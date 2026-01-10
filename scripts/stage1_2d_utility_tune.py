import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, routing_distribution


def wilson_ci(x, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    phat = x / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def load_thresholds(path: Path, track: str):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    if "thresholds" in data:
        t = data["thresholds"].get(track)
        if not isinstance(t, dict):
            return None
        if not all(k in t for k in ("t_lower", "t_upper", "tau")):
            return None
        return {"t_lower": float(t["t_lower"]), "t_upper": float(t["t_upper"]), "tau": float(t["tau"])}
    if all(k in data for k in ("t_lower", "t_upper", "tau")):
        return {"t_lower": float(data["t_lower"]), "t_upper": float(data["t_upper"]), "tau": float(data["tau"])}
    return None


def compute_metrics(logits, cs_ret, y, t_lower, t_upper, c_accept, c_reject):
    y_arr = np.asarray(y, dtype=int)
    logits_arr = np.asarray(logits, dtype=float)
    cs_arr = np.asarray(cs_ret, dtype=float)
    accept = (logits_arr >= t_upper) & (cs_arr >= c_accept)
    reject = (logits_arr <= t_lower) | (cs_arr <= c_reject)
    uncertain = ~(accept | reject)
    n = len(y_arr)
    fp = np.sum(accept & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    tp = np.sum(accept & (y_arr == 1))
    tn = np.sum(reject & (y_arr == 0))
    dist = {
        "ACCEPT": float(np.mean(accept)) if n else 0.0,
        "REJECT": float(np.mean(reject)) if n else 0.0,
        "UNCERTAIN": float(np.mean(uncertain)) if n else 0.0,
    }
    fp_rate = fp / n if n else 0.0
    fn_rate = fn / n if n else 0.0
    ok_rate = 1.0 - fp_rate - fn_rate
    coverage = 1.0 - dist["UNCERTAIN"]
    decided = tp + tn + fp + fn
    accuracy_on_decided = (tp + tn) / decided if decided else 0.0
    return {
        "accept_rate": dist["ACCEPT"],
        "reject_rate": dist["REJECT"],
        "uncertain_rate": dist["UNCERTAIN"],
        "fp_accept_rate": fp_rate,
        "fn_reject_rate": fn_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": accuracy_on_decided,
        "n": n,
        "fp_count": int(fp),
        "fn_count": int(fn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--cs_ret_col")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--tau_source", choices=["config", "manual"], default="config")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--tau_grid_min", type=float, default=0.2)
    ap.add_argument("--tau_grid_max", type=float, default=2.0)
    ap.add_argument("--tau_grid_steps", type=int, default=20)
    ap.add_argument("--threshold_steps", type=int, default=40)
    ap.add_argument("--budgets", type=float, nargs="+", required=True)
    ap.add_argument("--max_fp_accept_rate", type=float, default=0.05)
    ap.add_argument("--max_fn_reject_rate", type=float, default=0.05)
    ap.add_argument("--constraint_ci", choices=["none", "wilson"], default="none")
    ap.add_argument("--lambda_fp", type=float, default=10.0)
    ap.add_argument("--lambda_fn", type=float, default=10.0)
    ap.add_argument("--lambda_unc", type=float, default=1.0)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_yaml", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    thresholds_path = Path("configs/thresholds.yaml")
    base = load_thresholds(thresholds_path, args.track)

    df = pd.read_parquet(in_path)
    if args.logit_col not in df.columns:
        raise ValueError(f"missing logit_col {args.logit_col}")
    cs_col = args.cs_ret_col
    if cs_col is None:
        for cand in ["cs_ret", "retrieval_score", "sim", "similarity", "score_ret", "max_sim"]:
            if cand in df.columns:
                cs_col = cand
                break
    if cs_col is None:
        out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_2d_utility_tuning_{args.track}.md")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        msg = [
            f"# Stage1 2D Utility Tuning ({args.track})",
            "",
            f"- in_path: `{in_path}`",
            f"- n: `{len(df)}`",
            f"- reason: missing cs_ret-like column",
        ]
        out_md.write_text("\n".join(msg) + "\n", encoding="utf-8")
        sys.exit(2)
    if args.y_col not in df.columns:
        raise ValueError(f"missing y_col {args.y_col}")

    y = pd.to_numeric(df[args.y_col], errors="coerce")
    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    cs_val = pd.to_numeric(df[cs_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna() & cs_val.notna()
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    df[args.logit_col] = logits[mask].astype(float)
    df[cs_col] = cs_val[mask].astype(float)
    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    y_vals = df[args.y_col].astype(int).tolist()
    logits_list = df[args.logit_col].astype(float).tolist()
    cs_ret_col = cs_col
    cs_vals = df[cs_ret_col].astype(float).tolist()

    def constraints_ok(m):
        fp_upper = wilson_ci(m["fp_count"], m["n"])[1] if args.constraint_ci == "wilson" else m["fp_accept_rate"]
        fn_upper = wilson_ci(m["fn_count"], m["n"])[1] if args.constraint_ci == "wilson" else m["fn_reject_rate"]
        m["fp_upper95"] = fp_upper
        m["fn_upper95"] = fn_upper
        ok = True
        if fp_upper > args.max_fp_accept_rate:
            ok = False
        if fn_upper > args.max_fn_reject_rate:
            ok = False
        m["constraint_met"] = ok
        return ok

    step = args.threshold_steps
    grid_lower = np.linspace(0.0, 0.99, step)
    grid_upper = np.linspace(0.01, 1.0, step)
    budgets = sorted(args.budgets)

    def utility(m):
        return m["coverage"] - args.lambda_fp * m["fp_accept_rate"] - args.lambda_fn * m["fn_reject_rate"] - args.lambda_unc * m["uncertain_rate"]

    def evaluate(tau):
        cs_from_tau = [compute_cs_ret_from_logit(x, tau) for x in logits_list]
        best = None
        top_rows = {}
        feasible_counts = {}
        for b in budgets:
            rows = []
            for tl in grid_lower:
                for tu in grid_upper:
                    if tl >= tu:
                        continue
                    for ca in grid_upper:
                        for cr in grid_lower:
                            if ca <= cr:
                                continue
                            m = compute_metrics(logits_list, cs_from_tau, y_vals, float(tl), float(tu), float(ca), float(cr))
                            if m["uncertain_rate"] > b:
                                continue
                            m["utility"] = utility(m)
                            m["tau"] = float(tau)
                            m["t_lower"] = float(tl)
                            m["t_upper"] = float(tu)
                            m["c_accept"] = float(ca)
                            m["c_reject"] = float(cr)
                            rows.append(m)
            for r in rows:
                constraints_ok(r)
            feasible = [r for r in rows if r["constraint_met"]]
            feasible_counts[b] = len(feasible)
            ranked = sorted(
                feasible if feasible else rows,
                key=lambda r: (-r["utility"], r["fp_accept_rate"] + r["fn_reject_rate"], -r["coverage"], r["uncertain_rate"]),
            )
            top_rows[b] = ranked[:10]
            if feasible:
                candidate = ranked[0]
                if best is None or candidate["utility"] > best["utility"]:
                    best = candidate
        return best, top_rows, feasible_counts

    tau_list = [float(base["tau"])]
    if args.tau_source == "manual" and args.tau is not None:
        tau_list = [float(args.tau)]
    elif args.tau_source == "fit_grid":
        tau_list = np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)

    overall_best = None
    best_reason = None
    overall_rows = {}
    feasible_counts = {}
    for tau in tau_list:
        best, rows, feas = evaluate(tau)
        for k, v in rows.items():
            overall_rows.setdefault(k, []).extend(v)
        for k, v in feas.items():
            feasible_counts[k] = feasible_counts.get(k, 0) + v
        if best and (overall_best is None or best["utility"] > overall_best["utility"]):
            overall_best = best
    if overall_best is None:
        best_reason = "no feasible config under constraints"

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_2d_utility_tuning_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Stage1 2D Utility Tuning ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: `{len(df)}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- cs_ret_col: `{cs_ret_col}`")
    lines.append(f"- tau_source: `{args.tau_source}`")
    lines.append(f"- constraint_ci: `{args.constraint_ci}`")
    lines.append(f"- max_fp_accept_rate: `{args.max_fp_accept_rate}`")
    lines.append(f"- max_fn_reject_rate: `{args.max_fn_reject_rate}`")
    lines.append("")
    for b in budgets:
        rows = overall_rows.get(b, [])
        lines.append(f"## Budget <= {b}")
        lines.append("")
        lines.append(f"- feasible_configs: `{feasible_counts.get(b,0)}`")
        lines.append("| tau | t_lower | t_upper | c_accept | c_reject | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | coverage | utility |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        if not rows:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            rows_sorted = sorted(rows, key=lambda r: (-r["utility"], r["fp_accept_rate"] + r["fn_reject_rate"], -r["coverage"], r["uncertain_rate"]))[:10]
            for r in rows_sorted:
                lines.append(
                    "| {tau:.4f} | {t_lower:.4f} | {t_upper:.4f} | {c_accept:.4f} | {c_reject:.4f} | {uncertain_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {ok_rate:.4f} | {coverage:.4f} | {utility:.4f} |".format(
                        tau=r["tau"],
                        t_lower=r["t_lower"],
                        t_upper=r["t_upper"],
                        c_accept=r["c_accept"],
                        c_reject=r["c_reject"],
                        uncertain_rate=r["uncertain_rate"],
                        fp_accept_rate=r["fp_accept_rate"],
                        fn_reject_rate=r["fn_reject_rate"],
                        ok_rate=r["ok_rate"],
                        coverage=r["coverage"],
                        utility=r["utility"],
                    )
                )
        lines.append("")
    lines.append("Selected config")
    lines.append("")
    if overall_best is None:
        lines.append("- selected: none")
        if best_reason:
            lines.append(f"- reason: {best_reason}")
    else:
        lines.append(f"- tau: `{overall_best['tau']}`")
        lines.append(f"- t_lower: `{overall_best['t_lower']}`")
        lines.append(f"- t_upper: `{overall_best['t_upper']}`")
        lines.append(f"- c_accept: `{overall_best['c_accept']}`")
        lines.append(f"- c_reject: `{overall_best['c_reject']}`")
        lines.append(f"- utility: `{overall_best['utility']}`")
        lines.append(f"- uncertain_rate: `{overall_best['uncertain_rate']}`")
        lines.append(f"- fp_accept_rate: `{overall_best['fp_accept_rate']}`")
        lines.append(f"- fn_reject_rate: `{overall_best['fn_reject_rate']}`")
        lines.append(f"- fp_upper95: `{overall_best.get('fp_upper95')}`")
        lines.append(f"- fn_upper95: `{overall_best.get('fn_upper95')}`")
        lines.append(f"- constraint_met: `{overall_best.get('constraint_met', False)}`")
    lines.append("")
    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if overall_best is not None:
        out_yaml = Path(args.out_yaml) if args.out_yaml else Path(f"configs/thresholds_stage1_product_2d_{args.track}.yaml")
        payload = {
            "tau": float(overall_best["tau"]),
            "t_lower": float(overall_best["t_lower"]),
            "t_upper": float(overall_best["t_upper"]),
            "c_accept": float(overall_best["c_accept"]),
            "c_reject": float(overall_best["c_reject"]),
            "lambda_fp": float(args.lambda_fp),
            "lambda_fn": float(args.lambda_fn),
            "lambda_unc": float(args.lambda_unc),
            "max_fp_accept_rate": float(args.max_fp_accept_rate),
            "max_fn_reject_rate": float(args.max_fn_reject_rate),
            "constraint_ci": args.constraint_ci,
        }
        out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
