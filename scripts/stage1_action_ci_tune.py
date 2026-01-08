import argparse
import sys
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution


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
    else:
        t = data
    if not isinstance(t, dict):
        return None
    if not all(k in t for k in ("tau", "t_lower", "t_upper")):
        return None
    return {"tau": float(t["tau"]), "t_lower": float(t["t_lower"]), "t_upper": float(t["t_upper"])}


def fit_tau_grid(logits, y, tau_min, tau_max, steps):
    y_arr = np.asarray(y, dtype=float)
    taus = np.linspace(tau_min, tau_max, steps)
    best_tau = None
    best_nll = None
    eps = 1e-12
    for tau in taus:
        probs = np.array([compute_cs_ret_from_logit(v, tau) for v in logits], dtype=float)
        probs = np.clip(probs, eps, 1 - eps)
        nll = -np.mean(y_arr * np.log(probs) + (1 - y_arr) * np.log(1 - probs))
        if best_nll is None or nll < best_nll:
            best_nll = nll
            best_tau = float(tau)
    return best_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--tau_source", choices=["config", "fit_grid", "joint_grid", "manual"], default="config")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--tau_grid_min", type=float, default=0.2)
    ap.add_argument("--tau_grid_max", type=float, default=2.0)
    ap.add_argument("--tau_grid_steps", type=int, default=25)
    ap.add_argument("--threshold_steps", type=int, default=50)
    ap.add_argument("--min_accept_count", type=int, default=50)
    ap.add_argument("--min_reject_count", type=int, default=50)
    ap.add_argument("--min_coverage", type=float, default=0.30)
    ap.add_argument("--max_fp_given_accept_upper95", type=float, default=0.10)
    ap.add_argument("--max_fn_given_reject_upper95", type=float, default=0.10)
    ap.add_argument("--max_reject_rate", type=float, default=0.95)
    ap.add_argument("--certify", choices=["both", "accept_only", "reject_only"], default="both")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_yaml", default=None)
    ap.add_argument("--candidate_yaml")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))
    df = pd.read_parquet(in_path)
    if args.logit_col not in df.columns or args.y_col not in df.columns:
        raise ValueError("missing required columns")
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna()
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    df[args.logit_col] = logits[mask].astype(float)
    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)
    y_vals = df[args.y_col].astype(int).to_numpy()
    log_vals = df[args.logit_col].astype(float).to_numpy()
    n_used = len(df)

    thresholds_path = Path("configs/thresholds.yaml")
    base = load_thresholds(thresholds_path, args.track)
    tau_list = []
    if args.tau_source == "manual":
        if args.tau is None:
            raise ValueError("tau_source manual requires --tau")
        tau_list = [float(args.tau)]
    elif args.tau_source == "fit_grid":
        tau_list = [fit_tau_grid(log_vals, y_vals, args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)]
    elif args.tau_source == "joint_grid":
        tau_list = [float(t) for t in np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)]
    else:
        tau_list = [float(base["tau"])] if base else [1.0]

    grid_lower = np.linspace(0.0, 0.99, args.threshold_steps)
    grid_upper = np.linspace(0.01, 1.0, args.threshold_steps)

    def rank_key(row):
        if args.certify == "accept_only":
            return (
                row["coverage"],
                -row["fp_given_accept_upper95"],
                row["accuracy_on_decided"],
                row["accept_count"],
                -row["t_lower"],
                -row["t_upper"],
            )
        if args.certify == "reject_only":
            return (
                row["coverage"],
                -row["fn_given_reject_upper95"],
                row["accuracy_on_decided"],
                row["reject_count"],
                -row["t_lower"],
                -row["t_upper"],
            )
        return (
            row["coverage"],
            -(row["fp_given_accept_upper95"] + row["fn_given_reject_upper95"]),
            row["accuracy_on_decided"],
            -row["t_lower"],
            -row["t_upper"],
        )

    def violation_score(row):
        score = max(0.0, args.min_coverage - row["coverage"])
        if args.certify in ("both", "accept_only"):
            if row["accept_count"] < args.min_accept_count:
                score += (args.min_accept_count - row["accept_count"]) / max(1, args.min_accept_count)
            score += max(0.0, row["fp_given_accept_upper95"] - args.max_fp_given_accept_upper95)
        if args.certify in ("both", "reject_only"):
            if row["reject_count"] < args.min_reject_count:
                score += (args.min_reject_count - row["reject_count"]) / max(1, args.min_reject_count)
            score += max(0.0, row["fn_given_reject_upper95"] - args.max_fn_given_reject_upper95)
        return score

    def compute_row(tau, tl, tu, cs=None):
        cs_vals = cs if cs is not None else np.array([compute_cs_ret_from_logit(v, tau) for v in log_vals], dtype=float)
        decisions = []
        for p in cs_vals:
            d, _ = decide_route(p, tl, tu)
            decisions.append(d)
        d_arr = np.asarray(decisions)
        accept = d_arr == "ACCEPT"
        reject = d_arr == "REJECT"
        uncertain = d_arr == "UNCERTAIN"
        accept_count = int(np.sum(accept))
        reject_count = int(np.sum(reject))
        coverage = 1.0 - float(np.mean(uncertain))
        fp = int(np.sum(accept & (y_vals == 0)))
        fn = int(np.sum(reject & (y_vals == 1)))
        decided = accept_count + reject_count
        acc_decided = (np.sum(accept & (y_vals == 1)) + np.sum(reject & (y_vals == 0))) / decided if decided else 0.0
        fp_given_accept = fp / accept_count if accept_count else 0.0
        fn_given_reject = fn / reject_count if reject_count else 0.0
        fp_accept_ci = wilson_ci(fp, accept_count)
        fn_reject_ci = wilson_ci(fn, reject_count)
        fp_total_ci = wilson_ci(fp, n_used)
        fn_total_ci = wilson_ci(fn, n_used)
        if args.certify == "accept_only":
            accept_status = "INSUFFICIENT_N" if accept_count == 0 or accept_count < args.min_accept_count else ("PASS" if fp_accept_ci[1] <= args.max_fp_given_accept_upper95 else "FAIL")
            reject_status = "N/A"
        elif args.certify == "reject_only":
            accept_status = "N/A"
            reject_status = "INSUFFICIENT_N" if reject_count == 0 or reject_count < args.min_reject_count else ("PASS" if fn_reject_ci[1] <= args.max_fn_given_reject_upper95 else "FAIL")
        else:
            accept_status = "INSUFFICIENT_N" if accept_count == 0 or accept_count < args.min_accept_count else ("PASS" if fp_accept_ci[1] <= args.max_fp_given_accept_upper95 else "FAIL")
            reject_status = "INSUFFICIENT_N" if reject_count == 0 or reject_count < args.min_reject_count else ("PASS" if fn_reject_ci[1] <= args.max_fn_given_reject_upper95 else "FAIL")
        reject_rate = reject_count / n_used if n_used else 0.0
        feasible = True
        if coverage < args.min_coverage:
            feasible = False
        if args.certify in ("both", "accept_only"):
            if accept_count < args.min_accept_count:
                feasible = False
            if fp_accept_ci[1] > args.max_fp_given_accept_upper95:
                feasible = False
        if args.certify in ("both", "reject_only"):
            if reject_count < args.min_reject_count:
                feasible = False
            if fn_reject_ci[1] > args.max_fn_given_reject_upper95:
                feasible = False
        if args.certify == "reject_only":
            if args.max_reject_rate is not None and reject_rate > args.max_reject_rate:
                feasible = False
        dist = routing_distribution(cs_vals.tolist(), float(tl), float(tu))
        return {
            "tau": float(tau),
            "t_lower": float(tl),
            "t_upper": float(tu),
            "accept_rate": dist["ACCEPT"],
            "reject_rate": dist["REJECT"],
            "uncertain_rate": dist["UNCERTAIN"],
            "coverage": coverage,
            "accuracy_on_decided": acc_decided,
            "accept_count": accept_count,
            "reject_count": reject_count,
            "fp_given_accept": fp_given_accept,
            "fn_given_reject": fn_given_reject,
            "fp_given_accept_upper95": fp_accept_ci[1],
            "fn_given_reject_upper95": fn_reject_ci[1],
            "fp_upper95": fp_total_ci[1],
            "fn_upper95": fn_total_ci[1],
            "feasible": feasible,
            "accept_status": accept_status,
            "reject_status": reject_status,
            "reject_rate": reject_rate,
        }

    def evaluate(tau):
        cs = np.array([compute_cs_ret_from_logit(v, tau) for v in log_vals], dtype=float)
        best = None
        rows = []
        for tl in grid_lower:
            for tu in grid_upper:
                if tl >= tu:
                    continue
                row = compute_row(tau, float(tl), float(tu), cs=cs)
                rows.append(row)
                if row["feasible"]:
                    if best is None or rank_key(row) > rank_key(best):
                        best = row
        return best, rows

    overall_best = None
    all_rows = []
    candidate_rows = []
    base_candidate = load_thresholds(Path("configs/thresholds.yaml"), args.track)
    if base_candidate:
        candidate_rows.append(("baseline_candidate", base_candidate))
    if args.candidate_yaml:
        cand = load_thresholds(Path(args.candidate_yaml), args.track)
        if cand:
            candidate_rows.append(("candidate_yaml", cand))
    for tau in tau_list:
        best, rows = evaluate(tau)
        all_rows.extend(rows)
        if best and (overall_best is None or rank_key(best) > rank_key(overall_best)):
            overall_best = best
    candidate_results = []
    for name, cand in candidate_rows:
        row = compute_row(float(cand["tau"]), float(cand["t_lower"]), float(cand["t_upper"]))
        row["candidate_name"] = name
        candidate_results.append(row)
        all_rows.append(row)
        if row["feasible"] and (overall_best is None or rank_key(row) > rank_key(overall_best)):
            overall_best = row

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_action_ci_tuning_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Action-CI Tuning ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: `{n_used}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- tau_source: `{args.tau_source}`")
    lines.append(f"- tau_grid_min: `{args.tau_grid_min}`")
    lines.append(f"- tau_grid_max: `{args.tau_grid_max}`")
    lines.append(f"- tau_grid_steps: `{args.tau_grid_steps}`")
    lines.append(f"- threshold_steps: `{args.threshold_steps}`")
    lines.append(f"- min_coverage: `{args.min_coverage}`")
    lines.append(f"- min_accept_count: `{args.min_accept_count}`")
    lines.append(f"- min_reject_count: `{args.min_reject_count}`")
    lines.append(f"- max_fp_given_accept_upper95: `{args.max_fp_given_accept_upper95}`")
    lines.append(f"- max_fn_given_reject_upper95: `{args.max_fn_given_reject_upper95}`")
    lines.append(f"- max_reject_rate: `{args.max_reject_rate}`")
    lines.append(f"- certify: `{args.certify}`")
    if args.candidate_yaml:
        lines.append(f"- candidate_yaml: `{args.candidate_yaml}`")
    lines.append("")

    feasible_rows = [r for r in all_rows if r["feasible"]]
    lines.append(f"- feasible_count: `{len(feasible_rows)}`")
    lines.append("")
    lines.append("Candidate feasibility")
    lines.append("")
    lines.append("| candidate | tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | failures |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    if candidate_results:
        for r in candidate_results:
            failures = []
            if r["coverage"] < args.min_coverage:
                failures.append("coverage")
            if args.certify in ("both", "accept_only"):
                if r["accept_count"] < args.min_accept_count:
                    failures.append("accept_count")
                if r["fp_given_accept_upper95"] > args.max_fp_given_accept_upper95:
                    failures.append("fp_upper95")
            if args.certify in ("both", "reject_only"):
                if r["reject_count"] < args.min_reject_count:
                    failures.append("reject_count")
                if r["fn_given_reject_upper95"] > args.max_fn_given_reject_upper95:
                    failures.append("fn_upper95")
            fail_text = ",".join(failures) if failures else "ok"
            lines.append(
                "| {candidate_name} | {tau:.4f} | {t_lower:.4f} | {t_upper:.4f} | {coverage:.4f} | {accept_count} | {reject_count} | {fp_given_accept_upper95:.4f} | {fn_given_reject_upper95:.4f} | {accept_status} | {reject_status} | {fail_text} |".format(
                    **r,
                    fail_text=fail_text,
                )
            )
    else:
        lines.append("| none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    lines.append("Top candidates")
    lines.append("")
    lines.append("| tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_status | reject_status | accuracy_on_decided |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ranked = sorted(
        feasible_rows if feasible_rows else all_rows,
        key=rank_key,
        reverse=True,
    )[:10]
    if ranked:
        for r in ranked:
            lines.append(
                "| {tau:.4f} | {t_lower:.4f} | {t_upper:.4f} | {coverage:.4f} | {accept_count} | {reject_count} | {fp_given_accept_upper95:.4f} | {fn_given_reject_upper95:.4f} | {accept_status} | {reject_status} | {accuracy_on_decided:.4f} |".format(
                    **r
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")

    lines.append("Selected config")
    lines.append("")
    if overall_best is None:
        lines.append("- selected: none")
        near = sorted(all_rows, key=violation_score)[:5]
        lines.append("")
        lines.append("Closest (not feasible)")
        lines.append("")
        lines.append("| tau | t_lower | t_upper | coverage | accept_count | reject_count | fp_given_accept_upper95 | fn_given_reject_upper95 | coverage_shortfall | accept_shortfall | reject_shortfall | fp_excess | fn_excess |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in near:
            cov_short = max(0.0, args.min_coverage - r["coverage"])
            acc_short = max(0.0, args.min_accept_count - r["accept_count"]) / max(1, args.min_accept_count) if args.certify in ("both", "accept_only") else 0.0
            rej_short = max(0.0, args.min_reject_count - r["reject_count"]) / max(1, args.min_reject_count) if args.certify in ("both", "reject_only") else 0.0
            fp_ex = max(0.0, r["fp_given_accept_upper95"] - args.max_fp_given_accept_upper95) if args.certify in ("both", "accept_only") else 0.0
            fn_ex = max(0.0, r["fn_given_reject_upper95"] - args.max_fn_given_reject_upper95) if args.certify in ("both", "reject_only") else 0.0
            lines.append(
                "| {tau:.4f} | {t_lower:.4f} | {t_upper:.4f} | {coverage:.4f} | {accept_count} | {reject_count} | {fp_given_accept_upper95:.4f} | {fn_given_reject_upper95:.4f} | {cov_short:.4f} | {acc_short:.4f} | {rej_short:.4f} | {fp_ex:.4f} | {fn_ex:.4f} |".format(
                    **r,
                    cov_short=cov_short,
                    acc_short=acc_short,
                    rej_short=rej_short,
                    fp_ex=fp_ex,
                    fn_ex=fn_ex,
                )
            )
    else:
        lines.append(f"- accept_status: `{overall_best['accept_status']}`")
        lines.append(f"- reject_status: `{overall_best['reject_status']}`")
        lines.append(f"- trivial_reject_all: `{overall_best['reject_rate'] >= 0.999}`")
        lines.append(f"- trivial_accept_none: `{overall_best['accept_count'] == 0}`")
        lines.append(f"- tau: `{overall_best['tau']}`")
        lines.append(f"- t_lower: `{overall_best['t_lower']}`")
        lines.append(f"- t_upper: `{overall_best['t_upper']}`")
        lines.append(f"- coverage: `{overall_best['coverage']}`")
        lines.append(f"- accept_count: `{overall_best['accept_count']}`")
        lines.append(f"- reject_count: `{overall_best['reject_count']}`")
        lines.append(f"- fp_given_accept_upper95: `{overall_best['fp_given_accept_upper95']}`")
        lines.append(f"- fn_given_reject_upper95: `{overall_best['fn_given_reject_upper95']}`")
        lines.append(f"- accuracy_on_decided: `{overall_best['accuracy_on_decided']}`")
    lines.append("")
    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if overall_best is not None:
        if args.out_yaml:
            out_yaml = Path(args.out_yaml)
        elif args.certify == "accept_only":
            out_yaml = Path(f"configs/thresholds_stage1_action_accept_{args.track}.yaml")
        elif args.certify == "reject_only":
            out_yaml = Path(f"configs/thresholds_stage1_action_reject_{args.track}.yaml")
        else:
            out_yaml = Path(f"configs/thresholds_stage1_action_ci_{args.track}.yaml")
        payload = {
            "tau": float(overall_best["tau"]),
            "t_lower": float(overall_best["t_lower"]),
            "t_upper": float(overall_best["t_upper"]),
            "min_coverage": float(args.min_coverage),
            "min_accept_count": int(args.min_accept_count),
            "min_reject_count": int(args.min_reject_count),
            "max_fp_given_accept_upper95": float(args.max_fp_given_accept_upper95),
            "max_fn_given_reject_upper95": float(args.max_fn_given_reject_upper95),
            "tau_source": args.tau_source,
            "certify": args.certify,
            "max_reject_rate": None if args.max_reject_rate is None else float(args.max_reject_rate),
        }
        out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
