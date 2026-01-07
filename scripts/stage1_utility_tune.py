import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution


def load_thresholds(path: Path, track: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "thresholds" not in data:
        raise ValueError(f"thresholds missing in {path}")
    t = data["thresholds"].get(track)
    if not isinstance(t, dict):
        raise ValueError(f"track {track} missing in {path}")
    for k in ("t_lower", "t_upper", "tau"):
        if k not in t:
            raise ValueError(f"{k} missing for track {track} in {path}")
    return t


def compute_metrics(cs_ret, y, t_lower, t_upper):
    decisions = []
    for p in cs_ret:
        d, _ = decide_route(p, t_lower, t_upper)
        decisions.append(d)
    y_arr = np.asarray(y, dtype=int)
    d_arr = np.asarray(decisions)
    accept = d_arr == "ACCEPT"
    reject = d_arr == "REJECT"
    uncertain = d_arr == "UNCERTAIN"
    fp = np.sum(accept & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    tp = np.sum(accept & (y_arr == 1))
    tn = np.sum(reject & (y_arr == 0))
    n = len(y_arr)
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    fp_accept_rate = fp / n if n else 0.0
    fn_reject_rate = fn / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    coverage = 1.0 - dist["UNCERTAIN"]
    decided = tp + tn + fp + fn
    accuracy_on_decided = (tp + tn) / decided if decided else 0.0
    return {
        "accept_rate": dist["ACCEPT"],
        "reject_rate": dist["REJECT"],
        "uncertain_rate": dist["UNCERTAIN"],
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": accuracy_on_decided,
    }


def compute_utility(metrics, lambda_fp, lambda_fn, lambda_unc):
    return metrics["coverage"] - lambda_fp * metrics["fp_accept_rate"] - lambda_fn * metrics["fn_reject_rate"] - lambda_unc * metrics["uncertain_rate"]


def select_best(candidates, max_fp, max_fn, lambda_fp, lambda_fn, lambda_unc):
    feasible = [c for c in candidates if (max_fp is None or c["fp_accept_rate"] <= max_fp) and (max_fn is None or c["fn_reject_rate"] <= max_fn)]
    target = feasible if feasible else candidates
    for c in target:
        c["utility"] = compute_utility(c, lambda_fp, lambda_fn, lambda_unc)
    ranked = sorted(
        target,
        key=lambda r: (-r["utility"], -(r["ok_rate"]), r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]),
    )
    pick = ranked[0] if ranked else None
    return pick, feasible


def pareto_front(candidates):
    front = []
    for c in candidates:
        dominated = False
        for o in candidates:
            if o is c:
                continue
            better_or_equal = o["fp_accept_rate"] <= c["fp_accept_rate"] and o["fn_reject_rate"] <= c["fn_reject_rate"] and o["uncertain_rate"] <= c["uncertain_rate"]
            strictly_better = o["fp_accept_rate"] < c["fp_accept_rate"] or o["fn_reject_rate"] < c["fn_reject_rate"] or o["uncertain_rate"] < c["uncertain_rate"]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--tau_source", choices=["config", "manual", "grid"], default="config")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--tau_grid_min", type=float, default=0.2)
    ap.add_argument("--tau_grid_max", type=float, default=2.0)
    ap.add_argument("--tau_grid_steps", type=int, default=50)
    ap.add_argument("--grid_lower_steps", type=int, default=50)
    ap.add_argument("--grid_upper_steps", type=int, default=50)
    ap.add_argument("--max_fp_accept_rate", type=float, default=0.05)
    ap.add_argument("--max_fn_reject_rate", type=float, default=0.05)
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
    n_raw = len(df)
    if args.logit_col not in df.columns:
        raise ValueError(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise ValueError(f"missing y_col {args.y_col}")

    y = pd.to_numeric(df[args.y_col], errors="coerce")
    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna()
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    df[args.logit_col] = logits[mask].astype(float)

    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    n_used = len(df)
    y_vals = df[args.y_col].astype(int).tolist()
    logits = df[args.logit_col].astype(float).tolist()

    tau_list = []
    if args.tau_source == "manual":
        if args.tau is None:
            raise ValueError("tau_source=manual requires --tau")
        tau_list = [float(args.tau)]
    elif args.tau_source == "grid":
        tau_list = [float(t) for t in np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)]
    else:
        tau_list = [float(base["tau"])]

    grid_lower = np.linspace(0.0, 0.99, args.grid_lower_steps)
    grid_upper = np.linspace(0.01, 1.0, args.grid_upper_steps)

    baseline_metrics = compute_metrics(
        [compute_cs_ret_from_logit(x, float(base["tau"])) for x in logits],
        y_vals,
        float(base["t_lower"]),
        float(base["t_upper"]),
    )

    candidates = []
    for tau in tau_list:
        cs_ret = [compute_cs_ret_from_logit(x, tau) for x in logits]
        for t_lower in grid_lower:
            for t_upper in grid_upper:
                if t_lower >= t_upper:
                    continue
                m = compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper))
                cand = dict(m)
                cand["tau"] = float(tau)
                cand["t_lower"] = float(t_lower)
                cand["t_upper"] = float(t_upper)
                candidates.append(cand)
    base_candidate = dict(baseline_metrics)
    base_candidate["tau"] = float(base["tau"])
    base_candidate["t_lower"] = float(base["t_lower"])
    base_candidate["t_upper"] = float(base["t_upper"])
    base_candidate["label"] = "baseline_candidate"
    candidates.append(base_candidate)

    pick, feasible = select_best(
        candidates,
        args.max_fp_accept_rate,
        args.max_fn_reject_rate,
        args.lambda_fp,
        args.lambda_fn,
        args.lambda_unc,
    )

    for c in candidates:
        c["utility"] = compute_utility(c, args.lambda_fp, args.lambda_fn, args.lambda_unc)
    ranked = sorted(candidates, key=lambda r: (-r["utility"], -(r["ok_rate"]), r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]))[:20]
    pareto = pareto_front(feasible if feasible else candidates)
    pareto_ranked = sorted(pareto, key=lambda r: (-r["utility"], r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]))[:20]

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_utility_tuning_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Utility Tuning ({args.track})")
    lines.append("")
    lines.append("Config")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n_raw: `{n_raw}`")
    lines.append(f"- n_used: `{n_used}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- tau_source: `{args.tau_source}`")
    lines.append(f"- tau_grid_min: `{args.tau_grid_min}`")
    lines.append(f"- tau_grid_max: `{args.tau_grid_max}`")
    lines.append(f"- tau_grid_steps: `{args.tau_grid_steps}`")
    lines.append(f"- max_fp_accept_rate: `{args.max_fp_accept_rate}`")
    lines.append(f"- max_fn_reject_rate: `{args.max_fn_reject_rate}`")
    lines.append(f"- lambda_fp: `{args.lambda_fp}`")
    lines.append(f"- lambda_fn: `{args.lambda_fn}`")
    lines.append(f"- lambda_unc: `{args.lambda_unc}`")
    lines.append("")
    lines.append("Baseline thresholds")
    lines.append("")
    lines.append(f"- tau: `{base['tau']}`")
    lines.append(f"- t_lower: `{base['t_lower']}`")
    lines.append(f"- t_upper: `{base['t_upper']}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ("accept_rate", "reject_rate", "uncertain_rate", "fp_accept_rate", "fn_reject_rate", "ok_rate", "coverage", "accuracy_on_decided"):
        lines.append(f"| {k} | {baseline_metrics[k]:.4f} |")
    lines.append("")
    lines.append(f"- feasible_configs: `{len(feasible)}`")
    lines.append("")
    lines.append("Top configs by utility")
    lines.append("")
    lines.append("| tau | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | coverage | accuracy_on_decided | utility | label |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ranked:
        lines.append(
            "| {tau:.4f} | {t_lower:.4f} | {t_upper:.4f} | {accept_rate:.4f} | {reject_rate:.4f} | {uncertain_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {ok_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} | {utility:.4f} | {label} |".format(
                tau=r["tau"],
                t_lower=r["t_lower"],
                t_upper=r["t_upper"],
                accept_rate=r["accept_rate"],
                reject_rate=r["reject_rate"],
                uncertain_rate=r["uncertain_rate"],
                fp_accept_rate=r["fp_accept_rate"],
                fn_reject_rate=r["fn_reject_rate"],
                ok_rate=r["ok_rate"],
                coverage=r["coverage"],
                accuracy_on_decided=r["accuracy_on_decided"],
                utility=r["utility"],
                label=r.get("label", "grid"),
            )
        )
    lines.append("")
    lines.append("Pareto summary")
    lines.append("")
    lines.append("| tau | t_lower | t_upper | fp_accept_rate | fn_reject_rate | uncertain_rate | ok_rate | utility |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in pareto_ranked:
        lines.append(
            "| {tau:.4f} | {t_lower:.4f} | {t_upper:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {ok_rate:.4f} | {utility:.4f} |".format(
                tau=r["tau"],
                t_lower=r["t_lower"],
                t_upper=r["t_upper"],
                fp_accept_rate=r["fp_accept_rate"],
                fn_reject_rate=r["fn_reject_rate"],
                uncertain_rate=r["uncertain_rate"],
                ok_rate=r["ok_rate"],
                utility=r["utility"],
            )
        )
    lines.append("")
    lines.append("Selected config")
    lines.append("")
    if pick is None:
        lines.append("- selected: none")
    else:
        feasible_flag = (args.max_fp_accept_rate is None or pick["fp_accept_rate"] <= args.max_fp_accept_rate) and (args.max_fn_reject_rate is None or pick["fn_reject_rate"] <= args.max_fn_reject_rate)
        lines.append(f"- tau: `{pick['tau']}`")
        lines.append(f"- t_lower: `{pick['t_lower']}`")
        lines.append(f"- t_upper: `{pick['t_upper']}`")
        lines.append(f"- utility: `{pick['utility']}`")
        lines.append(f"- feasible: `{feasible_flag}`")
        lines.append("")
    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    out_yaml = Path(args.out_yaml) if args.out_yaml else Path(f"configs/thresholds_stage1_product_{args.track}.yaml")
    if pick is not None:
        payload = {
            "tau": float(pick["tau"]),
            "t_lower": float(pick["t_lower"]),
            "t_upper": float(pick["t_upper"]),
            "lambda_fp": float(args.lambda_fp),
            "lambda_fn": float(args.lambda_fn),
            "lambda_unc": float(args.lambda_unc),
            "max_fp_accept_rate": float(args.max_fp_accept_rate),
            "max_fn_reject_rate": float(args.max_fn_reject_rate),
        }
        out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
