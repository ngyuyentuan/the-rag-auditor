import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution, stage1_outcomes


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
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    for p, yi in zip(cs_ret, y):
        decision, _ = decide_route(p, t_lower, t_upper)
        o = stage1_outcomes(decision, yi)
        for k in counts:
            counts[k] += o[k]
    n = len(y)
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    fp_accept_rate = counts["fp"] / n if n else 0.0
    fn_reject_rate = counts["fn"] / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    coverage = 1.0 - dist["UNCERTAIN"]
    decided = counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"]
    accuracy_on_decided = (counts["tp"] + counts["tn"]) / decided if decided else 0.0
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


def quantiles(arr):
    qs = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
    vals = np.quantile(arr, qs)
    return list(zip(qs, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--baseline_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--candidate_yaml")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    base = load_thresholds(Path(args.baseline_yaml), args.track)

    df = pd.read_parquet(in_path)
    n_raw = len(df)
    if args.logit_col not in df.columns:
        raise ValueError(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise ValueError(f"missing y_col {args.y_col}")

    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna()
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    df[args.logit_col] = logits[mask].astype(float)

    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    n_used = len(df)
    logits = df[args.logit_col].astype(float).tolist()
    y_vals = df[args.y_col].astype(int).tolist()

    base_cs = np.asarray([compute_cs_ret_from_logit(x, float(base["tau"])) for x in logits], dtype=float)
    base_metrics = compute_metrics(base_cs, y_vals, float(base["t_lower"]), float(base["t_upper"]))
    base_q = quantiles(base_cs)

    candidate = None
    cand_metrics = None
    cand_q = None
    if args.candidate_yaml:
        candidate_path = Path(args.candidate_yaml)
        if candidate_path.exists():
            with candidate_path.open("r", encoding="utf-8") as f:
                candidate = yaml.safe_load(f)
        if candidate is not None and all(k in candidate for k in ("tau", "t_lower", "t_upper")):
            cand_cs = np.asarray([compute_cs_ret_from_logit(x, float(candidate["tau"])) for x in logits], dtype=float)
            cand_metrics = compute_metrics(cand_cs, y_vals, float(candidate["t_lower"]), float(candidate["t_upper"]))
            cand_q = quantiles(cand_cs)

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_tau_regression_explainer_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Tau Regression Explainer ({args.track})")
    lines.append("")
    lines.append(f"Generated: `{dt.datetime.utcnow().isoformat()}+00:00`")
    lines.append("")
    lines.append("Config")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n_raw: `{n_raw}`")
    lines.append(f"- n_used: `{n_used}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
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
        lines.append(f"| {k} | {base_metrics[k]:.4f} |")
    lines.append("")
    lines.append("Baseline cs_ret quantiles")
    lines.append("")
    lines.append("| q | value |")
    lines.append("|---:|---:|")
    for q, v in base_q:
        lines.append(f"| {q:.2f} | {v:.6f} |")
    lines.append("")

    if cand_metrics is None:
        lines.append("Candidate thresholds")
        lines.append("")
        lines.append("- candidate: not available")
        lines.append("")
    else:
        lines.append("Candidate thresholds")
        lines.append("")
        lines.append(f"- tau: `{candidate['tau']}`")
        lines.append(f"- t_lower: `{candidate['t_lower']}`")
        lines.append(f"- t_upper: `{candidate['t_upper']}`")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k in ("accept_rate", "reject_rate", "uncertain_rate", "fp_accept_rate", "fn_reject_rate", "ok_rate", "coverage", "accuracy_on_decided"):
            lines.append(f"| {k} | {cand_metrics[k]:.4f} |")
        lines.append("")
        lines.append("Candidate cs_ret quantiles")
        lines.append("")
        lines.append("| q | value |")
        lines.append("|---:|---:|")
        for q, v in cand_q:
            lines.append(f"| {q:.2f} | {v:.6f} |")
        lines.append("")
        lines.append("Delta (candidate - baseline)")
        lines.append("")
        lines.append("| metric | delta |")
        lines.append("|---|---:|")
        for k in ("accept_rate", "reject_rate", "uncertain_rate", "fp_accept_rate", "fn_reject_rate", "ok_rate", "coverage", "accuracy_on_decided"):
            lines.append(f"| {k} | {(cand_metrics[k] - base_metrics[k]):.4f} |")
        lines.append("")

    lines.append("Why no feasible configs can happen")
    lines.append("")
    if cand_metrics is None:
        lines.append(f"- baseline fp_accept_rate={base_metrics['fp_accept_rate']:.4f}, fn_reject_rate={base_metrics['fn_reject_rate']:.4f}, uncertain_rate={base_metrics['uncertain_rate']:.4f}")
        lines.append("- if constraints are tighter than these rates, the feasible set can be empty.")
    else:
        lines.append(f"- baseline fp_accept_rate={base_metrics['fp_accept_rate']:.4f}, fn_reject_rate={base_metrics['fn_reject_rate']:.4f}, uncertain_rate={base_metrics['uncertain_rate']:.4f}")
        lines.append(f"- candidate fp_accept_rate={cand_metrics['fp_accept_rate']:.4f}, fn_reject_rate={cand_metrics['fn_reject_rate']:.4f}, uncertain_rate={cand_metrics['uncertain_rate']:.4f}")
        lines.append("- if fp or fn increases while uncertain decreases, budgets with fp/fn caps can become infeasible.")
    lines.append("")
    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
