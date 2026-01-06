import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route


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


def compute_metrics(decisions, y):
    n = len(y)
    y_arr = np.asarray(y, dtype=int)
    d = np.asarray(decisions)
    accept = d == "ACCEPT"
    reject = d == "REJECT"
    uncertain = d == "UNCERTAIN"
    fp = np.sum(accept & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    tp = np.sum(accept & (y_arr == 1))
    tn = np.sum(reject & (y_arr == 0))
    accept_rate = float(np.mean(accept)) if n else 0.0
    reject_rate = float(np.mean(reject)) if n else 0.0
    uncertain_rate = float(np.mean(uncertain)) if n else 0.0
    fp_accept_rate = fp / n if n else 0.0
    fn_reject_rate = fn / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    coverage = 1.0 - uncertain_rate
    decided = tp + tn + fp + fn
    accuracy_on_decided = (tp + tn) / decided if decided else 0.0
    return {
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": accuracy_on_decided,
    }


def bootstrap_ci(decisions, y, rng, n_boot):
    n = len(y)
    decisions_arr = np.asarray(decisions)
    y_arr = np.asarray(y, dtype=int)
    metrics = {
        "uncertain_rate": [],
        "fp_accept_rate": [],
        "fn_reject_rate": [],
        "ok_rate": [],
        "coverage": [],
        "accuracy_on_decided": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = decisions_arr[idx]
        yb = y_arr[idx]
        m = compute_metrics(d, yb)
        for k in metrics:
            metrics[k].append(m[k])
    out = {}
    for k, vals in metrics.items():
        arr = np.asarray(vals)
        out[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def eval_config(logits, y_vals, tau, t_lower, t_upper):
    cs_ret = [compute_cs_ret_from_logit(x, tau) for x in logits]
    decisions = []
    for p in cs_ret:
        d, _ = decide_route(p, t_lower, t_upper)
        decisions.append(d)
    metrics = compute_metrics(decisions, y_vals)
    return decisions, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--tuned_thresholds_yaml")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    thresholds_path = Path(args.thresholds_yaml)
    base = load_thresholds(thresholds_path, args.track)

    df = pd.read_parquet(in_path)
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

    logits = df[args.logit_col].astype(float).tolist()
    y_vals = df[args.y_col].astype(int).tolist()

    base_decisions, base_metrics = eval_config(
        logits,
        y_vals,
        float(base["tau"]),
        float(base["t_lower"]),
        float(base["t_upper"]),
    )

    tuned_path = Path(args.tuned_thresholds_yaml) if args.tuned_thresholds_yaml else None
    tuned = None
    tuned_decisions = None
    tuned_metrics = None
    if tuned_path is not None and tuned_path.exists() and tuned_path.stat().st_size > 0:
        with tuned_path.open("r", encoding="utf-8") as f:
            tuned = yaml.safe_load(f)
        if not isinstance(tuned, dict) or any(k not in tuned for k in ("tau", "t_lower", "t_upper")):
            tuned = None
        else:
            tuned_decisions, tuned_metrics = eval_config(
                logits,
                y_vals,
                float(tuned["tau"]),
                float(tuned["t_lower"]),
                float(tuned["t_upper"]),
            )

    rng = np.random.default_rng(args.seed)
    base_ci = bootstrap_ci(base_decisions, y_vals, rng, args.bootstrap)
    tuned_ci = None
    if tuned_decisions is not None:
        tuned_ci = bootstrap_ci(tuned_decisions, y_vals, rng, args.bootstrap)

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_product_eval_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Product Eval ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: {len(df)}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append("")
    lines.append("Baseline thresholds")
    lines.append("")
    lines.append(f"- tau: `{base['tau']}`")
    lines.append(f"- t_lower: `{base['t_lower']}`")
    lines.append(f"- t_upper: `{base['t_upper']}`")
    lines.append("")
    lines.append("| metric | value | 95% CI |")
    lines.append("|---|---:|---:|")
    for k in ("uncertain_rate", "fp_accept_rate", "fn_reject_rate", "ok_rate", "coverage", "accuracy_on_decided"):
        lo, hi = base_ci[k]
        lines.append(f"| {k} | {base_metrics[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines.append("")
    if tuned_metrics is None:
        lines.append("Tuned thresholds")
        lines.append("")
        lines.append("- tuned: not available")
        lines.append("")
    else:
        lines.append("Tuned thresholds")
        lines.append("")
        lines.append(f"- tau: `{tuned['tau']}`")
        lines.append(f"- t_lower: `{tuned['t_lower']}`")
        lines.append(f"- t_upper: `{tuned['t_upper']}`")
        lines.append("")
        lines.append("| metric | value | 95% CI |")
        lines.append("|---|---:|---:|")
        for k in ("uncertain_rate", "fp_accept_rate", "fn_reject_rate", "ok_rate", "coverage", "accuracy_on_decided"):
            lo, hi = tuned_ci[k]
            lines.append(f"| {k} | {tuned_metrics[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.append("")
    lines.append("Interpretation")
    lines.append("")
    lines.append("ok_rate treats UNCERTAIN as safe deferral, so it can look high even when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use these together to judge product readiness.")
    lines.append("")
    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
