import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit


def ece_score(probs, y, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if not np.any(mask):
            continue
        bin_y = y[mask]
        bin_p = probs[mask]
        acc = float(np.mean(bin_y))
        conf = float(np.mean(bin_p))
        weight = float(len(bin_y)) / float(total)
        ece += abs(acc - conf) * weight
    return ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--tau_grid_min", type=float, default=0.2)
    ap.add_argument("--tau_grid_max", type=float, default=2.0)
    ap.add_argument("--tau_grid_steps", type=int, default=200)
    ap.add_argument("--metric", choices=["nll", "brier", "ece"], default="nll")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

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
    y_vals = df[args.y_col].astype(int).to_numpy()

    taus = np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)
    eps = 1e-12
    results = []
    for tau in taus:
        probs = np.asarray([compute_cs_ret_from_logit(v, float(tau)) for v in logits], dtype=float)
        probs = np.clip(probs, eps, 1.0 - eps)
        if args.metric == "nll":
            val = -np.mean(y_vals * np.log(probs) + (1.0 - y_vals) * np.log(1.0 - probs))
        elif args.metric == "brier":
            val = np.mean((probs - y_vals) ** 2)
        else:
            val = ece_score(probs, y_vals, bins=15)
        results.append((float(tau), float(val)))

    results_sorted = sorted(results, key=lambda x: x[1])
    best_tau, best_val = results_sorted[0]
    top10 = results_sorted[:10]

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_tau_fit_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Stage1 Tau Fit ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: {len(df)}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- metric: `{args.metric}`")
    lines.append(f"- tau_grid_min: `{args.tau_grid_min}`")
    lines.append(f"- tau_grid_max: `{args.tau_grid_max}`")
    lines.append(f"- tau_grid_steps: `{args.tau_grid_steps}`")
    lines.append(f"- best_tau: `{best_tau}`")
    lines.append(f"- best_metric: `{best_val}`")
    lines.append("")
    lines.append("| tau | metric |")
    lines.append("|---:|---:|")
    for t, v in top10:
        lines.append(f"| {t:.6f} | {v:.6f} |")
    lines.append("")
    lines.append("## Repro command")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
