"""
Generate synthetic calibration data for testing Stage1 tuning.
This creates realistic calibration data based on typical distributions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_scifact_like(n=1000, seed=14):
    """
    Generate SciFact-like calibration data.
    SciFact typically has ~65% positive rate (relevant docs found).
    Logits from cross-encoder reranking follow a bimodal distribution.
    """
    rng = np.random.default_rng(seed)
    
    # ~65% positive rate for SciFact
    y = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    
    # Generate logits (raw_max_top3 from cross-encoder)
    # Positive cases: higher logits, centered around 0.75-0.90
    # Negative cases: lower logits, centered around 0.50-0.70
    logits = []
    for label in y:
        if label == 1:
            # Positive: higher confidence, some noise
            logit = rng.beta(8, 3) * 0.5 + 0.5  # Range ~0.5-1.0, peak ~0.85
        else:
            # Negative: lower but overlapping
            logit = rng.beta(4, 4) * 0.6 + 0.3  # Range ~0.3-0.9, peak ~0.6
        logits.append(logit)
    
    df = pd.DataFrame({
        "qid": [f"q{i}" for i in range(n)],
        "raw_max_top3": logits,
        "y": y,
        # Additional features that may be used
        "top1": logits,
        "top2": [max(0.1, l - rng.uniform(0.05, 0.15)) for l in logits],
        "top3": [max(0.05, l - rng.uniform(0.1, 0.25)) for l in logits],
    })
    
    # Add derived features
    df["gap12"] = df["top1"] - df["top2"]
    df["mean_top3"] = (df["top1"] + df["top2"] + df["top3"]) / 3
    df["std_top3"] = df[["top1", "top2", "top3"]].std(axis=1)
    
    return df


def generate_fever_like(n=1000, seed=14):
    """
    Generate FEVER-like calibration data.
    FEVER typically has ~50% positive rate (evidence supports claim).
    Uses logit_platt from calibrated retrieval.
    """
    rng = np.random.default_rng(seed)
    
    # ~50% positive rate for FEVER
    y = rng.choice([0, 1], size=n, p=[0.50, 0.50])
    
    # Generate logits (logit_platt - can be negative)
    # FEVER has more challenging separation
    logits = []
    for label in y:
        if label == 1:
            # Positive: higher logits, centered around 0-2
            logit = rng.normal(1.0, 0.8)
        else:
            # Negative: lower logits, centered around -2 to 0
            logit = rng.normal(-0.5, 1.0)
        logits.append(logit)
    
    df = pd.DataFrame({
        "qid": range(n),
        "logit_platt": logits,
        "y": y,
        "top1": [1 / (1 + np.exp(-l)) for l in logits],  # Sigmoid transform for similarity
    })
    
    # Add more features
    df["top2"] = df["top1"] * rng.uniform(0.7, 0.95, n)
    df["top3"] = df["top2"] * rng.uniform(0.7, 0.95, n)
    df["gap12"] = df["top1"] - df["top2"]
    df["mean_top3"] = (df["top1"] + df["top2"] + df["top3"]) / 3
    
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate calibration data for testing")
    ap.add_argument("--track", choices=["scifact", "fever", "both"], default="both")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out_dir", default="data/calibration")
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.track in ("scifact", "both"):
        df = generate_scifact_like(args.n, args.seed)
        out_path = out_dir / "scifact_stage1_dev_train.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Generated SciFact data: {out_path}")
        print(f"  n={len(df)}, positive_rate={df['y'].mean():.2%}")
        print(f"  logit range: [{df['raw_max_top3'].min():.3f}, {df['raw_max_top3'].max():.3f}]")
    
    if args.track in ("fever", "both"):
        df = generate_fever_like(args.n, args.seed)
        out_path = out_dir / "fever_stage1_dev_train.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Generated FEVER data: {out_path}")
        print(f"  n={len(df)}, positive_rate={df['y'].mean():.2%}")
        print(f"  logit range: [{df['logit_platt'].min():.3f}, {df['logit_platt'].max():.3f}]")


if __name__ == "__main__":
    main()
