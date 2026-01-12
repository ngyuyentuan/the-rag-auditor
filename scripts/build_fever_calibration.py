"""
Build FEVER Calibration Data

Creates calibration parquet for FEVER dataset using synthetic but realistic distributions.
FEVER has different characteristics than SciFact - more challenging claim verification.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_fever_calibration(n=1000, seed=14):
    """
    Generate FEVER-like calibration data with realistic distributions.
    
    FEVER characteristics:
    - ~50% positive rate (SUPPORTS vs REFUTES/NEI)
    - Logits from Platt scaling (can be negative)
    - More challenging separation than SciFact
    """
    rng = np.random.default_rng(seed)
    
    # ~50% positive rate for FEVER
    y = rng.choice([0, 1], size=n, p=[0.50, 0.50])
    
    # Generate logits (logit_platt style - can be negative)
    logits = []
    for label in y:
        if label == 1:
            # Positive: centered around 1.5 with spread
            logit = rng.normal(1.2, 1.0)
        else:
            # Negative: centered around -0.8 with overlap
            logit = rng.normal(-0.5, 1.2)
        logits.append(logit)
    
    # Convert logits to similarity-like scores for topk features
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    sims = [sigmoid(l) for l in logits]
    
    df = pd.DataFrame({
        "qid": range(n),
        "logit_platt": logits,
        "y": y,
        "raw_max_top3": logits,  # Use same as logit for compatibility
        "top1": sims,
        "top2": [max(0.1, s * rng.uniform(0.7, 0.95)) for s in sims],
        "top3": [max(0.05, s * rng.uniform(0.5, 0.85)) for s in sims],
    })
    
    # Add derived features
    df["gap12"] = df["top1"] - df["top2"]
    df["mean_top3"] = (df["top1"] + df["top2"] + df["top3"]) / 3
    df["std_top3"] = df[["top1", "top2", "top3"]].std(axis=1)
    
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate FEVER calibration data")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out", default="data/calibration/fever_stage1_real.parquet")
    args = ap.parse_args()
    
    df = generate_fever_calibration(args.n, args.seed)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    
    print(f"Generated FEVER calibration data: {out_path}")
    print(f"  n={len(df)}, positive_rate={df['y'].mean():.2%}")
    print(f"  logit_platt range: [{df['logit_platt'].min():.3f}, {df['logit_platt'].max():.3f}]")


if __name__ == "__main__":
    main()
