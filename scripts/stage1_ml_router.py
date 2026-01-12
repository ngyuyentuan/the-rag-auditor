"""
Stage1 ML Router with Enhanced Features

This script implements:
1. Enhanced feature engineering (topk_entropy, score_span, margin_ratio)
2. ML-based routing using LogisticRegression
3. Comparison with threshold-only baseline

Key improvement: Use ML to learn optimal decision boundary instead of manual thresholds.
"""
import argparse
import sys
from pathlib import Path
from math import sqrt, log

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route


def compute_entropy(probs):
    """Compute entropy of probability distribution."""
    probs = np.asarray(probs)
    probs = probs[probs > 0]  # Filter zeros
    if len(probs) == 0:
        return 0.0
    probs = probs / probs.sum()  # Normalize
    return -np.sum(probs * np.log(probs + 1e-10))


def extract_features(df, logit_col):
    """Extract enhanced features from calibration data."""
    features = pd.DataFrame(index=df.index)
    
    # Base logit/score
    logits = df[logit_col].astype(float)
    features['logit'] = logits
    features['abs_logit'] = logits.abs()
    
    # Sigmoid transformation
    features['logit_sigmoid'] = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    
    # Top-k similarity features (if available)
    if 'top1' in df.columns:
        features['top1'] = df['top1'].astype(float)
    else:
        features['top1'] = features['logit_sigmoid']
    
    if 'top2' in df.columns:
        features['top2'] = df['top2'].astype(float)
    else:
        features['top2'] = features['top1'] * 0.9
    
    if 'top3' in df.columns:
        features['top3'] = df['top3'].astype(float)
    else:
        features['top3'] = features['top2'] * 0.9
    
    # Delta features
    features['delta12'] = features['top1'] - features['top2']
    features['delta23'] = features['top2'] - features['top3']
    
    # New: margin_ratio (relative gap)
    features['margin_ratio'] = features['delta12'] / (features['top1'] + 1e-6)
    
    # New: score_span (range of scores)
    features['score_span'] = features['top1'] - features['top3']
    
    # New: topk_entropy (uncertainty measure)
    def row_entropy(row):
        probs = [row['top1'], row['top2'], row['top3']]
        return compute_entropy(probs)
    features['topk_entropy'] = features.apply(row_entropy, axis=1)
    
    # Mean and std
    features['topk_mean'] = (features['top1'] + features['top2'] + features['top3']) / 3
    features['topk_std'] = features[['top1', 'top2', 'top3']].std(axis=1)
    
    # Confidence ratio
    features['confidence_ratio'] = features['top1'] / (features['topk_mean'] + 1e-6)
    
    return features


def train_ml_router(X_train, y_train, X_test, y_test, feature_names):
    """Train ML router and evaluate."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train LogisticRegression
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_prob': y_prob,
        'y_pred': y_pred,
        'feature_importance': importance
    }


def compute_routing_metrics(y_true, decisions, y_prob=None):
    """Compute Stage1 routing metrics."""
    n = len(y_true)
    y_arr = np.asarray(y_true)
    d_arr = np.asarray(decisions)
    
    accept = d_arr == "ACCEPT"
    reject = d_arr == "REJECT"
    uncertain = d_arr == "UNCERTAIN"
    
    fp = np.sum(accept & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    tp = np.sum(accept & (y_arr == 1))
    tn = np.sum(reject & (y_arr == 0))
    
    coverage = 1.0 - np.mean(uncertain)
    ok_rate = 1.0 - (fp + fn) / n
    decided = tp + tn + fp + fn
    accuracy_on_decided = (tp + tn) / decided if decided else 0.0
    
    return {
        'coverage': coverage,
        'ok_rate': ok_rate,
        'accuracy_on_decided': accuracy_on_decided,
        'fp_rate': fp / n,
        'fn_rate': fn / n,
        'accept_rate': np.mean(accept),
        'reject_rate': np.mean(reject),
        'uncertain_rate': np.mean(uncertain),
    }


def ml_router_decision(prob_positive, t_accept=0.7, t_reject=0.3):
    """Make routing decision based on ML probability."""
    if prob_positive >= t_accept:
        return "ACCEPT"
    elif prob_positive <= t_reject:
        return "REJECT"
    else:
        return "UNCERTAIN"


def main():
    ap = argparse.ArgumentParser(description="Stage1 ML Router with Enhanced Features")
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True, help="Path to calibration parquet")
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t_accept", type=float, default=0.7)
    ap.add_argument("--t_reject", type=float, default=0.3)
    ap.add_argument("--out_report", default=None)
    args = ap.parse_args()
    
    # Load data
    df = pd.read_parquet(args.in_path)
    
    # Clean
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna()
    df = df.loc[mask].copy()
    
    n_samples = len(df)
    print(f"Loaded {n_samples} samples, positive rate: {df[args.y_col].mean():.2%}")
    
    # Extract features
    features = extract_features(df, args.logit_col)
    feature_cols = [
        'logit', 'abs_logit', 'logit_sigmoid',
        'top1', 'delta12', 'margin_ratio', 'score_span',
        'topk_entropy', 'topk_mean', 'topk_std', 'confidence_ratio'
    ]
    
    X = features[feature_cols].fillna(0).values
    y = df[args.y_col].astype(int).values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train ML router
    print("\n[Phase 1] Training ML Router...")
    ml_results = train_ml_router(X_train, y_train, X_test, y_test, feature_cols)
    
    print(f"\n=== ML Router Results ===")
    print(f"Accuracy:  {ml_results['accuracy']:.2%}")
    print(f"Precision: {ml_results['precision']:.2%}")
    print(f"Recall:    {ml_results['recall']:.2%}")
    print(f"F1:        {ml_results['f1']:.2%}")
    print(f"CV Mean:   {ml_results['cv_mean']:.2%} (+/- {ml_results['cv_std']:.2%})")
    
    print("\nTop Features:")
    print(ml_results['feature_importance'].head(5).to_string(index=False))
    
    # Make routing decisions with ML
    y_prob_positive = ml_results['y_prob'][:, 1]
    ml_decisions = [ml_router_decision(p, args.t_accept, args.t_reject) for p in y_prob_positive]
    ml_metrics = compute_routing_metrics(y_test, ml_decisions)
    
    print(f"\n=== ML Router Routing Metrics ===")
    print(f"Coverage:           {ml_metrics['coverage']:.2%}")
    print(f"OK Rate:            {ml_metrics['ok_rate']:.2%}")
    print(f"Accuracy (decided): {ml_metrics['accuracy_on_decided']:.2%}")
    print(f"FP Rate:            {ml_metrics['fp_rate']:.2%}")
    print(f"FN Rate:            {ml_metrics['fn_rate']:.2%}")
    
    # Compare with baseline threshold
    print("\n[Phase 2] Comparing with Threshold Baseline...")
    
    # Load existing threshold config
    config_path = Path(f"configs/thresholds_stage1_real_scifact.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        tau = config.get('tau', 1.0)
        t_lower = config.get('t_lower', 0.3)
        t_upper = config.get('t_upper', 0.7)
    else:
        tau, t_lower, t_upper = 1.0, 0.3, 0.7
    
    # Test set baseline
    test_indices = range(len(X_train), len(X))
    test_logits = features.iloc[list(test_indices)]['logit'].values
    baseline_decisions = []
    for logit in test_logits:
        cs_ret = compute_cs_ret_from_logit(logit, tau)
        d, _ = decide_route(cs_ret, t_lower, t_upper)
        baseline_decisions.append(d)
    
    baseline_metrics = compute_routing_metrics(y_test, baseline_decisions)
    
    print(f"\n=== Baseline Threshold Routing Metrics ===")
    print(f"Coverage:           {baseline_metrics['coverage']:.2%}")
    print(f"OK Rate:            {baseline_metrics['ok_rate']:.2%}")
    print(f"Accuracy (decided): {baseline_metrics['accuracy_on_decided']:.2%}")
    
    # Comparison
    print("\n" + "="*50)
    print("COMPARISON: ML Router vs Baseline Threshold")
    print("="*50)
    print(f"{'Metric':<20} {'ML Router':<15} {'Baseline':<15} {'Diff':<10}")
    print("-"*50)
    for m in ['coverage', 'ok_rate', 'accuracy_on_decided', 'fp_rate', 'fn_rate']:
        ml_val = ml_metrics[m]
        bl_val = baseline_metrics[m]
        diff = ml_val - bl_val
        sign = "+" if diff > 0 else ""
        print(f"{m:<20} {ml_val:>12.2%} {bl_val:>12.2%} {sign}{diff:>8.2%}")
    
    # Save report
    out_report = args.out_report or f"reports/ml_router_{args.track}.md"
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# ML Router Results - {args.track}",
        "",
        "## Model Performance",
        f"- Accuracy: {ml_results['accuracy']:.2%}",
        f"- F1 Score: {ml_results['f1']:.2%}",
        f"- CV Mean: {ml_results['cv_mean']:.2%}",
        "",
        "## Routing Comparison",
        "",
        "| Metric | ML Router | Baseline | Improvement |",
        "|--------|-----------|----------|-------------|",
    ]
    
    for m in ['coverage', 'ok_rate', 'accuracy_on_decided']:
        ml_val = ml_metrics[m]
        bl_val = baseline_metrics[m]
        diff = ml_val - bl_val
        sign = "+" if diff > 0 else ""
        lines.append(f"| {m} | {ml_val:.2%} | {bl_val:.2%} | {sign}{diff:.2%} |")
    
    lines.extend([
        "",
        "## Top Features",
        "",
        "| Feature | Importance |",
        "|---------|------------|",
    ])
    for _, row in ml_results['feature_importance'].head(5).iterrows():
        lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
    
    Path(out_report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report: {out_report}")


if __name__ == "__main__":
    main()
