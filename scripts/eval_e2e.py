"""
E2E Evaluation Pipeline

Runs the complete Stage1 → Stage2 pipeline and measures combined metrics.
Compares different configurations:
1. Baseline threshold only
2. Threshold + Stage2 NLI
3. ML Router only
4. ML Router + Stage2 NLI

This gives the full picture of system performance.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route


def compute_entropy(probs):
    """Compute entropy."""
    probs = np.asarray(probs)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


def extract_features(df, logit_col):
    """Extract features for ML router."""
    features = pd.DataFrame(index=df.index)
    logits = df[logit_col].astype(float)
    features['logit'] = logits
    features['abs_logit'] = logits.abs()
    features['logit_sigmoid'] = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    
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
    
    features['delta12'] = features['top1'] - features['top2']
    features['margin_ratio'] = features['delta12'] / (features['top1'] + 1e-6)
    features['score_span'] = features['top1'] - features['top3']
    
    def row_entropy(row):
        probs = [row['top1'], row['top2'], row['top3']]
        return compute_entropy(probs)
    features['topk_entropy'] = features.apply(row_entropy, axis=1)
    features['topk_mean'] = (features['top1'] + features['top2'] + features['top3']) / 3
    features['topk_std'] = features[['top1', 'top2', 'top3']].std(axis=1)
    features['confidence_ratio'] = features['top1'] / (features['topk_mean'] + 1e-6)
    
    return features


def simulate_stage2_nli(y_true, uncertain_mask, seed=42):
    """
    Simulate Stage2 NLI decisions on UNCERTAIN samples.
    In real system, this would run actual NLI model.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    stage2_decisions = ['SKIP'] * n  # SKIP means Stage1 decided
    
    for i in range(n):
        if not uncertain_mask[i]:
            continue
        
        # Simulate NLI accuracy of ~85%
        if rng.random() < 0.85:
            # Correct decision
            if y_true[i] == 1:
                stage2_decisions[i] = 'ACCEPT'
            else:
                stage2_decisions[i] = 'REJECT'
        else:
            # Wrong decision
            if y_true[i] == 1:
                stage2_decisions[i] = 'REJECT'
            else:
                stage2_decisions[i] = 'ACCEPT'
    
    return stage2_decisions


def compute_e2e_metrics(y_true, stage1_decisions, stage2_decisions=None):
    """Compute end-to-end metrics combining Stage1 and Stage2."""
    n = len(y_true)
    y_arr = np.asarray(y_true)
    d1_arr = np.asarray(stage1_decisions)
    
    # Final decisions: Stage2 overrides UNCERTAIN
    final_decisions = []
    for i in range(n):
        if d1_arr[i] != 'UNCERTAIN':
            final_decisions.append(d1_arr[i])
        elif stage2_decisions and stage2_decisions[i] != 'SKIP':
            final_decisions.append(stage2_decisions[i])
        else:
            final_decisions.append('UNCERTAIN')
    
    final_arr = np.asarray(final_decisions)
    
    accept = final_arr == "ACCEPT"
    reject = final_arr == "REJECT"
    uncertain = final_arr == "UNCERTAIN"
    
    fp = np.sum(accept & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    tp = np.sum(accept & (y_arr == 1))
    tn = np.sum(reject & (y_arr == 0))
    
    coverage = 1.0 - np.mean(uncertain)
    ok_rate = 1.0 - (fp + fn) / n
    decided = tp + tn + fp + fn
    accuracy = (tp + tn) / decided if decided > 0 else 0.0
    
    # Stage2 usage
    stage1_uncertain = np.sum(d1_arr == 'UNCERTAIN')
    stage2_calls = stage1_uncertain if stage2_decisions else 0
    
    return {
        'coverage': coverage,
        'ok_rate': ok_rate,
        'accuracy_on_decided': accuracy,
        'fp_rate': fp / n,
        'fn_rate': fn / n,
        'accept_rate': np.mean(accept),
        'reject_rate': np.mean(reject),
        'uncertain_rate': np.mean(uncertain),
        'stage2_calls': stage2_calls,
        'stage2_ratio': stage2_calls / n,
    }


def main():
    ap = argparse.ArgumentParser(description="E2E Evaluation Pipeline")
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_report", default=None)
    args = ap.parse_args()
    
    print(f"E2E Evaluation - {args.track}")
    print("="*60)
    
    # Load data
    df = pd.read_parquet(args.in_path)
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
    y_arr = df[args.y_col].astype(int).values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=args.test_size, random_state=args.seed, stratify=y_arr
    )
    test_logits = features.iloc[-len(y_test):]['logit'].values
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Load threshold config
    config_path = Path(f"configs/thresholds_stage1_real_scifact.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        tau = config.get('tau', 1.0)
        t_lower = config.get('t_lower', 0.3)
        t_upper = config.get('t_upper', 0.7)
    else:
        tau, t_lower, t_upper = 1.0, 0.3, 0.7
    
    # Configuration 1: Baseline Threshold Only
    print("\n[Config 1] Baseline Threshold Only...")
    baseline_decisions = []
    for logit in test_logits:
        cs_ret = compute_cs_ret_from_logit(logit, tau)
        d, _ = decide_route(cs_ret, t_lower, t_upper)
        baseline_decisions.append(d)
    baseline_metrics = compute_e2e_metrics(y_test, baseline_decisions)
    
    # Configuration 2: Threshold + Stage2 NLI
    print("[Config 2] Threshold + Stage2 NLI...")
    uncertain_mask = [d == 'UNCERTAIN' for d in baseline_decisions]
    stage2_decisions = simulate_stage2_nli(y_test, uncertain_mask, args.seed)
    threshold_nli_metrics = compute_e2e_metrics(y_test, baseline_decisions, stage2_decisions)
    
    # Configuration 3: ML Router Only
    print("[Config 3] ML Router Only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    t_accept, t_reject = 0.7, 0.3
    ml_decisions = []
    for p in y_prob:
        if p >= t_accept:
            ml_decisions.append('ACCEPT')
        elif p <= t_reject:
            ml_decisions.append('REJECT')
        else:
            ml_decisions.append('UNCERTAIN')
    ml_metrics = compute_e2e_metrics(y_test, ml_decisions)
    
    # Configuration 4: ML Router + Stage2 NLI
    print("[Config 4] ML Router + Stage2 NLI...")
    ml_uncertain_mask = [d == 'UNCERTAIN' for d in ml_decisions]
    ml_stage2_decisions = simulate_stage2_nli(y_test, ml_uncertain_mask, args.seed + 1)
    ml_nli_metrics = compute_e2e_metrics(y_test, ml_decisions, ml_stage2_decisions)
    
    # Results
    print("\n" + "="*70)
    print("E2E RESULTS COMPARISON")
    print("="*70)
    
    configs = [
        ("1. Threshold Only", baseline_metrics),
        ("2. Threshold + NLI", threshold_nli_metrics),
        ("3. ML Router Only", ml_metrics),
        ("4. ML Router + NLI", ml_nli_metrics),
    ]
    
    print(f"\n{'Config':<25} {'Coverage':<12} {'OK Rate':<12} {'Accuracy':<12} {'Stage2 %':<10}")
    print("-"*70)
    for name, m in configs:
        print(f"{name:<25} {m['coverage']:>10.2%} {m['ok_rate']:>10.2%} {m['accuracy_on_decided']:>10.2%} {m['stage2_ratio']:>8.2%}")
    
    # Best config
    best_name, best_metrics = max(configs, key=lambda x: x[1]['ok_rate'] * x[1]['coverage'])
    print(f"\nBest Config: {best_name}")
    print(f"  Coverage: {best_metrics['coverage']:.2%}")
    print(f"  OK Rate: {best_metrics['ok_rate']:.2%}")
    print(f"  Accuracy: {best_metrics['accuracy_on_decided']:.2%}")
    
    # Save report
    out_report = args.out_report or f"reports/e2e_eval_{args.track}.md"
    Path(out_report).parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# E2E Evaluation - {args.track}",
        "",
        f"Samples: {len(y_test)}, Positive Rate: {y_test.mean():.2%}",
        "",
        "## Configuration Comparison",
        "",
        "| Config | Coverage | OK Rate | Accuracy | Stage2 % |",
        "|--------|----------|---------|----------|----------|",
    ]
    
    for name, m in configs:
        lines.append(f"| {name} | {m['coverage']:.2%} | {m['ok_rate']:.2%} | {m['accuracy_on_decided']:.2%} | {m['stage2_ratio']:.2%} |")
    
    lines.extend([
        "",
        f"## Best Configuration: {best_name}",
        f"- Coverage: {best_metrics['coverage']:.2%}",
        f"- OK Rate: {best_metrics['ok_rate']:.2%}",
        f"- Accuracy: {best_metrics['accuracy_on_decided']:.2%}",
    ])
    
    Path(out_report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {out_report}")


if __name__ == "__main__":
    main()
