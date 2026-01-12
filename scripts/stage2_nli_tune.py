"""
Stage2 NLI Threshold Tuning

This script tunes the NLI output thresholds for Stage2 verification.
Stage2 runs on UNCERTAIN samples from Stage1 and uses NLI to determine:
- ENTAILMENT → SUPPORTS (ACCEPT)
- CONTRADICTION → REFUTES (REJECT)
- NEUTRAL → NEI (UNCERTAIN)

We tune thresholds for each class to optimize accuracy.
"""
import argparse
import sys
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def wilson_ci(x, n):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    phat = x / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def make_nli_decision(probs, t_entail=0.5, t_contra=0.5, t_neutral=0.5):
    """
    Make NLI-based decision from probability distribution.
    
    Args:
        probs: dict with keys 'entailment', 'contradiction', 'neutral'
        t_entail: threshold for ENTAILMENT → ACCEPT
        t_contra: threshold for CONTRADICTION → REJECT
        t_neutral: threshold for NEUTRAL → NEI
    
    Returns:
        decision: 'ACCEPT', 'REJECT', or 'UNCERTAIN'
    """
    p_entail = probs.get('entailment', 0.0)
    p_contra = probs.get('contradiction', 0.0)
    p_neutral = probs.get('neutral', 0.0)
    
    # Check strongest signal above threshold
    if p_entail >= t_entail and p_entail > p_contra and p_entail > p_neutral:
        return 'ACCEPT', 'entailment'
    elif p_contra >= t_contra and p_contra > p_entail and p_contra > p_neutral:
        return 'REJECT', 'contradiction'
    elif p_neutral >= t_neutral:
        return 'UNCERTAIN', 'neutral'
    else:
        # Default: use argmax with no threshold
        max_label = max(probs, key=probs.get)
        if max_label == 'entailment':
            return 'ACCEPT', 'entailment_argmax'
        elif max_label == 'contradiction':
            return 'REJECT', 'contradiction_argmax'
        else:
            return 'UNCERTAIN', 'neutral_argmax'


def compute_nli_metrics(y_true, decisions, stage1_decisions=None):
    """Compute metrics for NLI-based decisions."""
    n = len(y_true)
    y_arr = np.asarray(y_true)
    d_arr = np.asarray(decisions)
    
    accept = d_arr == "ACCEPT"
    reject = d_arr == "REJECT"
    uncertain = d_arr == "UNCERTAIN"
    
    # Stage2 only runs on UNCERTAIN from Stage1
    # y=1 means evidence supports claim, y=0 means doesn't support
    
    # For Stage2: correct ACCEPT when y=1, correct REJECT when y=0
    tp = np.sum(accept & (y_arr == 1))
    fp = np.sum(accept & (y_arr == 0))
    tn = np.sum(reject & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    defer = np.sum(uncertain)
    
    coverage = 1.0 - defer / n if n > 0 else 0.0
    decided = tp + fp + tn + fn
    accuracy = (tp + tn) / decided if decided > 0 else 0.0
    ok_rate = 1.0 - (fp + fn) / n if n > 0 else 0.0
    
    return {
        'n': n,
        'coverage': coverage,
        'accuracy_on_decided': accuracy,
        'ok_rate': ok_rate,
        'fp_rate': fp / n if n > 0 else 0.0,
        'fn_rate': fn / n if n > 0 else 0.0,
        'accept_rate': np.mean(accept),
        'reject_rate': np.mean(reject),
        'uncertain_rate': np.mean(uncertain),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def grid_search_nli(samples, y_true, steps=20):
    """Grid search over NLI thresholds."""
    grid = np.linspace(0.3, 0.9, steps)
    
    best_config = None
    best_utility = -float('inf')
    all_configs = []
    
    for t_entail in grid:
        for t_contra in grid:
            for t_neutral in [0.5]:  # Keep neutral fixed
                decisions = []
                for probs in samples:
                    d, _ = make_nli_decision(probs, t_entail, t_contra, t_neutral)
                    decisions.append(d)
                
                metrics = compute_nli_metrics(y_true, decisions)
                
                # Utility: maximize coverage and accuracy, minimize errors
                utility = metrics['coverage'] * 0.5 + metrics['accuracy_on_decided'] * 0.5 - 5 * (metrics['fp_rate'] + metrics['fn_rate'])
                
                config = {
                    't_entail': t_entail,
                    't_contra': t_contra,
                    't_neutral': t_neutral,
                    'utility': utility,
                    **metrics
                }
                all_configs.append(config)
                
                if utility > best_utility:
                    best_utility = utility
                    best_config = config
    
    return best_config, all_configs


def generate_synthetic_nli_data(n=500, seed=14):
    """
    Generate synthetic NLI probability data for testing.
    Simulates what an NLI model would output.
    """
    rng = np.random.default_rng(seed)
    
    samples = []
    y_true = []
    
    for i in range(n):
        label = rng.choice([0, 1])
        y_true.append(label)
        
        if label == 1:
            # Evidence supports claim - higher entailment
            p_entail = rng.beta(5, 2)  # Skewed toward high
            p_contra = rng.beta(1, 4)  # Skewed toward low
            p_neutral = 1 - p_entail - p_contra
            if p_neutral < 0:
                p_neutral = 0.1
                total = p_entail + p_contra + p_neutral
                p_entail /= total
                p_contra /= total
                p_neutral /= total
        else:
            # Evidence doesn't support - could be contra or neutral
            if rng.random() < 0.5:
                # Contradiction case
                p_contra = rng.beta(5, 2)
                p_entail = rng.beta(1, 4)
                p_neutral = 1 - p_entail - p_contra
            else:
                # Neutral case
                p_neutral = rng.beta(5, 2)
                p_entail = rng.beta(1, 4)
                p_contra = 1 - p_entail - p_neutral
            
            if p_neutral < 0:
                p_neutral = 0.1
            total = p_entail + p_contra + p_neutral
            p_entail /= total
            p_contra /= total
            p_neutral /= total
        
        samples.append({
            'entailment': float(p_entail),
            'contradiction': float(p_contra),
            'neutral': float(p_neutral)
        })
    
    return samples, y_true


def main():
    ap = argparse.ArgumentParser(description="Stage2 NLI Threshold Tuning")
    ap.add_argument("--track", choices=["scifact", "fever"], default="scifact")
    ap.add_argument("--n", type=int, default=500, help="Number of synthetic samples")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--steps", type=int, default=15, help="Grid search steps per threshold")
    ap.add_argument("--out_yaml", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()
    
    print(f"Stage2 NLI Tuning - {args.track}")
    print("="*50)
    
    # Generate synthetic NLI data (in real scenario, use actual NLI outputs)
    print(f"\nGenerating {args.n} synthetic NLI samples...")
    samples, y_true = generate_synthetic_nli_data(args.n, args.seed)
    
    pos_rate = sum(y_true) / len(y_true)
    print(f"Samples: {len(samples)}, Positive rate: {pos_rate:.2%}")
    
    # Baseline: use argmax only
    print("\n[Baseline] Using argmax...")
    baseline_decisions = []
    for probs in samples:
        max_label = max(probs, key=probs.get)
        if max_label == 'entailment':
            baseline_decisions.append('ACCEPT')
        elif max_label == 'contradiction':
            baseline_decisions.append('REJECT')
        else:
            baseline_decisions.append('UNCERTAIN')
    
    baseline_metrics = compute_nli_metrics(y_true, baseline_decisions)
    print(f"Baseline Coverage: {baseline_metrics['coverage']:.2%}")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy_on_decided']:.2%}")
    print(f"Baseline OK Rate:  {baseline_metrics['ok_rate']:.2%}")
    
    # Grid search
    print(f"\n[Tuning] Grid search ({args.steps}^2 configs)...")
    best, all_configs = grid_search_nli(samples, y_true, args.steps)
    
    print("\n" + "="*50)
    print("BEST CONFIG FOUND")
    print("="*50)
    print(f"  t_entail:  {best['t_entail']:.4f}")
    print(f"  t_contra:  {best['t_contra']:.4f}")
    print(f"  t_neutral: {best['t_neutral']:.4f}")
    print()
    print(f"METRICS:")
    print(f"  Coverage:           {best['coverage']:.2%}")
    print(f"  Accuracy (decided): {best['accuracy_on_decided']:.2%}")
    print(f"  OK Rate:            {best['ok_rate']:.2%}")
    print(f"  FP Rate:            {best['fp_rate']:.2%}")
    print(f"  FN Rate:            {best['fn_rate']:.2%}")
    print(f"  Utility:            {best['utility']:.4f}")
    
    # Save YAML
    out_yaml = args.out_yaml or f"configs/thresholds_stage2_nli_{args.track}.yaml"
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        't_entail': float(best['t_entail']),
        't_contra': float(best['t_contra']),
        't_neutral': float(best['t_neutral']),
        'metrics': {
            'coverage': float(best['coverage']),
            'accuracy_on_decided': float(best['accuracy_on_decided']),
            'ok_rate': float(best['ok_rate']),
            'fp_rate': float(best['fp_rate']),
            'fn_rate': float(best['fn_rate']),
        }
    }
    Path(out_yaml).write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    print(f"\nSaved: {out_yaml}")
    
    # Save report
    out_md = args.out_md or f"reports/stage2_nli_{args.track}.md"
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# Stage2 NLI Tuning - {args.track}",
        "",
        "## Best Thresholds",
        f"- t_entail: `{best['t_entail']:.4f}`",
        f"- t_contra: `{best['t_contra']:.4f}`",
        f"- t_neutral: `{best['t_neutral']:.4f}`",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Tuned | Improvement |",
        "|--------|----------|-------|-------------|",
        f"| Coverage | {baseline_metrics['coverage']:.2%} | {best['coverage']:.2%} | {best['coverage']-baseline_metrics['coverage']:+.2%} |",
        f"| Accuracy | {baseline_metrics['accuracy_on_decided']:.2%} | {best['accuracy_on_decided']:.2%} | {best['accuracy_on_decided']-baseline_metrics['accuracy_on_decided']:+.2%} |",
        f"| OK Rate | {baseline_metrics['ok_rate']:.2%} | {best['ok_rate']:.2%} | {best['ok_rate']-baseline_metrics['ok_rate']:+.2%} |",
    ]
    
    Path(out_md).write_text("\n".join(lines), encoding='utf-8')
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
