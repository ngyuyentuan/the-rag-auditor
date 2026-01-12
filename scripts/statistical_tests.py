"""
Statistical Significance Tests for RAG Auditor
"""
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUPPORTS = [
    ("Vaccines are safe.", "Research confirms vaccines are safe and effective."),
    ("Exercise improves health.", "Studies show exercise improves health."),
]

REFUTES = [
    ("Vaccines are dangerous.", "Research confirms vaccines are safe and effective."),
    ("Earth is flat.", "Scientific evidence proves Earth is spherical."),
]

NEI = [
    ("AI will replace jobs.", "The impact is debated and uncertain. Experts disagree."),
    ("Coffee is good for health.", "Studies show mixed and inconclusive results."),
]


def generate_data(n: int) -> List[Tuple[str, str, str]]:
    data = []
    for _ in range(n // 3):
        data.append((*random.choice(SUPPORTS), "SUPPORTS"))
        data.append((*random.choice(REFUTES), "REFUTES"))
        data.append((*random.choice(NEI), "NEI"))
    random.shuffle(data)
    return data


def run_single_experiment(auditor, test_data: List) -> float:
    correct = 0
    for claim, evidence, expected in test_data:
        try:
            result = auditor.audit(claim, evidence, skip_cache=True)
            if result.verdict == expected:
                correct += 1
        except:
            pass
    return 100 * correct / len(test_data)


def run_statistical_tests(n_runs: int = 10, samples_per_run: int = 300, output_file: str = None):
    print("=" * 60)
    print(f"STATISTICAL SIGNIFICANCE TESTS ({n_runs} runs)")
    print("=" * 60)
    
    from src.auditor.ultimate_auditor import UltimateAuditor
    auditor = UltimateAuditor(device="cpu")
    
    results = []
    for i in range(n_runs):
        test_data = generate_data(samples_per_run)
        acc = run_single_experiment(auditor, test_data)
        results.append(acc)
        print(f"  Run {i+1}/{n_runs}: {acc:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results, ddof=1)
    se_acc = std_acc / np.sqrt(n_runs)
    ci_95 = stats.t.interval(0.95, n_runs - 1, loc=mean_acc, scale=se_acc)
    
    print(f"\n{'=' * 60}")
    print("STATISTICAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean Accuracy:     {mean_acc:.2f}%")
    print(f"Std Deviation:     {std_acc:.2f}%")
    print(f"Standard Error:    {se_acc:.2f}%")
    print(f"95% CI:            [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%]")
    
    baseline_acc = 50.0
    t_stat, p_value = stats.ttest_1samp(results, baseline_acc)
    print(f"\nT-test vs baseline (50%):")
    print(f"  t-statistic:     {t_stat:.4f}")
    print(f"  p-value:         {p_value:.2e}")
    print(f"  Significant:     {'Yes (p < 0.001)' if p_value < 0.001 else 'No'}")
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Statistical Significance Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Experiment Setup\n\n")
            f.write(f"- Number of runs: {n_runs}\n")
            f.write(f"- Samples per run: {samples_per_run}\n\n")
            f.write("## Results\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Mean Accuracy | {mean_acc:.2f}% |\n")
            f.write(f"| Std Deviation | {std_acc:.2f}% |\n")
            f.write(f"| Standard Error | {se_acc:.2f}% |\n")
            f.write(f"| 95% CI | [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%] |\n\n")
            f.write("## T-Test (vs 50% baseline)\n\n")
            f.write(f"- t-statistic: {t_stat:.4f}\n")
            f.write(f"- p-value: {p_value:.2e}\n")
            f.write(f"- **Statistically significant: {'Yes' if p_value < 0.05 else 'No'}**\n")
        print(f"\nSaved: {output_file}")
    
    return {"mean": mean_acc, "std": std_acc, "ci": ci_95, "p_value": p_value}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--output", type=str, default="reports/statistical_tests.md")
    args = parser.parse_args()
    
    run_statistical_tests(args.runs, args.samples, args.output)
