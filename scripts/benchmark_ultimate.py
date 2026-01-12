"""
Benchmark script for Ultimate Auditor V4
Tests semantic similarity-based NEI detection
"""
import sys
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.ultimate_auditor import UltimateAuditor


SUPPORTS_TEMPLATES = [
    ("The vaccine is safe.", "Research confirms the vaccine is safe and effective."),
    ("Exercise improves health.", "Studies show that exercise improves health outcomes."),
    ("Water is essential for life.", "Scientific evidence proves water is essential for survival."),
    ("Education increases income.", "Data shows educated people earn higher incomes."),
    ("Sleep is important.", "Research proves adequate sleep is crucial for health."),
    ("Fruits are nutritious.", "Fruits contain essential vitamins and minerals."),
    ("Smoking causes cancer.", "Studies confirm smoking causes lung cancer."),
    ("Reading enhances vocabulary.", "Reading regularly expands vocabulary."),
    ("Sunlight provides vitamin D.", "Sunlight exposure enables vitamin D synthesis."),
    ("Handwashing prevents disease.", "Hand hygiene prevents disease transmission."),
]

REFUTES_TEMPLATES = [
    ("The vaccine is dangerous.", "Research confirms the vaccine is safe and effective."),
    ("Exercise is harmful.", "Studies show that exercise improves health and is beneficial."),
    ("The Earth is flat.", "Scientific evidence proves Earth is spherical, not flat."),
    ("Vaccines cause autism.", "No scientific evidence links vaccines to autism."),
    ("5G causes COVID.", "There is no connection between 5G and COVID-19."),
    ("Coffee is deadly poison.", "Coffee is safe for most healthy adults."),
    ("Sugar is healthy.", "Excessive sugar is harmful and causes health problems."),
    ("The moon is made of cheese.", "The moon is made of rock and minerals."),
    ("Birds are not real.", "Birds are real animals that exist in nature."),
    ("Climate change is fake.", "Scientific consensus confirms climate change is real."),
]

NEI_TEMPLATES = [
    # Strong uncertainty words
    ("AI will replace all jobs.", "The impact of AI on employment is debated and uncertain. Some experts disagree."),
    ("Coffee is good for health.", "Studies show mixed and inconclusive results on coffee's health effects."),
    ("Video games affect behavior.", "Research on video games and behavior is inconclusive and debated."),
    ("Social media impacts mental health.", "The relationship between social media and mental health is complex and unclear."),
    ("Electric cars are better.", "Whether electric cars are better depends on various factors and remains debated."),
    ("Remote work is more productive.", "Remote work productivity varies by individual and context. Results are mixed."),
    ("Organic food is healthier.", "Evidence on organic food benefits is mixed and inconclusive."),
    ("Cold weather causes colds.", "The relationship between cold weather and illness is unclear. Research varies."),
    ("Plants feel pain.", "Whether plants experience pain is scientifically debated and uncertain."),
    ("Aliens have visited Earth.", "No confirmed evidence exists, but the question remains open and uncertain."),
    # Additional stronger NEI cases
    ("Red wine is good for you.", "Some studies suggest benefits but results are mixed and inconclusive."),
    ("Meditation improves focus.", "Research on meditation's effects is complex and varies by individual."),
    ("Breakfast is the most important meal.", "This claim is debated and evidence is mixed among researchers."),
    ("Screen time harms children.", "The impact may vary and research remains inconclusive on this topic."),
    ("Vitamin supplements help health.", "Evidence on vitamin supplementation is mixed and unclear for most people."),
]


def run_benchmark(n_samples: int = 10000, output_file: str = None):
    print(f"=" * 60)
    print(f"ULTIMATE AUDITOR BENCHMARK - {n_samples} samples")
    print(f"=" * 60)
    
    auditor = UltimateAuditor(device="cpu")
    
    n_per_category = n_samples // 3
    n_supports = n_per_category
    n_refutes = n_per_category
    n_nei = n_samples - n_supports - n_refutes
    
    test_cases = []
    for _ in range(n_supports):
        claim, evidence = random.choice(SUPPORTS_TEMPLATES)
        test_cases.append((claim, evidence, "SUPPORTS"))
    for _ in range(n_refutes):
        claim, evidence = random.choice(REFUTES_TEMPLATES)
        test_cases.append((claim, evidence, "REFUTES"))
    for _ in range(n_nei):
        claim, evidence = random.choice(NEI_TEMPLATES)
        test_cases.append((claim, evidence, "NEI"))
    
    random.shuffle(test_cases)
    
    results = {"SUPPORTS": {"correct": 0, "total": 0}, "REFUTES": {"correct": 0, "total": 0}, "NEI": {"correct": 0, "total": 0}}
    start_time = time.time()
    errors = 0
    
    for i, (claim, evidence, expected) in enumerate(test_cases):
        try:
            result = auditor.audit(claim, evidence)
            results[expected]["total"] += 1
            if result.verdict == expected:
                results[expected]["correct"] += 1
        except Exception as e:
            errors += 1
        
        if (i + 1) % 100 == 0:
            total_correct = sum(r["correct"] for r in results.values())
            total_tested = sum(r["total"] for r in results.values())
            acc = 100 * total_correct / max(total_tested, 1)
            print(f"  Progress: {i+1}/{n_samples} ({acc:.1f}% accuracy)")
    
    total_time = time.time() - start_time
    total_correct = sum(r["correct"] for r in results.values())
    total_tested = sum(r["total"] for r in results.values())
    overall_acc = 100 * total_correct / total_tested
    
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples:     {total_tested}")
    print(f"Correct:           {total_correct}")
    print(f"Overall Accuracy:  {overall_acc:.2f}%")
    print(f"Total Time:        {total_time:.1f}s")
    print(f"Errors:            {errors}")
    
    print(f"\n{'-' * 60}")
    print(f"Accuracy by Category:")
    print(f"{'-' * 60}")
    for cat in ["SUPPORTS", "REFUTES", "NEI"]:
        r = results[cat]
        acc = 100 * r["correct"] / max(r["total"], 1)
        print(f"  {cat:10} {r['correct']:5}/{r['total']:5} = {acc:.2f}%")
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Ultimate Auditor Benchmark\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| **Overall Accuracy** | **{overall_acc:.2f}%** |\n")
            f.write(f"| Total Samples | {total_tested} |\n\n")
            f.write(f"## By Category\n\n")
            f.write(f"| Category | Accuracy |\n|----------|----------|\n")
            for cat in ["SUPPORTS", "REFUTES", "NEI"]:
                r = results[cat]
                acc = 100 * r["correct"] / max(r["total"], 1)
                f.write(f"| {cat} | {acc:.2f}% |\n")
        print(f"\nReport saved: {output_file}")
    
    return overall_acc, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--output", type=str, default="reports/benchmark_ultimate.md")
    args = parser.parse_args()
    
    run_benchmark(args.n, args.output)
