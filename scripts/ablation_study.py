"""
Ablation Study for RAG Auditor
Tests contribution of each component
"""
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class AblationResult:
    name: str
    accuracy: float
    supports_acc: float
    refutes_acc: float
    nei_acc: float


SUPPORTS_TEMPLATES = [
    ("Vaccines are safe.", "Research confirms vaccines are safe and effective."),
    ("Exercise improves health.", "Studies show exercise improves health."),
    ("Water is essential.", "Scientific evidence proves water is essential."),
]

REFUTES_TEMPLATES = [
    ("Vaccines are dangerous.", "Research confirms vaccines are safe and effective."),
    ("Earth is flat.", "Scientific evidence proves Earth is spherical."),
    ("Vaccines cause autism.", "No evidence links vaccines to autism."),
]

NEI_TEMPLATES = [
    ("AI will replace jobs.", "The impact is debated and uncertain. Experts disagree."),
    ("Coffee is good for health.", "Studies show mixed and inconclusive results."),
    ("Remote work is productive.", "Productivity varies by individual. Results are mixed."),
]


def generate_test_data(n_per_category: int = 200) -> List[Tuple[str, str, str]]:
    data = []
    for _ in range(n_per_category):
        data.append((*random.choice(SUPPORTS_TEMPLATES), "SUPPORTS"))
        data.append((*random.choice(REFUTES_TEMPLATES), "REFUTES"))
        data.append((*random.choice(NEI_TEMPLATES), "NEI"))
    random.shuffle(data)
    return data


def run_ablation(n_samples: int = 600, output_file: str = None):
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    test_data = generate_test_data(n_samples // 3)
    results = []
    
    configs = [
        ("Full Model", {"use_nli": True, "use_patterns": True, "use_similarity": True, "use_antonyms": True}),
        ("No Semantic Similarity", {"use_nli": True, "use_patterns": True, "use_similarity": False, "use_antonyms": True}),
        ("No Pattern Matching", {"use_nli": True, "use_patterns": False, "use_similarity": True, "use_antonyms": True}),
        ("No Antonym Detection", {"use_nli": True, "use_patterns": True, "use_similarity": True, "use_antonyms": False}),
        ("NLI Only (Baseline)", {"use_nli": True, "use_patterns": False, "use_similarity": False, "use_antonyms": False}),
    ]
    
    from src.auditor.ultimate_auditor import UltimateAuditor
    
    for config_name, config in configs:
        print(f"\nTesting: {config_name}")
        auditor = UltimateAuditor(device="cpu")
        
        category_results = {"SUPPORTS": {"correct": 0, "total": 0}, "REFUTES": {"correct": 0, "total": 0}, "NEI": {"correct": 0, "total": 0}}
        
        for claim, evidence, expected in test_data:
            try:
                if not config.get("use_patterns", True):
                    auditor.SUPPORT_WORDS = []
                    auditor.NEGATION_WORDS = []
                    auditor.CONTRADICTION_WORDS = []
                    auditor.NEI_WORDS = []
                
                if not config.get("use_antonyms", True):
                    auditor.ANTONYM_PAIRS = []
                
                result = auditor.audit(claim, evidence)
                category_results[expected]["total"] += 1
                if result.verdict == expected:
                    category_results[expected]["correct"] += 1
            except:
                pass
        
        total_correct = sum(r["correct"] for r in category_results.values())
        total_tested = sum(r["total"] for r in category_results.values())
        overall_acc = 100 * total_correct / max(total_tested, 1)
        
        sup_acc = 100 * category_results["SUPPORTS"]["correct"] / max(category_results["SUPPORTS"]["total"], 1)
        ref_acc = 100 * category_results["REFUTES"]["correct"] / max(category_results["REFUTES"]["total"], 1)
        nei_acc = 100 * category_results["NEI"]["correct"] / max(category_results["NEI"]["total"], 1)
        
        results.append(AblationResult(config_name, overall_acc, sup_acc, ref_acc, nei_acc))
        print(f"  Overall: {overall_acc:.2f}%, SUPPORTS: {sup_acc:.2f}%, REFUTES: {ref_acc:.2f}%, NEI: {nei_acc:.2f}%")
    
    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Config':<30} {'Overall':>10} {'Delta':>10}")
    print("-" * 50)
    baseline = results[-1].accuracy
    for r in results:
        delta = r.accuracy - baseline
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        print(f"{r.name:<30} {r.accuracy:>9.2f}% {delta_str:>10}")
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Ablation Study\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## Results\n\n")
            f.write("| Configuration | Overall | SUPPORTS | REFUTES | NEI | Delta |\n")
            f.write("|--------------|---------|----------|---------|-----|-------|\n")
            for r in results:
                delta = r.accuracy - baseline
                delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
                f.write(f"| {r.name} | {r.accuracy:.2f}% | {r.supports_acc:.2f}% | {r.refutes_acc:.2f}% | {r.nei_acc:.2f}% | {delta_str} |\n")
            f.write("\n## Key Findings\n\n")
            most_important = max(results[:-1], key=lambda x: abs(x.accuracy - baseline))
            f.write(f"- Most important component: **{most_important.name.replace('No ', '')}** (removing it causes {abs(most_important.accuracy - baseline):.2f}% drop)\n")
        print(f"\nSaved: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--output", type=str, default="reports/ablation_study.md")
    args = parser.parse_args()
    
    run_ablation(args.n, args.output)
