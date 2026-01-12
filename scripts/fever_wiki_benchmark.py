"""
FEVER Benchmark with REAL Wikipedia text evidence instead of just titles
"""
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_fever_with_wiki(fever_file: Path, max_samples: int = 500) -> List[Dict]:
    from src.utils.wikipedia_client import WikipediaClient
    
    wiki = WikipediaClient(cache_enabled=True)
    samples = []
    label_counts = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    max_per_class = max_samples // 3
    
    print(f"Loading FEVER and fetching Wikipedia text...")
    
    with open(fever_file, "r", encoding="utf-8") as f:
        for line in f:
            if all(c >= max_per_class for c in label_counts.values()):
                break
                
            try:
                item = json.loads(line.strip())
                label = item.get("label", "")
                if label == "NOT ENOUGH INFO":
                    label = "NEI"
                if label not in label_counts or label_counts[label] >= max_per_class:
                    continue
                
                evidence_titles = set()
                if item.get("evidence"):
                    for ev_group in item["evidence"]:
                        if ev_group:
                            for ev in ev_group:
                                if len(ev) >= 3 and ev[2]:
                                    evidence_titles.add(str(ev[2]))
                
                if not evidence_titles and label != "NEI":
                    continue
                
                evidence_text = ""
                for title in list(evidence_titles)[:2]:
                    article = wiki.get_article(title)
                    if article and article.text:
                        evidence_text += article.text[:500] + " "
                
                if not evidence_text and label != "NEI":
                    continue
                
                samples.append({
                    "claim": item["claim"],
                    "evidence": evidence_text.strip() if evidence_text else item["claim"],
                    "label": label,
                    "titles": list(evidence_titles)
                })
                label_counts[label] += 1
                
                if len(samples) % 50 == 0:
                    print(f"  Loaded {len(samples)} samples: {label_counts}")
                    
            except Exception as e:
                continue
    
    print(f"Loaded {len(samples)} samples with Wikipedia text")
    return samples


def run_benchmark_with_wiki(n_samples: int = 300, output_file: str = None):
    print("=" * 60)
    print("FEVER BENCHMARK WITH REAL WIKIPEDIA TEXT")
    print("=" * 60)
    
    from src.auditor.ultra_auditor import UltraAuditor
    auditor = UltraAuditor(device="cpu", nei_semantic_check=True)
    
    fever_file = ROOT / "data" / "fever_dev.jsonl"
    samples = load_fever_with_wiki(fever_file, n_samples)
    
    correct = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    total = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    errors = 0
    
    print(f"\nEvaluating on {len(samples)} samples...")
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        try:
            result = auditor.audit(sample["claim"], sample["evidence"])
            total[sample["label"]] += 1
            if result.verdict == sample["label"]:
                correct[sample["label"]] += 1
                
            if (i + 1) % 50 == 0:
                curr_total = sum(total.values())
                curr_correct = sum(correct.values())
                print(f"  Progress: {i+1}/{len(samples)} ({100*curr_correct/curr_total:.1f}%)")
                
        except Exception as e:
            errors += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("FEVER WITH WIKIPEDIA TEXT RESULTS")
    print("=" * 60)
    
    total_samples = sum(total.values())
    total_correct = sum(correct.values())
    overall_acc = 100 * total_correct / total_samples if total_samples > 0 else 0
    
    print(f"Total samples:     {total_samples}")
    print(f"Correct:           {total_correct}")
    print(f"Overall Accuracy:  {overall_acc:.2f}%")
    print(f"Time:              {elapsed:.1f}s")
    print(f"Errors:            {errors}")
    print()
    print("-" * 60)
    
    for label in ["SUPPORTS", "REFUTES", "NEI"]:
        if total[label] > 0:
            acc = 100 * correct[label] / total[label]
            print(f"  {label:12} {correct[label]:3}/{total[label]:3} = {acc:.2f}%")
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# FEVER with Wikipedia Text Results\n\n")
            f.write(f"**Overall Accuracy: {overall_acc:.2f}%**\n\n")
            f.write(f"| Category | Correct | Total | Accuracy |\n")
            f.write(f"|----------|---------|-------|----------|\n")
            for label in ["SUPPORTS", "REFUTES", "NEI"]:
                if total[label] > 0:
                    acc = 100 * correct[label] / total[label]
                    f.write(f"| {label} | {correct[label]} | {total[label]} | {acc:.2f}% |\n")
        print(f"\nSaved: {output_file}")
    
    return overall_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--output", type=str, default="reports/fever_wiki.md")
    args = parser.parse_args()
    
    run_benchmark_with_wiki(args.n, args.output)
