"""
Optimized grid search with progress bars
"""
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def progress_bar(current, total, prefix="", length=40):
    percent = current / total
    filled = int(length * percent)
    bar = "#" * filled + "-" * (length - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({100*percent:.1f}%)", end="", flush=True)


def load_samples_with_progress(n: int = 150) -> List[Dict]:
    import json
    
    fever_file = ROOT / "data" / "fever_dev.jsonl"
    samples = []
    label_counts = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    max_per_class = n // 3
    
    print("Loading FEVER samples...")
    with open(fever_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if all(c >= max_per_class for c in label_counts.values()):
            break
        
        progress_bar(len(samples), n, "Loading")
        
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
            
            evidence_text = " ".join(evidence_titles) if evidence_titles else item["claim"]
            
            samples.append({
                "claim": item["claim"],
                "evidence": evidence_text,
                "label": label,
            })
            label_counts[label] += 1
        except:
            continue
    
    print(f"\nLoaded {len(samples)} samples: {label_counts}")
    return samples


def test_thresholds(auditor, samples, strong_e, strong_c, adj_e, adj_c, neutral) -> Dict:
    from src.auditor.constants import Verdict, PatternWeights
    
    correct = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    total = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    
    for sample in samples:
        probs = auditor._ensemble_inference(sample["claim"], sample["evidence"])
        
        p_e = probs.get('entailment', 0)
        p_c = probs.get('contradiction', 0)
        p_n = probs.get('neutral', 0)
        
        is_nei, _ = auditor._detect_nei_semantic(sample["claim"], sample["evidence"])
        if is_nei:
            verdict = "NEI"
        else:
            refutes_boost = auditor._calculate_refutes_boost(sample["claim"])
            supports_boost = auditor._calculate_supports_boost(sample["claim"], sample["evidence"])
            p_c_adj = min(p_c + refutes_boost, 1.0)
            p_e_adj = min(p_e + supports_boost, 1.0)
            
            if p_c > strong_c:
                verdict = "REFUTES"
            elif p_e > strong_e:
                verdict = "SUPPORTS"
            elif refutes_boost >= 0.18 and p_c > 0.15:
                verdict = "REFUTES"
            elif p_c_adj > adj_c and p_c_adj > p_e_adj:
                verdict = "REFUTES"
            elif p_e_adj > adj_e and p_e_adj > p_c_adj:
                verdict = "SUPPORTS"
            elif p_n > neutral:
                verdict = "NEI"
            else:
                scores = [(p_e_adj, "SUPPORTS"), (p_c_adj, "REFUTES"), (p_n, "NEI")]
                verdict = max(scores, key=lambda x: x[0])[1]
        
        total[sample["label"]] += 1
        if verdict == sample["label"]:
            correct[sample["label"]] += 1
    
    total_n = sum(total.values())
    total_c = sum(correct.values())
    
    return {
        "overall": total_c / total_n if total_n > 0 else 0,
        "supports": correct["SUPPORTS"] / total["SUPPORTS"] if total["SUPPORTS"] > 0 else 0,
        "refutes": correct["REFUTES"] / total["REFUTES"] if total["REFUTES"] > 0 else 0,
        "nei": correct["NEI"] / total["NEI"] if total["NEI"] > 0 else 0,
    }


def grid_search():
    print("=" * 60)
    print("GRID SEARCH FOR OPTIMAL THRESHOLDS")
    print("=" * 60)
    
    samples = load_samples_with_progress(120)
    
    print("\nLoading auditor...")
    from src.auditor.ensemble_v2 import EnsembleAuditorV2
    auditor = EnsembleAuditorV2(device="cpu", nei_semantic_check=True, use_single_model=True)
    
    print("\nWarming up model...")
    _ = auditor._ensemble_inference("test", "test")
    print("Model ready!")
    
    strong_e_range = [0.45, 0.48, 0.50]
    strong_c_range = [0.48, 0.50, 0.52]
    adj_e_range = [0.32, 0.35, 0.38]
    adj_c_range = [0.35, 0.38, 0.40]
    neutral_range = [0.40, 0.42]
    
    combos = list(product(strong_e_range, strong_c_range, adj_e_range, adj_c_range, neutral_range))
    total_combos = len(combos)
    
    print(f"\nTesting {total_combos} combinations...")
    
    best_result = None
    best_accuracy = 0
    best_params = None
    
    start_time = time.time()
    
    for i, (strong_e, strong_c, adj_e, adj_c, neutral) in enumerate(combos):
        progress_bar(i + 1, total_combos, "Grid Search")
        
        result = test_thresholds(auditor, samples, strong_e, strong_c, adj_e, adj_c, neutral)
        
        if result["overall"] > best_accuracy:
            best_accuracy = result["overall"]
            best_result = result
            best_params = {
                "strong_e": strong_e, "strong_c": strong_c,
                "adj_e": adj_e, "adj_c": adj_c, "neutral": neutral
            }
    
    elapsed = time.time() - start_time
    
    print(f"\n\n{'='*60}")
    print("BEST RESULT")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s")
    print(f"Accuracy: {100*best_result['overall']:.2f}%")
    print(f"  SUPPORTS: {100*best_result['supports']:.2f}%")
    print(f"  REFUTES: {100*best_result['refutes']:.2f}%")
    print(f"  NEI: {100*best_result['nei']:.2f}%")
    print(f"\nOptimal params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params


if __name__ == "__main__":
    grid_search()
