"""
Benchmark V2 - Test improved RAG Auditor

Tests RAG Auditor V2 with ensemble approach.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.rag_auditor_v2 import RAGAuditorV2


def generate_benchmark_data(n=100, seed=42) -> List[Tuple[str, str, str]]:
    """Generate diverse test cases."""
    rng = random.Random(seed)
    
    categories = {
        "easy_supports": [
            ("The sun rises in the east.", "The sun rises in the east and sets in the west.", "SUPPORTS"),
            ("Water freezes at 0 degrees Celsius.", "At standard pressure, water freezes at 0 degrees Celsius.", "SUPPORTS"),
            ("Tokyo is the capital of Japan.", "Tokyo has been the capital of Japan since 1868.", "SUPPORTS"),
            ("Einstein developed relativity.", "Albert Einstein published his theory of relativity in 1905.", "SUPPORTS"),
            ("The moon orbits Earth.", "The Moon orbits Earth at an average distance of 384,400 km.", "SUPPORTS"),
            ("Python is a programming language.", "Python is a high-level programming language.", "SUPPORTS"),
            ("The sky is blue.", "Rayleigh scattering makes the sky appear blue.", "SUPPORTS"),
            ("2 + 2 equals 4.", "Basic arithmetic confirms two plus two equals four.", "SUPPORTS"),
        ],
        "easy_refutes": [
            ("The earth is flat.", "Scientific evidence confirms Earth is a sphere.", "REFUTES"),
            ("Humans only use 10% of their brain.", "Brain imaging shows all areas are active.", "REFUTES"),
            ("Lightning never strikes twice.", "Lightning often strikes the same place multiple times.", "REFUTES"),
            ("Goldfish have 3-second memory.", "Studies show goldfish remember for months.", "REFUTES"),
            ("The Great Wall is visible from space.", "Astronauts report it is not visible without aid.", "REFUTES"),
            ("Vaccines cause autism.", "Research has found no link between vaccines and autism.", "REFUTES"),
            ("Sugar causes hyperactivity.", "Double-blind studies show no behavioral difference.", "REFUTES"),
            ("Dinosaurs and humans coexisted.", "Dinosaurs went extinct 66 million years ago.", "REFUTES"),
        ],
        "nei": [
            ("Chocolate is healthy.", "Cocoa has antioxidants but also sugar and fat.", "NEI"),
            ("AI will replace all jobs.", "AI automates tasks but also creates new jobs.", "NEI"),
            ("Coffee is good for you.", "Moderate consumption may have benefits and downsides.", "NEI"),
            ("Organic food is more nutritious.", "Evidence is mixed on nutritional differences.", "NEI"),
        ],
    }
    
    test_data = []
    for category, cases in categories.items():
        test_data.extend(cases)
    
    while len(test_data) < n:
        category = rng.choice(list(categories.keys()))
        case = rng.choice(categories[category])
        test_data.append(case)
    
    rng.shuffle(test_data)
    return test_data[:n]


def run_benchmark(auditor, test_data):
    """Run benchmark."""
    results = {'total': len(test_data), 'correct': 0, 
               'supports_correct': 0, 'refutes_correct': 0, 'nei_correct': 0,
               'supports_total': 0, 'refutes_total': 0, 'nei_total': 0, 'errors': []}
    
    start = time.time()
    for i, (claim, evidence, expected) in enumerate(test_data):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_data)}")
        
        result = auditor.audit(claim, evidence)
        predicted = result.verdict
        if predicted == "UNCERTAIN":
            predicted = "NEI"
        
        if expected == "SUPPORTS":
            results['supports_total'] += 1
            if predicted == "SUPPORTS":
                results['supports_correct'] += 1
                results['correct'] += 1
            else:
                results['errors'].append((claim[:40], expected, predicted))
        elif expected == "REFUTES":
            results['refutes_total'] += 1
            if predicted == "REFUTES":
                results['refutes_correct'] += 1
                results['correct'] += 1
            else:
                results['errors'].append((claim[:40], expected, predicted))
        else:
            results['nei_total'] += 1
            if predicted in ("NEI", "UNCERTAIN"):
                results['nei_correct'] += 1
                results['correct'] += 1
            else:
                results['errors'].append((claim[:40], expected, predicted))
    
    elapsed = time.time() - start
    
    acc = results['correct'] / results['total']
    supp_acc = results['supports_correct'] / max(results['supports_total'], 1)
    ref_acc = results['refutes_correct'] / max(results['refutes_total'], 1)
    nei_acc = results['nei_correct'] / max(results['nei_total'], 1)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK V2 RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['total']}, Correct: {results['correct']}")
    print(f"Accuracy: {acc:.2%}")
    print(f"\nBy Category:")
    print(f"  SUPPORTS: {results['supports_correct']}/{results['supports_total']} ({supp_acc:.2%})")
    print(f"  REFUTES:  {results['refutes_correct']}/{results['refutes_total']} ({ref_acc:.2%})")
    print(f"  NEI:      {results['nei_correct']}/{results['nei_total']} ({nei_acc:.2%})")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/results['total']*1000:.0f}ms per sample)")
    
    if results['errors'][:5]:
        print("\nSample Errors:")
        for claim, exp, pred in results['errors'][:5]:
            print(f"  [{exp}->{pred}] {claim}...")
    
    return {'accuracy': acc, 'supports': supp_acc, 'refutes': ref_acc, 'nei': nei_acc, 'time': elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-ensemble", action="store_true")
    args = ap.parse_args()
    
    print(f"{'='*60}")
    print(f"RAG AUDITOR V2 BENCHMARK - {args.n} test cases")
    print(f"{'='*60}")
    
    print("\nGenerating test data...")
    test_data = generate_benchmark_data(args.n, args.seed)
    
    print(f"Distribution: SUPPORTS={sum(1 for _,_,v in test_data if v=='SUPPORTS')}, "
          f"REFUTES={sum(1 for _,_,v in test_data if v=='REFUTES')}, "
          f"NEI={sum(1 for _,_,v in test_data if v=='NEI')}")
    
    print("\nInitializing V2 auditor...")
    auditor = RAGAuditorV2(
        device=args.device,
        use_ensemble=not args.no_ensemble,
    )
    
    print("\nRunning benchmark...")
    results = run_benchmark(auditor, test_data)
    
    # Save report
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark_v2_results.md").write_text(f"""# RAG Auditor V2 Benchmark

**Samples:** {args.n}
**Overall Accuracy:** {results['accuracy']:.2%}

| Category | Accuracy |
|----------|----------|
| SUPPORTS | {results['supports']:.2%} |
| REFUTES | {results['refutes']:.2%} |
| NEI | {results['nei']:.2%} |

**Time:** {results['time']:.1f}s
""", encoding="utf-8")
    print("\nSaved: reports/benchmark_v2_results.md")


if __name__ == "__main__":
    main()
