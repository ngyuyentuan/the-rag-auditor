"""
Benchmark V3 - Test final optimized RAG Auditor
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

from src.auditor.rag_auditor_v3 import RAGAuditorV3


def generate_benchmark_data(n=100, seed=42) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)
    
    categories = {
        "supports": [
            ("The sun rises in the east.", "The sun rises in the east and sets in the west.", "SUPPORTS"),
            ("Water freezes at 0 degrees.", "At standard pressure, water freezes at 0 degrees Celsius.", "SUPPORTS"),
            ("Tokyo is the capital of Japan.", "Tokyo has been the capital of Japan since 1868.", "SUPPORTS"),
            ("Python is a programming language.", "Python is a high-level programming language.", "SUPPORTS"),
            ("The sky is blue.", "Rayleigh scattering makes the sky appear blue.", "SUPPORTS"),
            ("The moon orbits Earth.", "The Moon orbits Earth at 384,400 km distance.", "SUPPORTS"),
            ("Einstein developed relativity.", "Einstein published the theory of relativity in 1905.", "SUPPORTS"),
            ("DNA is a double helix.", "Watson and Crick described DNA's double helix structure.", "SUPPORTS"),
        ],
        "refutes": [
            ("The earth is flat.", "Scientific evidence confirms Earth is a sphere.", "REFUTES"),
            ("Humans use only 10% of brain.", "Brain imaging shows all areas are active.", "REFUTES"),
            ("Vaccines cause autism.", "Research found no link between vaccines and autism.", "REFUTES"),
            ("Lightning never strikes twice.", "Lightning often strikes the same place.", "REFUTES"),
            ("Great Wall visible from space.", "Astronauts report it is not visible without aid.", "REFUTES"),
            ("Goldfish have 3-second memory.", "Studies show goldfish remember for months.", "REFUTES"),
            ("Sugar causes hyperactivity.", "Double-blind studies show no behavioral difference.", "REFUTES"),
            ("Dinosaurs and humans coexisted.", "Dinosaurs went extinct 66 million years ago.", "REFUTES"),
        ],
        "nei": [
            ("Chocolate is healthy.", "Cocoa has antioxidants but also sugar and fat.", "NEI"),
            ("AI will replace all jobs.", "AI automates tasks but also creates new jobs.", "NEI"),
            ("Coffee is good for you.", "Studies show mixed effects depending on consumption.", "NEI"),
            ("Organic food is more nutritious.", "Evidence is mixed on nutritional differences.", "NEI"),
            ("Social media affects mental health.", "Effects depend on usage patterns and individuals.", "NEI"),
            ("Video games cause violence.", "Research shows mixed and inconclusive results.", "NEI"),
            ("Working from home is better.", "Productivity varies by individual and task type.", "NEI"),
            ("Electric cars are eco-friendly.", "Depends on electricity source and battery production.", "NEI"),
        ],
    }
    
    test_data = []
    for cat, cases in categories.items():
        test_data.extend(cases)
    
    while len(test_data) < n:
        cat = rng.choice(list(categories.keys()))
        test_data.append(rng.choice(categories[cat]))
    
    rng.shuffle(test_data)
    return test_data[:n]


def run_benchmark(auditor, test_data, use_batch=True):
    results = {'total': len(test_data), 'correct': 0,
               'supports_correct': 0, 'refutes_correct': 0, 'nei_correct': 0,
               'supports_total': 0, 'refutes_total': 0, 'nei_total': 0, 'errors': []}
    
    start = time.time()
    
    if use_batch:
        pairs = [(claim, evidence) for claim, evidence, _ in test_data]
        all_results = auditor.audit_batch(pairs, batch_size=8)
        
        for i, (claim, evidence, expected) in enumerate(test_data):
            predicted = all_results[i].verdict
            if expected == "SUPPORTS":
                results['supports_total'] += 1
                if predicted == "SUPPORTS":
                    results['supports_correct'] += 1
                    results['correct'] += 1
                else:
                    results['errors'].append((claim[:35], expected, predicted))
            elif expected == "REFUTES":
                results['refutes_total'] += 1
                if predicted == "REFUTES":
                    results['refutes_correct'] += 1
                    results['correct'] += 1
                else:
                    results['errors'].append((claim[:35], expected, predicted))
            else:
                results['nei_total'] += 1
                if predicted == "NEI":
                    results['nei_correct'] += 1
                    results['correct'] += 1
                else:
                    results['errors'].append((claim[:35], expected, predicted))
    else:
        for i, (claim, evidence, expected) in enumerate(test_data):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            result = auditor.audit(claim, evidence)
            predicted = result.verdict
            
            if expected == "SUPPORTS":
                results['supports_total'] += 1
                if predicted == "SUPPORTS":
                    results['supports_correct'] += 1
                    results['correct'] += 1
            elif expected == "REFUTES":
                results['refutes_total'] += 1
                if predicted == "REFUTES":
                    results['refutes_correct'] += 1
                    results['correct'] += 1
            else:
                results['nei_total'] += 1
                if predicted == "NEI":
                    results['nei_correct'] += 1
                    results['correct'] += 1
    
    elapsed = time.time() - start
    
    acc = results['correct'] / results['total']
    supp = results['supports_correct'] / max(results['supports_total'], 1)
    ref = results['refutes_correct'] / max(results['refutes_total'], 1)
    nei = results['nei_correct'] / max(results['nei_total'], 1)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK V3 RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['total']}, Correct: {results['correct']}")
    print(f"Accuracy: {acc:.2%}")
    print(f"\nBy Category:")
    print(f"  SUPPORTS: {results['supports_correct']}/{results['supports_total']} ({supp:.2%})")
    print(f"  REFUTES:  {results['refutes_correct']}/{results['refutes_total']} ({ref:.2%})")
    print(f"  NEI:      {results['nei_correct']}/{results['nei_total']} ({nei:.2%})")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/results['total']*1000:.0f}ms per sample)")
    
    if results['errors'][:5]:
        print("\nSample Errors:")
        for claim, exp, pred in results['errors'][:5]:
            print(f"  [{exp}->{pred}] {claim}...")
    
    return {'accuracy': acc, 'supports': supp, 'refutes': ref, 'nei': nei, 'time': elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-batch", action="store_true")
    args = ap.parse_args()
    
    print(f"{'='*60}")
    print(f"RAG AUDITOR V3 BENCHMARK - {args.n} test cases")
    print(f"{'='*60}")
    
    test_data = generate_benchmark_data(args.n, args.seed)
    print(f"Distribution: SUPPORTS={sum(1 for _,_,v in test_data if v=='SUPPORTS')}, "
          f"REFUTES={sum(1 for _,_,v in test_data if v=='REFUTES')}, "
          f"NEI={sum(1 for _,_,v in test_data if v=='NEI')}")
    
    auditor = RAGAuditorV3(device=args.device)
    
    print("\nRunning benchmark...")
    results = run_benchmark(auditor, test_data, use_batch=not args.no_batch)
    
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark_v3_results.md").write_text(f"""# RAG Auditor V3 Benchmark

**Samples:** {args.n}
**Overall Accuracy:** {results['accuracy']:.2%}

| Category | Accuracy |
|----------|----------|
| SUPPORTS | {results['supports']:.2%} |
| REFUTES | {results['refutes']:.2%} |
| NEI | {results['nei']:.2%} |

**Time:** {results['time']:.1f}s ({results['time']/args.n*1000:.0f}ms/sample)
""", encoding="utf-8")
    print(f"\nSaved: reports/benchmark_v3_results.md")


if __name__ == "__main__":
    main()
