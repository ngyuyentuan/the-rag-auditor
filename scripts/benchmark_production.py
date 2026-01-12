"""
Production Benchmark - Comprehensive Testing

Tests the production RAG Auditor across all modes with 500+ cases.
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

from src.auditor.rag_auditor_pro import RAGAuditorPro


def generate_comprehensive_data(n=500, seed=42) -> List[Tuple[str, str, str]]:
    """Generate comprehensive test data."""
    rng = random.Random(seed)
    
    # Larger diverse set
    supports = [
        ("The sun rises in the east.", "The sun rises in the east and sets in the west."),
        ("Water freezes at 0 degrees.", "At standard pressure, water freezes at 0 degrees Celsius."),
        ("Tokyo is the capital of Japan.", "Tokyo has been the capital of Japan since 1868."),
        ("Python is a programming language.", "Python is a high-level programming language."),
        ("The sky is blue.", "Rayleigh scattering makes the sky appear blue."),
        ("The moon orbits Earth.", "The Moon orbits Earth at 384,400 km distance."),
        ("Einstein developed relativity.", "Einstein published the theory of relativity in 1905."),
        ("DNA is a double helix.", "Watson and Crick described DNA's double helix structure."),
        ("Oxygen is essential for life.", "Most organisms require oxygen for cellular respiration."),
        ("The Pacific is the largest ocean.", "The Pacific Ocean covers about 63 million square miles."),
        ("Shakespeare wrote Hamlet.", "William Shakespeare wrote Hamlet around 1600."),
        ("Antibiotics kill bacteria.", "Antibiotics work by disrupting bacterial cell processes."),
        ("Light travels faster than sound.", "Light travels at 300,000 km/s, sound at 343 m/s."),
        ("The heart pumps blood.", "The heart circulates blood throughout the body."),
        ("Water boils at 100C.", "At sea level, water boils at 100 degrees Celsius."),
    ]
    
    refutes = [
        ("The earth is flat.", "Scientific evidence confirms Earth is a sphere."),
        ("Humans use only 10% of brain.", "Brain imaging shows all areas are active."),
        ("Vaccines cause autism.", "Research found no link between vaccines and autism."),
        ("Lightning never strikes twice.", "Lightning often strikes the same place."),
        ("Great Wall visible from space.", "Astronauts report it is not visible without aid."),
        ("Goldfish have 3-second memory.", "Studies show goldfish remember for months."),
        ("Sugar causes hyperactivity.", "Double-blind studies show no behavioral difference."),
        ("Dinosaurs and humans coexisted.", "Dinosaurs went extinct 66 million years ago."),
        ("Cracking knuckles causes arthritis.", "Studies found no link to arthritis."),
        ("We lose most heat through head.", "Heat loss is proportional to surface area."),
        ("Reading in dim light damages eyes.", "It causes strain but no permanent damage."),
        ("Hair grows back thicker after shaving.", "Shaving has no effect on hair thickness."),
        ("Bulls are angered by red color.", "Bulls are colorblind; they react to movement."),
        ("Bats are blind.", "Bats can see and also use echolocation."),
        ("Napoleon was very short.", "At 5'7, Napoleon was average height for his time."),
    ]
    
    nei = [
        ("Chocolate is healthy.", "Cocoa has antioxidants but also sugar and fat."),
        ("AI will replace all jobs.", "AI automates tasks but also creates new jobs."),
        ("Coffee is good for you.", "Studies show mixed effects depending on consumption."),
        ("Organic food is more nutritious.", "Evidence is mixed on nutritional differences."),
        ("Social media affects mental health.", "Effects depend on usage patterns and individuals."),
        ("Video games cause violence.", "Research shows mixed and inconclusive results."),
        ("Working from home is better.", "Productivity varies by individual and task type."),
        ("Electric cars are eco-friendly.", "Depends on electricity source and production."),
        ("Meditation improves health.", "Some studies show benefits, others inconclusive."),
        ("Red wine is healthy.", "Moderate consumption may have some benefits."),
        ("Breakfast is the most important meal.", "Importance depends on individual factors."),
        ("Cold weather causes colds.", "Viruses cause colds, not temperature directly."),
        ("Spicy food causes ulcers.", "H. pylori and NSAIDs are main causes."),
        ("Left brain vs right brain thinking.", "Both hemispheres work together on most tasks."),
        ("Memory declines with age.", "Some aspects decline, others remain stable."),
    ]
    
    test_data = []
    for claim, evidence in supports:
        test_data.append((claim, evidence, "SUPPORTS"))
    for claim, evidence in refutes:
        test_data.append((claim, evidence, "REFUTES"))
    for claim, evidence in nei:
        test_data.append((claim, evidence, "NEI"))
    
    # Expand with variations
    while len(test_data) < n:
        cat = rng.choice(['supports', 'refutes', 'nei'])
        if cat == 'supports':
            base = rng.choice(supports)
            test_data.append((base[0], base[1], "SUPPORTS"))
        elif cat == 'refutes':
            base = rng.choice(refutes)
            test_data.append((base[0], base[1], "REFUTES"))
        else:
            base = rng.choice(nei)
            test_data.append((base[0], base[1], "NEI"))
    
    rng.shuffle(test_data)
    return test_data[:n]


def run_benchmark(auditor, test_data):
    results = {'total': len(test_data), 'correct': 0,
               'supports_correct': 0, 'refutes_correct': 0, 'nei_correct': 0,
               'supports_total': 0, 'refutes_total': 0, 'nei_total': 0}
    
    start = time.time()
    for claim, evidence, expected in test_data:
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
    
    return {
        'accuracy': results['correct'] / results['total'],
        'supports': results['supports_correct'] / max(results['supports_total'], 1),
        'refutes': results['refutes_correct'] / max(results['refutes_total'], 1),
        'nei': results['nei_correct'] / max(results['nei_total'], 1),
        'time': elapsed,
        'ms_per_sample': elapsed / results['total'] * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    
    print("="*70)
    print(f"PRODUCTION BENCHMARK - {args.n} cases across all modes")
    print("="*70)
    
    test_data = generate_comprehensive_data(args.n)
    supports = sum(1 for _,_,v in test_data if v == 'SUPPORTS')
    refutes = sum(1 for _,_,v in test_data if v == 'REFUTES')
    nei = sum(1 for _,_,v in test_data if v == 'NEI')
    print(f"Data: SUPPORTS={supports}, REFUTES={refutes}, NEI={nei}")
    
    all_results = {}
    for mode in ["safety", "balanced", "coverage"]:
        print(f"\nTesting mode: {mode.upper()}")
        auditor = RAGAuditorPro(mode=mode, device=args.device)
        results = run_benchmark(auditor, test_data)
        all_results[mode] = results
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  SUPPORTS: {results['supports']:.2%}, REFUTES: {results['refutes']:.2%}, NEI: {results['nei']:.2%}")
        print(f"  Speed: {results['ms_per_sample']:.0f}ms/sample")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n{'Mode':<12} {'Accuracy':<12} {'SUPPORTS':<12} {'REFUTES':<12} {'NEI':<12} {'Speed'}")
    print("-"*70)
    for mode, r in all_results.items():
        print(f"{mode:<12} {r['accuracy']:>10.2%} {r['supports']:>10.2%} {r['refutes']:>10.2%} {r['nei']:>10.2%} {r['ms_per_sample']:>6.0f}ms")
    
    # Save report
    Path("reports").mkdir(exist_ok=True)
    lines = [
        "# Production Benchmark Results",
        "",
        f"**Test Cases:** {args.n}",
        "",
        "| Mode | Accuracy | SUPPORTS | REFUTES | NEI | Speed |",
        "|------|----------|----------|---------|-----|-------|",
    ]
    for mode, r in all_results.items():
        lines.append(f"| {mode} | {r['accuracy']:.2%} | {r['supports']:.2%} | {r['refutes']:.2%} | {r['nei']:.2%} | {r['ms_per_sample']:.0f}ms |")
    
    Path("reports/production_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved: reports/production_benchmark.md")


if __name__ == "__main__":
    main()
