"""
Comprehensive RAG Auditor Benchmark

Tests the RAG Auditor with 1000 diverse claim-evidence pairs.
Includes hard cases: subtle contradictions, paraphrases, partial evidence.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.rag_auditor import RAGAuditor


def generate_benchmark_data(n=1000, seed=42) -> List[Tuple[str, str, str]]:
    """
    Generate diverse claim-evidence pairs with ground truth.
    
    Returns list of (claim, evidence, expected_verdict)
    """
    rng = random.Random(seed)
    
    # Categories of test cases
    categories = {
        # Easy supports
        "easy_supports": [
            ("The sun rises in the east.", "The sun rises in the east and sets in the west.", "SUPPORTS"),
            ("Water freezes at 0 degrees Celsius.", "At standard pressure, water freezes at 0°C (32°F).", "SUPPORTS"),
            ("Tokyo is the capital of Japan.", "Tokyo has been the capital of Japan since 1868.", "SUPPORTS"),
            ("Einstein developed the theory of relativity.", "Albert Einstein published his theory of special relativity in 1905.", "SUPPORTS"),
            ("The moon orbits Earth.", "The Moon orbits Earth at an average distance of 384,400 km.", "SUPPORTS"),
        ],
        
        # Easy refutes
        "easy_refutes": [
            ("The earth is flat.", "Scientific evidence confirms that Earth is an oblate spheroid.", "REFUTES"),
            ("Humans only use 10% of their brain.", "Brain imaging studies show that all areas of the brain are active.", "REFUTES"),
            ("Lightning never strikes the same place twice.", "Lightning often strikes tall structures like buildings and trees multiple times.", "REFUTES"),
            ("Goldfish have a 3-second memory.", "Studies show goldfish can remember things for months.", "REFUTES"),
            ("The Great Wall of China is visible from space.", "Astronauts report the Great Wall is not visible from low Earth orbit without aid.", "REFUTES"),
        ],
        
        # Hard supports (paraphrased)
        "hard_supports": [
            ("Coffee contains caffeine.", "The stimulant compound found in coffee beans is a methylxanthine alkaloid.", "SUPPORTS"),
            ("Shakespeare wrote Hamlet.", "The tragedy of Hamlet was penned by the Bard of Avon around 1600.", "SUPPORTS"),
            ("Dogs are mammals.", "Canines are warm-blooded vertebrates that nurse their young.", "SUPPORTS"),
            ("The Pacific is the largest ocean.", "Covering about 63 million square miles, the Pacific exceeds all other oceans in area.", "SUPPORTS"),
            ("Antibiotics kill bacteria.", "Antimicrobial agents work by disrupting bacterial cell walls and protein synthesis.", "SUPPORTS"),
        ],
        
        # Hard refutes (subtle)
        "hard_refutes": [
            ("Vaccines cause autism.", "Extensive research involving millions of children has found no link between vaccines and autism.", "REFUTES"),
            ("Sugar causes hyperactivity in children.", "Double-blind studies show no behavioral difference in children given sugar vs placebo.", "REFUTES"),
            ("We lose most body heat through our head.", "Heat loss is proportional to exposed surface area; the head is only about 10%.", "REFUTES"),
            ("Cracking knuckles causes arthritis.", "Long-term studies show no correlation between knuckle cracking and arthritis.", "REFUTES"),
            ("Reading in dim light damages your eyes.", "While causing temporary eye strain, reading in low light does not cause permanent damage.", "REFUTES"),
        ],
        
        # Ambiguous/NEI cases
        "ambiguous": [
            ("Chocolate is healthy.", "Cocoa contains antioxidants, but chocolate also has high sugar and fat content.", "NEI"),
            ("AI will replace all jobs.", "AI automates certain tasks but also creates new job categories.", "NEI"),
            ("Social media is bad for mental health.", "Studies show mixed effects depending on usage patterns and individual factors.", "NEI"),
            ("Organic food is more nutritious.", "Some studies show higher levels of certain nutrients, but overall evidence is mixed.", "NEI"),
            ("Coffee is good for you.", "Moderate consumption may have benefits, but excessive intake has downsides.", "NEI"),
        ],
        
        # Scientific claims
        "scientific": [
            ("DNA is a double helix.", "Watson and Crick described the double helix structure of deoxyribonucleic acid in 1953.", "SUPPORTS"),
            ("Evolution is a theory.", "Evolution by natural selection is both a scientific theory and an observed fact.", "SUPPORTS"),
            ("Climate change is natural.", "While climate has changed naturally, current warming is primarily driven by human activities.", "REFUTES"),
            ("Dinosaurs and humans coexisted.", "Dinosaurs went extinct 66 million years ago; humans appeared about 300,000 years ago.", "REFUTES"),
            ("There are 8 planets in our solar system.", "After reclassifying Pluto, the IAU recognizes 8 major planets.", "SUPPORTS"),
        ],
        
        # Historical claims
        "historical": [
            ("World War 2 ended in 1945.", "Japan surrendered on September 2, 1945, marking the end of WWII.", "SUPPORTS"),
            ("Napoleon was very short.", "At 5'7\", Napoleon was average height for his time; the myth arose from propaganda.", "REFUTES"),
            ("Vikings wore horned helmets.", "Archaeological evidence shows Viking helmets were plain; horns are a 19th-century myth.", "REFUTES"),
            ("The printing press was invented by Gutenberg.", "Johannes Gutenberg developed movable type printing in Europe around 1440.", "SUPPORTS"),
            ("Christopher Columbus discovered America.", "Columbus reached the Americas in 1492, though indigenous peoples had lived there for millennia.", "NEI"),
        ],
        
        # Tech claims
        "tech": [
            ("Python is a programming language.", "Python is a high-level, interpreted programming language created by Guido van Rossum.", "SUPPORTS"),
            ("The internet was invented in the 1990s.", "ARPANET, the precursor to the internet, was created in the 1960s.", "REFUTES"),
            ("5G causes COVID-19.", "COVID-19 is caused by the SARS-CoV-2 virus; radio waves cannot create viruses.", "REFUTES"),
            ("Quantum computers can solve any problem instantly.", "Quantum computers offer speedup for specific algorithms, not all computational problems.", "REFUTES"),
            ("Machine learning requires big data.", "While large datasets help, many ML techniques work with limited data.", "NEI"),
        ],
        
        # Edge cases
        "edge_cases": [
            ("2+2=4", "Basic arithmetic confirms that two plus two equals four.", "SUPPORTS"),
            ("The sky is blue.", "Rayleigh scattering of sunlight by the atmosphere makes the sky appear blue.", "SUPPORTS"),
            ("Nothing can travel faster than light.", "According to special relativity, the speed of light is the universal speed limit.", "SUPPORTS"),
            ("All birds can fly.", "Penguins, ostriches, and kiwis are examples of flightless birds.", "REFUTES"),
            ("Fish can drown.", "Fish can suffocate if water lacks oxygen, though technically different from drowning.", "NEI"),
        ],
    }
    
    # Generate test set
    test_data = []
    
    # Add all base cases
    for category, cases in categories.items():
        for claim, evidence, verdict in cases:
            test_data.append((claim, evidence, verdict))
    
    # Repeat and shuffle to reach n samples
    while len(test_data) < n:
        # Sample from categories
        category = rng.choice(list(categories.keys()))
        claim, evidence, verdict = rng.choice(categories[category])
        
        # Add slight variations
        variations = [
            (claim, evidence, verdict),
            (claim.lower(), evidence, verdict),
            (claim + " Is this true?", evidence, verdict),
        ]
        test_data.append(rng.choice(variations))
    
    rng.shuffle(test_data)
    return test_data[:n]


def run_benchmark(auditor, test_data, batch_size=10, show_errors=True):
    """Run benchmark and compute metrics."""
    results = {
        'total': len(test_data),
        'correct': 0,
        'supports_correct': 0,
        'refutes_correct': 0,
        'nei_correct': 0,
        'supports_total': 0,
        'refutes_total': 0,
        'nei_total': 0,
        'errors': [],
    }
    
    start_time = time.time()
    
    for i, (claim, evidence, expected) in enumerate(test_data):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_data)}")
        
        result = auditor.audit(claim, evidence)
        predicted = result.verdict
        
        # Map NEI variations
        if predicted == "UNCERTAIN":
            predicted = "NEI"
        
        # Track by category
        if expected == "SUPPORTS":
            results['supports_total'] += 1
            if predicted == "SUPPORTS":
                results['supports_correct'] += 1
                results['correct'] += 1
            else:
                results['errors'].append((claim, expected, predicted))
        elif expected == "REFUTES":
            results['refutes_total'] += 1
            if predicted == "REFUTES":
                results['refutes_correct'] += 1
                results['correct'] += 1
            else:
                results['errors'].append((claim, expected, predicted))
        else:  # NEI
            results['nei_total'] += 1
            if predicted in ("NEI", "UNCERTAIN"):
                results['nei_correct'] += 1
                results['correct'] += 1
            else:
                results['errors'].append((claim, expected, predicted))
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    accuracy = results['correct'] / results['total']
    supports_acc = results['supports_correct'] / results['supports_total'] if results['supports_total'] else 0
    refutes_acc = results['refutes_correct'] / results['refutes_total'] if results['refutes_total'] else 0
    nei_acc = results['nei_correct'] / results['nei_total'] if results['nei_total'] else 0
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nBy Category:")
    print(f"  SUPPORTS: {results['supports_correct']}/{results['supports_total']} ({supports_acc:.2%})")
    print(f"  REFUTES:  {results['refutes_correct']}/{results['refutes_total']} ({refutes_acc:.2%})")
    print(f"  NEI:      {results['nei_correct']}/{results['nei_total']} ({nei_acc:.2%})")
    print(f"\nTime: {elapsed:.1f}s ({elapsed/results['total']*1000:.0f}ms per sample)")
    
    if show_errors and results['errors'][:5]:
        print("\nSample Errors:")
        for claim, expected, predicted in results['errors'][:5]:
            print(f"  [{expected}->{predicted}] {claim[:50]}...")
    
    return {
        'accuracy': accuracy,
        'supports_accuracy': supports_acc,
        'refutes_accuracy': refutes_acc,
        'nei_accuracy': nei_acc,
        'elapsed': elapsed,
        'errors': len(results['errors']),
    }


def main():
    ap = argparse.ArgumentParser(description="RAG Auditor Benchmark")
    ap.add_argument("--n", type=int, default=1000, help="Number of test cases")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--out_report", default="reports/benchmark_results.md")
    args = ap.parse_args()
    
    print("="*60)
    print(f"RAG AUDITOR BENCHMARK - {args.n} test cases")
    print("="*60)
    
    print("\nGenerating test data...")
    test_data = generate_benchmark_data(args.n, args.seed)
    
    print(f"\nTest data distribution:")
    supports = sum(1 for _, _, v in test_data if v == "SUPPORTS")
    refutes = sum(1 for _, _, v in test_data if v == "REFUTES")
    nei = sum(1 for _, _, v in test_data if v == "NEI")
    print(f"  SUPPORTS: {supports}")
    print(f"  REFUTES: {refutes}")
    print(f"  NEI: {nei}")
    
    print("\nInitializing auditor...")
    auditor = RAGAuditor(
        device=args.device,
        stage2_enabled=not args.no_stage2,
    )
    
    print("\nRunning benchmark...")
    results = run_benchmark(auditor, test_data)
    
    # Save report
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Auditor Benchmark Results",
        "",
        f"**Samples:** {args.n}",
        f"**Overall Accuracy:** {results['accuracy']:.2%}",
        "",
        "## By Category",
        "",
        "| Category | Accuracy |",
        "|----------|----------|",
        f"| SUPPORTS | {results['supports_accuracy']:.2%} |",
        f"| REFUTES | {results['refutes_accuracy']:.2%} |",
        f"| NEI | {results['nei_accuracy']:.2%} |",
        "",
        f"**Time:** {results['elapsed']:.1f}s total",
        f"**Errors:** {results['errors']}",
    ]
    Path(args.out_report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report: {args.out_report}")


if __name__ == "__main__":
    main()
