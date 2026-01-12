"""
Large Scale Benchmark - 3000 Queries
Comprehensive evaluation of RAG Auditor system
"""
import argparse
import sys
import time
import random
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Test Data Generation
# =============================================================================

SUPPORTS_TEMPLATES = [
    # Science
    ("Water boils at 100 degrees Celsius.", "At standard atmospheric pressure, water boils at 100C (212F)."),
    ("The sun is a star.", "The Sun is the star at the center of the Solar System."),
    ("Humans have 23 pairs of chromosomes.", "Human cells contain 23 pairs of chromosomes for a total of 46."),
    ("DNA contains genetic information.", "DNA stores genetic instructions for living organisms."),
    ("Gravity causes objects to fall.", "Gravitational force pulls objects toward Earth's center."),
    ("Light travels faster than sound.", "Light travels at 300,000 km/s while sound travels at 343 m/s."),
    ("The Earth orbits the Sun.", "Earth completes one orbit around the Sun every 365.25 days."),
    ("Antibiotics kill bacteria.", "Antibiotics work by disrupting bacterial cell processes."),
    ("Plants produce oxygen.", "Through photosynthesis, plants release oxygen as a byproduct."),
    ("The brain controls body functions.", "The brain regulates all body activities through the nervous system."),
    
    # Technology
    ("Python is a programming language.", "Python is a high-level programming language created by Guido van Rossum."),
    ("The internet uses TCP/IP.", "TCP/IP is the fundamental communication protocol of the internet."),
    ("Machine learning is a type of AI.", "Machine learning is a subset of artificial intelligence."),
    ("Smartphones have processors.", "Modern smartphones contain multi-core CPUs and GPUs."),
    ("Cloud computing uses remote servers.", "Cloud services run on distributed server infrastructure."),
    
    # History
    ("World War II ended in 1945.", "WWII concluded in September 1945 with Japan's surrender."),
    ("The moon landing was in 1969.", "Apollo 11 landed on the Moon on July 20, 1969."),
    ("The internet was invented in the 20th century.", "ARPANET, the precursor to the internet, was created in 1969."),
]

REFUTES_TEMPLATES = [
    # Science myths
    ("The Earth is flat.", "Scientific evidence confirms Earth is an oblate spheroid."),
    ("Humans only use 10% of their brains.", "Brain imaging shows all areas of the brain are active."),
    ("Vaccines cause autism.", "Extensive research found no link between vaccines and autism."),
    ("Lightning never strikes twice.", "Lightning frequently strikes the same location multiple times."),
    ("The Great Wall is visible from space.", "The Great Wall is not visible from space without optical aid."),
    ("Goldfish have 3-second memory.", "Studies show goldfish can remember things for months."),
    ("Sugar causes hyperactivity.", "Scientific studies found no behavioral changes from sugar."),
    ("Cracking knuckles causes arthritis.", "Research shows no connection between knuckle cracking and arthritis."),
    ("We have only 5 senses.", "Humans have more than 5 senses including balance and temperature."),
    ("Hair grows back thicker after shaving.", "Shaving does not change hair thickness or color."),
    
    # Tech myths
    ("AI can think like humans.", "Current AI systems do not have consciousness or true understanding."),
    ("More RAM always means faster computer.", "Performance depends on many factors, not just RAM."),
    ("Macs cannot get viruses.", "Macs can be infected with malware, though less common."),
    
    # History myths
    ("Napoleon was extremely short.", "At 5'7, Napoleon was average height for his time."),
    ("Vikings wore horned helmets.", "Archaeological evidence shows Vikings wore plain helmets."),
]

NEI_TEMPLATES = [
    # Mixed evidence
    ("Coffee is healthy.", "Studies show mixed effects of coffee on health depending on consumption."),
    ("Social media causes depression.", "Research shows complex relationships between social media and mental health."),
    ("Video games cause violence.", "Studies have produced mixed results on video games and aggression."),
    ("Organic food is more nutritious.", "Evidence on nutritional differences is inconclusive."),
    ("Working from home is more productive.", "Productivity varies significantly between individuals."),
    ("Electric cars are better for environment.", "Environmental impact depends on electricity source and production."),
    ("AI will replace all jobs.", "AI will automate some tasks while creating new job categories."),
    ("Meditation improves health.", "Some studies support benefits, but evidence varies by condition."),
]

# Vietnamese test cases
VI_SUPPORTS = [
    ("Nuoc soi o 100 do C.", "Nuoc soi tai 100 do C o ap suat tieu chuan."),
    ("Trai dat quay quanh mat troi.", "Trai dat hoan thanh mot vong quay quanh mat troi trong 365 ngay."),
    ("Python la ngon ngu lap trinh.", "Python la mot ngon ngu lap trinh bac cao."),
]

VI_REFUTES = [
    ("Trai dat phang.", "Bang chung khoa hoc xac nhan trai dat hinh cau."),
    ("Vac-xin gay tu ky.", "Nghien cuu khong tim thay moi lien he giua vac-xin va tu ky."),
]

VI_NEI = [
    ("Ca phe tot cho suc khoe.", "Nghien cuu cho thay ca phe co tac dong khac nhau tuy nguoi."),
]


def generate_test_data(n: int = 3000, seed: int = 42) -> List[Tuple[str, str, str]]:
    """Generate n test cases with balanced distribution."""
    rng = random.Random(seed)
    data = []
    
    # Base distribution: 40% SUPPORTS, 30% REFUTES, 30% NEI
    n_supports = int(n * 0.4)
    n_refutes = int(n * 0.3)
    n_nei = n - n_supports - n_refutes
    
    # Generate SUPPORTS
    for _ in range(n_supports):
        if rng.random() < 0.1 and VI_SUPPORTS:  # 10% Vietnamese
            c, e = rng.choice(VI_SUPPORTS)
        else:
            c, e = rng.choice(SUPPORTS_TEMPLATES)
        # Add small variations
        if rng.random() < 0.3:
            c = c.replace(".", "").strip() + "."
        data.append((c, e, "SUPPORTS"))
    
    # Generate REFUTES
    for _ in range(n_refutes):
        if rng.random() < 0.1 and VI_REFUTES:
            c, e = rng.choice(VI_REFUTES)
        else:
            c, e = rng.choice(REFUTES_TEMPLATES)
        data.append((c, e, "REFUTES"))
    
    # Generate NEI
    for _ in range(n_nei):
        if rng.random() < 0.1 and VI_NEI:
            c, e = rng.choice(VI_NEI)
        else:
            c, e = rng.choice(NEI_TEMPLATES)
        data.append((c, e, "NEI"))
    
    rng.shuffle(data)
    return data


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(auditor, data: List[Tuple[str, str, str]], batch_size: int = 100) -> Dict:
    """Run benchmark and collect statistics."""
    results = {
        'total': len(data),
        'correct': 0,
        'by_category': {
            'SUPPORTS': {'total': 0, 'correct': 0},
            'REFUTES': {'total': 0, 'correct': 0},
            'NEI': {'total': 0, 'correct': 0},
        },
        'latencies': [],
        'errors': [],
    }
    
    start_time = time.time()
    
    for i, (claim, evidence, expected) in enumerate(data):
        try:
            result = auditor.audit(claim, evidence)
            latency = result.latency_ms
            predicted = result.verdict
            
            results['latencies'].append(latency)
            results['by_category'][expected]['total'] += 1
            
            if predicted == expected:
                results['correct'] += 1
                results['by_category'][expected]['correct'] += 1
        except Exception as e:
            results['errors'].append(str(e))
        
        # Progress
        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            acc = results['correct'] / (i + 1)
            print(f"  Progress: {i+1}/{len(data)} ({acc:.1%} accuracy, {elapsed:.0f}s)")
    
    total_time = time.time() - start_time
    
    # Compute final metrics
    results['accuracy'] = results['correct'] / results['total']
    results['total_time_s'] = total_time
    results['avg_latency_ms'] = sum(results['latencies']) / len(results['latencies']) if results['latencies'] else 0
    
    for cat in ['SUPPORTS', 'REFUTES', 'NEI']:
        cat_data = results['by_category'][cat]
        cat_data['accuracy'] = cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0
    
    return results


def print_results(results: Dict):
    """Print formatted results."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Total samples:     {results['total']}")
    print(f"Correct:           {results['correct']}")
    print(f"Overall Accuracy:  {results['accuracy']:.2%}")
    print(f"Total Time:        {results['total_time_s']:.1f}s")
    print(f"Avg Latency:       {results['avg_latency_ms']:.0f}ms")
    print(f"Errors:            {len(results['errors'])}")
    
    print("\n" + "-"*70)
    print("Accuracy by Category:")
    print("-"*70)
    for cat in ['SUPPORTS', 'REFUTES', 'NEI']:
        cat_data = results['by_category'][cat]
        print(f"  {cat:10} {cat_data['correct']:4}/{cat_data['total']:<4} = {cat_data['accuracy']:.2%}")


def save_report(results: Dict, output_path: Path):
    """Save detailed report."""
    report = f"""# Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {results['total']} |
| Correct | {results['correct']} |
| **Overall Accuracy** | **{results['accuracy']:.2%}** |
| Total Time | {results['total_time_s']:.1f}s |
| Avg Latency | {results['avg_latency_ms']:.0f}ms |
| Throughput | {results['total']/results['total_time_s']:.1f} samples/s |

## Accuracy by Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| SUPPORTS | {results['by_category']['SUPPORTS']['correct']} | {results['by_category']['SUPPORTS']['total']} | {results['by_category']['SUPPORTS']['accuracy']:.2%} |
| REFUTES | {results['by_category']['REFUTES']['correct']} | {results['by_category']['REFUTES']['total']} | {results['by_category']['REFUTES']['accuracy']:.2%} |
| NEI | {results['by_category']['NEI']['correct']} | {results['by_category']['NEI']['total']} | {results['by_category']['NEI']['accuracy']:.2%} |

## Errors

{len(results['errors'])} errors occurred during benchmark.
"""
    output_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="Number of test cases")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="reports/benchmark_3000.md")
    args = parser.parse_args()
    
    print("="*70)
    print(f"RAG AUDITOR BENCHMARK - {args.n} queries")
    print("="*70)
    
    # Generate data
    print(f"\nGenerating {args.n} test cases...")
    data = generate_test_data(args.n)
    print(f"  SUPPORTS: {sum(1 for _,_,v in data if v=='SUPPORTS')}")
    print(f"  REFUTES:  {sum(1 for _,_,v in data if v=='REFUTES')}")
    print(f"  NEI:      {sum(1 for _,_,v in data if v=='NEI')}")
    
    # Load auditor
    print("\nLoading auditor...")
    from src.auditor.multilingual_auditor import MultilingualAuditor
    auditor = MultilingualAuditor(device=args.device)
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(auditor, data)
    
    # Print results
    print_results(results)
    
    # Save report
    Path("reports").mkdir(exist_ok=True)
    save_report(results, Path(args.output))


if __name__ == "__main__":
    main()
