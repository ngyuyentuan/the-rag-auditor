"""
AI Response Checker - Interactive CLI

Check AI-generated responses for factual accuracy.

Usage:
    python scripts/check_ai_response.py
    python scripts/check_ai_response.py --claim "Your claim" --evidence "Evidence"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.rag_auditor import RAGAuditor


def colorize(text, color):
    """Add color to text for terminal output."""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m',
    }
    return f"{colors.get(color, '')}{text}{colors.get('reset', '')}"


def print_result(result):
    """Pretty print audit result."""
    verdict_colors = {
        'SUPPORTS': 'green',
        'REFUTES': 'red',
        'NEI': 'yellow',
        'UNCERTAIN': 'yellow',
    }
    color = verdict_colors.get(result.verdict, 'blue')
    
    print("\n" + "="*60)
    print(colorize(f"VERDICT: {result.verdict}", color))
    print(f"Confidence: {result.confidence:.1%}")
    print("-"*60)
    print(f"Stage1: {result.stage1_decision}")
    if result.stage2_ran:
        print(f"Stage2: {result.stage2_decision}")
    print(f"\nExplanation: {result.explanation}")
    print("="*60)


def interactive_mode(auditor):
    """Run in interactive mode."""
    print("\n" + "="*60)
    print("AI RESPONSE CHECKER - Interactive Mode")
    print("="*60)
    print("Enter claims and evidence to check. Type 'quit' to exit.\n")
    
    while True:
        print("-"*60)
        claim = input("Claim (or 'quit'): ").strip()
        if claim.lower() in ('quit', 'exit', 'q'):
            break
        if not claim:
            continue
            
        evidence = input("Evidence: ").strip()
        if not evidence:
            print("Evidence required!")
            continue
        
        result = auditor.audit(claim, evidence)
        print_result(result)
    
    print("\nGoodbye!")


def main():
    ap = argparse.ArgumentParser(description="AI Response Checker")
    ap.add_argument("--claim", help="Claim to verify")
    ap.add_argument("--evidence", help="Evidence text")
    ap.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--no-stage2", action="store_true", help="Disable Stage2 NLI")
    args = ap.parse_args()
    
    print("Initializing AI Response Checker...")
    auditor = RAGAuditor(
        device=args.device,
        stage2_enabled=not args.no_stage2,
    )
    
    if args.interactive or (not args.claim):
        interactive_mode(auditor)
    else:
        if not args.evidence:
            print("Error: --evidence required when using --claim")
            return
        
        print(f"\nClaim: {args.claim}")
        print(f"Evidence: {args.evidence[:100]}...")
        
        result = auditor.audit(args.claim, args.evidence)
        print_result(result)
        
        # Return exit code based on verdict
        if result.verdict == "SUPPORTS":
            sys.exit(0)
        elif result.verdict == "REFUTES":
            sys.exit(1)
        else:
            sys.exit(2)


if __name__ == "__main__":
    main()
