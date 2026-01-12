"""
Analyze FEVER errors to understand failure modes
"""
import sys
import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyze_fever_errors():
    from src.auditor.commercial_auditor import CommercialAuditor
    
    fever_file = ROOT / "data" / "fever_dev.jsonl"
    if not fever_file.exists():
        print("FEVER file not found")
        return
    
    samples = []
    with open(fever_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 300:
                break
            try:
                item = json.loads(line.strip())
                if item.get("label") in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                    evidence_text = ""
                    if item.get("evidence"):
                        for ev_group in item["evidence"]:
                            if ev_group:
                                for ev in ev_group:
                                    if len(ev) >= 3:
                                        evidence_text += str(ev[2]) + " "
                    
                    label = item["label"]
                    if label == "NOT ENOUGH INFO":
                        label = "NEI"
                    samples.append({
                        "claim": item["claim"],
                        "evidence": evidence_text.strip() if evidence_text.strip() else item["claim"],
                        "label": label
                    })
            except:
                continue
    
    print(f"Loaded {len(samples)} samples")
    
    auditor = CommercialAuditor(device="cpu")
    
    errors = {"SUPPORTS": [], "REFUTES": [], "NEI": []}
    
    for sample in samples:
        result = auditor.audit(sample["claim"], sample["evidence"])
        if result.verdict != sample["label"]:
            errors[sample["label"]].append({
                "claim": sample["claim"],
                "evidence": sample["evidence"],
                "expected": sample["label"],
                "predicted": result.verdict,
                "confidence": result.confidence,
                "explanation": result.explanation,
            })
    
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    for label, errs in errors.items():
        print(f"\n{label} Errors: {len(errs)}")
        for e in errs[:5]:
            print(f"  Claim: {e['claim'][:60]}...")
            print(f"  Evidence: {e['evidence'][:60]}...")
            print(f"  Expected: {e['expected']}, Got: {e['predicted']}")
            print(f"  Reason: {e['explanation']}")
            print()
    
    print("\n" + "=" * 60)
    print("COMMON PATTERNS IN ERRORS")
    print("=" * 60)
    
    for label, errs in errors.items():
        if not errs:
            continue
        print(f"\n{label} -> predicted as:")
        pred_counter = Counter(e["predicted"] for e in errs)
        for pred, count in pred_counter.most_common():
            print(f"  {pred}: {count}")


if __name__ == "__main__":
    analyze_fever_errors()
