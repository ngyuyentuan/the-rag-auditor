"""
Train a simple classifier on FEVER training data to learn optimal decision boundaries.
Uses the NLI model's outputs as features.
"""
import sys
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_fever_data(file_path: Path, max_samples: int = 5000) -> List[Dict]:
    samples = []
    label_counts = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    max_per_class = max_samples // 3
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                label = item.get("label", "")
                if label == "NOT ENOUGH INFO":
                    label = "NEI"
                if label not in label_counts:
                    continue
                if label_counts[label] >= max_per_class:
                    continue
                    
                samples.append({
                    "claim": item["claim"],
                    "label": label
                })
                label_counts[label] += 1
                
                if all(c >= max_per_class for c in label_counts.values()):
                    break
            except:
                continue
    
    print(f"Loaded {len(samples)} samples: {label_counts}")
    return samples


def extract_features(claim: str, nli_probs: Dict[str, float]) -> np.ndarray:
    p_e = nli_probs.get('entailment', 0)
    p_c = nli_probs.get('contradiction', 0)
    p_n = nli_probs.get('neutral', 0)
    
    claim_lower = claim.lower()
    has_not = 1.0 if ' not ' in claim_lower or "n't" in claim_lower else 0.0
    has_only = 1.0 if 'only' in claim_lower or 'exclusively' in claim_lower else 0.0
    has_never = 1.0 if 'never' in claim_lower else 0.0
    
    import re
    year_match = re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', claim)
    has_year = 1.0 if year_match else 0.0
    
    nat_words = ['american', 'british', 'english', 'german', 'french', 'italian', 'spanish']
    has_nationality = 1.0 if any(w in claim_lower for w in nat_words) else 0.0
    
    features = [
        p_e, p_c, p_n,
        p_e - p_c, p_c - p_n, p_e - p_n,
        max(p_e, p_c, p_n),
        has_not, has_only, has_never,
        has_year, has_nationality,
        has_not * p_c,
        has_only * p_c,
    ]
    return np.array(features)


def train_classifier(train_data: List[Dict], model_path: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    from src.auditor.fever_auditor import FeverOptimizedAuditor
    auditor = FeverOptimizedAuditor(device="cpu")
    
    X_list = []
    y_list = []
    label_to_idx = {"SUPPORTS": 0, "REFUTES": 1, "NEI": 2}
    
    print("Extracting features...")
    for i, sample in enumerate(train_data):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(train_data)}")
        
        probs = auditor._nli_inference(sample["claim"], sample["claim"])
        features = extract_features(sample["claim"], probs)
        X_list.append(features)
        y_list.append(label_to_idx[sample["label"]])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Training on {len(X)} samples...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        solver='lbfgs'
    )
    clf.fit(X_scaled, y)
    
    train_acc = clf.score(X_scaled, y)
    print(f"Training accuracy: {train_acc:.2%}")
    
    with open(model_path, 'wb') as f:
        pickle.dump({'classifier': clf, 'scaler': scaler}, f)
    print(f"Saved model to {model_path}")
    
    return clf, scaler


def evaluate_classifier(test_data: List[Dict], model_path: Path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['classifier']
    scaler = model_data['scaler']
    
    from src.auditor.fever_auditor import FeverOptimizedAuditor
    auditor = FeverOptimizedAuditor(device="cpu")
    
    idx_to_label = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
    
    correct = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    total = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0}
    
    print("Evaluating...")
    for i, sample in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_data)}")
        
        probs = auditor._nli_inference(sample["claim"], sample["claim"])
        features = extract_features(sample["claim"], probs)
        X = scaler.transform([features])
        pred_idx = clf.predict(X)[0]
        pred_label = idx_to_label[pred_idx]
        
        total[sample["label"]] += 1
        if pred_label == sample["label"]:
            correct[sample["label"]] += 1
    
    print("\n" + "=" * 50)
    print("TRAINED CLASSIFIER RESULTS")
    print("=" * 50)
    total_correct = sum(correct.values())
    total_samples = sum(total.values())
    print(f"Overall: {total_correct}/{total_samples} = {total_correct/total_samples:.2%}")
    for label in ["SUPPORTS", "REFUTES", "NEI"]:
        if total[label] > 0:
            print(f"  {label}: {correct[label]}/{total[label]} = {correct[label]/total[label]:.2%}")


if __name__ == "__main__":
    fever_file = ROOT / "data" / "fever_dev.jsonl"
    model_path = ROOT / "models" / "fever_classifier.pkl"
    model_path.parent.mkdir(exist_ok=True)
    
    all_data = load_fever_data(fever_file, max_samples=3000)
    random.shuffle(all_data)
    
    split = int(len(all_data) * 0.7)
    train_data = all_data[:split]
    test_data = all_data[split:]
    
    print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}")
    
    train_classifier(train_data, model_path)
    evaluate_classifier(test_data, model_path)
