# Complete Profile Comparison - All Tracks

## Summary Table

### SciFact (300 queries, 69% positive rate)

| Profile | Coverage | OK Rate | Accuracy | FP | FN |
|---------|----------|---------|----------|----|----|
| Conservative | 44.00% | **96.33%** | 91.67% | 2.00% | 1.67% |
| Balanced | 67.33% | 90.67% | 86.14% | 4.67% | 4.67% |
| Aggressive | **78.33%** | 87.33% | 83.83% | 6.33% | 6.33% |

### FEVER (1000 queries, 51% positive rate)

| Profile | Coverage | OK Rate | Accuracy | FP | FN |
|---------|----------|---------|----------|----|----|
| Conservative | 45.40% | **96.10%** | 91.41% | 3.00% | 0.90% |
| Balanced | 72.40% | 87.80% | 83.15% | 6.00% | 6.20% |
| Aggressive | **82.40%** | 84.10% | 80.70% | 8.10% | 7.80% |

---

## Profile Selection Guide

| Use Case | Recommended Profile | Expected Metrics |
|----------|---------------------|------------------|
| Medical/Legal (high stakes) | **Conservative** | ~45% coverage, ~96% OK |
| General Search/Q&A | **Balanced** | ~70% coverage, ~88% OK |
| Chatbot/Casual | **Aggressive** | ~80% coverage, ~84% OK |

---

## Config Files Created

### SciFact
- `configs/thresholds_scifact_conservative.yaml`
- `configs/thresholds_stage1_real_scifact.yaml` (balanced)
- `configs/thresholds_scifact_aggressive.yaml`

### FEVER
- `configs/thresholds_fever_conservative.yaml`
- `configs/thresholds_fever_balanced.yaml`
- `configs/thresholds_fever_aggressive.yaml`
