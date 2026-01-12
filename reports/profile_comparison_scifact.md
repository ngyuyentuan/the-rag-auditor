# SciFact Profile Comparison Report

## Dataset
- Source: BEIR SciFact (test split)
- Queries: 300
- Positive Rate: 69.33%

## Profile Comparison

| Profile | Coverage | OK Rate | Accuracy | FP Rate | FN Rate | Trade-off |
|---------|----------|---------|----------|---------|---------|-----------|
| **Conservative** | 44.00% | **96.33%** | 91.67% | 2.00% | 1.67% | Low coverage, very safe |
| **Balanced** | 67.33% | 90.67% | 86.14% | 4.67% | 4.67% | Good balance |
| **Aggressive** | **78.33%** | 87.33% | 83.83% | 6.33% | 6.33% | High coverage, more risk |

## Thresholds

| Profile | tau | t_lower | t_upper |
|---------|-----|---------|---------|
| Conservative | 3.0000 | 0.2679 | 0.8538 |
| Balanced | 1.5796 | 0.3878 | 0.8837 |
| Aggressive | 0.8436 | 0.5115 | 0.9513 |

## Recommendation

- **High-stakes applications** (medical, legal): Use **Conservative** (96% OK Rate)
- **General Q&A**: Use **Balanced** (67% coverage, 91% OK)
- **Chatbot/casual**: Use **Aggressive** (78% coverage)
