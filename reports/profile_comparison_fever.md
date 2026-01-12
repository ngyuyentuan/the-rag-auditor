# FEVER Profile Comparison Report

## Dataset
- Source: FEVER-style calibration data
- Queries: 1000
- Positive Rate: 50.70%

## Profile Comparison

| Profile | Coverage | OK Rate | Accuracy | FP Rate | FN Rate | Trade-off |
|---------|----------|---------|----------|---------|---------|-----------|
| **Conservative** | 45.40% | **96.10%** | 91.41% | 3.00% | 0.90% | Low coverage, very safe |
| **Balanced** | 72.40% | 87.80% | 83.15% | 6.00% | 6.20% | Good balance |
| **Aggressive** | **82.40%** | 84.10% | 80.70% | 8.10% | 7.80% | High coverage, more risk |

## Thresholds

| Profile | tau | t_lower | t_upper |
|---------|-----|---------|---------|
| Conservative | 1.8103 | 0.3897 | 0.6833 |
| Balanced | 1.5872 | 0.5115 | 0.6590 |
| Aggressive | 1.2897 | 0.5359 | 0.6590 |

## Comparison: SciFact vs FEVER

| Metric | SciFact (Balanced) | FEVER (Balanced) |
|--------|-------------------|------------------|
| Coverage | 67.33% | 72.40% |
| OK Rate | 90.67% | 87.80% |
| Accuracy | 86.14% | 83.15% |
| FP Rate | 4.67% | 6.00% |
| FN Rate | 4.67% | 6.20% |

> FEVER shows slightly lower accuracy due to harder claim verification task.
