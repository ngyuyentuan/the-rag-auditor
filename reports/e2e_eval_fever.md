# E2E Evaluation - fever

Samples: 300, Positive Rate: 50.67%

## Configuration Comparison

| Config | Coverage | OK Rate | Accuracy | Stage2 % |
|--------|----------|---------|----------|----------|
| 1. Threshold Only | 23.67% | 90.00% | 57.75% | 0.00% |
| 2. Threshold + NLI | 100.00% | 80.33% | 80.33% | 76.33% |
| 3. ML Router Only | 66.00% | 92.33% | 88.38% | 0.00% |
| 4. ML Router + NLI | 100.00% | 87.33% | 87.33% | 34.00% |

## Best Configuration: 4. ML Router + NLI
- Coverage: 100.00%
- OK Rate: 87.33%
- Accuracy: 87.33%