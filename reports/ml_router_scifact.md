# ML Router Results - scifact

## Model Performance
- Accuracy: 68.33%
- F1 Score: 75.32%
- CV Mean: 74.17%

## Routing Comparison

| Metric | ML Router | Baseline | Improvement |
|--------|-----------|----------|-------------|
| coverage | 66.67% | 61.67% | +5.00% |
| ok_rate | 86.67% | 68.33% | +18.33% |
| accuracy_on_decided | 80.00% | 48.65% | +31.35% |

## Top Features

| Feature | Importance |
|---------|------------|
| logit | 0.5402 |
| top1 | 0.5402 |
| topk_mean | 0.5305 |
| topk_std | 0.4949 |
| topk_entropy | 0.3518 |