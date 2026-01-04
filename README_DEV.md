# README_DEV

## Setup (WSL/Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## Day12 stage1

This branch only includes the Day12 E2E scripts. The Day12 stage1 scripts are not present here and may exist on a different branch or were removed from scope.

Sample, run, and stats:

```bash
python3 scripts/day12_sample_500.py --track scifact --n 500 --seed 12 --out runs/day12_scifact_sample.jsonl
python3 scripts/day12_run_500.py --track scifact --sample runs/day12_scifact_sample.jsonl --baseline calibrated --stage2 stub --out runs/day12_scifact_500.jsonl
python3 scripts/day12_sample_500.py --track fever --n 500 --seed 12 --out runs/day12_fever_sample.jsonl
python3 scripts/day12_run_500.py --track fever --sample runs/day12_fever_sample.jsonl --baseline calibrated --stage2 stub --out runs/day12_fever_500.jsonl
python3 scripts/day12_stats.py --tracks scifact fever --runs_dir runs --out_md reports/day12_stage1_stats.md --out_json reports/day12_compare.json
```

## Day12 e2e

Dry-run smoke:

```bash
python3 scripts/day12_e2e_run_500.py --track scifact --n 5 --seed 12 --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --dry_run_stage2 --out runs/day12_scifact_5_e2e_smoke.jsonl
python3 scripts/day12_e2e_run_500.py --track fever --n 5 --seed 12 --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --dry_run_stage2 --out runs/day12_fever_5_e2e_smoke.jsonl
python3 scripts/day12_e2e_stats.py --tracks scifact fever --runs_dir runs --out_md reports/day12_e2e_stats.md --out_json reports/day12_e2e_compare.json
```

Full run on CPU:

```bash
python3 scripts/day12_e2e_run_500.py --track scifact --n 500 --seed 12 --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 20 --rerank_keep 5 --batch_size 16 --out runs/day12_scifact_500_e2e.jsonl
python3 scripts/day12_e2e_run_500.py --track fever --n 500 --seed 12 --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 20 --rerank_keep 5 --batch_size 16 --out runs/day12_fever_500_e2e.jsonl
```

Optional CUDA run (if available):

```bash
python3 scripts/day12_e2e_run_500.py --track scifact --n 500 --seed 12 --device cuda --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 50 --rerank_keep 5 --batch_size 16 --out runs/day12_scifact_500_e2e.jsonl
```

## Troubleshooting

- numpy missing: run `python3 -m pip install -r requirements.txt`, or `python3 -m ensurepip --upgrade`, then install.
- python3-venv missing: on Ubuntu, install `python3-venv` via your package manager.
- artifacts/*.faiss.index missing: build SciFact with `python3 scripts/scifact_faiss_baseline.py --index_path artifacts/scifact.faiss.index --meta_path artifacts/scifact_faiss_meta.json`, and provide a FEVER FAISS index via `--index_path` if you use FEVER.
- CUDA OOM: lower `--batch_size` and `--rerank_topk`, or use `--device cpu`.

## Known limitations

- Day12 uses TRAIN-only sampling for sanity checks; Day10 test already consumed.
- Temperature scaling uses cs_ret = sigmoid(logit / tau); tau < 1 sharpens and tau > 1 softens.
- FEVER is expected to be weak with current thresholds.
