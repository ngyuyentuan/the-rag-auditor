#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SCIFACT_PARQUET:-}" || -z "${FEVER_PARQUET:-}" ]]; then
  if [[ -f reports/demo_stage1.md ]]; then
    mapfile -t _paths < <(.venv/bin/python - <<'PY'
from pathlib import Path
p = Path("reports/demo_stage1.md")
lines = p.read_text(encoding="utf-8").splitlines()
paths = []
for line in lines:
    s = line.strip()
    if s.startswith("- in_path:"):
        parts = s.split("`")
        if len(parts) >= 2:
            paths.append(parts[1])
if len(paths) >= 2:
    print(paths[0])
    print(paths[1])
PY
)
    if [[ -z "${SCIFACT_PARQUET:-}" && -n "${_paths[0]:-}" ]]; then
      SCIFACT_PARQUET="${_paths[0]}"
    fi
    if [[ -z "${FEVER_PARQUET:-}" && -n "${_paths[1]:-}" ]]; then
      FEVER_PARQUET="${_paths[1]}"
    fi
  fi
fi

if [[ -z "${SCIFACT_PARQUET:-}" || -z "${FEVER_PARQUET:-}" ]]; then
  echo "SCIFACT_PARQUET and FEVER_PARQUET must be set or present in reports/demo_stage1.md"
  exit 1
fi

echo "=== joint tuning scifact ==="
stdbuf -oL -eL .venv/bin/python -u scripts/joint_tune_stage1_tau_thresholds.py --track scifact --in_path "$SCIFACT_PARQUET" --logit_col raw_max_top3 --y_col y --budgets 0.30 0.40 0.50 0.60 --tau_grid_steps 60 --grid_lower_steps 60 --grid_upper_steps 60 --out_md reports/stage1_joint_tuning_scifact.md --out_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --progress

echo "=== joint tuning fever ==="
stdbuf -oL -eL .venv/bin/python -u scripts/joint_tune_stage1_tau_thresholds.py --track fever --in_path "$FEVER_PARQUET" --logit_col logit_platt --y_col y --budgets 0.30 0.40 0.50 0.60 --tau_grid_steps 60 --grid_lower_steps 60 --grid_upper_steps 60 --out_md reports/stage1_joint_tuning_fever.md --out_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --progress

echo "=== utility tuning scifact ==="
stdbuf -oL -eL .venv/bin/python -u scripts/stage1_utility_tune.py --track scifact --in_path "$SCIFACT_PARQUET" --logit_col raw_max_top3 --y_col y --n 1000 --seed 14 --tau_source joint_grid --tau_grid_steps 60 --threshold_steps 60 --constraint_ci wilson --out_md reports/stage1_utility_tuning_scifact.md --out_yaml configs/thresholds_stage1_product_scifact.yaml --progress

echo "=== utility tuning fever ==="
stdbuf -oL -eL .venv/bin/python -u scripts/stage1_utility_tune.py --track fever --in_path "$FEVER_PARQUET" --logit_col logit_platt --y_col y --n 1000 --seed 14 --tau_source joint_grid --tau_grid_steps 60 --threshold_steps 60 --constraint_ci wilson --out_md reports/stage1_utility_tuning_fever.md --out_yaml configs/thresholds_stage1_product_fever.yaml --progress

echo "=== action ci scifact accept_only ==="
stdbuf -oL -eL .venv/bin/python -u scripts/stage1_action_ci_tune.py --track scifact --in_path "$SCIFACT_PARQUET" --logit_col raw_max_top3 --y_col y --n 1000 --seed 14 --certify accept_only --min_accept_count 50 --min_coverage 0.30 --max_fp_given_accept_upper95 0.15 --out_md reports/stage1_action_ci_tune_scifact_accept.md --out_yaml configs/thresholds_stage1_action_accept_scifact.yaml

echo "=== action ci fever reject_only ==="
stdbuf -oL -eL .venv/bin/python -u scripts/stage1_action_ci_tune.py --track fever --in_path "$FEVER_PARQUET" --logit_col logit_platt --y_col y --n 1000 --seed 14 --certify reject_only --min_accept_count 10 --min_reject_count 200 --min_coverage 0.30 --max_fn_given_reject_upper95 0.15 --out_md reports/stage1_action_ci_tune_fever_reject.md --out_yaml configs/thresholds_stage1_action_reject_fever.yaml

echo "=== eval v2 scifact ==="
stdbuf -oL -eL .venv/bin/python -u scripts/eval_stage1_product_v2.py --track scifact --in_path "$SCIFACT_PARQUET" --logit_col raw_max_top3 --y_col y --baseline_yaml configs/thresholds.yaml --joint_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --product_yaml configs/thresholds_stage1_product_scifact.yaml --n 1000 --bootstrap 2000 --cheap_checks auto --out_md reports/stage1_product_eval_v2_scifact.md

echo "=== eval v2 fever ==="
stdbuf -oL -eL .venv/bin/python -u scripts/eval_stage1_product_v2.py --track fever --in_path "$FEVER_PARQUET" --logit_col logit_platt --y_col y --baseline_yaml configs/thresholds.yaml --joint_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --product_yaml configs/thresholds_stage1_product_fever.yaml --n 1000 --bootstrap 2000 --cheap_checks auto --out_md reports/stage1_product_eval_v2_fever.md

echo "=== product table ==="
stdbuf -oL -eL .venv/bin/python -u scripts/print_stage1_product_table.py --track all --n 1000 --seed 14 --max_fp_given_accept_upper95 0.15 --max_fn_given_reject_upper95 0.15 --out_md reports/stage1_product_table.md

echo "=== decision memo ==="
stdbuf -oL -eL .venv/bin/python -u scripts/stage1_product_decision.py --scifact_report reports/stage1_product_eval_v2_scifact.md --fever_report reports/stage1_product_eval_v2_fever.md --scifact_action_accept_yaml configs/thresholds_stage1_action_accept_scifact.yaml --fever_action_reject_yaml configs/thresholds_stage1_action_reject_fever.yaml --out_md reports/stage1_product_decision.md

echo "=== pytest ==="
stdbuf -oL -eL .venv/bin/python -m pytest -q

echo "=== DONE ==="
echo "time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "reports: reports/stage1_joint_tuning_scifact.md reports/stage1_joint_tuning_fever.md reports/stage1_utility_tuning_scifact.md reports/stage1_utility_tuning_fever.md reports/stage1_action_ci_tune_scifact_accept.md reports/stage1_action_ci_tune_fever_reject.md reports/stage1_product_eval_v2_scifact.md reports/stage1_product_eval_v2_fever.md reports/stage1_product_table.md reports/stage1_product_decision.md"
