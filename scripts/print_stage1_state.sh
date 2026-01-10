#!/usr/bin/env bash
set -euo pipefail

cd ~/the-rag-auditor

echo "==== REPO STATE ===="
echo "pwd: $(pwd)"
echo "top-level: $(git rev-parse --show-toplevel)"
echo "git status -sb:"
git status -sb
echo "git log -1:"
git log -1 --oneline --decorate
echo "git diff --stat:"
git diff --stat || true

echo
echo "==== ARTIFACTS: reports (latest) ===="
ls -lt reports | head -50 || true
echo
echo "==== ARTIFACTS: configs (latest) ===="
ls -lt configs | head -50 || true

PATTERN='^- n:|^- n_total:|^- seed:|logit_col|tau:|t_lower:|t_upper:|export_budget|ci_safe|decided_ci_status|action_ci_status|accept_ci_status|reject_ci_status|safety_selected|coverage_selected|PASS|FAIL|INSUFFICIENT|N/A'

REPORTS=(
  reports/stage1_joint_tuning_scifact.md
  reports/stage1_joint_tuning_fever.md
  reports/stage1_utility_tuning_scifact.md
  reports/stage1_utility_tuning_fever.md
  reports/stage1_action_ci_tune_scifact_accept.md
  reports/stage1_action_ci_tune_fever_reject.md
  reports/stage1_product_eval_v2_scifact.md
  reports/stage1_product_eval_v2_fever.md
  reports/stage1_product_table.md
  reports/stage1_product_decision.md
)

echo
echo "==== REPORT EXCERPTS ===="
for f in "${REPORTS[@]}"; do
  echo
  echo "==== $f ===="
  if [ -f "$f" ]; then
    sed -n '1,60p' "$f" || true
    echo "---- grep ----"
    grep -nE "$PATTERN" "$f" | head -120 || true
  else
    echo "MISSING"
  fi
done

YAMLS=(
  configs/thresholds.yaml
  configs/thresholds_stage1_joint_tuned_scifact.yaml
  configs/thresholds_stage1_joint_tuned_fever.yaml
  configs/thresholds_stage1_product_scifact.yaml
  configs/thresholds_stage1_product_fever.yaml
  configs/thresholds_stage1_action_accept_scifact.yaml
  configs/thresholds_stage1_action_reject_fever.yaml
)

echo
echo "==== YAML CONTENTS ===="
for y in "${YAMLS[@]}"; do
  echo
  echo "==== $y ===="
  if [ -f "$y" ]; then
    cat "$y"
  else
    echo "MISSING"
  fi
done

echo
echo "==== SUMMARY ===="

missing=()
for f in "${REPORTS[@]}" "${YAMLS[@]}"; do
  if [ ! -f "$f" ]; then
    missing+=("$f")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "missing_files:"
  printf ' - %s\n' "${missing[@]}"
else
  echo "missing_files: none"
fi

echo
echo "n/seed:"
for f in reports/stage1_product_table.md reports/stage1_product_decision.md reports/stage1_product_eval_v2_scifact.md reports/stage1_product_eval_v2_fever.md; do
  if [ -f "$f" ]; then
    n_line=$(grep -m1 -E "^- n:|^- n_total:" "$f" || true)
    seed_line=$(grep -m1 -E "^- seed:" "$f" || true)
    echo "- $f: ${n_line:-n: n/a} ${seed_line:-seed: n/a}"
  else
    echo "- $f: missing"
  fi
done

if [ -f reports/stage1_product_table.md ]; then
  echo
  echo "decided_ci_status=PASS rows (by track):"
  awk -F'\\|' '
    function trim(s){ gsub(/^[ \t]+|[ \t]+$/,"",s); return s }
    /^## / {track=$0; gsub(/^## /,"",track); next}
    /^\| config / {
      for (i=1;i<=NF;i++) {
        h=trim($i)
        if (h=="config") cfg=i
        if (h=="decided_ci_status") dci=i
      }
      next
    }
    /^\|---/ {next}
    /^\|/ {
      c=trim($cfg); d=trim($dci)
      if (d=="PASS") print "- " track ": " c
    }
  ' reports/stage1_product_table.md || true

  echo
  echo "action rows (certify + action_ci_status + accept/reject counts):"
  awk -F'\\|' '
    function trim(s){ gsub(/^[ \t]+|[ \t]+$/,"",s); return s }
    /^## / {track=$0; gsub(/^## /,"",track); next}
    /^\| config / {
      for (i=1;i<=NF;i++) {
        h=trim($i)
        if (h=="config") cfg=i
        if (h=="action_ci_status") aci=i
        if (h=="certify") cert=i
        if (h=="accept_count") acc=i
        if (h=="reject_count") rcc=i
      }
      next
    }
    /^\|---/ {next}
    /^\|/ {
      c=trim($cfg); a=trim($aci); ce=trim($cert)
      ac=trim($acc); rc=trim($rcc)
      if (c=="action_accept" || c=="action_reject") {
        print "- " track ": " c " certify=" ce " action_ci_status=" a " accept_count=" ac " reject_count=" rc
      }
    }
  ' reports/stage1_product_table.md || true
else
  echo "product_table missing; cannot summarize decided/action CI status"
fi
