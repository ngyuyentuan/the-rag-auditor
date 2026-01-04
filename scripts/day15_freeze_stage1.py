import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution, stage1_outcomes


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


def parse_in_paths(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = []
    for line in lines:
        s = line.strip()
        if s.startswith("- in_path:"):
            parts = s.split(chr(96))
            if len(parts) >= 2 and parts[1]:
                paths.append(parts[1])
    if len(paths) < 2:
        raise SystemExit("missing in_path entries in reports/demo_stage1.md")
    return paths[0], paths[1]


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_version(mod_name):
    try:
        mod = __import__(mod_name)
    except Exception:
        return "not installed"
    return getattr(mod, "__version__", "unknown")


def load_thresholds(thresholds_yaml: Path, tuned_yaml: Path | None, track: str):
    if tuned_yaml and tuned_yaml.exists():
        data = yaml.safe_load(tuned_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise SystemExit(f"invalid tuned thresholds in {tuned_yaml}")
        for k in ("tau", "t_lower", "t_upper"):
            if k not in data:
                raise SystemExit(f"missing {k} in tuned thresholds {tuned_yaml}")
        return {
            "tau": float(data["tau"]),
            "t_lower": float(data["t_lower"]),
            "t_upper": float(data["t_upper"]),
            "source": str(tuned_yaml),
        }
    data = yaml.safe_load(thresholds_yaml.read_text(encoding="utf-8")) or {}
    t = (data.get("thresholds") or {}).get(track)
    if not t:
        raise SystemExit(f"missing thresholds for track={track} in {thresholds_yaml}")
    for k in ("tau", "t_lower", "t_upper"):
        if k not in t:
            raise SystemExit(f"missing thresholds.{track}.{k} in {thresholds_yaml}")
    return {
        "tau": float(t["tau"]),
        "t_lower": float(t["t_lower"]),
        "t_upper": float(t["t_upper"]),
        "source": str(thresholds_yaml),
    }


def load_rows(in_path: Path, logit_col: str, n: int, seed: int):
    df = pd.read_parquet(in_path)
    if logit_col not in df.columns:
        raise SystemExit(f"missing logit_col {logit_col} in {in_path}")
    if "y" not in df.columns:
        raise SystemExit(f"missing y in {in_path}")
    df[logit_col] = pd.to_numeric(df[logit_col], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=[logit_col, "y"]).copy()
    df = df[np.isfinite(df[logit_col].to_numpy(np.float64))]
    df = df[(df["y"] == 0) | (df["y"] == 1)]
    df["y"] = df["y"].astype(int)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    rows = [(float(row[logit_col]), int(row["y"])) for _, row in df.iterrows()]
    if not rows:
        raise SystemExit(f"no valid rows in {in_path}")
    return rows


def compute_metrics(rows, tau, t_lower, t_upper):
    cs_list = []
    decisions = []
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    for logit, y in rows:
        cs_ret = compute_cs_ret_from_logit(logit, tau)
        decision, _ = decide_route(cs_ret, t_lower, t_upper)
        cs_list.append(cs_ret)
        decisions.append(decision)
        out = stage1_outcomes(decision, y)
        for k in counts:
            counts[k] += out[k]
    dist = routing_distribution(cs_list, t_lower, t_upper)
    n = len(rows) or 1
    fp = counts["fp"] / n
    fn = counts["fn"] / n
    return {
        "dist": dist,
        "fp_accept_rate": fp,
        "fn_reject_rate": fn,
        "ok_rate": 1.0 - fp - fn,
    }


def tag_info():
    head = run(["git", "rev-parse", "HEAD"])
    try:
        tag_obj = run(["git", "rev-parse", "stage1_freeze"])
        tag_target = run(["git", "rev-parse", "stage1_freeze^{}"])
        return {
            "head": head,
            "tag": "stage1_freeze",
            "tag_obj": tag_obj,
            "tag_target": tag_target,
            "tag_matches": tag_target == head,
        }
    except Exception:
        return {
            "head": head,
            "tag": "stage1_freeze",
            "tag_obj": "not present",
            "tag_target": "not present",
            "tag_matches": False,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", default=None)
    ap.add_argument("--fever_in_path", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--tuned_scifact_yaml", default=None)
    ap.add_argument("--tuned_fever_yaml", default=None)
    ap.add_argument("--out_md", default="reports/stage1_freeze_checklist.md")
    args = ap.parse_args()

    scifact_path = args.scifact_in_path
    fever_path = args.fever_in_path
    if not scifact_path or not fever_path:
        demo_md = Path("reports/demo_stage1.md")
        scifact_path, fever_path = parse_in_paths(demo_md)

    tuned_scifact = Path(args.tuned_scifact_yaml) if args.tuned_scifact_yaml else Path("configs/thresholds_stage1_tuned_scifact.yaml")
    tuned_fever = Path(args.tuned_fever_yaml) if args.tuned_fever_yaml else Path("configs/thresholds_stage1_tuned_fever.yaml")
    thresholds_yaml = Path(args.thresholds_yaml)

    scifact_thresholds = load_thresholds(thresholds_yaml, tuned_scifact, "scifact")
    fever_thresholds = load_thresholds(thresholds_yaml, tuned_fever, "fever")

    scifact_rows = load_rows(Path(scifact_path), "raw_max_top3", args.n, args.seed)
    fever_rows = load_rows(Path(fever_path), "logit_platt", args.n, args.seed)

    scifact_metrics = compute_metrics(scifact_rows, scifact_thresholds["tau"], scifact_thresholds["t_lower"], scifact_thresholds["t_upper"])
    fever_metrics = compute_metrics(fever_rows, fever_thresholds["tau"], fever_thresholds["t_lower"], fever_thresholds["t_upper"])

    files_to_hash = [
        Path("src/utils/buffer.py"),
        Path("configs/thresholds.yaml"),
        Path("scripts/day12_e2e_run_500.py"),
        Path("scripts/tune_stage1_thresholds.py"),
        Path("scripts/demo_stage1.py"),
    ]
    if tuned_scifact.exists():
        files_to_hash.append(tuned_scifact)
    if tuned_fever.exists():
        files_to_hash.append(tuned_fever)

    hashes = {str(p): sha256_file(p) for p in files_to_hash}
    tag = tag_info()

    lines = []
    lines.append("# Stage1 Freeze Checklist")
    lines.append("")
    lines.append("## Repo state")
    lines.append(f"- head: `{tag['head']}`")
    lines.append(f"- tag: `{tag['tag']}`")
    lines.append(f"- tag_object: `{tag['tag_obj']}`")
    lines.append(f"- tag_target: `{tag['tag_target']}`")
    lines.append(f"- tag_matches_head: `{tag['tag_matches']}`")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- python: `{sys.version.splitlines()[0]}`")
    lines.append(f"- numpy: `{get_version('numpy')}`")
    lines.append(f"- pandas: `{get_version('pandas')}`")
    lines.append(f"- torch: `{get_version('torch')}`")
    lines.append(f"- transformers: `{get_version('transformers')}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- scifact_in_path: `{scifact_path}`")
    lines.append(f"- fever_in_path: `{fever_path}`")
    lines.append(f"- n: `{args.n}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- scifact_source: `{scifact_thresholds['source']}`")
    lines.append(f"- scifact_tau: `{scifact_thresholds['tau']}`")
    lines.append(f"- scifact_t_lower: `{scifact_thresholds['t_lower']}`")
    lines.append(f"- scifact_t_upper: `{scifact_thresholds['t_upper']}`")
    lines.append(f"- fever_source: `{fever_thresholds['source']}`")
    lines.append(f"- fever_tau: `{fever_thresholds['tau']}`")
    lines.append(f"- fever_t_lower: `{fever_thresholds['t_lower']}`")
    lines.append(f"- fever_t_upper: `{fever_thresholds['t_upper']}`")
    lines.append("")
    lines.append("## Stage1 distribution")
    lines.append("### scifact")
    lines.append(f"- accept_rate: `{scifact_metrics['dist']['ACCEPT']}`")
    lines.append(f"- reject_rate: `{scifact_metrics['dist']['REJECT']}`")
    lines.append(f"- uncertain_rate: `{scifact_metrics['dist']['UNCERTAIN']}`")
    lines.append(f"- fp_accept_rate: `{scifact_metrics['fp_accept_rate']}`")
    lines.append(f"- fn_reject_rate: `{scifact_metrics['fn_reject_rate']}`")
    lines.append(f"- ok_rate: `{scifact_metrics['ok_rate']}`")
    lines.append("### fever")
    lines.append(f"- accept_rate: `{fever_metrics['dist']['ACCEPT']}`")
    lines.append(f"- reject_rate: `{fever_metrics['dist']['REJECT']}`")
    lines.append(f"- uncertain_rate: `{fever_metrics['dist']['UNCERTAIN']}`")
    lines.append(f"- fp_accept_rate: `{fever_metrics['fp_accept_rate']}`")
    lines.append(f"- fn_reject_rate: `{fever_metrics['fn_reject_rate']}`")
    lines.append(f"- ok_rate: `{fever_metrics['ok_rate']}`")
    lines.append("")
    lines.append("## File hashes (sha256)")
    for k in sorted(hashes):
        lines.append(f"- {k}: `{hashes[k]}`")
    if not tuned_scifact.exists():
        lines.append(f"- {tuned_scifact}: `not present`")
    if not tuned_fever.exists():
        lines.append(f"- {tuned_fever}: `not present`")
    lines.append("")
    lines.append("## Reproduce commands")
    lines.append("```bash")
    lines.append(f".venv/bin/python scripts/demo_stage1.py --track scifact --n {args.n} --seed {args.seed} --scifact_in_path '{scifact_path}' --logit_col raw_max_top3 --out_md reports/demo_stage1.md")
    lines.append(f".venv/bin/python scripts/demo_stage1.py --track fever --n {args.n} --seed {args.seed} --fever_in_path '{fever_path}' --logit_col logit_platt --out_md reports/demo_stage1.md")
    lines.append(f".venv/bin/python scripts/tune_stage1_thresholds.py --track scifact --in_path '{scifact_path}' --logit_col raw_max_top3 --y_col y --budgets 0.30 0.40 0.50 0.60")
    lines.append(f".venv/bin/python scripts/tune_stage1_thresholds.py --track fever --in_path '{fever_path}' --logit_col logit_platt --y_col y --budgets 0.30 0.40 0.50 0.60")
    lines.append(f".venv/bin/python scripts/day12_e2e_run_500.py --track scifact --n 50 --seed {args.seed} --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 20 --rerank_keep 5 --batch_size 16 --out runs/day12_scifact_50_e2e_stage1only.jsonl --in_path '{scifact_path}' --dry_run_stage2")
    lines.append(f".venv/bin/python scripts/day12_e2e_run_500.py --track fever --n 50 --seed {args.seed} --device cpu --stage2_policy uncertain_only --baseline_mode calibrated --rerank_topk 20 --rerank_keep 5 --batch_size 16 --out runs/day12_fever_50_e2e_stage1only.jsonl --in_path '{fever_path}' --logit_col logit_platt --y_col y --dry_run_stage2")
    lines.append(".venv/bin/python scripts/day12_e2e_stats.py --tracks scifact fever --runs_dir runs --out_md reports/day12_e2e_stats.md --out_json reports/day12_e2e_compare.json")
    lines.append("```")
    lines.append("")
    lines.append("## Expected outputs")
    lines.append("- reports/demo_stage1.md")
    lines.append("- reports/stage1_threshold_tuning_scifact.md")
    lines.append("- reports/stage1_threshold_tuning_fever.md")
    lines.append("- configs/thresholds_stage1_tuned_scifact.yaml")
    lines.append("- configs/thresholds_stage1_tuned_fever.yaml")
    lines.append("- runs/day12_scifact_50_e2e_stage1only.jsonl")
    lines.append("- runs/day12_fever_50_e2e_stage1only.jsonl")
    lines.append("- reports/day12_e2e_stats.md")
    lines.append("- reports/day12_e2e_compare.json")
    lines.append("")
    lines.append("## Freeze definition")
    lines.append("Do not change the parquet inputs, their sha256, or the threshold sources after this tag. Stage1 routing logic and tuned thresholds are locked for Stage2 work.")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
