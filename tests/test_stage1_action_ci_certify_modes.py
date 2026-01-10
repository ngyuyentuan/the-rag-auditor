from pathlib import Path
import pandas as pd
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_accept_only_feasible_both_not(tmp_path):
    df = pd.DataFrame({"logit": [2.0] * 8, "y": [1] * 8})
    pq = tmp_path / "data.parquet"
    df.to_parquet(pq)
    out_md_accept = tmp_path / "accept.md"
    out_yaml_accept = tmp_path / "accept.yaml"
    out_md_both = tmp_path / "both.md"
    out_yaml_both = tmp_path / "both.yaml"
    base_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "stage1_action_ci_tune.py"),
        "--track",
        "scifact",
        "--in_path",
        str(pq),
        "--logit_col",
        "logit",
        "--y_col",
        "y",
        "--n",
        "8",
        "--min_coverage",
        "0.0",
        "--tau_source",
        "manual",
        "--tau",
        "1.0",
        "--threshold_steps",
        "5",
        "--max_fp_given_accept_upper95",
        "1.0",
        "--max_fn_given_reject_upper95",
        "1.0",
    ]
    cmd_accept = base_cmd + [
        "--certify",
        "accept_only",
        "--min_accept_count",
        "1",
        "--min_reject_count",
        "5",
        "--out_md",
        str(out_md_accept),
        "--out_yaml",
        str(out_yaml_accept),
    ]
    cmd_both = base_cmd + [
        "--certify",
        "both",
        "--min_accept_count",
        "1",
        "--min_reject_count",
        "5",
        "--out_md",
        str(out_md_both),
        "--out_yaml",
        str(out_yaml_both),
    ]
    subprocess.run(cmd_accept, check=True)
    subprocess.run(cmd_both, check=True)
    assert out_md_accept.exists()
    assert out_yaml_accept.exists()
    assert out_md_both.exists()
    assert not out_yaml_both.exists()
