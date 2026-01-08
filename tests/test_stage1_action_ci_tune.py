from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_action_ci_feasible(tmp_path):
    df = pd.DataFrame({"logit": [2.0, 2.0, -2.0, -2.0, 1.5, -1.5], "y": [1, 1, 0, 0, 1, 0]})
    pq = tmp_path / "d.parquet"
    df.to_parquet(pq)
    out_md = tmp_path / "r.md"
    out_yaml = tmp_path / "t.yaml"
    cmd = [
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
        "6",
        "--min_accept_count",
        "0",
        "--min_reject_count",
        "0",
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
        "--out_md",
        str(out_md),
        "--out_yaml",
        str(out_yaml),
    ]
    subprocess.run(cmd, check=True)
    assert out_md.exists()
    assert out_yaml.exists()
    data = out_yaml.read_text()
    assert "tau" in data and "t_lower" in data and "t_upper" in data


def test_accept_zero_sets_status(tmp_path):
    df = pd.DataFrame({"logit": [0.0, 0.0], "y": [1, 0]})
    conf = {"tau": 1.0, "t_lower": -1.0, "t_upper": 1.0}
    sys.path.insert(0, str(ROOT / "scripts"))
    import print_stage1_product_table as mod

    m = mod.evaluate(df, "logit", "y", conf)
    assert m["accept_count"] == 0
