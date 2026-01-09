from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_eval(path, n, seed):
    text = [
        "# Stage1 Product Eval V2 (scifact)",
        "",
        f"- in_path: `{path.parent / 'data.parquet'}`",
        f"- n: {n}",
        f"- seed: {seed}",
        "- logit_col: `logit`",
        "- y_col: `y`",
        "",
        "## baseline (checks off)",
        "",
        "- tau: `1.0`",
        "- t_lower: `0.2`",
        "- t_upper: `0.8`",
        "",
        "| metric | value | 95% CI |",
        "|---|---:|---:|",
        "| uncertain_rate | 0.1 | [0.0, 0.2] |",
        "| fp_accept_rate | 0.0 | [0.0, 0.1] |",
        "| fn_reject_rate | 0.0 | [0.0, 0.1] |",
        "| ok_rate | 1.0 | [0.9, 1.0] |",
        "| coverage | 0.9 | [0.8, 1.0] |",
        "| accuracy_on_decided | 1.0 | [0.9, 1.0] |",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def test_decision_meta_lines(tmp_path):
    scifact_report = tmp_path / "scifact.md"
    fever_report = tmp_path / "fever.md"
    write_eval(scifact_report, 10, 7)
    write_eval(fever_report, 20, 7)
    out_md = tmp_path / "decision.md"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "stage1_product_decision.py"),
        "--scifact_report",
        str(scifact_report),
        "--fever_report",
        str(fever_report),
        "--out_md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True)
    text = out_md.read_text(encoding="utf-8")
    assert "- n_scifact:" in text
    assert "- n_fever:" in text
    assert "- seed:" in text or "- seed_scifact:" in text
