import json
import pathlib
import subprocess
import sys

import pytest


def run_e2e(track, out_path, tmp_path, n=5, dry_run=True, baseline_mode="calibrated", stage2_policy="uncertain_only", in_path=None, sample_path=None):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/day12_e2e_run_500.py",
        "--track",
        track,
        "--n",
        str(n),
        "--seed",
        "12",
        "--device",
        "cpu",
        "--stage2_policy",
        stage2_policy,
        "--baseline_mode",
        baseline_mode,
        "--rerank_topk",
        "5",
        "--rerank_keep",
        "2",
        "--nli_topn",
        "1",
        "--batch_size",
        "2",
        "--out",
        str(out_path),
    ]
    if in_path:
        cmd.extend(["--in_path", str(in_path)])
    if sample_path:
        cmd.extend(["--sample", str(sample_path)])
    if dry_run:
        cmd.append("--dry_run_stage2")
    subprocess.check_call(cmd, cwd=repo_root)


def check_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert lines
    for line in lines:
        row = json.loads(line)
        for key in ["metadata", "stage1", "stage2", "timing_ms"]:
            assert key in row


@pytest.mark.slow
def test_day12_e2e_smoke(tmp_path):
    import pandas as pd

    scifact_path = tmp_path / "scifact_train.parquet"
    fever_path = tmp_path / "fever_train.parquet"
    df_scifact = pd.DataFrame({
        "qid": ["0", "1"],
        "raw_max_top3": [0.5, 0.2],
        "y": [1, 0],
    })
    df_fever = pd.DataFrame({
        "qid": [0, 1],
        "raw_max_top3": [0.3, -0.1],
        "logit_platt": [0.3, -0.1],
        "y": [1, 0],
    })
    df_scifact.to_parquet(scifact_path, index=False)
    df_fever.to_parquet(fever_path, index=False)

    scifact_sample = tmp_path / "scifact_sample.jsonl"
    fever_sample = tmp_path / "fever_sample.jsonl"
    scifact_sample.write_text("{\"row_idx\":0}\n{\"row_idx\":1}\n", encoding="utf-8")
    fever_sample.write_text("{\"row_idx\":0}\n{\"row_idx\":1}\n", encoding="utf-8")

    out_scifact = tmp_path / "day12_scifact_500_e2e.jsonl"
    out_fever = tmp_path / "day12_fever_500_e2e.jsonl"
    run_e2e("scifact", out_scifact, tmp_path, n=2, dry_run=True, in_path=scifact_path, sample_path=scifact_sample, stage2_policy="always")
    run_e2e("fever", out_fever, tmp_path, n=2, dry_run=True, in_path=fever_path, sample_path=fever_sample, stage2_policy="always")
    check_jsonl(out_scifact)
    check_jsonl(out_fever)
    with open(out_scifact, "r", encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["stage2"]["rerank"].get("skipped") is True
    assert row["stage2"]["nli"].get("skipped") is True
    assert "route_requested" in row["stage2"]
    assert "capped" in row["stage2"]
    assert "cap_budget" in row["stage2"]
    assert "capped_reason" in row["stage2"]

    out_real = tmp_path / "day12_scifact_500_e2e_real.jsonl"
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    scifact_index = repo_root / "artifacts" / "scifact.faiss.index"
    scifact_corpus = repo_root / "data" / "beir_scifact" / "scifact" / "corpus.jsonl"
    scifact_queries = repo_root / "data" / "beir_scifact" / "scifact" / "queries.jsonl"
    if not (scifact_index.exists() and scifact_corpus.exists() and scifact_queries.exists()):
        pytest.skip("missing artifacts for real E2E run")
    run_e2e("scifact", out_real, tmp_path, n=2, dry_run=False, stage2_policy="always", in_path=scifact_path, sample_path=scifact_sample)
    with open(out_real, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    found = False
    for row in rows:
        if row.get("stage2", {}).get("ran"):
            rerank = row["stage2"].get("rerank", {})
            nli = row["stage2"].get("nli", {})
            if rerank.get("model") and rerank.get("scores") and nli.get("model") and nli.get("top_label"):
                found = True
                break
    assert found
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    stats_md = tmp_path / "day12_e2e_stats.md"
    stats_json = tmp_path / "day12_e2e_compare.json"
    cmd = [
        sys.executable,
        "scripts/day12_e2e_stats.py",
        "--tracks",
        "scifact",
        "fever",
        "--runs_dir",
        str(tmp_path),
        "--out_md",
        str(stats_md),
        "--out_json",
        str(stats_json),
    ]
    subprocess.check_call(cmd, cwd=repo_root)
    assert stats_md.exists()
    assert stats_json.exists()


def test_router_budget_cap_keeps_uncertain(tmp_path):
    from types import SimpleNamespace

    class DummyModel:
        def predict_proba(self, x):
            import numpy as np
            return np.array([[0.5, 0.5]] * len(x))

    from scripts.run_product_proof_router_v2_matrix import apply_router_v2

    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    row = {
        "metadata": {"qid": "1"},
        "stage1": {"route_decision": "UNCERTAIN", "route_reason": "baseline"},
        "stage2": {"ran": True, "rerank": {"model": "x"}, "nli": {"model": "x"}},
        "timing_ms": {"stage1_ms": 1.0, "rerank_ms": 1.0, "nli_ms": 1.0, "total_ms": 3.0},
        "ground_truth": {"y": 1},
        "pred": {"pred_verdict": "SUPPORTS"},
    }
    in_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    router = SimpleNamespace(model=DummyModel(), feature_cols=["f1"], logit_col="f1", prob_flip=False)
    import numpy as np
    feature_map = {"1": np.array([0.0])}
    apply_router_v2(in_path, out_path, router, feature_map, 0.8, 0.2, "uncertain_only", 0.0, {"f1": True})
    out = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert out["stage1"]["route_decision"] == "UNCERTAIN"
    assert out["stage2"]["ran"] is False
    assert out["stage2"]["capped"] is True
    assert out["stage2"]["capped_reason"] == "budget_cap"
    assert out["stage2"]["rerank"]["reason"] == "budget_cap"
    assert out["stage2"]["skipped_reason"] == "budget_cap"


def test_router_budget_cap_deterministic_priority(tmp_path):
    from types import SimpleNamespace
    import numpy as np

    class DummyModel:
        def predict_proba(self, x):
            probs = []
            for row in x:
                p = float(row[0])
                probs.append([1.0 - p, p])
            return np.array(probs)

    from scripts.run_product_proof_router_v2_matrix import apply_router_v2

    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    rows = [
        {"metadata": {"qid": "1"}, "stage1": {"route_decision": "UNCERTAIN", "route_reason": "baseline", "cs_ret": 0.5}, "stage2": {"ran": True}, "timing_ms": {"rerank_ms": 1.0, "nli_ms": 1.0, "total_ms": 2.0}, "ground_truth": {"y": 1}},
        {"metadata": {"qid": "2"}, "stage1": {"route_decision": "UNCERTAIN", "route_reason": "baseline", "cs_ret": 0.9}, "stage2": {"ran": True}, "timing_ms": {"rerank_ms": 1.0, "nli_ms": 1.0, "total_ms": 2.0}, "ground_truth": {"y": 0}},
    ]
    in_path.write_text("\n".join([json.dumps(r) for r in rows]) + "\n", encoding="utf-8")
    router = SimpleNamespace(model=DummyModel(), feature_cols=["f1"], logit_col="f1", prob_flip=False)
    feature_map = {"1": np.array([0.5]), "2": np.array([0.9])}
    apply_router_v2(in_path, out_path, router, feature_map, 0.8, 0.2, "uncertain_only", 0.5, {"f1": True})
    out = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ran = [r["metadata"]["qid"] for r in out if r["stage2"]["ran"]]
    assert ran == ["1"]
