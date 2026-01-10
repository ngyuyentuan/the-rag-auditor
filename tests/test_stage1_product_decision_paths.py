from pathlib import Path
import stage1_product_decision as mod


def test_output_paths_include_track():
    assert "scifact" in Path("configs/thresholds_stage1_product_safety_scifact.yaml").name
    assert "fever" in Path("configs/thresholds_stage1_product_safety_fever.yaml").name
    assert "scifact" in Path("configs/thresholds_stage1_product_coverage_scifact.yaml").name
    assert "fever" in Path("configs/thresholds_stage1_product_coverage_fever.yaml").name
