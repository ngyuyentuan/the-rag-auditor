from pathlib import Path

from scripts.aggregate_scifact_budget_seed_sweeps import parse_report


def test_parse_report_rows(tmp_path):
    md = tmp_path / "r.md"
    md.write_text(
        "\n".join([
            "# X",
            "",
            "## router_v2_min_route_hgb",
            "",
            "| budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| 0.30 | 0.45 | 0.07 | 0.48 | 0.48 | 0.20 | 0.15 | 0.01 | 0.01 | 0.98 | 100.0 | 200.0 |",
        ]),
        encoding="utf-8",
    )
    rows = parse_report(md)
    assert len(rows) == 1
    assert rows[0]["budget"] == 0.30
    assert rows[0]["fp_accept_rate"] == 0.01
