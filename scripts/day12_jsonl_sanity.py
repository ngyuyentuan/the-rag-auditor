import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    args = ap.parse_args()

    total = 0
    parsed = 0
    bad = 0
    stage2_ran = 0
    rerank_gt0 = 0
    nli_gt0 = 0
    missing_stage2 = 0

    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                bad += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            parsed += 1
            st = row.get("stage2", {}) or {}
            if st.get("ran") is True:
                stage2_ran += 1
            timing = row.get("timing_ms", {}) or {}
            if float(timing.get("rerank_ms", 0.0)) > 0:
                rerank_gt0 += 1
            if float(timing.get("nli_ms", 0.0)) > 0:
                nli_gt0 += 1
            if st.get("rerank", {}).get("reason") == "missing_stage2_artifacts":
                missing_stage2 += 1

    print(
        "total_lines",
        total,
        "parsed_rows",
        parsed,
        "bad_lines",
        bad,
        "stage2_ran_count",
        stage2_ran,
        "rerank_ms_gt0_count",
        rerank_gt0,
        "nli_ms_gt0_count",
        nli_gt0,
        "missing_stage2_artifacts_count",
        missing_stage2,
    )


if __name__ == "__main__":
    main()
