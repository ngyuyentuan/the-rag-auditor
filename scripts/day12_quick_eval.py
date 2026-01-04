import argparse
import json
import os


def new_counts():
    return {
        "total": 0,
        "parsed": 0,
        "bad": 0,
        "accept": 0,
        "reject": 0,
        "uncertain": 0,
        "fp": 0,
        "fn": 0,
        "stage2_ran": 0,
        "rerank_gt0": 0,
        "nli_gt0": 0,
    }


def update_counts(c, row):
    c["parsed"] += 1
    stage1 = row.get("stage1", {}) or {}
    decision = stage1.get("route_decision")
    if decision == "ACCEPT":
        c["accept"] += 1
    elif decision == "REJECT":
        c["reject"] += 1
    else:
        c["uncertain"] += 1
    y = (row.get("ground_truth") or {}).get("y")
    if decision == "ACCEPT" and y == 0:
        c["fp"] += 1
    if decision == "REJECT" and y == 1:
        c["fn"] += 1
    stage2 = row.get("stage2", {}) or {}
    if stage2.get("ran") is True:
        c["stage2_ran"] += 1
    timing = row.get("timing_ms", {}) or {}
    if float(timing.get("rerank_ms", 0.0)) > 0:
        c["rerank_gt0"] += 1
    if float(timing.get("nli_ms", 0.0)) > 0:
        c["nli_gt0"] += 1


def rates(c):
    n = c["parsed"] or 1
    fp_rate = c["fp"] / n
    fn_rate = c["fn"] / n
    return {
        "accept_rate": c["accept"] / n,
        "reject_rate": c["reject"] / n,
        "uncertain_rate": c["uncertain"] / n,
        "stage2_rate": c["stage2_ran"] / n,
        "fp_accept_rate": fp_rate,
        "fn_reject_rate": fn_rate,
        "ok_rate_stage1": 1.0 - fp_rate - fn_rate,
    }


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_path", required=True)
    ap.add_argument("--out_md", default="reports/day12_quick_eval.md")
    args = ap.parse_args()

    if not os.path.exists(args.run_path):
        note = "missing run file"
        if "fever" in args.run_path.lower():
            note = "FEVER skipped: missing artifacts"
        out_lines = [
            "# Day12 Quick Eval",
            "",
            f"- run_path: `{args.run_path}`",
            f"- status: `{note}`",
            "",
        ]
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        return

    overall = new_counts()
    per_baseline = {}

    with open(args.run_path, "r", encoding="utf-8") as f:
        for line in f:
            overall["total"] += 1
            line = line.strip()
            if not line:
                overall["bad"] += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                overall["bad"] += 1
                continue
            update_counts(overall, row)
            baseline = (row.get("baseline") or {}).get("name") or "unknown"
            c = per_baseline.get(baseline)
            if c is None:
                c = new_counts()
                per_baseline[baseline] = c
            c["total"] += 1
            update_counts(c, row)

    overall_rates = rates(overall)

    lines = []
    lines.append("# Day12 Quick Eval")
    lines.append("")
    lines.append(f"- run_path: `{args.run_path}`")
    lines.append(f"- total_lines: `{overall['total']}`")
    lines.append(f"- parsed_rows: `{overall['parsed']}`")
    lines.append(f"- bad_lines: `{overall['bad']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| accept_rate | {format_float(overall_rates['accept_rate'])} |")
    lines.append(f"| reject_rate | {format_float(overall_rates['reject_rate'])} |")
    lines.append(f"| uncertain_rate | {format_float(overall_rates['uncertain_rate'])} |")
    lines.append(f"| stage2_rate | {format_float(overall_rates['stage2_rate'])} |")
    lines.append(f"| fp_accept_rate | {format_float(overall_rates['fp_accept_rate'])} |")
    lines.append(f"| fn_reject_rate | {format_float(overall_rates['fn_reject_rate'])} |")
    lines.append(f"| ok_rate_stage1 | {format_float(overall_rates['ok_rate_stage1'])} |")
    lines.append(f"| rerank_ms_gt0_count | {overall['rerank_gt0']} |")
    lines.append(f"| nli_ms_gt0_count | {overall['nli_gt0']} |")
    lines.append("")
    lines.append("## By baseline")
    lines.append("")
    lines.append("| baseline | total_lines | parsed_rows | bad_lines | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | rerank_ms_gt0_count | nli_ms_gt0_count |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for baseline in sorted(per_baseline.keys()):
        c = per_baseline[baseline]
        r = rates(c)
        lines.append("| {b} | {t} | {p} | {bad} | {a} | {rj} | {u} | {s2} | {fp} | {fn} | {ok} | {rg} | {ng} |".format(
            b=baseline,
            t=c["total"],
            p=c["parsed"],
            bad=c["bad"],
            a=format_float(r["accept_rate"]),
            rj=format_float(r["reject_rate"]),
            u=format_float(r["uncertain_rate"]),
            s2=format_float(r["stage2_rate"]),
            fp=format_float(r["fp_accept_rate"]),
            fn=format_float(r["fn_reject_rate"]),
            ok=format_float(r["ok_rate_stage1"]),
            rg=c["rerank_gt0"],
            ng=c["nli_gt0"],
        ))
    lines.append("")

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
