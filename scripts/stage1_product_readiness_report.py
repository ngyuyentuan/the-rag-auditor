import argparse
import json
from pathlib import Path


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


def summarize(path: Path):
    c = new_counts()
    if not path.exists():
        return c
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            c["total"] += 1
            line = line.strip()
            if not line:
                c["bad"] += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                c["bad"] += 1
                continue
            update_counts(c, row)
    return c


def rates(c):
    n = c["parsed"] or 1
    fp = c["fp"] / n
    fn = c["fn"] / n
    return {
        "accept_rate": c["accept"] / n,
        "reject_rate": c["reject"] / n,
        "uncertain_rate": c["uncertain"] / n,
        "fp_accept_rate": fp,
        "fn_reject_rate": fn,
        "ok_rate": 1.0 - fp - fn,
    }


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--tuned_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    baseline_path = Path(args.baseline_jsonl)
    tuned_path = Path(args.tuned_jsonl)
    base_counts = summarize(baseline_path)
    tuned_counts = summarize(tuned_path)

    base_rates = rates(base_counts)
    tuned_rates = rates(tuned_counts)

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Product Readiness ({args.track})")
    lines.append("")
    lines.append(f"- baseline_jsonl: `{baseline_path}`")
    lines.append(f"- tuned_jsonl: `{tuned_path}`")
    lines.append("")
    lines.append("| variant | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append("| baseline | {a} | {r} | {u} | {fp} | {fn} | {ok} |".format(
        a=format_float(base_rates["accept_rate"]),
        r=format_float(base_rates["reject_rate"]),
        u=format_float(base_rates["uncertain_rate"]),
        fp=format_float(base_rates["fp_accept_rate"]),
        fn=format_float(base_rates["fn_reject_rate"]),
        ok=format_float(base_rates["ok_rate"]),
    ))
    lines.append("| tuned | {a} | {r} | {u} | {fp} | {fn} | {ok} |".format(
        a=format_float(tuned_rates["accept_rate"]),
        r=format_float(tuned_rates["reject_rate"]),
        u=format_float(tuned_rates["uncertain_rate"]),
        fp=format_float(tuned_rates["fp_accept_rate"]),
        fn=format_float(tuned_rates["fn_reject_rate"]),
        ok=format_float(tuned_rates["ok_rate"]),
    ))
    lines.append("")
    fp_delta = tuned_rates["fp_accept_rate"] - base_rates["fp_accept_rate"]
    unc_delta = tuned_rates["uncertain_rate"] - base_rates["uncertain_rate"]
    lines.append("Conclusion")
    lines.append("")
    lines.append("Tuned fp_accept_rate {fp_trend} by {fp_delta}. Uncertain_rate {unc_trend} by {unc_delta}.".format(
        fp_trend="decreased" if fp_delta < 0 else "increased" if fp_delta > 0 else "unchanged",
        fp_delta=format_float(abs(fp_delta)),
        unc_trend="decreased" if unc_delta < 0 else "increased" if unc_delta > 0 else "unchanged",
        unc_delta=format_float(abs(unc_delta)),
    ))
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
