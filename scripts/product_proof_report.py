import argparse
import json
from pathlib import Path


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def summarize(path):
    n = 0
    accept = reject = uncertain = 0
    fp = fn = 0
    stage2 = 0
    total_ms = []
    for row in iter_jsonl(path):
        n += 1
        stage1 = row.get("stage1", {}) or {}
        decision = stage1.get("route_decision")
        if decision == "ACCEPT":
            accept += 1
        elif decision == "REJECT":
            reject += 1
        else:
            uncertain += 1
        y = (row.get("ground_truth") or {}).get("y")
        if decision == "ACCEPT" and y == 0:
            fp += 1
        if decision == "REJECT" and y == 1:
            fn += 1
        if (row.get("stage2") or {}).get("ran") is True:
            stage2 += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
    n = n or 1
    return {
        "accept_rate": accept / n,
        "reject_rate": reject / n,
        "uncertain_rate": uncertain / n,
        "fp_accept_rate": fp / n,
        "fn_reject_rate": fn / n,
        "ok_rate": 1.0 - (fp / n) - (fn / n),
        "stage2_rate": stage2 / n,
        "mean_ms": sum(total_ms) / n if total_ms else 0.0,
    }


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--tuned_jsonl", required=True)
    ap.add_argument("--matrix_json", required=True)
    ap.add_argument("--out_md", default="reports/product_proof.md")
    ap.add_argument("--max_fp_accept", type=float, default=0.02)
    args = ap.parse_args()

    base = summarize(Path(args.baseline_jsonl))
    tuned = summarize(Path(args.tuned_jsonl))
    matrix = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))

    best = None
    best_name = None
    for name, m in matrix.items():
        if m.get("fp_accept_rate") is None:
            continue
        if m["fp_accept_rate"] <= args.max_fp_accept:
            if best is None or m["uncertain_rate"] < best["uncertain_rate"]:
                best = m
                best_name = name

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Product Proof")
    lines.append("")
    lines.append(f"- track: `{args.track}`")
    lines.append("")
    lines.append("| variant | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | stage2_rate | mean_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append("| baseline | {a} | {r} | {u} | {fp} | {fn} | {ok} | {s2} | {m} |".format(
        a=format_float(base["accept_rate"]),
        r=format_float(base["reject_rate"]),
        u=format_float(base["uncertain_rate"]),
        fp=format_float(base["fp_accept_rate"]),
        fn=format_float(base["fn_reject_rate"]),
        ok=format_float(base["ok_rate"]),
        s2=format_float(base["stage2_rate"]),
        m=f"{base['mean_ms']:.2f}",
    ))
    lines.append("| tuned | {a} | {r} | {u} | {fp} | {fn} | {ok} | {s2} | {m} |".format(
        a=format_float(tuned["accept_rate"]),
        r=format_float(tuned["reject_rate"]),
        u=format_float(tuned["uncertain_rate"]),
        fp=format_float(tuned["fp_accept_rate"]),
        fn=format_float(tuned["fn_reject_rate"]),
        ok=format_float(tuned["ok_rate"]),
        s2=format_float(tuned["stage2_rate"]),
        m=f"{tuned['mean_ms']:.2f}",
    ))
    lines.append("")
    if best is not None:
        lines.append(f"- best_mode_under_fp_{args.max_fp_accept}: `{best_name}`")
        lines.append(f"- best_uncertain_rate: `{format_float(best['uncertain_rate'])}`")
        lines.append(f"- best_stage2_rate: `{format_float(best['stage2_rate'])}`")
        lines.append(f"- best_mean_ms: `{best['mean_ms']:.2f}`")
    else:
        lines.append("- best_mode_under_fp: `none`")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
