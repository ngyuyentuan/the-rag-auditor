import argparse
import json
from pathlib import Path

import yaml


def summarize(path: Path):
    counts = {"parsed": 0, "bad": 0, "accept": 0, "reject": 0, "uncertain": 0, "fp": 0, "fn": 0}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                counts["bad"] += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                counts["bad"] += 1
                continue
            counts["parsed"] += 1
            stage1 = row.get("stage1", {}) or {}
            decision = stage1.get("route_decision")
            if decision == "ACCEPT":
                counts["accept"] += 1
            elif decision == "REJECT":
                counts["reject"] += 1
            else:
                counts["uncertain"] += 1
            y = (row.get("ground_truth") or {}).get("y")
            if decision == "ACCEPT" and y == 0:
                counts["fp"] += 1
            if decision == "REJECT" and y == 1:
                counts["fn"] += 1
    return counts


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


def load_prod(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "t_lower": float(data.get("t_lower")),
        "t_upper": float(data.get("t_upper")),
        "tau": float(data.get("tau")),
        "mode": data.get("mode"),
        "max_fp_accept": data.get("max_fp_accept"),
        "max_fn_reject": data.get("max_fn_reject"),
        "max_uncertain": data.get("max_uncertain"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--prod_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    baseline_path = Path(args.baseline_jsonl)
    prod_path = Path(args.prod_jsonl)
    prod_yaml = Path(f"configs/thresholds_stage1_prod_{args.track}.yaml")
    prod_meta = load_prod(prod_yaml) if prod_yaml.exists() else {}

    base_counts = summarize(baseline_path)
    prod_counts = summarize(prod_path)
    base_rates = rates(base_counts)
    prod_rates = rates(prod_counts)

    max_fp = prod_meta.get("max_fp_accept")
    max_fn = prod_meta.get("max_fn_reject")
    max_unc = prod_meta.get("max_uncertain")

    gate_fp = (max_fp is None) or (prod_rates["fp_accept_rate"] <= float(max_fp))
    gate_fn = (max_fn is None) or (prod_rates["fn_reject_rate"] <= float(max_fn))
    gate_unc = (max_unc is None) or (prod_rates["uncertain_rate"] <= float(max_unc))

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Stage1 Release Gate ({args.track})")
    lines.append("")
    lines.append(f"- baseline_jsonl: `{baseline_path}`")
    lines.append(f"- prod_jsonl: `{prod_path}`")
    lines.append(f"- prod_yaml: `{prod_yaml if prod_yaml.exists() else 'missing'}`")
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
    lines.append("| prod | {a} | {r} | {u} | {fp} | {fn} | {ok} |".format(
        a=format_float(prod_rates["accept_rate"]),
        r=format_float(prod_rates["reject_rate"]),
        u=format_float(prod_rates["uncertain_rate"]),
        fp=format_float(prod_rates["fp_accept_rate"]),
        fn=format_float(prod_rates["fn_reject_rate"]),
        ok=format_float(prod_rates["ok_rate"]),
    ))
    lines.append("")
    lines.append("Release gate")
    lines.append("")
    lines.append(f"- fp_accept_rate <= max_fp_accept: `{gate_fp}`")
    lines.append(f"- fn_reject_rate <= max_fn_reject: `{gate_fn}`")
    lines.append(f"- uncertain_rate <= max_uncertain: `{gate_unc}`")
    lines.append("")
    lines.append("If fp_accept_rate fails, raise t_upper or tighten max_fp_accept. If fn_reject_rate fails, lower t_lower or switch to accept_only. If uncertain_rate fails, relax max_uncertain or lower t_upper.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
