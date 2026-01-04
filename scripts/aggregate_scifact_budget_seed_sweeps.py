import argparse
import math
import statistics
from pathlib import Path


def parse_table(lines, start_idx):
    rows = []
    i = start_idx + 1
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    if i >= len(lines):
        return rows
    header = lines[i].strip()
    if "budget" not in header:
        return rows
    i += 2
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 12:
            i += 1
            continue
        rows.append(parts)
        i += 1
    return rows


def parse_report(path, model_name="router_v2_min_route_hgb"):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if line.strip() == f"## {model_name}":
            rows = parse_table(lines, i)
            break
    data = []
    for parts in rows:
        budget = float(parts[0])
        data.append({
            "budget": budget,
            "accept_rate": float(parts[1]),
            "reject_rate": float(parts[2]),
            "uncertain_rate": float(parts[3]),
            "stage2_route_rate": float(parts[4]),
            "stage2_ran_rate": float(parts[5]),
            "capped_rate": float(parts[6]),
            "fp_accept_rate": float(parts[7]),
            "fn_reject_rate": float(parts[8]),
            "ok_rate_stage1": float(parts[9]),
            "mean_ms": float(parts[10]),
            "p95_ms": float(parts[11]),
        })
    return data


def aggregate(values):
    if not values:
        return {"mean": None, "std": None, "worst": None}
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    worst = max(values)
    return {"mean": mean, "std": std, "worst": worst}


def find_out_path(path):
    out = Path(path)
    if not out.exists():
        return out
    stem = out.stem
    suffix = out.suffix
    parent = out.parent
    idx = 2
    while True:
        cand = parent / f"{stem}_v{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_md", default="reports/product_proof_router_v2_scifact_budget_robust_summary_v1.md")
    ap.add_argument("--model_name", default="router_v2_min_route_hgb")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"missing in_dir {in_dir}")
    out_md = find_out_path(args.out_md)

    reports = []
    for path in sorted(in_dir.glob("*.md")):
        reports.append(path)
    if not reports:
        raise SystemExit(f"no reports in {in_dir}")

    per_budget = {}
    for path in reports:
        rows = parse_report(path, model_name=args.model_name)
        if not rows:
            continue
        for row in rows:
            b = row["budget"]
            per_budget.setdefault(b, []).append(row)

    if not per_budget:
        raise SystemExit("no data rows found")

    lines = []
    lines.append("# SciFact Budget Sweep Robust Summary")
    lines.append("")
    lines.append(f"- in_dir: `{in_dir}`")
    lines.append(f"- model_name: `{args.model_name}`")
    lines.append("")
    lines.append("| budget | fp_mean | fp_std | fp_worst | fn_mean | fn_std | fn_worst | ran_mean | ran_std | ran_worst | cap_mean | cap_std | cap_worst | ok_mean | ok_std | ok_worst | p95_mean | p95_std | p95_worst |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    candidates = []
    for b in sorted(per_budget.keys()):
        rows = per_budget[b]
        fp = aggregate([r["fp_accept_rate"] for r in rows])
        fn = aggregate([r["fn_reject_rate"] for r in rows])
        ran = aggregate([r["stage2_ran_rate"] for r in rows])
        cap = aggregate([r["capped_rate"] for r in rows])
        ok = aggregate([r["ok_rate_stage1"] for r in rows])
        p95 = aggregate([r["p95_ms"] for r in rows])
        lines.append("| {b:.2f} | {fp_m:.4f} | {fp_s:.4f} | {fp_w:.4f} | {fn_m:.4f} | {fn_s:.4f} | {fn_w:.4f} | {r_m:.4f} | {r_s:.4f} | {r_w:.4f} | {c_m:.4f} | {c_s:.4f} | {c_w:.4f} | {ok_m:.4f} | {ok_s:.4f} | {ok_w:.4f} | {p_m:.2f} | {p_s:.2f} | {p_w:.2f} |".format(
            b=b,
            fp_m=fp["mean"], fp_s=fp["std"], fp_w=fp["worst"],
            fn_m=fn["mean"], fn_s=fn["std"], fn_w=fn["worst"],
            r_m=ran["mean"], r_s=ran["std"], r_w=ran["worst"],
            c_m=cap["mean"], c_s=cap["std"], c_w=cap["worst"],
            ok_m=ok["mean"], ok_s=ok["std"], ok_w=ok["worst"],
            p_m=p95["mean"], p_s=p95["std"], p_w=p95["worst"],
        ))
        feasible = fp["worst"] <= 0.01 and fn["worst"] <= 0.01 and ran["worst"] <= 0.25 and cap["worst"] <= 0.20
        if feasible:
            candidates.append(( -ok["mean"], ran["mean"], p95["mean"], b, ok, ran, cap, fp, fn ))
    lines.append("")
    lines.append("Ship recommendation")
    lines.append("")
    if candidates:
        candidates.sort()
        _, _, _, b, ok, ran, cap, fp, fn = candidates[0]
        lines.append(f"- budget={b:.2f} ok_mean={ok['mean']:.4f} fp_worst={fp['worst']:.4f} fn_worst={fn['worst']:.4f} ran_worst={ran['worst']:.4f} cap_worst={cap['worst']:.4f}")
    else:
        lines.append("- no budget meets worst-case constraints fp<=0.01 fn<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
