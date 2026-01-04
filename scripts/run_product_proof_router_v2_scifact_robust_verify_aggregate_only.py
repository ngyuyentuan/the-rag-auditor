import argparse
import json
import re
import secrets
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_new_dir(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def unique_run_id():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"robust_verify_aggregate_{stamp}_{suffix}"


def write_text_create(path, text):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"temp path exists: {tmp}")
    with tmp.open("x", encoding="utf-8") as f_out:
        f_out.write(text)
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    tmp.rename(path)
    return path


def parse_seed_budget(name):
    seed = None
    budget = None
    m = re.search(r"seed(\d+)", name)
    if m:
        seed = int(m.group(1))
    m = re.search(r"budget([0-9.]+)", name)
    if m:
        try:
            budget = float(m.group(1))
        except ValueError:
            budget = None
    return seed, budget


def iter_jsonl_safe(path):
    parsed = 0
    bad = 0
    with Path(path).open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parsed += 1
                yield obj
            except Exception:
                bad += 1
                return
    if parsed == 0 or bad > 0:
        return


def get_decision(row):
    stage1 = row.get("stage1") or {}
    return stage1.get("route_decision") or stage1.get("router_decision")


def get_y(row):
    gt = row.get("ground_truth") or {}
    return gt.get("y")


def stage2_route_requested(row):
    stage2 = row.get("stage2") or {}
    if "route_requested" in stage2:
        return bool(stage2.get("route_requested"))
    decision = get_decision(row)
    return decision == "UNCERTAIN"


def stage2_ran(row):
    stage2 = row.get("stage2") or {}
    return bool(stage2.get("ran"))


def stage2_capped(row):
    stage2 = row.get("stage2") or {}
    if stage2.get("capped") is True:
        return True
    if stage2.get("capped_reason") == "budget_cap":
        return True
    rerank = stage2.get("rerank") or {}
    nli = stage2.get("nli") or {}
    return rerank.get("reason") == "budget_cap" or nli.get("reason") == "budget_cap"


def timing_total_ms(row):
    timing = row.get("timing_ms") or {}
    val = timing.get("total_ms")
    try:
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def summarize_rows(rows):
    total = 0
    accept = 0
    reject = 0
    uncertain = 0
    fp = 0
    fn = 0
    route = 0
    ran = 0
    capped = 0
    timings = []
    for row in rows:
        total += 1
        decision = get_decision(row)
        y = get_y(row)
        if decision == "ACCEPT":
            accept += 1
            if y == 0:
                fp += 1
        elif decision == "REJECT":
            reject += 1
            if y == 1:
                fn += 1
        else:
            uncertain += 1
        if stage2_route_requested(row):
            route += 1
        if stage2_ran(row):
            ran += 1
        if stage2_capped(row):
            capped += 1
        timings.append(timing_total_ms(row))
    if total == 0:
        return None
    timings_sorted = sorted(timings)
    p95 = timings_sorted[int(0.95 * (len(timings_sorted) - 1))] if timings_sorted else 0.0
    mean_ms = statistics.mean(timings_sorted) if timings_sorted else 0.0
    return {
        "n": total,
        "accept_rate": accept / total,
        "reject_rate": reject / total,
        "uncertain_rate": uncertain / total,
        "stage2_route_rate": route / total,
        "stage2_ran_rate": ran / total,
        "capped_rate": capped / total,
        "fp_accept_rate": fp / total,
        "fn_reject_rate": fn / total,
        "ok_rate_stage1": 1.0 - (fp / total) - (fn / total),
        "mean_ms": mean_ms,
        "p95_ms": p95,
    }


def aggregate_budget_rows(rows):
    grouped = {}
    for r in rows:
        if r["budget"] is None:
            continue
        grouped.setdefault(r["budget"], []).append(r)
    out = {}
    for budget, items in grouped.items():
        n = len(items) or 1
        def mean(key):
            return sum(x[key] for x in items) / n
        def worst(key):
            return max(x[key] for x in items)
        out[budget] = {
            "budget": budget,
            "count": len(items),
            "ok_mean": mean("ok_rate_stage1"),
            "fp_worst": worst("fp_accept_rate"),
            "fn_worst": worst("fn_reject_rate"),
            "ran_worst": worst("stage2_ran_rate"),
            "cap_worst": worst("capped_rate"),
            "p95_worst": worst("p95_ms"),
        }
    return out


def select_ship(agg, fp_max, fn_max, ran_max, cap_max):
    best = None
    for budget, row in agg.items():
        feasible = row["fp_worst"] <= fp_max and row["fn_worst"] <= fn_max and row["ran_worst"] <= ran_max and row["cap_worst"] <= cap_max
        if not feasible:
            continue
        cand = (-row["ok_mean"], row["ran_worst"], row["p95_worst"], budget)
        if best is None or cand < best:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dirs", nargs="+", default=["runs/product_proof_router_v2", "runs/sweeps"])
    ap.add_argument("--glob", default="scifact*budget*.jsonl")
    ap.add_argument("--out_md")
    ap.add_argument("--constraints_fp", type=float, default=0.01)
    ap.add_argument("--constraints_fn", type=float, default=0.01)
    ap.add_argument("--constraints_ran", type=float, default=0.25)
    ap.add_argument("--constraints_cap", type=float, default=0.20)
    args = ap.parse_args()

    run_id = unique_run_id()
    models_dir = ensure_new_dir(Path("models") / "sweeps" / run_id)
    runs_dir = ensure_new_dir(Path("runs") / "sweeps" / run_id)
    reports_dir = ensure_new_dir(Path("reports") / "sweeps" / run_id)

    files = []
    for d in args.in_dirs:
        root = Path(d)
        if not root.exists():
            continue
        files.extend(root.rglob(args.glob))
    files = sorted({p for p in files})

    rows = []
    skipped = []
    for path in files:
        parsed = list(iter_jsonl_safe(path))
        if not parsed:
            skipped.append(str(path))
            continue
        stats = summarize_rows(parsed)
        if not stats:
            skipped.append(str(path))
            continue
        seed, budget = parse_seed_budget(path.name)
        stats["seed"] = seed
        stats["budget"] = budget
        stats["file"] = path.name
        rows.append(stats)

    agg = aggregate_budget_rows(rows)
    best = select_ship(agg, args.constraints_fp, args.constraints_fn, args.constraints_ran, args.constraints_cap)

    lines = []
    lines.append("# Product Proof Router v2 SciFact Robust Verify (aggregate-only)")
    lines.append("")
    lines.append(f"- in_dirs: `{', '.join(args.in_dirs)}`")
    lines.append(f"- glob: `{args.glob}`")
    lines.append(f"- files_found: {len(files)}")
    lines.append(f"- files_skipped: {len(skipped)}")
    if skipped:
        lines.append(f"- skipped: {', '.join(skipped[:10])}")
    lines.append("")
    lines.append("## Per-file metrics")
    lines.append("")
    lines.append("| file | seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append("| {file} | {seed} | {b} | {a:.4f} | {rj:.4f} | {u:.4f} | {route:.4f} | {ran:.4f} | {cap:.4f} | {fp:.4f} | {fn:.4f} | {ok:.4f} | {mean:.2f} | {p95:.2f} |".format(
            file=r["file"],
            seed=r["seed"] if r["seed"] is not None else "unknown",
            b=f"{r['budget']:.3f}" if r["budget"] is not None else "unknown",
            a=r["accept_rate"],
            rj=r["reject_rate"],
            u=r["uncertain_rate"],
            route=r["stage2_route_rate"],
            ran=r["stage2_ran_rate"],
            cap=r["capped_rate"],
            fp=r["fp_accept_rate"],
            fn=r["fn_reject_rate"],
            ok=r["ok_rate_stage1"],
            mean=r["mean_ms"],
            p95=r["p95_ms"],
        ))
    lines.append("")
    lines.append("## Budget summary")
    lines.append("")
    lines.append("| budget | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for budget in sorted(agg.keys()):
        row = agg[budget]
        lines.append("| {b:.3f} | {ok:.4f} | {fp:.4f} | {fn:.4f} | {ran:.4f} | {cap:.4f} | {p95:.2f} |".format(
            b=row["budget"],
            ok=row["ok_mean"],
            fp=row["fp_worst"],
            fn=row["fn_worst"],
            ran=row["ran_worst"],
            cap=row["cap_worst"],
            p95=row["p95_worst"],
        ))
    lines.append("")
    lines.append("## Ship recommendation")
    lines.append("")
    if best is None:
        lines.append("- no budget met fp_worst<=0.01 fn_worst<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20 on worst-case")
        penalties = []
        for budget, row in agg.items():
            penalty = max(0.0, row["fp_worst"] - args.constraints_fp) + max(0.0, row["fn_worst"] - args.constraints_fn) + max(0.0, row["ran_worst"] - args.constraints_ran) + max(0.0, row["cap_worst"] - args.constraints_cap)
            penalties.append((penalty, budget))
        if penalties:
            penalties.sort()
            _, budget = penalties[0]
            row = agg[budget]
            lines.append("- closest: budget={b:.3f} fp_worst={fp:.4f} fn_worst={fn:.4f} ran_worst={ran:.4f} cap_worst={cap:.4f} ok_mean={ok:.4f}".format(
                b=row["budget"],
                fp=row["fp_worst"],
                fn=row["fn_worst"],
                ran=row["ran_worst"],
                cap=row["cap_worst"],
                ok=row["ok_mean"],
            ))
    else:
        _, _, _, budget = best
        row = agg[budget]
        lines.append("- budget={b:.3f} fp_worst={fp:.4f} fn_worst={fn:.4f} ran_worst={ran:.4f} cap_worst={cap:.4f} ok_mean={ok:.4f}".format(
            b=row["budget"],
            fp=row["fp_worst"],
            fn=row["fn_worst"],
            ran=row["ran_worst"],
            cap=row["cap_worst"],
            ok=row["ok_mean"],
        ))

    report = Path(args.out_md) if args.out_md else (reports_dir / "product_proof_router_v2_scifact_robust_verify.md")
    if report.exists():
        raise SystemExit(f"output path exists: {report}")
    if report.parent != reports_dir:
        raise SystemExit("out_md must be under reports/sweeps/<run_id>/")
    write_text_create(report, "\n".join(lines))
    print(f"run_id={run_id}")
    print(f"models_dir={models_dir}")
    print(f"runs_dir={runs_dir}")
    print(f"reports_dir={reports_dir}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
