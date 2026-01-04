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


def parse_tag(name):
    base = name
    for token in ["_seed", "_budget", ".jsonl"]:
        if token in base:
            base = base.split(token)[0]
    return base


def iter_jsonl_strict(path, strict_schema, skip_partial):
    parsed = 0
    bad = 0
    rows = []
    with Path(path).open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parsed += 1
                if strict_schema:
                    if not isinstance(obj.get("stage1"), dict):
                        return None, "missing_stage1"
                    if not isinstance(obj.get("ground_truth"), dict):
                        return None, "missing_ground_truth"
                    if not isinstance(obj.get("timing_ms"), dict):
                        return None, "missing_timing_ms"
                    if not isinstance(obj.get("stage2"), dict):
                        return None, "missing_stage2"
                rows.append(obj)
            except Exception:
                bad += 1
                if skip_partial:
                    return None, "partial_jsonl"
    if parsed == 0:
        return None, "empty"
    if bad > 0 and skip_partial:
        return None, "partial_jsonl"
    return rows, None


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


def group_rows(rows, group_by):
    grouped = {}
    keys = [k.strip() for k in group_by.split(",") if k.strip()]
    for r in rows:
        key_parts = []
        for k in keys:
            key_parts.append(r.get(k))
        key = tuple(key_parts)
        grouped.setdefault(key, []).append(r)
    return grouped


def aggregate_group(rows):
    n = len(rows) or 1
    def mean(key):
        return sum(x[key] for x in rows) / n
    def worst(key):
        return max(rows, key=lambda x: x[key])
    return {
        "count": len(rows),
        "ok_mean": mean("ok_rate_stage1"),
        "fp_worst": worst("fp_accept_rate"),
        "fn_worst": worst("fn_reject_rate"),
        "ran_worst": worst("stage2_ran_rate"),
        "cap_worst": worst("capped_rate"),
        "p95_worst": worst("p95_ms"),
    }


def select_ship(agg_map, fp_max, fn_max, ran_max, cap_max):
    best = None
    for key, agg in agg_map.items():
        fp = agg["fp_worst"]["fp_accept_rate"]
        fn = agg["fn_worst"]["fn_reject_rate"]
        ran = agg["ran_worst"]["stage2_ran_rate"]
        cap = agg["cap_worst"]["capped_rate"]
        if fp > fp_max or fn > fn_max or ran > ran_max or cap > cap_max:
            continue
        ok = agg["ok_mean"]
        p95 = agg["p95_worst"]["p95_ms"]
        cand = (-ok, ran, p95, key)
        if best is None or cand < best:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dirs", nargs="+", default=["runs/product_proof_router_v2", "runs/sweeps"])
    ap.add_argument("--glob", default="scifact*budget*.jsonl")
    ap.add_argument("--include_tag")
    ap.add_argument("--group_by", default="tag,budget")
    ap.add_argument("--strict_schema", default="true")
    ap.add_argument("--skip_partial", default="true")
    ap.add_argument("--out_md")
    ap.add_argument("--constraints_fp", type=float, default=0.01)
    ap.add_argument("--constraints_fn", type=float, default=0.01)
    ap.add_argument("--constraints_ran", type=float, default=0.25)
    ap.add_argument("--constraints_cap", type=float, default=0.20)
    args = ap.parse_args()

    strict_schema = str(args.strict_schema).lower() == "true"
    skip_partial = str(args.skip_partial).lower() == "true"

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
        tag = parse_tag(path.name)
        if args.include_tag and args.include_tag not in tag:
            continue
        parsed, reason = iter_jsonl_strict(path, strict_schema, skip_partial)
        if parsed is None:
            skipped.append((str(path), reason))
            continue
        stats = summarize_rows(parsed)
        if not stats:
            skipped.append((str(path), "empty"))
            continue
        seed, budget = parse_seed_budget(path.name)
        stats["seed"] = seed
        stats["budget"] = budget
        stats["file"] = path.name
        stats["tag"] = tag
        rows.append(stats)

    grouped = group_rows(rows, args.group_by)
    agg_map = {}
    for key, items in grouped.items():
        agg_map[key] = aggregate_group(items)

    best = select_ship(agg_map, args.constraints_fp, args.constraints_fn, args.constraints_ran, args.constraints_cap)

    lines = []
    lines.append("# Product Proof Router v2 SciFact Robust Verify (aggregate-filtered)")
    lines.append("")
    lines.append(f"- in_dirs: `{', '.join(args.in_dirs)}`")
    lines.append(f"- glob: `{args.glob}`")
    lines.append(f"- include_tag: `{args.include_tag}`")
    lines.append(f"- group_by: `{args.group_by}`")
    lines.append(f"- strict_schema: {str(strict_schema).lower()}")
    lines.append(f"- skip_partial: {str(skip_partial).lower()}")
    lines.append(f"- files_found: {len(files)}")
    lines.append(f"- files_used: {len(rows)}")
    lines.append(f"- files_skipped: {len(skipped)}")
    if skipped:
        lines.append(f"- skipped: {', '.join([f'{p}:{r}' for p,r in skipped[:10]])}")
    lines.append("")
    lines.append("## Per-file metrics")
    lines.append("")
    lines.append("| file | tag | seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append("| {file} | {tag} | {seed} | {b} | {a:.4f} | {rj:.4f} | {u:.4f} | {route:.4f} | {ran:.4f} | {cap:.4f} | {fp:.4f} | {fn:.4f} | {ok:.4f} | {mean:.2f} | {p95:.2f} |".format(
            file=r["file"],
            tag=r["tag"],
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
    lines.append("## Group summary")
    lines.append("")
    lines.append("| group | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst | worst_fp_file | worst_fn_file | worst_ran_file | worst_cap_file | worst_p95_file |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for key in sorted(agg_map.keys(), key=lambda k: str(k)):
        agg = agg_map[key]
        group = ",".join([str(k) for k in key])
        lines.append("| {group} | {ok:.4f} | {fp:.4f} | {fn:.4f} | {ran:.4f} | {cap:.4f} | {p95:.2f} | {fp_file} | {fn_file} | {ran_file} | {cap_file} | {p95_file} |".format(
            group=group,
            ok=agg["ok_mean"],
            fp=agg["fp_worst"]["fp_accept_rate"],
            fn=agg["fn_worst"]["fn_reject_rate"],
            ran=agg["ran_worst"]["stage2_ran_rate"],
            cap=agg["cap_worst"]["capped_rate"],
            p95=agg["p95_worst"]["p95_ms"],
            fp_file=agg["fp_worst"]["file"],
            fn_file=agg["fn_worst"]["file"],
            ran_file=agg["ran_worst"]["file"],
            cap_file=agg["cap_worst"]["file"],
            p95_file=agg["p95_worst"]["file"],
        ))
    lines.append("")
    lines.append("## Ship recommendation")
    lines.append("")
    if best is None:
        lines.append("- no group met fp_worst<=0.01 fn_worst<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20 on worst-case")
        penalties = []
        for key, agg in agg_map.items():
            fp = agg["fp_worst"]["fp_accept_rate"]
            fn = agg["fn_worst"]["fn_reject_rate"]
            ran = agg["ran_worst"]["stage2_ran_rate"]
            cap = agg["cap_worst"]["capped_rate"]
            penalty = max(0.0, fp - args.constraints_fp) + max(0.0, fn - args.constraints_fn) + max(0.0, ran - args.constraints_ran) + max(0.0, cap - args.constraints_cap)
            penalties.append((penalty, key))
        if penalties:
            penalties.sort()
            _, key = penalties[0]
            agg = agg_map[key]
            group = ",".join([str(k) for k in key])
            lines.append("- closest: group={g} fp_worst={fp:.4f} fn_worst={fn:.4f} ran_worst={ran:.4f} cap_worst={cap:.4f} ok_mean={ok:.4f}".format(
                g=group,
                fp=agg["fp_worst"]["fp_accept_rate"],
                fn=agg["fn_worst"]["fn_reject_rate"],
                ran=agg["ran_worst"]["stage2_ran_rate"],
                cap=agg["cap_worst"]["capped_rate"],
                ok=agg["ok_mean"],
            ))
            lines.append("- worst_files: fp={fp} fn={fn} ran={ran} cap={cap} p95={p95}".format(
                fp=agg["fp_worst"]["file"],
                fn=agg["fn_worst"]["file"],
                ran=agg["ran_worst"]["file"],
                cap=agg["cap_worst"]["file"],
                p95=agg["p95_worst"]["file"],
            ))
    else:
        _, _, _, key = best
        agg = agg_map[key]
        group = ",".join([str(k) for k in key])
        lines.append("- group={g} fp_worst={fp:.4f} fn_worst={fn:.4f} ran_worst={ran:.4f} cap_worst={cap:.4f} ok_mean={ok:.4f}".format(
            g=group,
            fp=agg["fp_worst"]["fp_accept_rate"],
            fn=agg["fn_worst"]["fn_reject_rate"],
            ran=agg["ran_worst"]["stage2_ran_rate"],
            cap=agg["cap_worst"]["capped_rate"],
            ok=agg["ok_mean"],
        ))
        lines.append("- worst_files: fp={fp} fn={fn} ran={ran} cap={cap} p95={p95}".format(
            fp=agg["fp_worst"]["file"],
            fn=agg["fn_worst"]["file"],
            ran=agg["ran_worst"]["file"],
            cap=agg["cap_worst"]["file"],
            p95=agg["p95_worst"]["file"],
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
