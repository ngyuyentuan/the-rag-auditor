import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dirs", nargs="+", default=["runs"])
    p.add_argument("--glob", default="*.jsonl")
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--include_tag")
    p.add_argument("--out_md")
    p.add_argument("--debug", action="store_true", default=False)
    return p.parse_args()


def run_id_now():
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    entropy = os.urandom(3).hex()
    return f"robust_verify_aggregate_{stamp}_{entropy}"


def ensure_new_dir(path: Path):
    if path.exists():
        raise FileExistsError(f"output path exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def safe_write_md(path: Path, content: str):
    if path.exists():
        raise FileExistsError(f"output file exists: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"temp file exists: {tmp}")
    with open(tmp, "x", encoding="utf-8") as f:
        f.write(content)
    os.rename(tmp, path)


def discover_files(in_dirs, glob_pat, recursive):
    paths = []
    for d in in_dirs:
        root = Path(d)
        if not root.exists():
            continue
        if recursive:
            paths.extend(root.rglob(glob_pat))
        else:
            paths.extend(root.glob(glob_pat))
    dedup = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        dedup.append(p)
    return dedup


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield s


def detect_track(path: Path):
    if "scifact" in str(path).lower():
        return "scifact"
    for s in iter_jsonl(path):
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            return None
        meta = row.get("metadata") or {}
        tr = meta.get("track")
        if isinstance(tr, str):
            return tr.lower()
        return None
    return None


TAG_RE = re.compile(r"(d\d+_lr[\d.]+_leaf\d+_l[\d.]+)")
BUDGET_RE = re.compile(r"budget[_-]?([01](?:\\.[0-9]+)?)|\\bb([0-9]*\\.[0-9]+)\\b")
SEED_RE = re.compile(r"seed[_-]?([0-9]+)")


def extract_tag(path: Path, rows):
    for row in rows:
        meta = row.get("metadata") or {}
        tag = meta.get("tag")
        if isinstance(tag, str) and tag.strip():
            return tag
    m = TAG_RE.search(str(path))
    if m:
        return m.group(1)
    return "unknown"


def extract_budget(path: Path, rows):
    for row in rows:
        meta = row.get("metadata") or {}
        b = meta.get("stage2_budget", meta.get("budget"))
        if isinstance(b, (int, float)) and 0.0 < float(b) <= 1.0:
            return float(b)
        stage2 = row.get("stage2") or {}
        b2 = stage2.get("cap_budget", stage2.get("budget"))
        if isinstance(b2, (int, float)) and 0.0 < float(b2) <= 1.0:
            return float(b2)
    m = BUDGET_RE.search(str(path))
    if m:
        val = m.group(1) or m.group(2)
        try:
            b = float(val)
            if 0.0 < b <= 1.0:
                return b
            return None
        except ValueError:
            return None
    return None


def extract_seed(path: Path, rows):
    for row in rows:
        meta = row.get("metadata") or {}
        s = meta.get("seed")
        if isinstance(s, int):
            return int(s)
    m = SEED_RE.search(str(path))
    if m:
        return int(m.group(1))
    return None


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, math.ceil(q * len(xs)) - 1))
    return xs[k]


def compute_metrics(rows):
    total = len(rows)
    if total == 0:
        return None
    accept = reject = uncertain = 0
    fp = fn = 0
    ran = route = capped = 0
    total_ms = []
    for row in rows:
        stage1 = row.get("stage1") or {}
        decision = stage1.get("route_decision") or stage1.get("router_decision")
        if decision == "ACCEPT":
            accept += 1
        elif decision == "REJECT":
            reject += 1
        else:
            uncertain += 1
        gt = row.get("ground_truth") or {}
        y = gt.get("y")
        if decision == "ACCEPT" and y == 0:
            fp += 1
        if decision == "REJECT" and y == 1:
            fn += 1
        stage2 = row.get("stage2") or {}
        if stage2.get("route_requested") is True:
            route += 1
        elif decision == "UNCERTAIN":
            route += 1
        if stage2.get("ran") is True:
            ran += 1
        if stage2.get("capped") is True:
            capped += 1
        timing = row.get("timing_ms") or {}
        tm = timing.get("total_ms")
        if isinstance(tm, (int, float)):
            total_ms.append(float(tm))
    accept_rate = accept / total
    reject_rate = reject / total
    uncertain_rate = uncertain / total
    fp_rate = fp / total
    fn_rate = fn / total
    ok_rate = 1 - fp_rate - fn_rate
    route_rate = route / total
    ran_rate = ran / total
    capped_rate = capped / total
    mean_ms = mean(total_ms) if total_ms else 0.0
    p95_ms = percentile(total_ms, 0.95) or 0.0
    return {
        "n": total,
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "fp_accept_rate": fp_rate,
        "fn_reject_rate": fn_rate,
        "ok_rate_stage1": ok_rate,
        "stage2_route_rate": route_rate,
        "stage2_ran_rate": ran_rate,
        "capped_rate": capped_rate,
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
    }


def read_jsonl_rows(path: Path, strict_schema, skip_partial):
    rows = []
    for s in iter_jsonl(path):
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            if skip_partial:
                return None, "bad_json"
            continue
        if strict_schema:
            if not isinstance(row, dict):
                return None, "bad_schema"
            if "stage1" not in row or "stage2" not in row or "timing_ms" not in row or "ground_truth" not in row:
                return None, "missing_keys"
        rows.append(row)
    return rows, None


def summarize_groups(groups, constraints):
    summary = []
    for (tag, budget), items in groups.items():
        ok_mean = mean([i["ok_rate_stage1"] for i in items])
        fp_worst = max(i["fp_accept_rate"] for i in items)
        fn_worst = max(i["fn_reject_rate"] for i in items)
        ran_worst = max(i["stage2_ran_rate"] for i in items)
        cap_worst = max(i["capped_rate"] for i in items)
        p95_worst = max(i["p95_ms"] for i in items)
        def worst_files(key):
            v = max(i[key] for i in items)
            return [i["file"] for i in items if i[key] == v]
        summary.append({
            "tag": tag,
            "budget": budget,
            "ok_mean": ok_mean,
            "fp_worst": fp_worst,
            "fn_worst": fn_worst,
            "ran_worst": ran_worst,
            "cap_worst": cap_worst,
            "p95_worst": p95_worst,
            "n_files": len(items),
            "worst_files": {
                "fp": worst_files("fp_accept_rate"),
                "fn": worst_files("fn_reject_rate"),
                "ran": worst_files("stage2_ran_rate"),
                "cap": worst_files("capped_rate"),
                "p95": worst_files("p95_ms"),
            },
        })
    def penalty(row):
        return (
            max(0.0, row["fp_worst"] - constraints["fp"]) +
            max(0.0, row["fn_worst"] - constraints["fn"]) +
            max(0.0, row["ran_worst"] - constraints["ran"]) +
            max(0.0, row["cap_worst"] - constraints["cap"])
        )
    feasible = [r for r in summary if (
        r["fp_worst"] <= constraints["fp"] and
        r["fn_worst"] <= constraints["fn"] and
        r["ran_worst"] <= constraints["ran"] and
        r["cap_worst"] <= constraints["cap"]
    )]
    if feasible:
        feasible.sort(key=lambda r: (-r["ok_mean"], r["ran_worst"], r["p95_worst"]))
        pick = feasible[0]
        return summary, pick, None
    if not summary:
        return summary, None, None
    summary.sort(key=lambda r: (penalty(r), -r["ok_mean"], r["ran_worst"], r["p95_worst"]))
    return summary, None, summary[0]


def format_float(x):
    return f"{x:.4f}"


def main():
    args = parse_args()
    run_id = run_id_now()
    models_dir = Path("models") / "sweeps" / run_id
    runs_dir = Path("runs") / "sweeps" / run_id
    reports_dir = Path("reports") / "sweeps" / run_id
    ensure_new_dir(models_dir)
    ensure_new_dir(runs_dir)
    ensure_new_dir(reports_dir)
    out_md = Path(args.out_md) if args.out_md else reports_dir / "product_proof_router_v2_scifact_robust_verify.md"
    if out_md.exists():
        raise FileExistsError(f"output file exists: {out_md}")
    if not out_md.resolve().as_posix().startswith(reports_dir.resolve().as_posix()):
        raise ValueError("out_md must be under reports/sweeps/<run_id>")
    files = discover_files(args.in_dirs, args.glob, args.recursive)
    files_matched = len(files)
    scifact_files = []
    for p in files:
        tr = detect_track(p)
        if tr == "scifact":
            scifact_files.append(p)
    strict_schema = True
    skip_partial = True
    kept = []
    dropped_tag = []
    unknown_tag_files = []
    missing_budget_files = []
    per_file_rows = {}
    for p in scifact_files:
        rows, err = read_jsonl_rows(p, strict_schema, skip_partial)
        if err is not None or rows is None:
            continue
        tag = extract_tag(p, rows[:5])
        if args.include_tag and tag != args.include_tag:
            dropped_tag.append(str(p))
            if tag == "unknown":
                unknown_tag_files.append(str(p))
            continue
        budget = extract_budget(p, rows[:5])
        seed = extract_seed(p, rows[:5])
        metrics = compute_metrics(rows)
        if metrics is None:
            continue
        metrics.update({
            "file": str(p),
            "tag": tag,
            "budget": budget,
            "seed": seed,
        })
        kept.append(metrics)
        per_file_rows[str(p)] = rows
        if budget is None:
            missing_budget_files.append(str(p))
    files_with_budget = [k for k in kept if k["budget"] is not None]
    groups = defaultdict(list)
    for k in files_with_budget:
        b = round(float(k["budget"]), 3)
        groups[(k["tag"], b)].append(k)
    constraints = {"fp": 0.01, "fn": 0.01, "ran": 0.25, "cap": 0.20}
    summary, pick, closest = summarize_groups(groups, constraints)
    if args.debug:
        print(f"DEBUG files_matched_total={files_matched}")
        print(f"DEBUG files_kept_after_tag_filter={len(kept)}")
        for m in kept[:5]:
            print(f"DEBUG file={m['file']} tag={m['tag']} budget={m['budget']}")
    lines = []
    lines.append("# SciFact Robust Verify Aggregate (Filtered v2)")
    lines.append("")
    lines.append("Discovery summary")
    lines.append("")
    lines.append(f"- in_dirs: {args.in_dirs}")
    lines.append(f"- glob: `{args.glob}`")
    lines.append(f"- recursive: {args.recursive}")
    lines.append(f"- files_matched_total: {files_matched}")
    lines.append(f"- files_kept_scifact: {len(scifact_files)}")
    lines.append(f"- files_kept_after_tag_filter: {len(kept)}")
    lines.append(f"- files_with_budget: {len(files_with_budget)}")
    lines.append(f"- files_missing_budget: {len(missing_budget_files)}")
    if args.include_tag:
        lines.append(f"- include_tag: `{args.include_tag}`")
        lines.append(f"- dropped_tag_mismatch: {len(dropped_tag)}")
        lines.append(f"- unknown_tag_dropped: {len(unknown_tag_files)}")
    lines.append("")
    lines.append("Top 20 kept file paths")
    lines.append("")
    for p in kept[:20]:
        lines.append(f"- {p['file']}")
    lines.append("")
    lines.append("Budget summary")
    lines.append("")
    lines.append("| tag | budget | ok_mean | fp_worst | fn_worst | ran_worst | cap_worst | p95_worst | n_files |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(summary, key=lambda r: (r["tag"], r["budget"])):
        lines.append(
            f"| {row['tag']} | {row['budget']:.3f} | {format_float(row['ok_mean'])} | "
            f"{format_float(row['fp_worst'])} | {format_float(row['fn_worst'])} | "
            f"{format_float(row['ran_worst'])} | {format_float(row['cap_worst'])} | "
            f"{format_float(row['p95_worst'])} | {row['n_files']} |"
        )
    lines.append("")
    lines.append("Ship recommendation")
    lines.append("")
    if pick:
        lines.append(
            f"- group=(tag={pick['tag']}, budget={pick['budget']:.3f}) "
            f"fp_worst={format_float(pick['fp_worst'])} fn_worst={format_float(pick['fn_worst'])} "
            f"ran_worst={format_float(pick['ran_worst'])} cap_worst={format_float(pick['cap_worst'])} "
            f"ok_mean={format_float(pick['ok_mean'])} p95_worst={format_float(pick['p95_worst'])}"
        )
        wf = pick["worst_files"]
        lines.append("- worst_files:")
        lines.append(f"  - fp: {wf['fp']}")
        lines.append(f"  - fn: {wf['fn']}")
        lines.append(f"  - ran: {wf['ran']}")
        lines.append(f"  - cap: {wf['cap']}")
        lines.append(f"  - p95: {wf['p95']}")
    else:
        lines.append("- no group met constraints")
        if closest:
            lines.append(
                f"- closest: group=(tag={closest['tag']}, budget={closest['budget']:.3f}) "
                f"fp_worst={format_float(closest['fp_worst'])} fn_worst={format_float(closest['fn_worst'])} "
                f"ran_worst={format_float(closest['ran_worst'])} cap_worst={format_float(closest['cap_worst'])} "
                f"ok_mean={format_float(closest['ok_mean'])} p95_worst={format_float(closest['p95_worst'])}"
            )
            wf = closest["worst_files"]
            lines.append("- worst_files:")
            lines.append(f"  - fp: {wf['fp']}")
            lines.append(f"  - fn: {wf['fn']}")
            lines.append(f"  - ran: {wf['ran']}")
            lines.append(f"  - cap: {wf['cap']}")
            lines.append(f"  - p95: {wf['p95']}")
    lines.append("")
    lines.append("Unusable files (missing budget)")
    lines.append("")
    for p in missing_budget_files:
        lines.append(f"- {p}")
    if args.include_tag:
        lines.append("")
        lines.append("Dropped by tag mismatch")
        lines.append("")
        for p in dropped_tag[:50]:
            lines.append(f"- {p}")
    safe_write_md(out_md, "\n".join(lines) + "\n")
    print(str(models_dir))
    print(str(runs_dir))
    print(str(reports_dir))
    print(str(out_md))


if __name__ == "__main__":
    main()
