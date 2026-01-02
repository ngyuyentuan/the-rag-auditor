import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def iter_jsonl(path, cap=None):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if cap is not None and count >= cap:
                return


def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[int(f)] * (c - k)
    d1 = vals[int(c)] * (k - f)
    return d0 + d1


def summarize_rows(rows):
    n = 0
    accept = 0
    reject = 0
    uncertain = 0
    fp = 0
    fn = 0
    stage2_ran = 0
    rerank_ms = []
    nli_ms = []
    total_ms = []
    rerank_gt0 = 0
    nli_gt0 = 0
    for row in rows:
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
        stage2 = row.get("stage2", {}) or {}
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        r = float(timing.get("rerank_ms", 0.0))
        nli = float(timing.get("nli_ms", 0.0))
        t = float(timing.get("total_ms", 0.0))
        rerank_ms.append(r)
        nli_ms.append(nli)
        total_ms.append(t)
        if r > 0:
            rerank_gt0 += 1
        if nli > 0:
            nli_gt0 += 1
    if n == 0:
        return None
    return {
        "n": n,
        "accept_rate": accept / n,
        "reject_rate": reject / n,
        "uncertain_rate": uncertain / n,
        "stage2_rate": stage2_ran / n,
        "fp_accept": fp / n,
        "fn_reject": fn / n,
        "ok_rate": 1.0 - (fp / n) - (fn / n),
        "mean_ms": statistics.mean(total_ms) if total_ms else 0.0,
        "p95_ms": percentile(total_ms, 95) or 0.0,
        "p99_ms": percentile(total_ms, 99) or 0.0,
        "max_ms": max(total_ms) if total_ms else 0.0,
        "rerank_gt0": rerank_gt0,
        "nli_gt0": nli_gt0,
        "keys": None,
    }


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--tracks", nargs="+", default=["scifact", "fever"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    baselines = [
        ("calibrated", "day12_{track}_500_e2e.jsonl"),
        ("always_stage2", "day12_{track}_500_e2e_always.jsonl"),
        ("random", "day12_{track}_500_e2e_random.jsonl"),
    ]

    found_any = False
    summary = {}
    print("schema_check:")
    for track in args.tracks:
        summary[track] = {}
        for baseline, pattern in baselines:
            path = runs_dir / pattern.format(track=track)
            if not path.exists():
                print(f"missing: {path}")
                continue
            found_any = True
            rows = list(iter_jsonl(path, cap=args.n))
            if not rows:
                print(f"missing: {path}")
                continue
            keys = sorted(rows[0].keys())
            required = {"baseline", "metadata", "stage1", "stage2", "timing_ms", "ground_truth"}
            print(path)
            print("keys", keys)
            print("required_present", required.issubset(set(keys)))
            metrics = summarize_rows(rows)
            metrics["keys"] = keys
            summary[track][baseline] = metrics
            print("stage2_ran", metrics["stage2_rate"], "rerank_ms_gt0", metrics["rerank_gt0"], "nli_ms_gt0", metrics["nli_gt0"])

    if not found_any:
        sys.exit(1)

    print("metrics_table:")
    for track in args.tracks:
        print(f"track={track}")
        print("baseline\tn\taccept\treject\tuncertain\tstage2_rate\tok_rate\tfp_accept\tfn_reject\tmean_ms\tp95_ms\tp99_ms\tmax_ms")
        for baseline, _ in baselines:
            m = summary.get(track, {}).get(baseline)
            if not m:
                print(f"{baseline}\t0\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a\tn/a")
                continue
            print("{baseline}\t{n}\t{accept}\t{reject}\t{uncertain}\t{stage2_rate}\t{ok_rate}\t{fp}\t{fn}\t{mean_ms}\t{p95}\t{p99}\t{max_ms}".format(
                baseline=baseline,
                n=m["n"],
                accept=format_float(m["accept_rate"]),
                reject=format_float(m["reject_rate"]),
                uncertain=format_float(m["uncertain_rate"]),
                stage2_rate=format_float(m["stage2_rate"]),
                ok_rate=format_float(m["ok_rate"]),
                fp=format_float(m["fp_accept"]),
                fn=format_float(m["fn_reject"]),
                mean_ms=f"{m['mean_ms']:.2f}",
                p95=f"{m['p95_ms']:.2f}",
                p99=f"{m['p99_ms']:.2f}",
                max_ms=f"{m['max_ms']:.2f}",
            ))

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Day12 E2E Product Summary")
        lines.append("")
        lines.append("Included runs:")
        for track in args.tracks:
            for _, pattern in baselines:
                lines.append(f"- `runs/{pattern.format(track=track)}`")
        lines.append("")
        for track in args.tracks:
            lines.append(f"## Track: {track}")
            lines.append("")
            lines.append("| baseline | n | accept | reject | uncertain | stage2_rate | ok_rate | fp_accept | fn_reject | mean_ms | p95_ms | p99_ms | max_ms |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for baseline, _ in baselines:
                m = summary.get(track, {}).get(baseline)
                if not m:
                    lines.append(f"| {baseline} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                lines.append("| {baseline} | {n} | {accept} | {reject} | {uncertain} | {stage2_rate} | {ok_rate} | {fp} | {fn} | {mean_ms} | {p95} | {p99} | {max_ms} |".format(
                    baseline=baseline,
                    n=m["n"],
                    accept=format_float(m["accept_rate"]),
                    reject=format_float(m["reject_rate"]),
                    uncertain=format_float(m["uncertain_rate"]),
                    stage2_rate=format_float(m["stage2_rate"]),
                    ok_rate=format_float(m["ok_rate"]),
                    fp=format_float(m["fp_accept"]),
                    fn=format_float(m["fn_reject"]),
                    mean_ms=f"{m['mean_ms']:.2f}",
                    p95=f"{m['p95_ms']:.2f}",
                    p99=f"{m['p99_ms']:.2f}",
                    max_ms=f"{m['max_ms']:.2f}",
                ))
            lines.append("")
            cal = summary.get(track, {}).get("calibrated")
            alw = summary.get(track, {}).get("always_stage2")
            if cal and alw:
                delta_ok = alw["ok_rate"] - cal["ok_rate"]
                delta_ms = alw["mean_ms"] - cal["mean_ms"]
                lines.append("Product interpretation")
                lines.append("")
                lines.append("UNCERTAIN is a deferral to stage2 and is not counted as error in ok_rate.")
                lines.append("ok_rate reflects stage1 label-only correctness on accept/reject decisions, not end-to-end factual correctness without gold evidence/verdict.")
                lines.append(f"always_stage2 vs calibrated: delta ok_rate `{delta_ok:.4f}`, delta mean_ms `{delta_ms:.2f}`.")
                lines.append("")
        out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
