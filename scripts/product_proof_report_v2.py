import argparse
import json
import math
from pathlib import Path

import yaml


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


def percentile(values, p):
    if not values:
        return 0.0
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


def summarize(path):
    n = 0
    accept = 0
    reject = 0
    uncertain = 0
    fp = 0
    fn = 0
    stage2_ran = 0
    total_ms = []
    stage2_compute = 0
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
        stage2 = row.get("stage2", {}) or {}
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
        if float(timing.get("rerank_ms", 0.0)) > 0 and float(timing.get("nli_ms", 0.0)) > 0:
            stage2_compute += 1
    n = n or 1
    return {
        "n": n,
        "accept_rate": accept / n,
        "reject_rate": reject / n,
        "uncertain_rate": uncertain / n,
        "stage2_rate": stage2_ran / n,
        "fp_accept_rate": fp / n,
        "fn_reject_rate": fn / n,
        "ok_rate_stage1": 1.0 - (fp / n) - (fn / n),
        "mean_ms": sum(total_ms) / n if total_ms else 0.0,
        "p95_ms": percentile(total_ms, 95),
        "stage2_compute_count": stage2_compute,
    }


def format_float(x):
    return f"{x:.4f}"


def summarize_stage2(path):
    n = 0
    stage2_ran = 0
    rerank_gt0 = 0
    nli_gt0 = 0
    total_ms = []
    for row in iter_jsonl(path):
        n += 1
        stage2 = row.get("stage2", {}) or {}
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
        if float(timing.get("rerank_ms", 0.0)) > 0:
            rerank_gt0 += 1
        if float(timing.get("nli_ms", 0.0)) > 0:
            nli_gt0 += 1
    n = n or 1
    return {
        "n": n,
        "stage2_rate": stage2_ran / n,
        "mean_ms": sum(total_ms) / n if total_ms else 0.0,
        "p95_ms": percentile(total_ms, 95),
        "rerank_ms_gt0": rerank_gt0,
        "nli_ms_gt0": nli_gt0,
    }


def load_gates(track):
    report_path = Path(f"reports/stage1_product_gates_{track}.md")
    if not report_path.exists():
        return None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("Gate fp<="):
            parts = s.replace("Gate", "").strip().split()
            vals = {}
            for p in parts:
                if "<=" in p:
                    k, v = p.split("<=")
                    vals[k] = float(v)
            if "fp" in vals and "fn" in vals and "uncertain" in vals:
                return {
                    "max_fp_accept": vals["fp"],
                    "max_fn_reject": vals["fn"],
                    "max_uncertain": vals["uncertain"],
                }
    return None


def load_prod_yaml(track):
    path = Path(f"configs/thresholds_stage1_prod_{track}.yaml")
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return None
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["scifact", "fever"])
    ap.add_argument("--baseline_jsonl_scifact")
    ap.add_argument("--baseline_jsonl_fever")
    ap.add_argument("--tuned_jsonl_scifact")
    ap.add_argument("--tuned_jsonl_fever")
    ap.add_argument("--matrix_stats_json_scifact")
    ap.add_argument("--matrix_stats_json_fever")
    ap.add_argument("--stage1_only_jsonl_scifact")
    ap.add_argument("--stage1_only_jsonl_fever")
    ap.add_argument("--stage2_jsonl_scifact")
    ap.add_argument("--stage2_jsonl_fever")
    ap.add_argument("--out_md", default="reports/product_proof_v2.md")
    args = ap.parse_args()

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Product Proof v2")
    lines.append("")

    for track in args.tracks:
        baseline_path = Path(getattr(args, f"baseline_jsonl_{track}", "") or "")
        tuned_path = Path(getattr(args, f"tuned_jsonl_{track}", "") or "")
        stage1_only_path = Path(getattr(args, f"stage1_only_jsonl_{track}", "") or f"runs/product_proof/{track}_baseline_uncertain_only_50_dry.jsonl")
        stage2_path = Path(getattr(args, f"stage2_jsonl_{track}", "") or f"runs/product_proof/{track}_baseline_uncertain_only_50_real.jsonl")
        lines.append(f"## {track}")
        lines.append("")
        if not baseline_path.exists() or not tuned_path.exists():
            lines.append("- missing baseline or tuned jsonl")
            lines.append("")
            continue
        base = summarize(baseline_path)
        tuned = summarize(tuned_path)
        lines.append("Baseline vs tuned")
        lines.append("")
        lines.append("| metric | baseline | tuned | delta |")
        lines.append("|---|---:|---:|---:|")
        for key in [
            "accept_rate",
            "reject_rate",
            "uncertain_rate",
            "stage2_rate",
            "fp_accept_rate",
            "fn_reject_rate",
            "ok_rate_stage1",
            "mean_ms",
            "p95_ms",
        ]:
            delta = tuned[key] - base[key]
            lines.append("| {k} | {b} | {t} | {d} |".format(
                k=key,
                b=format_float(base[key]) if "rate" in key else f"{base[key]:.2f}",
                t=format_float(tuned[key]) if "rate" in key else f"{tuned[key]:.2f}",
                d=format_float(delta) if "rate" in key else f"{delta:.2f}",
            ))
        lines.append("")
        lines.append(f"- stage2_compute_count baseline: `{base['stage2_compute_count']}` tuned: `{tuned['stage2_compute_count']}`")
        lines.append("")

        gates = load_gates(track)
        if gates:
            violates = []
            if tuned["fp_accept_rate"] > gates["max_fp_accept"]:
                violates.append("fp_accept_rate")
            if tuned["fn_reject_rate"] > gates["max_fn_reject"]:
                violates.append("fn_reject_rate")
            if tuned["uncertain_rate"] > gates["max_uncertain"]:
                violates.append("uncertain_rate")
            if violates:
                lines.append(f"- gate violation: `{', '.join(violates)}` (fp<={gates['max_fp_accept']}, fn<={gates['max_fn_reject']}, uncertain<={gates['max_uncertain']})")
            else:
                lines.append(f"- gate status: pass (fp<={gates['max_fp_accept']}, fn<={gates['max_fn_reject']}, uncertain<={gates['max_uncertain']})")
        else:
            lines.append("- gate status: unavailable (stage1_product_gates report missing)")
        lines.append("")
        lines.append("Stage2 scenario")
        lines.append("")
        if stage1_only_path.exists():
            s1 = summarize_stage2(stage1_only_path)
            lines.append(f"- stage1_only: stage2_rate={format_float(s1['stage2_rate'])} mean_ms={s1['mean_ms']:.2f} p95_ms={s1['p95_ms']:.2f} rerank_ms_gt0={s1['rerank_ms_gt0']} nli_ms_gt0={s1['nli_ms_gt0']}")
        else:
            lines.append("- stage1_only: missing")
        if stage2_path.exists():
            s2 = summarize_stage2(stage2_path)
            if s2["rerank_ms_gt0"] == 0 and s2["nli_ms_gt0"] == 0:
                lines.append("- stage2_enabled: missing artifacts or stage2 skipped")
            else:
                lines.append(f"- stage2_enabled: stage2_rate={format_float(s2['stage2_rate'])} mean_ms={s2['mean_ms']:.2f} p95_ms={s2['p95_ms']:.2f} rerank_ms_gt0={s2['rerank_ms_gt0']} nli_ms_gt0={s2['nli_ms_gt0']}")
        else:
            lines.append("- stage2_enabled: missing run")
        lines.append("")
        lines.append("Policy comparisons")
        lines.append("")
        prod_yaml = load_prod_yaml(track)
        if prod_yaml and prod_yaml.get("fallback"):
            lines.append("- prod thresholds: fallback baseline")
        elif prod_yaml:
            lines.append("- prod thresholds: applied")
        else:
            lines.append("- prod thresholds: missing")
        lines.append("")
        def add_pair(label, base_path, prod_path):
            if not base_path.exists() or not prod_path.exists():
                lines.append(f"- {label}: missing")
                return
            b = summarize(base_path)
            t = summarize(prod_path)
            lines.append(f"- {label}")
            lines.append("  | metric | baseline | prod | delta |")
            lines.append("  |---|---:|---:|---:|")
            for key in [
                "accept_rate",
                "reject_rate",
                "uncertain_rate",
                "stage2_rate",
                "fp_accept_rate",
                "fn_reject_rate",
                "ok_rate_stage1",
                "mean_ms",
                "p95_ms",
            ]:
                delta = t[key] - b[key]
                lines.append("  | {k} | {b} | {t} | {d} |".format(
                    k=key,
                    b=format_float(b[key]) if "rate" in key else f"{b[key]:.2f}",
                    t=format_float(t[key]) if "rate" in key else f"{t[key]:.2f}",
                    d=format_float(delta) if "rate" in key else f"{delta:.2f}",
                ))
        out_dir = Path("runs/product_proof")
        n_candidates = []
        for p in out_dir.glob(f"{track}_baseline_uncertain_only_*_real.jsonl"):
            try:
                n_candidates.append(int(p.name.split("_")[-2]))
            except Exception:
                pass
        n_val = max(n_candidates) if n_candidates else None
        if n_val is None:
            lines.append("- product_proof runs missing")
        else:
            base_u = out_dir / f"{track}_baseline_uncertain_only_{n_val}_real.jsonl"
            prod_u = out_dir / f"{track}_prod_uncertain_only_{n_val}.jsonl"
            base_a = out_dir / f"{track}_baseline_always_{n_val}_real.jsonl"
            prod_a = out_dir / f"{track}_prod_always_{n_val}.jsonl"
            add_pair("uncertain_only", base_u, prod_u)
            add_pair("always", base_a, prod_a)
        lines.append("")
        lines.append("Ship recommendation")
        lines.append("")
        lines.append("- consider uncertain_only if stage2_rate is acceptable and p95_ms stays within product latency budget")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
