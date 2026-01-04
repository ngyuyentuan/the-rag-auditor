import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_product_proof_router_v2_matrix import apply_router_v2, build_feature_map, summarize, write_sample
from src.stage1.router_v2 import load


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="runs/product_proof_router_v2_sweep_scifact")
    ap.add_argument("--model_current", required=True)
    ap.add_argument("--model_min_route_logreg", required=True)
    ap.add_argument("--model_min_route_hgb", required=True)
    ap.add_argument("--out_md", default="reports/product_proof_router_v2_scifact_sweep.md")
    ap.add_argument("--stage2_budget", type=float, default=0.45)
    ap.add_argument("--baseline_jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = out_dir / f"scifact_sample_{args.n}.jsonl"
    if not sample.exists():
        write_sample(sample, args.n)

    baseline = out_dir / f"scifact_baseline_uncertain_only_{args.n}_real.jsonl"
    if not baseline.exists():
        if args.baseline_jsonl:
            src = Path(args.baseline_jsonl)
            if src.exists():
                baseline.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        if baseline.exists():
            pass
        else:
            fallback = Path("runs/product_proof_router_v2") / f"scifact_baseline_uncertain_only_{args.n}_real.jsonl"
            if fallback.exists():
                baseline.write_text(fallback.read_text(encoding="utf-8"), encoding="utf-8")
    if not baseline.exists():
        cmd = [
            ".venv/bin/python",
            "scripts/day12_e2e_run_500.py",
            "--track",
            "scifact",
            "--n",
            str(args.n),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--stage2_policy",
            "uncertain_only",
            "--baseline_mode",
            "calibrated",
            "--rerank_topk",
            "20",
            "--rerank_keep",
            "5",
            "--batch_size",
            "16",
            "--out",
            str(baseline),
            "--in_path",
            args.scifact_in_path,
            "--sample",
            str(sample),
        ]
        run(cmd)

    current_model = load(args.model_current)
    logreg_model = load(args.model_min_route_logreg)
    hgb_model = load(args.model_min_route_hgb)

    current_map, current_present = build_feature_map(args.scifact_in_path, current_model.logit_col, current_model.feature_cols)
    logreg_map, logreg_present = build_feature_map(args.scifact_in_path, logreg_model.logit_col, logreg_model.feature_cols)
    hgb_map, hgb_present = build_feature_map(args.scifact_in_path, hgb_model.logit_col, hgb_model.feature_cols)

    out_current = out_dir / f"scifact_router_v2_current_uncertain_only_{args.n}.jsonl"
    out_logreg = out_dir / f"scifact_router_v2_min_route_logreg_uncertain_only_{args.n}.jsonl"
    out_hgb = out_dir / f"scifact_router_v2_min_route_hgb_uncertain_only_{args.n}.jsonl"

    apply_router_v2(baseline, out_current, current_model, current_map, float(current_model.t_accept), float(current_model.t_reject), "uncertain_only", args.stage2_budget, current_present)
    apply_router_v2(baseline, out_logreg, logreg_model, logreg_map, float(logreg_model.t_accept), float(logreg_model.t_reject), "uncertain_only", args.stage2_budget, logreg_present)
    apply_router_v2(baseline, out_hgb, hgb_model, hgb_map, float(hgb_model.t_accept), float(hgb_model.t_reject), "uncertain_only", args.stage2_budget, hgb_present)

    rows = [
        ("baseline", baseline),
        ("router_v2_current", out_current),
        ("router_v2_min_route_logreg", out_logreg),
        ("router_v2_min_route_hgb", out_hgb),
    ]
    report = Path(args.out_md)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Product Proof Router v2 SciFact Sweep")
    lines.append("")
    lines.append("Note: Budget sweep is authoritative for shipping because stage2_budget changes stage2_ran/capped tradeoff.")
    lines.append("")
    lines.append("| name | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, path in rows:
        s = summarize(list(iter_jsonl(path)), "uncertain_only")
        lines.append("| {n} | {a} | {r} | {u} | {sr} | {rr} | {cr} | {fp} | {fn} | {ok} | {mean} | {p95} |".format(
            n=name,
            a=f"{s['accept_rate']:.4f}",
            r=f"{s['reject_rate']:.4f}",
            u=f"{s['uncertain_rate']:.4f}",
            sr=f"{s['stage2_route_rate']:.4f}",
            rr=f"{s['stage2_ran_rate']:.4f}",
            cr=f"{s['capped_rate']:.4f}",
            fp=f"{s['fp_accept_rate']:.4f}",
            fn=f"{s['fn_reject_rate']:.4f}",
            ok=f"{s['ok_rate_stage1']:.4f}",
            mean=f"{s['mean_ms']:.2f}",
            p95=f"{s['p95_ms']:.2f}",
        ))
    lines.append("")
    lines.append("Ship recommendation")
    lines.append("")
    best = None
    for name, path in rows[1:]:
        s = summarize(list(iter_jsonl(path)), "uncertain_only")
        feasible = s["fp_accept_rate"] <= 0.01 and s["fn_reject_rate"] <= 0.01
        if not feasible:
            continue
        if s["stage2_ran_rate"] > 0.25 or s["capped_rate"] > 0.20:
            continue
        cand = (s["stage2_ran_rate"], s["fp_accept_rate"], s["fn_reject_rate"], name, s)
        if best is None or cand < best:
            best = cand
    if best is None:
        lines.append("- no candidate met fp<=0.01 fn<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20")
    else:
        _, _, _, name, s = best
        lines.append(f"- {name} fp={s['fp_accept_rate']:.4f} fn={s['fn_reject_rate']:.4f} stage2_ran_rate={s['stage2_ran_rate']:.4f} capped_rate={s['capped_rate']:.4f}")
    report.write_text("\n".join(lines), encoding="utf-8")


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


import json


if __name__ == "__main__":
    main()
