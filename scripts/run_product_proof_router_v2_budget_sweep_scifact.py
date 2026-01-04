import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_product_proof_router_v2_matrix import summarize, write_sample
from src.stage1.router_v2 import load, decide_router, build_feature_frame


def run(cmd):
    subprocess.run(cmd, check=True)


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def prepare_rows(baseline_rows, router, feature_map, features_present):
    prepared = []
    for row in baseline_rows:
        qid_raw = (row.get("metadata") or {}).get("qid")
        qid = str(qid_raw)
        if qid not in feature_map:
            try:
                qid = str(int(float(qid)))
            except Exception:
                pass
        p_hat = None
        if qid in feature_map:
            x = feature_map[qid].reshape(1, -1)
            p_hat = float(router.model.predict_proba(x)[0, 1])
            if getattr(router, "prob_flip", False):
                p_hat = 1.0 - p_hat
        if p_hat is None:
            decision = row.get("stage1", {}).get("route_decision", "UNCERTAIN")
        else:
            decision = decide_router(p_hat, float(router.t_accept), float(router.t_reject))
        prepared.append({
            "row": row,
            "qid": qid,
            "p_hat": p_hat,
            "decision": decision,
            "features_present": features_present,
        })
    return prepared


def apply_budget(prepared, out_path, router, budget):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(prepared)
    budget_k = int((budget or 0.0) * n)
    uncertain_list = []
    for idx, entry in enumerate(prepared):
        if entry["decision"] != "UNCERTAIN":
            continue
        p_hat = entry["p_hat"]
        if p_hat is None:
            cs_ret = (entry["row"].get("stage1") or {}).get("cs_ret")
            if cs_ret is None:
                cs_ret = 0.5
            score = 1.0 - abs(float(cs_ret) - 0.5)
        else:
            score = 1.0 - abs(p_hat - 0.5)
        uncertain_list.append((score, idx))
    uncertain_list.sort(key=lambda x: (-x[0], x[1]))
    keep = {idx for _, idx in uncertain_list[:budget_k]}
    rank_map = {idx: i + 1 for i, (_, idx) in enumerate(uncertain_list)}

    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(prepared):
            row = json.loads(json.dumps(entry["row"]))
            stage1 = row.get("stage1") or {}
            router_decision = entry["decision"]
            route_decision = router_decision
            budget_capped = False
            routed_to_stage2 = False
            if router_decision == "UNCERTAIN":
                if idx not in keep:
                    route_decision = "UNCERTAIN"
                    stage1["route_reason"] = "stage2_budget_cap"
                    budget_capped = True
                else:
                    routed_to_stage2 = True
            stage1["route_decision"] = route_decision
            stage1["router_decision"] = router_decision
            stage1["route_source"] = "router_v2"
            stage1["t_lower"] = float(router.t_reject)
            stage1["t_upper"] = float(router.t_accept)
            stage1["router"] = {
                "name": "router_v2",
                "p_hat": entry["p_hat"],
                "t_accept": float(router.t_accept),
                "t_reject": float(router.t_reject),
                "features_present": entry["features_present"],
                "stage2_budget": budget,
                "routed_to_stage2": routed_to_stage2,
                "budget_rank": rank_map.get(idx),
                "budget_capped": budget_capped,
            }
            row["stage1"] = stage1
            stage2 = row.get("stage2") or {}
            timing = row.get("timing_ms") or {}
            has_stage2_data = bool(float(timing.get("rerank_ms", 0.0)) > 0 or float(timing.get("nli_ms", 0.0)) > 0)
            should_run = route_decision == "UNCERTAIN"
            if budget_capped:
                should_run = False
            stage2["route_requested"] = bool(route_decision == "UNCERTAIN")
            stage2["capped"] = bool(budget_capped)
            stage2["cap_budget"] = budget
            stage2["capped_reason"] = "budget_cap" if budget_capped else None
            if should_run and has_stage2_data:
                stage2["ran"] = True
            else:
                stage2["ran"] = False
                stage2["rerank"] = {}
                stage2["nli"] = {}
                if budget_capped:
                    stage2["rerank"] = {"skipped": True, "reason": "budget_cap"}
                    stage2["nli"] = {"skipped": True, "reason": "budget_cap"}
                    stage2["skipped_reason"] = "budget_cap"
                timing["rerank_ms"] = 0.0
                timing["nli_ms"] = 0.0
                if "stage1_ms" in timing:
                    timing["total_ms"] = float(timing.get("stage1_ms", 0.0))
                row["timing_ms"] = timing
            row["stage2"] = stage2
            pred = row.get("pred") or {}
            if budget_capped:
                pred["pred_verdict"] = None
                pred["pred_doc_id"] = None
                pred["pred_has_evidence"] = None
                row["pred"] = pred
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="runs/product_proof_router_v2_budget_sweep_scifact")
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--model_hgb", required=True)
    ap.add_argument("--model_logreg")
    ap.add_argument("--model_current")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.42, 0.45])
    ap.add_argument("--out_md", default="reports/product_proof_router_v2_scifact_budget_sweep.md")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = Path(args.baseline_jsonl)
    if not baseline.exists():
        raise SystemExit(f"missing baseline_jsonl {baseline}")

    models = []
    if args.model_current:
        models.append(("router_v2_current", args.model_current))
    if args.model_logreg:
        models.append(("router_v2_min_route_logreg", args.model_logreg))
    models.append(("router_v2_min_route_hgb", args.model_hgb))

    baseline_rows = list(iter_jsonl(baseline))
    qid_list = []
    for row in baseline_rows:
        qid = str((row.get("metadata") or {}).get("qid"))
        qid_list.append(qid)
    qid_set = set(qid_list)
    df = pd.read_parquet(args.scifact_in_path)
    if "qid" not in df.columns:
        raise SystemExit("missing qid in parquet")
    df["qid"] = df["qid"].astype(str)
    df = df[df["qid"].isin(qid_set)]
    qids = df["qid"].astype(str).tolist()
    reports = {}
    routers = []
    union_cols = set()
    for name, path in models:
        router = load(path)
        routers.append((name, router))
        union_cols.update(router.feature_cols)
    union_cols = list(union_cols)
    feat_all = build_feature_frame(df, routers[0][1].logit_col, union_cols)
    feature_map_all = dict(zip(qids, feat_all.to_numpy()))
    present_all = {col: True for col in union_cols}
    for name, router in routers:
        col_idx = [union_cols.index(c) for c in router.feature_cols]
        feature_map = {qid: feature_map_all[qid][col_idx] for qid in qids}
        features_present = {k: present_all.get(k, False) for k in router.feature_cols}
        prepared = prepare_rows(baseline_rows, router, feature_map, features_present)
        rows = []
        for b in args.budgets:
            out_path = out_dir / f"{name}_budget_{b:.2f}_{args.n}.jsonl"
            apply_budget(prepared, out_path, router, b)
            stats = summarize(list(iter_jsonl(out_path)), "uncertain_only")
            stats["budget"] = b
            rows.append(stats)
        reports[name] = rows

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Product Proof Router v2 SciFact Budget Sweep")
    lines.append("")
    for name, rows in reports.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append("| {b:.2f} | {a:.4f} | {rj:.4f} | {u:.4f} | {sr:.4f} | {rr:.4f} | {cr:.4f} | {fp:.4f} | {fn:.4f} | {ok:.4f} | {mean:.2f} | {p95:.2f} |".format(
                b=r["budget"],
                a=r["accept_rate"],
                rj=r["reject_rate"],
                u=r["uncertain_rate"],
                sr=r["stage2_route_rate"],
                rr=r["stage2_ran_rate"],
                cr=r["capped_rate"],
                fp=r["fp_accept_rate"],
                fn=r["fn_reject_rate"],
                ok=r["ok_rate_stage1"],
                mean=r["mean_ms"],
                p95=r["p95_ms"],
            ))
        lines.append("")
    lines.append("Ship recommendation")
    lines.append("")
    best = None
    for name, rows in reports.items():
        for r in rows:
            if r["fp_accept_rate"] > 0.01 or r["fn_reject_rate"] > 0.01:
                continue
            if r["stage2_ran_rate"] > 0.25 or r["capped_rate"] > 0.20:
                continue
            cand = (-r["ok_rate_stage1"], r["stage2_ran_rate"], r["p95_ms"], name, r)
            if best is None or cand < best:
                best = cand
    if best is None:
        lines.append("- no candidate met fp<=0.01 fn<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20")
    else:
        _, _, _, name, r = best
        lines.append("- {name} budget={b:.2f} fp={fp:.4f} fn={fn:.4f} stage2_ran_rate={rr:.4f} capped_rate={cr:.4f} ok_rate_stage1={ok:.4f}".format(
            name=name,
            b=r["budget"],
            fp=r["fp_accept_rate"],
            fn=r["fn_reject_rate"],
            rr=r["stage2_ran_rate"],
            cr=r["capped_rate"],
            ok=r["ok_rate_stage1"],
        ))
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
