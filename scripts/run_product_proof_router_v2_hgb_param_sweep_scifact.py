import argparse
import datetime as dt
import json
import os
import secrets
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.router_v2 import RouterV2, build_feature_frame, save, decide_router
from scripts.train_stage1_router_v2 import select_feature_cols, select_thresholds, select_best_candidate, compute_ece


def create_run_dir(base, name):
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def unique_run_id():
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{secrets.token_hex(3)}"


def write_text_atomic(path, text):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"output exists: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"temp exists: {tmp}")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(text)
    if path.exists():
        raise SystemExit(f"output exists: {path}")
    tmp.rename(path)


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
            decision = (row.get("stage1") or {}).get("route_decision", "UNCERTAIN")
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
    if out_path.exists():
        raise SystemExit(f"output exists: {out_path}")
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

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"temp exists: {tmp}")
    with tmp.open("x", encoding="utf-8") as f_out:
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
    if out_path.exists():
        raise SystemExit(f"output exists: {out_path}")
    tmp.rename(out_path)


def compute_stats(path):
    from scripts.run_product_proof_router_v2_matrix import summarize
    return summarize(list(iter_jsonl(path)), "uncertain_only")


def train_hgb_model(df_feat, y_col, seed, params):
    train_df, val_df = train_test_split(df_feat, test_size=0.2, random_state=seed, stratify=df_feat[y_col])
    hgb = HistGradientBoostingClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
    )
    calib = CalibratedClassifierCV(hgb, method="sigmoid", cv=3)
    calib.fit(train_df.drop(columns=[y_col]).to_numpy(), train_df[y_col].to_numpy())
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stage2_budgets", type=float, nargs="+", default=[0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35])
    ap.add_argument("--out_root", default="runs/sweeps")
    ap.add_argument("--out_md")
    args = ap.parse_args()

    run_id = unique_run_id()
    models_dir = create_run_dir("models/sweeps", run_id)
    runs_dir = create_run_dir(args.out_root, run_id)
    reports_dir = create_run_dir("reports/sweeps", run_id)

    out_md = args.out_md
    if out_md is None:
        out_md = reports_dir / "product_proof_router_v2_scifact_hgb_param_sweep.md"
    else:
        out_md = Path(out_md)
        if out_md.exists():
            raise SystemExit(f"output exists: {out_md}")
        if out_md.parent != reports_dir:
            raise SystemExit("out_md must be under reports/sweeps/<run_id>/")

    baseline_path = Path(args.baseline_jsonl)
    if not baseline_path.exists():
        raise SystemExit(f"missing baseline_jsonl {baseline_path}")

    baseline_rows = list(iter_jsonl(baseline_path))
    qid_list = [str((r.get("metadata") or {}).get("qid")) for r in baseline_rows]
    qid_set = set(qid_list)

    df = pd.read_parquet(args.scifact_in_path)
    if "qid" not in df.columns:
        raise SystemExit("missing qid in parquet")
    if "y" not in df.columns:
        raise SystemExit("missing y in parquet")
    df["qid"] = df["qid"].astype(str)
    df = df[df["qid"].isin(qid_set)]
    y = pd.to_numeric(df["y"], errors="coerce")
    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    df["y"] = y[mask].astype(int)

    feature_cols = select_feature_cols(df, "raw_max_top3")
    features = build_feature_frame(df, "raw_max_top3", feature_cols)
    df_feat = pd.concat([features, df[["y"]]], axis=1)

    full_probs_ref = None
    full_y = df_feat["y"].to_numpy()
    alphas = [0.01]
    betas = [0.01]
    c_fp = 10.0
    c_fn = 10.0

    grid = [
        {"max_depth": 2, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"max_depth": 2, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"max_depth": 2, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"max_depth": 2, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"max_depth": 3, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"max_depth": 3, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]

    summary_rows = []
    for params in grid:
        tag = f"d{params['max_depth']}_lr{params['learning_rate']}_leaf{params['min_samples_leaf']}_l2{params['l2_regularization']}"
        model = train_hgb_model(df_feat, "y", args.seed, params)
        router = RouterV2(model, feature_cols, "raw_max_top3", prob_flip=False)
        full_probs = router.predict_proba(df_feat.drop(columns=["y"]))
        auc = float(roc_auc_score(full_y, full_probs)) if len(np.unique(full_y)) > 1 else 0.0
        ece, _ = compute_ece(full_y, full_probs, bins=10)
        t_reject_max = float(np.quantile(full_probs[full_y == 1], 0.01)) if (full_y == 1).any() else None
        best_by_gate, results = select_thresholds(full_y, full_probs, alphas, betas, c_fp, c_fn, c_stage2=1.0, t_reject_max=t_reject_max)
        best, _, feasible = best_by_gate.get((0.01, 0.01), (None, None, None))
        if best is None:
            continue
        selected, feasible_sel = select_best_candidate(results, 0.01, 0.01, "min_route")
        if selected is not None:
            best = selected
            feasible = feasible_sel
        if best["t_accept"] <= best["t_reject"]:
            raise SystemExit("invalid thresholds")
        router.t_accept = float(best["t_accept"])
        router.t_reject = float(best["t_reject"])

        model_path = models_dir / f"stage1_router_v2_scifact_{tag}.joblib"
        save(router, model_path)
        report_path = reports_dir / f"stage1_router_v2_train_scifact_{tag}.md"
        lines = []
        lines.append(f"# Stage1 Router v2 Train (scifact {tag})")
        lines.append("")
        lines.append(f"- in_path: `{args.scifact_in_path}`")
        lines.append(f"- n: `{len(df_feat)}`")
        lines.append(f"- seed: `{args.seed}`")
        lines.append(f"- feature_cols: `{', '.join(feature_cols)}`")
        lines.append(f"- auc: `{auc:.4f}`")
        lines.append(f"- ece: `{ece:.4f}`")
        lines.append(f"- t_accept: `{best['t_accept']:.4f}`")
        lines.append(f"- t_reject: `{best['t_reject']:.4f}`")
        lines.append(f"- fp_accept_rate: `{best['fp_accept_rate']:.4f}`")
        lines.append(f"- fn_reject_rate: `{best['fn_reject_rate']:.4f}`")
        lines.append(f"- uncertain_rate: `{best['uncertain_rate']:.4f}`")
        lines.append(f"- stage2_route_rate: `{best['uncertain_rate']:.4f}`")
        lines.append(f"- ok_rate: `{best['ok_rate']:.4f}`")
        write_text_atomic(report_path, "\n".join(lines))

        feature_map = dict(zip(df["qid"].astype(str).tolist(), build_feature_frame(df, "raw_max_top3", feature_cols).to_numpy()))
        features_present = {k: True for k in feature_cols}
        prepared = prepare_rows(baseline_rows, router, feature_map, features_present)
        best_row = None
        for budget in args.stage2_budgets:
            out_sub = runs_dir / tag
            out_file = out_sub / f"scifact_router_v2_{tag}_budget_{budget:.2f}.jsonl"
            apply_budget(prepared, out_file, router, budget)
            stats = compute_stats(out_file)
            if stats["fp_accept_rate"] <= 0.01 and stats["fn_reject_rate"] <= 0.01 and stats["stage2_ran_rate"] <= 0.25 and stats["capped_rate"] <= 0.20:
                cand = (-stats["ok_rate_stage1"], stats["stage2_ran_rate"], stats["p95_ms"], budget, stats)
                if best_row is None or cand < best_row:
                    best_row = cand
        if best_row is None:
            summary_rows.append((tag, params, None, None, False))
        else:
            _, _, _, budget, stats = best_row
            summary_rows.append((tag, params, budget, stats, True))

    lines = []
    lines.append("# SciFact HGB Param Sweep Summary")
    lines.append("")
    lines.append("Constraints: fp<=0.01, fn<=0.01, stage2_ran_rate<=0.25, capped_rate<=0.20")
    lines.append("")
    lines.append("| tag | max_depth | learning_rate | min_samples_leaf | l2 | best_budget | ok_rate_stage1 | fp | fn | stage2_ran_rate | capped_rate | p95_ms | feasible |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    best = None
    for tag, params, budget, stats, feasible in summary_rows:
        if stats is None:
            lines.append(f"| {tag} | {params['max_depth']} | {params['learning_rate']} | {params['min_samples_leaf']} | {params['l2_regularization']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | False |")
            continue
        lines.append("| {tag} | {d} | {lr} | {leaf} | {l2} | {b:.2f} | {ok:.4f} | {fp:.4f} | {fn:.4f} | {sr:.4f} | {cr:.4f} | {p95:.2f} | {f} |".format(
            tag=tag,
            d=params["max_depth"],
            lr=params["learning_rate"],
            leaf=params["min_samples_leaf"],
            l2=params["l2_regularization"],
            b=budget,
            ok=stats["ok_rate_stage1"],
            fp=stats["fp_accept_rate"],
            fn=stats["fn_reject_rate"],
            sr=stats["stage2_ran_rate"],
            cr=stats["capped_rate"],
            p95=stats["p95_ms"],
            f=feasible,
        ))
        if feasible:
            cand = (-stats["ok_rate_stage1"], stats["stage2_ran_rate"], stats["p95_ms"], tag, budget, stats)
            if best is None or cand < best:
                best = cand
    lines.append("")
    lines.append("Ship recommendation")
    lines.append("")
    if best is None:
        lines.append("- No feasible candidate found")
    else:
        _, _, _, tag, budget, stats = best
        lines.append(f"- {tag} budget={budget:.2f} ok_rate_stage1={stats['ok_rate_stage1']:.4f} fp={stats['fp_accept_rate']:.4f} fn={stats['fn_reject_rate']:.4f} stage2_ran_rate={stats['stage2_ran_rate']:.4f} capped_rate={stats['capped_rate']:.4f}")
    lines.append("")
    lines.append("Repro commands")
    lines.append("")
    lines.append(f"- scifact_in_path: `{args.scifact_in_path}`")
    lines.append(f"- baseline_jsonl: `{args.baseline_jsonl}`")
    lines.append(f"- budgets: `{', '.join([str(b) for b in args.stage2_budgets])}`")
    write_text_atomic(out_md, "\n".join(lines))
    print(str(models_dir))
    print(str(runs_dir))
    print(str(reports_dir))


if __name__ == "__main__":
    main()
