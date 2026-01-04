import argparse
import json
import math
import os
import subprocess
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.router_v2 import build_feature_frame, load, decide_router


def write_sample(path, n):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"row_idx": i}) + "\n")
    return str(path)


def run(cmd):
    subprocess.run(cmd, check=True)


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
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


def summarize(rows, policy):
    n = 0
    accept = 0
    reject = 0
    uncertain = 0
    final_accept = 0
    final_reject = 0
    final_uncertain = 0
    final_fp = 0
    final_fn = 0
    fp = 0
    fn = 0
    stage2_ran = 0
    stage2_route = 0
    capped = 0
    rerank_gt0 = 0
    nli_gt0 = 0
    total_ms = []
    for row in rows:
        n += 1
        stage1 = row.get("stage1", {}) or {}
        router_decision = stage1.get("router_decision") or stage1.get("route_decision")
        decision = stage1.get("route_decision") or router_decision
        if decision == "ACCEPT":
            accept += 1
        elif decision == "REJECT":
            reject += 1
        else:
            uncertain += 1
        stage2 = row.get("stage2", {}) or {}
        route_requested = stage2.get("route_requested")
        if route_requested is None:
            if policy == "always":
                stage2_route += 1
            else:
                if router_decision == "UNCERTAIN":
                    stage2_route += 1
        else:
            stage2_route += 1 if route_requested else 0
        y = (row.get("ground_truth") or {}).get("y")
        if decision == "ACCEPT" and y == 0:
            fp += 1
        if decision == "REJECT" and y == 1:
            fn += 1
        router = stage1.get("router") or {}
        if router.get("budget_capped"):
            capped += 1
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
        if float(timing.get("rerank_ms", 0.0)) > 0:
            rerank_gt0 += 1
        if float(timing.get("nli_ms", 0.0)) > 0:
            nli_gt0 += 1
        pred = row.get("pred", {}) or {}
        pred_verdict = pred.get("pred_verdict")
        if pred_verdict in ("SUPPORTS", "ACCEPT", "ENTAILMENT"):
            final = "ACCEPT"
        elif pred_verdict in ("REFUTES", "REJECT", "CONTRADICTION"):
            final = "REJECT"
        else:
            final = decision if decision in ("ACCEPT", "REJECT") else "UNCERTAIN"
        if final == "ACCEPT":
            final_accept += 1
            if y == 0:
                final_fp += 1
        elif final == "REJECT":
            final_reject += 1
            if y == 1:
                final_fn += 1
        else:
            final_uncertain += 1
    n = n or 1
    return {
        "accept_rate": accept / n,
        "reject_rate": reject / n,
        "uncertain_rate": uncertain / n,
        "stage2_rate": stage2_ran / n,
        "stage2_ran_rate": stage2_ran / n,
        "stage2_route_rate": stage2_route / n,
        "capped_count": capped,
        "capped_rate": capped / n,
        "fp_accept_rate": fp / n,
        "fn_reject_rate": fn / n,
        "ok_rate_stage1": 1.0 - (fp / n) - (fn / n),
        "final_accept_rate": final_accept / n,
        "final_reject_rate": final_reject / n,
        "final_uncertain_rate": final_uncertain / n,
        "final_ok_rate": 1.0 - (final_fp / n) - (final_fn / n),
        "abstain_rate": final_uncertain / n,
        "mean_ms": sum(total_ms) / n if total_ms else 0.0,
        "p95_ms": percentile(total_ms, 95),
        "rerank_ms_gt0": rerank_gt0,
        "nli_ms_gt0": nli_gt0,
    }


def format_float(x):
    return f"{x:.4f}"


def build_feature_map(parquet_path, logit_col, feature_cols):
    df = pd.read_parquet(parquet_path)
    if "qid" not in df.columns:
        raise SystemExit("missing qid in parquet")
    feat_df = build_feature_frame(df, logit_col, feature_cols)
    qids = df["qid"].astype(str).tolist()
    feats = feat_df.to_numpy()
    present = {}
    for col in feature_cols:
        if col in df.columns:
            present[col] = True
        elif col in ["abs_logit", "logit_sq", "logit_sigmoid", "logit_cube", "cs_ret"]:
            present[col] = logit_col in df.columns
        elif col in ["top1_sim", "top2_sim", "top3_sim", "delta12", "count_sim_ge_t"]:
            present[col] = "raw_max_top3" in df.columns or "gap12" in df.columns
        elif col in ["top1", "top2", "top3", "margin", "topk_gap", "topk_ratio", "topk_entropy", "score_span"]:
            present[col] = col in df.columns or ("top1" in df.columns and "top2" in df.columns)
        elif col in ["topk_mean", "topk_std"]:
            present[col] = "mean_top3" in df.columns or "mean_top5" in df.columns or "std_top3" in df.columns or "std_top5" in df.columns
        else:
            present[col] = False
    return dict(zip(qids, feats)), present


def apply_router_v2(in_jsonl, out_jsonl, router, feature_map, t_accept, t_reject, policy, stage2_budget, features_present):
    in_path = Path(in_jsonl)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if t_accept <= t_reject:
        raise SystemExit("router thresholds must satisfy t_accept > t_reject")
    debug = os.environ.get("RAG_AUDITOR_DEBUG") == "1"
    rows = []
    with in_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid_raw = (row.get("metadata") or {}).get("qid")
            qid = str(qid_raw)
            if qid not in feature_map:
                try:
                    qid = str(int(float(qid)))
                except Exception:
                    pass
            stage1 = row.get("stage1") or {}
            router_decision = stage1.get("route_decision")
            p_hat = None
            reason = stage1.get("route_reason")
            if qid in feature_map:
                x = feature_map[qid].reshape(1, -1)
                p_hat = float(router.model.predict_proba(x)[0, 1])
                if getattr(router, "prob_flip", False):
                    p_hat = 1.0 - p_hat
                router_decision = decide_router(p_hat, t_accept, t_reject)
                if router_decision == "ACCEPT":
                    reason = "router_v2_accept"
                elif router_decision == "REJECT":
                    reason = "router_v2_reject"
                else:
                    reason = "router_v2_uncertain"
            rows.append({
                "row": row,
                "qid": qid,
                "router_decision": router_decision,
                "route_reason": reason,
                "p_hat": p_hat,
            })

    budget_k = None
    if stage2_budget is not None:
        budget_k = int(math.floor(stage2_budget * len(rows)))
    uncertain_list = []
    for idx, entry in enumerate(rows):
        if entry["router_decision"] == "UNCERTAIN" and entry["p_hat"] is not None:
            score = 1.0 - abs(entry["p_hat"] - 0.5)
            uncertain_list.append((score, idx))
        elif entry["router_decision"] == "UNCERTAIN":
            stage1 = entry["row"].get("stage1") or {}
            cs_ret = stage1.get("cs_ret")
            if cs_ret is None:
                cs_ret = 0.5
            score = 1.0 - abs(float(cs_ret) - 0.5)
            uncertain_list.append((score, idx))
    uncertain_list.sort(key=lambda x: (-x[0], x[1]))
    keep = set()
    if budget_k is None:
        keep = {idx for _, idx in uncertain_list}
    else:
        keep = {idx for _, idx in uncertain_list[:budget_k]}
    rank_map = {idx: i + 1 for i, (_, idx) in enumerate(uncertain_list)}

    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(rows):
            row = entry["row"]
            stage1 = row.get("stage1") or {}
            router_decision = entry["router_decision"]
            route_decision = router_decision
            budget_capped = False
            routed_to_stage2 = False
            if router_decision == "UNCERTAIN":
                if budget_k is not None and idx not in keep:
                    route_decision = "UNCERTAIN"
                    entry["route_reason"] = "stage2_budget_cap"
                    budget_capped = True
                else:
                    routed_to_stage2 = True
            stage1["route_decision"] = route_decision
            stage1["route_reason"] = entry["route_reason"]
            stage1["t_lower"] = t_reject
            stage1["t_upper"] = t_accept
            stage1["router"] = {
                "name": "router_v2",
                "p_hat": entry["p_hat"],
                "t_accept": float(t_accept),
                "t_reject": float(t_reject),
                "features_present": features_present,
                "stage2_budget": stage2_budget,
                "routed_to_stage2": routed_to_stage2,
                "budget_rank": rank_map.get(idx),
                "budget_capped": budget_capped,
            }
            stage1["router_decision"] = router_decision
            stage1["route_source"] = "router_v2"
            if debug:
                debug_features = None
                if entry["p_hat"] is not None and entry["qid"] in feature_map:
                    debug_features = dict(zip(router.feature_cols, feature_map[entry["qid"]].tolist()))
                stage1["router_debug"] = {
                    "p_hat": entry["p_hat"],
                    "t_accept": float(t_accept),
                    "t_reject": float(t_reject),
                    "router_decision": router_decision,
                    "route_decision": route_decision,
                    "route_source": "router_v2",
                    "budget_capped": budget_capped,
                    "routed_to_stage2": routed_to_stage2,
                    "features": debug_features,
                }
            row["stage1"] = stage1
            stage2 = row.get("stage2") or {}
            timing = row.get("timing_ms") or {}
            has_stage2_data = bool(float(timing.get("rerank_ms", 0.0)) > 0 or float(timing.get("nli_ms", 0.0)) > 0)
            if policy == "always":
                should_run = True
            elif policy == "uncertain_only":
                should_run = route_decision == "UNCERTAIN"
            else:
                should_run = stage2.get("ran") is True
            if budget_capped:
                should_run = False
            stage2["route_requested"] = bool(route_decision == "UNCERTAIN")
            stage2["capped"] = bool(budget_capped)
            stage2["cap_budget"] = stage2_budget
            stage2["capped_reason"] = "budget_cap" if budget_capped else None
            if should_run and has_stage2_data:
                stage2["ran"] = True
            else:
                stage2["ran"] = False
                stage2["rerank"] = {}
                stage2["nli"] = {}
                if should_run and not has_stage2_data:
                    stage2["rerank"] = {"skipped": True, "reason": "missing_stage2_outputs"}
                    stage2["nli"] = {"skipped": True, "reason": "missing_stage2_outputs"}
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


def annotate_baseline(path):
    path = Path(path)
    if not path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as f_in, tmp.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stage1 = row.get("stage1") or {}
            decision = stage1.get("route_decision")
            stage1["router"] = {
                "name": "baseline",
                "p_hat": None,
                "t_accept": None,
                "t_reject": None,
                "features_present": {},
            }
            stage1["router_decision"] = decision
            stage1["route_source"] = "baseline"
            row["stage1"] = stage1
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--fever_in_path", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="runs/product_proof_router_v2")
    ap.add_argument("--model_scifact", default="models/stage1_router_v2_scifact.joblib")
    ap.add_argument("--model_fever", default="models/stage1_router_v2_fever.joblib")
    ap.add_argument("--out_md", default="reports/product_proof_router_v2_matrix.md")
    ap.add_argument("--stage2_budget_scifact", type=float, default=0.45)
    ap.add_argument("--stage2_budget_fever", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scifact_sample = out_dir / f"scifact_sample_{args.n}.jsonl"
    fever_sample = out_dir / f"fever_sample_{args.n}.jsonl"
    if not scifact_sample.exists():
        write_sample(scifact_sample, args.n)
    if not fever_sample.exists():
        write_sample(fever_sample, args.n)

    def base_cmd(track, in_path, sample_path):
        cmd = [
            ".venv/bin/python",
            "scripts/day12_e2e_run_500.py",
            "--track",
            track,
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
            "--in_path",
            in_path,
            "--sample",
            str(sample_path),
        ]
        if track == "fever":
            cmd += ["--logit_col", "logit_platt", "--y_col", "y"]
        return cmd

    scifact_base = base_cmd("scifact", args.scifact_in_path, scifact_sample)
    fever_base = base_cmd("fever", args.fever_in_path, fever_sample)

    scifact_u = out_dir / f"scifact_baseline_uncertain_only_{args.n}_real.jsonl"
    fever_u = out_dir / f"fever_baseline_uncertain_only_{args.n}_real.jsonl"
    scifact_a = out_dir / f"scifact_baseline_always_{args.n}_real.jsonl"
    fever_a = out_dir / f"fever_baseline_always_{args.n}_real.jsonl"

    fallback_dir = Path("runs/product_proof_constrained")

    def ensure_run(path, cmd, fallback_name):
        if path.exists():
            return
        fallback = fallback_dir / fallback_name
        if fallback.exists():
            shutil.copyfile(fallback, path)
            return
        run(cmd)

    ensure_run(scifact_u, scifact_base + ["--out", str(scifact_u)], f"scifact_baseline_uncertain_only_{args.n}_real.jsonl")
    ensure_run(fever_u, fever_base + ["--out", str(fever_u)], f"fever_baseline_uncertain_only_{args.n}_real.jsonl")
    ensure_run(scifact_a, scifact_base + ["--stage2_policy", "always", "--out", str(scifact_a)], f"scifact_baseline_always_{args.n}_real.jsonl")
    ensure_run(fever_a, fever_base + ["--stage2_policy", "always", "--out", str(fever_a)], f"fever_baseline_always_{args.n}_real.jsonl")
    annotate_baseline(scifact_u)
    annotate_baseline(fever_u)
    annotate_baseline(scifact_a)
    annotate_baseline(fever_a)

    sc_router = load(args.model_scifact)
    fv_router = load(args.model_fever)

    scifact_map, scifact_present = build_feature_map(args.scifact_in_path, sc_router.logit_col, sc_router.feature_cols)
    fever_map, fever_present = build_feature_map(args.fever_in_path, fv_router.logit_col, fv_router.feature_cols)

    sc_ta = float(sc_router.t_accept)
    sc_tr = float(sc_router.t_reject)
    fv_ta = float(fv_router.t_accept)
    fv_tr = float(fv_router.t_reject)

    sc_prod_u = out_dir / f"scifact_router_v2_uncertain_only_{args.n}.jsonl"
    sc_prod_a = out_dir / f"scifact_router_v2_always_{args.n}.jsonl"
    fv_prod_u = out_dir / f"fever_router_v2_uncertain_only_{args.n}.jsonl"
    fv_prod_a = out_dir / f"fever_router_v2_always_{args.n}.jsonl"

    if scifact_u.exists() and not sc_prod_u.exists():
        apply_router_v2(scifact_u, sc_prod_u, sc_router, scifact_map, sc_ta, sc_tr, "uncertain_only", args.stage2_budget_scifact, scifact_present)
    if scifact_a.exists() and not sc_prod_a.exists():
        apply_router_v2(scifact_a, sc_prod_a, sc_router, scifact_map, sc_ta, sc_tr, "always", args.stage2_budget_scifact, scifact_present)
    if fever_u.exists() and not fv_prod_u.exists():
        apply_router_v2(fever_u, fv_prod_u, fv_router, fever_map, fv_ta, fv_tr, "uncertain_only", args.stage2_budget_fever, fever_present)
    if fever_a.exists() and not fv_prod_a.exists():
        apply_router_v2(fever_a, fv_prod_a, fv_router, fever_map, fv_ta, fv_tr, "always", args.stage2_budget_fever, fever_present)

    report = Path(args.out_md)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Product Proof Router v2 Matrix")
    lines.append("")
    for track in ["scifact", "fever"]:
        lines.append(f"## {track}")
        lines.append("")
        lines.append("| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | stage2_ran_rate | stage2_route_rate | capped_count | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | final_accept_rate | final_reject_rate | final_uncertain_rate | final_ok_rate | abstain_rate | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in [
            f"{track}_baseline_uncertain_only_{args.n}_real.jsonl",
            f"{track}_baseline_always_{args.n}_real.jsonl",
            f"{track}_router_v2_uncertain_only_{args.n}.jsonl",
            f"{track}_router_v2_always_{args.n}.jsonl",
        ]:
            path = out_dir / name
            if not path.exists():
                lines.append(f"| {name} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
                continue
            policy = "always" if "always" in name else "uncertain_only"
            s = summarize(list(iter_jsonl(path)), policy)
            lines.append("| {f} | {a} | {r} | {u} | {s2} | {s2r} | {s2route} | {cap} | {capr} | {fp} | {fn} | {ok} | {fa} | {fr} | {fu} | {fok} | {ab} | {mean} | {p95} | {rg} | {ng} |".format(
                f=name,
                a=format_float(s["accept_rate"]),
                r=format_float(s["reject_rate"]),
                u=format_float(s["uncertain_rate"]),
                s2=format_float(s["stage2_rate"]),
                s2r=format_float(s["stage2_ran_rate"]),
                s2route=format_float(s["stage2_route_rate"]),
                cap=s["capped_count"],
                capr=format_float(s["capped_rate"]),
                fp=format_float(s["fp_accept_rate"]),
                fn=format_float(s["fn_reject_rate"]),
                ok=format_float(s["ok_rate_stage1"]),
                fa=format_float(s["final_accept_rate"]),
                fr=format_float(s["final_reject_rate"]),
                fu=format_float(s["final_uncertain_rate"]),
                fok=format_float(s["final_ok_rate"]),
                ab=format_float(s["abstain_rate"]),
                mean=f"{s['mean_ms']:.2f}",
                p95=f"{s['p95_ms']:.2f}",
                rg=s["rerank_ms_gt0"],
                ng=s["nli_ms_gt0"],
            ))
        lines.append("")
    lines.append("Ship recommendation")
    lines.append("")
    for track in ["scifact", "fever"]:
        best = None
        for name in [
            f"{track}_baseline_uncertain_only_{args.n}_real.jsonl",
            f"{track}_baseline_always_{args.n}_real.jsonl",
            f"{track}_router_v2_uncertain_only_{args.n}.jsonl",
            f"{track}_router_v2_always_{args.n}.jsonl",
        ]:
            path = out_dir / name
            if not path.exists():
                continue
            policy = "always" if "always" in name else "uncertain_only"
            s = summarize(list(iter_jsonl(path)), policy)
            cost = 10.0 * s["fp_accept_rate"] + 10.0 * s["fn_reject_rate"] + 1.0 * s["stage2_ran_rate"]
            feasible = s["fp_accept_rate"] <= 0.01 and s["fn_reject_rate"] <= 0.01
            cand = (feasible, cost, name, s)
            if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):
                best = cand
        if best is None:
            lines.append(f"- {track}: no runs found")
        else:
            feasible, cost, name, s = best
            if track == "fever":
                preferred = f"{track}_baseline_uncertain_only_{args.n}_real.jsonl"
                if (out_dir / preferred).exists():
                    s_pref = summarize(list(iter_jsonl(out_dir / preferred)), "uncertain_only")
                    lines.append(f"- {track}: {preferred} cost={10.0 * s_pref['fp_accept_rate'] + 10.0 * s_pref['fn_reject_rate'] + 1.0 * s_pref['stage2_ran_rate']:.4f} fp={s_pref['fp_accept_rate']:.4f} fn={s_pref['fn_reject_rate']:.4f} stage2_ran_rate={s_pref['stage2_ran_rate']:.4f} feasible={s_pref['fp_accept_rate'] <= 0.01 and s_pref['fn_reject_rate'] <= 0.01}")
                else:
                    lines.append(f"- {track}: {name} cost={cost:.4f} fp={s['fp_accept_rate']:.4f} fn={s['fn_reject_rate']:.4f} stage2_ran_rate={s['stage2_ran_rate']:.4f} feasible={feasible}")
            else:
                lines.append(f"- {track}: {name} cost={cost:.4f} fp={s['fp_accept_rate']:.4f} fn={s['fn_reject_rate']:.4f} stage2_ran_rate={s['stage2_ran_rate']:.4f} feasible={feasible}")
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
