import argparse
import json
import math
import os
import random
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_product_proof_router_v2_matrix import apply_router_v2, iter_jsonl, summarize
from src.stage1.router_v2 import load


def ensure_new_dir(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def unique_run_id():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{stamp}_{suffix}"


def write_jsonl_create(path, rows):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"temp path exists: {tmp}")
    with tmp.open("x", encoding="utf-8") as f_out:
        for row in rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    tmp.rename(path)
    return path


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


def sample_rows(rows, n, seed):
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(rows)), n)
    return [rows[i] for i in idxs]


def get_parquet_columns(path):
    try:
        import pyarrow.parquet as pq
        return [n for n in pq.ParquetFile(path).schema.names]
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(path, columns=[])
            return list(df.columns)
        except Exception:
            import pandas as pd
            df = pd.read_parquet(path)
            return list(df.columns)


def required_columns(feature_cols, logit_col, available):
    cols = set([logit_col, "qid"])
    def add_if_present(name):
        if name in available:
            cols.add(name)
    for col in feature_cols:
        if col in available:
            cols.add(col)
    for name in ["raw_max_top3", "gap12", "mean_top3", "mean_top5", "std_top3", "std_top5", "top1", "top2", "top3", "cs_ret"]:
        add_if_present(name)
    return [c for c in cols if c in available]


def build_feature_map_subset(parquet_path, logit_col, feature_cols, qid_set):
    import pandas as pd
    available = get_parquet_columns(parquet_path)
    cols = required_columns(feature_cols, logit_col, available)
    df = None
    if "qid" in cols:
        try:
            df = pd.read_parquet(parquet_path, columns=cols, filters=[("qid", "in", list(qid_set))])
        except Exception:
            df = pd.read_parquet(parquet_path, columns=cols)
    else:
        df = pd.read_parquet(parquet_path, columns=cols)
    if "qid" not in df.columns:
        raise SystemExit("missing qid in parquet")
    df["qid"] = df["qid"].astype(str)
    from src.stage1.router_v2 import build_feature_frame
    feat_df = build_feature_frame(df, logit_col, feature_cols)
    qids = df["qid"].astype(str).tolist()
    feats = feat_df.to_numpy()
    present = {}
    for col in feature_cols:
        present[col] = col in df.columns or col in feat_df.columns
    return dict(zip(qids, feats)), present


def aggregate_budget_rows(rows):
    grouped = {}
    for r in rows:
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
            "fp_mean": mean("fp_accept_rate"),
            "fp_worst": worst("fp_accept_rate"),
            "fn_mean": mean("fn_reject_rate"),
            "fn_worst": worst("fn_reject_rate"),
            "ran_mean": mean("stage2_ran_rate"),
            "ran_worst": worst("stage2_ran_rate"),
            "cap_mean": mean("capped_rate"),
            "cap_worst": worst("capped_rate"),
            "ok_mean": mean("ok_rate_stage1"),
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
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 12, 21, 33, 42])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.26, 0.28, 0.30])
    ap.add_argument("--out_md")
    args = ap.parse_args()

    baseline_path = Path(args.baseline_jsonl)
    if not baseline_path.exists():
        raise SystemExit(f"missing baseline_jsonl {baseline_path}")
    parquet_path = Path(args.scifact_in_path)
    if not parquet_path.exists():
        raise SystemExit(f"missing scifact_in_path {parquet_path}")
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"missing model {model_path}")

    run_id = unique_run_id()
    models_dir = ensure_new_dir(Path("models") / "sweeps" / run_id)
    runs_dir = ensure_new_dir(Path("runs") / "sweeps" / run_id)
    reports_dir = ensure_new_dir(Path("reports") / "sweeps" / run_id)

    if args.out_md:
        out_md = Path(args.out_md)
        if reports_dir not in out_md.parents:
            raise SystemExit("out_md must be under reports/sweeps/<run_id>/")
    else:
        out_md = reports_dir / "product_proof_router_v2_scifact_robust_verify.md"

    if out_md.exists():
        raise SystemExit(f"output path exists: {out_md}")

    baseline_rows = list(iter_jsonl(baseline_path))
    router = load(model_path)
    qid_set = set()
    for row in baseline_rows:
        qid = str((row.get("metadata") or {}).get("qid"))
        qid_set.add(qid)
    feature_map, features_present = build_feature_map_subset(parquet_path, router.logit_col, router.feature_cols, qid_set)
    t_accept = float(router.t_accept)
    t_reject = float(router.t_reject)

    all_rows = []
    for seed in args.seeds:
        sampled = sample_rows(baseline_rows, args.n, seed)
        sample_path = runs_dir / f"baseline_seed_{seed}_{args.n}.jsonl"
        write_jsonl_create(sample_path, sampled)
        for budget in args.budgets:
            out_path = runs_dir / f"router_v2_budget_{budget:.2f}_seed_{seed}_{args.n}.jsonl"
            if out_path.exists():
                raise SystemExit(f"output path exists: {out_path}")
            apply_router_v2(sample_path, out_path, router, feature_map, t_accept, t_reject, "uncertain_only", budget, features_present)
            stats = summarize(list(iter_jsonl(out_path)), "uncertain_only")
            stats["seed"] = seed
            stats["budget"] = budget
            all_rows.append(stats)

    agg = aggregate_budget_rows(all_rows)
    fp_max = 0.01
    fn_max = 0.01
    ran_max = 0.25
    cap_max = 0.20
    best = select_ship(agg, fp_max, fn_max, ran_max, cap_max)

    lines = []
    lines.append("# Product Proof Router v2 SciFact Robust Verify")
    lines.append("")
    lines.append(f"- model: `{model_path}`")
    lines.append(f"- baseline_jsonl: `{baseline_path}`")
    lines.append(f"- scifact_in_path: `{parquet_path}`")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- budgets: {args.budgets}")
    lines.append("")
    lines.append("## Per-seed metrics")
    lines.append("")
    lines.append("| seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        lines.append("| {seed} | {b:.2f} | {a:.4f} | {rj:.4f} | {u:.4f} | {route:.4f} | {ran:.4f} | {cap:.4f} | {fp:.4f} | {fn:.4f} | {ok:.4f} | {mean:.2f} | {p95:.2f} |".format(
            seed=r["seed"],
            b=r["budget"],
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
    lines.append("| budget | fp_mean | fp_worst | fn_mean | fn_worst | ran_mean | ran_worst | cap_mean | cap_worst | ok_mean | p95_worst |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for budget in sorted(agg.keys()):
        row = agg[budget]
        lines.append("| {b:.2f} | {fp_m:.4f} | {fp_w:.4f} | {fn_m:.4f} | {fn_w:.4f} | {ran_m:.4f} | {ran_w:.4f} | {cap_m:.4f} | {cap_w:.4f} | {ok:.4f} | {p95:.2f} |".format(
            b=row["budget"],
            fp_m=row["fp_mean"],
            fp_w=row["fp_worst"],
            fn_m=row["fn_mean"],
            fn_w=row["fn_worst"],
            ran_m=row["ran_mean"],
            ran_w=row["ran_worst"],
            cap_m=row["cap_mean"],
            cap_w=row["cap_worst"],
            ok=row["ok_mean"],
            p95=row["p95_worst"],
        ))
    lines.append("")
    lines.append("## Ship recommendation")
    lines.append("")
    if best is None:
        lines.append("- no budget met fp<=0.01 fn<=0.01 stage2_ran_rate<=0.25 capped_rate<=0.20 on worst-case")
        penalties = []
        for budget, row in agg.items():
            penalty = max(0.0, row["fp_worst"] - fp_max) + max(0.0, row["fn_worst"] - fn_max) + max(0.0, row["ran_worst"] - ran_max) + max(0.0, row["cap_worst"] - cap_max)
            penalties.append((penalty, budget))
        if penalties:
            penalties.sort()
            _, budget = penalties[0]
            row = agg[budget]
            lines.append("- closest: budget={b:.2f} fp_worst={fp:.4f} fn_worst={fn:.4f} ran_worst={ran:.4f} cap_worst={cap:.4f} ok_mean={ok:.4f}".format(
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
        lines.append("- budget={b:.2f} fp_worst={fp:.4f} fn_worst={fn:.4f} ran_worst={ran:.4f} cap_worst={cap:.4f} ok_mean={ok:.4f}".format(
            b=row["budget"],
            fp=row["fp_worst"],
            fn=row["fn_worst"],
            ran=row["ran_worst"],
            cap=row["cap_worst"],
            ok=row["ok_mean"],
        ))

    write_text_create(out_md, "\n".join(lines))
    print(f"run_id={run_id}")
    print(f"models_dir={models_dir}")
    print(f"runs_dir={runs_dir}")
    print(f"reports_dir={reports_dir}")
    print(f"report={out_md}")


if __name__ == "__main__":
    main()
