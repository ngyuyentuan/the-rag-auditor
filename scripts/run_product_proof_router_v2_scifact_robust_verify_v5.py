import argparse
import json
import math
import random
import secrets
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_product_proof_router_v2_matrix import apply_router_v2, iter_jsonl, summarize
from src.stage1.router_v2 import load, build_feature_frame


def ensure_new_dir(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"output path exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def unique_run_id():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"robust_verify_{stamp}_{suffix}"


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
    cols = set([logit_col, "qid", "y"])
    for col in feature_cols:
        if col in available:
            cols.add(col)
    for name in ["raw_max_top3", "gap12", "mean_top3", "mean_top5", "std_top3", "std_top5", "top1", "top2", "top3", "cs_ret"]:
        if name in available:
            cols.add(name)
    return [c for c in cols if c in available]


def load_feature_frame(parquet_path, logit_col, feature_cols, fit_n, fit_seed):
    import pandas as pd
    available = get_parquet_columns(parquet_path)
    cols = required_columns(feature_cols, logit_col, available)
    df = pd.read_parquet(parquet_path, columns=cols)
    if "qid" not in df.columns:
        raise SystemExit("missing qid in parquet")
    if "y" not in df.columns:
        raise SystemExit("missing y in parquet")
    if fit_n is not None and fit_n < len(df):
        rng = np.random.RandomState(fit_seed)
        df_pos = df[df["y"] == 1]
        df_neg = df[df["y"] == 0]
        n_pos = int(round(fit_n * (len(df_pos) / len(df)))) if len(df) > 0 else 0
        n_neg = max(0, fit_n - n_pos)
        pos_idx = rng.choice(len(df_pos), size=min(n_pos, len(df_pos)), replace=False) if len(df_pos) else []
        neg_idx = rng.choice(len(df_neg), size=min(n_neg, len(df_neg)), replace=False) if len(df_neg) else []
        df = pd.concat([df_pos.iloc[pos_idx], df_neg.iloc[neg_idx]], ignore_index=True)
    df["qid"] = df["qid"].astype(str)
    feat_df = build_feature_frame(df, logit_col, feature_cols)
    return df, feat_df


def build_feature_map_subset(parquet_path, logit_col, feature_cols, qid_set):
    import pandas as pd
    available = get_parquet_columns(parquet_path)
    cols = required_columns(feature_cols, logit_col, available)
    try:
        df = pd.read_parquet(parquet_path, columns=cols, filters=[("qid", "in", list(qid_set))])
    except Exception:
        df = pd.read_parquet(parquet_path, columns=cols)
    if "qid" not in df.columns:
        raise SystemExit("missing qid in parquet")
    df["qid"] = df["qid"].astype(str)
    feat_df = build_feature_frame(df, logit_col, feature_cols)
    qids = df["qid"].astype(str).tolist()
    feats = feat_df.to_numpy()
    present = {}
    for col in feature_cols:
        present[col] = col in df.columns or col in feat_df.columns
    return dict(zip(qids, feats)), present


def ensemble_probs(models, flip_flags, x):
    acc = None
    for model, flip in zip(models, flip_flags):
        p = model.predict_proba(x)[:, 1]
        if flip:
            p = 1.0 - p
        acc = p if acc is None else acc + p
    return acc / float(len(models))


def fit_thresholds(p_hat, y, alpha, beta):
    qs = np.linspace(0.0, 1.0, 101)
    candidates = np.unique(np.quantile(p_hat, qs))
    best = None
    for t_accept in candidates:
        for t_reject in candidates:
            if t_reject >= t_accept:
                continue
            accept = p_hat >= t_accept
            reject = p_hat <= t_reject
            uncertain = ~(accept | reject)
            fp = float(np.mean(accept & (y == 0)))
            fn = float(np.mean(reject & (y == 1)))
            if fp > alpha or fn > beta:
                continue
            unc = float(np.mean(uncertain))
            ok = 1.0 - fp - fn
            cand = (unc, -ok, t_accept, t_reject)
            if best is None or cand < best[0]:
                best = (cand, {"t_accept": float(t_accept), "t_reject": float(t_reject), "fp": fp, "fn": fn, "unc": unc, "ok": ok})
    if best is None:
        raise SystemExit("no feasible thresholds for alpha/beta")
    return best[1]


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
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 12, 21, 33, 42])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.26, 0.28, 0.30])
    ap.add_argument("--fit_n", type=int, default=20000)
    ap.add_argument("--fit_seed", type=int, default=14)
    ap.add_argument("--out_md")
    args = ap.parse_args()

    baseline_path = Path(args.baseline_jsonl)
    if not baseline_path.exists():
        raise SystemExit(f"missing baseline_jsonl {baseline_path}")
    parquet_path = Path(args.scifact_in_path)
    if not parquet_path.exists():
        raise SystemExit(f"missing scifact_in_path {parquet_path}")
    model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    if not model_paths:
        raise SystemExit("missing models")
    for p in model_paths:
        if not Path(p).exists():
            raise SystemExit(f"missing model {p}")

    run_id = unique_run_id()
    models_dir = ensure_new_dir(Path("models") / "sweeps" / run_id)
    runs_dir = ensure_new_dir(Path("runs") / "sweeps" / run_id)
    reports_dir = ensure_new_dir(Path("reports") / "sweeps" / run_id)

    routers = [load(p) for p in model_paths]
    base = routers[0]
    for r in routers[1:]:
        if r.logit_col != base.logit_col:
            raise SystemExit("ensemble logit_col mismatch")
        if r.feature_cols != base.feature_cols:
            raise SystemExit("ensemble feature_cols mismatch")
    flip_flags = [bool(r.prob_flip) for r in routers]

    df_fit, feat_fit = load_feature_frame(parquet_path, base.logit_col, base.feature_cols, args.fit_n, args.fit_seed)
    x_fit = feat_fit.to_numpy()
    y_fit = df_fit["y"].astype(int).to_numpy()
    p_hat = ensemble_probs([r.model for r in routers], flip_flags, x_fit)
    fitted = fit_thresholds(p_hat, y_fit, 0.01, 0.01)

    class EnsembleModel:
        def __init__(self, models, flips):
            self.models = models
            self.flips = flips
        def predict_proba(self, x):
            p = ensemble_probs(self.models, self.flips, x)
            return np.vstack([1.0 - p, p]).T

    class RouterWrapper:
        pass

    wrapper = RouterWrapper()
    wrapper.model = EnsembleModel([r.model for r in routers], flip_flags)
    wrapper.feature_cols = list(base.feature_cols)
    wrapper.logit_col = base.logit_col
    wrapper.t_accept = fitted["t_accept"]
    wrapper.t_reject = fitted["t_reject"]
    wrapper.prob_flip = False

    baseline_rows = list(iter_jsonl(baseline_path))
    qid_set = set()
    for row in baseline_rows:
        qid = str((row.get("metadata") or {}).get("qid"))
        qid_set.add(qid)
    feature_map, features_present = build_feature_map_subset(parquet_path, base.logit_col, base.feature_cols, qid_set)

    all_rows = []
    for seed in args.seeds:
        sampled = sample_rows(baseline_rows, args.n, seed)
        sample_path = runs_dir / f"baseline_seed_{seed}_{args.n}.jsonl"
        write_jsonl_create(sample_path, sampled)
        for budget in args.budgets:
            out_path = runs_dir / f"scifact_ens_seed{seed}_budget{budget:.3f}.jsonl"
            if out_path.exists():
                raise SystemExit(f"output path exists: {out_path}")
            apply_router_v2(sample_path, out_path, wrapper, feature_map, fitted["t_accept"], fitted["t_reject"], "uncertain_only", budget, features_present)
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
    lines.append(f"- models: `{','.join(model_paths)}`")
    lines.append(f"- baseline_jsonl: `{baseline_path}`")
    lines.append(f"- scifact_in_path: `{parquet_path}`")
    lines.append(f"- fitted_t_accept: {fitted['t_accept']:.4f}")
    lines.append(f"- fitted_t_reject: {fitted['t_reject']:.4f}")
    lines.append(f"- fit_n: {args.fit_n}")
    lines.append(f"- fit_seed: {args.fit_seed}")
    lines.append("runtime mitigation: thresholds were fit on fit_n subsample; robust verify metrics are computed from n=200 runs per seed/budget")
    lines.append("")
    lines.append("## Per-seed metrics")
    lines.append("")
    lines.append("| seed | budget | accept_rate | reject_rate | uncertain_rate | stage2_route_rate | stage2_ran_rate | capped_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        lines.append("| {seed} | {b:.3f} | {a:.4f} | {rj:.4f} | {u:.4f} | {route:.4f} | {ran:.4f} | {cap:.4f} | {fp:.4f} | {fn:.4f} | {ok:.4f} | {mean:.2f} | {p95:.2f} |".format(
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
            penalty = max(0.0, row["fp_worst"] - fp_max) + max(0.0, row["fn_worst"] - fn_max) + max(0.0, row["ran_worst"] - ran_max) + max(0.0, row["cap_worst"] - cap_max)
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
