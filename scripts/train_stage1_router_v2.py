import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.router_v2 import build_feature_frame, fit, save


def compute_ece(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            rows.append((lo, hi, 0, 0.0, 0.0))
            continue
        yp = y_prob[mask]
        yt = y_true[mask]
        acc = float(yt.mean())
        conf = float(yp.mean())
        frac = float(mask.mean())
        ece += frac * abs(acc - conf)
        rows.append((lo, hi, int(mask.sum()), acc, conf))
    return float(ece), rows


def select_thresholds(y_true, y_prob, alphas, betas, c_fp, c_fn, c_stage2, t_reject_max=None):
    grid = np.linspace(0.0, 1.0, 201)
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    results = []
    for t_reject in grid:
        if t_reject_max is not None and t_reject > t_reject_max:
            continue
        reject_mask = p <= t_reject
        fn_rate = float(((reject_mask) & (y == 1)).mean())
        for t_accept in grid:
            if t_reject >= t_accept:
                continue
            accept_mask = p >= t_accept
            uncertain_mask = ~(accept_mask | reject_mask)
            fp_rate = float(((accept_mask) & (y == 0)).mean())
            uncertain_rate = float(uncertain_mask.mean())
            ok_rate = 1.0 - fp_rate - fn_rate
            expected_cost = c_fp * fp_rate + c_fn * fn_rate + c_stage2 * uncertain_rate
            results.append({
                "t_accept": float(t_accept),
                "t_reject": float(t_reject),
                "fp_accept_rate": fp_rate,
                "fn_reject_rate": fn_rate,
                "uncertain_rate": uncertain_rate,
                "ok_rate": ok_rate,
                "expected_cost": expected_cost,
            })
    best_by_gate = {}
    w_fp, w_fn, w_unc = 5.0, 3.0, 1.0
    for a in alphas:
        for b in betas:
            feasible = [r for r in results if r["fp_accept_rate"] <= a and r["fn_reject_rate"] <= b]
            feasible.sort(key=lambda r: (r["expected_cost"], r["uncertain_rate"]))
            if feasible:
                best_by_gate[(a, b)] = (feasible[0], 0.0, True)
                continue
            closest = []
            for r in results:
                penalty = (
                    w_fp * max(0.0, r["fp_accept_rate"] - a)
                    + w_fn * max(0.0, r["fn_reject_rate"] - b)
                    + w_unc * max(0.0, r["uncertain_rate"] - 0.3)
                )
                closest.append((penalty, r))
            closest.sort(key=lambda x: (x[0], x[1]["expected_cost"]))
            if closest:
                best_by_gate[(a, b)] = (closest[0][1], float(closest[0][0]), False)
    return best_by_gate, results


def select_feature_cols(df, logit_col):
    feature_cols = [logit_col, "abs_logit", "logit_sq", "logit_sigmoid", "logit_cube"]
    feature_cols += ["cs_ret", "top1_sim", "top2_sim", "top3_sim", "delta12", "count_sim_ge_t"]
    if "top1" in df.columns:
        feature_cols.append("top1")
    if "top2" in df.columns:
        feature_cols.append("top2")
    if "gap12" in df.columns or ("top1" in df.columns and "top2" in df.columns):
        feature_cols.append("margin")
    if "mean_top3" in df.columns or "mean_top5" in df.columns:
        feature_cols.append("topk_mean")
    if "std_top3" in df.columns or "std_top5" in df.columns:
        feature_cols.append("topk_std")
    return feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--c_stage2_list", type=float, nargs="+")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    if args.logit_col not in df.columns:
        raise SystemExit(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise SystemExit(f"missing y_col {args.y_col}")

    y = pd.to_numeric(df[args.y_col], errors="coerce")
    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    feature_cols = select_feature_cols(df, args.logit_col)
    features = build_feature_frame(df, args.logit_col, feature_cols)
    df_feat = pd.concat([features, df[[args.y_col]]], axis=1)

    train_df, val_df = train_test_split(
        df_feat, test_size=0.2, random_state=args.seed, stratify=df_feat[args.y_col]
    )

    class_weight = "balanced" if args.track == "fever" else None
    model = fit(train_df, feature_cols, args.y_col, args.logit_col, class_weight=class_weight)
    val_probs = model.predict_proba(val_df)
    val_y = val_df[args.y_col].to_numpy()
    auc = float(roc_auc_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else 0.0
    prob_flip = False
    mean_y1 = float(val_probs[val_y == 1].mean()) if (val_y == 1).any() else 0.0
    mean_y0 = float(val_probs[val_y == 0].mean()) if (val_y == 0).any() else 0.0
    if auc < 0.5:
        prob_flip = True
        val_probs = 1.0 - val_probs
        auc = 1.0 - auc
        model.prob_flip = True
    if not prob_flip and mean_y1 < mean_y0:
        prob_flip = True
        val_probs = 1.0 - val_probs
        auc = 1.0 - auc
        model.prob_flip = True
    ece, bins = compute_ece(val_y, val_probs, bins=10)

    full_probs = model.predict_proba(df_feat)
    if prob_flip:
        full_probs = 1.0 - full_probs
    full_y = df_feat[args.y_col].to_numpy()
    t_reject_max = None
    if (full_y == 1).any():
        t_reject_max = float(np.quantile(full_probs[full_y == 1], 0.01))

    alphas = [0.01, 0.02]
    betas = [0.01, 0.02]
    c_fp = 10.0
    c_fn = 10.0
    c_stage2_list = args.c_stage2_list if args.c_stage2_list else [1.0]
    default_gate = (0.01, 0.01)
    candidates = []
    for c_stage2 in c_stage2_list:
        best_by_gate, results = select_thresholds(full_y, full_probs, alphas, betas, c_fp, c_fn, c_stage2, t_reject_max=t_reject_max)
        best, penalty, feasible = best_by_gate.get(default_gate, (None, None, None))
        if best is not None and best["fn_reject_rate"] > 0.02:
            filtered = [r for r in results if r["fn_reject_rate"] <= 0.02]
            filtered.sort(key=lambda r: (r["expected_cost"], r["uncertain_rate"]))
            if filtered:
                best = filtered[0]
                penalty = 0.0
                feasible = best["fp_accept_rate"] <= default_gate[0] and best["fn_reject_rate"] <= default_gate[1]
            else:
                raise SystemExit("no thresholds satisfy fn_reject_rate <= 0.02")
        if best is None:
            continue
        candidates.append((c_stage2, best, penalty, feasible))
    if not candidates:
        raise SystemExit("no threshold candidates")
    candidates.sort(key=lambda x: (not x[3], x[1]["expected_cost"], x[1]["uncertain_rate"]))
    ship_c, best, penalty, feasible = candidates[0]
    if best is None:
        raise SystemExit("no threshold candidates")
    if best["t_accept"] <= best["t_reject"]:
        raise SystemExit("invalid thresholds: t_accept must be > t_reject")
    model.t_accept = float(best["t_accept"])
    model.t_reject = float(best["t_reject"])

    save(model, args.out_model)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Stage1 Router v2 Train ({args.track})")
    lines.append("")
    lines.append(f"Generated: `{dt.datetime.utcnow().isoformat()}+00:00`")
    lines.append("")
    lines.append("Config")
    lines.append("")
    lines.append(f"- in_path: `{args.in_path}`")
    lines.append(f"- n: `{len(df_feat)}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- feature_cols: `{', '.join(feature_cols)}`")
    lines.append(f"- out_model: `{args.out_model}`")
    lines.append(f"- threshold_selection: `full_data`")
    lines.append("")
    lines.append("Validation")
    lines.append("")
    lines.append(f"- auc: `{auc:.4f}`")
    lines.append(f"- ece: `{ece:.4f}`")
    lines.append(f"- mean_prob_y1: `{mean_y1:.4f}`")
    lines.append(f"- mean_prob_y0: `{mean_y0:.4f}`")
    lines.append("")
    lines.append("Thresholds")
    lines.append("")
    lines.append(f"- alpha_list: `{', '.join([str(a) for a in alphas])}`")
    lines.append(f"- beta_list: `{', '.join([str(b) for b in betas])}`")
    lines.append(f"- selected_gate: `fp<=0.01 fn<=0.01`")
    lines.append(f"- gate_feasible: `{feasible}`")
    lines.append(f"- gate_penalty: `{penalty:.4f}`")
    lines.append(f"- c_fp: `{c_fp:.2f}`")
    lines.append(f"- c_fn: `{c_fn:.2f}`")
    lines.append(f"- c_stage2_list: `{', '.join([str(x) for x in c_stage2_list])}`")
    lines.append(f"- ship_c_stage2: `{ship_c:.2f}`")
    lines.append(f"- prob_flip: `{prob_flip}`")
    lines.append(f"- t_reject_max: `{t_reject_max}`")
    stage2_budget = 0.45 if args.track == "scifact" else None
    lines.append(f"- stage2_budget: `{stage2_budget}`")
    lines.append(f"- t_accept: `{best['t_accept']:.4f}`")
    lines.append(f"- t_reject: `{best['t_reject']:.4f}`")
    lines.append(f"- fp_accept_rate: `{best['fp_accept_rate']:.4f}`")
    lines.append(f"- fn_reject_rate: `{best['fn_reject_rate']:.4f}`")
    lines.append(f"- uncertain_rate: `{best['uncertain_rate']:.4f}`")
    lines.append(f"- stage2_route_rate: `{best['uncertain_rate']:.4f}`")
    lines.append(f"- expected_cost: `{best['expected_cost']:.4f}`")
    lines.append(f"- ok_rate: `{best['ok_rate']:.4f}`")
    lines.append("")
    lines.append("Per c_stage2")
    lines.append("")
    lines.append("| c_stage2 | t_accept | t_reject | fp_accept_rate | fn_reject_rate | uncertain_rate | stage2_route_rate | expected_cost | feasible | penalty |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c_stage2, b, pen, feas in candidates:
        lines.append("| {c} | {ta} | {tr} | {fp} | {fn} | {u} | {sr} | {ec} | {f} | {p} |".format(
            c=f"{c_stage2:.2f}",
            ta=f"{b['t_accept']:.4f}",
            tr=f"{b['t_reject']:.4f}",
            fp=f"{b['fp_accept_rate']:.4f}",
            fn=f"{b['fn_reject_rate']:.4f}",
            u=f"{b['uncertain_rate']:.4f}",
            sr=f"{b['uncertain_rate']:.4f}",
            ec=f"{b['expected_cost']:.4f}",
            f=str(feas),
            p=f"{pen:.4f}",
        ))
    lines.append("")
    lines.append("Calibration bins")
    lines.append("")
    lines.append("| bin | count | avg_prob | frac_pos |")
    lines.append("|---|---:|---:|---:|")
    for i, (lo, hi, count, acc, conf) in enumerate(bins):
        lines.append("| {b} | {c} | {p:.4f} | {a:.4f} |".format(
            b=f"[{lo:.2f},{hi:.2f})",
            c=count,
            p=conf,
            a=acc,
        ))
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
