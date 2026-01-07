import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution, stage1_outcomes


def load_thresholds(path, track):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid thresholds yaml: {path}")
    block = data.get("thresholds") or {}
    cfg = block.get(track)
    if not isinstance(cfg, dict):
        if all(k in data for k in ["t_lower", "t_upper", "tau"]):
            cfg = data
        else:
            raise SystemExit(f"missing thresholds for track={track} in {path}")
    t_lower = cfg.get("t_lower")
    t_upper = cfg.get("t_upper")
    tau = cfg.get("tau")
    if t_lower is None or t_upper is None or tau is None:
        raise SystemExit(f"incomplete thresholds for track={track} in {path}")
    return float(t_lower), float(t_upper), float(tau)


def bootstrap_ci(decisions, y, n_boot, seed):
    if n_boot <= 0:
        return {}
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y, dtype=int)
    dec_arr = np.asarray(decisions)
    acc = []
    rej = []
    unc = []
    fp = []
    fn = []
    ok = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d = dec_arr[idx]
        yb = y_arr[idx]
        acc.append(float(np.mean(d == "ACCEPT")) if n else 0.0)
        rej.append(float(np.mean(d == "REJECT")) if n else 0.0)
        unc.append(float(np.mean(d == "UNCERTAIN")) if n else 0.0)
        fp_rate = float(np.mean((d == "ACCEPT") & (yb == 0))) if n else 0.0
        fn_rate = float(np.mean((d == "REJECT") & (yb == 1))) if n else 0.0
        fp.append(fp_rate)
        fn.append(fn_rate)
        ok.append(1.0 - fp_rate - fn_rate)
    def pct(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return {
        "accept_rate": pct(acc),
        "reject_rate": pct(rej),
        "uncertain_rate": pct(unc),
        "fp_accept_rate": pct(fp),
        "fn_reject_rate": pct(fn),
        "ok_rate": pct(ok),
    }


def bootstrap_delta_ci(dec_a, dec_b, y, n_boot, seed):
    if n_boot <= 0:
        return {}
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y, dtype=int)
    a = np.asarray(dec_a)
    b = np.asarray(dec_b)
    delta = {"ok_rate": [], "fp_accept_rate": [], "fn_reject_rate": [], "uncertain_rate": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        da = a[idx]
        db = b[idx]
        yb = y_arr[idx]
        fp_a = float(np.mean((da == "ACCEPT") & (yb == 0))) if n else 0.0
        fn_a = float(np.mean((da == "REJECT") & (yb == 1))) if n else 0.0
        fp_b = float(np.mean((db == "ACCEPT") & (yb == 0))) if n else 0.0
        fn_b = float(np.mean((db == "REJECT") & (yb == 1))) if n else 0.0
        ok_a = 1.0 - fp_a - fn_a
        ok_b = 1.0 - fp_b - fn_b
        unc_a = float(np.mean(da == "UNCERTAIN")) if n else 0.0
        unc_b = float(np.mean(db == "UNCERTAIN")) if n else 0.0
        delta["ok_rate"].append(ok_b - ok_a)
        delta["fp_accept_rate"].append(fp_b - fp_a)
        delta["fn_reject_rate"].append(fn_b - fn_a)
        delta["uncertain_rate"].append(unc_b - unc_a)
    out = {}
    for k, vals in delta.items():
        out[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


def compute_metrics(cs_ret, y, t_lower, t_upper):
    decisions = []
    reasons = []
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    for p, yi in zip(cs_ret, y):
        d, r = decide_route(float(p), t_lower, t_upper)
        decisions.append(d)
        reasons.append(r)
        o = stage1_outcomes(d, yi)
        for k in counts:
            counts[k] += o[k]
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    y_arr = np.asarray(y, dtype=int)
    dec_arr = np.asarray(decisions)
    fp = float(np.mean((dec_arr == "ACCEPT") & (y_arr == 0))) if len(y_arr) else 0.0
    fn = float(np.mean((dec_arr == "REJECT") & (y_arr == 1))) if len(y_arr) else 0.0
    ok = 1.0 - fp - fn
    return {
        "decisions": decisions,
        "reasons": reasons,
        "dist": dist,
        "fp": fp,
        "fn": fn,
        "ok": ok,
        "counts": counts,
    }


def cs_ret_summary(cs_ret):
    arr = np.asarray(cs_ret, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def proximity_counts(cs_ret, t_lower, t_upper, eps=0.01):
    arr = np.asarray(cs_ret, dtype=float)
    near_lower = float(np.mean(np.abs(arr - t_lower) <= eps)) if len(arr) else 0.0
    near_upper = float(np.mean(np.abs(arr - t_upper) <= eps)) if len(arr) else 0.0
    return near_lower, near_upper


def histogram_table(cs_ret, bins=20):
    arr = np.asarray(cs_ret, dtype=float)
    hist, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    total = float(np.sum(hist)) if len(arr) else 0.0
    rows = []
    for i in range(len(hist)):
        left = edges[i]
        right = edges[i + 1]
        count = int(hist[i])
        frac = (count / total) if total > 0 else 0.0
        rows.append((left, right, count, frac))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["scifact", "fever"])
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--tuned_thresholds_yaml")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"missing in_path: {in_path}")

    tuned_path = None
    if args.tuned_thresholds_yaml:
        tuned_path = Path(args.tuned_thresholds_yaml)
    else:
        candidate = Path(f"configs/thresholds_stage1_tuned_{args.track}.yaml")
        tuned_path = candidate if candidate.exists() else None

    df = pd.read_parquet(in_path)
    missing_logit = int(df[args.logit_col].isna().sum()) if args.logit_col in df.columns else None
    y_raw = df[args.y_col] if args.y_col in df.columns else None
    y_num = pd.to_numeric(y_raw, errors="coerce") if y_raw is not None else None
    missing_y = int(y_num.isna().sum()) if y_num is not None else None
    unique_y = sorted(set(y_num.dropna().astype(int).tolist())) if y_num is not None else []
    nonfinite_logit = int((~np.isfinite(pd.to_numeric(df[args.logit_col], errors="coerce"))).sum()) if args.logit_col in df.columns else None

    if args.logit_col not in df.columns or args.y_col not in df.columns:
        raise SystemExit("missing required columns in parquet")

    y = pd.to_numeric(df[args.y_col], errors="coerce")
    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)

    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)

    base_t_lower, base_t_upper, base_tau = load_thresholds(args.thresholds_yaml, args.track)
    tau = float(args.tau) if args.tau is not None else base_tau

    logits = df[args.logit_col].astype(float).to_numpy()
    y_vals = df[args.y_col].astype(int).to_numpy()
    cs_ret = [compute_cs_ret_from_logit(x, tau) for x in logits]

    base = compute_metrics(cs_ret, y_vals, base_t_lower, base_t_upper)
    base_ci = bootstrap_ci(base["decisions"], y_vals, args.bootstrap, args.seed)

    tuned = None
    tuned_ci = None
    delta_ci = None
    tuned_cfg = None
    if tuned_path and tuned_path.exists():
        t_lower_tuned, t_upper_tuned, tau_tuned = load_thresholds(tuned_path, args.track)
        if args.tau is not None:
            tau_tuned = float(args.tau)
        cs_ret_tuned = [compute_cs_ret_from_logit(x, tau_tuned) for x in logits]
        tuned = compute_metrics(cs_ret_tuned, y_vals, t_lower_tuned, t_upper_tuned)
        tuned_ci = bootstrap_ci(tuned["decisions"], y_vals, args.bootstrap, args.seed + 1)
        delta_ci = bootstrap_delta_ci(base["decisions"], tuned["decisions"], y_vals, args.bootstrap, args.seed + 2)
        tuned_cfg = (t_lower_tuned, t_upper_tuned, tau_tuned)

    cs_summary = cs_ret_summary(cs_ret)
    near_lower, near_upper = proximity_counts(cs_ret, base_t_lower, base_t_upper, eps=0.01)
    near_lower_005, near_upper_005 = proximity_counts(cs_ret, base_t_lower, base_t_upper, eps=0.005)
    hist_rows = histogram_table(cs_ret, bins=20)

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_regression_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Regression Diagnosis ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n_total: {len(df)}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- pos_rate: {float(np.mean(y_vals)):.4f}")
    lines.append(f"- missing_y: {missing_y}")
    lines.append(f"- unique_y_values: {unique_y}")
    lines.append(f"- missing_logit: {missing_logit}")
    lines.append(f"- nonfinite_logit_count: {nonfinite_logit}")
    lines.append("")
    lines.append("## cs_ret summary")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k in ["min", "p01", "p10", "p50", "p90", "p99", "max"]:
        lines.append(f"| {k} | {cs_summary[k]:.4f} |")
    lines.append("")
    lines.append("## cs_ret histogram (20 bins)")
    lines.append("| bin_left | bin_right | count | fraction |")
    lines.append("|---:|---:|---:|---:|")
    for left, right, count, frac in hist_rows:
        lines.append(f"| {left:.3f} | {right:.3f} | {count} | {frac:.4f} |")
    lines.append("")
    lines.append("## Threshold proximity counts")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| within_0.005_t_lower | {near_lower_005:.4f} |")
    lines.append(f"| within_0.005_t_upper | {near_upper_005:.4f} |")
    lines.append(f"| within_0.01_t_lower | {near_lower:.4f} |")
    lines.append(f"| within_0.01_t_upper | {near_upper:.4f} |")
    lines.append("")
    lines.append("## Baseline config")
    lines.append(f"- thresholds_yaml: `{args.thresholds_yaml}`")
    lines.append(f"- t_lower: {base_t_lower}")
    lines.append(f"- t_upper: {base_t_upper}")
    lines.append(f"- tau: {tau}")
    lines.append("")
    lines.append("| metric | value | 95% CI |")
    lines.append("|---|---:|---:|")
    lines.append(f"| accept_rate | {base['dist']['ACCEPT']:.4f} | [{base_ci['accept_rate'][0]:.4f}, {base_ci['accept_rate'][1]:.4f}] |")
    lines.append(f"| reject_rate | {base['dist']['REJECT']:.4f} | [{base_ci['reject_rate'][0]:.4f}, {base_ci['reject_rate'][1]:.4f}] |")
    lines.append(f"| uncertain_rate | {base['dist']['UNCERTAIN']:.4f} | [{base_ci['uncertain_rate'][0]:.4f}, {base_ci['uncertain_rate'][1]:.4f}] |")
    lines.append(f"| fp_accept_rate | {base['fp']:.4f} | [{base_ci['fp_accept_rate'][0]:.4f}, {base_ci['fp_accept_rate'][1]:.4f}] |")
    lines.append(f"| fn_reject_rate | {base['fn']:.4f} | [{base_ci['fn_reject_rate'][0]:.4f}, {base_ci['fn_reject_rate'][1]:.4f}] |")
    lines.append(f"| ok_rate | {base['ok']:.4f} | [{base_ci['ok_rate'][0]:.4f}, {base_ci['ok_rate'][1]:.4f}] |")
    lines.append("")
    lines.append("| count | value |")
    lines.append("|---|---:|")
    for k in ["tp", "fp", "tn", "fn", "defer_pos", "defer_neg"]:
        lines.append(f"| {k} | {base['counts'][k]} |")

    if tuned is not None:
        lines.append("")
        lines.append("## Tuned config")
        lines.append(f"- tuned_thresholds_yaml: `{tuned_path}`")
        lines.append(f"- t_lower: {tuned_cfg[0]}")
        lines.append(f"- t_upper: {tuned_cfg[1]}")
        lines.append(f"- tau: {tuned_cfg[2]}")
        lines.append("")
        lines.append("| metric | value | 95% CI |")
        lines.append("|---|---:|---:|")
        lines.append(f"| accept_rate | {tuned['dist']['ACCEPT']:.4f} | [{tuned_ci['accept_rate'][0]:.4f}, {tuned_ci['accept_rate'][1]:.4f}] |")
        lines.append(f"| reject_rate | {tuned['dist']['REJECT']:.4f} | [{tuned_ci['reject_rate'][0]:.4f}, {tuned_ci['reject_rate'][1]:.4f}] |")
        lines.append(f"| uncertain_rate | {tuned['dist']['UNCERTAIN']:.4f} | [{tuned_ci['uncertain_rate'][0]:.4f}, {tuned_ci['uncertain_rate'][1]:.4f}] |")
        lines.append(f"| fp_accept_rate | {tuned['fp']:.4f} | [{tuned_ci['fp_accept_rate'][0]:.4f}, {tuned_ci['fp_accept_rate'][1]:.4f}] |")
        lines.append(f"| fn_reject_rate | {tuned['fn']:.4f} | [{tuned_ci['fn_reject_rate'][0]:.4f}, {tuned_ci['fn_reject_rate'][1]:.4f}] |")
        lines.append(f"| ok_rate | {tuned['ok']:.4f} | [{tuned_ci['ok_rate'][0]:.4f}, {tuned_ci['ok_rate'][1]:.4f}] |")
        lines.append("")
        lines.append("| count | value |")
        lines.append("|---|---:|")
        for k in ["tp", "fp", "tn", "fn", "defer_pos", "defer_neg"]:
            lines.append(f"| {k} | {tuned['counts'][k]} |")
        lines.append("")
        lines.append("## Route flips (tuned vs baseline)")
        base_dec = np.asarray(base["decisions"], dtype=object)
        tuned_dec = np.asarray(tuned["decisions"], dtype=object)
        flip_mask = base_dec != tuned_dec
        flip_count = int(np.sum(flip_mask))
        lines.append(f"- flips_total: {flip_count}")
        if flip_count > 0:
            qids = df["qid"].astype(str).tolist() if "qid" in df.columns else [str(i) for i in range(len(df))]
            logit_vals = df[args.logit_col].astype(float).tolist()
            y_list = df[args.y_col].astype(int).tolist()
            tuned_cs = cs_ret
            dist_base = np.minimum(np.abs(np.asarray(cs_ret) - base_t_lower), np.abs(np.asarray(cs_ret) - base_t_upper))
            if tuned_cfg:
                dist_tuned = np.minimum(np.abs(np.asarray(tuned_cs) - tuned_cfg[0]), np.abs(np.asarray(tuned_cs) - tuned_cfg[1]))
            else:
                dist_tuned = dist_base
            dist_min = np.minimum(dist_base, dist_tuned)
            flip_idx = np.where(flip_mask)[0]
            order = flip_idx[np.argsort(dist_min[flip_idx])]
            base_reason = np.asarray(base["reasons"], dtype=object)
            tuned_reason = np.asarray(tuned["reasons"], dtype=object)
            lines.append("| qid | y | logit | cs_ret | baseline_decision | tuned_decision | baseline_reason | tuned_reason |")
            lines.append("|---|---:|---:|---:|---|---|---|---|")
            for i in order[:25]:
                lines.append(f"| {qids[i]} | {y_list[i]} | {logit_vals[i]:.6f} | {tuned_cs[i]:.6f} | {base_dec[i]} | {tuned_dec[i]} | {base_reason[i]} | {tuned_reason[i]} |")
            lines.append("")
        else:
            lines.append("no flips observed")
            lines.append("")
        lines.append("## Bootstrap delta CI (tuned - baseline)")
        lines.append("| metric | 95% CI |")
        lines.append("|---|---:|")
        lines.append(f"| ok_rate | [{delta_ci['ok_rate'][0]:.4f}, {delta_ci['ok_rate'][1]:.4f}] |")
        lines.append(f"| fp_accept_rate | [{delta_ci['fp_accept_rate'][0]:.4f}, {delta_ci['fp_accept_rate'][1]:.4f}] |")
        lines.append(f"| fn_reject_rate | [{delta_ci['fn_reject_rate'][0]:.4f}, {delta_ci['fn_reject_rate'][1]:.4f}] |")
        lines.append(f"| uncertain_rate | [{delta_ci['uncertain_rate'][0]:.4f}, {delta_ci['uncertain_rate'][1]:.4f}] |")

    lines.append("")
    lines.append("## Conclusion")
    if tuned is None:
        lines.append("Root cause: tuned thresholds missing; no tuned comparison available")
    else:
        d_fp = tuned["fp"] - base["fp"]
        d_fn = tuned["fn"] - base["fn"]
        d_unc = tuned["dist"]["UNCERTAIN"] - base["dist"]["UNCERTAIN"]
        if d_fn >= d_fp and d_fn >= 0:
            cause = f"fn_reject increased by {d_fn:.4f}"
        elif d_fp > d_fn and d_fp >= 0:
            cause = f"fp_accept increased by {d_fp:.4f}"
        elif d_unc > 0:
            cause = f"uncertain increased by {d_unc:.4f}"
        else:
            cause = "no clear degradation driver"
        lines.append(f"Root cause: tuned thresholds push decision boundary such that {cause}")

    lines.append("")
    lines.append("## Repro command")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
