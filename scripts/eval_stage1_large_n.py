import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution


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


def compute_cs_ret_array(logits, tau):
    return np.asarray([compute_cs_ret_from_logit(float(v), float(tau)) for v in logits], dtype=np.float64)


def wilson_interval(k, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def mcnemar_exact_p(b, c):
    total = b + c
    if total == 0:
        return 1.0
    k = min(b, c)
    acc = 0.0
    for i in range(k + 1):
        acc += math.comb(total, i)
    p = 2.0 * acc * (0.5 ** total)
    return min(1.0, p)


def evaluate_config(cs_ret, y, t_lower, t_upper):
    decisions = [decide_route(float(v), t_lower, t_upper)[0] for v in cs_ret]
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    y_arr = np.asarray(y, dtype=int)
    accept = np.array([d == "ACCEPT" for d in decisions])
    reject = np.array([d == "REJECT" for d in decisions])
    fp = float(np.mean((accept) & (y_arr == 0))) if len(y_arr) else 0.0
    fn = float(np.mean((reject) & (y_arr == 1))) if len(y_arr) else 0.0
    tp = float(np.mean((accept) & (y_arr == 1))) if len(y_arr) else 0.0
    precision_accept = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    risk = fp + fn
    coverage = 1.0 - dist["UNCERTAIN"]
    ok = 1.0 - risk
    return {
        "decisions": decisions,
        "dist": dist,
        "fp": fp,
        "fn": fn,
        "precision_accept": precision_accept,
        "risk": risk,
        "coverage": coverage,
        "ok": ok,
    }


def sweep_grid(cs_ret, y, steps_lower, steps_upper):
    grid = []
    lowers = np.linspace(0.0, 0.99, steps_lower)
    uppers = np.linspace(0.01, 1.0, steps_upper)
    for tl in lowers:
        for tu in uppers:
            if tl >= tu:
                continue
            res = evaluate_config(cs_ret, y, float(tl), float(tu))
            grid.append({
                "t_lower": float(tl),
                "t_upper": float(tu),
                "risk": res["risk"],
                "coverage": res["coverage"],
            })
    return grid


def top_by_coverage(grid, risk_cap, topk=20):
    pts = [g for g in grid if g["risk"] <= risk_cap]
    pts.sort(key=lambda x: (-x["coverage"], x["risk"], x["t_lower"], x["t_upper"]))
    return pts[:topk]


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
        acc_rate = float(np.mean(d == "ACCEPT")) if n else 0.0
        rej_rate = float(np.mean(d == "REJECT")) if n else 0.0
        unc_rate = float(np.mean(d == "UNCERTAIN")) if n else 0.0
        fp_rate = float(np.mean((d == "ACCEPT") & (yb == 0))) if n else 0.0
        fn_rate = float(np.mean((d == "REJECT") & (yb == 1))) if n else 0.0
        ok_rate = 1.0 - fp_rate - fn_rate
        acc.append(acc_rate)
        rej.append(rej_rate)
        unc.append(unc_rate)
        fp.append(fp_rate)
        fn.append(fn_rate)
        ok.append(ok_rate)
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
    deltas = {"ok_rate": [], "fp_accept_rate": [], "fn_reject_rate": [], "uncertain_rate": []}
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
        deltas["ok_rate"].append(ok_b - ok_a)
        deltas["fp_accept_rate"].append(fp_b - fp_a)
        deltas["fn_reject_rate"].append(fn_b - fn_a)
        deltas["uncertain_rate"].append(unc_b - unc_a)
    out = {}
    for k, vals in deltas.items():
        out[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["scifact", "fever"])
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--tuned_thresholds_yaml")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--t_lower", type=float)
    ap.add_argument("--t_upper", type=float)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--grid_lower_steps", type=int, default=25)
    ap.add_argument("--grid_upper_steps", type=int, default=25)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    out_md = args.out_md or f"reports/stage1_eval_{args.track}_large.md"
    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [args.logit_col, args.y_col]
    df = pd.read_parquet(args.in_path, columns=cols)
    if args.logit_col not in df.columns or args.y_col not in df.columns:
        raise SystemExit("missing required columns in parquet")
    df = df.dropna(subset=[args.logit_col, args.y_col]).copy()
    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)
    y = df[args.y_col].astype(int).to_numpy()
    logits = df[args.logit_col].astype(float).to_numpy()

    t_lower_base, t_upper_base, tau_base = load_thresholds(args.thresholds_yaml, args.track)
    if args.tau is not None:
        tau_base = float(args.tau)
    if args.t_lower is not None:
        t_lower_base = float(args.t_lower)
    if args.t_upper is not None:
        t_upper_base = float(args.t_upper)

    t_lower_tuned = None
    t_upper_tuned = None
    tau_tuned = None
    tuned_path = Path(args.tuned_thresholds_yaml) if args.tuned_thresholds_yaml else None
    if tuned_path and tuned_path.exists() and tuned_path.stat().st_size > 0:
        t_lower_tuned, t_upper_tuned, tau_tuned = load_thresholds(tuned_path, args.track)
        if args.tau is not None:
            tau_tuned = float(args.tau)

    cs_ret_base = compute_cs_ret_array(logits, tau_base)
    base = evaluate_config(cs_ret_base, y, t_lower_base, t_upper_base)
    n_total = len(y)
    base_ci = bootstrap_ci(base["decisions"], y, args.bootstrap, args.seed)

    tuned = None
    tuned_ci = None
    delta_ci = None
    if t_lower_tuned is not None and t_upper_tuned is not None and tau_tuned is not None:
        cs_ret_tuned = compute_cs_ret_array(logits, tau_tuned)
        tuned = evaluate_config(cs_ret_tuned, y, t_lower_tuned, t_upper_tuned)
        tuned_ci = bootstrap_ci(tuned["decisions"], y, args.bootstrap, args.seed + 1)
        delta_ci = bootstrap_delta_ci(base["decisions"], tuned["decisions"], y, args.bootstrap, args.seed + 2)

    grid = sweep_grid(cs_ret_base, y, args.grid_lower_steps, args.grid_upper_steps)

    lines = []
    lines.append(f"# Stage1 Large-N Eval ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{args.in_path}`")
    lines.append(f"- n_total: {n_total}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append("")
    lines.append("## Baseline config")
    lines.append(f"- thresholds_yaml: `{args.thresholds_yaml}`")
    lines.append(f"- t_lower: {t_lower_base}")
    lines.append(f"- t_upper: {t_upper_base}")
    lines.append(f"- tau: {tau_base}")
    lines.append("")
    lines.append("| metric | value | 95% CI |")
    lines.append("|---|---:|---:|")
    lines.append(f"| accept_rate | {base['dist']['ACCEPT']:.4f} | [{base_ci['accept_rate'][0]:.4f}, {base_ci['accept_rate'][1]:.4f}] |")
    lines.append(f"| reject_rate | {base['dist']['REJECT']:.4f} | [{base_ci['reject_rate'][0]:.4f}, {base_ci['reject_rate'][1]:.4f}] |")
    lines.append(f"| uncertain_rate | {base['dist']['UNCERTAIN']:.4f} | [{base_ci['uncertain_rate'][0]:.4f}, {base_ci['uncertain_rate'][1]:.4f}] |")
    lines.append(f"| fp_accept_rate | {base['fp']:.4f} | [{base_ci['fp_accept_rate'][0]:.4f}, {base_ci['fp_accept_rate'][1]:.4f}] |")
    lines.append(f"| fn_reject_rate | {base['fn']:.4f} | [{base_ci['fn_reject_rate'][0]:.4f}, {base_ci['fn_reject_rate'][1]:.4f}] |")
    lines.append(f"| risk | {base['risk']:.4f} | |")
    lines.append(f"| coverage | {base['coverage']:.4f} | [{1.0 - base_ci['uncertain_rate'][1]:.4f}, {1.0 - base_ci['uncertain_rate'][0]:.4f}] |")
    lines.append(f"| ok_rate | {base['ok']:.4f} | [{base_ci['ok_rate'][0]:.4f}, {base_ci['ok_rate'][1]:.4f}] |")
    lines.append(f"| precision_accept | {base['precision_accept']:.4f} | |")

    if tuned is not None:
        lines.append("")
        lines.append("## Tuned config")
        lines.append(f"- tuned_thresholds_yaml: `{args.tuned_thresholds_yaml}`")
        lines.append(f"- t_lower: {t_lower_tuned}")
        lines.append(f"- t_upper: {t_upper_tuned}")
        lines.append(f"- tau: {tau_tuned}")
        lines.append("")
        lines.append("| metric | value | 95% CI |")
        lines.append("|---|---:|---:|")
        lines.append(f"| accept_rate | {tuned['dist']['ACCEPT']:.4f} | [{tuned_ci['accept_rate'][0]:.4f}, {tuned_ci['accept_rate'][1]:.4f}] |")
        lines.append(f"| reject_rate | {tuned['dist']['REJECT']:.4f} | [{tuned_ci['reject_rate'][0]:.4f}, {tuned_ci['reject_rate'][1]:.4f}] |")
        lines.append(f"| uncertain_rate | {tuned['dist']['UNCERTAIN']:.4f} | [{tuned_ci['uncertain_rate'][0]:.4f}, {tuned_ci['uncertain_rate'][1]:.4f}] |")
        lines.append(f"| fp_accept_rate | {tuned['fp']:.4f} | [{tuned_ci['fp_accept_rate'][0]:.4f}, {tuned_ci['fp_accept_rate'][1]:.4f}] |")
        lines.append(f"| fn_reject_rate | {tuned['fn']:.4f} | [{tuned_ci['fn_reject_rate'][0]:.4f}, {tuned_ci['fn_reject_rate'][1]:.4f}] |")
        lines.append(f"| risk | {tuned['risk']:.4f} | |")
        lines.append(f"| coverage | {tuned['coverage']:.4f} | [{1.0 - tuned_ci['uncertain_rate'][1]:.4f}, {1.0 - tuned_ci['uncertain_rate'][0]:.4f}] |")
        lines.append(f"| ok_rate | {tuned['ok']:.4f} | [{tuned_ci['ok_rate'][0]:.4f}, {tuned_ci['ok_rate'][1]:.4f}] |")
        lines.append(f"| precision_accept | {tuned['precision_accept']:.4f} | |")

        a_wrong = np.array([
            (d == "ACCEPT" and y[i] == 0) or (d == "REJECT" and y[i] == 1)
            for i, d in enumerate(base["decisions"])
        ])
        b_wrong = np.array([
            (d == "ACCEPT" and y[i] == 0) or (d == "REJECT" and y[i] == 1)
            for i, d in enumerate(tuned["decisions"])
        ])
        b = int(np.sum((~a_wrong) & (b_wrong)))
        c = int(np.sum((a_wrong) & (~b_wrong)))
        pval = mcnemar_exact_p(b, c)
        lines.append("")
        lines.append("## McNemar exact test")
        lines.append(f"- b (A correct, B wrong): {b}")
        lines.append(f"- c (A wrong, B correct): {c}")
        lines.append(f"- p_value: {pval:.6f}")
        if delta_ci:
            lines.append("")
            lines.append("## Bootstrap delta CI (tuned - baseline)")
            lines.append("| metric | 95% CI |")
            lines.append("|---|---:|")
            lines.append(f"| ok_rate | [{delta_ci['ok_rate'][0]:.4f}, {delta_ci['ok_rate'][1]:.4f}] |")
            lines.append(f"| fp_accept_rate | [{delta_ci['fp_accept_rate'][0]:.4f}, {delta_ci['fp_accept_rate'][1]:.4f}] |")
            lines.append(f"| fn_reject_rate | [{delta_ci['fn_reject_rate'][0]:.4f}, {delta_ci['fn_reject_rate'][1]:.4f}] |")
            lines.append(f"| uncertain_rate | [{delta_ci['uncertain_rate'][0]:.4f}, {delta_ci['uncertain_rate'][1]:.4f}] |")

    lines.append("")
    lines.append("## Risk-coverage sweep")
    for risk_cap in [0.02, 0.05, 0.10]:
        top = top_by_coverage(grid, risk_cap)
        lines.append(f"### Top coverage under risk <= {risk_cap}")
        if not top:
            lines.append("none")
            lines.append("")
            continue
        lines.append("| t_lower | t_upper | risk | coverage |")
        lines.append("|---:|---:|---:|---:|")
        for row in top:
            lines.append(f"| {row['t_lower']:.4f} | {row['t_upper']:.4f} | {row['risk']:.4f} | {row['coverage']:.4f} |")
        lines.append("")

    if args.track == "fever":
        top05 = top_by_coverage(grid, 0.05, topk=1)
        best_cov = top05[0]["coverage"] if top05 else 0.0
        lines.append("## FEVER verdict")
        if best_cov <= 0.10:
            lines.append("negative result: coverage is extremely low under risk<=0.05; stage1-only is weak")
        else:
            lines.append("coverage under risk<=0.05 is not extremely low at baseline thresholds")

    lines.append("")
    lines.append("## Repro command")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
