import os, json, math, argparse
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None

def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def compute_scores(logits, tau):
    z = logits / float(tau)
    v = np.vectorize(stable_sigmoid)(z.astype(np.float64))
    return v.astype(np.float64)

def nll_for_tau(tau, logits, y):
    s = compute_scores(logits, tau)
    eps = 1e-12
    s = np.clip(s, eps, 1 - eps)
    y = y.astype(np.float64)
    return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))

def fit_tau(val_logits, val_y):
    if minimize_scalar is None:
        return 1.0, None, None
    f = lambda t: nll_for_tau(t, val_logits, val_y)
    res = minimize_scalar(f, bounds=(0.05, 10.0), method="bounded")
    tau = float(res.x)
    nll_before = f(1.0)
    nll_after = f(tau)
    if (not np.isfinite(tau)) or (tau <= 0) or (nll_after >= nll_before):
        return 1.0, float(nll_before), float(nll_before)
    return tau, float(nll_before), float(nll_after)

def sanitize(df, logit_col, y_col, name):
    n0 = len(df)
    d = df[[logit_col, y_col]].copy()
    d[logit_col] = pd.to_numeric(d[logit_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[logit_col, y_col])
    d = d[np.isfinite(d[logit_col].to_numpy(np.float64))]
    d = d[(d[y_col] == 0) | (d[y_col] == 1)]
    d[y_col] = d[y_col].astype(int)
    n1 = len(d)
    kept = f"{n1}/{n0}"
    if n0 > 0 and (n0 - n1) / n0 > 0.10:
        print(f"[warn] sanitize({name}): kept={kept}")
    else:
        print(f"[ok] sanitize({name}): {kept} rows kept")
    return d

def metrics_at_threshold(scores, y, t):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    pred = scores >= float(t)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    prec = None
    rec = None
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    if (tp + fn) > 0:
        rec = tp / (tp + fn)
    f1 = 0.0
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)
    acc = None
    denom = tp + fp + fn + tn
    if denom > 0:
        acc = (tp + tn) / denom
    return {
        "precision": float(prec) if prec is not None else None,
        "recall": float(rec) if rec is not None else None,
        "f1": float(f1),
        "accuracy": float(acc) if acc is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support_pos": int(tp + fn),
        "pred_pos": int(tp + fp),
    }

def choose_thresholds(scores, y, target_precision, target_recall):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    uniq = np.unique(scores)
    cand_upper = []
    cand_lower = []
    for t in uniq:
        m = metrics_at_threshold(scores, y, t)
        p = m["precision"]
        r = m["recall"]
        if p is not None and r is not None:
            if p >= target_precision:
                cand_upper.append((r, p, float(t)))
            if r >= target_recall:
                cand_lower.append((p, r, float(t)))
    if len(cand_upper) == 0:
        t_upper = 1.0
    else:
        cand_upper.sort(key=lambda x: (x[0], x[2]), reverse=True)
        t_upper = float(cand_upper[0][2])
    if len(cand_lower) == 0:
        t_lower = 0.0
    else:
        cand_lower.sort(key=lambda x: (x[0], x[2]), reverse=True)
        t_lower = float(cand_lower[0][2])
    if t_lower > t_upper:
        t_lower = min(t_lower, t_upper)
    return t_lower, t_upper

def compute_ece(scores, y, n_bins=10):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    info = []
    n = len(scores)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        inb = (scores > lo) & (scores <= hi)
        k = int(np.sum(inb))
        if k == 0:
            continue
        conf = float(np.mean(scores[inb]))
        acc = float(np.mean(y[inb]))
        gap = abs(conf - acc)
        ece += (k / n) * gap
        info.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": k, "avg_conf": conf, "avg_acc": acc, "gap": float(gap)})
    return float(ece), info

def plot_reliability(scores, y, out_path, title):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, 11)
    confs = []
    accs = []
    for i in range(10):
        lo = bins[i]
        hi = bins[i+1]
        inb = (scores > lo) & (scores <= hi)
        if int(np.sum(inb)) == 0:
            continue
        confs.append(float(np.mean(scores[inb])))
        accs.append(float(np.mean(y[inb])))
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    if len(confs) > 0:
        plt.plot(confs, accs, marker="o")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_pr(y_true, scores, out_path, title):
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if int(np.sum(y_true)) == 0:
        plt.figure(figsize=(6, 5))
        plt.text(0.1, 0.5, "No positive samples in y_true")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(title)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return None
    p, r, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True

def load_split(base, split):
    base2 = base
    if base2.endswith(".parquet"):
        base2 = base2[:-8]
    p1 = f"{base2}_{split}.parquet"
    if os.path.exists(p1):
        return p1
    p2 = f"{base2}{split}.parquet"
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"missing split file: tried {p1} and {p2}")


def write_thresholds_yaml(path, track, obj):
    try:
        import yaml
        if os.path.exists(path):
            data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        else:
            data = {}
        if "thresholds" not in data or data["thresholds"] is None:
            data["thresholds"] = {}
        data["thresholds"][track] = obj
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return
    except Exception:
        pass
    lines = []
    lines.append("thresholds:")
    lines.append(f"  {track}:")
    for k in ["t_lower","t_upper","tau","target_precision","target_recall"]:
        v = obj.get(k, None)
        if v is None:
            continue
        lines.append(f"    {k}: {v}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--in_base", required=True)
    ap.add_argument("--out_yaml", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--fig_pr", required=True)
    ap.add_argument("--fig_rel_before", required=True)
    ap.add_argument("--fig_rel_after", required=True)
    ap.add_argument("--target_precision", type=float, default=0.90)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--logit_col", default=None)
    ap.add_argument("--y_col", default="y")
    args = ap.parse_args()

    tr_path = load_split(args.in_base, "train")
    va_path = load_split(args.in_base, "val")
    te_path = load_split(args.in_base, "test")

    df_tr = pd.read_parquet(tr_path)
    df_va = pd.read_parquet(va_path)
    df_te = pd.read_parquet(te_path)

    logit_candidates = ["raw_max_top3", "max_top3_logit", "ce_max_top3", "max_top3_score", "max_top3", "cs_ret"]
    logit_col = args.logit_col
    if logit_col is None:
        for c in logit_candidates:
            if c in df_tr.columns:
                logit_col = c
                break
    if logit_col is None:
        raise SystemExit("no logit_col found")

    df_tr = sanitize(df_tr, logit_col, args.y_col, "train")
    df_va = sanitize(df_va, logit_col, args.y_col, "val")
    df_te = sanitize(df_te, logit_col, args.y_col, "test")

    tr_logits = df_tr[logit_col].to_numpy(np.float64)
    va_logits = df_va[logit_col].to_numpy(np.float64)
    te_logits = df_te[logit_col].to_numpy(np.float64)

    y_tr = df_tr[args.y_col].to_numpy(np.int32)
    y_va = df_va[args.y_col].to_numpy(np.int32)
    y_te = df_te[args.y_col].to_numpy(np.int32)

    tau, nll_before, nll_after = fit_tau(va_logits, y_va)

    s_tr = compute_scores(tr_logits, tau)
    s_va_uncal = compute_scores(va_logits, 1.0)
    s_va = compute_scores(va_logits, tau)
    s_te_uncal = compute_scores(te_logits, 1.0)
    s_te = compute_scores(te_logits, tau)

    ece_before, bins_before = compute_ece(s_te_uncal, y_te, n_bins=10)
    ece_after, bins_after = compute_ece(s_te, y_te, n_bins=10)

    t_lower, t_upper = choose_thresholds(s_tr, y_tr, args.target_precision, args.target_recall)

    m_tr_upper = metrics_at_threshold(s_tr, y_tr, t_upper)
    m_tr_lower = metrics_at_threshold(s_tr, y_tr, t_lower)
    m_te_upper = metrics_at_threshold(s_te, y_te, t_upper)
    m_te_lower = metrics_at_threshold(s_te, y_te, t_lower)

    uncertain = float(np.mean((s_tr >= t_lower) & (s_tr < t_upper)) * 100.0) if len(s_tr) > 0 else 0.0

    plot_pr(y_te, s_te, args.fig_pr, f"{args.track} PR (test)")
    plot_reliability(s_te_uncal, y_te, args.fig_rel_before, f"{args.track} reliability before")
    plot_reliability(s_te, y_te, args.fig_rel_after, f"{args.track} reliability after (tau={tau:.3f})")

    report = {
        "track": args.track,
        "paths": {"train": tr_path, "val": va_path, "test": te_path},
        "columns": {"logit_col": logit_col, "y_col": args.y_col},
        "temperature_scaling": {
            "tau": float(tau),
            "nll_before": float(nll_before) if nll_before is not None else None,
            "nll_after": float(nll_after) if nll_after is not None else None,
        },
        "calibration": {
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "bins_after": bins_after,
        },
        "thresholds": {
            "t_lower": float(t_lower),
            "t_upper": float(t_upper),
            "target_precision": float(args.target_precision),
            "target_recall": float(args.target_recall),
            "uncertain_zone_coverage_pct": float(uncertain),
        },
        "train_metrics": {"upper": m_tr_upper, "lower": m_tr_lower},
        "test_metrics": {"upper": m_te_upper, "lower": m_te_lower},
        "support": {
            "n_train": int(len(y_tr)), "pos_train": int(np.sum(y_tr)),
            "n_val": int(len(y_va)), "pos_val": int(np.sum(y_va)),
            "n_test": int(len(y_te)), "pos_test": int(np.sum(y_te)),
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_thresholds_yaml(args.out_yaml, args.track, {
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "tau": float(tau),
        "target_precision": float(args.target_precision),
        "target_recall": float(args.target_recall),
    })

    print("[ok] wrote:", args.out_json)
    print("[ok] wrote:", args.out_yaml)
    print("[ok] figs:", args.fig_pr, "|", args.fig_rel_before, "|", args.fig_rel_after)

if __name__ == "__main__":
    main()
