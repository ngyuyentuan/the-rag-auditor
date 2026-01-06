import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route
from scripts.stage1_cheap_checks import apply_cheap_checks


def load_thresholds(path: Path, track: str):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    if "thresholds" in data:
        t = data["thresholds"].get(track)
        if not isinstance(t, dict):
            return None
        for k in ("t_lower", "t_upper", "tau"):
            if k not in t:
                return None
        return {"tau": float(t["tau"]), "t_lower": float(t["t_lower"]), "t_upper": float(t["t_upper"])}
    if all(k in data for k in ("tau", "t_lower", "t_upper")):
        return {"tau": float(data["tau"]), "t_lower": float(data["t_lower"]), "t_upper": float(data["t_upper"])}
    return None


def compute_metrics(decisions, y):
    n = len(y)
    y_arr = np.asarray(y, dtype=int)
    d = np.asarray(decisions)
    accept = d == "ACCEPT"
    reject = d == "REJECT"
    uncertain = d == "UNCERTAIN"
    fp = np.sum(accept & (y_arr == 0))
    fn = np.sum(reject & (y_arr == 1))
    tp = np.sum(accept & (y_arr == 1))
    tn = np.sum(reject & (y_arr == 0))
    accept_rate = float(np.mean(accept)) if n else 0.0
    reject_rate = float(np.mean(reject)) if n else 0.0
    uncertain_rate = float(np.mean(uncertain)) if n else 0.0
    fp_accept_rate = fp / n if n else 0.0
    fn_reject_rate = fn / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    coverage = 1.0 - uncertain_rate
    decided = tp + tn + fp + fn
    accuracy_on_decided = (tp + tn) / decided if decided else 0.0
    risk = fp_accept_rate + fn_reject_rate
    return {
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": accuracy_on_decided,
        "risk": risk,
    }


def bootstrap_ci(decisions, y, rng, n_boot):
    n = len(y)
    decisions_arr = np.asarray(decisions)
    y_arr = np.asarray(y, dtype=int)
    metrics = {
        "uncertain_rate": [],
        "fp_accept_rate": [],
        "fn_reject_rate": [],
        "ok_rate": [],
        "coverage": [],
        "accuracy_on_decided": [],
        "risk": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = decisions_arr[idx]
        yb = y_arr[idx]
        m = compute_metrics(d, yb)
        for k in metrics:
            metrics[k].append(m[k])
    out = {}
    for k, vals in metrics.items():
        arr = np.asarray(vals)
        out[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def eval_config(df, logit_col, y_col, tau, t_lower, t_upper, cheap_checks, claim_col, passage_col, score_cols):
    logits = df[logit_col].astype(float).tolist()
    y_vals = df[y_col].astype(int).tolist()
    cs_ret = [compute_cs_ret_from_logit(x, tau) for x in logits]
    decisions = []
    for p in cs_ret:
        d, _ = decide_route(p, t_lower, t_upper)
        decisions.append(d)
    if cheap_checks == "off":
        return decisions, compute_metrics(decisions, y_vals)
    if claim_col is None or passage_col is None:
        return decisions, compute_metrics(decisions, y_vals)
    claims = df[claim_col].tolist()
    passages = df[passage_col].tolist()
    scores = None
    if score_cols:
        scores = df[score_cols].values.tolist()
    adjusted = []
    for idx, d in enumerate(decisions):
        claim = claims[idx]
        passage = passages[idx]
        score_list = scores[idx] if scores is not None else None
        new_d, _ = apply_cheap_checks(d, claim, passage, score_list)
        adjusted.append(new_d)
    return adjusted, compute_metrics(adjusted, y_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--baseline_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--joint_yaml")
    ap.add_argument("--product_yaml")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--cheap_checks", choices=["off", "on", "both"], default="off")
    ap.add_argument("--claim_col")
    ap.add_argument("--passage_col")
    ap.add_argument("--score_cols", nargs="+")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    df = pd.read_parquet(in_path)
    if args.logit_col not in df.columns:
        raise ValueError(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise ValueError(f"missing y_col {args.y_col}")

    y = pd.to_numeric(df[args.y_col], errors="coerce")
    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna()
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    df[args.logit_col] = logits[mask].astype(float)

    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    claim_col = args.claim_col
    passage_col = args.passage_col
    if claim_col is None:
        claim_col = pick_column(df, ["claim", "query", "question", "hypothesis", "sentence"])
    if passage_col is None:
        passage_col = pick_column(df, ["passage", "evidence", "context", "top_passage", "text"])
    if claim_col is not None and claim_col not in df.columns:
        claim_col = None
    if passage_col is not None and passage_col not in df.columns:
        passage_col = None

    baseline = load_thresholds(Path(args.baseline_yaml), args.track)
    joint_path = Path(args.joint_yaml) if args.joint_yaml else Path(f"configs/thresholds_stage1_joint_tuned_{args.track}.yaml")
    product_path = Path(args.product_yaml) if args.product_yaml else Path(f"configs/thresholds_stage1_product_{args.track}.yaml")
    joint = load_thresholds(joint_path, args.track)
    product = load_thresholds(product_path, args.track)

    configs = [("baseline", baseline)]
    if joint is not None:
        configs.append(("joint_tuned", joint))
    if product is not None:
        configs.append(("product_tuned", product))

    rng = np.random.default_rng(args.seed)
    results = []
    for name, cfg in configs:
        for mode in ("off", "on"):
            if args.cheap_checks == "off" and mode == "on":
                continue
            if args.cheap_checks == "on" and mode == "off":
                continue
            decisions, metrics = eval_config(
                df,
                args.logit_col,
                args.y_col,
                cfg["tau"],
                cfg["t_lower"],
                cfg["t_upper"],
                mode,
                claim_col,
                passage_col,
                args.score_cols,
            )
            ci = bootstrap_ci(decisions, df[args.y_col].tolist(), rng, args.bootstrap)
            results.append((name, mode, cfg, metrics, ci))

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_product_eval_v2_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Stage1 Product Eval V2 ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: {len(df)}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- claim_col: `{claim_col}`")
    lines.append(f"- passage_col: `{passage_col}`")
    lines.append(f"- cheap_checks: `{args.cheap_checks}`")
    lines.append("")
    for name, mode, cfg, metrics, ci in results:
        lines.append(f"## {name} (checks {mode})")
        lines.append("")
        lines.append(f"- tau: `{cfg['tau']}`")
        lines.append(f"- t_lower: `{cfg['t_lower']}`")
        lines.append(f"- t_upper: `{cfg['t_upper']}`")
        lines.append("")
        lines.append("| metric | value | 95% CI |")
        lines.append("|---|---:|---:|")
        for k in ("uncertain_rate", "fp_accept_rate", "fn_reject_rate", "risk", "ok_rate", "coverage", "accuracy_on_decided"):
            lo, hi = ci[k]
            lines.append(f"| {k} | {metrics[k]:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.append("")
    lines.append("Interpretation")
    lines.append("")
    lines.append("ok_rate treats UNCERTAIN as safe deferral, so it can look high when many cases are deferred. coverage and accuracy_on_decided show how many decisions are made and how accurate they are among decided cases. Use risk, coverage, and accuracy_on_decided together when choosing a product default.")
    lines.append("")
    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
