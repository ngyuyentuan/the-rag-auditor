import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import (
    compute_cs_ret_from_logit,
    decide_route,
    routing_distribution,
    stage1_outcomes,
)


def load_thresholds(path: Path, track: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "thresholds" not in data:
        raise ValueError(f"thresholds missing in {path}")
    t = data["thresholds"].get(track)
    if not isinstance(t, dict):
        raise ValueError(f"track {track} missing in {path}")
    for k in ("t_lower", "t_upper", "tau"):
        if k not in t:
            raise ValueError(f"{k} missing for track {track} in {path}")
    return t


def compute_rates(cs_ret, y, t_lower, t_upper):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    decisions = []
    for p, yi in zip(cs_ret, y):
        decision, _ = decide_route(p, t_lower, t_upper)
        decisions.append(decision)
        o = stage1_outcomes(decision, yi)
        for k in counts:
            counts[k] += o[k]
    n = len(y)
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    fp_accept_rate = counts["fp"] / n if n else 0.0
    fn_reject_rate = counts["fn"] / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    return {
        "accept_rate": dist["ACCEPT"],
        "reject_rate": dist["REJECT"],
        "uncertain_rate": dist["UNCERTAIN"],
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "counts": counts,
    }


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--grid_lower_steps", type=int, default=50)
    ap.add_argument("--grid_upper_steps", type=int, default=50)
    ap.add_argument("--budgets", type=float, nargs="+", required=True)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_yaml", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    thresholds_path = Path("configs/thresholds.yaml")
    base = load_thresholds(thresholds_path, args.track)
    tau = args.tau if args.tau is not None else float(base["tau"])

    df = pd.read_parquet(in_path)
    if args.logit_col not in df.columns:
        raise ValueError(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise ValueError(f"missing y_col {args.y_col}")

    y = pd.to_numeric(df[args.y_col], errors="coerce")
    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)

    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    logits = df[args.logit_col].astype(float).tolist()
    y_vals = df[args.y_col].astype(int).tolist()
    cs_ret = [compute_cs_ret_from_logit(x, tau) for x in logits]

    grid_lower = np.linspace(0.0, 0.99, args.grid_lower_steps)
    grid_upper = np.linspace(0.01, 1.0, args.grid_upper_steps)

    results = []
    for t_lower in grid_lower:
        for t_upper in grid_upper:
            if t_lower >= t_upper:
                continue
            r = compute_rates(cs_ret, y_vals, float(t_lower), float(t_upper))
            results.append({
                "t_lower": float(t_lower),
                "t_upper": float(t_upper),
                "accept_rate": r["accept_rate"],
                "reject_rate": r["reject_rate"],
                "uncertain_rate": r["uncertain_rate"],
                "fp_accept_rate": r["fp_accept_rate"],
                "fn_reject_rate": r["fn_reject_rate"],
                "ok_rate": r["ok_rate"],
            })

    budgets = sorted(args.budgets)
    picks = {}
    top10 = {}
    for b in budgets:
        candidates = [r for r in results if r["uncertain_rate"] <= b]
        candidates.sort(key=lambda r: (-r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]))
        picks[b] = candidates[0] if candidates else None
        top10[b] = candidates[:10] if candidates else []

    baseline = compute_rates(cs_ret, y_vals, float(base["t_lower"]), float(base["t_upper"]))

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_threshold_tuning_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Threshold Tuning ({args.track})")
    lines.append("")
    lines.append(f"Generated: `{dt.datetime.utcnow().isoformat()}+00:00`")
    lines.append("")
    lines.append("Config")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: `{len(df)}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- tau: `{tau}`")
    lines.append("")
    lines.append("Baseline thresholds")
    lines.append("")
    lines.append(f"- t_lower: `{base['t_lower']}`")
    lines.append(f"- t_upper: `{base['t_upper']}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| accept_rate | {format_float(baseline['accept_rate'])} |")
    lines.append(f"| reject_rate | {format_float(baseline['reject_rate'])} |")
    lines.append(f"| uncertain_rate | {format_float(baseline['uncertain_rate'])} |")
    lines.append(f"| fp_accept_rate | {format_float(baseline['fp_accept_rate'])} |")
    lines.append(f"| fn_reject_rate | {format_float(baseline['fn_reject_rate'])} |")
    lines.append(f"| ok_rate | {format_float(baseline['ok_rate'])} |")
    lines.append("")

    for b in budgets:
        lines.append(f"Top 10 configs for budget <= {b}")
        lines.append("")
        lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
        lines.append("|---|---|---|---|---|---|---|---|")
        rows = top10[b]
        if not rows:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            for r in rows:
                lines.append("| " + " | ".join([
                    format_float(r["t_lower"]),
                    format_float(r["t_upper"]),
                    format_float(r["accept_rate"]),
                    format_float(r["reject_rate"]),
                    format_float(r["uncertain_rate"]),
                    format_float(r["fp_accept_rate"]),
                    format_float(r["fn_reject_rate"]),
                    format_float(r["ok_rate"]),
                ]) + " |")
        lines.append("")
        pick = picks[b]
        lines.append("Recommended thresholds")
        lines.append("")
        if pick is None:
            lines.append("- none (no configs met uncertain budget)")
        else:
            lines.append(f"- t_lower: `{pick['t_lower']}`")
            lines.append(f"- t_upper: `{pick['t_upper']}`")
            lines.append(f"- ok_rate: `{pick['ok_rate']}`")
            lines.append(f"- uncertain_rate: `{pick['uncertain_rate']}`")
        lines.append("")

    lines.append("Interpretation")
    lines.append("")
    lines.append("Lower uncertain budgets reduce deferrals and can increase fp or fn rates. Higher budgets allow more deferrals and typically improve ok_rate. Use the budget that matches expected stage2 capacity.")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    smallest_budget = budgets[0]
    best = picks[smallest_budget]
    out_yaml = Path(args.out_yaml) if args.out_yaml else Path(f"configs/thresholds_stage1_tuned_{args.track}.yaml")
    if best is not None:
        out_yaml.write_text(yaml.safe_dump({
            "tau": float(tau),
            "t_lower": float(best["t_lower"]),
            "t_upper": float(best["t_upper"]),
        }, sort_keys=False), encoding="utf-8")
    else:
        out_yaml.write_text(yaml.safe_dump({
            "tau": float(tau),
            "t_lower": None,
            "t_upper": None,
        }, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
