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

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution, stage1_outcomes


def load_thresholds(path: Path, track: str) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    t = (data.get("thresholds") or {}).get(track)
    if not t:
        raise SystemExit(f"missing thresholds for track={track} in {path}")
    for k in ("t_lower", "t_upper", "tau"):
        if k not in t:
            raise SystemExit(f"{k} missing for track {track} in {path}")
    return t


def compute_metrics(cs_ret, y, t_lower, t_upper):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    for p, yi in zip(cs_ret, y):
        decision, _ = decide_route(p, t_lower, t_upper)
        o = stage1_outcomes(decision, yi)
        for k in counts:
            counts[k] += o[k]
    n = len(y) or 1
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    fp_accept_rate = counts["fp"] / n
    fn_reject_rate = counts["fn"] / n
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    return {
        "accept_rate": dist["ACCEPT"],
        "reject_rate": dist["REJECT"],
        "uncertain_rate": dist["UNCERTAIN"],
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "t_lower": t_lower,
        "t_upper": t_upper,
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
    ap.add_argument("--grid_lower_steps", type=int, default=80)
    ap.add_argument("--grid_upper_steps", type=int, default=80)
    ap.add_argument("--cost_fp", type=float, default=10.0)
    ap.add_argument("--cost_fn", type=float, default=5.0)
    ap.add_argument("--cost_defer", type=float, default=1.0)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_yaml", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    thresholds_path = Path("configs/thresholds.yaml")
    base = load_thresholds(thresholds_path, args.track)
    tau = args.tau if args.tau is not None else float(base["tau"])

    df = pd.read_parquet(in_path)
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
            metrics = compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper))
            expected_cost = (
                args.cost_fp * metrics["fp_accept_rate"]
                + args.cost_fn * metrics["fn_reject_rate"]
                + args.cost_defer * metrics["uncertain_rate"]
            )
            metrics["expected_cost"] = expected_cost
            results.append(metrics)

    results.sort(key=lambda r: (
        r["expected_cost"],
        r["fp_accept_rate"],
        r["fn_reject_rate"],
        r["uncertain_rate"],
    ))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Cost Tuning ({args.track})")
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
    lines.append(f"- cost_fp: `{args.cost_fp}`")
    lines.append(f"- cost_fn: `{args.cost_fn}`")
    lines.append(f"- cost_defer: `{args.cost_defer}`")
    lines.append("")

    best = results[0] if results else None
    lines.append("Top 20 configs")
    lines.append("")
    lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | expected_cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results[:20]:
        lines.append("| " + " | ".join([
            format_float(r["t_lower"]),
            format_float(r["t_upper"]),
            format_float(r["accept_rate"]),
            format_float(r["reject_rate"]),
            format_float(r["uncertain_rate"]),
            format_float(r["fp_accept_rate"]),
            format_float(r["fn_reject_rate"]),
            format_float(r["ok_rate"]),
            format_float(r["expected_cost"]),
        ]) + " |")
    lines.append("")

    lines.append("Pareto frontier (fp/fn/uncertain)")
    lines.append("")
    pareto = []
    for r in results:
        dominated = False
        for o in results:
            if o is r:
                continue
            if (
                o["fp_accept_rate"] <= r["fp_accept_rate"]
                and o["fn_reject_rate"] <= r["fn_reject_rate"]
                and o["uncertain_rate"] <= r["uncertain_rate"]
                and (
                    o["fp_accept_rate"] < r["fp_accept_rate"]
                    or o["fn_reject_rate"] < r["fn_reject_rate"]
                    or o["uncertain_rate"] < r["uncertain_rate"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: (r["fp_accept_rate"], r["fn_reject_rate"], r["uncertain_rate"]))
    lines.append("| fp_accept_rate | fn_reject_rate | uncertain_rate | accept_rate | reject_rate | ok_rate | expected_cost |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in pareto[:50]:
        lines.append("| " + " | ".join([
            format_float(r["fp_accept_rate"]),
            format_float(r["fn_reject_rate"]),
            format_float(r["uncertain_rate"]),
            format_float(r["accept_rate"]),
            format_float(r["reject_rate"]),
            format_float(r["ok_rate"]),
            format_float(r["expected_cost"]),
        ]) + " |")
    lines.append("")

    if best:
        lines.append("Recommended config")
        lines.append("")
        lines.append("- " + " ".join([
            f"t_lower={format_float(best['t_lower'])}",
            f"t_upper={format_float(best['t_upper'])}",
            f"expected_cost={format_float(best['expected_cost'])}",
            f"fp_accept_rate={format_float(best['fp_accept_rate'])}",
            f"fn_reject_rate={format_float(best['fn_reject_rate'])}",
            f"uncertain_rate={format_float(best['uncertain_rate'])}",
        ]))
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    if best is None:
        raise SystemExit("no configs evaluated")
    out_yaml.write_text(yaml.safe_dump({
        "tau": float(tau),
        "t_lower": float(best["t_lower"]),
        "t_upper": float(best["t_upper"]),
    }, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
