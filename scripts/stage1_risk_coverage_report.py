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
    ap.add_argument("--out_md", default=None)
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
            results.append(compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper)))

    budgets = [0.10, 0.20, 0.30, 0.50, 0.70]
    fp_caps = [0.01, 0.02, 0.05, 0.10]

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_risk_coverage_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Risk Coverage ({args.track})")
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

    base_metrics = compute_metrics(cs_ret, y_vals, float(base["t_lower"]), float(base["t_upper"]))
    lines.append("Baseline thresholds")
    lines.append("")
    lines.append(f"- t_lower: `{base['t_lower']}`")
    lines.append(f"- t_upper: `{base['t_upper']}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| accept_rate | {format_float(base_metrics['accept_rate'])} |")
    lines.append(f"| reject_rate | {format_float(base_metrics['reject_rate'])} |")
    lines.append(f"| uncertain_rate | {format_float(base_metrics['uncertain_rate'])} |")
    lines.append(f"| fp_accept_rate | {format_float(base_metrics['fp_accept_rate'])} |")
    lines.append(f"| fn_reject_rate | {format_float(base_metrics['fn_reject_rate'])} |")
    lines.append(f"| ok_rate | {format_float(base_metrics['ok_rate'])} |")
    lines.append("")

    lines.append("Best ok_rate under uncertain budgets")
    lines.append("")
    lines.append("| max_uncertain | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in budgets:
        feasible = [r for r in results if r["uncertain_rate"] <= b]
        if not feasible:
            lines.append(f"| {b} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        feasible.sort(key=lambda r: (-r["ok_rate"], r["fp_accept_rate"], r["fn_reject_rate"], r["uncertain_rate"]))
        r = feasible[0]
        lines.append("| {b} | {tl} | {tu} | {a} | {rj} | {u} | {fp} | {fn} | {ok} |".format(
            b=format_float(b),
            tl=format_float(r["t_lower"]),
            tu=format_float(r["t_upper"]),
            a=format_float(r["accept_rate"]),
            rj=format_float(r["reject_rate"]),
            u=format_float(r["uncertain_rate"]),
            fp=format_float(r["fp_accept_rate"]),
            fn=format_float(r["fn_reject_rate"]),
            ok=format_float(r["ok_rate"]),
        ))
    lines.append("")

    lines.append("Best ok_rate under fp_accept caps")
    lines.append("")
    lines.append("| max_fp_accept | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cap in fp_caps:
        feasible = [r for r in results if r["fp_accept_rate"] <= cap]
        if not feasible:
            lines.append(f"| {cap} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        feasible.sort(key=lambda r: (-r["ok_rate"], r["fn_reject_rate"], r["uncertain_rate"]))
        r = feasible[0]
        lines.append("| {c} | {tl} | {tu} | {a} | {rj} | {u} | {fp} | {fn} | {ok} |".format(
            c=format_float(cap),
            tl=format_float(r["t_lower"]),
            tu=format_float(r["t_upper"]),
            a=format_float(r["accept_rate"]),
            rj=format_float(r["reject_rate"]),
            u=format_float(r["uncertain_rate"]),
            fp=format_float(r["fp_accept_rate"]),
            fn=format_float(r["fn_reject_rate"]),
            ok=format_float(r["ok_rate"]),
        ))
    lines.append("")

    lines.append("Recommended operating points")
    lines.append("")
    conservative = min(results, key=lambda r: (r["fp_accept_rate"], r["fn_reject_rate"], r["uncertain_rate"]))
    balanced = max(results, key=lambda r: r["ok_rate"])
    aggressive = min(results, key=lambda r: (r["uncertain_rate"], r["fp_accept_rate"], r["fn_reject_rate"]))
    lines.append("| profile | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in [("conservative_accept", conservative), ("balanced", balanced), ("aggressive", aggressive)]:
        lines.append("| {name} | {tl} | {tu} | {a} | {rj} | {u} | {fp} | {fn} | {ok} |".format(
            name=name,
            tl=format_float(r["t_lower"]),
            tu=format_float(r["t_upper"]),
            a=format_float(r["accept_rate"]),
            rj=format_float(r["reject_rate"]),
            u=format_float(r["uncertain_rate"]),
            fp=format_float(r["fp_accept_rate"]),
            fn=format_float(r["fn_reject_rate"]),
            ok=format_float(r["ok_rate"]),
        ))
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
