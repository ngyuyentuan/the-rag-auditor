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
    ap.add_argument("--grid_lower_steps", type=int, default=100)
    ap.add_argument("--grid_upper_steps", type=int, default=100)
    ap.add_argument("--max_fp_accept", type=float, nargs="+", required=True)
    ap.add_argument("--max_fn_reject", type=float, nargs="+", required=True)
    ap.add_argument("--objective", choices=["min_uncertain", "min_stage2_cost"], default="min_uncertain")
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
            results.append(compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper)))

    fp_limits = sorted(float(x) for x in args.max_fp_accept)
    fn_limits = sorted(float(x) for x in args.max_fn_reject)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Constrained Tuning ({args.track})")
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
    lines.append(f"- objective: `{args.objective}`")
    lines.append("")

    best_default = None
    best_default_source = None
    default_gate = (0.02, 0.01)
    for fp_lim in fp_limits:
        for fn_lim in fn_limits:
            feasible = [
                r for r in results
                if r["fp_accept_rate"] <= fp_lim
                and r["fn_reject_rate"] <= fn_lim
            ]
            feasible.sort(key=lambda r: (
                r["uncertain_rate"],
                -r["ok_rate"],
                -r["accept_rate"],
                -r["reject_rate"],
            ))
            lines.append(f"Gate fp<={fp_lim} fn<={fn_lim}")
            lines.append("")
            lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
            lines.append("|---|---|---|---|---|---|---|---|")
            if not feasible:
                lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                lines.append("")
                closest = []
                for r in results:
                    penalty = max(0.0, r["fp_accept_rate"] - fp_lim) + max(0.0, r["fn_reject_rate"] - fn_lim) + max(0.0, r["uncertain_rate"] - 0.3)
                    closest.append((penalty, r))
                closest.sort(key=lambda x: (x[0], x[1]["uncertain_rate"]))
                lines.append("Closest configs (penalty)")
                lines.append("")
                lines.append("| penalty | t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for penalty, r in closest[:20]:
                    lines.append("| " + " | ".join([
                        format_float(penalty),
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
                if (fp_lim, fn_lim) == default_gate and closest:
                    best_default = closest[0][1]
                    best_default_source = "closest"
                continue
            for r in feasible[:20]:
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
            if (fp_lim, fn_lim) == default_gate:
                best_default = feasible[0]
                best_default_source = "feasible"

    if best_default:
        if best_default_source == "closest":
            lines.append("Recommended config (closest)")
        else:
            lines.append("Recommended config")
        lines.append("")
        lines.append("- " + " ".join([
            f"t_lower={format_float(best_default['t_lower'])}",
            f"t_upper={format_float(best_default['t_upper'])}",
            f"fp_accept_rate={format_float(best_default['fp_accept_rate'])}",
            f"fn_reject_rate={format_float(best_default['fn_reject_rate'])}",
            f"uncertain_rate={format_float(best_default['uncertain_rate'])}",
        ]))
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    if best_default is None:
        raise SystemExit("no feasible config for default gate")
    out_yaml.write_text(yaml.safe_dump({
        "tau": float(tau),
        "t_lower": float(best_default["t_lower"]),
        "t_upper": float(best_default["t_upper"]),
    }, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
