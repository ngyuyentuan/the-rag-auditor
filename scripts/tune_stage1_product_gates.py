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
    decisions = []
    for p, yi in zip(cs_ret, y):
        decision, _ = decide_route(p, t_lower, t_upper)
        decisions.append(decision)
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


def apply_mode(metrics, mode):
    accept_rate = metrics["accept_rate"]
    reject_rate = metrics["reject_rate"]
    uncertain_rate = metrics["uncertain_rate"]
    if mode == "accept_only":
        uncertain_rate = uncertain_rate + reject_rate
        reject_rate = 0.0
    elif mode == "reject_only":
        uncertain_rate = uncertain_rate + accept_rate
        accept_rate = 0.0
    return {
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
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
    ap.add_argument("--mode", choices=["accept_only", "reject_only", "two_sided"], default="accept_only")
    ap.add_argument("--max_fp_accept", type=float, nargs="+", required=True)
    ap.add_argument("--max_fn_reject", type=float, nargs="+", required=True)
    ap.add_argument("--max_uncertain", type=float, nargs="+", required=True)
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
    results_two_sided = []
    for t_lower in grid_lower:
        for t_upper in grid_upper:
            if t_lower >= t_upper:
                continue
            metrics = compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper))
            dist_mode = apply_mode(metrics, args.mode)
            results.append({
                "accept_rate": dist_mode["accept_rate"],
                "reject_rate": dist_mode["reject_rate"],
                "uncertain_rate": dist_mode["uncertain_rate"],
                "fp_accept_rate": metrics["fp_accept_rate"],
                "fn_reject_rate": metrics["fn_reject_rate"],
                "ok_rate": metrics["ok_rate"],
                "t_lower": metrics["t_lower"],
                "t_upper": metrics["t_upper"],
            })
            results_two_sided.append(metrics)

    fp_limits = sorted(float(x) for x in args.max_fp_accept)
    fn_limits = sorted(float(x) for x in args.max_fn_reject)
    unc_limits = sorted(float(x) for x in args.max_uncertain)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    baseline_metrics = compute_metrics(cs_ret, y_vals, float(base["t_lower"]), float(base["t_upper"]))
    baseline_mode = apply_mode(baseline_metrics, args.mode)

    lines = []
    lines.append(f"# Stage1 Product Gates ({args.track})")
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
    lines.append(f"- mode: `{args.mode}`")
    lines.append("")
    lines.append("Baseline thresholds")
    lines.append("")
    lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append("| " + " | ".join([
        format_float(float(base["t_lower"])),
        format_float(float(base["t_upper"])),
        format_float(baseline_mode["accept_rate"]),
        format_float(baseline_mode["reject_rate"]),
        format_float(baseline_mode["uncertain_rate"]),
        format_float(baseline_metrics["fp_accept_rate"]),
        format_float(baseline_metrics["fn_reject_rate"]),
        format_float(baseline_metrics["ok_rate"]),
    ]) + " |")
    lines.append("")

    weights = {"fp": 5.0, "fn": 3.0, "unc": 1.0}
    best_overall = None
    best_key = None
    best_two_sided = None
    default_gate = (0.02, 0.01, 0.5)
    best_default = None
    for unc_lim in unc_limits:
        for fp_lim in fp_limits:
            for fn_lim in fn_limits:
                feasible = [
                    r for r in results
                    if r["fp_accept_rate"] <= fp_lim
                    and r["fn_reject_rate"] <= fn_lim
                    and r["uncertain_rate"] <= unc_lim
                ]
                feasible.sort(key=lambda r: (
                    -r["ok_rate"],
                    r["fp_accept_rate"] + r["fn_reject_rate"],
                    r["uncertain_rate"],
                ))
                lines.append(f"Gate fp<={fp_lim} fn<={fn_lim} uncertain<={unc_lim}")
                lines.append("")
                lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
                lines.append("|---|---|---|---|---|---|---|---|")
                if not feasible:
                    closest = []
                    for r in results:
                        penalty = (
                            weights["fp"] * max(0.0, r["fp_accept_rate"] - fp_lim)
                            + weights["fn"] * max(0.0, r["fn_reject_rate"] - fn_lim)
                            + weights["unc"] * max(0.0, r["uncertain_rate"] - unc_lim)
                        )
                        closest.append((penalty, r))
                    closest.sort(key=lambda x: (x[0], -x[1]["ok_rate"]))
                    lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                    lines.append("")
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
                    continue
                for r in feasible[:10]:
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
                if best_overall is None and unc_lim == unc_limits[0] and fp_lim == fp_limits[0] and fn_lim == fn_limits[0]:
                    best_overall = feasible[0]
                    best_key = (fp_lim, fn_lim, unc_lim)
                if (fp_lim, fn_lim, unc_lim) == default_gate:
                    best_default = feasible[0]
                if args.mode == "accept_only" and best_two_sided is None:
                    feasible_two_sided = [
                        r for r in results_two_sided
                        if r["fp_accept_rate"] <= fp_lim
                        and r["fn_reject_rate"] <= fn_lim
                        and r["uncertain_rate"] <= unc_lim
                    ]
                    if feasible_two_sided:
                        feasible_two_sided.sort(key=lambda r: (
                            -r["ok_rate"],
                            r["fp_accept_rate"] + r["fn_reject_rate"],
                            r["uncertain_rate"],
                        ))
                        best_two_sided = feasible_two_sided[0]

    if args.mode == "accept_only":
        lines.append("Best two_sided config (comparison)")
        lines.append("")
        lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
        lines.append("|---|---|---|---|---|---|---|---|")
        if best_two_sided is None:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append("| " + " | ".join([
                format_float(best_two_sided["t_lower"]),
                format_float(best_two_sided["t_upper"]),
                format_float(best_two_sided["accept_rate"]),
                format_float(best_two_sided["reject_rate"]),
                format_float(best_two_sided["uncertain_rate"]),
                format_float(best_two_sided["fp_accept_rate"]),
                format_float(best_two_sided["fn_reject_rate"]),
                format_float(best_two_sided["ok_rate"]),
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
    lines.append("| fp_accept_rate | fn_reject_rate | uncertain_rate | accept_rate | reject_rate | ok_rate |")
    lines.append("|---|---|---|---|---|---|")
    for r in pareto[:50]:
        lines.append("| " + " | ".join([
            format_float(r["fp_accept_rate"]),
            format_float(r["fn_reject_rate"]),
            format_float(r["uncertain_rate"]),
            format_float(r["accept_rate"]),
            format_float(r["reject_rate"]),
            format_float(r["ok_rate"]),
        ]) + " |")
    lines.append("")

    lines.append("Interpretation")
    lines.append("")
    lines.append("Lower uncertain increases coverage but can increase risk. The selected operating point favors safety gates first.")
    lines.append("")
    lines.append("Recommended config")
    lines.append("")
    if best_default is None:
        lines.append("- no feasible config under strictest gates; fallback to baseline thresholds")
    else:
        lines.append("- " + " ".join([
            f"t_lower={format_float(best_default['t_lower'])}",
            f"t_upper={format_float(best_default['t_upper'])}",
            f"fp_accept_rate={format_float(best_default['fp_accept_rate'])}",
            f"fn_reject_rate={format_float(best_default['fn_reject_rate'])}",
            f"uncertain_rate={format_float(best_default['uncertain_rate'])}",
            f"ok_rate={format_float(best_default['ok_rate'])}",
        ]))
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    if best_default is None:
        out_yaml.write_text(yaml.safe_dump({
            "tau": float(tau),
            "t_lower": float(base["t_lower"]),
            "t_upper": float(base["t_upper"]),
            "mode": args.mode,
            "fallback": "baseline",
        }, sort_keys=False), encoding="utf-8")
    else:
        out_yaml.write_text(yaml.safe_dump({
            "tau": float(tau),
            "t_lower": float(best_default["t_lower"]),
            "t_upper": float(best_default["t_upper"]),
            "mode": args.mode,
        }, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
