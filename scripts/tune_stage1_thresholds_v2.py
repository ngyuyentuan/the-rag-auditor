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
    }


def format_float(x):
    return f"{x:.4f}"


def select_best(rows):
    rows.sort(key=lambda r: (
        r["uncertain_rate"],
        -r["ok_rate"],
        r["fp_accept_rate"],
        r["t_upper"],
        r["t_lower"],
    ))
    return rows[0] if rows else None


def top10(rows):
    rows.sort(key=lambda r: (
        r["uncertain_rate"],
        -r["ok_rate"],
        r["fp_accept_rate"],
        r["t_upper"],
        r["t_lower"],
    ))
    return rows[:10]


def validate_choice(cs_ret, y_vals, choice, fp_lim, fn_lim, unc_lim):
    if choice is None:
        return None
    m = compute_metrics(cs_ret, y_vals, float(choice["t_lower"]), float(choice["t_upper"]))
    ok = (
        m["fp_accept_rate"] <= fp_lim
        and m["fn_reject_rate"] <= fn_lim
        and m["uncertain_rate"] <= unc_lim
    )
    return m if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--mode", choices=["accept_only", "accept_reject"], default="accept_only")
    ap.add_argument("--grid_upper_steps", type=int, default=200)
    ap.add_argument("--grid_lower_steps", type=int, default=50)
    ap.add_argument("--max_fp_accept", type=float, nargs="+", required=True)
    ap.add_argument("--max_fn_reject", type=float, nargs="+", required=True)
    ap.add_argument("--max_uncertain", type=float, nargs="+", required=True)
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

    base_metrics = compute_metrics(cs_ret, y_vals, float(base["t_lower"]), float(base["t_upper"]))

    grid_upper = np.linspace(0.01, 1.0, args.grid_upper_steps)
    grid_lower = np.linspace(0.0, 0.99, args.grid_lower_steps)

    candidates = []
    if args.mode == "accept_only":
        t_lower = 0.0
        for t_upper in grid_upper:
            r = compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper))
            candidates.append({
                "t_lower": float(t_lower),
                "t_upper": float(t_upper),
                **r,
            })
    else:
        for t_lower in grid_lower:
            for t_upper in grid_upper:
                if t_lower >= t_upper:
                    continue
                r = compute_metrics(cs_ret, y_vals, float(t_lower), float(t_upper))
                candidates.append({
                    "t_lower": float(t_lower),
                    "t_upper": float(t_upper),
                    **r,
                })

    fp_limits = sorted(float(x) for x in args.max_fp_accept)
    fn_limits = sorted(float(x) for x in args.max_fn_reject)
    unc_limits = sorted(float(x) for x in args.max_uncertain)

    results = {}
    for fp_lim in fp_limits:
        for fn_lim in fn_limits:
            for unc_lim in unc_limits:
                feasible = [
                    r for r in candidates
                    if r["fp_accept_rate"] <= fp_lim
                    and r["fn_reject_rate"] <= fn_lim
                    and r["uncertain_rate"] <= unc_lim
                    and (args.mode != "accept_only" or r["t_upper"] > 0.05)
                ]
                results[(fp_lim, fn_lim, unc_lim)] = {
                    "best": select_best(list(feasible)),
                    "top10": top10(list(feasible)),
                }

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_threshold_tuning_v2_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Threshold Tuning v2 ({args.track})")
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

    lines.append("Constraints and best configs")
    lines.append("")
    for fp_lim in fp_limits:
        for fn_lim in fn_limits:
            for unc_lim in unc_limits:
                key = (fp_lim, fn_lim, unc_lim)
                lines.append(f"Constraints fp<={fp_lim} fn<={fn_lim} uncertain<={unc_lim}")
                lines.append("")
                best = results[key]["best"]
                validated = validate_choice(cs_ret, y_vals, best, fp_lim, fn_lim, unc_lim)
                if validated is None:
                    lines.append("- no feasible configs")
                    lines.append("")
                    continue
                lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
                lines.append("|---|---|---|---|---|---|---|---|")
                lines.append("| {tl} | {tu} | {a} | {r} | {u} | {fp} | {fn} | {ok} |".format(
                    tl=format_float(best["t_lower"]),
                    tu=format_float(best["t_upper"]),
                    a=format_float(best["accept_rate"]),
                    r=format_float(best["reject_rate"]),
                    u=format_float(best["uncertain_rate"]),
                    fp=format_float(best["fp_accept_rate"]),
                    fn=format_float(best["fn_reject_rate"]),
                    ok=format_float(best["ok_rate"]),
                ))
                lines.append("")
                lines.append("Top 10 feasible configs")
                lines.append("")
                lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for r in results[key]["top10"]:
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

    lines.append("Pareto view")
    lines.append("")
    for fp_lim in fp_limits:
        for unc_lim in unc_limits:
            feasible = [
                r for r in candidates
                if r["fp_accept_rate"] <= fp_lim and r["uncertain_rate"] <= unc_lim
            ]
            best = select_best(list(feasible))
            if best is None:
                lines.append(f"- fp<={fp_lim} uncertain<={unc_lim}: no feasible configs")
            else:
                lines.append("- fp<={fp} uncertain<={unc}: t_lower={tl} t_upper={tu} fp={fpv} fn={fnv} uncertain={uncv}".format(
                    fp=fp_lim,
                    unc=unc_lim,
                    tl=format_float(best["t_lower"]),
                    tu=format_float(best["t_upper"]),
                    fpv=format_float(best["fp_accept_rate"]),
                    fnv=format_float(best["fn_reject_rate"]),
                    uncv=format_float(best["uncertain_rate"]),
                ))
    lines.append("")
    lines.append("Interpretation")
    lines.append("")
    lines.append("accept_only is recommended for product safety because it avoids catastrophic false rejects by deferring most cases to Stage2. Choose lower max_fp_accept for higher safety and adjust max_uncertain to balance Stage2 capacity and user experience.")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    prod_key = (0.02, 0.01, 0.50)
    prod_best = results.get(prod_key, {}).get("best")
    prod_validated = validate_choice(cs_ret, y_vals, prod_best, *prod_key) if prod_best else None
    out_yaml = Path(args.out_yaml) if args.out_yaml else Path(f"configs/thresholds_stage1_prod_{args.track}.yaml")
    if prod_validated is None:
        lines.append("No feasible config for prod constraints; falling back to baseline thresholds.")
        lines.append("")
        out_yaml.write_text(yaml.safe_dump({
            "tau": float(tau),
            "t_lower": float(base["t_lower"]),
            "t_upper": float(base["t_upper"]),
            "mode": args.mode,
            "max_fp_accept": prod_key[0],
            "max_fn_reject": prod_key[1],
            "max_uncertain": prod_key[2],
            "fallback": "baseline",
        }, sort_keys=False), encoding="utf-8")
    else:
        out_yaml.write_text(yaml.safe_dump({
            "tau": float(tau),
            "t_lower": float(prod_best["t_lower"]),
            "t_upper": float(prod_best["t_upper"]),
            "mode": args.mode,
            "max_fp_accept": prod_key[0],
            "max_fn_reject": prod_key[1],
            "max_uncertain": prod_key[2],
        }, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
