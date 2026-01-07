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


def fit_tau(logits, y, tau_min=0.2, tau_max=2.0, tau_steps=200):
    y_arr = np.asarray(y, dtype=float)
    taus = np.linspace(tau_min, tau_max, tau_steps)
    best_tau = None
    best_val = None
    scores = []
    eps = 1e-12
    for tau in taus:
        probs = np.asarray([compute_cs_ret_from_logit(v, tau) for v in logits], dtype=float)
        probs = np.clip(probs, eps, 1.0 - eps)
        nll = -np.mean(y_arr * np.log(probs) + (1.0 - y_arr) * np.log(1.0 - probs))
        scores.append((float(tau), float(nll)))
        if best_val is None or nll < best_val:
            best_val = nll
            best_tau = float(tau)
    return best_tau, scores


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
    ap.add_argument("--max_fp_accept_rate", type=float)
    ap.add_argument("--max_fn_reject_rate", type=float)
    ap.add_argument("--tau_source", choices=["config", "fit", "manual"], default="config")
    ap.add_argument("--tau_fit_report_md")
    ap.add_argument("--objective", choices=["ok_rate", "ok_only", "pareto", "weighted"], default="ok_rate")
    ap.add_argument("--lambda_uncertain", type=float, default=0.0)
    ap.add_argument("--export_budget", type=float)
    ap.add_argument("--keep_existing_yaml", action="store_true")
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_yaml", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    thresholds_path = Path("configs/thresholds.yaml")
    base = load_thresholds(thresholds_path, args.track)
    tau = float(base["tau"])

    df = pd.read_parquet(in_path)
    n_raw = len(df)
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

    n_used = len(df)
    logits = df[args.logit_col].astype(float).tolist()
    y_vals = df[args.y_col].astype(int).tolist()
    tau_source = args.tau_source
    tau_fit_scores = None
    if tau_source == "manual":
        if args.tau is None:
            raise ValueError("tau_source=manual requires --tau")
        tau = float(args.tau)
    elif tau_source == "fit":
        tau, tau_fit_scores = fit_tau(logits, y_vals)
    else:
        if args.tau is not None:
            tau = float(args.tau)

    if tau_fit_scores is not None and args.tau_fit_report_md:
        tau_report = Path(args.tau_fit_report_md)
        tau_report.parent.mkdir(parents=True, exist_ok=True)
        top = sorted(tau_fit_scores, key=lambda x: x[1])[:10]
        lines_tau = []
        lines_tau.append(f"# Stage1 Tau Fit ({args.track})")
        lines_tau.append("")
        lines_tau.append(f"- in_path: `{in_path}`")
        lines_tau.append(f"- n: {len(df)}")
        lines_tau.append(f"- seed: {args.seed}")
        lines_tau.append(f"- logit_col: `{args.logit_col}`")
        lines_tau.append(f"- y_col: `{args.y_col}`")
        lines_tau.append(f"- metric: nll")
        lines_tau.append(f"- best_tau: {tau}")
        lines_tau.append("")
        lines_tau.append("| tau | nll |")
        lines_tau.append("|---:|---:|")
        for t, s in top:
            lines_tau.append(f"| {t:.6f} | {s:.6f} |")
        lines_tau.append("")
        lines_tau.append("## Repro command")
        lines_tau.append("```")
        lines_tau.append(" ".join(sys.argv))
        lines_tau.append("```")
        tau_report.write_text("\n".join(lines_tau) + "\n", encoding="utf-8")

    cs_ret = [compute_cs_ret_from_logit(x, tau) for x in logits]

    grid_lower = np.linspace(0.0, 0.99, args.grid_lower_steps)
    grid_upper = np.linspace(0.01, 1.0, args.grid_upper_steps)

    baseline = compute_rates(cs_ret, y_vals, float(base["t_lower"]), float(base["t_upper"]))

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
                "label": "grid",
            })

    budgets = sorted(args.budgets)
    if args.export_budget is not None and args.export_budget not in budgets:
        budgets.append(float(args.export_budget))
        budgets = sorted(budgets)
    results.append({
        "t_lower": float(base["t_lower"]),
        "t_upper": float(base["t_upper"]),
        "accept_rate": baseline["accept_rate"],
        "reject_rate": baseline["reject_rate"],
        "uncertain_rate": baseline["uncertain_rate"],
        "fp_accept_rate": baseline["fp_accept_rate"],
        "fn_reject_rate": baseline["fn_reject_rate"],
        "ok_rate": baseline["ok_rate"],
        "label": "baseline_candidate",
    })
    picks = {}
    top10 = {}
    feasible_counts = {}
    baseline_feasible = {}
    fallback_notes = {}
    best_near = {}
    best_near_penalty = {}

    def select_pick(candidates):
        pick = None
        if args.objective in ("ok_only", "ok_rate"):
            ranked = sorted(candidates, key=lambda r: (-r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]))
            pick = ranked[0] if ranked else None
            return pick, ranked
        elif args.objective == "weighted":
            lam = float(args.lambda_uncertain)
            for r in candidates:
                r["score"] = r["ok_rate"] - lam * r["uncertain_rate"]
            ranked = sorted(candidates, key=lambda r: (-r["score"], -r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]))
            pick = ranked[0] if ranked else None
            return pick, ranked
        else:
            pareto = []
            for r in candidates:
                dominated = False
                for o in candidates:
                    if o is r:
                        continue
                    if (o["fp_accept_rate"] + o["fn_reject_rate"]) <= (r["fp_accept_rate"] + r["fn_reject_rate"]) and o["uncertain_rate"] <= r["uncertain_rate"]:
                        if (o["fp_accept_rate"] + o["fn_reject_rate"]) < (r["fp_accept_rate"] + r["fn_reject_rate"]) or o["uncertain_rate"] < r["uncertain_rate"]:
                            dominated = True
                            break
                if not dominated:
                    pareto.append(r)
            ranked = sorted(pareto, key=lambda r: (-r["ok_rate"], r["uncertain_rate"], r["fp_accept_rate"] + r["fn_reject_rate"]))
            if ranked:
                pick = ranked[0]
                return pick, ranked
            ranked = sorted(candidates, key=lambda r: (-r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]))
            pick = ranked[0] if ranked else None
            return pick, ranked
    for b in budgets:
        def penalty(r):
            p = max(0.0, r["uncertain_rate"] - b) / max(b, 1e-12)
            if args.max_fp_accept_rate is not None:
                p += max(0.0, r["fp_accept_rate"] - args.max_fp_accept_rate) / max(args.max_fp_accept_rate, 1e-12)
            if args.max_fn_reject_rate is not None:
                p += max(0.0, r["fn_reject_rate"] - args.max_fn_reject_rate) / max(args.max_fn_reject_rate, 1e-12)
            return p

        near = min(
            results,
            key=lambda r: (penalty(r), -r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"], r["uncertain_rate"]),
        )
        best_near[b] = near
        best_near_penalty[b] = penalty(near)

        candidates = [r for r in results if r["uncertain_rate"] <= b]
        if args.max_fp_accept_rate is not None:
            candidates = [r for r in candidates if r["fp_accept_rate"] <= args.max_fp_accept_rate]
        if args.max_fn_reject_rate is not None:
            candidates = [r for r in candidates if r["fn_reject_rate"] <= args.max_fn_reject_rate]
        feasible_counts[b] = len(candidates)
        baseline_feasible[b] = any(r.get("label") == "baseline_candidate" for r in candidates)
        fallback_notes[b] = None
        pick, ranked = select_pick(candidates)
        picks[b] = pick
        top10[b] = ranked[:10] if ranked else []

    export_budget = None
    if args.export_budget is not None:
        export_budget = float(args.export_budget)
    else:
        for b in budgets:
            if feasible_counts[b] > 0:
                export_budget = b
                break

    best = picks[export_budget] if export_budget is not None else None
    if export_budget is not None and best is not None and feasible_counts[export_budget] > 0:
        tl = best["t_lower"]
        tu = best["t_upper"]
        lower_grid = np.linspace(max(0.0, tl - 0.02), min(0.99, tl + 0.02), 50)
        upper_grid = np.linspace(max(0.01, tu - 0.02), min(1.0, tu + 0.02), 50)
        refined = []
        for t_lower in lower_grid:
            for t_upper in upper_grid:
                if t_lower >= t_upper:
                    continue
                r = compute_rates(cs_ret, y_vals, float(t_lower), float(t_upper))
                if r["uncertain_rate"] > export_budget:
                    continue
                if args.max_fp_accept_rate is not None and r["fp_accept_rate"] > args.max_fp_accept_rate:
                    continue
                if args.max_fn_reject_rate is not None and r["fn_reject_rate"] > args.max_fn_reject_rate:
                    continue
                refined.append({
                    "t_lower": float(t_lower),
                    "t_upper": float(t_upper),
                    "accept_rate": r["accept_rate"],
                    "reject_rate": r["reject_rate"],
                    "uncertain_rate": r["uncertain_rate"],
                    "fp_accept_rate": r["fp_accept_rate"],
                    "fn_reject_rate": r["fn_reject_rate"],
                    "ok_rate": r["ok_rate"],
                    "label": "refined",
                })
        refined_pick, _ = select_pick(refined)
        if refined_pick is not None:
            best = refined_pick
            picks[export_budget] = refined_pick

    yaml_written = False
    yaml_payload = None
    out_yaml_preexisted = False
    out_yaml_removed = False

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
    lines.append(f"- n_raw: `{n_raw}`")
    lines.append(f"- n_used: `{n_used}`")
    lines.append(f"- n: `{len(df)}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- tau: `{tau}`")
    lines.append(f"- tau_source: `{tau_source}`")
    if args.tau_fit_report_md:
        lines.append(f"- tau_fit_report_md: `{args.tau_fit_report_md}`")
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
        lines.append(f"- feasible_configs: `{feasible_counts[b]}`")
        lines.append(f"- baseline_candidate_feasible: `{baseline_feasible[b]}`")
        near = best_near[b]
        lines.append(f"- best_near_feasible: t_lower={near['t_lower']} t_upper={near['t_upper']} fp={near['fp_accept_rate']} fn={near['fn_reject_rate']} uncertain={near['uncertain_rate']} penalty={best_near_penalty[b]}")
        if fallback_notes[b]:
            lines.append(f"- note: `{fallback_notes[b]}`")
        lines.append("")
        score_header = " | score" if args.objective == "weighted" else ""
        score_align = " |---:" if args.objective == "weighted" else ""
        lines.append("| t_lower | t_upper | accept_rate | reject_rate | uncertain_rate | fp_accept_rate | fn_reject_rate | ok_rate | label" + score_header + " |")
        lines.append("|---|---|---|---|---|---|---|---|---|" + ("---|" if args.objective == "weighted" else ""))
        rows = top10[b]
        if not rows:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a" + (" | n/a" if args.objective == "weighted" else "") + " |")
        else:
            for r in rows:
                base_cols = [
                    format_float(r["t_lower"]),
                    format_float(r["t_upper"]),
                    format_float(r["accept_rate"]),
                    format_float(r["reject_rate"]),
                    format_float(r["uncertain_rate"]),
                    format_float(r["fp_accept_rate"]),
                    format_float(r["fn_reject_rate"]),
                    format_float(r["ok_rate"]),
                    r.get("label", "grid"),
                ]
                if args.objective == "weighted":
                    base_cols.append(format_float(r.get("score", 0.0)))
                lines.append("| " + " | ".join(base_cols) + " |")
        lines.append("")
        pick = picks[b]
        lines.append("Recommended thresholds")
        lines.append("")
        if pick is None:
            lines.append("- none (no configs met uncertain budget)")
        else:
            lines.append(f"- t_lower: `{pick['t_lower']}`")
            lines.append(f"- t_upper: `{pick['t_upper']}`")
            lines.append(f"- fp_accept_rate: `{pick['fp_accept_rate']}`")
            lines.append(f"- fn_reject_rate: `{pick['fn_reject_rate']}`")
            lines.append(f"- ok_rate: `{pick['ok_rate']}`")
            lines.append(f"- uncertain_rate: `{pick['uncertain_rate']}`")
        lines.append("")

    out_yaml = Path(args.out_yaml) if args.out_yaml else Path(f"configs/thresholds_stage1_tuned_{args.track}.yaml")
    out_yaml_preexisted = out_yaml.exists()
    if export_budget is None or best is None or feasible_counts.get(export_budget, 0) == 0:
        lines.append("YAML output")
        lines.append("")
        lines.append("- written_to_yaml: none (no feasible config for export budget)")
        lines.append(f"- out_yaml_preexisted: {out_yaml_preexisted}")
        if out_yaml_preexisted and not args.keep_existing_yaml:
            out_yaml.unlink()
            out_yaml_removed = True
        lines.append(f"- out_yaml_removed: {out_yaml_removed}")
        lines.append("")
    else:
        yaml_written = True
        yaml_payload = {
            "tau": float(tau),
            "t_lower": float(best["t_lower"]),
            "t_upper": float(best["t_upper"]),
        }
        lines.append("YAML output")
        lines.append("")
        lines.append(f"- written_to_yaml: t_lower={best['t_lower']} t_upper={best['t_upper']} tau={tau} export_budget={export_budget}")
        lines.append(f"- out_yaml_preexisted: {out_yaml_preexisted}")
        lines.append(f"- out_yaml_removed: {out_yaml_removed}")
        lines.append("")

    lines.append("Interpretation")
    lines.append("")
    lines.append("Selection uses constraints if provided, then objective. Pareto objective uses the frontier of lower (fp+fn) and lower uncertain, then picks the highest ok_rate. Lower uncertain budgets reduce deferrals but can increase fp or fn; higher budgets allow more deferrals and typically improve ok_rate. Use the budget that matches expected stage2 capacity.")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    if yaml_written and yaml_payload is not None:
        out_yaml.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
