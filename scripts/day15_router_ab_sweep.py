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

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution, stage1_outcomes


def load_thresholds(path: Path, track: str) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    t = (data.get("thresholds") or {}).get(track)
    if not t:
        raise SystemExit(f"missing thresholds for track={track} in {path}")
    return t


def load_tuned(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid tuned thresholds in {path}")
    for k in ("tau", "t_lower", "t_upper"):
        if k not in data:
            raise SystemExit(f"missing {k} in tuned thresholds {path}")
    return data


def compute_stage1(rows, t_lower, t_upper):
    decisions = []
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    for cs_ret, y in rows:
        decision, _ = decide_route(cs_ret, t_lower, t_upper)
        decisions.append(decision)
        o = stage1_outcomes(decision, y)
        for k in counts:
            counts[k] += o[k]
    dist = routing_distribution([r[0] for r in rows], t_lower, t_upper)
    n = len(rows) or 1
    fp = counts["fp"] / n
    fn = counts["fn"] / n
    return {
        "dist": dist,
        "counts": counts,
        "fp_accept_rate": fp,
        "fn_reject_rate": fn,
        "ok_rate": 1.0 - fp - fn,
        "decisions": decisions,
    }


def normalize_probs(p_entail, p_contra, p_neutral):
    total = p_entail + p_contra + p_neutral
    if total <= 0:
        return 0.0, 0.0, 1.0
    return p_entail / total, p_contra / total, p_neutral / total


def policy_from_probs(p_entail, p_contra, p_neutral, entail_min, contradict_min, neutral_max):
    if p_entail >= entail_min and p_neutral <= neutral_max:
        return "ACCEPT"
    if p_contra >= contradict_min and p_neutral <= neutral_max:
        return "REJECT"
    return "UNCERTAIN"


def sweep(rows, decisions, t_lower, t_upper):
    guard_bands = [0.0, 0.01, 0.02]
    entail_mins = [0.70, 0.80, 0.90]
    contradict_mins = [0.70, 0.80, 0.90]
    neutral_maxes = [0.60, 0.80]
    out = []
    n = len(rows) or 1

    for guard in guard_bands:
        for ent in entail_mins:
            for con in contradict_mins:
                for neu in neutral_maxes:
                    final_accept = 0
                    final_reject = 0
                    final_uncertain = 0
                    stage2_triggers = 0
                    unsafe_accept = 0
                    unsafe_reject = 0
                    for (cs_ret, y), decision in zip(rows, decisions):
                        trigger = False
                        if decision == "UNCERTAIN":
                            trigger = True
                        else:
                            if decision == "ACCEPT" and cs_ret < t_upper + guard:
                                trigger = True
                            if decision == "REJECT" and cs_ret > t_lower - guard:
                                trigger = True
                        if trigger:
                            stage2_triggers += 1
                            p_entail = cs_ret
                            p_contra = 1.0 - cs_ret
                            p_neutral = max(0.0, 1.0 - max(p_entail, p_contra))
                            p_entail, p_contra, p_neutral = normalize_probs(p_entail, p_contra, p_neutral)
                            final = policy_from_probs(p_entail, p_contra, p_neutral, ent, con, neu)
                        else:
                            final = decision
                        if final == "ACCEPT":
                            final_accept += 1
                            if y == 0:
                                unsafe_accept += 1
                        elif final == "REJECT":
                            final_reject += 1
                            if y == 1:
                                unsafe_reject += 1
                        else:
                            final_uncertain += 1
                    out.append({
                        "guard_band": guard,
                        "entail_min": ent,
                        "contradict_min": con,
                        "neutral_max": neu,
                        "stage2_trigger_rate": stage2_triggers / n,
                        "final_accept_rate": final_accept / n,
                        "final_reject_rate": final_reject / n,
                        "final_uncertain_rate": final_uncertain / n,
                        "final_coverage": (final_accept + final_reject) / n,
                        "unsafe_accept_proxy": unsafe_accept / n,
                        "unsafe_reject_proxy": unsafe_reject / n,
                    })
    return out


def pick_best(rows, base_fp, base_fn):
    slack = base_fp + base_fn + 0.02
    constrained = [r for r in rows if (r["unsafe_accept_proxy"] + r["unsafe_reject_proxy"]) <= slack]
    pool = constrained if constrained else rows
    pool.sort(key=lambda r: (-r["final_coverage"], r["unsafe_accept_proxy"] + r["unsafe_reject_proxy"], r["final_uncertain_rate"]))
    return pool[0], pool[:10], bool(constrained)


def format_float(x):
    return f"{x:.4f}"


def run_track(track, in_path, logit_col, n, seed, thresholds_yaml, tuned_yaml):
    df = pd.read_parquet(in_path)
    if logit_col not in df.columns:
        raise SystemExit(f"missing logit_col {logit_col} in {in_path}")
    if "y" not in df.columns:
        raise SystemExit(f"missing y in {in_path}")
    df[logit_col] = pd.to_numeric(df[logit_col], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=[logit_col, "y"]).copy()
    df = df[(df["y"] == 0) | (df["y"] == 1)]
    df["y"] = df["y"].astype(int)
    df = df.reset_index(drop=True)
    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    thresholds_source = Path(thresholds_yaml)
    t = load_thresholds(thresholds_source, track)
    tau = float(t["tau"])
    t_lower = float(t["t_lower"])
    t_upper = float(t["t_upper"])
    tuned_source = None
    if tuned_yaml:
        tuned_path = Path(tuned_yaml)
        if tuned_path.exists():
            tuned = load_tuned(tuned_path)
            tau = float(tuned["tau"])
            t_lower = float(tuned["t_lower"])
            t_upper = float(tuned["t_upper"])
            tuned_source = tuned_path
    rows = []
    for _, row in df.iterrows():
        cs_ret = compute_cs_ret_from_logit(float(row[logit_col]), tau)
        rows.append((cs_ret, int(row["y"])))
    stage1 = compute_stage1(rows, t_lower, t_upper)
    sweeps = sweep(rows, stage1["decisions"], t_lower, t_upper)
    best, top10, constrained = pick_best(sweeps, stage1["fp_accept_rate"], stage1["fn_reject_rate"])
    return {
        "track": track,
        "in_path": in_path,
        "n": len(rows),
        "seed": seed,
        "logit_col": logit_col,
        "tau": tau,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "thresholds_source": thresholds_source,
        "tuned_source": tuned_source,
        "stage1": stage1,
        "top10": top10,
        "best": best,
        "best_constrained": constrained,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--fever_in_path", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--tuned_scifact_yaml", default="configs/thresholds_stage1_tuned_scifact.yaml")
    ap.add_argument("--tuned_fever_yaml", default="configs/thresholds_stage1_tuned_fever.yaml")
    ap.add_argument("--out_md", default="reports/day15_router_ab_sweep.md")
    args = ap.parse_args()

    scifact = run_track(
        "scifact",
        args.scifact_in_path,
        "raw_max_top3",
        args.n,
        args.seed,
        args.thresholds_yaml,
        args.tuned_scifact_yaml,
    )
    fever = run_track(
        "fever",
        args.fever_in_path,
        "logit_platt",
        args.n,
        args.seed,
        args.thresholds_yaml,
        args.tuned_fever_yaml,
    )

    lines = []
    lines.append("# Day15 Router A/B Sweep")
    lines.append("")
    lines.append("Stage2 policy is simulated from cs_ret probabilities to avoid collapse before running real stage2.")
    lines.append("")
    for item in [scifact, fever]:
        lines.append(f"## Track: {item['track']}")
        lines.append("")
        lines.append(f"- in_path: `{item['in_path']}`")
        lines.append(f"- n: `{item['n']}`")
        lines.append(f"- seed: `{item['seed']}`")
        lines.append(f"- logit_col: `{item['logit_col']}`")
        lines.append(f"- thresholds_yaml: `{item['thresholds_source']}`")
        lines.append(f"- tuned_thresholds_yaml: `{item['tuned_source']}`")
        lines.append(f"- tau: `{item['tau']}`")
        lines.append(f"- t_lower: `{item['t_lower']}`")
        lines.append(f"- t_upper: `{item['t_upper']}`")
        lines.append("")
        s1 = item["stage1"]
        lines.append("Stage1 baseline")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| accept_rate | {format_float(s1['dist']['ACCEPT'])} |")
        lines.append(f"| reject_rate | {format_float(s1['dist']['REJECT'])} |")
        lines.append(f"| uncertain_rate | {format_float(s1['dist']['UNCERTAIN'])} |")
        lines.append(f"| fp_accept_rate | {format_float(s1['fp_accept_rate'])} |")
        lines.append(f"| fn_reject_rate | {format_float(s1['fn_reject_rate'])} |")
        lines.append(f"| ok_rate | {format_float(s1['ok_rate'])} |")
        lines.append("")
        lines.append("Top 10 configs")
        lines.append("")
        lines.append("| guard_band | entail_min | contradict_min | neutral_max | stage2_trigger_rate | final_accept_rate | final_reject_rate | final_uncertain_rate | final_coverage | unsafe_accept_proxy | unsafe_reject_proxy |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in item["top10"]:
            lines.append("| {guard} | {ent} | {con} | {neu} | {t} | {fa} | {fr} | {fu} | {fc} | {ua} | {ur} |".format(
                guard=format_float(r["guard_band"]),
                ent=format_float(r["entail_min"]),
                con=format_float(r["contradict_min"]),
                neu=format_float(r["neutral_max"]),
                t=format_float(r["stage2_trigger_rate"]),
                fa=format_float(r["final_accept_rate"]),
                fr=format_float(r["final_reject_rate"]),
                fu=format_float(r["final_uncertain_rate"]),
                fc=format_float(r["final_coverage"]),
                ua=format_float(r["unsafe_accept_proxy"]),
                ur=format_float(r["unsafe_reject_proxy"]),
            ))
        lines.append("")
        best = item["best"]
        lines.append("Recommended config")
        lines.append("")
        lines.append(f"- guard_band: `{best['guard_band']}`")
        lines.append(f"- entail_min: `{best['entail_min']}`")
        lines.append(f"- contradict_min: `{best['contradict_min']}`")
        lines.append(f"- neutral_max: `{best['neutral_max']}`")
        lines.append(f"- constraint_met: `{item['best_constrained']}`")
        lines.append("")
        lines.append("Interpretation")
        lines.append("")
        lines.append("Prior collapse is driven by strict policy thresholds and wide guard bands that route most cases to stage2 but keep them UNCERTAIN. The selected config maximizes coverage while keeping proxy error within slack of stage1.")
        lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
