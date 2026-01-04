import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, stage1_outcomes

try:
    from src.utils.policy import finalize_decision as policy_finalize
except Exception:
    policy_finalize = None


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


def load_runs_jsonl(path: Path):
    mapping = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = (row.get("metadata") or {}).get("qid")
            if qid is None:
                continue
            mapping[str(qid)] = row
    return mapping


def decide_stage2_trigger(decision, cs_ret, t_lower, t_upper, guard_band, mode):
    if mode == "off":
        return False, "off"
    if mode == "always":
        return True, "always"
    if mode == "uncertain_only":
        return decision == "UNCERTAIN", "uncertain_only"
    near_lower = abs(cs_ret - t_lower) <= guard_band
    near_upper = abs(cs_ret - t_upper) <= guard_band
    if decision == "UNCERTAIN":
        return True, "uncertain"
    if near_lower:
        return True, "guard_band_lower"
    if near_upper:
        return True, "guard_band_upper"
    return False, "guarded_no"


def policy_decision(stage1_decision, stage2, entail_min, contradict_min, neutral_max):
    if policy_finalize:
        return policy_finalize(
            stage1_decision,
            stage2,
            entail_min=entail_min,
            contradict_min=contradict_min,
            neutral_max=neutral_max,
        )
    nli = (stage2 or {}).get("nli") or {}
    label_probs = nli.get("label_probs") or {}
    p_entail = float(label_probs.get("ENTAILMENT", 0.0))
    p_contra = float(label_probs.get("CONTRADICTION", 0.0))
    p_neutral = float(label_probs.get("NEUTRAL", 0.0))
    if p_entail >= entail_min and p_neutral <= neutral_max:
        return "ACCEPT", "stage2_entailment"
    if p_contra >= contradict_min and p_neutral <= neutral_max:
        return "REJECT", "stage2_contradiction"
    return "UNCERTAIN", "stage2_ambiguous"


def format_float(x):
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--tuned_thresholds_yaml", default=None)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--guard_band", type=float, default=0.02)
    ap.add_argument("--stage2_mode", choices=["off", "uncertain_only", "guarded", "always"], default="guarded")
    ap.add_argument("--runs_jsonl", default=None)
    ap.add_argument("--entail_min", type=float, default=0.9)
    ap.add_argument("--contradict_min", type=float, default=0.9)
    ap.add_argument("--neutral_max", type=float, default=0.8)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"missing in_path: {in_path}")

    df = pd.read_parquet(in_path)
    if args.logit_col not in df.columns:
        raise SystemExit(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise SystemExit(f"missing y_col {args.y_col}")

    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    df[args.logit_col] = pd.to_numeric(df[args.logit_col], errors="coerce")
    df = df.dropna(subset=[args.logit_col, args.y_col]).copy()
    df = df[(df[args.y_col] == 0) | (df[args.y_col] == 1)]
    df[args.y_col] = df[args.y_col].astype(int)
    df = df.reset_index(drop=True)
    df["row_idx"] = np.arange(len(df), dtype=np.int64)

    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    thresholds_source = Path(args.thresholds_yaml)
    t = load_thresholds(thresholds_source, args.track)
    tau = float(t.get("tau"))
    t_lower = float(t.get("t_lower"))
    t_upper = float(t.get("t_upper"))
    tuned_source = None
    if args.tuned_thresholds_yaml:
        tuned_path = Path(args.tuned_thresholds_yaml)
        if tuned_path.exists():
            tuned = load_tuned(tuned_path)
            tau = float(tuned["tau"])
            t_lower = float(tuned["t_lower"])
            t_upper = float(tuned["t_upper"])
            tuned_source = tuned_path

    runs_map = {}
    if args.runs_jsonl:
        runs_map = load_runs_jsonl(Path(args.runs_jsonl))

    stage1_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}
    final_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "defer_pos": 0, "defer_neg": 0}

    stage2_triggers = 0
    final_accept = 0
    final_reject = 0
    final_uncertain = 0

    risky = []

    for _, row in df.iterrows():
        y = int(row[args.y_col])
        qid = row["qid"] if "qid" in row else int(row["row_idx"])
        logit = float(row[args.logit_col])
        cs_ret = compute_cs_ret_from_logit(logit, tau)
        decision, _ = decide_route(cs_ret, t_lower, t_upper)
        s_out = stage1_outcomes(decision, y)
        for k in stage1_counts:
            stage1_counts[k] += s_out[k]

        trigger, trigger_reason = decide_stage2_trigger(
            decision, cs_ret, t_lower, t_upper, args.guard_band, args.stage2_mode
        )
        if trigger:
            stage2_triggers += 1

        final_decision = decision
        final_reason = "stage1_only"
        if trigger:
            if runs_map:
                row_key = str(qid)
                run_row = runs_map.get(row_key)
                stage2 = run_row.get("stage2") if run_row else None
                if stage2:
                    final_decision, final_reason = policy_decision(
                        decision,
                        stage2,
                        args.entail_min,
                        args.contradict_min,
                        args.neutral_max,
                    )
                else:
                    final_decision = "UNCERTAIN"
                    final_reason = "stage2_missing"
            else:
                final_decision = "UNCERTAIN"
                final_reason = "stage2_unavailable"

        if final_decision == "ACCEPT":
            final_accept += 1
        elif final_decision == "REJECT":
            final_reject += 1
        else:
            final_uncertain += 1

        f_out = stage1_outcomes(final_decision, y)
        for k in final_counts:
            final_counts[k] += f_out[k]

        closeness = min(abs(cs_ret - t_lower), abs(cs_ret - t_upper))
        risky.append({
            "qid": qid,
            "y": y,
            "logit": logit,
            "cs_ret": cs_ret,
            "stage1": decision,
            "final": final_decision,
            "why_stage2": trigger_reason,
            "closeness": closeness,
        })

    n_total = len(df)
    stage1_accept = stage1_counts["tp"] + stage1_counts["fp"]
    stage1_reject = stage1_counts["tn"] + stage1_counts["fn"]
    stage1_uncertain = stage1_counts["defer_pos"] + stage1_counts["defer_neg"]
    stage1_fp = stage1_counts["fp"]
    stage1_fn = stage1_counts["fn"]

    final_fp = final_counts["fp"]
    final_fn = final_counts["fn"]

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/day15_router_report_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Day15 Router Report ({args.track})")
    lines.append("")
    lines.append("Config")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n: `{n_total}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- thresholds_yaml: `{thresholds_source}`")
    lines.append(f"- tuned_thresholds_yaml: `{tuned_source}`")
    lines.append(f"- tau: `{tau}`")
    lines.append(f"- t_lower: `{t_lower}`")
    lines.append(f"- t_upper: `{t_upper}`")
    lines.append(f"- guard_band: `{args.guard_band}`")
    lines.append(f"- stage2_mode: `{args.stage2_mode}`")
    lines.append(f"- runs_jsonl: `{args.runs_jsonl}`")
    lines.append(f"- entail_min: `{args.entail_min}`")
    lines.append(f"- contradict_min: `{args.contradict_min}`")
    lines.append(f"- neutral_max: `{args.neutral_max}`")
    lines.append("")

    lines.append("Stage1 distribution")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| accept_rate | {format_float(stage1_accept / n_total)} |")
    lines.append(f"| reject_rate | {format_float(stage1_reject / n_total)} |")
    lines.append(f"| uncertain_rate | {format_float(stage1_uncertain / n_total)} |")
    lines.append(f"| fp_accept_rate | {format_float(stage1_fp / n_total)} |")
    lines.append(f"| fn_reject_rate | {format_float(stage1_fn / n_total)} |")
    lines.append(f"| ok_rate | {format_float(1.0 - (stage1_fp / n_total) - (stage1_fn / n_total))} |")
    lines.append("")

    lines.append("Stage2 trigger")
    lines.append("")
    lines.append(f"- trigger_rate: `{format_float(stage2_triggers / n_total)}`")
    lines.append("")

    lines.append("Final distribution")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| final_accept_rate | {format_float(final_accept / n_total)} |")
    lines.append(f"| final_reject_rate | {format_float(final_reject / n_total)} |")
    lines.append(f"| final_uncertain_rate | {format_float(final_uncertain / n_total)} |")
    lines.append(f"| final_fp_accept_rate | {format_float(final_fp / n_total)} |")
    lines.append(f"| final_fn_reject_rate | {format_float(final_fn / n_total)} |")
    lines.append(f"| final_ok_rate | {format_float(1.0 - (final_fp / n_total) - (final_fn / n_total))} |")
    lines.append("")

    lines.append("Confusion breakdown")
    lines.append("")
    lines.append("| split | tp | fp | tn | fn | defer_pos | defer_neg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append("| stage1 | {tp} | {fp} | {tn} | {fn} | {dp} | {dn} |".format(
        tp=stage1_counts["tp"],
        fp=stage1_counts["fp"],
        tn=stage1_counts["tn"],
        fn=stage1_counts["fn"],
        dp=stage1_counts["defer_pos"],
        dn=stage1_counts["defer_neg"],
    ))
    lines.append("| final | {tp} | {fp} | {tn} | {fn} | {dp} | {dn} |".format(
        tp=final_counts["tp"],
        fp=final_counts["fp"],
        tn=final_counts["tn"],
        fn=final_counts["fn"],
        dp=final_counts["defer_pos"],
        dn=final_counts["defer_neg"],
    ))
    lines.append("")

    lines.append("Risky cases (closest to thresholds)")
    lines.append("")
    lines.append("| qid | y | logit | cs_ret | stage1 | final | why_stage2 |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    risky_sorted = sorted(risky, key=lambda r: r["closeness"])[:20]
    for r in risky_sorted:
        lines.append("| {qid} | {y} | {logit} | {cs_ret} | {stage1} | {final} | {why} |".format(
            qid=r["qid"],
            y=r["y"],
            logit=f"{r['logit']:.4f}",
            cs_ret=f"{r['cs_ret']:.4f}",
            stage1=r["stage1"],
            final=r["final"],
            why=r["why_stage2"],
        ))
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
