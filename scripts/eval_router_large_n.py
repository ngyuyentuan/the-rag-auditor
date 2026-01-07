import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution


def load_thresholds(path, track):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid thresholds yaml: {path}")
    block = data.get("thresholds") or {}
    cfg = block.get(track)
    if not isinstance(cfg, dict):
        if all(k in data for k in ["t_lower", "t_upper", "tau"]):
            cfg = data
        else:
            raise SystemExit(f"missing thresholds for track={track} in {path}")
    t_lower = cfg.get("t_lower")
    t_upper = cfg.get("t_upper")
    tau = cfg.get("tau")
    if t_lower is None or t_upper is None or tau is None:
        raise SystemExit(f"incomplete thresholds for track={track} in {path}")
    return float(t_lower), float(t_upper), float(tau)


def load_router_config(path):
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def parse_demo_paths():
    path = Path("reports/demo_stage1.md")
    if not path.exists():
        return None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = []
    for line in lines:
        s = line.strip()
        if s.startswith("- in_path:"):
            parts = s.split("`")
            if len(parts) >= 2 and parts[1]:
                paths.append(parts[1])
    if len(paths) >= 2:
        return paths[0], paths[1]
    return None, None


def find_runs_jsonl(track):
    runs_dir = Path("runs")
    candidates = [
        runs_dir / f"day12_{track}_500_e2e.jsonl",
        runs_dir / f"day12_{track}_500_e2e_always.jsonl",
        runs_dir / f"day12_{track}_500_e2e_random.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in runs_dir.glob(f"day12_{track}_*_e2e*.jsonl"):
        return p
    return None


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_cs_ret_array(logits, tau):
    return np.asarray([compute_cs_ret_from_logit(float(v), float(tau)) for v in logits], dtype=np.float64)


def percentile(values, p):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p))


def compute_rates(decisions, y):
    n = len(decisions)
    dec = np.asarray(decisions, dtype=object)
    y_arr = np.asarray(y, dtype=int) if y is not None else None
    accept_rate = float(np.mean(dec == "ACCEPT")) if n else 0.0
    reject_rate = float(np.mean(dec == "REJECT")) if n else 0.0
    uncertain_rate = float(np.mean(dec == "UNCERTAIN")) if n else 0.0
    fp_rate = None
    fn_rate = None
    ok_rate = None
    if y_arr is not None and len(y_arr) == n:
        fp_rate = float(np.mean((dec == "ACCEPT") & (y_arr == 0)))
        fn_rate = float(np.mean((dec == "REJECT") & (y_arr == 1)))
        ok_rate = 1.0 - fp_rate - fn_rate
    return {
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "fp_accept_rate": fp_rate,
        "fn_reject_rate": fn_rate,
        "ok_rate": ok_rate,
    }


def bootstrap_ci(policy_a, policy_b, y, n_boot, seed):
    if n_boot <= 0 or y is None:
        return {}, {}
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y, dtype=int)
    a = np.asarray(policy_a, dtype=object)
    b = np.asarray(policy_b, dtype=object)
    stats_a = {"ok_rate": [], "fp_accept_rate": [], "fn_reject_rate": [], "uncertain_rate": []}
    stats_b = {"ok_rate": [], "fp_accept_rate": [], "fn_reject_rate": [], "uncertain_rate": []}
    delta = {"ok_rate": [], "fp_accept_rate": [], "fn_reject_rate": [], "uncertain_rate": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ya = y_arr[idx]
        da = a[idx]
        db = b[idx]
        fp_a = float(np.mean((da == "ACCEPT") & (ya == 0))) if n else 0.0
        fn_a = float(np.mean((da == "REJECT") & (ya == 1))) if n else 0.0
        fp_b = float(np.mean((db == "ACCEPT") & (ya == 0))) if n else 0.0
        fn_b = float(np.mean((db == "REJECT") & (ya == 1))) if n else 0.0
        ok_a = 1.0 - fp_a - fn_a
        ok_b = 1.0 - fp_b - fn_b
        unc_a = float(np.mean(da == "UNCERTAIN")) if n else 0.0
        unc_b = float(np.mean(db == "UNCERTAIN")) if n else 0.0
        stats_a["ok_rate"].append(ok_a)
        stats_a["fp_accept_rate"].append(fp_a)
        stats_a["fn_reject_rate"].append(fn_a)
        stats_a["uncertain_rate"].append(unc_a)
        stats_b["ok_rate"].append(ok_b)
        stats_b["fp_accept_rate"].append(fp_b)
        stats_b["fn_reject_rate"].append(fn_b)
        stats_b["uncertain_rate"].append(unc_b)
        delta["ok_rate"].append(ok_b - ok_a)
        delta["fp_accept_rate"].append(fp_b - fp_a)
        delta["fn_reject_rate"].append(fn_b - fn_a)
        delta["uncertain_rate"].append(unc_b - unc_a)
    def pct(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    out_a = {k: pct(v) for k, v in stats_a.items()}
    out_b = {k: pct(v) for k, v in stats_b.items()}
    out_delta = {k: pct(v) for k, v in delta.items()}
    return {"policy_a": out_a, "policy_b": out_b, "delta": out_delta}


def compute_stage2_final(stage2, entail_min, contradict_min, neutral_max):
    nli = (stage2 or {}).get("nli") or {}
    probs = nli.get("label_probs") or {}
    if probs:
        p_e = float(probs.get("ENTAILMENT", 0.0))
        p_c = float(probs.get("CONTRADICTION", 0.0))
        p_n = float(probs.get("NEUTRAL", 0.0))
        if p_e >= entail_min and p_n <= neutral_max:
            return "ACCEPT"
        if p_c >= contradict_min and p_n <= neutral_max:
            return "REJECT"
        return "UNCERTAIN"
    label = nli.get("top_label")
    prob = nli.get("top_prob")
    if label and prob is not None:
        p = float(prob)
        if label == "ENTAILMENT" and p >= entail_min:
            return "ACCEPT"
        if label == "CONTRADICTION" and p >= contradict_min:
            return "REJECT"
        if label == "NEUTRAL" and p <= neutral_max:
            return "UNCERTAIN"
    return None


def guard_trigger(cs_ret, decision, t_lower, t_upper, guard_band):
    if decision == "UNCERTAIN":
        return True
    if cs_ret is None:
        return False
    v = float(cs_ret)
    return (abs(v - t_lower) <= guard_band) or (abs(v - t_upper) <= guard_band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["scifact", "fever"])
    ap.add_argument("--runs_jsonl")
    ap.add_argument("--in_path")
    ap.add_argument("--logit_col")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--thresholds_yaml", default="configs/thresholds.yaml")
    ap.add_argument("--router_config_yaml")
    ap.add_argument("--tuned_thresholds_yaml")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--guard_band", type=float, default=0.0)
    ap.add_argument("--entail_min", type=float, default=0.7)
    ap.add_argument("--contradict_min", type=float, default=0.7)
    ap.add_argument("--neutral_max", type=float, default=0.6)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    out_md = args.out_md or f"reports/router_eval_{args.track}_large.md"
    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_lower, t_upper, tau = load_thresholds(args.thresholds_yaml, args.track)
    if args.tuned_thresholds_yaml and Path(args.tuned_thresholds_yaml).exists():
        t_lower, t_upper, tau = load_thresholds(args.tuned_thresholds_yaml, args.track)
    if args.tau is not None:
        tau = float(args.tau)

    router_cfg = load_router_config(args.router_config_yaml)
    guard_band = float(router_cfg.get("guard_band", args.guard_band))
    entail_min = float(router_cfg.get("entail_min", args.entail_min))
    contradict_min = float(router_cfg.get("contradict_min", args.contradict_min))
    neutral_max = float(router_cfg.get("neutral_max", args.neutral_max))

    runs_path = Path(args.runs_jsonl) if args.runs_jsonl else None
    if runs_path is None:
        runs_path = find_runs_jsonl(args.track)

    lines = []
    lines.append(f"# Router Eval Large-N ({args.track})")
    lines.append("")
    lines.append(f"- thresholds_yaml: `{args.thresholds_yaml}`")
    if args.tuned_thresholds_yaml:
        lines.append(f"- tuned_thresholds_yaml: `{args.tuned_thresholds_yaml}`")
    if args.router_config_yaml:
        lines.append(f"- router_config_yaml: `{args.router_config_yaml}`")
    lines.append(f"- t_lower: {t_lower}")
    lines.append(f"- t_upper: {t_upper}")
    lines.append(f"- tau: {tau}")
    lines.append(f"- guard_band: {guard_band}")
    lines.append(f"- entail_min: {entail_min}")
    lines.append(f"- contradict_min: {contradict_min}")
    lines.append(f"- neutral_max: {neutral_max}")
    lines.append("")

    if runs_path and runs_path.exists():
        rows = list(iter_jsonl(runs_path))
        if args.n is not None and args.n < len(rows):
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(rows), size=args.n, replace=False)
            rows = [rows[i] for i in idx]
        y_vals = []
        stage1_decisions = []
        final_a = []
        final_b = []
        trigger_b = []
        stage2_ran = []
        total_ms = []
        rerank_ms = []
        nli_ms = []
        missing_final = 0
        for row in rows:
            gt = row.get("ground_truth") or {}
            gold = row.get("gold") or {}
            y_val = gt.get("y", gold.get("y"))
            if y_val is not None:
                y_vals.append(int(y_val))
            stage1 = row.get("stage1") or {}
            cs_ret = stage1.get("cs_ret")
            if cs_ret is None:
                decision = stage1.get("route_decision")
                if decision is None:
                    decision = "UNCERTAIN"
            else:
                decision = decide_route(float(cs_ret), t_lower, t_upper)[0]
            stage1_decisions.append(decision)
            final_a.append(decision)
            trigger = guard_trigger(cs_ret, decision, t_lower, t_upper, guard_band)
            trigger_b.append(trigger)
            if trigger:
                stage2_final = compute_stage2_final(row.get("stage2"), entail_min, contradict_min, neutral_max)
                if stage2_final is None:
                    final_b.append("UNCERTAIN")
                    missing_final += 1
                else:
                    final_b.append(stage2_final)
            else:
                final_b.append(decision)
            stage2 = row.get("stage2") or {}
            stage2_ran.append(bool(stage2.get("ran")))
            timing = row.get("timing_ms") or {}
            if "total_ms" in timing:
                total_ms.append(float(timing.get("total_ms", 0.0)))
            if "rerank_ms" in timing:
                rerank_ms.append(float(timing.get("rerank_ms", 0.0)))
            if "nli_ms" in timing:
                nli_ms.append(float(timing.get("nli_ms", 0.0)))

        stage1_dist = routing_distribution([1.0 if d == "ACCEPT" else (0.0 if d == "REJECT" else 0.5) for d in stage1_decisions], 0.25, 0.75)
        y_arr = y_vals if y_vals else None
        metrics_a = compute_rates(final_a, y_arr)
        metrics_b = compute_rates(final_b, y_arr)
        trigger_rate = float(np.mean(trigger_b)) if trigger_b else 0.0
        ci = bootstrap_ci(final_a, final_b, y_arr, args.bootstrap, args.seed)

        lines.append(f"- runs_jsonl: `{runs_path}`")
        lines.append("")
        lines.append("## Stage1 distribution")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| accept_rate | {stage1_dist['ACCEPT']:.4f} |")
        lines.append(f"| reject_rate | {stage1_dist['REJECT']:.4f} |")
        lines.append(f"| uncertain_rate | {stage1_dist['UNCERTAIN']:.4f} |")

        lines.append("")
        lines.append("## Policy A: pure stage1")
        lines.append("| metric | value | 95% CI |")
        lines.append("|---|---:|---:|")
        lines.append(f"| final_accept_rate | {metrics_a['accept_rate']:.4f} | |")
        lines.append(f"| final_reject_rate | {metrics_a['reject_rate']:.4f} | |")
        lines.append(f"| final_uncertain_rate | {metrics_a['uncertain_rate']:.4f} | [{ci['policy_a']['uncertain_rate'][0]:.4f}, {ci['policy_a']['uncertain_rate'][1]:.4f}] |")
        if metrics_a["fp_accept_rate"] is not None:
            lines.append(f"| final_fp_accept_rate | {metrics_a['fp_accept_rate']:.4f} | [{ci['policy_a']['fp_accept_rate'][0]:.4f}, {ci['policy_a']['fp_accept_rate'][1]:.4f}] |")
            lines.append(f"| final_fn_reject_rate | {metrics_a['fn_reject_rate']:.4f} | [{ci['policy_a']['fn_reject_rate'][0]:.4f}, {ci['policy_a']['fn_reject_rate'][1]:.4f}] |")
            lines.append(f"| final_ok_rate | {metrics_a['ok_rate']:.4f} | [{ci['policy_a']['ok_rate'][0]:.4f}, {ci['policy_a']['ok_rate'][1]:.4f}] |")
        else:
            lines.append("| final_fp_accept_rate | missing | |")
            lines.append("| final_fn_reject_rate | missing | |")
            lines.append("| final_ok_rate | missing | |")

        lines.append("")
        lines.append("## Policy B: guarded router")
        lines.append("| metric | value | 95% CI |")
        lines.append("|---|---:|---:|")
        lines.append(f"| final_accept_rate | {metrics_b['accept_rate']:.4f} | |")
        lines.append(f"| final_reject_rate | {metrics_b['reject_rate']:.4f} | |")
        lines.append(f"| final_uncertain_rate | {metrics_b['uncertain_rate']:.4f} | [{ci['policy_b']['uncertain_rate'][0]:.4f}, {ci['policy_b']['uncertain_rate'][1]:.4f}] |")
        lines.append(f"| trigger_rate | {trigger_rate:.4f} | |")
        if metrics_b["fp_accept_rate"] is not None:
            lines.append(f"| final_fp_accept_rate | {metrics_b['fp_accept_rate']:.4f} | [{ci['policy_b']['fp_accept_rate'][0]:.4f}, {ci['policy_b']['fp_accept_rate'][1]:.4f}] |")
            lines.append(f"| final_fn_reject_rate | {metrics_b['fn_reject_rate']:.4f} | [{ci['policy_b']['fn_reject_rate'][0]:.4f}, {ci['policy_b']['fn_reject_rate'][1]:.4f}] |")
            lines.append(f"| final_ok_rate | {metrics_b['ok_rate']:.4f} | [{ci['policy_b']['ok_rate'][0]:.4f}, {ci['policy_b']['ok_rate'][1]:.4f}] |")
        else:
            lines.append("| final_fp_accept_rate | missing | |")
            lines.append("| final_fn_reject_rate | missing | |")
            lines.append("| final_ok_rate | missing | |")
        lines.append("")
        lines.append("## Bootstrap delta (B - A)")
        lines.append("| metric | 95% CI |")
        lines.append("|---|---:|")
        lines.append(f"| ok_rate | [{ci['delta']['ok_rate'][0]:.4f}, {ci['delta']['ok_rate'][1]:.4f}] |")
        lines.append(f"| fp_accept_rate | [{ci['delta']['fp_accept_rate'][0]:.4f}, {ci['delta']['fp_accept_rate'][1]:.4f}] |")
        lines.append(f"| fn_reject_rate | [{ci['delta']['fn_reject_rate'][0]:.4f}, {ci['delta']['fn_reject_rate'][1]:.4f}] |")
        lines.append(f"| uncertain_rate | [{ci['delta']['uncertain_rate'][0]:.4f}, {ci['delta']['uncertain_rate'][1]:.4f}] |")
        lines.append("")
        lines.append("## Stage2 and latency")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| stage2_ran_rate | {float(np.mean(stage2_ran)):.4f} |")
        mean_ms = float(np.mean(total_ms)) if total_ms else None
        p95_ms = percentile(total_ms, 95.0)
        p99_ms = percentile(total_ms, 99.0)
        if mean_ms is not None:
            lines.append(f"| mean_total_ms | {mean_ms:.2f} |")
        else:
            lines.append("| mean_total_ms | missing |")
        if p95_ms is not None:
            lines.append(f"| p95_total_ms | {p95_ms:.2f} |")
        else:
            lines.append("| p95_total_ms | missing |")
        if p99_ms is not None:
            lines.append(f"| p99_total_ms | {p99_ms:.2f} |")
        else:
            lines.append("| p99_total_ms | missing |")
        if rerank_ms:
            lines.append(f"| rerank_ms_gt0_rate | {float(np.mean(np.asarray(rerank_ms) > 0.0)):.4f} |")
        if nli_ms:
            lines.append(f"| nli_ms_gt0_rate | {float(np.mean(np.asarray(nli_ms) > 0.0)):.4f} |")
        lines.append("")
        lines.append("## Over-engineering check")
        ok_delta = (metrics_b["ok_rate"] - metrics_a["ok_rate"]) if metrics_a["ok_rate"] is not None else None
        trig_delta = trigger_rate - metrics_a["uncertain_rate"]
        if ok_delta is None:
            lines.append("insufficient labels to judge router effectiveness")
        else:
            if ok_delta >= 0.005 or trig_delta <= -0.05:
                lines.append("router shows material improvement or trigger reduction vs baseline")
            else:
                lines.append("router does not materially improve ok_rate or trigger_rate vs baseline")
        if args.track == "fever":
            lines.append("## FEVER verdict")
            if metrics_a["accept_rate"] <= 0.05 or metrics_a["uncertain_rate"] >= 0.9:
                lines.append("negative result: accept_rate remains near zero or uncertain is very high; stage1-only is weak")
            else:
                lines.append("coverage is not extremely low at baseline thresholds")
        lines.append("")
        lines.append("## Repro command")
        lines.append("```")
        lines.append(" ".join(sys.argv))
        lines.append("```")
    else:
        if args.in_path is None:
            scifact_path, fever_path = parse_demo_paths()
            args.in_path = scifact_path if args.track == "scifact" else fever_path
        if args.in_path is None:
            raise SystemExit("in_path is required when runs_jsonl is missing")
        if not args.logit_col:
            raise SystemExit("logit_col is required for proxy mode")
        cols = [args.logit_col, args.y_col]
        df = pd.read_parquet(args.in_path, columns=cols)
        if args.logit_col not in df.columns or args.y_col not in df.columns:
            raise SystemExit("missing required columns in parquet")
        if args.n is not None and args.n < len(df):
            df = df.sample(n=args.n, random_state=args.seed)
        y = df[args.y_col].astype(int).to_numpy()
        logits = df[args.logit_col].astype(float).to_numpy()
        cs_ret = compute_cs_ret_array(logits, tau)
        decisions = [decide_route(float(v), t_lower, t_upper)[0] for v in cs_ret]
        stage1_dist = routing_distribution(cs_ret, t_lower, t_upper)
        trigger = [guard_trigger(v, d, t_lower, t_upper, guard_band) for v, d in zip(cs_ret, decisions)]
        metrics_a = compute_rates(decisions, y)
        final_b = ["UNCERTAIN" if t else d for t, d in zip(trigger, decisions)]
        metrics_b = compute_rates(final_b, y)
        trigger_rate = float(np.mean(trigger)) if trigger else 0.0
        lines.append(f"- proxy_mode: true")
        lines.append(f"- in_path: `{args.in_path}`")
        lines.append("")
        lines.append("## Stage1 distribution")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| accept_rate | {stage1_dist['ACCEPT']:.4f} |")
        lines.append(f"| reject_rate | {stage1_dist['REJECT']:.4f} |")
        lines.append(f"| uncertain_rate | {stage1_dist['UNCERTAIN']:.4f} |")
        lines.append(f"| fp_accept_rate | {metrics_a['fp_accept_rate']:.4f} |")
        lines.append(f"| fn_reject_rate | {metrics_a['fn_reject_rate']:.4f} |")
        lines.append(f"| ok_rate | {metrics_a['ok_rate']:.4f} |")
        lines.append("")
        lines.append("## Policy A: pure stage1")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| final_accept_rate | {metrics_a['accept_rate']:.4f} |")
        lines.append(f"| final_reject_rate | {metrics_a['reject_rate']:.4f} |")
        lines.append(f"| final_uncertain_rate | {metrics_a['uncertain_rate']:.4f} |")
        lines.append("")
        lines.append("## Policy B: guarded router (proxy)")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| final_accept_rate | {metrics_b['accept_rate']:.4f} |")
        lines.append(f"| final_reject_rate | {metrics_b['reject_rate']:.4f} |")
        lines.append(f"| final_uncertain_rate | {metrics_b['uncertain_rate']:.4f} |")
        lines.append(f"| trigger_rate | {trigger_rate:.4f} |")
        lines.append("")
        lines.append("## Recommendation")
        lines.append("proxy results only; use runs_jsonl for end-to-end evaluation")
        lines.append("")
        lines.append("## Repro command")
        lines.append("```")
        lines.append(" ".join(sys.argv))
        lines.append("```")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
