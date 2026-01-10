import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, stable_sigmoid


def compute_metrics_fast(cs_sorted, pos_prefix, neg_prefix, t_lower, t_upper):
    n = len(cs_sorted)
    idx_low = int(np.searchsorted(cs_sorted, t_lower, side="left"))
    idx_upper = int(np.searchsorted(cs_sorted, t_upper, side="left"))
    reject_count = idx_low
    accept_count = n - idx_upper
    uncertain_count = idx_upper - idx_low
    fp_accept = neg_prefix[n] - neg_prefix[idx_upper]
    fn_reject = pos_prefix[idx_low]
    accept_rate = accept_count / n if n else 0.0
    reject_rate = reject_count / n if n else 0.0
    uncertain_rate = uncertain_count / n if n else 0.0
    fp_accept_rate = fp_accept / n if n else 0.0
    fn_reject_rate = fn_reject / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    coverage = 1.0 - uncertain_rate
    decided = accept_count + reject_count
    accuracy_on_decided = (accept_count + reject_count - fp_accept - fn_reject) / decided if decided else 0.0
    return {
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": accuracy_on_decided,
    }


def prepare_sorted(cs, y):
    order = np.argsort(cs)
    cs_sorted = cs[order]
    y_sorted = y[order]
    pos_prefix = np.zeros(len(y_sorted) + 1, dtype=int)
    neg_prefix = np.zeros(len(y_sorted) + 1, dtype=int)
    pos_prefix[1:] = np.cumsum(y_sorted == 1)
    neg_prefix[1:] = np.cumsum(y_sorted == 0)
    return cs_sorted, pos_prefix, neg_prefix


def compute_metrics_slow(cs_ret, y, t_lower, t_upper):
    fp = 0
    fn = 0
    accept = 0
    reject = 0
    uncertain = 0
    for p, yi in zip(cs_ret, y):
        decision, _ = decide_route(p, t_lower, t_upper)
        if decision == "ACCEPT":
            accept += 1
            if yi == 0:
                fp += 1
        elif decision == "REJECT":
            reject += 1
            if yi == 1:
                fn += 1
        else:
            uncertain += 1
    n = len(y)
    accept_rate = accept / n if n else 0.0
    reject_rate = reject / n if n else 0.0
    uncertain_rate = uncertain / n if n else 0.0
    fp_accept_rate = fp / n if n else 0.0
    fn_reject_rate = fn / n if n else 0.0
    ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
    coverage = 1.0 - uncertain_rate
    decided = accept + reject
    accuracy_on_decided = (accept + reject - fp - fn) / decided if decided else 0.0
    return {
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "fp_accept_rate": fp_accept_rate,
        "fn_reject_rate": fn_reject_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": accuracy_on_decided,
    }


def self_check(cs_ret, y_vals, lower_grid, upper_grid):
    n = min(200, len(y_vals))
    cs_slice = np.asarray(cs_ret[:n], dtype=float)
    y_slice = np.asarray(y_vals[:n], dtype=int)
    cs_sorted, pos_prefix, neg_prefix = prepare_sorted(cs_slice, y_slice)
    for t_lower in lower_grid[:3]:
        for t_upper in upper_grid[:3]:
            if t_lower >= t_upper:
                continue
            slow = compute_metrics_slow(cs_slice, y_slice, float(t_lower), float(t_upper))
            fast = compute_metrics_fast(cs_sorted, pos_prefix, neg_prefix, float(t_lower), float(t_upper))
            for k in slow:
                if abs(slow[k] - fast[k]) > 1e-9:
                    raise ValueError("fast_metrics_mismatch")


def near_feasible_score(r, budget, max_fp, max_fn):
    score = max(0.0, r["uncertain_rate"] - budget)
    score += max(0.0, r["fp_accept_rate"] - max_fp)
    score += max(0.0, r["fn_reject_rate"] - max_fn)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--tau_grid_min", type=float, default=0.2)
    ap.add_argument("--tau_grid_max", type=float, default=2.0)
    ap.add_argument("--tau_grid_steps", type=int, default=60)
    ap.add_argument("--grid_lower_steps", type=int, default=50)
    ap.add_argument("--grid_upper_steps", type=int, default=50)
    ap.add_argument("--budgets", type=float, nargs="+", required=True)
    ap.add_argument("--max_fp_accept_rate", type=float, default=0.05)
    ap.add_argument("--max_fn_reject_rate", type=float, default=0.05)
    ap.add_argument("--objective", choices=["feasible_best_ok", "weighted"], default="feasible_best_ok")
    ap.add_argument("--lambda_uncertain", type=float, default=0.2)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_yaml", default=None)
    ap.add_argument("--progress", dest="progress", action="store_true")
    ap.add_argument("--no_progress", dest="progress", action="store_false")
    ap.set_defaults(progress=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    df = pd.read_parquet(in_path)
    n_raw = len(df)
    if args.logit_col not in df.columns:
        raise ValueError(f"missing logit_col {args.logit_col}")
    if args.y_col not in df.columns:
        raise ValueError(f"missing y_col {args.y_col}")

    logits = pd.to_numeric(df[args.logit_col], errors="coerce")
    y = pd.to_numeric(df[args.y_col], errors="coerce")
    mask = y.isin([0, 1]) & logits.notna()
    df = df.loc[mask].copy()
    df[args.y_col] = y[mask].astype(int)
    df[args.logit_col] = logits[mask].astype(float)

    if args.n is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    n_used = len(df)
    logits = df[args.logit_col].astype(float).tolist()
    y_vals = df[args.y_col].astype(int).tolist()

    tau_grid = np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)
    lower_grid = np.linspace(0.0, 0.99, args.grid_lower_steps)
    upper_grid = np.linspace(0.01, 1.0, args.grid_upper_steps)

    results = []
    logits_arr = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=int)
    lower_grid = np.asarray(lower_grid, dtype=float)
    upper_grid = np.asarray(upper_grid, dtype=float)
    valid_pairs = [
        (i, j)
        for i in range(len(lower_grid))
        for j in range(len(upper_grid))
        if lower_grid[i] < upper_grid[j]
    ]
    total_iters = len(tau_grid) * len(valid_pairs)
    progress_enabled = args.progress if args.progress is not None else sys.stderr.isatty()
    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None
    progress_step = max(1, total_iters // 100) if total_iters else 1
    progress_count = 0
    start_time = time.time()
    pbar = _tqdm(total=total_iters, desc="grid", file=sys.stderr, disable=not progress_enabled) if _tqdm else None

    for tau in tau_grid:
        tau_val = float(tau)
        cs = stable_sigmoid(logits_arr / tau_val)
        cs = np.clip(cs, 0.0, 1.0)
        self_check(cs, y_arr, lower_grid, upper_grid)
        cs_sorted, pos_prefix, neg_prefix = prepare_sorted(cs, y_arr)
        idx_low = np.searchsorted(cs_sorted, lower_grid, side="left")
        idx_upper = np.searchsorted(cs_sorted, upper_grid, side="left")
        n = len(cs_sorted)
        total_pos = int(pos_prefix[n])
        total_neg = int(neg_prefix[n])
        for i, j in valid_pairs:
            t_lower = lower_grid[i]
            t_upper = upper_grid[j]
            low_idx = int(idx_low[i])
            up_idx = int(idx_upper[j])
            reject_count = low_idx
            accept_count = n - up_idx
            uncertain_count = up_idx - low_idx
            fp_accept = total_neg - int(neg_prefix[up_idx])
            fn_reject = int(pos_prefix[low_idx])
            accept_rate = accept_count / n if n else 0.0
            reject_rate = reject_count / n if n else 0.0
            uncertain_rate = uncertain_count / n if n else 0.0
            fp_accept_rate = fp_accept / n if n else 0.0
            fn_reject_rate = fn_reject / n if n else 0.0
            ok_rate = 1.0 - fp_accept_rate - fn_reject_rate
            coverage = 1.0 - uncertain_rate
            decided = accept_count + reject_count
            accuracy_on_decided = (accept_count + reject_count - fp_accept - fn_reject) / decided if decided else 0.0
            results.append({
                "accept_rate": accept_rate,
                "reject_rate": reject_rate,
                "uncertain_rate": uncertain_rate,
                "fp_accept_rate": fp_accept_rate,
                "fn_reject_rate": fn_reject_rate,
                "ok_rate": ok_rate,
                "coverage": coverage,
                "accuracy_on_decided": accuracy_on_decided,
                "tau": float(tau),
                "t_lower": float(t_lower),
                "t_upper": float(t_upper),
            })
            progress_count += 1
            if pbar:
                pbar.update(1)
            elif progress_enabled and (progress_count % progress_step == 0 or progress_count == total_iters):
                pct = (progress_count / total_iters) * 100 if total_iters else 100.0
                elapsed = time.time() - start_time
                print(f"progress {progress_count}/{total_iters} ({pct:.1f}%) elapsed {elapsed:.1f}s", file=sys.stderr)
    if pbar:
        pbar.close()
    if progress_enabled:
        elapsed = time.time() - start_time
        print(f"done {progress_count} iterations in {elapsed:.1f}s", file=sys.stderr)

    budgets = sorted(args.budgets)
    best_per_budget = {}
    near_per_budget = {}
    feasible_counts = {}

    for b in budgets:
        feasible = [
            r for r in results
            if r["uncertain_rate"] <= b
            and r["fp_accept_rate"] <= args.max_fp_accept_rate
            and r["fn_reject_rate"] <= args.max_fn_reject_rate
        ]
        feasible_counts[b] = len(feasible)
        if feasible:
            if args.objective == "weighted":
                for r in feasible:
                    r["score"] = r["ok_rate"] - args.lambda_uncertain * r["uncertain_rate"]
                feasible.sort(key=lambda r: (-r["score"], -r["coverage"], -r["accuracy_on_decided"], -r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"]))
            else:
                feasible.sort(key=lambda r: (-r["coverage"], -r["accuracy_on_decided"], -r["ok_rate"], r["fp_accept_rate"] + r["fn_reject_rate"]))
            best_per_budget[b] = feasible[0]
        else:
            best_per_budget[b] = None
        near = min(results, key=lambda r: (near_feasible_score(r, b, args.max_fp_accept_rate, args.max_fn_reject_rate), -r["coverage"], -r["accuracy_on_decided"], -r["ok_rate"]))
        near_per_budget[b] = near

    export_budget = None
    for b in budgets:
        if best_per_budget[b] is not None:
            export_budget = b
            break

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_joint_tuning_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage1 Joint Tuning ({args.track})")
    lines.append("")
    lines.append(f"Generated: `{dt.datetime.now(dt.UTC).isoformat()}`")
    lines.append("")
    lines.append("Config")
    lines.append("")
    lines.append(f"- in_path: `{in_path}`")
    lines.append(f"- n_raw: `{n_raw}`")
    lines.append(f"- n_used: `{n_used}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- tau_grid_min: `{args.tau_grid_min}`")
    lines.append(f"- tau_grid_max: `{args.tau_grid_max}`")
    lines.append(f"- tau_grid_steps: `{args.tau_grid_steps}`")
    lines.append(f"- grid_lower_steps: `{args.grid_lower_steps}`")
    lines.append(f"- grid_upper_steps: `{args.grid_upper_steps}`")
    lines.append(f"- max_fp_accept_rate: `{args.max_fp_accept_rate}`")
    lines.append(f"- max_fn_reject_rate: `{args.max_fn_reject_rate}`")
    lines.append(f"- objective: `{args.objective}`")
    lines.append("")

    for b in budgets:
        lines.append(f"Budget <= {b}")
        lines.append("")
        lines.append(f"- feasible_configs: `{feasible_counts[b]}`")
        best = best_per_budget[b]
        if best is None:
            lines.append("- recommended: none (no feasible config)")
        else:
            lines.append(f"- recommended: tau={best['tau']} t_lower={best['t_lower']} t_upper={best['t_upper']}")
            lines.append(f"  coverage={best['coverage']:.4f} accuracy_on_decided={best['accuracy_on_decided']:.4f} ok_rate={best['ok_rate']:.4f} fp={best['fp_accept_rate']:.4f} fn={best['fn_reject_rate']:.4f} uncertain={best['uncertain_rate']:.4f}")
        near = near_per_budget[b]
        lines.append(f"- best_near_feasible: tau={near['tau']} t_lower={near['t_lower']} t_upper={near['t_upper']} fp={near['fp_accept_rate']:.4f} fn={near['fn_reject_rate']:.4f} uncertain={near['uncertain_rate']:.4f}")
        lines.append("")

    out_yaml = Path(args.out_yaml) if args.out_yaml else Path(f"configs/thresholds_stage1_joint_tuned_{args.track}.yaml")
    if export_budget is None:
        lines.append("YAML output")
        lines.append("")
        lines.append("- written_to_yaml: none (no feasible config for any budget)")
        lines.append("")
    else:
        best = best_per_budget[export_budget]
        payload = {
            "tau": float(best["tau"]),
            "t_lower": float(best["t_lower"]),
            "t_upper": float(best["t_upper"]),
            "export_budget": float(export_budget),
            "max_fp_accept_rate": float(args.max_fp_accept_rate),
            "max_fn_reject_rate": float(args.max_fn_reject_rate),
        }
        out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        lines.append("YAML output")
        lines.append("")
        lines.append(f"- written_to_yaml: tau={payload['tau']} t_lower={payload['t_lower']} t_upper={payload['t_upper']} export_budget={payload['export_budget']}")
        lines.append("")

    lines.append("Repro command")
    lines.append("")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
