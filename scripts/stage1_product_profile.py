import argparse
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


def compute_rates(cs_ret, y, t_lower, t_upper):
    decisions = [decide_route(float(v), t_lower, t_upper)[0] for v in cs_ret]
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    y_arr = np.asarray(y, dtype=int)
    dec_arr = np.asarray(decisions, dtype=object)
    fp = float(np.mean((dec_arr == "ACCEPT") & (y_arr == 0))) if len(y_arr) else 0.0
    fn = float(np.mean((dec_arr == "REJECT") & (y_arr == 1))) if len(y_arr) else 0.0
    ok = 1.0 - fp - fn
    return {
        "decisions": decisions,
        "dist": dist,
        "fp": fp,
        "fn": fn,
        "ok": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["scifact", "fever"])
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--logit_col", required=True)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--n", type=int)
    ap.add_argument("--grid_lower_steps", type=int, default=50)
    ap.add_argument("--grid_upper_steps", type=int, default=50)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    out_md = Path(args.out_md) if args.out_md else Path(f"reports/stage1_product_profile_{args.track}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    t_lower_base, t_upper_base, tau_base = load_thresholds(Path("configs/thresholds.yaml"), args.track)
    tau = float(args.tau) if args.tau is not None else tau_base

    df = pd.read_parquet(args.in_path, columns=[args.logit_col, args.y_col])
    df = df.dropna(subset=[args.logit_col, args.y_col]).copy()
    if args.n is not None and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed)
    y = df[args.y_col].astype(int).to_numpy()
    logits = df[args.logit_col].astype(float).to_numpy()
    cs_ret = np.asarray([compute_cs_ret_from_logit(float(v), tau) for v in logits], dtype=float)

    grid_lower = np.linspace(0.0, 0.99, args.grid_lower_steps)
    grid_upper = np.linspace(0.01, 1.0, args.grid_upper_steps)
    candidates = []
    for tl in grid_lower:
        for tu in grid_upper:
            if tl >= tu:
                continue
            r = compute_rates(cs_ret, y, float(tl), float(tu))
            candidates.append({
                "t_lower": float(tl),
                "t_upper": float(tu),
                "accept_rate": r["dist"]["ACCEPT"],
                "reject_rate": r["dist"]["REJECT"],
                "uncertain_rate": r["dist"]["UNCERTAIN"],
                "fp_accept_rate": r["fp"],
                "fn_reject_rate": r["fn"],
                "ok_rate": r["ok"],
            })

    def pick_safety():
        feasible = [c for c in candidates if c["uncertain_rate"] <= 0.95]
        feasible.sort(key=lambda c: (c["fp_accept_rate"] + c["fn_reject_rate"], c["uncertain_rate"], -c["ok_rate"]))
        return feasible[0] if feasible else None

    def pick_coverage():
        feasible = [c for c in candidates if c["fp_accept_rate"] <= 0.05 and c["fn_reject_rate"] <= 0.05]
        relaxed = False
        if not feasible:
            feasible = [c for c in candidates if c["fp_accept_rate"] <= 0.10 and c["fn_reject_rate"] <= 0.10]
            relaxed = True
        feasible.sort(key=lambda c: (c["uncertain_rate"], c["fp_accept_rate"] + c["fn_reject_rate"], -c["ok_rate"]))
        return (feasible[0] if feasible else None), relaxed

    safety = pick_safety()
    coverage, relaxed = pick_coverage()

    lines = []
    lines.append(f"# Stage1 Product Profiles ({args.track})")
    lines.append("")
    lines.append(f"- in_path: `{args.in_path}`")
    lines.append(f"- n: {len(df)}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- logit_col: `{args.logit_col}`")
    lines.append(f"- y_col: `{args.y_col}`")
    lines.append(f"- tau: {tau}")
    lines.append("")
    lines.append("## Profile: safety_first")
    if safety is None:
        lines.append("no feasible config under uncertain<=0.95")
    else:
        lines.append(f"- t_lower: {safety['t_lower']}")
        lines.append(f"- t_upper: {safety['t_upper']}")
        lines.append(f"- accept_rate: {safety['accept_rate']:.4f}")
        lines.append(f"- reject_rate: {safety['reject_rate']:.4f}")
        lines.append(f"- uncertain_rate: {safety['uncertain_rate']:.4f}")
        lines.append(f"- fp_accept_rate: {safety['fp_accept_rate']:.4f}")
        lines.append(f"- fn_reject_rate: {safety['fn_reject_rate']:.4f}")
        lines.append(f"- ok_rate: {safety['ok_rate']:.4f}")
        lines.append("Product interpretation: prioritizes minimizing wrong decisions; high defer if needed.")
    lines.append("")
    lines.append("## Profile: coverage_first")
    if coverage is None:
        lines.append("no feasible config under specified error caps")
    else:
        lines.append(f"- t_lower: {coverage['t_lower']}")
        lines.append(f"- t_upper: {coverage['t_upper']}")
        lines.append(f"- accept_rate: {coverage['accept_rate']:.4f}")
        lines.append(f"- reject_rate: {coverage['reject_rate']:.4f}")
        lines.append(f"- uncertain_rate: {coverage['uncertain_rate']:.4f}")
        lines.append(f"- fp_accept_rate: {coverage['fp_accept_rate']:.4f}")
        lines.append(f"- fn_reject_rate: {coverage['fn_reject_rate']:.4f}")
        lines.append(f"- ok_rate: {coverage['ok_rate']:.4f}")
        if relaxed:
            lines.append("Product interpretation: coverage-first; relaxed error caps to 0.10; defer less but accept higher risk.")
        else:
            lines.append("Product interpretation: coverage-first under 0.05 fp/fn caps.")
    lines.append("")
    lines.append("## Repro command")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
