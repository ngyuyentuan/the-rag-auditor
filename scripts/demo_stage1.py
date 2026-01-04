import argparse
import os
import sys
from datetime import datetime, timezone

P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [P]

import numpy as np
import pandas as pd
import yaml

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution


def load_thresholds(path, track):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    t = (data.get("thresholds") or {}).get(track)
    if not t:
        raise SystemExit(f"missing thresholds for track={track}")
    for key in ["tau", "t_lower", "t_upper"]:
        if key not in t:
            raise SystemExit(f"missing thresholds.{track}.{key}")
    return t


def pick_defaults(track, in_path, logit_col):
    if track == "scifact":
        in_path = in_path or "data/calibration/scifact_stage1_dev_train.parquet"
        logit_col = logit_col or "raw_max_top3"
    else:
        in_path = in_path or "data/calibration/fever_stage1_dev_train.parquet"
        logit_col = logit_col or "logit_platt"
    return in_path, logit_col


def find_parquet(root_dir, keyword):
    keyword = keyword.lower()
    matches = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if not name.lower().endswith(".parquet"):
                continue
            if keyword not in name.lower():
                continue
            path = os.path.join(root, name)
            matches.append(path)
    if not matches:
        return None
    def score(path):
        lname = os.path.basename(path).lower()
        return (
            2 if "dev_train" in lname else 0,
            1 if "train" in lname else 0,
            1 if keyword in lname else 0,
            -len(path),
        )
    matches.sort(key=score, reverse=True)
    return matches[0]


def resolve_in_path(track, args):
    if track == "scifact":
        if args.scifact_in_path:
            return args.scifact_in_path
        if args.track == "scifact" and args.in_path:
            return args.in_path
    else:
        if args.fever_in_path:
            return args.fever_in_path
        if args.track == "fever" and args.in_path:
            return args.in_path
    default_path, _ = pick_defaults(track, None, None)
    if os.path.exists(default_path):
        return default_path
    search_root = "/mnt/c/Users/nguye/Downloads"
    found = find_parquet(search_root, track)
    if found:
        return found
    raise SystemExit(f"missing in_path for track={track}")


def sanitize(df, logit_col, y_col):
    cols = [logit_col, y_col]
    if "qid" in df.columns:
        cols.append("qid")
    d = df[cols].copy()
    d[logit_col] = pd.to_numeric(d[logit_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[logit_col, y_col])
    d = d[np.isfinite(d[logit_col].to_numpy(np.float64))]
    d = d[(d[y_col] == 0) | (d[y_col] == 1)]
    d[y_col] = d[y_col].astype(int)
    d = d.reset_index(drop=True)
    d["row_idx"] = np.arange(len(d), dtype=np.int64)
    return d


def format_float(v, digits=4):
    return f"{v:.{digits}f}"


def to_md_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_track(track, args):
    in_path = resolve_in_path(track, args)
    _, default_logit = pick_defaults(track, None, None)
    logit_col = args.logit_col or default_logit
    y_col = args.y_col

    df = pd.read_parquet(in_path)
    for col in [logit_col, y_col]:
        if col not in df.columns:
            raise SystemExit(f"missing column: {col}")
    df = sanitize(df, logit_col, y_col)
    if df.empty:
        raise SystemExit("no valid rows after sanitize")

    n = int(args.n)
    if n <= 0:
        raise SystemExit("n must be > 0")
    if len(df) > n:
        df = df.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)

    t = load_thresholds(args.thresholds, track)
    tau = float(t.get("tau"))
    t_lower = float(t.get("t_lower"))
    t_upper = float(t.get("t_upper"))

    records = []
    cs_list = []
    decisions = []

    for _, row in df.iterrows():
        qid = row["qid"] if "qid" in row else int(row["row_idx"])
        logit = float(row[logit_col])
        y = int(row[y_col])
        cs_ret = compute_cs_ret_from_logit(logit, tau)
        decision, reason = decide_route(cs_ret, t_lower, t_upper)
        cs_list.append(cs_ret)
        decisions.append(decision)
        records.append({
            "qid": qid,
            "y": y,
            "logit": logit,
            "cs_ret": cs_ret,
            "decision": decision,
            "reason": reason,
        })

    n_total = len(records)
    if n_total == 0:
        raise SystemExit("no records to report")

    dist = routing_distribution(cs_list, t_lower, t_upper)
    accept_rate = decisions.count("ACCEPT") / n_total
    reject_rate = decisions.count("REJECT") / n_total
    uncertain_rate = decisions.count("UNCERTAIN") / n_total

    return {
        "track": track,
        "in_path": in_path,
        "logit_col": logit_col,
        "y_col": y_col,
        "tau": tau,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "n": n_total,
        "seed": int(args.seed),
        "dist": dist,
        "accept_rate": accept_rate,
        "reject_rate": reject_rate,
        "uncertain_rate": uncertain_rate,
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default=None, choices=["scifact", "fever"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--in_path", default=None)
    ap.add_argument("--scifact_in_path", default=None)
    ap.add_argument("--fever_in_path", default=None)
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--logit_col", default=None)
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--out_md", default="reports/demo_stage1.md")
    args = ap.parse_args()

    results = []
    for track in ["scifact", "fever"]:
        results.append(run_track(track, args))

    lines = []
    lines.append("# Demo Stage1 Report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    for res in results:
        lines.append(f"## Track: {res['track']}")
        lines.append("")
        lines.append(f"- in_path: `{res['in_path']}`")
        lines.append(f"- n: `{res['n']}`")
        lines.append(f"- seed: `{res['seed']}`")
        lines.append(f"- logit_col: `{res['logit_col']}`")
        lines.append(f"- y_col: `{res['y_col']}`")
        lines.append(f"- tau: `{res['tau']}`")
        lines.append(f"- t_lower: `{res['t_lower']}`")
        lines.append(f"- t_upper: `{res['t_upper']}`")
        lines.append("")
        lines.append("Routing distribution:")
        lines.append("")
        dist_rows = [
            ["accept_rate", format_float(res["accept_rate"])],
            ["reject_rate", format_float(res["reject_rate"])],
            ["uncertain_rate", format_float(res["uncertain_rate"])],
        ]
        lines.append(to_md_table(["metric", "value"], dist_rows))
        lines.append("")
        lines.append("First 5 examples:")
        lines.append("")
        headers = ["qid", "y", "logit", "cs_ret", "decision", "reason"]
        rows = []
        for r in res["records"][:5]:
            rows.append([
                str(r["qid"]),
                str(r["y"]),
                format_float(r["logit"]),
                format_float(r["cs_ret"]),
                r["decision"],
                r["reason"],
            ])
        lines.append(to_md_table(headers, rows))
        lines.append("")

    out_path = args.out_md
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("wrote:", out_path)


if __name__ == "__main__":
    main()
