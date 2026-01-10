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


def wilson_ci(x, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    phat = x / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def parse_demo_paths():
    demo = Path("reports/demo_stage1.md")
    if not demo.exists():
        return None, None
    lines = demo.read_text(encoding="utf-8").splitlines()
    paths = []
    for line in lines:
        if line.strip().startswith("- in_path:"):
            segs = line.split("`")
            if len(segs) >= 2:
                paths.append(segs[1])
            else:
                paths.append(line.split(":", 1)[1].strip())
        if len(paths) >= 2:
            break
    sci = paths[0] if paths else None
    fever = paths[1] if len(paths) > 1 else None
    return sci, fever


def load_yaml(path: Path, track: str):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    if "thresholds" in data:
        t = data["thresholds"].get(track)
    else:
        t = data
    if not isinstance(t, dict):
        return None
    if not all(k in t for k in ("tau", "t_lower", "t_upper")):
        return None
    return t


def evaluate(df, logit_col, y_col, config):
    tau = float(config["tau"])
    t_lower = float(config["t_lower"])
    t_upper = float(config["t_upper"])
    logits = pd.to_numeric(df[logit_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = logits.notna() & y.isin([0, 1])
    logits = logits[mask].to_numpy(dtype=float)
    y_arr = y[mask].to_numpy(dtype=int)
    cs_ret = np.array([compute_cs_ret_from_logit(v, tau) for v in logits], dtype=float)
    if "c_accept" in config and "c_reject" in config:
        accept = (logits >= float(config["t_upper"])) & (cs_ret >= float(config["c_accept"]))
        reject = (logits <= float(config["t_lower"])) | (cs_ret <= float(config["c_reject"]))
        uncertain = ~(accept | reject)
    else:
        decisions = []
        for p in cs_ret:
            d, _ = decide_route(p, t_lower, t_upper)
            decisions.append(d)
        d_arr = np.asarray(decisions)
        accept = d_arr == "ACCEPT"
        reject = d_arr == "REJECT"
        uncertain = d_arr == "UNCERTAIN"
    n = len(y_arr)
    fp = int(np.sum(accept & (y_arr == 0)))
    fn = int(np.sum(reject & (y_arr == 1)))
    tp = int(np.sum(accept & (y_arr == 1)))
    tn = int(np.sum(reject & (y_arr == 0)))
    dist = routing_distribution(cs_ret, t_lower, t_upper)
    fp_rate = fp / n if n else 0.0
    fn_rate = fn / n if n else 0.0
    coverage = 1.0 - dist["UNCERTAIN"]
    ok_rate = 1.0 - fp_rate - fn_rate
    decided = tp + tn + fp + fn
    acc_decided = (tp + tn) / decided if decided else 0.0
    fp_ci = wilson_ci(fp, n)
    fn_ci = wilson_ci(fn, n)
    fp_decided = fp / decided if decided else 0.0
    fn_decided = fn / decided if decided else 0.0
    fp_dec_ci = wilson_ci(fp, decided)[1] if decided else fp_decided
    fn_dec_ci = wilson_ci(fn, decided)[1] if decided else fn_decided
    ci_safe = fp_ci[1] <= 0.05 and fn_ci[1] <= 0.05
    decided_ci_safe = fp_dec_ci <= config.get("max_fp_decided_upper95", 0.10) and fn_dec_ci <= config.get("max_fn_decided_upper95", 0.10)
    accept_rate = dist["ACCEPT"]
    reject_rate = dist["REJECT"]
    accept_count = int(round(accept_rate * n))
    reject_count = int(round(reject_rate * n))
    fp_given_accept = fp / accept_count if accept_count else 0.0
    fn_given_reject = fn / reject_count if reject_count else 0.0
    fp_accept_ci = wilson_ci(fp, accept_count) if accept_count else (0.0, 0.0)
    fn_reject_ci = wilson_ci(fn, reject_count) if reject_count else (0.0, 0.0)
    return {
        "accept_rate": dist["ACCEPT"],
        "reject_rate": dist["REJECT"],
        "uncertain_rate": dist["UNCERTAIN"],
        "fp_accept_rate": fp_rate,
        "fn_reject_rate": fn_rate,
        "ok_rate": ok_rate,
        "coverage": coverage,
        "accuracy_on_decided": acc_decided,
        "fp_upper95": fp_ci[1],
        "fn_upper95": fn_ci[1],
        "ci_safe": ci_safe,
        "fp_decided_rate": fp_decided,
        "fn_decided_rate": fn_decided,
        "fp_decided_upper95": fp_dec_ci,
        "fn_decided_upper95": fn_dec_ci,
        "decided_ci_safe": decided_ci_safe,
        "n": n,
        "decided_count": decided,
        "accept_count": accept_count,
        "reject_count": reject_count,
        "fp_given_accept": fp_given_accept,
        "fn_given_reject": fn_given_reject,
        "fp_given_accept_upper95": fp_accept_ci[1],
        "fn_given_reject_upper95": fn_reject_ci[1],
    }


def compute_statuses(m, args, certify_mode, config):
    max_fp_dec = float(config.get("max_fp_decided_upper95", args.max_fp_decided_upper95))
    max_fn_dec = float(config.get("max_fn_decided_upper95", args.max_fn_decided_upper95))
    min_accept = int(config.get("min_accept_count", args.min_accept_count))
    min_reject = int(config.get("min_reject_count", args.min_reject_count))
    max_fp_accept = float(config.get("max_fp_given_accept_upper95", args.max_fp_given_accept_upper95))
    max_fn_reject = float(config.get("max_fn_given_reject_upper95", args.max_fn_given_reject_upper95))
    decided_status = "INSUFFICIENT_N" if m["decided_count"] < args.min_decided_count else ("PASS" if (m["fp_decided_upper95"] <= max_fp_dec and m["fn_decided_upper95"] <= max_fn_dec) else "FAIL")
    if certify_mode == "accept_only":
        accept_status = "INSUFFICIENT_N" if m["accept_count"] == 0 or m["accept_count"] < min_accept else ("PASS" if m["fp_given_accept_upper95"] <= max_fp_accept else "FAIL")
        reject_status = "N/A"
        action_status = accept_status
        return "N/A", accept_status, reject_status, action_status, "N/A", "N/A"
    if certify_mode == "reject_only":
        accept_status = "N/A"
        reject_status = "INSUFFICIENT_N" if m["reject_count"] == 0 or m["reject_count"] < min_reject else ("PASS" if m["fn_given_reject_upper95"] <= max_fn_reject else "FAIL")
        action_status = reject_status
        return "N/A", accept_status, reject_status, action_status, "N/A", "N/A"
    accept_status = "INSUFFICIENT_N" if m["accept_count"] == 0 or m["accept_count"] < min_accept else ("PASS" if m["fp_given_accept_upper95"] <= max_fp_accept else "FAIL")
    reject_status = "INSUFFICIENT_N" if m["reject_count"] == 0 or m["reject_count"] < min_reject else ("PASS" if m["fn_given_reject_upper95"] <= max_fn_reject else "FAIL")
    if accept_status == "PASS" and reject_status == "PASS":
        action_status = "PASS"
    elif "INSUFFICIENT_N" in (accept_status, reject_status):
        action_status = "INSUFFICIENT_N"
    else:
        action_status = "FAIL"
    return decided_status, accept_status, reject_status, action_status, m["ci_safe"], m["decided_ci_safe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever", "all"], default="all")
    ap.add_argument("--scifact_in_path")
    ap.add_argument("--fever_in_path")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out_md", default="reports/stage1_product_table.md")
    ap.add_argument("--min_decided_count", type=int, default=0)
    ap.add_argument("--max_fp_decided_upper95", type=float, default=0.10)
    ap.add_argument("--max_fn_decided_upper95", type=float, default=0.10)
    ap.add_argument("--min_accept_count", type=int, default=0)
    ap.add_argument("--min_reject_count", type=int, default=0)
    ap.add_argument("--max_fp_given_accept_upper95", type=float, default=0.10)
    ap.add_argument("--max_fn_given_reject_upper95", type=float, default=0.10)
    args = ap.parse_args()

    sci_path = args.scifact_in_path
    fever_path = args.fever_in_path
    if not sci_path or not fever_path:
        sci_auto, fever_auto = parse_demo_paths()
        sci_path = sci_path or sci_auto
        fever_path = fever_path or fever_auto

    configs = {
        "baseline": Path("configs/thresholds.yaml"),
        "joint_tuned": Path("configs/thresholds_stage1_joint_tuned_{track}.yaml"),
        "product_tuned": Path("configs/thresholds_stage1_product_{track}.yaml"),
        "product_safety": Path("configs/thresholds_stage1_product_safety_{track}.yaml"),
        "product_coverage": Path("configs/thresholds_stage1_product_coverage_{track}.yaml"),
        "product_2d": Path("configs/thresholds_stage1_product_2d_{track}.yaml"),
        "action_accept": Path("configs/thresholds_stage1_action_accept_{track}.yaml"),
        "action_reject": Path("configs/thresholds_stage1_action_reject_{track}.yaml"),
    }

    tracks = ["scifact", "fever"] if args.track == "all" else [args.track]
    rows_out = []
    for track in tracks:
        in_path = sci_path if track == "scifact" else fever_path
        if not in_path:
            rows_out.append((track, "missing_parquet", None))
            continue
        p = Path(in_path)
        if not p.exists():
            rows_out.append((track, "missing_parquet", None))
            continue
        df = pd.read_parquet(p)
        if args.n is not None and args.n < len(df):
            df = df.sample(n=args.n, random_state=args.seed)
        logit_col = "raw_max_top3" if track == "scifact" else "logit_platt"
        y_col = "y"
        for name, template in configs.items():
            yaml_path = Path(str(template).format(track=track))
            conf = load_yaml(yaml_path, track)
            if conf is None:
                rows_out.append((track, name, "missing"))
                continue
            metrics = evaluate(df, logit_col, y_col, conf)
            rows_out.append((track, name, metrics))

    lines = []
    lines.append("# Stage1 Product Table")
    lines.append("")
    lines.append(f"- n: `{args.n if args.n else 'full'}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append("")
    for track in tracks:
        lines.append(f"## {track}")
        lines.append("")
        lines.append("| config | certify | ok_rate | coverage | decided_rate | decided_count | accuracy_on_decided | fp | fn | fp_upper95 | fn_upper95 | fp_decided | fn_decided | fp_decided_upper95 | fn_decided_upper95 | ci_safe | decided_ci_safe | decided_ci_status | accept_count | reject_count | fp_given_accept | fn_given_reject | fp_given_accept_upper95 | fn_given_reject_upper95 | accept_ci_status | reject_ci_status | action_ci_status |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
        for t, name, m in rows_out:
            if t != track:
                continue
            if m is None:
                lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            if m == "missing":
                lines.append(f"| {name} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
                continue
            certify_mode = "both"
            if name == "action_accept":
                certify_mode = "accept_only"
            elif name == "action_reject":
                certify_mode = "reject_only"
            config = load_yaml(Path(str(configs[name]).format(track=track)), track) or {}
            certify_label = config.get("certify", "") if name.startswith("action_") else ""
            if certify_label:
                certify_mode = certify_label
            decided_status, accept_status, reject_status, action_status, ci_safe, decided_ci_safe = compute_statuses(m, args, certify_mode, config)
            lines.append(
                "| {cfg} | {certify} | {ok_rate:.4f} | {coverage:.4f} | {decided_rate:.4f} | {decided_count} | {accuracy_on_decided:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {fp_upper95:.4f} | {fn_upper95:.4f} | {fp_decided_rate:.4f} | {fn_decided_rate:.4f} | {fp_decided_upper95:.4f} | {fn_decided_upper95:.4f} | {ci_safe} | {decided_ci_safe} | {decided_status} | {accept_count} | {reject_count} | {fp_given_accept:.4f} | {fn_given_reject:.4f} | {fp_given_accept_upper95:.4f} | {fn_given_reject_upper95:.4f} | {accept_status} | {reject_status} | {action_status} |".format(
                    cfg=name,
                    certify=certify_label,
                    ok_rate=m["ok_rate"],
                    coverage=m["coverage"],
                    decided_rate=m["coverage"],
                    decided_count=m["decided_count"],
                    accuracy_on_decided=m["accuracy_on_decided"],
                    fp_accept_rate=m["fp_accept_rate"],
                    fn_reject_rate=m["fn_reject_rate"],
                    fp_upper95=m["fp_upper95"],
                    fn_upper95=m["fn_upper95"],
                    fp_decided_rate=m["fp_decided_rate"],
                    fn_decided_rate=m["fn_decided_rate"],
                    fp_decided_upper95=m["fp_decided_upper95"],
                    fn_decided_upper95=m["fn_decided_upper95"],
                    ci_safe=ci_safe,
                    decided_ci_safe=decided_ci_safe,
                    decided_status=decided_status,
                    accept_count=m["accept_count"],
                    reject_count=m["reject_count"],
                    fp_given_accept=m["fp_given_accept"],
                    fn_given_reject=m["fn_given_reject"],
                    fp_given_accept_upper95=m["fp_given_accept_upper95"],
                    fn_given_reject_upper95=m["fn_given_reject_upper95"],
                    accept_status=accept_status,
                    reject_status=reject_status,
                    action_status=action_status,
                )
            )
        lines.append("")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
