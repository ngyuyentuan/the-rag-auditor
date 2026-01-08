import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route, routing_distribution


def parse_eval(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = {}
    cur = None
    in_table = False
    meta = {}
    for line in lines:
        if line.startswith("- in_path:"):
            meta["in_path"] = line.split("`")[1] if "`" in line else line.split(":", 1)[1].strip()
        if line.startswith("- logit_col:"):
            meta["logit_col"] = line.split("`")[1] if "`" in line else line.split(":", 1)[1].strip()
        if line.startswith("- y_col:"):
            meta["y_col"] = line.split("`")[1] if "`" in line else line.split(":", 1)[1].strip()
        if line.startswith("- n:"):
            try:
                meta["n"] = int(line.split("`")[1]) if "`" in line else int(line.split(":", 1)[1])
            except Exception:
                pass
        if line.startswith("## "):
            cur = line[3:].strip()
            sections[cur] = {}
            in_table = False
            continue
        if line.strip().startswith("| metric |"):
            in_table = True
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 2 and cur:
                metric = parts[0]
                try:
                    sections[cur][metric] = float(parts[1])
                    if len(parts) >= 3 and parts[2].startswith("[") and "," in parts[2]:
                        seg = parts[2].strip("[]").split(",")
                        sections[cur][metric + "_ci"] = (float(seg[0]), float(seg[1]))
                except Exception:
                    pass
        if in_table and line.strip() == "":
            in_table = False
    for sec in sections.values():
        if "n" not in sec and "n" in meta:
            sec["n"] = meta["n"]
        if "in_path" not in sec and "in_path" in meta:
            sec["in_path"] = meta["in_path"]
        if "logit_col" not in sec and "logit_col" in meta:
            sec["logit_col"] = meta["logit_col"]
        if "y_col" not in sec and "y_col" in meta:
            sec["y_col"] = meta["y_col"]
    return sections


def load_yaml_thresholds(path: Path, track: str):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    if "thresholds" in data:
        t = data["thresholds"].get(track)
        if not isinstance(t, dict):
            return None
        if not all(k in t for k in ("tau", "t_lower", "t_upper")):
            return None
        return {"tau": float(t["tau"]), "t_lower": float(t["t_lower"]), "t_upper": float(t["t_upper"])}
    if all(k in data for k in ("tau", "t_lower", "t_upper")):
        return {"tau": float(data["tau"]), "t_lower": float(data["t_lower"]), "t_upper": float(data["t_upper"])}
    return None


def load_product_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def wilson_ci(x, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    phat = x / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def utility(m, lam_fp, lam_fn, lam_unc):
    return m["coverage"] - lam_fp * m["fp_accept_rate"] - lam_fn * m["fn_reject_rate"] - lam_unc * m["uncertain_rate"]


def pick_safety(candidates):
    ranked = sorted(candidates.items(), key=lambda kv: (kv[1]["fp_accept_rate"] + kv[1]["fn_reject_rate"], -kv[1]["coverage"], -kv[1]["ok_rate"]))
    return ranked[0] if ranked else (None, None)


def pick_coverage(candidates):
    valid = [(k, v) for k, v in candidates.items() if (v["fp_accept_rate"] + v["fn_reject_rate"]) <= 0.05]
    pool = valid if valid else list(candidates.items())
    ranked = sorted(pool, key=lambda kv: (-kv[1]["coverage"], kv[1]["fp_accept_rate"] + kv[1]["fn_reject_rate"], -kv[1]["ok_rate"]))
    return ranked[0] if ranked else (None, None)


def pick_recommendation(baseline, product, weights):
    if not product:
        return "baseline", "product thresholds missing"
    lam_fp = float(weights.get("lambda_fp", 10.0))
    lam_fn = float(weights.get("lambda_fn", 10.0))
    lam_unc = float(weights.get("lambda_unc", 1.0))

    def util(m):
        return utility(m, lam_fp, lam_fn, lam_unc)

    if not baseline:
        return "product", "baseline metrics missing"
    if util(product) >= util(baseline):
        return "product", f"utility weights fp={lam_fp} fn={lam_fn} unc={lam_unc}"
    return "baseline", f"utility weights fp={lam_fp} fn={lam_fn} unc={lam_unc}"


def write_section(lines, title, baseline, joint, safety_name, safety_metrics, coverage_name, coverage_metrics, extras):
    lines.append(title)
    lines.append("")
    lines.append("| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    if baseline:
        lines.append("| baseline | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**baseline))
    if joint:
        lines.append("| joint_tuned | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**joint))
    for label, metrics in extras:
        if metrics == "missing":
            lines.append(f"| {label} | missing | missing | missing | missing | missing | missing |")
        elif metrics:
            lines.append("| {label} | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(label=label, **metrics))
    if safety_metrics:
        lines.append("| product_safety | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**safety_metrics))
    if coverage_metrics:
        lines.append("| product_coverage | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**coverage_metrics))
    lines.append("")
    if safety_name:
        lines.append(f"- safety_selected: `{safety_name}`")
    if coverage_name:
        lines.append(f"- coverage_selected: `{coverage_name}`")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_report", required=True)
    ap.add_argument("--fever_report", required=True)
    ap.add_argument("--scifact_product_yaml", default="configs/thresholds_stage1_product_scifact.yaml")
    ap.add_argument("--fever_product_yaml", default="configs/thresholds_stage1_product_fever.yaml")
    ap.add_argument("--scifact_2d_yaml")
    ap.add_argument("--fever_2d_yaml")
    ap.add_argument("--scifact_accept_yaml")
    ap.add_argument("--scifact_reject_yaml")
    ap.add_argument("--fever_accept_yaml")
    ap.add_argument("--fever_reject_yaml")
    ap.add_argument("--scifact_action_accept_yaml")
    ap.add_argument("--scifact_action_reject_yaml")
    ap.add_argument("--fever_action_accept_yaml")
    ap.add_argument("--fever_action_reject_yaml")
    ap.add_argument("--max_fp", type=float, default=0.05)
    ap.add_argument("--max_fn", type=float, default=0.05)
    ap.add_argument("--constraint_ci", choices=["none", "wilson"], default="wilson")
    ap.add_argument("--min_coverage_safety_scifact", type=float, default=0.35)
    ap.add_argument("--min_coverage_coverage_scifact", type=float, default=0.45)
    ap.add_argument("--min_coverage_safety_fever", type=float, default=0.15)
    ap.add_argument("--min_coverage_coverage_fever", type=float, default=0.30)
    ap.add_argument("--max_fp_decided_coverage", type=float, default=0.10)
    ap.add_argument("--max_fn_decided_coverage", type=float, default=0.10)
    ap.add_argument("--out_md", default="reports/stage1_product_decision.md")
    args = ap.parse_args()

    scifact_eval = parse_eval(Path(args.scifact_report))
    fever_eval = parse_eval(Path(args.fever_report))

    baseline_yaml = Path("configs/thresholds.yaml")
    joint_scifact_yaml = Path("configs/thresholds_stage1_joint_tuned_scifact.yaml")
    joint_fever_yaml = Path("configs/thresholds_stage1_joint_tuned_fever.yaml")
    prod_scifact_yaml = Path(args.scifact_product_yaml)
    prod_fever_yaml = Path(args.fever_product_yaml)
    safety_out_scifact = Path("configs/thresholds_stage1_product_safety_scifact.yaml")
    safety_out_fever = Path("configs/thresholds_stage1_product_safety_fever.yaml")
    cov_out_scifact = Path("configs/thresholds_stage1_product_coverage_scifact.yaml")
    cov_out_fever = Path("configs/thresholds_stage1_product_coverage_fever.yaml")

    def thresholds_for(name, track):
        if name == "baseline":
            return load_yaml_thresholds(baseline_yaml, track)
        if name == "joint_tuned":
            return load_yaml_thresholds(Path(f"configs/thresholds_stage1_joint_tuned_{track}.yaml"), track)
        if name == "product_tuned":
            return load_yaml_thresholds(Path(f"configs/thresholds_stage1_product_{track}.yaml"), track)
        if name == "product_2d":
            path = args.scifact_2d_yaml if track == "scifact" else args.fever_2d_yaml
            return load_yaml_thresholds(Path(path), track) if path else None
        return None

    def eval_yaml(track, yaml_path, meta):
        if not yaml_path:
            return None
        path = Path(yaml_path)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        certify = data.get("certify", "both")
        min_accept = int(data.get("min_accept_count", 0))
        min_reject = int(data.get("min_reject_count", 0))
        max_fp_accept = float(data.get("max_fp_given_accept_upper95", 1.0))
        max_fn_reject = float(data.get("max_fn_given_reject_upper95", 1.0))
        if "thresholds" in data:
            th = data["thresholds"].get(track)
        else:
            th = data
        if not isinstance(th, dict):
            return None
        if not all(k in th for k in ("tau", "t_lower", "t_upper")):
            return None
        if not meta.get("in_path") or not meta.get("logit_col") or not meta.get("y_col"):
            return None
        p = Path(meta["in_path"])
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if meta["logit_col"] not in df.columns or meta["y_col"] not in df.columns:
            return None
        y = pd.to_numeric(df[meta["y_col"]], errors="coerce")
        logits = pd.to_numeric(df[meta["logit_col"]], errors="coerce")
        mask = y.isin([0, 1]) & logits.notna()
        df = df.loc[mask].copy()
        df[meta["y_col"]] = y[mask].astype(int)
        df[meta["logit_col"]] = logits[mask].astype(float)
        if "c_accept" in th and "c_reject" in th:
            cs_col = None
            for cand in ["cs_ret", "retrieval_score", "sim", "similarity", "score_ret", "max_sim"]:
                if cand in df.columns:
                    cs_col = cand
                    break
            if cs_col is None:
                return None
            cs_vals = pd.to_numeric(df[cs_col], errors="coerce")
            mask2 = cs_vals.notna()
            df = df.loc[mask2].copy()
            df[cs_col] = cs_vals[mask2].astype(float)
        if "n" in meta and meta["n"] and meta["n"] < len(df):
            df = df.sample(n=meta["n"], random_state=14)
        y_arr = df[meta["y_col"]].astype(int).to_numpy()
        logits_arr = df[meta["logit_col"]].astype(float).to_numpy()
        cs_ret = np.asarray([compute_cs_ret_from_logit(v, float(th["tau"])) for v in logits_arr], dtype=float)
        if "c_accept" in th and "c_reject" in th:
            accept = (logits_arr >= float(th["t_upper"])) & (cs_ret >= float(th["c_accept"]))
            reject = (logits_arr <= float(th["t_lower"])) | (cs_ret <= float(th["c_reject"]))
            uncertain = ~(accept | reject)
            n = len(y_arr)
            fp = np.sum(accept & (y_arr == 0))
            fn = np.sum(reject & (y_arr == 1))
            tp = np.sum(accept & (y_arr == 1))
            tn = np.sum(reject & (y_arr == 0))
            accept_count = int(np.sum(accept))
            reject_count = int(np.sum(reject))
            dist = {"ACCEPT": float(np.mean(accept)) if n else 0.0, "REJECT": float(np.mean(reject)) if n else 0.0, "UNCERTAIN": float(np.mean(uncertain)) if n else 0.0}
        else:
            dist = routing_distribution(cs_ret, float(th["t_lower"]), float(th["t_upper"]))
            decisions = []
            for pval in cs_ret:
                d, _ = decide_route(pval, float(th["t_lower"]), float(th["t_upper"]))
                decisions.append(d)
            d_arr = np.asarray(decisions)
            n = len(d_arr)
            fp = int(np.sum((d_arr == "ACCEPT") & (y_arr == 0)))
            fn = int(np.sum((d_arr == "REJECT") & (y_arr == 1)))
            tp = int(np.sum((d_arr == "ACCEPT") & (y_arr == 1)))
            tn = int(np.sum((d_arr == "REJECT") & (y_arr == 0)))
            accept_count = int(np.sum(d_arr == "ACCEPT"))
            reject_count = int(np.sum(d_arr == "REJECT"))
        fp_rate = fp / n if n else 0.0
        fn_rate = fn / n if n else 0.0
        ok_rate = 1.0 - fp_rate - fn_rate
        coverage = 1.0 - dist["UNCERTAIN"]
        decided = tp + tn + fp + fn
        accuracy_on_decided = (tp + tn) / decided if decided else 0.0
        fp_accept_upper95 = wilson_ci(fp, accept_count)[1] if accept_count else 0.0
        fn_reject_upper95 = wilson_ci(fn, reject_count)[1] if reject_count else 0.0
        return {
            "ok_rate": ok_rate,
            "fp_accept_rate": fp_rate,
            "fn_reject_rate": fn_rate,
            "uncertain_rate": dist["UNCERTAIN"],
            "coverage": coverage,
            "accuracy_on_decided": accuracy_on_decided,
            "n": n,
            "accept_count": accept_count,
            "reject_count": reject_count,
            "fp_given_accept_upper95": fp_accept_upper95,
            "fn_given_reject_upper95": fn_reject_upper95,
            "certify": certify,
            "min_accept_count": min_accept,
            "min_reject_count": min_reject,
            "max_fp_given_accept_upper95": max_fp_accept,
            "max_fn_given_reject_upper95": max_fn_reject,
        }

    def constraint_ok(m, max_fp, max_fn):
        n = int(m.get("n", 0))
        fp = m.get("fp_accept_rate", 0.0)
        fn = m.get("fn_reject_rate", 0.0)
        fp_ci = m.get("fp_accept_rate_ci")
        fn_ci = m.get("fn_reject_rate_ci")
        if args.constraint_ci == "wilson":
            fp_upper = fp_ci[1] if fp_ci else (wilson_ci(fp * n, n)[1] if n else fp)
            fn_upper = fn_ci[1] if fn_ci else (wilson_ci(fn * n, n)[1] if n else fn)
        else:
            fp_upper = fp
            fn_upper = fn
        m["fp_upper95"] = fp_upper
        m["fn_upper95"] = fn_upper
        return fp_upper <= max_fp and fn_upper <= max_fn

    def parse_candidates(eval_data, track):
        mapping = {}
        for name, sec in eval_data.items():
            key = None
            if name.startswith("baseline"):
                key = "baseline"
            elif name.startswith("joint_tuned"):
                key = "joint_tuned"
            elif name.startswith("product_tuned"):
                key = "product_tuned"
            if key:
                mapping[key] = sec
        meta = eval_data.get("baseline (checks off)", {})
        yaml_path = args.scifact_2d_yaml if track == "scifact" else args.fever_2d_yaml
        if yaml_path:
            metrics = eval_yaml(track, yaml_path, meta)
            if metrics:
                mapping["product_2d"] = metrics
        return mapping

    def resolve_optional(path_arg, track, kind):
        if path_arg:
            return Path(path_arg)
        default_path = Path(f"configs/thresholds_stage1_action_{kind}_{track}.yaml")
        return default_path if default_path.exists() else None

    def eval_optional(track, yaml_path, eval_data):
        if yaml_path is None:
            return None
        if not yaml_path.exists():
            return "missing"
        meta = eval_data.get("baseline (checks off)", {})
        return eval_yaml(track, str(yaml_path), meta)

    def action_ci_status(metrics):
        certify = metrics.get("certify", "both")
        accept_count = int(metrics.get("accept_count", 0))
        reject_count = int(metrics.get("reject_count", 0))
        min_accept = int(metrics.get("min_accept_count", 0))
        min_reject = int(metrics.get("min_reject_count", 0))
        max_fp = float(metrics.get("max_fp_given_accept_upper95", 1.0))
        max_fn = float(metrics.get("max_fn_given_reject_upper95", 1.0))
        fp_up = float(metrics.get("fp_given_accept_upper95", 0.0))
        fn_up = float(metrics.get("fn_given_reject_upper95", 0.0))
        if certify == "accept_only":
            if accept_count == 0 or accept_count < min_accept:
                return "INSUFFICIENT_N"
            return "PASS" if fp_up <= max_fp else "FAIL"
        if certify == "reject_only":
            if reject_count == 0 or reject_count < min_reject:
                return "INSUFFICIENT_N"
            return "PASS" if fn_up <= max_fn else "FAIL"
        if accept_count == 0 or accept_count < min_accept or reject_count == 0 or reject_count < min_reject:
            return "INSUFFICIENT_N"
        return "PASS" if fp_up <= max_fp and fn_up <= max_fn else "FAIL"

    def filter_constraint(candidates, min_cov, enforce_decided):
        filtered = {}
        for k, v in candidates.items():
            if min_cov and v.get("coverage", 0.0) < min_cov:
                continue
            if not constraint_ok(v, args.max_fp, args.max_fn):
                continue
            if enforce_decided:
                n = int(v.get("n", 0))
                decided = int((1 - v.get("uncertain_rate", 0.0)) * n) if n else 0
                fp = v.get("fp_accept_rate", 0.0) * n
                fn = v.get("fn_reject_rate", 0.0) * n
                fp_dec = fp / decided if decided else 0.0
                fn_dec = fn / decided if decided else 0.0
                fp_dec_up = wilson_ci(fp_dec * decided, decided)[1] if decided else fp_dec
                fn_dec_up = wilson_ci(fn_dec * decided, decided)[1] if decided else fn_dec
                if fp_dec_up > args.max_fp_decided_coverage or fn_dec_up > args.max_fn_decided_coverage:
                    continue
            filtered[k] = v
        return filtered

    def process(track, eval_data, prod_yaml_path, safety_out, cov_out, min_cov_safety, min_cov_cov, accept_yaml, reject_yaml):
        candidates = parse_candidates(eval_data, track)
        filtered_safety = filter_constraint(candidates, min_cov_safety, False)
        if not filtered_safety and "baseline" in candidates:
            filtered_safety["baseline"] = candidates["baseline"]
        filtered_cov = filter_constraint(candidates, min_cov_cov, True)
        if not filtered_cov and "baseline" in candidates:
            filtered_cov["baseline"] = candidates["baseline"]
        safety_name, safety_metrics = pick_safety(filtered_safety)
        coverage_name, coverage_metrics = pick_coverage(filtered_cov)
        if safety_metrics:
            th = thresholds_for(safety_name, track)
            if th:
                safety_out.write_text(yaml.safe_dump(th, sort_keys=False), encoding="utf-8")
        if coverage_metrics:
            th = thresholds_for(coverage_name, track)
            if th:
                cov_out.write_text(yaml.safe_dump(th, sort_keys=False), encoding="utf-8")
        baseline = candidates.get("baseline")
        joint = candidates.get("joint_tuned")
        note = None
        if not filtered_cov:
            note = "No CI-safe coverage candidate; fallback to baseline"
        extras = []
        action_notes = []
        accept_metrics = eval_optional(track, accept_yaml, eval_data)
        if accept_yaml is not None or accept_metrics is not None:
            extras.append(("action_accept", accept_metrics))
            if isinstance(accept_metrics, dict):
                action_notes.append(f"- action_accept_certify: `{accept_metrics.get('certify','both')}`")
                action_notes.append(f"- action_accept_action_ci_status: `{action_ci_status(accept_metrics)}`")
        reject_metrics = eval_optional(track, reject_yaml, eval_data)
        if reject_yaml is not None or reject_metrics is not None:
            extras.append(("action_reject", reject_metrics))
            if isinstance(reject_metrics, dict):
                action_notes.append(f"- action_reject_certify: `{reject_metrics.get('certify','both')}`")
                action_notes.append(f"- action_reject_action_ci_status: `{action_ci_status(reject_metrics)}`")
        return baseline, joint, safety_name, safety_metrics, coverage_name, coverage_metrics, note, extras, action_notes

    sci_accept_arg = args.scifact_accept_yaml or args.scifact_action_accept_yaml
    sci_reject_arg = args.scifact_reject_yaml or args.scifact_action_reject_yaml
    fev_accept_arg = args.fever_accept_yaml or args.fever_action_accept_yaml
    fev_reject_arg = args.fever_reject_yaml or args.fever_action_reject_yaml
    sci_accept_yaml = resolve_optional(sci_accept_arg, "scifact", "accept")
    sci_reject_yaml = resolve_optional(sci_reject_arg, "scifact", "reject")
    fev_accept_yaml = resolve_optional(fev_accept_arg, "fever", "accept")
    fev_reject_yaml = resolve_optional(fev_reject_arg, "fever", "reject")

    sci_baseline, sci_joint, sci_safety_name, sci_safety_metrics, sci_cov_name, sci_cov_metrics, sci_note, sci_extras, sci_action_notes = process(
        "scifact",
        scifact_eval,
        prod_scifact_yaml,
        safety_out_scifact,
        cov_out_scifact,
        args.min_coverage_safety_scifact,
        args.min_coverage_coverage_scifact,
        sci_accept_yaml,
        sci_reject_yaml,
    )
    fev_baseline, fev_joint, fev_safety_name, fev_safety_metrics, fev_cov_name, fev_cov_metrics, fev_note, fev_extras, fev_action_notes = process(
        "fever",
        fever_eval,
        prod_fever_yaml,
        safety_out_fever,
        cov_out_fever,
        args.min_coverage_safety_fever,
        args.min_coverage_coverage_fever,
        fev_accept_yaml,
        fev_reject_yaml,
    )

    lines = []
    lines.append("# Stage1 Product Decision")
    lines.append("")
    write_section(lines, "SciFact", sci_baseline, sci_joint, sci_safety_name, sci_safety_metrics, sci_cov_name, sci_cov_metrics, sci_extras)
    if sci_action_notes:
        lines.extend(sci_action_notes)
        lines.append("")
    if sci_note:
        lines.append(f"- note: {sci_note}")
        lines.append("")
    write_section(lines, "FEVER", fev_baseline, fev_joint, fev_safety_name, fev_safety_metrics, fev_cov_name, fev_cov_metrics, fev_extras)
    if fev_action_notes:
        lines.extend(fev_action_notes)
        lines.append("")
    if fev_note:
        lines.append(f"- note: {fev_note}")
        lines.append("")
    lines.append("Risks & mitigations")
    lines.append("")
    lines.append("- ok_rate can remain high while coverage is low; coverage and accuracy_on_decided are required for product readiness.")
    lines.append("- fp_accept_rate and fn_reject_rate should be monitored for drift.")
    lines.append("- Stage2 should report evidence hit and verdict accuracy to validate end-to-end correctness.")
    lines.append("")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
