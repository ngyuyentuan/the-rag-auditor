import argparse
from pathlib import Path


def parse_eval(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = {"baseline": {}, "tuned": {}, "meta": {}}
    cur = None
    in_table = False
    for line in lines:
        if line.strip().startswith("- in_path:"):
            parts = line.split("`")
            if len(parts) >= 2:
                sections["meta"]["in_path"] = parts[1]
        if line.strip().startswith("- logit_col:"):
            parts = line.split("`")
            if len(parts) >= 2:
                sections["meta"]["logit_col"] = parts[1]
        if line.strip().lower() == "baseline thresholds":
            cur = "baseline"
            in_table = False
            continue
        if line.strip().lower() == "tuned thresholds":
            cur = "tuned"
            in_table = False
            continue
        if line.strip().startswith("| metric |"):
            in_table = True
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3 and cur:
                metric, val, ci = parts[0], parts[1], parts[2]
                try:
                    sections[cur][metric] = {"value": float(val), "ci": ci}
                except Exception:
                    pass
        if in_table and line.strip() == "":
            in_table = False
    return sections


def format_row(label, metrics):
    if not metrics:
        return f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return "| {label} | {ok} | {fp} | {fn} | {unc} | {cov} | {acc} | {ci_ok} |".format(
        label=label,
        ok=f"{metrics['ok_rate']['value']:.4f}",
        fp=f"{metrics['fp_accept_rate']['value']:.4f}",
        fn=f"{metrics['fn_reject_rate']['value']:.4f}",
        unc=f"{metrics['uncertain_rate']['value']:.4f}",
        cov=f"{metrics['coverage']['value']:.4f}",
        acc=f"{metrics['accuracy_on_decided']['value']:.4f}",
        ci_ok=metrics['ok_rate']['ci'],
    )


def recommend(metrics):
    if not metrics:
        return "insufficient data"
    fp = metrics["fp_accept_rate"]["value"]
    fn = metrics["fn_reject_rate"]["value"]
    cov = metrics["coverage"]["value"]
    if fp <= 0.05 and fn <= 0.05 and cov >= 0.30:
        return "Stage1-only OK"
    if fp > 0.05 or fn > 0.05:
        return "Stage2 required (risk too high)"
    return "Stage2 required (coverage too low)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_report", required=True)
    ap.add_argument("--fever_report", required=True)
    ap.add_argument("--out_md", default="reports/stage1_joint_decision.md")
    args = ap.parse_args()

    scifact = parse_eval(Path(args.scifact_report))
    fever = parse_eval(Path(args.fever_report))

    lines = []
    lines.append("# Stage1 Joint Decision")
    lines.append("")
    lines.append("SciFact summary")
    lines.append("")
    lines.append("| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided | ok_rate CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(format_row("baseline", scifact.get("baseline")))
    lines.append(format_row("joint_tuned", scifact.get("tuned")))
    lines.append("")
    lines.append("FEVER summary")
    lines.append("")
    lines.append("| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided | ok_rate CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(format_row("baseline", fever.get("baseline")))
    lines.append(format_row("joint_tuned", fever.get("tuned")))
    lines.append("")
    lines.append("Recommendation")
    lines.append("")
    lines.append(f"- scifact: {recommend(scifact.get('tuned') or scifact.get('baseline'))}")
    lines.append(f"- fever: {recommend(fever.get('tuned') or fever.get('baseline'))}")
    lines.append("")
    lines.append("Risks & mitigations")
    lines.append("")
    lines.append("- ok_rate can look strong while coverage remains low; use coverage and accuracy_on_decided for product readiness.")
    lines.append("- Stage2 must report evidence hit and verdict accuracy to validate end-to-end correctness.")
    lines.append("- Monitor fp_accept_rate and fn_reject_rate drift; alert when exceeding product gates.")
    lines.append("")
    lines.append("Repro commands")
    lines.append("")
    lines.append("```")
    sci_in = scifact.get("meta", {}).get("in_path", "")
    sci_logit = scifact.get("meta", {}).get("logit_col", "raw_max_top3")
    fev_in = fever.get("meta", {}).get("in_path", "")
    fev_logit = fever.get("meta", {}).get("logit_col", "logit_platt")
    lines.append(f"wsl -e bash -lc \"cd ~/the-rag-auditor && .venv/bin/python scripts/eval_stage1_product.py --track scifact --in_path '{sci_in}' --logit_col {sci_logit} --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_joint_tuned_scifact.yaml --n 1000 --out_md reports/stage1_product_eval_scifact_joint.md\"")
    lines.append(f"wsl -e bash -lc \"cd ~/the-rag-auditor && .venv/bin/python scripts/eval_stage1_product.py --track fever --in_path '{fev_in}' --logit_col {fev_logit} --y_col y --tuned_thresholds_yaml configs/thresholds_stage1_joint_tuned_fever.yaml --n 1000 --out_md reports/stage1_product_eval_fever_joint.md\"")
    lines.append("```")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
