import argparse
from pathlib import Path

import yaml


def parse_eval(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = {}
    cur = None
    in_table = False
    for line in lines:
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
                except Exception:
                    pass
        if in_table and line.strip() == "":
            in_table = False
    return sections


def load_product_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def pick_recommendation(baseline, product, weights):
    if not product:
        return "baseline", "product thresholds missing"
    lam_fp = float(weights.get("lambda_fp", 10.0))
    lam_fn = float(weights.get("lambda_fn", 10.0))
    lam_unc = float(weights.get("lambda_unc", 1.0))
    def util(m):
        return m["coverage"] - lam_fp * m["fp_accept_rate"] - lam_fn * m["fn_reject_rate"] - lam_unc * m["uncertain_rate"]
    if not baseline:
        return "product", "baseline metrics missing"
    if util(product) >= util(baseline):
        return "product", f"utility weights fp={lam_fp} fn={lam_fn} unc={lam_unc}"
    return "baseline", f"utility weights fp={lam_fp} fn={lam_fn} unc={lam_unc}"


def write_section(lines, title, baseline, product, joint, rec, reason):
    lines.append(title)
    lines.append("")
    lines.append("| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate | coverage | accuracy_on_decided |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    if baseline:
        lines.append("| baseline | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**baseline))
    if joint:
        lines.append("| joint_tuned | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**joint))
    if product:
        lines.append("| product_tuned | {ok_rate:.4f} | {fp_accept_rate:.4f} | {fn_reject_rate:.4f} | {uncertain_rate:.4f} | {coverage:.4f} | {accuracy_on_decided:.4f} |".format(**product))
    lines.append("")
    lines.append(f"- recommendation: `{rec}`")
    lines.append(f"- rationale: `{reason}`")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_report", required=True)
    ap.add_argument("--fever_report", required=True)
    ap.add_argument("--scifact_product_yaml", default="configs/thresholds_stage1_product_scifact.yaml")
    ap.add_argument("--fever_product_yaml", default="configs/thresholds_stage1_product_fever.yaml")
    ap.add_argument("--out_md", default="reports/stage1_product_decision.md")
    args = ap.parse_args()

    scifact_eval = parse_eval(Path(args.scifact_report))
    fever_eval = parse_eval(Path(args.fever_report))

    scifact_prod = load_product_yaml(Path(args.scifact_product_yaml))
    fever_prod = load_product_yaml(Path(args.fever_product_yaml))

    scifact_baseline = scifact_eval.get("baseline (checks off)")
    scifact_joint = scifact_eval.get("joint_tuned (checks off)")
    scifact_product = scifact_eval.get("product_tuned (checks off)")
    fever_baseline = fever_eval.get("baseline (checks off)")
    fever_joint = fever_eval.get("joint_tuned (checks off)")
    fever_product = fever_eval.get("product_tuned (checks off)")

    scifact_rec, scifact_reason = pick_recommendation(scifact_baseline, scifact_product, scifact_prod)
    fever_rec, fever_reason = pick_recommendation(fever_baseline, fever_product, fever_prod)

    lines = []
    lines.append("# Stage1 Product Decision")
    lines.append("")
    write_section(lines, "SciFact", scifact_baseline, scifact_product, scifact_joint, scifact_rec, scifact_reason)
    write_section(lines, "FEVER", fever_baseline, fever_product, fever_joint, fever_rec, fever_reason)
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
