import argparse
import sys
from pathlib import Path

import yaml


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def parse_eval(path):
    lines = read_lines(path)
    blocks = []
    current = {}
    for line in lines:
        if line.strip().startswith("| ok_rate"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                try:
                    current["ok_rate"] = float(parts[1])
                except ValueError:
                    pass
        if line.strip().startswith("| fp_accept_rate"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                try:
                    current["fp_accept_rate"] = float(parts[1])
                except ValueError:
                    pass
        if line.strip().startswith("| fn_reject_rate"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                try:
                    current["fn_reject_rate"] = float(parts[1])
                except ValueError:
                    pass
        if line.strip().startswith("| uncertain_rate"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts:
                try:
                    current["uncertain_rate"] = float(parts[1])
                except ValueError:
                    pass
        if line.startswith("## Tuned config"):
            if current:
                blocks.append(current)
                current = {}
    if current:
        blocks.append(current)
    if len(blocks) == 1:
        return {"baseline": blocks[0], "tuned": None}
    return {"baseline": blocks[0], "tuned": blocks[1]}


def parse_profile(path, profile_key):
    lines = read_lines(path)
    data = {}
    capture = False
    for line in lines:
        if line.strip().lower().startswith(f"## profile: {profile_key}".lower()):
            capture = True
            continue
        if capture:
            if line.startswith("## "):
                break
            if line.strip().startswith("- t_lower:"):
                data["t_lower"] = float(line.split(":")[1].strip())
            if line.strip().startswith("- t_upper:"):
                data["t_upper"] = float(line.split(":")[1].strip())
            if line.strip().startswith("- uncertain_rate:"):
                data["uncertain_rate"] = float(line.split(":")[1].strip())
            if line.strip().startswith("- fp_accept_rate:"):
                data["fp_accept_rate"] = float(line.split(":")[1].strip())
            if line.strip().startswith("- fn_reject_rate:"):
                data["fn_reject_rate"] = float(line.split(":")[1].strip())
            if line.strip().startswith("- ok_rate:"):
                data["ok_rate"] = float(line.split(":")[1].strip())
    return data


def load_yaml_thresholds(path):
    p = Path(path)
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return {
        "tau": float(data.get("tau")),
        "t_lower": float(data.get("t_lower")),
        "t_upper": float(data.get("t_upper")),
    }


def build_table_row(label, metrics, thresholds=None):
    if metrics is None:
        return f"| {label} | n/a | n/a | n/a | n/a |"
    return "| {label} | {ok:.4f} | {fp:.4f} | {fn:.4f} | {unc:.4f} |".format(
        label=label,
        ok=metrics.get("ok_rate", 0.0),
        fp=metrics.get("fp_accept_rate", 0.0),
        fn=metrics.get("fn_reject_rate", 0.0),
        unc=metrics.get("uncertain_rate", 0.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_report", required=True)
    ap.add_argument("--fever_report", required=True)
    ap.add_argument("--scifact_tuning", required=True)
    ap.add_argument("--fever_tuning", required=True)
    ap.add_argument("--scifact_profile", required=True)
    ap.add_argument("--fever_profile", required=True)
    ap.add_argument("--out_md", default="reports/stage1_decision_memo.md")
    args = ap.parse_args()

    sci_eval = parse_eval(args.scifact_report)
    fev_eval = parse_eval(args.fever_report)

    sci_tuned_yaml = load_yaml_thresholds("configs/thresholds_stage1_tuned_scifact.yaml")
    fev_tuned_yaml = load_yaml_thresholds("configs/thresholds_stage1_tuned_fever.yaml")

    sci_safety = parse_profile(args.scifact_profile, "safety_first")
    sci_cov = parse_profile(args.scifact_profile, "coverage_first")
    fev_safety = parse_profile(args.fever_profile, "safety_first")
    fev_cov = parse_profile(args.fever_profile, "coverage_first")

    lines = []
    lines.append("# Stage1 Decision Memo")
    lines.append("")
    lines.append("## SciFact summary")
    lines.append("")
    lines.append("| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(build_table_row("baseline", sci_eval["baseline"]))
    lines.append(build_table_row("tuned", sci_eval["tuned"]))
    lines.append(build_table_row("profile_safety_first", sci_safety))
    lines.append(build_table_row("profile_coverage_first", sci_cov))
    lines.append("")
    lines.append("### SciFact thresholds")
    if sci_tuned_yaml:
        lines.append(f"- tuned_yaml: t_lower={sci_tuned_yaml['t_lower']}, t_upper={sci_tuned_yaml['t_upper']}, tau={sci_tuned_yaml['tau']}")
    lines.append("")
    lines.append("### SciFact product tradeoff")
    lines.append("Baseline keeps risk balanced; tuned raises uncertain slightly but improves ok_rate marginally; coverage_first profile defers less but would require Stage2 to manage risk.")
    lines.append("")
    lines.append("## FEVER summary")
    lines.append("")
    lines.append("| config | ok_rate | fp_accept_rate | fn_reject_rate | uncertain_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(build_table_row("baseline", fev_eval["baseline"]))
    lines.append(build_table_row("tuned", fev_eval["tuned"]))
    lines.append(build_table_row("profile_safety_first", fev_safety))
    lines.append(build_table_row("profile_coverage_first", fev_cov))
    lines.append("")
    lines.append("### FEVER thresholds")
    if fev_tuned_yaml:
        lines.append(f"- tuned_yaml: t_lower={fev_tuned_yaml['t_lower']}, t_upper={fev_tuned_yaml['t_upper']}, tau={fev_tuned_yaml['tau']}")
    lines.append("")
    lines.append("### FEVER product tradeoff")
    lines.append("Baseline has near-zero accepts; tuned raises coverage but increases fn; coverage_first profile reduces uncertain but adds risk and should be paired with strong Stage2.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- SciFact: retain tuned weighted config (export budget=0.50) if Stage2 can handle similar deferral; ok_rate improves slightly and fp/fn stay bounded.")
    lines.append("- FEVER: prefer baseline for safety; tuned raises fn; only ship tuned if Stage2 evidence is strong and monitored.")
    lines.append("")
    lines.append("## Risks & mitigations")
    lines.append("- ok_rate alone hides evidence quality; Stage2 must measure evidence hit and verdict accuracy.")
    lines.append("- Monitor fp_accept and fn_reject drift over time; alert if exceeding product gates.")
    lines.append("- Ensure abstention/UNCERTAIN flows route to human/Stage2 with clear policy.")
    lines.append("")
    lines.append("## Repro commands")
    lines.append("```")
    lines.append(" ".join(sys.argv))
    lines.append("```")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
