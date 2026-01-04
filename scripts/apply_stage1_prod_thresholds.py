import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route


def load_prod(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"invalid prod thresholds in {path}")
    for k in ("tau", "t_lower", "t_upper", "mode"):
        if k not in data:
            raise SystemExit(f"missing {k} in {path}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--config_yaml")
    ap.add_argument("--prod_yaml")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    config_path = args.config_yaml or args.prod_yaml
    if not config_path:
        raise SystemExit("missing --config_yaml")
    prod = load_prod(Path(config_path))
    prod_fallback_used = bool(prod.get("fallback"))
    t_lower = float(prod["t_lower"])
    t_upper = float(prod["t_upper"])
    tau = float(prod["tau"])

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stage1 = row.get("stage1") or {}
            cs_ret = stage1.get("cs_ret")
            if cs_ret is None:
                logit = stage1.get("logit")
                if logit is None:
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                cs_ret = compute_cs_ret_from_logit(float(logit), tau)
                stage1["cs_ret"] = cs_ret
            decision, reason = decide_route(float(cs_ret), t_lower, t_upper)
            stage1["route_decision"] = decision
            stage1["route_reason"] = reason
            stage1["t_lower"] = t_lower
            stage1["t_upper"] = t_upper
            stage1["tau"] = tau
            stage1["prod_fallback_used"] = prod_fallback_used
            row["stage1"] = stage1
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
