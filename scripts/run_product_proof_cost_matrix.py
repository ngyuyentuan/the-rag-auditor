import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buffer import compute_cs_ret_from_logit, decide_route


def write_sample(path, n):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"row_idx": i}) + "\n")
    return str(path)


def run(cmd):
    subprocess.run(cmd, check=True)


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[int(f)] * (c - k)
    d1 = vals[int(c)] * (k - f)
    return d0 + d1


def summarize(rows):
    n = 0
    accept = 0
    reject = 0
    uncertain = 0
    fp = 0
    fn = 0
    stage2_ran = 0
    rerank_gt0 = 0
    nli_gt0 = 0
    total_ms = []
    for row in rows:
        n += 1
        stage1 = row.get("stage1", {}) or {}
        decision = stage1.get("route_decision")
        if decision == "ACCEPT":
            accept += 1
        elif decision == "REJECT":
            reject += 1
        else:
            uncertain += 1
        y = (row.get("ground_truth") or {}).get("y")
        if decision == "ACCEPT" and y == 0:
            fp += 1
        if decision == "REJECT" and y == 1:
            fn += 1
        stage2 = row.get("stage2", {}) or {}
        if stage2.get("ran") is True:
            stage2_ran += 1
        timing = row.get("timing_ms", {}) or {}
        total_ms.append(float(timing.get("total_ms", 0.0)))
        if float(timing.get("rerank_ms", 0.0)) > 0:
            rerank_gt0 += 1
        if float(timing.get("nli_ms", 0.0)) > 0:
            nli_gt0 += 1
    n = n or 1
    return {
        "accept_rate": accept / n,
        "reject_rate": reject / n,
        "uncertain_rate": uncertain / n,
        "stage2_rate": stage2_ran / n,
        "fp_accept_rate": fp / n,
        "fn_reject_rate": fn / n,
        "ok_rate_stage1": 1.0 - (fp / n) - (fn / n),
        "mean_ms": sum(total_ms) / n if total_ms else 0.0,
        "p95_ms": percentile(total_ms, 95),
        "rerank_ms_gt0": rerank_gt0,
        "nli_ms_gt0": nli_gt0,
    }


def format_float(x):
    return f"{x:.4f}"


def apply_cost_thresholds(in_jsonl, out_jsonl, t_lower, t_upper, tau, policy):
    in_path = Path(in_jsonl)
    out_path = Path(out_jsonl)
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
                if logit is not None:
                    cs_ret = compute_cs_ret_from_logit(float(logit), tau)
                    stage1["cs_ret"] = cs_ret
            if cs_ret is not None:
                decision, reason = decide_route(float(cs_ret), t_lower, t_upper)
                stage1["route_decision"] = decision
                stage1["route_reason"] = reason
                stage1["t_lower"] = t_lower
                stage1["t_upper"] = t_upper
                stage1["tau"] = tau
                row["stage1"] = stage1
            stage2 = row.get("stage2") or {}
            if policy == "always":
                should_run = True
            elif policy == "uncertain_only":
                should_run = stage1.get("route_decision") == "UNCERTAIN"
            else:
                should_run = stage2.get("ran") is True
            if should_run:
                stage2["ran"] = True
            else:
                stage2["ran"] = False
                stage2["rerank"] = {}
                stage2["nli"] = {}
                timing = row.get("timing_ms") or {}
                timing["rerank_ms"] = 0.0
                timing["nli_ms"] = 0.0
                if "stage1_ms" in timing:
                    timing["total_ms"] = float(timing.get("stage1_ms", 0.0))
                row["timing_ms"] = timing
            row["stage2"] = stage2
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scifact_in_path", required=True)
    ap.add_argument("--fever_in_path", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="runs/product_proof_cost")
    ap.add_argument("--cost_yaml_scifact", default="configs/thresholds_stage1_cost_scifact.yaml")
    ap.add_argument("--cost_yaml_fever", default="configs/thresholds_stage1_cost_fever.yaml")
    ap.add_argument("--out_md", default="reports/product_proof_cost_matrix.md")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scifact_sample = out_dir / f"scifact_sample_{args.n}.jsonl"
    fever_sample = out_dir / f"fever_sample_{args.n}.jsonl"
    if not scifact_sample.exists():
        write_sample(scifact_sample, args.n)
    if not fever_sample.exists():
        write_sample(fever_sample, args.n)

    def base_cmd(track, in_path, sample_path):
        cmd = [
            ".venv/bin/python",
            "scripts/day12_e2e_run_500.py",
            "--track",
            track,
            "--n",
            str(args.n),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--stage2_policy",
            "uncertain_only",
            "--baseline_mode",
            "calibrated",
            "--rerank_topk",
            "20",
            "--rerank_keep",
            "5",
            "--batch_size",
            "16",
            "--out",
            str(out_dir / f"{track}_baseline_uncertain_only_{args.n}_real.jsonl"),
            "--in_path",
            in_path,
            "--sample",
            str(sample_path),
        ]
        if track == "fever":
            cmd += ["--logit_col", "logit_platt", "--y_col", "y"]
        return cmd

    scifact_base = base_cmd("scifact", args.scifact_in_path, scifact_sample)
    fever_base = base_cmd("fever", args.fever_in_path, fever_sample)

    scifact_u = out_dir / f"scifact_baseline_uncertain_only_{args.n}_real.jsonl"
    fever_u = out_dir / f"fever_baseline_uncertain_only_{args.n}_real.jsonl"
    scifact_a = out_dir / f"scifact_baseline_always_{args.n}_real.jsonl"
    fever_a = out_dir / f"fever_baseline_always_{args.n}_real.jsonl"

    if not scifact_u.exists():
        run(scifact_base + ["--out", str(scifact_u)])
    if not fever_u.exists():
        run(fever_base + ["--out", str(fever_u)])
    if not scifact_a.exists():
        run(scifact_base + ["--stage2_policy", "always", "--out", str(scifact_a)])
    if not fever_a.exists():
        run(fever_base + ["--stage2_policy", "always", "--out", str(fever_a)])

    def load_cost_yaml(path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k in ("tau", "t_lower", "t_upper"):
            if k not in data:
                raise SystemExit(f"missing {k} in {path}")
        return float(data["tau"]), float(data["t_lower"]), float(data["t_upper"])

    import yaml
    sc_tau, sc_l, sc_u = load_cost_yaml(args.cost_yaml_scifact)
    fv_tau, fv_l, fv_u = load_cost_yaml(args.cost_yaml_fever)

    sc_prod_u = out_dir / f"scifact_cost_uncertain_only_{args.n}.jsonl"
    sc_prod_a = out_dir / f"scifact_cost_always_{args.n}.jsonl"
    fv_prod_u = out_dir / f"fever_cost_uncertain_only_{args.n}.jsonl"
    fv_prod_a = out_dir / f"fever_cost_always_{args.n}.jsonl"

    if scifact_u.exists() and not sc_prod_u.exists():
        apply_cost_thresholds(scifact_u, sc_prod_u, sc_l, sc_u, sc_tau, "uncertain_only")
    if scifact_a.exists() and not sc_prod_a.exists():
        apply_cost_thresholds(scifact_a, sc_prod_a, sc_l, sc_u, sc_tau, "always")
    if fever_u.exists() and not fv_prod_u.exists():
        apply_cost_thresholds(fever_u, fv_prod_u, fv_l, fv_u, fv_tau, "uncertain_only")
    if fever_a.exists() and not fv_prod_a.exists():
        apply_cost_thresholds(fever_a, fv_prod_a, fv_l, fv_u, fv_tau, "always")

    report = Path(args.out_md)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Product Proof Cost Matrix")
    lines.append("")
    for track in ["scifact", "fever"]:
        lines.append(f"## {track}")
        lines.append("")
        lines.append("| file | accept_rate | reject_rate | uncertain_rate | stage2_rate | fp_accept_rate | fn_reject_rate | ok_rate_stage1 | mean_ms | p95_ms | rerank_ms_gt0 | nli_ms_gt0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in [
            f"{track}_baseline_uncertain_only_{args.n}_real.jsonl",
            f"{track}_baseline_always_{args.n}_real.jsonl",
            f"{track}_cost_uncertain_only_{args.n}.jsonl",
            f"{track}_cost_always_{args.n}.jsonl",
        ]:
            path = out_dir / name
            if not path.exists():
                lines.append(f"| {name} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
                continue
            s = summarize(list(iter_jsonl(path)))
            lines.append("| {f} | {a} | {r} | {u} | {s2} | {fp} | {fn} | {ok} | {mean} | {p95} | {rg} | {ng} |".format(
                f=name,
                a=format_float(s["accept_rate"]),
                r=format_float(s["reject_rate"]),
                u=format_float(s["uncertain_rate"]),
                s2=format_float(s["stage2_rate"]),
                fp=format_float(s["fp_accept_rate"]),
                fn=format_float(s["fn_reject_rate"]),
                ok=format_float(s["ok_rate_stage1"]),
                mean=f"{s['mean_ms']:.2f}",
                p95=f"{s['p95_ms']:.2f}",
                rg=s["rerank_ms_gt0"],
                ng=s["nli_ms_gt0"],
            ))
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
