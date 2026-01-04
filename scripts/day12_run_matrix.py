import argparse
import json
import os
import subprocess
from pathlib import Path


def write_sample(path, n):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"row_idx": i}) + "\n")
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["scifact", "fever"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--out_dir", default="runs/day12_matrix")
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--rerank_topk", type=int, default=20)
    ap.add_argument("--rerank_keep", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dry_run_stage2", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = args.sample
    if not sample_path:
        sample_path = out_dir / f"{args.track}_sample_{args.n}.jsonl"
        sample_path = write_sample(sample_path, args.n)

    base_cmd = [
        ".venv/bin/python",
        "scripts/day12_e2e_run_500.py",
        "--track",
        args.track,
        "--n",
        str(args.n),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--rerank_topk",
        str(args.rerank_topk),
        "--rerank_keep",
        str(args.rerank_keep),
        "--batch_size",
        str(args.batch_size),
        "--in_path",
        args.in_path,
        "--sample",
        str(sample_path),
    ]
    if args.track == "fever":
        base_cmd += ["--logit_col", "logit_platt", "--y_col", "y"]
    if args.dry_run_stage2:
        base_cmd.append("--dry_run_stage2")

    for baseline in args.baselines:
        for policy in args.policies:
            out_path = out_dir / f"{args.track}_{baseline}_{policy}_{args.n}.jsonl"
            cmd = list(base_cmd) + [
                "--baseline_mode",
                baseline,
                "--stage2_policy",
                policy,
                "--out",
                str(out_path),
            ]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
