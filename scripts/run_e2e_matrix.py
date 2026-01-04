import argparse
import json
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
    ap.add_argument("--tracks", nargs="+", default=["scifact", "fever"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--scifact_in_path", default=None)
    ap.add_argument("--fever_in_path", default=None)
    ap.add_argument("--out_dir", default="runs/matrix")
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--rerank_topk", type=int, default=20)
    ap.add_argument("--rerank_keep", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dry_run_stage2", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for track in args.tracks:
        in_path = args.scifact_in_path if track == "scifact" else args.fever_in_path
        if not in_path:
            raise SystemExit(f"missing in_path for track={track}")
        sample_path = out_dir / f"{track}_sample_{args.n}.jsonl"
        if not sample_path.exists():
            write_sample(sample_path, args.n)
        base_cmd = [
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
            "--rerank_topk",
            str(args.rerank_topk),
            "--rerank_keep",
            str(args.rerank_keep),
            "--batch_size",
            str(args.batch_size),
            "--in_path",
            in_path,
            "--sample",
            str(sample_path),
        ]
        if track == "fever":
            base_cmd += ["--logit_col", "logit_platt", "--y_col", "y"]
        if args.dry_run_stage2:
            base_cmd.append("--dry_run_stage2")
        for baseline in args.baselines:
            for policy in args.policies:
                out_path = out_dir / f"{track}_{baseline}_{policy}_{args.n}.jsonl"
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
