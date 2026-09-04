"""Run math curriculum SFT followed by binary-correctness token-policy GRPO."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="artifacts/experiments/math_v2_seed42")
    parser.add_argument("--results-dir", default="artifacts/results/math_v2_seed42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sft-steps", type=int, default=12500)
    parser.add_argument("--synthetic-steps", type=int, default=1000)
    parser.add_argument("--grpo-steps", type=int, default=500)
    parser.add_argument("--resume-sft", help="Supervised checkpoint to continue from.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root_dir)
    supervised = root / "supervised"
    grpo = root / "grpo"
    sft_command = [
        sys.executable, "-m", "tiny_deepseek.training.math_sft",
        "--experiment-dir", str(supervised),
        "--device", args.device,
        "--max-steps", str(args.sft_steps),
        "--synthetic-steps", str(args.synthetic_steps),
        "--defer-readiness",
        "--seed", str(args.seed),
    ]
    if args.resume_sft:
        sft_command.extend(("--resume", args.resume_sft))
    subprocess.run(sft_command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m", "tiny_deepseek.cli.evaluate_math_readiness",
            "--checkpoint",
            str(supervised / "checkpoints" / "best_exact_match.pt"),
            "--experiment-dir",
            str(supervised),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    with (supervised / "summary.json").open(encoding="utf-8") as handle:
        supervised_summary = json.load(handle)
    if not supervised_summary.get("grpo_ready", False):
        print(
            "GRPO was not started: exact match, parse rate, pass@8, or mixed-group "
            "coverage did not meet the capability-first readiness gate."
        )
        return
    subprocess.run(
        [
            sys.executable, "-m", "tiny_deepseek.training.math_grpo",
            "--checkpoint", str(supervised / "checkpoints" / "best_exact_match.pt"),
            "--experiment-dir", str(grpo),
            "--device", args.device,
            "--max-steps", str(args.grpo_steps),
            "--seed", str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "tiny_deepseek.workflows.finalize_math_grpo",
            "--checkpoint", str(grpo / "checkpoints" / "latest.pt"),
            "--supervised-checkpoint", str(
                supervised / "checkpoints" / "best_exact_match.pt"
            ),
            "--supervised-dir", str(supervised),
            "--grpo-dir", str(grpo),
            "--results-dir", args.results_dir,
            "--device", args.device,
            "--seed", str(args.seed),
        ],
        check=True,
    )
    print(f"completed full math pipeline; report written to {Path(args.results_dir) / 'REPORT.md'}")


if __name__ == "__main__":
    main()
