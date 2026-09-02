"""Run math curriculum SFT followed by exact-answer token-policy GRPO."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="experiments/math_grpo_seed42")
    parser.add_argument("--results-dir", default="results/math_grpo_seed42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sft-steps", type=int, default=200)
    parser.add_argument("--synthetic-steps", type=int, default=100)
    parser.add_argument("--grpo-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root_dir)
    supervised = root / "supervised"
    grpo = root / "grpo"
    subprocess.run(
        [
            sys.executable, "train_math.py",
            "--experiment-dir", str(supervised),
            "--device", args.device,
            "--max-steps", str(args.sft_steps),
            "--synthetic-steps", str(args.synthetic_steps),
            "--seed", str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "train_math_grpo.py",
            "--checkpoint", str(supervised / "checkpoints" / "best_val_loss.pt"),
            "--experiment-dir", str(grpo),
            "--device", args.device,
            "--max-steps", str(args.grpo_steps),
            "--seed", str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "finalize_math_grpo.py",
            "--checkpoint", str(grpo / "checkpoints" / "latest.pt"),
            "--supervised-checkpoint", str(
                supervised / "checkpoints" / "best_val_loss.pt"
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
