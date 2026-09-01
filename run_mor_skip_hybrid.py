"""Run supervised inner skipping, hybrid GRPO, evaluation, and reporting."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resume_or_skip(experiment: Path) -> tuple[bool, list[str]]:
    if (experiment / "summary.json").exists():
        print(f"skipping completed experiment: {experiment}", flush=True)
        return True, []
    latest = experiment / "checkpoints/latest.pt"
    return False, ["--resume", str(latest)] if latest.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mor-checkpoint",
        default=(
            "experiments/mor_comparison_seed42/mor_middle_cycle_R3_seed42/"
            "checkpoints/best_val_loss.pt"
        ),
    )
    parser.add_argument(
        "--experiments-dir", default="experiments/mor_comparison_seed42"
    )
    parser.add_argument("--results-dir", default="results/mor_comparison_seed42")
    parser.add_argument("--supervised-max-steps", type=int, default=1000)
    parser.add_argument("--grpo-max-steps", type=int, default=300)
    parser.add_argument("--target-skip-density", type=float, default=0.5)
    parser.add_argument("--lambda-skip-density", type=float, default=0.1)
    parser.add_argument("--lambda-grpo", type=float, default=1.0)
    parser.add_argument("--exploration-epsilon", type=float, default=0.8)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.5)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--generation-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).parent
    experiments = Path(args.experiments_dir)
    experiments.mkdir(parents=True, exist_ok=True)
    supervised = experiments / (
        f"mor_inner_skiplayer_P{int(round(100 * args.target_skip_density)):03d}_"
        f"lam{args.lambda_skip_density:g}_seed{args.seed}"
    )
    complete, resume = resume_or_skip(supervised)
    if not complete:
        source = ["--checkpoint", args.mor_checkpoint] if not resume else []
        run(
            [
                sys.executable, str(root / "train_mor_skip.py"), *source, *resume,
                "--experiment-dir", str(supervised),
                "--max-steps", str(args.supervised_max_steps),
                "--batch-size", "32",
                "--target-skip-density", str(args.target_skip_density),
                "--lambda-skip-density", str(args.lambda_skip_density),
                "--eval-interval", "100",
                "--eval-iters", str(args.eval_iters),
                "--seed", str(args.seed),
                "--device", args.device,
            ]
        )

    supervised_checkpoint = supervised / "checkpoints/best_quality_compute.pt"
    grpo = experiments / (
        f"mor_inner_skiplayer_grpo_lam{args.lambda_grpo:g}_seed{args.seed}"
    )
    complete, resume = resume_or_skip(grpo)
    if not complete:
        source = ["--checkpoint", str(supervised_checkpoint)] if not resume else []
        run(
            [
                sys.executable, str(root / "train_mor_skip_grpo.py"), *source, *resume,
                "--experiment-dir", str(grpo),
                "--max-steps", str(args.grpo_max_steps),
                "--batch-size", "8",
                "--skip-density-budgets", "0.25", "0.5", "0.75", "1.0",
                "--exploration-epsilon", str(args.exploration_epsilon),
                "--lambda-compute-grpo", str(args.lambda_grpo),
                "--clip-epsilon", str(args.ppo_clip_epsilon),
                "--eval-interval", "50",
                "--eval-iters", str(args.eval_iters),
                "--seed", str(args.seed),
                "--device", args.device,
            ]
        )

    run(
        [
            sys.executable, str(root / "evaluate.py"), "--all",
            "--experiments-dir", str(experiments),
            "--batch-size", "8", "--eval-iters", str(args.eval_iters),
            "--generation-tokens", str(args.generation_tokens),
            "--device", args.device,
        ]
    )
    run(
        [
            sys.executable, str(root / "generate_report.py"),
            "--experiments-dir", str(experiments),
            "--results-dir", args.results_dir,
        ]
    )


if __name__ == "__main__":
    main()
