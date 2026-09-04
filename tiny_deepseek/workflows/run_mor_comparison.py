"""Run the matched five-way Dense/SkipLayer/MoR comparison and all reporting."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resume_args(experiment: Path) -> tuple[bool, list[str]]:
    if (experiment / "summary.json").exists():
        print(f"skipping completed experiment: {experiment}", flush=True)
        return True, []
    latest = experiment / "checkpoints" / "latest.pt"
    return False, ["--resume", str(latest)] if latest.exists() else []


def ensure_run_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="artifacts/experiments/mor_comparison_seed42")
    parser.add_argument("--results-dir", default="artifacts/results/mor_comparison_seed42")
    parser.add_argument(
        "--skip-source",
        default=(
            "artifacts/experiments/paper_core_scaled_lr001_1000/"
            "paper_skip_L08_P050_Eff04_seed42"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--grpo-max-steps", type=int, default=300)
    parser.add_argument("--lambda-grpo", type=float, default=1.0)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.5)
    parser.add_argument("--exploration-epsilon", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--generation-tokens", type=int, default=50)
    args = parser.parse_args()
    clip_tag = f"{args.ppo_clip_epsilon:g}".replace(".", "p")

    experiments = Path(args.experiments_dir)
    experiments.mkdir(parents=True, exist_ok=True)
    common = [
        "--paper-reproduction",
        "--paper-learning-rate", "0.01",
        "--context-length", "128",
        "--d-model", "128",
        "--n-heads", "2",
        "--n-layers", "8",
        "--batch-size", "32",
        "--max-steps", str(args.max_steps),
        "--eval-interval", "200",
        "--eval-iters", str(args.eval_iters),
        "--log-interval", "20",
        "--seed", str(args.seed),
        "--device", args.device,
    ]

    dense_dir = experiments / f"full_dense_L08_seed{args.seed}"
    complete, resume = resume_args(dense_dir)
    if not complete:
        run(
            [
                sys.executable, "-m", "tiny_deepseek.training.supervised",
                "--model", "dense", "--experiment-dir", str(dense_dir),
                *common, *resume,
            ]
        )

    skip_source = Path(args.skip_source)
    if not (skip_source / "summary.json").exists():
        raise FileNotFoundError(f"completed SkipLayer source not found: {skip_source}")
    skip_link = experiments / f"skiplayer_L08_seed{args.seed}"
    ensure_run_link(skip_link, skip_source)
    skip_checkpoint = skip_source / "checkpoints" / "best_val_loss.pt"

    skip_grpo_dir = experiments / f"skiplayer_grpo_clip{clip_tag}_lam{args.lambda_grpo:g}_seed{args.seed}"
    complete, resume = resume_args(skip_grpo_dir)
    if not complete:
        source_options = ["--checkpoint", str(skip_checkpoint)] if not resume else []
        run(
            [
                sys.executable, "-m", "tiny_deepseek.training.paper_grpo",
                *source_options, *resume,
                "--experiment-dir", str(skip_grpo_dir),
                "--max-steps", str(args.grpo_max_steps),
                "--batch-size", "8",
                "--depth-budgets", "3", "4", "5", "8",
                "--exploration-epsilon", str(args.exploration_epsilon),
                "--lambda-compute-grpo", str(args.lambda_grpo),
                "--clip-epsilon", str(args.ppo_clip_epsilon),
                "--eval-interval", "50",
                "--eval-iters", str(args.eval_iters),
                "--seed", str(args.seed),
                "--device", args.device,
            ]
        )

    mor_dir = experiments / f"mor_middle_cycle_R3_seed{args.seed}"
    complete, resume = resume_args(mor_dir)
    if not complete:
        run(
            [
                sys.executable, "-m", "tiny_deepseek.training.supervised",
                "--model", "mor", "--router", "linear",
                "--recursion-steps", "3", "--recursion-block-layers", "2",
                "--mor-capacity-factors", "1", "0.6666666667", "0.3333333333",
                "--mor-router-alpha", "0.1",
                "--mor-aux-loss-coefficient", "0.001",
                "--mor-capacity-warmup-steps", "0",
                "--experiment-dir", str(mor_dir),
                *common, *resume,
            ]
        )
    mor_checkpoint = mor_dir / "checkpoints" / "best_val_loss.pt"

    mor_grpo_dir = experiments / f"mor_grpo_clip{clip_tag}_lam{args.lambda_grpo:g}_seed{args.seed}"
    complete, resume = resume_args(mor_grpo_dir)
    if not complete:
        source_options = ["--checkpoint", str(mor_checkpoint)] if not resume else []
        run(
            [
                sys.executable, "-m", "tiny_deepseek.training.mor_grpo",
                *source_options, *resume,
                "--experiment-dir", str(mor_grpo_dir),
                "--max-steps", str(args.grpo_max_steps),
                "--batch-size", "8",
                "--recursion-budgets", "1", "2", "3",
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
            sys.executable, "-m", "tiny_deepseek.cli.evaluate", "--all",
            "--experiments-dir", str(experiments),
            "--batch-size", "8", "--eval-iters", str(args.eval_iters),
            "--generation-tokens", str(args.generation_tokens),
            "--device", args.device,
        ]
    )
    run(
        [
            sys.executable, "-m", "tiny_deepseek.workflows.generate_report",
            "--experiments-dir", str(experiments),
            "--results-dir", args.results_dir,
        ]
    )


if __name__ == "__main__":
    main()
