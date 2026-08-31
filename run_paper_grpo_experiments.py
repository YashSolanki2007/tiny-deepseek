"""Run a compute-penalty sweep for budget-guided GRPO and build all reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/paper_core_scaled_lr001_1000/"
            "paper_skip_L08_P050_Eff04_seed42/checkpoints/best_val_loss.pt"
        ),
    )
    parser.add_argument(
        "--experiments-dir", default="experiments/paper_core_scaled_lr001_1000"
    )
    parser.add_argument("--results-dir", default="results/paper_grpo_seed42")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 1.0, 2.0])
    parser.add_argument("--depth-budgets", type=int, nargs="+", default=[3, 4, 5, 8])
    parser.add_argument("--exploration-epsilon", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--generation-tokens", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).parent
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    for lam in args.lambdas:
        name = f"paper_skip_L08_P050_Eff04_seed{args.seed}_budget_grpo_lam{lam:g}"
        command = [
            sys.executable,
            str(root / "train_paper_grpo.py"),
            "--checkpoint", args.checkpoint,
            "--experiment-dir", str(experiments_dir / name),
            "--lambda-compute-grpo", str(lam),
            "--exploration-epsilon", str(args.exploration_epsilon),
            "--max-steps", str(args.max_steps),
            "--batch-size", str(args.batch_size),
            "--policy-epochs", str(args.policy_epochs),
            "--learning-rate", str(args.learning_rate),
            "--beta-kl", str(args.beta_kl),
            "--clip-epsilon", str(args.clip_epsilon),
            "--eval-interval", str(args.eval_interval),
            "--eval-iters", str(args.eval_iters),
            "--device", args.device,
            "--seed", str(args.seed),
            "--depth-budgets", *[str(value) for value in args.depth_budgets],
        ]
        run(command)

    run(
        [
            sys.executable,
            str(root / "evaluate.py"),
            "--all",
            "--experiments-dir", str(experiments_dir),
            "--batch-size", str(args.batch_size),
            "--eval-iters", str(args.eval_iters),
            "--generation-tokens", str(args.generation_tokens),
            "--device", args.device,
        ]
    )
    run(
        [
            sys.executable,
            str(root / "generate_report.py"),
            "--experiments-dir", str(experiments_dir),
            "--results-dir", args.results_dir,
        ]
    )


if __name__ == "__main__":
    main()
