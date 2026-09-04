"""Run the bounded Shakespeare SkipLayer+GRPO+MoE+MTP experiment."""

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
        "--source-checkpoint",
        default=(
            "artifacts/experiments/mor_comparison_seed42/"
            "skiplayer_grpo_clip0p5_lam1_seed42/checkpoints/best_quality_compute.pt"
        ),
    )
    parser.add_argument("--experiment-root")
    parser.add_argument("--attention", choices=["mla", "mha"], default="mla")
    parser.add_argument("--supervised-steps", type=int, default=250)
    parser.add_argument("--grpo-steps", type=int, default=120)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiment_root = Path(
        args.experiment_root
        or f"artifacts/experiments/skiplayer_moe_mtp_{'mla_rope' if args.attention == 'mla' else 'mha'}_seed{args.seed}"
    )
    supervised = experiment_root / "supervised"
    grpo = experiment_root / "grpo"
    print(
        "Live TensorBoard:\n"
        f"  tensorboard --logdir {experiment_root.resolve()} --port 6006\n"
        "  then open http://localhost:6006",
        flush=True,
    )
    if not (supervised / "summary.json").exists():
        run(
            [
                sys.executable, "-m", "tiny_deepseek.training.moe_mtp",
                "--source-checkpoint", args.source_checkpoint,
                "--experiment-dir", str(supervised),
                "--num-experts", "10", "--top-k", "2",
                "--attention", args.attention,
                "--max-steps", str(args.supervised_steps),
                "--batch-size", "16", "--eval-interval", "50",
                "--eval-iters", str(args.eval_iters), "--log-interval", "10",
                "--device", args.device, "--seed", str(args.seed),
            ]
        )
    supervised_checkpoint = supervised / "checkpoints" / "best_val_loss.pt"
    if not (grpo / "summary.json").exists():
        run(
            [
                sys.executable, "-m", "tiny_deepseek.training.paper_grpo",
                "--checkpoint", str(supervised_checkpoint),
                "--experiment-dir", str(grpo),
                "--max-steps", str(args.grpo_steps), "--batch-size", "8",
                "--depth-budgets", "3", "4", "5", "8",
                "--exploration-epsilon", "0.8",
                "--lambda-compute-grpo", "1.0", "--clip-epsilon", "0.5",
                "--eval-interval", "40", "--eval-iters", str(args.eval_iters),
                "--log-interval", "10", "--device", args.device,
                "--seed", str(args.seed),
            ]
        )
    run(
        [
            sys.executable, "-m", "tiny_deepseek.cli.evaluate",
            "--checkpoint", str(grpo / "checkpoints" / "best_quality_compute.pt"),
            "--batch-size", "8", "--eval-iters", str(args.eval_iters),
            "--generation-tokens", "50", "--device", args.device,
        ]
    )
    print(f"completed bounded experiment in {experiment_root}", flush=True)


if __name__ == "__main__":
    main()
