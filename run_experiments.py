"""Run reproducible core or full multi-seed sparse-depth experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def completed(path: Path, force: bool) -> bool:
    if not force and (path / "summary.json").exists():
        print(f"skipping completed experiment: {path}", flush=True)
        return True
    return False


def resume_arguments(path: Path, force: bool) -> list[str]:
    latest = path / "checkpoints" / "latest.pt"
    if latest.exists() and not force:
        print(f"resuming incomplete experiment: {path}", flush=True)
        return ["--resume", str(latest)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["core", "full", "supervised"], default="core")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--densities", type=float, nargs="+", default=None)
    parser.add_argument("--lambda-densities", type=float, nargs="+", default=None)
    parser.add_argument("--grpo-lambdas", type=float, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--grpo-max-steps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args, extra_train_args = parser.parse_known_args()
    root = Path(__file__).parent
    experiments = Path(args.experiments_dir)
    densities = args.densities or ([0.75, 0.5, 0.25] if args.preset == "full" else [0.5])
    lambda_densities = args.lambda_densities or ([0.03, 0.1, 0.3] if args.preset == "full" else [0.1])
    grpo_lambdas = args.grpo_lambdas or ([0.01, 0.05, 0.1, 0.2] if args.preset == "full" else [0.1])

    for seed in args.seeds:
        dense_dir = experiments / f"dense_seed{seed}"
        if not completed(dense_dir, args.force):
            resume = resume_arguments(dense_dir, args.force)
            run([
                sys.executable, str(root / "train.py"), "--model", "dense",
                "--seed", str(seed), "--max-steps", str(args.max_steps),
                "--device", args.device, "--experiment-dir", str(dense_dir),
                *resume, *extra_train_args,
            ])
        for density in densities:
            density_label = f"P{int(round(100 * density)):03d}"
            for lambda_density in lambda_densities:
                lambda_label = f"lam{lambda_density:g}"
                for router in ("linear", "gru"):
                    experiment = experiments / f"{router}_{density_label}_{lambda_label}_seed{seed}"
                    if not completed(experiment, args.force):
                        resume = resume_arguments(experiment, args.force)
                        run([
                            sys.executable, str(root / "train.py"), "--model", "sparse",
                            "--router", router, "--target-density", str(density),
                            "--lambda-density", str(lambda_density), "--seed", str(seed),
                            "--max-steps", str(args.max_steps), "--device", args.device,
                            "--experiment-dir", str(experiment), *resume, *extra_train_args,
                        ])
                if args.preset == "supervised":
                    continue
                supervised = experiments / f"gru_{density_label}_{lambda_label}_seed{seed}"
                checkpoint = supervised / "checkpoints" / "best_val_loss.pt"
                for grpo_lambda in grpo_lambdas:
                    grpo_dir = experiments / f"gru_grpo_{density_label}_{lambda_label}_rl{grpo_lambda:g}_seed{seed}"
                    if completed(grpo_dir, args.force):
                        continue
                    resume = resume_arguments(grpo_dir, args.force)
                    run([
                        sys.executable, str(root / "train_grpo.py"), "--checkpoint", str(checkpoint),
                        "--lambda-compute-grpo", str(grpo_lambda), "--seed", str(seed),
                        "--max-steps", str(args.grpo_max_steps), "--device", args.device,
                        "--experiment-dir", str(grpo_dir), "--grpo-router-only", *resume,
                    ])

    run([
        sys.executable, str(root / "evaluate.py"), "--all",
        "--experiments-dir", str(experiments), "--device", args.device,
    ])
    run([
        sys.executable, str(root / "generate_report.py"),
        "--experiments-dir", str(experiments), "--results-dir", args.results_dir,
    ])


if __name__ == "__main__":
    main()
