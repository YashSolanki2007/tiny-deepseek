"""Run matched-effective-depth experiments following the SkipLayer paper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resume_or_skip(experiment: Path, force: bool) -> tuple[bool, list[str]]:
    if (experiment / "summary.json").exists() and not force:
        print(f"skipping completed experiment: {experiment}", flush=True)
        return True, []
    latest = experiment / "checkpoints" / "latest.pt"
    if latest.exists() and not force:
        return False, ["--resume", str(latest)]
    return False, []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "core", "table1"], default="core")
    parser.add_argument("--experiments-dir", default="artifacts/experiments/paper_core")
    parser.add_argument("--results-dir", default="artifacts/results/paper_core")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--effective-layers", type=int)
    parser.add_argument("--densities", type=float, nargs="+")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args, extra = parser.parse_known_args()

    if args.preset == "smoke":
        effective_layers = args.effective_layers or 2
        densities = args.densities or [0.5]
        max_steps = args.max_steps or 20
        architecture = [
            "--context-length", "16", "--d-model", "32", "--n-heads", "1",
            "--batch-size", "4", "--eval-interval", "10", "--eval-iters", "2",
            "--log-interval", "2",
        ]
        evaluation = ["--batch-size", "4", "--eval-iters", "2", "--generation-tokens", "8"]
    elif args.preset == "table1":
        effective_layers = args.effective_layers or 6
        densities = args.densities or [0.5, 0.25, 0.125]
        max_steps = args.max_steps or 5000
        architecture = [
            "--context-length", "128", "--d-model", "128", "--n-heads", "2",
            "--batch-size", "32", "--eval-interval", "250", "--eval-iters", "50",
            "--log-interval", "20",
        ]
        evaluation = ["--batch-size", "32", "--eval-iters", "50", "--generation-tokens", "100"]
    else:
        effective_layers = args.effective_layers or 4
        densities = args.densities or [0.5]
        max_steps = args.max_steps or 5000
        architecture = [
            "--context-length", "128", "--d-model", "128", "--n-heads", "2",
            "--batch-size", "32", "--eval-interval", "250", "--eval-iters", "50",
            "--log-interval", "20",
        ]
        evaluation = ["--batch-size", "32", "--eval-iters", "50", "--generation-tokens", "100"]

    experiments = Path(args.experiments_dir)
    for density in densities:
        total_layers = effective_layers / density
        if not total_layers.is_integer():
            raise ValueError(
                f"effective_layers / density must be integral, got {total_layers}"
            )

    for seed in args.seeds:
        dense_dir = experiments / f"paper_dense_L{effective_layers:02d}_seed{seed}"
        skip, resume = resume_or_skip(dense_dir, args.force)
        if not skip:
            run([
                sys.executable, "-m", "tiny_deepseek.training.supervised", "--paper-reproduction",
                "--model", "dense", "--n-layers", str(effective_layers),
                "--seed", str(seed), "--max-steps", str(max_steps),
                "--device", args.device, "--experiment-dir", str(dense_dir),
                *architecture, *resume, *extra,
            ])
        for density in densities:
            layers = int(effective_layers / density)
            density_label = f"P{int(round(100 * density)):03d}"
            sparse_dir = experiments / (
                f"paper_skip_L{layers:02d}_{density_label}_Eff{effective_layers:02d}_seed{seed}"
            )
            skip, resume = resume_or_skip(sparse_dir, args.force)
            if not skip:
                run([
                    sys.executable, "-m", "tiny_deepseek.training.supervised", "--paper-reproduction",
                    "--model", "sparse", "--router", "linear",
                    "--n-layers", str(layers), "--target-density", str(density),
                    "--seed", str(seed), "--max-steps", str(max_steps),
                    "--device", args.device, "--experiment-dir", str(sparse_dir),
                    *architecture, *resume, *extra,
                ])

    run([
        sys.executable, "-m", "tiny_deepseek.cli.evaluate", "--all",
        "--experiments-dir", str(experiments), "--device", args.device, *evaluation,
    ])
    run([
        sys.executable, "-m", "tiny_deepseek.workflows.generate_report",
        "--experiments-dir", str(experiments), "--results-dir", args.results_dir,
    ])


if __name__ == "__main__":
    main()
