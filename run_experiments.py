"""Run the dense baseline and the requested dynamic-depth lambda sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/sweep")
    parser.add_argument("--router", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--skip-dense", action="store_true")
    args, train_args = parser.parse_known_args()

    script = str(Path(__file__).with_name("train.py"))
    runs_dir = Path(args.runs_dir)
    commands = []
    if not args.skip_dense:
        commands.append(
            [sys.executable, script, "--model", "dense", "--output-dir", str(runs_dir / "dense")]
        )
    for value in args.lambdas:
        name = f"dynamic_{args.router}_lambda_{value:g}"
        commands.append(
            [
                sys.executable,
                script,
                "--model",
                "dynamic",
                "--router",
                args.router,
                "--lambda-compute",
                str(value),
                "--output-dir",
                str(runs_dir / name),
            ]
        )
    for command in commands:
        full_command = command + train_args
        print("running:", " ".join(full_command), flush=True)
        subprocess.run(full_command, check=True)

    aggregate = str(Path(__file__).with_name("aggregate_results.py"))
    subprocess.run(
        [
            sys.executable,
            aggregate,
            "--runs-dir",
            str(runs_dir),
            "--csv",
            str(runs_dir / "results.csv"),
            "--plot",
            str(runs_dir / "pareto.png"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
