"""Aggregate results, generate plots, and build both reports."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="artifacts/experiments")
    parser.add_argument("--results-dir", default="artifacts/results")
    args = parser.parse_args()
    shared = ["--experiments-dir", args.experiments_dir, "--results-dir", args.results_dir]
    for module in (
        "tiny_deepseek.evaluation.aggregation",
        "tiny_deepseek.evaluation.plots",
        "tiny_deepseek.evaluation.report",
    ):
        subprocess.run([sys.executable, "-m", module, *shared], check=True)


if __name__ == "__main__":
    main()
