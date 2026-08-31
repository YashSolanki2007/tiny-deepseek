"""Aggregate results, generate plots, and build both reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    root = Path(__file__).parent
    shared = ["--experiments-dir", args.experiments_dir, "--results-dir", args.results_dir]
    for script in ("aggregate_results.py", "plots.py", "report.py"):
        subprocess.run([sys.executable, str(root / script), *shared], check=True)


if __name__ == "__main__":
    main()
