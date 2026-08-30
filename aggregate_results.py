"""Aggregate real experiment summaries into CSV and a Pareto plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLUMNS = [
    "model",
    "router",
    "lambda",
    "val_loss",
    "val_perplexity",
    "ppl",
    "parameters",
    "layers_per_token",
    "compute_fraction",
    "skip_fraction",
    "dense_block_flops_per_sequence",
    "estimated_executed_block_flops_per_sequence",
    "training_tokens_per_second",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--plot", default="pareto.png")
    args = parser.parse_args()

    summaries = []
    for path in sorted(Path(args.runs_dir).glob("**/summary.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        values.setdefault("ppl", values.get("val_perplexity"))
        values["source"] = str(path)
        summaries.append(values)
    if not summaries:
        raise SystemExit(f"No summary.json files found below {args.runs_dir!r}; refusing to fabricate results")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + ["source"], extrasaction="ignore")
        writer.writeheader()
        for values in summaries:
            writer.writerow(values)

    fig, axis = plt.subplots(figsize=(7, 5))
    for values in summaries:
        label = values["model"]
        if values["model"] == "dynamic":
            label += f" {values.get('router', '')} λ={values.get('lambda', 0):g}"
        axis.scatter(values["layers_per_token"], values["val_perplexity"], s=55)
        axis.annotate(label, (values["layers_per_token"], values["val_perplexity"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Average executed layers per token (theoretical)")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Dynamic-depth quality/compute tradeoff")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"wrote {csv_path} and {plot_path} from {len(summaries)} actual run(s)")


if __name__ == "__main__":
    main()
