"""Benchmark Stage A generation speed and theoretical routing for checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from generate import generate_tokens
from utils import load_checkpoint, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--warmup-tokens", type=int, default=5)
    parser.add_argument("--output", default="benchmark.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    rows = []
    for checkpoint_path in args.checkpoints:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        stoi = checkpoint["stoi"]
        unknown = sorted(set(args.prompt) - stoi.keys())
        if unknown:
            raise ValueError(f"Prompt contains unknown characters for {checkpoint_path}: {unknown!r}")
        ids = torch.tensor([[stoi[ch] for ch in args.prompt]], dtype=torch.long, device=device)
        if args.warmup_tokens:
            generate_tokens(model, ids.clone(), args.warmup_tokens, temperature=0.0, top_k=None)
        _, gates, elapsed = generate_tokens(model, ids, args.tokens, temperature=0.0, top_k=None)
        compute_fraction = gates.mean().item() if gates is not None else 1.0
        rows.append(
            {
                "checkpoint": checkpoint_path,
                "model": model.config.model_type,
                "router": model.config.router_type if model.config.model_type == "sparse" else "none",
                "tokens_per_second": args.tokens / max(elapsed, 1e-9),
                "layers_per_token": compute_fraction * model.config.n_layers,
                "compute_fraction": compute_fraction,
                "skip_fraction": 1.0 - compute_fraction,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['model']:7s} {row['router']:4s} | {row['tokens_per_second']:.2f} tok/s "
            f"| depth {row['layers_per_token']:.2f} | theoretical skip {100*row['skip_fraction']:.1f}%"
        )
    print(f"saved {output}")
    print("Stage A evaluates every block, so speed numbers do not represent sparse execution.")


if __name__ == "__main__":
    main()
