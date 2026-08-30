"""Evaluate a saved checkpoint on the Tiny Shakespeare validation split."""

from __future__ import annotations

import argparse
import json

from config import TrainConfig
from data import TinyShakespeareData
from train import evaluate
from utils import load_checkpoint, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = TinyShakespeareData(args.data_path, model.config.context_length)
    if dataset.stoi != checkpoint["stoi"]:
        raise ValueError("Checkpoint and dataset vocabularies differ")
    saved = checkpoint.get("train_config") or {}
    cfg = TrainConfig(
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        compute_loss=saved.get("compute_loss", "linear"),
        target_compute=saved.get("target_compute", 0.5),
    )
    current_lambda = saved.get("lambda_compute", 0.0) if model.config.model_type == "dynamic" else 0.0
    metrics = evaluate(model, dataset, cfg, device, current_lambda)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
