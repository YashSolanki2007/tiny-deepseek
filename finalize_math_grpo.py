"""Finalize evaluation/reporting for a completed math-GRPO checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from math_data import MathData
from math_training_utils import evaluate_math_answers, evaluate_math_model
from model import SparseMoEMTPTransformer
from utils import load_checkpoint, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="experiments/math_grpo_seed42/grpo/checkpoints/latest.pt"
    )
    parser.add_argument(
        "--supervised-checkpoint",
        default="experiments/math_grpo_seed42/supervised/checkpoints/best_val_loss.pt",
    )
    parser.add_argument("--supervised-dir", default="experiments/math_grpo_seed42/supervised")
    parser.add_argument("--grpo-dir", default="experiments/math_grpo_seed42/grpo")
    parser.add_argument("--results-dir", default="results/math_grpo_seed42")
    parser.add_argument("--data-dir", default="data/gsm8k")
    parser.add_argument("--answer-examples", type=int, default=6)
    parser.add_argument("--answer-tokens", type=int, default=48)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = select_device(args.device)
    model, grpo_checkpoint = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, SparseMoEMTPTransformer):
        raise ValueError("expected the trained math MLA+RoPE MoE+MTP model")
    data = MathData(args.data_dir, model.config.context_length, seed=args.seed)
    validation = evaluate_math_model(model, data, device, batch_size=2, eval_iters=4)
    answers, samples = evaluate_math_answers(
        model, data, device, split="test", count=args.answer_examples,
        max_new_tokens=args.answer_tokens, seed=args.seed + 1,
    )
    grpo_summary = {
        "stage": "math_quality_grpo",
        **validation,
        **answers,
        "samples": samples,
        "checkpoint": args.checkpoint,
    }
    grpo_dir = Path(args.grpo_dir)
    write_json(grpo_dir / "summary.json", grpo_summary)
    write_json(grpo_dir / "samples.json", samples)
    with SummaryWriter(str(grpo_dir / "tensorboard")) as writer:
        for key, value in {**validation, **answers}.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"final/{key}", value, int(grpo_checkpoint["step"]))

    with (Path(args.supervised_dir) / "summary.json").open(encoding="utf-8") as handle:
        supervised_training = json.load(handle)
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    supervised_model, supervised_checkpoint = load_checkpoint(
        args.supervised_checkpoint, device
    )
    supervised_validation = evaluate_math_model(
        supervised_model, data, device, batch_size=2, eval_iters=4
    )
    supervised_answers, supervised_samples = evaluate_math_answers(
        supervised_model, data, device, split="test", count=args.answer_examples,
        max_new_tokens=args.answer_tokens, seed=args.seed + 1,
    )
    supervised = {
        **supervised_training,
        **supervised_validation,
        **supervised_answers,
        "samples": supervised_samples,
        "checkpoint": args.supervised_checkpoint,
    }
    write_json(Path(args.supervised_dir) / "matched_evaluation.json", supervised)
    with SummaryWriter(str(Path(args.supervised_dir) / "tensorboard")) as writer:
        for key, value in {**supervised_validation, **supervised_answers}.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(
                    f"final/{key}", value, int(supervised_checkpoint["step"])
                )
    result = {
        "experiment": "larger MLA+RoPE SkipLayer+MoE+MTP math model with quality GRPO",
        "supervised": supervised,
        "grpo": grpo_summary,
    }
    results_dir = Path(args.results_dir)
    write_json(results_dir / "results.json", result)
    lines = [
        "# Math GRPO result", "",
        "| Stage | Validation CE | Byte accuracy | Exact answer | Parse rate | Layers/token | FLOPs vs dense |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (("Supervised", supervised), ("Quality GRPO", grpo_summary)):
        lines.append(
            f"| {name} | {values.get('val_loss', float('nan')):.4f} "
            f"| {100 * values.get('val_accuracy', 0):.2f}% "
            f"| {100 * values.get('answer_exact_match', 0):.2f}% "
            f"| {100 * values.get('answer_parse_rate', 0):.2f}% "
            f"| {values.get('layers_per_token', float('nan')):.3f} "
            f"| {values.get('estimated_flops_vs_full_dense', float('nan')):.3f}x |"
        )
    supervised_by_question = {
        sample["question"]: sample for sample in supervised_samples
    }
    lines.extend(["", "## Matched held-out samples", ""])
    for sample in samples:
        before = supervised_by_question[sample["question"]]
        lines.extend(
            [
                f"### {sample['question']}", "",
                f"Gold: `{sample['gold_answer']}`; supervised prediction: "
                f"`{before['prediction']}`; GRPO prediction: `{sample['prediction']}`", "",
                "Supervised:", "", "```text", before["completion"], "```", "",
                "Quality GRPO:", "", "```text", sample["completion"], "```", "",
            ]
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"finalized math report at {results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
