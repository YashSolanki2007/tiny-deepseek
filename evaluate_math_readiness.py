"""Evaluate a supervised math checkpoint for binary-GRPO readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from math_data import MathData
from math_training_utils import (
    evaluate_math_answers,
    evaluate_math_model,
    evaluate_math_pass_at_k,
)
from model import SparseMoEMTPTransformer
from utils import load_checkpoint, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="experiments/math_v2_seed42/supervised/checkpoints/best_exact_match.pt",
    )
    parser.add_argument("--experiment-dir", default="experiments/math_v2_seed42/supervised")
    parser.add_argument("--data-dir", default="data/gsm8k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--answer-examples", type=int, default=64)
    parser.add_argument("--pass-examples", type=int, default=200)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minimum-exact-rate", type=float, default=0.05)
    parser.add_argument("--minimum-parse-rate", type=float, default=0.95)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.20)
    parser.add_argument("--minimum-mixed-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, SparseMoEMTPTransformer):
        raise ValueError("expected an MLA+RoPE MoE+MTP math checkpoint")
    dataset_config = checkpoint.get("training_config", {}).get("dataset", {})
    data = MathData(
        args.data_dir,
        model.config.context_length,
        seed=args.seed,
        tokenizer_type=dataset_config.get("tokenizer_type", "byte"),
        bpe_vocab_size=int(dataset_config.get("bpe_vocab_size", 4096)),
    )
    if checkpoint["stoi"] != data.stoi:
        raise ValueError("checkpoint tokenizer does not match the math data tokenizer")

    validation = evaluate_math_model(
        model,
        data,
        device,
        batch_size=4,
        eval_iters=args.eval_iters,
        force_full_depth=True,
    )
    answers, samples = evaluate_math_answers(
        model,
        data,
        device,
        split="validation",
        count=args.answer_examples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        force_full_depth=True,
    )
    pass_metrics, pass_samples = evaluate_math_pass_at_k(
        model,
        data,
        device,
        split="validation",
        count=args.pass_examples,
        k=args.pass_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed + 10_000,
        force_full_depth=True,
    )
    ready = (
        answers["answer_exact_match"] >= args.minimum_exact_rate
        and answers["answer_parse_rate"] >= args.minimum_parse_rate
        and pass_metrics[f"pass_at_{args.pass_k}"] >= args.minimum_pass_rate
        and pass_metrics["mixed_outcome_group_rate"] >= args.minimum_mixed_rate
    )
    summary = {
        "stage": "math_supervised_grpo_readiness",
        "parameter_count": model.parameter_count(),
        **validation,
        **answers,
        **pass_metrics,
        "grpo_ready": ready,
        "grpo_readiness_thresholds": {
            "exact_rate": args.minimum_exact_rate,
            "parse_rate": args.minimum_parse_rate,
            "pass_rate": args.minimum_pass_rate,
            "mixed_outcome_group_rate": args.minimum_mixed_rate,
        },
        "samples": samples,
        "pass_at_k_samples": pass_samples,
        "checkpoint": args.checkpoint,
    }
    experiment_dir = Path(args.experiment_dir)
    write_json(experiment_dir / "summary.json", summary)
    write_json(experiment_dir / "samples.json", samples)
    write_json(experiment_dir / "pass_at_k_samples.json", pass_samples)
    with SummaryWriter(str(experiment_dir / "tensorboard")) as writer:
        for key, value in {**validation, **answers, **pass_metrics}.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"readiness/{key}", value, int(checkpoint["step"]))
        writer.add_scalar("readiness/grpo_ready", float(ready), int(checkpoint["step"]))
    print(
        f"readiness complete | pass@{args.pass_k} "
        f"{pass_metrics[f'pass_at_{args.pass_k}']:.3f} | mixed "
        f"{pass_metrics['mixed_outcome_group_rate']:.3f} | exact "
        f"{answers['answer_exact_match']:.3f} | parse "
        f"{answers['answer_parse_rate']:.3f} | GRPO ready={ready}"
    )


if __name__ == "__main__":
    main()
