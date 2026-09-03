"""Train the larger MLA+RoPE SkipLayer+MoE+MTP model on arithmetic and GSM8K."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from config import ModelConfig
from logging_utils import StructuredLogger
from losses import density_loss, scheduled_coefficient
from math_data import MathData
from math_training_utils import (
    evaluate_math_answers,
    evaluate_math_model,
    evaluate_math_pass_at_k,
)
from model import SparseMoEMTPTransformer, build_model
from utils import (
    gradient_norm,
    load_checkpoint,
    routing_metrics,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    top1_accuracy,
    write_json,
)


FIELDS = [
    "step", "split", "stage", "train_loss", "train_accuracy", "mtp_loss",
    "mtp_accuracy", "density_loss", "density_coefficient", "moe_aux_loss",
    "moe_router_entropy", "total_loss", "learning_rate", "gradient_norm",
    "seconds_per_step", "tokens_per_second", "layers_per_token",
    "compute_fraction", "skip_fraction", "routing_entropy",
    "expert_utilization_cv", "val_loss", "val_perplexity", "val_accuracy",
    "estimated_flops_vs_full_dense", "answer_exact_match", "answer_parse_rate",
    "generation_layers_per_token", "supervised_tokens", "complete_example_fraction",
    "pass_at_8", "sample_exact_match", "sample_parse_rate",
    "mixed_outcome_group_rate", "grpo_ready",
    "answer_repetition_rate", "arithmetic_operation_accuracy",
    "difficulty_easy_exact_match", "difficulty_medium_exact_match",
    "difficulty_hard_exact_match", "operation_addition_exact_match",
    "operation_subtraction_exact_match", "operation_multiplication_exact_match",
    "operation_division_exact_match", "effective_batch_size",
    "gsm_batch_fraction", "synthetic_batch_fraction",
]


def curriculum_source(
    step: int, synthetic_steps: int, seed: int, micro_step: int = 0
) -> str:
    """Deterministically move from easy arithmetic to 90% GSM8K batches."""
    if step < synthetic_steps:
        progress = step / max(synthetic_steps, 1)
        difficulty = "easy" if progress < 1 / 3 else "medium" if progress < 2 / 3 else "hard"
        return f"synthetic_{difficulty}"
    rng = random.Random(seed + step * 10_007 + micro_step * 101)
    if rng.random() >= 0.10:
        return "gsm_train"
    return "synthetic_hard"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiments/math_v2_seed42/supervised")
    parser.add_argument("--data-dir", default="data/gsm8k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--synthetic-steps", type=int, default=1000)
    parser.add_argument(
        "--full-depth-steps",
        type=int,
        default=-1,
        help="Number of initial always-execute steps; negative means all SFT steps.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--max-microbatch-tokens",
        type=int,
        default=4096,
        help="Split long length buckets while preserving the effective batch size.",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--tokenizer-type", choices=("byte", "bpe"), default="bpe")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume-learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--resume", help="Checkpoint to continue from.")
    parser.add_argument("--target-density", type=float, default=0.70)
    parser.add_argument("--lambda-density", type=float, default=0.10)
    parser.add_argument("--mtp-weight", type=float, default=0.30)
    parser.add_argument("--moe-aux-weight", type=float, default=0.0001)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--answer-eval-examples", type=int, default=32)
    parser.add_argument("--answer-eval-interval", type=int, default=500)
    parser.add_argument("--pass-monitor-examples", type=int, default=16)
    parser.add_argument("--pass-monitor-interval", type=int, default=500)
    parser.add_argument("--answer-tokens", type=int, default=128)
    parser.add_argument("--pass-eval-examples", type=int, default=200)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--pass-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-ready-pass-rate", type=float, default=0.10)
    parser.add_argument("--grpo-ready-mixed-rate", type=float, default=0.10)
    parser.add_argument(
        "--defer-readiness",
        action="store_true",
        help="Leave answer/pass@k evaluation to evaluate_math_readiness.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.max_microbatch_tokens < 1:
        raise ValueError("max-microbatch-tokens must be positive")
    if args.answer_eval_interval < 1 or args.pass_monitor_interval < 1:
        raise ValueError("answer/pass monitoring intervals must be positive")
    if args.answer_eval_interval % args.eval_interval != 0:
        raise ValueError("answer-eval-interval must be a multiple of eval-interval")
    if args.pass_monitor_interval % args.eval_interval != 0:
        raise ValueError("pass-monitor-interval must be a multiple of eval-interval")
    if not 0 <= args.synthetic_steps <= args.max_steps:
        raise ValueError("synthetic-steps must lie between zero and max-steps")
    full_depth_steps = args.max_steps if args.full_depth_steps < 0 else args.full_depth_steps
    if not 0 <= full_depth_steps <= args.max_steps:
        raise ValueError("full-depth-steps must lie between zero and max-steps")
    device = select_device(args.device)
    set_seed(args.seed)
    data = MathData(
        args.data_dir,
        args.context_length,
        seed=args.seed,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
    )
    config = ModelConfig(
        vocab_size=data.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=0.05,
        model_type="sparse_moe_mtp_mla",
        router_type="linear",
        router_dim=32,
        initial_execute_probability=0.90,
        router_input_norm=True,
        router_bias=True,
        paper_reproduction=False,
        sparse_inference=True,
        tie_weights=True,
        moe_num_experts=args.num_experts,
        moe_top_k=args.top_k,
        moe_expert_d_ff=args.d_ff // args.top_k,
        moe_bias_update_speed=0.001,
        moe_aux_loss_coefficient=args.moe_aux_weight,
        mtp_loss_coefficient=args.mtp_weight,
        attention_type="mla",
        position_embedding_type="rope",
        mla_q_lora_rank=0,
        mla_kv_lora_rank=args.d_model // 4,
        mla_qk_nope_head_dim=args.d_model // args.n_heads // 2,
        mla_qk_rope_head_dim=args.d_model // args.n_heads // 2,
        mla_v_head_dim=args.d_model // args.n_heads,
    )
    resume_checkpoint = None
    start_step = 0
    if args.resume:
        model, resume_checkpoint = load_checkpoint(args.resume, device)
        if model.config.to_dict() != config.to_dict():
            raise ValueError("resume checkpoint model configuration does not match arguments")
        start_step = int(resume_checkpoint["step"])
        if start_step >= args.max_steps:
            raise ValueError("max-steps must exceed the resume checkpoint step")
    else:
        model = build_model(config).to(device)
    if not isinstance(model, SparseMoEMTPTransformer):
        raise TypeError("math model must be SparseMoEMTPTransformer")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    if resume_checkpoint and resume_checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = args.resume_learning_rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.max_steps - start_step, 1),
        eta_min=args.min_lr,
    )
    experiment_dir = Path(args.experiment_dir)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(
        experiment_dir, FIELDS, purge_step=start_step if args.resume else None
    )
    run_config = {
        "stage": "math_supervised_curriculum",
        "model": config.to_dict(),
        "training": vars(args),
        "dataset": {
            "name": "GSM8K plus deterministic synthetic arithmetic",
            "train_examples": len(data.train_examples),
            "validation_examples": len(data.validation_examples),
            "test_examples": len(data.test_examples),
            "synthetic_train_examples": len(data.synthetic_train_examples),
            "tokenizer": "train-only byte-level BPE with BOS/EOS/PAD/UNK",
            "tokenizer_type": args.tokenizer_type,
            "bpe_vocab_size": args.bpe_vocab_size,
            "batching": "one complete example per row with prompt and padding loss masked",
            "response_format": "reasoning before final tagged answer",
            "curriculum": (
                "easy-to-hard verified synthetic warmup, then 90% GSM8K and "
                "10% hard verified synthetic"
            ),
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "selection_checkpoint": "best_exact_match with parse/repetition/CE tie-breaks",
            "complete_train_fraction": data.complete_example_fraction("mixed"),
            "complete_validation_fraction": data.complete_example_fraction("validation"),
            "full_depth_steps": full_depth_steps,
        },
        "device": str(device),
    }
    write_json(experiment_dir / "config.json", run_config)
    resume_best = resume_checkpoint.get("best_metrics", {}) if resume_checkpoint else {}
    best_loss = float(resume_best.get("val_loss", float("inf")))
    best_answer_score = (
        float(resume_best.get("answer_exact_match", -1.0)),
        float(resume_best.get("answer_parse_rate", -1.0)),
        -float(resume_best.get("answer_repetition_rate", 1.0)),
        -best_loss,
    )
    latest_eval: dict[str, float] = {}
    latest_answers: dict[str, float] = {}
    latest_pass: dict[str, float] = {}
    latest_samples: list[dict[str, object]] = []
    total_seconds = 0.0
    print(
        f"device={device} math-SFT parameters={model.parameter_count():,} "
        f"context={config.context_length} synthetic_steps={args.synthetic_steps} "
        f"start_step={start_step}"
    )
    try:
        for step in range(start_step, args.max_steps):
            model.train()
            synchronize_device(device)
            started = time.perf_counter()
            full_depth = step < full_depth_steps
            density_coefficient = (
                0.0
                if full_depth
                else scheduled_coefficient(
                    step - full_depth_steps,
                    args.max_steps - full_depth_steps,
                    args.lambda_density,
                    start=0.0,
                    end=0.25,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            accumulated = {
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "mtp_loss": 0.0,
                "mtp_accuracy": 0.0,
                "density_loss": 0.0,
                "moe_aux_loss": 0.0,
                "moe_router_entropy": 0.0,
                "total_loss": 0.0,
                "layers_per_token": 0.0,
                "compute_fraction": 0.0,
                "skip_fraction": 0.0,
                "routing_entropy": 0.0,
                "expert_utilization_cv": 0.0,
            }
            supervised_tokens = total_tokens = gsm_examples = 0
            sources = []
            effective_batch_size = (
                args.batch_size * args.gradient_accumulation_steps
            )
            step_source = curriculum_source(
                step, args.synthetic_steps, args.seed
            )
            processed_examples = micro_step = 0
            while processed_examples < effective_batch_size:
                source = step_source
                sources.append(source)
                requested_batch = min(
                    args.batch_size, effective_batch_size - processed_examples
                )
                x, y = data.get_supervised_batch(
                    source,
                    requested_batch,
                    device,
                    max_batch_tokens=args.max_microbatch_tokens,
                )
                microbatch_size = x.shape[0]
                weight = microbatch_size / effective_batch_size
                gsm_examples += microbatch_size * int(source == "gsm_train")
                actions = (
                    torch.ones(
                        (*x.shape, config.n_layers), dtype=torch.long, device=device
                    )
                    if full_depth else None
                )
                output = model(
                    x,
                    y,
                    routing_mode="greedy" if full_depth else "gumbel",
                    actions=actions,
                )
                density = density_loss(output.hard_gates, args.target_density)
                total = (
                    output.lm_loss
                    + config.mtp_loss_coefficient * output.mtp_loss
                    + config.moe_aux_loss_coefficient * output.moe_aux_loss
                    + density_coefficient * density
                )
                (total * weight).backward()
                route = routing_metrics(output, config.n_layers)
                current = {
                    "train_loss": float(output.lm_loss.item()),
                    "train_accuracy": float(top1_accuracy(output.logits, y).item()),
                    "mtp_loss": float(output.mtp_loss.item()),
                    "mtp_accuracy": float(output.mtp_accuracy.item()),
                    "density_loss": float(density.item()),
                    "moe_aux_loss": float(output.moe_aux_loss.item()),
                    "moe_router_entropy": float(output.moe_router_entropy.item()),
                    "total_loss": float(total.item()),
                    **{
                        key: float(route[key]) for key in (
                            "layers_per_token", "compute_fraction", "skip_fraction",
                            "routing_entropy", "expert_utilization_cv",
                        )
                    },
                }
                for key, value in current.items():
                    accumulated[key] += value * weight
                supervised_tokens += int(y.ne(-100).sum().item())
                total_tokens += x.numel()
                processed_examples += microbatch_size
                micro_step += 1
            grad = gradient_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_moe_selection_biases()
            scheduler.step()
            synchronize_device(device)
            seconds = time.perf_counter() - started
            total_seconds += seconds
            stage = "synthetic_curriculum" if step < args.synthetic_steps else "gsm_heavy"
            row = {
                "split": "train",
                "stage": stage,
                **accumulated,
                "density_coefficient": density_coefficient,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": grad,
                "seconds_per_step": seconds,
                "tokens_per_second": total_tokens / max(seconds, 1e-9),
                "supervised_tokens": supervised_tokens,
                "complete_example_fraction": min(
                    data.complete_example_fraction(source) for source in set(sources)
                ),
                "effective_batch_size": effective_batch_size,
                "gsm_batch_fraction": gsm_examples / effective_batch_size,
                "synthetic_batch_fraction": (
                    1.0 - gsm_examples / effective_batch_size
                ),
            }
            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(row, step)
                print(
                    f"step {step:5d} {stage:20s} | CE {row['train_loss']:.3f} "
                    f"| acc {row['train_accuracy']:.3f} | MTP {row['mtp_loss']:.3f} "
                    f"| GSM {row['gsm_batch_fraction']:.2f} "
                    f"| depth {row['layers_per_token']:.2f}/{config.n_layers} | {seconds:.2f}s"
                )
            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = evaluate_math_model(
                    model,
                    data,
                    device,
                    args.batch_size,
                    args.eval_iters,
                    force_full_depth=full_depth_steps == args.max_steps,
                )
                monitor_answers = (
                    (step + 1) % args.answer_eval_interval == 0
                    or step == args.max_steps - 1
                )
                monitor_pass = (
                    (step + 1) % args.pass_monitor_interval == 0
                    or step == args.max_steps - 1
                )
                if monitor_answers:
                    latest_answers, latest_samples = evaluate_math_answers(
                        model,
                        data,
                        device,
                        split="validation",
                        count=args.answer_eval_examples,
                        max_new_tokens=args.answer_tokens,
                        seed=args.seed,
                        force_full_depth=full_depth_steps == args.max_steps,
                    )
                if monitor_pass:
                    latest_pass, _ = evaluate_math_pass_at_k(
                        model,
                        data,
                        device,
                        split="validation",
                        count=args.pass_monitor_examples,
                        k=args.pass_k,
                        max_new_tokens=args.answer_tokens,
                        temperature=args.pass_temperature,
                        seed=args.seed + 10_000,
                        force_full_depth=full_depth_steps == args.max_steps,
                    )
                logger.log(
                    {
                        "split": "validation",
                        "stage": stage,
                        **latest_eval,
                        **(latest_answers if monitor_answers else {}),
                        **(latest_pass if monitor_pass else {}),
                    },
                    step,
                )
                if latest_eval["val_loss"] < best_loss:
                    best_loss = latest_eval["val_loss"]
                    save_checkpoint(
                        experiment_dir / "checkpoints" / "best_val_loss.pt",
                        model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                        stoi=data.stoi, itos=data.itos, training_config=run_config,
                        best_metrics={
                            "val_loss": best_loss,
                            **latest_answers,
                        },
                    )
                if monitor_answers:
                    answer_score = (
                        latest_answers["answer_exact_match"],
                        latest_answers["answer_parse_rate"],
                        -latest_answers["answer_repetition_rate"],
                        -latest_eval["val_loss"],
                    )
                    if answer_score > best_answer_score:
                        best_answer_score = answer_score
                        save_checkpoint(
                            experiment_dir / "checkpoints" / "best_exact_match.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step + 1,
                            stoi=data.stoi,
                            itos=data.itos,
                            training_config=run_config,
                            best_metrics={
                                "val_loss": latest_eval["val_loss"],
                                **latest_answers,
                            },
                        )
                save_checkpoint(
                    experiment_dir / "checkpoints" / "latest.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=data.stoi, itos=data.itos, training_config=run_config,
                    best_metrics={
                        "val_loss": best_loss,
                        "answer_exact_match": best_answer_score[0],
                        "answer_parse_rate": best_answer_score[1],
                        "answer_repetition_rate": -best_answer_score[2],
                    },
                )
                print(
                    f"validation | CE {latest_eval['val_loss']:.3f} "
                    f"| acc {latest_eval['val_accuracy']:.3f} "
                    f"| depth {latest_eval['layers_per_token']:.2f} "
                    f"| FLOPs {latest_eval['estimated_flops_vs_full_dense']:.3f}x"
                )
                if monitor_answers:
                    print(
                        f"answers | exact {latest_answers['answer_exact_match']:.3f} "
                        f"| parse {latest_answers['answer_parse_rate']:.3f} "
                        f"| repeat {latest_answers['answer_repetition_rate']:.3f}"
                    )
    finally:
        logger.close()

    if args.defer_readiness:
        summary = {
            "stage": "math_supervised_curriculum",
            "parameter_count": model.parameter_count(),
            "training_time_sec": total_seconds,
            **latest_eval,
            **latest_answers,
            **latest_pass,
            "readiness_pending": True,
            "checkpoint": str(experiment_dir / "checkpoints" / "best_exact_match.pt"),
        }
        write_json(experiment_dir / "summary.json", summary)
        print("completed math SFT; deferred pass@k readiness evaluation")
        return

    answer_metrics, samples = evaluate_math_answers(
        model, data, device, split="validation", count=args.answer_eval_examples,
        max_new_tokens=args.answer_tokens, seed=args.seed,
        force_full_depth=full_depth_steps == args.max_steps,
    )
    pass_metrics, pass_samples = evaluate_math_pass_at_k(
        model,
        data,
        device,
        split="validation",
        count=args.pass_eval_examples,
        k=args.pass_k,
        max_new_tokens=args.answer_tokens,
        temperature=args.pass_temperature,
        seed=args.seed + 10_000,
        force_full_depth=full_depth_steps == args.max_steps,
    )
    grpo_ready = (
        pass_metrics[f"pass_at_{args.pass_k}"] >= args.grpo_ready_pass_rate
        and pass_metrics["mixed_outcome_group_rate"] >= args.grpo_ready_mixed_rate
    )
    logger = StructuredLogger(experiment_dir, FIELDS)
    logger.log(
        {"split": "readiness", **pass_metrics, "grpo_ready": float(grpo_ready)},
        args.max_steps,
    )
    logger.close()
    summary = {
        "stage": "math_supervised_curriculum",
        "parameter_count": model.parameter_count(),
        "training_time_sec": total_seconds,
        **latest_eval,
        **answer_metrics,
        **pass_metrics,
        "grpo_ready": grpo_ready,
        "grpo_readiness_thresholds": {
            "pass_rate": args.grpo_ready_pass_rate,
            "mixed_outcome_group_rate": args.grpo_ready_mixed_rate,
        },
        "samples": samples,
        "pass_at_k_samples": pass_samples,
        "checkpoint": str(experiment_dir / "checkpoints" / "latest.pt"),
    }
    write_json(experiment_dir / "summary.json", summary)
    write_json(experiment_dir / "samples.json", samples)
    print(
        f"completed math SFT | exact {answer_metrics['answer_exact_match']:.3f} "
        f"| pass@{args.pass_k} {pass_metrics[f'pass_at_{args.pass_k}']:.3f} "
        f"| mixed groups {pass_metrics['mixed_outcome_group_rate']:.3f} "
        f"| GRPO ready={grpo_ready}"
    )


if __name__ == "__main__":
    main()
