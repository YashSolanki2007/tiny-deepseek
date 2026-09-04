"""Binary-correctness token-policy GRPO for the math MLA+RoPE sparse model."""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tiny_deepseek.training.logging import StructuredLogger
from tiny_deepseek.core.losses import group_relative_advantages
from tiny_deepseek.data.math import MathData, MathExample
from tiny_deepseek.training.math_utils import (
    evaluate_math_answers,
    evaluate_math_model,
    generate_math_group,
    score_math_completions,
)
from tiny_deepseek.core.model import SparseMoE, SparseMoEMTPTransformer
from tiny_deepseek.core.utils import (
    gradient_norm,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    write_json,
)


FIELDS = [
    "step", "split", "mean_reward", "reward_std", "best_reward", "worst_reward",
    "exact_reward", "group_exact_match", "group_parse_rate", "policy_loss",
    "kl_loss", "sft_loss", "mtp_loss", "moe_aux_loss", "total_loss",
    "mean_probability_ratio", "clip_fraction", "advantage_std", "learning_rate",
    "gradient_norm", "seconds_per_step", "rollout_tokens_per_second",
    "generation_layers_per_token", "compute_fraction", "val_loss",
    "val_perplexity", "val_accuracy", "mtp_accuracy", "layers_per_token",
    "skip_fraction", "expert_utilization_cv", "estimated_flops_vs_full_dense",
    "answer_exact_match", "answer_parse_rate",
    "curriculum_difficulty", "mixed_prompt_pool_size", "mixed_group",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--checkpoint",
        default="artifacts/experiments/math_v2_seed42/supervised/checkpoints/best_exact_match.pt",
    )
    result.add_argument(
        "--experiment-dir", default="artifacts/experiments/math_v2_seed42/grpo"
    )
    result.add_argument("--data-dir", default="data/gsm8k")
    result.add_argument("--device", default="auto")
    result.add_argument("--max-steps", type=int, default=500)
    result.add_argument("--group-size", type=int, default=16)
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--learning-rate", type=float, default=3e-6)
    result.add_argument("--clip-epsilon", type=float, default=0.2)
    result.add_argument("--beta-kl", type=float, default=0.04)
    result.add_argument("--sft-coefficient", type=float, default=0.5)
    result.add_argument("--sft-batch-size", type=int, default=4)
    result.add_argument(
        "--prompt-screen-examples",
        type=int,
        default=128,
        help="Training prompts screened with separate rollouts for mixed outcomes.",
    )
    result.add_argument("--eval-interval", type=int, default=10)
    result.add_argument("--eval-iters", type=int, default=3)
    result.add_argument("--log-interval", type=int, default=1)
    result.add_argument("--answer-eval-examples", type=int, default=6)
    result.add_argument("--seed", type=int, default=42)
    return result


@torch.inference_mode()
def build_mixed_prompt_pool(
    model: SparseMoEMTPTransformer,
    data: MathData,
    examples: list[MathExample],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    count: int,
    seed: int,
) -> tuple[list[MathExample], list[dict[str, Any]]]:
    """Screen prompts with rollouts that are separate from policy updates."""
    candidates = list(examples)
    random.Random(seed).shuffle(candidates)
    selected = candidates[: min(count, len(candidates))]
    pool = []
    diagnostics = []
    for index, example in enumerate(selected, start=1):
        torch.manual_seed(seed + index)
        _, _, completions, _ = generate_math_group(
            model,
            data,
            example,
            group_size,
            max_new_tokens,
            device,
            temperature,
        )
        _, _, predictions = score_math_completions(
            completions, example.answer, device
        )
        correct = sum(prediction == example.answer for prediction in predictions)
        if 0 < correct < group_size:
            pool.append(example)
        diagnostics.append(
            {
                "question": example.question,
                "answer": example.answer,
                "difficulty": example.difficulty,
                "operation": example.operation,
                "correct_samples": correct,
            }
        )
        if index % 16 == 0 or index == len(selected):
            print(
                f"GRPO prompt screening {index}/{len(selected)} | "
                f"mixed pool {len(pool)}"
            )
    return pool, diagnostics


def curriculum_candidates(
    pool: list[MathExample], step: int, max_steps: int
) -> list[MathExample]:
    """Start with easy mixed prompts, then admit medium and hard prompts."""
    progress = step / max(max_steps, 1)
    allowed = (
        {"easy"} if progress < 1 / 3
        else {"easy", "medium"} if progress < 2 / 3
        else {"easy", "medium", "hard", "unknown"}
    )
    candidates = [example for example in pool if example.difficulty in allowed]
    return candidates or pool


def freeze_routing(model: SparseMoEMTPTransformer) -> None:
    for parameter in model.router.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks:
        if isinstance(block.mlp, SparseMoE):
            for parameter in block.mlp.router.parameters():
                parameter.requires_grad_(False)


def completion_log_probabilities(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_length: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    completion_targets = token_ids[:, prompt_length:]
    completion_logits = logits[
        :, prompt_length - 1 : prompt_length - 1 + completion_targets.shape[1]
    ] / max(temperature, 1e-6)
    log_probabilities = F.log_softmax(completion_logits, dim=-1)
    chosen = log_probabilities.gather(-1, completion_targets.unsqueeze(-1)).squeeze(-1)
    return chosen, log_probabilities


def main() -> None:
    args = parser().parse_args()
    if args.group_size < 2:
        raise ValueError("group-size must be at least two")
    device = select_device(args.device)
    set_seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, SparseMoEMTPTransformer) or model.config.attention_type != "mla":
        raise ValueError("math GRPO requires the MLA+RoPE MoE+MTP checkpoint")
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
    reference = copy.deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    freeze_routing(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1), eta_min=args.learning_rate * 0.1
    )
    experiment_dir = Path(args.experiment_dir)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(experiment_dir, FIELDS)
    config: dict[str, Any] = {
        "stage": "math_binary_correctness_grpo",
        "model": model.config.to_dict(),
        "dataset": dataset_config,
        "grpo": vars(args),
        "reward": {
            "exact_normalized_answer": 1.0,
            "otherwise": 0.0,
            "advantage": "(reward - group mean) / population standard deviation",
            "tied_group_behavior": "all advantages are zero",
        },
        "frozen": "SkipLayer router, MoE selection routers, and reference policy",
        "optimized": "token policy/backbone, active experts, LM head, and MTP module",
        "regularization": "reference KL plus GSM8K supervised replay and MTP loss",
        "prompt_curriculum": (
            "separately screen mixed-outcome prompts; easy, then easy+medium, "
            "then all difficulties"
        ),
        "device": str(device),
    }
    write_json(experiment_dir / "config.json", config)
    eligible = data.eligible_examples(
        "train", model.config.context_length - args.max_new_tokens
    )
    rng = random.Random(args.seed + 900)
    latest_eval: dict[str, float] = {}
    latest_samples: list[dict[str, Any]] = []
    print(
        f"device={device} quality-GRPO group={args.group_size} steps={args.max_steps} "
        f"trainable={sum(p.numel() for p in trainable):,}/{model.parameter_count():,}"
    )
    mixed_pool, pool_diagnostics = build_mixed_prompt_pool(
        model,
        data,
        eligible,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
        count=args.prompt_screen_examples,
        seed=args.seed + 20_000,
    )
    write_json(experiment_dir / "prompt_screening.json", pool_diagnostics)
    if not mixed_pool:
        logger.close()
        raise RuntimeError(
            "No separately screened training prompt produced both correct and "
            "incorrect trajectories; GRPO has no binary advantage signal."
        )
    write_json(
        experiment_dir / "mixed_prompt_pool.json",
        [
            {
                "question": example.question,
                "answer": example.answer,
                "difficulty": example.difficulty,
                "operation": example.operation,
            }
            for example in mixed_pool
        ],
    )
    print(f"GRPO curriculum pool contains {len(mixed_pool)} mixed-outcome prompts")
    try:
        for step in range(args.max_steps):
            candidates = curriculum_candidates(mixed_pool, step, args.max_steps)
            example = rng.choice(candidates)
            synchronize_device(device)
            started = time.perf_counter()
            token_ids, old_logp, completions, rollout_depth = generate_math_group(
                model, data, example, args.group_size, args.max_new_tokens,
                device, args.temperature,
            )
            # Rollout helpers use inference_mode; clone the sampled trajectory
            # back into an ordinary tensor before the differentiable PPO pass.
            token_ids = torch.tensor(
                token_ids.tolist(), dtype=torch.long, device=device
            )
            compute = rollout_depth.mean(dim=1)
            rewards, reward_parts, predictions = score_math_completions(
                completions, example.answer, device
            )
            advantages = group_relative_advantages(rewards[None]).squeeze(0).detach()
            prompt_length = token_ids.shape[1] - args.max_new_tokens

            model.eval()
            current = model(
                token_ids[:, :-1], routing_mode="greedy", compute_mtp=False
            )
            new_logp, current_distribution = completion_log_probabilities(
                current.logits, token_ids, prompt_length, args.temperature
            )
            ratio = torch.exp(new_logp - old_logp.detach())
            clipped_ratio = ratio.clamp(
                1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon
            )
            surrogate = torch.minimum(
                ratio * advantages[:, None], clipped_ratio * advantages[:, None]
            )
            policy_loss = -surrogate.mean()
            with torch.no_grad():
                reference_output = reference(
                    token_ids[:, :-1], routing_mode="greedy", compute_mtp=False
                )
                _, reference_distribution = completion_log_probabilities(
                    reference_output.logits, token_ids, prompt_length, args.temperature
                )
            current_probability = current_distribution.exp()
            kl_loss = (
                current_probability * (current_distribution - reference_distribution)
            ).sum(dim=-1).mean()

            model.train()
            sft_x, sft_y = data.get_supervised_batch(
                "gsm_train", args.sft_batch_size, device
            )
            sft = model(sft_x, sft_y, routing_mode="greedy")
            moe_aux_loss = (
                sft.moe_aux_loss
                if sft.moe_aux_loss is not None
                else sft.lm_loss.new_zeros(())
            )
            total = (
                policy_loss
                + args.beta_kl * kl_loss
                + args.sft_coefficient * sft.lm_loss
                + model.config.mtp_loss_coefficient * sft.mtp_loss
                + model.config.moe_aux_loss_coefficient * moe_aux_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = gradient_norm(trainable)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            synchronize_device(device)
            seconds = time.perf_counter() - started
            exact = sum(prediction == example.answer for prediction in predictions)
            parseable = sum(prediction is not None for prediction in predictions)
            row = {
                "split": "train",
                "mean_reward": float(rewards.mean().item()),
                "reward_std": float(rewards.std(unbiased=False).item()),
                "best_reward": float(rewards.max().item()),
                "worst_reward": float(rewards.min().item()),
                **reward_parts,
                "group_exact_match": exact / args.group_size,
                "group_parse_rate": parseable / args.group_size,
                "mixed_group": float(0 < exact < args.group_size),
                "curriculum_difficulty": example.difficulty,
                "mixed_prompt_pool_size": len(mixed_pool),
                "policy_loss": float(policy_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "sft_loss": float(sft.lm_loss.item()),
                "mtp_loss": float(sft.mtp_loss.item()),
                "moe_aux_loss": float(moe_aux_loss.item()),
                "total_loss": float(total.item()),
                "mean_probability_ratio": float(ratio.mean().item()),
                "clip_fraction": float(ratio.ne(clipped_ratio).float().mean().item()),
                "advantage_std": float(advantages.std(unbiased=False).item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": grad,
                "seconds_per_step": seconds,
                "rollout_tokens_per_second": (
                    args.group_size * args.max_new_tokens / max(seconds, 1e-9)
                ),
                "generation_layers_per_token": float(
                    compute.mean().item() * model.config.n_layers
                ),
                "compute_fraction": float(compute.mean().item()),
            }
            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(row, step)
                print(
                    f"step {step:3d} | reward {row['mean_reward']:.3f} "
                    f"| exact {row['group_exact_match']:.2f} | parse {row['group_parse_rate']:.2f} "
                    f"| KL {row['kl_loss']:.4f} | SFT {row['sft_loss']:.3f} | {seconds:.1f}s"
                )
            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = evaluate_math_model(
                    model, data, device, batch_size=2, eval_iters=args.eval_iters
                )
                logger.log({"split": "validation", **latest_eval}, step)
                save_checkpoint(
                    experiment_dir / "checkpoints" / "latest.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=data.stoi, itos=data.itos, training_config=config,
                    best_metrics={"val_loss": latest_eval["val_loss"]},
                )
                print(
                    f"validation | CE {latest_eval['val_loss']:.3f} "
                    f"| acc {latest_eval['val_accuracy']:.3f} "
                    f"| depth {latest_eval['layers_per_token']:.2f}"
                )
    finally:
        logger.close()

    answer_metrics, latest_samples = evaluate_math_answers(
        model, data, device, split="test", count=args.answer_eval_examples,
        max_new_tokens=args.max_new_tokens, seed=args.seed + 1,
    )
    summary = {
        "stage": "math_binary_correctness_grpo",
        **latest_eval,
        **answer_metrics,
        "samples": latest_samples,
        "checkpoint": str(experiment_dir / "checkpoints" / "latest.pt"),
    }
    write_json(experiment_dir / "summary.json", summary)
    write_json(experiment_dir / "samples.json", latest_samples)
    save_checkpoint(
        experiment_dir / "checkpoints" / "final.pt",
        model=model, optimizer=optimizer, scheduler=scheduler, step=args.max_steps,
        stoi=data.stoi, itos=data.itos, training_config=config,
        best_metrics={"val_loss": latest_eval.get("val_loss", math.inf)}, summary=summary,
    )
    print(
        f"completed quality GRPO | exact {answer_metrics['answer_exact_match']:.3f} "
        f"| parse {answer_metrics['answer_parse_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
