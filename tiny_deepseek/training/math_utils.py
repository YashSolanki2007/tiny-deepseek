"""Evaluation, rollout, reward, and reporting helpers for math experiments."""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F

from tiny_deepseek.data.math import MathData, MathExample, extract_tagged_answer, repetition_rate
from tiny_deepseek.core.model import SparseMoEMTPTransformer
from tiny_deepseek.core.utils import (
    estimate_dense_block_flops,
    estimate_mla_skiplayer_moe_mtp_flops,
    routing_metrics,
    synchronize_device,
    top1_accuracy,
)


@torch.inference_mode()
def evaluate_math_model(
    model: SparseMoEMTPTransformer,
    data: MathData,
    device: torch.device,
    batch_size: int,
    eval_iters: int,
    force_full_depth: bool = False,
) -> dict[str, Any]:
    model.eval()
    previous_sparse_inference = model.config.sparse_inference
    # Vectorized hard gating is algebraically identical and avoids Python
    # selected-token dispatch when a nearly dense router executes ~all layers.
    model.config.sparse_inference = False
    totals: dict[str, float] = {}
    route_totals: dict[str, float] = {}
    try:
        for _ in range(eval_iters):
            x, y = data.get_supervised_batch("validation", batch_size, device)
            actions = (
                torch.ones(
                    (*x.shape, model.config.n_layers), dtype=torch.long, device=device
                )
                if force_full_depth else None
            )
            output = model(x, y, routing_mode="greedy", actions=actions)
            current = {
                "val_loss": float(output.lm_loss.item()),
                "val_accuracy": float(top1_accuracy(output.logits, y).item()),
                "mtp_loss": float(output.mtp_loss.item()),
                "mtp_accuracy": float(output.mtp_accuracy.item()),
                "moe_aux_loss": (
                    float(output.moe_aux_loss.item())
                    if output.moe_aux_loss is not None else 0.0
                ),
                "moe_router_entropy": (
                    float(output.moe_router_entropy.item())
                    if output.moe_router_entropy is not None else 0.0
                ),
            }
            route = routing_metrics(output, model.config.n_layers)
            for key, value in current.items():
                totals[key] = totals.get(key, 0.0) + value
            for key in (
                "layers_per_token", "compute_fraction", "skip_fraction",
                "routing_entropy", "expert_utilization_cv",
            ):
                route_totals[key] = route_totals.get(key, 0.0) + float(
                    route.get(key, 0.0)
                )
    finally:
        model.config.sparse_inference = previous_sparse_inference
    metrics = {key: value / eval_iters for key, value in totals.items()}
    metrics.update({key: value / eval_iters for key, value in route_totals.items()})
    metrics["val_perplexity"] = math.exp(min(metrics["val_loss"], 20.0))
    dense = estimate_dense_block_flops(model.config, model.config.context_length)
    sparse = estimate_mla_skiplayer_moe_mtp_flops(
        model.config, model.config.context_length, metrics["compute_fraction"]
    )
    metrics["estimated_flops_vs_full_dense"] = sparse / dense
    return metrics


@torch.inference_mode()
def generate_math_group_cached(
    model: SparseMoEMTPTransformer,
    data: MathData,
    example: MathExample,
    group_size: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    stop_on_eos: bool,
    stop_on_answer: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Generate a full-depth group using the MLA KV cache."""
    prompt = data.tokenizer.encode(example.prompt, bos=True)
    if len(prompt) + max_new_tokens > model.config.context_length:
        raise ValueError("prompt plus rollout exceeds model context")
    token_ids = torch.tensor(prompt, dtype=torch.long, device=device)[None].repeat(
        group_size, 1
    )
    model.eval()
    logits, caches = model.full_depth_prefill(token_ids)
    old_log_probabilities = []
    depth_rows = []
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        scaled_logits = logits / max(temperature, 1e-6)
        log_probability = F.log_softmax(scaled_logits, dim=-1)
        if temperature <= 0:
            next_token = scaled_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(log_probability.exp(), 1)
        if stop_on_eos or stop_on_answer:
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, data.tokenizer.eos_id),
                next_token,
            )
        old_log_probabilities.append(
            log_probability.gather(-1, next_token).squeeze(-1)
        )
        token_ids = torch.cat((token_ids, next_token), dim=1)
        depth_rows.append(torch.ones(group_size, device=device))
        if stop_on_eos or stop_on_answer:
            finished |= next_token.squeeze(-1).eq(data.tokenizer.eos_id)
            if stop_on_answer:
                generated = token_ids[:, len(prompt) :].detach().cpu().tolist()
                finished |= torch.tensor(
                    [
                        extract_tagged_answer(
                            data.tokenizer.decode(row, stop_at_eos=True)
                        )
                        is not None
                        for row in generated
                    ],
                    dtype=torch.bool,
                    device=device,
                )
            if bool(finished.all()):
                break
        logits, caches = model.full_depth_decode(next_token, caches)
    completion_ids = token_ids[:, len(prompt) :]
    completions = [
        data.tokenizer.decode(row.tolist(), stop_at_eos=True)
        for row in completion_ids.detach().cpu()
    ]
    return (
        token_ids,
        torch.stack(old_log_probabilities, dim=1),
        completions,
        torch.stack(depth_rows, dim=1),
    )


@torch.inference_mode()
def generate_math_group(
    model: SparseMoEMTPTransformer,
    data: MathData,
    example: MathExample,
    group_size: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    force_full_depth: bool = False,
    stop_on_eos: bool = False,
    stop_on_answer: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Generate a fixed-length on-policy group and retain sampled log probabilities."""
    if force_full_depth and model.config.attention_type == "mla":
        return generate_math_group_cached(
            model,
            data,
            example,
            group_size,
            max_new_tokens,
            device,
            temperature,
            stop_on_eos,
            stop_on_answer,
        )
    prompt = data.tokenizer.encode(example.prompt, bos=True)
    if len(prompt) + max_new_tokens > model.config.context_length:
        raise ValueError("prompt plus rollout exceeds model context")
    token_ids = torch.tensor(prompt, dtype=torch.long, device=device)[None].repeat(
        group_size, 1
    )
    old_log_probabilities = []
    depth_rows = []
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    model.eval()
    previous_sparse_inference = model.config.sparse_inference
    model.config.sparse_inference = False
    try:
        for _ in range(max_new_tokens):
            actions = (
                torch.ones(
                    (*token_ids.shape, model.config.n_layers),
                    dtype=torch.long,
                    device=device,
                )
                if force_full_depth else None
            )
            output = model(
                token_ids,
                routing_mode="greedy",
                actions=actions,
                compute_mtp=False,
            )
            logits = output.logits[:, -1] / max(temperature, 1e-6)
            log_probability = F.log_softmax(logits, dim=-1)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(log_probability.exp(), 1)
            if stop_on_eos or stop_on_answer:
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, data.tokenizer.eos_id),
                    next_token,
                )
            old_log_probabilities.append(log_probability.gather(-1, next_token).squeeze(-1))
            token_ids = torch.cat((token_ids, next_token), dim=1)
            depth_rows.append(output.hard_gates[:, -1].float().mean(dim=-1))
            if stop_on_eos or stop_on_answer:
                finished |= next_token.squeeze(-1).eq(data.tokenizer.eos_id)
                if stop_on_answer:
                    generated = token_ids[:, len(prompt) :].detach().cpu().tolist()
                    completed_answers = torch.tensor(
                        [
                            extract_tagged_answer(
                                data.tokenizer.decode(row, stop_at_eos=True)
                            )
                            is not None
                            for row in generated
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    finished |= completed_answers
                if bool(finished.all()):
                    break
    finally:
        model.config.sparse_inference = previous_sparse_inference
    completion_ids = token_ids[:, len(prompt) :]
    completions = [
        data.tokenizer.decode(row.tolist(), stop_at_eos=True)
        for row in completion_ids.detach().cpu()
    ]
    return (
        token_ids,
        torch.stack(old_log_probabilities, dim=1),
        completions,
        torch.stack(depth_rows, dim=1),
    )


def score_math_completions(
    completions: list[str], gold_answer: str, device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], list[str | None]]:
    """Apply the strict outcome reward: one if exactly correct, else zero."""
    predictions = [extract_tagged_answer(text) for text in completions]
    gold = extract_tagged_answer(f"<answer>{gold_answer}</answer>")
    exact = torch.tensor(
        [float(prediction == gold) for prediction in predictions],
        dtype=torch.float32,
        device=device,
    )
    components = {
        "exact_reward": float(exact.mean().item()),
    }
    return exact, components, predictions


@torch.inference_mode()
def evaluate_math_answers(
    model: SparseMoEMTPTransformer,
    data: MathData,
    device: torch.device,
    *,
    split: str,
    count: int,
    max_new_tokens: int,
    seed: int,
    force_full_depth: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    eligible = data.eligible_examples(
        split, model.config.context_length - max_new_tokens
    )
    selected = random.Random(seed).sample(eligible, min(count, len(eligible)))
    rows = []
    correct = parseable = 0
    depths = []
    repetitions = []
    difficulty_totals: dict[str, int] = {}
    difficulty_correct: dict[str, int] = {}
    operation_totals: dict[str, int] = {}
    operation_correct: dict[str, int] = {}
    synchronize_device(device)
    for example in selected:
        _, _, completions, depth = generate_math_group(
            model,
            data,
            example,
            1,
            max_new_tokens,
            device,
            temperature=0.0,
            force_full_depth=force_full_depth,
            stop_on_eos=True,
            stop_on_answer=True,
        )
        prediction = extract_tagged_answer(completions[0])
        exact = prediction == example.answer
        correct += int(exact)
        parseable += int(prediction is not None)
        repetitions.append(repetition_rate(completions[0]))
        difficulty_totals[example.difficulty] = (
            difficulty_totals.get(example.difficulty, 0) + 1
        )
        difficulty_correct[example.difficulty] = (
            difficulty_correct.get(example.difficulty, 0) + int(exact)
        )
        operation_totals[example.operation] = (
            operation_totals.get(example.operation, 0) + 1
        )
        operation_correct[example.operation] = (
            operation_correct.get(example.operation, 0) + int(exact)
        )
        depths.append(float(depth.mean().item()) * model.config.n_layers)
        rows.append(
            {
                "question": example.question,
                "gold_answer": example.answer,
                "prediction": prediction,
                "correct": exact,
                "completion": completions[0],
                "layers_per_token": depths[-1],
                "difficulty": example.difficulty,
                "operation": example.operation,
                "repetition_rate": repetitions[-1],
            }
        )
    total = max(len(rows), 1)
    metrics = {
        "answer_exact_match": correct / total,
        "answer_parse_rate": parseable / total,
        "answer_repetition_rate": sum(repetitions) / total,
        "generation_layers_per_token": sum(depths) / total,
        "answer_examples": len(rows),
    }
    for difficulty in ("easy", "medium", "hard"):
        denominator = difficulty_totals.get(difficulty, 0)
        metrics[f"difficulty_{difficulty}_exact_match"] = (
            difficulty_correct.get(difficulty, 0) / denominator
            if denominator else 0.0
        )
    single_operation_correct = 0
    single_operation_total = 0
    for operation in ("addition", "subtraction", "multiplication", "division"):
        denominator = operation_totals.get(operation, 0)
        numerator = operation_correct.get(operation, 0)
        metrics[f"operation_{operation}_exact_match"] = (
            numerator / denominator if denominator else 0.0
        )
        single_operation_correct += numerator
        single_operation_total += denominator
    metrics["arithmetic_operation_accuracy"] = (
        single_operation_correct / single_operation_total
        if single_operation_total else 0.0
    )
    return metrics, rows


@torch.inference_mode()
def evaluate_math_pass_at_k(
    model: SparseMoEMTPTransformer,
    data: MathData,
    device: torch.device,
    *,
    split: str,
    count: int,
    k: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    force_full_depth: bool = True,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Sample k answers for each problem and measure GRPO reward diversity."""
    if count < 1 or k < 2:
        raise ValueError("pass@k evaluation needs count >= 1 and k >= 2")
    eligible = data.eligible_examples(
        split, model.config.context_length - max_new_tokens
    )
    selected = random.Random(seed).sample(eligible, min(count, len(eligible)))
    passed = mixed = correct_samples = parseable_samples = 0
    rows = []
    for index, example in enumerate(selected, start=1):
        torch.manual_seed(seed + index)
        _, _, completions, _ = generate_math_group(
            model,
            data,
            example,
            k,
            max_new_tokens,
            device,
            temperature=temperature,
            force_full_depth=force_full_depth,
            stop_on_eos=True,
            stop_on_answer=True,
        )
        predictions = [extract_tagged_answer(text) for text in completions]
        gold = extract_tagged_answer(f"<answer>{example.answer}</answer>")
        correct = [prediction == gold for prediction in predictions]
        correct_count = sum(correct)
        passed += int(correct_count > 0)
        mixed += int(0 < correct_count < k)
        correct_samples += correct_count
        parseable_samples += sum(prediction is not None for prediction in predictions)
        rows.append(
            {
                "question": example.question,
                "gold_answer": gold,
                "predictions": predictions,
                "correct_samples": correct_count,
            }
        )
        if index % 10 == 0 or index == len(selected):
            print(
                f"pass@{k} evaluation {index}/{len(selected)} | "
                f"running pass rate {passed / index:.3f} | mixed {mixed / index:.3f}"
            )
    problems = max(len(selected), 1)
    samples = max(len(selected) * k, 1)
    return {
        f"pass_at_{k}": passed / problems,
        "sample_exact_match": correct_samples / samples,
        "sample_parse_rate": parseable_samples / samples,
        "mixed_outcome_group_rate": mixed / problems,
        "pass_evaluation_problems": len(selected),
        "pass_evaluation_k": k,
    }, rows
