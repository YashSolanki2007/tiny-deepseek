"""Evaluation, rollout, reward, and reporting helpers for math experiments."""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F

from math_data import MathData, MathExample, extract_tagged_answer, repetition_rate
from model import SparseMoEMTPTransformer
from utils import (
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
            x, y = data.get_batch("validation", batch_size, device)
            output = model(x, y, routing_mode="greedy")
            current = {
                "val_loss": float(output.lm_loss.item()),
                "val_accuracy": float(top1_accuracy(output.logits, y).item()),
                "mtp_loss": float(output.mtp_loss.item()),
                "mtp_accuracy": float(output.mtp_accuracy.item()),
                "moe_aux_loss": float(output.moe_aux_loss.item()),
                "moe_router_entropy": float(output.moe_router_entropy.item()),
            }
            route = routing_metrics(output, model.config.n_layers)
            for key, value in current.items():
                totals[key] = totals.get(key, 0.0) + value
            for key in (
                "layers_per_token", "compute_fraction", "skip_fraction",
                "routing_entropy", "expert_utilization_cv",
            ):
                route_totals[key] = route_totals.get(key, 0.0) + float(route[key])
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
def generate_math_group(
    model: SparseMoEMTPTransformer,
    data: MathData,
    example: MathExample,
    group_size: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Generate a fixed-length on-policy group and retain sampled log probabilities."""
    prompt = data.tokenizer.encode(example.prompt, bos=True)
    if len(prompt) + max_new_tokens > model.config.context_length:
        raise ValueError("prompt plus rollout exceeds model context")
    token_ids = torch.tensor(prompt, dtype=torch.long, device=device)[None].repeat(
        group_size, 1
    )
    old_log_probabilities = []
    depth_rows = []
    model.eval()
    previous_sparse_inference = model.config.sparse_inference
    model.config.sparse_inference = False
    try:
        for _ in range(max_new_tokens):
            output = model(token_ids, routing_mode="greedy", compute_mtp=False)
            logits = output.logits[:, -1] / max(temperature, 1e-6)
            log_probability = F.log_softmax(logits, dim=-1)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(log_probability.exp(), 1)
            old_log_probabilities.append(log_probability.gather(-1, next_token).squeeze(-1))
            token_ids = torch.cat((token_ids, next_token), dim=1)
            depth_rows.append(output.hard_gates[:, -1].float().mean(dim=-1))
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
    completions: list[str], gold_answer: str, compute_fraction: torch.Tensor,
    compute_target: float,
) -> tuple[torch.Tensor, dict[str, float], list[str | None]]:
    predictions = [extract_tagged_answer(text) for text in completions]
    gold = extract_tagged_answer(f"<answer>{gold_answer}</answer>")
    exact = torch.tensor(
        [float(prediction == gold) for prediction in predictions],
        dtype=compute_fraction.dtype,
        device=compute_fraction.device,
    )
    formatted = torch.tensor(
        [float(prediction is not None) for prediction in predictions],
        dtype=compute_fraction.dtype,
        device=compute_fraction.device,
    )
    closeness_values = []
    for prediction in predictions:
        if prediction is None or gold is None:
            closeness_values.append(0.0)
            continue
        predicted_value, gold_value = float(prediction), float(gold)
        relative_error = abs(predicted_value - gold_value) / (abs(gold_value) + 1.0)
        closeness_values.append(math.exp(-relative_error))
    closeness = torch.tensor(
        closeness_values, dtype=compute_fraction.dtype, device=compute_fraction.device
    )
    repetition = torch.tensor(
        [repetition_rate(text) for text in completions],
        dtype=compute_fraction.dtype,
        device=compute_fraction.device,
    )
    compute_violation = (compute_fraction - compute_target).clamp_min(0.0)
    rewards = (
        exact + 0.10 * formatted + 0.20 * closeness
        - 0.20 * repetition - 0.10 * compute_violation
    )
    components = {
        "exact_reward": float(exact.mean().item()),
        "format_reward": float(formatted.mean().item()),
        "closeness_reward": float(closeness.mean().item()),
        "repetition_penalty": float(repetition.mean().item()),
        "compute_violation": float(compute_violation.mean().item()),
    }
    return rewards, components, predictions


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
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    eligible = data.eligible_examples(
        split, model.config.context_length - max_new_tokens
    )
    selected = random.Random(seed).sample(eligible, min(count, len(eligible)))
    rows = []
    correct = parseable = 0
    depths = []
    synchronize_device(device)
    for example in selected:
        _, _, completions, depth = generate_math_group(
            model, data, example, 1, max_new_tokens, device, temperature=0.0
        )
        prediction = extract_tagged_answer(completions[0])
        exact = prediction == example.answer
        correct += int(exact)
        parseable += int(prediction is not None)
        depths.append(float(depth.mean().item()) * model.config.n_layers)
        rows.append(
            {
                "question": example.question,
                "gold_answer": example.answer,
                "prediction": prediction,
                "correct": exact,
                "completion": completions[0],
                "layers_per_token": depths[-1],
            }
        )
    total = max(len(rows), 1)
    return {
        "answer_exact_match": correct / total,
        "answer_parse_rate": parseable / total,
        "generation_layers_per_token": sum(depths) / total,
        "answer_examples": len(rows),
    }, rows
