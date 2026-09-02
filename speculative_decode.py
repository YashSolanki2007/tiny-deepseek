"""Lossless one-depth MTP speculative decoding and latency benchmarking."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from data import TinyShakespeareData
from generate import generate_tokens
from model import SparseMoEMTPTransformer
from utils import load_checkpoint, select_device, set_seed, synchronize_device, write_json


def sampling_distribution(
    logits: torch.Tensor, temperature: float, top_k: int | None
) -> torch.Tensor:
    if temperature <= 0:
        probabilities = torch.zeros_like(logits)
        return probabilities.scatter(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    scaled = logits / temperature
    if top_k is not None:
        count = min(top_k, scaled.shape[-1])
        cutoff = torch.topk(scaled, count, dim=-1).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    return torch.softmax(scaled, dim=-1)


def sample_probability(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probabilities, num_samples=1)


def residual_distribution(
    target: torch.Tensor, draft: torch.Tensor
) -> torch.Tensor:
    residual = (target - draft).clamp_min(0.0)
    mass = residual.sum(dim=-1, keepdim=True)
    normalized = residual / mass.clamp_min(torch.finfo(residual.dtype).tiny)
    return torch.where(mass > 0, normalized, target)


@torch.inference_mode()
def speculative_generate(
    model: SparseMoEMTPTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate exactly from the target distribution using one MTP draft token."""
    if input_ids.shape[0] != 1:
        raise ValueError("speculative_generate currently supports batch size 1")
    if max_new_tokens <= 0:
        return input_ids, {
            "tokens_generated": 0, "target_forward_calls": 0, "draft_calls": 0,
            "accepted_drafts": 0, "rejected_drafts": 0, "acceptance_rate": 0.0,
            "tokens_per_second": 0.0,
        }
    model.eval()
    prompt_length = input_ids.shape[1]
    generated = input_ids
    target_calls = draft_calls = accepted = rejected = bonus_tokens = 0
    fallback_tokens = 0
    acceptance_probabilities: list[float] = []
    synchronize_device(input_ids.device)
    started = time.perf_counter()

    target_context = generated[:, -model.config.context_length :]
    target_output = model(target_context, compute_mtp=False)
    target_calls += 1
    seed_distribution = sampling_distribution(
        target_output.logits[:, -1], temperature, top_k
    )
    seed_token = sample_probability(seed_distribution)
    generated = torch.cat((generated, seed_token), dim=1)

    while generated.shape[1] - prompt_length < max_new_tokens:
        # Exact parallel verification needs both appended positions inside one
        # causal window. Fall back at the sliding-window boundary.
        if target_context.shape[1] + 2 > model.config.context_length:
            remaining = max_new_tokens - (generated.shape[1] - prompt_length)
            for _ in range(remaining):
                context = generated[:, -model.config.context_length :]
                output = model(context, compute_mtp=False)
                target_calls += 1
                token = sample_probability(
                    sampling_distribution(output.logits[:, -1], temperature, top_k)
                )
                generated = torch.cat((generated, token), dim=1)
                fallback_tokens += 1
            break

        draft_logits = model.mtp_draft_logits(
            target_context, target_output.hidden_states, seed_token
        )
        draft_calls += 1
        draft_probability = sampling_distribution(draft_logits, temperature, top_k)
        draft_token = sample_probability(draft_probability)

        verification_input = torch.cat((target_context, seed_token, draft_token), dim=1)
        verification = model(verification_input, compute_mtp=False)
        target_calls += 1
        target_probability = sampling_distribution(
            verification.logits[:, -2], temperature, top_k
        )
        token_index = draft_token.item()
        acceptance_probability = min(
            1.0,
            float(
                target_probability[0, token_index]
                / draft_probability[0, token_index].clamp_min(1e-12)
            ),
        )
        acceptance_probabilities.append(acceptance_probability)
        if torch.rand((), device=input_ids.device).item() <= acceptance_probability:
            generated = torch.cat((generated, draft_token), dim=1)
            accepted += 1
            target_context = verification_input
            target_output = verification
            if generated.shape[1] - prompt_length >= max_new_tokens:
                break
            bonus_probability = sampling_distribution(
                verification.logits[:, -1], temperature, top_k
            )
            seed_token = sample_probability(bonus_probability)
            generated = torch.cat((generated, seed_token), dim=1)
            bonus_tokens += 1
        else:
            correction = sample_probability(
                residual_distribution(target_probability, draft_probability)
            )
            generated = torch.cat((generated, correction), dim=1)
            rejected += 1
            if generated.shape[1] - prompt_length >= max_new_tokens:
                break
            target_context = generated[:, -model.config.context_length :]
            target_output = model(target_context, compute_mtp=False)
            target_calls += 1
            seed_token = sample_probability(
                sampling_distribution(target_output.logits[:, -1], temperature, top_k)
            )
            generated = torch.cat((generated, seed_token), dim=1)

    synchronize_device(input_ids.device)
    elapsed = time.perf_counter() - started
    generated = generated[:, : prompt_length + max_new_tokens]
    drafted = accepted + rejected
    stats = {
        "tokens_generated": max_new_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": max_new_tokens / max(elapsed, 1e-9),
        "target_forward_calls": target_calls,
        "draft_calls": draft_calls,
        "accepted_drafts": accepted,
        "rejected_drafts": rejected,
        "acceptance_rate": accepted / max(drafted, 1),
        "mean_acceptance_probability": (
            sum(acceptance_probabilities) / max(len(acceptance_probabilities), 1)
        ),
        "bonus_tokens": bonus_tokens,
        "fallback_tokens": fallback_tokens,
        "tokens_per_target_call": max_new_tokens / max(target_calls, 1),
        "target_call_reduction": 1.0 - target_calls / max(max_new_tokens, 1),
    }
    return generated, stats


def encode_prompt(dataset: TinyShakespeareData, prompt: str, device: torch.device) -> torch.Tensor:
    unknown = sorted(set(prompt) - dataset.stoi.keys())
    if unknown:
        raise ValueError(f"prompt contains unknown characters: {unknown!r}")
    return torch.tensor([[dataset.stoi[ch] for ch in prompt]], device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/skiplayer_moe_mtp_mla_rope_seed42/"
            "grpo/checkpoints/best_quality_compute.pt"
        ),
    )
    parser.add_argument("--data-path", default="data/input.txt")
    parser.add_argument("--output-dir", default="results/speculative_mla_rope_seed42")
    parser.add_argument("--prompts", nargs="+", default=["ROMEO:", "JULIET:", "KING RICHARD:"])
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--benchmark-tokens", type=int, default=96)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be at least 1")
    top_k = None if args.top_k == 0 else args.top_k

    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, SparseMoEMTPTransformer) or model.config.attention_type != "mla":
        raise ValueError("checkpoint must be the MLA+RoPE SkipLayer+MoE+MTP model")
    dataset = TinyShakespeareData(args.data_path, model.config.context_length)
    if dataset.stoi != checkpoint["stoi"]:
        raise ValueError("checkpoint vocabulary differs from dataset")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup = encode_prompt(dataset, args.prompts[0], device)
    _ = model(warmup, compute_mtp=False)
    benchmark_prompt = encode_prompt(dataset, args.prompts[0], device)
    if benchmark_prompt.shape[1] + args.benchmark_tokens > model.config.context_length:
        raise ValueError("benchmark must fit within context_length for exact parallel verification")
    # Exercise every prefix length before recording MPS/CUDA timings. Without
    # this, lazy kernel/graph setup dominates the first short benchmark.
    set_seed(args.seed - 1)
    _ = generate_tokens(
        model, benchmark_prompt.clone(), args.benchmark_tokens,
        temperature=args.temperature, top_k=top_k,
    )
    set_seed(args.seed - 1)
    _ = speculative_generate(
        model, benchmark_prompt.clone(), args.benchmark_tokens,
        temperature=args.temperature, top_k=top_k,
    )
    baseline_runs: list[float] = []
    speculative_runs: list[dict[str, Any]] = []
    baseline_ids = speculative_ids = None
    for repeat in range(args.benchmark_repeats):
        repeat_seed = args.seed + repeat
        if repeat % 2 == 0:
            set_seed(repeat_seed)
            current_baseline_ids, _, baseline_seconds = generate_tokens(
                model, benchmark_prompt.clone(), args.benchmark_tokens,
                temperature=args.temperature, top_k=top_k,
            )
            set_seed(repeat_seed)
            current_speculative_ids, current_stats = speculative_generate(
                model, benchmark_prompt.clone(), args.benchmark_tokens,
                temperature=args.temperature, top_k=top_k,
            )
        else:
            set_seed(repeat_seed)
            current_speculative_ids, current_stats = speculative_generate(
                model, benchmark_prompt.clone(), args.benchmark_tokens,
                temperature=args.temperature, top_k=top_k,
            )
            set_seed(repeat_seed)
            current_baseline_ids, _, baseline_seconds = generate_tokens(
                model, benchmark_prompt.clone(), args.benchmark_tokens,
                temperature=args.temperature, top_k=top_k,
            )
        baseline_runs.append(baseline_seconds)
        speculative_runs.append(current_stats)
        if repeat == 0:
            baseline_ids = current_baseline_ids
            speculative_ids = current_speculative_ids

    assert baseline_ids is not None and speculative_ids is not None
    baseline_seconds = statistics.median(baseline_runs)
    speculative_seconds = statistics.median(
        run["elapsed_seconds"] for run in speculative_runs
    )
    baseline_speed = args.benchmark_tokens / max(baseline_seconds, 1e-9)
    speculative_speed = args.benchmark_tokens / max(speculative_seconds, 1e-9)
    accepted = sum(run["accepted_drafts"] for run in speculative_runs)
    drafted = sum(run["draft_calls"] for run in speculative_runs)
    target_calls = sum(run["target_forward_calls"] for run in speculative_runs)
    speculative_stats = {
        **speculative_runs[0],
        "elapsed_seconds": speculative_seconds,
        "tokens_per_second": speculative_speed,
        "target_forward_calls": target_calls / args.benchmark_repeats,
        "draft_calls": drafted / args.benchmark_repeats,
        "accepted_drafts": accepted / args.benchmark_repeats,
        "rejected_drafts": (drafted - accepted) / args.benchmark_repeats,
        "acceptance_rate": accepted / max(drafted, 1),
        "mean_acceptance_probability": sum(
            run["mean_acceptance_probability"] for run in speculative_runs
        ) / args.benchmark_repeats,
        "bonus_tokens": sum(run["bonus_tokens"] for run in speculative_runs)
        / args.benchmark_repeats,
        "fallback_tokens": sum(run["fallback_tokens"] for run in speculative_runs)
        / args.benchmark_repeats,
        "tokens_per_target_call": (
            args.benchmark_tokens * args.benchmark_repeats / max(target_calls, 1)
        ),
        "target_call_reduction": 1.0 - target_calls / (
            args.benchmark_tokens * args.benchmark_repeats
        ),
        "elapsed_seconds_by_repeat": [
            run["elapsed_seconds"] for run in speculative_runs
        ],
    }
    benchmark = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "architecture": {
            "target_model": model.config.model_type,
            "attention": model.config.attention_type,
            "positions": model.config.position_embedding_type,
            "target_layers": model.config.n_layers,
            "draft_layers": 1,
            "experts": model.config.moe_num_experts,
            "active_experts": model.config.moe_top_k,
        },
        "prompt": args.prompts[0],
        "tokens": args.benchmark_tokens,
        "repeats": args.benchmark_repeats,
        "temperature": args.temperature,
        "top_k": top_k,
        "baseline": {
            "elapsed_seconds": baseline_seconds,
            "tokens_per_second": baseline_speed,
            "target_forward_calls": args.benchmark_tokens,
            "elapsed_seconds_by_repeat": baseline_runs,
            "text": dataset.decode(baseline_ids[0].cpu().tolist()),
        },
        "speculative": {
            **speculative_stats,
            "text": dataset.decode(speculative_ids[0].cpu().tolist()),
        },
        "wall_clock_speedup": speculative_speed / baseline_speed,
    }

    samples = []
    for index, prompt in enumerate(args.prompts):
        if len(prompt) + args.sample_tokens > model.config.context_length:
            raise ValueError("sample prompt plus sample-tokens must fit inside context_length")
        set_seed(args.seed + index + 1)
        ids, stats = speculative_generate(
            model, encode_prompt(dataset, prompt, device), args.sample_tokens,
            temperature=args.temperature, top_k=top_k,
        )
        text = dataset.decode(ids[0].cpu().tolist())
        samples.append({"prompt": prompt, "seed": args.seed + index + 1, "text": text, **stats})
        (output_dir / f"sample_{index + 1}.txt").write_text(text + "\n", encoding="utf-8")

    result = {"benchmark": benchmark, "samples": samples}
    write_json(output_dir / "results.json", result)
    (output_dir / "generations.txt").write_text(
        "\n\n".join(sample["text"] for sample in samples) + "\n", encoding="utf-8"
    )
    print(json.dumps({"benchmark": {k: v for k, v in benchmark.items() if k not in {"baseline", "speculative"}}, "baseline": {k: v for k, v in benchmark["baseline"].items() if k != "text"}, "speculative": {k: v for k, v in benchmark["speculative"].items() if k != "text"}}, indent=2))
    for sample in samples:
        print("\n" + "=" * 72)
        print(sample["text"])


if __name__ == "__main__":
    main()
