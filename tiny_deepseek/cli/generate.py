"""Autoregressive sampling with dynamic-depth utilization reporting."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from tiny_deepseek.core.utils import load_checkpoint, select_device, synchronize_device


def sample_next(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        cutoff = torch.topk(logits, k).values[:, -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_tokens(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    model.eval()
    generated_gate_rows = []
    synchronize_device(input_ids.device)
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.context_length :]
        output = model(context)
        next_id = sample_next(output.logits[:, -1, :], temperature, top_k)
        input_ids = torch.cat((input_ids, next_id), dim=1)
        if output.hard_gates is not None:
            # These gates produced the distribution from which next_id was sampled.
            generated_gate_rows.append(output.hard_gates[:, -1, :].float().cpu())
    synchronize_device(input_ids.device)
    elapsed = time.perf_counter() - started
    gates = torch.cat(generated_gate_rows, dim=0) if generated_gate_rows else None
    return input_ids, gates, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    unknown = sorted(set(args.prompt) - stoi.keys())
    if unknown:
        raise ValueError(f"Prompt contains unknown characters: {unknown!r}")
    input_ids = torch.tensor([[stoi[ch] for ch in args.prompt]], dtype=torch.long, device=device)
    tokens, gates, elapsed = generate_tokens(
        model,
        input_ids,
        args.max_new_tokens,
        args.temperature,
        None if args.top_k == 0 else args.top_k,
    )
    print("".join(itos[int(index)] for index in tokens[0].cpu()))
    print(f"\nGeneration speed: {args.max_new_tokens / max(elapsed, 1e-9):.2f} tokens/s")
    if gates is None:
        print(f"Average layers/token: {model.config.n_layers:.2f} / {model.config.n_layers}")
        print("Estimated block utilization: 100.00%")
        print("Estimated block skipping: 0.00%")
        print("Per-layer utilization: " + " ".join(["1.000"] * model.config.n_layers))
    else:
        per_layer = gates.mean(dim=0)
        fraction = gates.mean().item()
        print(f"Average layers/token: {fraction * model.config.n_layers:.2f} / {model.config.n_layers}")
        print(f"Estimated block utilization: {100 * fraction:.2f}%")
        print(f"Estimated block skipping: {100 * (1 - fraction):.2f}%")
        print("Per-layer utilization: " + " ".join(f"{x:.3f}" for x in per_layer.tolist()))
    if model.config.sparse_inference:
        print(
            "Note: greedy inference executes selected queries/FFNs only; FLOP utilization "
            "is analytical and should still be checked against measured latency."
        )
    else:
        print("Note: Stage A evaluates every block; utilization is theoretical, not a wall-clock saving.")


if __name__ == "__main__":
    main()
