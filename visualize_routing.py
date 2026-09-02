"""Save hard- or soft-gate routing heatmaps for an input string."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import load_checkpoint, select_device


def display_character(character: str) -> str:
    return {"\n": "\\n", "\t": "\\t", " ": "·"}.get(character, character)


@torch.inference_mode()
def save_routing_heatmap(model, stoi, text: str, mode: str, output_path: str | Path) -> Path:
    if model.config.model_type not in {
        "sparse", "sparse_moe_mtp", "sparse_moe_mtp_mla", "mor", "mor_skip"
    }:
        raise ValueError("Routing visualization requires a routed checkpoint")
    unknown = sorted(set(text) - stoi.keys())
    if unknown:
        raise ValueError(f"Text contains unknown characters: {unknown!r}")
    text = text[-model.config.context_length :]
    device = next(model.parameters()).device
    ids = torch.tensor([[stoi[ch] for ch in text]], dtype=torch.long, device=device)
    model.eval()
    output = model(ids, routing_mode="greedy")
    gates = output.soft_gates if mode == "soft" else output.hard_gates
    matrix = gates[0].transpose(0, 1).float().cpu().numpy()
    width = max(10.0, 0.32 * len(text))
    height = max(3.5, 0.55 * model.config.n_layers)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(np.arange(model.config.n_layers))
    axis.set_yticklabels([f"Layer {i}" for i in range(model.config.n_layers)])
    axis.set_xticks(np.arange(len(text)))
    axis.set_xticklabels([display_character(ch) for ch in text], fontsize=8)
    axis.set_xlabel("Input character")
    axis.set_title(f"{mode.capitalize()} dynamic-depth gates")
    fig.colorbar(image, ax=axis, label="Gate value", fraction=0.025, pad=0.02)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default="ROMEO:\nWhat light through yonder window breaks?")
    parser.add_argument("--mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--output", default="routing_heatmap.png")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    stoi = checkpoint["stoi"]
    output_path = save_routing_heatmap(model, stoi, args.text, args.mode, args.output)
    print(f"saved {output_path}")

    if args.mode == "hard":
        text = args.text[-model.config.context_length :]
        ids = torch.tensor([[stoi[ch] for ch in text]], dtype=torch.long, device=device)
        matrix = model(ids, routing_mode="greedy").hard_gates[0].transpose(0, 1).float().cpu().numpy()
        labels = " ".join(display_character(ch) for ch in text)
        print("tokens  " + labels)
        for layer_idx, row in enumerate(matrix.astype(int)):
            print(f"layer {layer_idx:<2} " + " ".join(str(value) for value in row))


if __name__ == "__main__":
    main()
