"""Tiny Shakespeare download and character-level data loading."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import torch


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str | Path, url: str = TINY_SHAKESPEARE_URL) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(url, path)
    return path


class TinyShakespeareData:
    """In-memory character data with a deterministic 90/10 split."""

    def __init__(self, path: str | Path, context_length: int, download: bool = True):
        path = Path(path)
        if download:
            download_tiny_shakespeare(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        text = path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.context_length = context_length

        encoded = torch.tensor(self.encode(text), dtype=torch.long)
        split = int(0.9 * len(encoded))
        self.train_data = encoded[:split]
        self.val_data = encoded[split:]
        if min(len(self.train_data), len(self.val_data)) <= context_length:
            raise ValueError("Dataset splits must be longer than context_length")

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"Input contains characters outside the vocabulary: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.train_data if split == "train" else self.val_data
        starts = torch.randint(
            0, len(source) - self.context_length, (batch_size,), generator=generator
        )
        x = torch.stack([source[i : i + self.context_length] for i in starts])
        y = torch.stack([source[i + 1 : i + self.context_length + 1] for i in starts])
        return x.to(device), y.to(device)
