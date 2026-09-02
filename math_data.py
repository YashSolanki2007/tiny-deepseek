"""Byte-level synthetic arithmetic and GSM8K data for causal language modeling."""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


GSM8K_BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data"
)
ANSWER_PATTERN = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE)
GSM_ANSWER_PATTERN = re.compile(r"####\s*([^\n]+)")
CALCULATION_PATTERN = re.compile(r"<<.*?>>")


class ByteTokenizer:
    """Lossless UTF-8 tokenizer with three special tokens and no fitted state."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    byte_offset = 3
    vocab_size = 259

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        values = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if bos:
            values.insert(0, self.bos_id)
        if eos:
            values.append(self.eos_id)
        return values

    def decode(self, token_ids: Iterable[int], *, stop_at_eos: bool = False) -> str:
        values: list[int] = []
        for token in token_ids:
            token = int(token)
            if stop_at_eos and token == self.eos_id:
                break
            if token >= self.byte_offset:
                values.append(token - self.byte_offset)
        return bytes(values).decode("utf-8", errors="replace")

    @property
    def stoi(self) -> dict[str, int]:
        mapping = {"<pad>": self.pad_id, "<bos>": self.bos_id, "<eos>": self.eos_id}
        mapping.update({f"<byte:{value}>": value + self.byte_offset for value in range(256)})
        return mapping

    @property
    def itos(self) -> dict[int, str]:
        mapping = {self.pad_id: "<pad>", self.bos_id: "<bos>", self.eos_id: "<eos>"}
        mapping.update({value + self.byte_offset: f"<byte:{value}>" for value in range(256)})
        return mapping


@dataclass(frozen=True)
class MathExample:
    question: str
    reasoning: str
    answer: str

    @property
    def prompt(self) -> str:
        return f"Question: {self.question.strip()}\nResponse:"

    @property
    def response(self) -> str:
        reasoning = self.reasoning.strip()
        suffix = f"\nReasoning: {reasoning}" if reasoning else ""
        return f" <answer>{self.answer}</answer>{suffix}"

    @property
    def text(self) -> str:
        return self.prompt + self.response


def normalize_numeric_answer(value: str) -> str | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(". ")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not torch.isfinite(torch.tensor(number)):
        return None
    if number.is_integer():
        return str(int(number))
    return format(number, ".10g")


def extract_tagged_answer(text: str) -> str | None:
    match = ANSWER_PATTERN.search(text)
    return normalize_numeric_answer(match.group(1)) if match else None


def repetition_rate(text: str, width: int = 4) -> float:
    compact = " ".join(text.split())
    if len(compact) < width:
        return 0.0
    grams = [compact[index : index + width] for index in range(len(compact) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _download_jsonl(path: Path, split: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GSM8K_BASE_URL}/{split}.jsonl"
    print(f"Downloading GSM8K {split} split to {path} ...")
    urllib.request.urlretrieve(url, path)


def _read_gsm8k(path: Path) -> list[MathExample]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            answer_match = GSM_ANSWER_PATTERN.search(record["answer"])
            if answer_match is None:
                continue
            answer = normalize_numeric_answer(answer_match.group(1))
            if answer is None:
                continue
            reasoning = GSM_ANSWER_PATTERN.sub("", record["answer"]).strip()
            reasoning = CALCULATION_PATTERN.sub("", reasoning)
            examples.append(MathExample(record["question"], reasoning, answer))
    return examples


def generate_synthetic_examples(count: int, seed: int) -> list[MathExample]:
    rng = random.Random(seed)
    names = ["Ava", "Ben", "Chloe", "Dev", "Emma", "Finn", "Grace", "Hugo"]
    objects = ["apples", "books", "coins", "marbles", "pencils", "tickets"]
    examples: list[MathExample] = []
    for index in range(count):
        name, item = rng.choice(names), rng.choice(objects)
        kind = index % 5
        if kind == 0:
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            question = f"{name} has {a} {item} and gets {b} more. How many {item} are there?"
            answer = a + b
            reasoning = f"Add the two amounts: {a} + {b} = {answer}."
        elif kind == 1:
            b = rng.randint(2, 60)
            answer = rng.randint(2, 80)
            a = b + answer
            question = f"{name} has {a} {item} and gives away {b}. How many remain?"
            reasoning = f"Subtract what was given away: {a} - {b} = {answer}."
        elif kind == 2:
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            answer = a * b
            question = f"There are {a} boxes with {b} {item} in each. How many {item} are there?"
            reasoning = f"Multiply boxes by items per box: {a} * {b} = {answer}."
        elif kind == 3:
            groups, answer = rng.randint(2, 15), rng.randint(2, 15)
            total = groups * answer
            question = f"{total} {item} are shared equally among {groups} people. How many does each get?"
            reasoning = f"Divide the total equally: {total} / {groups} = {answer}."
        else:
            a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 20)
            subtotal, answer = a + b, a + b - c
            question = (
                f"{name} has {a} {item}, gets {b} more, then uses {c}. "
                f"How many {item} remain?"
            )
            reasoning = (
                f"First add: {a} + {b} = {subtotal}. Then subtract: "
                f"{subtotal} - {c} = {answer}."
            )
        examples.append(MathExample(question, reasoning, str(answer)))
    return examples


class MathData:
    """GSM8K examples plus deterministic synthetic streams for LM training."""

    def __init__(
        self,
        data_dir: str | Path,
        context_length: int,
        *,
        synthetic_train_size: int = 10_000,
        synthetic_val_size: int = 500,
        validation_fraction: float = 0.1,
        seed: int = 42,
        download: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        train_path = self.data_dir / "train.jsonl"
        test_path = self.data_dir / "test.jsonl"
        if download:
            _download_jsonl(train_path, "train")
            _download_jsonl(test_path, "test")
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("GSM8K train.jsonl and test.jsonl are required")
        self.tokenizer = ByteTokenizer()
        self.context_length = context_length
        all_train = _read_gsm8k(train_path)
        order = list(range(len(all_train)))
        random.Random(seed).shuffle(order)
        validation_count = max(1, round(len(order) * validation_fraction))
        self.validation_examples = [all_train[index] for index in order[:validation_count]]
        self.train_examples = [all_train[index] for index in order[validation_count:]]
        self.test_examples = _read_gsm8k(test_path)
        self.synthetic_train_examples = generate_synthetic_examples(
            synthetic_train_size, seed + 101
        )
        self.synthetic_validation_examples = generate_synthetic_examples(
            synthetic_val_size, seed + 202
        )
        self.streams = {
            "synthetic": self._stream(self.synthetic_train_examples),
            "gsm_train": self._stream(self.train_examples),
            "mixed": self._stream(
                self.synthetic_train_examples[: len(self.train_examples)] + self.train_examples
            ),
            "validation": self._stream(self.validation_examples),
        }

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def stoi(self) -> dict[str, int]:
        return self.tokenizer.stoi

    @property
    def itos(self) -> dict[int, str]:
        return self.tokenizer.itos

    def _stream(self, examples: Iterable[MathExample]) -> torch.Tensor:
        values: list[int] = []
        for example in examples:
            values.extend(self.tokenizer.encode(example.text, bos=True, eos=True))
        if len(values) <= self.context_length + 1:
            raise ValueError("math stream is shorter than the requested context")
        return torch.tensor(values, dtype=torch.long)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.streams[split]
        starts = torch.randint(
            0, len(source) - self.context_length - 1, (batch_size,), generator=generator
        )
        x = torch.stack([source[index : index + self.context_length] for index in starts])
        y = torch.stack([source[index + 1 : index + self.context_length + 1] for index in starts])
        return x.to(device), y.to(device)

    def eligible_examples(
        self, split: str, max_prompt_tokens: int
    ) -> list[MathExample]:
        examples = {
            "train": self.train_examples,
            "validation": self.validation_examples,
            "test": self.test_examples,
            "synthetic": self.synthetic_validation_examples,
        }[split]
        return [
            example for example in examples
            if len(self.tokenizer.encode(example.prompt, bos=True)) <= max_prompt_tokens
        ]
