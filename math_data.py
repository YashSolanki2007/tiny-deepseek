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


class MathBPETokenizer:
    """Train-only byte-level BPE persisted as a standard Tokenizers JSON file."""

    special_tokens = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(
        self,
        path: str | Path,
        training_texts: Iterable[str],
        vocab_size: int = 4096,
    ) -> None:
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as error:
            raise RuntimeError(
                "BPE math training requires `pip install tokenizers`"
            ) from error
        self.path = Path(path)
        if self.path.exists():
            self.backend = Tokenizer.from_file(str(self.path))
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.backend = Tokenizer(models.BPE(unk_token="<unk>"))
            self.backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.backend.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=list(self.special_tokens),
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False,
            )
            self.backend.train_from_iterator(training_texts, trainer=trainer)
            self.backend.save(str(self.path))
        self.pad_id = self._required_id("<pad>")
        self.bos_id = self._required_id("<bos>")
        self.eos_id = self._required_id("<eos>")

    def _required_id(self, token: str) -> int:
        value = self.backend.token_to_id(token)
        if value is None:
            raise ValueError(f"BPE tokenizer is missing required token {token}")
        return value

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        values = self.backend.encode(text, add_special_tokens=False).ids
        if bos:
            values.insert(0, self.bos_id)
        if eos:
            values.append(self.eos_id)
        return values

    def decode(self, token_ids: Iterable[int], *, stop_at_eos: bool = False) -> str:
        values = [int(token) for token in token_ids]
        if stop_at_eos and self.eos_id in values:
            values = values[: values.index(self.eos_id)]
        return self.backend.decode(values, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    @property
    def stoi(self) -> dict[str, int]:
        return self.backend.get_vocab()

    @property
    def itos(self) -> dict[int, str]:
        return {index: token for token, index in self.stoi.items()}


@dataclass(frozen=True)
class MathExample:
    question: str
    reasoning: str
    answer: str
    difficulty: str = "unknown"
    operation: str = "mixed"

    @property
    def prompt(self) -> str:
        return f"Question: {self.question.strip()}\nResponse:\n"

    @property
    def response(self) -> str:
        reasoning = self.reasoning.strip()
        prefix = f"Reasoning: {reasoning}\n" if reasoning else ""
        return f"{prefix}<answer>{self.answer}</answer>"

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
            calculation_count = len(CALCULATION_PATTERN.findall(record["answer"]))
            calculations = " ".join(CALCULATION_PATTERN.findall(record["answer"]))
            operators = {
                name for symbol, name in (
                    ("+", "addition"), ("-", "subtraction"),
                    ("*", "multiplication"), ("/", "division"),
                ) if symbol in calculations
            }
            operation = next(iter(operators)) if len(operators) == 1 else "mixed"
            difficulty = (
                "easy" if calculation_count <= 1
                else "medium" if calculation_count <= 3
                else "hard"
            )
            reasoning = GSM_ANSWER_PATTERN.sub("", record["answer"]).strip()
            reasoning = CALCULATION_PATTERN.sub("", reasoning)
            example = MathExample(
                record["question"], reasoning, answer, difficulty, operation
            )
            if example.question.strip() and example.reasoning.strip():
                examples.append(example)
    return examples


def generate_synthetic_examples(count: int, seed: int) -> list[MathExample]:
    rng = random.Random(seed)
    names = ["Ava", "Ben", "Chloe", "Dev", "Emma", "Finn", "Grace", "Hugo"]
    objects = ["apples", "books", "coins", "marbles", "pencils", "tickets"]
    examples: list[MathExample] = []
    for index in range(count):
        name, item = rng.choice(names), rng.choice(objects)
        kind = index % 10
        if kind == 0:
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            question = f"{name} has {a} {item} and gets {b} more. How many {item} are there?"
            answer = a + b
            reasoning = f"Add the two amounts: {a} + {b} = {answer}."
            difficulty, operation = "easy", "addition"
        elif kind == 1:
            b = rng.randint(2, 60)
            answer = rng.randint(2, 80)
            a = b + answer
            question = f"{name} has {a} {item} and gives away {b}. How many remain?"
            reasoning = f"Subtract what was given away: {a} - {b} = {answer}."
            difficulty, operation = "easy", "subtraction"
        elif kind == 2:
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            answer = a * b
            question = f"There are {a} boxes with {b} {item} in each. How many {item} are there?"
            reasoning = f"Multiply boxes by items per box: {a} * {b} = {answer}."
            difficulty, operation = "easy", "multiplication"
        elif kind == 3:
            groups, answer = rng.randint(2, 15), rng.randint(2, 15)
            total = groups * answer
            question = f"{total} {item} are shared equally among {groups} people. How many does each get?"
            reasoning = f"Divide the total equally: {total} / {groups} = {answer}."
            difficulty, operation = "easy", "division"
        elif kind == 4:
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
            difficulty, operation = "medium", "mixed"
        elif kind == 5:
            groups, each, extra = rng.randint(2, 12), rng.randint(2, 15), rng.randint(2, 20)
            subtotal, answer = groups * each, groups * each + extra
            question = (
                f"{name} fills {groups} boxes with {each} {item} each and has "
                f"{extra} loose {item}. How many {item} are there altogether?"
            )
            reasoning = (
                f"First multiply: {groups} * {each} = {subtotal}. Then add the "
                f"loose items: {subtotal} + {extra} = {answer}."
            )
            difficulty, operation = "medium", "mixed"
        elif kind == 6:
            price, quantity, fee = rng.randint(2, 20), rng.randint(2, 12), rng.randint(2, 15)
            subtotal, answer = price * quantity, price * quantity + fee
            question = (
                f"A ticket costs ${price}. {name} buys {quantity} tickets and pays "
                f"a ${fee} booking fee. What is the total cost?"
            )
            reasoning = (
                f"The tickets cost {price} * {quantity} = {subtotal}. Add the fee: "
                f"{subtotal} + {fee} = {answer}."
            )
            difficulty, operation = "medium", "mixed"
        elif kind == 7:
            percent = rng.choice([10, 20, 25, 50])
            base_unit = 100 // percent
            total = base_unit * rng.randint(2, 40)
            answer = total * percent // 100
            question = f"What is {percent}% of {total}?"
            reasoning = (
                f"Convert the percent to a fraction and multiply: "
                f"{total} * {percent} / 100 = {answer}."
            )
            difficulty, operation = "hard", "mixed"
        elif kind == 8:
            groups, per_group, used = rng.randint(3, 12), rng.randint(4, 18), rng.randint(2, 15)
            total = groups * per_group
            answer = total - used
            question = (
                f"There are {groups} packs of {per_group} {item}. After {used} are "
                f"used, how many {item} remain?"
            )
            reasoning = (
                f"First find the total: {groups} * {per_group} = {total}. Then "
                f"subtract the used items: {total} - {used} = {answer}."
            )
            difficulty, operation = "hard", "mixed"
        else:
            days, daily, bonus, spent = (
                rng.randint(3, 10), rng.randint(4, 20),
                rng.randint(3, 30), rng.randint(2, 20),
            )
            earned = days * daily
            subtotal = earned + bonus
            answer = subtotal - spent
            question = (
                f"{name} earns ${daily} each day for {days} days, receives a ${bonus} "
                f"bonus, and spends ${spent}. How much money remains?"
            )
            reasoning = (
                f"Daily earnings are {days} * {daily} = {earned}. With the bonus, "
                f"{earned} + {bonus} = {subtotal}. After spending, "
                f"{subtotal} - {spent} = {answer}."
            )
            difficulty, operation = "hard", "mixed"
        examples.append(
            MathExample(question, reasoning, str(answer), difficulty, operation)
        )
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
        tokenizer_type: str = "byte",
        bpe_vocab_size: int = 4096,
        bpe_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        train_path = self.data_dir / "train.jsonl"
        test_path = self.data_dir / "test.jsonl"
        if download:
            _download_jsonl(train_path, "train")
            _download_jsonl(test_path, "test")
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("GSM8K train.jsonl and test.jsonl are required")
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
        self.context_length = context_length
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == "byte":
            self.tokenizer = ByteTokenizer()
        elif tokenizer_type == "bpe":
            tokenizer_path = Path(bpe_path) if bpe_path else (
                self.data_dir / f"math_bpe_{bpe_vocab_size}.json"
            )
            tokenizer_training_examples = (
                self.train_examples + self.synthetic_train_examples
            )
            self.tokenizer = MathBPETokenizer(
                tokenizer_path,
                (example.text for example in tokenizer_training_examples),
                vocab_size=bpe_vocab_size,
            )
        else:
            raise ValueError("tokenizer_type must be 'byte' or 'bpe'")
        mixed_examples = (
            self.synthetic_train_examples[: len(self.train_examples)]
            + self.train_examples
        )
        synthetic_by_difficulty = {
            difficulty: [
                example for example in self.synthetic_train_examples
                if example.difficulty == difficulty
            ]
            for difficulty in ("easy", "medium", "hard")
        }
        self.example_splits = {
            "synthetic": self.synthetic_train_examples,
            **{
                f"synthetic_{difficulty}": examples
                for difficulty, examples in synthetic_by_difficulty.items()
            },
            "mixed": mixed_examples,
            "gsm_train": self.train_examples,
            "validation": self.validation_examples,
        }
        self._complete_cache: dict[str, list[tuple[list[int], int]]] = {}
        self.streams = {
            "synthetic": self._stream(self.synthetic_train_examples),
            **{
                f"synthetic_{difficulty}": self._stream(examples)
                for difficulty, examples in synthetic_by_difficulty.items()
            },
            "gsm_train": self._stream(self.train_examples),
            "mixed": self._stream(mixed_examples),
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

    def _complete_sequences(self, split: str) -> list[tuple[list[int], int]]:
        """Return complete examples and prompt lengths that fit one context row."""
        if split not in self.example_splits:
            raise KeyError(f"unknown supervised split: {split}")
        if split not in self._complete_cache:
            sequences = []
            for example in self.example_splits[split]:
                prompt = self.tokenizer.encode(example.prompt, bos=True)
                complete = self.tokenizer.encode(example.text, bos=True, eos=True)
                if len(complete) - 1 <= self.context_length:
                    sequences.append((complete, len(prompt)))
            if not sequences:
                raise ValueError(f"no complete {split} examples fit the context")
            # Sorting enables length-bucketed sampling below. Sampling a random
            # anchor preserves uniform example coverage while grouping similarly
            # sized rows, substantially reducing padding at larger batch sizes.
            self._complete_cache[split] = sorted(
                sequences, key=lambda item: len(item[0])
            )
        return self._complete_cache[split]

    def get_supervised_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        max_batch_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample complete rows, padding inputs and masking prompt/padding targets."""
        sequences = self._complete_sequences(split)
        bucket_size = min(len(sequences), max(batch_size * 8, batch_size))
        anchor = int(
            torch.randint(0, len(sequences), (1,), generator=generator).item()
        )
        bucket_start = (anchor // bucket_size) * bucket_size
        bucket_end = min(bucket_start + bucket_size, len(sequences))
        if max_batch_tokens is not None:
            if max_batch_tokens < 1:
                raise ValueError("max_batch_tokens must be positive")
            bucket_longest = len(sequences[bucket_end - 1][0]) - 1
            aligned_longest = min(
                self.context_length, ((bucket_longest + 31) // 32) * 32
            )
            batch_size = min(
                batch_size, max(1, max_batch_tokens // aligned_longest)
            )
        selected = torch.randint(
            bucket_start, bucket_end, (batch_size,), generator=generator
        )
        selected_sequences = [sequences[index] for index in selected.tolist()]
        longest = max(len(complete) - 1 for complete, _ in selected_sequences)
        # A small set of aligned shapes avoids spending most compute on padding while
        # remaining friendlier to accelerator kernel caches than arbitrary lengths.
        padded_length = min(self.context_length, ((longest + 31) // 32) * 32)
        inputs = torch.full(
            (batch_size, padded_length), self.tokenizer.pad_id, dtype=torch.long
        )
        targets = torch.full(
            (batch_size, padded_length), -100, dtype=torch.long
        )
        for row, (complete, prompt_length) in enumerate(selected_sequences):
            x = complete[:-1]
            y = complete[1:]
            length = len(x)
            inputs[row, :length] = torch.tensor(x, dtype=torch.long)
            response_target_start = max(prompt_length - 1, 0)
            targets[row, response_target_start:length] = torch.tensor(
                y[response_target_start:], dtype=torch.long
            )
        return inputs.to(device), targets.to(device)

    def complete_example_fraction(self, split: str) -> float:
        return len(self._complete_sequences(split)) / len(self.example_splits[split])

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
