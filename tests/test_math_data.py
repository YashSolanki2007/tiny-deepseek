from __future__ import annotations

import torch
import pytest

from math_data import (
    ByteTokenizer,
    extract_tagged_answer,
    generate_synthetic_examples,
    normalize_numeric_answer,
    repetition_rate,
)
from math_training_utils import score_math_completions


def test_byte_tokenizer_round_trip_handles_unseen_unicode() -> None:
    tokenizer = ByteTokenizer()
    text = "Compute 17 × 4 = 68."
    encoded = tokenizer.encode(text, bos=True, eos=True)
    assert max(encoded) < tokenizer.vocab_size
    assert tokenizer.decode(encoded, stop_at_eos=True) == text


def test_numeric_answer_extraction_normalizes_equivalent_forms() -> None:
    assert extract_tagged_answer("work <answer>$1,072.0</answer>") == "1072"
    assert normalize_numeric_answer("-2.5000") == "-2.5"
    assert extract_tagged_answer("there is no tagged result") is None


def test_synthetic_curriculum_is_deterministic_and_verified() -> None:
    first = generate_synthetic_examples(20, seed=8)
    second = generate_synthetic_examples(20, seed=8)
    assert first == second
    assert all(normalize_numeric_answer(example.answer) is not None for example in first)


def test_math_reward_prefers_exact_parseable_nonrepetitive_answer() -> None:
    completions = [
        " <answer>72</answer>",
        " <answer>70</answer>",
        " answer answer answer answer answer answer",
    ]
    rewards, parts, predictions = score_math_completions(
        completions, "72", torch.tensor([0.6, 0.6, 0.8]), compute_target=0.7
    )
    assert predictions == ["72", "70", None]
    assert rewards[0] > rewards[1] > rewards[2]
    assert parts["exact_reward"] == pytest.approx(1 / 3)
    assert repetition_rate(completions[2]) > repetition_rate(completions[0])
