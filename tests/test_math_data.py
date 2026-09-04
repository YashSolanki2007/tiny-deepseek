from __future__ import annotations

import torch
import pytest

from tiny_deepseek.data.math import (
    ByteTokenizer,
    MathData,
    MathBPETokenizer,
    MathExample,
    extract_tagged_answer,
    generate_synthetic_examples,
    normalize_numeric_answer,
)
from tiny_deepseek.training.math_utils import score_math_completions


def test_byte_tokenizer_round_trip_handles_unseen_unicode() -> None:
    tokenizer = ByteTokenizer()
    text = "Compute 17 × 4 = 68."
    encoded = tokenizer.encode(text, bos=True, eos=True)
    assert max(encoded) < tokenizer.vocab_size
    assert tokenizer.decode(encoded, stop_at_eos=True) == text


def test_bpe_tokenizer_round_trip_and_persistence(tmp_path) -> None:
    tokenizer_path = tmp_path / "math_bpe.json"
    corpus = [
        "Question: What is 2 + 3?\nResponse:\nReasoning: 2 + 3 = 5.\n<answer>5</answer>",
        "Question: What is 8 - 2?\nResponse:\nReasoning: 8 - 2 = 6.\n<answer>6</answer>",
    ]
    tokenizer = MathBPETokenizer(tokenizer_path, corpus, vocab_size=300)
    text = "Compute 17 × 4 = 68."
    encoded = tokenizer.encode(text, bos=True, eos=True)
    assert tokenizer.decode(encoded, stop_at_eos=True) == text
    reloaded = MathBPETokenizer(tokenizer_path, (), vocab_size=300)
    assert reloaded.encode(text, bos=True, eos=True) == encoded


def test_math_example_places_reasoning_before_answer() -> None:
    example = MathExample("What is 2 + 3?", "2 + 3 = 5.", "5")
    assert example.response.index("Reasoning:") < example.response.index("<answer>")


def test_complete_example_batch_masks_prompt_and_padding() -> None:
    data = MathData.__new__(MathData)
    data.context_length = 128
    data.tokenizer = ByteTokenizer()
    example = MathExample("What is 2 + 3?", "2 + 3 = 5.", "5")
    data.example_splits = {"synthetic": [example]}
    data._complete_cache = {}
    inputs, targets = data.get_supervised_batch("synthetic", 1, "cpu")
    prompt_length = len(data.tokenizer.encode(example.prompt, bos=True))
    assert targets[0, : prompt_length - 1].eq(-100).all()
    assert targets[0, prompt_length - 1 :].ne(-100).any()
    assert targets[0, len(data.tokenizer.encode(example.text, bos=True, eos=True)) - 1 :].eq(
        -100
    ).all()
    assert data.tokenizer.decode(inputs[0].tolist(), stop_at_eos=True) == example.text


def test_complete_batch_respects_token_budget_with_adaptive_microbatch() -> None:
    data = MathData.__new__(MathData)
    data.context_length = 256
    data.tokenizer = ByteTokenizer()
    data.example_splits = {
        "synthetic": [
            MathExample(f"What is {index} + 1?", "x " * (index + 4), str(index + 1))
            for index in range(16)
        ]
    }
    data._complete_cache = {}
    inputs, targets = data.get_supervised_batch(
        "synthetic", 8, "cpu", max_batch_tokens=256
    )
    assert inputs.shape == targets.shape
    assert inputs.numel() <= 256
    assert 1 <= inputs.shape[0] <= 8


def test_numeric_answer_extraction_normalizes_equivalent_forms() -> None:
    assert extract_tagged_answer("work <answer>$1,072.0</answer>") == "1072"
    assert normalize_numeric_answer("-2.5000") == "-2.5"
    assert extract_tagged_answer("there is no tagged result") is None


def test_synthetic_curriculum_is_deterministic_and_verified() -> None:
    first = generate_synthetic_examples(20, seed=8)
    second = generate_synthetic_examples(20, seed=8)
    assert first == second
    assert all(normalize_numeric_answer(example.answer) is not None for example in first)
    assert {example.difficulty for example in first} == {"easy", "medium", "hard"}
    assert all(example.question.strip() and example.reasoning.strip() for example in first)


def test_math_reward_is_strict_binary_exact_correctness() -> None:
    completions = [
        " <answer>72</answer>",
        " <answer>70</answer>",
        " answer answer answer answer answer answer",
    ]
    rewards, parts, predictions = score_math_completions(
        completions, "72", torch.device("cpu")
    )
    assert predictions == ["72", "70", None]
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0]))
    assert parts["exact_reward"] == pytest.approx(1 / 3)
