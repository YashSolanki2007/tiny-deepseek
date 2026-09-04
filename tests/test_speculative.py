from __future__ import annotations

import torch

from tiny_deepseek.core.config import ModelConfig
from tiny_deepseek.core.model import SparseMoEMTPTransformer, build_model
from tiny_deepseek.cli.speculative_decode import (
    residual_distribution,
    sampling_distribution,
    speculative_generate,
)


def tiny_mla_model() -> SparseMoEMTPTransformer:
    config = ModelConfig(
        vocab_size=19,
        context_length=16,
        d_model=16,
        n_heads=4,
        n_layers=3,
        d_ff=32,
        dropout=0.0,
        router_dim=8,
        router_type="linear",
        model_type="sparse_moe_mtp_mla",
        moe_num_experts=4,
        moe_top_k=2,
    )
    model = build_model(config).eval()
    assert isinstance(model, SparseMoEMTPTransformer)
    return model


def test_mtp_draft_logits_uses_one_block_and_returns_next_distribution() -> None:
    model = tiny_mla_model()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    target = model(token_ids, compute_mtp=False)
    draft = model.mtp_draft_logits(
        token_ids,
        target.hidden_states,
        torch.randint(0, model.config.vocab_size, (2, 1)),
    )
    assert draft.shape == (2, model.config.vocab_size)
    assert torch.isfinite(draft).all()


def test_residual_distribution_is_normalized_positive_target_minus_draft() -> None:
    target = torch.tensor([[0.6, 0.4], [0.25, 0.75]])
    draft = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    residual = residual_distribution(target, draft)
    torch.testing.assert_close(residual.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(residual, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_sampling_distribution_greedy_and_top_k() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    torch.testing.assert_close(
        sampling_distribution(logits, temperature=0.0, top_k=None),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
    probability = sampling_distribution(logits, temperature=1.0, top_k=2)
    assert probability[0, 0] == 0
    torch.testing.assert_close(probability.sum(dim=-1), torch.ones(1))


def test_speculative_generation_returns_requested_length_and_accounting() -> None:
    torch.manual_seed(7)
    model = tiny_mla_model()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    generated, stats = speculative_generate(
        model, prompt, max_new_tokens=8, temperature=0.8, top_k=10
    )
    assert generated.shape == (1, 12)
    assert stats["tokens_generated"] == 8
    assert stats["draft_calls"] == (
        stats["accepted_drafts"] + stats["rejected_drafts"]
    )
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
    assert stats["target_forward_calls"] >= 1
