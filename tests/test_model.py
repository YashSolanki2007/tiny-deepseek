from __future__ import annotations

import torch

from config import ModelConfig
from model import MultiHeadLatentAttention, TransformerBlock, apply_block_gate, build_model
from utils import estimate_dense_block_flops, estimate_mor_flops, estimate_mor_skip_flops


def tiny_config(**overrides) -> ModelConfig:
    values = dict(
        vocab_size=19, context_length=8, d_model=16, n_heads=4,
        n_layers=3, d_ff=32, dropout=0.0, router_dim=8,
        model_type="sparse",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_zero_gate_is_exact_identity() -> None:
    x, candidate = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    torch.testing.assert_close(
        apply_block_gate(x, candidate, torch.zeros(2, 8, 1)), x, rtol=0, atol=0
    )


def test_one_gate_equals_normal_block_output() -> None:
    block = TransformerBlock(tiny_config()).eval()
    x = torch.randn(2, 8, 16)
    candidate = block(x)
    torch.testing.assert_close(
        apply_block_gate(x, candidate, torch.ones(2, 8, 1)), candidate, rtol=0, atol=0
    )


def test_dense_model_has_no_routing_tensors() -> None:
    output = build_model(tiny_config(model_type="dense"))(torch.randint(0, 19, (2, 8)))
    assert output.soft_gates is None
    assert output.hard_gates is None


def test_sparse_output_shapes_and_hard_binary_forward() -> None:
    for router_type in ("linear", "gru"):
        output = build_model(tiny_config(router_type=router_type))(
            torch.randint(0, 19, (2, 8)), routing_mode="gumbel"
        )
        assert output.soft_gates.shape == (2, 8, 3)
        assert output.hard_gates.shape == (2, 8, 3)
        assert output.route_logits.shape == (2, 8, 3, 2)
        assert set(output.hard_gates.detach().unique().tolist()) <= {0.0, 1.0}


def test_sparse_moe_mtp_outputs_and_expert_accounting() -> None:
    config = tiny_config(
        model_type="sparse_moe_mtp", router_type="linear",
        moe_num_experts=10, moe_top_k=2,
    )
    model = build_model(config)
    token_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    targets = torch.randint(0, config.vocab_size, token_ids.shape)
    output = model(token_ids, targets, routing_mode="gumbel")
    assert output.mtp_logits.shape == (2, config.context_length - 1, config.vocab_size)
    assert output.expert_utilization.shape == (config.n_layers, 10)
    torch.testing.assert_close(output.expert_utilization.sum(dim=-1), torch.ones(config.n_layers))
    assert torch.isfinite(output.mtp_loss)
    assert torch.isfinite(output.moe_aux_loss)


def test_sparse_moe_mtp_can_discard_training_only_mtp_module() -> None:
    config = tiny_config(model_type="sparse_moe_mtp", router_type="linear")
    model = build_model(config).eval()
    token_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    output = model(token_ids, token_ids, routing_mode="greedy", compute_mtp=False)
    assert output.mtp_logits is None
    assert output.mtp_loss is None


def test_mla_variant_replaces_absolute_positions_and_runs_mtp() -> None:
    config = tiny_config(
        model_type="sparse_moe_mtp_mla", router_type="linear",
        moe_num_experts=10, moe_top_k=2,
    )
    model = build_model(config).eval()
    assert model.position_embedding is None
    assert isinstance(model.blocks[0].attn, MultiHeadLatentAttention)
    assert model.config.position_embedding_type == "rope"
    token_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    output = model(token_ids, token_ids, routing_mode="greedy")
    assert output.logits.shape == (2, config.context_length, config.vocab_size)
    assert output.mtp_logits.shape == (2, config.context_length - 1, config.vocab_size)
    assert torch.isfinite(output.logits).all()


def test_mla_sparse_selected_queries_match_dense_masked_semantics() -> None:
    torch.manual_seed(91)
    config = tiny_config(model_type="sparse_moe_mtp_mla", router_type="linear")
    block = TransformerBlock(config).eval()
    x = torch.randn(2, config.context_length, config.d_model)
    active = torch.tensor(
        [
            [False, True, False, True, True, False, True, False],
            [True, False, False, True, False, True, False, True],
        ]
    )
    dense_masked = apply_block_gate(x, block(x), active.unsqueeze(-1).float())
    sparse = block.forward_selected(x, active)
    torch.testing.assert_close(sparse, dense_masked, rtol=2e-5, atol=2e-6)


def test_budget_controller_hits_exact_depth_when_fully_enabled() -> None:
    model = build_model(tiny_config(router_type="linear")).eval()
    token_ids = torch.randint(0, 19, (3, 8))
    targets = torch.tensor([1.0, 2.0, 3.0])
    output = model(
        token_ids,
        routing_mode="budget",
        target_depths=targets,
        exploration_epsilon=torch.ones(3),
    )
    torch.testing.assert_close(output.hard_gates.sum(dim=-1), targets[:, None].expand(3, 8))
    assert torch.isfinite(output.behavior_log_probs).all()


def test_budget_behavior_matches_policy_when_epsilon_is_zero() -> None:
    model = build_model(tiny_config(router_type="linear")).eval()
    token_ids = torch.randint(0, 19, (2, 8))
    output = model(
        token_ids,
        routing_mode="budget",
        target_depths=torch.tensor([1.0, 2.0]),
        exploration_epsilon=0.0,
    )
    torch.testing.assert_close(output.behavior_log_probs, output.action_log_probs)


def test_initial_soft_execute_probability_is_about_ninety_percent() -> None:
    for router_type in ("linear", "gru"):
        model = build_model(tiny_config(router_type=router_type)).eval()
        output = model(torch.randint(0, 19, (8, 8)), routing_mode="greedy")
        assert 0.88 < output.soft_gates.mean().item() < 0.92
        assert output.hard_gates.mean().item() == 1.0


def test_paper_router_uses_pre_normalized_layer_input_without_bias() -> None:
    config = tiny_config(
        router_type="linear", router_input_norm=True, router_bias=False,
        initial_execute_probability=0.5, paper_reproduction=True,
    )
    model = build_model(config).eval()
    token_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    embedded = model.embed(token_ids)
    expected = model.router.gate_heads[0](model.blocks[0].ln1(embedded))
    output = model(token_ids, routing_mode="greedy")
    torch.testing.assert_close(output.route_logits[:, :, 0], expected)
    assert model.router.gate_heads[0].bias is None


def test_paper_router_uses_standard_model_initialization_scale() -> None:
    torch.manual_seed(22)
    model = build_model(
        tiny_config(
            router_type="linear", router_bias=False,
            initial_execute_probability=0.5, paper_reproduction=True,
        )
    )
    std = model.router.gate_heads[0].weight.std().item()
    assert 0.012 < std < 0.028


def test_skipped_token_remains_context_for_active_attention_query() -> None:
    torch.manual_seed(12)
    block = TransformerBlock(tiny_config()).eval()
    x = torch.randn(1, 2, 16)
    gate = torch.tensor([[[0.0], [1.0]]])
    output = apply_block_gate(x, block(x), gate)
    changed_context = x.clone()
    changed_context[:, 0, 0] += 3.0
    changed_output = apply_block_gate(changed_context, block(changed_context), gate)
    torch.testing.assert_close(output[:, 0], x[:, 0], rtol=0, atol=0)
    assert not torch.allclose(output[:, 1], changed_output[:, 1])


def test_sparse_greedy_block_matches_dense_masked_semantics() -> None:
    torch.manual_seed(31)
    block = TransformerBlock(tiny_config()).eval()
    x = torch.randn(2, 8, 16)
    active = torch.tensor(
        [
            [False, True, False, True, True, False, True, False],
            [True, False, False, True, False, True, False, True],
        ]
    )
    dense_masked = apply_block_gate(x, block(x), active.unsqueeze(-1).float())
    sparse = block.forward_selected(x, active)
    torch.testing.assert_close(sparse, dense_masked, rtol=1e-5, atol=1e-6)


def test_mor_middle_cycle_has_four_unique_blocks_and_eight_effective_layers() -> None:
    config = tiny_config(
        model_type="mor",
        router_type="linear",
        n_layers=8,
        recursion_steps=3,
        recursion_block_layers=2,
        mor_reproduction=True,
    )
    model = build_model(config)
    assert len(model.blocks) == 4
    output = model(
        torch.randint(0, config.vocab_size, (2, config.context_length)),
        routing_mode="topk",
    )
    assert output.hard_gates.shape == (2, config.context_length, 8)
    assert output.actions.shape == (2, config.context_length, 3)
    torch.testing.assert_close(
        output.recursion_utilization,
        torch.tensor([1.0, 5 / 8, 2 / 8]),
    )
    assert torch.isfinite(output.mor_aux_loss)


def test_mor_budget_anchor_and_minimum_recursion_are_exact() -> None:
    config = tiny_config(
        model_type="mor",
        router_type="linear",
        n_layers=8,
        recursion_steps=3,
        recursion_block_layers=2,
    )
    model = build_model(config).eval()
    output = model(
        torch.randint(0, config.vocab_size, (2, config.context_length)),
        routing_mode="budget",
        target_recursions=torch.tensor([1, 3]),
        exploration_epsilon=torch.ones(2),
    )
    torch.testing.assert_close(
        output.hard_gates.sum(dim=-1),
        torch.tensor([[4.0], [8.0]]).expand(2, config.context_length),
    )
    assert torch.isfinite(output.behavior_log_probs).all()


def test_mor_flops_equal_dense_when_all_effective_layers_execute() -> None:
    config = tiny_config(
        model_type="mor",
        router_type="linear",
        n_layers=8,
        recursion_steps=3,
        recursion_block_layers=2,
    )
    mor = estimate_mor_flops(config, config.context_length, [1.0, 1.0, 1.0])
    dense = estimate_dense_block_flops(config, config.context_length)
    # MoR adds only its lightweight scalar routers beyond the dense blocks.
    assert dense < mor < dense * 1.01


def test_recursion_wise_attention_saves_more_than_linear_depth_fraction() -> None:
    config = tiny_config(
        model_type="mor",
        router_type="linear",
        n_layers=8,
        recursion_steps=3,
        recursion_block_layers=2,
    )
    routed = estimate_mor_flops(config, config.context_length, [1.0, 0.625, 0.25])
    full = estimate_dense_block_flops(config, config.context_length)
    assert routed < full


def test_mor_skip_has_independent_inner_actions_and_combined_gates() -> None:
    config = tiny_config(
        model_type="mor_skip", router_type="linear", n_layers=8,
        recursion_steps=3, recursion_block_layers=2,
        initial_execute_probability=0.9, router_bias=True,
    )
    model = build_model(config).eval()
    output = model(torch.randint(0, 19, (2, 8)), routing_mode="greedy")
    assert len(model.blocks) == 4
    assert len(model.skip_router.gate_heads) == 6
    assert output.hard_gates.shape == (2, 8, 8)
    assert output.actions.shape == (2, 8, 6)
    assert output.mor_actions.shape == (2, 8, 3)
    assert output.route_logits.shape == (2, 8, 6, 2)
    for inner in range(6):
        recursion = inner // 2
        assert torch.equal(
            output.routing_decision_mask[..., inner],
            output.mor_actions[..., recursion].bool(),
        )
        assert not bool(
            output.skip_hard_gates[..., inner][
                ~output.routing_decision_mask[..., inner]
            ].any()
        )


def test_mor_skip_budget_zero_and_execute_all_eligible_are_exact() -> None:
    config = tiny_config(
        model_type="mor_skip", router_type="linear", n_layers=8,
        recursion_steps=3, recursion_block_layers=2,
        initial_execute_probability=0.9, router_bias=True,
    )
    model = build_model(config).eval()
    output = model(
        torch.randint(0, 19, (2, 8)),
        routing_mode="budget",
        target_skip_densities=torch.tensor([0.0, 1.0]),
        exploration_epsilon=torch.ones(2),
    )
    assert output.skip_hard_gates[0].sum().item() == 0
    assert output.hard_gates[0].sum(dim=-1).eq(2).all()
    torch.testing.assert_close(
        output.skip_hard_gates[1],
        output.routing_decision_mask[1].float(),
    )
    assert torch.isfinite(
        output.behavior_log_probs[output.routing_decision_mask]
    ).all()


def test_mor_skip_flops_reduce_to_mor_when_all_inner_gates_execute() -> None:
    config = tiny_config(
        model_type="mor_skip", router_type="linear", n_layers=8,
        recursion_steps=3, recursion_block_layers=2,
    )
    recursion = [1.0, 0.625, 0.25]
    combined = [value for value in recursion for _ in range(2)]
    hybrid = estimate_mor_skip_flops(
        config, config.context_length, recursion, combined
    )
    mor = estimate_mor_flops(config, config.context_length, recursion)
    assert mor < hybrid < mor * 1.01


def test_mor_skip_execute_all_matches_the_same_mor_backbone() -> None:
    torch.manual_seed(91)
    mor_config = tiny_config(
        model_type="mor", router_type="linear", n_layers=8,
        recursion_steps=3, recursion_block_layers=2, dropout=0.0,
    )
    mor = build_model(mor_config).eval()
    hybrid_values = mor_config.to_dict()
    hybrid_values.update(
        model_type="mor_skip", router_bias=True,
        initial_execute_probability=0.9,
    )
    hybrid = build_model(ModelConfig.from_dict(hybrid_values)).eval()
    missing, unexpected = hybrid.load_state_dict(mor.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(key.startswith("skip_router.") for key in missing)
    with torch.no_grad():
        for head in hybrid.skip_router.gate_heads:
            head.weight.zero_()
            head.bias.copy_(torch.tensor([-10.0, 10.0]))
    token_ids = torch.randint(0, mor_config.vocab_size, (2, 8))
    expected = mor(token_ids, routing_mode="greedy")
    actual = hybrid(token_ids, routing_mode="greedy")
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.hard_gates, expected.hard_gates)
