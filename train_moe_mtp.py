"""Warm-start SkipLayer+MoE+MTP from an existing SkipLayer+GRPO checkpoint."""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from config import ModelConfig
from data import TinyShakespeareData
from evaluation import evaluate_model
from logging_utils import StructuredLogger
from model import (
    MLP, MultiHeadLatentAttention, SparseMoE, SparseMoEMTPTransformer, build_model,
)
from utils import (
    estimate_dense_block_flops,
    estimate_mla_skiplayer_moe_mtp_flops,
    estimate_sparse_moe_mtp_flops,
    gradient_norm,
    load_checkpoint,
    perplexity,
    routing_metrics,
    save_checkpoint,
    select_device,
    set_seed,
    synchronize_device,
    top1_accuracy,
    write_json,
)


def initialize_from_skiplayer(
    model: SparseMoEMTPTransformer, source: torch.nn.Module
) -> dict[str, Any]:
    """Copy the backbone and split each dense FFN across complementary experts."""
    source_state = source.state_dict()
    destination = model.state_dict()
    copied = {
        key: value for key, value in source_state.items()
        if key in destination and destination[key].shape == value.shape
    }
    model.load_state_dict(copied, strict=False)
    for source_block, destination_block in zip(source.blocks, model.blocks):
        if not isinstance(source_block.mlp, MLP) or not isinstance(
            destination_block.mlp, SparseMoE
        ):
            raise TypeError("Expected dense source FFNs and sparse destination FFNs")
        first = source_block.mlp.network[0]
        second = source_block.mlp.network[2]
        expert_width = model.config.moe_expert_d_ff
        with torch.no_grad():
            for expert_index, expert in enumerate(destination_block.mlp.experts):
                partition = expert_index % model.config.moe_top_k
                start = partition * expert_width
                end = start + expert_width
                expert.network[0].weight.copy_(first.weight[start:end])
                expert.network[0].bias.copy_(first.bias[start:end])
                expert.network[2].weight.copy_(
                    second.weight[:, start:end] * model.config.moe_top_k
                )
                expert.network[2].bias.copy_(second.bias)
            destination_block.mlp.router.weight.zero_()
            for expert_index in range(model.config.moe_num_experts):
                pair_index = expert_index // model.config.moe_top_k
                destination_block.mlp.selection_bias[expert_index] = -1e-6 * pair_index
            if isinstance(destination_block.attn, MultiHeadLatentAttention):
                d = model.config.d_model
                old_q, old_k, old_v = source_block.attn.qkv.weight.split(d, dim=0)
                destination_block.attn.q_proj.weight.copy_(old_q)
                old_k_heads = old_k.view(model.config.n_heads, d // model.config.n_heads, d)
                k_nope = old_k_heads[:, : destination_block.attn.nope_dim].reshape(-1, d)
                k_rope = old_k_heads[:, destination_block.attn.nope_dim :].mean(dim=0)
                target = torch.cat((k_nope, old_v), dim=0)
                u, singular, vh = torch.linalg.svd(
                    target.detach().float().cpu(), full_matrices=False
                )
                rank = destination_block.attn.kv_rank
                root = singular[:rank].sqrt()
                destination_block.attn.kv_down.weight[:rank].copy_(
                    (root[:, None] * vh[:rank]).to(
                        device=first.weight.device, dtype=first.weight.dtype
                    )
                )
                destination_block.attn.kv_down.weight[rank:].copy_(k_rope)
                destination_block.attn.kv_up.weight.copy_(
                    (u[:, :rank] * root[None, :]).to(
                        device=first.weight.device, dtype=first.weight.dtype
                    )
                )
    return {
        "compatible_tensors_copied": len(copied),
        "expert_initialization": (
            "Each expert clones one complementary 1/top-k partition of the source FFN; "
            "the output projection is scaled by top-k so the initial preferred expert pair "
            "reconstructs the original dense FFN."
        ),
        "attention_initialization": (
            "Query/output projections are copied from MHA. KV content uses a joint truncated-SVD "
            "initialization into the MLA latent bottleneck; shared rotary keys start from the mean "
            "of the source per-head key subspaces. Learned absolute positions are not copied."
        ),
    }


def flop_metrics(model: SparseMoEMTPTransformer, metrics: dict[str, Any]) -> dict[str, float]:
    dense = estimate_dense_block_flops(model.config, model.config.context_length)
    estimator = (
        estimate_mla_skiplayer_moe_mtp_flops
        if model.config.attention_type == "mla"
        else estimate_sparse_moe_mtp_flops
    )
    inference = estimator(
        model.config, model.config.context_length, metrics["compute_fraction"]
    )
    d, t = model.config.d_model, model.config.context_length - 1
    mtp_block = 8 * t * d * d + 4 * t * t * d + 4 * t * d * model.config.d_ff
    mtp_projection = 4 * t * d * d
    return {
        "dense_block_flops_per_sequence": dense,
        "estimated_executed_block_flops_per_sequence": inference,
        "estimated_flops_vs_full_dense": inference / dense,
        "estimated_training_flops_per_sequence": inference + mtp_block + mtp_projection,
        "estimated_training_flops_vs_full_dense": (inference + mtp_block + mtp_projection) / dense,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-checkpoint",
        default=(
            "experiments/mor_comparison_seed42/"
            "skiplayer_grpo_clip0p5_lam1_seed42/checkpoints/best_quality_compute.pt"
        ),
    )
    p.add_argument(
        "--experiment-dir",
        default="experiments/skiplayer_moe_mtp_mla_rope_seed42/supervised",
    )
    p.add_argument("--data-path", default="data/input.txt")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-experts", type=int, default=10)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--attention", choices=["mla", "mha"], default="mla")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--mtp-loss-weight", type=float, default=0.3)
    p.add_argument("--moe-aux-weight", type=float, default=0.0001)
    p.add_argument("--moe-bias-update-speed", type=float, default=0.001)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = parser().parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    source_model, source_checkpoint = load_checkpoint(args.source_checkpoint, device)
    if source_model.config.model_type != "sparse":
        raise ValueError("source checkpoint must be the SkipLayer or SkipLayer+GRPO model")
    config: ModelConfig = replace(
        source_model.config,
        model_type=(
            "sparse_moe_mtp_mla" if args.attention == "mla" else "sparse_moe_mtp"
        ),
        moe_num_experts=args.num_experts,
        moe_top_k=args.top_k,
        moe_expert_d_ff=source_model.config.d_ff // args.top_k,
        moe_bias_update_speed=args.moe_bias_update_speed,
        moe_aux_loss_coefficient=args.moe_aux_weight,
        mtp_loss_coefficient=args.mtp_loss_weight,
    )
    model = build_model(config).to(device)
    if not isinstance(model, SparseMoEMTPTransformer):
        raise TypeError("failed to construct SparseMoEMTPTransformer")
    transfer = initialize_from_skiplayer(model, source_model)
    dataset = TinyShakespeareData(args.data_path, config.context_length)
    if dataset.stoi != source_checkpoint["stoi"]:
        raise ValueError("source checkpoint vocabulary differs from dataset")

    # Preserve the already learned compute policy during expert/MTP adaptation.
    for parameter in model.router.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1), eta_min=args.min_lr
    )
    source_training = source_checkpoint.get("training_config") or {}
    target_density = float(source_training.get("target_density", 0.5))
    experiment_dir = Path(args.experiment_dir)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    before = evaluate_model(model, dataset, device, args.batch_size, args.eval_iters, target_density)
    before.update(flop_metrics(model, before))
    full_config = {
        "stage": "moe_mtp_supervised_adaptation",
        "model": model.config.to_dict(),
        "device": str(device),
        "experiment_dir": str(experiment_dir),
        "source_checkpoint": str(args.source_checkpoint),
        "target_density": target_density,
        "transfer": transfer,
        "training": vars(args),
        "before_adaptation": before,
        "design_note": (
            "Ten top-2 half-width routed FFN experts preserve approximately one dense FFN's "
            "active multiply-adds. MLA compresses KV content and uses decoupled partial RoPE; "
            "causal attention uses fused SDPA. MTP depth D=1 predicts t+2 and is discarded at inference."
        ),
    }
    write_json(experiment_dir / "config.json", full_config)
    scalar_fields = [
        "step", "split", "train_loss", "train_perplexity", "train_accuracy",
        "mtp_loss", "mtp_accuracy", "total_loss", "moe_aux_loss",
        "moe_router_entropy", "expert_utilization_min", "expert_utilization_max",
        "expert_utilization_cv", "mean_soft_gate", "mean_hard_gate",
        "layers_per_token", "compute_fraction", "skip_fraction", "routing_entropy",
        "val_loss", "val_perplexity", "val_accuracy", "learning_rate",
        "gradient_norm", "tokens_processed", "seconds_per_step", "tokens_per_second",
        "validation_time_sec", "estimated_executed_block_flops_per_sequence",
        "estimated_flops_vs_full_dense", "estimated_training_flops_per_sequence",
        "estimated_training_flops_vs_full_dense", "mla_kv_cache_ratio", "rope_fraction",
    ]
    expert_fields = [
        f"layer_{layer}_expert_{expert}_{kind}"
        for layer in range(config.n_layers)
        for expert in range(config.moe_num_experts)
        for kind in ("utilization", "affinity")
    ]
    logger = StructuredLogger(experiment_dir, scalar_fields + expert_fields)
    best_loss = float("inf")
    best = {"val_loss": best_loss, "val_perplexity": float("inf"), "quality_compute_score": float("inf")}
    tokens_processed = 0
    total_step_seconds = 0.0
    started = time.perf_counter()
    latest_eval = before
    latest_train: dict[str, Any] = {}
    print(
        f"device={device} SkipLayer+MoE+MTP experts={config.moe_num_experts} "
        f"top_k={config.moe_top_k} parameters={model.parameter_count():,}"
    )
    try:
        for step in range(args.max_steps):
            model.train()
            synchronize_device(device)
            step_started = time.perf_counter()
            x, y = dataset.get_batch("train", args.batch_size, device)
            output = model(x, y, routing_mode="gumbel")
            total = (
                output.lm_loss
                + config.mtp_loss_coefficient * output.mtp_loss
                + config.moe_aux_loss_coefficient * output.moe_aux_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = gradient_norm(trainable)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            model.update_moe_selection_biases()
            scheduler.step()
            synchronize_device(device)
            seconds = time.perf_counter() - step_started
            total_step_seconds += seconds
            tokens_processed += x.numel()
            route = routing_metrics(output, config.n_layers)
            latest_train = {
                "split": "train", "train_loss": output.lm_loss.item(),
                "train_perplexity": perplexity(output.lm_loss.item()),
                "train_accuracy": top1_accuracy(output.logits, y).item(),
                "mtp_loss": output.mtp_loss.item(), "mtp_accuracy": output.mtp_accuracy.item(),
                "total_loss": total.item(), "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": grad, "tokens_processed": tokens_processed,
                "seconds_per_step": seconds, "tokens_per_second": x.numel() / max(seconds, 1e-9),
                "mla_kv_cache_ratio": (
                    (config.mla_kv_lora_rank + config.mla_qk_rope_head_dim)
                    / (2 * config.n_heads * (config.d_model // config.n_heads))
                ),
                "rope_fraction": config.mla_qk_rope_head_dim / (
                    config.mla_qk_nope_head_dim + config.mla_qk_rope_head_dim
                ),
                **route, **flop_metrics(model, route),
            }
            for layer, values in enumerate(route["expert_utilization"]):
                for expert, value in enumerate(values):
                    latest_train[f"layer_{layer}_expert_{expert}_utilization"] = value
            for layer, values in enumerate(route["expert_affinity"]):
                for expert, value in enumerate(values):
                    latest_train[f"layer_{layer}_expert_{expert}_affinity"] = value
            if step % args.log_interval == 0 or step == args.max_steps - 1:
                logger.log(latest_train, step)
                print(
                    f"step {step:4d} | CE {output.lm_loss.item():.4f} | MTP {output.mtp_loss.item():.4f} "
                    f"| depth {route['layers_per_token']:.2f}/{config.n_layers} "
                    f"| expert CV {route['expert_utilization_cv']:.3f} | {seconds:.3f}s"
                )
            if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
                latest_eval = evaluate_model(
                    model, dataset, device, args.batch_size, args.eval_iters, target_density
                )
                latest_eval.update(flop_metrics(model, latest_eval))
                logger.log({"split": "validation", **latest_eval}, step)
                checkpoint_args = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step + 1,
                    stoi=dataset.stoi, itos=dataset.itos, training_config=full_config,
                    best_metrics=best,
                )
                if latest_eval["val_loss"] < best["val_loss"]:
                    best["val_loss"] = latest_eval["val_loss"]
                    save_checkpoint(experiment_dir / "checkpoints" / "best_val_loss.pt", **checkpoint_args)
                save_checkpoint(experiment_dir / "checkpoints" / "latest.pt", **checkpoint_args)
                print(
                    f"validation | CE {latest_eval['val_loss']:.4f} | MTP {latest_eval['mtp_loss']:.4f} "
                    f"| acc {latest_eval['val_accuracy']:.3f} | depth {latest_eval['layers_per_token']:.2f} "
                    f"| FLOPs {latest_eval['estimated_flops_vs_full_dense']:.3f}x"
                )
    finally:
        logger.close()

    elapsed = time.perf_counter() - started
    summary = {
        "model": config.model_type, "router_type": config.router_type,
        "training_method": "supervised_moe_mtp_adaptation", "seed": args.seed,
        "target_density": target_density, "lambda_grpo": source_training.get("grpo", {}).get("lambda_compute_grpo", 1.0),
        "parameter_count": model.parameter_count(), "training_time_sec": elapsed,
        "moe_num_experts": config.moe_num_experts, "moe_top_k": config.moe_top_k,
        "moe_expert_d_ff": config.moe_expert_d_ff,
        "mtp_prediction_depth": 1, "mtp_loss_coefficient": config.mtp_loss_coefficient,
        "attention_type": config.attention_type,
        "position_embedding_type": config.position_embedding_type,
        "mla_kv_lora_rank": config.mla_kv_lora_rank,
        "mla_qk_nope_head_dim": config.mla_qk_nope_head_dim,
        "mla_qk_rope_head_dim": config.mla_qk_rope_head_dim,
        "mla_v_head_dim": config.mla_v_head_dim,
        "seconds_per_step": total_step_seconds / max(args.max_steps, 1),
        "tokens_per_sec": tokens_processed / max(elapsed, 1e-9),
        "checkpoint": str(experiment_dir / "checkpoints" / "best_val_loss.pt"),
        "source_checkpoint": str(args.source_checkpoint), "before_adaptation": before,
        **latest_eval,
    }
    write_json(experiment_dir / "summary.json", summary)
    save_checkpoint(
        experiment_dir / "checkpoints" / "latest.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, step=args.max_steps, stoi=dataset.stoi, itos=dataset.itos,
        training_config=full_config, best_metrics=best, summary=summary,
    )
    print(f"completed MoE+MTP adaptation; artifacts saved in {experiment_dir}")


if __name__ == "__main__":
    main()
