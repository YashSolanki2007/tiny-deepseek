# Mixture-of-Recursions + SkipLayer/GRPO Comparison

## 1. Objective

This matched Tiny Shakespeare experiment compares the five requested systems: a full eight-layer dense Transformer, supervised SkipLayer, SkipLayer + GRPO, Mixture-of-Recursions (MoR), and MoR + GRPO.

## 2. MoR Architecture

The implementation follows the paper's selected expert-choice design at small scale: Middle-Cycle sharing with `1 + 2×3 + 1 = 8` effective layers, three recursions, capacities `1, 2/3, 1/3`, linear sigmoid routers, scale `α=0.1`, auxiliary BCE coefficient `0.001`, no capacity warmup, and recursion-wise attention/KV restriction. MoR therefore stores four unique Transformer blocks while exposing eight effective layers. Supervised training uses hierarchical top-k routing; greedy evaluation uses the learned `0.5` threshold and is reported separately from oracle top-k validation.

## 3. GRPO Extension

Both GRPO variants freeze their Transformer weights and update only routing parameters. SkipLayer uses explicit low/medium/full layer budgets. MoR uses one/two/three-recursion budgets, corresponding to nominal effective depths four/six/eight. The full path is a quality anchor excluded from the policy loss. Every sampled action retains its exact behavior probability, and PPO ratios are computed per valid token-routing decision. The final protocol uses PPO clip ε=0.5 because ε=0.2 clipped nearly every controller-guided MoR decision in the smoke test.

## 4. Main Results

| Model | Seeds | Validation CE | Perplexity | Layers/token | Skipped | Unique parameters | Params vs dense | Estimated block FLOPs | FLOPs vs dense | Generation tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense | 1 | 2.156 ± 0.000 | 8.637 ± 0.000 | 8.000 ± 0.000 | 0.000 ± 0.000% | 2.664M ± 0.000M | 1.000× | 738.2M ± 0.0M | 1.000× | 98.498 ± 0.000 |
| MoR + GRPO | 1 | 2.172 ± 0.000 | 8.776 ± 0.000 | 5.196 ± 0.000 | 35.052 ± 0.000% | 1.345M ± 0.000M | 0.505× | 475.5M ± 0.0M | 0.644× | 55.483 ± 0.000 |
| MoR | 1 | 2.170 ± 0.000 | 8.761 ± 0.000 | 5.489 ± 0.000 | 31.385 ± 0.000% | 1.345M ± 0.000M | 0.505× | 502.9M ± 0.0M | 0.681× | 66.725 ± 0.000 |
| SkipLayer | 1 | 2.139 ± 0.000 | 8.490 ± 0.000 | 4.463 ± 0.000 | 44.218 ± 0.000% | 2.666M ± 0.000M | 1.001× | 442.0M ± 0.0M | 0.599× | 31.705 ± 0.000 |
| SkipLayer + GRPO | 1 | 2.145 ± 0.000 | 8.544 ± 0.000 | 4.385 ± 0.000 | 45.193 ± 0.000% | 2.666M ± 0.000M | 1.001× | 435.4M ± 0.0M | 0.590× | 32.237 ± 0.000 |

With one seed, `± 0` means across-seed uncertainty is unavailable.

## 5. Quality vs FLOPs

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

All FLOPs are forward-pass block estimates. MoR recursion-wise attention scales quadratically with the surviving token fraction because both queries and KV entries are restricted. Python/MPS wall-clock throughput is reported separately and is not an optimized-kernel claim.

## 6. Supervised to GRPO Movement

![Supervised to GRPO movement](sl_vs_grpo.png)

GRPO produced a small or inconsistent improvement across the measured runs. Paired changes: ΔCE=+0.0017, Δcompute=-0.0367; ΔCE=+0.0063, Δcompute=-0.0097.

## 7. Routing Diagnostics

Each routed run contains layer heatmaps, token-skip analyses, difficulty/depth correlations, and per-layer utilization. MoR runs additionally log recursion utilization, soft router probabilities, auxiliary BCE, threshold accuracy, greedy FLOPs, and oracle top-k validation in separate TensorBoard namespaces.

## 8. Training Dynamics

full_dense_L08_seed42: validation CE 2.5338→2.1601; mor_grpo_clip0p5_lam1_seed42: validation CE 2.1706→2.1720; mor_grpo_clip0p5_lam1_seed42: reward -2.8245→-2.8264, entropy 0.579→0.572; mor_middle_cycle_R3_seed42: validation CE 2.5140→2.1747; skiplayer_L08_seed42: validation CE 2.5306→2.1479; skiplayer_grpo_clip0p5_lam1_seed42: validation CE 2.1407→2.1526; skiplayer_grpo_clip0p5_lam1_seed42: reward -3.1896→-3.1942, entropy 0.160→0.245.

## 9. Conclusion

The lowest validation CE was produced by SkipLayer (2.1389). The lowest estimated inference FLOPs were produced by SkipLayer + GRPO. These are separate criteria: MoR reduces unique parameters through sharing, while routing reduces executed recursion work. A point improves the global frontier only if no other model has both lower CE and lower paper-style FLOPs. This is a one-seed, character-level architecture study and not evidence that the ranking transfers directly to LLM scale.
