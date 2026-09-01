# Sparse Transformer Routing Experiment

## 1. Objective

Sparse depth lets each token skip Transformer blocks it may not need. The linear SkipLayer baseline makes an independent skip/execute decision at each layer. The GRU router also remembers its earlier routing decisions across depth. GRPO is tested as a second-stage optimizer because final routing decisions are discrete and can be scored directly for quality and compute. H1 asks whether linear routing retains quality with less compute; H2 asks whether the GRU improves that tradeoff; H3 asks whether GRPO moves the Pareto frontier. H3 is not assumed true.

## 2. Experimental Setup

Tiny Shakespeare, character tokens, 90/10 train/validation split. The shared backbone has 8 layers, width 256, 4 attention heads, FFN width 1024, and context 128. Supervised training used up to 5000 steps with batch size 32. Devices: mps. Seeds: 42, 43, 44. Density targets: 0.50, 1.00. GRPO group size: 4; compute penalty: 0.1; KL coefficient: 0.01. Missing values are reported as `NaN`; no results are synthesized.

## 3. Models Compared

- **Dense Transformer:** every token executes every block.
- **Linear SkipLayer:** one two-logit linear gate per block, trained with straight-through Gumbel-Softmax.
- **GRU SkipLayer:** a per-token hidden state propagates across depth before producing each gate.
- **GRU SkipLayer + GRPO:** starts from the supervised GRU checkpoint and normally freezes the Transformer while updating only the router.

## 4. Supervised Routing Objective

$$L = L_{\mathrm{CE}} + \lambda\frac{1}{L}\sum_l(r_l-P)^2.$$

Here, $L_{\mathrm{CE}}$ is cross entropy, $r_l$ is hard executed-token density at layer $l$, $P$ is requested density, and $\lambda$ weights the density penalty. The penalty is warmed up rather than applied immediately.

## 5. GRPO Objective

The state $s_{t,l}$ contains the current token representation and GRU routing state. The action is skip or execute, and the GRU router is the policy:

$$R = -L_{\mathrm{CE}}-\lambda_c C-\beta\,KL.$$

The router is rewarded for maintaining prediction quality while using fewer layers. Advantages are normalized within each group, and trajectory log probability is the mean over token-layer decisions before applying the clipped GRPO objective.

## 6. Main Results

| Model | Seeds | Validation CE | Perplexity | Accuracy | Layers/token | Compute fraction | Training seconds | Generation tokens/s | ΔCE vs dense | ΔPPL vs dense |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense | 3 | 1.521 ± 0.004 | 4.578 ± 0.018 | 54.879 ± 0.072% | 8.000 ± 0.000 | 100.000 ± 0.000% | 1404.623 ± 27.939 | 113.665 ± 83.870 | +0.0000 | +0.0000 |
| GRU P=0.50, λ=0.1 | 3 | 1.523 ± 0.004 | 4.588 ± 0.018 | 54.711 ± 0.103% | 8.000 ± 0.000 | 100.000 ± 0.000% | 1594.017 ± 145.945 | 70.541 ± 25.898 | +0.0022 | +0.0102 |
| GRU + GRPO P=0.50, λ=0.1 | 3 | 1.523 ± 0.004 | 4.588 ± 0.018 | 54.711 ± 0.103% | 8.000 ± 0.000 | 100.000 ± 0.000% | 564.969 ± 41.107 | 97.888 ± 10.487 | +0.0022 | +0.0102 |
| LINEAR P=0.50, λ=0.1 | 3 | 1.536 ± 0.005 | 4.644 ± 0.023 | 54.327 ± 0.167% | 7.754 ± 0.111 | 96.930 ± 1.388% | 1556.551 ± 41.598 | 99.534 ± 28.750 | +0.0143 | +0.0659 |

With one seed, “± 0” does not quantify uncertainty; it means across-seed deviation is unavailable.

## 7. Quality vs Compute

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

Points must be compared at matched compute. Lower CE with more layers is a tradeoff, not an unconditional improvement.

Paired supervised comparisons: seed 42, P=0.50: GRU−Linear ΔCE=-0.0193, Δcompute=+0.0288 (not compute-matched); seed 43, P=0.50: GRU−Linear ΔCE=-0.0080, Δcompute=+0.0179 (matched closely); seed 44, P=0.50: GRU−Linear ΔCE=-0.0089, Δcompute=+0.0454 (not compute-matched).

## 8. Effect of GRPO

**GRPO made no meaningful, consistent difference in the measured Pareto frontier. Paired changes: ΔCE=+0.0021, Δcompute=+0.0000; ΔCE=+0.0050, Δcompute=+0.0000; ΔCE=-0.0011, Δcompute=+0.0000.** This compares each GRPO run to its initializing supervised checkpoint. Exact equal-compute or equal-quality claims require overlapping sweep points.

![Supervised to GRPO movement](sl_vs_grpo.png)

## 9. Routing Behavior

Mean absolute target-density error was 0.490. Mean Spearman correlation between token NLL and depth was -0.163. Across available runs, layer 0 was used most (1.000) and layer 4 least (0.958). Token maps are stored with each experiment. Routing heatmaps contain hard decisions and soft execute probabilities for the same validation text.

## 10. Training Dynamics

Each experiment's `plots/` directory contains CE, perplexity, accuracy, density, depth, and per-layer histories. GRPO runs additionally contain reward, component, KL, and entropy histories. Persistent near-zero entropy suggests policy collapse; no collapse is inferred when logs are absent.

Observed endpoint changes: dense_seed42: validation CE 2.3705→1.5250; dense_seed43: validation CE 2.3780→1.5214; dense_seed44: validation CE 2.3910→1.5172; gru_P050_lam0.1_seed42: validation CE 2.4192→1.5204; gru_P050_lam0.1_seed43: validation CE 2.4182→1.5219; gru_P050_lam0.1_seed44: validation CE 2.4173→1.5279; gru_grpo_P050_lam0.1_rl0.1_seed42: validation CE 1.5184→1.5184; gru_grpo_P050_lam0.1_rl0.1_seed42: reward -1.4151→-1.3260, entropy 0.024→0.020; gru_grpo_P050_lam0.1_rl0.1_seed43: validation CE 1.5170→1.5170; gru_grpo_P050_lam0.1_rl0.1_seed43: reward -1.3968→-1.2636, entropy 0.027→0.020; gru_grpo_P050_lam0.1_rl0.1_seed44: validation CE 1.5290→1.5290; gru_grpo_P050_lam0.1_rl0.1_seed44: reward -1.4595→-1.3649, entropy 0.025→0.018; linear_P050_lam0.1_seed42: validation CE 2.4332→1.5397; linear_P050_lam0.1_seed43: validation CE 2.4276→1.5300; linear_P050_lam0.1_seed44: validation CE 2.4176→1.5368.

## 11. Computational Efficiency

`compute_fraction` is the theoretical fraction of token-layer block executions. Stage A still evaluates every candidate and masks its output, so it demonstrates logical sparsity but **does not provide sparse-kernel wall-clock acceleration**. Generation latency is measured separately.

## 12. Limitations

Tiny Shakespeare is extremely small and character-level modeling is simplistic. Routing overhead matters, ordinary kernels may not exploit token sparsity, results may not transfer to LLMs, GRPO adds substantial training compute, and small seed counts do not justify statistical-significance claims.

## 13. Conclusion

The selected linear point used 0.982 compute with ΔCE=+0.0088 versus dense. Paired supervised comparisons: seed 42, P=0.50: GRU−Linear ΔCE=-0.0193, Δcompute=+0.0288 (not compute-matched); seed 43, P=0.50: GRU−Linear ΔCE=-0.0080, Δcompute=+0.0179 (matched closely); seed 44, P=0.50: GRU−Linear ΔCE=-0.0089, Δcompute=+0.0454 (not compute-matched). GRPO made no meaningful, consistent difference in the measured Pareto frontier. Paired changes: ΔCE=+0.0021, Δcompute=+0.0000; ΔCE=+0.0050, Δcompute=+0.0000; ΔCE=-0.0011, Δcompute=+0.0000. These measured tradeoffs—not raw loss alone—determine whether scaling is justified.

## 14. Recommended Next Experiment

Tune `lambda_density` before scaling: mean absolute target error is 0.490, so router comparisons are not yet compute-matched.
