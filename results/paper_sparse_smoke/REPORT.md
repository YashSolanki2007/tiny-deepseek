# Sparse Transformer Routing Experiment

## 1. Objective

Sparse depth lets each token skip Transformer blocks it may not need. The linear SkipLayer baseline makes an independent skip/execute decision at each layer. The GRU router also remembers its earlier routing decisions across depth. GRPO is tested as a second-stage optimizer because final routing decisions are discrete and can be scored directly for quality and compute. H1 asks whether linear routing retains quality with less compute; H2 asks whether the GRU improves that tradeoff; H3 asks whether GRPO moves the Pareto frontier. H3 is not assumed true.

## 2. Experimental Setup

Tiny Shakespeare, character tokens, 90/10 train/validation split. The shared backbone has 2 layers, width 32, 1 attention heads, FFN width 256, and context 16. Supervised training used up to 2 steps with batch size 4. Devices: mps. Seeds: 42. Density targets: 0.50, 1.00. GRPO group size: not run; compute penalty: not run; KL coefficient: not run. Missing values are reported as `NaN`; no results are synthesized.

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
| Dense | 1 | 4.020 ± 0.000 | 55.682 ± 0.000 | 9.375 ± 0.000% | 2.000 ± 0.000 | 100.000 ± 0.000% | 0.570 ± 0.000 | 79.312 ± 0.000 | +0.0000 | +0.0000 |
| LINEAR P=0.50, λ=0.1 | 1 | 4.030 ± 0.000 | 56.286 ± 0.000 | 9.375 ± 0.000% | 2.305 ± 0.000 | 57.617 ± 0.000% | 2.516 ± 0.000 | 7.628 ± 0.000 | +0.0108 | +0.6043 |

With one seed, “± 0” does not quantify uncertainty; it means across-seed deviation is unavailable.

## 7. Quality vs Compute

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

Points must be compared at matched compute. Lower CE with more layers is a tradeoff, not an unconditional improvement.

No paired Linear/GRU result at the same seed, target, and density coefficient is available, so H2 cannot yet be judged fairly.

## 8. Effect of GRPO

**No completed GRPO run with paired before/after measurements was found, so H3 cannot yet be evaluated.** This compares each GRPO run to its initializing supervised checkpoint. Exact equal-compute or equal-quality claims require overlapping sweep points.

![Supervised to GRPO movement](sl_vs_grpo.png)

## 9. Routing Behavior

Mean absolute target-density error was 0.076. Mean Spearman correlation between token NLL and depth was -0.185. Across available runs, layer 3 was used most (0.844) and layer 0 least (0.219). Token maps are stored with each experiment. Routing heatmaps contain hard decisions and soft execute probabilities for the same validation text.

## 10. Training Dynamics

Each experiment's `plots/` directory contains CE, perplexity, accuracy, density, depth, and per-layer histories. GRPO runs additionally contain reward, component, KL, and entropy histories. Persistent near-zero entropy suggests policy collapse; no collapse is inferred when logs are absent.

Observed endpoint changes: paper_dense_L02_seed42: validation CE 4.0197→4.0197; paper_skip_L04_P050_Eff02_seed42: validation CE 4.0304→4.0304.

## 11. Computational Efficiency

`compute_fraction` is the theoretical fraction of token-layer block executions. Stage A still evaluates every candidate and masks its output, so it demonstrates logical sparsity but **does not provide sparse-kernel wall-clock acceleration**. Generation latency is measured separately.

## 12. Limitations

Tiny Shakespeare is extremely small and character-level modeling is simplistic. Routing overhead matters, ordinary kernels may not exploit token sparsity, results may not transfer to LLMs, GRPO adds substantial training compute, and small seed counts do not justify statistical-significance claims.

## 13. Conclusion

The selected linear point used 0.576 compute with ΔCE=+0.0108 versus dense. No paired Linear/GRU result at the same seed, target, and density coefficient is available, so H2 cannot yet be judged fairly. No completed GRPO run with paired before/after measurements was found, so H3 cannot yet be evaluated. These measured tradeoffs—not raw loss alone—determine whether scaling is justified.

## 14. Recommended Next Experiment

First complete the core four-way comparison with at least three seeds; the present seed count is insufficient for a stable ranking.
