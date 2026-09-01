# Sparse Transformer Routing Experiment

## 1. Objective

Sparse depth lets each token skip Transformer blocks it may not need. The linear SkipLayer baseline makes an independent skip/execute decision at each layer. The GRU router also remembers its earlier routing decisions across depth. GRPO is tested as a second-stage optimizer because final routing decisions are discrete and can be scored directly for quality and compute. H1 asks whether linear routing retains quality with less compute; H2 asks whether the GRU improves that tradeoff; H3 asks whether GRPO moves the Pareto frontier. H3 is not assumed true.

## 2. Experimental Setup

Tiny Shakespeare, character tokens, 90/10 train/validation split. The shared backbone has 4 layers, width 128, 2 attention heads, FFN width 1024, and context 128. Supervised training used up to 1000 steps with batch size 32. Devices: mps. Seeds: 42. Density targets: 0.50, 1.00. GRPO group size: not run; compute penalty: not run; KL coefficient: not run. Missing values are reported as `NaN`; no results are synthesized.

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
| Dense | 1 | 2.002 ± 0.000 | 7.403 ± 0.000 | 41.233 ± 0.000% | 4.000 ± 0.000 | 100.000 ± 0.000% | 66.264 ± 0.000 | 51.486 ± 0.000 | +0.0000 | +0.0000 |
| LINEAR P=0.50, λ=0.1 | 1 | 2.864 ± 0.000 | 17.539 ± 0.000 | 19.648 ± 0.000% | 3.994 ± 0.000 | 49.919 ± 0.000% | 234.944 ± 0.000 | 12.013 ± 0.000 | +0.8626 | +10.1362 |

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

Mean absolute target-density error was 0.001. Mean Spearman correlation between token NLL and depth was +0.071. Across available runs, layer 5 was used most (0.667) and layer 0 least (0.336). Token maps are stored with each experiment. Routing heatmaps contain hard decisions and soft execute probabilities for the same validation text.

## 10. Training Dynamics

Each experiment's `plots/` directory contains CE, perplexity, accuracy, density, depth, and per-layer histories. GRPO runs additionally contain reward, component, KL, and entropy histories. Persistent near-zero entropy suggests policy collapse; no collapse is inferred when logs are absent.

Observed endpoint changes: paper_dense_L04_seed42: validation CE 2.5771→2.0053; paper_skip_L08_P050_Eff04_seed42: validation CE 2.8685→3.0642.

## 11. Computational Efficiency

`compute_fraction` is the theoretical fraction of token-layer block executions. Stage A still evaluates every candidate and masks its output, so it demonstrates logical sparsity but **does not provide sparse-kernel wall-clock acceleration**. Generation latency is measured separately.

## 12. Limitations

Tiny Shakespeare is extremely small and character-level modeling is simplistic. Routing overhead matters, ordinary kernels may not exploit token sparsity, results may not transfer to LLMs, GRPO adds substantial training compute, and small seed counts do not justify statistical-significance claims.

## 13. Conclusion

The selected linear point used 0.499 compute with ΔCE=+0.8626 versus dense. No paired Linear/GRU result at the same seed, target, and density coefficient is available, so H2 cannot yet be judged fairly. No completed GRPO run with paired before/after measurements was found, so H3 cannot yet be evaluated. These measured tradeoffs—not raw loss alone—determine whether scaling is justified.

## 14. Recommended Next Experiment

First complete the core four-way comparison with at least three seeds; the present seed count is insufficient for a stable ranking.
