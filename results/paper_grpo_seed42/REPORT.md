# SkipLayer Paper Reproduction on Tiny Shakespeare

## 1. Objective

This experiment first isolates the method in *Learning to Skip for Language Modeling*, then evaluates a router-only budget-guided GRPO extension without changing the Transformer weights. It asks whether a deeper, sparsely activated Transformer can match a shallower dense Transformer and whether RL fine-tuning moves that quality/compute point.

## 2. Scope and Scale Substitutions

Tiny Shakespeare with character tokenization and a contiguous 90/10 split. Widths: [128]; attention heads: [2]; contexts: [128]; batch sizes: [32]; training steps: [1000]; Adafactor learning rates: [0.01]; seeds: [42]. The attention head dimension is 64 and FFN width is 8× model width in the core protocol. The original private 1.6T-token corpus, 32K SentencePiece tokenizer, TPU grouped kernels, billion-parameter models, and 24-task one-shot evaluation suite are unavailable here. Results are therefore a method reproduction at small scale, not a reproduction of the paper's headline numbers.

## 3. Paper-Faithful SkipLayer

- Every physical Transformer layer has an independent bias-free linear `d_model → 2` router.
- The router consumes the pre-normalized layer input and emits skip/execute logits.
- Straight-through Gumbel-Softmax gives a hard binary forward path and soft backward surrogate.
- A skipped token is exact identity. An active token executes attention and FFN.
- Keys and values are retained for all causal-context tokens; only active queries and FFN inputs are gathered during greedy sparse evaluation.
- Greedy decoding selects the larger router logit.

## 4. Objective and Optimization

The paper's layerwise capacity objective is implemented as a sum, not an average:

$$L = L_{\mathrm{CE}} + 0.1\sum_l(r_l-P)^2.$$

There is no density-loss warmup. Training uses fixed-decay Adafactor with β1=0, β2=0.99, no separate gradient clipping, dropout 0, and inverse-square-root decay after 10,000 updates. A learning rate other than 0.1 is explicitly a scale-adapted ablation.

## 5. Main Results

| Model | Seeds | Total layers | Validation CE | Perplexity | Accuracy | Layers/token | Skipped | Estimated block FLOPs | FLOPs vs dense | Generation tokens/s |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense | 1 | 4 | 2.141 ± 0.000 | 8.510 ± 0.000 | 37.061 ± 0.000% | 4.000 ± 0.000 | 0.000 ± 0.000% | 369.1M ± 0.0M | 1.000× | 98.688 ± 0.000 |
| LINEAR P=0.50, λ=0.1 | 1 | 8 | 2.139 ± 0.000 | 8.490 ± 0.000 | 37.212 ± 0.000% | 4.463 ± 0.000 | 44.218 ± 0.000% | 442.0M ± 0.0M | 1.197× | 26.313 ± 0.000 |
| LINEAR + GRPO P=0.50, λ=0.1 | 1 | 8 | 2.143 ± 0.000 | 8.523 ± 0.000 | 37.178 ± 0.000% | 4.255 ± 0.000 | 46.809 ± 0.000% | 424.6M ± 0.0M | 1.150× | 26.204 ± 0.000 |
| LINEAR + GRPO P=0.50, λ=1 | 1 | 8 | 2.148 ± 0.000 | 8.571 ± 0.000 | 36.987 ± 0.000% | 4.213 ± 0.000 | 47.340 ± 0.000% | 421.0M ± 0.0M | 1.141× | 30.568 ± 0.000 |
| LINEAR + GRPO P=0.50, λ=2 | 1 | 8 | 2.160 ± 0.000 | 8.668 ± 0.000 | 36.504 ± 0.000% | 4.071 ± 0.000 | 49.111 ± 0.000% | 409.1M ± 0.0M | 1.108× | 28.167 ± 0.000 |

With one seed, “± 0” means across-seed uncertainty is unavailable, not zero uncertainty.

## 6. Quality vs Compute

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

`compute_fraction` is router density relative to the sparse model's physical depth. It is not FLOPs relative to the shallower dense baseline. Paper-style estimated FLOPs include always-on key/value projections.

## 7. Routing Behavior

Mean absolute target-density error was 0.031. Mean Spearman correlation between token NLL and depth was -0.048. Across available runs, layer 0 was used most (0.854) and layer 6 least (0.404). Token maps are stored with each experiment. Each sparse experiment additionally contains hard/soft routing heatmaps and a `token_skip_behavior.csv`/bubble plot analogous to the paper's token-skipping analysis.

## 8. Training Dynamics

Observed endpoint changes: paper_dense_L04_seed42: validation CE 2.5225→2.1536; paper_skip_L08_P050_Eff04_seed42: validation CE 2.5306→2.1479; paper_skip_L08_P050_Eff04_seed42_budget_grpo_lam0.1: validation CE 2.1427→2.1643; paper_skip_L08_P050_Eff04_seed42_budget_grpo_lam0.1: reward -2.6262→-2.6355, entropy 0.160→0.244; paper_skip_L08_P050_Eff04_seed42_budget_grpo_lam1: validation CE 2.1414→2.1609; paper_skip_L08_P050_Eff04_seed42_budget_grpo_lam1: reward -3.1896→-3.1985, entropy 0.160→0.259; paper_skip_L08_P050_Eff04_seed42_budget_grpo_lam2: validation CE 2.1416→2.1596; paper_skip_L08_P050_Eff04_seed42_budget_grpo_lam2: reward -3.8156→-3.8203, entropy 0.160→0.248.

## 9. Computational Efficiency

Training keeps the dense candidate computation to preserve the straight-through gradient. Greedy evaluation uses real gathered active queries and FFN inputs while retaining all key/value projections. The Python gather/scatter implementation on MPS is a semantic reference and can be slower than dense execution; it is not comparable to the paper's specialized TPU grouped kernels.

## 10. Budget-Guided GRPO Extension

Each group used explicit low/medium/full depth budgets. Non-anchor actions were sampled from a recorded mixture of the router policy and a remaining-budget controller, and PPO ratios were computed at every token-layer decision. The deterministic full-depth rollout served as a quality reference but was excluded from the policy-gradient loss. Measured changes: λ=0.1: depth 4.463→4.255, ΔCE=+0.0038; λ=1: depth 4.463→4.213, ΔCE=+0.0094; λ=2: depth 4.463→4.071, ΔCE=+0.0207. The lowest selected depth within ΔCE≤0.02 was 4.213/8. The joint target of depth≤4 and ΔCE≤0.02 was not reached. Every GRPO point remained dominated by the matched dense model in the paper-style CE/FLOPs comparison.

![Supervised to GRPO movement](sl_vs_grpo.png)

## 11. Conclusion

The selected SkipLayer point changed CE by -0.0023 relative to dense and executed 4.463 layers/token versus a 4.000 target. Its paper-style estimated block FLOPs were 1.197× dense because key/value projections remain active at every physical layer. This is a matched-effective-depth comparison, not a claim of equal FLOPs.

## 12. Recommended Next Experiment

Repeat the matched supervised pair and the selected GRPO point for at least three seeds, then add 25% and 12.5% density rows while holding effective depth fixed. The present one-seed result is not a stable ranking.
