# SkipLayer Paper Reproduction on Tiny Shakespeare

## 1. Objective

This experiment isolates the method in *Learning to Skip for Language Modeling* before any GRU-router or GRPO extension. It asks whether a deeper, sparsely activated Transformer can match a shallower dense Transformer at approximately the same expected number of active layers.

## 2. Scope and Scale Substitutions

Tiny Shakespeare with character tokenization and a contiguous 90/10 split. Widths: [32]; attention heads: [1]; contexts: [16]; batch sizes: [4]; training steps: [20]; Adafactor learning rates: [0.1]; seeds: [42]. The attention head dimension is 64 and FFN width is 8× model width in the core protocol. The original private 1.6T-token corpus, 32K SentencePiece tokenizer, TPU grouped kernels, billion-parameter models, and 24-task one-shot evaluation suite are unavailable here. Results are therefore a method reproduction at small scale, not a reproduction of the paper's headline numbers.

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
| Dense | 1 | 2 | 3.515 ± 0.000 | 33.600 ± 0.000 | 10.938 ± 0.000% | 2.000 ± 0.000 | 0.000 ± 0.000% | 1.4M ± 0.0M | 1.000× | 4.992 ± 0.000 |
| LINEAR P=0.50, λ=0.1 | 1 | 4 | 3.477 ± 0.000 | 32.368 ± 0.000 | 11.719 ± 0.000% | 3.859 ± 0.000 | 3.516 ± 0.000% | 2.7M ± 0.0M | 1.942× | 58.825 ± 0.000 |

With one seed, “± 0” means across-seed uncertainty is unavailable, not zero uncertainty.

## 6. Quality vs Compute

![Validation CE Pareto frontier](pareto_ce.png)

![Validation perplexity Pareto frontier](pareto_perplexity.png)

`compute_fraction` is router density relative to the sparse model's physical depth. It is not FLOPs relative to the shallower dense baseline. Paper-style estimated FLOPs include always-on key/value projections.

## 7. Routing Behavior

Mean absolute target-density error was 0.465. Mean Spearman correlation between token NLL and depth was -0.328. Across available runs, layer 1 was used most (0.977) and layer 0 least (0.930). Token maps are stored with each experiment. Each sparse experiment additionally contains hard/soft routing heatmaps and a `token_skip_behavior.csv`/bubble plot analogous to the paper's token-skipping analysis.

## 8. Training Dynamics

Observed endpoint changes: paper_dense_L02_seed42: validation CE 3.5145→3.5916; paper_skip_L04_P050_Eff02_seed42: validation CE 3.4809→3.4772.

## 9. Computational Efficiency

Training keeps the dense candidate computation to preserve the straight-through gradient. Greedy evaluation uses real gathered active queries and FFN inputs while retaining all key/value projections. The Python gather/scatter implementation on MPS is a semantic reference and can be slower than dense execution; it is not comparable to the paper's specialized TPU grouped kernels.

## 10. Conclusion

The selected SkipLayer point changed CE by -0.0374 relative to dense and executed 3.859 layers/token versus a 2.000 target. Its paper-style estimated block FLOPs were 1.942× dense because key/value projections remain active at every physical layer. This is a matched-effective-depth comparison, not a claim of equal FLOPs.

## 11. Next Paper-Only Experiment

Repeat the matched pair for at least three seeds, then add 25% and 12.5% density rows while holding effective depth fixed. GRPO should remain out of scope until those supervised paper baselines are stable. The present one-seed result is not a stable ranking.
