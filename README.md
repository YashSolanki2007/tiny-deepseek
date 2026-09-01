# SkipLayer Reproduction and Dynamic-Routing Extensions

The primary baseline is now a method-faithful, small-scale reproduction of
**Learning to Skip for Language Modeling** (arXiv:2311.15436). GRU routing and
GRPO remain available as explicitly separate extensions and should not be used
to judge whether the paper's supervised SkipLayer method was reproduced.

## Paper-first workflow

Run the end-to-end correctness smoke test:

```bash
python run_paper_experiments.py \
  --preset smoke \
  --experiments-dir experiments/paper_smoke \
  --results-dir results/paper_smoke
```

Run the matched-effective-depth Tiny Shakespeare comparison:

```bash
python run_paper_experiments.py \
  --preset core \
  --seeds 42 43 44
```

The exact protocol uses the paper's Adafactor learning rate of `0.1`. Because
that rate is unstable at this much smaller scale, a scale-adapted run must be
explicitly labeled:

```bash
python run_paper_experiments.py \
  --preset core \
  --paper-learning-rate 0.01
```

See [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) for the equation-by-equation
mapping, scale substitutions, sparse greedy execution semantics, and Table-1
analogue.

A correctness-first PyTorch research codebase for comparing four character-level language models on Tiny Shakespeare:

1. dense decoder-only Transformer;
2. SkipLayer-style linear router;
3. depth-aware GRU router;
4. supervised GRU router followed by router-only GRPO.

The scientific question is whether learned routing improves validation quality at **matched compute**, not simply whether one run has lower loss. Results, timing, routing maps, seed aggregates, and the final report are generated from real run artifacts; missing metrics remain `NaN`.

## Important implementation scope

Sparse models apply each hard gate as:

```python
candidate = block(x)
x = (1 - gate) * x + gate * candidate
```

This makes skip exactly identity and execute exactly the complete residual block. Training is a **Stage-A logical sparsity implementation**: all block candidates are still evaluated to preserve the straight-through gradient. Paper-mode greedy evaluation gathers active queries and FFN inputs while retaining key/value projections for all tokens. Density and estimated block FLOPs remain theoretical; generation latency is measured independently. The project does not claim optimized sparse-kernel wall-clock speedup.

The linear and GRU policies both emit `[skip, execute]` logits. Supervised training uses hard-forward, soft-backward Gumbel-Softmax. The GRU maintains one `[B, T, router_dim]` state per token and propagates it across model depth, never across text positions.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`data/input.txt` is downloaded from the Tiny Shakespeare source if absent. Tokenization is character-level with a sorted `stoi`/`itos` vocabulary and a contiguous 90/10 train/validation split. Device selection is CUDA, then Apple MPS, then CPU; override it with `--device`.

## Core models

Dense baseline:

```bash
python train.py \
  --model dense \
  --experiment-dir experiments/dense_seed42
```

Linear SkipLayer at 50% target density:

```bash
python train.py \
  --model sparse \
  --router linear \
  --target-density 0.5 \
  --lambda-density 0.1 \
  --experiment-dir experiments/linear_P050_lam0.1_seed42
```

GRU SkipLayer at the same target:

```bash
python train.py \
  --model sparse \
  --router gru \
  --router-dim 32 \
  --target-density 0.5 \
  --lambda-density 0.1 \
  --experiment-dir experiments/gru_P050_lam0.1_seed42
```

Defaults are context 128, width 256, 4 heads, 8 layers, FFN width 1024, dropout 0.1, batch size 32, and 5,000 optimization steps. All three models share the backbone configuration. Input/output embeddings are tied unless `--no-tie-weights` is supplied.

The supervised objective is:

```text
CE + lambda_density × mean_layer((hard_density_layer - target_density)²)
```

The density coefficient is zero during the first 10% of training, ramps from 10–30%, and stays at its configured value afterward. Change the schedule with `--density-warmup-start` and `--density-warmup-end`.

## GRPO router fine-tuning

## Mixture-of-Recursions comparison

The paper-aligned MoR implementation uses Middle-Cycle parameter sharing:
one unique entry block, two shared middle blocks applied for three recursive
steps, and one unique exit block. This gives eight effective layers from four
stored Transformer blocks. Hierarchical expert-choice capacities are
`1, 2/3, 1/3`; validation logs learned greedy routing separately from oracle
top-k routing.

Run the complete requested comparison—full dense, SkipLayer, SkipLayer + GRPO,
MoR, and MoR + GRPO—with:

```bash
python run_mor_comparison.py
```

The default output locations are:

```text
experiments/mor_comparison_seed42/
results/mor_comparison_seed42/
```

For a live, clearly grouped TensorBoard view:

```bash
tensorboard \
  --logdir experiments/mor_comparison_seed42 \
  --port 6006
```

Open `http://localhost:6006`. Supervised runs log `train`, `validation`, and,
for MoR, `validation_oracle_topk`. GRPO runs additionally log reward, policy
ratio/clipping, KL, entropy, achieved depth and FLOPs for every rollout budget,
plus per-recursion utilization and router diagnostics.

For the paper-faithful linear SkipLayer checkpoint, use the budget-guided,
per-decision variant. It samples explicit 3/4/5/8-layer rollout groups from a
recorded router/controller mixture, uses 8/8 as a quality anchor, freezes the
Transformer, and excludes that deterministic anchor from the policy loss:

```bash
python run_paper_grpo_experiments.py \
  --checkpoint experiments/paper_core_scaled_lr001_1000/paper_skip_L08_P050_Eff04_seed42/checkpoints/best_val_loss.pt \
  --lambdas 0.1 1.0 2.0 \
  --depth-budgets 3 4 5 8 \
  --exploration-epsilon 0.8
```

The budget-guided objective uses the exact behavior log probability for every
sampled token-layer action and applies PPO clipping per decision. CSV and
TensorBoard logs include achieved depth, CE, reward, and estimated FLOPs for
each requested budget. `train_grpo.py` below remains the older GRU-router
extension and is not used for the paper-faithful experiment.

Start GRPO from the best supervised GRU validation-loss checkpoint:

```bash
python train_grpo.py \
  --checkpoint experiments/gru_P050_lam0.1_seed42/checkpoints/best_val_loss.pt \
  --experiment-dir experiments/gru_grpo_P050_lam0.1_rl0.1_seed42 \
  --group-size 4 \
  --lambda-compute-grpo 0.1 \
  --beta-kl 0.01 \
  --grpo-router-only
```

For each source sequence, GRPO samples a group of routing trajectories. Its default reward is:

```text
-sequence_CE - lambda_compute_grpo × compute_fraction
```

The KL to the original supervised router is added to the clipped policy objective by default. `--kl-in-reward` additionally puts it in the sampled reward. Advantages are normalized within each group. A trajectory log probability is the **mean** log probability over all token-layer decisions, avoiding scale growth with context length or depth.

The default freezes embeddings, attention, FFNs, layer norms, and the LM head. `--grpo-unfreeze-transformer` is the optional ablation; it assigns Transformer parameters one tenth the router learning rate.

## Automated experiments

The core four-way comparison is one command:

```bash
python run_experiments.py --preset core
```

For three seeds:

```bash
python run_experiments.py \
  --preset core \
  --seeds 42 43 44
```

The full supervised density/coefficient sweep plus GRPO compute sweep is:

```bash
python run_experiments.py \
  --preset full \
  --seeds 42 43 44
```

`full` uses density targets 0.75, 0.50, 0.25; supervised coefficients 0.03, 0.1, 0.3; and GRPO compute coefficients 0.01, 0.05, 0.1, 0.2. Override these with `--densities`, `--lambda-densities`, or `--grpo-lambdas`. Completed `summary.json` runs are skipped unless `--force` is supplied. Unrecognized runner options are forwarded to supervised `train.py`, which is useful for smaller integration experiments.

The default core preset is still a substantial training job. For a quick pipeline check:

```bash
python run_experiments.py \
  --preset core \
  --max-steps 2 \
  --grpo-max-steps 2 \
  --context-length 8 \
  --d-model 16 \
  --n-heads 4 \
  --n-layers 3 \
  --d-ff 32 \
  --batch-size 2 \
  --eval-interval 1 \
  --eval-iters 1 \
  --log-interval 1 \
  --warmup-steps 0 \
  --device cpu
```

## Evaluation, generation, and routing maps

Evaluate one checkpoint:

```bash
python evaluate.py \
  --checkpoint experiments/gru_P050_lam0.1_seed42/checkpoints/best_val_loss.pt
```

Evaluate every completed experiment, including deterministic greedy routing, generation latency, routing maps, and token-difficulty analysis:

```bash
python evaluate.py --all
```

Generate text:

```bash
python generate.py \
  --checkpoint experiments/gru_P050_lam0.1_seed42/checkpoints/best_val_loss.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 500 \
  --temperature 0.8 \
  --top-k 40
```

Create a routing heatmap manually:

```bash
python visualize_routing.py \
  --checkpoint experiments/gru_P050_lam0.1_seed42/checkpoints/best_val_loss.pt \
  --text $'ROMEO:\nWhat light through yonder window breaks?' \
  --mode hard \
  --output experiments/gru_P050_lam0.1_seed42/routing_visualizations/custom_hard.png
```

## Results and reports

Regenerate all aggregate outputs from completed experiment summaries:

```bash
python generate_report.py
```

This creates:

```text
results/summary.csv
results/aggregate_results.csv
results/REPORT.md
results/REPORT.html
results/*.png
results/*.pdf
```

`REPORT.html` embeds available PNGs as base64 data URIs and works offline. It explicitly distinguishes theoretical density from latency and only evaluates GRPO when paired before/after metrics exist.

Each experiment directory contains:

```text
config.json
training_metrics.csv
tensorboard/
checkpoints/latest.pt
checkpoints/best_val_loss.pt
checkpoints/best_val_perplexity.pt
checkpoints/best_quality_compute.pt
plots/
samples/
routing_visualizations/
summary.json
```

Inspect live metrics with:

```bash
tensorboard --logdir experiments
```

## Resume

Supervised:

```bash
python train.py \
  --resume experiments/gru_P050_lam0.1_seed42/checkpoints/latest.pt \
  --experiment-dir experiments/gru_P050_lam0.1_seed42
```

GRPO:

```bash
python train_grpo.py \
  --resume experiments/gru_grpo_P050_lam0.1_rl0.1_seed42/checkpoints/latest.pt \
  --experiment-dir experiments/gru_grpo_P050_lam0.1_rl0.1_seed42
```

Checkpoints include model, optimizer, scheduler, step, seed/configuration, RNG state, best metrics, vocabulary, and—during GRPO—the frozen supervised reference router. Resume occurs at evaluation/checkpoint boundaries. First-generation one-logit checkpoints under the older `runs/` layout remain untouched but are intentionally rejected by this two-logit architecture with a clear compatibility error.

## Tests

```bash
pytest -q
```

Tests cover exact skip/execute semantics, dense and sparse output shapes, hard binary gates, initial execute probability, linear and GRU router gradients, GRU depth-state propagation, target-density loss, schedule warmup, routing-trajectory diversity, GRPO reward/advantages/clipping, and numerically exact greedy checkpoint restoration.

## File map

- `model.py`, `router.py`: backbone, exact gating, linear and GRU policies.
- `losses.py`: supervised density and GRPO objectives.
- `train.py`, `train_grpo.py`, `train_paper_grpo.py`: supervised, legacy GRU-GRPO, and budget-guided paper-router stages.
- `evaluation.py`, `evaluate.py`, `analysis.py`: metrics and difficulty analysis.
- `plots.py`, `report.py`, `generate_report.py`: plots and offline reports.
- `run_experiments.py`, `run_paper_grpo_experiments.py`, `aggregate_results.py`: orchestration and tables.
- `utils.py`, `logging_utils.py`: device, reproducibility, timing, checkpoints, CSV, TensorBoard.

## Experimental integrity

- Validation data is never used for optimization.
- Backbone size stays fixed across router comparisons unless the experiment name/config says otherwise.
- Dense and sparse timing is measured, while theoretical FLOP savings are labeled separately.
- A lower validation loss at higher compute is not called an improvement.
- Two or three seeds are summarized but not presented as proof of statistical significance.
- If GRPO makes routing worse, the generated report says so.
