# Tiny-deepseek

Tiny-deepseek is a small, research-oriented decoder-only Transformer for studying
dynamic computation and DeepSeek-inspired components on a laptop. The repository
combines token-wise layer skipping, Group Relative Policy Optimization (GRPO),
Mixture-of-Experts (MoE), Multi-Token Prediction (MTP), Multi-Head Latent Attention
(MLA), rotary position embeddings (RoPE), and speculative decoding.

This is an educational small-scale implementation. It is not an official DeepSeek
model, and the experiments are not intended to reproduce large-model benchmark
results.

## What is implemented

- A standard dense decoder-only Transformer baseline.
- [SkipLayer](https://arxiv.org/abs/2311.15436)-style token-wise execution gates.
- Linear and GRU depth routers with hard-forward, soft-backward Gumbel routing.
- Budget-guided router-only GRPO using explicit depth trajectories.
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) with shared middle blocks.
- A literal MoR + inner SkipLayer + GRPO hybrid.
- Configurable sparse MoE feed-forward layers with top-2 routing.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)-style one-depth MTP.
- MLA with compressed latent KV state and partial RoPE.
- Exact one-token speculative sampling using the MTP block as the draft model.
- A synthetic-arithmetic-to-GSM8K curriculum.
- Token-policy GRPO with machine-verifiable numerical rewards.
- CSV, JSON, plots, checkpoints, samples, and TensorBoard logging.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/YashSolanki2007/MoR-with-Layer_Skipping.git
cd MoR-with-Layer_Skipping

python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Device selection uses CUDA first, Apple MPS second, and CPU otherwise. Pass
`--device cpu`, `--device mps`, or `--device cuda` to override it.

## Current Tiny-deepseek architectures

### Mathematical-reasoning model

The math experiment increases model capacity while retaining the same components:

| Component | Configuration |
|---|---|
| Dataset | Synthetic arithmetic followed by GSM8K |
| Tokenizer | Train-only byte-level BPE with BOS, EOS, PAD, and UNK |
| Vocabulary | 4,096 tokens |
| Context | 512 BPE tokens |
| Main decoder | 8 layers, width 256, 4 heads, FFN width 1024 |
| Dropout | 0.05 |
| Dynamic depth | Linear SkipLayer router; all eight layers forced on during SFT |
| Feed-forward variants | 4-expert top-2 MoE (expert width 512), or one dense FFN of width 1,024 |
| Attention | MLA, KV latent rank 64 |
| MLA heads | 32 content dimensions, 32 RoPE dimensions, 64 value dimensions |
| Positions | RoPE with theta 10,000 |
| MTP | One causal Transformer block predicting `t+2` |
| Stored parameters | 11,764,560 with MoE; 7,547,728 with dense FFNs |

The BPE tokenizer is trained only on the GSM8K training partition and synthetic
training examples. Validation and test text never fit the tokenizer. Byte-level BPE
retains lossless coverage while making every current GSM8K training and validation
example fit completely inside the 512-token context.

## Math data and training curriculum

The data loader downloads the official
[GSM8K](https://github.com/openai/grade-school-math) train and test JSONL files. The
official test split is never optimized. Ten percent of the official training examples
form a deterministic validation split.

The synthetic curriculum generates verified examples covering:

- Addition and subtraction.
- Multiplication and exact division.
- Two-step add/subtract word problems.
- Multiple names and object types.

Every response performs the reasoning before emitting the machine-readable result:

```text
Question: Ava has 17 apples and gets 8 more. How many apples are there?
Response:
Reasoning: Add the two amounts: 17 + 8 = 25.
<answer>25</answer>
```

The initial v2 run used 10,000 supervised steps. The capability-first continuation
then processed 80,000 additional complete examples with an effective batch size of
32. This is equivalent to 20,000 iterations at the original batch size of four:

- Steps 0–999: synthetic arithmetic only.
- Steps 1,000 onward: 90% GSM8K and 10% hard verified synthetic arithmetic.
- The synthetic warmup progresses through easy, medium, and hard generators.
- Physical batch size up to 32, with token-budgeted adaptive microbatches and
  gradient accumulation for long examples.
- One complete example per row with prompt and padding targets masked from loss.
- Dynamic padding to a multiple of 32 tokens.
- Full eight-layer execution throughout supervised capability training.
- AdamW with peak learning rate `3e-4`; resumed training uses `1e-4` and cosine decay.
- Response CE + `0.3 × MTP CE` + `0.0001 × MoE balance loss`.
- Every 500 steps, fixed held-out prompts track exact match, parse rate, repetition,
  pass@8, difficulty buckets, and arithmetic-operation accuracy.
- Checkpoint selection prioritizes greedy exact match, then parse rate, repetition,
  and validation CE.

Run the supervised pipeline and automatically start GRPO only if the readiness gate
passes:

```bash
python -m tiny_deepseek.workflows.run_math_grpo \
  --root-dir artifacts/experiments/math_v2_seed42 \
  --sft-steps 12500 \
  --synthetic-steps 1000 \
  --grpo-steps 500 \
  --resume-sft artifacts/experiments/math_v2_seed42/supervised/checkpoints/latest.pt \
  --device auto
```

Run the directly comparable no-MoE ablation with the same active FFN width and
training schedule:

```bash
python -m tiny_deepseek.training.math_sft \
  --dense-ffn \
  --experiment-dir artifacts/experiments/math_v2_seed42/dense_ffn_supervised \
  --max-steps 12500 \
  --synthetic-steps 1000 \
  --defer-readiness \
  --device auto
```

This keeps SkipLayer, MLA, RoPE, and the one-block MTP head but replaces every
top-2 MoE with one ordinary width-1,024 FFN. Placing the output under the same
`math_v2_seed42` directory makes the existing TensorBoard instance display the MoE
and dense-FFN runs together.

After SFT, the runner samples eight responses for each of 200 held-out validation
problems. GRPO starts only if greedy exact match is at least 5%, parse rate is at
least 95%, pass@8 is at least 20%, and at least 15% of groups contain both correct
and incorrect trajectories. Otherwise the checkpoint and diagnostics are retained
and the runner stops before reinforcement learning.

## Binary-correctness math GRPO

This is different from the earlier router-only GRPO. It optimizes generated token
probabilities so it can, in principle, improve answers rather than only selecting a
depth.

Before optimization, separate group-of-16 rollouts screen training prompts for both
correct and incorrect outcomes. GRPO then uses fresh on-policy rollouts from this
prompt pool, progressing from easy prompts to medium and hard prompts. This avoids
conditioning updates on the same trajectories used to select prompts.

For each selected GSM8K question, the policy samples 16 continuations of 128 BPE
tokens. Numerical strings are normalized so values such as `72`, `72.0`, `$72`, and
`1,072` can be compared correctly. Only exact correctness is rewarded:

```text
reward = 1 if normalized_prediction == normalized_gold_answer else 0

advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

Population standard deviation is used. When every completion receives the same
reward, the standard deviation is zero and every advantage is explicitly set to zero.
Training uses a per-token clipped policy objective with:

| Setting | Value |
|---|---:|
| GRPO steps | 500, conditional on readiness |
| Group size | 16 |
| Maximum completion | 128 BPE tokens |
| Rollout temperature | 1.0 |
| Learning rate | `3e-6` |
| PPO clip epsilon | 0.2 |
| Frozen-reference KL coefficient | 0.04 |
| Supervised replay coefficient | 0.5 |
| MTP replay coefficient | 0.3 |

The SkipLayer router and MoE selection routers remain frozen during quality GRPO.
Attention, active experts, token embeddings/shared LM head, normalizations, and MTP
remain trainable. Every RL step includes a GSM8K supervised replay batch to limit policy
drift, while the original supervised checkpoint supplies the frozen KL reference.

## TensorBoard

Start TensorBoard for both math stages:

```bash
tensorboard \
  --logdir /Users/yashsolanki/Desktop/layer-skipper/artifacts/experiments/math_v2_seed42 \
  --port 6008
```

Open [http://localhost:6008](http://localhost:6008).

The supervised dashboard includes:

- Training and validation CE, perplexity, and response-token accuracy.
- MTP CE and `t+2` accuracy.
- Average depth, density loss, skip fraction, and estimated FLOPs.
- MoE balance, entropy, and expert-utilization variation.
- Learning rate, gradient norm, step latency, and throughput.

The GRPO dashboard additionally includes:

- Group reward mean, standard deviation, best, and worst values.
- Exact-answer and parseable-answer rates.
- Binary exact-correctness reward and the fraction of correct group trajectories.
- Group advantage, policy ratio, clipping, and frozen-reference KL.
- Supervised replay and MTP replay losses.
- Final matched evaluation metrics under the `final/` group.

If port 6007 is occupied, select another port and open that address instead.

## Generate answers yourself

Run the built-in sample questions against the completed GRPO checkpoint:

```bash
python -m tiny_deepseek.cli.generate_math --max-new-tokens 128 --device auto
```

Supply one or more custom questions by repeating `--question`:

```bash
python -m tiny_deepseek.cli.generate_math \
  --question "If a store has 25 books and sells 9, how many remain?" \
  --question "There are 8 boxes with 6 pencils in each. How many pencils are there?" \
  --question "Sarah has 30 dollars, spends 12, and earns 7 more. How much does she have?" \
  --max-new-tokens 128 \
  --temperature 0 \
  --device auto
```

Request several stochastic answers with:

```bash
python -m tiny_deepseek.cli.generate_math \
  --question "What is 17 multiplied by 6?" \
  --samples-per-question 4 \
  --temperature 0.8 \
  --seed 123 \
  --max-new-tokens 128 \
  --device auto
```

The CLI defaults to the best supervised v2 checkpoint because its readiness gate did
not permit GRPO. It displays the raw completion, parsed numerical answer, and average
executed layers per generated token.

## Completed math results

### BPE/full-depth capability continuation

The initial 10,000-step checkpoint was continued with the GSM8K-heavy curriculum and
80,000 additional complete examples. Both evaluations ran at full eight-layer depth;
the final pass@8 evaluation used 200 GSM8K validation problems and 1,600 sampled
responses:

| Metric | Initial 10k | Capability continuation |
|---|---:|---:|
| Parameters | 11,764,560 | 11,764,560 |
| Validation response-token CE | 3.7237 | **2.7923** |
| Validation response-token accuracy | 36.59% | **47.44%** |
| MTP `t+2` accuracy | 38.00% | **48.30%** |
| Greedy exact match | 0/16 | **3/64 (4.69%)** |
| Greedy parse rate | 87.50% | **89.06%** |
| Sample-level exact match | 0.6875% | **1.3125% (21/1,600)** |
| Sample parse rate | 72.38% | **83.50%** |
| pass@8 | 5.50% | **9.00% (18/200)** |
| Mixed correct/incorrect groups | 5.50% | **9.00%** |
| Executed layers/token | 8.0/8.0 | 8.0/8.0 |
| Analytical FLOPs vs full dense | 0.9233× | 0.9233× |

The final readiness thresholds were 5% greedy exact match, 95% parse rate, 20% pass@8,
and 15% mixed groups. The final values were 4.69%, 89.06%, 9%, and 9%, respectively,
so the orchestrator correctly stopped before GRPO. The continuation substantially
improved token metrics and sampled correctness, but it still did not produce enough
correct trajectories for stable binary-reward GRPO. The selected checkpoint, all 200
rollout groups, summary JSON, and TensorBoard scalars remain under
`artifacts/experiments/math_v2_seed42/supervised/`.

The validation CE above comes from the final fixed 20-batch readiness evaluation.
Periodic training validation uses freshly sampled batches, while checkpoint selection
uses the fixed greedy-answer set; those measurements should not be treated as
identical evaluations.

To reproduce only the readiness evaluation:

```bash
python -m tiny_deepseek.cli.evaluate_math_readiness \
  --checkpoint artifacts/experiments/math_v2_seed42/supervised/checkpoints/best_exact_match.pt \
  --pass-examples 200 \
  --pass-k 8 \
  --max-new-tokens 128 \
  --device auto
```

### Historical short math runs

The original shaped-reward comparison evaluated the best supervised checkpoint and
the final GRPO checkpoint on the same six held-out GSM8K questions with identical
greedy decoding.

| Stage | Validation CE ↓ | Perplexity ↓ | Byte accuracy ↑ | MTP accuracy ↑ | Exact answers ↑ | Parse rate | Layers/token ↓ | FLOPs vs dense ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best supervised | **2.0175** | **7.519** | **46.00%** | **44.66%** | 0/6 | 100% | 7.976 | 0.911× |
| Quality GRPO | 2.2338 | 9.335 | 41.75% | 39.41% | 0/6 | 100% | **7.954** | **0.909×** |

A second matched trial replaced every auxiliary reward with strict binary correctness.
Across 20 groups and 80 sampled trajectories, no completion was correct. Consequently,
all group advantages and all policy losses were exactly zero. On ten deterministic
matched validation batches, the only small change came from supervised replay:

| Stage | Validation CE ↓ | Byte accuracy ↑ | MTP accuracy ↑ | Exact answers ↑ | Layers/token ↓ | FLOPs vs dense ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Best supervised | 2.1543 | 43.71% | 42.08% | 0/6 | 7.964 | 0.910× |
| Binary GRPO trial | **2.1494** | **43.98%** | **42.35%** | 0/6 | 7.964 | 0.910× |

This strict reward is less gameable, but it cannot train this checkpoint until at
least one sampled trajectory in a group is correct. Increasing the number of GRPO
steps alone does not solve that missing-signal problem efficiently.

The original shaped-reward run did not improve mathematical reasoning. The model learned to emit
valid `<answer>` tags but collapsed toward frequent synthetic values, especially `12`.
No exact-correct trajectory appeared during the 20 GRPO steps, so GRPO never received
an exact-correctness advantage to reinforce.

GRPO worsened validation CE by `0.2163`, reduced byte accuracy by `4.25` percentage
points, and reduced estimated dense-relative FLOPs by only `0.0023`. Greedy routing
remained almost fully dense. The result indicates that 300 from-scratch supervised
steps are below the capability threshold needed for correctness GRPO on GSM8K.

Representative held-out result:

```text
Question: Patricia has 30 roses. She gave 24 roses to her mother.
She bought 15 more roses. How many roses did she have now?

Gold answer: 21
Supervised:   <answer>1</answer>
Quality GRPO: <answer>1</answer>
```

Complete matched samples and raw values are generated at:

```text
artifacts/results/math_grpo_seed42/REPORT.md
artifacts/results/math_grpo_seed42/results.json
artifacts/experiments/math_binary_grpo_seed42/grpo/summary.json
artifacts/experiments/math_binary_grpo_seed42/grpo/training_metrics.csv
```

To regenerate the matched final report from saved checkpoints:

```bash
python -m tiny_deepseek.workflows.finalize_math_grpo --device auto
```

## Shakespeare quality and compute results

The earlier seed-42, 20-validation-batch comparison produced:

| Model | Validation CE ↓ | Accuracy ↑ | Layers/token ↓ | FLOPs vs dense ↓ | Stored parameters |
|---|---:|---:|---:|---:|---:|
| SkipLayer + GRPO | 2.1452 | 37.06% | 4.3846 | 0.5899× | 2.666M |
| + MoE + MTP, MHA | **1.9854** | **41.11%** | 4.7030 | 0.6281× | 11.470M |
| + MoE + MTP, MLA + RoPE | 2.0679 | 38.65% | 4.5936 | **0.5645×** | 11.283M |

The MLA+RoPE checkpoint improves both quality and estimated FLOPs over the original
SkipLayer+GRPO point, although the MHA MoE variant produced the best quality in this
short run.

Run the Shakespeare MoE/MTP/MLA experiment with:

```bash
python -m tiny_deepseek.workflows.run_skiplayer_moe_mtp \
  --supervised-steps 250 \
  --grpo-steps 120 \
  --eval-iters 10 \
  --device auto
```

## MTP speculative decoding

The trained one-block MTP head drafts the token after a target-model token. The target
verifies the draft and a bonus position in one causal pass. A draft is accepted with
`min(1, p_target / p_draft)`; rejection samples from the normalized positive residual
`max(0, p_target - p_draft)`. This preserves the temperature/top-k transformed target
distribution.

Benchmark the Shakespeare MLA+RoPE checkpoint and generate samples with:

```bash
python -m tiny_deepseek.cli.speculative_decode \
  --checkpoint artifacts/experiments/skiplayer_moe_mtp_mla_rope_seed42/grpo/checkpoints/best_quality_compute.pt \
  --benchmark-tokens 96 \
  --benchmark-repeats 3 \
  --sample-tokens 80 \
  --device auto
```

The warmed three-repeat Apple MPS benchmark measured:

| Decoder | Median time for 96 tokens | Tokens/s | Mean full-target calls |
|---|---:|---:|---:|
| Ordinary autoregressive | 6.519 s | 14.73 | 96.00 |
| One-block MTP speculative | 4.253 s | 22.57 | 63.33 |

That is a local `1.53×` speedup, `69.44%` draft acceptance, and `34.03%` fewer
full-target calls. The implementation currently recomputes full prefixes and does not
have a production KV cache, so these timings are specific to this repository and
hardware.

## Other included comparisons

The repository also retains the completed dense, SkipLayer, SkipLayer-GRPO, MoR,
MoR-GRPO, and MoR + inner SkipLayer experiment code. The earlier matched Tiny
Shakespeare comparison was:

| Model | Validation CE ↓ | Accuracy ↑ | FLOPs vs dense ↓ | Unique parameters |
|---|---:|---:|---:|---:|
| Full dense | 2.1560 | 36.15% | 1.000× | 2.664M |
| SkipLayer | **2.1389** | **37.21%** | 0.599× | 2.666M |
| SkipLayer + GRPO | 2.1452 | 37.06% | **0.590×** | 2.666M |
| MoR | 2.1704 | 35.50% | 0.681× | **1.345M** |
| MoR + recursion GRPO | 2.1720 | 35.56% | 0.644× | **1.345M** |

These are one-seed, small-model results and should not be treated as evidence that the
same rankings or savings transfer directly to LLM scale.

## Tests

Run the complete suite with:

```bash
python -m pytest -q
```

The current suite contains 60 tests covering routing semantics, checkpoints, GRPO
objectives, MoR, MoE accounting, MLA/RoPE, MTP, speculative residual sampling, math
tokenization and persistence, length-bucketed token-budgeted batching,
complete-example response masking, curriculum staging, cached MLA decoding,
numerical answer parsing, deterministic synthetic data, and reward ordering.

## Repository map

```text
tiny_deepseek/
├── core/        # Models, routing, losses, optimizers, checkpoints
├── data/        # Tiny Shakespeare and GSM8K data pipelines
├── training/    # Supervised, GRPO, MoR, MoE, and MTP trainers
├── evaluation/  # Metrics, analysis, aggregation, plots, reports
├── workflows/   # End-to-end reproducible experiment runners
└── cli/         # Generation, evaluation, benchmarks, visualization
tests/             # Automated correctness tests
data/              # Versioned input datasets and generated tokenizer cache
artifacts/         # Ignored checkpoints, TensorBoard logs, plots, and run outputs
README.md          # Architecture, commands, results, and limitations
docs/PAPER_REPRODUCTION.md
requirements.txt
pyproject.toml      # Package metadata and installed command entry points
```

All executable commands use Python modules, for example:

```bash
python -m tiny_deepseek.training.math_sft --help
python -m tiny_deepseek.workflows.run_math_grpo --help
python -m tiny_deepseek.cli.generate_math --help
```

## Experimental limitations

- The reported experiments use one seed and short training budgets.
- Analytical FLOP estimates are not equivalent to measured latency.
- Hard sparse training still evaluates candidate blocks for straight-through gradients.
- Python selected-token dispatch can be slower than vectorized dense execution when
  almost every layer is active.
- The math model is trained from scratch and does not yet solve held-out GSM8K.
- No claim is made that the small-scale MoE, MLA, MTP, GRPO, or routing behavior will
  transfer unchanged to production-scale language models.
- Checkpoints, TensorBoard event files, generated reports, plots, and temporary files
  are intentionally excluded from Git.
