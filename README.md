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
- Ten-expert sparse MoE feed-forward layers with top-2 routing.
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
python -m pip install -r requirements.txt
```

Device selection uses CUDA first, Apple MPS second, and CPU otherwise. Pass
`--device cpu`, `--device mps`, or `--device cuda` to override it.

## Current Tiny-deepseek architectures

### Tiny Shakespeare model

The latest Shakespeare model uses the following configuration:

| Component | Configuration |
|---|---|
| Dataset | Tiny Shakespeare, character-level |
| Vocabulary | 65 characters |
| Context | 128 tokens |
| Main decoder | 8 layers, width 128, 2 heads, FFN width 1024 |
| Dynamic depth | Linear SkipLayer router with greedy sparse inference |
| MoE | 10 experts per layer, top-2 active, expert width 512 |
| Attention | MLA, KV latent rank 32 |
| MLA heads | 32 content dimensions, 32 RoPE dimensions, 64 value dimensions |
| Positions | RoPE with theta 10,000 |
| MTP | One causal Transformer block predicting `t+2` |
| Stored parameters | 11.283M |

The MTP module receives a normalized main-model hidden state and the embedding of
the observed next token. These representations are concatenated, projected, passed
through one causal Transformer block, and decoded with the shared language-model
head. The MTP loss coefficient is `0.3`.

### Mathematical-reasoning model

The math experiment increases model capacity while retaining the same components:

| Component | Configuration |
|---|---|
| Dataset | Synthetic arithmetic followed by GSM8K |
| Tokenizer | Lossless UTF-8 bytes with BOS, EOS, and PAD |
| Vocabulary | 259 tokens |
| Context | 256 byte tokens |
| Main decoder | 8 layers, width 256, 4 heads, FFN width 1024 |
| Dropout | 0.05 |
| Dynamic depth | Linear SkipLayer router, initial execute probability 0.90 |
| Density target | 0.70 with coefficient 0.10 |
| MoE | 10 experts per layer, top-2 active, expert width 512 |
| Attention | MLA, KV latent rank 64 |
| MLA heads | 32 content dimensions, 32 RoPE dimensions, 64 value dimensions |
| Positions | RoPE with theta 10,000 |
| MTP | One causal Transformer block predicting `t+2` |
| Stored parameters | 23,414,352 |

The UTF-8 byte tokenizer is used instead of the Shakespeare character vocabulary so
that unseen names, symbols, decimals, and numbers remain representable without fitting
a tokenizer on the test set.

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

Every response begins with a machine-readable result:

```text
Question: Ava has 17 apples and gets 8 more. How many apples are there?
Response: <answer>25</answer>
Reasoning: Add the two amounts: 17 + 8 = 25.
```

The completed run used 300 supervised steps:

- Steps 0–149: synthetic arithmetic only.
- Steps 150–299: an equal-sized synthetic/GSM8K mixture.
- Batch size 4.
- AdamW with peak learning rate `3e-4` and cosine decay.
- Main CE + `0.3 × MTP CE` + `0.0001 × MoE balance loss` + scheduled density loss.

Run the same bounded supervised and GRPO pipeline with:

```bash
python run_math_grpo.py \
  --sft-steps 300 \
  --synthetic-steps 150 \
  --grpo-steps 20 \
  --device auto
```

## Quality-focused math GRPO

This is different from the earlier router-only GRPO. It optimizes generated token
probabilities so it can, in principle, improve answers rather than only selecting a
depth.

For each GSM8K training question, the policy samples four continuations of 48 byte
tokens. Numerical strings are normalized so values such as `72`, `72.0`, `$72`, and
`1,072` can be compared correctly. The reward is:

```text
reward =
    1.00 × exact_answer
  + 0.10 × valid_answer_format
  + 0.20 × numeric_closeness
  - 0.20 × repeated_fourgram_rate
  - 0.10 × max(0, compute_fraction - 0.70)
```

The group mean and standard deviation produce the group-relative advantage. Training
uses a per-token clipped policy objective with:

| Setting | Value |
|---|---:|
| GRPO steps | 20 |
| Group size | 4 |
| Rollout temperature | 1.0 |
| Learning rate | `3e-6` |
| PPO clip epsilon | 0.2 |
| Frozen-reference KL coefficient | 0.04 |
| Supervised replay coefficient | 0.5 |
| MTP replay coefficient | 0.3 |

The SkipLayer router and MoE selection routers remain frozen during quality GRPO.
Attention, active experts, token embeddings/shared LM head, normalizations, and MTP
remain trainable. Every RL step includes a supervised replay batch to limit policy
drift, while the original supervised checkpoint supplies the frozen KL reference.

## TensorBoard

Start TensorBoard for both math stages:

```bash
tensorboard \
  --logdir /Users/yashsolanki/Desktop/layer-skipper/experiments/math_grpo_seed42 \
  --port 6007
```

Open [http://localhost:6007](http://localhost:6007).

The supervised dashboard includes:

- Training and validation CE, perplexity, and byte accuracy.
- MTP CE and `t+2` accuracy.
- Average depth, density loss, skip fraction, and estimated FLOPs.
- MoE balance, entropy, and expert-utilization variation.
- Learning rate, gradient norm, step latency, and throughput.

The GRPO dashboard additionally includes:

- Group reward mean, standard deviation, best, and worst values.
- Exact-answer and parseable-answer rates.
- Numeric-closeness, format, repetition, and compute reward components.
- Group advantage, policy ratio, clipping, and frozen-reference KL.
- Supervised replay and MTP replay losses.
- Final matched evaluation metrics under the `final/` group.

If port 6007 is occupied, select another port and open that address instead.

## Generate answers yourself

Run the built-in sample questions against the completed GRPO checkpoint:

```bash
python generate_math.py --max-new-tokens 48 --device auto
```

Supply one or more custom questions by repeating `--question`:

```bash
python generate_math.py \
  --question "If a store has 25 books and sells 9, how many remain?" \
  --question "There are 8 boxes with 6 pencils in each. How many pencils are there?" \
  --question "Sarah has 30 dollars, spends 12, and earns 7 more. How much does she have?" \
  --max-new-tokens 48 \
  --temperature 0 \
  --device auto
```

Request several stochastic answers with:

```bash
python generate_math.py \
  --question "What is 17 multiplied by 6?" \
  --samples-per-question 4 \
  --temperature 0.8 \
  --seed 123 \
  --max-new-tokens 48 \
  --device auto
```

The CLI displays the raw completion, parsed numerical answer, and average executed
layers per generated token.

## Completed math result

The comparison below evaluates the best supervised checkpoint and the final GRPO
checkpoint on the same six held-out GSM8K questions with identical greedy decoding.

| Stage | Validation CE ↓ | Perplexity ↓ | Byte accuracy ↑ | MTP accuracy ↑ | Exact answers ↑ | Parse rate | Layers/token ↓ | FLOPs vs dense ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best supervised | **2.0175** | **7.519** | **46.00%** | **44.66%** | 0/6 | 100% | 7.976 | 0.911× |
| Quality GRPO | 2.2338 | 9.335 | 41.75% | 39.41% | 0/6 | 100% | **7.954** | **0.909×** |

This bounded run did not improve mathematical reasoning. The model learned to emit
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
results/math_grpo_seed42/REPORT.md
results/math_grpo_seed42/results.json
```

To regenerate the matched final report from saved checkpoints:

```bash
python finalize_math_grpo.py --device auto
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
python run_skiplayer_moe_mtp.py \
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
python speculative_decode.py \
  --checkpoint experiments/skiplayer_moe_mtp_mla_rope_seed42/grpo/checkpoints/best_quality_compute.pt \
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
pytest -q
```

The current suite contains 50 tests covering routing semantics, checkpoints, GRPO
objectives, MoR, MoE accounting, MLA/RoPE, MTP, speculative residual sampling, math
tokenization, numerical answer parsing, deterministic synthetic data, and reward
ordering.

## Repository map

| Files | Purpose |
|---|---|
| `model.py`, `config.py`, `router.py` | Transformer, MLA, MoE, MTP, depth routing, and configuration |
| `math_data.py` | GSM8K download, byte tokenizer, deterministic splits, and synthetic curriculum |
| `train_math.py` | Arithmetic/GSM8K supervised training |
| `train_math_grpo.py` | Token-policy quality GRPO with exact numerical rewards |
| `math_training_utils.py` | Math rollouts, rewards, validation, and generation metrics |
| `generate_math.py` | Interactive command-line math generation |
| `run_math_grpo.py` | End-to-end supervised + GRPO orchestration |
| `finalize_math_grpo.py` | Matched checkpoint evaluation and final report generation |
| `speculative_decode.py` | One-block MTP speculative decoding and latency benchmark |
| `train.py`, `train_grpo.py`, `train_paper_grpo.py` | Dense/SkipLayer supervised and routing-GRPO training |
| `train_moe_mtp.py`, `run_skiplayer_moe_mtp.py` | Shakespeare MoE/MTP/MLA adaptation and orchestration |
| `train_mor_grpo.py`, `train_mor_skip.py`, `train_mor_skip_grpo.py` | MoR and hybrid training |
| `evaluation.py`, `evaluate.py`, `report.py`, `plots.py` | Evaluation and reporting |
| `logging_utils.py` | CSV and TensorBoard logging |
| `tests/` | Automated correctness tests |

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
