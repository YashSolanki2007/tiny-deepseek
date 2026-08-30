# Learned Dynamic Depth on Tiny Shakespeare

This project implements a small character-level decoder-only Transformer and a
dynamic-depth variant whose binary per-token, per-layer gates are learned with a
straight-through estimator (STE). Training is ordinary backpropagation with AdamW;
there is no reinforcement learning or policy-gradient code.

The implementation intentionally uses **Stage A routing**: every Transformer block
is evaluated during training and inference, and the binary gate chooses whether its
candidate state is accepted. Reported utilization, skipped blocks, and block FLOPs
are therefore theoretical. The project does not claim wall-clock sparse acceleration.

## Architecture

The dense model is an 8-layer pre-norm GPT-style Transformer by default. The dynamic
model computes a full block candidate and applies

```python
candidate = block(x)
x = (1 - gate) * x + gate * candidate
```

Thus `gate=0` is exactly identity and `gate=1` exactly matches the complete block.
The GRU router keeps a `[batch, time, router_dim]` state and advances it across
Transformer depth. The MLP router ablation has no depth memory. Both produce
deterministic hard gates in the forward pass while gradients flow through sigmoid
probabilities:

```python
soft = sigmoid(logit)
hard = (soft >= 0.5).float()
gate = soft + (hard - soft).detach()
```

The gate output bias starts at 2.2, so the initial soft probability is about 0.9.
Because thresholding is deterministic rather than Bernoulli sampling, this means
essentially all hard gates begin open; 0.9 is the differentiable expected utilization.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first training or validation command downloads Tiny Shakespeare to
`data/input.txt`, builds the sorted character vocabulary, and uses a deterministic
90/10 contiguous split.

## Train

Dense baseline:

```bash
python train.py \
  --model dense \
  --output-dir runs/dense
```

GRU dynamic-depth model with the default linear compute penalty:

```bash
python train.py \
  --model dynamic \
  --router gru \
  --lambda-compute 0.01 \
  --compute-loss linear \
  --output-dir runs/dynamic_gru_lambda_0.01
```

MLP router ablation:

```bash
python train.py \
  --model dynamic \
  --router mlp \
  --lambda-compute 0.01 \
  --output-dir runs/dynamic_mlp_lambda_0.01
```

Target-compute objective (50% expected utilization):

```bash
python train.py \
  --model dynamic \
  --compute-loss target \
  --target-compute 0.50 \
  --lambda-compute 0.01 \
  --output-dir runs/dynamic_target_0.50
```

Defaults match the requested research configuration: context 128, width 256,
4 heads, 8 layers, FFN width 1024, dropout 0.1, router width 32, AdamW at 3e-4,
and 5,000 steps. `--device auto` selects CUDA, then Apple MPS, then CPU. Use
`python train.py --help` for all model, optimizer, schedule, and evaluation options.

The compute coefficient is zero for the first 10% of steps, ramps linearly from
10%-30%, and remains at its target afterward. Logs include CE, total loss, compute
loss, soft/hard utilization, effective depth, skipped fraction, and every layer's
utilization. Each run saves `checkpoint.pt` and `summary.json`.

## Validate and generate

```bash
python validate.py \
  --checkpoint runs/dynamic_gru_lambda_0.01/checkpoint.pt \
  --eval-iters 100

python generate.py \
  --checkpoint runs/dynamic_gru_lambda_0.01/checkpoint.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 500 \
  --temperature 0.8 \
  --top-k 40
```

Generation reports actual tokens/second plus average layers per generated token,
the theoretical skip fraction, and per-layer utilization. `--temperature 0` uses
greedy decoding; `--top-k 0` disables top-k filtering.

## Routing visualization

```bash
python visualize_routing.py \
  --checkpoint runs/dynamic_gru_lambda_0.01/checkpoint.pt \
  --text $'ROMEO:\nWhat light through yonder window breaks?' \
  --mode soft \
  --output runs/dynamic_gru_lambda_0.01/routing_soft.png

python visualize_routing.py \
  --checkpoint runs/dynamic_gru_lambda_0.01/checkpoint.pt \
  --mode hard \
  --output runs/dynamic_gru_lambda_0.01/routing_hard.png
```

Hard mode also prints an ASCII token/layer gate matrix.

## Full lambda sweep, CSV, and Pareto curve

The sweep runs one dense model and dynamic models at
`0, 0.001, 0.005, 0.01, 0.02, 0.05`, then aggregates only real run summaries:

```bash
python run_experiments.py \
  --runs-dir runs/gru_sweep \
  --router gru \
  --max-steps 5000
```

Unknown arguments are forwarded to `train.py`, so a quick integration sweep is:

```bash
python run_experiments.py \
  --runs-dir runs/smoke_sweep \
  --max-steps 20 \
  --eval-interval 20 \
  --eval-iters 2 \
  --batch-size 4 \
  --context-length 32 \
  --d-model 64 \
  --d-ff 128 \
  --n-layers 4
```

Outputs are `results.csv` and `pareto.png` inside the sweep directory. To aggregate
selected runs later:

```bash
python aggregate_results.py \
  --runs-dir runs \
  --csv results.csv \
  --plot pareto.png
```

The aggregator exits if there are no summaries; it never invents result rows.

## Speed comparison

Compare actual Stage A generation throughput and theoretical routing side by side:

```bash
python benchmark.py \
  --checkpoints \
    runs/dense/checkpoint.pt \
    runs/dynamic_gru_lambda_0.01/checkpoint.pt \
  --tokens 100 \
  --output benchmark.csv
```

The benchmark performs five unreported warmup tokens by default before timing.

Dynamic Stage A may be slower than dense because it still evaluates all blocks and
adds router work. Genuine per-token sparse self-attention execution is deliberately
out of scope for this correctness-first prototype.

## Tests

```bash
pytest -q
```

Tests cover exact identity/full-block gate semantics, binary gate shape, both router
types, nonzero STE router gradients, initial ~0.9 soft probability, zero compute
gradient at lambda zero, compute warmup, and utilization pressure from a large
linear penalty.
