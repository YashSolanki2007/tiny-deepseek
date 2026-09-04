# SkipLayer paper reproduction protocol

This protocol targets the method in **Learning to Skip for Language Modeling**
(arXiv:2311.15436) before any GRU-router or GRPO extension is considered.

## Faithful method choices

- A separate linear `d_model -> 2` router is attached to every Transformer layer.
- The router consumes the layer-normalized input and emits `[skip, execute]` logits.
- Straight-through Gumbel-Softmax produces a hard binary forward decision and a
  soft backward surrogate.
- Skipping is exact identity; execution applies the complete attention-then-FFN
  residual block.
- Active attention queries retain keys and values from every causal-context token.
- The capacity loss is the paper's **sum** over layers:

  `CE + 0.1 * sum_l (density_l - P)^2`.

- There is no density-loss warmup in the paper protocol.
- Training uses fixed-decay Adafactor with beta1=0, beta2=0.99, learning rate
  0.1 for 10,000 updates, then inverse-square-root decay.
- Adafactor's update-RMS clipping is retained; no separate gradient clipping is
  applied because the paper does not report one.
- Dropout is zero and FFN width is eight times the model width.
- Evaluation uses greedy router decisions.
- Dense and SkipLayer models are compared at matched expected effective depth,
  e.g. dense 4L versus SkipLayer 8L at 50% density.

## Necessary scale substitutions

The paper's private 1.6T-token corpus, 32K SentencePiece model, TPU gather/scatter
kernels, 408M-5.79B parameter models, and 24 one-shot benchmark suite are not
publicly reproducible from this repository. This analogue therefore uses Tiny
Shakespeare, character tokens, smaller widths/depths, and PyTorch/MPS/CUDA/CPU.
These substitutions are always recorded in each experiment's `config.json`.

The PyTorch training path preserves the paper's routing semantics but evaluates
the full candidate block before applying the hard gate. Consequently its logged
FLOPs are theoretical and include the paper's always-on key/value projection
overhead; it does not claim sparse-kernel wall-clock acceleration.

## Commands

Fast end-to-end verification:

```bash
python -m tiny_deepseek.workflows.run_paper_experiments --preset smoke
```

Meaningful Tiny Shakespeare comparison:

```bash
python -m tiny_deepseek.workflows.run_paper_experiments --preset core --seeds 42 43 44
```

The exact `0.1` learning rate can be too aggressive after reducing the paper's
batch/model/data scale by orders of magnitude. A scale-adapted ablation must be
explicitly labeled and can be run without changing any routing semantics:

```bash
python -m tiny_deepseek.workflows.run_paper_experiments --preset core --paper-learning-rate 0.01
```

Small analogue of the depth/density rows in Table 1:

```bash
python -m tiny_deepseek.workflows.run_paper_experiments --preset table1 --seeds 42 43 44
```

TensorBoard:

```bash
tensorboard --logdir artifacts/experiments/paper_core
```
