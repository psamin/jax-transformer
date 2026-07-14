<div align="center">

# jax-transformer

**A decoder-only (GPT-style) transformer implemented from scratch in [JAX](https://github.com/google/jax).**

The architecture follows *Attention Is All You Need*

![JAX](https://img.shields.io/badge/JAX-pure-blue)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Background

I built this after working through the original transformer paper,
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
and DeepMind's Scaling book,
[*How To Scale Your Model*]([https://arxiv.org/abs/2203.15556](https://jax-ml.github.io/scaling-book/index)) (Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.).

The goal was to implement the decoder-only transformer directly from the paper:
token + positional embeddings, multi-head causal self-attention, the position-wise
feed-forward network, residual connections, and layer normalization — then train it
end to end on next-character prediction. [`gpt.py`](gpt.py) is the whole model;
[`train.py`](train.py) is the training loop and sampler.

## Architecture

```
tokens ─► embed (token + positional)
        └─► [ block ] × n_layer
              ├─ x + Attention(LayerNorm(x))   # causal multi-head self-attention
              └─ x + MLP(LayerNorm(x))         # GELU feed-forward
        └─► LayerNorm ─► linear head ─► logits over the vocabulary
```

- **Causal self-attention** — each position attends only to itself and earlier
  positions (lower-triangular mask), computed for all heads at once with `einsum`.
- **Pre-norm residual blocks** — LayerNorm *before* each sublayer, as in GPT-2,
  which trains more stably as depth grows.
- **Pure functional params** — a nested dict of arrays; the forward pass, loss,
  and sampler are plain functions, so `jax.jit`, `jax.grad`, and `optax` just work.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
```

That trains a small model on the built-in corpus ([`input.txt`](input.txt)) on CPU
in a couple of minutes and prints a text sample at the end. Point it at your own
corpus by replacing `input.txt`, or scale it up:

```bash
python train.py --input shakespeare.txt --steps 5000 --n_layer 6 --n_head 6 --n_embd 384
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--steps` | 2000 | training iterations |
| `--block_size` | 128 | context length |
| `--n_layer` / `--n_head` / `--n_embd` | 4 / 4 / 128 | model size |
| `--batch_size` | 32 | sequences per step |
| `--lr` | 3e-4 | AdamW learning rate |

## What training looks like

```
[data] 8,000 tokens · vocab 42
[model] 4L·4H·128D · ctx 128
[model] 0.82M parameters
step     1 · loss 3.7612 · 0.9s
step   500 · loss 1.8433 · 21.4s
step  2000 · loss 1.2107 · 84.7s

--- sample ---
the transformer reads the sequence and predicts the next token, and attention
lets every position look back at what came before ...
```

(Numbers vary by machine and corpus — the point is the loss falls and the samples
start to look like the training text.)

## Project structure

```
gpt.py            model: params init, attention, MLP, blocks, forward, loss, generate
train.py          char-level tokenizer, batching, AdamW training loop, sampling
input.txt         small built-in corpus so it runs with zero setup
requirements.txt  jax[cpu], optax, numpy
```

## Notes

- Runs on CPU; it just goes faster on a GPU/TPU (JAX picks the accelerator up
  automatically — no code change).
- Sampling supports `temperature` and `top_k`.
- Natural next steps: weight tying, dropout, a cosine LR schedule, and a proper
  train/val split.

## License

MIT — see [LICENSE](LICENSE).
