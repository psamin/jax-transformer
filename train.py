"""Train the from-scratch JAX GPT as a character-level language model.

By default it trains on `input.txt` (a small built-in corpus ships in the repo,
so it runs out of the box). Drop in your own `input.txt` for something bigger.

    python train.py                       # train + sample with the defaults
    python train.py --steps 5000 --n_layer 6 --n_embd 256

Everything is CPU-friendly at the default size; it just trains faster on a GPU/TPU.
"""
from __future__ import annotations

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gpt import GPTConfig, forward, generate, init_params, loss_fn

FALLBACK_TEXT = (
    "the transformer reads a sequence of tokens and predicts the next one. "
    "attention lets every position look back at every earlier position, so the "
    "model learns which words matter for what comes next. stack a few of these "
    "blocks, train on enough text, and it starts to speak.\n"
) * 200


def load_data(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError("empty")
    except (FileNotFoundError, ValueError):
        print(f"[data] {path} not found or empty — using the built-in fallback corpus.")
        text = FALLBACK_TEXT
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    return data, stoi, itos, len(chars)


def get_batch(rng: np.random.Generator, data, batch_size, block_size):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    data, stoi, itos, vocab_size = load_data(args.input)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    print(f"[data] {len(data):,} tokens · vocab {vocab_size}")
    print(f"[model] {cfg.n_layer}L·{cfg.n_head}H·{cfg.n_embd}D · ctx {cfg.block_size}")

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, cfg)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[model] {n_params/1e6:.2f}M parameters")

    optimizer = optax.adamw(args.lr)
    opt_state = optimizer.init(params)

    @partial(jax.jit, static_argnums=())
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, cfg)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for it in range(1, args.steps + 1):
        x, y = get_batch(rng, data, args.batch_size, cfg.block_size)
        params, opt_state, loss = step(params, opt_state, x, y)
        if it % 100 == 0 or it == 1:
            print(f"step {it:>5} · loss {float(loss):.4f} · {(time.time()-t0):.1f}s")

    # Sample a bit of text from the trained model.
    print("\n--- sample ---")
    key, sub = jax.random.split(key)
    start = jnp.zeros((1, 1), dtype=jnp.int32)
    out = generate(params, cfg, sub, start, max_new_tokens=400, temperature=0.8, top_k=20)
    print("".join(itos[int(i)] for i in out[0]))


if __name__ == "__main__":
    main()
