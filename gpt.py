"""A decoder-only (GPT-style) transformer written from scratch in pure JAX.

No Flax / Haiku — the parameters are a plain pytree of dicts and every layer
(embeddings, causal multi-head self-attention, MLP, LayerNorm) is spelled out so
the math is visible. Everything is batched over (B, T) and jit/grad-friendly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


# ---------------------------------------------------------------------------
# Parameter initialization (GPT-style: small normal init, zeros for biases).
# ---------------------------------------------------------------------------
def _normal(key, shape, std=0.02):
    return random.normal(key, shape) * std


def _linear(key, fan_in, fan_out):
    return {"w": _normal(key, (fan_in, fan_out)), "b": jnp.zeros((fan_out,))}


def _ln():
    # gamma/beta are created per-call with the right width below.
    return None


def init_params(key, cfg: GPTConfig):
    keys = iter(random.split(key, 4 + cfg.n_layer * 4))
    C = cfg.n_embd
    params = {
        "wte": _normal(next(keys), (cfg.vocab_size, C)),      # token embedding
        "wpe": _normal(next(keys), (cfg.block_size, C)),      # positional embedding
        "blocks": [],
        "ln_f": {"g": jnp.ones((C,)), "b": jnp.zeros((C,))},
        "head": _normal(next(keys), (C, cfg.vocab_size)),     # final projection to logits
    }
    for _ in range(cfg.n_layer):
        params["blocks"].append({
            "ln1": {"g": jnp.ones((C,)), "b": jnp.zeros((C,))},
            "attn": {
                "qkv": _linear(next(keys), C, 3 * C),
                "proj": _linear(next(keys), C, C),
            },
            "ln2": {"g": jnp.ones((C,)), "b": jnp.zeros((C,))},
            "mlp": {
                "fc": _linear(next(keys), C, 4 * C),
                "proj": _linear(next(keys), 4 * C, C),
            },
        })
    return params


# ---------------------------------------------------------------------------
# Layers.
# ---------------------------------------------------------------------------
def layer_norm(x, p, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return p["g"] * (x - mean) / jnp.sqrt(var + eps) + p["b"]


def linear(x, p):
    return x @ p["w"] + p["b"]


def causal_self_attention(x, p, n_head):
    # x: (B, T, C)
    B, T, C = x.shape
    hs = C // n_head
    qkv = linear(x, p["qkv"])                       # (B, T, 3C)
    q, k, v = jnp.split(qkv, 3, axis=-1)            # each (B, T, C)
    # (B, T, C) -> (B, n_head, T, head_size)
    split_heads = lambda t: t.reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    att = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(hs)   # (B, nh, T, T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    att = jnp.where(mask, att, -1e10)               # causal: no peeking ahead
    att = jax.nn.softmax(att, axis=-1)
    y = jnp.einsum("bhqk,bhkd->bhqd", att, v)       # (B, nh, T, hs)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)    # re-merge heads
    return linear(y, p["proj"])


def mlp(x, p):
    return linear(jax.nn.gelu(linear(x, p["fc"])), p["proj"])


def block(x, p, n_head):
    # Pre-norm residual blocks (as in GPT-2).
    x = x + causal_self_attention(layer_norm(x, p["ln1"]), p["attn"], n_head)
    x = x + mlp(layer_norm(x, p["ln2"]), p["mlp"])
    return x


# ---------------------------------------------------------------------------
# Forward / loss.
# ---------------------------------------------------------------------------
def forward(params, idx, cfg: GPTConfig):
    # idx: (B, T) int32 token ids -> logits (B, T, vocab)
    B, T = idx.shape
    x = params["wte"][idx] + params["wpe"][:T]      # token + positional embeddings
    for p in params["blocks"]:
        x = block(x, p, cfg.n_head)
    x = layer_norm(x, params["ln_f"])
    return x @ params["head"]


def loss_fn(params, idx, targets, cfg: GPTConfig):
    logits = forward(params, idx, cfg)              # (B, T, vocab)
    return cross_entropy(logits, targets)


def cross_entropy(logits, targets):
    # Mean next-token cross-entropy.
    logp = jax.nn.log_softmax(logits, axis=-1)
    tok = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -tok.mean()


# ---------------------------------------------------------------------------
# Autoregressive sampling.
# ---------------------------------------------------------------------------
def generate(params, cfg: GPTConfig, key, idx, max_new_tokens, temperature=1.0, top_k=None):
    """idx: (B, T) prompt tokens. Returns (B, T + max_new_tokens)."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]         # crop to the context window
        logits = forward(params, idx_cond, cfg)[:, -1, :] / temperature
        if top_k is not None:
            kth = jnp.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = jnp.where(logits < kth, -jnp.inf, logits)
        key, sub = random.split(key)
        next_id = random.categorical(sub, logits, axis=-1)[:, None]
        idx = jnp.concatenate([idx, next_id], axis=1)
    return idx
