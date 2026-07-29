<div align="center">

# transformers
**A repo for all the different transformers I experiment with
**A (GPT-style) transformer built with [JAX](https://github.com/google/jax) and trained on the OpenWebText dataset.**

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

The goal was to train a small transformer on JAX, using Flax NNX to build the model architecture, Optax for loss function and optimizer creation, and training on accelerated hardware with the help of Orbax and XLA. 

```

## Notes

- If you use Kaggle instead of Colab, then experiment with data parallelism by utilizing the TPU v5e-8 (This is during 2026, things may change in future)

## License

MIT — see [LICENSE](LICENSE).
