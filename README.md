<div align="center">

# transformers

**A repo for all the different types of transformers I experiment with, including implementations in PyTorch and JAX.**

The PyTorch implementation is a GPT-style transformer based on [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1463s) and is trained to generate Shakespearean text.

The JAX implementation is a GPT-style transformer trained on the OpenWebText dataset.

Both architectures reference *Attention Is All You Need*.

![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-pure-blue)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Background

I started this repo after working through the original transformer paper,  
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al., [Andrej Karpathy's transformer tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1463s), and DeepMind's Scaling Book,  
[*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/).

The goal is to experiment with different types of transformers in PyTorch and JAX while learning how their architectures, training processes, and optimizations work from the ground up.

The PyTorch implementation references Andrej Karpathy's tutorial and recreates a GPT-style transformer trained to generate Shakespearean text.

For the JAX implementation, I use Flax NNX to build the model architecture, Optax for the loss function and optimizer, Orbax for checkpointing, and XLA for accelerated training.

## Notes

- If you use Kaggle instead of Colab, you can experiment with data parallelism using the TPU v5e-8.
- This information is accurate as of 2026, but the available hardware may change in the future.

## License

MIT. See [LICENSE](LICENSE).
```
