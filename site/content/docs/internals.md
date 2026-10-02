---
title: Reading the engine
description: Five crates, meant to be read in order. By the end you will have built a transformer rather than configured one.
---

Kvad has two jobs. One is to be worth running. The other is to explain how
this works: every matrix multiply, every derivative, every attention head and
every rotation is code in the repository, with the reasoning for each constant
next to it.

Those sound opposed and mostly are not. Almost everything that makes inference
fast is arithmetic you can read: block-wise quantisation, an integer dot
product, a tiled matrix product, a cache you do not recompute. Where
legibility and speed do conflict, the documentation says which was chosen and
measures what it cost.

## The crates, in order

| | Crate | What it is | Depends on |
|---|---|---|---|
| 1 | [`nervus`](/docs/internals/nervus/) | A neural network and backpropagation, from scratch. Trains on MNIST, then trains a GPT on a text file. | nothing |
| 2 | [`kvad`](/docs/internals/engine/) | Transformer inference from scratch. Six architectures, real Hugging Face weights. | a hub client, a tokeniser, safetensors |
| 3 | [`kvad-gpu`](/docs/internals/gpu/) | The same forward passes on the GPU, and the image and video models. | candle, Metal |
| 4 | [`kvad-tui`](/docs/internals/tui/) | The terminal app. | ratatui |
| 5 | [`kvad-serve`](/docs/internals/serve/) | The HTTP server and web UI. | axum, SQLite, Svelte |

The first two use no machine-learning framework at all. The third is the same
model handed to one, so the two can be compared, and so the hand-written
version has something honest to be measured against. The last two are
applications, and the place to see what an engine has to expose before anything
can be built on it.

## What these pages are

The five pages that follow are the repository's own long-form documentation,
rendered here as it is written there. They are longer and more discursive than
the guide: they explain why a kernel is shaped the way it is, and they keep the
measurements that turned out to be wrong, because the method that caught them
is the useful part.

- [Training a LoRA](/docs/internals/tune/) describes the SDXL training loop.
- [The roadmap](/docs/internals/roadmap/) says what has to be true before Kvad
  is a serving engine, in dependency order.

## Running from a checkout

```bash
git clone https://github.com/bisand/kvad && cd kvad
./scripts/get-mnist.sh
cargo test                                      # includes a gradient check
cargo run --release -p nervus --bin train_mnist
cargo run --release -p nervus --bin train_text -- --data docs/nervus.md
cargo run --release -p kvad -- run --prompt "Why is the sky blue?"
```

```text
epoch  1  loss 0.2085  test accuracy 96.32%  (2.1s)
epoch 10  loss 0.0072  test accuracy 97.98%  (21.0s)
```

Design notes for individual features, such as GGUF, images, video, LoRA and
the cluster, are in the repository's
[`docs/`](https://github.com/bisand/kvad/tree/master/docs) as `*-plan.md`.
