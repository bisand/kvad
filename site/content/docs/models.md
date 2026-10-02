---
title: Models and backends
description: Finding and downloading models, the six backends a language model can load on, and what the server does when several models want the same memory.
---

## Finding and downloading

Models come from [Hugging Face](https://huggingface.co), as safetensors, and
are named by their repository.

```bash
kvad search smollm                 # says which results this build can run
kvad info --model Qwen/Qwen3-14B   # the config, without downloading weights
kvad pull Qwen/Qwen2.5-1.5B-Instruct
kvad ls                            # what is on this machine, and how big
kvad use Qwen/Qwen2.5-1.5B-Instruct
kvad rm  HuggingFaceTB/SmolLM2-135M-Instruct
```

`search` reads each result's config and reports a status: `chat` or
`completion`, the precision at which it fits in memory, or the reason it cannot
run. A model published only in a format the engine does not read, such as AWQ's
`I32` weights, says so and suggests looking for a bf16 or f16 publication.

For a gated model, sign in to Hugging Face the usual way. Kvad reads the same
token other tools do: `HF_TOKEN`, or the `token` file under `HF_HOME`.

Once a model is on disk, nothing asks the network for it again. Loads are from
the cache first, and only `pull` talks to the Hub.

## Architectures

```bash
kvad arch
```

| Architecture | Covers |
|---|---|
| `gpt2` | GPT-2 |
| `llama` | Llama 2/3, Mistral, Qwen2/2.5/3, Qwen3-MoE, SmolLM2 |
| `deepseek_v2` | DeepSeek V2: latent attention, 64 routed experts |
| `deepseek_v3` | DeepSeek V3/R1: 256 experts in 8 groups |
| `qwen3_5` | Qwen3.5/3.8: gated delta net with gated attention |
| `qwen3_next` | Qwen3-Next and Coder-Next |

All six run on the CPU. The GPU runs five of them; `deepseek_v3` is CPU only.

## Backends

A language model loads on one of six backends: two devices, three precisions.

| Backend | Device | Weights |
|---|---|---|
| `cpu-f32` | CPU | 32-bit floats, as published |
| `cpu-q8` | CPU | 8-bit, block-wise quantised |
| `cpu-q4` | CPU | 4-bit |
| `gpu-bf16` | Metal | 16-bit floats |
| `gpu-q8` | Metal | 8-bit |
| `gpu-q4` | Metal | 4-bit |

With nothing chosen, the server loads on `gpu-q8` wherever the GPU can run the
architecture, and on `cpu-q8` otherwise. That default came from measuring, on
an M5 Pro, medians of interleaved rounds:

| Model | `cpu-q8` | `gpu-q8` |
|---|---|---|
| GPT-2 medium | 127.9 tok/s | 292.5 tok/s |
| Qwen2.5-0.5B | 117.0 tok/s | 209.6 tok/s |
| DeepSeek-V2-Lite | 26.7 tok/s | 39.1 tok/s |

Choose one yourself per load:

```bash
kvad load Qwen/Qwen3-14B --backend gpu-q4
kvad run --quant q4 --prompt "…"        # the CPU backend at that precision
```

In the terminal app, `p` cycles through all six. In the web UI, the Models
page has a picker on each model.

Quantising is done once. The quantised weights are written to
`~/.cache/kvad/quant` and memory-mapped on later loads, so the second load of a model is
much faster than the first. `kvad cache` lists those files and `kvad cache
clear` deletes them.

## What is in memory

```bash
kvad ps
```

```text
MODEL                                              CHARGED  CONTEXT  CACHED
stabilityai/stable-diffusion-xl-base-1.0@gpu-bf16  6.4 GB   0        0       base model

memory: 29.6 GB of 36.0 GB left · each charged for 32768 tokens of KV cache
```

Several models can be in memory at once. Each is charged for its weights and
for a full context of KV cache, so a model that has been admitted can always
finish its conversation.

Three rules, all of them deliberate:

- A model is loaded **if it fits** beside what is already there.
- Nothing is **ever unloaded to make room**. A load that does not fit is
  refused with a message naming what holds the memory.
- A request that names a model on disk and not in memory **loads it**, under
  the same rule.

```bash
kvad load Qwen/Qwen2.5-1.5B-Instruct
kvad unload Qwen/Qwen2.5-1.5B-Instruct@gpu-q8
kvad unload                        # all of them
```

## Image and video models

These are named the same way and pulled with the same command. A repository
whose model is one file is named by the repository; one with several, as
`repo:file.safetensors`; a file on this machine, by its path.

```bash
kvad pull stabilityai/stable-diffusion-xl-base-1.0
kvad pull black-forest-labs/FLUX.1-schnell
kvad pull city96/Qwen-Image-gguf:Q4_K_S
kvad pull Lightricks/LTX-2.5
```

Community GGUF files of Qwen-Image's, FLUX.1-schnell's and LTX-2.5's
transformers load by the repository and the quantisation together. See
[Images](/docs/images/) and [Video](/docs/video/).
