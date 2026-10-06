---
title: Platforms
description: What runs where, and what each machine gets.
---

Kvad is developed and measured on an M5 Pro with 48 GB. That is the machine
the numbers in this documentation come from, unless a number says otherwise.

## Support

| | macOS, Apple silicon | Linux, x86-64 and arm64 |
|---|---|---|
| `kvad` command line | Yes | Yes |
| `kvad-serve` and the web UI | Yes | Yes, CPU only |
| `kvad-tui` | Yes | No |
| Language models on the CPU | Yes | Yes |
| Language models on the GPU | Yes, Metal | No |
| Images, video, LoRA training | Yes, Metal | No |
| Background service | launchd | systemd |

Intel Macs and Windows have no build.

There is no CUDA build. The GPU engine is written for Metal, and its fastest
kernels are specific to Apple's GPUs.

## Apple silicon

Any M-series Mac runs everything. What changes between them is speed and how
large a model fits.

**M5 and later.** The M5's GPU has matrix units that ordinary Metal shaders
never reach. Kvad has its own kernels for them, used for dense and quantised
matrix products in image models, video models and language-model prefill. They
need the Apple10 GPU family, and Kvad checks for it when it loads a model.
[Apple silicon](/apple-silicon/) has the measurements.

**M1 to M4.** The same models run through the standard Metal kernels. Nothing
is switched off, and nothing needs configuring: the accelerated path is simply
not taken.

To turn the M5 kernels off, for a comparison of your own:

```bash
KVAD_GPU_MPP=0 kvad serve
```

## Memory

Unified memory is what decides which models you can run. As a rule, a model
needs its weights at the precision you load it at, plus a KV cache that grows
with the conversation.

| Model | Precision | In memory |
|---|---|---|
| Qwen2.5-0.5B | q8 | 0.6 GB |
| Qwen3-14B | q8 | about 15 GB |
| DeepSeek-V2-Lite (15.7 B parameters) | q8 | 17.7 GB |
| Stable Diffusion 1.5 | f16 | 1.9 GB |
| SDXL | f16 | 6.4 GB |
| FLUX.1-schnell | q8 | 18.3 GB |
| FLUX.1-dev | q8 | 17 GB |
| Qwen-Image | q8 | about 29 GB |

The server will not load a model that does not fit beside the ones already in
memory. It says what is holding the memory instead, and never unloads
something to make room. `kvad search` marks each result with the precision at
which it fits this machine.

## Linux

The hand-written CPU engine is the same code on both systems. On arm64 it uses
NEON, the dot-product extension and `i8mm` where the processor has them. A
Linux build is what the release workflow produces; it is tested less than the
Mac build, because nobody develops on it.
