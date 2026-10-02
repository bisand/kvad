---
title: Introduction
description: Kvad runs language, image and video models on your own machine, from one install, with every kernel in a repository you can read.
---

Kvad is a local inference engine written in Rust. It runs chat models, image
models and a video model on your own hardware, and it comes with a command
line, a terminal app, a web UI and an OpenAI-compatible HTTP API. Nothing in it
is Python, and nothing leaves the machine unless you ask for a model from
Hugging Face.

It is developed and measured on Apple silicon, and the M5's GPU is the machine
it is tuned for. It also runs on Linux, on the CPU.

## What you get

Four binaries, installed together:

| Binary | What it is |
|---|---|
| `kvad` | The command line: find, download, run, chat, train, and drive a server. |
| `kvad-serve` | The HTTP server and the web UI, in one file. |
| `kvad-tui` | The terminal app: browse, download, switch backend, chat. |
| `kvad-gpu` | The GPU engine on its own, and the LoRA trainer. |

macOS on Apple silicon gets all four. Linux gets `kvad` and a CPU-only
`kvad-serve`; [Platforms](/docs/platforms/) has the details.

## What it runs

- **Language models** in six architectures: GPT-2, the Llama family (Llama,
  Mistral, Qwen2/2.5/3, SmolLM2), DeepSeek V2 and V3, Qwen3.5/3.8 and
  Qwen3-Next. Weights come straight from Hugging Face as safetensors.
- **Image models**: Stable Diffusion 1.5 and SDXL with their fine-tunes,
  FLUX.1-schnell and Qwen-Image, with LoRAs applied per request.
- **Video**: LTX-2.5, with sound, from a prompt or from a picture.

## Where to start

1. [Install](/docs/install/) it. One command, no root.
2. Follow the [quick start](/docs/quickstart/) to a first reply and a first image.
3. Open the [web UI](/docs/web-ui/), or point a client at the [API](/docs/api/).

## What it is not, yet

Kvad runs one generation at a time. Requests queue behind one another and the
queue depth is a number on the dashboard. There is no continuous batching and
no paged KV cache, so it is a very good engine for one person and a poor one
for a team. The [roadmap](/docs/internals/roadmap/) says what has to be built
before that changes.

## If you want to know how it works

Every matrix multiply, every quantisation kernel and every attention head is
code in the repository, with the reasoning beside it and the measurements that
turned out wrong left in. [Reading the engine](/docs/internals/) is the way in.
