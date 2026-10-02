---
title: The web UI
description: Twelve pages over the same server the API runs on, built into the binary.
---

The web UI is part of `kvad-serve`. There is nothing to install beside it and
nothing to serve it with: open the address the server listens on.

```bash
kvad serve          # then http://127.0.0.1:5823
```

With the background service installed it is already there.

## The pages

| Page | What it is for |
|---|---|
| **Dashboard** | Throughput, queue depth, memory and disk, and whatever is running. |
| **Models** | What is on this machine, what is on the Hub, and what is loaded. Download, load on a chosen backend, unload, delete. |
| **Chat** | Conversations with the loaded model, kept on the server, with the numbers for every reply. |
| **Playground** | Raw completion, two models side by side, the tokeniser, and the probability of every token the model considered. |
| **Images** | A prompt, the denoiser's settings, each step as it happens, and a gallery. |
| **Videos** | Text or a picture to video, with sound: the generation as it goes, and a gallery. |
| **Training** | Train a small model on a dataset, with its loss curve drawn as it falls. |
| **Datasets** | Upload text, or crawl a documentation site into a corpus. |
| **Evals** | Prompt suites across models, and perplexity on held-out text. |
| **Benchmarks** | Variants measured against each other: interleaved rounds, medians and ranges. |
| **Monitoring** | The machine, recent requests and what each route costs, and the log. |
| **Settings** | The server's address, the data directory, accounts, sessions and API keys. |

The API's own reference is at `/api` on the same address.

## Chat

Pick a model at the top; one on disk and not in memory is loaded when you send
the first message, if it fits. Replies render as Markdown, with code
highlighted. Under each reply are its numbers: prefill time, decode rate, and
how many tokens came from the cache.

A reasoning model's working is shown collapsed above its answer, and open while
the answer has not started. It is not saved: the message kept in the
conversation is the answer.

## Playground

Generation computes a distribution over the whole vocabulary at every step and
throws away everything except the token it drew. The playground keeps it. Click
any token to see what else was in the running, with the model's own
probabilities, and a mark on each showing whether top-k and top-p left it in
play.

Beside it is the tokeniser. `2026` is five tokens in Qwen's vocabulary: a
space, then each digit alone.

## Images and videos

Both show the work as it happens. An image has a preview at every denoising
step, made from the latent without a decode, so you can stop a picture that is
going wrong at step five rather than step thirty. A video is a job: it carries
on if you close the page, and the gallery shows how far along it is.

Every image and video is kept with the settings that made it, seed included,
so any of them can be made again.

## Benchmarks

A benchmark run visits every variant once per round, for several rounds, and
reports the median and the range with every sample kept. It drops the KV cache
before each timed generation, and it refuses to start while a training run, an
eval or another benchmark is going. Two ranges that overlap are two numbers
that have not been told apart, and the page shows that.

## Light and dark

The UI follows the system's setting and has a switch for the other one.

## Using it from another machine

By default the server listens on loopback only, with no authentication. To
reach it from elsewhere, configure an [authentication](/docs/authentication/)
mode and bind a reachable address.
