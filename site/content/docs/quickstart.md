---
title: Quick start
description: From an empty machine to a reply, a conversation in the browser, and a picture, in about five minutes.
---

This assumes you have [installed](/docs/install/) Kvad and said yes to the
background service. Without the service every command below still works: it
runs in the process instead of going to a server.

## 1. Ask something

```bash
kvad run --prompt "Why is the sky blue?"
```

With no model chosen this downloads `HuggingFaceTB/SmolLM2-135M-Instruct`, a
270 MB model that is small enough to arrive in seconds, and answers. The first
line of output says where the command ran:

```text
kvad: http://127.0.0.1:5823 (the kvad service; --local to run in this process instead)
```

## 2. Get a better model

`search` asks Hugging Face and says which results this build can run, and at
which precision they fit in this machine's memory.

```bash
kvad search qwen2.5
kvad pull Qwen/Qwen2.5-1.5B-Instruct
kvad use Qwen/Qwen2.5-1.5B-Instruct
```

`use` makes it the default for every command that names no model.

With [Tab completion](/docs/cli/#tab-completion) set up, `kvad use <Tab>`
lists the models you have, and `kvad help` lists everything `kvad` does.

## 3. Have a conversation

```bash
kvad chat
```

Or in the terminal app, which also browses and downloads models:

```bash
kvad-tui
```

## 4. Open the web UI

The service is already listening. Open
[http://127.0.0.1:5823](http://127.0.0.1:5823).

If you did not install the service, start a server by hand:

```bash
kvad serve
```

Chat, the playground, images, videos, training, evals and benchmarks are all
pages there. See [The web UI](/docs/web-ui/).

## 5. Make a picture

On a Mac:

```bash
kvad pull stabilityai/stable-diffusion-xl-base-1.0
kvad images make "a lighthouse on a cliff at dusk, oil painting" \
    --model stabilityai/stable-diffusion-xl-base-1.0 --out lighthouse.png
```

SDXL is a 6.3 GB download. On an M5 Pro a 1024² picture at 30 steps takes about
a minute and a half.

## 6. Point a client at it

Anything that speaks OpenAI's API works against `http://127.0.0.1:5823/v1`:

```bash
curl http://127.0.0.1:5823/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in Norwegian."}]
  }'
```

## Where next

- [Models and backends](/docs/models/): CPU or GPU, f32 or q8 or q4, and what fits.
- [OpenAI-compatible API](/docs/api/): chat, images and video over HTTP.
- [Tool calls and agents](/docs/agents/): driving a coding agent with a local model.
