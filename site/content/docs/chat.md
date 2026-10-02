---
title: Chat and completion
description: One-shot answers and conversations from the command line, the sampling options, and what the numbers after a reply mean.
---

## One answer

```bash
kvad run --prompt "Explain backpropagation in one sentence."
```

```text
llama · 30 layers · 9 heads (3 KV heads, 3x grouped) · 576 embd · 8192 ctx
134.5M parameters · instruction-tuned

The sky appears blue because of the way our eyes detect light. [...]
[prefill 41 tokens in 1.10s · generated 56 in 1.62s = 34.6 tok/s]
```

An instruction-tuned model gets its own chat template around the prompt. A
base model continues the text. To continue the text with an instruct model
too, on a server, add `--raw`.

## A conversation

```bash
kvad chat
kvad chat --system "You are terse."
kvad chat --model Qwen/Qwen3-14B
```

On a server, `--save` keeps the conversation there, where the web UI shows it,
and `--conversation ID` carries on with a kept one:

```bash
kvad chat --save
kvad conversations                 # ls
kvad chat --conversation 12
```

## Sampling

| Option | Default | |
|---|---|---|
| `--max-tokens N` | 256 | The generation budget. |
| `--temperature F` | 0.7 | 0 is greedy. |
| `--top-k N` | 40 | Keep the N best candidates. |
| `--top-p F` | 0.95 | The nucleus threshold. |
| `--seed N` | 7 | The same seed draws the same reply. |
| `--greedy` | | Shorthand for `--temperature 0`. |

## The numbers after a reply

**Prefill** is reading the prompt: every token of it goes through the model
once, in batches. It is bound by arithmetic, which is what a GPU is for.

**Decode** is writing the answer, one token at a time. It is bound by how fast
the weights can be read from memory, which is why fewer bits per weight helps
and why the CPU at q8 can keep up with the GPU at bf16.

**Cached** tokens are the part of the conversation the model had already read.
A conversation keeps its KV cache across turns: each turn the new transcript is
compared with what is cached, and only the new message is prefilled.

## Reasoning models

Qwen3 and its relatives think before they answer, between `<think>` and
`</think>`. Kvad separates the two as they stream. The command line prints the
working to stderr and the answer to stdout, the web UI shows the working
collapsed above the answer, and the API sends it as `reasoning_content`,
leaving `content` the answer.

A small reasoning model can spend its whole budget thinking. If a reply is
empty, raise `--max-tokens`.

## Where it runs

With the service running, `kvad run` and `kvad chat` send their work to it, so
a second copy of the weights is not loaded beside the one the server holds.
The first line of output says which happened.

```bash
kvad chat --local                           # in this process, whatever is running
kvad chat --remote http://gpu-box.local:5823
```

See [The server](/docs/server/#the-command-line-is-a-client) for the order
these are decided in.
