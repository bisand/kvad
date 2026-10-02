---
title: Benchmarks and evals
description: Measure variants against each other honestly, score models on prompt suites, and put a perplexity number on held-out text.
---

Three numbers in Kvad's own documentation were once wrong, each from timing a
single run. That is why the server has a benchmark in it rather than a shell
script, and why it is strict.

## Benchmarks

```bash
kvad bench run Qwen/Qwen2.5-1.5B-Instruct@cpu-q8 Qwen/Qwen2.5-1.5B-Instruct@gpu-q8
kvad bench                 # runs
kvad bench show 4
```

| Option | |
|---|---|
| `--rounds N` | How many rounds. |
| `--tokens N` | Tokens to generate each time. |
| `--prompt T` | The prompt. |
| `--seed N` | |

What a run does, and why:

- **A round visits every variant once**, and a run is several rounds. Five of
  A and then five of B blames the model for whatever changed about the machine
  in between.
- **Median and range**, from samples that are all kept. Two ranges that
  overlap are two numbers that have not been told apart.
- **The KV cache is dropped before every timed generation**, or the second
  round reports a time to first token that no first run would ever see.
- **It will not start** while a training run, an eval or another benchmark is
  going. The refusal names what is in the way.

One limit: interleaving is right when the variants are small enough to be in
memory together. Two variants that are each a third of the machine's memory
evict each other, and each round then measures the other's footprint. Run
those as two runs.

The web UI's Benchmarks page starts the same runs and draws every sample.

## Prompt suites

A suite is a file of prompts, each with what a right answer contains:

```json
{
  "name": "capitals",
  "cases": [
    { "prompt": "What is the capital of Norway?", "expect": "Oslo" },
    { "prompt": "What is the capital of France?", "expect": "Paris" }
  ]
}
```

```bash
kvad evals add capitals.json
kvad evals suites
kvad evals run capitals Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen3-14B@gpu-q8
kvad evals show 2
```

Each model is loaded in turn and every verdict is kept. A model with no
`@BACKEND` runs on `--backend`, or on the server's default.

## Perplexity

Perplexity scores how well a model predicts text it has not seen. Lower is
better. It is the number to use when comparing two training runs on the same
corpus, or the same model at two precisions.

```bash
kvad datasets add heldout.txt
kvad evals perplexity heldout my-model my-model-v2
```

Scoring runs the output head over a whole window as one matrix product, so it
is about four times as fast as decoding the same number of tokens.
