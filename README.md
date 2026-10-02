<img src="docs/brand/kvad.svg" width="96" align="right" alt="">

# Kvad

*Kvad* — Old Norse for a composed, recited poem: what a skald performs from
memory, one line at a time.

An LLM engine in Rust, written from the arithmetic up, with two jobs.

**Explain how this works.** Every matrix multiply, every derivative, every
attention head and every rotation is code in this repo, with the reasoning for
each constant next to it. Read the crates in order and you will have built a
transformer rather than configured one.

**Be worth running.** The target is vLLM and SGLang. Not as a gesture — as the
bar this has to clear before it is a real alternative rather than a nice
explanation.

Those sound opposed and mostly are not. Almost everything that makes inference
fast is arithmetic you can read: block-wise quantisation, an integer dot
product, a tiled GEMM, a cache you do not recompute. The unreadable part of a
fast engine tends to be the last stretch — hand-tuned CUDA, a dozen code paths
per operator — and the stretch before it is where the lessons are. Where
legibility and speed genuinely conflict, this repo says which it chose and
measures what it cost.

Five crates, meant to be read in order:

| Crate | What it is | Dependencies |
|---|---|---|
| [`nervus`](crates/nervus) | A neural network and backpropagation, from scratch. Trains on MNIST. | **none** |
| [`kvad`](crates/llm) | Transformer inference from scratch. Four architectures, real HuggingFace weights. | hub client, tokenizer, safetensors |
| [`kvad-gpu`](crates/gpu) | The same three forward passes on the GPU, in candle. | candle (Metal/CUDA) |
| [`kvad-tui`](crates/tui) | Terminal app: browse, download, activate, chat. | ratatui |
| [`kvad-serve`](crates/serve) | HTTP server and web UI: manage, train, score, benchmark, watch. | axum, rusqlite, Svelte |

The first two use no ML framework at all. The third is the same model handed to
one, so the two can be compared — and so the hand-written version has something
honest to be measured against. The last two are applications: no ML in either,
and the place to see what an engine has to expose before anything can be built
on it.

### Where this actually stands

Kvad runs one sequence at a time. On an M5 Pro, Qwen2.5-0.5B decodes at
**191 tok/s** on Metal at q8 and **112 tok/s** on the hand-written CPU engine —
which is also, for now, what this machine's GPU does in bf16. It has weight and
activation quantisation, an i8mm integer kernel, a hand-written 4-bit decode
kernel, a tiled f32 GEMM, batched prefill, prefix caching across chat turns,
and a memory-mapped cache of pre-quantised weights.

It runs four architectures, and the fourth is the one that says most about
where inference has gone. **DeepSeek-V2-Lite** — 15.7 billion parameters, 2.4
billion of them used on any given token — loads in 92 seconds, occupies 17.7 GB
at q8, and decodes at **23.4 tok/s** on the CPU. It needed multi-head latent
attention and a 64-expert mixture, neither of which the GPT-2-to-Llama skeleton
had any room for, which is why architectures are
[plugin modules](docs/engine.md#six-years-of-architecture-progress-as-a-table) now rather
than arms in a `match`.

There is now an HTTP API and a web UI around it: OpenAI-compatible completions
— including tool calling, so an agent like OpenCode drives it as a provider —
plus model management, training, evals, benchmarks and monitoring. None of that
makes it a fast server, and it is built not to pretend otherwise — requests
queue behind one another because the engine runs one generation at a time, and
the queue depth is a number on the dashboard rather than a mutex nobody can
see. What it did change is that the decode numbers in this README now have a
button that reproduces them: interleaved rounds, medians and ranges, and a
refusal to measure a machine that is busy doing something else.

It still has none of what makes a server *fast*: no continuous batching, no
paged KV cache, no kernels of its own on CUDA, no speculative decoding. The KV
cache is a `Vec<f32>` per layer that grows by appending, and 805 MB of it at
Qwen's full context. Nothing here has been benchmarked against vLLM or SGLang,
because a single-sequence engine and a serving engine do not yet have a number
in common.

So the second job is a direction, not a claim. [The roadmap](docs/roadmap.md#where-to-go-next)
says what would have to become true first, starting with the one thing every
measurement in this repo keeps pointing at.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/bisand/kvad/master/install.sh | sh
```

Prebuilt binaries from the latest [release](https://github.com/bisand/kvad/releases),
checked against that release's `SHA256SUMS` and put in `~/.local/bin`. Nothing
needs root. Run the same command again to upgrade.

macOS (Apple Silicon) gets all four binaries; Linux gets `kvad` and a CPU-only
`kvad-serve`. What the installer asks, its flags, the background service and
where the data lives are in [docs/install.md](docs/install.md).

## Quick start

From a checkout, with Rust and (for the web UI) Node installed:

```bash
./scripts/get-mnist.sh
cargo test                                      # includes a gradient check
cargo run --release -p nervus --bin train_mnist
cargo run --release -p nervus --bin train_text -- --data docs/nervus.md
cargo run --release -p kvad -- train --data docs/nervus.md --name nervus
cargo run --release -p kvad -- run --model nervus --prompt "## "
cargo run --release -p kvad -- run --prompt "Why is the sky blue?"
cargo run --release -p kvad-gpu -- run --prompt "Why is the sky blue?"
cargo run --release -p kvad-tui

cd web && npm ci && npm run build && cd ..   # once, for the web UI
cargo run --release -p kvad-serve            # then http://127.0.0.1:5823
```

Verified on an M5 Pro:

```
epoch  1  loss 0.2085  test accuracy 96.32%  (2.1s)
epoch 10  loss 0.0072  test accuracy 97.98%  (21.0s)
```

```
llama · 30 layers · 9 heads (3 KV heads, 3x grouped) · 576 embd · 8192 ctx
134.5M parameters · instruction-tuned

The sky appears blue because of the way our eyes detect light. [...]
[prefill 41 tokens in 1.10s · generated 56 in 1.62s = 34.6 tok/s]
```

## Documentation

The long form lives in [`docs/`](docs), one file per crate, meant to be read
in order:

| Read | What is in it |
|---|---|
| [Install](docs/install.md) | The installer's questions and flags, upgrades, `kvad service`, the data directory, platforms. |
| [1. `nervus`](docs/nervus.md) | Backpropagation from scratch, then a GPT trained on a text file: checkpoints, where the time went, seeds, `kvad crawl`, answering from a corpus, `kvad train`, and models that draw images and clips. |
| [2. `kvad`](docs/engine.md) | The inference engine: four architectures as plugins, quantisation, the integer and f32 kernels, batched prefill, the KV cache, and the measurements that turned out wrong. |
| [3. `kvad-gpu`](docs/gpu.md) | The same model in candle on Metal/CUDA: numbers, quantised weights on the GPU, one trait over two backends, and image generation. |
| [4. `kvad-tui`](docs/tui.md) | The terminal app. |
| [5. `kvad-serve`](docs/serve.md) | The HTTP server and web UI: reasoning output, tool calls, model residency, the default backend, benchmarks, the playground, the CLI as a client, testing. |
| [Where to go next](docs/roadmap.md) | The roadmap, one track per job, and what to read alongside. |

Design notes for individual features — GGUF, images, video, LoRA, the cluster,
the UI — sit next to them in [`docs/`](docs) as `*-plan.md`,
[tune.md](docs/tune.md) covers training a LoRA for an image model, and
[releasing.md](docs/releasing.md) covers cutting a release.

## Licence

MIT — see [LICENSE](LICENSE).
