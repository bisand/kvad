# ai-llm

Learning how neural networks and language models work by building them in Rust,
from the arithmetic up.

Three crates, meant to be read in order:

| Crate | What it is | Dependencies |
|---|---|---|
| [`nanograd`](crates/nanograd) | A neural network and backpropagation, from scratch. Trains on MNIST. | **none** |
| [`llm`](crates/llm) | Transformer inference from scratch. Two architectures, real HuggingFace weights. | hub client, tokenizer, safetensors |
| [`llm-tui`](crates/tui) | Terminal app: browse, download, activate, chat. | ratatui |

No ML framework anywhere. Every matrix multiply, every derivative, every
attention head and every rotation is code in this repo.

## Quick start

```bash
./scripts/get-mnist.sh
cargo test                                      # includes a gradient check
cargo run --release -p nanograd --bin train_mnist
cargo run --release -p llm -- run --prompt "Why is the sky blue?"
cargo run --release -p llm-tui
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

---

## Crate 1: `nanograd` — where the learning actually happens

Zero dependencies. Read in this order:

1. **[`matrix.rs`](crates/nanograd/src/matrix.rs)** — three matrix products.
   `A@B` forward, `Aᵀ@B` for weight gradients, `A@Bᵀ` for input gradients.
2. **[`nn.rs`](crates/nanograd/src/nn.rs)** — the important one. Every layer
   implements `forward(x) -> y` and `backward(dL/dy) -> dL/dx`. Chaining the
   second one backwards *is* backpropagation.
3. **[`bin/train_mnist.rs`](crates/nanograd/src/bin/train_mnist.rs)** — the
   training loop: predict, score, blame, adjust.

### The test worth running first

```bash
cargo test -p nanograd analytic_gradient_matches_numerical
```

It nudges one weight by ±0.001, measures how the loss actually moves, and checks
that against what `backward()` claimed. A subtly wrong gradient still trains,
just badly — this is the only thing that catches it.

### Things to try

- `--hidden 16` — how small before accuracy collapses?
- `--lr 0.5` — watch it diverge. `--lr 0.0001` — watch it crawl.
- `--momentum 0` — see how much of the speed was momentum.
- Note that training loss keeps falling after epoch 5 while test accuracy
  flatlines. That gap is overfitting, live.

**Momentum is a multiplier on your learning rate.** At steady state the velocity
converges to `lr·g/(1−μ)`, so `--lr 0.1 --momentum 0.9` really steps at ~1.0.
That is why the default is 0.02: at 0.1 this model plateaus at 95%, at 0.02 it
reaches 98%.

---

## Crate 2: `llm` — the same ideas, at scale

The examples below assume the binaries are on your `PATH`:

```bash
cargo install --path crates/llm --path crates/tui
```

Otherwise prefix each one with `cargo run --release -p llm --` (or `-p llm-tui`).

```bash
llm search smollm                 # find models; says which we can run
llm pull HuggingFaceTB/SmolLM2-360M-Instruct
llm ls                            # what is downloaded, and how big
llm use  Qwen/Qwen2.5-0.5B-Instruct
llm run  --prompt "Explain backpropagation in one sentence."
llm chat --system "You are terse."
llm info --model openai-community/gpt2-medium   # config only, no weights
```

Read in this order:

1. **[`tensor.rs`](crates/llm/src/tensor.rs)** — matmul, LayerNorm, GELU,
   softmax, then RMSNorm, SwiGLU and RoPE. Nine functions, two architectures.
2. **[`weights.rs`](crates/llm/src/weights.rs)** — safetensors is a length, a
   JSON header, and raw floats. Plus shard indexes and bf16 widening.
3. **[`model/mod.rs`](crates/llm/src/model/mod.rs)** — the skeleton both
   architectures share, including attention itself.
4. **[`model/gpt2.rs`](crates/llm/src/model/gpt2.rs)** — read first, it is
   simpler. Then **[`model/llama.rs`](crates/llm/src/model/llama.rs)**, written
   to be read as a diff against it.
5. **[`quant.rs`](crates/llm/src/quant.rs)** — block-wise int8/int4 weights
   and the kernels that consume them, then
   **[`simd.rs`](crates/llm/src/simd.rs)** for the one instruction the compiler
   will not reach on its own.
6. **[`sampler.rs`](crates/llm/src/sampler.rs)**, then
   **[`chat.rs`](crates/llm/src/chat.rs)**.

### Five years of architecture progress, as a table

| GPT-2 (2019) | Llama family (2023+) | Why |
|---|---|---|
| learned position rows (`wpe`) | RoPE: rotate Q and K by angle ∝ position | no hard context ceiling; position becomes *relative* for free |
| LayerNorm (centre, scale, bias) | RMSNorm (scale only) | the centring was never load-bearing |
| GELU MLP, 2 matrices | SwiGLU, 3 matrices | a learned gate per channel |
| multi-head attention | grouped-query attention | KV cache shrinks by the group factor |
| one fused QKV matrix | three projections | Q and KV now have different widths |

What did *not* change: the residual stream, the alternation of attention and
MLP, tied embeddings, the causal mask, scaled dot-product attention. `attend()`
in `model/mod.rs` is shared verbatim between the two — grouped-query attention
is just a smaller `kv_dim`, and GPT-2 is the `n_kv_head == n_head` case.

And the detail worth sitting with: **SmolLM2-135M is smaller than GPT-2-medium
and holds a conversation, while GPT-2 cannot.** The architecture changes above
are real but marginal. Nearly all of that gap is training data and
post-training. Running both in the same binary makes the point better than any
benchmark.

### Base models versus instruction-tuned

GPT-2 is a **base** model: pure next-token prediction, no instruction tuning.
Ask it a question and it writes more questions, because that is what its
training data looked like. `llm ls` and `llm search` label which is which, and
`llm chat` warns you.

An instruction-tuned model only behaves like an assistant when wrapped in the
exact marker tokens it was trained on. Those live as a **Jinja template** in
`tokenizer_config.json`, one per model, and they genuinely differ — so
[`chat.rs`](crates/llm/src/chat.rs) renders the model's own template rather than
hardcoding one. "The model is dumb" is very often "the template is wrong".

### Quantisation

```bash
llm run --quant q8 --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
```

`--quant q8|q4` quantises the weight matrices as they load. Each row is chopped
into blocks of 32 with its own scale, so a single outlier only ruins its own 32
neighbours instead of flattening the whole tensor — which is what per-tensor
scaling does, and why it fails.

Measured on Qwen2.5-0.5B (494M parameters, M5 Pro, 18 threads):

| | weights | tok/s | `first 8 primes` |
|---|---|---|---|
| f32 | 1976 MB | ~34 | `2, 3, 5, 7, 11, 13, 17, 19` |
| q8 | 556 MB (3.6x) | ~38 | `2, 3, 5, 7, 11, 13, 17, 19` |
| q4 | 309 MB (6.4x) | ~37 | `11, 13, 17, 19, 23, 29, 31, 37` |

**q8 is free; q4 is not.** q8 reproduced f32 exactly on every test — including
with the activations quantised as well. q4 broke all three: it misses the
sequence start above, turns `17 + 25 = 42` into `40` on SmolLM2, and sends
GPT-2 into `"The jury's jury's jury's"`. Still fluent, still confident, quietly
wrong. Half-billion-parameter models have less redundancy to spare than the
7B+ models where q4 is usually judged.

```bash
cargo run --release -p llm --example bench_matvec
```

The kernel benchmark is where the interesting part is. On the output head
(151936 x 896, the largest single matmul per token):

| | ms/call | vs f32 |
|---|---|---|
| f32 | 4.0 | 1.00x |
| q8 + quantised activations | 0.83 | **4.9x** |
| q4 + quantised activations | 0.90–1.6 | 2.4–4.5x |

The 4.9x comes from quantising the *activations* too, which turns the inner
loop into an integer dot product. On this machine that compiles to exactly what
you would hope for — six instructions per 32 weights:

```asm
ldp     q2, q3, [x10], #0x20   ; 32 weight bytes
movi.2d v4, #0
sdot.4s v4, v3, v1             ; 16 int8 multiply-accumulates
sdot.4s v4, v2, v0             ; 16 more
addv.4s s0, v4                 ; horizontal sum
```

Getting there took four fixes, each a general lesson:

- **Keep the accumulator in registers.** The first kernel read and wrote all 32
  outputs on every input row: 256 bytes of output traffic per 32 bytes of
  weights. Quantising made it *slower than f32* (0.90x).
- **Use more than one accumulator.** Float addition is not associative, so the
  compiler cannot reorder a `sum +=` chain; the loop runs at FMA latency rather
  than throughput. Integer addition *is* associative — one reason the integer
  path is easier to vectorise.
- **Check what the inner loop doesn't depend on.** The 4-bit offset correction
  needs a per-block sum of the activations, which depends only on the input.
  Computing it inside the row loop repeated it 151,936 times per call.
- **Never mix floats into an integer reduction.** This was the big one. With
  the per-block scaling inline, the loop compiled to scalar loads — no `sdot`
  at all, despite the same expression vectorising perfectly as a standalone
  function. Splitting it into an integer pass and a scaling pass was worth
  roughly 2x on its own. Worth knowing that a correct, innocuous-looking line
  can silently cost you the vector units.

### Two kernels, chosen by size

There is a crossover. The integer path wins big on a matrix streamed from main
memory and *loses* on one that fits in cache — a cache-resident matmul is not
bandwidth-bound, so shrinking the weights buys nothing and the extra work is
pure overhead. Measured here: 4.9x on the 153 MB output head, but below 1x on a
5 MB MLP matrix. So `matvec_bt` dispatches on weight size, and
`INTEGER_PATH_MIN_BYTES` is the (empirical, cache-dependent) threshold.

### What it is worth end to end

| | f32 | q8 |
|---|---|---|
| Qwen2.5-0.5B | ~34 tok/s | ~38 tok/s |
| GPT-2-medium | ~35 tok/s | ~49 tok/s |

**A 4.9x kernel bought well under 1.4x overall.** That gap is the whole lesson in
optimisation: the output head is one matmul out of 169 per token, and
everything else — the cache-resident per-layer matrices, attention, the norms,
the softmax — did not get faster. Amdahl's law, measured rather than quoted.

These were taken with a loaded machine (the f32 baseline itself varied between
19.7 and 25.2 tok/s), so treat them as approximate. The kernel numbers are far
more repeatable than the end-to-end ones.

And note what q4 does *not* buy: it is no faster than q8 — unpacking nibbles
costs more than the halved bytes save — while being much less accurate. Its
only advantage is memory.

### Batched prefill, and i8mm

A prompt used to go through the model one token at a time, re-reading every
weight matrix once per token. Feeding tokens through together reads each weight
once and reuses it across the batch — a memory-bound operation becomes a
compute-bound one. `forward_batch` does that, in chunks of 64.

Causal masking comes for free, which is the neat part: push all the batch's
keys and values into the cache first, then let query `i` attend over exactly
`pos0 + i + 1` positions. No mask anywhere. (Every tutorial's triangular matrix
is only needed when you *don't* have a KV cache to be careful with.)

Once prefill is a real matrix-matrix product, ARM's `i8mm` extension applies.
`SMMLA` multiplies a 2x8 block of `i8` by an 8x2 block and accumulates a 2x2
`i32` result — 32 multiply-accumulates in one instruction, twice `SDOT`'s 16.

**But look at the shape it wants: two independent activation rows.** Generating
one token at a time gives you exactly one, so half the result lanes would be
duplicates and the useful throughput collapses back to `SDOT`'s. `i8mm` cannot
help decoding. Only prefill.

Prefill, 654 tokens, Qwen2.5-0.5B, q8:

| | time | |
|---|---|---|
| one token at a time (`LLM_PREFILL_CHUNK=1`) | 18.7 s | |
| batched f32 GEMM | 2.81 s | **6.6x** from batching |
| batched q8, `SDOT` (`LLM_NO_I8MM=1`) | 2.72 s | |
| batched q8, `SMMLA` | 2.18 s | **1.25x** more from i8mm |

Prefill drops from ~28 ms/token to ~3.3 ms/token. Decoding is untouched — it is
still a matrix-vector product, and still the same speed.

Note where f32 lands: **2.81 s against q8's 2.72 s.** Once the weights are
reused across a batch, prefill is compute-bound, and quantisation is an
answer to a memory problem. On the smaller `q_proj` matrix the f32 GEMM is
actually *faster* than the q8 `SDOT` path, because unpacking and rescaling cost
more than the bytes they save. Quantisation is a decode optimisation.

`i8mm` is worth more than 1.27x on a single core: 2.04x at
`RAYON_NUM_THREADS=1`. With all 18 threads running, memory bandwidth becomes
the limit again and a faster instruction has less to offer. Both knobs above
are environment variables precisely so this is measurable rather than asserted.

Rust exposes `SMMLA` only through an unstable intrinsic, so
[`simd.rs`](crates/llm/src/simd.rs) emits it with inline assembly — stable, and
four lines. Availability is detected at runtime, so one binary still runs on
CPUs without it.

### The bug worth stealing

The first version of the `SMMLA` kernel was **slower than the `SDOT` path it
was meant to beat** — 65% slower on one core. The cause was one missing
attribute:

```rust
#[target_feature(enable = "i8mm")]   // <- this line
unsafe fn smmla_row_pair(...)
```

A `#[target_feature]` function can only be inlined into a caller declaring at
least the same features. Without it, every single `smmla` became a real
function call. It compiles, it is correct, it is tested, and it quietly throws
away everything the instruction was for.

Two measurement traps on the way there, both of which produced confident wrong
answers:

- The first comparison said `SMMLA` was **2.01x faster**. It was being compared
  against a fallback that had floats mixed into its integer loop, so the
  baseline was scalar code rather than `SDOT`. Fixing the baseline turned the
  result into a 7% *loss*.
- An isolated benchmark with a single accumulator said `SMMLA` was **2.4x
  slower** — that loop was latency-bound on its own accumulator chain. Four
  independent chains turned it into a 1.39x win.

Three different numbers for the same instruction, all measured, none of them
right until the last one. Worth remembering before quoting a speedup.

### The f32 GEMM, and register tiling

Batching only helped the quantised paths at first: with f32 weights `matmul_bt`
still called the matrix-vector kernel once per token, which reads the whole
weight matrix `m` times. [`gemm_bt`](crates/llm/src/tensor.rs) fixes that, and
the fix is the classic one.

A first attempt kept one weight row live and ran it against four activation
rows. That loads 5 vectors per 4 multiply-accumulates — the loop spends its
time fetching, not computing. The **4x4 micro-kernel** loads four weight
vectors and four activation vectors, then does sixteen multiply-accumulates
with them:

| tile | loads per 4-wide step | FMAs | GMAC/s |
|---|---|---|---|
| 1 row x 4 tokens | 5 | 4 | 78.9 |
| 4 rows x 4 tokens | 8 | 16 | **204.5** |

Same arithmetic, same memory traffic from RAM, 2.6x the throughput. The only
thing that changed is how many times each fetched value gets used before it is
thrown away. That ratio — arithmetic per byte fetched — is what decides whether
a GEMM runs at memory speed or at FPU speed, and 4x4 is sized so the 16
accumulator vectors plus 8 operand vectors fit in aarch64's 32 vector
registers.

### The optimisation that wasn't

Rust compiles `acc += a * b` to a separate `fmul` and `fadd`. It will not fuse
them on its own, because fusing changes the result: `a * b + c` rounds twice,
`fma(a, b, c)` rounds once. `f32::mul_add` asks for the fused form explicitly.

One instruction instead of two, *and* more accurate. It should be free money.
Measured across four tile shapes, it was consistently **worse** — 204 GMAC/s
became 93–116:

| | 4x4 | 4x2 | 2x4 | 2x2 |
|---|---|---|---|---|
| `mul_add` | 93.3 | 116.3 | 103.1 | 80.4 |
| `+= a * b` | **204.5** | | | |

An `fmla` needs all three operands live simultaneously. With 16 accumulators
and 8 operands already in flight, that tips the tile past the register file and
it spills to the stack. The cheaper instruction lost to the extra memory
traffic it caused.

In [`matvec_bt`](crates/llm/src/tensor.rs), where only four accumulators are
live, `mul_add` made no measurable difference at all — that kernel is
bandwidth-bound at ~220 GB/s, so its instruction count is irrelevant. Which is
the more useful half of the lesson: an optimisation is a claim about the
bottleneck, and if you have not measured the bottleneck you are guessing.

### Verifying you got it right

```bash
llm run --model openai-community/gpt2 --greedy --prompt "1, 2, 3, 4, 5, 6,"
llm run --model Qwen/Qwen2.5-0.5B-Instruct --greedy \
    --prompt "List the first 8 prime numbers, comma separated."
```

The first continues `7, 8, 9, ... 16`. The second answers
`2, 3, 5, 7, 11, 13, 17, 19`. A wrong transpose, a wrong RoPE convention or a
mishandled bias degrades output to *plausible-looking noise* rather than failing
loudly, so arithmetic is the sharp test. The GPT-2 one is also the regression
test for the Llama refactor, and both are the acceptance test for `--quant q8`.

---

## Crate 3: `llm-tui` — the app

```bash
cargo run --release -p llm-tui
```

`/` search · `↑↓` select · `enter` download and load · `p` cycle precision ·
`d` delete · `tab` switch to chat · `esc` interrupt generation.

`p` is the quickest way to feel the quantisation trade-off: load a model at
f32, ask it something arithmetic, then reload at q4 and ask again.

Three concerns on three threads: the UI loop only draws and reads keys, the
engine thread downloads and generates, and `rayon` fans each matmul across cores
underneath. They talk over channels — except cancellation, which a channel
cannot express because the worker is busy inside `generate`, so that is a shared
`AtomicBool` the per-token callback checks.

The chat keeps its KV cache **across turns**. Each turn it re-encodes the
conversation, finds the common prefix with what is already cached
(`Llm::common_prefix`), truncates to there, and prefills only the new message.
The status bar reports how many tokens that saved — `[33.3 tok/s · 96 cached]`.
Without it, turn *N* re-reads the entire transcript.

This crate is the least educational of the three. It is `ratatui` plumbing and
state management — good Rust, no ML. Build it last.

---

## Where to go next

**4. Serialise quantised weights.** Quantisation currently happens on every
load, which means still reading the full f32 checkpoint off disk. Writing the
blocks out once would make startup and disk footprint match the memory win.

**5. GPU, via [`candle`](https://github.com/huggingface/candle).** HuggingFace's
Rust framework, Metal backend. You will recognise every operation because you
wrote them by hand first. 48 GB of unified memory holds a quantised 30B model.

**6. Train your own.** A character-level transformer, 10–30M parameters, on a
corpus you pick. Needs backprop through attention, layernorm and softmax, plus
Adam. The gradient check from crate 1 is how you will debug it — extend
`nanograd` (hard, most educational) or use
[`burn`](https://github.com/tracel-ai/burn).

**7. Fine-tune with LoRA.** Freeze the model, train two small low-rank matrices
per weight matrix. This is what "custom model" means in practice, and unlike
full fine-tuning it fits on a laptop.

### Worth reading alongside

- Karpathy, *Let's build GPT: from scratch, in code, spelled out*.
- *The Illustrated Transformer*, Jay Alammar — the diagrams.
- Vaswani et al., *Attention Is All You Need* (2017).
- Su et al., *RoFormer* (2021) — where RoPE comes from.
- Ainslie et al., *GQA* (2023) — grouped-query attention.
