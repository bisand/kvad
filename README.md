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

Four crates, meant to be read in order:

| Crate | What it is | Dependencies |
|---|---|---|
| [`nanograd`](crates/nanograd) | A neural network and backpropagation, from scratch. Trains on MNIST. | **none** |
| [`kvad`](crates/llm) | Transformer inference from scratch. Two architectures, real HuggingFace weights. | hub client, tokenizer, safetensors |
| [`kvad-gpu`](crates/gpu) | The same Llama forward pass on the GPU, in candle. | candle (Metal/CUDA) |
| [`kvad-tui`](crates/tui) | Terminal app: browse, download, activate, chat. | ratatui |

The first two use no ML framework at all. The third is the same model handed to
one, so the two can be compared — and so the hand-written version has something
honest to be measured against.

### Where this actually stands

Kvad runs one sequence at a time. On an M5 Pro, Qwen2.5-0.5B decodes at
**193 tok/s** on Metal at q8 and 35 tok/s on the CPU engine, and prefills 640
tokens in 0.18 s. It has weight and activation quantisation, an i8mm integer
kernel, a tiled f32 GEMM, batched prefill, prefix caching across chat turns, and
a memory-mapped cache of pre-quantised weights.

It has none of what makes a *server* fast: no continuous batching, no paged KV
cache, no HTTP API, no kernels of its own on CUDA, no speculative decoding. The
KV cache is a `Vec<f32>` per layer that grows by appending, and 805 MB of it at
Qwen's full context. Nothing here has been benchmarked against vLLM or SGLang,
because a single-sequence engine and a serving engine do not yet have a number
in common.

So the second job is a direction, not a claim. [The roadmap](#where-to-go-next)
says what would have to become true first, starting with the one thing every
measurement in this repo keeps pointing at.

## Quick start

```bash
./scripts/get-mnist.sh
cargo test                                      # includes a gradient check
cargo run --release -p nanograd --bin train_mnist
cargo run --release -p kvad -- run --prompt "Why is the sky blue?"
cargo run --release -p kvad-gpu -- run --prompt "Why is the sky blue?"
cargo run --release -p kvad-tui
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
4. **[`attention.rs`](crates/nanograd/src/attention.rs)** — causal
   self-attention, forward and backward. Read it after the rest: it is three
   `Linear` layers and two of the products from step 1, and the only new
   derivative in it is the softmax's.

### The test worth running first

```bash
cargo test -p nanograd analytic_gradient_matches_numerical
```

It nudges one weight by ±0.001, measures how the loss actually moves, and checks
that against what `backward()` claimed. A subtly wrong gradient still trains,
just badly — this is the only thing that catches it.

Attention has its own, which checks every weight of all four projections and
the gradient handed back to the layer below. With correct derivatives the
analytic and numerical gradients differ by about 0.0007; drop the row average
from the softmax derivative, forget the `1/sqrt(d)` on the way back, or miss the
transpose in `dK`, and that becomes 0.2 to 0.8. Two more tests pin the causal
mask from both sides: the future cannot change the past's output, and the past's
loss cannot blame the future's input.

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

## Crate 2: `kvad` — the same ideas, at scale

The examples below assume the binaries are on your `PATH`:

```bash
cargo install --path crates/llm --path crates/tui
```

Otherwise prefix each one with `cargo run --release -p kvad --` (or `-p kvad-tui`).

```bash
kvad search smollm                 # find models; says which we can run
kvad pull HuggingFaceTB/SmolLM2-360M-Instruct
kvad ls                            # what is downloaded, and how big
kvad use  Qwen/Qwen2.5-0.5B-Instruct
kvad run  --prompt "Explain backpropagation in one sentence."
kvad chat --system "You are terse."
kvad info --model openai-community/gpt2-medium   # config only, no weights
kvad cache                         # pre-quantised weight files
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
   will not reach on its own, and
   **[`qcache.rs`](crates/llm/src/qcache.rs)** for doing that work once
   instead of once per load.
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
training data looked like. `kvad ls` and `kvad search` label which is which, and
`kvad chat` warns you.

An instruction-tuned model only behaves like an assistant when wrapped in the
exact marker tokens it was trained on. Those live as a **Jinja template** in
`tokenizer_config.json`, one per model, and they genuinely differ — so
[`chat.rs`](crates/llm/src/chat.rs) renders the model's own template rather than
hardcoding one. "The model is dumb" is very often "the template is wrong".

### Quantisation

```bash
kvad run --quant q8 --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
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
| q4 | 309 MB (6.4x) | ~37 | `2, 3, 5, 7, 11, 13, 17` |

**q8 is free; q4 mostly is not.** q8 reproduces f32 exactly on every test,
including with the activations quantised as well. q4 gets GPT-2's counting and
SmolLM2's arithmetic right but still drops a prime above — fluent, confident,
quietly wrong. Half-billion-parameter models have less redundancy to spare than
the 7B+ models where q4 is usually judged.

> **A correction.** q4 used to fail *all three* tests, and this README used to
> say so. That was a bug in the scale, not a property of 4-bit weights — see
> [the scale that wasted a code](#the-scale-that-wasted-a-code).

### The scale that wasted a code

Four bits span sixteen codes, `-8..7`. The obvious scale is
`block_max_magnitude / 7`, which is symmetric and reads naturally. It also never
produces `-8`: one code in sixteen is unused and every step is 1/7 of the range
instead of 1/8.

Dividing the *signed* extreme by `-8` uses all sixteen. The sign is what makes
it work — whichever end the extreme sits at, it lands on `-8` and the rest of
the block spreads over the remaining codes. The trade is that the *opposite*
extreme now clips to 7/8 of its value, so it is not a free win; measured over
random blocks it is a **9.54% → 8.49%** mean relative error, about what the
8/7 step ratio predicts.

On real models that margin decided three test cases. Before the fix, q4 turned
`17 + 25 = 42` into `40`, sent GPT-2 into `"The jury's jury's jury's"`, and
started the primes at 11. After it, the first two are correct.

I found this by comparing against GGML's `Q4_0` through candle, which produced
better output than my own q4 from the same checkpoint. That is the argument for
porting a model to a framework even when you have written it yourself: the
framework is a second opinion, and it disagreed for a reason.

```bash
cargo run --release -p kvad --example bench_matvec
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
| one token at a time (`KVAD_PREFILL_CHUNK=1`) | 18.7 s | |
| batched f32 GEMM | 2.81 s | **6.6x** from batching |
| batched q8, `SDOT` (`KVAD_NO_I8MM=1`) | 2.72 s | |
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
kvad run --model openai-community/gpt2 --greedy --prompt "1, 2, 3, 4, 5, 6,"
kvad run --model Qwen/Qwen2.5-0.5B-Instruct --greedy \
    --prompt "List the first 8 prime numbers, comma separated."
```

The first continues `7, 8, 9, ... 16`. The second answers
`2, 3, 5, 7, 11, 13, 17, 19`. A wrong transpose, a wrong RoPE convention or a
mishandled bias degrades output to *plausible-looking noise* rather than failing
loudly, so arithmetic is the sharp test. The GPT-2 one is also the regression
test for the Llama refactor, and both are the acceptance test for `--quant q8`.

### Doing the quantising once

Every `--quant q8` load re-derived the same 494 million codes from the same
bf16 checkpoint and threw them away on exit. [`qcache.rs`](crates/llm/src/qcache.rs)
writes them out instead, and maps the file back on later loads:

```bash
kvad run --quant q8 --prompt hi      # first: "quantising to q8 (first load)"
kvad run --quant q8 --prompt hi      # after: "mapping q8 weights (556 MB)"
kvad cache                           # what has been written, and where
kvad cache clear                     # it is all derived data; delete freely
```

| model | quantise | map | |
|---|---|---|---|
| SmolLM2-135M q8 | 0.30 s | 8 ms | 37x |
| Qwen2.5-0.5B q8 | 1.17 s | 20 ms | 58x |
| Qwen2.5-0.5B q4 | 1.11 s | 24 ms | 46x |
| GPT-2-medium q8 | 1.85 s | 2 ms | 925x |

The ratios are silly because the denominator is *nothing happening*. `mmap`
does not read the file; it wires the pages into the address space and returns.
The bytes arrive later, on demand, faulted in from the page cache they are
already sitting in. Startup stops scaling with model size, because startup
stops doing work — a 7B model maps in the same 2 ms as a 355M one.

That property is why the weights are stored in the layout the kernels already
want. A format that needed a fix-up pass — endian swapping, unpacking, even
just a `Vec` copy — would give most of it back.

**What it cost in the code.** `Weight` could no longer assume it owns its
arrays, so `Vec<i8>` became `Store<i8>`: either a `Vec` or a window onto a
mapping, `Deref`ing to a slice either way. Every kernel in `quant.rs` is
untouched — they all took `&[i8]` and still do. The loaders lost their
`Checkpoint` and gained a `Source` trait with two implementations, which is
also where GPT-2's transpose went: the cache stores post-transpose matrices,
so `matrix_t` is a property of the *checkpoint*, and the mapped source ignores
the distinction.

**The interesting hazard is staleness, not speed.** A cache that can serve
bytes written under different rules is worse than no cache at all — and this
is not hypothetical here. The q4 scale fix two sections up changed every
4-bit code in every file; had this existed then, the old weights would have
gone on quietly answering `17 + 25 = 40` long after the bug was fixed. So the
header records a format version, the precision, the model's shape, and the
source checkpoint's file sizes, and anything that does not match is rebuilt
with a line saying why.

**Where the time goes now.** With the quantising gone, the load breakdown is
worth reading:

```
  [ 0.752s] reading weights          <- 0.75 s of hf-hub resolving five files
  [ 0.752s] mapping q8 weights (556 MB)
  [ 0.772s] reading tokenizer        <- 20 ms, nearly all of it RoPE tables
```

Those 20 ms are not the weights. Qwen precomputes a sin/cos table for 32768
positions at load (`Rope::new`, 12.6 ms measured on its own); the rest is the
tokenizer. The dominant cost is now the Hub client resolving file paths —
0.75 s, the same whether the model is 135M or 7B, and the same offline.
Which is the usual shape of these things: remove the obvious cost and the
bottleneck moves somewhere you were not looking.

The GPU backend does not use this. candle's `quantize_onto` takes about 0.2 s
for the same model — fast enough not to be worth a second format, and its
natural on-disk form would be GGUF rather than ours.

---

## Crate 3: `kvad-gpu` — the same model, handed to a framework

```bash
kvad-gpu run  --prompt "Why is the sky blue?"
kvad-gpu run  --device metal --dtype bf16 --model Qwen/Qwen2.5-0.5B-Instruct
kvad-gpu chat
```

[`model.rs`](crates/gpu/src/model.rs) is the Llama forward pass again, in
[candle](https://github.com/huggingface/candle). Read it next to
[`llama.rs`](crates/llm/src/model/llama.rs): the structure is line for line the
same, and every operation is one you already wrote.

- `matvec_bt` becomes `Tensor::matmul`.
- The loop over attention heads becomes one batched matmul over a
  `[1, n_head, seq, head_dim]` tensor.
- `candle_nn::rotary_emb::rope` is the same `rotate_half` convention as
  `Rope::apply`, and `ops::rms_norm` the same scale-only normalisation.

Two differences are worth noticing because they run *against* the framework:

- **Batching is free.** The CPU engine needed a separate `forward_batch`,
  because a matrix-vector product and a matrix-matrix product are genuinely
  different kernels. On the GPU there is one `forward`, and a single token is
  just the narrow case.
- **The causal mask comes back.** With a KV cache the CPU engine got masking
  for free — the cache only ever held earlier positions. Here the whole batch
  is one matmul, so the future has to be masked out explicitly, with the
  triangular `-inf` matrix every tutorial shows.

### Numbers

| | CPU (q8) | Metal (bf16) | |
|---|---|---|---|
| Qwen2.5-0.5B, decode | 35 tok/s | 115 tok/s | 3.3x |
| SmolLM2-135M, decode | ~35 tok/s | ~168 tok/s | 4.7x |
| Qwen2.5-0.5B, prefill (640 tokens) | 2.15 s | 0.18 s | **12x** |

**The GPU wins far more on prefill than on decode**, and that is the same
distinction as everywhere else in this repo. Prefill is compute-bound, which is
what a GPU is for. Decode is memory-bound, and there the CPU is carrying int8
weights (556 MB) against the GPU's bf16 (1260 MB) — quantisation claws back
most of a hardware generation.

The other oddity in that table: SmolLM2-135M decodes no faster than Qwen-494M on
the CPU, despite being a quarter the size. It has *more* layers (30 vs 24), and
our CPU decode has a fixed cost per matmul — a rayon dispatch and an
allocation — that does not shrink with the matrix. At this size that overhead,
not arithmetic, sets the speed. The GPU, which scales with actual work, shows
the expected 4.7x instead.

Metal `f32` runs at about two thirds of bf16's rate, for exactly double the
memory. `bf16` is the default because it is also what the checkpoints ship as.

The GPU backend covers the **Llama family only**; GPT-2 stays on the CPU engine,
and asking for it says so rather than failing obscurely.

### Quantised weights on the GPU

```bash
kvad-gpu run --quant q8   # or q4, q4k, q6k
```

candle's `QTensor` holds GGML's block formats — the same scheme as
[`quant.rs`](crates/llm/src/quant.rs): blocks with a shared scale, `Q8_0` and
`Q4_0` being 32-wide exactly like ours. `QMatMul` keeps HuggingFace's
`[out, in]` layout and transposes inside its kernel, where the dense path wants
the transpose done once at load, so [`Proj`](crates/gpu/src/model.rs) hides the
difference.

All six backends, Qwen2.5-0.5B, decode:

| backend | weights | tok/s |
|---|---|---|
| cpu f32 | 1976 MB | 34 |
| cpu q8 | 556 MB | 35 |
| cpu q4 | 309 MB | 36 |
| metal bf16 | 1260 MB | 115 |
| **metal q8** | **525 MB** | **193** |
| metal q4 | 278 MB | 236 |

Quantisation is worth **1.7x** on the GPU (115 → 193 tok/s) where it was worth
nothing on the CPU. Our CPU decode never became bandwidth-bound in the first
place: it has a fixed per-matmul cost — a rayon dispatch and an allocation —
that sets the speed at this model size, so shrinking the weights changed
nothing. The GPU has no such floor, so the bytes actually matter.

And then prefill, where it stops being free — eventually:

| prompt | metal bf16 | metal q8 | |
|---|---|---|---|
| 140 tokens | 0.070 s | **0.040 s** | q8 1.75x faster |
| 640 tokens | 0.180 s | **0.160 s** | q8 1.12x faster |
| 1340 tokens | 0.390 s | 0.380 s | even |
| 5040 tokens | **2.590 s** | 2.750 s | q8 1.06x *slower* |

There is a crossover, at around 1300 tokens on this machine. A short prompt is
still narrow enough that reading the weights dominates, so smaller weights win;
a long one turns every matmul into real work over a wide batch, and then the
dequantisation is pure overhead. Same principle as
[the two CPU kernels](#two-kernels-chosen-by-size), one bottleneck moving under
a different axis: there it was the size of the *matrix*, here it is the size of
the *batch*.

> **Correction.** An earlier version of this section claimed a flat "1.6x
> slower on prefill" from a single pair of measurements at 654 tokens (0.24 s
> against 0.39 s). That does not reproduce: five runs at each of four prompt
> lengths give the table above, with a median equal to the minimum every time.
> Re-running the old arrangement behind `KVAD_GPU_DENSE_EMBED=1` rules out the
> embedding change as the cause, so the original figure was simply a bad
> measurement on a loaded machine. The *shape* of the claim survives — there is
> a length past which quantising costs you — but it arrives much later and much
> more gently than reported. Measure the curve, not two points on it.

One practical note. The k-quants (`q4k`, `q6k`) need dimensions divisible by
256 — Qwen is 896 wide, so they are refused up front with a message naming
`q8` and `q4` instead of failing halfway through the load.

### The table that was stored twice

```bash
kvad-gpu run --quant q8                          # quantised table
KVAD_GPU_DENSE_EMBED=1 kvad-gpu run --quant q8   # the dense one, for comparison
```

The embedding table was the last dense tensor in a quantised GPU model, and on
a small model it is not a small one: 136M of Qwen2.5-0.5B's 494M parameters live
in `[151936, 896]`. At bf16 that is 272 MB.

It stayed dense because the lookup needs an operation nothing else in the engine
wants — *gather rows and dequantise only those*. A `QTensor` is a run of blocks,
not a matrix, so row `t` is a span of blocks that has to be decoded on its own.
candle turns out to ship exactly that kernel (`QTensor::embedding`, which is
GGML's `get_rows`), so the change is one method call rather than a Metal shader.

**The compression is the smaller half of it.** Qwen ties its embeddings — small
models nearly always do — so the lookup table and the output head are the *same
matrix*. They were stored twice anyway, because a dense `index_select` wants
`[vocab, n_embd]` and a matmul wants the transpose. Quantise the lookup and both
want the identical thing: one `Arc<QTensor>`, two uses.

| | before | after |
|---|---|---|
| metal q8 | 797 MB | **525 MB** |
| metal q4 | 550 MB | **278 MB** |

525 MB for 494M parameters is 8.5 bits each, which is exactly `Q8_0` — 8-bit
codes plus an f16 scale per 32. The model is now at the format's floor, with
nothing left dense but the norms, and the win over bf16 goes from 1.6x to
**2.4x** (4.5x at q4).

**And it changed the speed by nothing at all:** 196.6 → 195.7 tok/s at q8,
239.6 → 237.3 at q4, prefill identical to three decimals. Exactly as it should
be. One row of 151936 is read per token, so the table was never on the
bandwidth path — it was pure capacity. Everywhere else in this repo,
quantisation traded accuracy for speed. Here it trades accuracy for *room*, and
that is the whole point: memory is what stops you loading a bigger model, and a
bigger model is worth far more than 1% of tok/s.

The CPU engine needed no change for any of this. `Weight::row` has dequantised
one row at a time since quantisation was added, and the tied head has always
been the same `Weight` — the hand-written version got here by the path of least
resistance, because one type did both jobs. The framework version duplicated
precisely because its two operations wanted two different types.

### One trait, two backends

Adding the GPU needed a change to the CPU engine's shape.
[`Transformer`](crates/llm/src/model/mod.rs) deliberately keeps the KV cache
*outside* the model, because on the CPU it is a `Vec<f32>` the caller can own.
A GPU backend cannot work that way — its cache lives in device memory and must
never round-trip through the host between layers.

So the cache moved inside, behind a `Session` trait: `forward`, `cached`,
`truncate`. Everything above it — prefix reuse, sampling, streaming, the chat
loop, the TUI — stopped caring which backend it was talking to. The CLI keeps
its `--quant`, the GPU CLI gets `--device` and `--dtype`, and in the TUI `p`
cycles through all six: `cpu f32`, `cpu q8`, `cpu q4`, `gpu bf16`, `gpu q8`,
`gpu q4`.

That is the usual shape of this kind of work: the interesting part was not the
new backend, it was the seam the old one had to grow.

---

## Crate 4: `kvad-tui` — the app

```bash
cargo run --release -p kvad-tui
```

`/` search · `↑↓` select · `enter` download and load · `p` cycle backend ·
`d` delete · `tab` switch to chat · `esc` interrupt generation.

`p` cycles all six backends — `cpu f32`, `cpu q8`, `cpu q4`, `gpu bf16`,
`gpu q8`, `gpu q4` — and is the quickest way to feel the trade-offs: load a model at
f32, ask it something arithmetic, reload at q4 and ask again, then reload on
the GPU and watch the token rate.

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

Two tracks, one per job. They are independent: the learning track finishes the
story, the serving track is what the second half of the mission actually costs.

### Finishing the story

**1. Train your own.** A character-level transformer, 10–30M parameters, on a
corpus you pick. Backprop through attention is done
([`attention.rs`](crates/nanograd/src/attention.rs)); still needed are an
embedding, layernorm, the residual wiring of a block, and Adam. The gradient
check from crate 1 is how you will debug each one — extend `nanograd` (hard,
most educational) or use [`burn`](https://github.com/tracel-ai/burn).

**2. Fine-tune with LoRA.** Freeze the model, train two small low-rank matrices
per weight matrix. This is what "custom model" means in practice, and unlike
full fine-tuning it fits on a laptop.

### Becoming a server

In rough dependency order. The first one is not optional: without it the
kernels already written stay invisible.

**3. Remove the per-matmul floor.** CPU decode runs at ~35 tok/s whether the
weights are 1976 MB, 556 MB or 309 MB, and SmolLM2-135M is no faster than
Qwen-494M despite being a quarter the size and having *more* layers. Both say
the same thing: a fixed cost per matmul — a rayon dispatch and an allocation —
sets the speed, not bandwidth and not arithmetic. It is why `sdot`, SMMLA and
the tiled GEMM all measure well in isolation and vanish end to end, while the
same quantisation is worth 1.7x on the GPU. Reusable output buffers, coarser
work units, and one parallel region per layer rather than per matmul.

**4. A paged KV cache.** Today it is a `Vec<f32>` per layer that grows by
appending, which is 805 MB at Qwen's full context and cannot be shared between
sequences or reclaimed in pieces. Paging it into fixed blocks is what makes
several conversations fit in the memory of one, and it is a prerequisite for
everything below. Quantising it is a second, separate win.

**5. Continuous batching.** Half the machinery exists: `forward_batch` already
runs many positions through one set of weights, which is the whole reason
prefill is fast. Serving needs the same thing across *different sequences* at
different positions, which means per-sequence positions in RoPE and attention,
and admitting new requests between steps instead of between batches.

**6. `kvad-serve`.** An OpenAI-compatible HTTP endpoint, so the thing can be
pointed at by something that already exists. Deliberately last: a server around
a single-sequence engine measures nothing interesting.

**7. Then the hardware.** Real CUDA kernels, flash attention, speculative
decoding. This is the stretch where legibility and speed start to fight, and
the point at which this README owes an honest account of the trade.

### Worth reading alongside

- Karpathy, *Let's build GPT: from scratch, in code, spelled out*.
- *The Illustrated Transformer*, Jay Alammar — the diagrams.
- Vaswani et al., *Attention Is All You Need* (2017).
- Su et al., *RoFormer* (2021) — where RoPE comes from.
- Ainslie et al., *GQA* (2023) — grouped-query attention.

And for the serving track:

- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* (2023) — the vLLM paper, and step 4 above.
- Yu et al., *Orca* (2022) — where continuous batching comes from.
- Dao et al., *FlashAttention* (2022).
