[← Back to the README](../README.md)

# Crate 2: `kvad` — the same ideas, at scale

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
kvad arch                          # architectures this build can run
kvad cache                         # pre-quantised weight files
```

Read in this order:

1. **[`tensor.rs`](../crates/llm/src/tensor.rs)** — matmul, LayerNorm, GELU,
   softmax, then RMSNorm, SwiGLU and RoPE. Nine functions, four architectures.
2. **[`weights.rs`](../crates/llm/src/weights.rs)** — safetensors is a length, a
   JSON header, and raw floats. Plus shard indexes and bf16 widening.
3. **[`model/mod.rs`](../crates/llm/src/model/mod.rs)** — the skeleton the
   architectures share, including attention itself.
4. **[`model/gpt2.rs`](../crates/llm/src/model/gpt2.rs)** — read first, it is
   simplest. Then **[`model/llama.rs`](../crates/llm/src/model/llama.rs)**,
   written to be read as a diff against it, and
   **[`model/deepseek.rs`](../crates/llm/src/model/deepseek.rs)**, a diff against
   *that*. **[`model/arch.rs`](../crates/llm/src/model/arch.rs)** is the registry
   they plug into, and where to look when adding a fifth.
5. **[`quant.rs`](../crates/llm/src/quant.rs)** — block-wise int8/int4 weights
   and the kernels that consume them, then
   **[`simd.rs`](../crates/llm/src/simd.rs)** for the one instruction the compiler
   will not reach on its own, and
   **[`qcache.rs`](../crates/llm/src/qcache.rs)** for doing that work once
   instead of once per load.
6. **[`sampler.rs`](../crates/llm/src/sampler.rs)**, then
   **[`chat.rs`](../crates/llm/src/chat.rs)**.

## Six years of architecture progress, as a table

| GPT-2 (2019) | Llama family (2023+) | DeepSeek V2/V3 (2024+) | Why |
|---|---|---|---|
| learned position rows (`wpe`) | RoPE: rotate Q and K by angle ∝ position | RoPE on *part* of each head, YaRN-interpolated | the rest of the head has to survive being multiplied by a matrix |
| LayerNorm (centre, scale, bias) | RMSNorm (scale only) | RMSNorm, plus one on the compressed vector | the centring was never load-bearing |
| GELU MLP, 2 matrices | SwiGLU, 3 matrices | 64 or 256 SwiGLUs, a router, and 1–2 that always run | parameters you have, without arithmetic you pay for |
| multi-head attention | grouped-query attention | multi-head *latent* attention | the KV cache is what a server runs out of |
| one fused QKV matrix | three projections | two, one of which is a rank-512 bottleneck | Q and KV now have different widths, then different ranks |

The last row of that table is the interesting one, and it is worth stating as
numbers. For DeepSeek-V2-Lite — 16 heads, 128-wide values — ordinary
multi-head attention would store 5120 floats per position per layer.
Grouped-query attention would divide that by the group factor. MLA stores
**576**: one 512-wide compressed vector, and one 64-wide rotary key shared by
every head.

It gets away with it because the per-head matrix that would turn that vector
into a key can be moved onto the query instead — `q · (W c) = (Wᵀ q) · c` —
and there is one query per step against thousands of keys. That identity is
the whole architecture, and [`model/deepseek.rs`](../crates/llm/src/model/deepseek.rs)
is mostly an explanation of it.

What did *not* change across all three: the residual stream, the alternation
of attention and MLP, the causal mask, scaled dot-product attention. `attend()`
in `model/mod.rs` is shared verbatim between GPT-2 and Llama — grouped-query
attention is just a smaller `kv_dim`, and GPT-2 is the `n_kv_head == n_head`
case. DeepSeek is the first one that needed its own, which is what
[`model/arch.rs`](../crates/llm/src/model/arch.rs) exists for.

And the detail worth sitting with: **SmolLM2-135M is smaller than GPT-2-medium
and holds a conversation, while GPT-2 cannot.** The architecture changes above
are real but marginal. Nearly all of that gap is training data and
post-training. Running both in the same binary makes the point better than any
benchmark.

## The one that needed its own file

Everything above this point is true of GPT-2 and of Llama, and none of it was
true of the first mixture-of-experts checkpoint pointed at it.
DeepSeek-V2-Lite is the model this engine grew a registry for, and it is worth
looking at what it actually costs to run:

```
deepseek_v2 · 27 layers · 16 heads · 2048 embd · 163840 ctx · 102400 vocab
15706.5M parameters · weights 17670 MB (cpu q8) · loaded in 92.2s

def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

15.7 billion parameters, of which the router picks **2.4 billion** per token:
six experts of sixty-four, plus two that always run. It costs a 2.4B model to
decode and knows what a 16B model knows, and that is the entire argument for a
mixture of experts.

Decode, five interleaved rounds each, medians and the range across all
samples — [the protocol the benchmark page enforces](serve.md#what-it-measures-and-what-it-refuses-to),
run from a shell because one of these takes 17 GB of RAM:

| | weights | decode | prefill (13 tokens) |
|---|---|---|---|
| q8 | 16.5 GB | **23.4 tok/s** (22.8–24.7) | 0.47s (0.46–0.58) |
| q4 | 9.1 GB | **18.0 tok/s** (17.7–18.2) | 0.44s (0.44–0.49) |

**q8 decoded faster than q4 here, by 30%, while reading twice the bytes.** A
mixture of experts is the case where you would most expect the memory argument
to win — decoding touches only six experts of sixty-four, so q4 saves more than
a gigabyte of traffic per token — and it lost anyway.

The explanation this README gave was that decoding is `m = 1`, where there is
no [i8mm tile](#batched-prefill-and-i8mm) to take: `matvec_bt` walks one weight
row at a time, q8's goes straight into `sdot`, and q4's has to be masked and
shifted apart first. Reading half as much memory does not help once you are no
longer waiting on memory.

That was the right mechanism and much too large a number for it, and the
difference was a kernel nobody had written. The nibble unpacking was costing
about five times what it should, because of *which* loop LLVM chose to
vectorise; written out by hand it costs 9%, and
[the section that chases that down](#the-nibble-kernel-the-compiler-would-not-write)
is the more useful half of this story. Re-measured on the same model, five
interleaved rounds, with q8 as the control:

| | q8 (untouched) | q4 before | q4 after |
|---|---|---|---|
| decode | 24.8 tok/s (24.3–25.0) | 19.2 (18.0–19.5) | **26.8** (26.4–27.6) |

q4 goes from 22% behind q8 to **8% ahead**, on 7.4 GB less memory. The control
arm varies by 1% across ten runs, so the 40% is not the machine.

(Both tables were measured idle. The second reads a little faster on q8 too,
which is a different prompt and generation length rather than anything
changing: the q8 kernel was not touched, so take the control's agreement
between arms as the thing to trust and not the absolute rate. The first sample
of each run is discarded — faulting 16.5 GB of weights in from the page cache
costs a 6.5 s prefill against 0.7 s warm, and that is a disk measurement.)

q4 on this model is now what the name suggests — and it took two sessions and a
disassembler to stop it being a trade.

## The one that needed no file at all

Qwen3-30B-A3B is also a mixture of experts — 128 of them, eight per token — and
it added no architecture to this engine. Its attention is the `qwen3` block
already here, unchanged, and a mixture is not a kind of model: it is one of the
two answers to what a block does *after* it has attended. So the two questions
were separated. [`model/ffn.rs`](../crates/llm/src/model/ffn.rs) is the second
axis on its own, and `qwen3_moe` is the Llama block with the other answer on
it — a config read and a branch in the loader, and the same file runs both.

Even the routing came for free. DeepSeek scores its experts with a sigmoid and
a learned balancing bias, picks groups before it picks experts, and rescales
what it chose; Qwen3 does none of that — softmax over a flat list, the best
eight, renormalised to sum to one. Those are not different code, they are
different rows of the same config, and the router already defaulted every one
of them to the plain answer. The family that motivated the split added nothing
to it.

What it is worth being exact about: the two families disagree about how to
count layers. `moe_layer_freq` asks `layer % freq == 0` and
`decoder_sparse_step` asks `(layer + 1) % step == 0`. Both ship with the period
set to one, where the two rules are the same rule — so a single implementation
of it is correct today and silently wrong for half the layers of the first
checkpoint that ships a two.

## The one that stopped keeping a cache

`qwen3_5` — Qwen3.8 — is the first architecture here that does not keep its
past. Sixteen of its sixty-four layers are ordinary attention. The other
forty-eight are a **gated delta net**, and they hold one matrix per head
instead of a row per token:

```text
    S <- S · e^g       decay what is already there
    δ  = (v − Sᵀk)·β   how wrong S is about this key
    S <- S + k ⊗ δ     write the correction
    y  = Sᵀq           read it back
```

That is the delta rule. Rather than appending `k ⊗ v` and hoping, it asks what
`S` already returns for this key and stores only the difference, scaled by a
learned per-token `β`. `g` is a learned decay, so the layer chooses per token
how much to forget.

The cost is constant. `S` is the same size at position one and at position
262144 — 157 MB across those forty-eight layers, whatever the context. Sixty-
four layers of ordinary attention at full context would be paying per token for
all of it.

What it costs instead is prefix reuse, which is
[the decision recorded above](gpu.md#the-seam-that-cannot-rewind): the state has
already absorbed the tokens you want to drop, so `truncate` refuses and says
so. This is the architecture that made that seam necessary, and the first
backend to ever answer anything but "yes".

## Two implementations agreeing all the way to the wrong answer

`tests/qwen3_5.rs` writes the whole forward pass out a second time, longhand,
and checks the engine against it. Both passed on the first run. Both were
wrong.

Qwen3.5's RMSNorm scales by `1 + w`, not by `w`: the stored vector is an
*offset from one*, initialised to zeros. Every other architecture in this
repository scales by `w` directly, and so did both of these — the engine and
the second implementation written to check it, because the same misreading was
behind both. They agreed with each other to seven decimal places and disagreed
with the model by a factor of two on every norm in the network.

What caught it was running the real thing. `scripts/check-qwen3-5.py` loads the
same fixture into `transformers`' own `Qwen3_5ForCausalLM` and compares:

```text
    positions (7, 64)  worst absolute difference 5.018e+00  (scale 2.614)
      position 0: 3.228e+00  DISAGREE
```

and after the fix, on the same fixture:

```text
    positions (7, 64)  worst absolute difference 1.520e-05  (scale 2.739)
      position 0: 4.292e-06  ok
```

The lesson is not "write more tests". It is that a second implementation by the
same author checks the *algebra* and cannot check the *reading*, and that those
are different kinds of mistake. The same file, incidentally, uses the ordinary
`w` convention for its gated norm — two conventions in one checkpoint, and
nothing announces which is which.

## The second family in the same file

`qwen3_next` — Qwen3-Next-80B and Qwen3-Coder-Next — is the same mixer as
Qwen3.8 with a mixture behind it, and it went into the same module rather than
a new one. That was a finding, not a preference. Put the two published
implementations side by side with the names normalised away and the diff is
three things: where the decoder lives in the checkpoint, how the delta net's
inputs are spelled, and whether the feed-forward is one MLP or five hundred and
twelve. The convolution, the delta rule, the gated norm, the quarter-rotated
gated attention — identical, line for line.

So the module gained a three-line `Family` enum and two branches, and the
second architecture cost about as much as the first one cost after it.

The spelling is the part worth knowing about. Qwen3.5 writes four input
projections, each already in head order. Qwen3-Next writes two, laid out one
*key* head at a time — its query, its key, then the values and output gates of
the value heads that share them — and the engine takes them apart per token
rather than permuting the rows once at load, because those rows may be
quantised and slicing a quantised matrix apart is a kernel this does not have.
The cost is one pass over twelve thousand floats against the `[12288, 2048]`
matrix-vector product that just produced them.

The other difference is a shared expert with a mind of its own. DeepSeek's runs
for every token and is added as it comes; Qwen3-Next's is scaled by
`sigmoid(x · w)` first, so a token can decline it. Those are near enough alike
that running either under the other's rule would load, run, and be wrong by an
amount that varies per token — which is why
[`Shared`](../crates/llm/src/model/ffn.rs) is an enum with three variants rather
than an `Option<usize>` with two meanings.

And PyTorch settled it again, as it did for Qwen3.8. The Rust reference agrees
with the engine, which proves the algebra and not the reading; `Qwen3NextForCausalLM`
on the same fixture agrees to 4.7e-05 across every position, which proves the
reading.

## The shape a test fixture cannot have

Neither of those two architectures had ever loaded a real checkpoint. The tests
build their own, and the first real `qwen3_next` file refused to load:

```text
    Error: tensor `layers.0.linear_attn.conv1d.weight` not found in checkpoint
```

It was in the checkpoint. PyTorch stores a `Conv1d` with `groups = channels` as
`[channels, 1, kernel]` — the middle axis is the input channels *per group*,
which is one — and the loader read rank three, gave up, and reported the tensor
as missing from a file it was sitting in. The fixtures write the
two-dimensional spelling because that is the shape the engine wants, so nothing
in the suite could have caught it.

Two fixes, and the second is the one that matters. Accepting `[c, 1, k]` is a
line. Separating *not there* from *there and unreadable* is the change that
stops the next one of these being debugged from a wrong error message — a dtype
this engine cannot decode used to report as a missing tensor too.

The smallest real `qwen3_next` on the Hub is a 16 MB random-weight test model,
and it is the one that found this. It says nothing sensible, and both backends
say the same nothing, byte for byte.

## Base models versus instruction-tuned

GPT-2 is a **base** model: pure next-token prediction, no instruction tuning.
Ask it a question and it writes more questions, because that is what its
training data looked like. `kvad ls` and `kvad search` label which is which, and
`kvad chat` warns you.

An instruction-tuned model only behaves like an assistant when wrapped in the
exact marker tokens it was trained on. Those live as a **Jinja template** in
`tokenizer_config.json`, one per model, and they genuinely differ — so
[`chat.rs`](../crates/llm/src/chat.rs) renders the model's own template rather than
hardcoding one. "The model is dumb" is very often "the template is wrong".

## Quantisation

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

## The scale that wasted a code

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

| | ms/call | GB/s | vs f32 |
|---|---|---|---|
| f32 | 2.69 | 202 | 1.00x |
| q8 + quantised activations | 0.91 | 169 | **2.96x** |
| q4 + quantised activations | 1.28 | 67 | 2.11x |

That is close to the ceiling: 545 MB in 2.69 ms is 202 GB/s, and the q8 version
moves a third of the bytes at nearly the same rate. The speedup comes from
quantising the *activations* too, which turns the inner
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

## One kernel, after a threshold that measured the wrong thing

There used to be two kernels here, chosen by weight size. The integer path
ran 4.9x faster on the 153 MB output head and *slower* on a 5 MB MLP matrix,
so `matvec_bt` dispatched on `INTEGER_PATH_MIN_BYTES` and the explanation
wrote itself: a cache-resident matmul is not bandwidth-bound, so shrinking the
weights buys nothing and the unpacking is pure overhead.

Plausible, and wrong. The benchmark called the kernels from the main thread,
which is not a rayon worker, so every call paid a flat ~0.17 ms of cold-path
dispatch — invisible beside a 153 MB matmul, and the entire runtime of a 5 MB
one. The tell was sitting in the output the whole time:

```
qwen mlp.down  [896 x 4864]
  f32       17 MB     0.17 ms/call
  q8         5 MB     0.17 ms/call
  q4         3 MB     0.17 ms/call
```

Six times the bytes, the same time, three times over. That is not a kernel
being bandwidth-bound; that is a kernel that is not being measured at all.
With the harness fixed to run inside a pool ([`bench_matvec`](../crates/llm/examples/bench_matvec.rs)),
the same matrix drops to 0.09 ms and the integer path wins at every size —
1.13x to 1.70x end to end across three models and both precisions, nothing
slower. So the threshold is gone and there is one kernel. The dequantising one
survives as the baseline the integer path is measured against, behind
`KVAD_DEQUANT=1`.

## The nibble kernel the compiler would not write

For most of this project q4 decoded *slower* than q8, on every model, and this
README said so in three places with an explanation attached: `m = 1` decode
walks one weight row at a time, q8's row goes straight into `sdot`, and q4's
has to be masked and shifted apart first. Reading half the bytes does not help
once you are not waiting on bytes.

The mechanism was right. The size of it was not — and the size was an accident
of what LLVM decided to vectorise.

`sdot` multiplies `i8` by `i8`. A 4-bit code is stored unsigned, `0` to `15`,
with the bias taken off afterwards; mask one out of a byte and widen it and you
get a *zero*-extend, and a zero-extended operand against a sign-extended one is
not a shape `sdot` can take. So the q4 loop got `smull`/`smlal` instead —
eight lanes an instruction where `sdot` does sixteen, four of them per block,
with a widening step in front of each.

The obvious fix is to subtract the 8 inside the loop, which makes the weight
genuinely signed. It is exactly equivalent arithmetic — `sum((n-8)v)` and
`sum(nv) - 8 sum(v)` agree over the integers — and it made things **five times
slower**. Casting through `i8` without subtracting does nothing at all: LLVM
knows the top bits are zero and folds the sign-extend straight back. Measured
on one core, a 896-wide row, block dots only:

| | GMAC/s |
|---|---|
| unsigned nibbles, bias per block *(what shipped)* | 24 |
| signed nibbles, same iterator form | 4 |
| signed, one fused accumulator | 7 |
| signed, with the block sizes as array types | 2 |
| q8, for scale | 88 |

Every attempt to make the inner loop signed made the autovectoriser abandon the
reduction and vectorise *across blocks* instead — loading weights a byte at a
time into lanes with `ld1.b`, dozens of single-byte loads where there had been
one `ldr q`. The disassembly is unambiguous about it, and no amount of
rephrasing the Rust talked it out of the idea.

So the kernel is written down, next to `SMMLA` in [`simd.rs`](../crates/llm/src/simd.rs),
and it is ten lines:

```rust
let packed = vld1q_u8(w);                      // 16 bytes = 32 weights
let bias = vdupq_n_s8(-8);
let lo = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f))), bias);
let hi = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), bias);
let acc = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x));
vdotq_s32(acc, hi, vld1q_s8(x.add(16)))
```

Two ops remove the bias from thirty-two weights at once, which is the thing the
scalar loop could not express. Four blocks run per iteration so that `vpaddq`
reduces four accumulators in three instructions instead of four `addv`s. That
lands at **80 GMAC/s against the portable path's 24** — and within 9% of q8,
which is what unpacking nibbles should actually cost.

Note what this is *not*. It is not an instruction Rust cannot name; `vdotq_s32`
is stable and the compiler emits `sdot` for q8 without being asked. It is the
compiler declining to *choose* it — the same reason `SMMLA` is hand-written one
screen up, arrived at from the opposite direction.

The old path is still there for CPUs without `dotprod`, and `KVAD_NO_DOTPROD=1`
selects it, which is how everything below was measured. Both produce
bit-identical output — verified, not assumed — because the two ways of removing
the bias are the same integers.

## What the kernel is worth end to end

Interleaved rounds, `KVAD_NO_DOTPROD=1` against the kernel, same binary and
the same weights, medians with the range across all samples. **q8 is the
control**: its kernel was not touched, so whatever its two arms differ by is
what this machine's noise is worth.

| | q8 (control) | q4 before | q4 after | |
|---|---|---|---|---|
| Qwen2.5-0.5B | 110.7 tok/s (107–111) | 91.3 (84–96) | **116.4** (114–117) | 1.27x |
| DeepSeek-V2-Lite 15.7B | 24.8 (24.3–25.0) | 19.2 (18.0–19.5) | **26.8** (26.4–27.6) | 1.40x |
| SmolLM2-135M | 173.7 (169–176) | 159.2 (155–171) | **175.9** (170–182) | 1.10x |

The control's two arms agree within 1% on all three models; q4 moves 10–40%.
The gain tracks matrix size, which is what it should do — a 135M model spends
most of a decode step in dispatch, and there is no kernel for that.

**q4 is now the fastest CPU precision on every model here**, which has not been
true before: 116 against q8's 111 on Qwen, 176 against 174 on SmolLM2, 26.8
against 24.8 on DeepSeek.

And the microbenchmark, which is where it is cleanest:

```
qwen lm_head  [151936 x 896]
  f32      545 MB     2.67 ms/call   1.00x
  q8       153 MB     0.89 ms/call   2.99x
  q4        85 MB     0.74 ms/call   3.60x
```

## What it is worth end to end

| | f32 | q8 | |
|---|---|---|---|
| Qwen2.5-0.5B | 64 tok/s | 112 tok/s | 1.75x |
| SmolLM2-135M | 165 tok/s | 194 tok/s | 1.18x |
| GPT-2-medium | 67 tok/s | 125 tok/s | 1.86x |

Quantisation buys most of what the byte counts say it should, which is worth
stating because for most of this project's life it bought nothing at all —
every one of those f32 and q8 numbers used to be about 35 tok/s, whatever the
model and whatever the precision. [Removing the floor](#the-floor-under-everything)
is what let the kernels through; before that, the honest summary of this
section was "a 4.9x kernel bought under 1.4x overall", and the Amdahl's-law
moral drawn from it was measuring a scheduler.

For a long time the honest footnote here was that q4 bought none of this: at
91 tok/s it was *slower* than q8's 112, so its only advantage was memory. That
was [a missing kernel, not a price quantisation charges](#the-nibble-kernel-the-compiler-would-not-write).
q4 now decodes at 116, ahead of q8. It is still much less accurate, and that
part is real.

## Batched prefill, and i8mm

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

Prefill, 640 tokens, Qwen2.5-0.5B, q8:

| | time | |
|---|---|---|
| one token at a time (`KVAD_PREFILL_CHUNK=1`) | 6.20 s | |
| batched f32 GEMM | 1.93 s | **3.2x** from batching |
| batched q8, `SDOT` (`KVAD_NO_I8MM=1`) | 1.81 s | |
| batched q8, `SMMLA` | 1.20 s | **1.51x** more from i8mm |

Prefill drops from ~9.7 ms/token to ~1.9 ms/token.

Note where f32 lands: **1.93 s against q8's 1.81 s.** Once the weights are
reused across a batch, prefill is compute-bound, and quantisation is an
answer to a memory problem. On the smaller `q_proj` matrix the f32 GEMM is
actually *faster* than the q8 `SDOT` path, because unpacking and rescaling cost
more than the bytes they save. Quantisation is mostly a decode optimisation.

`i8mm` is worth more on a single core than across the pool, because with every
thread running memory bandwidth becomes the limit again and a faster
instruction has less to offer. Both knobs above are environment variables
precisely so this is measurable rather than asserted.

(Every figure in this table used to be two to three times larger — the
token-at-a-time row was 18.7 s. Batching was never the only thing making
prefill slow; see [the floor](#the-floor-under-everything).)

Rust exposes `SMMLA` only through an unstable intrinsic, so
[`simd.rs`](../crates/llm/src/simd.rs) emits it with inline assembly — stable, and
four lines. Availability is detected at runtime, so one binary still runs on
CPUs without it.

**q4 takes this kernel too, and for a while it did not.** The gate was one
condition — `matches!(self.data, Data::Q8 { .. })` — and everything at q4
prefilled on the row-at-a-time fallback while unpacking nibbles on top. The web
UI's benchmark page is what turned it up, by measuring time to first token
beside decode rate and showing q4 losing the one while winning the other.

The fix is a format problem rather than a kernel problem. `SMMLA` wants eight
contiguous `i8`; q4 stores two weights per byte, with the *low* nibble of each
byte belonging to the first half of a block and the high nibble to the second,
and the sign carried as a bias removed later during scaling. Unpacking a row
pair into plain `i8` — subtracting the 8 on the way, which is exactly the
correction `row_dot` applies afterwards — hands the existing kernel an operand
it already understands, and costs `n` bytes of work reused across all `m`
activation rows.

Prefill, 862 tokens of this README, Qwen2.5-0.5B, five rounds interleaved:

| | median | range | |
|---|---|---|---|
| q4, row at a time (`KVAD_NO_I8MM=1`) | 4.58 s | 4.05–4.77 | |
| q4, `SMMLA` | 2.54 s | 2.02–2.74 | **1.8x** |
| q8, `SMMLA` | 2.52 s | 2.03–2.78 | |

A second run of five put q4 at 2.15 s against 4.68 s, so the gain is 1.8–2.2x
depending on the round. The absolute times are worse than the table above
because the machine was not quiet; the ratios are the claim, and interleaving
the settings is what makes them survive that. The line that matters is the
third: q4 prefill went from roughly half q8's speed to level with it, while
still decoding faster. In the server's own benchmark, time to first token at q4
moved from 41 ms to 33.8 against q8's 34.3 — a difference that has stopped
existing.

### Checking that, after the decode kernel turned up a five-fold hole

[The 4-bit decode kernel](#the-nibble-kernel-the-compiler-would-not-write) was
5x off from what the source suggested, and the section above makes the same
kind of claim about prefill on a noisier measurement — so it is worth asking
whether prefill has a hole of its own. It does not, and the way that was
established is the point.

The first problem was that nothing measured it. `bench_batch` had rows for
`f32 gemm`, `q8 sdot` and `q8 smmla` and no q4 at all: the benchmark's blind
spot was exactly where the question lived. With q4 in it, and swept across the
batch size rather than measured at one:

| m | q8 `SMMLA` | q4 `SMMLA` | q8 `SDOT` | q4 `SDOT` |
|---|---|---|---|---|
| 2 | 77.9 | 71.6 | 71.1 | 66.7 |
| 8 | 153.6 | 151.6 | 130.5 | 131.2 |
| 32 | 285.4 | 279.7 | 141.9 | 146.1 |
| 64 | 306.3 | 306.0 | 161.0 | 161.5 |
| 128 | 319.7 | 309.6 | 207.4 | 210.8 |

GMAC/s on `mlp.down`, `[896 x 4864]`. q4 is level with q8 from `m = 8` up and
at worst 8% behind at `m = 2`, where the unpack is amortised across exactly one
pass of the kernel. `SMMLA` beats the row-at-a-time path at every batch size,
so the `m >= 2` gate is right too. End to end, 685 tokens on Qwen2.5-0.5B, six
rounds interleaved: **q4 1.34 s against q8 1.36 s.**

And the disassembly says why there was nothing to find. The nibble unpack
compiles to `ldr q` / `and.16b` / `usra.16b` / two `str q` — sixteen bytes in,
thirty-two out, fully vectorised. The split packing is the reason: the low
nibbles of sixteen bytes *are* sixteen consecutive weights, so unpacking writes
two contiguous runs and never scatters. The decode kernel's problem was never
that nibbles are awkward; it was one `zext` in a reduction the vectoriser was
looking at.

**One thing did change, in the path nobody was looking at.** A CPU without
`i8mm` prefills on `row_dot`, which calls the same block-dot the decode kernel
replaced. So the decode work lifted prefill too, on hardware that cannot reach
`SMMLA` at all — `KVAD_NO_DOTPROD=1` against it, `m = 64`:

| | q4 `SDOT` before | after | | vs q8 |
|---|---|---|---|---|
| `mlp.down` | 117.1 | **161.5** | 1.38x | 0.75x → 1.00x |
| `q_proj` | 95.3 | **138.1** | 1.45x | 0.73x → 1.02x |

On `q_proj` that also puts q4 ahead of the f32 GEMM (138.1 against 134.9),
which the section above notes was the one shape where quantisation lost on the
fallback. It no longer does.

So: no kernel written here, and the measurement that would have caught it if
there were is now in the benchmark instead of absent from it. Worth writing
down at the same length as a fix, because "we looked and there was nothing"
is only useful if you can see how hard anyone looked.

## The KV cache, and a fix that was slower than the bug

The [roadmap](roadmap.md#where-to-go-next) has called the KV cache the next bottleneck for
a while, on the grounds that it is the only part of a decode step that grows
with the conversation. Everything else — every matmul, all 169 of them — is the
same size at position 10 and at position 4000. So it is worth knowing what it
actually costs, and until `bench_attend` existed nothing measured it.

`attend` scores this token's query against every cached key. The loop was
written the obvious way:

```rust
let mut dot = 0.0f32;
for i in 0..hd {
    dot += q_head[i] * k_head[i];
}
```

One running sum — which [`tensor::matvec_bt`](../crates/llm/src/tensor.rs), in the
next file along, already carries a comment warning against: *"four lanes, not
one running sum: a single chain would stall on FMA latency rather than run at
its throughput."* DeepSeek's latent attention had the same loop over a
512-wide vector, so its chain was 512 dependent adds long.

**Then the fix turned out to be slower than the bug.** Four lanes, the form the
repo already uses, on a 64-long dot over a 2048-entry cache, one core:

| | |
|---|---|
| one running sum *(what shipped)* | 17.6 us |
| four lanes | 22.8 us |
| **eight lanes** | **6.3 us** |

The disassembly says why, and it is not what the comment assumes. LLVM cannot
reorder float additions, but nothing stops it vectorising the *multiplies*: the
scalar loop compiles to four `FMUL.4S` and a chain of scalar `FADD`s, and
because consecutive positions are independent, the out-of-order engine overlaps
those chains across `t`. Four lanes replaces that with a single 128-bit
accumulator carried around the loop — one vector FMA per iteration, each
waiting on the last, and nothing to overlap it with. Eight lanes is two
independent chains, which is the first shape that actually beats doing nothing.

So the rule is not "lanes are better than a running sum". It is *two
accumulators or more*, and four lanes only looks like a fix because a 128-bit
vector makes it look like four.

`tensor::dot` is now eight lanes and shared, and `attend` and DeepSeek's MLA
call it instead of writing the loop out. Both versions of `attend` live in
[`bench_attend`](../crates/llm/examples/bench_attend.rs), alternating in one
process and checked against each other every round:

| ctx | running sum | eight lanes | | cache/layer | attention alone |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B shape** — 14 heads, 2 KV, `head_dim` 64, 24 layers ||||||
| 512 | 22.2 us | 18.6 | 1.19x | 0.52 MB | 0.45 ms/token |
| 2048 | 74.8 | 60.3 | 1.24x | 2.10 MB | 1.45 |
| 8192 | 332.3 | 248.1 | **1.34x** | 8.39 MB | 5.95 |
| **Llama-8B shape** — 32 heads, 8 KV, `head_dim` 128, 32 layers ||||||
| 128 | 49.6 | 37.2 | **1.33x** | 1.05 MB | 1.19 |
| 512 | 148.8 | 108.3 | **1.37x** | 4.19 MB | 3.47 |
| 2048 | 505.4 | 420.3 | 1.20x | 16.78 MB | 13.45 |
| 8192 | 2293.1 | 2275.6 | **1.01x** | 67.11 MB | 72.82 |

**The last row is the more useful result.** At 67 MB of cache per layer the fix
buys nothing at all, because the loop has stopped being latency-bound and
become DRAM-bound — and the last column says why that matters: attention alone
is 73 ms per token there, which is most of the token.

**What it is worth end to end is smaller, and today unmeasurable.** Attention
is a minority of a decode step at these lengths — 1.45 ms of a ~12 ms token at
2048 on Qwen — so 1.24x of it predicts about 3%. Driving a real session at 4096
tokens of context, five rounds alternating, that is what the paired rounds
suggested (1.00x, 1.06x, 1.02x, 1.06x, 1.05x) — but the control, the same
measurement at 64 tokens of context where the fix has almost nothing to do,
swung 0.88x to 1.03x across those same rounds. The control's noise is wider than the effect, so
the honest answer is that this machine could not resolve it, and the isolated
numbers above are the claim. It gets decisive at long context on a big model,
and that is exactly where the second finding says the fix stops helping.

Which is the argument for the paged cache, arrived at from the measurement
rather than from the memory figure. Under grouped-query attention each cached
key is read **once per query head in its group** — seven times on Qwen, four on
that Llama shape — because `attend` parallelises over query heads and each one
walks the cache for itself. The bytes are shared; the reads are not. Fixing
that means restructuring attention around the KV head rather than the query
head, with the softmax split across position blocks, which is a different and
much larger change than a dot product. It is now a measured number instead of
an assumption, which is the part that was missing.

## The bug worth stealing

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

## The f32 GEMM, and register tiling

Batching only helped the quantised paths at first: with f32 weights `matmul_bt`
still called the matrix-vector kernel once per token, which reads the whole
weight matrix `m` times. [`gemm_bt`](../crates/llm/src/tensor.rs) fixes that, and
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

## The optimisation that wasn't

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

In [`matvec_bt`](../crates/llm/src/tensor.rs), where only four accumulators are
live, `mul_add` made no measurable difference at all — that kernel is
bandwidth-bound at ~220 GB/s, so its instruction count is irrelevant. Which is
the more useful half of the lesson: an optimisation is a claim about the
bottleneck, and if you have not measured the bottleneck you are guessing.

## Verifying you got it right

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

## Doing the quantising once

Every `--quant q8` load re-derived the same 494 million codes from the same
bf16 checkpoint and threw them away on exit. [`qcache.rs`](../crates/llm/src/qcache.rs)
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

The GPU backend used to be exempt from all this, on the grounds that candle's
`quantize_onto` takes about 0.2 s for the same model — fast enough not to be
worth a second format. That was true of the model it was measured on and of no
other: [it does not survive a 16B checkpoint](gpu.md#the-quantising-done-once-over-there-too).

## The floor under everything

For most of this project, CPU decode ran at about 35 tok/s. Not approximately
— *exactly* that, whatever you changed:

| | f32 | q8 | q4 |
|---|---|---|---|
| Qwen2.5-0.5B (1976 / 556 / 309 MB) | 34 | 35 | 36 |
| SmolLM2-135M, a quarter the size | ~35 | ~35 | ~35 |

Six times the bytes, the same speed. A model a quarter the size, the same
speed. Every kernel in this chapter — the `sdot` integer path, the SMMLA
tile, the tiled GEMM — measured faster in isolation and changed nothing here.
The explanations on offer were all plausible (a fixed cost per matmul; more
layers in the smaller model) and none of them were checked.

Six seconds with a sampling profiler ended it. Top of stack:

```
swtch_pri (kernel yield)                12776
the actual matvec kernel                 5186
__psynch_cvwait                          3652
__workq_kernreturn                       1585
```

One fifth of the CPU was doing arithmetic. And the main thread's stack was the
same every time:

```
matvec_bt -> Registry::in_worker_cold -> LockLatch::wait_and_reset
          -> pthread_cond_wait
```

**`in_worker_cold`.** A `par_iter()` called from a thread that is not a pool
worker cannot push onto a worker's deque; it has to inject the job into a
shared queue, wake the workers, and block the caller on a condition variable
until they are done. Two kernel transitions and a scheduler round trip. Fine
once. Decoding one token runs 169 matmuls, so it happened 169 times per token,
each one costing more than the arithmetic it was waiting for.

Three changes, measured one at a time:

| | | |
|---|---|---|
| `pool.install()` around the forward pass | 35 → 70 | caller blocks once per *token*, not per matmul |
| one job per thread instead of a split tree | 70 → 90 | no adaptive `join` tree, no steals between leaves, no `collect` |
| size the pool for big.LITTLE | — | 18 threads was slower than 11; see below |
| delete the integer/dequant threshold | 90 → 112 | [it measured this same bug](#one-kernel-after-a-threshold-that-measured-the-wrong-thing) |

**3.2x on q8, and the anomalies went with it.** SmolLM2-135M now decodes at
194 tok/s against Qwen-494M's 112, which is what a quarter the parameters
should look like. Quantisation now buys 1.75x where it used to buy nothing.
Prefill of 640 tokens went from 2.15 s to 1.16 s without being touched.

### Not all cores are worth using

The pool is deliberately smaller than the machine. This host reports 18
logical CPUs, of which six are performance cores and twelve are efficiency
cores. Rayon splits a matmul into equal pieces regardless, and a parallel
section ends when its *slowest* piece does — so the E-cores set the pace of
every one of those 169 barriers:

| threads | 4 | 6 | 8 | 10 | 12 | 14 | 15 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| tok/s | 36 | 46 | 54 | 61 | 65 | 69 | 70 | 68 | 52 |

Using the whole machine is the worst setting above four threads. The default
is now the performance cores plus two thirds of the efficiency cores, and
`KVAD_THREADS` overrides it.

### What this cost in credibility

Three things in this README had to be rewritten because of one bug: the
integer/dequant threshold, the "Amdahl's law" moral drawn from a 4.9x kernel
buying 1.4x, and the claim that SmolLM2's layer count explained its speed.
All three were reasonable stories told about a number that was really a
scheduler. The profiler took six seconds and would have found it at any point
in the previous four chapters.
