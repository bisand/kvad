# Plan: LittleBit — weights below one bit

Written 2026-11-21. The method is Samsung Research's, from two papers and one
repository:

- **LittleBit** — Lee, Kim, You & Kim, *Ultra Low-Bit Quantization via
  Latent Factorization*, NeurIPS 2025 ([arXiv 2506.13771](https://arxiv.org/abs/2506.13771)).
- **LittleBit-2** — Lee & Kim, *Maximizing the Spectral Energy Gain in
  Sub-1-Bit LLMs via Latent Geometry Alignment*, ICML 2026
  ([arXiv 2603.00042](https://arxiv.org/abs/2603.00042)).
- [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit), read
  at `42d658b`. PyTorch, training and evaluation, CC BY-NC 4.0.

Step 1 below is built. The rest is what it would take to go further, and the
questions that decide whether to.

## Why

Every format Kvad had stores the weight matrix: f32, or eight or four bits a
weight plus a scale per block of 32. One bit a weight is where that line ends,
and an 8B model at one bit is still a gigabyte. LittleBit goes under it by
storing a low-rank factorisation of each matrix and quantising the *factors*
to signs:

```text
W  ≈  diag(u1) · U · diag(u2 ⊙ v1) · V · diag(v2)        U: [out, s] of ±1,  V: [s, in] of ±1
```

The rank `s` sets the size. The checkpoints published so far add a second,
residual factorisation of what the first missed, at the same rank. Qwen3-8B
at 0.55 bits per weight is rank 544 on the 4096-wide projections and 824 on
the MLP, and the whole model is 1.9 GB in memory, embeddings and head at q8
included. At q8 throughout it would be 9.2 GB.

## What the method is

Two parts, and they are very different amounts of work.

**The format, and the arithmetic to run it.** Each `nn.Linear` becomes the
tensors below, per path, with `_R` on the residual path's names:

| tensor | dtype | shape | |
|---|---|---|---|
| `U_packed` | int32 | `[out, ⌈s/32⌉]` | signs, 32 to a word, LSB first; a set bit is −1 |
| `U_shape` | int64 | `[2]` | `[out, s]` |
| `V_packed` | int32 | `[s, ⌈in/32⌉]` | |
| `V_shape` | int64 | `[2]` | `[s, in]` |
| `u1`, `u2`, `v1`, `v2` | bf16 | `[1, out]`, `[1, s]`, `[1, s]`, `[1, in]` | |
| `_eff_bit_target`, `_eff_bit_actual`, `_split_dim_final` | | scalars | bookkeeping |

`lm_head` and the embedding table stay dense. The forward pass never rebuilds
`W`: `((((x ⊙ v2) Vᵀ) ⊙ (v1 ⊙ u2)) Uᵀ) ⊙ u1`, and the two matrix products are
sums of signed values. `crates/llm/src/littlebit.rs` explains this at length.

**Making the factors.** Three stages, from `quantization/modules/littlebit.py`
and `main.py`:

1. *Initialise* from the dense weight: a rank-`s` truncated SVD
   `W ≈ (U_t √S)(√S V_tᵀ)`; then a rank-one SVD of `|U|` and of `|V|` to get
   the four scale vectors. With `--residual`, the same again on what that
   leaves over.
2. *LittleBit-2's rotation* (`--use_itq`), before the scales: Joint
   Iterative Quantization. Stack `U` on `Vᵀ`, and look for the orthogonal
   `R` that brings the rows closest to the corners of the hypercube —
   `min ‖sign(XR) − XR‖`, alternating `B = sign(XR)` with an orthogonal
   Procrustes step (an SVD of `BᵀX`), fifty rounds. Then `U ← UR`,
   `V ← RᵀV`. The product `UV` is unchanged, so the rotation is free at
   inference and only changes how much the rounding to signs loses.
3. *Quantisation-aware distillation*: train `U`, `V` and the four vectors
   with the signs in the forward pass, against the original model as
   teacher. The sign's gradient is SmoothSign, `α(1 − tanh²(αx))` with
   α = 100. The loss is the KL divergence of the logits plus ten times the
   mean squared error of every layer's hidden state. The paper trains five
   epochs of C4 and WikiText-2 at 2048 tokens; the published 8B was trained
   half an epoch on one H200.

The third stage is the method. Without it the factors are a rank-544
approximation with every number in it rounded to a sign, and the model does
not survive it.

## Step 1: running them — built

A checkpoint that holds `{stem}.U_packed` where a loader asks for
`{stem}.weight` is read as LittleBit factors and runs as a `Weight` like any
other, so every architecture that loads its projections through
`Source::matrix` runs LittleBit checkpoints without knowing it. Unpacked
checkpoints — from before the reference packed, with the latent `U` and `V`
in bf16 — are read too, by taking their sign once.

- `crates/llm/src/littlebit.rs`: the format, the two kernels, the reader.
- `crates/llm/src/quant.rs`: `Weight` gains the variant, and dispatches to
  it from `matvec_bt` and `matmul_bt`.
- `crates/llm/src/qcache.rs`: `Live` reads the factors when there is no
  dense weight, and the cache file gains a `littlebit` entry, so the second
  load maps the signs rather than reading the checkpoint.
- `crates/llm/src/weights.rs`: `Checkpoint::read_words`, for integer tensors
  that are bits rather than numbers.

### How it was checked

- `crates/llm/tests/littlebit.rs` writes a tiny Qwen3 twice: once as packed
  LittleBit tensors in the reference's exact layout, with residual paths and
  ranks that are not whole words, and once as the dense matrices those
  factors multiply out to, computed in the test entry by entry. The packing
  there is written out independently of `littlebit.rs`. The two models agree
  on logits through prefill, through decode a token at a time, and through
  the cache file and back.
- Three projections of the real Qwen3-8B checkpoint (the first layer's
  `k_proj`, a middle layer's `down_proj`, the last layer's `o_proj`) are
  run through Kvad by an ignored test in `tests/littlebit.rs`, and through
  `scripts/check-littlebit.py` — a numpy transcription of the reference's
  `binary_unpacker` and `LittleBitLinear.forward`, in float64, reading the
  safetensors bytes itself. Largest relative difference: 3.6·10⁻⁷, both
  kernels.
- Perplexity on WikiText-2's test split, the reference's protocol
  (2048-token windows, the rows joined by blank lines): see the table below.
- `bits_per_weight` reproduces the checkpoint's own `_eff_bit_actual`:
  0.5479, which is 0.546875 in the bf16 it is stored as.

### What it measured

[`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw),
a community checkpoint made with the reference code: LittleBit-2, residual,
0.55 bits a weight, half an epoch of distillation. On a 4-core x86-64 cloud VM
(SSE2 build, which is what Kvad's Linux release is), q8 embeddings and head:

| | |
|---|---|
| weights in memory | 1889 MB for 8.19 B parameters |
| first load (reads the checkpoint, writes the cache) | 29 s |
| later loads (maps the cache) | 0.4 s |
| prefill, 20 tokens | 2.1 s |
| decode | 4.4 – 4.7 tok/s |
| WikiText-2 perplexity, first 3 windows of 2048 | 26.9 (the windows alone: 18.0, 28.0, 38.6) |
| the model card's, every window, PyTorch on an A40 | 24.48 |
| prefill of one 2048-token window, logits for every position | 10.6 min |

Three windows are not the whole test set, and these three disagree with each
other by a factor of two, so 26.9 against 24.48 is agreement and not a
finding. What it rules out is the failure that matters: a misread sign, scale
or bit order gives perplexities in the hundreds or thousands, not in the
twenties.

Quality is the checkpoint's, not this engine's: its card puts ARC-Challenge
at chance, and greedy answers to simple questions wander off. A checkpoint
trained the paper's five epochs would be the fair test of the method; none is
published yet.

### What turned out to matter

- **The batched kernel, twice over.** Prefill first ran at the same cost per
  token as decode. Tiling the activations for cache did nothing; blocking
  four sign rows against each activation load, eight lanes each, was 5x. And
  then a copy of that same loop ran at a third of the speed inside Kvad: the
  compiler vectorised it across the four *rows*, with a shuffle per column,
  because the tile's width was a runtime value. Fixed-size arrays took the
  choice away from it. The table is in `sign_gemm`'s documentation.
- **Which way to turn a bit into a sign** depends on the instruction set:
  SSE2 has no per-lane shift, NEON does. `signed_sum` picks per
  architecture, and says what each choice measured.
- **The integer path that did not pay.** An `i8` version of the decode
  kernel — signs expanded through a 256-entry table, the activation
  quantised to `i8` per block of 32, which is a word of signs — ran at
  2–2.5 G sign-adds a second against the float version's 5 here. It might
  be another story with `sdot` on the M5; it was not measured there.

## What is not done, in the order it would be

### 2. Measure it on the M5

Measured 2026-11-22 on an M5 Pro (6 + 12 cores, 48 GB), CPU, same checkpoint,
q8 embeddings and head, a 27-token prompt and 64 greedy tokens, four runs:

| | |
|---|---|
| first load (reads the checkpoint, writes the cache) | 15.3 s, 4.57 GB peak footprint |
| later loads (maps the cache) | 0.1 s, 176 MB footprint, 1.4 GB resident |
| prefill, 27 tokens | 1.13 – 1.18 s |
| decode, default threads | 10.6 – 11.1 tok/s |

Decode by `KVAD_THREADS`, two runs each:

| threads | decode, tok/s | system time, s |
|---|---|---|
| 4 | 6.7 – 6.8 | 1.2 – 1.5 |
| 6 | 8.0 – 8.1 | 4.7 – 4.8 |
| 8 | 8.6 – 8.8 | 9.9 |
| 12 | 8.4 – 9.9 | 18.5 – 19.7 |
| 18 | 9.4 – 10.1 | 36.9 |

Going from 4 threads to 18 buys 1.5x while system time grows about 25x, so
this does not yet show the compute-bound scaling the arithmetic predicted.
There is no dense or q4 Qwen3-8B on this machine to compare with, so whether
LittleBit is *faster* than the other formats here is still open. The numpy
check and the perplexity were not repeated on this machine.

**Where the system time goes.** `sample` during decode puts it in `swtch_pri`,
rayon's idle workers yielding while they wait for work. It is a symptom of
waiting rather than a cost of its own:

| threads | in `signed_sum` | yielding | sleeping |
|---|---|---|---|
| 4 | 86% | 2 – 4% | 2 – 3% |
| 14 (default) | 40 – 53% | 28 – 43% | 4 – 11% |

Cutting each section into 2, 4, 8 or 16 times as many tasks, so fast cores
could steal from slow ones, changed decode by less than the run-to-run spread
(four interleaved rounds), so uneven chunks are not the cause.

**What they wait for is the start and end of each parallel section.** A token
makes 252 projections, so 504 sections. Timers inside every task, in ms per
token:

| threads | token | arithmetic, if perfectly shared | waiting inside sections | outside sections |
|---|---|---|---|---|
| 4 | 141.3 | 123.3 | 6.3 | 11.7 |
| 6 | 108.9 | 87.0 | 11.9 | 10.0 |
| 14 | 91.3 | 41.2 | 41.1 | 9.0 |

At 14 threads the average section's last task starts 51 µs after the section
opens, and the section closes 25 µs after its last task ends, against about
80 µs of arithmetic per task. Those two delays are 39 of the 41 ms. Workers
also miss sections: each of 13 ran about 97% of its share, and the first
worker picked up the rest.

Removing all of the wait would take a token to about 50 ms, or 20 tok/s. That
is a ceiling, not a projection.

**Merging sections did not reach it.** Running Q, K and V as one job, and gate
and up as another, cuts a token from 504 sections to 288. If each section had
a fixed 76 µs price that would save about 16 ms a token. Eight alternating
rounds of the old and new builds measured no gain: median 10.14 tok/s before,
10.02 after, paired ratios from 0.89 to 1.05, and the text identical in every
round. So the wait is not a fixed price per section. It grows with the work
in the section, and what causes that is still not identified. The merge was
not kept.

### 3. Prefill on the matrix units

`sign_gemm` is f32 multiply-adds against `±1.0`. A row pair of signs
unpacked to `i8` is exactly what the q4 kernel already builds for `SMMLA`
(`unpack_q4_pair`), so prefill could run on i8mm like every other quantised
weight here, at the cost of quantising the activations — which every other
quantised weight here already pays.

### 4. The GPU

`kvad-gpu` reads checkpoints itself and asks for `{stem}.weight`, so a
LittleBit checkpoint fails there with a missing tensor. Supporting it is a
Metal kernel for the two sign products and a `Loader` that recognises the
format. Until then the CPU is the only backend, and the error could at least
say why.

### 5. Making LittleBit models here

Stages 1 and 2 (SVD, Joint-ITQ) are linear algebra this repository does not
have: a truncated SVD of every weight (randomised subspace iteration and a
small dense SVD would do), and fifty orthogonal Procrustes steps of an
`s x s` SVD per layer. On the CPU that is a long afternoon per 8B model,
not a week.

Stage 3 is training. `kvad-gpu` already differentiates through candle, runs
a fused AdamW, and keeps frozen q8 matrices with their own backward for
LoRA (`grad.rs`, `adam.rs`): the teacher is a frozen model, and the student
is the same model with every projection replaced by trainable factors and a
SmoothSign with a custom backward. What it would not have is room: a teacher
and a student of the same model, the student's latent factors in f32 with
two AdamW moments each, and activations at 2048 tokens. A 0.6B or 1.7B
model on an M5 Pro is plausible. 8B is what the reference does on an H200.

## Licence

The reference code is CC BY-NC 4.0. Nothing in Kvad is copied from it: the
reader is written against the file format, and the arithmetic against the
paper. But every checkpoint made with that code inherits the licence, and
the base model's — so a LittleBit model is a non-commercial model whatever
runs it. `littlebit.rs` says so too.

## For the owner

1. **Is step 5 wanted?** Running other people's LittleBit checkpoints is
   done. Making them is a training project the size of the LoRA work, and
   its output would be non-commercial whatever Kvad's own licence is — the
   method is not, but the only reference implementation to check against is.
2. **Is a CC BY-NC model one Kvad should offer** in its model search, or
   only run when given one? Nothing here searches for them.
3. **Which checkpoint is the benchmark?** The 8B used here is one community
   upload at half an epoch. A model trained to the paper's recipe would say
   much more about the method; making one is step 5.
