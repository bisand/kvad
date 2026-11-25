# Plan: Nemotron-H, and Munin 2.0 on a 48 GB Mac

Written 2026-11-25. The reference is NVIDIA's `modeling_nemotron_h.py` as
shipped beside `danish-foundation-models/munin-2.0` (the `trust_remote_code`
file), and `transformers`' own `models/nemotron_h` (main, November 2026). They
agree on everything below except one line, noted under *Mamba-2*.

## Munin is not dense

The starting assumption was that Munin 2.0 is a dense 124B model, so a token
needs every weight and streaming it from the SSD costs ~30 s a token in bf16.
Its `config.json` and model card say otherwise. It is NVIDIA's Nemotron 3 Super,
post-trained: **120B parameters, 12B active**, 88 layers of three kinds:

| | layers | per layer | all of them |
|---|---|---|---|
| `M` Mamba-2 | 40 | 110 M | 4.4 B |
| `*` attention (GQA, 32 q / 2 kv heads, no RoPE) | 8 | 36 M | 0.3 B |
| `E` latent mixture: 512 experts, 22 a token, one shared | 40 | 2.87 B | 115 B |
| embeddings + head | | | 1.1 B |

Of a mixture layer's 2.87 B, the routed experts are 2.82 B (512 × 5.5 M) and
everything else (router, two latent projections, the shared expert) is 55 M.
So the weights every token needs are about **8 B** (Mamba, attention, the
mixtures' fixed parts, head), and the experts a token needs are 22 × 40 × 5.5 M
= **4.8 B** of the 113 B.

| | always needed | experts a token reads | all experts |
|---|---|---|---|
| bf16 | 16 GB | 9.7 GB | 226 GB |
| q8 | 8.5 GB | 5.1 GB | 120 GB |
| q4 | 4.5 GB | 2.7 GB | 64 GB |

That is the shape the existing expert path was built for (#295, #297): keep
the fixed part resident and read the experts a token picks from the
quantised cache on the SSD, through `experts::Resident`. At q4 the floor with
every expert a miss is 2.7 GB at ~6–13 GB/s, **0.2–0.45 s a token**, and the
expert cache's hits only lower it. Not 10–30 s.

Dense streaming, a layer at a time, only matters for a dense model bigger
than memory, and NVIDIA has published no such Nemotron-H (the dense ones are
4–12 B). André wanted it built anyway: `crates/llm/src/layers.rs`, measured on
the dense 4B below.

## The layer pattern

`hybrid_override_pattern` has one character a layer: `M` Mamba-2, `*`
attention, `-` a dense MLP, `E` a mixture. **Every layer is one mixer**, with
one RMSNorm in front and a residual round it:

```text
x  <-  x + mixer(RMSNorm(x))
```

There is no separate feed-forward half; an MLP is a layer of its own. Then
`norm_f` and `lm_head` (untied).

Tensor names: `backbone.embeddings`, `backbone.layers.{i}.norm`,
`backbone.layers.{i}.mixer.*`, `backbone.norm_f`, `lm_head`. The checkpoint
also carries `mtp.*`, a multi-token-prediction head (one attention and one
mixture layer). It is skipped by name, as Qwen3-Next's is.

### Mamba-2

Shapes, with Munin's numbers: `H` = `mamba_num_heads` (128) heads of `P` =
`mamba_head_dim` (64); `I = H·P` (8192); `G` = `n_groups` (8) groups of state
`N` = `ssm_state_size` (128); `conv_dim = I + 2·G·N` (10240).

```text
in_proj: hidden -> [ z (I) | xBC (conv_dim) | dt (H) ]
xBC    <- silu(causal depthwise conv1d(xBC), kernel 4, with bias)
x, B, C = split(xBC, [I, G·N, G·N])           head h reads group h / (H/G)
dt     <- softplus(dt + dt_bias), clamped to time_step_limit (default (0, ∞))
A       = -exp(A_log)
per head h, per p in 0..P, per n in 0..N:
    S[h,p,n] <- S[h,p,n] · exp(dt[h]·A[h]) + dt[h] · B[g,n] · x[h,p]
    y[h,p]    = Σ_n S[h,p,n] · C[g,n]  +  D[h] · x[h,p]
y       <- RMSNorm_grouped(y · silu(z)) · norm.weight   groups of I/G, gate *before* the norm
out_proj: I -> hidden
```

State a layer keeps: the convolution's last 3 inputs, `3·conv_dim` floats,
and `S`, `H·P·N` floats: 1.08 M floats, 4.3 MB in f32. Forty layers of it are
172 MB whatever the context length. The reference keeps `S` in f32
(`mamba_ssm_cache_dtype`) and so does this.

The one disagreement: the remote code's *pure-PyTorch* fallback clamps `dt`
below at `time_step_min` (0.001); its CUDA path, and `transformers`, clamp to
`time_step_limit`, which defaults to (0, ∞) and so does nothing after a
softplus. The CUDA path is what NVIDIA trained and evaluated with, so this
follows it.

### Attention

Plain grouped-query attention, `head_dim` 128, scale `1/√head_dim`, no bias,
**no position encoding at all**: the Mamba layers' convolution and recurrence
are what carry order. The KV cache holds the 8 attention layers only: 2 heads ×
128 × 2 × 4 bytes = 2 KB a position a layer in f32, 16 KB a position for the
model, 2.1 GB at the full 131 072.

### MLP and mixture

An MLP is `down(relu(up(x))²)`: no gate, so not this engine's SwiGLU `Ffn`.
Routed and shared experts are the same shape.

A mixture layer:

```text
scores  = sigmoid(x · Wrouterᵀ)                          in f32
choice  = top 22 of (scores + e_score_correction_bias)   n_group 1: no group mask
w       = scores[choice] / Σ scores[choice] · 5.0         normalised, then scaled
latent  = fc1_latent_proj(x)                              4096 -> 1024
routed  = fc2_latent_proj( Σ w_e · expert_e(latent) )     experts run in 1024, back to 4096
out     = routed + shared_experts(x)                      the shared expert reads x, not latent
```

The routing rule is DeepSeek-V3's (`noaux_tc`: sigmoid, a bias that picks but
does not weigh, scale after normalising), which `ffn::Router` already has; the
config just does not name it, so `configure` does. The latent projection is
new: the routed sum has to go through `fc2` before the shared expert's output
is added, where `Moe::run` adds both into one accumulator.

Nemotron-3-Nano-30B-A3B is the same mixture without the latent projection
(`moe_latent_size` null), 128 experts, 6 a token, scale 2.5.

## Steps

1. **`nemotron_h` on the CPU** (`crates/llm/src/model/nemotron_h.rs`).
   `Ffn` learns an ungated ReLU² form; the mixture reuses `ffn::Router`,
   `Moe::routed` and `Moe::with_experts` (so the expert cache comes for free)
   and adds the latent projections around them. `configure` turns the pattern
   into a per-layer cache layout: a recurrent state for `M`, keys and values
   for `*`, nothing for `-` and `E`. One token at a time, as Qwen3.5's delta
   net is: a chunked prefill is its own piece of work.
2. **Checked against `transformers`** on a tiny random-weight model:
   `scripts/nemotron-h-fixture.py` builds it with `NemotronHForCausalLM`
   (one layer of each kind, the latent mixture included), runs it in f32, and
   writes weights, config and logits to `crates/llm/tests/nemotron_h/`. That
   fixture is small and committed, so `cargo test` checks the engine against
   the reference with no Python and no download.
3. **The expert cache reads this naming**: `experts.rs` finds experts under
   `mixer.experts` as well as `mlp.experts`, with or without a gate.
4. **Measure** on Nemotron-H-4B-Instruct (9 GB, dense: Mamba, attention, MLP)
   for sense and speed, then on Nemotron-3-Nano-30B-A3B (63 GB bf16, the
   mixture) with the expert cache. Munin itself (230 GB bf16; ~64 GB at q4)
   only after asking: it needs the download, and a q4 cache written from it.

## Measured

M5 Pro, 48 GB, CPU engine, 2026-11-25. The machine's load average sat between
7 and 22 through this (another session was building caches), so absolute
timings carry ±25%; ratios and byte counts do not move.

**Correctness.** The tiny fixture agrees with `transformers` 5.24.1 at every
position to 1e-4 of the logits' scale, whole prompt or a token at a time or in
two pieces (`tests/nemotron_h.rs`). On the real Nemotron-H-4B-Instruct, the
engine at f32 and `transformers` at f32 produce the same 30 greedy tokens for
a Danish prompt. Its Danish is poor either way ("for to grundlæggende
grundlægger"); that is the 4B model, not the engine. Nemotron-3-Nano-30B-A3B
at q4 reasons in English and answers "Hovedstaden i Danmark er København."

**Two loading traps, both in the published files.** Nemotron-H's configs hold
`"time_step_limit": [0.0, Infinity]`, which is Python's JSON and not JSON:
`read_json` now reads bare `Infinity` and `NaN` as `null`, absent. The 4B's
config spells `head_dim` as `attention_head_dim`, and the fallback, hidden over
heads, is 96 where the heads are 128. And Nemotron 3's chat template defines
macros, which minijinja renders only with its `macros` feature.

**The first load held the whole quantised model.** It quantised every weight
into memory and wrote the cache as it went: 21.9 GB peak footprint for Nano at
q4, and Munin would want 64 GB of 48. It now writes each weight, keeps none,
and maps the file it wrote: **1.84 GB peak**, same answer, same tokens. Writing
Nano's 19.7 GB q4 cache from 59 GB of bf16 takes 220 s.

**Nano decode, q4, 150 greedy tokens.** Warm (the cache file in the page
cache), against cold (evicted with `msync(MS_INVALIDATE)` before every run),
three rounds each, rotating order:

| experts from | hits | read a token | warm tok/s | cold tok/s | footprint |
|---|---|---|---|---|---|
| the mapping | | | 20.6 20.4 20.9 | 9.6 15.8 10.0 | 0.25 GB |
| a 2 GB expert cache | 46.5% | 0.40 GB | 18.0 9.1 16.9 | 9.4 11.1 10.8 | 2.2 GB |
| an 8 GB expert cache | 77.6% | 0.17 GB | 19.3 17.6 19.1 | 12.5 15.4 14.4 | 8.3 GB |

Cold at 2 GB, about half of a token's 100 ms is reading: 0.4 GB in ~50 ms is
~8 GB/s while it reads, against 13.8 GB/s the SSD gives a sequential `dd`.

**Reading the next mixture's experts ahead** (`KVAD_EXPERT_GUESS=1`, #295),
cold, 2 GB cache, four pairs alternating order: hits rise from 46.5% to 71.3%,
and bytes read from 75 to 102 GB, because the guess is 68% right (the next
mixture is two layers up, past a Mamba layer). Decode: 11.3 10.5 9.1 10.0
tok/s with it, 11.6 10.1 7.1 10.2 without. It does not pay here either.

**CPU or Metal.** The CPU. Metal has no Nemotron-H, and with the experts on
the SSD half of a token is reading, which a GPU does not shorten; the other
half is the CPU's arithmetic, so a port is worth at most 2× there. Streaming a
dense model, the CPU already reads at 93% of the disk's speed, so a GPU would
gain nothing. It is worth measuring Munin first.

**Dense layers from the SSD** (`crates/llm/src/layers.rs`). Each layer is one
run of the q-cache file, read with the expert cache's uncached `Fetcher` in
8 MB requests, four in flight, into a pool of slabs, and the layer's weights
pointed at its slab with `Weight::rebased`. One thread reads the next
`KVAD_LAYER_AHEAD` layers (default 2), wrapping to the first for the next
token. Mixture layers are left to the expert cache. It is on when the cache
file is larger than usable memory, or with `KVAD_LAYER_STREAM=1`. The answer
is bit for bit the mapping's (`tests/nemotron_h.rs`).

Nemotron-H-4B at q8, forced on, cold, 40 greedy tokens, three rounds:

| | decode tok/s | footprint |
|---|---|---|
| mapped (fits, so warm after the first token) | 21.5 22.6 21.4 | 0.24 GB |
| streamed, nothing ahead | 2.6 2.1 2.6 | 0.34 GB |
| streamed, 1 layer ahead | 3.1 2.9 2.5 | 0.42 GB |
| streamed, 2 ahead | 3.1 3.0 2.9 | 0.51 GB |
| streamed, 4 ahead | 3.1 2.7 2.6 | 0.68 GB |

A token reads 4.25 GB (every layer; the embeddings and head stay mapped), so
3.0 tok/s is 12.8 GB/s, against 13.8 GB/s from a sequential `dd`: the floor
is 3.25 tok/s and this is within 10% of it. Reading ahead is worth ~20%, one
layer gets most of it, and more only costs memory. A model that fits is seven
times faster mapped, which is why this is off until the file does not fit.

### What that projects for Munin (not measured)

A token reads 22 experts in each of 40 layers, 3.1 MB each at q4: **2.7 GB**
with no hits. At the 8–14 GB/s above that is a floor of **0.2–0.35 s a token**.
The default expert cache is half of usable memory, 19.3 GB here, so 31% of
Munin's 61 GB of experts; on Nano 12% of them gave 46.5% hits and 45% gave
78%, so call it ~65%: ~1 GB a token, ~0.1 s. Arithmetic scales with active
parameters, 12.7 B against Nano's ~3.5 B, so ~3.6 × Nano's 50 ms. Both
together: **~0.3 s a token, ~3 tok/s**, in a footprint of about 20 GB: the
expert cache's slabs, plus 172 MB of state. The 4.5 GB every token needs is
mapped, and is file pages rather than footprint.
To measure it needs 230 GB of bf16 and a 64 GB q4 cache beside it, 294 GB;
the disk had 222 GB free.
