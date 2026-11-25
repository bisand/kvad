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

Dense streaming, a layer at a time, would only matter for a dense Nemotron-H
bigger than memory, and NVIDIA has published none (the dense ones are 4–12 B).
It is left out of this plan unless André wants it for its own sake.

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

What is measured goes at the end of this file.
