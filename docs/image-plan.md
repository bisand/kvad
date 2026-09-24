# Plan: text to image

Written 2026-09-23 for [#8](https://github.com/bisand/kvad/issues/8), before
any of its code, from the checkpoints themselves: every shape below was read
off a `config.json` or a safetensors header on this machine, and every
scheduler constant off a `scheduler_config.json`. Where a line explains *why*
a model does something, it was checked against the reference implementation
in `diffusers` (`main`, September 2026), because a config says what the
numbers are and not what is done with them.

[#7](https://github.com/bisand/kvad/issues/7) asked what this repository is
before any of this was built. It is still open as an issue, but the owner asked
for #8 in full, and that settles it for this modality as #7's first option: in
this repo, beside the text path, behind the `gpu` feature the server already
has. `Session` and `Llm` are untouched. Images are a peer of them, not a mode
of them.

## What a text-to-image model is, in this engine's terms

Three models and a loop that is not a token loop.

1. A **text encoder** turns the prompt into conditioning: a matrix, one row
   per prompt token, and sometimes one pooled vector for the whole prompt.
2. A **denoiser** starts from Gaussian noise in a small latent space and is
   run 20–50 times. Each run is a full forward pass that predicts which way
   the latent should move, conditioned on the text and on *how noisy it is
   now*. A **scheduler**, which has no weights, turns that prediction into
   the next latent.
3. A **VAE decoder** turns the final latent into pixels, once.

Nothing is autoregressive. There is no KV cache, no sampler, no stopping
condition. The number of forward passes is fixed before the first one starts,
which is why progress is `step 12 of 30` and not a stream of tokens.

**Guidance** doubles the work. Every step runs the denoiser twice, with the
prompt and without it (or with a negative prompt), and moves further along
the difference: `uncond + g · (cond − uncond)`. With `g = 5` the model is
pushed five times as hard towards "what the prompt changes". It is the single
knob that most changes how much an image looks like its prompt, and it is why
`guidance_scale` is a request parameter with no counterpart in `Sampling`.

## The decision: GPU first, and no CPU path for now

`tensor.rs` has nothing two-dimensional: no convolution, no group norm, no
upsampling. Both VAEs below are convolution stacks from end to end, and the
SDXL denoiser is one in its outer levels. candle has all of those on Metal,
in f16 and bf16 (`conv2d` by `im2col` then matmul, `upsample_nearest2d`,
and `group_norm` written in candle's generic ops). So on the GPU a pipeline
is assembled from pieces that exist, and on the CPU it is a new kernel chapter
before the first image.

So:

- **Every image model is written on candle, in `kvad-gpu`.** This is the first
  part of the engine with no hand-written CPU version, and that is deliberate
  rather than something that happened.
- **What is still written by hand is the model.** Nothing comes from
  `candle-transformers`: CLIP, the UNet, both VAEs, the MMDiT, both schedulers
  and the PNG encoder are in this repository, readable end to end, the way
  `model.rs` is. candle provides arithmetic, not architectures — the same rule
  the text backend follows.
- **A CPU path would be for reading, not for use.** A correct, slow conv2d and
  group norm in `tensor.rs` would let the SD pipeline run with no framework at
  all, and that is worth doing *for that reason* if somebody wants it. It
  would take minutes per image, and nobody should mistake it for a way to
  serve images. It is not planned. A convolution that is obviously faster on
  paper has already been five times slower in practice here once (see the
  autovectoriser notes); a fast one is a project of its own.

## SDXL, as read off `stabilityai/stable-diffusion-xl-base-1.0`

`model_index.json` names the pieces: `CLIPTextModel`,
`CLIPTextModelWithProjection`, `UNet2DConditionModel`, `AutoencoderKL`,
`EulerDiscreteScheduler`. Files used, all fp16:

| Piece | File | Tensors | Params | Bytes |
|---|---|---|---|---|
| CLIP ViT-L text | `text_encoder/model.fp16.safetensors` | 196 | 123 M | 246 MB |
| OpenCLIP bigG text | `text_encoder_2/model.fp16.safetensors` | 517 | 695 M | 1.39 GB |
| UNet | `unet/diffusion_pytorch_model.fp16.safetensors` | 1680 | 2.57 B | 5.14 GB |
| VAE | `madebyollin/sdxl-vae-fp16-fix`, f32 on disk | 248 | 84 M | 335 MB |

About 6.9 GB resident in f16. The VAE is not the one in the base repo:
SDXL's own VAE overflows f16 in its decoder and returns black or NaN images
unless it runs in f32 (its config says `force_upcast: true` for exactly this
reason). The fp16-fix checkpoint is the same decoder retrained to keep its
activations in range, which lets the whole pipeline stay in one dtype.

### Tokenizers

The base repo ships `vocab.json` and `merges.txt` but no `tokenizer.json`,
and CLIP's tokenizer is not plain byte-level BPE (it lowercases, normalises
whitespace and marks word ends with `</w>`). `openai/clip-vit-large-patch14`
ships the same vocabulary as a `tokenizer.json`, so both encoders read that.
The two tokenizers differ only in padding: the first pads with
`<|endoftext|>` (49407), the second with `!` (0). Both wrap the prompt in
`<|startoftext|>` (49406) … `<|endoftext|>` and pad or truncate to 77.

One trap: both text configs say `eos_token_id: 2`. That is wrong — a known
leftover from the conversion — and the pooled output is taken at the
position of the *highest token id*, which is where `<|endoftext|>` is.

### Text encoders

Both are pre-LN transformers with a causal mask, learned position embeddings
(77 of them) and biases everywhere.

| | ViT-L | bigG |
|---|---|---|
| width, heads, layers | 768, 12, 12 | 1280, 20, 32 |
| MLP | 3072, `quick_gelu` (`x·σ(1.702x)`) | 5120, `gelu` |
| what SDXL takes | hidden state *before* the last layer | the same, plus pooled |

SDXL conditions on the **penultimate** layer of both, without the final
layer norm, concatenated along the width: `[77, 768 + 1280] = [77, 2048]`,
which is the UNet's `cross_attention_dim`. The pooled vector comes from bigG
alone: run all 32 layers, apply `final_layer_norm`, take the row at the
end-of-text position, multiply by `text_projection` (1280×1280, no bias).

For an empty negative prompt SDXL does not encode `""`:
`force_zeros_for_empty_prompt: true` means the unconditional context and
pooled vector are zeros.

### UNet

`block_out_channels [320, 640, 1280]`, `layers_per_block 2`,
`transformer_layers_per_block [1, 2, 10]`, `norm_num_groups 32`,
`use_linear_projection: true`, `cross_attention_dim 2048`.

`attention_head_dim [5, 10, 20]` is misnamed in the config: it is the number
of **heads** per level (a diffusers legacy the library itself documents), so
every head is 64 wide: 320/5, 640/10, 1280/20.

```text
latent [4, 128, 128]  ─ conv_in 3×3 ─▶ 320
down 0  DownBlock2D        2 resnets 320,          downsample ─▶ 64×64
down 1  CrossAttnDown      2 × (resnet → transformer ×2)  640, downsample ─▶ 32×32
down 2  CrossAttnDown      2 × (resnet → transformer ×10) 1280
mid     resnet, transformer ×10, resnet             1280
up 0    CrossAttnUp        3 × (concat skip, resnet, transformer ×10) 1280, upsample
up 1    CrossAttnUp        3 × (concat skip, resnet, transformer ×2)   640, upsample
up 2    UpBlock2D          3 × (concat skip, resnet)                   320
out     group norm, silu, conv_out 3×3 ─▶ 4
```

Every level's input and every resnet's output is kept for the way back up,
and each up-resnet concatenates one of them on the channel axis — nine skips
in, nine out. Getting their order wrong still produces an image, only a bad
one, which is the kind of bug this plan exists to avoid.

- **ResnetBlock2D**: GN → silu → conv3×3 → add `Linear(silu(temb))` per
  channel → GN → silu → conv3×3, plus a 1×1 `conv_shortcut` when the widths
  differ.
- **Transformer2D**: GN (eps 1e-6) → `proj_in` linear on `[HW, C]` → N blocks
  → `proj_out` → add the input. A block is LN → self-attention, LN →
  cross-attention to the text (`to_k`, `to_v` are `[C, 2048]`), LN → GEGLU
  (`proj` to 8C, split in two, `a · gelu(b)`, back to C). `to_q/k/v` have no
  bias; `to_out.0` has one.
- **Time**: the step number, as a 320-wide sinusoid (`flip_sin_to_cos`, so
  cos first; `freq_shift 0`), through `linear → silu → linear` to 1280.
- **SDXL's extra conditioning** (`addition_embed_type: text_time`): six
  numbers — original size, crop top-left, target size, here `[H, W, 0, 0, H, W]`
  — each as a 256-wide sinusoid, concatenated with the 1280 pooled vector:
  `6·256 + 1280 = 2816`, which is `projection_class_embeddings_input_dim`.
  `linear → silu → linear` to 1280 and added to the time embedding.

### Scheduler

`EulerDiscreteScheduler`, `prediction_type: epsilon` (the UNet predicts the
noise), `beta_schedule: scaled_linear` from 0.00085 to 0.012 over 1000
steps, `timestep_spacing: leading` with `steps_offset: 1`.

- `βᵢ = (lerp(√0.00085, √0.012, i/999))²`, `ᾱᵢ = Π(1 − β)`,
  `σᵢ = √((1 − ᾱᵢ)/ᾱᵢ)`. σ runs from 0.029 at step 0 to 14.6 at step 999.
- For *n* steps, `leading` picks timesteps `999 … 1` spaced `⌊1000/n⌋` apart,
  counted up from 0 and shifted by the offset, then run from the top: 30
  steps is `958, 925, 892, …, 67, 34, 1`.
  Each timestep's σ is interpolated from the table; a final σ of 0 is appended.
- Start: `x = noise · √(σ_max² + 1)`.
- Each step: the UNet sees `x / √(σ² + 1)` and the timestep; the Euler
  update is `x ← x + ε̂ · (σ_next − σ)`.
- The latent is decoded as `vae(x / 0.13025)` — `scaling_factor` from the VAE
  config — and pixels come back in `[−1, 1]`.

### VAE decoder

`AutoencoderKL`, `latent_channels 4`, `block_out_channels [128, 256, 512, 512]`,
three resnets per up block (`layers_per_block + 1`), GN with 32 groups.

`post_quant_conv` 1×1 → `conv_in` 4→512 → mid (resnet, one single-head
attention over all 128×128 = 16384 positions, resnet) → four up blocks at
512, 512, 256, 128, the first three ending in nearest-2× then conv3×3 →
GN → silu → `conv_out` 128→3. The encoder half of the checkpoint is not
needed to make an image and is skipped by name, so the unread-weights guard
still sees the decoder read in full.

The 1024×1024 decode is the peak of the whole pipeline: 128 channels at full
resolution in f16 is 256 MB per activation, and the mid-block attention is a
16384×16384 score matrix (512 MB in f16) unless it is done in chunks.

## Qwen-Image, as read off `Qwen/Qwen-Image`

`model_index.json`: `Qwen2_5_VLForConditionalGeneration`,
`QwenImageTransformer2DModel`, `AutoencoderKLQwenImage`,
`FlowMatchEulerDiscreteScheduler`. Everything is bf16 on disk.

| Piece | Files | Params | Bytes (bf16) |
|---|---|---|---|
| Qwen2.5-VL-7B | `text_encoder/`, 4 shards, 729 tensors | 8.3 B (7.6 B language) | 16.6 GB |
| MMDiT | `transformer/`, 9 shards, 1933 tensors | 20.4 B | 40.9 GB |
| VAE | `vae/`, 194 tensors | 127 M | 254 MB |

This machine has 48 GiB. The denoiser alone does not fit in bf16 next to
anything else, so Qwen-Image runs quantised or not at all: q8 is about
21.7 GB for the denoiser and 7 GB for the language tower, which with the VAE
and activations is about 30 GB. `quant.rs`'s point about DiT matrices being
matrices is exactly what makes it possible. The model is admitted by the
same accounting as the text models ([#2](https://github.com/bisand/kvad/issues/2)),
and on this machine it will generally be the only thing resident.

### Text encoder

Only the language tower is needed: text-to-image never shows the encoder an
image, so the `visual.*` tree (0.67 B) is skipped, and so is `lm_head`,
because the pipeline wants hidden states, not logits.

The language tower is a Qwen2 decoder — 28 layers, width 3584, 28 query
heads and 4 KV heads, biases on q/k/v, RMSNorm, SwiGLU 18944, rope θ 10⁶ —
with *multimodal* RoPE (`mrope_section [16, 24, 24]`). For text alone the
three position components are all the token index, and M-RoPE reduces to
ordinary RoPE, which is why `llama.rs` claiming `qwen2` is nearly the whole
of it.

The prompt is wrapped in a fixed chat template —

```text
<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

— run through all 28 layers and the final norm, and the first **34** rows,
which are the system prompt, are dropped. What is left, one 3584-wide row per
prompt token, is the conditioning. The repo ships `vocab.json`/`merges.txt`
and no `tokenizer.json`; `Qwen/Qwen2.5-VL-7B-Instruct` has one with the same
vocabulary.

### MMDiT

`num_layers 60`, `num_attention_heads 24`, `attention_head_dim 128` (so
width 3072), `in_channels 64`, `out_channels 16`, `patch_size 2`,
`joint_attention_dim 3584`, `axes_dims_rope [16, 56, 56]`, no guidance
embedding.

- **In**: the 16-channel latent is cut into 2×2 patches, each flattened
  channel-major to 64 numbers (`c·4 + dy·2 + dx`), and `img_in` maps them to
  3072. The text goes through `txt_norm` (RMSNorm) then `txt_in` to 3072.
- **Time**: σ·1000 as a 256-wide sinusoid (cos first), `linear → silu →
  linear` to 3072.
- **Each of the 60 blocks is two streams that meet only in attention.** Image
  and text each have their own `mod` (`silu → Linear(3072, 6·3072)` of the
  time embedding: shift, scale, gate twice), own LayerNorms without weights,
  own q/k/v, own MLP (GELU-tanh, 12288). Per-head RMSNorm on q and k. Text and
  image queries, keys and values are concatenated — text first — and one
  attention runs over both; the result is split back and each stream adds it
  through its own output projection and gate. Then each stream's MLP, gated.
- **Out**: an adaptive LayerNorm from the time embedding (note the order:
  **scale, then shift**, the opposite of the blocks), then `proj_out` to 64,
  unpatchified to 16 channels.

**RoPE here is three-dimensional.** Each 128-wide head is split 16 / 56 / 56
between frame, row and column. Image rows and columns are centred — a 64-patch
side uses positions −32 … 31 — and text tokens are placed on the diagonal
*after* the image: token *i* has position `max(h/2, w/2) + i` on all three
axes. Rotation is on adjacent pairs `(x₂ᵢ, x₂ᵢ₊₁)`, not halves, which is the
opposite of the convention `model.rs` uses for Llama.

### Scheduler

Flow matching, which is simpler than DDPM-style noise: the latent at noise
level σ is `(1 − σ)·image + σ·noise`, and the model predicts the velocity
`noise − image`. Euler: `x ← x + v̂ · (σ_next − σ)`.

- `σ = linspace(1, 1/n, n)`, then shifted towards noise by an amount that
  grows with the image: `μ = lerp(0.5, 0.9)` as the patch count goes from 256
  to 8192, `σ ← eᵘ / (eᵘ + 1/σ − 1)`.
- `shift_terminal 0.02`: the schedule is stretched so its last σ is 0.02
  rather than 1/n, then 0 is appended.
- The denoiser is told the timestep as σ (it multiplies by 1000 itself).
- Default 50 steps. Guidance is "true CFG" at 4.0, and only when a negative
  prompt is given (the reference passes `" "`). The combined prediction is
  rescaled to the conditional one's norm, row by row.

### VAE

A Wan-2.1-style video VAE: `z_dim 16`, `base_dim 96`, `dim_mult [1, 2, 4, 4]`,
causal 3D convolutions, and **RMS norm, not group norm** — each position's
channel vector is L2-normalised and scaled by `√C · γ`. Latents are stored
normalised and are unnormalised per channel with `latents_mean` and
`latents_std` before decoding.

An image is a one-frame video, and that collapses most of the 3D machinery:

- A causal 3D convolution pads *two* zero frames in front, so for a single
  frame only the last temporal slice of each `[out, in, 3, 3, 3]` kernel ever
  touches data. It is a 2D convolution with `weight[:, :, 2]`.
- The temporal upsamplers (`time_conv`) are skipped on the first frame by
  the reference itself. For an image they never run; their weights are
  recorded as deliberately unread.

What is left is a 2D decoder: `conv_in` 16→384, mid (resnet, one single-head
attention at width 384, resnet), up blocks at 384, 384, 192, 96 with three
resnets each and a nearest-2× + conv that *halves* the width after each of
the first three, RMS norm, silu, `conv_out` 96→3, clamp to `[−1, 1]`.

## FLUX.1-schnell, as read off `black-forest-labs/FLUX.1-schnell`

Added 2026-09-24. The repo is gated, and was read after its licence had been
accepted and a token saved on this machine. Before that, this section said
FLUX could not be read, which was true.

`model_index.json`: `CLIPTextModel`, `T5EncoderModel`, `FluxTransformer2DModel`,
`AutoencoderKL`, `FlowMatchEulerDiscreteScheduler`. Files as loaded:

| Piece | Files | Bytes (bf16) |
|---|---|---|
| CLIP ViT-L text | `text_encoder/model.safetensors` | 0.25 GB |
| T5 v1.1 XXL encoder | `text_encoder_2/`, 2 shards, 219 tensors | 9.5 GB |
| MMDiT | `transformer/`, 3 shards, 1156 tensors | 23.8 GB |
| VAE | `vae/diffusion_pytorch_model.safetensors` | 0.17 GB |

The repo's root also holds a 24 GB single-file copy of the transformer, which
is not read.

- **Transformer**: `num_layers 19` double blocks and `num_single_layers 38`
  single blocks, 24 heads of 128 (width 3072), `in_channels 64` (a 16-channel
  latent in 2×2 patches, packed exactly as Qwen-Image packs), `patch_size 1`
  at the transformer, `joint_attention_dim 4096` (T5's width),
  `pooled_projection_dim 768` (CLIP's), `guidance_embeds: false`.
- **The double blocks are Qwen-Image's blocks.** Same modulation (shift,
  scale, gate twice), same per-head RMSNorm on q and k, same joint attention
  with text first, same GELU-tanh MLP; only the weight names differ
  (`norm1.linear` for `img_mod.1`, `ff` for `img_mlp`). So the block lives
  once, in `mmdit.rs`, and both models load it.
- **A single block** runs on text and image as one sequence: one modulation
  (shift, scale, gate), then attention and a 4× MLP *side by side* from the
  same normalised input, concatenated to 5 × 3072 and projected back by one
  `proj_out`, times the gate.
- **Conditioning**: the time embedding (σ·1000, 256-wide, cos first) plus
  CLIP's pooled vector, each through its own two-layer MLP, summed. CLIP's
  pooled vector here is `pooler_output` — the end-of-text row after the final
  norm, with no projection, unlike SDXL's second encoder.
- **RoPE**: axes 16 / 56 / 56 over each head, adjacent pairs, θ = 10000. A
  patch is at `(0, row, col)` counted from the top left; every text token is
  at `(0, 0, 0)`. Nothing is centred, unlike Qwen-Image.
- **T5**: 24 layers, width 4096, 64 heads of 64, gated-GELU MLP of 10240,
  RMSNorm with no bias, no position embeddings but a relative position bias
  (32 buckets, max distance 128) shared by every layer, and no `1/√d` on the
  attention scores. The prompt is padded to 256 tokens with id 0 and the
  padding is **not** masked, because the reference pipeline does not mask it.
- **Scheduler**: flow matching with `use_dynamic_shifting: false` and
  `shift 1.0`, so four steps are σ = 1, 0.75, 0.5, 0.25, then 0.
- **VAE**: `AutoencoderKL` with 16 latent channels, no `post_quant_conv`,
  `scaling_factor 0.3611` and `shift_factor 0.1159`: pixels are
  `decode(latent / 0.3611 + 0.1159)`.
- **Schnell takes no guidance.** It was distilled to make an image in one to
  four steps without it, so a guidance scale or a negative prompt is refused
  rather than ignored. FLUX.1-dev, which takes guidance as an input to the time
  embedding, is refused by name until it has an implementation and a test.

At q8 the transformer is about 12.6 GB and T5 about 5 GB.

## Where the code goes

`kvad` (the CPU crate) gets the parts that are not arithmetic on a GPU, so
that the service layer can name them without depending on candle:

- `kvad::image` — `ImageRequest` (prompt, negative prompt, width, height,
  steps, guidance, seed; a second parameter type, not an extension of
  `Sampling`), `Image` (RGB8), the `Painter` trait a pipeline implements,
  `Step` progress, and a PNG encoder written out by hand (stored deflate
  blocks, CRC-32, Adler-32 — about a hundred lines and no dependency).

`kvad-gpu` gets the models, under `src/image/`:

| File | What |
|---|---|
| `nn.rs` | the 2D vocabulary: conv, group norm, resnet, attention in chunks |
| `clip.rs` | CLIP text encoders |
| `unet.rs` | SDXL's UNet |
| `vae.rs` | `AutoencoderKL` decoder |
| `schedule.rs` | Euler (ε) and flow-match Euler (v) |
| `sdxl.rs` | the SDXL pipeline |
| `qwen.rs` | Qwen-Image: text tower, MMDiT, Wan decoder, pipeline |
| `mmdit.rs` | the double-stream block Qwen-Image and FLUX share, and FLUX's single-stream block |
| `t5.rs` | T5's encoder |
| `flux.rs` | FLUX.1-schnell |

`examples/sdxl.rs` is the milestone: prompt in, PNG on disk, no server.

## The service layer

- `Cmd::Paint(ImageRequest)`; `Evt::Step { done, total }` and
  `Evt::Painted(Image)`. The engine thread holds either an `Llm` or a
  `Painter`, never both; a `Cmd::Chat` to a painter is an error that says so,
  and the other way round.
- The scheduler routes by what is loaded, and residents say what they are:
  `kind: "chat" | "image"`. `/v1/models` carries it, so a client can tell
  before it asks.
- `POST /v1/images/generations`, OpenAI's shape: `model`, `prompt`, `n`,
  `size` (`"1024x1024"`), `response_format` (`b64_json` or `url`), plus
  kvad's own `negative_prompt`, `steps`, `guidance_scale`, `seed`. With
  `stream: true` it answers in server-sent events, one per step, then the
  image.
- **Where images live**: decided rather than defaulted. PNGs go to
  `data_dir()/images/<id>.png`, a row per image in an `images` table (owner,
  model, prompt, parameters, seed, size, bytes). They are the user's work, like
  a trained model, so `data`, not cache. A `url` response points at
  `/api/images/<id>.png`, served to the owner only. Nothing expires them;
  the gallery deletes them.
- Generation goes through the scheduler like a chat, because the scheduler
  owns the GPU. A generation holds the queue for its whole length, so a chat
  sent meanwhile waits for it. That is the honest cost of one GPU.

## The order it is built in

1. This document.
2. SDXL on Metal as `examples/sdxl.rs`. **The milestone is one image.**
3. The service shape: `Cmd`/`Evt`, the scheduler, `/v1/images/generations`,
   `/v1/models` kinds.
4. The UI page: prompt, parameters, per-step progress, a gallery.
5. Qwen-Image, quantised, with its memory accounted for.

Step 5 moved after the UI against #8's order: the UI is testable against
SDXL in seconds per image, and Qwen-Image is minutes per image on this machine.
Proving the plumbing on the slow model is how a week goes to finding a
transposed tensor.

## What was built, and what it measured

Added 2026-09-23, after the fact. Everything above is the plan as written
before the code. Where the code went differently, it says so here.

- **All six steps are done.** SDXL on Metal (`examples/sdxl.rs`, which also
  takes `--repo Qwen/Qwen-Image --quant q8`), `Cmd::Paint` and
  `Evt::Painting`/`Evt::Painted`, `/v1/images/generations` with storage and a
  gallery (`crates/serve/src/images.rs`, migration 008), the Images page, and
  `kvad images make|ls|rm` — the CLI reaches every route, as a test requires.
- **SDXL's first image was right.** 1024², 30 steps: 3.93 s a step with
  guidance, 9.4 s to decode, on an M5 Pro in f16 (single runs).
- **Qwen-Image's first image was noise**, and the cause was in candle, not
  here. Its Metal quantised matrix-matrix kernel ignores the input's start
  offset, and the image half of the joint attention is a `narrow` after the
  text. The text tower was cleared first by pointing the checkpoint's
  `lm_head` at it ("The capital of France is" → " Paris"). The denoiser was
  then caught by one number: the neighbour correlation of its predicted clean
  latent from pure noise, 0.65 before and 0.996 after the fix in
  `common::Proj::forward`. It now draws legible lettering at 1024²: 28.3 s a
  step with true CFG, 5.8 s to decode, at q8.
- **Memory is as estimated.** q8 denoiser 21.7 GB, language tower 7.5 GB (both
  cached after the first load). The first run's peak footprint was 32.7 GB,
  quantising as it loaded.
- **Previews** are the latent mixed to RGB by a fixed matrix per model, fitted
  by least squares against one decoded image each. They explain 76–83% of the
  colour variance for SDXL's four channels and 96–98% for Qwen-Image's
  sixteen.
- **FLUX.1-schnell** (added 2026-09-24) was right the first time: 1024², 4
  steps, 13.1 s a step, 9.6 s to decode, 18.3 GB at q8 (12.6 GB transformer,
  5.1 GB T5, both cached). Its preview fit explains 97–99% of the colour
  variance. It takes no guidance, and a request that asks for guidance or a
  negative prompt is refused with a 400 before it is queued (`Defaults::
  takes_guidance`), rather than failing after the model has run.
- **Not done:** FLUX.1-dev (see the FLUX section), a CPU path (decided against, as above),
  and any speed work. Qwen-Image at 1024² is about ten minutes an image at
  20 steps. Nothing has been profiled yet, and that is where to start.
