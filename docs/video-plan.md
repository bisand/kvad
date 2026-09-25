# Plan: text to video, starting with LTX-2.5

Written 2026-09-24 for [#51](https://github.com/bisand/kvad/issues/51),
before any of its code. Every shape below was read from a safetensors header
of `Lightricks/LTX-2.5`, fetched with range requests after the licence had
been accepted on this machine. Every config value was read from the
`__metadata__` those headers carry. Where a line explains *why* the model does
something, it was checked against the reference implementation,
`Lightricks/LTX-2` at `a95ab856` (2026-08-26), and against Hugging Face
`transformers` for the Gemma tower, which LTX imports rather than defines.

Nothing here has been run. torch is not installed on this machine, and no
tensor below has been compared with the reference numerically. A few weight
tensors were read, each to settle one question:
- the keyframe embedding (see The DiT);
- the latent statistics of both video VAE files and of the audio VAE (see
  Decoding).

"How it will be tested" says what to prove first and how.

This is the third modality to follow the shape settled in
[#7](https://github.com/bisand/kvad/issues/7) and built in #8: a peer trait, a
`Model` variant, a `kind`, and an OpenAI-shaped endpoint. The decision in
`docs/image-plan.md` to go GPU first with no CPU path applies here unchanged,
for the same reasons and a few more: three-dimensional convolutions, and a
denoiser of 21B parameters.

## What #51 got wrong

The issue was written from community copies before the gated repo could be
read. These points are corrected here and should be corrected there.

- **The text-encoder file does not hold the connectors.** It holds Gemma and
  the two *aggregate projections* (188160 → 4096 and → 2048, 1.16B
  parameters). The two 8-layer connectors (2.02B parameters) are in the
  **DiT** file.
- **The tokenizer is inside the text-encoder file.** It is a `U8` tensor
  named `tokenizer_json`, 32 MB. So `Lightricks/LTX-2.5-Diffusers`, which is
  gated separately and not accepted on this machine, is not needed for
  anything.
- **Generation is two stages, not one.** The reference `DistilledPipeline` has
  no one-stage path. It denoises at half the width and height, upsamples the
  latent ×2 with a learned upsampler, then runs 3 more steps at full size.
  So the upsampler is part of the pipeline, not a later extra.
- **Stage 1 on 2.5 samples ancestrally.** It is not deterministic Euler.
  Fresh noise is added at every step from a second RNG.
- **The audio rate is 48 kHz.** The BWE vocoder config says
  `output_sampling_rate: 48000`, which settles the 24-vs-48 question.
- **The distilled pipeline predicts the frame count by default.** When no
  frame count is given, a 2M-parameter duration head chooses one from the
  prompt, between 1 and 20 s.
- **Width and height must be multiples of 64 for the two-stage pipeline**, as
  the code asserts. The model card says 32, which holds only for one stage.

## The files

The model card calls `Lightricks/LTX-2.5` a "split, Comfy-aligned pack": one
safetensors file per component, and no `model_index.json`. The configs are in
each file's metadata under `config`. The text encoder's is under
`gemma_config`, and the DiT carries `model_version: 2.5.0` and
`gemma_version: gemma4-12b-ltx-v1`, which the reference checks against the
encoder.

What text-to-video reads:

| Piece | File | Params | bf16 |
|---|---|---|---|
| DiT, distilled, with both connectors | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 21.00 B | 42.0 GB |
| Gemma 4 12B + aggregate projections + tokenizer | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 13.15 B | 26.3 GB |
| Spatial latent upsampler ×2 | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 0.50 B | 1.0 GB |
| Video VAE, conv decoder | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 0.72 B | 1.45 GB |
| Audio VAE + vocoder + BWE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | 0.18 B | 0.37 GB |
| Duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 1.9 M | 4 MB |

That is about 71 GB to download.

The repo also holds files that text-to-video does not read:
- the dev DiT, 42.0 GB, for later;
- three Comfy-only int8 and NVFP4 files. The model card says these are not
  for PyTorch, and they are not for us;
- the DiffVAE decoder (`vae/ltx-2.5-video-vae-bf16`, 1.47 GB), for later;
- the temporal upsampler, 0.26 GB;
- the distilled LoRA, 8.9 GB, used only by the dev model's pipelines.

**kvad must fetch exactly the files it needs.** A whole-repo download here is
227 GB. [#20](https://github.com/bisand/kvad/issues/20) is the same mistake on
a smaller scale.

Parts of the downloaded files are skipped by name:
- the encoder halves of both VAEs, 0.63 GB in the conv VAE file;
- in the text-encoder file: `vision_model.*`, `multi_modal_projector.*`,
  `audio_projector.*`, and the embedded chat and processor configs.

The unread-weights guard should see these listed as deliberately unread,
the way `qwen.rs` lists `time_conv`.

**Recognising the repo.** `kvad-gpu/src/image/mod.rs` recognises a pipeline
from `model_index.json`, and this repo has none. The recognition should rest
on the DiT's own metadata: a `diffusion_models/*.safetensors` whose `config`
names `AVTransformer3DModel` and whose `model_version` is `2.5.x`.

## What a text-to-video model is, in this engine's terms

It is `docs/image-plan.md`'s three models, plus sound, plus a second pass:

```text
prompt ─ Gemma 4 (48 layers, all 49 hidden states kept)
       ─ aggregate projection ─▶ 4096 (video) and 2048 (audio)
       ─ two 8-layer connectors ─▶ video context [1024, 4096], audio context [1024, 2048]
       ─ [duration head ─▶ frame count, if none was asked for]
stage 1, W/2 × H/2: video latent + audio latent from noise, 8 steps, one DiT call each
upsampler ×2 on the video latent
stage 2, W × H: both latents re-noised to σ = 0.909375, 3 steps
video VAE ─▶ frames;  audio VAE ─▶ mel ─▶ vocoder ─▶ BWE ─▶ 48 kHz stereo
```

**Video and sound are one model, not two.** Every DiT block carries a video
stream and an audio stream, and each reads the other in every block. There is
no video-only switch in the reference. Dropping the audio stream would change
the video, so the audio stream always runs. Whether to *decode* it is a
request option.

The number of DiT calls is fixed before the first one, as for images: 8 plus
3, with no guidance on the distilled model. Progress reports steps, as for
images, but the steps of the two stages differ in cost by about 5×. See the
cost section.

## Text: Gemma 4 and the connectors

### Tokens

- **No chat template and no system prompt.** Encoding strips the prompt,
  tokenises it with the embedded `tokenizer.json`, and prepends `<bos>` (2)
  by hand, because the tokenizer's post-processor adds nothing. There is no
  EOS.
- The sequence is truncated to 1024 and **left**-padded with `<pad>` (0) to
  exactly 1024. The mask is 1 for BOS and the prompt.
- In Rust this is the `tokenizers` crate with `add_special_tokens = false`,
  then the BOS. That BPE has not yet been checked against the Python one.
- `<|image|>`, `<|audio|>` and `<|video|>` written literally in a prompt are
  swapped for pad before embedding.

### The tower

The config is `gemma4_unified_text`:
- 48 layers, width 3840;
- 16 query heads;
- an MLP of 15360, gated GELU-tanh;
- RMS eps 1e-6.

kvad has no Gemma of any version. This tower is new code, and it has more
traps than any tower here so far.

- **Embedding** × bf16(√3840), which is exactly **62.0**.
- **RMSNorm multiplies by `w`, not `1 + w`**, and is computed in f32. The
  checkpoint confirms this: `q_norm` is a constant 1.0234, and `model.norm`
  runs up to 600.
- **Sandwich norms.** Each layer normalises the input before attention and
  the output after it, then does the same around the MLP:
  `h += post_attn_ln(attn(input_ln(h)))`,
  `h += post_ff_ln(mlp(pre_ff_ln(h)))`.
- **`layer_scalar` scales the whole residual stream at the end of every
  layer.** The values run from 0.0045 (layer 11) to 0.92. Leaving it out
  would still produce numbers, just wrong ones.
- **Two kinds of layer.** Layers 5, 11, 17, 23, 29, 35, 41 and 47 are global;
  the other 40 are sliding (window 1024, which at 1024 tokens is plain
  causal). A dump that shows only layer 0's shapes hides the difference:

  | | sliding (40) | global (8) |
  |---|---|---|
  | q_proj | [4096, 3840]: 16 × 256 | [8192, 3840]: 16 × 512 |
  | k_proj | [2048, 3840]: 8 × 256 | [512, 3840]: **1** × 512 |
  | v_proj | [2048, 3840] | **none** |
  | o_proj | [3840, 4096] | [3840, 8192] |

- **`attention_k_eq_v`.** In the global layers, V is the raw `k_proj`
  output, taken *before* `k_norm` and RoPE. Every layer then applies a
  weightless RMS `v_norm` to V.
- **Attention scale is 1.0.** The temperature is in the constant `q_norm` and
  `k_norm` weights. There is no softcapping; the config's 30 applies to
  logits, which are never computed here.
- **Causal plus padding mask.** The `bidirectional: "vision"` setting does
  nothing for text.
- **RoPE is rotate-half**, which is the convention `model.rs` already uses:
  - Sliding layers rotate all 256 dimensions with θ = 10⁴.
  - Global layers use "proportional" RoPE with θ = 10⁶. There are 64
    frequencies `1/10⁶^(2i/512)`, and the remaining 192 are zero. So
    dimensions 0–63 pair with 256–319 and rotate, and every other dimension
    passes through unchanged.
- Positions are `0..1024`, **counting the left padding**. The prompt's tokens
  sit at `1024 − n … 1023`. Running only the *n* real tokens is exact in
  exact arithmetic, because attention is causal and pad-masked and RoPE is
  relative. Keep those positions anyway, so that bf16 rounding matches the
  reference.
- **It returns 49 hidden states:** the scaled embeddings, the outputs of
  layers 0–46, and `model.norm` applied to layer 47's output. There is no LM
  head. The embeddings are tied, and nothing here needs logits.

### Aggregate projection

Stack the 49 states as `[1024, 3840, 49]`. RMS-normalise each (token, layer)
over the 3840 axis, with no weight. Flatten **dimension-major, layer-minor**:
column `d·49 + l`, which is not one layer after another. Zero the pad rows.
Then:

- video: `x · √(4096/3840)` through `video_aggregate_embed`
  ([4096, 188160] + bias);
- audio: `x · √(2048/3840)` through `audio_aggregate_embed`
  ([2048, 188160] + bias).

K = 188160 needs f32 accumulation. These two matrices are 1.16B parameters
for one matmul each per prompt.

### Connectors

These are the `video_embeddings_connector` and `audio_embeddings_connector`
in the DiT file:

| | video | audio |
|---|---|---|
| Heads | 32 × 128 | 32 × 64 |
| Blocks | 8 | 8 |
| FFN | 4096 → 16384 → 4096, with bias | 2048 → 8192 → 2048, with bias |
| Registers | [128, 4096] | [128, 2048] |

1. Move the real rows to the front. It is a stable sort by mask, and audio
   uses the same order.
2. Replace every pad row *p* with `registers[p % 128]`. Then **clear the
   mask**: from here on everything attends to everything.
3. Each block: weightless RMS norm → gated self-attention → residual;
   weightless RMS norm → GELU-tanh FFN → residual.
   - `q_norm` and `k_norm` are weighted RMS norms over the **full width**,
     all heads at once, not per head.
   - RoPE is "split" with θ = 10⁴ and max position 4096. The frequencies are
     `10⁴^(k/(F−1)) · π/2`, built in f64 and cast to f32. The angle is
     `f · (2p/4096 − 1)`, and head *h* takes its own slice of the frequency
     vector.
   - The gate is `2 · sigmoid(to_gate_logits(normed input))`, one per head,
     applied before `to_out`.
4. A final weightless RMS norm. The config keys `connector_norm_output`,
   `connector_learnable_registers_std` and `text_encoder_norm_type` are never
   read by the reference.

The result is `[1024, 4096]` and `[1024, 2048]`, handed to the DiT with no
mask. The duration head reads the same two tensors.

## The DiT

The DiT is `AVTransformer3DModel` with 48 blocks:

| | Parameters |
|---|---|
| The 48 blocks (386.7 M each) | 18.56 B |
| Modulation and projections outside the blocks | 0.43 B |
| The two connectors | 2.02 B |

### Tokens

- **Video:** latent `[128, F, H, W]` with F = (frames − 1)/8 + 1, H = h/32,
  W = w/32. Patch size is 1, so tokens are `(f, h, w)` in that order, 128
  wide, and `patchify_proj` is 128 → 4096. A 768×512 frame is 24 × 16 = 384
  tokens. 121 frames is 16 latent frames, 6144 tokens.
- **Audio:** latent `[8, T, 16]` (channels, time, mel bins), tokens
  `b t (c f)`, 128 wide, and `audio_patchify_proj` is 128 → 2048.
  T = round(frames / fps × 25), which is 126 for 121 frames at 24 fps.
- **Keyframe embedding.** A learned `[1, 4096]` vector is added to the 384
  tokens of latent frame 0 in **every** generation, text-to-video included.
  The weight was read to check it is not zero: its RMS is 0.0008, small but
  real.

### Positions

RoPE is the part to get exactly right before anything else.

- **Video positions are in seconds and pixels, not latent indices.** Each
  latent cell covers a range, and the position is its midpoint:
  - Time: 0.5/fps for frame 0, and (8f − 3)/fps after it. That follows from
    frame 0 being one pixel frame and each later latent frame covering 8.
  - Rows and columns: 32h + 16 and 32w + 16.
- **Audio positions** are seconds too: 0.005 for latent 0, and
  (4i − 1) · 0.01 after it.
- **The rotation spans the whole attention width, not each head.**
  1. Build one frequency vector for the whole 4096: `ω_k =
     (π/2)·10⁴^(k/(N−1))`, in f64. For video self-attention, N = 682
     frequencies per axis.
  2. Interleave the axes, t h w t h w …
  3. Pad **two** unrotated entries at the front, to 2048.
  4. Cut into 32 heads of 64, so each head has its own slice of frequencies.
  5. Within a head, rotate-half pairs (j, j + 64).

  The angle is `ω_k · (2p/max_pos − 1)`, with max_pos (20, 2048, 2048) for
  (t, h, w).
- Audio self-attention, and both audio↔video attentions, use 1D tables over
  time alone, with max_pos 20. Text cross-attention has no RoPE.
- **Angles reach about 15,700 rad.** The tables must be computed on the CPU,
  in f64 or f32, and uploaded. candle's Metal backend has no f64, and bf16 at
  that magnitude has a resolution of 64 rad. The reference computes in f32
  and stores cos and sin as bf16. They depend only on shape and fps, so build
  them once per generation.

### Conditioning

- The timestep is σ · 1000 as a 256-wide sinusoid, cos first, then
  `linear → silu → linear`.
- Eight "adaLN single" modules turn that into modulation rows:

  | Module | Rows × width | Driven by |
  |---|---|---|
  | `adaln_single` | 9 × 4096 | video σ |
  | `audio_adaln_single` | 9 × 2048 | audio σ |
  | `prompt_adaln_single` | 2 × 4096 | video σ |
  | `audio_prompt_adaln_single` | 2 × 2048 | audio σ |
  | `av_ca_video_scale_shift_adaln_single` | 4 × 4096 | video σ |
  | `av_ca_audio_scale_shift_adaln_single` | 4 × 2048 | audio σ |
  | `av_ca_a2v_gate_adaln_single` | 1 × 4096 | **audio** σ |
  | `av_ca_v2a_gate_adaln_single` | 1 × 2048 | **video** σ |

  Each gate is driven by the *other* stream's σ. It is multiplied by the
  config's `av_ca_timestep_scale_multiplier` of 1000, not the code default
  of 1.
- **Each block adds its own f32 table to these rows**, cast to bf16:
  - `scale_shift_table [9, W]`:

    | Rows | Meaning |
    |---|---|
    | 0, 1, 2 | shift, scale, gate for self-attention |
    | 3, 4, 5 | shift, scale, gate for the FFN |
    | 6, 7 | shift and scale for the text cross-attention query |
    | 8 | gate for the text cross-attention output |

  - `prompt_scale_shift_table [2, W]`: shift and scale for the text K and V.
  - `scale_shift_table_a2v_ca_{video,audio} [5, W]`. **These are (scale,
    shift)**, the opposite order to the main table:

    | Rows | Meaning |
    |---|---|
    | 0, 1 | scale and shift for a2v |
    | 2, 3 | scale and shift for v2a |
    | 4 | the gate |
- **The reference evaluates σ per token**, as `mask · σ`, so that conditioning
  frames can sit at σ = 0. In text-to-video every token has the same σ.
  Compute each module once and broadcast. The reference materialises it per
  token, at about 3 TFLOP a forward, and that cost can be skipped.

### A block

Notation: `rms` is weightless RMSNorm with eps 1e-6, reduced in f32;
`ada(x, scale, shift) = rms(x)·(1 + scale) + shift`.

Every attention works the same way:
- `to_q`, `to_k`, `to_v` with bias;
- `q_norm` and `k_norm` as weighted RMS over the full width;
- RoPE where there is one;
- softmax attention at scale 1/√head_dim, with no mask;
- the output times `2·sigmoid(to_gate_logits(query input))` per head, then
  `to_out.0`.

In order:

1. **Video self-attention.** `vx += attn1(ada(vx, s₁, s₀)) · g₂`.
2. **Video cross-attention to the text.** The query is
   `rms(vx)·(1 + s₇) + s₆`. The context is modulated too,
   `ctx·(1 + scale_kv) + shift_kv`, and **not normalised**. Because that
   modulation depends on σ, **the text K and V cannot be cached across
   steps**. `vx += attn2(q, ctx) · g₈`.
3. **Audio self-attention and cross-attention**, the same at width 2048
   (32 × 64).
4. **Audio ↔ video.** Take both streams' states *before* either update.
   - **a2v:** `audio_to_video_attn`, whose q comes from video and k and v
     from audio, all in the audio head layout of 32 × 64:
     - to_q [2048, 4096], to_k and to_v [2048, 2048], to_out [4096, 2048];
     - the gate is from the video table's row 4;
     - the update goes into `vx`.
   - **v2a** is the mirror image, using rows 2 and 3 of each table, and goes
     into `ax`.
5. **FFNs.** `vx += ff(ada(vx, s₄, s₃)) · g₅`.
   - Video: 4096 → 16384 → 4096, GELU-tanh, **no bias** (`ff_bias: false`).
   - Audio: 2048 → 8192 → 2048, **with bias**.

**Output.**
1. shift, scale = `scale_shift_table [2, W]` + the *embedded* timestep. That
   is the vector before the adaLN's SiLU, not after.
2. A **LayerNorm**, not RMS, weightless, eps 1e-6.
3. `x·(1 + scale) + shift`.
4. `proj_out` to 128.

The model predicts **velocity**, `v = ε − x₀`, with `x_σ = (1 − σ)x₀ + σε`.

**Precision.** The reference keeps the residual stream in bf16. f16 is not
safe to assume, because adaLN-scaled activations can overflow 65504, so the
DiT runs in bf16 on Metal. Whether f16 is actually unsafe is unverified; bf16
avoids the question.

## Sampling

Everything below is from the reference `DistilledPipeline`, with its
LTX-2.5 branches.

### Shapes and noise

- Video latent `[128, (F−1)/8 + 1, H/32, W/32]`; audio latent
  `[8, round(F/fps·25), 16]`.
- The latent is kept in bf16 between steps, and each update is computed in
  f32.
- The reference uses one seeded generator, drawn in a fixed order: stage-1
  video, stage-1 audio, stage-2 video, stage-2 audio. The ancestral noise
  comes from a second generator seeded `seed + 10000`. Bit-matching the
  reference would mean reproducing torch's Philox RNG. **We do not**: a kvad
  seed gives a repeatable kvad video, not the reference's video. Parity tests
  inject the reference's noise instead.

### Stage 1: 8 steps, ancestral

σ = `[1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]`,
fixed, with no shift. From the prediction, x₀ = x − σ·v. For each step to σₙ:

```text
σ_down = σₙ · (1 + (σₙ/σ − 1)·η)          η = 1
r      = σ_down / σ
x      = ((1 − σₙ)/(1 − σ_down)) · (r·x + (1 − r)·x₀)
         + ε · √max(σₙ² − σ_down²(1 − σₙ)²/(1 − σ_down)², 0)
```

The last step, to σ = 0, returns x₀. The coefficients are constants of the
schedule. Written as `x = A·(r·x + (1 − r)·x₀) + c·ε`, they are:

| step | σ → σₙ | A | r | c |
|---|---|---|---|---|
| 0 | 1 → 0.99375 | 0.501567 | 0.987539 | 0.861510 |
| 1 | 0.99375 → 0.9875 | 0.668067 | 0.987461 | 0.738504 |
| 2 | 0.9875 → 0.98125 | 0.751189 | 0.987382 | 0.652982 |
| 3 | 0.98125 → 0.975 | 0.801020 | 0.987302 | 0.590269 |
| 4 | 0.975 → 0.909375 | 0.596873 | 0.869915 | 0.755431 |
| 5 | 0.909375 → 0.725 | 0.651669 | 0.635609 | 0.619472 |
| 6 | 0.725 → 0.421875 | 0.766223 | 0.338604 | 0.377621 |
| 7 | 0.421875 → 0 | returns x₀ | | |

The code should compute these, and a test should check them against this
table. The values were derived from the reference's formula, not printed by
it.

### Upsampler, then stage 2: 3 steps, deterministic

- The upsampler works on **un-normalised** latents: `x·std + mean`, upsample,
  then `(x − mean)/std`. The per-channel `std` and `mean` are the
  `per_channel_statistics` in the video VAE file.
- Its structure:
  1. conv3d 128 → 1024, then group norm (32) and SiLU;
  2. 4 residual blocks of conv3d, group norm and SiLU;
  3. a per-frame conv2d 1024 → 4096 and a 2D pixel shuffle ×2;
  4. 4 more residual blocks;
  5. conv3d 1024 → 128.

  All padding is zeros.
- Stage 2 re-noises **both** latents, `0.090625·x + 0.909375·ε`, then steps
  σ = `[0.909375, 0.725, 0.421875, 0]` by plain Euler,
  `x ← x + v·(σₙ − σ)`. The audio is refined again, not frozen.
- The docs say stage 2 has 4 steps. The code runs 3: four sigmas make three
  steps.

### Guidance

None. The distilled model makes one DiT call per step, 11 in all, with no
negative prompt. So `takes_guidance: false`, as for FLUX.1-schnell, and a
request with guidance or a negative prompt is refused before it is queued.

The dev model's guidance is CFG 3 (video) and 7 (audio), STG on block 28,
modality guidance 3, rescale 0.7, over 30 steps: up to four DiT calls a step.
That belongs in its own issue.

### Duration head

The head reads the two connector outputs:
1. Project each to 256 and add a modality embedding.
2. Concatenate them.
3. Pool with one learned query through a 4-head `nn.MultiheadAttention`,
   whose `in_proj` is packed as q, k, v.
4. GELU-tanh MLP, then exp.

The result is in seconds. `round(s · fps)` is clamped to 1–20 s and floored
to 8k + 1 frames.

The CLI uses the head whenever no frame count is given. kvad should do the
same, and let a request give `seconds` or `frames` instead. A 20 s ceiling at
24 fps is 481 frames, which is not what anyone wants to wait for at first. The
first milestone should take an explicit frame count.

### One stage

A single stage at the requested size is not in `ltx-pipelines`. Diffusers
documents one for the distilled model, at 960×544 with the same 8 sigmas.
That is stage 1 with no upsampler, decoded directly, and it is the cheapest
complete path. **It is the first milestone**, because it exercises every
model but the upsampler. Two stages follow once it works.

## Decoding: video and sound

### Video: the conv decoder

`CausalVideoAutoencoder`, `vae/ltx-2.5-video-vae-conv-bf16`. For images,
Qwen-Image's decoder was a video VAE that collapsed to 2D. This one does not
collapse, and it is where three-dimensional convolution arrives.

- **Un-normalise first**: `z·std[c] + mean[c]` with
  `per_channel_statistics.{std,mean}-of-means` (`[128]`, at the top level of
  the file). `scaling_factor` (1.0) is never read.
  - The upsampler uses the same statistics.
  - Both video VAE files carry them, and they were read and compared: they
    are equal.
- The config says `causal_decoder: false` and `timestep_conditioning: false`.
  So the decoder takes no noise and no timestep, and is deterministic.
- **The stack**, from the config's `decoder_blocks` reversed. The grid is
  given for 121 frames at 768×512, and every conv is 3×3×3, stride 1, with
  bias. There are 42 of them.

  | Block | Channels | Grid (t × h × w) |
  |---|---|---|
  | `conv_in` | 128 → 1024 | 16 × 16 × 24 |
  | res × 2 | 1024 | |
  | up, all axes ×2 | → 512 | 31 × 32 × 48 |
  | res × 2 | 512 | |
  | up, all axes ×2 | → 512 | 61 × 64 × 96 |
  | res × 4 | 512 | |
  | up, time ×2 | → 256 | 121 × 64 × 96 |
  | res × 6 | 256 | |
  | up, space ×2 | → 128 | 121 × 128 × 192 |
  | res × 4 | 128 | |
  | PixelNorm, SiLU, `conv_out` | 128 → 48 | |
  | unpatchify 4 × 4 | → 3 | 121 × 512 × 768 |

- **An "up" block** is a conv to `stride product × C / multiplier`
  channels, then depth-to-space `(c p₁ p₂ p₃) → c (t·p₁)(h·p₂)(w·p₃)`. When
  time is doubled, **the first output frame is dropped**. That is how 16
  latent frames become 121: latent 0 decodes to one frame, and every later
  latent to eight. There is no residual path around the up blocks.
- **A res block** is `x + conv(silu(pn(conv(silu(pn(x))))))`. PixelNorm is
  `x / √(mean_c(x²) + 1e-8)`, over channels, with no weight.
- **The padding**: the edge frame is repeated once at each end in time, and
  space is padded with zeros.
- **Unpatchify has a trap.** Channel `c·16 + 4·r + q` goes to pixel
  `(4h + q, 4w + r)`. The **width** offset is the middle factor, not the
  height offset.
- **Output**: `(x + 1)/2`, clamped to [0, 1].

**Conv3d, with no conv3d.** A 3×3×3 conv with that time padding is exactly the
sum of three 2D convs over neighbouring frames:
`y[t] = b + Σₖ conv2d(x[t + k − 1], W[:, :, k])`. There are two ways to write
it:
- as the literal sum of three 2D convs;
- as one 2D conv over the three frames stacked on the channel axis, with the
  kernel reshaped to `[Cout, 3·Cin, 3, 3]`.

Either is exact, needs no new kernel, and turns into an im2col plus a matmul
inside candle.

**It has to run over chunks of frames.** candle's Metal conv2d builds an
im2col buffer, and all 121 frames at once would need about 20 GB. A chunk
needs a one-frame halo at each end and is exact. Which of the two forms is
faster is a measurement to make, not a guess; see the autovectoriser notes.

**Tiling.** By default the reference tiles the decode:
- in time, 80 frames with 24 overlapping;
- in space, 768 px with 64 overlapping;
- blending the tiles with linear ramps.

That is **not exact**. Each output frame depends on about ±13 latent frames.
The reference also has an exact memory-efficient path, the default in
`blocks.py`, which runs whole layers in chunks of frames. kvad should do that,
and tests should compare against the reference with tiling off.

**Size.**

| Size | Work | Largest tensor | Peak |
|---|---|---|---|
| 768×512×121 | 117.5 TFLOP, as much as half a DiT forward | 726 MB bf16 | about 3 GB |
| 1536×1024×121 | about 470 TFLOP | 2.9 GB | about 12 GB, the reference's figure |

The 1536×1024 peak should be smaller here, because each layer's input can be
dropped as soon as its chunked output exists.

The encoder half, `encoder.*` with 319M parameters, is unread for
text-to-video.

### The DiffVAE decoder is for later

`vae/ltx-2.5-video-vae-bf16` is a neighbourhood-attention transformer:
- about 420M parameters;
- a 3D window of 11×11×11 in its last stage, over about 3M tokens;
- one denoising step from seeded noise, so its output depends on the seed.

Its arithmetic is about the same as the conv decoder's. What it needs is a
3D neighbourhood-attention kernel, which candle does not have. On macOS the
reference itself falls back to a path it calls the slowest.

The model card's example uses it, but the conv decoder is a documented,
supported alternative, and the pipeline picks the decoder from the file's
`_class_name`. **kvad starts with the conv decoder.**

### Sound

Four steps: an audio VAE decoder to a mel spectrogram, a vocoder to 16 kHz,
and a bandwidth-extension stage to 48 kHz. **No FFT is needed anywhere.** The
one STFT, in the bandwidth extension, is a strided 1D convolution by a basis
stored in the checkpoint.

1. **Un-normalise.** Flatten the latent `[8, T, 16]` to `[T, 128]` with
   index `c·16 + f`. Apply `x·std + mean` with
   `audio_vae.per_channel_statistics`, then unflatten.
2. **The audio VAE decoder** turns the latent into a log-mel spectrogram
   `[2, 4T − 3, 64]`: stereo, natural log. It is a small 2D UNet-style
   decoder.

   | Step | Channels | Time × mel |
   |---|---|---|
   | `conv_in` | 8 → 512 | 126 × 16 |
   | mid, 2 res blocks | 512 | |
   | up | 512 | 251 × 32 |
   | up | → 256 | 501 × 64 |
   | → 128 | 128 | |
   | PixelNorm, SiLU, `conv_out` | → 2 | 501 × 64 |

   - Each level has three res blocks.
   - Its convs are **causal in time**: padded two rows before and none
     after, and one each side along the mel axis.
   - PixelNorm eps here is **1e-6**, where the video decoder's is 1e-8.
   - Upsampling is nearest ×2 followed by a causal conv, and **drops the
     first time row**, which is where 4T − 3 comes from.
3. **The vocoder**, BigVGAN-v2 style, takes the two channels' mels packed as
   channel `side·64 + mel`, 128 in.
   - `conv_pre` goes to 1536 channels.
   - Six transposed-1D upsamplings, ×5 then ×2 five times (160 in all), halve
     the channels down to 24.
   - Each stage has three residual blocks (kernels 3, 7, 11; dilations 1, 3,
     5), averaged.
   - The activation is SnakeBeta, `x + sin²(e^α·x) / (e^β + 1e-9)`, applied
     anti-aliased: upsample ×2, activate, downsample ×2, with 12-tap filters
     stored in the checkpoint.
   - `conv_post` goes to 2 channels, then a clamp to [−1, 1].
   - The result is 16 kHz stereo, 160 samples per mel frame.
   - **It runs in f32**, as the reference does.
4. **Bandwidth extension, 16 → 48 kHz.**
   1. Take a causal mel of the 16 kHz signal: pad 432 on the left, then a
      1D conv by the stored `forward_basis` `[514, 1, 512]` (Hann window
      included) with stride 80. Then the magnitude, the stored `mel_basis`
      `[64, 257]`, and `log(max(·, 1e-5))`.
   2. Run a second, smaller vocoder: 512 channels, ×240, no final
      activation. This gives a **residual**.
   3. Add it to the 16 kHz signal upsampled ×3 by a 43-tap Hann-windowed
      sinc. The filter is computed at run time, not stored:
      - `tₙ = (n/3 − 7)·0.99` for n in 0..43;
      - `hₙ = sinc(tₙ) · cos²(clamp(tₙ, ±6)·π/12) · 0.99/3`.

      Replicate-pad 7 at each end, apply it as a transposed conv with
      stride 3, multiply by 3, then crop 42 from the left and 40 from the
      right.
   4. Clamp to [−1, 1] again.

   The result is **48000 Hz stereo**, 240480 samples for 121 frames at 24
   fps: 5.01 s.

The audio chain is 160M parameters and small work next to everything else.
What it needs from candle:
- `conv1d` with dilation and `conv_transpose1d`, which both exist;
- per-channel ("depthwise") filtering in the anti-aliasing. candle does
  grouped convolutions as a loop over groups, which is far too slow at 1536
  channels. Reshape `[B, C, L]` to `[B·C, 1, L]` and use one single-channel
  conv instead, which is exact because every channel uses the same filter;
- a transposed conv with padding. Metal's fast path wants padding 0, so pad
  0 and trim afterwards.

Replicate padding (`pad_with_same`) and `upsample_nearest2d` are in candle
already. The audio VAE encoder and the STFT's `inverse_basis` are unread.

## Memory, on this machine (48 GiB)

Sizes at kvad's q8 (Q8_0, 1.0625 bytes a weight) for everything with large
matrices, and bf16 for the small convolutional parts:

| Phase | What is resident | Size |
|---|---|---|
| Text | Gemma tower 12.65 GB, aggregate projections 1.23 GB, connectors 2.14 GB | 16.0 GB |
| Denoise | DiT blocks and modulation, 19.0B at q8 | 20.2 GB |
| Upsample | Upsampler at bf16 | 1.0 GB |
| Decode | conv VAE decoder 0.8 GB, audio decoder, vocoder and BWE 0.3 GB | 1.1 GB |

**All of it resident is about 38 GB before activations.** That is more than
this machine should give the GPU at once: Qwen-Image's 32.7 GB peak was the
most anything has used here so far. So the pipeline runs in phases.

1. Load the text phase and encode.
2. Drop the text phase.
3. Load the DiT and run both stages. The reference loads the DiT twice, once
   per stage; kvad should keep it resident across both.
4. Run the upsampler between the stages.
5. Decode.

The peak weights are the denoise phase: about 22 GB with the upsampler and
decoders resident beside it. The peak *activations* come in the decode at
full size, a few GB at 1536×1024 even when chunked (see Decoding). If the DiT
stays resident through the decode, 1536×1024 comes near the 32.7 GB
Qwen-Image reached. Dropping the DiT before the decode is the fallback if it
does not fit.

This is new for a pipeline here. `Painter`s so far hold everything from load
to drop. Two points need saying before the code:

- **It is not eviction in the #2 sense.** Nothing that belongs to another
  model is touched. Admission accounts the *peak* phase plus whatever stays
  resident between requests.
- **Something has to decide what stays resident between requests.** The
  choice is to keep the DiT and reload the text phase per request (16 GB
  mapped from the q8 cache each time, which is seconds against a generation
  of minutes), or to reload both. The first is the plan. Measure the reload
  before settling it.

**Activations** are small next to the weights, apart from attention.
- Stage 1 at 768×512×121 has 6144 video tokens. A full score matrix for one
  block is 32 × 6144² × 2 bytes = 2.4 GB.
- Stage 2 at 1536×1024 has 24576 tokens, and the full score matrix would be
  39 GB.

**Attention must be chunked** (`nn.rs` already has a chunked attention for the
VAE) or fused. The residual stream at stage 2 is 200 MB, and the FFN's hidden
state is 800 MB.

**Disk.**
- The download is 71 GB.
- The q8 caches add about 36 GB.
- The machine has 176 GiB free.

Whether the bf16 originals are worth keeping once the q8 caches exist is a
question for `kvad rm`, not for this plan.

## Cost

This section is an extrapolation, not a measurement.

**FLOPs.** One DiT forward over 768×512×121 is about **199 TFLOP**: 6144
video tokens, 126 audio tokens and 1024 text tokens. Of each block's share:

| Part of the block | Share |
|---|---|
| Video FFN (6144 × 4096 × 16384 and back) | 40% |
| Video attention projections | 20% |
| Video self-attention scores | 15% |

Audio is about 1% of the total. At 1536×1024 the linear work is 4× and the
self-attention 16×, about 1.15 PFLOP a forward.

**Time.** Qwen-Image here does about 13 TFLOP/s effective (14 s a pass over
about 4.3k tokens at 20.4B parameters). On that basis:

| Run | Estimate |
|---|---|
| Stage 1 at 768×512 | 15 s a step, 2 minutes for 8 |
| Stage 2 at 1536×1024 | 90 s a step, 4.5 minutes for 3 |
| One stage at 768×512×121 (the first milestone) | 2 minutes plus decode |

[#52](https://github.com/bisand/kvad/issues/52) (the M5's neural
accelerators) is aimed at exactly these matmuls. Every large matmul in this
pipeline, including the DiT, the Gemma tower, the connectors and the
aggregate projections, goes through `Proj::forward`
(`crates/gpu/src/common.rs:33`), so it gains from #52 without further work.

## What does not exist here yet

- **Gemma 4.** It is a new tower; see above.
- **Three-dimensional convolution.** It is needed by the upsampler and the
  conv VAE, and candle 0.11 has none. It can be written as 2D convolutions
  over neighbouring frames, which is exact (see Decoding).
- **3D depth-to-space.** It is a reshape and a permute of rank 7, then
  dropping one frame. Whether candle's Metal strided copy handles rank 7 is
  unchecked.
- **Fast depthwise 1D filtering** for the vocoder's anti-aliasing (see
  Sound).
- **Split RoPE**, and a RoPE that spans heads.
- **Gated attention.** It is small, but new.
- **An audio path of any kind:** 1D convolutions exist in candle, but
  nothing here has used them.
- **A video file.** This is now done in
  [#54](https://github.com/bisand/kvad/pull/54) (`kvad::video`).
  - The video is an all-`I_PCM` H.264 stream in an MP4 written by hand,
    tagged BT.709 limited range as the reference's is.
  - The sound is **FLAC with `VERBATIM` subframes inside the same MP4**,
    rather than only a WAV beside it. That plays in Chromium and
    AVFoundation, and a WAV can still be written on its own.
  - Re-encoding to a small file with `ffmpeg`, if it is installed, is still
    to do and belongs with the service. The reference writes H.264 at CRF 19
    and AAC through PyAV.
- **Serving a video.** The server must answer HTTP Range requests. Without
  them Chromium cannot seek in a video it has not fully downloaded: with the
  file written in #54, every seek went back to 0 until the test server
  answered Range. A paused Chromium video also shows the frame *nearest* the
  seek time, so a scrubber seeks to a frame's start, not its middle.
- **Loading a repo without `model_index.json`**, and fetching only named
  files from it.

## How it will be tested, with no reference run

Qwen-Image was cleared with invariants rather than golden tensors: an LM head
pointed at the text tower, and the neighbour correlation of the predicted x₀.
The same kind of checks apply here:

- **Gemma.** Tie `embed_tokens` as an LM head and ask it to continue "The
  capital of France is". The LTX variant is an encoder, and whether it still
  predicts text is unverified, but it is cheap to try.
- **The DiT.** From pure noise at σ = 1, the predicted x₀ should be smooth
  in time and space, as Qwen-Image's was.
- **The decoders.** Decoding a constant latent, and one latent from a real
  run, should give plausible frames and silence.
- **Golden tensors per block.** The strongest check is still a reference run.
  A Python environment in a scratch directory could load *one* DiT block
  (386 M parameters) and the VAE decoder, and write inputs and outputs for
  kvad to compare against. That needs neither the 42 GB file in memory nor a
  CUDA machine. Do this before step 5 below, not after a wrong video.

## The order it is built in

1. This document.
2. **The file writer**, on synthetic frames. Everything else is invisible
   without it. Done in #54: an I_PCM MP4 with FLAC sound, plus a WAV. It was
   decoded frame by frame by ffmpeg, AVFoundation and Chromium at 262×122,
   768×512 and 1536×1024 (level 6.1). Real-time playback in a browser is
   still to be watched by eye.
3. **The decoders**, with per-component fixtures: the conv video VAE decoder
   (where conv3d, chunking and tiling are proven), then audio VAE → vocoder →
   BWE. Done in #57; see the measurements below.
4. **Text:** Gemma 4, the aggregate projection and the connectors. Done in
   #68; see below.
5. **The DiT and one-stage distilled sampling** as `examples/ltx.rs`: prompt
   in, MP4 on disk, no server. **The milestone is one clip, with sound**, at
   512×320 × 25 frames first and 768×512 × 121 second. Done in #70: both
   clips, with sound; see below.
6. **Two stages:** the upsampler and stage 2. Done in #79, at 1536×1024 ×
   121; see below.
7. **The service:** the `video` kind, `/v1/videos`, storage, the CLI, the UI
   page. #51 has the shape. Video files are served with Range support, and
   `ffmpeg` re-encoding is optional.
8. **Later, each in its own issue:**
   - the duration head as the default;
   - image-to-video;
   - the dev model with its guidance;
   - the DiffVAE decoder;
   - the temporal upsampler;
   - the community GGUFs ([#42](https://github.com/bisand/kvad/issues/42));
   - LoRAs ([#41](https://github.com/bisand/kvad/issues/41)).

## What was built, and what it measured

Added 2026-09-24, as the steps land. Everything above is the plan as
written before the code. Where the code went differently, it says so here.
Every comparison is against Lightricks' own code on the same weights and
inputs. `scripts/ltx-fixtures.py` writes the reference outputs, and its
header lists the commands and the numbers to expect. dB is signal to error:
every 10 dB is ten times less error power.

**Step 3, the decoders (#57).**
- **In f32 they are exact.**
  - Video: 122–124 dB PSNR.
  - Audio: 124 dB at the spectrogram, 91–97 dB at 16 and 48 kHz.
- **The video decoder in bf16 on Metal is 59 dB from the reference's f32**,
  and the reference's own bf16 on MPS is 58.
- **The audio runs in f32 throughout**, not only the vocoders. In bf16, the
  decoder's 53 dB spectrogram becomes a 20 dB waveform after the vocoder.
  The three models are 160 M parameters, about a second of work.
- **The conv3d built from 2D convolutions was right the first time.** Its
  speed is not: 768×512 × 121 frames decodes in 71.7 s at an 18.3 GB peak,
  against the reference's 5.2 s and 11.2 GB on MPS. candle's `conv2d` runs
  at 1.9 TFLOP/s, where the same matmul alone runs at 7.2
  ([#56](https://github.com/bisand/kvad/issues/56)). The M5 matmul work
  (#58–#67) changed nothing here, because convolutions do not go through
  `Proj`.
- **Found on the way:** candle 0.11's CPU matmul silently gives wrong
  numbers when its *left* operand is broadcast along the batch. Metal is
  unaffected. A test in `ltx_audio.rs` reports whether that is still so.

**Step 4, the text path (#68).**
- **Tokens are identical** to LTX's own tokenizer, including a prompt cut at
  1024 tokens and one that mixes scripts and stray whitespace.
- **Gemma's first six layers in f32 are exact**: 108–115 dB at every hidden
  state, global layer 5 included. That covers its single key-value head,
  its values taken from its keys, and its RoPE turning a quarter of the head.
- **In bf16 on Metal, those layers are 42–55 dB from the f32 reference.**
  That is within half a dB of the reference's own bf16 on MPS at every
  state, so bf16 costs kvad nothing the reference does not also pay.
- **All 48 layers, against the reference's bf16 on MPS:**
  - bf16: 36–56 dB (55.7 dB at the final state);
  - q8 weights with bf16 activations, through the M5 path
    (`Loader::accelerated`): 27–46 dB.

  These compare two drifting computations, not one against exact. Whether
  q8's extra drift matters is for the contexts and, in the end, for
  generations to say.
- **Time and memory for a 1024-token prompt (the longest there is):**

  | Gemma weights | Time | Peak |
  |---|---|---|
  | bf16 | 2.43 s | 25.6 GB |
  | q8 | 2.75 s | 15.3 GB |

  q8 maps from an 11.6 GB cache in 5–13 s after the first load, which
  quantises.
- **The memory table above assumed the connectors at q8.** They are dense
  bf16 for now, about 4 GB, so the text phase at q8 is about 19 GB rather
  than 16. Worth quantising if the phase has to shrink.
- **The projections and connectors in f32 are exact**: 115.6 dB (video)
  and 117.6 dB (audio), from the reference's own hidden states.
- **The whole path on Metal**, against the reference's f32 contexts from its
  bf16 states: 46.6 / 45.4 dB in bf16 and 46.3 / 44.0 dB at q8, cosine
  0.99997. (These are after step 5's GELU fix; before it, 44.7 / 46.5 and
  45.7 / 43.5.) q8's extra drift inside Gemma does not reach the contexts.
- **q8 is the default for the text path**: 2.75 s rather than 2.43 s for
  the longest prompt, at 10 GB less.

**Step 5, the DiT and one stage (#70).**
- **The DiT's first two blocks and its output heads in f32 are exact**, at
  512×320 × 25 and σ = 0.9875: positions and tokens identical, 115–120 dB
  on the CPU and 118–123 on Metal. Two blocks of 386 M parameters and the
  0.43 B around them are 1.2 B, which the reference can run in f32 on the
  CPU; the other 46 blocks are more of the same block.
- **In bf16 on Metal they are as close as the reference's own bf16:**

  | | video, blocks 0 / 1 | video velocity | audio |
  |---|---|---|---|
  | kvad, bf16 | 46.2 / 45.0 dB | 42.0 dB | 44.6–45.1 dB |
  | kvad, q8 | 44.7 / 45.7 dB | 42.4 dB | 44.0–44.9 dB |
  | the reference's bf16 on MPS | 46.8 / 46.6 dB | 43.6 dB | 45.5–46.1 dB |

- **candle's Metal GELU in bf16 cost the video stream 8 dB.** Its kernel
  evaluates the tanh polynomial in the tensor's own type, where PyTorch
  computes in f32 and rounds once. In the 16 384-wide video feed-forward
  that alone put block 0 at 38.7 dB. GELU now runs in f32 everywhere in the
  video code. It changes nothing measurable in Gemma, whose gated MLP was
  within ±0.4 dB either way.
- **candle's Metal pool frees a dropped tensor's buffer only at the next
  `synchronize()`.** "Drop the text phase" (above) therefore needs a
  synchronise after the drop. Without one, the text path's 13 GB stayed
  beside the DiT's 20 GB and 768×512 × 121 ran out of memory in its first
  step at 37 GB.
- **Two q8 caches under one name overwrite each other**, each finding the
  other's spec stale and quantising again. The Gemma-only check now has its
  own name, and a DiT loaded in part is quantised without a cache.
- **Generations**, on an M5 Pro at q8, seed 1, with a prompt about a golden
  retriever on a beach at sunset that "barks twice":

  | | 512×320 × 25 | 768×512 × 121 (5 s) |
  |---|---|---|
  | Text path, load and encode | 37 s, 2.4 s | 30 s, 0.6 s |
  | DiT load | 106 s (quantising) | 40 s (from the 20 GB cache) |
  | A step | 2.9 s | 29 s |
  | Video decode | 6.8 s | 69 s |
  | All told | 224 s | 382 s |
  | Peak memory footprint | 39.2 GB (first run) | 24.2 GB |

  The first step of a run after a build or a new cache is slower (50 s at
  512×320): shaders compile and the cache pages in.
- **The pictures are right and the sound is plausible.** The dog runs from
  the waterline to the camera over the whole five seconds. The short clip's
  sound has two broadband bursts half a second apart. Both clips peak at
  full scale, and the reference's own decoder and vocoder give the same
  peak and loudness from the same latents.
- **A step is twice the estimate above.** 29 s at 768×512 × 121 is about
  6.9 TFLOP/s against the 13 assumed from Qwen-Image. That is for
  profiling before two stages quadruple the tokens.
- **How memory is measured here**, from step 6 on: the peak *footprint*
  (`/usr/bin/time -l`), which counts GPU memory. The resident set size,
  given for step 5 at first (27.7 and 37.2 GB), counts the mapped q8 cache
  files and misses private GPU buffers.

**Step 6, two stages (#79).**
- **The upsampler matches the reference**: 98 dB in f32, on a real latent
  from a generation. Its conv3d pads time with **zeros**, not edge frames,
  and its group norm takes each group's statistics over the **whole clip**,
  as PyTorch's does on a 5D tensor. **In bf16 it is only 27 dB from exact**,
  and so is the reference's own bf16 on MPS. It is 498 M parameters and a
  few seconds of work, so kvad runs it in f32.
- **Stage 2** re-noises both latents to σ = 0.909375 and takes three Euler
  steps, sound included.
- **The decoder had to go chunked by blocks, as this plan said it
  should.** Chunking each convolution was not enough:
  - a residual step done whole keeps about five full-size tensors alive,
    3 GB each at 1536×1024;
  - each convolution padded a copy of its whole input;
  - PixelNorm made f32 copies of whole stages;
  - candle's pool keeps freed buffers until a synchronise.

  Decoding 1536×1024 × 121 ran out of memory at 77 GB. Now each residual
  step, up block and the output tail runs a chunk of frames at a time, with
  the halo frames its convolutions read, into a preallocated output. It is
  still exact (122–124 dB in f32, including with one-frame chunks).

  | Decode alone | Before | After |
  |---|---|---|
  | 768×512 × 121 | 18.3 GB, 68 s | 8.7 GB, 72 s |
  | 1536×1024 × 121 | out of memory at 77 GB | 24.2 GB, 285 s |

  That is twice the reference's 12 GB. The blocks' input and output are
  still whole, and candle rounds every buffer up to a power of two.
- **Generations**, same prompt and seed:

  | | 768×512, 1 stage | 768×512, 2 stages | 1536×1024, 2 stages |
  |---|---|---|---|
  | Stage 1 | 8 × 29 s | 8 × 6.5 s | 8 × 30 s |
  | Upsampler | – | 3.9 s | 7.7 s |
  | Stage 2 | – | 3 × 29 s | 3 × 193 s |
  | Video decode | 69 s | 69 s | 297 s |
  | All told | 382 s | 309 s | 1224 s |
  | Peak footprint | 24.2 GB | 26.8 GB | 36.6 GB |

  Two stages at 768×512 are faster than one and more detailed. At 1536×1024
  the clip holds together for all five seconds, and at full resolution the
  fur and sand are sharp.
- **Where the 20 minutes go:** stage 2 (48%) and the decode (24%).
  - A stage-2 step is about 1.1 PFLOP, done at 5.6 TFLOP/s. The Cost
    section's estimate was 90 s a step; it takes 193.
  - The decode's convolutions are #56.

  Both are for profiling before the service (step 7) makes them a user's
  wait.


**Re-measured after rebasing onto #80**, which reads weights and the q8
cache past the page cache. Same prompt and seed, two runs each; the latents
are byte-for-byte those above.

| | 768×512, 1 stage | 768×512, 2 stages | 1536×1024, 2 stages |
|---|---|---|---|
| Loads: text path + DiT | 9 s + 3.6 s | 9 s + 3.6 s | 9 s + 3.6 s |
| All told | 338–341 s | 241–244 s | 1149–1160 s |
| Peak footprint | 26.8 GB | 29.2 GB | 35.8–39.0 GB |

- **The loads were 70–87 s and are now 13 s.** That is where the time went.
  Steps and the decode are unchanged against the pre-rebase build measured
  the same day.
- **The earlier peaks were too low.** Through the old mapped reads, macOS
  kept the 33 GB of q8 cache files in its file cache and compressed memory
  to make room: 48 GB during one 768×512 load, including part of the DiT,
  which showed 20.3 GB resident instead of 22.8. Nothing is compressed now.
  A footprint compares across a change to how weights are read only with
  the compressor's pages beside it.
- **1536×1024 peaks in stage 2's first step**, just after the upsampler ran
  in f32 beside the DiT, and the peak varies by 3 GB between runs. Up to
  39 GB of 48 is the pipeline's tightest point; freeing the upsampler before
  stage 2 would give some room.
