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
- **Both stages on 2.5 sample ancestrally.** It is not deterministic Euler.
  Fresh noise is added at every step from a second RNG. Stage 2 has done so
  since the reference's 1.4.0; see "The reference moved" below.
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
| Video VAE, diffusion decoder (the default since step 3 below; its encoder is the conv file's) | `vae/ltx-2.5-video-vae-bf16.safetensors` | 0.74 B | 1.47 GB |
| Audio VAE + vocoder + BWE | `vae/ltx-2.5-audio-vae-bf16.safetensors` | 0.18 B | 0.37 GB |
| Duration head | `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 1.9 M | 4 MB |

That is about 72.5 GB to download.

The repo also holds files that text-to-video does not read:
- the dev DiT, 42.0 GB, for later;
- three Comfy-only int8 and NVFP4 files. The model card says these are not
  for PyTorch, and they are not for us;
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
  comes from a second generator seeded `seed + 10000`, and stage 2's from a
  third, `seed + 20000`. Bit-matching the
  reference would mean reproducing torch's Philox RNG. **We do not**: a kvad
  seed gives a repeatable kvad video, not the reference's video. Parity tests
  inject the reference's noise instead.
- Every draw of a generation has a stream of its own, `image::nn::stream`:
  the seed plus the draw's number *mixed*. Until #167 it was the seed plus
  the draw's number times γ, and γ is the step of SplitMix's own counter, so
  each draw was the first one started a few numbers late, and its noise the
  first draw's moved that many elements along. See "The sound was a train of
  clicks" below.

### Stage 1: 8 steps, ancestral

σ = `[1, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0]`,
fixed, with no shift. From the prediction, x₀ = x − σ·v. For each step to σₙ:

```text
σ_down = σₙ · (1 + (σₙ/σ − 1)·η)          η = 1
r      = σ_down / σ
x      = ((1 − σₙ)/(1 − σ_down)) · (r·x + (1 − r)·x₀)
         + ε · √max(σₙ² − σ_down²(1 − σₙ)²/(1 − σ_down)², 0)
```

The last step, to σ = 0, returns x₀. The video and the sound are stepped
alike, each with noise of its own at every step: the reference's
`euler_ancestral_denoising_loop` draws the video's and then the sound's. The
coefficients are constants of the schedule. Written as `x = A·(r·x + (1 − r)·x₀) + c·ε`, they are:

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

### Upsampler, then stage 2: 3 steps, ancestral

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
  σ = `[0.909375, 0.725, 0.421875, 0]` as stage 1 steps its last three:
  rows 5 and 6 of the table above, each with noise of its own for the video
  and for the sound, and then the prediction. The audio is refined again,
  not frozen. Until the reference's 1.4.0 these were plain Euler,
  `x ← x + v·(σₙ − σ)`, and the dev model's stage 2 still is.
- The docs say stage 2 has 4 steps. The code runs 3: four sigmas make three
  steps.

### Guidance

None. The distilled model makes one DiT call per step, 11 in all, with no
negative prompt. So `takes_guidance: false`, as for FLUX.1-schnell, and a
request with guidance or a negative prompt is refused before it is queued.

The dev model's guidance is CFG 3 (video) and 7 (audio), STG on block 28,
modality guidance 3, rescale 0.7, over 30 steps: up to four DiT calls a step.
That belongs in its own issue. (It came later; see "The dev model with its
guidance" below.)

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
first milestone should take an explicit frame count. (It is the default now,
under this machine's own ceiling; see "The duration head as the default"
below.)

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

### The DiffVAE decoder

At first this said: `vae/ltx-2.5-video-vae-bf16` is a neighbourhood-attention
transformer of about 420M parameters, and kvad starts with the conv decoder.
It did. Read in full, the file's own config (`_class_name`
`NADiffusionDecoder`) is this:

| Stage | Blocks | Width | Window (t, h, w) | Then |
|---|---|---|---|---|
| in | | 128 → 2048 | | `conv_in`, a linear layer, after un-normalising |
| 1 | 4 | 2048 | 3×7×7 | ×2 in space, to 1024 |
| 2 | 6 | 1024 | 3×7×7 | ×2 in time, to 512, the first frame dropped |
| 3 | 4 | 512 | 3×5×5 | ×2 in all three, 512, the first frame dropped |
| 4 | 2 | 512 | 3×5×5 | ×2 in all three, to 256: the *context* |
| 5 | 8 | 256 | **11×11×11** | `norm_out`, `conv_out` to 48, unpatchify 4×4 |

417M parameters, 0.83 GB in bf16, beside an encoder of 319M that image-to-video does not need (the conv VAE's serves).

- **A block** is pre-norm: `x + NA(rms(x))`, then `x + SwiGLU(rms(x))`, the
  MLP `w_down(silu(w_gate·x) · w_up·x)` 4× wide, no biases.
- **NA** is 3D neighbourhood attention, NATTEN's `na3d`: each token attends
  to exactly the window around it, which is shifted inward at the edges
  rather than cut, `start = clamp(i − ⌊k/2⌋, 0, n − k)` on each axis, so
  every axis must be at least its window. Heads of 64; the fused `qkv`
  splits q, k, v in that order; `q_norm` and `k_norm` are RMS norms over a
  head, and q is scaled by 1/8 before the rotation.
- **RoPE** is absolute and 3D, with positions 0, 1, 2 … on each axis: 16 of
  a head's 64 dimensions for time and 24 each for rows and columns, rotating
  *adjacent pairs* (2i, 2i + 1), with frequencies `10000^(−2i/d)`. The
  reference cuts the columns into four slabs to rotate them, but keeps the
  positions running across them, so the slabs change nothing.
- **Upsampling** is a linear layer to `p₁·p₂·p₃` times the channels, divided
  by the reduction, then depth-to-space in the conv decoder's channel order;
  doubling time drops the first frame.
- **Stage 5 is one step of diffusion.** Its input is noise the size of the
  frames, patchified 4×4 to 48 channels, at `t = 1` (`× 1000` into the
  embedding). The step's embedding is PixArt's: a 256-wide sinusoid, cosines
  first, `linear → silu → linear` to 384; a shared adaLN turns
  `silu` of that into seven rows, and each block adds its own table. A block
  adds `context_proj(context)` to x, then `x + NA(rms(x)(1 + s₀) + s₁)`, then
  `x + SwiGLU(rms(x)(1 + s₃) + s₄)`; the gate rows are unused. The output is
  the clean frames (`model_output_type: x0`, one step), so **the frames
  depend on the seed** the noise is drawn from.
- **Edges.** The last latent frame is repeated twice through stages 1–4 and
  the copies are cut from the context before stage 5, to at least 11
  frames; a latent smaller than 3×7×7 is edge-padded first. The reference
  runs stages 1–3 on the whole clip and stages 4–5 in overlapping tiles,
  blended.

**The cost is stage 5.** At 768×512 × 121 it is 121×128×192 = 3M tokens,
each attending to 1331 others: about 33 TFLOP of attention a clip, four
times that at 1536×1024, where one stage-5 activation is 6 GB in bf16. candle has
no neighbourhood attention, and the reference itself, on a Mac, falls back
to a path it calls its slowest.

In three steps:
1. **Exact.** Every layer, with neighbourhood attention done plainly (a loop
   over the window's offsets with an online softmax), checked stage by stage
   against the reference's own eager path on small latents.
2. **Fast.** A Metal kernel for neighbourhood attention, and stages 4–5 in
   tiles, so that a real clip decodes.
3. **In the pipeline**: a choice of decoder, the seed, and what it looks like
   beside the conv decoder's frames.

**Step 1, exact.** `video::ltx_diffvae` is every layer above, with
neighbourhood attention as a loop over the window's offsets: each query's
key at offset `(a, b, c)` is its window's corner plus a constant, so one
gather per offset, and an online softmax in f32. It is checked against the
reference's own modules on their eager path (`scripts/ltx-fixtures.py
--diffvae`, `examples/ltx_diffvae.rs`), each stage from the reference's
input to it, on a seeded 3×8×8 latent (17 frames of 256×256; stage 5 is
17×64×64 tokens):

| Stage | f32 on Metal | bf16 on Metal | The reference's own bf16 on MPS |
|---|---|---|---|
| 1–3, the latent to stage 4's input | 120.1 dB | 43.4 dB | 43.6 dB |
| 4, to the context | 125.0 dB | 47.9 dB | 48.8 dB |
| 5, the context and noise to the frames | 123.8 dB, 0.00 of a level | 47.5 dB, 3.4 levels at most | 50.8 dB |

- **Exact in f32 at every stage**, the first time it ran: the layout of each
  upsampling, the rotary pairs and their split of a head, the windows at the
  edges, the fused `qkv`, the step's rows.
- **Stage 5 in bf16 is 3.3 dB short of the reference's own bf16**, where the
  earlier stages are within 1 dB of it. It is 3.4 of 255 levels at the
  worst pixel. Step 2 found why: every layer rounded before its bias.
- **The plain attention is far too slow**: stage 5 took 233 s here, where
  the reference's eager path took 7.3 s on the CPU and its whole decode 2.2 s
  on MPS. That path groups queries whose windows share a geometry and does
  each group as one masked attention over their keys' bounding box, on the
  matrix units. A real clip's stage 5 is 43 times this one.

**Step 2, fast.** Five things, each measured on its own.

1. **A neighbourhood-attention kernel** (`mpp_neighbourhood`), flash
   attention on the M5's matrix units as `mpp_attention` does it. A SIMD
   group takes a 4×4 patch of queries in one frame. They share their time
   window, and with windows up to 13 wide their keys fit a box of runs of 16
   consecutive tokens, one fragment each. So the group walks its box two
   runs at a time and hides each query's keys outside its own window. At
   11×11×11 that is 54% of the products used. It agrees with the plain loop
   in f32 to 57.0–59.4 dB in bf16 (74.2–76.4 in f16), the answer's own
   rounding. Stage 5's attention at 768×512 runs at 11.4 TFLOP/s of the
   window's products, about 21 counting the hidden ones, where
   `mpp_attention` runs: 47 ms for 16 frames of one block.
2. **Every bias added before the rounding.** The 3.3 dB was rounding: the
   dense product was rounded to bf16, the bias added, and the sum rounded
   again, where PyTorch's `addmm` rounds once. `mpp::dense_bias` fills the
   answer with the bias's rows and accumulates the product into it
   (`matmul2d`'s multiply-accumulate), and `Linear` uses it for every dense
   layer with a bias, one row included. The one-row products make the
   step's modulation rows, and an error there is every token's: they were
   2.0 of the 3.2 dB. It is faster too, 1.65–7× over the product and
   candle's broadcast add (`dense_bias_race`). This is every image and video
   model's dense `Linear`; the DiT's bf16 check is at or above the
   reference's own bf16 at every point with it (velocity 44.2/46.1 dB,
   the reference's 43.5/45.7).
3. **Three fused kernels** in `ltx_fused`, where candle's chains of f32
   casts and strided broadcasts took 5.5 s a stage-5 block for the norms and
   RoPE alone: `norm_affine` (a weighted RMS norm and its modulation),
   `head_norm_rope` (q and k from the fused projection, normed a head at a
   time and turned on adjacent pairs, written with v straight into the
   grid's q, k and v) and `swiglu`.
4. **Memory.** Everything but the attention goes a few frames at a time.
   And candle's pool rounds each buffer up to a power of two and keeps it
   until a synchronise: a 1.52 GB activation took 2.15 GB, and q, k, v
   together 8.6 GB for 4.56. So the whole-grid tensors are allocated at
   their size outside the pool (`fused::metal::exact`), and each block ends
   with a synchronise.
5. **Tiles**, as the reference tiles: stages 1–3 on the whole clip, stages
   4–5 on tiles of stage 4's input of at most `BUDGET` (3M) stage-5 tokens,
   overlapping by the halo (20 stage-4 tokens: 40 frames, 160 pixels),
   blended with complementary linear ramps. The noise is one field over the
   whole clip, each tile drawing its own block of it
   (`image::nn::noise_block`), where the reference draws each tile's afresh.
   So tiled and whole decodes differ only by the edges.

With all five, each stage from the reference's input, on the step-1
fixture:

| Stage | bf16 on Metal | The reference's own bf16 on MPS |
|---|---|---|
| 1–3 | 44.2 dB | 43.6 dB |
| 4 | 49.2 dB | 48.8 dB |
| 5 | 51.4 dB, 2.4 levels at most | 50.8 dB |

f32 is still exact (120.1, 125.0, 123.8 dB). Whole clips, from latents the
pipeline made, against the conv decoder on the same latents, decode alone,
peak footprint:

| Clip | DiffVAE | Conv decoder |
|---|---|---|
| 768×512 × 121 | 10.9–11.5 s, 15.9 GB; one tile of 2.97M tokens | 7.9 s, 8.2 GB |
| 1536×1024 × 121 | 61.7–70.3 s, 22.3 GB; 1×2×3 tiles, 16.8M tokens | 30.7 s, 17.9 GB |

Tiling's own cost, the 768×512 clip tiled against whole: 63.2 dB as 1×1×3
tiles, at most 8 levels apart; 58.5 dB as 2×2×3, at most 14. Nothing shows
where the tiles meet. Profiled at 768×512, a stage-5 block takes about
1.5 s: 0.48 for the norms, `qkv` and the rotation, 0.40 for the attention,
0.47 for the feed-forward and 0.14 for the context.

**Step 3, in the pipeline.** The diffusion decoder is LTX-2.5's default, as
it is the reference's: its README's split-file examples read
`vae/ltx-2.5-video-vae-bf16`, and calls the conv file "lighter". A request can
name either: `decoder: "diffusion" | "conv"` on `/v1/videos`, `--decoder` on
`kvad videos make` and on `examples/ltx.rs`, and a selector beside the seed in
the web UI. A model with one decoder refuses the field by name. The video
keeps which decoder made it (`kvad.decoder`; migration 014). Videos made
before this have none recorded, and all were conv.

- **The file is fetched at load** with the rest, 1.47 GB more. The conv file
  stays too: its encoder starts a video from a picture, and the upsampler
  reads its latent statistics. Both are the same, byte for byte, in the
  diffusion decoder's file (84 of 84 encoder tensors and both statistics), so
  image-to-video is the same whichever decoder is used.
- **The seed.** Stage 5's noise comes from the request's seed, turned
  (`ltx_diffvae::noise_seed`) so that it is not the stream the DiT's noise
  came from. The same seed and settings give the same video.
- **Progress** counts the decode tile by tile, and a cancel stops it
  between tiles.

Whole generations, `examples/ltx.rs`, distilled, a dog on a beach, seed 1:

| Clip | With the diffusion decoder | With the conv decoder, as before |
|---|---|---|
| 768×512 × 121 | 77.6 s, the decode 12.5 s; peak 26.9 GB | about 75 s; peak 27.0 GB |
| 1536×1024 × 121 | 391 s, the decode 88.0 s; peak 31.4 GB | 366 s, the decode 35.3 s; peak 31.3 GB (#94's image-to-video run, another prompt) |

The peak is still stage 2's, so admission's line stands. In the service, the
same 2-second fox clip decoded in 8.6 s with the diffusion decoder and 5.9 s
with the conv one. The frames are the same scene. The diffusion decoder's are
a little finer in texture, sand and fur, and the difference is subtle at this
size. At 1536×1024 the decode takes 88 s in the pipeline against 62–70 s on
its own. That is not yet explained.

### The temporal upsampler, and DFR

`latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0` doubles a
latent's frames. The reference runs it in one place only: the temporal rounds
of its DFR pipeline ("Diffusion Fidelity Rendering", `dfr_pipeline.py`), which
it calls its production pipeline. Each round upsamples the finished latent in
time, splits the clip into time tiles that meet at keyframes carried over
from the round before, and denoises each tile again with the distilled DiT.
Those keyframes only exist because DFR's own two stages generate them, and
its decode reads them too. So the upsampler is one piece of DFR, which is
built here in steps, each checked against the reference:

1. the temporal upsampler;
2. conditioning tokens in the DiT: generated keyframe slots, anchor
   keyframes, a reference latent, each token at its own σ, and frozen sound;
3. DFR's stages 1 and 2: its canvas and keyframe layout, a deterministic
   stage 1, the keyframes upsampled together, and stage 2 with the detailing
   IC-LoRA against the half-size video;
4. the temporal rounds: tiles that meet at keyframes, four ancestral steps
   at η 0.5 re-blended after each, the sound time-compressed per tile, the
   tiles stitched and their keyframes carried on;
5. the keyframe-aware decode: the DiffVAE attending to the keyframes as a
   second stream;
6. in the pipeline: 48 and 96 fps.

What each needs was read from the reference's code; the riskiest are the
keyframe-aware decode and the temporal rounds.

**Step 1, the upsampler.** It is the spatial upsampler's class,
`LatentUpsampler`, with `mid_channels` 512 rather than 1024 and time in place
of space. Its upsampling is a conv3d to twice the channels, then channel
`2c + p` of frame `f` goes to frame `2f + p`, and the first frame is dropped:
a latent's first frame stands for one pixel frame, and each after it for
eight. So `F` latent frames become `2F − 1`, and `8(F − 1) + 1` pixel frames
become `16(F − 1) + 1`, twice the frame rate. 131M parameters. The config also
says `rational_resampler: true`, which only the reference's spatial branch
reads.

`ltx_upsample::Upsampler` now loads either file. Against the reference's
`upsample_video`, on a 768×512 × 121 latent from the pipeline (16 frames to 31;
`scripts/ltx-fixtures.py --upsampler`, `examples/ltx_upsample.rs --temporal`):

| | Against the reference's f32 |
|---|---|
| f32, CPU | 92.0 dB |
| f32, Metal | 90.8 dB, in 1.4 s |
| bf16, Metal | 26.9 dB |
| the reference's own bf16, MPS | 26.8 dB |

Exact in f32 the first time it ran. Like the spatial upsampler, it loses most
of its precision in bf16, the reference's own included, so it runs in f32.

**Step 2, conditioning tokens.** DFR's stages append tokens after the
video's own, and the DiT reads each by its own σ, place and keyframe mark
(`video::ltx_cond`, whose module notes have the table):

- **Anchor keyframes** (`VideoConditionByKeyframeIndex`): a given latent
  frame at one pixel frame, clean, its mask `1 − 0.95` rounded to bf16 as the
  reference rounds it, 0.050048828125. Unmarked.
- **Generated keyframes** (`VideoGeneratedKeyframeSlots`): one latent frame
  of tokens each at one pixel frame, denoised like the video, and marked, so
  the DiT adds its learned keyframe vector, as it always does to the first
  latent frame. Read back out after a stage.
- **A reference latent** (`VideoConditionByReferenceLatent`): the half-size
  video, clean at σ 0, its rows and columns scaled to the target's pixels.

Each token's σ is the step's times its mask, so a state has up to three
sets of the DiT's per-token rows. `ltx_fused::Held` had two, in a fixed
order: the rows of a picture held first, then the rest. It now also takes an
index, each token reading the row it names, in the kernels and the chain
alike. Frozen sound is the audio at σ 0, which the reference also feeds to
the video's audio gate. Noising is the reference's `lerp`, which on the CPU
is two fused multiply-adds: without the fusing, 159 dB; with it, identical.

Against the reference's own items and its DiT's first two blocks, on its
512×320 × 25 latent at 60 fps with two anchors, two generated keyframes and
a half-size reference, 1440 tokens, the sound frozen
(`scripts/ltx-fixtures.py --conditioned`, `examples/ltx_cond.rs`):

| | Video | Audio |
|---|---|---|
| The state: every place, mask, mark, clean and noised latent | identical | |
| f32, CPU | 114.0 dB (the 800 appended tokens 113.5) | 119.4 dB |
| bf16, Metal | 44.5 dB (appended 44.7) | 45.3 dB |
| the reference's own bf16, MPS | 44.0 dB | 44.8 dB |

The DiT's other checks (plain, a picture held, both perturbations) are
unchanged by the index, to the tenth of a dB.

**Step 3, DFR's stages 1 and 2** (`video::ltx_dfr`). The canvas pads
`frames − 1` to whole segments of 24 or 32 pixel frames, whichever pads
less and the longer on a tie, and puts a keyframe at the end of each: 121
frames are five segments of 24, keyframes at 24, 48, 72, 96 and 120. Above
30 fps the DiT is told 60, with the sound still timed at the clip's own
rate. Stage 1 is the distilled schedule's eight steps at half size with the
keyframe slots, ancestral at η 1 as the plain pipeline's are (plain Euler
until the reference's 1.4.0, and when this step was checked). The
video and the keyframes are then upsampled apart, the keyframes as a clip of
their own. Stage 2 re-noises both to 0.909375 and takes three steps with
stage 1's video appended as a clean reference latent at half size and the
detailing IC-LoRA fused in at 0.5. Its sound is denoised with the video,
which reads it, and then dropped: DFR ships stage 1's.

A step is the reference's `X0Model` and `post_process_latent`. Each token's
prediction is `x − σ·m·v`, rounded, blended `x₀·m + clean·(1 − m)` in f32 and
rounded again, then one Euler step for every token at the step's σ. That
is η 0, which the table below was measured at. At η 1, as the pipeline now
runs, the step is the rounds' ancestral one (step 4), blended again after
its noise.

The LoRA is `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`, 0.33
GB, a gated repo apart from LTX-2.5's own: its terms have to be accepted on
the Hub by the account whose token fetches it. It is rank 32 on the blocks'
attention and feed-forward only. Its metadata says `reference_downscale_factor`
2, which the reference latent's places are scaled by.

Against the reference's own `DiffusionStage`, stage 1, both upsamplings and
stage 2, as `DFRPipeline` chains them, with the DiT cut to two blocks: a
512×320 × 49 clip at 48 fps (keyframes at 24 and 48, the DiT told 60), from
the noise the reference drew, stage 2 from Kvad's own stage 1
(`scripts/ltx-fixtures.py --dfr --detailing`, `examples/ltx_dfr.rs`):

| | Stage 1: video, keyframes, sound | Upsampled: video, keyframes | Stage 2: video, keyframes, sound |
|---|---|---|---|
| f32, CPU | 104.0, 104.2, 112.8 dB | 100.6, 100.9 dB | 103.1, 103.4, 113.7 dB |
| bf16, Metal (upsampler f32) | 34.2, 34.2, 35.8 dB | 34.1, 32.7 dB | 33.3, 33.3, 30.5 dB |
| the reference's own bf16, MPS | 33.7, 34.0, 34.3 dB | 31.6, 31.4 dB | 32.4, 32.5, 27.9 dB |

Exact in f32 the first time it ran; in bf16, eleven steps from the
reference's f32, a little closer than the reference's own bf16. Without the
LoRA, stage 2 is 28.5 dB from the reference's with it, so it does its part.

**Step 4, the temporal rounds** (`ltx_dfr::round`). A round doubles the
frames with the temporal upsampler, 49 at 48 fps to 97 at 96, and the
keyframes' places double with them into seams. The clip is cut at the seams
into `2^round` tiles, the leftover segments going to the first. Every tile
after the first starts a segment and one latent frame early, a lead-in that
it denoises for context and then drops, so the tile before keeps the seam's
frame and nothing is blended. Each tile holds its seams as anchors at 0.95
and generates a keyframe halfway between each pair of marks, seeded from the
video's nearest latent frame. It is re-noised to 0.975 and takes four
ancestral steps at η 0.5, the tokens blended towards their clean latents
again after each step's noise. Its sound is stage 1's, frozen at σ 0 and
resampled linearly over the seconds the tile plays. The seams and the new
keyframes, the earlier tile's where two tiles made the same one, are the
next round's seams.

Kvad's `Ancestral` now takes an η, and groups `σ_down²·α_next²/α_down²` as
the reference does; η 1, the plain pipeline's, is otherwise unchanged. An
anchor's latent is rounded to bf16, as DFR carries its keyframes, whatever
the DiT runs in.

Against the reference's own round loop, copied from `DFRPipeline.__call__`
with its helpers (`TemporalTilePlan`, `_audio_latent_for_tile`, the carry
merge) and its `euler_ancestral_denoising_loop`, on from step 3's stage 2:
two rounds, 97 frames at 96 fps in two tiles, then 193 at 192 in four,
keyframes at the reference's places, and every noise draw the reference made
replayed and used. The fixture keeps the loop's latents in the run's dtype,
where DFR keeps them in bf16, so that the f32 run is f32 throughout.

| | Round 1: video, keyframes | Round 2: video, keyframes |
|---|---|---|
| f32, CPU | 105.7, 104.5 dB | 106.5, 105.4 dB |
| bf16, Metal (upsamplers f32) | 33.2, 33.2 dB | 37.4, 34.8 dB |
| the reference's own bf16, MPS | 33.5, 32.9 dB | 36.9, 34.5 dB |

Exact in f32 the first time it ran. Unit tests pin the tile plans for both
rounds and for five segments in two tiles, and a tile's sound.

**Step 5, the keyframe-aware decode** (`ltx_diffvae::DiffDecoder::decode_keyed`).
DFR decodes with its keyframes beside the video, as a second stream of
*planes*:
- **The planes' path:** one latent frame each, tagged with the file's
  `type_emb` (its only keyframe weight), then through the video's own
  weights.
- **Upsampling:** each plane is upsampled as a clip of one frame, so in
  space only.
- **Place in time:** each plane has a place in each stage's time, the
  middle of the cell that holds its pixel frame.
- **Attention:** the streams meet only in the attention, which becomes
  joint (`joint_na3d`), with one softmax per query:
  - a frame's query sees its own window and the same rows and columns on
    its two nearest planes;
  - a plane's query sees its own window and the same on its two nearest
    frames.
- **Stage 5:** the planes are pixels of their own noise, denoised with the
  video and dropped.
- **Tiling:** each tile carries the planes inside it and the nearest one on
  each side, at times counted from its own first frame.

One thing differs from the plain decode: the reference's joint kernels cut
a window at the grid's edges instead of shifting it inward as NATTEN does.
Its eager and Triton versions agree on this, and Kvad follows them.

The M5 kernel (`mpp_neighbourhood`) gained a joint specialisation. It walks
the query's cut window, then the same rows of the other stream's frames, in
one online softmax. The planes' side is the same kernel with the roles
turned: a window of one plane, then the video's frames. Off the M5 each
part is its own attention with its log-sum-exp, and the parts are merged
into one softmax (`plain_part`, `merge`), which is what the kernel is
tested against. A first version ran the video's side that way on the GPU
too: three whole f32 activations and their merge at stage 5 made the decode
2.1× slower and 13 GB larger, and merging them one part at a time ran out
of memory.

Against the reference's own keyframe decode, stage by stage, each stage
from its input: the 3 × 8 × 8 latent of the plain check and two planes at
pixel frames 8 and 16 (`scripts/ltx-fixtures.py --diffvae --keyframes`,
`examples/ltx_diffvae.rs --keyframes`).

| Stage | f32, CPU (video, planes) | bf16, Metal (video, planes) | the reference's own bf16 |
|---|---|---|---|
| 1–3 | 119.7, 119.2 dB | 44.2, 44.0 dB | 43.6, 43.3 dB |
| 4 | 125.0, 125.1 dB | 49.2, 49.1 dB | 48.8, 48.7 dB |
| 5 | 125.4 dB | 50.6 dB, 3.1 levels at most | 49.7 dB |

The planes' stage-5 times are the reference's. Without the planes, the
same step is 16.4 dB from the reference's with them, so they matter.

Exact in f32 the first time it ran, and above the reference's own bf16 at
every stage in bf16. The kernel is tested against the parts in f32 at
57.1–58.8 dB, which is its answer's rounding to bf16.

Whole clips from latents the pipeline made. There are no DFR keyframes
until step 6, so the planes are the latent's own frames at DFR's places,
24 to 120. That measures what planes cost, not what DFR's planes make.
Decode alone, peak footprint:

| Clip | Plain | Five planes |
|---|---|---|
| 768×512 × 121, one tile | 10.7–10.8 s, 18.4 GB | 11.5–11.8 s, 20.6 GB |
| 1536×1024 × 121, 1×2×3 tiles | 64.1–73.4 s, 22.3 GB | 77.6–79.9 s, 24.6 GB |

On their own at stage 5's size, the joint kernel with two planes a frame
takes 45.4 ms where the plain one takes 44.2 (16 frames of 768×512). Its cut
windows do less work at the time edges than shifted ones, which pays for
the planes' rows. Tiling costs no more with planes: the 768×512 clip as
2×2×2 tiles against whole is 61.5 dB PSNR with them and 59.2 without.

The 15.9 GB in step 2's table for the 768×512 clip does not reproduce
today: `master`'s own decode of the same latent peaks at 18.9–19.1 GB, and
this branch's at 18.4. The cause is not found yet.

**Step 6, in the pipeline.** A request picks its unguided pipeline:
- `pipeline: "fast" | "dfr"` on `/v1/videos`;
- `--pipeline` on `kvad videos make`;
- a selector in the web UI.

The fast pipeline stays the default at 30 fps and below. Above 30, DFR runs
whatever is asked, since only its temporal rounds make those rates. `fps`
must halve to 30 or less: 48 and 96 are 24 doubled once and twice, and 60
and 120 are 30 so doubled. `frames` and `fps` describe the clip as it is
delivered, so at 48 fps frames are on a grid of 16, and 241 is 121 at 24
doubled. Guidance runs the dev model, which DFR does not, so a guided
request above 30 fps is refused by name. The video keeps which pipeline
made it (`kvad.pipeline`, migration 015).

- **The detailing LoRA is fetched by the first DFR request**, not at load:
  its repo is gated, and an account without access keeps the fast
  pipeline. Without access, the request is a 400 that names the page
  whose terms to accept.
- **Three DiT loads a generation.** Stage 1 runs on the plain DiT, stage 2
  on the DiT with the LoRA fused in (cached apart, as the dev model's
  second DiT is), and the rounds on the plain one again. Each is dropped
  before the next loads.
- **The sound is stage 1's**, as the reference ships it, cut to the clip's
  length. The canvas pads the end of a clip to whole segments, and that
  padding is trimmed from the video too.
- **A picture to start from** holds the first latent frame in stages 1 and
  2 (`ltx_cond::State::held`), and in every temporal tile that starts at
  the clip's first frame, as the reference re-attaches it.
- **The conv decoder** has no keyframes: with `decoder: "conv"`, DFR's
  clip is decoded without them.

**Admission.** DFR's largest DiT call is not always stage 2. Each temporal
round's tiles carry a segment of lead-in, their anchors and their new
keyframes. For 121 frames, stage 2 holds 25 latent frames of tokens: 16 of
video, 5 keyframes and a reference of 4. At 48 fps the round's second tile
holds 26, and at 96 fps a tile holds 34. `kvad::video::dfr_frames` counts
the largest call from the canvas and the tile plans, which now live in the
`kvad` crate so that a request is checked before it is queued. A DFR clip is
admitted when that call is no more than the fast pipeline's largest at the
same size. That is conservative: DFR drops the upsampler before stage 2,
and its measured peaks sit below the fast pipeline's at the same tokens
(below). The web form counts the same way.

Measured through the service on an M5 Pro, as requests: "A red fox trots
through fresh snow in a pine forest at dawn…", seed 1. Peak footprint is the
server's (`footprint`):

| Clip | Text | Denoise | Decode | Peak |
|---|---|---|---|---|
| 768×512, 121 frames at 24 fps | 12.7 s | 113.6 s* | 18.3 s | 26 GB |
| 768×512, 241 frames at 48 fps | 9.1 s | 212.3 s | 38.8 s | 27 GB |
| 768×512, 481 frames at 96 fps | 9.0 s | 533.5 s | 78.9 s | 28 GB |
| 1536×1024, 145 frames at 48 fps | 10.6 s | 975.9 s | 106.2 s | 30 GB |
| 768×512, 97 frames at 48 fps, from a picture | 9.9 s | 102.9 s | 12.5 s | — |

\* The first DFR request also fuses and quantises the detailing DiT's q8
cache, 20 GB on disk.

For comparison, the fast pipeline makes the first clip in about 77 s and
peaks at 27 GB, and 1536×1024 × 121 at 31–35 GB. The 96 fps clip's second
round is most of its time. Each of its four tiles denoises three segments
and a fourth of lead-in, the reference's own layout, which its docs warn
grows much faster than the four steps suggest.

- **The clips are coherent** at every rate: the fox trots towards the
  camera. At 48 and 96 fps the new frames are real in-betweens. Frame to
  frame differences are even, not alternating as duplicates would make
  them.
- **The seams:** at 48 fps there is no jump where the tiles meet (frame
  144: 1.44 against a mean of 1.53). At 96 fps the three seams of the
  second round rise a little over their neighbours (0.87–0.92 against about
  0.7), still under the clip's own largest steps (1.27–1.46).
- **From a picture** the first frame is the picture's, and the clip moves
  on from it.
- **Refusals:** a guided request at 48 fps and a rate that does not halve
  (45) are refused, each saying what to change.

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
   `ffmpeg` re-encoding is optional. Done; see below.
8. **Later, each in its own issue:**
   - the duration head as the default (done; see below);
   - image-to-video (done; see below);
   - the dev model with its guidance (done; see below);
   - the DiffVAE decoder (done; see below);
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
- **The pictures are right. The sound was not, and this called it
  plausible.** The dog runs from the waterline to the camera over the whole
  five seconds. The short clip's sound has two broadband bursts half a
  second apart. Both clips peak at full scale, and the reference's own
  decoder and vocoder give the same peak and loudness from the same latents.
  That checked the decoders and nothing before them, and nobody listened:
  the bursts were clicks, and every clip from this sampler had them until
  #167.
- **The sound was a train of clicks** (#167): broadband impulses about ten
  a second with near silence between, from the first clip made here until
  0.14.0. The stepping was the reference's, which steps the sound
  ancestrally as it does the video. The noise was not fresh. A draw's seed
  was `seed + draw·γ`, and SplitMix's `j`th number is a function of
  `seed + j·γ`, so draw `d` was draw 0 started `d` numbers late: an even
  `d` gives draw 0's noise moved `d` elements along. The sound began as
  draw 1 and step `i` added draw `3 + 2i`, the noise it began as moved
  `2 + 2i` elements along, and an audio token is 8 channels of 16 mel bins,
  so that is the same noise two more bins up at every step. (The video had
  the same fault along its 128 channels, which are not an axis of anything,
  and showed no harm that was noticed.) With the draw's number mixed before
  it is added, on an M5 Pro at q8, 768×512, two stages, from
  `ffmpeg -af astats` and a spectrogram:

  | Clip | | Peak | RMS | Crest factor | Spectrogram |
  |---|---|---|---|---|---|
  | A fox in falling snow, seed 3, 113 frames | before | 0.0 dB | −27.0 dB | 27.1 | vertical lines, black between |
  | | after | −41.5 dB | −54.8 dB | 4.3 | an even floor |
  | A retriever on a beach that barks, seed 1, 121 frames | before | 0.0 dB | −15.6 dB | 6.1 | vertical lines, black between |
  | | after | −1.6 dB | −18.3 dB | 6.8 | eight low bursts with harmonics over a steady floor of surf |

  The levels tell the two apart only where the scene is quiet; the
  spectrogram does for both. The dog's clip was made on the guided
  pipeline and on DFR as well, with bursts and a floor like these on both,
  and all three were listened to afterwards, on 7 October 2026. A seed's
  clip is another clip than before, since every draw but the first changed.
  `ltx_sample::tests::the_sound_of_a_fast_clip_is_not_a_train_of_clicks`
  is the check, on the weights of the machine: 5.8 with the fix and 11.1
  without.
- **The reference moved, and Kvad followed.** Since its 1.4.0 (29
  September 2026; read at 1.4.2, `9ec55f9`) the reference samples with
  Euler ancestral, η 1 and noise scale 1, wherever a distilled LTX-2.5
  checkpoint denoises: the distilled pipeline's stage 2 as well as its
  stage 1, and DFR's stages 1 and 2 and its spatial epilogue. Each pass
  seeds its noise apart (`seed + 10000`, `+ 20000`, `+ 30000`). Its 1.3 had
  said of stage 2 that "its 3-step refinement schedule is too short to
  remove freshly injected noise"; the changelog gives no reason for the
  change of mind, only that the output differs. DFR's temporal tiles stay
  at η 0.5, and `ti2vid_two_stages`, the dev model's, stays Euler.

  Kvad's `refine` and DFR's `first` and `second` take an η now:
  `ltx_sample::ETA`, 1, for the distilled model, and 0 for the dev model's
  stage 2. Three steps are two that add noise and the prediction. Each
  draw has a stream of its own: 102 to 105 in `refine`, after the two that
  re-noise, and the next of the request's counter in DFR. There is no
  spatial epilogue here to change.

  Before and after, on an M5 Pro at q8, 768×512 × 121 at 24 fps, same
  prompt and seed, from `ffmpeg -af astats`, a `showspectrumpic`
  spectrogram, and the frames looked at:

  | Clip | | Peak | RMS | Crest factor (L, R) | Spectrogram | Edges |
  |---|---|---|---|---|---|---|
  | Fast, a fox in falling snow, seed 3 | Euler | −12.6 dB | −30.0 dB | 8.1, 6.4 | an even floor, no lines | 22.0 |
  | | ancestral | −16.3 dB | −31.4 dB | 5.9, 4.8 | the same | 20.1 |
  | Fast, a retriever on a beach that barks, seed 1 | Euler | −0.1 dB | −19.5 dB | 9.4, 9.3 | three bursts with harmonics over surf | 33.8 |
  | | ancestral | −0.8 dB | −20.2 dB | 9.1, 9.3 | the same three, at the same times | 31.7 |
  | DFR, the fox, seed 3 | Euler | −7.4 dB | −23.7 dB | 5.4, 5.5 | a low floor that steps up twice | 27.1 |
  | | ancestral | −10.9 dB | −25.4 dB | 4.6, 4.8 | a low floor that steps up three times | 28.9 |

  "Edges" is the mean of a Sobel filter over every frame's luma, a rough
  figure for detail.

  - **The sound is not harmed.** No clip has the vertical broadband lines
    of a click train, and no crest factor rose. That was the worry: an
    ancestral stage 2 puts noise into the sound at σ 0.725 and 0.42 with
    one step left to remove it.
  - **The fast pipeline's picture is the same picture**, since stage 1 is
    unchanged: the same fox, the same turn of its head, the same dog and
    waves. It is a little softer, 9% and 6% fewer edges and a latent 3%
    smaller (RMS 0.906 to 0.875, and 1.006 to 0.978), which shows at 2× on
    the fox's fur and not at full size. Nothing is broken and nothing is
    plainly better.
  - **DFR's is another clip**, since its stage 1 changed: a closer, greyer
    fox with more texture where Euler made a small pale one by a tree. One
    pair says nothing about which pipeline makes the better clips.
  - A seed's clip is another clip than before in both pipelines.

  So this follows the reference because it is the reference, and because
  nothing measured says not to: it is not a measured gain. One prompt for
  DFR and two for the fast pipeline, and nobody listened.

  `examples/ltx_dfr.rs` still checks DFR's stages at η 0: its fixtures are
  the reference's 1.3, and `scripts/ltx-fixtures.py` is written against
  1.3's modules (`dfr_layout`, the helpers in `dfr_pipeline`), which 1.4.0
  moved. The ancestral step it would check is the one the rounds' check
  already covers. Porting the script is not done.
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
  steps, sound included. (Ancestral steps since the reference's 1.4.0: see
  "The reference moved" above.)
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
- **1536×1024 peaks in stage 2's own working set**: the DiT's 22.8 GB and
  about 13 GB of activations at 24 576 tokens, 35–37 GB, once with a spike
  to 39 GB as stage 2 started. The upsampler is already dropped and
  synchronised before stage 2. Its memory comes back 0.1–0.3 s after the
  synchronise, not at it, which may be what the spike overlapped. Shrinking
  the upsampler does not lower this peak. Shrinking the DiT's activations
  would. Up to 39 GB of 48 is the pipeline's tightest point.
- **The upsampler was larger than it needed to be** (3345a2b, in #79).
  `Conv3d::load` folded each weight on the device, so the upload stayed
  allocated beside the folded copy until a synchronise, and the copy was
  rounded up to a power of two: 4.5 GB for 2 GB of f32 weights. Folded on
  the host it is 2.5 GB, and with a synchronise after each block the
  upsampler peaks at about 6 GB instead of 8. 768×512 in two stages peaked
  while it ran beside the DiT, and now peaks at 27.1 GB instead of 29.2. The
  video decoder shares the loader: 0.55 GB less to load, 0.3 GB more at its
  peak (9.0 against 8.7 GB at 768×512), at the same speed and output.

**Profiling stage 2 and the decode (#84).** `kvad_gpu::prof` synchronises
around each labelled piece when it is on, and costs one atomic load when it
is off. `examples/ltx_cost.rs` runs the first blocks of the DiT at q8 on
noise at stage 2's shape. `ltx_decode --profile` times the decoder's steps.
- **A stage-2 block takes 3.96 s** at 24 576 video tokens, against 4.04 s
  a block in a real step. It does about 23.7 TFLOP, so 5.9 TFLOP/s.

  | Where | s/block | Share |
  |---|---|---|
  | Video self-attention, the attention itself (candle's `sdpa`) | 1.77 | 35% |
  | Projections, all streams | ~1.4 | ~30% |
  | Element-wise: norms and RoPE, modulation, residuals, GELU, gates | ~1.5 | ~30% |
  | Text attention, audio, the rest | ~0.3 | ~6% |

  The projections run on the neural accelerators (`Proj::Blocks`), at
  8–11.5 TFLOP/s. The attention runs on candle's fused kernel, MLX's, at
  5.6 TFLOP/s on the ordinary ALUs. The element-wise work was chains of
  small candle ops with casts to f32 and back between them: the q/k norms
  and RoPE alone took 0.33 s for about 2 GB of traffic.
- **The 1536×1024 decode takes 282 s**, and 86% of it is three residual
  stages: 512 channels at 2.6 TFLOP/s, 256 at 1.6 and 128 at 0.9. The rate
  falls with the channels because candle's `conv2d` builds an im2col copy
  the size of the frame while the multiply's work per byte shrinks. That is
  #56's finding at full size, and a convolution that reads neighbourhoods
  straight from its input is its fix.

**Fusing the DiT's element-wise work (#85).** Five Metal kernels in
`video/ltx_fused.rs`, each reading its inputs once, computing in f32 and
rounding once: `modulate` (every modulated norm), `gated_modulate` (a gated
residual and the norm after it, from one buffer), `gated_add`, `norm_rope`
(the q/k norm and each head's rotation, read straight from a q8
projection's f32 answer) and `gelu`. Anything they can't take falls back to
the chains, as does everything under `KVAD_GPU_FUSED=0`.

| | Chain | Fused |
|---|---|---|
| A stage-2 block (2 rounds each) | 3.68 / 3.70 s | 2.88 / 2.99 s |
| 768×512, 2 stages, all told (2 runs each) | 231.5 / 232.8 s | 187.7 / 188.6 s |
| 1536×1024, 2 stages, all told (1 run each) | 1085 s | 892 s |
| its stage-2 steps | 181 s | 138–144 s |
| its peak footprint | 38.3 GB | 33.3 GB |
| Velocity, bf16 against f32, video / audio | 44.2 / 44.5 dB | 44.7 / 44.9 dB |

- **Rounding once brings bf16 closer to f32** after every block, by 0.4–1.4
  dB. kvad's f32 DiT agrees with the reference to 115–123 dB, so it stands
  in for it (`ltx_cost --accuracy`).
- **The clips are the same scene with the same motion**, 30 dB PSNR apart
  at 768×512 and 33 dB at 1536×1024, mostly in fine detail. The chain run
  twice gives identical clips, so the difference is the rounding's.
- **1536×1024 peaked at 33.3 GB**, from one run. The chain's peak ranged
  35.8–39.0 GB over three runs. *Corrected with #86:* two later runs with
  these kernels peaked at 35.1 and 35.3 GB, so 33.3 was a low reading. The
  kernels lower the peak by a GB or two, not five.
- **Attention is now 48% of a stage-2 block.** An attention kernel on the
  neural accelerators is the largest lever left in the DiT, and the decode's
  convolutions (#56) the largest outside it.

**Attention on the neural accelerators (#86).** A flash-attention kernel
in `mpp_attention.rs` does both of attention's products through
`matmul2d`, and reads q, k and v in the projections' own layout, so the
copies that split the heads are gone. It is laid out as MLX's M5 attention
is: 64 queries a threadgroup in four SIMD groups, 32 keys a step, an online
softmax in powers of two, and one rounding at the end.

| At stage 2's shapes, bf16 | candle | This |
|---|---|---|
| Video self-attention, 24 576 × 24 576 | 6.5 TFLOP/s | about 16 (2.5×) |
| Video to text, 24 576 × 1024 | 3.1 TFLOP/s | about 19 (6×) |
| Between video and audio | 0.6–0.8 TFLOP/s | 16–19 (20–25×) |

| Same binary, `KVAD_GPU_MPP_ATTENTION=0` for candle's | candle's | This |
|---|---|---|
| A stage-2 block (2 rounds each) | 2.76 / 2.78 s | 1.71 / 1.85 s |
| 768×512, 2 stages, all told (2 runs each) | 186.3 / 190.4 s | 158.7 / 162.3 s |
| 1536×1024, 2 stages, all told (1 run each) | 936.9 s | 712.7 s |
| its stage 1 / stage 2 / decode | 167 / 448 / 297 s | 124 / 281 / 284 s |
| its peak footprint | 35.1 GB | 35.3 GB |

- **Two things made it fast.** Every index into an array of matrix
  fragments has to be a compile-time constant: one the compiler cannot
  resolve puts the array in memory, and the first version ran at 3.3
  TFLOP/s. And a barrier each step keeps the SIMD groups sharing K and V
  in the core's cache: 9.6 TFLOP/s without it.
- **Accuracy is unchanged.** The kernel agrees with attention written out
  in f32 to 55–58 dB in bf16, where candle's gives 44–49. Through the DiT
  the velocity is 44.7 / 44.9 dB either way. The clips are the same scene,
  28–30 dB apart, the rounding difference #85 showed too.
- **MLX's own M5 attention runs the self-attention at 21.5 TFLOP/s.** With
  every key served from cache this kernel reaches 26. *Resolved in #87,
  below: the gap was not in how K and V arrive.*
- **1536×1024 now takes 12 minutes, and the decode is as large as stage 2.**
  #56's convolutions are as big a lever now as anything in the DiT.

**Level with MLX, and faster matmuls (#87).** Staging K and V in
threadgroup memory was slower: 11 TFLOP/s copying then computing, and 7
with a register prefetch. MLX's kernel, compiled from its source in kvad's
harness and changed a piece at a time, showed what was:
- **The threadgroup's shape.** Dispatched 128 × 1 instead of 32 × 4, the
  same threads in the same SIMD groups, MLX's kernel fell from 21.4 to 15.6
  TFLOP/s, which was #86's rate.
- **Loading fragments element by element**, not as vectors copied out:
  16.2 → 18.6 TFLOP/s with the right shape.

The attention now runs stage 2's self-attention at 20.2 TFLOP/s, 3.2×
candle and level with MLX's 19.4–19.8, raced in turn. `mpp.rs`'s matmuls
were dispatched the same way, and 32 × n runs them faster too: Q8_0 by
1.14–1.22× and bf16 by 1.09–1.31× at stage 2's shapes.

| Against #86's build, alternated | #86 | #87 |
|---|---|---|
| A stage-2 block | 2.01 / 2.10 s | 1.58 / 1.76 s |
| 768×512, 2 stages, all told (2 runs each) | 183.0 / 170.0 s | 169.2 / 165.0 s |
| 1536×1024, 2 stages, all told (1 run each) | 803.4 s | 661.4 s |
| its stage 1 / stage 2 steps / decode | 132 s / 106–124 s / 299 s | 113 s / 79–80 s / 284 s |

- **The output is byte-for-byte the same as #86's** at both sizes: neither
  change touches the arithmetic.
- **The machine ran slower through this session:** #86's own 1536×1024 run
  took 713 s earlier and 803 s here. The ratios are what compare.
- **At 1536×1024 the decode is now 43% of the time**, 284 s against stage
  2's 240. #56 is the largest lever left.

**The decoder's convolutions on the matrix units (#88, for #56).** The
Decoding section planned 3×3×3 convolutions as 2D ones over neighbouring
frames, and #57 built them that way on candle's `conv2d`. They were exact,
but three quarters of their time went to `conv2d`'s `im2col` copy, and the
residual stages ran at 0.9–2.6 TFLOP/s. `mpp_conv3d` is an implicit-GEMM
convolution instead: the kernel, stored `[out, 27·in]`, times the input's
shifted neighbourhoods, which each threadgroup gathers into threadgroup
memory 32 channels at a time as it multiplies. Nothing the size of
`im2col` is built. The pixel norm and SiLU before each convolution became
one kernel too (`ltx_fused::norm_silu`).

| Decode, bf16 | Before | Convolution kernel | Both kernels |
|---|---|---|---|
| 768×512 (2 runs each) | 69.85 / 69.84 s | 12.16 / 12.16 s | 7.79 / 7.80 s |
| 1536×1024 (1 run each) | 278.4 s | 45.5 s | 31.3 s |
| its peak footprint | 24.0 GB | 19.1 GB | 18.3 GB |

| Against `master`'s build, alternated | `master` | #88 |
|---|---|---|
| 768×512, 2 stages, all told | 160.2 s | 91.3 s |
| 1536×1024, 2 stages, all told | 662.5 s | 432.0 s |
| its decode | 281.3 s | 35.6 s |

- **Per convolution it runs at 15–18 TFLOP/s**, 7–17× the folded
  `conv2d`, and agrees with it to the output's rounding: 55.6 dB in bf16,
  73.7 dB in f16. The folded convolution still runs everywhere else,
  including the f32 latent upsampler.
- **The clips are 54.4–54.7 dB PSNR from before**, and the latents are
  byte-for-byte the same: only the decode changed. #88's 1536×1024 run
  drifted about 20 s in stage 2, which it does not touch.
- **Against the reference:** `ltx-core` decoded 768×512 in 5.2–5.5 s on
  MPS (#56). kvad now takes 7.8 s, where it took 68.8.
- **Memory:** the decode's peak no longer comes near the DiT's. At
  768×512 it is 8.8 GB, below the reference's 11.2, and a generation's
  peak is stage 2's.
- **Where 1536×1024's 7 minutes go now:** stage 2 about 240 s, stage 1
  110 s, the decode 31–36 s. The DiT is the lever again, and the service
  (step 7) is the step left.

**Stage 2's projections at MLX's rate, and the gates inside attention.**
Profiled again on `master`, a stage-2 block took 1.56 s. Its q8 projections
ran at 10–12 TFLOP/s, against 18 for the feed-forward's. Alone, the Q8_0
kernel ran every shape at about 21. The rest went to what came after it:
the kernel answered in f32, then candle added the bias and cast to bf16 in
passes of their own. At `[24576, 4096] × [4096, 4096]` that was 65 ms
against the kernel's 39. The per-head gates, `2·sigmoid(g)` over the whole
attention output, took another 0.13 s, 9% of the block.
- **The store finishes each sum.** It adds the bias, takes the feed-forward's
  GELU where asked, and rounds once to the dtype asked for. Nothing reads
  an f32 answer back, and the 16 384-wide GELU pass is gone. The sums are
  the same as before, bit for bit.
- **The kernel is laid out as the attention kernel is.** Each SIMD group
  keeps a `32 × 32` corner of the tile as `16 × 16` fragments in registers,
  loads its rows of the activations element by element, and multiplies on
  the matrix unit `16 × 32 × 16` at a time. The weights are unpacked into a
  padded slab in threadgroup memory as before. It reads bf16 as it comes
  and rounds it to f16 in registers, so the f16 copy of every input is
  gone too. The attention kernel and this one now share the fragment code.
- **Attention applies the gates** to each row before it rounds. The gates'
  logits stay in f32 from the q8 projection.

| Q8_0, bias, bf16 in and out, `[24576, k] × [k, n]` | Before | Now | MLX 8-bit |
|---|---|---|---|
| 4096 → 4096 | 12.6 TFLOP/s | 23–25.5 | 26.1–26.4 |
| 4096 → 16 384 | 13.1 | 23.5–25.6 | 24.5–25.1 |
| 16 384 → 4096 | 17.1 | 20.7–24.4 | 21.8–21.9 |
| 4096 → 2048 | 11.5 | 21.7–24.8 | 22.8 |
| 2048 → 4096 | 8.5 | 20.7–23.2 | 22.4–22.8 |
| 4096 → 32, the gates' logits | 1.7 | 5.4–6.0 | – |

MLX's numbers are its `quantized_matmul`, 8 bits in groups of 32 or 64, on
bf16 input, in the same session. Stepping 64 along `K` at a time, as MLX
does, ran at 21 here against 25. A `128 × 64` tile over eight SIMD groups
was no faster.

| Against `master`'s build, alternated | `master` | This |
|---|---|---|
| A stage-2 block (`ltx_cost`) | 1.56 s | 1.09 s |
| 768×512, 2 stages, all told | 101.7 / 106.6 s | 78.7 / 79.0 s |
| its stage 1 / stage 2 | 28.6–31.2 / 44.7–45.8 s | 18.0–20.7 / 30.3–31.7 s |
| 1536×1024, 2 stages, all told (1 run each) | 458.4 s | 368.5 s |
| its stage 1 / stage 2 | 125.4 / 269.1 s | 87.0 / 214.3 s |
| its peak footprint | 35.3 GB | 33.0 GB |

- **A 1536×1024 step gains less than a 768×512 one** (1.26× against 1.47×).
  Attention grows with the square of the tokens and was already fast, and at
  24 576 tokens it is now 40% of a block.
- **Accuracy:** bf16 against f32 through two blocks is 44.7 / 44.9 dB at the
  velocity, as before, and 0.1 dB closer after each block from rounding the
  gated output once. Each build gives byte-identical latents run to run.
  Between builds the clips are 31 dB PSNR apart at 768×512 and 35 dB at
  1536×1024, the same scene and motion. That is the rounding difference
  #85 and #86 showed.
- **The peak** is one run each, and has moved by two GB between runs of one
  build before. The f32 answers no longer exist, so a lower peak is
  plausible, but this does not show it.
- **The element-wise kernels are not worth rewriting yet.** In isolation
  they move 78–134 GB/s against 127–154 for candle's plainest ops, and all
  of them together are 3–4% of a block.
- **The tests passed without the kernels.** A source that failed to compile
  left `mpp::available` false, and every test skipped. A test now asserts
  that the kernels build wherever the GPU has matrix units.
- **Where 1536×1024's 6 minutes go now:** stage 2 about 215 s, of which
  attention is about 40%; stage 1 about 87 s; the decode about 40 s.

**The service (step 7).** LTX-2.5 is now a model the server loads and a
client asks for a video, the way it asks an image model for a picture.

- **A peer, not a mode:** `kvad::video::Director` beside
  `kvad::image::Painter`, `Model::Video`, `Cmd::Film` and its events, and a
  third `Kind`, `video`, which every route that picks a model by kind
  refuses or offers by name. `kvad-gpu`'s `video::ltx::Ltx` is
  `examples/ltx.rs` phase for phase. The repo has no `model_index.json`, so
  `hub::pipeline` knows it by its DiT's file and names it `LTX2Pipeline`, as
  diffusers names LTX-2's pipeline; it lists as a runnable video model, and
  a load fetches only the five files a generation reads.
- **Nothing is kept between generations.** The plan was to keep the DiT and
  reload the text path. Measured on `master` before this, from the q8 caches:
  the text path loads in 8.9 s and the DiT in 3.5 s, of 75 s for 768×512 ×
  121 all told. So the model holds paths and loads each phase as it gets
  to it. A load builds the two q8 caches if they are missing, so that the
  first generation does not spend two minutes quantising.
- **Admission charges a generation's peak**, since that is what the next
  request needs, not what is held: `24.2 GB + 58.2 bytes × w·h·frames`, the
  line through 768×512 × 121 at 27.0 GB (the run above) and 1536×1024 × 121
  at 35.3 GB, the higher of the two peaks measured there (#88; #89's was
  33.0). A request past what was measured, 121 frames and 1536×1024 × 121
  pixels × frames, is refused before it is queued, and so is one past what
  the machine's usable memory holds by that line. A larger clip has never
  run here, and a Metal allocation past memory reboots the Mac.
- **`/v1/videos` is OpenAI's job shape**, checked against their published
  spec: `POST` answers at once with a `queued` video, `GET /v1/videos/{id}`
  reports `status` and `progress`, `/content` is the MP4 and
  `?variant=thumbnail` its middle frame, as a PNG where OpenAI's is a WebP.
  OpenAI's SDKs send the request as `multipart/form-data`, with or without
  a file in it, so that is read by hand beside JSON. Beside their fields:
  `frames` (or `num_frames`), `fps`, `seed`, `audio: false`. A length in
  seconds becomes the nearest 8k + 1 frames. `input_reference`, a negative
  prompt, guidance and steps are refused by name. Every endpoint in the spec
  is marked deprecated, and the SDK warns that the Sora API was to shut down
  on 2026-09-24; the shape is still the only one its clients know.
- **Checked with OpenAI's own Python SDK** (3.19.2): `videos.create`,
  `retrieve`, `download_content` for both variants, and `list`, unchanged,
  against this server. A 4 s clip at 768×512 came back in 62.7 s, and
  `ffprobe` reads 97 H.264 frames and FLAC sound.
- **Progress is weighted by time**, from the measured cost of each part,
  since stage 2's steps are four times stage 1's at 768×512 and the loads
  and the decode are a quarter of the total. In that run it read 15% at
  10 s, 43% at 28 s, 83% at 52 s, done at 62.7 s: within 3 points of the
  share of time spent.
- **Stored as images are:** a `videos` table and `videos/<id>.mp4` with a
  poster PNG. The row is written when the video is asked for, and is the
  job's state as well as its record. A restart marks what was running as
  failed.
- **Range requests are answered by hand**, streamed from the file a
  megabyte at a time. Chromium seeks: a `<video>` in the new page reached
  3 s with `readyState` 4 on 206 answers.
- **Deleting a video stops it, promptly.** Measured with three queued: the
  first, deleted in stage 1, and the second, deleted while queued, then the
  third to finish. It finished at 89 s at first, because a queued video
  only noticed its deletion at its first step, and a running one stopped
  two steps after it. Now the task that watches a video asks every second
  whether its row is there, the scheduler skips a video nobody is waiting
  for, and the drain sets the engine's cancel flag as soon as the watcher
  is gone rather than at the next step nobody received: the third finished
  at 49 s. A generation that stops mid-stage now synchronises the device as
  it returns. Without that, the DiT's 20 GB stayed in candle's pool, and
  the next video's text phase took 15 s rather than 10.
- **`kvad videos ls|make|show|get|rm`**: `make` waits and writes the file,
  and says that stopping the wait does not stop the video. **The Videos
  page** has a size, a length, sound, a seed, a gallery of players with
  posters, and progress with a rough time left, and it says that a video
  holds the GPU for its whole length.
- **1536×1024 × 121 through the server** took 328 s (text 9.3 s, denoise
  278.5 s, decode 38.6 s), for a 288 MB file. The server's peak footprint
  over its whole life, the load and the generation, was 33.5 GB, under the
  35.3 GB admission charged. One run.
- **Not done:** image-to-video. Compression, a preview, progress as a
  stream and image-to-video came after; see below.

**Compressed with `ffmpeg`, when there is one.** The MP4 `kvad::video` writes
compresses nothing: 14 MB a second at 768×512, 57 at 1536×1024. Each video
is now re-encoded as the reference writes its own, H.264 at CRF 19 and AAC,
with the BT.709 tags and the index in front.

| Re-encoded on an M5 Pro | Time | Size | PSNR from the original |
|---|---|---|---|
| 768×512 × 97 | 0.3 s | 58.3 → 1.9 MB | 43.6 dB |
| 1536×1024 × 121 | 0.9 s | 287.9 → 9.1 MB | 44.2 dB |

- **So the uncompressed file is not kept.** A thirtieth of the size for
  under a second, at 44 dB, is not a trade anybody would decline per video.
  If `ffmpeg` fails, the uncompressed file is kept, and the failure logged.
- **`[videos] ffmpeg`** in `kvad.toml`: `auto` (the default), `off`, or a
  path. `auto` looks on `PATH` and then in `/opt/homebrew/bin`,
  `/usr/local/bin` and `/usr/bin`. The installed launchd service runs with
  `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, where a `PATH` lookup alone never
  finds Homebrew's `ffmpeg`. Started with that `PATH`, the server found it,
  and said so as it started.
- **Through the server:** a 5 s clip at 768×512 was served at 1.7 MB, with
  121 H.264 High frames and AAC sound, and Chromium seeked in it to 4 s and
  played on.

**A rough preview while a video is made.** After each denoising step, the
DiT's prediction of the clean video has its middle latent frame mixed
straight into colour by a fixed 128 × 3 matrix, as the image models'
previews are. That is 12×8 pixels in stage 1 at 768×512 and 24×16 in
stage 2. The server keeps the latest as `videos/<id>.preview.png`, serves it
as `/content?variant=preview` while the video is in progress, and drops it
when the video is done; the Videos page shows it, scaled up smoothly.

- **The matrix is fitted here**, by least squares from the final latents of
  three 768×512 × 121 clips (a fox in snow, a city street at night in the
  rain, a dog on a beach at sunset) to their decoded frames, each averaged
  over the 32×32 pixels and 8 frames a latent cell covers. Fitted on two and
  scored on the third, it explains 84–96% of the variance in each of red,
  green and blue, 22–26 dB from the pooled frames; on all three, 98%. Ridge
  regression did no better on the clip left out.
- **On a fourth scene, not in the fit** (a red balloon over hills and a
  lake), the previews score 26.9–27.0 dB against the finished clip's middle
  frame, pooled to their size: at the end of stage 1, after stage 2's first
  step, and after its last. The first preview, after stage 1's first step
  and about 15 s in, already shows the balloon, the sky, the hills and the
  lake where the finished clip has them.
- **It costs nothing that was not already paid:** reading the frame back is
  the synchronise each step's progress report already made.

**Progress as a stream.** OpenAI's shape is polled, and stays. Beside it,
`GET /v1/videos/{id}/events` is kvad's own: server-sent events carrying the
same video object, once as it stands and again at every change, ending with
`video.completed`, `video.failed` or `video.deleted`. A step can be most of a
minute from the next, so the page's poll every two seconds asked the list
thirty times for each answer that changed.

- **Each video being made has a `broadcast` channel carrying nothing.** The
  task making it tells the channel after every change it writes, and a
  watcher that hears it reads the row again, so the row stays the one
  account of a video and a watcher that falls behind skips to the latest.
  The channel is made before the generation starts and closed when its task
  ends, which a watcher takes as the last word. A watcher subscribes before
  it reads the row, so a change between the two is not lost.
- **Traced on the test server:** events at the start and after each step,
  and a deletion in stage 1 reached the watcher in the same second. A
  finished video's stream is one `video.completed` event.
- **The Videos page** fetched the list once and followed the one video being
  made on its stream, loading each step's preview as its event arrived. The
  time left ticks on a clock of its own, since the events come once a step.
- **`kvad videos make`** follows the stream, and `kvad videos watch ID`
  follows one again after the wait was stopped.

**Image-to-video.** A video can start from a picture: OpenAI's
`input_reference`, which becomes the first frame. The reference's distilled
pipeline does it with a `VideoConditionByLatentIndex` at frame 0 and
strength 1, and so does kvad, piece for piece:

1. **The picture is put through one frame of H.264** at CRF 18, the value
   the reference's `detect_params` gives a checkpoint of 2.4 or later (33
   before). LTX was trained on frames of compressed video, and a picture
   that is too clean does not look like the frames after it.
2. **It is scaled to cover each stage's size and cut from the middle**:
   bilinear, `align_corners=False`, no antialiasing, as the reference's
   `resize_and_center_crop` calls torch; from the picture itself for each
   stage, not from stage 1's copy.
3. **The video VAE's encoder** makes it one latent frame, `[128, 1, h, w]`,
   in bf16 as the reference runs it.
4. **Its tokens are held**: they start as the picture instead of noise, and
   after every update they are put back, as the reference's
   `post_process_latent` does with a denoise mask of 0 there. In stage 2
   the picture encoded at the full size replaces the upsampled first frame.
5. **The DiT modulates them at σ = 0.** The reference evaluates σ per token,
   `mask · σ`, and this is the one place that matters. What reads a token's
   σ — the nine adaLN rows, the audio–video scale and shift, the output
   head — gets a second set of rows at σ = 0, which the first `h·w` tokens
   read. What reads the stream's σ as a whole — the text keys and values,
   both audio–video gates — stays at the step's σ, as in the reference. The
   fused kernels take a split point, so the residual stream is not cut and
   joined around each norm.

What was measured against the reference (`scripts/ltx-fixtures.py
--picture`, `--contexts random`; `examples/ltx_encode.rs`,
`examples/ltx_dit.rs --held`):

| Piece | kvad | The reference's own bf16 |
|---|---|---|
| Scaling and cutting, 1000×700 to 384×256 and 768×512 | 150 dB, 0.00 of a level apart | — |
| Encoder, f32 on the CPU | 105.9–108.0 dB | — |
| Encoder, f32 on Metal | 105.6–107.6 dB | — |
| Encoder, bf16 on Metal (0.15 s at 768×512) | 37.4–41.0 dB | 37.0–40.1 dB |
| DiT, first frame held, f32 on the CPU | 107.4–118.3 dB | — |
| DiT, first frame held, bf16 on Metal | video 47.1/46.0, velocity 42.9 dB | 46.9/46.3, 43.7 dB |

- **The held check tests what it says.** The reference's two fixtures,
  held and not, differ by 2.1 dB after the first block and 13.3 dB at the
  video velocity; kvad matches the held one to 107–118 dB in f32.
- **candle's Metal reductions are wrong over five axes.** The encoder's
  space-to-depth residual averages channel groups, and as a mean over the
  middle of a `[t, c, g, h, w]` tensor it was 1e38 out on Metal (mean, sum
  and max alike) and right on the CPU: −27 dB, then NaN. At four axes,
  `[t·c/g, g, h, w]`, it is exact.
- **The encoder is causal**, unlike the decoder: each convolution reads
  its frame and the two before it, with the first frame standing in for
  those before the clip (`Time::Causal`; on the M5's matrix units, the same
  gather one frame earlier). For a picture that changes nothing: every tap
  reads the one frame. A whole clip (#73) is encoded the same way:
  `VideoEncoder::encode_clip` on a 384×256×25 clip from
  `scripts/ltx-fixtures.py --clip` is 108.4 dB from the reference in f32
  on the CPU and 107.9 on Metal; in bf16 on Metal 36.9 dB, where the
  reference's own bf16 on MPS is 35.9. The first 9 frames alone make the
  whole clip's first two latent frames, to 346 dB. With the decoder's
  padding instead it is 0 dB. 121 frames at 768×512 take 10.2 s in bf16,
  with an 11.6 GB peak footprint.
- **The H.264 round trip is `ffmpeg`'s** (`kvad::video::picture_from_file`),
  which also reads the picture: kvad writes PNGs and has no decoder, and
  the round trip needs an encoder anyway. On a 1000×700 picture it lands
  48.2 dB PSNR from the reference's own round trip (by PyAV), where the
  round trip itself moves the picture 37.8 dB from what it was. That is as
  close as the reference comes to itself: PyAV lets x264 cut a frame into a
  slice per core, and its answer on one core is 50.1 dB from its answer on
  eighteen; the two x264 builds, each on one core, are 49.1 dB apart. At
  `ffmpeg`'s default 25 fps rather than the reference's 1, it was 43.9 dB:
  the frame rate moves the rate control even for one frame.
- **`ffmpeg` turns a JPEG by its EXIF orientation**, as the reference does,
  in the same direction (checked with orientation 6, ffmpeg 9.0.2). It does
  not convert an ICC profile to sRGB, which the reference does.

End to end, on an M5 Pro, from a 1024×768 picture (a red car in a street,
made by FLUX.1-schnell here):

- **768×512 × 121** took 77.8 s, with a peak footprint of 26.8 GB, where
  text-to-video's was 27.0. The picture, both encodes and the `ffmpeg`
  round trip, took 1.0 s. The clip's first frame is 29.0 dB PSNR from the
  picture as scaled and cut (the VAE's reconstruction, after H.264), and
  its ninth 17.4 dB: the picture is held, and the clip moves on from it.
- **1536×1024 × 121** peaked at 31.3 GB, under the 35.3 GB admission
  charges. The picture took 1.5 s. One run, 366 s, with a build running
  beside part of it.
- **Through the server:** `input_reference` as a file in a form (the page,
  as OpenAI's SDK sends one) and as a `data:` URL in JSON (`kvad videos
  make --image`). A URL elsewhere, a `file_id`, any other file, and a file
  that is not a picture are each a 400 with its own sentence; `ffmpeg`'s
  words, which name a path in the data directory, go to the log. The
  picture is kept as sent, as `videos/<id>.input`, served as
  `/content?variant=input` and linked as `kvad.picture_url`, and deleted
  with the video. A 4 s clip at 768×512 through the CLI: picture and text
  11.4 s, denoise 44.0 s, decode 10.8 s.
- **The page** takes a picture beside the prompt and says what the cut will
  take (a 4:3 picture for a 3:2 video loses 11% of it, top and bottom). It
  shows the picture in place of the preview until the first step has one,
  marks a video made from a picture in the gallery, and "Reuse" fetches the
  picture back into the form.

**The duration head as the default.** A request that gives neither
`seconds` nor `frames` now gets the length the model chooses from the
prompt, as the reference's pipelines do with their `AutoDuration`: the
duration head reads the two contexts the text path has just made and
predicts seconds, which become `round(s · fps)` frames, clamped to 1–20 s,
floored to 8k + 1.

- **Here the clamp's top is lower**: the longest clip the model makes here
  at the request's size, 121 frames at 768×512 and at 1536×1024 on this
  machine. A request is checked, and admitted, at that longest, since that
  is the most the model may choose; a size that leaves room for less than a
  second is refused, since the head never chooses less.
- **The length is known after the text phase**, not when the video is asked
  for. So the row holds 0 frames and a `chosen` flag until then; OpenAI's
  `seconds` is `null` and `kvad.frames` too, and `kvad.length_chosen` says
  the model chose it. The step after the head carries the length, and the
  row, the plan the progress is weighted by, and the shapes the DiT runs at
  are all set from it.
- **The head is optional.** It is 4 MB in its own file, which a load fetches
  as it fetches the other five; a load that cannot get it makes 121 frames,
  as before, and says so.
- **Against the reference**, on eight prompts' contexts from kvad's text
  path (`examples/ltx_duration.rs`, `scripts/ltx-fixtures.py --duration`):
  within 5e-7 of its seconds in f32, on the CPU and on Metal, and the same
  frames for all eight. The reference's own bf16 head is within 1%, and the
  same frames too. kvad runs it in f32: it is a few hundred million
  multiply-adds, and bf16 could move a prediction near a frame boundary to
  the other side.

| Prompt | Head | Frames at 24 fps (up to 121) |
|---|---|---|
| a door slams shut | 4.25 s | 97 |
| a glass falls off a table and shatters on the floor | 4.73 s | 113 |
| a fox in the snow | 4.34 s | 97 |
| the red car drives towards the camera and past it (two sentences) | 4.40 s | 105 |
| two chefs argue in a kitchen (dialogue, four sentences) | 3.88 s | 89 |
| a lighthouse at dusk, waves, gulls | 5.69 s | 121 |
| a long, slow aerial shot over mountains at sunrise | 8.17 s | 121 |
| a woman reads out a letter's worth of dialogue | 8.54 s | 121 |

- **So it mostly shortens.** Five of the eight are shorter than the 121
  frames every request used to get, and the rest are at the ceiling. "A
  door slams shut" took 63.2 s at 97 frames, where 121 took 77.8 s.
- **Through the server**, with OpenAI's Python SDK and no `seconds`: the
  video came back `queued` with `seconds: None`, read 4.71 s from its first
  stage-1 step, and finished at 113 frames, which is what ffprobe counts.
  `kvad videos` shows "4.71 s, from the prompt"; the page's Seconds field
  reads "auto" and says what the model may choose at the size in the form,
  and "Reuse" leaves a chosen length for the model to choose again.

**The dev model with its guidance.** A request that gives `steps`,
`guidance_scale` or `negative_prompt` now runs LTX-2.5's dev model, as the
reference's `ti2vid_two_stages` does; one that gives none runs the distilled
model, as before.

1. **The text path** encodes the prompt and the negative prompt (the
   reference's `DEFAULT_NEGATIVE_PROMPT`, word for word, when none is given).
   The connectors are read from the distilled DiT's file for both models:
   the dev file's are the same, all 258 tensors byte for byte, and so are
   the two configs, so the text path keeps one cache.
2. **Stage 1** runs the dev DiT at half size, from noise, in 30 Euler steps
   on `LTX2Scheduler`'s levels (shifted as for 4096 tokens, which is what the
   reference's pipelines always ask for, and stretched to end at 0.1). Each
   step predicts the clean latents four times:
   - with the prompt;
   - with the negative prompt;
   - with block 28's self-attention skipped, video and audio, its output
     the value projection alone (spatio-temporal guidance, STG);
   - with every audio–video attention skipped (modality guidance).

   Then `cond + (cfg − 1)(cond − uncond) + stg(cond − blind) + (modality −
   1)(cond − deaf)`, with CFG 3 for the video and 7 for the sound, STG 1 and
   modality 3, rescaled so that its spread is 0.7 of the way back to
   `cond`'s. The four calls run one after another, not batched, so memory at
   its largest is one call's.
3. **Stage 2** is the dev DiT with the distilled LoRA (rank 450, strength 1)
   fused in: `W + B·A` in f32, rounded to the checkpoint's bf16, then
   quantised to a q8 cache of its own. Its three unguided steps are the
   distilled pipeline's, and, as in the reference, its sound is discarded
   and stage 1's kept.

What was checked against the reference, on the first two blocks and the
heads (`scripts/ltx-fixtures.py --contexts random` and `--lora`/`--guided`;
`examples/ltx_dit.rs`):

| Piece | kvad, f32 on the CPU | kvad, bf16 on Metal | The reference's own bf16 |
|---|---|---|---|
| STG on the last block loaded | 111.7–120.1 dB | video 41.0, audio 45.5 at the velocity | 41.6, 45.8 |
| Audio–video attention skipped | 114.6–120.0 dB | 42.4, 45.3 | 43.7, 45.6 |
| The distilled LoRA fused | 114.8–120.7 dB | 44.4, 45.3 | 44.6, 45.3 |
| One guided prediction, all four passes combined | video 102.6, audio 99.2 dB | 35.6, 27.4 | 35.7, 27.1 |
| The 30-step schedule | within 3e-7 of every level | | |

- **Each check tests what it says.** Skipping block 1's self-attention
  leaves block 0 untouched and moves block 1 to −2.3 dB from the plain
  pass; skipping the audio–video attention moves the outputs by 39–52 dB.
- **A fused LoRA has to be rounded to bf16**, as the reference rounds it.
  Kept in f32 when the rest of a check read in f32, it came out 55–62 dB
  from the reference, where everything else is 110 dB and more.
- **The guided prediction is further from exact in f32** (99–103 dB) than
  one pass, because guidance multiplies the passes' tiny differences by as
  much as 7; in bf16 it is where the reference's own bf16 is.

On an M5 Pro, the lighthouse prompt at seed 1, with every cache built:

| 768×512 × 121 | Distilled | Dev, guided |
|---|---|---|
| Text | 8.7 s load | 8.7 s load, and a second prompt |
| Stage 1 | 17.8 s (8 steps) | 283.8 s (30 steps, 9.5 s each) |
| Stage 2 | 28.8 s | 34.3 s, the LoRA'd DiT's 3.7 s load included |
| All told | about 75 s | 347.2 s |
| Peak footprint | 27.0 GB | 24.6 GB |

- **1536×1024 × 121, guided**: 1486.8 s all told (stage 1 1238.9 s, 41 s
  a step; stage 2 187.4 s; decode 38 s), with a peak footprint of 31.6 GB,
  under the 35.3 GB admission charges for that size. One run, with a build
  beside part of it. The distilled model's measured peak there was 33.0–35.3
  GB; stage 1's four calls a step run one after another, so they add time,
  not memory.
- **The same seed makes the same file**: a second run was byte for byte
  the first, whose caches it built.
- **Building the caches**, once: the dev DiT loaded and quantised in 23.9 s,
  and the dev DiT with the LoRA fused in 54.5 s; 20 GB each.
- **Through the service**, the three knobs select the pipeline, and its
  files come only from `kvad pull Lightricks/LTX-2.5 --dev` (51 GB): a
  guided request before that is a 400 naming the command, and so is one to
  a model without a guided pipeline. `kvad.guided` says how a video was
  guided; the page has the three fields folded under "Guidance". Through
  the test server, `kvad videos make … --seconds 2 --steps 12 --guidance 4`
  made a 49-frame clip (text 9.9 s, denoise 74.5 s, decode 5.8 s), stored
  with `guided: { steps: 12, guidance: 4 }`, and "Reuse" put both back.
  `kvad pull … --dev` fetches the two files alone: the plain pull of LTX's
  repo looks for a model index it does not have, and a load fetches the
  distilled model's files.
