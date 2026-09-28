# Plan: LoRAs for the image and video models

Written 2026-09-28 for [#41](https://github.com/bisand/kvad/issues/41). A
LoRA is how most people change what one of these models draws: a style, a
subject, or a distillation that draws in 8 steps what the model draws in 50.
For a set of linear layers `W` it is a pair of thin matrices, `A` (r × in)
and `B` (out × r), and a scale, so that the layer computes
`W·x + scale·B·(A·x)`.

## Decided

Asked 2026-09-28:
- **Applied at run time**, as a side path beside each layer it adapts, as
  diffusers applies one unless told to fuse it. A q8 or GGUF base takes one
  unchanged. A request chooses its LoRAs and strengths with no reload, and
  no quantised cache is written per LoRA. Fusing, which LTX-2.5's distilled
  LoRA still does (`common::Lora`), would mean a reload per LoRA and
  strength, and re-quantising from bf16 for every model but SDXL and SD 1.5.
- **Named per request:** `loras: [{name, scale}]` on `/v1/images` and
  `/v1/videos`, `--lora NAME[:SCALE]` in the CLI, and a picker on the Images
  and Videos pages. Names are a checkpoint's (`repo`, `repo:file.safetensors`
  or a path). The resident model stays as it is, and admission adds the
  LoRA's bytes.
- **Qwen-Image first, with Lightning**, #41's own example and much the most
  downloaded LoRA of any model here (`lightx2v/Qwen-Image-Lightning`, 389k).
- **The common formats, model by model**, and every key in a file must land
  on a layer or the LoRA is refused, as the unread-weights guard refuses a
  model. What people publish, read off the headers of each model's most
  downloaded LoRAs on the Hub:

  | Model | Formats |
  |---|---|
  | Qwen-Image | `lora_down`/`lora_up` + `alpha` under diffusers' names (Lightning); PEFT under `diffusion_model.` or `transformer.`; kohya's underscores |
  | SDXL, SD 1.5 | kohya: `lora_unet_…` under `ldm`'s names or diffusers', `lora_te_`/`lora_te1_`/`lora_te2_` for the text encoders, some convolutions |
  | FLUX | PEFT under `transformer.`; kohya in Black Forest Labs' layout, one `qkv` where diffusers has three |
  | LTX-2.5 | PEFT under `diffusion_model.`, some with `alpha` |

  LyCORIS' kinds (LoKr, LoHa) and DoRA are refused by name.

## How it works

- **A registry of adaptable layers.** A reader can carry one; every
  `nn::Linear` loaded through it registers its full name and shape and
  keeps a slot. A request's LoRAs fill the slots it names and are cleared
  after it.
- **The side path**, `y + (x·A)·B`, added after the layer's own answer and
  bias, as PEFT adds it. A layer with a LoRA set declines the fused GELU
  its kernel could store, so the GELU comes after the LoRA's term.
- **Half-precision factors:** the pipeline's dtype where it is half, and
  bf16 where it runs in f32 beside quantised weights, as Qwen-Image at q8
  does. The scale is folded into `B` once, in f32, as the LoRA is set.
- **Added in place:** on the M5, `(x·A)·B` is summed straight into the
  layer's answer by the dense kernel's store (`mpp::dense_acc`), into f32
  where the answer is f32. Separately, the product, its scaling, its cast
  and the sum were four passes over the answer, and most of the time.
- **Names:** a file's module names are matched against the registry's after
  their prefix (`transformer.`, `diffusion_model.`, `lora_unet_`) is taken
  off, as written or with kohya's underscores for dots. Where a layout is
  not the model's own (`ldm`'s, Black Forest Labs'), a map translates it
  first, as `single.rs` does for checkpoints.
- **The scale** is `alpha / rank` where the file gives `alpha`, and one
  where it does not, times the request's strength.

## Lightning's schedule

Lightning was distilled on a fixed shift of 3: its card sets diffusers'
`base_shift` and `max_shift` both to ln 3 and drops the terminal stretch.
Kvad's Qwen-Image schedule shifts by the image's size, about 0.69 at 1024²,
and stretches to 0.02. Over 8 steps the two are far apart: Lightning's
card's schedule ends at σ 0.30 and jumps to 0 from there, Qwen-Image's
steps down to 0.02. Step 1 drew with both (below): the same picture in
composition and quality, 21.3 dB apart in PSNR. Neither draws worse, so a
request needs no way to ask for Lightning's to draw well; whether to offer
it anyway, to match the recipe on Lightning's card, is step 2's question.

## How it is checked

Against diffusers with PEFT, as each model was checked. The whole
Qwen-Image transformer is 82 GB in f32, so the reference runs its first
blocks only, from the cached bf16 shards, with and without the LoRA:
`scripts/lora-fixtures.py`, then `qwen::tests`.

## The order it is built in

1. **The side path and the registry,** Lightning on Qwen-Image through
   `examples/sdxl.rs --lora`: checked against diffusers with PEFT, its cost
   a step measured, and pictures at 8 steps with both schedules.
2. **The request and the service:** `loras` on `/v1/images/generations`,
   the CLI, pulls of a LoRA by name, admission, the Images page.
3. **SDXL and SD 1.5:** kohya's names in both layouts, the text encoders'
   LoRAs, and the convolutions'.
4. **FLUX.1-schnell:** PEFT, and Black Forest Labs' layout.
5. **LTX-2.5:** its DiT's own linear layers, and `/v1/videos`. The last of
   #51's list.

## What was built, and what it measured

Added 2026-09-28, on an M5 Pro (48 GB).

**Step 1, the side path and the registry** (`image::lora`):
- **`lora::File`** reads a LoRA's pairs from its header in any of the three
  spellings, with `alpha` where given, and refuses LyCORIS' kinds, DoRA,
  whole-weight differences, a pair missing a half, and any other tensor, by
  name.
- **`lora::Adapters`**, carried by a `Reader`: each `nn::Linear` read
  through it registers and keeps a `Slot`. Setting a LoRA matches its names
  after a prefix, dotted or with kohya's underscores. A pair that names no
  layer, or does not fit one, refuses the whole LoRA and leaves nothing
  set.
- **Qwen-Image** registers its transformer's layers under `transformer.`,
  `diffusion_model.` and `lora_unet_`, and gains `set_loras` and
  `set_shift`. `examples/sdxl.rs --lora FILE[:STRENGTH] --shift F`.
- **`mpp::dense_acc`**, `c += x·w` in place, into an f32 `c` from halves:
  the dense kernel's accumulating store, templated on `C`'s type.

Checked:
- **Against diffusers with PEFT** (`scripts/lora-fixtures.py`, then
  `qwen::tests::a_lora_agrees_with_peft`), Qwen-Image's first two blocks
  with Lightning's 24 pairs for them, from the cached bf16 shards:

  | | Plain | With Lightning | At half strength | Lightning's own part |
  |---|---|---|---|---|
  | CPU, f32 | 95.1 dB | 95.8 dB | 95.7 dB | 42.7 dB |
  | Metal, q8, bf16 factors | 50.3 dB | 50.3 dB | 50.3 dB | 21.5 dB |

  The part is the adapted output less the plain one, 47.4 dB below the
  output, so each output's own rounding bounds it: about 45 dB in f32,
  where it reads 42.7. Lightning's `alpha` is 8 at rank 64, a scale of
  1/8, which diffusers folds into the factors and Kvad into `B`.
- **Unit tests:** `W·x + (alpha/r)·strength·B·A·x` for known numbers, in each
  spelling, through each prefix and kohya's underscores; each refusal;
  `dense_acc` into f32 and into its own dtype, within a rounding.

Measured, Qwen-Image at q8, 1024², Lightning 8-step V2.0 (bf16, 0.85 GB):
- **Pictures:** the base at 8 steps is a blur. With Lightning at 8 steps it
  is sharp, and finer than the base's own 50, on either schedule.
- **Setting it:** 720 layers in 1.2 s. The peak grows by the factors'
  0.85 GB, 38.8 → 39.7 GB. Kept in f32 they were 1.7 GB, and a run then
  went to swap on this machine: a 212 s decode.
- **A step:** +0.37 s, +4.5%, the median of eight back-to-back pairs
  (`examples/lora_cost.rs`), where the side path as four separate passes
  cost +14.9%. At Qwen-Image's shapes, the passes the in-place sum
  removed were two to three times the products' own time
  (`lora::cost`): 2.1 → 1.0 ms a 3072-wide layer, 6.4 → 2.0 ms a
  12288-wide one.
- **So Lightning's 8 steps took 68 s** to denoise, where the base's 50
  took 448 s in the same session: 6.6× less, for a sharper picture.

**Step 2, the request and the service.**
- **The request:** `loras: [{name, scale}]` on `ImageRequest`, at most four,
  each once, with a scale between −10 and 10, and refused by a model whose
  `Defaults` say it `takes_loras` not (every one but Qwen-Image, so far).
  `/v1/images/generations` takes it beside OpenAI's fields; `kvad images
  make --lora NAME[:SCALE]`, as often as wanted.
- **Names** (`kvad::lora`), as a checkpoint's: `repo`, `repo:file.safetensors`
  or a path. A file is a LoRA if every tensor is one half of a pair or an
  `alpha`. A pull reads the header on the Hub first, and by the repo's name
  finds the one LoRA among its files and records it (`kvad-only-lora`), as
  a checkpoint's pull does. `kvad pull` tries a checkpoint first and a LoRA
  when the header says it is none; a LoRA needs nothing beside it.
- **What a LoRA is for**, guessed from its layers' names (`lora::adapts`):
  a UNet's blocks are SD 1.5's or SDXL's, Qwen-Image's name each stream's
  MLP, FLUX's have single-stream blocks, LTX-2.5's attend in
  `transformer_blocks`. The file does not say, and a model it does not fit
  refuses it whatever this answers; the Images page offers a model the
  LoRAs for its pipeline and those it cannot place.
- **Set for the request and taken off after**, error or not. So nothing
  stays on the device uncharged: a request's LoRAs must fit in what no
  resident has been charged, and pay Lightning's 1.2 s of setting each time,
  2% of an 8-step picture at 1024².
- **Loads are cache-first:** a LoRA not on this machine is a 400 that names
  the pull.
- **A LoRA that does not fit** is refused as the asker's to change: a
  painter's `image::Refused` travels from the engine as its own event, and
  is a 400 (an error event when streaming), naming the LoRA as asked, how
  many of its layers the model has not, and the first few.
- **Kept with the image:** migration 016 adds `images.loras`, the gallery
  and each image's `kvad` object carry them, and "Reuse" puts them back.
- **The listing:** a LoRA is a row of its own, `lora: true` and what it
  `adapts`, never runnable; `kvad ls` calls it `lora`, the Models page
  badges it ("LoRA for Qwen-Image") without Load or Default, and `kvad rm`
  deletes its one file.
- **The Images page** has a LoRA section: add, choose, strength, remove.

Checked through a test server:
- Lightning by `repo:file`, here already, fetched nothing; `nerijs/pixel-art-xl`
  by its repo's name was found among its files by header, fetched and
  recorded, in 13.7 s with its 170 MB.
- Through the service, with Lightning, Qwen-Image drew the example's picture
  byte for byte; admission charged the model its 27.3 GB and the LoRA
  nothing standing. The next request, without it, drew the plain picture of
  step 1 byte for byte, so taking it off leaves nothing behind.
- From the Images page, Lightning at 8 steps and 768², 30.7 s; the gallery
  lists the LoRA with the image.
- A LoRA not here, a scale of 99, and pixel-art-xl on Qwen-Image (722 of its
  722 layers none of the model's) are each a 400 that says why.

**Step 3, SDXL and SD 1.5.** What their LoRAs are, read off the headers of
the 24 most downloaded SDXL LoRAs and SD 1.5's: 16 of SDXL's are kohya's
under `ldm`'s names, 5 under diffusers', 7 adapt both text encoders too, 4
carry 3×3 convolutions (LCM, TCD, a slider, ikea-instructions), 3 use
diffusers' older `unet.…lora.down` and one its `LoRAAttnProcessor`'s names.
SD 1.5's are kohya's under diffusers' names, some with the text encoder, and
the most downloaded of all, LCM-LoRA (95k), has convolutions.
- **Parts:** a registry holds each part of a model apart (`unet`, `te1`,
  `te2`, Qwen-Image's `transformer`), and a LoRA's prefix says which it
  adapts: `lora_unet_` or `unet.`, `lora_te_`/`lora_te1_` or `text_encoder.`,
  `lora_te2_` or `text_encoder_2.`.
- **`ldm`'s names:** each UNet layer is also registered by its name in
  Stability's layout (`single::ldm_of`, `single::unet` backwards), which
  kohya's SDXL LoRAs use.
- **Convolutions:** `nn::Conv2d` registers too. LoCon's `A` is a convolution
  of the layer's own kernel, stride and padding into `r` channels, and `B`
  a product across them at each pixel, summed in place as a linear layer's
  is. A 1×1 kernel's factors on a layer Kvad runs as linear (SD 1.5's
  projections) are read as the matrices they are.
- **diffusers' `processor` names** (`attn1.processor.to_out_lora`) are read
  as the layers' own (`attn1.to_out.0`).
- **What the model skips:** SDXL reads CLIP-L's second-to-last layer and
  never computes its last. A LoRA's pairs there are the model's, known,
  and change nothing, as in the reference; `Reader::skip_under` says so.
- **One wrapper** sets a request's LoRAs, draws and takes them off, for all
  three pipelines (`lora::painting`), and SDXL and SD 1.5 take LoRAs.

Checked against diffusers with PEFT, in f32 on the CPU, the base's text
encoders whole and one UNet call on the reference's own hidden states
(`scripts/lora-fixtures.py --pipeline sdxl|sd15`, then
`lora::sd_tests::sd_loras_agree_with_peft`):

| LoRA | Kinds | Text encoders | UNet | Its own part |
|---|---|---|---|---|
| `nerijs/pixel-art-xl` | kohya, `ldm`'s names | unchanged | 112.7 dB | 98.0 dB |
| `thejagstudio/3d-animation-style-sdxl` | kohya, `ldm`'s, both text encoders | 116.8 dB | 117.0 dB | 93.1 / 98.9 dB |
| `latent-consistency/lcm-lora-sdxl` | kohya, diffusers', 3×3 convolutions | unchanged | 108.2 dB | 94.0 dB |
| `DiffusionLight/TurboLoRA` | diffusers' `unet.…lora.down` | unchanged | 112.2 dB | 95.6 dB |
| `jbilcke-hf/sdxl-cinematic-1` | diffusers' `processor` | unchanged | 111.1 dB | 90.4 dB |
| `latent-consistency/lcm-lora-sdv1-5` | kohya, diffusers', convolutions | unchanged | 117.0 dB | 104.9 dB |
| ColoringBook Redmond 1.5 | kohya, diffusers', the text encoder | 113.7 dB | 120.6 dB | 94.0 / 89.5 dB |

At half strength, 108.8–120.7 dB. In f16 on Metal, as the pipelines run,
the LoRAs on attention and text encoders sit where the plain f16 run does
(57–67 dB, as plain), and the two with convolutions a little under it:
LCM-LoRA at 54.5 dB on SDXL against 57.2 plain, and 62.3 on SD 1.5 against
67.5, its pairs in f16 across a full-resolution grid.

Two faults in the reference, found on the way:
- **diffusers 0.40 cannot load kohya's text-encoder pairs:** its
  `get_peft_kwargs` fails on an empty rank list. The script loads the UNet's
  through diffusers and merges the text encoders' into their weights in
  f32, which is the same arithmetic.
- **SDXL's second tokenizer was half in the cache,** its config without its
  vocabulary, and transformers 5.14 built an empty tokenizer from it
  without a word: the prompt became `[0, 1, 1, …]` for bigG, and the first
  SDXL references were −1 dB from Kvad's. Kvad reads CLIP-L's tokenizer for
  both encoders and never needed those files; with them fetched, the
  references agree.

Pictures, each against the same seed without its LoRA: pixel-art-xl draws
a cleaner sprite on a finer grid, 3d-animation-style a Pixar-like fox
where the base drew a photograph, ColoringBook on DreamShaper 8 an inked
drawing where it painted. SDXL's step with pixel-art-xl took 2.97 s, the
base's 3.02 s, single runs.

**Step 4, FLUX.1-schnell.** Its LoRAs, read off the headers of the 84 most
downloaded for FLUX.1-dev and schnell (most are trained on dev, and applied
to schnell as they are): 46 are PEFT's under `transformer.`, diffusers'
names; 28 kohya's in Black Forest Labs' layout, 10 of those with CLIP-L's
pairs too; one ComfyUI's, one XLabs'; 5 LyCORIS', refused by name.
- **Black Forest Labs' layout** fuses a double block's q, k and v into one
  `qkv`, and a single block's q, k, v and the MLP's input into `linear1`.
  The rows of each that each of diffusers' layers is are already written
  down, for city96's GGUFs (`flux::gguf_map`), and a LoRA's pair for a fused
  layer is split the same way (`bfl_loras`, `Adapters::fused`): `A` whole,
  since every row of the fused answer reads the same input, and `B` cut to
  the layer's rows, in the layer's order, which also turns the last
  modulation's halves the other way round, as the weight's are.
- **CLIP-L** registers as `te1`, and the transformer, dense or GGUF, as
  `transformer`; prefixes `transformer.`, `diffusion_model.`, `lora_unet_`,
  `lora_transformer_`, `lora_te1_` and `text_encoder.`.

Checked against diffusers with PEFT (`scripts/lora-fixtures.py --pipeline
flux --blocks 1`, then `flux::tests::a_lora_agrees_with_peft`), the first
double and single blocks from the cached bf16 shards, and each LoRA's pairs
for them, which diffusers' own FLUX loader converts from kohya's names:

| LoRA | Kind | CPU, f32 | Its own part | Metal, q8 |
|---|---|---|---|---|
| `furaidosu/flux-lora-gliff-tosti-vector-1` | PEFT | 111.2 dB | 81.3 dB | 35.3 dB |
| `prithivMLmods/Canopus-LoRA-Flux-FaceRealism` | kohya, fused layout | 110.9 dB | 74.1 dB | 35.4 dB |
| `hugovntr/flux-schnell-realism` | kohya, fused layout, CLIP-L | 111.0 dB | 68.0 dB | 35.4 dB |

Without a LoRA, 111.0 dB and 35.4 dB; each part as near as the outputs'
own rounding lets it be (29–42 dB below them). The q8 figures are FLUX's
own at q8, a LoRA or none.

Pictures at 1024², 4 steps, each against the same seed without its LoRA:
the vector LoRA draws flat, cut-out shapes where the base draws a soft
illustration (and letters the trigger word into a corner); Canopus a more
photographic face; the realism LoRA, CLIP-L's pairs and all, a close
variant of the base's. Steps took 7.8–8.6 s with a LoRA, 7.0–8.1 without,
single runs on a busy machine.

**Found on the way:** the first Canopus picture was black. The installed
service had its own FLUX at q8 loaded, 17 GB, beside a virtual machine's
8 GB and QEMU's 7 GB, and the second FLUX went 41 GB into swap: its decode
took 193 s, and a command buffer that fails under that pressure is served
as zeros, with no error. Unloaded, the same run drew its portrait and
decoded in 9.2 s. The pipelines check the latent before the decode, and
not the pixels after it.

**Step 5, LTX-2.5.** Its LoRAs on the Hub are all PEFT's under
`diffusion_model.`, the reference's own names, some with an `alpha`: a few
video only, most with the audio stream and its cross-attention too. Most of
Lightricks' own, and several of the community's (Ripple, AKUSPACE, the
cel-character one), are IC-LoRAs, trained to be conditioned on a video or a
sound beside the prompt; they load and apply here, but what they are for
needs that conditioning, which a request cannot give yet.
- **The DiT is built of `nn::Linear`**, as the image models are, so its
  reader carries the registry and every projection takes a LoRA with
  nothing new in the layers. The checkpoint names them under
  `model.diffusion_model.` and LoRAs under `diffusion_model.`; a part can
  now leave a prefix off what it registers (`Adapters::part_under`).
- **The connectors are the text path's,** skipped by the DiT's loader
  before the registry is attached, so a LoRA's pairs for them are refused
  rather than taken as known and ignored.
- **Every DiT a video loads takes the request's LoRAs**: both stages', the
  dev model's and its distilled stage 2's, DFR's detailing DiT and its
  temporal rounds', and a GGUF's, which a fused LoRA cannot be (not run
  here: no GGUF of LTX-2.5 is on this machine). They are
  opened once, before anything loads (`Ltx::adapted`); a DiT goes with its
  stage, and them with it.
- **The request:** `loras` on `VideoRequest`, checked as a picture's are
  (`image::check_loras`); `/v1/videos` takes it as JSON's list or a form's
  field of JSON text; `kvad videos make --lora`; the Videos page's picker,
  now one component with the Images page's (`Loras.svelte`), offering the
  LoRAs whose names say LTX-2.5 and those they place nowhere. Migration 017
  keeps them with the video, and `kvad.loras` says them.

Checked against the reference, with its own LoRA, the distilled LoRA
(rank 450) on the distilled DiT's first two blocks, the fixture #97 checked
fusing against (`examples/ltx_dit --lora … --at-run-time`):

| | Fused | At run time |
|---|---|---|
| f32, CPU: blocks / velocity | 117.7–119.5 / 114.8–120.7 dB | 58.1–61.6 / 55.4–56.1 dB |
| bf16, Metal | 43.5–47.6 dB | 43.4–47.0 dB |
| q8, Metal | 43.5–47.6 dB | 43.0–47.2 dB |

The reference's own bf16 is 44.6–47.7 dB. The f32 gap is the reference's
fusing, not the side path: it rounds `W + B·A` to bf16, the checkpoint's
dtype, and Kvad's fused path does the same. Fused without that rounding,
the six figures are the run-time ones to a tenth of a dB, so the side path
is the exact product; in bf16 and q8, as videos run, the two are level
with each other and with the reference's own bf16.

Through the service, `SOLRICKS/LTX-2.5-BTS-Movie-Set` (both streams, pulled
by its repo's name) at 0.8, 768×512 and 49 frames, against the same seed
without it: a studio soundstage, lighting truss, a camera on a dolly with
its monitor and a crew in matching jackets, where the base drew a crew in
a snowy forest; denoising took 36.5 s against 35.9, and the video keeps its
LoRA. A LoRA that does not fit the DiT fails the video, since a video is a
job already accepted, with the same message a picture's refusal gives.

**Found on the way:** `lora::adapts` called two of kohya's FLUX LoRAs
Qwen-Image's. Black Forest Labs' double blocks name `img_mlp` and
`img_mod`, as Qwen-Image's do, and FLUX's block names are now asked about
first. And `kvad pull`'s hint for a video LoRA said `kvad images make`.
