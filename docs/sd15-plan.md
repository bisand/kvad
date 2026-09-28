# Plan: Stable Diffusion 1.5

Written 2026-09-27 for [#40](https://github.com/bisand/kvad/issues/40). SD 1.5
is still among the most downloaded text-to-image repos, mostly for its
fine-tunes (DreamShaper, Realistic Vision and hundreds more). Everything in
it is a smaller configuration of what `sdxl.rs` runs already.

## What differs from SDXL

Read off `stable-diffusion-v1-5/stable-diffusion-v1-5` and
`Lykon/dreamshaper-8`, whose files are the same in kind: `.fp16` safetensors
for the text encoder, UNet and VAE, 2.1 GB together.

- **One text encoder, CLIP-L**, read to its last layer and through its final
  norm, where SDXL reads two encoders' second-to-last layers. No pooled
  vector, so no size conditioning either: the UNet's config has no
  `addition_embed_type`.
- **The UNet's config** differs in form more than in kind:
  - four levels, the last without attention (`DownBlock2D`); `unet.rs`
    counts levels from the config already;
  - `attention_head_dim` is one number, 8, for every level, where SDXL's is
    a list;
  - no `transformer_layers_per_block`: one transformer a level;
  - `use_linear_projection` is false: the transformers' `proj_in` and
    `proj_out` are 1×1 convolutions, `[c, c, 1, 1]`, which are the same
    arithmetic as SDXL's linear ones on the same weights reshaped.
- **The VAE is the repo's own**, in f16. SDXL's own VAE overflows in f16
  and Kvad decodes SDXL with the fp16-fix; SD 1.5's does not, and its latent
  space is not SDXL's. Its scaling factor is 0.18215, read from its config.
- **The scheduler is PNDM** in the base repo and DEIS in DreamShaper's. Kvad
  runs Euler on their noise levels, as #40 says: `schedule::euler` reads no
  class, only the settings. The base's config leaves out `prediction_type`
  and `timestep_spacing`, which then mean diffusers' defaults for its class:
  `epsilon`, and `leading` for PNDM. The Euler schedule now reads a missing
  setting as its class's default, and still refuses a setting it does not
  implement.
- **Defaults:** 512², 25 steps, guidance 7.5.
- **The safety checker and feature extractor** are components of the
  pipeline. Neither is loaded or fetched.

## How it is checked

Against diffusers itself, on the same weights and inputs, each part in dB as
LTX-2.5's were: the text encoder's hidden states, one UNet call, and the
VAE's decode. `scripts/sd15-fixtures.py` writes the reference's, and
`examples/sd15.rs` compares.

## The order it is built in

1. **The pipeline** in diffusers' layout: `StableDiffusionPipeline`, its
   parts checked against the reference, a picture from the base and one
   from DreamShaper.
2. **Single files** in the original layout, which most SD 1.5 fine-tunes
   are: `cond_stage_model.transformer.` for CLIP-L, the UNet under
   `model.diffusion_model.` with four levels, and the VAE under
   `first_stage_model.`, which here is read, since a fine-tune's VAE is
   often its own.
3. **Names and the service:** the pipeline in the listing, admission and
   the Images page, and SD 1.5 files through the names `kvad::checkpoint`
   gives SDXL's.

## What was built, and what it measured

Added 2026-09-27, on an M5 Pro (48 GB).

**Step 1, the pipeline** (`image::sd15`), from the parts SDXL has, each made
general where SDXL had fixed it:
- **The UNet** takes its size conditioning as optional, 1×1 projections in
  its transformers (`Linear::load_1x1`), a head count and a depth that are
  one number or a list, and its middle block's depth from the config
  rather than the last level's, which SD 1.5's does not attend at.
- **CLIP** gains `Pooled::Last`: every layer, and the whole sequence through
  the final norm.
- **The Euler schedule** reads a setting the config leaves out as its
  class's diffusers default, and still refuses what it does not implement.
- **The VAE's** `scaling_factor`, left out of SD 1.5's config, is diffusers'
  default for `AutoencoderKL`, 0.18215, SD 1.5's own.
- **CLIP's `embeddings.position_ids`**, a buffer SD 1.5's text encoder file
  keeps, joins the unread-weights guard's list of constants a loader
  builds itself.
- **Previews** have SD 1.5's own colours, fitted as SDXL's were
  (`scripts/sd15-fixtures.py --preview`): 74%, 72% and 62% of R, G and B.

Against diffusers, on the same weights and inputs
(`scripts/sd15-fixtures.py`, then `sd15::tests::agrees_with_diffusers`):

| Part | f32 on the CPU | f16 on Metal |
|---|---|---|
| Text encoder (the tokens identical) | 113.4 dB | 58.2 dB |
| UNet, one call | 120.5 dB | 67.5 dB |
| VAE, a decode | 112.4 dB | 60.4 dB |

- **The base and DreamShaper 8** draw what their prompts ask, each in its
  own style: 512², 25 steps, 0.74–0.75 s a step, a 1.1 s load, a 7.6 GB
  peak.
- **Through the service** both are listed as image models, and DreamShaper
  drew the example's picture pixel for pixel, at `metal f16`, charged
  2.1 GB. Nothing in the service changed: it asks the pipeline.
- **SDXL is unchanged:** the same picture as before the change, pixel for
  pixel.

**Step 2, single files** (`single::sd15`, `Sd15::load_with`). The base's
configs and scheduler, and the file's three models, each through a map:
- **The UNet** is SDXL's map. Its rules count levels from the names, so the
  fourth level, and the first level's upsampler after a resnet rather than
  an attention, come out as diffusers names them.
- **CLIP-L** is Hugging Face's names under `cond_stage_model.transformer.`.
- **The VAE** has a map of its own. `ldm` numbers the decoder's levels from
  the bottom (`up.3` is diffusers' `up_blocks.0`), calls the shortcut
  `nin_shortcut`, and makes the attention's projections 1×1 convolutions,
  which a map's source now reads as the matrix they are (`Src::matrix`). The
  encoder is left unread, as the diffusers VAE's is.
- **Left unread, and said so:** `ldm`'s twelve tables of the noise schedule,
  the EMA's bookkeeping, and in a file that kept it the EMA's copy of the
  UNet under `model_ema.`, which diffusers' conversion leaves unread too.
  Anything else is refused by name.
- **An inpainting UNet**, nine channels in, is refused as one, not as a shape
  the loader did not expect.
- **The EMA's step counter is an `I32`**, which the safetensors reader now
  accepts.

Checked:
- **The base's own `v1-5-pruned-emaonly.safetensors`** (f32) against its
  diffusers folders (f16), through the maps
  (`single::tests::sd15_s_base_file_is_its_own_folders_tensor_for_tensor`):
  all 1022 tensors the loaders read are the same bit for bit, and every name
  in the folders is found. The 123 left unread are the VAE's encoder (108),
  CLIP's buffer of positions, the schedule's 12 tables and the EMA's 2.
- **Pictures:** the base from its single file draws the same PNG as from its
  folders, byte for byte; so does DreamShaper 8 from
  `Lykon/DreamShaper`'s `DreamShaper_8_pruned.safetensors` against
  `Lykon/dreamshaper-8`. DreamShaper's VAE differs from the base's in all
  248 of its tensors, so reading the file's own is what makes its picture
  its own.
- **Load and memory:** 1.5 s from the base's 4.3 GB of f32 and 1.1 s from
  DreamShaper's 2.1 GB of f16, a 7.7 GB peak, where the folders' is 7.6 GB.
  0.75 s a step, as from the folders.

**Step 3, names and the service.** `kvad::checkpoint` tells the two kinds
apart by the header: the UNet beside SDXL's bigG, or beside SD 1.5's CLIP-L
(`checkpoint::Kind`). SD 2's files, whose text encoder is OpenCLIP's under
`cond_stage_model.model.`, are neither and are refused, as a LoRA or a UNet
alone is. Everything SDXL's single files had follows from the kind:
- **Names:** `repo`, `repo:file.safetensors` or a path, found and checked by
  a range request on the Hub before a byte of weights is fetched.
- **Listing:** each file is an image model of its kind's pipeline
  (`LocalModel.single` now carries the kind).
- **Loading and pulls:** its kind's base gives its configs; a pull fetches
  them, and for SD 1.5 no VAE besides.
- **Admission:** charged what the loaders read, in f16: not an f32 file's
  size, the VAE's encoder, `ldm`'s training state or an EMA's copy of the
  UNet.

Measured through the service, M5 Pro:
- The base's file by `repo:file`, a copy of it by its path, and DreamShaper's
  by its repo alone each drew the example's picture byte for byte, and were
  charged 1.9 GB, the pipeline's 1.03 B parameters in f16.
- `ckpt/anything-v5.0`, one file and nothing else, was pulled by its repo's
  name and drew in its own style.

**Found on the way:** a repo named alone with no `config.json` was fetched
in full before the pull found out it was no language model and asked
whether it was a checkpoint: a UNet alone, 1.7 GB, was downloaded and then
refused, in 1 min 50 s. The language model's pull now fetches the config
before the weights, so the same refusal comes from the header, in 2.3 s,
with nothing fetched; a language model's pull is otherwise as it was.

**Names, corrected.** A checkpoint was listed by its repo's name alone
whenever it was the only one in the cache entry, beside no model index or
config. The cache holds only what was fetched, so one of
`Lykon/DreamShaper`'s 37 single files, fetched by `repo:file`, was listed as
`Lykon/DreamShaper`, a name that means the repo's diffusers pipeline on the
Hub, and fetching a second file of the repo would have renamed it. Now:
- **A pull by the repo's name**, which asks the Hub and finds one checkpoint,
  records it in the cache entry (`checkpoint::ONLY`), and only a recorded
  file is listed by the repo's name. Anything else is `repo:file`.
- **Only a checkpoint counts** as a repo's model: the headers of the files at
  its top say which are, up to eight of them. Pony's repo keeps SDXL's VAE,
  `sdxl_vae.safetensors`, beside its checkpoint, so a pull by its name was
  refused as ambiguous; it is now found, and recorded, in 3.8 s from the
  cache, and draws by that name.
