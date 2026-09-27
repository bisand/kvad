# Plan: SDXL fine-tunes, in diffusers' layout and in one file

Written 2026-09-27 for the second half of
[#42](https://github.com/bisand/kvad/issues/42), after its GGUF half
(`docs/gguf-plan.md`). The headers below were read from the Hub with range
requests, and the repos' file lists through its API.

## What is out there

#42 said the fine-tunes are single files. Most of the popular ones on the Hub
are not only that: they ship diffusers folders beside the file.

| Repo | Downloads | Diffusers folders | Single file |
|---|---|---|---|
| `RunDiffusion/Juggernaut-XL-v9` | 784k | yes, `.fp16` files | 7.1 GB |
| `cagliostrolab/animagine-xl-4.0` | 360k | yes, plain files | 6.9 GB, and an "opt" one |
| `LyliaEngine/Pony_Diffusion_V6_XL` | 264k | no | 6.9 GB |
| `Laxhar/noobai-XL-1.1` | 125k | yes, plain files | 7.1 GB |
| `John6666/prefect-illustrious-xl-v3-sdxl` | 47k | yes, plain files | none |
| `OnomaAIResearch/Illustrious-xl-early-release-v0` | 80k | yes, plain files | two |

- **Kvad cannot load most of those folders either.** Its SDXL loader, and the
  admission that sizes it, name `model.fp16.safetensors` and
  `diffusion_pytorch_model.fp16.safetensors`, which is how
  `stabilityai/stable-diffusion-xl-base-1.0` ships them. The fine-tunes ship
  `model.safetensors` and `diffusion_pytorch_model.safetensors`, in f16
  already (the UNet is 5.1 GB), and John6666's merges, thousands of them, the
  same. Juggernaut's are `.fp16` and would load.
- **The single file is the original layout**, Stability's own, in four
  parts. `stabilityai/stable-diffusion-xl-base-1.0` ships one,
  `sd_xl_base_1.0.safetensors`, 2515 tensors in f16:
  - `model.diffusion_model.*`, the UNet in `ldm`'s names (1680):
    `input_blocks.N.M`, `middle_block`, `output_blocks`, `time_embed`,
    `label_emb`;
  - `conditioner.embedders.0.transformer.text_model.*`, CLIP-L, already in
    Hugging Face's names under a prefix;
  - `conditioner.embedders.1.model.*`, OpenCLIP's bigG: `resblocks.N` with
    one fused `attn.in_proj_weight` for q, k and v (`[3840, 1280]`),
    `ln_1`, `mlp.c_fc`, `mlp.c_proj`, and a `text_projection` that
    diffusers stores transposed;
  - `first_stage_model.*`, the VAE, its attention as 1×1 convolutions.
- **NoobAI's "Vpred" models predict velocity**, where SDXL predicts noise:
  a scheduler Kvad does not have yet.

## Decisions

- **Names.** A Hub repo whose only model is one checkpoint (Pony's) is named
  by the repo. A repo with several is `repo:file.safetensors`. A path to a
  `.safetensors` file on this machine works too, for Civitai's, which Kvad
  cannot fetch itself.
- **The VAE is madebyollin's fp16-fix**, as for every SDXL here: the same
  latent space, and safe in f16, where SDXL's own VAE, which most files
  carry, overflows. A file's own VAE is left unread, and says so; a
  fine-tune with a VAE trained apart decodes with the standard one.

## The order it is built in

1. **The diffusers-layout fine-tunes:** each file as `.fp16` where the repo
   has it and plain where it does not, read in f16 whatever it is stored
   in, and admission sized from the files there.
2. **The single file:** a map from the original layout to diffusers' names
   (the UNet, CLIP-L, bigG with its `in_proj` split by rows), checked
   against `stabilityai/stable-diffusion-xl-base-1.0`'s own diffusers
   folders, tensor for tensor.
3. **Names:** `repo`, `repo:file`, and local paths, through pull, the
   listing, admission, load, delete, the CLI and the web page.
4. **Later, each on its own:** v-prediction (NoobAI's Vpred), and SD 1.5
   ([#40](https://github.com/bisand/kvad/issues/40)), whose fine-tunes are
   single files too.

## What was built, and what it measured

Added 2026-09-27, on an M5 Pro (48 GB).

**Step 1, the diffusers-layout fine-tunes.** Each component's weights are
the `.fp16` file where the repo has one and the plain file where it does
not, asked of the cache first and then of the Hub, `.fp16` first, so a repo
with both never downloads its f32 file. Either is read in f16. Admission
sizes the pipeline from the headers of whichever files are here, at two
bytes a weight.
- **`cagliostrolab/animagine-xl-4.0`**, plain files only, loads and draws
  what its prompt asks, in its own style: 1024², 30 steps, 2.83 s a step,
  a 29.3 GB peak. The first load took 430 s, most of it the 6.7 GB download.
  It is listed as runnable, which is admission sizing it.
- **`stabilityai/stable-diffusion-xl-base-1.0`** still loads its `.fp16`
  files, offline, in 5.4 s, at 2.79 s a step.
- **A velocity-predicting model is refused by name**, as it was: the Euler
  schedule reads `prediction_type` and takes only `epsilon`.

**Step 2, the single file.** `image::single` maps each name the three
loaders ask for to where it is in the file, built from the file's own
names, and a reader serves it from there: a tensor, rows of one, or a
transpose. The UNet's `ldm` names become diffusers' by level; CLIP-L's lose
a prefix; bigG's `in_proj` is cut into q, k and v by rows, and its
`text_projection` is transposed. The configs, the tokenizer and the
scheduler come from `stabilityai/stable-diffusion-xl-base-1.0`, since a
single file carries none, and the VAE is madebyollin's fp16-fix.
- **The unread-weights guard covers the file whole:** everything in it is
  read through a map, skipped under a prefix a loader skips (CLIP-L's last
  layer and final norm, which SDXL does not read), or on the file's own
  list of what nothing reads: the VAE, CLIP-L's `position_ids` buffer and
  bigG's `logit_scale`. It caught the skipped layer the first time: the
  folders' guard knew of it, and the file's did not yet.
- **Against the base's own folders**, from
  `stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors`
  (`single::tests::the_base_file…`, ignored, since it needs both copies):
  all 2393 tensors the loaders read are the folders' bit for bit, and 250
  are left unread (the VAE's 248, and the two above). The same prompt and
  seed at 1024² in 20 steps draw the same picture pixel for pixel, at the
  same 3.43 B parameters, a 5.8 s load and 2.98 s a step.
- **Pony Diffusion V6 XL** (`LyliaEngine/Pony_Diffusion_V6_XL`, a single
  file only) loads and draws what its prompt asks, in its own style:
  1024², 30 steps, 3.09 s a step, a 28.1 GB peak.
- **Other files' headers**, read from the Hub: NoobAI-XL 1.1 (bf16, without
  CLIP-L's `position_ids`) and Juggernaut XL v9 (without `logit_scale`) are
  the same four parts and nothing else, which the maps take as they come.

**Step 3, names.** `kvad::checkpoint` finds the file a name means: `repo`
when the repo's only model is one checkpoint, `repo:file.safetensors`, or a
path on this machine. A file is one only if its header is SDXL's, the UNet
and bigG under their prefixes, which on the Hub is read with two range
requests, so a file that is not one costs kilobytes to refuse.
- **The listing** makes each checkpoint at the top of a cache entry a model
  of its own, and a repo whose only model is one checkpoint that model by
  its own name. A language model's entry is never read for them: its files
  are its shards. Pony is listed as `LyliaEngine/Pony_Diffusion_V6_XL`, and
  the base's file as `stabilityai/stable-diffusion-xl-base-1.0:sd_xl_base_1.0.safetensors`
  beside the base itself, each an image model that admission sizes.
- **A pull** of `repo:file` or a path is a checkpoint's. One of `repo`
  alone is a language model's first, and a checkpoint's only when that
  fails and the repo holds `.safetensors` files and nothing that makes it
  a model of its own, so a language model's pull asks the Hub nothing
  more. It fetches the file and what goes beside it: SDXL base's configs,
  its tokenizer and scheduler, and the fp16-fix VAE.
- **A path** is loaded by name, the request's autoload and its default
  backend taking it as the listing takes a cached file. It is never in the
  listing, so `kvad rm` cannot delete a file of the user's by its path.
- **Delete** takes the one file, and the cache entry once no model is left
  in it (`hub::remove_cached`, which the GGUFs' delete now shares).
- **The Models page** offers a pull of any name typed whole: a GGUF, a
  checkpoint file, or a repo, since its search lists text models.

Checked through a test server on this branch:
- Pony by its repo name, and by its path, drew the example's picture pixel
  for pixel;
- the base's `sd_xl_offset_example-lora_1.0.safetensors`, a LoRA, is refused
  as no SDXL checkpoint, from its header;
- `morikomorizz/Pony-Diffusion-V6-XL-GGUF`, five `.safetensors` files at its
  top, is refused with their names to choose from;
- a pull of Pony, here already, and of a path fetched nothing but what goes
  beside them, and `kvad pull` names `kvad images make` after both;
- a language model's pull is as it was;
- `kvad rm` of the base's single file took that 6.9 GB file and left the
  base's folders and its listing;
- the Models page offers the pull for a checkpoint's name and for a repo's.
