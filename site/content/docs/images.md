---
title: Images
description: Text to image with Stable Diffusion 1.5, SDXL, FLUX.1-schnell and Qwen-Image, on the Mac's GPU.
---

Image generation runs on the GPU engine, so it needs a Mac. The models are
written out in this repository: CLIP, T5, the UNet, the MMDiT, the VAEs and
both schedulers. Only convolution and matrix multiplication come from a
framework.

```bash
kvad pull stabilityai/stable-diffusion-xl-base-1.0
kvad images make "a red fox in fresh snow" \
    --model stabilityai/stable-diffusion-xl-base-1.0
```

The picture is written to `image-ID.png`, or to `--out FILE`. It is also kept
on the server with the settings that made it, where the web UI's Images page
and `kvad images` list it.

## Models

| Model | In memory | A step, on an M5 Pro | Steps |
|---|---|---|---|
| Stable Diffusion 1.5 | 1.9 GB | 0.75 s at 512² | 25 |
| SDXL | 6.4 GB | 3.1 s at 1024² | 30 |
| FLUX.1-schnell, q8 | 18.3 GB | about 9 s at 1024² | 4 |
| Qwen-Image, q8 | about 29 GB | 28 s at 1024² | 20 |

SDXL and Qwen-Image run the denoiser twice per step while guidance is on, and
those figures include both passes. FLUX.1-schnell needs no guidance. FLUX.1-dev
is not supported.

### Fine-tunes

SDXL and Stable Diffusion 1.5 fine-tunes load in any of the three layouts
people publish: a diffusers folder, a single checkpoint file in Stability's
own layout, or a file on your disk.

```bash
kvad pull Lykon/dreamshaper-8
kvad images make "a lighthouse at dusk" --model Lykon/dreamshaper-8
kvad images make "a lighthouse at dusk" --model ~/Downloads/some-sdxl-finetune.safetensors
```

A repository with several checkpoints is named as `repo:file.safetensors`.

### GGUF

Community GGUF files of Qwen-Image's and FLUX.1-schnell's transformers load by
the repository and the quantisation together. The rest of the model comes from
the `base_model` on the model card.

```bash
kvad pull city96/Qwen-Image-gguf:Q4_K_S
kvad images make "a red fox in fresh snow" --model city96/Qwen-Image-gguf:Q4_K_S
```

A Q4_K_S holds 9.1 GB less than q8 for Qwen-Image, and 5.5 GB less for FLUX,
for a step that is 5 to 18% longer.

## Options

```text
kvad images make PROMPT [--out FILE] [--model MODEL] [--size WxH] [--steps N]
                        [--guidance F] [--negative TEXT] [--seed N]
                        [--lora NAME[:SCALE]]...
```

Anything left out is the model's own default. `--lora` applies a
[LoRA](/docs/lora/) for this picture only.

```bash
kvad images            # the gallery, as a list
kvad images rm 12
```

## From the API

`POST /v1/images/generations` is OpenAI's images endpoint, with the settings
OpenAI has no field for beside its own.

```bash
curl http://127.0.0.1:5823/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "a red fox in fresh snow",
    "size": "1024x1024",
    "steps": 30,
    "guidance_scale": 5,
    "seed": 5,
    "response_format": "url"
  }'
```

With `"stream": true` the server sends an event per denoising step, a rough
preview beside each when `partial_images` asks for one, and a completed event
per image. See [OpenAI-compatible API](/docs/api/#images).

## Memory

The VAE decode is the most memory an image asks for: it is the only stage at
full resolution. If a picture fails with a message about GPU memory, another
model is probably loaded. `kvad ps` shows what is, and `kvad unload` frees it.

## The pictures on the front page

The four pictures on [the front page](/) were made with the commands below,
on an M5 Pro, with SDXL's defaults: 1024², 30 steps, guidance 5. Each took 82
to 86 seconds to denoise and 9 to decode. The same seed draws the same
picture.

```bash
M=stabilityai/stable-diffusion-xl-base-1.0
kvad images make "a lighthouse on a rocky Norwegian coast at dusk, calm sea, warm light in the lamp room, long exposure photograph" --model $M --seed 11
kvad images make "a red fox in fresh snow at the edge of a birch forest, early morning light, wildlife photograph" --model $M --seed 5
kvad images make "a wooden stave church in a green valley under low clouds, watercolour" --model $M --seed 3
kvad images make "an open book on a wooden desk by a window, a single candle, oil painting, Dutch golden age" --model $M --seed 21
```
