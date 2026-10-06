---
title: LoRAs
description: Apply published LoRAs to image and video models per request, and train one for SDXL on your own pictures.
---

## Applying one

LoRAs apply to Qwen-Image, FLUX, SDXL, Stable Diffusion 1.5 and LTX-2.5. They
are applied at run time, per request: the model stays loaded as it was, and
each request chooses its own LoRAs and strengths. Nothing is merged into the
weights and nothing is reloaded.

```bash
kvad pull lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors

kvad images make "a tiny astronaut hatching from an egg on the moon" \
    --model Qwen/Qwen-Image --steps 8 \
    --lora lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
```

A strength goes after a colon, `--lora NAME:0.8`, and is 1 if left out. Up to
four can be applied to one request. A LoRA is named the way it was pulled:
`repo`, `repo:file.safetensors`, or a path on this machine.

The formats people publish are all read: PEFT's, kohya's, diffusers' and Black
Forest Labs', including text-encoder and convolution pairs.

That Lightning LoRA is worth having. With it Qwen-Image draws in 8 steps
instead of 50: 68 seconds at 1024² on an M5 Pro against 448, and a sharper
picture.

A LoRA that does not fit the model fails the request, and the message names
the layers it adapts that the model does not have.

## Training one

A LoRA for SDXL can be trained on a folder of pictures, each with a caption in
a `.txt` file of the same name beside it. That is the kohya and diffusers
convention.

```text
my-photos/
  01.jpg
  01.txt      "a photo of a red boat, my-style"
  02.jpg
  02.txt
```

```bash
kvad-gpu tune --data ./my-photos --name my-style
```


The result is `~/.local/share/kvad/loras/my-style.safetensors`, which
`kvad images --lora` applies and so does diffusers.

```bash
kvad images make "a lighthouse, my-style" \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora ~/.local/share/kvad/loras/my-style.safetensors
```

### From the web UI

The Training page runs the same training as a job on the server. Upload the
pictures and their captions on the Datasets page, choose them and a name on
the Training page, and add a prompt or two to draw as it learns. The page
shows the validation loss and a row of pictures for each prompt, one at
every measurement. The finished LoRA is then a choice on the Images page.

This arrived after v0.12.0. Until the next release it needs a build from
source.

### What a run takes

Measured on an M5 Pro with 48 GB, one picture a step:

| Size | A step | While training | At most |
|---|---|---|---|
| 512² | 1.9 s | 6.4 GB | 7.2 GB |
| 1024² | 7.3 s | 7.6 GB | 10.6 GB |

A thousand steps at 1024², SDXL's own size, is about two hours. At 512² it is
about half an hour, and SDXL draws less well there.

### Options

| Option | Default | |
|---|---|---|
| `--size N` | 1024 | Pixels a side, a multiple of 64. |
| `--rank N` | 16 | |
| `--alpha F` | the rank | The LoRA is scaled by alpha / rank. |
| `--steps N` | 1000 | |
| `--lr F` | 1e-4 | |
| `--eval-every N` | 100 | Steps between validation measurements. |
| `--holdout N` | a tenth | Pictures kept out of training, to measure on. |
| `--caption TEXT` | | The caption of every picture that has none. |
| `--from FILE` | | Go on from a LoRA this wrote. |
| `--cap GB` | ¾ of memory | End the run if its memory passes this. |

### Which step is kept

The training loss is not the number to watch: it keeps falling while the LoRA
memorises your pictures. So some pictures are held out, the loss on those is
measured every `--eval-every` steps, and the LoRA written to
`NAME.safetensors` is the step where that was lowest. The last step is written
beside it as `NAME.last.safetensors` when it is a different one.

`ffmpeg` is needed, to decode the pictures. `--cap` is there because a
backward pass that does not fit in memory is not refused by macOS: it takes
the machine down.

[Training a LoRA](/docs/internals/tune/) has the long form, with one run
described from start to finish.
