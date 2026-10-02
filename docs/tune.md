# Training a LoRA for an image model

Written 2026-10-01 for [#75](https://github.com/bisand/kvad/issues/75). The
first part of it: the training loop itself, for SDXL, run in the process
that asks for it. What is not here yet is at the end.

```bash
kvad-gpu tune --data ./my-photos --name my-style
kvad images make "a lighthouse, my-style" --model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora ~/.local/share/kvad/loras/my-style.safetensors
```

The folder holds pictures (`jpg`, `png`, `webp`, `bmp`), each with its
caption in a `.txt` of the same name beside it, which is the kohya and
diffusers convention. What comes out is one `.safetensors` file in PEFT's
key names under diffusers' `unet.`, which `kvad images --lora` applies and
so does diffusers.

## What a run does

**1. Reads every picture and caption once.** The UNet is never shown a
picture. It is shown a *latent*, what the VAE's encoder makes of one, and
what the two text encoders make of its caption, and neither changes while a
LoRA on the UNet is trained. So each is read before the first step, kept
under the data directory's `tune-cache/` for the next run, and the encoders
are gone before the UNet is loaded. A picture is fitted to a square first:
shrunk until its short side is `--size`, each new pixel the mean of the
ones it covers, and cut from its middle. `ffmpeg` decodes it, as it does a
video's first frame.

**2. Takes steps.** Each one:

```text
x₀  a latent drawn from the picture's Gaussian (the encoder gives a mean and a spread)
t   a noise level, any of SDXL's thousand, each as likely as another
ε   noise, the latent's shape

x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε          the picture, noised
loss = mean((UNet(x_t, t, caption) − ε)²)   the noise it failed to find
```

`ᾱ_t` is how much of the picture level `t` keeps: 0.999 at the first and
0.005 at the last (`schedule::Noising`). It is the same mix drawing starts
from and walks down, so the LoRA is trained on what drawing shows the model.

The LoRA is a pair of thin matrices on each of the UNet's 560 attention
projections (`to_q`, `to_k`, `to_v`, `to_out.0`): the layer answers
`W·x + (alpha/rank) · B·(A·x)`. `A` starts random and `B` at zero, so the
first step draws exactly what the model draws. Only `A` and `B` are trained,
23 M numbers at rank 16 beside 2.57 B that are not, by AdamW.

**3. Measures, and keeps the best.** Every `--eval-every` steps the
validation loss is taken, and the LoRA is written if it is the lowest yet.
The last step's is written beside it as `NAME.last.safetensors` where that
is another.

## Why the training loss is not the number to watch

How much noise there is to find decides most of a step's loss. At level 900
the input is nearly all noise and the answer is nearly the input; at 100 the
noise is a faint grain to pick out of a picture. So a step's loss says which
level it drew, and little else.

The validation loss is on the same pictures, at the same four levels (125,
375, 625, 875), with the same noise, every time. Two measurements differ by
what the LoRA learned between them and by nothing else. It is taken on
pictures the run never trains on: a tenth of the set, from one to four,
unless `--holdout` says otherwise. A set of fewer than five has none to
spare, and the loss is then on pictures it trained on, which says how well
those are fitted and nothing about any other; the run says so.

It is printed against the model's own loss on those pictures before the
first step, because that is what a LoRA has to beat.

## One run

Twelve made-up pictures in one made-up style, flat discs with thick dark
rings on a flat ground, each captioned `a kvadring picture of three red
discs on a purple ground` and the like. 300 steps at 512², rank 16, on an
M5 Pro:

| after | validation loss | against the model's own |
|---|---|---|
| 0 steps | 0.0147 | |
| 50 | 0.0128 | −12.9% |
| 100 | 0.0126 | −14.4% |
| 200 | 0.0123 | −16.7% |
| 250 | 0.0125 | −14.9% |
| 300 | 0.0126 | −14.7% |

The step kept is 200. After it the loss on the held-out picture rises while
the training loss goes on falling: the LoRA has started to learn the eleven
pictures rather than what they share. With the LoRA, `a kvadring picture of
three red discs on a blue ground` draws three flat red discs with dark rings
on a flat blue ground; without it, the same seed draws shaded hoops.

## What it takes

Measured on an M5 Pro with 48 GB, one picture a step:

| size | a step | while training | at most |
|---|---|---|---|
| 512² | 1.9 s | 6.4 GB | 7.2 GB |
| 1024² | 7.3 s | 7.6 GB | 13.7 GB |

"At most" at 1024² is the VAE's encoder reading the pictures, before the
UNet is loaded; a run whose pictures were read before does not pay it. A
measurement is four forward passes for each held-out picture. A thousand
steps at 1024², SDXL's own size, is about two hours; at 512² about half an
hour, and SDXL draws less well there.

The run ends itself if its memory passes `--cap` gigabytes, three quarters
of the machine's unless told. A backward pass that does not fit is not
refused by macOS: it takes the machine down.

## Three things found on the way

- **A loss that is a mean loses its gradient in half precision.** The mean's
  gradient for each number of the UNet's answer is the error there divided
  by how many numbers there are, 65 536 at 1024², so about 1e-5. The UNet
  is in f16, and so is the gradient passing back between its stages, and
  f16 rounds numbers that small to a few levels. Against the gradients the
  UNet finds in f32, the cosine was 0.934 and the length 6.6% over. So the
  loss goes into `backward` as a sum and the factors' gradients, which are
  in f32, are divided back: 0.999988.
- **A step's gradient is the loss's own slope**: all the `B`s moved a little
  way along their gradient and back, the loss rises by 0.999 of what
  `backward` says it should, at a low, a middle and a high noise level. In
  f32. In f16 the same measurement gives anywhere from 0.35 to 0.98, because
  a nudge small enough for the slope to hold is mostly rounded away, so it
  is a test and not a check every run makes. What every run checks at its
  first step is that each layer's `B` has a gradient that is a number and
  is not nothing.
- **A long run on one thread needs an autorelease pool a step.** Metal hands
  back command buffers and the like to be freed when the thread's pool is
  emptied, and a command-line thread has none. The run grew 5 MB a step,
  and by its factors' size at every save, until each step, measurement and
  save had a pool of its own (`common::pooled`).

`cargo test --release -p kvad-gpu tune::tests -- --ignored --test-threads 1`
runs the four checks that need SDXL's weights; the last is that the file a
run writes, set on the UNet as a request sets a LoRA, gives the answer the
run's own factors gave, to the bit.

## Not here yet

- **`kvad tune` through the service**, as a job with its loss on a chart
  (#77). `kvad-gpu tune` runs in its own process and needs the UNet's
  memory to itself.
- **Samples at checkpoints**: a few fixed prompts and seeds drawn at each
  measurement. They need the VAE's decoder, whose peak at 1024² is 21 GB.
- **Other models.** FLUX and Qwen-Image are refused by name: their blocks'
  gradients are checked (#74), and a step through all of them has not been
  made to fit or been measured. An SDXL checkpoint in one file is refused
  too; a fine-tune in diffusers' layout trains.
- **One square size.** No aspect-ratio buckets, no flips, no caption
  dropout, one picture a step, a constant learning rate.
- **The text encoders** are not trained.
