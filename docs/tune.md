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

**4. Draws samples, if asked.** Each `--sample "a prompt"` is drawn before
the first step and at every measurement, into `NAME.samples/` as
`N-STEP.png` for the Nth prompt. A prompt's seed never changes, so a row of
its samples differs by what the LoRA learned and by nothing else, and the
first of the row is the model's own. It is the loop `kvad images` runs, on
the same noise: a sample is what a request with that LoRA, seed, size and
step count draws. The prompts are read by the text encoders with the
captions, before the UNet is loaded, and the VAE's decoder, 0.1 GB, is kept
for the run. Samples are the run's own size and 20 steps unless
`--sample-size` and `--sample-steps` say otherwise.

They cost time and no memory. Measured on an M5 Pro beside a run at 1024²,
whose own peak is 10.4 GB:

| a sample | takes | reaches | the run, at most |
|---|---|---|---|
| 512² | 16 s | 8.0 GB | 10.4 GB |
| 768² | 39 s | 8.4 GB | 10.4 GB |
| 1024² | 72 s | 8.7 GB | 10.4 GB |

Beside a run at 512², whose own peak is 7.1 GB, two 512² samples reach
8.7. The steps between are as fast and their losses the same to the last
digit, and two runs' samples are the same file.

It was not so at first: a 768² sample reached 18.3 GB, nearly all of it the
VAE's decoder, and a 1024² one would have reached about 28. The decoder now
runs its convolutions in bands of rows and its norms a few groups at a time
(`nn::BAND`), which gives the same picture to the bit.

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
| 1024² | 7.3 s | 7.6 GB | 10.6 GB |

"At most" is a step's backward pass. At 1024² it was 13.7 GB, the VAE's
encoder reading the pictures before the UNet is loaded, until the encoder
too ran its convolutions in bands (`nn::BAND`); reading them now reaches
less than a step does. A
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

## From the web UI

The Training page starts the same run and follows it (#77). The pictures are
a dataset: the Datasets page uploads them with their `.txt` captions into a
folder under the data directory's `datasets/`, shows each beside its
caption, and lets one be written there. A run is refused before it starts
for a picture with no caption, naming every one, unless it is given a
caption for those.

The server does not train in its own process. It starts `kvad-gpu tune
--progress json` beside it and reads what that writes, one JSON object a
line: every step, every measurement, every sample drawn. Three reasons:

- **The memory comes back.** What a run takes is the system's again the
  moment the process ends. In the server it would stay in candle's buffer
  pool and the allocator for as long as the server ran.
- **The ceiling ends the run, not the server.** `--cap` kills the process
  that passes it.
- **A failed backward pass costs one job**, and not everybody's models.

The run's memory is set aside in the server's budget before it starts, so
no model is loaded into it meanwhile, and a run that does not fit beside
what is loaded is refused with what holds the rest. What is set aside is
the measured peak and a fifth: 8.6 GB at 512², 10.3 at 768² and 12.7 at
1024², and at least 10.4 where samples are drawn. The sizes offered are
those three, the ones measured.

The page draws the validation loss alone, for the reason above, against the
model's own before the first step, and a row of pictures for each sample
prompt: the model's own first, then one for each measurement. Stopping a
run sends it the interrupt Ctrl-C would; it finishes its step, measures,
and keeps what it has. A run started by a server ends when that server
does.

The LoRA is written to the data directory's `loras/NAME.safetensors`. LoRAs
there are listed with the models, `kvad ls` among "trained here", and the
Images page offers them for the models they fit.

Checked on an M5 Pro with six made-up pictures at 512²: a 6-step run to its
end (validation loss 0.0054 to 0.0051, three samples), and a 40-step run
stopped from the page at step 10, which kept that step. The second LoRA was
then applied by a request, 768² in 12 steps, by the path it is listed
under. No long run has been made through the page.

## From `kvad`, through the server

`kvad tune` asks a running server for the same job, from this machine or
another:

```bash
kvad tune --data ./my-photos --name my-style --sample "a lighthouse, my-style"
kvad images make "a lighthouse, my-style" --lora my-style
kvad jobs pictures 12 --out ./samples      # what job 12 drew as it learned
```

`--data` sends the folder to the server as a dataset first, a request a
file; `--dataset` names one it has. `kvad datasets add DIR`, `put`, `get`
and `rm ID FILE...` are the Datasets page's upload, and `kvad tune options`
says what a run there can be asked for and what memory each size is set
aside. A LoRA trained on the server is applied by the one word it was
named, as above: the path it has there means nothing on another machine.

Checked against a server with the six made-up pictures at 512²: the folder
sent, a 6-step run followed to its end, its three samples fetched, and an
image drawn with `--lora` and the LoRA's name.

## Not here yet

- **Going on from a LoRA** (`--from`), `--alpha` and `--holdout` from the
  page or `kvad tune`. `kvad-gpu tune` has them.
- **Other models.** FLUX and Qwen-Image are refused by name: their blocks'
  gradients are checked (#74), and a step through all of them has not been
  made to fit or been measured. An SDXL checkpoint in one file is refused
  too; a fine-tune in diffusers' layout trains.
- **One square size.** No aspect-ratio buckets, no flips, no caption
  dropout, one picture a step, a constant learning rate.
- **The text encoders** are not trained.
