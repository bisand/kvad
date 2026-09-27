# Plan: the community's GGUFs of the image and video models

Written 2026-09-27 for [#42](https://github.com/bisand/kvad/issues/42). The
headers below were read from the Hub with range requests, and the model cards
through its API. #42 also asks for original-layout single-file safetensors
(SDXL and SD 1.5 fine-tunes). That is left for later: it is a key map with no
quantisation in it, and nothing here depends on it.

## Why

Kvad quantises the big models itself, from their bf16, at load, and keeps
the blocks in `qcache`. That only ever makes Q8_0: nothing here decides which
matrices can take fewer bits. The community's GGUFs have decided that per
matrix. Below q8 they are how the big models fit beside anything else:

| Model | Q8_0, the size of Kvad's own q8 | Q4_K_S |
|---|---|---|
| Qwen-Image transformer | 21.8 GB | 12.1 GB |
| FLUX.1 transformer | 12.7 GB | 6.8 GB |
| LTX-2.5 distilled DiT | 23.6 GB | 15.3 GB |

The sizes are the files'.

## What #42 got wrong, or did not know

- **Only FLUX's files need a name map.** city96's FLUX files use Black
  Forest Labs' own names (`double_blocks.N.img_attn.qkv`), where diffusers
  splits q, k and v. The Qwen-Image files (city96's, QuantStack's,
  unsloth's) use diffusers' names, which Kvad reads already. The LTX-2.5
  file (`Abiray/LTX-2.5-Distilled-GGUF`) uses the reference's own names,
  which Kvad also reads, and carries the reference's config in its header.
- **A file's name is not its make-up.** city96's Qwen-Image "Q4_K_S" is 716
  Q4_K matrices and 124 Q5_K. unsloth's "Q4_K_M" mixes Q4_K, Q5_K, Q6_K,
  Q8_0 and six bf16 matrices. Every file keeps its norms and biases in f32,
  and the six small matrices at each end of Qwen-Image in bf16. So the
  loader takes each tensor as it comes.
- **Every repo names its base.** Each model card's `base_model` names the
  diffusers repo it was made from, which is where the text encoder, the
  VAE, the scheduler and the transformer's config come from.
- **candle has Metal kernels for every type these files use,** Q4_0 to
  Q6_K, taken from llama.cpp. They run on the GPU's ordinary cores. Kvad's
  own Q8_0 runs on the M5's matrix units (`mpp`), 1.5× faster for FLUX, so
  a Q4_K file could fit in less and still run slower. Step 1 measured it:
  twice as slow a step for Qwen-Image.
- **Z-Image**, #42's most-downloaded example, is not a pipeline Kvad
  implements, so its GGUFs have nothing to load into. That is its own
  issue.

## How a GGUF is read

`kvad_gpu::gguf::Gguf` parses the header with candle's reader and reads the
data past the page cache, as every other weight is read. It is a
`SimpleBackend`, so a `Reader` over it reads a norm or a bias like a
safetensors file's. A matrix is offered as blocks first
(`Reader::blocks`): the `Loader` puts a quantised one on the device as its
maker wrote it, and never quantises it again. A Q8_0, Q4_K, Q5_K or Q6_K
matrix goes to the M5's matrix units, as Kvad's own Q8_0 does (step 5);
any other type goes to candle's `QMatMul`. A plain matrix, like
Qwen-Image's bf16 ones, is read dense.

A GGUF's matrices are quantised, which on candle means f32 activations, so
a pipeline with a GGUF denoiser computes in f32. Its text encoder is
quantised to q8 when nothing else is asked for, since in f32 Qwen-Image's
would be 28 GB.

The unread-weights guard (`finish_gguf`) holds for these files as for any
checkpoint: a tensor in the file that nothing read refuses the load.

## Naming and pulling

A GGUF model is named `repo:QUANT`, as Ollama and the Hub already do:
`city96/Qwen-Image-gguf:Q4_K_S`. The quant picks the one file in the repo
whose name ends in `-Q4_K_S.gguf` or `_Q4_K_S.gguf`, whatever the case.
Pulling it fetches that file and, from the model card's `base_model`,
everything of the base except its transformer. The base's transformer, 41 GB
for Qwen-Image, is never downloaded for a GGUF.

## The order it is built in

1. **The reader, and Qwen-Image from a GGUF**, as `examples/gguf.rs` and a
   `--gguf` option on `examples/sdxl.rs`. It is checked four ways:
   - the reader against candle's own GGUF reader;
   - candle's Metal kernel for each type in a file against f32 on the CPU;
   - a Q8_0 file's blocks against Kvad's own Q8_0 of the base;
   - and a picture, with its speed and peak memory, against Kvad's q8.
2. **Names:** `repo:QUANT` through pull, the model list, admission by
   memory, load, the CLI and the web page.
3. **FLUX** from city96's files, with the name map to diffusers' layout.
4. **LTX-2.5's distilled DiT** from a GGUF, for the fast pipeline. The dev
   model, and DFR's DiT with its LoRA fused in, stay Kvad's own.
5. **The K-quants on the M5's matrix units.** Step 1 measured a Q4_K_S
   Qwen-Image at twice the time a step of Kvad's q8, and step 4 an LTX-2.5
   Q4_K_M at 3.3 times the denoise. Done; see below.

## What was built, and what it measured

Added 2026-09-27, as the steps land. Every run below is on the M5 Pro
(48 GB): Qwen-Image, "a red fox sitting in fresh snow, photograph", seed 5,
1024² in 20 steps, the text encoder at q8. Peaks are the process's peak
memory footprint, from `/usr/bin/time -l`.

**Step 1, the reader and Qwen-Image from a GGUF.** The files are city96's,
`city96/Qwen-Image-gguf`, made from `Qwen/Qwen-Image`.
- **The reader** gives the same bytes and numbers as candle's own GGUF
  reader, for Q8_0, Q4_0, Q4_K, Q5_K, Q6_K and plain F32, F16 and BF16
  (`gguf::tests`).
- **city96's Q8_0 is Kvad's own q8.** All 840 of its Q8_0 matrices are
  byte-identical to Kvad's quantisation of the base's bf16: 637,009,920
  blocks, none different. The file differs from Kvad's q8 only in its six
  small matrices, which it keeps in bf16 where Kvad quantises them.
- **candle's Metal kernels are right for every type in the files.** Each is
  against the same matrix dequantised and multiplied in f32 on the CPU:

  | Type | 4096 rows | One row | 4096 rows, narrowed | Speed, 4096 rows |
  |---|---|---|---|---|
  | Q8_0 | 73.7 dB | 119.4 dB | 73.7 dB | 6.9 TFLOP/s |
  | Q5_K | 73.7 dB | 119.6 dB | 73.7 dB | 5.9 TFLOP/s |
  | Q4_K | 58.4 dB | 119.7 dB | 58.4 dB | 6.3 TFLOP/s |

  The many-row kernel dequantises into half precision, which is why it is
  below the one-row kernel.
- **What the types cost in the weights**, against the base's bf16, every
  10th matrix of each type:
  - Q4_K: 22.1–22.9 dB, mean 22.7;
  - Q5_K: 28.2–28.6 dB, mean 28.5;
  - Kvad's Q8_0 of the same matrices: 44.1–45.4 dB, mean 45.1.

  The kernels' own error is 35 dB below the four-bit rounding, so what a
  Q4_K file draws is the file's, not the kernels'.
- **The pictures:**

  | Transformer | Load | Step | Peak | Against Kvad's q8 picture |
  |---|---|---|---|---|
  | Kvad's q8 | 5.0 s | 7.10–7.61 s | 38.8 GB | — |
  | Q8_0 GGUF | 5.1 s | 7.12–7.44 s | 39.2 GB | 45.3 dB |
  | Q4_K_S GGUF | 3.5 s | 13.85–14.39 s | 29.7 GB | 25.2 dB |

  - Loads are from a warm disk: the q8 run's first load, cold, took 50.6 s.
  - The Q8_0 file draws the same fox. The 0.44 GB more at the peak is its
    six bf16 matrices, held in f32 for the f32 pipeline.
  - The Q4_K_S file draws the same fox in the same pose, with the tail a
    flatter grey. It saves 9.1 GB and takes twice as long a step: its
    matrices run on candle's kernels, on the GPU's ordinary cores, where
    Kvad's q8 and the Q8_0 file's run on the M5's matrix units.

So step 5 is needed. A Q4_K file is how Qwen-Image fits beside other
work, but for now it costs twice the time.

**Step 2, names.** A GGUF is a model named `repo:QUANT` everywhere a model
is named: pull, the listing, admission, load, delete, the CLI, the Models
and Images pages.
- **`kvad::gguf`** splits the name and finds the one file in the repo that
  is that quantisation, by the end of its name. It reads the base from the
  model card, whose `base_model` is a string, an inline list or a block
  list, depending on who wrote it. It takes no GGUF split into parts.
- **The listing** makes each GGUF in a cache entry a model of its own, and
  the entry itself one only if it is something else as well. A GGUF is its
  base's pipeline (`hub::pipeline`), so what the service already does with
  a pipeline (its kind, its default backend, whether its files are all
  here) it does for a GGUF unchanged. One whose card is missing, or whose
  base is not downloaded, says which.
- **A pull** asks the Hub's API for the repo's files, fetches the card, and
  asks the base's `model_index.json` what it is before anything large
  moves. So a GGUF of a model with no GGUF loader here, FLUX for now, costs
  8 KB to refuse. Then the file, and the base's files but its transformer's
  weights: `qwen::fetch_base`, the list the load reads. The Qwen-Image base's
  41 GB transformer is never fetched.
- **Admission** charges the base's text encoder at the quantisation asked
  (q8 if none), the VAE, and the file's own blocks with its plain tensors
  widened to f32. Through the service the Q4_K_S was charged 19.9 GB, the
  same as it counted once loaded, and peaked at 28 GB making a 1024² image.
  A pipeline is charged its weights, not its working memory, as Kvad's own
  q8 is.
- **Delete** takes the one file, link and blob, and the cache entry once
  nothing is left in it. `hub::remove` does it for the service, the CLI and
  the engine's own command, which each used to delete the entry whole.
- **The Models page** offers a pull when the search box holds a GGUF's
  name, since the Hub search it runs lists text models. After a pull, the
  CLI names the command for the model's kind: `kvad images make` here, where
  it used to say `kvad run`.

Checked through a test server on this branch:
- the Q4_K_S, pulled by name with its card missing, found its base, fetched
  nothing else, and drew the same picture as the example, pixel for pixel;
- a quantisation the repo does not have is refused with the ones it has;
- a FLUX GGUF is refused before its file is fetched;
- a name with a colon that is not a quantisation is refused as such;
- the Models page's pull, and `kvad rm` of the Q8_0, which left the Q4_K_S
  and the card.

**Step 3, FLUX from city96's files.** `city96/FLUX.1-schnell-gguf`, made from
`black-forest-labs/FLUX.1-schnell`. A GGUF of FLUX.1-dev is refused for its
guidance input, as its diffusers repo is: a pull reads the base's
transformer config before anything large, and `city96/FLUX.1-dev-gguf:Q4_K_S`
was refused after 20 KB.
- **`Gguf::mapped`** presents a file under the names a loader asks for.
  Each name is some rows of one or more of the file's tensors, and the
  rows of a quantised matrix are whole blocks, so a mapped matrix is a
  span of the file's bytes and nothing is dequantised to split it. Every
  row of a tensor the map names must land in exactly one name, so a map
  that drops rows or uses them twice fails to open.
- **`flux::gguf_map`** is diffusers' own conversion of Black Forest Labs'
  checkpoints, backwards. Three things differ besides the names:
  - each double block's two streams have one `qkv` matrix, and diffusers'
    q, k and v are its thirds;
  - each single block's `linear1` is q, k, v and the MLP's input, at four
    times the width;
  - the last modulation's shift and scale are the other way round.
- **Checked against the base**, every tensor, with the Q8_0 file:
  - the mapped file is exactly the 1156 tensors diffusers' transformer has;
  - all 494 Q8_0 matrices are Kvad's own Q8_0 of the base's bf16, byte for
    byte (369,819,648 blocks), q, k and v and `proj_mlp` included;
  - of its 662 plain tensors, 365 are the base's exactly, and the rest,
    which the file narrowed to f16, are 133 dB or closer. The swapped
    modulation is among them; the other way round it would be about 0 dB.
- **The pictures**, 1024² in schnell's 4 steps:

  | Transformer | Load, warm | Step | Peak | Against Kvad's q8 picture |
  |---|---|---|---|---|
  | Kvad's q8 | 3.3 s | 7.01–7.93 s | 39.3–39.5 GB | — |
  | Q8_0 GGUF | 3.5–3.6 s | 7.42–8.08 s | 39.4–39.9 GB | 44.4 dB |
  | Q4_K_S GGUF | 2.5 s | 15.40–15.50 s | 33.9 GB | 24.3 dB |

  - Four runs each of the first two, alternating and then in the other
    order: the step times overlap, and neither order made one faster.
  - The Q8_0 draws the same fox; the difference is the file's embedders and
    final layer, kept in f16 where Kvad quantises them.
  - The Q4_K_S draws the same fox in the same pose, with a shorter tail. Its
    Q4_K matrices are 20.4–24.3 dB from the bf16, and candle's kernel is
    53.7 dB from their own numbers. It saves 5.5 GB and, as Qwen-Image's
    did, takes about twice as long a step.
- **Through the service**, `city96/FLUX.1-schnell-gguf:Q4_K_S` was pulled by
  name, which fetched nothing more of the base, and drew the example's
  picture pixel for pixel, at a peak of 32 GB. Admission charged it
  12.5 GB, as it counted itself once loaded.

**Step 4, LTX-2.5's distilled DiT.** `Abiray/LTX-2.5-Distilled-GGUF`, whose
card names `Lightricks/LTX-2.5`. Its "Q4_K_M" is 1292 Q5_K matrices, 350
Q4_K and 102 Q6_K, and its header carries the reference's config and the
Gemma checkpoint the DiT wants, as Lightricks' own file's metadata does.
- **Names:** the reference's own, without the `model.diffusion_model.` the
  safetensors file files them under, so the file is read with that prefix
  (`Gguf::prefixed`). `metadata()` reads a GGUF's header too, so the DiT's
  loader and the text path, which reads its config and its connectors from
  the DiT's file, take the GGUF's path where they took the file's.
- **The connectors are quantised in the file** and are read that way.
  Their feed-forward added its answer to the bf16 stream as it came, which
  only a dense matrix gives in bf16; it asks for the stream's dtype now
  (`Linear::forward_in`), which changes nothing for a dense one.
- **bf16 into a K-quant:** candle's quantised kernels take f32 only, and on
  Metal anything else trips an assertion. LTX's DiT computes in bf16, so a
  K-quant projection widens its input and answers in f32, as Kvad's Q8_0
  on the M5's matrix units always has.
- **The fast pipeline only.** DFR fuses the detailing LoRA into the DiT and
  the guided pipeline runs the dev model, and neither can be made from the
  file, so a GGUF's defaults have neither: guidance and `pipeline: dfr` are
  refused. A rate above 30 fps runs the fast pipeline, as LTX-2.5 did
  before DFR, with the DiT told that rate.
- **Identity:** LTX-2.5 is known by its DiT's file, which is the file a
  GGUF of it is there not to need. So a GGUF whose base is Lightricks'
  repo is LTX by that name. A pull fetches the file and everything a load
  reads but the DiT (the text encoder, the three decoders, the upsampler
  and the duration head); the 39 GB DiT is never fetched. Admission charges
  the same generation's peak as Kvad's own, which was measured with it.
- **Checked against the distilled DiT,** every 20th matrix of each type:
  - all 2605 of its plain tensors are the base's, 2603 exactly and the two
    registers within f16's rounding;
  - Q5_K 27.5–28.9 dB from the bf16, Q4_K 19.8–24.0, Q6_K 32.2–36.9, and
    Kvad's own Q8_0 of the same 40.7–47.6: quantised from these weights;
  - candle's kernels are 73.5–73.7 dB from the matrices' own numbers,
    except on the 32-row gate matrices in Q4_K, 40.7 dB, still 18 dB below
    what the four bits cost.
- **Through the service**, 768×512 × 121 frames at 24 fps, seed 1, "a red
  fox trots through fresh snow, photograph":

  | DiT | Text | Denoise | Decode | Total | Peak |
  |---|---|---|---|---|---|
  | Kvad's q8 | 9.9 s | 61.6 s | 17.4 s | 89.5 s | 25 GB at most |
  | Q4_K_M GGUF | 4.7 s | 202.7 s | 21.7 s | 229.8 s | 21 GB |

  The same woods and the same fox on the same path, towards the camera and
  away to the left; the frames are 15.0–17.5 dB apart, as two clips that
  diverge in detail are. Single runs. The q8 peak is the process's after
  both, so an upper bound. The denoise is 3.3 times as long, more than the
  image models' 2: Kvad's q8 DiT runs near MLX's rate on the matrix units,
  and the K-quants' kernels, on the GPU's ordinary cores, run the gates'
  32-row matrices at 0.4–0.5 TFLOP/s.

**Step 5, the k-quants on the M5's matrix units.** The Q8_0 kernel
(`mpp`) unpacks each step's `64 × 32` slab of weights into threadgroup
memory as f16, then multiplies on the matrix unit. Only the unpacking knows
what a block is, so it became one decoder per format, and the kernel a
template over them: Q8_0, Q4_K, Q5_K and Q6_K, each with f16 or bf16 in and
f32, f16 or bf16 out.
- **A run of eight weights decodes on its own.** A k-quant super-block is
  256 weights in sub-blocks of 32 (Q4_K, Q5_K) or 16 (Q6_K), each with its
  own scale, and eight weights never straddle two. Each decoder is GGML's
  own dequantisation, in f32, rounded once to f16.
- **Q8_0 is unchanged bit for bit:** its decoder is the multiply in f16 it
  always did, and its outputs on fixed inputs, 836 KB of them over four
  shapes and three output dtypes, were the same before and after.
- **The k-quants against their own numbers**, dequantised by candle and
  multiplied in f32 on the CPU, f32 and bf16 in, at every ragged edge:
  87–98 dB, where candle's own many-row kernel is 53–58. A Q6_K decoder with
  its quarters' top bits taken from the wrong shift fails the test.
- **Rates** at the LTX-2.5 DiT's shapes (24576 rows), bias and bf16 in and
  out, in TFLOP/s:

  | `[k] → [n]` | Q8_0 | Q4_K | Q5_K | Q6_K | candle's, Q4_K–Q6_K |
  |---|---|---|---|---|---|
  | 4096 → 4096 | 24.1 | 20.8 | 19.2 | 18.9 | 6.5–6.9 |
  | 4096 → 16384 | 22.6 | 19.3 | 16.8 | 16.5 | 6.4–6.9 |
  | 16384 → 4096 | 21.3 | 19.0 | 17.2 | 16.4 | 6.0–7.0 |
  | 4096 → 2048 | 21.7 | 19.0 | 17.5 | 17.3 | 6.2–6.6 |
  | 2048 → 4096 | 20.5 | 17.4 | 16.7 | 16.2 | 6.0–6.5 |
  | 4096 → 32 | 4.9 | 5.8 | 5.8 | 5.8 | 3.0–3.1 |

  The k-quants are 75–90% of Q8_0's rate, their decoding being more reads
  and more arithmetic a run, and 2.6–3 times candle's.
- **The models**, the same runs as the steps before:

  | Model | Kvad's q8 | GGUF on candle's kernels | GGUF now | Peak, GGUF |
  |---|---|---|---|---|
  | Qwen-Image Q4_K_S, a step | 7.10–7.61 s | 13.85–14.39 s | 8.31–8.41 s | 29.6 GB, against q8's 38.8 |
  | FLUX Q4_K_S, a step | 7.01–7.93 s | 15.40–15.50 s | 7.56–7.77 s | 33.9 GB, against 39.3–39.5 |
  | LTX-2.5 Q4_K_M, the denoise | 51.1–61.6 s | 202.7 s | 59.0–62.1 s | 20 GB, against 25 at most |

  - Qwen-Image's step is 9–18% longer than q8's, and FLUX's 5–9% than q8's
    in the same session; LTX's denoise is within q8's own spread. Each is the model's same picture as
    on candle's kernels: 47.7 dB for Qwen-Image, 44.5 for FLUX, and for
    LTX the same clip, 22.9–26.2 dB a frame, as a video drifts over two
    stages of slightly different rounding.
  - So a GGUF now buys its memory for about q8's time: 9.1 GB for
    Qwen-Image, 5.5 GB for FLUX and about 4 GB for LTX-2.5.
