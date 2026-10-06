[← Back to the README](../README.md)

# Crate 3: `kvad-gpu` — the same model, handed to a framework

```bash
kvad-gpu run  --prompt "Why is the sky blue?"
kvad-gpu run  --device metal --dtype bf16 --model Qwen/Qwen2.5-0.5B-Instruct
kvad-gpu chat
```

[`model.rs`](../crates/gpu/src/model.rs) is the Llama forward pass again, in
[candle](https://github.com/huggingface/candle). Read it next to
[`llama.rs`](../crates/llm/src/model/llama.rs): the structure is line for line the
same, and every operation is one you already wrote.

It is three forward passes now rather than one.
[`gpt2.rs`](../crates/gpu/src/gpt2.rs) and
[`deepseek.rs`](../crates/gpu/src/deepseek.rs) sit beside it, with
[`common.rs`](../crates/gpu/src/common.rs) holding what they share — which is a
shorter list than it looks: two weight layouts behind one `forward`, an
embedding table that is dense or quantised, and the reader every checkpoint is
loaded through. `session` picks which one from the config, and that same list
is what the server asks before it offers the GPU as a default, rather than
keeping a copy of the answer.

- `matvec_bt` becomes `Tensor::matmul`.
- The loop over attention heads becomes one batched matmul over a
  `[1, n_head, seq, head_dim]` tensor.
- `candle_nn::rotary_emb::rope` is the same `rotate_half` convention as
  `Rope::apply`, and `ops::rms_norm` the same scale-only normalisation.

Two differences are worth noticing because they run *against* the framework:

- **Batching is free.** The CPU engine needed a separate `forward_batch`,
  because a matrix-vector product and a matrix-matrix product are genuinely
  different kernels. On the GPU there is one `forward`, and a single token is
  just the narrow case.
- **The causal mask comes back.** With a KV cache the CPU engine got masking
  for free — the cache only ever held earlier positions. Here the whole batch
  is one matmul, so the future has to be masked out explicitly, with the
  triangular `-inf` matrix every tutorial shows.

## Numbers

| | CPU (q8) | Metal (bf16) | |
|---|---|---|---|
| Qwen2.5-0.5B, decode | 112 tok/s | 113 tok/s | 1.0x |
| SmolLM2-135M, decode | 194 tok/s | ~168 tok/s | 0.9x |
| Qwen2.5-0.5B, prefill (640 tokens) | 1.16 s | 0.19 s | **6.1x** |

**The GPU wins on prefill and, at this model size, not at all on decode.**
That is the same distinction as everywhere else in this repo. Prefill is
compute-bound, which is what a GPU is for. Decode is memory-bound, and there
the CPU is carrying int8 weights (556 MB) against the GPU's bf16 (1260 MB) —
quantisation claws back a whole hardware generation, and on the smaller model
the CPU is ahead.

That row used to read 35 tok/s and 3.3x, and the smaller model used to be
*slower* on the CPU than the larger one. Both were
[a scheduling bug](engine.md#the-floor-under-everything), not a property of the
hardware. It is the same lesson as the rest of the repo, learned the
expensive way: the number you are explaining may not be a number about the
thing you think it is about.

Metal `f32` runs at about two thirds of bf16's rate, for exactly double the
memory. `bf16` is the default because it is also what the checkpoints ship as.

The GPU backend covers **the Llama family, GPT-2, DeepSeek V2, Qwen3.5/3.8 and
Qwen3-Next** — five of the six architectures the CPU engine has. `deepseek_v3`
is the sixth: [`deepseek.rs`](../crates/gpu/src/deepseek.rs) implements it and
[`session`](../crates/gpu/src/model.rs) has no arm that routes a V3 checkpoint to
it, so asking for one on the GPU says which architectures there are rather
than failing obscurely. One list answers that, and the server asks it rather
than keeping a copy.

## Quantised weights on the GPU

```bash
kvad-gpu run --quant q8   # or q4, q4k, q6k
```

candle's `QTensor` holds GGML's block formats — the same scheme as
[`quant.rs`](../crates/llm/src/quant.rs): blocks with a shared scale, `Q8_0` and
`Q4_0` being 32-wide exactly like ours. `QMatMul` keeps HuggingFace's
`[out, in]` layout and transposes inside its kernel, where the dense path wants
the transpose done once at load, so [`Proj`](../crates/gpu/src/model.rs) hides the
difference.

All six backends, Qwen2.5-0.5B, decode:

| backend | weights | tok/s |
|---|---|---|
| cpu f32 | 1976 MB | 64 |
| cpu q8 | 556 MB | 112 |
| cpu q4 | 309 MB | 116 |
| metal bf16 | 1260 MB | 113 |
| **metal q8** | **525 MB** | **191** |
| metal q4 | 278 MB | 228 |

Quantisation is worth **1.7x** on both now — 113 → 191 on the GPU, 64 → 112 on
the CPU. It used to be worth nothing on the CPU, and the explanation given here
(a per-matmul cost that the bytes could not touch) was correct about the
mechanism and wrong about the conclusion: that cost was
[removable](engine.md#the-floor-under-everything), not inherent. With it gone, both
backends are bandwidth-bound in decode and behave the same way.

And then prefill, where it stops being free — eventually:

| prompt | metal bf16 | metal q8 | |
|---|---|---|---|
| 140 tokens | 0.070 s | **0.040 s** | q8 1.75x faster |
| 640 tokens | 0.180 s | **0.160 s** | q8 1.12x faster |
| 1340 tokens | 0.390 s | 0.380 s | even |
| 5040 tokens | **2.590 s** | 2.750 s | q8 1.06x *slower* |

There is a crossover, at around 1300 tokens on this machine. A short prompt is
still narrow enough that reading the weights dominates, so smaller weights win;
a long one turns every matmul into real work over a wide batch, and then the
dequantisation is pure overhead. The bottleneck moves with the batch size, and
the point where it crosses is a property of this GPU rather than of
quantisation.

(The CPU engine once claimed a similar crossover against *matrix* size, and
that one turned out not to exist —
[it was measuring a scheduler](engine.md#one-kernel-after-a-threshold-that-measured-the-wrong-thing).
This one is measured across four prompt lengths with five runs each, which is
the standard the other claim failed.)

> **Correction.** An earlier version of this section claimed a flat "1.6x
> slower on prefill" from a single pair of measurements at 654 tokens (0.24 s
> against 0.39 s). That does not reproduce: five runs at each of four prompt
> lengths give the table above, with a median equal to the minimum every time.
> Re-running the old arrangement behind `KVAD_GPU_DENSE_EMBED=1` rules out the
> embedding change as the cause, so the original figure was simply a bad
> measurement on a loaded machine. The *shape* of the claim survives — there is
> a length past which quantising costs you — but it arrives much later and much
> more gently than reported. Measure the curve, not two points on it.

One practical note. The k-quants (`q4k`, `q6k`) need dimensions divisible by
256 — Qwen is 896 wide, so they are refused up front with a message naming
`q8` and `q4` instead of failing halfway through the load.

## The table that was stored twice

```bash
kvad-gpu run --quant q8                          # quantised table
KVAD_GPU_DENSE_EMBED=1 kvad-gpu run --quant q8   # the dense one, for comparison
```

The embedding table was the last dense tensor in a quantised GPU model, and on
a small model it is not a small one: 136M of Qwen2.5-0.5B's 494M parameters live
in `[151936, 896]`. At bf16 that is 272 MB.

It stayed dense because the lookup needs an operation nothing else in the engine
wants — *gather rows and dequantise only those*. A `QTensor` is a run of blocks,
not a matrix, so row `t` is a span of blocks that has to be decoded on its own.
candle turns out to ship exactly that kernel (`QTensor::embedding`, which is
GGML's `get_rows`), so the change is one method call rather than a Metal shader.

**The compression is the smaller half of it.** Qwen ties its embeddings — small
models nearly always do — so the lookup table and the output head are the *same
matrix*. They were stored twice anyway, because a dense `index_select` wants
`[vocab, n_embd]` and a matmul wants the transpose. Quantise the lookup and both
want the identical thing: one `Arc<QTensor>`, two uses.

| | before | after |
|---|---|---|
| metal q8 | 797 MB | **525 MB** |
| metal q4 | 550 MB | **278 MB** |

525 MB for 494M parameters is 8.5 bits each, which is exactly `Q8_0` — 8-bit
codes plus an f16 scale per 32. The model is now at the format's floor, with
nothing left dense but the norms, and the win over bf16 goes from 1.6x to
**2.4x** (4.5x at q4).

**And it changed the speed by nothing at all:** 196.6 → 195.7 tok/s at q8,
239.6 → 237.3 at q4, prefill identical to three decimals. Exactly as it should
be. One row of 151936 is read per token, so the table was never on the
bandwidth path — it was pure capacity. Everywhere else in this repo,
quantisation traded accuracy for speed. Here it trades accuracy for *room*, and
that is the whole point: memory is what stops you loading a bigger model, and a
bigger model is worth far more than 1% of tok/s.

The CPU engine needed no change for any of this. `Weight::row` has dequantised
one row at a time since quantisation was added, and the tied head has always
been the same `Weight` — the hand-written version got here by the path of least
resistance, because one type did both jobs. The framework version duplicated
precisely because its two operations wanted two different types.

## The quantising, done once over there too

The CPU engine got a cache of pre-quantised weights
[some way back](engine.md#doing-the-quantising-once), and this backend did not, because
candle quantises Qwen2.5-0.5B in about 0.2 s and 0.2 s is not a problem worth a
file format. That judgement was correct and it did not scale: the same code on
DeepSeek-V2-Lite spends **66 seconds** turning 15.7 billion bf16 weights into
4-bit blocks, every single load, and throws the result away on exit.

So [`crates/gpu/src/qcache.rs`](../crates/gpu/src/qcache.rs) writes them out.
Steady state, three runs each way, both orders:

| model | quantise | map | file | |
|---|---|---|---|---|
| GPT-2-medium q8 | 1.7 s | 0.8 s | 358 MB | 2.1x |
| Qwen2.5-0.5B q8 | 1.2 s | 0.9 s | 500 MB | 1.3x |
| Qwen3-0.6B q8 | 1.3 s | 0.9 s | 604 MB | 1.4x |
| DeepSeek-coder-7B q8 | 6.6 s | 2.8 s | 6.8 GB | 2.4x |
| **DeepSeek-V2-Lite q4** | **65.5 s** | **3.9 s** | 8.2 GB | **17x** |

Those are whole-process times and the small ones are mostly floor — about
0.7 s of it is the Hub client resolving five file paths, which the CPU section
already noticed is the thing left standing once the weights stop costing
anything. The row that matters is the last one.

**Same box, different contents.** The file is the same `.nq` container: magic,
data section, a JSON header at the end, a `u64` saying where the header starts.
What goes in it is not the same bytes. Both engines call their eight-bit format
`q8` and they disagree about what that means — ours carries an `f32` scale per
32 weights in the order `quant.rs` reads, GGML's carries an `f16` — so handing
one to the other would be a model made of noise. They are told apart by name
before anything else: `Qwen--Qwen3-0.6B.q8.nq` and
`Qwen--Qwen3-0.6B.gpu-q8.nq`, in one directory, both listed by `kvad cache` and
both forgotten by `kvad cache <repo>`. `Container` and `Writer` moved out of
`qcache.rs` to be shared, which is the whole of the code reuse: a second copy of
the trailer arithmetic is a second place for it to be wrong.

**The part that got simpler.** The CPU cache has to stamp the checkpoint's
entire tensor list into every file, because a mapped cache never reopens a
checkpoint — so a build that learns to read one more *optional* weight gets
`None` back and runs without it, at full speed, saying nothing. That was
[the last route by which an unread weight could hide](engine.md#doing-the-quantising-once).

This one has no such problem, and the reason is that it caches *less*. Only the
quantised matrices go in the file. The norms, the biases and DeepSeek's two
dense halves of `kv_b_proj` still come from the checkpoint, which therefore is
still open, which means a name the file does not hold is simply quantised from
the source on the spot. A miss is answered rather than survived. The file is
then deleted at the end of that load, so the next one writes a complete one, and
the unread-tensor guard goes on subtracting from the *checkpoint's* list exactly
as it did before — a cache hit records the name it answered, so a weight served
from disk still counts as read.

That also means there is nothing to cache at `--quant none`: the checkpoint
already is the weights, and a copy of them would be a copy of the checkpoint.
The same rule the CPU cache applies at f32.

**Where the win is, and is not.** It is in the arithmetic, not the I/O. Run the
two arrangements alternately on a machine whose page cache cannot hold both the
29 GB checkpoint and the 8.2 GB cache file, and V2-Lite still wins (66 s to
17 s) while DeepSeek-coder-7B comes out even: the smaller file is being read
cold either way, and 6.9B parameters is only a few seconds of quantising to
save. Load the same model twice in a row — which is what a cache is for — and
the table above is what you get.

## One trait, two backends

Adding the GPU needed a change to the CPU engine's shape.
[`Transformer`](../crates/llm/src/model/mod.rs) deliberately keeps the KV cache
*outside* the model, because on the CPU it is a `Vec<f32>` the caller can own.
A GPU backend cannot work that way — its cache lives in device memory and must
never round-trip through the host between layers.

So the cache moved inside, behind a `Session` trait: `forward`, `cached`,
`truncate`. Everything above it — prefix reuse, sampling, streaming, the chat
loop, the TUI — stopped caring which backend it was talking to. The CLI keeps
its `--quant`, the GPU CLI gets `--device` and `--dtype`, and in the TUI `p`
cycles through all six: `cpu f32`, `cpu q8`, `cpu q4`, `gpu bf16`, `gpu q8`,
`gpu q4`.

That is the usual shape of this kind of work: the interesting part was not the
new backend, it was the seam the old one had to grow.

## The seam that cannot rewind

`truncate` returns a number now, and the number is the interesting part.

Every architecture released since Qwen3 — Qwen3.8, Qwen3-Coder-Next,
GLM-5.3-Flash, Kimi-Linear — carries a **recurrent state** on three layers in
four, with ordinary attention on the fourth. A state is not a cache. Keys and
values are stored *per position*, so the state of a prefix is exactly the rows
belonging to that prefix and dropping the rest is a `Vec::truncate`. A
recurrent state is one fixed-size vector that has already absorbed every token
it has seen, and **there is no subtraction that takes the unwanted ones back
out**.

Prefix reuse is built on being able to take them back out. So this engine's
answer is to refuse: a cache with a recurrent state rewinds to zero or not at
all. Every turn of such a conversation is a fresh prefill — honest and slow,
rather than fast and wrong.

Which is why `truncate` reports where it *actually* landed instead of returning
`()`. The caller used to ask for a rewind to N, assume it happened, and forward
from N. On a state that refused, that would run the new tail over a state that
had already read it: a model that is wrong rather than slow, and wrong in a way
nothing prints. Now the caller forwards from the position it is given and
reports that as `cached_tokens`, so the refusal shows up as a number on the
dashboard rather than as a model quietly talking nonsense.

The alternative is to snapshot the state every N tokens and rewind to the
nearest one, at a memory cost proportional to context / N. That is a real
design with a real tuning knob, and nobody can choose N without a model to
measure. Returning a number is what makes it a change to one cache later
instead of a change to everything that calls it.

Counting what the cache holds turned up a sevenfold error in what the server
was reporting. `kv_bytes_per_token` read `kv_dim()` — what ordinary
attention *would* store — where it should have read what the cache actually
stores. Those agree for GPT-2 and Llama and disagree for DeepSeek, which sets
`n_kv_head = n_head` because every head really does have its own key, and then
keeps none of them: 576 floats a position against the 2048 `kv_dim()`
describes. The server was reporting **7.1x** the memory it was using, on the
one architecture whose entire argument is that it uses less.

## Pictures, which are not tokens

```bash
cargo run --release -p kvad-gpu --example sdxl -- \
    --prompt "a lighthouse on a cliff at dusk, oil painting" --out lighthouse.png
kvad images make "a red fox in fresh snow" --model stabilityai/stable-diffusion-xl-base-1.0
```

[`image/`](../crates/gpu/src/image/mod.rs) is text-to-image, and it is the one
part of the engine with no CPU version. That is a decision, written down with
every shape it depends on in [`docs/image-plan.md`](image-plan.md): an
image model is a convolution stack, `tensor.rs` has no convolution, and a fast
one is a project of its own. What is still written here is the model. CLIP,
the UNet, the VAE, both schedulers, Qwen-Image's MMDiT and its video VAE,
FLUX's T5 and single-stream blocks, and the PNG encoder are all in this
repository; candle supplies `conv2d` and the
matmuls, and nothing from `candle-transformers` is used.

Nothing about it is a token loop. A text encoder runs once. A denoiser runs
20–50 times from noise, twice per step while guidance is on, and a scheduler
with no weights at all turns each prediction into the next latent. A VAE turns
the last latent into pixels. So a painter is not a `Session`: it implements
[`kvad::image::Painter`](../crates/llm/src/image.rs), a peer of `Llm` that the
engine thread, the scheduler and the server hold beside it.

The first image out of `examples/sdxl.rs` was the right one: a lighthouse,
1024², 30 steps. What that took on an M5 Pro, in f16, from single runs (the
second with a 58 GB download going on in the background):

| | 1024², 30 steps | 768², 20 steps |
|---|---|---|
| denoise, per step (guidance on) | 3.93 s | 2.2 s |
| VAE decode | 9.4 s | 5.3 s |
| load, from the disk | 19 s | — |

The decode is the most memory the pipeline ever asks for: it is the only
stage at full resolution, 128 channels of 1024×1024.

Qwen-Image is the same story at twenty billion parameters: Qwen2.5-VL-7B as the
text encoder, a 60-block MMDiT with a three-axis RoPE as the denoiser, a video
VAE run on one frame, flow matching instead of noise prediction. 41 GB of bf16
denoiser does not fit beside anything on a 48 GB machine, so it runs at q8
through the same loader and quantised-weight cache as the text models: 21.7 GB
of transformer and 7.5 GB of language tower, quantised once (130 s with the
quantising, 66 s from the cache after). Again single runs, true CFG at 4:

| | 512², 20 steps | 1024², 20 steps |
|---|---|---|
| denoise, per step (two 20B passes) | 6.5 s | 28.3 s |
| VAE decode | 1.5 s | 5.8 s |
| peak memory footprint | 32.7 GB, quantising as it loaded | — |

Its first image was structured noise, and the model was not the reason. The
language tower checked out at once — it is a chat model, so the checkpoint's
own output head could be pointed at it, and "The capital of France is" came
back " Paris". The denoiser was probed instead: from pure noise at σ = 1, the
clean latent it predicts should be smooth, and its neighbouring pixels
correlated at 0.65. candle's Metal kernel for a quantised matrix-matrix product
reads its input from the start of the buffer whatever the tensor's offset — and
the image half of joint attention is a `narrow` that starts where the text
ends. So every patch was projected from the wrong rows. `Tensor::copy` keeps the
offset too; `force_contiguous` does not. With that in `Proj::forward` the same
probe read 0.996, and the next image was a fox.

FLUX.1-schnell came after, and needed less new code than either: its first
nineteen blocks *are* Qwen-Image's blocks under other names, so the block moved
into [`mmdit.rs`](../crates/gpu/src/image/mmdit.rs) and both models load it. What
FLUX adds is a T5-XXL encoder ([`t5.rs`](../crates/gpu/src/image/t5.rs): relative
position buckets instead of positions, no `1/√d` on the scores) and 38
single-stream blocks that run attention and MLP side by side. Its first image
was right. At q8 it holds 18.3 GB (12.6 GB of transformer, 5.1 GB of T5), and
schnell's four steps need no guidance, so each is one forward pass:

| | 768², 4 steps | 1024², 4 steps |
|---|---|---|
| denoise, per step | 7.3 s | 13.1 s |
| VAE decode | 5.3 s | 9.6 s |

(Single runs again.) FLUX.1-dev is the same transformer with the guidance
scale as one more input, embedded as the timestep is, so it is guided in
one pass a step and takes no negative prompt: 28 steps at guidance 3.5,
3.7 s a step at 768² and 7.2 s at 1024²
([docs/image-plan.md](image-plan.md)).

The community's GGUFs of Qwen-Image's and FLUX.1-schnell's transformers load
too, and of LTX-2.5's distilled DiT for its fast pipeline, by the repo and
the quantisation together; the model card's `base_model` supplies the rest
([docs/gguf-plan.md](gguf-plan.md)):

```bash
kvad pull city96/Qwen-Image-gguf:Q4_K_S
kvad images make "a red fox in fresh snow" --model city96/Qwen-Image-gguf:Q4_K_S
```

city96's FLUX files keep Black Forest Labs' layout, one `qkv` matrix where
diffusers has three, and are read through a map of rows, so no block is
requantised. city96's Q8_0 of either model is Kvad's own q8 block for block.
The k-quants (Q4_K, Q5_K, Q6_K) run on the M5's matrix units in the same
kernel as Kvad's q8, one decoder a format, at 75–90% of its rate. So a
Q4_K_S holds 9.1 GB less for Qwen-Image and 5.5 GB less for FLUX at 1024²
for a step 5–18% longer than q8's, and an LTX-2.5 Q4_K_M holds about 4 GB
less and denoises as fast.

SDXL fine-tunes load too: the diffusers folders nearly all of them ship,
whichever names their files carry, and a checkpoint in one file in
Stability's own layout, as Pony and Civitai's are. A repo whose only model
is one file is named by the repo; one with several, `repo:file.safetensors`;
a file on this machine, by its path. Its configs come from SDXL's base, and
its VAE is the fp16-fix ([docs/checkpoint-plan.md](checkpoint-plan.md)):

```bash
kvad pull LyliaEngine/Pony_Diffusion_V6_XL
kvad images make "score_9, a red fox in fresh snow" --model LyliaEngine/Pony_Diffusion_V6_XL
kvad images make "a lighthouse at dusk" --model ~/Downloads/some-sdxl-finetune.safetensors
```

Stable Diffusion 1.5 is SDXL's parts at a smaller size: one text encoder,
read to its last layer; a UNet of four levels with no size conditioning,
whose transformers project with 1×1 convolutions; and its own VAE, which,
unlike SDXL's, stays in range in f16. Each part matches diffusers at
112–121 dB in f32 ([docs/sd15-plan.md](sd15-plan.md)). It is here for
its fine-tunes, which load by the same three kinds of name as SDXL's, and
from one file its VAE is the file's own, since a fine-tune's often is: all
248 of DreamShaper 8's VAE tensors differ from the base's. At 512² and 25
steps it takes 0.75 s a step on an M5 Pro, and the server charges 1.9 GB.

```bash
kvad images make "a red fox in fresh snow" --model stable-diffusion-v1-5/stable-diffusion-v1-5
kvad pull ckpt/anything-v5.0
kvad images make "1girl, reading in a library" --model ckpt/anything-v5.0
```

SDXL and SD 1.5 also make an image from a picture
(`crates/gpu/src/image/edit.rs`, `/v1/images/edits`, #43). The picture is
encoded by the VAE's other half, noised to the level a `strength` names, and
the walk down the schedule goes on from there with the prompt: SDEdit, which
diffusers calls image-to-image. The prompt says what the picture should be,
not what to change in it; these are not models trained to follow an
instruction. With a mask it is inpainting as diffusers does it for a UNet
that was not trained to: the kept part is put back after every step, noised
to the level the step came down to, and once more as pixels at the end, so
it comes back to the byte. On SD 1.5 at 512², the model's own drawing edited
with another prompt comes back 10 levels of 255 from itself, on average, at
strength 0.2, and 41 at strength 1; under a mask of its right half, the left
half is the same bytes
(`edit::tests::sd15_edits_a_picture_and_keeps_what_a_mask_keeps`). A mask's
edge can show: nothing here was trained to meet one.

```bash
kvad images edit "a green apple on a wooden table" --image apples.jpg --strength 0.5
kvad images edit "a blue ceramic bowl" --image apples.jpg --mask right-half.png
```

LoRAs apply to Qwen-Image, FLUX, SDXL, SD 1.5 and LTX-2.5, at run time, per
request: the model stays as it was loaded, q8 or a GGUF, and each request
chooses its own LoRAs and strengths ([docs/lora-plan.md](lora-plan.md)).
Each is a side path `(x·A)·B` beside a layer it adapts, added into the
layer's answer by the M5 dense kernel's store, for 4.5% a step on Qwen-Image.
They are read in the spellings people publish: PEFT's, kohya's under `ldm`'s
names, diffusers' or Black Forest Labs' (whose fused `qkv` is split by rows),
diffusers' two older ones, with their text encoders' and their convolutions'
pairs. Against diffusers with PEFT, Qwen-Image's first two blocks with
Lightning agree to 95.8 dB in f32, three FLUX LoRAs to 111 dB, and seven SDXL
and SD 1.5 LoRAs of every one of those kinds to 108–121 dB. On LTX-2.5 a
video's LoRAs go on every DiT its pipeline loads, and in bf16 they are level
with the reference's own run of its distilled LoRA. Lightning's 8 steps take
68 s at 1024², where the base's 50 took 448 s, and draw a sharper picture:

```bash
kvad pull lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
kvad images make "a tiny astronaut hatching from an egg on the moon" --model Qwen/Qwen-Image \
    --steps 8 --lora lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
```

A LoRA for SDXL can be trained here too, on a folder of pictures with a
caption beside each ([docs/tune.md](tune.md)): 1.9 s a step in 7 GB at 512²,
7.3 s in 14 GB at 1024², keeping the step with the lowest loss on pictures
it did not train on.

```bash
kvad-gpu tune --data ./my-photos --name my-style
```

Two things in it are measurements rather than code:

- **The previews.** Each step can carry a picture of where it is heading,
  without a decode: the latent's four channels mixed into RGB by a fixed
  4×3 matrix. The matrix was fitted by least squares from one generated
  image, latent against pixels, and explains 76–83% of the variance per
  colour — blurry and slightly wrong, which is what a preview is for.
- **The schedule.** SDXL's Euler schedule is checked against the timesteps
  diffusers picks (`958, 925, …, 34, 1` for 30 steps), because a scheduler
  that visits the wrong noise levels still draws a picture, only a worse one.
