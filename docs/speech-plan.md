# Plan: speech to text

Written 2026-10-16 for [#11](https://github.com/bisand/kvad/issues/11), with
the first half of it built: the engine and `kvad transcribe`. Every shape
below was read off a `config.json` or a safetensors header on this machine,
and every rule of the decoding loop off openai-whisper's `decoding.py` and
`transcribe.py`. Every stage was then checked against transformers running the
same checkpoint on the same audio.

The first checkpoints it was built for are the National Library of Norway's
[NB-Whisper](https://huggingface.co/NbAiLab/nb-whisper-large), fine-tunes of
OpenAI's Whisper on 66,000 hours of Norwegian. They are Whispers in every
respect that matters here, so what loads them loads `openai/whisper-*` too.

## Where it lives, and why there

On the CPU, written out, in `crates/llm/src/speech/`, beside the text path
and not inside it. That puts it on the other side of
[the image plan](image-plan.md#the-decision-gpu-first-and-no-cpu-path-for-now)'s
decision, for the reason that decision gave: there it was convolutions all the
way down, and there was no CPU path to write. Whisper is matmuls, LayerNorm
and GELU. Its two convolutions are one-dimensional, and a one-dimensional
convolution is a matmul over stacked windows. Every kernel it needs was
already here: `Weight::matmul_bt` for the encoder's 1500-frame batches, the
tiled `gemm_bt` for its attention, and q8/q4 for its weights.

It is not a `Transformer` in the architecture registry. A chat session keeps
a conversation's cache between turns, and a transcriber keeps nothing but its
weights: a recording goes in and a transcript comes out. So
`speech::Transcriber` is a peer of `Llm`, the way `Painter` is.

## The architecture, as read off the checkpoints

`WhisperForConditionalGeneration`, `model_type: whisper`.

| | tiny | base | small | medium | large (v3) |
|---|---|---|---|---|---|
| `d_model` | 384 | 512 | 768 | 1024 | 1280 |
| encoder / decoder blocks | 4 / 4 | 6 / 6 | 12 / 12 | 24 / 24 | 32 / 32 |
| heads (both) | 6 | 8 | 12 | 16 | 20 |
| `num_mel_bins` | 80 | 80 | 80 | 80 | 128 |
| `vocab_size` | 51865 | 51865 | 51865 | 51865 | 51866 |
| weights | 1 file | 1 file | 1 file | 1 file | 2 shards |

All of them have 1500 audio positions (`max_source_positions`), 448 text
positions (`max_target_positions`) and a 4× MLP. Every tensor is f32 in the
NB-Whisper repos.

**Why not `Spec`.** Its single `n_ctx` would mean two things (1500 frames,
448 tokens), and its single `n_layer` would be wrong for the distilled
Whispers, which keep 32 encoder blocks and drop to 2 decoder ones. The mel
bands have no home in it at all. So `whisper::Config` is a sibling: eleven
numbers, read by name.

**What the tensors say that the config does not:**

- `encoder.conv1.weight` is `[d, n_mels, 3]` and `conv2.weight` is
  `[d, d, 3]`, stride 1 and 2, padding 1. These are the only rank-3 tensors
  in the file. `Checkpoint` now reads any `[out, in, k]` as the
  `[out, in × k]` matrix that im2col multiplies by. The depthwise
  `[c, 1, k]` that Qwen3-Next needed is the case `in = 1`.
- `encoder.embed_positions` is stored as a weight even though it is fixed
  sinusoids. The decoder's is learned, as GPT-2's is.
- Every `k_proj` lacks a bias, and every other projection has one.
- There is no `proj_out.weight`: the output head is `embed_tokens`.
- `activation_function: gelu` is the exact, `erf` one. GPT-2's tanh
  approximation differs by up to 5e-4, so `tensor::gelu_erf_inplace` is new.
  Rust has no stable `erf`, so it uses Abramowitz–Stegun 7.1.26, good to
  1.5e-7.

**The vocabulary moves by one at large-v3.** Large-v3 added Cantonese, which
shifts `<|translate|>`, `<|transcribe|>`, `<|notimestamps|>` and the first
timestamp up by one. So `transcribe::Tokens` looks every special token up in
the tokenizer by name, and none is a constant.

## The spectrogram

`preprocessor_config.json` gives `n_fft 400`, `hop_length 160`,
`chunk_length 30` and `feature_size` (80 or 128). transformers' extractor adds
the rest:

- periodic Hann window;
- reflect-padded centring;
- power spectrum;
- Slaney-scale, Slaney-normalised mel filters from 0 to 8 kHz;
- `log10` floored at 1e-10;
- everything more than 8 below the loudest value raised to that floor;
- `(x + 4) / 4`;
- the last frame dropped.

The floor is taken over the whole recording, so `mel::log_mel` runs once per
file, not once per window.

400 is not a power of two. `mel::fft` halves while the length is even
(400 → 200 → 100 → 50 → 25) and does the 25 directly. That is the only new
mathematics in the chapter, and a test holds it to the plain DFT.

## Verification

`scripts/whisper-fixtures.py` runs transformers in f32 on a WAV and writes
four things:

- its mel spectrogram;
- its encoder output;
- its first logits after the prompt;
- its greedy tokens, with and without timestamps.

`speech::tests::whisper_agrees_with_transformers` compares each stage of ours
against them. On the first 20 seconds of NB-Whisper's own sample
(`king.mp3`):

| | mel | encoder | logits | tokens, plain | tokens, timed |
|---|---|---|---|---|---|
| nb-whisper-tiny | 2.4e-5 | 2.2e-4 of 16.8 | 9.7e-5 of 29 | identical | identical |
| nb-whisper-small | 2.4e-5 | 7.3e-3 of 32.9 | 1.2e-4 of 67 | identical | identical |
| nb-whisper-large | 2.2e-5 | 1.2e-3 of 18.3 | 3.6e-5 of 32 | identical | identical |

The gaps are maximum absolute differences. "Of" gives the largest magnitude
in the reference.

**Long-form check.** Over the whole 85-second sample, greedy and with
timestamps, tiny's segments agree with transformers' long-form generation
segment for segment, times included, for the first two windows. They part
ways in the third, for the reason under *Known differences*.

## Known differences from the reference

- **The last window's padding.** openai-whisper, and transformers in long
  form, pad the last window's *spectrogram* with zeros. A zero there is
  `log10 = -4`, which is not what silence looks like. Here the *audio* is
  padded with silence before the spectrogram is taken, which is what a short
  clip looked like in training, and what transformers does for a clip under
  30 seconds. On `king.mp3` this changed tiny's last window from repetition
  into the right sentence once fallback ran.
- **No gzip test.** openai-whisper decodes a window again when its text
  compresses better than 2.4:1, a sign of repetition. There is no compressor
  in this crate. Instead, a window that runs into the 224-token limit without
  ending counts as failed, which is how the repetition loop shows up.
  Log-probability below -1 is checked as in the reference.
- **Greedy only.** No beam search. NB-Whisper's card recommends 5 beams for
  accuracy.
- **NB-Whisper and `--translate`.** The model ignores the translate task, and
  transformers gives the same Norwegian. NB-Whisper was trained to write
  English for `language=en`.
- **A lone closing timestamp skips the rest of the window.** This is
  openai-whisper's rule, followed exactly. On `king.mp3`, large ends its
  second window at 54.8 s this way and loses "Nordmenn liker Grieg, Kygo"
  before 60 s. transformers' chunked pipeline, which NB-Whisper's card uses,
  overlaps windows and does not lose it. It is a different algorithm.

## How fast, on the CPU

On an M5 Pro, `kvad transcribe` on the 85-second sample unless it says
otherwise, fallback included:

| model | precision | | wall time | × real time |
|---|---|---|---|---|
| nb-whisper-tiny | f32 | plain text | 1.6 s | 55× |
| nb-whisper-tiny | f32 | `--timestamps` | 7.1 s | 12×: windows that fail are decoded again, warmer |
| nb-whisper-small | f32 | first 20 s | 2.4 s | 8× |
| nb-whisper-large | f32 | `--timestamps` | 58 s | 1.5× |
| nb-whisper-large | q8 | `--timestamps` | 41 s | 2.1× |

The encoder runs at the speed of the f32 `gemm_bt`, about 200–250 GFLOP/s
here. Small's 12 blocks over one window take 1.7 s, which is that rate times
their 340 GFLOP. Nothing is waiting on anything else.

The work runs on its own rayon pool, a window at a time. Without the pool,
every parallel loop in a decoder step takes rayon's cold path, as the text
engine's did before 2026-09-19, and tiny ran at half its speed.

## What is next

1. **`/v1/audio/transcriptions` and `/v1/audio/translations`** in
   `kvad-serve`. This needs:
   - a `Kind` and the `Cmd`/`Evt`/`Work` arms for it;
   - detection by `model_type: whisper`, since `hub::pipeline` takes every
     non-pipeline for a chat model;
   - multipart upload through the hand-written `videos::form`;
   - the five response formats, which `Transcript::render` already writes;
   - `kvad transcribe` routed to a running server, as `run` and `chat` are.

   `speech::audio::from_bytes` is there for the upload.
2. **The GPU.** Large at 2× real time is usable for a file and not for a
   meeting. On Metal the encoder is one large prefill, which is what
   kvad-gpu does best.
3. **Beam search**, and the gzip test, if a measured word error rate says
   they are worth it.
