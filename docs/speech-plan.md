# Plan: speech to text

Written 2026-10-16 for [#11](https://github.com/bisand/kvad/issues/11), with
both halves of it built: the engine and `kvad transcribe`, then the server's
endpoints (*On the server*). Every shape
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

- **Held to UTF-8.** Whisper writes bytes. A character outside ASCII is
  two to four of them, which need not be in one token, and nothing in the
  model makes what it writes UTF-8. A hallucinating window can write `å`
  and then a lone continuation byte, or end a segment halfway through `ø`.
  The reference prints `�`. Here, the decoder rules out any token that
  cannot continue the bytes written so far, and rules out end-of-text and
  timestamps in the middle of a character (`transcribe::utf8`).
  - Text that was already UTF-8 comes out as it did, since nothing it chose
    is ruled out. On NB-Whisper tiny, small and large, the tokens are still
    transformers' own.
  - What `avg_logprob` reports moves a little, because it is taken after the
    rules, as in openai-whisper, and the rules now remove some impossible
    tokens' share.
  - On a 79-second `say -v Nora` recording, tiny's two `�` are gone. One
    window that had come out as noise is now the sentence that was read.
- **The last window's padding.** openai-whisper, and transformers in long
  form, pad the last window's *spectrogram* with zeros. A zero there is
  `log10 = -4`, which is not what silence looks like. Here the *audio* is
  padded with silence before the spectrogram is taken, which is what a short
  clip looked like in training, and what transformers does for a clip under
  30 seconds. On `king.mp3` this changed tiny's last window from repetition
  into the right sentence once fallback ran.
- **Repeated 4-grams for the gzip test.** openai-whisper decodes a window
  again when its text compresses better than 2.4:1, a sign of repetition.
  There is no compressor in this crate. Instead, a window fails when it runs
  into the 224-token limit and more than half of its text's 4-grams were
  written earlier in it, which is what a compressor would point back to.
  Log-probability below -1 is checked as in the reference.
  - The two are far apart. Speech measured 0.01 to 0.07, including NB-Whisper
    large on `king.mp3` played 2 to 3 times as fast, where a window fills
    the limit. The kitchen's page of spaces and stray letters measured 0.72,
    taken from its text tokenized again, as the audio was not kept. Tiny
    writing one special token over and over on pink noise measured 1.
  - Running into the limit alone used to count as failure. Speech that
    filled the limit was then tried at every temperature, and kept at 1.0.
  - Real radio does not come near the threshold. Three NRK podcast
    episodes, 2 h 17 min in all (Abels tårn with listeners phoning in,
    P3morgen, and the first hour of Lørdagsrådet with its music), large at
    gpu-q8, plain and timed: 275 windows each way, none decoded again, none
    at the limit, none skipped. Of the windows with 60 tokens or more, the
    highest score was 0.099, the show's introduction with its name said
    twice, and the next 0.075. (A window of a few words scores high, and
    ends, so it is never asked.)
  - Radio fills a window more than a speech does. With timestamps, a window
    held 147 to 158 tokens at the median and 212 at most, where `king.mp3`
    holds about 87. Speech that fills the limit is not rare in the way
    hallucination is.
- **A loop that never ends is skipped.** When a window loops at every
  temperature, openai-whisper writes down its last try anyway. Here nothing
  of it is written, as for silence, and the next window starts 30 seconds
  on. On 24 minutes of kitchen microphone, large at gpu-q8 had one such
  window, over noise: 224 tokens of spaces and stray letters at each of the
  six temperatures, at an average log-probability of -0.03 to -0.14,
  confident throughout. It was the only window skipped. Speech before the
  loop in such a window is lost with it.
  - On `king.mp3` at 2.6 times the speed, the first window fills the limit
    with speech. Skipping every window that never ended, as #246 first did,
    left only the last sentence. Now the window is kept at temperature 0, and
    the next starts at its last timestamp.
- **NB-Whisper never says `<|nospeech|>`.** On 30 seconds of pink noise, and
  of digital silence, its probability is 0.00 on large, where
  openai/whisper-tiny gives 0.97 and 0.93. So silence is never skipped as
  silence on NB-Whisper, and a loop is the only window that is.
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

## On the server

`POST /v1/audio/transcriptions` and `/v1/audio/translations`, in
`crates/serve/src/speech.rs`. OpenAI's form: `file`, `model`, `language`,
`prompt`, `response_format` and `temperature`, and beside them `seed`,
`timestamps` and `stream`. `kvad transcribe` sends to them when a server is
running, as `run` and `chat` do, and `--local` keeps it in the process.

- **A fourth kind of model.** `hub::pipeline` sorts a model on the disk by
  what it is for, and a Whisper has no `model_index.json` to say. So its
  `model_type` names it `WhisperForConditionalGeneration`
  (`speech::PIPELINE`), and the server's `Kind::Speech` follows from that as
  `Kind::Video` follows from LTX's files: `/v1/models` and `/api/models` say
  `speech`, `resident_for` refuses it a chat and a chat model a recording,
  and the engine holds a `Held::Speech` beside the other three.
- **The GPU at q8 unless asked**, since *On the GPU* below; the CPU at f32
  on a machine without one. It was `cpu-f32`, what `kvad transcribe --local`
  loads, so that the two wrote the same transcript, and a GPU backend was
  refused. On large that agreement cost 11 to 17 times the time. Asked for
  `cpu-f32` (`--backend cpu-f32`, or `--quant f32`), the server and
  `--local` still write the same bytes.
- **Admission from the config.** What a Whisper's weights will take is
  counted from `config.json` (`whisper::Config::params`) at the precision
  asked for, before anything is loaded. On nb-whisper-tiny at f32 and
  nb-whisper-small at q8 it came to the byte what the loaded model reports.
  On the GPU it is `kvad_gpu::speech::weight_bytes`, held to the byte by
  `the_estimate_is_what_loads` at f16 and q8.
- **The same bytes.** On `king.mp3` with nb-whisper-tiny, all five formats
  from `curl`, from `kvad transcribe` routed to the server and from
  `kvad transcribe --local` are identical files, and so is small at q8.
  The client writes the server's answer as it came: JSON parsed and written
  out again was not, because serde_json's parser can come back a digit short
  on an `avg_logprob`.
- **`temperature` is OpenAI's.** 0, or none, is greedy with the fallback to
  warmer temperatures; above 0, the one temperature tried. `kvad transcribe
  --local` reads it the same way, so the same command line is the same
  transcript in either process. Greedy with no fallback at all is the
  engine's `Request { temperature: Some(0.0) }`, which the reference test
  uses and no command line asks for.
- **A hang-up stops it** after the window under way, as a video stops after
  its step: on a 340-second recording with small at q8, 11 seconds after the
  client left.
- **Nothing is kept**, neither the recording nor the transcript.

## On the GPU

`crates/gpu/src/speech.rs`, in candle on Metal: the same model, block for
block, and none of the rest. `Transcriber` drives a `speech::Model`:

- `hear`: a window's spectrogram to what the decoder reads of it;
- `said`: an empty text cache;
- `decode`: tokens to the next-token scores at the rows asked for.

`whisper::Cpu` is one Model, and kvad-gpu's `GpuWhisper` the other. The
spectrogram, the tokenizer, the rules and the walk through a recording are
host code, and they are here once. The two caches cross the boundary as
`Heard` and `Said`, which only the model that made them opens. `decode`
answers in scores rather than hidden states, because a GPU's hidden state
would otherwise come to the host only to go back for the output head: a
second wait a token.

- **The encoder** is one 1500-row batch with no mask. Its attention is
  MLX's fused kernel, and at q8 its projections run on the M5's matrix
  units, as an image model's do. A language model cannot use those, because
  they have no one-row kernel, and nothing in the encoder is one row. Nor
  are the cross-attention's key and value projections, which read the
  encoder's output once a window.
- **The decoder** keeps its own key and value cache (kvad-gpu's
  `KvCache`) and reads the window's cross keys and values unchanged.
  - A prompt of more than one token is the one case where the fused
    kernel's causal mask is wrong (`model::fused` says why). So a masked
    prompt is written out, with its scores and softmax in f32. Single steps
    and the cross-attention are fused.
  - Around q8 both caches are kept in f16, as a quantised language model's
    are. The cross keys and values are read whole at every step: at large,
    in f32, that was 490 MB a token beside 850 MB of weights. f16 took a
    step from 7.6 to 6.7 ms and changed no token.
- **The residual stream is f32** whatever the matrices are, and so are the
  norms and the two position tables, read from the checkpoint unrounded.
  This is not where two bytes lose their accuracy: with the stream in two
  bytes, or the attention in f32, the gaps below barely move.
- **Dense means f16, not bf16** (`kvad_gpu::speech::DENSE`). A server's
  `gpu-bf16` loads a Whisper in f16 and says `metal f16`, as SDXL runs in
  f16 whatever it is asked. Whisper was trained in f16 and its numbers stay
  well inside f16's range, so bf16's extra range buys nothing. Its three
  fewer mantissa bits put it four to thirteen times further from
  transformers at every stage below, except tiny's encoder, where it is
  twice.

### Against transformers

kvad-gpu's `speech::tests::whisper_agrees_with_transformers` is the CPU's
test at each precision the GPU loads at (`KVAD_WHISPER_GPU=f32,bf16,f16,q8`),
on the same fixtures. Both tests now go through `speech::reference`, which
holds the stages and the tokens. Each cell is the largest gap, then whether
the greedy tokens are transformers' own, plain and timed:

| | precision | encoder | logits | plain | timed |
|---|---|---|---|---|---|
| nb-whisper-tiny | f32 | 2.4e-4 | 3.6e-5 | same | same |
| | bf16 | 0.29 | 0.29 | same | same |
| | f16 | 0.16 | 0.048 | from token 38 | same |
| | q8 | 1.4 | 0.66 | same | same |
| | q4 | 10 | 13 | (not decoded: the scores are noise) | |
| nb-whisper-small | f32 | 4.8e-3 | 1.0e-4 | same | same |
| | bf16 | 4.4 | 0.26 | same | from token 32 |
| | f16 | 1.1 | 0.024 | same | from token 32 |
| | q8 | 3.1 | 0.21 | from token 20 | from token 22 |
| | q4 | 22 | 4.1 | from token 20 | from token 21 |
| nb-whisper-large | f32 | 9.9e-4 | 3.1e-5 | same | same |
| | bf16 | 6.3 | 0.084 | from token 36 | same |
| | f16 | 0.50 | 0.013 | same | same |
| | q8 | 7.1 | 0.093 | same | same |
| | q4 | 13 | 1.3 | same | same |

The reference's largest magnitudes are 16.8, 32.9 and 18.3 for the
encoder, and 29, 67 and 32 for the logits. On the CPU at q8
(`KVAD_WHISPER_PRECISION=q8`) the same test gives 3.6, 5.5 and 9.9 for the
encoder and 0.66, 0.25 and 0.091 for the logits. Its tokens are the GPU's
q8 tokens in every cell: small's plain from token 20 and timed from token
22, and the rest the same.

- **f32 on the GPU is transformers'**, token for token, on all three, and
  as close at every stage as the CPU. That is the check that the model is
  right.
- **What differs at two bytes or at q8 is a near tie**, and the CPU's own
  q8 lands on the same ones. On small, q8 drops the comma in "sørlendinger,
  og" at token 20, and the CPU at q8 drops the same comma at the same
  token. f16 and bf16 put one timestamp a step (20 ms) late. tiny at f16
  writes "innvandrert" for "innvandrere", and large at bf16 writes
  "innvandret".
- **q4** is unusable on tiny, where its encoder is 10 off of 17, and on
  small it shares q8's comma. On large it matches transformers token for
  token. It is offered as the server offers it for everything, and not
  chosen for anyone.

### How fast

nb-whisper-large on all 85 seconds of `king.mp3`, `--timestamps`, fallback
included, on an M5 Pro. Every row's transcript is the same text. Runs come
from `cargo run --release -p kvad-gpu --example transcribe_speed`, each set
of backends in its own process, interleaved in rounds; the range is over
the rounds:

| backend | wall time | × real time | a window heard | a decoder step |
|---|---|---|---|---|
| cpu f32 | 48.1–51.5 s | 1.7–1.8× | 11.1–12.3 s | 45 ms |
| cpu q8 | 33.1–33.5 s | 2.5–2.6× | 8.0 s | 27–29 ms |
| metal f32 | 6.5–6.7 s | 12.7–13.3× | 0.58–0.60 s | 17–18 ms |
| metal f16 (`gpu-bf16`) | 4.23 s | 20.2× | 0.27–0.28 s | 12.2–12.3 ms |
| **metal q8** (`gpu-q8`) | **2.91–2.92 s** | **29×** | 0.30 s | 6.7 ms |

gpu-q8 is 11× cpu-q8 and 17× cpu-f32. "A window heard" is the encoder and
every block's cross keys and values; three windows were heard, and 261
one-token steps taken, in every row. bf16 runs at f16's speed (4.4–4.8 s in
the first, crowded attempt below). The CPU's numbers here are a little under
the table above, which was `kvad transcribe` on another day.

- **A window's encoder is 0.3 s on the GPU.** The two convolutions, 32
  blocks over 1500 frames and every block's cross keys and values take
  270–300 ms at f16 or q8, and about 590 at f32. On the CPU the encoder is
  most of a window.
- **The decoder is now most of the time**: 261 one-token steps, at 6.7 ms
  each at q8, 12.3 at f16 and 17 at f32, 60 to 80% of a run. That is above
  what the bytes cost. At q8 a step reads about 850 MB of weights and 250 MB
  of cross keys and values, about 4 ms at 280 GB/s. The likeliest place for
  the rest is the hundreds of small kernels in 32 blocks of LayerNorm,
  residual adds and GELU, unfused. The text models' fused kernels
  (`fused.rs`) are RMSNorm and SiLU, so they do not apply. That is a guess
  until it is ablated, and it is the next thing to take apart.
- **The GPU does not go cold between steps.** The host's share of a step is
  the rules and the sampler, about 0.3 ms. Idling the GPU a further 1 ms
  before every step made a step 0.15 ms slower, and 3 ms made it 0.5 ms
  slower. The keeper a split model needs (`model::stage`'s `Warm`) would buy
  well under 1% here.
- **One process holding six of them is not a measurement.** The first
  attempt loaded all six backends, 24.6 GB in one process. Its third round
  of q8 took 12.5 s where the first took 3.8, and the CPU's f32 went from
  48.7 to 61.9 s. Split into a process for the GPU and one for the CPU, no
  row drifts by more than a few percent.

## Live, from the phone

Planned 2026-10-17, not built. The owner's ask: record on the phone, and see
the words come back on the phone *while speaking*, over a return channel,
not as one transcript once the recording is done. Today's `stream: true`
cannot do that. The whole file is uploaded first, and then a delta comes
back for each 30-second window. Getting there takes audio streaming in,
text streaming out, and a decoding loop that writes words before their
window is complete. The phone reaches the Mac through the tunnel in
`docs/relay-plan.md`. This section is what runs at either end of it.

**One session, two directions.** The phone opens one WebSocket to the
server. Audio goes up it as it is recorded; events come down it as they are
decoded. Through the tunnel it is one QUIC stream whose two directions are
independent, so a slow upload never holds back the return channel, and the
reverse. Over the LAN, a direct path and the relay alike, it is the same
session, and a change of path mid-session is QUIC's business, not this
one's.

**The shape: OpenAI's Realtime API, transcription only.** The house rule is
OpenAI's shapes, so `GET /v1/realtime?intent=transcription` (upgraded to a
WebSocket), a session update choosing the model and language, audio
appended in chunks, and transcript deltas and completions coming back. The
exact event names are taken from OpenAI's reference when this is built, and
tested against their SDK, as the other endpoints were. Two additions,
marked as Kvad's own:

- *Binary audio frames.* Realtime sends audio as base64 inside JSON, a third
  more bytes for nothing on a phone's uplink. A binary frame is taken as raw
  audio.
- *A replaceable partial.* Beside the append-only deltas, an event carrying
  the current guess at the words not yet settled. Each one replaces the
  last.

**Audio.** 16 kHz mono PCM, 16-bit, little-endian: what Whisper eats, so the
server resamples nothing. That is 256 kbit/s, which a phone's uplink and the
relay both carry. Opus would cut it to a tenth, but would mean an Opus
decoder, written here or taken as a dependency. It waits until a measured
uplink says it is needed. Every chunk carries its sample offset, so audio
that is resent after a reconnect is placed and not doubled.

**Decoding before the window is full.** Whisper hears 30 seconds at a time,
and its encoder takes the full 30 whether or not they hold sound. So the
live loop decodes the window again as it grows. Measured by `live_speed`
(below), it is:

1. About once a second, run the window from its start: the audio so far,
   padded to 30 s as always. Hand the decoder the tokens already settled as
   its prompt, so that only the new tail is decoded one step at a time. The
   tail is decoded *plain*, with `<|notimestamps|>`. Whisper leaves out a
   segment that runs into the end of its window, as it learnt to in
   training, and the segment being spoken always does. Timed, the partial
   is often empty while someone speaks: on tiny, for 16 s in a row.
2. A word is *settled* once two runs in a row agree on it, along with every
   word before it. That is LocalAgreement-2, from Macháček et al.'s
   whisper_streaming (2023). Settled words go down as deltas and never
   change. The rest goes down as the partial. Words are compared, not
   tokens with timestamps: those move a step or two between runs whose
   words do not, and compared, they never settle.
3. When the window holds more than 20 s, it is finished where a settled
   segment ends and the next begins, and the next window starts there with
   what was settled after the cut kept, as the file path's windows are cut
   (`#247`). The plain tail has no timestamps, so one timed decode of the
   window finds them, and its words are matched to the settled ones. A
   lone closing timestamp is not a cut: the model closes a segment early
   when it has not heard the next one, and a window started there hears
   the end of it again.
4. The rules that already hold stay in force: UTF-8 kept whole (`#245`),
   loops caught (`#246`, `#247`), a hang-up stops the work after the step
   under way. The example stops a tail that writes eight tokens it has
   already written, which is a stand-in for that.

**What it costs, measured.** `cargo run --release -p kvad-gpu --example
live_speed` plays all 85 seconds of `king.mp3` in at real time on an M5 Pro,
with an update due every second. It runs each set of backends in its own
process, interleaved in rounds: three for large, two for the others. The
range is over the rounds. "Shown" is how long after a word was spoken it
first showed, unchanged from then on, as a partial. "Settled" is when it
went down as a delta. Each is the median, then the 90th percentile, over
every word. "Spoken" is where the same backend's timed transcript of the
whole file puts the end of the word. Whisper's segment times are a second
or two out at worst, so read these to about a second.

| model | backend | an update (min, median, max) | busy | shown | settled | keeps up |
|---|---|---|---|---|---|---|
| large | gpu-q8 | 0.35, 0.45, 1.4 s | 46–47% | 1.5–1.6, 3.9–4.0 s | 2.5–2.6, 5.0–5.1 s | yes |
| | gpu-f16 | 0.31, 0.43, 1.8 s | 46% | 1.6, 3.9–4.0 s | 2.6, 5.1–5.2 s | yes |
| | cpu-q8 | 7.5, 18–20, 22.5 s | 96–97% | 31–51, 46–64 s | 33–51, 46–64 s | **no** |
| **small** | **gpu-q8** | 0.10, 0.13, 0.40 s | 14% | **0.4–0.5, 1.7–1.8 s** | **1.4–1.5, 2.7–2.8 s** | yes |
| | gpu-f16 | 0.09, 0.13, 0.45 s | 14–15% | 0.5, 1.8–1.9 s | 1.4–1.5, 2.8 s | yes |
| | cpu-q8 | 1.5, 1.8, 4.2 s | 84–88% | 4.9–5.3, 10–14 s | 7.8–8.4, 14–18 s | no |
| tiny | gpu-q8 | 0.04, 0.06, 0.17 s | 7% | 0.7–0.9, 1.8–2.9 s | 1.7–1.9, 2.8–3.9 s | yes |
| | gpu-f16 | 0.04, 0.06, 0.20 s | 6–7% | 1.1–1.5, 3.9–6.7 s | 2.1–2.4, 4.9–7.7 s | yes |
| | cpu-q8 | 0.26, 0.30, 1.0 s | 36% | 0.9, 2.4–2.5 s | 1.8–1.9, 3.4–3.5 s | yes |

`gpu-f16` is what a server's `gpu-bf16` loads a Whisper at (`DENSE`).

- **The budget held. The lag did not, on large.** An update on large at
  gpu-q8 is 0.45 s: 25 ms of spectrogram on the host, 370 ms of encoder and
  prompt (about 120 tokens), and 7 new steps at 6.6 ms. That is inside the
  estimate's 0.35 to 0.5 s, and the GPU is busy under half the time. Within
  a sentence, a word shows about a second after it is spoken, and settles
  a second later, as estimated. But large writes nothing of a new sentence
  until it is nearly finished. "Det er ikke alltid så lett å si hvor vi er
  fra", which the file transcript puts at 23.8 to 28.7 s, first shows at
  28.4 s, whole. That is the 4 s at the 90th percentile. A guess at why:
  NB-Whisper was trained largely on subtitles, which come a sentence at a
  time.
- **Small is the model for live.** It shows a word in half a second at the
  median, under 2 s at the 90th percentile, and settles it a second after
  that, with the GPU busy a seventh of the time. Of the 132 words it
  settles, 129 are the file transcript's, in order. Large's are 126 to 127
  of 132. Tiny is quicker still, but its text is worse: 78 to 82 of its
  110 to 121 words are the file's, and tiny's file transcript itself goes
  wrong at the end of the recording.
- **The CPU keeps up with tiny only.** Small on the CPU manages an update
  every 2.5 s or so, with words 5 s behind. Large manages 7 updates in 85 s,
  each needing 8 s for the encoder alone, and ends 25 to 42 s behind. So a
  server without a GPU offers tiny for live.
- **Timed partials are worse everywhere**, which is why step 1 is plain.
  On gpu-q8, large shows words at 1.6–1.7 s and 4.2–5.4 s, against plain's
  1.5–1.6 s and 3.9–4.0 s. Small's 90th percentile goes from 1.8 s to
  2.2–2.9 s, and to 9.5 s in one round at f16. Tiny's window filled 48
  times with nothing settled to cut at, and it settled 68 words, 31 of
  them the file's.
- **The cut costs an update.** The timed decode that finds the cut takes
  0.76 s on large at q8 and 1.0 s at f16. With the update it comes before,
  that is the 1.4 and 1.8 s at the top of the range, and the next update,
  falling due meanwhile, is skipped. Here it hears the window again. The
  endpoint can reuse the update's encoder output instead, which should save
  the 0.37 s of hearing. That is not measured.
- **Idling between updates costs little.** Taken back to back (`--fast`),
  large's median update at gpu-q8 is 0.42–0.45 s, against 0.44–0.46 s at
  one a second. Most of that is the spectrogram on the host, 14 ms back to
  back and 24 ms after a second's idle: the CPU coming back cold, not the
  GPU.

**When the Mac falls behind.** Audio is never dropped. Partials are: when an
update is still running as the next is due, the next is skipped, and a
`lag` field on each event says how far behind the voice the text is, so the
phone can show it.

**When the link drops.** The phone keeps the whole recording on the phone,
whatever happens to the link, in storage only the app can read (iOS Data
Protection class Complete, or Android's app-private files). The server holds
the session for a minute after the link goes. On reconnect, the phone names
the session and the last offset that was settled, and resends from there.
After that minute, or with the Mac asleep, the phone has a file, and the
existing `POST /v1/audio/transcriptions` takes it as it does today. The
live path never loses anything that the file path would have kept.

**Nothing is kept, still.** The server keeps the audio in memory for as long
as the window needs it and the session lives, and writes none of it to
disk. The transcript goes down the return channel and nowhere else. A
device paired for transcription is scoped to this endpoint and the file one
(`docs/relay-plan.md`, *Least privilege for devices*).

**Tests.** A recording played into the WebSocket at real time gives, once
settled, the same text the file endpoint gives for the same audio. Not
bit-for-bit, since the windows fall differently; the check is a word error
rate between them, with the bound set from the first measurement
(`live_speed` matched 126 to 129 of 132 words, small and large). Settled
text never changes after it is sent. A link cut mid-sentence and resumed
gives the same transcript as one never cut. A slow decoder (an injected
delay) skips partials but never audio. And the same session runs over the
tunnel's three paths, relayed, direct and LAN, with a change from one to
another mid-sentence.

## What is next

1. **The decoder's step.** At q8 a step is about 6.7 ms against about 4 ms
   of bytes. Ablate it first, to find where the rest is. A fused LayerNorm
   with its residual add, and the GELU fused into the first MLP matrix, are
   the likely candidates.
2. **Beam search**, and the gzip test, if a measured word error rate says
   they are worth it.
3. **The web UI** is built: the Transcribe page. It streams, and writes
   the subtitle files itself from the segments, byte for byte the server's.
4. **Live transcription** over a WebSocket, words back while speaking, as
   *Live, from the phone* above lays out. `live_speed` has measured the
   cadence (small at gpu-q8, once a second). Next is the endpoint, which the
   Transcribe page can use with the browser's microphone before any phone
   exists.
