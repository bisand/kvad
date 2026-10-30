# Plan: who spoke when

Written 2026-10-18 as research and a plan; steps 1 and 2 of *The order
it is built in* are built (`speech::score`, `speech::align`), the rest not
yet. It
follows [the speech plan](speech-plan.md), whose Whisper, CPU and Metal,
file and live, it extends.

## What the owner asked for

To tell speakers apart in the transcript of a meeting, both a meeting
recorded in a room and one held online, on the owner's own machines. A
transcript today is a list of segments, each with a start, an end and its
text. What is asked for is the same transcript with a speaker on every
line, "who spoke when", which the field calls speaker diarization.

Three things do not exist yet, and the plan builds them in this order:

1. **Words with times.** Whisper's segments are a sentence or two, and two
   people often share one. A speaker change has to land between two words,
   so each word needs its own start and end.
2. **Speakers, offline.** A recording in, and "speaker A from 3.2 to 7.9 s,
   speaker B from 7.6 to 12.0 s" out, from models that hear voices and not
   words. Then each word goes to the speaker who was talking when it was
   said.
3. **Speakers, live.** The same while the meeting is under way, with labels
   that may change for a few seconds and then settle, as the words already
   do.

An online meeting can skip the second and third, when the meeting gives one
audio track per participant (*One track a person*, below).

## Words with times

Whisper never writes a word's time. Its timestamp tokens fall between
segments, 20 ms apart, and nothing in between has one. But the decoder
reads the audio through cross-attention, and in a few of its heads the
weight a text token puts on the 1500 audio frames peaks where that token is
spoken. OpenAI found which heads by hand, model by model, and ships them as
`alignment_heads` in each `generation_config.json`. openai-whisper's
`timing.py` turns them into word times, and transformers'
`_extract_token_timestamps` does the same arithmetic. That is what is
written here, once, on the host, with each backend answering one new
question.

### The arithmetic, from `timing.py`

For one window, once its tokens are decoded:

1. Run the decoder once more over `sot_sequence`, `<|notimestamps|>`, the
   window's text tokens (timestamps left out) and end-of-text, from an
   empty cache, against the window's cross keys and values.
2. Keep the cross-attention weights of each alignment head, softmaxed, as
   a `[tokens × frames]` matrix. Keep only the frames that hold the
   recording, `size / 2` of them, where `size` is the window's spectrogram
   columns (`transcribe`'s `size`); the rest are the silence the last
   window is padded with.
3. Standardise each head's matrix along the token axis: for every frame,
   subtract the mean over tokens and divide by the standard deviation
   (population, not sample).
4. Median-filter each row along time, width 7.
5. Average the heads. Drop the rows of the prompt and of end-of-text.
6. Dynamic time warping over the negated matrix, steps right, down and
   diagonal, from the first token at the first frame to the last at the
   last. Where the path first reaches each token is that token's time, at
   50 frames a second.
7. Split the tokens into words: a new word at each token that begins with
   a space or is punctuation, after grouping tokens into whole UTF-8
   characters (`transcribe::utf8` already knows which tokens are
   fragments). A word starts at its first token's time and ends at the next
   word's start.
8. openai-whisper's clean-ups, each a few lines: words longer than twice
   the median word (capped at 0.7 s) are cut back at a sentence's end;
   leading quotes and brackets join the next word and trailing punctuation
   the previous one; and the first and last words of a segment are held to
   the segment's own timestamps when they run more than half a second past
   them.

The matrix is at most 1500 frames by 224 tokens. The filter and the warp
are a few hundred thousand cells, under a millisecond on the host.

### What each backend adds

`speech::Model` gets a fourth question beside `hear`, `said` and `decode`:

```rust
/// The cross-attention weights of `heads` (layer, head) for `tokens`,
/// decoded from an empty cache against `heard`: `[head][token][frame]`,
/// softmaxed, in f32.
fn attention(&self, tokens: &[u32], heard: &Heard, heads: &[(usize, usize)]) -> Res<Vec<f32>>;
```

- **On the CPU**, `whisper::Whisper::decode`'s cross-attention is
  `attend_rows`, which already holds every (query, head) pair's softmaxed
  scores in a vector before it multiplies them into the values. A variant
  writes the asked-for heads' rows out as well. Nothing else changes.
- **On the GPU**, the cross-attention is MLX's fused kernel, and the
  weights never leave it. openai-whisper has the same problem with
  PyTorch's fused attention and turns it off for this one pass
  (`disable_sdpa`). Here, in each layer that holds an alignment head, the
  cross-attention is written out as `softmax(q kᵀ · scale)` in f32 for
  those heads, and the fused kernel still does the layer's work. The pass
  stops after the last layer with an alignment head: the logits are not
  needed, since OpenAI's `words` carry no probability. That is layer 25 of
  32 on large, 10 of 12 on small, 3 of 4 on tiny.

It costs one masked prompt-sized pass per window. On large at gpu-q8 a
window's encoder and its 120-token prompt took 370 ms in `live_speed`,
most of it the encoder, so the guess is a few tens of milliseconds a
window: a few percent of a file. It is measured in the step below, not
assumed.

### The heads are OpenAI's, and they still work

Every NB-Whisper ships the alignment heads of the OpenAI model it was
fine-tuned from, unchanged. Read off the cache:

| model | `alignment_heads` (layer, head) |
|---|---|
| nb-whisper-tiny, whisper-tiny | (2,2) (3,0) (3,2) (3,3) (3,4) (3,5) |
| nb-whisper-small | (5,3) (5,9) (8,0) (8,4) (8,7) (8,8) (9,0) (9,7) (9,9) (10,5) |
| nb-whisper-large | (7,0) (10,17) (12,18) (13,12) (16,1) (17,14) (19,11) (21,4) (24,1) (25,6), large-v3's |

66,000 hours of fine-tuning could have moved what those heads attend to.
So a throwaway probe checked, before anything else was planned on them: a
synthetic dialogue, eight turns between two voices, 27 s, with gaps of 0.2
to 0.35 s between turns. Voice A is macOS's Nora; voice B is Nora pitched
down and retimed, so the words keep their length. Each clip's silence was
trimmed, so the true turns are known to the sample. transformers'
word timestamps (the same DTW, these heads) on NB-Whisper in f32 on the
CPU, each word matched to the script and given to the turn its time falls
in:

| | words given the right speaker, by word times | by whole segments | turn change to the nearest gap between words |
|---|---|---|---|
| nb-whisper-tiny | 72 of 74 | (transformers' timed tiny wrote one segment) | 0.09–0.43 s, median 0.22 |
| nb-whisper-small | 74 of 75 | 61 of 75 | 0.01–0.27 s, median 0.05 |
| nb-whisper-large | 76 of 76 | 70 of 76 | 0.00–0.09 s, median 0.06 |

- **Segments are not enough.** Large put "Det er godt å høre. Men vi må
  fortsatt kutte litt i reisekostnadene." in one segment, and "Ja. Kan vi
  ta det opp med styret på torsdag?" in another: four turns in two
  segments. Small did the same and worse. Giving a segment to whoever
  speaks most of it gave every fifth word to the wrong speaker on small,
  and every thirteenth on large, on clean audio with polite turns.
- **The heads still align.** Every misattributed word by word time was a
  one-word turn or the word next to one: tiny stretched "fjor." 0.45 s
  past the end of its turn and ran "Ja," into the next, and small gave
  "Kan" to the "Ja." before it.
- **What this does not show.** One voice, pitched, read by a synthesiser
  with clean gaps and no overlap. It says the heads did not break in
  fine-tuning. It says nothing about real meetings, which the
  measurements under *How it is measured* are for. The scripts are in the
  session notes, and the dialogue maker becomes `scripts/dialogue.py`
  (below), as a fixture any later step can use.

**A model with no `alignment_heads`.** openai-whisper's default then is
every head in the second half of the decoder's layers. kvad does the same,
and says so in the server's log, since such times are worse.

### In the API

OpenAI's shape: `timestamp_granularities[]=word` with
`response_format=verbose_json` adds a `words` array, each
`{"word", "start", "end"}`, beside `segments`. `kvad transcribe --words`
asks for it. Word times work without timestamps too: the alignment pass
decodes with `<|notimestamps|>` whatever the window was decoded with, so
the live loop's plain tail gets word times (*Live labels*).

### Done, and measured

- `scripts/whisper-fixtures.py` writes transformers' word times
  (`return_token_timestamps`) beside its other stages. In f32, kvad's
  agree with them to within one frame (20 ms) for every word on tiny,
  small and large, on the CPU and on the GPU. At f16 and q8 the bound is
  set from the first measurement, as the token tables were.
- On `scripts/dialogue.py`'s recording, words given the right speaker by
  their times are at least the probe's: 72 of 74 on tiny, 74 of 75 on
  small, 76 of 76 on large.
- `transcribe_speed` gains a `--words` column: the cost of the alignment
  pass per window and over the whole file, at every backend.

### As built, and what it measured

`crates/llm/src/speech/align.rs` is the arithmetic above, steps 1 to 8;
`Model::attention` is the fourth question, on the CPU
(`Whisper::attention`) and the GPU (`GpuWhisper::attention`, the alignment
layers' cross-attention written out in f32, the pass stopped after the last
of them). One clean-up is left out on purpose: openai-whisper also moves a
segment's times to its words'; here a segment keeps the times it was
decoded with, so a transcript is the same with words or without.

**transformers is not quite `timing.py`.** Its `token_timestamps[k]` is
when token `k` *ends*, which is when `k + 1` starts in `timing.py`. It
drops the `<|notimestamps|>` row as well as the prompt's, so its first
token ends at frame 0, and it standardises over the generated rows only.
And the pipeline's `return_timestamps="word"` decodes *with* timestamp
tokens, so the probe's numbers came from a timed decode.
`scripts/whisper-fixtures.py` therefore writes three things: the alignment
heads' attention, the token starts by `timing.py`'s arithmetic over that
attention (`token_starts`, its median filter and DTW written out as
there), and transformers' own `token_ends`, with the attention mask passed
so that it knows where the recording stops. The test holds kvad to the
first two and reports the third.

On the dialogue, one window, in f32:

| model | attention, largest gap | token starts from `timing.py`'s | transformers' own ends, more than a frame out |
|---|---|---|---|
| tiny, CPU and GPU | 7e-6 / 4e-6 | 0 frames, all 136 | 1 of 134 (the second token, 5 frames: its first is pinned at 0) |
| small, CPU and GPU | 7e-6 | 0 frames, all 141 | 4 of 139, 2 frames each |
| large, CPU and GPU | 5e-6 | 0 frames, all 134 | 3 of 132, 2 frames each |

On the GPU at the other precisions, the most any token start moved from
`timing.py`'s: f16 1, 0, 0 frames (tiny, small, large); q8 1, 1, 1; bf16
15, 3, 1; q4 8 on large. bf16 is what the server's `gpu-bf16` is not: it
loads f16 (`speech.rs`'s `DENSE`).

Words given the right speaker on `scripts/dialogue.py`'s recording, by
`crates/gpu/examples/word_turns.rs` (the probe's rule, matching by fewest
edits instead of difflib), the same at cpu-f32, gpu-f16 and gpu-q8:

| | by word times | by whole segments | the probe, by word times |
|---|---|---|---|
| tiny | 73 of 73 | 38 of 73 (one segment) | 72 of 74 |
| small | 75 of 75 | 61 of 75 | 74 of 75 |
| large | 76 of 76 | 70 of 76 | 76 of 76 |

Small wrote no "Ja." this time, so its one-word turn was not tested.

What it costs, nb-whisper-large on the 85 s of `king.wav`, three windows,
two rounds each, a backend a process, with and without `--words`:

| backend | an alignment pass | of the run | the run, without and with |
|---|---|---|---|
| gpu-q8 | 66–67 ms | 4.6–4.7% | 4.2 s, 4.2–4.3 s |
| gpu-bf16 (f16) | 37–39 ms | 1.8–1.9% | 5.5–6.5 s, 6.1 s |
| gpu-f32 | 52–57 ms | 2.1–2.2% | 8.4–8.6 s, 7.5–7.8 s |
| cpu-f32 | 0.7–0.9 s | 2.7–3.6% | 61–93 s, 60–95 s |
| cpu-q8 | 1.8–1.9 s | 4.5–8.1% | 34–43 s, 66–129 s |

The GPU rows are a few percent, as guessed. The CPU rows were taken with
the machine's load average near 28 from other builds, and a q8 step ran
from 30 to 206 ms within one process: they say the pass is a prompt's
cost, and nothing finer, until taken again on a quiet machine. q8's pass
being slower than f32's is in that noise or is the prompt-shaped matmul
at q8; it is not yet known which.

## Speakers, offline: the pipeline

pyannote's, which is the field's reference and whose every stage is
published, MIT-licensed code. Version 3.1 and community-1 (2025) run the
same two networks; community-1 changed only the clustering, and gained
1.8 points of DER on AMI for it. Four stages:

1. **Who is speaking, locally.** A segmentation model hears 10 s at a
   time, stepping 1 s, so every moment is heard ten times. For each of
   589 frames (17 ms apart) it gives the probability of each of seven
   classes: nobody, one of three local speakers, or one of the three
   pairs. That is pyannote's *powerset*: up to three speakers a chunk and
   two at once. "Local speaker 2" means only "the second voice in this
   chunk"; the next chunk numbers its own.
2. **What each local speaker sounds like.** For each chunk, and each local
   speaker active in it, an embedding model turns the chunk's audio into a
   256-number voiceprint, its statistics pooled over that speaker's frames
   only, overlap left out (`embedding_exclude_overlap`).
3. **Which local speakers are the same person.** Clustering the
   embeddings of the whole recording. 3.1: agglomerative, centroid
   linkage, cosine distance under 0.7046, clusters of fewer than 12
   members merged into their nearest. community-1: the same at 0.6 as a
   start, then a PLDA transform to 128 dimensions and VBx (a variational
   Gaussian mixture, at most 20 iterations, `Fa` 0.07, `Fb` 0.8, no HMM).
   Then each chunk's local speakers are assigned to clusters by the
   Hungarian algorithm, so that two voices in one chunk are never one
   person.
4. **Back to time.** Each frame's speaker count is the segmentation's,
   averaged over the ten chunks that heard it. Each frame goes to that
   many clusters, the strongest first. Runs of frames become intervals.

kvad builds 3.1's clustering first, because it is a page of arithmetic,
and community-1's VBx after it, measured against it.

### The models, and their licences

The owner asked for models whose licences allow redistribution and
commercial use. Every licence below was read off the model card and the
Hub's metadata on 2026-10-18; the session notes hold the quotes and links.

| model | role | parameters | weights' licence | on the Hub | verdict |
|---|---|---|---|---|---|
| `pyannote/segmentation-3.0` | segmentation | 1,473,265 | MIT | `pytorch_model.bin`, 5.9 MB; gated by a contact form | **chosen** |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | embedding | 6,634,336 | CC-BY-4.0 | `pytorch_model.bin`, 26.6 MB; not gated | **chosen** |
| community-1's PLDA (`plda.npz`, `xvec_transform.npz`) | clustering | 2 small matrices | CC-BY-4.0 (BUT) | inside the gated community-1 repo | **chosen**, for VBx |
| `nvidia/Nemotron-3-Diarization` | all of it, offline and streaming | 99,226,504 | OpenMDW-1.1 | `model.safetensors`, 397 MB; not gated | **chosen**, as the contender (below) |
| Silero VAD | speech or not | 309,633 | MIT | `silero_vad_16k.safetensors` on GitHub | in reserve, for tracks |
| `speechbrain/spkrec-ecapa-voxceleb` | embedding | about 20.8 M | Apache-2.0 | `.ckpt` (pickle) only | not chosen: three times the size, no better for this |
| `nvidia/speakerverification_en_titanet_large` | embedding | about 23 M | CC-BY-4.0 | `.nemo` | not chosen: same reason |
| 3D-Speaker CAM++ | embedding | 7.2 M | Apache-2.0 (ModelScope) | ModelScope, not the Hub | not chosen: off the Hub; its training data's licence unread |
| `nvidia/diar_streaming_sortformer_4spk-v2` | streaming diarizer | about 117 M | CC-BY-4.0 | `.nemo`, gguf | superseded by Nemotron; 4 speakers |
| `nvidia/diar_sortformer_4spk-v1` | offline diarizer | 123.5 M | **CC-BY-NC-4.0** | safetensors | **excluded**: not commercial |
| `BUT-FIT/diarizen-wavlm-large-s80-md` | segmentation | about 69.5 M | **CC-BY-NC-4.0** | `.bin` | **excluded**, though the best on AMI |
| Rev's reverb-diarization v1, v2 | segmentation | — | **Rev non-production** | via sherpa-onnx | **excluded** |
| pyannoteAI precision-2 | all | — | closed, hosted | API only | **excluded**: not on premise |

- **A caveat that applies to every speaker-embedding model on the Hub.**
  They are all trained on VoxCeleb, which its publishers offer "for
  research purposes" under CC-BY-4.0, with the videos' copyright left
  with their owners. The weights' own licences (CC-BY-4.0, Apache-2.0)
  allow commercial use, and whether a dataset's terms reach a model
  trained on it is unsettled. That is the owner's call, not this plan's;
  it is said plainly in the documentation, beside the attribution
  CC-BY-4.0 asks for. Nemotron's card lists no VoxCeleb: its real
  conversations are public corpora (AMI, ICSI, VoxConverse and others),
  LDC corpora (Fisher, CALLHOME part 1, DIHARD III's development set) and
  data NVIDIA licensed by agreement, and OpenMDW-1.1 puts no limits on
  use or on outputs. It needs no embedding model at all unless speakers
  are named.
- **WeSpeaker's own repos disagree with themselves** (some cards say
  apache-2.0, others cc-by-4.0, and WeSpeaker's documentation says a model
  follows its dataset's licence). pyannote's copy of ResNet34-LM is
  unambiguous (CC-BY-4.0), ungated, and is the one pyannote's published
  numbers were made with, so it is the one used.
- **The gate.** segmentation-3.0 is MIT but sits behind a form that asks
  for a company name. `kvad pull` uses the user's Hub token as it does for
  any gated model, and when the gate is not yet accepted it says which
  page to accept it on. MIT allows a mirror, and ungated ones exist
  (`onnx-community/pyannote-segmentation-3.0`, others); a converted
  safetensors mirror under the project's own name is an open question for
  the owner, not something this plan does.

### Reading the weights

pyannote's checkpoints are PyTorch `.bin` files: a zip holding a pickle
and the raw tensor bytes. `weights.rs` says why kvad reads safetensors
only: a pickle is a program, and loading one runs it. So a third reader,
in `weights.rs`, never runs anything:

- The zip is read by hand. PyTorch stores its members uncompressed, so
  this is the central directory and a slice per member, under a hundred
  lines; anything compressed is refused.
- The pickle is read as data. Its opcodes build a tree; a `GLOBAL` names
  a callable, and here it becomes a node with that name, never a call. Of
  the tree, only `collections.OrderedDict` and
  `torch._utils._rebuild_tensor_v2` over a `torch.FloatStorage` are
  understood: a name, a storage key, an offset, a shape and strides,
  which say where in the zip a tensor's bytes are. Everything else in a
  Lightning checkpoint (hyper-parameters, pyannote's task description) is
  left as an opaque node, and nothing asks for it.
- community-1's PLDA files are `.npz`: the same zip, of `.npy` arrays,
  whose header is a Python dict literal of three keys.

`scripts/diar-fixtures.py` writes a tiny checkpoint with torch, and a test
reads it and compares every tensor. Nemotron ships safetensors and needs
none of this.

### What the layers are, in kvad's terms

**Segmentation** (PyanNet). Raw 16 kHz audio in, which suits it to the CPU:

| layer | shape | in kvad |
|---|---|---|
| InstanceNorm1d on the waveform | 1 channel | new: a mean and a variance over time, a dozen lines |
| SincNet: 40 learnt band-pass pairs | conv1d 1→80, kernel 251, stride 10 | the 80 filters are computed once at load from the 80 learnt cut-offs (`low_hz`, `band_hz`) and a Hamming half-window, and become an ordinary convolution, which `Checkpoint`'s `[out, in, k]` and the im2col matmul already do |
| `abs`, MaxPool1d(3), InstanceNorm1d(80), LeakyReLU | | new, each a loop |
| conv1d 80→60 and 60→60, kernel 5, each with pool, norm, LeakyReLU | | im2col, as Whisper's convolutions |
| LSTM, 4 layers, both directions, 128 wide | 60 → 256 | **new.** Per layer and direction, the input projection for every frame is one matmul (`gemm_bt`); the recurrence is 589 steps of a 512×128 matrix times the state, and the gates (σ, tanh) in PyTorch's order, input, forget, cell, output. Run over many chunks at once, each step is a matmul with a row per chunk |
| Linear 256→128→128, LeakyReLU; Linear→7, log-softmax | | `matmul_bt` |
| powerset to speakers | 7 → 3 | a table |

It is about 2.6 GFLOP a chunk, 60% of it the LSTM, which is sequential
within a chunk and batched across chunks.

**Embedding** (WeSpeaker ResNet34). Spectrogram in, image-shaped:

| layer | in kvad |
|---|---|
| Kaldi fbank: audio ×32768, 25 ms frames every 10 ms, Hamming, 512-point FFT, 80 triangular filters on the HTK mel scale, log; then each bin's mean over the chunk subtracted | **new**, beside `mel.rs`, whose FFT it shares. Held to `torchaudio.compliance.kaldi.fbank` by a fixture, as `mel.rs` is held to transformers' extractor |
| conv2d 3×3, 1→32; BatchNorm; ReLU | BatchNorm folds into the convolution's weight and bias at load, so it costs nothing. conv2d on the CPU is im2col over frequency and time, then `gemm_bt`: new, the same idea as conv1d. On the GPU it is kvad-gpu's image `Conv2d` |
| 16 residual blocks [3, 4, 6, 3], 32→64→128→256 channels, stride 2 into each stage | the same, plus adds |
| statistics pooling: mean and standard deviation over time of 256 channels × 10 bands, weighted by a local speaker's frames | new, a few lines |
| Linear 5120→256 | `matmul_bt` |

By its shapes it is about 45 GFLOP for each 10 s chunk, 17 of it the
third stage. An hour is 3,600 chunks, so about 160 TFLOP an hour if the
trunk runs once a chunk, and three times that if, as in pyannote, it runs
once per local speaker. It need not: the weights enter only at the
pooling, so one trunk serves all three, and the answer is the same
arithmetic. Even so it is the expensive stage: about 13 minutes an hour of
audio at the CPU's 200 GFLOP/s, so four times real time, and ten seconds
or so an hour on the GPU at f16. These are counts from the shapes, to be
replaced by measurements.

**Clustering** is host code on thousands of vectors of 256:
cosine distances, centroid linkage, the Hungarian algorithm (which the
scorer needs too), and VBx's few matrix products a step. A 3-hour
recording is about 30,000 embeddings, whose pairwise distances are 1.8 GB
in f32. So linkage runs on the embeddings of the cleanest stretches
first, as pyannote's `min_cluster_size` already implies, and the rest are
assigned to the nearest centroid. Measured on the longest recording in
the sets below.

### The contender: Nemotron-3-Diarization

Published by NVIDIA on 2026-09-01: one network that does the whole job,
offline or streaming, for up to eight speakers. Its tensors were read off
the safetensors header:

- 128 mel bands every 10 ms, eight frames stacked into one (1024 numbers)
  and projected to 512, so the encoder runs at 80 ms a step.
- 31 pre-norm transformer blocks, 512 wide, 8 heads, rotary positions,
  LayerNorm, a GELU MLP 2048 wide, no bias on the key, query and value
  projections. This is exactly the kind of model kvad already runs, on
  both backends.
- A 192-wide head, a convolution that unpacks each 80 ms step into eight
  10 ms ones, and eight sigmoids: one per speaker, in the order they
  first spoke. No clustering.
- For streaming, a FIFO of recent frames and an arrival-order speaker
  cache, so a speaker keeps their channel across chunks. That is host
  bookkeeping around the network.

It is about 9 TFLOP an hour of audio offline (2 × 99 M × 45,000 steps),
a twentieth of the ResNet path. Its card reports 12.7% DER on DIHARD III
offline and 13.2% at 1 s latency, where pyannote's cards report 21.7% for
3.1 and 20.2% for community-1 on the same set, each by its authors' own
run, no collar, overlap scored. The two cards' AMI numbers use
different references and cannot be compared.

**It was trained on VoxConverse's test set** (the card lists VoxConverse
v0.3 "development and test" among its training data), so VoxConverse
says nothing about Nemotron, and its tables leave that column out. AMI's
and NOTSOFAR's training and development sets and CALLHOME part 1 are in
its training data too; AMI's test set, and both Norwegian sets, are
not.

What it does not do: more than eight speakers (a Stortinget sitting has
dozens), and Norwegian is not among the languages its card lists. Speaker
turns depend on voices more than on words, but that is a guess until
measured.

The owner's meetings stay under eight people (*Open questions*), so
Nemotron is built first, as the whole diarization, offline and live
(reordered 2026-10-30). The pyannote pipeline follows for what Nemotron
cannot do: the segmentation's speech activity cuts per-person tracks,
the embedding model names speakers from reference clips, and the two
with clustering handle more than eight. Both sit behind the same
interface, and the measurements choose the default per case. A transformers port
(`Nemotron3DiarizationForAudioFrameClassification`) exists to hold it to.

## Words to speakers

The diarization gives each speaker a set of intervals, overlapping where
two talk at once. Whisper gives one stream of words. Each word goes to the
speaker whose intervals cover the most of it. A word that falls in no
interval (a gap, or a word time a frame or two out) goes to the nearest
speaker within half a second, and otherwise to the speaker of the word
before it. Where two speakers overlap, Whisper usually writes only one of
them, and the word goes to whichever covers more of it. That is
whisperX's rule (`assign_word_speakers`), and the probe above used it.

Then consecutive words with one speaker make a turn, and Whisper's
segments are cut where the speaker changes, keeping Whisper's own
boundaries everywhere else. A segment that one speaker holds throughout
comes out as it went in.

One correction is worth measuring before it is kept: a speaker change one
word away from the end of a sentence moves to the sentence's end, because
a word time is likelier to be a frame out than a person to say one word of
someone else's sentence. Kept only if it lowers the word-level error on
the measured sets.

**Done** when `speech::speakers` (host code) turns a transcript with words
and a diarization into labelled segments, with unit tests on hand-made
cases: a turn inside a segment, a word in a gap, an overlap, a one-word
turn. Measured as word diarization error (*How it is measured*).

## Live labels

The live loop the speech plan measured (`live_speed`, not yet an endpoint)
decodes a growing window again about once a second, plain, and settles a
word once two runs agree on it. Speakers follow the same idea: a label is
a guess until it settles, and only settled labels are final.

**Online clustering, as diart does it.** Coria, Bredin et al.'s *diart*
(ASRU 2021, MIT licence) runs pyannote's segmentation on a rolling 5 s
buffer every 500 ms, takes an embedding for each local speaker in the
buffer, and assigns it to the closest known speaker if their cosine
distance is under a threshold, or makes a new one. Two local speakers in
one buffer may not go to the same known speaker, so the assignment is a
small matching problem. Each known speaker's centroid is the running sum
of what was assigned to it. Its latency runs from 0.5 to 5 s, a trade
against accuracy. Here:

1. Every update (once a second), the segmentation model runs on the last
   10 s of audio, its training length. Its frames for the new second are
   aggregated with the earlier, overlapping runs as the offline path does.
2. Each local speaker active for at least a second in that buffer gets an
   embedding, and is matched to the known speakers (Hungarian on cosine
   distance, with a "new speaker" column at the threshold).
3. Words get the speaker under them by the rule in *Words to speakers*,
   with the word times from the alignment pass over the plain tail.

**When a label settles.** A frame's segmentation is final once the last
buffer that holds it has run, 10 s later. A word's label settles when its
word has settled and either its frames are final or the last three updates
gave it the same speaker, whichever is first. Until then, it goes down
with the partial and may change.

**When speakers merge.** Online clustering makes two speakers out of one
voice when the first second of them was unlike the rest (a laugh, a cough,
a new microphone position). Every 30 s, and when the session ends, the
centroids are clustered again offline, and two that are closer than the
threshold become one. That renames a settled label, which settled text
never does. So a label is a speaker's id, and a `speakers` event says
"speaker C is now speaker A", which the client applies to everything it
has shown. Words never change; the name over them may, until the session
ends.

**What it costs.** Each update adds a 10 s segmentation and one embedding
per active speaker. Both models are a few million parameters: guesses of
under 20 ms together on the GPU and under 100 ms on the CPU, beside the
130 ms an update takes on small at gpu-q8 today. Measured in its step.

**Done** when `live_speed` gains a `--speakers` flag and reports, beside
its word lags, when a word's label was first shown and when it settled
(median and 90th percentile), how many labels changed after first being
shown, and the session's DER against the reference, beside the offline
DER of the same recording. Bounds: settled labels' DER within 5 points of
the offline path's; the update still under the one-second budget on small
at gpu-q8.

**A model built for this.** Nemotron-3-Diarization (*The contender*)
labels speakers frame by frame as audio streams in, in the order they
first spoke, keeping each one's channel through its speaker cache, at
0.32 to 1.04 s of latency by its card. It is the shape this problem
wants, and step 6 builds the live labels on it; the diart loop above
waits for the pyannote models (steps 8 and 9) and is measured beside it then.

## One track a person

An online meeting already knows who is talking: each participant's audio
comes from their own microphone. Where a meeting platform hands those
tracks over separately, there is nothing to cluster. Transcribe each track,
label its words with its participant, and merge the words by time.
Diarization is then exact, overlap included, and the names are the
platform's.

**Every track needs a speech detector.** A participant's track is silence
most of the meeting, and NB-Whisper never says `<|nospeech|>` (speech
plan, *Known differences*): it writes something for every window it is
given. So each track is cut to its speech first, by the segmentation
model's speech activity (or a smaller detector), and only the speech is
transcribed, with each piece's offset kept. That also makes the cost the
meeting's speech, not its length times its participants.

**Bleed.** A participant in a room with a loudspeaker, or two people at one
laptop, puts others' voices in their track. Platforms cancel echo, so this
is the exception. A word written at the same time, within a frame or two,
on two tracks, with the same text, is kept on the louder track only.

**What it takes as input.** Two shapes, both Kvad's own, since OpenAI has
neither:

- *A file of several tracks.* `POST /v1/audio/transcriptions` with
  `file` repeated, one part a participant, and `speaker[]` naming each in
  order (the file name if absent). Or one file with a channel a
  participant, the common shape of a recorded phone call (agent on the
  left, caller on the right), asked for with `channels=separate`. The
  answer is one transcript with every segment's `speaker` set.
- *Several live streams.* One live session carrying several tracks: each
  binary audio frame starts with a track number, and the session update
  names the tracks. A meeting bot sends every participant's audio up one
  WebSocket. Several phones in one room each opening their own session,
  joined by a meeting id, is the same thing with one track a session,
  and waits until someone asks for it.

Getting the tracks out of a meeting platform is the platform's business:
whether its recordings or its bot interfaces give unmixed audio per
participant differs by platform and plan, and is checked against each one
when a user asks for it. Nothing here depends on one.

**Done** when a two-track recording (two `say` voices on two channels,
with overlap) comes back with every word on the right speaker, and when a
track that is all silence comes back with no words. Measured on AMI's
individual headset microphones, which are exactly this input (*How it is
measured*).

## In the API, the CLI and the web UI

**Asking for speakers.** OpenAI's shape is a model of its own,
`gpt-4o-transcribe-diarize`, answering `response_format=diarized_json`.
Here a model is a Whisper and stays one, so:

- `response_format=diarized_json` asks for speakers, in OpenAI's format:
  `task`, `duration`, `text`, and `segments`, each
  `{"type": "transcript.text.segment", "id", "start", "end", "text",
  "speaker"}`. Speakers are `A`, `B`, … in order of first appearance, as
  OpenAI labels them. `text` puts each turn on its own line after its
  speaker, `A: …`, as OpenAI's example does.
- `diarize=true` (Kvad's own) adds speakers to the other formats, which
  OpenAI's diarizing model does not offer:
  - `verbose_json`: each segment, and each word when words were asked for,
    gains `"speaker"`. A client that does not know the field ignores it.
  - `vtt`: WebVTT's voice span, `<v A>…`, one cue a turn. Players show the
    name, and `::cue(v[voice="A"])` can colour it.
  - `srt`: SubRip has no voice markup, so `A: ` before the text, as in
    `text`.
  - `text` and `json`: `A: …` lines, as `diarized_json`'s `text`.
- `num_speakers`, or `min_speakers` and `max_speakers` (Kvad's own, the
  names pyannote uses), when the caller knows. Unknown is the default.
- `known_speaker_names[]` and `known_speaker_references[]`, OpenAI's
  fields: up to four 2–10 s clips of named speakers, sent with the
  request. Each reference's embedding is compared to each cluster's, and a
  match under the threshold names it. The references are used for that
  request and dropped with it (*Privacy*).
- **Streaming.** OpenAI's diarizing stream sends `transcript.text.delta`
  events with no speaker and a `transcript.text.segment` event, speaker
  included, when a segment is final. The file path can only cluster once
  it has heard the whole recording, so the deltas come as today, one a
  window, and the segments with their speakers come after the last window,
  then `transcript.text.done`.

**The CLI.** `kvad transcribe --speakers` (`--speakers 2` or
`--speakers 2-4` when known) and `--words`, writing any format as above.
`kvad transcribe --speakers --format text` is the meeting minutes: one line
a turn.

**The Transcribe page.** A "Speakers" switch beside "Timestamps", off by
default. With it on, each segment has a coloured chip with its speaker,
one colour a speaker. Clicking a chip renames the speaker everywhere ("A"
to "Kari"), and dragging one chip onto another merges them, for the case
the clustering split one person in two. Both are edits on the page only:
the page already writes its subtitle files itself from the segments, so
renamed speakers go into the downloads with no round trip, and nothing
about the names reaches the server. Live, a label that has not settled is
shown in its colour's outline only, and filled in when it settles.

**Done** when each format has a test that renders a hand-made diarized
transcript byte for byte, `diarized_json` is checked against OpenAI's
OpenAPI schema (`TranscriptionDiarizedSegment`) and parsed by their Python
SDK, and the server, the CLI routed to it and `--local` write the same
bytes, as the speech plan's formats do. The page gets a screenshot from a
clean demo server in the PR.

## Privacy

A voiceprint is biometric data. Under the GDPR, which binds Norway through
the EEA, biometric data processed "for the purpose of uniquely identifying
a natural person" is a special category (Article 9), which may be
processed only on one of the article's narrow grounds, explicit consent
being the one that applies here. Telling "speaker A" from "speaker B"
inside one recording, without knowing who either is, is arguably not
identification. Kvad does not lean on that argument. It treats every
embedding as biometric and arranges never to keep one it does not have to.

**By default, nothing is kept.** Embeddings exist in memory for the request
or the live session that made them, and are dropped with it. They are
never written to disk, never logged, never sent to a client, and never in
an error message. Labels are letters. What the transcript says is personal
data as it was before, and is kept the way transcripts are today: not at
all, on the server.

**Named speakers per request.** `known_speaker_references[]` come with the
request and go with it. Whoever sends a voice sample is responsible for
having the speaker's consent; the API documentation says so plainly.

**Saved speakers only with consent.** Naming people across meetings
("this is Kari, every time") means keeping a voiceprint, which needs the
speaker's explicit consent. Built last, and only if asked for:

- Off unless the server's configuration turns it on (`speech.enrollment =
  true`).
- `POST /v1/speakers` with a name, 10–60 s of the person's speech, and a
  consent record: who consented, when, and to which text, stored with the
  embedding. The audio is turned into an embedding and dropped. Only the
  embedding and the record are stored, in the data directory, readable by
  the server's user only, and scoped to the identity that created it.
- An embedding belongs to the model that made it. A new embedding model
  cannot read old ones, so the person enrols again; keeping their audio to
  re-embed it would mean keeping more than needed.
- `GET /v1/speakers` lists names, dates and consent records, never
  embeddings. `DELETE /v1/speakers/{id}` deletes the embedding and the
  record from the disk at once, and a deletion log keeps only the id and
  the time. Every saved speaker has an expiry (a year unless configured),
  after which it is deleted the same way.
- Whoever runs a server with enrollment on is the controller of that data,
  and processing special-category data at scale generally calls for a data
  protection impact assessment (Article 35). The documentation says this,
  and does not pretend to be legal advice.

**Done** when tests show that a diarized request leaves nothing behind in
the data directory, that no embedding reaches a log at the most verbose
level, and that a deleted speaker is gone from the disk.

## How it is measured

**Three numbers, for three questions.**

- *Diarization error rate (DER)*: of all the time someone is speaking in
  the reference, the share that is missed (speech called silence), false
  alarm (silence called speech) or confused (speech given to the wrong
  speaker), after the hypothesis's speakers are mapped one to one onto the
  reference's to make that least. Overlap counts per speaker: two people
  talking for a second is two seconds of reference speech. It measures
  the diarization alone, without Whisper.
- *Word diarization error (WDER)*: of the words Whisper got right
  (aligned to the reference transcript), the share given to the wrong
  speaker, after the same mapping. Shafey et al. (Google, 2019). It is
  what a reader of the transcript sees, and measures the word times and
  the merge on top of the diarization.
- *cpWER*: the word error rate of each speaker's text against the
  reference speaker's it maps to, under the mapping that makes it least.
  The CHiME-6 challenge's metric for transcripts with speakers. It
  measures everything at once, Whisper included.

**The protocol.** The VBx paper names three, and the scorer does all
three with two settings:

- *Full*: no collar, overlap scored. What pyannote's and NVIDIA's cards
  report, and the headline here.
- *Fair*: 0.25 s either side of each reference boundary not scored,
  overlap scored. For the Norwegian sets, whose references are
  sentence-level and include the pauses inside a sentence.
- *Forgiving*: the same collar, overlap not scored. The protocol of most
  papers before 2021, for comparing with them.

A collar trap, checked in both tools' source: `md-eval` and `dscore` take
`--collar 0.25` to mean 0.25 s either side, and `pyannote.metrics` takes
`collar=0.5` to mean the same, half on each side. The scorer takes the
either-side number and says so.

Beside DER, the Jaccard error rate (JER, DIHARD's second metric), which
weighs every speaker equally, so a speaker who says little counts as much
as one who says a lot: in a meeting, the quiet person matters.

pyannote's own pipeline is run on the same audio by a fixture script, so
that every table has its reference beside it, as every Whisper table has
transformers beside it.

**The sets.** Each licence was read off the publisher's page on
2026-10-18 (quotes and links in the session notes). None of them is ever
committed: a test or an example reads a directory it is pointed at.

| set | language | what it is | licence | what it measures here |
|---|---|---|---|---|
| AMI, test (16 meetings, about 9 h) | English | scenario meetings, 3–5 people, a room; references from `pyannote/AMI-diarization-setup` (`only_words`) | CC-BY-4.0 (corpus); setup repo Apache-2.0 | the headline offline DER against pyannote's own numbers (3.1: 18.8, community-1: 17.0, Full, headset mix); its individual headset tracks are *One track a person*'s test; its word transcripts give WDER and cpWER |
| VoxConverse, test v0.3 (232 files, 43.5 h) | mostly English | debates, news, panels from YouTube, 1–21 speakers | annotations CC-BY-4.0 "for research purposes"; the audio's copyright stays with the videos' owners | many speakers, long files (3.1: 11.3, community-1: 11.2); evaluation only, never redistributed; not for Nemotron, which trained on it |
| NB Samtale (Språkbanken, 35 conversations, 24 h) | **Norwegian** | podcasts and live events, 2–8 speakers, overlap marked (2.5% of speech), each speaker transcribed | **CC0**, audio and annotations | the Norwegian headline: DER, WDER, and cpWER with NB-Whisper |
| NPSC, test (Stortinget, 4 days, 12.3 h of speech) | **Norwegian** (Bokmål and Nynorsk) | parliament sittings, 9 to 50 speakers a day, sentence times corrected by hand | **CC0** | clustering many speakers; Nemotron's eight-speaker limit |

- **The Norwegian sets need rebuilding.** NB Samtale is distributed as
  one WAV per segment, not as whole recordings, and the Hub's NPSC as one
  MP3 per sentence. `scripts/diar-sets.py` lays each segment back at its
  time, and checks that overlapping segments carry the same samples,
  refusing any conversation where they do not. The few per cent of a
  conversation no segment covers become silence, which makes false alarms
  easier to avoid than in the original: the numbers say so wherever they
  appear.
- **NB Samtale's reference started as a machine's.** Its first-pass
  segments came from the National Library's own ASR and diarization, and
  were then corrected by hand. A system like that one may score a little
  better than it deserves. NPSC has no overlap at all, so it measures the
  clustering and not crosstalk.
- **Excluded**: DIHARD III and CALLHOME (LDC, paid), MSDWild (no
  commercial use), Ego4D (a signed agreement with disputed terms), and
  Earnings-21 (its audio's licence is not stated).
- **Published Norwegian results: none found**, for any of these models.
  The tables here would be the first.

**The scorer is written here.** DER is time counted after an optimal
one-to-one mapping, which is the Hungarian algorithm on a matrix of a few
speakers a side, with no dependency. `scripts/der-fixtures.py` scores the
same reference and hypothesis RTTM files with `pyannote.metrics` (MIT) and
writes its numbers, and a test holds the Rust scorer to them to 0.01 points
on every fixture. `md-eval.pl`, NIST's scorer, is the historical reference,
and is not needed when pyannote.metrics agrees with it.

As built (`crates/llm/src/speech/score.rs`): the time is exact, between
every boundary on either side, not a frame count at 10 ms as first planned,
because frames move each boundary by up to 5 ms, which on the 27 s dialogue
is about 0.3 points, thirty times the bound. pyannote's choices are kept
where they change a number: the scored stretch runs from the first turn of
either side to the last; speakers map by shared time, and a pair sharing
none is not mapped; JER uses DER's mapping and ignores a hypothesis speaker
with nothing to map to (DIHARD's `dscore` does neither). cpWER is held to
`meeteval` (MIT), the CHiME scorer. No package scores WDER, so the fixture
script writes Shafey et al.'s definition a second time; which words pair
depends on how an alignment breaks ties, and both take a substitution
before a deletion before an insertion. On the 15 fixtures (hand-made, the
dialogue, six random meetings with overlap, edge cases) the largest gap to
pyannote is 3·10⁻¹⁴ at all three protocols.

**Tests never download.** Every test that needs a model finds it with
`crate::weights::local_file` and skips, saying so, when it is not in the
cache, as the Whisper tests do. Datasets are the same: a test reads a
directory named by an environment variable (`KVAD_DIAR_DATA`) and skips
when it is unset. Models and datasets are fetched by hand, by the
commands the fixture scripts' headers give.

**Synthetic speech for the unit tests.** `scripts/dialogue.py` makes the
probe's recording, generalised: any number of voices from `say`, pitched
apart, with chosen gaps and chosen overlaps, and the reference RTTM and
transcript written beside it. It is not a benchmark. It is a fixture
whose right answer is known to the sample, for tests that must not depend
on a dataset being present.

### First numbers

Nemotron-3-Diarization in kvad, f32 on the CPU (M5, all cores), on the
three sets as `scripts/diar-sets.py` lays them out, scored by
`score::tests::diarization_sets` (2026-10-30). Pooled over each set's
recordings; points of reference speech; *s / h* is seconds of wall time
an hour of audio, one run each, with the machine not otherwise idle.

| set | files | hours | DER full | missed | false alarm | confusion | DER fair | DER forgiving | JER | s / h |
|---|---|---|---|---|---|---|---|---|---|---|
| AMI test | 16 | 9.1 | 26.0 | 24.0 | 1.2 | 0.7 | 24.1 | 20.6 | 28.2 | 176 |
| NB Samtale | 35 | 24.5 | 9.3 | 8.6 | 0.4 | 0.3 | 7.2 | 6.5 | 12.2 | 176 |
| NPSC test | 4 | 13.1 | 61.7 | 11.1 | 0.1 | 50.4 | 60.9 | 60.9 | 78.3 | 159 |

- **Norwegian is not the problem.** NB Samtale, 2 to 8 people in podcasts
  and live events, the meetings Kvad is for, has almost no confusion:
  0.3 points. What error there is, is speech not found, and some of that
  is the reference's: its turns are sentences, pauses included.
- **AMI is missed speech, and it is the model's.** 24 points missed, 0.7
  confused: Nemotron hears who but not all of when on the far-field
  headset mix, with `only_words` references that count every short word.
  On IS1009a transformers' own run gives the same 379 turns and the same
  26.7% to the tenth (`scripts/nemotron-fixtures.py`, the agreement test
  passing), so this is not the port. Its card's AMI number uses another
  reference, and is not this one.
- **NPSC is the eight-speaker limit, as expected.** By sitting: 21.9%
  with 9 speakers, 51.9% with 41, 71.0% with 50, 67.9% with 30, nearly
  all of it confusion. A sitting of Stortinget is not a meeting; the
  pyannote pipeline (steps 8–11) is for it.
- **The cost**: 159–176 s an hour, about 21 times real time on the CPU,
  slower than the 27 times of the short dialogue; the GPU is next.
- **pyannote 3.1 beside it** waits for its gated models' terms to be
  accepted on the Hub; `scripts/pyannote-diarize.py` runs it then, into
  the same table.
- **WDER** waits for step 4: it needs the words given to speakers, which
  is that step's merge.

## The order it is built in

Each step is a PR of its own, and none begins before the one it needs is
measured.

1. **The yardstick.** The DER, JER, WDER and cpWER scorer as a test
   helper, the Hungarian algorithm it needs, `scripts/der-fixtures.py`
   and `scripts/dialogue.py`. *Done* when the Rust scorer agrees with
   `pyannote.metrics` to 0.01 points on every fixture, at all three
   protocols. **Built**: `speech::score`; agrees to 3·10⁻¹⁴, and
   `dialogue.py` with no script remakes the probe's turns to the
   millisecond.
2. **Words with times**, on the CPU, then the GPU: the `attention`
   question, the alignment on the host, `timestamp_granularities[]=word`,
   `--words`. *Done* as in *Words with times*: within a frame of
   transformers in f32, the probe's word attribution or better on the
   dialogue, and the cost in `transcribe_speed`. **Built**: 0 frames from
   `timing.py` over transformers' attention in f32 on both backends and
   all three sizes, every matched word on the right speaker, and a pass
   that is 2–5% of a GPU run (*As built, and what it measured*).
3. **Nemotron-3-Diarization**, offline, on the CPU and the GPU.
   `kvad diarize FILE` prints RTTM, which is what the scorer reads.
   *Done* when it agrees with the transformers port in f32 (largest gap
   set from the first run), and its DER, WDER and cost per hour of audio
   are in a table on AMI test, NB Samtale and NPSC (not VoxConverse,
   which it was trained on), beside pyannote 3.1 run in Python on the
   same audio, with the first Norwegian numbers recorded here.
   **Built on the CPU** (`speech::nemotron`, `kvad diarize`): on a
   70-second, three-voice Norwegian dialogue from `scripts/dialogue.py`,
   3.8·10⁻⁶ from transformers in probability and the same 21 turns
   (`scripts/nemotron-fixtures.py`), across two compressions of the
   speaker cache; no speaker confused. On the three sets (*First numbers*,
   above): 26.0% DER on AMI, 9.3% on NB Samtale, 61.7% on NPSC, at about
   21 times real time. The GPU and pyannote's column are next.
4. **Speakers in the transcript**: the merge, `diarized_json`,
   `diarize=true` and the four formats, `num_speakers` (more than eight
   refused, saying so), the CLI. *Done* as in *In the API, the CLI and
   the web UI*, with WDER and cpWER on AMI and NB Samtale beside the
   DER. The eight-speaker limit is in the documentation.
5. **The Transcribe page**: the switch, the chips, renaming and merging.
   *Done* with a screenshot from a clean demo server, and the page's
   downloads byte for byte the server's for the same transcript.
6. **Live labels** on Nemotron's streaming mode, on the live endpoint
   (#260), measured by `live_speed --speakers`. *Done* as in *Live
   labels*.

The pyannote pipeline, for tracks, names and more than eight speakers:

7. **Reading pickled checkpoints** without running them (`weights.rs`).
   *Done* when the fixture checkpoint and segmentation-3.0's file read
   tensor for tensor as torch reads them, and a pickle that asks for any
   callable outside the three it knows is still read as data, with
   nothing called.
8. **Segmentation** on the CPU, with the speech activity it gives.
   *Done* when its seven-class output agrees with pyannote's on fixture
   chunks (largest gap set from the first run), and its speech activity
   on AMI's test set misses and falsely finds speech no more than
   pyannote's own run does, to 0.1 points. Timed per hour of audio.
9. **Embedding** on the CPU and the GPU, with the Kaldi fbank. *Done*
   when the fbank agrees with torchaudio's to 1e-4, and the embeddings
   with pyannote's to a cosine of 0.9999 in f32 on fixture chunks. Timed
   per hour of audio on both backends.
10. **Clustering and the whole offline diarization**: 3.1's
    agglomerative clustering, the constrained assignment, the
    reconstruction, behind `kvad diarize`. *Done* when DER (Full) on AMI
    test and VoxConverse test is within 0.5 points of pyannote 3.1 run
    on the same audio, and in step 3's table beside Nemotron. The
    default for files stays Nemotron unless this is better on NB
    Samtale, and is this pipeline when more than eight speakers are
    asked for.
11. **VBx**, community-1's clustering. *Done* when it is within 0.5
    points of community-1 run on the same audio, and kept over step 10's
    clustering if it beats it on the Norwegian sets.
12. **One track a person**: several files, `channels=separate`, the
    speech cut, the bleed rule. *Done* as in *One track a person*, and on
    AMI's individual headsets, cpWER beside the single-mix pipeline's on
    the same meetings.
13. **Named speakers.** `known_speaker_references[]` first, which keeps
    nothing. Saved speakers only if the owner asks for them, as *Privacy*
    lays out. The diart loop (*Live labels*) is measured beside
    Nemotron's streaming here too, now that its models exist.

## Open questions

- **The VoxCeleb question.** *Answered 2026-10-30:* the owner uses kvad
  as an open-source, non-commercial project, so VoxCeleb's research terms
  do not stand in the way of the embedding model. Kvad itself is MIT, so
  the documentation still says plainly that the speaker models carry
  their own terms, for anyone who uses kvad commercially.
- **A mirror.** Should the project publish segmentation-3.0, converted
  to safetensors, under its own name on the Hub, as MIT allows? It would
  remove the contact form and the pickle reader. It is outward-facing, so
  it is the owner's decision. *Undecided as of 2026-10-30.*
- **How many speakers is a meeting?** *Answered 2026-10-30:* the owner's
  meetings normally stay under eight people. Nemotron's limit of eight
  is acceptable, and documented where diarization is, so Nemotron is
  built first and the pyannote pipeline after it.
- **Norwegian dialects.** Neither model was trained on Norwegian. NB
  Samtale's speakers come from across the country; if the numbers there
  lag the English ones badly, fine-tuning the segmentation model (1.5 M
  parameters, and kvad trains models) on Norwegian conversations is the
  lever, and NB Samtale's CC0 licence permits it, with the conversations
  it is measured on held out. That is a plan of its own.
