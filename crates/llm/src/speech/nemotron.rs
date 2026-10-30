//! Nemotron-3-Diarization (NVIDIA, 2026): who spoke when, for up to eight
//! speakers, by one network with no clustering.
//!
//! ```text
//!   audio ─ log-mel [T × 128] ─ stack 8 ─▶ [T/8 × 1024] ─ project ─▶ [T/8 × 512]
//!                                                                      │
//!   speaker cache ++ FIFO ++ chunk ++ look-ahead ─▶ 31 blocks ─▶ [n × 512]
//!                                                                      │
//!        8 sigmoids a 10 ms frame ◀─ head ◀─ 8× upsample ◀─ project to 192
//! ```
//!
//! The encoder is the kind of transformer this crate already runs: pre-norm
//! blocks, LayerNorm, an exact-GELU MLP, eight heads with no bias on the
//! query, key and value. Its positions are rotary, and restart at 0 for
//! every chunk. Each 80 ms step is eight spectrogram frames side by side, so
//! a minute is 750 steps.
//!
//! The head says, for every 10 ms, how likely each of eight speakers is to be
//! talking, numbered **in the order they first spoke**. That is how it gets
//! away with no clustering: "speaker 0" means the first voice it heard, and
//! the training taught it to keep that numbering.
//!
//! # The speaker cache
//!
//! A long recording is read 340 steps (27.2 s) at a time, with 40 more after
//! as look-ahead. What carries the numbering from one chunk to the next is
//! what the chunk is read *with*: before it go the encoder inputs of up to
//! 264 earlier steps chosen to show each speaker so far (the cache), and the
//! 40 steps just before it (the FIFO). The network sees "speaker 0 sounded
//! like this" and labels the new chunk consistently.
//!
//! When the FIFO overflows its oldest steps move to the cache, and when the
//! cache overflows it is compressed back to 264 ([`Cache::compress`]): each
//! speaker keeps the steps where it was most clearly the one speaking, a
//! budget of 32 apiece, the rest of the cache filled with a learned
//! "silence" step. That is AOSC, the arrival-order speaker cache, and it is
//! host bookkeeping around the network: here it is transformers' code
//! (`Nemotron3DiarizationSpeakerCache`) step for step.
//!
//! Offline and streaming are the same loop with different sizes; this is the
//! offline one, as transformers runs it for a whole recording.

use super::mel::{self, Complex};
use super::whisper::{add, attend_all, Mlp, Norm};
use crate::qcache::Source;
use crate::quant::{Precision, Weight};
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The model on the Hub, and the only one in this layout so far.
pub const REPO: &str = "nvidia/Nemotron-3-Diarization";

/// Samples per spectrogram frame's transform, and per window inside it.
const N_FFT: usize = 512;
const WIN: usize = 400;
/// What `log` is kept off zero by: `2^-24`.
const LOG_GUARD: f32 = 1.0 / 16_777_216.0;
const PREEMPHASIS: f32 = 0.97;

/// The sizes `config.json` gives, and the speaker cache's policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub n_mels: usize,
    /// Spectrogram frames a step: 8.
    pub stack: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub rope_theta: f32,
    /// The head's width: 192.
    pub head_d: usize,
    pub speakers: usize,
    /// Steps a chunk scores, and the look-ahead it reads after them.
    pub chunk: usize,
    pub right: usize,
    /// Offline sizes of the FIFO, and how many steps leave it at a time.
    pub fifo: usize,
    pub update: usize,
    /// Steps the cache holds, and the silence steps each speaker is given.
    pub cache: usize,
    pub silence: usize,
    pub threshold: f32,
    pub latest_boost: f32,
    pub min_positive_rate: f32,
    pub strong_rate: f32,
    pub weak_rate: f32,
}

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Res<Config> {
        if v["model_type"].as_str() != Some("nemotron3_diarization") {
            return Err(format!("model_type is `{}`, not `nemotron3_diarization`", v["model_type"].as_str().unwrap_or("")).into());
        }
        let (a, h, s) = (&v["audio_config"], &v["head_config"], &v["streaming_config"]);
        let num = |o: &serde_json::Value, k: &str| -> Res<usize> { o[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("config.json has no `{k}`").into()) };
        let float = |o: &serde_json::Value, k: &str| -> Res<f32> { o[k].as_f64().map(|x| x as f32).ok_or_else(|| format!("config.json has no `{k}`").into()) };
        if a["hidden_act"].as_str() != Some("gelu") || num(a, "num_key_value_heads")? != num(a, "num_attention_heads")? {
            return Err("only an exact-GELU encoder with as many key heads as query heads is implemented".into());
        }
        if a["rope_parameters"]["partial_rotary_factor"].as_f64().unwrap_or(1.0) != 1.0 {
            return Err("only rotary positions over the whole head are implemented".into());
        }
        Ok(Config {
            n_mels: num(a, "num_mel_bins")?,
            stack: num(a, "subsampling_factor")?,
            d: num(a, "hidden_size")?,
            layers: num(a, "num_hidden_layers")?,
            heads: num(a, "num_attention_heads")?,
            ffn: num(a, "intermediate_size")?,
            rope_theta: float(&a["rope_parameters"], "rope_theta")?,
            head_d: num(h, "hidden_size")?,
            speakers: num(h, "num_speakers")?,
            chunk: num(v, "chunk_length")?,
            right: num(v, "chunk_right_context")?,
            fifo: num(v, "fifo_length")?,
            update: num(v, "speaker_cache_update_period")?,
            cache: num(s, "speaker_cache_length")?,
            silence: num(s, "speaker_cache_silence_frames_per_speaker")?,
            threshold: float(s, "prediction_score_threshold")?,
            latest_boost: float(s, "latest_frames_score_boost")?,
            min_positive_rate: float(s, "min_positive_scores_rate")?,
            strong_rate: float(s, "strong_boost_rate")?,
            weak_rate: float(s, "weak_boost_rate")?,
        })
    }
}

/// The log-mel spectrogram of 16 kHz mono `audio`, `[frames][n_mels]`, and how
/// many of its frames are the audio's.
///
/// Not Whisper's ([`mel::log_mel`]), though built of the same parts. The
/// samples are first sharpened, `x[t] − 0.97 x[t−1]`, which lifts the high
/// frequencies speech carries its consonants in. Each frame is 400 samples
/// under a *symmetric* Hann window, centred in a 512-sample transform; the
/// recording is padded with zeros rather than mirrored; and the bands are a
/// natural log, `+ 2^-24`, with no floor and no rescaling.
///
/// Centring gives `len / 160 + 1` frames. The last reaches past the audio by
/// more than the reference counts, so it is zeroed, and reported as not the
/// audio's: the model reads it, and nothing it says there is kept.
pub fn features(audio: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
    let hop = mel::HOP;
    let frames = audio.len() / hop + 1;
    let valid = audio.len() / hop;
    let sharp: Vec<f32> = (0..audio.len()).map(|t| if t == 0 { audio[0] } else { audio[t] - PREEMPHASIS * audio[t - 1] }).collect();
    // torch.hann_window(400, periodic=False), placed in the middle of 512.
    let lead = (N_FFT - WIN) / 2;
    let window: Vec<f32> = (0..WIN).map(|t| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * t as f64 / (WIN - 1) as f64).cos()) as f32).collect();
    let bank = mel::filterbank_for(n_mels, N_FFT);
    let mut out = vec![0.0f32; frames * n_mels];
    out.par_chunks_mut(n_mels).enumerate().take(valid).for_each(|(t, row)| {
        let start = (t * hop) as isize - (N_FFT / 2) as isize;
        let frame: Vec<Complex> = (0..N_FFT)
            .map(|i| {
                let at = start + i as isize;
                let x = if i < lead || i >= lead + WIN || at < 0 || at as usize >= sharp.len() { 0.0 } else { sharp[at as usize] * window[i - lead] };
                Complex { re: x, im: 0.0 }
            })
            .collect();
        let spectrum = mel::fft(&frame);
        let power: Vec<f32> = spectrum[..N_FFT / 2 + 1].iter().map(|c| c.re * c.re + c.im * c.im).collect();
        for (o, band) in row.iter_mut().zip(&bank) {
            *o = (band.iter().zip(&power).map(|(w, p)| w * p).sum::<f32>() + LOG_GUARD).ln();
        }
    });
    (out, valid)
}

/// One encoder block's weights: Whisper's, with no biases on the query, key
/// and value, and rotary positions.
struct Block {
    ln1: Norm,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    o_b: Vec<f32>,
    ln2: Norm,
    mlp: Mlp,
}

pub struct Nemotron {
    pub config: Config,
    /// `[d, n_mels × stack]`: eight frames side by side, to one step.
    embed: Weight,
    ln_in: Norm,
    blocks: Vec<Block>,
    ln_out: Norm,
    proj: Weight,
    proj_b: Vec<f32>,
    /// `[head_d × stack, head_d × 3]`: a convolution over three steps whose
    /// output is eight 10 ms frames for each, side by side.
    up: Weight,
    up_b: Vec<f32>,
    dense: Weight,
    dense_b: Vec<f32>,
    out: Weight,
    out_b: Vec<f32>,
    /// The step a compressed cache's spare places hold.
    silence: Vec<f32>,
    /// `cos` and `sin` of each position's angle for each rotated pair, built
    /// once for the longest input a chunk can be.
    rope: (Vec<f32>, Vec<f32>),
    pool: rayon::ThreadPool,
    precision: Precision,
}

impl Nemotron {
    /// The model from the cache, or the Hub: its two files, not the `.nemo`
    /// or the GGUF beside them.
    pub fn fetch(repo: &str, precision: Precision, watch: &crate::weights::Watcher) -> Res<Nemotron> {
        let config = crate::weights::fetch_file(repo, "config.json", watch)?;
        let weights = crate::weights::fetch_file(repo, "model.safetensors", watch)?;
        let config = Config::from_json(&crate::weights::read_json(&config)?).map_err(|e| format!("{repo}: {e}"))?;
        Nemotron::load(config, &[weights], precision)
    }

    pub fn load(config: Config, files: &[PathBuf], precision: Precision) -> Res<Nemotron> {
        let ckpt = crate::weights::Checkpoint::open(files)?;
        let src = crate::qcache::Live::new(&ckpt, precision);
        let model = Nemotron::from_source(config, &src, precision)?;
        let left = src.unread();
        if !left.is_empty() {
            return Err(format!("this checkpoint holds tensors the Nemotron loader never reads: {}", crate::weights::collapsed(&left).join(", ")).into());
        }
        Ok(model)
    }

    fn from_source(config: Config, src: &dyn Source, precision: Precision) -> Res<Nemotron> {
        let c = &config;
        let t = "model.audio_tower";
        let blocks = (0..c.layers)
            .map(|i| {
                let p = format!("{t}.layers.{i}");
                let w = |s: &str| src.matrix(&format!("{p}.self_attn.{s}.weight"));
                Ok(Block {
                    ln1: Norm::load(src, &format!("{p}.layer_norm1"))?,
                    q: w("q_proj")?,
                    k: w("k_proj")?,
                    v: w("v_proj")?,
                    o: w("o_proj")?,
                    o_b: src.vector(&format!("{p}.self_attn.o_proj.bias"))?,
                    ln2: Norm::load(src, &format!("{p}.layer_norm2"))?,
                    mlp: Mlp::load(src, &format!("{p}.mlp"))?,
                })
            })
            .collect::<Res<Vec<_>>>()?;
        // PyTorch stores a Conv1d as [out, in, k], which read flat is the
        // [out, in × k] matrix im2col wants (see `whisper::Conv`).
        let up = src.vector("model.upsampler.conv.weight")?;
        if up.len() != c.head_d * c.stack * c.head_d * 3 {
            return Err("the upsampler is not the shape config.json gives".into());
        }
        let longest = c.cache + c.fifo + c.chunk + c.right;
        Ok(Nemotron {
            embed: src.matrix(&format!("{t}.embedder.projection.weight"))?,
            ln_in: Norm::load(src, &format!("{t}.input_layer_norm"))?,
            blocks,
            ln_out: Norm::load(src, &format!("{t}.layer_norm"))?,
            proj: src.matrix("model.proj.weight")?,
            proj_b: src.vector("model.proj.bias")?,
            up: Weight::from_f32(Tensor::new(c.head_d * c.stack, c.head_d * 3, up)),
            up_b: src.vector("model.upsampler.conv.bias")?,
            dense: src.matrix("classifier.dense.weight")?,
            dense_b: src.vector("classifier.dense.bias")?,
            out: src.matrix("classifier.out_proj.weight")?,
            out_b: src.vector("classifier.out_proj.bias")?,
            silence: src.vector("silence_embeds")?,
            rope: rope(c.d / c.heads, c.rope_theta, longest),
            pool: crate::model::pool(),
            precision,
            config,
        })
    }

    pub fn params(&self) -> usize {
        let b = |b: &Block| b.q.param_count() + b.k.param_count() + b.v.param_count() + b.o.param_count() + b.mlp.params();
        self.embed.param_count() + self.blocks.iter().map(b).sum::<usize>() + self.proj.param_count() + self.up.param_count() + self.dense.param_count() + self.out.param_count()
    }

    pub fn summary(&self) -> String {
        let c = &self.config;
        format!("nemotron-3-diarization: {}M params, {} blocks, d={}, up to {} speakers, cpu {}", self.params() / 1_000_000, c.layers, c.d, c.speakers, self.precision)
    }

    /// The spectrogram's steps: each run of `stack` frames (zeros after the
    /// last), as one row, projected to the encoder's width. `[steps][d]`.
    pub fn embed(&self, features: &[f32]) -> Vec<f32> {
        let row = self.config.n_mels * self.config.stack;
        let steps = features.len().div_ceil(row);
        let mut stacked = features.to_vec();
        stacked.resize(steps * row, 0.0);
        self.embed.matmul_bt(&stacked, steps, None)
    }

    /// The encoder over `x`, `[n][d]`, positions from 0, its keys the first
    /// `keys` rows: the rest are past the recording's end.
    pub fn encode(&self, x: &[f32], keys: usize) -> Vec<f32> {
        let c = &self.config;
        let (n, hd) = (x.len() / c.d, c.d / c.heads);
        let mut x = self.ln_in.rows(x);
        let scale = 1.0 / (hd as f32).sqrt();
        for b in &self.blocks {
            let h = b.ln1.rows(&x);
            let mut q = b.q.matmul_bt(&h, n, None);
            let mut k = b.k.matmul_bt(&h, n, None);
            let v = b.v.matmul_bt(&h, n, None);
            self.rotate(&mut q, hd);
            self.rotate(&mut k, hd);
            q.iter_mut().for_each(|v| *v *= scale);
            let a = attend_all(&q, &k[..keys * c.d], &v[..keys * c.d], n, keys, c.heads, hd);
            add(&mut x, &b.o.matmul_bt(&a, n, Some(&b.o_b)));
            let h = b.ln2.rows(&x);
            add(&mut x, &b.mlp.forward(&h, n));
        }
        self.ln_out.rows(&x)
    }

    /// Rotary positions, row `i` at position `i`: in each head, dimension
    /// `j` and `j + hd/2` turned together by `i θ_j` (transformers'
    /// `rotate_half`, not the interleaved pairs of the original paper).
    fn rotate(&self, x: &mut [f32], hd: usize) {
        let (cos, sin) = &self.rope;
        let half = hd / 2;
        x.par_chunks_mut(self.config.d).enumerate().for_each(|(i, row)| {
            let (cos, sin) = (&cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
            for head in row.chunks_mut(hd) {
                for j in 0..half {
                    let (a, b) = (head[j], head[j + half]);
                    head[j] = a * cos[j] - b * sin[j];
                    head[j + half] = b * cos[j] + a * sin[j];
                }
            }
        });
    }

    /// The head over the encoder's output: each speaker's score for every
    /// 10 ms frame, `[n × stack][speakers]`, before the sigmoid.
    ///
    /// Down to 192 wide, then a convolution over each step and its two
    /// neighbours that makes `stack` × 192 numbers a step: eight frames'
    /// worth, laid side by side and read as eight rows. Then ReLU, a square
    /// layer, ReLU, and one score per speaker.
    pub fn head(&self, h: &[f32]) -> Vec<f32> {
        let c = &self.config;
        let n = h.len() / c.d;
        let p = self.proj.matmul_bt(h, n, Some(&self.proj_b));
        let w = c.head_d;
        let mut cols = vec![0.0f32; n * w * 3];
        cols.par_chunks_mut(w * 3).enumerate().for_each(|(t, row)| {
            for k in 0..3 {
                let at = t as isize + k as isize - 1;
                if at < 0 || at as usize >= n {
                    continue;
                }
                for (i, &v) in p[at as usize * w..(at as usize + 1) * w].iter().enumerate() {
                    row[i * 3 + k] = v;
                }
            }
        });
        let mut x = self.up.matmul_bt(&cols, n, Some(&self.up_b));
        let frames = n * c.stack;
        relu(&mut x);
        let mut x = self.dense.matmul_bt(&x, frames, Some(&self.dense_b));
        relu(&mut x);
        self.out.matmul_bt(&x, frames, Some(&self.out_b))
    }

    /// Each speaker's probability in every 10 ms frame of `audio`,
    /// `[frames][speakers]`: the recording read a chunk at a time, each with
    /// the speaker cache and FIFO before it.
    pub fn diarize(&self, audio: &[f32], progress: &mut (dyn FnMut(f32, f32) + Send)) -> Vec<f32> {
        self.pool.install(|| self.offline(audio, progress))
    }

    fn offline(&self, audio: &[f32], progress: &mut (dyn FnMut(f32, f32) + Send)) -> Vec<f32> {
        let c = &self.config;
        let (features, valid) = features(audio, c.n_mels);
        let embeds = self.embed(&features);
        drop(features);
        let steps = embeds.len() / c.d;
        // A step is the recording's if its first frame is.
        let real = valid.div_ceil(c.stack);
        let mut cache = Cache::new(c);
        let mut probs = Vec::with_capacity(steps * c.stack * c.speakers);
        for start in (0..steps).step_by(c.chunk) {
            let end = (start + c.chunk).min(steps);
            let reach = (end + c.right).min(steps);
            let before = cache.len();
            let mut input = Vec::with_capacity((before + reach - start) * c.d);
            input.extend_from_slice(&cache.embeds);
            input.extend_from_slice(&cache.fifo);
            input.extend_from_slice(&embeds[start * c.d..reach * c.d]);
            let n = input.len() / c.d;
            let keys = n - (reach - real.min(reach));
            let logits = self.head(&self.encode(&input, keys));
            cache.update(&self.silence, &input, &pool(&logits, c, keys), end - start);
            let out = &logits[before * c.stack * c.speakers..(before + end - start) * c.stack * c.speakers];
            probs.extend(out.iter().map(|&x| sigmoid(x)));
            progress((end * c.stack) as f32 / 100.0, (steps * c.stack) as f32 / 100.0);
        }
        probs.truncate(valid * c.speakers);
        probs
    }
}

fn relu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `cos` and `sin` of `i θ_j` for positions `0..n`, `θ_j = theta^(-2j / hd)`,
/// rounded as transformers rounds them: the angle is an f32 product.
fn rope(hd: usize, theta: f32, n: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half).map(|j| 1.0 / theta.powf((2 * j) as f32 / hd as f32)).collect();
    let angles: Vec<f32> = (0..n).flat_map(|i| inv.iter().map(move |f| i as f32 * f)).collect();
    (angles.iter().map(|a| (*a as f64).cos() as f32).collect(), angles.iter().map(|a| (*a as f64).sin() as f32).collect())
}

/// Each step's speaker probabilities, `[n][speakers]`: the mean of its
/// frames' sigmoids, and zero for the steps past the recording's end.
fn pool(logits: &[f32], c: &Config, keys: usize) -> Vec<f32> {
    let s = c.speakers;
    let n = logits.len() / (c.stack * s);
    let mut out = vec![0.0f32; n * s];
    for (t, row) in out.chunks_mut(s).enumerate().take(keys) {
        for f in 0..c.stack {
            for (k, o) in row.iter_mut().enumerate() {
                *o += sigmoid(logits[(t * c.stack + f) * s + k]);
            }
        }
        row.iter_mut().for_each(|o| *o /= c.stack as f32);
    }
    out
}

/// The arrival-order speaker cache and the FIFO: encoder inputs, `[n][d]`
/// each, read before every chunk.
struct Cache {
    d: usize,
    speakers: usize,
    embeds: Vec<f32>,
    /// The speaker probabilities the cache's steps were kept for.
    probs: Vec<f32>,
    fifo: Vec<f32>,
    compressed: bool,
    fifo_len: usize,
    update: usize,
    cache_len: usize,
    silence: usize,
    threshold: f32,
    latest_boost: f32,
    min_positive: usize,
    strong: usize,
    weak: usize,
}

impl Cache {
    fn new(c: &Config) -> Cache {
        let budget = (c.cache / c.speakers - c.silence) as f32;
        Cache {
            d: c.d,
            speakers: c.speakers,
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
            fifo_len: c.fifo,
            update: c.update,
            cache_len: c.cache,
            silence: c.silence,
            threshold: c.threshold,
            latest_boost: c.latest_boost,
            min_positive: (budget * c.min_positive_rate).floor() as usize,
            strong: (budget * c.strong_rate).floor() as usize,
            weak: (budget * c.weak_rate).floor() as usize,
        }
    }

    /// Steps read before a chunk: the cache's, then the FIFO's.
    fn len(&self) -> usize {
        (self.embeds.len() + self.fifo.len()) / self.d
    }

    /// After a chunk: its `chunk` steps join the FIFO, and when that holds
    /// more than it may, its oldest move to the cache, at least `update` of
    /// them. `input` is what the chunk was read with, `probs` its steps'
    /// speaker probabilities.
    fn update(&mut self, silence: &[f32], input: &[f32], probs: &[f32], chunk: usize) {
        let (d, s) = (self.d, self.speakers);
        let (cached, queued) = (self.embeds.len() / d, self.fifo.len() / d);
        let start = cached + queued;
        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(&input[start * d..(start + chunk) * d]);
        let total = queued + chunk;
        if total > self.fifo_len {
            let popped = self.update.max(total - self.fifo_len).min(total);
            let mut fifo_probs = probs[cached * s..(cached + queued) * s].to_vec();
            fifo_probs.extend_from_slice(&probs[start * s..(start + chunk) * s]);
            // Until the cache is first compressed it is plain earlier steps,
            // which this chunk's run has scored again; after, it is a
            // selection, and the scores it was chosen by are the ones kept.
            let mut kept = if self.compressed { std::mem::take(&mut self.probs) } else { probs[..cached * s].to_vec() };
            let mut embeds = std::mem::take(&mut self.embeds);
            embeds.extend_from_slice(&fifo[..popped * d]);
            kept.extend_from_slice(&fifo_probs[..popped * s]);
            fifo.drain(..popped * d);
            if embeds.len() / d > self.cache_len {
                (embeds, kept) = self.compress(&embeds, &kept, silence);
                self.compressed = true;
            }
            self.embeds = embeds;
            self.probs = kept;
        }
        self.fifo = fifo;
    }

    /// How much each step is worth keeping for each speaker, `[n][speakers]`:
    /// the log-odds that this speaker alone is speaking, if it is speaking at
    /// all (probability over a half), else nothing. Once a speaker has
    /// enough clearly-theirs steps, its unclear ones are dropped too.
    fn scores(&self, probs: &[f32]) -> Vec<f32> {
        let s = self.speakers;
        let th = self.threshold;
        let half = 0.5f32.ln();
        let mut scores: Vec<f32> = probs
            .chunks(s)
            .flat_map(|row| {
                let alone: f32 = row.iter().map(|p| (1.0 - p).max(th).ln()).sum();
                row.iter().map(move |&p| match p > 0.5 {
                    true => p.max(th).ln() - (1.0 - p).max(th).ln() + alone - half,
                    false => f32::NEG_INFINITY,
                })
            })
            .collect();
        for k in 0..s {
            let positive = scores.iter().skip(k).step_by(s).filter(|&&v| v > 0.0).count();
            if positive >= self.min_positive {
                scores.iter_mut().skip(k).step_by(s).filter(|v| **v <= 0.0).for_each(|v| *v = f32::NEG_INFINITY);
            }
        }
        scores
    }

    /// The cache cut back to `cache_len` steps: for each speaker, in order of
    /// arrival, its best-scored steps in time order and a silence step, then
    /// silence for the places nobody earned.
    ///
    /// Before the choosing, the steps just arrived from the FIFO get a
    /// little more, and each speaker's best 24 and best 48 a lot more
    /// (`2 ln 2`, then `ln 2`), so that every speaker heard keeps some steps
    /// however loud the others are.
    fn compress(&self, embeds: &[f32], probs: &[f32], silence: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (d, s) = (self.d, self.speakers);
        let n = embeds.len() / d;
        let mut scores = self.scores(probs);
        scores[self.cache_len * s..].iter_mut().for_each(|v| *v += self.latest_boost);
        boost(&mut scores, s, self.strong, -2.0 * 0.5f32.ln());
        boost(&mut scores, s, self.weak, -0.5f32.ln());

        // Speaker-major, each speaker's steps then its silence places, which
        // always win.
        let m = n + self.silence;
        let score = |i: usize| if i % m >= n { f32::INFINITY } else { scores[(i % m) * s + i / m] };
        let mut order: Vec<usize> = (0..m * s).collect();
        order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
        let sentinel = m * s;
        let mut chosen: Vec<usize> = order[..self.cache_len].iter().map(|&i| if score(i) == f32::NEG_INFINITY { sentinel } else { i }).collect();
        chosen.sort_unstable();

        let mut out_e = Vec::with_capacity(self.cache_len * d);
        let mut out_p = Vec::with_capacity(self.cache_len * s);
        for i in chosen {
            let step = if i == sentinel { n } else { (i % m).min(n) };
            match step < n {
                true => {
                    out_e.extend_from_slice(&embeds[step * d..(step + 1) * d]);
                    out_p.extend_from_slice(&probs[step * s..(step + 1) * s]);
                }
                false => {
                    out_e.extend_from_slice(silence);
                    out_p.extend(std::iter::repeat_n(0.0, s));
                }
            }
        }
        (out_e, out_p)
    }
}

/// Add `by` to each speaker's `count` best scores.
fn boost(scores: &mut [f32], s: usize, count: usize, by: f32) {
    let n = scores.len() / s;
    for k in 0..s {
        let mut steps: Vec<usize> = (0..n).collect();
        steps.sort_by(|&a, &b| scores[b * s + k].total_cmp(&scores[a * s + k]).then(a.cmp(&b)));
        for &t in steps.iter().take(count) {
            scores[t * s + k] += by;
        }
    }
}

/// Who spoke when, from [`Nemotron::diarize`]'s probabilities: each run of
/// frames over `threshold` for a speaker is one turn, `speaker_N` numbered
/// by arrival. Turns of different speakers may overlap.
pub fn turns(probs: &[f32], speakers: usize, threshold: f32) -> Vec<super::score::Turn> {
    let frames = probs.len() / speakers;
    let at = |f: usize| f as f64 * mel::HOP as f64 / mel::SAMPLE_RATE as f64;
    let mut turns = Vec::new();
    for k in 0..speakers {
        let mut began = None;
        for f in 0..=frames {
            let on = f < frames && probs[f * speakers + k] > threshold;
            match (on, began) {
                (true, None) => began = Some(f),
                (false, Some(b)) => {
                    turns.push(super::score::Turn { speaker: format!("speaker_{k}"), start: at(b), end: at(f) });
                    began = None;
                }
                _ => {}
            }
        }
    }
    turns.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against `scripts/nemotron-fixtures.py`: transformers' own numbers for
    /// one WAV, stage by stage, on the CPU at f32.
    ///
    /// ```text
    /// KVAD_NEMOTRON_FIXTURES=/tmp/nemotron-fx cargo test --release -p kvad speech::nemotron -- --ignored --nocapture
    /// ```
    ///
    /// The model is read from the cache only; the script will have put it
    /// there.
    #[test]
    #[ignore]
    fn nemotron_agrees_with_transformers() {
        let dir = PathBuf::from(std::env::var("KVAD_NEMOTRON_FIXTURES").expect("KVAD_NEMOTRON_FIXTURES is not set"));
        let theirs = crate::weights::Checkpoint::open(&[dir.join("nemotron.safetensors")]).unwrap();
        let get = |name: &str| theirs.get(name).unwrap().data;
        let meta = crate::weights::read_json(&dir.join("meta.json")).unwrap();
        let audio = super::super::audio::from_wav(&std::fs::read(dir.join("audio.wav")).unwrap()).unwrap();
        let file = |f: &str| crate::weights::local_file(REPO, f).expect("the model is not in the cache: run the script first");
        let config = Config::from_json(&crate::weights::read_json(&file("config.json")).unwrap()).unwrap();
        let model = Nemotron::load(config, &[file("model.safetensors")], Precision::F32).unwrap();
        let c = &model.config;

        // The largest difference, and the largest number it is measured against.
        let gap = |ours: &[f32], theirs: &[f32]| -> (f32, f32) {
            assert_eq!(ours.len(), theirs.len());
            let big = theirs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            (ours.iter().zip(theirs).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())), big)
        };

        let (features, valid) = features(&audio, c.n_mels);
        assert_eq!(valid, meta["frames"].as_u64().unwrap() as usize);
        let f = gap(&features, &get("features"));
        eprintln!("features: {:.2e} of {:.1}", f.0, f.1);
        let embeds = model.embed(&features);
        let e = gap(&embeds, &get("embeds"));
        eprintln!("embeds:   {:.2e} of {:.1}", e.0, e.1);
        let first = &get("embeds")[..(c.chunk + c.right) * c.d];
        let encoded = model.pool.install(|| model.encode(first, c.chunk + c.right));
        let h = gap(&encoded, &get("encoded"));
        eprintln!("encoded:  {:.2e} of {:.1}", h.0, h.1);
        let probs = model.diarize(&audio, &mut |_, _| {});
        let p = gap(&probs, &get("probs")[..valid * c.speakers]);
        eprintln!("probs:    {:.2e}", p.0);

        // Turns, against transformers' `extract_speaker_dict`.
        let ours = turns(&probs, c.speakers, 0.5);
        let theirs: Vec<(f64, f64, String)> = meta["segments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s["Start"].as_f64().unwrap(), s["End"].as_f64().unwrap(), format!("speaker_{}", s["Speaker"])))
            .collect();
        eprintln!("turns:    {} of theirs {}", ours.len(), theirs.len());

        assert!(f.0 < 1e-3 * f.1.max(1.0), "the spectrogram is not transformers'");
        assert!(e.0 < 1e-3 * e.1, "the steps are not transformers'");
        // The first run was 1.1e-6 of the encoder's largest and 3.8e-6 in
        // probability.
        assert!(h.0 < 1e-4 * h.1, "the encoder is not transformers'");
        assert!(p.0 < 1e-4, "the probabilities are not transformers'");
        assert_eq!(ours.len(), theirs.len(), "a different number of turns");
        for (o, t) in ours.iter().zip(&theirs) {
            assert_eq!(o.speaker, t.2);
            assert!((o.start - t.0).abs() < 0.015 && (o.end - t.1).abs() < 0.015, "{o:?} against {t:?}");
        }
    }

    /// The cache's bookkeeping on its own: it overflows into a compression
    /// that keeps each speaker's clearest steps, grouped by speaker, padded
    /// with silence.
    #[test]
    fn compression_keeps_each_speakers_clearest_steps() {
        let c = Config {
            n_mels: 0,
            stack: 8,
            d: 1,
            layers: 0,
            heads: 1,
            ffn: 0,
            rope_theta: 1.0,
            head_d: 0,
            speakers: 2,
            chunk: 0,
            right: 0,
            fifo: 0,
            update: 0,
            cache: 6,
            silence: 1,
            threshold: 0.25,
            latest_boost: 0.05,
            min_positive_rate: 0.5,
            strong_rate: 0.75,
            weak_rate: 1.5,
        };
        let cache = Cache::new(&c);
        // Ten steps, each embed its own index; speaker 0 clear in 0..4,
        // speaker 1 clear in 6..9, nobody in 4 and 5.
        let embeds: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let probs: Vec<f32> = (0..10).flat_map(|i| if i < 4 { [0.95, 0.05] } else if i >= 6 { [0.05, 0.9] } else { [0.1, 0.1] }).collect();
        let (e, p) = cache.compress(&embeds, &probs, &[-1.0]);
        assert_eq!(e.len(), 6);
        // Grouped by speaker in time order, each group ending with silence.
        let zero: Vec<f32> = e.iter().take_while(|&&v| v >= 0.0).copied().collect();
        assert!(zero.iter().all(|&v| v < 4.0) && zero.windows(2).all(|w| w[0] < w[1]), "{e:?}");
        let one: Vec<f32> = e[zero.len() + 1..].iter().take_while(|&&v| v >= 0.0).copied().collect();
        assert!(!one.is_empty() && one.iter().all(|&v| v >= 6.0), "{e:?}");
        assert_eq!(p[zero.len() * 2..zero.len() * 2 + 2], [0.0, 0.0]);
    }

    #[test]
    fn turns_are_runs_over_the_threshold() {
        let probs = [0.9, 0.1, 0.9, 0.6, 0.2, 0.7, 0.2, 0.2];
        let t = turns(&probs, 2, 0.5);
        let t: Vec<(String, f64, f64)> = t.into_iter().map(|t| (t.speaker, t.start, t.end)).collect();
        assert_eq!(t, vec![("speaker_0".into(), 0.0, 0.02), ("speaker_1".into(), 0.01, 0.03)]);
    }
}
