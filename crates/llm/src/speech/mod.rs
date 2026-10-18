//! Speech to text: Whisper, on the CPU, written out, and the loop that
//! drives kvad-gpu's Whisper on Metal too.
//!
//! The one encoder-decoder in this crate, and a peer of the text path rather
//! than one of its architectures: it is not a [`Transformer`]
//! (crate::model::Transformer) in the registry, because nothing about a
//! chat session — a prompt in, tokens out, a conversation's cache kept
//! between turns — is what it does. It takes a recording and gives back its
//! transcript, and what it holds between calls is its weights.
//!
//! Read in this order:
//!
//! 1. [`audio`] — a file to 16 kHz mono samples: WAV by hand, anything else
//!    by `ffmpeg`, and a resampler.
//! 2. [`mel`] — samples to the log-mel spectrogram the encoder reads, with
//!    the FFT it takes.
//! 3. [`whisper`] — the model: an encoder with no mask, a decoder with
//!    cross-attention, and the two caches that are not one.
//! 4. [`transcribe`] — the decoding loop: the prompt, the rules about what
//!    may be said, the walk through a long recording, and the formats a
//!    transcript is written out in.
//! 5. [`live`] — the same loop over audio that is still arriving: a window
//!    decoded again as it grows, and words settled once two runs agree.
//! 6. [`align`] — when each word was said, from where the decoder's
//!    cross-attention looks.
//! 7. [`score`] — how wrong a "who spoke when" is: the diarization error
//!    rates, and the Hungarian algorithm they map speakers with.
//!
//! Every Whisper on the Hub in transformers' layout loads: OpenAI's own
//! (`openai/whisper-*`), the distilled ones, and fine-tunes of them such as
//! the National Library of Norway's NB-Whisper (`NbAiLab/nb-whisper-*`).
//!
//! The loop drives a [`Model`], and [`whisper::Cpu`] is one. kvad-gpu has
//! the other, on Metal (`kvad_gpu::speech`): the spectrogram, the tokens,
//! the rules and the walk through a recording are the same arithmetic on the
//! host whichever answers, so they are here once.

pub mod align;
pub mod audio;
pub mod live;
pub mod mel;
pub mod reference;
pub mod score;
pub mod transcribe;
pub mod whisper;

pub use live::{Live, Update};
pub use transcribe::{Format, Progress, Request, Segment, Task, Transcriber, Transcript};

use crate::quant::Precision;
use crate::weights::{fetch_file, Watcher};
use std::any::Any;
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A Whisper, as the decoding loop sees one: a window of spectrogram in,
/// next-token scores out.
///
/// Four questions and no more, because that is all the loop asks: what
/// the decoder will read of this window ([`Model::hear`]), an empty text
/// cache to begin a pass with ([`Model::said`]), the scores after some
/// tokens ([`Model::decode`]), and, for words with times, where some of the
/// decoder's heads looked ([`Model::attention`]). What each keeps between calls is its own
/// business: host vectors on the CPU, tensors on a GPU. So the two caches
/// cross this boundary as [`Heard`] and [`Said`], which hold whatever the
/// model put in them and which only it opens.
///
/// `decode` answers in scores rather than hidden states, at the rows
/// asked for. A GPU's hidden state would otherwise come to the host only to
/// go back for the output head, a second wait a token for nothing.
pub trait Model: Send + Sync {
    fn config(&self) -> &whisper::Config;

    /// One window of spectrogram, `[3000][n_mels]` time-major as
    /// [`mel::window`] lays it out, through the encoder and every decoder
    /// block's cross-attention keys and values.
    fn hear(&self, window: &[f32]) -> Res<Heard>;

    /// A text cache with nothing in it.
    fn said(&self) -> Said;

    /// Run `tokens` through the decoder after what `said` holds, keeping
    /// them there, and return the scores for the next token after each row
    /// of `tokens` that `rows` names, in that order.
    fn decode(&self, tokens: &[u32], said: &mut Said, heard: &Heard, rows: &[usize]) -> Res<Vec<Vec<f32>>>;

    /// The cross-attention weights of `heads`, each `(layer, head)`, for
    /// `tokens` decoded from an empty cache against `heard`:
    /// `[head][token][frame]`, softmaxed, in f32, over all of a window's
    /// frames. What [`align`] times words by. No scores: nothing after the
    /// last layer with one of `heads` need run.
    fn attention(&self, tokens: &[u32], heard: &Heard, heads: &[(usize, usize)]) -> Res<Vec<f32>>;

    /// Run `work` where this model's work runs best: the CPU's on its own
    /// threads (see [`transcribe::Transcriber`]), a GPU's as it comes.
    fn within(&self, work: &mut (dyn FnMut() + Send));

    /// Weights in the matrices, convolutions and position tables; norms and
    /// biases left out, as [`whisper::Config::params`] leaves them.
    fn params(&self) -> usize;
    /// What those take where they are held, counted the same way.
    fn weight_bytes(&self) -> usize;
    /// Where it runs, as a language model's session says it: `cpu f32`,
    /// `metal q8`.
    fn backend(&self) -> String;
}

/// A model shared: what a test holds to look inside one while a
/// [`Transcriber`] drives it.
impl<M: Model> Model for std::sync::Arc<M> {
    fn config(&self) -> &whisper::Config {
        (**self).config()
    }
    fn hear(&self, window: &[f32]) -> Res<Heard> {
        (**self).hear(window)
    }
    fn said(&self) -> Said {
        (**self).said()
    }
    fn decode(&self, tokens: &[u32], said: &mut Said, heard: &Heard, rows: &[usize]) -> Res<Vec<Vec<f32>>> {
        (**self).decode(tokens, said, heard, rows)
    }
    fn attention(&self, tokens: &[u32], heard: &Heard, heads: &[(usize, usize)]) -> Res<Vec<f32>> {
        (**self).attention(tokens, heard, heads)
    }
    fn within(&self, work: &mut (dyn FnMut() + Send)) {
        (**self).within(work)
    }
    fn params(&self) -> usize {
        (**self).params()
    }
    fn weight_bytes(&self) -> usize {
        (**self).weight_bytes()
    }
    fn backend(&self) -> String {
        (**self).backend()
    }
}

/// What a [`Model`] made of a window for its decoder to read: the
/// cross-attention's keys and values. Opened by that model only.
pub struct Heard(pub Box<dyn Any + Send + Sync>);

/// A [`Model`]'s decoder cache for one pass over one window. Opened by that
/// model only.
pub struct Said(pub Box<dyn Any + Send + Sync>);

impl Heard {
    /// What `T` this holds, or an error saying it is another model's.
    pub fn get<T: 'static>(&self) -> Res<&T> {
        self.0.downcast_ref().ok_or_else(|| "a window heard by another model".into())
    }
}

impl Said {
    pub fn get_mut<T: 'static>(&mut self) -> Res<&mut T> {
        self.0.downcast_mut().ok_or_else(|| "a text cache from another model".into())
    }
}

/// What a Whisper is called where an image model's diffusers pipeline
/// would be: transformers' class for it. A server sorts the models on its
/// disk by that name, and this one has no `model_index.json` to give it one,
/// so it is known by its `config.json`'s `model_type` instead
/// ([`crate::hub::pipeline`]).
pub const PIPELINE: &str = "WhisperForConditionalGeneration";

/// What `repo`'s weights will take once loaded at `precision`, from its
/// `config.json` in the cache: `None` for anything that is not a Whisper
/// there. Asked before the load, to decide whether it fits.
pub fn weight_bytes(repo: &str, precision: Precision) -> Option<u64> {
    let (matrices, floats) = local_config(repo)?.params();
    Some(precision.weight_bytes(matrices as u64) + 4 * floats as u64)
}

/// `repo`'s shape, from its `config.json` in the cache: `None` for anything
/// that is not a Whisper there.
pub fn local_config(repo: &str) -> Option<whisper::Config> {
    let config = crate::weights::local_file(repo, "config.json")?;
    whisper::Config::from_json(&crate::weights::read_json(&config).ok()?).ok()
}

/// Whether `repo` is a Whisper, asking the disk only: a directory, or a repo
/// whose `config.json` is in the Hub cache.
pub fn is_whisper(repo: &str) -> bool {
    crate::weights::local_file(repo, "config.json").is_some_and(|c| whisper::Config::is_whisper(&c))
}

/// The files a Whisper is loaded from: its config, tokenizer and weights,
/// and the generation config where the repo has one.
pub struct Files {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub generation: Option<PathBuf>,
    pub weights: Vec<PathBuf>,
}

impl Files {
    /// The files of `repo`, from the cache where they are there and the Hub
    /// where they are not.
    pub fn fetch(repo: &str, watch: &Watcher) -> Res<Files> {
        let config = fetch_file(repo, "config.json", watch)?;
        let tokenizer = fetch_file(repo, "tokenizer.json", watch)?;
        let generation = fetch_file(repo, "generation_config.json", watch).ok();
        let weights = match fetch_file(repo, "model.safetensors.index.json", watch) {
            Ok(index) => {
                let map = crate::weights::read_json(&index)?;
                let mut shards: Vec<String> = map["weight_map"]
                    .as_object()
                    .ok_or("a shard index with no weight_map")?
                    .values()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect();
                shards.sort();
                shards.dedup();
                shards.iter().map(|s| fetch_file(repo, s, watch)).collect::<Res<Vec<_>>>()?
            }
            Err(_) => vec![fetch_file(repo, "model.safetensors", watch)
                .map_err(|e| format!("{repo} has no safetensors weights (model.safetensors): {e}"))?],
        };
        Ok(Files { config, tokenizer, generation, weights })
    }
}

impl Transcriber {
    /// A Whisper from `repo`, on the CPU, its matrices at `precision`.
    pub fn load(repo: &str, precision: Precision, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Transcriber> {
        Transcriber::load_with(repo, progress, watch, &mut |config, files, progress| {
            progress(&format!("at {precision}, on the CPU"));
            Ok(Box::new(whisper::Cpu::load(config, &files.weights, precision)?))
        })
    }

    /// A Whisper from `repo`, its model made by `model` from its config and
    /// files: what a GPU's loader uses, with everything around the model
    /// read here as for the CPU.
    pub fn load_with(
        repo: &str,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
        model: &mut dyn FnMut(whisper::Config, &Files, &mut dyn FnMut(&str)) -> Res<Box<dyn Model>>,
    ) -> Res<Transcriber> {
        let files = Files::fetch(repo, watch)?;
        let config_json = crate::weights::read_json(&files.config)?;
        let config = whisper::Config::from_json(&config_json).map_err(|e| format!("{repo}: {e}"))?;
        let generation = files.generation.as_deref().map(crate::weights::read_json).transpose()?;
        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer).map_err(|e| e.to_string())?;
        let tokens = transcribe::Tokens::read(&tokenizer, &config_json, generation.as_ref())?;
        if tokens.heads_guessed {
            progress(&format!("{repo} names no alignment heads: words are timed by every head of the decoder's second half, and less well"));
        }
        progress(&format!(
            "loading {repo}: {} + {} blocks, d={}, {} mel bands",
            config.encoder_layers, config.decoder_layers, config.d, config.n_mels
        ));
        let model = model(config, &files, progress)?;
        Ok(Transcriber { model, tokenizer, tokens })
    }

    /// Where it runs, as a language model's session says it: `cpu f32`.
    pub fn backend(&self) -> String {
        self.model.backend()
    }

    pub fn summary(&self) -> String {
        let c = self.model.config();
        format!(
            "whisper: {}M params, {} encoder + {} decoder blocks, d={}, {} mel bands",
            self.model.params() / 1_000_000,
            c.encoder_layers,
            c.decoder_layers,
            c.d,
            c.n_mels
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against `scripts/whisper-fixtures.py`: transformers' own numbers for
    /// one WAV, stage by stage ([`reference`]), on the CPU at f32.
    ///
    /// ```text
    /// KVAD_WHISPER_FIXTURES=/tmp/whisper-fx cargo test --release -p kvad speech::tests -- --ignored --nocapture
    /// ```
    ///
    /// The model is read from the cache only; the script will have put it
    /// there. `KVAD_WHISPER_PRECISION=q8` (or q4) quantises it first, and
    /// then the stages are only reported, against a looser bound; the
    /// tokens are held to the same. kvad-gpu's `speech::tests` is the same
    /// on Metal.
    #[test]
    #[ignore]
    fn whisper_agrees_with_transformers() {
        let dir = PathBuf::from(std::env::var("KVAD_WHISPER_FIXTURES").expect("KVAD_WHISPER_FIXTURES is not set"));
        let fx = reference::Fixtures::open(&dir).unwrap();
        let repo = fx.repo().unwrap();
        // The model is made here and shared with the transcriber, so that
        // the encoder, which only its own type shows, can be asked for too.
        let precision = std::env::var("KVAD_WHISPER_PRECISION").ok().map_or(Precision::F32, |p| Precision::parse(&p).expect("f32, q8 or q4"));
        let mut made = None;
        let t = Transcriber::load_with(&repo, &mut |m| eprintln!("{m}"), &Watcher::none(), &mut |config, files, _| {
            let model = std::sync::Arc::new(whisper::Cpu::load(config, &files.weights, precision)?);
            made = Some(model.clone());
            Ok(Box::new(model))
        })
        .unwrap();
        let model = made.unwrap();
        let encode = |window: &[f32]| -> Res<Vec<f32>> { Ok(model.whisper.encode(window)) };
        let tol = match precision {
            Precision::F32 => reference::Tolerance { encoder: 2e-3, logits: 2e-3, frames: 1 },
            _ => reference::Tolerance { encoder: 0.75, logits: 0.1, frames: usize::MAX },
        };
        reference::compare(&fx, &t, &encode, tol).unwrap();
    }
}
