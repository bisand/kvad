//! Speech to text: Whisper, on the CPU, written out.
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
//!
//! Every Whisper on the Hub in transformers' layout loads: OpenAI's own
//! (`openai/whisper-*`), the distilled ones, and fine-tunes of them such as
//! the National Library of Norway's NB-Whisper (`NbAiLab/nb-whisper-*`).

pub mod audio;
pub mod mel;
pub mod transcribe;
pub mod whisper;

pub use transcribe::{Format, Progress, Request, Segment, Task, Transcriber, Transcript};

use crate::quant::Precision;
use crate::weights::{fetch_file, Watcher};
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

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
    let config = crate::weights::local_file(repo, "config.json")?;
    let config = whisper::Config::from_json(&crate::weights::read_json(&config).ok()?).ok()?;
    let (matrices, floats) = config.params();
    Some(precision.weight_bytes(matrices as u64) + 4 * floats as u64)
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
    /// A Whisper from `repo`, its matrices at `precision`.
    pub fn load(repo: &str, precision: Precision, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Transcriber> {
        let files = Files::fetch(repo, watch)?;
        let config_json = crate::weights::read_json(&files.config)?;
        let config = whisper::Config::from_json(&config_json).map_err(|e| format!("{repo}: {e}"))?;
        let generation = files.generation.as_deref().map(crate::weights::read_json).transpose()?;
        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer).map_err(|e| e.to_string())?;
        let tokens = transcribe::Tokens::read(&tokenizer, &config_json, generation.as_ref())?;
        progress(&format!(
            "loading {repo}: {} + {} blocks, d={}, {} mel bands, at {precision}",
            config.encoder_layers, config.decoder_layers, config.d, config.n_mels
        ));
        let model = whisper::Whisper::load(config, &files.weights, precision)?;
        Ok(Transcriber { model, tokenizer, tokens, pool: crate::model::pool(), precision })
    }

    /// Where it runs, as a language model's session says it: `cpu f32`.
    pub fn backend(&self) -> String {
        format!("cpu {}", self.precision)
    }

    pub fn summary(&self) -> String {
        let c = &self.model.config;
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
    /// one WAV, stage by stage, so that a difference says where it starts.
    ///
    /// ```text
    /// KVAD_WHISPER_FIXTURES=/tmp/whisper-fx cargo test --release -p kvad speech::tests -- --ignored --nocapture
    /// ```
    ///
    /// The model is read from the cache only; the script will have put it
    /// there.
    #[test]
    #[ignore]
    fn whisper_agrees_with_transformers() {
        let dir = PathBuf::from(std::env::var("KVAD_WHISPER_FIXTURES").expect("KVAD_WHISPER_FIXTURES is not set"));
        let meta = crate::weights::read_json(&dir.join("tokens.json")).unwrap();
        let repo = meta["model"].as_str().unwrap();
        crate::weights::local_file(repo, "config.json").unwrap_or_else(|| panic!("{repo} is not in the cache"));
        let fx = crate::weights::Checkpoint::open(&[dir.join("whisper.safetensors")]).unwrap();
        let ids = |k: &str| -> Vec<u32> { meta[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
        let gap = |a: &[f32], b: &[f32]| -> (f32, f32) {
            assert_eq!(a.len(), b.len());
            let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            let scale = b.iter().map(|v| v.abs()).fold(0.0, f32::max);
            (max, scale)
        };

        let t = Transcriber::load(repo, Precision::F32, &mut |m| eprintln!("{m}"), &Watcher::none()).unwrap();
        let audio = audio::from_wav(&std::fs::read(dir.join("audio.wav")).unwrap()).unwrap();
        let audio = &audio[..audio.len().min(mel::N_SAMPLES)];

        // The spectrogram, against transformers' [n_mels, 3000].
        let ours = mel::log_mel(audio, mel::N_SAMPLES - audio.len(), t.model.config.n_mels);
        let theirs = fx.get("mel").unwrap();
        let (max, _) = gap(&ours.concat(), &theirs.data);
        eprintln!("mel: max gap {max:.2e}");
        assert!(max < 1e-4, "the spectrogram is {max} from transformers'");

        // The encoder, from transformers' spectrogram so that only the
        // encoder is being compared.
        let n_mels = t.model.config.n_mels;
        let mut window = vec![0.0f32; mel::N_FRAMES * n_mels];
        for m in 0..n_mels {
            for f in 0..mel::N_FRAMES {
                window[f * n_mels + m] = theirs.data[m * mel::N_FRAMES + f];
            }
        }
        let encoded = t.model.encode(&window);
        let (max, scale) = gap(&encoded, &fx.get("encoded").unwrap().data);
        eprintln!("encoder: max gap {max:.2e} on values up to {scale:.1}");
        assert!(max < 2e-3 * scale.max(1.0), "the encoder is {max} from transformers'");

        // The decoder's first scores after the prompt.
        let heard = t.model.listen(&encoded);
        let mut cache = whisper::TextCache::new(t.model.config.decoder_layers);
        let prompt = ids("prompt");
        let h = t.model.decode(&prompt, &mut cache, &heard).unwrap();
        let logits = t.model.logits(&h[(prompt.len() - 1) * t.model.config.d..]);
        let (max, scale) = gap(&logits, &fx.get("logits").unwrap().data);
        eprintln!("decoder: max gap {max:.2e} on logits up to {scale:.1}");
        assert!(max < 2e-3 * scale.max(1.0), "the decoder is {max} from transformers'");

        // Greedy tokens, with timestamps and without, for the one window.
        let language = meta["language"].as_str().unwrap().to_string();
        for (timestamps, key) in [(false, "plain"), (true, "timed")] {
            let req = Request { language: Some(language.clone()), timestamps, temperature: Some(0.0), ..Request::default() };
            let got = t.transcribe(audio, &req, &mut |_| true).unwrap();
            let ours: Vec<u32> = got.segments.iter().flat_map(|s| s.tokens.iter().copied()).collect();
            let want = ids(key);
            eprintln!("{key}: {}", got.text);
            assert_eq!(ours, want, "{key} tokens differ from transformers'");
        }
    }
}
