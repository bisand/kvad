//! A Whisper held against transformers', stage by stage.
//!
//! `scripts/whisper-fixtures.py` runs transformers on one WAV and writes what
//! each stage made of it: the spectrogram, the encoder's output, the scores
//! after the prompt, the greedy tokens with and without timestamps, and the
//! alignment heads' attention with the frame each token starts at.
//! [`compare`] runs a [`Transcriber`] on the same audio and says how far each
//! stage is from those, so that a difference says where it starts.
//!
//! Public because two crates' tests use it: this one's for the CPU, and
//! kvad-gpu's for Metal. The stages it compares are the ones every [`Model`]
//! has, plus the encoder's output, which only the model's own type can show
//! and so is asked for by the caller.

use super::{align, audio, mel, Model, Request, Transcriber};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What a model is allowed to differ from transformers' f32 by, as a
/// fraction of the largest number in the reference: the encoder's output and
/// the scores after the prompt; and the frames a token's start may be from
/// `timing.py`'s over transformers' attention. The tokens may not differ at
/// all.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub encoder: f32,
    pub logits: f32,
    pub frames: usize,
}

/// How far each stage was: the largest difference, and the largest number in
/// the reference it is measured against.
#[derive(Debug, Clone, Default)]
pub struct Gaps {
    pub mel: f32,
    pub encoder: (f32, f32),
    pub logits: (f32, f32),
    /// The alignment heads' weights, and the most frames any token's start
    /// is from the reference's.
    pub attention: (f32, f32),
    pub frames: usize,
}

/// The fixtures in `dir`: which model they were made with, the audio, and
/// transformers' numbers.
pub struct Fixtures {
    meta: serde_json::Value,
    numbers: crate::weights::Checkpoint,
    /// The first 30 seconds of `audio.wav`, as the script read it.
    pub audio: Vec<f32>,
}

impl Fixtures {
    pub fn open(dir: &Path) -> Res<Fixtures> {
        let meta = crate::weights::read_json(&dir.join("tokens.json"))?;
        let numbers = crate::weights::Checkpoint::open(&[dir.join("whisper.safetensors")])?;
        let mut audio = audio::from_wav(&std::fs::read(dir.join("audio.wav"))?)?;
        audio.truncate(mel::N_SAMPLES);
        Ok(Fixtures { meta, numbers, audio })
    }

    /// The model they were made with, which the test reads from the cache
    /// only: the script will have put it there.
    pub fn repo(&self) -> Res<String> {
        let repo = self.meta["model"].as_str().ok_or("tokens.json names no model")?;
        crate::weights::local_file(repo, "config.json").ok_or_else(|| format!("{repo} is not in the cache"))?;
        Ok(repo.to_string())
    }

    fn ids(&self, key: &str) -> Vec<u32> {
        self.meta[key].as_array().map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect()).unwrap_or_default()
    }

    /// transformers' spectrogram, `[n_mels][3000]`, laid out as
    /// [`mel::window`] lays a window out, time-major: what an encoder is
    /// given to compare encoders alone.
    pub fn window(&self, n_mels: usize) -> Res<Vec<f32>> {
        let theirs = self.numbers.get("mel")?;
        let mut window = vec![0.0f32; mel::N_FRAMES * n_mels];
        for m in 0..n_mels {
            for f in 0..mel::N_FRAMES {
                window[f * n_mels + m] = theirs.data[m * mel::N_FRAMES + f];
            }
        }
        Ok(window)
    }
}

/// The largest difference between `a` and `b`, and the largest magnitude in
/// `b`.
pub fn gap(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "the two are not the same size");
    // `fold` with `max` would pass over a NaN: a comparison with one is
    // false. So a NaN anywhere is the largest gap there is.
    let max = a.iter().zip(b).map(|(x, y)| if (x - y).is_nan() { f32::INFINITY } else { (x - y).abs() }).fold(0.0, f32::max);
    let scale = b.iter().map(|v| v.abs()).fold(0.0, f32::max);
    (max, scale)
}

/// `t` against the fixtures, stage by stage, each said as it is measured.
///
/// `encode` is the encoder's output for a window, `[1500][d]`, from the
/// model's own type: the one stage [`Model`] does not show.
pub fn compare(fx: &Fixtures, t: &Transcriber, encode: &(dyn Fn(&[f32]) -> Res<Vec<f32>> + Sync), tol: Tolerance) -> Res<Gaps> {
    let mut gaps = Gaps::default();
    let model: &dyn Model = t.model.as_ref();
    let n_mels = model.config().n_mels;
    let audio = &fx.audio;

    // The spectrogram, against transformers' [n_mels, 3000]. Host
    // arithmetic, the same whichever model.
    let ours = mel::log_mel(audio, mel::N_SAMPLES - audio.len(), n_mels);
    let (max, _) = gap(&ours.concat(), &fx.numbers.get("mel")?.data);
    eprintln!("mel: max gap {max:.2e}");
    if max >= 1e-4 {
        return Err(format!("the spectrogram is {max} from transformers'").into());
    }
    gaps.mel = max;

    // The encoder, from transformers' spectrogram, so that only the encoder
    // is being compared.
    let window = fx.window(n_mels)?;
    let encoded = t.on_pool(|| encode(&window))?;
    let (max, scale) = gap(&encoded, &fx.numbers.get("encoded")?.data);
    eprintln!("encoder: max gap {max:.2e} on values up to {scale:.1}");
    gaps.encoder = (max, scale);
    if max >= tol.encoder * scale.max(1.0) {
        return Err(format!("the encoder is {max} from transformers'").into());
    }

    // The decoder's first scores after the prompt, from this model's own
    // encoder over the same spectrogram.
    let prompt = fx.ids("prompt");
    let logits = t.on_pool(|| {
        let heard = model.hear(&window)?;
        let mut said = model.said();
        Ok(model.decode(&prompt, &mut said, &heard, &[prompt.len() - 1])?.remove(0))
    })?;
    let (max, scale) = gap(&logits, &fx.numbers.get("logits")?.data);
    eprintln!("decoder: max gap {max:.2e} on logits up to {scale:.1}");
    gaps.logits = (max, scale);
    if max >= tol.logits * scale.max(1.0) {
        return Err(format!("the decoder is {max} from transformers'").into());
    }

    // Words with times: the alignment heads' weights over the plain text,
    // and the frame each token starts at by them. In fixtures that have
    // them; older ones were made before words were.
    let rows = fx.ids("align_rows");
    if rows.is_empty() {
        eprintln!("attention: these fixtures have none; make them again for words");
    } else {
        let heads: Vec<(usize, usize)> = fx.meta["heads"]
            .as_array()
            .ok_or("tokens.json has align_rows and no heads")?
            .iter()
            .filter_map(|p| Some((p[0].as_u64()? as usize, p[1].as_u64()? as usize)))
            .collect();
        let weights = t.on_pool(|| {
            let heard = model.hear(&window)?;
            model.attention(&rows, &heard, &heads)
        })?;
        let (max, scale) = gap(&weights, &fx.numbers.get("attention")?.data);
        eprintln!("attention: max gap {max:.2e} on weights up to {scale:.2}");
        gaps.attention = (max, scale);
        let text = fx.ids("plain").len();
        let sot = rows.len() - text - 2;
        let frames = weights.len() / (heads.len() * rows.len());
        let ours = align::token_frames(&weights, heads.len(), rows.len(), frames, audio.len() / mel::HOP / 2, sot);
        let theirs = fx.ids("token_starts");
        let worst = ours.iter().zip(&theirs).map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs() as usize).max().unwrap_or(0);
        let late = ours.iter().zip(&theirs).filter(|(&a, &b)| a.abs_diff(b as usize) > 1).count();
        // transformers' own token timestamps end each token where the next
        // starts here; its first token ends at frame 0 by construction
        // (`align`'s module comment), so that one is left out.
        let ends = fx.ids("token_ends");
        let off: Vec<String> =
            ends.iter().enumerate().skip(1).filter(|&(k, &e)| ours.get(k + 1).is_some_and(|&s| s.abs_diff(e as usize) > 1)).map(|(k, &e)| format!("{k}: {} against {e}", ours[k + 1])).collect();
        eprintln!(
            "token starts: {} tokens, most {worst} frames from timing.py's, {late} more than a frame; against transformers' own ends, {} of {} more than a frame{}",
            ours.len(),
            off.len(),
            ends.len().saturating_sub(1),
            if off.is_empty() { String::new() } else { format!(" ({})", off.join(", ")) }
        );
        if ours.len() != theirs.len() {
            return Err(format!("{} token starts against timing.py's {}", ours.len(), theirs.len()).into());
        }
        gaps.frames = worst;
        if worst > tol.frames {
            return Err(format!("a token starts {worst} frames from where timing.py has it").into());
        }
    }

    // Greedy tokens, with timestamps and without, for the one window: both
    // asked before either is judged, so that a failure says which.
    let language = fx.meta["language"].as_str().ok_or("tokens.json names no language")?.to_string();
    let mut differ = Vec::new();
    for (timestamps, key) in [(false, "plain"), (true, "timed")] {
        let req = Request { language: Some(language.clone()), timestamps, temperature: Some(0.0), ..Request::default() };
        let got = t.transcribe(audio, &req, &mut |_| true)?;
        let ours: Vec<u32> = got.segments.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        eprintln!("{key}: {}", got.text);
        let want = fx.ids(key);
        if ours != want {
            let at = ours.iter().zip(&want).take_while(|(a, b)| a == b).count();
            differ.push(format!("{key} tokens differ from transformers' from token {at}: {:?} against {:?}", &ours[at..], &want[at..]));
        }
    }
    match differ.is_empty() {
        true => Ok(gaps),
        false => Err(differ.join("; ").into()),
    }
}
