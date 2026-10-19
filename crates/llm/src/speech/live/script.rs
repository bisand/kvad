//! A Whisper that is not one, for tests that drive a [`Live`] session: a
//! model that writes a fixed script, one word for every second of sound it
//! hears, as a real one would write what is said. It times its segments three words
//! at a time when asked to, leaves out the segment running into the end of
//! its window as Whisper does, and writes half of a two-token word whose
//! second is not yet heard. So what is settled, and where a window is cut,
//! can be checked against what the script says, with no model downloaded.
//! [`Script::restless`] is one that never writes the same plain guess
//! twice running, so that no update agrees with the last;
//! [`Script::punctuating`] writes the same words each time, with a full
//! stop and a question mark after them by turns, as Whisper writes a "Ja"
//! in a meeting.
//!
//! Built for this crate's tests, and with the `script` feature for
//! kvad-serve's, which play it into `/v1/realtime`. Not a part of the API.
//!
//! [`Live`]: super::Live

use super::super::mel::SAMPLE_RATE;
use super::super::transcribe::{Tokens, Transcriber};
use super::super::{whisper, Heard, Model, Said};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The script's words: `Ġw0` to `Ġw49`, ids 0 to 49, and every fourth word
/// is written in two tokens, its second `x` (id 50).
pub const WORDS: u32 = 50;
const X: u32 = 50;
const EOT: u32 = 52;
const SOT: u32 = 53;
const SOT_PREV: u32 = 54;
const TRANSCRIBE: u32 = 55;
const TRANSLATE: u32 = 56;
const NO_TIMESTAMPS: u32 = 57;
const NO_SPEECH: u32 = 58;
const NO: u32 = 59;
const DOT: u32 = 60;
const ASK: u32 = 61;
const TB: u32 = 62;
const VOCAB: usize = TB as usize + 1501;
/// Timestamp steps in a second.
const STEPS: u32 = 50;

fn script_word(k: u32) -> Vec<u32> {
    match k % 4 {
        3 => vec![k, X],
        _ => vec![k],
    }
}

fn tokenizer() -> tokenizers::Tokenizer {
    let mut vocab = serde_json::Map::new();
    for k in 0..WORDS {
        vocab.insert(format!("Ġw{k}"), k.into());
    }
    vocab.insert("x".into(), X.into());
    vocab.insert(".".into(), DOT.into());
    vocab.insert("?".into(), ASK.into());
    vocab.insert("<unk>".into(), 51.into());
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": null, "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" },
    });
    json.to_string().parse().unwrap()
}

fn tokens() -> Tokens {
    Tokens {
        eot: EOT,
        sot: SOT,
        sot_prev: SOT_PREV,
        transcribe: TRANSCRIBE,
        translate: TRANSLATE,
        no_timestamps: NO_TIMESTAMPS,
        no_speech: Some(NO_SPEECH),
        timestamp_begin: TB,
        languages: vec![("no".into(), NO)],
        suppress: vec![SOT, SOT_PREV, TRANSCRIBE, TRANSLATE, NO_SPEECH],
        suppress_begin: vec![EOT],
        default_language: Some("no".into()),
        max_initial_timestamp: 50,
        multilingual: true,
        utf8: Vec::new(),
        heads: vec![(0, 0)],
        heads_guessed: false,
    }
}

pub struct Script {
    config: whisper::Config,
    restless: bool,
    punctuating: bool,
}

impl Script {
    /// A transcriber over the script.
    pub fn transcriber() -> Transcriber {
        Script::over(false, false)
    }

    /// A transcriber over the script that writes every word with an `x`
    /// after it, plain, when it hears an odd number of seconds, and none
    /// with one when it hears an even number. A word is not the same word
    /// with an `x`, so two updates a second apart agree on nothing. Timed,
    /// it writes the script as [`Script::transcriber`]'s does. It goes on
    /// from the last word written, as Whisper goes on from what it is
    /// handed, so a settled word may be given an `x` after it.
    pub fn restless() -> Transcriber {
        Script::over(true, false)
    }

    /// A transcriber over the script that writes, plain, a full stop after
    /// every word when it hears an odd number of seconds and a question mark
    /// when it hears an even number: the same words, punctuated by turns.
    pub fn punctuating() -> Transcriber {
        Script::over(false, true)
    }

    fn over(restless: bool, punctuating: bool) -> Transcriber {
        let config = whisper::Config {
            n_mels: 80,
            d: 8,
            encoder_layers: 1,
            encoder_heads: 1,
            encoder_ffn: 8,
            decoder_layers: 1,
            decoder_heads: 1,
            decoder_ffn: 8,
            audio_ctx: 1500,
            text_ctx: 448,
            vocab: VOCAB,
        };
        Transcriber { model: Box::new(Script { config, restless, punctuating }), tokenizer: tokenizer(), tokens: tokens() }
    }

    /// What the model would write after `seq`, heard `seconds` of sound.
    fn next(&self, seq: &[u32], seconds: f32) -> Vec<f32> {
        let mut logits = vec![-10.0f32; VOCAB];
        let sot = seq.iter().position(|&i| i == SOT).expect("a prompt has a start");
        if seq.len() == sot + 1 {
            // The row that says whether there is speech.
            logits[NO_SPEECH as usize] = if seconds < 0.5 { 10.0 } else { -10.0 };
            return logits;
        }
        // The window's words follow the last word of its context.
        let base = seq[..sot].iter().rev().find(|&&i| i < WORDS).map_or(0, |&i| i + 1);
        let plain = seq[sot..].contains(&NO_TIMESTAMPS);
        let begin = sot + if plain { 4 } else { 3 };
        if seq.len() < begin {
            return logits;
        }
        let mut target = Vec::new();
        let whole = (seconds.floor() as u32).min(WORDS - base);
        if plain && self.restless {
            let x = seconds.round() as u32 % 2 == 1;
            let next = match seq[begin..].last() {
                Some(&k) if k < WORDS && x => X,
                _ => match seq[begin..].iter().rev().find(|&&i| i < WORDS).map_or(base, |&i| i + 1) {
                    k if k < base + whole => k,
                    _ => EOT,
                },
            };
            logits[next as usize] = 10.0;
            return logits;
        }
        if plain && self.punctuating {
            // Goes on from what it is handed, whatever marks are in it: a
            // word's `x` if it is two tokens, its mark, then the next word.
            let mark = if seconds.round() as u32 % 2 == 1 { DOT } else { ASK };
            let written = &seq[begin..];
            let last = written.iter().rev().find(|&&i| i < WORDS).copied();
            let next = match (last, written.last()) {
                (Some(k), Some(&t)) if t == k && k % 4 == 3 => X,
                (Some(_), Some(&t)) if t < WORDS || t == X => mark,
                _ => match last.map_or(base, |k| k + 1) {
                    k if k < base + whole => k,
                    _ => EOT,
                },
            };
            logits[next as usize] = 10.0;
            return logits;
        }
        if plain {
            for j in 0..whole {
                target.extend(script_word(base + j));
            }
            // A word half heard is written half, where it is two tokens.
            if seconds - seconds.floor() > 0.5 && base + whole < WORDS {
                target.push(base + whole);
            }
        } else {
            for s in 0..whole / 3 {
                target.push(TB + 3 * s * STEPS);
                for j in 3 * s..3 * s + 3 {
                    target.extend(script_word(base + j));
                }
                target.push(TB + 3 * (s + 1) * STEPS);
            }
        }
        target.push(EOT);
        let written = &seq[begin..];
        let next = match target.starts_with(written) {
            true => target.get(written.len()).copied().unwrap_or(EOT),
            false => EOT,
        };
        logits[next as usize] = 10.0;
        logits
    }
}

impl Model for Script {
    fn config(&self) -> &whisper::Config {
        &self.config
    }
    /// Seconds of sound in the window: its loud frames. [`sound`] is about
    /// 1.4 in the log-mel, and silence or padding under 0, whether or not
    /// the window is full.
    fn hear(&self, window: &[f32]) -> Res<Heard> {
        let frames: Vec<f32> = window.chunks(80).map(|f| f.iter().copied().fold(f32::MIN, f32::max)).collect();
        let loud = frames.iter().filter(|&&f| f > 0.0).count();
        Ok(Heard(Box::new(loud as f32 / 100.0)))
    }
    fn said(&self) -> Said {
        Said(Box::new(Vec::<u32>::new()))
    }
    fn decode(&self, tokens: &[u32], said: &mut Said, heard: &Heard, rows: &[usize]) -> Res<Vec<Vec<f32>>> {
        let seconds = *heard.get::<f32>()?;
        let seq = said.get_mut::<Vec<u32>>()?;
        let before = seq.len();
        seq.extend_from_slice(tokens);
        Ok(rows.iter().map(|&r| self.next(&seq[..before + r + 1], seconds)).collect())
    }
    /// Not asked: a live session times no words.
    fn attention(&self, _: &[u32], _: &Heard, _: &[(usize, usize)]) -> Res<Vec<f32>> {
        Err("the script times no words".into())
    }
    fn within(&self, work: &mut (dyn FnMut() + Send)) {
        work()
    }
    fn params(&self) -> usize {
        0
    }
    fn weight_bytes(&self) -> usize {
        0
    }
    fn backend(&self) -> String {
        "script".into()
    }
}

/// `seconds` of sound, which the script hears as that many words.
pub fn sound(seconds: f32) -> Vec<f32> {
    (0..(seconds * SAMPLE_RATE as f32) as usize).map(|i| 0.5 * (i as f32 * 0.17).sin()).collect()
}

/// The script's first `n` words, as the transcriber writes them.
pub fn script_text(n: u32) -> String {
    (0..n).map(|k| if k % 4 == 3 { format!(" w{k}x") } else { format!(" w{k}") }).collect()
}
