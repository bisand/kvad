//! A live session driven over a Whisper that is not one: a model that
//! writes a fixed script, one word for every second of sound it hears, as
//! a real one would write what is said. It times its segments three words
//! at a time when asked to, leaves out the segment running into the end of
//! its window as Whisper does, and writes half of a two-token word whose
//! second is not yet heard. So what is settled, and where a window is cut,
//! can be checked against what the script says, with no model downloaded.

use super::*;
use crate::speech::transcribe::Tokens;
use crate::speech::{whisper, Heard, Model, Said};

/// The script's words: `Ġw0` to `Ġw49`, ids 0 to 49, and every fourth word
/// is written in two tokens, its second `x` (id 50).
const WORDS: u32 = 50;
const X: u32 = 50;
const EOT: u32 = 52;
const SOT: u32 = 53;
const SOT_PREV: u32 = 54;
const TRANSCRIBE: u32 = 55;
const TRANSLATE: u32 = 56;
const NO_TIMESTAMPS: u32 = 57;
const NO_SPEECH: u32 = 58;
const NO: u32 = 59;
const TB: u32 = 60;
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
    }
}

struct Script {
    config: whisper::Config,
}

impl Script {
    fn transcriber() -> Transcriber {
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
        Transcriber { model: Box::new(Script { config }), tokenizer: tokenizer(), tokens: tokens() }
    }

    /// What the model would write after `seq`, heard `seconds` of sound.
    fn next(seq: &[u32], seconds: f32) -> Vec<f32> {
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
    /// Seconds of sound in the window: its frames louder than the padding.
    fn hear(&self, window: &[f32]) -> Res<Heard> {
        let frames: Vec<f32> = window.chunks(80).map(|f| f.iter().copied().fold(f32::MIN, f32::max)).collect();
        let floor = frames.iter().copied().fold(f32::MAX, f32::min);
        let loud = frames.iter().filter(|&&f| f > floor + 0.5).count();
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
        Ok(rows.iter().map(|&r| Script::next(&seq[..before + r + 1], seconds)).collect())
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

fn sound(seconds: f32) -> Vec<f32> {
    (0..(seconds * SAMPLE_RATE as f32) as usize).map(|i| 0.5 * (i as f32 * 0.17).sin()).collect()
}

fn script_text(n: u32) -> String {
    (0..n).map(|k| if k % 4 == 3 { format!(" w{k}x") } else { format!(" w{k}") }).collect()
}

/// Sound pushed a second at a time, an update after each: what is settled
/// is the script, in order, never taken back, and the window is cut where
/// a segment ends rather than forced.
#[test]
fn settled_text_is_the_script_and_never_changes() {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    let (mut settled, mut cuts) = (String::new(), 0);
    let mut shown_first = 0;
    for second in 1..=40 {
        live.push(&sound(1.0));
        let u = live.update(&tr).unwrap();
        assert!(!u.forced, "forced at {second} s");
        cuts += u.cut as usize;
        settled.push_str(&u.settled);
        let now = format!("{settled}{}", u.partial);
        assert!(script_text(40).starts_with(&settled), "settled `{settled}` at {second} s");
        // The partial is the script too, a word or two ahead.
        assert!(script_text(40).starts_with(&now), "shown `{now}` at {second} s");
        if second == 1 {
            shown_first = u.partial.len();
        }
        assert_eq!(u.heard, second as f32);
    }
    assert!(shown_first > 0, "the first word showed as a partial before it settled");
    assert!(cuts >= 1, "a 40 s session is cut at least once");
    settled.push_str(&live.finish(&tr).unwrap().settled);
    assert_eq!(settled, script_text(40));
}

/// A word whose first half is heard, and then all of it, is not settled
/// on the half the two guesses share: whole words are compared.
#[test]
fn half_a_word_is_not_agreed_with_all_of_it() {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    // Word 3 is two tokens; at 3.7 s its first is written.
    live.push(&sound(3.7));
    let a = live.update(&tr).unwrap();
    assert_eq!((a.settled.as_str(), a.partial.as_str()), ("", " w0 w1 w2 w3"));
    live.push(&sound(0.6));
    let b = live.update(&tr).unwrap();
    assert_eq!((b.settled.as_str(), b.partial.as_str()), (" w0 w1 w2", " w3x"));
    let c = live.update(&tr).unwrap();
    assert_eq!((c.settled.as_str(), c.partial.as_str()), (" w3x", ""));
}

/// Silence settles nothing, and is not kept: the window never fills with
/// it, and speech after it settles as before.
#[test]
fn silence_is_let_go_of() {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    for _ in 0..45 {
        live.push(&vec![0.0; SAMPLE_RATE]);
        let u = live.update(&tr).unwrap();
        assert!(u.silent && u.settled.is_empty() && !u.forced);
    }
    assert!(live.window.len() <= QUIET_KEPT);
    assert_eq!(live.finish(&tr).unwrap().settled, "", "finishing on silence settles nothing");
    live.push(&sound(2.0));
    live.update(&tr).unwrap();
    assert_eq!(live.finish(&tr).unwrap().settled, " w0 w1");
    assert_eq!(live.received(), 47.0);
}
