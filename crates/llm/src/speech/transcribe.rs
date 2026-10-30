//! The decoding loop: what the decoder is told, what it may say, and how a
//! long recording is walked through 30 seconds at a time.
//!
//! None of this is learned and none of it can be guessed. Each rule below is
//! openai-whisper's (`decoding.py` and `transcribe.py`), and each is a
//! quiet quality bug when it is wrong — a transcript that is fluent and
//! slightly off, rather than an error.
//!
//! # The prompt is tokens
//!
//! ```text
//!   [<|startofprev|> …the previous window's text…]
//!   <|startoftranscript|> <|no|> <|transcribe|> [<|notimestamps|>]
//! ```
//!
//! The language token says what language the speech is in (and the text
//! should be), the task token whether to write it down or translate it to
//! English, and `<|notimestamps|>` turns off the timestamp tokens that
//! otherwise bracket every segment. The part before `<|startoftranscript|>`
//! is context: the text so far, so the model keeps names and spelling
//! consistent across a seam between windows.
//!
//! # What the model may not say
//!
//! Before each token is chosen, some are taken off the table:
//!
//! - the tokens the checkpoint lists as `suppress_tokens` (mostly symbols
//!   that are not speech), and the special tokens that only belong in a
//!   prompt;
//! - at the first position, a bare space and end-of-text, so a window is not
//!   "transcribed" as nothing;
//! - with timestamps on, everything [`Rules::timestamps`] says: they come in
//!   pairs, never go backwards, open the text, and win whenever all of them
//!   together are likelier than any one word.
//!
//! # Long audio
//!
//! The encoder reads 30 seconds. A longer recording is read a window at a
//! time, and with timestamps on, where each window *ends* is the model's
//! decision: the last complete segment's end timestamp is where the next
//! window starts, so a word cut in half at the 30-second mark is read again
//! whole in the next window rather than mangled in both.
//!
//! # When a window goes wrong
//!
//! A window whose tokens the model was unsure of on average (log-probability
//! below −1), or that goes round in circles, is decoded again at a higher
//! temperature, up to five times. openai-whisper measures the circling by
//! how well the text gzips; this crate counts repeated 4-grams instead
//! (`repeated`). A loop usually runs into the token limit, but not always:
//! on AMI's meetings whisper-large-v3-turbo wrote thirty confident one-second
//! "Okay." segments and stopped, window after window.

use super::align::Word;
use super::mel::{self, HOP, N_FRAMES, SAMPLE_RATE};
use super::{Heard, Model};
use crate::sampler::Sampler;
use tokenizers::Tokenizer;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Seconds per timestamp step: `<|0.02|>` is one step after `<|0.00|>`.
pub const TIME_STEP: f32 = 0.02;
/// Spectrogram columns per encoder frame: the second convolution's stride.
const FRAMES_PER_STEP: usize = 2;
/// The temperatures a window is tried at, in order.
const TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
/// Below this average log-probability a window is decoded again.
const LOGPROB_THRESHOLD: f32 = -1.0;
/// Above this probability of `<|nospeech|>`, with a low log-probability as
/// well, a window is taken to be silence and skipped.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
/// Above this share of repeated 4-grams, a window that ran into the token
/// limit is a loop rather than speech too dense to fit. Speech measured 0.01
/// to 0.07, loops and pages of spaces 0.72 to 1.
const REPEATED_THRESHOLD: f32 = 0.5;
/// A window that ended is a loop too when this many of its 4-grams repeat
/// as well: AMI's meetings stopped cleanly after thirty one-second "Okay."
/// segments, 57 repeats, where NB Samtale's speech never passed 32 nor a
/// share of 0.31. A short "ja. ja. ja." is kept, as gzip's overhead keeps
/// it for openai-whisper.
const LOOP_MIN: usize = 12;

/// What to do with the speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// Write it down in the language it is in.
    #[default]
    Transcribe,
    /// Write it down in English.
    Translate,
}

/// What a caller asks a transcription for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Request {
    /// A language code, `no` or `en`. `None` takes the checkpoint's own
    /// default where it has one (NB-Whisper's is Norwegian) and otherwise
    /// asks the model; `Some("auto")` always asks the model.
    pub language: Option<String>,
    pub task: Task,
    /// Whether to ask for timestamp tokens. Segments have times either way;
    /// without them, a segment is a whole window.
    pub timestamps: bool,
    /// Text to read as if it had come before the recording: names, spelling,
    /// a style to follow.
    pub prompt: Option<String>,
    /// The first temperature to try; `0` is greedy, and a window that fails
    /// is tried again warmer. Set, it is the only one tried.
    pub temperature: Option<f32>,
    /// Feed each window's text to the next as context. On by default, as in
    /// openai-whisper.
    pub no_context: bool,
    pub seed: Option<u64>,
    /// Time every word as well as every segment ([`super::align`]): one
    /// more pass of the decoder over each window's text.
    pub words: bool,
}

/// One stretch of the transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    /// Its words with their times, when [`Request::words`] asked for them.
    pub words: Vec<Word>,
    /// Who says it, `A`, `B`, …, once [`super::speakers::assign`] has been
    /// given who spoke when.
    pub speaker: Option<String>,
}

/// A whole transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The language code that was read: asked for, the checkpoint's own, or
    /// detected.
    pub language: String,
    /// Seconds of audio.
    pub duration: f32,
    pub segments: Vec<Segment>,
}

/// How far along a transcription is, after each window.
#[derive(Debug, Clone)]
pub struct Progress {
    pub done: f32,
    pub total: f32,
    /// The segments that window added.
    pub segments: Vec<Segment>,
}

/// The token ids the loop needs by name, read from the tokenizer rather than
/// written down: they move by one between large-v3's vocabulary and the
/// others'.
#[derive(Debug, Clone)]
pub struct Tokens {
    pub eot: u32,
    pub sot: u32,
    pub sot_prev: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    pub no_speech: Option<u32>,
    pub timestamp_begin: u32,
    /// Every language token, as `(code, id)`, in id order.
    pub languages: Vec<(String, u32)>,
    /// What the checkpoint says may never be produced.
    pub suppress: Vec<u32>,
    /// What may not be produced first.
    pub suppress_begin: Vec<u32>,
    /// The language a prompt carries when none was asked for, if the
    /// checkpoint names one (`generation_config.json`'s `language`).
    pub default_language: Option<String>,
    /// How far into a window the first timestamp may be, in steps.
    pub max_initial_timestamp: usize,
    /// Whether the prompt carries a language and a task at all: the
    /// English-only checkpoints (`*.en`) have neither.
    pub multilingual: bool,
    /// What each token does to a character being written: see [`utf8`].
    /// Empty for a tokenizer that is not byte-level, which turns the rule
    /// off.
    pub utf8: Vec<[u8; 4]>,
    /// The decoder's heads whose cross-attention words are timed by,
    /// `(layer, head)`: see [`super::align::heads`].
    pub heads: Vec<(usize, usize)>,
    /// Whether the checkpoint named none, and `heads` is a guess.
    pub heads_guessed: bool,
}

impl Tokens {
    /// Read off a Whisper's tokenizer, `config.json` and, where there is one,
    /// `generation_config.json`.
    pub fn read(tok: &Tokenizer, config: &serde_json::Value, generation: Option<&serde_json::Value>) -> Res<Tokens> {
        let id = |s: &str| tok.token_to_id(s).ok_or_else(|| format!("the tokenizer has no {s}"));
        let sot = id("<|startoftranscript|>")?;
        let translate = id("<|translate|>")?;
        // Every token between the two is a language: `<|en|>`, `<|no|>`, ….
        let languages = (sot + 1..translate)
            .filter_map(|i| {
                let t = tok.id_to_token(i)?;
                let code = t.strip_prefix("<|")?.strip_suffix("|>")?;
                Some((code.to_string(), i))
            })
            .collect();
        // The checkpoint's lists, from wherever it keeps them: newer exports
        // in `generation_config.json`, older ones in `config.json`.
        let list = |key: &str| -> Vec<u32> {
            [generation, Some(config)]
                .into_iter()
                .flatten()
                .find_map(|v| v.get(key)?.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()))
                .unwrap_or_default()
        };
        let gen = |key: &str| generation.and_then(|g| g.get(key));
        let default_language = gen("language").and_then(|v| v.as_str()).map(|s| s.trim_start_matches("<|").trim_end_matches("|>").to_string());
        let layer_count = |k: &str| config.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let (heads, heads_guessed) = super::align::heads(generation, layer_count("decoder_layers"), layer_count("decoder_attention_heads"));
        let multilingual = gen("is_multilingual").and_then(|v| v.as_bool()).unwrap_or_else(|| tok.token_to_id("<|en|>").is_some());
        let mut t = Tokens {
            eot: id("<|endoftext|>")?,
            sot,
            sot_prev: id("<|startofprev|>")?,
            transcribe: id("<|transcribe|>")?,
            translate,
            no_timestamps: id("<|notimestamps|>")?,
            no_speech: tok.token_to_id("<|nospeech|>").or_else(|| tok.token_to_id("<|nocaptions|>")),
            timestamp_begin: id("<|0.00|>")?,
            languages,
            suppress: list("suppress_tokens"),
            suppress_begin: list("begin_suppress_tokens"),
            default_language,
            max_initial_timestamp: gen("max_initial_timestamp_index").and_then(|v| v.as_u64()).map_or(50, |v| v as usize),
            multilingual,
            utf8: utf8(tok),
            heads,
            heads_guessed,
        };
        // What openai-whisper suppresses besides the checkpoint's list: the
        // tokens that only ever belong in a prompt.
        let sot_lm = tok.token_to_id("<|startoflm|>");
        t.suppress.extend([t.sot, t.sot_prev, t.transcribe, t.translate].into_iter().chain(sot_lm).chain(t.no_speech));
        if t.suppress_begin.is_empty() {
            t.suppress_begin = vec![tok.token_to_id("Ġ").unwrap_or(220), t.eot];
        }
        Ok(t)
    }

    pub fn language(&self, code: &str) -> Option<u32> {
        self.languages.iter().find(|(c, _)| c == code).map(|&(_, i)| i)
    }
}

/// One window, decoded.
pub(super) struct Window {
    /// What was produced, without the end-of-text: what it was handed as
    /// written already, and then what it wrote.
    pub(super) tokens: Vec<u32>,
    pub(super) avg_logprob: f32,
    pub(super) no_speech_prob: f32,
    pub(super) temperature: f32,
    /// Whether it stopped by itself rather than at the token limit.
    pub(super) ended: bool,
}

impl Window {
    /// Whether it went round in circles: this crate's gzip test. Speech
    /// dense enough to fill the limit is not a loop, and is kept.
    fn looped(&self, eot: u32) -> bool {
        let (again, share) = repeated(&self.tokens, eot);
        share > REPEATED_THRESHOLD && (!self.ended || again >= LOOP_MIN)
    }

    /// Whether none of it is written down, as fallback left it: silence, or
    /// a loop at every temperature it was tried at. openai-whisper keeps the
    /// second, and its last try over noise is a page of spaces and stray
    /// letters.
    fn skipped(&self, eot: u32) -> bool {
        self.silent() || self.looped(eot)
    }

    /// Whether the model took the window for silence: likely no speech, and
    /// unsure of what it wrote over it.
    pub(super) fn silent(&self) -> bool {
        self.no_speech_prob > NO_SPEECH_THRESHOLD && self.avg_logprob < LOGPROB_THRESHOLD
    }
}

/// How many of the text's 4-grams in `tokens` were written earlier in them,
/// and what share: what a compressor would find to point back to.
/// Timestamps and special tokens are not text, and a window of nothing else
/// repeats all of it.
fn repeated(tokens: &[u32], eot: u32) -> (usize, f32) {
    let text: Vec<u32> = tokens.iter().copied().filter(|&i| i < eot).collect();
    if text.len() < 4 {
        return (0, 1.0);
    }
    let mut seen = std::collections::HashSet::new();
    let again = text.windows(4).filter(|g| !seen.insert(*g)).count();
    (again, again as f32 / (text.len() - 3) as f32)
}

/// The rules about what the next token may be, given what has been produced.
pub struct Rules<'a> {
    pub tokens: &'a Tokens,
    pub timestamps: bool,
}

impl Rules<'_> {
    /// Rule out, in `logits`, everything that may not come after `sampled`.
    pub fn apply(&self, logits: &mut [f32], sampled: &[u32]) {
        let t = self.tokens;
        let ban = |logits: &mut [f32], range: std::ops::Range<usize>| {
            let end = range.end.min(logits.len());
            logits[range.start.min(end)..end].fill(f32::NEG_INFINITY);
        };
        if sampled.is_empty() {
            for &i in &t.suppress_begin {
                if let Some(l) = logits.get_mut(i as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        for &i in &t.suppress {
            if let Some(l) = logits.get_mut(i as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
        if !t.utf8.is_empty() {
            // Owed by what has been written so far, in this window.
            let owed = sampled.iter().fold(0, |owed, &i| match t.utf8.get(i as usize).map(|s| s[owed as usize]) {
                Some(next) if next != BROKEN => next,
                _ => 0,
            });
            for (l, steps) in logits.iter_mut().zip(&t.utf8) {
                if steps[owed as usize] == BROKEN {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        let tb = t.timestamp_begin as usize;
        if !self.timestamps {
            // `<|notimestamps|>` was in the prompt; a timestamp now would be
            // a token the transcript has no place for.
            ban(logits, tb..logits.len());
            return;
        }
        self.timestamps(logits, sampled);
    }

    /// The timestamp rules, in openai-whisper's order.
    ///
    /// 1. `<|notimestamps|>` is never produced: it is a prompt token.
    /// 2. Timestamps pair up: `<|1.00|> text <|2.40|><|2.40|> text …`. After
    ///    one that closes a segment (two in a row), text must follow; after
    ///    one that opens or closes a lone segment, another timestamp or the
    ///    end must.
    /// 3. They never go backwards, and a closing one is never the same as
    ///    the one before it.
    /// 4. The first token is a timestamp, no later than
    ///    `max_initial_timestamp` steps into the window.
    /// 5. If all the timestamps together are likelier than the likeliest
    ///    text token, it is a timestamp.
    pub fn timestamps(&self, logits: &mut [f32], sampled: &[u32]) {
        let t = self.tokens;
        let tb = t.timestamp_begin as usize;
        let n = logits.len();
        logits[t.no_timestamps as usize] = f32::NEG_INFINITY;

        let is_ts = |i: u32| i as usize >= tb;
        let last = sampled.last().is_some_and(|&i| is_ts(i));
        let penultimate = sampled.len() < 2 || is_ts(sampled[sampled.len() - 2]);
        if last {
            if penultimate {
                logits[tb..].fill(f32::NEG_INFINITY);
            } else {
                logits[..t.eot as usize].fill(f32::NEG_INFINITY);
            }
        }

        if let Some(&latest) = sampled.iter().rev().find(|&&i| is_ts(i)) {
            let floor = if last && !penultimate { latest } else { latest + 1 } as usize;
            logits[tb..floor.min(n)].fill(f32::NEG_INFINITY);
        }

        if sampled.is_empty() {
            logits[..tb].fill(f32::NEG_INFINITY);
            let last_allowed = tb + t.max_initial_timestamp;
            if last_allowed + 1 < n {
                logits[last_allowed + 1..].fill(f32::NEG_INFINITY);
            }
        }

        let lp = log_softmax(logits);
        let ts = log_sum_exp(&lp[tb..]);
        let text = lp[..tb].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if ts > text {
            logits[..tb].fill(f32::NEG_INFINITY);
        }
    }
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let lse = log_sum_exp(x);
    x.iter().map(|v| v - lse).collect()
}

fn log_sum_exp(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + x.iter().map(|v| (v - max).exp()).sum::<f32>().ln()
}

/// A token that cannot come next: see [`utf8`].
const BROKEN: u8 = u8::MAX;

/// For every token, and for each number of bytes a character being written
/// still owes (0 to 3), how many it owes after that token, or [`BROKEN`]
/// where the token cannot come next.
///
/// Whisper writes bytes, not characters. Its vocabulary is GPT-2's
/// byte-level BPE, and a character outside ASCII is two to four bytes that
/// need not be in one token: `ø` is C3 B8, and the model may write `Ã` (C3)
/// and then `¸` (B8) as two. Nothing makes the bytes it writes UTF-8. A
/// hallucinating window does write `å` (C3 A5) and then a lone B8, which is
/// no character, and the transcript shows `�`. Or it writes C3, a timestamp,
/// and B8 in the next segment, and both segments show one.
///
/// So the decoder is held to UTF-8, as the [`Rules`] hold it to timestamps
/// that pair up: a continuation byte only where one is owed, a new character
/// only where none is, and no end of text and no timestamp in the middle of
/// a character. Text that was UTF-8 already is decoded exactly as it was,
/// since nothing it chose is ruled out. Overlong forms and surrogates are
/// not checked for: a model does not write them by accident, and they would
/// come out as `�` and no worse.
pub fn utf8(tok: &Tokenizer) -> Vec<[u8; 4]> {
    if !matches!(tok.get_decoder(), Some(tokenizers::DecoderWrapper::ByteLevel(_))) {
        return Vec::new();
    }
    let unbyte = unbyte();
    let step = |bytes: &[u8], mut owed: u8| -> u8 {
        for &b in bytes {
            match (b, owed) {
                (0x80..=0xBF, 0) => return BROKEN,
                (0x80..=0xBF, _) => owed -= 1,
                (_, 1..) => return BROKEN,
                (0x00..=0x7F, _) => {}
                (0xC2..=0xDF, _) => owed = 1,
                (0xE0..=0xEF, _) => owed = 2,
                (0xF0..=0xF4, _) => owed = 3,
                _ => return BROKEN,
            }
        }
        owed
    };
    (0..tok.get_vocab_size(true) as u32)
        .map(|id| {
            // A token that is not bytes — end of text, a timestamp, a
            // language — leaves nothing owed, and so cannot come mid-character.
            let bytes: Option<Vec<u8>> = tok.id_to_token(id).and_then(|t| t.chars().map(|c| unbyte.get(&c).copied()).collect());
            match bytes {
                Some(b) if tok.get_added_vocabulary().simple_id_to_token(id).is_none() => [0, 1, 2, 3].map(|owed| step(&b, owed)),
                _ => [0, BROKEN, BROKEN, BROKEN],
            }
        })
        .collect()
}

/// GPT-2's byte-level alphabet, read backwards: the character a vocabulary
/// writes for each byte, to the byte.
///
/// Printable bytes stand for themselves (`!`–`~`, `¡`–`¬`, `®`–`ÿ`), and the
/// other 68 — controls, the space, the soft hyphen — are written as the
/// characters from U+0100 up, in byte order, so that a space is `Ġ`.
fn unbyte() -> &'static std::collections::HashMap<char, u8> {
    static TABLE: std::sync::OnceLock<std::collections::HashMap<char, u8>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        let mut table = std::collections::HashMap::new();
        let mut next = 256u32;
        for b in 0..=255u8 {
            let c = if printable(b) {
                char::from(b)
            } else {
                next += 1;
                char::from_u32(next - 1).expect("U+0100 to U+0143")
            };
            table.insert(c, b);
        }
        table
    })
}

/// A loaded Whisper with what it takes to drive it.
pub struct Transcriber {
    /// On the CPU or a GPU: see [`Model`].
    pub model: Box<dyn Model>,
    pub tokenizer: Tokenizer,
    pub tokens: Tokens,
}

impl Transcriber {
    /// `work`, where the model's work runs best ([`Model::within`]): on the
    /// CPU, its own threads, handed a window at a time.
    pub(crate) fn on_pool<T: Send>(&self, work: impl FnOnce() -> Res<T> + Send) -> Res<T> {
        let mut work = Some(work);
        let mut out: Option<Result<T, String>> = None;
        self.model.within(&mut || {
            if let Some(w) = work.take() {
                out = Some(w().map_err(|e| e.to_string()));
            }
        });
        out.ok_or("the model ran nothing it was given")?.map_err(Into::into)
    }

    /// The tokens a window's prompt is made of, after any context.
    pub(super) fn start(&self, language: u32, req: &Request) -> Vec<u32> {
        let t = &self.tokens;
        let mut s = vec![t.sot];
        if t.multilingual {
            s.push(language);
            s.push(if req.task == Task::Translate { t.translate } else { t.transcribe });
        }
        if !req.timestamps {
            s.push(t.no_timestamps);
        }
        s
    }

    /// The language spoken in a window, by the model's first step: given
    /// only `<|startoftranscript|>`, which language token is likeliest?
    pub fn detect(&self, audio: &Heard) -> Res<(String, f32)> {
        let mut said = self.model.said();
        let logits = self.model.decode(&[self.tokens.sot], &mut said, audio, &[0])?.remove(0);
        let ids: Vec<f32> = self.tokens.languages.iter().map(|&(_, i)| logits[i as usize]).collect();
        let lp = log_softmax(&ids);
        let (best, p) = lp.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        Ok((self.tokens.languages[best].0.clone(), p.exp()))
    }

    /// Decode one window at one temperature.
    ///
    /// `written` is what the window is taken to have written already: it
    /// goes in the prompt after `start`, the rules count it as written, and
    /// the window's tokens begin with it. A file's windows have none; a live
    /// one's is what it has settled ([`super::live`]). `lap` stops the
    /// window where it begins to write that many tokens a second time.
    pub(super) fn window(
        &self,
        audio: &Heard,
        context: &[u32],
        start: &[u32],
        written: &[u32],
        lap: Option<usize>,
        temperature: f32,
        sampler: &mut Sampler,
    ) -> Res<Window> {
        let t = &self.tokens;
        let ctx = self.model.config().text_ctx;
        let mut prompt = Vec::new();
        if !context.is_empty() {
            prompt.push(t.sot_prev);
            // At most half the decoder's positions are context, as in
            // openai-whisper: the rest are for what is being written.
            let keep = (ctx / 2 - 1).min(context.len());
            prompt.extend_from_slice(&context[context.len() - keep..]);
        }
        let sot_at = prompt.len();
        prompt.extend_from_slice(start);
        prompt.extend_from_slice(written);

        let mut said = self.model.said();
        // The scores at `<|startoftranscript|>` say whether there is speech
        // at all, and those at the end of the prompt what it begins with.
        let rows = match t.no_speech {
            Some(_) => vec![sot_at, prompt.len() - 1],
            None => vec![prompt.len() - 1],
        };
        let mut scores = self.model.decode(&prompt, &mut said, audio, &rows)?;
        let mut logits = scores.pop().expect("a row asked for");
        let no_speech_prob = match (t.no_speech, scores.pop()) {
            (Some(ns), Some(mut l)) => {
                crate::tensor::softmax_inplace(&mut l);
                l[ns as usize]
            }
            _ => 0.0,
        };

        let rules = Rules { tokens: t, timestamps: !start.contains(&t.no_timestamps) };
        sampler.temperature = temperature;
        let limit = (ctx / 2).min(ctx.saturating_sub(prompt.len()));
        let mut out = written.to_vec();
        let fresh = |out: &Vec<u32>| out.len() - written.len();
        let mut sum_logprob = 0.0f32;
        let mut ended = false;
        while fresh(&out) < limit {
            rules.apply(&mut logits, &out);
            let next = sampler.sample(&logits);
            sum_logprob += log_softmax(&logits)[next as usize];
            if next == t.eot {
                ended = true;
                break;
            }
            out.push(next);
            if let Some(lap) = lap {
                let n = out.len();
                if fresh(&out) >= lap && out[..n - 1].windows(lap).any(|w| w == &out[n - lap..]) {
                    out.truncate(n - lap);
                    break;
                }
            }
            if prompt.len() + fresh(&out) > ctx {
                break;
            }
            logits = self.model.decode(&[next], &mut said, audio, &[0])?.remove(0);
        }
        Ok(Window { avg_logprob: sum_logprob / (fresh(&out) + 1) as f32, tokens: out, no_speech_prob, temperature, ended })
    }

    /// One window, tried warmer until it comes out well or there are no
    /// temperatures left.
    fn window_with_fallback(&self, audio: &Heard, context: &[u32], start: &[u32], req: &Request, sampler: &mut Sampler) -> Res<Window> {
        let temps: Vec<f32> = match req.temperature {
            Some(t) => vec![t],
            None => TEMPERATURES.to_vec(),
        };
        let mut last = None;
        for &temp in &temps {
            let w = self.on_pool(|| self.window(audio, context, start, &[], None, temp, sampler))?;
            let unsure = w.avg_logprob < LOGPROB_THRESHOLD;
            // Silence is allowed to be unsure, and to ramble: it is skipped
            // below, not retried. So is a window that loops at every
            // temperature.
            let silent = w.no_speech_prob > NO_SPEECH_THRESHOLD;
            let good = (!w.looped(self.tokens.eot) && !unsure) || (silent && unsure);
            last = Some(w);
            if good {
                break;
            }
        }
        Ok(last.expect("at least one temperature"))
    }

    /// The text of `tokens`, timestamps and special tokens left out.
    pub fn text(&self, tokens: &[u32]) -> Res<String> {
        let words: Vec<u32> = tokens.iter().copied().filter(|&i| i < self.tokens.eot).collect();
        self.tokenizer.decode(&words, true).map_err(|e| e.to_string().into())
    }

    /// The language `req` asks for, checked: `None` to ask the model.
    pub(super) fn language_asked(&self, req: &Request) -> Res<Option<String>> {
        let t = &self.tokens;
        let language: Option<String> = match req.language.as_deref() {
            Some("auto") => None,
            Some(code) => Some(code.to_string()),
            None => t.default_language.clone(),
        };
        if let Some(code) = &language {
            if t.multilingual && t.language(code).is_none() {
                // The asker's to change, and a server answers it as such.
                return Err(Box::new(crate::image::Refused(format!("`{code}` is not a language this model knows"))));
            }
        }
        Ok(language)
    }

    /// The context a recording is read after: whatever the caller wanted,
    /// with a space before it as the model saw text in training.
    pub(super) fn prompted(&self, req: &Request) -> Res<Vec<u32>> {
        Ok(match req.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
            Some(p) => self.tokenizer.encode(format!(" {p}"), false).map_err(|e| e.to_string())?.get_ids().to_vec(),
            None => Vec::new(),
        })
    }

    /// Transcribe 16 kHz mono `audio`.
    ///
    /// `progress` is told after each window, and stops the transcription by
    /// returning `false`.
    pub fn transcribe(&self, audio: &[f32], req: &Request, progress: &mut dyn FnMut(&Progress) -> bool) -> Res<Transcript> {
        let n_mels = self.model.config().n_mels;
        let mel = self.on_pool(|| Ok(mel::log_mel(audio, mel::N_SAMPLES, n_mels)))?;
        let content = audio.len() / HOP;
        let duration = audio.len() as f32 / SAMPLE_RATE as f32;
        let t = &self.tokens;

        let mut sampler = Sampler::new(0.0, 0, 1.0, req.seed.unwrap_or(0));
        let mut language = self.language_asked(req)?;
        let mut context = self.prompted(req)?;
        let mut reset_at = 0;
        // Where the words timed so far stopped: see `align::tidy`.
        let mut last_speech = 0.0;

        let mut segments: Vec<Segment> = Vec::new();
        let mut seek = 0;
        while seek < content {
            let offset = (seek * HOP) as f32 / SAMPLE_RATE as f32;
            let size = (content - seek).min(N_FRAMES);
            let window_len = (size * HOP) as f32 / SAMPLE_RATE as f32;
            let heard = self.on_pool(|| self.model.hear(&mel::window(&mel, seek)))?;

            let code = match &language {
                Some(c) => c.clone(),
                None if t.multilingual => {
                    let (c, _) = self.on_pool(|| self.detect(&heard))?;
                    language = Some(c.clone());
                    c
                }
                None => "en".to_string(),
            };
            let start = self.start(t.language(&code).unwrap_or(0), req);
            let context_now = if req.no_context { &[][..] } else { &context[reset_at..] };
            let w = self.window_with_fallback(&heard, context_now, &start, req, &mut sampler)?;

            if w.skipped(t.eot) {
                seek += size;
                if !progress(&Progress { done: ((seek * HOP) as f32 / SAMPLE_RATE as f32).min(duration), total: duration, segments: Vec::new() }) {
                    return Err("stopped".into());
                }
                continue;
            }

            let before = segments.len();
            let mut add = |start: f32, end: f32, tokens: &[u32]| -> Res<()> {
                let text = self.text(tokens)?;
                if !text.trim().is_empty() {
                    segments.push(Segment {
                        start,
                        end,
                        text,
                        tokens: tokens.to_vec(),
                        temperature: w.temperature,
                        avg_logprob: w.avg_logprob,
                        no_speech_prob: w.no_speech_prob,
                        words: Vec::new(),
                        speaker: None,
                    });
                }
                Ok(())
            };
            let time = |i: u32| offset + i.saturating_sub(t.timestamp_begin) as f32 * TIME_STEP;
            let tokens = &w.tokens;
            let is_ts = |i: u32| i >= t.timestamp_begin;
            let lone_end = tokens.len() >= 2 && !is_ts(tokens[tokens.len() - 2]) && is_ts(tokens[tokens.len() - 1]);
            // Where two timestamps sit side by side: one segment's end and the
            // next one's start.
            let mut cuts: Vec<usize> = (1..tokens.len()).filter(|&i| is_ts(tokens[i]) && is_ts(tokens[i - 1])).collect();
            if !cuts.is_empty() {
                if lone_end {
                    cuts.push(tokens.len());
                }
                let mut from = 0;
                for &cut in &cuts {
                    let piece = &tokens[from..cut];
                    if piece.len() >= 2 {
                        add(time(piece[0]), time(piece[piece.len() - 1]), piece)?;
                    }
                    from = cut;
                }
                if lone_end {
                    seek += size;
                } else {
                    // Start the next window where the last whole segment
                    // ended, so whatever was cut off is heard again whole.
                    let last_ts = tokens[from - 1] - t.timestamp_begin;
                    seek += (last_ts as usize * FRAMES_PER_STEP).max(1);
                }
            } else {
                let mut end = offset + window_len;
                if let Some(&last) = tokens.iter().rev().find(|&&i| is_ts(i)) {
                    if last != t.timestamp_begin {
                        end = time(last);
                    }
                }
                add(offset, end, tokens)?;
                seek += size;
            }

            if req.words {
                // The prompt as far as the task: the alignment pass puts
                // `<|notimestamps|>` after it whatever this window had.
                let sot: Vec<u32> = start.iter().copied().filter(|&i| i != t.no_timestamps).collect();
                self.time_words(&heard, &sot, &mut segments[before..], size, offset, &mut last_speech)?;
            }
            context.extend_from_slice(tokens);
            if w.temperature > 0.5 {
                // A window that needed a lot of randomness is a poor guide
                // to the next one.
                reset_at = context.len();
            }
            let added = segments[before..].to_vec();
            if !progress(&Progress { done: ((seek * HOP) as f32 / SAMPLE_RATE as f32).min(duration), total: duration, segments: added }) {
                return Err("stopped".into());
            }
        }

        let text = segments.iter().map(|s| s.text.as_str()).collect::<String>();
        Ok(Transcript { text, language: language.unwrap_or_else(|| "en".into()), duration, segments })
    }
}

// ---------------------------------------------------------------------------
// What a transcript is written out as
// ---------------------------------------------------------------------------

/// The formats OpenAI's transcription endpoint answers in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    #[default]
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
    /// OpenAI's for a transcript with speakers: its segments, each with who
    /// says it.
    DiarizedJson,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        Some(match s {
            "json" => Format::Json,
            "text" => Format::Text,
            "verbose_json" => Format::VerboseJson,
            "srt" => Format::Srt,
            "vtt" => Format::Vtt,
            "diarized_json" => Format::DiarizedJson,
            _ => return None,
        })
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Format::Json | Format::VerboseJson | Format::DiarizedJson => "application/json",
            Format::Text => "text/plain; charset=utf-8",
            Format::Srt => "application/x-subrip; charset=utf-8",
            Format::Vtt => "text/vtt; charset=utf-8",
        }
    }
}

impl Transcript {
    /// The transcript in `format`. `task` is what `verbose_json` reports it
    /// was asked to do.
    ///
    /// With speakers, each format says who: `text` and `json` a turn a line
    /// after its speaker, `A: …`; `srt` the same before each cue's text;
    /// `vtt` in WebVTT's voice span, `<v A>`; `verbose_json` a `speaker` on
    /// every segment and word. `diarized_json` is OpenAI's.
    pub fn render(&self, format: Format, task: Task) -> String {
        let text = match self.has_speakers() {
            true => self.turns(),
            false => self.text.trim().to_string(),
        };
        match format {
            Format::Text => format!("{text}\n"),
            Format::Json => serde_json::json!({ "text": text }).to_string(),
            Format::VerboseJson => self.verbose(task).to_string(),
            Format::Srt => self.cues(",", true),
            Format::Vtt => format!("WEBVTT\n\n{}", self.cues(".", false)),
            Format::DiarizedJson => self.diarized(task).to_string(),
        }
    }

    pub fn has_speakers(&self) -> bool {
        self.segments.iter().any(|s| s.speaker.is_some())
    }

    /// A turn a line, `A: …`: the segments one speaker says in a row, joined.
    fn turns(&self) -> String {
        let mut lines: Vec<(&str, String)> = Vec::new();
        for s in &self.segments {
            let who = s.speaker.as_deref().unwrap_or("?");
            match lines.last_mut() {
                Some((w, text)) if *w == who => text.push_str(&s.text),
                _ => lines.push((who, s.text.clone())),
            }
        }
        lines.iter().map(|(w, t)| format!("{w}: {}", t.trim())).collect::<Vec<_>>().join("\n")
    }

    /// OpenAI's `diarized_json` (`TranscriptionDiarized`).
    pub fn diarized(&self, task: Task) -> serde_json::Value {
        let segments: Vec<serde_json::Value> = self
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({
                    "type": "transcript.text.segment",
                    "id": format!("seg_{i}"),
                    "start": round2(s.start),
                    "end": round2(s.end),
                    "text": s.text.trim(),
                    "speaker": s.speaker.as_deref().unwrap_or("?"),
                })
            })
            .collect();
        serde_json::json!({
            "task": if task == Task::Translate { "translate" } else { "transcribe" },
            "duration": round2(self.duration),
            "text": self.turns(),
            "segments": segments,
        })
    }

    pub fn verbose(&self, task: Task) -> serde_json::Value {
        let segments: Vec<serde_json::Value> = self
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut v = serde_json::json!({
                    "id": i,
                    "start": round2(s.start),
                    "end": round2(s.end),
                    "text": s.text,
                    "tokens": s.tokens,
                    "temperature": s.temperature,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                });
                if let Some(who) = &s.speaker {
                    v["speaker"] = who.as_str().into();
                }
                v
            })
            .collect();
        let mut out = serde_json::json!({
            "task": if task == Task::Translate { "translate" } else { "transcribe" },
            "language": self.language,
            "duration": round2(self.duration),
            "text": self.text.trim(),
            "segments": segments,
        });
        // OpenAI's `timestamp_granularities[]=word`: every word, beside the
        // segments rather than in them, without the space before it.
        if self.segments.iter().any(|s| !s.words.is_empty()) {
            let words: Vec<serde_json::Value> = self
                .segments
                .iter()
                .flat_map(|s| s.words.iter().map(move |w| (w, &s.speaker)))
                .map(|(w, who)| {
                    let mut v = serde_json::json!({ "word": w.word.trim(), "start": round2(w.start), "end": round2(w.end) });
                    if let Some(who) = who {
                        v["speaker"] = who.as_str().into();
                    }
                    v
                })
                .collect();
            out["words"] = words.into();
        }
        out
    }

    /// Numbered (SRT) or bare (VTT) cues, one per segment.
    fn cues(&self, comma: &str, numbered: bool) -> String {
        let mut out = String::new();
        for (i, s) in self.segments.iter().enumerate() {
            if numbered {
                out.push_str(&format!("{}\n", i + 1));
            }
            let text = match (&s.speaker, numbered) {
                (None, _) => s.text.trim().to_string(),
                (Some(who), true) => format!("{who}: {}", s.text.trim()),
                (Some(who), false) => format!("<v {who}>{}", s.text.trim()),
            };
            out.push_str(&format!("{} --> {}\n{text}\n\n", clock(s.start, comma), clock(s.end, comma)));
        }
        out
    }
}

fn round2(x: f32) -> f64 {
    (x as f64 * 100.0).round() / 100.0
}

/// `hh:mm:ss,mmm` — with a full stop instead of the comma for VTT.
pub fn clock(seconds: f32, comma: &str) -> String {
    let ms = (seconds.max(0.0) as f64 * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{comma}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens() -> Tokens {
        // A miniature vocabulary: text 0..10, eot 10, specials 11..15,
        // timestamps from 20.
        Tokens {
            eot: 10,
            sot: 11,
            sot_prev: 12,
            transcribe: 13,
            translate: 14,
            no_timestamps: 15,
            no_speech: Some(16),
            timestamp_begin: 20,
            languages: vec![],
            suppress: vec![3, 11, 12, 13, 14, 16],
            suppress_begin: vec![0, 10],
            default_language: None,
            max_initial_timestamp: 2,
            multilingual: true,
            utf8: Vec::new(),
            heads: vec![(0, 0)],
            heads_guessed: false,
        }
    }

    fn allowed(logits: &[f32]) -> Vec<usize> {
        logits.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect()
    }

    #[test]
    fn a_window_opens_with_an_early_timestamp() {
        let t = tokens();
        let rules = Rules { tokens: &t, timestamps: true };
        let mut l = vec![0.0f32; 30];
        rules.apply(&mut l, &[]);
        assert_eq!(allowed(&l), vec![20, 21, 22]);
    }

    #[test]
    fn timestamps_pair_up_and_never_go_back() {
        let t = tokens();
        let rules = Rules { tokens: &t, timestamps: true };
        // After `<|t5|> word`: text or a timestamp, but none before 5.
        let mut l = vec![0.0f32; 30];
        rules.apply(&mut l, &[25, 4]);
        assert!(allowed(&l).iter().all(|&i| i >= 25 || i < 10), "{:?}", allowed(&l));
        // After `<|t5|> word <|t7|>`, a lone closing timestamp: another
        // timestamp (from 7) or the end, never text.
        let mut l = vec![0.0f32; 30];
        rules.apply(&mut l, &[25, 4, 27]);
        assert!(allowed(&l).iter().all(|&i| i >= 27 || i == 10), "{:?}", allowed(&l));
        // After `… <|t7|><|t7|>`: text must follow.
        let mut l = vec![0.0f32; 30];
        rules.apply(&mut l, &[25, 4, 27, 27]);
        assert!(allowed(&l).iter().all(|&i| i < 20), "{:?}", allowed(&l));
    }

    #[test]
    fn without_timestamps_none_is_produced_and_the_first_is_not_blank() {
        let t = tokens();
        let rules = Rules { tokens: &t, timestamps: false };
        let mut l = vec![0.0f32; 30];
        rules.apply(&mut l, &[]);
        assert_eq!(allowed(&l), vec![1, 2, 4, 5, 6, 7, 8, 9, 15, 17, 18, 19]);
    }

    /// In the miniature vocabulary, token 4 is the lead byte C3 and token 5
    /// the continuation B8, so 4 then 5 is `ø`; everything else is ASCII or
    /// not bytes at all.
    fn with_bytes() -> Tokens {
        let mut t = tokens();
        t.utf8 = (0..30).map(|i| match i {
            4 => [1, BROKEN, BROKEN, BROKEN],
            5 => [BROKEN, 0, 1, 2],
            _ => [0, BROKEN, BROKEN, BROKEN],
        }).collect();
        t
    }

    #[test]
    fn a_character_begun_is_finished_before_anything_else() {
        let t = with_bytes();
        for timestamps in [false, true] {
            let rules = Rules { tokens: &t, timestamps };
            // After the lead byte only its continuation may come: not text,
            // not the end, not a timestamp.
            let mut l = vec![0.0f32; 30];
            rules.apply(&mut l, &[20, 4]);
            assert_eq!(allowed(&l), vec![5], "timestamps {timestamps}");
            // And once it is whole, a continuation is never the first byte
            // of anything, while what else may come is the other rules'
            // business again.
            let mut l = vec![0.0f32; 30];
            rules.apply(&mut l, &[20, 4, 5]);
            assert!(!allowed(&l).contains(&5) && !allowed(&l).is_empty(), "{:?}", allowed(&l));
        }
    }

    /// NB-Whisper's own vocabulary, where it is on this machine (it is never
    /// downloaded here).
    #[test]
    fn the_table_knows_whispers_bytes() {
        let Some(path) = crate::weights::local_file("NbAiLab/nb-whisper-tiny", "tokenizer.json") else {
            eprintln!("NbAiLab/nb-whisper-tiny is not on this machine: skipped");
            return;
        };
        let tok = Tokenizer::from_file(path).unwrap();
        let table = utf8(&tok);
        assert_eq!(table.len(), tok.get_vocab_size(true));
        let id = |s: &str| tok.token_to_id(s).unwrap() as usize;
        // `Ã` begins `ø` and owes one byte; `¸` is that byte, and nothing
        // else; `Ã¥` is a whole `å`.
        assert_eq!(table[id("Ã")][0], 1);
        assert_eq!(table[id("¸")][0], BROKEN);
        assert_eq!(table[id("¸")][1], 0);
        assert_eq!(table[id("Ã¥")][0], 0);
        assert_eq!(table[id("Ġs")][0], 0);
        assert_eq!(table[id("Ġs")][1], BROKEN);
        for special in ["<|endoftext|>", "<|0.00|>", "<|30.00|>", "<|no|>"] {
            assert_eq!(table[id(special)], [0, BROKEN, BROKEN, BROKEN], "{special}");
        }
        // Every token of ordinary text is allowed where a character is not
        // half written, so text that was UTF-8 is decoded as it was.
        let text = " Nordmenn er sørlendinger, trøndere – og «folk» fra Ålesund. 1 000 €!";
        let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
        let owed = ids.iter().fold(0u8, |owed, &i| {
            let next = table[i as usize][owed as usize];
            assert_ne!(next, BROKEN, "token {i} after {owed} owed");
            next
        });
        assert_eq!(owed, 0);
    }

    #[test]
    fn silence_and_loops_are_skipped_and_dense_speech_is_not() {
        const EOT: u32 = 50257;
        let w = |tokens: Vec<u32>, ended, avg_logprob, no_speech_prob| Window { tokens, avg_logprob, no_speech_prob, temperature: 0.0, ended };
        // 224 tokens of words that do not come round again, with a
        // timestamp pair every 20: speech too dense for the limit.
        let mut x = 7u32;
        let mut dense: Vec<u32> = (0..224).map(|i| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            if i % 20 == 19 { EOT + 100 + i } else { 300 + (x >> 8) % 20_000 }
        }).collect();
        assert!(repeated(&dense, EOT).1 < 0.05, "{:?}", repeated(&dense, EOT));
        assert!(!w(dense.clone(), false, -0.25, 0.0).looped(EOT));
        assert!(!w(dense.clone(), false, -0.25, 0.0).skipped(EOT));
        // The same words ending in a phrase said over and over.
        dense.truncate(40);
        let mut circling = dense.clone();
        while circling.len() < 224 {
            circling.extend([1000, 2000, 3000, 4000, 5000, 6000, 13]);
        }
        circling.truncate(224);
        assert!(w(circling, false, -0.1, 0.0).skipped(EOT));
        // A page of spaces and stray letters, with confidence, as written
        // over a kitchen's noise.
        let page: Vec<u32> = (0..224).map(|i| if i % 9 == 4 { [258, 275, 256, 385][i % 4] } else { 220 }).collect();
        assert!(w(page, false, -0.03, 0.0).skipped(EOT));
        // Nothing but a special token, over and over: no text at all.
        assert!(w(vec![EOT + 31; 224], false, -3.25, 0.0).skipped(EOT));
        // A window that ended is kept when it repeats itself a little;
        // fallback has had its say on how sure it is.
        let ja = [1000, 13, 1000, 13, 1000, 13, 1000, 13, 1000, 13].to_vec();
        assert!(!w(ja.clone(), true, -0.3, 0.0).skipped(EOT));
        assert!(!w(ja, true, -1.5, 0.0).skipped(EOT));
        // Not when it is a confident page of one-second "Okay."s, each
        // segment timed, as AMI's meetings came out.
        let okay: Vec<u32> = (0..30).flat_map(|i| [EOT + 100 + 50 * i, 4120, 13, EOT + 150 + 50 * i]).collect();
        assert!(w(okay, true, -0.1, 0.0).skipped(EOT));
        // A likely `<|nospeech|>` alone is not enough, and with an unsure
        // window it is silence, which may ramble too.
        assert!(!w(vec![1000], true, -0.3, 0.9).skipped(EOT));
        assert!(w(vec![1000], true, -1.5, 0.9).skipped(EOT));
        assert!(w(vec![1000; 224], false, -1.5, 0.9).skipped(EOT));
    }

    #[test]
    fn subtitles_are_numbered_and_timed() {
        let seg = |start, end, text: &str| Segment {
            start,
            end,
            text: text.into(),
            tokens: vec![],
            temperature: 0.0,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            words: vec![],
            speaker: None,
        };
        let tr = Transcript {
            text: " Hei. Hallo.".into(),
            language: "no".into(),
            duration: 3725.5,
            segments: vec![seg(0.0, 1.5, " Hei."), seg(3723.25, 3725.5, " Hallo.")],
        };
        assert_eq!(tr.render(Format::Srt, Task::Transcribe), "1\n00:00:00,000 --> 00:00:01,500\nHei.\n\n2\n01:02:03,250 --> 01:02:05,500\nHallo.\n\n");
        assert!(tr.render(Format::Vtt, Task::Transcribe).starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHei."));
        assert_eq!(tr.render(Format::Text, Task::Transcribe), "Hei. Hallo.\n");
    }

    /// OpenAI's `words`: beside the segments, every word without the space
    /// before it, at two decimals; and no key at all when none were asked.
    #[test]
    fn verbose_json_carries_the_words_when_there_are_any() {
        let word = |w: &str, start, end| Word { word: w.into(), start, end };
        let mut seg = Segment { start: 0.0, end: 1.5, text: " Hei, sa han.".into(), tokens: vec![], temperature: 0.0, avg_logprob: 0.0, no_speech_prob: 0.0, words: vec![], speaker: None };
        let mut tr = Transcript { text: seg.text.clone(), language: "no".into(), duration: 1.5, segments: vec![seg.clone()] };
        assert!(tr.verbose(Task::Transcribe).get("words").is_none());
        seg.words = vec![word(" Hei,", 0.0, 0.42), word(" sa", 0.42, 0.7000001), word(" han.", 0.7, 1.24)];
        tr.segments = vec![seg];
        let v = tr.verbose(Task::Transcribe);
        assert_eq!(v["words"], serde_json::json!([
            { "word": "Hei,", "start": 0.0, "end": 0.42 },
            { "word": "sa", "start": 0.42, "end": 0.7 },
            { "word": "han.", "start": 0.7, "end": 1.24 },
        ]));
        assert!(v["segments"][0].get("words").is_none(), "words go beside the segments, as OpenAI's");
    }

    /// Each format with speakers, byte for byte, from one hand-made
    /// transcript: two turns of A's, one of B's between them.
    #[test]
    fn every_format_says_who_speaks() {
        let word = |w: &str, start, end| Word { word: w.into(), start, end };
        let seg = |start, end, text: &str, who: &str, words| Segment {
            start,
            end,
            text: text.into(),
            tokens: vec![],
            temperature: 0.0,
            avg_logprob: -0.25,
            no_speech_prob: 0.5,
            words,
            speaker: Some(who.into()),
        };
        let tr = Transcript {
            text: " Hva synes du? Bra. Takk.".into(),
            language: "no".into(),
            duration: 4.0,
            segments: vec![
                seg(0.0, 1.4, " Hva synes du?", "A", vec![word(" Hva", 0.1, 0.4), word(" synes", 0.5, 0.9), word(" du?", 1.0, 1.4)]),
                seg(2.0, 2.6, " Bra.", "B", vec![word(" Bra.", 2.0, 2.6)]),
                seg(3.0, 4.0, " Takk.", "A", vec![word(" Takk.", 3.0, 3.6)]),
            ],
        };
        let render = |f| tr.render(f, Task::Transcribe);
        assert_eq!(render(Format::Text), "A: Hva synes du?\nB: Bra.\nA: Takk.\n");
        assert_eq!(render(Format::Json), r#"{"text":"A: Hva synes du?\nB: Bra.\nA: Takk."}"#);
        assert_eq!(
            render(Format::Srt),
            "1\n00:00:00,000 --> 00:00:01,400\nA: Hva synes du?\n\n2\n00:00:02,000 --> 00:00:02,600\nB: Bra.\n\n3\n00:00:03,000 --> 00:00:04,000\nA: Takk.\n\n"
        );
        assert_eq!(
            render(Format::Vtt),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.400\n<v A>Hva synes du?\n\n00:00:02.000 --> 00:00:02.600\n<v B>Bra.\n\n00:00:03.000 --> 00:00:04.000\n<v A>Takk.\n\n"
        );
        assert_eq!(
            render(Format::DiarizedJson),
            concat!(
                r#"{"task":"transcribe","duration":4.0,"text":"A: Hva synes du?\nB: Bra.\nA: Takk.","segments":["#,
                r#"{"type":"transcript.text.segment","id":"seg_0","start":0.0,"end":1.4,"text":"Hva synes du?","speaker":"A"},"#,
                r#"{"type":"transcript.text.segment","id":"seg_1","start":2.0,"end":2.6,"text":"Bra.","speaker":"B"},"#,
                r#"{"type":"transcript.text.segment","id":"seg_2","start":3.0,"end":4.0,"text":"Takk.","speaker":"A"}]}"#,
            )
        );
        let v = tr.verbose(Task::Transcribe);
        assert_eq!(v["text"], "Hva synes du? Bra. Takk.", "verbose_json's text stays the words alone");
        assert_eq!(v["segments"].as_array().unwrap().iter().map(|s| s["speaker"].as_str().unwrap()).collect::<Vec<_>>(), ["A", "B", "A"]);
        assert_eq!(v["words"][2], serde_json::json!({ "word": "du?", "start": 1.0, "end": 1.4, "speaker": "A" }));
        assert_eq!(v["words"][3]["speaker"], "B");
    }
}
