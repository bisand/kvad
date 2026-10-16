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
//! below −1), or that never stopped — repeating a phrase until the token
//! limit, the usual way a Whisper fails — is decoded again at a higher
//! temperature, up to five times. openai-whisper also measures repetition by
//! how well the text gzips; that needs a compressor this crate does not have,
//! and a window caught in a loop runs into the limit regardless.

use super::mel::{self, HOP, N_FRAMES, SAMPLE_RATE};
use super::whisper::{AudioCache, TextCache, Whisper};
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
struct Window {
    /// What was produced, without the end-of-text.
    tokens: Vec<u32>,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
    /// Whether it stopped by itself rather than at the token limit.
    ended: bool,
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

/// A loaded Whisper with what it takes to drive it.
pub struct Transcriber {
    pub model: Whisper,
    pub tokenizer: Tokenizer,
    pub tokens: Tokens,
    /// The threads its matmuls run on. Work is handed to it whole, a window
    /// at a time: a parallel loop started from a thread outside a pool takes
    /// rayon's slow path at every call, and a decoder step is dozens of
    /// them. See [`crate::model::threads`].
    pub pool: rayon::ThreadPool,
    /// What its matrices were loaded at, for a server's account of it.
    pub precision: crate::quant::Precision,
}

impl Transcriber {
    /// `work`, on this transcriber's threads.
    fn on_pool<T: Send>(&self, work: impl FnOnce() -> Res<T> + Send) -> Res<T> {
        self.pool.install(|| work().map_err(|e| e.to_string())).map_err(Into::into)
    }

    /// The tokens a window's prompt is made of, after any context.
    fn start(&self, language: u32, req: &Request) -> Vec<u32> {
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
    pub fn detect(&self, audio: &AudioCache) -> Res<(String, f32)> {
        let mut cache = TextCache::new(self.model.config.decoder_layers);
        let h = self.model.decode(&[self.tokens.sot], &mut cache, audio)?;
        let logits = self.model.logits(&h);
        let ids: Vec<f32> = self.tokens.languages.iter().map(|&(_, i)| logits[i as usize]).collect();
        let lp = log_softmax(&ids);
        let (best, p) = lp.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        Ok((self.tokens.languages[best].0.clone(), p.exp()))
    }

    /// Decode one window at one temperature.
    fn window(&self, audio: &AudioCache, context: &[u32], start: &[u32], temperature: f32, sampler: &mut Sampler) -> Res<Window> {
        let t = &self.tokens;
        let ctx = self.model.config.text_ctx;
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

        let d = self.model.config.d;
        let mut cache = TextCache::new(self.model.config.decoder_layers);
        let h = self.model.decode(&prompt, &mut cache, audio)?;
        let no_speech_prob = match t.no_speech {
            Some(ns) => {
                let mut l = self.model.logits(&h[sot_at * d..(sot_at + 1) * d]);
                crate::tensor::softmax_inplace(&mut l);
                l[ns as usize]
            }
            None => 0.0,
        };
        let mut logits = self.model.logits(&h[(prompt.len() - 1) * d..]);

        let rules = Rules { tokens: t, timestamps: !start.contains(&t.no_timestamps) };
        sampler.temperature = temperature;
        let limit = (ctx / 2).min(ctx - prompt.len());
        let mut out = Vec::new();
        let mut sum_logprob = 0.0f32;
        let mut ended = false;
        while out.len() < limit {
            rules.apply(&mut logits, &out);
            let next = sampler.sample(&logits);
            sum_logprob += log_softmax(&logits)[next as usize];
            if next == t.eot {
                ended = true;
                break;
            }
            out.push(next);
            if cache.len() >= ctx {
                break;
            }
            logits = self.model.logits(&self.model.decode(&[next], &mut cache, audio)?);
        }
        Ok(Window { avg_logprob: sum_logprob / (out.len() + 1) as f32, tokens: out, no_speech_prob, temperature, ended })
    }

    /// One window, tried warmer until it comes out well or there are no
    /// temperatures left.
    fn window_with_fallback(&self, audio: &AudioCache, context: &[u32], start: &[u32], req: &Request, sampler: &mut Sampler) -> Res<Window> {
        let temps: Vec<f32> = match req.temperature {
            Some(t) => vec![t],
            None => TEMPERATURES.to_vec(),
        };
        let mut last = None;
        for &temp in &temps {
            let w = self.on_pool(|| self.window(audio, context, start, temp, sampler))?;
            let unsure = w.avg_logprob < LOGPROB_THRESHOLD;
            // Silence is allowed to be unsure, and to ramble: it is skipped
            // below, not retried.
            let silent = w.no_speech_prob > NO_SPEECH_THRESHOLD;
            let good = (w.ended && !unsure) || (silent && unsure);
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

    /// Transcribe 16 kHz mono `audio`.
    ///
    /// `progress` is told after each window, and stops the transcription by
    /// returning `false`.
    pub fn transcribe(&self, audio: &[f32], req: &Request, progress: &mut dyn FnMut(&Progress) -> bool) -> Res<Transcript> {
        let n_mels = self.model.config.n_mels;
        let mel = self.on_pool(|| Ok(mel::log_mel(audio, mel::N_SAMPLES, n_mels)))?;
        let content = audio.len() / HOP;
        let duration = audio.len() as f32 / SAMPLE_RATE as f32;
        let t = &self.tokens;

        let mut sampler = Sampler::new(0.0, 0, 1.0, req.seed.unwrap_or(0));
        let mut language: Option<String> = match req.language.as_deref() {
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

        // The prompt's context starts as whatever the caller wanted the
        // recording read after, with a space before it as the model saw text
        // in training.
        let mut context: Vec<u32> = match req.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
            Some(p) => self.tokenizer.encode(format!(" {p}"), false).map_err(|e| e.to_string())?.get_ids().to_vec(),
            None => Vec::new(),
        };
        let mut reset_at = 0;

        let mut segments: Vec<Segment> = Vec::new();
        let mut seek = 0;
        while seek < content {
            let offset = (seek * HOP) as f32 / SAMPLE_RATE as f32;
            let size = (content - seek).min(N_FRAMES);
            let window_len = (size * HOP) as f32 / SAMPLE_RATE as f32;
            let heard = self.on_pool(|| Ok(self.model.listen(&self.model.encode(&mel::window(&mel, seek)))))?;

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

            if w.no_speech_prob > NO_SPEECH_THRESHOLD && w.avg_logprob < LOGPROB_THRESHOLD {
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
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        Some(match s {
            "json" => Format::Json,
            "text" => Format::Text,
            "verbose_json" => Format::VerboseJson,
            "srt" => Format::Srt,
            "vtt" => Format::Vtt,
            _ => return None,
        })
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Format::Json | Format::VerboseJson => "application/json",
            Format::Text => "text/plain; charset=utf-8",
            Format::Srt => "application/x-subrip; charset=utf-8",
            Format::Vtt => "text/vtt; charset=utf-8",
        }
    }
}

impl Transcript {
    /// The transcript in `format`. `task` is what `verbose_json` reports it
    /// was asked to do.
    pub fn render(&self, format: Format, task: Task) -> String {
        match format {
            Format::Text => format!("{}\n", self.text.trim()),
            Format::Json => serde_json::json!({ "text": self.text.trim() }).to_string(),
            Format::VerboseJson => self.verbose(task).to_string(),
            Format::Srt => self.cues(",", true),
            Format::Vtt => format!("WEBVTT\n\n{}", self.cues(".", false)),
        }
    }

    pub fn verbose(&self, task: Task) -> serde_json::Value {
        let segments: Vec<serde_json::Value> = self
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({
                    "id": i,
                    "start": round2(s.start),
                    "end": round2(s.end),
                    "text": s.text,
                    "tokens": s.tokens,
                    "temperature": s.temperature,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                })
            })
            .collect();
        serde_json::json!({
            "task": if task == Task::Translate { "translate" } else { "transcribe" },
            "language": self.language,
            "duration": round2(self.duration),
            "text": self.text.trim(),
            "segments": segments,
        })
    }

    /// Numbered (SRT) or bare (VTT) cues, one per segment.
    fn cues(&self, comma: &str, numbered: bool) -> String {
        let mut out = String::new();
        for (i, s) in self.segments.iter().enumerate() {
            if numbered {
                out.push_str(&format!("{}\n", i + 1));
            }
            out.push_str(&format!("{} --> {}\n{}\n\n", clock(s.start, comma), clock(s.end, comma), s.text.trim()));
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
}
