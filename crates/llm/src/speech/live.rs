//! Live transcription: audio that is still arriving, written down as it is
//! spoken.
//!
//! Whisper hears 30 seconds at a time, and its encoder takes the full 30
//! whether or not they hold sound. So a [`Live`] session decodes its window
//! again as the window grows, about once a second (the caller's clock, not
//! this module's), and each [`Live::update`] says what is now certain and
//! what is still a guess. The design is `docs/speech-plan.md`'s, *Live, from
//! the phone*, as `live_speed` measured it:
//!
//! 1. An update hears the window from its start, padded to 30 s as always.
//!    The tokens already settled go to the decoder as written, so only the
//!    tail is decoded a step at a time. The tail is decoded *plain*, with
//!    `<|notimestamps|>`: Whisper leaves out a segment that runs into the
//!    end of its window, and the one being spoken always does.
//! 2. A word is *settled* once two updates in a row write it, and every
//!    word before it, the same: LocalAgreement-2, from Macháček et al.'s
//!    whisper_streaming (2023). Settled text never changes. The rest is the
//!    *partial*, which the next update replaces.
//! 3. A window holding more than [`CUT`] seconds is finished where a
//!    settled segment ends and the next begins, two timestamps side by
//!    side, as a file's windows are. The plain tail has none, so one timed
//!    decode of what the update has just heard finds them, and its words
//!    are matched to the settled ones. A window that reaches 30 s with
//!    nowhere to cut begins again at its newest 30 s, losing its partial.
//! 4. A window the model takes for silence, with nothing settled in it, is
//!    let go of but for its last second, so that silence never fills one.
//!
//! The file path's rules hold here as they do there, since the window is
//! decoded by the same [`Transcriber::window`]: what may be said, UTF-8 kept
//! whole. A tail that starts to write again the eight tokens it has just
//! written is stopped there, which stands in for the file path's loop test
//! (that one needs the window to run into the token limit, and a live
//! update must not).

use super::mel::{self, SAMPLE_RATE};
use super::transcribe::{Request, Transcriber, TIME_STEP};
use crate::sampler::Sampler;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A window holding more than this many seconds is finished at its last
/// settled segment end. whisper_streaming trims at 15; the encoder costs
/// the same whatever the window holds, and a longer window only lengthens
/// the prompt, so 20 leaves more segments to cut at.
pub const CUT: f32 = 20.0;

/// How many tokens written again make a loop.
const LAP: usize = 8;

/// What of a silent window is kept: enough for a word just beginning.
const QUIET_KEPT: usize = SAMPLE_RATE;

/// One live transcription: what has been heard, what is settled, and the
/// last guess at the rest.
///
/// It holds no model. Each [`Live::update`] is handed the [`Transcriber`],
/// so a server can keep many sessions and run each one's update where the
/// model is, one at a time.
pub struct Live {
    request: Request,
    /// The language token, once known: asked for, the checkpoint's own, or
    /// detected at the first update.
    language: Option<(String, u32)>,
    /// The window's audio, from its start.
    window: Vec<f32>,
    /// Samples before the window's start, since the session began.
    before: usize,
    /// The finished windows' tokens, as a file's windows have them: the
    /// context each window is read after.
    context: Vec<u32>,
    /// This window's settled tokens.
    settled: Vec<u32>,
    /// The last update's tokens, settled and partial.
    last: Vec<u32>,
    sampler: Sampler,
}

/// What one update came to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Update {
    /// Text settled by this update, to be added to what was settled before
    /// it. It never changes once sent.
    pub settled: String,
    /// The guess at what follows the settled text. Each replaces the last.
    pub partial: String,
    /// How far into the audio this update heard, in seconds since the
    /// session began.
    pub heard: f32,
    /// The window was finished at a segment's end ([`CUT`]).
    pub cut: bool,
    /// The window reached 30 s with nowhere to cut, and began again.
    pub forced: bool,
    /// The model took the window for silence.
    pub silent: bool,
}

impl Live {
    /// A session for `request`: its language, task and prompt. Timestamps
    /// and temperature are the session's own business (plain and greedy),
    /// and are not read.
    pub fn new(tr: &Transcriber, request: &Request) -> Res<Live> {
        let language = match tr.language_asked(request)? {
            Some(code) => Some((code.clone(), tr.tokens.language(&code).unwrap_or(0))),
            None if !tr.tokens.multilingual => Some(("en".into(), 0)),
            None => None,
        };
        Ok(Live {
            request: Request { timestamps: false, temperature: None, ..request.clone() },
            language,
            window: Vec::new(),
            before: 0,
            context: tr.prompted(request)?,
            settled: Vec::new(),
            last: Vec::new(),
            sampler: Sampler::new(0.0, 0, 1.0, 0),
        })
    }

    /// More audio, 16 kHz mono, as it arrives.
    pub fn push(&mut self, samples: &[f32]) {
        self.window.extend_from_slice(samples);
    }

    /// Seconds of audio pushed since the session began.
    pub fn received(&self) -> f32 {
        (self.before + self.window.len()) as f32 / SAMPLE_RATE as f32
    }

    /// The language being transcribed, once it is known.
    pub fn language(&self) -> Option<&str> {
        self.language.as_ref().map(|(c, _)| c.as_str())
    }

    /// Decode the window as it now stands, and settle what two updates in
    /// a row agree on.
    pub fn update(&mut self, tr: &Transcriber) -> Res<Update> {
        self.step(tr, false)
    }

    /// Settle everything: the end of the audio, or of a turn. What was the
    /// partial is decoded once more and settled as it comes, and the next
    /// audio pushed begins a new window, read after this one.
    pub fn finish(&mut self, tr: &Transcriber) -> Res<Update> {
        let mut u = self.step(tr, true)?;
        self.context.extend_from_slice(&self.settled);
        self.trim_context(tr);
        self.before += self.window.len();
        self.window.clear();
        self.settled.clear();
        self.last.clear();
        u.partial.clear();
        Ok(u)
    }

    fn step(&mut self, tr: &Transcriber, finishing: bool) -> Res<Update> {
        let mut u = Update { heard: self.received(), ..Update::default() };
        if self.window.is_empty() {
            return Ok(u);
        }
        if self.window.len() > mel::N_SAMPLES {
            // Nothing settled to cut at in time: begin again at the newest
            // 30 s, what was settled kept as context.
            let gone = self.window.len() - mel::N_SAMPLES;
            self.window.drain(..gone);
            self.before += gone;
            self.context.append(&mut self.settled);
            self.trim_context(tr);
            self.last.clear();
            u.forced = true;
        }

        let m = &*tr.model;
        let n_mels = m.config().n_mels;
        let window = &self.window;
        let spectrogram = tr.on_pool(|| Ok(mel::window(&mel::log_mel(window, mel::N_SAMPLES - window.len(), n_mels), 0)))?;
        let heard = tr.on_pool(|| m.hear(&spectrogram))?;
        let (_, language) = match &self.language {
            Some(l) => l.clone(),
            None => {
                let (code, _) = tr.on_pool(|| tr.detect(&heard))?;
                let id = tr.tokens.language(&code).unwrap_or(0);
                self.language = Some((code, id));
                self.language.clone().expect("just set")
            }
        };
        let plain = tr.start(language, &self.request);
        let sampler = &mut self.sampler;
        let (context, settled) = (&self.context, &self.settled);
        let w = tr.on_pool(|| tr.window(&heard, context, &plain, settled, Some(LAP), 0.0, sampler))?;

        if self.settled.is_empty() && w.silent() {
            // Nothing said, and nothing to keep: let the window go but for
            // its last second, which may hold a word beginning. Finishing,
            // nothing is settled: what a model writes over silence is not
            // speech.
            let gone = self.window.len().saturating_sub(QUIET_KEPT);
            self.window.drain(..gone);
            self.before += gone;
            self.last.clear();
            u.silent = true;
            return Ok(u);
        }
        let guess = w.tokens;

        // Settled is every word this guess and the last agree on, from the
        // start, compared whole: a word the two begin the same and end
        // differently is not agreed.
        let upto = match finishing {
            true => guess.len(),
            false => {
                let (ours, theirs) = (words(tr, &guess), words(tr, &self.last));
                let agreed = ours.iter().zip(&theirs).take_while(|(a, b)| guess[(*a).clone()] == self.last[(*b).clone()]).last();
                agreed.map_or(0, |(a, _)| a.end).max(self.settled.len())
            }
        };
        u.settled = tr.text(&guess[self.settled.len()..upto])?;
        u.partial = tr.text(&guess[upto..])?;
        self.settled = guess[..upto].to_vec();
        self.last = guess;

        if !finishing && self.window.len() as f32 / SAMPLE_RATE as f32 > CUT {
            // Where to finish the window: one timed decode of what was just
            // heard.
            let timed = tr.start(language, &Request { timestamps: true, ..self.request.clone() });
            let (sampler, context) = (&mut self.sampler, &self.context);
            let w = tr.on_pool(|| tr.window(&heard, context, &timed, &[], Some(LAP), 0.0, sampler))?;
            if let Some((i, at)) = cut(tr, &w.tokens, &self.settled) {
                let gone = ((at * SAMPLE_RATE as f32) as usize).min(self.window.len());
                self.window.drain(..gone);
                self.before += gone;
                self.context.extend(self.settled.drain(..i));
                self.last.drain(..i);
                self.trim_context(tr);
                u.cut = true;
            }
        }
        Ok(u)
    }

    /// Keep no more context than a window can be read after.
    fn trim_context(&mut self, tr: &Transcriber) {
        let keep = tr.model.config().text_ctx / 2;
        if self.context.len() > keep {
            self.context.drain(..self.context.len() - keep);
        }
    }
}

/// The words of `ids`, as the tokens each takes: a word begins at a token
/// whose text begins with a space. Whisper writes one before every word.
fn words(tr: &Transcriber, ids: &[u32]) -> Vec<std::ops::Range<usize>> {
    let begins = |i: u32| i >= tr.tokens.eot || tr.tokenizer.id_to_token(i).is_some_and(|p| p.starts_with('Ġ') || p.starts_with(' '));
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (k, &id) in ids.iter().enumerate() {
        match out.last_mut() {
            Some(w) if !begins(id) => w.end = k + 1,
            _ => out.push(k..k + 1),
        }
    }
    out
}

/// A word as compared between a timed guess and a plain one: its letters
/// and digits, in lower case.
fn plain(word: &str) -> String {
    word.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Each word of `ids` as [`plain`] has it, with how many of `ids` come up
/// to its end. Words of no letters, as a dash, are left out.
fn word_ends(tr: &Transcriber, ids: &[u32]) -> Vec<(String, usize)> {
    words(tr, ids).into_iter().filter_map(|w| Some((plain(&tr.text(&ids[w.clone()]).ok()?), w.end))).filter(|(w, _)| !w.is_empty()).collect()
}

/// Where a window can be finished, from a timed decode of it: the end of
/// its last segment that another follows and whose words are all settled,
/// as how many of the settled tokens it takes and how many seconds into the
/// window it ends. A lone closing timestamp is not a cut: the model closes
/// a segment early when it has not heard the next, and a window started
/// there hears the end of it again.
fn cut(tr: &Transcriber, timed: &[u32], settled: &[u32]) -> Option<(usize, f32)> {
    let t = &tr.tokens;
    let is_ts = |i: u32| i >= t.timestamp_begin;
    let text: Vec<u32> = timed.iter().copied().filter(|&i| !is_ts(i)).collect();
    let heard = word_ends(tr, &text);
    let have = word_ends(tr, settled);
    let pairs = align(&heard.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &have.iter().map(|w| w.0.as_str()).collect::<Vec<_>>());
    // Segment ends, two timestamps side by side, as how many text tokens
    // come before each.
    let mut ends = Vec::new();
    let mut n = 0;
    for i in 0..timed.len() {
        if !is_ts(timed[i]) {
            n += 1;
        } else if i >= 1 && !is_ts(timed[i - 1]) && timed.get(i + 1).is_some_and(|&x| is_ts(x)) {
            ends.push((n, (timed[i] - t.timestamp_begin) as f32 * TIME_STEP));
        }
    }
    for &(n, at) in ends.iter().rev() {
        // The segment's last word, among the settled ones.
        let Some(last) = heard.iter().rposition(|w| w.1 <= n) else { continue };
        if let Some(&(_, j)) = pairs.iter().find(|&&(i, _)| i == last) {
            if at > 0.0 {
                return Some((have[j].1, at));
            }
        }
    }
    None
}

/// Which of `a`'s words are which of `b`'s: the longest run of words the
/// two have in common, in order.
fn align(a: &[&str], b: &[&str]) -> Vec<(usize, usize)> {
    let mut best = vec![vec![0u32; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            best[i][j] = if a[i] == b[j] { best[i + 1][j + 1] + 1 } else { best[i + 1][j].max(best[i][j + 1]) };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push((i, j));
            (i, j) = (i + 1, j + 1);
        } else if best[i + 1][j] >= best[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

#[cfg(any(test, feature = "script"))]
#[doc(hidden)]
pub mod script;

#[cfg(test)]
mod tests;
