//! Words with times: where in a window each word was said.
//!
//! Whisper never writes a word's time. Its timestamp tokens fall between
//! segments, and a segment is a sentence or two. But the decoder reads the
//! audio through cross-attention, and in a few of its heads the weight a
//! text token puts on the 1500 frames of a window peaks where that token is
//! spoken. OpenAI found which heads by hand, model by model, and ships them
//! as `alignment_heads` in each `generation_config.json`; every NB-Whisper
//! kept the heads of the model it was tuned from, and they still align (see
//! `docs/diarization-plan.md`, *The heads are OpenAI's*).
//!
//! The arithmetic is openai-whisper's `timing.py`, step by step:
//!
//! 1. The window's text is decoded once more, from an empty cache, as
//!    `<|startoftranscript|> <|no|> <|transcribe|> <|notimestamps|> text…
//!    <|endoftext|>`, timestamps left out whatever the window was decoded
//!    with ([`super::Model::attention`]).
//! 2. Each alignment head's softmaxed weights, `[tokens × frames]`, cut to
//!    the frames that hold the recording.
//! 3. Each head standardised along the token axis: for every frame, the
//!    mean over tokens taken away and the result divided by their
//!    standard deviation, the population's.
//! 4. Each row median-filtered along time, seven frames wide.
//! 5. The heads averaged, and the rows of the prompt and of end-of-text
//!    dropped, all but `<|notimestamps|>`'s: the scores while the first text
//!    token is being chosen are where it starts.
//! 6. Dynamic time warping over the negated matrix, from the first row at
//!    the first frame to the last row at the last. Where the path first
//!    reaches a row is when that token starts, at 50 frames a second.
//! 7. Tokens grouped into words, and a word timed from its first token's
//!    start to the next word's.
//! 8. openai-whisper's clean-ups ([`tidy`]).
//!
//! transformers' `_extract_token_timestamps` is the same arithmetic over a
//! different set of rows: it drops `<|notimestamps|>`'s too, so its first
//! token ends at frame 0, and standardises over the generated tokens only.
//! Its `token_timestamps[k]` is when token `k` ends, which here is when
//! token `k + 1` starts.
//!
//! Every word's time costs one more pass of the decoder over the window's
//! text, as a prompt: a few percent of the window's encoder.

use super::transcribe::{Segment, Transcriber};
use super::Heard;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Encoder frames a second: two spectrogram columns, of 10 ms each, a frame.
const FRAMES_PER_SECOND: f64 = 50.0;
/// The median filter's width along time, `timing.py`'s.
const MEDIAN_WIDTH: usize = 7;
/// What joins the word after it, and what joins the word before it:
/// openai-whisper's defaults.
const PREPENDED: &str = "\"'“¿([{-";
const APPENDED: &str = "\"'.。,，!！?？:：”)]}、";
/// Python's `string.punctuation`: a piece of text that is one of these, or a
/// run of them in this order, is a word of its own.
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const SENTENCE_END: &str = ".。!！?？";

/// A word, and when it was said, in seconds from the start of the
/// recording. `word` is as Whisper wrote it, with the space before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

/// The heads that align, as `(layer, head)`: the checkpoint's
/// `alignment_heads` where it has them, and otherwise every head of the
/// second half of the decoder's layers, as openai-whisper does. The second
/// says so, since such times are worse.
pub fn heads(generation: Option<&serde_json::Value>, layers: usize, per_layer: usize) -> (Vec<(usize, usize)>, bool) {
    let listed: Option<Vec<(usize, usize)>> = generation.and_then(|g| g.get("alignment_heads")?.as_array()).map(|a| {
        a.iter().filter_map(|p| Some((p.get(0)?.as_u64()? as usize, p.get(1)?.as_u64()? as usize))).filter(|&(l, h)| l < layers && h < per_layer).collect()
    });
    match listed {
        Some(h) if !h.is_empty() => (h, false),
        _ => ((layers / 2..layers).flat_map(|l| (0..per_layer).map(move |h| (l, h))).collect(), true),
    }
}

/// Steps 2 to 6: from the heads' weights, `[head][row][frames]`, to the
/// frame each text token starts at.
///
/// The rows are `sot` prompt tokens, `<|notimestamps|>`, the text and
/// end-of-text; `keep` frames of each hold the recording. The answer has a
/// frame for each text token and one more, where end-of-text starts, which
/// is where the last word ends.
pub fn token_frames(weights: &[f32], heads: usize, rows: usize, frames: usize, keep: usize, sot: usize) -> Vec<usize> {
    assert_eq!(weights.len(), heads * rows * frames, "weights are not [{heads}][{rows}][{frames}]");
    assert!(rows >= sot + 2, "no text to align");
    let keep = keep.clamp(1, frames);
    let mut mean = vec![0.0f32; rows * keep];
    for h in 0..heads {
        let w = |r: usize, f: usize| weights[(h * rows + r) * frames + f];
        // Standardised along the tokens, frame by frame.
        let mut m = vec![0.0f32; rows * keep];
        for f in 0..keep {
            let mu = (0..rows).map(|r| w(r, f)).sum::<f32>() / rows as f32;
            let var = (0..rows).map(|r| (w(r, f) - mu).powi(2)).sum::<f32>() / rows as f32;
            let sd = var.sqrt();
            for r in 0..rows {
                m[r * keep + f] = (w(r, f) - mu) / sd;
            }
        }
        for r in 0..rows {
            let filtered = median_filter(&m[r * keep..(r + 1) * keep], MEDIAN_WIDTH);
            mean[r * keep..(r + 1) * keep].iter_mut().zip(&filtered).for_each(|(a, b)| *a += b);
        }
    }
    mean.iter_mut().for_each(|v| *v /= heads as f32);
    // From `<|notimestamps|>` to the last text token.
    let text = &mean[sot * keep..(rows - 1) * keep];
    let n = rows - 1 - sot;
    let path = dtw(&text.iter().map(|v| -(*v as f64)).collect::<Vec<_>>(), n, keep);
    // Where the path first reaches each row. It reaches every row, unless
    // the matrix was not numbers; then a row it skipped starts where the
    // one before it did.
    let mut starts: Vec<Option<usize>> = vec![None; n];
    for &(row, frame) in &path {
        starts[row].get_or_insert(frame);
    }
    let mut last = 0;
    starts.into_iter().map(|s| {
        last = s.unwrap_or(last);
        last
    }).collect()
}

/// A median along `x`, `width` wide, the edges reflected as PyTorch's
/// `reflect` padding does (the edge itself not repeated). A row no longer
/// than half the width is left as it is.
pub fn median_filter(x: &[f32], width: usize) -> Vec<f32> {
    let pad = width / 2;
    if x.len() <= pad {
        return x.to_vec();
    }
    let n = x.len() as isize;
    let at = |i: isize| -> f32 {
        let j = if i < 0 { -i } else if i >= n { 2 * (n - 1) - i } else { i };
        x[j as usize]
    };
    let mut window = vec![0.0f32; width];
    (0..n)
        .map(|i| {
            for (k, w) in window.iter_mut().enumerate() {
                *w = at(i - pad as isize + k as isize);
            }
            window.sort_by(f32::total_cmp);
            window[pad]
        })
        .collect()
}

/// The cheapest path through `cost`, `[rows][cols]`, from its top left to its
/// bottom right, a step at a time right, down or both, as `(row, col)`
/// pairs in order. Ties go as `timing.py`'s `dtw_cpu` breaks them: the
/// diagonal only when it is strictly cheapest, then down, and otherwise
/// right. The running cost is kept in f32 as there.
pub fn dtw(cost: &[f64], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let w = cols + 1;
    let mut acc = vec![f32::INFINITY; (rows + 1) * w];
    let mut trace = vec![0u8; (rows + 1) * w];
    acc[0] = 0.0;
    for j in 1..=cols {
        for i in 1..=rows {
            let (c0, c1, c2) = (acc[(i - 1) * w + j - 1], acc[(i - 1) * w + j], acc[i * w + j - 1]);
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            acc[i * w + j] = (cost[(i - 1) * cols + j - 1] + c as f64) as f32;
            trace[i * w + j] = t;
        }
    }
    // Every finite path ends by the diagonal into the corner, so it never
    // runs along row or column 0; one through costs that are not numbers
    // might, and stops there.
    let (mut i, mut j) = (rows, cols);
    let mut path = Vec::with_capacity(rows + cols);
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}

/// A word as the alignment makes it: its text, how many tokens it took, and
/// its time in seconds from the start of the window.
#[derive(Debug, Clone, PartialEq)]
pub struct Timing {
    pub word: String,
    pub tokens: usize,
    pub start: f64,
    pub end: f64,
}

/// Step 7: `text`'s tokens as words, each with the decoded text of its
/// tokens, as openai-whisper's `split_tokens_on_spaces` makes them: a new
/// word at each piece that begins with a space or is punctuation, a piece
/// being the tokens that make whole characters together.
pub fn split(t: &Transcriber, text: &[u32]) -> Res<Vec<(String, Vec<u32>)>> {
    let mut pieces: Vec<Vec<u32>> = Vec::new();
    let mut owed = 0u8;
    for &id in text {
        if owed == 0 {
            pieces.push(Vec::new());
        }
        pieces.last_mut().expect("a piece").push(id);
        owed = match t.tokens.utf8.get(id as usize).map(|s| s[owed as usize]) {
            Some(next) if next != u8::MAX => next,
            _ => 0,
        };
    }
    let mut words: Vec<(String, Vec<u32>)> = Vec::new();
    for piece in pieces {
        let s = t.tokenizer.decode(&piece, false).map_err(|e| e.to_string())?;
        let new = s.starts_with(' ') || PUNCTUATION.contains(s.trim()) || words.is_empty();
        match words.last_mut() {
            Some(last) if !new => {
                last.0.push_str(&s);
                last.1.extend(piece);
            }
            _ => words.push((s, piece)),
        }
    }
    Ok(words)
}

/// Opening punctuation joined to the word after it, closing punctuation to
/// the word before, as openai-whisper's `merge_punctuations` does: the
/// joined word keeps its own time, and the punctuation's word is left empty.
pub fn merge_punctuation(words: &mut [Timing]) {
    if words.len() < 2 {
        return;
    }
    let (mut i, mut j) = (words.len() as isize - 2, words.len() - 1);
    while i >= 0 {
        let p = i as usize;
        if words[p].word.starts_with(' ') && PREPENDED.contains(words[p].word.trim()) {
            let (word, tokens) = (std::mem::take(&mut words[p].word), std::mem::take(&mut words[p].tokens));
            words[j].word = word + &words[j].word;
            words[j].tokens += tokens;
        } else {
            j = p;
        }
        i -= 1;
    }
    let (mut i, mut j) = (0, 1);
    while j < words.len() {
        if !words[i].word.ends_with(' ') && APPENDED.contains(words[j].word.as_str()) {
            let (word, tokens) = (std::mem::take(&mut words[j].word), std::mem::take(&mut words[j].tokens));
            words[i].word.push_str(&word);
            words[i].tokens += tokens;
        } else {
            i = j;
        }
        j += 1;
    }
}

/// Step 8, the first half: openai-whisper's hack for a word stretched over a
/// pause, before the punctuation is merged. A word longer than twice the
/// median (the median at most 0.7 s) is cut back where it ends a sentence or
/// follows one's end. Returns the median and the longest a word may be.
fn cut_long_words(words: &mut [Timing]) -> (f64, f64) {
    let mut lengths: Vec<f64> = words.iter().map(|w| w.end - w.start).filter(|&d| d != 0.0).collect();
    lengths.sort_by(f64::total_cmp);
    let median = match lengths.len() {
        0 => 0.0,
        n if n % 2 == 1 => lengths[n / 2],
        n => (lengths[n / 2 - 1] + lengths[n / 2]) / 2.0,
    }
    .min(0.7);
    let longest = 2.0 * median;
    if !lengths.is_empty() {
        for i in 1..words.len() {
            if words[i].end - words[i].start > longest {
                if SENTENCE_END.contains(words[i].word.as_str()) {
                    words[i].end = words[i].start + longest;
                } else if SENTENCE_END.contains(words[i - 1].word.as_str()) {
                    words[i].start = words[i].end - longest;
                }
            }
        }
    }
    (median, longest)
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Step 8, the second half, for one segment's words: the first word after a
/// pause cut back when it is too long, and the first and last words held to
/// the segment's own times when they run more than half a second past them.
/// openai-whisper also moves the segment's times to its words'; here a
/// segment keeps the times it was decoded with, so a transcript is the same
/// whether words were asked for or not. Returns where speech last ended.
pub fn tidy(words: &mut [Word], segment: (f32, f32), median: f64, longest: f64, last_speech: f64) -> f64 {
    let (seg_start, seg_end) = (segment.0 as f64, segment.1 as f64);
    if words.is_empty() {
        return last_speech;
    }
    let mut w: Vec<(f64, f64)> = words.iter().map(|w| (w.start as f64, w.end as f64)).collect();
    if w[0].1 - last_speech > median * 4.0 && (w[0].1 - w[0].0 > longest || (w.len() > 1 && w[1].1 - w[0].0 > longest * 2.0)) {
        if w.len() > 1 && w[1].1 - w[1].0 > longest {
            let boundary = (w[1].1 / 2.0).max(w[1].1 - longest);
            w[0].1 = boundary;
            w[1].0 = boundary;
        }
        w[0].0 = (w[0].1 - longest).max(0.0);
    }
    if seg_start < w[0].1 && seg_start - 0.5 > w[0].0 {
        w[0].0 = (w[0].1 - median).min(seg_start).max(0.0);
    }
    let last = w.len() - 1;
    let mut end = seg_end;
    if seg_end > w[last].0 && seg_end + 0.5 < w[last].1 {
        w[last].1 = (w[last].0 + median).max(seg_end);
    } else {
        end = w[last].1;
    }
    for (word, (s, e)) in words.iter_mut().zip(w) {
        word.start = s as f32;
        word.end = e as f32;
    }
    end
}

impl Transcriber {
    /// The words of `segments`, all from the window `heard` is of, which
    /// starts `offset` seconds in and holds `columns` spectrogram columns of
    /// the recording. `sot` is the window's prompt up to its task, without
    /// `<|notimestamps|>`. `last_speech` is where the words before ended,
    /// and is moved on.
    pub(crate) fn time_words(&self, heard: &Heard, sot: &[u32], segments: &mut [Segment], columns: usize, offset: f32, last_speech: &mut f64) -> Res<()> {
        let t = &self.tokens;
        let per_segment: Vec<Vec<u32>> = segments.iter().map(|s| s.tokens.iter().copied().filter(|&i| i < t.eot).collect()).collect();
        let text: Vec<u32> = per_segment.concat();
        if text.is_empty() {
            return Ok(());
        }
        let mut tokens = sot.to_vec();
        tokens.push(t.no_timestamps);
        tokens.extend_from_slice(&text);
        tokens.push(t.eot);
        let weights = self.on_pool(|| self.model.attention(&tokens, heard, &t.heads))?;
        let frames = weights.len() / (t.heads.len() * tokens.len());
        let starts = token_frames(&weights, t.heads.len(), tokens.len(), frames, columns / 2, sot.len());
        let mut timings = self.timings(&text, &starts)?;
        let (median, longest) = cut_long_words(&mut timings);
        merge_punctuation(&mut timings);

        // The words back to the segments they were written in, by their
        // tokens, as `add_word_timestamps` deals them out.
        let mut next = 0;
        for (segment, own) in segments.iter_mut().zip(&per_segment) {
            let mut words = Vec::new();
            let mut taken = 0;
            while next < timings.len() && taken < own.len() {
                let w = &timings[next];
                if !w.word.is_empty() {
                    let at = |s: f64| round2(offset as f64 + s) as f32;
                    words.push(Word { word: w.word.clone(), start: at(w.start), end: at(w.end) });
                }
                taken += w.tokens;
                next += 1;
            }
            *last_speech = tidy(&mut words, (segment.start, segment.end), median, longest, *last_speech);
            segment.words = words;
        }
        Ok(())
    }

    /// Step 7: `text` as words, each from its first token's start to the
    /// next word's, by `starts` (a frame a token, and one for end-of-text).
    pub fn timings(&self, text: &[u32], starts: &[usize]) -> Res<Vec<Timing>> {
        assert_eq!(starts.len(), text.len() + 1, "a start for every token and the end");
        let mut out = Vec::new();
        let mut at = 0;
        let words = split(self, text)?;
        for (word, tokens) in &words {
            let next = at + tokens.len();
            out.push(Timing { word: word.clone(), tokens: tokens.len(), start: starts[at] as f64 / FRAMES_PER_SECOND, end: starts[next] as f64 / FRAMES_PER_SECOND });
            at = next;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_filter_reflects_its_edges() {
        // Reflected, [1, 5, 2, 8, 3] is … 8 2 5 | 1 5 2 8 3 | 8 2 5 … at
        // width 3: medians of (5,1,5), (1,5,2), (5,2,8), (2,8,3), (8,3,8).
        assert_eq!(median_filter(&[1.0, 5.0, 2.0, 8.0, 3.0], 3), [5.0, 2.0, 5.0, 3.0, 8.0]);
        // Too short to filter: as it was.
        assert_eq!(median_filter(&[3.0, 1.0, 2.0], 7), [3.0, 1.0, 2.0]);
    }

    #[test]
    fn dtw_walks_the_cheap_cells() {
        // Three rows over six columns, cheap along a staircase.
        let cheap = [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)];
        let cost: Vec<f64> = (0..18).map(|k| if cheap.contains(&(k / 6, k % 6)) { -1.0 } else { 1.0 }).collect();
        assert_eq!(dtw(&cost, 3, 6), cheap);
        // From corner to corner, whatever the costs.
        let path = dtw(&[0.0; 12], 4, 3);
        assert_eq!((path[0], *path.last().unwrap()), ((0, 0), (3, 2)));
    }

    #[test]
    fn a_token_starts_where_its_head_looks_first() {
        // One head, rows: sot, notimestamps, two text tokens, eot; forty
        // frames, each row looking at its own stretch, wider than the
        // median filter.
        let (rows, frames) = (5, 40);
        let looks = [(0, 4), (0, 10), (10, 20), (20, 30), (30, 40)];
        let mut w = vec![0.01f32; rows * frames];
        for (r, &(a, b)) in looks.iter().enumerate() {
            for f in a..b {
                w[r * frames + f] = 1.0;
            }
        }
        // `<|notimestamps|>` from 0, the two words from 10 and 20, and the
        // end where end-of-text would begin.
        assert_eq!(token_frames(&w, 1, rows, frames, frames, 1), [0, 10, 20]);
        // Only the frames that hold the recording are read: with 25 kept,
        // the second word is squeezed into what is left.
        let short = token_frames(&w, 1, rows, frames, 25, 1);
        assert_eq!(&short[..2], [0, 10]);
        assert!(short[2] < 25, "{short:?}");
    }

    fn timing(word: &str, start: f64, end: f64) -> Timing {
        Timing { word: word.into(), tokens: 1, start, end }
    }

    #[test]
    fn punctuation_joins_its_word() {
        let mut w = vec![timing(" «", 0.0, 0.1), timing("Hei", 0.1, 0.4), timing(",", 0.4, 0.5), timing(" sa", 0.5, 0.8), timing(" han", 0.8, 1.0), timing(".", 1.0, 1.1)];
        // `«` is not one of openai-whisper's opening marks, and stays.
        merge_punctuation(&mut w);
        let words: Vec<&str> = w.iter().filter(|w| !w.word.is_empty()).map(|w| w.word.as_str()).collect();
        assert_eq!(words, [" «", "Hei,", " sa", " han."]);
        let mut w = vec![timing(" (", 0.0, 0.1), timing(" ja", 0.1, 0.4), timing(")", 0.4, 0.5)];
        merge_punctuation(&mut w);
        assert_eq!((w[1].word.as_str(), w[1].tokens, w[1].start), (" ( ja)", 3, 0.1));
    }

    #[test]
    fn words_are_held_to_their_segment() {
        let word = |s, e| Word { word: " x".into(), start: s, end: e };
        // The first word starts 1.5 s before its segment does, and the last
        // ends 1 s after: both are brought in.
        let mut w = vec![word(0.5, 2.2), word(2.2, 2.5), word(2.5, 4.0)];
        let end = tidy(&mut w, (2.0, 3.0), 0.3, 0.6, 0.0);
        assert_eq!((w[0].start, w[2].end), (1.6, 3.0));
        assert_eq!(end, 3.0);
        // Words inside their segment are left alone, and the end of speech
        // is the last word's.
        let mut w = vec![word(2.0, 2.4), word(2.4, 2.9)];
        assert_eq!(tidy(&mut w, (2.0, 3.0), 0.3, 0.6, 1.9), 2.9f32 as f64);
        assert_eq!((w[0].start, w[1].end), (2.0, 2.9));
    }

    #[test]
    fn the_heads_are_the_checkpoints_or_the_second_half() {
        let g = serde_json::json!({ "alignment_heads": [[2, 2], [3, 0], [9, 9]] });
        // A head outside the model is dropped.
        assert_eq!(heads(Some(&g), 4, 6), (vec![(2, 2), (3, 0)], false));
        let (all, guessed) = heads(None, 4, 2);
        assert_eq!((all, guessed), (vec![(2, 0), (2, 1), (3, 0), (3, 1)], true));
    }
}
