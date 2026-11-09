//! Paragraphs: a blank line in a transcript wherever the speaker paused.
//!
//! Whisper's segments are sentences, but their times are a guess: across a
//! pause it often stretches the one before and starts the next early, so two
//! can meet with the silence inside one of them. So the pauses are heard in
//! the audio itself, and each begins its paragraph at the gap between two
//! segments nearest it. Kvad for iPhone does the same for a take
//! (`apps/phone/core/src/speak.rs`, `Quiet`, and `api.rs`, `paragraphs`):
//! the two are twins, and a change to one is a change to both.

use super::mel::SAMPLE_RATE;
use super::transcribe::Segment;

/// The shortest quiet that begins a new paragraph, in seconds: longer than a
/// breath or a comma, shorter than a turn.
pub const PAUSE: f32 = 1.5;

/// How far a pause may lie from the gap between two segments and still be
/// that gap's, in seconds.
const NEAR: f32 = 1.0;

/// The blocks loudness is judged in, in samples: 50 ms.
const BLOCK: usize = SAMPLE_RATE / 20;

/// Below this amplitude a block is quiet in any room: about -66 dBFS.
const QUIETEST: f32 = 0.0005;

/// Where the noise floor starts, before the room has been heard: about
/// -54 dBFS.
const FLOOR: f32 = 0.002;

/// Where 16 kHz `audio` was quiet for [`PAUSE`] or more, start and end in
/// seconds. A block is quiet under 2.5 times the noise floor's amplitude
/// (8 dB). The floor falls at once to a quieter block, and rises only toward
/// a block that was quiet, over a second or so, so a voice, however steady,
/// never raises it.
pub fn pauses(audio: &[f32]) -> Vec<(f32, f32)> {
    let mut floor = FLOOR;
    let mut since: Option<usize> = None;
    let mut out = Vec::new();
    let seconds = |block: usize| (block * BLOCK) as f32 / SAMPLE_RATE as f32;
    let mut block = 0;
    for b in audio.chunks_exact(BLOCK) {
        let rms = (b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / BLOCK as f64).sqrt() as f32;
        let quiet = rms < (floor * 2.5).max(QUIETEST);
        if rms < floor {
            floor = rms;
        } else if quiet {
            floor += (rms - floor) * 0.05;
        }
        match (quiet, since) {
            (true, None) => since = Some(block),
            (false, Some(from)) => {
                if seconds(block) - seconds(from) >= PAUSE {
                    out.push((seconds(from), seconds(block)));
                }
                since = None;
            }
            _ => {}
        }
        block += 1;
    }
    // A recording that ends in silence has no paragraph after it to begin.
    out
}

/// The segments' text as one, with a blank line between two of them where
/// the speaker paused: at the gap nearest each of `pauses` if one is within
/// [`NEAR`], and wherever a segment begins [`PAUSE`] or more after the last
/// ended. Between the others the text is joined as Whisper wrote it.
pub fn paragraphs(segments: &[Segment], pauses: &[(f32, f32)]) -> String {
    let segments: Vec<&Segment> = segments.iter().filter(|s| !s.text.trim().is_empty()).collect();
    // Before segment i, the gap from where the last ended to where it begins,
    // either way round, since the two can overlap.
    let gaps: Vec<(f32, f32)> = (1..segments.len()).map(|i| {
        let (end, start) = (segments[i - 1].end, segments[i].start);
        (end.min(start), end.max(start))
    }).collect();
    let mut breaks: Vec<bool> = (1..segments.len()).map(|i| segments[i].start - segments[i - 1].end >= PAUSE).collect();
    for &(from, to) in pauses {
        let far = |&(lo, hi): &(f32, f32)| (lo - to).max(from - hi).max(0.0);
        let nearest = (0..gaps.len()).min_by(|&a, &b| far(&gaps[a]).total_cmp(&far(&gaps[b])));
        if let Some(i) = nearest.filter(|&i| far(&gaps[i]) <= NEAR) {
            breaks[i] = true;
        }
    }
    let mut text = String::new();
    for (i, s) in segments.iter().enumerate() {
        if i > 0 && breaks[i - 1] {
            text.truncate(text.trim_end().len());
            text.push_str("\n\n");
            text.push_str(s.text.trim_start());
        } else {
            text.push_str(&s.text);
        }
    }
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, text: &str) -> Segment {
        Segment { start, end, text: text.into(), tokens: Vec::new(), temperature: 0.0, avg_logprob: 0.0, no_speech_prob: 0.0, words: Vec::new(), speaker: None }
    }

    #[test]
    fn a_gap_between_segments_begins_a_paragraph() {
        let text = paragraphs(&[seg(0.0, 2.0, " Hei."), seg(2.4, 4.0, " Hvordan går det?"), seg(6.0, 7.0, " Bra."), seg(7.1, 7.5, " ")], &[]);
        assert_eq!(text, "Hei. Hvordan går det?\n\nBra.");
        assert_eq!(paragraphs(&[], &[(1.0, 3.0)]), "");
    }

    /// Whisper stretched both segments into the pause, so they nearly meet;
    /// the pause the audio had still begins the paragraph, and one that fell
    /// inside a segment, far from any gap, begins none.
    #[test]
    fn a_pause_in_the_audio_begins_a_paragraph_at_the_nearest_gap() {
        let segments = [seg(0.0, 4.2, " Hei."), seg(4.6, 9.0, " Hvordan går det?"), seg(9.0, 14.0, " Bra."), seg(14.0, 20.0, " Og du?")];
        assert_eq!(paragraphs(&segments, &[]), "Hei. Hvordan går det? Bra. Og du?");
        assert_eq!(paragraphs(&segments, &[(3.0, 5.5)]), "Hei.\n\nHvordan går det? Bra. Og du?");
        assert_eq!(paragraphs(&segments, &[(14.5, 16.0)]), "Hei. Hvordan går det? Bra.\n\nOg du?");
        assert_eq!(paragraphs(&segments, &[(16.5, 18.5)]), "Hei. Hvordan går det? Bra. Og du?", "inside a segment, 2.5 s from a gap");
    }

    /// Text with no spaces between its segments, as Chinese is written, keeps
    /// none where there is no paragraph.
    #[test]
    fn segments_are_joined_as_written() {
        assert_eq!(paragraphs(&[seg(0.0, 1.0, "你好。"), seg(1.0, 2.0, "再见。")], &[]), "你好。再见。");
    }

    /// Two seconds of a tone, two of silence, one of the tone: one pause,
    /// where the silence was; a breath of half a second is none.
    #[test]
    fn silence_in_the_audio_is_a_pause() {
        let tone = |s: f32| (0..(s * SAMPLE_RATE as f32) as usize).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect::<Vec<f32>>();
        let hush = |s: f32| vec![0.0001f32; (s * SAMPLE_RATE as f32) as usize];
        let audio = [tone(2.0), hush(2.0), tone(1.0), hush(0.5), tone(1.0)].concat();
        let p = pauses(&audio);
        assert_eq!(p.len(), 1, "{p:?}");
        assert!((p[0].0 - 2.0).abs() < 0.06 && (p[0].1 - 4.0).abs() < 0.06, "{p:?}");
    }
}
