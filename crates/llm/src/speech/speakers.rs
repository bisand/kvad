//! Words to speakers: a transcript with its words timed, and a diarization's
//! turns, made into a transcript whose segments each have one speaker.
//!
//! The diarization gives each speaker intervals, overlapping where two talk
//! at once; Whisper gives one stream of words. Each word goes to the speaker
//! whose turns cover the most of it. A word that falls in no turn (a gap, or
//! a word timed a frame or two out) goes to the speaker of the nearest turn
//! within half a second, and otherwise to the speaker of the word before it.
//! That is whisperX's rule (`assign_word_speakers`).
//!
//! Then each segment is cut where its speaker changes, at a word, and
//! Whisper's own boundaries are kept everywhere else: a segment one speaker
//! holds throughout comes out as it went in. A piece of a cut segment keeps
//! no tokens, which were its window's, not its turn's.
//!
//! Speakers are named `A`, `B`, … in the order they are first heard in the
//! transcript, as OpenAI's diarizing model names them; a speaker the
//! diarization found but no word went to is not named at all.

use super::score::Turn;
use super::transcribe::{Segment, Transcript};

/// How far from a turn a word in a gap may be and still be given to it.
const NEAREST: f64 = 0.5;

/// `transcript`, its segments cut and labelled by `turns`. Its words must
/// be timed ([`super::Request::words`]); a segment without words keeps its
/// text and takes the speaker of the turn that covers most of it.
pub fn assign(transcript: &Transcript, turns: &[Turn]) -> Transcript {
    let mut names: Vec<String> = Vec::new();
    let mut name = |speaker: &str| -> String {
        let i = names.iter().position(|n| n == speaker).unwrap_or_else(|| {
            names.push(speaker.to_string());
            names.len() - 1
        });
        label(i)
    };
    let mut segments = Vec::new();
    let mut last: Option<String> = None;
    for s in &transcript.segments {
        if s.words.is_empty() {
            let who = speaker_of(turns, s.start as f64, s.end as f64).or(last.clone());
            last = who.clone().or(last);
            segments.push(Segment { speaker: Some(name(who.as_deref().unwrap_or(""))), ..s.clone() });
            continue;
        }
        let mut pieces: Vec<(String, Vec<super::align::Word>)> = Vec::new();
        for w in &s.words {
            let who = speaker_of(turns, w.start as f64, w.end as f64).or_else(|| last.clone());
            last = who.clone();
            let who = who.unwrap_or_default();
            match pieces.last_mut() {
                Some((p, words)) if *p == who => words.push(w.clone()),
                _ => pieces.push((who, vec![w.clone()])),
            }
        }
        // A first word heard before anyone's turn takes the speaker of the
        // words after it.
        if pieces.len() > 1 && pieces[0].0.is_empty() {
            let (_, words) = pieces.remove(0);
            pieces[0].1.splice(0..0, words);
        }
        if pieces.len() == 1 {
            let (who, _) = pieces.pop().unwrap();
            segments.push(Segment { speaker: Some(name(&who)), ..s.clone() });
            continue;
        }
        let n = pieces.len();
        for (i, (who, words)) in pieces.into_iter().enumerate() {
            segments.push(Segment {
                start: if i == 0 { s.start } else { words[0].start },
                end: if i == n - 1 { s.end } else { words[words.len() - 1].end },
                text: words.iter().map(|w| w.word.as_str()).collect(),
                tokens: Vec::new(),
                speaker: Some(name(&who)),
                words,
                ..s.clone()
            });
        }
    }
    Transcript { segments, ..transcript.clone() }
}

/// `transcript` with its speakers, found by `model` in `audio`, the
/// recording it was transcribed from; its words must be timed. `words`
/// false drops them again, for a caller that asked for speakers and not for
/// words. `progress` hears seconds done of seconds in all.
pub fn diarize(
    transcript: &Transcript,
    audio: &[f32],
    model: &super::nemotron::Nemotron,
    words: bool,
    progress: &mut (dyn FnMut(f32, f32) + Send),
) -> Transcript {
    let probs = model.diarize(audio, progress);
    let turns = super::nemotron::turns(&probs, model.config.speakers, 0.5);
    let mut out = assign(transcript, &turns);
    if !words {
        out.segments.iter_mut().for_each(|s| s.words.clear());
    }
    out
}

/// `A` to `Z`, then `AA`, `AB`, …
fn label(i: usize) -> String {
    match i {
        0..26 => ((b'A' + i as u8) as char).to_string(),
        _ => format!("{}{}", label(i / 26 - 1), label(i % 26)),
    }
}

/// The speaker whose turns cover most of `start..end`, or, with none
/// covering any of it, the speaker of the nearest turn within [`NEAREST`].
fn speaker_of(turns: &[Turn], start: f64, end: f64) -> Option<String> {
    let mut covered: Vec<(&str, f64)> = Vec::new();
    for t in turns.iter().filter(|t| t.start < end && t.end > start) {
        let s = t.end.min(end) - t.start.max(start);
        match covered.iter_mut().find(|(n, _)| *n == t.speaker) {
            Some((_, c)) => *c += s,
            None => covered.push((&t.speaker, s)),
        }
    }
    if let Some((n, _)) = covered.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
        return Some(n.to_string());
    }
    turns
        .iter()
        .map(|t| (t, (t.start - end).max(start - t.end)))
        .filter(|&(_, gap)| gap <= NEAREST)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(t, _)| t.speaker.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::align::Word;

    fn turn(speaker: &str, start: f64, end: f64) -> Turn {
        Turn { speaker: speaker.into(), start, end }
    }

    fn word(w: &str, start: f32, end: f32) -> Word {
        Word { word: w.into(), start, end }
    }

    fn segment(start: f32, end: f32, words: Vec<Word>) -> Segment {
        Segment {
            start,
            end,
            text: words.iter().map(|w| w.word.as_str()).collect(),
            tokens: vec![1, 2, 3],
            temperature: 0.0,
            avg_logprob: -0.2,
            no_speech_prob: 0.01,
            words,
            speaker: None,
        }
    }

    fn transcript(segments: Vec<Segment>) -> Transcript {
        Transcript { text: String::new(), language: "no".into(), duration: 10.0, segments }
    }

    fn said(t: &Transcript) -> Vec<(String, f32, f32, String)> {
        t.segments.iter().map(|s| (s.speaker.clone().unwrap(), s.start, s.end, s.text.clone())).collect()
    }

    #[test]
    fn a_segment_one_speaker_holds_comes_out_as_it_went_in() {
        let t = transcript(vec![segment(0.0, 2.0, vec![word(" Hei", 0.1, 0.5), word(" der.", 0.6, 1.2)])]);
        let out = assign(&t, &[turn("speaker_3", 0.0, 2.0)]);
        assert_eq!(out.segments[0], Segment { speaker: Some("A".into()), ..t.segments[0].clone() });
    }

    #[test]
    fn a_turn_inside_a_segment_cuts_it() {
        let t = transcript(vec![segment(
            0.0,
            4.0,
            vec![word(" Hva", 0.1, 0.4), word(" synes", 0.5, 0.9), word(" du?", 1.0, 1.4), word(" Bra.", 2.0, 2.6), word(" Takk.", 3.0, 3.6)],
        )]);
        let out = assign(&t, &[turn("speaker_0", 0.0, 1.5), turn("speaker_1", 1.8, 2.8), turn("speaker_0", 2.9, 4.0)]);
        assert_eq!(
            said(&out),
            vec![
                ("A".into(), 0.0, 1.4, " Hva synes du?".into()),
                ("B".into(), 2.0, 2.6, " Bra.".into()),
                ("A".into(), 3.0, 4.0, " Takk.".into()),
            ]
        );
        assert!(out.segments.iter().all(|s| s.tokens.is_empty()));
        assert_eq!(out.segments[0].avg_logprob, -0.2);
    }

    #[test]
    fn a_word_in_a_gap_goes_to_the_nearest_turn_or_the_one_before() {
        let t = transcript(vec![segment(0.0, 6.0, vec![word(" en", 0.2, 0.6), word(" to", 1.2, 1.4), word(" tre", 4.0, 4.4)])]);
        // `to` is 0.3 s after A's turn and 0.4 s before B's: A. `tre` is
        // far from both: A, the word before it.
        let out = assign(&t, &[turn("a", 0.0, 0.9), turn("b", 1.8, 2.5)]);
        assert_eq!(said(&out), vec![("A".into(), 0.0, 6.0, " en to tre".into())]);
        let t = transcript(vec![segment(0.0, 3.0, vec![word(" en", 0.2, 0.6), word(" to", 1.5, 1.7)])]);
        let out = assign(&t, &[turn("a", 0.0, 0.9), turn("b", 1.9, 2.5)]);
        assert_eq!(said(&out).iter().map(|s| s.0.as_str()).collect::<Vec<_>>(), ["A", "B"]);
    }

    #[test]
    fn in_an_overlap_the_word_goes_to_who_covers_more_of_it() {
        let t = transcript(vec![segment(0.0, 2.0, vec![word(" nei", 0.2, 0.5), word(" ja", 1.0, 1.4)])]);
        let who = |turns: &[Turn]| said(&assign(&t, turns)).into_iter().map(|s| s.0).collect::<Vec<_>>();
        assert_eq!(who(&[turn("a", 0.0, 1.1), turn("b", 0.9, 2.0)]), ["A", "B"]);
        assert_eq!(who(&[turn("a", 0.0, 1.3), turn("b", 1.2, 2.0)]), ["A"]);
    }

    #[test]
    fn a_one_word_turn_is_a_segment_of_its_own() {
        let t = transcript(vec![segment(0.0, 3.0, vec![word(" Så", 0.1, 0.4), word(" ja.", 1.0, 1.2), word(" Videre.", 1.5, 2.5)])]);
        let out = assign(&t, &[turn("x", 0.0, 0.8), turn("y", 0.9, 1.3), turn("x", 1.4, 3.0)]);
        assert_eq!(said(&out).iter().map(|s| (s.0.as_str(), s.3.as_str())).collect::<Vec<_>>(), [("A", " Så"), ("B", " ja."), ("A", " Videre.")]);
    }

    #[test]
    fn speakers_are_named_in_the_order_they_are_heard() {
        let t = transcript(vec![
            segment(0.0, 1.0, vec![word(" en", 0.1, 0.5)]),
            segment(1.0, 2.0, vec![word(" to", 1.1, 1.5)]),
            segment(2.0, 3.0, vec![word(" tre", 2.1, 2.5)]),
        ]);
        // speaker_5 has a turn but no word, and gets no name.
        let out = assign(&t, &[turn("speaker_2", 0.0, 1.0), turn("speaker_0", 1.0, 2.0), turn("speaker_5", 5.0, 6.0), turn("speaker_2", 2.0, 3.0)]);
        assert_eq!(said(&out).iter().map(|s| s.0.as_str()).collect::<Vec<_>>(), ["A", "B", "A"]);
        assert_eq!((label(25), label(26), label(27)), ("Z".into(), "AA".into(), "AB".into()));
    }

    #[test]
    fn a_segment_without_words_takes_who_covers_it() {
        let mut s = segment(0.0, 2.0, vec![]);
        s.text = " Hei.".into();
        let out = assign(&transcript(vec![s.clone()]), &[turn("q", 0.5, 2.0)]);
        assert_eq!(out.segments, vec![Segment { speaker: Some("A".into()), ..s }]);
    }

    /// The plan's word-level table: each Whisper's transcript of each
    /// recording of `KVAD_DIAR_DATA`'s sets (`<set>/asr/<whisper>/<name>.json`,
    /// `kvad transcribe --words`'s verbose_json), given speakers by each
    /// diarization's turns (`<set>/hyp/<system>/<name>.rttm`) with
    /// [`assign`], against the reference's words (`<set>/<name>.json`, from
    /// `scripts/diar-sets.py`). WER is Whisper's alone; WDER and cpWER are
    /// the speakers'. WDER is each recording's, weighed by its words.
    ///
    /// ```text
    /// KVAD_DIAR_DATA=~/data/diar/sets cargo test --release -p kvad speech::speakers::tests::word_sets -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn word_sets() {
        use crate::speech::score::{self, Spoken};
        use std::path::{Path, PathBuf};
        let root = std::env::var("KVAD_DIAR_DATA").expect("KVAD_DIAR_DATA is not set");
        let dirs = |p: &Path| -> Vec<PathBuf> {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p).map(|d| d.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
            v.sort();
            v
        };
        let json = |p: &Path| -> serde_json::Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
        let spoken = |who: &str, text: &str| score::words(text).into_iter().map(|word| Spoken { speaker: who.to_string(), word }).collect::<Vec<_>>();
        println!("| set | whisper | diarization | files | words | WER | WDER | cpWER |");
        println!("|---|---|---|---|---|---|---|---|");
        for set in dirs(Path::new(&root)) {
            for asr in dirs(&set.join("asr")) {
                for system in dirs(&set.join("hyp")) {
                    let (mut words, mut wer, mut wder, mut cp, mut files) = (0usize, 0usize, 0.0, 0usize, 0);
                    let mut names: Vec<PathBuf> = std::fs::read_dir(&asr).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
                    names.sort();
                    for path in names {
                        let name = path.file_stem().unwrap().to_str().unwrap();
                        let Ok(rttm) = std::fs::read_to_string(system.join(format!("{name}.rttm"))) else { continue };
                        let turns = score::rttm(&rttm).unwrap();
                        let v = json(&path);
                        let heard: Vec<Word> = v["words"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|w| Word { word: format!(" {}", w["word"].as_str().unwrap()), start: w["start"].as_f64().unwrap() as f32, end: w["end"].as_f64().unwrap() as f32 })
                            .collect();
                        let t = transcript(vec![segment(0.0, v["duration"].as_f64().unwrap() as f32, heard)]);
                        let out = assign(&t, &turns);
                        let hypothesis: Vec<Spoken> = out.segments.iter().flat_map(|s| s.words.iter().flat_map(|w| spoken(s.speaker.as_deref().unwrap(), &w.word))).collect();
                        let reference: Vec<Spoken> = json(&set.join(format!("{name}.json")))
                            .as_array()
                            .unwrap()
                            .iter()
                            .flat_map(|e| spoken(e["speaker"].as_str().unwrap(), e["text"].as_str().unwrap()))
                            .collect();
                        let (r, h): (Vec<&str>, Vec<&str>) = (reference.iter().map(|w| w.word.as_str()).collect(), hypothesis.iter().map(|w| w.word.as_str()).collect());
                        wer += score::align(&r, &h).0;
                        wder += score::wder(&reference, &hypothesis).unwrap_or(0.0) * reference.len() as f64;
                        cp += score::cpwer(&reference, &hypothesis).0;
                        words += reference.len();
                        files += 1;
                    }
                    if files == 0 {
                        continue;
                    }
                    let pc = |x: f64| 100.0 * x / words as f64;
                    let name = |p: &Path| p.file_name().unwrap().to_string_lossy().to_string();
                    println!(
                        "| {} | {} | {} | {files} | {words} | {:.1} | {:.1} | {:.1} |",
                        name(&set),
                        name(&asr),
                        name(&system),
                        pc(wer as f64),
                        pc(wder),
                        pc(cp as f64)
                    );
                }
            }
        }
    }
}
