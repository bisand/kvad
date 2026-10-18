//! Whether word times put each word in the right speaker's turn: the
//! diarization plan's probe, on kvad's own word times.
//!
//! `scripts/dialogue.py` makes a conversation whose turns are known to the
//! sample. This transcribes it with timestamps and words, matches each word
//! Whisper wrote to the script's words (the same word, in order, by the
//! fewest edits), and gives it the speaker of the turn its time overlaps
//! most, or of the nearest turn when it falls in a gap. Beside that, the
//! same by whole segments, each given to the turn it overlaps most: what a
//! transcript without word times could do. Run:
//!
//!     python3 scripts/dialogue.py --out /tmp/dialogue
//!     cargo run --release -p kvad-gpu --example word_turns -- \
//!         NbAiLab/nb-whisper-small /tmp/dialogue cpu-f32,gpu-q8
//!
//! The model is read from the cache. The probe's numbers, from
//! transformers' word timestamps in f32, were 72 of 74 words on tiny, 74 of
//! 75 on small and 76 of 76 on large.
use kvad::quant::Precision;
use kvad::speech::score::{align, words};
use kvad::speech::{audio, Request, Transcriber};
use kvad::weights::Watcher;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Turn {
    speaker: String,
    start: f32,
    end: f32,
    text: String,
}

fn load(repo: &str, backend: &str) -> Res<Transcriber> {
    let quiet = &mut |_: &str| {};
    match backend.split_once('-') {
        Some(("cpu", p)) => Transcriber::load(repo, Precision::parse(p).ok_or("cpu-f32, cpu-q8 or cpu-q4")?, quiet, &Watcher::none()),
        Some(("gpu", p)) => {
            let (dtype, quant) = match kvad_gpu::model::parse_dtype(p) {
                Some(dt) => (dt, None),
                None => (candle_core::DType::F32, kvad_gpu::model::parse_quant(p).ok_or("gpu-f32, gpu-f16, gpu-bf16, gpu-q8 or gpu-q4")?),
            };
            kvad_gpu::speech::load(repo, dtype, quant, quiet, &Watcher::none())
        }
        _ => Err(format!("`{backend}`: a backend is cpu-PRECISION or gpu-PRECISION").into()),
    }
}

/// Whose turn `(t0, t1)` is: the one it overlaps most, or in a gap the one
/// nearest its middle.
fn speaker_at(turns: &[Turn], t0: f32, t1: f32) -> &str {
    let overlap = |t: &Turn| t1.min(t.end) - t0.max(t.start);
    let best = turns.iter().max_by(|a, b| overlap(a).total_cmp(&overlap(b))).expect("a turn");
    if overlap(best) > 0.0 {
        return &best.speaker;
    }
    let mid = (t0 + t1) / 2.0;
    let gap = |t: &Turn| (mid - t.start).abs().min((mid - t.end).abs());
    &turns.iter().min_by(|a, b| gap(a).total_cmp(&gap(b))).expect("a turn").speaker
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [repo, dir, backends] = &args[..] else {
        return Err("usage: word_turns REPO DIALOGUE_DIR BACKEND,BACKEND,…".into());
    };
    let dir = std::path::Path::new(dir);
    let samples = audio::from_wav(&std::fs::read(dir.join("dialogue.wav"))?)?;
    let truth = kvad::weights::read_json(&dir.join("dialogue.json"))?;
    let turns: Vec<Turn> = truth
        .as_array()
        .ok_or("dialogue.json is not a list of turns")?
        .iter()
        .map(|t| Turn {
            speaker: t["speaker"].as_str().unwrap_or("").into(),
            start: t["start"].as_f64().unwrap_or(0.0) as f32,
            end: t["end"].as_f64().unwrap_or(0.0) as f32,
            text: t["text"].as_str().unwrap_or("").into(),
        })
        .collect();
    // The script's words, each with its speaker.
    let script: Vec<(String, &str)> = turns.iter().flat_map(|t| words(&t.text).into_iter().map(move |w| (w, t.speaker.as_str()))).collect();
    let script_words: Vec<&str> = script.iter().map(|(w, _)| w.as_str()).collect();

    let req = Request { timestamps: true, words: true, temperature: Some(0.0), ..Request::default() };
    for backend in backends.split(',') {
        let t = load(repo, backend)?;
        let done = t.transcribe(&samples, &req, &mut |_| true)?;

        // By word times. A word Whisper wrote that is several script words
        // ("styret," is one) is matched by its normalised parts.
        let mut ours: Vec<(String, f32, f32)> = Vec::new();
        for w in done.segments.iter().flat_map(|s| &s.words) {
            for part in words(&w.word) {
                ours.push((part, w.start, w.end));
            }
        }
        let hyp: Vec<&str> = ours.iter().map(|(w, _, _)| w.as_str()).collect();
        let (_, pairs) = align(&script_words, &hyp);
        let matched: Vec<(usize, usize)> = pairs.into_iter().filter(|&(i, j)| script_words[i] == hyp[j]).collect();
        let mut wrong = Vec::new();
        for &(i, j) in &matched {
            let (w, s, e) = &ours[j];
            if speaker_at(&turns, *s, *e) != script[i].1 {
                wrong.push(format!("{w}@{s:.2}-{e:.2}"));
            }
        }

        // By whole segments.
        let mut by_segment: Vec<(String, &str)> = Vec::new();
        for s in &done.segments {
            let who = speaker_at(&turns, s.start, s.end);
            by_segment.extend(words(&s.text).into_iter().map(|w| (w, who)));
        }
        let seg_words: Vec<&str> = by_segment.iter().map(|(w, _)| w.as_str()).collect();
        let (_, pairs) = align(&script_words, &seg_words);
        let seg_matched: Vec<(usize, usize)> = pairs.into_iter().filter(|&(i, j)| script_words[i] == seg_words[j]).collect();
        let seg_right = seg_matched.iter().filter(|&&(i, j)| by_segment[j].1 == script[i].1).count();

        println!(
            "{repo} {backend}: {} words, {} matched to the script; right speaker by word times {} of {}, by whole segments {seg_right} of {} ({} segments)",
            ours.len(),
            matched.len(),
            matched.len() - wrong.len(),
            matched.len(),
            seg_matched.len(),
            done.segments.len()
        );
        if !wrong.is_empty() {
            println!("  wrong: {}", wrong.join(" "));
        }
        for s in &done.segments {
            let ws: Vec<String> = s.words.iter().map(|w| format!("{}[{:.2}-{:.2}]", w.word.trim(), w.start, w.end)).collect();
            println!("  [{:.2} → {:.2}] {}", s.start, s.end, ws.join(" "));
        }
    }
    Ok(())
}
