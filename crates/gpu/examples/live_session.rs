//! What the shipped live session does with a recording played in at real
//! time: `kvad::speech::Live`, the loop the server's `/v1/realtime` runs,
//! where `live_speed` measured a copy of it before it was built.
//!
//! The recording is pushed in as the wall clock reaches it, and an update
//! is run about once a `--cadence` (1 s by default), as the server runs
//! them: one still running when the next is due skips it, and audio is
//! never dropped. At the end, the session is finished, as a client ending
//! its turn has it. Run:
//!
//!     cargo run --release -p kvad-gpu --example live_session -- \
//!         NbAiLab/nb-whisper-small king.wav gpu-q8,gpu-f16 3
//!
//! Each run reports its updates' wall time, and the real-time factor:
//! seconds of work for each second of audio, the share of the time the
//! backend was busy. Then, for every word, how long after it was spoken it
//! first showed as a partial (unchanged from then on, with every word
//! before it) and when it settled. "Spoken" is where the same backend's
//! timed transcript of the whole file puts the end of the word, its
//! segment's time shared out by letters, as `live_speed` has it, and is
//! good to a second or so: read the median and the 90th percentile.
//! `--trace` prints every update.
use kvad::quant::Precision;
use kvad::speech::mel::SAMPLE_RATE;
use kvad::speech::{audio, Live, Request, Transcriber, Transcript};
use kvad::weights::Watcher;
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn load(repo: &str, backend: &str) -> Res<Transcriber> {
    let quiet = &mut |_: &str| {};
    match backend.split_once('-') {
        Some(("cpu", p)) => {
            let precision = Precision::parse(p).ok_or("cpu-f32, cpu-q8 or cpu-q4")?;
            Transcriber::load_with(repo, quiet, &Watcher::none(), &mut |config, files, _| {
                Ok(Box::new(kvad::speech::whisper::Cpu::load(config, &files.weights, precision)?))
            })
        }
        Some(("gpu", p)) => {
            let (dtype, quant) = match kvad_gpu::model::parse_dtype(p) {
                Some(dt) => (dt, None),
                None => (candle_core::DType::F32, kvad_gpu::model::parse_quant(p).ok_or("gpu-f32, gpu-f16, gpu-bf16, gpu-q8 or gpu-q4")?),
            };
            let device = kvad_gpu::model::pick_device(None)?;
            Transcriber::load_with(repo, quiet, &Watcher::none(), &mut |config, files, progress| {
                Ok(Box::new(kvad_gpu::speech::GpuWhisper::load(repo, config, &files.weights, device.clone(), dtype, quant, progress)?))
            })
        }
        _ => Err(format!("`{backend}`: a backend is cpu-PRECISION or gpu-PRECISION").into()),
    }
}

/// A word as compared: its letters and digits, in lower case.
fn plain(word: &str) -> String {
    word.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

fn plain_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(plain).filter(|w| !w.is_empty()).collect()
}

/// Each word of the file's own transcript, and when it was spoken.
fn spoken_words(done: &Transcript) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for seg in &done.segments {
        let words = plain_words(&seg.text);
        let total: usize = words.iter().map(|w| w.chars().count()).sum();
        let mut upto = 0;
        for w in words {
            upto += w.chars().count();
            out.push((w, seg.start + (seg.end - seg.start) * upto as f32 / total.max(1) as f32));
        }
    }
    out
}

/// Which of `a`'s words are which of `b`'s: the longest common run.
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

fn spread(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!("{:.2}/{:.2}/{:.2}/{:.2}", at(0.0), at(0.5), at(0.9), at(1.0))
}

/// What one run came to.
struct Run {
    walls: Vec<f64>,
    busy: f64,
    skipped: usize,
    cuts: usize,
    forced: usize,
    /// Each settled word, with since when it showed and when it settled.
    words: Vec<(String, f64, f64)>,
    text: String,
    end: f64,
}

fn live(tr: &Transcriber, samples: &[f32], cadence: f64, trace: bool) -> Res<Run> {
    let duration = samples.len() as f64 / SAMPLE_RATE as f64;
    let mut session = Live::new(tr, &Request::default())?;
    let start = Instant::now();
    let now = || start.elapsed().as_secs_f64();
    let mut run = Run { walls: Vec::new(), busy: 0.0, skipped: 0, cuts: 0, forced: 0, words: Vec::new(), text: String::new(), end: 0.0 };
    // The words showing, settled and partial, with since when each has
    // shown unchanged, every word before it too.
    let mut showing: Vec<(String, f64)> = Vec::new();
    let mut pushed = 0;
    let mut tick = 1usize;
    loop {
        std::thread::sleep(Duration::from_secs_f64((tick as f64 * cadence - now()).max(0.0)));
        let have = ((now() * SAMPLE_RATE as f64) as usize).min(samples.len());
        session.push(&samples[pushed..have]);
        pushed = have;
        let ended = have == samples.len();
        let t0 = Instant::now();
        let u = if ended { session.finish(tr)? } else { session.update(tr)? };
        let wall = t0.elapsed().as_secs_f64();
        let t = now();
        run.walls.push(wall);
        run.busy += wall;
        run.cuts += u.cut as usize;
        run.forced += u.forced as usize;

        run.text.push_str(&u.settled);
        let settled_n = plain_words(&run.text).len();
        let shown = plain_words(&format!("{}{}", run.text, u.partial));
        let same = shown.iter().zip(&showing).take_while(|(a, b)| *a == &b.0).count();
        showing.truncate(same);
        showing.extend(shown[same..].iter().map(|w| (w.clone(), t)));
        for (w, since) in &showing[run.words.len()..settled_n.min(showing.len())] {
            run.words.push((w.clone(), *since, t));
        }
        if trace {
            println!("  {t:6.2} s  heard {:.2} s  {:.0} ms{}{}  settled +[{}]  partial [{}]", u.heard, 1000.0 * wall, if u.cut { "  cut" } else { "" }, if u.silent { "  silent" } else { "" }, u.settled, u.partial);
        }
        if ended {
            run.end = t - duration;
            break;
        }
        let next = (t / cadence).floor() as usize + 1;
        run.skipped += next - tick - 1;
        tick = next;
    }
    Ok(run)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trace = args.iter().any(|a| a == "--trace");
    let cadence: f64 = args.iter().find_map(|a| a.strip_prefix("--cadence=")).map_or(Ok(1.0), str::parse)?;
    let args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [repo, wav, backends, rounds] = args[..] else {
        return Err("usage: live_session REPO WAV BACKEND,BACKEND,… ROUNDS [--cadence=SECONDS] [--trace]".into());
    };
    let samples = audio::from_wav(&std::fs::read(wav)?)?;
    let seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let rounds: usize = rounds.parse()?;
    let backends: Vec<&str> = backends.split(',').collect();
    let (mut loaded, mut spoken) = (Vec::new(), Vec::new());
    for b in &backends {
        let t = Instant::now();
        let model = load(repo, b)?;
        // One update over five seconds, so that no round is the first time
        // each kernel runs.
        let mut warm = Live::new(&model, &Request::default())?;
        warm.push(&samples[..5 * SAMPLE_RATE]);
        warm.update(&model)?;
        println!("{b}: {} loaded in {:.1} s", model.backend(), t.elapsed().as_secs_f32());
        let done = model.transcribe(&samples, &Request { timestamps: true, ..Request::default() }, &mut |_| true)?;
        spoken.push(spoken_words(&done));
        loaded.push(model);
    }
    println!("{seconds:.1} s of audio, an update due every {cadence} s");
    println!("columns: min/median/p90/max; a word's lag is from where the file's timed transcript puts its end");
    let mut texts: Vec<Option<String>> = vec![None; backends.len()];
    for round in 1..=rounds {
        for (n, b) in backends.iter().enumerate() {
            let run = live(&loaded[n], &samples, cadence, trace)?;
            let reference = &spoken[n];
            let pairs = align(&run.words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &reference.iter().map(|w| w.0.as_str()).collect::<Vec<_>>());
            let shown: Vec<f64> = pairs.iter().map(|&(i, j)| run.words[i].1 - reference[j].1 as f64).collect();
            let settled: Vec<f64> = pairs.iter().map(|&(i, j)| run.words[i].2 - reference[j].1 as f64).collect();
            println!(
                "round {round} {b:8} {} updates, wall {} s, real-time factor {:.2}, {} skipped, {} cuts{}, ends {:.2} s after the audio",
                run.walls.len(),
                spread(run.walls.clone()),
                run.busy / (seconds + run.end),
                run.skipped,
                run.cuts,
                if run.forced > 0 { format!(" ({} forced)", run.forced) } else { String::new() },
                run.end,
            );
            println!("         {b:8} {} words, {} of {} as the file's, lag shown {} s, settled {} s", run.words.len(), pairs.len(), reference.len(), spread(shown), spread(settled));
            texts[n].get_or_insert(run.text);
        }
    }
    for (b, text) in backends.iter().zip(texts) {
        println!("{b}: {}", text.unwrap_or_default().trim());
    }
    Ok(())
}
