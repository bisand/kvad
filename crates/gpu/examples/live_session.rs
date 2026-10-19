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
//!
//! A backend may end in `+gain=X`, the recording made X times louder and
//! clipped, or in `+level`, or `+level=TARGET:MOST`: the recording is
//! then heard through Kvad for iPhone's gain stage first
//! (`apps/phone/core/src/level.rs`, the same file), as a take with it on is
//! sent. Each such entry is a column of its own, interleaved with the rest,
//! so `gpu-q8,gpu-q8+level` compares the two on one recording. `--from=S`
//! and `--to=S` play a part of it. `--private` prints no text, only the
//! numbers, for a recording that is not to be read; with it, the stretches
//! with nothing settled are listed by where they are, with how many of the
//! file's words were spoken in them. `--script=FILE` is the text the
//! recording is known to say, for one played in a room: each column then
//! says how many of its words the live text has, and the file's. It may be
//! another model's transcript: `--save=PREFIX` writes each entry's file
//! transcript to PREFIX-N.txt, the first being 1.
use kvad::quant::Precision;
use kvad::speech::mel::SAMPLE_RATE;
use kvad::speech::{audio, Live, Request, Transcriber, Transcript};
use kvad::weights::Watcher;
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[path = "../../../apps/phone/core/src/level.rs"]
#[allow(dead_code)]
mod level;

/// `samples` as the phone's gain stage leaves them: at 16 bits, as the
/// microphone gives them to it, a tenth of a second at a time.
fn levelled(samples: &[f32], target: f32, most: f32) -> Vec<f32> {
    let mut pcm: Vec<i16> = samples.iter().map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
    let mut stage = level::Level::new(target, most);
    for chunk in pcm.chunks_mut(1600) {
        stage.feed(chunk);
    }
    pcm.iter().map(|&x| x as f32 / 32768.0).collect()
}

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

/// The longest waits between one settled word and the next, as (from,
/// to) on the audio's clock, longest first.
fn waits(settled: &[f64], end: f64, n: usize) -> Vec<(f64, f64)> {
    let mut at: Vec<f64> = settled.to_vec();
    at.push(end);
    let mut gaps: Vec<(f64, f64)> = std::iter::once(0.0).chain(at.iter().copied()).zip(at.iter().copied()).filter(|(a, b)| b > a).collect();
    gaps.sort_by(|x, y| (y.1 - y.0).total_cmp(&(x.1 - x.0)));
    gaps.truncate(n);
    gaps
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trace = args.iter().any(|a| a == "--trace");
    let private = args.iter().any(|a| a == "--private");
    let cadence: f64 = args.iter().find_map(|a| a.strip_prefix("--cadence=")).map_or(Ok(1.0), str::parse)?;
    let from: f64 = args.iter().find_map(|a| a.strip_prefix("--from=")).map_or(Ok(0.0), str::parse)?;
    let to: f64 = args.iter().find_map(|a| a.strip_prefix("--to=")).map_or(Ok(f64::MAX), str::parse)?;
    let save = args.iter().find_map(|a| a.strip_prefix("--save="));
    let script = args.iter().find_map(|a| a.strip_prefix("--script=")).map(std::fs::read_to_string).transpose()?.map(|t| plain_words(&t));
    let args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [repo, wav, backends, rounds] = args[..] else {
        return Err("usage: live_session REPO WAV BACKEND[+level[=TARGET:MOST]],… ROUNDS [--cadence=SECONDS] [--from=S] [--to=S] [--trace] [--private]".into());
    };
    let all = audio::from_wav(&std::fs::read(wav)?)?;
    let cut = |s: f64| ((s * SAMPLE_RATE as f64) as usize).min(all.len());
    let raw = &all[cut(from)..cut(to)];
    let seconds = raw.len() as f64 / SAMPLE_RATE as f64;
    let rounds: usize = rounds.parse()?;
    let entries: Vec<&str> = backends.split(',').collect();
    let mut models: Vec<(String, Transcriber)> = Vec::new();
    let (mut which, mut heard, mut spoken) = (Vec::new(), Vec::new(), Vec::new());
    for e in &entries {
        let (b, stage) = e.split_once('+').map_or((*e, None), |(b, s)| (b, Some(s)));
        let samples = match stage {
            None => raw.to_vec(),
            Some("level") => levelled(raw, level::TARGET, level::MOST),
            // A fixed gain, clipped: what turning the microphone up does.
            Some(g) if g.starts_with("gain=") => {
                let g: f32 = g[5..].parse()?;
                raw.iter().map(|&x| (x * g).clamp(-1.0, 1.0)).collect()
            }
            Some(s) => {
                let (t, m) = s.strip_prefix("level=").and_then(|v| v.split_once(':')).ok_or("+level or +level=TARGET:MOST")?;
                levelled(raw, t.parse()?, m.parse()?)
            }
        };
        let n = match models.iter().position(|(name, _)| name == b) {
            Some(n) => n,
            None => {
                let t = Instant::now();
                let model = load(repo, b)?;
                // One update over five seconds, so that no round is the first
                // time each kernel runs.
                let mut warm = Live::new(&model, &Request::default())?;
                warm.push(&samples[..5 * SAMPLE_RATE]);
                warm.update(&model)?;
                println!("{b}: {} loaded in {:.1} s", model.backend(), t.elapsed().as_secs_f32());
                models.push((b.to_string(), model));
                models.len() - 1
            }
        };
        let done = models[n].1.transcribe(&samples, &Request { timestamps: true, ..Request::default() }, &mut |_| true)?;
        spoken.push(spoken_words(&done));
        let words: Vec<&str> = spoken.last().unwrap().iter().map(|w| w.0.as_str()).collect();
        // How much of the first entry's it keeps, and how much of it is
        // written twice, which is what a model makes up over noise does.
        let first: Vec<&str> = spoken[0].iter().map(|w| w.0.as_str()).collect();
        let mut seen = std::collections::HashSet::new();
        let again = words.windows(4).filter(|g| !seen.insert(*g)).count();
        let mut line = format!("{e}: the file's transcript has {} words, {} of the first's {}, {:.1}% of its 4-grams again", words.len(), align(&words, &first).len(), first.len(), 100.0 * again as f64 / words.len().max(1) as f64);
        if let Some(known) = &script {
            line += &format!(", {} of the script's {}", align(&words, &known.iter().map(String::as_str).collect::<Vec<_>>()).len(), known.len());
        }
        println!("{line}");
        if let Some(prefix) = save {
            std::fs::write(format!("{prefix}-{}.txt", spoken.len()), words.join(" "))?;
        }
        which.push(n);
        heard.push(samples);
    }
    println!("{seconds:.1} s of audio from {from:.0} s, an update due every {cadence} s");
    println!("columns: min/median/p90/max; a word's lag is from where the file's timed transcript puts its end");
    let mut texts: Vec<Option<String>> = vec![None; entries.len()];
    for round in 1..=rounds {
        for (n, b) in entries.iter().enumerate() {
            let run = live(&models[which[n]].1, &heard[n], cadence, trace && !private)?;
            let reference = &spoken[n];
            let pairs = align(&run.words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &reference.iter().map(|w| w.0.as_str()).collect::<Vec<_>>());
            let shown: Vec<f64> = pairs.iter().map(|&(i, j)| run.words[i].1 - reference[j].1 as f64).collect();
            let settled: Vec<f64> = pairs.iter().map(|&(i, j)| run.words[i].2 - reference[j].1 as f64).collect();
            // Against the first entry's file transcript too, which is the
            // same for every entry.
            let common = align(&run.words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &spoken[0].iter().map(|w| w.0.as_str()).collect::<Vec<_>>()).len();
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
            println!("         {b:8} {} words, {} of {} as the file's ({} of the first's {}), lag shown {} s, settled {} s", run.words.len(), pairs.len(), reference.len(), common, spoken[0].len(), spread(shown), spread(settled));
            let times: Vec<f64> = run.words.iter().map(|w| w.2).collect();
            let longest: Vec<String> = waits(&times, seconds + run.end, 3)
                .iter()
                .map(|&(a, b)| {
                    let said = reference.iter().filter(|w| (w.1 as f64) > a && (w.1 as f64) <= b).count();
                    format!("{:.1} s to {:.0} s ({said} words)", b - a, from + b)
                })
                .collect();
            println!("         {b:8} longest with nothing settled: {}", longest.join(", "));
            if let Some(known) = &script {
                let found = align(&run.words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &known.iter().map(String::as_str).collect::<Vec<_>>()).len();
                println!("         {b:8} {found} of the script's {} words", known.len());
            }
            texts[n].get_or_insert(run.text);
        }
    }
    if !private {
        for (b, text) in entries.iter().zip(texts) {
            println!("{b}: {}", text.unwrap_or_default().trim());
        }
    }
    Ok(())
}
