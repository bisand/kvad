//! What live transcription would cost: a recording played in at real time,
//! decoded again about once a second as it grows, on each backend, in
//! rounds.
//!
//! This is the loop `docs/speech-plan.md` lays out under "Live, from the
//! phone", run on a file instead of a phone, to see what it costs before
//! the endpoint is built. At each update:
//!
//! 1. The audio heard so far since the window's start, padded to 30 s as
//!    always, is turned into a spectrogram and heard (the encoder, and every
//!    block's cross keys and values).
//! 2. The tokens already settled in this window go to the decoder as one
//!    prompt, after the window's start tokens and the finished windows'
//!    text as context, as a file's windows have it.
//! 3. Greedy one-token steps write the rest, the partial, until the end of
//!    text.
//!
//! A word is settled once two updates in a row agree on it and on every
//! word before it (LocalAgreement-2, from whisper_streaming). Words are
//! compared, not timestamps: those move a step or two between guesses
//! whose words do not, and compared, they would never settle. What is
//! settled is the newer guess's tokens through the last agreed word,
//! timestamps between included, and never a timestamp after it (see
//! `live`). A tail that starts going round in circles is stopped there.
//!
//! When the window holds more than [`CUT`] seconds, it is finished where a
//! settled segment ends and the next begins, and the next window starts
//! there with what was settled after it kept. A window that reaches 30 s
//! with nowhere to cut begins again at the newest 30 s, and the run counts
//! it as forced. When the recording ends, the last update's words are all
//! settled, as a phone hanging up would have them.
//!
//! The clock is the wall's: an update is due every `--cadence` seconds (1 by
//! default), and an update still running when the next is due skips it, as
//! the plan says the server will. So the backend idles between updates as
//! it would live, and comes back cold as it would. `--fast` runs the
//! updates back to back instead, the clock jumping to when each was due,
//! to see what the idling costs. Run:
//!
//!     cargo run --release -p kvad-gpu --example live_speed -- \
//!         NbAiLab/nb-whisper-large king.wav gpu-q8,gpu-f16 3
//!
//! The model is read from the cache. Each run reports its updates' wall
//! time, the share of the time the backend was busy (its duty), how many
//! updates were due while one was running, and, for every word, how long
//! after it was spoken it first showed as a partial and when it settled.
//! "Shown" is since when the partial has shown the word unchanged, with
//! every word before it. "Spoken" is where the same backend's transcript of
//! the whole file, timed, puts the word's end, its segment's time shared
//! out by letters; the live words are matched to those by the longest
//! common run. Whisper's segment times are themselves a second or two out
//! at times, so a word's lag is good to about that, and some come out
//! negative: read the median and the 90th percentile, not the extremes.
//! `--trace` prints every update: what it heard, what it settled, and its
//! partial.
//!
//! As with `transcribe_speed`, load one process per set of backends that
//! fit beside each other: the GPU's in one, the CPU's in another.
use kvad::quant::Precision;
use kvad::speech::mel::{self, SAMPLE_RATE};
use kvad::speech::transcribe::{Rules, TIME_STEP};
use kvad::speech::{audio, Model, Request, Transcriber, Transcript};
use kvad::weights::Watcher;
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A window holding more than this many seconds is finished at its last
/// settled segment end. whisper_streaming trims at 15; the encoder costs
/// the same whatever the window holds, and a longer window only lengthens
/// the prompt, so 20 leaves more segments to cut at.
const CUT: f32 = 20.0;

/// How many tokens written again make a loop.
const LAP: usize = 8;

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

/// `work`, where the model's work runs best (`Model::within`), as the
/// transcriber's own loop runs it.
fn within<T: Send>(model: &dyn Model, work: impl FnOnce() -> Res<T> + Send) -> Res<T> {
    let mut work = Some(work);
    let mut out: Option<Result<T, String>> = None;
    model.within(&mut || {
        if let Some(w) = work.take() {
            out = Some(w().map_err(|e| e.to_string()));
        }
    });
    out.ok_or("the model ran nothing it was given")?.map_err(Into::into)
}

/// Seconds since the recording began to play: the wall's, or, with
/// `--fast`, one moved only by what the updates cost.
struct Clock {
    start: Instant,
    fast: Option<f64>,
}

impl Clock {
    fn now(&self) -> f64 {
        self.fast.unwrap_or_else(|| self.start.elapsed().as_secs_f64())
    }
    fn wait_until(&mut self, t: f64) {
        match &mut self.fast {
            Some(v) => *v = v.max(t),
            None => std::thread::sleep(Duration::from_secs_f64((t - self.now()).max(0.0))),
        }
    }
    fn spent(&mut self, d: f64) {
        if let Some(v) = &mut self.fast {
            *v += d;
        }
    }
}

/// What one update cost.
#[derive(Clone, Copy)]
struct Cost {
    wall: f64,
    /// The spectrogram, on the host.
    mel: f64,
    /// Heard, and the prompt through the decoder: one wait on a GPU, since
    /// hearing does not wait for itself.
    heard_and_prompt: f64,
    prompt: usize,
    steps: usize,
    stepping: f64,
    /// A plain window's timed decode, to find where to cut it.
    cutting: f64,
}

/// One update's guess: the window's tokens after its start tokens, timed
/// or plain as `start` says.
fn update(tr: &Transcriber, window: &[f32], context: &[u32], start: &[u32], settled: &[u32]) -> Res<(Vec<u32>, Cost)> {
    let t0 = Instant::now();
    let m = &*tr.model;
    let t = &tr.tokens;
    let ctx = m.config().text_ctx;
    let mel = within(m, || Ok(mel::window(&mel::log_mel(window, mel::N_SAMPLES - window.len(), m.config().n_mels), 0)))?;
    let mel_s = t0.elapsed().as_secs_f64();
    let mut prompt = Vec::new();
    if !context.is_empty() {
        prompt.push(t.sot_prev);
        let keep = (ctx / 2 - 1).min(context.len());
        prompt.extend_from_slice(&context[context.len() - keep..]);
    }
    prompt.extend_from_slice(start);
    prompt.extend_from_slice(settled);
    let rules = Rules { tokens: t, timestamps: !start.contains(&t.no_timestamps) };
    let limit = (ctx / 2).min(ctx - prompt.len());
    let mut out = settled.to_vec();
    let (mut steps, mut stepping, mut heard_and_prompt) = (0, Duration::ZERO, 0.0);
    within(m, || {
        let t1 = Instant::now();
        let heard = m.hear(&mel)?;
        let mut said = m.said();
        let mut logits = m.decode(&prompt, &mut said, &heard, &[prompt.len() - 1])?.remove(0);
        heard_and_prompt = t1.elapsed().as_secs_f64();
        while out.len() - settled.len() < limit {
            rules.apply(&mut logits, &out);
            let next = logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
            if next == t.eot || prompt.len() + out.len() - settled.len() >= ctx {
                break;
            }
            out.push(next);
            // A tail going round in circles is stopped where it starts its
            // second lap, a stand-in for the file path's loop rules: eight
            // tokens again, which speech repeats less than Whisper does.
            let n = out.len();
            if n - settled.len() >= LAP && out[..n - 1].windows(LAP).any(|w| w == &out[n - LAP..]) {
                out.truncate(n - LAP);
                break;
            }
            let ts = Instant::now();
            logits = m.decode(&[next], &mut said, &heard, &[0])?.remove(0);
            stepping += ts.elapsed();
            steps += 1;
        }
        Ok(())
    })?;
    let cost = Cost { wall: t0.elapsed().as_secs_f64(), mel: mel_s, heard_and_prompt, prompt: prompt.len(), steps, stepping: stepping.as_secs_f64(), cutting: 0.0 };
    Ok((out, cost))
}

/// Each word of `ids` as compared ([`plain`]), with where its last token
/// ends; a word is a token beginning with a space and those after it that
/// do not. Words of no letters, as a dash, are left out.
fn word_ends(tr: &Transcriber, ids: &[u32]) -> Vec<(String, usize)> {
    let space = |i: u32| tr.tokenizer.id_to_token(i).is_some_and(|p| p.starts_with('Ġ'));
    let mut out = Vec::new();
    let mut begin = 0;
    for k in 0..ids.len() {
        if ids.get(k + 1).is_none_or(|&n| space(n) || n >= tr.tokens.eot) {
            let word = plain(&tr.text(&ids[begin..=k]).unwrap_or_default());
            if !word.is_empty() {
                out.push((word, k + 1));
            }
            begin = k + 1;
        }
    }
    out
}

/// Where a plain window can be finished: one timed decode of it, and the
/// end of its last segment that is followed by another and whose words
/// are all settled, as how many of the settled tokens it takes and how far
/// into the window it ends.
fn plain_cut(tr: &Transcriber, window: &[f32], context: &[u32], timed: &[u32], settled: &[u32]) -> Res<Option<(usize, f32)>> {
    let t = &tr.tokens;
    let is_ts = |i: u32| i >= t.timestamp_begin;
    let (guess, _) = update(tr, window, context, timed, &[])?;
    let text: Vec<u32> = guess.iter().copied().filter(|&i| !is_ts(i)).collect();
    let heard = word_ends(tr, &text);
    let have = word_ends(tr, settled);
    let pairs = align(&heard.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &have.iter().map(|w| w.0.as_str()).collect::<Vec<_>>());
    // Segment ends, two timestamps side by side, as how many text tokens
    // come before each.
    let mut ends = Vec::new();
    let mut n = 0;
    for i in 0..guess.len() {
        if !is_ts(guess[i]) {
            n += 1;
        } else if i >= 1 && !is_ts(guess[i - 1]) && guess.get(i + 1).is_some_and(|&x| is_ts(x)) {
            ends.push((n, (guess[i] - t.timestamp_begin) as f32 * TIME_STEP));
        }
    }
    for &(n, at) in ends.iter().rev() {
        // The segment's last word, among the settled ones.
        let Some(last) = heard.iter().rposition(|w| w.1 <= n) else { continue };
        if let Some(&(_, j)) = pairs.iter().find(|&&(i, _)| i == last) {
            return Ok(Some((have[j].1, at)));
        }
    }
    Ok(None)
}

/// A settled token, as the phone would have been sent it.
struct Sent {
    id: u32,
    /// Since when the partial has shown it, and when it was settled.
    shown: f64,
    settled: f64,
}

/// What one run of the recording came to.
struct Run {
    costs: Vec<Cost>,
    /// Updates that were due while one was running, and so never ran.
    skipped: usize,
    cuts: usize,
    forced: usize,
    busy: f64,
    end: f64,
    sent: Vec<Sent>,
}

fn live(tr: &Transcriber, samples: &[f32], cadence: f64, fast: bool, plain: bool, trace: bool) -> Res<Run> {
    let t = &tr.tokens;
    let is_ts = |i: u32| i >= t.timestamp_begin;
    let language = t.default_language.as_deref().and_then(|c| t.language(c)).or_else(|| t.language("no")).unwrap_or(0);
    let mut start = vec![t.sot];
    if t.multilingual {
        start.extend([language, t.transcribe]);
    }
    let timed = start.clone();
    if plain {
        start.push(t.no_timestamps);
    }
    let duration = samples.len() as f64 / SAMPLE_RATE as f64;
    let mut clock = Clock { start: Instant::now(), fast: fast.then_some(0.0) };
    let mut run = Run { costs: Vec::new(), skipped: 0, cuts: 0, forced: 0, busy: 0.0, end: 0.0, sent: Vec::new() };
    // The window's start in samples, the finished windows' tokens, this
    // window's settled tokens, and the last guess's words (its tokens
    // other than timestamps) with since when each has been shown.
    let (mut from, mut context, mut settled) = (0usize, Vec::<u32>::new(), Vec::<u32>::new());
    let (mut last, mut shown) = (Vec::<u32>::new(), Vec::<f64>::new());
    let words = |ids: &[u32]| ids.iter().copied().filter(|&i| !is_ts(i)).collect::<Vec<u32>>();
    let mut tick = 1usize;
    loop {
        clock.wait_until(tick as f64 * cadence);
        let have = ((clock.now() * SAMPLE_RATE as f64) as usize).min(samples.len());
        let ended = have == samples.len();
        let seconds = |n: usize| n as f32 / SAMPLE_RATE as f32;

        // Finish a full window where a settled segment ends and the next
        // begins, two timestamps side by side, as a file's windows are cut.
        // A lone closing timestamp is not enough: the model closes a
        // segment early when it has not heard the next, and a window
        // started there hears the end of it again. What was settled after
        // the cut is kept, its timestamps moved to the new start.
        let cut_at = |s: &[u32]| (2..s.len()).rev().find(|&i| is_ts(s[i]) && is_ts(s[i - 1]) && !is_ts(s[i - 2]));
        let mut cutting = 0.0;
        if plain && seconds(have - from) > CUT {
            // A plain window has no timestamps to cut at: one timed decode
            // finds them, and it is paid for in this update.
            let t0 = Instant::now();
            let at = plain_cut(tr, &samples[from..have], &context, &timed, &settled)?;
            cutting = t0.elapsed().as_secs_f64();
            if let Some((i, at)) = at {
                context.extend_from_slice(&settled[..i]);
                settled.drain(..i);
                from += (at * SAMPLE_RATE as f32) as usize;
                last.drain(..i.min(last.len()));
                shown.drain(..i.min(shown.len()));
                run.cuts += 1;
                if trace {
                    println!("  cut at {:.2} s", seconds(from));
                }
            }
        } else if seconds(have - from) > CUT {
            if let Some(i) = cut_at(&settled) {
                let step = settled[i - 1] - t.timestamp_begin;
                let back = |x: u32| if is_ts(x) { x.saturating_sub(step).max(t.timestamp_begin) } else { x };
                let gone = words(&settled[..i]).len();
                context.extend_from_slice(&settled[..i]);
                settled = settled[i..].iter().map(|&x| back(x)).collect();
                from += step as usize * (SAMPLE_RATE as f32 * TIME_STEP) as usize;
                last.drain(..gone.min(last.len()));
                shown.drain(..gone.min(shown.len()));
                run.cuts += 1;
                if trace {
                    println!("  cut at {:.2} s", seconds(from));
                }
            }
        }
        if seconds(have - from) > 30.0 {
            // Nothing settled to cut at: begin again at the newest 30 s,
            // and count it, since a phone would lose its partial.
            context.extend_from_slice(&settled);
            from = have - mel::N_SAMPLES;
            settled.clear();
            last.clear();
            shown.clear();
            run.forced += 1;
        }

        let (guess, mut cost) = update(tr, &samples[from..have], &context, &start, &settled)?;
        cost.cutting = cutting;
        cost.wall += cutting;
        clock.spent(cost.wall);
        let now = clock.now();
        run.busy += cost.wall;
        run.costs.push(cost);

        // Two guesses are compared by their words, not their timestamps,
        // which move by a step or two from one guess to the next though
        // the words do not. Settled is everything this guess writes up to
        // the last word the two agree on, its timestamps between included.
        // A timestamp after that word is not: it is often the model closing
        // a segment where the audio stops, and settled, it would bind every
        // later guess to close there, since a timestamp or the end must
        // follow it and the end is what the model picks.
        let said = words(&guess);
        let agree = said.iter().zip(&last).take_while(|(a, b)| a == b).count();
        shown.truncate(agree);
        shown.resize(said.len(), now);
        let upto = match ended {
            true => guess.len(),
            false => guess.iter().enumerate().filter(|&(_, &x)| !is_ts(x)).nth(agree.wrapping_sub(1)).map_or(0, |(i, _)| i + 1).max(settled.len()),
        };
        let offset = seconds(from);
        let mut nth = words(&settled).len();
        for &id in &guess[settled.len()..upto] {
            let at = if is_ts(id) { now } else { nth += 1; shown[nth - 1] };
            run.sent.push(Sent { id, shown: at, settled: now });
        }
        if trace {
            let show = |ids: &[u32]| {
                ids.iter().map(|&i| if is_ts(i) { format!("<{:.2}>", offset + (i - t.timestamp_begin) as f32 * TIME_STEP) } else { tr.text(&[i]).unwrap_or_default() }).collect::<String>()
            };
            println!("  {now:6.2} s  heard {:.2}-{:.2}  {:.0} ms  settled +[{}]  partial [{}]", offset, seconds(have), 1000.0 * cost.wall, show(&guess[settled.len()..upto.max(settled.len())]), show(&guess[upto.max(settled.len())..]));
        }
        if upto > settled.len() {
            settled = guess[..upto].to_vec();
        }
        last = said;
        if ended {
            run.end = now - duration;
            break;
        }
        let next = (now / cadence).floor() as usize + 1;
        run.skipped += next - tick - 1;
        tick = next;
    }
    Ok(run)
}

/// A word as compared: its letters and digits, in lower case.
fn plain(word: &str) -> String {
    word.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Each word of what was sent, with when it was first shown and settled:
/// the times of its last token.
fn sent_words(tr: &Transcriber, sent: &[Sent]) -> Vec<(String, f64, f64)> {
    let text: Vec<&Sent> = sent.iter().filter(|s| s.id < tr.tokens.eot).collect();
    let space = |s: &Sent| tr.tokenizer.id_to_token(s.id).is_some_and(|p| p.starts_with('Ġ'));
    let mut out = Vec::new();
    let mut begin = 0;
    for k in 0..text.len() {
        if text.get(k + 1).is_none_or(|n| space(n)) {
            let ids: Vec<u32> = text[begin..=k].iter().map(|s| s.id).collect();
            let word = plain(&tr.text(&ids).unwrap_or_default());
            if !word.is_empty() {
                out.push((word, text[k].shown, text[k].settled));
            }
            begin = k + 1;
        }
    }
    out
}

/// Each word of the file's own transcript, timed, and when it was spoken:
/// by where its last letter falls in its segment, the segment's time
/// shared out by letters.
fn spoken_words(done: &Transcript) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for seg in &done.segments {
        let words: Vec<String> = seg.text.split_whitespace().map(plain).filter(|w| !w.is_empty()).collect();
        let total: usize = words.iter().map(|w| w.chars().count()).sum();
        let mut upto = 0;
        for w in words {
            upto += w.chars().count();
            out.push((w, seg.start + (seg.end - seg.start) * upto as f32 / total.max(1) as f32));
        }
    }
    out
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

fn spread(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!("{:.2}/{:.2}/{:.2}/{:.2}", at(0.0), at(0.5), at(0.9), at(1.0))
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fast = args.iter().any(|a| a == "--fast");
    let trace = args.iter().any(|a| a == "--trace");
    // Partials timed, as the plan first had them, or plain; `--both` takes
    // each backend through both in every round.
    let modes: Vec<bool> = match (args.iter().any(|a| a == "--both"), args.iter().any(|a| a == "--plain")) {
        (true, _) => vec![false, true],
        (false, plain) => vec![plain],
    };
    let cadence: f64 = args.iter().find_map(|a| a.strip_prefix("--cadence=")).map_or(Ok(1.0), str::parse)?;
    let args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [repo, wav, backends, rounds] = args[..] else {
        return Err("usage: live_speed REPO WAV BACKEND,BACKEND,… ROUNDS [--cadence=SECONDS] [--plain | --both] [--fast] [--trace]".into());
    };
    let samples = audio::from_wav(&std::fs::read(wav)?)?;
    let seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let rounds: usize = rounds.parse()?;
    let backends: Vec<&str> = backends.split(',').collect();
    let mut loaded = Vec::new();
    // What each backend makes of the whole file, timed: the words' times
    // to measure a lag from.
    let mut spoken = Vec::new();
    for b in &backends {
        let t = Instant::now();
        let model = load(repo, b)?;
        // One update over the first five seconds, so that the first round
        // is not also the first time each kernel runs.
        update(&model, &samples[..5 * SAMPLE_RATE], &[], &[model.tokens.sot], &[])?;
        println!("{b}: {} loaded in {:.1} s", model.backend(), t.elapsed().as_secs_f32());
        let done = model.transcribe(&samples, &Request { timestamps: true, ..Request::default() }, &mut |_| true)?;
        spoken.push(spoken_words(&done));
        loaded.push(model);
    }
    println!("{seconds:.1} s of audio, an update due every {cadence} s{}", if fast { ", back to back (--fast)" } else { "" });
    println!("columns: min/median/p90/max; a word's lag is from where the file's timed transcript puts its end to the update that sent it");
    let runs: Vec<(usize, bool, String)> =
        (0..backends.len()).flat_map(|n| modes.iter().map(move |&p| (n, p))).map(|(n, p)| (n, p, format!("{} {}", backends[n], if p { "plain" } else { "timed" }))).collect();
    let mut texts: Vec<Option<String>> = vec![None; runs.len()];
    for round in 1..=rounds {
        for (k, (n, plain, b)) in runs.iter().enumerate() {
            let (n, tr) = (*n, &loaded[*n]);
            let run = live(tr, &samples, cadence, fast, *plain, trace)?;
            let walls: Vec<f64> = run.costs.iter().map(|c| c.wall).collect();
            let avg = |f: fn(&Cost) -> f64| run.costs.iter().map(f).sum::<f64>() / run.costs.len() as f64;
            let steps: usize = run.costs.iter().map(|c| c.steps).sum();
            let stepping: f64 = run.costs.iter().map(|c| c.stepping).sum();
            let prompt = run.costs.iter().map(|c| c.prompt).sum::<usize>() / run.costs.len();
            let cutting: Vec<f64> = run.costs.iter().map(|c| c.cutting).filter(|&c| c > 0.0).collect();
            let ws = sent_words(tr, &run.sent);
            let reference = &spoken[n];
            let pairs = align(&ws.iter().map(|w| w.0.as_str()).collect::<Vec<_>>(), &reference.iter().map(|w| w.0.as_str()).collect::<Vec<_>>());
            let shown: Vec<f64> = pairs.iter().map(|&(i, j)| ws[i].1 - reference[j].1 as f64).collect();
            let settled: Vec<f64> = pairs.iter().map(|&(i, j)| ws[i].2 - reference[j].1 as f64).collect();
            println!(
                "round {round} {b:15} {} updates, wall {} s, duty {:3.0}%, {} skipped, {} cuts{}, ends {:.2} s after the audio",
                run.costs.len(),
                spread(walls),
                100.0 * run.busy / (seconds + run.end),
                run.skipped,
                run.cuts,
                if run.forced > 0 { format!(" ({} forced)", run.forced) } else { String::new() },
                run.end,
            );
            println!(
                "         {b:15} per update: mel {:.0} ms, heard+prompt {:.0} ms ({prompt} tokens), {:.1} steps at {:.1} ms{} | {} words, {} as the file's, lag shown {} s, settled {} s",
                1000.0 * avg(|c| c.mel),
                1000.0 * avg(|c| c.heard_and_prompt),
                steps as f64 / run.costs.len() as f64,
                1000.0 * stepping / steps.max(1) as f64,
                match cutting.len() {
                    0 => String::new(),
                    n => format!(", {n} timed decodes to cut at {:.0} ms", 1000.0 * cutting.iter().sum::<f64>() / n as f64),
                },
                ws.len(),
                pairs.len(),
                spread(shown),
                spread(settled),
            );
            let ids: Vec<u32> = run.sent.iter().map(|s| s.id).collect();
            texts[k].get_or_insert(tr.text(&ids)?);
        }
    }
    for ((_, _, b), text) in runs.iter().zip(texts) {
        println!("{b}: {}", text.unwrap_or_default().trim());
    }
    Ok(())
}
