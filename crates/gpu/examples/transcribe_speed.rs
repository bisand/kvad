//! How fast a Whisper transcribes on each backend, CPU and GPU, in rounds.
//!
//! Every backend is loaded once, then each round transcribes the recording
//! on each in turn, so that a machine that drifts between runs drifts on all
//! of them alike. Reports each run's wall time and its multiple of real
//! time, the transcript's tokens, and whether every backend's transcript is
//! the first one's. Run:
//!
//!     cargo run --release -p kvad-gpu --example transcribe_speed -- \
//!         NbAiLab/nb-whisper-large king.wav cpu-f32,cpu-q8,gpu-f16,gpu-q8 3
//!
//! The model is read from the cache. `--plain` leaves timestamps off; they
//! are on by default, as for the table in `docs/speech-plan.md`.
//!
//! Each run is broken down as well: the windows heard (encoder and
//! cross-attention keys and values), the prompts, the one-token steps, and
//! the rest, which is the host's.
use kvad::quant::Precision;
use kvad::speech::whisper::Config;
use kvad::speech::{audio, mel, Heard, Model, Request, Said, Transcriber};
use kvad::weights::Watcher;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Where a run's time went: windows heard, and decoder passes of one token
/// (a pass over a prompt is counted apart).
#[derive(Default, Clone)]
struct Spent {
    windows: usize,
    hearing: Duration,
    prompts: usize,
    prompting: Duration,
    steps: usize,
    stepping: Duration,
}

/// A model, timed.
struct Timed<M> {
    model: M,
    spent: Arc<Mutex<Spent>>,
}

impl<M: Model> Model for Timed<M> {
    fn config(&self) -> &Config {
        self.model.config()
    }
    fn hear(&self, window: &[f32]) -> Res<Heard> {
        let t = Instant::now();
        let heard = self.model.hear(window);
        let mut s = self.spent.lock().unwrap();
        s.windows += 1;
        s.hearing += t.elapsed();
        heard
    }
    fn said(&self) -> Said {
        self.model.said()
    }
    fn decode(&self, tokens: &[u32], said: &mut Said, heard: &Heard, rows: &[usize]) -> Res<Vec<Vec<f32>>> {
        // `KVAD_SPEED_GAP_US`: idle this long before each step, beyond
        // the host's own work, to see what an idle GPU costs the next one.
        if let Some(us) = std::env::var("KVAD_SPEED_GAP_US").ok().and_then(|v| v.parse().ok()) {
            std::thread::sleep(Duration::from_micros(us));
        }
        let t = Instant::now();
        let out = self.model.decode(tokens, said, heard, rows);
        let mut s = self.spent.lock().unwrap();
        match tokens.len() {
            1 => (s.steps += 1, s.stepping += t.elapsed()),
            _ => (s.prompts += 1, s.prompting += t.elapsed()),
        };
        out
    }
    fn within(&self, work: &mut (dyn FnMut() + Send)) {
        self.model.within(work)
    }
    fn params(&self) -> usize {
        self.model.params()
    }
    fn weight_bytes(&self) -> usize {
        self.model.weight_bytes()
    }
    fn backend(&self) -> String {
        self.model.backend()
    }
}

fn load(repo: &str, backend: &str, spent: &Arc<Mutex<Spent>>) -> Res<Transcriber> {
    let quiet = &mut |_: &str| {};
    let timed = |model| -> Box<dyn Model> { Box::new(Timed { model, spent: spent.clone() }) };
    match backend.split_once('-') {
        Some(("cpu", p)) => {
            let precision = Precision::parse(p).ok_or("cpu-f32, cpu-q8 or cpu-q4")?;
            Transcriber::load_with(repo, quiet, &Watcher::none(), &mut |config, files, _| {
                Ok(Box::new(Timed { model: kvad::speech::whisper::Cpu::load(config, &files.weights, precision)?, spent: spent.clone() }))
            })
        }
        Some(("gpu", p)) => {
            let (dtype, quant) = match kvad_gpu::model::parse_dtype(p) {
                Some(dt) => (dt, None),
                None => (candle_core::DType::F32, kvad_gpu::model::parse_quant(p).ok_or("gpu-f32, gpu-f16, gpu-bf16, gpu-q8 or gpu-q4")?),
            };
            let device = kvad_gpu::model::pick_device(None)?;
            Transcriber::load_with(repo, quiet, &Watcher::none(), &mut |config, files, progress| {
                Ok(timed(kvad_gpu::speech::GpuWhisper::load(repo, config, &files.weights, device.clone(), dtype, quant, progress)?))
            })
        }
        _ => Err(format!("`{backend}`: a backend is cpu-PRECISION or gpu-PRECISION").into()),
    }
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let plain = args.iter().any(|a| a == "--plain");
    let args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [repo, wav, backends, rounds] = args[..] else {
        return Err("usage: transcribe_speed REPO WAV BACKEND,BACKEND,… ROUNDS [--plain]".into());
    };
    let samples = audio::from_wav(&std::fs::read(wav)?)?;
    let seconds = samples.len() as f32 / mel::SAMPLE_RATE as f32;
    let rounds: usize = rounds.parse()?;
    let backends: Vec<&str> = backends.split(',').collect();
    let mut loaded = Vec::new();
    for b in &backends {
        let t = std::time::Instant::now();
        let spent = Arc::new(Mutex::new(Spent::default()));
        let model = load(repo, b, &spent)?;
        println!("{b}: {} loaded in {:.1} s", model.backend(), t.elapsed().as_secs_f32());
        loaded.push((model, spent));
    }
    let req = Request { timestamps: !plain, ..Request::default() };
    let mut first: Option<String> = None;
    for round in 1..=rounds {
        for (b, (model, spent)) in backends.iter().zip(&loaded) {
            *spent.lock().unwrap() = Spent::default();
            let t = std::time::Instant::now();
            let done = model.transcribe(&samples, &req, &mut |_| true)?;
            let wall = t.elapsed().as_secs_f32();
            let s = spent.lock().unwrap().clone();
            let ms = |d: Duration, n: usize| d.as_secs_f64() * 1000.0 / n.max(1) as f64;
            let tokens: usize = done.segments.iter().map(|s| s.tokens.len()).sum();
            let same = match &first {
                None => {
                    first = Some(done.text.clone());
                    "first"
                }
                Some(f) if *f == done.text => "same",
                Some(_) => "differs",
            };
            println!(
                "round {round} {b:9} {wall:6.2} s  {:5.1}x real time  {tokens} tokens  {same}  | {} windows {:.0} ms each, {} prompts {:.1} ms, {} steps {:.2} ms, rest {:.2} s",
                seconds / wall,
                s.windows,
                ms(s.hearing, s.windows),
                s.prompts,
                ms(s.prompting, s.prompts),
                s.steps,
                ms(s.stepping, s.steps),
                wall - (s.hearing + s.prompting + s.stepping).as_secs_f32(),
            );
        }
    }
    if let Some(text) = first {
        println!("{}", text.trim());
    }
    Ok(())
}
