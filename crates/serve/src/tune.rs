//! A LoRA run for an image model, as a job (#77).
//!
//! `kvad-gpu tune` trains a LoRA in the process that asks for it. This is
//! the same loop, [`kvad_gpu::image::tune::run`], asked for over HTTP and
//! followed like any other job; and it does not run in the server.
//!
//! # A process of its own
//!
//! The server starts itself again as `kvad-serve tune-worker`, hands that
//! process the run on its standard input, and reads what it does a line at a
//! time from its standard output. Three reasons, in the order they matter:
//!
//! - **A run that does not fit must end itself, and only itself.** On macOS a
//!   Metal allocation past physical memory is not refused; the machine
//!   compresses, swaps and stops answering (`kvad_gpu::cap`). The trainer's
//!   guard is a ceiling on the process's footprint that ends the process at
//!   it. In the server that would be the server, and every model it holds,
//!   and the footprint watched would be theirs as well as the run's.
//! - **What a run took is given back.** A process that has held 10 GB of
//!   GPU buffers and the allocator's large blocks does not return all of it
//!   while it lives. A server is up for weeks; a worker is gone with its
//!   run.
//! - **A run that crashes is a failed job**, not a server that has to be
//!   started again and a row that says "running" until it is.
//!
//! What it costs is that the worker loads the model itself: a model the
//! server already holds for drawing is not shared with it.
//!
//! # Memory
//!
//! The run is charged to the server's budget before it starts
//! ([`crate::scheduler::Scheduler::reserve`]), as a model is, and is refused
//! if what is left beside the models in memory is less than [`need`]. The
//! worker's own ceiling is that much and [`OVER`] more.
//!
//! # Stopping
//!
//! The job's cancel flag becomes the line `stop` on the worker's input,
//! which raises the flag the loop reads: once a step, and once a denoising
//! step while it draws a sample. The run then measures where it is, writes
//! its last step beside its best, and says so. If the server goes, the
//! worker's input closes, and it ends at once.

use crate::jobs::{Drawn, Jobs, LoraParams, Measure, Params, Update};
use crate::scheduler::Reservation;
use rusqlite::params;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The first argument that makes `kvad-serve` a worker and not a server.
pub const WORKER: &str = "tune-worker";

/// The sizes a run may be asked for: what was measured (docs/tune.md), and
/// what lies under it.
pub const SIZES: std::ops::RangeInclusive<usize> = 256..=1024;

/// How far past [`need`] the worker's footprint may go before it ends
/// itself. The charge is a peak measured on single runs; this is the room
/// for it to have been low.
pub const OVER: u64 = 2_000_000_000;

/// What a run at `side` pixels is charged, in bytes, drawing samples or not.
///
/// From docs/tune.md's measurements of SDXL at rank 16 on an M5 Pro: a
/// run's own peak is 7.1 GB at 512² and 10.4 at 1024² (10.6 the first time,
/// when the pictures are still to be read), and it grows with the picture's
/// area, which this draws a line through: 7.1, 8.5 and 10.6 GB. Samples
/// reach 8.7 GB whatever the run's size, so a small run that draws them is
/// charged for that. And 1.5 GB over, because every one of those is one
/// run's figure.
pub fn need(side: usize, sampling: bool) -> u64 {
    let run = 5.9e9 + 4.7e9 * (side as f64 / 1024.0).powi(2);
    let drawing = if sampling { 8.7e9 } else { 0.0 };
    (run.max(drawing) + 1.5e9) as u64
}

/// Everything the worker needs to run, as the one line it is handed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Work {
    pub repo: String,
    pub data: PathBuf,
    pub out: PathBuf,
    pub from: Option<PathBuf>,
    pub size: usize,
    pub rank: usize,
    pub alpha: Option<f64>,
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    pub eval_every: usize,
    pub holdout: Option<usize>,
    pub samples: Vec<String>,
    pub sample_size: Option<usize>,
    pub sample_steps: usize,
    pub sample_dir: PathBuf,
    pub ffmpeg: PathBuf,
    /// The footprint the worker ends itself at, in gigabytes.
    pub cap_gb: f64,
    /// The server's data directory, where it named one: the worker has read
    /// no config file, and must look where the server does.
    pub data_dir: Option<PathBuf>,
}

/// How a run ended, in numbers. A loss that is not a number is `None`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Outcome {
    pub out: PathBuf,
    pub last: Option<PathBuf>,
    pub layers: usize,
    pub trained: usize,
    pub pictures: usize,
    pub held_out: usize,
    pub base_val: Option<f64>,
    pub best_val: Option<f64>,
    pub best_step: usize,
    pub last_val: Option<f64>,
    pub steps: usize,
    pub elapsed_secs: f64,
    pub stopped: bool,
}

/// One line of the worker's output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "is", rename_all = "snake_case")]
pub enum Line {
    Say { line: String },
    Step { step: usize, steps: usize, loss: f64, secs: f64 },
    Measured { step: usize, val_loss: f64, elapsed_secs: f64 },
    Saved { step: usize },
    Sampled { step: usize, prompt: usize, file: PathBuf },
    Done(Outcome),
    Failed { error: String },
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// Be the worker: read one [`Work`] from standard input, run it, and write a
/// [`Line`] for everything it does. Never returns.
#[cfg(feature = "gpu")]
pub fn worker() -> ! {
    use std::io::{BufRead, Write};
    fn emit(line: &Line) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{}", serde_json::to_string(line).unwrap_or_default());
        let _ = out.flush();
    }
    let finite = |v: f32| v.is_finite().then_some(v as f64);

    let stop = Arc::new(AtomicBool::new(false));
    let result = (|| -> Res<Outcome> {
        let mut first = String::new();
        std::io::stdin().lock().read_line(&mut first)?;
        let work: Work = serde_json::from_str(&first).map_err(|e| format!("the run was not handed over whole: {e}"))?;

        // The rest of the input is the server's to speak on: `stop`, or its
        // end, which is the server's own.
        let raised = Arc::clone(&stop);
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                match line.as_deref() {
                    Ok("stop") => raised.store(true, Ordering::Relaxed),
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            std::process::exit(1);
        });

        kvad::weights::set_data_dir(work.data_dir.clone());
        kvad_gpu::cap::at(work.cap_gb);
        let device = kvad_gpu::model::pick_device(None)?;
        if !device.is_metal() && !device.is_cuda() {
            return Err("an image model is trained on the GPU and nowhere else, and this machine has none that candle can use".into());
        }

        let mut opts = kvad_gpu::image::tune::Options::new(&work.repo, &work.data, &work.out, &work.ffmpeg);
        opts.from = work.from.clone();
        opts.size = work.size;
        opts.rank = work.rank;
        opts.alpha = work.alpha;
        opts.steps = work.steps;
        opts.lr = work.lr;
        opts.seed = work.seed;
        opts.eval_every = work.eval_every;
        opts.holdout = work.holdout;
        opts.samples = work.samples.clone();
        opts.sample_size = work.sample_size;
        opts.sample_steps = work.sample_steps;
        opts.sample_dir = Some(work.sample_dir.clone());
        opts.cancel = Some(Arc::clone(&stop));

        let started = std::time::Instant::now();
        let mut say = |line: &str| emit(&Line::Say { line: line.to_string() });
        let mut watch = |e: kvad_gpu::image::tune::Event| {
            use kvad_gpu::image::tune::Event;
            match e {
                Event::Step { step, steps, loss, secs } => emit(&Line::Step { step, steps, loss: loss as f64, secs }),
                // A loss that is not a number has no place in a chart, and
                // is said in words.
                Event::Measured { step, val_loss } => match finite(val_loss) {
                    Some(val_loss) => emit(&Line::Measured { step, val_loss, elapsed_secs: started.elapsed().as_secs_f64() }),
                    None => emit(&Line::Say { line: format!("step {step}: the validation loss is not a number") }),
                },
                Event::Saved { step, .. } => emit(&Line::Saved { step }),
                Event::Sampled { step, prompt, path } => emit(&Line::Sampled { step, prompt, file: path }),
            }
        };
        let s = kvad_gpu::image::tune::run(&opts, &device, &mut say, &mut watch)?;
        Ok(Outcome {
            out: s.out,
            last: s.last,
            layers: s.layers,
            trained: s.trained,
            pictures: s.pictures,
            held_out: s.held_out,
            base_val: finite(s.base_val),
            best_val: finite(s.best_val),
            best_step: s.best_step,
            last_val: finite(s.last_val),
            steps: s.steps,
            elapsed_secs: s.elapsed_secs,
            stopped: s.stopped,
        })
    })();
    match result {
        Ok(outcome) => emit(&Line::Done(outcome)),
        Err(e) => emit(&Line::Failed { error: e.to_string() }),
    }
    std::process::exit(0)
}

#[cfg(not(feature = "gpu"))]
pub fn worker() -> ! {
    eprintln!("this kvad-serve was built without the GPU backend, and trains no image model");
    std::process::exit(2)
}

// ---------------------------------------------------------------------------
// The server's side
// ---------------------------------------------------------------------------

/// What the worker's lines are made into: rows, and updates for whoever is
/// watching. Apart from the worker itself, so that the whole of what the
/// server does with a run can be given lines and looked at.
pub struct Follower {
    db: crate::db::Db,
    id: i64,
    events: broadcast::Sender<Update>,
    /// The losses and seconds of the steps since the last measurement.
    since: Vec<(f64, f64)>,
    /// The last measurement, for the line that says it was saved.
    last: Option<Measure>,
    /// How the worker said it ended, once it has.
    pub ended: Option<Result<Outcome, String>>,
}

impl Follower {
    pub fn new(db: crate::db::Db, id: i64, events: broadcast::Sender<Update>) -> Self {
        Follower { db, id, events, since: Vec::new(), last: None, ended: None }
    }

    fn send(&self, update: Update) {
        let _ = self.events.send(update);
    }

    fn keep(&self, m: &Measure) {
        let _ = self.db.with(|c| {
            c.execute(
                "INSERT OR REPLACE INTO tune_metrics (job, step, val_loss, train_loss, secs_per_step, elapsed_secs, saved)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![self.id, m.step, m.val_loss, m.train_loss, m.secs_per_step, m.elapsed_secs, m.saved as i64],
            )
        });
    }

    /// One line of the worker's output. A line that is not one of
    /// [`Line`]'s is something a library printed, and is passed on as said.
    pub fn take(&mut self, line: &str) {
        let line = match serde_json::from_str::<Line>(line) {
            Ok(line) => line,
            Err(_) if line.trim().is_empty() => return,
            Err(_) => Line::Say { line: line.to_string() },
        };
        match line {
            Line::Say { line } => self.send(Update::Status { message: line }),
            Line::Step { step, steps, loss, secs } => {
                self.since.push((loss, secs));
                self.send(Update::Stepped { step, steps, loss, secs });
            }
            Line::Measured { step, val_loss, elapsed_secs } => {
                let n = self.since.len() as f64;
                let mean = |of: fn(&(f64, f64)) -> f64| (n > 0.0).then(|| self.since.iter().map(of).sum::<f64>() / n);
                let m = Measure { step: step as i64, val_loss, train_loss: mean(|s| s.0), secs_per_step: mean(|s| s.1), elapsed_secs, saved: false };
                self.since.clear();
                self.keep(&m);
                self.send(Update::Measured(m.clone()));
                self.last = Some(m);
            }
            Line::Saved { step } => {
                // The whole measurement again, marked, so that a watcher
                // keying by step replaces the row with one that is complete.
                if let Some(m) = self.last.as_mut().filter(|m| m.step == step as i64) {
                    m.saved = true;
                    let m = m.clone();
                    self.keep(&m);
                    self.send(Update::Measured(m));
                }
            }
            Line::Sampled { step, prompt, file } => {
                let (step, prompt) = (step as i64, prompt as i64);
                let _ = self.db.with(|c| {
                    c.execute(
                        "INSERT OR REPLACE INTO tune_samples (job, step, prompt, file) VALUES (?1, ?2, ?3, ?4)",
                        params![self.id, step, prompt, file.to_string_lossy()],
                    )
                });
                self.send(Update::Picture(Drawn { step, prompt, url: crate::jobs::sample_url(self.id, step, prompt) }));
            }
            Line::Done(outcome) => self.ended = Some(Ok(outcome)),
            Line::Failed { error } => self.ended = Some(Err(error)),
        }
    }

    /// The job's last state, its error and its result, from how the worker
    /// ended; `otherwise` is what is said of one that ended without a word.
    pub fn verdict(self, otherwise: String) -> (&'static str, Option<String>, Option<serde_json::Value>) {
        match self.ended {
            Some(Ok(o)) => {
                // A run stopped before any step wrote nothing.
                let wrote = o.best_step > 0;
                let result = serde_json::json!({
                    // What a request names to draw with it: the file, where
                    // there is one.
                    "handle": wrote.then(|| o.out.display().to_string()),
                    "last": o.last.as_ref().map(|p| p.display().to_string()),
                    // False for a run stopped before any step: nothing was
                    // written, and there is no best loss to report.
                    "measured": o.best_step > 0 && o.best_val.is_some(),
                    "base_val": o.base_val,
                    "best_val": o.best_val,
                    "best_step": o.best_step,
                    "last_val": o.last_val,
                    // Whether the LoRA beat the model without it, on the
                    // pictures it was measured on.
                    "improved": matches!((o.best_val, o.base_val), (Some(best), Some(base)) if best < base),
                    "layers": o.layers,
                    "params": o.trained,
                    "pictures": o.pictures,
                    "held_out": o.held_out,
                    "steps": o.steps,
                    "elapsed_secs": o.elapsed_secs,
                    "stopped": o.stopped,
                });
                (if o.stopped { "cancelled" } else { "done" }, None, Some(result))
            }
            Some(Err(error)) => ("failed", Some(error), None),
            None => ("failed", Some(otherwise), None),
        }
    }
}

/// Start a LoRA run in a worker, and answer with its job.
///
/// `hold` is the memory it was admitted on, kept until the worker is gone.
/// `work` is built by the caller, for the reason [`crate::jobs::train`]'s
/// options are: every part of working out what a request means is a
/// different refusal.
pub fn start(jobs: &Arc<Jobs>, hold: Reservation, mut work: Work, params: LoraParams, owner: Option<i64>) -> Res<crate::jobs::Job> {
    if jobs.training() {
        return Err("a training run is already going; wait for it or stop it".into());
    }
    let label = params.name.clone();
    let id = jobs.create("train", &label, &serde_json::to_value(Params::Lora(params))?, owner)?;
    let (cancel, events) = jobs.register(id);
    work.sample_dir = samples_dir(id);

    let worker = Arc::clone(jobs);
    let db = jobs.db.clone();
    std::thread::Builder::new().name(format!("kvad-tune-{id}")).spawn(move || {
        let mut follower = Follower::new(db, id, events);
        let otherwise = match run(&work, &cancel, &mut follower) {
            Ok(said) => said,
            Err(e) => format!("the training process could not be run: {e}"),
        };
        // Given back before the job is said to be over, so that whoever
        // hears it end and starts another finds the room.
        drop(hold);
        let (state, error, result) = follower.verdict(otherwise);
        worker.finish(id, state, error, result);
    })?;

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

/// Where a run's samples are written: under the data directory, by its job.
pub fn samples_dir(job: i64) -> PathBuf {
    kvad::weights::data_dir().join("tune-samples").join(job.to_string())
}

/// Run the worker to its end, feeding `follower` its lines. What comes back
/// is what to say of it if it ended without a [`Line::Done`] or a
/// [`Line::Failed`]: the last of what it wrote to its standard error, which
/// is where a run over its ceiling says so.
fn run(work: &Work, cancel: &Arc<AtomicBool>, follower: &mut Follower) -> Res<String> {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};

    let mut child = Command::new(std::env::current_exe()?)
        .arg(WORKER)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().ok_or("the worker has no input")?;
    writeln!(input, "{}", serde_json::to_string(work)?)?;
    input.flush()?;

    // Held open until the worker is gone: its closing is how the worker
    // knows the server is. `stop` goes down it when the job is cancelled.
    let over = Arc::new(AtomicBool::new(false));
    let asking = {
        let (cancel, over) = (Arc::clone(cancel), Arc::clone(&over));
        std::thread::spawn(move || {
            let mut asked = false;
            while !over.load(Ordering::Relaxed) {
                if !asked && cancel.load(Ordering::Relaxed) {
                    asked = writeln!(input, "stop").and_then(|()| input.flush()).is_ok();
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        })
    };
    let errors = child.stderr.take().map(|stderr| {
        std::thread::spawn(move || {
            let mut tail = std::collections::VecDeque::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                tracing::info!("tune worker: {line}");
                if tail.len() == 8 {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
            tail
        })
    });

    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            follower.take(&line);
        }
    }
    let status = child.wait()?;
    over.store(true, Ordering::Relaxed);
    let _ = asking.join();
    let tail = errors.and_then(|t| t.join().ok()).unwrap_or_default();
    Ok(match tail.is_empty() {
        true => format!("the training process ended without saying why ({status})"),
        false => format!("the training process ended ({status}): {}", tail.into_iter().collect::<Vec<_>>().join(" / ")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;

    fn following() -> (Arc<Jobs>, i64, Follower, broadcast::Receiver<Update>) {
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let id = jobs.create("train", "my-style", &serde_json::json!({ "loop": "lora" }), None).unwrap();
        let (events, heard) = broadcast::channel(64);
        let follower = Follower::new(jobs.db.clone(), id, events);
        (jobs, id, follower, heard)
    }

    fn line(l: &Line) -> String {
        serde_json::to_string(l).unwrap()
    }

    /// A run as its worker tells it, line by line, is the chart, the samples
    /// and the result somebody comes back to: the validation loss at each
    /// measurement, the first with no training loss because no step had been
    /// taken; the mean of the steps between for the others; and the step
    /// whose LoRA is on disk marked.
    #[test]
    fn a_runs_lines_become_its_chart_its_samples_and_its_result() {
        let (jobs, id, mut f, mut heard) = following();
        f.take(&line(&Line::Say { line: "loading the UNet".into() }));
        f.take(&line(&Line::Measured { step: 0, val_loss: 0.2, elapsed_secs: 30.0 }));
        f.take(&line(&Line::Sampled { step: 0, prompt: 0, file: "/data/tune-samples/1/1-00000.png".into() }));
        f.take(&line(&Line::Step { step: 1, steps: 2, loss: 0.3, secs: 2.0 }));
        f.take(&line(&Line::Step { step: 2, steps: 2, loss: 0.1, secs: 4.0 }));
        f.take(&line(&Line::Measured { step: 2, val_loss: 0.15, elapsed_secs: 40.0 }));
        f.take(&line(&Line::Saved { step: 2 }));
        // Something a library printed, and a blank line.
        f.take("compiled 3 kernels");
        f.take("");

        let measures = jobs.measures(id).unwrap();
        assert_eq!(measures.len(), 2);
        assert_eq!(measures[0], Measure { step: 0, val_loss: 0.2, train_loss: None, secs_per_step: None, elapsed_secs: 30.0, saved: false });
        assert_eq!(measures[1], Measure { step: 2, val_loss: 0.15, train_loss: Some(0.2), secs_per_step: Some(3.0), elapsed_secs: 40.0, saved: true });
        assert_eq!(jobs.pictures(id).unwrap(), [Drawn { step: 0, prompt: 0, url: format!("/api/jobs/{id}/samples/0/0") }]);
        assert_eq!(jobs.picture_file(id, 0, 0).unwrap().as_deref(), Some("/data/tune-samples/1/1-00000.png"));
        assert_eq!(jobs.picture_file(id, 0, 1).unwrap(), None);

        // A watcher hears each as it happens, and the saved step whole.
        let mut kinds = Vec::new();
        while let Ok(u) = heard.try_recv() {
            kinds.push(match u {
                Update::Status { .. } => "status",
                Update::Measured(m) if m.saved => {
                    assert_eq!(m.train_loss, Some(0.2), "the saved mark lost the rest of its row");
                    "saved"
                }
                Update::Measured(_) => "measured",
                Update::Picture(_) => "picture",
                Update::Stepped { .. } => "stepped",
                other => panic!("{other:?}"),
            });
        }
        assert_eq!(kinds, ["status", "measured", "picture", "stepped", "stepped", "measured", "saved", "status"]);

        // And a late one is told the same from the rows.
        let past = crate::jobs::history(&jobs, id).unwrap();
        assert_eq!(past.len(), 3);

        let outcome = Outcome {
            out: "/data/loras/my-style.safetensors".into(), last: None, layers: 560, trained: 23_000_000, pictures: 9, held_out: 1,
            base_val: Some(0.2), best_val: Some(0.15), best_step: 2, last_val: Some(0.15), steps: 2, elapsed_secs: 41.0, stopped: false,
        };
        f.take(&line(&Line::Done(outcome)));
        let (state, error, result) = f.verdict("unsaid".into());
        let result = result.unwrap();
        assert_eq!((state, error), ("done", None));
        assert_eq!(result["handle"], "/data/loras/my-style.safetensors");
        assert_eq!((result["measured"].as_bool(), result["improved"].as_bool(), result["best_step"].as_i64()), (Some(true), Some(true), Some(2)));
    }

    /// The three ways a run ends that are not a finished run: stopped, which
    /// keeps what it reached; failed, with the worker's own sentence; and
    /// gone without a word, which is what a run over its ceiling is, with
    /// what it wrote to its standard error for the reason.
    #[test]
    fn a_run_that_did_not_finish_says_how_it_ended() {
        let stopped = Outcome {
            out: "x".into(), last: None, layers: 1, trained: 1, pictures: 1, held_out: 0,
            base_val: Some(0.2), best_val: None, best_step: 0, last_val: Some(0.2), steps: 0, elapsed_secs: 1.0, stopped: true,
        };
        let (_, _, mut f, _heard) = following();
        f.take(&line(&Line::Done(stopped)));
        let (state, error, result) = f.verdict("unsaid".into());
        assert_eq!((state, error), ("cancelled", None));
        let result = result.unwrap();
        assert_eq!(result["measured"], false, "a run stopped before a step has no best loss");
        assert!(result["handle"].is_null(), "a run that wrote nothing named a file");

        let (_, _, mut f, _heard) = following();
        f.take(&line(&Line::Failed { error: "step 3: the loss is not a number".into() }));
        assert_eq!(f.verdict("unsaid".into()), ("failed", Some("step 3: the loss is not a number".into()), None));

        let (_, _, f, _heard) = following();
        let (state, error, _) = f.verdict("the training process ended (exit status: 137): stopped: the memory footprint reached 14.2 GB".into());
        assert!(state == "failed" && error.unwrap().contains("14.2 GB"));
    }

    /// What a run is charged follows what was measured: the run's own peak
    /// where that is the larger, the samples' where a small run draws them.
    #[test]
    fn a_run_is_charged_its_measured_peak_and_a_margin() {
        let gb = |b: u64| b as f64 / 1e9;
        assert!((gb(need(1024, false)) - 12.1).abs() < 0.05, "{}", gb(need(1024, false)));
        assert!((gb(need(512, false)) - 8.6).abs() < 0.1, "{}", gb(need(512, false)));
        // 8.7 GB of samples over a 512² run's 7.1.
        assert!((gb(need(512, true)) - 10.2).abs() < 0.05, "{}", gb(need(512, true)));
        assert_eq!(need(1024, true), need(1024, false), "samples are under a 1024² run's own peak");
    }
}
