//! Work that outlives the request that asked for it.
//!
//! A download takes minutes and a training run can take an hour. Both are
//! longer than an HTTP request should live and longer than a browser tab is
//! likely to stay open, so the work belongs to the server and a request only
//! watches it. Close the tab, come back tomorrow, and the run is either still
//! going or in the history with its loss curve intact.
//!
//! # Why not the scheduler
//!
//! [`crate::scheduler`] exists because the *engine* can do one thing at a
//! time and somebody has to own its channel. These jobs are not engine work —
//! a download is network and disk, and training runs on `nervus`'s own
//! threads — so putting them behind that queue would make a download wait for
//! a generation for no reason. What they need instead is a place to live and
//! a way to be watched, which is this.
//!
//! # Watching
//!
//! Each live job has a `broadcast` channel. A watcher subscribes, then reads
//! whatever is already in the database, then follows the live tail; a step
//! can arrive between those two, so the same step may be seen twice and the
//! client keys metrics by step number. That is cheaper than a lock held
//! across a database read.
//!
//! # One run at a time
//!
//! Two training runs would fight for the same cores and each take twice as
//! long, so the second is refused rather than queued — an hour of waiting
//! with no way to see why is worse than being told now. Downloads have no
//! such limit; they are waiting on a network.

use crate::db::Db;
use crate::scheduler::Progress;
use rusqlite::{params, OptionalExtension};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Job {
    pub id: i64,
    pub kind: String,
    pub state: String,
    pub label: String,
    pub params: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
}

impl Job {
    pub fn live(&self) -> bool {
        matches!(self.state.as_str(), "queued" | "running")
    }
}

/// One checkpoint of a training run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Metric {
    pub step: i64,
    pub train_loss: f64,
    pub val_loss: f64,
    pub chars_per_sec: f64,
    pub elapsed_secs: f64,
    pub saved: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Sample {
    pub step: i64,
    pub text: String,
}

/// One case of a prompt suite, run against one variant.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Case {
    pub variant: String,
    pub idx: i64,
    pub prompt: String,
    pub expect: String,
    pub got: String,
    pub passed: bool,
    pub decode_per_sec: Option<f64>,
    pub generated_tokens: Option<i64>,
}

/// What one variant scored on a held-out text.
///
/// Read back as well as written: a perplexity run keeps its scores in the
/// job's result rather than in rows of its own, because one number per
/// variant is the whole of it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Scored {
    pub variant: String,
    pub tokens: usize,
    pub scored: usize,
    pub perplexity: f64,
    pub bits_per_token: f64,
    pub took_secs: f64,
}

/// One timed generation in a benchmark.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Timing {
    pub variant: String,
    pub round: i64,
    pub decode_per_sec: f64,
    pub prefill_per_sec: f64,
    pub ttft_millis: f64,
    pub generated_tokens: i64,
    /// What it wrote, for a side-by-side comparison. Absent when the run is a
    /// measurement rather than a comparison.
    pub text: Option<String>,
}

/// Something that happened, as a watcher sees it.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Update {
    Status { message: String },
    Download { file: String, bytes: u64, total: u64 },
    Metric(Metric),
    Sample(Sample),
    /// Work done out of work to do, in whatever units the job counts in.
    Progress { done: usize, total: usize },
    Case(Case),
    Scored(Scored),
    Timing(Timing),
    /// How fast this machine turned out to be, sent once and early, while
    /// there is still time to do something about the answer.
    Pace { chars_per_sec: f64, remaining_secs: f64 },
    /// Terminal. The job row has the rest.
    Ended { state: String, error: Option<String> },
}

struct Live {
    cancel: Arc<AtomicBool>,
    events: broadcast::Sender<Update>,
}

pub struct Jobs {
    pub db: Db,
    live: Mutex<HashMap<i64, Live>>,
}

impl Jobs {
    pub fn new(db: Db) -> Self {
        Jobs { db, live: Mutex::new(HashMap::new()) }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, HashMap<i64, Live>> {
        self.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Mark every job that was running when the server stopped.
    ///
    /// Nothing is resumed: a training run's optimiser state was in memory and
    /// a half-written download is `hf-hub`'s to sort out. What matters is
    /// that a row does not claim to be running when nothing is.
    pub fn abandon_orphans(&self) -> Res<usize> {
        self.db.with(|c| {
            c.execute(
                "UPDATE jobs SET state = 'failed', ended_at = datetime('now'),
                        error = 'the server stopped while this was running'
                 WHERE state IN ('queued', 'running')",
                [],
            )
        })
    }

    pub fn list(&self, limit: usize) -> Res<Vec<Job>> {
        self.db.with(|c| {
            let mut q = c.prepare(
                "SELECT id, kind, state, label, params, result, error,
                        created_at, started_at, ended_at
                 FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?1",
            )?;
            let rows = q.query_map([limit as i64], row)?.collect();
            rows
        })
    }

    pub fn get(&self, id: i64) -> Res<Option<Job>> {
        self.db.with(|c| {
            c.query_row(
                "SELECT id, kind, state, label, params, result, error,
                        created_at, started_at, ended_at
                 FROM jobs WHERE id = ?1",
                [id],
                row,
            )
            .optional()
        })
    }

    pub fn metrics(&self, id: i64) -> Res<Vec<Metric>> {
        self.db.with(|c| {
            let mut q = c.prepare(
                "SELECT step, train_loss, val_loss, chars_per_sec, elapsed_secs, saved
                 FROM train_metrics WHERE job = ?1 ORDER BY step",
            )?;
            let rows = q
                .query_map([id], |r| {
                    Ok(Metric {
                        step: r.get(0)?,
                        train_loss: r.get(1)?,
                        val_loss: r.get(2)?,
                        chars_per_sec: r.get(3)?,
                        elapsed_secs: r.get(4)?,
                        saved: r.get::<_, i64>(5)? != 0,
                    })
                })?
                .collect();
            rows
        })
    }

    pub fn samples(&self, id: i64) -> Res<Vec<Sample>> {
        self.db.with(|c| {
            let mut q =
                c.prepare("SELECT step, text FROM train_samples WHERE job = ?1 ORDER BY step")?;
            let rows =
                q.query_map([id], |r| Ok(Sample { step: r.get(0)?, text: r.get(1)? }))?.collect();
            rows
        })
    }

    /// The verdicts an eval run has reached so far.
    pub fn cases(&self, id: i64) -> Res<Vec<Case>> {
        self.db.with(|c| {
            let mut q = c.prepare(
                "SELECT variant, idx, prompt, expect, got, passed, decode_per_sec,
                        generated_tokens
                 FROM eval_results WHERE job = ?1 ORDER BY variant, idx",
            )?;
            let rows = q
                .query_map([id], |r| {
                    Ok(Case {
                        variant: r.get(0)?,
                        idx: r.get(1)?,
                        prompt: r.get(2)?,
                        expect: r.get(3)?,
                        got: r.get(4)?,
                        passed: r.get::<_, i64>(5)? != 0,
                        decode_per_sec: r.get(6)?,
                        generated_tokens: r.get(7)?,
                    })
                })?
                .collect();
            rows
        })
    }

    /// The samples a benchmark has taken so far.
    pub fn timings(&self, id: i64) -> Res<Vec<Timing>> {
        self.db.with(|c| {
            let mut q = c.prepare(
                "SELECT variant, round, decode_per_sec, prefill_per_sec, ttft_millis,
                        generated_tokens, text
                 FROM bench_samples WHERE job = ?1 ORDER BY round, variant",
            )?;
            let rows = q
                .query_map([id], |r| {
                    Ok(Timing {
                        variant: r.get(0)?,
                        round: r.get(1)?,
                        decode_per_sec: r.get(2)?,
                        prefill_per_sec: r.get(3)?,
                        ttft_millis: r.get(4)?,
                        generated_tokens: r.get(5)?,
                        text: r.get(6)?,
                    })
                })?
                .collect();
            rows
        })
    }

    /// Is a training run going? The one thing that is not allowed twice.
    pub fn training(&self) -> bool {
        self.live_kind(&["train"]).is_some()
    }

    /// A live job of one of these kinds, named, or `None`.
    ///
    /// Used to refuse the second of two things that would fight over the same
    /// hardware: two training runs, or a benchmark started while anything
    /// else is on the machine. The answer names what is in the way, because
    /// "busy" without a subject is a message nobody can act on.
    pub fn live_kind(&self, kinds: &[&str]) -> Option<String> {
        let list: Vec<String> = kinds.iter().map(|k| format!("'{k}'")).collect();
        self.db
            .with(|c| {
                c.query_row(
                    &format!(
                        "SELECT kind, label FROM jobs
                         WHERE kind IN ({}) AND state IN ('queued','running')
                         ORDER BY id LIMIT 1",
                        list.join(", ")
                    ),
                    [],
                    |r| Ok(format!("{} ({})", r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                )
                .optional()
            })
            .ok()
            .flatten()
    }

    pub fn create(&self, kind: &str, label: &str, params: &serde_json::Value, owner: Option<i64>) -> Res<i64> {
        self.db.with(|c| {
            c.execute(
                "INSERT INTO jobs (kind, state, label, params, owner)
                 VALUES (?1, 'running', ?2, ?3, ?4)",
                params![kind, label, params.to_string(), owner],
            )?;
            let id = c.last_insert_rowid();
            c.execute("UPDATE jobs SET started_at = datetime('now') WHERE id = ?1", [id])?;
            Ok(id)
        })
    }

    pub fn finish(&self, id: i64, state: &str, error: Option<String>, result: Option<serde_json::Value>) {
        let _ = self.db.with(|c| {
            c.execute(
                "UPDATE jobs SET state = ?2, error = ?3, result = ?4, ended_at = datetime('now')
                 WHERE id = ?1",
                params![id, state, error, result.map(|r| r.to_string())],
            )
        });
        let held = self.held().remove(&id);
        if let Some(live) = held {
            // A watcher that has already gone is not an error; there may be
            // nobody left to tell, and the row says it all anyway.
            let _ = live.events.send(Update::Ended { state: state.to_string(), error });
        }
    }

    /// Follow a job. `None` for one that is already over — read the rows.
    pub fn watch(&self, id: i64) -> Option<broadcast::Receiver<Update>> {
        self.held().get(&id).map(|l| l.events.subscribe())
    }

    /// Ask a job to stop. True if there was one to ask.
    ///
    /// Advisory: a download finishes the file it is on, and a training run
    /// stops at its next step. Neither is instant and neither is a kill.
    pub fn cancel(&self, id: i64) -> bool {
        match self.held().get(&id) {
            Some(live) => {
                live.cancel.store(true, Ordering::Relaxed);
                true
            }
            false_positive => false_positive.is_some(),
        }
    }

    pub fn register(&self, id: i64) -> (Arc<AtomicBool>, broadcast::Sender<Update>) {
        // 256: a training run sends a handful of updates a minute and a
        // download a few a second. A watcher that falls this far behind is
        // one that has stopped reading, and `broadcast` drops the oldest for
        // it rather than blocking the work.
        let (events, _) = broadcast::channel(256);
        let cancel = Arc::new(AtomicBool::new(false));
        self.held()
            .insert(id, Live { cancel: Arc::clone(&cancel), events: events.clone() });
        (cancel, events)
    }
}

pub fn row(r: &rusqlite::Row) -> rusqlite::Result<Job> {
    let json = |s: Option<String>| s.and_then(|s| serde_json::from_str(&s).ok());
    Ok(Job {
        id: r.get("id")?,
        kind: r.get("kind")?,
        state: r.get("state")?,
        label: r.get("label")?,
        params: json(r.get("params")?).unwrap_or(serde_json::Value::Null),
        result: json(r.get("result")?),
        error: r.get("error")?,
        created_at: r.get("created_at")?,
        started_at: r.get("started_at")?,
        ended_at: r.get("ended_at")?,
    })
}

// ---------------------------------------------------------------------------
// The kinds of job
// ---------------------------------------------------------------------------

/// Download a model in the background.
pub fn pull(jobs: &Arc<Jobs>, repo: String, owner: Option<i64>) -> Res<Job> {
    let id = jobs.create("pull", &repo, &serde_json::json!({ "repo": repo }), owner)?;
    let (cancel, events) = jobs.register(id);
    let worker = Arc::clone(jobs);

    std::thread::Builder::new().name(format!("kvad-pull-{id}")).spawn(move || {
        let say = events.clone();
        let watch = {
            let events = events.clone();
            kvad::weights::Watcher::new(move |f: kvad::weights::Fetch| {
                if let Some(u) = from_fetch(f) {
                    let _ = events.send(u);
                }
            })
        };
        let mut progress = |message: &str| {
            let _ = say.send(Update::Status { message: message.to_string() });
        };
        let outcome = kvad::weights::pull_watched(&repo, &mut progress, &watch);

        // `hf-hub` has no way to be interrupted, so a cancelled download is
        // one that is noticed as finished rather than stopped. Saying so is
        // better than a button that quietly does nothing.
        match (outcome, cancel.load(Ordering::Relaxed)) {
            (Ok(_), true) => worker.finish(id, "cancelled", None, None),
            (Ok(_), false) => {
                worker.finish(id, "done", None, Some(serde_json::json!({ "repo": repo })))
            }
            (Err(e), _) => worker.finish(id, "failed", Some(e.to_string()), None),
        }
    })?;

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

/// Read a website into a dataset, in the background.
///
/// Network and disk, like a pull, so several may run at once — but not two
/// onto the same name, which would be two crawls writing one file.
pub fn crawl(
    jobs: &Arc<Jobs>,
    request: kvad::crawl::Request,
    owner: Option<i64>,
) -> Res<Job> {
    let taken = jobs.db.with(|c| {
        c.query_row(
            "SELECT 1 FROM jobs
             WHERE kind = 'crawl' AND state IN ('queued','running') AND label = ?1 LIMIT 1",
            [&request.name],
            |_| Ok(()),
        )
        .optional()
    })?;
    if taken.is_some() {
        return Err(format!("`{}` is already being read; wait for it or stop it", request.name).into());
    }

    let id = jobs.create("crawl", &request.name, &serde_json::to_value(&request)?, owner)?;
    let (cancel, events) = jobs.register(id);
    let worker = Arc::clone(jobs);
    let db = jobs.db.clone();

    std::thread::Builder::new().name(format!("kvad-crawl-{id}")).spawn(move || {
        let mut say = |note: kvad::crawl::Note| match note {
            kvad::crawl::Note::Say(message) => {
                let _ = events.send(Update::Status { message });
            }
            kvad::crawl::Note::Page { url, title, done, total } => {
                // The bar and the line under it: how far along, and what it
                // is reading, which is the half anybody actually watches.
                let _ = events.send(Update::Progress { done, total });
                let name = match title.is_empty() {
                    true => url,
                    false => title,
                };
                let _ = events.send(Update::Status { message: name });
            }
        };

        let outcome = kvad::crawl::run(&request, &mut say, &cancel)
            .and_then(|crawled| keep(&db, &request, &crawled, owner));

        // A stopped crawl keeps the pages it read, the same bargain a stopped
        // training run makes with its best checkpoint. There is no reason to
        // throw away four hundred pages because somebody wanted the four
        // hundredth to be the last.
        match (outcome, cancel.load(Ordering::Relaxed)) {
            (Ok(result), stopped) => {
                worker.finish(id, if stopped { "cancelled" } else { "done" }, None, Some(result))
            }
            (Err(e), _) => worker.finish(id, "failed", Some(e.to_string()), None),
        }
    })?;

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

/// Write what a crawl came back with: the text, and the manifest beside it.
///
/// The manifest is written second and its failure is not the crawl's: a
/// corpus with no record of where it came from is worse than one with, and
/// much better than no corpus at all after twenty minutes of fetching.
fn keep(
    db: &Db,
    request: &kvad::crawl::Request,
    crawled: &kvad::crawl::Crawled,
    owner: Option<i64>,
) -> Res<serde_json::Value> {
    let dataset =
        crate::datasets::save(db, &request.name, &crawled.text, owner, Some(&request.url))?;

    let mut wrote = None;
    if let Ok(path) = crate::datasets::manifest_of(&request.name) {
        match serde_json::to_vec_pretty(&crawled.manifest)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
        {
            Ok(()) => wrote = Some(path.display().to_string()),
            Err(e) => tracing::warn!("could not write the crawl manifest: {e}"),
        }
    }

    let manifest = &crawled.manifest;
    Ok(serde_json::json!({
        "dataset": dataset.name,
        "id": dataset.id,
        "start": manifest.start,
        "scope": manifest.scope,
        "pages": manifest.pages.len(),
        "skipped": manifest.skipped.len(),
        "stopped": manifest.stopped,
        "characters": manifest.characters,
        "distinct": manifest.distinct,
        "mapped": manifest.mapped.iter().map(|m| m.count).sum::<usize>(),
        "dropped": manifest.dropped.len(),
        "alphabet": manifest.alphabet.iter().take(200).collect::<Vec<_>>(),
        "manifest": wrote,
    }))
}

/// What a training run was asked for. Stored as the job's `params`, so a run
/// can be read back and repeated.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainParams {
    pub name: Option<String>,
    pub from: Option<String>,
    pub dataset: i64,
    pub dataset_name: String,
    pub size: String,
    pub steps: usize,
    pub lr: f32,
    pub eval_every: usize,
    pub threads: usize,
    pub sample: usize,
    pub seed: u64,
}

/// Train a model in the background.
///
/// `opts` is built by the caller, because working out what a request means —
/// which dataset, which size, whether the model it continues from exists — is
/// the handler's job and every one of those answers is a different refusal.
pub fn train(
    jobs: &Arc<Jobs>,
    mut opts: kvad::train::Options,
    params: TrainParams,
    owner: Option<i64>,
) -> Res<Job> {
    if jobs.training() {
        return Err("a training run is already going; wait for it or stop it".into());
    }
    let label = params
        .name
        .clone()
        .or_else(|| params.from.clone())
        .unwrap_or_else(|| "training".into());
    let id = jobs.create("train", &label, &serde_json::to_value(&params)?, owner)?;
    let (cancel, events) = jobs.register(id);
    // The flag Phase 0 put on `Training::stop`, read once a step.
    opts.cancel = Some(Arc::clone(&cancel));

    let worker = Arc::clone(jobs);
    let db = jobs.db.clone();
    std::thread::Builder::new()
        .name(format!("kvad-train-{id}"))
        // `nervus` recurses through the model twice per step and holds the
        // graph on the stack; 8 MB for the same reason the engine thread has
        // it.
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut out = |line: &str| {
                let _ = events.send(Update::Status { message: line.to_string() });
            };
            // Every checkpoint is written down as it happens rather than at
            // the end, so a run watched in a tab yesterday is a chart today
            // and a run that dies halfway still has its curve.
            let mut watch = |e: kvad::train::Event| {
                let update = match e {
                    kvad::train::Event::Pace { chars_per_sec, remaining_secs } => Update::Pace {
                        chars_per_sec: chars_per_sec as f64,
                        remaining_secs: remaining_secs as f64,
                    },
                    kvad::train::Event::Step {
                        step, train_loss, val_loss, chars_per_sec, elapsed_secs, ..
                    } => {
                        let metric = Metric {
                            step: step as i64,
                            train_loss: train_loss as f64,
                            val_loss: val_loss as f64,
                            chars_per_sec: chars_per_sec as f64,
                            elapsed_secs: elapsed_secs as f64,
                            saved: false,
                        };
                        let _ = db.with(|c| {
                            c.execute(
                                "INSERT OR REPLACE INTO train_metrics
                                 (job, step, train_loss, val_loss, chars_per_sec, elapsed_secs)
                                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                                params![
                                    id,
                                    metric.step,
                                    metric.train_loss,
                                    metric.val_loss,
                                    metric.chars_per_sec,
                                    metric.elapsed_secs
                                ],
                            )
                        });
                        Update::Metric(metric)
                    }
                    kvad::train::Event::Saved { step, val_loss } => {
                        // Marks the step whose model is the one on disk, which
                        // the chart draws as the point that matters.
                        let _ = db.with(|c| {
                            c.execute(
                                "UPDATE train_metrics SET saved = 1 WHERE job = ?1 AND step = ?2",
                                params![id, step as i64],
                            )
                        });
                        Update::Metric(Metric {
                            step: step as i64,
                            train_loss: f64::NAN,
                            val_loss: val_loss as f64,
                            chars_per_sec: 0.0,
                            elapsed_secs: 0.0,
                            saved: true,
                        })
                    }
                    kvad::train::Event::Sample { step, text } => {
                        let _ = db.with(|c| {
                            c.execute(
                                "INSERT OR REPLACE INTO train_samples (job, step, text)
                                 VALUES (?1, ?2, ?3)",
                                params![id, step as i64, text],
                            )
                        });
                        Update::Sample(Sample { step: step as i64, text })
                    }
                };
                let _ = events.send(update);
            };

            match kvad::train::run_watched(&opts, &mut out, &mut watch) {
                Ok(summary) => {
                    // A run stopped before its first checkpoint has no best
                    // loss: the bar starts at infinity and nothing beat it.
                    // `serde_json` cannot encode infinity and would quietly
                    // write `null` anyway, so say `null` on purpose — and
                    // `measured` says which kind of null this is.
                    let finite = |v: f32| v.is_finite().then_some(v);
                    let result = serde_json::json!({
                        "handle": summary.handle,
                        "dir": summary.dir.display().to_string(),
                        "params": summary.params,
                        "measured": summary.best_val.is_finite(),
                        "best_val": finite(summary.best_val),
                        "best_step": summary.best_step,
                        // False for a continuation no checkpoint improved on:
                        // `best_val` is then the model that was already there
                        // and `reached` is the best this run managed.
                        "improved": summary.improved(),
                        "reached": finite(summary.reached),
                        "last_val": finite(summary.last_val),
                        "elapsed_secs": summary.elapsed_secs,
                        "stopped": summary.stopped,
                    });
                    // A stopped run is not a failed one: it kept the best
                    // model it reached, and that model is usable.
                    let state = if summary.stopped { "cancelled" } else { "done" };
                    worker.finish(id, state, None, Some(result));
                }
                Err(e) => worker.finish(id, "failed", Some(e.to_string()), None),
            }
        })?;

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

fn from_fetch(f: kvad::weights::Fetch) -> Option<Update> {
    match f {
        kvad::weights::Fetch::Download { total: 0, .. }
        | kvad::weights::Fetch::Local
        | kvad::weights::Fetch::Shards(_) => None,
        kvad::weights::Fetch::Download { file, bytes, total } => {
            Some(Update::Download { file, bytes, total })
        }
        kvad::weights::Fetch::Fetched { file } => {
            Some(Update::Status { message: format!("fetched {file}") })
        }
    }
}

/// The same events a watcher would see, out of the database.
///
/// What a browser arriving late is sent before it starts following the live
/// tail, and the whole of what a finished run has to show.
pub fn history(jobs: &Jobs, id: i64) -> Res<Vec<Update>> {
    let mut updates: Vec<Update> = jobs.metrics(id)?.into_iter().map(Update::Metric).collect();
    updates.extend(jobs.samples(id)?.into_iter().map(Update::Sample));
    updates.extend(jobs.cases(id)?.into_iter().map(Update::Case));
    updates.extend(jobs.timings(id)?.into_iter().map(Update::Timing));
    Ok(updates)
}

/// Turn `Progress` from the engine into an update, so that a load and a pull
/// look the same to a watcher.
impl From<Progress> for Update {
    fn from(p: Progress) -> Update {
        match p {
            Progress::Status { message } => Update::Status { message },
            Progress::Download { file, bytes, total } => Update::Download { file, bytes, total },
            Progress::Fetched { file } => Update::Status { message: format!("fetched {file}") },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jobs() -> Arc<Jobs> {
        Arc::new(Jobs::new(Db::in_memory().unwrap()))
    }

    #[test]
    fn a_job_is_a_row_before_it_is_anything_else() {
        let jobs = jobs();
        let id = jobs.create("pull", "a/b", &serde_json::json!({ "repo": "a/b" }), None).unwrap();
        let job = jobs.get(id).unwrap().unwrap();
        assert_eq!((job.kind.as_str(), job.state.as_str(), job.label.as_str()), ("pull", "running", "a/b"));
        assert!(job.live());
        assert!(job.started_at.is_some() && job.ended_at.is_none());
        assert_eq!(job.params["repo"], "a/b");

        jobs.finish(id, "done", None, Some(serde_json::json!({ "repo": "a/b" })));
        let job = jobs.get(id).unwrap().unwrap();
        assert!(!job.live());
        assert_eq!(job.state, "done");
        assert!(job.ended_at.is_some());
        assert_eq!(job.result.unwrap()["repo"], "a/b");
    }

    /// Two training runs would fight for the same cores, so the second is
    /// refused rather than queued behind an hour of silence.
    #[test]
    fn only_one_training_run_at_a_time() {
        let jobs = jobs();
        assert!(!jobs.training());
        let id = jobs.create("train", "x", &serde_json::json!({}), None).unwrap();
        assert!(jobs.training());

        // A download alongside is fine: it is waiting on a network.
        jobs.create("pull", "a/b", &serde_json::json!({}), None).unwrap();
        assert!(jobs.training());

        jobs.finish(id, "done", None, None);
        assert!(!jobs.training());
    }

    /// A row that says "running" when nothing is would be a run somebody
    /// waits for forever.
    #[test]
    fn a_restart_ends_the_jobs_it_interrupted() {
        let jobs = jobs();
        let running = jobs.create("train", "x", &serde_json::json!({}), None).unwrap();
        let done = jobs.create("pull", "a/b", &serde_json::json!({}), None).unwrap();
        jobs.finish(done, "done", None, None);

        assert_eq!(jobs.abandon_orphans().unwrap(), 1);
        let job = jobs.get(running).unwrap().unwrap();
        assert_eq!(job.state, "failed");
        assert!(job.error.unwrap().contains("server stopped"));
        // And the one that had finished is left alone.
        assert_eq!(jobs.get(done).unwrap().unwrap().state, "done");
    }

    /// Cancelling is advisory and only a live job has anything to ask.
    #[test]
    fn only_a_live_job_can_be_asked_to_stop() {
        let jobs = jobs();
        let id = jobs.create("train", "x", &serde_json::json!({}), None).unwrap();
        assert!(!jobs.cancel(id), "a job with no thread behind it had something to ask");

        let (cancel, _events) = jobs.register(id);
        assert!(!cancel.load(Ordering::Relaxed));
        assert!(jobs.cancel(id));
        assert!(cancel.load(Ordering::Relaxed), "the flag the run reads was not raised");

        // Finishing takes the registration with it.
        jobs.finish(id, "cancelled", None, None);
        assert!(!jobs.cancel(id));
        assert!(jobs.watch(id).is_none());
    }

    /// The history a late watcher is handed is the same shape as the live
    /// tail, so the client has one thing to parse.
    #[test]
    fn what_was_written_down_reads_back_as_the_events_that_wrote_it() {
        let jobs = jobs();
        let id = jobs.create("train", "x", &serde_json::json!({}), None).unwrap();
        jobs.db
            .with(|c| {
                c.execute(
                    "INSERT INTO train_metrics
                     (job, step, train_loss, val_loss, chars_per_sec, elapsed_secs, saved)
                     VALUES (?1, 10, 2.5, 2.4, 1000.0, 1.5, 1)",
                    [id],
                )?;
                c.execute("INSERT INTO train_samples (job, step, text) VALUES (?1, 10, 'abc')", [id])
            })
            .unwrap();

        let past = history(&jobs, id).unwrap();
        assert_eq!(past.len(), 2);
        match &past[0] {
            Update::Metric(m) => {
                assert_eq!(m.step, 10);
                assert_eq!(m.val_loss, 2.4);
                assert!(m.saved, "the step whose model is on disk lost its mark");
            }
            other => panic!("expected a metric, got {other:?}"),
        }
        match &past[1] {
            Update::Sample(s) => assert_eq!(s.text, "abc"),
            other => panic!("expected a sample, got {other:?}"),
        }
    }

    /// Deleting a job takes its chart with it, which needs foreign keys on.
    #[test]
    fn deleting_a_job_takes_its_metrics() {
        let jobs = jobs();
        let id = jobs.create("train", "x", &serde_json::json!({}), None).unwrap();
        jobs.db
            .with(|c| {
                c.execute(
                    "INSERT INTO train_metrics
                     (job, step, train_loss, val_loss, chars_per_sec, elapsed_secs)
                     VALUES (?1, 1, 1.0, 1.0, 1.0, 1.0)",
                    [id],
                )
            })
            .unwrap();
        jobs.db.with(|c| c.execute("DELETE FROM jobs WHERE id = ?1", [id])).unwrap();
        assert!(jobs.metrics(id).unwrap().is_empty(), "the metrics outlived their job");
    }
}
