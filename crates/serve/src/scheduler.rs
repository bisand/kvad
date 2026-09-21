//! The one owner of the engine.
//!
//! `kvad::service::Engine` already runs one thing at a time — that is what
//! the engine underneath it does — but it speaks over a single pair of
//! channels with exactly one receiver. An HTTP server has many requests in
//! flight and each needs its own stream of tokens, so somebody has to own
//! that receiver and hand the events to whoever asked for them. That is all
//! this is.
//!
//! # Why a queue and not a lock
//!
//! Two requests arriving at once have to wait for each other either way. A
//! queue makes the waiting visible: [`Scheduler::depth`] is a number the
//! dashboard can show, and the order is first-in-first-out rather than
//! whatever the mutex felt like. When continuous batching lands it replaces
//! the inside of [`run`] and the API above it does not change.
//!
//! # What does *not* come through here
//!
//! Searching the Hub, listing what is on disk, deleting a model and pulling
//! one are not engine work. They go through `spawn_blocking` in the handlers
//! instead, so that a Hub search does not wait behind a thirty-second
//! generation. Only loading, unloading and generating are serialised.

use kvad::chat::Message;
use kvad::runtime::{Chosen, Perplexity, Stats, Token};
use kvad::service::{Backend, Cmd, Engine, Evt, Loader, Sampling};
use kvad::weights::Fetch;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

/// What a load produced, and what `/v1/models` and the dashboard report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Loaded {
    pub repo: String,
    pub summary: String,
    pub params: usize,
    pub instruct: bool,
    /// Whether this model can be offered tools — a fact about its chat
    /// template, reported so that a client finds out before it asks for a
    /// call rather than after it gets a paragraph instead of one.
    pub tools: bool,
    pub backend: String,
    pub weight_bytes: usize,
    /// The longest conversation this model can hold.
    pub n_ctx: usize,
    /// What one token of context costs in the KV cache. The cache grows by
    /// this much per token and is not pre-allocated, so the interesting
    /// number is this times the tokens actually held — see
    /// [`Scheduler::last_cached`].
    pub kv_bytes_per_token: usize,
}

/// A step of a load, as it happens.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Progress {
    /// A line of text for a status bar.
    Status { message: String },
    /// Bytes, for a bar.
    Download { file: String, bytes: u64, total: u64 },
    /// The file is on disk, downloaded or already cached.
    Fetched { file: String },
}

/// A fragment of a reply, or the end of one.
#[derive(Debug, Clone)]
pub enum Piece {
    Token(String),
    /// The same fragment, with the candidates behind it, for a request that
    /// asked to see the choice.
    Chose(Chosen),
    Done(Stats),
    Failed(String),
}

type Answer<T> = oneshot::Sender<Result<T, String>>;

enum Job {
    Load { repo: String, backend: Backend, progress: tokio_mpsc::Sender<Progress>, done: Answer<Loaded> },
    Unload { done: Answer<Option<String>> },
    Chat {
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
        out: tokio_mpsc::Sender<Piece>,
    },
    Complete {
        prompt: String,
        sampling: Sampling,
        explain: usize,
        fresh: bool,
        out: tokio_mpsc::Sender<Piece>,
    },
    Tokenize { text: String, done: Answer<Vec<Token>> },
    Score { text: String, window: usize, progress: tokio_mpsc::Sender<(usize, usize)>, done: Answer<Perplexity> },
}

pub struct Scheduler {
    jobs: Sender<Job>,
    /// Jobs submitted and not yet started. The running one is not counted;
    /// `busy` says whether there is one.
    waiting: Arc<AtomicUsize>,
    busy: Arc<AtomicBool>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    /// Tokens in the KV cache after the last generation.
    ///
    /// Read from the last `Stats` rather than from the session, because the
    /// session lives on the engine thread and asking it during a generation
    /// would mean a lock on the thing being measured. A number from the end
    /// of the last reply is the honest one to show.
    cached: Arc<AtomicUsize>,
    /// The engine's interrupt flag, raised to stop a generation mid-stream.
    cancel: Arc<AtomicBool>,
}

impl Scheduler {
    pub fn spawn(loader: Loader) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let waiting = Arc::new(AtomicUsize::new(0));
        let busy = Arc::new(AtomicBool::new(false));
        let loaded: Arc<Mutex<Option<Loaded>>> = Arc::new(Mutex::new(None));

        let engine = Engine::spawn(loader);
        let cancel = Arc::clone(&engine.cancel);

        let cached = Arc::new(AtomicUsize::new(0));
        let (w, b, l, c) =
            (Arc::clone(&waiting), Arc::clone(&busy), Arc::clone(&loaded), Arc::clone(&cached));
        std::thread::Builder::new()
            .name("kvad-scheduler".into())
            .spawn(move || run(rx, engine, w, b, l, c))
            .expect("failed to spawn the scheduler thread");

        Scheduler { jobs: tx, waiting, busy, loaded, cached, cancel }
    }

    /// Tokens the KV cache held after the last generation, or 0.
    pub fn last_cached(&self) -> usize {
        self.cached.load(Ordering::Relaxed)
    }

    /// How many requests are waiting for the engine, the running one
    /// included.
    pub fn depth(&self) -> usize {
        self.waiting.load(Ordering::Relaxed) + usize::from(self.busy.load(Ordering::Relaxed))
    }

    pub fn loaded(&self) -> Option<Loaded> {
        self.loaded.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Stop the generation that is running, if one is.
    ///
    /// Affects whatever the engine is doing *now*, not a particular request:
    /// with one generation at a time those are the same thing, and when they
    /// stop being the same thing this grows an argument.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn submit(&self, job: Job) -> Result<(), String> {
        self.waiting.fetch_add(1, Ordering::Relaxed);
        self.jobs.send(job).map_err(|_| {
            self.waiting.fetch_sub(1, Ordering::Relaxed);
            "the engine thread has stopped".to_string()
        })
    }

    /// Load a model, reporting progress as it goes.
    pub async fn load(
        &self,
        repo: String,
        backend: Backend,
        progress: tokio_mpsc::Sender<Progress>,
    ) -> Result<Loaded, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Load { repo, backend, progress, done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }

    /// Drop the loaded model. The name of what went, or `None` for nothing.
    pub async fn unload(&self) -> Result<Option<String>, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Unload { done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }

    /// Generate a reply, a fragment at a time.
    ///
    /// The receiver is returned immediately and the request may still be
    /// queued: the first [`Piece`] arrives when the engine reaches it. A
    /// caller that drops the receiver ends the generation, because the
    /// forwarding send then fails.
    pub fn chat(
        &self,
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
    ) -> Result<tokio_mpsc::Receiver<Piece>, String> {
        // Bounded, so a client that reads slowly slows the generation down
        // rather than filling memory with tokens it has not asked for. 64 is
        // a second or so of decoding at the rates this engine reaches.
        let (out, rx) = tokio_mpsc::channel(64);
        self.submit(Job::Chat { messages, tools, sampling, out })?;
        Ok(rx)
    }

    /// Continue a prompt, with no chat template involved.
    ///
    /// `explain` is how many candidates to report per token, and `fresh`
    /// drops the KV cache first — see [`kvad::service::Cmd::Complete`] for why
    /// that is a measurement question rather than an output one.
    pub fn complete(
        &self,
        prompt: String,
        sampling: Sampling,
        explain: usize,
        fresh: bool,
    ) -> Result<tokio_mpsc::Receiver<Piece>, String> {
        let (out, rx) = tokio_mpsc::channel(64);
        self.submit(Job::Complete { prompt, sampling, explain, fresh, out })?;
        Ok(rx)
    }

    /// How the loaded model's tokenizer splits a text.
    ///
    /// Engine work, although it is only a tokenizer: the tokenizer belongs to
    /// the loaded model and the loaded model lives on that thread. So this
    /// queues behind a generation, which is a millisecond of work waiting on
    /// thirty seconds of somebody else's — and the alternative is a second
    /// copy of the tokenizer that can disagree with the first.
    pub async fn tokenize(&self, text: String) -> Result<Vec<Token>, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Tokenize { text, done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }

    /// Score a text the model did not write.
    pub async fn score(
        &self,
        text: String,
        window: usize,
        progress: tokio_mpsc::Sender<(usize, usize)>,
    ) -> Result<Perplexity, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Score { text, window, progress, done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }
}

/// The scheduler thread: one job at a time, in the order they arrived.
fn run(
    jobs: Receiver<Job>,
    engine: Engine,
    waiting: Arc<AtomicUsize>,
    busy: Arc<AtomicBool>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    cached: Arc<AtomicUsize>,
) {
    let set = |to: Option<Loaded>| *loaded.lock().unwrap_or_else(|e| e.into_inner()) = to;

    while let Ok(job) = jobs.recv() {
        waiting.fetch_sub(1, Ordering::Relaxed);
        busy.store(true, Ordering::Relaxed);

        match job {
            Job::Load { repo, backend, progress, done } => {
                engine.send(Cmd::Load { repo, backend });
                let result = drain_load(&engine.rx, &progress);
                if let Ok(model) = &result {
                    set(Some(model.clone()));
                    // A fresh session holds nothing.
                    cached.store(0, Ordering::Relaxed);
                }
                let _ = done.send(result);
            }

            Job::Unload { done } => {
                engine.send(Cmd::Unload);
                let was = drain_unload(&engine.rx);
                if matches!(was, Ok(Some(_))) {
                    set(None);
                    cached.store(0, Ordering::Relaxed);
                }
                let _ = done.send(was);
            }

            Job::Chat { messages, tools, sampling, out } => {
                engine.send(Cmd::Chat { messages, tools, sampling });
                if let Some(stats) = drain_chat(&engine.rx, &out) {
                    // What the cache holds now: the prompt it prefilled plus
                    // everything it generated.
                    cached.store(stats.prompt_tokens + stats.generated_tokens, Ordering::Relaxed);
                }
            }

            Job::Complete { prompt, sampling, explain, fresh, out } => {
                engine.send(Cmd::Complete { prompt, sampling, explain, fresh });
                if let Some(stats) = drain_chat(&engine.rx, &out) {
                    cached.store(stats.prompt_tokens + stats.generated_tokens, Ordering::Relaxed);
                }
            }

            Job::Tokenize { text, done } => {
                engine.send(Cmd::Tokenize(text));
                let _ = done.send(drain_tokens(&engine.rx));
            }

            Job::Score { text, window, progress, done } => {
                engine.send(Cmd::Perplexity { text, window });
                let _ = done.send(drain_score(&engine.rx, &progress));
                // Scoring ends by clearing the cache, so nothing is held.
                cached.store(0, Ordering::Relaxed);
            }
        }

        busy.store(false, Ordering::Relaxed);
    }
}

/// Read events until the load has an answer.
///
/// Every drain loop here skips events it did not ask for. The engine sends
/// `Local(..)` of its own accord — at startup, and after anything that
/// changes what is on disk — and a loop that treated an unexpected event as
/// an error would fail a load because the cache listing arrived first.
fn drain_load(rx: &Receiver<Evt>, progress: &tokio_mpsc::Sender<Progress>) -> Result<Loaded, String> {
    loop {
        match rx.recv() {
            Ok(Evt::Loaded {
                repo,
                summary,
                params,
                instruct,
                tools,
                backend,
                weight_bytes,
                n_ctx,
                kv_bytes_per_token,
            }) => {
                return Ok(Loaded {
                    repo,
                    summary,
                    params,
                    instruct,
                    tools,
                    backend,
                    weight_bytes,
                    n_ctx,
                    kv_bytes_per_token,
                })
            }
            Ok(Evt::Error(e)) => return Err(e),
            Ok(Evt::Status(message)) => report(progress, Progress::Status { message }),
            Ok(Evt::Fetching(f)) => {
                if let Some(p) = fetch_progress(f) {
                    report(progress, p);
                }
            }
            Ok(_) => {}
            Err(_) => return Err("the engine thread has stopped".into()),
        }
    }
}

fn drain_unload(rx: &Receiver<Evt>) -> Result<Option<String>, String> {
    loop {
        match rx.recv() {
            Ok(Evt::Unloaded(repo)) => return Ok(Some(repo)),
            Ok(Evt::Error(e)) => return Err(e),
            // What the engine says when there was nothing loaded. The command
            // has no failure of its own, so "nothing happened" is the answer.
            Ok(Evt::Status(_)) => return Ok(None),
            Ok(_) => {}
            Err(_) => return Err("the engine thread has stopped".into()),
        }
    }
}

fn drain_tokens(rx: &Receiver<Evt>) -> Result<Vec<Token>, String> {
    loop {
        match rx.recv() {
            Ok(Evt::Tokens(tokens)) => return Ok(tokens),
            Ok(Evt::Error(e)) => return Err(e),
            // What the engine says when nothing is loaded.
            Ok(Evt::Status(message)) => return Err(message),
            Ok(_) => {}
            Err(_) => return Err("the engine thread has stopped".into()),
        }
    }
}

fn drain_score(
    rx: &Receiver<Evt>,
    progress: &tokio_mpsc::Sender<(usize, usize)>,
) -> Result<Perplexity, String> {
    loop {
        match rx.recv() {
            Ok(Evt::Scored(p)) => return Ok(p),
            Ok(Evt::Scoring { done, total }) => {
                let _ = progress.try_send((done, total));
            }
            Ok(Evt::Error(e)) => return Err(e),
            Ok(Evt::Status(message)) => return Err(message),
            Ok(_) => {}
            Err(_) => return Err("the engine thread has stopped".into()),
        }
    }
}

/// Forward a generation's tokens, and return the statistics it ended with.
fn drain_chat(rx: &Receiver<Evt>, out: &tokio_mpsc::Sender<Piece>) -> Option<Stats> {
    loop {
        let piece = match rx.recv() {
            Ok(Evt::Token(text)) => Piece::Token(text),
            Ok(Evt::Chose(chosen)) => Piece::Chose(chosen),
            Ok(Evt::Done(stats)) => Piece::Done(stats),
            Ok(Evt::Error(e)) => Piece::Failed(e),
            // Sent when no model is loaded, which for a chat is a refusal
            // rather than a status line.
            Ok(Evt::Status(message)) => Piece::Failed(message),
            Ok(_) => continue,
            Err(_) => Piece::Failed("the engine thread has stopped".into()),
        };
        let ended = match &piece {
            Piece::Token(_) | Piece::Chose(_) => None,
            Piece::Done(stats) => Some(Some(*stats)),
            Piece::Failed(_) => Some(None),
        };
        // A closed receiver is a client that has gone away. Stop forwarding,
        // but keep draining until the engine finishes this generation, or the
        // next request would read this one's leftovers.
        let _ = out.blocking_send(piece);
        if let Some(stats) = ended {
            return stats;
        }
    }
}

fn fetch_progress(f: Fetch) -> Option<Progress> {
    match f {
        // A file answered from the cache reports no size; there is no bar to
        // draw and the status line above already said what is happening.
        Fetch::Download { total: 0, .. } | Fetch::Local | Fetch::Shards(_) => None,
        Fetch::Download { file, bytes, total } => Some(Progress::Download { file, bytes, total }),
        Fetch::Fetched { file } => Some(Progress::Fetched { file }),
    }
}

/// Progress is advisory: a client that stopped reading should not stop the
/// load it started, and a full channel means it is reading more slowly than
/// the bytes arrive.
fn report(progress: &tokio_mpsc::Sender<Progress>, p: Progress) {
    let _ = progress.try_send(p);
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::quant::Precision;

    /// A scheduler with an engine that cannot load anything: enough to test
    /// the queue, which is what this module is.
    fn refusing() -> Scheduler {
        Scheduler::spawn(Box::new(|_, _, _, _| Err("no backend in tests".into())))
    }

    #[tokio::test]
    async fn a_failed_load_comes_back_as_an_error_and_leaves_nothing_loaded() {
        let sched = refusing();
        let (progress, mut seen) = tokio_mpsc::channel(16);
        let err = sched
            .load("nobody/nothing".into(), Backend::Cpu(Precision::Q8), progress)
            .await
            .unwrap_err();
        assert_eq!(err, "no backend in tests");
        assert!(sched.loaded().is_none());

        // The status line the engine emits before trying is reported, so a
        // load that fails slowly is not a silent one.
        let first = seen.recv().await.expect("no progress at all");
        assert!(matches!(first, Progress::Status { .. }), "{first:?}");

        // And the queue is empty again afterwards.
        assert_eq!(sched.depth(), 0);
    }

    /// Unloading nothing is not an error; it is `None`.
    #[tokio::test]
    async fn unloading_an_empty_engine_says_nothing_was_loaded() {
        let sched = refusing();
        assert_eq!(sched.unload().await.unwrap(), None);
    }

    /// A chat with no model loaded ends with a refusal rather than hanging.
    #[tokio::test]
    async fn chatting_without_a_model_fails_the_stream() {
        let sched = refusing();
        let mut pieces = sched
            .chat(vec![Message::user("hello")], Vec::new(), Sampling::default())
            .expect("the scheduler refused to queue the job");
        let piece = pieces.recv().await.expect("the stream ended with nothing in it");
        match piece {
            Piece::Failed(why) => assert!(why.contains("no model loaded"), "{why}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(pieces.recv().await.is_none(), "the stream continued past its end");
    }

    /// Jobs run in the order they were submitted, and the depth says how many
    /// are outstanding.
    #[tokio::test]
    async fn work_queues_rather_than_overlapping() {
        let sched = refusing();
        let mut waits = Vec::new();
        for _ in 0..4 {
            let (progress, _seen) = tokio_mpsc::channel(1);
            waits.push(sched.load("x/y".into(), Backend::Cpu(Precision::Q8), progress));
        }
        // Nothing has been awaited yet, so every one of them is outstanding.
        for w in waits {
            assert!(w.await.is_err());
        }
        assert_eq!(sched.depth(), 0, "the queue did not drain");
    }
}
