//! The owner of the engines, one per model in memory.
//!
//! `kvad::service::Engine` holds one model and runs one thing at a time —
//! that is what the engine underneath it does — and speaks over a single
//! pair of channels with exactly one receiver. The server holds several
//! models, so it holds several engines, and an HTTP server has many requests
//! in flight that each need their own stream of tokens. Somebody has to own
//! those receivers, decide which engine a request is for, and decide whether
//! a model may be loaded at all. That is this.
//!
//! # Residents
//!
//! A model in memory is a *resident*, and it is keyed on the model *and* the
//! backend: the same weights at `cpu-q8` and at `gpu-q8` are two residents,
//! with two engines. A client names one as `repo@backend` — see [`id_of`] —
//! or as the bare repo, which means the one of that repo used most recently.
//!
//! Whether a load is allowed is [`crate::memory`]'s question. Nothing here
//! ever unloads a model to make room for another.
//!
//! # Why a queue and not a lock
//!
//! Two requests arriving at once have to wait for each other either way. A
//! queue makes the waiting visible: [`Scheduler::depth`] is a number the
//! dashboard can show, and the order is first-in-first-out rather than
//! whatever the mutex felt like.
//!
//! There is still one queue for every resident, and so one job at a time
//! across the whole server. Two residents on different hardware — one on
//! the CPU and one on the GPU — could decode at once, and that is the next
//! step; but two CPU residents decoding at once would fight over every core,
//! and how to share them is a measurement nobody has made yet. A load is a
//! job like any other, so a load never runs beside a decode.
//!
//! # What does *not* come through here
//!
//! Searching the Hub, listing what is on disk, deleting a model and pulling
//! one are not engine work. They go through `spawn_blocking` in the handlers
//! instead, so that a Hub search does not wait behind a thirty-second
//! generation. Only loading, unloading and generating are serialised.

use crate::memory::{Admission, Budget, Held};
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
    /// [`Resident::cached_tokens`].
    pub kv_bytes_per_token: usize,
    /// `chat` or `image`: which requests this model can answer.
    pub kind: Kind,
    /// For an image model, what a request that leaves a knob out gets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<kvad::image::Defaults>,
}

/// What a model is for. Language models chat; image pipelines paint; and a
/// request for the one sent to the other is refused before it is queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Chat,
    Image,
}

/// Which model, on which backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Key {
    pub repo: String,
    pub backend: Backend,
}

/// A model in memory, and what it costs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Resident {
    /// What a client names to reach this one: `repo@backend`.
    pub id: String,
    #[serde(flatten)]
    pub model: Loaded,
    /// What admission charged it; see [`crate::memory`].
    pub commit: u64,
    /// Whether it reads its experts from the disk through a cache, because
    /// it did not fit in what was left when it loaded.
    pub streams: bool,
    /// Tokens in its KV cache after its last generation.
    ///
    /// Read from the last `Stats` rather than from the session, because the
    /// session lives on the engine thread and asking it during a generation
    /// would mean a lock on the thing being measured. A number from the end
    /// of the last reply is the honest one to show.
    pub cached_tokens: usize,
    #[serde(skip)]
    pub key: Key,
    /// When it was last loaded or used, as a count of jobs. The bare repo
    /// name, and a page that names no model at all, mean the most recent.
    #[serde(skip)]
    used: u64,
}

/// `repo@backend`, the name a client uses for one resident.
pub fn id_of(repo: &str, backend: Backend) -> String {
    format!("{repo}@{}", crate::engine::id_of(backend))
}

/// A name a client sent, split into the repo and, if it said, the backend.
///
/// `@` cannot appear in a Hub repo id, so a name whose tail after one is
/// not a backend this build has is taken whole, as a repo nobody has.
pub fn parse_id(name: &str) -> (&str, Option<Backend>) {
    match name.rsplit_once('@') {
        Some((repo, backend)) => match crate::engine::parse(backend) {
            Some(b) => (repo, Some(b)),
            None => (name, None),
        },
        None => (name, None),
    }
}

/// Why a load did not happen.
#[derive(Debug, Clone)]
pub enum LoadError {
    /// It would not fit beside what is already resident.
    Full(String),
    /// It was tried, and failed.
    Failed(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Full(why) | LoadError::Failed(why) => f.write_str(why),
        }
    }
}

impl From<LoadError> for String {
    fn from(e: LoadError) -> String {
        e.to_string()
    }
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

/// A step of an image, or the image.
#[derive(Debug, Clone)]
pub enum Stroke {
    Step(kvad::image::Step),
    Done(Box<kvad::image::Painted>),
    Failed(String),
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
    Load {
        repo: String,
        backend: Backend,
        progress: tokio_mpsc::Sender<Progress>,
        done: oneshot::Sender<Result<Loaded, LoadError>>,
    },
    /// `None` unloads every resident.
    Unload { which: Option<Key>, done: Answer<Vec<String>> },
    Chat {
        on: Key,
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
        fresh: bool,
        out: tokio_mpsc::Sender<Piece>,
    },
    Complete {
        on: Key,
        prompt: String,
        sampling: Sampling,
        explain: usize,
        fresh: bool,
        out: tokio_mpsc::Sender<Piece>,
    },
    Tokenize { on: Key, text: String, done: Answer<Vec<Token>> },
    Score {
        on: Key,
        text: String,
        window: usize,
        progress: tokio_mpsc::Sender<(usize, usize)>,
        done: Answer<Perplexity>,
    },
    Paint { on: Key, request: kvad::image::ImageRequest, out: tokio_mpsc::Sender<Stroke> },
}

/// Makes the loader for each new engine. One per engine, because a
/// [`Loader`] is `FnMut` and moves to the thread that calls it.
type Loaders = Box<dyn Fn() -> Loader + Send>;

pub struct Scheduler {
    jobs: Sender<Job>,
    shared: Shared,
    budget: Budget,
}

/// What the scheduler thread shares with the handle.
#[derive(Clone, Default)]
struct Shared {
    /// Jobs submitted and not yet started. The running one is not counted;
    /// `busy` says whether there is one.
    waiting: Arc<AtomicUsize>,
    busy: Arc<AtomicBool>,
    /// In the order they were loaded.
    residents: Arc<Mutex<Vec<Resident>>>,
    /// The interrupt flag of whichever engine is generating.
    running: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

impl Shared {
    fn residents(&self) -> std::sync::MutexGuard<'_, Vec<Resident>> {
        self.residents.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn running(&self) -> std::sync::MutexGuard<'_, Option<Arc<AtomicBool>>> {
        self.running.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Scheduler {
    pub fn spawn(loaders: impl Fn() -> Loader + Send + 'static, budget: Budget) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let shared = Shared::default();
        let (theirs, loaders) = (shared.clone(), Box::new(loaders) as Loaders);
        std::thread::Builder::new()
            .name("kvad-scheduler".into())
            .spawn(move || run(rx, loaders, budget, theirs))
            .expect("failed to spawn the scheduler thread");
        Scheduler { jobs: tx, shared, budget }
    }

    /// How many requests are waiting for an engine, the running one
    /// included.
    pub fn depth(&self) -> usize {
        self.shared.waiting.load(Ordering::Relaxed)
            + usize::from(self.shared.busy.load(Ordering::Relaxed))
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Every resident, in the order they were loaded.
    pub fn residents(&self) -> Vec<Resident> {
        self.shared.residents().clone()
    }

    /// Bytes of the budget no resident has been charged.
    pub fn left(&self) -> u64 {
        let spent: u64 = self.shared.residents().iter().map(|r| r.commit).sum();
        self.budget.total.saturating_sub(spent)
    }

    /// The resident a name means: `repo@backend` exactly, or the bare repo's
    /// most recently used.
    pub fn find(&self, name: &str) -> Option<Resident> {
        let (repo, backend) = parse_id(name);
        self.residents()
            .into_iter()
            .filter(|r| r.key.repo.eq_ignore_ascii_case(repo))
            .filter(|r| backend.is_none_or(|b| r.key.backend == b))
            .max_by_key(|r| r.used)
    }

    /// The resident used most recently: what a page that names no model
    /// means.
    pub fn current(&self) -> Option<Resident> {
        self.residents().into_iter().max_by_key(|r| r.used)
    }

    /// [`Scheduler::current`]'s model, for the places that report one.
    pub fn loaded(&self) -> Option<Loaded> {
        self.current().map(|r| r.model)
    }

    /// Stop the generation that is running, if one is.
    ///
    /// Affects whatever an engine is doing *now*, not a particular request:
    /// with one job at a time across every resident those are the same
    /// thing, and when residents run at once this grows an argument.
    pub fn cancel(&self) {
        if let Some(flag) = self.shared.running().as_ref() {
            flag.store(true, Ordering::Relaxed);
        }
    }

    fn submit(&self, job: Job) -> Result<(), String> {
        self.shared.waiting.fetch_add(1, Ordering::Relaxed);
        self.jobs.send(job).map_err(|_| {
            self.shared.waiting.fetch_sub(1, Ordering::Relaxed);
            "the engine thread has stopped".to_string()
        })
    }

    /// Load a model beside the others, if it fits, reporting progress as it
    /// goes. A model already resident on that backend is answered at once.
    pub async fn load(
        &self,
        repo: String,
        backend: Backend,
        progress: tokio_mpsc::Sender<Progress>,
    ) -> Result<Loaded, LoadError> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Load { repo, backend, progress, done }).map_err(LoadError::Failed)?;
        wait.await.map_err(|_| LoadError::Failed("the engine stopped before it answered".into()))?
    }

    /// Drop one resident, or every one for `None`. The ids of what went,
    /// which is empty when nothing matched.
    pub async fn unload(&self, which: Option<Key>) -> Result<Vec<String>, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Unload { which, done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }

    /// Generate a reply from `on`, a fragment at a time.
    ///
    /// The receiver is returned immediately and the request may still be
    /// queued: the first [`Piece`] arrives when the engine reaches it. A
    /// caller that drops the receiver ends the generation, because the
    /// forwarding send then fails.
    ///
    /// `fresh` drops the KV cache first, as for [`Scheduler::complete`]; a
    /// conversation leaves it off so the next turn reuses this one's cache.
    pub fn chat(
        &self,
        on: &Key,
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
        fresh: bool,
    ) -> Result<tokio_mpsc::Receiver<Piece>, String> {
        // Bounded, so a client that reads slowly slows the generation down
        // rather than filling memory with tokens it has not asked for. 64 is
        // a second or so of decoding at the rates this engine reaches.
        let (out, rx) = tokio_mpsc::channel(64);
        self.submit(Job::Chat { on: on.clone(), messages, tools, sampling, fresh, out })?;
        Ok(rx)
    }

    /// Continue a prompt, with no chat template involved.
    ///
    /// `explain` is how many candidates to report per token, and `fresh`
    /// drops the KV cache first — see [`kvad::service::Cmd::Complete`] for why
    /// that is a measurement question rather than an output one.
    pub fn complete(
        &self,
        on: &Key,
        prompt: String,
        sampling: Sampling,
        explain: usize,
        fresh: bool,
    ) -> Result<tokio_mpsc::Receiver<Piece>, String> {
        let (out, rx) = tokio_mpsc::channel(64);
        self.submit(Job::Complete { on: on.clone(), prompt, sampling, explain, fresh, out })?;
        Ok(rx)
    }

    /// How a resident's tokenizer splits a text.
    ///
    /// Engine work, although it is only a tokenizer: the tokenizer belongs to
    /// the loaded model and the loaded model lives on that thread. So this
    /// queues behind a generation, which is a millisecond of work waiting on
    /// thirty seconds of somebody else's — and the alternative is a second
    /// copy of the tokenizer that can disagree with the first.
    pub async fn tokenize(&self, on: &Key, text: String) -> Result<Vec<Token>, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Tokenize { on: on.clone(), text, done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }

    /// Make an image on `on`, a step at a time.
    ///
    /// As [`Scheduler::chat`]: the receiver comes back at once, the first
    /// [`Stroke`] when the engine reaches the job, and dropping the receiver
    /// stops the generation at the next step.
    pub fn paint(
        &self,
        on: &Key,
        request: kvad::image::ImageRequest,
    ) -> Result<tokio_mpsc::Receiver<Stroke>, String> {
        // A step is seconds, so a handful of them buffered is plenty.
        let (out, rx) = tokio_mpsc::channel(8);
        self.submit(Job::Paint { on: on.clone(), request, out })?;
        Ok(rx)
    }

    /// Score a text the model did not write.
    pub async fn score(
        &self,
        on: &Key,
        text: String,
        window: usize,
        progress: tokio_mpsc::Sender<(usize, usize)>,
    ) -> Result<Perplexity, String> {
        let (done, wait) = oneshot::channel();
        self.submit(Job::Score { on: on.clone(), text, window, progress, done })?;
        wait.await.map_err(|_| "the engine stopped before it answered".to_string())?
    }
}

/// One engine and the model it holds.
struct Slot {
    key: Key,
    engine: Engine,
}

/// The scheduler thread: one job at a time, in the order they arrived.
fn run(jobs: Receiver<Job>, loaders: Loaders, budget: Budget, shared: Shared) {
    let mut slots: Vec<Slot> = Vec::new();
    let mut clock: u64 = 0;

    // Mark `key` as just used, and say what its cache held afterwards.
    let touch = |key: &Key, clock: u64, cached: Option<usize>| {
        if let Some(r) = shared.residents().iter_mut().find(|r| &r.key == key) {
            r.used = clock;
            if let Some(c) = cached {
                r.cached_tokens = c;
            }
        }
    };
    let missing = |key: &Key| format!("{} is not loaded", id_of(&key.repo, key.backend));

    while let Ok(job) = jobs.recv() {
        shared.waiting.fetch_sub(1, Ordering::Relaxed);
        shared.busy.store(true, Ordering::Relaxed);
        clock += 1;

        match job {
            Job::Load { repo, backend, progress, done } => {
                let key = Key { repo, backend };
                let result = match slots.iter().any(|s| s.key == key) {
                    // Already here: the one swap that costs nothing.
                    true => {
                        touch(&key, clock, None);
                        let found = shared.residents().iter().find(|r| r.key == key).cloned();
                        found.map(|r| r.model).ok_or_else(|| LoadError::Failed(missing(&key)))
                    }
                    false => load(&key, &loaders, budget, &shared, &progress).map(|(slot, r)| {
                        slots.push(slot);
                        let model = r.model.clone();
                        shared.residents().push(Resident { used: clock, ..r });
                        model
                    }),
                };
                let _ = done.send(result);
            }

            Job::Unload { which, done } => {
                let (going, staying): (Vec<Slot>, Vec<Slot>) =
                    slots.drain(..).partition(|s| which.as_ref().is_none_or(|k| &s.key == k));
                slots = staying;
                let mut gone = Vec::new();
                let mut failed = None;
                for slot in going {
                    slot.engine.send(Cmd::Unload);
                    // Waited for rather than left to the engine's thread, so
                    // that the memory is back before the next job asks what
                    // is left.
                    if let Err(e) = drain_unload(&slot.engine.rx) {
                        failed = Some(e);
                    }
                    shared.residents().retain(|r| r.key != slot.key);
                    gone.push(id_of(&slot.key.repo, slot.key.backend));
                }
                let _ = done.send(failed.map_or(Ok(gone), Err));
            }

            Job::Chat { on, messages, tools, sampling, fresh, out } => {
                match slots.iter().find(|s| s.key == on) {
                    Some(slot) => {
                        *shared.running() = Some(Arc::clone(&slot.engine.cancel));
                        slot.engine.send(Cmd::Chat { messages, tools, sampling, fresh });
                        // What the cache holds now: the prompt it prefilled
                        // plus everything it generated.
                        let cached = drain_chat(&slot.engine.rx, &out)
                            .map(|stats| stats.prompt_tokens + stats.generated_tokens);
                        *shared.running() = None;
                        touch(&on, clock, cached);
                    }
                    None => {
                        let _ = out.blocking_send(Piece::Failed(missing(&on)));
                    }
                }
            }

            Job::Complete { on, prompt, sampling, explain, fresh, out } => {
                match slots.iter().find(|s| s.key == on) {
                    Some(slot) => {
                        *shared.running() = Some(Arc::clone(&slot.engine.cancel));
                        slot.engine.send(Cmd::Complete { prompt, sampling, explain, fresh });
                        let cached = drain_chat(&slot.engine.rx, &out)
                            .map(|stats| stats.prompt_tokens + stats.generated_tokens);
                        *shared.running() = None;
                        touch(&on, clock, cached);
                    }
                    None => {
                        let _ = out.blocking_send(Piece::Failed(missing(&on)));
                    }
                }
            }

            Job::Tokenize { on, text, done } => {
                let answer = match slots.iter().find(|s| s.key == on) {
                    Some(slot) => {
                        slot.engine.send(Cmd::Tokenize(text));
                        drain_tokens(&slot.engine.rx)
                    }
                    None => Err(missing(&on)),
                };
                let _ = done.send(answer);
            }

            Job::Score { on, text, window, progress, done } => {
                let answer = match slots.iter().find(|s| s.key == on) {
                    Some(slot) => {
                        slot.engine.send(Cmd::Perplexity { text, window });
                        let scored = drain_score(&slot.engine.rx, &progress);
                        // Scoring ends by clearing the cache, so nothing is
                        // held.
                        touch(&on, clock, Some(0));
                        scored
                    }
                    None => Err(missing(&on)),
                };
                let _ = done.send(answer);
            }

            Job::Paint { on, request, out } => match slots.iter().find(|s| s.key == on) {
                Some(slot) => {
                    *shared.running() = Some(Arc::clone(&slot.engine.cancel));
                    slot.engine.send(Cmd::Paint(request));
                    drain_paint(&slot.engine.rx, &out, &slot.engine.cancel);
                    *shared.running() = None;
                    touch(&on, clock, None);
                }
                None => {
                    let _ = out.blocking_send(Stroke::Failed(missing(&on)));
                }
            },
        }

        shared.busy.store(false, Ordering::Relaxed);
    }
}

/// Admit `key`, and if it is admitted, give it an engine and load it there.
fn load(
    key: &Key,
    loaders: &Loaders,
    budget: Budget,
    shared: &Shared,
    progress: &tokio_mpsc::Sender<Progress>,
) -> Result<(Slot, Resident), LoadError> {
    let id = id_of(&key.repo, key.backend);
    let held: Vec<Held> =
        shared.residents().iter().map(|r| Held { id: r.id.clone(), commit: r.commit }).collect();
    let need = crate::memory::need(&key.repo, key.backend, budget.context);
    let admission = crate::memory::admit(&budget, &held, &id, &need);
    let room = match admission {
        Admission::Refused(why) => return Err(LoadError::Full(why)),
        Admission::Streams { room } => room,
        Admission::Fits { .. } => budget.total.saturating_sub(held.iter().map(|h| h.commit).sum()),
    };

    // The expert cache is decided many calls below the loader, so what is
    // left is said to it this way rather than passed; see `set_room`. Said
    // for every load and not only a streaming one, because a mixture whose
    // estimate came in low would otherwise size its cache against the whole
    // machine rather than against what the other residents left of it.
    kvad::experts::set_room(Some(room));
    let engine = Engine::spawn(loaders());
    engine.send(Cmd::Load { repo: key.repo.clone(), backend: key.backend });
    let result = drain_load(&engine.rx, progress);
    kvad::experts::set_room(None);
    let model = result.map_err(LoadError::Failed)?;

    // Charged what it turned out to be, now that it is loaded, rather than
    // what the files suggested. A model over the budget alone is charged
    // the budget, which is all there is to charge.
    let kv = match need.kv {
        0 => (model.kv_bytes_per_token * model.n_ctx.min(budget.context)) as u64,
        kv => kv,
    };
    let (commit, streams) = match admission {
        Admission::Streams { room } => (room, true),
        _ => ((model.weight_bytes as u64 + kv).min(budget.total), false),
    };
    let resident =
        Resident { id, model, commit, streams, cached_tokens: 0, key: key.clone(), used: 0 };
    Ok((Slot { key: key.clone(), engine }, resident))
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
                image,
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
                    kind: if image.is_some() { Kind::Image } else { Kind::Chat },
                    image,
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

/// Forward an image's steps, then the image.
///
/// A client that has gone away is noticed here rather than by the engine: the
/// engine only learns of it through the cancel flag, which is set on the
/// first step nobody received. Draining carries on to the engine's answer
/// either way, or the next job would read this one's leftovers.
fn drain_paint(rx: &Receiver<Evt>, out: &tokio_mpsc::Sender<Stroke>, cancel: &AtomicBool) {
    loop {
        let stroke = match rx.recv() {
            Ok(Evt::Painting(step)) => Stroke::Step(step),
            Ok(Evt::Painted(p)) => Stroke::Done(Box::new(p)),
            Ok(Evt::Error(e)) => Stroke::Failed(e),
            Ok(Evt::Status(message)) => Stroke::Failed(message),
            Ok(_) => continue,
            Err(_) => Stroke::Failed("the engine thread has stopped".into()),
        };
        let last = !matches!(stroke, Stroke::Step(_));
        if out.blocking_send(stroke).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
        if last {
            return;
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

    const Q8: Backend = Backend::Cpu(Precision::Q8);
    const F32: Backend = Backend::Cpu(Precision::F32);

    fn roomy() -> Budget {
        Budget { total: 1 << 40, context: 1024 }
    }

    /// A scheduler with an engine that cannot load anything: enough to test
    /// the queue.
    fn refusing() -> Scheduler {
        Scheduler::spawn(|| Box::new(|_, _, _, _| Err("no backend in tests".into())), roomy())
    }

    fn nowhere() -> tokio_mpsc::Sender<Progress> {
        let (progress, mut seen) = tokio_mpsc::channel(64);
        tokio::spawn(async move { while seen.recv().await.is_some() {} });
        progress
    }

    #[tokio::test]
    async fn a_failed_load_comes_back_as_an_error_and_leaves_nothing_loaded() {
        let sched = refusing();
        let (progress, mut seen) = tokio_mpsc::channel(16);
        let err = sched.load("nobody/nothing".into(), Q8, progress).await.unwrap_err();
        assert!(matches!(err, LoadError::Failed(_)), "{err:?}");
        assert_eq!(err.to_string(), "no backend in tests");
        assert!(sched.residents().is_empty());

        // The status line the engine emits before trying is reported, so a
        // load that fails slowly is not a silent one.
        let first = seen.recv().await.expect("no progress at all");
        assert!(matches!(first, Progress::Status { .. }), "{first:?}");

        // And the queue is empty again afterwards.
        assert_eq!(sched.depth(), 0);
    }

    /// Unloading nothing is not an error; it is nothing.
    #[tokio::test]
    async fn unloading_an_empty_engine_says_nothing_was_loaded() {
        let sched = refusing();
        assert!(sched.unload(None).await.unwrap().is_empty());
    }

    /// A chat with a model that is not loaded ends with a refusal naming it,
    /// rather than hanging.
    #[tokio::test]
    async fn chatting_with_no_such_resident_fails_the_stream() {
        let sched = refusing();
        let on = Key { repo: "a/b".into(), backend: Q8 };
        let mut pieces = sched
            .chat(&on, vec![Message::user("hello")], Vec::new(), Sampling::default(), false)
            .expect("the scheduler refused to queue the job");
        let piece = pieces.recv().await.expect("the stream ended with nothing in it");
        match piece {
            Piece::Failed(why) => assert_eq!(why, "a/b@cpu-q8 is not loaded"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(pieces.recv().await.is_none(), "the stream continued past its end");
        assert_eq!(sched.depth(), 0);
    }

    /// Jobs run in the order they were submitted, and the depth says how many
    /// are outstanding.
    #[tokio::test]
    async fn work_queues_rather_than_overlapping() {
        let sched = refusing();
        let mut waits = Vec::new();
        for _ in 0..4 {
            waits.push(sched.load("x/y".into(), Q8, nowhere()));
        }
        // Nothing has been awaited yet, so every one of them is outstanding.
        for w in waits {
            assert!(w.await.is_err());
        }
        assert_eq!(sched.depth(), 0, "the queue did not drain");
    }

    #[test]
    fn a_name_is_a_repo_and_perhaps_a_backend() {
        assert_eq!(id_of("Qwen/Qwen3-0.6B", Q8), "Qwen/Qwen3-0.6B@cpu-q8");
        assert_eq!(parse_id("Qwen/Qwen3-0.6B@cpu-q8"), ("Qwen/Qwen3-0.6B", Some(Q8)));
        assert_eq!(parse_id("Qwen/Qwen3-0.6B"), ("Qwen/Qwen3-0.6B", None));
        // Not a backend, so not split: the whole of it is what was asked for.
        assert_eq!(parse_id("a/b@nonsense"), ("a/b@nonsense", None));
    }

    /// Two residents at once, each answering for itself; the bare name means
    /// the one used last; one can go without the other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_models_are_resident_at_once() {
        let dir = crate::compare::tests::tiny_model("residents");
        let repo = dir.to_string_lossy().into_owned();
        let sched = Scheduler::spawn(kvad::service::cpu_loader, roomy());

        sched.load(repo.clone(), F32, nowhere()).await.unwrap();
        sched.load(repo.clone(), Q8, nowhere()).await.unwrap();
        let ids: Vec<String> = sched.residents().into_iter().map(|r| r.id).collect();
        assert_eq!(ids, [id_of(&repo, F32), id_of(&repo, Q8)]);
        assert!(sched.residents().iter().all(|r| r.commit > 0), "a resident was charged nothing");

        // Loading one that is already here is free, and changes nothing but
        // which is the most recent.
        assert_eq!(sched.find(&repo).unwrap().key.backend, Q8);
        sched.load(repo.clone(), F32, nowhere()).await.unwrap();
        assert_eq!(sched.residents().len(), 2);
        assert_eq!(sched.find(&repo).unwrap().key.backend, F32);

        // Each generates, on its own engine.
        for backend in [F32, Q8] {
            let on = Key { repo: repo.clone(), backend };
            let sampling = Sampling { temperature: 0.0, max_tokens: 4, ..Sampling::default() };
            let mut pieces = sched.complete(&on, "the cat".into(), sampling, 0, true).unwrap();
            let mut done = None;
            while let Some(piece) = pieces.recv().await {
                match piece {
                    Piece::Done(stats) => done = Some(stats),
                    Piece::Failed(why) => panic!("{backend}: {why}"),
                    _ => {}
                }
            }
            assert_eq!(done.expect("no end to the generation").generated_tokens, 4);
        }
        // The last to generate is the one a page naming nothing gets.
        assert_eq!(sched.current().unwrap().key.backend, Q8);
        assert!(sched.find(&id_of(&repo, F32)).unwrap().cached_tokens > 0);

        let gone = sched.unload(Some(Key { repo: repo.clone(), backend: F32 })).await.unwrap();
        assert_eq!(gone, [id_of(&repo, F32)]);
        assert_eq!(sched.residents().iter().map(|r| r.id.clone()).collect::<Vec<_>>(), [id_of(&repo, Q8)]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A budget that holds one model and not two refuses the second, says
    /// why, and leaves the first where it was.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_model_that_does_not_fit_beside_another_is_refused() {
        let dir = crate::compare::tests::tiny_model("admission");
        let repo = dir.to_string_lossy().into_owned();

        // Learn what one costs, then allow a budget of that and a byte.
        let probe = Scheduler::spawn(kvad::service::cpu_loader, roomy());
        probe.load(repo.clone(), F32, nowhere()).await.unwrap();
        let one = probe.residents()[0].commit;
        drop(probe);

        let sched =
            Scheduler::spawn(kvad::service::cpu_loader, Budget { total: one + 1, context: 1024 });
        sched.load(repo.clone(), F32, nowhere()).await.unwrap();
        let again = sched.load(repo.clone(), F32, nowhere()).await;
        assert!(again.is_ok(), "the same resident again must be free, not refused: {again:?}");

        let err = sched.load(repo.clone(), Q8, nowhere()).await.unwrap_err();
        let LoadError::Full(why) = &err else { panic!("expected a refusal, got {err:?}") };
        assert!(why.contains(&id_of(&repo, F32)), "the refusal does not say what is resident: {why}");
        assert_eq!(sched.residents().len(), 1, "a refused load changed what was resident");

        // Once the first is gone, the second fits.
        sched.unload(None).await.unwrap();
        sched.load(repo.clone(), Q8, nowhere()).await.unwrap();
        assert_eq!(sched.residents().iter().map(|r| r.key.backend).collect::<Vec<_>>(), [Q8]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
