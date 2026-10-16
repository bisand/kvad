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
//! queue makes the waiting visible: [`Scheduler::queue`] says whose work is
//! waiting and for how long, and the order is first-in-first-out rather
//! than whatever the mutex felt like.
//!
//! # Whose turn it is
//!
//! First-in-first-out is fair to requests, which is not the same as fair to
//! people: somebody who sends ten gets ten times the machine. So the next
//! generation to run is the earliest one of whichever account was served
//! longest ago — round-robin over accounts, see [`pick`] — and an account
//! may have only [`Scheduler::share`] of them waiting. With no accounts
//! there is one person and the order is arrival order, as it was.
//!
//! Loads and unloads are not reordered, and nothing passes one: a chat
//! queued behind an unload runs after it, whoever asked.
//!
//! The queue has an end. A request for a generation that arrives when
//! [`Scheduler::limit`] of them are already waiting is refused there and
//! then — see [`Refused::Full`] — because an answer now that says "later" is
//! worth more to a client than a connection that sits for four minutes and
//! then times out somewhere in between.
//!
//! There is still one queue for every resident, and one resident at work at
//! a time. Two residents on different hardware — one on the CPU and one on
//! the GPU — could decode at once, and that is a later step; but two CPU
//! residents decoding at once would fight over every core, and how to share
//! them is a measurement nobody has made yet. A load is a job like any
//! other, so a load never runs beside a decode.
//!
//! # Several replies at once
//!
//! On one resident, though, replies run together: up to
//! [`Scheduler::batch`] of them, each a token further every step, and a
//! reply that ends gives its place to the next in line without the others
//! waiting for it. That is the *continuous* in continuous batching. Whose
//! turn it is to join is the same question as whose turn it is to run, and
//! has the same answer — [`pick`] — with one addition: if the job whose
//! turn it is cannot join, because it is for another model or is not a
//! reply at all, nobody joins and the batch runs down, so that a queue of
//! chats cannot keep a load or another model's request waiting for ever.
//!
//! A request that asked for no cache is a measurement, and runs alone.
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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
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
    /// `chat`, `image`, `video` or `speech`: which requests this model can
    /// answer.
    pub kind: Kind,
    /// For an image model, what a request that leaves a knob out gets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<kvad::image::Defaults>,
    /// For a video model, the same, and its limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<kvad::video::Defaults>,
}

/// What a model is for. Language models chat, image pipelines paint, video
/// pipelines film, Whispers listen; and a request for one sent to another is
/// refused before it is queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Chat,
    Image,
    Video,
    Speech,
}

impl Kind {
    /// The kind of a model on disk, from the pipeline it is, if it is one.
    pub fn of_pipeline(pipeline: Option<&str>) -> Kind {
        match pipeline {
            Some(kvad::speech::PIPELINE) => Kind::Speech,
            Some(p) if kvad::video::is_video_pipeline(p) => Kind::Video,
            Some(_) => Kind::Image,
            None => Kind::Chat,
        }
    }
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
    /// Tokens in its KV cache after its last generation, across every
    /// conversation it is keeping.
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
    /// A model across several machines, and where one of them has got to
    /// with its layers. They load at once, so these come interleaved; this
    /// machine is one of them. `state` is `asked`, `loading`, `ready` or
    /// `failed`: `loading` carries a line in `message` or a download in
    /// `file`, `bytes` and `total`; `ready` the memory its layers took, in
    /// `bytes`; `failed` why, in `message`.
    Machine {
        machine: String,
        layers: [usize; 2],
        state: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file: Option<String>,
        bytes: u64,
        total: u64,
    },
}

/// A step of an image, or the image.
#[derive(Debug, Clone)]
pub enum Stroke {
    Step(kvad::image::Step),
    Done(Box<kvad::image::Painted>),
    Failed(String),
    /// The request asked for what the model cannot do, found only once it
    /// tried: a LoRA that does not fit it.
    Refused(String),
}

/// Where a transcription has got to, or the transcript.
#[derive(Debug, Clone)]
pub enum Heard {
    Window(kvad::speech::Progress),
    Done(Box<kvad::speech::Transcript>),
    Failed(String),
    /// Asked for what the model cannot do, found once it tried: a language
    /// it does not know.
    Refused(String),
}

/// Where a video has got to, or the video.
#[derive(Debug, Clone)]
pub enum Reel {
    Step(kvad::video::Step),
    Done(Box<kvad::video::Filmed>),
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

/// What a generation that was stopped before the engine reached it ends
/// with, in place of a reply.
pub const CANCELLED: &str = "cancelled";

/// Whose a generation is: the account's row, or `None` with no
/// authentication, where the one person there is owns all of them.
pub type Owner = Option<i64>;

/// A generation's place in the queue's books, which travels with its job.
///
/// `stop` is this generation's and nobody else's. The engine's own flag
/// cannot be, because the engine has one for everything it runs and clears
/// it as each command starts.
struct Ticket {
    id: u64,
    owner: Owner,
    stop: Arc<AtomicBool>,
}

/// What a generation is making.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Work {
    Chat,
    Completion,
    Image,
    Video,
    Transcription,
}

/// Whether a generation may be turned away for want of room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    /// A request arriving: refused when the queue is full.
    Request,
    /// The next piece of work already let in — a request's second image, an
    /// evaluation's next case. Refusing it would throw away what the first
    /// piece cost, so it is not counted against the limit at the door,
    /// though it takes a place in the queue like any other.
    Continued,
}

/// Why a generation was not queued.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// As many are waiting as are allowed to.
    Full { waiting: usize, limit: usize },
    /// This account has as many waiting as one account may.
    Yours { waiting: usize, limit: usize },
    /// This account has started as many in the last minute as one account
    /// may, and may start another in `retry_secs`.
    Hurried { limit: usize, retry_secs: u64 },
    /// The scheduler's thread has ended, which it does not do.
    Stopped,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::Full { waiting, limit } => write!(
                f,
                "the queue is full: {waiting} generations are waiting and {limit} is the limit. \
                 Try again in a moment"
            ),
            Refused::Yours { waiting, limit } => write!(
                f,
                "you have {waiting} generations waiting, and {limit} is as many as one account may. \
                 Try again when one has started"
            ),
            Refused::Hurried { limit, retry_secs } => write!(
                f,
                "you have asked for {limit} generations in the last minute, which is as many as \
                 one account may. Try again in {retry_secs} seconds"
            ),
            Refused::Stopped => f.write_str("the engine thread has stopped"),
        }
    }
}

/// A generation that is waiting or running, as [`Scheduler::cancel`] and
/// [`Scheduler::queue`] find it. Kept in the order they were let in, which
/// is the order they run in.
struct Live {
    id: u64,
    owner: Owner,
    kind: Work,
    /// The resident it is for, as a client names one.
    model: String,
    stop: Arc<AtomicBool>,
    /// The interrupt flag of the engine running it, once one is.
    engine: Option<Arc<AtomicBool>>,
    queued: Instant,
    started: Option<Instant>,
}

/// One generation, as the queue's view reports it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Queued {
    pub id: u64,
    #[serde(skip)]
    pub owner: Owner,
    pub kind: Work,
    pub model: String,
    pub running: bool,
    /// Generations in front of it, the running one included.
    pub ahead: usize,
    /// How long it waited for the engine, or has so far.
    pub waited_secs: f64,
    /// How long it has been running, once it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub running_secs: Option<f64>,
}

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
        ticket: Ticket,
        on: Key,
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
        fresh: bool,
        out: tokio_mpsc::Sender<Piece>,
    },
    Complete {
        ticket: Ticket,
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
    Paint { ticket: Ticket, on: Key, request: kvad::image::ImageRequest, out: tokio_mpsc::Sender<Stroke> },
    Film { ticket: Ticket, on: Key, request: kvad::video::VideoRequest, out: tokio_mpsc::Sender<Reel> },
    Transcribe { ticket: Ticket, on: Key, audio: Vec<f32>, request: kvad::speech::Request, out: tokio_mpsc::Sender<Heard> },
}

impl Job {
    fn ticket(&self) -> Option<&Ticket> {
        match self {
            Job::Chat { ticket, .. }
            | Job::Complete { ticket, .. }
            | Job::Paint { ticket, .. }
            | Job::Film { ticket, .. }
            | Job::Transcribe { ticket, .. } => Some(ticket),
            _ => None,
        }
    }

    /// For a reply or a continuation: the resident it is for, and whether
    /// it asked for no cache.
    fn text(&self) -> Option<(&Key, bool)> {
        match self {
            Job::Chat { on, fresh, .. } | Job::Complete { on, fresh, .. } => Some((on, *fresh)),
            _ => None,
        }
    }

    /// End a generation that was stopped before it started. `try_send`,
    /// because this runs on the runtime's threads and the channel is empty:
    /// nothing has been sent to a generation that has not begun.
    fn cancelled(self) {
        match self {
            Job::Chat { out, .. } | Job::Complete { out, .. } => {
                let _ = out.try_send(Piece::Failed(CANCELLED.into()));
            }
            Job::Paint { out, .. } => {
                let _ = out.try_send(Stroke::Failed(CANCELLED.into()));
            }
            Job::Film { out, .. } => {
                let _ = out.try_send(Reel::Failed(CANCELLED.into()));
            }
            Job::Transcribe { out, .. } => {
                let _ = out.try_send(Heard::Failed(CANCELLED.into()));
            }
            _ => {}
        }
    }
}

/// The jobs nobody has started, and the word that there is one.
///
/// A deque under a lock rather than a channel, because a channel can only be
/// read from its front: counting what waits, taking a stopped generation out
/// of the middle, and — when the order stops being arrival order — choosing
/// whose turn it is all need to see the whole of it.
#[derive(Default)]
struct Queue {
    waiting: Mutex<Waiting>,
    arrived: Condvar,
}

#[derive(Default)]
struct Waiting {
    jobs: VecDeque<Job>,
    /// Nobody will take another job: the scheduler was dropped, or its
    /// thread ended.
    closed: bool,
    /// When each account last had a generation started, as a count of
    /// generations started. An account not in it has never had one.
    served: Served,
    turn: u64,
    /// When each account's requests of the last minute were let in, oldest
    /// first. Under the queue's lock because being let in is one decision:
    /// room in the queue, room in the account's share, and this.
    asked: std::collections::HashMap<i64, VecDeque<Instant>>,
}

/// The stretch a rate is counted over.
const MINUTE: std::time::Duration = std::time::Duration::from_secs(60);

/// How long `asked`'s owner must wait to be under `limit` a minute at
/// `now`, having forgotten what is older than a minute. `None` is no wait.
fn wait_for(asked: &mut VecDeque<Instant>, limit: usize, now: Instant) -> Option<u64> {
    while asked.front().is_some_and(|at| now.duration_since(*at) >= MINUTE) {
        asked.pop_front();
    }
    if asked.len() < limit {
        return None;
    }
    // The one that has to age out for there to be room: with exactly the
    // limit held, the oldest.
    let leaving = asked[asked.len() - limit];
    Some((MINUTE - now.duration_since(leaving)).as_secs_f64().ceil() as u64)
}

type Served = std::collections::HashMap<Owner, u64>;

/// Which of the waiting jobs runs next.
///
/// `waiting` is the queue in arrival order: a generation's owner, or `None`
/// for a job that is not a generation — a load, an unload. The answer is an
/// index into it.
///
/// Only the generations in front of the first other job are candidates,
/// because that job changes what is in memory and the ones behind it were
/// queued for the memory as it will be afterwards. Among the candidates,
/// each account's earliest stands for it, and the account served longest
/// ago goes first; never served is longest of all, and a tie goes to the
/// earlier arrival. One account alone is therefore arrival order.
///
/// This is round-robin over accounts by turns, not by tokens: two accounts
/// alternate whatever the length of what they asked for. Weighing a turn by
/// what it cost is deficit round-robin, and is the next step if turns turn
/// out to differ enough to matter.
fn pick(waiting: &[Option<Owner>], served: &Served) -> Option<usize> {
    let stretch = waiting.iter().take_while(|w| w.is_some()).count();
    if stretch == 0 {
        // Nothing, or a job that is not a generation at the front.
        return (!waiting.is_empty()).then_some(0);
    }
    let mut best: Option<(u64, usize)> = None;
    let mut seen: Vec<Owner> = Vec::new();
    for (at, owner) in waiting[..stretch].iter().flatten().enumerate() {
        if seen.contains(owner) {
            continue;
        }
        seen.push(*owner);
        // Zero is before every turn there has been.
        let last = served.get(owner).copied().unwrap_or(0);
        if best.is_none_or(|(least, _)| last < least) {
            best = Some((last, at));
        }
    }
    best.map(|(_, at)| at)
}

impl Waiting {
    fn owners(&self) -> Vec<Option<Owner>> {
        self.jobs.iter().map(|j| j.ticket().map(|t| t.owner)).collect()
    }

    /// Take the job whose turn it is, and note whose turn it was.
    fn take(&mut self) -> Option<Job> {
        self.take_if(|_| true)
    }

    /// Take the job whose turn it is, if it is one that `fits`. A job that
    /// does not is left, and so is everything behind it in turn: nothing
    /// that fits is looked for past it.
    fn take_if(&mut self, fits: impl Fn(&Job) -> bool) -> Option<Job> {
        let at = pick(&self.owners(), &self.served)?;
        if !fits(&self.jobs[at]) {
            return None;
        }
        let job = self.jobs.remove(at)?;
        if let Some(ticket) = job.ticket() {
            self.turn += 1;
            self.served.insert(ticket.owner, self.turn);
        }
        Some(job)
    }

    /// The ids of the waiting generations, in the order [`Waiting::take`]
    /// would run them if nothing else arrived.
    fn order(&self) -> Vec<u64> {
        let mut left: Vec<(Option<Owner>, Option<u64>)> =
            self.jobs.iter().map(|j| (j.ticket().map(|t| t.owner), j.ticket().map(|t| t.id))).collect();
        let (mut served, mut turn) = (self.served.clone(), self.turn);
        let mut order = Vec::new();
        while let Some(at) = pick(&left.iter().map(|(owner, _)| *owner).collect::<Vec<_>>(), &served) {
            let (owner, id) = left.remove(at);
            if let (Some(owner), Some(id)) = (owner, id) {
                turn += 1;
                served.insert(owner, turn);
                order.push(id);
            }
        }
        order
    }

    /// How many generations wait, and how many of them are `who`'s.
    fn count(&self, who: Owner) -> (usize, usize) {
        let tickets = self.jobs.iter().filter_map(|j| j.ticket());
        tickets.fold((0, 0), |(all, theirs), t| (all + 1, theirs + usize::from(t.owner == who)))
    }
}

impl Queue {
    fn waiting(&self) -> std::sync::MutexGuard<'_, Waiting> {
        self.waiting.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Stop taking jobs, and drop the ones waiting: their callers find their
    /// channels closed, which is how they hear.
    fn close(&self) {
        let mut waiting = self.waiting();
        waiting.closed = true;
        waiting.jobs.clear();
        self.arrived.notify_all();
    }

    /// The job whose turn it is, without waiting, if there is one and it
    /// `fits`: what a batch under way asks, to see who may join it.
    fn joining(&self, fits: impl Fn(&Job) -> bool) -> Option<Job> {
        let mut waiting = self.waiting();
        match waiting.closed {
            true => None,
            false => waiting.take_if(fits),
        }
    }

    /// The next job, waiting for one if there is none. `None` once closed.
    fn next(&self) -> Option<Job> {
        let mut waiting = self.waiting();
        loop {
            if waiting.closed {
                return None;
            }
            if let Some(job) = waiting.take() {
                return Some(job);
            }
            waiting = self.arrived.wait(waiting).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Closes the queue when the scheduler's thread ends, however it ends, so
/// that a job submitted afterwards is refused rather than left waiting for
/// nobody.
struct Closing(Arc<Queue>);

impl Drop for Closing {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// How many generations may wait when nothing says otherwise.
///
/// Each one waiting is a client holding a connection open, and at the pace
/// of a large model each is most of a minute of somebody's patience. Sixteen
/// is already a long wait for the last of them; it is a ceiling for a burst,
/// not a target.
pub const DEFAULT_LIMIT: usize = 16;

/// How many of them may be one account's, when nothing says otherwise: a
/// quarter of the queue, so that it takes four accounts to fill it.
pub const DEFAULT_SHARE: usize = 4;

/// How many replies run at once on one resident, when nothing says
/// otherwise.
///
/// Every reply in a batch waits for the whole step, so each is slower than
/// it would be alone and all of them together are faster than in turn: on
/// Qwen2.5-1.5B at `cpu-q8`, four at once are 39 ms a token each against 21
/// alone and 83 in turn. Four is where that trade is still plainly a good
/// one; past it the wait for a token grows faster than the total does.
pub const DEFAULT_BATCH: usize = 4;

/// Makes the loader for each new engine. One per engine, because a
/// [`Loader`] is `FnMut` and moves to the thread that calls it.
type Loaders = Box<dyn Fn() -> Loader + Send>;

pub struct Scheduler {
    queue: Arc<Queue>,
    shared: Shared,
    budget: Budget,
    /// How many generations may wait. Zero is no limit.
    limit: usize,
    /// How many of them may be one account's. Zero is no limit.
    share: usize,
    /// How many generations one account may ask for in a minute. Zero is
    /// no limit, which is what it is unless somebody sets one.
    rate: usize,
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.queue.close();
    }
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
    /// Every generation submitted and not yet finished, the waiting ones
    /// included: a generation can be stopped before the engine reaches it.
    live: Arc<Mutex<Vec<Live>>>,
    /// The id the next generation gets. They start at one.
    issued: Arc<AtomicU64>,
    /// Memory set aside for work that is not a resident; see [`Reserved`].
    reserved: Arc<Mutex<Option<Held>>>,
    /// Layers held for models on other machines; see [`Lodged`]. Each with
    /// the number its guard knows it by.
    guests: Arc<Mutex<Vec<(u64, Held)>>>,
    /// How many replies run at once on one resident; see
    /// [`Scheduler::batching`]. Here because the thread that runs them
    /// reads it.
    batch: Arc<AtomicUsize>,
}

/// Memory of the budget held for something that is no model in this
/// process: a LoRA training run, which is a process of its own on the same
/// machine and the same GPU. Admission counts it as it counts a resident,
/// so a model is not loaded into the memory a backward pass is about to
/// take. Given back when this is dropped.
pub struct Reserved(Arc<Mutex<Option<Held>>>);

/// Memory this server has given to a model that is another machine's: the
/// layers it holds as a cluster worker, and the cache they will grow.
/// Admission counts it as it counts a resident, so a server's own models
/// and the layers it holds for others share one budget. Given back when
/// this is dropped, which is when the coordinator lets go.
pub struct Lodged {
    guests: Arc<Mutex<Vec<(u64, Held)>>>,
    number: u64,
}

impl Lodged {
    /// Charge what it turned out to be, now that it is loaded.
    pub fn resize(&self, bytes: u64) {
        let mut guests = self.guests.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, held)) = guests.iter_mut().find(|(number, _)| *number == self.number) {
            held.commit = bytes;
        }
    }
}

impl Drop for Lodged {
    fn drop(&mut self) {
        self.guests.lock().unwrap_or_else(|e| e.into_inner()).retain(|(number, _)| *number != self.number);
    }
}

impl Drop for Reserved {
    fn drop(&mut self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

impl Shared {
    fn residents(&self) -> std::sync::MutexGuard<'_, Vec<Resident>> {
        self.residents.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn live(&self) -> std::sync::MutexGuard<'_, Vec<Live>> {
        self.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Enter a generation in the books, and hand back what its job carries.
    fn admit(&self, owner: Owner, kind: Work, on: &Key) -> Ticket {
        let id = self.issued.fetch_add(1, Ordering::Relaxed) + 1;
        let stop = Arc::new(AtomicBool::new(false));
        self.live().push(Live {
            id,
            owner,
            kind,
            model: id_of(&on.repo, on.backend),
            stop: Arc::clone(&stop),
            engine: None,
            queued: Instant::now(),
            started: None,
        });
        Ticket { id, owner, stop }
    }

    /// `id` has left the queue and is the scheduler thread's now: it has
    /// stopped waiting, whatever becomes of it.
    fn taken(&self, id: u64) {
        if let Some(live) = self.live().iter_mut().find(|l| l.id == id) {
            live.started = Some(Instant::now());
        }
    }

    /// The engine has reached `id`: from here stopping it means raising
    /// `engine`'s flag as well.
    fn begin(&self, id: u64, engine: &Arc<AtomicBool>) {
        if let Some(live) = self.live().iter_mut().find(|l| l.id == id) {
            live.engine = Some(Arc::clone(engine));
        }
    }

    /// `id` is over, however it ended.
    fn finish(&self, id: u64) {
        self.live().retain(|l| l.id != id);
    }

    fn reserved(&self) -> std::sync::MutexGuard<'_, Option<Held>> {
        self.reserved.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Everything admission counts: the residents, and what is reserved.
    fn held(&self) -> Vec<Held> {
        let mut held: Vec<Held> = self.residents().iter().map(|r| Held { id: r.id.clone(), commit: r.commit }).collect();
        held.extend(self.reserved().clone());
        held.extend(self.guests.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(_, held)| held.clone()));
        held
    }
}

impl Scheduler {
    pub fn spawn(loaders: impl Fn() -> Loader + Send + 'static, budget: Budget) -> Self {
        let queue = Arc::new(Queue::default());
        let shared = Shared::default();
        let (theirs, loaders) = (shared.clone(), Box::new(loaders) as Loaders);
        shared.batch.store(DEFAULT_BATCH, Ordering::Relaxed);
        let closing = Closing(Arc::clone(&queue));
        std::thread::Builder::new()
            .name("kvad-scheduler".into())
            .spawn(move || run(closing, loaders, budget, theirs))
            .expect("failed to spawn the scheduler thread");
        Scheduler { queue, shared, budget, limit: DEFAULT_LIMIT, share: DEFAULT_SHARE, rate: 0 }
    }

    /// Run up to `batch` replies at once on one resident. One, or zero, is
    /// one at a time, as it was before there was a choice.
    pub fn batching(self, batch: usize) -> Self {
        self.shared.batch.store(batch.max(1), Ordering::Relaxed);
        self
    }

    /// How many replies run at once on one resident.
    pub fn batch(&self) -> usize {
        self.shared.batch.load(Ordering::Relaxed)
    }

    /// Let one account ask for `rate` generations a minute, and refuse the
    /// next until one of them is a minute old. Zero lets it ask freely.
    ///
    /// Counted when a request is let in, not when it runs, and not for the
    /// rest of a request already in: four images are one request. Like the
    /// share, it is for accounts, and with none there is nobody to hold to
    /// it.
    pub fn pacing(mut self, rate: usize) -> Self {
        self.rate = rate;
        self
    }

    /// How many generations one account may ask for in a minute, or zero
    /// for any number.
    pub fn rate(&self) -> usize {
        self.rate
    }

    /// Let one account have `share` generations waiting, and refuse its
    /// next. Zero lets it have any number, up to the whole queue.
    pub fn sharing(mut self, share: usize) -> Self {
        self.share = share;
        self
    }

    /// How many generations one account may have waiting, or zero for any
    /// number.
    pub fn share(&self) -> usize {
        self.share
    }

    /// Whether `who` may queue another generation, given what waits.
    ///
    /// The whole queue first: when there is no room for anybody, that is
    /// the truer thing to say. The account's share applies only where there
    /// are accounts — with none, `who` is everybody, and a share of the
    /// queue for everybody is just a smaller queue.
    ///
    /// The account's rate is asked last and only noted when `counted`: a
    /// request turned away for want of room has not used one of its
    /// minute's, and [`Scheduler::room`] is a question, not a request.
    fn admits(&self, waiting: &mut Waiting, who: Owner, counted: bool) -> Result<(), Refused> {
        let (all, theirs) = waiting.count(who);
        if self.limit > 0 && all >= self.limit {
            return Err(Refused::Full { waiting: all, limit: self.limit });
        }
        if who.is_some() && self.share > 0 && theirs >= self.share {
            return Err(Refused::Yours { waiting: theirs, limit: self.share });
        }
        if let (Some(id), true) = (who, self.rate > 0) {
            let (asked, now) = (waiting.asked.entry(id).or_default(), Instant::now());
            if let Some(retry_secs) = wait_for(asked, self.rate, now) {
                return Err(Refused::Hurried { limit: self.rate, retry_secs });
            }
            if counted {
                asked.push_back(now);
            }
        }
        Ok(())
    }

    /// Let `limit` generations wait, and refuse the one after. Zero lets
    /// any number.
    pub fn bounded(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// How many generations may wait, or zero for any number.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// How many jobs are waiting for an engine, the running one included.
    /// Loads and the like are counted; [`Scheduler::queue`] is the
    /// generations alone, and whose they are.
    pub fn depth(&self) -> usize {
        self.shared.waiting.load(Ordering::Relaxed)
            + usize::from(self.shared.busy.load(Ordering::Relaxed))
    }

    /// Whether a request for a generation would be let in now.
    ///
    /// For a handler that answers before it queues — a stream whose headers
    /// have gone by the time the job is submitted, a video whose row is made
    /// first — so that a full queue is still a status code. The submission
    /// itself asks again, under the lock, and that is the one that counts.
    pub fn room(&self, who: Owner) -> Result<(), Refused> {
        self.admits(&mut self.queue.waiting(), who, false)
    }

    /// Every generation waiting or running, in the order they run — which
    /// is whose turn it is, not who arrived first; see [`pick`]. The order
    /// is as things stand: a request arriving from an account that has had
    /// no turn lately goes in ahead of some of these.
    pub fn queue(&self) -> Vec<Queued> {
        let now = Instant::now();
        let secs = |from: Instant, to: Instant| to.duration_since(from).as_secs_f64();
        // The queue's lock before the books', as everywhere.
        let order = self.queue.waiting().order();
        let live = self.shared.live();
        // What has left the queue comes first; what the scheduler's thread
        // has taken and not yet marked is between the two for a moment, and
        // is counted with the first.
        let place = |l: &Live| order.iter().position(|id| *id == l.id).map_or(0, |at| at + 1);
        let mut rows: Vec<&Live> = live.iter().collect();
        rows.sort_by_key(|l| place(l));
        rows.into_iter()
            .enumerate()
            .map(|(ahead, l)| Queued {
                id: l.id,
                owner: l.owner,
                kind: l.kind,
                model: l.model.clone(),
                running: l.started.is_some(),
                ahead,
                waited_secs: secs(l.queued, l.started.unwrap_or(now)),
                running_secs: l.started.map(|s| secs(s, now)),
            })
            .collect()
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    /// Every resident, in the order they were loaded.
    pub fn residents(&self) -> Vec<Resident> {
        self.shared.residents().clone()
    }

    /// Bytes of the budget no resident has been charged, and nothing has
    /// reserved.
    pub fn left(&self) -> u64 {
        let spent: u64 = self.shared.held().iter().map(|h| h.commit).sum();
        self.budget.total.saturating_sub(spent)
    }

    /// Set `bytes` of the budget aside for `what`, or say why not: what is
    /// left, and who holds the rest. One at a time.
    ///
    /// Unlike a model, never admitted "alone, whatever it takes": what asks
    /// is a training run, whose memory is a figure measured beforehand and
    /// whose overrun takes the machine down.
    pub fn reserve(&self, what: &str, bytes: u64) -> Result<Reserved, String> {
        let gb = |b: u64| format!("{:.1} GB", b as f64 / 1e9);
        let mut reserved = self.shared.reserved();
        if let Some(other) = reserved.as_ref() {
            return Err(format!("{} already has {} set aside for it", other.id, gb(other.commit)));
        }
        let residents = self.shared.residents();
        let spent: u64 = residents.iter().map(|r| r.commit).sum::<u64>() + self.guests().iter().map(|g| g.commit).sum::<u64>();
        let left = self.budget.total.saturating_sub(spent);
        if bytes > left {
            let holding = match residents.is_empty() {
                true => String::new(),
                false => format!(
                    ": {} in memory. Unload {} first",
                    residents.iter().map(|r| format!("{} ({})", r.id, gb(r.commit))).collect::<Vec<_>>().join(", "),
                    if residents.len() == 1 { "it" } else { "one" }
                ),
            };
            return Err(format!("{what} takes about {}, and {} is left of {}{holding}", gb(bytes), gb(left), gb(self.budget.total)));
        }
        *reserved = Some(Held { id: what.to_string(), commit: bytes });
        Ok(Reserved(Arc::clone(&self.shared.reserved)))
    }

    /// What is held here for models on other machines.
    pub fn guests(&self) -> Vec<Held> {
        self.shared.guests.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(_, held)| held.clone()).collect()
    }

    /// Take `bytes` of the budget for `what`, which another machine's model
    /// asks this one to hold, or say why not. Never "alone, whatever it
    /// takes": the model is not this server's, and the coordinator that
    /// asked can place the layers somewhere else.
    pub fn lodge(&self, what: &str, bytes: u64) -> Result<Lodged, String> {
        let gb = |b: u64| format!("{:.1} GB", b as f64 / 1e9);
        let held = self.shared.held();
        let left = self.budget.total.saturating_sub(held.iter().map(|h| h.commit).sum());
        if bytes > left {
            let holding = match held.is_empty() {
                true => String::new(),
                false => format!(", with {} in memory", held.iter().map(|h| format!("{} ({})", h.id, gb(h.commit))).collect::<Vec<_>>().join(", ")),
            };
            return Err(format!("{what} takes about {}, and this machine has {} left of {} for models{holding}", gb(bytes), gb(left), gb(self.budget.total)));
        }
        let mut guests = self.shared.guests.lock().unwrap_or_else(|e| e.into_inner());
        let number = guests.iter().map(|(number, _)| number + 1).max().unwrap_or(1);
        guests.push((number, Held { id: what.to_string(), commit: bytes }));
        Ok(Lodged { guests: Arc::clone(&self.shared.guests), number })
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

    /// Stop `who`'s generations: the one with this id, or every one of
    /// theirs for `None`. How many were told to stop.
    ///
    /// Somebody else's generation is not counted and not touched, so naming
    /// one reads the same as naming one that does not exist. A waiting
    /// generation is stopped as well as the running one: it leaves the queue
    /// at once and ends with [`CANCELLED`], having cost nothing, and its
    /// place is there for the next request.
    pub fn cancel(&self, who: Owner, id: Option<u64>) -> usize {
        let theirs = |owner: Owner, of: u64| owner == who && id.is_none_or(|id| of == id);

        // The queue's lock before the books', here and in `generate`.
        let mut waiting = self.queue.waiting();
        let (stopped, staying): (VecDeque<Job>, VecDeque<Job>) = std::mem::take(&mut waiting.jobs)
            .into_iter()
            .partition(|job| job.ticket().is_some_and(|t| theirs(t.owner, t.id)));
        waiting.jobs = staying;
        drop(waiting);

        let mut count = stopped.len();
        self.shared.waiting.fetch_sub(count, Ordering::Relaxed);
        for job in stopped {
            if let Some(ticket) = job.ticket() {
                self.shared.finish(ticket.id);
            }
            job.cancelled();
        }

        // What is left of theirs has been taken by the scheduler's thread:
        // running, or about to be.
        for l in self.shared.live().iter().filter(|l| theirs(l.owner, l.id)) {
            l.stop.store(true, Ordering::Relaxed);
            if let Some(engine) = &l.engine {
                engine.store(true, Ordering::Relaxed);
            }
            count += 1;
        }
        count
    }

    fn submit(&self, job: Job) -> Result<(), String> {
        let mut waiting = self.queue.waiting();
        if waiting.closed {
            return Err(Refused::Stopped.to_string());
        }
        self.shared.waiting.fetch_add(1, Ordering::Relaxed);
        waiting.jobs.push_back(job);
        self.queue.arrived.notify_one();
        Ok(())
    }

    /// [`Scheduler::submit`] for a generation: counted against the limit,
    /// and entered in the books under the same lock that queues it, so the
    /// books' order is the queue's.
    fn generate(
        &self,
        who: Owner,
        entry: Entry,
        kind: Work,
        on: &Key,
        job: impl FnOnce(Ticket) -> Job,
    ) -> Result<u64, Refused> {
        let mut waiting = self.queue.waiting();
        if waiting.closed {
            return Err(Refused::Stopped);
        }
        if entry == Entry::Request {
            self.admits(&mut waiting, who, true)?;
        }
        let ticket = self.shared.admit(who, kind, on);
        let id = ticket.id;
        self.shared.waiting.fetch_add(1, Ordering::Relaxed);
        waiting.jobs.push_back(job(ticket));
        self.queue.arrived.notify_one();
        Ok(id)
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
    ///
    /// The number beside the receiver is the generation's id, which is what
    /// `who` names to [`Scheduler::cancel`] it.
    pub fn chat(
        &self,
        who: Owner,
        entry: Entry,
        on: &Key,
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
        fresh: bool,
    ) -> Result<(u64, tokio_mpsc::Receiver<Piece>), Refused> {
        // Bounded, so a client that reads slowly slows the generation down
        // rather than filling memory with tokens it has not asked for. 64 is
        // a second or so of decoding at the rates this engine reaches.
        let (out, rx) = tokio_mpsc::channel(64);
        let id = self.generate(who, entry, Work::Chat, on, |ticket| Job::Chat {
            ticket,
            on: on.clone(),
            messages,
            tools,
            sampling,
            fresh,
            out,
        })?;
        Ok((id, rx))
    }

    /// Continue a prompt, with no chat template involved.
    ///
    /// `explain` is how many candidates to report per token, and `fresh`
    /// drops the KV cache first — see [`kvad::service::Cmd::Complete`] for why
    /// that is a measurement question rather than an output one.
    pub fn complete(
        &self,
        who: Owner,
        entry: Entry,
        on: &Key,
        prompt: String,
        sampling: Sampling,
        explain: usize,
        fresh: bool,
    ) -> Result<(u64, tokio_mpsc::Receiver<Piece>), Refused> {
        let (out, rx) = tokio_mpsc::channel(64);
        let id = self.generate(who, entry, Work::Completion, on, |ticket| Job::Complete {
            ticket,
            on: on.clone(),
            prompt,
            sampling,
            explain,
            fresh,
            out,
        })?;
        Ok((id, rx))
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
        who: Owner,
        entry: Entry,
        on: &Key,
        request: kvad::image::ImageRequest,
    ) -> Result<(u64, tokio_mpsc::Receiver<Stroke>), Refused> {
        // A step is seconds, so a handful of them buffered is plenty.
        let (out, rx) = tokio_mpsc::channel(8);
        let id =
            self.generate(who, entry, Work::Image, on, |ticket| Job::Paint { ticket, on: on.clone(), request, out })?;
        Ok((id, rx))
    }

    /// Make a video on `on`, as [`Scheduler::paint`] makes an image.
    ///
    /// The engine is this job's for minutes, and every other request on the
    /// same engine waits behind it.
    pub fn film(
        &self,
        who: Owner,
        entry: Entry,
        on: &Key,
        request: kvad::video::VideoRequest,
    ) -> Result<(u64, tokio_mpsc::Receiver<Reel>), Refused> {
        let (out, rx) = tokio_mpsc::channel(8);
        let id =
            self.generate(who, entry, Work::Video, on, |ticket| Job::Film { ticket, on: on.clone(), request, out })?;
        Ok((id, rx))
    }

    /// Write down what is said in `audio`, 16 kHz mono samples, on `on`, as
    /// [`Scheduler::paint`] makes an image: a [`Heard::Window`] for every 30
    /// seconds of it, then the transcript.
    pub fn transcribe(
        &self,
        who: Owner,
        entry: Entry,
        on: &Key,
        audio: Vec<f32>,
        request: kvad::speech::Request,
    ) -> Result<(u64, tokio_mpsc::Receiver<Heard>), Refused> {
        let (out, rx) = tokio_mpsc::channel(8);
        let id = self.generate(who, entry, Work::Transcription, on, |ticket| Job::Transcribe {
            ticket,
            on: on.clone(),
            audio,
            request,
            out,
        })?;
        Ok((id, rx))
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
fn run(queue: Closing, loaders: Loaders, budget: Budget, shared: Shared) {
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

    // The replies running together on one resident, if any are.
    let mut batch: Option<Batch> = None;

    loop {
        // With replies under way: let in who may join, pass on what the
        // engine has said, and come round again. Nothing else is taken from
        // the queue until they have ended.
        if let Some(b) = batch.as_mut() {
            // Streams that have ended, dropped only once `busy` is settled.
            let mut ended: Vec<tokio_mpsc::Sender<Piece>> = Vec::new();
            match slots.iter().find(|s| s.key == b.on) {
                Some(slot) => {
                    let most = shared.batch.load(Ordering::Relaxed).max(1);
                    while !b.alone && b.streams.len() < most {
                        let fits = |job: &Job| job.text().is_some_and(|(on, fresh)| on == &b.on && !fresh);
                        let Some(job) = queue.0.joining(fits) else { break };
                        shared.waiting.fetch_sub(1, Ordering::Relaxed);
                        clock += 1;
                        ended.extend(b.admit(job, slot, &shared));
                    }
                    let on = b.on.clone();
                    ended.extend(b.pump(slot, &shared, &|cached| touch(&on, clock, Some(cached))));
                }
                // Unloaded from under them, which nothing here does while
                // they run.
                None => ended.extend(b.fail(&missing(&b.on), &shared)),
            }
            if b.streams.is_empty() {
                batch = None;
                shared.busy.store(false, Ordering::Relaxed);
            }
            drop(ended);
            continue;
        }

        let Some(job) = queue.0.next() else { break };
        // Busy before the job leaves the waiting count, so that for the
        // moment between the two the depth reads one too many rather than one
        // too few: a queue that looks empty while it is not is the worse lie.
        shared.busy.store(true, Ordering::Relaxed);
        shared.waiting.fetch_sub(1, Ordering::Relaxed);
        clock += 1;

        // A reply begins a batch, of one for now. What may join it is asked
        // at the top of the loop.
        if let Some((on, fresh)) = job.text().map(|(on, fresh)| (on.clone(), fresh)) {
            let mut b = Batch { on, alone: fresh, streams: Vec::new() };
            let ended = match slots.iter().find(|s| s.key == b.on) {
                Some(slot) => b.admit(job, slot, &shared),
                None => {
                    shared.taken(job.ticket().map_or(0, |t| t.id));
                    job.refused(Piece::Failed(missing(&b.on)), &shared)
                }
            };
            match b.streams.is_empty() {
                true => shared.busy.store(false, Ordering::Relaxed),
                false => batch = Some(b),
            }
            drop(ended);
            continue;
        }
        if let Some(ticket) = job.ticket() {
            shared.taken(ticket.id);
        }

        // What the caller is waiting for: an answer, or the end of its
        // stream. Handed over only once `busy` is clear, because a caller
        // that has its answer and then reads the depth must not find its own
        // job still counted in it.
        let reply: Box<dyn FnOnce()> = match job {
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
                Box::new(move || drop(done.send(result)))
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
                Box::new(move || drop(done.send(failed.map_or(Ok(gone), Err))))
            }

            // Begun above, as a batch.
            Job::Chat { .. } | Job::Complete { .. } => Box::new(|| {}),

            Job::Tokenize { on, text, done } => {
                let answer = match slots.iter().find(|s| s.key == on) {
                    Some(slot) => {
                        slot.engine.send(Cmd::Tokenize(text));
                        drain_tokens(&slot.engine.rx)
                    }
                    None => Err(missing(&on)),
                };
                Box::new(move || drop(done.send(answer)))
            }

            Job::Score { on, text, window, progress, done } => {
                let answer = match slots.iter().find(|s| s.key == on) {
                    Some(slot) => {
                        slot.engine.send(Cmd::Perplexity { text, window });
                        let scored = drain_score(&slot.engine.rx, &progress);
                        // Scoring holds nothing of its own afterwards, and
                        // what it found the model holding is still held.
                        touch(&on, clock, None);
                        scored
                    }
                    None => Err(missing(&on)),
                };
                Box::new(move || drop(done.send(answer)))
            }

            Job::Paint { ticket, on, request, out } => {
                match slots.iter().find(|s| s.key == on) {
                    _ if ticket.stop.load(Ordering::Relaxed) => {
                        let _ = out.blocking_send(Stroke::Failed(CANCELLED.into()));
                    }
                    Some(slot) => {
                        shared.begin(ticket.id, &slot.engine.cancel);
                        slot.engine.send(Cmd::Paint(request));
                        drain_paint(&slot.engine.rx, &out, &ticket.stop, &slot.engine.cancel);
                        touch(&on, clock, None);
                    }
                    None => {
                        let _ = out.blocking_send(Stroke::Failed(missing(&on)));
                    }
                }
                shared.finish(ticket.id);
                Box::new(move || drop(out))
            }

            Job::Film { ticket, on, request, out } => {
                match slots.iter().find(|s| s.key == on) {
                    // Deleted while it waited: a minute of the GPU saved.
                    _ if out.is_closed() => {}
                    _ if ticket.stop.load(Ordering::Relaxed) => {
                        let _ = out.blocking_send(Reel::Failed(CANCELLED.into()));
                    }
                    Some(slot) => {
                        shared.begin(ticket.id, &slot.engine.cancel);
                        slot.engine.send(Cmd::Film(request));
                        drain_film(&slot.engine.rx, &out, &ticket.stop, &slot.engine.cancel);
                        touch(&on, clock, None);
                    }
                    None => {
                        let _ = out.blocking_send(Reel::Failed(missing(&on)));
                    }
                }
                shared.finish(ticket.id);
                Box::new(move || drop(out))
            }

            Job::Transcribe { ticket, on, audio, request, out } => {
                match slots.iter().find(|s| s.key == on) {
                    // Nobody is waiting for it any more.
                    _ if out.is_closed() => {}
                    _ if ticket.stop.load(Ordering::Relaxed) => {
                        let _ = out.blocking_send(Heard::Failed(CANCELLED.into()));
                    }
                    Some(slot) => {
                        shared.begin(ticket.id, &slot.engine.cancel);
                        slot.engine.send(Cmd::Transcribe { audio, request });
                        drain_transcribe(&slot.engine.rx, &out, &ticket.stop, &slot.engine.cancel);
                        touch(&on, clock, None);
                    }
                    None => {
                        let _ = out.blocking_send(Heard::Failed(missing(&on)));
                    }
                }
                shared.finish(ticket.id);
                Box::new(move || drop(out))
            }
        };

        shared.busy.store(false, Ordering::Relaxed);
        reply();
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
    let held = shared.held();
    let need = crate::memory::need(&key.repo, key.backend, budget.context);
    let left = budget.total.saturating_sub(held.iter().map(|h| h.commit).sum());
    let admission = match crate::memory::admit(&budget, &held, &id, &need) {
        // Too large for what is left here, and `[cluster]` names machines
        // to put the rest on: that is the model they are for. It goes to
        // the placing, which is told what is left and refuses it in its
        // own words if no placing fits.
        Admission::Refused(_) if left > 0 && need.bytes().is_some() && crate::cluster::spreads(&key.repo, key.backend) => Admission::Fits { commit: left },
        other => other,
    };
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
    crate::cluster::set_room(Some(room));
    let engine = Engine::spawn(loaders());
    engine.send(Cmd::Load { repo: key.repo.clone(), backend: key.backend });
    // What the model is charged for below is what it may keep: every
    // conversation's cache together, not each one's.
    engine.send(Cmd::Hold(budget.context));
    let result = drain_load(&engine.rx, progress);
    kvad::experts::set_room(None);
    crate::cluster::set_room(None);
    let model = result.map_err(LoadError::Failed)?;

    // Charged what it turned out to be, now that it is loaded, rather than
    // what the files suggested. A model over the budget alone is charged
    // the budget, which is all there is to charge.
    let kv = match need.kv {
        0 => (model.kv_bytes_per_token * model.n_ctx.min(budget.context)) as u64,
        kv => kv,
    };
    // A model across machines is charged what of it is here: this
    // machine's layers and the ends, and the cache of those layers. The
    // rest is the workers' to count, and they do.
    let (weights, kv) = match crate::cluster::here(&key.repo, key.backend) {
        Some(spread) => (spread.bytes_here as u64, kv * spread.layers_here as u64 / spread.layers.max(1) as u64),
        None => (model.weight_bytes as u64, kv),
    };
    let (commit, streams) = match admission {
        Admission::Streams { room } => (room, true),
        _ => ((weights + kv).min(budget.total), false),
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
                video,
                speech,
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
                    kind: match (&image, &video) {
                        _ if speech => Kind::Speech,
                        (Some(_), _) => Kind::Image,
                        (_, Some(_)) => Kind::Video,
                        _ => Kind::Chat,
                    },
                    image,
                    video,
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

/// A reply under way: its place in the books, and where its pieces go.
struct Stream {
    ticket: Ticket,
    out: tokio_mpsc::Sender<Piece>,
    /// Pieces the reader has not had room for yet. A reader slower than the
    /// engine used to slow the engine down; with several replies on one
    /// engine that would be one reader slowing everybody's, so what it has
    /// not taken waits here instead. At most a reply's worth.
    backlog: VecDeque<Piece>,
}

/// The replies running together on one resident.
struct Batch {
    on: Key,
    /// Nothing may join: the reply asked for no cache, which makes it a
    /// measurement, and a measurement shares the engine with nobody.
    alone: bool,
    streams: Vec<Stream>,
}

impl Job {
    /// End a reply the engine will never hear of, with `piece`, and hand
    /// back its stream for the caller to drop when it is ready to.
    ///
    /// The send waits for room, as it did when replies ran one at a time:
    /// the channel of a reply that has not started is empty unless a test
    /// filled it to hold this thread still.
    fn refused(self, piece: Piece, shared: &Shared) -> Option<tokio_mpsc::Sender<Piece>> {
        match self {
            Job::Chat { ticket, out, .. } | Job::Complete { ticket, out, .. } => {
                let _ = out.blocking_send(piece);
                shared.finish(ticket.id);
                Some(out)
            }
            _ => None,
        }
    }
}

impl Batch {
    /// Send a reply to the engine to run beside the others. One stopped
    /// while it waited is ended instead, and its stream handed back.
    fn admit(&mut self, job: Job, slot: &Slot, shared: &Shared) -> Option<tokio_mpsc::Sender<Piece>> {
        let ticket = job.ticket()?;
        shared.taken(ticket.id);
        if ticket.stop.load(Ordering::Relaxed) {
            // Stopped while it waited: the engine never hears of it.
            return job.refused(Piece::Failed(CANCELLED.into()), shared);
        }
        let (ticket, what, out) = match job {
            Job::Chat { ticket, messages, tools, sampling, fresh, out, .. } => {
                (ticket, Cmd::Chat { messages, tools, sampling, fresh }, out)
            }
            Job::Complete { ticket, prompt, sampling, explain, fresh, out, .. } => {
                (ticket, Cmd::Complete { prompt, sampling, explain, fresh }, out)
            }
            _ => return None,
        };
        // The reply's own flag goes with it: the engine stops this one when
        // it is raised, and nobody else's.
        slot.engine.send(Cmd::Beside { id: ticket.id, stop: Arc::clone(&ticket.stop), what: Box::new(what) });
        self.streams.push(Stream { ticket, out, backlog: VecDeque::new() });
        None
    }

    /// Pass on what the engine has said since last time, waiting a moment
    /// for it to say something. The streams that ended are handed back.
    ///
    /// A moment and no longer, because this is also how often the queue is
    /// asked who may join: a token is tens of milliseconds, a prefill can
    /// be seconds, and somebody arriving during one should not wait for it
    /// to end before they are even sent to the engine.
    fn pump(
        &mut self,
        slot: &Slot,
        shared: &Shared,
        held: &dyn Fn(usize),
    ) -> Vec<tokio_mpsc::Sender<Piece>> {
        const MOMENT: std::time::Duration = std::time::Duration::from_millis(20);
        let mut over: Vec<u64> = Vec::new();
        match slot.engine.rx.recv_timeout(MOMENT) {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return self.fail("the engine thread has stopped", shared);
            }
            Ok(first) => {
                let mut next = Some(first);
                while let Some(evt) = next {
                    over.extend(self.hear(evt, held));
                    next = slot.engine.rx.try_recv().ok();
                }
            }
        }

        for stream in &mut self.streams {
            while let Some(piece) = stream.backlog.pop_front() {
                match stream.out.try_send(piece) {
                    Ok(()) => {}
                    Err(tokio_mpsc::error::TrySendError::Full(piece)) => {
                        stream.backlog.push_front(piece);
                        break;
                    }
                    // Nobody is reading: stop writing, and stop the reply.
                    Err(tokio_mpsc::error::TrySendError::Closed(_)) => stream.backlog.clear(),
                }
            }
            if stream.out.is_closed() {
                stream.ticket.stop.store(true, Ordering::Relaxed);
            }
        }

        let (ended, running): (Vec<Stream>, Vec<Stream>) =
            std::mem::take(&mut self.streams).into_iter().partition(|s| over.contains(&s.ticket.id));
        self.streams = running;
        ended.into_iter().map(|stream| stream.end(shared)).collect()
    }

    /// One thing the engine said: put it on its reply's backlog, and say
    /// whose reply it ended, if it ended one.
    fn hear(&mut self, evt: Evt, held: &dyn Fn(usize)) -> Option<u64> {
        // Anything not under an id is not a reply's: the engine's own news.
        let Evt::Of(id, evt) = evt else { return None };
        let stream = self.streams.iter_mut().find(|s| s.ticket.id == id)?;
        let piece = match *evt {
            Evt::Token(text) => Piece::Token(text),
            Evt::Chose(chosen) => Piece::Chose(chosen),
            Evt::Done(stats) => {
                // What the cache holds now, across every conversation the
                // model is keeping.
                held(stats.held_tokens);
                Piece::Done(stats)
            }
            Evt::Error(e) => Piece::Failed(e),
            // Sent when no model is loaded, which for a reply is a refusal
            // rather than a status line.
            Evt::Status(message) => Piece::Failed(message),
            _ => return None,
        };
        let last = !matches!(piece, Piece::Token(_) | Piece::Chose(_));
        stream.backlog.push_back(piece);
        last.then_some(id)
    }

    /// End every reply with `why`.
    fn fail(&mut self, why: &str, shared: &Shared) -> Vec<tokio_mpsc::Sender<Piece>> {
        let streams = std::mem::take(&mut self.streams);
        streams
            .into_iter()
            .map(|mut stream| {
                stream.backlog.push_back(Piece::Failed(why.to_string()));
                stream.end(shared)
            })
            .collect()
    }
}

impl Stream {
    /// The reply is over as far as the engine goes. Its place in the books
    /// goes now; what its reader has not taken yet follows on a thread of
    /// its own, so that a slow reader of a finished reply holds nobody up.
    fn end(mut self, shared: &Shared) -> tokio_mpsc::Sender<Piece> {
        shared.finish(self.ticket.id);
        while let Some(piece) = self.backlog.pop_front() {
            match self.out.try_send(piece) {
                Ok(()) => {}
                Err(tokio_mpsc::error::TrySendError::Closed(_)) => {
                    self.backlog.clear();
                }
                Err(tokio_mpsc::error::TrySendError::Full(piece)) => {
                    self.backlog.push_front(piece);
                    let (out, rest) = (self.out.clone(), std::mem::take(&mut self.backlog));
                    std::thread::spawn(move || {
                        for piece in rest {
                            if out.blocking_send(piece).is_err() {
                                break;
                            }
                        }
                    });
                }
            }
        }
        self.out
    }
}

/// Forward a transcription's windows, then the transcript, as
/// [`drain_film`] does a video's, and for its reason: a window of a large
/// Whisper on the CPU is tens of seconds, and the engine checks its flag only
/// between them.
fn drain_transcribe(rx: &Receiver<Evt>, out: &tokio_mpsc::Sender<Heard>, stop: &AtomicBool, cancel: &AtomicBool) {
    loop {
        let heard = match rx.recv_timeout(std::time::Duration::from_millis(250)) {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if out.is_closed() || stop.load(Ordering::Relaxed) {
                    cancel.store(true, Ordering::Relaxed);
                }
                continue;
            }
            Ok(Evt::Transcribing(p)) => Heard::Window(p),
            Ok(Evt::Transcribed(t)) => Heard::Done(t),
            Ok(Evt::Error(e)) => Heard::Failed(e),
            Ok(Evt::Refused(e)) => Heard::Refused(e),
            Ok(Evt::Status(message)) => Heard::Failed(message),
            Ok(_) => continue,
            Err(_) => Heard::Failed("the engine thread has stopped".into()),
        };
        let last = !matches!(heard, Heard::Window(_));
        if out.blocking_send(heard).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
        if last {
            return;
        }
    }
}

/// Forward a video's progress, then the video, as [`drain_paint`] does an
/// image's.
///
/// Unlike an image's steps, a video's can be most of a minute apart, and
/// the engine checks the cancel flag only when it reports one. So the flag
/// is set as soon as the receiver is gone, not at the first event nobody
/// received: that would let the engine run one more step, which at
/// 1536×1024 is fifty seconds of a video nobody wants.
fn drain_film(rx: &Receiver<Evt>, out: &tokio_mpsc::Sender<Reel>, stop: &AtomicBool, cancel: &AtomicBool) {
    loop {
        let reel = match rx.recv_timeout(std::time::Duration::from_millis(250)) {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if out.is_closed() || stop.load(Ordering::Relaxed) {
                    cancel.store(true, Ordering::Relaxed);
                }
                continue;
            }
            Ok(Evt::Filming(step)) => Reel::Step(step),
            Ok(Evt::Filmed(f)) => Reel::Done(f),
            Ok(Evt::Error(e)) => Reel::Failed(e),
            Ok(Evt::Status(message)) => Reel::Failed(message),
            Ok(_) => continue,
            Err(_) => Reel::Failed("the engine thread has stopped".into()),
        };
        let last = !matches!(reel, Reel::Step(_));
        if out.blocking_send(reel).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
        if last {
            return;
        }
    }
}

/// Forward an image's steps, then the image.
///
/// A client that has gone away is noticed here rather than by the engine: the
/// engine only learns of it through the cancel flag, which is set on the
/// first step nobody received. Draining carries on to the engine's answer
/// either way, or the next job would read this one's leftovers.
fn drain_paint(rx: &Receiver<Evt>, out: &tokio_mpsc::Sender<Stroke>, stop: &AtomicBool, cancel: &AtomicBool) {
    loop {
        let evt = rx.recv();
        if stop.load(Ordering::Relaxed) {
            cancel.store(true, Ordering::Relaxed);
        }
        let stroke = match evt {
            Ok(Evt::Painting(step)) => Stroke::Step(step),
            Ok(Evt::Painted(p)) => Stroke::Done(Box::new(p)),
            Ok(Evt::Error(e)) => Stroke::Failed(e),
            Ok(Evt::Refused(e)) => Stroke::Refused(e),
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
        Fetch::Part { machine, layers, step } => {
            use kvad::weights::Part;
            let (state, message, file, bytes, total) = match step {
                Part::Asked => ("asked", None, None, 0, 0),
                Part::Said(line) => ("loading", Some(line), None, 0, 0),
                Part::Download { file, bytes, total } => ("loading", None, Some(file), bytes, total),
                Part::Fetched { file } => ("loading", Some(format!("fetched {file}")), None, 0, 0),
                Part::Ready { bytes } => ("ready", None, None, bytes, 0),
                Part::Failed(why) => ("failed", Some(why), None, 0, 0),
            };
            Some(Progress::Machine { machine, layers, state, message, file, bytes, total })
        }
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

    #[test]
    fn a_pipeline_is_an_image_a_video_or_a_speech_model_by_its_name() {
        assert_eq!(Kind::of_pipeline(None), Kind::Chat);
        assert_eq!(Kind::of_pipeline(Some("FluxPipeline")), Kind::Image);
        assert_eq!(Kind::of_pipeline(Some(kvad::video::LTX_PIPELINE)), Kind::Video);
        assert_eq!(Kind::of_pipeline(Some(kvad::speech::PIPELINE)), Kind::Speech);
    }

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
        let (_, mut pieces) = sched
            .chat(None, Entry::Request, &on, vec![Message::user("hello")], Vec::new(), Sampling::default(), false)
            .expect("the scheduler refused to queue the job");
        let piece = pieces.recv().await.expect("the stream ended with nothing in it");
        match piece {
            Piece::Failed(why) => assert_eq!(why, "a/b@cpu-q8 is not loaded"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(pieces.recv().await.is_none(), "the stream continued past its end");
        assert_eq!(sched.depth(), 0);
    }

    /// Stopping reaches the caller's generations and nobody else's, by id or
    /// all at once, and one that has ended is no longer there to stop.
    #[test]
    fn a_generation_is_stopped_only_by_its_owner() {
        let sched = refusing();
        let on = Key { repo: "a/b".into(), backend: Q8 };
        let (mine, theirs) =
            (sched.shared.admit(Some(1), Work::Chat, &on), sched.shared.admit(Some(2), Work::Chat, &on));
        assert_ne!(mine.id, theirs.id);
        let engine = Arc::new(AtomicBool::new(false));
        sched.shared.begin(mine.id, &engine);

        // Somebody else's id reads as no generation at all.
        assert_eq!(sched.cancel(Some(2), Some(mine.id)), 0);
        assert_eq!(sched.cancel(None, Some(mine.id)), 0);
        assert!(!mine.stop.load(Ordering::Relaxed) && !engine.load(Ordering::Relaxed));

        // Everything of one account's is that account's only.
        assert_eq!(sched.cancel(Some(2), None), 1);
        assert!(theirs.stop.load(Ordering::Relaxed));
        assert!(!mine.stop.load(Ordering::Relaxed) && !engine.load(Ordering::Relaxed));

        // The running one takes its engine's flag with it.
        assert_eq!(sched.cancel(Some(1), Some(mine.id)), 1);
        assert!(mine.stop.load(Ordering::Relaxed) && engine.load(Ordering::Relaxed));

        sched.shared.finish(mine.id);
        assert_eq!(sched.cancel(Some(1), Some(mine.id)), 0);
    }

    /// A job that holds the scheduler's thread until `Held` is let go: its
    /// answer has nowhere to go while the channel it would go to is full.
    struct Held(tokio_mpsc::Receiver<Piece>);

    impl Held {
        fn by(sched: &Scheduler, owner: Owner) -> Held {
            let on = Key { repo: "a/b".into(), backend: Q8 };
            let (out, rx) = tokio_mpsc::channel(1);
            out.try_send(Piece::Token("in the way".into())).unwrap();
            let ticket = sched.shared.admit(owner, Work::Chat, &on);
            let job = Job::Chat {
                ticket,
                on,
                messages: Vec::new(),
                tools: Vec::new(),
                sampling: Sampling::default(),
                fresh: false,
                out,
            };
            sched.submit(job).unwrap();
            // Taken off the queue, so that what waits behind it can be counted.
            while !sched.shared.busy.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            Held(rx)
        }

        async fn release(mut self) {
            while self.0.recv().await.is_some() {}
        }
    }

    fn ask(sched: &Scheduler, who: Owner, entry: Entry) -> Result<(u64, tokio_mpsc::Receiver<Piece>), Refused> {
        let on = Key { repo: "a/b".into(), backend: Q8 };
        sched.chat(who, entry, &on, vec![Message::user("hello")], Vec::new(), Sampling::default(), false)
    }

    /// A generation stopped while it waits leaves the queue at once and ends
    /// as cancelled, without waiting for its turn or the engine being asked.
    #[tokio::test]
    async fn a_waiting_generation_can_be_stopped() {
        let sched = refusing();
        let held = Held::by(&sched, Some(1));

        let (id, mut pieces) = ask(&sched, Some(2), Entry::Request).unwrap();
        assert_eq!(sched.cancel(Some(1), Some(id)), 0, "another account stopped it");
        assert_eq!(sched.cancel(Some(2), Some(id)), 1);

        // The job in front of it is still there: this did not wait its turn.
        match pieces.recv().await.expect("the stream ended with nothing in it") {
            // Not "a/b@cpu-q8 is not loaded": it never got as far as asking.
            Piece::Failed(why) => assert_eq!(why, CANCELLED),
            other => panic!("expected a cancellation, got {other:?}"),
        }
        assert!(pieces.recv().await.is_none(), "the stream continued past its end");
        assert_eq!(sched.cancel(Some(2), Some(id)), 0, "an ended generation was still there to stop");
        assert_eq!(sched.depth(), 1, "the stopped generation is still counted as waiting");
        held.release().await;
    }

    /// The queue has an end: the request after the last place is refused,
    /// with the numbers, and a place given back is a place again.
    #[tokio::test]
    async fn a_full_queue_refuses_the_next_request() {
        let sched = refusing().bounded(2);
        let held = Held::by(&sched, Some(1));

        let (first, _a) = ask(&sched, Some(1), Entry::Request).unwrap();
        let (_, mut b) = ask(&sched, Some(2), Entry::Request).unwrap();
        assert_eq!(sched.room(Some(3)), Err(Refused::Full { waiting: 2, limit: 2 }));
        assert_eq!(ask(&sched, Some(3), Entry::Request).unwrap_err(), Refused::Full { waiting: 2, limit: 2 });

        // Work already let in is not turned away at the door.
        let (_, mut more) = ask(&sched, Some(2), Entry::Continued).unwrap();
        assert!(ask(&sched, Some(3), Entry::Request).is_err());

        // Stopping two of the three leaves one, and so a place.
        assert_eq!(sched.cancel(Some(2), None), 2);
        assert!(matches!(b.recv().await, Some(Piece::Failed(why)) if why == CANCELLED));
        assert!(matches!(more.recv().await, Some(Piece::Failed(why)) if why == CANCELLED));
        let (last, _c) = ask(&sched, Some(3), Entry::Request).expect("a place given back was not a place");

        // The view, in the order they run: the one holding the engine, then
        // the two that wait, each with what is ahead of it. The later
        // arrival is first of those, because the earlier one's account is
        // the one that has the engine now.
        let view = sched.queue();
        assert_eq!(view.iter().map(|g| (g.id, g.running, g.ahead)).collect::<Vec<_>>(), [
            (view[0].id, true, 0),
            (last, false, 1),
            (first, false, 2),
        ]);
        assert_eq!(view.iter().map(|g| g.owner).collect::<Vec<_>>(), [Some(1), Some(3), Some(1)]);
        assert!(view.iter().all(|g| g.model == "a/b@cpu-q8" && g.kind == Work::Chat));
        assert!(view[0].running_secs.is_some() && view[1].running_secs.is_none());

        held.release().await;
    }

    /// Whose turn it is: the account served longest ago, by its earliest
    /// request, and nothing passes a job that is not a generation.
    #[test]
    fn the_next_generation_is_the_one_whose_account_has_waited_longest() {
        let (a, b, c) = (Some(Some(1)), Some(Some(2)), Some(Some(3)));
        let never = Served::new();
        assert_eq!(pick(&[], &never), None);
        // Nobody has had a turn: arrival order.
        assert_eq!(pick(&[a, a, b], &never), Some(0));
        // One account alone is arrival order however often it is served.
        assert_eq!(pick(&[a, a, a], &Served::from([(Some(1), 9)])), Some(0));
        // The account just served waits for the one that was not.
        assert_eq!(pick(&[a, a, b], &Served::from([(Some(1), 1)])), Some(2));
        // Of two that have been, the one longer ago; and its earliest request.
        assert_eq!(pick(&[a, b, c, c], &Served::from([(Some(1), 5), (Some(2), 4), (Some(3), 2)])), Some(2));
        // A load or an unload is not passed, and goes when it is at the front.
        assert_eq!(pick(&[a, a, None, b], &Served::from([(Some(1), 1)])), Some(0));
        assert_eq!(pick(&[None, b, a], &never), Some(0));
        // No accounts is one owner, and so arrival order.
        assert_eq!(pick(&[Some(None), Some(None)], &Served::from([(None, 3)])), Some(0));
    }

    /// Ten requests from one account do not hold up one from another: the
    /// order they run in takes turns.
    #[tokio::test]
    async fn accounts_take_turns_whatever_order_they_asked_in() {
        let sched = refusing();
        let held = Held::by(&sched, Some(9));
        let mut streams = Vec::new();
        let mut ask_as = |who: i64| {
            let (id, rx) = ask(&sched, Some(who), Entry::Request).unwrap();
            streams.push(rx);
            id
        };
        let (a1, a2, a3) = (ask_as(1), ask_as(1), ask_as(1));
        let (b1, b2) = (ask_as(2), ask_as(2));
        let c1 = ask_as(3);

        let waiting: Vec<u64> = sched.queue().into_iter().filter(|g| !g.running).map(|g| g.id).collect();
        assert_eq!(waiting, [a1, b1, c1, a2, b2, a3]);

        // And that is the order they are taken in: each ends as it is
        // reached, so the books empty in turn order.
        held.release().await;
        for mut stream in streams {
            while stream.recv().await.is_some() {}
        }
        assert!(sched.queue().is_empty());
    }

    /// One account may have its share waiting and no more; another account
    /// is not held to what the first has used, and with no accounts there
    /// is no share to keep to.
    #[tokio::test]
    async fn one_account_cannot_have_the_whole_queue() {
        let sched = refusing().bounded(8).sharing(2);
        let held = Held::by(&sched, Some(1));

        let _a = (ask(&sched, Some(1), Entry::Request).unwrap(), ask(&sched, Some(1), Entry::Request).unwrap());
        assert_eq!(ask(&sched, Some(1), Entry::Request).unwrap_err(), Refused::Yours { waiting: 2, limit: 2 });
        assert_eq!(sched.room(Some(1)), Err(Refused::Yours { waiting: 2, limit: 2 }));
        assert!(sched.room(Some(2)).is_ok());
        let _b = ask(&sched, Some(2), Entry::Request).unwrap();
        // The rest of a request already let in is not counted at the door.
        let _more = ask(&sched, Some(1), Entry::Continued).unwrap();

        let _local: Vec<_> = (0..4).map(|_| ask(&sched, None, Entry::Request).unwrap()).collect();
        // Eight wait now, and that is what the next is told, whoever it is.
        assert_eq!(ask(&sched, Some(2), Entry::Request).unwrap_err(), Refused::Full { waiting: 8, limit: 8 });

        held.release().await;
    }

    /// An account may ask so often and no more, is told how long until it
    /// may again, and is not held to what another has asked.
    #[tokio::test]
    async fn one_account_may_ask_only_so_often() {
        let sched = refusing().sharing(0).pacing(2);
        let _a = (ask(&sched, Some(1), Entry::Request).unwrap(), ask(&sched, Some(1), Entry::Request).unwrap());
        // Asking whether there is room is not asking for a generation.
        assert!(matches!(sched.room(Some(1)), Err(Refused::Hurried { limit: 2, retry_secs: 59..=60 })));
        assert!(matches!(
            ask(&sched, Some(1), Entry::Request).unwrap_err(),
            Refused::Hurried { limit: 2, retry_secs: 59..=60 }
        ));
        // The rest of a request already in is not another request.
        let _more = ask(&sched, Some(1), Entry::Continued).unwrap();
        // Another account has its own minute, and no account has none.
        let _b = ask(&sched, Some(2), Entry::Request).unwrap();
        let _local: Vec<_> = (0..5).map(|_| ask(&sched, None, Entry::Request).unwrap()).collect();
    }

    /// The wait is until the request that has to age out has: the oldest
    /// of the last `limit`, and what is over a minute old is forgotten.
    #[test]
    fn the_wait_is_until_the_oldest_of_the_minute_is_a_minute_old() {
        let start = Instant::now();
        let at = |secs: u64| start + std::time::Duration::from_secs(secs);
        let mut asked: VecDeque<Instant> = [at(0), at(10), at(50)].into();
        assert_eq!(wait_for(&mut asked, 4, at(55)), None);
        assert_eq!(wait_for(&mut asked, 3, at(55)), Some(5));
        // Two a minute, and three held: the second has to go, not the first.
        assert_eq!(wait_for(&mut asked, 2, at(55)), Some(15));
        // At the minute the first is forgotten, and there is room for one.
        assert_eq!(wait_for(&mut asked, 3, at(60)), None);
        assert_eq!(asked.len(), 2);
    }

    /// No limit is no limit.
    #[tokio::test]
    async fn a_limit_of_zero_lets_any_number_wait() {
        let sched = refusing().bounded(0);
        let held = Held::by(&sched, None);
        let waiting: Vec<_> = (0..50).map(|_| ask(&sched, None, Entry::Request).unwrap()).collect();
        assert_eq!(sched.queue().len(), 51);
        assert!(sched.room(None).is_ok());
        drop(waiting);
        held.release().await;
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
            let (_, mut pieces) = sched.complete(None, Entry::Request, &on, "the cat".into(), sampling, 0, true).unwrap();
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

    /// Two conversations taking turns on one model each find their own
    /// cache when they speak again, and say what they would have said alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn conversations_taking_turns_keep_their_caches() {
        let dir = crate::compare::tests::tiny_model("turns");
        let repo = dir.to_string_lossy().into_owned();
        let sched = Scheduler::spawn(kvad::service::cpu_loader, roomy());

        // One continuation: what was said, and what the cache had to offer.
        async fn say(sched: &Scheduler, on: &Key, prompt: &str, fresh: bool) -> (String, Stats) {
            let sampling = Sampling { temperature: 0.0, max_tokens: 3, ..Sampling::default() };
            let (_, mut pieces) = sched.complete(None, Entry::Request, on, prompt.into(), sampling, 0, fresh).unwrap();
            let (mut text, mut done) = (String::new(), None);
            while let Some(piece) = pieces.recv().await {
                match piece {
                    Piece::Token(t) => text.push_str(&t),
                    Piece::Done(stats) => done = Some(stats),
                    Piece::Failed(why) => panic!("{why}"),
                    Piece::Chose(_) => {}
                }
            }
            (text, done.expect("no end to the generation"))
        }

        for backend in [F32, Q8] {
            sched.load(repo.clone(), backend, nowhere()).await.unwrap();
            let on = Key { repo: repo.clone(), backend };

            // The tiny model reads characters, so a token is a character, and
            // its context is 32 of them: two conversations fit and three do
            // not.
            let (a, b) = ("the cat", "on the");
            let (a1, first) = say(&sched, &on, a, false).await;
            assert_eq!((first.cached_tokens, first.conversations), (0, 1));
            let (b1, other) = say(&sched, &on, b, false).await;
            assert_eq!(other.cached_tokens, 0, "{backend}: one conversation was read as another's prefix");
            assert_eq!(other.conversations, 2, "{backend}: the first conversation was not kept");
            assert_eq!(other.held_tokens, a.len() + b.len() + 6);

            // Each speaks again, and finds everything it said and was told.
            let a_next = format!("{a}{a1} s");
            let (a2, again) = say(&sched, &on, &a_next, false).await;
            assert_eq!(again.cached_tokens, a.len() + 3, "{backend}: the first conversation lost its cache");
            let b_next = format!("{b}{b1} m");
            let (b2, _) = say(&sched, &on, &b_next, false).await;

            // A third has no room beside both: the one that spoke longest
            // ago goes, and what is held stays inside the context.
            let (_, third) = say(&sched, &on, "sat on the mat", false).await;
            assert_eq!(third.cached_tokens, 0);
            assert!(third.held_tokens <= 32, "{backend}: {} tokens held of a context of 32", third.held_tokens);
            assert_eq!(third.conversations, 2, "{backend}: nothing was let go to make room");

            // With no cache at all, the same prompts get the same replies.
            assert_eq!(say(&sched, &on, &a_next, true).await.0, a2, "{backend}: a kept cache changed a reply");
            assert_eq!(say(&sched, &on, &b_next, true).await.0, b2, "{backend}: a kept cache changed a reply");

            sched.unload(Some(on)).await.unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Several replies on one model run together, and each says what it
    /// would have said alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replies_run_together_and_say_what_each_says_alone() {
        let dir = crate::compare::tests::tiny_model("together");
        let repo = dir.to_string_lossy().into_owned();
        let prompts = ["the cat", "on the", "sat on", "the mat"];

        async fn read(mut pieces: tokio_mpsc::Receiver<Piece>) -> (String, Stats) {
            let (mut text, mut done) = (String::new(), None);
            while let Some(piece) = pieces.recv().await {
                match piece {
                    Piece::Token(t) => text.push_str(&t),
                    Piece::Done(stats) => done = Some(stats),
                    Piece::Failed(why) => panic!("{why}"),
                    Piece::Chose(_) => {}
                }
            }
            (text, done.expect("no end to the generation"))
        }
        // Greedy, so that what is said depends on the prompt and nothing else.
        let sampling = Sampling { temperature: 0.0, max_tokens: 20, ..Sampling::default() };

        for backend in [F32, Q8] {
            let on = Key { repo: repo.clone(), backend };

            // What each says with the engine to itself: one at a time.
            let alone = Scheduler::spawn(kvad::service::cpu_loader, roomy()).batching(1);
            alone.load(repo.clone(), backend, nowhere()).await.unwrap();
            let mut want = Vec::new();
            for prompt in prompts {
                let (_, pieces) = alone.complete(None, Entry::Request, &on, prompt.into(), sampling, 0, false).unwrap();
                let (text, stats) = read(pieces).await;
                assert_eq!(stats.beside, 0, "{backend}: one at a time ran beside another");
                want.push(text);
            }
            drop(alone);

            // All four asked for before the scheduler looks: the thread is
            // held until they are queued, so they are there to join.
            let sched = Scheduler::spawn(kvad::service::cpu_loader, roomy()).bounded(0).batching(3);
            sched.load(repo.clone(), backend, nowhere()).await.unwrap();
            let held = Held::by(&sched, None);
            let streams: Vec<_> = prompts
                .iter()
                .map(|p| sched.complete(None, Entry::Request, &on, p.to_string(), sampling, 0, false).unwrap().1)
                .collect();
            held.release().await;

            let mut most = 0;
            for (stream, want) in streams.into_iter().zip(&want) {
                let (text, stats) = read(stream).await;
                assert_eq!(&text, want, "{backend}: a reply written beside others is not the one written alone");
                most = most.max(stats.beside);
            }
            // Three at once and never four: the fourth waits for a place.
            assert_eq!(most, 2, "{backend}: the batch was not three");
            assert!(sched.queue().is_empty());
            assert_eq!(sched.depth(), 0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Layers held for another machine's model take of the budget as a
    /// model of this server's does: they are refused when they do not fit,
    /// a model is refused for want of the room they have, and the room is
    /// back when they are let go.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn layers_held_for_another_machine_are_counted_against_this_ones_models() {
        let dir = crate::compare::tests::tiny_model("lodging");
        let repo = dir.to_string_lossy().into_owned();
        let probe = Scheduler::spawn(kvad::service::cpu_loader, roomy());
        probe.load(repo.clone(), F32, nowhere()).await.unwrap();
        let one = probe.residents()[0].commit;
        drop(probe);

        let sched = Scheduler::spawn(kvad::service::cpu_loader, Budget { total: one + 10, context: 1024 });
        let why = sched.lodge("layers 0..2 of a model, for another machine", one + 11).err().expect("refused");
        assert!(why.contains("layers 0..2 of a model") && why.contains("left of"), "{why}");
        assert!(sched.guests().is_empty());

        let lodged = sched.lodge("layers 0..2 of a model, for another machine", 4).unwrap();
        assert_eq!(sched.left(), one + 6);
        lodged.resize(11);
        assert_eq!((sched.left(), sched.guests()[0].commit), (one - 1, 11));

        // Not alone any more, so not admitted whatever it takes.
        let err = sched.load(repo.clone(), F32, nowhere()).await.unwrap_err();
        let LoadError::Full(why) = &err else { panic!("expected a refusal, got {err:?}") };
        assert!(why.contains("for another machine"), "who holds the rest is said: {why}");

        drop(lodged);
        assert_eq!(sched.left(), one + 10);
        sched.load(repo, F32, nowhere()).await.unwrap();
        // And the other way: a model in memory is room a guest does not have.
        assert!(sched.lodge("more layers", 11).is_err());
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
