//! The engine as a background service: commands in, events out.
//!
//! Everything interesting an engine does is slow. Downloading a model takes
//! tens of seconds, loading it takes more, and generation produces a token
//! every few tens of milliseconds. None of that can happen on the thread that
//! draws a screen or answers an HTTP request, or the caller freezes solid.
//!
//! So the engine lives on a thread of its own, and the two sides talk over a
//! pair of channels: [`Cmd`] in, [`Evt`] out. The caller never blocks — it
//! drains whatever events have arrived and goes back to what it was doing.
//!
//! Cancellation is the one thing a channel cannot express, because the worker
//! is busy inside `generate` and not reading its inbox. That uses a shared
//! [`AtomicBool`] which the per-token callback checks.
//!
//! # One model, one generation
//!
//! The worker holds at most one loaded model and runs one generation at a
//! time, because that is what the engine underneath it does. Commands queue
//! in the order they were sent. Continuous batching, when it lands, replaces
//! the inside of this module; the channels either side of it do not change.
//!
//! # Why the loader is the caller's
//!
//! `kvad` cannot build a GPU session: the GPU crate depends on this one, not
//! the other way round. So [`Backend::Gpu`] is a request this crate can
//! describe and not fulfil, and whoever spawns an engine hands it a
//! [`Loader`] that can. A build with no GPU crate in it passes
//! [`cpu_loader`], and asking that one for a GPU says what to do instead.
//!
//! # Kinds of model
//!
//! A loader hands back a [`Model`]: a language model, an image pipeline
//! ([`crate::image::Painter`]), a video pipeline
//! ([`crate::video::Director`]) or a Whisper
//! ([`crate::speech::Transcriber`]). They are peers, not modes of one another.
//! [`Llm`] knows nothing about images and a painter knows nothing about
//! tokens; what they share is this thread, the load and unload around them,
//! and the channel their progress goes down. Asking one for the other's work
//! is an [`Evt::Error`] that says which kind of model it is.

use crate::chat::Message;
use crate::hub::{self, HubModel, LocalModel};
use crate::image::{Defaults, ImageRequest, Painted, Painter, Step};
use crate::quant::Precision;
use crate::runtime::{Chosen, Llm, Perplexity, Run, Stats, Token};
use crate::sampler::Sampler;
use crate::video::{Director, Filmed, VideoRequest};
use crate::weights::{Fetch, Watcher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub enum Cmd {
    Search(String),
    Load { repo: String, backend: Backend },
    /// How many tokens of cache the loaded model may keep across every
    /// conversation: what whoever loaded it has charged it for. Not said,
    /// the model's own context. See [`crate::runtime::Llm::hold`].
    Hold(usize),
    /// Drop the loaded model, freeing its weights and its KV cache.
    ///
    /// Not the same as loading something else: a server needs to be able to
    /// give the memory back without being told what to spend it on next.
    Unload,
    /// Reply to a conversation, optionally with tools on offer.
    ///
    /// `tools` is a list of function schemas in OpenAI's shape, passed to the
    /// model's template untouched. A model whose template ignores them —
    /// [`crate::runtime::Llm::takes_tools`] — generates the same reply it
    /// would have without them, so a caller that means it checks first.
    ///
    /// `fresh` drops the KV cache first, as it does for [`Cmd::Complete`]. A
    /// conversation wants the opposite — its next turn is prefilled out of
    /// this one's cache — but an eval case is a first turn every time, and
    /// one served from the previous case's cache is not.
    Chat {
        messages: Vec<Message>,
        tools: Vec<serde_json::Value>,
        sampling: Sampling,
        fresh: bool,
    },
    /// Continue a prompt, with no chat template anywhere near it.
    ///
    /// The base-model view of a model, which for an instruct model is a
    /// different and more revealing thing than talking to it.
    Complete {
        prompt: String,
        sampling: Sampling,
        /// How many candidates to report per token; 0 for none.
        explain: usize,
        /// Drop the KV cache before starting.
        ///
        /// Matters for measurement rather than for output: the same prompt
        /// twice in a row would otherwise be prefilled from the first run's
        /// cache and report a time to first token that no first run would
        /// ever see. A benchmark sets this; a person typing does not.
        fresh: bool,
    },
    /// A [`Cmd::Chat`] or a [`Cmd::Complete`] that may run beside others.
    ///
    /// On their own those two are one generation at a time: the next
    /// command waits for it. Sent like this, a generation is let in while
    /// others sent like this are under way, and all of them are advanced
    /// together, a token each a step — in one pass over the model where the
    /// backend can ([`crate::model::Session::forward_each`]).
    ///
    /// Everything it has to say comes back as [`Evt::Of`] under `id`, since
    /// the channel is one and the generations are several. It stops when
    /// `stop` is raised, and nobody else's does.
    ///
    /// Where the backend keeps one conversation, it waits its turn instead.
    Beside { id: u64, stop: Arc<AtomicBool>, what: Box<Cmd> },
    /// How the loaded model's tokenizer splits a text.
    Tokenize(String),
    /// Score a text the model did not write.
    Perplexity { text: String, window: usize },
    RefreshLocal,
    Delete(String),
    /// Make an image, reporting each denoising step as [`Evt::Painting`] and
    /// ending with [`Evt::Painted`]. Only an image model can.
    Paint(ImageRequest),
    /// Make a video, reporting its progress as [`Evt::Filming`] and ending
    /// with [`Evt::Filmed`]. Only a video model can.
    Film(VideoRequest),
    /// Write down what is said in a recording, 16 kHz mono samples,
    /// reporting each 30-second window as [`Evt::Transcribing`] and ending
    /// with [`Evt::Transcribed`]. Only a Whisper can.
    Transcribe { audio: Vec<f32>, request: crate::speech::Request },
    /// One update of a live transcription ([`crate::speech::Live`]): the
    /// audio that has come since the last pushed into the session, and the
    /// window decoded again, or with `finish`, everything in it settled.
    /// `session` is `None` the first time, and one is begun for `request`.
    /// The session comes back with what the update said, as
    /// [`Evt::Listened`], to be sent again with the next. Only a Whisper can.
    Listen { session: Option<Box<crate::speech::Live>>, request: crate::speech::Request, audio: Vec<f32>, finish: bool },
}

/// What a [`Loader`] hands back.
pub enum Model {
    Text(Llm),
    Image(Box<dyn Painter>),
    Video(Box<dyn Director>),
    Speech(Box<crate::speech::Transcriber>),
}

impl From<Llm> for Model {
    fn from(llm: Llm) -> Self {
        Model::Text(llm)
    }
}

/// How to turn logits into tokens, chosen per request.
///
/// [`Sampler`] itself owns a random generator and so cannot be built by a
/// caller and sent across a channel; this is the part of it that is a
/// decision rather than state.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// `None` continues the loaded model's generator where the last request
    /// left it, so two identical requests differ. `Some(n)` restarts it, so
    /// two identical requests agree — which is what a benchmark, a test or a
    /// bug report needs, and what nobody wants by default.
    pub seed: Option<u64>,
    pub max_tokens: usize,
}

/// How many tokens a command line writes when `--max-tokens` does not say.
///
/// Two answers, because there are two kinds of writing. A reply ends
/// itself: an instruction-tuned model says its end-of-turn token when it is
/// done, so the budget is a ceiling that is rarely met, and gets the same
/// one a request to the server does. A continuation does not end: a base
/// model, or any model handed a bare prompt, writes until it is stopped, and
/// 4096 tokens of that is a screenful nobody asked for, and minutes of a
/// CPU. That keeps the 256 every command had until now.
pub fn token_budget(given: Option<usize>, continuing: bool) -> usize {
    match (given, continuing) {
        (Some(n), _) => n,
        (None, false) => Sampling::default().max_tokens,
        (None, true) => CONTINUATION_TOKENS,
    }
}

/// What a continuation is written to when nothing says how far.
pub const CONTINUATION_TOKENS: usize = 256;

impl Default for Sampling {
    fn default() -> Self {
        // Not 512, because of reasoning models. Qwen3 writes its working
        // before its answer, and on a grounded question that working alone
        // ran to about 650 tokens — so at 512 a request that names no limit
        // gets a trace and no answer. Measured: at 400 the same model spent
        // every token thinking and never reached one.
        //
        // And not 1024, which is room for the working and a short answer and
        // cuts a long one off in the middle. 4096 is what the web UI's chat
        // starts at too (`DEFAULTS` in web/src/lib/chat.svelte.js); a model
        // that is done sooner stops sooner, so the budget costs nothing
        // until it is needed.
        Sampling { temperature: 0.7, top_k: 40, top_p: 0.95, seed: None, max_tokens: 4096 }
    }
}

/// The generator a model starts with when nothing asks for a seed.
const DEFAULT_SEED: u64 = 7;

/// Where a model should run.
///
/// One knob covers both axes, because they are the same question — how much
/// hardware to spend — and the CPU's quantisation levels and the GPU's float
/// widths are just different answers to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu(Precision),
    /// The best GPU available, at this dtype.
    Gpu(GpuMode),
}

/// How the GPU should hold the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMode {
    /// Dense half precision — what the checkpoint ships as.
    Bf16,
    Q8,
    Q4,
}

impl Backend {
    pub const ALL: [Backend; 6] = [
        Backend::Cpu(Precision::F32),
        Backend::Cpu(Precision::Q8),
        Backend::Cpu(Precision::Q4),
        Backend::Gpu(GpuMode::Bf16),
        Backend::Gpu(GpuMode::Q8),
        Backend::Gpu(GpuMode::Q4),
    ];

    pub fn next(self) -> Backend {
        let i = Backend::ALL.iter().position(|b| *b == self).unwrap_or(0);
        Backend::ALL[(i + 1) % Backend::ALL.len()]
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu(p) => write!(f, "cpu {p}"),
            Backend::Gpu(GpuMode::Bf16) => f.write_str("gpu bf16"),
            Backend::Gpu(GpuMode::Q8) => f.write_str("gpu q8"),
            Backend::Gpu(GpuMode::Q4) => f.write_str("gpu q4"),
        }
    }
}

pub enum Evt {
    /// Transient progress text for a status bar.
    Status(String),
    /// The same progress, in numbers, for a progress bar.
    Fetching(Fetch),
    SearchResults(Vec<HubModel>),
    Local(Vec<LocalModel>),
    Loaded {
        repo: String,
        summary: String,
        params: usize,
        instruct: bool,
        /// Whether this model's template has anywhere to put tools, and so
        /// whether it can be asked to call one. See
        /// [`crate::runtime::Llm::takes_tools`].
        tools: bool,
        backend: String,
        weight_bytes: usize,
        /// The longest conversation this model can hold.
        n_ctx: usize,
        /// What one token of context costs in the KV cache.
        ///
        /// The cache is not pre-allocated — a 32k-context model would reserve
        /// gigabytes for a conversation of fifty tokens — so what it actually
        /// occupies is this times the tokens held, and what it *could* occupy
        /// is this times `n_ctx`. A caller reporting memory needs both.
        kv_bytes_per_token: usize,
        /// For an image model, what it does when a request leaves a knob
        /// out; `None` for a language model. This is how a caller tells the
        /// two apart.
        image: Option<Defaults>,
        /// For a video model, the same; `None` for any other.
        video: Option<crate::video::Defaults>,
        /// A Whisper, which has no knobs to report and is told apart by
        /// this alone.
        speech: bool,
    },
    /// No model is loaded any more, and what was loaded is named.
    Unloaded(String),
    /// A fragment of the assistant's reply.
    Token(String),
    /// The same fragment, with the candidates it was chosen from. Sent
    /// instead of [`Evt::Token`] when a request asked to be shown the
    /// choice.
    Chose(Chosen),
    Done(Stats),
    Tokens(Vec<Token>),
    /// Progress through a text being scored, in tokens.
    Scoring { done: usize, total: usize },
    Scored(Perplexity),
    /// One denoising step of an image, done.
    Painting(Step),
    Painted(Painted),
    /// Where a video has got to.
    Filming(crate::video::Step),
    /// Boxed, because a video is the frames themselves, and every other
    /// event would otherwise be sent at its size.
    Filmed(Box<Filmed>),
    /// A window of a recording, done, and the stretches of text it added.
    Transcribing(crate::speech::Progress),
    Transcribed(Box<crate::speech::Transcript>),
    /// A live transcription's update, and the session to send with the
    /// next.
    Listened(Box<crate::speech::Live>, crate::speech::Update),
    Error(String),
    /// Something a generation sent with [`Cmd::Beside`] has to say — a
    /// token, its end, its failure — with the id it was sent under.
    Of(u64, Box<Evt>),
    /// A request refused by what served it, for the asker to change:
    /// [`crate::image::Refused`].
    Refused(String),
}

/// Turns a repo id and a [`Backend`] into a loaded model.
///
/// See the module docs for why this is the caller's to supply. `progress` is
/// the line of text a status bar shows; `watch` is the same thing in bytes,
/// and both are already wired to the event channel by the time a loader sees
/// them.
pub type Loader = Box<
    dyn FnMut(&str, Backend, &mut dyn FnMut(&str), &Watcher) -> Res<Model> + Send,
>;

/// The loader for a build with no GPU crate in it.
///
/// CPU backends load; a GPU one is refused with the key that changes it,
/// rather than with a type error at the other end of the program.
///
/// A model trained by `nervus`'s `train_digits` or `train_video` is the
/// exception to both: it makes pictures or clips, and it runs on the CPU
/// whatever it is asked for ([`crate::dit`]).
pub fn cpu_loader() -> Loader {
    Box::new(|repo, backend, progress, watch| match backend {
        _ if crate::dit::is_one(repo) => Ok(Model::Image(Box::new(crate::dit::Drawer::load(repo, progress)?))),
        _ if crate::dit::is_clips(repo) => Ok(Model::Video(Box::new(crate::dit::Filmer::load(repo, progress)?))),
        Backend::Cpu(precision) if crate::speech::is_whisper(repo) => {
            Ok(Model::Speech(Box::new(crate::speech::Transcriber::load(repo, precision, progress, watch)?)))
        }
        // Before the GPU's refusal below, which would not say that a CPU
        // backend is there for it.
        Backend::Gpu(_) if crate::speech::is_whisper(repo) => Err(WHISPER_ON_CPU.into()),
        Backend::Cpu(precision) => Llm::load_watched(repo, precision, progress, watch).map(Model::from),
        Backend::Gpu(_) => {
            Err("this build has no GPU backend; pick a CPU precision instead".into())
        }
    })
}

/// What a Whisper asked for at a GPU backend is told by a build with none.
const WHISPER_ON_CPU: &str =
    "this build has no GPU backend; load the Whisper at a cpu backend (cpu-f32, cpu-q8 or cpu-q4)";

pub struct Engine {
    tx: Sender<Cmd>,
    pub rx: Receiver<Evt>,
    /// Set by the caller to interrupt generation mid-stream.
    pub cancel: Arc<AtomicBool>,
}

impl Engine {
    pub fn spawn(loader: Loader) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let (evt_tx, evt_rx) = mpsc::channel::<Evt>();
        let cancel = Arc::new(AtomicBool::new(false));

        let worker_cancel = Arc::clone(&cancel);
        std::thread::Builder::new()
            .name("kvad-engine".into())
            // Generation recurses through 30 layers of closures and rayon
            // scopes; the default 2 MB is enough, but be explicit.
            .stack_size(8 * 1024 * 1024)
            .spawn(move || worker(cmd_rx, evt_tx, worker_cancel, loader))
            .expect("failed to spawn engine thread");

        Engine { tx: cmd_tx, rx: evt_rx, cancel }
    }

    pub fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// What one token of context costs in the KV cache, per layer, at four bytes
/// a float.
///
/// The same arithmetic as [`crate::model::KvCache::max_bytes`], divided by the
/// context length — stated here so that a caller can multiply it by the tokens
/// actually held rather than by the ones that might be.
///
/// It reads [`Spec::cache`], which is what the cache actually stores, and not
/// `kv_dim()`, which is what ordinary attention would have stored. Those agree
/// for GPT-2 and the Llama family and disagree by a factor of seven for
/// DeepSeek: MLA declares `n_kv_head == n_head` because every head really does
/// have its own key, and then keeps none of them — 576 floats a position
/// against the 2048 `kv_dim()` describes. Saying so made this function claim
/// seven times the memory the engine was using.
///
/// A recurrent state is deliberately not in here. It is a cost *per layer*,
/// not per token, so no per-token figure can carry it; see
/// [`crate::model::CacheShape`].
///
/// `number` is how wide one cached number is,
/// [`crate::model::Session::kv_number_bytes`]:
/// 4 on the CPU, often 2 on a GPU.
fn kvad_kv_bytes_per_token(spec: &crate::model::Spec, number: usize) -> usize {
    spec.cache.bytes_per_token_as(spec.n_layer, number)
}

/// The model currently loaded, plus its sampler.
struct Loaded {
    llm: Llm,
    sampler: Sampler,
}

/// Whichever kind of model this engine holds.
enum Held {
    Text(Loaded),
    Image { repo: String, painter: Box<dyn Painter> },
    Video { repo: String, director: Box<dyn Director> },
    Speech { repo: String, transcriber: Box<crate::speech::Transcriber> },
}

impl Held {
    fn repo(&self) -> &str {
        match self {
            Held::Text(s) => &s.llm.repo,
            Held::Image { repo, .. } | Held::Video { repo, .. } | Held::Speech { repo, .. } => repo,
        }
    }

    /// What this model says when asked to `what` and it cannot: where to
    /// ask it instead.
    fn cannot(&self, what: &str) -> String {
        match self {
            Held::Text(s) => format!("{} is a language model; it cannot {what}", s.llm.repo),
            Held::Image { repo, .. } => format!("{repo} makes images; ask it at /v1/images/generations"),
            Held::Video { repo, .. } => format!("{repo} makes videos; ask it at /v1/videos"),
            Held::Speech { repo, .. } => format!("{repo} transcribes speech; ask it at /v1/audio/transcriptions"),
        }
    }

    /// The language model, or an error saying this is not one.
    fn text(&mut self) -> Result<&mut Loaded, String> {
        match self {
            Held::Text(s) => Ok(s),
            Held::Image { repo, .. } => Err(format!(
                "{repo} makes images from text; it cannot chat, complete, tokenize or score. \
                 Ask it for an image instead (POST /v1/images/generations)."
            )),
            Held::Video { repo, .. } => Err(format!(
                "{repo} makes videos from text; it cannot chat, complete, tokenize or score. \
                 Ask it for a video instead (POST /v1/videos)."
            )),
            Held::Speech { repo, .. } => Err(format!(
                "{repo} transcribes speech; it cannot chat, complete, tokenize or score. \
                 Send it a recording instead (POST /v1/audio/transcriptions)."
            )),
        }
    }
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Evt>, cancel: Arc<AtomicBool>, mut load: Loader) {
    let say = |msg: &str| {
        let _ = tx.send(Evt::Status(msg.to_string()));
    };
    let fail = |e: Box<dyn std::error::Error>| {
        let _ = tx.send(Evt::Error(e.to_string()));
    };

    let mut session: Option<Held> = None;
    // The generations under way. More than one only when every one of them
    // was sent with `Cmd::Beside`.
    let mut generating: Vec<Generating> = Vec::new();
    let _ = tx.send(Evt::Local(hub::local_models()));

    // A step loop, not a command loop: each time round it either advances
    // the generation by a token or, with none under way, waits for a
    // command. A generation used to be one call that kept the thread until
    // it ended; in steps, the thread comes back between tokens, which is
    // where a second generation will be let in.
    //
    //
    // A generation sent with `Cmd::Beside` may be joined by others sent the
    // same way: between steps the channel is looked at without waiting, and
    // one that can join is prefilled and joins. Anything else that arrives
    // — a load, a plain chat, a generation the backend has no room to run
    // beside the rest — is kept in `held`, and nothing more is heard until
    // the generations have ended: commands are still taken in order.
    let mut held: Option<Cmd> = None;
    loop {
        let cmd = match generating.is_empty() {
            true => match held.take() {
                Some(cmd) => cmd,
                None => match rx.recv() {
                    Ok(cmd) => cmd,
                    Err(_) => break,
                },
            },
            false => {
                // What was held is asked again each time round: a generation
                // that could not join while another was reading its prompt
                // can once that one has.
                let joining = match held.take() {
                    Some(cmd) => Some(cmd),
                    None => match rx.try_recv() {
                        Ok(cmd) => Some(cmd),
                        Err(mpsc::TryRecvError::Empty) => None,
                        Err(mpsc::TryRecvError::Disconnected) => break,
                    },
                };
                let Some(Ok(s)) = session.as_mut().map(Held::text) else {
                    // The model went from under them, which nothing here does.
                    generating.clear();
                    continue;
                };
                match joining {
                    Some(cmd) if can_join(&cmd, &mut generating, s) => cmd,
                    waits => {
                        held = waits;
                        advance(&mut generating, s, &tx, &cancel);
                        continue;
                    }
                }
            }
        };
        // A generation sent to run beside others answers under its id.
        let (tag, cmd) = match cmd {
            Cmd::Beside { id, stop, what } => (Some(Tag { id, stop }), *what),
            alone => (None, alone),
        };
        match cmd {
            Cmd::RefreshLocal => {
                let _ = tx.send(Evt::Local(hub::local_models()));
            }

            Cmd::Search(query) => {
                say(&format!("searching for “{query}”"));
                match hub::search(&query, 40) {
                    Ok(results) => {
                        let _ = tx.send(Evt::Status(format!("{} results", results.len())));
                        let _ = tx.send(Evt::SearchResults(results));
                    }
                    Err(e) => fail(e),
                }
            }

            Cmd::Delete(id) => match hub::find_local(&id) {
                Some(local) => match hub::remove(&local) {
                    Ok(()) => {
                        if hub::State::active().as_deref() == Some(id.as_str()) {
                            let _ = hub::State::clear();
                        }
                        say(&format!("deleted {id}"));
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Err(e) => fail(Box::new(e)),
                },
                None => say(&format!("{id} is not in the cache")),
            },

            Cmd::Unload => match session.take() {
                Some(was) => {
                    let _ = tx.send(Evt::Unloaded(was.repo().to_string()));
                }
                None => say("nothing is loaded"),
            },

            Cmd::Load { repo, backend } => {
                // Drop the previous model before loading the next one, or two
                // sets of weights are briefly resident at once.
                session = None;
                say(&format!("loading {repo}"));

                let mut progress = |msg: &str| {
                    let _ = tx.send(Evt::Status(format!("{repo}: {msg}")));
                };
                // `hf-hub` reports bytes from its own download threads, so the
                // watcher forwards down the channel rather than touching
                // anything on this one.
                let watch = {
                    let tx = tx.clone();
                    Watcher::new(move |f: Fetch| {
                        let _ = tx.send(Evt::Fetching(f));
                    })
                };
                match load(&repo, backend, &mut progress, &watch) {
                    Ok(Model::Image(painter)) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: painter.summary(),
                            params: painter.params(),
                            instruct: false,
                            tools: false,
                            backend: painter.backend(),
                            weight_bytes: painter.weight_bytes(),
                            n_ctx: 0,
                            kv_bytes_per_token: 0,
                            image: Some(painter.defaults()),
                            video: None,
                            speech: false,
                        });
                        // Not made the active model: that is what `kvad run`
                        // talks to when it is not told, and it cannot talk
                        // to this.
                        session = Some(Held::Image { repo, painter });
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Ok(Model::Video(director)) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: director.summary(),
                            params: director.params(),
                            instruct: false,
                            tools: false,
                            backend: director.backend(),
                            weight_bytes: director.weight_bytes(),
                            n_ctx: 0,
                            kv_bytes_per_token: 0,
                            image: None,
                            video: Some(director.defaults()),
                            speech: false,
                        });
                        // Not the active model either, for the same reason.
                        session = Some(Held::Video { repo, director });
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Ok(Model::Speech(transcriber)) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: transcriber.summary(),
                            params: transcriber.model.params(),
                            instruct: false,
                            tools: false,
                            backend: transcriber.backend(),
                            weight_bytes: transcriber.model.weight_bytes(),
                            n_ctx: 0,
                            kv_bytes_per_token: 0,
                            image: None,
                            video: None,
                            speech: true,
                        });
                        // Not the active model either: `kvad run` cannot
                        // talk to it.
                        session = Some(Held::Speech { repo, transcriber });
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Ok(Model::Text(llm)) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: llm.spec.summary(),
                            params: llm.param_count,
                            instruct: llm.is_instruct(),
                            tools: llm.takes_tools(),
                            backend: llm.backend(),
                            weight_bytes: llm.weight_bytes,
                            n_ctx: llm.spec.n_ctx,
                            kv_bytes_per_token: kvad_kv_bytes_per_token(
                                &llm.spec,
                                llm.session.kv_number_bytes(),
                            ),
                            image: None,
                            video: None,
                            speech: false,
                        });
                        let _ = hub::State::set_active(&repo);
                        let d = Sampling::default();
                        let sampler =
                            Sampler::new(d.temperature, d.top_k, d.top_p, DEFAULT_SEED);
                        session = Some(Held::Text(Loaded { llm, sampler }));
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Err(e) => fail(e),
                }
            }

            Cmd::Hold(tokens) => {
                if let Some(Ok(s)) = session.as_mut().map(|held| held.text()) {
                    s.llm.hold(tokens);
                }
            }

            Cmd::Chat { messages, tools, sampling, fresh } => {
                let s = match session.as_mut().map(Held::text) {
                    None => {
                        tell(&tx, &tag, Evt::Status("no model loaded".into()));
                        continue;
                    }
                    Some(Err(e)) => {
                        tell(&tx, &tag, Evt::Error(e));
                        continue;
                    }
                    Some(Ok(s)) => s,
                };
                let ids = match fresh.then(|| s.llm.fresh()).transpose().and_then(|_| s.llm.encode_chat(&messages, &tools)) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tell(&tx, &tag, Evt::Error(e.to_string()));
                        continue;
                    }
                };
                // The session reuses whatever of the previous turn's cache
                // still matches, so a chat prefills only the newest message.
                generating.extend(begin(s, &ids, &sampling, 0, tag, &tx, &cancel));
            }

            Cmd::Complete { prompt, sampling, explain, fresh } => {
                let s = match session.as_mut().map(Held::text) {
                    None => {
                        tell(&tx, &tag, Evt::Status("no model loaded".into()));
                        continue;
                    }
                    Some(Err(e)) => {
                        tell(&tx, &tag, Evt::Error(e));
                        continue;
                    }
                    Some(Ok(s)) => s,
                };
                // No template: the prompt is the text, which is what makes
                // this a different view of the model rather than a second
                // chat window.
                let ids = match fresh.then(|| s.llm.fresh()).transpose().and_then(|_| s.llm.encode(&prompt)) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tell(&tx, &tag, Evt::Error(e.to_string()));
                        continue;
                    }
                };
                generating.extend(begin(s, &ids, &sampling, explain, tag, &tx, &cancel));
            }

            // Unwrapped above; one inside another is nothing anybody sends.
            Cmd::Beside { .. } => {
                tell(&tx, &tag, Evt::Error("a generation cannot be sent beside itself".into()));
            }

            Cmd::Tokenize(text) => match session.as_mut().map(Held::text) {
                None => say("no model loaded"),
                Some(Err(e)) => fail(e.into()),
                Some(Ok(s)) => match s.llm.tokenize(&text) {
                    Ok(tokens) => {
                        let _ = tx.send(Evt::Tokens(tokens));
                    }
                    Err(e) => fail(e),
                },
            },

            Cmd::Perplexity { text, window } => {
                let Some(held) = session.as_mut() else {
                    say("no model loaded");
                    continue;
                };
                let s = match held.text() {
                    Ok(s) => s,
                    Err(e) => {
                        fail(e.into());
                        continue;
                    }
                };
                cancel.store(false, Ordering::Relaxed);
                let tx2 = tx.clone();
                let cancel2 = Arc::clone(&cancel);
                let scored = s.llm.perplexity(&text, window, |done, total| {
                    let _ = tx2.send(Evt::Scoring { done, total });
                    !cancel2.load(Ordering::Relaxed)
                });
                match scored {
                    Ok(p) => {
                        let _ = tx.send(Evt::Scored(p));
                    }
                    Err(e) => fail(e),
                }
            }

            Cmd::Paint(request) => {
                let painter = match session.as_mut() {
                    None => {
                        say("no model loaded");
                        continue;
                    }
                    Some(Held::Image { painter, .. }) => painter,
                    Some(other) => {
                        fail(other.cannot("make images").into());
                        continue;
                    }
                };
                cancel.store(false, Ordering::Relaxed);
                let tx2 = tx.clone();
                let cancel2 = Arc::clone(&cancel);
                let painted = painter.paint(&request, &mut |step| {
                    tx2.send(Evt::Painting(step)).is_ok() && !cancel2.load(Ordering::Relaxed)
                });
                match painted {
                    Ok(p) => {
                        let _ = tx.send(Evt::Painted(p));
                    }
                    Err(e) => match e.downcast_ref::<crate::image::Refused>() {
                        Some(r) => {
                            let _ = tx.send(Evt::Refused(r.0.clone()));
                        }
                        None => fail(e),
                    },
                }
            }

            Cmd::Film(request) => {
                let director = match session.as_mut() {
                    None => {
                        say("no model loaded");
                        continue;
                    }
                    Some(Held::Video { director, .. }) => director,
                    Some(other) => {
                        fail(other.cannot("make videos").into());
                        continue;
                    }
                };
                cancel.store(false, Ordering::Relaxed);
                let tx2 = tx.clone();
                let cancel2 = Arc::clone(&cancel);
                let filmed = director.film(&request, &mut |step| {
                    tx2.send(Evt::Filming(step)).is_ok() && !cancel2.load(Ordering::Relaxed)
                });
                match filmed {
                    Ok(f) => {
                        let _ = tx.send(Evt::Filmed(Box::new(f)));
                    }
                    Err(e) => fail(e),
                }
            }

            Cmd::Transcribe { audio, request } => {
                let transcriber = match session.as_mut() {
                    None => {
                        say("no model loaded");
                        continue;
                    }
                    Some(Held::Speech { transcriber, .. }) => transcriber,
                    Some(other) => {
                        fail(other.cannot("transcribe speech").into());
                        continue;
                    }
                };
                cancel.store(false, Ordering::Relaxed);
                let tx2 = tx.clone();
                let cancel2 = Arc::clone(&cancel);
                let heard = transcriber.transcribe(&audio, &request, &mut |p| {
                    tx2.send(Evt::Transcribing(p.clone())).is_ok() && !cancel2.load(Ordering::Relaxed)
                });
                match heard {
                    Ok(t) => {
                        let _ = tx.send(Evt::Transcribed(Box::new(t)));
                    }
                    Err(e) => match e.downcast_ref::<crate::image::Refused>() {
                        Some(r) => {
                            let _ = tx.send(Evt::Refused(r.0.clone()));
                        }
                        None => fail(e),
                    },
                }
            }

            Cmd::Listen { session: live, request, audio, finish } => {
                let transcriber = match session.as_mut() {
                    None => {
                        say("no model loaded");
                        continue;
                    }
                    Some(Held::Speech { transcriber, .. }) => transcriber,
                    Some(other) => {
                        fail(other.cannot("transcribe speech").into());
                        continue;
                    }
                };
                let listened = (|| -> Res<_> {
                    let mut live = match live {
                        Some(l) => l,
                        None => Box::new(crate::speech::Live::new(transcriber, &request)?),
                    };
                    live.push(&audio);
                    let update = if finish { live.finish(transcriber)? } else { live.update(transcriber)? };
                    Ok((live, update))
                })();
                match listened {
                    Ok((live, update)) => {
                        let _ = tx.send(Evt::Listened(live, update));
                    }
                    Err(e) => match e.downcast_ref::<crate::image::Refused>() {
                        Some(r) => {
                            let _ = tx.send(Evt::Refused(r.0.clone()));
                        }
                        None => fail(e),
                    },
                }
            }
        }
    }
}

/// What a generation sent with [`Cmd::Beside`] is known by.
struct Tag {
    id: u64,
    stop: Arc<AtomicBool>,
}

/// Send `evt`, under the generation's id if it has one.
fn tell(tx: &Sender<Evt>, tag: &Option<Tag>, evt: Evt) -> bool {
    let evt = match tag {
        Some(tag) => Evt::Of(tag.id, Box::new(evt)),
        None => evt,
    };
    tx.send(evt).is_ok()
}

/// A generation between its steps.
struct Generating {
    run: Run,
    /// How many candidates each token is reported with; none sends the
    /// token's text alone.
    explain: usize,
    /// Its own, because several generations cannot share one stream of
    /// draws; see [`Sampler::for_one`].
    sampler: Sampler,
    tag: Option<Tag>,
    /// The most others it has run beside so far.
    beside: usize,
}

/// Whether `cmd` may start beside the generations under way, having made
/// room for it if so.
///
/// It may when it and all of them were sent with [`Cmd::Beside`], and the
/// one whose conversation is in the session can carry it instead — which a
/// backend that keeps one conversation cannot, and then the answer is no and
/// the command waits, as every command did before.
fn can_join(cmd: &Cmd, generating: &mut [Generating], s: &mut Loaded) -> bool {
    matches!(cmd, Cmd::Beside { .. })
        && generating.iter().all(|g| g.tag.is_some())
        // One prompt is read at a time, in the session. The next waits for
        // it, and is asked again when it is done.
        && generating.iter().all(|g| g.run.ready())
        && generating.iter_mut().all(|g| s.llm.detach(&mut g.run))
}

/// Start generating from token ids: the conversation found, and its prompt
/// still to read — [`advance`] does that, a pass at a time when there are
/// others to keep writing in between.
///
/// Shared by chat and raw completion, which differ only in how the ids were
/// arrived at — and that difference belongs above this line, not inside the
/// loop. `None` is a generation that could not start, which has been said.
fn begin(
    s: &mut Loaded,
    ids: &[u32],
    sampling: &Sampling,
    explain: usize,
    tag: Option<Tag>,
    tx: &Sender<Evt>,
    cancel: &Arc<AtomicBool>,
) -> Option<Generating> {
    // The engine's own flag is for a generation running alone, whose
    // caller has nothing else to stop it by. One with a tag has its own.
    if tag.is_none() {
        cancel.store(false, Ordering::Relaxed);
    }
    let sampler = s.sampler.for_one(sampling.temperature, sampling.top_k, sampling.top_p, sampling.seed);
    match s.llm.open(ids, sampling.max_tokens, explain) {
        Ok(run) => Some(Generating { run, explain, sampler, tag, beside: 0 }),
        Err(e) => {
            tell(tx, &tag, Evt::Error(e.to_string()));
            None
        }
    }
}

/// Take every generation one token further and send each its token. The
/// ones that are over leave, with their statistics or their failure sent.
///
/// A generation still reading its prompt reads on instead: all of it when
/// there is nobody else, and one pass when there is, so that the others
/// write a token between each pass rather than waiting for the whole
/// prompt. That is chunked prefill, and it is what keeps a long prompt
/// arriving from stopping every reply under way for as long as it takes to
/// read.
fn advance(generating: &mut Vec<Generating>, s: &mut Loaded, tx: &Sender<Evt>, cancel: &AtomicBool) {
    let (writing, reading): (Vec<Generating>, Vec<Generating>) =
        generating.drain(..).partition(|g| g.run.ready());
    let alone = writing.is_empty();
    for mut g in reading {
        let stopped = match &g.tag {
            Some(tag) => tag.stop.load(Ordering::Relaxed),
            None => cancel.load(Ordering::Relaxed),
        };
        match stopped {
            // Stopped before its prompt was read: over, with nothing said.
            true => {
                let (stats, _) = s.llm.finish(g.run);
                tell(tx, &g.tag, Evt::Done(stats));
            }
            false => match s.llm.prefill(&mut g.run, alone) {
                // Read or not, it writes nothing this time round: its first
                // token is the next step's.
                Ok(_) => generating.push(g),
                Err(e) => {
                    tell(tx, &g.tag, Evt::Error(e.to_string()));
                    s.llm.abandon(g.run);
                }
            },
        }
    }
    if writing.is_empty() {
        return;
    }
    let reading = std::mem::replace(generating, writing);
    step(generating, s, tx, cancel);
    generating.extend(reading);
}

/// Take every generation that has read its prompt one token further.
fn step(generating: &mut Vec<Generating>, s: &mut Loaded, tx: &Sender<Evt>, cancel: &AtomicBool) {
    // One alone, with its conversation in the session, is the step it
    // always was; several are one pass. A failure is the pass's, and so
    // every generation's that was in it.
    let others = generating.len().saturating_sub(1);
    generating.iter_mut().for_each(|g| g.beside = g.beside.max(others));
    let chosen = match generating.as_mut_slice() {
        [alone] => s.llm.step(&mut alone.run, &mut alone.sampler).map(|chosen| vec![chosen]),
        several => {
            // The one that began last still has its conversation in the
            // session; it carries it from here, like the rest. `can_join`
            // found the backend able to, or there would not be several.
            for g in several.iter_mut() {
                s.llm.detach(&mut g.run);
            }
            let (mut runs, mut samplers): (Vec<&mut Run>, Vec<&mut Sampler>) =
                several.iter_mut().map(|g| (&mut g.run, &mut g.sampler)).unzip();
            s.llm.step_each(&mut runs, &mut samplers)
        }
    };
    let chosen = match chosen {
        Ok(chosen) => chosen,
        Err(e) => {
            for g in generating.drain(..) {
                tell(tx, &g.tag, Evt::Error(e.to_string()));
                s.llm.abandon(g.run);
            }
            return;
        }
    };

    let mut still = Vec::with_capacity(generating.len());
    for (g, chosen) in generating.drain(..).zip(chosen) {
        let stopped = match &g.tag {
            Some(tag) => tag.stop.load(Ordering::Relaxed),
            None => cancel.load(Ordering::Relaxed),
        };
        let over = match chosen {
            None => true,
            // Asked to stop: the token is not sent. Nobody listening: the same.
            Some(_) if stopped => true,
            Some(chosen) => {
                let evt = match g.explain {
                    0 => Evt::Token(chosen.text),
                    _ => Evt::Chose(chosen),
                };
                !tell(tx, &g.tag, evt)
            }
        };
        match over {
            false => still.push(g),
            true => {
                let (mut stats, _) = s.llm.finish(g.run);
                stats.beside = g.beside;
                tell(tx, &g.tag, Evt::Done(stats));
            }
        }
    }
    *generating = still;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply gets the server's own budget, a continuation the short one,
    /// and `--max-tokens` is taken at its word either way.
    #[test]
    fn a_reply_and_a_continuation_are_written_to_different_lengths() {
        assert_eq!(token_budget(None, false), Sampling::default().max_tokens);
        assert_eq!(token_budget(None, false), 4096);
        assert_eq!(token_budget(None, true), 256);
        assert_eq!(token_budget(Some(12), false), 12);
        assert_eq!(token_budget(Some(9000), true), 9000);
    }

    /// The per-token figure and the cache's own total have to be the same
    /// arithmetic, because the doc comment on one of them says they are.
    ///
    /// They were not. `kv_bytes_per_token` read `kv_dim()` — what ordinary
    /// attention *would* store — and DeepSeek stores something else entirely:
    /// MLA sets `n_kv_head = n_head` because every head really does have its
    /// own key, then keeps none of them, caching 576 floats a position where
    /// `kv_dim()` says 2048. The server reported 7.1x the memory it was
    /// using, on the one architecture whose whole argument is that it uses
    /// less.
    #[test]
    fn the_per_token_cost_agrees_with_the_cache_it_describes() {
        use crate::model::{CacheLayout, CacheShape, KvCache};

        let mut spec = tiny_spec();

        for (what, cache, n_kv_head, head_dim) in [
            // Llama: what is cached is what `kv_dim()` describes.
            ("llama", CacheShape::kv(256, 256), 2, 128),
            // DeepSeek: 64 rotated + 512 latent, against a `kv_dim()` of 2048.
            ("deepseek", CacheShape::kv(64, 512), 16, 128),
        ] {
            spec.cache = CacheLayout::uniform(cache);
            spec.n_kv_head = n_kv_head;
            spec.head_dim = head_dim;
            assert_eq!(
                kvad_kv_bytes_per_token(&spec, 4) * spec.n_ctx,
                KvCache::max_bytes(&spec),
                "{what}: the per-token figure does not scale to the cache's own total"
            );
        }
    }

    /// A spec with nothing architecture-specific in it, for the arithmetic
    /// above.
    fn tiny_spec() -> crate::model::Spec {
        crate::model::Spec {
            arch: crate::model::Arch::require("llama"),
            n_layer: 4,
            n_head: 4,
            n_kv_head: 2,
            n_embd: 256,
            head_dim: 128,
            n_ctx: 1024,
            vocab_size: 32,
            intermediate: 512,
            eps: 1e-5,
            rope_theta: 10000.0,
            tie_embeddings: true,
            cache: crate::model::CacheLayout::uniform(crate::model::CacheShape::kv(256, 256)),
            config: crate::model::Json::default(),
        }
    }

    /// An engine that can load nothing, for the tests that are about the
    /// channels either side of the loader rather than about loading.
    fn refusing_loader(why: &'static str) -> Loader {
        Box::new(move |_, _, _, _| Err(why.into()))
    }

    /// An engine answers on the channel rather than on the calling thread,
    /// and keeps answering after a load fails.
    #[test]
    fn a_failed_load_is_an_event_and_not_the_end_of_the_engine() {
        let engine = Engine::spawn(refusing_loader("no backend here"));
        engine.send(Cmd::Load {
            repo: "nobody/nothing".into(),
            backend: Backend::Cpu(Precision::Q8),
        });
        engine.send(Cmd::Unload);

        let mut errors = Vec::new();
        let mut statuses = Vec::new();
        // Local(..) first, then the load's status, its error, and the
        // "nothing is loaded" the unload answers with.
        for _ in 0..8 {
            match engine.rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Evt::Error(e)) => errors.push(e),
                Ok(Evt::Status(s)) => statuses.push(s),
                Ok(_) => {}
                Err(_) => break,
            }
            if !errors.is_empty() && statuses.iter().any(|s| s == "nothing is loaded") {
                break;
            }
        }
        assert_eq!(errors, ["no backend here"]);
        assert!(
            statuses.iter().any(|s| s == "nothing is loaded"),
            "the engine stopped serving commands after a failed load: {statuses:?}"
        );
    }

    /// The default loader cannot build a GPU session, and says so in terms of
    /// what to do about it.
    #[test]
    fn the_cpu_loader_refuses_a_gpu_backend_by_name() {
        let watch = Watcher::none();
        let err = match cpu_loader()("anything", Backend::Gpu(GpuMode::Bf16), &mut |_| {}, &watch) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("the CPU loader built a GPU session"),
        };
        assert!(err.contains("CPU precision"), "{err}");
    }

    #[test]
    fn backends_cycle_through_every_one_and_come_back() {
        let mut b = Backend::ALL[0];
        for _ in 0..Backend::ALL.len() {
            b = b.next();
        }
        assert_eq!(b, Backend::ALL[0]);
        assert_eq!(Backend::Cpu(Precision::Q8).to_string(), "cpu q8");
        assert_eq!(Backend::Gpu(GpuMode::Q4).to_string(), "gpu q4");
    }
}
