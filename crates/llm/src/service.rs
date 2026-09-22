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

use crate::chat::Message;
use crate::hub::{self, HubModel, LocalModel};
use crate::quant::Precision;
use crate::runtime::{Chosen, Llm, Perplexity, Stats, Token};
use crate::sampler::Sampler;
use crate::weights::{Fetch, Watcher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub enum Cmd {
    Search(String),
    Load { repo: String, backend: Backend },
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
    Chat { messages: Vec<Message>, tools: Vec<serde_json::Value>, sampling: Sampling },
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
    /// How the loaded model's tokenizer splits a text.
    Tokenize(String),
    /// Score a text the model did not write.
    Perplexity { text: String, window: usize },
    RefreshLocal,
    Delete(String),
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

impl Default for Sampling {
    fn default() -> Self {
        // 1024 rather than 512 because of reasoning models. Qwen3 writes its
        // working before its answer, and on a grounded question that working
        // alone ran to about 650 tokens — so at 512 a request that names no
        // limit gets a trace and no answer. Measured: at 400 the same model
        // spent every token thinking and never reached one.
        Sampling { temperature: 0.7, top_k: 40, top_p: 0.95, seed: None, max_tokens: 1024 }
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
    Error(String),
}

/// Turns a repo id and a [`Backend`] into a loaded model.
///
/// See the module docs for why this is the caller's to supply. `progress` is
/// the line of text a status bar shows; `watch` is the same thing in bytes,
/// and both are already wired to the event channel by the time a loader sees
/// them.
pub type Loader = Box<
    dyn FnMut(&str, Backend, &mut dyn FnMut(&str), &Watcher) -> Res<Llm> + Send,
>;

/// The loader for a build with no GPU crate in it.
///
/// CPU backends load; a GPU one is refused with the key that changes it,
/// rather than with a type error at the other end of the program.
pub fn cpu_loader() -> Loader {
    Box::new(|repo, backend, progress, watch| match backend {
        Backend::Cpu(precision) => Llm::load_watched(repo, precision, progress, watch),
        Backend::Gpu(_) => {
            Err("this build has no GPU backend; pick a CPU precision instead".into())
        }
    })
}

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
fn kvad_kv_bytes_per_token(spec: &crate::model::Spec) -> usize {
    spec.n_layer * (spec.cache.k + spec.cache.v) * std::mem::size_of::<f32>()
}

/// The model currently loaded, plus its sampler.
struct Loaded {
    llm: Llm,
    sampler: Sampler,
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Evt>, cancel: Arc<AtomicBool>, mut load: Loader) {
    let say = |msg: &str| {
        let _ = tx.send(Evt::Status(msg.to_string()));
    };
    let fail = |e: Box<dyn std::error::Error>| {
        let _ = tx.send(Evt::Error(e.to_string()));
    };

    let mut session: Option<Loaded> = None;
    let _ = tx.send(Evt::Local(hub::local_models()));

    while let Ok(cmd) = rx.recv() {
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
                Some(local) => match std::fs::remove_dir_all(&local.path) {
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
                    let _ = tx.send(Evt::Unloaded(was.llm.repo.clone()));
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
                    Ok(llm) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: llm.spec.summary(),
                            params: llm.param_count,
                            instruct: llm.is_instruct(),
                            tools: llm.takes_tools(),
                            backend: llm.backend(),
                            weight_bytes: llm.weight_bytes,
                            n_ctx: llm.spec.n_ctx,
                            kv_bytes_per_token: kvad_kv_bytes_per_token(&llm.spec),
                        });
                        let _ = hub::State::set_active(&repo);
                        let d = Sampling::default();
                        let sampler =
                            Sampler::new(d.temperature, d.top_k, d.top_p, DEFAULT_SEED);
                        session = Some(Loaded { llm, sampler });
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Err(e) => fail(e),
                }
            }

            Cmd::Chat { messages, tools, sampling } => {
                let Some(s) = session.as_mut() else {
                    say("no model loaded");
                    continue;
                };
                aim(s, &sampling);
                let ids = match s.llm.encode_chat(&messages, &tools) {
                    Ok(ids) => ids,
                    Err(e) => {
                        fail(e);
                        continue;
                    }
                };
                // The session reuses whatever of the previous turn's cache
                // still matches, so a chat prefills only the newest message.
                run(s, &ids, &sampling, 0, &tx, &cancel);
            }

            Cmd::Complete { prompt, sampling, explain, fresh } => {
                let Some(s) = session.as_mut() else {
                    say("no model loaded");
                    continue;
                };
                aim(s, &sampling);
                if fresh {
                    if let Err(e) = s.llm.reset() {
                        fail(e);
                        continue;
                    }
                }
                // No template: the prompt is the text, which is what makes
                // this a different view of the model rather than a second
                // chat window.
                let ids = match s.llm.encode(&prompt) {
                    Ok(ids) => ids,
                    Err(e) => {
                        fail(e);
                        continue;
                    }
                };
                run(s, &ids, &sampling, explain, &tx, &cancel);
            }

            Cmd::Tokenize(text) => match session.as_ref() {
                None => say("no model loaded"),
                Some(s) => match s.llm.tokenize(&text) {
                    Ok(tokens) => {
                        let _ = tx.send(Evt::Tokens(tokens));
                    }
                    Err(e) => fail(e),
                },
            },

            Cmd::Perplexity { text, window } => {
                let Some(s) = session.as_mut() else {
                    say("no model loaded");
                    continue;
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
        }
    }
}

/// Point the session's sampler at what this request asked for.
///
/// The knobs are the request's; the generator is the session's, unless the
/// request asked for one of its own.
fn aim(s: &mut Loaded, sampling: &Sampling) {
    s.sampler.temperature = sampling.temperature;
    s.sampler.top_k = sampling.top_k;
    s.sampler.top_p = sampling.top_p;
    if let Some(seed) = sampling.seed {
        s.sampler.reseed(seed);
    }
}

/// Generate from token ids, forwarding every piece down the channel.
///
/// Shared by chat and raw completion, which differ only in how the ids were
/// arrived at — and that difference belongs above this line, not inside the
/// loop.
fn run(
    s: &mut Loaded,
    ids: &[u32],
    sampling: &Sampling,
    explain: usize,
    tx: &Sender<Evt>,
    cancel: &Arc<AtomicBool>,
) {
    cancel.store(false, Ordering::Relaxed);
    let tx2 = tx.clone();
    let cancel2 = Arc::clone(cancel);
    let result =
        s.llm.generate_explained(ids, &mut s.sampler, sampling.max_tokens, explain, |chosen| {
            if cancel2.load(Ordering::Relaxed) {
                return false;
            }
            let evt = match explain {
                0 => Evt::Token(chosen.text.clone()),
                _ => Evt::Chose(chosen.clone()),
            };
            tx2.send(evt).is_ok()
        });

    match result {
        Ok((stats, _)) => {
            let _ = tx.send(Evt::Done(stats));
        }
        Err(e) => {
            let _ = tx.send(Evt::Error(e.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        use crate::model::{CacheShape, KvCache};

        let mut spec = tiny_spec();

        for (what, cache, n_kv_head, head_dim) in [
            // Llama: what is cached is what `kv_dim()` describes.
            ("llama", CacheShape::kv(256, 256), 2, 128),
            // DeepSeek: 64 rotated + 512 latent, against a `kv_dim()` of 2048.
            ("deepseek", CacheShape::kv(64, 512), 16, 128),
        ] {
            spec.cache = cache;
            spec.n_kv_head = n_kv_head;
            spec.head_dim = head_dim;
            assert_eq!(
                kvad_kv_bytes_per_token(&spec) * spec.n_ctx,
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
            cache: crate::model::CacheShape::kv(256, 256),
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
