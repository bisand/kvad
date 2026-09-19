//! The background worker.
//!
//! Everything interesting here is slow: downloading a model takes tens of
//! seconds, loading it takes more, and generation produces a token every few
//! tens of milliseconds. None of that can happen on the thread that draws the
//! screen, or the UI would freeze solid and keys would go unanswered.
//!
//! So the engine lives on its own thread, and the two sides talk over a pair
//! of channels: [`Cmd`] in, [`Evt`] out. The UI thread never blocks — it drains
//! whatever events have arrived, redraws, and goes back to polling the
//! keyboard.
//!
//! Cancellation is the one thing a channel cannot express, because the worker
//! is busy inside `generate` and not reading its inbox. That uses a shared
//! [`AtomicBool`] which the per-token callback checks.

use llm::chat::Message;
use llm::hub::{self, HubModel, LocalModel};
use llm::model::Session;
use llm::quant::Precision;
use llm::runtime::{Llm, Stats};
use llm::sampler::Sampler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

pub enum Cmd {
    Search(String),
    Load { repo: String, backend: Backend },
    Chat(Vec<Message>),
    RefreshLocal,
    Delete(String),
}

/// Where a model should run. Cycled from the UI with `p`.
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
    /// Transient progress text for the status bar.
    Status(String),
    SearchResults(Vec<HubModel>),
    Local(Vec<LocalModel>),
    Loaded {
        repo: String,
        summary: String,
        params: usize,
        instruct: bool,
        backend: String,
        weight_bytes: usize,
    },
    /// A fragment of the assistant's reply.
    Token(String),
    Done(Stats),
    Error(String),
}

pub struct Engine {
    tx: Sender<Cmd>,
    pub rx: Receiver<Evt>,
    /// Set by the UI to interrupt generation mid-stream.
    pub cancel: Arc<AtomicBool>,
}

impl Engine {
    pub fn spawn() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let (evt_tx, evt_rx) = mpsc::channel::<Evt>();
        let cancel = Arc::new(AtomicBool::new(false));

        let worker_cancel = Arc::clone(&cancel);
        std::thread::Builder::new()
            .name("llm-engine".into())
            // Generation recurses through 30 layers of closures and rayon
            // scopes; the default 2 MB is enough, but be explicit.
            .stack_size(8 * 1024 * 1024)
            .spawn(move || worker(cmd_rx, evt_tx, worker_cancel))
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

/// The model currently loaded, plus its sampler.
struct Loaded {
    llm: Llm,
    sampler: Sampler,
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Evt>, cancel: Arc<AtomicBool>) {
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

            Cmd::Load { repo, backend } => {
                // Drop the previous model before loading the next one, or two
                // sets of weights are briefly resident at once.
                session = None;
                say(&format!("loading {repo}"));

                let mut progress = |msg: &str| {
                    let _ = tx.send(Evt::Status(format!("{repo}: {msg}")));
                };
                let loaded = match backend {
                    Backend::Cpu(precision) => Llm::load_with(&repo, precision, &mut progress),
                    Backend::Gpu(mode) => {
                        let dtype = llm_gpu::model::parse_dtype("bf16").expect("known dtype");
                        let quant = match mode {
                            GpuMode::Bf16 => None,
                            GpuMode::Q8 => llm_gpu::model::parse_quant("q8").expect("known quant"),
                            GpuMode::Q4 => llm_gpu::model::parse_quant("q4").expect("known quant"),
                        };
                        // The GPU backend covers the Llama family only; the
                        // error names the alternative rather than just failing.
                        Llm::load_custom(&repo, &mut progress, &mut |files, spec| {
                            if spec.arch != llm::model::Arch::Llama {
                                return Err(format!(
                                    "the GPU backend implements the Llama family only; this model is {}. Press p to pick a CPU backend.",
                                    spec.arch
                                )
                                .into());
                            }
                            let device = llm_gpu::model::pick_device(None)?;
                            let m = llm_gpu::model::GpuLlama::load(
                                &files.weights,
                                spec.clone(),
                                dtype,
                                quant,
                                device,
                            )?;
                            Ok(Box::new(m) as Box<dyn Session>)
                        })
                    }
                };
                match loaded {
                    Ok(llm) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: llm.spec.summary(),
                            params: llm.param_count,
                            instruct: llm.is_instruct(),
                            backend: llm.backend(),
                            weight_bytes: llm.weight_bytes,
                        });
                        let _ = hub::State::set_active(&repo);
                        session = Some(Loaded { llm, sampler: Sampler::new(0.7, 40, 0.95, 7) });
                        let _ = tx.send(Evt::Local(hub::local_models()));
                    }
                    Err(e) => fail(e),
                }
            }

            Cmd::Chat(messages) => {
                let Some(s) = session.as_mut() else {
                    say("no model loaded — pick one on the Models tab");
                    continue;
                };
                cancel.store(false, Ordering::Relaxed);

                let ids = match s.llm.encode_chat(&messages) {
                    Ok(ids) => ids,
                    Err(e) => {
                        fail(e);
                        continue;
                    }
                };

                // The session reuses whatever of the previous turn's cache
                // still matches, so a chat prefills only the newest message.
                let tx2 = tx.clone();
                let cancel2 = Arc::clone(&cancel);
                let result = s.llm.generate(&ids, &mut s.sampler, 512, |piece| {
                    if cancel2.load(Ordering::Relaxed) {
                        return false;
                    }
                    tx2.send(Evt::Token(piece.to_string())).is_ok()
                });

                match result {
                    Ok((stats, _)) => {
                        let _ = tx.send(Evt::Done(stats));
                    }
                    Err(e) => fail(e),
                }
            }
        }
    }
}
