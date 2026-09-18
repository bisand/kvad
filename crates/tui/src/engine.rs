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
use llm::model::KvCache;
use llm::quant::Precision;
use llm::runtime::{Llm, Stats};
use llm::sampler::Sampler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

pub enum Cmd {
    Search(String),
    Load { repo: String, precision: Precision },
    Chat(Vec<Message>),
    RefreshLocal,
    Delete(String),
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
        precision: Precision,
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

struct Session {
    llm: Llm,
    cache: KvCache,
    /// The exact tokens currently represented in `cache`.
    cached_ids: Vec<u32>,
    sampler: Sampler,
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Evt>, cancel: Arc<AtomicBool>) {
    let say = |msg: &str| {
        let _ = tx.send(Evt::Status(msg.to_string()));
    };
    let fail = |e: Box<dyn std::error::Error>| {
        let _ = tx.send(Evt::Error(e.to_string()));
    };

    let mut session: Option<Session> = None;
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

            Cmd::Load { repo, precision } => {
                // Drop the previous model before loading the next one, or two
                // sets of weights are briefly resident at once.
                session = None;
                say(&format!("loading {repo}"));

                let mut progress = |msg: &str| {
                    let _ = tx.send(Evt::Status(format!("{repo}: {msg}")));
                };
                match Llm::load_with(&repo, precision, &mut progress) {
                    Ok(llm) => {
                        let _ = tx.send(Evt::Loaded {
                            repo: repo.clone(),
                            summary: llm.spec.summary(),
                            params: llm.param_count,
                            instruct: llm.is_instruct(),
                            precision: llm.precision,
                            weight_bytes: llm.weight_bytes,
                        });
                        let _ = hub::State::set_active(&repo);
                        let cache = llm.new_cache();
                        session = Some(Session {
                            llm,
                            cache,
                            cached_ids: Vec::new(),
                            sampler: Sampler::new(0.7, 40, 0.95, 7),
                        });
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

                // Reuse as much of the previous turn's cache as still matches.
                // In a chat the whole history is a shared prefix, so this
                // usually means prefilling only the newest message.
                let shared = Llm::common_prefix(&s.cached_ids, &ids);
                s.cache.truncate(shared);

                let tx2 = tx.clone();
                let cancel2 = Arc::clone(&cancel);
                let result = s.llm.generate(&ids, &mut s.cache, &mut s.sampler, 512, |piece| {
                    if cancel2.load(Ordering::Relaxed) {
                        return false;
                    }
                    tx2.send(Evt::Token(piece.to_string())).is_ok()
                });

                match result {
                    Ok((stats, final_ids)) => {
                        s.cached_ids = final_ids;
                        let _ = tx.send(Evt::Done(stats));
                    }
                    Err(e) => fail(e),
                }
            }
        }
    }
}
