//! UI state, and what each key does to it.
//!
//! The app never calls the model directly. Keys mutate state and may push a
//! [`Cmd`] onto the engine's queue; everything that comes back arrives as an
//! [`Evt`] and is applied here. Keeping that one-directional makes the slow
//! parts impossible to accidentally call from the draw path.

use crate::engine::{Backend, Cmd, Engine, Evt};
use llm::chat::Message;
use llm::hub::{self, HubModel, LocalModel};
use llm::model::Arch;
use llm::runtime::Stats;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Tab {
    Models,
    Chat,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Focus {
    Search,
    List,
}

/// One row in the model list, from either the Hub or the local cache.
pub struct Entry {
    pub id: String,
    pub arch: Option<Arch>,
    pub detail: String,
    pub runnable: bool,
    pub local: bool,
}

pub struct Active {
    pub repo: String,
    pub summary: String,
    pub params: usize,
    pub instruct: bool,
    pub backend: String,
    pub weight_bytes: usize,
}

pub struct App {
    pub tab: Tab,
    pub focus: Focus,
    pub query: String,
    pub results: Vec<HubModel>,
    /// True once a search has run, so the list shows Hub results rather than
    /// the local cache.
    pub showing_results: bool,
    pub local: Vec<LocalModel>,
    pub selected: usize,
    pub confirm_delete: Option<String>,

    pub messages: Vec<Message>,
    pub input: String,
    /// The assistant's reply as it arrives, before it becomes a Message.
    pub streaming: Option<String>,
    pub scroll: u16,

    pub active: Option<Active>,
    /// Where the *next* load will run. Changing it does not touch the model
    /// already in memory — the backend is chosen when weights are read.
    pub backend: Backend,
    pub busy: bool,
    pub status: String,
    pub error: Option<String>,
    pub last_stats: Option<Stats>,
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        App {
            tab: Tab::Models,
            focus: Focus::List,
            query: String::new(),
            results: Vec::new(),
            showing_results: false,
            local: Vec::new(),
            selected: 0,
            confirm_delete: None,
            messages: Vec::new(),
            input: String::new(),
            streaming: None,
            scroll: 0,
            active: None,
            backend: Backend::Cpu(llm::quant::Precision::Q8),
            busy: false,
            status: "press / to search the Hub, or pick a downloaded model".into(),
            error: None,
            last_stats: None,
            should_quit: false,
        }
    }

    pub fn entries(&self) -> Vec<Entry> {
        let local_ids: Vec<&str> = self.local.iter().map(|l| l.id.as_str()).collect();
        if self.showing_results {
            self.results
                .iter()
                .map(|m| Entry {
                    local: local_ids.contains(&m.id.as_str()),
                    detail: match m.blocker() {
                        Some(reason) => reason,
                        None => format!(
                            "{} downloads · {}",
                            m.downloads,
                            if m.looks_instruct { "chat" } else { "completion" }
                        ),
                    },
                    runnable: m.runnable(),
                    arch: m.arch,
                    id: m.id.clone(),
                })
                .collect()
        } else {
            self.local
                .iter()
                .map(|m| Entry {
                    id: m.id.clone(),
                    arch: m.arch,
                    detail: format!(
                        "{}{}",
                        hub::human_bytes(m.bytes),
                        if m.complete { "" } else { " · incomplete" }
                    ),
                    runnable: m.arch.is_some() && m.complete,
                    local: true,
                })
                .collect()
        }
    }

    pub fn selected_entry(&self) -> Option<Entry> {
        self.entries().into_iter().nth(self.selected)
    }

    // -- events from the engine ------------------------------------------

    pub fn apply(&mut self, evt: Evt) {
        match evt {
            Evt::Status(s) => self.status = s,
            Evt::Error(e) => {
                self.error = Some(e);
                self.busy = false;
                // A failed generation leaves a half-written reply; keep it
                // rather than silently discarding what the model did say.
                if let Some(partial) = self.streaming.take() {
                    if !partial.is_empty() {
                        self.messages.push(Message::assistant(partial));
                    }
                }
            }
            Evt::SearchResults(r) => {
                self.results = r;
                self.showing_results = true;
                self.selected = 0;
            }
            Evt::Local(l) => {
                self.local = l;
                if !self.showing_results {
                    self.selected = self.selected.min(self.local.len().saturating_sub(1));
                }
            }
            Evt::Loaded { repo, summary, params, instruct, backend, weight_bytes } => {
                self.status = format!("loaded {repo}");
                self.active =
                    Some(Active { repo, summary, params, instruct, backend, weight_bytes });
                self.busy = false;
                self.messages.clear();
                self.tab = Tab::Chat;
            }
            Evt::Token(piece) => {
                self.streaming.get_or_insert_with(String::new).push_str(&piece);
                self.scroll = u16::MAX; // pinned to the bottom
            }
            Evt::Done(stats) => {
                if let Some(reply) = self.streaming.take() {
                    self.messages.push(Message::assistant(reply));
                }
                self.last_stats = Some(stats);
                self.busy = false;
            }
        }
    }

    // -- keys --------------------------------------------------------------

    pub fn on_key(&mut self, key: ratatui::crossterm::event::KeyEvent, engine: &Engine) {
        use ratatui::crossterm::event::{KeyCode as K, KeyModifiers as M};

        // Ctrl+C always quits, wherever the cursor is.
        if key.modifiers.contains(M::CONTROL) && matches!(key.code, K::Char('c')) {
            self.should_quit = true;
            return;
        }
        self.error = None;

        if let Some(id) = self.confirm_delete.clone() {
            match key.code {
                K::Char('y') | K::Char('Y') => {
                    engine.send(Cmd::Delete(id));
                    self.confirm_delete = None;
                }
                _ => {
                    self.confirm_delete = None;
                    self.status = "cancelled".into();
                }
            }
            return;
        }

        if key.code == K::Tab {
            self.tab = if self.tab == Tab::Models { Tab::Chat } else { Tab::Models };
            return;
        }

        match self.tab {
            Tab::Models => self.models_key(key, engine),
            Tab::Chat => self.chat_key(key, engine),
        }
    }

    fn models_key(&mut self, key: ratatui::crossterm::event::KeyEvent, engine: &Engine) {
        use ratatui::crossterm::event::KeyCode as K;

        if self.focus == Focus::Search {
            match key.code {
                K::Enter => {
                    if !self.query.trim().is_empty() {
                        engine.send(Cmd::Search(self.query.trim().to_string()));
                        self.focus = Focus::List;
                    }
                }
                K::Esc => self.focus = Focus::List,
                K::Backspace => {
                    self.query.pop();
                }
                K::Char(c) => self.query.push(c),
                _ => {}
            }
            return;
        }

        let count = self.entries().len();
        match key.code {
            K::Char('/') => self.focus = Focus::Search,
            K::Char('q') => self.should_quit = true,
            K::Char('j') | K::Down => {
                if count > 0 {
                    self.selected = (self.selected + 1) % count;
                }
            }
            K::Char('k') | K::Up => {
                if count > 0 {
                    self.selected = (self.selected + count - 1) % count;
                }
            }
            K::Char('l') => {
                // Back to the local cache listing.
                self.showing_results = false;
                self.selected = 0;
                engine.send(Cmd::RefreshLocal);
            }
            K::Char('r') => engine.send(Cmd::RefreshLocal),
            K::Char('p') => {
                // Cycle where the next load runs. q8 is usually
                // indistinguishable from f32 and faster; q4 halves memory
                // again but visibly degrades small models; the GPU is faster
                // than any of them but only handles the Llama family.
                self.backend = self.backend.next();
                self.status = format!("next load will use {}", self.backend);
            }
            K::Char('d') => {
                if let Some(e) = self.selected_entry() {
                    if e.local {
                        self.confirm_delete = Some(e.id);
                    } else {
                        self.status = "not downloaded".into();
                    }
                }
            }
            K::Enter => {
                let Some(entry) = self.selected_entry() else { return };
                if !entry.runnable {
                    self.status = format!("cannot run {}: {}", entry.id, entry.detail);
                    return;
                }
                if self.busy {
                    self.status = "busy — wait for the current operation".into();
                    return;
                }
                self.busy = true;
                self.status = format!("loading {} on {}", entry.id, self.backend);
                engine.send(Cmd::Load { repo: entry.id, backend: self.backend });
            }
            _ => {}
        }
    }

    fn chat_key(&mut self, key: ratatui::crossterm::event::KeyEvent, engine: &Engine) {
        use ratatui::crossterm::event::{KeyCode as K, KeyModifiers as M};

        match key.code {
            K::Esc => {
                if self.busy {
                    engine.request_cancel();
                    self.status = "stopping…".into();
                }
            }
            K::Char('l') if key.modifiers.contains(M::CONTROL) => {
                self.messages.clear();
                self.streaming = None;
                self.status = "conversation cleared".into();
            }
            K::PageUp => self.scroll = self.scroll.saturating_sub(5),
            K::PageDown => self.scroll = self.scroll.saturating_add(5),
            K::Enter => {
                if self.busy || self.input.trim().is_empty() {
                    return;
                }
                if self.active.is_none() {
                    self.status = "no model loaded — press Tab and pick one".into();
                    return;
                }
                let text = self.input.trim().to_string();
                self.input.clear();
                self.messages.push(Message::user(text));
                self.streaming = Some(String::new());
                self.busy = true;
                self.scroll = u16::MAX;
                engine.send(Cmd::Chat(self.messages.clone()));
            }
            K::Backspace => {
                self.input.pop();
            }
            K::Char(c) => self.input.push(c),
            _ => {}
        }
    }
}
