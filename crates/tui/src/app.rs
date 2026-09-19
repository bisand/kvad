//! UI state, and what each key does to it.
//!
//! The app never calls the model directly. Keys mutate state and may push a
//! [`Cmd`] onto the engine's queue; everything that comes back arrives as an
//! [`Evt`] and is applied here. Keeping that one-directional makes the slow
//! parts impossible to accidentally call from the draw path.

use kvad::chat::Message;
use kvad::hub::{self, HubModel, LocalModel};
use kvad::model::Arch;
use kvad::runtime::Stats;
use kvad::service::{Backend, Cmd, Engine, Evt};
use kvad::weights::Fetch;

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
            backend: Backend::Cpu(kvad::quant::Precision::Q8),
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

    /// Ask the engine for `repo`: a repo id, or a directory holding a model.
    pub fn load(&mut self, repo: String, engine: &Engine) {
        if self.busy {
            self.status = "busy — wait for the current operation".into();
            return;
        }
        self.busy = true;
        self.status = format!("loading {repo} on {}", self.backend);
        engine.send(Cmd::Load { repo, backend: self.backend });
    }

    pub fn selected_entry(&self) -> Option<Entry> {
        self.entries().into_iter().nth(self.selected)
    }

    // -- events from the engine ------------------------------------------

    pub fn apply(&mut self, evt: Evt) {
        match evt {
            Evt::Status(s) => self.status = s,
            // A download of several gigabytes says nothing for minutes if it
            // only reports which file it is on. Bytes go to the same line.
            Evt::Fetching(f) => {
                if let Some(line) = describe(&f) {
                    self.status = line;
                }
            }
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
            Evt::Loaded { repo, summary, params, instruct, backend, weight_bytes, .. } => {
                self.status = format!("loaded {repo}");
                self.active =
                    Some(Active { repo, summary, params, instruct, backend, weight_bytes });
                self.busy = false;
                self.messages.clear();
                self.tab = Tab::Chat;
            }
            Evt::Unloaded(repo) => {
                self.active = None;
                self.messages.clear();
                self.streaming = None;
                self.last_stats = None;
                self.busy = false;
                self.status = format!("unloaded {repo}");
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
            // Raw completion, the tokeniser inspector and scoring are things
            // the web UI asks for and this one has no command to send. Taking
            // the text of a `Chose` keeps the terminal honest if that ever
            // changes; the rest is nothing a terminal would draw.
            Evt::Chose(chosen) => {
                self.streaming.get_or_insert_with(String::new).push_str(&chosen.text);
                self.scroll = u16::MAX;
            }
            Evt::Tokens(_) | Evt::Scoring { .. } | Evt::Scored(_) => {}
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
            K::Char('u') => {
                if self.active.is_some() {
                    engine.send(Cmd::Unload);
                } else {
                    self.status = "nothing is loaded".into();
                }
            }
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
                self.load(entry.id, engine);
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
                engine.send(Cmd::Chat {
                    messages: self.messages.clone(),
                    sampling: Default::default(),
                });
            }
            K::Backspace => {
                self.input.pop();
            }
            K::Char(c) => self.input.push(c),
            _ => {}
        }
    }
}

/// A fetch event as one line of status text, or `None` for one with nothing
/// to say that the line above it did not already.
fn describe(f: &Fetch) -> Option<String> {
    match f {
        Fetch::Local => None,
        Fetch::Shards(n) => Some(format!("{n} shards to fetch")),
        // A file whose size is unknown was answered from the cache, and
        // "0 B of 0 B" would be a worse answer than the one already shown.
        Fetch::Download { total: 0, .. } => None,
        Fetch::Download { file, bytes, total } => Some(format!(
            "{file}  {} of {}  ({:.0}%)",
            hub::human_bytes(*bytes),
            hub::human_bytes(*total),
            100.0 * *bytes as f64 / *total as f64
        )),
        Fetch::Fetched { file } => Some(format!("fetched {file}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fetch event either says something a person can read or says
    /// nothing at all; none of them produce a line about zero bytes.
    #[test]
    fn a_download_reads_as_bytes_and_a_percentage() {
        let line = describe(&Fetch::Download {
            file: "model.safetensors".into(),
            bytes: 512 * 1024 * 1024,
            total: 1024 * 1024 * 1024,
        });
        assert_eq!(line.unwrap(), "model.safetensors  512.0 MB of 1.0 GB  (50%)");

        // Cached files report completion and no size; there is nothing to draw.
        assert_eq!(describe(&Fetch::Download { file: "config.json".into(), bytes: 0, total: 0 }), None);
        assert_eq!(describe(&Fetch::Local), None);
        assert_eq!(describe(&Fetch::Shards(4)).unwrap(), "4 shards to fetch");
    }

    /// An unload leaves the app with nothing selected and nothing to send,
    /// rather than a chat pointing at a model that is gone.
    #[test]
    fn unloading_clears_the_conversation_as_well_as_the_model() {
        let mut app = App::new();
        app.active = Some(Active {
            repo: "openai-community/gpt2".into(),
            summary: String::new(),
            params: 0,
            instruct: false,
            backend: "cpu q8".into(),
            weight_bytes: 0,
        });
        app.messages.push(Message::user("hello"));
        app.busy = true;

        app.apply(Evt::Unloaded("openai-community/gpt2".into()));
        assert!(app.active.is_none());
        assert!(app.messages.is_empty());
        assert!(!app.busy);
        assert_eq!(app.status, "unloaded openai-community/gpt2");
    }
}
