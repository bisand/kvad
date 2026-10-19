//! What the app asks for and what it is told: one command in, events out,
//! each a JSON object with one key, its name (`messages.json` lists them).
//!
//! Every command that waits on the network runs on a thread of its own, so
//! `handle` returns at once. Its answers come through `emit`, from whichever
//! thread has them.
//!
//! A server is an address on the LAN, reached over plain HTTP, or a paired
//! `kvad://` server, reached through the tunnel of docs/relay-plan.md from
//! anywhere. A pairing link from `kvad remote pair` pairs the phone first;
//! its device key, and the key and relay voucher each server gives it, are
//! kept in the Keychain (`pairings`, `secrets`). A key typed for a LAN
//! server is held in memory only.

use crate::api::{Model, Server, Turn};
use crate::{chats, pairings, probe, speak};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Where events go: the C callback, or a test's list.
pub type Emit = Arc<dyn Fn(Value) + Send + Sync>;

/// An event: `{name: body}`.
pub fn event(name: &str, body: Value) -> Value {
    json!({ name: body })
}

pub struct Session {
    emit: Emit,
    state: Arc<Mutex<State>>,
    probe: probe::Probe,
    pub(crate) speak: speak::Speak,
}

#[derive(Default)]
struct State {
    server: Option<Server>,
    turns: Vec<Turn>,
    stop: Arc<AtomicBool>,
    replying: bool,
    /// The conversation's id, once it has a first question: what it is kept
    /// as on the phone.
    chat: Option<String>,
    /// Counts connections. An answer for an earlier one, which a disconnect
    /// has since ended, is dropped.
    connection: u64,
}

impl Session {
    /// A session, which says at once what server the Connect screen should
    /// offer: a paired one first, since a phone that has one most likely
    /// wants it, else the last one connected to.
    pub fn new(emit: Emit) -> Session {
        let server = pairings::Store::load().first().or_else(remembered).unwrap_or_default();
        emit(event("ready", json!({ "server": server.trim_start_matches("http://"), "core": env!("CARGO_PKG_VERSION") })));
        let probe = probe::Probe::new(emit.clone());
        let speak = speak::Speak::new(emit.clone());
        Session { emit, state: Arc::default(), probe, speak }
    }

    /// One command, as the app sends it.
    pub fn handle(&self, command: &Value) {
        let Some((name, args)) = command.as_object().filter(|o| o.len() == 1).and_then(|o| o.iter().next()) else {
            return (self.emit)(event("error", json!({ "error": "A command is an object with one key, its name." })));
        };
        let text = |k: &str| args[k].as_str().unwrap_or_default().to_string();
        match name.as_str() {
            "connect" => self.connect(text("url"), text("key"), text("device")),
            "models" => self.models(),
            "chat" => self.chat(text("model"), text("text"), text("folder")),
            "stop" => self.state.lock().unwrap().stop.store(true, Ordering::Relaxed),
            "new_chat" => {
                let mut st = self.state.lock().unwrap();
                st.turns.clear();
                st.chat = None;
            }
            "chats" => (self.emit)(event("chats", json!({ "chats": chats::list(&text("folder")) }))),
            "chat_open" => self.open_chat(&text("folder"), &text("id")),
            "chat_delete" => self.delete_chat(&text("folder"), &text("id")),
            "disconnect" => {
                self.end();
                (self.emit)(event("disconnected", json!({})));
            }
            "unpair" => self.unpair(),
            "network_changed" => {
                if let Some(server) = self.state.lock().unwrap().server.as_ref() {
                    server.network_changed();
                }
            }
            "opened" => (self.emit)(opened(&text("link"))),
            "probe_open" => self.probe.open(),
            "probe_start" => self.probe.start(&text("ticket"), args["secs"].as_u64().unwrap_or(60)),
            "probe_stop" => self.probe.stop(),
            "speak_start" => self.speak.start(self.server(), &text("model"), &text("language"), &text("folder")),
            "speak_stop" => self.speak.stop(),
            "speak_cancel" => self.speak.cancel(),
            "takes" => self.speak.takes(&text("folder")),
            "take_delete" => self.speak.delete(&text("folder"), &text("id")),
            "retranscribe" => self.speak.retranscribe(self.server(), &text("folder"), &text("id"), &text("model"), &text("language")),
            other => (self.emit)(event("error", json!({ "error": format!("No command is called {other}.") }))),
        }
    }

    /// Ends the connection, and any reply on it. A take being recorded
    /// ends too, on the link it has.
    fn end(&self) {
        self.speak.stop();
        reset(&self.state);
    }

    fn server(&self) -> Option<Server> {
        self.state.lock().unwrap().server.clone()
    }

    /// Samples from the microphone for the take being recorded.
    pub fn audio(&self, samples: &[i16]) {
        self.speak.audio(samples);
    }

    fn connect(&self, url: String, key: String, device: String) {
        self.end();
        let (emit, state) = (self.emit.clone(), self.state.clone());
        let connection = state.lock().unwrap().connection;
        std::thread::spawn(move || {
            let result = open(&url, &key, &device).and_then(|s| {
                let health = s.health()?;
                let models = s.models()?;
                Ok((s, health, models))
            });
            let mut st = state.lock().unwrap();
            if st.connection != connection {
                return;
            }
            match result {
                Err(e) => emit(event("connect_failed", json!({ "error": short_ids(&e) }))),
                Ok((server, health, models)) => {
                    remember(server.name());
                    let name = server.name().trim_start_matches("http://").to_string();
                    let path = server.path().unwrap_or_default();
                    emit(event(
                        "connected",
                        json!({
                            "name": name,
                            "label": short_name(server.name()),
                            "paired": server.name().starts_with("kvad://"),
                            "version": health.version,
                            "who": health.you,
                            "role": health.role,
                            "path": path,
                            "models": models_json(&models),
                            "default": default_model(&models),
                        }),
                    ));
                    st.server = Some(server);
                    drop(st);
                    watch_path(emit, state, connection, path);
                }
            }
        });
    }

    fn models(&self) {
        let Some(server) = self.state.lock().unwrap().server.clone() else {
            return (self.emit)(event("models_failed", json!({ "error": "Not connected to a server." })));
        };
        let emit = self.emit.clone();
        std::thread::spawn(move || match server.models() {
            Ok(models) => emit(event("models", json!({ "models": models_json(&models) }))),
            Err(e) => emit(event("models_failed", json!({ "error": e }))),
        });
    }

    /// Conversation `id` from the phone, to go on with.
    fn open_chat(&self, folder: &str, id: &str) {
        let mut st = self.state.lock().unwrap();
        if st.replying {
            return (self.emit)(event("chats_failed", json!({ "error": "A reply is still coming. Stop it first." })));
        }
        match chats::load(folder, id) {
            Ok((model, turns)) => {
                let shown: Vec<Value> = turns.iter().map(|t| json!({ "role": t.role, "content": t.content })).collect();
                st.turns = turns;
                st.chat = Some(id.to_string());
                drop(st);
                (self.emit)(event("chat_opened", json!({ "id": id, "model": model, "turns": shown })));
            }
            Err(e) => (self.emit)(event("chats_failed", json!({ "error": e }))),
        }
    }

    fn delete_chat(&self, folder: &str, id: &str) {
        let mut st = self.state.lock().unwrap();
        if st.chat.as_deref() == Some(id) {
            if st.replying {
                return (self.emit)(event("chats_failed", json!({ "error": "A reply is still coming. Stop it first." })));
            }
            st.turns.clear();
            st.chat = None;
        }
        drop(st);
        match chats::delete(folder, id) {
            Ok(()) => (self.emit)(event("chat_deleted", json!({ "id": id }))),
            Err(e) => (self.emit)(event("chats_failed", json!({ "error": e }))),
        }
    }

    /// A question in the conversation, and the reply to it as it comes. The
    /// conversation is kept in `folder` once the reply is in, or as much of
    /// it as came.
    fn chat(&self, model: String, text: String, folder: String) {
        let text = text.trim().to_string();
        let fail = |e: &str| (self.emit)(event("chat_failed", json!({ "error": e, "nothing": true })));
        let (server, turns, stop, connection) = {
            let mut st = self.state.lock().unwrap();
            let Some(server) = st.server.clone() else { return fail("Not connected to a server.") };
            if st.replying {
                return fail("A reply is still coming. Stop it first.");
            }
            if text.is_empty() || model.is_empty() {
                return fail("A chat needs a model and something to say.");
            }
            st.turns.push(Turn { role: "user", content: text });
            st.stop = Arc::new(AtomicBool::new(false));
            st.replying = true;
            st.chat.get_or_insert_with(chats::new_id);
            (server, st.turns.clone(), st.stop.clone(), st.connection)
        };
        let (emit, state) = (self.emit.clone(), self.state.clone());
        std::thread::spawn(move || {
            let mut reply = String::new();
            let result = server.chat(&model, &turns, &stop, |piece| {
                reply.push_str(piece);
                emit(event("piece", json!({ "text": piece })));
            });
            let mut st = state.lock().unwrap();
            if st.connection != connection {
                return;
            }
            st.replying = false;
            if !reply.is_empty() {
                st.turns.push(Turn { role: "assistant", content: reply.clone() });
            }
            let kept = (st.chat.clone(), st.turns.clone());
            drop(st);
            match result {
                Ok(()) => emit(event("reply_done", json!({}))),
                // Nothing came back: the error says why, and an empty reply
                // would only say it less well.
                Err(e) => emit(event("chat_failed", json!({ "error": e, "nothing": reply.is_empty() }))),
            }
            if let (Some(id), false) = (kept.0, folder.is_empty()) {
                match chats::save(&folder, &id, &model, &kept.1) {
                    Ok(about) => emit(event("chat_saved", json!({ "chat": about }))),
                    Err(e) => emit(event("chats_failed", json!({ "error": e }))),
                }
            }
        });
    }

    fn unpair(&self) {
        let Some(server) = self.state.lock().unwrap().server.clone() else {
            return (self.emit)(event("unpair_failed", json!({ "error": "Not connected to a server." })));
        };
        let (emit, state) = (self.emit.clone(), self.state.clone());
        std::thread::spawn(move || {
            let name = server.name().to_string();
            let mut store = pairings::Store::load();
            let outcome = match store.get(&name).and_then(|p| p.device) {
                Some(id) => server.unpair(id).and_then(|_| store.forget(&name)),
                None => store.forget(&name),
            };
            match outcome {
                Ok(()) => {
                    reset(&state);
                    emit(event("unpaired", json!({})));
                }
                Err(e) => emit(event("unpair_failed", json!({ "error": format!("Could not unpair: {e}") }))),
            }
        });
    }
}

/// No connection, and an end to any reply on the one there was.
fn reset(state: &Mutex<State>) {
    let mut st = state.lock().unwrap();
    st.stop.store(true, Ordering::Relaxed);
    let connection = st.connection + 1;
    *st = State { connection, ..State::default() };
}

/// Says which path the tunnel is on whenever it changes, every 2 s, for as
/// long as the connection lasts.
fn watch_path(emit: Emit, state: Arc<Mutex<State>>, connection: u64, mut said: String) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(2));
        let now = {
            let st = state.lock().unwrap();
            match (&st.server, st.connection == connection) {
                (Some(server), true) => server.path().unwrap_or_default(),
                _ => return,
            }
        };
        if now != said {
            emit(event("path", json!({ "now": now })));
            said = now;
        }
    });
}

fn models_json(models: &[Model]) -> Value {
    models.iter().map(|m| json!({ "id": m.id, "kind": m.kind, "resident": m.resident })).collect()
}

/// The server `url` names. A pairing link pairs first; a `kvad://` name is a
/// paired server, through the tunnel; anything else is an address on the
/// LAN, over plain HTTP, with `key` if it wants one. `device` is what the
/// server's list of devices calls this phone, if it pairs.
pub fn open(url: &str, key: &str, device: &str) -> Result<Server, String> {
    let url = url.trim();
    if url.contains("&pair=") {
        let name = join(url, device)?;
        return through(&name);
    }
    if url.starts_with("kvad://") {
        return through(url.split('?').next().unwrap_or(url).trim_end_matches('/'));
    }
    Server::new(url, Some(key.to_string()))
}

/// Pairs this phone with the server a link from `kvad remote pair` names,
/// through the relay if it is not on the LAN, and keeps what the server
/// gives it. The server's name, `kvad://<node id>`.
pub fn join(link: &str, device: &str) -> Result<String, String> {
    let link: kvad_tunnel::PairLink = link.parse()?;
    let mut store = pairings::Store::load();
    let key = store.device_key()?;
    let device = if device.trim().is_empty() { "Kvad for iPhone" } else { device.trim() };
    let request = json!({ "v": 1, "token": link.token, "name": device }).to_string();
    let answer = kvad_tunnel::blocking::pair_as(&key, &link.address, request.as_bytes(), link.voucher.as_ref())?;
    let answer: Value = serde_json::from_slice(&answer).map_err(|_| "The server's answer was not one a kvad gives.".to_string())?;
    if let Some(e) = answer["error"].as_str() {
        return Err(e.to_string());
    }
    let text = |k: &str| answer[k].as_str().map(str::to_string);
    let name = link.address.label();
    let pairing = pairings::Pairing {
        address: link.address.to_string(),
        key: text("key").ok_or("The server paired this phone and sent no key.")?,
        voucher: text("voucher"),
        account: text("account").unwrap_or_default(),
        role: text("role").unwrap_or_default(),
        device: answer["paired"].as_i64(),
    };
    store.put(&name, &pairing)?;
    Ok(name)
}

/// A paired server, through the tunnel, showing relays the voucher it gave
/// this phone, and swapping that for a fresh one once half is spent.
pub fn through(name: &str) -> Result<Server, String> {
    let mut store = pairings::Store::load();
    let mut pairing = store
        .get(name)
        .ok_or_else(|| format!("This phone is not paired with {name}. Paste a pairing link from `kvad remote pair` on the Mac."))?;
    let key = store.device_key()?;
    let address: kvad_tunnel::Address = pairing.address.parse()?;
    let voucher = pairing.voucher.as_deref().and_then(|v| v.parse::<kvad_tunnel::Voucher>().ok());
    let tunnel = kvad_tunnel::blocking::Tunnel::connect_as(&key, &address, voucher.as_ref())?;
    let server = Server::through(tunnel, pairing.key.clone());
    if voucher.is_none_or(|v| v.left() < kvad_tunnel::VOUCHER_LIFE / 2) {
        if let Ok(fresh) = server.voucher() {
            pairing.voucher = Some(fresh);
            let _ = store.put(name, &pairing);
        }
    }
    Ok(server)
}

/// What a `kvad://` link opened in the app is: a server, or a pairing link,
/// and the line that asks the person to go on. It is never acted on by
/// itself: it only fills in the Server field, and pairing waits for Connect.
pub fn opened(link: &str) -> Value {
    let link = link.trim();
    let notice = if link.len() > 4096 {
        None
    } else if link.contains("&pair=") {
        link.parse::<kvad_tunnel::PairLink>().ok().map(|l| format!("Pair with {}? Tap Connect to pair this phone.", short_name(&format!("kvad://{}", l.address.node))))
    } else {
        link.parse::<kvad_tunnel::Address>().ok().map(|a| format!("Opened {}. Tap Connect to reach it.", short_name(&format!("kvad://{}", a.node))))
    };
    match notice {
        Some(notice) => event("link", json!({ "server": link, "notice": notice })),
        None => event("link_refused", json!({ "error": "That link is not a Kvad server or pairing link." })),
    }
}

/// `text` with every `kvad://` node id in it cut short, so that an error can
/// wrap: 64 hex digits are one word, wider than the screen.
pub fn short_ids(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find("kvad://") {
        out.push_str(&rest[..i + 7]);
        rest = &rest[i + 7..];
        let hex = rest.bytes().take_while(u8::is_ascii_hexdigit).count();
        if hex > 12 {
            out.push_str(&rest[..12]);
            out.push('…');
            rest = &rest[hex..];
        }
    }
    out.push_str(rest);
    out
}

/// A server's name as a row has room for: `kvad://` and the first twelve
/// digits of its node id, or an address as it is.
pub fn short_name(name: &str) -> String {
    match name.strip_prefix("kvad://") {
        Some(id) if id.len() > 12 => format!("kvad://{}…", &id[..12]),
        _ => name.trim_start_matches("http://").to_string(),
    }
}

/// A chat model already in memory if there is one, else the first chat model.
pub fn default_model(models: &[Model]) -> String {
    let chat = models.iter().filter(|m| m.kind == "chat");
    chat.clone().find(|m| m.resident).or_else(|| chat.clone().next()).map(|m| m.id.clone()).unwrap_or_default()
}

/// Where the last server's address is kept: in the app's own container on
/// the phone. It is an address, not a secret; the key is never written.
fn remembered_path() -> Option<std::path::PathBuf> {
    if !cfg!(target_os = "ios") {
        return None;
    }
    let home = std::env::var_os("HOME")?;
    Some(std::path::PathBuf::from(home).join("Library/Application Support/kvad-phone/server"))
}

pub fn remembered() -> Option<String> {
    let s = std::fs::read_to_string(remembered_path()?).ok()?;
    Some(s.trim().to_string()).filter(|s| !s.is_empty())
}

pub fn remember(base: &str) {
    if let Some(p) = remembered_path() {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(p, base);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_ids_cut_node_ids_to_twelve_digits() {
        let id = "bdbc76ccbacc9b027f1da0edb724d8110b1f22672510a63091c527ff9445cbd0";
        assert_eq!(short_ids(&format!("not paired with kvad://{id}.")), "not paired with kvad://bdbc76ccbacc….");
        assert_eq!(short_ids("kvad://abc and more"), "kvad://abc and more");
        assert_eq!(short_name(&format!("kvad://{id}")), "kvad://bdbc76ccbacc…");
        assert_eq!(short_name("http://192.168.1.20:5823"), "192.168.1.20:5823");
    }

    #[test]
    fn the_default_model_is_a_resident_chat_model() {
        let m = |id: &str, kind: &str, resident| Model { id: id.into(), kind: kind.into(), resident };
        let models = [m("a/whisper", "speech", true), m("a/one", "chat", false), m("a/two", "chat", true)];
        assert_eq!(default_model(&models), "a/two");
        assert_eq!(default_model(&models[..2]), "a/one");
        assert_eq!(default_model(&models[..1]), "");
    }
}
