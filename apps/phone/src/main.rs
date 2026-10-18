//! Kvad on the iPhone, Phase 1 of docs/phone-plan.md: connect to a
//! kvad-serve, list its models, and chat with a reply that streams in.
//!
//! The UI is `ui/app.slint`. This file wires its callbacks to `api`, each
//! call on a worker thread, and hands the answers back to the UI's thread
//! with `upgrade_in_event_loop`.
//!
//! A server is an address on the LAN, reached over plain HTTP, or a paired
//! `kvad://` server, reached through the tunnel of docs/relay-plan.md from
//! anywhere (Phase 4). Pasting a pairing link from `kvad remote pair` pairs
//! the phone; its device key, and the key and relay voucher each server
//! gives it, are kept in the Keychain (`pairings`, `secrets`). A key typed
//! for a LAN server is held in memory only.

mod api;
mod ios;
mod pairings;
mod probe;
mod secrets;

use api::{Model, Server, Turn};
use slint::{Model as _, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

slint::include_modules!();

#[derive(Default)]
struct State {
    server: Option<Server>,
    turns: Vec<Turn>,
    stop: Arc<AtomicBool>,
}

/// The server `url` names. A pairing link pairs first; a `kvad://` name is a
/// paired server, through the tunnel; anything else is an address on the
/// LAN, over plain HTTP, with `key` if it wants one.
fn open(url: &str, key: &str) -> Result<Server, String> {
    let url = url.trim();
    if url.contains("&pair=") {
        let name = join(url)?;
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
fn join(link: &str) -> Result<String, String> {
    let link: kvad_tunnel::PairLink = link.parse()?;
    let mut store = pairings::Store::load();
    let key = store.device_key()?;
    let request = serde_json::json!({ "v": 1, "token": link.token, "name": ios::device_name() }).to_string();
    let answer = kvad_tunnel::blocking::pair_as(&key, &link.address, request.as_bytes(), link.voucher.as_ref())?;
    let answer: serde_json::Value = serde_json::from_slice(&answer).map_err(|_| "The server's answer was not one a kvad gives.".to_string())?;
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
fn through(name: &str) -> Result<Server, String> {
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

fn main() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    let state = Arc::new(Mutex::new(State::default()));
    ui.set_bubbles(ModelRc::from(Rc::new(VecModel::<Bubble>::default())));
    ui.set_models(ModelRc::from(Rc::new(VecModel::<ModelRow>::default())));
    // A paired server first: a phone that has one most likely wants it.
    if let Some(url) = pairings::Store::load().first().or_else(remembered) {
        ui.set_server_url(url.trim_start_matches("http://").into());
    }

    // Which path the tunnel is on, kept up to date while connected.
    let path_timer = slint::Timer::default();
    path_timer.start(slint::TimerMode::Repeated, std::time::Duration::from_secs(2), {
        let weak = ui.as_weak();
        let state = state.clone();
        move || {
            let Some(ui) = weak.upgrade() else { return };
            let path = state.lock().unwrap().server.as_ref().and_then(|s| s.path()).unwrap_or_default();
            if ui.get_server_path() != path.as_str() {
                ui.set_server_path(path.into());
            }
        }
    });

    ui.on_paste_link({
        let weak = ui.as_weak();
        move || {
            if let (Some(ui), Some(text)) = (weak.upgrade(), ios::paste()) {
                ui.set_server_url(text.trim().into());
            }
        }
    });

    ui.on_unpair({
        let weak = ui.as_weak();
        let state = state.clone();
        move || {
            let Some(server) = state.lock().unwrap().server.clone() else { return };
            let weak = weak.clone();
            std::thread::spawn(move || {
                let name = server.name().to_string();
                let mut store = pairings::Store::load();
                let outcome = match store.get(&name).and_then(|p| p.device) {
                    Some(id) => server.unpair(id).and_then(|_| store.forget(&name)),
                    None => store.forget(&name),
                };
                let _ = weak.upgrade_in_event_loop(move |ui| match outcome {
                    Ok(()) => {
                        ui.set_server_url("".into());
                        ui.invoke_disconnect();
                    }
                    Err(e) => ui.set_chat_error(format!("Could not unpair: {e}").into()),
                });
            });
        }
    });

    ui.on_connect({
        let weak = ui.as_weak();
        let state = state.clone();
        move |url, key| {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_connecting(true);
            ui.set_connect_error("".into());
            let (weak, state) = (weak.clone(), state.clone());
            std::thread::spawn(move || {
                let result = open(&url, &key).and_then(|s| {
                    let health = s.health()?;
                    let models = s.models()?;
                    Ok((s, health, models))
                });
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_connecting(false);
                    match result {
                        Err(e) => ui.set_connect_error(short_ids(&e).into()),
                        Ok((server, health, models)) => {
                            remember(server.name());
                            ui.set_server_url(server.name().trim_start_matches("http://").into());
                            ui.set_server_path(server.path().unwrap_or_default().into());
                            ui.set_server_name(short_name(server.name()).into());
                            ui.set_paired(server.name().starts_with("kvad://"));
                            ui.set_server_version(format!("kvad {}", health.version).into());
                            let who = if health.you.is_empty() { health.role.clone() } else { format!("{} ({})", health.you, health.role) };
                            ui.set_server_who(who.into());
                            show_models(&ui, &models);
                            if ui.get_model().is_empty() {
                                choose(&ui, &default_model(&models));
                            }
                            state.lock().unwrap().server = Some(server);
                            ui.set_connected(true);
                            #[cfg(debug_assertions)]
                            script::after_connect(&ui);
                        }
                    }
                });
            });
        }
    });

    ui.on_refresh_models({
        let weak = ui.as_weak();
        let state = state.clone();
        move || {
            let Some(server) = state.lock().unwrap().server.clone() else { return };
            if let Some(ui) = weak.upgrade() {
                ui.set_models_loading(true);
            }
            let weak = weak.clone();
            std::thread::spawn(move || {
                let result = server.models();
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_models_loading(false);
                    if let Ok(models) = result {
                        show_models(&ui, &models);
                    }
                });
            });
        }
    });

    ui.on_choose_model({
        let weak = ui.as_weak();
        move |id| {
            if let Some(ui) = weak.upgrade() {
                choose(&ui, &id);
            }
        }
    });

    ui.on_send({
        let weak = ui.as_weak();
        let state = state.clone();
        move |text| {
            let text = text.trim().to_string();
            let Some(ui) = weak.upgrade() else { return };
            if text.is_empty() || ui.get_replying() {
                return;
            }
            let model = ui.get_model().to_string();
            let (server, turns, stop) = {
                let mut st = state.lock().unwrap();
                let Some(server) = st.server.clone() else { return };
                st.turns.push(Turn { role: "user", content: text.clone() });
                st.stop = Arc::new(AtomicBool::new(false));
                (server, st.turns.clone(), st.stop.clone())
            };
            let b = bubbles(&ui);
            rows(&b).push(Bubble { mine: true, text: text.into(), streaming: false });
            rows(&b).push(Bubble { mine: false, text: "".into(), streaming: true });
            ui.set_draft("".into());
            ui.set_chat_error("".into());
            ui.set_replying(true);

            let (weak, state) = (weak.clone(), state.clone());
            std::thread::spawn(move || {
                let mut reply = String::new();
                let result = server.chat(&model, &turns, &stop, |piece| {
                    reply.push_str(piece);
                    let now: SharedString = reply.as_str().into();
                    let _ = weak.upgrade_in_event_loop(move |ui| set_last(&ui, now, true));
                });
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_replying(false);
                    if let Err(e) = &result {
                        ui.set_chat_error(e.as_str().into());
                    }
                    let b = bubbles(&ui);
                    if reply.is_empty() && result.is_err() {
                        // Nothing came back: the error says why, and the
                        // empty bubble would only say it less well.
                        rows(&b).remove(b.row_count() - 1);
                    } else {
                        set_last(&ui, reply.as_str().into(), false);
                        state.lock().unwrap().turns.push(Turn { role: "assistant", content: reply });
                    }
                });
            });
        }
    });

    ui.on_stop({
        let state = state.clone();
        move || state.lock().unwrap().stop.store(true, Ordering::Relaxed)
    });

    ui.on_new_chat({
        let weak = ui.as_weak();
        let state = state.clone();
        move || {
            state.lock().unwrap().turns.clear();
            if let Some(ui) = weak.upgrade() {
                rows(&bubbles(&ui)).clear();
                ui.set_chat_error("".into());
            }
        }
    });

    ui.on_disconnect({
        let weak = ui.as_weak();
        let state = state.clone();
        move || {
            let mut st = state.lock().unwrap();
            st.stop.store(true, Ordering::Relaxed);
            *st = State::default();
            if let Some(ui) = weak.upgrade() {
                rows(&bubbles(&ui)).clear();
                ui.set_api_key("".into());
                choose(&ui, "");
                ui.set_tab(0);
                ui.set_connected(false);
            }
        }
    });

    probe::wire(&ui);

    #[cfg(debug_assertions)]
    script::at_launch(&ui);

    ui.run()
}

/// For screenshots and a scripted look at each state, in debug builds only:
/// `simctl launch` passes `SIMCTL_CHILD_KVAD_PHONE_*` through as these.
/// `KVAD_PHONE_CONNECT` is a server to connect to at launch,
/// `KVAD_PHONE_TAB` the tab to open (0 to 4), `KVAD_PHONE_MODEL` the chat
/// model, `KVAD_PHONE_SAY` a message to send once connected, and
/// `KVAD_PHONE_UNPAIR` unpairs the phone from the server it connected to.
/// `KVAD_PHONE_PROBE` opens the Network test, and when it is a ticket, runs
/// it, as long as `KVAD_PHONE_PROBE_LENGTH` (0 to 2: 30 s, 1 min, 5 min) says.
#[cfg(debug_assertions)]
mod script {
    use super::*;

    pub fn at_launch(ui: &AppWindow) {
        if let Ok(probe) = std::env::var("KVAD_PHONE_PROBE") {
            ui.invoke_open_probe();
            if let Some(length) = std::env::var("KVAD_PHONE_PROBE_LENGTH").ok().and_then(|l| l.parse().ok()) {
                ui.set_probe_length(length);
            }
            if probe.starts_with("kvadprobe:") {
                ui.set_probe_ticket(probe.into());
                ui.invoke_probe_start();
            }
        }
        if let Ok(url) = std::env::var("KVAD_PHONE_CONNECT") {
            ui.set_server_url(url.as_str().into());
            ui.invoke_connect(url.into(), "".into());
        }
    }

    pub fn after_connect(ui: &AppWindow) {
        if let Some(tab) = std::env::var("KVAD_PHONE_TAB").ok().and_then(|t| t.parse().ok()) {
            ui.set_tab(tab);
            if tab == 3 {
                ui.invoke_refresh_models();
            }
        }
        if let Ok(model) = std::env::var("KVAD_PHONE_MODEL") {
            choose(ui, &model);
        }
        if let Ok(say) = std::env::var("KVAD_PHONE_SAY") {
            ui.invoke_send(say.into());
        }
        if std::env::var("KVAD_PHONE_SHEET").is_ok() {
            ui.set_picking_model(true);
        }
        if std::env::var("KVAD_PHONE_UNPAIR").is_ok() {
            ui.invoke_unpair();
        }
    }
}

/// The conversation's rows, which `main` made a `VecModel` at the start.
fn bubbles(ui: &AppWindow) -> ModelRc<Bubble> {
    ui.get_bubbles()
}

fn rows(m: &ModelRc<Bubble>) -> &VecModel<Bubble> {
    m.as_any().downcast_ref::<VecModel<Bubble>>().expect("the bubbles are a VecModel")
}

fn set_last(ui: &AppWindow, text: SharedString, streaming: bool) {
    let b = bubbles(ui);
    if let Some(i) = b.row_count().checked_sub(1) {
        rows(&b).set_row_data(i, Bubble { mine: false, text, streaming });
    }
}

/// The models grouped by what they are for, chat first, as the Models
/// screen's sections show them.
fn show_models(ui: &AppWindow, models: &[Model]) {
    let order = |k: &str| ["chat", "speech", "image", "video", "embedding"].iter().position(|x| *x == k).unwrap_or(9);
    let mut sorted: Vec<&Model> = models.iter().collect();
    sorted.sort_by(|a, b| (order(&a.kind), &a.kind, &a.id).cmp(&(order(&b.kind), &b.kind, &b.id)));
    let rows: Vec<ModelRow> = sorted
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let (owner, name) = m.id.split_once('/').unwrap_or(("", &m.id));
            ModelRow {
                id: m.id.as_str().into(),
                name: name.into(),
                owner: owner.into(),
                kind: m.kind.as_str().into(),
                resident: m.resident,
                first: i == 0 || sorted[i - 1].kind != m.kind,
                last: i + 1 == sorted.len() || sorted[i + 1].kind != m.kind,
            }
        })
        .collect();
    ui.set_models(ModelRc::from(Rc::new(VecModel::from(rows))));
}

/// The model the chat asks, and the name its title shows.
fn choose(ui: &AppWindow, id: &str) {
    ui.set_model(id.into());
    ui.set_model_name(id.split_once('/').map_or(id, |(_, name)| name).into());
}

/// `text` with every `kvad://` node id in it cut short, so that an error can
/// wrap: 64 hex digits are one word, wider than the screen.
fn short_ids(text: &str) -> String {
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
fn short_name(name: &str) -> String {
    match name.strip_prefix("kvad://") {
        Some(id) if id.len() > 12 => format!("kvad://{}…", &id[..12]),
        _ => name.trim_start_matches("http://").to_string(),
    }
}

/// A chat model already in memory if there is one, else the first chat model.
fn default_model(models: &[Model]) -> String {
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

fn remembered() -> Option<String> {
    let s = std::fs::read_to_string(remembered_path()?).ok()?;
    Some(s.trim().to_string()).filter(|s| !s.is_empty())
}

fn remember(base: &str) {
    if let Some(p) = remembered_path() {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(p, base);
    }
}
