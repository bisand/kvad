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

mod ios;
mod probe;

use kvad_phone_core::api::{Model, Server, Turn};
use kvad_phone_core::pairings;
use kvad_phone_core::session::{default_model, remember, remembered, short_ids, short_name};
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

/// The server `url` names, paired first if it is a pairing link.
fn open(url: &str, key: &str) -> Result<Server, String> {
    kvad_phone_core::session::open(url, key, &ios::device_name())
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

    // A `kvad://` link opened in the app, from the Camera's QR scanner or a
    // tap, goes in the Server field and waits for Connect: pairing is the
    // person's act, never the link's. Coming back to the foreground looks for
    // paths again, since the phone may be on another network by then.
    ios::listen(ios::Events {
        opened: Box::new({
            let weak = ui.as_weak();
            move |link| {
                let _ = weak.upgrade_in_event_loop(move |ui| opened(&ui, &link));
            }
        }),
        active: Box::new({
            let state = state.clone();
            move || {
                if let Some(server) = state.lock().unwrap().server.as_ref() {
                    server.network_changed();
                }
            }
        }),
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

/// Puts an opened link in front of the person, once it parses as a server or
/// a pairing link; on the Connect screen, or, while connected to a server,
/// as a line in the chat that says how to get to it.
fn opened(ui: &AppWindow, link: &str) {
    let said = kvad_phone_core::session::opened(link);
    let say = |text: &str| match ui.get_connected() {
        true => ui.set_chat_error(text.into()),
        false => ui.set_connect_error(text.into()),
    };
    let Some(notice) = said["link"]["notice"].as_str() else {
        say(said["link_refused"]["error"].as_str().unwrap_or_default());
        return;
    };
    ui.set_server_url(link.trim().into());
    ui.set_api_key("".into());
    if ui.get_connected() {
        say("A Kvad link was opened. Disconnect in More to use it.");
    } else {
        ui.set_connect_error("".into());
        ui.set_connect_notice(notice.into());
    }
}
