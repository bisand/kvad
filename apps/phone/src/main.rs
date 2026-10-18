//! Kvad on the iPhone, Phase 1 of docs/phone-plan.md: connect to a
//! kvad-serve, list its models, and chat with a reply that streams in.
//!
//! The UI is `ui/app.slint`. This file wires its callbacks to `api`, each
//! call on a worker thread, and hands the answers back to the UI's thread
//! with `upgrade_in_event_loop`.
//!
//! The API key is held in memory only, for now: it goes to the Keychain
//! once the app has a Keychain binding (the plan's first open question), and
//! never to a file.

mod api;

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

fn main() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    let state = Arc::new(Mutex::new(State::default()));
    ui.set_bubbles(ModelRc::from(Rc::new(VecModel::<Bubble>::default())));
    ui.set_models(ModelRc::from(Rc::new(VecModel::<ModelRow>::default())));
    if let Some(url) = remembered() {
        ui.set_server_url(url.into());
    }

    ui.on_connect({
        let weak = ui.as_weak();
        let state = state.clone();
        move |url, key| {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_connecting(true);
            ui.set_connect_error("".into());
            let (weak, state) = (weak.clone(), state.clone());
            std::thread::spawn(move || {
                let key = Some(key.to_string());
                let result = Server::new(&url, key).and_then(|s| {
                    let health = s.health()?;
                    let models = s.models()?;
                    Ok((s, health, models))
                });
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    ui.set_connecting(false);
                    match result {
                        Err(e) => ui.set_connect_error(e.into()),
                        Ok((server, health, models)) => {
                            remember(server.base());
                            ui.set_server_url(server.base().trim_start_matches("http://").into());
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

    #[cfg(debug_assertions)]
    script::at_launch(&ui);

    ui.run()
}

/// For screenshots and a scripted look at each state, in debug builds only:
/// `simctl launch` passes `SIMCTL_CHILD_KVAD_PHONE_*` through as these.
/// `KVAD_PHONE_CONNECT` is a server to connect to at launch,
/// `KVAD_PHONE_TAB` the tab to open (0 to 4), `KVAD_PHONE_MODEL` the chat
/// model, and `KVAD_PHONE_SAY` a message to send once connected.
#[cfg(debug_assertions)]
mod script {
    use super::*;

    pub fn at_launch(ui: &AppWindow) {
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
