//! The Network test screen: `relay-probe dial`, run from the phone itself,
//! for the phone's column of Phase 0 in docs/relay-plan.md. It is the
//! probe's own code (tools/relay-probe's library), so the phone dials as the
//! laptop does: post-quantum only, the listener's key pinned by the ticket,
//! no port mapping, and the same path names and table.
//!
//! The run is async, on a small tokio runtime of its own, made at the first
//! run; what it says comes back to the UI's thread as the other screens'
//! answers do.
//!
//! The phone's probe key is a file in the app's container, made on first
//! use with the probe's own checks, until the app has the Keychain
//! (docs/phone-plan.md). Its id is shown, to be added to the relay's allow
//! list.

use crate::{ios, AppWindow, ProbeRow};
use relay_probe::{Event, Net, Pace, Summary, Ticket};
use slint::{ComponentHandle, Model as _, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The lengths a run can be, as the screen offers them, in seconds.
const LENGTHS: [u64; 3] = [30, 60, 300];
/// How often a run pings, in ms: the command's default.
const EVERY: u64 = 250;

#[derive(Default)]
struct Probe {
    rt: Option<tokio::runtime::Runtime>,
    stop: Option<tokio::sync::watch::Sender<bool>>,
    task: Option<tokio::task::JoinHandle<()>>,
    connected: bool,
}

/// Which run the UI is showing. An answer from any other is dropped.
static RUN: AtomicU64 = AtomicU64::new(0);

pub fn wire(ui: &AppWindow) {
    let probe = Arc::new(Mutex::new(Probe::default()));
    ui.set_probe_rows(ModelRc::from(Rc::new(VecModel::<ProbeRow>::default())));

    ui.on_open_probe({
        let weak = ui.as_weak();
        move || {
            let Some(ui) = weak.upgrade() else { return };
            ui.set_probe_open(true);
            if ui.get_probe_id().is_empty() {
                match key_path().and_then(|p| relay_probe::key(&p)) {
                    Ok(k) => ui.set_probe_id(k.public().to_string().into()),
                    Err(e) => ui.set_probe_error(e.into()),
                }
            }
        }
    });

    ui.on_probe_paste({
        let weak = ui.as_weak();
        move || {
            let Some(ui) = weak.upgrade() else { return };
            match ios::paste() {
                Some(t) => {
                    ui.set_probe_ticket(t.trim().into());
                    ui.set_probe_error("".into());
                }
                None => ui.set_probe_error("Nothing to paste. Copy the ticket on the Mac first.".into()),
            }
        }
    });

    ui.on_probe_copy_id({
        let weak = ui.as_weak();
        move || {
            if let Some(ui) = weak.upgrade() {
                ios::copy(&ui.get_probe_id());
            }
        }
    });

    ui.on_probe_start({
        let weak = ui.as_weak();
        let probe = probe.clone();
        move || {
            let Some(ui) = weak.upgrade() else { return };
            if ui.get_probe_running() {
                return;
            }
            ui.set_probe_error("".into());
            let ticket = match Ticket::read(&ui.get_probe_ticket()) {
                Ok(t) => t,
                Err(e) => return ui.set_probe_error(e.into()),
            };
            let secret = match key_path().and_then(|p| relay_probe::key(&p)) {
                Ok(k) => k,
                Err(e) => return ui.set_probe_error(e.into()),
            };
            let secs = LENGTHS.get(ui.get_probe_length() as usize).copied().unwrap_or(60);
            ui.set_probe_log("".into());
            vec(&ui.get_probe_rows()).clear();
            ui.set_probe_totals("".into());
            ui.set_probe_path("".into());
            ui.set_probe_last("".into());
            ui.set_probe_elapsed(0.0);
            ui.set_probe_secs(secs as f32);
            let asking = ticket.addr.relay_urls().next().is_some();
            ui.set_probe_status(if asking { "Asking the relay about this network…".into() } else { "Connecting…".into() });
            ui.set_probe_running(true);
            ios::keep_awake(true);

            let run = RUN.fetch_add(1, Ordering::SeqCst) + 1;
            let (tx, mut rx) = tokio::sync::watch::channel(false);
            let mut p = probe.lock().unwrap();
            p.connected = false;
            p.stop = Some(tx);
            let rt = p.rt.get_or_insert_with(|| {
                tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("a tokio runtime")
            });
            let (weak, probe) = (weak.clone(), probe.clone());
            p.task = Some(rt.spawn(async move {
                let stop = async move {
                    let _ = rx.wait_for(|s| *s).await;
                };
                let pace = Pace { secs, every: EVERY };
                let on = |e: &Event, s: &Summary| {
                    if matches!(e, Event::Connected { .. }) {
                        probe.lock().unwrap().connected = true;
                    }
                    let (e, s) = (e.clone(), s.clone());
                    let _ = weak.upgrade_in_event_loop(move |ui| {
                        if RUN.load(Ordering::SeqCst) == run {
                            show(&ui, &e, &s);
                        }
                    });
                };
                let result = relay_probe::dial(secret, &Net::default(), ticket, pace, stop, on).await;
                let _ = weak.upgrade_in_event_loop(move |ui| {
                    if RUN.load(Ordering::SeqCst) == run {
                        finish(&ui, result);
                    }
                });
            }));
        }
    });

    ui.on_probe_stop({
        let weak = ui.as_weak();
        let probe = probe.clone();
        move || {
            let mut p = probe.lock().unwrap();
            if let Some(stop) = p.stop.take() {
                let _ = stop.send(true);
            }
            // Before the connection is up there is no summary to wait for,
            // and a connect can take 30 s to give up: end it now.
            if !p.connected {
                if let Some(task) = p.task.take() {
                    task.abort();
                }
                if let Some(ui) = weak.upgrade() {
                    RUN.fetch_add(1, Ordering::SeqCst);
                    ui.set_probe_running(false);
                    ui.set_probe_status("Stopped before it connected.".into());
                    ios::keep_awake(false);
                }
            }
        }
    });
}

/// Where the phone's probe key is: the app's container on the phone, the
/// user's Application Support on the Mac.
fn key_path() -> Result<std::path::PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
    Ok(std::path::PathBuf::from(home).join("Library/Application Support/kvad-phone/probe.key"))
}

/// A model `wire` made a `VecModel`.
fn vec<T: Clone + 'static>(m: &ModelRc<T>) -> &VecModel<T> {
    m.as_any().downcast_ref::<VecModel<T>>().expect("the probe's models are VecModels")
}

fn route(path: &Option<relay_probe::Route>) -> SharedString {
    path.as_ref().map_or("?".into(), |(kind, to)| format!("{kind} · {}", to.trim_start_matches("https://").trim_end_matches('/')).into())
}

/// One thing the run said: a line for the log, as the command prints it,
/// and the table so far.
fn show(ui: &AppWindow, e: &Event, s: &Summary) {
    match e {
        Event::Network(_) => ui.set_probe_status("Connecting…".into()),
        Event::Connected { path, .. } => {
            ui.set_probe_status("".into());
            ui.set_probe_path(route(path));
        }
        Event::Moved { path, .. } => ui.set_probe_path(route(&Some(path.clone()))),
        Event::Ping { at, rtt } => {
            ui.set_probe_last(relay_probe::ms(*rtt).into());
            ui.set_probe_elapsed(at.as_secs_f32());
            set_table(ui, s);
            return;
        }
    }
    add_line(ui, e.to_string().trim_start());
}

fn add_line(ui: &AppWindow, line: &str) {
    let mut log = ui.get_probe_log().to_string();
    if !log.is_empty() {
        log.push('\n');
    }
    log.push_str(line);
    ui.set_probe_log(log.into());
}

fn set_table(ui: &AppWindow, s: &Summary) {
    let rows: Vec<ProbeRow> = s
        .rows()
        .iter()
        .map(|r| {
            let ms = |d: std::time::Duration| SharedString::from(format!("{:.1}", d.as_secs_f64() * 1e3));
            ProbeRow { path: r.kind.into(), pings: r.pings.to_string().into(), median: ms(r.median), p90: ms(r.p90), max: ms(r.max) }
        })
        .collect();
    let table = ui.get_probe_rows();
    vec(&table).set_vec(rows);
    ui.set_probe_totals(s.totals().into());
}

fn finish(ui: &AppWindow, result: Result<Summary, String>) {
    ui.set_probe_running(false);
    ios::keep_awake(false);
    match result {
        Ok(s) => {
            set_table(ui, &s);
            ui.set_probe_elapsed(s.elapsed.as_secs_f32());
            let status = if s.failure.is_some() {
                "Stopped early."
            } else if s.elapsed.as_secs_f32() + 0.5 < ui.get_probe_secs() {
                "Stopped."
            } else {
                "Done."
            };
            ui.set_probe_status(status.into());
            if let Some(e) = &s.failure {
                add_line(ui, &format!("stopped early: {e}"));
            }
        }
        Err(e) => {
            ui.set_probe_status("".into());
            ui.set_probe_error(e.into());
        }
    }
}
