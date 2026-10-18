//! The Network test: `relay-probe dial`, run from the phone itself, for the
//! phone's column of Phase 0 in docs/relay-plan.md. It is the probe's own
//! code (tools/relay-probe's library), so the phone dials as the laptop
//! does: post-quantum only, the listener's key pinned by the ticket, no port
//! mapping, and the same path names and table.
//!
//! The run is async, on a small tokio runtime of its own, made at the first
//! run. What it says goes out as events: `probe_status`, `probe_path`,
//! `probe_line` for the log, `probe_ping` with the table so far, and at the
//! end `probe_done`, or `probe_failed`, which ends a run too.
//!
//! The phone's probe key is a file in the app's container, made on first use
//! with the probe's own checks. Its id is shown, to be added to the relay's
//! allow list.

use crate::session::{event, Emit};
use relay_probe::{Event, Net, Pace, Summary, Ticket};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// How often a run pings, in ms: the command's default.
const EVERY: u64 = 250;

pub struct Probe {
    emit: Emit,
    run: Arc<Mutex<Run>>,
    /// Which run is current. What any other says is dropped.
    current: Arc<AtomicU64>,
}

#[derive(Default)]
struct Run {
    rt: Option<tokio::runtime::Runtime>,
    stop: Option<tokio::sync::watch::Sender<bool>>,
    task: Option<tokio::task::JoinHandle<()>>,
    connected: bool,
    running: bool,
}

impl Probe {
    pub fn new(emit: Emit) -> Probe {
        Probe { emit, run: Arc::default(), current: Arc::default() }
    }

    /// The phone's probe id, made the first time.
    pub fn open(&self) {
        (self.emit)(match key_path().and_then(|p| relay_probe::key(&p)) {
            Ok(k) => event("probe_id", json!({ "id": k.public().to_string() })),
            Err(e) => event("probe_failed", json!({ "error": e })),
        });
    }

    /// A run against the listener a `kvadprobe:` ticket names, for `secs`.
    pub fn start(&self, ticket: &str, secs: u64) {
        let fail = |e: String| (self.emit)(event("probe_failed", json!({ "error": e })));
        let mut r = self.run.lock().unwrap();
        if r.running {
            return fail("A test is running. Stop it first.".into());
        }
        let ticket = match Ticket::read(ticket.trim()) {
            Ok(t) => t,
            Err(e) => return fail(e),
        };
        let secret = match key_path().and_then(|p| relay_probe::key(&p)) {
            Ok(k) => k,
            Err(e) => return fail(e),
        };
        let asking = ticket.addr.relay_urls().next().is_some();
        let status = if asking { "Asking the relay about this network…" } else { "Connecting…" };
        (self.emit)(event("probe_started", json!({ "secs": secs, "status": status })));

        let run = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, mut rx) = tokio::sync::watch::channel(false);
        r.connected = false;
        r.running = true;
        r.stop = Some(tx);
        let rt = r.rt.get_or_insert_with(|| {
            tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("a tokio runtime")
        });
        let (emit, state, current) = (self.emit.clone(), self.run.clone(), self.current.clone());
        r.task = Some(rt.spawn(async move {
            let stop = async move {
                let _ = rx.wait_for(|s| *s).await;
            };
            let pace = Pace { secs, every: EVERY };
            let on = |e: &Event, s: &Summary| {
                if matches!(e, Event::Connected { .. }) {
                    state.lock().unwrap().connected = true;
                }
                if current.load(Ordering::SeqCst) == run {
                    for ev in said(e, s) {
                        emit(ev);
                    }
                }
            };
            let result = relay_probe::dial(secret, &Net::default(), ticket, pace, stop, on).await;
            state.lock().unwrap().running = false;
            if current.load(Ordering::SeqCst) == run {
                emit(finished(result, secs));
            }
        }));
    }

    pub fn stop(&self) {
        let mut r = self.run.lock().unwrap();
        if let Some(stop) = r.stop.take() {
            let _ = stop.send(true);
        }
        // Before the connection is up there is no summary to wait for, and a
        // connect can take 30 s to give up: end it now.
        if r.running && !r.connected {
            if let Some(task) = r.task.take() {
                task.abort();
            }
            r.running = false;
            self.current.fetch_add(1, Ordering::SeqCst);
            (self.emit)(event("probe_done", json!({ "status": "Stopped before it connected." })));
        }
    }
}

/// Where the phone's probe key is: the app's container on the phone, the
/// user's Application Support on the Mac.
fn key_path() -> Result<std::path::PathBuf, String> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
    Ok(std::path::PathBuf::from(home).join("Library/Application Support/kvad-phone/probe.key"))
}

fn route(path: &Option<relay_probe::Route>) -> String {
    path.as_ref().map_or("?".into(), |(kind, to)| format!("{kind} · {}", to.trim_start_matches("https://").trim_end_matches('/')))
}

/// The table so far: a row for each path, and the totals under it.
fn table(s: &Summary) -> Value {
    let ms = |d: std::time::Duration| format!("{:.1}", d.as_secs_f64() * 1e3);
    let rows: Vec<Value> = s
        .rows()
        .iter()
        .map(|r| json!({ "path": r.kind, "pings": r.pings, "median": ms(r.median), "p90": ms(r.p90), "max": ms(r.max) }))
        .collect();
    json!({ "rows": rows, "totals": s.totals() })
}

/// What one thing the run said shows: a line for the log, as the command
/// prints it, and what changed above it.
fn said(e: &Event, s: &Summary) -> Vec<Value> {
    let line = || event("probe_line", json!({ "text": e.to_string().trim_start() }));
    match e {
        Event::Network(_) => vec![event("probe_status", json!({ "text": "Connecting…" })), line()],
        Event::Connected { path, .. } => {
            vec![event("probe_status", json!({ "text": "" })), event("probe_path", json!({ "text": route(path) })), line()]
        }
        Event::Moved { path, .. } => vec![event("probe_path", json!({ "text": route(&Some(path.clone())) })), line()],
        Event::Ping { at, rtt } => {
            let mut ping = table(s);
            ping["last"] = json!(relay_probe::ms(*rtt));
            ping["elapsed"] = json!(at.as_secs_f64());
            vec![event("probe_ping", ping)]
        }
    }
}

/// How the run ended.
fn finished(result: Result<Summary, String>, secs: u64) -> Value {
    match result {
        Ok(s) => {
            let status = if s.failure.is_some() {
                "Stopped early."
            } else if s.elapsed.as_secs_f64() + 0.5 < secs as f64 {
                "Stopped."
            } else {
                "Done."
            };
            let mut done = table(&s);
            done["status"] = json!(status);
            done["elapsed"] = json!(s.elapsed.as_secs_f64());
            if let Some(e) = &s.failure {
                done["line"] = json!(format!("stopped early: {e}"));
            }
            event("probe_done", done)
        }
        Err(e) => event("probe_failed", json!({ "error": e })),
    }
}
