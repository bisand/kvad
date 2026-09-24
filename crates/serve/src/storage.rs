//! Where the server keeps its data, and moving it from the web UI.
//!
//! The data directory holds the database, generated images, datasets and
//! trained models; the Hugging Face cache comes along when the data
//! directory was chosen rather than defaulted (see `kvad::hub::cache_dir`).
//! Changing `[data] dir` alone would start the server empty somewhere else,
//! so the page moves the data instead, and the setting is the last step.
//!
//! # How a move goes
//!
//! 1. **While serving.** Whatever lives on another disk from the target is
//!    copied there, symlinks as symlinks (the Hub's cache is built of them),
//!    with progress. Nothing the server reads is touched yet.
//! 2. **Switching.** Under the database's lock, so that nothing can write
//!    after the copy is taken: the database is copied with `VACUUM INTO`,
//!    files that changed during step 1 are copied again, and what is on the
//!    same disk as the target is renamed there — instant, and the reason
//!    that step waits until now. Then `[data] dir` is written to kvad.toml
//!    and the server restarts into it.
//! 3. **Afterwards.** What was copied is still where it was. The new
//!    database remembers where, and the page offers to delete it. Nothing is
//!    deleted before somebody says so.
//!
//! A move that fails or is cancelled before the switch removes what it made
//! in the target, which was empty when it started, and leaves everything
//! else as it was.

use crate::api::{blocking, blocking_or, Fail};
use crate::auth::{Admin, State};
use crate::db::Db;
use crate::settings::Running;
use axum::extract::{Query, State as St};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/data", get(overview))
        .route("/api/data/plan", get(plan_for))
        .route("/api/data/move", post(start).delete(cancel))
        .route("/api/data/previous", delete(forget_previous))
}

/// The key in the settings table that remembers the copies moves left
/// behind: a list, since a second move can come before the first one's copy
/// is deleted.
const PREVIOUS: &str = "storage.previous";

/// Names in the data directory that are the database, which is moved by
/// `VACUUM INTO` rather than as files.
const DATABASE_FILES: [&str; 4] = ["kvad.db", "kvad.db-wal", "kvad.db-shm", "kvad.db-journal"];

// ---------------------------------------------------------------------------
// What there is
// ---------------------------------------------------------------------------

/// Where this server's data is, as the page shows it.
#[derive(Debug, Clone, serde::Serialize)]
struct Overview {
    data_dir: PathBuf,
    /// Whether somebody chose the data directory; a defaulted one does not
    /// hold the Hub's cache.
    chosen: bool,
    hub: Hub,
    database: PathBuf,
    /// `false` when `[database] path` or `--db` put it somewhere of its own,
    /// and a move leaves it there.
    database_moves: bool,
    items: Vec<Item>,
    total_bytes: u64,
    /// Why this server's data cannot be moved from the page, if it cannot.
    blocked: Option<String>,
    previous: Vec<Previous>,
    moving: Option<Progress>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Hub {
    path: PathBuf,
    /// Whether a move takes it along. Not when `HF_HUB_CACHE` or `HF_HOME`
    /// decides where it is, which is `env`.
    follows: bool,
    env: Option<&'static str>,
}

/// One thing a move carries: a directory or file in the data directory, the
/// database, or the Hub's cache.
#[derive(Debug, Clone, serde::Serialize)]
struct Item {
    name: String,
    from: PathBuf,
    bytes: u64,
    kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Data,
    Database,
    Hub,
}

/// A copy left behind by a move, until somebody deletes it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Previous {
    pub from: PathBuf,
    pub to: PathBuf,
    /// Exactly what is left, and all that a delete removes.
    pub paths: Vec<PathBuf>,
    pub bytes: u64,
}

fn hub_env() -> Option<&'static str> {
    let set = |name| std::env::var_os(name).is_some_and(|v| !v.is_empty());
    if set("HF_HUB_CACHE") {
        Some("HF_HUB_CACHE")
    } else if set("HF_HOME") {
        Some("HF_HOME")
    } else {
        None
    }
}

/// Everything a move of `running`'s data would carry, sized.
fn items(running: &Running) -> Vec<Item> {
    let data = &running.data_dir;
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(data) {
        let mut names: Vec<_> = entries.flatten().map(|e| e.file_name()).collect();
        names.sort();
        for name in names {
            let name = name.to_string_lossy().to_string();
            if DATABASE_FILES.contains(&name.as_str()) {
                continue;
            }
            let from = data.join(&name);
            out.push(Item { bytes: size_of(&from), name, from, kind: Kind::Data });
        }
    }
    if running.database_moves() {
        let bytes = DATABASE_FILES.iter().map(|f| size_of(&data.join(f))).sum();
        out.push(Item { name: "kvad.db".into(), from: running.database.clone(), bytes, kind: Kind::Database });
    }
    let hub = kvad::hub::cache_dir();
    if hub_env().is_none() && !hub.starts_with(data) && hub.exists() {
        out.push(Item { name: "huggingface/hub".into(), bytes: size_of(&hub), from: hub, kind: Kind::Hub });
    }
    out
}

fn describe(running: &Running, db: &Db) -> Overview {
    let items = items(running);
    Overview {
        data_dir: running.data_dir.clone(),
        chosen: kvad::weights::chosen_data_dir().is_some(),
        hub: Hub { path: kvad::hub::cache_dir(), follows: hub_env().is_none(), env: hub_env() },
        database: running.database.clone(),
        database_moves: running.database_moves(),
        total_bytes: items.iter().map(|i| i.bytes).sum(),
        items,
        blocked: blocked(running).err(),
        previous: previous(db),
        moving: running.storage.progress(),
    }
}

/// Whether this server can move its data from the page at all.
fn blocked(running: &Running) -> Result<(), String> {
    if !cfg!(unix) {
        return Err("this build cannot restart itself, which a move ends with".into());
    }
    if std::env::var_os("KVAD_DATA_DIR").is_some_and(|v| !v.is_empty()) {
        return Err("the server is started with KVAD_DATA_DIR, which wins over kvad.toml; \
                    change it where the server is started"
            .into());
    }
    if running.is_restarting() {
        return Err("the server is restarting".into());
    }
    Ok(())
}

fn previous(db: &Db) -> Vec<Previous> {
    db.setting(PREVIOUS).ok().flatten().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

async fn overview(_: Admin, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let running = Arc::clone(&state.settings);
    let db = state.db.clone();
    let overview = blocking(move || Ok(describe(&running, &db))).await?;
    Ok(Json(serde_json::to_value(overview).map_err(|e| Fail::internal(e.to_string()))?))
}

// ---------------------------------------------------------------------------
// A plan
// ---------------------------------------------------------------------------

/// What moving to `to` would do, worked out before anything is.
#[derive(Debug, Clone, serde::Serialize)]
struct Plan {
    to: PathBuf,
    steps: Vec<Step>,
    /// What has to be copied rather than renamed, the database included.
    copy_bytes: u64,
    /// Whether anything but the database is copied, which is what makes a
    /// move take time: the database is always copied, and small.
    crosses_disk: bool,
    /// Free space where it is going, when anything is copied.
    free_bytes: Option<u64>,
    /// Whether this makes the Hub's cache move somewhere the Hub's other
    /// tools will not look.
    leaves_shared_cache: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Step {
    name: String,
    kind: Kind,
    from: PathBuf,
    to: PathBuf,
    bytes: u64,
    /// Same disk: renamed at the switch. Otherwise copied first.
    rename: bool,
}

/// A path as somebody typed it: `~/` is the home directory, and it has to
/// be absolute otherwise, since a relative one means nothing to a server.
fn typed_path(to: &str) -> Result<PathBuf, String> {
    let to = to.trim();
    let home = || std::env::var_os("HOME").map(PathBuf::from).ok_or("there is no HOME to put ~ in");
    let path = match to.strip_prefix("~/") {
        Some(rest) => home()?.join(rest),
        None if to == "~" => home()?,
        None => PathBuf::from(to),
    };
    if !path.is_absolute() {
        return Err(format!("`{to}` is not a full path; start it with / or ~/"));
    }
    // Without `..` and friends, so that nesting below is judged on the path
    // that will be used.
    Ok(path.components().collect())
}

/// Whether `a` and `b` are one inside the other, or the same.
fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// A directory with its symlinks resolved, as far as it exists.
fn resolved(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        match (existing.file_name().map(|n| n.to_os_string()), existing.parent()) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => break,
        }
    }
    let mut out = existing.canonicalize().unwrap_or(existing);
    for name in rest.into_iter().rev() {
        out.push(name);
    }
    out
}

fn plan(running: &Running, to: &str) -> Result<Plan, String> {
    blocked(running)?;
    let to = typed_path(to)?;
    let here = resolved(&running.data_dir);
    let target = resolved(&to);
    if target == here {
        return Err(format!("the data is already in {}", to.display()));
    }
    let items = items(running);
    for item in &items {
        if overlaps(&target, &resolved(&item.from)) {
            return Err(format!("{} overlaps {}, which is being moved", to.display(), item.from.display()));
        }
    }
    if overlaps(&target, &here) {
        return Err(format!("{} overlaps the data directory {}", to.display(), running.data_dir.display()));
    }
    match std::fs::read_dir(&to) {
        Ok(mut entries) => {
            // `.DS_Store` is Finder's, and not anybody's data.
            if entries.any(|e| e.is_ok_and(|e| e.file_name() != ".DS_Store")) {
                return Err(format!("{} is not empty; a move goes into an empty or new directory", to.display()));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotADirectory => {
            return Err(format!("{} is a file", to.display()))
        }
        Err(e) => return Err(format!("could not read {}: {e}", to.display())),
    }

    // The device the target will be on: its own if it exists, or its
    // nearest existing parent's.
    let mut probe = to.clone();
    while !probe.exists() {
        probe = probe.parent().ok_or("no part of that path exists")?.to_path_buf();
    }
    let target_dev = device(&probe)?;

    let steps: Vec<Step> = items
        .iter()
        .map(|item| {
            let dest = match item.kind {
                Kind::Data => to.join(&item.name),
                Kind::Database => to.join("kvad.db"),
                Kind::Hub => to.join("huggingface/hub"),
            };
            // The database is always copied, by `VACUUM INTO`: an open SQLite
            // file with a write-ahead log is not something to rename.
            let rename = item.kind != Kind::Database && device(&item.from).ok() == Some(target_dev);
            Step { name: item.name.clone(), kind: item.kind, from: item.from.clone(), to: dest, bytes: item.bytes, rename }
        })
        .collect();
    let copy_bytes: u64 = steps.iter().filter(|s| !s.rename).map(|s| s.bytes).sum();
    let crosses_disk = steps.iter().any(|s| !s.rename && s.kind != Kind::Database);
    let free_bytes = free_space(&probe);
    if let Some(free) = free_bytes {
        // A gigabyte to spare, so that the move does not fill the disk the
        // server is about to write to.
        let spare = 1_000_000_000;
        if copy_bytes + spare > free {
            return Err(format!(
                "{} needs {} free and has {}",
                probe.display(),
                gb(copy_bytes + spare),
                gb(free)
            ));
        }
    }
    let default_hub = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"));
    let leaves_shared_cache = steps.iter().any(|s| s.kind == Kind::Hub && Some(&s.from) == default_hub.as_ref());
    Ok(Plan { to, steps, copy_bytes, crosses_disk, free_bytes, leaves_shared_cache })
}

/// In the web UI's units, so that the page and this say the same number.
fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
}

#[cfg(unix)]
fn device(path: &Path) -> Result<u64, String> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).map(|m| m.dev()).map_err(|e| format!("could not look at {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn device(_: &Path) -> Result<u64, String> {
    Ok(0)
}

#[cfg(unix)]
fn free_space(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a valid NUL-terminated path and `stat` is ours to fill.
    if unsafe { libc::statvfs(c.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn free_space(_: &Path) -> Option<u64> {
    None
}

#[derive(serde::Deserialize)]
struct To {
    to: String,
}

async fn plan_for(_: Admin, St(state): St<State>, Query(q): Query<To>) -> Result<Json<serde_json::Value>, Fail> {
    let running = Arc::clone(&state.settings);
    let planned = blocking_or(move || plan(&running, &q.to).map_err(Fail::bad)).await?;
    Ok(Json(serde_json::to_value(planned).map_err(|e| Fail::internal(e.to_string()))?))
}

// ---------------------------------------------------------------------------
// Moving
// ---------------------------------------------------------------------------

/// How a move is going, for the page to poll.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Progress {
    to: PathBuf,
    /// `copying`, `switching`, `restarting`, `failed` or `cancelled`.
    stage: &'static str,
    done_bytes: u64,
    total_bytes: u64,
    current: Option<String>,
    error: Option<String>,
}

/// The one move there can be at a time. Lives in [`Running`].
#[derive(Default)]
pub struct Mover {
    progress: Mutex<Option<Progress>>,
    cancel: AtomicBool,
}

impl Mover {
    fn progress(&self) -> Option<Progress> {
        self.progress.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn update(&self, f: impl FnOnce(&mut Progress)) {
        if let Some(p) = self.progress.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            f(p);
        }
    }

    pub fn busy(&self) -> bool {
        matches!(self.progress().map(|p| p.stage), Some("copying" | "switching" | "restarting"))
    }
}

async fn start(_: Admin, St(state): St<State>, Json(q): Json<To>) -> Result<(StatusCode, Json<serde_json::Value>), Fail> {
    let running = Arc::clone(&state.settings);
    let db = state.db.clone();
    let runtime = tokio::runtime::Handle::current();
    let planned = blocking_or({
        let running = Arc::clone(&running);
        move || {
            if running.storage.busy() {
                return Err(Fail::conflict("a move is already under way"));
            }
            if running.is_restarting() {
                return Err(Fail::conflict("the server is restarting"));
            }
            let planned = plan(&running, &q.to).map_err(Fail::bad)?;
            // The file has to take the new setting before anything moves.
            crate::settings::data_dir_edit(&running, &planned.to).map_err(Fail::bad)?;
            Ok(planned)
        }
    })
    .await?;

    *running.storage.progress.lock().unwrap_or_else(|e| e.into_inner()) = Some(Progress {
        to: planned.to.clone(),
        stage: "copying",
        done_bytes: 0,
        total_bytes: planned.copy_bytes,
        current: None,
        error: None,
    });
    running.storage.cancel.store(false, Ordering::SeqCst);
    tracing::info!("moving the data to {}", planned.to.display());
    let body = json!({ "moving": planned.to });
    std::thread::spawn(move || {
        let outcome = carry(&running, &db, &planned, &runtime);
        match outcome {
            Ok(()) => {}
            Err(Stop::Cancelled) => {
                tracing::info!("the move to {} was cancelled", planned.to.display());
                running.storage.update(|p| p.stage = "cancelled");
            }
            Err(Stop::Failed(why)) => {
                tracing::warn!("the move to {} failed: {why}", planned.to.display());
                running.storage.update(|p| {
                    p.stage = "failed";
                    p.error = Some(why);
                });
            }
        }
    });
    Ok((StatusCode::ACCEPTED, Json(body)))
}

async fn cancel(_: Admin, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let mover = &state.settings.storage;
    match mover.progress().map(|p| p.stage) {
        Some("copying") => {
            mover.cancel.store(true, Ordering::SeqCst);
            Ok(Json(json!({ "cancelling": true })))
        }
        Some("switching" | "restarting") => Err(Fail::conflict("the move is switching over and cannot stop now")),
        _ => Err(Fail::missing("there is no move under way")),
    }
}

enum Stop {
    Cancelled,
    Failed(String),
}

impl From<String> for Stop {
    fn from(why: String) -> Self {
        Stop::Failed(why)
    }
}

/// The move itself; see the module comment for its steps.
fn carry(running: &Arc<Running>, db: &Db, plan: &Plan, runtime: &tokio::runtime::Handle) -> Result<(), Stop> {
    let made_target = !plan.to.exists();
    let copied = || plan.steps.iter().filter(|s| !s.rename && s.kind != Kind::Database);
    let undo = |renamed: &[&Step]| {
        for step in renamed.iter().rev() {
            let _ = std::fs::rename(&step.to, &step.from);
        }
        empty_out(&plan.to, made_target);
    };

    std::fs::create_dir_all(&plan.to).map_err(|e| format!("could not make {}: {e}", plan.to.display()))?;

    // 1. Copies, while the server goes on serving from where it is.
    let mut done = 0u64;
    for step in copied() {
        running.storage.update(|p| p.current = Some(step.name.clone()));
        if let Err(stop) = copy_tree(&step.from, &step.to, &mut |n| {
            done += n;
            running.storage.update(|p| p.done_bytes = done);
            !running.storage.cancel.load(Ordering::SeqCst)
        }) {
            undo(&[]);
            return Err(stop);
        }
    }
    if running.storage.cancel.load(Ordering::SeqCst) {
        undo(&[]);
        return Err(Stop::Cancelled);
    }

    // 2. The switch, with the database held so that nothing writes after
    //    its copy is taken.
    running.storage.update(|p| {
        p.stage = "switching";
        p.current = None;
    });
    let switched: Result<Result<(), String>, _> = db.with(|conn| {
        let mut renamed: Vec<&Step> = Vec::new();
        let result = (|| -> Result<(), String> {
            if let Some(step) = plan.steps.iter().find(|s| s.kind == Kind::Database) {
                conn.execute("VACUUM INTO ?1", [step.to.to_string_lossy()])
                    .map_err(|e| format!("could not copy the database to {}: {e}", step.to.display()))?;
            }
            for step in copied() {
                sync_tree(&step.from, &step.to)?;
            }
            for step in plan.steps.iter().filter(|s| s.rename) {
                if let Some(parent) = step.to.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("could not make {}: {e}", parent.display()))?;
                }
                std::fs::rename(&step.from, &step.to)
                    .map_err(|e| format!("could not move {} to {}: {e}", step.from.display(), step.to.display()))?;
                renamed.push(step);
            }

            // What is left behind, for the new server to offer to delete.
            let mut left: Vec<PathBuf> = copied().map(|s| s.from.clone()).collect();
            if plan.steps.iter().any(|s| s.kind == Kind::Database) {
                left.extend(DATABASE_FILES.iter().map(|f| running.data_dir.join(f)).filter(|p| p.exists()));
            }
            let previous = Previous {
                from: running.data_dir.clone(),
                to: plan.to.clone(),
                bytes: left.iter().map(|p| size_of(p)).sum(),
                paths: left,
            };
            let restore = remember(running, &plan.to, previous)?;
            crate::settings::set_data_dir(running, &plan.to).inspect_err(|_| restore())
        })();
        if let Err(why) = &result {
            // The file was not written, or this would not be reached: the
            // server stays where it is, with everything back in place.
            let _ = why;
            undo(&renamed);
        }
        Ok(result)
    });
    match switched {
        Ok(Ok(())) => {}
        Ok(Err(why)) => return Err(Stop::Failed(why)),
        Err(e) => {
            undo(&[]);
            return Err(Stop::Failed(e.to_string()));
        }
    }

    // 3. Into the new place.
    running.storage.update(|p| p.stage = "restarting");
    tracing::info!("the data is in {}; restarting into it", plan.to.display());
    let _guard = runtime.enter();
    crate::settings::begin_restart(Arc::clone(running)).map_err(|why| {
        format!("the data was moved and kvad.toml says so, but the restart failed: {why}. Restart the server to finish")
    })?;
    Ok(())
}

/// Add the copy left behind to the list, in the database the server will
/// open next: the new one, or this one when the database is not moving.
/// Either way it already holds the list so far, since `VACUUM INTO` copied
/// it. Answers with what puts the list back, for a switch that fails after.
fn remember(running: &Running, to: &Path, previous: Previous) -> Result<impl FnOnce(), String> {
    let path = if running.database_moves() { to.join("kvad.db") } else { running.database.clone() };
    let conn = rusqlite::Connection::open(&path).map_err(|e| format!("could not open {}: {e}", path.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(5)).map_err(|e| e.to_string())?;
    let before: Option<String> = conn
        .query_row("SELECT value FROM settings WHERE key = ?1", [PREVIOUS], |r| r.get(0))
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(other),
        })
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let mut list: Vec<Previous> = before.as_deref().and_then(|t| serde_json::from_str(t).ok()).unwrap_or_default();
    list.push(previous);
    let write = |conn: &rusqlite::Connection, value: Option<&str>| match value {
        Some(value) => conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')",
            [PREVIOUS, value],
        ),
        None => conn.execute("DELETE FROM settings WHERE key = ?1", [PREVIOUS]),
    };
    let value = serde_json::to_string(&list).map_err(|e| e.to_string())?;
    write(&conn, Some(&value)).map_err(|e| format!("could not note the old copy in {}: {e}", path.display()))?;
    Ok(move || {
        let _ = write(&conn, before.as_deref());
    })
}

/// Take back what a move put in `to`, which was empty when it started.
fn empty_out(to: &Path, made: bool) {
    if made {
        let _ = std::fs::remove_dir_all(to);
    } else if let Ok(entries) = std::fs::read_dir(to) {
        for entry in entries.flatten() {
            if entry.file_name() == ".DS_Store" {
                continue;
            }
            let path = entry.path();
            let _ = match entry.file_type() {
                Ok(t) if t.is_dir() => std::fs::remove_dir_all(&path),
                _ => std::fs::remove_file(&path),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Copying
// ---------------------------------------------------------------------------

/// Bytes a tree holds, not following symlinks: the Hub's cache links every
/// snapshot to a blob, and counting both would count it twice.
fn size_of(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else { return 0 };
    if meta.is_dir() {
        std::fs::read_dir(path).map(|es| es.flatten().map(|e| size_of(&e.path())).sum()).unwrap_or(0)
    } else if meta.is_file() {
        meta.len()
    } else {
        0
    }
}

/// Copy `from` to `to`, which does not exist yet: directories, files with
/// their permissions and modification times, and symlinks as symlinks.
/// `step` is told of every chunk, and stops the copy by answering false.
fn copy_tree(from: &Path, to: &Path, step: &mut dyn FnMut(u64) -> bool) -> Result<(), Stop> {
    let meta = std::fs::symlink_metadata(from).map_err(|e| format!("could not read {}: {e}", from.display()))?;
    if meta.file_type().is_symlink() {
        link_like(from, to)?;
    } else if meta.is_dir() {
        std::fs::create_dir_all(to).map_err(|e| format!("could not make {}: {e}", to.display()))?;
        let entries = std::fs::read_dir(from).map_err(|e| format!("could not read {}: {e}", from.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("could not read {}: {e}", from.display()))?;
            copy_tree(&entry.path(), &to.join(entry.file_name()), step)?;
        }
        let _ = std::fs::set_permissions(to, meta.permissions());
    } else if meta.is_file() {
        copy_file(from, to, &meta, step)?;
    }
    Ok(())
}

fn copy_file(from: &Path, to: &Path, meta: &std::fs::Metadata, step: &mut dyn FnMut(u64) -> bool) -> Result<(), Stop> {
    use std::io::{Read, Write};
    let fail = |what: &str, p: &Path, e: std::io::Error| Stop::Failed(format!("could not {what} {}: {e}", p.display()));
    let mut src = std::fs::File::open(from).map_err(|e| fail("read", from, e))?;
    let mut dst = std::fs::File::create(to).map_err(|e| fail("write", to, e))?;
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = src.read(&mut buf).map_err(|e| fail("read", from, e))?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).map_err(|e| fail("write", to, e))?;
        if !step(n as u64) {
            return Err(Stop::Cancelled);
        }
    }
    dst.sync_all().map_err(|e| fail("write", to, e))?;
    let _ = dst.set_permissions(meta.permissions());
    if let Ok(modified) = meta.modified() {
        let _ = dst.set_modified(modified);
    }
    Ok(())
}

#[cfg(unix)]
fn link_like(from: &Path, to: &Path) -> Result<(), String> {
    let target = std::fs::read_link(from).map_err(|e| format!("could not read {}: {e}", from.display()))?;
    std::os::unix::fs::symlink(&target, to).map_err(|e| format!("could not make {}: {e}", to.display()))
}

#[cfg(not(unix))]
fn link_like(from: &Path, _: &Path) -> Result<(), String> {
    Err(format!("{} is a symlink, which this build cannot copy", from.display()))
}

/// Make `to` what `from` is now, after `from` changed during a copy: new
/// and changed files copied again, and what `from` no longer has removed.
/// Small by the time it runs, since the copy has just been made.
fn sync_tree(from: &Path, to: &Path) -> Result<(), String> {
    let unchanged = |a: &std::fs::Metadata, b: &std::fs::Metadata| {
        a.len() == b.len() && a.modified().ok() == b.modified().ok()
    };
    let src = match std::fs::symlink_metadata(from) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return remove_any(to);
        }
        Err(e) => return Err(format!("could not read {}: {e}", from.display())),
    };
    let dst = std::fs::symlink_metadata(to).ok();
    let never = &mut |_| true;
    match (src.is_dir() && !src.file_type().is_symlink(), dst) {
        (true, Some(d)) if d.is_dir() => {
            let mut seen = std::collections::HashSet::new();
            for entry in std::fs::read_dir(from).map_err(|e| format!("could not read {}: {e}", from.display()))? {
                let entry = entry.map_err(|e| format!("could not read {}: {e}", from.display()))?;
                seen.insert(entry.file_name());
                sync_tree(&entry.path(), &to.join(entry.file_name()))?;
            }
            for entry in std::fs::read_dir(to).map_err(|e| format!("could not read {}: {e}", to.display()))?.flatten() {
                if !seen.contains(&entry.file_name()) {
                    remove_any(&entry.path())?;
                }
            }
            Ok(())
        }
        (false, Some(d)) if !src.file_type().is_symlink() && d.is_file() && unchanged(&src, &d) => Ok(()),
        (_, Some(_)) => {
            remove_any(to)?;
            copy_tree(from, to, never).map_err(stop_to_string)
        }
        (_, None) => copy_tree(from, to, never).map_err(stop_to_string),
    }
}

fn stop_to_string(stop: Stop) -> String {
    match stop {
        Stop::Cancelled => "cancelled".into(),
        Stop::Failed(why) => why,
    }
}

fn remove_any(path: &Path) -> Result<(), String> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    let removed =
        if meta.is_dir() && !meta.file_type().is_symlink() { std::fs::remove_dir_all(path) } else { std::fs::remove_file(path) };
    removed.map_err(|e| format!("could not remove {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// The copy left behind
// ---------------------------------------------------------------------------

async fn forget_previous(_: Admin, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let running = Arc::clone(&state.settings);
    let db = state.db.clone();
    blocking_or(move || {
        let list = previous(&db);
        if list.is_empty() {
            return Err(Fail::missing("there is no old copy to delete"));
        }
        let freed = delete_previous(&running, &list).map_err(Fail::bad)?;
        db.with(|c| c.execute("DELETE FROM settings WHERE key = ?1", [PREVIOUS]))
            .map_err(|e| Fail::internal(e.to_string()))?;
        let deleted: Vec<&PathBuf> = list.iter().flat_map(|p| &p.paths).collect();
        Ok(Json(json!({ "deleted": deleted, "bytes": freed })))
    })
    .await
}

/// Delete exactly the paths moves left behind, and nothing the server is
/// using now: all of them are checked before any is deleted. Then each old
/// data directory, if that left it empty.
fn delete_previous(running: &Running, list: &[Previous]) -> Result<u64, String> {
    let live = [resolved(&running.data_dir), resolved(&kvad::hub::cache_dir()), resolved(&running.database)];
    for path in list.iter().flat_map(|p| &p.paths) {
        let path = resolved(path);
        if let Some(inside) = live.iter().find(|l| overlaps(&path, l)) {
            return Err(format!(
                "{} overlaps {}, which the server is using now; nothing was deleted",
                path.display(),
                inside.display()
            ));
        }
    }
    let mut freed = 0;
    for previous in list {
        for path in &previous.paths {
            freed += size_of(path);
            remove_any(path)?;
        }
        let emptied = std::fs::read_dir(&previous.from)
            .is_ok_and(|mut e| e.all(|e| e.is_ok_and(|e| e.file_name() == ".DS_Store")));
        if emptied {
            let _ = std::fs::remove_dir_all(&previous.from);
        }
    }
    Ok(freed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvad-storage-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The Hub's cache is blobs and symlinks to them. A copy that followed
    /// the links would double it and break the layout the Hub reads.
    #[test]
    fn a_copy_keeps_symlinks_as_symlinks_and_files_as_they_were() {
        let root = scratch("copy");
        let from = root.join("hub/models--a--b");
        std::fs::create_dir_all(from.join("blobs")).unwrap();
        std::fs::create_dir_all(from.join("snapshots/abc")).unwrap();
        std::fs::write(from.join("blobs/123"), b"weights").unwrap();
        std::os::unix::fs::symlink("../../blobs/123", from.join("snapshots/abc/model.safetensors")).unwrap();

        let mut seen = 0;
        assert!(copy_tree(&root.join("hub"), &root.join("moved"), &mut |n| {
            seen += n;
            true
        })
        .is_ok());
        let to = root.join("moved/models--a--b");
        assert!(std::fs::symlink_metadata(to.join("snapshots/abc/model.safetensors")).unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read(to.join("snapshots/abc/model.safetensors")).unwrap(), b"weights");
        assert_eq!(seen, 7, "the blob is counted once, and the link not at all");
        assert_eq!(size_of(&root.join("hub")), 7);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// What changed while a copy was being made reaches the copy at the
    /// switch, and what was deleted meanwhile leaves it.
    #[test]
    fn a_sync_catches_up_with_what_changed_during_the_copy() {
        let root = scratch("sync");
        let from = root.join("images");
        std::fs::create_dir_all(&from).unwrap();
        std::fs::write(from.join("1.png"), b"one").unwrap();
        std::fs::write(from.join("2.png"), b"two").unwrap();
        copy_tree(&from, &root.join("to"), &mut |_| true).map_err(stop_to_string).unwrap();

        std::fs::write(from.join("3.png"), b"three").unwrap();
        std::fs::remove_file(from.join("1.png")).unwrap();
        std::fs::write(from.join("2.png"), b"two, again").unwrap();
        sync_tree(&from, &root.join("to")).unwrap();

        let mut names: Vec<_> =
            std::fs::read_dir(root.join("to")).unwrap().flatten().map(|e| e.file_name().into_string().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["2.png", "3.png"]);
        assert_eq!(std::fs::read(root.join("to/2.png")).unwrap(), b"two, again");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A cancelled copy says so rather than failing, and stops.
    #[test]
    fn a_copy_stops_when_told_to() {
        let root = scratch("cancel");
        std::fs::create_dir_all(root.join("from")).unwrap();
        std::fs::write(root.join("from/a"), b"aaaa").unwrap();
        std::fs::write(root.join("from/b"), b"bbbb").unwrap();
        let stopped = copy_tree(&root.join("from"), &root.join("to"), &mut |_| false);
        assert!(matches!(stopped, Err(Stop::Cancelled)));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_typed_path_is_full_or_starts_at_home() {
        let home = PathBuf::from(std::env::var("HOME").unwrap());
        assert_eq!(typed_path("~/kvad").unwrap(), home.join("kvad"));
        assert_eq!(typed_path(" /srv/kvad/ ").unwrap(), PathBuf::from("/srv/kvad"));
        assert_eq!(typed_path("/srv/x/../kvad").unwrap(), PathBuf::from("/srv/x/../kvad").components().collect::<PathBuf>());
        assert!(typed_path("kvad").is_err());
    }

    #[test]
    fn nesting_either_way_is_an_overlap() {
        assert!(overlaps(Path::new("/a/b"), Path::new("/a")));
        assert!(overlaps(Path::new("/a"), Path::new("/a/b")));
        assert!(overlaps(Path::new("/a"), Path::new("/a")));
        assert!(!overlaps(Path::new("/a/b"), Path::new("/a/bc")));
    }

    /// A move into a directory with something in it is refused before
    /// anything happens, and so is one into the data directory itself.
    #[test]
    fn a_plan_refuses_a_target_that_is_not_empty_or_is_the_data_itself() {
        let root = scratch("plan");
        let data = root.join("data");
        std::fs::create_dir_all(data.join("images")).unwrap();
        std::fs::write(data.join("images/1.png"), b"png").unwrap();
        let running = Running::for_data(data.clone());

        std::fs::create_dir_all(root.join("full")).unwrap();
        std::fs::write(root.join("full/x"), b"x").unwrap();
        let full = plan(&running, root.join("full").to_str().unwrap()).unwrap_err();
        assert!(full.contains("not empty"), "{full}");

        let same = plan(&running, data.to_str().unwrap()).unwrap_err();
        assert!(same.contains("already"), "{same}");
        let inside = plan(&running, data.join("images/deeper").to_str().unwrap()).unwrap_err();
        assert!(inside.contains("overlaps"), "{inside}");

        let fine = plan(&running, root.join("new").to_str().unwrap()).unwrap();
        let images = fine.steps.iter().find(|s| s.name == "images").unwrap();
        assert!(images.rename, "the same disk is a rename");
        assert_eq!(images.to, root.join("new/images"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Deleting the old copy never reaches what the server is using now.
    #[test]
    fn the_old_copy_is_not_deleted_when_it_is_in_use() {
        let root = scratch("previous");
        let data = root.join("data");
        std::fs::create_dir_all(data.join("images")).unwrap();
        let running = Running::for_data(data.clone());
        let previous =
            Previous { from: root.join("old"), to: data.clone(), paths: vec![data.join("images")], bytes: 0 };
        assert!(delete_previous(&running, &[previous]).is_err());
        assert!(data.join("images").exists());

        std::fs::create_dir_all(root.join("old/images")).unwrap();
        std::fs::write(root.join("old/images/1.png"), b"png").unwrap();
        let previous =
            Previous { from: root.join("old"), to: data.clone(), paths: vec![root.join("old/images")], bytes: 3 };
        assert_eq!(delete_previous(&running, &[previous]).unwrap(), 3);
        assert!(!root.join("old").exists(), "the emptied old directory goes too");
        let _ = std::fs::remove_dir_all(&root);
    }
}
