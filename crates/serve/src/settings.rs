//! The server's own settings, changed from the web UI, and a restart to put
//! them in force.
//!
//! `kvad.toml` stays what [`crate::config`] says it is: read once, before the
//! socket and the database, and never again. Nothing here changes a setting
//! in the running server. What changes is the file, and a restart reads it —
//! the same restart a person would do by hand, done by the server itself so
//! that the page that changed the file can also apply it.
//!
//! Only `[server]` is edited as settings here. The auth mode is shown and not
//! offered: it decides who may use this page, so it is not this page's to
//! change. `[data] dir` is written too, but only as the last step of moving
//! the data there, which is [`crate::storage`]'s; changing it alone would
//! start the server empty somewhere else.
//!
//! A save is refused rather than written when the file it would leave behind
//! is one the server could not start from. That is the whole risk of a
//! settings page for a service: a restart into a config that fails its own
//! check, and a supervisor starting it again every ten seconds, with the page
//! that could fix it gone.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, State};
use crate::config::{Config, Server};
use axum::extract::State as St;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/settings", get(read).put(write))
        .route("/api/restart", post(restart))
}

/// How long a restart waits for requests in flight before it goes anyway.
///
/// A graceful shutdown waits for every open response, and some never end on
/// their own: the page asking for the restart holds a log tail and a job's
/// event stream open. Five seconds lets an answer that is being written
/// finish, and nothing that has no end hold the restart hostage.
const GRACE: Duration = Duration::from_secs(5);

/// What this process was started with, and the way to ask it to start again.
pub struct Running {
    /// The file the settings come from, whether or not it exists.
    pub path: PathBuf,
    /// The settings in force, after the command line has had its say.
    pub server: Server,
    /// `--bind`, which wins over the file's `bind`. Worth saying on the page:
    /// the launchd service passes one, so changing `bind` there changes
    /// nothing until the service is installed without it.
    pub bind_flag: Option<SocketAddr>,
    /// `--insecure`, which the check a save runs has to know about.
    pub insecure: bool,
    pub data_dir: PathBuf,
    pub database: PathBuf,
    /// Whether `[database] path` or `--db` named the database, which then
    /// stays where it is when the data moves.
    pub database_named: bool,
    pub auth: String,
    /// A move of the data, if one is under way or has just ended.
    pub storage: crate::storage::Mover,
    restart: tokio::sync::Notify,
    restarting: AtomicBool,
}

impl Running {
    pub fn new(
        path: PathBuf,
        cfg: &Config,
        bind_flag: Option<SocketAddr>,
        insecure: bool,
        data_dir: PathBuf,
    ) -> Self {
        Running {
            path,
            server: cfg.server.clone(),
            bind_flag,
            insecure,
            data_dir,
            database: cfg.database_path(),
            database_named: cfg.database.path.is_some(),
            auth: cfg.auth.mode.to_string(),
            storage: crate::storage::Mover::default(),
            restart: tokio::sync::Notify::new(),
            restarting: AtomicBool::new(false),
        }
    }

    /// Whether moving the data takes the database with it: it is
    /// `kvad.db` in the data directory because nothing said otherwise.
    pub fn database_moves(&self) -> bool {
        !self.database_named && self.database == self.data_dir.join("kvad.db")
    }

    /// Whether a restart has been asked for. `main` reads this when the
    /// server stops, to know whether to start again.
    pub fn is_restarting(&self) -> bool {
        self.restarting.load(Ordering::SeqCst)
    }

    /// Resolves when a restart has been asked for; one of the two things
    /// that end the server, beside Ctrl-C.
    pub async fn restart_asked(&self) {
        self.restart.notified().await
    }

    /// The file as the server would read it at its next start, command line
    /// included, and checked the way startup checks it.
    fn startable(&self, cfg: &Config) -> Result<(), String> {
        let mut cfg = cfg.clone();
        if let Some(bind) = self.bind_flag {
            cfg.server.bind = bind;
        }
        cfg.check(self.insecure).map_err(|e| e.to_string())
    }
}

/// In tests, a server nobody will restart, reading a file that is not there.
impl Default for Running {
    fn default() -> Self {
        let cfg = Config::default();
        let path = std::env::temp_dir().join("kvad-no-such-settings.toml");
        Running::new(path, &cfg, None, false, std::env::temp_dir())
    }
}

#[cfg(test)]
impl Running {
    /// A server whose data is in `data`, with its database there too.
    pub fn for_data(data: PathBuf) -> Self {
        Running {
            database: data.join("kvad.db"),
            database_named: false,
            path: data.join("kvad.toml"),
            data_dir: data,
            ..Running::default()
        }
    }
}

/// The `[server]` settings, as the page edits them.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Values {
    pub autoload: bool,
    pub load_on_request: bool,
    /// `None` is this machine's usable memory.
    pub memory_gb: Option<f64>,
    /// `None` is `memory::DEFAULT_CONTEXT`.
    pub context: Option<usize>,
    pub bind: SocketAddr,
}

impl From<&Server> for Values {
    fn from(s: &Server) -> Self {
        Values {
            autoload: s.autoload,
            load_on_request: s.load_on_request,
            memory_gb: s.memory_gb,
            context: s.context,
            bind: s.bind,
        }
    }
}

#[derive(serde::Serialize)]
struct Settings {
    path: PathBuf,
    exists: bool,
    /// What the file says, with the defaults for what it leaves out. `None`
    /// when it cannot be read, and then `problem` says why.
    saved: Option<Values>,
    problem: Option<String>,
    /// What this process is running with. Where it differs from `saved`, a
    /// restart is what would change it.
    running: Values,
    /// What an unset `memory_gb` and `context` come to on this machine.
    defaults: Defaults,
    bind_flag: Option<SocketAddr>,
    /// Shown, not offered; see the module comment.
    fixed: Fixed,
    /// Whether this build can restart itself.
    can_restart: bool,
}

#[derive(serde::Serialize)]
struct Defaults {
    memory_gb: Option<f64>,
    context: usize,
}

#[derive(serde::Serialize)]
struct Fixed {
    auth: String,
    data_dir: PathBuf,
    database: PathBuf,
}

/// The file's text, or empty when there is no file. Every other failure is
/// an error: a file that exists and cannot be read is not the same as none.
fn read_text(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("could not read {}: {e}", path.display())),
    }
}

fn parse(path: &Path, text: &str) -> Result<Config, String> {
    toml::from_str(text).map_err(|e| format!("{} does not parse: {e}", path.display()))
}

fn describe(running: &Running) -> Settings {
    let (exists, saved, problem) = match read_text(&running.path) {
        Ok(None) => (false, Some(Values::from(&Server::default())), None),
        Ok(Some(text)) => match parse(&running.path, &text) {
            Ok(cfg) => (true, Some(Values::from(&cfg.server)), None),
            Err(why) => (true, None, Some(why)),
        },
        Err(why) => (true, None, Some(why)),
    };
    Settings {
        path: running.path.clone(),
        exists,
        saved,
        problem,
        running: Values::from(&running.server),
        defaults: Defaults {
            memory_gb: kvad::machine::usable_memory_cached().map(|b| b as f64 / 1e9),
            context: crate::memory::DEFAULT_CONTEXT,
        },
        bind_flag: running.bind_flag,
        fixed: Fixed {
            auth: running.auth.clone(),
            data_dir: running.data_dir.clone(),
            database: running.database.clone(),
        },
        can_restart: cfg!(unix),
    }
}

async fn read(_: Admin, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let running = Arc::clone(&state.settings);
    let settings = blocking(move || Ok(describe(&running))).await?;
    Ok(Json(serde_json::to_value(settings).map_err(|e| Fail::internal(e.to_string()))?))
}

async fn write(
    _: Admin,
    St(state): St<State>,
    Json(want): Json<Values>,
) -> Result<Json<serde_json::Value>, Fail> {
    let running = Arc::clone(&state.settings);
    let outcome = tokio::task::spawn_blocking(move || save(&running, &want).map(|()| describe(&running)))
        .await
        .map_err(|e| Fail::internal(format!("a background task failed: {e}")))?;
    match outcome {
        Ok(settings) => Ok(Json(serde_json::to_value(settings).map_err(|e| Fail::internal(e.to_string()))?)),
        Err(why) => Err(Fail::bad(why)),
    }
}

/// Write `want` into the file's `[server]` table, or say why not.
fn save(running: &Running, want: &Values) -> Result<(), String> {
    if let Some(gb) = want.memory_gb {
        if !(gb.is_finite() && gb > 0.0) {
            return Err("memory for models has to be a number of gigabytes above zero".into());
        }
    }
    if want.context == Some(0) {
        return Err("the context charged to each model has to be at least one token".into());
    }

    let path = &running.path;
    let text = read_text(path)?.unwrap_or_default();
    let before = parse(path, &text)?;
    let had = Values::from(&before.server);

    // Only what changed is touched, so a key the file leaves out stays out
    // when it is not being changed, and its comment stays where it was.
    let literal = |v: &dyn std::fmt::Debug| format!("{v:?}");
    let mut edits: Vec<(&str, Option<String>)> = Vec::new();
    if want.autoload != had.autoload {
        edits.push(("autoload", Some(want.autoload.to_string())));
    }
    if want.load_on_request != had.load_on_request {
        edits.push(("load_on_request", Some(want.load_on_request.to_string())));
    }
    if want.memory_gb != had.memory_gb {
        edits.push(("memory_gb", want.memory_gb.map(|gb| literal(&gb))));
    }
    if want.context != had.context {
        edits.push(("context", want.context.map(|n| n.to_string())));
    }
    if want.bind != had.bind {
        edits.push(("bind", Some(format!("\"{}\"", want.bind))));
    }
    if edits.is_empty() {
        return Ok(());
    }

    let edited = edit_table(&text, "server", &edits);
    // The edit is by lines, so check it the only way that counts: read the
    // result back. Anything laid out in a way the edit did not expect — a
    // dotted `server.bind` at the top, an inline table — shows up here as a
    // file that does not say what was asked, and is refused rather than
    // written.
    let after = parse(path, &edited)
        .map_err(|e| format!("{e}; the change was not written. Edit the file by hand"))?;
    let untouched = |c: &Config| format!("{:?} {:?} {:?} {:?}", c.data, c.database, c.auth, c.client);
    if Values::from(&after.server) != *want || untouched(&after) != untouched(&before) {
        return Err(format!(
            "{} is laid out in a way this page cannot edit safely; the change was not written. \
             Edit the file by hand",
            path.display()
        ));
    }
    running.startable(&after).map_err(|why| format!("the server could not start with that: {why}"))?;

    // A new address has to be one this machine will listen on, or the
    // restart ends in a server that cannot start. The address in use now
    // needs no test, and could not pass one: this process holds it.
    if running.bind_flag.is_none() && want.bind != running.server.bind {
        std::net::TcpListener::bind(want.bind)
            .map_err(|e| format!("nothing here can listen on {}: {e}", want.bind))?;
    }

    replace(path, &edited)
}

/// Write `text` to `path` by renaming a finished file over it, so that a
/// crash halfway through leaves the old file rather than half of the new one.
/// The old file's permissions are kept: it may hold an OIDC client secret.
fn replace(path: &Path, text: &str) -> Result<(), String> {
    let fail = |what: &str, e: std::io::Error| format!("could not {what} {}: {e}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| fail("make the directory for", e))?;
    }
    let partial = path.with_extension("toml.partial");
    std::fs::write(&partial, text).map_err(|e| fail("write beside", e))?;
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&partial, meta.permissions());
    }
    std::fs::rename(&partial, path).map_err(|e| fail("replace", e))
}

/// The table a line opens, if it is a `[table]` header.
fn header(line: &str) -> Option<&str> {
    let line = line.trim();
    let rest = line.strip_prefix('[').filter(|r| !r.starts_with('['))?;
    Some(rest.split(']').next()?.trim())
}

/// The key a line sets, if it is a `key = value` line.
fn key(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.starts_with('#') || line.starts_with('[') {
        return None;
    }
    let (k, _) = line.split_once('=')?;
    Some(k.trim().trim_matches('"'))
}

/// A trailing `# comment` on a `key = value` line, outside any quotes.
fn trailing_comment(line: &str) -> Option<&str> {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return Some(&line[i..]),
            _ => {}
        }
    }
    None
}

/// Set or remove keys in one table of a TOML document, leaving every other
/// line as it was. `None` removes the key.
///
/// By lines rather than through a TOML editor, because the edits are scalar
/// keys in one table and the result is read back and compared before it is
/// written (see [`save`]); a layout this does not understand is refused
/// there, not mangled here.
fn edit_table(text: &str, table: &str, edits: &[(&str, Option<String>)]) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let start = match lines.iter().position(|l| header(l) == Some(table)) {
        Some(h) => h + 1,
        None => {
            if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                lines.push(String::new());
            }
            lines.push(format!("[{table}]"));
            lines.len()
        }
    };
    for (name, value) in edits {
        let end = lines[start..].iter().position(|l| header(l).is_some()).map_or(lines.len(), |i| start + i);
        let at = (start..end).find(|&i| key(&lines[i]) == Some(*name));
        match (at, value) {
            (Some(i), Some(value)) => {
                let line = &lines[i];
                let indent = &line[..line.len() - line.trim_start().len()];
                let comment = trailing_comment(line).map(|c| format!("  {c}")).unwrap_or_default();
                lines[i] = format!("{indent}{name} = {value}{comment}");
            }
            (Some(i), None) => {
                lines.remove(i);
            }
            (None, Some(value)) => {
                // After the table's last key, so it lands above any comment
                // block that belongs to the table after it.
                let after = (start..end).rev().find(|&i| key(&lines[i]).is_some()).map_or(start, |i| i + 1);
                lines.insert(after, format!("{name} = {value}"));
            }
            (None, None) => {}
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// `kvad.toml` as it would be with `[data] dir` set to `to`, checked the
/// way a save is: it reads back as exactly that, every other table
/// unchanged, and the server could start from it.
fn with_data_dir(running: &Running, to: &Path) -> Result<String, String> {
    let path = &running.path;
    let text = read_text(path)?.unwrap_or_default();
    let before = parse(path, &text)?;
    let written = to.to_str().ok_or_else(|| format!("{} is not a path kvad.toml can hold", to.display()))?;
    if written.contains(['"', '\\']) || written.chars().any(char::is_control) {
        return Err(format!("{} has a quote, backslash or control character in it, which kvad.toml cannot hold here", to.display()));
    }
    let edited = edit_table(&text, "data", &[("dir", Some(format!("\"{written}\"")))]);
    let after = parse(path, &edited)
        .map_err(|e| format!("{e}; the data was not moved. Edit the file by hand"))?;
    let untouched = |c: &Config| format!("{:?} {:?} {:?} {:?}", c.server, c.database, c.auth, c.client);
    if after.data.dir.as_deref() != Some(written) || untouched(&after) != untouched(&before) {
        return Err(format!(
            "{} is laid out in a way this page cannot edit safely; the data was not moved. \
             Set [data] dir by hand",
            path.display()
        ));
    }
    running.startable(&after).map_err(|why| format!("the server could not start with that: {why}"))?;
    Ok(edited)
}

/// Whether kvad.toml can take `[data] dir = to`, asked before a move starts.
pub fn data_dir_edit(running: &Running, to: &Path) -> Result<(), String> {
    with_data_dir(running, to).map(|_| ())
}

/// Write `[data] dir = to` into kvad.toml: the last step of a move.
pub fn set_data_dir(running: &Running, to: &Path) -> Result<(), String> {
    let edited = with_data_dir(running, to)?;
    replace(&running.path, &edited)
}

async fn restart(_: Admin, St(state): St<State>) -> Result<(StatusCode, Json<serde_json::Value>), Fail> {
    if !cfg!(unix) {
        return Err(Fail::conflict("this build cannot restart itself; restart the service by hand"));
    }
    if state.settings.storage.busy() {
        return Err(Fail::conflict("the data is being moved; the server restarts when that is done"));
    }
    let settings = Arc::clone(&state.settings);
    crate::api::blocking_or(move || begin_restart(settings).map_err(Fail::conflict)).await?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "restarting": true }))))
}

/// Check that the server can start from kvad.toml, and restart into it.
/// Needs to be inside the tokio runtime, which it spawns the restart on.
pub fn begin_restart(settings: Arc<Running>) -> Result<(), String> {
    // The file it is about to read has to be one it can start from, or this
    // is a restart into a server that exits at once.
    let again = |why: String| format!("the server could not start again: {why}");
    let cfg = Config::load(&settings.path).map_err(|e| again(e.to_string()))?;
    settings.startable(&cfg).map_err(again)?;

    if settings.restarting.swap(true, Ordering::SeqCst) {
        return Err("a restart is already under way".into());
    }
    tracing::info!("restarting, as asked through the API");
    tokio::spawn(async move {
        // Long enough for this answer to leave.
        tokio::time::sleep(Duration::from_millis(300)).await;
        settings.restart.notify_one();
        tokio::time::sleep(GRACE).await;
        tracing::info!("requests still open after {}s; restarting without them", GRACE.as_secs());
        exec_self();
    });
    Ok(())
}

/// Replace this process with a fresh start of the same program, with the
/// same arguments.
///
/// `exec` rather than exiting and being started again, because only a
/// supervisor would start it again, and not every server has one: the launchd
/// unit restarts a server that exits with an error and not one that exits
/// cleanly, and a server started in a terminal has nobody. The process id
/// stays the same, so launchd and systemd see one process throughout.
///
/// The program is `argv[0]` rather than [`std::env::current_exe`]: after an
/// upgrade has replaced the binary, Linux names the running one
/// `kvad-serve (deleted)`, and the point of a restart after an upgrade is the
/// new one.
pub fn exec_self() -> ! {
    static ONCE: AtomicBool = AtomicBool::new(false);
    if ONCE.swap(true, Ordering::SeqCst) {
        // The other caller is already replacing the process.
        loop {
            std::thread::park();
        }
    }
    println!("restarting");
    let error = exec();
    eprintln!("kvad-serve: could not restart: {error}");
    std::process::exit(1)
}

#[cfg(unix)]
fn exec() -> std::io::Error {
    use std::os::unix::process::CommandExt;
    let mut args = std::env::args_os();
    let program = args.next().map(PathBuf::from).or_else(|| std::env::current_exe().ok());
    match program {
        Some(program) => std::process::Command::new(program).args(args).exec(),
        None => std::io::Error::other("cannot tell which program this is"),
    }
}

#[cfg(not(unix))]
fn exec() -> std::io::Error {
    std::io::Error::other("restarting in place needs a Unix exec")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str, text: Option<&str>) -> Running {
        let dir = std::env::temp_dir().join(format!("kvad-settings-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kvad.toml");
        if let Some(text) = text {
            std::fs::write(&path, text).unwrap();
        }
        let cfg = Config::load(&path).unwrap();
        Running::new(path, &cfg, None, false, dir)
    }

    fn values(r: &Running) -> Values {
        Values::from(&r.server)
    }

    /// A key that is changed is changed in place, and everything around it —
    /// comments, other keys, other tables — is left as it was written.
    #[test]
    fn a_change_touches_only_its_own_line() {
        let text = "# mine\n[server]\n# why it is on\nautoload = true   # really\nload_on_request = true\n\n[auth]\nmode = \"none\"\n";
        let r = scratch("in-place", Some(text));
        let want = Values { autoload: false, ..values(&r) };
        save(&r, &want).unwrap();
        let written = std::fs::read_to_string(&r.path).unwrap();
        assert_eq!(
            written,
            "# mine\n[server]\n# why it is on\nautoload = false  # really\nload_on_request = true\n\n[auth]\nmode = \"none\"\n"
        );
    }

    #[test]
    fn a_missing_file_or_table_is_made() {
        let r = scratch("made", None);
        let want = Values { memory_gb: Some(24.5), context: Some(8192), ..values(&r) };
        save(&r, &want).unwrap();
        assert_eq!(std::fs::read_to_string(&r.path).unwrap(), "[server]\nmemory_gb = 24.5\ncontext = 8192\n");

        let r = scratch("table", Some("[auth]\nmode = \"none\"\n"));
        save(&r, &Values { autoload: true, ..values(&r) }).unwrap();
        assert_eq!(
            std::fs::read_to_string(&r.path).unwrap(),
            "[auth]\nmode = \"none\"\n\n[server]\nautoload = true\n"
        );
    }

    /// Unsetting goes back to the default by taking the line out, not by
    /// writing the default's value in, which would stop tracking it.
    #[test]
    fn unsetting_removes_the_key() {
        let r = scratch("unset", Some("[server]\nmemory_gb = 20.0\nautoload = true\n"));
        save(&r, &Values { memory_gb: None, ..values(&r) }).unwrap();
        assert_eq!(std::fs::read_to_string(&r.path).unwrap(), "[server]\nautoload = true\n");
    }

    /// The check startup runs, run before the file is written: an address
    /// anybody can reach, with nobody asked who they are, is refused here
    /// rather than at the restart.
    #[test]
    fn a_file_the_server_would_refuse_is_not_written() {
        let text = "[server]\nautoload = true\n";
        let r = scratch("refused", Some(text));
        let open: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let error = save(&r, &Values { bind: open, ..values(&r) }).unwrap_err();
        assert!(error.contains("could not start"), "{error}");
        assert_eq!(std::fs::read_to_string(&r.path).unwrap(), text, "the file was changed anyway");
    }

    /// A layout the line edit does not understand is caught by reading the
    /// result back, and nothing is written.
    #[test]
    fn a_layout_it_cannot_edit_is_left_alone() {
        let text = "server.autoload = true\n";
        let r = scratch("dotted", Some(text));
        let error = save(&r, &Values { autoload: false, ..values(&r) }).unwrap_err();
        assert!(error.contains("by hand"), "{error}");
        assert_eq!(std::fs::read_to_string(&r.path).unwrap(), text);
    }

    #[test]
    fn nonsense_numbers_are_refused() {
        let r = scratch("numbers", None);
        assert!(save(&r, &Values { memory_gb: Some(0.0), ..values(&r) }).is_err());
        assert!(save(&r, &Values { memory_gb: Some(f64::NAN), ..values(&r) }).is_err());
        assert!(save(&r, &Values { context: Some(0), ..values(&r) }).is_err());
        assert!(!r.path.exists(), "a refused save wrote a file");
    }

    /// An address something else is listening on is refused, because the
    /// restart would end in a server that cannot bind.
    #[test]
    fn an_address_in_use_is_refused() {
        let r = scratch("in-use", None);
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let error = save(&r, &Values { bind: taken.local_addr().unwrap(), ..values(&r) }).unwrap_err();
        assert!(error.contains("listen on"), "{error}");
    }
}
