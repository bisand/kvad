//! The server as a service that starts at login: a launchd agent on macOS, a
//! systemd user unit on Linux.
//!
//! This knowledge started in `install.sh`, and most of what is here was
//! learned there by getting it wrong. It lives here now so that there is one
//! copy: a shell script and a binary that both know how to stop a launchd job
//! will eventually disagree about whether it stopped. The installer asks
//! `kvad service` to do this whenever the binary it just laid down is new
//! enough to have it, and keeps its own version only for upgrading from one
//! that is not.
//!
//! Three lessons from the script, each of which is a function below:
//!
//! * `launchctl bootout` returns while the job is still on its way out, and
//!   anything that touches the label in that gap fails. [`stop`] waits for
//!   launchd to let go.
//! * `launchctl`'s exit status does not say whether the job is there:
//!   `launchctl load -w` prints "Load failed: 5: Input/output error" and then
//!   exits 0. So nothing here infers success from a status; it *asks*
//!   afterwards, with [`state`].
//! * A service is stopped before the unit under it is rewritten, or it keeps
//!   running the old one. [`install`] stops first.
//!
//! What this module does not know is whether the server is *answering*. That
//! is an HTTP question and [`crate::client`] asks it; `kvad service status`
//! puts the two answers side by side, because a job launchd believes is
//! running and that does not answer on its port is the failure people
//! actually hit.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// launchd's name for the job. The same one `install.sh` has always used, so
/// a service it installed is the service this finds.
pub const LABEL: &str = "net.kvad.serve";

/// systemd's.
pub const UNIT: &str = "kvad-serve.service";

/// The label every function here acts on: [`LABEL`], except in the test that
/// drives a real launchd, which must never touch the service somebody is
/// actually running.
fn label() -> String {
    #[cfg(test)]
    if let Ok(label) = std::env::var("KVAD_TEST_LABEL") {
        return label;
    }
    LABEL.to_string()
}

/// How long a stop or a start is given to be seen to have happened.
///
/// Five seconds is what the installer settled on for a bootout. An install
/// that hangs on a job that is never leaving is worse than one that says so.
const SETTLE: Duration = Duration::from_secs(5);

/// Which service manager this machine has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Manager {
    Launchd,
    Systemd,
}

impl Manager {
    /// Decided by the platform rather than by looking: kvad builds for macOS
    /// and Linux, and on each there is one answer.
    pub fn here() -> Manager {
        if cfg!(target_os = "macos") {
            Manager::Launchd
        } else {
            Manager::Systemd
        }
    }
}

/// What the service manager has, and what the unit file says.
#[derive(Debug, Clone, Default)]
pub struct State {
    /// The unit file is on disk.
    pub installed: bool,
    /// The manager has the job: loaded into launchd, enabled in systemd. A
    /// file on disk that the manager does not have is common enough (a stop
    /// by hand, a bootout that did not take) that the two are kept apart.
    pub loaded: bool,
    pub running: bool,
    pub pid: Option<u32>,
    /// How it last ended, as the manager says it. A service failing to start
    /// in a loop is visible here and nowhere else short of the log.
    pub last_exit: Option<String>,
    /// The binary the unit runs.
    pub program: Option<PathBuf>,
    /// The address the unit passes it, as `HOST:PORT`.
    pub bind: Option<String>,
}

/// Where the unit file goes.
pub fn unit_path() -> PathBuf {
    match Manager::here() {
        Manager::Launchd => home().join("Library/LaunchAgents").join(format!("{}.plist", label())),
        Manager::Systemd => config_home().join("systemd/user").join(UNIT),
    }
}

/// Where a launchd agent's output goes: standard output, then standard error.
///
/// Empty on Linux, where the journal has it and `journalctl` is the way in.
pub fn log_paths() -> Vec<PathBuf> {
    match Manager::here() {
        Manager::Launchd => {
            let dir = home().join("Library/Logs/kvad");
            vec![dir.join("kvad-serve.log"), dir.join("kvad-serve.err")]
        }
        Manager::Systemd => Vec::new(),
    }
}

/// Everything the manager and the unit file will say, without changing
/// anything.
pub fn state() -> State {
    let path = unit_path();
    let text = std::fs::read_to_string(&path).ok();
    let (program, bind) = match &text {
        Some(t) => read_unit(Manager::here(), t),
        None => (None, None),
    };
    let mut state = State { installed: text.is_some(), program, bind, ..State::default() };

    match Manager::here() {
        Manager::Launchd => {
            if let Some(out) = output("launchctl", &["print", &target()]) {
                state.loaded = true;
                read_launchctl(&out, &mut state);
            }
        }
        Manager::Systemd => {
            let props = output(
                "systemctl",
                &["--user", "show", UNIT, "-p", "ActiveState,MainPID,UnitFileState,ExecMainStatus"],
            );
            if let Some(out) = props {
                read_systemctl(&out, &mut state);
            }
        }
    }
    state
}

/// The address an installed service was given, read off its unit file.
///
/// What the rest of the machine should look for, and what an upgrade keeps:
/// a reinstall that quietly moved the server back to the default port is a
/// server that stopped answering where everything expects it.
pub fn installed_bind() -> Option<String> {
    let text = std::fs::read_to_string(unit_path()).ok()?;
    read_unit(Manager::here(), &text).1
}

/// Start the installed service, and see that it started.
///
/// Returns false when it was already running, which is not an error but is
/// worth a different sentence.
pub fn start() -> Res<bool> {
    let before = state();
    if !before.installed {
        return Err(format!(
            "the service is not installed ({} does not exist). Install it with:\n    kvad service install",
            unit_path().display()
        )
        .into());
    }
    if before.running {
        return Ok(false);
    }

    match Manager::here() {
        Manager::Launchd => {
            if before.loaded {
                // Loaded and not running: it exited, or was never kicked.
                // A second bootstrap would fail on a label that is there.
                quiet("launchctl", &["kickstart", &target()]);
            } else {
                let unit = unit_path();
                let unit = unit.to_string_lossy();
                if !quiet("launchctl", &["bootstrap", &domain(), &unit]) {
                    quiet("launchctl", &["load", "-w", &unit]);
                }
            }
        }
        Manager::Systemd => {
            systemctl(&["start", UNIT])?;
        }
    }

    // Asked, not inferred from anything above.
    if settle(|| state().running) {
        return Ok(true);
    }
    let after = state();
    let mut why = match (after.loaded, &after.last_exit) {
        (false, _) => format!("{} did not take the job.", manager_name()),
        (true, Some(code)) => format!("it was started and is not running; it last exited with {code}."),
        (true, None) => "it was started and is not running.".to_string(),
    };
    why.push_str(&format!("\n{}", where_the_log_is()));
    Err(why.into())
}

/// Stop the service, and wait until the manager has let go of it.
///
/// On macOS this is a `bootout`: the job is unloaded, and launchd loads it
/// again at the next login because the file is still in `LaunchAgents`. On
/// Linux it is a `stop`, and the unit stays enabled. Either way "stop" means
/// for now; [`uninstall`] means for good.
///
/// Returns false when there was nothing to stop.
pub fn stop() -> Res<bool> {
    let before = state();
    match Manager::here() {
        Manager::Launchd => {
            if !before.loaded {
                return Ok(false);
            }
            quiet("launchctl", &["bootout", &target()]);
            // `bootout` returns while the job is still leaving. Anything that
            // touches the label in that gap — a bootstrap straight after, as
            // a restart does — fails.
            if !settle(|| !state().loaded) {
                return Err(format!(
                    "{} is still loaded after {} seconds. Boot it out yourself with:\n    launchctl bootout {}",
                    label(),
                    SETTLE.as_secs(),
                    target()
                )
                .into());
            }
        }
        Manager::Systemd => {
            if !before.running {
                return Ok(false);
            }
            systemctl(&["stop", UNIT])?;
            if !settle(|| !state().running) {
                return Err(format!(
                    "{UNIT} is still running after {} seconds. Stop it yourself with:\n    systemctl --user stop kvad-serve",
                    SETTLE.as_secs()
                )
                .into());
            }
        }
    }
    Ok(true)
}

/// Write the unit, hand it to the manager, and see that it is running.
///
/// `program` is the `kvad-serve` to run, and `bind` the address to give it.
/// Whatever was installed before is stopped first, so that a rewritten unit
/// is the one that runs.
pub fn install(program: &Path, bind: &str) -> Res<()> {
    stop()?;
    let path = unit_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    }
    let text = match Manager::here() {
        Manager::Launchd => {
            for log in log_paths() {
                if let Some(dir) = log.parent() {
                    std::fs::create_dir_all(dir)
                        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
                }
            }
            plist(program, bind)
        }
        Manager::Systemd => systemd_unit(program, bind),
    };
    std::fs::write(&path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;

    match Manager::here() {
        Manager::Launchd => {
            let unit = path.to_string_lossy();
            if !quiet("launchctl", &["bootstrap", &domain(), &unit]) {
                quiet("launchctl", &["load", "-w", &unit]);
            }
            // Asked rather than inferred, for the reason at the top of this
            // module: `load -w` fails and exits 0.
            if !settle(|| state().loaded) {
                return Err(format!(
                    "wrote {}, but launchd did not take it. Load it yourself with:\n    launchctl bootstrap {} {}",
                    path.display(),
                    domain(),
                    path.display()
                )
                .into());
            }
        }
        Manager::Systemd => {
            if systemctl(&["daemon-reload"]).is_err() {
                return Err(format!(
                    "wrote {}, but systemd --user is not available here. Start the server yourself with:\n    {}",
                    path.display(),
                    program.display()
                )
                .into());
            }
            systemctl(&["enable", "--now", UNIT])?;
        }
    }

    if !settle(|| state().running) {
        let exit = state().last_exit.map(|c| format!(" It last exited with {c}.")).unwrap_or_default();
        return Err(format!(
            "installed, and it is not running.{exit}\n{}",
            where_the_log_is()
        )
        .into());
    }
    Ok(())
}

/// Stop the service and remove its unit. Returns false if there was none.
pub fn uninstall() -> Res<bool> {
    let path = unit_path();
    if !path.exists() {
        // Not installed. A job loaded with no file behind it is still worth
        // taking down, and costs nothing to try.
        let _ = stop();
        return Ok(false);
    }
    match Manager::here() {
        Manager::Launchd => {
            stop()?;
        }
        Manager::Systemd => {
            // `disable --now` rather than `stop`: the unit is going, and an
            // enabled unit whose file vanished leaves a dangling symlink.
            let _ = systemctl(&["disable", "--now", UNIT]);
        }
    }
    std::fs::remove_file(&path).map_err(|e| format!("could not remove {}: {e}", path.display()))?;
    if Manager::here() == Manager::Systemd {
        let _ = systemctl(&["daemon-reload"]);
    }
    Ok(true)
}

/// Whether a systemd user service outlives the last login session here.
///
/// It does not by default, which on a headless machine means the server dies
/// when the ssh session that started it closes. `None` where the question
/// does not arise, or `loginctl` cannot answer it.
pub fn lingering() -> Option<bool> {
    if Manager::here() != Manager::Systemd {
        return None;
    }
    let user = output("id", &["-un"])?;
    let out = output("loginctl", &["show-user", user.trim(), "-p", "Linger", "--value"])?;
    Some(out.trim() == "yes")
}

/// How to read the service's log, as a sentence.
pub fn where_the_log_is() -> String {
    match Manager::here() {
        Manager::Launchd => {
            let logs = log_paths();
            format!("Its log is {}, and `kvad service logs` shows it.", logs[0].display())
        }
        Manager::Systemd => {
            "Its log is in the journal: `kvad service logs`, or journalctl --user -u kvad-serve.".into()
        }
    }
}

/// The manager's own name, for sentences.
pub fn manager_name() -> &'static str {
    match Manager::here() {
        Manager::Launchd => "launchd",
        Manager::Systemd => "systemd",
    }
}

// ---------------------------------------------------------------------------
// Unit files
// ---------------------------------------------------------------------------

/// The launchd agent, as `install.sh` has always written it.
///
/// `KeepAlive` on an unsuccessful exit only, so a server that crashes comes
/// back and one that was asked to stop stays stopped. `Interactive` because
/// it is somebody's model server rather than a background chore, and macOS
/// throttles the other kind.
fn plist(program: &Path, bind: &str) -> String {
    let logs = log_paths();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>              <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{program}</string>
        <string>--bind</string>
        <string>{bind}</string>
    </array>
    <key>RunAtLoad</key>          <true/>
    <key>KeepAlive</key>
    <dict><key>SuccessfulExit</key><false/></dict>
    <key>ProcessType</key>        <string>Interactive</string>
    <key>StandardOutPath</key>    <string>{out}</string>
    <key>StandardErrorPath</key>  <string>{err}</string>
    <key>WorkingDirectory</key>   <string>{home}</string>
</dict>
</plist>
"#,
        label = xml(&label()),
        program = xml(&program.to_string_lossy()),
        bind = xml(bind),
        out = xml(&logs[0].to_string_lossy()),
        err = xml(&logs[1].to_string_lossy()),
        home = xml(&home().to_string_lossy()),
    )
}

fn systemd_unit(program: &Path, bind: &str) -> String {
    let program = program.to_string_lossy();
    // systemd splits `ExecStart` on whitespace unless the word is quoted.
    let program = match program.contains(char::is_whitespace) {
        true => format!("\"{program}\""),
        false => program.into_owned(),
    };
    format!(
        "[Unit]\n\
         Description=kvad — HTTP server and web UI\n\
         After=network.target\n\
         \n\
         [Service]\n\
         ExecStart={program} --bind {bind}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    )
}

fn xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// The program and the `--bind` address a unit file names.
///
/// Every `<string>` in a plist, in order, rather than a line-by-line read: a
/// plist somebody reformatted by hand can put the whole array on one line
/// and still be the plist launchd is running. The program is the first
/// string after `ProgramArguments`, and the address is the one after
/// `--bind`.
fn read_unit(manager: Manager, text: &str) -> (Option<PathBuf>, Option<String>) {
    let words: Vec<String> = match manager {
        Manager::Launchd => {
            let Some(at) = text.find("ProgramArguments") else { return (None, None) };
            let mut words = Vec::new();
            let mut rest = &text[at..];
            while let Some(open) = rest.find("<string>") {
                rest = &rest[open + "<string>".len()..];
                let Some(close) = rest.find("</string>") else { break };
                words.push(
                    rest[..close].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"),
                );
                rest = &rest[close..];
            }
            words
        }
        Manager::Systemd => {
            let Some(line) = text.lines().find_map(|l| l.trim().strip_prefix("ExecStart=")) else {
                return (None, None);
            };
            split_exec(line)
        }
    };
    let program = words.first().map(PathBuf::from);
    let bind = words.iter().enumerate().find_map(|(i, w)| match w.strip_prefix("--bind") {
        Some("") => words.get(i + 1).cloned(),
        Some(v) => v.strip_prefix('=').map(str::to_string),
        None => None,
    });
    (program, bind)
}

/// An `ExecStart` line in words, honouring double quotes.
fn split_exec(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            }
            c => word.push(c),
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// Read `launchctl print`'s answer.
///
/// Only the lines at the top level of the job, one tab in: the same keys
/// appear again inside nested blocks, and an endpoint's `state` is not the
/// job's.
fn read_launchctl(out: &str, state: &mut State) {
    for line in out.lines() {
        let Some(line) = line.strip_prefix('\t') else { continue };
        if line.starts_with('\t') {
            continue;
        }
        if let Some(v) = line.strip_prefix("state = ") {
            state.running = v.trim() == "running";
        } else if let Some(v) = line.strip_prefix("pid = ") {
            state.pid = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("last exit code = ") {
            let v = v.trim();
            if !v.starts_with('(') {
                state.last_exit = Some(format!("status {v}"));
            }
        }
    }
}

fn read_systemctl(out: &str, state: &mut State) {
    for line in out.lines() {
        let Some((key, value)) = line.split_once('=') else { continue };
        match key {
            "ActiveState" => state.running = matches!(value, "active" | "activating" | "reloading"),
            "MainPID" => state.pid = value.parse().ok().filter(|&p: &u32| p != 0),
            "UnitFileState" => state.loaded = matches!(value, "enabled" | "enabled-runtime" | "linked"),
            "ExecMainStatus" => {
                if value != "0" && !value.is_empty() {
                    state.last_exit = Some(format!("status {value}"));
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Talking to the manager
// ---------------------------------------------------------------------------

/// `gui/UID`: the per-user domain a login session's agents live in.
fn domain() -> String {
    // SAFETY: getuid cannot fail and has no preconditions.
    format!("gui/{}", unsafe { libc::getuid() })
}

/// `gui/UID/net.kvad.serve`: this one job.
fn target() -> String {
    format!("{}/{}", domain(), label())
}

/// Run a command and keep what it printed, if it succeeded.
fn output(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Run a command for its effect, saying only whether it claimed success.
/// Callers check the effect themselves; see the top of this module.
fn quiet(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// `systemctl --user`, with what it said on failure.
fn systemctl(args: &[&str]) -> Res<()> {
    let out = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .map_err(|e| format!("could not run systemctl: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let said = String::from_utf8_lossy(&out.stderr);
    Err(format!("systemctl --user {} failed: {}", args.join(" "), said.trim()).into())
}

/// Poll until `done` or [`SETTLE`] runs out.
fn settle(mut done: impl FnMut() -> bool) -> bool {
    let until = Instant::now() + SETTLE;
    loop {
        if done() {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

fn config_home() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".config"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `install.sh` writes and what this writes have to read back the
    /// same, or an upgrade forgets the address somebody chose.
    #[test]
    fn a_unit_this_writes_reads_back_as_what_was_written() {
        let program = Path::new("/home/ada/.local/bin/kvad-serve");
        let (p, b) = read_unit(Manager::Launchd, &plist(program, "0.0.0.0:5823"));
        assert_eq!(p.as_deref(), Some(program));
        assert_eq!(b.as_deref(), Some("0.0.0.0:5823"));

        let (p, b) = read_unit(Manager::Systemd, &systemd_unit(program, "[::1]:9000"));
        assert_eq!(p.as_deref(), Some(program));
        assert_eq!(b.as_deref(), Some("[::1]:9000"));
    }

    /// A plist reformatted onto one line is still the plist launchd runs.
    #[test]
    fn a_plist_on_one_line_still_has_its_address() {
        let one_line = "<plist><dict><key>Label</key><string>net.kvad.serve</string>\
                        <key>ProgramArguments</key><array><string>/opt/kvad-serve</string>\
                        <string>--bind</string><string>127.0.0.1:7000</string></array></dict></plist>";
        let (p, b) = read_unit(Manager::Launchd, one_line);
        assert_eq!(p.as_deref(), Some(Path::new("/opt/kvad-serve")));
        assert_eq!(b.as_deref(), Some("127.0.0.1:7000"));
    }

    #[test]
    fn exec_start_keeps_a_quoted_path_whole() {
        let unit = systemd_unit(Path::new("/home/a b/kvad-serve"), "127.0.0.1:5823");
        let (p, b) = read_unit(Manager::Systemd, &unit);
        assert_eq!(p.as_deref(), Some(Path::new("/home/a b/kvad-serve")));
        assert_eq!(b.as_deref(), Some("127.0.0.1:5823"));
        assert_eq!(split_exec("x --bind=1.2.3.4:5"), ["x", "--bind=1.2.3.4:5"]);
        let (_, b) = read_unit(Manager::Systemd, "ExecStart=x --bind=1.2.3.4:5");
        assert_eq!(b.as_deref(), Some("1.2.3.4:5"));
    }

    /// The job's own `state`, not an endpoint's inside a nested block.
    #[test]
    fn launchctl_is_read_at_the_top_level_only() {
        let out = "gui/501/net.kvad.serve = {\n\
                   \tactive count = 1\n\
                   \tstate = running\n\
                   \tendpoints = {\n\
                   \t\tstate = waiting\n\
                   \t}\n\
                   \tpid = 4242\n\
                   \tlast exit code = (never exited)\n\
                   }\n";
        let mut s = State::default();
        read_launchctl(out, &mut s);
        assert!(s.running);
        assert_eq!(s.pid, Some(4242));
        assert_eq!(s.last_exit, None);

        let mut s = State::default();
        read_launchctl("x = {\n\tstate = not running\n\tlast exit code = 1\n}\n", &mut s);
        assert!(!s.running);
        assert_eq!(s.last_exit.as_deref(), Some("status 1"));
    }

    #[test]
    fn systemctl_show_is_read_by_key() {
        let mut s = State::default();
        read_systemctl(
            "ActiveState=active\nMainPID=812\nUnitFileState=enabled\nExecMainStatus=0\n",
            &mut s,
        );
        assert!(s.running && s.loaded);
        assert_eq!(s.pid, Some(812));
        assert_eq!(s.last_exit, None);

        let mut s = State::default();
        read_systemctl("ActiveState=failed\nMainPID=0\nUnitFileState=enabled\nExecMainStatus=1\n", &mut s);
        assert!(!s.running);
        assert_eq!(s.pid, None);
        assert_eq!(s.last_exit.as_deref(), Some("status 1"));
    }

    /// The whole life of a service, against the real service manager.
    ///
    /// Ignored, because it loads a job into this login's launchd: run it by
    /// hand with `cargo test -p kvad --lib daemon -- --ignored`. It cannot
    /// touch a kvad service that is installed for real — the label is its
    /// own, `HOME` is a directory of its own, and what it runs is `sleep`.
    /// macOS only for now: on Linux the unit name would need the same
    /// treatment as the label.
    #[test]
    #[ignore = "drives this login's real launchd; run by hand"]
    #[cfg(target_os = "macos")]
    fn a_service_installs_starts_stops_and_goes() {
        let home = std::env::temp_dir().join(format!("kvad-daemon-test-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        std::env::set_var("HOME", &home);
        std::env::set_var("KVAD_TEST_LABEL", "net.kvad.serve.test");
        let program = home.join("fake-serve");
        std::fs::write(&program, "#!/bin/sh\nexec sleep 600\n").unwrap();
        std::fs::set_permissions(&program, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();

        // Whatever an earlier failed run left behind.
        let _ = uninstall();

        install(&program, "127.0.0.1:5897").expect("install");
        let s = state();
        assert!(s.installed && s.loaded && s.running, "{s:?}");
        assert_eq!(s.bind.as_deref(), Some("127.0.0.1:5897"));
        assert_eq!(s.program.as_deref(), Some(program.as_path()));
        assert_eq!(installed_bind().as_deref(), Some("127.0.0.1:5897"));

        assert!(!start().expect("start while running"), "a running service started again");
        assert!(stop().expect("stop"));
        let s = state();
        assert!(s.installed && !s.loaded && !s.running, "{s:?}");
        assert!(!stop().expect("stop while stopped"), "a stopped service stopped again");

        assert!(start().expect("start"));
        assert!(state().running);

        // Installing over a running service replaces it rather than failing
        // on a label that is already there.
        install(&program, "127.0.0.1:5896").expect("reinstall");
        assert_eq!(state().bind.as_deref(), Some("127.0.0.1:5896"));

        assert!(uninstall().expect("uninstall"));
        let s = state();
        assert!(!s.installed && !s.loaded && !s.running, "{s:?}");
        assert!(!uninstall().expect("uninstall twice"));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn paths_are_escaped_in_a_plist() {
        let text = plist(Path::new("/tmp/a&b/kvad-serve"), "127.0.0.1:5823");
        assert!(text.contains("/tmp/a&amp;b/kvad-serve"));
        let (p, _) = read_unit(Manager::Launchd, &text);
        assert_eq!(p.as_deref(), Some(Path::new("/tmp/a&b/kvad-serve")));
    }
}
