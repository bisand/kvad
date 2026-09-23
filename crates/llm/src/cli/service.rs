//! `kvad service` — the server as something that starts at login.
//!
//! The mechanics are `kvad::daemon`'s. What this adds is the second opinion:
//! the service manager says whether the job is running, and `/api/health`
//! says whether anything answers, and `status` prints both and says when
//! they disagree. A job launchd believes is running and that does not answer
//! on its port is the failure people actually hit, and a status command that
//! only asked launchd would call it fine.

use super::out;
use crate::Args;
use kvad::client::{self, Remote, Settings, Why};
use kvad::daemon;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub const USAGE: &str = "usage: kvad service <command>

  status                is it running, where is it listening, what is loaded
  start | stop | restart
  logs [-f] [-n N]      the server's log; -f to follow it
  install [--host H] [--port P] [--force]
                        run kvad-serve at login, on H:P (default: the address
                        it already has, else server.bind in kvad.toml, else
                        127.0.0.1:5823). --force installs over the objections:
                        a non-loopback address with no auth, a taken address.
  uninstall             stop it, and stop it starting at login

`stop` is for now: it starts again at the next login. `uninstall` is for good.

`status` exits 0 when the server answers, 1 when the service manager says it
is running and nothing answers, and 3 when it is not running.";

pub fn run(args: &Args) -> Res<()> {
    let (sub, _) = super::split(args, "status");
    match sub {
        "status" => status(args),
        "start" => start(args),
        "stop" => stop(),
        "restart" => {
            stop()?;
            start(args)
        }
        "logs" => logs(args),
        "install" => install(args),
        "uninstall" => uninstall(),
        "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => super::unknown("service", other, USAGE),
    }
}

/// This machine's service address, as a server to ask.
fn this_machine() -> Res<(Remote, &'static str)> {
    let settings = Settings::read()?;
    let (address, from) = client::service_address(&settings);
    Ok((Remote::new(&address, Why::Service)?, from))
}

/// Ask a server for `/api/health`.
///
/// Three answers: the document, "it answered and wants a credential", and
/// nothing at all.
enum Health {
    Answered(Value),
    Refused(String),
    Silent(String),
}

fn health(remote: &Remote) -> Health {
    match remote.get("/api/health") {
        Ok(value) => Health::Answered(value),
        Err(e) => match e.downcast_ref::<client::Failure>() {
            Some(f) => Health::Refused(f.message.clone()),
            None => Health::Silent(e.to_string()),
        },
    }
}

/// `kvad service status`.
fn status(args: &Args) -> Res<()> {
    // Another machine's service manager is not ours to ask. Its health is.
    if let client::Choice::Remote(url) = &args.choice {
        let remote = Remote::new(url, Why::Flag)?;
        let answer = health(&remote);
        if args.json {
            out::json(&json!({ "address": remote.base, "health": health_json(&answer) }));
        } else {
            println!("address   {}", remote.base);
            print_health(&answer);
        }
        if let Health::Silent(_) = answer {
            std::process::exit(3);
        }
        return Ok(());
    }

    let state = daemon::state();
    let (remote, from) = this_machine()?;
    let answer = health(&remote);
    let answers = !matches!(answer, Health::Silent(_));

    let verdict = match (state.running, answers) {
        (_, true) => 0,
        (true, false) => 1,
        (false, false) => 3,
    };

    if args.json {
        out::json(&json!({
            "service": {
                "manager": daemon::manager_name(),
                "unit": daemon::unit_path(),
                "installed": state.installed,
                "loaded": state.loaded,
                "running": state.running,
                "pid": state.pid,
                "last_exit": state.last_exit,
                "program": state.program,
                "bind": state.bind,
            },
            "address": remote.base,
            "address_from": from,
            "health": health_json(&answer),
        }));
        std::process::exit(verdict);
    }

    let what = match (state.installed, state.loaded, state.running) {
        (false, _, _) => "not installed".to_string(),
        (true, false, _) => "installed, not loaded".to_string(),
        (true, true, false) => match &state.last_exit {
            Some(code) => format!("installed, not running (last exit: {code})"),
            None => "installed, not running".to_string(),
        },
        (true, true, true) => match state.pid {
            Some(pid) => format!("running, pid {pid}"),
            None => "running".to_string(),
        },
    };
    println!("service   {} ({}) — {what}", daemon::LABEL, daemon::manager_name());
    if let Some(program) = &state.program {
        let bind = state.bind.as_deref().map(|b| format!(" --bind {b}")).unwrap_or_default();
        println!("          {}{bind}", program.display());
    }
    println!("address   {} ({from})", remote.base);
    print_health(&answer);

    // The disagreements, which are the reason this command exists.
    println!();
    match (state.installed, state.running, &answer) {
        (_, true, Health::Silent(_)) => {
            println!(
                "{} says the service is running, and nothing answers at {}.\n\
                 It may have only just started — `kvad service status` again in a moment —\n\
                 or it is failing to bind, or crashing and being restarted.\n{}",
                daemon::manager_name(),
                remote.base,
                daemon::where_the_log_is()
            );
        }
        (true, false, Health::Answered(_) | Health::Refused(_)) | (false, _, Health::Answered(_) | Health::Refused(_)) => {
            println!(
                "Something answers at {}, and it is not the installed service: a `kvad serve`\n\
                 started by hand, most likely. Commands go to it all the same.",
                remote.base
            );
        }
        (false, false, Health::Silent(_)) => {
            println!("Not running. Run kvad-serve at login with:  kvad service install");
        }
        (true, false, Health::Silent(_)) => {
            println!("Not running. Start it with:  kvad service start");
        }
        (true, true, _) => {
            if let Some(note) = version_note(&state.program, &answer) {
                println!("{note}");
            } else {
                println!("Running, and answering. Commands go to it; --local runs one here instead.");
            }
        }
    }
    std::process::exit(verdict);
}

fn health_json(answer: &Health) -> Value {
    match answer {
        Health::Answered(h) => h.clone(),
        Health::Refused(why) => json!({ "refused": why }),
        Health::Silent(why) => json!({ "silent": why }),
    }
}

fn print_health(answer: &Health) {
    match answer {
        Health::Answered(h) => {
            println!(
                "answers   kvad-serve {}, up {}, auth {}, signed in as {}",
                out::s(&h["version"]),
                uptime(h["uptime_secs"].as_u64().unwrap_or(0)),
                out::s(&h["auth"]),
                out::s(&h["you"]["name"]),
            );
            let residents = out::items(&h["residents"]);
            if residents.is_empty() {
                println!("models    none in memory");
            }
            for (i, r) in residents.iter().enumerate() {
                println!(
                    "{}{} ({})",
                    if i == 0 { "models    " } else { "          " },
                    out::s(&r["id"]),
                    out::bytes(&r["commit"]),
                );
            }
            if let Some(depth) = h["queue_depth"].as_u64().filter(|&d| d > 0) {
                println!("queue     {depth} waiting");
            }
        }
        Health::Refused(why) => {
            println!("answers   yes, and wants a credential: {why}");
            println!("          sign in with:  kvad auth login");
        }
        Health::Silent(why) => println!("answers   no — {why}"),
    }
}

/// A service running a different version from this CLI, said once.
///
/// The usual cause is an upgrade that replaced the binaries while the
/// service kept running the old ones, and the fix is a restart.
fn version_note(program: &Option<PathBuf>, answer: &Health) -> Option<String> {
    let Health::Answered(h) = answer else { return None };
    let theirs = h["version"].as_str()?;
    let ours = env!("CARGO_PKG_VERSION");
    if theirs == ours {
        return None;
    }
    let beside = serve_beside_us();
    let other_binary = match (program, &beside) {
        (Some(p), Some(b)) if p != b => format!("\nThe service runs {}; this kvad is beside {}.", p.display(), b.display()),
        _ => String::new(),
    };
    Some(format!(
        "The service is kvad-serve {theirs} and this is kvad {ours}.{other_binary}\n\
         If the binaries were upgraded underneath it:  kvad service restart"
    ))
}

fn uptime(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m {}s", s / 60, s % 60),
        s if s < 86_400 => format!("{}h {}m", s / 3600, s % 3600 / 60),
        s => format!("{}d {}h", s / 86_400, s % 86_400 / 3600),
    }
}

/// Wait for a server to answer, for up to `budget`.
///
/// The socket opens early — before migrations have a chance to be slow, and
/// long before the autoloaded model is in memory — so this is seconds, not
/// the minute a load takes.
fn await_answer(remote: &Remote, budget: Duration) -> bool {
    let until = Instant::now() + budget;
    loop {
        if !matches!(health(remote), Health::Silent(_)) {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn start(_args: &Args) -> Res<()> {
    let (remote, _) = this_machine()?;
    match daemon::start()? {
        false => println!("{} is already running", daemon::LABEL),
        true => println!("started {}", daemon::LABEL),
    }
    match await_answer(&remote, Duration::from_secs(15)) {
        true => println!("  answering at {}", remote.base),
        false => {
            println!("  not answering at {} yet.", remote.base);
            println!("  {}", daemon::where_the_log_is());
        }
    }
    Ok(())
}

fn stop() -> Res<()> {
    match daemon::stop()? {
        true => {
            println!("stopped {}", daemon::LABEL);
            println!("  it starts again at the next login; `kvad service uninstall` stops that too");
        }
        false => println!("{} was not running", daemon::LABEL),
    }
    Ok(())
}

/// `kvad service logs`: `tail` on macOS, `journalctl` on Linux.
///
/// Handed over with `exec` rather than run as a child, for the reason
/// `kvad serve` is: Ctrl-C on `-f` reaches the program doing the following,
/// and nothing sits in between forwarding it.
fn logs(args: &Args) -> Res<()> {
    let lines = args.limit.unwrap_or(50).to_string();
    let mut command = match daemon::Manager::here() {
        daemon::Manager::Launchd => {
            let files: Vec<PathBuf> = daemon::log_paths().into_iter().filter(|p| p.exists()).collect();
            if files.is_empty() {
                println!(
                    "no log yet: nothing has been written to {}",
                    daemon::log_paths()[0].parent().map(|p| p.display().to_string()).unwrap_or_default()
                );
                return Ok(());
            }
            let mut c = std::process::Command::new("tail");
            c.arg("-n").arg(&lines);
            if args.follow {
                // -F rather than -f: it follows the name, so a log that is
                // rotated or recreated by a restart keeps being followed.
                c.arg("-F");
            }
            c.args(files);
            c
        }
        daemon::Manager::Systemd => {
            let mut c = std::process::Command::new("journalctl");
            c.args(["--user", "-u", daemon::UNIT, "-n", &lines]);
            c.arg(if args.follow { "-f" } else { "--no-pager" });
            c
        }
    };

    #[cfg(unix)]
    let failure = {
        use std::os::unix::process::CommandExt;
        command.exec()
    };
    #[cfg(not(unix))]
    let failure = match command.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => e,
    };
    Err(format!("could not read the log: {failure}").into())
}

/// The `kvad-serve` beside this binary, which is the one to install.
fn serve_beside_us() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    let path = exe.parent()?.join("kvad-serve");
    path.is_file().then_some(path)
}

fn install(args: &Args) -> Res<()> {
    let program = serve_beside_us().ok_or(
        "there is no kvad-serve beside this kvad to install. Build it once:\n\n    \
         cargo build --release -p kvad-serve\n\nand it will be found there.",
    )?;

    let settings = Settings::read()?;
    let bind = choose_bind(args, &settings)?;
    let addr: std::net::SocketAddr = bind.parse().map_err(|_| {
        format!("`{bind}` is not an address kvad-serve can listen on: it wants an IP and a port, like 127.0.0.1:5823")
    })?;

    if !args.force {
        objections(addr, &settings)?;
    }

    println!("installing {} ({})", daemon::LABEL, daemon::manager_name());
    println!("  {} --bind {bind}", program.display());
    daemon::install(&program, &bind)?;
    println!("  written to {}", daemon::unit_path().display());

    let remote = Remote::new(&reachable(addr), Why::Service)?;
    match await_answer(&remote, Duration::from_secs(15)) {
        true => println!("  answering at {}", remote.base),
        false => println!("  running, and not answering at {} yet", remote.base),
    }
    println!("  {}", daemon::where_the_log_is());
    println!("  stop it with: kvad service stop · remove it with: kvad service uninstall");
    // A user service dies with the last session unless lingering is on,
    // which is surprising enough on a headless box to be worth saying.
    if daemon::lingering() == Some(false) {
        println!(
            "  to keep it running after you log out:  sudo loginctl enable-linger {}",
            std::env::var("USER").unwrap_or_else(|_| "$USER".into())
        );
    }
    Ok(())
}

/// The address to install on: flags, else the one already installed, else
/// the config file's, else the default.
///
/// An upgrade keeps the address it had. A reinstall that quietly moved the
/// server back to the default port is a server that stopped answering where
/// the rest of the machine expects it.
fn choose_bind(args: &Args, settings: &Settings) -> Res<String> {
    let before = daemon::installed_bind().or_else(|| settings.server_bind.clone());
    let (old_host, old_port) = match &before {
        Some(b) => match b.parse::<std::net::SocketAddr>() {
            Ok(a) => (a.ip().to_string(), a.port()),
            Err(_) => ("127.0.0.1".to_string(), client::DEFAULT_PORT),
        },
        None => ("127.0.0.1".to_string(), client::DEFAULT_PORT),
    };
    let host = args.host.clone().unwrap_or(old_host);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.contains("://") || host.contains('/') {
        return Err(format!("--host `{host}` is not a host; the address on its own is enough, e.g. 127.0.0.1").into());
    }
    let port = args.port.unwrap_or(old_port);
    if port == 0 {
        return Err("--port 0 means any free port, and a service at an address nobody can predict is not a service".into());
    }
    // IPv6 gets its brackets back here and nowhere else.
    Ok(match host.contains(':') {
        true => format!("[{host}]:{port}"),
        false => format!("{host}:{port}"),
    })
}

/// The two ways an address produces a service that looks installed and
/// never serves anything.
fn objections(addr: std::net::SocketAddr, settings: &Settings) -> Res<()> {
    if !addr.ip().is_loopback() && auth_mode(settings) == "none" {
        return Err(format!(
            "{addr} is not a loopback address, and kvad-serve refuses to listen anywhere else while\n\
             auth.mode is \"none\" — everyone who could reach it would be an administrator. The\n\
             service would fail to start and be restarted for as long as it is loaded.\n\n\
             Set an auth mode in {} first (kvad.example.toml says how),\n\
             or install it anyway with --force.",
            settings.path.display()
        )
        .into());
    }

    // Whether something already holds this exact address, found by trying
    // to take it. The installed service does not count: it is stopped
    // before the new unit starts.
    let ours = daemon::state();
    let ours_there = ours.running && ours.bind.as_deref().and_then(|b| b.parse().ok()) == Some(addr);
    if !ours_there {
        if let Err(e) = std::net::TcpListener::bind(addr) {
            if e.kind() == std::io::ErrorKind::AddrInUse {
                return Err(format!(
                    "something is already listening on {addr}. Two servers cannot share an address, so\n\
                     kvad-serve would fail to bind and be restarted in a loop. Pick another with\n\
                     --host or --port, stop what is there, or install anyway with --force."
                )
                .into());
            }
        }
    }
    Ok(())
}

/// `auth.mode` from `kvad.toml`, as the server would read it.
fn auth_mode(settings: &Settings) -> String {
    std::fs::read_to_string(&settings.path)
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
        .and_then(|t| t.get("auth")?.get("mode")?.as_str().map(str::to_string))
        .unwrap_or_else(|| "none".into())
}

/// Where to connect to reach something listening on `addr`.
fn reachable(addr: std::net::SocketAddr) -> String {
    match addr.ip().is_unspecified() {
        true if addr.is_ipv4() => format!("127.0.0.1:{}", addr.port()),
        true => format!("[::1]:{}", addr.port()),
        false => addr.to_string(),
    }
}

fn uninstall() -> Res<()> {
    let path = daemon::unit_path();
    match daemon::uninstall()? {
        true => println!("removed {}\n  {} no longer starts at login", path.display(), daemon::LABEL),
        false => println!("{} is not installed", daemon::LABEL),
    }
    Ok(())
}
