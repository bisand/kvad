//! `kvad-serve`: the HTTP server, and the web UI it carries.
//!
//!     kvad-serve                          loopback on 8080, no auth
//!     kvad-serve --bind 0.0.0.0:8080      needs an auth mode, or --insecure
//!     kvad-serve --config path/kvad.toml
//!
//! It loads config, opens and migrates a database, decides who a request is
//! from, manages and runs models, and serves the UI. Training, evals and
//! monitoring are still to come; see `docs/ui-plan.md`.
//!
//! # The shape of it
//!
//! * [`config`] — the few things that must be known before anything starts.
//! * [`db`] — SQLite, and the migrations that shape it.
//! * [`auth`] — who a request is from, and the seam the other modes fit into.
//! * [`engine`] — which backends this build can load.
//! * [`scheduler`] — the one owner of the engine thread, and the queue in
//!   front of it.
//! * [`models`] — what is on this machine and what the engine holds.
//! * [`chat`] / [`conversations`] — conversations, stored and served.
//! * [`openai`] — `/v1`, shaped by somebody else's documentation.
//! * [`api`] — the routing table, and what every handler shares.
//! * [`assets`] — the built UI, embedded, with everything else falling back
//!   to it so a client-side router survives a reload.

mod api;
mod assets;
mod auth;
mod chat;
mod config;
mod conversations;
mod db;
mod engine;
mod models;
mod openai;
mod scheduler;

use axum::Router;
use std::net::SocketAddr;
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Args {
    config: Option<PathBuf>,
    /// Overrides `server.bind` from the file. The common case is one address
    /// for one run, and editing a file to move a port is a poor trade.
    bind: Option<SocketAddr>,
    db: Option<PathBuf>,
    insecure: bool,
}

fn usage() -> ! {
    eprintln!(
        "kvad-serve — HTTP server and web UI for kvad

    --config PATH   configuration file (default: {})
    --bind ADDR     address to listen on, e.g. 127.0.0.1:8080
    --db PATH       SQLite database file
    --insecure      allow a non-loopback bind with no authentication
    -h, --help",
        config::default_path().display()
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args { config: None, bind: None, db: None, insecure: false };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        if matches!(flag, "-h" | "--help") {
            usage();
        }
        if flag == "--insecure" {
            a.insecure = true;
            i += 1;
            continue;
        }
        let Some(value) = argv.get(i + 1) else {
            eprintln!("missing value for {flag}");
            usage();
        };
        match flag {
            "--config" => a.config = Some(value.into()),
            "--db" => a.db = Some(value.into()),
            "--bind" => {
                a.bind = Some(value.parse().unwrap_or_else(|_| {
                    eprintln!("--bind expects an address and port, got `{value}`");
                    std::process::exit(2);
                }))
            }
            other => {
                eprintln!("unknown flag {other}");
                usage();
            }
        }
        i += 2;
    }
    a
}

#[tokio::main]
async fn main() {
    // `RUST_LOG` if it is set, otherwise our own requests and warnings from
    // everything else. A default of `info` across every dependency would bury
    // the line somebody is looking for.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kvad_serve=info,tower_http=warn,warn".into()),
        )
        .init();

    if let Err(e) = run(parse_args()).await {
        // Startup failures are the ones a person reads, so they go to stderr
        // plainly rather than through the log's formatting.
        eprintln!("kvad-serve: {e}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Res<()> {
    let config_path = args.config.clone().unwrap_or_else(config::default_path);
    let mut cfg = config::Config::load(&config_path)?;
    if let Some(bind) = args.bind {
        cfg.server.bind = bind;
    }
    if let Some(path) = args.db {
        cfg.database.path = path;
    }
    // Before the socket, not after: the check exists to stop the server
    // listening somewhere it should not, and a check after `bind` has already
    // happened is a check that ran too late.
    cfg.check(args.insecure)?;

    let db = db::Db::open(&cfg.database.path)?;
    let schema = db.version()?;
    let state = auth::State {
        auth: auth::provider(cfg.auth.mode)?.into(),
        engine: std::sync::Arc::new(scheduler::Scheduler::spawn(engine::loader())),
        started: std::time::Instant::now(),
        db,
    };

    let app = Router::new()
        .merge(api::routes())
        // Everything that is not the API is the UI, including paths that do
        // not exist as files; see `assets::serve`.
        .fallback(assets::serve)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(cfg.server.bind).await.map_err(|e| {
        format!("could not bind {}: {e}", cfg.server.bind)
    })?;
    let bound = listener.local_addr()?;

    println!("kvad-serve listening on http://{bound}");
    println!("  config     {}", config_path.display());
    println!("  database   {} (schema {schema})", cfg.database.path.display());
    println!("  auth       {}", cfg.auth.mode);
    println!(
        "  backends   {}",
        engine::available().iter().map(|c| c.id.clone()).collect::<Vec<_>>().join(", ")
    );
    if !assets::is_embedded() {
        println!("  web UI     not built into this binary — open the address above to see how");
    }

    axum::serve(listener, app).with_graceful_shutdown(interrupted()).await?;
    println!("stopped");
    Ok(())
}

/// Resolves on Ctrl-C, so an in-flight request finishes rather than being cut.
async fn interrupted() {
    let _ = tokio::signal::ctrl_c().await;
    println!();
}
