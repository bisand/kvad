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

mod accounts;
mod api;
mod assets;
mod auth;
mod bench;
mod chat;
mod compare;
mod config;
mod conversations;
mod datasets;
mod db;
mod engine;
mod evals;
mod jobs;
mod machine;
mod metrics;
mod models;
mod monitoring;
mod oidc;
mod openai;
mod playground;
mod scheduler;
mod secret;
mod watching;
mod training;
mod users;

use axum::Router;
use std::net::SocketAddr;
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// How long per-request rows are kept. Long enough to answer "what happened
/// last Tuesday", short enough that the table stays readable.
const KEEP_REQUESTS_DAYS: u32 = 7;

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
    // The tail is kept where a browser can read it as well as printed, so
    // that "what did the server say" is answerable from a machine nobody is
    // sitting at.
    let metrics = std::sync::Arc::new(metrics::Metrics::new());
    tracing_subscriber::fmt()
        .with_writer(watching::LogTail(std::sync::Arc::clone(&metrics)))
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kvad_serve=info,tower_http=warn,warn".into()),
        )
        .init();

    if let Err(e) = run(parse_args(), metrics).await {
        // Startup failures are the ones a person reads, so they go to stderr
        // plainly rather than through the log's formatting.
        eprintln!("kvad-serve: {e}");
        std::process::exit(1);
    }
}

async fn run(args: Args, metrics: std::sync::Arc<metrics::Metrics>) -> Res<()> {
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
    // Sessions that ran out are refused whether or not they are still rows;
    // this only keeps the table from growing forever.
    let swept = users::sweep(&db).unwrap_or(0);

    // A mode with accounts and no accounts needs a way in. One token, printed
    // where only somebody with access to this terminal can read it, gone when
    // it is used or when the server restarts.
    // Anything that was running when the server last stopped is not running
    // now, whatever its row says.
    let job_runner = std::sync::Arc::new(jobs::Jobs::new(db.clone()));
    let orphans = job_runner.abandon_orphans().unwrap_or(0);

    let setup = std::sync::Arc::new(auth::Setup::default());
    let accounts = users::count(&db)?;
    // Not in OIDC mode: the first account there is made by the first person
    // on the allow-list who signs in, so a setup token would be a second way
    // in that nobody needs.
    let first_run =
        matches!(cfg.auth.mode, config::Mode::Local | config::Mode::Basic) && accounts == 0;

    let state = auth::State {
        auth: auth::provider(cfg.auth.mode, &cfg.auth.oidc)?.into(),
        engine: std::sync::Arc::new(scheduler::Scheduler::spawn(engine::loader())),
        jobs: std::sync::Arc::clone(&job_runner),
        metrics: std::sync::Arc::clone(&metrics),
        setup: std::sync::Arc::clone(&setup),
        oidc: std::sync::Arc::new((cfg.auth.oidc.clone(), oidc::Flows::default())),
        started: std::time::Instant::now(),
        db,
    };

    let app = Router::new()
        .merge(api::routes())
        // Everything that is not the API is the UI, including paths that do
        // not exist as files; see `assets::serve`.
        .fallback(assets::serve)
        // Outside the trace layer, so the time recorded is the time the
        // client waited rather than the time the handler ran.
        .layer(axum::middleware::from_fn_with_state(state.clone(), watching::timed))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state.clone());

    // Requests go to memory as they happen and to the database in batches:
    // a lock on the database in the middle of every response would be a lock
    // on the server.
    tokio::spawn({
        let (metrics, db) = (std::sync::Arc::clone(&metrics), state.db.clone());
        async move {
            let mut tick = tokio::time::interval(metrics::FLUSH_EVERY);
            let mut since_prune = 0u32;
            loop {
                tick.tick().await;
                let (m, d) = (std::sync::Arc::clone(&metrics), db.clone());
                // An hour's worth of ticks between prunes. Cheap either way;
                // this just keeps it off the flush path.
                since_prune += 1;
                let prune = since_prune >= 720;
                if prune {
                    since_prune = 0;
                }
                let _ = tokio::task::spawn_blocking(move || {
                    let _ = m.flush(&d);
                    if prune {
                        let _ = metrics::Metrics::prune(&d, KEEP_REQUESTS_DAYS);
                    }
                })
                .await;
            }
        }
    });

    let listener = tokio::net::TcpListener::bind(cfg.server.bind).await.map_err(|e| {
        format!("could not bind {}: {e}", cfg.server.bind)
    })?;
    let bound = listener.local_addr()?;

    // Through `tracing` as well as stdout, so that the log tail on the
    // Monitoring page starts with something rather than with nothing.
    tracing::info!("listening on http://{bound}, auth {}, schema {schema}", cfg.auth.mode);
    println!("kvad-serve listening on http://{bound}");
    println!("  config     {}", config_path.display());
    println!("  database   {} (schema {schema})", cfg.database.path.display());
    if cfg.auth.mode == config::Mode::Oidc {
        println!("  provider   {}", cfg.auth.oidc.issuer);
    }
    println!("  auth       {}{}", cfg.auth.mode, match accounts {
        0 => String::new(),
        1 => " · 1 account".into(),
        n => format!(" · {n} accounts"),
    });
    println!(
        "  backends   {}",
        engine::available().iter().map(|c| c.id.clone()).collect::<Vec<_>>().join(", ")
    );
    if !assets::is_embedded() {
        println!("  web UI     not built into this binary — open the address above to see how");
    }

    if first_run {
        // After the address, so that the two things somebody needs — where to
        // go and what to type when they get there — are together at the end.
        println!();
        println!("This server has no accounts yet. Open the address above and use this");
        println!("one-time setup token to make the first one:");
        println!();
        println!("    {}", setup.issue());
        println!();
        println!("It is not stored anywhere, and a restart issues a new one.");
    }
    if swept > 0 {
        tracing::info!("swept {swept} expired session(s)");
    }
    if orphans > 0 {
        tracing::info!("{orphans} job(s) were interrupted by a restart and are marked failed");
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
