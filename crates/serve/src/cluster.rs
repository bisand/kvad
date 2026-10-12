//! The server's part in a cluster: a worker for other machines' models,
//! and the route that says what there is.
//!
//! The other part, spreading this server's own models over workers, is a
//! few lines of [`crate::engine`]: it is a way of loading a model, and
//! that is where models are loaded.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, State};
use axum::extract::State as St;
use axum::routing::get;
use axum::{Json, Router};
use kvad::cluster::{discover, key, wire, worker};

pub fn routes() -> Router<State> {
    Router::new().route("/api/cluster", get(overview))
}

/// What `[cluster]` and `[worker]` came to in this process, set once at
/// startup.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Running {
    /// The machines `[cluster] workers` names, as it names them.
    pub workers: Vec<String>,
    pub place: String,
    pub split: Option<String>,
    /// This server as a worker, if it is one.
    pub worker: Option<Listening>,
}

/// A worker that is listening.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Listening {
    pub port: u16,
    /// How a coordinator names this machine: see `kvad::cluster::make_id`.
    pub id: String,
    pub name: String,
    /// Whether the link was told: finding a worker by name is written for
    /// macOS, and elsewhere it is named by its address.
    pub advertised: bool,
}

static RUNNING: std::sync::OnceLock<Running> = std::sync::OnceLock::new();

pub fn running() -> Running {
    RUNNING.get().cloned().unwrap_or_default()
}

/// Read the two sections, start the worker if there is to be one, and
/// remember what came of it. Once, at startup.
pub fn start(cluster: &crate::config::Cluster, as_worker: &crate::config::Worker) {
    // Said at startup and not at the first load, which may be an hour off.
    if let Err(why) = kvad::cluster::plan::Mix::parse(&cluster.place) {
        tracing::warn!("[cluster] place: {why}");
    }
    if let Some(Err(why)) = cluster.split.as_deref().map(kvad::cluster::session::parse_split) {
        tracing::warn!("[cluster] split: {why}");
    }
    let listening = match as_worker.enabled {
        true => listen(as_worker.port).map_err(|why| tracing::error!("not a worker: {why}")).ok(),
        false => None,
    };
    let _ = RUNNING.set(Running { workers: cluster.workers.clone(), place: cluster.place.clone(), split: cluster.split.clone(), worker: listening });
}

/// Become a worker: the key, the id, the port, the advertisement, and a
/// thread that answers coordinators until the process ends.
fn listen(port: u16) -> Result<Listening, String> {
    let path = key::path();
    let (key, new) = key::Key::read_or_create(&path)?;
    if new {
        // Where it is and not what it is: this goes to a log, and a key in
        // a log is a key no longer.
        tracing::info!("a cluster key was made in {}; every machine of the cluster needs the same file", path.display());
    }
    let (name, id) = (kvad::cluster::hostname(), kvad::cluster::make_id()?);
    // `::` and not `0.0.0.0`: a cable's addresses are link-local IPv6, and
    // an IPv6 socket takes IPv4 as well.
    let listener = std::net::TcpListener::bind(("::", port))
        .or_else(|_| std::net::TcpListener::bind(("0.0.0.0", port)))
        .map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    let advert = discover::advertise(&name, port, &id, &kvad::cluster::build()).map_err(|why| tracing::info!("{why}")).ok();
    let advertised = advert.is_some();
    tracing::info!("{name} ({id}, kvad {}) is a worker on port {port}", kvad::cluster::build());

    std::thread::Builder::new()
        .name("kvad-worker".into())
        .spawn(move || {
            // Kept for as long as the thread is, which is as long as the
            // process: it is withdrawn when either ends.
            let _advert = advert;
            let served = worker::serve(&listener, &key, &mut worker::from_disk(), &mut |event| match event {
                worker::Event::Came(from, hello) => tracing::info!("worker: {} ({}, kvad {}) connected", from.ip(), hello.name, hello.version),
                worker::Event::Progress(message) => tracing::info!("worker: {message}"),
                worker::Event::Loaded(load, ready) => tracing::info!(
                    "worker: holding layers {}..{} of {} at {}, {:.0} MB",
                    load.layers[0],
                    load.layers[1],
                    load.repo,
                    load.precision,
                    ready.bytes as f64 / 1e6
                ),
                worker::Event::Went(from, how) => tracing::info!("worker: {} gone: {how}", from.ip()),
            });
            if let Err(e) = served {
                tracing::error!("the worker stopped: {e}");
            }
        })
        .map_err(|e| format!("no thread for the worker: {e}"))?;
    Ok(Listening { port, id, name, advertised })
}

#[derive(serde::Serialize)]
pub struct Overview {
    /// This build, as a worker and a coordinator compare it.
    build: String,
    /// Where a worker listens unless told otherwise.
    default_port: u16,
    #[serde(flatten)]
    running: Running,
    /// Whether this machine holds a cluster key, without which it can
    /// neither be a worker nor use one.
    has_key: bool,
    /// The workers the link says are there, whoever named them: found by
    /// asking, which takes a couple of seconds.
    found: Vec<Found>,
}

#[derive(serde::Serialize)]
pub struct Found {
    name: String,
    id: String,
    build: String,
    /// Whether it is this build: a coordinator refuses a worker of another.
    same_build: bool,
    /// Whether it is this very server.
    this_machine: bool,
    /// Whether `[cluster] workers` names it.
    named: bool,
    port: u16,
    addresses: Vec<String>,
}

/// `GET /api/cluster`: what this server's part in a cluster is, and who
/// else on the link says they are a worker.
pub async fn overview(_: Admin, St(_): St<State>) -> Result<Json<Overview>, Fail> {
    let running = running();
    let (ours, named) = (running.worker.as_ref().map(|w| w.id.clone()), running.workers.clone());
    let found = blocking(move || {
        Ok(discover::workers()
            .into_iter()
            .map(|w| Found {
                same_build: w.build == kvad::cluster::build(),
                this_machine: ours.as_deref() == Some(w.id.as_str()),
                named: named.iter().any(|word| w.answers_to(word)),
                name: w.name,
                id: w.id,
                build: w.build,
                port: w.port,
                addresses: w.addresses,
            })
            .collect())
    })
    .await?;
    Ok(Json(Overview {
        build: kvad::cluster::build(),
        default_port: wire::PORT,
        has_key: key::path().is_file() || std::env::var(key::ENV).is_ok_and(|k| !k.trim().is_empty()),
        running,
        found,
    }))
}
