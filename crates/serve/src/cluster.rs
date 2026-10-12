//! The server's part in a cluster: a worker for other machines' models,
//! and the route that says what there is.
//!
//! The other part, spreading this server's own models over workers, is a
//! few lines of [`crate::engine`]: it is a way of loading a model, and
//! that is where models are loaded.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, State};
use axum::extract::{Query, State as St};
use axum::routing::get;
use axum::{Json, Router};
use crate::scheduler::Scheduler;
use kvad::cluster::stage::{Fault, Stage};
use kvad::cluster::{discover, key, session, wire, worker};
use kvad::service::Backend;

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

/// What the load now running has of the budget, for the placing of a
/// model across machines to count as this machine's memory; zero when no
/// load is running. Process-wide and not an argument for the reason
/// `kvad::experts::set_room` is: one load runs at a time, and the loader's
/// signature is every loader's.
static ROOM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_room(bytes: Option<u64>) {
    ROOM.store(bytes.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

pub fn room() -> Option<u64> {
    Some(ROOM.load(std::sync::atomic::Ordering::Relaxed)).filter(|&bytes| bytes > 0)
}

/// Whether a load of `repo` at `backend` is one `[cluster]` would place
/// across machines: workers are named, the backend is a CPU's, and the
/// model is a language model of an architecture that can be cut. One not
/// on this disk yet has no config to ask, and is tried.
pub fn spreads(repo: &str, backend: Backend) -> bool {
    let Backend::Cpu(_) = backend else { return false };
    if running().workers.is_empty() || kvad::dit::is_one(repo) || kvad::dit::is_clips(repo) {
        return false;
    }
    let name = kvad::weights::model_id(repo);
    let local = kvad::hub::find_local(&name).or_else(|| kvad::hub::find_trained(&name));
    !local.and_then(|model| model.arch).is_some_and(|arch| !arch.splits())
}

/// The model of this name loaded across machines now, if it is: what of
/// it is on this machine is in there.
pub fn here(repo: &str, backend: Backend) -> Option<session::Report> {
    let Backend::Cpu(precision) = backend else { return None };
    let (name, precision) = (kvad::weights::model_id(repo), precision.to_string());
    session::live().into_iter().find(|report| report.repo == name && report.precision == precision)
}

/// Whose budget a worker's layers are counted against: this server's
/// scheduler, said once it exists, which is after the worker is listening.
static HOST: std::sync::OnceLock<std::sync::Arc<Scheduler>> = std::sync::OnceLock::new();

pub fn host(scheduler: std::sync::Arc<Scheduler>) {
    let _ = HOST.set(scheduler);
}

/// What a `Load` will take here, weights and cache, from what this disk
/// knows of the model: nothing, for one it has yet to fetch.
fn sizes(load: &wire::Load) -> Option<(u64, u64)> {
    let local = kvad::hub::find_local(&load.repo).or_else(|| kvad::hub::find_trained(&load.repo))?;
    let spec = kvad::hub::model_file(&local.path, "config.json").and_then(|config| kvad::model::Spec::from_json(&config).ok())?;
    let precision = kvad::quant::Precision::parse(&load.precision)?;
    let count = load.layers[1].saturating_sub(load.layers[0]);
    let weights = precision.weight_bytes(local.params?) * count as u64 / spec.n_layer.max(1) as u64;
    let context = match load.context {
        0 => spec.n_ctx,
        said => said.min(spec.n_ctx),
    };
    Some((weights, spec.cache.bytes_as(count, context, 4, 4) as u64))
}

/// Stages from this machine's disk, as any worker makes them, counted
/// against this server's budget for as long as they are held: asked for
/// before anything is read where the size can be told, so that a server
/// full of its own models says so and loads nothing, and set to what the
/// layers came to once they are in memory.
fn budgeted() -> worker::Factory {
    let mut from_disk = worker::from_disk();
    Box::new(move |load, progress, watch| {
        let what = format!("layers {}..{} of {} at {}, for another machine", load.layers[0], load.layers[1], load.repo, load.precision);
        let ask = sizes(load).map_or(0, |(weights, cache)| weights + cache);
        let lodged = HOST.get().map(|host| host.lodge(&what, ask)).transpose()?;
        let stage = from_disk(load, progress, watch)?;
        if let Some(lodged) = &lodged {
            // The files are here now, whether they were or not.
            lodged.resize(stage.holds().1 as u64 + sizes(load).map_or(0, |(_, cache)| cache));
        }
        Ok(Box::new(Lodging { stage, _lodged: lodged }) as Box<dyn Stage>)
    })
}

/// A stage, and its place in the budget: both go together.
struct Lodging {
    stage: Box<dyn Stage>,
    _lodged: Option<crate::scheduler::Lodged>,
}

impl Stage for Lodging {
    fn shard(&self) -> kvad::model::Shard {
        self.stage.shard()
    }
    fn run(&mut self, seq: u64, pos0: usize, m: usize, xs: Vec<f32>) -> Result<Vec<f32>, Fault> {
        self.stage.run(seq, pos0, m, xs)
    }
    fn truncate(&mut self, seq: u64, len: usize) -> Result<usize, Fault> {
        self.stage.truncate(seq, len)
    }
    fn holds(&self) -> (usize, usize) {
        self.stage.holds()
    }
    fn inside(&mut self) -> Option<wire::Spent> {
        self.stage.inside()
    }
    fn place(&self) -> kvad::cluster::stage::Place {
        self.stage.place()
    }
    fn elsewhere(&self) -> bool {
        self.stage.elsewhere()
    }
}

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
            let served = worker::serve_with(&listener, &key, &mut budgeted(), &mut |event| match event {
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
            }, &|| HOST.get().map(|host| host.left()));
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
    /// asking, which takes a couple of seconds. `None` when not asked.
    found: Option<Vec<Found>>,
    /// The models loaded across machines now: each stage, where it is,
    /// and where its time has gone since the model was loaded.
    models: Vec<kvad::cluster::session::Report>,
}

#[derive(serde::Deserialize)]
pub struct Asking {
    /// Whether to ask the network who is a worker. Yes unless told not
    /// to: a page that looks every few seconds has no use for an answer
    /// that takes three.
    find: Option<bool>,
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
pub async fn overview(_: Admin, St(_): St<State>, Query(asking): Query<Asking>) -> Result<Json<Overview>, Fail> {
    let running = running();
    let (ours, named) = (running.worker.as_ref().map(|w| w.id.clone()), running.workers.clone());
    let find = asking.find.unwrap_or(true);
    let found = blocking(move || {
        if !find {
            return Ok(None);
        }
        Ok(Some(discover::workers()
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
            .collect()))
    })
    .await?;
    Ok(Json(Overview {
        build: kvad::cluster::build(),
        default_port: wire::PORT,
        has_key: key::path().is_file() || std::env::var(key::ENV).is_ok_and(|k| !k.trim().is_empty()),
        running,
        found,
        models: kvad::cluster::session::live(),
    }))
}
