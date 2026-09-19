//! Starting runs, watching them, and the text they are trained on.
//!
//! Everything slow here is a [`crate::jobs`] job, so a request either starts
//! one or follows one. Nothing in this file blocks for longer than a database
//! read.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
use crate::datasets::{self, Dataset};
use crate::jobs::{self, Job, TrainParams, Update};
use axum::extract::{DefaultBodyLimit, Path, Query, State as St};
use axum::response::sse::Event;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/{id}", get(job).delete(cancel_job))
        .route("/api/jobs/{id}/events", get(job_events))
        .route("/api/train", post(start))
        .route("/api/train/options", get(options))
        .route("/api/datasets", get(list_datasets).post(upload))
        .route("/api/datasets/{id}", get(dataset).delete(remove_dataset))
        .route("/api/datasets/{id}/check", get(check))
        // Uploads are text and the store has its own limit; axum's default of
        // 2 MB would refuse a corpus long before that.
        .layer(DefaultBodyLimit::max(datasets::MAX_BYTES + 1024))
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct Limit {
    #[serde(default)]
    limit: Option<usize>,
}

async fn list_jobs(_: Identity, St(state): St<State>, Query(q): Query<Limit>) -> Result<Json<Vec<Job>>, Fail> {
    let jobs = state.jobs.clone();
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    blocking(move || jobs.list(limit)).await.map(Json)
}

#[derive(serde::Serialize)]
pub struct Whole {
    #[serde(flatten)]
    job: Job,
    metrics: Vec<jobs::Metric>,
    samples: Vec<jobs::Sample>,
}

async fn job(_: Identity, St(state): St<State>, Path(id): Path<i64>) -> Result<Json<Whole>, Fail> {
    let jobs = state.jobs.clone();
    let found = blocking(move || {
        Ok(match jobs.get(id)? {
            None => None,
            Some(job) => Some(Whole { metrics: jobs.metrics(id)?, samples: jobs.samples(id)?, job }),
        })
    })
    .await?;
    found.map(Json).ok_or_else(|| Fail::missing(format!("there is no job {id}")))
}

/// Ask a job to stop.
///
/// A `DELETE` because that is what the plan says cancelling is, and because
/// the job's row stays: a cancelled run is history, not an absence.
async fn cancel_job(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    let jobs = state.jobs.clone();
    let job = blocking(move || jobs.get(id)).await?;
    let Some(job) = job else { return Err(Fail::missing(format!("there is no job {id}"))) };
    if !job.live() {
        return Err(Fail::bad(format!("job {id} already {}", job.state)));
    }
    Ok(Json(json!({ "asked": state.jobs.cancel(id) })))
}

/// Follow a job: everything that has happened, then everything that does.
///
/// The subscription is taken *before* the history is read, so an update that
/// lands between the two is seen rather than lost. It may therefore be seen
/// twice, which is why metrics and samples are keyed by step — a repeat
/// replaces rather than appends.
async fn job_events(
    _: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<impl IntoResponse, Fail> {
    let live = state.jobs.watch(id);
    let jobs = state.jobs.clone();
    let (job, past) =
        blocking(move || Ok((jobs.get(id)?, jobs::history(&jobs, id)?))).await?;
    let Some(job) = job else { return Err(Fail::missing(format!("there is no job {id}"))) };

    let (events, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        for update in past {
            if events.send(crate::models::sse("update", &update)).await.is_err() {
                return;
            }
        }
        let Some(mut live) = live else {
            // Already over by the time we looked. The history was the whole
            // of it; say how it ended and close.
            let _ = events
                .send(crate::models::sse(
                    "update",
                    &Update::Ended { state: job.state, error: job.error },
                ))
                .await;
            return;
        };
        loop {
            match live.recv().await {
                Ok(update) => {
                    let ended = matches!(update, Update::Ended { .. });
                    if events.send(crate::models::sse("update", &update)).await.is_err() || ended {
                        return;
                    }
                }
                // Lagged: this watcher fell behind and lost updates. The
                // database has the metrics, so a reload recovers; keep going.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => return,
            }
        }
    });

    Ok(crate::models::stream(rx))
}

// ---------------------------------------------------------------------------
// Starting a run
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
pub struct Options {
    sizes: Vec<Size>,
    /// Models a run could continue from: trained here, with a character
    /// tokeniser beside their weights.
    continuable: Vec<String>,
    defaults: Defaults,
    /// How many threads this machine has, so the picker has a ceiling.
    cores: usize,
    /// Whether a run is going, which is the one thing that stops another.
    training: bool,
}

#[derive(serde::Serialize)]
pub struct Size {
    name: &'static str,
    shape: String,
}

#[derive(serde::Serialize)]
pub struct Defaults {
    steps: usize,
    lr: f32,
    eval_every: usize,
    threads: usize,
    sample: usize,
    seed: u64,
}

async fn options(_: Admin, St(state): St<State>) -> Result<Json<Options>, Fail> {
    let jobs = state.jobs.clone();
    let (continuable, training) =
        blocking(move || Ok((datasets::continuable_models(), jobs.training()))).await?;
    let d = nanograd::text::Training::default();
    Ok(Json(Options {
        sizes: kvad::train::SIZES.iter().map(|s| Size { name: s.name, shape: s.shape() }).collect(),
        continuable,
        defaults: Defaults {
            steps: d.steps,
            lr: d.lr,
            eval_every: d.eval_every,
            threads: d.threads,
            sample: 160,
            seed: 1337,
        },
        cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        training,
    }))
}

#[derive(serde::Deserialize)]
pub struct StartRequest {
    dataset: i64,
    /// Where the result is kept. Without it, `from` trains in place.
    #[serde(default)]
    name: Option<String>,
    /// Continue this model instead of starting one.
    #[serde(default)]
    from: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    lr: Option<f32>,
    #[serde(default)]
    eval_every: Option<usize>,
    #[serde(default)]
    threads: Option<usize>,
    #[serde(default)]
    sample: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

async fn start(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<StartRequest>,
) -> Result<Json<Job>, Fail> {
    let db = state.db.clone();
    let jobs = state.jobs.clone();
    let owner = who.0.id;

    blocking(move || {
        let size = match &body.size {
            None => &kvad::train::SIZES[kvad::train::DEFAULT_SIZE],
            Some(name) => kvad::train::size(name).ok_or_else(|| {
                let names: Vec<_> = kvad::train::SIZES.iter().map(|s| s.name).collect();
                format!("`{name}` is not a size; there are {}", names.join(", "))
            })?,
        };
        // Named before anything slow happens, because every one of these is a
        // different thing to go and fix.
        if body.name.is_none() && body.from.is_none() {
            return Err("a new model needs a name".into());
        }
        if let Some(name) = &body.name {
            if !kvad::weights::is_model_name(name) {
                return Err(format!("`{name}` is not a model name: one word, no `/`").into());
            }
        }
        let dataset = datasets::get(&db, body.dataset)?
            .ok_or_else(|| format!("there is no dataset {}", body.dataset))?;
        let data = datasets::file_for(&db, body.dataset)?;

        // The check that would otherwise fail an hour in — or, with `--from`,
        // immediately but after somebody has already committed to a run.
        if let Some(from) = &body.from {
            if !datasets::continuable(from) {
                return Err(format!(
                    "`{from}` is not a model trained here, so there is no vocabulary to continue"
                )
                .into());
            }
            let text = std::fs::read_to_string(&data)?;
            let missing = datasets::unseen(from, &text)?;
            if !missing.is_empty() {
                let shown: String = missing.iter().take(12).collect();
                return Err(format!(
                    "`{}` contains {} character{} `{from}` has no token for ({shown}{}). \
                     A vocabulary is fixed at first training: train a new model on both texts \
                     instead.",
                    dataset.name,
                    missing.len(),
                    if missing.len() == 1 { "" } else { "s" },
                    if missing.len() > 12 { "…" } else { "" }
                )
                .into());
            }
        }

        let d = nanograd::text::Training::default();
        let training = nanograd::text::Training {
            steps: body.steps.unwrap_or(d.steps).clamp(1, 1_000_000),
            lr: body.lr.unwrap_or(d.lr),
            eval_every: body.eval_every.unwrap_or(d.eval_every).max(1),
            threads: body.threads.unwrap_or(d.threads).clamp(1, 256),
            ..d
        };
        let params = TrainParams {
            name: body.name.clone(),
            from: body.from.clone(),
            dataset: dataset.id,
            dataset_name: dataset.name.clone(),
            size: size.name.to_string(),
            steps: training.steps,
            lr: training.lr,
            eval_every: training.eval_every,
            threads: training.threads,
            sample: body.sample.unwrap_or(160),
            seed: body.seed.unwrap_or(1337),
        };
        let opts = kvad::train::Options {
            data,
            from: body.from.clone(),
            name: body.name.clone(),
            size,
            training,
            seed: params.seed,
            sample: params.sample,
            temperature: 0.8,
            cancel: None, // `jobs::train` puts its own flag here.
        };
        jobs::train(&jobs, opts, params, owner)
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

async fn list_datasets(_: Admin, St(state): St<State>) -> Result<Json<Vec<Dataset>>, Fail> {
    let db = state.db.clone();
    blocking(move || datasets::list(&db)).await.map(Json)
}

#[derive(serde::Deserialize)]
pub struct Named {
    name: String,
}

/// Upload a corpus.
///
/// The body is the text, not a multipart form: what is being sent is one
/// file's contents and nothing else, and a browser can read a file and post
/// it in three lines.
async fn upload(
    who: Admin,
    St(state): St<State>,
    Query(q): Query<Named>,
    text: String,
) -> Result<Json<Dataset>, Fail> {
    let db = state.db.clone();
    let owner = who.0.id;
    blocking(move || datasets::save(&db, q.name.trim(), &text, owner))
        .await
        .map(Json)
        .map_err(|e| Fail::bad(e.1))
}

#[derive(serde::Serialize)]
pub struct WithText {
    #[serde(flatten)]
    dataset: Dataset,
    /// The first of it, for a look before committing an hour to it.
    preview: String,
}

async fn dataset(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<WithText>, Fail> {
    let db = state.db.clone();
    let (dataset, text) =
        blocking(move || datasets::read(&db, id)).await.map_err(|e| Fail::missing(e.1))?;
    Ok(Json(WithText { preview: text.chars().take(2000).collect(), dataset }))
}

async fn remove_dataset(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    match blocking(move || datasets::delete(&db, id)).await? {
        true => Ok(Json(json!({ "deleted": id }))),
        false => Err(Fail::missing(format!("there is no dataset {id}"))),
    }
}

#[derive(serde::Deserialize)]
pub struct Against {
    /// A model trained here, which a run might continue.
    model: String,
}

/// What this dataset would cost the named model.
///
/// The answer somebody wants *before* choosing both and pressing a button:
/// every character the model has no token for, not just the first.
async fn check(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
    Query(q): Query<Against>,
) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let answer = blocking(move || {
        let (dataset, text) = datasets::read(&db, id)?;
        let missing = datasets::unseen(&q.model, &text)?;
        Ok(json!({
            "dataset": dataset.name,
            "model": q.model,
            "continuable": datasets::continuable(&q.model),
            "unseen": missing.iter().collect::<String>(),
            "unseen_count": missing.len(),
        }))
    })
    .await
    .map_err(|e| Fail::bad(e.1))?;
    Ok(Json(answer))
}
