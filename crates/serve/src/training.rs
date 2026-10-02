//! Starting runs, watching them, and the text they are trained on.
//!
//! Everything slow here is a [`crate::jobs`] job, so a request either starts
//! one or follows one. Nothing in this file blocks for longer than a database
//! read.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
use crate::datasets::{self, Dataset};
use crate::jobs::{self, Job, LoraParams, TrainParams, Update};
use axum::extract::{DefaultBodyLimit, Path, Query, State as St};
use axum::http::header;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde_json::json;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/jobs", get(list_jobs))
        .route("/api/jobs/{id}", get(job).delete(cancel_job))
        .route("/api/jobs/{id}/events", get(job_events))
        .route("/api/jobs/{id}/samples/{step}/{prompt}", get(job_sample))
        .route("/api/train", post(start))
        .route("/api/train/options", get(options))
        .route("/api/datasets", get(list_datasets).post(upload))
        .route("/api/datasets/crawl", post(start_crawl))
        .route("/api/datasets/{id}", get(dataset).delete(remove_dataset))
        .route("/api/datasets/{id}/check", get(check))
        .route("/api/datasets/{id}/search", get(search))
        .route("/api/datasets/{id}/pictures", get(pictures))
        .route("/api/datasets/{id}/pictures/{file}", get(picture))
        // Uploads are text and the store has its own limit; axum's default of
        // 2 MB would refuse a corpus long before that.
        .layer(DefaultBodyLimit::max(datasets::MAX_BYTES + 1024))
        // A picture is larger than a text may be, so these have a limit of
        // their own, and are added after the layer above so as not to be
        // under it.
        .merge(
            Router::new()
                .route("/api/datasets/pictures/{name}", post(keep_pictures))
                .route("/api/datasets/pictures/{name}", delete(drop_pictures))
                .route("/api/datasets/pictures/{name}/{file}", put(stage_picture))
                .layer(DefaultBodyLimit::max(datasets::MAX_PICTURE_BYTES + 1024)),
        )
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
    /// A LoRA run's measurements and the pictures it drew; empty for
    /// everything else.
    measures: Vec<jobs::Measure>,
    pictures: Vec<jobs::Drawn>,
}

async fn job(_: Identity, St(state): St<State>, Path(id): Path<i64>) -> Result<Json<Whole>, Fail> {
    let jobs = state.jobs.clone();
    let found = blocking(move || {
        Ok(match jobs.get(id)? {
            None => None,
            Some(job) => Some(Whole {
                metrics: jobs.metrics(id)?,
                samples: jobs.samples(id)?,
                measures: jobs.measures(id)?,
                pictures: jobs.pictures(id)?,
                job,
            }),
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

/// A picture a LoRA run drew, by the measurement it was drawn at and which
/// prompt it is of.
async fn job_sample(
    _: Identity,
    St(state): St<State>,
    Path((id, step, prompt)): Path<(i64, i64, i64)>,
) -> Result<Response, Fail> {
    let jobs = state.jobs.clone();
    let bytes = blocking(move || match jobs.picture_file(id, step, prompt)? {
        Some(file) => Ok(std::fs::read(file).ok()),
        None => Ok(None),
    })
    .await?
    .ok_or_else(|| Fail::missing(format!("job {id} drew no such sample")))?;
    // A sample is drawn once and never again, so it can be kept.
    Ok(([(header::CONTENT_TYPE, "image/png"), (header::CACHE_CONTROL, "private, max-age=31536000, immutable")], bytes).into_response())
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
    /// What a LoRA run for an image model can be asked for.
    lora: LoraOptions,
}

#[derive(serde::Serialize)]
pub struct LoraOptions {
    /// Why no LoRA can be trained on this server at all, if none can.
    unavailable: Option<String>,
    /// The models on this machine one can be trained for.
    models: Vec<String>,
    /// LoRAs trained here, which a run can go on from.
    continuable: Vec<String>,
    /// The sizes that have been measured, each with what a run at it is
    /// charged, with samples and without.
    sizes: Vec<LoraSize>,
    /// Bytes of the budget nothing is charged: what a run has to fit in.
    left: u64,
    defaults: LoraDefaults,
}

#[derive(serde::Serialize)]
pub struct LoraSize {
    size: usize,
    bytes: u64,
    bytes_sampling: u64,
}

#[derive(serde::Serialize)]
pub struct LoraDefaults {
    size: usize,
    rank: usize,
    steps: usize,
    lr: f64,
    eval_every: usize,
    sample_steps: usize,
    seed: u64,
}

/// SDXL's pipeline, which is the one a LoRA is trained for so far.
const TRAINABLE: &str = "StableDiffusionXLPipeline";

/// The models on this machine a LoRA can be trained for: SDXL, or a
/// fine-tune of it, as a repo in diffusers' layout. A checkpoint in one file
/// and a GGUF are not: `kvad_gpu::image::tune` says why of each.
fn trainable_models() -> Vec<String> {
    let mut models: Vec<String> = kvad::hub::local_models()
        .into_iter()
        .filter(|m| m.complete && m.gguf.is_none() && m.single.is_none() && m.lora.is_none())
        .filter(|m| kvad::hub::pipeline(m).as_deref() == Some(TRAINABLE))
        .map(|m| m.id)
        .collect();
    // SDXL itself first, where it is here: it is what a run that names no
    // model trains a LoRA for. A fine-tune is somebody's choice, and named.
    if let Some(at) = models.iter().position(|m| m == BASE) {
        models[..=at].rotate_right(1);
    }
    models
}

const BASE: &str = "stabilityai/stable-diffusion-xl-base-1.0";

/// Where LoRAs trained here are kept, as `kvad-gpu tune --name` keeps them.
fn loras_dir() -> std::path::PathBuf {
    kvad::weights::data_dir().join("loras")
}

/// The LoRAs trained here, by name. A run's last step, kept beside its best
/// as `NAME.last`, is one of them: going on from where a run stopped is
/// what it is for.
fn trained_loras() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(loras_dir())
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|f| f.strip_suffix(".safetensors").map(str::to_string))
        .collect();
    names.sort();
    names
}

/// Why this server trains no LoRA, if it does not.
fn lora_unavailable() -> Option<String> {
    if !cfg!(feature = "gpu") {
        return Some("this kvad-serve was built without the GPU backend, and an image model is trained on the GPU".into());
    }
    if crate::videos::ffmpeg().is_none() {
        return Some("the pictures are read by ffmpeg, and this server has none: install it, or name one under [videos] ffmpeg".into());
    }
    None
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
    let (continuable, training, models, loras) = blocking(move || {
        Ok((datasets::continuable_models(), jobs.training(), trainable_models(), trained_loras()))
    })
    .await?;
    let d = nervus::text::Training::default();
    let lora = LoraOptions {
        unavailable: lora_unavailable(),
        models,
        continuable: loras,
        sizes: [512, 768, 1024]
            .into_iter()
            .map(|size| LoraSize { size, bytes: crate::tune::need(size, false), bytes_sampling: crate::tune::need(size, true) })
            .collect(),
        left: state.engine.left(),
        // `kvad-gpu tune`'s own, but for the size: 512² is a quarter of the
        // time a step and two thirds of the memory, and the place to find
        // out whether a set of pictures teaches anything.
        defaults: LoraDefaults { size: 512, rank: 16, steps: 1000, lr: 1e-4, eval_every: 100, sample_steps: 20, seed: 1337 },
    };
    Ok(Json(Options {
        lora,
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

/// What a LoRA run is asked for; see [`LoraParams`] for what each means.
#[derive(serde::Deserialize)]
pub struct LoraRequest {
    dataset: i64,
    name: String,
    #[serde(default)]
    from: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    size: Option<usize>,
    #[serde(default)]
    rank: Option<usize>,
    #[serde(default)]
    alpha: Option<f64>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    lr: Option<f64>,
    #[serde(default)]
    eval_every: Option<usize>,
    #[serde(default)]
    holdout: Option<usize>,
    #[serde(default)]
    samples: Vec<String>,
    #[serde(default)]
    sample_size: Option<usize>,
    #[serde(default)]
    sample_steps: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

/// Whether `size` is one a run or a sample may be: a multiple of 64 in
/// what has been measured.
fn check_size(what: &str, size: usize) -> Result<(), String> {
    match size % 64 == 0 && crate::tune::SIZES.contains(&size) {
        true => Ok(()),
        false => Err(format!(
            "{what} {size} is not a multiple of 64 from {} to {}: what a run larger than that takes has not been measured, and one that does not fit takes the machine down",
            crate::tune::SIZES.start(),
            crate::tune::SIZES.end()
        )),
    }
}

/// Work out what a LoRA request means, and refuse it for everything that
/// can be known to be wrong before a model is loaded.
fn plan_lora(db: &crate::db::Db, body: LoraRequest) -> Result<(crate::tune::Work, LoraParams), String> {
    if let Some(why) = lora_unavailable() {
        return Err(why);
    }
    let ffmpeg = crate::videos::ffmpeg().ok_or("no ffmpeg")?;
    let name = body.name.trim().to_string();
    if !kvad::weights::is_model_name(&name) || name.ends_with(".last") {
        return Err(format!("`{name}` is not a name for a LoRA: one word, no `/`, and not ending in `.last`, which is what a run calls its last step"));
    }
    let models = trainable_models();
    let model = match body.model {
        Some(model) => model,
        None => models.first().cloned().ok_or_else(|| format!("there is no model on this machine to train a LoRA for: pull {BASE}"))?,
    };
    if !models.contains(&model) {
        return Err(format!(
            "`{model}` is not a model on this machine that a LoRA can be trained for. That is SDXL, or a fine-tune of it, as a repo and not as one file or a GGUF{}",
            match models.is_empty() {
                true => "; none is here".to_string(),
                false => format!(": {}", models.join(", ")),
            }
        ));
    }
    let from = match body.from.as_deref().map(str::trim).filter(|f| !f.is_empty()) {
        None => None,
        Some(from) => {
            let file = loras_dir().join(format!("{from}.safetensors"));
            if !kvad::weights::is_model_name(from) || !file.is_file() {
                return Err(format!("`{from}` is not a LoRA trained here"));
            }
            Some((from.to_string(), file))
        }
    };

    let (dataset, data) = datasets::folder_for(db, body.dataset).map_err(|e| e.to_string())?;
    let pictures = dataset.items.unwrap_or(0).max(0) as usize;
    if body.holdout.is_some_and(|h| h >= pictures) {
        return Err(format!("holding out {} leaves none of `{}`'s {pictures} pictures to train on", body.holdout.unwrap_or(0), dataset.name));
    }

    let size = body.size.unwrap_or(512);
    check_size("a size of", size)?;
    let samples: Vec<String> = body.samples.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if samples.len() > 4 {
        return Err(format!("{} prompts to draw at every measurement is more than four, and each is up to a minute every time", samples.len()));
    }
    if let Some(sample_size) = body.sample_size {
        check_size("a sample size of", sample_size)?;
    }
    let between = |what: &str, v: usize, most: usize| match (1..=most).contains(&v) {
        true => Ok(v),
        false => Err(format!("{what} is from 1 to {most}, and {v} was asked for")),
    };
    let rank = between("the rank", body.rank.unwrap_or(16), 128)?;
    let steps = between("the number of steps", body.steps.unwrap_or(1000), 100_000)?;
    let eval_every = between("the steps between measurements", body.eval_every.unwrap_or(100), 100_000)?;
    let sample_steps = between("a sample's steps", body.sample_steps.unwrap_or(20), 100)?;
    let lr = body.lr.unwrap_or(1e-4);
    if !(lr.is_finite() && lr > 0.0 && lr <= 1.0) {
        return Err(format!("a learning rate of {lr} is not one: more than 0, and 1e-4 unless there is a reason"));
    }
    if body.alpha.is_some_and(|a| !(a.is_finite() && a > 0.0)) {
        return Err("alpha is more than 0, or left out to be the rank".into());
    }

    let charged = crate::tune::need(size, !samples.is_empty());
    let params = LoraParams {
        name: name.clone(),
        from: from.as_ref().map(|f| f.0.clone()),
        model: model.clone(),
        dataset: dataset.id,
        dataset_name: dataset.name,
        size,
        rank,
        alpha: body.alpha,
        steps,
        lr,
        eval_every,
        holdout: body.holdout,
        samples: samples.clone(),
        sample_size: body.sample_size,
        sample_steps,
        seed: body.seed.unwrap_or(1337),
        charged,
    };
    let work = crate::tune::Work {
        repo: model,
        data,
        out: loras_dir().join(format!("{name}.safetensors")),
        from: from.map(|f| f.1),
        size,
        rank,
        alpha: body.alpha,
        steps,
        lr,
        seed: params.seed,
        eval_every,
        holdout: body.holdout,
        samples,
        sample_size: body.sample_size,
        sample_steps,
        // `tune::start` names it, once the job has a number.
        sample_dir: std::path::PathBuf::new(),
        ffmpeg,
        cap_gb: (charged + crate::tune::OVER) as f64 / 1e9,
        data_dir: kvad::weights::chosen_data_dir(),
    };
    Ok((work, params))
}

/// Start a LoRA run: plan it, charge it, and hand it to a worker.
async fn start_lora(who: Admin, state: State, body: LoraRequest) -> Result<Json<Job>, Fail> {
    let db = state.db.clone();
    let jobs = state.jobs.clone();
    let (work, params) = blocking(move || {
        // Before anything is charged: a second run is refused for being a
        // second run, not for the memory the first is holding.
        if jobs.training() {
            return Err("a training run is already going; wait for it or stop it".into());
        }
        Ok(plan_lora(&db, body)?)
    })
    .await
    .map_err(|e| Fail::bad(e.1))?;

    let what = format!("training the LoRA {} at {}×{}", params.name, params.size, params.size);
    let hold = state.engine.reserve(what, params.charged).await.map_err(Fail::conflict)?;
    let jobs = state.jobs.clone();
    let owner = who.0.id;
    blocking(move || crate::tune::start(&jobs, hold, work, params, owner)).await.map(Json).map_err(|e| Fail::bad(e.1))
}

/// Start a run. `loop` says which: `text`, or left out, for a language model
/// trained from scratch on a text, and `lora` for a LoRA on an image model.
async fn start(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<Job>, Fail> {
    let read = |e: serde_json::Error| Fail::bad(format!("that is not a run this can start: {e}"));
    let body: StartRequest = match body.get("loop").and_then(|l| l.as_str()) {
        None | Some("text") => serde_json::from_value(body).map_err(read)?,
        Some("lora") => return start_lora(who, state, serde_json::from_value(body).map_err(read)?).await,
        Some(other) => return Err(Fail::bad(format!("`{other}` is not a training loop this server has: there are text and lora"))),
    };
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

        let d = nervus::text::Training::default();
        let training = nervus::text::Training {
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
    blocking(move || datasets::save(&db, q.name.trim(), &text, owner, None))
        .await
        .map(Json)
        .map_err(|e| Fail::bad(e.1))
}

/// Read a website into a dataset.
///
/// Answers with the job, not with the dataset: this takes minutes and several
/// hundred requests, and the page that asked follows it the way it follows a
/// training run.
async fn start_crawl(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<kvad::crawl::Request>,
) -> Result<Json<Job>, Fail> {
    let request = body.sane();
    datasets::check_name(request.name.trim()).map_err(Fail::bad)?;
    let request = kvad::crawl::Request { name: request.name.trim().to_string(), ..request };
    let jobs = state.jobs.clone();
    blocking(move || {
        // Before the job exists: an address that cannot be fetched is this
        // request's answer, not a row that fails a second after it is made.
        kvad::crawl::check(&request)?;
        jobs::crawl(&jobs, request, who.0.id)
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

/// One file of a set of pictures being uploaded: a picture, or the `.txt`
/// that captions one.
///
/// The body is the file, as an uploaded text's is. Nothing is a dataset
/// until [`keep_pictures`] is asked, which looks at all of them together.
async fn stage_picture(
    _: Admin,
    Path((name, file)): Path<(String, String)>,
    bytes: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, Fail> {
    blocking(move || {
        datasets::stage(name.trim(), &file, &bytes)?;
        Ok(json!({ "staged": file, "bytes": bytes.len() }))
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

#[derive(serde::Deserialize)]
pub struct Keeping {
    /// The caption of every picture that has none beside it.
    #[serde(default)]
    caption: Option<String>,
}

/// Make a dataset of the pictures uploaded under this name, if every one of
/// them decodes and has a caption; and if not, say everything that is wrong
/// with the lot and keep none of it.
async fn keep_pictures(
    who: Admin,
    St(state): St<State>,
    Path(name): Path<String>,
    Query(q): Query<Keeping>,
) -> Result<Json<Dataset>, Fail> {
    let db = state.db.clone();
    let owner = who.0.id;
    blocking(move || {
        let ffmpeg = crate::videos::ffmpeg().ok_or("the pictures are read by ffmpeg, to see that each is one, and this server has none: install it, or name one under [videos] ffmpeg")?;
        // What the trainer will read each with, so that what passes here is
        // what it can read.
        // Not ffmpeg's own account of why: it is a dozen lines a picture,
        // and what there is to do about it is the same whatever they say.
        let decodes = |path: &std::path::Path| kvad::video::picture_from_file(&ffmpeg, path, 0).map(|_| ()).map_err(|_| "ffmpeg could not decode it".to_string());
        datasets::keep(&db, name.trim(), q.caption.as_deref(), owner, &decodes)
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

/// Throw away an upload that was not kept.
async fn drop_pictures(_: Admin, Path(name): Path<String>) -> Result<Json<serde_json::Value>, Fail> {
    blocking(move || {
        datasets::unstage(name.trim())?;
        Ok(json!({ "dropped": name }))
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

/// The pictures of a set, each with its caption.
async fn pictures(_: Admin, St(state): St<State>, Path(id): Path<i64>) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let (dataset, pictures) = blocking(move || datasets::pictures(&db, id)).await.map_err(|e| Fail::missing(e.1))?;
    Ok(Json(json!({ "dataset": dataset, "pictures": pictures })))
}

/// One picture of a set, as it was uploaded.
async fn picture(_: Admin, St(state): St<State>, Path((id, file)): Path<(i64, String)>) -> Result<Response, Fail> {
    let db = state.db.clone();
    let kind = mime_guess::from_path(&file).first_or_octet_stream().to_string();
    let bytes = blocking(move || Ok(std::fs::read(datasets::picture_file(&db, id, &file)?)?)).await.map_err(|e| Fail::missing(e.1))?;
    Ok(([(header::CONTENT_TYPE, kind)], bytes).into_response())
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
    let (dataset, text) = blocking(move || match datasets::get(&db, id)? {
        // A set of pictures has no text to show the start of; its pictures
        // and their captions are `/api/datasets/{id}/pictures`.
        Some(d) if d.kind == datasets::Kind::Pictures => Ok((d, String::new())),
        _ => datasets::read(&db, id),
    })
    .await
    .map_err(|e| Fail::missing(e.1))?;
    Ok(Json(WithText { preview: text.chars().take(2000).collect(), dataset }))
}

async fn remove_dataset(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    match blocking(move || datasets::delete(&db, id)).await? {
        true => {
            crate::retrieval::forget(id);
            Ok(Json(json!({ "deleted": id })))
        }
        false => Err(Fail::missing(format!("there is no dataset {id}"))),
    }
}

#[derive(serde::Deserialize)]
pub struct Question {
    q: String,
    #[serde(default)]
    k: Option<usize>,
}

/// What this dataset has to say about a question.
///
/// The half of retrieval that can be judged on its own. Whether the right
/// passage comes back and whether a model then reads it properly are
/// different questions that fail for different reasons, and only the first
/// one has an answer you can look at.
async fn search(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
    Query(q): Query<Question>,
) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let k = q.k.unwrap_or(5).clamp(1, 50);
    let found = blocking(move || {
        let index = crate::retrieval::index_for(&db, id)?;
        // What the model would be given, not the chunks behind it: a passage
        // is the unit that gets read.
        let passages = index.passages(&q.q, k, kvad::retrieve::RADIUS);
        Ok(json!({
            "chunks": index.len(),
            "vocabulary": index.vocabulary(),
            "passages": passages,
        }))
    })
    .await
    .map_err(|e| Fail::bad(e.1))?;
    Ok(Json(found))
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
