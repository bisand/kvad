//! The measurement protocol, as a page.
//!
//! The README's numbers were wrong three times before this project adopted a
//! protocol: interleave the variants, run each several times, report the
//! median and the range, and check the machine is idle first. Doing that by
//! hand is a shell script nobody reruns; doing it here means the numbers in
//! the README have a button that reproduces them.
//!
//! What this page refuses to do is as much of the point as what it does. It
//! will not start while a training run, an eval or another benchmark is going
//! — see [`crate::compare::machine_is_busy`] — because a benchmark of a busy
//! machine is a measurement of the other job.
//!
//! The ordering is in [`crate::compare`]: a round visits every variant once,
//! and a run is several rounds, so drift shows up as a trend rather than as a
//! winner.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
use crate::compare::{self, BenchParams, Variant};
use crate::jobs::{Job, Timing};
use axum::extract::{Path, State as St};
use axum::routing::{get, post};
use axum::{Json, Router};

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/bench/run", post(start))
        .route("/api/bench/runs", get(list))
        .route("/api/bench/runs/{id}", get(samples))
}

#[derive(serde::Deserialize)]
pub struct StartRequest {
    variants: Vec<Variant>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    rounds: Option<usize>,
    #[serde(default)]
    tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// Keep what each variant wrote, for a side-by-side comparison. The
    /// Playground's compare button sets this; the Benchmarks page does not.
    #[serde(default)]
    keep_text: bool,
}

/// Long enough to prefill meaningfully, short enough that the prompt is not
/// what is being measured.
const DEFAULT_PROMPT: &str = "The history of the transformer architecture begins with";

async fn start(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<StartRequest>,
) -> Result<Json<Job>, Fail> {
    let variants = compare::resolve(&body.variants).map_err(Fail::bad)?;
    let jobs = state.jobs.clone();
    let engine = state.engine.clone();

    // The idle check, first and as a refusal. A number measured next to a
    // training run is a number about the training run.
    if let Some(busy) = compare::machine_is_busy(&jobs) {
        return Err(Fail::bad(format!(
            "{busy} is running, and a benchmark of a busy machine measures the other job. \
             Wait for it or stop it."
        )));
    }

    let p = BenchParams {
        prompt: match body.prompt {
            Some(p) if !p.trim().is_empty() => p,
            _ => DEFAULT_PROMPT.to_string(),
        },
        rounds: body.rounds.unwrap_or(5).clamp(1, 20),
        tokens: body.tokens.unwrap_or(64).clamp(1, 1024),
        seed: body.seed.unwrap_or(1337),
        keep_text: body.keep_text,
    };
    compare::bench(&jobs, &engine, p, variants, who.0.id)
        .map(Json)
        .map_err(|e| Fail::bad(e.to_string()))
}

async fn list(_: Identity, St(state): St<State>) -> Result<Json<Vec<Job>>, Fail> {
    let jobs = state.jobs.clone();
    blocking(move || jobs.db.with(|c| crate::evals::kind_rows(c, "bench", 50))).await.map(Json)
}

/// Every sample of a run, and the summary computed from them.
///
/// The summary is recomputed here rather than read from the job's result, so
/// that a run still going has one too — watching five rounds arrive with no
/// median until the end would be a page that says nothing while it works.
#[derive(serde::Serialize)]
pub struct Samples {
    #[serde(flatten)]
    job: Job,
    samples: Vec<Timing>,
    summary: serde_json::Value,
}

async fn samples(
    _: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<Samples>, Fail> {
    let jobs = state.jobs.clone();
    let found = blocking(move || {
        let Some(job) = jobs.get(id)? else { return Ok(None) };
        let samples = jobs.timings(id)?;
        let summary = jobs.db.with(|c| compare::summarise(c, id))?;
        Ok(Some(Samples { job, samples, summary }))
    })
    .await?;
    found.map(Json).ok_or_else(|| Fail::missing(format!("there is no benchmark {id}")))
}
