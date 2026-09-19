//! What the Dashboard and the Monitoring pages read.
//!
//! Everything here comes from the ring buffer in [`crate::metrics`] or from a
//! directory walk, so a page polling every few seconds costs a mutex and not
//! a query.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, State};
use crate::machine::{self, Disk};
use crate::metrics::{self, ByRoute, Generation, Request, Summary};
use axum::extract::{Query, State as St};
use axum::routing::get;
use axum::{Json, Router};

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/metrics", get(overview))
        .route("/api/metrics/requests", get(requests))
        .route("/api/metrics/log", get(log))
}

#[derive(serde::Serialize)]
pub struct Overview {
    uptime_secs: u64,
    loaded: Option<crate::scheduler::Loaded>,
    queue_depth: usize,
    /// Resident memory, or null on a platform we cannot ask. Null is not
    /// zero, and a dashboard should not draw it as though it were.
    resident_bytes: Option<u64>,
    /// What the loaded model's KV cache would cost at its full context, and
    /// what the last generation actually left in it.
    kv: Option<Kv>,
    disk: Disk,
    /// Jobs running right now, so the dashboard can say the machine is busy
    /// without the Training page being open.
    running: Vec<crate::jobs::Job>,
    /// Newest last, for sparklines.
    decode_per_sec: Vec<f64>,
    ttft_millis: Vec<f64>,
    latency: Summary,
    requests: usize,
    errors: usize,
}

#[derive(serde::Serialize)]
pub struct Kv {
    bytes_per_token: usize,
    max_bytes: usize,
    n_ctx: usize,
    /// Tokens the cache held after the last generation. Not live: the engine
    /// is busy generating when it would be interesting to ask, and a number
    /// from a moment ago is better than a lock on the engine thread.
    cached_tokens: usize,
    cached_bytes: usize,
}

pub async fn overview(_: Admin, St(state): St<State>) -> Result<Json<Overview>, Fail> {
    let loaded = state.engine.loaded();
    let generations: Vec<Generation> = state.metrics.generations();
    let recent = state.metrics.recent(2000);

    let jobs = state.jobs.clone();
    let (disk, running) = blocking(move || {
        Ok((machine::disk(), jobs.list(50)?.into_iter().filter(|j| j.live()).collect::<Vec<_>>()))
    })
    .await?;

    let kv = loaded.as_ref().map(|l| {
        let cached = state.engine.last_cached();
        Kv {
            bytes_per_token: l.kv_bytes_per_token,
            max_bytes: l.kv_bytes_per_token * l.n_ctx,
            n_ctx: l.n_ctx,
            cached_tokens: cached,
            cached_bytes: cached * l.kv_bytes_per_token,
        }
    });

    Ok(Json(Overview {
        uptime_secs: state.started.elapsed().as_secs(),
        queue_depth: state.engine.depth(),
        resident_bytes: machine::resident_bytes(),
        kv,
        disk,
        running,
        // Oldest first, which is the direction a sparkline is read.
        decode_per_sec: generations.iter().map(|g| g.decode_per_sec).collect(),
        ttft_millis: generations.iter().map(|g| g.ttft_millis).collect(),
        latency: metrics::summarise(recent.iter().map(|r| r.millis).collect()),
        errors: recent.iter().filter(|r| r.status >= 400).count(),
        requests: recent.len(),
        loaded,
    }))
}

#[derive(serde::Deserialize)]
pub struct Limit {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(serde::Serialize)]
pub struct Requests {
    recent: Vec<Request>,
    by_route: Vec<ByRoute>,
}

pub async fn requests(
    _: Admin,
    St(state): St<State>,
    Query(q): Query<Limit>,
) -> Json<Requests> {
    let all = state.metrics.recent(2000);
    let limit = q.limit.unwrap_or(100).clamp(1, 2000);
    Json(Requests {
        by_route: metrics::by_route(&all),
        recent: all.into_iter().take(limit).collect(),
    })
}

pub async fn log(_: Admin, St(state): St<State>, Query(q): Query<Limit>) -> Json<Vec<String>> {
    Json(state.metrics.log(q.limit.unwrap_or(200).clamp(1, 500)))
}
