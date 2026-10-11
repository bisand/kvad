//! What the Dashboard and the Monitoring pages read.
//!
//! Everything here comes from the ring buffer in [`crate::metrics`] or from a
//! directory walk, so a page polling every few seconds costs a mutex and not
//! a query.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
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
        .route("/api/metrics/usage", get(usage))
}

#[derive(serde::Serialize)]
pub struct Overview {
    uptime_secs: u64,
    loaded: Option<crate::scheduler::Loaded>,
    /// Every model in memory, and what each was charged.
    residents: Vec<crate::scheduler::Resident>,
    memory: crate::models::Budgeted,
    queue_depth: usize,
    /// Resident memory, or null on a platform we cannot ask. Null is not
    /// zero, and a dashboard should not draw it as though it were.
    resident_bytes: Option<u64>,
    /// What the most recently used model's KV cache would cost at its full
    /// context, and what its last generation actually left in it.
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
    let current = state.engine.current();
    let loaded = current.as_ref().map(|r| r.model.clone());
    let generations: Vec<Generation> = state.metrics.generations();
    let recent = state.metrics.recent(2000);

    let jobs = state.jobs.clone();
    let (disk, running) = blocking(move || {
        Ok((machine::disk(), jobs.list(50)?.into_iter().filter(|j| j.live()).collect::<Vec<_>>()))
    })
    .await?;

    let kv = current.as_ref().map(|r| {
        let (l, cached) = (&r.model, r.cached_tokens);
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
        residents: state.engine.residents(),
        memory: crate::models::Budgeted::of(&state.engine),
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

#[derive(serde::Deserialize)]
pub struct Days {
    #[serde(default)]
    days: Option<u32>,
}

/// One account's usage, with whose it is.
#[derive(serde::Serialize)]
pub struct Used {
    /// The account's name: `local` for a server with no accounts, and null
    /// for an account that has since been removed.
    name: Option<String>,
    mine: bool,
    #[serde(flatten)]
    usage: metrics::Usage,
}

#[derive(serde::Serialize)]
pub struct UsageReport {
    days: u32,
    accounts: Vec<Used>,
}

/// Who has used the engine, and what they waited for it.
///
/// A signed-in user is shown their own figures; an administrator
/// everybody's. From the table rather than the ring buffer, since the
/// question is as often about last week as about now — and what has not
/// been written yet is written first, so the last few seconds are in it.
pub async fn usage(who: Identity, St(state): St<State>, Query(q): Query<Days>) -> Result<Json<UsageReport>, Fail> {
    let days = q.days.unwrap_or(7).clamp(1, 365);
    let (db, metrics) = (state.db.clone(), state.metrics.clone());
    let admin = who.is_admin();
    let (all, users) = blocking(move || {
        metrics.flush(&db)?;
        let users = match admin {
            true => crate::users::list(&db)?,
            false => Vec::new(),
        };
        Ok((metrics::usage(&db, days)?, users))
    })
    .await?;
    let accounts = all
        .into_iter()
        .filter(|u| admin || u.owner == who.id)
        .map(|usage| Used {
            name: match usage.owner {
                None => Some("local".to_string()),
                Some(id) if Some(id) == who.id => Some(who.name.clone()),
                Some(id) => users.iter().find(|u| u.id == id).map(|u| u.name.clone()),
            },
            mine: usage.owner == who.id,
            usage,
        })
        .collect();
    Ok(Json(UsageReport { days, accounts }))
}
