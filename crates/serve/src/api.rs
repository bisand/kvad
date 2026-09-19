//! The JSON API. One route so far.
//!
//! Everything under `/api` is this server's own; `/v1` is reserved for the
//! OpenAI-compatible surface that arrives with chat in Phase 2, and the two
//! are kept apart so that the compatible half can stay compatible.

use crate::auth::{Identity, State};
use axum::routing::get;
use axum::{Json, Router};

pub fn routes() -> Router<State> {
    Router::new().route("/api/health", get(health))
}

#[derive(serde::Serialize)]
pub struct Health {
    /// Always `"ok"`: a handler that runs at all is a server that is serving.
    /// Whether the *engine* is healthy is a different question, and gets its
    /// own fields once there is an engine.
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
    /// Which schema version the database is at. Cheap, and it is the first
    /// thing to look at when a deployment behaves like an older one.
    schema: u32,
    /// Whether a built UI is in this binary. A person who sees the stub page
    /// and thinks the server is broken can be pointed here.
    ui_embedded: bool,
    /// Who the server thinks is asking. With `auth.mode = "none"` this is
    /// always the local operator, and saying so out loud is the point.
    you: Identity,
}

async fn health(who: Identity, axum::extract::State(state): axum::extract::State<State>) -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.started.elapsed().as_secs(),
        // A database that cannot answer is worth a zero rather than a 500:
        // health is what you call when things are already wrong.
        schema: state.db.version().unwrap_or(0),
        ui_embedded: crate::assets::is_embedded(),
        you: who,
    })
}
