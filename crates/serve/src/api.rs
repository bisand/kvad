//! The JSON API, and the plumbing every handler in it shares.
//!
//! Two surfaces, kept apart on purpose:
//!
//! * `/api/**` is this server's own — models, conversations, health. It can
//!   change whenever the UI needs it to.
//! * `/v1/**` is the OpenAI-compatible surface. It is shaped by somebody
//!   else's documentation and has to stay that way, which is the whole point
//!   of it: anything that already speaks to OpenAI speaks to this.
//!
//! The UI's chat goes through `/v1/chat/completions` like any other client,
//! so the compatible path is the one that gets exercised every day rather
//! than the one that quietly rots.

use crate::auth::{Identity, State};
use axum::extract::State as St;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/openapi.json", get(crate::openapi::document))
        .route("/api/models", get(crate::models::list).delete(crate::models::remove))
        .route("/api/models/search", get(crate::models::search))
        .route("/api/models/detail", get(crate::models::hub_detail))
        .route("/api/models/load", post(crate::models::load))
        .route("/api/models/unload", post(crate::models::unload))
        .route("/api/models/active", post(crate::models::set_active))
        .route("/api/models/pull", post(crate::models::pull))
        .route("/api/qcache", delete(crate::models::forget_qcache))
        .route("/api/generation", delete(crate::openai::cancel))
        .route(
            "/api/conversations",
            get(crate::conversations::list).post(crate::conversations::create),
        )
        .route(
            "/api/conversations/{id}",
            get(crate::conversations::get)
                .patch(crate::conversations::update)
                .delete(crate::conversations::remove),
        )
        .route("/api/conversations/{id}/messages", post(crate::conversations::append))
        .route("/v1/models", get(crate::openai::models))
        .route("/v1/chat/completions", post(crate::openai::completions))
        // Signing in, and the one route that has to answer a request with no
        // credential at all.
        .route("/api/auth", get(crate::accounts::situation))
        .route("/api/auth/login", post(crate::accounts::sign_in))
        .route("/api/auth/logout", post(crate::accounts::sign_out))
        .route("/api/auth/setup", post(crate::accounts::setup))
        .route("/api/auth/oidc/start", get(crate::accounts::oidc_start))
        .route("/api/auth/oidc/callback", get(crate::accounts::oidc_callback))
        .route("/api/auth/password", post(crate::accounts::change_password))
        .route("/api/users", get(crate::accounts::list_users).post(crate::accounts::create_user))
        .route(
            "/api/users/{id}",
            axum::routing::patch(crate::accounts::update_user)
                .delete(crate::accounts::delete_user),
        )
        .route("/api/sessions", get(crate::accounts::list_sessions))
        .route("/api/sessions/{hash}", delete(crate::accounts::revoke_session))
        .route("/api/keys", get(crate::accounts::list_keys).post(crate::accounts::create_key))
        .route("/api/keys/{id}", delete(crate::accounts::revoke_key))
        .merge(crate::training::routes())
        .merge(crate::monitoring::routes())
        .merge(crate::playground::routes())
        .merge(crate::evals::routes())
        .merge(crate::bench::routes())
        .merge(crate::images::routes())
        .merge(crate::videos::routes())
        .merge(crate::settings::routes())
        .merge(crate::storage::routes())
}

/// A request that could not be answered, as a status and a sentence.
///
/// One shape for every failure, so a client has one thing to parse:
/// `{"error": "..."}`. The message is for a person to read — it is what lands
/// in a toast — so it says what went wrong rather than which function it
/// happened in.
#[derive(Debug)]
pub struct Fail(pub StatusCode, pub String);

impl Fail {
    pub fn bad(why: impl Into<String>) -> Self {
        Fail(StatusCode::BAD_REQUEST, why.into())
    }
    pub fn missing(what: impl Into<String>) -> Self {
        Fail(StatusCode::NOT_FOUND, what.into())
    }
    /// Refused because of what the server is holding, not because of what
    /// was asked: a model that would fit if something else were unloaded.
    pub fn conflict(why: impl Into<String>) -> Self {
        Fail(StatusCode::CONFLICT, why.into())
    }
    pub fn internal(why: impl Into<String>) -> Self {
        Fail(StatusCode::INTERNAL_SERVER_ERROR, why.into())
    }
    /// Signed in, and not allowed this. Distinct from [`crate::auth::Denied`],
    /// which answers a request that failed to say who it was at all.
    pub fn denied(why: impl Into<String>) -> Self {
        Fail(StatusCode::FORBIDDEN, why.into())
    }
}

impl IntoResponse for Fail {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

/// Run blocking work — SQLite, the filesystem, the Hub — off the runtime.
///
/// Everything under this server that touches a disk is synchronous, because
/// the engine is. Rather than pretend otherwise, each handler says so by
/// going through here, and the error comes back as text because
/// `Box<dyn Error>` is not `Send` and cannot cross a thread boundary.
pub async fn blocking<T, F>(f: F) -> Result<T, Fail>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'static,
{
    match tokio::task::spawn_blocking(move || f().map_err(|e| e.to_string())).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(why)) => Err(Fail::internal(why)),
        Err(e) => Err(Fail::internal(format!("a background task failed: {e}"))),
    }
}

/// [`blocking`] for work that answers with its own [`Fail`], so that a
/// refusal stays a 400 or a 409 rather than becoming a 500.
pub async fn blocking_or<T, F>(f: F) -> Result<T, Fail>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, Fail> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Fail::internal(format!("a background task failed: {e}")))?
}

#[derive(serde::Serialize)]
pub struct Health {
    /// Always `"ok"`: a handler that runs at all is a server that is serving.
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
    /// Which schema version the database is at. Cheap, and it is the first
    /// thing to look at when a deployment behaves like an older one.
    schema: u32,
    /// Whether a built UI is in this binary. A person who sees the stub page
    /// and thinks the server is broken can be pointed here.
    ui_embedded: bool,
    /// The model in memory used most recently, every model in memory, and
    /// how much is waiting for them.
    loaded: Option<crate::scheduler::Loaded>,
    residents: Vec<crate::scheduler::Resident>,
    queue_depth: usize,
    /// Which authentication mode is in force. The UI shows or hides the
    /// sign-out button by it.
    auth: String,
    /// Who the server thinks is asking. With `auth.mode = "none"` this is
    /// always the local operator, and saying so out loud is the point.
    you: Identity,
}

async fn health(who: Identity, St(state): St<State>) -> Json<Health> {
    Json(Health {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.started.elapsed().as_secs(),
        // A database that cannot answer is worth a zero rather than a 500:
        // health is what you call when things are already wrong.
        schema: state.db.version().unwrap_or(0),
        ui_embedded: crate::assets::is_embedded(),
        loaded: state.engine.loaded(),
        residents: state.engine.residents(),
        queue_depth: state.engine.depth(),
        auth: state.auth.mode().to_string(),
        you: who,
    })
}
