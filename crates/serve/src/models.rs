//! What is on this machine, what is on the Hub, and what the engine holds.
//!
//! The filesystem is the truth here and the database has no say in it: what
//! `kvad ls` lists, this lists, by calling the same functions. Two answers to
//! "is this model downloaded?" would disagree the first time somebody deleted
//! a directory by hand.
//!
//! # Long operations
//!
//! Pulling a model takes minutes and loading one takes tens of seconds, so
//! both stream their progress rather than making the browser wait on a
//! request with nothing in it. Both are `POST` with an SSE body: a browser's
//! `EventSource` can only issue `GET`, but `fetch` can read a stream, and the
//! chat endpoint needs that reader anyway.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
use crate::scheduler::Progress;
use axum::extract::{Query, State as St};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use kvad::hub;
use kvad::weights;
use serde_json::json;
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

/// The settings key holding the backend a load uses when nobody says.
///
/// In the database rather than the config file because it is a preference,
/// changed from the UI while the server runs — which is exactly the line the
/// config module draws.
const BACKEND_KEY: &str = "models.backend";

#[derive(serde::Serialize)]
pub struct Model {
    pub id: String,
    /// `gpt2` or `llama`, read from the model's own config.json, or `null`
    /// for a directory this engine cannot run.
    pub arch: Option<String>,
    pub bytes: u64,
    /// False for a cache entry with a config and no weights: a half-finished
    /// download, or a training run stopped before its first checkpoint.
    pub complete: bool,
    /// Trained here, rather than downloaded. The distinction matters because
    /// one of them can be fetched again and the other cannot.
    pub trained: bool,
    pub runnable: bool,
    /// Why not, when not.
    pub blocker: Option<String>,
}

fn describe(m: &hub::LocalModel, trained: bool) -> Model {
    let blocker = match (m.arch, m.complete) {
        (None, _) => Some("this engine runs GPT-2 and the Llama family; this is neither".into()),
        (_, false) => Some(match trained {
            true => "no weights yet — the run was stopped before its first checkpoint".into(),
            false => "the download did not finish".to_string(),
        }),
        _ => None,
    };
    Model {
        id: m.id.clone(),
        arch: m.arch.map(|a| a.to_string()),
        bytes: m.bytes,
        complete: m.complete,
        trained,
        runnable: blocker.is_none(),
        blocker,
    }
}

#[derive(serde::Serialize)]
pub struct Listing {
    downloaded: Vec<Model>,
    trained: Vec<Model>,
    /// Pre-quantised weights, which are a cache and can always be deleted.
    qcache: Vec<QCache>,
    /// The model `kvad run` would pick with no `--model`. Shared with the CLI
    /// and the TUI, so setting it here sets it there.
    active: Option<String>,
    loaded: Option<crate::scheduler::Loaded>,
    queue_depth: usize,
    /// Every backend this build can actually load; see `engine::available`.
    backends: Vec<crate::engine::Choice>,
    /// Which of them a load uses when the request does not say.
    backend: String,
}

#[derive(serde::Serialize)]
pub struct QCache {
    repo: String,
    precision: String,
    bytes: u64,
}

/// The backend a load uses when the request does not name one.
fn default_backend(db: &crate::db::Db) -> String {
    let stored = db.setting(BACKEND_KEY).ok().flatten();
    let stored = stored.as_ref().and_then(|v| v.as_str()).unwrap_or("");
    match crate::engine::parse(stored) {
        // A stored backend that this build cannot load — a GPU one in a
        // `--no-default-features` binary — is treated as unset rather than as
        // an error every load has to explain.
        Some(b) => crate::engine::id_of(b),
        None => crate::engine::id_of(kvad::service::Backend::Cpu(kvad::quant::Precision::Q8)),
    }
}

/// Readable by anyone signed in, because the Chat page needs to know what is
/// loaded. Everything that *changes* a model takes `Admin` instead. Nothing
/// here is a secret: model names, sizes, and which backends this build has.
pub async fn list(_: Identity, St(state): St<State>) -> Result<Json<Listing>, Fail> {
    let db = state.db.clone();
    let scanned = blocking(move || {
        Ok((
            hub::local_models(),
            hub::trained_models(),
            kvad::qcache::entries(),
            hub::State::active(),
            default_backend(&db),
        ))
    })
    .await?;
    let (downloaded, trained, qcache, active, backend) = scanned;

    Ok(Json(Listing {
        downloaded: downloaded.iter().map(|m| describe(m, false)).collect(),
        trained: trained.iter().map(|m| describe(m, true)).collect(),
        qcache: qcache
            .into_iter()
            .map(|(_, repo, precision, bytes)| QCache { repo, precision, bytes })
            .collect(),
        active,
        loaded: state.engine.loaded(),
        queue_depth: state.engine.depth(),
        backends: crate::engine::available(),
        backend,
    }))
}

#[derive(serde::Deserialize)]
pub struct SearchQuery {
    q: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(serde::Serialize)]
pub struct Found {
    id: String,
    arch: Option<String>,
    model_type: Option<String>,
    downloads: u64,
    likes: u64,
    /// From the name alone. Only the tokenizer config can confirm it, which
    /// means downloading — so the listing guesses and says it is guessing.
    looks_instruct: bool,
    runnable: bool,
    blocker: Option<String>,
    /// Already on this machine, so the button says "load" rather than "pull".
    local: bool,
}

pub async fn search(
    _: Admin,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<Found>>, Fail> {
    let q = query.q.trim().to_string();
    if q.is_empty() {
        return Err(Fail::bad("a search needs something to search for"));
    }
    let limit = query.limit.unwrap_or(40).clamp(1, 100);
    let (results, local) =
        blocking(move || Ok((hub::search(&q, limit)?, hub::local_models()))).await?;

    let here: Vec<&str> = local.iter().map(|m| m.id.as_str()).collect();
    Ok(Json(
        results
            .iter()
            .map(|m| Found {
                arch: m.arch.map(|a| a.to_string()),
                model_type: m.model_type.clone(),
                downloads: m.downloads,
                likes: m.likes,
                looks_instruct: m.looks_instruct,
                runnable: m.runnable(),
                blocker: m.blocker(),
                local: here.contains(&m.id.as_str()),
                id: m.id.clone(),
            })
            .collect(),
    ))
}

#[derive(serde::Deserialize)]
pub struct LoadRequest {
    repo: String,
    /// A backend id from the listing, e.g. `cpu-q8`. Omitted means whichever
    /// was used last.
    #[serde(default)]
    backend: Option<String>,
}

/// Load a model, streaming what it is doing while it does it.
pub async fn load(
    _: Admin,
    St(state): St<State>,
    Json(body): Json<LoadRequest>,
) -> Result<impl IntoResponse, Fail> {
    let repo = body.repo.trim().to_string();
    if repo.is_empty() {
        return Err(Fail::bad("no model was named"));
    }
    let wanted = match &body.backend {
        Some(id) => id.clone(),
        None => {
            let db = state.db.clone();
            blocking(move || Ok(default_backend(&db))).await?
        }
    };
    let backend = crate::engine::parse(&wanted).ok_or_else(|| {
        Fail::bad(format!(
            "`{wanted}` is not a backend this build can load; it has {}",
            crate::engine::available().iter().map(|c| c.id.clone()).collect::<Vec<_>>().join(", ")
        ))
    })?;

    // Remembered before the load rather than after, so that a load which
    // fails halfway still leaves the picker showing what was asked for.
    let db = state.db.clone();
    let remember = crate::engine::id_of(backend);
    blocking(move || db.set_setting(BACKEND_KEY, &json!(remember))).await?;

    let (progress, updates) = tokio::sync::mpsc::channel(64);
    let engine = state.engine.clone();
    let (events, rx) = tokio::sync::mpsc::channel::<Event>(64);

    tokio::spawn(async move {
        let mut updates = ReceiverStream::new(updates);
        let finished = tokio::spawn({
            let engine = engine.clone();
            async move { engine.load(repo, backend, progress).await }
        });
        while let Some(p) = updates.next().await {
            if events.send(sse("progress", &p)).await.is_err() {
                // Nobody is reading. The load carries on — it is the server's
                // now, not this request's — but there is nothing to report to.
                break;
            }
        }
        let outcome = match finished.await {
            Ok(Ok(model)) => sse("loaded", &model),
            Ok(Err(why)) => sse("error", &json!({ "error": why })),
            Err(e) => sse("error", &json!({ "error": format!("the load was interrupted: {e}") })),
        };
        let _ = events.send(outcome).await;
    });

    Ok(stream(rx))
}

pub async fn unload(_: Admin, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let was = state.engine.unload().await.map_err(Fail::internal)?;
    Ok(Json(json!({ "unloaded": was })))
}

#[derive(serde::Deserialize)]
pub struct ActiveRequest {
    /// `null` clears the setting, which is how `kvad rm` leaves it when the
    /// active model is the one being deleted.
    repo: Option<String>,
}

pub async fn set_active(
    _: Admin,
    Json(body): Json<ActiveRequest>,
) -> Result<Json<serde_json::Value>, Fail> {
    let repo = body.repo.clone();
    blocking(move || match &repo {
        Some(repo) => hub::State::set_active(repo),
        None => hub::State::clear(),
    })
    .await?;
    Ok(Json(json!({ "active": body.repo })))
}

#[derive(serde::Deserialize)]
pub struct PullRequest {
    repo: String,
}

/// Download a model without loading it.
///
/// Not scheduler work: a download is network and disk, and making it wait
/// behind a generation would be a queue for no reason. It is also why the
/// Models page can pull one model while chatting with another.
pub async fn pull(
    _: Admin,
    Json(body): Json<PullRequest>,
) -> Result<impl IntoResponse, Fail> {
    let repo = body.repo.trim().to_string();
    if repo.is_empty() {
        return Err(Fail::bad("no model was named"));
    }

    let (events, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let reporter = events.clone();
    tokio::task::spawn_blocking(move || {
        let watch = {
            let reporter = reporter.clone();
            weights::Watcher::new(move |f| {
                if let Some(p) = progress_of(f) {
                    // `try_send` rather than blocking: this runs on hf-hub's
                    // own download threads, which must not be held up by a
                    // browser that is reading slowly.
                    let _ = reporter.try_send(sse("progress", &p));
                }
            })
        };
        let mut say = |message: &str| {
            let _ = reporter.blocking_send(sse(
                "progress",
                &Progress::Status { message: message.to_string() },
            ));
        };
        let outcome = match weights::fetch_watched(&repo, &mut say, &watch) {
            Ok(_) => sse("pulled", &json!({ "repo": repo })),
            Err(e) => sse("error", &json!({ "error": e.to_string() })),
        };
        let _ = reporter.blocking_send(outcome);
    });
    drop(events);

    Ok(stream(rx))
}

fn progress_of(f: weights::Fetch) -> Option<Progress> {
    match f {
        weights::Fetch::Download { total: 0, .. }
        | weights::Fetch::Local
        | weights::Fetch::Shards(_) => None,
        weights::Fetch::Download { file, bytes, total } => {
            Some(Progress::Download { file, bytes, total })
        }
        weights::Fetch::Fetched { file } => Some(Progress::Fetched { file }),
    }
}

#[derive(serde::Deserialize)]
pub struct IdQuery {
    /// A repo id or a trained model's name. In the query string rather than
    /// the path because repo ids contain slashes.
    id: String,
}

/// Delete a downloaded or trained model.
pub async fn remove(
    _: Admin,
    St(state): St<State>,
    Query(query): Query<IdQuery>,
) -> Result<Json<serde_json::Value>, Fail> {
    let id = query.id.trim().to_string();
    if id.is_empty() {
        return Err(Fail::bad("no model was named"));
    }
    // Refusing to delete what is loaded, rather than deleting it and leaving
    // the engine holding memory-mapped weights whose file is gone.
    if state.engine.loaded().is_some_and(|l| l.repo == id) {
        return Err(Fail::bad(format!("{id} is loaded; unload it first")));
    }

    // A trained model first: its name has no slash in it and so cannot
    // collide with a repo id, and a model trained here is the one whose
    // deletion is irreversible.
    let target = {
        let id = id.clone();
        blocking(move || {
            Ok(match hub::find_trained(&id) {
                Some(m) => Some((m.path, true)),
                None => hub::find_local(&id).map(|m| (m.path, false)),
            })
        })
        .await?
    };
    let Some((path, trained)) = target else {
        return Err(Fail::missing(format!("{id} is not on this machine")));
    };

    let gone = id.clone();
    blocking(move || {
        std::fs::remove_dir_all(&path)?;
        // The quantised weights are derived from what just went; keeping them
        // would mean a re-download silently reusing weights from a file that
        // no longer exists to compare against.
        let forgotten = kvad::qcache::forget(&gone);
        if hub::State::active().as_deref() == Some(gone.as_str()) {
            hub::State::clear()?;
        }
        Ok(forgotten)
    })
    .await
    .map(|forgotten| Json(json!({ "deleted": id, "trained": trained, "qcache_files": forgotten })))
}

#[derive(serde::Deserialize)]
pub struct RepoQuery {
    repo: String,
}

/// Throw away the pre-quantised weights for a model. They rebuild on the next
/// load, more slowly; nothing is lost but time.
pub async fn forget_qcache(
    _: Admin,
    Query(query): Query<RepoQuery>,
) -> Result<Json<serde_json::Value>, Fail> {
    let repo = query.repo.clone();
    let files = blocking(move || Ok(kvad::qcache::forget(&repo))).await?;
    Ok(Json(json!({ "repo": query.repo, "files": files })))
}

/// One SSE event, named, carrying JSON.
///
/// Named events rather than one stream of anonymous ones because the client
/// has to tell "still going" from "finished" from "failed", and a `kind`
/// field inside the payload would make every reader parse before it can
/// dispatch.
pub fn sse(name: &str, data: &impl serde::Serialize) -> Event {
    Event::default().event(name).json_data(data).unwrap_or_else(|e| {
        Event::default().event("error").data(format!(r#"{{"error":"could not encode: {e}"}}"#))
    })
}

/// Wrap a channel of events as an SSE response.
pub fn stream(
    rx: tokio::sync::mpsc::Receiver<Event>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // A keep-alive comment every fifteen seconds, so that a proxy between
    // here and the browser does not decide a quiet load has died.
    Sse::new(ReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::model::Arch;
    use std::path::PathBuf;

    fn local(id: &str, arch: Option<Arch>, complete: bool) -> hub::LocalModel {
        hub::LocalModel { id: id.into(), path: PathBuf::new(), bytes: 1, arch, complete }
    }

    /// Every unrunnable model says why in a sentence somebody can act on, and
    /// a trained one gets a different sentence from a downloaded one: you
    /// cannot re-download a model you trained.
    #[test]
    fn a_model_that_cannot_run_says_why() {
        let good = describe(&local("a/b", Some(Arch::Llama), true), false);
        assert!(good.runnable && good.blocker.is_none());
        assert_eq!(good.arch.as_deref(), Some("llama"));

        let strange = describe(&local("a/b", None, true), false);
        assert!(!strange.runnable);
        assert!(strange.blocker.unwrap().contains("neither"));

        let half = describe(&local("a/b", Some(Arch::Gpt2), false), false);
        assert!(half.blocker.unwrap().contains("download did not finish"));

        let stopped = describe(&local("mine", Some(Arch::Gpt2), false), true);
        assert!(stopped.blocker.unwrap().contains("first checkpoint"));
    }

    /// An unset, unreadable or impossible stored backend all mean the same
    /// thing: use the one that works everywhere.
    #[test]
    fn the_default_backend_falls_back_rather_than_failing() {
        let db = crate::db::Db::in_memory().unwrap();
        assert_eq!(default_backend(&db), "cpu-q8");

        db.set_setting(BACKEND_KEY, &json!("cpu-f32")).unwrap();
        assert_eq!(default_backend(&db), "cpu-f32");

        db.set_setting(BACKEND_KEY, &json!("nonsense")).unwrap();
        assert_eq!(default_backend(&db), "cpu-q8");

        // Not a string at all — something wrote the wrong shape.
        db.set_setting(BACKEND_KEY, &json!(17)).unwrap();
        assert_eq!(default_backend(&db), "cpu-q8");
    }
}
