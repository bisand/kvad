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
use crate::engine::For;
use axum::extract::{Query, State as St};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use kvad::hub;
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
    // Ordered by how specific the answer is. A half-finished download is the
    // likeliest reason a directory here cannot be run, and it used to be
    // answered with the architecture message instead — the GPTQ repo that has
    // only its `model.safetensors.index.json` was being told it was the wrong
    // kind of model rather than an unfinished one.
    let blocker = match (m.complete, &m.model_type, m.arch) {
        (false, _, _) => Some(match trained {
            true => "no weights yet — the run was stopped before its first checkpoint".into(),
            false => "the download did not finish".to_string(),
        }),
        (_, None, _) => Some("no config.json, so there is nothing to say what this is".into()),
        // Named by the list the loader dispatches on rather than by a copy of
        // it kept here, which is how this came to be offering GPT-2 and the
        // Llama family long after DeepSeek arrived.
        (_, Some(t), None) => Some(format!(
            "`{t}` is not an architecture this build runs; it runs {}",
            kvad::model::arch::supported()
        )),
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
///
/// A stored setting is somebody's choice and wins. With none, the answer is
/// what this build prefers for this model — see [`crate::engine::preferred`],
/// which is where the measurements are.
///
/// `repo` is the model about to be loaded, where there is one. It decides
/// nothing but the architecture, and the architecture decides only whether
/// the GPU is an option.
///
/// Blocking, and for a repo that is not on this disk it may spend a Hub round
/// trip finding out what that repo is — see [`known_about`]. Callers already
/// run it on a blocking thread; the point is that it is on the way to a load,
/// where seconds are the unit.
pub fn default_backend(db: &crate::db::Db, repo: Option<&str>) -> String {
    backend_for(db, repo, hub::remote_arch)
}

/// [`default_backend`], with the Hub lookup handed in.
///
/// Only so the tests can have both of its answers without a network. Note
/// that `ask` is never reached when somebody has stored a choice, which is
/// what keeps a configured server from calling out at all.
fn backend_for(
    db: &crate::db::Db,
    repo: Option<&str>,
    ask: impl FnOnce(&str) -> Option<kvad::model::Arch>,
) -> String {
    let stored = db.setting(BACKEND_KEY).ok().flatten();
    let stored = stored.as_ref().and_then(|v| v.as_str()).unwrap_or("");
    match crate::engine::parse(stored) {
        // A stored backend that this build cannot load — a GPU one in a
        // `--no-default-features` binary — is treated as unset rather than as
        // an error every load has to explain.
        Some(b) => crate::engine::id_of(b),
        None => crate::engine::id_of(crate::engine::preferred(known_about(repo, ask))),
    }
}

/// What can be said about `repo` before anything is loaded.
///
/// Three sources, cheapest first. A model on this disk has a config that has
/// already been read. A model named but not downloaded has one on the Hub,
/// and asking costs a metadata request — worth it, because the alternative is
/// assuming the worst about every model the moment before downloading it, and
/// that assumption is the difference between a first load on the GPU and a
/// first load on the CPU. Anything else leaves nothing to go on: a path or a
/// trained name is not a repo id and [`hub::remote_arch`] refuses it without
/// a request, and a Hub that cannot be reached says nothing either.
///
/// The two ways of knowing nothing stay apart, because [`For::Anything`] is
/// the picker asking what this build likes and [`For::Unknown`] is a specific
/// model nobody can describe.
fn known_about(repo: Option<&str>, ask: impl FnOnce(&str) -> Option<kvad::model::Arch>) -> For {
    let Some(repo) = repo else { return For::Anything };
    if let Some(arch) =
        hub::find_local(repo).or_else(|| hub::find_trained(repo)).and_then(|m| m.arch)
    {
        return For::This(arch);
    }
    match ask(repo) {
        Some(arch) => For::This(arch),
        None => For::Unknown,
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
            // No model in hand: the picker is asking what this build
            // prefers in general.
            default_backend(&db, None),
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
    /// Weights, as the Hub counts them from the safetensors headers.
    params: Option<u64>,
    /// Bytes to download.
    bytes: Option<u64>,
    /// What the weights would occupy here, per precision — which is not the
    /// download size, because loading quantises.
    memory: Option<Memory>,
    /// The best precision whose weights fit in this machine's memory, or
    /// `null` for "nothing fits" / "we could not tell".
    fits_at: Option<String>,
    /// Which of those two `fits_at: null` means.
    size_known: bool,
}

/// What a model costs here, at each precision this engine offers.
#[derive(serde::Serialize)]
pub struct Memory {
    f32: u64,
    q8: u64,
    q4: u64,
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
                params: m.params,
                bytes: m.download_bytes,
                memory: m.params.map(|p| Memory {
                    f32: kvad::quant::Precision::F32.weight_bytes(p),
                    q8: kvad::quant::Precision::Q8.weight_bytes(p),
                    q4: kvad::quant::Precision::Q4.weight_bytes(p),
                }),
                fits_at: match m.fit() {
                    hub::Fit::At(p) => Some(p.to_string()),
                    hub::Fit::TooBig { .. } | hub::Fit::Unknown => None,
                },
                size_known: m.params.is_some(),
                id: m.id.clone(),
            })
            .collect(),
    ))
}

#[derive(serde::Deserialize)]
pub struct LoadRequest {
    repo: String,
    /// A backend id from the listing, e.g. `cpu-q8`. Naming one chooses it
    /// and remembers it; omitting it uses whichever was chosen last, or —
    /// with nothing ever chosen — what this build prefers for this model.
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
            let named = repo.clone();
            blocking(move || Ok(default_backend(&db, Some(&named)))).await?
        }
    };
    let backend = crate::engine::parse(&wanted).ok_or_else(|| {
        Fail::bad(format!(
            "`{wanted}` is not a backend this build can load; it has {}",
            crate::engine::available().iter().map(|c| c.id.clone()).collect::<Vec<_>>().join(", ")
        ))
    })?;

    // Remembered only when the request named one. A backend that was worked
    // out from `default_backend` is not a choice anybody made, and storing
    // it would turn this machine's preference into this machine's setting —
    // after which the preference stops being consulted, and a model whose
    // architecture wants a different answer gets the one the last load
    // happened to use.
    //
    // Remembered before the load rather than after, so that a load which
    // fails halfway still leaves the picker showing what was asked for.
    if body.backend.is_some() {
        let db = state.db.clone();
        let remember = crate::engine::id_of(backend);
        blocking(move || db.set_setting(BACKEND_KEY, &json!(remember))).await?;
    }

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
/// A job rather than a stream, since Phase 4. A checkpoint of several
/// gigabytes takes longer than a browser tab reliably stays open, and a
/// download that dies because somebody navigated away is a download that has
/// to start again. The answer is the job; watch it at
/// `/api/jobs/{id}/events`.
///
/// Still not scheduler work: a download is network and disk, so making it
/// wait behind a generation would be a queue for no reason. That is why the
/// Models page can pull one model while somebody chats with another.
pub async fn pull(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<PullRequest>,
) -> Result<Json<crate::jobs::Job>, Fail> {
    let repo = body.repo.trim().to_string();
    if repo.is_empty() {
        return Err(Fail::bad("no model was named"));
    }
    let jobs = state.jobs.clone();
    let owner = who.0.id;
    blocking(move || crate::jobs::pull(&jobs, repo, owner))
        .await
        .map(Json)
        .map_err(|e| Fail::bad(e.1))
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
    /// Which of that model's files, by the tag the listing shows: `q8`,
    /// `gpu-q4`.
    ///
    /// Required, and deliberately. Every row in the listing is its own file,
    /// and an endpoint that deletes all of them when a caller forgets a
    /// parameter is the bug this field exists to close — the button used to
    /// send the repo alone, so throwing away one row threw away every
    /// precision that model had been quantised at.
    precision: String,
}

/// Throw away one file of pre-quantised weights: a model at one precision.
/// It rebuilds on the next load at that precision, more slowly; nothing is
/// lost but time.
pub async fn forget_qcache(
    _: Admin,
    Query(query): Query<RepoQuery>,
) -> Result<Json<serde_json::Value>, Fail> {
    let (repo, precision) = (query.repo.clone(), query.precision.clone());
    let files = blocking(move || Ok(kvad::qcache::forget_one(&repo, &precision))).await?;
    if files == 0 {
        // The row came from a listing, so nothing there means somebody else
        // got to it first. Saying so beats reporting a deletion that did not
        // happen.
        return Err(Fail::missing(format!(
            "there is no {} cache for {}",
            query.precision, query.repo
        )));
    }
    Ok(Json(json!({ "repo": query.repo, "precision": query.precision, "files": files })))
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

    /// `arch` is derived from `model_type` exactly as `local_models` derives
    /// it, so a test cannot disagree with the registry about which types this
    /// build answers to.
    fn local(id: &str, model_type: Option<&str>, complete: bool) -> hub::LocalModel {
        hub::LocalModel {
            id: id.into(),
            path: PathBuf::new(),
            bytes: 1,
            arch: model_type.and_then(Arch::from_model_type),
            model_type: model_type.map(str::to_string),
            complete,
        }
    }

    /// Every unrunnable model says why in a sentence somebody can act on, and
    /// the three reasons are three sentences.
    ///
    /// The one that was wrong is the fourth case below. A directory with no
    /// config and no weights — the ordinary shape of an interrupted download,
    /// and what `Qwen2.5-Coder-7B-Instruct-GPTQ-Int4` looked like on this
    /// machine — has no `model_type` either, so it was matching the
    /// architecture arm and being told it was the wrong kind of model rather
    /// than an unfinished one.
    #[test]
    fn a_model_that_cannot_run_says_why() {
        let good = describe(&local("a/b", Some("llama"), true), false);
        assert!(good.runnable && good.blocker.is_none());
        assert_eq!(good.arch.as_deref(), Some("llama"));

        // Downloaded, readable, and something this build has no loader for.
        // It names the model's own `model_type` and the list the loader
        // dispatches on, rather than a sentence kept here about which
        // architectures existed when it was written.
        let strange = describe(&local("a/b", Some("whisper"), true), false);
        assert!(!strange.runnable);
        let why = strange.blocker.unwrap();
        assert!(why.contains("`whisper`"), "{why}");
        assert!(why.contains("llama") && why.contains("gpt2"), "{why}");

        let nothing = describe(&local("a/b", None, true), false);
        assert!(nothing.blocker.unwrap().contains("no config.json"));

        let interrupted = describe(&local("a/b", None, false), false);
        assert!(interrupted.blocker.unwrap().contains("download did not finish"));

        let half = describe(&local("a/b", Some("gpt2"), false), false);
        assert!(half.blocker.unwrap().contains("download did not finish"));

        let stopped = describe(&local("mine", Some("gpt2"), false), true);
        assert!(stopped.blocker.unwrap().contains("first checkpoint"));
    }

    /// An unset, unreadable or impossible stored backend all mean the same
    /// thing: fall back to what this build prefers rather than fail.
    #[test]
    fn the_default_backend_falls_back_rather_than_failing() {
        let db = crate::db::Db::in_memory().unwrap();
        // Nothing stored and no model named: the build's preference, which
        // is the GPU where there is one.
        let prefers = crate::engine::id_of(crate::engine::preferred(For::Anything));
        assert_eq!(default_backend(&db, None), prefers);

        db.set_setting(BACKEND_KEY, &json!("cpu-f32")).unwrap();
        assert_eq!(default_backend(&db, None), "cpu-f32", "a chosen backend is a choice");

        db.set_setting(BACKEND_KEY, &json!("nonsense")).unwrap();
        assert_eq!(default_backend(&db, None), prefers);

        // Not a string at all — something wrote the wrong shape.
        db.set_setting(BACKEND_KEY, &json!(17)).unwrap();
        assert_eq!(default_backend(&db, None), prefers);
    }

    /// A model nobody has downloaded is asked about, not guessed at — and
    /// when nobody can say what it is, it gets the backend that loads
    /// anything.
    ///
    /// The Hub lookup is handed in here so that both of its answers can be
    /// had without a network. A test that reached huggingface.co would be
    /// testing the network.
    #[test]
    fn a_model_this_machine_does_not_have_is_asked_about_not_guessed_at() {
        let db = crate::db::Db::in_memory().unwrap();
        let llama = kvad::model::Arch::require("llama");

        // Nobody can say what it is: offline, or no such repo. Guessing the
        // GPU here would be a default that fails to load.
        assert_eq!(backend_for(&db, Some("nobody/has-this-model"), |_| None), "cpu-q8");
        assert!(matches!(known_about(Some("nobody/has-this-model"), |_| None), For::Unknown));

        // Not downloaded, and the Hub knows what it is, which is enough to
        // choose properly rather than conservatively.
        assert_eq!(
            backend_for(&db, Some("somebody/a-llama"), |_| Some(llama)),
            crate::engine::id_of(crate::engine::preferred(For::This(llama))),
        );

        // A stored choice is a choice, and settles it before anyone is asked.
        db.set_setting(BACKEND_KEY, &json!("cpu-f32")).unwrap();
        let answer = backend_for(&db, Some("somebody/a-llama"), |_| {
            panic!("asked the Hub about a model somebody had already chosen a backend for")
        });
        assert_eq!(answer, "cpu-f32");

        assert!(matches!(
            known_about(None, |_| panic!("asked the Hub about no model in particular")),
            For::Anything
        ));
    }
}
