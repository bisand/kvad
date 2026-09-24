//! `/v1/images/generations`, and where the images it makes are kept.
//!
//! The endpoint is OpenAI's shape, for the same reason `/v1/chat/completions`
//! is: a client that already makes images with OpenAI makes them here by
//! changing a URL. What OpenAI has no field for — steps, guidance, a seed, a
//! negative prompt — is accepted beside its fields under the names diffusers
//! uses, and what this engine measured goes back in a `kvad` object.
//!
//! # Where images live
//!
//! Every image made is kept, because an image is somebody's work in a way a
//! chat reply is not: it took a minute of a GPU, it cannot be made again
//! without its seed, and the person who asked for it will want it tomorrow.
//! A PNG goes to `images/<id>.png` in the data directory — beside the
//! database and the datasets, not in a cache, because it is the only copy —
//! and a row in `images` says how it was made. Nothing expires them; the
//! gallery's delete does.
//!
//! `response_format: "url"` answers with a link to that file rather than the
//! bytes. The link is this server's, and it asks for the same credential the
//! request did. That is not what OpenAI's links are (theirs are signed and
//! public for an hour), so `b64_json` is the default here: a client that
//! follows a link without its key would otherwise get a 401 where it
//! expected a picture.
//!
//! # Streaming
//!
//! With `stream: true` the answer is server-sent events: one
//! `image_generation.step` per denoising step (kvad's own), an
//! `image_generation.partial_image` beside it when previews were asked for
//! (OpenAI's name, via `partial_images`, or kvad's `preview`), and one
//! `image_generation.completed` per image. A client that goes away stops the
//! generation at the next step.

use crate::api::{blocking, Fail};
use crate::auth::{Identity, State};
use crate::db::Db;
use crate::models::{sse, stream};
use crate::scheduler::{Kind, Resident, Stroke};
use axum::extract::{Path, State as St};
use axum::http::{header, HeaderMap};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use kvad::image::{ImageRequest, Painted};
use rusqlite::{params, Row};
use serde_json::json;
use std::path::{Path as FsPath, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/v1/images/generations", post(generations))
        .route("/api/images", get(gallery))
        .route("/api/images/{id}", get(file).delete(remove))
}

/// Where the PNGs are: `images` in the data directory.
pub fn dir() -> PathBuf {
    kvad::weights::data_dir().join("images")
}

/// The most images one request may ask for. Each is a minute or so of the
/// GPU that every other request waits behind.
const MAX_N: usize = 4;

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// One image, as the gallery lists it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Stored {
    pub id: i64,
    pub model: String,
    pub backend: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: u64,
    pub bytes: u64,
    pub secs: f64,
    pub created_at: String,
    /// Where the picture is, on this server.
    pub url: String,
}

const COLUMNS: &str = "id, model, backend, prompt, negative_prompt, width, height, steps, guidance, \
                       seed, bytes, secs, created_at";

fn stored_from(r: &Row<'_>) -> rusqlite::Result<Stored> {
    let id: i64 = r.get(0)?;
    Ok(Stored {
        id,
        model: r.get(1)?,
        backend: r.get(2)?,
        prompt: r.get(3)?,
        negative_prompt: r.get(4)?,
        width: r.get(5)?,
        height: r.get(6)?,
        steps: r.get(7)?,
        guidance: r.get(8)?,
        seed: r.get::<_, i64>(9)? as u64,
        bytes: r.get::<_, i64>(10)? as u64,
        secs: r.get(11)?,
        url: url_of(id, &r.get::<_, String>(12)?),
        created_at: r.get(12)?,
    })
}

/// The picture's link, which is served as immutable and so has to name one
/// picture forever. The id alone did not: ids were reused before migration
/// 009, and start again at 1 in a data directory that was wiped. The time it
/// was made, as digits, is what tells those apart.
fn url_of(id: i64, created_at: &str) -> String {
    let made: String = created_at.chars().filter(char::is_ascii_digit).collect();
    format!("/api/images/{id}.png?v={made}")
}

/// Keep an image: its row, then its file, and neither if the file cannot be
/// written.
pub fn save(db: &Db, dir: &FsPath, owner: Option<i64>, model: &str, backend: &str, p: &Painted) -> Res<Stored> {
    let png = p.image.png();
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let r = &p.request;
    let secs = p.encode_secs + p.denoise_secs + p.decode_secs;
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO images (owner, model, backend, prompt, negative_prompt, width, height, steps, \
             guidance, seed, bytes, secs) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                owner,
                model,
                backend,
                r.prompt,
                r.negative_prompt,
                r.width as i64,
                r.height as i64,
                r.steps as i64,
                r.guidance as f64,
                r.seed as i64,
                png.len() as i64,
                secs
            ],
        )?;
        Ok(c.last_insert_rowid())
    })?;

    // Written aside and renamed into place, so that nobody is ever served
    // half a picture.
    let path = dir.join(format!("{id}.png"));
    let aside = dir.join(format!("{id}.png.part"));
    let written = std::fs::write(&aside, &png).and_then(|()| std::fs::rename(&aside, &path));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&aside);
        let _ = db.with(|c| c.execute("DELETE FROM images WHERE id = ?1", [id]));
        return Err(format!("could not write {}: {e}", path.display()).into());
    }
    get_one(db, id, owner)?.ok_or_else(|| "the image was saved and then was not there".into())
}

pub fn list(db: &Db, owner: Option<i64>) -> Res<Vec<Stored>> {
    db.with(|c| {
        let mut q = c.prepare(&format!("SELECT {COLUMNS} FROM images WHERE owner IS ?1 ORDER BY id DESC"))?;
        let rows = q.query_map([owner], stored_from)?.collect();
        rows
    })
}

pub fn get_one(db: &Db, id: i64, owner: Option<i64>) -> Res<Option<Stored>> {
    db.with(|c| {
        match c.query_row(
            &format!("SELECT {COLUMNS} FROM images WHERE id = ?2 AND owner IS ?1"),
            params![owner, id],
            stored_from,
        ) {
            Ok(s) => Ok(Some(s)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    })
}

/// Forget an image, row and file. `false` when there was no such image of
/// `owner`'s — somebody else's is not found rather than refused.
pub fn delete(db: &Db, dir: &FsPath, id: i64, owner: Option<i64>) -> Res<bool> {
    let gone = db.with(|c| c.execute("DELETE FROM images WHERE id = ?2 AND owner IS ?1", params![owner, id]))?;
    if gone == 0 {
        return Ok(false);
    }
    match std::fs::remove_file(dir.join(format!("{id}.png"))) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(true),
    }
}

// ---------------------------------------------------------------------------
// The gallery
// ---------------------------------------------------------------------------

async fn gallery(who: Identity, St(state): St<State>) -> Result<Json<Vec<Stored>>, Fail> {
    let db = state.db.clone();
    blocking(move || list(&db, who.id)).await.map(Json)
}

/// `12` or `12.png`: the id in a path, whichever way it was written.
fn id_in(name: &str) -> Result<i64, Fail> {
    name.strip_suffix(".png").unwrap_or(name).parse().map_err(|_| Fail::missing(format!("there is no image {name}")))
}

async fn file(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Response, Fail> {
    let id = id_in(&name)?;
    let db = state.db.clone();
    let bytes = blocking(move || match get_one(&db, id, who.id)? {
        Some(_) => Ok(Some(std::fs::read(dir().join(format!("{id}.png")))?)),
        None => Ok(None),
    })
    .await?
    .ok_or_else(|| Fail::missing(format!("there is no image {id}")))?;
    Ok((
        [
            (header::CONTENT_TYPE, "image/png"),
            // The link names one picture forever (see `url_of`), so a browser
            // may keep it for as long as it likes — but only for this person.
            (header::CACHE_CONTROL, "private, max-age=31536000, immutable"),
        ],
        bytes,
    )
        .into_response())
}

async fn remove(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Json<serde_json::Value>, Fail> {
    let id = id_in(&name)?;
    let db = state.db.clone();
    match blocking(move || delete(&db, &dir(), id, who.id)).await? {
        true => Ok(Json(json!({ "deleted": id }))),
        false => Err(Fail::missing(format!("there is no image {id}"))),
    }
}

// ---------------------------------------------------------------------------
// /v1/images/generations
// ---------------------------------------------------------------------------

/// OpenAI's request, and diffusers' names for what it does not have.
#[derive(serde::Deserialize)]
pub struct Generations {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    n: Option<usize>,
    /// `1024x1024`, or `auto` for the model's own.
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    stream: bool,
    /// OpenAI's way of asking for previews while it streams. Any number above
    /// zero means one per step here: they cost nothing to make.
    #[serde(default)]
    partial_images: Option<usize>,
    #[serde(default)]
    preview: Option<bool>,
    #[serde(default)]
    negative_prompt: Option<String>,
    /// `num_inference_steps` in diffusers; either name is taken.
    #[serde(default, alias = "num_inference_steps")]
    steps: Option<usize>,
    #[serde(default)]
    guidance_scale: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    B64,
    Url,
}

impl Generations {
    fn request(&self) -> Result<ImageRequest, Fail> {
        let (width, height) = match self.size.as_deref().map(str::trim) {
            None | Some("") | Some("auto") => (None, None),
            Some(s) => {
                let (w, h) = s
                    .split_once(['x', 'X', '×'])
                    .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
                    .ok_or_else(|| Fail::bad(format!("size is WIDTHxHEIGHT, like 1024x1024, not {s:?}")))?;
                (Some(w), Some(h))
            }
        };
        Ok(ImageRequest {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            width,
            height,
            steps: self.steps,
            guidance: self.guidance_scale,
            seed: self.seed,
            preview: self.preview.unwrap_or(false) || self.partial_images.unwrap_or(0) > 0,
        })
    }

    fn format(&self) -> Result<Format, Fail> {
        match self.response_format.as_deref() {
            None | Some("b64_json") => Ok(Format::B64),
            Some("url") => Ok(Format::Url),
            Some(other) => Err(Fail::bad(format!("response_format is b64_json or url, not {other:?}"))),
        }
    }
}

pub async fn generations(
    who: Identity,
    headers: HeaderMap,
    St(state): St<State>,
    Json(body): Json<Generations>,
) -> Result<Response, Fail> {
    let mut request = body.request()?;
    let format = body.format()?;
    let n = body.n.unwrap_or(1);
    if n == 0 || n > MAX_N {
        return Err(Fail::bad(format!("n is between 1 and {MAX_N}, not {n}")));
    }

    let resident = crate::openai::resident_for(&state, body.model.as_deref(), Kind::Image).await?;
    let defaults = resident
        .model
        .image
        .ok_or_else(|| Fail::internal(format!("{} loaded as an image model with no defaults", resident.model.repo)))?;
    // Checked here, against the model's own limits, so that a size the model
    // cannot draw is a 400 before anything is queued rather than a failure
    // after a wait. The seed is fixed now too: several images are that seed
    // and the ones after it, so each can be made again on its own.
    let resolved = request.resolved(&defaults).map_err(|e| Fail::bad(e.to_string()))?;
    request.seed = Some(resolved.seed);

    let job = Job { state, owner: who.id, resident, request, n, format, base: base_url(&headers) };
    Ok(match body.stream {
        true => streamed(job).into_response(),
        false => whole(job).await?.into_response(),
    })
}

/// What a generation needs once the request has been read.
struct Job {
    state: State,
    owner: Option<i64>,
    resident: Resident,
    request: ImageRequest,
    n: usize,
    format: Format,
    /// `http://host:port` as the client reached us, for absolute links.
    base: Option<String>,
}

impl Job {
    /// The request for the `i`th image: the same, one seed along.
    fn nth(&self, i: usize) -> ImageRequest {
        ImageRequest { seed: self.request.seed.map(|s| s.wrapping_add(i as u64)), ..self.request.clone() }
    }

    async fn keep(&self, painted: &Painted) -> Result<Stored, Fail> {
        let (db, owner, model) = (self.state.db.clone(), self.owner, self.resident.model.repo.clone());
        let backend = self.resident.model.backend.clone();
        let painted = painted.clone();
        blocking(move || save(&db, &dir(), owner, &model, &backend, &painted)).await
    }

    /// One image in OpenAI's shape, with how it was made in `kvad`.
    fn datum(&self, stored: &Stored, painted: &Painted) -> serde_json::Value {
        let link = match &self.base {
            Some(base) => format!("{base}{}", stored.url),
            None => stored.url.clone(),
        };
        let mut d = match self.format {
            Format::B64 => json!({ "b64_json": b64(&painted.image.png()) }),
            Format::Url => json!({ "url": link }),
        };
        d["kvad"] = json!({
            "id": stored.id,
            "url": stored.url,
            "seed": stored.seed,
            "width": stored.width,
            "height": stored.height,
            "steps": stored.steps,
            "guidance": stored.guidance,
            "encode_secs": painted.encode_secs,
            "denoise_secs": painted.denoise_secs,
            "decode_secs": painted.decode_secs,
        });
        d
    }
}

async fn whole(job: Job) -> Result<Json<serde_json::Value>, Fail> {
    let mut data = Vec::with_capacity(job.n);
    for i in 0..job.n {
        let mut strokes = job.state.engine.paint(&job.resident.key, job.nth(i)).map_err(Fail::internal)?;
        let painted = loop {
            match strokes.recv().await {
                Some(Stroke::Step(_)) => continue,
                Some(Stroke::Done(p)) => break *p,
                Some(Stroke::Failed(e)) => return Err(Fail::internal(e)),
                None => return Err(Fail::internal("the engine stopped before it answered")),
            }
        };
        let stored = job.keep(&painted).await?;
        data.push(job.datum(&stored, &painted));
    }
    Ok(Json(json!({
        "created": now_secs(),
        "data": data,
        "kvad": { "model": job.resident.model.repo, "backend": job.resident.model.backend },
    })))
}

fn streamed(job: Job) -> impl IntoResponse {
    let (events, rx) = tokio::sync::mpsc::channel::<Event>(16);
    tokio::spawn(async move {
        for i in 0..job.n {
            let mut strokes = match job.state.engine.paint(&job.resident.key, job.nth(i)) {
                Ok(s) => s,
                Err(e) => {
                    let _ = events.send(sse("error", &json!({ "error": e }))).await;
                    return;
                }
            };
            while let Some(stroke) = strokes.recv().await {
                let sent = match stroke {
                    Stroke::Step(step) => {
                        let tick = json!({
                            "type": "image_generation.step",
                            "index": i,
                            "step": step.done,
                            "total": step.total,
                        });
                        let mut ok = events.send(sse("image_generation.step", &tick)).await.is_ok();
                        if let Some(preview) = step.preview {
                            let partial = json!({
                                "type": "image_generation.partial_image",
                                "index": i,
                                "partial_image_index": step.done - 1,
                                "b64_json": b64(&preview.png()),
                                "size": format!("{}x{}", preview.width, preview.height),
                                "output_format": "png",
                                "created_at": now_secs(),
                            });
                            ok = ok && events.send(sse("image_generation.partial_image", &partial)).await.is_ok();
                        }
                        ok
                    }
                    Stroke::Done(painted) => {
                        let done = match job.keep(&painted).await {
                            Ok(stored) => {
                                let mut d = job.datum(&stored, &painted);
                                d["type"] = json!("image_generation.completed");
                                d["index"] = json!(i);
                                d["created_at"] = json!(now_secs());
                                d["size"] = json!(format!("{}x{}", stored.width, stored.height));
                                d["output_format"] = json!("png");
                                sse("image_generation.completed", &d)
                            }
                            Err(Fail(_, why)) => sse("error", &json!({ "error": why })),
                        };
                        let _ = events.send(done).await;
                        break;
                    }
                    Stroke::Failed(e) => {
                        let _ = events.send(sse("error", &json!({ "error": e }))).await;
                        return;
                    }
                };
                // The client has gone. Dropping `strokes` is what tells the
                // scheduler to stop at the next step.
                if !sent {
                    return;
                }
            }
        }
    });
    stream(rx)
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// `scheme://host` as the client addressed this server, when it said.
fn base_url(headers: &HeaderMap) -> Option<String> {
    let host = headers.get(header::HOST)?.to_str().ok()?;
    let scheme = headers.get("x-forwarded-proto").and_then(|v| v.to_str().ok()).unwrap_or("http");
    Some(format!("{scheme}://{host}"))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::image::{Image, Resolved};

    fn painted(seed: u64) -> Painted {
        Painted {
            image: Image { width: 2, height: 1, rgb: vec![255, 0, 0, 0, 0, 255] },
            request: Resolved {
                prompt: "a red and a blue pixel".into(),
                negative_prompt: None,
                width: 2,
                height: 1,
                steps: 3,
                guidance: 5.0,
                seed,
                preview: false,
            },
            encode_secs: 0.1,
            denoise_secs: 1.0,
            decode_secs: 0.2,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvad-images-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A kept image is a row and a file, both, and a u64 seed survives the
    /// trip through SQLite's i64 bit for bit.
    #[test]
    fn an_image_is_kept_as_a_row_and_a_file_and_forgotten_as_both() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("kept");
        let seed = u64::MAX - 7;
        let s = save(&db, &dir, None, "stabilityai/sdxl", "metal f16", &painted(seed)).unwrap();
        assert_eq!(s.seed, seed);
        assert_eq!((s.width, s.height, s.steps), (2, 1, 3));
        assert!(s.url.starts_with(&format!("/api/images/{}.png?v=", s.id)), "{}", s.url);
        let file = dir.join(format!("{}.png", s.id));
        assert_eq!(std::fs::read(&file).unwrap(), painted(seed).image.png());
        assert_eq!(list(&db, None).unwrap().len(), 1);

        assert!(delete(&db, &dir, s.id, None).unwrap());
        assert!(!file.exists());
        assert!(list(&db, None).unwrap().is_empty());
        assert!(!delete(&db, &dir, s.id, None).unwrap(), "a second delete finds nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Deleting the newest image and making another must not give the new
    /// one the old id: the link is cached as immutable, and a browser that
    /// saw the old picture there would go on showing it.
    #[test]
    fn a_deleted_images_id_is_not_handed_out_again() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("reused");
        let first = save(&db, &dir, None, "m", "b", &painted(1)).unwrap();
        assert!(delete(&db, &dir, first.id, None).unwrap());
        let second = save(&db, &dir, None, "m", "b", &painted(2)).unwrap();
        assert!(second.id > first.id, "id {} was handed out again", first.id);
        assert_ne!(second.url, first.url);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same id in a wiped data directory is a different picture, and its
    /// link has to say so.
    #[test]
    fn a_link_names_the_moment_as_well_as_the_id() {
        assert_eq!(url_of(1, "2026-09-24 05:09:48"), "/api/images/1.png?v=20260924050948");
        assert_ne!(url_of(1, "2026-09-24 05:09:48"), url_of(1, "2026-09-25 10:00:00"));
    }

    /// Somebody else's image is not found, the same way somebody else's
    /// conversation is not: the answer does not confirm it exists.
    #[test]
    fn another_accounts_images_are_not_found() {
        let db = Db::in_memory().unwrap();
        db.with(|c| {
            c.execute("INSERT INTO users (id, name, role) VALUES (1, 'a', 'user'), (2, 'b', 'user')", [])
        })
        .unwrap();
        let dir = scratch("owned");
        let mine = save(&db, &dir, Some(1), "m", "b", &painted(1)).unwrap();
        assert!(get_one(&db, mine.id, Some(2)).unwrap().is_none());
        assert!(list(&db, Some(2)).unwrap().is_empty());
        assert!(!delete(&db, &dir, mine.id, Some(2)).unwrap());
        assert!(get_one(&db, mine.id, Some(1)).unwrap().is_some(), "and the owner still has it");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn body(v: serde_json::Value) -> Generations {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn a_request_is_read_in_openai_s_names_and_diffusers_names() {
        let g = body(json!({
            "prompt": "a cat", "size": "768x512", "num_inference_steps": 20,
            "guidance_scale": 6.5, "seed": 3, "partial_images": 2, "response_format": "url"
        }));
        let r = g.request().unwrap();
        assert_eq!((r.width, r.height, r.steps, r.guidance, r.seed), (Some(768), Some(512), Some(20), Some(6.5), Some(3)));
        assert!(r.preview);
        assert!(g.format().unwrap() == Format::Url);

        let auto = body(json!({ "prompt": "a cat", "size": "auto" })).request().unwrap();
        assert_eq!((auto.width, auto.height), (None, None));
        assert!(body(json!({ "prompt": "a cat" })).format().unwrap() == Format::B64);
        assert!(body(json!({ "prompt": "a cat", "size": "big" })).request().is_err());
        assert!(body(json!({ "prompt": "a cat", "response_format": "gif" })).format().is_err());
    }

    #[test]
    fn an_id_is_read_with_or_without_its_extension() {
        assert_eq!(id_in("12.png").unwrap(), 12);
        assert_eq!(id_in("12").unwrap(), 12);
        assert!(id_in("../etc/passwd").is_err());
    }
}

#[cfg(test)]
mod against_a_real_model {
    use super::*;
    use crate::compare::tests::{roomy, tiny_model};
    use kvad::quant::Precision;
    use kvad::service::Backend;

    /// A language model is not asked for a picture, whether it is named or
    /// is simply the only thing in memory, and the refusal says where to go.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_language_model_is_refused_an_image_request_by_name_and_by_default() {
        let dir = tiny_model("images");
        let db = Db::in_memory().unwrap();
        let state = State {
            db: db.clone(),
            auth: std::sync::Arc::new(crate::auth::Local),
            engine: std::sync::Arc::new(crate::scheduler::Scheduler::spawn(kvad::service::cpu_loader, roomy())),
            jobs: std::sync::Arc::new(crate::jobs::Jobs::new(db.clone())),
            metrics: std::sync::Arc::new(crate::metrics::Metrics::new()),
            setup: std::sync::Arc::new(Default::default()),
            oidc: std::sync::Arc::new(Default::default()),
            started: std::time::Instant::now(),
            load_on_request: false,
        };
        let (progress, _ignored) = tokio::sync::mpsc::channel(8);
        let repo = dir.to_string_lossy().into_owned();
        let loaded = state.engine.load(repo.clone(), Backend::Cpu(Precision::F32), progress).await.unwrap();
        assert_eq!(loaded.kind, Kind::Chat);
        assert!(loaded.image.is_none());

        let ask = |model: Option<&str>| {
            let state = state.clone();
            let body: Generations = serde_json::from_value(json!({ "model": model, "prompt": "a cat" })).unwrap();
            async move {
                let who = Identity { id: None, name: "local".into(), role: crate::auth::Role::Admin };
                match generations(who, HeaderMap::new(), St(state), Json(body)).await {
                    Ok(_) => panic!("a language model answered an image request"),
                    Err(Fail(status, why)) => (status, why),
                }
            }
        };

        let (status, why) = ask(Some(&repo)).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(why.contains("is a language model") && why.contains("/v1/chat/completions"), "{why}");

        // Unnamed, the one model in memory is not a candidate: it is the
        // wrong kind, and saying "no image model" is the useful answer.
        let (_, why) = ask(None).await;
        assert!(why.contains("no image model is loaded"), "{why}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
