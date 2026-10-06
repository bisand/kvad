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
//! # Edits
//!
//! `/v1/images/edits` is OpenAI's too: a picture, a prompt, and a mask if
//! only part of the picture is to change. What it runs is image-to-image on
//! the model that draws (`kvad_gpu::image::edit`): the picture is noised
//! part of the way, by `strength`, and drawn over from there, so the prompt
//! says what the picture should be and not what to change in it. A model
//! trained to follow an instruction, as OpenAI's own and FLUX Kontext are,
//! is a different thing this endpoint will also serve when one is here.
//!
//! The picture arrives as a file in a form, which is what OpenAI's SDK
//! sends, or as a `data:` URL in JSON. It is kept beside the image it made,
//! `<id>.input`, and the mask as `<id>.mask`, as they were sent: an edit is
//! looked at beside what it was made from, and made again from it.
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
use crate::videos::Fields;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State as St};
use axum::http::{header, HeaderMap};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use kvad::image::{Edit, ImageRequest, Painted};
use rusqlite::{params, Row};
use serde_json::json;
use std::path::{Path as FsPath, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/v1/images/generations", post(generations))
        .route("/v1/images/edits", post(edits))
        .route("/api/images", get(gallery))
        .route("/api/images/{id}", get(file).delete(remove))
        .route("/api/images/{id}/input", get(input))
        .route("/api/images/{id}/mask", get(mask))
        // A picture and its mask, each sent as base64 at its largest;
        // axum's default of 2 MB would refuse a photograph.
        .layer(DefaultBodyLimit::max(2 * (MAX_PICTURE / 3 * 4) + (1 << 16)))
}

/// The largest picture an edit may start from, and the largest mask: 20 MiB,
/// OpenAI's limit on a picture sent to it and the one a video's is held to.
const MAX_PICTURE: usize = 20 << 20;

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
    /// The LoRAs that made it, each at its strength; none for most.
    pub loras: Vec<kvad::image::Lora>,
    /// For an edit: how far its picture was noised, where that picture is,
    /// and where its mask is if it had one. All `None` for an image made
    /// from a prompt alone.
    pub strength: Option<f64>,
    pub input_url: Option<String>,
    pub mask_url: Option<String>,
}

const COLUMNS: &str = "id, model, backend, prompt, negative_prompt, width, height, steps, guidance, \
                       seed, bytes, secs, created_at, loras, strength, masked";

/// What an edit was made from, as it was sent: the picture's file, and the
/// mask's if there was one.
#[derive(Debug, Clone, PartialEq)]
pub struct Sources {
    pub image: Vec<u8>,
    pub mask: Option<Vec<u8>>,
}

fn stored_from(r: &Row<'_>) -> rusqlite::Result<Stored> {
    let id: i64 = r.get(0)?;
    let strength: Option<f64> = r.get(14)?;
    let masked = r.get::<_, i64>(15)? != 0;
    Ok(Stored {
        id,
        input_url: strength.map(|_| format!("/api/images/{id}/input")),
        mask_url: strength.filter(|_| masked).map(|_| format!("/api/images/{id}/mask")),
        strength,
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
        // Written by `save` only, so JSON it can read; anything else would
        // be a row edited by hand, shown as made without.
        loras: r.get::<_, Option<String>>(13)?.and_then(|j| serde_json::from_str(&j).ok()).unwrap_or_default(),
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
/// written. An edit's picture and mask are kept beside it.
pub fn save(db: &Db, dir: &FsPath, owner: Option<i64>, model: &str, backend: &str, p: &Painted, sources: Option<&Sources>) -> Res<Stored> {
    let png = p.image.png();
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let r = &p.request;
    let secs = p.encode_secs + p.denoise_secs + p.decode_secs;
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO images (owner, model, backend, prompt, negative_prompt, width, height, steps, \
             guidance, seed, bytes, secs, loras, strength, masked) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
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
                secs,
                (!r.loras.is_empty()).then(|| serde_json::to_string(&r.loras).unwrap_or_default()),
                r.strength.map(|s| s as f64),
                r.masked as i64
            ],
        )?;
        Ok(c.last_insert_rowid())
    })?;

    // Written aside and renamed into place, so that nobody is ever served
    // half a picture.
    let path = dir.join(format!("{id}.png"));
    let aside = dir.join(format!("{id}.png.part"));
    let written = std::fs::write(&aside, &png).and_then(|()| std::fs::rename(&aside, &path)).and_then(|()| {
        let Some(sources) = sources else { return Ok(()) };
        std::fs::write(dir.join(format!("{id}.input")), &sources.image)?;
        match &sources.mask {
            Some(mask) => std::fs::write(dir.join(format!("{id}.mask")), mask),
            None => Ok(()),
        }
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&aside);
        remove_files(dir, id);
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
    let _ = std::fs::remove_file(dir.join(format!("{id}.input")));
    let _ = std::fs::remove_file(dir.join(format!("{id}.mask")));
    match std::fs::remove_file(dir.join(format!("{id}.png"))) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(true),
    }
}

/// Every file an image has, gone: for a save that failed part of the way.
fn remove_files(dir: &FsPath, id: i64) {
    for ending in ["png", "input", "mask"] {
        let _ = std::fs::remove_file(dir.join(format!("{id}.{ending}")));
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

/// What an edit was made from: its picture, or its mask, as it was sent.
async fn source(who: Identity, state: State, name: String, ending: &'static str) -> Result<Response, Fail> {
    let id = id_in(&name)?;
    let db = state.db.clone();
    let bytes = blocking(move || match get_one(&db, id, who.id)? {
        Some(_) => Ok(std::fs::read(dir().join(format!("{id}.{ending}"))).ok()),
        None => Ok(None),
    })
    .await?
    .ok_or_else(|| Fail::missing(format!("image {id} has no {ending}")))?;
    // Written once, with the image, and never again.
    Ok(([(header::CONTENT_TYPE, crate::videos::sniff(&bytes)), (header::CACHE_CONTROL, "private, max-age=31536000, immutable")], bytes).into_response())
}

async fn input(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Response, Fail> {
    source(who, state, name, "input").await
}

async fn mask(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Response, Fail> {
    source(who, state, name, "mask").await
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
    /// kvad's own: LoRAs to apply for these images, `[{name, scale}]`,
    /// each by the name it was pulled by.
    #[serde(default)]
    loras: Vec<kvad::image::Lora>,
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    B64,
    Url,
}

/// `1024x1024` as a width and a height; `auto`, or nothing, as neither.
fn size_of(size: Option<&str>) -> Result<(Option<usize>, Option<usize>), Fail> {
    match size.map(str::trim) {
        None | Some("") | Some("auto") => Ok((None, None)),
        Some(s) => {
            let (w, h) = s
                .split_once(['x', 'X', '×'])
                .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
                .ok_or_else(|| Fail::bad(format!("size is WIDTHxHEIGHT, like 1024x1024, not {s:?}")))?;
            Ok((Some(w), Some(h)))
        }
    }
}

fn format_of(response_format: Option<&str>) -> Result<Format, Fail> {
    match response_format {
        None | Some("b64_json") => Ok(Format::B64),
        Some("url") => Ok(Format::Url),
        Some(other) => Err(Fail::bad(format!("response_format is b64_json or url, not {other:?}"))),
    }
}

impl Generations {
    fn request(&self) -> Result<ImageRequest, Fail> {
        let (width, height) = size_of(self.size.as_deref())?;
        Ok(ImageRequest {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            width,
            height,
            steps: self.steps,
            guidance: self.guidance_scale,
            seed: self.seed,
            preview: self.preview.unwrap_or(false) || self.partial_images.unwrap_or(0) > 0,
            loras: self.loras.clone(),
            edit: None,
        })
    }

    fn format(&self) -> Result<Format, Fail> {
        format_of(self.response_format.as_deref())
    }
}

pub async fn generations(
    who: Identity,
    headers: HeaderMap,
    St(state): St<State>,
    Json(body): Json<Generations>,
) -> Result<Response, Fail> {
    let request = body.request()?;
    let asked = Asked { model: body.model.clone(), request, n: body.n.unwrap_or(1), format: body.format()?, stream: body.stream, sources: None };
    make(who, headers, state, asked).await
}

/// A request for images once it has been read, whichever route read it.
struct Asked {
    model: Option<String>,
    request: ImageRequest,
    n: usize,
    format: Format,
    stream: bool,
    /// An edit's picture and mask as they were sent, to keep.
    sources: Option<Sources>,
}

/// Check a request against the model it names and make its images.
async fn make(who: Identity, headers: HeaderMap, state: State, asked: Asked) -> Result<Response, Fail> {
    let Asked { model, mut request, n, format, stream, sources } = asked;
    if n == 0 || n > MAX_N {
        return Err(Fail::bad(format!("n is between 1 and {MAX_N}, not {n}")));
    }

    let resident = crate::openai::resident_for(&state, model.as_deref(), Kind::Image).await?;
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
    loras_fit(&state, &request.loras)?;

    let job = Job { state, owner: who.id, resident, request, n, format, base: base_url(&headers), sources: sources.map(std::sync::Arc::new) };
    Ok(match stream {
        true => streamed(job).into_response(),
        false => whole(job).await?.into_response(),
    })
}

// ---------------------------------------------------------------------------
// /v1/images/edits
// ---------------------------------------------------------------------------

/// A picture a request sent under `name`: a file in a form, or in JSON a
/// `data:` URL, bare or as OpenAI's `{"image_url": …}`. `None` where it
/// sent none.
fn sent(f: &Fields, name: &str) -> Result<Option<Vec<u8>>, Fail> {
    // OpenAI's SDK names a form's file `image`, or `image[]` for several.
    let array = format!("{name}[]");
    let mut files = f.files.iter().filter(|(n, _)| *n == name || *n == array);
    let bytes = match (files.next(), f.get(&[name])) {
        (Some((_, bytes)), _) => {
            if files.next().is_some() {
                return Err(Fail::bad(format!("{name}: one picture is edited at a time here, and several were sent")));
            }
            bytes.clone()
        }
        (None, None) => return Ok(None),
        (None, Some((_, v))) => {
            let url = match v {
                serde_json::Value::String(u) => u.clone(),
                serde_json::Value::Object(o) if o.contains_key("file_id") => {
                    return Err(Fail::bad(format!("{name}.file_id: there is no Files API here; send the picture itself, as a file or a data: URL")))
                }
                serde_json::Value::Object(o) => o
                    .get("image_url")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| Fail::bad(format!("{name} wants an image_url")))?,
                _ => return Err(Fail::bad(format!("{name} is a file, or a data: URL"))),
            };
            crate::videos::data_url(url.trim(), name)?
        }
    };
    if bytes.is_empty() {
        return Err(Fail::bad(format!("{name} is empty")));
    }
    if bytes.len() > MAX_PICTURE {
        return Err(Fail::bad(format!("{name} is {} MB, and the most a picture may be is {} MB", bytes.len() >> 20, MAX_PICTURE >> 20)));
    }
    Ok(Some(bytes))
}

/// What an edit asked for, but for its picture and mask, which are files
/// still to be read.
fn edit_request(f: &Fields) -> Result<(ImageRequest, Option<f32>), Fail> {
    let prompt = f.text(&["prompt"]).ok_or_else(|| Fail::bad("an edit needs a prompt: what the picture should be"))?;
    let (width, height) = size_of(f.text(&["size"]).as_deref())?;
    let request = ImageRequest {
        prompt,
        negative_prompt: f.text(&["negative_prompt"]),
        width,
        height,
        steps: f.number(&["steps", "num_inference_steps"])?,
        guidance: f.number(&["guidance_scale"])?,
        seed: f.number(&["seed"])?,
        preview: f.flag("preview")?.unwrap_or(false) || f.number::<usize>(&["partial_images"])?.unwrap_or(0) > 0,
        loras: f.loras()?,
        edit: None,
    };
    Ok((request, f.number(&["strength"])?))
}

/// Read a picture and its mask with `ffmpeg`, which reads files: each is
/// written aside, read, and removed.
fn read_sources(ffmpeg: &FsPath, sources: &Sources, strength: Option<f32>) -> Res<Edit> {
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let aside = |what: &str| std::env::temp_dir().join(format!("kvad-edit-{}-{n}.{what}", std::process::id()));
    let read = |what: &str, bytes: &[u8]| -> Res<PathBuf> {
        let path = aside(what);
        std::fs::write(&path, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))?;
        Ok(path)
    };
    let picture = read("image", &sources.image)?;
    let image = kvad::video::picture_from_file(ffmpeg, &picture, 0);
    let _ = std::fs::remove_file(&picture);
    let image = image.map_err(|e| format!("image is not a picture ffmpeg can read: {e}"))?;
    let mask = match &sources.mask {
        None => None,
        Some(bytes) => {
            let path = read("mask", bytes)?;
            let mask = kvad::video::mask_from_file(ffmpeg, &path);
            let _ = std::fs::remove_file(&path);
            Some(mask.map_err(|e| format!("mask is not a picture ffmpeg can read: {e}"))?)
        }
    };
    Ok(Edit { image, mask, strength })
}

pub async fn edits(who: Identity, headers: HeaderMap, St(state): St<State>, body: Bytes) -> Result<Response, Fail> {
    let f = Fields::read(&headers, &body)?;
    let (mut request, strength) = edit_request(&f)?;
    let image = sent(&f, "image")?.ok_or_else(|| Fail::bad("an edit needs an image: the picture to start from, as a file in a form or a data: URL"))?;
    let sources = Sources { image, mask: sent(&f, "mask")? };

    // Read now, so that a file that is not a picture is a 400 and not a
    // failure after a wait in the queue.
    let ffmpeg = crate::videos::ffmpeg().ok_or_else(|| Fail::bad("editing a picture needs ffmpeg on the server, to read it; `[videos] ffmpeg` in kvad.toml"))?;
    let read = sources.clone();
    let edit = blocking(move || read_sources(&ffmpeg, &read, strength)).await.map_err(|e| Fail::bad(e.1))?;
    request.edit = Some(edit);

    let asked = Asked {
        model: f.text(&["model"]),
        request,
        n: f.number(&["n"])?.unwrap_or(1),
        format: format_of(f.text(&["response_format"]).as_deref())?,
        stream: f.flag("stream")?.unwrap_or(false),
        sources: Some(sources),
    };
    make(who, headers, state, asked).await
}

/// Each LoRA on this machine, and all of them together in what no resident
/// has been charged.
///
/// A LoRA is set for its request and taken off after it, so it is charged
/// nothing standing; but while it is set its factors are on the device, and
/// they have to fit beside everything that is. Loads are cache-first, so
/// one that is not here is a 400 that says how to fetch it, not a download
/// in the middle of a request.
pub(crate) fn loras_fit(state: &State, loras: &[kvad::image::Lora]) -> Result<(), Fail> {
    let mut bytes = 0;
    for l in loras {
        let here = kvad::lora::local(&l.name).ok_or_else(|| Fail::bad(format!("the LoRA {} is not on this machine; `kvad pull {}` fetches it", l.name, l.name)))?;
        bytes += kvad::lora::device_bytes(&here.file).unwrap_or(0);
    }
    let left = state.engine.left();
    if bytes > left {
        return Err(Fail::bad(format!(
            "the LoRAs take {:.2} GB while they are set, and {:.2} GB is left beside the models in memory",
            bytes as f64 / 1e9,
            left as f64 / 1e9
        )));
    }
    Ok(())
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
    /// An edit's picture and mask as they were sent, kept with each image.
    sources: Option<std::sync::Arc<Sources>>,
}

impl Job {
    /// The request for the `i`th image: the same, one seed along.
    fn nth(&self, i: usize) -> ImageRequest {
        ImageRequest { seed: self.request.seed.map(|s| s.wrapping_add(i as u64)), ..self.request.clone() }
    }

    async fn keep(&self, painted: &Painted) -> Result<Stored, Fail> {
        let (db, owner, model) = (self.state.db.clone(), self.owner, self.resident.model.repo.clone());
        let backend = self.resident.model.backend.clone();
        let (painted, sources) = (painted.clone(), self.sources.clone());
        blocking(move || save(&db, &dir(), owner, &model, &backend, &painted, sources.as_deref())).await
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
            "loras": stored.loras,
            "strength": stored.strength,
            "input_url": stored.input_url,
            "mask_url": stored.mask_url,
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
                Some(Stroke::Refused(e)) => return Err(Fail::bad(e)),
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
                    Stroke::Failed(e) | Stroke::Refused(e) => {
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
                loras: Vec::new(),
                strength: None,
                masked: false,
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
        let s = save(&db, &dir, None, "stabilityai/sdxl", "metal f16", &painted(seed), None).unwrap();
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

    /// The LoRAs that made an image are kept with it, and an image made
    /// without any is kept as made without: `NULL`, as every image from
    /// before there were LoRAs is.
    #[test]
    fn an_image_keeps_the_loras_that_made_it() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("loras");
        let mut p = painted(1);
        let loras = vec![kvad::image::Lora { name: "lightx2v/Qwen-Image-Lightning:8steps.safetensors".into(), scale: 0.8 }];
        p.request.loras = loras.clone();
        assert_eq!(save(&db, &dir, None, "Qwen/Qwen-Image", "metal q8", &p, None).unwrap().loras, loras);
        let plain = save(&db, &dir, None, "Qwen/Qwen-Image", "metal q8", &painted(2), None).unwrap();
        assert!(plain.loras.is_empty());
        let stored: Option<String> = db.with(|c| c.query_row("SELECT loras FROM images WHERE id = ?1", [plain.id], |r| r.get(0))).unwrap();
        assert_eq!(stored, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An edit is kept with what it was made from, as it was sent, and the
    /// row says where; an image that is no edit has neither.
    #[test]
    fn an_edit_keeps_its_picture_and_its_mask() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("edits");
        let mut p = painted(1);
        p.request.strength = Some(0.5);
        p.request.masked = true;
        let sources = Sources { image: b"\x89PNG the picture".to_vec(), mask: Some(b"\x89PNG the mask".to_vec()) };
        let s = save(&db, &dir, None, "m", "b", &p, Some(&sources)).unwrap();
        assert_eq!(s.strength, Some(0.5));
        assert_eq!((s.input_url.as_deref(), s.mask_url.as_deref()), (Some(format!("/api/images/{}/input", s.id).as_str()), Some(format!("/api/images/{}/mask", s.id).as_str())));
        assert_eq!(std::fs::read(dir.join(format!("{}.input", s.id))).unwrap(), sources.image);
        assert_eq!(std::fs::read(dir.join(format!("{}.mask", s.id))).unwrap(), sources.mask.clone().unwrap());

        // Without a mask there is no link to one.
        p.request.masked = false;
        let plain = save(&db, &dir, None, "m", "b", &p, Some(&Sources { mask: None, ..sources })).unwrap();
        assert!(plain.input_url.is_some() && plain.mask_url.is_none());
        assert!(!dir.join(format!("{}.mask", plain.id)).exists());

        let drawn = save(&db, &dir, None, "m", "b", &painted(2), None).unwrap();
        assert_eq!((drawn.strength, drawn.input_url, drawn.mask_url), (None, None, None));

        assert!(delete(&db, &dir, s.id, None).unwrap());
        assert!(!dir.join(format!("{}.input", s.id)).exists() && !dir.join(format!("{}.mask", s.id)).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn fields(v: serde_json::Value) -> Fields {
        match v {
            serde_json::Value::Object(map) => Fields { map, files: Vec::new() },
            _ => unreachable!(),
        }
    }

    /// An edit's fields under OpenAI's names and diffusers', from JSON or
    /// from a form, where every value is text.
    #[test]
    fn an_edit_is_read_from_json_or_a_form() {
        let (r, strength) = edit_request(&fields(json!({ "prompt": "a green apple", "size": "768x512", "steps": 20, "guidance_scale": 6.5, "seed": 7, "strength": 0.4, "negative_prompt": "blurry", "partial_images": 1 }))).unwrap();
        assert_eq!((r.prompt.as_str(), r.width, r.height, r.steps, r.guidance, r.seed, r.preview), ("a green apple", Some(768), Some(512), Some(20), Some(6.5), Some(7), true));
        assert_eq!((r.negative_prompt.as_deref(), strength), (Some("blurry"), Some(0.4)));

        let form = fields(json!({ "prompt": "a green apple", "num_inference_steps": "12", "strength": "0.6", "loras": "[{\"name\":\"a/b\",\"scale\":0.5}]" }));
        let (r, strength) = edit_request(&form).unwrap();
        assert_eq!((r.steps, strength, r.width, r.loras.len()), (Some(12), Some(0.6), None, 1));

        assert!(edit_request(&fields(json!({ "size": "512x512" }))).unwrap_err().1.contains("prompt"));
        assert!(edit_request(&fields(json!({ "prompt": "x", "strength": "a lot" }))).unwrap_err().1.contains("strength"));
    }

    /// The picture, however it was sent; and what cannot be had is refused
    /// by name.
    #[test]
    fn an_edit_s_picture_is_a_file_or_a_data_url() {
        let mut form = fields(json!({ "prompt": "x" }));
        form.files.push(("image".into(), b"picture".to_vec()));
        form.files.push(("mask".into(), b"mask".to_vec()));
        assert_eq!(sent(&form, "image").unwrap().as_deref(), Some(&b"picture"[..]));
        assert_eq!(sent(&form, "mask").unwrap().as_deref(), Some(&b"mask"[..]));
        assert_eq!(sent(&fields(json!({})), "mask").unwrap(), None);

        // OpenAI's SDK sends a list of pictures as `image[]`.
        let mut list = fields(json!({}));
        list.files.push(("image[]".into(), b"one".to_vec()));
        assert_eq!(sent(&list, "image").unwrap().as_deref(), Some(&b"one"[..]));
        list.files.push(("image[]".into(), b"two".to_vec()));
        assert!(sent(&list, "image").unwrap_err().1.contains("one picture"));

        let url = json!({ "image": "data:image/png;base64,cGljdHVyZQ==" });
        assert_eq!(sent(&fields(url), "image").unwrap().as_deref(), Some(&b"picture"[..]));
        for (body, wrong) in [
            (json!({ "image": "https://example.com/a.png" }), "fetches nothing"),
            (json!({ "image": { "file_id": "file-1" } }), "no Files API"),
            (json!({ "image": 3 }), "a data: URL"),
        ] {
            let said = sent(&fields(body), "image").unwrap_err().1;
            assert!(said.contains(wrong), "{said}");
        }
    }

    /// Deleting the newest image and making another must not give the new
    /// one the old id: the link is cached as immutable, and a browser that
    /// saw the old picture there would go on showing it.
    #[test]
    fn a_deleted_images_id_is_not_handed_out_again() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("reused");
        let first = save(&db, &dir, None, "m", "b", &painted(1), None).unwrap();
        assert!(delete(&db, &dir, first.id, None).unwrap());
        let second = save(&db, &dir, None, "m", "b", &painted(2), None).unwrap();
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
        let mine = save(&db, &dir, Some(1), "m", "b", &painted(1), None).unwrap();
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
            settings: Default::default(),
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
