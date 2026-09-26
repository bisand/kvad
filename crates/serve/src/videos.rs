//! `/v1/videos`, and where the videos it makes are kept.
//!
//! OpenAI's shape, as `/v1/images/generations` is, so that a client written
//! for OpenAI's video API makes videos here by changing a URL. That shape is
//! a job, not an answer: `POST /v1/videos` returns at once with a video whose
//! `status` is `queued`; `GET /v1/videos/{id}` reports its `progress` until
//! it is `completed` or `failed`; and `GET /v1/videos/{id}/content` is the
//! MP4. A generation takes minutes, which is longer than a request should be
//! held open and longer than a browser tab can be trusted to stay, so the
//! work belongs to the server and a client only watches it.
//!
//! OpenAI's SDKs send the request as `multipart/form-data` whether or not a
//! file is in it, so that is read here as well as JSON; see [`form`].
//!
//! What OpenAI has no field for is taken beside its fields: `frames` (or
//! `num_frames`, diffusers' name) where OpenAI has only `seconds`, `fps`,
//! `seed`, and `audio: false` for a silent file. What this engine measured
//! comes back in a `kvad` object. Every endpoint in OpenAI's reference is
//! marked deprecated as this is written; its shape is still the only one a
//! client of theirs knows.
//!
//! # Where videos live
//!
//! `videos/<id>.mp4` in the data directory, beside `images/`, for the reason
//! images are there: it is somebody's work and the only copy. A poster frame
//! goes beside it as `<id>.png`, for a gallery to show before anything
//! plays.
//!
//! # Compressed, when there is an `ffmpeg`
//!
//! The MP4 `kvad::video` writes compresses nothing (it says why): 5 s at
//! 768×512 is 71 MB, and at 1536×1024 288 MB. Where an `ffmpeg` can be
//! found, each is re-encoded as LTX's reference writes its own, H.264 at
//! CRF 19 and AAC. Measured on an M5 Pro, that takes under a second, is a
//! thirtieth of the size (58 MB to 1.9, 288 MB to 9.1), and is 43.6–44.2 dB
//! PSNR from the uncompressed file. The uncompressed file is not kept: at
//! that price, nobody would keep it. If `ffmpeg` fails, it is kept instead,
//! and the failure is logged. `[videos] ffmpeg` in `kvad.toml` names one, or
//! turns this off; see [`find_ffmpeg`].
//!
//! # Seeking
//!
//! The file is served with HTTP Range requests answered. Without them a
//! browser cannot seek in a video it has not fully downloaded: with the
//! files `kvad::video` writes, every seek in Chromium went back to 0 until
//! the test server answered Range.
//!
//! # A generation's lifetime
//!
//! [`run`] is a task of its own, spawned by the request that asked for the
//! video and not tied to it. It writes each step's progress into the row, so
//! that `GET` reads nothing but the database. Deleting a video that is still
//! running stops it: the task notices the row is gone and drops its end of
//! the channel, and the engine stops at the next step. A server that stops
//! mid-generation marks what it was making as failed when it starts again
//! ([`abandon`]).
//!
//! # Watching instead of asking
//!
//! OpenAI's shape is polled, and that stays. Beside it,
//! `GET /v1/videos/{id}/events` is kvad's own: server-sent events carrying
//! the same video object, sent once as it stands and again at every change
//! — each step, the finish, a deletion — until it ends. A step can be a
//! minute apart from the next, so polling every two seconds asked thirty
//! times for each answer that changed; and a video that finishes is seen at
//! once rather than up to two seconds later. See [`Herald`].

use crate::api::{blocking, Fail};
use crate::auth::{Identity, State};
use crate::db::Db;
use crate::scheduler::{Key, Kind, Reel};
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State as St};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kvad::video::{Filmed, VideoRequest};
use rusqlite::{params, Row};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path as FsPath, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/v1/videos", post(create).get(list_route))
        .route("/v1/videos/{id}", get(retrieve).delete(remove))
        .route("/v1/videos/{id}/content", get(content))
        .route("/v1/videos/{id}/events", get(events))
}

/// Where the MP4s are: `videos` in the data directory.
pub fn dir() -> PathBuf {
    kvad::weights::data_dir().join("videos")
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// One video, as its row has it.
#[derive(Debug, Clone)]
pub struct Stored {
    pub id: i64,
    pub model: String,
    pub backend: String,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub seed: u64,
    pub audio: bool,
    pub status: String,
    pub progress: f64,
    pub phase: Option<String>,
    pub error: Option<String>,
    pub bytes: Option<u64>,
    pub encode_secs: Option<f64>,
    pub denoise_secs: Option<f64>,
    pub decode_secs: Option<f64>,
    pub created_at: String,
    /// When it was asked for, when the engine reached it, and when it was
    /// done, in Unix seconds, which is how OpenAI's object gives them.
    pub created: i64,
    pub started: Option<i64>,
    pub completed: Option<i64>,
}

const COLUMNS: &str = "id, model, backend, prompt, width, height, frames, fps, seed, audio, status, progress, \
                       phase, error, bytes, encode_secs, denoise_secs, decode_secs, created_at, \
                       CAST(strftime('%s', created_at) AS INTEGER), \
                       CAST(strftime('%s', started_at) AS INTEGER), CAST(strftime('%s', completed_at) AS INTEGER)";

fn stored_from(r: &Row<'_>) -> rusqlite::Result<Stored> {
    Ok(Stored {
        id: r.get(0)?,
        model: r.get(1)?,
        backend: r.get(2)?,
        prompt: r.get(3)?,
        width: r.get(4)?,
        height: r.get(5)?,
        frames: r.get(6)?,
        fps: r.get(7)?,
        seed: r.get::<_, i64>(8)? as u64,
        audio: r.get::<_, i64>(9)? != 0,
        status: r.get(10)?,
        progress: r.get(11)?,
        phase: r.get(12)?,
        error: r.get(13)?,
        bytes: r.get::<_, Option<i64>>(14)?.map(|b| b as u64),
        encode_secs: r.get(15)?,
        denoise_secs: r.get(16)?,
        decode_secs: r.get(17)?,
        created_at: r.get(18)?,
        created: r.get(19)?,
        started: r.get(20)?,
        completed: r.get(21)?,
    })
}

impl Stored {
    /// OpenAI's id for it.
    pub fn name(&self) -> String {
        format!("video_{}", self.id)
    }

    /// A link to the file, or to its poster frame, that names this video
    /// forever: the time it was made, as digits, tells apart two videos
    /// given the same id by two data directories, as it does for images.
    fn link(&self, variant: &str) -> String {
        let made: String = self.created_at.chars().filter(char::is_ascii_digit).collect();
        let variant = match variant {
            "video" => String::new(),
            v => format!("variant={v}&"),
        };
        format!("/v1/videos/{}/content?{variant}v={made}", self.name())
    }

    /// OpenAI's video object, and kvad's own fields in `kvad`.
    pub fn resource(&self) -> Value {
        let done = self.status == "completed";
        json!({
            "id": self.name(),
            "object": "video",
            "model": self.model,
            "status": self.status,
            // A whole percentage, as OpenAI gives it; kvad's own is finer.
            "progress": (self.progress * 100.0).floor() as i64,
            "created_at": self.created,
            "completed_at": self.completed,
            "expires_at": null,
            "prompt": self.prompt,
            "size": format!("{}x{}", self.width, self.height),
            "seconds": seconds(self.frames, self.fps),
            "remixed_from_video_id": null,
            "error": self.error.as_ref().map(|e| json!({
                "code": if e == CANCELLED { "cancelled" } else { "generation_failed" },
                "message": e,
            })),
            "kvad": {
                "id": self.id,
                "backend": self.backend,
                "width": self.width,
                "height": self.height,
                "frames": self.frames,
                "fps": self.fps,
                "seed": self.seed,
                "audio": self.audio,
                "progress": self.progress,
                "phase": self.phase,
                "started_at": self.started,
                "bytes": self.bytes,
                "encode_secs": self.encode_secs,
                "denoise_secs": self.denoise_secs,
                "decode_secs": self.decode_secs,
                "url": done.then(|| self.link("video")),
                "thumbnail_url": done.then(|| self.link("thumbnail")),
                // Named by how far along it is, so that each step's is a new
                // link and a page showing it asks again. A 404 until the
                // first denoising step is done.
                "preview_url": (self.status == "in_progress").then(|| format!(
                    "/v1/videos/{}/content?variant=preview&at={}",
                    self.name(),
                    (self.progress * 1000.0).round() as i64
                )),
            },
        })
    }
}

/// A clip's length as OpenAI writes it: a string of seconds, with no more
/// decimals than it needs.
fn seconds(frames: u32, fps: u32) -> String {
    let s = format!("{:.2}", frames as f64 / fps.max(1) as f64);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// What a video that was stopped says, and how [`Stored::resource`] tells it
/// from one that failed.
const CANCELLED: &str = "cancelled";

/// Write the row for a video about to be made.
pub fn queue(db: &Db, owner: Option<i64>, model: &str, backend: &str, r: &kvad::video::Resolved) -> Res<Stored> {
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO videos (owner, model, backend, prompt, width, height, frames, fps, seed, audio) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                owner,
                model,
                backend,
                r.prompt,
                r.width as i64,
                r.height as i64,
                r.frames as i64,
                r.fps as i64,
                r.seed as i64,
                r.audio as i64
            ],
        )?;
        Ok(c.last_insert_rowid())
    })?;
    get_one(db, id, owner)?.ok_or_else(|| "the video was queued and then was not there".into())
}

/// Record where a generation has got to. `false` when the row is gone:
/// somebody deleted the video, and the generation should stop.
pub fn advance(db: &Db, id: i64, step: &kvad::video::Step) -> Res<bool> {
    let n = db.with(|c| {
        c.execute(
            "UPDATE videos SET status = 'in_progress', progress = ?2, phase = ?3, \
             started_at = coalesce(started_at, datetime('now')) WHERE id = ?1",
            params![id, step.progress as f64, step.phase],
        )
    })?;
    Ok(n > 0)
}

/// Keep a finished video: the files, then the row that says they are there.
///
/// The files are written aside and renamed into place, so that nobody is
/// served half a video; `ffmpeg`, when there is one, compresses the MP4 on
/// its way into place. If the row went while the files were being written,
/// the video was deleted, and the files go too.
pub fn finish(db: &Db, dir: &FsPath, id: i64, f: &Filmed, ffmpeg: Option<&FsPath>) -> Res<bool> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let mp4 = f.video.mp4(f.audio.as_ref());
    // The middle frame, which says more about a clip than its first.
    let poster = f.video.frame(f.video.frames() / 2).png();
    for (ext, bytes) in [("mp4", &mp4), ("png", &poster)] {
        let path = dir.join(format!("{id}.{ext}"));
        let aside = dir.join(format!("{id}.{ext}.part"));
        let written = std::fs::write(&aside, bytes).and_then(|()| match (ext, ffmpeg) {
            ("mp4", Some(ffmpeg)) => compress(ffmpeg, &aside, &path, id),
            _ => std::fs::rename(&aside, &path),
        });
        if let Err(e) = written {
            let _ = std::fs::remove_file(&aside);
            remove_files(dir, id);
            return Err(format!("could not write {}: {e}", path.display()).into());
        }
    }
    let bytes = std::fs::metadata(dir.join(format!("{id}.mp4")))?.len();
    let n = db.with(|c| {
        c.execute(
            "UPDATE videos SET status = 'completed', progress = 1, phase = NULL, bytes = ?2, encode_secs = ?3, \
             denoise_secs = ?4, decode_secs = ?5, completed_at = datetime('now') WHERE id = ?1",
            params![id, bytes as i64, f.encode_secs, f.denoise_secs, f.decode_secs],
        )
    })?;
    // Done with: the video itself is the preview now.
    let _ = std::fs::remove_file(dir.join(format!("{id}.preview.png")));
    if n == 0 {
        remove_files(dir, id);
    }
    Ok(n > 0)
}

/// Re-encode the uncompressed MP4 at `raw` into `to`, as LTX's reference
/// writes its own; or, if `ffmpeg` fails, move `raw` there as it is.
///
/// Written to a third name and renamed, as every file here is. The colours
/// are said again for the encoder, which would otherwise write none, and
/// `faststart` puts the index in front, as `kvad::video` does, so that a
/// browser can start before the whole file is there.
fn compress(ffmpeg: &FsPath, raw: &FsPath, to: &FsPath, id: i64) -> std::io::Result<()> {
    let out = raw.with_extension("x264");
    let ran = std::process::Command::new(ffmpeg)
        .args(["-nostdin", "-y", "-v", "error", "-i"])
        .arg(raw)
        .args(["-c:v", "libx264", "-crf", "19", "-preset", "medium", "-pix_fmt", "yuv420p"])
        .args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"])
        .args(["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-f", "mp4"])
        .arg(&out)
        .stdin(std::process::Stdio::null())
        .output();
    let why = match ran {
        Ok(o) if o.status.success() => {
            std::fs::remove_file(raw)?;
            return std::fs::rename(&out, to);
        }
        Ok(o) => format!("{}: {}", o.status, String::from_utf8_lossy(&o.stderr).trim()),
        Err(e) => e.to_string(),
    };
    let _ = std::fs::remove_file(&out);
    tracing::warn!("video {id} is kept uncompressed: {} failed: {why}", ffmpeg.display());
    std::fs::rename(raw, to)
}

/// The `ffmpeg` every video is compressed with, chosen once at startup.
static FFMPEG: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();

/// Choose the `ffmpeg` from `[videos] ffmpeg`, and say what was chosen, for
/// the line the server prints as it starts.
pub fn use_ffmpeg(setting: &str) -> String {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let found = find_ffmpeg(setting, &std::env::split_paths(&path).collect::<Vec<_>>());
    let said = match (&found, setting.trim()) {
        (Some(p), _) => format!("compressed with {}", p.display()),
        (None, "off") => "kept uncompressed ([videos] ffmpeg = \"off\")".into(),
        (None, "auto" | "") => "kept uncompressed: no ffmpeg found".into(),
        (None, named) => format!("kept uncompressed: {named} is not a file"),
    };
    let _ = FFMPEG.set(found);
    said
}

/// `ffmpeg` as `[videos] ffmpeg` says to find it: `off` for none, a path
/// for that file if it is there, and `auto` for the first on `path` or in
/// the places a package manager puts one. The last matters: a launchd
/// service runs with `/usr/bin:/bin:/usr/sbin:/sbin` for its `PATH`, and
/// Homebrew's `ffmpeg` is in none of them.
pub fn find_ffmpeg(setting: &str, path: &[PathBuf]) -> Option<PathBuf> {
    match setting.trim() {
        "off" => None,
        "auto" | "" => path
            .iter()
            .cloned()
            .chain(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"].map(PathBuf::from))
            .map(|d| d.join("ffmpeg"))
            .find(|f| f.is_file()),
        named => Some(PathBuf::from(named)).filter(|f| f.is_file()),
    }
}

/// Who is told when a video changes: one `broadcast` a video being made,
/// carrying nothing. A watcher that hears it reads the row again, so the
/// row stays the one account of a video, and a watcher that fell behind
/// loses nothing by skipping to the latest.
static HERALDS: std::sync::LazyLock<std::sync::Mutex<HashMap<i64, tokio::sync::broadcast::Sender<()>>>> =
    std::sync::LazyLock::new(Default::default);

fn heralds() -> std::sync::MutexGuard<'static, HashMap<i64, tokio::sync::broadcast::Sender<()>>> {
    HERALDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A video being made, as far as its watchers are concerned. Made before the
/// generation starts, so that nobody can subscribe too early to hear it;
/// dropped when the generation's task ends, which closes the channel and
/// tells every watcher to read the row one last time.
pub struct Herald(i64);

impl Herald {
    pub fn new(id: i64) -> Self {
        heralds().insert(id, tokio::sync::broadcast::channel(16).0);
        Herald(id)
    }
}

impl Drop for Herald {
    fn drop(&mut self) {
        heralds().remove(&self.0);
    }
}

/// Tell whoever is watching video `id` that its row has changed.
fn tell(id: i64) {
    if let Some(h) = heralds().get(&id) {
        let _ = h.send(());
    }
}

/// Listen for video `id`'s changes; `None` when nothing is making it.
fn listen(id: i64) -> Option<tokio::sync::broadcast::Receiver<()>> {
    heralds().get(&id).map(|h| h.subscribe())
}

/// Whether a video's row is still there.
fn exists(db: &Db, id: i64) -> Res<bool> {
    db.with(|c| c.query_row("SELECT count(*) FROM videos WHERE id = ?1", [id], |r| r.get::<_, i64>(0)))
        .map(|n| n > 0)
}

/// Keep a step's preview as `<id>.preview.png`, in place of the last one.
pub fn look(dir: &FsPath, id: i64, preview: &kvad::image::Image) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    let (path, aside) = (dir.join(format!("{id}.preview.png")), dir.join(format!("{id}.preview.png.part")));
    std::fs::write(&aside, preview.png())?;
    Ok(std::fs::rename(&aside, &path)?)
}

/// Record that a video will not be made, and why.
pub fn fail(db: &Db, id: i64, why: &str) -> Res<()> {
    db.with(|c| {
        c.execute(
            "UPDATE videos SET status = 'failed', phase = NULL, error = ?2, completed_at = datetime('now') \
             WHERE id = ?1",
            params![id, why],
        )
    })?;
    Ok(())
}

/// Mark every video that was queued or being made when the server last
/// stopped as failed: nothing is making it now.
pub fn abandon(db: &Db) -> Res<usize> {
    db.with(|c| {
        c.execute(
            "UPDATE videos SET status = 'failed', phase = NULL, \
             error = 'the server stopped before this video was finished', completed_at = datetime('now') \
             WHERE status IN ('queued', 'in_progress')",
            [],
        )
    })
}

pub fn list(db: &Db, owner: Option<i64>) -> Res<Vec<Stored>> {
    db.with(|c| {
        let mut q = c.prepare(&format!("SELECT {COLUMNS} FROM videos WHERE owner IS ?1 ORDER BY id DESC"))?;
        let rows = q.query_map([owner], stored_from)?.collect();
        rows
    })
}

pub fn get_one(db: &Db, id: i64, owner: Option<i64>) -> Res<Option<Stored>> {
    db.with(|c| {
        match c.query_row(
            &format!("SELECT {COLUMNS} FROM videos WHERE id = ?2 AND owner IS ?1"),
            params![owner, id],
            stored_from,
        ) {
            Ok(s) => Ok(Some(s)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    })
}

/// Forget a video, row and files. `false` when there was no such video of
/// `owner`'s. A video still being made stops at its next step; see [`run`].
pub fn delete(db: &Db, dir: &FsPath, id: i64, owner: Option<i64>) -> Res<bool> {
    let gone = db.with(|c| c.execute("DELETE FROM videos WHERE id = ?2 AND owner IS ?1", params![owner, id]))?;
    if gone > 0 {
        remove_files(dir, id);
    }
    Ok(gone > 0)
}

fn remove_files(dir: &FsPath, id: i64) {
    for ext in ["mp4", "png", "preview.png"] {
        let _ = std::fs::remove_file(dir.join(format!("{id}.{ext}")));
    }
}

// ---------------------------------------------------------------------------
// Reading a request
// ---------------------------------------------------------------------------

/// One part of a `multipart/form-data` body.
#[derive(Debug, PartialEq)]
struct Part {
    name: String,
    /// Present when the part is a file.
    filename: Option<String>,
    data: Vec<u8>,
}

/// The parts of a `multipart/form-data` body (RFC 7578).
///
/// By hand, because the form is small and fixed: a boundary line, headers, a
/// blank line, the bytes, and the next boundary, which is always preceded by
/// CRLF. A part's own headers other than its name are not needed here, and
/// neither is anything before the first boundary or after the last.
fn form(body: &[u8], content_type: &str) -> Result<Vec<Part>, Fail> {
    let boundary = content_type
        .split(';')
        .filter_map(|p| p.trim().strip_prefix("boundary="))
        .next()
        .map(|b| b.trim_matches('"'))
        .filter(|b| !b.is_empty())
        .ok_or_else(|| Fail::bad("a multipart body with no boundary"))?;
    let delimiter = format!("--{boundary}").into_bytes();
    let next = format!("\r\n--{boundary}").into_bytes();
    let broken = || Fail::bad("a multipart body that does not parse");

    let mut at = find(body, &delimiter, 0).ok_or_else(broken)? + delimiter.len();
    let mut parts = Vec::new();
    loop {
        match body.get(at..at + 2) {
            Some(b"--") => return Ok(parts),
            Some(b"\r\n") => at += 2,
            _ => return Err(broken()),
        }
        let head_end = find(body, b"\r\n\r\n", at).ok_or_else(broken)?;
        let head = std::str::from_utf8(&body[at..head_end]).map_err(|_| broken())?;
        let end = find(body, &next, head_end + 4).ok_or_else(broken)?;
        let disposition = head
            .split("\r\n")
            .find(|l| l.to_ascii_lowercase().starts_with("content-disposition:"))
            .ok_or_else(broken)?;
        let param = |key: &str| {
            disposition.split(';').map(str::trim).find_map(|p| p.strip_prefix(key)).map(|v| v.trim_matches('"').to_string())
        };
        parts.push(Part {
            name: param("name=").ok_or_else(broken)?,
            filename: param("filename="),
            data: body[head_end + 4..end].to_vec(),
        });
        at = end + next.len();
    }
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    hay.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

/// A request's fields, whichever way it was sent: JSON's own values, or a
/// form's as strings. Files are named in `files`, to be refused.
struct Fields {
    map: Map<String, Value>,
    files: Vec<String>,
}

impl Fields {
    fn read(headers: &HeaderMap, body: &[u8]) -> Result<Fields, Fail> {
        let kind = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
        if kind.to_ascii_lowercase().starts_with("multipart/form-data") {
            let mut map = Map::new();
            let mut files = Vec::new();
            for part in form(body, kind)? {
                match part.filename {
                    Some(_) => files.push(part.name),
                    None => {
                        let text = String::from_utf8(part.data)
                            .map_err(|_| Fail::bad(format!("the form field {} is not text", part.name)))?;
                        map.insert(part.name, Value::String(text));
                    }
                }
            }
            return Ok(Fields { map, files });
        }
        match serde_json::from_slice(body) {
            Ok(Value::Object(map)) => Ok(Fields { map, files: Vec::new() }),
            Ok(_) => Err(Fail::bad("the request body is not a JSON object")),
            Err(e) => Err(Fail::bad(format!("the request body is not JSON or a form: {e}"))),
        }
    }

    /// The first of `names` that is present and not empty.
    fn get<'a>(&self, names: &[&'a str]) -> Option<(&'a str, &Value)> {
        names.iter().find_map(|n| {
            let v = self.map.get(*n)?;
            let blank = v.is_null() || v.as_str().is_some_and(|s| s.trim().is_empty());
            (!blank).then_some((*n, v))
        })
    }

    fn text(&self, names: &[&str]) -> Option<String> {
        self.get(names).map(|(_, v)| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }

    /// A number, whether JSON sent it as one or a form as text.
    fn number<T: std::str::FromStr>(&self, names: &[&str]) -> Result<Option<T>, Fail> {
        let Some((name, v)) = self.get(names) else { return Ok(None) };
        let text = match v {
            Value::String(s) => s.trim().to_string(),
            other => other.to_string(),
        };
        text.parse().map(Some).map_err(|_| Fail::bad(format!("{name} must be a number, not {text}")))
    }

    fn flag(&self, name: &str) -> Result<Option<bool>, Fail> {
        match self.get(&[name]) {
            None => Ok(None),
            Some((_, Value::Bool(b))) => Ok(Some(*b)),
            Some((_, v)) => match v.as_str().map(|s| s.trim().to_ascii_lowercase()).as_deref() {
                Some("true" | "1" | "yes") => Ok(Some(true)),
                Some("false" | "0" | "no") => Ok(Some(false)),
                _ => Err(Fail::bad(format!("{name} is true or false, not {v}"))),
            },
        }
    }
}

/// What a create request asked for, before the model's defaults are known.
#[derive(Debug, PartialEq)]
struct Asked {
    model: Option<String>,
    request: VideoRequest,
    /// OpenAI's way of giving a length, turned into frames once the frame
    /// rate is known.
    seconds: Option<f64>,
}

impl Asked {
    fn read(f: &Fields) -> Result<Asked, Fail> {
        // A file in a form, or OpenAI's JSON reference to one.
        let reference = f.files.first().cloned().or_else(|| f.get(&["input_reference"]).map(|_| "input_reference".into()));
        if let Some(name) = reference {
            return Err(Fail::bad(format!(
                "{name}: making a video from a picture is not implemented here yet; leave it out"
            )));
        }
        // Refused rather than ignored, as images refuse a guidance scale a
        // model has no use for: a video made without what was asked for
        // should not be filed as if it had been.
        for knob in ["negative_prompt", "guidance_scale", "steps", "num_inference_steps"] {
            if f.get(&[knob]).is_some() {
                return Err(Fail::bad(format!(
                    "{knob}: this model makes videos on a fixed schedule, without guidance; leave it out"
                )));
            }
        }
        let prompt = f.text(&["prompt"]).ok_or_else(|| Fail::bad("prompt is required"))?;
        let (width, height) = match f.text(&["size"]).as_deref().map(str::trim) {
            None | Some("auto") => (None, None),
            Some(s) => {
                let (w, h) = s
                    .split_once(['x', 'X', '×'])
                    .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
                    .ok_or_else(|| Fail::bad(format!("size is WIDTHxHEIGHT, like 768x512, not {s:?}")))?;
                (Some(w), Some(h))
            }
        };
        let frames = f.number(&["frames", "num_frames"])?;
        let seconds: Option<f64> = f.number(&["seconds"])?;
        if frames.is_some() && seconds.is_some() {
            return Err(Fail::bad("give seconds or frames, not both"));
        }
        if seconds.is_some_and(|s| !(s.is_finite() && s > 0.0)) {
            return Err(Fail::bad("seconds must be more than 0"));
        }
        Ok(Asked {
            model: f.text(&["model"]),
            request: VideoRequest {
                prompt,
                width,
                height,
                frames,
                fps: f.number(&["fps", "frame_rate"])?,
                seed: f.number(&["seed"])?,
                audio: f.flag("audio")?,
            },
            seconds,
        })
    }

    /// The request, with `seconds` turned into the nearest number of frames
    /// the model can make.
    fn request(mut self, d: &kvad::video::Defaults) -> VideoRequest {
        if let Some(s) = self.seconds {
            let fps = self.request.fps.unwrap_or(d.fps) as f64;
            let steps = (s * fps / d.frame_step as f64).round().max(1.0) as usize;
            self.request.frames = Some(steps * d.frame_step + 1);
        }
        self.request
    }
}

// ---------------------------------------------------------------------------
// The endpoints
// ---------------------------------------------------------------------------

pub async fn create(who: Identity, headers: HeaderMap, St(state): St<State>, body: Bytes) -> Result<Json<Value>, Fail> {
    let asked = Asked::read(&Fields::read(&headers, &body)?)?;
    let resident = crate::openai::resident_for(&state, asked.model.as_deref(), Kind::Video).await?;
    let defaults = resident
        .model
        .video
        .ok_or_else(|| Fail::internal(format!("{} loaded as a video model with no defaults", resident.model.repo)))?;
    let mut request = asked.request(&defaults);
    // Checked now, so that a size the model cannot make is a 400 and not a
    // failed video after a wait. The seed is fixed now too, so that the row
    // can say which one it is before the video exists.
    let resolved = request.resolved(&defaults).map_err(|e| Fail::bad(e.to_string()))?;
    request.seed = Some(resolved.seed);

    let (db, owner, model, backend) =
        (state.db.clone(), who.id, resident.model.repo.clone(), resident.model.backend.clone());
    let stored = blocking(move || queue(&db, owner, &model, &backend, &resolved)).await?;
    let herald = Herald::new(stored.id);
    tokio::spawn(run(state, stored.id, resident.key, request, herald));
    Ok(Json(stored.resource()))
}

/// Make the video, writing its progress into its row as it goes.
///
/// `_herald` is held for as long as this runs, and every change to the row is
/// told to it; see [`Herald`].
async fn run(state: State, id: i64, key: Key, request: VideoRequest, _herald: Herald) {
    let db = state.db.clone();
    let failed = |why: String| {
        let db = db.clone();
        async move {
            let _ = std::fs::remove_file(dir().join(format!("{id}.preview.png")));
            let _ = blocking(move || fail(&db, id, &why)).await;
            tell(id);
        }
    };
    let mut reels = match state.engine.film(&key, request) {
        Ok(r) => r,
        Err(e) => return failed(e).await,
    };
    // Whether the row is still there is asked every second as well as at
    // each step: a queued video has no steps, and one being made may go a
    // minute between them. Gone is deleted, and returning drops `reels`,
    // which is what tells the scheduler to skip it or the engine to stop.
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        let reel = tokio::select! {
            reel = reels.recv() => reel,
            _ = tick.tick() => {
                let db = db.clone();
                match blocking(move || exists(&db, id)).await {
                    Ok(false) => return,
                    _ => continue,
                }
            }
        };
        let Some(reel) = reel else { break };
        match reel {
            Reel::Step(step) => {
                let db = db.clone();
                let going = blocking(move || {
                    let going = advance(&db, id, &step)?;
                    if let (true, Some(preview)) = (going, &step.preview) {
                        // A preview that cannot be written is not worth a
                        // failed video.
                        if let Err(e) = look(&dir(), id, preview) {
                            tracing::warn!("video {id}: could not keep a preview: {e}");
                        }
                    }
                    Ok(going)
                });
                if !matches!(going.await, Ok(true)) {
                    return;
                }
                tell(id);
            }
            Reel::Done(filmed) => {
                let db = db.clone();
                let ffmpeg = FFMPEG.get().cloned().flatten();
                let finished = blocking(move || finish(&db, &dir(), id, &filmed, ffmpeg.as_deref())).await;
                match finished {
                    Err(Fail(_, why)) => failed(why).await,
                    Ok(_) => tell(id),
                }
                return;
            }
            Reel::Failed(why) => return failed(why).await,
        }
    }
    failed("the engine stopped before it answered".into()).await
}

/// `video_12`, or `12`: the id in a path.
fn id_in(name: &str) -> Result<i64, Fail> {
    name.strip_prefix("video_").unwrap_or(name).parse().map_err(|_| Fail::missing(format!("there is no video {name}")))
}

async fn one(who: &Identity, state: &State, name: &str) -> Result<Stored, Fail> {
    let id = id_in(name)?;
    let (db, owner) = (state.db.clone(), who.id);
    blocking(move || get_one(&db, id, owner)).await?.ok_or_else(|| Fail::missing(format!("there is no video {name}")))
}

async fn retrieve(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Json<Value>, Fail> {
    Ok(Json(one(&who, &state, &name).await?.resource()))
}

async fn remove(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Json<Value>, Fail> {
    let id = id_in(&name)?;
    let (db, owner) = (state.db.clone(), who.id);
    let gone = blocking(move || delete(&db, &dir(), id, owner)).await?;
    tell(id);
    match gone {
        true => Ok(Json(json!({ "id": format!("video_{id}"), "object": "video.deleted", "deleted": true }))),
        false => Err(Fail::missing(format!("there is no video {name}"))),
    }
}

/// A video's changes as server-sent events, until it ends: the video object
/// as `video.updated`, then `video.completed` or `video.failed` with the
/// last of it, or `video.deleted`. The first event is the video as it
/// stands, so a watcher needs no separate `GET`; one that ended long ago is
/// that event alone.
async fn events(who: Identity, St(state): St<State>, Path(name): Path<String>) -> Result<Response, Fail> {
    let id = id_in(&name)?;
    // Before the row is read, so that a change between the two is heard.
    let heard = listen(id);
    let first = one(&who, &state, &name).await?;
    let (tx, rx) = tokio::sync::mpsc::channel::<axum::response::sse::Event>(16);
    tokio::spawn(async move {
        let mut now = Some(first);
        let mut heard = heard;
        loop {
            let (event, data, last) = match &now {
                Some(v) => match v.status.as_str() {
                    "completed" => ("video.completed", v.resource(), true),
                    "failed" => ("video.failed", v.resource(), true),
                    _ => ("video.updated", v.resource(), false),
                },
                None => ("video.deleted", json!({ "id": format!("video_{id}"), "deleted": true }), true),
            };
            if tx.send(crate::models::sse(event, &data)).await.is_err() || last {
                return;
            }
            // Nothing is making it and it has not ended, or its maker has
            // just ended: either way nothing more will be told. (A video
            // left unfinished by a server that stopped is marked failed
            // when the server starts again.)
            let Some(h) = heard.as_mut() else { return };
            // Lagged is as good as heard: the row is read afresh either way.
            if let Err(tokio::sync::broadcast::error::RecvError::Closed) = h.recv().await {
                heard = None;
            }
            let (db, owner) = (state.db.clone(), who.id);
            now = match blocking(move || get_one(&db, id, owner)).await {
                Ok(v) => v,
                Err(_) => return,
            };
        }
    });
    Ok(crate::models::stream(rx).into_response())
}

/// OpenAI's list: newest first unless `order=asc`, `limit` at a time (20
/// unless asked, 100 at most), continuing `after` the id given.
async fn list_route(
    who: Identity,
    St(state): St<State>,
    Query(q): Query<HashMap<String, String>>,
) -> Result<Json<Value>, Fail> {
    let limit = match q.get("limit") {
        None => 20,
        Some(n) => n.parse::<usize>().ok().filter(|n| *n <= 100).ok_or_else(|| Fail::bad("limit is 0 to 100"))?,
    };
    let ascending = match q.get("order").map(String::as_str) {
        None | Some("desc") => false,
        Some("asc") => true,
        Some(o) => return Err(Fail::bad(format!("order is asc or desc, not {o}"))),
    };
    let after = q.get("after").map(|a| id_in(a)).transpose()?;
    let (db, owner) = (state.db.clone(), who.id);
    let mut all = blocking(move || list(&db, owner)).await?;
    if ascending {
        all.reverse();
    }
    let start = match after {
        Some(a) => all.iter().position(|v| v.id == a).map(|i| i + 1).unwrap_or(all.len()),
        None => 0,
    };
    let page: Vec<&Stored> = all[start..].iter().take(limit).collect();
    let has_more = start + page.len() < all.len();
    Ok(Json(json!({
        "object": "list",
        "data": page.iter().map(|v| v.resource()).collect::<Vec<_>>(),
        "first_id": page.first().map(|v| v.name()),
        "last_id": page.last().map(|v| v.name()),
        "has_more": has_more,
    })))
}

/// The MP4, or with `variant=thumbnail` its poster frame, answering Range.
///
/// OpenAI's thumbnail is a WebP; this one is a PNG, said so in its
/// `Content-Type`, because a PNG is what this engine can write. There is no
/// `spritesheet`.
async fn content(
    who: Identity,
    St(state): St<State>,
    Path(name): Path<String>,
    Query(q): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response, Fail> {
    let (ext, kind) = match q.get("variant").map(String::as_str) {
        None | Some("video") => ("mp4", "video/mp4"),
        Some("thumbnail") => ("png", "image/png"),
        Some("preview") => ("preview.png", "image/png"),
        Some(v) => return Err(Fail::bad(format!("variant is video, thumbnail or preview here, not {v}"))),
    };
    let v = one(&who, &state, &name).await?;
    // kvad's own variant: the latest step's rough look, while it is made.
    if ext == "preview.png" {
        let path = dir().join(format!("{}.preview.png", v.id));
        let bytes = match (v.status.as_str(), std::fs::read(&path)) {
            ("in_progress", Ok(bytes)) => bytes,
            _ => return Err(Fail::missing(format!("{} has no preview now", v.name()))),
        };
        return Ok(([(header::CONTENT_TYPE, "image/png"), (header::CACHE_CONTROL, "private, no-store")], bytes).into_response());
    }
    if v.status != "completed" {
        return Err(Fail::missing(format!("{} is {}, not ready", v.name(), v.status.replace('_', " "))));
    }
    let path = dir().join(format!("{}.{ext}", v.id));
    let len = std::fs::metadata(&path).map_err(|e| Fail::internal(format!("{}: {e}", path.display())))?.len();
    // The link names one video forever (see `Stored::link`) only with its
    // `v`; without it, a browser has to ask each time.
    let made: String = v.created_at.chars().filter(char::is_ascii_digit).collect();
    let cache = match q.get("v") == Some(&made) {
        true => "private, max-age=31536000, immutable",
        false => "private, no-cache",
    };
    let asked = headers.get(header::RANGE).and_then(|r| r.to_str().ok());
    let (status, from, to) = match range(asked, len) {
        Ok(None) => (StatusCode::OK, 0, len.saturating_sub(1)),
        Ok(Some((a, b))) => (StatusCode::PARTIAL_CONTENT, a, b),
        Err(()) => {
            return Ok((
                StatusCode::RANGE_NOT_SATISFIABLE,
                [(header::CONTENT_RANGE, format!("bytes */{len}"))],
            )
                .into_response())
        }
    };
    let count = if len == 0 { 0 } else { to - from + 1 };
    let mut response = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, kind)
        .header(header::CONTENT_LENGTH, count)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CACHE_CONTROL, cache);
    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header(header::CONTENT_RANGE, format!("bytes {from}-{to}/{len}"));
    }
    response.body(file_body(path, from, count)).map_err(|e| Fail::internal(e.to_string()))
}

/// `count` bytes of a file from `from`, read a megabyte at a time on a
/// blocking thread, so that a 285 MB video is never in memory at once.
fn file_body(path: PathBuf, from: u64, count: u64) -> Body {
    use std::io::{Read, Seek, SeekFrom};
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    tokio::task::spawn_blocking(move || {
        let send = |chunk| tx.blocking_send(chunk).is_ok();
        let mut file = match std::fs::File::open(&path).and_then(|mut f| f.seek(SeekFrom::Start(from)).map(|_| f)) {
            Ok(f) => f,
            Err(e) => {
                send(Err(e));
                return;
            }
        };
        let mut left = count;
        while left > 0 {
            let mut buf = vec![0u8; left.min(1 << 20) as usize];
            if let Err(e) = file.read_exact(&mut buf) {
                send(Err(e));
                return;
            }
            left -= buf.len() as u64;
            // The client went away.
            if !send(Ok(Bytes::from(buf))) {
                return;
            }
        }
    });
    Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx))
}

/// The bytes a `Range` header asks for, of a file `len` long: `None` for the
/// whole file, `Err` for a range that starts past its end (416).
///
/// One range only. A request for several is answered with the whole file,
/// which RFC 9110 allows, and so is a header that does not parse: a server
/// may ignore Range. `bytes=a-b`, `bytes=a-`, and `bytes=-n` for the last n.
fn range(header: Option<&str>, len: u64) -> Result<Option<(u64, u64)>, ()> {
    let Some(spec) = header.and_then(|h| h.trim().strip_prefix("bytes=")) else { return Ok(None) };
    let Some((a, b)) = spec.split_once('-').filter(|_| !spec.contains(',')) else { return Ok(None) };
    let num = |s: &str| s.trim().parse::<u64>().ok();
    let (first, last) = match (a.trim().is_empty(), b.trim().is_empty()) {
        // `-n`: the last n bytes, of which there are never none.
        (true, false) => match num(b) {
            Some(0) => return Err(()),
            Some(n) => (len.saturating_sub(n), u64::MAX),
            None => return Ok(None),
        },
        (false, true) => match num(a) {
            Some(a) => (a, u64::MAX),
            None => return Ok(None),
        },
        (false, false) => match (num(a), num(b)) {
            (Some(a), Some(b)) if a <= b => (a, b),
            _ => return Ok(None),
        },
        (true, true) => return Ok(None),
    };
    if first >= len {
        return Err(());
    }
    Ok(Some((first, last.min(len - 1))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::video::{Audio, Resolved, Video};

    fn resolved(seed: u64) -> Resolved {
        Resolved { prompt: "a red and a blue pixel".into(), width: 2, height: 2, frames: 3, fps: 24, seed, audio: true }
    }

    fn filmed(seed: u64) -> Filmed {
        Filmed {
            video: Video { width: 2, height: 2, fps: 24, rgb: (0..36).map(|i| (i * 7) as u8).collect() },
            audio: Some(Audio { rate: 48_000, channels: 2, samples: vec![0.0; 960] }),
            request: resolved(seed),
            encode_secs: 1.0,
            denoise_secs: 2.0,
            decode_secs: 3.0,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvad-videos-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A video's life in its row: queued, in progress, completed with its
    /// files beside it, and gone with them.
    #[test]
    fn a_video_is_queued_advanced_kept_and_forgotten() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("life");
        let seed = u64::MAX - 7;
        let q = queue(&db, None, "Lightricks/LTX-2.5", "metal q8", &resolved(seed)).unwrap();
        assert_eq!((q.status.as_str(), q.seed, q.started), ("queued", seed, None));
        assert_eq!(q.resource()["kvad"]["url"], Value::Null, "no link before there is a file");

        let step = kvad::video::Step { phase: "stage 2", done: 1, total: 3, progress: 0.5, elapsed: 30.0, preview: None };
        assert!(advance(&db, q.id, &step).unwrap());
        let s = get_one(&db, q.id, None).unwrap().unwrap();
        assert_eq!((s.status.as_str(), s.phase.as_deref(), s.progress), ("in_progress", Some("stage 2"), 0.5));
        assert!(s.started.is_some());
        assert_eq!(s.resource()["progress"], 50);
        assert_eq!(s.resource()["kvad"]["preview_url"], json!(format!("/v1/videos/video_{}/content?variant=preview&at=500", q.id)));
        let preview = kvad::image::Image { width: 2, height: 1, rgb: vec![1, 2, 3, 4, 5, 6] };
        look(&dir, q.id, &preview).unwrap();
        assert_eq!(std::fs::read(dir.join(format!("{}.preview.png", q.id))).unwrap(), preview.png());

        assert!(finish(&db, &dir, q.id, &filmed(seed), None).unwrap());
        let s = get_one(&db, q.id, None).unwrap().unwrap();
        let mp4 = std::fs::read(dir.join(format!("{}.mp4", q.id))).unwrap();
        assert_eq!(mp4, filmed(seed).video.mp4(filmed(seed).audio.as_ref()));
        assert_eq!(s.bytes, Some(mp4.len() as u64));
        let r = s.resource();
        assert_eq!((r["status"].as_str(), r["progress"].as_i64(), r["size"].as_str()), (Some("completed"), Some(100), Some("2x2")));
        assert!(r["kvad"]["url"].as_str().unwrap().starts_with(&format!("/v1/videos/video_{}/content?v=", q.id)));
        assert!(dir.join(format!("{}.png", q.id)).exists());
        assert!(!dir.join(format!("{}.preview.png", q.id)).exists(), "a finished video's preview goes");
        assert_eq!(r["kvad"]["preview_url"], Value::Null);

        assert!(delete(&db, &dir, q.id, None).unwrap());
        assert!(!dir.join(format!("{}.mp4", q.id)).exists());
        assert!(!dir.join(format!("{}.png", q.id)).exists());
        assert!(!advance(&db, q.id, &step).unwrap(), "a deleted video's generation is told to stop");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A video deleted while its files were being written leaves no files.
    #[test]
    fn a_video_deleted_before_it_finished_leaves_nothing() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("deleted");
        let q = queue(&db, None, "m", "b", &resolved(1)).unwrap();
        assert!(delete(&db, &dir, q.id, None).unwrap());
        assert!(!finish(&db, &dir, q.id, &filmed(1), None).unwrap());
        assert!(!dir.join(format!("{}.mp4", q.id)).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_restart_fails_what_was_running_and_leaves_what_was_done() {
        let db = Db::in_memory().unwrap();
        let dir = scratch("restart");
        let (a, b, c) = (
            queue(&db, None, "m", "b", &resolved(1)).unwrap(),
            queue(&db, None, "m", "b", &resolved(2)).unwrap(),
            queue(&db, None, "m", "b", &resolved(3)).unwrap(),
        );
        let step = kvad::video::Step { phase: "text", done: 0, total: 1, progress: 0.0, elapsed: 0.0, preview: None };
        advance(&db, b.id, &step).unwrap();
        finish(&db, &dir, c.id, &filmed(3), None).unwrap();
        assert_eq!(abandon(&db).unwrap(), 2);
        let status = |id| get_one(&db, id, None).unwrap().unwrap().status;
        assert_eq!((status(a.id), status(b.id), status(c.id)), ("failed".into(), "failed".into(), "completed".into()));
        let r = get_one(&db, a.id, None).unwrap().unwrap().resource();
        assert_eq!(r["error"]["code"], "generation_failed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn another_accounts_videos_are_not_found() {
        let db = Db::in_memory().unwrap();
        db.with(|c| c.execute("INSERT INTO users (id, name, role) VALUES (1, 'a', 'user'), (2, 'b', 'user')", []))
            .unwrap();
        let dir = scratch("owned");
        let mine = queue(&db, Some(1), "m", "b", &resolved(1)).unwrap();
        assert!(get_one(&db, mine.id, Some(2)).unwrap().is_none());
        assert!(list(&db, Some(2)).unwrap().is_empty());
        assert!(!delete(&db, &dir, mine.id, Some(2)).unwrap());
        assert!(get_one(&db, mine.id, Some(1)).unwrap().is_some());
    }

    #[test]
    fn ffmpeg_is_found_where_it_is_asked_for_and_nowhere_else() {
        let dir = scratch("ffmpeg");
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("ffmpeg");
        std::fs::write(&fake, b"").unwrap();
        assert_eq!(find_ffmpeg("off", &[dir.clone()]), None);
        assert_eq!(find_ffmpeg("auto", &[dir.clone()]), Some(fake.clone()), "PATH comes first");
        assert_eq!(find_ffmpeg(fake.to_str().unwrap(), &[]), Some(fake.clone()));
        assert_eq!(find_ffmpeg(dir.join("nothing").to_str().unwrap(), &[]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A video from a real `ffmpeg`, where there is one: whole, smaller, and
    /// what `bytes` says; and from one that fails, the uncompressed file.
    #[test]
    fn a_video_is_compressed_when_ffmpeg_works_and_kept_when_it_does_not() {
        let Some(ffmpeg) = find_ffmpeg("auto", &[]) else { return };
        let db = Db::in_memory().unwrap();
        let dir = scratch("compressed");
        let (w, h, n) = (64, 64, 9);
        let rgb = (0..w * h * 3 * n).map(|i| ((i / 3) % w * 4 + i / (w * h * 3) * 8) as u8).collect();
        let f = Filmed { video: Video { width: w, height: h, fps: 24, rgb }, ..filmed(1) };
        let raw = f.video.mp4(f.audio.as_ref());

        let q = queue(&db, None, "m", "b", &resolved(1)).unwrap();
        assert!(finish(&db, &dir, q.id, &f, Some(&ffmpeg)).unwrap());
        let mp4 = std::fs::read(dir.join(format!("{}.mp4", q.id))).unwrap();
        assert!(mp4.len() < raw.len(), "{} bytes compressed against {}", mp4.len(), raw.len());
        assert_eq!(get_one(&db, q.id, None).unwrap().unwrap().bytes, Some(mp4.len() as u64));
        assert!(std::fs::read_dir(&dir).unwrap().all(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            !name.contains("part") && !name.contains("x264")
        }), "nothing left aside");

        let q = queue(&db, None, "m", "b", &resolved(2)).unwrap();
        let broken = dir.join("not-ffmpeg");
        std::fs::write(&broken, b"").unwrap();
        assert!(finish(&db, &dir, q.id, &f, Some(&broken)).unwrap());
        assert_eq!(std::fs::read(dir.join(format!("{}.mp4", q.id))).unwrap(), raw);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A watcher hears each change while a video is made, and hears the
    /// channel close when the generation's task lets its herald go.
    #[tokio::test]
    async fn a_herald_tells_its_listeners_and_closes_when_dropped() {
        let id = -7; // no row, and no other test's
        assert!(listen(id).is_none());
        let herald = Herald::new(id);
        let mut rx = listen(id).unwrap();
        tell(id);
        assert!(rx.recv().await.is_ok());
        drop(herald);
        assert!(matches!(rx.recv().await, Err(tokio::sync::broadcast::error::RecvError::Closed)));
        assert!(listen(id).is_none());
    }

    #[test]
    fn seconds_are_written_as_openai_writes_them() {
        assert_eq!(seconds(121, 24), "5.04");
        assert_eq!(seconds(96, 24), "4");
        assert_eq!(seconds(125, 25), "5");
    }

    fn json_fields(v: Value) -> Fields {
        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        Fields::read(&h, v.to_string().as_bytes()).unwrap()
    }

    const LTX: kvad::video::Defaults = kvad::video::Defaults {
        width: 768,
        height: 512,
        frames: 121,
        fps: 24,
        multiple: 64,
        frame_step: 8,
        max_frames: 121,
        max_volume: 1536 * 1024 * 121,
    };

    #[test]
    fn a_request_is_read_in_openai_s_names_and_kvad_s() {
        let a = Asked::read(&json_fields(json!({
            "model": "Lightricks/LTX-2.5", "prompt": "a dog", "size": "1536x1024", "seconds": "4", "seed": 3, "audio": false
        })))
        .unwrap();
        assert_eq!(a.model.as_deref(), Some("Lightricks/LTX-2.5"));
        let r = a.request(&LTX);
        assert_eq!((r.width, r.height, r.frames, r.fps, r.seed, r.audio), (Some(1536), Some(1024), Some(97), None, Some(3), Some(false)));

        // Five seconds at 24 fps is LTX's 121; at 25, 125.
        let five = |fps: Value| Asked::read(&json_fields(json!({ "prompt": "a dog", "seconds": 5, "fps": fps }))).unwrap().request(&LTX).frames;
        assert_eq!(five(json!(24)), Some(121));
        // 125 frames is not 8k + 1, and 129 is the nearest that is.
        assert_eq!(five(json!("25")), Some(129));

        let r = Asked::read(&json_fields(json!({ "prompt": "a dog", "num_frames": 49 }))).unwrap().request(&LTX);
        assert_eq!(r.frames, Some(49));
    }

    #[test]
    fn what_this_model_cannot_do_is_refused_by_name() {
        let refuse = |v: Value| match Asked::read(&json_fields(v)) {
            Ok(_) => panic!("accepted"),
            Err(Fail(_, why)) => why,
        };
        assert!(refuse(json!({ "prompt": "a dog", "input_reference": { "image_url": "x" } })).contains("from a picture"));
        assert!(refuse(json!({ "prompt": "a dog", "negative_prompt": "blur" })).contains("without guidance"));
        assert!(refuse(json!({ "prompt": "a dog", "seconds": 4, "frames": 97 })).contains("not both"));
        assert!(refuse(json!({ "prompt": "a dog", "size": "big" })).contains("WIDTHxHEIGHT"));
        assert!(refuse(json!({ "size": "768x512" })).contains("prompt"));
        // Empty is the same as left out.
        assert!(Asked::read(&json_fields(json!({ "prompt": "a dog", "negative_prompt": "" }))).is_ok());
    }

    /// The body OpenAI's Python SDK sends for `videos.create`, with a file
    /// part added, as `httpx` writes it.
    #[test]
    fn a_form_is_read_as_the_sdk_sends_it() {
        let body = "--abc123\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na dog\r\non a beach\r\n\
                    --abc123\r\nContent-Disposition: form-data; name=\"seconds\"\r\n\r\n4\r\n\
                    --abc123\r\nContent-Disposition: form-data; name=\"input_reference\"; filename=\"a.png\"\r\n\
                    Content-Type: image/png\r\n\r\n\x01\x02\r\n--abc123--\r\n";
        let parts = form(body.as_bytes(), "multipart/form-data; boundary=abc123").unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], Part { name: "prompt".into(), filename: None, data: b"a dog\r\non a beach".to_vec() });
        assert_eq!(parts[2].filename.as_deref(), Some("a.png"));
        assert_eq!(parts[2].data, [1, 2]);

        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, "multipart/form-data; boundary=\"abc123\"".parse().unwrap());
        let f = Fields::read(&h, body.as_bytes()).unwrap();
        assert_eq!(f.files, ["input_reference"]);
        assert_eq!(f.number::<f64>(&["seconds"]).unwrap(), Some(4.0));
        assert!(form(b"--abc123\r\nno headers end", "multipart/form-data; boundary=abc123").is_err());
        assert!(form(body.as_bytes(), "multipart/form-data").is_err());
    }

    #[test]
    fn a_range_is_read_as_rfc_9110_says() {
        assert_eq!(range(None, 1000), Ok(None));
        assert_eq!(range(Some("bytes=0-"), 1000), Ok(Some((0, 999))));
        assert_eq!(range(Some("bytes=100-199"), 1000), Ok(Some((100, 199))));
        assert_eq!(range(Some("bytes=900-5000"), 1000), Ok(Some((900, 999))), "an end past the file is the end");
        assert_eq!(range(Some("bytes=-100"), 1000), Ok(Some((900, 999))));
        assert_eq!(range(Some("bytes=-5000"), 1000), Ok(Some((0, 999))));
        assert_eq!(range(Some("bytes=1000-"), 1000), Err(()));
        assert_eq!(range(Some("bytes=-0"), 1000), Err(()));
        // Ignored, and answered with the whole file.
        assert_eq!(range(Some("bytes=0-1,5-6"), 1000), Ok(None));
        assert_eq!(range(Some("bytes=5-1"), 1000), Ok(None));
        assert_eq!(range(Some("items=0-1"), 1000), Ok(None));
        assert_eq!(range(Some("bytes=a-b"), 1000), Ok(None));
    }

    #[test]
    fn an_id_is_read_with_or_without_its_prefix() {
        assert_eq!(id_in("video_12").unwrap(), 12);
        assert_eq!(id_in("12").unwrap(), 12);
        assert!(id_in("video_../x").is_err());
    }
}
