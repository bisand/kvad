//! Training a LoRA for an image model: the pictures, and the run.
//!
//! The text half of training is [`crate::training`]; this is the same shape
//! for pictures. A dataset of them is a folder, filled a file at a time by
//! the page that uploads it, and a run is a [`crate::jobs`] job of the kind
//! `tune`, followed over the same `/api/jobs/{id}/events` as any other.
//!
//! The run itself is `kvad-gpu tune`, started as a process of its own; why
//! is at [`crate::jobs::tune`]. What this file decides is everything that
//! can be refused before it starts: a model that is not one it trains, a
//! picture with no caption, and above all whether it fits. A backward pass
//! past the machine's memory is not refused by macOS, it takes the machine
//! down, so the memory a run is measured to take is set aside in the
//! scheduler's budget first, and a model is not loaded into it meanwhile.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
use crate::datasets::{self, Dataset};
use crate::jobs::{self, Job, TuneParams, TuneRun};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, State as St};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use std::path::PathBuf;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/tune", post(start))
        .route("/api/tune/options", get(options))
        .route("/api/datasets/pictures", post(create))
        .route("/api/datasets/{id}/pictures", get(pictures))
        .route("/api/datasets/{id}/files/{file}", get(file).put(put).delete(remove))
        .route("/api/jobs/{id}/pictures/{file}", get(sample))
        // One picture a request, and the store has its own limit.
        .layer(DefaultBodyLimit::max(datasets::MAX_PICTURE_BYTES + 1024))
}

/// The one pipeline a LoRA is trained for so far (`docs/tune.md`).
const TRAINS: &str = "StableDiffusionXLPipeline";

/// Pixels a side a run may be asked for: what has been measured.
const SIDES: [usize; 3] = [512, 768, 1024];

/// The most prompts a run draws at each measurement. Each is a picture's
/// worth of time, sixteen seconds at 512², at every one of them.
const MAX_SAMPLES: usize = 4;

/// What a run at `side` pixels is set aside, in bytes, with or without
/// samples drawn beside it.
///
/// From `docs/tune.md`'s measurements on an M5 Pro: a step's backward pass
/// reaches 7.2 GB at 512² and 10.6 GB at 1024², and between them it grows
/// with the picture's area (8.2 GB at 768²). Drawing a sample reaches 8.7 GB
/// whatever the run's size. A fifth is added, because the figure is one
/// machine's and the run ends itself at it.
fn takes(side: usize, samples: bool) -> u64 {
    let area = (side as f64 / 1024.0).powi(2);
    let step = 6.0 + 4.6 * area;
    let peak = if samples { step.max(8.7) } else { step };
    (peak * 1.2 * 1e9) as u64
}

/// `kvad-gpu`, where the installer and a build both put it: beside this
/// binary.
fn trainer() -> Option<PathBuf> {
    let beside = std::env::current_exe().ok()?.parent()?.join("kvad-gpu");
    beside.is_file().then_some(beside)
}

/// The `ffmpeg` a run decodes the pictures with: the server's own, or the
/// first found where `[videos] ffmpeg` said to use none for videos.
fn ffmpeg() -> Option<PathBuf> {
    crate::videos::ffmpeg().or_else(|| {
        let path = std::env::var_os("PATH").unwrap_or_default();
        kvad::video::ffmpeg_on(&std::env::split_paths(&path).collect::<Vec<_>>())
    })
}

/// Why no LoRA can be trained on this machine, if none can.
fn unavailable() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return Some("A LoRA is trained on a Mac's GPU, by kvad-gpu, which is not built for this system.".into());
    }
    if trainer().is_none() {
        return Some("kvad-gpu trains the LoRA, and it is not beside kvad-serve. The installer puts it there; from a checkout, cargo build --release -p kvad-gpu.".into());
    }
    if ffmpeg().is_none() {
        return Some("The pictures are decoded by ffmpeg, and none was found. Install it (brew install ffmpeg) and restart the server.".into());
    }
    None
}

/// The models on this machine a LoRA can be trained for: SDXL and its
/// fine-tunes, in diffusers' layout.
fn trainable() -> Vec<String> {
    kvad::hub::local_models()
        .into_iter()
        .filter(|m| m.complete && m.gguf.is_none() && m.single.is_none() && m.lora.is_none())
        .filter(|m| kvad::hub::pipeline(m).as_deref() == Some(TRAINS))
        .map(|m| m.id)
        .collect()
}

/// Where a LoRA trained here is written.
fn lora_file(name: &str) -> PathBuf {
    kvad::weights::data_dir().join("loras").join(format!("{name}.safetensors"))
}

#[derive(serde::Serialize)]
pub struct Options {
    /// Why not, when no LoRA can be trained here; the rest is then for
    /// showing, not for asking with.
    unavailable: Option<String>,
    models: Vec<String>,
    /// The model named when a request names none, and the one to pull when
    /// `models` is empty.
    base: &'static str,
    sides: Vec<Side>,
    defaults: Defaults,
    max_samples: usize,
    /// Bytes of the memory budget nothing holds.
    left: u64,
    /// Whether a run is going, of text or of a LoRA.
    training: bool,
}

#[derive(serde::Serialize)]
pub struct Side {
    side: usize,
    /// Bytes set aside for a run at this size, without and with samples.
    takes: u64,
    takes_sampling: u64,
}

#[derive(serde::Serialize)]
pub struct Defaults {
    size: usize,
    rank: usize,
    steps: usize,
    lr: f64,
    eval_every: usize,
    seed: u64,
    sample_size: usize,
    sample_steps: usize,
}

/// What a run is asked for when it does not say. `kvad-gpu tune`'s own, but
/// for the samples' size: 512², which is 16 s a sample where the run's own
/// 1024² is 72.
const DEFAULTS: Defaults = Defaults { size: 1024, rank: 16, steps: 1000, lr: 1e-4, eval_every: 100, seed: 1337, sample_size: 512, sample_steps: 20 };

/// SDXL's base, which `kvad-gpu tune` trains for when no model is named.
const BASE: &str = "stabilityai/stable-diffusion-xl-base-1.0";

async fn options(_: Admin, St(state): St<State>) -> Result<Json<Options>, Fail> {
    let jobs = state.jobs.clone();
    let (unavailable, models, training) = blocking(move || Ok((unavailable(), trainable(), jobs.training()))).await?;
    Ok(Json(Options {
        unavailable,
        models,
        base: BASE,
        sides: SIDES.iter().map(|&side| Side { side, takes: takes(side, false), takes_sampling: takes(side, true) }).collect(),
        defaults: DEFAULTS,
        max_samples: MAX_SAMPLES,
        left: state.engine.left(),
        training,
    }))
}

#[derive(serde::Deserialize)]
pub struct StartRequest {
    dataset: i64,
    name: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    size: Option<usize>,
    #[serde(default)]
    rank: Option<usize>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    lr: Option<f64>,
    #[serde(default)]
    eval_every: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// The caption of every picture that has none.
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    samples: Vec<String>,
    #[serde(default)]
    sample_size: Option<usize>,
    #[serde(default)]
    sample_steps: Option<usize>,
}

/// What a request means, with every refusal that needs no disk: a number
/// out of its range, a name that is not one.
fn resolved(body: &StartRequest, dataset: &Dataset) -> Result<TuneParams, String> {
    let name = body.name.trim();
    if !kvad::weights::is_model_name(name) || datasets::check_name(name).is_err() || name.contains(' ') {
        return Err(format!("`{name}` is not a name for a LoRA: one word of letters, digits and `-`, `_` or `.`"));
    }
    let size = body.size.unwrap_or(DEFAULTS.size);
    if !SIDES.contains(&size) {
        return Err(format!("a run is {} pixels a side, not {size}: those are the sizes whose memory has been measured", SIDES.map(|s| s.to_string()).join(", ")));
    }
    let sample_size = body.sample_size.unwrap_or(DEFAULTS.sample_size.min(size));
    if !SIDES.contains(&sample_size) {
        return Err(format!("a sample is {} pixels a side, not {sample_size}", SIDES.map(|s| s.to_string()).join(", ")));
    }
    let steps = body.steps.unwrap_or(DEFAULTS.steps);
    let eval_every = body.eval_every.unwrap_or(DEFAULTS.eval_every.min(steps.max(1)));
    let rank = body.rank.unwrap_or(DEFAULTS.rank);
    let lr = body.lr.unwrap_or(DEFAULTS.lr);
    let sample_steps = body.sample_steps.unwrap_or(DEFAULTS.sample_steps);
    for (what, n, most) in [("steps", steps, 100_000), ("rank", rank, 128), ("eval_every", eval_every, steps.max(1)), ("sample_steps", sample_steps, 100)] {
        if n == 0 || n > most {
            return Err(format!("{what} is from 1 to {most}, not {n}"));
        }
    }
    if !(lr.is_finite() && lr > 0.0 && lr <= 1.0) {
        return Err(format!("the learning rate is a number above 0, 0.0001 unless there is a reason, not {lr}"));
    }
    let samples: Vec<String> = body.samples.iter().map(|s| s.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|s| !s.is_empty()).collect();
    if samples.len() > MAX_SAMPLES || samples.iter().any(|s| s.len() > 1000) {
        return Err(format!("a run draws at most {MAX_SAMPLES} prompts, each of at most 1000 characters"));
    }
    let caption = body.caption.as_deref().map(str::trim).filter(|c| !c.is_empty()).map(str::to_string);
    Ok(TuneParams {
        name: name.to_string(),
        model: body.model.clone().unwrap_or_else(|| BASE.to_string()),
        dataset: dataset.id,
        dataset_name: dataset.name.clone(),
        pictures: dataset.pictures as usize,
        size,
        rank,
        steps,
        lr,
        eval_every,
        seed: body.seed.unwrap_or(DEFAULTS.seed),
        caption,
        samples,
        sample_size,
        sample_steps,
    })
}

/// The pictures of `held` with no caption, as a refusal that names them:
/// all of them, so that the fix is one visit to the dataset and not one a
/// picture.
fn uncaptioned(dataset: &str, held: &[datasets::Picture]) -> Option<String> {
    let bare: Vec<&str> = held.iter().filter(|p| p.caption.is_none()).map(|p| p.file.as_str()).collect();
    if bare.is_empty() {
        return None;
    }
    let more = match bare.len() > 12 {
        true => format!(" and {} more", bare.len() - 12),
        false => String::new(),
    };
    Some(format!(
        "{} of the {} pictures in `{dataset}` {} no caption: {}{more}. Give each one, or give the run a caption for every picture without.",
        bare.len(),
        held.len(),
        if bare.len() == 1 { "has" } else { "have" },
        bare.iter().take(12).copied().collect::<Vec<_>>().join(", "),
    ))
}

async fn start(who: Admin, St(state): St<State>, Json(body): Json<StartRequest>) -> Result<Json<Job>, Fail> {
    let (db, jobs, engine) = (state.db.clone(), state.jobs.clone(), state.engine.clone());
    let owner = who.0.id;
    blocking(move || {
        if let Some(why) = unavailable() {
            return Err(why.into());
        }
        let (dataset, data) = datasets::picture_dir(&db, body.dataset)?;
        let params = resolved(&body, &dataset)?;
        let held = datasets::pictures(&dataset.name)?;
        if held.is_empty() {
            return Err(format!("`{}` holds no pictures yet", dataset.name).into());
        }
        if params.caption.is_none() {
            if let Some(why) = uncaptioned(&dataset.name, &held) {
                return Err(why.into());
            }
        }
        if !trainable().contains(&params.model) {
            return Err(format!(
                "`{}` is not a model on this machine a LoRA is trained for: that is SDXL, `{BASE}`, or a fine-tune of it in diffusers' layout",
                params.model
            )
            .into());
        }
        let out = lora_file(&params.name);
        if out.exists() {
            return Err(format!("there is a LoRA called `{}` already, {}. Another name, or delete that file first.", params.name, out.display()).into());
        }
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if let Some(other) = jobs.live_kind(&["train", "tune"]) {
            return Err(format!("a training run is already going, {other}; wait for it or stop it").into());
        }

        // Last, so that nothing above can fail with the memory set aside.
        let bytes = takes(params.size, !params.samples.is_empty());
        let what = format!("A LoRA run at {}²", params.size);
        let hold = engine.reserve(&what, bytes).map_err(|mut why| {
            why.push('.');
            if params.size > SIDES[0] {
                why.push_str(&format!(" At {}² a run takes about {:.1} GB.", SIDES[0], takes(SIDES[0], !params.samples.is_empty()) as f64 / 1e9));
            }
            why
        })?;
        let run = TuneRun {
            program: trainer().ok_or("kvad-gpu is not beside kvad-serve")?,
            data,
            out,
            ffmpeg: ffmpeg().ok_or("no ffmpeg was found")?,
            cap_gb: bytes as f64 / 1e9,
            hold: Box::new(hold),
        };
        jobs::tune(&jobs, params, run, owner)
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

// ---------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct Named {
    name: String,
}

/// Start a dataset of pictures, or find the one of that name: an upload is
/// this and then a `PUT` a file, and a second upload adds to the first.
async fn create(who: Admin, St(state): St<State>, Query(q): Query<Named>) -> Result<Json<Dataset>, Fail> {
    let db = state.db.clone();
    let owner = who.0.id;
    blocking(move || datasets::create_pictures(&db, q.name.trim(), owner)).await.map(Json).map_err(|e| Fail::bad(e.1))
}

async fn pictures(_: Admin, St(state): St<State>, Path(id): Path<i64>) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    blocking(move || {
        let (dataset, _) = datasets::picture_dir(&db, id)?;
        let held = datasets::pictures(&dataset.name)?;
        Ok(json!({ "dataset": dataset, "pictures": held }))
    })
    .await
    .map(Json)
    .map_err(|e| Fail::missing(e.1))
}

/// Put one file: a picture, or the caption of one, by its name.
///
/// The body is the file, as an upload of text is the text: a browser reads
/// a file and sends it in three lines, and a folder of forty pictures is
/// forty requests that each succeed or say which picture was wrong.
async fn put(_: Admin, St(state): St<State>, Path((id, name)): Path<(i64, String)>, bytes: Bytes) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    blocking(move || {
        datasets::put_file(&db, id, &name, &bytes)?;
        Ok(json!({ "file": name, "bytes": bytes.len() }))
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

async fn file(_: Admin, St(state): St<State>, Path((id, name)): Path<(i64, String)>) -> Result<Response, Fail> {
    let db = state.db.clone();
    let shown = name.clone();
    let bytes = blocking(move || {
        let (_, dir) = datasets::picture_dir(&db, id)?;
        datasets::check_file(&name)?;
        Ok(std::fs::read(dir.join(&name)).ok())
    })
    .await
    .map_err(|e| Fail::missing(e.1))?
    .ok_or_else(|| Fail::missing(format!("there is no {shown} in that dataset")))?;
    let kind = match datasets::sniff(&bytes) {
        Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "text/plain; charset=utf-8",
    };
    // A file can be replaced under the same name, so nothing is kept.
    Ok(([(header::CONTENT_TYPE, kind), (header::CACHE_CONTROL, "no-store")], bytes).into_response())
}

async fn remove(_: Admin, St(state): St<State>, Path((id, name)): Path<(i64, String)>) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let shown = name.clone();
    match blocking(move || datasets::remove_file(&db, id, &name)).await.map_err(|e| Fail::bad(e.1))? {
        true => Ok(Json(json!({ "deleted": shown }))),
        false => Err(Fail::missing(format!("there is no {shown} in that dataset"))),
    }
}

/// A picture a run drew, by the name the run's events give it.
async fn sample(_: Identity, St(state): St<State>, Path((id, name)): Path<(i64, String)>) -> Result<Response, Fail> {
    let jobs = state.jobs.clone();
    let shown = name.clone();
    let bytes = blocking(move || {
        // Only a file the job's own rows name: the path is made of what a
        // run wrote down, never of what a request asked for.
        match jobs.pictures(id)?.iter().any(|p| p.file == name) {
            true => Ok(std::fs::read(jobs::samples_dir(id).join(&name)).ok()),
            false => Ok(None),
        }
    })
    .await?
    .ok_or_else(|| Fail::missing(format!("job {id} drew no {shown}")))?;
    // A step's picture never changes once it is drawn.
    Ok(([(header::CONTENT_TYPE, "image/png"), (header::CACHE_CONTROL, "private, max-age=31536000, immutable")], bytes).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> Dataset {
        Dataset {
            id: 3,
            name: "my dog".into(),
            kind: "pictures".into(),
            pictures: 12,
            uncaptioned: 0,
            bytes: 0,
            characters: 0,
            distinct: 0,
            created_at: String::new(),
            source: None,
            manifest: false,
            present: true,
        }
    }

    fn request(json: serde_json::Value) -> StartRequest {
        serde_json::from_value(json).unwrap()
    }

    /// The figures `docs/tune.md` measured, with the fifth on top: what a
    /// run is set aside is never under what one was seen to take.
    #[test]
    fn a_run_is_set_aside_more_than_it_was_measured_to_take() {
        for (side, measured) in [(512, 7.2e9), (768, 8.3e9), (1024, 10.6e9)] {
            let aside = takes(side, false) as f64;
            assert!(aside > measured * 1.15 && aside < measured * 1.25, "{side}: {aside}");
        }
        // Drawing a sample reaches 8.7 GB whatever the run's size.
        assert!(takes(512, true) as f64 > 8.7e9 * 1.15);
        assert_eq!(takes(1024, true), takes(1024, false), "a 1024² step is the larger of the two");
    }

    #[test]
    fn a_request_says_only_what_it_wants_changed() {
        let p = resolved(&request(json!({ "dataset": 3, "name": "my-style" })), &dataset()).unwrap();
        assert_eq!((p.model.as_str(), p.size, p.rank, p.steps, p.eval_every, p.sample_size), (BASE, 1024, 16, 1000, 100, 512));
        assert_eq!((p.dataset, p.dataset_name.as_str(), p.pictures), (3, "my dog", 12));
        assert!(p.samples.is_empty() && p.caption.is_none());

        // A short run measures at least once, and a small one samples small.
        let p = resolved(&request(json!({ "dataset": 3, "name": "s", "steps": 40, "size": 512, "samples": [" a  dog ", ""], "caption": "  " })), &dataset()).unwrap();
        assert_eq!((p.eval_every, p.sample_size), (40, 512));
        assert_eq!(p.samples, ["a dog"]);
        assert_eq!(p.caption, None);
    }

    #[test]
    fn a_request_out_of_range_is_refused_by_what_is_wrong() {
        for (body, wrong) in [
            (json!({ "dataset": 3, "name": "a/b" }), "not a name"),
            (json!({ "dataset": 3, "name": "two words" }), "not a name"),
            (json!({ "dataset": 3, "name": "s", "size": 640 }), "640"),
            (json!({ "dataset": 3, "name": "s", "steps": 0 }), "steps"),
            (json!({ "dataset": 3, "name": "s", "rank": 500 }), "rank"),
            (json!({ "dataset": 3, "name": "s", "steps": 50, "eval_every": 80 }), "eval_every"),
            (json!({ "dataset": 3, "name": "s", "lr": 0.0 }), "learning rate"),
            (json!({ "dataset": 3, "name": "s", "samples": ["a", "b", "c", "d", "e"] }), "at most 4"),
        ] {
            let said = resolved(&request(body.clone()), &dataset()).unwrap_err();
            assert!(said.contains(wrong), "{body}: {said}");
        }
    }

    /// Every picture without a caption is named, not the first.
    #[test]
    fn pictures_without_captions_are_all_named() {
        let p = |file: &str, caption: Option<&str>| datasets::Picture { file: file.into(), bytes: 1, caption: caption.map(str::to_string) };
        assert_eq!(uncaptioned("d", &[p("a.png", Some("a dog"))]), None);
        let said = uncaptioned("d", &[p("a.png", Some("a dog")), p("b.png", None), p("c.jpg", None)]).unwrap();
        assert!(said.starts_with("2 of the 3 pictures in `d` have no caption: b.png, c.jpg."), "{said}");
        let many: Vec<_> = (0..15).map(|i| p(&format!("{i:02}.png"), None)).collect();
        assert!(uncaptioned("d", &many).unwrap().contains("11.png and 3 more"));
    }
}
