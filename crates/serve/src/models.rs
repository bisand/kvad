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
use kvad::quant::Precision;
use kvad::service::Backend;
use serde_json::json;
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

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
    /// The precision a run here would use: the best whose weights fit, or
    /// the smallest there is when none of them do. `null` when the size
    /// could not be read.
    pub fits_at: Option<String>,
    /// The weights do not fit, so they arrive from the disk as the model
    /// runs. Worth saying on a downloaded model and not only on a search
    /// result: by the time it is on the disk the question is no longer
    /// whether to fetch it but what to expect when it starts.
    pub streams: bool,
    /// Streams, and pages in so much a token that it is not worth running.
    /// A dense model over memory, nearly always; a sparse mixture, not.
    pub crawls: bool,
    /// Roughly the bytes a token pages in from the disk, when it streams
    /// and how much of it a token reads is known. See `hub::Fit::per_token`.
    pub disk_per_token: Option<u64>,
    /// A mixture of experts, which is what lets a model over memory stream
    /// at all.
    pub mixture: bool,
    /// `chat` for a language model, `image` for a text-to-image pipeline.
    pub kind: crate::scheduler::Kind,
    /// The diffusers pipeline an image model is, by its `model_index.json`.
    pub pipeline: Option<String>,
    /// What the config says this is, for a row somebody opened. `None`
    /// when there is no config, or when it describes an architecture this
    /// build has no reader for.
    pub detail: Option<Detail>,
}

/// A model past its name and its size: the shape the config describes.
#[derive(serde::Serialize)]
pub struct Detail {
    /// The same line `kvad info` prints.
    pub summary: String,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub params: Option<u64>,
    /// Weights here at each precision, as the search results give it.
    pub memory: Option<Memory>,
    /// A mixture's shape, when it is one.
    pub experts: Option<Experts>,
    /// How the checkpoint stores its weights, when that is not plainly as
    /// floats: `fp8 (e4m3), 128×128 blocks`. Said because it decides the
    /// download and the accuracy and not the memory, which is the part
    /// nobody would guess.
    pub stored_as: Option<String>,
    /// The verdict with the whole config in hand: `kvad search`'s line, and
    /// the numbers behind it. `None` when the size is not known.
    pub fit: Option<Verdict>,
}

/// Whether the model runs here, and out of what, as `hub::Fit` says it.
#[derive(serde::Serialize)]
pub struct Verdict {
    /// The line `kvad search` prints: `fits at q8`, `streams — ...`.
    pub line: String,
    pub precision: Option<String>,
    pub streams: bool,
    pub crawls: bool,
    pub disk_per_token: Option<u64>,
}

impl Verdict {
    fn of(fit: hub::Fit) -> Option<Verdict> {
        (fit != hub::Fit::Unknown).then(|| Verdict {
            line: fit.to_string(),
            precision: fit.precision().map(|p| p.to_string()),
            streams: fit.streams(),
            crawls: fit.crawls(),
            disk_per_token: fit.per_token(),
        })
    }
}

/// How a mixture routes, and what that costs a cache.
#[derive(serde::Serialize)]
pub struct Experts {
    pub count: usize,
    pub per_token: usize,
    /// Experts one token reads across the whole model.
    ///
    /// The number that decides whether an expert cache is worth having:
    /// hold fewer than this and every entry is evicted before its next
    /// use, because a forward pass reads this many before it repeats
    /// itself. Measured exactly on Qwen3-30B-A3B — a cache of 374 experts
    /// hits 1.9% and one of 384, which is this number, hits 33.2%.
    pub working_set: usize,
    /// Layers that route. Not every layer does: DeepSeek's first is an
    /// ordinary MLP, so its working set is 6 × 26 and not 6 × 27.
    pub layers: usize,
    /// What `working_set` experts weigh here at each precision: three
    /// matrices of `hidden × moe_intermediate` apiece. Checked against a
    /// real cache: one Qwen3-30B expert at q8 is 3 × 2048 × 768 × 1.125
    /// bytes, 5.31 MB, exactly.
    pub working_set_bytes: Option<Memory>,
    /// And the whole expert store, for the proportion.
    pub store_bytes: Option<Memory>,
}

/// Read the config and say what it describes.
///
/// Best effort throughout: a model this build cannot run still lists, and
/// a row that cannot say what it is says nothing rather than failing.
fn detail_of(m: &hub::LocalModel) -> Option<Detail> {
    let config = hub::model_file(&m.path, "config.json")?;
    let spec = kvad::model::Spec::from_json(&config).ok()?;
    Some(detail_from(&spec, m.params))
}

/// Weights at each precision this engine offers.
fn memory(params: u64) -> Memory {
    Memory {
        f32: kvad::quant::Precision::F32.weight_bytes(params),
        q8: kvad::quant::Precision::Q8.weight_bytes(params),
        q4: kvad::quant::Precision::Q4.weight_bytes(params),
    }
}

/// How the weights are packed, for the ones this engine reads packed.
///
/// Only fp8 today. A packing it cannot read never gets this far: it is a
/// blocker, and the row says so in its own words.
fn stored_as(config: &kvad::model::Json) -> Option<String> {
    let quant = config.get("quantization_config")?;
    if !kvad::weights::reads_packing(quant) {
        return None;
    }
    let fmt = quant.get("fmt").and_then(|f| f.as_str()).unwrap_or("e4m3");
    // The loader derives the block from the tensors' shapes rather than
    // trusting this; it is only being repeated to a person here.
    let block = quant
        .get("weight_block_size")
        .and_then(|b| b.as_array())
        .map(|b| b.iter().filter_map(|n| n.as_u64()).map(|n| n.to_string()).collect::<Vec<_>>())
        .filter(|b| !b.is_empty());
    Some(match block {
        Some(b) => format!("fp8 ({fmt}), {} blocks", b.join("×")),
        None => format!("fp8 ({fmt})"),
    })
}

/// The shape a spec describes, with whatever the caller knows of its size.
///
/// A downloaded model knows its parameter count from the safetensors index;
/// a search result knows it from the Hub and passes `None`, because the row
/// is already showing it.
fn detail_from(spec: &kvad::model::Spec, params: Option<u64>) -> Detail {
    let experts = kvad::model::ffn::Router::count(&spec.config).and_then(|count| {
        let per_token = spec.config.num(&["num_experts_per_tok"])?;
        let layout = kvad::model::ffn::Layout::read(&spec.config);
        let layers = (0..spec.n_layer).filter(|&i| layout.is_moe(i, count)).count();
        let expert = spec
            .config
            .num(&["moe_intermediate_size"])
            .map(|width| 3 * spec.n_embd as u64 * width as u64);
        Some(Experts {
            count,
            per_token,
            working_set: per_token * layers,
            layers,
            working_set_bytes: expert.map(|e| memory(e * (per_token * layers) as u64)),
            store_bytes: expert.map(|e| memory(e * (count * layers) as u64)),
        })
    });
    let reads = hub::Reads::of(Some(spec.config.value()));
    Detail {
        summary: spec.summary(),
        n_layer: spec.n_layer,
        n_head: spec.n_head,
        n_kv_head: spec.n_kv_head,
        n_embd: spec.n_embd,
        n_ctx: spec.n_ctx,
        vocab_size: spec.vocab_size,
        params,
        memory: params.map(memory),
        experts,
        stored_as: stored_as(&spec.config),
        fit: Verdict::of(hub::fit_of(params, reads)),
    }
}

#[derive(serde::Deserialize)]
pub struct DetailQuery {
    repo: String,
    /// The parameter count the search row already has, so the verdict can
    /// be given without a second request to find it out. It only sizes an
    /// estimate shown back to whoever sent it, so it is taken as given.
    #[serde(default)]
    params: Option<u64>,
}

/// What a model on the Hub is, without downloading it.
///
/// One request for one `config.json`, asked when somebody opens a row
/// rather than for every result of every search: forty rows would be forty
/// round trips, and nearly all of them would be for a model the reader
/// scrolled straight past.
///
/// It is also where the verdict is exact. A search row judges a mixture by
/// the Hub's trimmed config, which drops DeepSeek's expert count; the file
/// has it, so the row that could only say "a mixture" can say how sparse.
pub async fn hub_detail(_: Identity, Query(q): Query<DetailQuery>) -> Result<Json<Detail>, Fail> {
    let repo = q.repo.trim().to_string();
    if repo.is_empty() {
        return Err(Fail::bad("which model?"));
    }
    let params = q.params;
    let detail = blocking(move || {
        let config = hub::remote_config(&repo)
            .ok_or("that repo has no config.json we could read")?;
        let spec = kvad::model::Spec::from_config(kvad::model::Json::new(config))?;
        Ok(detail_from(&spec, params))
    })
    .await?;
    Ok(Json(detail))
}

fn describe(m: &hub::LocalModel, trained: bool) -> Model {
    // Ordered by how specific the answer is. A half-finished download is the
    // likeliest reason a directory here cannot be run, and it used to be
    // answered with the architecture message instead — the GPTQ repo that has
    // only its `model.safetensors.index.json` was being told it was the wrong
    // kind of model rather than an unfinished one.
    // An image pipeline has no `config.json` at its root — each of its
    // models has its own — so it is recognised by its index before the
    // architecture checks below would call it a directory of nothing.
    let pipeline = hub::pipeline(m);
    let blocker = match (m.complete, &m.model_type, m.arch) {
        (false, _, _) if pipeline.is_some() => Some("the download did not finish".to_string()),
        _ if pipeline.is_some() => {
            let p = pipeline.as_deref().unwrap_or_default();
            match (crate::engine::paints(p), cfg!(feature = "gpu")) {
                (false, true) => Some(format!(
                    "`{p}` is not a pipeline this build implements; it implements {}",
                    crate::engine::pipelines()
                )),
                (false, false) => Some("this build has no GPU backend, and images are made on the GPU only".into()),
                // The pipeline's own answer to whether the files it reads are
                // here: SDXL borrows its VAE from another repo, and a
                // Qwen-Image download is 22 files.
                (true, _) => crate::engine::image_weight_bytes(&m.id, crate::engine::preferred_image(p))
                    .is_none()
                    .then(|| "not every file this pipeline reads is downloaded yet".to_string()),
            }
        }
        (false, _, _) => Some(match trained {
            true => "no weights yet — the run was stopped before its first checkpoint".into(),
            false => "the download did not finish".to_string(),
        }),
        (_, None, _) => Some("no config.json, so there is nothing to say what this is".into()),
        // Ahead of the architecture, and for the same reason it comes ahead
        // of it in search: an fp8 repack of a model this engine runs is not
        // an unsupported architecture.
        _ if m.unreadable().is_some() => m.unreadable(),
        // Named by the list the loader dispatches on rather than by a copy of
        // it kept here, which is how this came to be offering GPT-2 and the
        // Llama family long after DeepSeek arrived.
        (_, Some(t), None) => Some(format!(
            "`{t}` is not an architecture this build runs; it runs {}",
            kvad::model::arch::supported()
        )),
        _ => None,
    };
    let fit = m.fit();
    Model {
        id: m.id.clone(),
        arch: m.arch.map(|a| a.to_string()),
        bytes: m.bytes,
        complete: m.complete,
        trained,
        runnable: blocker.is_none(),
        blocker,
        fits_at: fit.precision().map(|p| p.to_string()),
        streams: fit.streams(),
        crawls: fit.crawls(),
        disk_per_token: fit.per_token(),
        mixture: m.reads.mixture(),
        kind: match pipeline {
            Some(_) => crate::scheduler::Kind::Image,
            None => crate::scheduler::Kind::Chat,
        },
        pipeline,
        detail: detail_of(m),
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
    /// The resident used most recently, which is what a page that names no
    /// model talks to. Kept beside `residents` for the pages that show one.
    loaded: Option<crate::scheduler::Loaded>,
    /// Every model in memory, in the order they were loaded.
    residents: Vec<crate::scheduler::Resident>,
    memory: Budgeted,
    queue_depth: usize,
    /// Every backend this build can actually load; see `engine::available`.
    backends: Vec<crate::engine::Choice>,
    /// Which of them a load uses when the request does not say.
    backend: String,
}

/// What the models in memory may take, and what they have left of it.
#[derive(serde::Serialize)]
pub struct Budgeted {
    total: u64,
    left: u64,
    /// Tokens of KV cache each resident is charged for.
    context: usize,
}

impl Budgeted {
    pub fn of(engine: &crate::scheduler::Scheduler) -> Budgeted {
        let budget = engine.budget();
        Budgeted { total: budget.total, left: engine.left(), context: budget.context }
    }
}

#[derive(serde::Serialize)]
pub struct QCache {
    repo: String,
    precision: String,
    bytes: u64,
}

/// The backend a load uses when the request does not name one: what this
/// build prefers for this model, which is `gpu-q8` wherever the GPU backend
/// can run the architecture. See [`crate::engine::preferred`], which is where
/// the measurements are.
///
/// Never a stored setting. There used to be one, and every load from the
/// Models page wrote it, because the page names its picker's backend on
/// every load. So one experiment at `cpu-f32` became the default for every
/// load after it — including the one the service does at startup, which then
/// put 59 GB of f32 weights on a 48 GB machine. A backend a request names is
/// for that load.
///
/// `repo` is the model about to be loaded, where there is one. Its
/// architecture decides whether the GPU is an option, and its size and the
/// way its weights are stored whether the GPU can take it; see [`fitted`].
///
/// Blocking, and for a repo that is not on this disk it may spend a Hub round
/// trip finding out what that repo is — see [`known_about`]. Callers already
/// run it on a blocking thread; the point is that it is on the way to a load,
/// where seconds are the unit.
pub fn default_backend(repo: Option<&str>) -> String {
    let pipeline = repo.and_then(|r| hub::find_local(r)).and_then(|m| hub::pipeline(&m));
    if let Some(p) = pipeline {
        return crate::engine::id_of(crate::engine::preferred_image(&p));
    }
    backend_for(repo, hub::remote_arch)
}

/// [`default_backend`], with the Hub lookup handed in.
///
/// Only so the tests can have both of its answers without a network.
fn backend_for(repo: Option<&str>, ask: impl FnOnce(&str) -> Option<kvad::model::Arch>) -> String {
    let local = repo.and_then(|r| hub::find_local(r).or_else(|| hub::find_trained(r)));
    let preferred = crate::engine::preferred(known_about(repo, local.as_ref(), ask));
    let packed = local.as_ref().is_some_and(packed);
    let usable = kvad::machine::usable_memory_cached();
    crate::engine::id_of(fitted(preferred, local.as_ref(), packed, usable))
}

/// Whether a checkpoint's weights are packed rather than plain floats: fp8,
/// with a `weight_scale_inv` beside every matrix.
///
/// The CPU engine decodes those as it loads. The GPU loader reads none of
/// them, and refuses a checkpoint holding tensors it would not read rather
/// than run a model with pieces missing. Qwen3-0.6B-FP8 is how this was
/// found: its architecture is one the GPU runs, so it was offered the GPU,
/// and the load failed on 196 scales.
fn packed(model: &hub::LocalModel) -> bool {
    hub::model_file(&model.path, "config.json")
        .and_then(|path| kvad::weights::read_json(&path).ok())
        .is_some_and(|config| config.get("quantization_config").is_some())
}

/// The CPU instead of the GPU, for a model the GPU cannot take: packed
/// weights (see [`packed`]), which go to `cpu-q8`, the same arithmetic as the
/// `gpu-q8` they were denied; or a model too big for memory.
///
/// The CPU reads memory-mapped weights, so a model bigger than memory still
/// runs there, from the disk, and a mixture streams its experts through a
/// cache; see `kvad::experts`. The GPU has no such path: its weights are
/// copied into buffers, and a model that does not fit in memory does not
/// load. Qwen3-Next-80B is the case that found this. At `gpu-q8` it is
/// about 85 GB on a 48 GB machine, so the GPU default refused it beside any
/// other model and would have tried to allocate all of it alone. Its expert
/// cache was measured at `cpu-q4`, 11-12 tok/s, and that is where this puts
/// it: the CPU, at the best precision [`hub::fit_of`] says it runs at, which
/// for a model over memory is the smallest.
///
/// Weighed at q8, because that is the GPU default, against usable memory
/// and not the server's budget: whether the GPU can hold a model at all is a
/// fact about the machine, not about what else happens to be loaded.
fn fitted(
    preferred: Backend,
    local: Option<&hub::LocalModel>,
    packed: bool,
    usable: Option<u64>,
) -> Backend {
    if matches!(preferred, Backend::Gpu(_)) && packed {
        return Backend::Cpu(Precision::Q8);
    }
    let (Backend::Gpu(_), Some(model), Some(usable)) = (preferred, local, usable) else {
        return preferred;
    };
    match model.params {
        Some(params) if Precision::Q8.weight_bytes(params) > usable => {
            Backend::Cpu(model.fit().precision().unwrap_or(Precision::Q4))
        }
        _ => preferred,
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
fn known_about(
    repo: Option<&str>,
    local: Option<&hub::LocalModel>,
    ask: impl FnOnce(&str) -> Option<kvad::model::Arch>,
) -> For {
    let Some(repo) = repo else { return For::Anything };
    if let Some(arch) = local.and_then(|m| m.arch) {
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
    let scanned = blocking(move || {
        Ok((
            hub::local_models(),
            hub::trained_models(),
            kvad::qcache::entries(),
            hub::State::active(),
            // No model in hand: the picker is asking what this build
            // prefers in general.
            default_backend(None),
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
        residents: state.engine.residents(),
        memory: Budgeted::of(&state.engine),
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
    /// The precision a run here would use: the best whose weights fit, or
    /// the smallest there is when none of them do. `null` only when the
    /// size is unknown, which `size_known` distinguishes.
    fits_at: Option<String>,
    /// The weights do not fit, so they arrive from the disk as the model
    /// runs. That works — a mixture a fifth over memory generated at 1.5
    /// tok/s here against 24 with everything resident — and it is much
    /// slower than fitting, which is why it is said out loud rather than
    /// left for someone to infer from two numbers.
    streams: bool,
    /// As on a downloaded model. From the trimmed config, so a mixture
    /// whose expert count the Hub dropped has `disk_per_token: null` and
    /// `crawls: false` until its row is opened.
    crawls: bool,
    disk_per_token: Option<u64>,
    mixture: bool,
    /// What this machine has to spend on weights, so the page can say what
    /// the model is being measured against.
    usable_memory: Option<u64>,
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
                memory: m.params.map(memory),
                fits_at: m.fit().precision().map(|p| p.to_string()),
                streams: m.fit().streams(),
                crawls: m.fit().crawls(),
                disk_per_token: m.fit().per_token(),
                mixture: m.reads.mixture(),
                usable_memory: kvad::machine::usable_memory_cached(),
                size_known: m.params.is_some(),
                id: m.id.clone(),
            })
            .collect(),
    ))
}

#[derive(serde::Deserialize)]
pub struct LoadRequest {
    repo: String,
    /// A backend id from the listing, e.g. `cpu-q8`, for this load only.
    /// Omitted, what this build prefers for this model; see
    /// [`default_backend`].
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
            let named = repo.clone();
            blocking(move || Ok(default_backend(Some(&named)))).await?
        }
    };
    let backend = crate::engine::parse(&wanted).ok_or_else(|| {
        Fail::bad(format!(
            "`{wanted}` is not a backend this build can load; it has {}",
            crate::engine::available().iter().map(|c| c.id.clone()).collect::<Vec<_>>().join(", ")
        ))
    })?;

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
            Ok(Err(why)) => sse("error", &json!({
                "error": why.to_string(),
                // Refused for want of memory, rather than tried and failed:
                // the page can say "unload something" instead of "retry".
                "full": matches!(why, crate::scheduler::LoadError::Full(_)),
            })),
            Err(e) => sse("error", &json!({ "error": format!("the load was interrupted: {e}") })),
        };
        let _ = events.send(outcome).await;
    });

    Ok(stream(rx))
}

#[derive(serde::Deserialize, Default)]
pub struct UnloadRequest {
    /// The resident to unload, as `repo@backend` or a bare repo. Absent,
    /// every resident goes.
    #[serde(default)]
    id: Option<String>,
}

/// Unload one resident, or all of them.
///
/// The body is optional, so that the old call with none still means what it
/// did when there was only ever one model: give the memory back.
pub async fn unload(
    _: Admin,
    St(state): St<State>,
    body: Option<Json<UnloadRequest>>,
) -> Result<Json<serde_json::Value>, Fail> {
    let which = match body.and_then(|Json(b)| b.id) {
        Some(id) => Some(
            state
                .engine
                .find(&id)
                .map(|r| r.key)
                .ok_or_else(|| Fail::missing(format!("{id} is not loaded")))?,
        ),
        None => None,
    };
    let gone = state.engine.unload(which).await.map_err(Fail::internal)?;
    Ok(Json(json!({ "unloaded": gone })))
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
    // an engine holding memory-mapped weights whose file is gone.
    if state.engine.residents().iter().any(|r| r.key.repo == id) {
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
            // These cases are about what a model says for itself, not about
            // what it weighs, so the size is deliberately unknown.
            params: None,
            unreadable_as: None,
            reads: hub::Reads::Everything,
        }
    }

    /// A model the GPU cannot hold at q8 goes to the CPU, where it can run
    /// from the disk, and so does one whose weights are packed; anything
    /// else keeps what the build prefers.
    #[test]
    fn a_model_the_gpu_cannot_take_defaults_to_the_cpu() {
        const GB: u64 = 1_000_000_000;
        let gpu = Backend::Gpu(kvad::service::GpuMode::Q8);
        let sized = |params: u64| hub::LocalModel {
            params: Some(params),
            reads: hub::Reads::Share(10.0 / 512.0),
            ..local("Qwen/Qwen3-Next-80B-A3B-Instruct", Some("qwen3_next"), true)
        };

        // Ten trillion parameters is over memory on any machine this runs
        // on, so the precision `fit` picks is the smallest, whatever the
        // machine.
        let huge = sized(10_000 * GB);
        assert_eq!(fitted(gpu, Some(&huge), false, Some(36 * GB)), Backend::Cpu(Precision::Q4));

        // The 80B against this machine's 36 GB: about 85 GB at q8.
        assert!(matches!(fitted(gpu, Some(&sized(80 * GB)), false, Some(36 * GB)), Backend::Cpu(_)));

        // One that fits keeps the GPU.
        assert_eq!(fitted(gpu, Some(&sized(14 * GB)), false, Some(36 * GB)), gpu);
        // A size nobody can read, or a machine nobody can ask, is no reason
        // to move it.
        let unknown = local("a/b", Some("qwen3"), true);
        assert_eq!(fitted(gpu, Some(&unknown), false, Some(36 * GB)), gpu);
        assert_eq!(fitted(gpu, Some(&huge), false, None), gpu);
        // And a CPU preference is left alone.
        let cpu = Backend::Cpu(Precision::Q8);
        assert_eq!(fitted(cpu, Some(&huge), false, Some(36 * GB)), cpu);

        // Packed weights are the CPU's to decode, however small the model,
        // and at q8, which is what the GPU would have run it at.
        let small = sized(GB);
        assert_eq!(fitted(gpu, Some(&small), true, Some(36 * GB)), cpu);
        assert_eq!(fitted(cpu, Some(&small), true, Some(36 * GB)), cpu);
    }

    /// As `local`, for a checkpoint whose weights are packed in a format this
    /// engine has no reader for.
    fn packed(id: &str, model_type: &str, packed_as: &str) -> hub::LocalModel {
        hub::LocalModel { unreadable_as: Some(packed_as.into()), ..local(id, Some(model_type), true) }
    }

    /// A repack that is already on the disk gets the same answer as one in
    /// the search results, and gets it in terms of the packing rather than of
    /// the architecture — `qwen3_moe` is an architecture this build runs, and
    /// saying it is not would send somebody looking for the wrong thing.
    #[test]
    fn a_downloaded_repack_says_it_is_the_packing_that_stops_it() {
        let m = describe(&packed("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "qwen3_moe", "fp8"), false);
        assert!(!m.runnable);
        let why = m.blocker.expect("a blocked model states a reason");
        assert!(why.contains("fp8"), "{why}");
        assert!(!why.contains("not an architecture"), "{why}");

        // An unfinished download is still answered as one. The packing is a
        // fact about weights, and there are none yet.
        let half = hub::LocalModel { complete: false, ..packed("a/b", "qwen3_moe", "fp8") };
        assert!(describe(&half, false).blocker.unwrap().contains("did not finish"));
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

    /// No model named: the build's preference, which is `gpu-q8` in a build
    /// with a GPU backend. Whatever an earlier load named does not come into
    /// it, because nothing remembers it.
    #[test]
    fn the_default_backend_is_the_builds_preference() {
        let prefers = if cfg!(feature = "gpu") { "gpu-q8" } else { "cpu-q8" };
        assert_eq!(default_backend(None), prefers);
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
        let llama = kvad::model::Arch::require("llama");

        // Nobody can say what it is: offline, or no such repo. Guessing the
        // GPU here would be a default that fails to load.
        assert_eq!(backend_for(Some("nobody/has-this-model"), |_| None), "cpu-q8");
        assert!(matches!(known_about(Some("nobody/has-this-model"), None, |_| None), For::Unknown));

        // Not downloaded, and the Hub knows what it is, which is enough to
        // choose properly rather than conservatively.
        assert_eq!(
            backend_for(Some("somebody/a-llama"), |_| Some(llama)),
            crate::engine::id_of(crate::engine::preferred(For::This(llama))),
        );

        assert!(matches!(
            known_about(None, None, |_| panic!("asked the Hub about no model in particular")),
            For::Anything
        ));
    }
}
