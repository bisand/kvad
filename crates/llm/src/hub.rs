//! Finding models on the HuggingFace Hub, and keeping track of local ones.
//!
//! # Searching
//!
//! The Hub's model index is a plain JSON endpoint — no SDK, no auth for public
//! models:
//!
//! ```text
//! GET https://huggingface.co/api/models?search=smollm&config=true&sort=downloads
//! ```
//!
//! `config=true` is the useful part: it returns each model's `model_type`
//! *before* you download several gigabytes, so the search can say which results
//! this engine can actually run. Of the roughly two million models on the Hub,
//! we handle two architecture families — being honest about that in the listing
//! is better than failing after the download.
//!
//! # Local models
//!
//! `hf-hub` stores downloads in the HuggingFace cache that [`cache_dir`]
//! names, laid out as `models--{owner}--{name}/`. Listing what is on disk is
//! a directory walk; nothing here maintains a database of its own.

use crate::model::Arch;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A search result. Everything here comes from the Hub and is display-only
/// data — nothing in it is trusted or executed.
#[derive(Debug, Clone)]
pub struct HubModel {
    pub id: String,
    pub model_type: Option<String>,
    pub arch: Option<Arch>,
    pub downloads: u64,
    pub likes: u64,
    pub gated: bool,
    /// Heuristic, from the name. Only loading the tokenizer config can
    /// actually confirm it, which `pull` does.
    pub looks_instruct: bool,
    /// Weights, counted by the Hub from the safetensors headers. `None` for a
    /// repo that ships no safetensors — a GGUF mirror, say — which is also a
    /// repo this engine cannot load.
    pub params: Option<u64>,
    /// Bytes to download: the parameter counts multiplied by the width of the
    /// dtype each is stored in. Not the same as what it costs in memory here,
    /// which depends on the precision it is loaded at.
    pub download_bytes: Option<u64>,
    /// How this checkpoint's weights are stored, when that is something
    /// [`crate::weights`] cannot decode: `F8_E4M3` for the fp8
    /// republications, `I32` for an AWQ repack, `nvfp4-pack-quantized` for a
    /// compressed-tensors one.
    ///
    /// Named here rather than discovered at load time, which is where it used
    /// to be discovered: the fp8 check in the DeepSeek loader is right and
    /// arrives seventy gigabytes too late. Both signals ride in the search
    /// response already, so this costs nothing beyond reading them.
    pub unreadable_as: Option<String>,
    /// How much of itself the model reads per token, from the trimmed
    /// config the search response already carries. See [`Reads`].
    pub reads: Reads,
}

impl HubModel {
    pub fn runnable(&self) -> bool {
        self.arch.is_some() && !self.gated && self.unreadable_as.is_none()
    }

    /// One-line reason a model cannot be run, if it cannot.
    pub fn blocker(&self) -> Option<String> {
        if self.gated {
            return Some("gated — needs licence acceptance on huggingface.co".into());
        }
        // Ahead of the architecture, because it is the more specific answer.
        // An fp8 repack of a model this engine runs is not an unsupported
        // architecture, and saying that it is would send somebody looking for
        // the wrong missing thing.
        if let Some(dtype) = &self.unreadable_as {
            return Some(format!(
                "weights are stored as `{dtype}`, which this engine cannot read —                  look for a bf16 or f16 publication of the same model"
            ));
        }
        match &self.model_type {
            None => Some("no config.json".into()),
            Some(t) if self.arch.is_none() => Some(format!("unsupported arch `{t}`")),
            _ => None,
        }
    }

    /// What this model's weights would occupy here, at each precision.
    ///
    /// Weights only — a KV cache needs the context length, and a search result
    /// does not carry one. For a 7B model that understates the real
    /// requirement by a gigabyte or two at a long context, which is worth
    /// knowing and is still the right number to show: it is the part that is
    /// fixed, and the part that decides whether the download is worth starting.
    pub fn memory_at(&self, precision: crate::quant::Precision) -> Option<u64> {
        self.params.map(|p| precision.weight_bytes(p))
    }

    /// The cheapest precision whose weights fit in this machine's memory.
    ///
    /// `None` means either that we do not know the size, or that nothing fits.
    /// [`HubModel::fit`] tells those apart.
    pub fn best_precision(&self) -> Option<crate::quant::Precision> {
        let usable = crate::machine::usable_memory_cached()?;
        let params = self.params?;
        // Largest first, so the answer is the *best* precision that fits
        // rather than merely the smallest.
        crate::quant::Precision::SMALLEST_FIRST
            .into_iter()
            .rev()
            .find(|p| p.weight_bytes(params) <= usable)
    }

    pub fn fit(&self) -> Fit {
        fit_of(self.params, self.reads)
    }
}

/// How much of a model one token reads.
///
/// The number that decides what running from the disk costs, and the one a
/// badge judging by total size cannot see. A dense model reads every weight
/// for every token. A mixture reads `top_k` of each routed layer's experts,
/// and Qwen3-Next-80B reads 10 of 512 -- which is why it generated 10.7
/// tok/s through plain mmap at two fifths over memory, while a dense 70B
/// the same distance over would page in gigabytes a token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reads {
    /// A dense model: all of it, every token.
    Everything,
    /// A mixture reading this fraction of its experts per token.
    Share(f64),
    /// A mixture whose expert count was not given, so how sparse it is
    /// cannot be said. The Hub's trimmed config does this to DeepSeek and
    /// Mixtral: it keeps `num_experts_per_tok` and drops the count, which
    /// they spell `n_routed_experts` and `num_local_experts`.
    SomeOf,
}

impl Reads {
    /// Read from a `config.json`, whole or as the Hub trims it.
    ///
    /// `num_experts_per_tok` is the mark of a mixture, and the one key
    /// every family's trimmed config was seen to keep.
    pub fn of(config: Option<&serde_json::Value>) -> Reads {
        let num = |k: &str| config?.get(k)?.as_u64().filter(|&n| n > 0);
        let Some(top_k) = num("num_experts_per_tok") else { return Reads::Everything };
        match ["n_routed_experts", "num_experts", "num_local_experts"].iter().find_map(|k| num(k)) {
            Some(count) if count > top_k => Reads::Share(top_k as f64 / count as f64),
            // One expert of one is a dense model with extra words.
            Some(_) => Reads::Everything,
            None => Reads::SomeOf,
        }
    }

    pub fn mixture(self) -> bool {
        !matches!(self, Reads::Everything)
    }
}

/// Bytes a token may page in from the disk before the model is better
/// described as crawling than as streaming.
///
/// Set between the only two measurements there are, both through plain mmap
/// on this machine. Qwen3-Next-80B at q4, whose estimate here is 0.29 GB a
/// token, generated 10.7 tok/s. DeepSeek-V2-Lite at f32, estimated at 2.5
/// GB, generated 1.5. The estimates are in the same ratio as the speeds to
/// within the compute both of them also pay, so the line sits between them
/// at a gigabyte: a few tokens a second, by the same arithmetic.
pub const CRAWL: u64 = 1_000_000_000;

/// Whether a model of this size will run here, and out of what.
///
/// Shared by the downloaded list and the search results, because a model
/// does not change size by being on the disk already and the two pages
/// disagreeing about it would be a bug waiting to happen.
pub fn fit_of(params: Option<u64>, reads: Reads) -> Fit {
    let (Some(params), Some(has)) = (params, crate::machine::usable_memory_cached()) else {
        return Fit::Unknown;
    };
    // Largest first, so the answer is the *best* precision that fits rather
    // than merely the smallest.
    match crate::quant::Precision::SMALLEST_FIRST
        .into_iter()
        .rev()
        .find(|p| p.weight_bytes(params) <= has)
    {
        Some(p) => Fit::At(p),
        None => {
            let needs = crate::quant::Precision::SMALLEST_FIRST[0].weight_bytes(params);
            Fit::Slow { needs, has, per_token: from_disk(needs, has, reads) }
        }
    }
}

/// Roughly what one token pages in when `needs` bytes of weights live in
/// `has` bytes of memory.
///
/// The part that does not fit is spread over the experts, and a token reads
/// its share of them, so it finds that share of the missing part missing.
/// That assumes the non-expert weights stay resident -- they are read by
/// every token, so they are the last thing the kernel evicts -- and that
/// routing is uniform, which it is not: skew makes a real cache do better,
/// so this errs slow. For a dense model the share is the whole, and the
/// answer, the part over, is a floor: a pass that scans more than the cache
/// holds can miss on every page of it.
fn from_disk(needs: u64, has: u64, reads: Reads) -> Option<u64> {
    let over = needs.saturating_sub(has);
    match reads {
        Reads::Everything => Some(over),
        Reads::Share(share) => Some((over as f64 * share) as u64),
        Reads::SomeOf => None,
    }
}

/// Parameters in a downloaded checkpoint.
///
/// The safetensors index names the total bytes of weights and the config
/// names the dtype they are stored in; the quotient is the count. A
/// single-file checkpoint has no index, so the file's own size stands in —
/// it overstates by the header, which is kilobytes against gigabytes.
///
/// `packed` gives up instead. A quantised repack's `torch_dtype` describes
/// what it was converted *from*: Qwen's fp8 publication of a 30B model says
/// `bfloat16` over bytes that hold one weight each, and dividing by two
/// called it a 15B model. An AWQ repack is further out still, at eight
/// weights to an `i32`. One wrong number here becomes a wrong memory
/// estimate, a wrong precision and a wrong badge, and none of those are
/// worth having in place of "unknown".
fn local_params(dir: &Path, config: Option<&serde_json::Value>, packed: bool) -> Option<u64> {
    if packed {
        return None;
    }
    // A quantised checkpoint has no single width to divide by. An fp8 one
    // holds its big matrices at a byte a weight, its norms and embeddings at
    // bf16, and an f32 scale for every block on top -- and its `torch_dtype`
    // says `bfloat16` throughout, that being what it was converted from.
    // Dividing the total by any one of those was wrong three ways: it called
    // Qwen3-0.6B-FP8 a 1.06B model against the same weights' 0.75B in bf16.
    // So this one counts instead of dividing.
    if config.and_then(|c| c.get("quantization_config")).is_some() {
        return header_params(dir);
    }
    // `torch_dtype` spells these differently from the Hub API's `safetensors`
    // block, which is why this is not `dtype_bytes`. Absent, assume the
    // half precision that nearly every checkpoint now ships in: guessing
    // wrong by a factor of two is better than saying nothing at all.
    let width = match config.and_then(|j| j.get("torch_dtype")?.as_str()) {
        Some("float64" | "int64") => 8,
        Some("float32" | "int32") => 4,
        Some("float8_e4m3fn" | "float8_e5m2" | "int8" | "uint8") => 1,
        _ => 2,
    };
    Some(index_bytes(dir)? / width)
}

/// Parameters in a checkpoint, counted from the shapes its safetensors
/// headers declare.
///
/// For the mixed-precision checkpoints, where no division gives the right
/// answer. A header is a JSON object at the front of each shard — eight
/// bytes of length, then that much text — so this reads kilobytes per shard
/// and none of the weights.
///
/// The scales are excluded. They are how a quantised checkpoint stores what
/// an unquantised one holds inline, and counting them would report the model
/// as larger for having been made smaller.
fn header_params(dir: &Path) -> Option<u64> {
    let mut total = 0u64;
    let mut found = false;
    for name in snapshot_files(dir) {
        if !name.ends_with(".safetensors") {
            continue;
        }
        let Some(path) = model_file(dir, &name) else { continue };
        let Ok(mut file) = std::fs::File::open(&path) else { continue };
        let mut len = [0u8; 8];
        if std::io::Read::read_exact(&mut file, &mut len).is_err() {
            continue;
        }
        let len = u64::from_le_bytes(len);
        // A header is kilobytes. Anything claiming to be enormous is a file
        // this has no business reading into memory.
        if len == 0 || len > 128 << 20 {
            continue;
        }
        let mut buf = vec![0u8; len as usize];
        if std::io::Read::read_exact(&mut file, &mut buf).is_err() {
            continue;
        }
        let Ok(header) = serde_json::from_slice::<serde_json::Value>(&buf) else { continue };
        let Some(entries) = header.as_object() else { continue };
        for (tensor, meta) in entries {
            if tensor == "__metadata__" || tensor.ends_with("_scale_inv") || tensor.ends_with("_scale") {
                continue;
            }
            let Some(shape) = meta.get("shape").and_then(|s| s.as_array()) else { continue };
            let n = shape.iter().filter_map(|d| d.as_u64()).product::<u64>();
            total += n;
            found = true;
        }
    }
    found.then_some(total)
}

/// Bytes of weights in a checkpoint, as its own index states them.
///
/// A single-file checkpoint has no index, so the file's own size stands in —
/// it overstates by the header, which is kilobytes against gigabytes.
fn index_bytes(dir: &Path) -> Option<u64> {
    match model_file(dir, "model.safetensors.index.json") {
        Some(index) => crate::weights::read_json(&index)
            .ok()?
            .get("metadata")?
            .get("total_size")?
            .as_u64(),
        None => Some(std::fs::metadata(model_file(dir, "model.safetensors")?).ok()?.len()),
    }
}

/// Whether a model will run on this machine, and how cheaply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Fits, at this precision or anything smaller.
    At(crate::quant::Precision),
    /// Runs, but out of the disk rather than out of memory: what the
    /// smallest precision needs, against what this machine has.
    ///
    /// Not "too big", which is what this used to say and what it is not.
    /// A model larger than memory is mapped, and the pages it wants are
    /// fetched as it touches them; it produces the same tokens, slower.
    /// Measured here: DeepSeek-V2-Lite at f32, a fifth larger than memory,
    /// generated at 1.5 tok/s against 24 with everything resident. Slow is
    /// the honest word, and refusing would have been the wrong answer to a
    /// model that works.
    ///
    /// The two numbers are what separate a model a third over from one
    /// eleven times over. Both are slow; only the first is worth running.
    /// See [`crate::residency`] for where that line falls.
    ///
    /// `per_token` is what separates those two now: roughly the bytes a
    /// token pages in (see [`from_disk`]), which depends on how much of the
    /// model a token reads and not on its size. `None` for a mixture whose
    /// sparsity is not known yet.
    Slow { needs: u64, has: u64, per_token: Option<u64> },
    /// The Hub did not say how big it is, or we cannot read this machine's
    /// memory. Saying nothing beats guessing.
    Unknown,
}

impl Fit {
    /// The precision a run here would use: the best whose weights fit, or
    /// the smallest there is when none of them do.
    ///
    /// `None` only when the size is unknown. A model larger than memory
    /// still has an answer, because it still runs — see [`Fit::Slow`] —
    /// and it runs at the smallest precision, that being the one which
    /// asks the disk for the least.
    pub fn precision(&self) -> Option<crate::quant::Precision> {
        match self {
            Fit::At(p) => Some(*p),
            Fit::Slow { .. } => Some(crate::quant::Precision::SMALLEST_FIRST[0]),
            Fit::Unknown => None,
        }
    }

    /// Whether the weights arrive from the disk as the model runs, rather
    /// than being held in memory.
    pub fn streams(&self) -> bool {
        matches!(self, Fit::Slow { .. })
    }

    /// Whether a token pages in so much that the model is not worth
    /// running from the disk. See [`CRAWL`].
    ///
    /// False when it is not known, which is not a promise that it streams
    /// well: a caller wanting to say so asks [`Fit::per_token`] as well.
    pub fn crawls(&self) -> bool {
        self.per_token().is_some_and(|b| b > CRAWL)
    }

    /// Roughly the bytes a token pages in from the disk, when it does.
    pub fn per_token(&self) -> Option<u64> {
        match self {
            Fit::Slow { per_token, .. } => *per_token,
            _ => None,
        }
    }
}

impl std::fmt::Display for Fit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fit::At(p) => write!(f, "fits at {p}"),
            Fit::Slow { needs, has, per_token } => {
                let word = match per_token {
                    Some(b) if *b > CRAWL => "crawls",
                    Some(_) => "streams",
                    None => "slow",
                };
                write!(
                    f,
                    "{word} — {} at {}, have {}",
                    human_bytes(*needs),
                    crate::quant::Precision::SMALLEST_FIRST[0],
                    human_bytes(*has)
                )?;
                match per_token {
                    Some(b) => write!(f, ", ~{} a token from disk", human_bytes(*b)),
                    None => f.write_str(", a mixture of unknown sparsity"),
                }
            }
            Fit::Unknown => f.write_str("?"),
        }
    }
}

/// Bytes a dtype takes per value, as the Hub names them.
fn dtype_bytes(name: &str) -> u64 {
    match name {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        // F8_E4M3, F8_E5M2, I8, U8, BOOL, and the 4-bit types, which the Hub
        // still counts one value per byte.
        _ => 1,
    }
}

/// Whether [`crate::weights`] can turn this dtype into the f32 the engine
/// computes in.
///
/// The list is exactly the one `decode` matches on, and the two have to stay
/// in step: this is a promise made from search-result metadata about what a
/// loader will do seventy gigabytes later, and a promise made from a stale
/// copy of the list would be worse than none.
fn readable_dtype(name: &str) -> bool {
    // `F8_E4M3` is readable because the scales that give it meaning ship in
    // the same file; see `Checkpoint::rescale`. `F8_E5M2` is not, and the
    // two are one letter apart in the Hub's spelling, which is the reason
    // this is a list rather than a prefix test.
    matches!(name, "F32" | "BF16" | "F16" | "F8_E4M3")
}

/// How a checkpoint's weights are stored, when the reader cannot decode
/// them. See [`HubModel::unreadable_as`].
///
/// Two signals, and the order between them is the point.
///
/// The `safetensors` block is evidence about the bytes themselves, so it
/// answers whenever it is there. Whatever holds the most parameters is what
/// the checkpoint *is*: an fp8 repack still ships its norms in bf16, and a
/// bf16 checkpoint may carry a few thousand i64 values of bookkeeping, so
/// asking which dtype dominates gets both right where asking whether
/// anything exotic is present gets the second one wrong.
///
/// `quantization_config` is only a claim in a file, and it is checked second
/// because a config can declare a method meaning "quantise this on load"
/// while shipping perfectly readable bf16 weights. It earns its place on the
/// repos the Hub has not indexed, which have no `safetensors` block at all
/// and would otherwise pass as runnable — an NVFP4 publication among them.
fn unreadable_as(safetensors: Option<&serde_json::Value>, config: Option<&serde_json::Value>) -> Option<String> {
    if let Some(by_dtype) = safetensors.and_then(|s| s.get("parameters")).and_then(|p| p.as_object()) {
        let (dtype, _) = by_dtype
            .iter()
            .filter_map(|(d, n)| Some((d, n.as_u64()?)))
            .max_by_key(|&(_, n)| n)?;
        return (!readable_dtype(dtype)).then(|| dtype.clone());
    }
    quant_format(config?)
}

/// What a `config.json` calls the format its weights are packed in, if it
/// says they are packed at all.
///
/// The same question asked of a search result and of a directory on the
/// disk, which is why it is here rather than inline in either.
fn quant_format(config: &serde_json::Value) -> Option<String> {
    let quant = config.get("quantization_config")?;
    // A packing this build reads is not a blocker. It stops being one the
    // moment `decode` grows a case for it, which is why the question is
    // asked of the reader rather than answered again here.
    if crate::weights::reads_packing(quant) {
        return None;
    }
    // `format` is the specific one where compressed-tensors uses both;
    // `quant_method` is what everything else names itself by.
    let name = ["format", "quant_method"]
        .iter()
        .find_map(|k| quant.get(k)?.as_str())
        .unwrap_or("a quantised format");
    Some(name.to_string())
}

/// Search the Hub, newest-first by download count.
pub fn search(query: &str, limit: usize) -> Res<Vec<HubModel>> {
    // `expand[]` *replaces* the default field set rather than adding to it, so
    // everything this function reads has to be named — including the fields
    // that used to arrive for free. Asking for one more thing and silently
    // losing `config` was the first version of this.
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit={}\
         &filter=text-generation\
         &expand[]=config&expand[]=downloads&expand[]=likes&expand[]=gated\
         &expand[]=safetensors",
        urlencode(query),
        limit.clamp(1, 100)
    );
    let body = ureq::get(&url).call()?.body_mut().read_to_string()?;
    let items: serde_json::Value = serde_json::from_str(&body)?;
    let items = items.as_array().ok_or("unexpected response from the Hub API")?;

    Ok(items
        .iter()
        .map(|m| {
            let id = m.get("modelId").or_else(|| m.get("id")).and_then(|v| v.as_str()).unwrap_or("?");
            let model_type = m
                .get("config")
                .and_then(|c| c.get("model_type"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let lower = id.to_ascii_lowercase();
            let safetensors = m.get("safetensors");
            let params = safetensors.and_then(|s| s.get("total")).and_then(|v| v.as_u64());
            // Each dtype's count times its width. A model stored half in bf16
            // and half in fp8 is neither one nor the other.
            let download_bytes = safetensors
                .and_then(|s| s.get("parameters"))
                .and_then(|p| p.as_object())
                .map(|by_dtype| {
                    by_dtype
                        .iter()
                        .filter_map(|(dtype, n)| Some(n.as_u64()? * dtype_bytes(dtype)))
                        .sum()
                });
            HubModel {
                unreadable_as: unreadable_as(safetensors, m.get("config")),
                reads: Reads::of(m.get("config")),
                arch: model_type.as_deref().and_then(Arch::from_model_type),
                model_type,
                downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                gated: !matches!(m.get("gated"), None | Some(serde_json::Value::Bool(false))),
                looks_instruct: ["instruct", "-it", "chat", "sft"]
                    .iter()
                    .any(|k| lower.contains(k)),
                params,
                download_bytes,
                id: id.to_string(),
            }
        })
        .collect())
}

/// What the Hub says one repo's architecture is, without downloading it.
///
/// The same `config.model_type` [`search`] reads, asked about one model rather
/// than a query: a metadata request, a few hundred bytes, and not a single
/// byte of weights. It exists so that a caller deciding *how* to run a model
/// it has never seen can find out rather than guess — the server choosing a
/// backend for a repo somebody has just named.
///
/// Every way of not knowing is `None`: no network, a repo that does not exist,
/// one that ships no `config.json`, and a `model_type` this build has no
/// architecture for. They are the same answer to the caller, which is "assume
/// nothing", and the caller has no channel to report them on anyway.
///
/// The five-second budget is the point of the timeout. This is a question
/// asked on the way to something slower, and an answer that takes longer than
/// that is worth less than the default it is refining.
pub fn remote_arch(id: &str) -> Option<Arch> {
    let path = repo_path(id)?;
    let agent = ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_connect(Some(std::time::Duration::from_secs(3)))
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build(),
    );
    let url = format!("https://huggingface.co/api/models/{path}?expand[]=config");
    let body = agent.get(&url).call().ok()?.body_mut().read_to_string().ok()?;
    let json: serde_json::Value = serde_json::from_str(&body).ok()?;
    let model_type = json.get("config")?.get("model_type")?.as_str()?;
    Arch::from_model_type(model_type)
}

/// A repo's `config.json`, without downloading a byte of its weights.
///
/// [`remote_arch`] asks the API's `expand[]=config`, which carries the
/// `model_type` and nothing else. The file itself carries the shape — layer
/// count, head count, how many experts and how many of them run — and that
/// is a separate request, which is why this is its own function and why it
/// is asked only when somebody wants the answer.
///
/// Every way of not knowing is `None`, as in [`remote_arch`], and for the
/// same reason: the caller has no channel to report them on.
pub fn remote_config(id: &str) -> Option<serde_json::Value> {
    let path = repo_path(id)?;
    let agent = ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_connect(Some(std::time::Duration::from_secs(3)))
            .timeout_global(Some(std::time::Duration::from_secs(5)))
            .build(),
    );
    let url = format!("https://huggingface.co/{path}/raw/main/config.json");
    let body = agent.get(&url).call().ok()?.body_mut().read_to_string().ok()?;
    serde_json::from_str(&body).ok()
}

/// `id` as a path under `/api/models/`, or `None` if it is not shaped like a
/// repo id at all.
///
/// The check is the point rather than a formality: this id arrives in an HTTP
/// request body, and it is about to be pasted into a URL. A repo id is
/// `owner/name` and both halves are drawn from a small alphabet, so anything
/// carrying a `..`, a second slash, a query string or a space is not one —
/// and a caller asking about `./out/readme` gets `None` here rather than a
/// round trip to be told there is no such repo.
fn repo_path(id: &str) -> Option<String> {
    let (owner, name) = id.split_once('/')?;
    let plain = |s: &str| {
        !s.is_empty()
            && s.len() <= 96
            && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            && s != "."
            && s != ".."
    };
    (plain(owner) && plain(name)).then(|| format!("{owner}/{name}"))
}

/// A model already downloaded to the local cache.
#[derive(Debug, Clone)]
pub struct LocalModel {
    pub id: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// Read from the cached config.json, so this is authoritative rather than
    /// a guess.
    pub arch: Option<Arch>,
    /// What that config called itself, whether or not this build has an
    /// architecture for it.
    ///
    /// Kept beside `arch` for the same reason [`HubModel`] keeps it: the two
    /// ways `arch` can be `None` want different things said about them, and
    /// "there is no config to read" is not "this build cannot run that".
    pub model_type: Option<String>,
    pub complete: bool,
    /// Weights in the checkpoint, so a downloaded model can say whether it
    /// will fit here — which the page listing them wants to show, and
    /// which nothing on disk states outright.
    pub params: Option<u64>,
    /// How the weights are packed, when that is something this engine cannot
    /// read. As [`HubModel::unreadable_as`], for a model already downloaded:
    /// blocking these in search stops somebody starting the download, and
    /// this is what stops the ones that are already here being offered.
    pub unreadable_as: Option<String>,
    /// How much of itself it reads per token, from the whole config on the
    /// disk -- so, unlike a search result, a downloaded mixture always
    /// knows its sparsity.
    pub reads: Reads,
}

impl LocalModel {
    /// Whether this will run from memory or from the disk. See [`Fit`].
    pub fn fit(&self) -> Fit {
        fit_of(self.params, self.reads)
    }

    /// One-line reason this cannot be run, for the reasons visible from the
    /// checkpoint's own metadata. The caller adds the ones only it knows —
    /// a half-finished download, an architecture this build has no reader
    /// for — and asks this for the rest.
    pub fn unreadable(&self) -> Option<String> {
        let packed = self.unreadable_as.as_ref()?;
        Some(format!(
            "weights are packed as `{packed}`, which this engine cannot read —              look for a bf16 or f16 publication of the same model"
        ))
    }
}

/// Root of the HuggingFace cache: where pulled models are, and go.
///
/// The first of these that says anything:
///
/// 1. `HF_HUB_CACHE`, then `$HF_HOME/hub` — the Hub's own variables, which
///    somebody set on purpose and which every other Hub tool honours too;
/// 2. `huggingface/hub` in the data directory, when one was chosen (see
///    [`crate::weights::chosen_data_dir`]), so that moving kvad's data moves
///    its models with it. The same layout as the default, so an existing
///    cache moves there with one `mv`;
/// 3. `~/.cache/huggingface/hub`, which the Hub's Python tools share.
///
/// A data directory that was only defaulted moves nothing, so an existing
/// cache is not stranded by an upgrade.
pub fn cache_dir() -> PathBuf {
    cache_dir_from(
        std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()).map(PathBuf::from),
        std::env::var_os("HF_HOME").filter(|v| !v.is_empty()).map(PathBuf::from),
        crate::weights::chosen_data_dir(),
    )
}

fn cache_dir_from(hub_cache: Option<PathBuf>, hf_home: Option<PathBuf>, data: Option<PathBuf>) -> PathBuf {
    hub_cache
        .or_else(|| hf_home.map(|h| h.join("hub")))
        .or_else(|| data.map(|d| d.join("huggingface/hub")))
        .unwrap_or_else(|| dirs_home().join(".cache/huggingface/hub"))
}

/// A Hub client that keeps its files in [`cache_dir`].
///
/// Said explicitly rather than left to `hf-hub`, which only knows the
/// environment and would download into one place while [`local_models`]
/// looked in another. The token is not moved: it stays where the Hub's own
/// tools read it, `HF_TOKEN` or `$HF_HOME/token`.
pub fn client() -> Result<hf_hub::HFClientSync, hf_hub::HFError> {
    hf_hub::HFClient::builder().cache_dir(cache_dir()).build_sync()
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

/// Everything currently in the cache.
pub fn local_models() -> Vec<LocalModel> {
    let root = cache_dir();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out: Vec<LocalModel> = entries
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            // `models--openai-community--gpt2` -> `openai-community/gpt2`.
            let rest = name.strip_prefix("models--")?;
            let id = rest.replacen("--", "/", 1);
            let path = entry.path();
            let bytes = dir_size(&path);
            let files = snapshot_files(&path);
            let image_weights = denoiser_is_here(&path);
            // Read once. It used to be parsed for the `model_type` here and
            // again inside `local_params`, and there are now three questions
            // to ask it.
            let config = find_config(&path).and_then(|c| crate::weights::read_json(&c).ok());
            let model_type = config
                .as_ref()
                .and_then(|j| j.get("model_type")?.as_str().map(str::to_string));
            let unreadable_as = config.as_ref().and_then(quant_format);
            let params = local_params(&path, config.as_ref(), unreadable_as.is_some());
            Some(LocalModel {
                params,
                unreadable_as,
                reads: Reads::of(config.as_ref()),
                id,
                path,
                bytes,
                arch: model_type.as_deref().and_then(Arch::from_model_type),
                model_type,
                // A cache entry with a config but no weights is a half-finished
                // `info` call, not a usable model.
                complete: files.iter().any(|f| f.ends_with(".safetensors")) || image_weights,
            })
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn find_local(id: &str) -> Option<LocalModel> {
    local_models().into_iter().find(|m| m.id.eq_ignore_ascii_case(id))
}

/// Every model trained on this machine, by name.
///
/// Kept apart from [`local_models`] rather than folded into it, because the
/// two are not the same kind of thing. A downloaded model can be deleted and
/// fetched again; a trained one is the only copy there is. `kvad ls` lists
/// them in a section of their own for that reason, and `kvad rm` says
/// "retraining" rather than "re-download" when asked to delete one.
pub fn trained_models() -> Vec<LocalModel> {
    let root = crate::weights::models_dir();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut out: Vec<LocalModel> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|entry| {
            let path = entry.path();
            let config = crate::weights::read_json(&path.join("config.json")).ok();
            let model_type = config
                .as_ref()
                .and_then(|j| j.get("model_type")?.as_str().map(str::to_string));
            LocalModel {
                id: entry.file_name().to_string_lossy().into_owned(),
                bytes: dir_size(&path),
                // Nothing this engine trains is packed, so this is `None` in
                // practice. Asked anyway rather than assumed, because a
                // directory here is whatever somebody put in it.
                unreadable_as: config.as_ref().and_then(quant_format),
                reads: Reads::of(config.as_ref()),
                // A trained model is laid out flat rather than in the
                // cache's blob-and-snapshot shape, which `model_file`
                // already handles by looking directly first.
                params: local_params(&path, config.as_ref(), false),
                arch: model_type.as_deref().and_then(Arch::from_model_type),
                model_type,
                // A directory left behind by a run that was stopped before
                // its first checkpoint has a tokeniser and no weights.
                complete: path.join("model.safetensors").is_file()
                    && path.join("config.json").is_file(),
                path,
            }
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn find_trained(name: &str) -> Option<LocalModel> {
    trained_models().into_iter().find(|m| m.id == name)
}

/// One named file belonging to a model, wherever that model keeps it.
///
/// A downloaded model is a cache entry whose real names live under
/// `snapshots/<revision>/`; a trained one is a plain directory. A caller that
/// wants the tokenizer config should not have to know which kind it has.
pub fn model_file(model_dir: &Path, name: &str) -> Option<PathBuf> {
    let direct = model_dir.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    for rev in std::fs::read_dir(model_dir.join("snapshots")).ok()?.filter_map(|e| e.ok()) {
        let candidate = rev.path().join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// The diffusers pipeline a model on disk is, by the `_class_name` in its
/// `model_index.json` — `StableDiffusionXLPipeline`, `QwenImagePipeline` —
/// or `None` for anything that has no such file, which is every language
/// model.
///
/// Read here rather than kept on [`LocalModel`], because every listing would
/// then pay for a file that only a server offering images asks about, and
/// that asks about it for the few models it is about to show.
///
/// A video pipeline laid out without an index — LTX-2.5 — is named by its
/// denoiser's file instead; see [`crate::video::LTX_PIPELINE`].
pub fn pipeline(model: &LocalModel) -> Option<String> {
    pipeline_in(&model.path)
}

fn pipeline_in(dir: &Path) -> Option<String> {
    let Some(index) = model_file(dir, "model_index.json") else {
        return model_file(dir, crate::video::LTX_DENOISER).map(|_| crate::video::LTX_PIPELINE.to_string());
    };
    let json = crate::weights::read_json(&index).ok()?;
    json.get("_class_name")?.as_str().map(str::to_string)
}

fn find_config(model_dir: &Path) -> Option<PathBuf> {
    model_file(model_dir, "config.json")
}

/// Whether an image pipeline's denoiser is on the disk in full.
///
/// A pipeline keeps its weights a directory down — `unet/`, `transformer/` —
/// so the check above, which looks for weights beside `config.json`, never
/// finds them. The denoiser is the part that is most of the download and the
/// part without which nothing else matters, so its presence is what
/// "downloaded" means here; whether every file a particular pipeline reads is
/// present is the pipeline's own question, asked when it is offered.
fn denoiser_is_here(model_dir: &Path) -> bool {
    if model_file(model_dir, "model_index.json").is_none() {
        // LTX's denoiser is one file, and one in the snapshot is whole:
        // `hf-hub` downloads to an `.incomplete` file and renames it, and
        // links it into the snapshot after that.
        return model_file(model_dir, crate::video::LTX_DENOISER).is_some();
    }
    let Ok(revisions) = std::fs::read_dir(model_dir.join("snapshots")) else { return false };
    revisions.filter_map(|e| e.ok()).any(|rev| {
        ["unet", "transformer"].iter().any(|part| {
            let dir = rev.path().join(part);
            let Ok(entries) = std::fs::read_dir(&dir) else { return false };
            let names: Vec<String> =
                entries.filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned())).collect();
            match names.iter().find(|n| n.ends_with(".safetensors.index.json")) {
                // Sharded: every shard the index names.
                Some(index) => shard_names_in(&dir.join(index)).is_some_and(|shards| {
                    !shards.is_empty() && shards.iter().all(|s| dir.join(s).is_file())
                }),
                None => names.iter().any(|n| n.ends_with(".safetensors")),
            }
        })
    })
}

/// The files a shard index points at.
fn shard_names_in(index: &Path) -> Option<Vec<String>> {
    let json = crate::weights::read_json(index).ok()?;
    let mut shards: Vec<String> =
        json.get("weight_map")?.as_object()?.values().filter_map(|v| v.as_str().map(String::from)).collect();
    shards.sort();
    shards.dedup();
    Some(shards)
}

/// The human-readable file names in a cache entry.
///
/// The cache stores each download twice over: `blobs/` holds the real files
/// under content hashes, and `snapshots/<revision>/` holds symlinks to them
/// under their proper names. So "is this model actually downloaded?" has to be
/// answered from the snapshot side — the blob side is all hashes.
fn snapshot_files(model_dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(revisions) = std::fs::read_dir(model_dir.join("snapshots")) else {
        return names;
    };
    for rev in revisions.filter_map(|e| e.ok()) {
        let Ok(entries) = std::fs::read_dir(rev.path()) else { continue };
        names.extend(entries.filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned())));
    }
    names
}

/// Bytes on disk, counting each file once.
///
/// `DirEntry::metadata` does *not* follow symlinks, which is exactly what is
/// wanted here: the links under `snapshots/` report as neither file nor
/// directory and are skipped, so only the real blobs are counted. Following
/// them would report every model at twice its true size.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.filter_map(|e| e.ok()) {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total += meta.len();
            }
        }
    }
    total
}

pub fn human_bytes(b: u64) -> String {
    // Up to TB, because a search result can now be a 1.5 TB checkpoint and
    // "1491.9 GB" is a number nobody reads as one and a half terabytes.
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

// ---------------------------------------------------------------------------
// Which model is "active"
// ---------------------------------------------------------------------------

/// The selected model, remembered between runs.
///
/// Deliberately a three-line JSON file rather than anything clever: the models
/// themselves live in the HuggingFace cache, and duplicating that state would
/// only create a second source of truth to keep in sync.
pub struct State;

/// Where settings somebody typed are kept: `$XDG_CONFIG_HOME/kvad`, or
/// `~/.config/kvad`. Config rather than data, because everything in here can
/// be written again from scratch — unlike a trained model. See
/// [`crate::weights::data_dir`] for the other side of that line.
pub fn config_dir() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join(".config"))
        .join("kvad")
}

impl State {
    pub fn path() -> PathBuf {
        config_dir().join("state.json")
    }

    pub fn active() -> Option<String> {
        let json = crate::weights::read_json(&Self::path()).ok()?;
        json.get("active")?.as_str().map(str::to_string)
    }

    pub fn set_active(id: &str) -> Res<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::json!({ "active": id }).to_string())?;
        Ok(())
    }

    pub fn clear() -> Res<()> {
        let path = Self::path();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            other => format!("%{other:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Hub's own variables first, then a data directory somebody chose,
    /// then the default everyone shares.
    #[test]
    fn the_hub_cache_follows_a_chosen_data_directory_and_nothing_less() {
        let p = |s: &str| Some(PathBuf::from(s));
        assert_eq!(cache_dir_from(p("/c"), p("/h"), p("/d")), PathBuf::from("/c"));
        assert_eq!(cache_dir_from(None, p("/h"), p("/d")), PathBuf::from("/h/hub"));
        assert_eq!(cache_dir_from(None, None, p("/d")), PathBuf::from("/d/huggingface/hub"));
        assert_eq!(cache_dir_from(None, None, None), dirs_home().join(".cache/huggingface/hub"));
    }

    /// LTX-2.5 has no `model_index.json`, and is known by its DiT's file in
    /// a snapshot: a video pipeline, downloaded, where a repo with neither
    /// is nothing of the kind.
    #[test]
    fn a_repo_laid_out_as_ltx_is_a_video_pipeline() {
        let dir = std::env::temp_dir().join(format!("kvad-hub-ltx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rev = dir.join("snapshots/abc");
        std::fs::create_dir_all(rev.join("vae")).unwrap();
        assert_eq!(pipeline_in(&dir), None);
        assert!(!denoiser_is_here(&dir));
        let dit = rev.join(crate::video::LTX_DENOISER);
        std::fs::create_dir_all(dit.parent().unwrap()).unwrap();
        std::fs::write(&dit, b"").unwrap();
        assert_eq!(pipeline_in(&dir).as_deref(), Some(crate::video::LTX_PIPELINE));
        assert!(crate::video::is_video_pipeline(&pipeline_in(&dir).unwrap()));
        assert!(denoiser_is_here(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn urlencoding_handles_spaces_and_slashes() {
        assert_eq!(urlencode("smollm2 instruct"), "smollm2+instruct");
        assert_eq!(urlencode("Qwen/Qwen2.5"), "Qwen%2FQwen2.5");
    }

    fn sized(params: Option<u64>) -> HubModel {
        HubModel {
            id: "a/b".into(),
            model_type: Some("llama".into()),
            arch: Some(Arch::require("llama")),
            downloads: 0,
            likes: 0,
            gated: false,
            looks_instruct: false,
            params,
            download_bytes: params.map(|p| p * 2),
            unreadable_as: None,
            reads: Reads::Everything,
        }
    }

    /// The counts are the ones the Hub returns for these repos, so this is a
    /// test of the shape the API actually sends rather than of one invented
    /// to suit the code.
    #[test]
    fn the_dtype_holding_the_weights_decides_whether_they_can_be_read() {
        let st = |json: &str| serde_json::from_str::<serde_json::Value>(json).unwrap();

        // Qwen/Qwen3-Coder-Next-FP8. Readable since `decode` learned e4m3,
        // and this asserted the opposite until it did.
        let fp8 = st(r#"{"parameters":{"BF16":683691264,"F8_E4M3":78995521536}}"#);
        assert_eq!(unreadable_as(Some(&fp8), None), None);

        // One letter apart, and not implemented.
        let e5m2 = st(r#"{"parameters":{"BF16":683691264,"F8_E5M2":78995521536}}"#);
        assert_eq!(unreadable_as(Some(&e5m2), None), Some("F8_E5M2".into()));

        // Qwen/Qwen3-30B-A3B, which loads.
        let bf16 = st(r#"{"parameters":{"BF16":30532122624}}"#);
        assert_eq!(unreadable_as(Some(&bf16), None), None);

        // Bookkeeping in a dtype we cannot decode does not make the
        // checkpoint unreadable, which is the whole reason this asks which
        // dtype dominates rather than whether an odd one is present.
        let mixed = st(r#"{"parameters":{"BF16":30532122624,"I64":4096}}"#);
        assert_eq!(unreadable_as(Some(&mixed), None), None);

        // A repo with neither signal says nothing either way.
        assert_eq!(unreadable_as(None, None), None);
        assert_eq!(unreadable_as(Some(&st("{}")), None), None);
    }

    /// A quantised checkpoint is counted, not divided.
    ///
    /// The regression this exists for: dividing `total_size` by a width
    /// reported Qwen3-0.6B-FP8 as a 1.06B model where the same weights in
    /// bf16 came to 0.75B, because an fp8 file mixes one-byte matrices with
    /// bf16 norms and f32 scales and no single divisor is right for it.
    #[test]
    fn a_quantised_checkpoint_counts_its_shapes_and_skips_its_scales() {
        let dir = std::env::temp_dir().join(format!("kvad-hdr-{}", std::process::id()));
        let snap = dir.join("snapshots").join("abc123");
        std::fs::create_dir_all(&snap).unwrap();

        // 4x8 of weights, 2x4 of scales. Only the first is the model.
        let header = r#"{"__metadata__":{"format":"pt"},
            "w.weight":{"dtype":"F8_E4M3","shape":[4,8],"data_offsets":[0,32]},
            "w.weight_scale_inv":{"dtype":"F32","shape":[2,4],"data_offsets":[32,64]}}"#;
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 64]);
        std::fs::write(snap.join("model.safetensors"), out).unwrap();

        // 32 weights, and not the 8 scales beside them, nor `__metadata__`.
        assert_eq!(header_params(&dir), Some(32));

        // And the whole way through, as `local_models` would ask it.
        let config = serde_json::json!({
            "torch_dtype": "bfloat16",
            "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
        });
        assert_eq!(local_params(&dir, Some(&config), false), Some(32));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// ucbye/Qwen3-Coder-Next-NVFP4-GB10 has no `safetensors` block at all,
    /// and passed as runnable until the config was consulted too.
    #[test]
    fn a_repo_the_hub_has_not_indexed_is_judged_by_its_config() {
        let st = |json: &str| serde_json::from_str::<serde_json::Value>(json).unwrap();

        let nvfp4 = st(
            r#"{"quantization_config":{"format":"nvfp4-pack-quantized",
                "quant_method":"compressed-tensors"}}"#,
        );
        assert_eq!(unreadable_as(None, Some(&nvfp4)), Some("nvfp4-pack-quantized".into()));

        // Named by `quant_method` when that is the only spelling on offer.
        let awq = st(r#"{"quantization_config":{"quant_method":"awq"}}"#);
        assert_eq!(unreadable_as(None, Some(&awq)), Some("awq".into()));

        // The bytes win when we have them. A config may declare a method
        // meaning "quantise this as you load it" over readable weights, and
        // refusing that would block a model that runs.
        let both = st(r#"{"parameters":{"BF16":30532122624}}"#);
        assert_eq!(unreadable_as(Some(&both), Some(&awq)), None);

        // A config with no quantisation block is no evidence of anything.
        assert_eq!(unreadable_as(None, Some(&st(r#"{"model_type":"llama"}"#))), None);
    }

    /// Both halves matter: the button has to go, and the row has to say why
    /// in terms of the dtype rather than of the architecture, which is fine.
    #[test]
    fn a_packing_this_build_cannot_read_is_blocked_before_it_is_downloaded() {
        let mut m = sized(Some(80_000_000_000));
        assert!(m.runnable(), "a bf16 llama is runnable");

        m.unreadable_as = Some("I32".into());
        assert!(!m.runnable());
        let why = m.blocker().expect("a blocked model states a reason");
        assert!(why.contains("I32"), "{why}");
        assert!(!why.contains("unsupported arch"), "{why}");
    }

    /// The quantised formats are not whole bytes, and the arithmetic that says
    /// how big a download will be has to know it.
    #[test]
    fn a_models_size_here_depends_on_the_precision_it_is_loaded_at() {
        use crate::quant::Precision;
        let m = sized(Some(8_000_000_000));

        // 4 bytes, 9 bits and 5 bits per weight.
        assert_eq!(m.memory_at(Precision::F32), Some(32_000_000_000));
        assert_eq!(m.memory_at(Precision::Q8), Some(9_000_000_000));
        assert_eq!(m.memory_at(Precision::Q4), Some(5_000_000_000));
        // Smaller is smaller, at every step.
        assert!(m.memory_at(Precision::Q4) < m.memory_at(Precision::Q8));
        assert!(m.memory_at(Precision::Q8) < m.memory_at(Precision::F32));

        // A repo with no safetensors says nothing rather than zero.
        assert_eq!(sized(None).memory_at(Precision::Q8), None);
        assert_eq!(sized(None).fit(), Fit::Unknown);
    }

    /// The verdict has to be the *best* precision that fits, not the smallest
    /// one that does — otherwise every model would report q4.
    #[test]
    fn the_fit_is_the_best_precision_that_will_run() {
        use crate::quant::Precision;
        let Some(usable) = crate::machine::usable_memory_cached() else { return };

        // A model whose f32 weights alone exceed memory, but whose q8 fit.
        let params = (usable as f64 / Precision::Q8.bytes_per_weight()) as u64;
        assert_eq!(sized(Some(params)).fit(), Fit::At(Precision::Q8));

        // Something that fits comfortably at full precision.
        let tiny = (usable as f64 / 4.0) as u64 / 100;
        assert_eq!(sized(Some(tiny)).fit(), Fit::At(Precision::F32));

        // And something no precision saves.
        let huge = (usable as f64 / Precision::Q4.bytes_per_weight()) as u64 * 4;
        assert!(matches!(sized(Some(huge)).fit(), Fit::Slow { .. }));
        // The numbers are the reason this variant carries anything: a model
        // barely over and a model many times over are both slow, and only
        // one of them is slow enough to still be worth running.
        let Fit::Slow { needs, has, .. } = sized(Some(huge)).fit() else { panic!("expected Slow") };
        assert!(needs > has, "{needs} should not fit in {has}");
        assert_eq!(needs, Precision::SMALLEST_FIRST[0].weight_bytes(huge));
    }

    /// Total size put a dense 70B and a streaming 80B mixture under the same
    /// badge. These are the real parameter counts, at the precision each was
    /// run at or would be, against the 36 GB this machine gives weights --
    /// and the two that were measured have to land on the sides of the line
    /// their speeds put them on.
    #[test]
    fn a_model_over_memory_is_judged_by_what_a_token_reads() {
        use crate::quant::Precision::{F32, Q4};
        let has = 36_000_000_000;
        let cfg = |json: &str| serde_json::from_str::<serde_json::Value>(json).unwrap();

        // The configs as the Hub's search response trims them.
        let next = Reads::of(Some(&cfg(r#"{"model_type":"qwen3_next","num_experts":512,"num_experts_per_tok":10}"#)));
        let llama = Reads::of(Some(&cfg(r#"{"model_type":"llama"}"#)));
        assert_eq!(next, Reads::Share(10.0 / 512.0));
        assert_eq!(llama, Reads::Everything);
        // DeepSeek's count is dropped from the trimmed copy and kept in the
        // file itself, under its own name.
        let trimmed = Reads::of(Some(&cfg(r#"{"model_type":"deepseek_v2","num_experts_per_tok":6}"#)));
        assert_eq!(trimmed, Reads::SomeOf);
        let deepseek = Reads::of(Some(&cfg(r#"{"n_routed_experts":64,"num_experts_per_tok":6}"#)));
        assert_eq!(deepseek, Reads::Share(6.0 / 64.0));
        let mixtral = Reads::of(Some(&cfg(r#"{"num_local_experts":8,"num_experts_per_tok":2}"#)));

        // Measured at 10.7 tok/s through plain mmap.
        let b = from_disk(Q4.weight_bytes(81_324_862_720), has, next).unwrap();
        assert!(b < CRAWL, "Qwen3-Next-80B at q4 pages {b} a token");
        // Measured at 1.5 tok/s.
        let b = from_disk(F32.weight_bytes(15_706_484_224), has, deepseek).unwrap();
        assert!(b > CRAWL, "DeepSeek-V2-Lite at f32 pages {b} a token");
        // Neither measured, and both what the old badge called the 80B's
        // equal. A mixture is not enough to stream: Mixtral reads a quarter
        // of itself a token.
        assert!(from_disk(Q4.weight_bytes(70_553_706_496), has, llama).unwrap() > CRAWL);
        assert!(from_disk(Q4.weight_bytes(140_630_071_296), has, mixtral).unwrap() > CRAWL);
        // And a mixture of unknown sparsity is not given a number.
        assert_eq!(from_disk(Q4.weight_bytes(140_630_071_296), has, trimmed), None);
    }

    /// The Hub reports a mixed-dtype checkpoint as counts per dtype, and the
    /// download is the sum of each times its width.
    #[test]
    fn download_size_counts_each_dtype_at_its_own_width() {
        assert_eq!(dtype_bytes("BF16"), 2);
        assert_eq!(dtype_bytes("F32"), 4);
        assert_eq!(dtype_bytes("F8_E4M3"), 1);
        // Anything unrecognised counts as a byte rather than as nothing, so an
        // unknown dtype understates rather than vanishing.
        assert_eq!(dtype_bytes("SOMETHING_NEW"), 1);
    }

    /// The id in a repo lookup comes out of an HTTP request body and goes
    /// into a URL, so the shape check is a boundary and not a nicety.
    ///
    /// It doubles as the answer to "is this a Hub repo at all": a path, a
    /// bare name, or a trained model asked about here is refused before any
    /// request is made rather than after one comes back empty.
    #[test]
    fn only_something_shaped_like_a_repo_id_is_asked_about() {
        assert_eq!(repo_path("Qwen/Qwen3-0.6B").as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(repo_path("openai-community/gpt2").as_deref(), Some("openai-community/gpt2"));
        for not_one in [
            "",
            "shakespeare",           // trained here, no owner
            "./out/readme",          // a path
            "/Users/someone/model",  // an absolute path
            "owner/name/extra",      // too many segments
            "owner/../../etc",       // traversal
            "owner/na me",           // a space
            "owner/name?expand[]=x", // a query of its own
            "owner/",
            "/name",
        ] {
            assert!(repo_path(not_one).is_none(), "`{not_one}` should not become a URL");
        }
    }

    /// The lookup itself, against the real Hub.
    ///
    /// Ignored by default for the same reason the crawler's is: a test that
    /// reaches the network is testing the network. Run it when this function
    /// changes, or when the Hub's API does.
    #[test]
    #[ignore = "fetches from the network"]
    fn the_hub_can_say_what_a_model_is_without_downloading_it() {
        assert_eq!(remote_arch("Qwen/Qwen3-0.6B"), Arch::from_model_type("qwen3"));
        assert_eq!(remote_arch("openai-community/gpt2"), Arch::from_model_type("gpt2"));
        // No such repo, and no such architecture: both are "assume nothing".
        assert!(remote_arch("nobody/has-this-model-at-all-9f3c").is_none());
        assert!(remote_arch("openai/whisper-tiny").is_none());
    }

    #[test]
    fn byte_formatting() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024 * 1024), "5.0 GB");
        assert_eq!(human_bytes(1536 * 1024 * 1024 * 1024), "1.5 TB");
    }
}
