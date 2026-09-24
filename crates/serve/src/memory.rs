//! What the models this server holds take, and whether one more fits.
//!
//! # Admission, not eviction
//!
//! A model loads if it fits in what is left, and nothing is ever unloaded to
//! make room: unloading is something a person asks for. The case this is
//! for is an agent in the middle of a session, which must not lose its model
//! because some other request named a different one, and then find out from
//! a timeout while it waits tens of seconds to get it back.
//!
//! # Counted, not asked
//!
//! "What is left" is this server's own sum, not the operating system's idea
//! of free memory. Free memory on a machine with a page cache moves with
//! whatever was last read, and would admit a model one minute and refuse the
//! same model the next. A resident is charged three things:
//!
//! * **Its weights**, including weights that are only memory-mapped. The
//!   kernel would let mapped weights be over-committed, and then evict one
//!   model's pages to make room for another's. `engine::preferred` records
//!   what that costs: a CPU model at 13.6 tok/s rather than 26.7, reading
//!   its own weights back off the disk while it decodes.
//! * **Its KV cache at [`Budget::context`] tokens**, or at the model's
//!   context if that is shorter. The cache is not pre-allocated and grows
//!   with the conversation, so a model admitted on what its cache holds now
//!   could run out of room halfway through a conversation. The whole
//!   context was the first plan, and measured against the models on this
//!   machine it is not a number anyone could admit: Qwen3-0.6B declares
//!   40,960 tokens, which at four bytes a float is 9.4 GB of cache for 0.6 GB
//!   of weights.
//! * **For a model that streams its experts, everything that was left** when
//!   it loaded. Its cache is sized from that (see
//!   [`kvad::experts::set_room`]), and the part of it that is not the cache
//!   is the dense weights, the KV cache and the page cache it reads through.
//!
//! # Alone, anything goes
//!
//! With nothing else resident a load is never refused. That is what the
//! server did before it could hold more than one model, and a machine with
//! one model on it has nobody else's memory to protect. A model bigger than
//! the budget is then charged the whole budget, so nothing is admitted
//! beside it.

use kvad::quant::Precision;
use kvad::service::{Backend, GpuMode};

/// How much memory the server may spend on models, and how much context each
/// is charged for.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Budget {
    /// Bytes, across every resident.
    pub total: u64,
    /// Tokens of KV cache charged to each resident.
    pub context: usize,
}

/// The context charged when the config does not say.
///
/// 32k because of agents. An agent's system prompt and tool schemas run to
/// ten thousand tokens or more before the first question, and a session of
/// reading files grows past twenty thousand. A reservation that such a
/// session outgrows in its first hour does not protect anyone.
pub const DEFAULT_CONTEXT: usize = 32_768;

/// The smallest share of memory worth giving a model that streams its
/// experts, before refusing it.
///
/// A guess, and stated as one: the only cache sizes that have been measured
/// are 16 GB and 26 GB (see `kvad::experts::default_budget`). Below this the
/// cache holds too few experts to be worth the memory it takes from the
/// models already resident.
pub const STREAMING_FLOOR: u64 = 8_000_000_000;

impl Budget {
    /// This machine's usable memory, unless the config names a figure.
    pub fn of_machine(gb: Option<f64>, context: Option<usize>) -> Budget {
        let total = match gb {
            Some(gb) => (gb * 1e9) as u64,
            // A machine we cannot ask is given nothing to protect, which is
            // the one-model behaviour: everything is admitted alone and
            // nothing beside it.
            None => kvad::machine::usable_memory_cached().unwrap_or(0),
        };
        Budget { total, context: context.unwrap_or(DEFAULT_CONTEXT).max(1) }
    }
}

/// What a model will take, worked out from its files before it is loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Need {
    /// Weights at the backend asked for. `None` when neither a parameter
    /// count nor a file on disk says.
    pub weights: Option<u64>,
    /// The KV cache at the context charged. Zero when the config cannot be
    /// read, which is a model the load is about to refuse anyway.
    pub kv: u64,
    /// Whether this model can run from the disk through an expert cache: a
    /// mixture, on a CPU backend that reads a quantised cache file.
    pub streams: bool,
}

impl Need {
    pub fn bytes(&self) -> Option<u64> {
        self.weights.map(|w| w + self.kv)
    }
}

/// What the scheduler should do with a load.
#[derive(Debug, Clone, PartialEq)]
pub enum Admission {
    /// Load it, and charge it this much.
    Fits { commit: u64 },
    /// Load it with an expert cache sized from `room`, and charge it all of
    /// `room`.
    Streams { room: u64 },
    /// Do not load it; the sentence says why and what would help.
    Refused(String),
}

/// A model already resident, as admission sees it.
#[derive(Debug, Clone)]
pub struct Held {
    pub id: String,
    pub commit: u64,
}

/// Whether `need` for `id` fits beside what is `held`.
pub fn admit(budget: &Budget, held: &[Held], id: &str, need: &Need) -> Admission {
    let total = budget.total;
    if held.is_empty() {
        return match need.bytes() {
            Some(bytes) if bytes <= total => Admission::Fits { commit: bytes },
            // Over the budget, or of a size nobody can say, and alone. The
            // same load the server did when it held one model, charged the
            // whole budget.
            _ if need.streams => Admission::Streams { room: total },
            _ => Admission::Fits { commit: total },
        };
    }

    let spent: u64 = held.iter().map(|h| h.commit).sum();
    let left = total.saturating_sub(spent);
    let holding = || {
        held.iter()
            .map(|h| format!("{} ({})", h.id, gb(h.commit)))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let Some(bytes) = need.bytes() else {
        return Admission::Refused(format!(
            "{id} is not downloaded, or its size cannot be read from its files, so there is no \
             telling whether it fits beside {}. Unload them first, or download it and try again.",
            holding()
        ));
    };
    if bytes <= left {
        return Admission::Fits { commit: bytes };
    }
    if need.streams && left >= STREAMING_FLOOR {
        return Admission::Streams { room: left };
    }

    let streaming = match need.streams {
        true => format!(
            " It could stream its experts, but that needs at least {} left.",
            gb(STREAMING_FLOOR)
        ),
        false => String::new(),
    };
    Admission::Refused(format!(
        "{id} needs {} ({} of weights and {} of KV cache), and {} is left of {}: {} holding {}.{streaming} \
         Unload something first.",
        gb(bytes),
        gb(bytes - need.kv),
        gb(need.kv),
        gb(left),
        gb(total),
        if held.len() == 1 { "one model is" } else { "these models are" },
        holding(),
    ))
}

/// What `repo` would take at `backend`, from what is on the disk.
///
/// Blocking: it reads a directory listing and a config. The scheduler calls
/// it on its own thread, before the load it is about to decide on.
pub fn need(repo: &str, backend: Backend, context: usize) -> Need {
    // An image pipeline holds no KV cache and reads its repo selectively.
    if let Some(weights) = crate::engine::image_weight_bytes(repo, backend) {
        return Need { weights: Some(weights), kv: 0, streams: false };
    }
    let found = kvad::hub::find_local(repo).or_else(|| kvad::hub::find_trained(repo));
    let Some(local) = found.or_else(|| at_path(repo)) else {
        return Need::default();
    };

    // A quantised cache file already written is what a CPU load will map,
    // byte for byte, so it wins over arithmetic on the parameter count.
    let written = match backend {
        Backend::Cpu(p @ (Precision::Q8 | Precision::Q4)) => {
            std::fs::metadata(kvad::qcache::path_for(repo, p)).ok().map(|m| m.len())
        }
        _ => None,
    };
    let from_params = local.params.map(|n| match backend {
        Backend::Cpu(p) => p.weight_bytes(n),
        Backend::Gpu(GpuMode::Bf16) => n * 2,
        Backend::Gpu(GpuMode::Q8) => Precision::Q8.weight_bytes(n),
        Backend::Gpu(GpuMode::Q4) => Precision::Q4.weight_bytes(n),
    });
    let weights = written.or(from_params).or(Some(local.bytes).filter(|&b| b > 0));

    let kv = kvad::hub::model_file(&local.path, "config.json")
        .and_then(|config| kvad::model::Spec::from_json(&config).ok())
        .map(|spec| kv_bytes(&spec, context, crate::engine::kv_number_bytes(&spec, backend)))
        .unwrap_or(0);

    let streams =
        local.reads.mixture() && matches!(backend, Backend::Cpu(Precision::Q8 | Precision::Q4));
    Need { weights, kv, streams }
}

/// A model named by the directory it is in, which the engine loads as
/// readily as a repo id and the Hub's listings do not know about.
///
/// Sized from its files rather than its parameter count, which nothing here
/// has read. That overstates a quantised load by the difference between the
/// checkpoint's floats and the cache file's, which is the safe way round to
/// be wrong.
fn at_path(repo: &str) -> Option<kvad::hub::LocalModel> {
    let path = std::path::Path::new(repo);
    let files = std::fs::read_dir(path).ok()?;
    let bytes = files
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    let config = kvad::weights::read_json(&path.join("config.json")).ok();
    Some(kvad::hub::LocalModel {
        id: repo.to_string(),
        path: path.to_path_buf(),
        bytes,
        arch: None,
        model_type: None,
        complete: true,
        params: None,
        unreadable_as: None,
        reads: kvad::hub::Reads::of(config.as_ref()),
    })
}

/// The KV cache a resident is charged for, with each cached number `number`
/// bytes wide: [`crate::engine::kv_number_bytes`].
pub fn kv_bytes(spec: &kvad::model::Spec, context: usize, number: usize) -> u64 {
    spec.cache.bytes_as(spec.n_layer, spec.n_ctx.min(context), number) as u64
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn budget(total: u64) -> Budget {
        Budget { total, context: DEFAULT_CONTEXT }
    }

    fn need(weights: u64, kv: u64, streams: bool) -> Need {
        Need { weights: Some(weights), kv, streams }
    }

    fn held(id: &str, commit: u64) -> Held {
        Held { id: id.into(), commit }
    }

    /// Alone, nothing is refused: a model that fits is charged what it
    /// takes, one that does not is charged everything, and a mixture that
    /// does not is given the whole budget to stream through.
    #[test]
    fn alone_everything_is_admitted() {
        let b = budget(36 * GB);
        assert_eq!(admit(&b, &[], "a", &need(16 * GB, 4 * GB, false)), Admission::Fits { commit: 20 * GB });
        assert_eq!(admit(&b, &[], "a", &need(34 * GB, 6 * GB, false)), Admission::Fits { commit: 36 * GB });
        assert_eq!(admit(&b, &[], "a", &need(50 * GB, 2 * GB, true)), Admission::Streams { room: 36 * GB });
        // A size nobody can read is what the server has always loaded.
        assert_eq!(admit(&b, &[], "a", &Need::default()), Admission::Fits { commit: 36 * GB });
    }

    /// Beside others, a model fits in what is left or is refused, and the
    /// refusal names what is holding the memory.
    #[test]
    fn beside_others_it_fits_in_what_is_left_or_is_refused() {
        let b = budget(36 * GB);
        let others = [held("big@gpu-q8", 20 * GB), held("small@cpu-q8", 4 * GB)];
        assert_eq!(admit(&b, &others, "a", &need(10 * GB, 2 * GB, false)), Admission::Fits { commit: 12 * GB });

        let Admission::Refused(why) = admit(&b, &others, "a", &need(10 * GB, 3 * GB, false)) else {
            panic!("13 GB was admitted into 12");
        };
        assert!(why.contains("big@gpu-q8 (20.0 GB)"), "{why}");
        assert!(why.contains("small@cpu-q8 (4.0 GB)"), "{why}");
        assert!(why.contains("12.0 GB is left of 36.0 GB"), "{why}");
        assert!(!why.contains("stream"), "a dense model was offered streaming: {why}");
    }

    /// A mixture that does not fit streams through what is left, if that is
    /// enough to be worth it, and is charged all of it.
    #[test]
    fn a_mixture_streams_through_what_is_left() {
        let b = budget(36 * GB);
        assert_eq!(
            admit(&b, &[held("x", 16 * GB)], "moe", &need(50 * GB, 2 * GB, true)),
            Admission::Streams { room: 20 * GB }
        );
        let Admission::Refused(why) = admit(&b, &[held("x", 30 * GB)], "moe", &need(50 * GB, 2 * GB, true))
        else {
            panic!("6 GB was offered to a model streaming 50");
        };
        assert!(why.contains("could stream its experts"), "{why}");

        // Once it is in, nothing else is.
        let after = [held("x", 16 * GB), held("moe", 20 * GB)];
        assert!(matches!(admit(&b, &after, "y", &need(GB / 2, 0, false)), Admission::Refused(_)));
    }

    /// A model whose size cannot be read is not admitted beside anything,
    /// because there is no saying whether it fits.
    #[test]
    fn an_unknown_size_is_refused_beside_others() {
        let b = budget(36 * GB);
        let got = admit(&b, &[held("x", GB)], "nobody/nothing", &Need::default());
        assert!(matches!(got, Admission::Refused(ref why) if why.contains("not downloaded")), "{got:?}");
    }

    /// The KV cache is charged at the budget's context, not the model's:
    /// measured against Qwen3-0.6B's config, the model's own would charge
    /// 9.4 GB.
    #[test]
    fn the_cache_is_charged_at_the_budget_context() {
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 28,
            "max_position_embeddings": 40960,
            "vocab_size": 151936,
            "intermediate_size": 3072,
        });
        let spec = kvad::model::Spec::from_config(kvad::model::Json::new(config)).unwrap();
        // 28 layers, a key and a value of 8 heads by 128, four bytes each.
        let per_token = 28 * 2 * 8 * 128 * 4;
        assert_eq!(kv_bytes(&spec, 40_960, 4), 40_960 * per_token);
        assert_eq!(kv_bytes(&spec, DEFAULT_CONTEXT, 4), DEFAULT_CONTEXT as u64 * per_token);
        // A model whose context is shorter than the budget's is charged its own.
        assert_eq!(kv_bytes(&spec, 1 << 20, 4), 40_960 * per_token);
    }
}
