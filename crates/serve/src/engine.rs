//! Which backends this build can load, and the loader that does it.
//!
//! `kvad` describes `Backend::Gpu` and cannot build one — the GPU crate
//! depends on `kvad`, so the dependency runs the wrong way — which is why
//! `Engine::spawn` takes a loader from whoever spawns it. This is the
//! server's, and it is the only file that knows whether this build has a GPU
//! backend in it.

use kvad::model::Arch;
use kvad::quant::Precision;
#[cfg(feature = "gpu")]
use kvad::service::Model;
use kvad::service::{Backend, Loader};

/// A loader for whatever this build can run.
pub fn loader() -> Loader {
    #[cfg(feature = "gpu")]
    {
        let mut cpu = kvad::service::cpu_loader();
        Box::new(move |repo, backend, progress, watch| {
            // An image pipeline is not a language model at any precision,
            // and has no CPU implementation at all; see docs/image-plan.md.
            let image = kvad_gpu::image::is_pipeline(repo, watch);
            match backend {
                Backend::Cpu(_) if image => Err(format!(
                    "{repo} makes images, and images are made on the GPU only; load it at a gpu backend"
                )
                .into()),
                Backend::Cpu(_) => cpu(repo, backend, progress, watch),
                Backend::Gpu(mode) if image => gpu::paint(repo, mode, progress, watch),
                Backend::Gpu(mode) => gpu::load(repo, mode, progress, watch).map(Model::from),
            }
        })
    }
    #[cfg(not(feature = "gpu"))]
    kvad::service::cpu_loader()
}

/// The backend to use when nobody has said which.
///
/// The GPU where there is one and it can run this architecture, and the CPU
/// otherwise. Measured on an M5 Pro at q8, medians of interleaved rounds:
///
/// | model | cpu-q8 | gpu-q8 |
/// |---|---|---|
/// | GPT-2 medium | 127.9 tok/s | 292.5 tok/s |
/// | Qwen2.5-0.5B | 117.0 tok/s | 209.6 tok/s |
/// | DeepSeek-V2-Lite | 26.7 tok/s | 39.1 tok/s |
///
/// The last row was run as two separate benchmarks rather than one
/// interleaved pair: at 17 GB a side on a 48 GB machine, the GPU variant
/// evicts the CPU variant's memory-mapped weights between rounds, and the
/// CPU then reads them off the disk while it decodes. Interleaved it reads
/// 13.6 rather than 26.7, which is a measurement of the memory and not of
/// the backend.
///
/// A model whose architecture the GPU backend has no implementation for gets
/// the CPU, because a default that fails to load is worse than one that is
/// slower, and [`kvad_gpu::model::supports`] is the same list the loader
/// dispatches on. [`For`] says how much is known.
///
/// q8 on both sides, so this is the same arithmetic in two places rather
/// than a quantisation trade. bf16 on the GPU is not offered as a default:
/// it is twice the memory and, on this machine, no faster than the CPU's q8.
pub fn preferred(what: For) -> Backend {
    #[cfg(feature = "gpu")]
    if match what {
        // Nobody named a model, so there is nothing to rule the GPU out.
        For::Anything => true,
        For::This(arch) => kvad_gpu::model::supports(arch),
        // Named, and this machine has never read its config. Its architecture
        // is whatever the download turns out to hold, and the GPU backend does
        // not implement all of them.
        For::Unknown => false,
    } {
        return Backend::Gpu(kvad::service::GpuMode::Q8);
    }
    let _ = what;
    Backend::Cpu(Precision::Q8)
}

/// How much is known about the model a backend is being chosen for.
///
/// Three questions, and the bug this replaced was two of them sharing a
/// spelling. The architecture used to arrive as an `Option<Arch>`, and `None`
/// meant both "no model was named" — the picker asking what this build likes
/// in general — and "a model was named that this machine has never seen".
/// Those want opposite answers, so an `Option` could not carry them: the
/// second was getting the first's, which is how naming an undownloaded model
/// came to default to a backend that might not be able to load it.
#[derive(Clone, Copy)]
pub enum For {
    /// No model in particular.
    Anything,
    /// A model on this disk, whose config names this architecture.
    ///
    /// Carried but not read in a build without a GPU backend, where there is
    /// nothing for an architecture to decide: everything gets the CPU.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    This(Arch),
    /// A model named but not downloaded, so there is no config to read.
    Unknown,
}

/// Every backend this build can be asked for, for the picker in the UI.
///
/// The list is what the build can do, not what the enum can name: offering a
/// GPU option that this binary cannot honour would be a picker that lies.
pub fn available() -> Vec<Choice> {
    Backend::ALL
        .iter()
        .filter(|b| cfg!(feature = "gpu") || matches!(b, Backend::Cpu(_)))
        .map(|b| Choice { id: id_of(*b), label: b.to_string() })
        .collect()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Choice {
    /// What the client sends back, e.g. `cpu-q8`.
    pub id: String,
    /// What a person reads, e.g. `cpu q8`.
    pub label: String,
}

/// A backend's wire name: its display name with the space made a hyphen, so
/// there is one spelling and it is safe in a URL.
pub fn id_of(backend: Backend) -> String {
    backend.to_string().replace(' ', "-")
}

/// The backend a wire name means, if this build has it.
pub fn parse(id: &str) -> Option<Backend> {
    Backend::ALL
        .iter()
        .copied()
        .filter(|b| cfg!(feature = "gpu") || matches!(b, Backend::Cpu(_)))
        .find(|b| id_of(*b) == id)
}

/// Whether this build can load an image pipeline of this class.
pub fn paints(pipeline: &str) -> bool {
    #[cfg(feature = "gpu")]
    return kvad_gpu::image::PIPELINES.contains(&pipeline);
    #[cfg(not(feature = "gpu"))]
    {
        let _ = pipeline;
        false
    }
}

/// The pipelines this build implements, for a message.
pub fn pipelines() -> String {
    #[cfg(feature = "gpu")]
    return kvad_gpu::image::PIPELINES.join(", ");
    #[cfg(not(feature = "gpu"))]
    String::from("none")
}

/// The backend an image pipeline gets when nobody has said which.
///
/// SDXL runs in f16 whatever it is asked, so any GPU backend is the same load
/// and `gpu-bf16` is the one whose name does not promise a quantisation that
/// will not happen. Qwen-Image does not fit in bf16 on a machine this server
/// is likely to be on (41 GB of denoiser alone) and gets q8. So does FLUX:
/// its 24 GB of transformer and 9.5 GB of T5 would fit in bf16 on their own,
/// and leave room for nothing else.
pub fn preferred_image(pipeline: &str) -> Backend {
    match pipeline {
        "QwenImagePipeline" | "FluxPipeline" => Backend::Gpu(kvad::service::GpuMode::Q8),
        _ => Backend::Gpu(kvad::service::GpuMode::Bf16),
    }
}

/// What an image pipeline on this disk will take at `backend`, or `None` for
/// anything that is not one (and for everything, in a build with no GPU).
///
/// Asked of the pipeline rather than worked out from the repo's size: each
/// reads only some of its repo, in its own precision. See
/// [`kvad_gpu::image::weight_bytes`].
pub fn image_weight_bytes(repo: &str, backend: Backend) -> Option<u64> {
    #[cfg(feature = "gpu")]
    if let Backend::Gpu(mode) = backend {
        return kvad_gpu::image::weight_bytes(repo, gpu::quant(mode));
    }
    let _ = (repo, backend);
    None
}

/// Bytes one cached key or value number will take for `spec` at `backend`:
/// what the loaded session's `kv_number_bytes` will report, asked before
/// there is one, so that admission charges what the cache will hold.
///
/// f32 on the CPU. On the GPU it is the GPU crate's answer, and it depends
/// on the architecture and the quantisation: 2 for a bf16 model, 2 for a
/// quantised Llama or Qwen3.5 whose cache is f16, 4 for a quantised GPT-2 or
/// DeepSeek. The server's GPU is Metal wherever it is built with one.
pub fn kv_number_bytes(spec: &kvad::model::Spec, backend: Backend) -> usize {
    #[cfg(feature = "gpu")]
    if let Backend::Gpu(mode) = backend {
        let dtype = kvad_gpu::model::parse_dtype(gpu::DTYPE).expect("known dtype");
        return kvad_gpu::model::kv_number_bytes(spec.arch, dtype, gpu::quant(mode), cfg!(target_os = "macos"));
    }
    let _ = (spec, backend);
    std::mem::size_of::<f32>()
}

/// Bytes one number of a recurrent state will take for `spec` at `backend`,
/// as [`kv_number_bytes`] is for the attention cache: what the loaded
/// session's `state_number_bytes` will report.
///
/// f32 on the CPU. On the GPU, the dtype the model computes in: 2 for a
/// bf16 model, 4 for a quantised one, which computes in f32 — whatever its
/// attention cache is kept in.
pub fn state_number_bytes(spec: &kvad::model::Spec, backend: Backend) -> usize {
    #[cfg(feature = "gpu")]
    if let Backend::Gpu(mode) = backend {
        let dtype = kvad_gpu::model::parse_dtype(gpu::DTYPE).expect("known dtype");
        return kvad_gpu::model::state_number_bytes(dtype, gpu::quant(mode));
    }
    let _ = (spec, backend);
    std::mem::size_of::<f32>()
}

#[cfg(feature = "gpu")]
mod gpu {
    use kvad::runtime::Llm;
    use kvad::service::{GpuMode, Model};
    use kvad::weights::Watcher;

    /// The dtype every GPU model is loaded at, as `kvad_gpu` spells it. One
    /// place, because `kv_number_bytes` and `state_number_bytes` have to
    /// agree with `load` about it.
    pub(super) const DTYPE: &str = "bf16";

    /// The quantisation a mode names, as `kvad_gpu` spells it.
    pub(super) fn quant(mode: GpuMode) -> Option<kvad_gpu::model::Quant> {
        let name = match mode {
            GpuMode::Bf16 => "none",
            GpuMode::Q8 => "q8",
            GpuMode::Q4 => "q4",
        };
        kvad_gpu::model::parse_quant(name).expect("known quant")
    }

    pub fn paint(
        repo: &str,
        mode: GpuMode,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Result<Model, Box<dyn std::error::Error>> {
        Ok(Model::Image(kvad_gpu::image::load(repo, quant(mode), progress, watch)?))
    }

    pub fn load(
        repo: &str,
        mode: GpuMode,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Result<Llm, Box<dyn std::error::Error>> {
        let dtype = kvad_gpu::model::parse_dtype(DTYPE).expect("known dtype");
        let quant = quant(mode);
        // Which architecture this is, and whether there is a GPU
        // implementation of it, is the GPU crate's question and is answered
        // in one place: `session` dispatches on the config and names what it
        // has if it cannot. The server kept its own copy of that answer
        // until GPT-2 and DeepSeek arrived on the GPU and the CLI could run
        // models this could not — a list in two places is a list that
        // disagrees.
        let id = kvad::weights::model_id(repo);
        Llm::load_custom(repo, progress, watch, &mut |files, spec, progress| {
            let device = kvad_gpu::model::pick_device(None)?;
            kvad_gpu::model::session(&id, &files.weights, spec, dtype, quant, device, progress)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preference is a backend this build actually has, whichever build
    /// this is, and it never offers the GPU for an architecture the GPU
    /// backend cannot load.
    #[test]
    fn the_preferred_backend_is_one_this_build_can_load() {
        let cases = [
            For::Anything,
            For::This(Arch::require("llama")),
            For::This(Arch::require("deepseek_v3")),
            For::Unknown,
        ];
        for what in cases {
            let id = id_of(preferred(what));
            assert!(parse(&id).is_some(), "preferred `{id}` and could not parse it");
        }
        assert_eq!(
            id_of(preferred(For::This(Arch::require("llama")))),
            if cfg!(feature = "gpu") { "gpu-q8" } else { "cpu-q8" },
        );
        // No GPU implementation of V3, so no GPU default for it — even in a
        // build that has the GPU backend.
        assert_eq!(id_of(preferred(For::This(Arch::require("deepseek_v3")))), "cpu-q8");
        // And knowing nothing about a named model is not the same as being
        // asked about no model: the architecture arrives with the download,
        // and the engine reads one the GPU backend has no implementation for.
        assert_eq!(id_of(preferred(For::Unknown)), "cpu-q8");
    }

    /// Whatever the picker offers, the server must be able to load. The two
    /// come from the same filter so that they cannot disagree.
    #[test]
    fn every_offered_backend_parses_back() {
        let offered = available();
        assert!(!offered.is_empty());
        for choice in &offered {
            assert!(parse(&choice.id).is_some(), "offered `{}` and could not parse it", choice.id);
        }
        assert!(offered.iter().any(|c| c.id == "cpu-q8"));
        assert_eq!(parse("cpu-q8").map(id_of).as_deref(), Some("cpu-q8"));
        assert!(parse("nonsense").is_none());
        assert_eq!(offered.iter().any(|c| c.id.starts_with("gpu-")), cfg!(feature = "gpu"));
    }
}
