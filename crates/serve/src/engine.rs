//! Which backends this build can load, and the loader that does it.
//!
//! `kvad` describes `Backend::Gpu` and cannot build one — the GPU crate
//! depends on `kvad`, so the dependency runs the wrong way — which is why
//! `Engine::spawn` takes a loader from whoever spawns it. This is the
//! server's, and it is the only file that knows whether this build has a GPU
//! backend in it.

use kvad::model::Arch;
use kvad::quant::Precision;
use kvad::service::{Backend, Loader};

/// A loader for whatever this build can run.
pub fn loader() -> Loader {
    #[cfg(feature = "gpu")]
    {
        let mut cpu = kvad::service::cpu_loader();
        Box::new(move |repo, backend, progress, watch| match backend {
            Backend::Cpu(_) => cpu(repo, backend, progress, watch),
            Backend::Gpu(mode) => gpu::load(repo, mode, progress, watch),
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
/// | DeepSeek-V2-Lite | 13.6 tok/s | 37.7 tok/s |
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

#[cfg(feature = "gpu")]
mod gpu {
    use kvad::runtime::Llm;
    use kvad::service::GpuMode;
    use kvad::weights::Watcher;

    pub fn load(
        repo: &str,
        mode: GpuMode,
        progress: &mut dyn FnMut(&str),
        watch: &Watcher,
    ) -> Result<Llm, Box<dyn std::error::Error>> {
        let dtype = kvad_gpu::model::parse_dtype("bf16").expect("known dtype");
        let quant = match mode {
            GpuMode::Bf16 => None,
            GpuMode::Q8 => kvad_gpu::model::parse_quant("q8").expect("known quant"),
            GpuMode::Q4 => kvad_gpu::model::parse_quant("q4").expect("known quant"),
        };
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
