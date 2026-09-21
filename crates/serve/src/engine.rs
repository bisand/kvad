//! Which backends this build can load, and the loader that does it.
//!
//! `kvad` describes `Backend::Gpu` and cannot build one — the GPU crate
//! depends on `kvad`, so the dependency runs the wrong way — which is why
//! `Engine::spawn` takes a loader from whoever spawns it. This is the
//! server's, and it is the only file that knows whether this build has a GPU
//! backend in it.

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
        Llm::load_custom(repo, progress, watch, &mut |files, spec, _| {
            let device = kvad_gpu::model::pick_device(None)?;
            kvad_gpu::model::session(&files.weights, spec, dtype, quant, device)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
