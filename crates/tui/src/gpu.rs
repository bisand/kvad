//! The half of the engine's loader that `kvad` cannot supply.
//!
//! `kvad::service` describes [`Backend::Gpu`] and cannot build one: the GPU
//! crate depends on `kvad`, so the dependency runs the wrong way for `kvad`
//! to call into it. Whoever spawns an engine supplies a [`Loader`] instead,
//! and this is the TUI's — CPU backends straight through to the core's own,
//! GPU ones through `kvad_gpu`.

use kvad::runtime::Llm;
use kvad::service::{cpu_loader, Backend, GpuMode, Loader};
use kvad::weights::Watcher;

/// A loader that can build either backend.
pub fn loader() -> Loader {
    let mut cpu = cpu_loader();
    Box::new(move |repo, backend, progress, watch| match backend {
        Backend::Cpu(_) => cpu(repo, backend, progress, watch),
        Backend::Gpu(mode) => load_on_gpu(repo, mode, progress, watch),
    })
}

fn load_on_gpu(
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
    let id = kvad::weights::model_id(repo);
    Llm::load_custom(repo, progress, watch, &mut |files, spec, progress| {
        // Which architectures the GPU backend has is its question, answered in
        // one place. This kept its own copy of the answer, and the copy said
        // "the Llama family only" long after GPT-2 and DeepSeek arrived there;
        // what is local to this screen is only the way out of it.
        if !kvad_gpu::model::supports(spec.arch) {
            return Err(format!(
                "the GPU backend implements {}; this model is {}. \
                 Press p to pick a CPU backend.",
                kvad_gpu::model::supported(),
                spec.arch
            )
            .into());
        }
        let device = kvad_gpu::model::pick_device(None)?;
        kvad_gpu::model::session(&id, &files.weights, spec, dtype, quant, device, progress)
    })
}
