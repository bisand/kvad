//! The registry: which architectures this build can run, and how one is added.
//!
//! # Why this is not an enum any more
//!
//! It used to be. `enum Arch { Gpt2, Llama }`, a `match` in the loader, and a
//! `match` in every backend that only implements some of them. That is the
//! right shape for two architectures and the wrong shape for five, because
//! every one of those `match`es is a place a new architecture has to be
//! remembered — and the compiler only catches the exhaustive ones.
//!
//! So an architecture is now a value: a name, the `model_type` strings it
//! answers to, and two function pointers. Adding one means writing a module
//! and adding a line to [`registry`]; nothing else in the engine changes, and
//! nothing else in the engine can be forgotten.
//!
//! # "Plugin" here means a Cargo feature, not a shared library
//!
//! Each architecture is behind a feature — `arch-gpt2`, `arch-llama`,
//! `arch-deepseek` — and a build with a feature off does not contain that
//! code at all. `kvad arch` prints what the binary you are holding actually
//! has.
//!
//! It is worth saying plainly why this is *not* `dlopen` and a stable ABI.
//! Rust has no stable ABI, so a real dynamic plugin would mean pinning the
//! compiler version and passing everything across the boundary as C types —
//! including [`Weight`](crate::quant::Weight), which exists to hand `&[i8]`
//! straight to a kernel. Every bit of that machinery would cost the thing
//! this repository is for: being readable end to end. Compile-time plugins
//! give the property actually being asked for — architectures are separable,
//! optional, and cannot reach into each other — and cost nothing.

use super::{Spec, Transformer};
use crate::qcache::Source;
use std::sync::LazyLock;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One architecture, as data.
pub struct Architecture {
    /// What this engine calls it. Appears in `Spec::summary`, in the model
    /// list, and in the GPU backend's "I only do llama" check.
    pub id: &'static str,

    /// The HuggingFace `model_type` values this module implements —
    /// **exactly**, never as a substring.
    ///
    /// The rule was learned the hard way: asking whether a type *contained*
    /// `qwen2` accepted `qwen2_moe`, and `smollm` accepted `smollm3`, which
    /// drops RoPE on every fourth layer. Both would have loaded and been
    /// quietly wrong. A model that refuses to load is a message; a model that
    /// runs and is wrong is a bug report from a confused user.
    pub model_types: &'static [&'static str],

    /// One line, for `kvad arch` and for the error an unknown config gets.
    pub about: &'static str,

    /// Fill in what the shared [`Spec`] cannot know by itself.
    ///
    /// Everything in `Spec` up to this point came from field names that most
    /// of the Hub agrees on. Anything else — the shape of the KV cache above
    /// all — is the architecture's own business, read from
    /// [`Spec::config`](Spec) here.
    pub configure: fn(&mut Spec) -> Res<()>,

    /// Build the thing that runs, given somewhere to get weights from.
    pub load: fn(&dyn Source, Spec) -> Res<Box<dyn Transformer>>,
}

/// A handle to one entry in the registry.
///
/// Copy, comparable, and `'static`: it is a pointer to a table the binary
/// already contains, so passing one around costs nothing and storing one in
/// [`Spec`] does not complicate its lifetime.
#[derive(Clone, Copy)]
pub struct Arch(&'static Architecture);

impl Arch {
    pub fn id(self) -> &'static str {
        self.0.id
    }

    pub fn about(self) -> &'static str {
        self.0.about
    }

    pub fn model_types(self) -> &'static [&'static str] {
        self.0.model_types
    }

    /// Whether this is the architecture called `id`.
    ///
    /// Used by the backends that implement some architectures and not others.
    /// The GPU crate has its own Llama, hand-written against Metal, and no
    /// MLA; `spec.arch.is("llama")` is how it says so without needing to know
    /// what else exists.
    pub fn is(self, id: &str) -> bool {
        self.0.id == id
    }

    pub(crate) fn configure(self, spec: &mut Spec) -> Res<()> {
        (self.0.configure)(spec)
    }

    pub(crate) fn load(self, src: &dyn Source, spec: Spec) -> Res<Box<dyn Transformer>> {
        (self.0.load)(src, spec)
    }

    /// Map a HuggingFace `model_type` (or an `architectures[0]` class name)
    /// onto an implementation, or `None` if this build cannot run it.
    ///
    /// Used both when loading a checkpoint and when searching the Hub, so the
    /// search can say up front which results are worth downloading.
    pub fn from_model_type(model_type: &str) -> Option<Arch> {
        let t = model_type.to_ascii_lowercase();
        // `architectures` entries are class names — `LlamaForCausalLM` — and
        // are the fallback for the few configs with no `model_type`. Reduce
        // them to the same stem rather than keeping two lists.
        let stem = t
            .trim_end_matches("forcausallm")
            .trim_end_matches("lmheadmodel")
            .trim_end_matches("model");
        // Underscores come and go between the two spellings of the same name:
        // the `model_type` is `deepseek_v2` and the class is
        // `DeepseekV2ForCausalLM`. Nothing else distinguishes them, so
        // neither should this.
        let plain = |s: &str| s.replace('_', "");
        let stem = plain(stem);
        registry()
            .iter()
            .copied()
            .find(|a| a.0.model_types.iter().any(|t| plain(t) == stem))
    }

    /// Look one up by the name this engine gives it.
    pub fn by_id(id: &str) -> Option<Arch> {
        registry().iter().copied().find(|a| a.0.id == id)
    }

    /// As [`Arch::by_id`], for callers that are naming a feature they know is
    /// on — tests, and the backends that only have one architecture anyway.
    ///
    /// # Panics
    ///
    /// If this build was compiled without that architecture's feature.
    pub fn require(id: &str) -> Arch {
        Arch::by_id(id)
            .unwrap_or_else(|| panic!("this build has no `{id}`; enable the arch-{id} feature"))
    }
}

/// Everything this binary can run, in the order it will be listed.
///
/// Pushes rather than a `vec![]`, because every line is behind its own `cfg`
/// and a literal cannot be — which is the whole mechanism.
#[allow(clippy::vec_init_then_push)]
pub fn registry() -> &'static [Arch] {
    static REGISTRY: LazyLock<Vec<Arch>> = LazyLock::new(|| {
        let mut all: Vec<Arch> = Vec::new();
        #[cfg(feature = "arch-gpt2")]
        all.push(Arch(&super::gpt2::ARCH));
        #[cfg(feature = "arch-llama")]
        all.push(Arch(&super::llama::ARCH));
        all
    });
    &REGISTRY
}

/// Every `model_type` this build answers to, for an error message.
pub fn supported() -> String {
    let mut types: Vec<&str> = registry()
        .iter()
        .flat_map(|a| a.0.model_types)
        .copied()
        .collect();
    types.sort_unstable();
    types.join(", ")
}

/// Two architectures are the same architecture when they are the same entry.
impl PartialEq for Arch {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for Arch {}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.id)
    }
}

/// Just the name. The table behind it is static and printing it would bury
/// every `Spec` it appears in.
impl std::fmt::Debug for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.id)
    }
}
