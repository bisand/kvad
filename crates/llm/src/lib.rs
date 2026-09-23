//! # kvad
//!
//! Transformer inference written from scratch: download real weights from the
//! HuggingFace Hub and run them with hand-written matrix code. Four
//! architectures, no ML framework.
//!
//! Read the modules in this order:
//!
//! 1. [`tensor`]  — the operations a transformer is built from.
//! 2. [`weights`] — the safetensors format, and how weights get off disk.
//! 3. [`model`]   — the skeleton the architectures share, then
//!    [`model::gpt2`] (simplest, read first), [`model::llama`] (read as a
//!    diff against it) and [`model::deepseek`] (a diff against *that*).
//!    [`model::arch`] is the registry they plug into.
//! 4. [`sampler`] — logits in, one token out.
//! 5. [`chat`]    — why an instruction-tuned model needs exact marker tokens.
//! 6. [`quant`]   — making the weights smaller, and the kernels that read
//!    them, then [`qcache`] — doing that work once instead of every load.
//! 7. [`train`]   — `kvad train`, which is the `nervus` crate's training
//!    loop with the models given names and a home.
//! 8. [`service`] — the same engine driven from another thread, which is what
//!    a terminal UI and a server both need.
//!
//! [`crawl`] and [`retrieve`] are off to one side: no tensors in either, just
//! the way a corpus gets made out of a documentation site and the way the
//! part of it that answers a question gets found again. So are [`client`]
//! and [`daemon`], which are how the command line talks to `kvad-serve` and
//! to the service manager that keeps it running.

/// The `serde_json` this crate was built against.
///
/// [`model::Json::new`] takes a `serde_json::Value`, so anything constructing a
/// [`model::Spec`] by hand needs the crate — and needs *this* copy of it, since
/// two versions of `serde_json` in one build are two unrelated `Value` types
/// that do not convert. Re-exported so a caller cannot pick the wrong one.
pub use serde_json;

pub mod chat;
pub mod client;
pub mod crawl;
pub mod daemon;
pub mod experts;
pub mod hub;
pub mod machine;
pub mod model;
pub mod qcache;
pub mod quant;
pub mod residency;
pub mod retrieve;
pub mod runtime;
pub mod service;
pub mod simd;
pub mod sampler;
pub mod tensor;
pub mod train;
pub mod weights;
