//! # kvad
//!
//! Transformer inference written from scratch: download real weights from the
//! HuggingFace Hub and run them with hand-written matrix code. Two
//! architectures, no ML framework.
//!
//! Read the modules in this order:
//!
//! 1. [`tensor`]  — the operations a transformer is built from.
//! 2. [`weights`] — the safetensors format, and how weights get off disk.
//! 3. [`model`]   — the skeleton both architectures share, then
//!    [`model::gpt2`] (simpler, read first) and [`model::llama`] (read as a
//!    diff against it).
//! 4. [`sampler`] — logits in, one token out.
//! 5. [`chat`]    — why an instruction-tuned model needs exact marker tokens.
//! 6. [`quant`]   — making the weights smaller, and the kernels that read
//!    them, then [`qcache`] — doing that work once instead of every load.

pub mod chat;
pub mod hub;
pub mod model;
pub mod qcache;
pub mod quant;
pub mod runtime;
pub mod simd;
pub mod sampler;
pub mod tensor;
pub mod weights;
