//! # gpt2
//!
//! GPT-2 inference written from scratch: download real weights from the
//! HuggingFace Hub, and run them with hand-written matrix code.
//!
//! Read the modules in this order:
//!
//! 1. [`tensor`]  — the five operations a transformer is built from.
//! 2. [`weights`] — the safetensors format, and how weights get off disk.
//! 3. [`model`]   — the architecture and the forward pass. This is the core.
//! 4. [`sampler`] — logits in, one token out.

pub mod model;
pub mod sampler;
pub mod tensor;
pub mod weights;
