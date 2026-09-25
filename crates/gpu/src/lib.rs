//! The transformer on the GPU, via candle.
//!
//! Three architectures so far — [`model::GpuLlama`], [`gpt2::GpuGpt2`] and
//! [`deepseek::GpuDeepSeek`] — each implementing [`kvad::model::Session`], so
//! any of them drops into the same runtime, CLI and TUI as the hand-written CPU
//! engine. Only the forward pass differs, and [`model::session`] picks which
//! one from the config.
//!
//! Two modules hold what they share. [`ffn`] is what a block does after it has
//! attended — one MLP, or a router and a hundred of them — which is a choice
//! every architecture here makes independently of how it attends, and is why
//! `qwen3_moe` is the Llama backend rather than a fourth one. [`common`] holds
//! the rest, and that is a smaller list than it looks:
//! two weight layouts behind one `forward`, an embedding table that is dense or
//! quantised, the [`Reader`](common::Reader) every checkpoint is read through —
//! which is what lets a loader be asked, afterwards, what it never looked at —
//! and the [`Loader`](common::Loader) that turns a stored matrix into one of
//! those layouts, going by way of [`qcache`] rather than the quantiser when the
//! quantiser has already been this way.
//!
//! `fused` holds a decode step's small ops as single Metal kernels — the
//! residual add and norm, SiLU and multiply, and everything between the Q/K/V
//! projection and the cache — each with candle's ops behind it for every
//! device and step it does not take.
//!
//! [`uncached`] reads weight files past the page cache, which a model more
//! than half the machine's memory needs (the module says why).
//!
//! [`image`] is the other thing this crate does, and the one thing in the
//! engine with no CPU version: text-to-image pipelines, which are convolution
//! stacks and so live where the convolutions are. They are not sessions —
//! nothing about them is tokens in, logits out — and implement
//! [`kvad::image::Painter`] instead.

mod common;
pub mod deepseek;
mod ffn;
mod fused;
pub mod gpt2;
pub mod image;
pub mod model;
#[cfg(target_os = "macos")]
mod mpp;
pub mod qwen3_5;
pub mod qcache;
mod uncached;
