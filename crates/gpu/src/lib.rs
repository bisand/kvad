//! The transformer on the GPU, via candle.
//!
//! Three architectures so far — [`model::GpuLlama`], [`gpt2::GpuGpt2`] and
//! [`deepseek::GpuDeepSeek`] — each implementing [`kvad::model::Session`], so
//! any of them drops into the same runtime, CLI and TUI as the hand-written CPU
//! engine. Only the forward pass differs, and [`model::session`] picks which
//! one from the config.
//!
//! [`common`] holds what they share. That is a smaller list than it looks:
//! two weight layouts behind one `forward`, an embedding table that is dense or
//! quantised, the [`Reader`](common::Reader) every checkpoint is read through —
//! which is what lets a loader be asked, afterwards, what it never looked at —
//! and the [`Loader`](common::Loader) that turns a stored matrix into one of
//! those layouts, going by way of [`qcache`] rather than the quantiser when the
//! quantiser has already been this way.

mod common;
pub mod deepseek;
pub mod gpt2;
pub mod model;
pub mod qcache;
