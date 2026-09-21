//! The transformer on the GPU, via candle.
//!
//! Two architectures so far — [`model::GpuLlama`] and [`gpt2::GpuGpt2`] — each
//! implementing [`kvad::model::Session`], so either drops into the same
//! runtime, CLI and TUI as the hand-written CPU engine. Only the forward pass
//! differs, and [`model::session`] picks which one from the config.
//!
//! [`common`] holds what they share. That is a smaller list than it looks:
//! two weight layouts behind one `forward`, an embedding table that is dense or
//! quantised, and the [`Reader`](common::Reader) every checkpoint is read
//! through — which is what lets a loader be asked, afterwards, what it never
//! looked at.

mod common;
pub mod deepseek;
pub mod gpt2;
pub mod model;
