//! The transformer on the GPU, via candle.
//!
//! [`model::GpuLlama`] implements [`kvad::model::Session`], so it drops into
//! the same runtime, CLI and TUI as the hand-written CPU engine. Only the
//! forward pass differs.
//!
//! [`common`] holds what is not about any one architecture: two weight layouts
//! behind one `forward`, an embedding table that is dense or quantised, and the
//! [`Reader`](common::Reader) every checkpoint is read through — which is what
//! lets a loader be asked, afterwards, what it never looked at.

mod common;
pub mod model;
