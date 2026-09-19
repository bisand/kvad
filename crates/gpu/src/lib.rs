//! The transformer on the GPU, via candle.
//!
//! [`model::GpuLlama`] implements [`kvad::model::Session`], so it drops into
//! the same runtime, CLI and TUI as the hand-written CPU engine. Only the
//! forward pass differs.

pub mod model;
