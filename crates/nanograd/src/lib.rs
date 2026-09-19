//! # nanograd
//!
//! A neural network, its training loop, and backpropagation, written from
//! scratch with no dependencies at all.
//!
//! Read the modules in this order:
//!
//! 1. [`matrix`] — the three matrix products everything else is built from.
//! 2. [`nn`]     — layers, the loss, and the chain rule. This is the core.
//! 3. [`mnist`]  — reading the dataset off disk.
//! 4. [`attention`] — the first piece of a transformer, built from 1 and 2.
//! 5. [`norm`]   — LayerNorm and RMSNorm, and why their gradients look as they do.
//! 6. [`embedding`] — token ids to vectors: a `Linear` layer with the zeros skipped.
//! 7. [`block`]  — the residual connection, and the transformer block made of two.
//! 8. [`model`]  — all of it assembled into a GPT that predicts the next token.
//! 9. [`optim`]  — AdamW, and why SGD is not enough for a transformer.
//! 10. [`text`]  — from a text file to training windows, and from a model to text.
//!
//! Then run `cargo test -p nanograd`: the gradient check in `nn` verifies the
//! hand-derived derivatives against numerically measured ones.

pub mod attention;
pub mod block;
pub mod embedding;
#[cfg(test)]
mod gradcheck;
pub mod matrix;
pub mod mnist;
pub mod model;
pub mod nn;
pub mod norm;
pub mod optim;
pub mod rng;
pub mod text;
