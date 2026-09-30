//! # nervus
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
//! 11. [`checkpoint`] — a trained model on disk, laid out so that `kvad` can run it.
//!     ([`json`] is there because it has to be; it teaches nothing about networks.)
//!
//! Then the same transformer, taught to draw instead of write:
//!
//! 12. [`dit`]   — a diffusion transformer: patches for tokens, no mask, and
//!     the conditioning steering every norm (adaLN-Zero).
//! 13. [`flow`]  — flow matching: the loss it is trained with, and the loop
//!     that draws with it. ([`png`] is [`json`]'s counterpart for pictures.)
//! 14. [`moving`] — clips of two digits bouncing in a box, made up on the
//!     spot, and what the same model needs to draw those: a time axis, and
//!     attention that is cheaper than all of it at once ([`attention::Scope`]).
//!
//! Then run `cargo test -p nervus`: the gradient check in `nn` verifies the
//! hand-derived derivatives against numerically measured ones.

pub mod attention;
pub mod block;
pub mod checkpoint;
pub mod dit;
pub mod embedding;
pub mod flow;
#[cfg(test)]
mod gradcheck;
pub mod json;
pub mod matrix;
pub mod mnist;
pub mod model;
pub mod moving;
pub mod nn;
pub mod norm;
pub mod optim;
pub mod png;
pub mod rng;
pub mod text;
