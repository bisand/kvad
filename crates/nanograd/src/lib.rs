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
//!
//! Then run `cargo test -p nanograd`: the gradient check in `nn` verifies the
//! hand-derived derivatives against numerically measured ones.

pub mod attention;
pub mod matrix;
pub mod mnist;
pub mod nn;
pub mod rng;
