//! Qwen3.8-Flash-Next (`qwen4_exp`) against transformers, on a tiny random
//! model.
//!
//! `scripts/qwen4-exp-fixtures.py` builds the model with transformers' own
//! class, shaped so that each of the three things this architecture adds is
//! reached — four streams, n-gram embeddings across an EOS and a table in
//! twelve shards, and a block selection with a budget of 8 under a 24-token
//! prompt — packs the experts the way the published checkpoint does, and
//! writes the logits at every position. This reads the same files and must
//! produce the same numbers, a token at a time through its own cache.
//!
//! Ignored, because the fixture is made by Python and not checked in:
//!
//! ```text
//! /opt/homebrew/bin/python3.14 -m venv /tmp/qwen4-venv
//! /tmp/qwen4-venv/bin/pip install torch transformers safetensors
//! /tmp/qwen4-venv/bin/python scripts/qwen4-exp-fixtures.py --out /tmp/qwen4-fx
//! KVAD_QWEN4_FIXTURES=/tmp/qwen4-fx cargo test --release -p kvad --test qwen4_exp -- --ignored
//! ```

use kvad::model::qwen3_5::Streams;
use kvad::model::{KvCache, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var("KVAD_QWEN4_FIXTURES").expect("KVAD_QWEN4_FIXTURES: run scripts/qwen4-exp-fixtures.py"))
}

fn spec() -> Spec {
    Spec::from_json(&fixtures().join("config.json")).unwrap()
}

/// The prompt, and transformers' logits at each of its positions.
fn reference() -> (Vec<u32>, Vec<f32>) {
    let r = Checkpoint::open(&[fixtures().join("reference.safetensors")]).unwrap();
    let (_, words) = r.read_words("tokens").unwrap().unwrap();
    // int64, low word first.
    let tokens = words.chunks(2).map(|w| w[0]).collect();
    (tokens, r.get("logits").unwrap().data)
}

fn load(precision: Precision) -> (Box<dyn Transformer>, Spec) {
    let spec = spec();
    let ckpt = Checkpoint::open(&[fixtures().join("model.safetensors")]).unwrap();
    let src = Live::new(&ckpt, precision);
    let model = kvad::model::qwen3_5::Model::load(&src, spec.clone()).unwrap();
    let left = src.unread();
    assert!(left.is_empty(), "the loader left tensors unread: {left:?}");
    (Box::new(model), spec)
}

#[test]
#[ignore = "needs KVAD_QWEN4_FIXTURES from scripts/qwen4-exp-fixtures.py"]
fn every_position_agrees_with_transformers() {
    let (model, spec) = load(Precision::F32);
    assert!(spec.arch.is("qwen4_exp"));
    let (tokens, want) = reference();
    let vocab = spec.vocab_size;
    let mut cache = KvCache::new(&spec);
    for (i, &t) in tokens.iter().enumerate() {
        let got = model.forward(t, &mut cache);
        let want = &want[i * vocab..(i + 1) * vocab];
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let scale = want.iter().map(|x| x.abs()).fold(1f32, f32::max);
        assert!(worst < 1e-4 * scale, "position {i}: worst disagreement {worst} (scale {scale})");
    }
}

/// q8, as the published model would run: not equal, but near.
///
/// Measured at 13.5% relative error over every position's logits: random
/// weights make a model q8 rounds badly, and the GPU's q8 lands at 9.5–13%.
/// A real mistake does not hide in that: every expert read as the first was
/// 76% on the GPU.
#[test]
#[ignore = "needs KVAD_QWEN4_FIXTURES from scripts/qwen4-exp-fixtures.py"]
fn q8_is_near_transformers() {
    let (model, spec) = load(Precision::Q8);
    let (tokens, want) = reference();
    let mut cache = KvCache::new(&spec);
    let got: Vec<f32> = tokens.iter().flat_map(|&t| model.forward(t, &mut cache)).collect();
    let d: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
    let off = (d / want.iter().map(|b| b * b).sum::<f32>()).sqrt();
    assert!(off < 0.25, "q8 is {:.1}% off transformers", off * 100.0);
}

/// The hash's primes and multipliers are derived, not read; the checkpoint
/// states them as int64 buffers, and the two must be the same numbers.
#[test]
#[ignore = "needs KVAD_QWEN4_FIXTURES from scripts/qwen4-exp-fixtures.py"]
fn the_derived_hash_constants_are_the_checkpoints() {
    let spec = spec();
    let streams = Streams::read(&spec.config, &spec).unwrap().unwrap();
    let ckpt = Checkpoint::open(&[fixtures().join("model.safetensors")]).unwrap();
    let int64 = |name: &str| -> Vec<u64> {
        let (_, w) = ckpt.read_words(&format!("model.language_model.layers.1.ple.ple_embedding.{name}")).unwrap().unwrap();
        w.chunks(2).map(|w| w[0] as u64 | (w[1] as u64) << 32).collect()
    };
    let heads = streams.heads(0);
    assert_eq!(heads.iter().map(|h| h.0).collect::<Vec<_>>(), int64("ngram_heads_vocab_sizes"));
    assert_eq!(heads.iter().map(|h| h.1).collect::<Vec<_>>(), int64("ngram_heads_offsets"));
    assert_eq!(streams.multipliers(0), int64("layer_multipliers"));
}
