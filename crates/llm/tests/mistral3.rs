//! Mistral 3 against `transformers`' own `Mistral3ForConditionalGeneration`.
//!
//! `tests/mistral3/` is a tiny random model, vision tower and all, and the
//! logits the reference computed for its language model in f32.
//! `scripts/mistral3-fixture.py` wrote all three; they are committed, so this
//! needs no Python and reads nothing from the network.
//!
//! The fixture is shaped so that both of Ministral 3's changes to Mistral are
//! in play within its 24 tokens: YaRN with an attention factor that is not 1,
//! and a query temperature that steps up at positions 8 and 16, where a real
//! checkpoint's would wait until 16 384.

use kvad::model::{Json, KvCache, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/mistral3")
}

/// The fixture's tokens and, per position, the reference's logits.
fn expected() -> (Vec<u32>, Vec<Vec<f32>>) {
    let text = std::fs::read_to_string(fixture().join("logits.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let tokens = v["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let rows = v["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
        .collect();
    (tokens, rows)
}

fn config() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap()
}

fn engine_with(config: serde_json::Value, precision: Precision) -> (Box<dyn Transformer>, Spec) {
    let spec = Spec::from_config(Json::new(config)).unwrap();
    assert_eq!(spec.arch.to_string(), "mistral3");
    let ckpt = Checkpoint::open(&[fixture().join("model.safetensors")]).unwrap();
    let src = Live::new(&ckpt, precision);
    let model = kvad::model::llama::Model::load(&src, spec.clone()).unwrap();
    let left = src.unread();
    assert!(left.is_empty(), "the loader left tensors unread: {left:?}");
    (Box::new(model), spec)
}

fn engine(precision: Precision) -> (Box<dyn Transformer>, Spec) {
    engine_with(config(), precision)
}

fn worst(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "different lengths");
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = want.iter().map(|x| x.abs()).fold(1f32, f32::max);
    (worst, scale)
}

fn close(got: &[f32], want: &[f32], what: &str) {
    let (worst, scale) = worst(got, want);
    assert!(worst < 1e-4 * scale, "{what}: worst disagreement {worst} (scale {scale})");
}

/// The whole prompt in one pass, every position.
#[test]
fn a_prompt_agrees_with_the_reference_at_every_position() {
    let (tokens, want) = expected();
    let (model, spec) = engine(Precision::F32);
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch_all(&tokens, &mut cache);
    for (i, row) in got.chunks(spec.vocab_size).enumerate() {
        close(row, &want[i], &format!("position {i}"));
    }
}

/// One token at a time, which is decode: the matrix-vector kernels, and the
/// query temperature taken from the cache's length rather than a row index.
#[test]
fn decoding_agrees_with_the_reference_too() {
    let (tokens, want) = expected();
    let (model, spec) = engine(Precision::F32);
    let mut cache = KvCache::new(&spec);
    for (i, &t) in tokens.iter().enumerate() {
        close(&model.forward(t, &mut cache), &want[i], &format!("token {i}"));
    }
}

/// The fixture is not passing by accident: without the query temperature the
/// first eight positions still agree, and the ones past it do not.
#[test]
fn the_query_temperature_is_what_the_later_positions_need() {
    let (tokens, want) = expected();
    let mut config = config();
    config["text_config"]["rope_parameters"]["llama_4_scaling_beta"] = 0.0.into();
    let (model, spec) = engine_with(config, Precision::F32);
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch_all(&tokens, &mut cache);
    let rows: Vec<&[f32]> = got.chunks(spec.vocab_size).collect();
    for (i, row) in rows.iter().enumerate().take(8) {
        close(row, &want[i], &format!("position {i}, before the first step"));
    }
    let (off, scale) = worst(rows[20], &want[20]);
    assert!(off > 1e-2 * scale, "position 20 agreed without the temperature: {off} of {scale}");
}

/// Nor is YaRN idle: with the plain rotation in its place, a position that
/// attends to others disagrees.
#[test]
fn yarn_is_what_the_rotation_needs() {
    let (tokens, want) = expected();
    let mut config = config();
    config["text_config"]["rope_parameters"]["rope_type"] = "default".into();
    config["text_config"]["rope_parameters"]["type"] = "default".into();
    let (model, spec) = engine_with(config, Precision::F32);
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch_all(&tokens, &mut cache);
    let (off, scale) = worst(&got[5 * spec.vocab_size..6 * spec.vocab_size], &want[5]);
    assert!(off > 1e-2 * scale, "position 5 agreed with a plain rotation: {off} of {scale}");
}

/// Quantised, the model is a little wrong everywhere and should still be the
/// same model: within 8% of the logits' scale at every position.
#[test]
fn quantised_it_stays_close() {
    let (tokens, want) = expected();
    let (model, spec) = engine(Precision::Q8);
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch_all(&tokens, &mut cache);
    for (i, row) in got.chunks(spec.vocab_size).enumerate() {
        let (off, scale) = worst(row, &want[i]);
        assert!(off < 0.08 * scale, "q8, position {i}: off by {off} (scale {scale})");
    }
}
