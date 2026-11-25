//! Nemotron-H against `transformers`' own `NemotronHForCausalLM`.
//!
//! `tests/nemotron_h/` is a tiny random model, one layer of every kind (Mamba,
//! attention, the latent mixture, a dense MLP, Mamba again), and the logits
//! the reference computed for it in f32. `scripts/nemotron-h-fixture.py`
//! wrote all three; they are committed, so this needs no Python and reads
//! nothing from the network.
//!
//! Qwen3.5's tests compare the engine with a second Rust implementation and
//! then, by hand, with the Python one. Here the reference *is* the fixture,
//! because a misreading — the gate before the norm or after, a `dt` clamp,
//! a scale applied once too often — sits equally in two versions written by
//! the same reader and only the published one settles it.

use kvad::model::{Json, KvCache, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/nemotron_h")
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

fn engine(precision: Precision) -> (Box<dyn Transformer>, Spec) {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("config.json")).unwrap()).unwrap();
    let spec = Spec::from_config(Json::new(config)).unwrap();
    let ckpt = Checkpoint::open(&[fixture().join("model.safetensors")]).unwrap();
    let src = Live::new(&ckpt, precision);
    let model = kvad::model::nemotron_h::Model::load(&src, spec.clone()).unwrap();
    let left = src.unread();
    assert!(left.is_empty(), "the loader left tensors unread: {left:?}");
    (Box::new(model), spec)
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: different lengths");
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = want.iter().map(|x| x.abs()).fold(1f32, f32::max);
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
    assert_eq!(cache.len, tokens.len());
}

/// One token at a time, which is decode: the matrix-vector kernels, and the
/// state carried through the cache rather than within a pass.
#[test]
fn decoding_agrees_with_the_reference_too() {
    let (tokens, want) = expected();
    let (model, spec) = engine(Precision::F32);
    let mut cache = KvCache::new(&spec);
    for (i, &t) in tokens.iter().enumerate() {
        close(&model.forward(t, &mut cache), &want[i], &format!("token {i}"));
    }
}

/// Three tokens, then four: the Mamba states, the convolution's window and
/// the attention rows are all that join the two calls.
#[test]
fn a_prompt_in_two_pieces_is_the_same_prompt() {
    let (tokens, want) = expected();
    let (model, spec) = engine(Precision::F32);
    let mut cache = KvCache::new(&spec);
    model.forward_batch(&tokens[..3], &mut cache);
    let got = model.forward_batch(&tokens[3..], &mut cache);
    close(&got, want.last().unwrap(), "three tokens then four");
}

/// Quantised, the model is a little wrong everywhere and should still be the
/// same model: the same next token at every position.
#[test]
fn quantised_it_still_picks_the_same_tokens() {
    let (tokens, want) = expected();
    let argmax = |r: &[f32]| r.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let (model, spec) = engine(Precision::Q8);
    let mut cache = KvCache::new(&spec);
    let got = model.forward_batch_all(&tokens, &mut cache);
    for (i, row) in got.chunks(spec.vocab_size).enumerate() {
        assert_eq!(argmax(row), argmax(&want[i]), "q8, position {i}");
    }
}

/// Munin 2.0's published `config.json`, the keys that decide its shape: the
/// state and the cache the plan quotes are checked here rather than asserted.
#[test]
fn munin_has_the_state_and_the_cache_the_plan_says() {
    let config = serde_json::json!({
        "model_type": "nemotron_h", "hidden_size": 4096, "num_hidden_layers": 88,
        "hybrid_override_pattern": "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 131072,
        "mamba_num_heads": 128, "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4,
        "n_routed_experts": 512, "num_experts_per_tok": 22, "n_shared_experts": 1, "moe_latent_size": 1024,
        "routed_scaling_factor": 5.0, "norm_topk_prob": true, "max_position_embeddings": 131072,
        "mlp_hidden_act": "relu2", "use_conv_bias": true, "attention_bias": false, "mlp_bias": false,
        "mamba_proj_bias": false, "use_bias": false, "residual_in_fp32": false
    });
    let spec = Spec::from_config(Json::new(config)).unwrap();
    use kvad::model::nemotron_h::{layer_kinds, Kind};
    let kinds = layer_kinds(&spec.config, spec.n_layer).unwrap();
    let count = |k: Kind| kinds.iter().filter(|&&x| x == k).count();
    assert_eq!((count(Kind::Mamba), count(Kind::Attention), count(Kind::Moe)), (40, 8, 40));

    // 3 × 10240 convolution inputs and 128 heads of 64 × 128: 4.3 MB a layer.
    let state: usize = (0..88).map(|i| spec.cache.at(i).state).sum();
    assert_eq!(state, 40 * (3 * 10240 + 128 * 64 * 128));
    assert_eq!(state * 4 / 1_000_000, 172, "MB of Mamba state, whatever the context");
    // Keys and values on 8 layers only: 16 KB a position in f32.
    let per_position: usize = (0..88).map(|i| spec.cache.at(i).bytes(1)).sum::<usize>() - state * 4;
    assert_eq!(per_position, 8 * 2 * 2 * 128 * 4);
    // And the router is told the rule the config leaves out.
    assert_eq!(spec.config.text("scoring_func"), Some("sigmoid"));
    assert_eq!(spec.config.text("topk_method"), Some("noaux_tc"));
}
