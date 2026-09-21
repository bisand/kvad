//! What a loader does *not* read.
//!
//! The Qwen3 bug that prompted this was not a wrong answer but a question never
//! asked: the GPU backend never looked for `self_attn.q_norm.weight`, so it ran
//! a Llama forward pass over a Qwen3 model at full speed and produced nonsense.
//! Nothing complained, because a checkpoint has no opinion about tensors nobody
//! wants and a loader asks only for what it already knows about.
//!
//! [`Live::unread`] closes that by subtracting what was asked for from what the
//! file holds. These tests are about the two ways that can go wrong: missing a
//! real omission, and crying about a tensor that is meant to be left alone.
//!
//! The architectures are covered together because the check is one mechanism
//! underneath all three, and because the two things it has to tell apart —
//! GPT-2's stored causal mask and a tied model's duplicate output head — live
//! in different families.

use kvad::model::{Json, Spec};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// A tensor to write: `[rows, cols]`, or `[cols]` when `rows` is 1.
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

/// Values that are not all the same, so nothing can pass by accident, and not
/// random, so a failure reproduces.
fn mat(rows: usize, cols: usize) -> Mat {
    let data = (0..rows * cols).map(|i| ((i % 17) as f32 - 8.0) / 64.0).collect();
    Mat { rows, cols, data }
}

fn vec1(n: usize) -> Mat {
    mat(1, n)
}

/// The smallest safetensors writer that will do: a JSON header, its length, and
/// the arrays. Writing one here keeps this test free of anything that could
/// also be wrong in the engine.
fn write_safetensors(path: &PathBuf, tensors: &BTreeMap<String, Mat>) {
    let mut header = String::from("{");
    let mut blob: Vec<u8> = Vec::new();
    for (i, (name, m)) in tensors.iter().enumerate() {
        let start = blob.len();
        for v in &m.data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let shape = match m.rows {
            1 => format!("[{}]", m.cols),
            r => format!("[{r},{}]", m.cols),
        };
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape},\"data_offsets\":[{start},{}]}}",
            blob.len()
        ));
    }
    header.push('}');
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&blob);
    std::fs::write(path, out).unwrap();
}

/// A one-layer Qwen3: grouped-query attention, SwiGLU, and the per-head Q/K
/// norms that started all this.
fn qwen3(tie: bool) -> (serde_json::Value, BTreeMap<String, Mat>) {
    let (e, hd, heads, kv, inter, vocab) = (32usize, 16usize, 4usize, 2usize, 64usize, 24usize);
    let config = serde_json::json!({
        "model_type": "qwen3",
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "hidden_size": e,
        "head_dim": hd,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tie,
    });

    let mut t = BTreeMap::new();
    t.insert("model.embed_tokens.weight".into(), mat(vocab, e));
    t.insert("model.norm.weight".into(), vec1(e));
    let p = "model.layers.0";
    t.insert(format!("{p}.input_layernorm.weight"), vec1(e));
    t.insert(format!("{p}.post_attention_layernorm.weight"), vec1(e));
    t.insert(format!("{p}.self_attn.q_proj.weight"), mat(heads * hd, e));
    t.insert(format!("{p}.self_attn.k_proj.weight"), mat(kv * hd, e));
    t.insert(format!("{p}.self_attn.v_proj.weight"), mat(kv * hd, e));
    t.insert(format!("{p}.self_attn.o_proj.weight"), mat(e, heads * hd));
    t.insert(format!("{p}.self_attn.q_norm.weight"), vec1(hd));
    t.insert(format!("{p}.self_attn.k_norm.weight"), vec1(hd));
    t.insert(format!("{p}.mlp.gate_proj.weight"), mat(inter, e));
    t.insert(format!("{p}.mlp.up_proj.weight"), mat(inter, e));
    t.insert(format!("{p}.mlp.down_proj.weight"), mat(e, inter));
    (config, t)
}

/// A one-block GPT-2, including the causal mask its exports store as a weight.
fn gpt2() -> (serde_json::Value, BTreeMap<String, Mat>) {
    let (e, ctx, vocab) = (32usize, 8usize, 24usize);
    let config = serde_json::json!({
        "model_type": "gpt2",
        "n_layer": 1,
        "n_head": 4,
        "n_embd": e,
        "n_ctx": ctx,
        "vocab_size": vocab,
        "layer_norm_epsilon": 1e-5,
    });

    let mut t = BTreeMap::new();
    t.insert("wte.weight".into(), mat(vocab, e));
    t.insert("wpe.weight".into(), mat(ctx, e));
    t.insert("ln_f.weight".into(), vec1(e));
    t.insert("ln_f.bias".into(), vec1(e));
    let p = "h.0";
    for (n, m) in [
        ("ln_1.weight", vec1(e)),
        ("ln_1.bias", vec1(e)),
        ("ln_2.weight", vec1(e)),
        ("ln_2.bias", vec1(e)),
        ("attn.c_attn.weight", mat(e, 3 * e)),
        ("attn.c_attn.bias", vec1(3 * e)),
        ("attn.c_proj.weight", mat(e, e)),
        ("attn.c_proj.bias", vec1(e)),
        ("mlp.c_fc.weight", mat(e, 4 * e)),
        ("mlp.c_fc.bias", vec1(4 * e)),
        ("mlp.c_proj.weight", mat(4 * e, e)),
        ("mlp.c_proj.bias", vec1(e)),
    ] {
        t.insert(format!("{p}.{n}"), m);
    }
    // The stored causal mask: a lower-triangular block `torch` had nowhere else
    // to put. Nothing reads it, and nothing should complain about it.
    t.insert(format!("{p}.attn.bias"), mat(ctx, ctx));
    (config, t)
}

/// Load through an architecture and report what the checkpoint was left holding.
///
/// `Err` is the load failing, which two of these tests are about.
fn load_and_list(
    tag: &str,
    config: serde_json::Value,
    tensors: BTreeMap<String, Mat>,
) -> Result<Vec<String>, String> {
    let path =
        std::env::temp_dir().join(format!("kvad-unread-{}-{tag}.safetensors", std::process::id()));
    write_safetensors(&path, &tensors);

    let out = (|| {
        let ckpt = Checkpoint::open(std::slice::from_ref(&path)).map_err(|e| e.to_string())?;
        let spec = Spec::from_config(Json::new(config)).map_err(|e| e.to_string())?;
        let live = Live::new(&ckpt, Precision::F32);
        // The registry's dispatch is crate-private, so the two architectures are
        // named here instead. It costs this branch and keeps `Arch` as it is.
        match spec.arch.is("llama") {
            true => {
                kvad::model::llama::Model::load(&live, spec.clone()).map_err(|e| e.to_string())?;
            }
            false => {
                kvad::model::gpt2::Model::load(&live, spec.clone()).map_err(|e| e.to_string())?;
            }
        }
        Ok(live.unread())
    })();

    let _ = std::fs::remove_file(&path);
    out
}

/// The bug itself, in the shape the check can see: a weight in the file that
/// the loader has no idea about.
#[test]
fn a_weight_the_loader_does_not_know_about_is_named() {
    let (config, mut t) = qwen3(true);
    t.insert("model.layers.0.self_attn.some_new_norm.weight".into(), vec1(16));

    let left = load_and_list("unknown", config, t).unwrap();
    assert_eq!(left, vec!["model.layers.0.self_attn.some_new_norm.weight"]);
}

/// Qwen3's per-head norms are read, so a correct load leaves nothing over.
///
/// The other half of the same check, and the one that keeps it honest: a test
/// that only ever sees failures would pass just as well if `unread` returned
/// every name in the file.
#[test]
fn a_fully_implemented_model_leaves_nothing_over() {
    let (config, t) = qwen3(true);
    let left = load_and_list("complete", config, t).unwrap();
    assert!(left.is_empty(), "nothing should be left, but: {left:?}");
}

/// GPT-2 stores its causal mask as a tensor, and recomputing it is not an
/// omission. Refusing that checkpoint would be the check doing the harm it
/// exists to prevent.
#[test]
fn gpt2s_stored_causal_mask_is_not_an_omission() {
    let (config, t) = gpt2();
    let left = load_and_list("gpt2-mask", config, t).unwrap();
    assert!(left.is_empty(), "nothing should be left, but: {left:?}");
}

/// A tied model's duplicate output head is skipped on purpose, which the check
/// has to be able to tell from a tensor forgotten.
///
/// Qwen3-0.6B really does ship one: 297 MB byte-identical to its embedding
/// table. Tying means the head *is* the table, so the copy is redundant — but
/// nothing reads it, and a check that could not tell the difference would refuse
/// the model.
#[test]
fn a_tied_models_duplicate_head_is_skipped_not_forgotten() {
    let (config, mut t) = qwen3(true);
    t.insert("lm_head.weight".into(), mat(24, 32));

    let left = load_and_list("tied-duplicate", config, t).unwrap();
    assert!(left.is_empty(), "the duplicate head is deliberate, but: {left:?}");
}

/// An untied model with no head of its own is a broken checkpoint, and all
/// three architectures used to fall through to the embedding table instead —
/// a different model from the one the config describes, run without a word.
#[test]
fn an_untied_model_without_a_head_says_so() {
    let (config, t) = qwen3(false);
    let err = load_and_list("untied-headless", config, t)
        .expect_err("an untied model with no `lm_head.weight` must not load");
    assert!(err.contains("lm_head.weight"), "unhelpful message: {err}");
    assert!(err.contains("tie_word_embeddings"), "should name the other possibility: {err}");
}

/// The same rule in GPT-2, which reaches it by a different line of code: its
/// head has an optional bias beside it, so it could not use the shared helper
/// for both halves.
#[test]
fn an_untied_gpt2_without_a_head_says_so() {
    let (mut config, t) = gpt2();
    config["tie_word_embeddings"] = serde_json::json!(false);

    let err = load_and_list("gpt2-headless", config, t)
        .expect_err("an untied GPT-2 with no `lm_head.weight` must not load");
    assert!(err.contains("lm_head.weight"), "unhelpful message: {err}");
}
