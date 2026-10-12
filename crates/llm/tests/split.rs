//! A model cut into shards is the same model.
//!
//! This is the claim everything about a cluster rests on, and it is a claim
//! about bits, not about closeness: a layer split moves where a matmul runs
//! and nothing about what it computes, so the logits of a model cut in
//! three have to be the logits of the model whole, exactly. "Close" would
//! pass with a row attending over the wrong cache, a position counted from
//! the wrong shard, or a norm applied on both sides of a cut.
//!
//! No sockets here. Every stage is in this process; what is under test is
//! the cut.

#![cfg(all(feature = "cluster", feature = "arch-llama", feature = "arch-gpt2"))]

mod common;

use kvad::cluster::session::PipelineSession;
use kvad::cluster::stage::{LocalStage, Stage};
use kvad::model::{gpt2, llama, Json, Shard, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Deterministic pseudo-random numbers, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 8_388_608.0 - 1.0) * 0.25
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: (0..rows * cols).map(|_| self.next()).collect() }
    }

    /// A bias: one row.
    fn vector(&mut self, n: usize) -> Mat {
        self.matrix(1, n)
    }

    /// Norm weights, which are near one rather than near zero.
    fn ones(&mut self, n: usize) -> Mat {
        Mat { rows: 1, cols: n, data: (0..n).map(|_| 1.0 + self.next()).collect() }
    }
}

struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

/// The smallest safetensors writer that will do: a JSON header, its length,
/// and the arrays.
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
        header.push_str(&format!("\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape},\"data_offsets\":[{start},{}]}}", blob.len()));
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

const VOCAB: usize = 64;

/// A checkpoint on disk for as long as a test wants shards read from it.
struct Fixture {
    path: PathBuf,
    ckpt: Checkpoint,
    spec: Spec,
}

impl Fixture {
    fn new(tag: &str, tensors: &BTreeMap<String, Mat>, config: serde_json::Value) -> Fixture {
        let path = std::env::temp_dir().join(format!("kvad-split-{}-{tag}.safetensors", std::process::id()));
        write_safetensors(&path, tensors);
        let ckpt = Checkpoint::open(std::slice::from_ref(&path)).unwrap();
        Fixture { path, ckpt, spec: Spec::from_config(Json::new(config)).unwrap() }
    }

    /// One shard, read through the same audit a whole model is: whatever it
    /// did not read it has to have said it was leaving.
    fn load(&self, precision: Precision, shard: &Shard) -> Box<dyn Transformer> {
        let live = Live::new(&self.ckpt, precision);
        let model: Box<dyn Transformer> = match self.spec.arch.id() {
            "gpt2" => Box::new(gpt2::Model::load_shard(&live, self.spec.clone(), shard).unwrap()),
            _ => Box::new(llama::Model::load_shard(&live, self.spec.clone(), shard).unwrap()),
        };
        let unread = live.unread();
        assert!(unread.is_empty(), "{shard}: neither read nor left to another shard: {unread:?}");
        assert_eq!(model.shard(), *shard);
        model
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A small Llama, four layers of it: grouped-query attention, an untied
/// head, and — for `qwen3` — the per-head norms on Q and K.
///
/// Every matrix has a row length that is a multiple of 32, because anything
/// narrower the quantiser leaves as f32, and a tiny model that quietly
/// skipped quantisation would test the wrong kernels.
fn llama(model_type: &str, tag: &str) -> Fixture {
    let (hidden, n_head, n_kv_head, n_layer, inter) = (64usize, 4usize, 2usize, 4usize, 128usize);
    let head_dim = hidden / n_head;
    let mut r = Rng(0x0bad_5eed_1234_5678);
    let mut t: BTreeMap<String, Mat> = BTreeMap::new();
    t.insert("model.embed_tokens.weight".into(), r.matrix(VOCAB, hidden));
    t.insert("model.norm.weight".into(), r.ones(hidden));
    t.insert("lm_head.weight".into(), r.matrix(VOCAB, hidden));
    for l in 0..n_layer {
        let p = format!("model.layers.{l}");
        t.insert(format!("{p}.input_layernorm.weight"), r.ones(hidden));
        t.insert(format!("{p}.post_attention_layernorm.weight"), r.ones(hidden));
        t.insert(format!("{p}.self_attn.q_proj.weight"), r.matrix(n_head * head_dim, hidden));
        t.insert(format!("{p}.self_attn.k_proj.weight"), r.matrix(n_kv_head * head_dim, hidden));
        t.insert(format!("{p}.self_attn.v_proj.weight"), r.matrix(n_kv_head * head_dim, hidden));
        t.insert(format!("{p}.self_attn.o_proj.weight"), r.matrix(hidden, n_head * head_dim));
        if model_type == "qwen3" {
            t.insert(format!("{p}.self_attn.q_norm.weight"), r.ones(head_dim));
            t.insert(format!("{p}.self_attn.k_norm.weight"), r.ones(head_dim));
        }
        t.insert(format!("{p}.mlp.gate_proj.weight"), r.matrix(inter, hidden));
        t.insert(format!("{p}.mlp.up_proj.weight"), r.matrix(inter, hidden));
        t.insert(format!("{p}.mlp.down_proj.weight"), r.matrix(hidden, inter));
    }
    let config = serde_json::json!({
        "model_type": model_type,
        "hidden_size": hidden,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_head,
        "num_hidden_layers": n_layer,
        "vocab_size": VOCAB,
        "intermediate_size": inter,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": false,
    });
    Fixture::new(tag, &t, config)
}

/// A small GPT-2, three layers: learned positions added at the door, which
/// is the one place a shard needs telling where in the sequence it is; and
/// tied embeddings, so the head is the table the other end of the model
/// reads.
fn gpt2_model(tag: &str) -> Fixture {
    let (e, n_head, n_layer, n_ctx) = (64usize, 4usize, 3usize, 48usize);
    let mut r = Rng(0x5eed_0f_a_9272);
    let mut t: BTreeMap<String, Mat> = BTreeMap::new();
    t.insert("wte.weight".into(), r.matrix(VOCAB, e));
    t.insert("wpe.weight".into(), r.matrix(n_ctx, e));
    t.insert("ln_f.weight".into(), r.ones(e));
    t.insert("ln_f.bias".into(), r.vector(e));
    for l in 0..n_layer {
        let p = format!("h.{l}");
        t.insert(format!("{p}.ln_1.weight"), r.ones(e));
        t.insert(format!("{p}.ln_1.bias"), r.vector(e));
        // Stored `[in, out]`, as a Conv1D writes them.
        t.insert(format!("{p}.attn.c_attn.weight"), r.matrix(e, 3 * e));
        t.insert(format!("{p}.attn.c_attn.bias"), r.vector(3 * e));
        t.insert(format!("{p}.attn.c_proj.weight"), r.matrix(e, e));
        t.insert(format!("{p}.attn.c_proj.bias"), r.vector(e));
        t.insert(format!("{p}.ln_2.weight"), r.ones(e));
        t.insert(format!("{p}.ln_2.bias"), r.vector(e));
        t.insert(format!("{p}.mlp.c_fc.weight"), r.matrix(e, 4 * e));
        t.insert(format!("{p}.mlp.c_fc.bias"), r.vector(4 * e));
        t.insert(format!("{p}.mlp.c_proj.weight"), r.matrix(4 * e, e));
        t.insert(format!("{p}.mlp.c_proj.bias"), r.vector(e));
    }
    let config = serde_json::json!({
        "model_type": "gpt2",
        "n_embd": e,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_positions": n_ctx,
        "vocab_size": VOCAB,
        "layer_norm_epsilon": 1e-5,
    });
    Fixture::new(tag, &t, config)
}

const PRECISIONS: [Precision; 3] = [Precision::F32, Precision::Q8, Precision::Q4];

#[test]
fn a_llama_cut_into_shards_answers_to_the_bit() {
    for model_type in ["llama", "qwen3"] {
        for precision in PRECISIONS {
            let fixture = llama(model_type, &format!("{model_type}-{precision}"));
            common::every_cut_is_the_model(&fixture.spec, precision, model_type, &|shard| fixture.load(precision, shard));
        }
    }
}

#[test]
fn a_gpt2_cut_into_shards_answers_to_the_bit() {
    for precision in PRECISIONS {
        let fixture = gpt2_model(&format!("gpt2-{precision}"));
        common::every_cut_is_the_model(&fixture.spec, precision, "gpt2", &|shard| fixture.load(precision, shard));
    }
}

/// A shard holds what it was asked for and no more, and says so in what it
/// counts: the parameters of the pieces are the parameters of the model.
#[test]
fn a_shard_holds_its_own_layers_and_nothing_else() {
    let fixture = llama("llama", "holds");
    let whole = fixture.load(Precision::F32, &Shard::whole(&fixture.spec));
    let middle = fixture.load(Precision::F32, &Shard { layers: 1..3, ends: false });
    let ends = fixture.load(Precision::F32, &Shard { layers: 0..0, ends: true });
    let per_layer = middle.param_count() / 2;
    assert_eq!(whole.param_count(), ends.param_count() + 4 * per_layer);
    assert_eq!(ends.param_count(), 2 * VOCAB * 64 + 64, "the table, the head and the final norm");
    assert!(middle.memory_bytes() > 0 && middle.memory_bytes() < whole.memory_bytes());
}

#[test]
#[should_panic(expected = "which another shard holds")]
fn a_shard_without_the_ends_cannot_embed() {
    let fixture = llama("llama", "no-ends");
    fixture.load(Precision::F32, &Shard { layers: 1..3, ends: false }).embed(&[1, 2, 3], 0);
}

/// Shards that are not one model are refused before a token is run.
#[test]
fn shards_that_are_not_one_model_are_not_a_pipeline() {
    let fixture = llama("llama", "refused");
    let pipeline = |shards: &[Shard]| {
        let loaded = shards.iter().map(|s| fixture.load(Precision::F32, s)).collect();
        PipelineSession::local(loaded, Precision::F32).map(|_| ()).map_err(|e| e.to_string())
    };
    let part = |layers: std::ops::Range<usize>, ends: bool| Shard { layers, ends };

    assert!(pipeline(&[part(0..2, true), part(2..4, false)]).is_ok());
    // In any order: a pipeline follows the layers, not the list.
    assert!(pipeline(&[part(2..4, false), part(0..2, true)]).is_ok());

    let gap = pipeline(&[part(0..1, true), part(2..4, false)]).unwrap_err();
    assert!(gap.contains("layer 1 is nowhere"), "{gap}");
    let overlap = pipeline(&[part(0..3, true), part(2..4, false)]).unwrap_err();
    assert!(overlap.contains("layer 2 is twice"), "{overlap}");
    let short = pipeline(&[part(0..3, true)]).unwrap_err();
    assert!(short.contains("3 layers of 4"), "{short}");
    let headless = pipeline(&[part(0..2, false), part(2..4, false)]).unwrap_err();
    assert!(headless.contains("0 shards hold the embedding"), "{headless}");
    let two_heads = pipeline(&[part(0..2, true), part(2..4, true)]).unwrap_err();
    assert!(two_heads.contains("2 shards hold the embedding"), "{two_heads}");
    assert!(PipelineSession::local(Vec::new(), Precision::F32).is_err());
}

/// A stage is told the position of the rows it is handed, and refuses rows
/// from anywhere but the end of what it holds: a token lost or run twice
/// between two machines is an error at that token, not a wrong reply.
#[test]
fn a_stage_refuses_rows_it_is_not_up_to() {
    let fixture = llama("llama", "stage");
    let e = fixture.spec.n_embd;
    let shard = Shard { layers: 1..3, ends: false };
    let mut stage = LocalStage::new(Arc::from(fixture.load(Precision::F32, &shard)));
    assert_eq!(stage.shard(), shard);

    let rows = |m: usize| vec![0.125f32; m * e];
    assert_eq!(stage.run(7, 0, 3, rows(3)).unwrap().len(), 3 * e);
    // The same rows again, and rows from further on.
    let again = stage.run(7, 0, 3, rows(3)).unwrap_err().to_string();
    assert!(again.contains("handed position 0 of a conversation it holds 3 positions of"), "{again}");
    assert!(stage.run(7, 5, 1, rows(1)).is_err());
    // A refusal changes nothing: the next rows are still the next rows.
    assert!(stage.run(7, 3, 1, rows(1)).is_ok());

    // Another conversation starts from nothing, beside the first.
    assert!(stage.run(8, 0, 2, rows(2)).is_ok());
    assert!(stage.run(7, 4, 1, rows(1)).is_ok());

    // Rows that are not the shape they were said to be.
    let ragged = stage.run(8, 2, 2, rows(1)).unwrap_err().to_string();
    assert!(ragged.contains("2 rows of 64 were announced"), "{ragged}");

    assert_eq!(stage.truncate(7, 2).unwrap(), 2);
    assert!(stage.run(7, 2, 1, rows(1)).is_ok());
    assert_eq!(stage.truncate(99, 5).unwrap(), 0, "a conversation never seen holds nothing");
}
