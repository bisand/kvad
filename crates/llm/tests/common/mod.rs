//! What a model cut into shards is held to, for every architecture that
//! can be cut: the answer of the model run whole, to the bit.

#![allow(dead_code)]

use kvad::cluster::session::PipelineSession;
use kvad::model::{gpt2, llama, CpuSession, Json, Session, Shard, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// Every way of cutting `n_layer` layers that is worth telling apart: not
/// at all, in two, in three, with the ends on a shard that has no layers —
/// a coordinator that only embeds and reads the head — and with the ends
/// on the last shard instead of the first.
pub fn cuts(n_layer: usize) -> Vec<Vec<Shard>> {
    let mut all = vec![Shard::cut(n_layer, 1), Shard::cut(n_layer, 2)];
    if n_layer >= 3 {
        all.push(Shard::cut(n_layer, 3));
    }
    all.push(vec![Shard { layers: 0..0, ends: true }, Shard { layers: 0..1, ends: false }, Shard { layers: 1..n_layer, ends: false }]);
    all.push(vec![Shard { layers: 0..1, ends: false }, Shard { layers: 1..n_layer, ends: true }]);
    all
}

/// The same tokens through both, and the same bits out of both: a prompt
/// read in several passes, tokens decoded one at a time, a rewind and a
/// different continuation, and every position's logits at once.
pub fn same_answers(whole: &mut dyn Session, cut: &mut dyn Session, what: &str) {
    let vocab = whole.spec().vocab_size as u32;
    let prompt: Vec<u32> = (0..21u32).map(|i| (i * 7 + 3) % vocab).collect();
    fn both(whole: &mut dyn Session, cut: &mut dyn Session, tokens: &[u32], what: &str, step: &str) -> Vec<f32> {
        let (a, b) = (whole.forward(tokens).unwrap(), cut.forward(tokens).unwrap());
        assert!(a.iter().all(|x| x.is_finite()), "{what}, {step}: not finite");
        assert_eq!(bits(&a), bits(&b), "{what}, {step}: the cut model answers differently");
        assert_eq!(whole.cached(), cut.cached(), "{what}, {step}: they hold different lengths");
        a
    }

    // Prefill: 21 tokens at 8 a pass is three passes, the last a short one.
    let mut logits = both(whole, cut, &prompt, what, "prefill");
    for i in 0..5 {
        logits = both(whole, cut, &[argmax(&logits)], what, &format!("decode {i}"));
    }

    // Back to the middle of the prompt, and on with something else.
    assert_eq!(whole.truncate(9).unwrap(), 9);
    assert_eq!(cut.truncate(9).unwrap(), 9, "{what}: every stage rewinds to where it was told");
    let other: Vec<u32> = (0..6u32).map(|i| (i * 11 + 1) % vocab).collect();
    logits = both(whole, cut, &other, what, "after a rewind");
    both(whole, cut, &[argmax(&logits)], what, "decode after a rewind");

    // Rewinding to more than is held changes nothing, and says what is held.
    assert_eq!(cut.truncate(999).unwrap(), whole.cached());

    // Every position's logits, which is the head over every row.
    assert_eq!(whole.truncate(0).unwrap(), 0);
    assert_eq!(cut.truncate(0).unwrap(), 0);
    let (a, b) = (whole.forward_all(&prompt[..19]).unwrap(), cut.forward_all(&prompt[..19]).unwrap());
    assert_eq!(a.len(), 19 * vocab as usize);
    assert_eq!(bits(&a), bits(&b), "{what}, every position: the cut model answers differently");
}

/// Hold every cut of a model to the model. `load` reads one shard from the
/// checkpoint and must leave nothing in it unaccounted for.
pub fn every_cut_is_the_model(spec: &Spec, precision: Precision, what: &str, load: &dyn Fn(&Shard) -> Box<dyn Transformer>) {
    let whole_shard = Shard::whole(spec);
    for cut in cuts(spec.n_layer) {
        let names: Vec<String> = cut.iter().map(Shard::to_string).collect();
        let what = format!("{what} {precision}, cut as [{}]", names.join("; "));
        let mut whole = CpuSession::new(load(&whole_shard), precision).reading(8);
        let shards: Vec<Box<dyn Transformer>> = cut.iter().map(load).collect();
        let (params, bytes) = shards.iter().fold((0, 0), |a, s| (a.0 + s.param_count(), a.1 + s.memory_bytes()));
        let mut split = PipelineSession::local(shards, precision).unwrap().reading(8);
        assert_eq!((split.param_count(), split.weight_bytes()), (params, bytes), "{what}: the pipeline holds what its shards hold");
        assert_eq!(split.param_count(), whole.param_count(), "{what}: and between them that is the model");
        same_answers(&mut whole, &mut split, &what);
    }
}

// ---------------------------------------------------------------------------
// Small models, built here
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random numbers, so a failure is reproducible.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 8_388_608.0 - 1.0) * 0.25
    }

    pub fn matrix(&mut self, rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: (0..rows * cols).map(|_| self.next()).collect() }
    }

    /// A bias: one row.
    pub fn vector(&mut self, n: usize) -> Mat {
        self.matrix(1, n)
    }

    /// Norm weights, which are near one rather than near zero.
    pub fn ones(&mut self, n: usize) -> Mat {
        Mat { rows: 1, cols: n, data: (0..n).map(|_| 1.0 + self.next()).collect() }
    }
}

pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// The smallest safetensors writer that will do: a JSON header, its length,
/// and the arrays.
pub fn write_safetensors(path: &PathBuf, tensors: &BTreeMap<String, Mat>) {
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

pub const VOCAB: usize = 64;

/// A checkpoint on disk for as long as a test wants shards read from it.
pub struct Fixture {
    pub path: PathBuf,
    pub ckpt: Checkpoint,
    pub spec: Spec,
}

impl Fixture {
    pub fn new(tag: &str, tensors: &BTreeMap<String, Mat>, config: serde_json::Value) -> Fixture {
        let path = std::env::temp_dir().join(format!("kvad-split-{}-{tag}.safetensors", std::process::id()));
        write_safetensors(&path, tensors);
        let ckpt = Checkpoint::open(std::slice::from_ref(&path)).unwrap();
        Fixture { path, ckpt, spec: Spec::from_config(Json::new(config)).unwrap() }
    }

    /// One shard, read through the same audit a whole model is: whatever it
    /// did not read it has to have said it was leaving.
    pub fn load(&self, precision: Precision, shard: &Shard) -> Box<dyn Transformer> {
        read_shard(&self.ckpt, &self.spec, precision, shard)
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
pub fn llama(model_type: &str, tag: &str) -> Fixture {
    let (tensors, config) = llama_parts(model_type);
    Fixture::new(tag, &tensors, config)
}

/// The same model as its tensors and its config, for a test that wants to
/// file them some other way than in one file.
pub fn llama_parts(model_type: &str) -> (BTreeMap<String, Mat>, serde_json::Value) {
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
    (t, config)
}

/// A small GPT-2, three layers: learned positions added at the door, which
/// is the one place a shard needs telling where in the sequence it is; and
/// tied embeddings, so the head is the table the other end of the model
/// reads.
pub fn gpt2_model(tag: &str) -> Fixture {
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

/// One shard out of a checkpoint, with nothing in the file unaccounted for.
pub fn read_shard(ckpt: &Checkpoint, spec: &Spec, precision: Precision, shard: &Shard) -> Box<dyn Transformer> {
    let live = Live::new(ckpt, precision);
    let model: Box<dyn Transformer> = match spec.arch.id() {
        "gpt2" => Box::new(gpt2::Model::load_shard(&live, spec.clone(), shard).unwrap()),
        _ => Box::new(llama::Model::load_shard(&live, spec.clone(), shard).unwrap()),
    };
    let unread = live.unread();
    assert!(unread.is_empty(), "{shard}: neither read nor left to another shard: {unread:?}");
    assert_eq!(model.shard(), *shard);
    model
}

/// A model as a directory, the way a large one is published: its tensors
/// across several weight files, an index saying which file holds which,
/// and a config. `file_of` says which file a tensor goes in.
pub fn sharded_dir(tag: &str, tensors: BTreeMap<String, Mat>, config: &serde_json::Value, file_of: &dyn Fn(&str) -> usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kvad-sharded-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut files: BTreeMap<usize, BTreeMap<String, Mat>> = BTreeMap::new();
    let mut weight_map = serde_json::Map::new();
    for (name, mat) in tensors {
        let file = file_of(&name);
        weight_map.insert(name.clone(), serde_json::json!(format!("model-{file:05}.safetensors")));
        files.entry(file).or_default().insert(name, mat);
    }
    for (file, tensors) in &files {
        write_safetensors(&dir.join(format!("model-{file:05}.safetensors")), tensors);
    }
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({ "weight_map": weight_map }).to_string()).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    // Asked for by name and never opened by anything here.
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    dir
}
