//! A LittleBit checkpoint, written the way the reference writes one, runs the
//! model its factors stand for.
//!
//! Two checkpoints of one tiny Qwen3: one stores every projection as packed
//! LittleBit factors — `int32` words of signs, `int64` shapes, a residual
//! path under `_R` — and the other stores the dense matrix those factors
//! multiply out to, computed here entry by entry. If the bit order, the sign
//! convention, the padding, a scale or the residual sum were read wrong, the
//! two would disagree; agreeing on logits through prefill, through decode one
//! token at a time and through the cache file is the claim that they were not.
//!
//! The packing below is written out independently of `kvad::littlebit` on
//! purpose: a reader checked against its own writer can agree with itself
//! about a wrong format.

use kvad::model::{llama, KvCache, Json, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::path::PathBuf;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
    }

    fn vec(&mut self, n: usize, around: f32, spread: f32) -> Vec<f32> {
        (0..n).map(|_| around + spread * self.next()).collect()
    }
}

/// One tensor as safetensors stores it.
struct Raw {
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn f32s(shape: Vec<usize>, v: &[f32]) -> Raw {
    Raw { dtype: "F32", shape, bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

fn write_safetensors(path: &PathBuf, tensors: &BTreeMap<String, Raw>) {
    let mut header = String::from("{");
    let mut blob: Vec<u8> = Vec::new();
    for (i, (name, t)) in tensors.iter().enumerate() {
        let start = blob.len();
        blob.extend_from_slice(&t.bytes);
        let shape = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{}\",\"shape\":[{shape}],\"data_offsets\":[{start},{}]}}",
            t.dtype,
            blob.len()
        ));
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&blob);
    std::fs::write(path, out).unwrap();
}

/// Signs as the reference's `binary_packer` stores them: `+1 -> 0`,
/// `-1 -> 1`, thirty-two to an `int32`, least significant bit first, each row
/// padded with `+1`.
fn packed(rows: usize, cols: usize, signs: &[f32]) -> Raw {
    let words = cols.div_ceil(32);
    let mut bytes = Vec::with_capacity(rows * words * 4);
    for r in 0..rows {
        for w in 0..words {
            let mut word = 0u32;
            for j in 0..32 {
                let c = w * 32 + j;
                if c < cols && signs[r * cols + c] < 0.0 {
                    word |= 1 << j;
                }
            }
            bytes.extend_from_slice(&(word as i32).to_le_bytes());
        }
    }
    Raw { dtype: "I32", shape: vec![rows, words], bytes }
}

fn shape(rows: usize, cols: usize) -> Raw {
    let bytes = [rows as i64, cols as i64].iter().flat_map(|x| x.to_le_bytes()).collect();
    Raw { dtype: "I64", shape: vec![2], bytes }
}

/// One projection, both ways: its LittleBit tensors into `lb`, and the dense
/// matrix they multiply out to into `dense`.
fn projection(
    r: &mut Rng,
    stem: &str,
    out: usize,
    inp: usize,
    rank: usize,
    lb: &mut BTreeMap<String, Raw>,
    dense: &mut BTreeMap<String, Raw>,
) {
    let mut w = vec![0.0f64; out * inp];
    for suffix in ["", "_R"] {
        let u: Vec<f32> = r.vec(out * rank, 0.0, 1.0).iter().map(|x| x.signum()).collect();
        let v: Vec<f32> = r.vec(rank * inp, 0.0, 1.0).iter().map(|x| x.signum()).collect();
        // Magnitudes chosen so a product of the five lands near the scale of
        // an ordinary initialisation, and the model's activations stay sane.
        let u1 = r.vec(out, 0.15, 0.05);
        let u2 = r.vec(rank, 0.6, 0.3);
        let v1 = r.vec(rank, 0.6, 0.3);
        let v2 = r.vec(inp, 0.4, 0.1);
        for o in 0..out {
            for i in 0..inp {
                let mut sum = 0.0f64;
                for k in 0..rank {
                    sum += (u[o * rank + k] * u2[k] * v1[k] * v[k * inp + i]) as f64;
                }
                w[o * inp + i] += u1[o] as f64 * sum * v2[i] as f64;
            }
        }
        lb.insert(format!("{stem}.U{suffix}_packed"), packed(out, rank, &u));
        lb.insert(format!("{stem}.U{suffix}_shape"), shape(out, rank));
        lb.insert(format!("{stem}.V{suffix}_packed"), packed(rank, inp, &v));
        lb.insert(format!("{stem}.V{suffix}_shape"), shape(rank, inp));
        lb.insert(format!("{stem}.u1{suffix}"), f32s(vec![1, out], &u1));
        lb.insert(format!("{stem}.u2{suffix}"), f32s(vec![1, rank], &u2));
        lb.insert(format!("{stem}.v1{suffix}"), f32s(vec![1, rank], &v1));
        lb.insert(format!("{stem}.v2{suffix}"), f32s(vec![1, inp], &v2));
    }
    // The bookkeeping scalars every converted layer carries, which the
    // loader must account for without needing.
    lb.insert(format!("{stem}._eff_bit_target"), f32s(vec![], &[0.55]));
    lb.insert(format!("{stem}._eff_bit_actual"), f32s(vec![], &[0.55]));
    lb.insert(
        format!("{stem}._split_dim_final"),
        Raw { dtype: "I64", shape: vec![], bytes: (rank as i64).to_le_bytes().to_vec() },
    );
    let w: Vec<f32> = w.iter().map(|&x| x as f32).collect();
    dense.insert(format!("{stem}.weight"), f32s(vec![out, inp], &w));
}

const VOCAB: usize = 64;

/// Both checkpoints, and the spec they share.
///
/// Ranks that are not whole words — 13 and 40 — so the padding at the end of
/// every row of `U` is exercised, as it is in every real checkpoint.
fn checkpoints(tag: &str) -> (PathBuf, PathBuf, Spec) {
    let (hidden, n_head, n_kv, n_layer, inter) = (64usize, 4usize, 2usize, 2usize, 96usize);
    let head_dim = hidden / n_head;
    let mut r = Rng(0x5eed_0f_11771e_b17);
    let mut shared: BTreeMap<String, Raw> = BTreeMap::new();
    let mut lb: BTreeMap<String, Raw> = BTreeMap::new();
    let mut dense: BTreeMap<String, Raw> = BTreeMap::new();

    // Embeddings and the head stay dense in a LittleBit model: the reference
    // excludes `lm_head` and has no linear layer to replace in a lookup table.
    shared.insert("model.embed_tokens.weight".into(), f32s(vec![VOCAB, hidden], &r.vec(VOCAB * hidden, 0.0, 1.0)));
    shared.insert("lm_head.weight".into(), f32s(vec![VOCAB, hidden], &r.vec(VOCAB * hidden, 0.0, 0.2)));
    shared.insert("model.norm.weight".into(), f32s(vec![hidden], &r.vec(hidden, 1.0, 0.1)));
    for l in 0..n_layer {
        let p = format!("model.layers.{l}");
        for n in ["input_layernorm", "post_attention_layernorm"] {
            shared.insert(format!("{p}.{n}.weight"), f32s(vec![hidden], &r.vec(hidden, 1.0, 0.1)));
        }
        for n in ["q_norm", "k_norm"] {
            shared.insert(format!("{p}.self_attn.{n}.weight"), f32s(vec![head_dim], &r.vec(head_dim, 1.0, 0.1)));
        }
        let a = format!("{p}.self_attn");
        projection(&mut r, &format!("{a}.q_proj"), n_head * head_dim, hidden, 13, &mut lb, &mut dense);
        projection(&mut r, &format!("{a}.k_proj"), n_kv * head_dim, hidden, 13, &mut lb, &mut dense);
        projection(&mut r, &format!("{a}.v_proj"), n_kv * head_dim, hidden, 13, &mut lb, &mut dense);
        projection(&mut r, &format!("{a}.o_proj"), hidden, n_head * head_dim, 13, &mut lb, &mut dense);
        let m = format!("{p}.mlp");
        projection(&mut r, &format!("{m}.gate_proj"), inter, hidden, 40, &mut lb, &mut dense);
        projection(&mut r, &format!("{m}.up_proj"), inter, hidden, 40, &mut lb, &mut dense);
        projection(&mut r, &format!("{m}.down_proj"), hidden, inter, 40, &mut lb, &mut dense);
    }

    let dir = std::env::temp_dir();
    let id = std::process::id();
    let (lb_path, dense_path) =
        (dir.join(format!("kvad-lb-{id}-{tag}.safetensors")), dir.join(format!("kvad-lbd-{id}-{tag}.safetensors")));
    for (path, own) in [(&lb_path, lb), (&dense_path, dense)] {
        let mut all = own;
        for (k, v) in &shared {
            all.insert(k.clone(), Raw { dtype: v.dtype, shape: v.shape.clone(), bytes: v.bytes.clone() });
        }
        write_safetensors(path, &all);
    }

    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": hidden,
        "head_dim": head_dim,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "num_hidden_layers": n_layer,
        "vocab_size": VOCAB,
        "intermediate_size": inter,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": false,
    });
    (lb_path, dense_path, Spec::from_config(Json::new(config)).unwrap())
}

fn load(path: &PathBuf, spec: &Spec) -> llama::Model {
    let ckpt = Checkpoint::open(std::slice::from_ref(path)).unwrap();
    let live = Live::new(&ckpt, Precision::F32);
    let model = llama::Model::load(&live, spec.clone()).unwrap();
    let unread = live.unread();
    assert!(unread.is_empty(), "the loader never read {unread:?}");
    model
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    assert!(a.iter().all(|v| v.is_finite()), "{what}: not finite");
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 1e-3 * scale.max(1.0), "{what}: worst disagreement {worst} on logits up to {scale}");
}

#[test]
fn a_littlebit_checkpoint_runs_the_model_its_factors_multiply_out_to() {
    let (lb_path, dense_path, spec) = checkpoints("runs");
    let factored = load(&lb_path, &spec);
    let dense = load(&dense_path, &spec);

    // The factored model is much smaller than the dense one, which is the
    // point; at this size the vectors dominate, so only "smaller" is claimed.
    assert!(factored.memory_bytes() < dense.memory_bytes());
    assert_eq!(factored.param_count(), dense.param_count());

    let tokens = [3u32, 17, 42, 8, 60, 1];

    // Prefill: every projection through the batched path.
    let want = dense.forward_batch_all(&tokens, &mut KvCache::new(&spec));
    let got = factored.forward_batch_all(&tokens, &mut KvCache::new(&spec));
    close(&got, &want, "prefill");

    // Decode: one token at a time, every projection through the matvec path.
    let mut cache = KvCache::new(&spec);
    for (i, &t) in tokens.iter().enumerate() {
        let got = factored.forward(t, &mut cache);
        close(&got, &want[i * VOCAB..(i + 1) * VOCAB], &format!("decode at {i}"));
    }

    let _ = std::fs::remove_file(&lb_path);
    let _ = std::fs::remove_file(&dense_path);
}

/// Through the cache file and back: the second load of a LittleBit model maps
/// its signs rather than unpacking a checkpoint, and must run the same model.
#[test]
fn a_littlebit_model_survives_the_quantised_cache() {
    let (lb_path, dense_path, spec) = checkpoints("cache");
    let dir = std::env::temp_dir().join(format!("kvad-lb-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cache_path = dir.join("model.q8");

    let ckpt = Checkpoint::open(std::slice::from_ref(&lb_path)).unwrap();
    let mut live = Live::new(&ckpt, Precision::Q8);
    live.record_to(&cache_path).unwrap();
    let first = llama::Model::load(&live, spec.clone()).unwrap();
    assert!(live.unread().is_empty());
    live.finish("test/littlebit", std::slice::from_ref(&lb_path), &spec).unwrap().unwrap();

    let mapped = kvad::qcache::Mapped::open(&cache_path, "test/littlebit", std::slice::from_ref(&lb_path), &spec, Precision::Q8)
        .unwrap()
        .expect("the cache was just written");
    let second = llama::Model::load(&mapped, spec.clone()).unwrap();
    assert!(mapped.unread().is_empty(), "the cache lost {:?}", mapped.unread());

    let tokens = [5u32, 9, 33, 2];
    let a = first.forward_batch_all(&tokens, &mut KvCache::new(&spec));
    let b = second.forward_batch_all(&tokens, &mut KvCache::new(&spec));
    assert_eq!(a, b, "the mapped model must compute exactly what the loaded one did");

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&lb_path);
    let _ = std::fs::remove_file(&dense_path);
}

// ---------------------------------------------------------------------------
// Against a real checkpoint, by hand
// ---------------------------------------------------------------------------

/// The checkpoint directory these two read, or `None` to skip.
fn real_checkpoint() -> Option<PathBuf> {
    match std::env::var("KVAD_LITTLEBIT_CHECKPOINT") {
        Ok(dir) => Some(PathBuf::from(dir)),
        Err(_) => {
            eprintln!("set KVAD_LITTLEBIT_CHECKPOINT to a LittleBit checkpoint's directory");
            None
        }
    }
}

/// A deterministic activation, the same one `scripts/check-littlebit.py`
/// builds, so the fixture need not carry it.
fn activation(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 2654435761 + seed * 40503) % 1000) as f32 / 500.0 - 1.0).collect()
}

/// Write what three of a real checkpoint's projections compute, for
/// `scripts/check-littlebit.py` to compare with the reference's arithmetic.
///
/// The tiny model above checks this reader against a packing written out a
/// second time in Rust, which catches a slip and cannot catch a
/// misunderstanding shared by both. The script reads the same file with a
/// transcription of the reference's `binary_unpacker` and
/// `LittleBitLinear.forward` in numpy, which shares nothing with this.
///
/// ```text
/// KVAD_LITTLEBIT_CHECKPOINT=/path/to/model KVAD_LITTLEBIT_FIXTURE=/tmp/lb \
///     cargo test --release -p kvad --test littlebit -- --ignored writes
/// python3 scripts/check-littlebit.py /path/to/model /tmp/lb
/// ```
#[test]
#[ignore = "needs a LittleBit checkpoint; writes a fixture for scripts/check-littlebit.py"]
fn writes_a_fixture_from_a_real_checkpoint() {
    let Some(dir) = real_checkpoint() else { return };
    let out = PathBuf::from(std::env::var("KVAD_LITTLEBIT_FIXTURE").expect("KVAD_LITTLEBIT_FIXTURE"));
    std::fs::create_dir_all(&out).unwrap();
    let ckpt = Checkpoint::open(&[dir.join("model.safetensors")]).unwrap();
    let n_layer = ckpt.names().filter(|n| n.ends_with("self_attn.q_proj.U_packed")).count();
    let mut cases = Vec::new();
    for (layer, what) in [(0, "self_attn.k_proj"), (n_layer / 2, "mlp.down_proj"), (n_layer - 1, "self_attn.o_proj")] {
        let stem = format!("layers.{layer}.{what}");
        let f = kvad::littlebit::read(&ckpt, &stem).unwrap().expect("LittleBit factors");
        let m = 3;
        let x = activation(f.cols(), 0);
        let xs: Vec<f32> = (0..m).flat_map(|i| activation(f.cols(), i + 1)).collect();
        cases.push(serde_json::json!({
            "stem": stem, "rows": f.rows(), "cols": f.cols(), "m": m,
            "bits_per_weight": f.bits_per_weight(),
            "matvec": f.matvec(&x), "matmul": f.matmul(&xs, m),
        }));
    }
    std::fs::write(out.join("littlebit.json"), serde_json::to_string(&cases).unwrap()).unwrap();
}

/// Perplexity over the first windows of a text, the way the reference's
/// `eval.py` measures it: tokenised whole, cut into windows of 2048, and the
/// mean cross-entropy of each next token.
///
/// For WikiText-2, the rows of its `test` split joined by blank lines — which
/// is what the published checkpoints report.
///
/// ```text
/// KVAD_LITTLEBIT_CHECKPOINT=/path/to/model KVAD_LITTLEBIT_TEXT=wikitext-2-test.txt \
///     KVAD_LITTLEBIT_WINDOWS=3 cargo test --release -p kvad --test littlebit \
///     -- --ignored --nocapture perplexity
/// ```
#[test]
#[ignore = "needs a LittleBit checkpoint and a text; minutes a window on a CPU"]
fn perplexity_of_a_real_checkpoint() {
    let Some(dir) = real_checkpoint() else { return };
    let text = std::fs::read_to_string(std::env::var("KVAD_LITTLEBIT_TEXT").expect("KVAD_LITTLEBIT_TEXT")).unwrap();
    let windows: usize = std::env::var("KVAD_LITTLEBIT_WINDOWS").map(|s| s.parse().unwrap()).unwrap_or(3);
    let seqlen = 2048;

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let ids = tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    assert!(ids.len() >= windows * seqlen, "{} tokens is fewer than {windows} windows", ids.len());
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let spec = Spec::from_config(Json::new(config)).unwrap();
    let ckpt = Checkpoint::open(&[dir.join("model.safetensors")]).unwrap();
    let live = Live::new(&ckpt, Precision::Q8);
    let model = llama::Model::load(&live, spec.clone()).unwrap();
    assert!(live.unread().is_empty());

    let vocab = spec.vocab_size;
    let (mut nll, mut n) = (0.0f64, 0usize);
    for w in 0..windows {
        let batch = &ids[w * seqlen..(w + 1) * seqlen];
        let logits = model.forward_batch_all(batch, &mut KvCache::new(&spec));
        for j in 0..seqlen - 1 {
            let row = &logits[j * vocab..(j + 1) * vocab];
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
            nll += max + sum.ln() - row[batch[j + 1] as usize] as f64;
        }
        n += seqlen - 1;
        eprintln!("window {w}: perplexity so far {:.3}", (nll / n as f64).exp());
    }
    eprintln!("perplexity over {windows} windows of {seqlen}: {:.3}", (nll / n as f64).exp());
}
