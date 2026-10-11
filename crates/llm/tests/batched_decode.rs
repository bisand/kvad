//! Decode, batched: several sequences advanced by one token in one pass.
//!
//! The batched pass shares its body with prefill and differs from it in one
//! thing — each row has a cache of its own, and attends over that. So the
//! mistake to look for is the one that still produces plausible numbers: a
//! row reading another sequence's history, or its own at the wrong position.
//! Sequences of different lengths are what tell those apart from the right
//! answer, and the loop over sequences, one token at a time, is the right
//! answer to hold it to.

use kvad::model::{llama, CpuSession, Json, KvCache, Session, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::collections::BTreeMap;
use std::path::PathBuf;

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

const VOCAB: usize = 64;

/// A small Llama: grouped-query attention, so the query and the cache have
/// different widths, and — for `qwen3` — the per-head norms on Q and K.
///
/// Every matrix has a row length that is a multiple of 32, because anything
/// narrower the quantiser leaves as f32, and a tiny model that quietly
/// skipped quantisation would test the wrong kernels.
fn tiny(model_type: &str, precision: Precision, tag: &str) -> (Box<dyn Transformer>, Spec) {
    let (hidden, n_head, n_kv_head, n_layer, inter) = (64usize, 4usize, 2usize, 3usize, 128usize);
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

    let path = std::env::temp_dir().join(format!("kvad-each-{}-{tag}.safetensors", std::process::id()));
    write_safetensors(&path, &t);
    let ckpt = Checkpoint::open(std::slice::from_ref(&path)).unwrap();
    let spec = Spec::from_config(Json::new(config)).unwrap();
    let live = Live::new(&ckpt, precision);
    let model = llama::Model::load(&live, spec.clone()).unwrap();
    let unread = live.unread();
    assert!(unread.is_empty(), "{tag}: the loader never read {unread:?}");
    let _ = std::fs::remove_file(&path);
    (Box::new(model), spec)
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "{what}: worst disagreement {worst}");
    assert!(a.iter().all(|v| v.is_finite()), "{what}: not finite");
}

/// Three conversations of different lengths, and what each says next.
const PROMPTS: [&[u32]; 3] = [&[3, 17, 8, 0, 29, 41, 5], &[12], &[60, 2, 2, 33]];
const NEXT: [[u32; 3]; 4] = [[7, 7, 7], [1, 44, 19], [63, 0, 30], [9, 21, 9]];

/// The claim: a pass over several sequences is each sequence's own pass.
///
/// The sequences are different lengths on purpose. With equal lengths a row
/// that read its neighbour's history at its own position, or its own history
/// at its neighbour's, would rotate by the right angle and attend over the
/// right count, and only the numbers would be wrong — by an amount a random
/// model does not make obvious.
#[test]
fn several_sequences_at_once_say_what_each_says_alone() {
    for (model_type, precision) in [
        ("llama", Precision::F32),
        ("llama", Precision::Q8),
        ("qwen3", Precision::F32),
        ("qwen3", Precision::Q8),
    ] {
        let tag = format!("{model_type}-{precision}");
        let (model, spec) = tiny(model_type, precision, &tag);

        let prefilled = || -> Vec<KvCache> {
            PROMPTS
                .iter()
                .map(|prompt| {
                    let mut cache = KvCache::new(&spec);
                    model.forward_batch(prompt, &mut cache);
                    cache
                })
                .collect()
        };
        let (mut alone, mut together) = (prefilled(), prefilled());

        for (step, tokens) in NEXT.iter().enumerate() {
            let want: Vec<Vec<f32>> =
                tokens.iter().zip(alone.iter_mut()).map(|(&t, cache)| model.forward(t, cache)).collect();
            let mut each: Vec<&mut KvCache> = together.iter_mut().collect();
            let got = model.forward_each(tokens, &mut each);

            assert_eq!(got.len(), tokens.len() * VOCAB, "{tag}: a row of logits for each sequence");
            for (i, row) in want.iter().enumerate() {
                close(&got[i * VOCAB..(i + 1) * VOCAB], row, &format!("{tag} step {step} sequence {i}"));
                assert_eq!(together[i].len, alone[i].len, "{tag}: sequence {i} is a different length");
            }
        }
    }
}

/// One sequence is a batch, and nothing is one too.
#[test]
fn a_batch_of_one_is_a_token_and_a_batch_of_none_is_nothing() {
    let (model, spec) = tiny("llama", Precision::F32, "one");
    let (mut a, mut b) = (KvCache::new(&spec), KvCache::new(&spec));
    model.forward_batch(PROMPTS[0], &mut a);
    model.forward_batch(PROMPTS[0], &mut b);
    close(&model.forward_each(&[11], &mut [&mut a]), &model.forward(11, &mut b), "one");
    assert!(model.forward_each(&[], &mut []).is_empty());
}

/// The session's batched step is the same conversations' own steps, and
/// leaves the conversation the session is holding alone.
#[test]
fn a_session_advances_conversations_it_has_set_aside() {
    let build = |tag: &str| {
        let (model, _) = tiny("qwen3", Precision::Q8, tag);
        CpuSession::new(model, Precision::Q8)
    };
    // Two sessions over the same weights: one steps each conversation in
    // turn by taking it back, the other all of them at once.
    let (mut one, mut all) = (build("session-one"), build("session-all"));

    let mut set_aside = |session: &mut CpuSession| -> Vec<kvad::model::Parked> {
        let parked = PROMPTS
            .iter()
            .map(|prompt| {
                session.forward(prompt).unwrap();
                session.park().unwrap()
            })
            .collect();
        // And one it keeps, which the batched step must not disturb.
        session.forward(&[5, 6]).unwrap();
        parked
    };
    let (mut theirs, mut ours) = (set_aside(&mut one), set_aside(&mut all));

    for (step, tokens) in NEXT.iter().enumerate() {
        let got = all.forward_each(tokens, &mut ours).unwrap();
        assert_eq!(all.cached(), 2, "the session's own conversation moved");

        let own = one.park().unwrap();
        for (i, (&token, cache)) in tokens.iter().zip(theirs.iter_mut()).enumerate() {
            one.resume(std::mem::replace(cache, Box::new(()))).unwrap();
            let want = one.forward(&[token]).unwrap();
            *cache = one.park().unwrap();
            close(&got[i], &want, &format!("step {step} conversation {i}"));
        }
        one.resume(own).unwrap();
    }

    // What was advanced is still each conversation's own: taken back, it is
    // as long as its prompt and the steps since.
    for (cache, prompt) in ours.into_iter().zip(PROMPTS) {
        all.resume(cache).unwrap();
        assert_eq!(all.cached(), prompt.len() + NEXT.len());
    }
}

/// A prompt read a pass at a time is the prompt read in one call: the same
/// logits to the last bit, and the same cache.
///
/// The engine reads a long prompt between other replies' tokens, a pass at
/// a time, and promises that this changes when the prompt is read and not
/// what the model makes of it. That holds only if the passes are cut where
/// the backend would have cut them itself, which is what
/// [`Session::prefill_pass`] is for. Exact, not close: at q8 a pass cut
/// somewhere else is a different order of additions, and rounding turns
/// that into a different reply.
#[test]
fn a_prompt_read_a_pass_at_a_time_is_the_prompt_read_at_once() {
    let prompt: Vec<u32> = (0..23u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    for (model_type, precision) in [("llama", Precision::F32), ("qwen3", Precision::Q8)] {
        let build = |tag: &str| {
            let (model, _) = tiny(model_type, precision, tag);
            // Five tokens a pass: four whole passes and three left over.
            CpuSession::new(model, precision).reading(5)
        };
        let (mut once, mut passes) = (build("read-once"), build("read-passes"));

        let want = once.forward(&prompt).unwrap();

        let (mut at, mut got, mut count) = (0, Vec::new(), 0);
        while at < prompt.len() {
            let pass = passes.prefill_pass(prompt.len() - at);
            assert!((1..=prompt.len() - at).contains(&pass), "a pass of {pass} with {} left", prompt.len() - at);
            got = passes.forward(&prompt[at..at + pass]).unwrap();
            // Something else's token, between the passes: a conversation
            // set aside is advanced and the prompt's cache is left alone.
            at += pass;
            count += 1;
        }
        assert_eq!(count, 5, "{model_type}: the prompt was not read in passes");
        assert_eq!(passes.cached(), once.cached());
        assert_eq!(got, want, "{model_type} {precision}: a pass at a time changed what the model made of the prompt");

        // And the next token after it agrees too, so the caches do.
        assert_eq!(passes.forward(&[9]).unwrap(), once.forward(&[9]).unwrap());
    }
}
