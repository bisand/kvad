//! A model saved by `nanograd` is a GPT-2 checkpoint, and this is the test
//! that says so.
//!
//! `nanograd` can save a model and load it back bit for bit, and that proves
//! less than it seems to: a writer and a reader that share a misunderstanding
//! — query and key in the wrong thirds of the fused matrix, a head transposed
//! the wrong way — agree with each other perfectly. The only real check of a
//! file format is a second implementation. Here there is one: this crate's
//! GPT-2, written to run OpenAI's weights and knowing nothing about
//! `nanograd`. Both are handed the same tokens, and have to produce the same
//! logits.

use kvad::model::gpt2;
use kvad::model::{KvCache, Spec, Transformer};
use kvad::qcache::Live;
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use nanograd::checkpoint;
use nanograd::model::{Gpt, GptConfig};
use nanograd::rng::Rng;
use nanograd::text::CharTokenizer;
use std::path::PathBuf;

const CONFIG: GptConfig = GptConfig { vocab: 23, context: 12, d_model: 16, n_heads: 4, n_layers: 3 };
const IDS: [usize; 12] = [5, 22, 0, 5, 9, 9, 17, 1, 22, 3, 5, 11];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kvad-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A freshly initialised model has every bias at 0 and every gamma at 1, and
/// would pass this test with its biases saved in the wrong order. So move
/// every number somewhere of its own first.
fn model() -> Gpt {
    let mut rng = Rng::new(7);
    let mut model = Gpt::new(CONFIG, &mut rng);
    for p in model.params() {
        p.value.iter_mut().for_each(|v| *v += 0.3 * rng.normal());
    }
    model
}

/// The largest difference between two rows of logits, relative to their size.
///
/// Not-a-number counts as infinitely far. `f32::max` quietly prefers the
/// other argument to a NaN, so without this a model that loaded as garbage
/// and produced nothing but NaN would measure a gap of zero — and did, when
/// the floats were deliberately written big-endian to see who would notice.
fn gap(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return f32::INFINITY;
    }
    let size = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs())) / size
}

#[test]
fn the_engine_computes_what_nanograd_computes() {
    let dir = scratch("logits");
    let mut trained = model();
    checkpoint::save(&dir, &mut trained).unwrap();
    let ours = trained.forward(&IDS);

    let spec = Spec::from_json(&dir.join(checkpoint::CONFIG_FILE)).unwrap();
    assert!(!spec.tie_embeddings, "the config must say the head is its own tensor");
    assert_eq!((spec.n_ctx, spec.vocab_size, spec.head_dim), (12, 23, 4));
    let ckpt = Checkpoint::open(&[dir.join(checkpoint::WEIGHTS_FILE)]).unwrap();
    let engine = gpt2::Model::load(&Live::new(&ckpt, Precision::F32), spec.clone()).unwrap();
    assert_eq!(engine.param_count(), trained.param_count());

    // One token at a time, as in generation: position i reads the cache that
    // positions 0..i left behind, and must match row i of our single pass.
    //
    // The two sum in different orders, so the floats differ in their last
    // bits: measured, the worst row is 5e-7. Every mistake in the layout that
    // was tried on purpose measured above 0.2, and the subtlest thing tried,
    // a wrong epsilon in config.json, 1.2e-3.
    let mut cache = KvCache::new(&spec);
    for (i, &id) in IDS.iter().enumerate() {
        let theirs = engine.forward(id as u32, &mut cache);
        let gap = gap(ours.row(i), &theirs);
        assert!(gap < 1e-5, "position {i}: logits differ by {gap:e} of their size");
    }

    // And the prompt path, which takes the whole sequence in one call.
    let tokens: Vec<u32> = IDS.iter().map(|&id| id as u32).collect();
    let theirs = engine.forward_batch(&tokens, &mut KvCache::new(&spec));
    let gap = gap(ours.row(IDS.len() - 1), &theirs);
    assert!(gap < 1e-5, "prompt: logits differ by {gap:e} of their size");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// The same question about the tokeniser: `nanograd` writes a `tokenizer.json`
/// and claims the `tokenizers` library reads it as a character tokeniser.
#[test]
fn the_tokenizers_library_agrees_on_every_id() {
    let text = "Two lines,\n\tone tab; \"quotes\", a back\\slash, a bell \u{7}, é → 😀 and  two spaces.\n";
    let dir = scratch("tokenizer");
    let ours = CharTokenizer::from_text(text);
    ours.save(&dir).unwrap();
    let theirs = tokenizers::Tokenizer::from_file(dir.join(nanograd::text::TOKENIZER_FILE)).unwrap();

    let ids = ours.encode(text).unwrap();
    let encoded = theirs.encode(text, false).unwrap();
    let their_ids: Vec<usize> = encoded.get_ids().iter().map(|&id| id as usize).collect();
    assert_eq!(their_ids, ids);
    assert_eq!(theirs.get_vocab_size(true), ours.vocab());

    let as_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    assert_eq!(theirs.decode(&as_u32, true).unwrap(), text);
    std::fs::remove_dir_all(&dir).unwrap();
}
