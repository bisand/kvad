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
use kvad::runtime::Llm;
use kvad::sampler::Sampler;
use kvad::weights::{self, Checkpoint, ModelFiles};
use nanograd::checkpoint;
use nanograd::model::{Gpt, GptConfig};
use nanograd::optim::AdamW;
use nanograd::rng::Rng;
use nanograd::text::{generate, train_step, CharTokenizer};
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

    // The scoring path, which keeps every row rather than the last. It is the
    // same batch through the same blocks with the output head run over all of
    // it, so every position must match what `nanograd` computed — and unlike
    // the two checks above, this one compares the whole matrix.
    let all = engine.forward_batch_all(&tokens, &mut KvCache::new(&spec));
    assert_eq!(all.len(), IDS.len() * spec.vocab_size);
    for i in 0..IDS.len() {
        let row = &all[i * spec.vocab_size..(i + 1) * spec.vocab_size];
        let off = crate::gap(ours.row(i), row);
        assert!(off < 1e-5, "scoring position {i}: logits differ by {off:e} of their size");
    }

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

/// The whole journey, as a user makes it: train a model on a text, save it,
/// and hand the directory to the engine by name, exactly as `kvad run --model
/// DIR` does. Everything is real — the engine's tokeniser reads the prompt,
/// its KV cache carries the generation — and with sampling switched off both
/// sides must write the same characters.
#[test]
fn a_trained_model_runs_from_its_directory() {
    let text = "the cat sat on the mat. ".repeat(40);
    let tok = CharTokenizer::from_text(&text);
    let tokens = tok.encode(&text).unwrap();
    let config = GptConfig { vocab: tok.vocab(), context: 32, d_model: 32, n_heads: 4, n_layers: 2 };

    let mut rng = Rng::new(3);
    let mut model = Gpt::new(config, &mut rng);
    let mut opt = AdamW::new(3e-3);
    for _ in 0..300 {
        train_step(&mut model, &mut opt, &tokens, 4, &mut rng);
    }

    let dir = scratch("journey");
    checkpoint::save(&dir, &mut model).unwrap();
    tok.save(&dir).unwrap();

    let prompt = "the cat";
    let count = 24;
    let ours = tok.decode(&generate(&mut model, &tok.encode(prompt).unwrap(), count, 0.0, &mut rng));
    // If it has not learned the sentence, agreeing about it proves little.
    assert_eq!(ours, "the cat sat on the mat. the cat", "the model did not learn its text");

    let mut llm = Llm::load_with(dir.to_str().unwrap(), Precision::F32, &mut |_| {}).unwrap();
    assert!(!llm.is_instruct());
    let ids = llm.encode(prompt).unwrap();
    let (stats, ids) = llm.generate(&ids, &mut Sampler::new(0.0, 0, 1.0, 0), count, |_| true).unwrap();
    assert_eq!(stats.generated_tokens, count);
    assert_eq!(llm.decode(&ids).unwrap(), ours);

    // Scoring, on a model whose whole world is one sentence. Its own text
    // should surprise it far less than a rearrangement of the same
    // characters — which is the only way to compare, since a character
    // tokeniser has no token for anything it was not trained on.
    let learnt = llm.perplexity(&text, 32, |_, _| true).unwrap();
    let shuffled = llm.perplexity(&"tam. eht no tas tac eht ".repeat(40), 32, |_, _| true).unwrap();
    assert!(learnt.scored > 0 && learnt.windows > 1);
    assert!(
        learnt.perplexity < shuffled.perplexity / 2.0,
        "the sentence it was trained on scored {:.2} and nonsense scored {:.2}",
        learnt.perplexity,
        shuffled.perplexity
    );
    // Perplexity is the exponential of the mean surprise, and bits are the
    // same number in another base. If those three ever disagree, one of them
    // is being computed twice.
    assert!((learnt.perplexity - learnt.nats.exp()).abs() < 1e-9);
    assert!((learnt.bits_per_token * std::f64::consts::LN_2 - learnt.nats).abs() < 1e-9);
    // Every window but its first token, and no window is scored twice.
    assert_eq!(learnt.scored, learnt.tokens - learnt.windows);

    // And a model that has learnt one sentence cannot be surprised by much:
    // under two nats is generous for a vocabulary this small.
    assert!(learnt.perplexity < 2.0, "{}", learnt.perplexity);

    // The tokeniser inspector, on the same model. A character tokeniser makes
    // this easy to check: one token per character, in order, covering the
    // whole string.
    let split = llm.tokenize("the cat").unwrap();
    assert_eq!(split.len(), 7);
    assert_eq!(split.iter().map(|t| t.piece.as_str()).collect::<String>(), "the cat");
    assert_eq!((split[0].start, split[0].end), (0, 1));
    assert_eq!(split[3].piece, " ");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// A fetch reports to two callbacks at once, and a model that is already
/// here downloads nothing.
///
/// The second half of that is what a progress bar needs to be told: a local
/// model finishes instantly and must not leave a bar at 0%. The download path
/// itself needs the network and so is not tested here.
#[test]
fn a_fetch_reports_in_words_and_in_events() {
    let dir = scratch("watched");
    let mut model = model();
    checkpoint::save(&dir, &mut model).unwrap();
    CharTokenizer::from_text("abc").save(&dir).unwrap();

    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let watch = {
        let seen = std::sync::Arc::clone(&seen);
        weights::Watcher::new(move |f| seen.lock().unwrap().push(f))
    };
    let mut lines = Vec::new();
    let files =
        weights::fetch_watched(dir.to_str().unwrap(), &mut |l| lines.push(l.to_string()), &watch)
            .unwrap();
    // Resolved, so compared by name rather than by the path as typed.
    assert_eq!(files.weights.len(), 1);
    assert!(files.weights[0].ends_with("model.safetensors"));

    assert_eq!(*seen.lock().unwrap(), [weights::Fetch::Local]);
    assert_eq!(lines, ["a directory on this machine; nothing to fetch"]);

    // And the watcher nobody supplied is the one `fetch_with` uses, which is
    // the same call with the events dropped.
    assert!(!weights::Watcher::none().is_listening());
    assert!(watch.is_listening());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_directory_is_a_model_only_if_the_files_are_there() {
    let dir = scratch("files");
    let mut model = model();
    checkpoint::save(&dir, &mut model).unwrap();

    // The weights and the config, but no tokeniser: say which file, and where.
    let error = ModelFiles::from_dir(&dir).err().unwrap().to_string();
    assert!(error.contains("tokenizer.json") && error.contains(dir.to_str().unwrap()), "{error}");

    CharTokenizer::from_text("abc").save(&dir).unwrap();
    let files = ModelFiles::from_dir(&dir).unwrap();
    assert_eq!(files.weights, [dir.join("model.safetensors")]);
    assert!(files.tokenizer_config.is_none(), "a base model: no chat template to find");

    // A model too large for one file: the index names its shards, many
    // tensors to each, and every shard has to be there.
    std::fs::rename(dir.join("model.safetensors"), dir.join("part-b.safetensors")).unwrap();
    let index = r#"{"weight_map":{"x":"part-b.safetensors","y":"part-a.safetensors","z":"part-b.safetensors"}}"#;
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
    let error = ModelFiles::from_dir(&dir).err().unwrap().to_string();
    assert!(error.contains("part-a.safetensors"), "{error}");
    std::fs::write(dir.join("part-a.safetensors"), b"").unwrap();
    let files = ModelFiles::from_dir(&dir).unwrap();
    assert_eq!(files.weights, [dir.join("part-a.safetensors"), dir.join("part-b.safetensors")]);

    // However the directory is spelled, it is one model with one name...
    let spelled = dir.join("..").join(dir.file_name().unwrap());
    assert!(weights::is_local(spelled.to_str().unwrap()));
    assert_eq!(weights::model_id(spelled.to_str().unwrap()), weights::model_id(dir.to_str().unwrap()));
    // ...a repo id is left alone, and a path to nowhere is not sent to the Hub.
    assert_eq!(weights::model_id("openai-community/gpt2"), "openai-community/gpt2");
    let error = weights::fetch_with("./no/such/model", &mut |_| {}).err().unwrap().to_string();
    assert!(error.contains("no such directory"), "{error}");

    std::fs::remove_dir_all(&dir).unwrap();
}
