//! A model trained here has a name, and the name has to work everywhere a
//! repo id does.
//!
//! The test that matters is the last one: train through `kvad train`, then
//! load the result by its bare name through the ordinary engine path and
//! require the same characters out as `nanograd` writes. That is the journey
//! a person makes, and it crosses every piece this feature touches — the
//! models home, name resolution, the checkpoint, the tokeniser, and the
//! engine's KV cache.
//!
//! These tests share one process, and so share its environment and its
//! working directory — and they change both. So they run one at a time,
//! behind [`alone`], which is also what points `XDG_DATA_HOME` at a scratch
//! directory before any of them can read it. Without that the first test to
//! run would train into the real models home, which it once did.

use kvad::quant::Precision;
use kvad::runtime::Llm;
use kvad::sampler::Sampler;
use kvad::train;
use kvad::weights;
use nanograd::checkpoint;
use nanograd::rng::Rng;
use nanograd::text::{self, CharTokenizer};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, Once};

/// The right to change the environment and the working directory. Every test
/// in this file takes it as its first line and holds it to the end.
fn alone() -> MutexGuard<'static, ()> {
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    static SET: Once = Once::new();
    // A test that failed poisoned nothing that matters here; the next one
    // still wants the lock.
    let guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    SET.call_once(|| {
        let root = scratch_root();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("kvad/models")).unwrap();
        std::env::set_var("XDG_DATA_HOME", &root);
    });
    guard
}

fn scratch_root() -> PathBuf {
    std::env::temp_dir().join(format!("kvad-trained-{}", std::process::id()))
}

fn models_home() -> PathBuf {
    scratch_root().join("kvad/models")
}

/// A text with one thing in it to learn, and a file holding it.
fn corpus_file(name: &str, text: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("kvad-{name}-{}.txt", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

const SENTENCE: &str = "the cat sat on the mat. ";

fn options(data: PathBuf, name: &str) -> train::Options {
    train::Options {
        data,
        name: Some(name.to_string()),
        sample: 0,
        seed: 3,
        training: nanograd::text::Training {
            steps: 300,
            batch: 4,
            lr: 3e-3,
            eval_every: 100,
            eval_windows: 10,
            threads: 1,
            save: None, // `train::run` fills this in from the name.
        },
        ..train::Options::default()
    }
}

/// A name is one path component, and everything else is refused before it can
/// be joined onto the models directory.
#[test]
fn a_model_name_is_one_word_and_cannot_leave_its_home() {
    let _alone = alone();
    assert!(weights::is_model_name("shakespeare"));
    assert!(weights::is_model_name("my-model.v2"));
    for bad in ["", ".", "..", "a/b", "../escape", "/etc", "./here", "a/../b"] {
        assert!(!weights::is_model_name(bad), "`{bad}` was accepted as a model name");
    }

    // And the lookup refuses them too, even where such a directory exists:
    // `models/../..` is a real directory on any machine.
    for bad in ["..", "../..", "a/b"] {
        assert!(weights::trained_dir(bad).is_none(), "`{bad}` resolved to a directory");
    }
}

/// The three ways of naming a model, in the order they are tried.
#[test]
fn a_directory_that_exists_wins_over_a_name_and_a_name_over_the_hub() {
    let _alone = alone();
    let home = models_home();
    let name = "precedence";
    std::fs::create_dir_all(home.join(name)).unwrap();

    // Nothing is called `precedence` here, so the trained model is found.
    let real = std::fs::canonicalize(home.join(name)).unwrap();
    assert_eq!(weights::local_dir(name), Some(real));
    assert_eq!(weights::model_id(name), name);
    // However the same model is spelled, it answers to the one name -- which
    // is what the quantised-weight cache is keyed on.
    assert_eq!(weights::model_id(home.join(name).to_str().unwrap()), name);

    // A directory of that name where we are standing takes it back.
    let elsewhere = std::env::temp_dir().join(format!("kvad-precedence-{}", std::process::id()));
    std::fs::create_dir_all(elsewhere.join(name)).unwrap();
    let here = std::env::current_dir().unwrap();
    std::env::set_current_dir(&elsewhere).unwrap();
    let won = weights::local_dir(name);
    std::env::set_current_dir(here).unwrap();
    assert_eq!(won, Some(std::fs::canonicalize(elsewhere.join(name)).unwrap()));

    // A directory that is not in the models home keeps its path for a name.
    // Two models called `out`, in two places, must not be one model to the
    // quantised-weight cache.
    let outside = std::fs::canonicalize(elsewhere.join(name)).unwrap();
    assert_eq!(weights::model_id(outside.to_str().unwrap()), outside.display().to_string());

    // A repo id is not a model name and is left for the Hub.
    assert!(weights::local_dir("openai-community/gpt2").is_none());
    assert_eq!(weights::model_id("openai-community/gpt2"), "openai-community/gpt2");

    // A bare word that is neither says so, and says where to look.
    let error = weights::fetch_with("no-such-model", &mut |_| {}).err().unwrap().to_string();
    assert!(error.contains("not a model trained here") && error.contains("kvad ls"), "{error}");

    std::fs::remove_dir_all(home.join(name)).unwrap();
    std::fs::remove_dir_all(elsewhere).unwrap();
}

/// The whole of `kvad train --name NAME` followed by `kvad run --model NAME`:
/// a text file in, and the engine writing the same characters `nanograd`
/// writes from the same weights.
#[test]
fn a_model_trained_by_name_runs_by_name() {
    let _alone = alone();
    let home = models_home();
    let name = "cats";
    let data = corpus_file("cats", &SENTENCE.repeat(40));

    let summary = train::run(&options(data, name), &mut |_| {}).unwrap();
    assert_eq!(summary.handle, name, "it should be reachable by the name it was given");
    assert_eq!(summary.dir, std::fs::canonicalize(home.join(name)).unwrap());
    for file in ["model.safetensors", "config.json", "tokenizer.json"] {
        assert!(summary.dir.join(file).is_file(), "no {file} in {}", summary.dir.display());
    }
    // It is listed, by name, as a model trained here rather than downloaded.
    assert!(kvad::hub::trained_models().iter().any(|m| m.id == name && m.complete));

    // What `nanograd` writes from the saved weights...
    let mut model = checkpoint::load(&summary.dir).unwrap();
    let tok = CharTokenizer::load(&summary.dir).unwrap();
    let prompt = "the cat";
    let count = 24;
    let mut rng = Rng::new(11);
    let ours = tok.decode(&text::generate(&mut model, &tok.encode(prompt).unwrap(), count, 0.0, &mut rng));
    // If it never learned the sentence, agreeing about it would prove little.
    assert_eq!(ours, "the cat sat on the mat. the cat", "the model did not learn its text");

    // ...and what the engine writes, given nothing but the name.
    let mut llm = Llm::load_with(name, Precision::F32, &mut |_| {}).unwrap();
    let ids = llm.encode(prompt).unwrap();
    let (stats, ids) = llm.generate(&ids, &mut Sampler::new(0.0, 0, 1.0, 0), count, |_| true).unwrap();
    assert_eq!(stats.generated_tokens, count);
    assert_eq!(llm.decode(&ids).unwrap(), ours);

    std::fs::remove_dir_all(&summary.dir).unwrap();
}

/// `--from` and the one thing it cannot do. The vocabulary was fixed when the
/// model was first trained, and there is no row in the embedding table for a
/// character that was not in that text; the error has to say which one.
#[test]
fn training_further_keeps_the_vocabulary_it_started_with() {
    let _alone = alone();
    let name = "further";
    let first = corpus_file("further-a", &SENTENCE.repeat(60));
    let mut opts = options(first, name);
    opts.training.steps = 40;
    let before = train::run(&opts, &mut |_| {}).unwrap();

    // More of the same text: fine, and it lands back in the same directory.
    let more = corpus_file("further-b", &"the mat sat on the cat. ".repeat(60));
    let again = train::Options { data: more, from: Some(name.into()), name: None, ..options(PathBuf::new(), name) };
    let mut again = again;
    again.name = None;
    again.training.steps = 40;
    let after = train::run(&again, &mut |_| {}).unwrap();
    assert_eq!(after.dir, before.dir);
    assert_eq!(after.handle, name);

    // A character it has never seen: refused, and named.
    let strange = corpus_file("further-c", &"the cat sat on the mat\u{2603} ".repeat(60));
    let mut refused = again;
    refused.data = strange;
    let error = train::run(&refused, &mut |_| {}).unwrap_err().to_string();
    assert!(error.contains('\u{2603}') && error.contains("fixed at first training"), "{error}");

    // ...and the refusal left the model alone.
    assert!(checkpoint::load(&before.dir).is_ok());
    std::fs::remove_dir_all(&before.dir).unwrap();
}

/// A new model needs a name; an old one may be trained in place.
#[test]
fn a_run_with_nowhere_to_put_its_model_does_not_start() {
    let _alone = alone();
    let data = corpus_file("nowhere", &SENTENCE.repeat(60));
    let nameless = train::Options { name: None, ..options(data.clone(), "unused") };
    let error = train::run(&nameless, &mut |_| {}).unwrap_err().to_string();
    assert!(error.contains("needs a name"), "{error}");

    let escaping = train::Options { name: Some("../escape".into()), ..options(data.clone(), "unused") };
    let error = train::run(&escaping, &mut |_| {}).unwrap_err().to_string();
    assert!(error.contains("not a model name"), "{error}");

    let missing = train::Options { name: None, from: Some("no-such-model".into()), ..options(data, "unused") };
    let error = train::run(&missing, &mut |_| {}).unwrap_err().to_string();
    assert!(error.contains("not a model on this machine"), "{error}");
}
