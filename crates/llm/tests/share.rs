//! Load only your share.
//!
//! A machine that holds layers 20 to 49 of a model has no use for the
//! other three hundred gigabytes of it, and must not need them: not to
//! download, not to open, and not in the file it keeps its quantised
//! weights in. These tests make the rest of the checkpoint unusable and
//! check that a shard does not notice; and they check the trap on the
//! other side, that what a shard leaves on disk is never taken for the
//! model.

#![cfg(all(feature = "cluster", feature = "arch-llama"))]

mod common;

use kvad::cluster::session::PipelineSession;
use kvad::model::{CpuSession, Shard, Spec, Transformer};
use kvad::qcache::{self, Mapped};
use kvad::quant::Precision;
use kvad::weights::{self, ModelFiles};
use std::path::{Path, PathBuf};

/// The small Llama of the other tests, four layers, filed as a large model
/// is: the embedding and layers 0 and 1 in one file, layers 2 and 3 in a
/// second, the final norm and the head in a third.
fn published(tag: &str) -> (PathBuf, Spec) {
    let (tensors, config) = common::llama_parts("llama");
    let dir = common::sharded_dir(tag, tensors, &config, &|name| match name {
        n if n.contains("layers.2.") || n.contains("layers.3.") => 2,
        n if n.contains("layers.") || n.contains("embed_tokens") => 1,
        _ => 3,
    });
    let spec = Spec::from_json(&dir.join("config.json")).unwrap();
    (dir, spec)
}

fn names(files: &ModelFiles) -> Vec<String> {
    files.weights.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect()
}

fn fetch(dir: &Path, shard: &Shard) -> ModelFiles {
    weights::fetch_for(dir.to_str().unwrap(), &|tensor| shard.wants(tensor), &mut |_| {}, &weights::Watcher::none()).unwrap()
}

/// The quantised cache is one directory for the whole process, named by a
/// variable, so the tests that move it take turns.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Cache {
    dir: PathBuf,
    _turn: std::sync::MutexGuard<'static, ()>,
}

impl Cache {
    fn new(tag: &str) -> Cache {
        let turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("kvad-share-cache-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("KVAD_QUANT_CACHE", &dir);
        Cache { dir, _turn: turn }
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        std::env::remove_var("KVAD_QUANT_CACHE");
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Load a shard and keep what the loader said on the way.
fn load(repo: &str, files: &ModelFiles, spec: &Spec, precision: Precision, shard: &Shard) -> (Box<dyn Transformer>, Vec<String>) {
    let mut said = Vec::new();
    let model = qcache::load_shard(repo, files, spec, precision, shard, &mut |line| said.push(line.to_string())).unwrap();
    (model, said)
}

/// Which files a run of layers needs is known from the index, before any
/// of them is opened.
#[test]
fn a_shard_asks_for_the_files_its_layers_are_in() {
    let (dir, spec) = published("files");
    let all = ["model-00001.safetensors", "model-00002.safetensors", "model-00003.safetensors"];
    assert_eq!(names(&weights::fetch_with(dir.to_str().unwrap(), &mut |_| {}).unwrap()), all);
    assert_eq!(names(&fetch(&dir, &Shard::whole(&spec))), all);

    assert_eq!(names(&fetch(&dir, &Shard { layers: 2..4, ends: false })), ["model-00002.safetensors"]);
    assert_eq!(names(&fetch(&dir, &Shard { layers: 0..1, ends: false })), ["model-00001.safetensors"]);
    // A run across two files is both, and the ends are where they are: the
    // table beside the first layers, the head in a file of its own.
    assert_eq!(names(&fetch(&dir, &Shard { layers: 1..3, ends: false })), ["model-00001.safetensors", "model-00002.safetensors"]);
    assert_eq!(names(&fetch(&dir, &Shard { layers: 0..0, ends: true })), ["model-00001.safetensors", "model-00003.safetensors"]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The proof that a shard opens only its own: every other file of the
/// checkpoint is taken away, and the shard loads and runs as it did.
#[test]
fn a_shard_loads_with_the_rest_of_the_checkpoint_gone() {
    let (dir, spec) = published("gone");
    let repo = dir.to_str().unwrap();
    let whole = weights::fetch_with(repo, &mut |_| {}).unwrap();
    let mut reference = CpuSession::new(qcache::load(repo, &whole, &spec, Precision::F32, &mut |_| {}).unwrap(), Precision::F32).reading(8);

    // The worker's machine: layers 2 and 3, and nothing else on its disk.
    let (later, ends) = (Shard { layers: 2..4, ends: false }, Shard { layers: 0..2, ends: true });
    let far = dir.with_file_name(format!("{}-far", dir.file_name().unwrap().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&far);
    std::fs::create_dir_all(&far).unwrap();
    for name in ["config.json", "tokenizer.json", "model.safetensors.index.json", "model-00002.safetensors"] {
        std::fs::copy(dir.join(name), far.join(name)).unwrap();
    }
    let theirs = fetch(&far, &later);
    assert_eq!(names(&theirs), ["model-00002.safetensors"]);
    let (far_model, _) = load(far.to_str().unwrap(), &theirs, &spec, Precision::F32, &later);

    // The whole model cannot be had there, and says which file it wanted.
    let missing = weights::fetch_with(far.to_str().unwrap(), &mut |_| {}).err().expect("two files are missing").to_string();
    assert!(missing.contains("model-00001.safetensors"), "{missing}");

    // The coordinator's: the files with the ends and layers 0 and 1.
    let ours = fetch(&dir, &ends);
    assert_eq!(names(&ours), ["model-00001.safetensors", "model-00003.safetensors"]);
    let (near_model, _) = load(repo, &ours, &spec, Precision::F32, &ends);

    let mut split = PipelineSession::local(vec![near_model, far_model], Precision::F32).unwrap().reading(8);
    common::same_answers(&mut reference, &mut split, "each shard from its own files");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&far);
}

/// A shard's quantised weights are kept, in a file that is that shard's:
/// the second load maps it, and nothing else ever does.
///
/// The trap this is here for: a cache file's reader takes a name the file
/// does not hold for a tensor the model does not have. A file written by
/// layers 2 and 3 and mapped for the whole model would fail on layer 0 —
/// or, for an untied model, answer that there is no head and run with the
/// embedding table instead.
#[test]
fn a_shards_cache_is_never_taken_for_the_models() {
    let _cache = Cache::new("identity");
    let (dir, spec) = published("identity");
    let repo = dir.to_str().unwrap();
    let files = weights::fetch_with(repo, &mut |_| {}).unwrap();
    let precision = Precision::Q8;
    let (whole, later, ends) = (Shard::whole(&spec), Shard { layers: 2..4, ends: false }, Shard { layers: 0..2, ends: true });
    let said_it = |said: &[String], what: &str| said.iter().any(|line| line.contains(what));

    // First a shard, which quantises and keeps; then again, which maps.
    let (_, said) = load(repo, &files, &spec, precision, &later);
    assert!(said_it(&said, "quantising to q8 (first load)") && said_it(&said, "cached for next time"), "{said:?}");
    let (later_mapped, said) = load(repo, &files, &spec, precision, &later);
    assert!(said_it(&said, "mapping q8 weights") && !said_it(&said, "stale"), "{said:?}");

    // Then the whole model, on the same machine, with that file sitting
    // there. It is not mapped, not even looked at as a candidate: the
    // model's file is another file.
    let later_path = qcache::path_for_shard(repo, precision, &spec, &later);
    let whole_path = qcache::path_for_shard(repo, precision, &spec, &whole);
    assert!(later_path.is_file() && later_path != whole_path && !whole_path.exists());
    assert_eq!(whole_path, qcache::path_for(repo, precision), "the model's file is the one it always was");
    let (model, said) = load(repo, &files, &spec, precision, &whole);
    assert!(said_it(&said, "quantising to q8 (first load)") && !said_it(&said, "mapping"), "{said:?}");
    assert!(whole_path.is_file());
    assert_eq!(model.param_count(), 4 * later_mapped.param_count() / 2 + 2 * common::VOCAB * 64 + 64);

    // And were a shard's file ever put where another's is looked for, it
    // would be refused for what it says it is.
    let as_whole = Mapped::open_shard(&later_path, repo, &files.weights, &spec, precision, &whole).err().expect("refused");
    assert!(as_whole.to_string().contains("`shard` changed"), "{as_whole}");
    assert!(Mapped::open_shard(&later_path, repo, &files.weights, &spec, precision, &ends).is_err());
    assert!(Mapped::open_shard(&whole_path, repo, &files.weights, &spec, precision, &later).is_err(), "nor the model's for a shard's");
    assert!(Mapped::open_shard(&later_path, repo, &files.weights, &spec, precision, &later).unwrap().is_some());
    assert!(Mapped::open(&whole_path, repo, &files.weights, &spec, precision).unwrap().is_some());

    // Mapped or quantised just now, whole or in shards: one model.
    let (ends_fresh, _) = load(repo, &files, &spec, precision, &ends);
    let mut split = PipelineSession::local(vec![ends_fresh, later_mapped], precision).unwrap().reading(8);
    let mut reference = CpuSession::new(qcache::load(repo, &files, &spec, precision, &mut |_| {}).unwrap(), precision).reading(8);
    common::same_answers(&mut reference, &mut split, "a mapped shard beside a fresh one");
    let _ = std::fs::remove_dir_all(&dir);
}
