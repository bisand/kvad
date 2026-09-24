//! A model the Hub cache holds loads without the Hub.
//!
//! What this guards against was seen on a service restart with DNS for
//! huggingface.co timing out: the loader asked the Hub about every file of a
//! 14B model the cache held in full, `hf-hub` retried each request for three
//! minutes before falling back to the file it already had, and nothing was
//! loaded. The rule since is that a load reads the cache first and asks the
//! Hub only about what the cache cannot answer; `kvad pull` is what asks.
//!
//! The Hub here is a socket on this machine that counts connections and
//! answers none, with `HF_ENDPOINT` pointed at it. Counting rather than a
//! closed port, because a refused connection looks transient to `hf-hub`: it
//! retries, falls back to the cache, and the load succeeds a second later —
//! a test on a closed port passes with the bug still in. A count of zero is
//! the claim. The last test is the control that shows the count can be more.
//!
//! These tests share one process and change its environment, so they run one
//! at a time behind [`alone`], which also points every cache at a scratch
//! directory before any of them can read the real ones.

use kvad::quant::Precision;
use kvad::runtime::Llm;
use kvad::sampler::Sampler;
use kvad::train;
use kvad::weights::{self, Cached};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, Once, OnceLock};

/// Connections the stand-in Hub has seen.
static ASKED: AtomicUsize = AtomicUsize::new(0);

fn alone() -> MutexGuard<'static, ()> {
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    static SET: Once = Once::new();
    let guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    SET.call_once(|| {
        let root = scratch_root();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("kvad/models")).unwrap();
        std::env::set_var("XDG_DATA_HOME", &root);
        std::env::set_var("HF_HOME", root.join("hf"));
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");

        let hub = TcpListener::bind("127.0.0.1:0").unwrap();
        std::env::set_var("HF_ENDPOINT", format!("http://{}", hub.local_addr().unwrap()));
        std::thread::spawn(move || {
            for connection in hub.incoming() {
                ASKED.fetch_add(1, Ordering::SeqCst);
                drop(connection);
            }
        });
    });
    guard
}

fn scratch_root() -> PathBuf {
    std::env::temp_dir().join(format!("kvad-offline-{}", std::process::id()))
}

fn hub_cache() -> PathBuf {
    scratch_root().join("hf/hub")
}

/// A small model trained here, once, and where it was saved.
fn trained() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let data = scratch_root().join("corpus.txt");
        std::fs::write(&data, "the cat sat on the mat. ".repeat(40)).unwrap();
        let options = train::Options {
            data,
            name: Some("offline".to_string()),
            sample: 0,
            seed: 3,
            training: nervus::text::Training {
                steps: 30,
                batch: 4,
                lr: 3e-3,
                eval_every: 30,
                eval_windows: 2,
                threads: 1,
                save: None,
                ..Default::default()
            },
            ..train::Options::default()
        };
        train::run(&options, &mut |_| {}).unwrap().dir
    })
}

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// How the model is laid out in the cache.
struct Layout {
    /// A shard index and one shard, rather than `model.safetensors`.
    split: bool,
    /// `.no_exist` markers for the files the Hub would have said are missing,
    /// as `hf-hub` 1.0 writes them. Older caches, and other tools, have none.
    markers: bool,
    /// Leave `tokenizer.json` out, marked as nothing.
    without_tokenizer: bool,
}

/// Put the trained model into the Hub cache as `kvad-test/<name>` at
/// [`REVISION`], the way `hf-hub` lays a download out, and return its id.
fn cache_as(name: &str, layout: Layout) -> String {
    let repo = hub_cache().join(format!("models--kvad-test--{name}"));
    let _ = std::fs::remove_dir_all(&repo);
    let snapshot = repo.join("snapshots").join(REVISION);
    let missing = repo.join(".no_exist").join(REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/main"), REVISION).unwrap();

    let dir = trained();
    let copy = |from: &str, to: &str| std::fs::copy(dir.join(from), snapshot.join(to)).unwrap();
    let mut absent: Vec<&str> = vec!["model_index.json"];
    if !layout.without_tokenizer {
        copy("tokenizer.json", "tokenizer.json");
    }
    copy("config.json", "config.json");
    for optional in ["tokenizer_config.json", "generation_config.json"] {
        match dir.join(optional).is_file() {
            true => {
                copy(optional, optional);
            }
            // Without markers, an optional file nobody can vouch for would
            // send the loader to the Hub; that is the next test's business.
            false if !layout.markers => std::fs::write(snapshot.join(optional), "{}").map(|_| ()).unwrap(),
            false => absent.push(optional),
        }
    }
    match layout.split {
        false => {
            copy("model.safetensors", "model.safetensors");
        }
        true => {
            let shard = "model-00001-of-00001.safetensors";
            copy("model.safetensors", shard);
            let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
            let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
            let map: serde_json::Map<String, serde_json::Value> =
                meta.tensors().into_keys().map(|name| (name, shard.into())).collect();
            let index = serde_json::json!({ "metadata": {}, "weight_map": map });
            std::fs::write(snapshot.join("model.safetensors.index.json"), index.to_string()).unwrap();
            absent.push("model.safetensors");
        }
    }
    if layout.markers {
        std::fs::create_dir_all(&missing).unwrap();
        for file in absent {
            std::fs::write(missing.join(file), "").unwrap();
        }
    }
    format!("kvad-test/{name}")
}

/// What the model writes from a prompt, greedily.
fn continuation(llm: &mut Llm) -> String {
    let ids = llm.encode("the cat").unwrap();
    let (_, ids) = llm.generate(&ids, &mut Sampler::new(0.0, 0, 1.0, 0), 16, |_| true).unwrap();
    llm.decode(&ids).unwrap()
}

/// Load `repo` and require that the Hub was not asked, and that what loaded
/// writes what the model loaded from its own directory writes.
fn loads_offline(repo: &str) {
    let before = ASKED.load(Ordering::SeqCst);
    let mut lines = Vec::new();
    let mut llm = Llm::load_with(repo, Precision::F32, &mut |l| lines.push(l.to_string())).unwrap();
    assert_eq!(ASKED.load(Ordering::SeqCst), before, "the Hub was asked while loading {repo}: {lines:?}");
    assert!(
        lines.iter().any(|l| l.contains("nothing to fetch")),
        "a load from the cache should say it fetched nothing: {lines:?}"
    );

    let mut original = Llm::load_with(trained().to_str().unwrap(), Precision::F32, &mut |_| {}).unwrap();
    assert_eq!(continuation(&mut llm), continuation(&mut original), "the cache did not give back the same model");
}

/// The case that was seen: a split checkpoint, with the markers `hf-hub`
/// left for `model.safetensors` and `model_index.json`.
#[test]
fn a_split_model_in_the_cache_loads_without_the_hub() {
    let _alone = alone();
    let repo = cache_as("split", Layout { split: true, markers: true, without_tokenizer: false });
    loads_offline(&repo);
}

#[test]
fn a_single_file_model_in_the_cache_loads_without_the_hub() {
    let _alone = alone();
    let repo = cache_as("single", Layout { split: false, markers: false, without_tokenizer: false });
    loads_offline(&repo);
}

/// A cache written by something that leaves no `.no_exist` markers still
/// says the checkpoint is split, by holding every shard its index names.
#[test]
fn a_complete_set_of_shards_needs_no_markers() {
    let _alone = alone();
    let repo = cache_as("unmarked", Layout { split: true, markers: false, without_tokenizer: false });
    assert_eq!(weights::cached(&repo, "model.safetensors"), Cached::Unknown);
    loads_offline(&repo);
}

/// The three answers the cache can give, none of which asks the Hub.
#[test]
fn the_cache_says_here_absent_or_unknown() {
    let _alone = alone();
    let repo = cache_as("answers", Layout { split: true, markers: true, without_tokenizer: false });
    let before = ASKED.load(Ordering::SeqCst);
    assert!(matches!(weights::cached(&repo, "config.json"), Cached::Here(p) if p.is_file()));
    assert_eq!(weights::cached(&repo, "model_index.json"), Cached::Absent);
    assert_eq!(weights::cached(&repo, "vocab.txt"), Cached::Unknown);
    assert_eq!(weights::cached("kvad-test/never-downloaded", "config.json"), Cached::Unknown);
    assert!(weights::in_cache(&repo).is_some());
    assert!(weights::in_cache("kvad-test/never-downloaded").is_none());
    assert_eq!(ASKED.load(Ordering::SeqCst), before, "looking in the cache asked the Hub");
}

/// The control: a file the cache cannot answer for does go to the Hub, so
/// the zero counted above is the loader not asking, not the count missing
/// requests. And a pull asks even when the cache has everything.
#[test]
fn what_the_cache_cannot_answer_is_asked_of_the_hub() {
    let _alone = alone();
    let repo = cache_as("partial", Layout { split: true, markers: true, without_tokenizer: true });
    assert!(weights::in_cache(&repo).is_none(), "a model without its tokenizer is not in the cache");
    let before = ASKED.load(Ordering::SeqCst);
    let error = weights::fetch_with(&repo, &mut |_| {}).err().expect("there is no tokenizer to be had");
    assert!(ASKED.load(Ordering::SeqCst) > before, "the missing tokenizer was never asked for: {error}");

    let whole = cache_as("pulled", Layout { split: true, markers: true, without_tokenizer: false });
    let before = ASKED.load(Ordering::SeqCst);
    // The Hub is unreachable, and `hf-hub` falls back to the cache for each
    // file after its retries, so the pull succeeds; what matters is that it asked.
    let _ = weights::pull_watched(&whole, &mut |_| {}, &weights::Watcher::none());
    assert!(ASKED.load(Ordering::SeqCst) > before, "a pull did not ask the Hub");
}
