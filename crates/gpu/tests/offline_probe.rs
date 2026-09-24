//! Whether a model is an image pipeline is answered from the cache when the
//! cache can answer it.
//!
//! `is_pipeline` runs before every load the server does, and it used to ask
//! the Hub for `model_index.json` about any model that was not a pipeline —
//! every language model, cached or not. With the Hub unreachable that was
//! three minutes of `hf-hub` retries before the load could start. See
//! `crates/llm/tests/offline_cache.rs` for the stand-in Hub and why it counts
//! connections rather than refusing them.

use kvad::weights::Watcher;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, Once};

static ASKED: AtomicUsize = AtomicUsize::new(0);

fn alone() -> MutexGuard<'static, ()> {
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    static SET: Once = Once::new();
    let guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    SET.call_once(|| {
        let root = std::env::temp_dir().join(format!("kvad-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
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

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// A language model's files in the cache as `kvad-test/<name>`. Only their
/// presence is read, so their contents are placeholders. `missing` are the
/// files to leave `.no_exist` markers for.
fn cache_language_model(name: &str, missing: &[&str]) -> String {
    let hub = PathBuf::from(std::env::var("HF_HOME").unwrap()).join("hub");
    let repo = hub.join(format!("models--kvad-test--{name}"));
    let snapshot = repo.join("snapshots").join(REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/main"), REVISION).unwrap();
    for file in ["config.json", "tokenizer.json", "model.safetensors", "tokenizer_config.json", "generation_config.json"] {
        std::fs::write(snapshot.join(file), "{}").unwrap();
    }
    if !missing.is_empty() {
        let markers = repo.join(".no_exist").join(REVISION);
        std::fs::create_dir_all(&markers).unwrap();
        for file in missing {
            std::fs::write(markers.join(file), "").unwrap();
        }
    }
    format!("kvad-test/{name}")
}

#[test]
fn a_cached_language_model_is_not_a_pipeline_and_the_hub_is_not_asked() {
    let _alone = alone();
    let marked = cache_language_model("marked", &["model_index.json"]);
    let unmarked = cache_language_model("unmarked", &[]);
    let before = ASKED.load(Ordering::SeqCst);
    assert!(!kvad_gpu::image::is_pipeline(&marked, &Watcher::none()));
    assert!(!kvad_gpu::image::is_pipeline(&unmarked, &Watcher::none()));
    assert_eq!(ASKED.load(Ordering::SeqCst), before, "the Hub was asked about a model the cache holds");
}

/// The control: a model that is not here is still asked about.
#[test]
fn a_model_that_is_not_here_is_asked_about() {
    let _alone = alone();
    let before = ASKED.load(Ordering::SeqCst);
    assert!(!kvad_gpu::image::is_pipeline("kvad-test/not-downloaded", &Watcher::none()));
    assert!(ASKED.load(Ordering::SeqCst) > before, "the Hub was never asked");
}
