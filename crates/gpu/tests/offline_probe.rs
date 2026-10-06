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

/// `files` in the cache as `owner/name`, each holding `text`, and `.no_exist`
/// markers for `missing`. What a load that came before would have left.
fn cache_repo(repo: &str, files: &[(&str, &str)], missing: &[&str]) -> PathBuf {
    let hub = PathBuf::from(std::env::var("HF_HOME").unwrap()).join("hub");
    let root = hub.join(format!("models--{}", repo.replace('/', "--")));
    let snapshot = root.join("snapshots").join(REVISION);
    std::fs::create_dir_all(root.join("refs")).unwrap();
    std::fs::write(root.join("refs/main"), REVISION).unwrap();
    for (dir, names) in [(&snapshot, files.iter().map(|f| *f).collect::<Vec<_>>()), (&root.join(".no_exist").join(REVISION), missing.iter().map(|m| (*m, "")).collect())] {
        for (file, text) in names {
            let path = dir.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
    }
    snapshot
}

/// A pull of `repo`, and the files it says it fetched, each by its path
/// under its snapshot. The Hub must not have been asked: everything a pull
/// wants is in the cache, so a request is a file the test did not expect.
fn pulled(repo: &str) -> Vec<String> {
    let before = ASKED.load(Ordering::SeqCst);
    let files = kvad_gpu::image::pull_pipeline(repo, &mut |_| {}, &Watcher::none()).unwrap().expect("a pipeline");
    assert_eq!(ASKED.load(Ordering::SeqCst), before, "the pull of {repo} asked the Hub for a file the cache does not hold");
    let mut names: Vec<String> = files
        .iter()
        .map(|p| {
            let p = p.to_string_lossy();
            let (repo, file) = p.split_once(&format!("/snapshots/{REVISION}/")).unwrap();
            format!("{}/{file}", repo.rsplit('/').next().unwrap())
        })
        .collect();
    names.sort();
    names
}

/// `files` of `repo` as [`pulled`] names them, with what every pipeline
/// here borrows: CLIP's tokenizer, and for SDXL its VAE.
fn expected(repo: &str, files: &[&str], borrowed: &[&str]) -> Vec<String> {
    let mut names: Vec<String> = files.iter().map(|f| format!("models--{}/{f}", repo.replace('/', "--"))).collect();
    names.extend(borrowed.iter().map(|b| b.to_string()));
    names.sort();
    names
}

const CLIP_TOKENIZER: &str = "models--openai--clip-vit-large-patch14/tokenizer.json";

/// The repos a pipeline borrows from, in the cache.
fn cache_borrowed() {
    cache_repo("openai/clip-vit-large-patch14", &[("tokenizer.json", "{}")], &[]);
    cache_repo("madebyollin/sdxl-vae-fp16-fix", &[("config.json", "{}"), ("diffusion_pytorch_model.safetensors", "")], &[]);
    cache_repo("Qwen/Qwen2.5-VL-7B-Instruct", &[("tokenizer.json", "{}")], &[]);
}

/// A pull of a pipeline in diffusers' layout fetches what its load reads and
/// nothing else: the `.fp16` weights where the repo has both, the plain ones
/// where it has only those, every shard an index names, and not SDXL's own
/// VAE. It used to be a language model's pull, and fail for want of a
/// `config.json`.
#[test]
fn a_pull_of_a_pipeline_fetches_what_its_load_reads() {
    let _alone = alone();
    cache_borrowed();
    let index = |class: &str| format!(r#"{{"_class_name": "{class}"}}"#);
    let shards = r#"{"weight_map": {"a": "w-00001-of-00002.safetensors", "b": "w-00002-of-00002.safetensors", "c": "w-00001-of-00002.safetensors"}}"#;

    // SDXL as Stability ships it: f32 and fp16 side by side, and a VAE.
    let read = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "unet/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    ];
    let unread = ["text_encoder/model.safetensors", "text_encoder_2/model.safetensors", "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"];
    let class = index("StableDiffusionXLPipeline");
    let files: Vec<(&str, &str)> = read.iter().chain(&unread).map(|f| (*f, if *f == "model_index.json" { class.as_str() } else { "{}" })).collect();
    cache_repo("kvad-test/xl", &files, &[]);
    let vae = ["models--madebyollin--sdxl-vae-fp16-fix/config.json", "models--madebyollin--sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors"];
    assert_eq!(pulled("kvad-test/xl"), expected("kvad-test/xl", &read, &[CLIP_TOKENIZER, vae[0], vae[1]]));

    // SD 1.5 as a fine-tune ships it: one set of weights, its own VAE.
    let read = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "unet/config.json",
        "vae/config.json",
        "text_encoder/model.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ];
    let class = index("StableDiffusionPipeline");
    let files: Vec<(&str, &str)> = read.iter().map(|f| (*f, if *f == "model_index.json" { class.as_str() } else { "{}" })).collect();
    let no_fp16 = ["text_encoder/model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"];
    cache_repo("kvad-test/fifteen", &files, &no_fp16);
    assert_eq!(pulled("kvad-test/fifteen"), expected("kvad-test/fifteen", &read, &[CLIP_TOKENIZER]));

    // FLUX: CLIP in one file, T5 and the transformer in shards.
    let read = [
        "model_index.json",
        "tokenizer_2/tokenizer.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/w-00001-of-00002.safetensors",
        "text_encoder_2/w-00002-of-00002.safetensors",
        "transformer/w-00001-of-00002.safetensors",
        "transformer/w-00002-of-00002.safetensors",
    ];
    let indexes = ["text_encoder_2/model.safetensors.index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"];
    let class = index("FluxPipeline");
    let mut files: Vec<(&str, &str)> = read.iter().map(|f| (*f, if *f == "model_index.json" { class.as_str() } else { "{}" })).collect();
    files.extend(indexes.iter().map(|i| (*i, shards)));
    // What is not a pipeline's file at all, and sits in the repo's root.
    files.push(("flux1-schnell.safetensors", ""));
    cache_repo("kvad-test/flux", &files, &["text_encoder/model.safetensors.index.json"]);
    assert_eq!(pulled("kvad-test/flux"), expected("kvad-test/flux", &read, &[CLIP_TOKENIZER]));

    // Qwen-Image: its text encoder and transformer in shards.
    let read = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/w-00001-of-00002.safetensors",
        "text_encoder/w-00002-of-00002.safetensors",
        "transformer/w-00001-of-00002.safetensors",
        "transformer/w-00002-of-00002.safetensors",
    ];
    let indexes = ["text_encoder/model.safetensors.index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"];
    let class = index("QwenImagePipeline");
    let mut files: Vec<(&str, &str)> = read.iter().map(|f| (*f, if *f == "model_index.json" { class.as_str() } else { "{}" })).collect();
    files.extend(indexes.iter().map(|i| (*i, shards)));
    cache_repo("kvad-test/qwen", &files, &[]);
    assert_eq!(pulled("kvad-test/qwen"), expected("kvad-test/qwen", &read, &["models--Qwen--Qwen2.5-VL-7B-Instruct/tokenizer.json"]));
}

/// A repo with no model index is left for the language model's pull, and a
/// pipeline with no implementation here is refused before its weights.
#[test]
fn a_pull_knows_what_is_not_its_pipeline() {
    let _alone = alone();
    let language = cache_language_model("pulled", &["model_index.json"]);
    let before = ASKED.load(Ordering::SeqCst);
    assert!(kvad_gpu::image::pull_pipeline(&language, &mut |_| {}, &Watcher::none()).unwrap().is_none());

    cache_repo("kvad-test/three", &[("model_index.json", r#"{"_class_name": "StableDiffusion3Pipeline"}"#)], &[]);
    let refused = kvad_gpu::image::pull_pipeline("kvad-test/three", &mut |_| {}, &Watcher::none()).unwrap_err().to_string();
    assert!(refused.contains("StableDiffusion3Pipeline"), "{refused}");
    assert_eq!(ASKED.load(Ordering::SeqCst), before, "the Hub was asked");
}

/// Every pipeline this backend loads has a pull: a model index naming one
/// is never the "not implemented here" a pull gives.
#[test]
fn every_pipeline_has_a_pull() {
    let _alone = alone();
    for class in kvad_gpu::image::PIPELINES {
        let repo = format!("kvad-test/bare-{class}");
        // A model index and nothing else, in a directory standing in for
        // the repo, where a file that is not there is an error and not a
        // request.
        let dir = std::env::temp_dir().join(format!("kvad-probe-{}", std::process::id())).join(&repo);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model_index.json"), format!(r#"{{"_class_name": "{class}"}}"#)).unwrap();
        cache_borrowed();
        let said = kvad_gpu::image::pull_pipeline(&dir.to_string_lossy(), &mut |_| {}, &Watcher::none()).unwrap_err().to_string();
        assert!(!said.contains("implemented here"), "{class}: {said}");
    }
}
