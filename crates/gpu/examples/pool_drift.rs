//! Whether a long-lived loop of GPU work grows: the footprint at rest after
//! each image, or each few hundred tokens, of one process.
//!
//!     cargo run --release -p kvad-gpu --example pool_drift -- image REPO [--count 20] [--steps 4] [--size 512] [--preview] [--pool]
//!     cargo run --release -p kvad-gpu --example pool_drift -- llm REPO [--count 20] [--steps 200] [--pool]
//!
//! `--pool` gives each image, or each token, an autorelease pool from out
//! here (`kvad_gpu::pooled`), which is the difference a pool makes whatever
//! the engine does inside. `--preview` asks for a preview a step, which reads
//! the latent back, as a server's requests do. `--thread` runs the loop on a spawned thread, as
//! a server does. Stops itself at 24 GB.
//!
//! Two columns, because the footprint alone cannot say. It has the GPU's
//! buffers in it and rests at one of a few levels tens of MB apart, and what
//! leaks without a pool is a few kB a command buffer: the heap in use shows
//! that to the kB. Before the engine had pools of its own, SDXL grew by
//! 160 kB an image and a 1.5 B LLM by 1.5 kB a token.
//!
//! Name a model that is in the cache, and set `HF_HUB_OFFLINE=1` to be sure
//! nothing is fetched.

use kvad::image::ImageRequest;
use kvad::weights::Watcher;
use kvad_gpu::{cap, model, pooled};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    cap::at(24.0);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let run = move || {
        if let Err(e) = probe(&argv) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    match std::env::args().any(|a| a == "--thread") {
        true => std::thread::Builder::new().stack_size(8 << 20).spawn(run).unwrap().join().unwrap(),
        false => run(),
    }
}

/// MB of the malloc heap in use, which is where what Metal autoreleases
/// lives. The footprint has the GPU's buffers in it and moves by tens of MB
/// from one reading to the next; this does not, and shows a leak of a few
/// kB a step that the footprint cannot.
fn heap() -> f64 {
    #[cfg(target_os = "macos")]
    {
        let mut stats = std::mem::MaybeUninit::<libc::malloc_statistics_t>::zeroed();
        // SAFETY: a null zone asks for all of them, and `stats` is the
        // struct the call fills, alive past it.
        unsafe {
            libc::malloc_zone_statistics(std::ptr::null_mut(), stats.as_mut_ptr());
            return stats.assume_init().size_in_use as f64 / 1e6;
        }
    }
    #[cfg(not(target_os = "macos"))]
    0.0
}

fn probe(argv: &[String]) -> Res<()> {
    let num = |f: &str, default: usize| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default);
    let pool = argv.iter().any(|a| a == "--pool");
    let preview = argv.iter().any(|a| a == "--preview");
    let (kind, repo) = (argv.first().ok_or("image or llm, then a repo")?.as_str(), argv.get(1).ok_or("a repo")?);
    let count = num("--count", 20);
    let mb = || cap::footprint() as f64 / 1e6;
    let (mut seen, mut heaps) = (Vec::new(), Vec::new());
    let mut report = |i: usize| {
        seen.push(mb());
        heaps.push(heap());
        println!("{:>4}  {:9.1} MB  heap {:9.3} MB", i, seen[seen.len() - 1], heaps[heaps.len() - 1]);
    };
    match kind {
        "image" => {
            let (steps, size) = (num("--steps", 4), num("--size", 512));
            let mut painter = kvad_gpu::image::load_with(repo, None, None, &mut |_| {}, &Watcher::none())?;
            println!("{} on {}, {steps} steps at {size}², {}", painter.summary(), painter.backend(), if pool { "a pool an image" } else { "no pool" });
            for i in 0..count {
                let mut req = ImageRequest::new("a lighthouse on a cliff at dusk, oil painting");
                (req.steps, req.width, req.height, req.seed) = (Some(steps), Some(size), Some(size), Some(i as u64));
                req.preview = preview;
                let mut draw = || painter.paint(&req, &mut |_| true).map(drop);
                if pool { pooled(&mut draw)? } else { draw()? };
                report(i + 1);
            }
        }
        "llm" => {
            let steps = num("--steps", 200);
            let device = model::pick_device(None)?;
            let id = kvad::weights::model_id(repo);
            let mut llm = kvad::runtime::Llm::load_custom(repo, &mut |_| {}, &Watcher::none(), &mut |files, spec, progress| {
                model::session(&id, &files.weights, spec, candle_core::DType::BF16, None, device.clone(), progress)
            })?;
            println!("{repo} on {}, {steps} tokens a round, {}", llm.backend(), if pool { "a pool a token" } else { "no pool" });
            for i in 0..count {
                // The same context every round, so the cache is the same size
                // at each reading.
                llm.session.truncate(0)?;
                for t in 0..steps {
                    let mut step = || llm.session.forward(&[1000 + t as u32]).map(drop);
                    if pool { pooled(&mut step)? } else { step()? };
                }
                report(i + 1);
            }
        }
        other => return Err(format!("`{other}` is neither image nor llm").into()),
    }
    // Medians of five, because a reading at rest is one of a few levels
    // some tens of MB apart (what the buffer pool happens to hold): rounds
    // 6 to 10, by when the pool has its sizes, against the last five.
    let n = seen.len();
    if n >= 15 {
        let median = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (early, late) = (median(&seen[5..10]), median(&seen[n - 5..]));
        println!("rounds 6-10 {early:.1} MB, the last five {late:.1} MB: {:+.2} MB a round", (late - early) / (n - 10) as f64);
        let (early, late) = (median(&heaps[5..10]), median(&heaps[n - 5..]));
        println!("the heap,   {early:.3} MB, the last five {late:.3} MB: {:+.1} kB a round", (late - early) * 1e3 / (n - 10) as f64);
    }
    Ok(())
}
