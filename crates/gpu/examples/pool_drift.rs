//! Whether a long-lived loop of GPU work grows: the footprint at rest after
//! each image, or each few hundred tokens, of one process.
//!
//!     cargo run --release -p kvad-gpu --example pool_drift -- image REPO [--count 20] [--steps 4] [--size 512] [--pool]
//!     cargo run --release -p kvad-gpu --example pool_drift -- llm REPO [--count 20] [--steps 200] [--pool]
//!
//! `--pool` gives each image, or each token, an autorelease pool from out
//! here (`kvad_gpu::pooled`), which is the difference a pool makes whatever
//! the engine does inside. `--thread` runs the loop on a spawned thread, as
//! a server does. Stops itself at 24 GB. Reads the cache and fetches nothing
//! it does not find there... so name a model that is.

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

fn probe(argv: &[String]) -> Res<()> {
    let num = |f: &str, default: usize| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default);
    let pool = argv.iter().any(|a| a == "--pool");
    let (kind, repo) = (argv.first().ok_or("image or llm, then a repo")?.as_str(), argv.get(1).ok_or("a repo")?);
    let count = num("--count", 20);
    let mb = || cap::footprint() as f64 / 1e6;
    let mut seen = Vec::new();
    let mut report = |i: usize| {
        seen.push(mb());
        println!("{:>4}  {:9.1} MB", i, seen[seen.len() - 1]);
    };
    match kind {
        "image" => {
            let (steps, size) = (num("--steps", 4), num("--size", 512));
            let mut painter = kvad_gpu::image::load_with(repo, None, None, &mut |_| {}, &Watcher::none())?;
            println!("{} on {}, {steps} steps at {size}², {}", painter.summary(), painter.backend(), if pool { "a pool an image" } else { "no pool" });
            for i in 0..count {
                let mut req = ImageRequest::new("a lighthouse on a cliff at dusk, oil painting");
                (req.steps, req.width, req.height, req.seed) = (Some(steps), Some(size), Some(size), Some(i as u64));
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
    let n = seen.len();
    if n >= 5 {
        // From the fifth reading, by when the buffer pool has its sizes.
        println!("growth from round 5 to round {n}: {:+.1} MB, {:+.2} MB a round", seen[n - 1] - seen[4], (seen[n - 1] - seen[4]) / (n - 5).max(1) as f64);
    }
    Ok(())
}
