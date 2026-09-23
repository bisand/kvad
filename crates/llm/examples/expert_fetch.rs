//! Replay a recorded trace against the disk, and time the reads.
//!
//!     cargo run --release -p kvad --example expert_fetch -- CACHE.nq TRACE [--gb N]
//!
//! [`kvad::residency`] answers how often a token would find its experts
//! already resident. It cannot say what the rest cost, because it never
//! opens a file: it divides missed bytes by a bandwidth measured elsewhere,
//! on another file, under another access pattern. Every tok/s figure it
//! prints rests on that division.
//!
//! This does the reads. Same trace, same LRU, same capacity — but each miss
//! becomes an actual `pread` against the actual quantised cache, with the
//! kernel's own caching turned off, and the clock running. What comes out
//! is milliseconds a token spends waiting for disk, measured rather than
//! divided.
//!
//! Nothing here runs the model. That is the point: the arithmetic is
//! already known to cost 62 ms a token on Qwen3-30B-A3B, and what was
//! unknown is whether the fetch fits beside it.
//!
//! Options:
//!
//!     --gb N         cache size; default sweeps a range
//!     --threads N    reads in flight (default 4, which saturates an M5 Pro)
//!     --compute-ms N what a token costs with everything resident, to say
//!                    whether the measured fetch hides behind it

use kvad::experts::{Cache, Extent, ExpertStore};
use kvad::residency::Log;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cache_path), Some(trace_path)) = (args.first(), args.get(1)) else {
        eprintln!("usage: expert_fetch CACHE.nq TRACE [--gb N] [--threads N] [--compute-ms N]");
        std::process::exit(2);
    };
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let threads = flag("--threads").and_then(|v| v.parse().ok()).unwrap_or(4usize);
    let compute = flag("--compute-ms").and_then(|v| v.parse::<f64>().ok());

    let store = match ExpertStore::open(std::path::Path::new(cache_path)) {
        Ok(Some(s)) => s,
        Ok(None) => {
            eprintln!("{cache_path} holds no mixture");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("could not index {cache_path}: {e}");
            std::process::exit(1);
        }
    };
    let log = match Log::read(std::path::Path::new(trace_path)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("could not read {trace_path}: {e}");
            std::process::exit(1);
        }
    };
    // A trace of one model replayed against another's cache would index
    // the wrong experts and report a plausible, meaningless number.
    if log.n_experts != store.n_experts() {
        eprintln!(
            "trace is of a {}-expert model; this cache holds {}",
            log.n_experts,
            store.n_experts()
        );
        std::process::exit(1);
    }

    let expert = store.expert_bytes() as f64;
    let blobs = store.len();
    let tokens = log.tokens().max(1);
    println!(
        "{} experts of {:.2} MB = {:.1} GB · trace of {tokens} tokens · {threads} reads in flight",
        blobs,
        expert / 1e6,
        blobs as f64 * expert / 1e9
    );
    let fetcher = match kvad::experts::Fetcher::new(&store, threads) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("could not open the cache for reading: {e}");
            std::process::exit(1);
        }
    };

    let caps: Vec<usize> = match flag("--gb") {
        Some(list) => list
            .split(',')
            .filter_map(|g| g.trim().parse::<f64>().ok())
            .map(|g| (g * 1e9 / expert) as usize)
            .collect(),
        None => [0.10, 0.20, 0.30, 0.50].iter().map(|f| (blobs as f64 * f) as usize).collect(),
    };

    println!(
        "\n  {:>7} {:>5} {:>7} {:>11} {:>10} {:>9}{}",
        "cap GB", "%", "hit", "read GB/tok", "fetch ms", "GB/s", if compute.is_some() { "    tok/s" } else { "" }
    );

    for capacity in caps {
        let mut cache = Cache::new(capacity);
        let mut slabs: Vec<Vec<u8>> = Vec::new();
        let mut want: Vec<Extent> = Vec::new();
        let mut moved = 0u64;
        let mut spent = std::time::Duration::ZERO;

        for step in &log.steps {
            want.clear();
            for &e in &step.experts {
                let key = step.layer * log.n_experts as u32 + e;
                if cache.touch(key) {
                    continue;
                }
                // Resident-but-empty: the slot is reserved, and this is the
                // read that fills it. Grouped with the rest of the token's
                // misses so they go to the device together.
                match store.get(step.layer as usize, e as usize) {
                    Some(extent) => want.push(extent),
                    None => {
                        eprintln!("trace names expert {e} of layer {}, which this cache has not", step.layer);
                        std::process::exit(1);
                    }
                }
            }
            if want.is_empty() {
                continue;
            }
            if slabs.len() < want.len() {
                slabs.resize(want.len(), Vec::new());
            }
            let started = Instant::now();
            match fetcher.fetch(&want, &mut slabs[..want.len()]) {
                Ok(n) => moved += n,
                Err(e) => {
                    eprintln!("fetch failed: {e}");
                    std::process::exit(1);
                }
            }
            spent += started.elapsed();
        }

        let seconds = spent.as_secs_f64();
        let hit = cache.hits as f64 / (cache.hits + cache.misses).max(1) as f64;
        let per_token_ms = 1e3 * seconds / tokens as f64;
        print!(
            "  {:>7.1} {:>4.0}% {:>6.1}% {:>11.3} {:>10.2} {:>9.2}",
            capacity as f64 * expert / 1e9,
            100.0 * capacity as f64 / blobs as f64,
            100.0 * hit,
            moved as f64 / tokens as f64 / 1e9,
            per_token_ms,
            moved as f64 / 1e9 / seconds.max(1e-9),
        );
        match compute {
            Some(ms) => println!("  {:>7.1}", 1e3 / (ms + per_token_ms)),
            None => println!(),
        }
    }

    match compute {
        Some(ms) => println!(
            "\nfetch ms is measured, not divided, and tok/s adds it to {ms:.0} ms of\n\
             arithmetic — the pessimistic end, where nothing is prefetched and the\n\
             reads wait their turn. Overlapping them is what a real engine does next."
        ),
        None => println!(
            "\nfetch ms is measured, not divided. Pass --compute-ms to see what it does\n\
             to a token that costs that much arithmetic."
        ),
    }
}
