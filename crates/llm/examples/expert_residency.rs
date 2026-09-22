//! Replay an expert trace against a cache that does not exist yet.
//!
//!     KVAD_EXPERT_TRACE=/tmp/m.trace kvad chat SOME-MOE-MODEL
//!     cargo run --release -p kvad --example expert_residency -- /tmp/m.trace
//!
//! The question it answers: a mixture-of-experts model too big for memory
//! could keep some experts resident and read the rest off the disk — how
//! much of what a token asks for would already be there?
//!
//! Three policies, because the gap between them is the actual finding.
//! `LRU` is what you get for free. `pinned` keeps the globally hottest
//! experts and never evicts, which needs no runtime machinery at all — if
//! it is close to LRU, that is the thing to build. `optimal` reads the
//! future and cannot be built, which is the point: it is the ceiling, and
//! if LRU is near it then no cleverer policy is worth writing.
//!
//! Options:
//!
//!     --bandwidth GB/s   what the disk streams (default 11.7, measured on
//!                        an M5 Pro at a 4 MB block size — measure yours,
//!                        it is the difference between this being a good
//!                        idea and a bad one)
//!     --ram GB           mark the capacities that would fit in this much
//!     --gb A,B,C         capacities to try, instead of the default sweep

use kvad::residency::{Log, Outcome, Policy};

const GB: f64 = 1e9;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: expert_residency TRACE [--bandwidth GB/s] [--ram GB] [--gb A,B,C]");
        std::process::exit(2);
    };
    let flag = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let bandwidth = flag("--bandwidth").and_then(|v| v.parse::<f64>().ok()).unwrap_or(11.7) * GB;
    let ram = flag("--ram").and_then(|v| v.parse::<f64>().ok()).map(|g| g * GB);

    let log = match Log::read(std::path::Path::new(path)) {
        Ok(log) => log,
        Err(e) => {
            eprintln!("could not read {path}: {e}");
            std::process::exit(1);
        }
    };

    let (layers, tokens) = (log.layers(), log.tokens());
    let expert = log.expert_bytes as f64;
    // What the model holds, not what the trace touched: the cache has to be
    // sized against the whole store, because anything it does not hold is
    // what the disk is for.
    let blobs = layers * log.n_experts;
    let store = blobs as f64 * expert;
    let reads: usize = log.steps.iter().map(|s| s.experts.len()).sum();

    println!(
        "{tokens} tokens · {layers} routed layers × {} experts = {blobs} blobs of {:.2} MB",
        log.n_experts,
        expert / 1e6
    );
    println!(
        "expert store {:.1} GB · {reads} reads · {} distinct blobs touched ({:.0}% of the store)",
        store / GB,
        log.distinct(),
        100.0 * log.distinct() as f64 / blobs as f64
    );
    println!(
        "every read from disk would cost {:.2} GB/token; disk assumed at {:.1} GB/s\n",
        reads as f64 * expert / tokens as f64 / GB,
        bandwidth / GB
    );

    let capacities: Vec<usize> = match flag("--gb") {
        Some(list) => list
            .split(',')
            .filter_map(|g| g.trim().parse::<f64>().ok())
            .map(|g| (g * GB / expert) as usize)
            .collect(),
        // Fractions of the store rather than absolute sizes, so the sweep
        // means the same thing for a 60 GB model and a 400 GB one.
        None => [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            .iter()
            .map(|f| (blobs as f64 * f) as usize)
            .collect(),
    };

    println!(
        "  {:>6} {:>5}   {:>15}   {:>15}   {:>15}   {:>13}",
        "cap GB", "%", "LRU", "pinned", "optimal", "tok/s ceiling"
    );
    println!(
        "  {:>6} {:>5}   {:>6} {:>8}   {:>6} {:>8}   {:>6} {:>8}   {:>6} {:>6}",
        "", "", "hit", "GB/tok", "hit", "GB/tok", "hit", "GB/tok", "LRU", "opt"
    );

    for capacity in capacities {
        let by: Vec<Outcome> = Policy::ALL.iter().map(|p| log.replay(*p, capacity)).collect();
        let held = capacity as f64 * expert;
        // A cache big enough for the whole store is the "just load it"
        // baseline, and the row is worth printing precisely because it is
        // the thing this is trying to avoid.
        let fits = match ram {
            Some(ram) if held <= ram => " ",
            Some(_) => "!",
            None => " ",
        };
        print!("{fits} {:>6.1} {:>4.0}%", held / GB, 100.0 * capacity as f64 / blobs as f64);
        for o in &by {
            print!("   {:>5.1}% {:>8.3}", 100.0 * o.hit_rate(), o.bytes_per_token() / GB);
        }
        println!(
            "   {:>6.1} {:>6.1}",
            by[0].ceiling(bandwidth).min(9999.9),
            by[2].ceiling(bandwidth).min(9999.9)
        );
    }

    if ram.is_some() {
        println!("\n`!` marks a capacity that would not fit the RAM given.");
    }
    println!(
        "\ntok/s is a ceiling, not a prediction: it assumes compute is free, every\n\
         fetch streams at full rate, and nothing is prefetched. A real engine lands\n\
         below it — so a row that looks bad here cannot be rescued by good code."
    );
}
