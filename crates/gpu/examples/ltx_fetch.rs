//! Fetch files of LTX-2.5's repo into the Hub cache, and print where each is.
//!
//!     cargo run --release -p kvad-gpu --example ltx_fetch -- [--repo REPO] PATH_IN_REPO …
//!
//! For files no example reads yet: the dev DiT and the distilled LoRA are
//! 51 GB, and worth starting before the code that reads them is written.
//! `--repo` names another of Lightricks' repos, such as the detailing
//! IC-LoRA's, in place of LTX-2.5's own.

use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::LTX_REPO;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    let repo = match argv.iter().position(|a| a == "--repo") {
        Some(i) if i + 1 < argv.len() => argv.drain(i..i + 2).nth(1).unwrap_or_default(),
        Some(_) => return Err("--repo wants a repo".into()),
        None => LTX_REPO.to_string(),
    };
    for f in argv {
        let t = std::time::Instant::now();
        let path = fetch_file(&repo, &f, &Watcher::none())?;
        println!("{} ({:.0} s)", path.display(), t.elapsed().as_secs_f64());
    }
    Ok(())
}
