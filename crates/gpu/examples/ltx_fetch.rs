//! Fetch files of LTX-2.5's repo into the Hub cache, and print where each is.
//!
//!     cargo run --release -p kvad-gpu --example ltx_fetch -- PATH_IN_REPO …
//!
//! For files no example reads yet: the dev DiT and the distilled LoRA are
//! 51 GB, and worth starting before the code that reads them is written.

use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::video::LTX_REPO;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for f in std::env::args().skip(1) {
        let t = std::time::Instant::now();
        let path = fetch_file(LTX_REPO, &f, &Watcher::none())?;
        println!("{} ({:.0} s)", path.display(), t.elapsed().as_secs_f64());
    }
    Ok(())
}
