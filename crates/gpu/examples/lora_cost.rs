//! What a LoRA applied at run time costs a step: Qwen-Image at q8 and 1024²,
//! one load, then denoising alternated without and with a LoRA, so that
//! whatever else the machine is doing lands on both alike.
//!
//!     cargo run --release -p kvad-gpu --example lora_cost -- LORA.safetensors [ROUNDS]
//!
//! Prints each run's time a step, the medians, and the median of each
//! round's difference.

use kvad::image::{ImageRequest, Painter};
use kvad::weights::Watcher;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let path = argv.first().ok_or("give a LoRA file")?;
    let rounds: usize = argv.get(1).map(|r| r.parse()).transpose()?.unwrap_or(4);
    let device = kvad_gpu::model::pick_device(None)?;
    let quant = kvad_gpu::model::parse_quant("q8").flatten();
    let mut q = kvad_gpu::image::qwen::QwenImage::load_with("Qwen/Qwen-Image", None, quant, device, &mut |m| eprintln!("  {m}"), &Watcher::none())?;
    let file = kvad_gpu::image::lora::File::open(std::path::Path::new(path))?;
    let mut req = ImageRequest::new("a tiny astronaut hatching from an egg on the moon");
    (req.width, req.height, req.steps, req.seed) = (Some(1024), Some(1024), Some(3), Some(1));
    let (mut plain, mut adapted) = (Vec::new(), Vec::new());
    for round in 0..rounds {
        for with in [false, true] {
            match with {
                true => q.set_loras(&[(&file, 1.0)])?,
                false => q.set_loras(&[])?,
            };
            let p = q.paint(&req, &mut |_| true)?;
            let per = p.denoise_secs / 3.0;
            eprintln!("round {round}, {}: {per:.2} s a step", if with { "with the LoRA" } else { "without" });
            if with { adapted.push(per) } else { plain.push(per) }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    // Each round's pair was run back to back, so its difference is the
    // least disturbed by whatever else the machine did in between.
    let mut diffs: Vec<f64> = plain.iter().zip(&adapted).map(|(p, a)| a - p).collect();
    let d = median(&mut diffs);
    let (p, a) = (median(&mut plain), median(&mut adapted));
    eprintln!("medians: {p:.2} s a step without, {a:.2} with; of each round's pair, {d:+.2} s, {:+.1}%", d / p * 100.0);
    Ok(())
}
