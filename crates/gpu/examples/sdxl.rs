//! Prompt in, PNG out: SDXL on the GPU with no server involved.
//!
//!     cargo run --release -p kvad-gpu --example sdxl -- \
//!         --prompt "a lighthouse on a cliff at dusk, oil painting" --out lighthouse.png
//!
//! Options: `--negative TEXT`, `--steps N`, `--guidance F`, `--seed N`,
//! `--width N`, `--height N`, `--repo REPO` (any repo whose
//! `model_index.json` names a pipeline `kvad_gpu::image` implements).
//!
//! Reports the time spent in each of the three models, because they are
//! nothing alike: the text encoders run once over 77 tokens, the denoiser
//! runs `steps` times (twice each with guidance), and the decoder runs once
//! at full resolution.

use kvad::image::{ImageRequest, Step};
use kvad::weights::Watcher;
use std::io::Write;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let mut req = ImageRequest::new("a lighthouse on a cliff at dusk, oil painting");
    let mut out = String::from("sdxl.png");
    let mut repo = kvad_gpu::image::sdxl::REPO.to_string();
    let mut quant = None;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let v = argv.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", argv[i]))?;
        match argv[i].as_str() {
            "--prompt" => req.prompt = v,
            "--negative" => req.negative_prompt = Some(v),
            "--steps" => req.steps = Some(v.parse()?),
            "--guidance" => req.guidance = Some(v.parse()?),
            "--seed" => req.seed = Some(v.parse()?),
            "--width" => req.width = Some(v.parse()?),
            "--height" => req.height = Some(v.parse()?),
            "--out" => out = v,
            "--repo" => repo = v,
            "--quant" => quant = kvad_gpu::model::parse_quant(&v).ok_or("--quant is none, q8, q4, q4k or q6k")?,
            other => return Err(format!("unknown option {other}").into()),
        }
        i += 2;
    }

    let t = Instant::now();
    let mut painter = kvad_gpu::image::load(&repo, quant, &mut |m| eprintln!("  {m}"), &Watcher::none())?;
    eprintln!("{} on {}, loaded in {:.1} s", painter.summary(), painter.backend(), t.elapsed().as_secs_f64());

    let mut last = Instant::now();
    let painted = painter.paint(&req, &mut |s: Step| {
        eprint!("\r  step {:>3}/{}  {:.2} s/step ", s.done, s.total, last.elapsed().as_secs_f64());
        let _ = std::io::stderr().flush();
        last = Instant::now();
        true
    })?;
    eprintln!();

    std::fs::write(&out, painted.image.png())?;
    let r = &painted.request;
    eprintln!(
        "{out}: {}×{}, {} steps, guidance {}, seed {}\n  encode {:.2} s, denoise {:.1} s ({:.2} s/step), decode {:.1} s",
        r.width,
        r.height,
        r.steps,
        r.guidance,
        r.seed,
        painted.encode_secs,
        painted.denoise_secs,
        painted.denoise_secs / r.steps as f64,
        painted.decode_secs
    );
    Ok(())
}
