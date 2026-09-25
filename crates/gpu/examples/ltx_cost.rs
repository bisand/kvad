//! Where an LTX-2.5 DiT step's time goes, on Metal at q8, piece by piece.
//!
//!     cargo run --release -p kvad-gpu --example ltx_cost -- \
//!         [--width 1536 --height 1024 --frames 121] [--blocks 2] [--rounds 3]
//!
//! Loads the first `--blocks` blocks (quantised afresh, not from the cache),
//! and runs them on noise at the shape given, which defaults to stage 2's.
//! The text contexts are noise too, at their real 1024 rows. Timing doesn't
//! depend on the numbers, only on the shapes.
//!
//! It times the forward pass twice over:
//! 1. **Whole**, the median of `--rounds` runs, as the pipeline runs it.
//! 2. **In pieces**, with [`kvad_gpu::prof`] synchronising around each
//!    attention's projections, norms and RoPE, attention, gate and output
//!    projection, and each feed-forward's up, GELU and down. What isn't in
//!    a piece (modulation, residuals, patchify, heads) is "the rest".
//!
//! The pieces synchronise, so their total runs a little over the whole
//! forward pass. Each piece is given per block, with the arithmetic it does
//! where that is known and the rate it does it at.

use candle_core::{DType, Device, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::prof;
use kvad_gpu::video::ltx_dit::{Dit, Shape};
use kvad_gpu::video::ltx_text::{Contexts, DIT_FILE, LENGTH};
use kvad_gpu::video::LTX_REPO;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Widths: video 4096 (32 heads of 128), audio 2048 (32 of 64), the
/// feed-forwards four times their stream's width.
const V: f64 = 4096.0;
const A: f64 = 2048.0;

/// The multiply-adds a piece does, ×2, for `t` video tokens, `a` audio
/// tokens and `c` text rows. `None` for what isn't a matmul.
fn flops(scope: &str, label: &str, t: f64, a: f64, c: f64) -> Option<f64> {
    let mm = |m: f64, k: f64, n: f64| 2.0 * m * k * n;
    Some(match (scope, label) {
        ("video self", "q, k, v") => 3.0 * mm(t, V, V),
        ("video self", "attention") => 2.0 * mm(t, t, V),
        ("video self", "out") | ("video text", "out") => mm(t, V, V),
        ("video text", "q, k, v") => mm(t, V, V) + 2.0 * mm(c, V, V),
        ("video text", "attention") => 2.0 * mm(t, c, V),
        ("video ff", "up") | ("video ff", "down") => mm(t, V, 4.0 * V),
        ("audio self", "q, k, v") => 3.0 * mm(a, A, A),
        ("audio self", "attention") => 2.0 * mm(a, a, A),
        ("audio self", "out") | ("audio text", "out") => mm(a, A, A),
        ("audio text", "q, k, v") => mm(a, A, A) + 2.0 * mm(c, A, A),
        ("audio text", "attention") => 2.0 * mm(a, c, A),
        ("audio ff", "up") | ("audio ff", "down") => mm(a, A, 4.0 * A),
        // Video queries at audio's width, audio keys and values.
        ("audio to video", "q, k, v") => mm(t, V, A) + 2.0 * mm(a, A, A),
        ("audio to video", "attention") => 2.0 * mm(t, a, A),
        ("audio to video", "out") => mm(t, A, V),
        // Audio queries, video keys and values at audio's width.
        ("video to audio", "q, k, v") => mm(a, A, A) + 2.0 * mm(t, V, A),
        ("video to audio", "attention") => 2.0 * mm(a, t, A),
        ("video to audio", "out") => mm(a, A, A),
        _ => return None,
    })
}

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let num = |f: &str, d: usize| -> Res<usize> { Ok(value(f).map(|v| v.parse()).transpose()?.unwrap_or(d)) };
    let shape = Shape::new(num("--width", 1536)?, num("--height", 1024)?, num("--frames", 121)?, 24.0)?;
    let (blocks, rounds) = (num("--blocks", 2)?, num("--rounds", 3)?);

    let device = Device::new_metal(0)?;
    let path = fetch_file(LTX_REPO, DIT_FILE, &Watcher::none())?;
    let q8 = kvad_gpu::model::parse_quant("q8").ok_or("no q8")?;
    let dit = Dit::load(&path, &device, DType::BF16, Some(blocks), q8, &mut |_| {})?;
    let grid = dit.grid(shape)?;
    let (t, a) = (shape.video_tokens(), shape.audio_latents());
    let noise = |rows: usize, cols: usize| -> Res<Tensor> { Ok(Tensor::randn(0f32, 1.0, (rows, cols), &device)?.to_dtype(DType::BF16)?) };
    let (video, audio) = (noise(t, 128)?, noise(a, 128)?);
    let ctx = Contexts { video: noise(LENGTH, 4096)?, audio: noise(LENGTH, 2048)? };
    eprintln!("{}×{} × {} frames: {t} video tokens, {a} audio; {blocks} blocks at q8", shape.width, shape.height, shape.frames);

    let forward = || -> Res<f64> {
        device.synchronize()?;
        let s = Instant::now();
        dit.forward(&video, &audio, (0.725, 0.725), &ctx, &grid)?;
        device.synchronize()?;
        Ok(s.elapsed().as_secs_f64())
    };
    forward()?;
    let mut whole: Vec<f64> = (0..rounds).map(|_| forward()).collect::<Res<_>>()?;
    whole.sort_by(f64::total_cmp);
    let whole = whole[rounds / 2];

    prof::start();
    let pieces = forward()?;
    let rows = prof::stop();

    let per = |s: f64| s / blocks as f64;
    println!("forward: {:.3} s a block whole (median of {rounds}), {:.3} s in pieces", per(whole), per(pieces));
    println!();
    println!("{:<16} {:<11} {:>8} {:>6} {:>8} {:>8}", "", "", "s/block", "share", "TFLOP", "TFLOP/s");
    let mut counted = 0.0;
    let (mut total_flops, mut group) = (0.0, "");
    for r in &rows {
        counted += r.seconds;
        let f = flops(r.scope, &r.label, t as f64, a as f64, LENGTH as f64);
        total_flops += f.unwrap_or(0.0);
        let rate = f.map(|f| format!("{:>8.3} {:>8.2}", f / 1e12, f / 1e12 / per(r.seconds))).unwrap_or_default();
        let name = if r.scope == group { "" } else { r.scope };
        group = r.scope;
        println!("{name:<16} {:<11} {:>8.3} {:>5.1}% {rate}", r.label, per(r.seconds), 100.0 * r.seconds / pieces);
    }
    let rest = pieces - counted;
    println!("{:<28} {:>8.3} {:>5.1}%", "the rest", per(rest), 100.0 * rest / pieces);
    println!();
    println!("matmuls and attention: {:.1} TFLOP a block, at {:.2} TFLOP/s over the whole forward", total_flops / 1e12, total_flops / 1e12 / per(whole));
    Ok(())
}
