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
//!    attention's projections, norms and RoPE, gates' logits, attention
//!    (which applies the gates) and output projection, and each
//!    feed-forward's up (with its GELU, where the projection takes it), GELU
//!    and down. What isn't in a piece (modulation, residuals, patchify,
//!    heads) is "the rest".
//!
//! The pieces synchronise, so their total runs a little over the whole
//! forward pass. Each piece is given per block, with the arithmetic it does
//! where that is known and the rate it does it at.
//!
//! `--accuracy` instead runs the same inputs through the blocks in f32 and
//! in bf16, unquantised, and says how far apart the two are after each
//! block and at the velocities, in dB. The f32 path agreed with Lightricks'
//! reference to 115–123 dB, so it stands in for it here. The inputs are the
//! same from run to run, so `KVAD_GPU_FUSED=0` gives the unfused numbers to
//! compare, and `KVAD_GPU_MPP_ATTENTION=0` candle's attention's.

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
    if argv.iter().any(|a| a == "--accuracy") {
        return accuracy(&path, shape, blocks, &device);
    }
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

/// `rows × cols` numbers of about unit spread, the same every run: a sum of
/// four uniforms from a fixed generator, centred.
fn fixed(rows: usize, cols: usize, seed: u64) -> Res<Tensor> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let v: Vec<f32> = (0..rows * cols).map(|_| (next() + next() + next() + next() - 2.0) * 1.7).collect();
    Ok(Tensor::from_vec(v, (rows, cols), &Device::Cpu)?)
}

/// Signal-to-error ratio in dB of `got` against `want`.
fn db(got: &Tensor, want: &Tensor) -> Res<f32> {
    let err = (got - want)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let sig = want.sqr()?.sum_all()?.to_scalar::<f32>()?;
    if !(err + sig).is_finite() {
        return Err("a comparison met a NaN or an infinity".into());
    }
    Ok(10.0 * (sig / err.max(1e-30)).log10())
}

fn accuracy(path: &std::path::Path, shape: Shape, blocks: usize, device: &Device) -> Res<()> {
    let (t, a) = (shape.video_tokens(), shape.audio_latents());
    let (video, audio) = (fixed(t, 128, 1)?, fixed(a, 128, 2)?);
    let (cv, ca) = (fixed(LENGTH, 4096, 3)?, fixed(LENGTH, 2048, 4)?);
    // Both streams after every block, then the velocities, on the host in f32.
    let run = |dt: DType| -> Res<Vec<(Tensor, Tensor)>> {
        let dit = Dit::load(path, device, dt, Some(blocks), None, &mut |_| {})?;
        let grid = dit.grid(shape)?;
        let on = |x: &Tensor| -> Res<Tensor> { Ok(x.to_device(device)?.to_dtype(dt)?) };
        let ctx = Contexts { video: on(&cv)?, audio: on(&ca)? };
        let host = |x: &Tensor| -> candle_core::Result<Tensor> { x.to_device(&Device::Cpu)?.to_dtype(DType::F32) };
        let mut out = Vec::new();
        let (v, au) = dit.forward_watched(&on(&video)?, &on(&audio)?, (0.725, 0.725), &ctx, &grid, &mut |_, vx, ax| {
            out.push((host(vx)?, host(ax)?));
            Ok(())
        })?;
        out.push((host(&v)?, host(&au)?));
        drop(dit);
        device.synchronize()?;
        Ok(out)
    };
    let want = run(DType::F32)?;
    let got = run(DType::BF16)?;
    let fused = !matches!(std::env::var("KVAD_GPU_FUSED").as_deref(), Ok("0") | Ok("false"));
    println!("{}×{} × {}: bf16 against f32, fused kernels {}", shape.width, shape.height, shape.frames, if fused { "on" } else { "off" });
    for (i, ((gv, ga), (wv, wa))) in got.iter().zip(&want).enumerate() {
        let what = if i < blocks { format!("block {i}") } else { "velocity".into() };
        println!("{what:<9} video {:>5.1} dB   audio {:>5.1} dB", db(gv, wv)?, db(ga, wa)?);
    }
    Ok(())
}
