//! Write a test clip: `kvad::video` on frames and sound whose content is
//! known, so that what a player decodes can be checked against it.
//!
//! ```text
//! cargo run -p kvad --example video_pattern -- 768 512 24 121 /tmp/pattern
//! ```
//!
//! writes `/tmp/pattern.mp4` (video and sound), `pattern.wav` (the sound
//! alone) and `pattern.rgb` (the frames as they went in, for comparing).
//!
//! Each frame is a diagonal colour gradient that moves one step a frame, with
//! the frame's number in binary across the top: sixteen 16×16 squares, white
//! for 1 and black for 0, most significant first. A decoder that drops,
//! repeats or reorders frames shows it there. The sound is 440 Hz on the
//! left and 660 Hz on the right, with a click at the start of every second.

use kvad::video::{Audio, Video};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [w, h, fps, frames] = [0, 1, 2, 3].map(|i| args.get(i).and_then(|a| a.parse::<usize>().ok()));
    let (Some(w), Some(h), Some(fps), Some(frames), Some(out)) = (w, h, fps, frames, args.get(4)) else {
        eprintln!("usage: video_pattern WIDTH HEIGHT FPS FRAMES OUT_PREFIX");
        std::process::exit(2);
    };

    let mut rgb = Vec::with_capacity(w * h * 3 * frames);
    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                let bit = if y < 16 && x < 256 { Some(f >> (15 - x / 16) & 1 == 1) } else { None };
                let px = match bit {
                    Some(true) => [255, 255, 255],
                    Some(false) => [0, 0, 0],
                    None => [((x + 4 * f) * 255 / (w + 4 * frames)) as u8, (y * 255 / h) as u8, ((x + y) * 255 / (w + h)) as u8],
                };
                rgb.extend_from_slice(&px);
            }
        }
    }
    let video = Video { width: w, height: h, fps: fps as u32, rgb };

    let rate = 48000;
    let n = frames * rate / fps;
    let mut samples = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        let click = if i % rate < 48 { 0.8 } else { 0.0 };
        samples.push(0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + click);
        samples.push(0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin() + click);
    }
    let audio = Audio { rate: rate as u32, channels: 2, samples };

    let t = std::time::Instant::now();
    let mp4 = video.mp4(Some(&audio));
    let took = t.elapsed();
    std::fs::write(format!("{out}.mp4"), &mp4).unwrap();
    std::fs::write(format!("{out}.wav"), audio.wav()).unwrap();
    std::fs::write(format!("{out}.rgb"), &video.rgb).unwrap();
    println!("{out}.mp4: {frames} frames of {w}x{h} at {fps} fps, {:.1} MB, written in {:.2?}", mp4.len() as f64 / 1e6, took);
}
