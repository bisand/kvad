//! Train a diffusion transformer to draw clips of two digits moving.
//!
//!     cargo run --release -p nervus --bin train_video
//!
//! Options: --steps N --batch N --lr F --d-model N --layers N --heads N
//!          --size apart|half|full --patch N --frames N --attention full|factorised
//!          --eval-every N --sample-steps N --guidance F --threads N --seed N
//!          --out DIR
//!
//! `--size apart` (the default) is two 14-pixel digits in a 32-pixel box,
//! starting together in the middle and moving apart, cut into 4×4 patches.
//! `half` is the same digits starting anywhere and going any way; after
//! 12,000 steps of one frame of those, the labels had hardly been learned
//! (see `moving::Motion::apart`). `full` is Moving MNIST as published,
//! 28-pixel digits in 64, and needs `--patch 8` to stay at 512 tokens. At 8 a token has 64
//! numbers to carry, twice what a model 128 wide can spare, and after 3000
//! steps it drew ink in patch-shaped blocks and no digits at all.
//!
//! The clips are Moving MNIST (see `nervus::moving`), made fresh for every
//! batch from the MNIST digits on disk. At every checkpoint the run draws the
//! same eight clips, each asked for two digits, and saves them as a film
//! strip: a row per clip, a column per frame.
//!
//! It also measures flicker, the way a video model most often goes wrong: a
//! clip whose every frame is a fine picture, but not the same picture a
//! moment later. Real clips change by a known amount from frame to frame;
//! a drawn clip that changes by much more is flickering, and one that
//! changes by much less is not moving.

use nervus::dit::{Attention, Dit, DitConfig};
use nervus::flow::{self, Replicas};
use nervus::mnist;
use nervus::moving::{self, Motion};
use nervus::optim::{AdamW, Schedule};
use nervus::png;
use nervus::rng::Rng;
use nervus::text::human_secs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct Args {
    steps: usize,
    batch: usize,
    lr: f32,
    d_model: usize,
    layers: usize,
    heads: usize,
    size: Motion,
    patch: usize,
    frames: usize,
    attention: Attention,
    eval_every: usize,
    sample_steps: usize,
    guidance: f32,
    threads: usize,
    seed: u64,
    out: PathBuf,
    /// A model to start from rather than a new one. With `--steps 0`, it
    /// only draws: the way to try a sampler setting without training again.
    load: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            steps: 3000,
            batch: 32,
            lr: 1e-3,
            d_model: 128,
            layers: 4,
            heads: 4,
            size: Motion::apart(),
            patch: 4,
            frames: 8,
            attention: Attention::Factorised,
            eval_every: 500,
            sample_steps: 20,
            guidance: 2.0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            seed: 1337,
            out: PathBuf::from("out/video"),
            load: None,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        let value = || -> String {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {flag}");
                std::process::exit(2);
            })
        };
        let number = || -> f64 {
            value().parse().unwrap_or_else(|_| {
                eprintln!("{flag} expects a number");
                std::process::exit(2);
            })
        };
        match flag.as_str() {
            "--steps" => a.steps = number() as usize,
            "--batch" => a.batch = number() as usize,
            "--lr" => a.lr = number() as f32,
            "--d-model" => a.d_model = number() as usize,
            "--layers" => a.layers = number() as usize,
            "--heads" => a.heads = number() as usize,
            "--patch" => a.patch = number() as usize,
            "--size" => {
                a.size = match value().as_str() {
                    "apart" => Motion::apart(),
                    "half" => Motion::half(),
                    "full" => Motion::default(),
                    other => {
                        eprintln!("--size is apart, half or full, not {other}");
                        std::process::exit(2);
                    }
                }
            }
            "--frames" => a.frames = number() as usize,
            "--attention" => {
                a.attention = match value().as_str() {
                    "full" => Attention::Full,
                    "factorised" | "factorized" => Attention::Factorised,
                    other => {
                        eprintln!("--attention is full or factorised, not {other}");
                        std::process::exit(2);
                    }
                }
            }
            "--eval-every" => a.eval_every = number() as usize,
            "--sample-steps" => a.sample_steps = number() as usize,
            "--guidance" => a.guidance = number() as f32,
            "--threads" => a.threads = number() as usize,
            "--seed" => a.seed = number() as u64,
            "--out" => a.out = PathBuf::from(value()),
            "--load" => a.load = Some(PathBuf::from(value())),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    a
}

/// The MNIST loader standardises; a clip is made of pixels from 0 to 1, and
/// of digits `digit` pixels wide.
fn unit_images(d: &mnist::Dataset, digit: usize) -> Vec<Vec<f32>> {
    (0..d.len())
        .map(|i| {
            let unit: Vec<f32> = d.images.row(i).iter().map(|&v| (v * 0.3081 + 0.1307).clamp(0.0, 1.0)).collect();
            moving::shrink(&unit, moving::DIGIT / digit)
        })
        .collect()
}

/// A clip as the model sees it: from -1 (paper) to 1 (ink).
fn signed(pixels: &[f32]) -> Vec<f32> {
    pixels.iter().map(|&v| 2.0 * v - 1.0).collect()
}

/// And back, clamped: what the model drew, as pixels from 0 to 1.
fn unit(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| ((v + 1.0) / 2.0).clamp(0.0, 1.0)).collect()
}

/// What each class is called, which is what a prompt asks for.
const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// The rate a clip is meant to be played at: eight frames, one second.
const FPS: u32 = 8;

/// The clips every checkpoint draws: two digits each, and a seed each.
const ASKED: [[usize; 2]; 8] = [[3, 7], [0, 1], [2, 5], [4, 9], [6, 8], [1, 1], [7, 2], [8, 3]];

/// A clip a row, a frame a column, two pixels apart, each pixel drawn as a
/// square big enough that a frame is 128 pixels across.
fn save_strip(path: &std::path::Path, clips: &[Vec<f32>], frames: usize, side: usize) -> std::io::Result<()> {
    let scale = (128 / side).max(1);
    const GAP: usize = 2;
    let cell = side + GAP;
    let (width, height) = (frames * cell * scale, clips.len() * cell * scale);
    let mut pixels = vec![40u8; width * height];
    for (row, clip) in clips.iter().enumerate() {
        for f in 0..frames {
            let frame = &clip[f * side * side..(f + 1) * side * side];
            let (top, left) = (row * cell * scale, f * cell * scale);
            for y in 0..side * scale {
                for x in 0..side * scale {
                    let v = frame[(y / scale) * side + x / scale];
                    pixels[(top + y) * width + left + x] = (v * 255.0).round() as u8;
                }
            }
        }
    }
    png::write_grey(path, width, height, &pixels)
}

/// What the model makes of a real clip after noise is added, in one step:
/// for each of `clips`, the clip, then the noisy clip and the model's guess
/// at the clean one, `x₀ ≈ xₜ − t·v`, at t = 0.5 and t = 0.9.
///
/// Drawing from pure noise needs the whole clip invented; this only needs
/// what is already there cleaned up. A model that does this well and draws
/// badly has learned to denoise and not yet what a clip should be; one that
/// does both badly has learned neither.
fn guess_rows(model: &mut Dit, clips: &[(&[f32], Vec<usize>)]) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    let mut rng = Rng::new(7);
    for (x0, labels) in clips {
        rows.push(unit(x0));
        let noise: Vec<f32> = (0..x0.len()).map(|_| rng.normal()).collect();
        for t in [0.5f32, 0.9] {
            let (xt, _) = flow::noisy(x0, &noise, t);
            let v = model.forward_labels(&xt, t, labels);
            let guess: Vec<f32> = xt.iter().zip(&v).map(|(x, v)| x - t * v).collect();
            rows.push(unit(&xt));
            rows.push(unit(&guess));
        }
    }
    rows
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let train = mnist::load(&data_dir, "train")?;
    let test = mnist::load(&data_dir, "t10k")?;
    let motion = Motion { frames: args.frames, ..args.size };
    let (train_images, test_images) = (unit_images(&train, motion.digit), unit_images(&test, motion.digit));
    std::fs::create_dir_all(&args.out)?;
    let config = DitConfig {
        image: motion.side,
        frames: args.frames,
        attention: args.attention,
        channels: 1,
        patch: args.patch,
        classes: 10,
        d_model: args.d_model,
        n_heads: args.heads,
        n_layers: args.layers,
    };
    let mut rng = Rng::new(args.seed);
    let mut model = match &args.load {
        Some(dir) => {
            // A model this binary saved, or the state of one.
            let model = match dir.join("model_index.json").exists() {
                true => nervus::dit::load_clips(dir)?.model,
                false => nervus::dit::load_state(dir)?,
            };
            if model.config() != config {
                eprintln!("{} is {:?}, and the flags ask for {config:?}", dir.display(), model.config());
                std::process::exit(2);
            }
            model
        }
        None => Dit::new(config, &mut rng),
    };
    println!("model: {}", model.summary());
    println!(
        "steps {} batch {} lr {} threads {}, drawing in {} steps at guidance {}\n",
        args.steps, args.batch, args.lr, args.threads, args.sample_steps, args.guidance
    );

    // The clips every checkpoint is measured on, from the test digits, made
    // once from a seed of their own; and how much real clips flicker.
    const VALIDATION_CLIPS: usize = 64;
    let mut held = Rng::new(20_260_930);
    let validation: Vec<moving::Clip> = (0..VALIDATION_CLIPS).map(|_| moving::clip(&motion, &test_images, &test.labels, &mut held)).collect();
    let real_flicker = validation.iter().map(|c| moving::flicker(&c.pixels, args.frames)).sum::<f32>() / VALIDATION_CLIPS as f32;
    let validation: Vec<(Vec<f32>, Vec<usize>)> = validation.into_iter().map(|c| (signed(&c.pixels), c.labels.to_vec())).collect();
    // The same clips told the wrong digits, each label moved on by five. How
    // much worse the model does on these is how much it listens to labels;
    // a model that ignores them does the same on both.
    let mislabelled: Vec<(&[f32], Vec<usize>)> =
        validation.iter().map(|(x, l)| (x.as_slice(), l.iter().map(|l| (l + 5) % 10).collect())).collect();
    let validation: Vec<(&[f32], Vec<usize>)> = validation.iter().map(|(x, l)| (x.as_slice(), l.clone())).collect();
    println!("real clips flicker {real_flicker:.4}: the mean change of a pixel from one frame to the next\n");

    let mut opt = AdamW::new(args.lr);
    opt.weight_decay = 0.0;
    opt.clip = Some(1.0);
    let schedule = Schedule { peak: args.lr, warmup: args.steps / 20, decay_to: 0.1 };
    let threads = args.threads.clamp(1, args.batch);
    let mut replicas = Replicas::new(&model, threads);

    let started = Instant::now();
    let mut training = Duration::ZERO;
    let (mut running, mut since, mut best) = (0.0, 0, f32::INFINITY);
    for step in 1..=args.steps {
        opt.lr = schedule.at(step, args.steps);
        let stepping = Instant::now();
        let examples: Vec<flow::Example> = (0..args.batch)
            .map(|_| {
                let c = moving::clip(&motion, &train_images, &train.labels, &mut rng);
                flow::noised(signed(&c.pixels), c.labels.to_vec(), config.unconditional(), &mut rng)
            })
            .collect();
        running += match threads {
            1 => flow::train_step(&mut model, &mut opt, &examples),
            _ => replicas.train_step(&mut model, &mut opt, &examples),
        };
        training += stepping.elapsed();
        since += 1;

        if step == 10.min(args.steps) {
            let rate = (step * args.batch) as f64 / training.as_secs_f64();
            let left = ((args.steps - step) * args.batch) as f64 / rate;
            println!("pace: {rate:.1} clips/s, about {} to go, not counting checkpoints\n", human_secs(left as f32));
        }

        if step % args.eval_every.max(1) == 0 || step == args.steps {
            let val = replicas.validate(&mut model, &validation);
            let gap = replicas.validate(&mut model, &mislabelled) - val;
            if val < best {
                best = val;
                nervus::dit::save_clips(&args.out.join("model"), &mut model, &DIGITS, 2, FPS)?;
            }
            let requests: Vec<(Vec<usize>, u64)> = ASKED.iter().enumerate().map(|(i, pair)| (pair.to_vec(), i as u64)).collect();
            let drawn: Vec<Vec<f32>> = replicas.sample(&mut model, &requests, args.sample_steps, args.guidance).iter().map(|x| unit(x)).collect();
            let drawn_flicker = drawn.iter().map(|c| moving::flicker(c, args.frames)).sum::<f32>() / drawn.len() as f32;
            save_strip(&args.out.join(format!("step-{step:06}.png")), &drawn, args.frames, motion.side)?;
            let guesses = guess_rows(&mut model, &validation[..2]);
            save_strip(&args.out.join(format!("guess-{step:06}.png")), &guesses, args.frames, motion.side)?;
            println!(
                "step {step:>6}  train {:.4}  val {val:.4}  label gap {gap:.4}  flicker {drawn_flicker:.4} (real {real_flicker:.4})  ({})",
                running / since as f32,
                human_secs(started.elapsed().as_secs_f32())
            );
            if step == args.steps {
                std::fs::copy(args.out.join(format!("step-{step:06}.png")), args.out.join("samples.png"))?;
            }
            (running, since) = (0.0, 0);
        }
    }
    if args.steps == 0 {
        let requests: Vec<(Vec<usize>, u64)> = ASKED.iter().enumerate().map(|(i, pair)| (pair.to_vec(), i as u64)).collect();
        let drawn: Vec<Vec<f32>> = replicas.sample(&mut model, &requests, args.sample_steps, args.guidance).iter().map(|x| unit(x)).collect();
        let name = format!("drawn-{}steps-g{}.png", args.sample_steps, args.guidance);
        save_strip(&args.out.join(&name), &drawn, args.frames, motion.side)?;
        println!("drew {} clips into {}", drawn.len(), args.out.join(name).display());
        return Ok(());
    }
    let rate = (args.steps * args.batch) as f64 / training.as_secs_f64();
    println!(
        "\ndone in {}. Best validation {best:.4}; samples in {}",
        human_secs(started.elapsed().as_secs_f32()),
        args.out.join("samples.png").display()
    );
    println!(
        "trained at {rate:.1} clips/s = {:.0} frames/s ({} in training steps)",
        rate * args.frames as f64,
        human_secs(training.as_secs_f32())
    );
    // kvad finds a model by name in its data directory's `models`; its
    // default is below.
    println!(
        "\nto draw with kvad, give it a name:\n  cp -R {} ~/.local/share/kvad/models/moving\n  kvad videos make \"3 and 7\" --model moving",
        args.out.join("model").display()
    );
    Ok(())
}
