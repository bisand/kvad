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
//!
//! Flicker cannot see a digit that turns into another a little at a time,
//! so for clips of digits moving apart a judge reads them back: a network
//! trained on windows of real clips, where each digit is known to be. At
//! every checkpoint it reads all 55 pairs of digits, drawn, and says how
//! often they were the digits asked for and how often each stayed the same
//! digit all the way through; and first, what it scores on real clips, which
//! is the most a model can.

use nervus::dit::{Attention, Dit, DitConfig};
use nervus::flow::{self, Replicas};
use nervus::matrix::Matrix;
use nervus::mnist;
use nervus::moving::{self, Motion, Start};
use nervus::nn::{softmax_cross_entropy, Mlp};
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

// ---------------------------------------------------------------------------
// The judge
// ---------------------------------------------------------------------------

/// Reads which digit is where in a clip of digits moving apart.
///
/// Every such clip makes the same journey, so where each digit is in every
/// frame is known, and reading a clip is reading one window at a time
/// ([`moving::window`]). The reader is `train_mnist`'s network, trained on
/// windows of real clips, overlaps and all, so that it learns these digits
/// where they are rather than as MNIST has them.
///
/// It cannot read every frame. In the first, the two windows are the same
/// window, holding both digits; they only come apart as the clip goes on.
/// So it is measured on real clips first, frame by frame, and only the
/// frames it reads reliably are used to judge anything.
struct Judge {
    net: Mlp,
    /// The frames it reads at least [`RELIABLE`] of real windows right in.
    frames: Vec<usize>,
    /// Its own score on real clips: the best a model could get from it.
    ceiling: Score,
}

/// What counts as a frame the judge can read.
const RELIABLE: f32 = 0.9;

/// A window as the reader wants it, standardised as MNIST is.
fn standardised(window: &[f32]) -> Vec<f32> {
    window.iter().map(|&v| (v - 0.1307) / 0.3081).collect()
}

/// What the judge reads in each of `frames`: the two digits, first slot first.
fn read(net: &mut Mlp, motion: &Motion, pixels: &[f32], frames: &[usize]) -> Vec<[usize; 2]> {
    let d = motion.digit;
    let mut x = Matrix::zeros(2 * frames.len(), d * d);
    for (i, &f) in frames.iter().enumerate() {
        for slot in 0..2 {
            x.row_mut(2 * i + slot).copy_from_slice(&standardised(&moving::window(pixels, motion, f, slot)));
        }
    }
    let logits = net.forward(&x);
    let best = |r: usize| (0..10).fold(0, |b, j| if logits.get(r, j) > logits.get(r, b) { j } else { b });
    (0..frames.len()).map(|i| [best(2 * i), best(2 * i + 1)]).collect()
}

/// How a set of clips reads, asked for two digits each.
#[derive(Clone, Copy)]
struct Score {
    /// Of all judged frames, the share where both digits read as asked,
    /// either way round. (The model adds its two labels up, so it cannot
    /// know which is meant to go where.)
    both: f32,
    /// Of the clips, the share whose most common reading in each place is
    /// what was asked for: right on the whole, forgiving a frame or two, of
    /// the model's and of the judge's.
    mostly: f32,
    /// Of the clips, the share where each digit reads the same in every
    /// judged frame: a digit that becomes another one on the way fails.
    held: f32,
    /// Both at once: the digits asked for, all the way through.
    held_right: f32,
}

fn score(net: &mut Mlp, motion: &Motion, frames: &[usize], clips: &[(Vec<f32>, [usize; 2])]) -> Score {
    let read: Vec<(Vec<[usize; 2]>, [usize; 2])> = clips.iter().map(|(pixels, asked)| (read(net, motion, pixels, frames), *asked)).collect();
    tally(&read)
}

/// [`Score`] from what was read in each judged frame of each clip, and what
/// that clip was asked for.
fn tally(clips: &[(Vec<[usize; 2]>, [usize; 2])]) -> Score {
    let same = |a: [usize; 2], b: [usize; 2]| a == b || a == [b[1], b[0]];
    // The digit read most often in one place; the lower one on a tie.
    let most = |reads: &[[usize; 2]], slot: usize| {
        let mut counts = [0; 10];
        reads.iter().for_each(|r| counts[r[slot]] += 1);
        (0..10).fold(0, |best, d| if counts[d] > counts[best] { d } else { best })
    };
    let (mut both, mut frames, mut mostly, mut held, mut held_right) = (0, 0, 0, 0, 0);
    for (reads, asked) in clips {
        both += reads.iter().filter(|&&r| same(r, *asked)).count();
        frames += reads.len();
        mostly += same([most(reads, 0), most(reads, 1)], *asked) as usize;
        if reads.iter().all(|&r| r == reads[0]) {
            held += 1;
            held_right += same(reads[0], *asked) as usize;
        }
    }
    let n = clips.len() as f32;
    Score {
        both: both as f32 / frames as f32,
        mostly: mostly as f32 / n,
        held: held as f32 / n,
        held_right: held_right as f32 / n,
    }
}

impl Score {
    fn line(&self) -> String {
        format!(
            "both right in {:.0}% of frames; of clips, {:.0}% mostly right, {:.0}% held every frame, {:.0}% both",
            100.0 * self.both,
            100.0 * self.mostly,
            100.0 * self.held,
            100.0 * self.held_right
        )
    }
}

fn train_judge(motion: &Motion, train: (&[Vec<f32>], &[usize]), test: (&[Vec<f32>], &[usize]), rng: &mut Rng) -> Judge {
    const CLIPS: usize = 6000;
    const TEST_CLIPS: usize = 500;
    let (d, frames) = (motion.digit, motion.frames);
    let mut x = Matrix::zeros(CLIPS * frames * 2, d * d);
    let mut y = Vec::with_capacity(CLIPS * frames * 2);
    for c in 0..CLIPS {
        let clip = moving::clip(motion, train.0, train.1, rng);
        for f in 0..frames {
            for slot in 0..2 {
                x.row_mut((c * frames + f) * 2 + slot).copy_from_slice(&standardised(&moving::window(&clip.pixels, motion, f, slot)));
                y.push(clip.labels[slot]);
            }
        }
    }
    let mut net = Mlp::new(&[d * d, 128, 10], rng);
    let mut order: Vec<usize> = (0..y.len()).collect();
    for _ in 0..3 {
        rng.shuffle(&mut order);
        for chunk in order.chunks(64) {
            let mut xb = Matrix::zeros(chunk.len(), d * d);
            for (r, &i) in chunk.iter().enumerate() {
                xb.row_mut(r).copy_from_slice(x.row(i));
            }
            let yb: Vec<usize> = chunk.iter().map(|&i| y[i]).collect();
            let (_, dlogits) = softmax_cross_entropy(&net.forward(&xb), &yb);
            net.zero_grad();
            net.backward(&dlogits);
            net.step(0.02, 0.9);
        }
    }

    // Measured on clips of digits it has not seen, frame by frame.
    let mut held_out = Rng::new(20_261_001);
    let real: Vec<(Vec<f32>, [usize; 2])> =
        (0..TEST_CLIPS).map(|_| moving::clip(motion, test.0, test.1, &mut held_out)).map(|c| (c.pixels, c.labels)).collect();
    let all: Vec<usize> = (0..frames).collect();
    let mut right = vec![0; frames];
    for (pixels, labels) in &real {
        for (f, r) in read(&mut net, motion, pixels, &all).into_iter().enumerate() {
            right[f] += (r[0] == labels[0]) as usize + (r[1] == labels[1]) as usize;
        }
    }
    let per_frame: Vec<f32> = right.iter().map(|&r| r as f32 / (2 * TEST_CLIPS) as f32).collect();
    let judged: Vec<usize> = (0..frames).filter(|&f| per_frame[f] >= RELIABLE).collect();
    println!(
        "judge: reads real windows {}; judging frames {:?}",
        per_frame.iter().enumerate().map(|(f, a)| format!("{f}:{:.0}%", 100.0 * a)).collect::<Vec<_>>().join(" "),
        judged
    );
    let ceiling = score(&mut net, motion, &judged, &real);
    println!("judge on real clips (the most a model can score): {}\n", ceiling.line());
    Judge { net, frames: judged, ceiling }
}

/// Every unordered pair of digits, doubles included: 55 clips, one seed each.
fn every_pair() -> Vec<(Vec<usize>, u64)> {
    let pairs = (0..10).flat_map(|a| (a..10).map(move |b| vec![a, b]));
    pairs.enumerate().map(|(i, p)| (p, 1000 + i as u64)).collect()
}

/// Draw every pair and read them back.
fn judge_model(judge: &mut Judge, replicas: &mut Replicas, model: &mut Dit, motion: &Motion, args: &Args) -> Score {
    let requests = every_pair();
    let drawn = replicas.sample(model, &requests, args.sample_steps, args.guidance);
    let clips: Vec<(Vec<f32>, [usize; 2])> =
        drawn.iter().zip(&requests).map(|(x, (asked, _))| (unit(x), [asked[0], asked[1]])).collect();
    score(&mut judge.net, motion, &judge.frames, &clips)
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
    let mut judge = match motion.start {
        Start::Apart => Some(train_judge(&motion, (&train_images, &train.labels), (&test_images, &test.labels), &mut Rng::new(args.seed ^ 0x5eed))),
        Start::Anywhere => None,
    };

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
            if let Some(judge) = judge.as_mut() {
                let s = judge_model(judge, &mut replicas, &mut model, &motion, &args);
                println!("             judged: {}", s.line());
            }
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
        if let Some(judge) = judge.as_mut() {
            let s = judge_model(judge, &mut replicas, &mut model, &motion, &args);
            println!("every pair of digits, drawn and read back: {}", s.line());
            println!("                           real clips, the most: {}", judge.ceiling.line());
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Four clips asked for a 3 and a 7, read over three frames: one right
    /// throughout (the other way round, which counts), one whose 7 turns into
    /// a 1 for the last frame, one with a misread in the middle, and one that
    /// is the wrong digits all the way.
    #[test]
    fn a_digit_that_changes_is_not_held_and_a_wrong_one_is_not_right() {
        let asked = [3, 7];
        let clips = vec![
            (vec![[7, 3], [7, 3], [7, 3]], asked),
            (vec![[3, 7], [3, 7], [3, 1]], asked),
            (vec![[3, 7], [8, 7], [3, 7]], asked),
            (vec![[5, 5], [5, 5], [5, 5]], asked),
        ];
        let s = tally(&clips);
        assert_eq!(s.both, 7.0 / 12.0);
        assert_eq!(s.mostly, 0.75, "the first three are right on the whole");
        assert_eq!(s.held, 0.5, "the first and the last never change");
        assert_eq!(s.held_right, 0.25, "and only the first is also right");
    }
}
