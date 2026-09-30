//! Train a diffusion transformer to draw handwritten digits from noise.
//!
//!     cargo run --release -p nervus --bin train_digits
//!
//! Options: --steps N --batch N --lr F --d-model N --layers N --heads N
//!          --patch N --eval-every N --sample-steps N --guidance F
//!          --threads N --seed N --out DIR
//!
//! At every checkpoint it draws one grid, the same seeds every time: a row
//! per digit, 0 at the top. It also hands those drawings to a classifier
//! trained on real MNIST first, and prints how many it recognises as the
//! digit that was asked for. That is a crude measure. A model could draw a
//! perfect 7 every time and score 100% while having no variety at all. But
//! it is a number, and "the digits look right" is not.

use nervus::dit::{self, Dit, DitConfig};
use nervus::flow::{self, Replicas};
use nervus::matrix::Matrix;
use nervus::mnist;
use nervus::nn::{softmax_cross_entropy, Mlp};
use nervus::optim::{AdamW, Schedule};
use nervus::png;
use nervus::rng::Rng;
use nervus::text::human_secs;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    steps: usize,
    batch: usize,
    lr: f32,
    d_model: usize,
    layers: usize,
    heads: usize,
    patch: usize,
    eval_every: usize,
    sample_steps: usize,
    guidance: f32,
    threads: usize,
    seed: u64,
    out: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            steps: 4000,
            batch: 64,
            lr: 1e-3,
            d_model: 128,
            layers: 4,
            heads: 4,
            patch: 4,
            eval_every: 500,
            sample_steps: 20,
            guidance: 2.0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            seed: 1337,
            out: PathBuf::from("out/digits"),
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
            "--eval-every" => a.eval_every = number() as usize,
            "--sample-steps" => a.sample_steps = number() as usize,
            "--guidance" => a.guidance = number() as f32,
            "--threads" => a.threads = number() as usize,
            "--seed" => a.seed = number() as u64,
            "--out" => a.out = PathBuf::from(value()),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    a
}

/// A pixel as the diffusion model sees it, from -1 (paper) to 1 (ink).
///
/// The MNIST loader standardises for the classifier's sake; undo that and
/// stretch to [-1, 1], which is the range the noise is scaled against.
fn to_signed(standardised: f32) -> f32 {
    2.0 * (standardised * 0.3081 + 0.1307) - 1.0
}

/// And back, for the classifier.
fn to_standardised(signed: f32) -> f32 {
    (((signed + 1.0) / 2.0).clamp(0.0, 1.0) - 0.1307) / 0.3081
}

/// An MLP that reads digits, trained here in a few seconds, to judge the
/// drawn ones. It is `train_mnist`'s network with its default settings.
fn classifier(train: &mnist::Dataset, test: &mnist::Dataset, rng: &mut Rng) -> Mlp {
    let mut net = Mlp::new(&[784, 128, 10], rng);
    let mut order: Vec<usize> = (0..train.len()).collect();
    for _ in 0..3 {
        rng.shuffle(&mut order);
        for chunk in order.chunks(64) {
            let (x, y) = train.batch(chunk);
            let (_, dlogits) = softmax_cross_entropy(&net.forward(&x), &y);
            net.zero_grad();
            net.backward(&dlogits);
            net.step(0.02, 0.9);
        }
    }
    let all: Vec<usize> = (0..test.len()).collect();
    let (x, y) = test.batch(&all);
    let right = recognised(&mut net, &x, &y);
    println!("judge: an MLP that reads {:.1}% of the MNIST test set correctly", 100.0 * right as f32 / y.len() as f32);
    net
}

/// How many rows of `x` the classifier reads as the digit in `y`.
fn recognised(net: &mut Mlp, x: &Matrix, y: &[usize]) -> usize {
    let logits = net.forward(x);
    (0..logits.rows)
        .filter(|&r| {
            let row = logits.row(r);
            (0..10).fold(0, |best, j| if row[j] > row[best] { j } else { best }) == y[r]
        })
        .count()
}

/// Draw the grid, save it, and say how much of it the classifier recognises.
fn checkpoint_grid(
    model: &mut Dit,
    replicas: &mut Replicas,
    judge: &mut Mlp,
    args: &Args,
    columns: usize,
    name: &str,
) -> std::io::Result<(f32, Vec<Vec<f32>>)> {
    let requests: Vec<(usize, u64)> = (0..10 * columns).map(|i| (i / columns, (i % columns) as u64)).collect();
    let drawn = replicas.sample(model, &requests, args.sample_steps, args.guidance);

    let data = drawn.iter().flat_map(|img| img.iter().map(|&v| to_standardised(v))).collect();
    let labels: Vec<usize> = requests.iter().map(|&(label, _)| label).collect();
    let right = recognised(judge, &Matrix::from_vec(drawn.len(), 784, data), &labels);

    save_grid(&args.out.join(name), &drawn, columns)?;
    Ok((right as f32 / drawn.len() as f32, drawn))
}

/// Ten rows of `columns` digits, two pixels apart, each pixel drawn 3×3 so
/// that the grid is big enough to look at.
fn save_grid(path: &std::path::Path, images: &[Vec<f32>], columns: usize) -> std::io::Result<()> {
    const SCALE: usize = 3;
    const GAP: usize = 2;
    let cell = 28 + GAP;
    let (width, height) = (columns * cell * SCALE, images.len().div_ceil(columns) * cell * SCALE);
    let mut pixels = vec![0u8; width * height];
    for (i, image) in images.iter().enumerate() {
        let (top, left) = ((i / columns) * cell * SCALE, (i % columns) * cell * SCALE);
        for y in 0..28 * SCALE {
            for x in 0..28 * SCALE {
                let v = image[(y / SCALE) * 28 + x / SCALE];
                pixels[(top + y) * width + left + x] = (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    png::write_grey(path, width, height, &pixels)
}

/// Digits side by side as text, at half resolution, so a run can be watched
/// in a terminal.
fn render_row(images: &[Vec<f32>]) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    let mut s = String::new();
    for y in (0..28).step_by(2) {
        for image in images {
            for x in 0..14 {
                let v = (image[y * 28 + 2 * x] + image[y * 28 + 2 * x + 1]) / 2.0;
                let level = (((v + 1.0) / 2.0).clamp(0.0, 1.0) * (RAMP.len() - 1) as f32).round() as usize;
                s.push(RAMP[level] as char);
            }
            s.push(' ');
        }
        s.push('\n');
    }
    s
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let train = mnist::load(&data_dir, "train")?;
    let test = mnist::load(&data_dir, "t10k")?;
    let signed = |d: &mnist::Dataset| -> Vec<Vec<f32>> {
        (0..d.len()).map(|i| d.images.row(i).iter().map(|&v| to_signed(v)).collect()).collect()
    };
    let (train_images, test_images) = (signed(&train), signed(&test));
    std::fs::create_dir_all(&args.out)?;

    let mut rng = Rng::new(args.seed);
    let mut judge = classifier(&train, &test, &mut rng);

    let config = DitConfig {
        image: 28,
        channels: 1,
        patch: args.patch,
        classes: 10,
        d_model: args.d_model,
        n_heads: args.heads,
        n_layers: args.layers,
    };
    let mut model = Dit::new(config, &mut rng);
    println!("model: {}", model.summary());
    println!(
        "steps {} batch {} lr {} threads {}, drawing in {} steps at guidance {}\n",
        args.steps, args.batch, args.lr, args.threads, args.sample_steps, args.guidance
    );

    // DiT trains without weight decay, and so does this.
    let mut opt = AdamW::new(args.lr);
    opt.weight_decay = 0.0;
    opt.clip = Some(1.0);
    let schedule = Schedule { peak: args.lr, warmup: args.steps / 20, decay_to: 0.1 };
    let threads = args.threads.clamp(1, args.batch);
    let mut replicas = Replicas::new(&model, threads);

    const COLUMNS: usize = 10;
    const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
    const VALIDATION_IMAGES: usize = 200;
    let started = Instant::now();
    let (mut running, mut since, mut best) = (0.0, 0, f32::INFINITY);

    for step in 1..=args.steps {
        opt.lr = schedule.at(step, args.steps);
        let examples = flow::draw(&train_images, &train.labels, config.unconditional(), args.batch, &mut rng);
        running += match threads {
            1 => flow::train_step(&mut model, &mut opt, &train_images, &examples),
            _ => replicas.train_step(&mut model, &mut opt, &train_images, &examples),
        };
        since += 1;

        if step == 20.min(args.steps) {
            let rate = (step * args.batch) as f32 / started.elapsed().as_secs_f32();
            let left = ((args.steps - step) * args.batch) as f32 / rate;
            println!("pace: {rate:.0} images/s, about {} to go, not counting checkpoints\n", human_secs(left));
        }

        if step % args.eval_every.max(1) == 0 || step == args.steps {
            let val = flow::validate(&mut model, &test_images, &test.labels, VALIDATION_IMAGES);
            let saved = val < best;
            if saved {
                best = val;
                dit::save(&args.out.join("model"), &mut model, &DIGITS)?;
            }
            let (read, drawn) = checkpoint_grid(&mut model, &mut replicas, &mut judge, &args, COLUMNS, &format!("step-{step:06}.png"))?;
            println!(
                "step {step:>6}  train {:.4}  val {val:.4}{}  recognised {:>5.1}%  ({})",
                running / since as f32,
                if saved { " *" } else { "  " },
                100.0 * read,
                human_secs(started.elapsed().as_secs_f32())
            );
            if step == args.steps {
                let firsts: Vec<Vec<f32>> = drawn.iter().step_by(COLUMNS).cloned().collect();
                println!("\n{}", render_row(&firsts));
                std::fs::copy(args.out.join(format!("step-{step:06}.png")), args.out.join("samples.png"))?;
            }
            (running, since) = (0.0, 0);
        }
    }
    println!(
        "done in {}. Best validation {best:.4}, model in {}, samples in {}",
        human_secs(started.elapsed().as_secs_f32()),
        args.out.join("model").display(),
        args.out.join("samples.png").display()
    );
    // kvad finds a model by name in its data directory's `models`, and this
    // crate does not know where kvad keeps that; its default is below.
    println!(
        "\nto draw with kvad, give it a name:\n  cp -R {} ~/.local/share/kvad/models/digits\n  kvad images make 7 --model digits",
        args.out.join("model").display()
    );
    Ok(())
}
