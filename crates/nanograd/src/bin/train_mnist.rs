//! Train a multilayer perceptron to recognise handwritten digits.
//!
//!     cargo run --release -p nanograd --bin train_mnist
//!
//! Options: --epochs N --batch N --lr F --hidden N --seed N

use nanograd::mnist;
use nanograd::nn::{softmax_cross_entropy, Mlp};
use nanograd::rng::Rng;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    epochs: usize,
    batch: usize,
    lr: f32,
    momentum: f32,
    hidden: usize,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Args { epochs: 8, batch: 64, lr: 0.02, momentum: 0.9, hidden: 128, seed: 1337 }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        let value = |i: usize| -> String {
            argv.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {flag}");
                std::process::exit(2);
            })
        };
        let parse = |i: usize| -> f64 {
            value(i).parse().unwrap_or_else(|_| {
                eprintln!("{flag} expects a number");
                std::process::exit(2);
            })
        };
        match argv[i].as_str() {
            "--epochs" => a.epochs = parse(i) as usize,
            "--batch" => a.batch = parse(i) as usize,
            "--lr" => a.lr = parse(i) as f32,
            "--momentum" => a.momentum = parse(i) as f32,
            "--hidden" => a.hidden = parse(i) as usize,
            "--seed" => a.seed = parse(i) as u64,
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    a
}

/// Fraction of examples the network gets right.
fn accuracy(net: &mut Mlp, data: &mnist::Dataset, batch: usize) -> f32 {
    let mut correct = 0usize;
    for start in (0..data.len()).step_by(batch) {
        let idx: Vec<usize> = (start..(start + batch).min(data.len())).collect();
        let (x, y) = data.batch(&idx);
        let logits = net.forward(&x);
        for r in 0..logits.rows {
            let pred = argmax(logits.row(r));
            if pred == y[r] {
                correct += 1;
            }
        }
    }
    correct as f32 / data.len() as f32
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");

    let train = mnist::load(&data_dir, "train")?;
    let test = mnist::load(&data_dir, "t10k")?;
    println!("train: {} examples   test: {} examples", train.len(), test.len());

    let mut rng = Rng::new(args.seed);
    let mut net = Mlp::new(&[784, args.hidden, 10], &mut rng);
    println!("model:\n  {}", net.summary());
    println!(
        "hyperparams: epochs={} batch={} lr={} momentum={}\n",
        args.epochs, args.batch, args.lr, args.momentum
    );

    let mut order: Vec<usize> = (0..train.len()).collect();
    let started = Instant::now();

    for epoch in 1..=args.epochs {
        // Reshuffle every epoch. Without this the network sees the same
        // batches in the same order and can memorise the sequence rather
        // than the digits.
        rng.shuffle(&mut order);

        let mut running_loss = 0.0;
        let mut steps = 0usize;

        for chunk in order.chunks(args.batch) {
            let (x, y) = train.batch(chunk);

            // The four lines that are the entire training loop.
            let logits = net.forward(&x);                        // predict
            let (loss, dlogits) = softmax_cross_entropy(&logits, &y); // score
            net.zero_grad();
            net.backward(&dlogits);                              // blame
            net.step(args.lr, args.momentum);                    // adjust

            running_loss += loss;
            steps += 1;
        }

        let acc = accuracy(&mut net, &test, 512);
        println!(
            "epoch {epoch:>2}  loss {:.4}  test accuracy {:.2}%  ({:.1}s)",
            running_loss / steps as f32,
            acc * 100.0,
            started.elapsed().as_secs_f32()
        );
    }

    // Show the network's work on a few test digits, including at least one
    // it gets wrong -- the mistakes are usually more interesting.
    println!("\n--- sample predictions ---");
    let mut shown_wrong = 0;
    let mut shown_right = 0;
    for i in 0..test.len() {
        let (x, y) = test.batch(&[i]);
        let logits = net.forward(&x);
        let pred = argmax(logits.row(0));
        let wrong = pred != y[0];
        if (wrong && shown_wrong < 2) || (!wrong && shown_right < 1) {
            if wrong {
                shown_wrong += 1;
            } else {
                shown_right += 1;
            }
            println!("{}", test.render(i));
            println!(
                "  true {}   predicted {}   {}\n",
                y[0],
                pred,
                if wrong { "WRONG" } else { "correct" }
            );
        }
        if shown_wrong >= 2 && shown_right >= 1 {
            break;
        }
    }

    Ok(())
}
