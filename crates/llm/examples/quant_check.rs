//! Compare one checkpoint's weights against another's, tensor by tensor.
//!
//!     cargo run --release -p kvad --example quant_check -- REPO_A REPO_B
//!
//! Written for one question. An fp8 checkpoint stores `X.weight` beside
//! `X.weight_scale_inv`, and the name says *inverse* while the convention
//! multiplies. Reading it the wrong way round does not fail: it returns
//! finite numbers, of the wrong magnitude, and a model built from them
//! still generates text. A quantised publication and its original are the
//! same weights at two precisions, so comparing them settles it in a way
//! that reading the output never could.
//!
//! Reports, per tensor and overall, the relative L2 distance and the ratio
//! of norms. The ratio is the diagnostic: an inverted scale shows up there
//! as a number nowhere near one, whatever the distance says.

use kvad::weights::Checkpoint;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(a_id), Some(b_id)) = (args.first(), args.get(1)) else {
        eprintln!("usage: quant_check REPO_A REPO_B");
        std::process::exit(2);
    };
    let show: usize = args
        .iter()
        .position(|a| a == "--show")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(8);

    let open = |id: &String| match kvad::weights::fetch(id).and_then(|f| Checkpoint::open(&f.weights)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not open {id}: {e}");
            std::process::exit(1);
        }
    };
    let (a, b) = (open(a_id), open(b_id));

    // The scale tensors have no counterpart in the original, by design:
    // they are how the quantised one stores what the original holds inline.
    let mut names: Vec<String> = a
        .names()
        .filter(|n| !n.ends_with("_scale_inv") && !n.ends_with("_scale"))
        .map(str::to_string)
        .collect();
    names.sort();

    println!("{a_id}\n  against {b_id}\n");
    println!("  {:>10} {:>10}  {}", "rel L2", "|a|/|b|", "tensor");

    let (mut worst, mut worst_name) = (0.0f64, String::new());
    let (mut sum, mut n) = (0.0f64, 0usize);
    let mut shown = 0;
    for name in &names {
        let (Some(ta), Some(tb)) = (a.try_get(name), b.try_get(name)) else { continue };
        if ta.data.len() != tb.data.len() {
            println!("  {:>10} {:>10}  {name}  (shapes differ)", "-", "-");
            continue;
        }
        let mut num = 0.0f64;
        let (mut da, mut db) = (0.0f64, 0.0f64);
        for (x, y) in ta.data.iter().zip(&tb.data) {
            let (x, y) = (*x as f64, *y as f64);
            num += (x - y) * (x - y);
            da += x * x;
            db += y * y;
        }
        let rel = if db > 0.0 { (num / db).sqrt() } else { f64::NAN };
        let ratio = if db > 0.0 { (da / db).sqrt() } else { f64::NAN };
        if rel > worst {
            worst = rel;
            worst_name = name.clone();
        }
        sum += rel;
        n += 1;
        if shown < show {
            println!("  {rel:>10.5} {ratio:>10.5}  {name}");
            shown += 1;
        }
    }

    if n == 0 {
        eprintln!("\nno tensor is in both checkpoints under the same name");
        std::process::exit(1);
    }
    println!("\n  {n} tensors · mean rel L2 {:.5} · worst {:.5} at {worst_name}", sum / n as f64, worst);
    // fp8 keeps three mantissa bits over a block of 128 by 128, so a few per
    // cent is what agreement looks like. An inverted scale is not off by a
    // few per cent; it is off by orders of magnitude, and says so here.
    println!(
        "\n  A correctly-read fp8 checkpoint sits a few per cent from its\n  \
         original, with the norm ratio near 1. Anything else is not noise."
    );
}
