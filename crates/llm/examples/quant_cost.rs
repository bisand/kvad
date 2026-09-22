//! What quantisation costs, in the only currency that matters.
//!
//!     cargo run --release -p kvad --example quant_cost -- MODEL FILE [FILE...]
//!
//! Scores the same text at every precision this build has and prints the
//! perplexities side by side. Not a benchmark of speed: a measurement of
//! how much worse the model's predictions get when its weights are
//! rounded.
//!
//! # Why it exists
//!
//! "q8 is free and q4 is close enough" is folklore, repeated about dense
//! models and then applied to mixtures without anyone checking. A mixture
//! has reason to be different: a dense feed-forward runs every weight on
//! every token, so rounding errors are averaged over the whole corpus,
//! while an expert that fires on one token in twenty sees a twentieth of
//! the traffic and averages its errors over that much less. Whether that
//! matters is a question about a specific model, and this answers it for
//! whichever one you point it at.
//!
//! # Reading the output
//!
//! Perplexity is exp of the mean negative log-likelihood: the number of
//! equally-likely tokens the model is effectively choosing between. Lower
//! is better, and the useful figure is the *ratio* between precisions
//! rather than any single value, because the absolute number says as much
//! about the text as about the model.
//!
//! Score several kinds of text. Quantisation does not degrade every domain
//! equally, and code -- which is full of exact tokens that are either right
//! or wrong -- tends to show damage that prose absorbs.

use kvad::quant::Precision;
use kvad::runtime::Llm;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model), files) = (args.first(), &args[1.min(args.len())..]) else {
        eprintln!("usage: quant_cost MODEL FILE [FILE...]");
        std::process::exit(2);
    };
    if files.is_empty() {
        eprintln!("usage: quant_cost MODEL FILE [FILE...]");
        std::process::exit(2);
    }

    // Read every text up front: a file that is missing should say so now,
    // not after twenty minutes of quantising.
    let skip = args.iter().position(|a| a == "--prec").map(|i| [i, i + 1]);
    let files: Vec<&String> = files
        .iter()
        .enumerate()
        .filter(|(i, _)| skip.is_none_or(|s| !s.contains(&(i + 1))))
        .map(|(_, f)| f)
        .collect();
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|f| {
            let body = std::fs::read_to_string(*f).unwrap_or_else(|e| {
                eprintln!("could not read {f}: {e}");
                std::process::exit(1);
            });
            let name = std::path::Path::new(f)
                .file_name()
                .map_or_else(|| (*f).clone(), |n| n.to_string_lossy().into_owned());
            (name, body)
        })
        .collect();

    // f32 first, so that the honest reference is measured before anything
    // is compared against it. A model too big to hold at f32 will simply be
    // slow here rather than wrong.
    let precisions: Vec<Precision> = match args.iter().position(|a| a == "--prec") {
        Some(i) => args[i + 1]
            .split(',')
            .map(|p| match p.trim() {
                "f32" => Precision::F32,
                "q8" => Precision::Q8,
                "q4" => Precision::Q4,
                other => {
                    eprintln!("unknown precision `{other}`: try f32, q8 or q4");
                    std::process::exit(2);
                }
            })
            .collect(),
        // A model that does not fit at f32 will thrash rather than fail, so
        // the full sweep is the default and `--prec` is how you say you
        // already know which end of it is affordable.
        None => vec![Precision::F32, Precision::Q8, Precision::Q4],
    };
    let mut table: Vec<(Precision, Vec<f64>)> = Vec::new();

    for precision in precisions.iter().copied() {
        eprintln!("loading {model} at {precision:?} ...");
        let mut llm = match Llm::load(model, precision) {
            Ok(llm) => llm,
            Err(e) => {
                eprintln!("  skipped: {e}");
                continue;
            }
        };
        let mut row = Vec::new();
        for (name, text) in &texts {
            match llm.perplexity(text, 512, |_, _| true) {
                Ok(p) => {
                    eprintln!(
                        "  {name}: ppl {:.4} over {} tokens in {} windows",
                        p.perplexity, p.scored, p.windows
                    );
                    row.push(p.perplexity);
                }
                Err(e) => {
                    eprintln!("  {name}: {e}");
                    row.push(f64::NAN);
                }
            }
        }
        table.push((precision, row));
    }

    if table.is_empty() {
        eprintln!("nothing loaded; no table to print");
        std::process::exit(1);
    }

    println!("\n{model}\n");
    print!("{:<8}", "");
    for (name, _) in &texts {
        print!("{name:>22}");
    }
    println!();

    // Everything is quoted against the first precision that loaded, which
    // is the least rounded one available.
    let (base_precision, base) = table[0].clone();
    for (precision, row) in &table {
        print!("{:<8}", format!("{precision:?}"));
        for (i, ppl) in row.iter().enumerate() {
            let delta = (ppl / base[i] - 1.0) * 100.0;
            print!("{:>13.4}{:>9}", ppl, format!("{delta:+.2}%"));
        }
        println!();
    }
    println!("\nperplexity, and the change against {base_precision:?}. Lower is better.");
}
