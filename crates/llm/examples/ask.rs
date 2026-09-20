//! Search a corpus from the command line, and see why each hit came back.
//!
//!     cargo run --release -p kvad --example ask -- CORPUS "a question"
//!
//! The half of retrieval that can be measured on its own: whether the right
//! chunk comes back has nothing to do with whether a model then reads it
//! properly, and the two fail for different reasons.

use kvad::retrieve::{self, Index};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(path), Some(question)) = (args.first(), args.get(1)) else {
        eprintln!("usage: ask CORPUS \"a question\"  [how many]");
        std::process::exit(2);
    };
    let k: usize = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(5);

    let corpus = std::fs::read_to_string(path).expect("could not read the corpus");
    // The manifest beside it, if a crawl left one, so hits can cite pages.
    let manifest = std::path::PathBuf::from(format!("{path}.crawl.json"));
    let pages = pages_of(&manifest);

    let started = std::time::Instant::now();
    let chunks = retrieve::chunk(&corpus, &pages);
    let chunked = started.elapsed();
    let index = Index::build(chunks);
    let built = started.elapsed();

    println!(
        "{} characters -> {} chunks, {} words: chunked in {:?}, indexed in {:?}",
        corpus.chars().count(),
        index.len(),
        index.vocabulary(),
        chunked,
        built - chunked
    );

    let asked = std::time::Instant::now();
    let hits = index.search(question, k);
    println!("searched in {:?}\n", asked.elapsed());

    for hit in &hits {
        let why: Vec<String> =
            hit.because.iter().take(4).map(|(w, s)| format!("{w} {s:.2}")).collect();
        println!("{:>7.2}  {}", hit.score, hit.chunk.heading);
        println!("         {}", hit.chunk.source.as_deref().unwrap_or("-"));
        println!("         {}", why.join(", "));
        let first = hit.chunk.text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        println!("         {}\n", first.chars().take(90).collect::<String>());
    }
}

fn pages_of(manifest: &std::path::Path) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(manifest) else { return Vec::new() };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    json.get("pages")
        .and_then(|p| p.as_array())
        .map(|pages| {
            pages
                .iter()
                .filter_map(|p| {
                    Some((p.get("title")?.as_str()?.to_string(), p.get("url")?.as_str()?.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}
