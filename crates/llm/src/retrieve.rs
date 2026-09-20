//! Finding the part of a corpus that answers a question.
//!
//! Training cannot put facts into one of these models — a character model has
//! nowhere to keep them, and the downloaded models cannot be trained here at
//! all. So the other way round: keep the text, find the part of it that bears
//! on the question, and put that in the prompt. The model does not know
//! anything new; it is reading.
//!
//! # Why the index is arithmetic and not a second model
//!
//! The obvious modern answer is embeddings — run every chunk through a model,
//! run the question through the same model, and compare the vectors. It needs
//! a model this repository does not have (a BERT-class encoder is a whole
//! architecture, not a feature) and it is worse at exactly the thing a
//! documentation corpus is made of: `Vec<T>`, `unwrap`, `trpl::join`. Those
//! are not fuzzy concepts to be matched by meaning, they are strings, and a
//! question containing one is nearly always about it.
//!
//! So: BM25, which is thirty years old, is four lines of arithmetic, and can
//! be read off the page. [`Index::search`] says what each term contributed,
//! which is the thing an embedding cannot do at all.
//!
//! # No stop-word list
//!
//! "the" and "a" are in every chunk, so their inverse document frequency is
//! near zero and they change no ranking. Dropping them by hand would be
//! keeping a list to do what the formula already does — and the list would be
//! wrong for a corpus about the word `if`.

use std::collections::HashMap;

/// One piece of a corpus, and where it came from.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Chunk {
    /// Position in the index, which is what a hit refers to.
    pub id: usize,
    /// The headings above it, outermost first: "Common Programming Concepts
    /// › Variables and Mutability". What a citation shows, and — because a
    /// heading says what a section is about in the words somebody searching
    /// would use — also part of what is searched.
    pub heading: String,
    pub text: String,
    /// The page this came from, as the crawl manifest names it, when it could
    /// be worked out. `None` for an uploaded corpus, which has no pages.
    pub source: Option<String>,
}

impl Chunk {
    /// Heading and body, which is what both the reader and the searcher see.
    fn searchable(&self) -> impl Iterator<Item = &str> {
        [self.heading.as_str(), self.text.as_str()].into_iter()
    }
}

/// How big a chunk should be, in characters.
///
/// About 300 tokens of English at four characters to the token — small enough
/// that five of them fit a small model's context with room for the question,
/// big enough to hold an explanation and the code under it. A section shorter
/// than this is left alone; only a longer one is split.
pub const TARGET: usize = 1200;

/// Split a corpus into chunks on its headings.
///
/// The crawler writes one `#` heading per page and leaves the page's own
/// `##`s below it, so the heading structure is already there to cut along.
/// A corpus with no headings at all — an uploaded text file — is cut into
/// [`TARGET`]-sized pieces on paragraph boundaries instead.
pub fn chunk(corpus: &str, pages: &[(String, String)]) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    // One entry per heading level, so that `##` after `# Page` gives the path
    // "Page › Section" and the next `#` clears what was under it.
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut body = String::new();

    let mut flush = |path: &[(usize, String)], body: &mut String| {
        let text = body.trim();
        if !text.is_empty() {
            let heading =
                path.iter().map(|(_, h)| h.as_str()).collect::<Vec<_>>().join(" \u{203a} ");
            let source = page_of(path, pages);
            for piece in split(text) {
                chunks.push(Chunk {
                    id: chunks.len(),
                    heading: heading.clone(),
                    text: piece,
                    source: source.clone(),
                });
            }
        }
        body.clear();
    };

    for line in corpus.lines() {
        let hashes = line.bytes().take_while(|b| *b == b'#').count();
        let is_heading = (1..=6).contains(&hashes) && line[hashes..].starts_with(' ');
        if !is_heading {
            body.push_str(line);
            body.push('\n');
            continue;
        }
        flush(&path, &mut body);
        // A heading closes every heading at its level or deeper.
        path.retain(|(level, _)| *level < hashes);
        path.push((hashes, line[hashes..].trim().to_string()));
    }
    flush(&path, &mut body);
    chunks
}

/// The page a heading path belongs to, by matching its outermost heading
/// against the titles in a crawl manifest.
///
/// Best effort, and it says so by returning `None`: the corpus keeps a page's
/// own `# Heading` where it has one, and a site's `<title>` is usually that
/// heading plus the site's name — "Ownership - The Rust Programming
/// Language". Matching one inside the other catches that and gives up
/// quietly on anything else, which is better than citing the wrong page.
fn page_of(path: &[(usize, String)], pages: &[(String, String)]) -> Option<String> {
    let top = path.first()?.1.to_lowercase();
    let top = top.trim();
    if top.is_empty() {
        return None;
    }
    pages
        .iter()
        .find(|(title, _)| {
            let title = title.to_lowercase();
            title == top || title.starts_with(&format!("{top} "))
        })
        .map(|(_, url)| url.clone())
}

/// Cut a section into pieces of about [`TARGET`], on blank lines.
///
/// Never inside a paragraph, and never inside a fenced code block: half a
/// code block is worse than no code block, and a model reading it will
/// finish the fence itself and invent the rest.
fn split(text: &str) -> Vec<String> {
    if text.chars().count() <= TARGET {
        return vec![text.to_string()];
    }
    let mut pieces = Vec::new();
    let mut piece = String::new();
    let mut fenced = false;
    for para in text.split("\n\n") {
        let long_enough = piece.chars().count() + para.chars().count() > TARGET;
        if long_enough && !fenced && !piece.is_empty() {
            pieces.push(piece.trim().to_string());
            piece.clear();
        }
        // An odd number of fences in this paragraph flips whether we are
        // inside a code block.
        fenced ^= para.lines().filter(|l| l.trim_start().starts_with("```")).count() % 2 == 1;
        piece.push_str(para);
        piece.push_str("\n\n");
    }
    if !piece.trim().is_empty() {
        pieces.push(piece.trim().to_string());
    }
    pieces
}

/// The words of a string, as the index knows them.
///
/// Lowercased, and cut at everything that is not a letter or a digit — so
/// `Vec<String>` is `vec` and `string`, and `trpl::join` is `trpl` and
/// `join`. A question asking about either finds both halves, which for code
/// is the behaviour worth having: nobody searching for `join` should miss the
/// page that spells it `trpl::join`.
///
/// Single characters are dropped, and that is not tidying. "How do I make a
/// vector" contains the word `i`, which this cutting rule also finds inside
/// `I/O` — and because almost no chunk holds a lone `i`, it is *rare*, so
/// BM25 pays well for it. Measured on the Rust book, `i` scored 5.1 against
/// `vector`'s 5.0, and a chapter on I/O with no vector in it outranked the
/// chapter on vectors. A passage is never relevant because it contains a
/// single letter, in prose or in code: `Box<T>` is found by `box`.
pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().nth(1).is_some())
        .map(|w| w.to_lowercase())
}

/// What one chunk scored, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hit {
    pub chunk: Chunk,
    pub score: f32,
    /// What each of the question's words contributed, largest first. The
    /// answer to "why did this come back", which is the question an embedding
    /// index cannot answer at all.
    pub because: Vec<(String, f32)>,
}

/// A searchable corpus.
pub struct Index {
    chunks: Vec<Chunk>,
    /// term -> the chunks holding it, and how often.
    postings: HashMap<String, Vec<(u32, u32)>>,
    /// Words in each chunk, by chunk id.
    lengths: Vec<u32>,
    average: f32,
}

/// How much a repeat of the same word is worth. 1.2 is the value everyone
/// uses; above it, the tenth `unwrap` counts nearly as much as the second.
const K1: f32 = 1.2;

/// How much to hold a long chunk's word counts against it. 0.75, likewise —
/// 0 ignores length entirely and 1 divides it out completely.
const B: f32 = 0.75;

/// What a hit has to be worth, as a share of the best one, to come back at
/// all. A backstop under the rule below, for hits that clear it and are
/// still nowhere.
const FLOOR: f32 = 0.05;

/// A word in more than this share of the corpus is not what a question is
/// about.
///
/// Most of a question is common words: "how do I make a vector" is four of
/// them and one that matters. Every chunk holding "the" scores *something* —
/// on a large corpus that something is 0.005 and harmless, but on a small one
/// it was 21% of the best hit, which is a passage in the model's context
/// earning its place by containing the word "the".
///
/// So: if the question contains anything specific, a passage has to have
/// matched something specific. If it contains nothing specific — "what is
/// it" — the rule is not applied at all, because then the common words are
/// all there is to go on and a corpus that is entirely about vectors should
/// still answer "vector".
const COMMON: f32 = 0.5;

impl Index {
    pub fn build(chunks: Vec<Chunk>) -> Index {
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut lengths = Vec::with_capacity(chunks.len());

        for chunk in &chunks {
            let mut counts: HashMap<String, u32> = HashMap::new();
            for part in chunk.searchable() {
                for word in words(part) {
                    *counts.entry(word).or_insert(0) += 1;
                }
            }
            lengths.push(counts.values().sum());
            for (word, n) in counts {
                postings.entry(word).or_default().push((chunk.id as u32, n));
            }
        }

        let total: u64 = lengths.iter().map(|n| *n as u64).sum();
        let average = match lengths.is_empty() {
            true => 0.0,
            false => total as f32 / lengths.len() as f32,
        };
        Index { chunks, postings, lengths, average }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// How many distinct words the corpus uses.
    pub fn vocabulary(&self) -> usize {
        self.postings.len()
    }

    /// Which page a chunk is on, for deciding what may be merged with it.
    ///
    /// The source when a crawl recorded one, and the outermost heading
    /// otherwise — an uploaded corpus has no pages but still has a shape.
    fn page_of(&self, id: usize) -> (Option<&str>, &str) {
        let chunk = &self.chunks[id];
        let top = chunk.heading.split(" \u{203a} ").next().unwrap_or("");
        (chunk.source.as_deref(), top)
    }

    /// The best passages for a question: the best chunks, grown to their
    /// neighbours and merged where they touch.
    ///
    /// Never across a page: the chunk after the last one of a page is the
    /// first of another, and dragging that in cites one page with another
    /// page's words.
    pub fn passages(&self, question: &str, k: usize, radius: usize) -> Vec<Passage> {
        let hits = self.search(question, k);
        if hits.is_empty() {
            return Vec::new();
        }

        let mut spans: Vec<Passage> = hits
            .iter()
            .map(|hit| {
                let page = self.page_of(hit.chunk.id);
                let (mut from, mut to) = (hit.chunk.id, hit.chunk.id);
                for _ in 0..radius {
                    if from > 0 && self.page_of(from - 1) == page {
                        from -= 1;
                    }
                    if to + 1 < self.chunks.len() && self.page_of(to + 1) == page {
                        to += 1;
                    }
                }
                Passage {
                    heading: hit.chunk.heading.clone(),
                    source: hit.chunk.source.clone(),
                    text: String::new(),
                    score: hit.score,
                    because: hit.because.clone(),
                    from,
                    to,
                    // The chunk itself, until the text is laid out below.
                    focus: hit.chunk.id,
                }
            })
            .collect();

        // Touching counts as overlapping: two spans that meet end to end are
        // one run of text, and a gap of nothing between them is not a gap.
        // Still never across a page — growing each span stopped at the page,
        // and merging two of them would walk straight over it.
        spans.sort_by_key(|s| s.from);
        let mut merged: Vec<Passage> = Vec::new();
        for span in spans {
            match merged.last_mut() {
                Some(last)
                    if span.from <= last.to + 1
                        && self.page_of(span.from) == self.page_of(last.to) =>
                {
                    last.to = last.to.max(span.to);
                    // The heading and the reason come from the best chunk in
                    // the run, not from whichever one started earliest.
                    if span.score > last.score {
                        last.score = span.score;
                        last.heading = span.heading;
                        last.because = span.because;
                        last.focus = span.focus;
                    }
                }
                _ => merged.push(span),
            }
        }

        for passage in &mut merged {
            let mut text = String::new();
            let mut chunk_at = 0;
            for chunk in &self.chunks[passage.from..=passage.to] {
                if chunk.id == passage.focus {
                    chunk_at = text.chars().count();
                }
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(&chunk.text);
            }
            passage.focus = earned(&text, &passage.because).unwrap_or(chunk_at);
            passage.text = text;
        }
        merged.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.from.cmp(&b.from)));
        merged
    }

    /// The `k` best chunks for a question, best first.
    ///
    /// BM25. For each word of the question: how rare it is across the corpus
    /// (`idf`), times how often this chunk uses it, damped so that the tenth
    /// use is worth less than the second and a long chunk does not win on
    /// length alone.
    pub fn search(&self, question: &str, k: usize) -> Vec<Hit> {
        let n = self.chunks.len() as f32;
        let mut scores: HashMap<u32, Vec<(String, f32)>> = HashMap::new();

        let mut asked: HashMap<String, u32> = HashMap::new();
        for word in words(question) {
            *asked.entry(word).or_insert(0) += 1;
        }

        // The words of the question that are about something. See `COMMON`.
        let telling: Vec<&String> = asked
            .keys()
            .filter(|w| {
                self.postings.get(*w).is_some_and(|p| (p.len() as f32) <= n * COMMON)
            })
            .collect();
        let telling: Vec<String> = telling.into_iter().cloned().collect();

        for (word, _) in &asked {
            let Some(postings) = self.postings.get(word) else { continue };
            // Rarer is worth more, and a word in every chunk is worth nothing
            // — which is why there is no stop-word list.
            let holders = postings.len() as f32;
            let idf = ((n - holders + 0.5) / (holders + 0.5) + 1.0).ln();
            for (chunk, count) in postings {
                let length = self.lengths[*chunk as usize] as f32;
                let f = *count as f32;
                let damped = (f * (K1 + 1.0)) / (f + K1 * (1.0 - B + B * length / self.average));
                scores.entry(*chunk).or_default().push((word.clone(), idf * damped));
            }
        }

        let mut hits: Vec<Hit> = scores
            .into_iter()
            .map(|(chunk, mut because)| {
                because.sort_by(|a, b| b.1.total_cmp(&a.1));
                Hit {
                    score: because.iter().map(|(_, s)| s).sum(),
                    chunk: self.chunks[chunk as usize].clone(),
                    because,
                }
            })
            .collect();
        // By score, then by id, so that two chunks scoring the same come back
        // in the same order twice.
        // Asked something specific, answer with something specific.
        if !telling.is_empty() {
            hits.retain(|hit| hit.because.iter().any(|(w, _)| telling.contains(w)));
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk.id.cmp(&b.chunk.id)));
        if let Some(best) = hits.first().map(|h| h.score) {
            hits.retain(|h| h.score >= best * FLOOR);
        }
        hits.truncate(k);
        hits
    }
}

/// A run of adjacent chunks, handed to a model as one piece.
///
/// Ranking wants small chunks and reading wants whole ones, and they are not
/// the same size. Measured on the Rust book: asked what happens to a
/// reference after a push, the three best hits were chunks 885, 886 and 888
/// of one section, arriving as three unrelated items in score order — and the
/// model answered from the first, which was about something else. The section
/// they came from says it properly, in order, and the passage that finished
/// the explanation had not ranked at all.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Passage {
    pub heading: String,
    pub source: Option<String>,
    pub text: String,
    /// The best score of any chunk in it, and what earned that.
    pub score: f32,
    pub because: Vec<(String, f32)>,
    /// The chunks it covers, first and last inclusive.
    pub from: usize,
    pub to: usize,
    /// Where in `text` the paragraph that earned the score begins.
    ///
    /// A passage too long for the budget is cut around this rather than from
    /// the front. The whole section on reading vector elements is 6,921
    /// characters and the part about the borrow checker is near the end of
    /// it, so a budget of 4,000 taken from the beginning removes precisely
    /// the answer and leaves the model the paragraphs before it.
    ///
    /// A paragraph and not the chunk it is in: a chunk is up to [`TARGET`]
    /// characters, which at this budget is most of what there is room for,
    /// so cutting around the chunk's first line lands somewhere arbitrary
    /// inside it.
    pub focus: usize,
}

/// How many chunks either side of a hit to take with it.
///
/// One. The case this exists for is an explanation that runs over a boundary,
/// and an explanation two chunks away from anything that matched is not about
/// the question.
pub const RADIUS: usize = 1;

/// Lay passages out for a prompt, stopping at `budget` characters.
///
/// Each keeps its heading and its source, because a model given a source will
/// cite it and a model given none will invent one.
///
/// A first passage too big for the whole budget is cut at a paragraph rather
/// than dropped: half an explanation is worth more than no explanation, and
/// dropping it would send the model the question with nothing attached, which
/// is the case it answers from memory.
pub fn context(passages: &[Passage], budget: usize) -> String {
    let mut out = String::new();
    for passage in passages {
        let source = passage.source.as_deref().unwrap_or("the corpus");
        let head = format!("[{}]\n(from {})\n", passage.heading, source);
        let room = budget.saturating_sub(out.chars().count() + head.chars().count());
        if passage.text.chars().count() > room {
            if !out.is_empty() {
                break;
            }
            let trimmed = around(&passage.text, passage.focus, room);
            if trimmed.is_empty() {
                break;
            }
            out.push_str(&head);
            out.push_str(&trimmed);
            break;
        }
        out.push_str(&head);
        out.push_str(&passage.text);
        out.push_str("\n\n");
    }
    out.trim_end().to_string()
}

/// Where the paragraph that earned the score begins, in characters.
///
/// The words are the question's, with what each contributed to this
/// passage's score; a paragraph is worth the sum of the ones it holds. Using
/// the scores rather than counting matches is what makes a paragraph with the
/// rare word beat one with three common ones — which is the same judgement
/// the ranking made to pick this passage in the first place.
fn earned(text: &str, because: &[(String, f32)]) -> Option<usize> {
    let mut at = 0;
    let mut best: Option<(f32, usize)> = None;
    for para in text.split("\n\n") {
        let lower = para.to_lowercase();
        let score: f32 = because
            .iter()
            .filter(|(word, _)| words(&lower).any(|w| &w == word))
            .map(|(_, score)| score)
            .sum();
        if score > best.map_or(0.0, |(s, _)| s) {
            best = Some((score, at));
        }
        at += para.chars().count() + 2;
    }
    best.map(|(_, at)| at)
}

/// As much of `text` as fits in `room`, around the paragraph at `focus`.
///
/// Grows outwards from there a paragraph at a time, forwards first — what
/// follows the matching passage explains it more often than what precedes it —
/// so the part that earned the hit is always in what comes back.
fn around(text: &str, focus: usize, room: usize) -> String {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    // Which paragraph `focus` lands in.
    let mut at = 0;
    let mut seen = 0;
    for (i, para) in paragraphs.iter().enumerate() {
        if seen > focus {
            break;
        }
        (at, seen) = (i, seen + para.chars().count() + 2);
    }

    let (mut first, mut last) = (at, at);
    let mut taken = paragraphs[at].chars().count();
    if taken > room {
        // One paragraph is already too much. Better half of the right one
        // than all of the wrong one: cut it at a line.
        let mut out = String::new();
        for line in paragraphs[at].lines() {
            if out.chars().count() + line.chars().count() > room {
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
        return out.trim_end().to_string();
    }

    loop {
        let forward = (last + 1 < paragraphs.len())
            .then(|| paragraphs[last + 1].chars().count() + 2)
            .filter(|n| taken + n <= room);
        if let Some(n) = forward {
            (last, taken) = (last + 1, taken + n);
            continue;
        }
        let back = (first > 0)
            .then(|| paragraphs[first - 1].chars().count() + 2)
            .filter(|n| taken + n <= room);
        match back {
            Some(n) => (first, taken) = (first - 1, taken + n),
            None => break,
        }
    }
    paragraphs[first..=last].join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "\
# Ownership

Ownership is a set of rules that govern how a Rust program manages memory.

## The Stack and the Heap

The stack stores values in the order it gets them. All data on the stack must
have a known fixed size.

# Vectors

## Creating a Vector

To create a new empty vector, we call the Vec::new function.

```rust
let v: Vec<i32> = Vec::new();
```
";

    fn index() -> Index {
        let pages = vec![
            ("Ownership - The Rust Programming Language".into(), "https://x.test/ch04.html".into()),
            ("Vectors - The Rust Programming Language".into(), "https://x.test/ch08.html".into()),
        ];
        Index::build(chunk(CORPUS, &pages))
    }

    #[test]
    fn headings_become_a_path_and_a_page() {
        let chunks = chunk(CORPUS, &[]);
        let headings: Vec<&str> = chunks.iter().map(|c| c.heading.as_str()).collect();
        assert_eq!(
            headings,
            [
                "Ownership",
                "Ownership \u{203a} The Stack and the Heap",
                "Vectors \u{203a} Creating a Vector"
            ],
            "a `#` has to clear what was under the last one"
        );
        // `# Vectors` has no body of its own, so it is not a chunk — but it
        // is still the path of what follows it.
        assert!(chunks.iter().all(|c| !c.text.trim().is_empty()));
    }

    /// The manifest titles the page "Ownership - The Rust Programming
    /// Language" and the corpus heads it "Ownership". Neither is the other.
    #[test]
    fn a_chunk_cites_the_page_it_came_from() {
        let chunks = index().chunks;
        assert_eq!(chunks[0].source.as_deref(), Some("https://x.test/ch04.html"));
        assert_eq!(chunks[1].source.as_deref(), Some("https://x.test/ch04.html"));
        assert_eq!(chunks[2].source.as_deref(), Some("https://x.test/ch08.html"));
        // No manifest, no citation — rather than a wrong one.
        assert!(chunk(CORPUS, &[]).iter().all(|c| c.source.is_none()));
    }

    #[test]
    fn the_question_finds_the_section_about_it() {
        let index = index();
        let best = |q: &str| index.search(q, 3)[0].chunk.heading.clone();
        assert_eq!(best("how do I create a vector"), "Vectors \u{203a} Creating a Vector");
        assert_eq!(best("what is on the stack"), "Ownership \u{203a} The Stack and the Heap");
        assert_eq!(best("rules for managing memory"), "Ownership");
        // The code is searchable as words: `Vec::new` is `vec` and `new`.
        assert_eq!(best("Vec::new"), "Vectors \u{203a} Creating a Vector");
    }

    /// The reason there is no stop-word list — and the reason there is a
    /// floor. A common word is worth nearly nothing, which fixes the order;
    /// nearly nothing is still more than zero, which without a floor drags
    /// every chunk containing "the" into the answer behind the real one.
    #[test]
    fn a_common_word_changes_neither_the_ranking_nor_the_answer() {
        let index = index();
        let with = index.search("the stack", 3);
        let without = index.search("stack", 3);
        assert_eq!(
            with.iter().map(|h| h.chunk.id).collect::<Vec<_>>(),
            without.iter().map(|h| h.chunk.id).collect::<Vec<_>>(),
            "`the` brought something back with it"
        );
        // Worth less than the word that matters, and the hit says so rather
        // than being told to. Relative and not absolute: in a corpus of three
        // chunks `the` is in two of them and scores 0.8, where in the 1,407
        // of the Rust book it is in nearly all and scores 0.005. The claim
        // that holds at both sizes is the comparison.
        let contributed = |w: &str| {
            with[0].because.iter().find(|(word, _)| word == w).map(|(_, s)| *s).unwrap_or(0.0)
        };
        assert!(
            contributed("the") < contributed("stack") / 1.5,
            "`the` {} against `stack` {}",
            contributed("the"),
            contributed("stack")
        );
    }

    /// The other half of that rule: a question with nothing specific in it
    /// still gets an answer, because then the common words are all there is.
    /// A corpus entirely about vectors must still answer "vector".
    #[test]
    fn a_question_of_nothing_but_common_words_still_answers() {
        let corpus = "# One\n\nthe stack is here\n\n# Two\n\nthe stack is there\n";
        let index = Index::build(chunk(corpus, &[]));
        // `stack` and `the` are both in every chunk, so nothing is specific.
        assert_eq!(index.search("the stack", 5).len(), 2, "asked with what there was, got nothing");
    }

    #[test]
    fn a_hit_says_which_words_earned_it() {
        let hits = index().search("create a vector", 1);
        let words: Vec<&str> = hits[0].because.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(words[0], "vector", "the rarest word should be worth the most");
        assert!(words.contains(&"create"));
        assert!(hits[0].score > 0.0);
    }

    /// The pronoun in "how do I make a vector" is also the `I` in `I/O`, and
    /// a lone letter is rare enough that BM25 pays handsomely for it.
    #[test]
    fn a_single_letter_is_not_why_a_passage_is_relevant() {
        assert_eq!(words("how do I make a Vec<T>").collect::<Vec<_>>(), ["how", "do", "make", "vec"]);
        // The name survives being cut up; only the single letters go.
        assert_eq!(words("trpl::join").collect::<Vec<_>>(), ["trpl", "join"]);
        assert_eq!(words("I/O").collect::<Vec<_>>(), Vec::<String>::new());
    }

    #[test]
    fn nothing_matches_and_nothing_comes_back() {
        assert!(index().search("kubernetes helm chart", 5).is_empty());
        assert!(index().search("", 5).is_empty());
        assert!(Index::build(Vec::new()).search("anything", 5).is_empty());
    }

    /// Half a code block is worse than none: the model finishes the fence
    /// itself and invents whatever it thinks was in it.
    #[test]
    fn a_long_section_splits_between_paragraphs_and_never_inside_a_fence() {
        let mut text = String::from("# Long\n\n");
        for i in 0..20 {
            text.push_str(&format!("Paragraph number {i} with enough words in it to take up room.\n\n"));
        }
        text.push_str("```rust\n");
        for i in 0..40 {
            text.push_str(&format!("let x{i} = {i};\n"));
        }
        text.push_str("```\n");

        let chunks = chunk(&text, &[]);
        assert!(chunks.len() > 1, "a long section should split");
        for c in &chunks {
            assert_eq!(
                c.text.matches("```").count() % 2,
                0,
                "a fence was cut in half: {}",
                c.text
            );
        }
        // Every paragraph survived somewhere.
        let all = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all.contains("Paragraph number 19"));
        assert!(all.contains("let x39 = 39;"));
    }

    /// Ranking wants small chunks and reading wants whole ones. A hit is
    /// grown to its neighbours so that an explanation running over a boundary
    /// arrives in one piece — but never past the page it is on, because the
    /// chunk after the last one of a page is the first of another.
    #[test]
    fn a_passage_grows_to_its_neighbours_and_stops_at_the_page() {
        let index = index();
        // `stack` is in chunk 1 alone. Chunk 0 is the same page and comes
        // with it; chunk 2 is a different page and does not.
        let passages = index.passages("stack", 5, RADIUS);
        assert_eq!(passages.len(), 1);
        assert_eq!((passages[0].from, passages[0].to), (0, 1));
        assert!(passages[0].text.contains("manages memory"), "the neighbour was left behind");
        assert!(passages[0].text.contains("stack stores values"));
        assert!(!passages[0].text.contains("empty vector"), "it walked into the next page");
        // Named after the chunk that earned it, not the one it starts at.
        assert_eq!(passages[0].heading, "Ownership \u{203a} The Stack and the Heap");
    }

    /// Two hits on the same page become one run of text; a hit on another
    /// page stays its own, even when the two are next to each other.
    #[test]
    fn touching_passages_merge_and_pages_still_do_not() {
        let index = index();
        let passages = index.passages("vector stack memory", 5, RADIUS);
        assert_eq!(passages.len(), 2, "{:?}", passages.iter().map(|p| (p.from, p.to)).collect::<Vec<_>>());
        let spans: Vec<(usize, usize)> = passages.iter().map(|p| (p.from, p.to)).collect();
        assert!(spans.contains(&(0, 1)), "the two chunks of one page did not merge: {spans:?}");
        assert!(spans.contains(&(2, 2)), "the other page was swallowed: {spans:?}");
        // No chunk is in two passages, so nothing is sent twice.
        assert!(passages[0].to < passages[1].from || passages[1].to < passages[0].from);
    }

    /// A budget smaller than the passage must not remove the answer.
    ///
    /// The whole section on reading vector elements is 6,921 characters and
    /// the part about the borrow checker is near the end of it, so a budget
    /// of 4,000 taken from the front keeps the paragraphs before the answer
    /// and drops the answer.
    #[test]
    fn a_passage_too_long_for_the_budget_is_cut_around_the_answer() {
        let mut corpus = String::from("# Long Page\n\n");
        for i in 0..30 {
            corpus.push_str(&format!("Filler paragraph {i} about nothing in particular at all.\n\n"));
        }
        corpus.push_str("The borrowchecker forbids this entirely.\n\n");
        for i in 30..60 {
            corpus.push_str(&format!("Filler paragraph {i} about nothing in particular at all.\n\n"));
        }

        let index = Index::build(chunk(&corpus, &[]));
        let passages = index.passages("borrowchecker", 5, RADIUS);
        assert_eq!(passages.len(), 1);
        assert!(passages[0].text.chars().count() > 600, "the passage should not fit in the budget");

        let prompt = context(&passages, 600);
        assert!(prompt.chars().count() <= 600, "the budget was {} over", prompt.chars().count());
        assert!(
            prompt.contains("The borrowchecker forbids this entirely."),
            "the budget removed the answer:\n{prompt}"
        );
        // And it is not simply the front of the passage.
        assert!(!prompt.contains("Filler paragraph 0 "), "it was cut from the front after all");
    }

    #[test]
    fn a_prompt_gets_the_source_and_stops_at_its_budget() {
        let hits = index().passages("vector stack memory", 3, RADIUS);
        assert_eq!(hits.len(), 2);

        let prompt = context(&hits, 10_000);
        assert!(prompt.contains("https://x.test/ch08.html"));
        assert!(prompt.contains("Creating a Vector"));

        // A budget for one passage takes one, and never nothing: a prompt
        // with no context at all is a question the model answers from memory.
        let tight = context(&hits, 200);
        assert!(!tight.is_empty());
        assert!(tight.chars().count() < prompt.chars().count());
        assert!(tight.chars().count() <= 200, "the budget was {} over", tight.chars().count());
    }
}
