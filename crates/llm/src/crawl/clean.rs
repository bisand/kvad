//! Making crawled text trainable.
//!
//! A character tokeniser gives an id to every distinct character it saw, and
//! a model has one row of its embedding table and one column of its output
//! head per id. That makes the size of a corpus's alphabet a hyperparameter,
//! and the web is very bad at keeping it small: the same page uses three
//! kinds of quotation mark, two kinds of dash, a non-breaking space between
//! every number and its unit, and one emoji in a warning box.
//!
//! Left alone, a crawl of a documentation site arrives with three or four
//! hundred distinct characters, of which the last two hundred were seen once
//! each. Those rows never train — they get a handful of gradient updates
//! between them — and they are the rows that make an existing model refuse
//! the text entirely: what `--from` asks of a corpus before it will continue
//! a model is exactly "does this contain a character your tokeniser has never
//! met".
//!
//! So: [`normalise`] maps what has an ASCII spelling onto it, and
//! [`drop_rare`] removes the tail that is left. Both are reported — the
//! manifest says what was mapped and what was dropped — because silently
//! editing somebody's corpus is worse than a big alphabet.

/// Characters that are never dropped however rare they are.
///
/// A corpus where `z` or `7` appears twice is a small corpus, not a corpus
/// with a junk character in it, and a model that cannot spell `zero` because
/// of a frequency threshold is a bug that would take a day to find.
fn always(c: char) -> bool {
    c == '\n' || (' '..='~').contains(&c)
}

/// What one character was turned into, for the report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Mapped {
    pub from: char,
    pub to: String,
    pub count: usize,
}

/// A character and how often it occurred.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Count {
    pub character: String,
    pub count: usize,
}

/// The ASCII spelling of a character, where it has one.
///
/// Deliberately not a Unicode normalisation: NFKD would also take the accents
/// off `café`, and a corpus of Norwegian would lose the three letters that
/// make it Norwegian. This is about typography — quotation marks, dashes,
/// spaces and bullets — and leaves letters alone.
fn ascii(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{a0}' | '\u{2007}' | '\u{202f}' | '\u{2009}' | '\u{200a}' | '\u{2002}' | '\u{2003}'
        | '\u{3000}' => " ",
        '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}' | '\u{ad}' => "",
        '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{2032}' | '\u{2035}' | '\u{ff07}' => "'",
        '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{2033}' | '\u{ff02}' => "\"",
        '\u{2013}' | '\u{2212}' | '\u{2012}' | '\u{2010}' | '\u{2011}' => "-",
        '\u{2014}' | '\u{2015}' => "--",
        '\u{2026}' => "...",
        '\u{2022}' | '\u{25cf}' | '\u{25aa}' | '\u{00b7}' | '\u{2043}' => "-",
        '\u{00d7}' => "x",
        '\u{2192}' => "->",
        '\u{2190}' => "<-",
        '\u{21d2}' => "=>",
        '\u{2264}' => "<=",
        '\u{2265}' => ">=",
        '\u{2260}' => "!=",
        '\u{00a9}' => "(c)",
        '\u{00ae}' => "(r)",
        '\u{2122}' => "(tm)",
        '\t' => "    ",
        '\r' => "",
        _ => return None,
    })
}

/// Map typography onto ASCII and tidy the whitespace.
///
/// Returns the text and what it replaced, commonest first.
pub fn normalise(text: &str) -> (String, Vec<Mapped>) {
    let mut out = String::with_capacity(text.len());
    let mut seen: Vec<(char, &'static str, usize)> = Vec::new();
    for c in text.chars() {
        match ascii(c) {
            Some(to) => {
                out.push_str(to);
                match seen.iter_mut().find(|(from, _, _)| *from == c) {
                    Some((_, _, n)) => *n += 1,
                    None => seen.push((c, to, 1)),
                }
            }
            None => out.push(c),
        }
    }
    seen.sort_by(|a, b| b.2.cmp(&a.2));
    let mapped =
        seen.into_iter().map(|(from, to, count)| Mapped { from, to: to.into(), count }).collect();
    (tidy(&out), mapped)
}

/// Trailing spaces off every line, and no more than one blank line in a row.
///
/// Whitespace is characters, and a corpus whose commonest bigram is "space
/// newline" teaches a model to write trailing spaces.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    // Leading and trailing blank lines are nobody's content.
    out.trim_start_matches('\n').trim_end().to_string() + "\n"
}

/// Every distinct character and how often it occurs, commonest first.
pub fn histogram(text: &str) -> Vec<Count> {
    let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for c in text.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    // By count, then by character, so the same text reports the same order
    // twice — a `HashMap`'s iteration order would not.
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts.into_iter().map(|(c, count)| Count { character: c.to_string(), count }).collect()
}

/// Remove the characters seen fewer than `min` times.
///
/// `min` of 0 or 1 keeps everything. Returns the text and what went, so that
/// the job's result can say "these 143 characters were removed, 2,400 of them
/// were one emoji" and somebody can disagree and run it again.
pub fn drop_rare(text: &str, min: usize) -> (String, Vec<Count>) {
    if min < 2 {
        return (text.to_string(), Vec::new());
    }
    let counts = histogram(text);
    let doomed: Vec<char> = counts
        .iter()
        .filter(|c| c.count < min)
        .filter_map(|c| c.character.chars().next())
        .filter(|c| !always(*c))
        .collect();
    if doomed.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let out: String = text.chars().filter(|c| !doomed.contains(c)).collect();
    let dropped = counts.into_iter().filter(|c| c.character.chars().all(|ch| doomed.contains(&ch))).collect();
    (tidy(&out), dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typography_becomes_ascii_and_letters_are_left_alone() {
        let (text, mapped) = normalise("\u{201c}Don\u{2019}t\u{201d} \u{2014} caf\u{e9}, blåbær\u{2026}");
        assert_eq!(text, "\"Don't\" -- café, blåbær...\n");
        // Reported, commonest first, so a corpus can be argued with.
        assert_eq!(mapped.len(), 5);
        assert!(mapped.iter().any(|m| m.from == '\u{2019}' && m.to == "'"));
    }

    #[test]
    fn whitespace_is_characters_too() {
        let (text, _) = normalise("a  \n\n\n\nb\t\r\n   \n");
        assert_eq!(text, "a\n\nb\n");
    }

    #[test]
    fn the_rare_tail_goes_and_the_alphabet_stays() {
        let text = "aaaa bbbb 🎉 çç\n";
        let (out, dropped) = drop_rare(text, 3);
        assert_eq!(out, "aaaa bbbb\n");
        assert_eq!(dropped.iter().map(|d| d.character.as_str()).collect::<Vec<_>>(), ["ç", "🎉"]);

        // ASCII is never rare, however seldom it turns up: a corpus with one
        // `z` in it is a small corpus.
        let (kept, dropped) = drop_rare("aaaaaaaa z\n", 3);
        assert_eq!(kept, "aaaaaaaa z\n");
        assert!(dropped.is_empty());

        // Below 2 the threshold means nothing, and says so by doing nothing.
        assert_eq!(drop_rare(text, 1).0, text);
    }

    #[test]
    fn a_histogram_is_ordered_the_same_way_twice() {
        let h = histogram("banana");
        assert_eq!(h[0].character, "a");
        assert_eq!(h[0].count, 3);
        assert_eq!(histogram("banana")[2].character, histogram("banana")[2].character);
    }
}
