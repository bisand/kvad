//! Porter's stemmer, so that a question may be phrased in its asker's words.
//!
//! The corpus says "push" fifty-five times and "pushing" five. A question
//! asking what happens "after pushing to the vector" shares no word with the
//! passage that answers it, so the one term that tells that section's two
//! halves apart counts for nothing, and the ranking is decided by `element`
//! and `vector`, which are spread evenly over both. Every measurement of that
//! question in this repository's history was really a measurement of this.
//!
//! # Why the whole algorithm and not three rules
//!
//! Stripping `-ing`, `-ed` and `-s` by hand is thirty lines and turns
//! `string` into `str`, which on a corpus of code does more damage than it
//! repairs. Porter's rules are conditioned on the *measure* of what would be
//! left — roughly, how many syllables — and `str` has a measure of zero and
//! not a vowel in it, so `string` is left alone while `pushing` reduces. That
//! condition is the reason to write the real thing rather than an
//! approximation of it.
//!
//! # What it is not
//!
//! Not a dictionary and not linguistics. `closure` becomes `closur` and
//! `creates` becomes `creat`, which are not words. It does not matter: both
//! sides of the index go through the same function, so what is required of it
//! is that it agree with itself. It is applied to the corpus and to the
//! question, never to anything anybody reads.

/// The stem of an English word, by Porter's algorithm.
///
/// Words of two letters or fewer come back untouched: step 1a would take the
/// `s` off `as` and leave a single letter, which is not a search term.
/// Anything that is not ASCII comes back untouched too — the rules are about
/// English letters, and a word that is not made of them has no suffix these
/// rules know.
pub fn stem(word: &str) -> String {
    if word.len() <= 2 || !word.is_ascii() {
        return word.to_string();
    }
    let mut w = word.as_bytes().to_vec();
    step1a(&mut w);
    step1b(&mut w);
    step1c(&mut w);
    step2(&mut w);
    step3(&mut w);
    step4(&mut w);
    step5(&mut w);
    String::from_utf8(w).unwrap_or_else(|_| word.to_string())
}

/// Whether the letter at `i` is a consonant.
///
/// `y` is the awkward one, and it is awkward in both directions: a consonant
/// at the start of a word or after a vowel (`yes`, `buoy`), a vowel after a
/// consonant (`sky`).
fn consonant(w: &[u8], i: usize) -> bool {
    match w[i] {
        b'a' | b'e' | b'i' | b'o' | b'u' => false,
        b'y' => i == 0 || !consonant(w, i - 1),
        _ => true,
    }
}

/// How many vowel-consonant sequences a stem has: `[C](VC)^m[V]`.
///
/// Porter's stand-in for syllables, and what every condition below is really
/// asking about. `tree` is 0, `trouble` is 1, `troubles` is 2.
fn measure(w: &[u8]) -> usize {
    let n = w.len();
    let mut i = 0;
    while i < n && consonant(w, i) {
        i += 1;
    }
    let mut m = 0;
    while i < n {
        while i < n && !consonant(w, i) {
            i += 1;
        }
        if i >= n {
            break;
        }
        while i < n && consonant(w, i) {
            i += 1;
        }
        m += 1;
    }
    m
}

fn has_vowel(w: &[u8]) -> bool {
    (0..w.len()).any(|i| !consonant(w, i))
}

/// Ends in two of the same consonant: `hopp`, `fall`.
fn doubled(w: &[u8]) -> bool {
    let n = w.len();
    n >= 2 && w[n - 1] == w[n - 2] && consonant(w, n - 1)
}

/// Ends consonant-vowel-consonant, the last not `w`, `x` or `y`: `hop`, but
/// not `snow` or `box`. The shape that takes an `e` when a suffix is removed.
fn cvc(w: &[u8]) -> bool {
    let n = w.len();
    n >= 3
        && consonant(w, n - 3)
        && !consonant(w, n - 2)
        && consonant(w, n - 1)
        && !matches!(w[n - 1], b'w' | b'x' | b'y')
}

fn ends(w: &[u8], suffix: &str) -> bool {
    w.ends_with(suffix.as_bytes())
}

/// The measure of what would be left if `len` letters came off the end.
fn stem_measure(w: &[u8], len: usize) -> usize {
    measure(&w[..w.len() - len])
}

fn swap(w: &mut Vec<u8>, len: usize, with: &str) {
    w.truncate(w.len() - len);
    w.extend_from_slice(with.as_bytes());
}

/// Plurals.
fn step1a(w: &mut Vec<u8>) {
    if ends(w, "sses") {
        swap(w, 4, "ss");
    } else if ends(w, "ies") {
        swap(w, 3, "i");
    } else if ends(w, "ss") {
        // Already a stem: `caress` is not `cares`.
    } else if ends(w, "s") {
        swap(w, 1, "");
    }
}

/// `-ed` and `-ing`, and the spelling repairs that removing them needs.
fn step1b(w: &mut Vec<u8>) {
    let mut removed = false;
    if ends(w, "eed") {
        // `agreed` keeps its `ee`; `feed` has nothing to spare.
        if stem_measure(w, 3) > 0 {
            swap(w, 3, "ee");
        }
        return;
    } else if ends(w, "ed") && has_vowel(&w[..w.len() - 2]) {
        swap(w, 2, "");
        removed = true;
    } else if ends(w, "ing") && has_vowel(&w[..w.len() - 3]) {
        // The condition that saves `string`: what is left is `str`, which has
        // no vowel in it, so nothing is removed.
        swap(w, 3, "");
        removed = true;
    }
    if !removed {
        return;
    }
    if ends(w, "at") || ends(w, "bl") || ends(w, "iz") {
        w.push(b'e');
    } else if doubled(w) && !matches!(w[w.len() - 1], b'l' | b's' | b'z') {
        // `hopping` became `hopp`; it is `hop`.
        w.pop();
    } else if measure(w) == 1 && cvc(w) {
        // `filing` became `fil`; it is `file`.
        w.push(b'e');
    }
}

/// `y` to `i`, so that `happy` and `happiness` meet.
fn step1c(w: &mut Vec<u8>) {
    if ends(w, "y") && has_vowel(&w[..w.len() - 1]) {
        swap(w, 1, "i");
    }
}

/// Longest first: a word ending `ational` also ends `tional`.
const STEP2: &[(&str, &str)] = &[
    ("ational", "ate"),
    ("tional", "tion"),
    ("ization", "ize"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("biliti", "ble"),
    ("ousli", "ous"),
    ("entli", "ent"),
    ("ation", "ate"),
    ("alism", "al"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("anci", "ance"),
    ("enci", "ence"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("ator", "ate"),
    ("logi", "log"),
    ("bli", "ble"),
    ("eli", "e"),
];

fn step2(w: &mut Vec<u8>) {
    for (from, to) in STEP2 {
        if ends(w, from) && stem_measure(w, from.len()) > 0 {
            swap(w, from.len(), to);
            return;
        }
    }
}

const STEP3: &[(&str, &str)] = &[
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ness", ""),
    ("ful", ""),
];

fn step3(w: &mut Vec<u8>) {
    for (from, to) in STEP3 {
        if ends(w, from) && stem_measure(w, from.len()) > 0 {
            swap(w, from.len(), to);
            return;
        }
    }
}

/// Suffixes that go only from a word with something left over: measure above
/// one. This is what keeps `iter` from becoming `it`.
const STEP4: &[&str] = &[
    "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous",
    "ive", "ize", "ion", "al", "er", "ic", "ou",
];

fn step4(w: &mut Vec<u8>) {
    for suffix in STEP4 {
        if !ends(w, suffix) || stem_measure(w, suffix.len()) <= 1 {
            continue;
        }
        // `ion` only after `s` or `t`: `adoption` yes, `lion` no.
        if *suffix == "ion" {
            let stem = &w[..w.len() - 3];
            if !matches!(stem.last(), Some(b's') | Some(b't')) {
                continue;
            }
        }
        swap(w, suffix.len(), "");
        return;
    }
}

/// A trailing `e`, and a doubled `l`.
fn step5(w: &mut Vec<u8>) {
    if ends(w, "e") {
        let m = stem_measure(w, 1);
        // `rate` keeps its `e` — `rat` is a different word — and `cease`
        // does not need one.
        if m > 1 || (m == 1 && !cvc(&w[..w.len() - 1])) {
            w.pop();
        }
    }
    if measure(w) > 1 && doubled(w) && ends(w, "l") {
        w.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measurement that started this: the corpus says `push` and the
    /// question says `pushing`, and until now those were different words.
    #[test]
    fn the_question_and_the_corpus_meet_in_the_middle() {
        for form in ["push", "pushes", "pushed", "pushing"] {
            assert_eq!(stem(form), "push", "{form}");
        }
        for form in ["borrow", "borrows", "borrowed", "borrowing"] {
            assert_eq!(stem(form), "borrow", "{form}");
        }
        for form in ["reference", "references", "referring", "referred"] {
            assert_eq!(stem(form), "refer", "{form}");
        }
        for form in ["allocate", "allocates", "allocated", "allocating", "allocation"] {
            assert_eq!(stem(form), "alloc", "{form}");
        }
    }

    /// The reason this is Porter's algorithm and not three rules. Every one of
    /// these is a word a naive `-ing`/`-ed`/`-s` stripper ruins, and ruining
    /// them on a corpus of code is worse than the problem being fixed.
    #[test]
    fn code_words_are_left_alone() {
        for word in [
            "string", "vec", "impl", "trait", "unwrap", "iter", "mut", "dyn", "async", "await",
            "slice", "clone", "panic", "enum", "struct", "crate", "macro", "thread", "stack",
            "heap", "box", "i32", "utf8",
        ] {
            assert_eq!(stem(word), word, "`{word}` was stemmed");
        }
        // And the plural still meets the singular.
        assert_eq!(stem("strings"), "string");
        assert_eq!(stem("traits"), "trait");
        assert_eq!(stem("slices"), "slice");
        assert_eq!(stem("threads"), "thread");
    }

    /// Two letters or fewer, and anything that is not ASCII, are left as they
    /// are: step 1a would make `as` into a single letter.
    #[test]
    fn the_short_and_the_foreign_are_untouched() {
        for word in ["as", "is", "us", "of", "in", "if", "to", "fn"] {
            assert_eq!(stem(word), word, "`{word}` was stemmed");
        }
        assert_eq!(stem("blåbær"), "blåbær");
        assert_eq!(stem(""), "");
    }

    /// Porter's own examples for the parts of the algorithm that are easy to
    /// get subtly wrong: the doubled consonant, the restored `e`, and the
    /// measure conditions that decide whether a suffix may go at all.
    #[test]
    fn the_awkward_rules_behave() {
        assert_eq!(stem("hopping"), "hop", "a doubled consonant should collapse");
        assert_eq!(stem("falling"), "fall", "but not one of l, s or z");
        assert_eq!(stem("filing"), "file", "a short stem takes its `e` back");
        // `eed` keeps its `ee` when there is a stem in front of it, and step
        // 5 then takes the last `e` off what is left: the intermediate is
        // `agree` and the answer is `agre`. What matters is that every form
        // arrives at the same place.
        assert_eq!(stem("agreed"), "agre");
        assert_eq!(stem("agree"), "agre");
        assert_eq!(stem("agreeing"), "agre");
        // And a word with nothing in front of the `eed` keeps all of it.
        assert_eq!(stem("feed"), "feed");
        assert_eq!(stem("caresses"), "caress");
        assert_eq!(stem("ponies"), "poni");
        assert_eq!(stem("caress"), "caress", "`ss` is not a plural");
        assert_eq!(stem("relational"), "relat");
        assert_eq!(stem("adoption"), "adopt", "`ion` goes after a t");
        assert_eq!(stem("lion"), "lion", "and stays otherwise");
        assert_eq!(stem("rate"), "rate", "a short stem keeps its `e`");
        assert_eq!(stem("controlling"), "control", "a doubled l, once");
    }

    /// `y` is a vowel after a consonant and a consonant otherwise, and the
    /// measure depends on getting that right.
    #[test]
    fn y_is_both_kinds_of_letter() {
        assert_eq!(measure(b"tree"), 0);
        assert_eq!(measure(b"trouble"), 1);
        assert_eq!(measure(b"troubles"), 2);
        assert!(!consonant(b"sky", 2), "the y of `sky` is a vowel");
        assert!(consonant(b"yes", 0), "the y of `yes` is a consonant");
        assert_eq!(stem("happy"), "happi");
        assert_eq!(stem("happiness"), "happi", "and so they meet");
    }
}
