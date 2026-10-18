//! How wrong a "who spoke when" is: the yardstick every later step of the
//! diarization plan (`docs/diarization-plan.md`) is measured with.
//!
//! Four numbers, each answering its own question:
//!
//! - [`der`], the diarization error rate: of all the time someone speaks in
//!   the reference, the share missed, falsely found or given to the wrong
//!   speaker. It measures the speakers alone, without any words.
//! - [`jer`], the Jaccard error rate, DIHARD's second number: the same
//!   errors, but averaged over speakers, so that a speaker who says little
//!   weighs as much as one who says a lot.
//! - [`wder`], the word diarization error: of the words the transcript and
//!   the reference agree are there, the share given to the wrong speaker.
//!   What a reader of a transcript with speakers sees.
//! - [`cpwer`], the word error rate of each speaker's text against the
//!   reference speaker's it is matched to: everything at once, Whisper
//!   included.
//!
//! A hypothesis calls its speakers whatever it likes, so each number first
//! maps its speakers one to one onto the reference's, the mapping that makes
//! it least. That is an assignment problem, solved by [`hungarian`].
//!
//! # Held to pyannote.metrics
//!
//! [`der`] and [`jer`] are `pyannote.metrics`' arithmetic, read off its
//! source (`diarization.py`, `identification.py`, `utils.py`), and
//! `scripts/der-fixtures.py` writes its numbers for the fixtures the tests
//! below hold these to, to 0.01 points. Its choices, which every one of them
//! changes a number:
//!
//! - What is scored runs from the first turn of either side to the last, as
//!   pyannote does when no scoring map is given.
//! - The speakers are mapped by the time each pair shares, scored before the
//!   mapping is made, and a pair that shares none is not mapped at all.
//! - The time is exact. The plan said a frame count at 10 ms, but frames
//!   would move every boundary by up to 5 ms, and on a short recording that
//!   is tenths of a point: further from pyannote than the bound.
//! - [`jer`] maps speakers as [`der`] does, and leaves a hypothesis speaker
//!   with no reference speaker out, where DIHARD's `dscore` maps by Jaccard
//!   and counts them.
//!
//! # The collar
//!
//! [`Protocol::collar`] is seconds either side of a reference boundary, as
//! `md-eval` and `dscore` take `--collar 0.25`. `pyannote.metrics` takes
//! `collar=0.5` to mean the same, half on each side: the fixture script
//! doubles it on the way.

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Someone speaking, from `start` to `end` seconds: one line of an RTTM file.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
}

/// A word, and who said it.
#[derive(Debug, Clone, PartialEq)]
pub struct Spoken {
    pub speaker: String,
    pub word: String,
}

/// What is scored, of the time there is: the three the VBx paper names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Protocol {
    /// Seconds either side of each reference boundary left unscored.
    pub collar: f64,
    /// Whether time where the reference has two speakers at once is left
    /// unscored.
    pub skip_overlap: bool,
}

impl Protocol {
    /// No collar, overlap scored: what pyannote's and NVIDIA's model cards
    /// report.
    pub const FULL: Protocol = Protocol { collar: 0.0, skip_overlap: false };
    /// A quarter second either side of every reference boundary left out,
    /// overlap scored: for references written at the sentence.
    pub const FAIR: Protocol = Protocol { collar: 0.25, skip_overlap: false };
    /// The same collar, and overlap left out: most papers before 2021.
    pub const FORGIVING: Protocol = Protocol { collar: 0.25, skip_overlap: true };
}

/// The four kinds of time a [`der`] counts, in seconds. Overlap counts per
/// speaker: two people at once for a second is two seconds of `total`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Der {
    /// Reference speech, speaker by speaker.
    pub total: f64,
    /// Speech the hypothesis gave nobody.
    pub missed: f64,
    /// Speakers the hypothesis heard where the reference has fewer.
    pub false_alarm: f64,
    /// Speech given to the wrong speaker.
    pub confusion: f64,
}

impl Der {
    /// The error, as a fraction: 0.12 is 12 points.
    pub fn rate(&self) -> f64 {
        let wrong = self.missed + self.false_alarm + self.confusion;
        match self.total {
            0.0 if wrong == 0.0 => 0.0,
            0.0 => 1.0,
            t => wrong / t,
        }
    }
}

/// Turns from RTTM text: `SPEAKER file 1 start duration <NA> <NA> name <NA> <NA>`,
/// one a line. Other line types, and turns that last no time, are passed
/// over, as pyannote's reader drops a segment that is empty.
pub fn rttm(text: &str) -> Res<Vec<Turn>> {
    let mut turns = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"SPEAKER") {
            continue;
        }
        let bad = || format!("RTTM line {}: {line}", n + 1);
        let start: f64 = f.get(3).and_then(|s| s.parse().ok()).ok_or_else(bad)?;
        let duration: f64 = f.get(4).and_then(|s| s.parse().ok()).ok_or_else(bad)?;
        let speaker = f.get(7).ok_or_else(bad)?.to_string();
        if duration > 0.0 {
            turns.push(Turn { speaker, start, end: start + duration });
        }
    }
    Ok(turns)
}

/// `turns` as RTTM, for a recording called `file`, in milliseconds.
pub fn to_rttm(file: &str, turns: &[Turn]) -> String {
    turns.iter().map(|t| format!("SPEAKER {file} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n", t.start, t.end - t.start, t.speaker)).collect()
}

/// The cheapest one-to-one assignment of rows to columns: for each row, its
/// column, or `None` where there are more rows than columns and it got none.
///
/// Kuhn and Munkres's algorithm with potentials, in the form of Jonker and
/// Volgenant: a row at a time, it grows the shortest augmenting path from the
/// new row through the columns, lowering each row's potential and raising
/// each column's so that every edge on a path costs nothing in the reduced
/// costs, and flips the path. O(n² m) for n rows of m columns, n ≤ m, which
/// a wider-than-tall matrix is turned into first. A diarization has a few
/// speakers a side; the clustering later has thousands of chunks against a
/// few speakers, which is the same shape.
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<Option<usize>> {
    let n = cost.len();
    let m = cost.first().map_or(0, Vec::len);
    if n == 0 || m == 0 {
        return vec![None; n];
    }
    if n > m {
        // Columns to rows, and back.
        let t: Vec<Vec<f64>> = (0..m).map(|j| (0..n).map(|i| cost[i][j]).collect()).collect();
        let mut rows = vec![None; n];
        for (j, i) in hungarian(&t).into_iter().enumerate() {
            rows[i.expect("every row of a tall matrix is assigned")] = Some(j);
        }
        return rows;
    }
    // One-based, with row and column 0 the path's virtual start.
    let (mut u, mut v) = (vec![0.0f64; n + 1], vec![0.0f64; m + 1]);
    // `owner[j]`: the row column `j` is assigned to, 0 for none.
    let mut owner = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        owner[0] = i;
        let mut j0 = 0;
        let mut least = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = owner[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let reduced = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if reduced < least[j] {
                    least[j] = reduced;
                    way[j] = j0;
                }
                if least[j] < delta {
                    delta = least[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[owner[j]] += delta;
                    v[j] -= delta;
                } else {
                    least[j] -= delta;
                }
            }
            j0 = j1;
            if owner[j0] == 0 {
                break;
            }
        }
        // Flip the path back to its start.
        while j0 != 0 {
            let j1 = way[j0];
            owner[j0] = owner[j1];
            j0 = j1;
        }
    }
    let mut rows = vec![None; n];
    for j in 1..=m {
        if owner[j] != 0 {
            rows[owner[j] - 1] = Some(j - 1);
        }
    }
    rows
}

/// The assignment that makes `gain` most, as [`hungarian`] gives one.
fn most(gain: &[Vec<f64>]) -> Vec<Option<usize>> {
    hungarian(&gain.iter().map(|r| r.iter().map(|g| -g).collect()).collect::<Vec<_>>())
}

/// Disjoint, sorted intervals covering what `spans` cover.
fn union(mut spans: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    spans.retain(|(a, b)| b > a);
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (a, b) in spans {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// `from` with what `take` covers taken out; both disjoint and sorted.
fn minus(from: &[(f64, f64)], take: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for &(mut a, b) in from {
        for &(c, d) in take {
            if d <= a || c >= b {
                continue;
            }
            if c > a {
                out.push((a, c));
            }
            a = a.max(d);
        }
        if b > a {
            out.push((a, b));
        }
    }
    out
}

/// How long two sets of disjoint, sorted intervals share.
fn shared(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let (mut i, mut j, mut sum) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        sum += (a[i].1.min(b[j].1) - a[i].0.max(b[j].0)).max(0.0);
        if a[i].1 < b[j].1 { i += 1 } else { j += 1 }
    }
    sum
}

fn length(a: &[(f64, f64)]) -> f64 {
    a.iter().map(|(s, e)| e - s).sum()
}

/// Both sides cut to what `protocol` scores, each turn as `(speaker index,
/// start, end)`, with each side's speakers in order of first appearance.
struct Scored {
    reference: Vec<(usize, f64, f64)>,
    hypothesis: Vec<(usize, f64, f64)>,
    n_ref: usize,
    n_hyp: usize,
}

fn scored(reference: &[Turn], hypothesis: &[Turn], protocol: Protocol) -> Scored {
    let all = reference.iter().chain(hypothesis);
    let extent = (all.clone().map(|t| t.start).fold(f64::INFINITY, f64::min), all.map(|t| t.end).fold(f64::NEG_INFINITY, f64::max));
    let mut unscored = Vec::new();
    if protocol.collar > 0.0 {
        for t in reference {
            for at in [t.start, t.end] {
                unscored.push((at - protocol.collar, at + protocol.collar));
            }
        }
    }
    if protocol.skip_overlap {
        for (i, a) in reference.iter().enumerate() {
            for b in &reference[i + 1..] {
                unscored.push((a.start.max(b.start), a.end.min(b.end)));
            }
        }
    }
    let map = minus(&union(vec![extent]), &union(unscored));
    let cut = |turns: &[Turn]| -> (Vec<(usize, f64, f64)>, usize) {
        let mut names: Vec<&str> = Vec::new();
        let mut out = Vec::new();
        for t in turns {
            for &(a, b) in &map {
                let (s, e) = (t.start.max(a), t.end.min(b));
                if e > s {
                    let k = names.iter().position(|n| *n == t.speaker).unwrap_or_else(|| {
                        names.push(&t.speaker);
                        names.len() - 1
                    });
                    out.push((k, s, e));
                }
            }
        }
        (out, names.len())
    };
    let (reference, n_ref) = cut(reference);
    let (hypothesis, n_hyp) = cut(hypothesis);
    Scored { reference, hypothesis, n_ref, n_hyp }
}

impl Scored {
    /// For each hypothesis speaker, the reference speaker it is mapped to:
    /// the mapping under which the most time is shared, pair by pair.
    fn mapping(&self) -> Vec<Option<usize>> {
        let mut together = vec![vec![0.0; self.n_ref]; self.n_hyp];
        for &(h, a, b) in &self.hypothesis {
            for &(r, c, d) in &self.reference {
                together[h][r] += (b.min(d) - a.max(c)).max(0.0);
            }
        }
        most(&together).into_iter().enumerate().map(|(h, r)| r.filter(|&r| together[h][r] > 0.0)).collect()
    }

    /// Each side's time for one speaker, as disjoint intervals.
    fn of(turns: &[(usize, f64, f64)], k: usize) -> Vec<(f64, f64)> {
        union(turns.iter().filter(|t| t.0 == k).map(|t| (t.1, t.2)).collect())
    }
}

/// The diarization error of `hypothesis` against `reference`, under
/// `protocol`.
pub fn der(reference: &[Turn], hypothesis: &[Turn], protocol: Protocol) -> Der {
    let s = scored(reference, hypothesis, protocol);
    let mapped = s.mapping();
    // Every boundary on either side, so that between two of them who is
    // speaking does not change.
    let mut at: Vec<f64> = s.reference.iter().chain(&s.hypothesis).flat_map(|t| [t.1, t.2]).collect();
    at.sort_by(f64::total_cmp);
    at.dedup();
    let mut der = Der::default();
    for w in at.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = (a + b) / 2.0;
        let mut speaking = vec![0usize; s.n_ref];
        let (mut n_ref, mut n_hyp) = (0, 0);
        for &(r, c, d) in &s.reference {
            if c <= mid && mid < d {
                speaking[r] += 1;
                n_ref += 1;
            }
        }
        let mut heard = vec![0usize; s.n_ref];
        for &(h, c, d) in &s.hypothesis {
            if c <= mid && mid < d {
                n_hyp += 1;
                // A speaker mapped to nobody is never right.
                if let Some(r) = mapped[h] {
                    heard[r] += 1;
                }
            }
        }
        let correct: usize = speaking.iter().zip(&heard).map(|(a, b)| *a.min(b)).sum();
        let span = b - a;
        der.total += span * n_ref as f64;
        der.confusion += span * (n_ref.min(n_hyp) - correct) as f64;
        der.missed += span * n_ref.saturating_sub(n_hyp) as f64;
        der.false_alarm += span * n_hyp.saturating_sub(n_ref) as f64;
    }
    der
}

/// The Jaccard error rate of `hypothesis` against `reference`, under
/// `protocol`, as a fraction: for each reference speaker, the time it and its
/// mapped hypothesis speaker do not share over the time either speaks, and
/// the mean of that over the reference's speakers. A reference speaker with
/// no hypothesis speaker is all error.
pub fn jer(reference: &[Turn], hypothesis: &[Turn], protocol: Protocol) -> f64 {
    let s = scored(reference, hypothesis, protocol);
    if s.n_ref == 0 {
        return 1.0;
    }
    let mapped = s.mapping();
    let mut sum = 0.0;
    for r in 0..s.n_ref {
        sum += match mapped.iter().position(|&m| m == Some(r)) {
            None => 1.0,
            Some(h) => {
                let (a, b) = (Scored::of(&s.reference, r), Scored::of(&s.hypothesis, h));
                let both = shared(&a, &b);
                let either = length(&a) + length(&b) - both;
                (either - both) / either
            }
        };
    }
    sum / s.n_ref as f64
}

/// How `hypothesis` lines up with `reference` word for word, by the fewest
/// edits: each pair `(reference index, hypothesis index)` is a word in both,
/// the same or substituted; the words in neither pair were deleted or
/// inserted. Where two alignments cost the same, a substitution is taken
/// before a deletion, and a deletion before an insertion.
pub fn align(reference: &[&str], hypothesis: &[&str]) -> (usize, Vec<(usize, usize)>) {
    let (n, m) = (reference.len(), hypothesis.len());
    // d[i][j]: the edits that turn the first i words into the first j.
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        d[i][0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]) as usize;
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    let (mut i, mut j, mut pairs) = (n, m, Vec::new());
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]) as usize {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if i > 0 && d[i][j] == d[i - 1][j] + 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    (d[n][m], pairs)
}

/// Speakers in order of first appearance, and each word's index among them.
fn speakers(words: &[Spoken]) -> (Vec<&str>, Vec<usize>) {
    let mut names: Vec<&str> = Vec::new();
    let index = words
        .iter()
        .map(|w| {
            names.iter().position(|n| *n == w.speaker).unwrap_or_else(|| {
                names.push(&w.speaker);
                names.len() - 1
            })
        })
        .collect();
    (names, index)
}

/// The word diarization error, Shafey et al.'s (2019), as a fraction: of the
/// words [`align`] pairs with a reference word (right or substituted), the
/// share whose speaker is not the reference word's, after mapping the
/// hypothesis's speakers onto the reference's so that most pairs agree.
/// `None` when no word pairs at all.
///
/// Which words pair depends on how [`align`] breaks ties, and moves the
/// number by a word now and then: jiwer's alignment put one fixture 0.2
/// points away. No package scores WDER, so the fixture script writes its own
/// with the same rule.
pub fn wder(reference: &[Spoken], hypothesis: &[Spoken]) -> Option<f64> {
    let (r, h): (Vec<&str>, Vec<&str>) = (reference.iter().map(|w| w.word.as_str()).collect(), hypothesis.iter().map(|w| w.word.as_str()).collect());
    let (_, pairs) = align(&r, &h);
    if pairs.is_empty() {
        return None;
    }
    let (refs, r_of) = speakers(reference);
    let (hyps, h_of) = speakers(hypothesis);
    let mut agree = vec![vec![0.0; refs.len()]; hyps.len()];
    for &(i, j) in &pairs {
        agree[h_of[j]][r_of[i]] += 1.0;
    }
    let mapped = most(&agree);
    let wrong = pairs.iter().filter(|&&(i, j)| mapped[h_of[j]] != Some(r_of[i])).count();
    Some(wrong as f64 / pairs.len() as f64)
}

/// The concatenated minimum-permutation word error rate, CHiME-6's, as
/// `(errors, reference words)`: each speaker's words joined in order, each
/// hypothesis speaker matched to at most one reference speaker so that the
/// edits come to the fewest, and a speaker on either side with no partner
/// counted as all deleted or all inserted. `meeteval`'s `cpwer` is the
/// reference it is held to.
pub fn cpwer(reference: &[Spoken], hypothesis: &[Spoken]) -> (usize, usize) {
    let (refs, hyps) = (by_speaker(reference), by_speaker(hypothesis));
    // Square, with empty speakers on the shorter side: a speaker matched to
    // one of those is all deletions or all insertions.
    let n = refs.len().max(hyps.len());
    let none = Vec::new();
    let (r_of, h_of) = (|k: usize| refs.get(k).unwrap_or(&none), |k: usize| hyps.get(k).unwrap_or(&none));
    let cost: Vec<Vec<f64>> = (0..n).map(|r| (0..n).map(|h| align(r_of(r), h_of(h)).0 as f64).collect()).collect();
    let errors = hungarian(&cost).iter().enumerate().map(|(r, h)| cost[r][h.expect("square")] as usize).sum();
    (errors, reference.len())
}

/// Each speaker's words, in order, the speakers in order of first appearance.
fn by_speaker(words: &[Spoken]) -> Vec<Vec<&str>> {
    let (names, of) = speakers(words);
    let mut out = vec![Vec::new(); names.len()];
    for (w, &k) in words.iter().zip(&of) {
        out[k].push(w.word.as_str());
    }
    out
}

/// Text as the word error rates compare it: lower case, split at white
/// space, each word stripped of the punctuation around it, and nothing left
/// that is only punctuation.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).filter(|w| !w.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator, so that the tests need no crate.
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn hungarian_finds_the_cheapest_assignment() {
        // Against every permutation, on square and rectangular matrices,
        // with ties among the costs.
        fn best(cost: &[Vec<f64>], row: usize, used: &mut Vec<bool>) -> f64 {
            if row == cost.len() {
                return 0.0;
            }
            let mut b = f64::INFINITY;
            for j in 0..used.len() {
                if !used[j] {
                    used[j] = true;
                    b = b.min(cost[row][j] + best(cost, row + 1, used));
                    used[j] = false;
                }
            }
            b
        }
        let mut seed = 7;
        for (n, m) in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (2, 5), (3, 6), (5, 2), (6, 3), (1, 4), (4, 1)] {
            for _ in 0..20 {
                let cost: Vec<Vec<f64>> = (0..n).map(|_| (0..m).map(|_| (lcg(&mut seed) * 10.0).floor()).collect()).collect();
                let got = hungarian(&cost);
                let assigned: Vec<usize> = got.iter().flatten().copied().collect();
                assert_eq!(assigned.len(), n.min(m), "{cost:?}: {got:?}");
                let mut distinct = assigned.clone();
                distinct.sort();
                distinct.dedup();
                assert_eq!(distinct.len(), assigned.len(), "a column assigned twice: {got:?}");
                let sum: f64 = got.iter().enumerate().filter_map(|(i, j)| j.map(|j| cost[i][j])).sum();
                let want = if n <= m {
                    best(&cost, 0, &mut vec![false; m])
                } else {
                    let t: Vec<Vec<f64>> = (0..m).map(|j| (0..n).map(|i| cost[i][j]).collect()).collect();
                    best(&t, 0, &mut vec![false; n])
                };
                assert_eq!(sum, want, "{cost:?}: {got:?}");
            }
        }
    }

    fn turn(speaker: &str, start: f64, end: f64) -> Turn {
        Turn { speaker: speaker.into(), start, end }
    }

    #[test]
    fn der_by_hand() {
        // A speaks 0–10, B 8–20. The hypothesis has x for 0–9, y 9–20 and a
        // stray z 20–22: x is A and y is B.
        let reference = [turn("A", 0.0, 10.0), turn("B", 8.0, 20.0)];
        let hypothesis = [turn("x", 0.0, 9.0), turn("y", 9.0, 20.0), turn("z", 20.0, 22.0)];
        let d = der(&reference, &hypothesis, Protocol::FULL);
        // 22 s of reference speech; 8–10 has two speakers and one heard, a
        // right one each second: 2 s missed and nothing confused; 20–22, z
        // is 2 s of false alarm.
        assert_eq!(d, Der { total: 22.0, missed: 2.0, false_alarm: 2.0, confusion: 0.0 });
        // A against x: 9 shared of 10; B against y: 11 of 12.
        assert!((jer(&reference, &hypothesis, Protocol::FULL) - (0.1 + 1.0 / 12.0) / 2.0).abs() < 1e-12);
        // Without the overlap, 8–10 goes; the collar takes 0.25 either side
        // of 0, 8, 10 and 20, and 20.25–22 is still a false alarm. Each
        // speaker's scored time is then its partner's exactly.
        let d = der(&reference, &hypothesis, Protocol::FORGIVING);
        // What is scored: 0.25–7.75, 10.25–19.75 and 20.25–22.
        assert!((d.total - 17.0).abs() < 1e-9, "{d:?}");
        assert!((d.false_alarm - 1.75).abs() < 1e-9 && d.missed == 0.0 && d.confusion == 0.0, "{d:?}");
        assert!(jer(&reference, &hypothesis, Protocol::FORGIVING).abs() < 1e-12);
        // A perfect answer, and none at all.
        assert_eq!(der(&reference, &reference, Protocol::FULL).rate(), 0.0);
        assert_eq!(der(&reference, &[], Protocol::FULL).rate(), 1.0);
        assert_eq!(jer(&reference, &[], Protocol::FULL), 1.0);
    }

    #[test]
    fn rttm_reads_what_it_writes() {
        let turns = vec![turn("A", 0.5, 5.469), turn("speaker_01", 5.719, 9.868)];
        assert_eq!(rttm(&to_rttm("dialogue", &turns)).unwrap(), turns);
        assert!(rttm("SPEAKER x 1 zero 1.0 <NA> <NA> A <NA> <NA>").is_err());
        assert_eq!(rttm("SPKR-INFO x 1 <NA> <NA> <NA> unknown A <NA> <NA>\nSPEAKER x 1 1.0 0.000 <NA> <NA> A <NA> <NA>").unwrap(), vec![]);
    }

    fn spoken(text: &str) -> Vec<Spoken> {
        text.split_whitespace().map(|w| {
            let (s, w) = w.split_once(':').unwrap();
            Spoken { speaker: s.into(), word: w.into() }
        }).collect()
    }

    #[test]
    fn word_errors_by_hand() {
        let reference = spoken("A:hei B:takk B:for B:sist A:ja");
        // One word substituted, one deleted, one given to the wrong speaker.
        let hypothesis = spoken("x:hei y:takk y:fra x:sist x:ja");
        // Pairs: hei, takk, for/fra, sist, ja; x maps to A, y to B; sist is
        // B's and given to x: one of five.
        assert_eq!(wder(&reference, &hypothesis), Some(0.2));
        // A: "hei ja" against x: "hei sist ja", one insertion; B: "takk for
        // sist" against y: "takk fra", a substitution and a deletion.
        assert_eq!(cpwer(&reference, &hypothesis), (3, 5));
        // One speaker for everything is best matched to B: two insertions,
        // and A's two words deleted.
        assert_eq!(cpwer(&reference, &spoken("x:hei x:takk x:for x:sist x:ja")), (4, 5));
        assert_eq!(wder(&reference, &[]), None);
        assert_eq!(words("Hei, og velkommen til «møtet». — Ja!"), ["hei", "og", "velkommen", "til", "møtet", "ja"]);
    }

    /// Held to `pyannote.metrics` (DER, JER) and `meeteval` (cpWER) on the
    /// cases `scripts/der-fixtures.py` writes, to 0.01 points, and to the
    /// script's own WDER.
    #[test]
    fn agrees_with_pyannote_metrics() {
        let fx: serde_json::Value = serde_json::from_str(include_str!("der-fixtures.json")).unwrap();
        let cases = fx["cases"].as_array().unwrap();
        assert!(cases.len() >= 10, "the fixtures have {} cases", cases.len());
        let mut worst = 0.0f64;
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let reference = rttm(case["reference"].as_str().unwrap()).unwrap();
            let hypothesis = rttm(case["hypothesis"].as_str().unwrap()).unwrap();
            for (key, protocol) in [("full", Protocol::FULL), ("fair", Protocol::FAIR), ("forgiving", Protocol::FORGIVING)] {
                let want = &case["scores"][key];
                let d = der(&reference, &hypothesis, protocol);
                for (what, ours) in [
                    ("der", 100.0 * d.rate()),
                    ("jer", 100.0 * jer(&reference, &hypothesis, protocol)),
                    ("total", d.total),
                    ("missed", d.missed),
                    ("false_alarm", d.false_alarm),
                    ("confusion", d.confusion),
                ] {
                    let theirs = want[what].as_f64().unwrap();
                    // Rates in points, times in seconds: both to 0.01.
                    let gap = (ours - theirs).abs();
                    worst = worst.max(gap);
                    assert!(gap < 0.01, "{name} {key} {what}: ours {ours:.4}, pyannote's {theirs:.4}");
                }
            }
            if let Some(w) = case.get("words") {
                let (reference, hypothesis) = (spoken(w["reference"].as_str().unwrap()), spoken(w["hypothesis"].as_str().unwrap()));
                let (errors, length) = cpwer(&reference, &hypothesis);
                assert_eq!((errors as u64, length as u64), (w["cpwer_errors"].as_u64().unwrap(), w["cpwer_length"].as_u64().unwrap()), "{name} cpWER");
                let ours = 100.0 * wder(&reference, &hypothesis).unwrap();
                let theirs = w["wder"].as_f64().unwrap();
                assert!((ours - theirs).abs() < 0.01, "{name} WDER: ours {ours:.4}, the script's {theirs:.4}");
            }
        }
        eprintln!("{} cases at three protocols, largest gap {worst:.2e}", cases.len());
    }
}
