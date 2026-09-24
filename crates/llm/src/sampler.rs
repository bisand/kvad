//! Turning logits into a token.
//!
//! The model gives you 50257 numbers. How you collapse those into one choice
//! is not part of the model at all, and it changes the output character more
//! than most people expect.
//!
//! * **Greedy** (always the argmax) is deterministic and tends to loop:
//!   "the the the". The most likely token at every step is not the most likely
//!   sentence.
//! * **Temperature** divides the logits before the softmax. Below 1.0 sharpens
//!   the distribution, above 1.0 flattens it. It is a confidence dial.
//! * **Top-k** keeps only the k best candidates. Cheap guard against sampling
//!   something absurd from the long tail.
//! * **Top-p** (nucleus) keeps the smallest set whose probabilities sum to p.
//!   Adapts to the distribution: wide where the model is unsure, narrow where
//!   it is confident.

use crate::tensor::softmax_inplace;
use nervus::rng::Rng;

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Sampler { temperature, top_k, top_p, rng: Rng::new(seed) }
    }

    /// Start the generator again from `seed`.
    ///
    /// Sampling is a stream, not a function: the same logits give different
    /// tokens depending on how many draws came before. Anything that has to
    /// be reproducible has to be able to say where the stream begins.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits) as u32;
        }

        let (idx, probs) = self.shortlist(logits);

        // Inverse-CDF sampling over whatever survived.
        let total: f32 = probs.iter().sum();
        let mut target = self.rng.uniform() * total;
        for (i, &p) in probs.iter().enumerate() {
            target -= p;
            if target <= 0.0 {
                return idx[i] as u32;
            }
        }
        *idx.last().unwrap() as u32
    }

    /// The candidates this sampler would choose between, best first, with the
    /// probabilities it would use.
    fn shortlist(&self, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        match self.top_k {
            // Only the best `k` can survive, so only they are put in order.
            // `select_nth_unstable_by` moves them to the front in one linear
            // pass, and then there are `k` to sort instead of the vocabulary.
            // `cut` sees exactly the list a full sort would have handed it.
            //
            // A full sort of Qwen2.5's 151,936 logits was 1.1–1.9 ms a
            // token on an M5 Pro (`examples/decode_breakdown`), next to about
            // 8 ms for the whole step of a 1.5B model at q8. It was once
            // "a rounding error next to the matmuls", with GPT-2's 50k
            // vocabulary and a slower engine; it stopped being one.
            k if k > 0 && k < idx.len() => {
                idx.select_nth_unstable_by(k - 1, |&a, &b| best_first(logits, a, b));
                idx.truncate(k);
                idx.sort_unstable_by(|&a, &b| best_first(logits, a, b));
            }
            // Without a top-k, top-p can reach anywhere down the order, so
            // there is nothing to leave unsorted.
            _ => idx.sort_unstable_by(|&a, &b| best_first(logits, a, b)),
        }
        let probs = self.cut(logits, &mut idx);
        (idx, probs)
    }

    /// Apply temperature, top-k and top-p to an order that is already sorted.
    ///
    /// Truncates `order` to what survives and returns the reweighted
    /// probabilities of exactly those. The one place the filters are
    /// implemented, so that what the sampler draws from and what the
    /// playground draws cannot drift apart.
    fn cut(&self, logits: &[f32], order: &mut Vec<usize>) -> Vec<f32> {
        let k = if self.top_k == 0 { order.len() } else { self.top_k.min(order.len()) };
        order.truncate(k);

        let mut probs: Vec<f32> = order.iter().map(|&i| logits[i] / self.temperature).collect();
        softmax_inplace(&mut probs);

        // Nucleus: walk down the sorted probabilities until they account for
        // top_p of the mass, and discard the rest.
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            order.truncate(keep);
        }
        probs
    }

    /// Sample, and say what the choice was between.
    ///
    /// The `prob` of each candidate is the **model's** probability — a plain
    /// softmax over every logit, at temperature 1 — rather than the sampler's
    /// reweighted one. That is the number worth showing: it is what the model
    /// believes, and it does not move when somebody drags the temperature
    /// slider. What the sampler did is in `kept`, which says whether top-k and
    /// top-p left that token in play at all.
    ///
    /// Costs one softmax over the vocabulary per token, which is tens of
    /// microseconds against tens of milliseconds of matmul.
    pub fn sample_explained(&mut self, logits: &[f32], k: usize) -> (u32, Vec<Ranked>) {
        let chosen = self.sample(logits);
        let mut top = self.explain(logits, k);
        if let Some(r) = top.iter_mut().find(|r| r.id == chosen) {
            r.chosen = true;
        }
        (chosen, top)
    }

    /// The `k` most likely tokens and what the model thinks of them.
    pub fn explain(&self, logits: &[f32], k: usize) -> Vec<Ranked> {
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_unstable_by(|&a, &b| best_first(logits, a, b));

        // The filters keep a prefix of this same order, so "which survived" is
        // a count rather than a set — and one sort does for both questions.
        let kept = self.kept_count(logits, &order);

        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        order
            .into_iter()
            .take(k.min(logits.len()))
            .enumerate()
            .map(|(rank, i)| Ranked {
                id: i as u32,
                prob: probs[i],
                kept: rank < kept,
                chosen: false,
            })
            .collect()
    }

    /// How many of `order` top-k and top-p leave in play.
    ///
    /// Greedy decoding keeps exactly one — there is no choice being made, and
    /// a view that showed ten candidates as live would be describing a
    /// sampler nobody asked for.
    fn kept_count(&self, logits: &[f32], order: &[usize]) -> usize {
        if self.temperature <= 0.0 {
            return 1;
        }
        let mut survivors = order.to_vec();
        self.cut(logits, &mut survivors);
        survivors.len()
    }
}

/// One candidate for the next token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ranked {
    pub id: u32,
    /// The model's own probability, over the whole vocabulary.
    pub prob: f32,
    /// Whether top-k and top-p left this token in play.
    pub kept: bool,
    /// Whether it is the one that was drawn.
    pub chosen: bool,
}

/// The order candidates are ranked in: highest logit first, and equal logits
/// by token id. The tie-break makes the order a function of the logits
/// alone, so the shortlist and the playground's view cannot disagree about
/// which of two equal tokens comes first, whichever way they got there.
fn best_first(logits: &[f32], a: usize, b: usize) -> std::cmp::Ordering {
    logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
}

/// The first index of the largest value.
///
/// The best value is kept in a register rather than read back from `v` at
/// every step, which is the difference between a loop the compiler can keep
/// tight and one that reloads from memory 151,936 times a token.
pub fn argmax(v: &[f32]) -> usize {
    let Some(&first) = v.first() else { return 0 };
    let (mut best, mut top) = (0, first);
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > top {
            best = i;
            top = x;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits with a clear order: token 3 is the model's favourite.
    fn logits() -> Vec<f32> {
        vec![0.0, 1.0, 2.0, 5.0, 1.5, -1.0, 0.5, 3.0]
    }

    /// The probabilities shown are the model's, over the whole vocabulary, so
    /// they sum to one and do not move when the temperature does.
    #[test]
    fn the_view_shows_what_the_model_believes_not_what_the_sampler_reweighted() {
        let cold = Sampler::new(0.1, 0, 1.0, 1);
        let hot = Sampler::new(2.0, 0, 1.0, 1);
        let cold = cold.explain(&logits(), 8);
        let hot = hot.explain(&logits(), 8);

        let total: f32 = cold.iter().map(|r| r.prob).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
        for (a, b) in cold.iter().zip(hot.iter()) {
            assert_eq!(a.id, b.id);
            assert!((a.prob - b.prob).abs() < 1e-6, "temperature moved the model's own number");
        }
        // Best first, and the best is the biggest logit.
        assert_eq!(cold[0].id, 3);
        assert!(cold[0].prob > cold[1].prob);
    }

    /// `kept` is the sampler's half of the answer: which of those candidates
    /// could actually have been drawn.
    #[test]
    fn the_filters_say_how_many_tokens_were_really_in_play() {
        let k3 = Sampler::new(1.0, 3, 1.0, 1).explain(&logits(), 8);
        assert_eq!(k3.iter().filter(|r| r.kept).count(), 3);
        // And they are the first three, because the order is the same one.
        assert!(k3[..3].iter().all(|r| r.kept));

        // Greedy is not a choice at all: exactly one token is live.
        let greedy = Sampler::new(0.0, 40, 0.95, 1).explain(&logits(), 8);
        assert_eq!(greedy.iter().filter(|r| r.kept).count(), 1);
        assert!(greedy[0].kept);

        // Nucleus keeps fewer than top-k when the distribution is peaked.
        let nucleus = Sampler::new(1.0, 8, 0.5, 1).explain(&logits(), 8);
        let live = nucleus.iter().filter(|r| r.kept).count();
        assert!((1..8).contains(&live), "{live}");
    }

    /// Explaining a draw must describe *that* draw, and the token it names
    /// must be one the filters allowed.
    #[test]
    fn the_token_drawn_is_marked_and_was_allowed() {
        let mut s = Sampler::new(0.8, 4, 0.95, 42);
        for _ in 0..20 {
            let (chosen, top) = s.sample_explained(&logits(), 8);
            let marked: Vec<u32> = top.iter().filter(|r| r.chosen).map(|r| r.id).collect();
            assert_eq!(marked, [chosen]);
            assert!(
                top.iter().find(|r| r.id == chosen).unwrap().kept,
                "drew a token the filters had removed"
            );
        }
    }

    /// Explaining must not change what the sampler draws: the same seed and
    /// the same logits give the same tokens either way.
    #[test]
    fn explaining_a_draw_does_not_change_it() {
        let plain: Vec<u32> = {
            let mut s = Sampler::new(0.8, 4, 0.95, 7);
            (0..10).map(|_| s.sample(&logits())).collect()
        };
        let explained: Vec<u32> = {
            let mut s = Sampler::new(0.8, 4, 0.95, 7);
            (0..10).map(|_| s.sample_explained(&logits(), 5).0).collect()
        };
        assert_eq!(plain, explained);
    }

    /// The shortlist as it was before `select_nth_unstable`: every logit
    /// sorted, then cut. With the same tie-break, so that the two can be held
    /// to the same answer.
    fn shortlist_by_full_sort(s: &Sampler, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| best_first(logits, a, b));
        let probs = s.cut(logits, &mut idx);
        (idx, probs)
    }

    /// `argmax` as it was, reading the best value back from the slice at
    /// every step.
    fn argmax_reloading(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    /// A vocabulary's worth of logits, with ties: rounded to a coarse grid,
    /// so that equal values are common and the tie-break is exercised.
    fn vocabulary(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| ((rng.uniform() * 24.0 - 12.0) * 4.0).round() / 4.0).collect()
    }

    /// Selecting the top k and sorting only those hands `cut` exactly what
    /// a full sort did: the same candidates in the same order, and the same
    /// probabilities bit for bit.
    #[test]
    fn the_shortlist_is_the_one_a_full_sort_gives() {
        for seed in 0..20 {
            let logits = vocabulary(5000, seed);
            for (t, k, p) in [(0.7, 40, 0.95), (1.0, 1, 1.0), (0.3, 5, 0.5), (1.5, 4999, 0.9), (0.7, 5000, 0.95), (0.7, 9000, 1.0), (0.7, 0, 0.95)] {
                let s = Sampler::new(t, k, p, 1);
                assert_eq!(s.shortlist(&logits), shortlist_by_full_sort(&s, &logits), "seed {seed}, t {t} k {k} p {p}");
            }
        }
    }

    /// And so the same seed draws the same tokens.
    #[test]
    fn the_same_seed_draws_the_same_tokens() {
        let logits = vocabulary(5000, 3);
        let mut fast = Sampler::new(0.7, 40, 0.95, 11);
        let reference = Sampler::new(0.7, 40, 0.95, 11);
        let mut rng = Rng::new(11);
        for _ in 0..200 {
            let (idx, probs) = shortlist_by_full_sort(&reference, &logits);
            let total: f32 = probs.iter().sum();
            let mut target = rng.uniform() * total;
            let mut want = *idx.last().unwrap() as u32;
            for (i, &p) in probs.iter().enumerate() {
                target -= p;
                if target <= 0.0 {
                    want = idx[i] as u32;
                    break;
                }
            }
            assert_eq!(fast.sample(&logits), want);
        }
    }

    /// The first of equal maxima, as before, and NaN never wins.
    #[test]
    fn argmax_takes_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 2);
        assert_eq!(argmax(&[-1.0, -0.5, -3.0]), 1);
        for seed in 0..10 {
            let v = vocabulary(5000, seed);
            assert_eq!(argmax(&v), argmax_reloading(&v));
        }
    }

    /// What a token's sampling costs at Qwen2.5's vocabulary, the old way
    /// and the new, taking turns. A measurement, not a test:
    ///
    ///     cargo test --release -p kvad sampler::tests::cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost() {
        let logits = vocabulary(151_936, 5);
        let s = Sampler::new(0.7, 40, 0.95, 1);
        let time = |f: &dyn Fn()| {
            let t = std::time::Instant::now();
            for _ in 0..50 {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / 50.0
        };
        let (mut old, mut new, mut greedy, mut greedy_old) = (vec![], vec![], vec![], vec![]);
        for _ in 0..7 {
            old.push(time(&|| drop(std::hint::black_box(shortlist_by_full_sort(&s, &logits)))));
            new.push(time(&|| drop(std::hint::black_box(s.shortlist(&logits)))));
            greedy.push(time(&|| { std::hint::black_box(argmax(&logits)); }));
            greedy_old.push(time(&|| { std::hint::black_box(argmax_reloading(&logits)); }));
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!("151,936 logits, t 0.7, top-k 40, top-p 0.95:");
        println!("  full sort   {:.3} ms", median(old));
        println!("  top-k first {:.3} ms", median(new));
        println!("greedy:");
        println!("  argmax, reloading the best   {:.3} ms", median(greedy_old));
        println!("  argmax, best in a register   {:.3} ms", median(greedy));
    }

}
