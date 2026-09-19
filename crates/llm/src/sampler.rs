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
use nanograd::rng::Rng;

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
        // Sort candidates by logit, descending. Sorting all 50k is wasteful
        // when top_k is small -- select_nth_unstable would be the fix -- but
        // it is a rounding error next to the matmuls.
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
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
        order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

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

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
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
}
