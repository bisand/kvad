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

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits) as u32;
        }

        // Sort candidates by logit, descending. Sorting all 50k is wasteful
        // when top_k is small -- select_nth_unstable would be the fix -- but
        // it is a rounding error next to the matmuls.
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

        let k = if self.top_k == 0 { logits.len() } else { self.top_k.min(logits.len()) };
        idx.truncate(k);

        let mut probs: Vec<f32> = idx.iter().map(|&i| logits[i] / self.temperature).collect();
        softmax_inplace(&mut probs);

        // Nucleus: walk down the sorted probabilities until they account for
        // top_p of the mass, and discard the rest.
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cut = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= self.top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut);
            idx.truncate(cut);
        }

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
