//! A GPT: every piece so far, assembled into something that predicts the next
//! token.
//!
//! ```text
//! ids ──> token embedding ──┐
//!                           + ──> block ──> ... ──> block ──> norm ──> head ──> logits
//! 0,1,2.. position embedding┘
//! ```
//!
//! Row `i` of the logits is the model's guess at token `i + 1`, made from
//! tokens `0..=i` only — the causal mask inside each block guarantees that. So
//! one forward pass over a sequence of length T is T predictions, and T
//! training examples, at once. That is the trick that makes training on text
//! cheap, and it is why the mask matters so much: a leak would let the model
//! read the answer.
//!
//! There is almost no new code here, because there is almost nothing new.
//! Three things are:
//!
//! # Positions
//!
//! Attention takes a weighted average of earlier positions, and an average
//! does not know what order its terms came in. To the last position, "dog
//! bites man" and "man bites dog" are the same bag of three words. So each
//! position gets a vector of its own, learned like any other embedding and
//! added to the token's. Going backwards, an addition hands the same gradient
//! to both sides, unchanged.
//!
//! # Starting out ignorant
//!
//! A model that has seen no data should predict every token with equal
//! probability, and cost exactly `ln(vocab)`. If the first loss you see is
//! higher than that, the model is starting out *confidently wrong*, and the
//! first stretch of training is spent unlearning its initialisation.
//!
//! He initialisation — right for the MNIST network — does exactly that here.
//! Its logits come out with a variance near 2, and measured over 40 seeds the
//! initial loss is 0.92 above `ln(vocab)`: nearly a whole nat of confidence
//! with nothing behind it. GPT-2's recipe is small weights everywhere,
//! N(0, 0.02), so the logits start near zero; the same measurement gives
//! 0.003. See `init`.
//!
//! # Depth, again
//!
//! The residual stream is a sum: every block adds two branch outputs to it, so
//! after N blocks it holds 2N of them, and its variance has grown 2N-fold. The
//! fix is to scale the *last* layer of each branch — the one that writes to
//! the stream — by `1/sqrt(2N)`, which makes the sum's variance independent of
//! depth.

use crate::block::Block;
use crate::embedding::Embedding;
use crate::matrix::Matrix;
use crate::nn::{prefixed, Layer, Linear, Param};
use crate::norm::LayerNorm;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct GptConfig {
    /// How many distinct tokens there are.
    pub vocab: usize,
    /// The longest sequence the model can take: the number of position vectors.
    pub context: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
}

pub struct Gpt {
    config: GptConfig,
    tokens: Embedding,
    positions: Embedding,
    blocks: Vec<Block>,
    /// The stream is never normalised on its way up, so it is normalised once
    /// here, before anything reads a prediction off it.
    norm: LayerNorm,
    /// `[d_model, vocab]`: one score per token.
    head: Linear,
}

impl Gpt {
    pub fn new(config: GptConfig, rng: &mut Rng) -> Self {
        let GptConfig { vocab, context, d_model, n_heads, n_layers } = config;
        let mut model = Gpt {
            config,
            tokens: Embedding::new(vocab, d_model, rng),
            positions: Embedding::new(context, d_model, rng),
            blocks: (0..n_layers).map(|_| Block::new(d_model, n_heads, rng)).collect(),
            norm: LayerNorm::new(d_model),
            head: Linear::new(d_model, vocab, rng),
        };
        model.init(rng);
        model
    }

    /// GPT-2's initialisation, applied over the top of what the layers chose
    /// for themselves. Picking tensors by the end of their name is how the
    /// reference implementations do it too.
    fn init(&mut self, rng: &mut Rng) {
        const STD: f32 = 0.02;
        let writes_to_stream = STD / (2.0 * self.config.n_layers as f32).sqrt();
        for p in self.params() {
            let std = if p.name.ends_with("wo.weight") || p.name.ends_with("mlp.3.weight") {
                writes_to_stream
            } else if p.name.ends_with(".weight") {
                STD
            } else {
                // Embedding tables are already N(0, 0.02). Biases stay at 0,
                // and a norm's gamma at 1.
                continue;
            };
            p.value.iter_mut().for_each(|v| *v = rng.normal() * std);
        }
    }

    /// Logits, `[ids.len(), vocab]`. Row `i` scores the candidates for token `i + 1`.
    pub fn forward(&mut self, ids: &[usize]) -> Matrix {
        assert!(
            ids.len() <= self.config.context,
            "{} tokens do not fit a context of {}",
            ids.len(),
            self.config.context
        );
        let places: Vec<usize> = (0..ids.len()).collect();

        let mut x = self.tokens.forward(ids);
        x.add_in_place(&self.positions.forward(&places));
        for block in self.blocks.iter_mut() {
            x = block.forward(&x);
        }
        let x = self.norm.forward(&x);
        self.head.forward(&x)
    }

    pub fn backward(&mut self, dlogits: &Matrix) {
        let dx = self.head.backward(dlogits);
        let mut dx = self.norm.backward(&dx);
        for block in self.blocks.iter_mut().rev() {
            dx = block.backward(&dx);
        }
        // x = tokens + positions: both sides of an addition get the gradient
        // as it is. Below them there is nothing left to blame.
        self.tokens.backward(&dx);
        self.positions.backward(&dx);
    }

    pub fn step(&mut self, lr: f32, momentum: f32) {
        self.tokens.step(lr, momentum);
        self.positions.step(lr, momentum);
        self.blocks.iter_mut().for_each(|b| b.step(lr, momentum));
        self.norm.step(lr, momentum);
        self.head.step(lr, momentum);
    }

    pub fn zero_grad(&mut self) {
        self.tokens.zero_grad();
        self.positions.zero_grad();
        self.blocks.iter_mut().for_each(|b| b.zero_grad());
        self.norm.zero_grad();
        self.head.zero_grad();
    }

    pub fn params(&mut self) -> Vec<Param<'_>> {
        let mut all = prefixed("tokens", self.tokens.params());
        all.extend(prefixed("positions", self.positions.params()));
        for (i, block) in self.blocks.iter_mut().enumerate() {
            all.extend(prefixed(&format!("blocks.{i}"), block.params()));
        }
        all.extend(prefixed("norm", self.norm.params()));
        all.extend(prefixed("head", self.head.params()));
        all
    }

    pub fn param_count(&mut self) -> usize {
        self.params().iter().map(|p| p.value.len()).sum()
    }

    pub fn summary(&mut self) -> String {
        let GptConfig { vocab, context, d_model, n_heads, n_layers } = self.config;
        format!(
            "Gpt({n_layers} layers, {n_heads} heads, d_model {d_model}, vocab {vocab}, context {context}, {} params)",
            self.param_count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{check_params, scramble};
    use crate::nn::softmax_cross_entropy;

    const CONFIG: GptConfig = GptConfig { vocab: 11, context: 6, d_model: 8, n_heads: 2, n_layers: 2 };
    /// Token 4 repeats, and the sequence is shorter than the context, so some
    /// token rows and one position row are never touched.
    const IDS: [usize; 5] = [4, 9, 4, 0, 7];
    /// The same sequence, shifted left by one: what comes next at each position.
    const NEXT: [usize; 5] = [9, 4, 0, 7, 2];

    fn loss(model: &mut Gpt) -> f32 {
        softmax_cross_entropy(&model.forward(&IDS), &NEXT).0
    }

    /// Every tensor in the model, from the head down to the embedding tables.
    ///
    /// The tolerance is looser than a single layer's, and measured rather than
    /// guessed. Through two blocks in f32 the worst tensor disagrees by 6.9e-4
    /// at this nudge; at 3e-2 curvature takes over (6.2e-3, growing with the
    /// square of the nudge) and at 1e-3 rounding does (5.4e-3). So 1e-2 is
    /// about the best this estimate can do, and 5e-3 sits 7x above it. The
    /// mildest wiring bug tried against it measures 0.28.
    #[test]
    fn analytic_gradient_matches_numerical() {
        let mut rng = Rng::new(51);
        let mut model = Gpt::new(CONFIG, &mut rng);
        // A freshly initialised GPT is a bad place to check: attention is
        // uniform and half the gradients are nearly nothing.
        scramble(model.params(), &mut rng);

        model.zero_grad();
        let (_, dlogits) = softmax_cross_entropy(&model.forward(&IDS), &NEXT);
        model.backward(&dlogits);

        let report = check_params(&mut model, Gpt::params, loss, 1e-2);
        const TOLERANCE: f32 = 5e-3;
        assert_eq!(report.len(), 2 + CONFIG.n_layers * 16 + 2 + 2);
        for c in report {
            if c.name.ends_with("wk.bias") {
                // Exactly zero by construction — see the attention tests.
                assert!(c.analytic_norm < 1e-6, "{}: gradient {:e}", c.name, c.analytic_norm);
                assert!(c.numerical_norm < 1e-4, "{}: numerical gradient {:e}", c.name, c.numerical_norm);
                continue;
            }
            assert!(c.rel < TOLERANCE, "{}: analytic and numerical gradients differ (rel {:.4})", c.name, c.rel);
        }
    }

    /// Averaged over seeds, because five predictions from one model are a
    /// noisy sample: measured over 40 seeds, a single fresh model lands up to
    /// 0.07 either side of ln(vocab), and their mean 0.003 above it. With the
    /// layers' own He initialisation left in place the mean is 0.92 above, and
    /// no single seed comes closer than 0.23.
    #[test]
    fn a_fresh_model_knows_nothing() {
        const SEEDS: u64 = 40;
        let ignorance = (CONFIG.vocab as f32).ln();
        let mean = (0..SEEDS).map(|seed| loss(&mut Gpt::new(CONFIG, &mut Rng::new(100 + seed)))).sum::<f32>()
            / SEEDS as f32;
        assert!((mean - ignorance).abs() < 0.05, "mean initial loss {mean}, ln(vocab) {ignorance}");
    }

    #[test]
    fn init_reaches_the_tensors_it_means_to() {
        let mut model = Gpt::new(GptConfig { d_model: 32, ..CONFIG }, &mut Rng::new(53));
        let std = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let scaled = 0.02 / (2.0 * CONFIG.n_layers as f32).sqrt();

        let mut writers = 0;
        for p in model.params() {
            let expected = if p.name.ends_with("wo.weight") || p.name.ends_with("mlp.3.weight") {
                writers += 1;
                scaled
            } else if p.name.ends_with(".weight") || p.name.ends_with(".table") {
                0.02
            } else if p.name.ends_with("gamma") {
                assert!(p.value.iter().all(|&v| v == 1.0), "{} was disturbed", p.name);
                continue;
            } else {
                assert!(p.value.iter().all(|&v| v == 0.0), "{} should start at zero", p.name);
                continue;
            };
            let measured = std(p.value);
            assert!((measured / expected - 1.0).abs() < 0.2, "{}: std {measured}, wanted {expected}", p.name);
        }
        // The names are matched by their endings. If a layer is renamed and
        // the match silently stops finding it, this is what notices.
        assert_eq!(writers, 2 * CONFIG.n_layers);
    }

    /// The whole model this time, not one layer: no logit may depend on a
    /// token that comes after it.
    #[test]
    fn the_model_cannot_read_ahead() {
        let mut model = Gpt::new(CONFIG, &mut Rng::new(54));
        scramble(model.params(), &mut Rng::new(55));
        let before = model.forward(&IDS);

        let mut changed = IDS;
        changed[4] = 1;
        let after = model.forward(&changed);

        for r in 0..4 {
            assert_eq!(before.row(r), after.row(r), "position {r} saw the last token");
        }
        assert_ne!(before.row(4), after.row(4));
    }

    /// `step` and `zero_grad` are forwarded by hand through every composite
    /// layer, and a forgotten line is silent: that tensor just never trains.
    #[test]
    fn step_and_zero_grad_reach_every_tensor() {
        let mut rng = Rng::new(57);
        let mut model = Gpt::new(CONFIG, &mut rng);
        scramble(model.params(), &mut rng);
        model.zero_grad();
        let (_, dlogits) = softmax_cross_entropy(&model.forward(&IDS), &NEXT);
        model.backward(&dlogits);

        let before: Vec<Vec<f32>> = model.params().iter().map(|p| p.value.to_vec()).collect();
        model.step(0.1, 0.0);
        for (p, before) in model.params().iter().zip(&before) {
            // The key bias has no gradient to move it; see the attention tests.
            if !p.name.ends_with("wk.bias") {
                assert!(p.value != before.as_slice(), "{} did not move", p.name);
            }
        }

        model.zero_grad();
        for p in model.params() {
            assert!(p.grad.iter().all(|&g| g == 0.0), "{} kept its gradient", p.name);
        }
    }

    /// The oldest sanity check there is: a model that cannot memorise one
    /// short sequence has something wrong with it. This is forward, backward
    /// and step, through every layer, all having to agree.
    #[test]
    fn it_can_memorise_one_sequence() {
        let mut model = Gpt::new(CONFIG, &mut Rng::new(56));
        let initial = loss(&mut model);
        for _ in 0..STEPS {
            model.zero_grad();
            let (_, dlogits) = softmax_cross_entropy(&model.forward(&IDS), &NEXT);
            model.backward(&dlogits);
            model.step(LR, 0.9);
        }
        let trained = loss(&mut model);
        assert!(trained < 0.05, "loss went from {initial} to only {trained}");
    }

    /// Measured: 2.41 at the start, 0.0014 after 25 steps, 0.0001 after 100.
    const STEPS: usize = 100;
    const LR: f32 = 0.05;
}
