//! Flow matching: how a diffusion model is trained, and how it then draws.
//!
//! # The one idea in this file
//!
//! Take a real image `x₀` and a picture of pure noise `x₁`, and draw a
//! straight line between them:
//!
//! ```text
//! xₜ = (1 − t) · x₀  +  t · x₁          t = 0 is the image, t = 1 the noise
//! ```
//!
//! Along that line the image moves at a constant velocity, `v = x₁ − x₀`. Show
//! the model a point `xₜ` somewhere on it, tell it `t`, and ask for `v`. The
//! loss is the squared error. That is the whole of training:
//!
//! ```text
//! loss = mean( (model(xₜ, t, label) − (x₁ − x₀))² )
//! ```
//!
//! The model never sees where the line started, so it can't know `v`
//! exactly: many images could have led to this `xₜ`. What minimises the
//! squared error is the *average* velocity over all of them, and that
//! average is exactly what drawing needs. Start from fresh noise at `t = 1`,
//! and walk backwards against the predicted velocity in small steps:
//!
//! ```text
//! x ← x − Δt · model(x, t, label)       from t = 1 down to t = 0
//! ```
//!
//! Each step lands a little closer to *some* image the model has learned,
//! and at `t = 0` there is one. That is Euler's method for an ordinary
//! differential equation, and the samplers in `crates/gpu` (FLUX, Qwen-Image,
//! LTX) are this loop, with a better step rule than Euler's.
//!
//! # Guidance
//!
//! During training the label is thrown away one time in ten and replaced by
//! "none". So the same model learns to draw a 7 and to draw a digit, any
//! digit. At drawing time, ask it both and push further in the direction the
//! label makes a difference in:
//!
//! ```text
//! v = v_none + g · (v_label − v_none)
//! ```
//!
//! `g = 1` is the labelled prediction as it is. Larger `g` exaggerates what
//! makes a 7 a 7, and gives cleaner, less varied digits. This is the
//! `guidance_scale` of every text-to-image model, and it is why they run the
//! model twice per step.
//!
//! # A loss that says little
//!
//! A batch's loss depends mostly on which `t` its examples happened to draw:
//! near `t = 0` the velocity is nearly all noise and easy to guess from the
//! input, near `t = 1` it is nearly all image and cannot be. So the training
//! loss is noisy in a way that has nothing to do with learning. [`validate`]
//! fixes the images, the noise and the `t`s, so two checkpoints are measured
//! on exactly the same questions.

use crate::dit::Dit;
use crate::optim::AdamW;
use crate::rng::Rng;

/// How often a label is dropped in training, so that guidance has an
/// unconditional model to push away from. The DiT and SD papers use 10%.
pub const DROP_LABEL: f32 = 0.1;

/// One training example, fully decided: which image, how noisy, what noise,
/// and which label the model is told. Drawn before any work starts, so that
/// the same draws happen however many threads share them.
pub struct Example {
    pub image: usize,
    pub t: f32,
    pub noise: Vec<f32>,
    pub label: usize,
}

/// `(xₜ, v)`: the point on the line, and the velocity along it.
pub fn noisy(x0: &[f32], noise: &[f32], t: f32) -> (Vec<f32>, Vec<f32>) {
    let xt = x0.iter().zip(noise).map(|(x, n)| (1.0 - t) * x + t * n).collect();
    let v = x0.iter().zip(noise).map(|(x, n)| n - x).collect();
    (xt, v)
}

/// Mean squared error, and its gradient: `2 (predicted − target) / n`.
pub fn mse(predicted: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let n = predicted.len() as f32;
    let loss = predicted.iter().zip(target).map(|(p, t)| (p - t) * (p - t)).sum::<f32>() / n;
    (loss, predicted.iter().zip(target).map(|(p, t)| 2.0 * (p - t) / n).collect())
}

/// Draw `batch` examples from `images`.
///
/// `t` is drawn uniformly. Stable Diffusion 3 found a distribution bunched
/// in the middle ("logit-normal") trained better at their scale, because the
/// ends of the line are easy. Whether it does here is a thing to measure.
pub fn draw(images: &[Vec<f32>], labels: &[usize], unconditional: usize, batch: usize, rng: &mut Rng) -> Vec<Example> {
    (0..batch)
        .map(|_| {
            let image = rng.below(images.len());
            let t = rng.uniform();
            let noise = (0..images[image].len()).map(|_| rng.normal()).collect();
            let label = if rng.uniform() < DROP_LABEL { unconditional } else { labels[image] };
            Example { image, t, noise, label }
        })
        .collect()
}

/// Forward and backward on some examples, adding into the model's gradients.
/// Returns the sum of their losses.
fn accumulate(model: &mut Dit, images: &[Vec<f32>], examples: &[&Example], batch: usize) -> f32 {
    let mut total = 0.0;
    for e in examples {
        let (xt, v) = noisy(&images[e.image], &e.noise, e.t);
        let (loss, mut dv) = mse(&model.forward(&xt, e.t, e.label), &v); //    predict, score
        // A mean over the whole batch, not over this thread's share of it.
        dv.iter_mut().for_each(|g| *g /= batch as f32);
        model.backward(&dv); //                                                  blame
        total += loss;
    }
    total
}

/// One step of training. Returns the batch's mean loss.
pub fn train_step(model: &mut Dit, opt: &mut AdamW, images: &[Vec<f32>], examples: &[Example]) -> f32 {
    model.zero_grad();
    let all: Vec<&Example> = examples.iter().collect();
    let total = accumulate(model, images, &all, examples.len());
    opt.step(model.params()); //                                                 adjust
    total / examples.len() as f32
}

/// Copies of a model, one to a thread, sharing each batch: data parallelism,
/// exactly as `text::Replicas` does it for the GPT. See there for why each
/// replica is a whole model.
pub struct Replicas {
    models: Vec<Dit>,
}

impl Replicas {
    pub fn new(model: &Dit, threads: usize) -> Self {
        assert!(threads > 0, "training needs at least one thread");
        Replicas { models: (0..threads).map(|_| Dit::new(model.config(), &mut Rng::new(0))).collect() }
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    fn broadcast(&mut self, model: &mut Dit) {
        self.models.iter_mut().for_each(|replica| replica.copy_from(model));
    }

    /// The same step as [`train_step`], with example `i` on replica `i mod n`.
    pub fn train_step(&mut self, model: &mut Dit, opt: &mut AdamW, images: &[Vec<f32>], examples: &[Example]) -> f32 {
        self.broadcast(model);
        let n = self.models.len();
        let batch = examples.len();
        let total: f32 = std::thread::scope(|scope| {
            let running: Vec<_> = self
                .models
                .iter_mut()
                .enumerate()
                .map(|(r, replica)| {
                    scope.spawn(move || {
                        replica.zero_grad();
                        let mine: Vec<&Example> = examples.iter().skip(r).step_by(n).collect();
                        accumulate(replica, images, &mine, batch)
                    })
                })
                .collect();
            running.into_iter().map(|thread| thread.join().expect("a training thread panicked")).sum()
        });

        model.zero_grad();
        for replica in self.models.iter_mut() {
            for (ours, theirs) in model.params().into_iter().zip(replica.params()) {
                ours.grad.iter_mut().zip(theirs.grad.iter()).for_each(|(sum, g)| *sum += g); // all-reduce
            }
        }
        opt.step(model.params());
        total / batch as f32
    }

    /// Draw one image per request, the requests shared out across the
    /// replicas. Request `i` is drawn from `Rng::new(seeds[i])`, so an image
    /// does not depend on how many threads drew it or what else was drawn.
    pub fn sample(&mut self, model: &mut Dit, requests: &[(usize, u64)], steps: usize, guidance: f32) -> Vec<Vec<f32>> {
        self.broadcast(model);
        let n = self.models.len();
        let mut out = vec![Vec::new(); requests.len()];
        std::thread::scope(|scope| {
            let running: Vec<_> = self
                .models
                .iter_mut()
                .enumerate()
                .map(|(r, replica)| {
                    scope.spawn(move || {
                        let mut drawn = Vec::new();
                        for (i, &(label, seed)) in requests.iter().enumerate().skip(r).step_by(n) {
                            drawn.push((i, sample(replica, label, steps, guidance, &mut Rng::new(seed))));
                        }
                        drawn
                    })
                })
                .collect();
            for thread in running {
                for (i, image) in thread.join().expect("a drawing thread panicked") {
                    out[i] = image;
                }
            }
        });
        out
    }
}

/// The `t`s every validation measures at: the middles of eight equal slices
/// of the line, so the ends and the middle count the same.
pub const VALIDATION_TIMES: usize = 8;

/// Mean loss on the first `count` of `images`, at fixed `t`s and fixed noise.
///
/// Every checkpoint of a run asks the same questions, so the numbers can be
/// compared with each other, which the training loss cannot (see the top of
/// this file). Labels are always given: this is the model that guidance
/// starts from.
pub fn validate(model: &mut Dit, images: &[Vec<f32>], labels: &[usize], count: usize) -> f32 {
    let mut rng = Rng::new(VALIDATION_SEED);
    let count = count.min(images.len());
    let mut total = 0.0;
    for i in 0..count {
        let noise: Vec<f32> = (0..images[i].len()).map(|_| rng.normal()).collect();
        for k in 0..VALIDATION_TIMES {
            let t = (k as f32 + 0.5) / VALIDATION_TIMES as f32;
            let (xt, v) = noisy(&images[i], &noise, t);
            total += mse(&model.forward(&xt, t, labels[i]), &v).0;
        }
    }
    total / (count * VALIDATION_TIMES) as f32
}

const VALIDATION_SEED: u64 = 20_260_929;

/// Draw one image of `label`, from noise, in `steps` Euler steps.
///
/// With `guidance` other than 1, the model is run twice a step — with the
/// label and without — and the two answers are extrapolated; see the top of
/// this file. At exactly 1 the second run could change nothing, so it is
/// skipped.
pub fn sample(model: &mut Dit, label: usize, steps: usize, guidance: f32, rng: &mut Rng) -> Vec<f32> {
    sample_watched(model, label, steps, guidance, rng, &mut |_, _| true).expect("nothing asked it to stop")
}

/// [`sample`], calling `on_step(done, x)` after every step with the image as
/// it stands. Returning `false` stops the drawing, which then returns `None`
/// rather than a half-drawn image.
pub fn sample_watched(
    model: &mut Dit,
    label: usize,
    steps: usize,
    guidance: f32,
    rng: &mut Rng,
    on_step: &mut dyn FnMut(usize, &[f32]) -> bool,
) -> Option<Vec<f32>> {
    let config = model.config();
    let mut x: Vec<f32> = (0..config.pixels()).map(|_| rng.normal()).collect();
    let dt = 1.0 / steps as f32;
    for k in 0..steps {
        let t = 1.0 - k as f32 * dt;
        let mut v = model.forward(&x, t, label);
        if guidance != 1.0 {
            let none = model.forward(&x, t, config.unconditional());
            v.iter_mut().zip(&none).for_each(|(v, n)| *v = n + guidance * (*v - n));
        }
        x.iter_mut().zip(&v).for_each(|(x, v)| *x -= dt * v);
        if !on_step(k + 1, &x) {
            return None;
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::DitConfig;

    const CONFIG: DitConfig = DitConfig { image: 8, channels: 1, patch: 4, classes: 2, d_model: 16, n_heads: 2, n_layers: 2 };

    /// Two "images": a bright top half, and a bright left half.
    fn two_images() -> (Vec<Vec<f32>>, Vec<usize>) {
        let n = CONFIG.image;
        let top = (0..n * n).map(|i| if i / n < n / 2 { 1.0 } else { -1.0 }).collect();
        let left = (0..n * n).map(|i| if i % n < n / 2 { 1.0 } else { -1.0 }).collect();
        (vec![top, left], vec![0, 1])
    }

    #[test]
    fn the_line_runs_from_the_image_to_the_noise_at_constant_speed() {
        let (x0, noise) = (vec![1.0, -2.0], vec![0.5, 3.0]);
        assert_eq!(noisy(&x0, &noise, 0.0).0, x0);
        assert_eq!(noisy(&x0, &noise, 1.0).0, noise);
        // One Euler step of the true velocity goes from t = 1 straight to t = 0.
        let (x1, v) = noisy(&x0, &noise, 1.0);
        let back: Vec<f32> = x1.iter().zip(&v).map(|(x, v)| x - v).collect();
        assert_eq!(back, x0);
    }

    /// The oldest check: a model that cannot learn two images has something
    /// wrong with it. Then check that it listens to the label, and draws the
    /// image it is asked for from noise it has never seen.
    #[test]
    fn it_learns_two_images_and_draws_the_one_it_is_asked_for() {
        let (images, labels) = two_images();
        let mut rng = Rng::new(81);
        let mut model = Dit::new(CONFIG, &mut rng);
        let mut opt = AdamW::new(3e-3);
        let before = validate(&mut model, &images, &labels, 2);
        for _ in 0..STEPS {
            let examples = draw(&images, &labels, CONFIG.unconditional(), 16, &mut rng);
            train_step(&mut model, &mut opt, &images, &examples);
        }
        let after = validate(&mut model, &images, &labels, 2);
        assert!(after < 0.25 * before, "validation loss went from {before} to only {after}");

        // Told the wrong label, it should be worse at every noise level: the
        // label is information, and it is being used all along the line. On
        // seeds 81..781 the smallest margin measured was 1.7 times.
        let noise: Vec<f32> = (0..CONFIG.pixels()).map(|_| rng.normal()).collect();
        for k in 0..VALIDATION_TIMES {
            let t = (k as f32 + 0.5) / VALIDATION_TIMES as f32;
            for label in 0..2 {
                let (xt, v) = noisy(&images[label], &noise, t);
                let right = mse(&model.forward(&xt, t, label), &v).0;
                let wrong = mse(&model.forward(&xt, t, 1 - label), &v).0;
                assert!(1.25 * right < wrong, "t {t}, label {label}: {right} with it and {wrong} with the other");
            }
        }

        // Most, not all. A model this small, shown two images, has learned
        // nothing about most of the noise it could be started from, and now
        // and then a draw wanders off and comes back as neither image. On
        // seeds 81..781 at this length, 9 to 12 of the 12 draws were right.
        let mut right = 0;
        for seed in 0..DRAWS {
            for label in 0..2 {
                let drawn = sample(&mut model, label, 20, 1.0, &mut Rng::new(900 + seed));
                let distance = |i: usize| drawn.iter().zip(&images[i]).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
                right += (distance(label) < 0.25 * distance(1 - label)) as u64;
            }
        }
        assert!(3 * right >= 2 * (2 * DRAWS), "{right} of {} draws were the image asked for", 2 * DRAWS);
    }

    /// Measured on seed 81: validation 2.39 at the start and 0.20 after 1500
    /// steps. At 600 it was 0.43, and seeds 281 and 481 still drew the same
    /// image whichever label they were given.
    const STEPS: usize = 1500;
    const DRAWS: u64 = 6;

    /// Threads must not change what is learned, only how fast.
    #[test]
    fn replicas_take_the_step_one_model_would() {
        let (images, labels) = two_images();
        let run = |threads: usize| {
            let mut rng = Rng::new(83);
            let mut model = Dit::new(CONFIG, &mut rng);
            let mut replicas = Replicas::new(&model, threads);
            let mut opt = AdamW::new(1e-2);
            for _ in 0..5 {
                let examples = draw(&images, &labels, CONFIG.unconditional(), 6, &mut rng);
                match threads {
                    1 => train_step(&mut model, &mut opt, &images, &examples),
                    _ => replicas.train_step(&mut model, &mut opt, &images, &examples),
                };
            }
            model.params().into_iter().flat_map(|p| p.value.to_vec()).collect::<Vec<f32>>()
        };
        let one = run(1);
        for threads in [2, 3, 4] {
            let worst = one.iter().zip(run(threads)).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(worst < 1e-4, "{threads} threads moved a weight by {worst:e}");
        }
    }

    /// And drawing on several threads draws the same images as drawing on one.
    #[test]
    fn an_image_depends_on_its_seed_and_not_on_the_threads() {
        let mut rng = Rng::new(84);
        let mut model = Dit::new(CONFIG, &mut rng);
        crate::gradcheck::scramble(model.params(), &mut rng);
        let requests: Vec<(usize, u64)> = (0..5).map(|i| (i % 3, 100 + i as u64)).collect();
        let alone: Vec<Vec<f32>> =
            requests.iter().map(|&(label, seed)| sample(&mut model, label, 4, 2.0, &mut Rng::new(seed))).collect();
        let shared = Replicas::new(&model, 3).sample(&mut model, &requests, 4, 2.0);
        assert_eq!(alone, shared);
    }
}
