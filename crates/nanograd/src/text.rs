//! Everything between a text file and a trained model: tokens, training
//! windows, one step of training, and sampling.
//!
//! # Where the training examples come from
//!
//! Nobody labels this data. The text is its own answer key: for every position
//! the "label" is simply the character that comes next. Take any window of
//! `context + 1` characters; the first `context` are the input and the last
//! `context`, shifted by one, are the targets.
//!
//! ```text
//! text      T  o     b  e  ,     o  r
//! input     T  o     b  e  ,     o
//! target    o     b  e  ,     o  r
//! ```
//!
//! One window is `context` training examples at once, because row `i` of the
//! model's output predicts from characters `0..=i` alone. A megabyte of text
//! holds a million windows. This is the whole reason language models could be
//! scaled: the supervision is free.
//!
//! # Characters, not words
//!
//! The tokeniser here gives every distinct character an id. Real models use
//! subword pieces (BPE), which pack about four characters into a token and so
//! see four times as far with the same context. Characters are used here
//! because they need no training of their own and no vocabulary file, and
//! because watching a model discover *spelling* from nothing is half the fun.
//!
//! A saved model is useless without the tokeniser it was trained with: id 17
//! means whatever character was seventeenth in *that* text. So the vocabulary
//! is saved beside the weights, as a `tokenizer.json` in the format the
//! HuggingFace `tokenizers` library reads. That library has no character
//! tokeniser, but it does not need one. BPE starts from single characters and
//! applies a list of learned merges; with an empty list it stops where it
//! started.

use crate::checkpoint::{invalid, read_text, write_whole};
use crate::json::{object, Json};
use crate::model::Gpt;
use crate::nn::softmax_cross_entropy;
use crate::optim::{AdamW, Schedule};
use crate::rng::Rng;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// One id per distinct character, in sorted order.
pub struct CharTokenizer {
    /// `chars[id]` is the character; sorted, so the reverse is a binary search.
    chars: Vec<char>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        CharTokenizer { chars }
    }

    pub fn vocab(&self) -> usize {
        self.chars.len()
    }

    /// Every character this tokeniser has an id for, in order.
    ///
    /// [`encode`](Self::encode) reports the *first* character it cannot
    /// handle, which is what a training run needs — it is about to fail and
    /// the message should name the culprit. Anything that wants to say what a
    /// text would cost *before* running it needs the whole set to compare
    /// against, and this is it.
    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// Fails with the offending character if the text holds one this
    /// tokeniser never saw: the model has no row for it.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, char> {
        text.chars().map(|c| self.chars.binary_search(&c).map_err(|_| c)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.chars[id]).collect()
    }

    /// BPE with no merges, and a decoder that joins the pieces with nothing
    /// between them.
    fn to_json(&self) -> Json {
        let vocab = self.chars.iter().enumerate().map(|(id, c)| (c.to_string(), id.into())).collect();
        object([
            ("version", "1.0".into()),
            ("truncation", Json::Null),
            ("padding", Json::Null),
            ("added_tokens", Json::Array(vec![])),
            ("normalizer", Json::Null),
            // No splitting into words first: spaces and newlines are
            // characters like any other, and the model predicts them too.
            ("pre_tokenizer", Json::Null),
            ("post_processor", Json::Null),
            ("decoder", object([("type", "Fuse".into())])),
            (
                "model",
                object([
                    ("type", "BPE".into()),
                    ("dropout", Json::Null),
                    ("unk_token", Json::Null),
                    ("continuing_subword_prefix", Json::Null),
                    ("end_of_word_suffix", Json::Null),
                    ("fuse_unk", Json::Bool(false)),
                    ("byte_fallback", Json::Bool(false)),
                    ("vocab", Json::Object(vocab)),
                    ("merges", Json::Array(vec![])),
                ]),
            ),
        ])
    }

    /// Only a vocabulary this tokeniser could have written: single
    /// characters, ids `0..n` in character order, no merges.
    fn from_json(json: &Json) -> Result<Self, String> {
        let model = json.get("model").ok_or("tokenizer: no model")?;
        if model.get("merges").and_then(Json::as_array).is_none_or(|m| !m.is_empty()) {
            return Err("tokenizer: it has merges, so it is not a character tokeniser".into());
        }
        let vocab = model.get("vocab").and_then(Json::as_object).ok_or("tokenizer: no vocab")?;

        let mut chars = vec![None; vocab.len()];
        for (token, id) in vocab {
            let mut letters = token.chars();
            let (Some(c), None) = (letters.next(), letters.next()) else {
                return Err(format!("tokenizer: {token:?} is not a single character"));
            };
            let slot = id.as_usize().and_then(|id| chars.get_mut(id)).ok_or("tokenizer: an id is out of range")?;
            if slot.replace(c).is_some() {
                return Err("tokenizer: two characters share an id".into());
            }
        }
        // Every slot was filled: as many distinct ids as there are slots.
        let chars: Vec<char> = chars.into_iter().flatten().collect();
        if !chars.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err("tokenizer: ids are not in character order".into());
        }
        Ok(CharTokenizer { chars })
    }

    /// Write `tokenizer.json` into `dir`, beside the model it belongs to.
    pub fn save(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        write_whole(&dir.join(TOKENIZER_FILE), self.to_json().to_string().as_bytes())
    }

    pub fn load(dir: &Path) -> io::Result<Self> {
        let text = read_text(&dir.join(TOKENIZER_FILE))?;
        Json::parse(&text).and_then(|json| Self::from_json(&json)).map_err(invalid)
    }
}

/// A tokenised text, split into a part to learn from and a part to be tested on.
pub struct Corpus {
    pub train: Vec<usize>,
    pub val: Vec<usize>,
}

impl Corpus {
    /// The validation set is the *end* of the text, not a random sample of it.
    /// Windows overlap, so a random sample would share most of its characters
    /// with training windows on either side, and the validation loss would
    /// measure memory. A model that has merely memorised does well on the
    /// training split and badly here; the gap between the two is overfitting.
    pub fn new(mut tokens: Vec<usize>, val_fraction: f32) -> Self {
        let cut = ((tokens.len() as f32) * (1.0 - val_fraction)) as usize;
        let val = tokens.split_off(cut);
        Corpus { train: tokens, val }
    }
}

/// A random window of `context` tokens, and the same window one step later.
pub fn window<'a>(tokens: &'a [usize], context: usize, rng: &mut Rng) -> (&'a [usize], &'a [usize]) {
    assert!(tokens.len() > context, "{} tokens is too few for a context of {context}", tokens.len());
    let start = rng.below(tokens.len() - context);
    (&tokens[start..start + context], &tokens[start + 1..start + context + 1])
}

/// One step of training on `batch` random windows. Returns their mean loss.
///
/// The model takes one sequence at a time, so a batch is a loop. That works
/// because gradients *accumulate*: `backward` adds into the gradient buffers,
/// and nothing clears them until `zero_grad`. Averaging over several windows
/// before stepping gives a steadier direction than any one window would.
pub fn train_step(model: &mut Gpt, opt: &mut AdamW, tokens: &[usize], batch: usize, rng: &mut Rng) -> f32 {
    let context = model.config().context;
    let mut total = 0.0;

    model.zero_grad();
    for _ in 0..batch {
        let (input, target) = window(tokens, context, rng);

        let logits = model.forward(input); //                          predict
        let (loss, mut dlogits) = softmax_cross_entropy(&logits, target); // score
        // Each window's gradient is already a mean over its positions; divide
        // by the batch so the sum over windows is a mean as well.
        dlogits.data.iter_mut().for_each(|g| *g /= batch as f32);
        model.backward(&dlogits); //                                   blame

        total += loss;
    }
    opt.step(model.params()); //                                       adjust

    total / batch as f32
}

/// Copies of a model, one to a thread, that share the work of each batch.
///
/// # Data parallelism
///
/// The windows in a batch have nothing to do with each other until their
/// gradients are added up, and addition does not care who did the work. So:
///
/// 1. **Broadcast.** Every replica is given the current weights.
/// 2. **Compute.** Each takes its share of the windows and runs forward and
///    backward on them, accumulating gradients of its own, on a thread of its
///    own. Nothing is shared and nothing is locked.
/// 3. **All-reduce.** The replicas' gradients are added into the one model
///    that has an optimiser, and it takes the step.
///
/// That is the whole of how a model is trained on a thousand GPUs, with a
/// core standing in for a GPU and a `memcpy` for the network. It is also why
/// each replica is a full model rather than a view of shared weights: a layer
/// keeps what it saw on the way forward for use on the way back, so two
/// threads cannot be inside the same layer at once.
///
/// The step it takes is the step [`train_step`] would have taken — the same
/// windows, drawn in the same order from the same generator — except that the
/// gradients are summed in a different order, and floating-point addition
/// notices. Window `i` always goes to replica `i mod n` and the replicas are
/// always reduced in order, so a run is reproducible for a given number of
/// threads, and differs in the last bits between one number and another.
pub struct Replicas {
    models: Vec<Gpt>,
}

impl Replicas {
    pub fn new(model: &Gpt, threads: usize) -> Self {
        assert!(threads > 0, "training needs at least one thread");
        // Built with throwaway weights; every step begins by replacing them.
        let models = (0..threads).map(|_| Gpt::new(model.config(), &mut Rng::new(0))).collect();
        Replicas { models }
    }

    /// One step on `batch` random windows, split across the replicas.
    /// Returns their mean loss.
    pub fn train_step(&mut self, model: &mut Gpt, opt: &mut AdamW, tokens: &[usize], batch: usize, rng: &mut Rng) -> f32 {
        let context = model.config().context;
        let windows: Vec<_> = (0..batch).map(|_| window(tokens, context, rng)).collect();
        let n = self.models.len();

        for replica in self.models.iter_mut() {
            for (theirs, ours) in replica.params().into_iter().zip(model.params()) {
                theirs.value.copy_from_slice(ours.value); //           broadcast
            }
        }

        let total: f32 = std::thread::scope(|scope| {
            let windows = &windows;
            let running: Vec<_> = self
                .models
                .iter_mut()
                .enumerate()
                .map(|(r, replica)| {
                    scope.spawn(move || {
                        let mut total = 0.0;
                        replica.zero_grad();
                        for (input, target) in windows.iter().skip(r).step_by(n) {
                            let logits = replica.forward(input);
                            let (loss, mut dlogits) = softmax_cross_entropy(&logits, target);
                            // By the whole batch, not by this replica's share
                            // of it: the mean is over every window there is.
                            dlogits.data.iter_mut().for_each(|g| *g /= batch as f32);
                            replica.backward(&dlogits); //               compute
                            total += loss;
                        }
                        total
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
}

/// Mean loss over `windows` random windows, with no learning.
pub fn evaluate(model: &mut Gpt, tokens: &[usize], windows: usize, rng: &mut Rng) -> f32 {
    let context = model.config().context;
    let mut total = 0.0;
    for _ in 0..windows {
        let (input, target) = window(tokens, context, rng);
        total += softmax_cross_entropy(&model.forward(input), target).0;
    }
    total / windows as f32
}

// ---------------------------------------------------------------------------
// The training loop
// ---------------------------------------------------------------------------

/// One idea, repeated: draw a batch, take a step, and every so often ask the
/// validation split how it is going.
///
/// The loop lives here rather than in a `main` because there are two front
/// ends to it — `train_text` in this crate and `kvad train` in the engine —
/// and a training loop copied into two places is two training loops.
///
/// # Keeping the best model, not the last
///
/// A run on a small text overfits in plain sight: validation loss bottoms out
/// and then climbs while training loss keeps falling. Saving at the end saves
/// the memoriser. So the model is written every time the validation loss
/// improves on the best this run has seen, and a run that gets worse leaves
/// the good model where it was. That is early stopping, done by keeping the
/// best rather than by halting: the run still finishes, and still prints what
/// happened, so the overfitting stays visible instead of being hidden by a
/// loop that quietly gave up.
///
/// For that comparison to mean anything the two losses must be measured on
/// the same windows. They are: every checkpoint evaluates on windows drawn
/// from a generator of its own, seeded the same way each time. A side effect
/// is that the training run no longer depends on how often it is evaluated,
/// because evaluation no longer takes draws from the training generator.
///
/// The bar starts at infinity, so the first checkpoint always writes. With
/// `--from` that matters: a model continuing on different text cannot be
/// compared with its old loss on its old text, and a run that saved nothing
/// because it never beat a number from another corpus would be a run that
/// silently did nothing.
#[derive(Clone)]
pub struct Training {
    pub steps: usize,
    pub batch: usize,
    pub lr: f32,
    /// Steps between validation checkpoints.
    pub eval_every: usize,
    /// Windows drawn at each checkpoint. Always the same ones.
    pub eval_windows: usize,
    /// Steps spent bringing the learning rate up from nothing to `lr`, and
    /// the fraction of `lr` left at the last step. See [`Schedule`], which
    /// says what each is for and what each measured; `Some(0)` with
    /// `decay_to: 1.0` holds `lr` flat, which is what every run here did
    /// before this existed.
    ///
    /// `None` is "you choose": a tenth of the run. A warm-up is a share of
    /// the run rather than a number of steps — ten steps of warm-up in a run
    /// of 750 measured worse than none at all — and only the run knows how
    /// long it is.
    pub warmup: Option<usize>,
    pub decay_to: f32,
    /// The longest the whole gradient may be before it is scaled down.
    pub clip: Option<f32>,
    /// Replicas to split each batch across; see [`Replicas`]. Clamped to the
    /// batch size, because a batch cannot be split finer than one window.
    pub threads: usize,
    /// Where to write the model whenever validation loss improves. `None`
    /// trains and keeps nothing.
    pub save: Option<std::path::PathBuf>,
    /// Whether `model` arrives already trained, as it does for a `kvad train
    /// --from` continuation.
    ///
    /// It decides what the first checkpoint has to beat, and the two answers
    /// are not the same promise. A new model has nothing worth keeping, so
    /// the bar starts at infinity and the first checkpoint always writes. A
    /// continuation *is* the model in `save` — so starting its bar at
    /// infinity means the first checkpoint overwrites it whatever it scored,
    /// and a run that helped nothing left you worse off than before it.
    ///
    /// True measures the model that arrived and makes its own loss the bar,
    /// which costs one validation pass and makes "keeps the best model" true
    /// across a continuation rather than only within a run.
    pub already_trained: bool,
    /// Set from another thread to end the run early. `None` is a run that
    /// cannot be stopped, which is what a command line wants: Ctrl-C.
    ///
    /// Read once a step, not once a checkpoint. Checkpoints are hundreds of
    /// steps apart by default, and a stop button that takes a minute to
    /// answer is a stop button nobody believes. A stopped run leaves the best
    /// model so far on disk, because `save` writes at every improvement
    /// rather than at the end.
    pub stop: Option<Arc<AtomicBool>>,
}

impl Default for Training {
    fn default() -> Self {
        Training {
            steps: 2000,
            batch: 16,
            lr: 3e-3,
            warmup: None,
            decay_to: 0.1,
            clip: Some(1.0),
            eval_every: 250,
            eval_windows: 50,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            save: None,
            already_trained: false,
            stop: None,
        }
    }
}

impl Training {
    /// The rate schedule this configuration asks for. A warm-up left unset is
    /// a tenth of the run, and one longer than the run is the whole of it.
    pub fn schedule(&self) -> Schedule {
        Schedule {
            peak: self.lr,
            warmup: self.warmup.unwrap_or(self.steps / 10).min(self.steps),
            decay_to: self.decay_to,
        }
    }
}

/// What a run says about itself while it runs.
pub enum Report<'a> {
    /// What the model handed to the run already scores, sent once and before
    /// the first step. Only for a continuation — see
    /// [`Training::already_trained`] — where it is the number every
    /// checkpoint is then measured against.
    Baseline { val_loss: f32 },
    /// How fast this machine is turning out to be, and what the rest of the
    /// run will cost at that rate. Sent once, early — the point of it is to
    /// arrive while there is still something to be done about the answer.
    Pace { chars_per_sec: f32, remaining_secs: f32 },
    /// A validation checkpoint. `model` is handed over so that the caller can
    /// write a sample from it, which is the part anyone actually watches.
    Step {
        step: usize,
        train_loss: f32,
        val_loss: f32,
        /// The best validation loss so far, and so written to disk if there
        /// is anywhere to write it.
        best: bool,
        saved: bool,
        elapsed_secs: f32,
        chars_per_sec: f32,
        model: &'a mut Gpt,
    },
}

/// What a finished run leaves behind.
#[derive(Debug)]
pub struct Trained {
    /// The loss of the model that is now on disk. For a continuation that
    /// improved nothing this is the loss of the model that arrived, which is
    /// still the one there.
    pub best_val: f32,
    /// The step whose model was written, or 0 for "the one that arrived, and
    /// nothing was written". See [`Trained::improved`].
    pub best_step: usize,
    /// The lowest validation loss this run measured, whether or not it beat
    /// the model the run started from. The same as `best_val` unless nothing
    /// did — and the number to report when nothing did, because "the best
    /// this run reached" is the question being asked at that point.
    pub reached: f32,
    pub last_val: f32,
    pub elapsed_secs: f32,
    /// True if [`Training::stop`] was raised and the run ended before its
    /// last step. The numbers above are still the truth about what happened;
    /// they are just about a shorter run than the one that was asked for.
    pub stopped: bool,
}

impl Trained {
    /// Whether any checkpoint beat the model the run started from.
    ///
    /// False only for a continuation that helped nothing: nothing was
    /// written, and the model in `save` is the one that arrived.
    pub fn improved(&self) -> bool {
        self.best_step > 0
    }
}

/// Steps to time before reporting a pace. Ten is enough for an estimate good
/// to a few per cent, and on the slowest model here costs about ten seconds.
const PACE_AFTER: usize = 10;

/// The windows every checkpoint is measured on. Any fixed number would do;
/// what matters is that it is the same one every time.
const EVAL_SEED: u64 = 20_260_919;

/// Train `model` on `corpus`, reporting as it goes and keeping the best.
pub fn train(
    model: &mut Gpt,
    tok: &CharTokenizer,
    corpus: &Corpus,
    cfg: &Training,
    rng: &mut Rng,
    report: &mut dyn FnMut(Report),
) -> io::Result<Trained> {
    let context = model.config().context;
    let batch = cfg.batch.max(1);
    let eval_every = cfg.eval_every.max(1);

    // A window is `context + 1` tokens, and both splits are drawn from. Say
    // so here, where the numbers are, rather than let `window` assert its way
    // out of a thread halfway through the first step.
    for (which, tokens) in [("training", &corpus.train), ("validation", &corpus.val)] {
        if tokens.len() <= context {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "too little text: the {which} split is {} characters and a context of {context} needs \
                     more than that. The validation split is the last tenth, so about {} characters in all.",
                    tokens.len(),
                    10 * (context + 1)
                ),
            ));
        }
    }
    if cfg.steps == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "a run of no steps would train nothing"));
    }
    let mut opt = AdamW::new(cfg.lr);
    opt.clip = cfg.clip;
    let schedule = cfg.schedule();
    let mut replicas = (cfg.threads.min(batch) > 1).then(|| Replicas::new(model, cfg.threads.min(batch)));

    // Characters seen after `steps` steps: every window is `context` of them,
    // and there are `batch` windows to a step.
    let chars = |steps: usize| (steps * batch * context) as f32;
    // What a checkpoint has to beat, measured before the clock starts so that
    // a continuation's extra validation pass is not charged to the pace
    // estimate. Step 0 is the model that arrived; see `Training::already_trained`.
    let (mut best_val, mut best_step) = match cfg.already_trained {
        true => (evaluate(model, &corpus.val, cfg.eval_windows, &mut Rng::new(EVAL_SEED)), 0),
        false => (f32::INFINITY, 0),
    };
    if cfg.already_trained {
        report(Report::Baseline { val_loss: best_val });
    }

    let started = std::time::Instant::now();
    let (mut running, mut since) = (0.0, 0);
    let mut reached = f32::INFINITY;
    let mut last_val = f32::INFINITY;

    let stopped = |cfg: &Training| cfg.stop.as_deref().is_some_and(|s| s.load(Ordering::Relaxed));
    let mut ended_early = false;

    for step in 1..=cfg.steps {
        // Asked before the step rather than after it, so that a stop raised
        // during step N is answered by not starting step N+1.
        if stopped(cfg) {
            ended_early = true;
            break;
        }

        // The rate is the loop's business, not the optimiser's: Adam decides
        // which way each weight goes, the schedule decides how far.
        opt.lr = schedule.at(step, cfg.steps);
        running += match &mut replicas {
            Some(replicas) => replicas.train_step(model, &mut opt, &corpus.train, batch, rng),
            None => train_step(model, &mut opt, &corpus.train, batch, rng),
        };
        since += 1;

        if step == PACE_AFTER.min(cfg.steps) {
            let rate = chars(step) / started.elapsed().as_secs_f32();
            report(Report::Pace { chars_per_sec: rate, remaining_secs: chars(cfg.steps - step) / rate });
        }

        if step % eval_every == 0 || step == cfg.steps {
            let val = evaluate(model, &corpus.val, cfg.eval_windows, &mut Rng::new(EVAL_SEED));
            reached = reached.min(val);
            let best = val < best_val;
            if best {
                (best_val, best_step) = (val, step);
            }
            let saved = match (&cfg.save, best) {
                (Some(dir), true) => {
                    save(dir, model, tok)?;
                    true
                }
                _ => false,
            };
            let elapsed = started.elapsed().as_secs_f32();
            report(Report::Step {
                step,
                train_loss: running / since as f32,
                val_loss: val,
                best,
                saved,
                elapsed_secs: elapsed,
                chars_per_sec: chars(step) / elapsed,
                model,
            });
            (running, since, last_val) = (0.0, 0, val);
        }
    }

    Ok(Trained {
        best_val,
        best_step,
        reached,
        last_val,
        elapsed_secs: started.elapsed().as_secs_f32(),
        stopped: ended_early,
    })
}

/// A number of seconds as something to read: "45s", "6m 20s", "1h 12m".
///
/// Training runs span four orders of magnitude here — a test finishes in a
/// second, the largest preset takes half an hour — and "1832s" makes nobody
/// any the wiser.
pub fn human_secs(secs: f32) -> String {
    let secs = secs.max(0.0).round() as u64;
    match (secs / 3600, (secs % 3600) / 60, secs % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m {s:02}s"),
        (h, m, _) => format!("{h}h {m:02}m"),
    }
}

/// A model and the tokeniser it was trained with, written into one directory.
///
/// They travel together because neither is any use alone: id 17 means
/// whatever character was seventeenth in *that* text. The tokeniser is
/// rewritten on every save, unchanged, rather than once at the start, so that
/// the directory is a complete model from the first write and after an
/// interrupted run — a few kilobytes against a directory nothing can load.
pub fn save(dir: &Path, model: &mut Gpt, tok: &CharTokenizer) -> io::Result<()> {
    crate::checkpoint::save(dir, model)?;
    tok.save(dir)
}

/// Continue `prompt` by `count` tokens, one at a time.
///
/// This is all generation is: predict a distribution over the next token,
/// draw from it, append the draw, and ask again. The model only ever sees the
/// last `context` tokens, so the window slides.
///
/// It is also wasteful, and instructively so. Every new token re-runs the
/// whole window through every layer, recomputing keys and values for
/// positions whose keys and values cannot have changed. Caching them — the KV
/// cache — is the first thing an inference engine does, and the rest of Kvad
/// is about what comes after that.
pub fn generate(model: &mut Gpt, prompt: &[usize], count: usize, temperature: f32, rng: &mut Rng) -> Vec<usize> {
    assert!(!prompt.is_empty(), "generation needs at least one token to start from");
    let context = model.config().context;
    let mut ids = prompt.to_vec();
    for _ in 0..count {
        let seen = &ids[ids.len().saturating_sub(context)..];
        let logits = model.forward(seen);
        // Only the last row is about a token we do not have yet.
        ids.push(sample(logits.row(logits.rows - 1), temperature, rng));
    }
    ids
}

/// Draw one token from `logits`.
///
/// Temperature divides the logits before the softmax. Below 1 it sharpens the
/// distribution towards the favourite, above 1 it flattens it towards chance,
/// and at 0 it is no longer a draw: take the most likely token, always.
pub fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    if temperature <= 0.0 {
        return (0..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits.iter().map(|&l| ((l - max) / temperature).exp()).collect();
    let total: f32 = weights.iter().sum();

    // Walk the probabilities until a uniform draw has been used up.
    let mut remaining = rng.uniform() * total;
    for (i, w) in weights.iter().enumerate() {
        remaining -= w;
        if remaining < 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// The loss of guessing the next token from how often each token appears, and
/// nothing else. A model that beats this has learned something about order.
pub fn unigram_loss(tokens: &[usize], vocab: usize) -> f32 {
    let mut counts = vec![0usize; vocab];
    tokens.iter().for_each(|&t| counts[t] += 1);
    let n = tokens.len() as f32;
    counts.iter().filter(|&&c| c > 0).map(|&c| -(c as f32 / n) * (c as f32 / n).ln()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GptConfig;

    #[test]
    fn text_survives_a_round_trip() {
        let text = "To be, or not to be: that is the question.\n";
        let tok = CharTokenizer::from_text(text);
        assert_eq!(tok.decode(&tok.encode(text).unwrap()), text);
        assert_eq!(tok.encode("zebra"), Err('z'));

        // Repeats share an id; 'l' and 'L' do not.
        let tok = CharTokenizer::from_text("hello, HELLO");
        assert_eq!(tok.vocab(), 4 + 4 + 2);
        assert_eq!(tok.encode("lol").unwrap(), tok.encode("lol").unwrap());
        assert_ne!(tok.encode("l"), tok.encode("L"));
    }

    /// The characters a file format is most likely to mangle, because the
    /// vocabulary is where they have to survive as JSON keys.
    #[test]
    fn a_saved_tokeniser_gives_every_character_its_old_id() {
        let text = "tab\t newline\n \"quotes\" back\\slash \u{1} é → 😀";
        let tok = CharTokenizer::from_text(text);
        let dir = std::env::temp_dir().join(format!("nanograd-tokeniser-{}", std::process::id()));
        tok.save(&dir).unwrap();
        let back = CharTokenizer::load(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(back.chars, tok.chars);
        assert_eq!(back.encode(text), tok.encode(text));
    }

    #[test]
    fn a_vocabulary_that_is_not_ours_is_refused() {
        let with = |vocab: &str, merges: &str| {
            let json = format!(r#"{{"model":{{"type":"BPE","vocab":{vocab},"merges":{merges}}}}}"#);
            CharTokenizer::from_json(&Json::parse(&json).unwrap()).map(|t| t.chars)
        };
        assert_eq!(with(r#"{"a":0,"b":1}"#, "[]"), Ok(vec!['a', 'b']));
        assert!(with(r#"{"a":0,"b":1}"#, r#"["a b"]"#).unwrap_err().contains("merges"));
        assert!(with(r#"{"a":0,"th":1}"#, "[]").unwrap_err().contains("single character"));
        assert!(with(r#"{"a":0,"b":2}"#, "[]").unwrap_err().contains("out of range"));
        assert!(with(r#"{"a":0,"b":0}"#, "[]").unwrap_err().contains("share an id"));
        // Our `encode` is a binary search, so the order is not a detail.
        assert!(with(r#"{"b":0,"a":1}"#, "[]").unwrap_err().contains("character order"));
    }

    #[test]
    fn a_target_is_its_input_one_step_later() {
        let tokens: Vec<usize> = (0..50).collect();
        let mut rng = Rng::new(71);
        for _ in 0..200 {
            let (input, target) = window(&tokens, 8, &mut rng);
            assert_eq!(input.len(), 8);
            assert_eq!(&input[1..], &target[..7]);
            assert_eq!(target[7], input[7] + 1);
        }
    }

    /// Both ends of the text must be reachable, or some of it is never trained on.
    #[test]
    fn windows_cover_the_whole_text() {
        let tokens: Vec<usize> = (0..20).collect();
        let mut rng = Rng::new(72);
        let (mut first, mut last) = (false, false);
        for _ in 0..500 {
            let (input, target) = window(&tokens, 8, &mut rng);
            first |= input[0] == 0;
            last |= target[7] == 19;
        }
        assert!(first && last, "reached the first token: {first}, the last: {last}");
    }

    #[test]
    fn validation_is_the_end_of_the_text_and_nothing_else() {
        let corpus = Corpus::new((0..100).collect(), 0.1);
        assert_eq!(corpus.train, (0..90).collect::<Vec<_>>());
        assert_eq!(corpus.val, (90..100).collect::<Vec<_>>());
    }

    #[test]
    fn sampling_follows_the_distribution() {
        // Probabilities 0.1, 0.2, 0.7.
        let logits = [0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln()];
        let mut rng = Rng::new(73);

        let mut counts = [0usize; 3];
        for _ in 0..20_000 {
            counts[sample(&logits, 1.0, &mut rng)] += 1;
        }
        for (count, expected) in counts.iter().zip([0.1, 0.2, 0.7]) {
            let got = *count as f32 / 20_000.0;
            assert!((got - expected).abs() < 0.01, "wanted {expected}, drew {got}");
        }

        // Temperature 0 is the favourite, every time.
        assert!((0..100).all(|_| sample(&logits, 0.0, &mut rng) == 2));

        // A low temperature sharpens: p^(1/T), renormalised. At T = 0.5 that
        // is 0.01, 0.04, 0.49 over 0.54.
        let mut counts = [0usize; 3];
        for _ in 0..20_000 {
            counts[sample(&logits, 0.5, &mut rng)] += 1;
        }
        let got = counts[2] as f32 / 20_000.0;
        assert!((got - 0.49 / 0.54).abs() < 0.01, "at T = 0.5 the favourite was drawn {got}");
    }

    #[test]
    fn the_unigram_baseline_is_the_entropy_of_the_counts() {
        // Four tokens, equally common: two bits, which is ln(4) nats.
        assert!((unigram_loss(&[0, 1, 2, 3, 0, 1, 2, 3], 4) - 4f32.ln()).abs() < 1e-6);
        // One token only: nothing to be unsure of.
        assert_eq!(unigram_loss(&[2, 2, 2], 4), 0.0);
    }

    /// A batch's gradient must be the *mean* of its windows' gradients, and
    /// must not include the batch before. No learning test can see either
    /// mistake: Adam cancels the size of the gradient, so a sum trains exactly
    /// like a mean, and stale gradients just look like extra momentum.
    ///
    /// So check the numbers. With exactly `context + 1` tokens there is only
    /// one window to draw, and a batch of three of it has to give the same
    /// gradient as a batch of one — twice running.
    #[test]
    fn a_batch_gradient_is_a_mean_and_starts_from_zero() {
        let tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5];
        let config = GptConfig { vocab: 10, context: 8, d_model: 8, n_heads: 2, n_layers: 1 };
        let mut rng = Rng::new(75);
        let mut model = Gpt::new(config, &mut rng);
        // A learning rate of zero: all of the gradient, none of the movement.
        let mut frozen = AdamW::new(0.0);
        frozen.weight_decay = 0.0;
        let grads = |model: &mut Gpt| -> Vec<f32> { model.params().iter().flat_map(|p| p.grad.to_vec()).collect() };

        train_step(&mut model, &mut frozen, &tokens, 1, &mut rng);
        let one = grads(&mut model);
        for _ in 0..2 {
            train_step(&mut model, &mut frozen, &tokens, 3, &mut rng);
            let three = grads(&mut model);
            let worst = worst_gap(&one, &three);
            assert!(worst < 1e-6, "a batch of three identical windows differs from one by {worst:e}");
        }
    }

    /// The largest difference between two lists of numbers. `f32::max` prefers
    /// anything to a NaN, so a fold over it reports two lists of NaN as
    /// identical; this reports them as infinitely far apart.
    fn worst_gap(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        if a.iter().chain(b).any(|v| !v.is_finite()) {
            return f32::INFINITY;
        }
        a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
    }

    /// A text that can be learned — it counts to seven, over and over — with
    /// an occasional wrong note, so that no two windows are quite alike.
    fn some_tokens(count: usize, vocab: usize) -> Vec<usize> {
        let mut rng = Rng::new(31);
        (0..count).map(|i| if rng.below(8) == 0 { rng.below(vocab) } else { i % 7 }).collect()
    }

    const SMALL: GptConfig = GptConfig { vocab: 10, context: 8, d_model: 8, n_heads: 2, n_layers: 2 };

    /// Splitting a batch across threads must not change what is computed,
    /// only who computes it. Seven windows do not divide evenly among two,
    /// three or four replicas, and ten replicas leave three with nothing to do.
    #[test]
    fn replicas_compute_the_gradient_one_model_would() {
        let tokens = some_tokens(200, SMALL.vocab);
        let grads = |model: &mut Gpt| -> Vec<f32> { model.params().iter().flat_map(|p| p.grad.to_vec()).collect() };
        let frozen = || {
            let mut opt = AdamW::new(0.0);
            opt.weight_decay = 0.0;
            opt
        };

        for threads in [1, 2, 3, 4, 10] {
            let (mut alone, mut shared) = (Gpt::new(SMALL, &mut Rng::new(5)), Gpt::new(SMALL, &mut Rng::new(5)));
            let (mut rng_a, mut rng_b) = (Rng::new(9), Rng::new(9));
            let (mut opt_a, mut opt_b) = (frozen(), frozen());
            let mut replicas = Replicas::new(&shared, threads);

            // Twice, so that a gradient left over from the first step, in the
            // model or in any replica, would show up in the second.
            for step in 0..2 {
                let loss_a = train_step(&mut alone, &mut opt_a, &tokens, 7, &mut rng_a);
                let loss_b = replicas.train_step(&mut shared, &mut opt_b, &tokens, 7, &mut rng_b);
                let (a, b) = (grads(&mut alone), grads(&mut shared));

                let size = a.iter().fold(0.0f32, |m, g| m.max(g.abs()));
                let gap = worst_gap(&a, &b) / size;
                assert!(gap < 1e-5, "{threads} threads, step {step}: gradients differ by {gap:e} of their size");
                assert!((loss_a - loss_b).abs() < 1e-5, "{threads} threads: loss {loss_a} against {loss_b}");
                // One replica adds the windows up in the same order, so there
                // is not even rounding to tell them apart.
                if threads == 1 {
                    assert_eq!(a, b);
                }
            }
        }
    }

    /// With a learning rate of zero the weights never move, and a replica
    /// still holding the weights it was built with would go unnoticed. So
    /// train for real, and require the loss to follow the same path step by
    /// step — which it only can if every replica sees every update.
    #[test]
    fn replicas_are_given_the_new_weights_every_step() {
        let tokens = some_tokens(200, SMALL.vocab);
        let (mut alone, mut shared) = (Gpt::new(SMALL, &mut Rng::new(5)), Gpt::new(SMALL, &mut Rng::new(5)));
        let (mut rng_a, mut rng_b) = (Rng::new(9), Rng::new(9));
        let (mut opt_a, mut opt_b) = (AdamW::new(0.01), AdamW::new(0.01));
        let mut replicas = Replicas::new(&shared, 3);

        let mut first = 0.0;
        for step in 0..30 {
            let loss_a = train_step(&mut alone, &mut opt_a, &tokens, 6, &mut rng_a);
            let loss_b = replicas.train_step(&mut shared, &mut opt_b, &tokens, 6, &mut rng_b);
            // Measured: they never part by more than 5e-7.
            assert!((loss_a - loss_b).abs() < 1e-5, "step {step}: loss {loss_a} alone, {loss_b} shared");
            if step == 0 {
                first = loss_a;
            }
        }
        let last = train_step(&mut alone, &mut opt_a, &tokens, 6, &mut rng_a);
        assert!(last < first - 0.3, "nothing was learned ({first} to {last}), so nothing was tested");
    }

    /// Everything in the crate at once: tokeniser, windows, model, AdamW and
    /// sampling, on a text with exactly one thing to learn.
    #[test]
    fn it_learns_a_pattern_and_continues_it() {
        let text = "abcd".repeat(200);
        let tok = CharTokenizer::from_text(&text);
        let corpus = Corpus::new(tok.encode(&text).unwrap(), 0.1);

        let mut rng = Rng::new(74);
        let config = GptConfig { vocab: tok.vocab(), context: 8, d_model: 16, n_heads: 2, n_layers: 1 };
        let mut model = Gpt::new(config, &mut rng);
        let mut opt = AdamW::new(1e-2);

        let before = evaluate(&mut model, &corpus.val, 20, &mut rng);
        for _ in 0..STEPS {
            train_step(&mut model, &mut opt, &corpus.train, 4, &mut rng);
        }
        let after = evaluate(&mut model, &corpus.val, 20, &mut rng);

        // Even the first row of a window has one character to go on, and in
        // this text one character settles what comes next.
        assert!(after < 0.1, "validation loss went from {before} to only {after}");

        // Longer than the context, so the window has to slide.
        let out = generate(&mut model, &tok.encode("ab").unwrap(), 30, 0.0, &mut rng);
        assert_eq!(tok.decode(&out), "abcd".repeat(8));
    }

    const STEPS: usize = 150;

    // -----------------------------------------------------------------------
    // The training loop
    // -----------------------------------------------------------------------

    #[test]
    fn a_duration_is_written_the_way_a_person_would_say_it() {
        assert_eq!(human_secs(0.0), "0s");
        assert_eq!(human_secs(45.4), "45s");
        assert_eq!(human_secs(59.6), "1m 00s");
        assert_eq!(human_secs(380.0), "6m 20s");
        assert_eq!(human_secs(4320.0), "1h 12m");
        // Never a negative duration, whatever a clock says.
        assert_eq!(human_secs(-3.0), "0s");
    }

    /// A text that contradicts its own validation split: the same three
    /// letters, in the opposite order. Learning the training half can only
    /// make the validation half worse, which is overfitting with the
    /// gradualness taken out — a run on this text gets worse from its first
    /// checkpoint, reliably, in a second.
    ///
    /// The two halves are built by hand rather than with [`Corpus::new`], so
    /// that not one window of the one is in the other.
    /// A raised [`Training::stop`] ends the run at the next step, not at the
    /// next checkpoint, and says so in the result.
    ///
    /// Asserted by counting checkpoints rather than by timing: a run of 400
    /// steps evaluating every 5 would report 80 times, and stopping from
    /// inside the first report has to leave it at exactly one. The model on
    /// disk is the one that first report saved, so a stopped run is still a
    /// run you can use.
    #[test]
    fn a_raised_stop_ends_the_run_at_the_next_step() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut model = Gpt::new(config, &mut Rng::new(4));
        let dir = scratch("stopped");

        let stop = Arc::new(AtomicBool::new(false));
        let cfg = Training {
            steps: 400,
            batch: 2,
            lr: 1e-2,
            eval_every: 5,
            eval_windows: 5,
            threads: 1,
            save: Some(dir.clone()),
            stop: Some(Arc::clone(&stop)),
            ..Training::default()
        };

        let mut checkpoints = 0;
        let done = train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| {
            if let Report::Step { .. } = report {
                checkpoints += 1;
                stop.store(true, Ordering::Relaxed);
            }
        })
        .unwrap();

        assert_eq!(checkpoints, 1, "the run kept going past the step the stop was raised on");
        assert!(done.stopped, "a run that was stopped reported itself as finished");
        assert_eq!(done.best_step, 5);
        assert!(dir.join(crate::checkpoint::WEIGHTS_FILE).is_file(), "a stopped run left no model behind");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stop that is already raised stops before the first step, and the
    /// result is a run that trained nothing rather than an error.
    #[test]
    fn a_stop_raised_before_the_start_trains_nothing() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut model = Gpt::new(config, &mut Rng::new(4));
        let cfg = Training {
            steps: 400,
            batch: 2,
            lr: 1e-2,
            eval_every: 5,
            eval_windows: 5,
            threads: 1,
            save: None,
            stop: Some(Arc::new(AtomicBool::new(true))),
            ..Training::default()
        };

        let mut reports = 0;
        let done =
            train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |_| reports += 1).unwrap();
        assert_eq!(reports, 0);
        assert!(done.stopped);
        assert_eq!(done.best_step, 0);
        assert!(done.best_val.is_infinite(), "a run of no steps measured a loss");
    }

    fn a_text_and_its_contradiction() -> (CharTokenizer, Corpus) {
        let tok = CharTokenizer::from_text("abc");
        let corpus =
            Corpus { train: tok.encode(&"abc".repeat(200)).unwrap(), val: tok.encode(&"acb".repeat(40)).unwrap() };
        (tok, corpus)
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nanograd-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    const TINY: GptConfig = GptConfig { vocab: 7, context: 16, d_model: 32, n_heads: 4, n_layers: 2 };

    /// The directory must hold the model from the step with the lowest
    /// validation loss, and not the one the run happened to stop on.
    ///
    /// Checking the loss alone would not do it: two steps can score the same
    /// to three decimals and be different models. So snapshot the weights at
    /// every checkpoint, and require the file to be the snapshot from the
    /// step the run said was best, float for float.
    #[test]
    fn the_saved_model_is_the_best_one_and_not_the_last() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut model = Gpt::new(config, &mut Rng::new(4));
        let dir = scratch("best");

        let cfg = Training {
            steps: 240,
            batch: 4,
            lr: 1e-2,
            eval_every: 40,
            eval_windows: 20,
            threads: 1,
            save: Some(dir.clone()),
            ..Training::default()
        };
        let mut seen: Vec<(usize, f32, Vec<f32>)> = Vec::new();
        let done = train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| {
            if let Report::Step { step, val_loss, model, .. } = report {
                seen.push((step, val_loss, model.params().iter().flat_map(|p| p.value.to_vec()).collect()));
            }
        })
        .unwrap();

        // The run has to have got worse, or there is nothing here to test.
        assert_eq!(seen.len(), 6);
        assert!(done.best_step < cfg.steps, "validation never got worse: {seen:?}", seen = seen.iter().map(|s| s.1).collect::<Vec<_>>());
        assert!(done.last_val > done.best_val, "last {} against best {}", done.last_val, done.best_val);
        assert_eq!(done.best_val, seen.iter().map(|s| s.1).fold(f32::INFINITY, f32::min));

        let best = seen.iter().find(|s| s.0 == done.best_step).expect("the best step was reported");
        let mut back = crate::checkpoint::load(&dir).unwrap();
        let on_disk: Vec<f32> = back.params().iter().flat_map(|p| p.value.to_vec()).collect();
        assert_eq!(on_disk, best.2, "the directory holds some step other than {}", done.best_step);

        // And its tokeniser, so that the directory is a model and not half of one.
        assert_eq!(CharTokenizer::load(&dir).unwrap().chars, tok.chars);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A continuation that helps nothing must change nothing.
    ///
    /// "Keeps the best model, not the last" was true within a run and false
    /// across one. The bar started at infinity on every run, including a
    /// `--from` continuation — and a continuation *is* the model in `save`,
    /// so its first checkpoint always wrote, whatever it scored. A run that
    /// only made the model worse replaced a good model with a worse one, and
    /// the good one was gone.
    ///
    /// The same text as the test above, for the same reason: on a corpus
    /// that contradicts its own validation split, training can only make
    /// validation worse, so every checkpoint of the second run is worse than
    /// the model it started from. Nothing may be written, and the bytes on
    /// disk say so better than any loss does.
    #[test]
    fn a_continuation_that_helps_nothing_writes_nothing() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut fresh = Gpt::new(config, &mut Rng::new(4));
        let dir = scratch("continued");

        let cfg = Training {
            steps: 240,
            batch: 4,
            lr: 1e-2,
            eval_every: 40,
            eval_windows: 20,
            threads: 1,
            save: Some(dir.clone()),
            ..Training::default()
        };

        // A first run: a new model, so the bar starts at infinity and the
        // first checkpoint writes. That is right, and stays right.
        let first = train(&mut fresh, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |_| {}).unwrap();
        assert!(first.improved() && first.best_step > 0, "the first run wrote nothing");
        assert_eq!(first.best_val, first.reached, "with no model to beat, the best is the best");
        let kept = std::fs::read(dir.join("model.safetensors")).unwrap();

        // Now continue what is on disk — the best step, not the step the
        // first run ended on.
        let mut continued = crate::checkpoint::load(&dir).unwrap();
        let again = Training { already_trained: true, ..cfg.clone() };
        let mut baseline = None;
        let second = train(&mut continued, &tok, &corpus, &again, &mut Rng::new(6), &mut |report| {
            if let Report::Baseline { val_loss } = report {
                baseline = Some(val_loss);
            }
        })
        .unwrap();

        // The bar is the model that arrived, and it is the same number the
        // first run reported for it: saving and loading lost nothing, and
        // the measurement is the same measurement.
        assert_eq!(baseline, Some(second.best_val), "the bar was not the model that arrived");
        assert_eq!(baseline, Some(first.best_val), "a saved model scores differently when loaded");

        // Nothing beat it, so nothing was written and step 0 says so.
        assert!(!second.improved(), "step {} was written over a better model", second.best_step);
        assert_eq!(second.best_step, 0);
        assert!(
            second.reached > second.best_val,
            "this text was supposed to make it worse: reached {} against the bar {}",
            second.reached,
            second.best_val
        );
        assert_eq!(
            std::fs::read(dir.join("model.safetensors")).unwrap(),
            kept,
            "the model on disk was replaced by a worse one"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Evaluating draws windows from a generator of its own, so asking more
    /// often cannot change the answer. Before that it shared the training
    /// generator, and `--eval-every 100` and `--eval-every 250` trained two
    /// different models.
    #[test]
    fn how_often_a_run_is_evaluated_does_not_change_what_it_learns() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let run = |eval_every: usize| -> (Vec<f32>, f32) {
            let mut model = Gpt::new(config, &mut Rng::new(4));
            let cfg = Training { steps: 60, batch: 4, lr: 1e-2, eval_every, eval_windows: 20, threads: 1, save: None, ..Training::default() };
            let mut last = 0.0;
            train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| {
                if let Report::Step { val_loss, .. } = report {
                    last = val_loss;
                }
            })
            .unwrap();
            (model.params().iter().flat_map(|p| p.value.to_vec()).collect(), last)
        };

        let (often, val_often) = run(10);
        let (seldom, val_seldom) = run(30);
        assert_eq!(often, seldom, "how often it was evaluated changed the weights");
        // ...and the same windows every time, so the last reading agrees too.
        assert_eq!(val_often, val_seldom);
    }

    /// Threads are a matter of who does the arithmetic, not of what it is,
    /// and `train` has to keep it that way — the loop that chooses between
    /// `Replicas` and `train_step` now lives here rather than in a `main`.
    ///
    /// The losses, not the weights. Replicas sum the same gradients in a
    /// different order, so they differ in the last bits, and Adam divides by
    /// the size of the gradient: where a gradient is near zero, a difference
    /// of 1e-7 in it can still be a step of a whole learning rate. Measured,
    /// individual weights part by up to 5e-3 after 40 steps, while every loss
    /// reported here stays within 2.6e-5 — which is why the threshold is on
    /// the loss and is 1e-4.
    #[test]
    fn a_run_learns_the_same_thing_however_many_threads_it_uses() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let run = |threads: usize| -> Vec<f32> {
            let mut model = Gpt::new(config, &mut Rng::new(4));
            let cfg = Training { steps: 60, batch: 4, lr: 1e-2, eval_every: 20, eval_windows: 10, threads, save: None, ..Training::default() };
            let mut losses = Vec::new();
            train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| {
                if let Report::Step { train_loss, val_loss, .. } = report {
                    losses.extend([train_loss, val_loss]);
                }
            })
            .unwrap();
            losses
        };

        let one = run(1);
        assert_eq!(one.len(), 6, "three checkpoints, two losses each");
        // Four windows to a batch, so eight threads is four with four idle.
        for threads in [2, 3, 4, 8] {
            let many = run(threads);
            let worst = worst_gap(&one, &many);
            assert!(worst < 1e-4, "{threads} threads changed a loss by {worst:e}");
        }
    }

    /// The training loss a checkpoint prints is the mean since the last one,
    /// including a final interval that is shorter than the rest.
    ///
    /// Dividing by `eval_every` instead of by the number of steps actually
    /// taken is the mistake, and it is invisible whenever the steps divide
    /// evenly — which the defaults do. So: the same run reported both ways.
    #[test]
    fn a_reported_training_loss_is_the_mean_since_the_last_checkpoint() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let run = |eval_every: usize| -> Vec<f32> {
            let mut model = Gpt::new(config, &mut Rng::new(4));
            let cfg = Training { steps: 7, batch: 2, lr: 1e-2, eval_every, eval_windows: 5, threads: 1, save: None, ..Training::default() };
            let mut losses = Vec::new();
            train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| {
                if let Report::Step { train_loss, .. } = report {
                    losses.push(train_loss);
                }
            })
            .unwrap();
            losses
        };

        // Seven steps, one at a time...
        let each = run(1);
        assert_eq!(each.len(), 7);
        // ...and all seven at once, which is a final interval of 7 where a
        // run of 7 with `eval_every` 10 would have expected 10.
        let whole = run(10);
        assert_eq!(whole.len(), 1);
        let mean = each.iter().sum::<f32>() / 7.0;
        assert!((whole[0] - mean).abs() < 1e-5, "reported {}, the mean of the seven was {mean}", whole[0]);
    }

    /// Both splits have to be longer than the context, and the message has
    /// to say which one is not — the validation split is a tenth of the
    /// text, so it is nearly always the one that runs out first.
    #[test]
    fn a_text_too_short_to_draw_a_window_from_is_refused_by_name() {
        let tok = CharTokenizer::from_text("abc");
        let config = GptConfig { vocab: 3, context: 16, d_model: 8, n_heads: 2, n_layers: 1 };
        let cfg = Training { steps: 1, batch: 1, lr: 1e-2, eval_every: 1, eval_windows: 1, threads: 1, save: None, ..Training::default() };

        let attempt = |train_len: usize, val_len: usize| -> String {
            let mut model = Gpt::new(config, &mut Rng::new(4));
            let corpus = Corpus {
                train: tok.encode(&"abc".repeat(train_len)).unwrap(),
                val: tok.encode(&"abc".repeat(val_len)).unwrap(),
            };
            train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |_| {})
                .err()
                .map_or_else(String::new, |e| e.to_string())
        };

        // Plenty to train on, not enough to be tested on: the case a caller
        // is least likely to have thought about.
        let error = attempt(100, 5);
        assert!(error.contains("validation split is 15"), "{error}");
        assert!(error.contains("170 characters in all"), "{error}");
        assert!(attempt(5, 100).contains("training split is 15"));
        // Exactly one window is enough.
        assert_eq!(attempt(6, 6), "");
    }

    /// A run that cannot train is a mistake, not a no-op. Left to itself it
    /// would report a best validation loss of infinity at step 0 and leave
    /// the directory it was pointed at empty.
    #[test]
    fn a_run_of_no_steps_is_refused() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut model = Gpt::new(config, &mut Rng::new(4));
        let cfg = Training { steps: 0, batch: 2, lr: 1e-2, eval_every: 10, eval_windows: 8, threads: 1, save: None, ..Training::default() };
        let error = train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |_| {}).unwrap_err();
        assert!(error.to_string().contains("no steps"), "{error}");
    }

    /// The loop has to actually use the schedule, and not just build one.
    ///
    /// Adam moves each weight by about `lr` a step whatever its gradient, so
    /// the distance a run travels from where it started is close to the sum
    /// of its rates. A run that spends its whole length climbing to the peak
    /// has half the rate, on average, of one that starts there — and must
    /// travel roughly half as far. Measured: 0.433 of it, rather than 0.5,
    /// because a shorter step also means a different place to step from. The
    /// bound is wide on both sides of that; what it has to rule out is a
    /// loop that built a schedule and then ignored it, which measures 1.0.
    #[test]
    fn the_loop_moves_at_the_rate_the_schedule_says() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let start: Vec<f32> =
            Gpt::new(config, &mut Rng::new(4)).params().iter().flat_map(|p| p.value.to_vec()).collect();

        let travelled = |warmup, decay_to| -> f32 {
            let mut model = Gpt::new(config, &mut Rng::new(4));
            let cfg = Training {
                steps: 60,
                batch: 2,
                lr: 1e-2,
                warmup,
                decay_to,
                clip: None,
                eval_every: 60,
                eval_windows: 5,
                threads: 1,
                save: None,
                already_trained: false,
                stop: None,
            };
            train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |_| {}).unwrap();
            let end: Vec<f32> = model.params().iter().flat_map(|p| p.value.to_vec()).collect();
            start.iter().zip(&end).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt()
        };

        let flat = travelled(Some(0), 1.0);
        let ramped = travelled(Some(60), 1.0);
        assert!(flat > 0.0, "nothing moved, so nothing was measured");
        let ratio = ramped / flat;
        assert!((0.30..0.60).contains(&ratio), "a run that ramped all the way travelled {ratio} of a flat one");
    }

    /// Every checkpoint in a run is measured on the same windows, so that its
    /// number can be compared with the one before — which is the whole basis
    /// for keeping the best model.
    ///
    /// The way to see it is to stop the model moving. At a learning rate of
    /// zero the weights never change, so anything that makes two checkpoints
    /// disagree is the measurement and not the model.
    #[test]
    fn every_checkpoint_measures_the_same_validation_windows() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut model = Gpt::new(config, &mut Rng::new(4));
        let cfg = Training { steps: 30, batch: 2, lr: 0.0, eval_every: 10, eval_windows: 8, threads: 1, save: None, ..Training::default() };

        let mut seen = Vec::new();
        train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| {
            if let Report::Step { val_loss, .. } = report {
                seen.push(val_loss);
            }
        })
        .unwrap();
        assert_eq!(seen.len(), 3);
        assert!(seen.windows(2).all(|p| p[0] == p[1]), "a still model was measured differently: {seen:?}");
    }

    /// The estimate has to arrive while it is still worth having.
    #[test]
    fn a_run_says_how_fast_it_is_going_before_it_is_over() {
        let (tok, corpus) = a_text_and_its_contradiction();
        let config = GptConfig { vocab: tok.vocab(), ..TINY };
        let mut model = Gpt::new(config, &mut Rng::new(4));
        let cfg = Training { steps: 100, batch: 4, lr: 1e-2, eval_every: 50, eval_windows: 10, threads: 1, save: None, ..Training::default() };

        let mut pace = Vec::new();
        let mut steps_at_pace = None;
        let mut latest = 0;
        train(&mut model, &tok, &corpus, &cfg, &mut Rng::new(5), &mut |report| match report {
            Report::Pace { chars_per_sec, remaining_secs } => {
                steps_at_pace = Some(latest);
                pace.push((chars_per_sec, remaining_secs));
            }
            Report::Step { step, .. } => latest = step,
            // A new model, so there is no baseline to report.
            Report::Baseline { val_loss } => panic!("a fresh run measured a baseline of {val_loss}"),
        })
        .unwrap();

        assert_eq!(pace.len(), 1, "the pace is reported once");
        // Before the first checkpoint at step 50, so before any of the run
        // has been paid for twice over.
        assert_eq!(steps_at_pace, Some(0));
        let (rate, remaining) = pace[0];
        assert!(rate > 0.0 && rate.is_finite(), "{rate} characters a second");
        // Nine tenths of the run was still to come.
        let whole = rate * remaining / 0.9;
        let want = (cfg.steps * cfg.batch * config.context) as f32;
        assert!((whole - want).abs() < 0.01 * want, "estimated {whole} characters in all, not {want}");
    }
}
