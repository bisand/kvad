//! `kvad train`: a text file in, a model with a name out.
//!
//! The learning itself is not here. Every float that moves is moved by
//! `nervus`, which has no dependencies and is meant to be read; this module
//! is the part that turns a command line into a call to
//! [`nervus::text::train`] and the result into a model the rest of `kvad`
//! can find by name.
//!
//! # A name, not a path
//!
//! `--name shakespeare` writes into `$XDG_DATA_HOME/kvad/models/shakespeare`,
//! and from then on `kvad run --model shakespeare`, `kvad use shakespeare`
//! and `kvad ls` all know it, from any working directory. See
//! [`crate::weights`] for how a name, a directory and a Hub repo id are told
//! apart.
//!
//! # Training an existing model further
//!
//! `--from NAME --data more.txt` picks up a trained model and keeps going.
//! Two things about it are worth knowing before you rely on it, and both are
//! in the command's help text as well:
//!
//! * **The vocabulary is fixed at first training.** A character tokeniser
//!   gives ids to the characters it saw; a model has one row per id, in its
//!   embedding table and in its output head. There is no row to give a
//!   character that was not there the first time, so new text containing one
//!   is refused, and the message says which character.
//! * **The optimiser's state is not saved.** AdamW keeps two running averages
//!   per weight and a resumed run starts them from nothing. Measured: over
//!   the first 50 steps that cost 0.06 and 0.04 of training loss on two seeds
//!   and nothing on a third, with nothing visible by step 100.

use nervus::checkpoint;
use nervus::model::{Gpt, GptConfig};
use nervus::rng::Rng;
use nervus::text::{self, CharTokenizer, Corpus, Report, Training};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The largest corpus a run will take.
///
/// `nervus` holds the whole of it in memory as a `Vec<usize>`, eight bytes
/// a character, so 64 MB of text is half a gigabyte before training starts.
/// The limit is about this machine, not about taste — and it is here rather
/// than beside either of the two things that enforce it, an upload and a
/// crawl, because it is a fact about the training loop and they would drift.
pub const MAX_CORPUS_BYTES: usize = 64 * 1024 * 1024;

/// A model shape, under a name somebody might pick without reading a paper.
///
/// Three sizes rather than five flags. `train_text` in `nervus` still takes
/// `--layers`, `--d-model`, `--heads` and `--context` one by one, for anyone
/// who wants to see what each of them does.
pub struct Size {
    pub name: &'static str,
    pub layers: usize,
    pub d_model: usize,
    pub heads: usize,
    pub context: usize,
}

impl Size {
    /// The shape in one line. The parameter count is not in it: it depends on
    /// the vocabulary, which depends on the text, and `Gpt::summary` prints
    /// the true number a moment later.
    pub fn shape(&self) -> String {
        format!("{} layers, {} heads, d_model {}, context {}", self.layers, self.heads, self.d_model, self.context)
    }
}

/// What each one costs is measured, not guessed — see the README. The run
/// also times itself after ten steps and says what it expects to take on
/// *this* machine, which is the only estimate that can be honest.
pub const SIZES: [Size; 3] = [
    Size { name: "small", layers: 2, d_model: 64, heads: 4, context: 64 },
    Size { name: "medium", layers: 4, d_model: 128, heads: 4, context: 128 },
    Size { name: "large", layers: 6, d_model: 256, heads: 8, context: 256 },
];

pub const DEFAULT_SIZE: usize = 0;

pub fn size(name: &str) -> Option<&'static Size> {
    SIZES.iter().find(|s| s.name == name)
}

pub struct Options {
    pub data: PathBuf,
    /// Continue this model instead of starting one: a name, or a directory.
    pub from: Option<String>,
    /// Where the result is kept. Without it, `--from` trains in place.
    pub name: Option<String>,
    pub size: &'static Size,
    pub training: Training,
    pub seed: u64,
    /// Characters to write at each checkpoint, to watch it learn. 0 for none.
    pub sample: usize,
    pub temperature: f32,
    /// Raised from another thread to end the run early; see
    /// [`nervus::text::Training::stop`], which this is copied into. A
    /// command line leaves it `None` and uses Ctrl-C; a server cannot, because
    /// the run it is cancelling is one of several things the process is doing.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            data: PathBuf::new(),
            from: None,
            name: None,
            size: &SIZES[DEFAULT_SIZE],
            training: Training::default(),
            seed: 1337,
            sample: 160,
            temperature: 0.8,
            cancel: None,
        }
    }
}

#[derive(Debug)]
pub struct Summary {
    /// What `kvad run --model` will take: a name, or a path for a model
    /// trained somewhere other than the models home.
    pub handle: String,
    pub dir: PathBuf,
    pub params: usize,
    /// The loss of the model now in `dir`. For a continuation that improved
    /// nothing, the loss of the model that was already there.
    pub best_val: f32,
    /// The step whose model was written, or 0 for "nothing was written, and
    /// the model in `dir` is the one this run started from".
    pub best_step: usize,
    /// The best this run measured, whether or not it beat what was already
    /// there. Worth reporting when `best_step` is 0 and `best_val` is not
    /// about this run at all.
    pub reached: f32,
    pub last_val: f32,
    pub elapsed_secs: f32,
    /// True if [`Options::cancel`] was raised and the run ended early. The
    /// model in `dir` is still the best step it reached.
    pub stopped: bool,
}

impl Summary {
    /// Whether any checkpoint beat the model the run started from, and so
    /// whether anything was written. See [`Summary::best_step`].
    pub fn improved(&self) -> bool {
        self.best_step > 0
    }
}

/// Where a run's model will be written, and what it will then be called.
///
/// Refuses to write a name that is not one path component. `models_dir()` is
/// built by joining, and a "name" of `../../.ssh` would join right out of it.
fn target(opts: &Options) -> Res<PathBuf> {
    if let Some(name) = &opts.name {
        if !crate::weights::is_model_name(name) {
            return Err(format!("`{name}` is not a model name: it has to be one word, with no `/` in it").into());
        }
        return Ok(crate::weights::models_dir().join(name));
    }
    match &opts.from {
        // Training in place. The model is already somewhere with a name; put
        // it back there.
        Some(from) => crate::weights::local_dir(from)
            .ok_or_else(|| format!("`{from}` is not a model on this machine. `kvad ls` lists what is.").into()),
        None => Err("a new model needs a name: kvad train --data FILE --name NAME".into()),
    }
}

/// What a run is doing, in numbers rather than words.
///
/// Beside the lines of text a run already writes, not instead of them —
/// [`crate::weights::Fetch`] stands in the same relation to a fetch, and for
/// the same reason. A terminal wants "step 250/2000 train loss 1.842"; a loss
/// chart wants the two floats and cannot parse them back out of that sentence.
///
/// Unlike a [`crate::weights::Watcher`] this is an ordinary `&mut FnMut`:
/// training runs on the thread that called it, so there is nothing to share
/// across threads.
#[derive(Debug, Clone)]
pub enum Event {
    /// How fast this machine turned out to be, and what is left at that rate.
    /// Sent once, early, while there is still time to do something about it.
    Pace { chars_per_sec: f32, remaining_secs: f32 },
    /// A validation checkpoint.
    Step {
        step: usize,
        /// The last step this run will take, so a bar has a denominator.
        steps: usize,
        train_loss: f32,
        val_loss: f32,
        chars_per_sec: f32,
        elapsed_secs: f32,
    },
    /// The model was written to disk, because this step is the best so far.
    /// Always follows the [`Event::Step`] it belongs to.
    Saved { step: usize, val_loss: f32 },
    /// What the model writes when asked, at a checkpoint. Empty unless
    /// [`Options::sample`] asked for some.
    Sample { step: usize, text: String },
}

/// Train, saving the best model as it goes, and say what happened.
///
/// Progress goes to `out` rather than to stdout so that a caller which owns
/// the screen — the TUI, a test — is not written over. The same reason
/// [`crate::weights::fetch_with`] takes one.
pub fn run(opts: &Options, out: &mut dyn FnMut(&str)) -> Res<Summary> {
    run_watched(opts, out, &mut |_| {})
}

/// As [`run`], and also reporting [`Event`]s to `watch`.
pub fn run_watched(
    opts: &Options,
    out: &mut dyn FnMut(&str),
    watch: &mut dyn FnMut(Event),
) -> Res<Summary> {
    let dir = target(opts)?;
    let text = std::fs::read_to_string(&opts.data)
        .map_err(|e| format!("could not read {}: {e}", opts.data.display()))?;
    let mut rng = Rng::new(opts.seed);

    // A trained model arrives with the tokeniser it was trained with, and
    // only that one will do: id 17 means whatever was seventeenth in *its*
    // text. A new model gets one made from this text.
    let (mut model, tok) = match &opts.from {
        Some(from) => {
            let source = crate::weights::local_dir(from)
                .ok_or_else(|| format!("`{from}` is not a model on this machine. `kvad ls` lists what is."))?;
            let model = checkpoint::load(&source)?;
            let tok = CharTokenizer::load(&source)?;
            if model.config().vocab != tok.vocab() {
                return Err(format!("{}: the model and the tokeniser disagree about the vocabulary", source.display()).into());
            }
            let called = crate::weights::trained_name(&source).unwrap_or_else(|| source.display().to_string());
            out(&format!("continuing `{called}`, whose vocabulary of {} characters is fixed", tok.vocab()));
            (model, tok)
        }
        None => {
            let tok = CharTokenizer::from_text(&text);
            let config = GptConfig {
                vocab: tok.vocab(),
                context: opts.size.context,
                d_model: opts.size.d_model,
                n_heads: opts.size.heads,
                n_layers: opts.size.layers,
            };
            (Gpt::new(config, &mut rng), tok)
        }
    };

    let tokens = tok.encode(&text).map_err(|c| {
        format!(
            "{} contains {c:?}, and this model has no token for it. The vocabulary is \
             fixed at first training: train a new model on the two texts together instead.",
            opts.data.display()
        )
    })?;
    let corpus = Corpus::new(tokens, 0.1);
    out(&format!(
        "text: {} characters, {} distinct   train: {}   validation: {}",
        corpus.train.len() + corpus.val.len(),
        tok.vocab(),
        corpus.train.len(),
        corpus.val.len()
    ));
    out(&format!("model: {}", model.summary()));
    out(&format!(
        "loss to beat: {:.3} knowing nothing, {:.3} knowing only letter frequencies",
        (tok.vocab() as f32).ln(),
        text::unigram_loss(&corpus.train, tok.vocab())
    ));

    let cfg = Training {
        save: Some(dir.clone()),
        stop: opts.cancel.clone(),
        // The model this run starts from is already on disk, so what a
        // checkpoint has to beat is that model rather than nothing at all.
        already_trained: opts.from.is_some(),
        ..opts.training.clone()
    };
    let prompt = tok.encode("\n").ok().filter(|ids| !ids.is_empty()).unwrap_or_else(|| vec![0]);
    let mut sampler = Rng::new(opts.seed ^ 0x5a5a);
    let done = text::train(&mut model, &tok, &corpus, &cfg, &mut rng, &mut |report| match report {
        Report::Baseline { val_loss } => {
            out(&format!(
                "the model you are continuing scores {val_loss:.3} on this text; nothing \
                 worse than that will be written over it"
            ));
        }
        Report::Pace { chars_per_sec, remaining_secs } => {
            watch(Event::Pace { chars_per_sec, remaining_secs });
            // Only worth interrupting a terminal for when the answer is long
            // enough to act on. A watcher gets it either way and decides for
            // itself.
            if remaining_secs >= 20.0 {
                out(&format!(
                    "{chars_per_sec:.0} characters a second here — about {} to go. Ctrl-C now if that is too long.",
                    text::human_secs(remaining_secs)
                ));
            }
        }
        Report::Step { step, train_loss, val_loss, best, saved, elapsed_secs, chars_per_sec, model } => {
            let mark = if saved { "  *saved" } else if best { "  *best" } else { "" };
            out(&format!(
                "step {step:>5}/{}  train loss {train_loss:.3}  validation loss {val_loss:.3}  ({}){mark}",
                cfg.steps,
                text::human_secs(elapsed_secs)
            ));
            watch(Event::Step { step, steps: cfg.steps, train_loss, val_loss, chars_per_sec, elapsed_secs });
            if saved {
                watch(Event::Saved { step, val_loss });
            }
            if opts.sample > 0 {
                let out_ids = text::generate(model, &prompt, opts.sample, opts.temperature, &mut sampler);
                let text = tok.decode(&out_ids).trim().to_string();
                out(&format!("---\n{text}\n---"));
                watch(Event::Sample { step, text });
            }
        }
    })?;

    // Resolved now that it certainly exists, so that a model reached by
    // `--name` and the same model reached by `--from` report one path.
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    Ok(Summary {
        handle: crate::weights::trained_name(&dir).unwrap_or_else(|| dir.display().to_string()),
        dir,
        params: model.param_count(),
        best_val: done.best_val,
        best_step: done.best_step,
        reached: done.reached,
        last_val: done.last_val,
        elapsed_secs: done.elapsed_secs,
        stopped: done.stopped,
    })
}

/// The directory a name would be trained into, whether or not it exists yet.
pub fn would_write(name: &str) -> PathBuf {
    crate::weights::models_dir().join(name)
}

/// Whether `dir` already holds a model, so that `kvad train --name` can say
/// so before spending an hour overwriting it.
pub fn holds_a_model(dir: &Path) -> bool {
    dir.join(checkpoint::WEIGHTS_FILE).is_file()
}
