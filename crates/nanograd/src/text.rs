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
use crate::optim::AdamW;
use crate::rng::Rng;
use std::io;
use std::path::Path;

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
            let worst = one.iter().zip(&three).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(worst < 1e-6, "a batch of three identical windows differs from one by {worst:e}");
        }
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
}
