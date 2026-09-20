//! A loaded model, ready to generate: weights, tokenizer, chat template and
//! stop tokens in one place.
//!
//! This is the API the CLI and the TUI both drive.

use crate::chat::{ChatTemplate, Message};
use crate::model::{CpuSession, Session, Spec};
use crate::qcache;
use crate::quant::Precision;
use crate::sampler::Sampler;
use crate::weights;
use std::time::Instant;
use tokenizers::Tokenizer;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Llm {
    pub repo: String,
    pub spec: Spec,
    pub session: Box<dyn Session>,
    pub tokenizer: Tokenizer,
    pub chat: Option<ChatTemplate>,
    /// Token ids that end generation. Usually one; Llama 3 has two.
    pub eos: Vec<u32>,
    pub param_count: usize,
    /// Bytes the weights occupy, on whichever device holds them.
    pub weight_bytes: usize,
    /// Exactly the tokens the session's cache currently represents, so the
    /// next turn can find the shared prefix.
    cached_ids: Vec<u32>,
}

/// Builds the backend for a model whose files are already on disk.
///
/// The `kvad` crate cannot depend on the GPU crate — the dependency runs the
/// other way — so choosing a backend is the caller's job.
pub type SessionFactory<'a> = &'a mut dyn FnMut(
    &weights::ModelFiles,
    &Spec,
    &mut dyn FnMut(&str),
) -> Res<Box<dyn Session>>;

/// One generated token, and what it was chosen from.
#[derive(Debug, Clone)]
pub struct Chosen {
    pub id: u32,
    /// The new text this token added, which for byte-level BPE is sometimes
    /// nothing and sometimes several characters at once.
    pub text: String,
    /// The candidates, best first. Empty unless the caller asked for them.
    pub top: Vec<Candidate>,
}

/// A token the model considered.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: u32,
    pub text: String,
    /// The model's own probability, over the whole vocabulary.
    pub prob: f32,
    /// Whether the sampler's top-k and top-p left it in play.
    pub kept: bool,
    pub chosen: bool,
}

/// One token as the tokenizer sees it.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    /// The vocabulary entry, `Ġthe` and all.
    pub token: String,
    /// The text it covers, cut from the input.
    pub piece: String,
    pub start: usize,
    pub end: usize,
}

/// What a model scored on a text it did not write.
#[derive(Debug, Clone, Copy)]
pub struct Perplexity {
    /// Tokens the text encoded to.
    pub tokens: usize,
    /// Tokens that were actually graded: all but the first of each window.
    pub scored: usize,
    pub windows: usize,
    pub window: usize,
    /// Mean negative log-likelihood, in nats.
    pub nats: f64,
    pub perplexity: f64,
    /// The same number as compression: bits needed per token.
    pub bits_per_token: f64,
    pub stopped: bool,
}

/// Log of the probability this row of logits gives `target`.
///
/// The stable form: subtract the maximum before exponentiating, or a logit of
/// 90 overflows `f32` and the answer is a NaN that propagates into every
/// score after it.
fn log_prob(logits: &[f32], target: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    (logits[target as usize] - max) - sum.ln()
}

/// Progress and timing for one generation run.
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub prompt_tokens: usize,
    /// Prompt tokens served from an existing KV cache rather than recomputed.
    pub cached_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f32,
    pub decode_secs: f32,
}

impl Stats {
    pub fn tokens_per_sec(&self) -> f32 {
        self.generated_tokens as f32 / self.decode_secs.max(1e-6)
    }
}

impl Llm {
    pub fn load(repo_id: &str, precision: Precision) -> Res<Self> {
        Self::load_with(repo_id, precision, &mut |msg| eprintln!("  {msg}"))
    }

    pub fn load_with(
        repo_id: &str,
        precision: Precision,
        progress: &mut dyn FnMut(&str),
    ) -> Res<Self> {
        Self::load_watched(repo_id, precision, progress, &weights::Watcher::none())
    }

    /// As [`Llm::load_with`], reporting the download in bytes as well as in
    /// words. A server draws a progress bar from `watch`; a terminal does not
    /// need one and passes [`weights::Watcher::none`].
    pub fn load_watched(
        repo_id: &str,
        precision: Precision,
        progress: &mut dyn FnMut(&str),
        watch: &weights::Watcher,
    ) -> Res<Self> {
        // What the quantised-weight cache files this model under.
        let repo = weights::model_id(repo_id);
        Self::load_custom(repo_id, progress, watch, &mut |files, spec, progress| {
            let model = qcache::load(&repo, files, spec, precision, progress)?;
            Ok(Box::new(CpuSession::new(model, precision)))
        })
    }

    /// Load a model with a backend of the caller's choosing.
    pub fn load_custom(
        repo_id: &str,
        progress: &mut dyn FnMut(&str),
        watch: &weights::Watcher,
        build: SessionFactory,
    ) -> Res<Self> {
        let files = weights::fetch_watched(repo_id, progress, watch)?;
        progress("reading weights");
        let spec = Spec::from_json(&files.config)?;
        let session = build(&files, &spec, progress)?;

        progress("reading tokenizer");
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(|e| e.to_string())?;

        let chat = match &files.tokenizer_config {
            Some(p) => ChatTemplate::from_tokenizer_config(p)?,
            None => None,
        };

        // Stop tokens can be declared in three places, and models disagree
        // about which. Take the union of whatever we find.
        let mut eos = Vec::new();
        for path in [files.generation_config.as_ref(), Some(&files.config)]
            .into_iter()
            .flatten()
        {
            if let Ok(json) = weights::read_json(path) {
                match json.get("eos_token_id") {
                    Some(serde_json::Value::Number(n)) => {
                        eos.extend(n.as_u64().map(|v| v as u32))
                    }
                    Some(serde_json::Value::Array(a)) => {
                        eos.extend(a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32))
                    }
                    _ => {}
                }
            }
        }
        eos.sort_unstable();
        eos.dedup();

        let param_count = session.param_count();
        let weight_bytes = session.weight_bytes();
        Ok(Llm {
            repo: repo_id.to_string(),
            spec,
            session,
            tokenizer,
            chat,
            eos,
            param_count,
            weight_bytes,
            cached_ids: Vec::new(),
        })
    }

    pub fn is_instruct(&self) -> bool {
        self.chat.is_some()
    }

    pub fn encode(&self, text: &str) -> Res<Vec<u32>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Res<String> {
        self.tokenizer.decode(ids, true).map_err(|e| e.to_string().into())
    }

    /// Turn a conversation into token ids, using the model's own template.
    ///
    /// Falls back to the raw last message for base models, which have no
    /// template and no notion of turns.
    pub fn encode_chat(&self, messages: &[Message]) -> Res<Vec<u32>> {
        match &self.chat {
            Some(t) => self.encode(&t.render(messages, true)?),
            None => self.encode(messages.last().map(|m| m.content.as_str()).unwrap_or("")),
        }
    }

    /// Where this model runs, e.g. `cpu q8` or `metal bf16`.
    pub fn backend(&self) -> String {
        self.session.label()
    }

    /// Forget the conversation so far, so the next prompt starts clean.
    pub fn reset(&mut self) -> Res<()> {
        self.cached_ids.clear();
        self.session.truncate(0)
    }

    /// How many leading tokens two sequences share.
    ///
    /// In a chat, turn N's tokens begin with all of turn N-1's, so this is
    /// usually the entire conversation so far.
    pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Generate from `prompt_ids`, calling `on_token` with each new fragment of
    /// text as it appears.
    ///
    /// The KV cache lives inside the session, and is reused across calls: the
    /// shared prefix with the previous prompt is kept and only the remainder
    /// is prefilled. On a chat that means re-reading just the newest message
    /// rather than the whole transcript.
    ///
    /// Returning `false` from `on_token` stops generation — that is how the
    /// TUI implements its interrupt key.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        sampler: &mut Sampler,
        max_tokens: usize,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> Res<(Stats, Vec<u32>)> {
        self.generate_explained(prompt_ids, sampler, max_tokens, 0, |c| on_token(&c.text))
    }

    /// As [`Llm::generate`], reporting what each token was chosen *from*.
    ///
    /// `explain` is how many candidates to report per step; 0 is
    /// [`Llm::generate`] and costs nothing extra. Anything above it costs one
    /// sort and one softmax over the vocabulary per token — tens of
    /// microseconds against tens of milliseconds of matmul, so the numbers
    /// this produces are not numbers about a slower engine.
    ///
    /// This is the whole of the playground's top-k view: the logits were
    /// always there, and generation was simply throwing away everything except
    /// the argument's winner.
    pub fn generate_explained(
        &mut self,
        prompt_ids: &[u32],
        sampler: &mut Sampler,
        max_tokens: usize,
        explain: usize,
        mut on_token: impl FnMut(&Chosen) -> bool,
    ) -> Res<(Stats, Vec<u32>)> {
        if prompt_ids.is_empty() {
            return Err("prompt encoded to zero tokens".into());
        }

        // Always leave at least one token to process, or there would be no
        // logits to sample the next token from.
        let reuse = Self::common_prefix(&self.cached_ids, prompt_ids)
            .min(self.session.cached())
            .min(prompt_ids.len() - 1);
        self.session.truncate(reuse)?;

        let mut stats =
            Stats { prompt_tokens: prompt_ids.len(), cached_tokens: reuse, ..Default::default() };

        let t0 = Instant::now();
        let mut logits = self.session.forward(&prompt_ids[reuse..])?;
        stats.prefill_secs = t0.elapsed().as_secs_f32();

        let mut ids: Vec<u32> = prompt_ids.to_vec();
        let budget = max_tokens.min(self.spec.n_ctx.saturating_sub(ids.len()));

        let t1 = Instant::now();
        for _ in 0..budget {
            let (next, top) = match explain {
                0 => (sampler.sample(&logits), Vec::new()),
                k => sampler.sample_explained(&logits, k),
            };
            if self.eos.contains(&next) {
                break;
            }
            ids.push(next);
            stats.generated_tokens += 1;

            // Decode the whole tail and emit only what is new. Byte-level BPE
            // tokens can be fragments of a UTF-8 character, so decoding them
            // one at a time would emit replacement characters mid-word.
            let text = self.decode(&ids)?;
            let previous = self.decode(&ids[..ids.len() - 1])?;
            let chosen = Chosen {
                id: next,
                text: new_text(finished(&text), finished(&previous)).to_string(),
                // Resolved here rather than by the caller, because the
                // tokenizer is here and an id means nothing without it.
                top: top.into_iter().map(|r| self.candidate(r)).collect(),
            };
            if !on_token(&chosen) {
                break;
            }

            logits = self.session.forward(&[next])?;
        }
        stats.decode_secs = t1.elapsed().as_secs_f32();

        self.cached_ids = ids.clone();
        Ok((stats, ids))
    }

    fn candidate(&self, r: crate::sampler::Ranked) -> Candidate {
        Candidate {
            // A single id can be half a UTF-8 character, which decodes to a
            // replacement char. That is the truth about byte-level BPE and
            // showing it is better than hiding it.
            text: self.decode(&[r.id]).unwrap_or_default(),
            id: r.id,
            prob: r.prob,
            kept: r.kept,
            chosen: r.chosen,
        }
    }

    /// How the tokenizer splits a piece of text.
    ///
    /// `piece` is the substring the token covers, taken from the offsets, so
    /// what is shown is the reader's own text cut up rather than a rendering
    /// of vocabulary entries. `token` is the vocabulary entry itself, which is
    /// where the `Ġ` and `Ċ` of byte-level BPE live — the two disagree, and
    /// seeing how is most of the point of an inspector.
    pub fn tokenize(&self, text: &str) -> Res<Vec<Token>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
        let offsets = enc.get_offsets();
        Ok(enc
            .get_ids()
            .iter()
            .zip(enc.get_tokens())
            .enumerate()
            .map(|(i, (&id, token))| {
                let (start, end) = offsets.get(i).copied().unwrap_or((0, 0));
                Token {
                    id,
                    token: token.clone(),
                    piece: text.get(start..end).unwrap_or("").to_string(),
                    start,
                    end,
                }
            })
            .collect())
    }

    /// How surprised this model is by a text it did not write.
    ///
    /// Perplexity is the exponential of the mean negative log-likelihood the
    /// model assigns to each actual next token: "on average, how many equally
    /// likely tokens was it choosing between?". Lower is better, 1.0 would be
    /// certainty, and the vocabulary size is what a model that had learnt
    /// nothing would score.
    ///
    /// Scored in windows rather than in one pass, because a file is usually
    /// longer than the context. Each window starts with an empty cache, and
    /// its **first token is not scored** — nothing precedes it, so there is no
    /// prediction to grade. That makes the number slightly pessimistic
    /// compared to a sliding window with overlap, and it makes it comparable
    /// between models, which is what it is for.
    ///
    /// `on_window` is called after each window with the tokens scored so far
    /// and the total; returning false stops, and what was measured up to that
    /// point comes back with `stopped` set.
    pub fn perplexity(
        &mut self,
        text: &str,
        window: usize,
        mut on_window: impl FnMut(usize, usize) -> bool,
    ) -> Res<Perplexity> {
        let ids = self.encode(text)?;
        if ids.len() < 2 {
            return Err("there is not enough text here to score: two tokens at least".into());
        }
        let window = window.clamp(2, self.spec.n_ctx);
        let vocab = self.spec.vocab_size;

        let mut nats = 0.0f64;
        let mut scored = 0usize;
        let mut windows = 0usize;
        let mut stopped = false;

        for part in ids.chunks(window) {
            if part.len() < 2 {
                // A last window holding one token has nothing to grade.
                break;
            }
            // Every window is independent, so the cache from the last one
            // would be a prefix this text never had.
            self.reset()?;
            let logits = self.session.forward_all(part)?;
            if logits.len() != part.len() * vocab {
                return Err(format!(
                    "the backend returned {} logits for {} tokens of a {vocab}-token vocabulary",
                    logits.len(),
                    part.len()
                )
                .into());
            }
            for i in 0..part.len() - 1 {
                nats += -log_prob(&logits[i * vocab..(i + 1) * vocab], part[i + 1]) as f64;
                scored += 1;
            }
            windows += 1;
            if !on_window(scored, ids.len()) {
                stopped = true;
                break;
            }
        }

        // The cache now holds a stretch of somebody's test set, which is not
        // a prefix of anyone's next conversation.
        self.reset()?;

        let mean = nats / scored.max(1) as f64;
        Ok(Perplexity {
            tokens: ids.len(),
            scored,
            windows,
            window,
            nats: mean,
            perplexity: mean.exp(),
            bits_per_token: mean / std::f64::consts::LN_2,
            stopped,
        })
    }
}

/// The part of a decode that is finished enough to send.
///
/// A character whose bytes are split across several tokens decodes, while it
/// is still incomplete, to U+FFFD. That replacement character is the front
/// half of something the next token completes, so sending it would put a
/// `\u{FFFD}` in the stream that the real character then appears after. It is
/// held back instead, and arrives as itself once the rest of it does.
///
/// A model that genuinely ends its output with U+FFFD loses it. That is the
/// trade against every multi-byte character in every other generation.
fn finished(text: &str) -> &str {
    text.trim_end_matches('\u{FFFD}')
}

/// What `text` has that `previous` did not.
///
/// Not `text[previous.len()..]`, which is what this was and which panics.
/// `previous` is not always a byte prefix of `text`: decoding `Hi 😊` a token
/// at a time gives `Hi`, then `Hi \u{FFFD}` at six bytes, then `Hi 😊` at seven —
/// so the previous length lands inside the emoji, and slicing a `str`
/// anywhere but a character boundary is a panic.
///
/// Walking characters cannot land inside one, and the first place the two
/// disagree is where the new text starts.
fn new_text<'a>(text: &'a str, previous: &str) -> &'a str {
    let mut previous = previous.chars();
    for (i, c) in text.char_indices() {
        if previous.next() != Some(c) {
            return &text[i..];
        }
    }
    ""
}

#[cfg(test)]
mod tests {
    use super::{finished, new_text, Llm};

    #[test]
    fn new_text_is_what_the_last_token_added() {
        assert_eq!(new_text("Hello world", "Hello"), " world");
        assert_eq!(new_text("Hello", "Hello"), "");
        assert_eq!(new_text("Hello", ""), "Hello");
        // A decode that shrank: nothing to send, rather than a panic.
        assert_eq!(new_text("Hi", "Hi there"), "");
    }

    #[test]
    fn a_character_split_across_tokens_arrives_once_and_whole() {
        // The real decodes of `Hi 😊`, which Qwen2.5 tokenises as
        // [13048, 26525, 232]: the emoji's bytes arrive in two tokens, and
        // the first of them decodes to a replacement character.
        let steps = ["Hi", "Hi \u{FFFD}", "Hi 😊"];

        // Why the old code died: 6 is not a character boundary in `Hi 😊`,
        // it is inside the emoji, and `text[6..]` on that is a panic.
        assert_eq!(steps[1].len(), 6);
        assert_eq!(steps[2].len(), 7);
        assert!(!steps[2].is_char_boundary(steps[1].len()));

        let mut stream = String::new();
        for i in 0..steps.len() {
            let previous = if i == 0 { "" } else { steps[i - 1] };
            stream.push_str(new_text(finished(steps[i]), finished(previous)));
        }
        assert_eq!(stream, "Hi 😊");
        assert!(!stream.contains('\u{FFFD}'), "a half-decoded character reached the stream");
    }

    #[test]
    fn an_incomplete_tail_is_held_back() {
        assert_eq!(finished("Hi \u{FFFD}"), "Hi ");
        assert_eq!(finished("Hi 😊"), "Hi 😊");
        assert_eq!(finished(""), "");
        // Nothing should leak if a decoder produces more than one of them.
        assert_eq!(finished("ab\u{FFFD}\u{FFFD}"), "ab");
    }

    #[test]
    fn common_prefix_finds_the_shared_history() {
        // In a chat, turn N's tokens begin with all of turn N-1's, so the
        // shared prefix is the entire conversation so far.
        let previous = [1u32, 2, 3, 4, 5];
        let next = [1u32, 2, 3, 4, 5, 9, 9, 9];
        assert_eq!(Llm::common_prefix(&previous, &next), 5);

        // A cleared or edited history diverges early and most of the cache
        // has to be thrown away.
        assert_eq!(Llm::common_prefix(&previous, &[1, 2, 7, 8]), 2);
        assert_eq!(Llm::common_prefix(&previous, &[9]), 0);
        assert_eq!(Llm::common_prefix(&[], &next), 0);
    }
}
