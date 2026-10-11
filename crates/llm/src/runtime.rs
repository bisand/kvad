//! A loaded model, ready to generate: weights, tokenizer, chat template and
//! stop tokens in one place.
//!
//! This is the API the CLI and the TUI both drive.

use crate::chat::{ChatTemplate, Message};
use crate::model::{CpuSession, Parked, Session, Spec};
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
    /// How many of those were the prompt, the rest being the reply to it.
    prompt_len: usize,
    /// When the conversation in the session last spoke; see `clock`.
    used: u64,
    /// Conversations set aside, each with its cache; see [`Llm::seat`].
    parked: Vec<Conversation>,
    /// Counts generations, to say which conversation has waited longest.
    clock: u64,
    /// Tokens of cache this model may hold across every conversation.
    hold: usize,
    /// Tokens of that spoken for by generations whose caches are out of the
    /// model's hands while they run; see [`Llm::detach`].
    lent: usize,
}

/// A generation under way: what [`Llm::begin`] hands out, [`Llm::step`]
/// advances and [`Llm::finish`] takes back.
pub struct Run {
    /// The prompt, and every token generated since.
    ids: Vec<u32>,
    prompt_len: usize,
    /// What the model made of everything it has been fed: the next token is
    /// sampled from these.
    logits: Vec<f32>,
    /// Whether the last of `ids` has been through the model. Not, between a
    /// token being handed out and the step after it.
    fed: bool,
    /// Tokens it may still generate.
    left: usize,
    explain: usize,
    stats: Stats,
    /// When prefill ended and decoding began.
    decoding: Instant,
    /// The generation's own cache, once it has been taken out of the
    /// session so that another generation can use it; see [`Llm::detach`].
    cache: Option<Parked>,
    /// Tokens of the model's hold this generation was counted for when its
    /// cache left: what it may grow to.
    reserved: usize,
}

impl Run {
    /// Whether the generation's cache is its own to carry rather than the
    /// session's: whether it can be advanced beside others.
    pub fn detached(&self) -> bool {
        self.cache.is_some()
    }
}

/// A conversation that is not the one in the session: its tokens, and the
/// cache that represents them, waiting for its next turn.
struct Conversation {
    ids: Vec<u32>,
    prompt_len: usize,
    used: u64,
    /// Positions its cache has memory for; see [`Session::room`].
    room: usize,
    cache: Parked,
}

/// As many conversations as are ever set aside, whatever room there is.
///
/// The tokens are the real limit — see [`Llm::hold`] — and this is only so
/// that a thousand three-token prompts do not leave a thousand of them to
/// search.
const MOST_PARKED: usize = 16;

/// Whether a prompt sharing `shared` leading tokens with a conversation
/// whose own prompt was `prompt_len` long is that conversation speaking
/// again.
///
/// Its next turn begins with all of its last prompt, and an edited or
/// regenerated turn with most of it. Another conversation under the same
/// system prompt shares only that, which is a small part of a long
/// conversation and most of a short one — and a short one costs little to
/// take over. So: at least half. Taking a conversation over cuts its cache
/// back to what is shared, so the half that is wrong about is the half
/// that was cheapest to lose.
fn continues(shared: usize, prompt_len: usize) -> bool {
    shared > 0 && shared * 2 >= prompt_len
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
    /// Tokens of cache the model holds once this is over, across every
    /// conversation it is keeping, and how many conversations that is.
    pub held_tokens: usize,
    pub conversations: usize,
    /// The most other generations this one ran beside at any step. Each
    /// step is everybody's, so this is why a reply was slower than the
    /// model's own pace: it was not the only one being written.
    pub beside: usize,
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
            prompt_len: 0,
            used: 0,
            parked: Vec::new(),
            clock: 0,
            hold: 0,
            lent: 0,
        }
        .holding(usize::MAX))
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
    ///
    /// `tools` is what the caller is offering the model this turn; pass an
    /// empty slice for a conversation with none. Whether the template has
    /// anywhere to put them is [`Llm::takes_tools`], and a caller that cares
    /// asks before generating rather than after.
    pub fn encode_chat(&self, messages: &[Message], tools: &[serde_json::Value]) -> Res<Vec<u32>> {
        match &self.chat {
            Some(t) => self.encode(&t.render(messages, tools, true)?),
            None => self.encode(messages.last().map(|m| m.content.as_str()).unwrap_or("")),
        }
    }

    /// Whether this model's template can be told about tools at all.
    ///
    /// False for every base model, which has no template, and for the
    /// instruct models trained before tool calling was a thing anyone
    /// expected of them.
    pub fn takes_tools(&self) -> bool {
        self.chat.as_ref().is_some_and(|t| t.takes_tools())
    }

    /// Where this model runs, e.g. `cpu q8` or `metal bf16`.
    pub fn backend(&self) -> String {
        self.session.label()
    }

    /// Forget every conversation, so the next prompt starts clean and
    /// nothing is held.
    pub fn reset(&mut self) -> Res<()> {
        self.parked.clear();
        self.clear()
    }

    /// Forget the conversation in the session, and leave the ones set aside.
    fn clear(&mut self) -> Res<()> {
        self.cached_ids.clear();
        self.prompt_len = 0;
        // Rewinding to zero is the one request every cache can honour, so the
        // answer is not worth checking here.
        self.session.truncate(0)?;
        Ok(())
    }

    /// Start the next prompt with an empty cache, without costing anybody
    /// theirs: the conversation in the session is set aside where the
    /// backend can do that, and forgotten where it cannot.
    ///
    /// For a request that asked for no cache — a measurement, mostly. It
    /// used to be [`Llm::reset`], which on a server is one account's
    /// benchmark throwing away every other account's conversation.
    pub fn fresh(&mut self) -> Res<()> {
        match self.set_aside() {
            true => Ok(()),
            false => self.clear(),
        }
    }

    /// Let the model hold `tokens` of cache across every conversation, or
    /// its own context if that is less: what a server has charged it for.
    ///
    /// One conversation may still grow to the model's context, as it always
    /// could; this is what the ones set aside have to fit in beside it.
    pub fn hold(&mut self, tokens: usize) {
        self.hold = tokens.min(self.spec.n_ctx);
    }

    fn holding(mut self, tokens: usize) -> Self {
        self.hold(tokens);
        self
    }

    /// Tokens of cache held, across the session and everything set aside.
    pub fn held(&self) -> usize {
        self.session.cached() + self.parked.iter().map(|c| c.ids.len()).sum::<usize>()
    }

    /// Move the conversation in the session out of it, leaving the session
    /// empty. `false` from a backend that cannot, with nothing changed.
    fn set_aside(&mut self) -> bool {
        if self.cached_ids.is_empty() {
            // Nothing to keep, and an empty session is what was wanted.
            return self.clear().is_ok();
        }
        let room = self.session.room();
        let Some(cache) = self.session.park() else {
            return false;
        };
        self.parked.push(Conversation {
            ids: std::mem::take(&mut self.cached_ids),
            prompt_len: std::mem::take(&mut self.prompt_len),
            used: self.used,
            room,
            cache,
        });
        true
    }

    /// Put the conversation `prompt_ids` continues into the session, so
    /// that what follows finds its cache there.
    ///
    /// The session holds one conversation, and until this existed the model
    /// did too: two people taking turns each found the other's conversation
    /// in the cache, shared nothing with it past the template's opening,
    /// and prefilled their own from the start, every turn. Measured on
    /// Qwen3-14B, the wait for a first token at turn eight was six times
    /// what it was alone, and growing with the conversation.
    ///
    /// So the conversations that are not speaking are set aside whole, and
    /// the one that is, is found by its tokens: whichever shares the longest
    /// prefix with the prompt, if that is enough of it to be the same
    /// conversation — see [`continues`]. Nothing names a conversation,
    /// because nothing has to: an OpenAI client sends the whole transcript
    /// and no id, and the transcript is the better key anyway.
    ///
    /// A prompt that continues none of them starts an empty cache, and the
    /// one in the session is set aside rather than overwritten. What is set
    /// aside is kept while it fits in [`Llm::hold`] beside what this
    /// generation may grow to; past that the one that spoke longest ago
    /// goes first.
    fn seat(&mut self, prompt_ids: &[u32], max_tokens: usize) -> Res<()> {
        let here = Self::common_prefix(&self.cached_ids, prompt_ids);
        let there = self
            .parked
            .iter()
            .enumerate()
            .map(|(at, c)| (Self::common_prefix(&c.ids, prompt_ids), at, c.prompt_len))
            // The longest, and of two as long the one that spoke last.
            .max_by_key(|&(shared, at, _)| (shared, self.parked[at].used));

        match there {
            // One set aside is the better match, and is this conversation.
            Some((shared, at, prompt_len)) if shared > here && continues(shared, prompt_len) => {
                let next = self.parked.remove(at);
                // Where the backend cannot set one aside it could not have
                // given this one out either, so this does not fail; if the
                // session's is empty there is nothing to keep.
                self.set_aside();
                self.session.resume(next.cache)?;
                (self.cached_ids, self.prompt_len, self.used) = (next.ids, next.prompt_len, next.used);
            }
            // The one in the session is, or the session is empty.
            _ if self.cached_ids.is_empty() || continues(here, self.prompt_len) => {}
            // None of them is: a conversation not seen before. Where the
            // backend keeps only one, this falls through to what it always
            // did, and the cache is cut back to whatever is shared.
            _ => {
                self.set_aside();
            }
        }

        // Room for what this generation may come to, beside the rest.
        self.make_room(prompt_ids.len().saturating_add(max_tokens));
        Ok(())
    }

    /// Let go of conversations set aside until `coming` tokens in the
    /// session fit beside the rest, the one that spoke longest ago first.
    ///
    /// Counted in the memory each one has, not the tokens in it, since the
    /// memory is what was charged for.
    fn make_room(&mut self, coming: usize) {
        let coming = coming.min(self.spec.n_ctx).max(self.session.room()) + self.lent;
        let waiting = |parked: &[Conversation]| parked.iter().map(|c| c.room).sum::<usize>();
        while !self.parked.is_empty()
            && (self.parked.len() > MOST_PARKED || waiting(&self.parked) + coming > self.hold)
        {
            let oldest = (0..self.parked.len()).min_by_key(|&at| self.parked[at].used).unwrap_or(0);
            self.parked.remove(oldest);
        }
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
        let mut run = self.begin(prompt_ids, max_tokens, explain)?;
        while let Some(chosen) = self.step(&mut run, sampler)? {
            if !on_token(&chosen) {
                break;
            }
        }
        Ok(self.finish(run))
    }

    /// Start a generation: find the prompt's conversation, prefill what of
    /// it the cache does not hold, and hand back the generation, ready for
    /// its first [`Llm::step`].
    ///
    /// A generation in three parts rather than one call, so that whoever
    /// drives it gets the thread back between tokens. [`Llm::generate`] is
    /// the three in a loop; an engine that advances several generations at
    /// once is the same three with something else between the steps.
    pub fn begin(&mut self, prompt_ids: &[u32], max_tokens: usize, explain: usize) -> Res<Run> {
        if prompt_ids.is_empty() {
            return Err("prompt encoded to zero tokens".into());
        }

        self.seat(prompt_ids, max_tokens)?;

        // Always leave at least one token to process, or there would be no
        // logits to sample the next token from.
        let wanted = Self::common_prefix(&self.cached_ids, prompt_ids)
            .min(self.session.cached())
            .min(prompt_ids.len() - 1);
        // What the cache *could* keep, which is not always what was asked for:
        // a recurrent state has already absorbed the tokens being dropped and
        // rewinds to zero instead. Forwarding from `wanted` after a refusal
        // would run the tail over a state that had already read it — a model
        // that is wrong rather than slow, and wrong in a way nothing reports.
        // See `KvCache::truncate`.
        let reuse = self.session.truncate(wanted)?;

        let mut stats =
            Stats { prompt_tokens: prompt_ids.len(), cached_tokens: reuse, ..Default::default() };

        let t0 = Instant::now();
        let logits = self.session.forward(&prompt_ids[reuse..])?;
        stats.prefill_secs = t0.elapsed().as_secs_f32();

        Ok(Run {
            ids: prompt_ids.to_vec(),
            prompt_len: prompt_ids.len(),
            logits,
            fed: true,
            left: max_tokens.min(self.spec.n_ctx.saturating_sub(prompt_ids.len())),
            explain,
            stats,
            decoding: Instant::now(),
            cache: None,
            reserved: 0,
        })
    }

    /// Take a generation's cache out of the session and give it to the
    /// generation to carry, leaving the session free for another's prefill.
    /// `false` from a backend that keeps one conversation, with nothing
    /// changed.
    ///
    /// `run` has to be the generation whose conversation the session holds:
    /// the one begun last, with none detached since. Until it is finished
    /// the model does not count its conversation among the ones it keeps,
    /// but does count the room it may come to against [`Llm::hold`].
    ///
    /// This is what lets several generations run at once. The session has
    /// one conversation in it; a generation that carries its own cache can
    /// be advanced beside others in one pass —
    /// [`Session::forward_each`] — and needs the session only to begin and
    /// to end.
    pub fn detach(&mut self, run: &mut Run) -> bool {
        if run.cache.is_some() {
            return true;
        }
        let Some(cache) = self.session.park() else {
            return false;
        };
        run.cache = Some(cache);
        run.reserved = (run.ids.len() + run.left).min(self.spec.n_ctx);
        self.lent += run.reserved;
        self.cached_ids.clear();
        self.prompt_len = 0;
        true
    }

    /// Give up on a generation that failed: what [`Llm::finish`] is to one
    /// that did not. Its cache, if it carried one, goes with it.
    pub fn abandon(&mut self, run: Run) {
        self.lent = self.lent.saturating_sub(run.reserved);
    }

    /// The next token of each of several generations, all of them detached:
    /// [`Llm::step`] for more than one, with the model read once for all of
    /// them where the backend can.
    ///
    /// `None` for a generation is the end of that one; the others go on. An
    /// error is everybody's, since the pass that failed was.
    pub fn step_each(&mut self, runs: &mut [&mut Run], samplers: &mut [&mut Sampler]) -> Res<Vec<Option<Chosen>>> {
        // Feed every token handed out last time, in one pass.
        let waiting: Vec<usize> = (0..runs.len()).filter(|&i| !runs[i].fed).collect();
        if !waiting.is_empty() {
            let mut tokens = Vec::with_capacity(waiting.len());
            let mut caches = Vec::with_capacity(waiting.len());
            for &i in &waiting {
                tokens.push(*runs[i].ids.last().expect("a run holds its prompt, which is not empty"));
                caches.push(runs[i].cache.take().ok_or("a generation advanced beside others has to be detached")?);
            }
            let fed = self.session.forward_each(&tokens, &mut caches);
            // The caches go back whatever happened, or they are lost.
            for (&i, cache) in waiting.iter().zip(caches) {
                runs[i].cache = Some(cache);
            }
            for (&i, logits) in waiting.iter().zip(fed?) {
                runs[i].logits = logits;
                runs[i].fed = true;
            }
        }
        runs.iter_mut().zip(samplers.iter_mut()).map(|(run, sampler)| self.choose(run, sampler)).collect()
    }

    /// The generation's next token, or `None` when it has ended: at an
    /// end-of-sequence token, or with its budget spent.
    ///
    /// The token a step hands out is not yet in the cache. The step after
    /// feeds it, which is what makes a generation that stops between two
    /// steps — an interrupt, a reader that went away — cost nothing for the
    /// token nobody will follow.
    pub fn step(&mut self, run: &mut Run, sampler: &mut Sampler) -> Res<Option<Chosen>> {
        if run.cache.is_some() {
            return Ok(self.step_each(&mut [run], &mut [sampler])?.pop().flatten());
        }
        if !run.fed {
            let last = *run.ids.last().expect("a run holds its prompt, which is not empty");
            run.logits = self.session.forward(&[last])?;
            run.fed = true;
        }
        self.choose(run, sampler)
    }

    /// Sample a generation's next token from the logits it holds.
    fn choose(&self, run: &mut Run, sampler: &mut Sampler) -> Res<Option<Chosen>> {
        if run.left == 0 {
            return Ok(None);
        }
        let (next, top) = match run.explain {
            0 => (sampler.sample(&run.logits), Vec::new()),
            k => sampler.sample_explained(&run.logits, k),
        };
        if self.eos.contains(&next) {
            run.left = 0;
            return Ok(None);
        }
        run.ids.push(next);
        run.fed = false;
        run.left -= 1;
        run.stats.generated_tokens += 1;

        Ok(Some(Chosen {
            id: next,
            text: added(&self.tokenizer, &run.ids)?,
            // Resolved here rather than by the caller, because the
            // tokenizer is here and an id means nothing without it.
            top: top.into_iter().map(|r| self.candidate(r)).collect(),
        }))
    }

    /// End a generation, wherever it got to, and say what it came to: its
    /// statistics, and every token of the prompt and the reply.
    ///
    /// This is where the model learns what its cache now holds, so a
    /// generation that was begun has to be finished, even one that was
    /// interrupted. One that failed is the exception: its cache is whatever
    /// the failure left, and dropping the [`Run`] leaves the model believing
    /// what it believed before, which the next prompt checks against the
    /// cache's own length.
    ///
    /// A generation that carried its own cache hands it back here: its
    /// conversation becomes the one in the session, and whatever was there
    /// is set aside.
    pub fn finish(&mut self, run: Run) -> (Stats, Vec<u32>) {
        let Run { ids, prompt_len, mut stats, decoding, cache, reserved, .. } = run;
        stats.decode_secs = decoding.elapsed().as_secs_f32();

        self.lent = self.lent.saturating_sub(reserved);
        if let Some(cache) = cache {
            // A backend that gave the cache out can set one aside and take
            // one back; if it somehow will not, the conversation is lost
            // and the next prompt finds an empty cache, which is slow and
            // not wrong.
            if !self.set_aside() || self.session.resume(cache).is_err() {
                let _ = self.clear();
                return (stats, ids);
            }
            self.make_room(self.session.room());
        }

        self.cached_ids = ids.clone();
        self.prompt_len = prompt_len;
        self.clock += 1;
        self.used = self.clock;
        stats.held_tokens = self.held();
        stats.conversations = self.parked.len() + 1;
        (stats, ids)
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
        // Whoever was mid-conversation keeps it, where there is room for it
        // beside a window of this.
        self.fresh()?;
        self.make_room(window);

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
            self.clear()?;
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
        self.clear()?;

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

/// Tokens of context [`added`] decodes behind the newest one.
///
/// Enough that whatever the newest token completes or changes lies inside
/// the window. A UTF-8 character is at most four bytes, so at most four
/// byte-level tokens; the rest is margin.
const WINDOW: usize = 8;

/// The text `ids`' last token adds to what the ones before it said.
///
/// Decoded one token at a time, byte-level BPE tokens can be fragments of a
/// UTF-8 character, and SentencePiece pieces lose or gain a leading space
/// depending on what comes before them. So the new text is the difference
/// between two decodes, with and without the newest token.
///
/// Those two decodes used to be of the **whole** sequence, prompt and all,
/// every token: a chat's history, or an agent's ten-thousand-token context,
/// decoded twice per token generated. `examples/decode_breakdown` measured
/// 0.3–0.4 ms a token at 2,048 tokens of context, growing with every token.
/// Only the last few tokens can differ between the two decodes, so only
/// they are decoded, **when the window can speak for everything before it**:
/// - It must begin on a whole character, so that nothing before it can
///   combine with its bytes. A window that decodes to `\u{FFFD}` first may
///   have cut one in half.
/// - It must say something of its own besides `\u{FFFD}`. `finished` holds
///   a trailing run of those back, in case the next token completes a
///   character, and releases the run once something whole follows. A run
///   longer than the window would be cut short. A window of nothing but
///   special tokens decodes to nothing, and then a SentencePiece decoder
///   strips the newest piece's leading space, where the whole sequence
///   would have kept it.
/// - It must not begin inside a run of byte tokens (`<0xF7>`, `<0x36>`, …),
///   which SentencePiece falls back to for anything without a piece of its
///   own. The decoder turns such a run into text as a whole. If the run is
///   not valid UTF-8, **every** byte in it becomes `\u{FFFD}`, ASCII
///   included, so a window that starts after the bad byte reads valid text
///   where the whole sequence reads none. Special tokens are skipped before
///   the decoder sees anything, so a run carries on across them:
///   `<0x6B> <unk> <0x0E>` is one run.
///
/// Where either fails, the window doubles backwards until both hold, or
/// until it is the whole sequence, which is what this always did.
/// `the_window_adds_what_the_whole_sequence_adds` holds the result to the
/// full decode, token by token.
fn added(tok: &Tokenizer, ids: &[u32]) -> Res<String> {
    let Some(last) = ids.len().checked_sub(1) else { return Ok(String::new()) };
    let decode = |ids: &[u32]| -> Res<String> { tok.decode(ids, true).map_err(|e| e.to_string().into()) };
    // `<0xF7>` and the like: SentencePiece's byte fallback. Byte-level BPE
    // has no such tokens, and this is always false for it.
    let byte = |id: u32| {
        tok.id_to_token(id).is_some_and(|t| t.len() == 6 && t.starts_with("<0x") && t.ends_with('>'))
    };
    // Whether a byte run crosses `from`: the last token before it and the
    // first in the window that the decoder will see are both bytes. Only
    // asked when the window starts on a byte, which in real text is rare,
    // so the special tokens are only looked up then.
    let splits_a_run = |from: usize| {
        if from == 0 || !ids[from..last].iter().any(|&id| byte(id)) {
            return false;
        }
        let special: std::collections::HashSet<u32> = tok
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, t)| t.special)
            .map(|(id, _)| id)
            .collect();
        let seen = |id: &&u32| !special.contains(id);
        let first = ids[from..last].iter().find(seen);
        let before = ids[..from].iter().rev().find(seen);
        matches!((first, before), (Some(&a), Some(&b)) if byte(a) && byte(b))
    };
    let mut back = WINDOW;
    loop {
        let from = last.saturating_sub(back);
        let previous = decode(&ids[from..last])?;
        let whole = !previous.starts_with('\u{FFFD}') && !finished(&previous).is_empty() && !splits_a_run(from);
        if whole || from == 0 {
            let text = decode(&ids[from..])?;
            return Ok(new_text(finished(&text), finished(&previous)).to_string());
        }
        back *= 2;
    }
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
    use super::{added, finished, new_text, Llm, WINDOW};
    use nervus::rng::Rng;
    use tokenizers::Tokenizer;

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

    /// What the newest token added, the way it was found before [`added`]:
    /// by decoding everything, twice.
    fn added_by_whole_decode(tok: &Tokenizer, ids: &[u32]) -> String {
        let text = tok.decode(ids, true).unwrap();
        let previous = tok.decode(&ids[..ids.len() - 1], true).unwrap();
        new_text(finished(&text), finished(&previous)).to_string()
    }

    /// A plain fixed window, without [`added`]'s checks: what they are for.
    fn added_by_fixed_window(tok: &Tokenizer, ids: &[u32]) -> String {
        let last = ids.len() - 1;
        let from = last.saturating_sub(WINDOW);
        let text = tok.decode(&ids[from..], true).unwrap();
        let previous = tok.decode(&ids[from..last], true).unwrap();
        new_text(finished(&text), finished(&previous)).to_string()
    }

    /// Every step of generating `ids` after the first `prompt` of them, asked
    /// of `how` and of the whole decode: the first step they differ, if any.
    fn first_difference(
        tok: &Tokenizer,
        ids: &[u32],
        prompt: usize,
        how: &dyn Fn(&Tokenizer, &[u32]) -> String,
    ) -> Option<(usize, String, String)> {
        (prompt.max(1)..=ids.len()).find_map(|n| {
            let (ours, whole) = (how(tok, &ids[..n]), added_by_whole_decode(tok, &ids[..n]));
            (ours != whole).then_some((n, ours, whole))
        })
    }

    fn windowed(tok: &Tokenizer, ids: &[u32]) -> String {
        added(tok, ids).unwrap()
    }

    /// Text that makes a tokenizer work: accents, three scripts, emoji built
    /// from several code points, combining marks, code and whitespace.
    const HARD_TEXT: &str = "Blåbærsyltetøy på skjærgården, naïve café. 東京の天気は晴れです。\
        مرحبا بالعالم 👩‍👩‍👧 family 🇳🇴 flag e\u{301} combined 😊😊\n\tfn main() { println!(\"{:?}\", x); }\n\n  \
        indented, then “quotes”, ellipsis…, and a tail of ❤️‍🔥.";

    /// Qwen2.5's tokenizer, byte-level BPE, as the server's default models
    /// have. From the Hub cache, so skipped where it has never been pulled.
    fn qwen() -> Option<Tokenizer> {
        let snapshots = crate::hub::cache_dir().join("models--Qwen--Qwen2.5-1.5B-Instruct").join("snapshots");
        let dir = std::fs::read_dir(snapshots).ok()?.flatten().next()?;
        Tokenizer::from_file(dir.path().join("tokenizer.json")).ok()
    }

    /// A SentencePiece tokenizer as Llama 2 and Mistral ship theirs:
    /// `▁` for a space, byte fallback for anything without a piece of its
    /// own, and the decoder's leading space stripped. The vocabulary is
    /// small and made up. Only the decoder is under test, and it is the one
    /// those models use.
    fn sentencepiece() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (i, t) in ["<unk>", "<s>", "</s>"].iter().enumerate() {
            vocab.insert(t.to_string(), i.into());
        }
        for b in 0..=255u32 {
            vocab.insert(format!("<0x{b:02X}>"), (3 + b).into());
        }
        let pieces = ["▁", "▁Hello", "▁world", ",", ".", "▁the", "▁a", "b", "é", "▁naïve", "▁東京", "の", "\n", "▁▁", "s", "▁café"];
        for (i, p) in pieces.iter().enumerate() {
            vocab.insert(p.to_string(), (259 + i).into());
        }
        let special = |id: u32, content: &str| serde_json::json!({
            "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        });
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [special(0, "<unk>"), special(1, "<s>"), special(2, "</s>")],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}
            ]},
            "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true,
                      "ignore_merges": false, "vocab": vocab, "merges": []}
        });
        json.to_string().parse().unwrap()
    }

    /// Ids drawn from anywhere in a vocabulary, special tokens included:
    /// fragments of characters in every order, which no real text produces.
    fn noise(tok: &Tokenizer, n: usize, seed: u64) -> Vec<u32> {
        let size = tok.get_vocab_size(true) as f32;
        let mut rng = Rng::new(seed);
        (0..n).map(|_| ((rng.uniform() * size) as u32).min(size as u32 - 1)).collect()
    }

    /// The point of [`added`]: token by token, the window says exactly what
    /// decoding everything said. On real text and on noise, with a prompt in
    /// front, for both kinds of tokenizer.
    #[test]
    fn the_window_adds_what_the_whole_sequence_adds() {
        let mut cases: Vec<(&str, Tokenizer, Vec<u32>)> = Vec::new();
        if let Some(q) = qwen() {
            let text = q.encode(HARD_TEXT, false).unwrap().get_ids().to_vec();
            cases.push(("qwen, text", q.clone(), text));
            for seed in 0..20 {
                cases.push(("qwen, noise", q.clone(), noise(&q, 200, seed)));
            }
        } else {
            eprintln!("Qwen2.5's tokenizer is not in the Hub cache; byte-level BPE is not exercised");
        }
        let sp = sentencepiece();
        for seed in 0..20 {
            cases.push(("sentencepiece, noise", sp.clone(), noise(&sp, 200, seed)));
        }
        for (what, tok, ids) in &cases {
            for prompt in [1, 5, ids.len() / 2] {
                if let Some((n, ours, whole)) = first_difference(tok, ids, prompt, &windowed) {
                    panic!("{what}, prompt {prompt}: at token {n} the window added {ours:?}, the whole decode {whole:?}");
                }
            }
        }
    }

    /// And the test above can tell. A fixed window, without the checks, cuts a
    /// long run of held-back `\u{FFFD}` short, starts inside runs of byte
    /// tokens, and drops a SentencePiece space after a window of special
    /// tokens.
    #[test]
    fn a_fixed_window_is_not_enough() {
        let sp = sentencepiece();
        let caught = (0..20).any(|seed| first_difference(&sp, &noise(&sp, 200, seed), 1, &added_by_fixed_window).is_some());
        assert!(caught, "the noise never needed the checks, so it cannot show they work");

        // `<s>` eight times, then `▁Hello`: the whole decode says " Hello",
        // and a window of nothing but `<s>` would say "Hello".
        let hello = sp.token_to_id("▁Hello").unwrap();
        let world = sp.token_to_id("▁world").unwrap();
        let ids: Vec<u32> = [world].into_iter().chain([1; WINDOW]).chain([hello]).collect();
        assert_eq!(added_by_whole_decode(&sp, &ids), " Hello");
        assert_eq!(added_by_fixed_window(&sp, &ids), "Hello");
        assert_eq!(added(&sp, &ids).unwrap(), " Hello");
    }

    /// What finding the new text costs a token, deep into a conversation.
    /// A measurement, not a test:
    ///
    ///     cargo test --release -p kvad runtime::tests::cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost() {
        let q = qwen().expect("Qwen2.5's tokenizer in the Hub cache");
        let one = q.encode(HARD_TEXT, false).unwrap().get_ids().to_vec();
        for context in [128, 2048, 8192, 32768] {
            let ids: Vec<u32> = one.iter().cycle().take(context).copied().collect();
            let time = |f: &dyn Fn() -> String| {
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    std::hint::black_box(f());
                }
                t.elapsed().as_secs_f64() * 1e3 / 20.0
            };
            let whole = time(&|| added_by_whole_decode(&q, &ids));
            let window = time(&|| added(&q, &ids).unwrap());
            println!("  context {context:6}: whole sequence twice {whole:8.3} ms   window {window:6.3} ms");
        }
    }



}
