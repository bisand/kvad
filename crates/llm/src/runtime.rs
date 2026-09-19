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
        let repo = repo_id.to_string();
        Self::load_custom(repo_id, progress, &mut |files, spec, progress| {
            let model = qcache::load(&repo, files, spec, precision, progress)?;
            Ok(Box::new(CpuSession::new(model, precision)))
        })
    }

    /// Load a model with a backend of the caller's choosing.
    pub fn load_custom(
        repo_id: &str,
        progress: &mut dyn FnMut(&str),
        build: SessionFactory,
    ) -> Res<Self> {
        let files = weights::fetch_with(repo_id, progress)?;
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
            let next = sampler.sample(&logits);
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
            if !on_token(&text[previous.len()..]) {
                break;
            }

            logits = self.session.forward(&[next])?;
        }
        stats.decode_secs = t1.elapsed().as_secs_f32();

        self.cached_ids = ids.clone();
        Ok((stats, ids))
    }
}

#[cfg(test)]
mod tests {
    use super::Llm;

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
