//! Chat templates.
//!
//! # Why a template is needed at all
//!
//! A base model like GPT-2 only continues text. An instruction-tuned model was
//! additionally trained on conversations wrapped in specific marker tokens, and
//! it only behaves like an assistant when it sees those exact markers. For
//! SmolLM2 and Qwen the format is ChatML:
//!
//! ```text
//! <|im_start|>system
//! You are a helpful assistant.<|im_end|>
//! <|im_start|>user
//! Why is the sky blue?<|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! Generation then starts after that final line. Get the markers wrong and the
//! model does not error — it drifts back towards being a base model, inventing
//! both halves of the conversation. "The model is dumb" is very often "the
//! template is wrong".
//!
//! # Why this needs a template engine
//!
//! HuggingFace ships the format as a **Jinja2 template string** inside
//! `tokenizer_config.json`, one per model, and they differ in real ways: some
//! inject a default system prompt, Llama 3 uses different markers, some models
//! interleave tool calls. Hardcoding one family's format works until you load
//! a model from another. So we render the model's own template with
//! `minijinja`, which is what the mainstream Rust runtimes do too.

use minijinja::{Environment, Value};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Message { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Message { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Message { role: "assistant".into(), content: content.into() }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos: Option<String>,
    eos: Option<String>,
}

/// Python's string methods, for templates written against Python's Jinja2.
///
/// HuggingFace's chat templates are rendered by Jinja2 inside CPython, where a
/// string carries every method `str` has. `minijinja` implements Jinja's
/// filters and tests, not Python's methods, so `content.startswith("<think>")`
/// — which Qwen3's template uses to find a reasoning block — is an unknown
/// method, and the render fails *after* the weights are in memory.
///
/// Only what templates in the wild actually reach for. An unknown method still
/// errors, because silently answering `undefined` would render a template that
/// looks fine and is subtly wrong, and a wrong template is the failure mode
/// this whole module exists to avoid.
///
/// `minijinja-contrib` ships a fuller version of this; it is one small
/// function against a dependency, and this way the list says what these models
/// actually need.
fn python_string_methods(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, minijinja::Error> {
    let unknown = || minijinja::Error::from(minijinja::ErrorKind::UnknownMethod);
    let Some(text) = value.as_str() else { return Err(unknown()) };
    let arg = |i: usize| args.get(i).and_then(|v| v.as_str()).unwrap_or("");

    Ok(match method {
        "startswith" => Value::from(text.starts_with(arg(0))),
        "endswith" => Value::from(text.ends_with(arg(0))),
        // Python's bare `strip()` takes no argument and trims whitespace; with
        // one it trims any of those characters, which is not the same thing.
        "strip" | "lstrip" | "rstrip" => {
            let chars: Vec<char> = arg(0).chars().collect();
            fn trim<'a>(s: &'a str, chars: &[char], end: bool) -> &'a str {
                match (chars.is_empty(), end) {
                    (true, false) => s.trim_start(),
                    (true, true) => s.trim_end(),
                    (false, false) => s.trim_start_matches(|c| chars.contains(&c)),
                    (false, true) => s.trim_end_matches(|c| chars.contains(&c)),
                }
            }
            Value::from(match method {
                "lstrip" => trim(text, &chars, false),
                "rstrip" => trim(text, &chars, true),
                _ => trim(trim(text, &chars, false), &chars, true),
            })
        }
        "lower" => Value::from(text.to_lowercase()),
        "upper" => Value::from(text.to_uppercase()),
        "title" => Value::from(
            text.split(' ')
                .map(|w| match w.chars().next() {
                    Some(c) => c.to_uppercase().collect::<String>() + &w[c.len_utf8()..].to_lowercase(),
                    None => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        "replace" => Value::from(text.replace(arg(0), arg(1))),
        "split" => Value::from(match args.first().and_then(|v| v.as_str()) {
            // Python's `split()` with no argument splits on runs of
            // whitespace and drops empties; `split(sep)` keeps them.
            None => text.split_whitespace().map(Value::from).collect::<Vec<_>>(),
            Some(sep) => text.split(sep).map(Value::from).collect::<Vec<_>>(),
        }),
        "find" | "rfind" => {
            let at = match method {
                "find" => text.find(arg(0)),
                _ => text.rfind(arg(0)),
            };
            // Python answers -1 rather than raising, and templates test for it.
            Value::from(at.map_or(-1i64, |i| i as i64))
        }
        _ => return Err(unknown()),
    })
}

impl ChatTemplate {
    /// Read the template out of a `tokenizer_config.json`, if it has one.
    pub fn from_tokenizer_config(path: &Path) -> Res<Option<Self>> {
        let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;

        // The field is usually a string. Some models ship several named
        // templates as a list, in which case "default" is the one we want.
        let template = match cfg.get("chat_template") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(items)) => {
                let pick = items
                    .iter()
                    .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| items.first());
                match pick.and_then(|t| t.get("template")).and_then(|t| t.as_str()) {
                    Some(s) => s.to_string(),
                    None => return Ok(None),
                }
            }
            _ => return Ok(None),
        };

        let mut env = Environment::new();
        // Templates sometimes validate their input and bail out. Without this
        // global, rendering fails with a confusing "unknown function" instead
        // of the model author's actual message.
        env.add_function("raise_exception", |msg: String| -> Result<(), minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        // Newer templates stamp a date into the system prompt.
        env.add_function("strftime_now", |_format: String| -> String {
            // Deliberately fixed: a template that wants today's date gets a
            // plausible one rather than a crash. Wire up a clock if you care.
            "2026-01-01".to_string()
        });
        env.set_unknown_method_callback(python_string_methods);
        env.add_template_owned("chat", template)?;

        let token_text = |key: &str| -> Option<String> {
            match cfg.get(key)? {
                serde_json::Value::String(s) => Some(s.clone()),
                // Sometimes a full AddedToken object.
                v => v.get("content")?.as_str().map(str::to_string),
            }
        };

        Ok(Some(ChatTemplate {
            env,
            bos: token_text("bos_token"),
            eos: token_text("eos_token"),
        }))
    }

    /// Render a conversation into the exact string the model expects.
    ///
    /// `add_generation_prompt` appends the opening marker for the assistant's
    /// turn — the cue that says "your line now".
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Res<String> {
        let tmpl = self.env.get_template("chat")?;
        Ok(tmpl.render(minijinja::context! {
            messages => Value::from_serialize(messages),
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos.clone().unwrap_or_default(),
            eos_token => self.eos.clone().unwrap_or_default(),
        })?)
    }
}

// ---------------------------------------------------------------------------
// Reasoning traces
// ---------------------------------------------------------------------------

/// The tags a reasoning model wraps its working in.
const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Where a reply is, between those tags.
enum Where {
    /// Nothing but whitespace so far, and an opening tag still possible.
    Before,
    Inside,
    /// Past the trace, or established that there is not one.
    After,
}

/// Separating a reasoning model's working from its answer, as it streams.
///
/// Qwen3 and its relatives open a reply with `<think>`, reason in the open,
/// close with `</think>`, and then answer. Handed to a client as one string
/// that is the whole reply, which is how it arrives from the engine, the
/// working reads as part of the answer — and it is not: it contradicts
/// itself, changes its mind, and is often longer than what follows.
///
/// # Why this is a state machine and not a `split`
///
/// The reply arrives a token at a time and a tag is several tokens: `<`,
/// `think`, `>`. So text that might still become a tag is held back until it
/// either completes one or cannot, and everything else goes out immediately —
/// a client that waited for the whole reply to split it would lose the
/// streaming it came for.
///
/// # Only at the beginning
///
/// A `<think>` after the model has already said something is not a trace,
/// it is a model writing about tags — this file's own documentation would
/// parse as one. So the opening tag counts only before any other text, and
/// once something else has arrived the rest of the reply is the answer,
/// literal tags and all. Nothing is ever dropped either way.
pub struct Thinking {
    held: String,
    at: Where,
}

impl Default for Thinking {
    fn default() -> Self {
        Thinking::new()
    }
}

impl Thinking {
    pub fn new() -> Thinking {
        Thinking { held: String::new(), at: Where::Before }
    }

    /// Feed a piece of the reply. Returns what of it is working, and what of
    /// it is answer — either may be empty, and both are empty for a piece
    /// held back as a possible tag.
    pub fn feed(&mut self, text: &str) -> (String, String) {
        self.held.push_str(text);
        let (mut working, mut answer) = (String::new(), String::new());
        loop {
            match self.at {
                Where::After => {
                    answer.push_str(&self.held);
                    self.held.clear();
                    return (working, answer);
                }
                Where::Before => match self.held.find(OPEN) {
                    // The same rule as below, and it has to be in both: a tag
                    // can arrive in the piece that also carries the text
                    // before it, and then the text before it decides.
                    Some(at) if !self.held[..at].trim().is_empty() => {
                        answer.push_str(&self.held);
                        self.held.clear();
                        self.at = Where::After;
                        return (working, answer);
                    }
                    Some(at) => {
                        answer.push_str(&self.held[..at]);
                        self.held = self.held[at + OPEN.len()..].to_string();
                        self.at = Where::Inside;
                    }
                    None => {
                        let keep = self.held.len() - partial(&self.held, OPEN);
                        // Anything but whitespace before a tag means there is
                        // no trace here and there will not be one.
                        if !self.held[..keep].trim().is_empty() {
                            answer.push_str(&self.held);
                            self.held.clear();
                            self.at = Where::After;
                            return (working, answer);
                        }
                        answer.push_str(&self.held[..keep]);
                        self.held = self.held[keep..].to_string();
                        return (working, answer);
                    }
                },
                Where::Inside => match self.held.find(CLOSE) {
                    Some(at) => {
                        working.push_str(&self.held[..at]);
                        self.held = self.held[at + CLOSE.len()..].to_string();
                        self.at = Where::After;
                    }
                    None => {
                        let keep = self.held.len() - partial(&self.held, CLOSE);
                        working.push_str(&self.held[..keep]);
                        self.held = self.held[keep..].to_string();
                        return (working, answer);
                    }
                },
            }
        }
    }

    /// End of the reply: whatever is still held back.
    ///
    /// A trace with no closing tag — the budget ran out mid-thought, which
    /// for a small reasoning model is common — is still the trace, and is
    /// returned as one rather than thrown away or shown as an answer.
    pub fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.held);
        match self.at {
            Where::Inside => (rest, String::new()),
            _ => (String::new(), rest),
        }
    }

    /// Whether a trace was opened and never closed.
    pub fn unfinished(&self) -> bool {
        matches!(self.at, Where::Inside)
    }
}

/// How much of the end of `s` could still become the start of `tag`.
fn partial(s: &str, tag: &str) -> usize {
    let most = (tag.len() - 1).min(s.len());
    (1..=most)
        .rev()
        .find(|n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Feeding the same reply whole and a character at a time has to give the
    /// same answer, because the stream is the case this exists for: a tag is
    /// several tokens and arrives in pieces.
    #[test]
    fn a_trace_splits_the_same_however_it_arrives() {
        let reply = "<think>I should check the listing.</think>The borrow checker rejects it.";
        assert_eq!(
            whole(reply),
            ("I should check the listing.".to_string(), "The borrow checker rejects it.".to_string())
        );
        assert_eq!(by_char(reply), whole(reply), "the stream split differently");
    }

    #[test]
    fn a_reply_with_no_trace_is_all_answer() {
        for reply in ["Just an answer.", "  leading space then text", ""] {
            let (working, answer) = whole(reply);
            assert!(working.is_empty(), "{reply:?} invented a trace");
            assert_eq!(answer, reply);
            assert_eq!(by_char(reply).1, reply);
        }
    }

    /// A small reasoning model runs out of budget mid-thought often enough
    /// that this is the ordinary case and not the strange one.
    #[test]
    fn a_trace_that_never_closes_is_still_a_trace() {
        let mut thinking = Thinking::new();
        let (working, answer) = thinking.feed("<think>I should check whether");
        assert_eq!(working, "I should check whether");
        assert!(answer.is_empty());
        assert!(thinking.unfinished());
        assert_eq!(thinking.finish(), (String::new(), String::new()));
    }

    /// This module's own documentation mentions the tag. A model writing
    /// about reasoning models must not be parsed as one.
    #[test]
    fn a_tag_after_the_answer_has_started_is_just_text() {
        let reply = "Reasoning models open with <think> and close with </think>.";
        let (working, answer) = whole(reply);
        assert!(working.is_empty());
        assert_eq!(answer, reply, "the tags were eaten");
        assert_eq!(by_char(reply).1, reply);
    }

    /// Whatever the split, every character comes out of one side or the
    /// other. A streamed reply may be cut anywhere, including inside a tag.
    #[test]
    fn nothing_is_lost_at_any_boundary() {
        let reply = "<think>abc</think>def";
        for at in 0..=reply.len() {
            if !reply.is_char_boundary(at) {
                continue;
            }
            let mut thinking = Thinking::new();
            let (w1, a1) = thinking.feed(&reply[..at]);
            let (w2, a2) = thinking.feed(&reply[at..]);
            let (w3, a3) = thinking.finish();
            assert_eq!(format!("{w1}{w2}{w3}"), "abc", "cut at {at}");
            assert_eq!(format!("{a1}{a2}{a3}"), "def", "cut at {at}");
        }
    }

    fn whole(reply: &str) -> (String, String) {
        let mut thinking = Thinking::new();
        let (mut working, mut answer) = thinking.feed(reply);
        let (w, a) = thinking.finish();
        working.push_str(&w);
        answer.push_str(&a);
        (working, answer)
    }

    fn by_char(reply: &str) -> (String, String) {
        let mut thinking = Thinking::new();
        let (mut working, mut answer) = (String::new(), String::new());
        for c in reply.chars() {
            let (w, a) = thinking.feed(&c.to_string());
            working.push_str(&w);
            answer.push_str(&a);
        }
        let (w, a) = thinking.finish();
        working.push_str(&w);
        answer.push_str(&a);
        (working, answer)
    }
}
