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
