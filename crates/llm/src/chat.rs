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

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// The tools this assistant turn asked for, if it asked for any.
    ///
    /// Empty and absent are the same thing to a template — `{%- if
    /// message.tool_calls %}` — so an empty list is left out rather than
    /// rendered as one, which keeps a plain conversation's JSON exactly what
    /// it was before tools existed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Which call a `tool` message is the result of. Qwen's template ignores
    /// it and Mistral's prints it; it is the client's id either way, carried
    /// through untouched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// The tool that produced a `tool` message, for the templates that name
    /// it in the transcript.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// One call, in the shape OpenAI gave it and the templates read.
///
/// `arguments` is held as JSON *text*, because that is what OpenAI's wire
/// format carries and what the model wrote — see [`Called`] for why a
/// template is nevertheless shown the object it spells.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCall {
    pub id: String,
    /// Always `function`. The field exists because OpenAI's schema has it and
    /// templates test it; there is no second kind.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: Called,
}

/// A call with no id yet: what the model actually wrote.
///
/// The id is the wire format's, not the model's — it exists so a client can
/// match a result to a call — so it is minted where the wire format is, and
/// this is what the parser produces.
#[derive(Debug, Clone, PartialEq)]
pub struct Called {
    pub name: String,
    /// The arguments as JSON text, e.g. `{"path":"src/main.rs"}`.
    pub arguments: String,
}

/// The arguments reach a template as the *object* they spell, not as the text
/// they arrived in.
///
/// Every Hermes-descended template — Qwen's included — writes a previous call
/// out with `{{ tool_call.arguments | tojson }}`, and `tojson` of a string is
/// a quoted, escaped string. Handing it the text put
/// `"{\"path\": \"main.rs\"}"` in the next turn's prompt, two lines under
/// the instructions saying a call is `{"name": …, "arguments": <args-json-object>}`.
/// The model imitates the example nearest to hand: it wrote its next call
/// double-encoded too, the client sent that back, and the escaping gained a
/// level per turn until a reply stopped parsing as a call at all. What it
/// looked like from outside was an agent that answered with the text of a tool
/// call instead of calling the tool.
///
/// Text that is not JSON is serialised as itself. A model that wrote something
/// unparseable is better represented by what it wrote than by an error here,
/// and the caller still gets the wire format's string either way: the OpenAI
/// response is built from these fields, not from this impl.
impl serde::Serialize for Called {
    fn serialize<S: serde::Serializer>(&self, out: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut call = out.serialize_struct("Called", 2)?;
        call.serialize_field("name", &self.name)?;
        match serde_json::from_str::<serde_json::Value>(&self.arguments) {
            Ok(object) => call.serialize_field("arguments", &object)?,
            Err(_) => call.serialize_field("arguments", &self.arguments)?,
        }
        call.end()
    }
}

impl ToolCall {
    pub fn new(id: impl Into<String>, called: Called) -> Self {
        ToolCall { id: id.into(), kind: "function".into(), function: called }
    }
}

impl Message {
    fn of(role: &str, content: impl Into<String>) -> Self {
        Message { role: role.into(), content: content.into(), ..Message::default() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Message::of("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Message::of("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::of("assistant", content)
    }
    /// An assistant turn that called tools. The content is whatever it said
    /// alongside the calls, which for most models is nothing.
    pub fn calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Message { tool_calls, ..Message::of("assistant", content) }
    }
    /// What a tool answered, on its way back to the model.
    pub fn tool(content: impl Into<String>, id: Option<String>, name: Option<String>) -> Self {
        Message { tool_call_id: id, name, ..Message::of("tool", content) }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    /// The template's own source, kept so that [`ChatTemplate::takes_tools`]
    /// can be answered without rendering anything.
    source: String,
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
        let source = template.clone();
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
            source,
            bos: token_text("bos_token"),
            eos: token_text("eos_token"),
        }))
    }

    /// Whether this model's template has anywhere to put tools.
    ///
    /// A template that never mentions `tools` renders the same string whether
    /// or not any were offered, so the model is never told the tools exist
    /// and cannot call them. That is worth knowing *before* generating a
    /// reply: the alternative is a client that offered a function, waited for
    /// a call, and got a paragraph.
    ///
    /// Read off the source rather than by rendering, because rendering needs
    /// a conversation and this is a fact about the model.
    pub fn takes_tools(&self) -> bool {
        self.source.contains("tools")
    }

    /// Render a conversation into the exact string the model expects.
    ///
    /// `add_generation_prompt` appends the opening marker for the assistant's
    /// turn — the cue that says "your line now".
    ///
    /// `tools` is the list of function schemas the client is offering, in
    /// OpenAI's shape, and goes in verbatim: every template that takes them
    /// dumps them with `tojson` into a system block of its own wording. An
    /// empty list is falsy in Jinja, which is why a conversation with no
    /// tools renders exactly as it did before there were any.
    pub fn render(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
        add_generation_prompt: bool,
    ) -> Res<String> {
        let tmpl = self.env.get_template("chat")?;
        Ok(tmpl.render(minijinja::context! {
            messages => Value::from_serialize(messages),
            tools => Value::from_serialize(tools),
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


// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

/// The tags a model wraps a call in.
///
/// Hermes' format, which Qwen adopted and most open models followed: the call
/// is a JSON object between these, `{"name": …, "arguments": {…}}`. It is not
/// universal — Llama 3.1 emits a bare object, DeepSeek has markers of its own
/// — and this parses the one its templates ask for. A model whose template
/// documents a different format will not be understood, which is a gap worth
/// naming rather than papering over with four half-tested parsers.
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// Pulling tool calls out of a reply as it streams.
///
/// The same problem as [`Thinking`] and the same shape of answer: a tag is
/// several tokens, so text that might still become one is held back and
/// everything else goes out at once. What differs is what happens inside the
/// tags — a trace is text to be forwarded, a call is JSON to be parsed, so
/// the block is buffered whole rather than streamed through.
///
/// # Only when tools were offered
///
/// A model that was offered no tools cannot be calling one, and a reply that
/// contains `<tool_call>` anyway is a model writing about the format — this
/// paragraph would parse as one. So the caller runs this only for a request
/// that offered tools; unlike [`Thinking`], there is no position rule that
/// could tell the two apart, because a real call legitimately follows text.
///
/// # Nothing is dropped
///
/// A block whose JSON does not parse, and a block the token budget cut off
/// before it closed, both come back out as content, tags and all. A client
/// then sees what the model wrote and can say so, which is worth more than a
/// reply that silently lost a paragraph.
pub struct ToolCalls {
    held: String,
    inside: bool,
}

impl Default for ToolCalls {
    fn default() -> Self {
        ToolCalls::new()
    }
}

impl ToolCalls {
    pub fn new() -> ToolCalls {
        ToolCalls { held: String::new(), inside: false }
    }

    /// Feed a piece of the reply. Returns what of it is content, and any
    /// calls that finished in it — both may be empty.
    pub fn feed(&mut self, text: &str) -> (String, Vec<Called>) {
        self.held.push_str(text);
        let (mut content, mut calls) = (String::new(), Vec::new());
        loop {
            match (self.inside, self.held.find(if self.inside { CALL_CLOSE } else { CALL_OPEN })) {
                (false, Some(at)) => {
                    content.push_str(&self.held[..at]);
                    self.held = self.held[at + CALL_OPEN.len()..].to_string();
                    self.inside = true;
                }
                (false, None) => {
                    let keep = self.held.len() - partial(&self.held, CALL_OPEN);
                    content.push_str(&self.held[..keep]);
                    self.held = self.held[keep..].to_string();
                    return (content, calls);
                }
                (true, Some(at)) => {
                    // A second block opening before this one closed, with a
                    // finished call in between: the model wrote the calls and
                    // forgot the closer between them. A 14B Qwen did exactly
                    // that, and taking the open as the closer it left out is
                    // the difference between both calls arriving and neither:
                    // the block would otherwise run to the *next* closer,
                    // swallowing the second call into a body that cannot
                    // parse.
                    //
                    // Only when what came before parses, because `<tool_call>`
                    // inside a block that is not a call is text like any
                    // other, and a model writing about the format writes
                    // exactly that.
                    if let Some(open) = self.held[..at].find(CALL_OPEN) {
                        if let Some(found) = parse_calls(&self.held[..open]) {
                            calls.extend(found);
                            self.held = self.held[open + CALL_OPEN.len()..].to_string();
                            continue;
                        }
                    }
                    let body = self.held[..at].to_string();
                    self.held = self.held[at + CALL_CLOSE.len()..].to_string();
                    self.inside = false;
                    match parse_calls(&body) {
                        Some(found) => calls.extend(found),
                        None => content.push_str(&format!("{CALL_OPEN}{body}{CALL_CLOSE}")),
                    }
                }
                // An open block: hold everything until it closes. Nothing of
                // a call can be sent early — half a JSON object is not half
                // an answer, it is nothing.
                (true, None) => return (content, calls),
            }
        }
    }

    /// End of the reply: whatever is still held back.
    pub fn finish(&mut self) -> (String, Vec<Called>) {
        let rest = std::mem::take(&mut self.held);
        if !std::mem::replace(&mut self.inside, false) {
            return (rest, Vec::new());
        }
        // The same rule as in `feed`, for a reply that ended without ever
        // closing: every finished call before an open that follows it is a
        // call the model wrote and did not close.
        let mut calls = Vec::new();
        let mut rest = rest.as_str();
        while let Some(open) = rest.find(CALL_OPEN) {
            match parse_calls(&rest[..open]) {
                Some(found) => {
                    calls.extend(found);
                    rest = &rest[open + CALL_OPEN.len()..];
                }
                None => break,
            }
        }
        // A call the budget cut off. Occasionally the model wrote the whole
        // object and only the closing tag is missing, which is a call; more
        // often it is a fragment, which is text.
        match parse_calls(rest) {
            Some(found) => {
                calls.extend(found);
                (String::new(), calls)
            }
            None => (format!("{CALL_OPEN}{rest}"), calls),
        }
    }

    /// Whether a call was opened and never closed.
    pub fn unfinished(&self) -> bool {
        self.inside
    }
}

/// One `<tool_call>` block's contents, if they are calls.
///
/// Usually one object. A model that means to make two calls sometimes writes
/// both into one block with nothing between them but a newline — the opening
/// tag is the token it drops, and it drops the same one whether there is a
/// closing tag after the first call or not. Both objects are whole, so both
/// are calls, and reading only the first would lose the second as surely as
/// reading neither.
///
/// It stays an all-or-nothing answer: every value in the body has to parse
/// and every one has to be a call, so a block that is a call followed by a
/// sentence is still text, and so is a block that is a sentence. The caller
/// hands those back with their tags on, which is what lets a client show what
/// the model actually wrote.
fn parse_calls(body: &str) -> Option<Vec<Called>> {
    let values = serde_json::Deserializer::from_str(body.trim()).into_iter::<serde_json::Value>();
    let mut calls = Vec::new();
    for value in values {
        calls.push(parse_call(&value.ok()?)?);
    }
    (!calls.is_empty()).then_some(calls)
}

/// One object, if it is a call.
///
/// `arguments` comes back as text in every case, because that is what the
/// wire format carries. An object is re-serialised — compactly, and this is
/// the one place the spelling changes — and a string is passed through,
/// which is the double-encoding some fine-tunes emit.
fn parse_call(value: &serde_json::Value) -> Option<Called> {
    let name = value.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = match value.get("arguments") {
        // A call with no arguments is a call. The field is required by the
        // format and left out by models anyway.
        None => "{}".to_string(),
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    };
    Some(Called { name, arguments })
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

    // -- tool calls ---------------------------------------------------------

    /// The stream is the case this exists for, so the two readings have to
    /// agree: a tag is several tokens and arrives in pieces.
    #[test]
    fn a_call_is_found_the_same_however_it_arrives() {
        let reply = "Let me look.\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"src/main.rs\"}}\n</tool_call>";
        let (content, calls) = calls_of(reply);
        assert_eq!(content, "Let me look.\n");
        assert_eq!(
            calls,
            vec![Called { name: "read".into(), arguments: r#"{"path":"src/main.rs"}"#.into() }]
        );
        assert_eq!(calls_by_char(reply), (content, calls), "the stream parsed differently");
    }

    /// Models offered several tools answer with several calls, and a client
    /// that got only the first would run half the turn.
    #[test]
    fn every_call_in_a_reply_comes_back() {
        let reply = "<tool_call>{\"name\": \"ls\", \"arguments\": {}}</tool_call>\
                     <tool_call>{\"name\": \"cat\", \"arguments\": {\"f\": 1}}</tool_call>";
        let (content, calls) = calls_of(reply);
        assert!(content.is_empty(), "{content:?}");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1], Called { name: "cat".into(), arguments: r#"{"f":1}"#.into() });
        assert_eq!(calls_by_char(reply).1, calls);
    }

    /// Arguments are text on the wire whichever way the model wrote them:
    /// an object is compacted, and a string that is itself JSON — which
    /// some fine-tunes emit — is passed through as it stands.
    #[test]
    fn arguments_are_always_json_text() {
        let (_, object) = calls_of(r#"<tool_call>{"name": "f", "arguments": {"a": [1, 2]}}</tool_call>"#);
        assert_eq!(object[0].arguments, r#"{"a":[1,2]}"#);

        let (_, double) = calls_of(r#"<tool_call>{"name": "f", "arguments": "{\"a\": 1}"}</tool_call>"#);
        assert_eq!(double[0].arguments, r#"{"a": 1}"#);

        let (_, none) = calls_of(r#"<tool_call>{"name": "f"}</tool_call>"#);
        assert_eq!(none[0].arguments, "{}");
    }

    /// A block that is not a call is text. Nothing a model wrote is dropped
    /// because this module failed to understand it.
    #[test]
    fn a_block_that_does_not_parse_is_content() {
        for reply in [
            "<tool_call>{\"name\": \"f\", </tool_call>",   // cut off
            "<tool_call>not json at all</tool_call>",      // never was
            "<tool_call>{\"arguments\": {}}</tool_call>",   // nothing named
        ] {
            let (content, calls) = calls_of(reply);
            assert!(calls.is_empty(), "{reply:?} parsed as a call");
            assert_eq!(content, reply, "{reply:?} lost text");
            assert_eq!(calls_by_char(reply).0, reply);
        }
    }

    /// The budget ran out mid-call. If the object is whole it is a call, and
    /// otherwise it is the text the model got as far as writing.
    #[test]
    fn a_call_that_never_closes() {
        let mut parser = ToolCalls::new();
        let (content, calls) = parser.feed("<tool_call>{\"name\": \"ls\", \"arguments\": {}}");
        assert!(content.is_empty() && calls.is_empty(), "a call was sent before it closed");
        assert!(parser.unfinished());
        assert_eq!(parser.finish().1, vec![Called { name: "ls".into(), arguments: "{}".into() }]);

        let mut cut = ToolCalls::new();
        cut.feed("<tool_call>{\"name\": \"l");
        assert_eq!(cut.finish(), ("<tool_call>{\"name\": \"l".to_string(), Vec::new()));
    }

    /// Two calls in one block, with no tags at all between them. The same
    /// omission as the test below and one token further: the model drops the
    /// opening tag of the second call, and here it had already written the
    /// closing tag of neither.
    #[test]
    fn two_calls_in_one_block() {
        let reply = "I will fix it and check it.\n\
                     <tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"main.rs\"}}\n\
                     {\"name\": \"bash\", \"arguments\": {\"command\": \"cargo fmt\"}}\n\
                     </tool_call>";
        let (content, calls) = calls_of(reply);
        assert_eq!(content, "I will fix it and check it.\n");
        assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["edit", "bash"]);
        assert_eq!(calls_by_char(reply), calls_of(reply), "the stream split differently");

        // All of the body or none of it. A call and then a sentence is a
        // model talking about a call, and it comes back with its tags on.
        let mixed = "<tool_call>{\"name\": \"f\", \"arguments\": {}} and then run it</tool_call>";
        assert!(calls_of(mixed).1.is_empty(), "{:?}", calls_of(mixed));
        assert_eq!(calls_of(mixed).0, mixed);
    }

    /// Two calls and one closing tag, which is what a model writes when it
    /// forgets the one in between. Both calls are there to be had, and reading
    /// the block through to the *next* closer lost both: what sat between the
    /// tags was then two objects with a tag in the middle, which parses as
    /// nothing and came back out as text. An agent saw its own tool call
    /// printed at it instead of run.
    #[test]
    fn a_closing_tag_left_out_between_two_calls() {
        let reply = "I will fix it and run it.\n\
                     <tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"main.rs\"}}\n\
                     <tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"cargo run\"}}\n\
                     </tool_call>";
        let (content, calls) = calls_of(reply);
        assert_eq!(content, "I will fix it and run it.\n");
        assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["edit", "bash"]);
        assert_eq!(calls_by_char(reply), calls_of(reply), "the stream split differently");

        // The same reply with the last closer gone too, which is what the
        // budget running out looks like.
        let cut = reply.strip_suffix("\n</tool_call>").unwrap();
        assert_eq!(calls_of(cut).1.len(), 2, "{:?}", calls_of(cut));

        // And the rule stays narrow: a block that is not a call is text, so
        // a model writing *about* the format still gets its words back.
        let prose = "<tool_call>as in <tool_call>{\"name\": \"f\"}</tool_call>";
        assert!(calls_of(prose).1.is_empty(), "{:?}", calls_of(prose));
        assert_eq!(calls_of(prose).0, prose);
    }

    /// A streamed reply may be cut anywhere, including inside either tag and
    /// inside the JSON. Every character has to end up on one side or the
    /// other, and the call has to be the same call.
    #[test]
    fn no_split_changes_the_answer() {
        let reply = "before<tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1}}</tool_call>after";
        let (whole_content, whole_calls) = calls_of(reply);
        for at in 0..=reply.len() {
            if !reply.is_char_boundary(at) {
                continue;
            }
            let mut parser = ToolCalls::new();
            let (c1, mut calls) = parser.feed(&reply[..at]);
            let (c2, more) = parser.feed(&reply[at..]);
            let (c3, last) = parser.finish();
            calls.extend(more);
            calls.extend(last);
            assert_eq!(format!("{c1}{c2}{c3}"), whole_content, "cut at {at}");
            assert_eq!(calls, whole_calls, "cut at {at}");
        }
    }

    /// Tools reach the model, and what the model called reaches the next
    /// turn's prompt. Rendered against a template of this file's own rather
    /// than a downloaded one, so the test says what it depends on.
    #[test]
    fn a_template_is_given_the_tools_and_the_calls() {
        let template = "\
            {%- if tools %}TOOLS:{% for t in tools %} {{ t.function.name }} {{ t | tojson }}{% endfor %}\n{% endif %}\
            {%- for m in messages %}\
            {{- m.role }}: {{ m.content }}\
            {%- for c in m.tool_calls %} CALL {{ c.function.name }}{{ c.function.arguments | tojson }}{% endfor %}\
            {%- if m.tool_call_id %} (for {{ m.tool_call_id }}){% endif %}\n\
            {%- endfor %}\
            {%- if add_generation_prompt %}assistant:{% endif %}";
        let dir = std::env::temp_dir().join(format!("kvad-tooltmpl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer_config.json");
        std::fs::write(&path, serde_json::json!({ "chat_template": template }).to_string()).unwrap();
        let chat = ChatTemplate::from_tokenizer_config(&path).unwrap().expect("no template read");
        std::fs::remove_dir_all(&dir).ok();

        assert!(chat.takes_tools());
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": { "type": "object" },
            },
        })];
        let conversation = [
            Message::user("what is in main.rs?"),
            Message::calls(
                "",
                vec![ToolCall::new(
                    "call_1",
                    Called { name: "read".into(), arguments: r#"{"path":"main.rs"}"#.into() },
                )],
            ),
            Message::tool("fn main() {}", Some("call_1".into()), Some("read".into())),
        ];

        let with = chat.render(&conversation, &tools, true).unwrap();
        assert!(with.starts_with("TOOLS: read "), "the tools never reached the template: {with}");
        // Through `tojson`, which is how every real template writes them out
        // and which is a minijinja feature this crate has to ask for. The
        // filter is missing from a default build and the render then fails
        // at request time, on the one path a conversation without tools
        // never reaches.
        assert!(with.contains("parameters"), "the schema was not rendered: {with}");
        // In the order the client wrote it, which is the order the model was
        // tuned on. Both minijinja and serde_json sort a map's keys unless
        // told not to, and sorted is a different prompt: `type` last,
        // `description` before `name`. Same schema, and not the same text.
        assert!(
            with.contains(
                r#"{"type":"function","function":{"name":"read","description":"Read a file""#
            ),
            "the schema's keys were reordered: {with}"
        );
        // Through `tojson` as well, and this is the assertion that matters:
        // the filter has to be handed the object, because `tojson` of the
        // *text* is a quoted escaped string and the model copies whatever
        // shape it is shown. See `impl Serialize for Called`.
        assert!(with.contains(r#"CALL read{"path":"main.rs"}"#), "{with}");
        assert!(!with.contains(r#"\""#), "the arguments were double-encoded: {with}");
        assert!(with.contains("tool: fn main() {} (for call_1)"), "{with}");
        assert!(with.ends_with("assistant:"), "{with}");

        // And with none offered the model is told about none — an empty list
        // is falsy in Jinja, which is what keeps a plain conversation's
        // prompt byte-for-byte what it was before tools existed.
        let without = chat.render(&conversation, &[], true).unwrap();
        assert!(!without.contains("TOOLS"), "{without}");
    }

    fn calls_of(reply: &str) -> (String, Vec<Called>) {
        let mut parser = ToolCalls::new();
        let (mut content, mut calls) = parser.feed(reply);
        let (rest, last) = parser.finish();
        content.push_str(&rest);
        calls.extend(last);
        (content, calls)
    }

    fn calls_by_char(reply: &str) -> (String, Vec<Called>) {
        let mut parser = ToolCalls::new();
        let (mut content, mut calls) = (String::new(), Vec::new());
        for c in reply.chars() {
            let (text, found) = parser.feed(&c.to_string());
            content.push_str(&text);
            calls.extend(found);
        }
        let (rest, last) = parser.finish();
        content.push_str(&rest);
        calls.extend(last);
        (content, calls)
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
