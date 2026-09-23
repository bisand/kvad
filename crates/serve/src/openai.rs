//! `/v1/models` and `/v1/chat/completions`, shaped the way OpenAI's are.
//!
//! The shape is somebody else's and is not ours to improve. Anything that
//! already speaks to OpenAI — a script, an editor plugin, a library — speaks
//! to this, and that is worth more than any field we might prefer.
//!
//! What we add, we add in one place: a `kvad` object beside `usage`, carrying
//! the numbers this engine knows and OpenAI does not — how much of the prompt
//! came out of the KV cache, how long prefill took as against decode, and
//! which backend ran it. A client that does not know about it ignores it.
//!
//! # The model field
//!
//! OpenAI's `model` selects from many that are all warm. Here several can be
//! in memory at once — see [`crate::scheduler`] — and `model` names one of
//! them, as a repo id or as `repo@backend` for a particular one.
//!
//! A model on this disk that is not in memory is loaded, if it fits beside
//! what is, and the request waits the tens of seconds that takes. Nothing is
//! ever unloaded to make room. A model that does not fit is refused with a
//! 409 that names the models holding the memory, because a client that
//! asks for a model and gets somebody else's answer is worse than a client
//! that is told no — and a client whose model vanished because another
//! request named a different one is worse than both. A model that is not on
//! this disk is a 404, and is not downloaded on the way: pulling several
//! gigabytes is not something a completion should start.
//!
//! Omitting `model` is allowed when exactly one model is in memory. With
//! several, the server would be guessing.
//!
//! # Tools
//!
//! `tools` are passed to the model's own chat template, which writes them
//! into a system block in whatever wording that model was trained on, and the
//! calls it writes back are parsed out of the reply as it streams — see
//! [`kvad::chat::ToolCalls`]. What arrives at the client is OpenAI's shape:
//! `tool_calls` on the message, `finish_reason` of `tool_calls`, and the same
//! ids handed back on the `tool` messages of the next turn.
//!
//! Two refusals rather than two pretences. A model whose template has no
//! place for tools cannot be offered them: the tools would vanish silently
//! and the reply would be a paragraph where the client expected a call. And
//! `tool_choice` beyond `auto` and `none` — `required`, or a named function —
//! would need the sampler to be constrained to the tokens that open a call,
//! which is real work and is not done here.

use crate::api::{blocking, Fail};
use crate::models::{sse, stream};
use crate::auth::{Identity, State};
use crate::scheduler::{LoadError, Piece, Resident};
use axum::extract::State as St;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::Json;
use kvad::chat::Message;
use kvad::runtime::{Chosen, Stats};
use kvad::service::Sampling;
use serde_json::json;

/// Every model on this disk that could be asked for, with the ones in memory
/// marked.
///
/// Not only the ones in memory, because naming one that is not loads it when
/// it fits; see the module docs. A model in memory is listed a second time
/// under `repo@backend`, which is how a client reaches that one when the same
/// weights are in memory twice.
pub async fn models(_: Identity, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let found = blocking(move || {
        let listed = |trained: bool| {
            move |m: kvad::hub::LocalModel| (m.id.clone(), trained, takes_tools(&m))
        };
        let mut all: Vec<(String, bool, bool)> = kvad::hub::local_models()
            .into_iter()
            .filter(|m| m.complete && m.arch.is_some())
            .map(listed(false))
            .collect();
        all.extend(
            kvad::hub::trained_models()
                .into_iter()
                .filter(|m| m.complete && m.arch.is_some())
                .map(listed(true)),
        );
        Ok(all)
    })
    .await?;

    let residents = state.engine.residents();
    let resident = |id: &str| residents.iter().any(|r| r.key.repo.eq_ignore_ascii_case(id));
    let mut data: Vec<serde_json::Value> = found
        .into_iter()
        .map(|(id, trained, tools)| json!({
            "id": id,
            "object": "model",
            // The field is required and means "when was this published".
            // Nothing here was published, so it is zero rather than a number
            // invented to look like a date.
            "created": 0,
            "owned_by": if trained { "kvad" } else { "huggingface" },
            // OpenAI's model object says nothing about what a model can do,
            // because there the answer is in the documentation. Here it is a
            // property of the checkpoint on this disk, and a client picking a
            // model for an agent needs it before it picks — so it goes in the
            // extension field, beside the one on a completion. Whether it is
            // in memory goes there too: naming one that is not costs a load.
            "kvad": { "tools": tools, "resident": resident(&id) },
        }))
        .collect();
    data.extend(residents.iter().map(|r| json!({
        "id": r.id,
        "object": "model",
        "created": 0,
        "owned_by": "kvad",
        "kvad": { "tools": r.model.tools, "resident": true, "backend": r.model.backend },
    })));

    Ok(Json(json!({ "object": "list", "data": data })))
}

/// Whether a model on disk could be offered tools, read from its template.
///
/// Costs a small JSON read and a template compile per model, on a route that
/// is already doing a directory walk. The alternative is loading the model,
/// which is tens of seconds and the thing a client is consulting this list to
/// avoid.
fn takes_tools(model: &kvad::hub::LocalModel) -> bool {
    kvad::hub::model_file(&model.path, "tokenizer_config.json")
        .and_then(|p| kvad::chat::ChatTemplate::from_tokenizer_config(&p).ok().flatten())
        .is_some_and(|t| t.takes_tools())
}

#[derive(serde::Deserialize)]
pub struct Completions {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Turn>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Not OpenAI's; this engine samples with it and there is nowhere else to
    /// put it. A client that does not send one gets the default.
    #[serde(default)]
    top_k: Option<usize>,
    /// The functions the client is offering this turn, in OpenAI's shape.
    /// Passed to the model's template untouched: every template that takes
    /// them dumps them with `tojson`, so a field this server has never heard
    /// of still reaches the model.
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    /// `auto` or `none`. See the module docs for why the other two are
    /// refused rather than ignored.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// What OpenAI renamed `max_tokens` to, and what the newer clients send.
    /// Both are accepted and mean the same thing here; a client that sends
    /// both gets `max_tokens`, which is the one it also sent on purpose.
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// Answer from this dataset: search it with the last thing the user
    /// said, and put what comes back in front of the model.
    ///
    /// Not OpenAI's either. It is the answer to "can I add knowledge to a
    /// model" that does not involve training one: the model learns nothing
    /// and reads something.
    #[serde(default)]
    dataset: Option<i64>,
}

/// How many passages to look for. The sixth is rarely better than the first
/// and always costs the same.
const GROUNDING_CHUNKS: usize = 5;

/// How much of a corpus to put in front of the model, in characters.
///
/// A share of what the model can hold rather than a fixed number: 4,000
/// characters is a comfortable fifth of a 2k context and an unusable
/// thirtieth of a 32k one. Found by measuring — the section of the book that
/// answers what happens to a reference after a push is 4,096 characters, and
/// a flat budget of 4,000 cut the answer off the end of it.
///
/// Four characters to the token, and two fifths of the window: the rest is
/// the conversation so far and the reply, which have to fit as well. Capped,
/// because a 32k model would otherwise be handed fifty thousand characters —
/// which costs prefill on every turn and buries the passage that matters
/// among four that do not.
fn grounding_budget(n_ctx: usize) -> usize {
    (n_ctx * 4 * 2 / 5).clamp(2_000, 12_000)
}

#[derive(serde::Deserialize)]
pub struct Turn {
    role: String,
    /// Absent on an assistant turn that did nothing but call a tool, where
    /// OpenAI sends `null`.
    #[serde(default)]
    content: Option<Content>,
    #[serde(default)]
    tool_calls: Vec<RequestedCall>,
    /// On a `tool` message: which call this is the result of.
    #[serde(default)]
    tool_call_id: Option<String>,
    /// On a `tool` message: which tool produced it.
    #[serde(default)]
    name: Option<String>,
}

/// A message's content: a string, or the list of parts a client sends when
/// there is more than one thing in the turn.
///
/// The second spelling is not an improvement on the first, it is what the
/// mainstream client libraries emit as soon as a message has an attachment
/// in it, and a server that took only strings would refuse them for a
/// difference the model never sees.
#[derive(serde::Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(serde::Deserialize)]
pub struct Part {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

impl Content {
    /// The text of it, or a refusal naming what could not be read.
    ///
    /// Anything that is not text — an image, audio, a file — is refused
    /// rather than dropped: this engine has no vision tower, and a client
    /// that sent a picture and got an answer about the sentence beside it
    /// would have no way of telling.
    fn text(&self) -> Result<String, Fail> {
        match self {
            Content::Text(text) => Ok(text.clone()),
            Content::Parts(parts) => {
                if let Some(other) = parts.iter().find(|p| p.kind != "text") {
                    return Err(Fail::bad(format!(
                        "a message part of type `{}`: this engine reads text",
                        other.kind
                    )));
                }
                Ok(parts.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join(""))
            }
        }
    }
}

/// A call on the way *back* in: what the assistant said last turn, echoed by
/// the client so the model can see its own call beside the result.
#[derive(serde::Deserialize)]
pub struct RequestedCall {
    #[serde(default)]
    id: Option<String>,
    function: RequestedFunction,
}

#[derive(serde::Deserialize)]
pub struct RequestedFunction {
    name: String,
    /// JSON text by the specification. Some clients send the object itself,
    /// which is the same information and is accepted as it arrives.
    #[serde(default)]
    arguments: serde_json::Value,
}

impl Completions {
    fn sampling(&self) -> Result<Sampling, Fail> {
        let d = Sampling::default();
        let temperature = self.temperature.unwrap_or(d.temperature);
        let top_p = self.top_p.unwrap_or(d.top_p);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(Fail::bad("temperature is between 0 and 2; 0 is greedy"));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(Fail::bad("top_p is between 0 and 1"));
        }
        let max_tokens = self.max_tokens.or(self.max_completion_tokens).unwrap_or(d.max_tokens);
        if max_tokens == 0 {
            return Err(Fail::bad("max_tokens of 0 would generate nothing"));
        }
        Ok(Sampling {
            temperature,
            top_p,
            top_k: self.top_k.unwrap_or(d.top_k),
            seed: self.seed,
            max_tokens: max_tokens.min(8192),
        })
    }

    fn turns(&self) -> Result<Vec<Message>, Fail> {
        if self.messages.is_empty() {
            return Err(Fail::bad("a completion needs at least one message"));
        }
        self.messages
            .iter()
            .enumerate()
            .map(|(i, t)| {
                // A turn that only called a tool has no content, and that is
                // not the same as an empty one: the template renders the
                // calls and nothing else.
                let content = match &t.content {
                    Some(c) => c.text()?,
                    None => String::new(),
                };
                match t.role.as_str() {
                    "system" => Ok(Message::system(content)),
                    "user" => Ok(Message::user(content)),
                    "assistant" if t.tool_calls.is_empty() => Ok(Message::assistant(content)),
                    "assistant" => Ok(Message::calls(content, t.calls(i))),
                    "tool" => Ok(Message::tool(content, t.tool_call_id.clone(), t.name.clone())),
                    other => Err(Fail::bad(format!("`{other}` is not a role a message can have"))),
                }
            })
            .collect()
    }

    /// The tools to offer the model, once `tool_choice` has had its say.
    ///
    /// Refusals rather than silent nonsense: a tool with no name is not a
    /// tool, and a `tool_choice` this engine cannot honour is told so here
    /// instead of being discovered by a client waiting for a call that was
    /// never going to come.
    fn tools(&self) -> Result<Vec<serde_json::Value>, Fail> {
        match self.tool_choice.as_ref().and_then(|c| c.as_str()) {
            // `none` means the tools are context, not an invitation. Dropping
            // them entirely is the honest reading: the model is told about no
            // tools and so calls none.
            Some("none") => return Ok(Vec::new()),
            None | Some("auto") => {}
            Some(other) => {
                return Err(Fail::bad(format!(
                    "tool_choice `{other}` would need the sampler constrained to a call; \
                     this engine offers `auto` and `none`"
                )))
            }
        }
        // An object is a named function, which is the same refusal.
        if self.tool_choice.as_ref().is_some_and(|c| c.is_object()) {
            return Err(Fail::bad(
                "naming a tool in tool_choice would need the sampler constrained to a call; \
                 this engine offers `auto` and `none`",
            ));
        }
        for tool in &self.tools {
            let named = tool
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .is_some_and(|n| !n.trim().is_empty());
            if !named {
                return Err(Fail::bad("every tool needs a `function.name`"));
            }
        }
        Ok(self.tools.clone())
    }
}

impl Turn {
    /// This turn's calls, in the engine's shape.
    ///
    /// A client that left an id out gets one made up, because the template
    /// may print it and the next turn's `tool` message will refer to it. The
    /// message's position makes it unique within the conversation, which is
    /// as far as an id has to reach.
    fn calls(&self, at: usize) -> Vec<kvad::chat::ToolCall> {
        self.tool_calls
            .iter()
            .enumerate()
            .map(|(n, c)| {
                let id = c.id.clone().unwrap_or_else(|| format!("call_{at}_{n}"));
                let arguments = match &c.function.arguments {
                    serde_json::Value::Null => "{}".to_string(),
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                kvad::chat::ToolCall::new(
                    id,
                    kvad::chat::Called { name: c.function.name.clone(), arguments },
                )
            })
            .collect()
    }
}

pub async fn completions(
    who: Identity,
    St(state): St<State>,
    Json(body): Json<Completions>,
) -> Result<Response, Fail> {
    let sampling = body.sampling()?;
    let mut turns = body.turns()?;
    let tools = body.tools()?;


    let resident = resident_for(&state, body.model.as_deref()).await?;
    let loaded = resident.model.clone();

    // Asked for a call from a model that cannot make one. The template is
    // where tools live, so a model whose template never mentions them would
    // be handed the conversation with the tools quietly missing and would
    // answer in prose — which a client cannot tell apart from a model that
    // considered the tools and declined.
    if !tools.is_empty() && !loaded.tools {
        return Err(Fail::bad(format!(
            "{}'s chat template has no place for tools, so it cannot be asked to call one",
            loaded.repo
        )));
    }

    // Retrieval goes in front of everything else the conversation says, so
    // that a system prompt the user wrote still has the last word on tone.
    // After the load, because how much may be put in front of the model is a
    // fact about the model.
    if let Some(dataset) = body.dataset {
        // Everything else that reads a corpus is behind `Admin` — the
        // preview, the search, the listing — and an answer built out of one
        // is that corpus read aloud. A different gate here would be a way
        // around the others.
        if !who.is_admin() {
            return Err(Fail::denied("answering from a dataset is an administrator's to ask for"));
        }
        // The turns rather than the request's own messages: by here the
        // content has been read out of whichever shape it arrived in, and
        // searching a corpus with the wrong one of the two would be a bug
        // that only showed up for one kind of client.
        let question = turns
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .ok_or_else(|| Fail::bad("answering from a dataset needs a question to search it with"))?;
        let db = state.db.clone();
        let budget = grounding_budget(loaded.n_ctx);
        let found = blocking(move || {
            crate::retrieval::grounding(&db, dataset, &question, GROUNDING_CHUNKS, budget)
        })
        .await
        .map_err(|e| Fail::bad(e.1))?;
        // Nothing matched: the model is told nothing rather than told that
        // nothing was found, and answers as it otherwise would.
        if let Some(grounding) = found {
            turns.insert(0, Message::system(&grounding));
        }
    }

    let offered = !tools.is_empty();
    let pieces =
        state.engine.chat(&resident.key, turns, tools, sampling, false).map_err(Fail::internal)?;
    let id = format!("chatcmpl-{}", now_millis());
    let metrics = state.metrics.clone();
    // Timed from here rather than by the middleware: for a streamed reply the
    // middleware sees only the moment the headers went out, which is the
    // moment before all the work.
    let started = std::time::Instant::now();

    let mut response = match body.stream {
        true => streamed(id, loaded, pieces, metrics, started, offered).into_response(),
        false => whole(id, loaded, pieces, metrics, started, offered).await?.into_response(),
    };
    response.extensions_mut().insert(crate::watching::RecordedItself);
    Ok(response)
}

/// The model in memory a request is for, loading it first if it may be.
async fn resident_for(state: &State, model: Option<&str>) -> Result<Resident, Fail> {
    let Some(name) = model else {
        if let Some(only) = state.engine.only() {
            return Ok(only);
        }
        let all = state.engine.residents();
        return match all.len() {
            0 => Err(Fail::bad("no model is loaded, and the request did not name one")),
            _ => Err(Fail::bad(format!(
                "{} models are loaded and the request did not name one: {}",
                all.len(),
                all.iter().map(|r| r.id.as_str()).collect::<Vec<_>>().join(", ")
            ))),
        };
    };
    if let Some(r) = state.engine.find(name) {
        return Ok(r);
    }

    let (repo, backend) = crate::scheduler::parse_id(name);
    let repo = repo.to_string();
    let (on_disk, default) = {
        let repo = repo.clone();
        blocking(move || {
            let here = kvad::hub::find_local(&repo)
                .or_else(|| kvad::hub::find_trained(&repo))
                .is_some_and(|m| m.complete);
            // The same answer the Models page would give, from the same
            // function: a load started from here and a load started from
            // there must not disagree about what "the default backend"
            // means.
            Ok((here, crate::models::default_backend(Some(&repo))))
        })
        .await?
    };
    if !on_disk {
        return Err(Fail::missing(format!("{name} is not a model on this machine")));
    }
    if !state.load_on_request {
        return Err(Fail::conflict(format!(
            "{name} is not loaded, and this server loads models only when somebody asks it to"
        )));
    }
    let backend = backend
        .or_else(|| crate::engine::parse(&default))
        .unwrap_or(kvad::service::Backend::Cpu(kvad::quant::Precision::Q8));
    let (progress, _ignored) = tokio::sync::mpsc::channel(1);
    match state.engine.load(repo.clone(), backend, progress).await {
        Ok(_) => {}
        Err(LoadError::Full(why)) => return Err(Fail::conflict(why)),
        Err(LoadError::Failed(why)) => return Err(Fail::bad(format!("could not load {name}: {why}"))),
    }
    state
        .engine
        .find(&crate::scheduler::id_of(&repo, backend))
        .ok_or_else(|| Fail::internal(format!("{name} loaded and then was not there")))
}

/// Where a completion's row is filed. The route pattern, matching what the
/// middleware would have recorded.
const ROUTE: &str = "/v1/chat/completions";

/// What the dashboard's sparklines are made of.
///
/// Prefill is time-to-first-token: the wait before anything appears, which is
/// what somebody watching an empty reply box actually experiences, and a
/// different complaint from a reply that arrives slowly.
fn measured(stats: &Stats) -> crate::metrics::Generation {
    crate::metrics::Generation {
        prompt_tokens: stats.prompt_tokens as u32,
        cached_tokens: stats.cached_tokens as u32,
        generated_tokens: stats.generated_tokens as u32,
        decode_per_sec: stats.tokens_per_sec() as f64,
        ttft_millis: stats.prefill_secs as f64 * 1000.0,
    }
}

/// Stop whatever the engine is generating.
///
/// A `DELETE` with no body, because there is one generation running at a
/// time across every model in memory, and the request that started it is
/// the one holding the stream. When several can run at once this grows an
/// id.
pub async fn cancel(_: Identity, St(state): St<State>) -> Json<serde_json::Value> {
    state.engine.cancel();
    Json(json!({ "cancelled": true }))
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn now_secs() -> u64 {
    (now_millis() / 1000) as u64
}

/// The numbers OpenAI has no field for.
fn extension(stats: &Stats, loaded: &crate::scheduler::Loaded) -> serde_json::Value {
    json!({
        "backend": loaded.backend,
        "cached_tokens": stats.cached_tokens,
        "prefill_secs": stats.prefill_secs,
        "decode_secs": stats.decode_secs,
        "prefill_tokens_per_sec":
            (stats.prompt_tokens - stats.cached_tokens) as f32 / stats.prefill_secs.max(1e-6),
        "decode_tokens_per_sec": stats.tokens_per_sec(),
    })
}

/// A tool call's id.
///
/// Unique within the conversation, which is the whole job: the client hands
/// the same id back on the `tool` message carrying the result. Derived from
/// the completion's own id so that a call can be traced to the reply it came
/// from in a log.
fn call_id(completion: &str, index: usize) -> String {
    format!("call_{}_{index}", completion.trim_start_matches("chatcmpl-"))
}

/// One call in OpenAI's shape. `index` is what a streaming client uses to
/// assemble the deltas; a whole reply carries it too, and it is ignored.
fn wire_call(completion: &str, index: usize, call: &kvad::chat::Called) -> serde_json::Value {
    json!({
        "index": index,
        "id": call_id(completion, index),
        "type": "function",
        "function": { "name": call.name, "arguments": call.arguments },
    })
}

fn usage(stats: &Stats) -> serde_json::Value {
    json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
    })
}

fn streamed(
    id: String,
    loaded: crate::scheduler::Loaded,
    mut pieces: tokio::sync::mpsc::Receiver<Piece>,
    metrics: std::sync::Arc<crate::metrics::Metrics>,
    started: std::time::Instant,
    // `offered` is whether this request offered tools, and so whether
    // `<tool_call>` in the reply is a call rather than a model writing
    // about one.
    offered: bool,
) -> impl IntoResponse {
    let (events, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let created = now_secs();

    tokio::spawn(async move {
        let chunk = |delta: serde_json::Value, finish: Option<&str>, extra: serde_json::Value| {
            let mut body = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": loaded.repo,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            });
            if let (Some(obj), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
                obj.extend(extra.clone());
            }
            body
        };

        // The first chunk announces the role and carries no text, which is
        // what OpenAI's stream does and what clients look for.
        let mut opened = false;
        // A reasoning model's working, told apart from its answer as it
        // arrives. See `kvad::chat::Thinking` for why this cannot be a split
        // at the end.
        let mut thinking = kvad::chat::Thinking::new();
        // And the calls out of what is left, when any were offered.
        let mut parsing = offered.then(kvad::chat::ToolCalls::new);
        // How many calls have gone out, which is also the next one's index.
        let mut sent = 0usize;
        let completion = id.clone();
        let mut delta_of = move |working: &str, answer: &str, calls: &[serde_json::Value]| {
            let mut delta = json!({});
            if !working.is_empty() {
                delta["reasoning_content"] = json!(working);
            }
            if !answer.is_empty() {
                delta["content"] = json!(answer);
            }
            if !calls.is_empty() {
                delta["tool_calls"] = json!(calls);
            }
            if delta.as_object().is_some_and(|d| d.is_empty()) {
                return None;
            }
            if !std::mem::replace(&mut opened, true) {
                delta["role"] = json!("assistant");
            }
            Some(delta)
        };

        while let Some(piece) = pieces.recv().await {
            let event = match piece {
                // A `Chose` is a token that also says what it was chosen
                // from. Chat never asks for that — it is the playground's
                // request — so the two are the same thing here.
                Piece::Token(text) | Piece::Chose(Chosen { text, .. }) => {
                    let (working, answer) = thinking.feed(&text);
                    let (answer, found) = match &mut parsing {
                        Some(p) => p.feed(&answer),
                        None => (answer, Vec::new()),
                    };
                    // A call goes out whole, once its closing tag has
                    // arrived. OpenAI's format allows the arguments to be
                    // streamed in fragments; there is nothing to stream
                    // here, because the JSON had to be complete before it
                    // could be known to be a call at all.
                    let calls: Vec<_> = found
                        .iter()
                        .map(|c| {
                            let one = wire_call(&completion, sent, c);
                            sent += 1;
                            one
                        })
                        .collect();
                    // All empty means the text is held back as a possible
                    // tag, and there is nothing to send yet.
                    let Some(delta) = delta_of(&working, &answer, &calls) else { continue };
                    Event::default().data(chunk(delta, None, json!({})).to_string())
                }
                Piece::Done(stats) => {
                    metrics.record_generation(
                        "POST",
                        ROUTE,
                        200,
                        started.elapsed(),
                        measured(&stats),
                    );
                    // Anything still held back — a trace the budget cut off
                    // mid-thought, a call whose closing tag never came —
                    // goes before the chunk that ends the stream, rather
                    // than being dropped with it.
                    let (working, answer) = thinking.finish();
                    let (mut answer, mut found) = match &mut parsing {
                        Some(p) => p.feed(&answer),
                        None => (answer, Vec::new()),
                    };
                    if let Some(p) = &mut parsing {
                        let (rest, last) = p.finish();
                        answer.push_str(&rest);
                        found.extend(last);
                    }
                    let calls: Vec<_> = found
                        .iter()
                        .map(|c| {
                            let one = wire_call(&completion, sent, c);
                            sent += 1;
                            one
                        })
                        .collect();
                    if let Some(delta) = delta_of(&working, &answer, &calls) {
                        let chunk = chunk(delta, None, json!({})).to_string();
                        if events.send(Event::default().data(chunk)).await.is_err() {
                            return;
                        }
                    }
                    Event::default().data(
                        chunk(
                            json!({}),
                            // The reply ended in a call, so the client's turn
                            // is to run it rather than to show an answer.
                            Some(if sent > 0 { "tool_calls" } else { "stop" }),
                            json!({ "usage": usage(&stats), "kvad": extension(&stats, &loaded) }),
                        )
                        .to_string(),
                    )
                }
                // An error mid-stream cannot become a status code — the
                // headers are long gone — so it is an event of its own, and
                // the stream ends without `[DONE]`.
                Piece::Failed(why) => {
                    // The headers said 200 a while ago; the row says what
                    // actually happened, which is the only place it can.
                    metrics.record("POST", ROUTE, 500, started.elapsed());
                    let _ = events.send(sse("error", &json!({ "error": why }))).await;
                    return;
                }
            };
            if events.send(event).await.is_err() {
                // The client hung up. Stop the generation rather than let it
                // run to `max_tokens` for nobody — and record it, because a
                // reply nobody waited for is still work this server did.
                metrics.record("POST", ROUTE, 499, started.elapsed());
                return;
            }
        }
        let _ = events.send(Event::default().data("[DONE]")).await;
    });

    stream(rx)
}

async fn whole(
    id: String,
    loaded: crate::scheduler::Loaded,
    mut pieces: tokio::sync::mpsc::Receiver<Piece>,
    metrics: std::sync::Arc<crate::metrics::Metrics>,
    started: std::time::Instant,
    offered: bool,
) -> Result<Json<serde_json::Value>, Fail> {
    let mut text = String::new();
    let mut working = String::new();
    let mut thinking = kvad::chat::Thinking::new();
    // The same two passes the stream makes, for the same reason: a reply is
    // assembled here from the pieces either way, so doing it differently
    // would be a second answer to the same question.
    let mut parsing = offered.then(kvad::chat::ToolCalls::new);
    let mut found: Vec<kvad::chat::Called> = Vec::new();
    let take = |parsing: &mut Option<kvad::chat::ToolCalls>,
                    found: &mut Vec<kvad::chat::Called>,
                    answer: String| match parsing {
        Some(p) => {
            let (content, calls) = p.feed(&answer);
            found.extend(calls);
            content
        }
        None => answer,
    };
    while let Some(piece) = pieces.recv().await {
        match piece {
            Piece::Token(t) | Piece::Chose(Chosen { text: t, .. }) => {
                let (w, a) = thinking.feed(&t);
                working.push_str(&w);
                text.push_str(&take(&mut parsing, &mut found, a));
            }
            Piece::Failed(why) => {
                metrics.record("POST", ROUTE, 500, started.elapsed());
                return Err(Fail::internal(why));
            }
            Piece::Done(stats) => {
                metrics.record_generation(
                    "POST",
                    ROUTE,
                    200,
                    started.elapsed(),
                    measured(&stats),
                );
                let (w, a) = thinking.finish();
                working.push_str(&w);
                text.push_str(&take(&mut parsing, &mut found, a));
                if let Some(p) = &mut parsing {
                    let (rest, last) = p.finish();
                    text.push_str(&rest);
                    found.extend(last);
                }
                let mut message = json!({ "role": "assistant", "content": text });
                // Only when there was one: a field that is always present and
                // usually empty teaches a client to ignore it.
                if !working.is_empty() {
                    message["reasoning_content"] = json!(working);
                }
                if !found.is_empty() {
                    let calls: Vec<_> = found
                        .iter()
                        .enumerate()
                        .map(|(n, c)| wire_call(&id, n, c))
                        .collect();
                    message["tool_calls"] = json!(calls);
                    // A call and nothing else: OpenAI sends a null content
                    // there rather than an empty string, and clients test
                    // for it.
                    if text.is_empty() {
                        message["content"] = serde_json::Value::Null;
                    }
                }
                return Ok(Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": now_secs(),
                    "model": loaded.repo,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": if found.is_empty() { "stop" } else { "tool_calls" },
                    }],
                    "usage": usage(&stats),
                    "kvad": extension(&stats, &loaded),
                })))
            }
        }
    }
    metrics.record("POST", ROUTE, 500, started.elapsed());
    Err(Fail::internal("the engine stopped without finishing the reply"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: serde_json::Value) -> Completions {
        serde_json::from_value(json).unwrap()
    }

    /// Defaults where nothing was asked for, and a refusal where something
    /// impossible was.
    #[test]
    fn sampling_defaults_and_refuses() {
        let plain = request(json!({ "messages": [{ "role": "user", "content": "hi" }] }));
        let s = plain.sampling().unwrap();
        assert_eq!(s.temperature, Sampling::default().temperature);
        assert_eq!(s.seed, None, "an unasked-for seed must not be invented");

        let asked = request(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "temperature": 0.0, "top_p": 0.5, "top_k": 3, "seed": 42, "max_tokens": 16,
        }));
        let s = asked.sampling().unwrap();
        assert_eq!((s.temperature, s.top_p, s.top_k, s.seed, s.max_tokens), (0.0, 0.5, 3, Some(42), 16));

        // The newer spelling, which is what an editor's agent sends. Ignoring
        // it would mean honouring the engine's default instead of the
        // client's limit, which a client has no way to notice.
        let renamed = request(json!({
            "messages": [{ "role": "user", "content": "hi" }], "max_completion_tokens": 24,
        }));
        assert_eq!(renamed.sampling().unwrap().max_tokens, 24);

        // A cap, so one request cannot ask the engine for an afternoon.
        let huge = request(json!({ "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 1_000_000 }));
        assert_eq!(huge.sampling().unwrap().max_tokens, 8192);

        for bad in [json!({ "temperature": 5.0 }), json!({ "top_p": 2.0 }), json!({ "max_tokens": 0 })] {
            let mut body = bad.as_object().unwrap().clone();
            body.insert("messages".into(), json!([{ "role": "user", "content": "hi" }]));
            assert!(request(serde_json::Value::Object(body)).sampling().is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn turns_are_checked_before_the_engine_sees_them() {
        let empty = request(json!({ "messages": [] }));
        assert!(empty.turns().is_err());

        let odd = request(json!({ "messages": [{ "role": "wizard", "content": "hi" }] }));
        let err = odd.turns().unwrap_err();
        assert!(err.1.contains("wizard"), "{}", err.1);

        let good = request(json!({ "messages": [
            { "role": "system", "content": "Be brief." },
            { "role": "user", "content": "hi" },
        ] }));
        assert_eq!(good.turns().unwrap().len(), 2);
    }

    /// An agentic client sends back what the model called and what the tool
    /// answered. Both have to survive the round trip, because the model is
    /// about to read its own call beside the result.
    #[test]
    fn a_tool_round_trip_becomes_a_conversation() {
        let body = request(json!({ "messages": [
            { "role": "user", "content": "what is in main.rs?" },
            { "role": "assistant", "content": null, "tool_calls": [{
                "id": "call_abc", "type": "function",
                "function": { "name": "read", "arguments": "{\"path\":\"main.rs\"}" },
            }] },
            { "role": "tool", "tool_call_id": "call_abc", "name": "read", "content": "fn main() {}" },
        ] }));
        let turns = body.turns().unwrap();
        assert_eq!(turns.len(), 3);

        let called = &turns[1];
        assert_eq!(called.content, "", "a call-only turn is empty, not absent");
        assert_eq!(called.tool_calls.len(), 1);
        assert_eq!(called.tool_calls[0].id, "call_abc");
        assert_eq!(called.tool_calls[0].kind, "function");
        assert_eq!(called.tool_calls[0].function.name, "read");
        assert_eq!(called.tool_calls[0].function.arguments, r#"{"path":"main.rs"}"#);

        let result = &turns[2];
        assert_eq!(result.role, "tool");
        assert_eq!(result.content, "fn main() {}");
        assert_eq!(result.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(result.name.as_deref(), Some("read"));
    }

    /// The client libraries send a list of parts as soon as a message has
    /// more than one thing in it. Text parts join; anything else is refused
    /// by name rather than quietly dropped.
    #[test]
    fn content_arrives_as_a_string_or_as_parts() {
        let parts = request(json!({ "messages": [
            { "role": "user", "content": [
                { "type": "text", "text": "read this: " },
                { "type": "text", "text": "fn main() {}" },
            ] },
        ] }));
        assert_eq!(parts.turns().unwrap()[0].content, "read this: fn main() {}");

        let picture = request(json!({ "messages": [
            { "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "x" } }] },
        ] }));
        let err = picture.turns().unwrap_err();
        assert!(err.1.contains("image_url"), "{}", err.1);
    }

    /// What this engine can honour, and what it says instead of pretending.
    #[test]
    fn tool_choice_is_honoured_or_refused() {
        let offered = json!([{ "type": "function", "function": { "name": "read" } }]);
        let with = |choice: serde_json::Value| {
            request(json!({
                "messages": [{ "role": "user", "content": "hi" }],
                "tools": offered, "tool_choice": choice,
            }))
        };

        let auto = request(json!({
            "messages": [{ "role": "user", "content": "hi" }], "tools": offered,
        }));
        assert_eq!(auto.tools().unwrap().len(), 1, "tools with no choice are on offer");
        assert_eq!(with(json!("auto")).tools().unwrap().len(), 1);

        // `none` means they are not on offer at all, so the model is never
        // told about them and cannot call one.
        assert!(with(json!("none")).tools().unwrap().is_empty());

        for refused in [json!("required"), json!({ "type": "function", "function": { "name": "read" } })] {
            let err = with(refused.clone()).tools().unwrap_err();
            assert!(err.1.contains("sampler"), "{refused} was accepted: {}", err.1);
        }

        // A tool with no name is not a tool, and the model would be handed a
        // function it cannot call.
        let nameless = request(json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "description": "does things" } }],
        }));
        assert!(nameless.tools().is_err());
    }

    /// The id is the client's handle on a call: it comes back on the `tool`
    /// message, so it has to be unique within a conversation and stable
    /// between the streamed and the whole reading of the same reply.
    #[test]
    fn a_call_on_the_wire_carries_an_id_and_an_index() {
        let called = kvad::chat::Called { name: "read".into(), arguments: r#"{"a":1}"#.into() };
        let first = wire_call("chatcmpl-42", 0, &called);
        assert_eq!(first["id"], json!("call_42_0"));
        assert_eq!(first["index"], json!(0));
        assert_eq!(first["type"], json!("function"));
        assert_eq!(first["function"]["name"], json!("read"));
        // Text, not an object: the client hands it to somebody else's
        // function and this server does not re-spell it.
        assert_eq!(first["function"]["arguments"], json!(r#"{"a":1}"#));
        assert_ne!(wire_call("chatcmpl-42", 1, &called)["id"], first["id"]);
    }

    /// The extension carries what OpenAI's schema has nowhere to put, and the
    /// rates are derived rather than left for a client to work out.
    #[test]
    fn the_kvad_extension_reports_both_halves_of_a_generation() {
        let loaded = crate::scheduler::Loaded {
            repo: "a/b".into(),
            summary: String::new(),
            params: 0,
            instruct: true,
            tools: true,
            backend: "cpu q8".into(),
            weight_bytes: 0,
            n_ctx: 2048,
            kv_bytes_per_token: 1024,
        };
        let stats = Stats {
            prompt_tokens: 100,
            cached_tokens: 60,
            generated_tokens: 20,
            prefill_secs: 0.5,
            decode_secs: 2.0,
        };
        let ext = extension(&stats, &loaded);
        // Prefill rate counts what was actually computed, not what was in the
        // prompt: 40 tokens in half a second, not 100.
        assert_eq!(ext["prefill_tokens_per_sec"], json!(80.0));
        assert_eq!(ext["decode_tokens_per_sec"], json!(10.0));
        assert_eq!(ext["cached_tokens"], json!(60));
        assert_eq!(ext["backend"], json!("cpu q8"));

        let u = usage(&stats);
        assert_eq!(u["total_tokens"], json!(120));
    }
}
