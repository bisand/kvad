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
//! OpenAI's `model` selects from many that are all warm. Here there is one
//! model in memory at a time, so naming a different one means loading it,
//! which takes tens of seconds and evicts what was there. That is what this
//! does, because a client that asks for a model and gets somebody else's
//! answer is worse than a client that waits. Omitting `model` uses whatever
//! is loaded.

use crate::api::{blocking, Fail};
use crate::models::{sse, stream};
use crate::auth::{Identity, State};
use crate::scheduler::Piece;
use axum::extract::State as St;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use axum::Json;
use kvad::chat::Message;
use kvad::runtime::Stats;
use kvad::service::Sampling;
use serde_json::json;

pub async fn models(_: Identity) -> Result<Json<serde_json::Value>, Fail> {
    let found = blocking(move || {
        let mut all: Vec<(String, bool)> = kvad::hub::local_models()
            .into_iter()
            .filter(|m| m.complete && m.arch.is_some())
            .map(|m| (m.id, false))
            .collect();
        all.extend(
            kvad::hub::trained_models()
                .into_iter()
                .filter(|m| m.complete && m.arch.is_some())
                .map(|m| (m.id, true)),
        );
        Ok(all)
    })
    .await?;

    Ok(Json(json!({
        "object": "list",
        "data": found
            .into_iter()
            .map(|(id, trained)| json!({
                "id": id,
                "object": "model",
                // The field is required and means "when was this published".
                // Nothing here was published, so it is zero rather than a
                // number invented to look like a date.
                "created": 0,
                "owned_by": if trained { "kvad" } else { "huggingface" },
            }))
            .collect::<Vec<_>>(),
    })))
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
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(serde::Deserialize)]
pub struct Turn {
    role: String,
    content: String,
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
        let max_tokens = self.max_tokens.unwrap_or(d.max_tokens);
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
            .map(|t| match t.role.as_str() {
                "system" => Ok(Message::system(&t.content)),
                "user" => Ok(Message::user(&t.content)),
                "assistant" => Ok(Message::assistant(&t.content)),
                other => Err(Fail::bad(format!("`{other}` is not a role a message can have"))),
            })
            .collect()
    }
}

pub async fn completions(
    _: Identity,
    St(state): St<State>,
    Json(body): Json<Completions>,
) -> Result<Response, Fail> {
    let sampling = body.sampling()?;
    let turns = body.turns()?;

    let loaded = match (&body.model, state.engine.loaded()) {
        (None, Some(l)) => l,
        (None, None) => {
            return Err(Fail::bad("no model is loaded, and the request did not name one"))
        }
        (Some(wanted), Some(l)) if &l.repo == wanted => l,
        (Some(wanted), _) => {
            // Naming a model that is not loaded loads it. The queue makes
            // that safe — the load goes in front of this request's own turn —
            // but it is slow, and a client that did not mean it should be
            // told why it waited.
            let db = state.db.clone();
            let backend = blocking(move || {
                Ok(db
                    .setting("models.backend")
                    .ok()
                    .flatten()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_else(|| "cpu-q8".into()))
            })
            .await?;
            let backend = crate::engine::parse(&backend)
                .unwrap_or(kvad::service::Backend::Cpu(kvad::quant::Precision::Q8));
            let (progress, _ignored) = tokio::sync::mpsc::channel(1);
            state
                .engine
                .load(wanted.clone(), backend, progress)
                .await
                .map_err(|why| Fail::bad(format!("could not load {wanted}: {why}")))?
        }
    };

    let pieces = state.engine.chat(turns, sampling).map_err(Fail::internal)?;
    let id = format!("chatcmpl-{}", now_millis());
    let metrics = state.metrics.clone();
    // Timed from here rather than by the middleware: for a streamed reply the
    // middleware sees only the moment the headers went out, which is the
    // moment before all the work.
    let started = std::time::Instant::now();

    let mut response = match body.stream {
        true => streamed(id, loaded, pieces, metrics, started).into_response(),
        false => whole(id, loaded, pieces, metrics, started).await?.into_response(),
    };
    response.extensions_mut().insert(crate::watching::RecordedItself);
    Ok(response)
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
/// A `DELETE` with no body, because there is one generation to cancel and the
/// request that started it is the one holding the stream. When several can
/// run at once this grows an id.
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
        while let Some(piece) = pieces.recv().await {
            let event = match piece {
                Piece::Token(text) => {
                    let first = !std::mem::replace(&mut opened, true);
                    let delta = match first {
                        true => json!({ "role": "assistant", "content": text }),
                        false => json!({ "content": text }),
                    };
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
                    Event::default().data(
                        chunk(
                            json!({}),
                            Some("stop"),
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
) -> Result<Json<serde_json::Value>, Fail> {
    let mut text = String::new();
    while let Some(piece) = pieces.recv().await {
        match piece {
            Piece::Token(t) => text.push_str(&t),
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
                return Ok(Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": now_secs(),
                    "model": loaded.repo,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": text },
                        "finish_reason": "stop",
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

    /// The extension carries what OpenAI's schema has nowhere to put, and the
    /// rates are derived rather than left for a client to work out.
    #[test]
    fn the_kvad_extension_reports_both_halves_of_a_generation() {
        let loaded = crate::scheduler::Loaded {
            repo: "a/b".into(),
            summary: String::new(),
            params: 0,
            instruct: true,
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
