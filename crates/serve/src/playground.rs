//! Raw completion, and what the tokenizer and the sampler were actually
//! doing.
//!
//! The chat page is the model being useful. This is the model being a model:
//! no template, no conversation, just text continued — and, if asked, the
//! distribution behind every token it picked.
//!
//! Both of the views here are nearly free, which is the argument for having
//! them. The logits were computed anyway and generation was throwing all but
//! the winner away; the tokenizer had already cut the prompt up and nobody
//! was shown the pieces. Neither needs a second code path through the engine,
//! and neither makes generation measurably slower — see
//! [`kvad::runtime::Llm::generate_explained`].

use crate::api::Fail;
use crate::auth::{Identity, State};
use crate::models::{sse, stream};
use crate::scheduler::Piece;
use axum::extract::State as St;
use axum::response::sse::Event;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use kvad::service::Sampling;
use serde_json::json;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/playground/complete", post(complete))
        .route("/api/playground/tokenize", post(tokenize))
}

/// The most candidates a request may ask to be shown per token.
///
/// Twenty rows is already more than anyone reads, and the cost of the view is
/// in the sort rather than in `k` — but a request asking for the whole
/// vocabulary would produce a megabyte of JSON per token.
const MAX_EXPLAIN: usize = 20;

#[derive(serde::Deserialize)]
pub struct CompleteRequest {
    prompt: String,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
    /// Fix the sampler's generator, so the same request twice gives the same
    /// text. The playground is where somebody is changing one knob at a time,
    /// which only means anything if the other variable is held still.
    #[serde(default)]
    seed: Option<u64>,
    /// How many candidates to report per token. 0, the default, is an
    /// ordinary completion.
    #[serde(default)]
    explain: Option<usize>,
    /// Start from an empty KV cache. On by default here, because the
    /// playground is where somebody runs the same prompt twice on purpose and
    /// a second run served out of the first one's cache is not a second run.
    #[serde(default = "yes")]
    fresh: bool,
}

fn yes() -> bool {
    true
}

/// One token, and what it was chosen from.
#[derive(serde::Serialize)]
struct Chose {
    id: u32,
    text: String,
    top: Vec<Candidate>,
}

#[derive(serde::Serialize)]
struct Candidate {
    id: u32,
    text: String,
    /// The model's own probability, over the whole vocabulary — not the
    /// sampler's reweighting of it. See [`kvad::sampler::Sampler::explain`].
    prob: f32,
    /// Whether top-k and top-p left it in play.
    kept: bool,
    chosen: bool,
}

async fn complete(
    _: Identity,
    St(state): St<State>,
    Json(body): Json<CompleteRequest>,
) -> Result<impl IntoResponse, Fail> {
    let prompt = body.prompt;
    if prompt.trim().is_empty() {
        return Err(Fail::bad("a completion needs a prompt to continue"));
    }
    if state.engine.loaded().is_none() {
        return Err(Fail::bad("no model is loaded; load one from the Models page"));
    }
    let d = Sampling::default();
    let sampling = Sampling {
        temperature: body.temperature.unwrap_or(d.temperature).clamp(0.0, 4.0),
        top_k: body.top_k.unwrap_or(d.top_k),
        top_p: body.top_p.unwrap_or(d.top_p).clamp(0.0, 1.0),
        seed: body.seed,
        max_tokens: body.max_tokens.unwrap_or(256).clamp(1, 4096),
    };
    let explain = body.explain.unwrap_or(0).min(MAX_EXPLAIN);

    let mut pieces = state
        .engine
        .complete(prompt, sampling, explain, body.fresh)
        .map_err(Fail::internal)?;

    let (events, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        while let Some(piece) = pieces.recv().await {
            let event = match piece {
                Piece::Token(text) => sse("token", &json!({ "text": text })),
                Piece::Chose(c) => sse(
                    "token",
                    &Chose {
                        id: c.id,
                        text: c.text,
                        top: c
                            .top
                            .into_iter()
                            .map(|t| Candidate {
                                id: t.id,
                                text: t.text,
                                prob: t.prob,
                                kept: t.kept,
                                chosen: t.chosen,
                            })
                            .collect(),
                    },
                ),
                Piece::Done(stats) => sse(
                    "done",
                    &json!({
                        "prompt_tokens": stats.prompt_tokens,
                        "cached_tokens": stats.cached_tokens,
                        "generated_tokens": stats.generated_tokens,
                        "prefill_secs": stats.prefill_secs,
                        "decode_secs": stats.decode_secs,
                        "decode_per_sec": stats.tokens_per_sec(),
                    }),
                ),
                Piece::Failed(why) => sse("error", &json!({ "error": why })),
            };
            if events.send(event).await.is_err() {
                // The reader has gone. Stopping the generation is the
                // scheduler's business: dropping this receiver does it.
                return;
            }
        }
    });

    Ok(stream(rx))
}

#[derive(serde::Deserialize)]
pub struct TokenizeRequest {
    text: String,
}

#[derive(serde::Serialize)]
struct Split {
    tokens: Vec<Token>,
    count: usize,
    /// Characters in, tokens out. The number that decides what a context
    /// window is worth, and it is different for every model.
    characters: usize,
}

#[derive(serde::Serialize)]
struct Token {
    id: u32,
    /// The vocabulary entry, `Ġthe` and all.
    token: String,
    /// The text it covers, cut from the input.
    piece: String,
    start: usize,
    end: usize,
}

/// How the loaded model's tokenizer splits a text.
async fn tokenize(
    _: Identity,
    St(state): St<State>,
    Json(body): Json<TokenizeRequest>,
) -> Result<Json<Split>, Fail> {
    if state.engine.loaded().is_none() {
        return Err(Fail::bad("no model is loaded; load one from the Models page"));
    }
    let characters = body.text.chars().count();
    let tokens = state.engine.tokenize(body.text).await.map_err(Fail::bad)?;
    Ok(Json(Split {
        count: tokens.len(),
        characters,
        tokens: tokens
            .into_iter()
            .map(|t| Token {
                id: t.id,
                token: t.token,
                piece: t.piece,
                start: t.start,
                end: t.end,
            })
            .collect(),
    }))
}
