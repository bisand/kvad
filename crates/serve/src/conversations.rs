//! Conversations over HTTP. Rows in, rows out.
//!
//! Nothing here talks to the engine. The UI sends a turn to
//! `/v1/chat/completions`, watches the tokens arrive, and then posts what was
//! said to these routes. That means the compatible endpoint stays stateless
//! and identical for every client, and it means a conversation is saved
//! because somebody chose to save it rather than as a side effect of asking a
//! question.
//!
//! Every route passes `who.id` down to [`crate::chat`], which puts it in the
//! `WHERE` clause. Somebody else's conversation is therefore not found rather
//! than found and refused, and the 404 does not confirm that it exists.

use crate::api::{blocking, Fail};
use crate::auth::{Identity, State};
use crate::chat::{self, Conversation, Stats, Stored};
use axum::extract::{Path, State as St};
use axum::Json;

#[derive(serde::Deserialize)]
pub struct NewConversation {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(serde::Serialize)]
pub struct Full {
    #[serde(flatten)]
    conversation: Conversation,
    messages: Vec<Stored>,
}

pub async fn list(who: Identity, St(state): St<State>) -> Result<Json<Vec<Conversation>>, Fail> {
    let db = state.db.clone();
    let me = who.id;
    blocking(move || chat::list(&db, me)).await.map(Json)
}

pub async fn create(
    who: Identity,
    St(state): St<State>,
    Json(body): Json<NewConversation>,
) -> Result<Json<Conversation>, Fail> {
    let db = state.db.clone();
    let me = who.id;
    blocking(move || {
        let title = body.title.as_deref().map(str::trim).filter(|t| !t.is_empty());
        chat::create(
            &db,
            me,
            title.unwrap_or("New conversation"),
            body.system.as_deref(),
            body.model.as_deref(),
        )
    })
    .await
    .map(Json)
}

pub async fn get(
    who: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<Full>, Fail> {
    let db = state.db.clone();
    let me = who.id;
    let found = blocking(move || {
        Ok(match chat::get(&db, id, me)? {
            None => None,
            Some(conversation) => Some((conversation, chat::messages(&db, id, me)?)),
        })
    })
    .await?;
    match found {
        Some((conversation, messages)) => Ok(Json(Full { conversation, messages })),
        None => Err(Fail::missing(format!("there is no conversation {id}"))),
    }
}

#[derive(serde::Deserialize)]
pub struct Patch {
    #[serde(default)]
    title: Option<String>,
    /// Three states, not two: absent leaves the system prompt alone, `null`
    /// clears it, a string replaces it. `Option<Option<_>>` with
    /// `deserialize_with` is how serde spells that.
    #[serde(default, deserialize_with = "double_option")]
    system: Option<Option<String>>,
    #[serde(default)]
    model: Option<String>,
}

fn double_option<'de, D>(d: D) -> Result<Option<Option<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(d).map(Some)
}

pub async fn update(
    who: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
    Json(body): Json<Patch>,
) -> Result<Json<Conversation>, Fail> {
    let db = state.db.clone();
    let me = who.id;
    let updated = blocking(move || {
        chat::update(
            &db,
            id,
            me,
            body.title.as_deref().map(str::trim).filter(|t| !t.is_empty()),
            body.system.as_ref().map(|s| s.as_deref()),
            body.model.as_deref(),
        )
    })
    .await?;
    updated.map(Json).ok_or_else(|| Fail::missing(format!("there is no conversation {id}")))
}

pub async fn remove(
    who: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let me = who.id;
    match blocking(move || chat::delete(&db, id, me)).await? {
        true => Ok(Json(serde_json::json!({ "deleted": id }))),
        false => Err(Fail::missing(format!("there is no conversation {id}"))),
    }
}

#[derive(serde::Deserialize)]
pub struct NewMessage {
    role: String,
    content: String,
    #[serde(default)]
    stats: Option<Stats>,
    /// Rename the conversation to this message, if it is still called
    /// whatever it was called when it was empty. The UI sets it on the first
    /// user message so a conversation gets a name without anyone typing one.
    #[serde(default)]
    title_if_unnamed: bool,
}

pub async fn append(
    who: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
    Json(body): Json<NewMessage>,
) -> Result<Json<Stored>, Fail> {
    let db = state.db.clone();
    let me = who.id;
    let exists = {
        let db = db.clone();
        blocking(move || chat::exists(&db, id, me)).await?
    };
    if !exists {
        return Err(Fail::missing(format!("there is no conversation {id}")));
    }

    blocking(move || {
        let stored = chat::append(&db, id, me, &body.role, &body.content, body.stats.as_ref())?;
        if body.title_if_unnamed {
            // Only a conversation still carrying the placeholder gets
            // renamed, so a title somebody typed is never overwritten.
            let current = chat::get(&db, id, me)?;
            if current.is_some_and(|c| c.title == "New conversation") {
                chat::update(&db, id, me, Some(&chat::title_from(&body.content)), None, None)?;
            }
        }
        Ok(stored)
    })
    .await
    .map(Json)
    .map_err(|e| match e.1.contains("is not a role") {
        true => Fail::bad(e.1),
        false => e,
    })
}
