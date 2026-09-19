//! Conversations, as they are kept.
//!
//! Reading and writing rows, and nothing else — no engine, no HTTP. The chat
//! that a person sees is assembled by the UI out of these rows plus a call to
//! `/v1/chat/completions`, which knows nothing about them.
//!
//! # Whose conversation
//!
//! Every function that touches one takes an `owner: Option<i64>` and matches
//! it with SQL's `IS`, which compares NULL to NULL as equal. So:
//!
//! * signed in as account 5, you see `owner = 5`;
//! * with `auth.mode = "none"`, where there are no accounts, you see
//!   `owner IS NULL` — which is everything, because that is all there is.
//!
//! The scope is in the `WHERE` clause of every statement rather than in a
//! check beside it, so a conversation that is not yours is not found rather
//! than found and then refused. That also means the 404 does not confirm that
//! somebody else's conversation exists.

use crate::db::Db;
use rusqlite::{params, Connection, Row};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Conversation {
    pub id: i64,
    pub title: String,
    pub system: Option<String>,
    pub model: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// How many messages are in it. Cheap enough to count, and it is the one
    /// thing a list of conversations can say that a title cannot.
    pub messages: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Stored {
    #[serde(default)]
    pub id: i64,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<Stats>,
}

/// What one reply cost. The same numbers `/v1/chat/completions` returns in
/// its `kvad` extension, kept so that a conversation reopened tomorrow still
/// shows them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Stats {
    pub model: Option<String>,
    pub backend: Option<String>,
    pub prompt_tokens: i64,
    pub cached_tokens: i64,
    pub generated_tokens: i64,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

/// A title made out of the first thing somebody said.
///
/// Every conversation needs a name in a list and nobody wants to type one.
/// Cut on a word boundary where there is one within reach, because
/// "How do I conf…" reads worse than "How do I…".
pub fn title_from(text: &str) -> String {
    const LIMIT: usize = 48;
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return "New conversation".into();
    }
    if text.chars().count() <= LIMIT {
        return text;
    }
    let cut: String = text.chars().take(LIMIT).collect();
    let trimmed = match cut.rsplit_once(' ') {
        Some((head, _)) if head.chars().count() >= LIMIT / 2 => head.to_string(),
        _ => cut,
    };
    format!("{}…", trimmed.trim_end_matches([' ', ',', '.']))
}

fn conversation_from(row: &Row) -> rusqlite::Result<Conversation> {
    Ok(Conversation {
        id: row.get("id")?,
        title: row.get("title")?,
        system: row.get("system")?,
        model: row.get("model")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        messages: row.get("messages")?,
    })
}

const LIST_SQL: &str = "SELECT c.id, c.title, c.system, c.model, c.created_at, c.updated_at,
        (SELECT count(*) FROM messages m WHERE m.conversation = c.id) AS messages
     FROM conversations c";

/// `IS` rather than `=`, so that NULL matches NULL: the mode with no accounts
/// owns the conversations that have no owner.
const MINE: &str = "c.owner IS ?1";

pub fn list(db: &Db, owner: Option<i64>) -> Res<Vec<Conversation>> {
    db.with(|c| {
        let mut q = c.prepare(&format!(
            "{LIST_SQL} WHERE {MINE} ORDER BY c.updated_at DESC, c.id DESC"
        ))?;
        let rows = q.query_map([owner], conversation_from)?.collect();
        rows
    })
}

pub fn get(db: &Db, id: i64, owner: Option<i64>) -> Res<Option<Conversation>> {
    db.with(|c| {
        c.query_row(
            &format!("{LIST_SQL} WHERE c.id = ?2 AND {MINE}"),
            params![owner, id],
            conversation_from,
        )
        .map(Some)
        .or_else(none_if_missing)
    })
}

pub fn create(
    db: &Db,
    owner: Option<i64>,
    title: &str,
    system: Option<&str>,
    model: Option<&str>,
) -> Res<Conversation> {
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO conversations (title, system, model, owner) VALUES (?1, ?2, ?3, ?4)",
            params![title, system, model, owner],
        )?;
        Ok(c.last_insert_rowid())
    })?;
    get(db, id, owner)?.ok_or_else(|| "the conversation vanished as it was created".into())
}

/// Change a conversation's title, system prompt or model.
///
/// Each argument is "leave it alone" when `None`, so a client that only wants
/// to rename something does not have to send back a system prompt it might
/// have got out of date.
pub fn update(
    db: &Db,
    id: i64,
    owner: Option<i64>,
    title: Option<&str>,
    system: Option<Option<&str>>,
    model: Option<&str>,
) -> Res<Option<Conversation>> {
    // Every statement carries the owner, so an update is silently nothing
    // rather than a change to somebody else's conversation.
    let changed = db.with(|c| {
        let mut n = 0;
        let set = |column: &str, value: &dyn rusqlite::ToSql| -> rusqlite::Result<usize> {
            c.execute(
                &format!("UPDATE conversations SET {column} = ?3 WHERE id = ?2 AND owner IS ?1"),
                params![owner, id, value],
            )
        };
        if let Some(title) = title {
            n += set("title", &title)?;
        }
        if let Some(system) = system {
            n += set("system", &system)?;
        }
        if let Some(model) = model {
            n += set("model", &model)?;
        }
        if n > 0 {
            c.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?2 AND owner IS ?1",
                params![owner, id],
            )?;
        }
        Ok(())
    });
    changed?;
    get(db, id, owner)
}

pub fn delete(db: &Db, id: i64, owner: Option<i64>) -> Res<bool> {
    // `ON DELETE CASCADE` takes the messages, and `foreign_keys` is turned on
    // for every connection in `Db::prepare` — without it SQLite would leave
    // them behind, silently.
    Ok(db.with(|c| {
        c.execute("DELETE FROM conversations WHERE id = ?2 AND owner IS ?1", params![owner, id])
    })? > 0)
}

/// The messages of a conversation of `owner`'s. Empty for one that is not.
pub fn messages(db: &Db, id: i64, owner: Option<i64>) -> Res<Vec<Stored>> {
    if !exists(db, id, owner)? {
        return Ok(Vec::new());
    }
    db.with(|c| {
        let mut q = c.prepare(
            "SELECT id, role, content, created_at, model, backend, prompt_tokens,
                    cached_tokens, generated_tokens, prefill_secs, decode_secs
             FROM messages WHERE conversation = ?1 ORDER BY id",
        )?;
        let rows = q.query_map([id], |row| {
            // Every stats column is written together or not at all, so one of
            // them standing in for the rest is safe.
            let generated: Option<i64> = row.get("generated_tokens")?;
            Ok(Stored {
                id: row.get("id")?,
                role: row.get("role")?,
                content: row.get("content")?,
                created_at: row.get("created_at")?,
                stats: match generated {
                    None => None,
                    Some(generated_tokens) => Some(Stats {
                        model: row.get("model")?,
                        backend: row.get("backend")?,
                        prompt_tokens: row.get("prompt_tokens")?,
                        cached_tokens: row.get("cached_tokens")?,
                        generated_tokens,
                        prefill_secs: row.get("prefill_secs")?,
                        decode_secs: row.get("decode_secs")?,
                    }),
                },
            })
        })?;
        rows.collect()
    })
}

pub fn append(
    db: &Db,
    id: i64,
    owner: Option<i64>,
    role: &str,
    content: &str,
    stats: Option<&Stats>,
) -> Res<Stored> {
    if !matches!(role, "system" | "user" | "assistant") {
        return Err(format!("`{role}` is not a role a message can have").into());
    }
    if !exists(db, id, owner)? {
        return Err(format!("there is no conversation {id}").into());
    }
    let row = db.with(|c| {
        c.execute(
            "INSERT INTO messages
                (conversation, role, content, model, backend, prompt_tokens,
                 cached_tokens, generated_tokens, prefill_secs, decode_secs)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                id,
                role,
                content,
                stats.and_then(|s| s.model.clone()),
                stats.and_then(|s| s.backend.clone()),
                stats.map(|s| s.prompt_tokens),
                stats.map(|s| s.cached_tokens),
                stats.map(|s| s.generated_tokens),
                stats.map(|s| s.prefill_secs),
                stats.map(|s| s.decode_secs),
            ],
        )?;
        let new = c.last_insert_rowid();
        // A conversation's order in the list is when it was last spoken to.
        c.execute("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?1", [id])?;
        Ok(new)
    })?;
    messages(db, id, owner)?
        .into_iter()
        .find(|m| m.id == row)
        .ok_or_else(|| "the message vanished as it was written".into())
}

fn none_if_missing<T>(e: rusqlite::Error) -> rusqlite::Result<Option<T>> {
    match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    }
}

/// Whether a conversation of `owner`'s exists, without reading it.
pub fn exists(db: &Db, id: i64, owner: Option<i64>) -> Res<bool> {
    db.with(|c: &Connection| {
        c.query_row(
            "SELECT 1 FROM conversations WHERE id = ?2 AND owner IS ?1",
            params![owner, id],
            |_| Ok(()),
        )
        .map(|_| true)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(false),
            other => Err(other),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The owner every test that is not about ownership uses: none, which is
    /// what `auth.mode = "none"` passes.
    const ALONE: Option<i64> = None;

    fn stats() -> Stats {
        Stats {
            model: Some("openai-community/gpt2".into()),
            backend: Some("cpu q8".into()),
            prompt_tokens: 12,
            cached_tokens: 8,
            generated_tokens: 40,
            prefill_secs: 0.12,
            decode_secs: 0.9,
        }
    }

    #[test]
    fn a_conversation_keeps_its_messages_in_order_and_their_numbers_with_them() {
        let db = Db::in_memory().unwrap();
        let c = create(&db, ALONE, "First", Some("Be brief."), Some("gpt2")).unwrap();
        assert_eq!(c.messages, 0);

        append(&db, c.id, ALONE, "user", "hello", None).unwrap();
        let reply = append(&db, c.id, ALONE, "assistant", "hi", Some(&stats())).unwrap();
        append(&db, c.id, ALONE, "user", "again", None).unwrap();

        let all = messages(&db, c.id, ALONE).unwrap();
        assert_eq!(
            all.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            ["hello", "hi", "again"]
        );
        // Only the assistant's message carries numbers, and they survive.
        assert!(all[0].stats.is_none() && all[2].stats.is_none());
        let kept = all[1].stats.as_ref().unwrap();
        assert_eq!(kept.generated_tokens, 40);
        assert_eq!(kept.cached_tokens, 8);
        assert_eq!(kept.backend.as_deref(), Some("cpu q8"));
        assert_eq!(reply.id, all[1].id);

        assert_eq!(get(&db, c.id, ALONE).unwrap().unwrap().messages, 3);
    }

    /// Deleting a conversation has to take its messages with it. SQLite
    /// enforces `ON DELETE CASCADE` only when foreign keys are on, which is a
    /// per-connection pragma and therefore easy to lose.
    #[test]
    fn deleting_a_conversation_takes_its_messages() {
        let db = Db::in_memory().unwrap();
        let a = create(&db, ALONE, "Going", None, None).unwrap();
        let b = create(&db, ALONE, "Staying", None, None).unwrap();
        append(&db, a.id, ALONE, "user", "one", None).unwrap();
        append(&db, b.id, ALONE, "user", "two", None).unwrap();

        assert!(delete(&db, a.id, ALONE).unwrap());
        assert!(!delete(&db, a.id, ALONE).unwrap(), "deleting it twice reported success twice");
        assert!(!exists(&db, a.id, ALONE).unwrap());

        let orphans: i64 = db
            .with(|c| {
                c.query_row("SELECT count(*) FROM messages WHERE conversation = ?1", [a.id], |r| {
                    r.get(0)
                })
            })
            .unwrap();
        assert_eq!(orphans, 0, "the messages outlived their conversation");
        assert_eq!(messages(&db, b.id, ALONE).unwrap().len(), 1);
    }

    /// Each field of an update is optional, and a system prompt can be
    /// cleared — which is not the same as leaving it alone.
    #[test]
    fn an_update_changes_only_what_it_was_given() {
        let db = Db::in_memory().unwrap();
        let c = create(&db, ALONE, "Old", Some("Be brief."), None).unwrap();

        let renamed = update(&db, c.id, ALONE, Some("New"), None, None).unwrap().unwrap();
        assert_eq!(renamed.title, "New");
        assert_eq!(renamed.system.as_deref(), Some("Be brief."), "a rename lost the system prompt");

        let cleared = update(&db, c.id, ALONE, None, Some(None), None).unwrap().unwrap();
        assert_eq!(cleared.title, "New");
        assert_eq!(cleared.system, None);

        assert!(update(&db, 9999, ALONE, Some("x"), None, None).unwrap().is_none());
    }

    #[test]
    fn a_title_is_the_first_thing_said_cut_at_a_word() {
        assert_eq!(title_from("Explain rotary embeddings"), "Explain rotary embeddings");
        assert_eq!(title_from("   "), "New conversation");
        assert_eq!(title_from("a\n\nb   c"), "a b c");

        let long = title_from("Explain rotary position embeddings and why they beat learned ones");
        assert!(long.ends_with('…'), "{long}");
        assert!(long.chars().count() <= 49, "{long} is {} characters", long.chars().count());
        assert!(long.starts_with("Explain rotary position embeddings"), "{long}");

        // A single long word has nowhere to cut, so it is cut anyway.
        let unbroken = title_from(&"x".repeat(80));
        assert_eq!(unbroken.chars().count(), 49);
    }

    /// The role column is checked in SQL as well; this is the error a caller
    /// gets rather than a constraint violation.
    #[test]
    fn a_message_needs_a_role_that_exists() {
        let db = Db::in_memory().unwrap();
        let c = create(&db, ALONE, "x", None, None).unwrap();
        let err = append(&db, c.id, ALONE, "wizard", "abracadabra", None).unwrap_err().to_string();
        assert!(err.contains("wizard"), "{err}");
    }

    /// The test this module exists for. Delete the `owner IS ?1` from any one
    /// of these statements and this fails.
    ///
    /// A conversation that is not yours is not *refused*, it is not *found* —
    /// so the answer to "does conversation 3 exist" is the same whether it
    /// does not exist at all or merely is not yours.
    #[test]
    fn one_persons_conversation_is_not_anothers_to_read_or_change() {
        let db = Db::in_memory().unwrap();
        let (ada, bob) = (Some(1), Some(2));
        // Real rows, because the foreign key is enforced.
        crate::users::create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
        crate::users::create(&db, "bob", Some("builder-2024"), "user", None).unwrap();

        let hers = create(&db, ada, "Ada's", Some("Be brief."), None).unwrap();
        append(&db, hers.id, ada, "user", "a secret", None).unwrap();
        let his = create(&db, bob, "Bob's", None, None).unwrap();

        // Each sees one conversation: their own.
        assert_eq!(list(&db, ada).unwrap().len(), 1);
        assert_eq!(list(&db, bob).unwrap().iter().map(|c| c.id).collect::<Vec<_>>(), [his.id]);

        // Bob cannot read hers, by id or by its messages.
        assert!(get(&db, hers.id, bob).unwrap().is_none());
        assert!(messages(&db, hers.id, bob).unwrap().is_empty(), "bob read ada's messages");
        assert!(!exists(&db, hers.id, bob).unwrap());

        // ...nor rename it, nor add to it, nor delete it.
        assert!(update(&db, hers.id, bob, Some("Bob's now"), None, None).unwrap().is_none());
        assert!(append(&db, hers.id, bob, "user", "hello?", None).is_err());
        assert!(!delete(&db, hers.id, bob).unwrap());

        // And none of that changed anything.
        let still = get(&db, hers.id, ada).unwrap().unwrap();
        assert_eq!(still.title, "Ada's");
        assert_eq!(messages(&db, hers.id, ada).unwrap().len(), 1);

        // The mode with no accounts sees the conversations with no owner, and
        // those alone.
        let shared = create(&db, None, "From before accounts", None, None).unwrap();
        assert_eq!(list(&db, None).unwrap().iter().map(|c| c.id).collect::<Vec<_>>(), [shared.id]);
        assert!(get(&db, hers.id, None).unwrap().is_none());
    }
}
