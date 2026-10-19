//! Conversations kept on the phone, as Speak keeps its takes: one JSON file
//! each, named by the millisecond it began, in the folder the app gives.
//! The app makes that folder, with its protection class and its backups
//! setting; the files take the folder's class. Nothing of a conversation is
//! kept on the server (`/v1/chat/completions` keeps nothing).

use crate::api::Turn;
use serde_json::{json, Value};
use std::path::Path;

/// A conversation's title: its first question, cut at about a line.
fn title(turns: &[Turn]) -> String {
    let first = turns.iter().find(|t| t.role == "user").map(|t| t.content.trim()).unwrap_or_default();
    let line = first.lines().next().unwrap_or_default();
    match line.char_indices().nth(60) {
        Some((at, _)) => format!("{}…", line[..at].trim_end()),
        None => line.to_string(),
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A new conversation's id: the millisecond it began.
pub fn new_id() -> String {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0).to_string()
}

/// A conversation's id names its file, so nothing in it can climb out of the
/// folder.
pub fn plain(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit())
}

/// What the list shows of a conversation, without its turns.
fn about(v: &Value) -> Value {
    json!({ "id": v["id"], "started": v["started"], "updated": v["updated"], "model": v["model"], "title": v["title"] })
}

/// Writes conversation `id` with `turns`, and says what the list shows of it.
pub fn save(folder: &str, id: &str, model: &str, turns: &[Turn]) -> Result<Value, String> {
    if !plain(id) {
        return Err("That is not a conversation's id.".into());
    }
    std::fs::create_dir_all(folder).map_err(|e| format!("There is nowhere to keep the conversation: {e}"))?;
    let started = id.parse::<u64>().map(|ms| ms / 1000).unwrap_or_else(|_| now());
    let v = json!({
        "id": id,
        "started": started,
        "updated": now(),
        "model": model,
        "title": title(turns),
        "turns": turns.iter().map(|t| json!({ "role": t.role, "content": t.content })).collect::<Vec<_>>(),
    });
    std::fs::write(Path::new(folder).join(format!("{id}.json")), v.to_string()).map_err(|e| format!("The conversation could not be kept: {e}"))?;
    Ok(about(&v))
}

/// Conversation `id`: its model and its turns.
pub fn load(folder: &str, id: &str) -> Result<(String, Vec<Turn>), String> {
    if !plain(id) {
        return Err("That is not a conversation's id.".into());
    }
    let bytes = std::fs::read(Path::new(folder).join(format!("{id}.json"))).map_err(|_| "That conversation is not on this phone.".to_string())?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|_| "That conversation could not be read.".to_string())?;
    let turns = v["turns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let role = match t["role"].as_str()? {
                "user" => "user",
                "assistant" => "assistant",
                _ => return None,
            };
            Some(Turn { role, content: t["content"].as_str()?.to_string() })
        })
        .collect();
    Ok((v["model"].as_str().unwrap_or_default().to_string(), turns))
}

/// Every conversation in `folder`, the latest first.
pub fn list(folder: &str) -> Vec<Value> {
    let mut all: Vec<Value> = std::fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
        .filter_map(|e| serde_json::from_slice::<Value>(&std::fs::read(e.path()).ok()?).ok())
        .filter(|v| v["id"].as_str().is_some_and(plain))
        .map(|v| about(&v))
        .collect();
    all.sort_by(|a, b| b["updated"].as_u64().cmp(&a["updated"].as_u64()).then(b["id"].as_str().cmp(&a["id"].as_str())));
    all
}

pub fn delete(folder: &str, id: &str) -> Result<(), String> {
    if !plain(id) {
        return Err("That is not a conversation's id.".into());
    }
    match std::fs::remove_file(Path::new(folder).join(format!("{id}.json"))) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("The conversation could not be deleted: {e}")),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &'static str, content: &str) -> Turn {
        Turn { role, content: content.into() }
    }

    #[test]
    fn a_conversation_is_kept_listed_opened_and_deleted() {
        let dir = std::env::temp_dir().join(format!("kvad-chats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let folder = dir.to_str().unwrap();
        let turns = [turn("user", "Hva er hovedstaden i Norge?\nKort, takk."), turn("assistant", "Oslo.")];
        let about = save(folder, "1760870000000", "Qwen/Qwen3-14B", &turns).unwrap();
        assert_eq!(about["title"], "Hva er hovedstaden i Norge?");
        assert_eq!(about["started"], 1760870000);
        save(folder, "1760870005000", "m", &[turn("user", &"x".repeat(100))]).unwrap();
        let all = list(folder);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0]["id"], "1760870005000", "the latest first");
        assert_eq!(all[0]["title"].as_str().unwrap().chars().count(), 61, "cut at 60 and an ellipsis");
        let (model, back) = load(folder, "1760870000000").unwrap();
        assert_eq!((model.as_str(), back), ("Qwen/Qwen3-14B", turns.to_vec()));
        delete(folder, "1760870000000").unwrap();
        delete(folder, "1760870000000").unwrap();
        assert_eq!(list(folder).len(), 1);
        assert!(load(folder, "../../etc").is_err() && save(folder, "a/b", "m", &turns).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
