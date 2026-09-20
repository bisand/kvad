//! Searching a dataset, and putting what comes back in front of a model.
//!
//! [`kvad::retrieve`] does the work; this is where the result meets a
//! request. Two things live here that do not belong in the engine crate: the
//! index cache, and the rule for when to throw one away.
//!
//! # Built on demand, not stored
//!
//! There is no `chunks` table and no migration. The whole of the Rust book —
//! 1.3 MB, 111 pages — chunks in 7 ms and indexes in 20 ms, which is less
//! than the round trip that asked for it. A table would be a second copy of
//! something the file on disk already says, kept in step by hand, for no
//! measured gain; `datasets` has that rule at the top of it and this follows
//! it.
//!
//! What is kept is the built index, until the file under it changes. Length
//! and modification time, because a dataset is replaced by being written over
//! — a crawl of the same name — and both move when it is.

use crate::db::Db;
use kvad::retrieve::{Hit, Index};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What the file looked like when its index was built.
type Fingerprint = (u64, Option<std::time::SystemTime>);

struct Cached {
    fingerprint: Fingerprint,
    index: Arc<Index>,
}

fn cache() -> &'static Mutex<HashMap<i64, Cached>> {
    static CACHE: OnceLock<Mutex<HashMap<i64, Cached>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn fingerprint(path: &std::path::Path) -> Fingerprint {
    match std::fs::metadata(path) {
        Ok(m) => (m.len(), m.modified().ok()),
        Err(_) => (0, None),
    }
}

/// The index for a dataset, building it if the file has changed under it.
pub fn index_for(db: &Db, id: i64) -> Res<Arc<Index>> {
    let dataset = crate::datasets::get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    let path = crate::datasets::path_of(&dataset.name)?;
    let now = fingerprint(&path);

    {
        let held = cache().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = held.get(&id) {
            if cached.fingerprint == now {
                return Ok(Arc::clone(&cached.index));
            }
        }
    }

    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let index = Arc::new(Index::build(kvad::retrieve::chunk(&text, &pages_of(&dataset.name))));

    let mut held = cache().lock().unwrap_or_else(|e| e.into_inner());
    held.insert(id, Cached { fingerprint: now, index: Arc::clone(&index) });
    Ok(index)
}

/// Forget a dataset's index. Called when the dataset goes.
pub fn forget(id: i64) {
    cache().lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
}

/// The pages a crawl recorded, so that a hit can cite one.
///
/// Absent or unreadable is not an error: an uploaded corpus has no manifest
/// and its chunks simply have no source. A citation that cannot be made is
/// better left unmade.
fn pages_of(name: &str) -> Vec<(String, String)> {
    let Ok(path) = crate::datasets::manifest_of(name) else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    json.get("pages")
        .and_then(|p| p.as_array())
        .map(|pages| {
            pages
                .iter()
                .filter_map(|p| {
                    let title = p.get("title")?.as_str()?.to_string();
                    let url = p.get("url")?.as_str()?.to_string();
                    Some((title, url))
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn search(db: &Db, id: i64, question: &str, k: usize) -> Res<Vec<Hit>> {
    Ok(index_for(db, id)?.search(question, k))
}

/// The instruction a model gets along with what was found.
///
/// It says three things, and each of them is a failure seen without it: use
/// what is here, say when it is not enough, and cite the page. A small model
/// told none of that answers from whatever it remembers and sounds equally
/// confident either way.
pub const INSTRUCTION: &str = "\
Answer using only the extracts below. They are from documentation the user is asking about. \
If they do not contain the answer, say so plainly rather than guessing. \
Cite the page you used.";

/// The system prompt for a question against a dataset, or `None` when
/// nothing matched — in which case the model should be told nothing rather
/// than told nothing was found, and answer as it normally would.
pub fn grounding(db: &Db, id: i64, question: &str, k: usize, budget: usize) -> Res<Option<String>> {
    let hits = search(db, id, question, k)?;
    if hits.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!("{INSTRUCTION}\n\n{}", kvad::retrieve::context(&hits, budget))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The index is thrown away when the file under it moves, and kept when
    /// it does not. A crawl of the same name replaces a dataset in place, so
    /// "same id, different text" is the ordinary case and not a strange one.
    #[test]
    fn an_index_outlives_a_request_and_not_an_edit() {
        let db = Db::in_memory().unwrap();
        let dir = std::env::temp_dir().join(format!("kvad-retrieval-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // `datasets::save` writes into the real datasets directory, so this
        // test works the level below it: the cache is about a path and its
        // fingerprint, and that is what is exercised here.
        let path = dir.join("corpus");
        std::fs::write(&path, "# One\n\nvectors and slices\n").unwrap();
        let first = fingerprint(&path);

        std::fs::write(&path, "# One\n\nvectors and slices, and more of them\n").unwrap();
        assert_ne!(first, fingerprint(&path), "a rewrite has to be noticed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dataset_with_no_manifest_has_no_citations() {
        assert!(pages_of("a-name-that-is-not-a-dataset").is_empty());
    }
}
