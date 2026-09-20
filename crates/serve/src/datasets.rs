//! Text to train on.
//!
//! The file on disk is the truth, as it is for models: these live in
//! `$XDG_DATA_HOME/kvad/datasets`, and the table holds what a listing wants
//! to show without opening every one of them again.
//!
//! # The check worth having
//!
//! A character tokeniser gives ids to the characters it saw, and a model has
//! one row per id. Training an existing model further on a text containing a
//! character it has never seen therefore cannot work, and `kvad train`
//! refuses — after reading the file, naming the first offender. That is the
//! right error at the wrong time: by then somebody has chosen a model, chosen
//! a dataset, and pressed a button.
//!
//! [`unseen`] answers the same question before any of that, and answers it
//! completely: not the first character that would fail but every one, so the
//! fix is one edit rather than a dozen rounds of the same message.

use crate::db::Db;
use nanograd::text::CharTokenizer;
use rusqlite::{params, OptionalExtension};
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The largest file that may be uploaded.
///
/// `nanograd` holds the whole corpus in memory as a `Vec<usize>`, eight bytes
/// a character, so 64 MB of text is half a gigabyte before training starts.
/// The limit is about this machine, not about taste.
pub const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Dataset {
    pub id: i64,
    pub name: String,
    pub bytes: i64,
    /// Characters, not bytes: a corpus of Norwegian is longer in bytes than
    /// it is in anything a model sees.
    pub characters: i64,
    pub distinct: i64,
    pub created_at: String,
    /// Where it came from: the address a crawl started at, or `None` for a
    /// file somebody uploaded.
    pub source: Option<String>,
    /// Whether a crawl's manifest is beside the file — every page it read, in
    /// the order it read them.
    pub manifest: bool,
    /// False when the row is here and the file is not — somebody tidied the
    /// directory by hand, and a listing that pretended otherwise would fail
    /// at the point of training.
    pub present: bool,
}

/// Where uploaded corpora live.
pub fn dir() -> PathBuf {
    kvad::weights::data_dir().join("datasets")
}

/// What a dataset may be called.
///
/// The same rule as a model name and for the same reason: the path is built
/// by joining, and a "name" of `../../.ssh` would join right out of the
/// directory.
pub fn check_name(name: &str) -> Result<(), String> {
    let ok = kvad::weights::is_model_name(name)
        && !name.starts_with('.')
        && name.len() <= 96
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ' '));
    match ok {
        true => Ok(()),
        false => Err(format!(
            "`{name}` is not a dataset name: one word of letters, digits, spaces and \
             `-`, `_` or `.`, not starting with a dot"
        )),
    }
}

pub fn path_of(name: &str) -> Res<PathBuf> {
    check_name(name)?;
    Ok(dir().join(name))
}

/// Where a crawl's record of itself goes: beside the text, under the same
/// name, so that moving one and forgetting the other takes an effort.
pub fn manifest_of(name: &str) -> Res<PathBuf> {
    check_name(name)?;
    Ok(dir().join(format!("{name}.crawl.json")))
}

fn row(r: &rusqlite::Row) -> rusqlite::Result<Dataset> {
    let name: String = r.get("name")?;
    Ok(Dataset {
        id: r.get("id")?,
        bytes: r.get("bytes")?,
        characters: r.get("characters")?,
        distinct: r.get("distinct_chars")?,
        created_at: r.get("created_at")?,
        source: r.get("source")?,
        manifest: manifest_of(&name).is_ok_and(|p| p.is_file()),
        present: path_of(&name).is_ok_and(|p| p.is_file()),
        name,
    })
}

const COLUMNS: &str = "id, name, bytes, characters, distinct_chars, created_at, source";

pub fn list(db: &Db) -> Res<Vec<Dataset>> {
    db.with(|c| {
        let mut q = c.prepare(&format!("SELECT {COLUMNS} FROM datasets ORDER BY name"))?;
        let rows = q.query_map([], row)?.collect();
        rows
    })
}

pub fn get(db: &Db, id: i64) -> Res<Option<Dataset>> {
    db.with(|c| {
        c.query_row(&format!("SELECT {COLUMNS} FROM datasets WHERE id = ?1"), [id], row).optional()
    })
}

/// Write a corpus and remember it.
///
/// Replaces a dataset of the same name, because the alternative is
/// `shakespeare-2` and then `shakespeare-2-final`. What it does not do is
/// touch any model already trained on the old text: that model has the
/// tokeniser it was trained with and does not care what this file says now.
pub fn save(
    db: &Db,
    name: &str,
    text: &str,
    owner: Option<i64>,
    source: Option<&str>,
) -> Res<Dataset> {
    let path = path_of(name)?;
    if text.len() > MAX_BYTES {
        return Err(format!(
            "that is {} and the limit is {}",
            kvad::hub::human_bytes(text.len() as u64),
            kvad::hub::human_bytes(MAX_BYTES as u64)
        )
        .into());
    }
    if text.trim().is_empty() {
        return Err("there is no text in that file".into());
    }

    let (characters, distinct) = count(text);
    std::fs::create_dir_all(dir())?;
    std::fs::write(&path, text)?;

    let id = db.with(|c| {
        c.execute(
            "INSERT INTO datasets (name, bytes, characters, distinct_chars, owner, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(name) DO UPDATE SET
                bytes = ?2, characters = ?3, distinct_chars = ?4, source = ?6,
                created_at = datetime('now')",
            params![name, text.len() as i64, characters, distinct, owner, source],
        )?;
        c.query_row("SELECT id FROM datasets WHERE name = ?1", [name], |r| r.get(0))
    })?;
    get(db, id)?.ok_or_else(|| "the dataset vanished as it was written".into())
}

/// How much text there is, and how many different characters are in it.
///
/// The second number is the vocabulary a model trained on this would have,
/// which is what decides the size of its embedding table and its output head.
pub fn count(text: &str) -> (i64, i64) {
    let mut seen: Vec<char> = text.chars().collect();
    let characters = seen.len() as i64;
    seen.sort_unstable();
    seen.dedup();
    (characters, seen.len() as i64)
}

pub fn delete(db: &Db, id: i64) -> Res<bool> {
    let Some(dataset) = get(db, id)? else { return Ok(false) };
    // The row goes whether or not the file does: a row pointing at nothing is
    // worse than a file nobody knows about. The manifest goes with the text
    // it describes; on its own it would be a record of a corpus that is not
    // here any more.
    let _ = std::fs::remove_file(path_of(&dataset.name)?);
    let _ = std::fs::remove_file(manifest_of(&dataset.name)?);
    Ok(db.with(|c| c.execute("DELETE FROM datasets WHERE id = ?1", [id]))? > 0)
}

pub fn read(db: &Db, id: i64) -> Res<(Dataset, String)> {
    let dataset = get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    let path = path_of(&dataset.name)?;
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    Ok((dataset, text))
}

/// Every character in `text` that `model` has no token for.
///
/// `model` is a model trained here, by name. Returns an empty list for a text
/// it could be trained further on, and for a model that is not one of ours —
/// the question does not apply to a downloaded checkpoint with a real BPE
/// tokeniser, which has a token for everything.
pub fn unseen(model: &str, text: &str) -> Res<Vec<char>> {
    let Some(dir) = kvad::weights::local_dir(model) else {
        return Err(format!("`{model}` is not a model on this machine").into());
    };
    let Ok(tok) = CharTokenizer::load(&dir) else {
        // No character tokeniser, so this is not a model `kvad train` made
        // and `--from` would refuse it for a different reason entirely.
        return Ok(Vec::new());
    };
    let known = tok.chars();
    let mut missing: Vec<char> =
        text.chars().filter(|c| known.binary_search(c).is_err()).collect();
    missing.sort_unstable();
    missing.dedup();
    Ok(missing)
}

/// Whether a directory holds a model `--from` could continue: one this
/// repository trained, with a character tokeniser beside its weights.
pub fn continuable(name: &str) -> bool {
    kvad::weights::local_dir(name).is_some_and(|d| CharTokenizer::load(&d).is_ok())
}

/// A path under [`dir`], for the training options.
pub fn file_for(db: &Db, id: i64) -> Res<PathBuf> {
    let dataset = get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    let path = path_of(&dataset.name)?;
    match path.is_file() {
        true => Ok(path),
        false => Err(format!("`{}` is listed but its file is gone", dataset.name).into()),
    }
}

/// The models a run could continue from, for the picker.
pub fn continuable_models() -> Vec<String> {
    kvad::hub::trained_models()
        .into_iter()
        .filter(|m| m.complete && continuable(&m.id))
        .map(|m| m.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name is one path component, and everything that could climb out of
    /// the datasets directory is refused before anything is joined.
    #[test]
    fn a_dataset_name_cannot_leave_its_directory() {
        for good in ["shakespeare", "my corpus", "notes.v2", "a-b_c"] {
            assert!(check_name(good).is_ok(), "`{good}` was refused");
        }
        for bad in ["", ".", "..", "../escape", "a/b", "/etc/passwd", ".hidden", "a\nb", "a;b"] {
            assert!(check_name(bad).is_err(), "`{bad}` was accepted");
            assert!(path_of(bad).is_err());
        }
    }

    #[test]
    fn counting_is_characters_and_distinct_characters_not_bytes() {
        // Six characters, nine bytes: the vocabulary a model would get is
        // about characters, and so is the context window.
        let (characters, distinct) = count("héllo!");
        assert_eq!(characters, 6);
        assert_eq!(distinct, 5, "the two l's share an id");
        assert_eq!("héllo!".len(), 7);

        assert_eq!(count(""), (0, 0));
    }

    /// The check that saves somebody from choosing a model, choosing a
    /// dataset, pressing a button and waiting for the same refusal.
    #[test]
    fn unseen_characters_are_all_reported_not_just_the_first() {
        let dir = std::env::temp_dir().join(format!("kvad-unseen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        CharTokenizer::from_text("abc ").save(&dir).unwrap();
        let name = dir.to_str().unwrap();

        assert!(unseen(name, "a cab").unwrap().is_empty());

        // Two characters missing, and both are named — `encode` would have
        // reported only the first.
        let missing = unseen(name, "a cab, a zebra!").unwrap();
        assert_eq!(missing, ['!', ',', 'e', 'r', 'z'], "sorted, so the message reads the same way twice");
        assert!(nanograd::text::CharTokenizer::load(&dir).unwrap().encode("a cab, a zebra!").is_err());

        assert!(unseen("no/such/model", "x").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
