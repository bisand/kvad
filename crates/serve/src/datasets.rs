//! Text to train on, and pictures.
//!
//! The file on disk is the truth, as it is for models: these live in
//! `$XDG_DATA_HOME/kvad/datasets`, and the table holds what a listing wants
//! to show without opening every one of them again.
//!
//! A dataset of pictures is a folder there instead of a file: the pictures,
//! and beside each a `.txt` of the same name with its caption, which is the
//! folder `kvad-gpu tune` reads (and kohya's scripts, and diffusers'). It is
//! filled a file at a time, so nothing about it is kept in the table but
//! that it is one; see [`pictures`].
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
use nervus::text::CharTokenizer;
use rusqlite::{params, OptionalExtension};
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The largest file that may be uploaded: what a training run can hold, and
/// the reason for the number is [`kvad::train::MAX_CORPUS_BYTES`]'s to give.
pub const MAX_BYTES: usize = kvad::train::MAX_CORPUS_BYTES;

/// What a picture's file may end in: `kvad_gpu::image::dataset::KINDS`,
/// said again because a server built without the GPU crate still keeps the
/// folders. A test holds the two together.
pub const PICTURE_KINDS: [&str; 5] = ["jpg", "jpeg", "png", "webp", "bmp"];

/// The largest picture that may be uploaded. A photograph from a phone is
/// 3 to 12 MB, and every picture is shrunk to at most 1536 a side before
/// anything is trained on it.
pub const MAX_PICTURE_BYTES: usize = 32 * 1024 * 1024;

/// The most pictures in one dataset. A LoRA is trained on tens, and a
/// thousand steps see each of a thousand pictures once.
pub const MAX_PICTURES: usize = 2000;

/// The longest caption: far past the 77 tokens CLIP reads of one.
pub const MAX_CAPTION_BYTES: usize = 8 * 1024;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Dataset {
    pub id: i64,
    pub name: String,
    /// `text`, or `pictures`.
    pub kind: String,
    /// How many pictures a dataset of them holds, and how many of those
    /// have no caption beside them. Both 0 for a text.
    pub pictures: i64,
    pub uncaptioned: i64,
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
    let kind: String = r.get("kind")?;
    // Counted from the folder each time: it is filled a file at a time, and
    // a count kept in the table would be wrong after any one of them failed.
    let held = match kind.as_str() {
        "pictures" => pictures(&name).ok(),
        _ => None,
    };
    let present = match kind.as_str() {
        "pictures" => held.is_some(),
        _ => path_of(&name).is_ok_and(|p| p.is_file()),
    };
    let held = held.unwrap_or_default();
    Ok(Dataset {
        id: r.get("id")?,
        bytes: match kind.as_str() {
            "pictures" => held.iter().map(|p| p.bytes as i64).sum(),
            _ => r.get("bytes")?,
        },
        pictures: held.len() as i64,
        uncaptioned: held.iter().filter(|p| p.caption.is_none()).count() as i64,
        characters: r.get("characters")?,
        distinct: r.get("distinct_chars")?,
        created_at: r.get("created_at")?,
        source: r.get("source")?,
        manifest: manifest_of(&name).is_ok_and(|p| p.is_file()),
        present,
        kind,
        name,
    })
}

const COLUMNS: &str = "id, name, kind, bytes, characters, distinct_chars, created_at, source";

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
    if path.is_dir() {
        return Err(format!("`{name}` is a dataset of pictures; a text needs another name").into());
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
    let path = path_of(&dataset.name)?;
    match dataset.kind.as_str() {
        "pictures" => drop(std::fs::remove_dir_all(&path)),
        _ => drop(std::fs::remove_file(&path)),
    }
    let _ = std::fs::remove_file(manifest_of(&dataset.name)?);
    Ok(db.with(|c| c.execute("DELETE FROM datasets WHERE id = ?1", [id]))? > 0)
}

pub fn read(db: &Db, id: i64) -> Res<(Dataset, String)> {
    let dataset = text_one(db, id)?;
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
    let dataset = text_one(db, id)?;
    let path = path_of(&dataset.name)?;
    match path.is_file() {
        true => Ok(path),
        false => Err(format!("`{}` is listed but its file is gone", dataset.name).into()),
    }
}

/// The dataset `id`, which has to be a text: what a language model is
/// trained on, searched and read back as.
fn text_one(db: &Db, id: i64) -> Res<Dataset> {
    let dataset = get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    match dataset.kind.as_str() {
        "text" => Ok(dataset),
        _ => Err(format!("`{}` is a dataset of pictures, not a text", dataset.name).into()),
    }
}

// ---------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------

/// One picture of a dataset, and the caption beside it if there is one.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Picture {
    pub file: String,
    pub bytes: u64,
    pub caption: Option<String>,
}

/// What a file in a dataset of pictures is, by its name: `Some(true)` a
/// picture, `Some(false)` a caption, `None` neither.
fn picture_or_caption(file: &str) -> Option<bool> {
    let ending = file.rsplit_once('.')?.1.to_ascii_lowercase();
    match ending.as_str() {
        "txt" => Some(false),
        e if PICTURE_KINDS.contains(&e) => Some(true),
        _ => None,
    }
}

/// What a file in a dataset may be called: one path component, of the
/// characters a dataset's own name may have, ending as a picture or a
/// caption does. Returns whether it is a picture.
///
/// Stricter than a file system needs, for the reason [`check_name`] is: the
/// path is built by joining. A browser's file names are freer than this,
/// and the page that uploads them renames what does not pass.
pub fn check_file(file: &str) -> Result<bool, String> {
    let ok = !file.starts_with('.')
        && file.len() <= 128
        && file.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ' '));
    match (ok, picture_or_caption(file)) {
        (true, Some(picture)) => Ok(picture),
        _ => Err(format!(
            "`{file}` is not a file a dataset of pictures holds: letters, digits, spaces and `-`, `_` or `.`, ending in {} for a picture or .txt for its caption",
            PICTURE_KINDS.join(", ")
        )),
    }
}

/// The format a picture's first bytes say it is, as one of
/// [`PICTURE_KINDS`]' families.
///
/// Not a decode: `ffmpeg` does that when a run reads the pictures, and names
/// the one it cannot. This catches what an upload gets wrong most, a file
/// that is not a picture at all under a picture's name, while somebody is
/// still looking at the page that sent it.
pub fn sniff(bytes: &[u8]) -> Option<&'static str> {
    match bytes {
        [0xFF, 0xD8, 0xFF, ..] => Some("jpeg"),
        [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, ..] => Some("png"),
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => Some("webp"),
        [b'B', b'M', ..] => Some("bmp"),
        _ => None,
    }
}

/// Start an empty dataset of pictures, or find the one of that name.
///
/// Not a replacement, where [`save`] is one: pictures arrive a file at a
/// time, and the second upload into `my-dog` adds to the first.
pub fn create_pictures(db: &Db, name: &str, owner: Option<i64>) -> Res<Dataset> {
    let path = path_of(name)?;
    if path.is_file() {
        return Err(format!("`{name}` is a text dataset; pictures need another name").into());
    }
    std::fs::create_dir_all(&path)?;
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO datasets (name, kind, bytes, characters, distinct_chars, owner)
             VALUES (?1, 'pictures', 0, 0, 0, ?2)
             ON CONFLICT(name) DO NOTHING",
            params![name, owner],
        )?;
        c.query_row("SELECT id FROM datasets WHERE name = ?1", [name], |r| r.get(0))
    })?;
    get(db, id)?.ok_or_else(|| "the dataset vanished as it was made".into())
}

/// The pictures in the dataset called `name`, by file name, each with its
/// caption as a run would read it: on one line.
pub fn pictures(name: &str) -> Res<Vec<Picture>> {
    pictures_at(&path_of(name)?)
}

fn pictures_at(dir: &std::path::Path) -> Res<Vec<Picture>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("could not read {}: {e}", dir.display()))? {
        let entry = entry?;
        let file = entry.file_name().to_string_lossy().into_owned();
        if check_file(&file) != Ok(true) || !entry.path().is_file() {
            continue;
        }
        let caption = std::fs::read_to_string(entry.path().with_extension("txt"))
            .ok()
            .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|t| !t.is_empty());
        found.push(Picture { file, bytes: entry.metadata().map_or(0, |m| m.len()), caption });
    }
    found.sort_by(|a, b| a.file.cmp(&b.file));
    Ok(found)
}

/// The dataset `id`, which has to be one of pictures, and its folder.
pub fn picture_dir(db: &Db, id: i64) -> Res<(Dataset, PathBuf)> {
    let dataset = get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    if dataset.kind != "pictures" {
        return Err(format!("`{}` is a text, not a dataset of pictures", dataset.name).into());
    }
    let dir = path_of(&dataset.name)?;
    Ok((dataset, dir))
}

/// Put one file into a dataset of pictures: a picture, or a caption.
///
/// A picture replaces one of the same name, and so does a caption. A second
/// picture with the same name before its ending (`a.jpg` beside `a.png`) is
/// refused: the two would share one caption file, and a run would read the
/// same caption for both without saying so.
pub fn put_file(db: &Db, id: i64, file: &str, bytes: &[u8]) -> Res<()> {
    let (dataset, dir) = picture_dir(db, id)?;
    put_at(&dir, &dataset.name, file, bytes)
}

fn put_at(dir: &std::path::Path, name: &str, file: &str, bytes: &[u8]) -> Res<()> {
    match check_file(file)? {
        true => {
            if bytes.len() > MAX_PICTURE_BYTES {
                return Err(format!(
                    "{file} is {} and the limit for a picture is {}",
                    kvad::hub::human_bytes(bytes.len() as u64),
                    kvad::hub::human_bytes(MAX_PICTURE_BYTES as u64)
                )
                .into());
            }
            if sniff(bytes).is_none() {
                return Err(format!("{file} is not a picture: it does not begin as a JPEG, PNG, WebP or BMP does").into());
            }
            let held = match dir.is_dir() {
                true => pictures_at(dir)?,
                false => Vec::new(),
            };
            let stem = |f: &str| f.rsplit_once('.').map_or(f.to_string(), |(s, _)| s.to_string());
            if let Some(twin) = held.iter().find(|p| p.file != file && stem(&p.file) == stem(file)) {
                return Err(format!("{file} would share its caption with {}, which is already here: rename one", twin.file).into());
            }
            if held.len() >= MAX_PICTURES && !held.iter().any(|p| p.file == file) {
                return Err(format!("`{name}` already holds {MAX_PICTURES} pictures, which is the most one may").into());
            }
        }
        false => {
            if bytes.len() > MAX_CAPTION_BYTES {
                return Err(format!("{file} is {} bytes, and a caption is at most {MAX_CAPTION_BYTES}", bytes.len()).into());
            }
            if std::str::from_utf8(bytes).is_err() {
                return Err(format!("{file} is not text in UTF-8, which a caption has to be").into());
            }
        }
    }
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(file), bytes)?;
    Ok(())
}

/// Take one file out: a picture, and its caption with it, or a caption
/// alone. False if it was not there.
pub fn remove_file(db: &Db, id: i64, file: &str) -> Res<bool> {
    let (_, dir) = picture_dir(db, id)?;
    remove_at(&dir, file)
}

fn remove_at(dir: &std::path::Path, file: &str) -> Res<bool> {
    let picture = check_file(file)?;
    let path = dir.join(file);
    if !path.is_file() {
        return Ok(false);
    }
    std::fs::remove_file(&path)?;
    if picture {
        let _ = std::fs::remove_file(path.with_extension("txt"));
    }
    Ok(true)
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

    /// A file's name is joined onto the dataset's folder, and says by its
    /// ending which of the two things in one it is.
    #[test]
    fn a_file_in_a_dataset_of_pictures_is_a_picture_or_a_caption() {
        for (file, picture) in [("one.jpg", true), ("Two 2.PNG", true), ("a-b_c.webp", true), ("one.txt", false)] {
            assert_eq!(check_file(file), Ok(picture), "{file}");
        }
        for bad in ["", "one", "one.gif", "../one.jpg", "a/b.png", ".hidden.png", "one.jpg\n", "é.png"] {
            assert!(check_file(bad).is_err(), "`{bad}` was accepted");
        }
    }

    #[test]
    fn a_picture_is_known_by_how_it_begins() {
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0]), Some("jpeg"));
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n...."), Some("png"));
        assert_eq!(sniff(b"RIFF\0\0\0\0WEBPVP8 "), Some("webp"));
        assert_eq!(sniff(b"BM......"), Some("bmp"));
        assert_eq!(sniff(b"<html>"), None);
        assert_eq!(sniff(b""), None);
    }

    /// The folder is what is counted, a file at a time: pictures with and
    /// without captions, a twin refused, a picture's caption gone with it.
    #[test]
    fn a_dataset_of_pictures_is_its_folder() {
        let dir = std::env::temp_dir().join(format!("kvad-pictures-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let png = b"\x89PNG\r\n\x1a\n....";

        put_at(&dir, "my dog", "one.png", png).unwrap();
        put_at(&dir, "my dog", "two.png", png).unwrap();
        put_at(&dir, "my dog", "one.txt", b"a  photo of\nsks dog ").unwrap();
        let held = pictures_at(&dir).unwrap();
        assert_eq!(held.iter().map(|p| p.file.as_str()).collect::<Vec<_>>(), ["one.png", "two.png"]);
        assert_eq!(held[0].caption.as_deref(), Some("a photo of sks dog"), "on one line, as a run reads it");
        assert_eq!((held[1].caption.as_deref(), held[1].bytes), (None, png.len() as u64));

        let said = put_at(&dir, "my dog", "one.jpg", &[0xFF, 0xD8, 0xFF, 0xE0]).unwrap_err().to_string();
        assert!(said.contains("one.png"), "{said}");
        assert!(put_at(&dir, "my dog", "three.png", b"<html>").is_err(), "not a picture");
        assert!(put_at(&dir, "my dog", "notes.md", b"x").is_err());
        assert!(put_at(&dir, "my dog", "two.txt", &[0xFF, 0xFE]).is_err(), "not UTF-8");

        assert!(remove_at(&dir, "one.png").unwrap());
        assert!(!remove_at(&dir, "one.png").unwrap());
        assert!(!dir.join("one.txt").exists(), "the caption went with its picture");
        assert_eq!(pictures_at(&dir).unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A text's routes refuse a dataset of pictures by what it is, where
    /// they used to have only "its file is gone" to say.
    #[test]
    fn a_dataset_of_pictures_is_not_read_as_a_text() {
        let db = Db::in_memory().unwrap();
        db.with(|c| c.execute("INSERT INTO datasets (name, kind, bytes, characters, distinct_chars) VALUES ('my dog', 'pictures', 0, 0, 0)", [])).unwrap();
        let id = list(&db).unwrap()[0].id;
        for said in [read(&db, id).unwrap_err().to_string(), file_for(&db, id).unwrap_err().to_string()] {
            assert!(said.contains("pictures, not a text"), "{said}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_pictures_kept_are_the_pictures_a_run_reads() {
        assert_eq!(PICTURE_KINDS, kvad_gpu::image::dataset::KINDS);
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
        assert!(nervus::text::CharTokenizer::load(&dir).unwrap().encode("a cab, a zebra!").is_err());

        assert!(unseen("no/such/model", "x").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
