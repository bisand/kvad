//! What runs are trained on: a text, or a folder of captioned pictures.
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
//!
//! # Pictures
//!
//! An image model is trained on a folder: pictures, and beside each a `.txt`
//! of the same name with its caption, which is the convention `kvad-gpu tune`
//! reads (`kvad_gpu::image::dataset`). Here that folder is
//! `datasets/NAME/`, and it arrives a file at a time: each is put in a
//! folder beside it that is not yet a dataset ([`stage`]), and [`keep`]
//! makes the dataset of them, or refuses the lot.
//!
//! The same check, at the same time, for the same reason. A picture with no
//! caption, a caption with no picture and a file that is not a picture each
//! end a run, and the run finds them one at a time and after loading two
//! text encoders. [`check_folder`] finds all of them before there is a
//! dataset at all, and names every one.

use crate::db::Db;
use nervus::text::CharTokenizer;
use rusqlite::{params, OptionalExtension};
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The largest file that may be uploaded: what a training run can hold, and
/// the reason for the number is [`kvad::train::MAX_CORPUS_BYTES`]'s to give.
pub const MAX_BYTES: usize = kvad::train::MAX_CORPUS_BYTES;

/// The largest picture that may be uploaded, and the most a folder of them
/// may hold.
///
/// Not the text's limit, because it is not the text's reason. A text run
/// holds its whole corpus in memory as token ids. An image run reads each
/// picture once, keeps a latent of it a few hundred kilobytes large, and
/// never opens the file again, so nothing about the loop limits a set. What
/// these keep out is a file that is not a photograph, and a disk filled by
/// accident: 64 MB is past any camera's JPEG, and 4 GB is several hundred of
/// those where a LoRA is trained on ten to fifty.
pub const MAX_PICTURE_BYTES: usize = 64 << 20;
pub const MAX_FOLDER_BYTES: u64 = 4 << 30;

/// What a picture's file may end in: what the trainer reads
/// (`kvad_gpu::image::dataset::KINDS`), said again here because a server
/// built without the GPU crate still keeps datasets.
pub const PICTURES: [&str; 5] = ["jpg", "jpeg", "png", "webp", "bmp"];

/// What a dataset is made of, which decides which loop can train on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Text,
    Pictures,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Pictures => "pictures",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Dataset {
    pub id: i64,
    pub name: String,
    pub kind: Kind,
    /// How many pictures a folder of them holds; `None` for a text.
    pub items: Option<i64>,
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
    let kind = match r.get::<_, String>("kind")?.as_str() {
        "pictures" => Kind::Pictures,
        _ => Kind::Text,
    };
    Ok(Dataset {
        id: r.get("id")?,
        kind,
        items: r.get("items")?,
        bytes: r.get("bytes")?,
        characters: r.get("characters")?,
        distinct: r.get("distinct_chars")?,
        created_at: r.get("created_at")?,
        source: r.get("source")?,
        manifest: manifest_of(&name).is_ok_and(|p| p.is_file()),
        // A text is a file and a set of pictures a folder, under one name.
        present: path_of(&name).is_ok_and(|p| match kind {
            Kind::Text => p.is_file(),
            Kind::Pictures => p.is_dir(),
        }),
        name,
    })
}

const COLUMNS: &str = "id, name, kind, items, bytes, characters, distinct_chars, created_at, source";

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
    // Replacing pictures of the same name: the folder goes, as a text would.
    if path.is_dir() {
        std::fs::remove_dir_all(&path)?;
    }
    std::fs::write(&path, text)?;

    let id = db.with(|c| {
        c.execute(
            "INSERT INTO datasets (name, bytes, characters, distinct_chars, owner, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(name) DO UPDATE SET
                bytes = ?2, characters = ?3, distinct_chars = ?4, source = ?6,
                kind = 'text', items = NULL, created_at = datetime('now')",
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
    let _ = match path.is_dir() {
        true => std::fs::remove_dir_all(&path),
        false => std::fs::remove_file(&path),
    };
    let _ = std::fs::remove_file(manifest_of(&dataset.name)?);
    Ok(db.with(|c| c.execute("DELETE FROM datasets WHERE id = ?1", [id]))? > 0)
}

pub fn read(db: &Db, id: i64) -> Res<(Dataset, String)> {
    let dataset = get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    if dataset.kind != Kind::Text {
        return Err(format!("`{}` is {}, not a text", dataset.name, dataset.kind.name()).into());
    }
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
    if dataset.kind != Kind::Text {
        return Err(format!("`{}` is pictures, and this trains on a text", dataset.name).into());
    }
    let path = path_of(&dataset.name)?;
    match path.is_file() {
        true => Ok(path),
        false => Err(format!("`{}` is listed but its file is gone", dataset.name).into()),
    }
}

// ---------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------

/// The folder a set of pictures is trained from, for the training options.
pub fn folder_for(db: &Db, id: i64) -> Res<(Dataset, PathBuf)> {
    let dataset = get(db, id)?.ok_or_else(|| format!("there is no dataset {id}"))?;
    if dataset.kind != Kind::Pictures {
        return Err(format!("`{}` is a text, and a LoRA for an image model is trained on pictures", dataset.name).into());
    }
    let path = path_of(&dataset.name)?;
    match path.is_dir() {
        true => Ok((dataset, path)),
        false => Err(format!("`{}` is listed but its folder is gone", dataset.name).into()),
    }
}

/// Where the files of an upload wait until [`keep`] makes a dataset of
/// them. Its name starts with a dot, which no dataset's may, so it is never
/// mistaken for one.
fn staging_of(root: &std::path::Path, name: &str) -> Res<PathBuf> {
    check_name(name)?;
    Ok(root.join(format!(".{name}.upload")))
}

fn ending(file: &str) -> String {
    std::path::Path::new(file).extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase()
}

/// What one file of a folder may be called: one path component, not hidden,
/// and a picture or a caption by its ending.
pub fn check_file(file: &str) -> Result<(), String> {
    let plain = !file.is_empty()
        && file.len() <= 200
        && !file.starts_with('.')
        && !file.chars().any(|c| matches!(c, '/' | '\\') || c.is_control());
    if !plain {
        return Err(format!("`{file}` is not a file's name: one name, with no folder in it, not starting with a dot"));
    }
    match ending(file).as_str() {
        e if e == "txt" || PICTURES.contains(&e) => Ok(()),
        _ => Err(format!("`{file}` is neither a picture ({}) nor a caption (txt)", PICTURES.join(", "))),
    }
}

/// Put one file of an upload beside the others, under `root`.
pub fn stage_in(root: &std::path::Path, name: &str, file: &str, bytes: &[u8]) -> Res<()> {
    let staging = staging_of(root, name)?;
    check_file(file)?;
    if bytes.len() > MAX_PICTURE_BYTES {
        return Err(format!("`{file}` is {} and one file may be {}", kvad::hub::human_bytes(bytes.len() as u64), kvad::hub::human_bytes(MAX_PICTURE_BYTES as u64)).into());
    }
    std::fs::create_dir_all(&staging)?;
    let held: u64 = std::fs::read_dir(&staging)?.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum();
    if held + bytes.len() as u64 > MAX_FOLDER_BYTES {
        return Err(format!("with `{file}` that is over {}, the most a set of pictures may hold", kvad::hub::human_bytes(MAX_FOLDER_BYTES)).into());
    }
    Ok(std::fs::write(staging.join(file), bytes)?)
}

pub fn stage(name: &str, file: &str, bytes: &[u8]) -> Res<()> {
    stage_in(&dir(), name, file, bytes)
}

/// Throw away an upload that was not kept.
pub fn unstage(name: &str) -> Res<()> {
    let staging = staging_of(&dir(), name)?;
    match std::fs::remove_dir_all(&staging) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

/// One picture of a set, and what it is captioned.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Picture {
    pub file: String,
    pub caption: String,
}

/// Every picture in `folder` with its caption, or everything that is wrong
/// with the folder, all of it, in one message.
///
/// `fallback` is the caption of a picture that has none beside it. `decodes`
/// is asked of every picture, and answers with why not; the server asks
/// `ffmpeg`, which is what the trainer will read them with.
///
/// What is wrong is one of four things, and each names its files: a picture
/// with no caption, a caption that is empty, a caption with no picture
/// (which is usually a picture with a mistyped name, and so also a picture
/// with no caption), and a picture that does not decode.
pub fn check_folder(folder: &std::path::Path, fallback: Option<&str>, decodes: &dyn Fn(&std::path::Path) -> Result<(), String>) -> Result<Vec<Picture>, String> {
    let mut files: Vec<String> = std::fs::read_dir(folder)
        .map_err(|e| format!("no files were uploaded: {e}"))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|f| !f.starts_with('.'))
        .collect();
    files.sort();
    let stem = |f: &str| std::path::Path::new(f).file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
    let pictures: Vec<&String> = files.iter().filter(|f| PICTURES.contains(&ending(f).as_str())).collect();
    if pictures.is_empty() {
        return Err(format!("there are no pictures in that: none of its files end in {}", PICTURES.join(", ")));
    }

    let (mut kept, mut bare, mut empty, mut broken) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for file in &pictures {
        let caption = match std::fs::read_to_string(folder.join(file).with_extension("txt")) {
            Ok(text) => Some(text.split_whitespace().collect::<Vec<_>>().join(" ")),
            Err(_) => fallback.map(|c| c.split_whitespace().collect::<Vec<_>>().join(" ")),
        };
        match caption {
            None => bare.push(file.as_str()),
            Some(c) if c.is_empty() => empty.push(file.as_str()),
            Some(caption) => kept.push(Picture { file: file.to_string(), caption }),
        }
        if let Err(why) = decodes(&folder.join(file)) {
            broken.push(format!("{file} ({why})"));
        }
    }
    let orphans: Vec<&str> = files
        .iter()
        .filter(|f| ending(f) == "txt" && !pictures.iter().any(|p| stem(p) == stem(f)))
        .map(String::as_str)
        .collect();
    // Two pictures of one name share one caption file, and the trainer's
    // cache: `a.jpg` and `a.png` are refused rather than guessed between.
    let twins: Vec<&str> = pictures.windows(2).filter(|w| stem(w[0]) == stem(w[1])).map(|w| w[1].as_str()).collect();

    let mut wrong = Vec::new();
    let listed = |files: &[&str]| files.join(", ");
    if !bare.is_empty() {
        wrong.push(format!("{} with no caption: {}. A caption is a .txt of the same name, or give one caption for every picture without", bare.len(), listed(&bare)));
    }
    if !empty.is_empty() {
        wrong.push(format!("{} whose caption is empty: {}", empty.len(), listed(&empty)));
    }
    if !orphans.is_empty() {
        wrong.push(format!("{} with no picture of the same name: {}", if orphans.len() == 1 { "a caption".to_string() } else { format!("{} captions", orphans.len()) }, listed(&orphans)));
    }
    if !twins.is_empty() {
        wrong.push(format!("{} sharing a name, and so a caption, with another picture: {}", twins.len(), listed(&twins)));
    }
    if !broken.is_empty() {
        wrong.push(format!("{} that could not be read as a picture: {}", broken.len(), broken.join(", ")));
    }
    match wrong.is_empty() {
        true => Ok(kept),
        false => Err(format!("{} pictures, and nothing was kept: {}", pictures.len(), wrong.join("; "))),
    }
}

/// Make a dataset of what was uploaded under `name`, if all of it is sound.
///
/// A picture that took the fallback caption is given a `.txt` saying so, so
/// that the folder means the same thing to anything else that reads it.
/// Refused, the upload stays where it is: the fix is usually one more file.
pub fn keep_in(db: &Db, root: &std::path::Path, name: &str, fallback: Option<&str>, owner: Option<i64>, decodes: &dyn Fn(&std::path::Path) -> Result<(), String>) -> Res<Dataset> {
    let staging = staging_of(root, name)?;
    let pictures = check_folder(&staging, fallback.filter(|c| !c.trim().is_empty()), decodes)?;
    for p in &pictures {
        let caption = staging.join(&p.file).with_extension("txt");
        if !caption.is_file() {
            std::fs::write(&caption, &p.caption)?;
        }
    }
    let bytes: u64 = std::fs::read_dir(&staging)?.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum();

    let path = root.join(name);
    match path.is_dir() {
        true => std::fs::remove_dir_all(&path)?,
        false => drop(std::fs::remove_file(&path)),
    }
    let _ = std::fs::remove_file(root.join(format!("{name}.crawl.json")));
    std::fs::rename(&staging, &path)?;

    let id = db.with(|c| {
        c.execute(
            "INSERT INTO datasets (name, bytes, characters, distinct_chars, owner, kind, items)
             VALUES (?1, ?2, 0, 0, ?3, 'pictures', ?4)
             ON CONFLICT(name) DO UPDATE SET
                bytes = ?2, characters = 0, distinct_chars = 0, source = NULL,
                kind = 'pictures', items = ?4, created_at = datetime('now')",
            params![name, bytes as i64, owner, pictures.len() as i64],
        )?;
        c.query_row("SELECT id FROM datasets WHERE name = ?1", [name], |r| r.get(0))
    })?;
    get(db, id)?.ok_or_else(|| "the dataset vanished as it was written".into())
}

pub fn keep(db: &Db, name: &str, fallback: Option<&str>, owner: Option<i64>, decodes: &dyn Fn(&std::path::Path) -> Result<(), String>) -> Res<Dataset> {
    keep_in(db, &dir(), name, fallback, owner, decodes)
}

/// The pictures of a set, by name, each with its caption.
pub fn pictures(db: &Db, id: i64) -> Res<(Dataset, Vec<Picture>)> {
    let (dataset, folder) = folder_for(db, id)?;
    let pictures = check_folder(&folder, None, &|_| Ok(()))?;
    Ok((dataset, pictures))
}

/// One picture's file, for showing it.
pub fn picture_file(db: &Db, id: i64, file: &str) -> Res<PathBuf> {
    let (_, folder) = folder_for(db, id)?;
    check_file(file)?;
    let path = folder.join(file);
    match PICTURES.contains(&ending(file).as_str()) && path.is_file() {
        true => Ok(path),
        false => Err(format!("there is no picture `{file}` in that dataset").into()),
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

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvad-datasets-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A file "decodes" in these tests unless it says it does not: what
    /// reads a real picture is `ffmpeg`, and the check under test is what is
    /// done with its answer.
    fn decodes(path: &std::path::Path) -> Result<(), String> {
        match std::fs::read(path).unwrap().starts_with(b"broken") {
            true => Err("not a picture".into()),
            false => Ok(()),
        }
    }

    /// The check #77 asks for: an upload with one picture that has no
    /// caption is refused, and the message names the picture. And every
    /// other thing wrong with it is in the same message.
    #[test]
    fn an_upload_is_refused_whole_and_every_fault_is_named() {
        let root = scratch("refused");
        let db = Db::in_memory().unwrap();
        for (file, body) in [("one.jpg", "x"), ("one.txt", "a dog"), ("two.png", "x")] {
            stage_in(&root, "dogs", file, body.as_bytes()).unwrap();
        }
        let refused = keep_in(&db, &root, "dogs", None, None, &decodes).unwrap_err().to_string();
        assert!(refused.contains("two.png") && refused.contains("no caption"), "{refused}");
        assert!(!refused.contains("one.jpg"), "a picture with a caption was named: {refused}");
        assert!(list(&db).unwrap().is_empty() && !root.join("dogs").exists(), "a refused upload was kept");

        // Three more faults, and all four are said at once.
        stage_in(&root, "dogs", "three.webp", b"broken").unwrap();
        stage_in(&root, "dogs", "three.txt", b" \n").unwrap();
        stage_in(&root, "dogs", "tow.txt", b"a cat").unwrap();
        let refused = keep_in(&db, &root, "dogs", None, None, &decodes).unwrap_err().to_string();
        for named in ["two.png", "three.webp (not a picture)", "caption is empty: three.webp", "tow.txt"] {
            assert!(refused.contains(named), "`{named}` is not in: {refused}");
        }

        // What cannot be a file of a set never reaches the folder.
        for bad in ["../x.png", "a/b.png", ".hidden.png", "notes.md", "", "x"] {
            assert!(stage_in(&root, "dogs", bad, b"x").is_err(), "`{bad}` was taken");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Kept, a set is a folder under its name and a row that says how many
    /// pictures; a fallback caption is written beside each picture it was
    /// given to; and a text of the same name replaces it, as it would a text.
    #[test]
    fn a_sound_upload_becomes_a_folder_and_a_row() {
        let root = scratch("kept");
        let db = Db::in_memory().unwrap();
        for (file, body) in [("b.PNG", "x"), ("a.jpg", "xx"), ("a.txt", " a photo\nof a dog ")] {
            stage_in(&root, "dogs", file, body.as_bytes()).unwrap();
        }
        let kept = keep_in(&db, &root, "dogs", Some("a thing"), None, &decodes).unwrap();
        assert_eq!((kept.kind, kept.items, kept.name.as_str()), (Kind::Pictures, Some(2), "dogs"));
        assert!(root.join("dogs").is_dir() && !root.join(".dogs.upload").exists());
        assert_eq!(std::fs::read_to_string(root.join("dogs/b.txt")).unwrap(), "a thing");
        let found = check_folder(&root.join("dogs"), None, &decodes).unwrap();
        assert_eq!(found, [Picture { file: "a.jpg".into(), caption: "a photo of a dog".into() }, Picture { file: "b.PNG".into(), caption: "a thing".into() }]);

        // A second upload of the name replaces the first, whole.
        stage_in(&root, "dogs", "c.png", b"x").unwrap();
        stage_in(&root, "dogs", "c.txt", b"a cat").unwrap();
        let again = keep_in(&db, &root, "dogs", None, None, &decodes).unwrap();
        assert_eq!((again.id, again.items), (kept.id, Some(1)));
        assert!(!root.join("dogs/a.jpg").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A dataset from before there were two kinds is a text.
    #[test]
    fn a_row_written_before_there_were_pictures_is_a_text() {
        let db = Db::in_memory().unwrap();
        db.with(|c| c.execute("INSERT INTO datasets (name, bytes, characters, distinct_chars) VALUES ('old', 3, 3, 2)", [])).unwrap();
        let old = &list(&db).unwrap()[0];
        assert_eq!((old.kind, old.items), (Kind::Text, None));
        assert!(folder_for(&db, old.id).unwrap_err().to_string().contains("is a text"));
    }

    /// The endings a set may hold are the ones the trainer reads.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_pictures_kept_are_the_pictures_the_trainer_reads() {
        assert_eq!(PICTURES, kvad_gpu::image::dataset::KINDS);
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
