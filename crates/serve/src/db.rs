//! SQLite, and the migrations that shape it.
//!
//! # Why the filesystem still wins
//!
//! The database never claims a model exists. The HuggingFace cache and
//! `weights::models_dir()` are the truth about what is on this machine, and
//! `kvad ls`, the TUI and the web UI all read them directly. What goes in
//! here is what the filesystem cannot answer: conversations, jobs, training
//! metrics, settings, users. Two sources of truth about which models are
//! present would drift the first time somebody deleted a directory by hand.
//!
//! # Migrations
//!
//! Numbered `.sql` files, embedded in the binary with `include_str!`, applied
//! in order inside one transaction each. `PRAGMA user_version` records how
//! many have run: it is four bytes in the database header, needs no table of
//! its own, and cannot get out of step with the schema it describes because
//! it is written by the same transaction.
//!
//! Migrations are append-only. Editing one that has already run would leave
//! two databases with the same `user_version` and different schemas, which is
//! the one failure mode this design cannot detect.
//!
//! # One connection, one thread
//!
//! `rusqlite` is synchronous, so a connection cannot be held across an
//! `.await`. This wraps one in a mutex and every use of it happens inside
//! `spawn_blocking`. That is enough while the writes are a settings change
//! and a chat message; when it stops being enough the answer is a pool, and
//! [`Db::with`] is the seam it goes behind.

use rusqlite::Connection;
use std::path::Path;
use std::sync::{Arc, Mutex};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Every migration, in the order they must run.
///
/// Explicit rather than globbed: the order is the whole contract, and a
/// directory listing sorts however the filesystem feels like it.
const MIGRATIONS: &[(&str, &str)] = &[
    ("001-settings", include_str!("migrations/001-settings.sql")),
    ("002-chat", include_str!("migrations/002-chat.sql")),
    ("003-users", include_str!("migrations/003-users.sql")),
    ("004-jobs", include_str!("migrations/004-jobs.sql")),
    ("005-requests", include_str!("migrations/005-requests.sql")),
    ("006-evals", include_str!("migrations/006-evals.sql")),
    ("007-crawl", include_str!("migrations/007-crawl.sql")),
    ("008-images", include_str!("migrations/008-images.sql")),
    ("009-image-ids", include_str!("migrations/009-image-ids.sql")),
    ("010-videos", include_str!("migrations/010-videos.sql")),
];

#[derive(Clone)]
pub struct Db(Arc<Mutex<Connection>>);

impl Db {
    /// Open (creating if need be) the database at `path`, and migrate it.
    pub fn open(path: &Path) -> Res<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .map_err(|e| format!("could not open {}: {e}", path.display()))?;
        Self::prepare(conn)
    }

    /// An empty database in memory. What the tests use.
    #[allow(dead_code)]
    pub fn in_memory() -> Res<Self> {
        Self::prepare(Connection::open_in_memory()?)
    }

    fn prepare(conn: Connection) -> Res<Self> {
        // WAL lets a reader and a writer work at once, which a server with an
        // SSE stream open needs. `foreign_keys` is off by default in SQLite
        // and has to be asked for on every connection.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", true)?;
        // Wait rather than fail when another connection holds the write lock.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        migrate(&conn)?;
        Ok(Db(Arc::new(Mutex::new(conn))))
    }

    /// Do something with the connection.
    ///
    /// Blocking, so callers on an async task go through `spawn_blocking`.
    pub fn with<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Res<T> {
        // A panic while holding the lock poisons it. The connection itself is
        // still fine — SQLite does not care who panicked — so take it back.
        let conn = self.0.lock().unwrap_or_else(|e| e.into_inner());
        Ok(f(&conn)?)
    }

    /// How many migrations this database has had.
    pub fn version(&self) -> Res<u32> {
        self.with(|c| c.pragma_query_value(None, "user_version", |r| r.get(0)))
    }

    /// A runtime setting, or `None` if it was never written.
    ///
    /// The settings table is here from the first migration and read from the
    /// UI in a later phase; the first reader is `storage`, which keeps the
    /// copy a move left behind here.
    pub fn setting(&self, key: &str) -> Res<Option<serde_json::Value>> {
        let raw: Option<String> = self.with(|c| {
            c.query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0))
                .map(Some)
                .or_else(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => Ok(None),
                    other => Err(other),
                })
        })?;
        match raw {
            None => Ok(None),
            Some(text) => Ok(Some(serde_json::from_str(&text)?)),
        }
    }

    #[allow(dead_code)]
    pub fn set_setting(&self, key: &str, value: &serde_json::Value) -> Res<()> {
        let text = value.to_string();
        self.with(|c| {
            c.execute(
                "INSERT INTO settings (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = ?2, updated_at = datetime('now')",
                rusqlite::params![key, text],
            )
        })?;
        Ok(())
    }
}

/// Run whatever migrations this database has not had.
///
/// Each runs in a transaction of its own, together with the `user_version`
/// bump, so an interrupted migration leaves the database at the version it
/// was at rather than halfway into the next one.
fn migrate(conn: &Connection) -> Res<()> {
    let at: u32 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    let total = MIGRATIONS.len() as u32;
    if at > total {
        return Err(format!(
            "this database is at schema version {at} and this build of kvad-serve only knows \
             {total}. It was written by a newer version; upgrade rather than downgrade."
        )
        .into());
    }

    for (i, (name, sql)) in MIGRATIONS.iter().enumerate().skip(at as usize) {
        let version = i as u32 + 1;
        tracing::info!("applying migration {name}");
        // Foreign keys off for the duration, which is what SQLite's own
        // "making other kinds of table schema changes" procedure requires: a
        // table is altered by building a new one, copying the rows and
        // dropping the old, and `DROP TABLE` with enforcement on runs every
        // ON DELETE CASCADE pointing at it first. It has to be set out here
        // rather than in the `.sql` file, because the pragma is a no-op
        // inside a transaction and every migration runs inside one.
        conn.pragma_update(None, "foreign_keys", false)?;
        let applied = conn
            .execute_batch(&format!("BEGIN; {sql}; PRAGMA user_version = {version}; COMMIT;"))
            .map_err(|e| format!("migration {name} failed: {e}"));
        conn.pragma_update(None, "foreign_keys", true)?;
        applied?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Migrations 006 and 007 rebuild the jobs table to widen its `kind`
    /// check, and a rebuild is a `DROP TABLE` — which, with foreign keys
    /// enforced, would take every training metric and sample on the machine
    /// with it. This is the upgrade an existing installation makes, run
    /// against rows.
    #[test]
    fn widening_the_kinds_of_job_keeps_the_jobs_and_their_charts() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();

        // A tripwire rather than a version number: this test is about
        // rebuilding `jobs`, and it should fail when a migration starts doing
        // that and nobody has looked here.
        let rebuilds: Vec<&str> = MIGRATIONS
            .iter()
            .filter(|(_, sql)| sql.contains("CREATE TABLE jobs_new"))
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(rebuilds, ["006-evals", "007-crawl"], "a migration rebuilds `jobs` unwatched");

        // Everything up to the first of them; `migrate` then runs the rest.
        let before: Vec<_> = MIGRATIONS.iter().take_while(|(n, _)| *n != rebuilds[0]).collect();
        for (i, (_, sql)) in before.iter().enumerate() {
            conn.execute_batch(&format!(
                "BEGIN; {sql}; PRAGMA user_version = {}; COMMIT;",
                i + 1
            ))
            .unwrap();
        }

        conn.execute(
            "INSERT INTO jobs (id, kind, state, label, params) VALUES (7, 'train', 'done', 'x', '{}')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO train_metrics (job, step, train_loss, val_loss, chars_per_sec, elapsed_secs)
             VALUES (7, 100, 2.0, 1.9, 500.0, 3.0)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO train_samples (job, step, text) VALUES (7, 100, 'hi')", [])
            .unwrap();

        migrate(&conn).unwrap();

        let (kind, label): (String, String) = conn
            .query_row("SELECT kind, label FROM jobs WHERE id = 7", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!((kind.as_str(), label.as_str()), ("train", "x"));
        let metrics: i64 =
            conn.query_row("SELECT count(*) FROM train_metrics WHERE job = 7", [], |r| r.get(0))
                .unwrap();
        let samples: i64 =
            conn.query_row("SELECT count(*) FROM train_samples WHERE job = 7", [], |r| r.get(0))
                .unwrap();
        assert_eq!((metrics, samples), (1, 1), "the rebuild cascaded through the chart");

        // The point of the rebuilds: kinds the old checks refused.
        for kind in ["bench", "crawl"] {
            conn.execute(
                "INSERT INTO jobs (kind, state, label, params) VALUES (?1, 'running', 'b', '{}')",
                [kind],
            )
            .unwrap();
        }
        // And the check is still a check.
        assert!(conn
            .execute(
                "INSERT INTO jobs (kind, state, label, params) VALUES ('nonsense', 'running', 'n', '{}')",
                [],
            )
            .is_err());

        // Deleting a job must still take its chart, which needs the foreign
        // keys the migration turned off to have come back on.
        conn.execute("DELETE FROM jobs WHERE id = 7", []).unwrap();
        let left: i64 =
            conn.query_row("SELECT count(*) FROM train_metrics WHERE job = 7", [], |r| r.get(0))
                .unwrap();
        assert_eq!(left, 0, "foreign keys did not come back on after the migration");
    }

    /// Migration 009 rebuilds `images` for AUTOINCREMENT. The pictures a
    /// machine already has keep their ids, since their files are named by
    /// them, and the newest id is not handed out again after the upgrade.
    #[test]
    fn remembering_image_ids_keeps_the_images() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        let before: Vec<_> = MIGRATIONS.iter().take_while(|(n, _)| *n != "009-image-ids").collect();
        for (i, (_, sql)) in before.iter().enumerate() {
            conn.execute_batch(&format!("BEGIN; {sql}; PRAGMA user_version = {}; COMMIT;", i + 1)).unwrap();
        }
        conn.execute(
            "INSERT INTO images (id, model, backend, prompt, width, height, steps, guidance, seed, bytes, secs)
             VALUES (3, 'm', 'b', 'a lighthouse', 64, 64, 4, 0.0, -1, 10, 1.5),
                    (5, 'm', 'b', 'a harbour', 64, 64, 4, 0.0, 7, 10, 1.5)",
            [],
        )
        .unwrap();

        migrate(&conn).unwrap();

        let kept: Vec<(i64, String, i64)> = conn
            .prepare("SELECT id, prompt, seed FROM images ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(kept, [(3, "a lighthouse".to_string(), -1), (5, "a harbour".to_string(), 7)]);

        conn.execute("DELETE FROM images WHERE id = 5", []).unwrap();
        conn.execute(
            "INSERT INTO images (model, backend, prompt, width, height, steps, guidance, seed, bytes, secs)
             VALUES ('m', 'b', 'a storm', 64, 64, 4, 0.0, 1, 10, 1.5)",
            [],
        )
        .unwrap();
        let next: i64 = conn.query_row("SELECT max(id) FROM images", [], |r| r.get(0)).unwrap();
        assert_eq!(next, 6, "the deleted newest id was handed out again");
    }

    #[test]
    fn a_fresh_database_runs_every_migration_and_stops_there() {
        let db = Db::in_memory().unwrap();
        assert_eq!(db.version().unwrap(), MIGRATIONS.len() as u32);

        // Migrating an already-migrated database does nothing, rather than
        // failing on a table that is already there.
        db.with(|c| {
            migrate(c).unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(db.version().unwrap(), MIGRATIONS.len() as u32);
    }

    #[test]
    fn settings_round_trip_and_replace_rather_than_accumulate() {
        let db = Db::in_memory().unwrap();
        assert!(db.setting("theme").unwrap().is_none());

        db.set_setting("theme", &serde_json::json!("dim")).unwrap();
        assert_eq!(db.setting("theme").unwrap().unwrap(), serde_json::json!("dim"));

        db.set_setting("theme", &serde_json::json!("light")).unwrap();
        assert_eq!(db.setting("theme").unwrap().unwrap(), serde_json::json!("light"));
        let rows: i64 = db.with(|c| c.query_row("SELECT count(*) FROM settings", [], |r| r.get(0))).unwrap();
        assert_eq!(rows, 1);
    }

    /// A database from a newer build is not something to guess at.
    #[test]
    fn a_database_from_the_future_is_refused() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "user_version", 99).unwrap();
        let err = migrate(&conn).unwrap_err().to_string();
        assert!(err.contains("newer version"), "{err}");
    }
}
