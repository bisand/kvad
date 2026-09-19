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
const MIGRATIONS: &[(&str, &str)] =
    &[("001-settings", include_str!("migrations/001-settings.sql"))];

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
    /// UI in a later phase; until then this and its writer are exercised only
    /// by the tests below.
    #[allow(dead_code)]
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
        conn.execute_batch(&format!("BEGIN; {sql}; PRAGMA user_version = {version}; COMMIT;"))
            .map_err(|e| format!("migration {name} failed: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
