//! Accounts, sessions and API keys: the rows behind every sign-in.
//!
//! Reading and writing, no HTTP. Which of these a request may use is
//! [`crate::auth`]'s business; what they mean is here.
//!
//! # The rules that are easy to get wrong
//!
//! * **A session and an API key are looked up by hash**, never by the token,
//!   so the query is the comparison and the database never holds anything
//!   that can be signed in with.
//! * **An expired session is not a session.** The expiry is checked in SQL,
//!   in the same statement as the lookup, so there is no window between
//!   finding a row and deciding to trust it.
//! * **The last admin cannot be removed or demoted.** A server with no
//!   administrator cannot be administered, and the way out of that is a
//!   database editor.

use crate::db::Db;
use crate::secret;
use rusqlite::{params, OptionalExtension, Row};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// How long a sign-in lasts. Long enough not to be a nuisance on a machine
/// somebody uses daily, short enough that a forgotten browser is not forever.
pub const SESSION_DAYS: i64 = 30;

#[derive(Debug, Clone, serde::Serialize)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub role: String,
    pub email: Option<String>,
    pub created_at: String,
    pub last_seen_at: Option<String>,
    /// Whether this account can sign in with a password at all.
    pub has_password: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Session {
    /// The hash, which is also the handle used to revoke it. Not the token:
    /// the token was shown to a browser once and never written down.
    pub token_hash: String,
    pub user: i64,
    pub created_at: String,
    pub expires_at: String,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ApiKey {
    pub id: i64,
    pub name: String,
    pub prefix: String,
    pub user: i64,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

fn user_from(row: &Row) -> rusqlite::Result<User> {
    let hash: Option<String> = row.get("password_hash")?;
    Ok(User {
        id: row.get("id")?,
        name: row.get("name")?,
        role: row.get("role")?,
        email: row.get("email")?,
        created_at: row.get("created_at")?,
        last_seen_at: row.get("last_seen_at")?,
        has_password: hash.is_some_and(|h| !h.is_empty()),
    })
}

const COLUMNS: &str = "id, name, password_hash, role, email, created_at, last_seen_at";

pub fn count(db: &Db) -> Res<i64> {
    db.with(|c| c.query_row("SELECT count(*) FROM users", [], |r| r.get(0)))
}

pub fn list(db: &Db) -> Res<Vec<User>> {
    db.with(|c| {
        let mut q = c.prepare(&format!("SELECT {COLUMNS} FROM users ORDER BY id"))?;
        let rows = q.query_map([], user_from)?.collect();
        rows
    })
}

pub fn get(db: &Db, id: i64) -> Res<Option<User>> {
    db.with(|c| {
        c.query_row(&format!("SELECT {COLUMNS} FROM users WHERE id = ?1"), [id], user_from)
            .optional()
    })
}

pub fn by_name(db: &Db, name: &str) -> Res<Option<User>> {
    db.with(|c| {
        c.query_row(&format!("SELECT {COLUMNS} FROM users WHERE name = ?1"), [name], user_from)
            .optional()
    })
}

/// Used to match an identity provider's `email` claim to an account.
pub fn by_email(db: &Db, email: &str) -> Res<Option<User>> {
    db.with(|c| {
        c.query_row(&format!("SELECT {COLUMNS} FROM users WHERE email = ?1"), [email], user_from)
            .optional()
    })
}

/// What a name has to be to be a name.
///
/// Deliberately narrow. A name is typed at a sign-in prompt and shown beside
/// other people's, so anything that could be mistaken for another name — a
/// leading space, a zero-width character, a slash — is refused rather than
/// normalised into something the person did not type.
pub fn check_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '@'));
    match ok {
        true => Ok(()),
        false => Err(
            "a name is 1 to 64 characters of letters, digits, and `-`, `_`, `.` or `@`".into(),
        ),
    }
}

pub fn check_role(role: &str) -> Result<(), String> {
    match role {
        "admin" | "user" => Ok(()),
        other => Err(format!("`{other}` is not a role; there are `admin` and `user`")),
    }
}

/// Make an account.
pub fn create(
    db: &Db,
    name: &str,
    password: Option<&str>,
    role: &str,
    email: Option<&str>,
) -> Res<User> {
    check_name(name)?;
    check_role(role)?;
    let hash = match password {
        Some(p) => {
            secret::check_password(p)?;
            Some(secret::hash_password(p)?)
        }
        None => None,
    };
    if by_name(db, name)?.is_some() {
        return Err(format!("there is already an account called `{name}`").into());
    }
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO users (name, password_hash, role, email) VALUES (?1, ?2, ?3, ?4)",
            params![name, hash, role, email],
        )?;
        Ok(c.last_insert_rowid())
    })?;
    get(db, id)?.ok_or_else(|| "the account vanished as it was created".into())
}

/// The first account on a server that has none, created against the one-time
/// setup token.
///
/// It adopts every conversation that has no owner, because those are from
/// before there was anybody to own them — this server running with no
/// authentication at all — and the person setting it up is the person who had
/// those conversations. Leaving them ownerless would show them to whoever
/// signed up second.
pub fn bootstrap(db: &Db, name: &str, password: &str) -> Res<User> {
    if count(db)? > 0 {
        return Err("this server already has an account; sign in instead".into());
    }
    let user = create(db, name, Some(password), "admin", None)?;
    db.with(|c| c.execute("UPDATE conversations SET owner = ?1 WHERE owner IS NULL", [user.id]))?;
    Ok(user)
}

/// Change a password, or set one on an account that had none.
pub fn set_password(db: &Db, id: i64, password: &str) -> Res<()> {
    secret::check_password(password)?;
    let hash = secret::hash_password(password)?;
    let changed =
        db.with(|c| c.execute("UPDATE users SET password_hash = ?2 WHERE id = ?1", params![id, hash]))?;
    match changed {
        0 => Err(format!("there is no account {id}").into()),
        // Every session was signed in with the old password. Changing it
        // because the old one leaked and leaving those alive would be
        // changing nothing.
        _ => {
            db.with(|c| c.execute("DELETE FROM sessions WHERE user = ?1", [id]))?;
            Ok(())
        }
    }
}

/// How many administrators there are. The number that must not reach zero.
pub fn admin_count(db: &Db) -> Res<i64> {
    db.with(|c| c.query_row("SELECT count(*) FROM users WHERE role = 'admin'", [], |r| r.get(0)))
}

fn is_last_admin(db: &Db, id: i64) -> Res<bool> {
    let user = get(db, id)?;
    Ok(user.is_some_and(|u| u.role == "admin") && admin_count(db)? <= 1)
}

pub fn set_role(db: &Db, id: i64, role: &str) -> Res<User> {
    check_role(role)?;
    if role != "admin" && is_last_admin(db, id)? {
        return Err("this is the only administrator; make somebody else one first".into());
    }
    db.with(|c| c.execute("UPDATE users SET role = ?2 WHERE id = ?1", params![id, role]))?;
    get(db, id)?.ok_or_else(|| format!("there is no account {id}").into())
}

pub fn delete(db: &Db, id: i64) -> Res<bool> {
    if is_last_admin(db, id)? {
        return Err("this is the only administrator; a server with none cannot be run".into());
    }
    // Sessions, keys and conversations go with them, by ON DELETE CASCADE.
    Ok(db.with(|c| c.execute("DELETE FROM users WHERE id = ?1", [id]))? > 0)
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// Sign somebody in: check the password, and hand back a token.
///
/// The token is returned and not stored. What is stored is its hash.
pub fn sign_in(db: &Db, name: &str, password: &str, agent: Option<&str>) -> Res<(User, String)> {
    let stored: Option<(i64, String)> = db.with(|c| {
        c.query_row(
            "SELECT id, coalesce(password_hash, '') FROM users WHERE name = ?1",
            [name],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
    })?;

    // One message for "no such account" and for "wrong password", because
    // telling them apart turns the sign-in form into a way to ask which
    // accounts exist. The hash is still verified against a made-up value when
    // the account does not exist, so the two do not take visibly different
    // amounts of time either.
    const REFUSED: &str = "that name and password do not go together";
    let Some((id, hash)) = stored else {
        let _ = secret::verify_password(password, DUMMY_HASH);
        return Err(REFUSED.into());
    };
    if !secret::verify_password(password, &hash) {
        return Err(REFUSED.into());
    }

    let token = open_session(db, id, agent)?;
    let user = get(db, id)?.ok_or("the account vanished as it signed in")?;
    Ok((user, token))
}

/// Start a session for an account whose identity has already been
/// established, and return the token.
///
/// Split out of [`sign_in`] because an identity provider establishes it
/// somewhere else entirely — see [`crate::oidc`] — and there is no password
/// here to check. Anything calling this is asserting that it has checked
/// something at least as good.
pub fn open_session(db: &Db, id: i64, agent: Option<&str>) -> Res<String> {
    let token = secret::token();
    let fingerprint = secret::fingerprint(&token);
    db.with(|c| {
        c.execute(
            "INSERT INTO sessions (token_hash, user, expires_at, user_agent)
             VALUES (?1, ?2, datetime('now', ?3), ?4)",
            params![fingerprint, id, format!("+{SESSION_DAYS} days"), agent],
        )?;
        c.execute("UPDATE users SET last_seen_at = datetime('now') WHERE id = ?1", [id])
    })?;
    Ok(token)
}

/// An argon2 hash of nothing in particular, verified against when an account
/// does not exist so that a missing account costs the same as a wrong
/// password. Generated once at the parameters `hash_password` uses.
const DUMMY_HASH: &str = "$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0c2E$\
                          Kq6VLBAoJ1EDvJpZTfJPNBSCNCHhBoQEuDXHJiXBNyE";

/// The account behind a session token, if the session is live.
///
/// The expiry is part of the lookup rather than a check afterwards, so an
/// expired session is simply not found.
pub fn from_session(db: &Db, token: &str) -> Res<Option<User>> {
    let fingerprint = secret::fingerprint(token);
    let id: Option<i64> = db.with(|c| {
        c.query_row(
            "SELECT user FROM sessions WHERE token_hash = ?1 AND expires_at > datetime('now')",
            [&fingerprint],
            |r| r.get(0),
        )
        .optional()
    })?;
    match id {
        None => Ok(None),
        Some(id) => get(db, id),
    }
}

pub fn sign_out(db: &Db, token: &str) -> Res<()> {
    let fingerprint = secret::fingerprint(token);
    db.with(|c| c.execute("DELETE FROM sessions WHERE token_hash = ?1", [&fingerprint]))?;
    Ok(())
}

pub fn sessions(db: &Db, user: i64) -> Res<Vec<Session>> {
    db.with(|c| {
        let mut q = c.prepare(
            "SELECT token_hash, user, created_at, expires_at, user_agent
             FROM sessions WHERE user = ?1 AND expires_at > datetime('now')
             ORDER BY created_at DESC",
        )?;
        let rows = q
            .query_map([user], |r| {
                Ok(Session {
                    token_hash: r.get("token_hash")?,
                    user: r.get("user")?,
                    created_at: r.get("created_at")?,
                    expires_at: r.get("expires_at")?,
                    user_agent: r.get("user_agent")?,
                })
            })?
            .collect();
        rows
    })
}

/// Revoke one session of `user`'s. Scoped to the user so that knowing a hash
/// is not enough to end somebody else's session.
pub fn revoke_session(db: &Db, user: i64, token_hash: &str) -> Res<bool> {
    Ok(db.with(|c| {
        c.execute(
            "DELETE FROM sessions WHERE user = ?1 AND token_hash = ?2",
            params![user, token_hash],
        )
    })? > 0)
}

/// Throw away sessions that have expired. Nothing depends on this — an
/// expired session is already refused — it just keeps the table from growing
/// forever.
pub fn sweep(db: &Db) -> Res<usize> {
    db.with(|c| c.execute("DELETE FROM sessions WHERE expires_at <= datetime('now')", []))
}

// ---------------------------------------------------------------------------
// API keys
// ---------------------------------------------------------------------------

/// Make a key. The token comes back once and is never recoverable.
pub fn create_key(db: &Db, user: i64, name: &str) -> Res<(ApiKey, String)> {
    let name = name.trim();
    if name.is_empty() {
        return Err("a key needs a name, so it can be told from the others".into());
    }
    let token = secret::token();
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO api_keys (name, token_hash, prefix, user) VALUES (?1, ?2, ?3, ?4)",
            params![name, secret::fingerprint(&token), secret::prefix(&token), user],
        )?;
        Ok(c.last_insert_rowid())
    })?;
    let key = keys(db, user)?
        .into_iter()
        .find(|k| k.id == id)
        .ok_or("the key vanished as it was made")?;
    Ok((key, token))
}

pub fn keys(db: &Db, user: i64) -> Res<Vec<ApiKey>> {
    db.with(|c| {
        let mut q = c.prepare(
            "SELECT id, name, prefix, user, created_at, last_used_at
             FROM api_keys WHERE user = ?1 ORDER BY id",
        )?;
        let rows = q
            .query_map([user], |r| {
                Ok(ApiKey {
                    id: r.get("id")?,
                    name: r.get("name")?,
                    prefix: r.get("prefix")?,
                    user: r.get("user")?,
                    created_at: r.get("created_at")?,
                    last_used_at: r.get("last_used_at")?,
                })
            })?
            .collect();
        rows
    })
}

pub fn revoke_key(db: &Db, user: i64, id: i64) -> Res<bool> {
    Ok(db.with(|c| c.execute("DELETE FROM api_keys WHERE user = ?1 AND id = ?2", params![user, id]))?
        > 0)
}

/// The account behind a bearer token, if there is one.
///
/// Also stamps `last_used_at`, which is how a settings page can show which
/// keys are still in use and which can be revoked without breaking anything.
pub fn from_key(db: &Db, token: &str) -> Res<Option<User>> {
    let fingerprint = secret::fingerprint(token);
    let id: Option<i64> = db.with(|c| {
        c.query_row("SELECT user FROM api_keys WHERE token_hash = ?1", [&fingerprint], |r| r.get(0))
            .optional()
    })?;
    let Some(id) = id else { return Ok(None) };
    db.with(|c| {
        c.execute("UPDATE api_keys SET last_used_at = datetime('now') WHERE token_hash = ?1", [&fingerprint])
    })?;
    get(db, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded() -> Db {
        let db = Db::in_memory().unwrap();
        create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
        db
    }

    #[test]
    fn an_account_stores_a_hash_and_never_the_password() {
        let db = seeded();
        let stored: String = db
            .with(|c| c.query_row("SELECT password_hash FROM users WHERE name = 'ada'", [], |r| r.get(0)))
            .unwrap();
        assert!(stored.starts_with("$argon2id$"));
        assert!(!stored.contains("lovelace"));
        assert!(by_name(&db, "ada").unwrap().unwrap().has_password);

        // The name is matched without regard to case, in the lookup and in
        // the uniqueness check, so `Ada` is `ada` and cannot be made twice.
        assert!(by_name(&db, "ADA").unwrap().is_some());
        let err = create(&db, "Ada", Some("something-else"), "user", None).unwrap_err().to_string();
        assert!(err.contains("already"), "{err}");
    }

    #[test]
    fn signing_in_needs_the_right_password_and_says_nothing_else() {
        let db = seeded();
        let (who, token) = sign_in(&db, "ada", "lovelace-1843", Some("curl")).unwrap();
        assert_eq!(who.name, "ada");
        assert_eq!(token.len(), 64);

        // The session is found by the token, and the table holds only a hash.
        assert_eq!(from_session(&db, &token).unwrap().unwrap().id, who.id);
        let rows: i64 = db
            .with(|c| c.query_row("SELECT count(*) FROM sessions WHERE token_hash = ?1", [&token], |r| r.get(0)))
            .unwrap();
        assert_eq!(rows, 0, "the token itself was stored");

        // A wrong password and a missing account are refused identically, so
        // the form cannot be used to ask which accounts exist.
        let wrong = sign_in(&db, "ada", "nope-nope-nope", None).unwrap_err().to_string();
        let absent = sign_in(&db, "nobody", "nope-nope-nope", None).unwrap_err().to_string();
        assert_eq!(wrong, absent);

        // And a token that was never issued is nobody.
        assert!(from_session(&db, &secret::token()).unwrap().is_none());
    }

    #[test]
    fn signing_out_ends_that_session_and_only_that_one() {
        let db = seeded();
        let (_, first) = sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        let (who, second) = sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        assert_eq!(sessions(&db, who.id).unwrap().len(), 2);

        sign_out(&db, &first).unwrap();
        assert!(from_session(&db, &first).unwrap().is_none());
        assert!(from_session(&db, &second).unwrap().is_some());
    }

    /// An expired session is refused, and it is refused by the lookup rather
    /// than by a check somebody could forget to write.
    #[test]
    fn an_expired_session_is_not_a_session() {
        let db = seeded();
        let (who, token) = sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        db.with(|c| {
            c.execute(
                "UPDATE sessions SET expires_at = datetime('now', '-1 second') WHERE user = ?1",
                [who.id],
            )
        })
        .unwrap();

        assert!(from_session(&db, &token).unwrap().is_none());
        assert!(sessions(&db, who.id).unwrap().is_empty(), "an expired session was listed as live");
        assert_eq!(sweep(&db).unwrap(), 1);
    }

    /// Changing a password ends every session, because the reason to change
    /// one is usually that the old one got out.
    #[test]
    fn a_new_password_ends_the_old_sessions() {
        let db = seeded();
        let (who, token) = sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        set_password(&db, who.id, "a-better-one-1843").unwrap();

        assert!(from_session(&db, &token).unwrap().is_none());
        assert!(sign_in(&db, "ada", "lovelace-1843", None).is_err());
        assert!(sign_in(&db, "ada", "a-better-one-1843", None).is_ok());
    }

    #[test]
    fn a_key_is_shown_once_and_recognised_by_its_hash() {
        let db = seeded();
        let ada = by_name(&db, "ada").unwrap().unwrap();
        let (key, token) = create_key(&db, ada.id, "laptop").unwrap();
        assert_eq!(key.prefix, token[..8]);
        assert!(key.last_used_at.is_none());

        assert_eq!(from_key(&db, &token).unwrap().unwrap().id, ada.id);
        // Using it is recorded, so a key nobody uses can be told apart from
        // one somebody does before it is revoked.
        assert!(keys(&db, ada.id).unwrap()[0].last_used_at.is_some());

        assert!(from_key(&db, &secret::token()).unwrap().is_none());
        assert!(revoke_key(&db, ada.id, key.id).unwrap());
        assert!(from_key(&db, &token).unwrap().is_none());
    }

    /// One person's key or session must not be another person's to revoke.
    #[test]
    fn keys_and_sessions_belong_to_the_person_who_made_them() {
        let db = seeded();
        let ada = by_name(&db, "ada").unwrap().unwrap();
        let bob = create(&db, "bob", Some("builder-2024"), "user", None).unwrap();

        let (key, _) = create_key(&db, ada.id, "ada's").unwrap();
        assert!(!revoke_key(&db, bob.id, key.id).unwrap(), "bob revoked ada's key");
        assert!(keys(&db, bob.id).unwrap().is_empty());

        let (_, token) = sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        let hash = secret::fingerprint(&token);
        assert!(!revoke_session(&db, bob.id, &hash).unwrap(), "bob ended ada's session");
        assert!(revoke_session(&db, ada.id, &hash).unwrap());
    }

    /// A server with no administrator cannot be administered.
    #[test]
    fn the_last_administrator_cannot_be_removed_or_demoted() {
        let db = seeded();
        let ada = by_name(&db, "ada").unwrap().unwrap();
        assert!(set_role(&db, ada.id, "user").is_err());
        assert!(delete(&db, ada.id).is_err());

        // With a second administrator, either is fine.
        let bob = create(&db, "bob", Some("builder-2024"), "admin", None).unwrap();
        assert_eq!(set_role(&db, ada.id, "user").unwrap().role, "user");
        assert!(delete(&db, bob.id).is_err(), "bob is now the only admin");

        // An ordinary user is deleted without argument, and takes their
        // sessions with them.
        sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        assert!(delete(&db, ada.id).unwrap());
        assert!(sessions(&db, ada.id).unwrap().is_empty());
    }

    /// The first account adopts what was there before there were accounts.
    #[test]
    fn bootstrapping_takes_over_the_conversations_from_before() {
        let db = Db::in_memory().unwrap();
        crate::chat::create(&db, None, "From before", None, None).unwrap();
        assert_eq!(count(&db).unwrap(), 0);

        let ada = bootstrap(&db, "ada", "lovelace-1843").unwrap();
        assert_eq!(ada.role, "admin");
        let owner: Option<i64> = db
            .with(|c| c.query_row("SELECT owner FROM conversations", [], |r| r.get(0)))
            .unwrap();
        assert_eq!(owner, Some(ada.id));

        // And it happens once.
        assert!(bootstrap(&db, "bob", "builder-2024").is_err());
    }

    #[test]
    fn names_and_roles_are_checked_before_anything_is_written() {
        for bad in ["", " ada", "ada bell", "ada/bell", "a".repeat(65).as_str()] {
            assert!(check_name(bad).is_err(), "`{bad}` was accepted");
        }
        for good in ["ada", "ada.bell", "ada_bell-2", "ada@example.com"] {
            assert!(check_name(good).is_ok(), "`{good}` was refused");
        }
        assert!(check_role("wizard").is_err());

        let db = Db::in_memory().unwrap();
        assert!(create(&db, "ada bell", Some("longenough"), "admin", None).is_err());
        assert!(create(&db, "ada", Some("short"), "admin", None).is_err());
        assert_eq!(count(&db).unwrap(), 0, "a refused account was written anyway");
    }
}
