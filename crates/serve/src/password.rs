//! `kvad-serve password NAME`: a new password for an account, set by
//! whoever holds the database.
//!
//! The way back in after a password is lost. There is no other: changing a
//! password over HTTP needs the current one, and an administrator setting
//! somebody else's needs an administrator who can still sign in, which on a
//! server with one account is the person who cannot.
//!
//! It adds no power. The database is the server — its accounts, its keys'
//! hashes, every conversation — and somebody who can open the file can
//! already rewrite any row in it with `sqlite3`. This is that, done without
//! getting the hash wrong, and with the account's sessions ended the way
//! [`crate::accounts::change_password`] ends them.

use crate::config::{self, Mode};
use crate::db::Db;
use crate::{secret, users};
use std::path::PathBuf;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What [`reset`] did, to be said.
#[derive(Debug)]
pub struct Reset {
    pub user: users::User,
    /// Sessions that ended with the old password.
    pub sessions: usize,
    /// API keys, which did not.
    pub keys: usize,
}

/// Open the database a server with these arguments would open, ask for a
/// password twice and set it.
pub fn run(name: &str, config_path: Option<PathBuf>, db_path: Option<PathBuf>) -> Res<()> {
    let config_path = config_path.unwrap_or_else(config::default_path);
    let mut cfg = config::Config::load(&config_path)?;
    // As `run` does, so that `kvad.db` is found in the same data directory.
    kvad::weights::set_data_dir(cfg.chosen_data_dir(&config_path));
    if let Some(path) = db_path {
        cfg.database.path = Some(path);
    }
    let path = cfg.database_path();
    // Not `Db::open` on a path that is not there: that would make an empty
    // database, and then say there is no such account, which is true and
    // no help. The likelier mistake is the wrong --config or --db.
    if !path.exists() {
        return Err(format!(
            "there is no database at {}; --config or --db says where the server keeps it",
            path.display()
        )
        .into());
    }
    let db = Db::open(&path)?;
    // Before the questions, so that a wrong name is not told so only after
    // its password was typed twice.
    let user = known(&db, name)?;

    eprintln!("A new password for {} ({}), in {}.", user.name, user.role, path.display());
    let first = ask("new password")?;
    secret::check_password(&first)?;
    if ask("the same again")? != first {
        return Err("those were not the same; nothing was changed".into());
    }
    let done = reset(&db, &user.name, &first)?;

    println!("{}'s password is changed.", done.user.name);
    match done.sessions {
        0 => println!("It had no sessions to end."),
        n => println!("Ended its {}: a browser signed in with the old password signs in again.", count(n, "session", "sessions")),
    }
    match done.keys {
        0 => println!("It has no API keys."),
        n => println!("Its {} still {}.", count(n, "API key", "API keys"), if n == 1 { "works" } else { "work" }),
    }
    // A password nothing asks for is worth saying, rather than leaving
    // somebody to find out at a sign-in page that is not there.
    match cfg.auth.mode {
        Mode::Local | Mode::Basic => {}
        Mode::None => println!("This server's auth.mode is \"none\": nothing asks for a password until it has a mode with accounts."),
        Mode::Oidc => println!("This server's auth.mode is \"oidc\": it signs in through the provider, not with this password."),
    }
    Ok(())
}

/// Set `name`'s password: every session it had ends, its keys stay.
pub fn reset(db: &Db, name: &str, password: &str) -> Res<Reset> {
    let user = known(db, name)?;
    let sessions = users::sessions(db, user.id)?.len();
    let keys = users::keys(db, user.id)?.len();
    users::set_password(db, user.id, password)?;
    Ok(Reset { user, sessions, keys })
}

/// The account called `name`, or a refusal that names the ones there are.
fn known(db: &Db, name: &str) -> Res<users::User> {
    if let Some(user) = users::by_name(db, name)? {
        return Ok(user);
    }
    let names: Vec<String> = users::list(db)?.into_iter().map(|u| u.name).collect();
    Err(match names.is_empty() {
        true => "this server has no accounts yet; the first is made with the setup token it prints".into(),
        false => format!("there is no account called `{name}`; there are {}", names.join(", ")).into(),
    })
}

fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// A password, asked on the terminal with echo off; from stdin, one line,
/// when there is no terminal.
///
/// The same as `kvad`'s `out::secret`, which is in the CLI's binary and so
/// out of this crate's reach.
fn ask(question: &str) -> Res<String> {
    #[cfg(unix)]
    if let Some(answer) = tty_secret(question)? {
        return Ok(answer);
    }
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(answer.trim_end_matches(['\n', '\r']).to_string())
}

#[cfg(unix)]
fn tty_secret(question: &str) -> Res<Option<String>> {
    use std::io::{BufRead, Write};
    use std::os::unix::io::AsRawFd;

    let Ok(tty) = std::fs::OpenOptions::new().read(true).write(true).open("/dev/tty") else {
        return Ok(None);
    };
    let fd = tty.as_raw_fd();
    // SAFETY: a zeroed termios is a valid value to hand tcgetattr to fill.
    let mut term: libc::termios = unsafe { std::mem::zeroed() };
    // SAFETY: fd is an open terminal for as long as `tty` lives.
    if unsafe { libc::tcgetattr(fd, &mut term) } != 0 {
        return Ok(None);
    }
    let before = term;
    term.c_lflag &= !libc::ECHO;
    term.c_lflag |= libc::ECHONL;
    // SAFETY: as above; `term` is the terminal's own settings, less echo.
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &term) };

    let mut writer = &tty;
    let _ = write!(writer, "{question}: ");
    let _ = writer.flush();
    let mut answer = String::new();
    let read = std::io::BufReader::new(&tty).read_line(&mut answer);
    // Echo back on before anything can go wrong, including the read.
    // SAFETY: restoring what tcgetattr returned.
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &before) };
    read?;
    Ok(Some(answer.trim_end_matches(['\n', '\r']).to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded() -> Db {
        let db = Db::in_memory().unwrap();
        users::create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
        users::create(&db, "grace", Some("hopper-1906"), "user", None).unwrap();
        db
    }

    #[test]
    fn the_password_is_changed_and_the_old_one_stops_working() {
        let db = seeded();
        let done = reset(&db, "ada", "analytical-engine").unwrap();
        assert_eq!(done.user.name, "ada");
        assert!(users::sign_in(&db, "ada", "analytical-engine", None).is_ok());
        assert!(users::sign_in(&db, "ada", "lovelace-1843", None).is_err());
        // Somebody else's is not touched.
        assert!(users::sign_in(&db, "grace", "hopper-1906", None).is_ok());
    }

    #[test]
    fn its_sessions_end_and_its_keys_survive() {
        let db = seeded();
        let (ada, one) = users::sign_in(&db, "ada", "lovelace-1843", Some("browser")).unwrap();
        let (_, two) = users::sign_in(&db, "ada", "lovelace-1843", Some("phone")).unwrap();
        let (_, hers) = users::sign_in(&db, "grace", "hopper-1906", None).unwrap();
        let (_, key) = users::create_key(&db, ada.id, "kvad CLI").unwrap();

        let done = reset(&db, "ada", "analytical-engine").unwrap();
        assert_eq!((done.sessions, done.keys), (2, 1));
        assert!(users::from_session(&db, &one).unwrap().is_none());
        assert!(users::from_session(&db, &two).unwrap().is_none());
        assert_eq!(users::from_key(&db, &key).unwrap().unwrap().id, ada.id);
        assert!(users::from_session(&db, &hers).unwrap().is_some(), "another account's session ended");
    }

    #[test]
    fn an_unknown_name_is_refused_and_changes_nothing() {
        let db = seeded();
        let err = reset(&db, "babbage", "difference-engine").unwrap_err().to_string();
        assert!(err.contains("no account called `babbage`") && err.contains("ada, grace"), "{err}");
        assert!(users::sign_in(&db, "ada", "lovelace-1843", None).is_ok());

        let empty = Db::in_memory().unwrap();
        let err = reset(&empty, "ada", "analytical-engine").unwrap_err().to_string();
        assert!(err.contains("no accounts yet"), "{err}");
    }

    #[test]
    fn a_short_password_is_refused_and_the_sessions_stay() {
        let db = seeded();
        let (_, token) = users::sign_in(&db, "ada", "lovelace-1843", None).unwrap();
        assert!(reset(&db, "ada", "short").is_err());
        assert!(users::from_session(&db, &token).unwrap().is_some());
        assert!(users::sign_in(&db, "ada", "lovelace-1843", None).is_ok());
    }
}
