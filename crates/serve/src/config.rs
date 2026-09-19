//! Bootstrap configuration: the handful of things that have to be known
//! before anything else can start.
//!
//! The split is deliberate and worth stating once. What is *here* is what the
//! server needs in order to open a socket and a database — where to bind,
//! where the database lives, how requests prove who they are. What lives in
//! the database instead is everything that can be changed while the server
//! runs, by somebody who is already logged in. Putting the auth mode in the
//! database would mean asking the database how to decide who may read the
//! database.
//!
//! The file is optional. With no file at all the server binds loopback with
//! no auth, which is the right default for `kvad-serve` on a laptop and is
//! refused outright the moment the bind address is not loopback — see
//! [`Config::check`].

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub server: Server,
    pub database: Database,
    pub auth: Auth,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Server {
    pub bind: SocketAddr,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Database {
    pub path: PathBuf,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Auth {
    pub mode: Mode,
}

/// How a request proves who it is.
///
/// Only [`Mode::None`] is implemented. The rest are named here rather than
/// added later because the name in the file is the thing people write down,
/// and a config that silently accepted `mode = "local"` and then let everyone
/// in would be worse than one that refuses to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Anyone who can reach the socket is an administrator. Loopback only.
    None,
    /// Username and password, against the server's own user table.
    Local,
    /// The same user table over HTTP Basic.
    Basic,
    /// An external identity provider, authorization code + PKCE.
    Oidc,
}

impl Default for Server {
    fn default() -> Self {
        Server { bind: SocketAddr::from(([127, 0, 0, 1], 8080)) }
    }
}

impl Default for Database {
    fn default() -> Self {
        Database { path: kvad::weights::data_dir().join("kvad.db") }
    }
}

impl Default for Auth {
    fn default() -> Self {
        Auth { mode: Mode::None }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::None => "none",
            Mode::Local => "local",
            Mode::Basic => "basic",
            Mode::Oidc => "oidc",
        })
    }
}

/// Where the config file lives when nobody says otherwise:
/// `$XDG_CONFIG_HOME/kvad/kvad.toml`, beside the active-model state the CLI
/// and the TUI already keep there.
pub fn default_path() -> PathBuf {
    kvad::hub::config_dir().join("kvad.toml")
}

impl Config {
    /// Read `path`, or return the defaults if there is no file there.
    ///
    /// A missing file is not an error — most people will never write one —
    /// but a file that exists and cannot be parsed is, and the error says
    /// which file. Silently falling back to the defaults after someone has
    /// written a config is how a server ends up listening somewhere nobody
    /// expected.
    pub fn load(path: &Path) -> Res<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
            Err(e) => return Err(format!("could not read {}: {e}", path.display()).into()),
        };
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// Refuse combinations that would be an accident rather than a choice.
    ///
    /// There is one, and it is the one that matters: `auth.mode = "none"`
    /// means every request is an administrator, which is fine when the only
    /// way to reach the socket is to be sitting at the machine and is a gift
    /// to the internet otherwise. `insecure` is the caller saying they know;
    /// it exists so that someone behind their own authenticating proxy is not
    /// stuck, and it has to be typed every time.
    pub fn check(&self, insecure: bool) -> Res<()> {
        if self.auth.mode == Mode::None && !is_loopback(&self.server.bind) && !insecure {
            return Err(format!(
                "refusing to bind {} with auth.mode = \"none\": everyone who can reach that \
                 address would be an administrator. Set an auth mode, bind 127.0.0.1, or pass \
                 --insecure if something in front of this is already authenticating.",
                self.server.bind
            )
            .into());
        }
        if self.auth.mode != Mode::None {
            return Err(format!(
                "auth.mode = \"{}\" is not implemented yet; only \"none\" is. \
                 Until it is, bind loopback.",
                self.auth.mode
            )
            .into());
        }
        Ok(())
    }
}

/// Whether an address can only be reached from this machine.
///
/// `::` and `0.0.0.0` are the ones worth being careful about: they are not
/// loopback, they are *every* interface, and a check that only compared
/// against `127.0.0.1` would wave them through.
fn is_loopback(addr: &SocketAddr) -> bool {
    match addr.ip() {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_file_is_the_defaults_and_a_broken_one_is_an_error() {
        let missing = std::env::temp_dir().join("kvad-no-such-config.toml");
        let _ = std::fs::remove_file(&missing);
        let cfg = Config::load(&missing).unwrap();
        assert_eq!(cfg.server.bind.port(), 8080);
        assert_eq!(cfg.auth.mode, Mode::None);
        assert!(is_loopback(&cfg.server.bind));

        let broken = std::env::temp_dir().join(format!("kvad-broken-{}.toml", std::process::id()));
        std::fs::write(&broken, "[server]\nbind = \"not an address\"\n").unwrap();
        let err = Config::load(&broken).unwrap_err().to_string();
        assert!(err.contains(broken.to_str().unwrap()), "the error does not say which file: {err}");
        std::fs::remove_file(&broken).unwrap();
    }

    /// A key nobody reads is a setting somebody thinks they changed.
    #[test]
    fn an_unknown_key_is_refused_rather_than_ignored() {
        let path = std::env::temp_dir().join(format!("kvad-typo-{}.toml", std::process::id()));
        std::fs::write(&path, "[server]\nbnid = \"0.0.0.0:80\"\n").unwrap();
        let err = Config::load(&path).unwrap_err().to_string();
        assert!(err.contains("bnid"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }

    /// Binding the world with no auth is the one mistake this can prevent.
    #[test]
    fn no_auth_off_loopback_is_refused_unless_insisted_on() {
        let mut cfg = Config::default();
        assert!(cfg.check(false).is_ok(), "loopback with no auth is the ordinary case");

        for address in ["0.0.0.0:8080", "[::]:8080", "192.168.1.10:8080"] {
            cfg.server.bind = address.parse().unwrap();
            let err = cfg.check(false).unwrap_err().to_string();
            assert!(err.contains("--insecure"), "{address}: {err}");
            assert!(cfg.check(true).is_ok(), "{address} should be allowed once insisted on");
        }

        // Loopback in either family is still loopback.
        for address in ["127.0.0.1:8080", "[::1]:8080"] {
            cfg.server.bind = address.parse().unwrap();
            assert!(cfg.check(false).is_ok(), "{address}");
        }
    }

    /// A mode that is named but not built refuses to start, rather than
    /// starting and letting everybody in.
    #[test]
    fn an_unimplemented_auth_mode_does_not_quietly_become_none() {
        let path = std::env::temp_dir().join(format!("kvad-oidc-{}.toml", std::process::id()));
        std::fs::write(&path, "[auth]\nmode = \"oidc\"\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.auth.mode, Mode::Oidc);
        let err = cfg.check(false).unwrap_err().to_string();
        assert!(err.contains("not implemented"), "{err}");
        std::fs::remove_file(&path).unwrap();
    }
}
