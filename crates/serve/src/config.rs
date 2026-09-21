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
    /// Load the active model as soon as the server is listening.
    ///
    /// Without this the engine holds nothing until somebody asks for
    /// something, and the first request after a restart pays the load — tens
    /// of seconds, or ninety for DeepSeek-V2-Lite — or is refused outright
    /// for naming no model. A service that comes back after a reboot should
    /// come back ready.
    ///
    /// On by default, and the cost is honest rather than hidden: the model
    /// that was active when the server stopped is in memory again a minute
    /// later, whether or not anybody turns up. A machine that would rather
    /// spend that memory on something else sets this to false, and the first
    /// request loads as before.
    ///
    /// It is here rather than in the database because it is read once, at
    /// startup, before there is a request to change it.
    pub autoload: bool,
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
    pub oidc: Oidc,
}

/// An external identity provider, and who it is allowed to let in.
///
/// The allow-list is not optional and there is no default that means
/// "anybody". An OIDC client pointed at a public provider with no allow-list
/// would let every Google account in the world sign in to this machine, so a
/// configuration that forgets one is refused at startup rather than at the
/// moment somebody notices.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Oidc {
    /// The provider's issuer URL, e.g. `https://accounts.google.com`. Its
    /// `/.well-known/openid-configuration` is read from here.
    pub issuer: String,
    pub client_id: String,
    pub client_secret: String,
    /// Where the provider sends the browser back to. Must be registered with
    /// the provider, and must be this server's own
    /// `…/api/auth/oidc/callback`.
    pub redirect_url: String,
    /// Exact addresses that may sign in, compared without regard to case.
    pub allow_emails: Vec<String>,
    /// Whole domains that may, e.g. `example.com`.
    pub allow_domains: Vec<String>,
    /// Who becomes an administrator, by address.
    pub admin_emails: Vec<String>,
    /// A claim in the ID token holding group or role names, e.g. `groups`.
    pub role_claim: Option<String>,
    /// Values of that claim which mean "administrator".
    pub admin_roles: Vec<String>,
    /// Scopes asked for beyond `openid email profile`.
    pub scopes: Vec<String>,
}

impl Oidc {
    /// What is missing, if anything.
    pub fn check(&self) -> Result<(), String> {
        for (name, value) in [
            ("issuer", &self.issuer),
            ("client_id", &self.client_id),
            ("redirect_url", &self.redirect_url),
        ] {
            if value.trim().is_empty() {
                return Err(format!("auth.oidc.{name} is not set"));
            }
        }
        if self.allow_emails.is_empty() && self.allow_domains.is_empty() {
            return Err(
                "auth.oidc needs allow_emails or allow_domains: without one, everybody with \
                 an account at the provider could sign in to this machine"
                    .into(),
            );
        }
        if self.role_claim.is_some() && self.admin_roles.is_empty() {
            return Err("auth.oidc.role_claim is set but admin_roles is empty, so it decides \
                        nothing"
                .into());
        }
        Ok(())
    }

    /// Whether this address is on the list.
    pub fn allows(&self, email: &str) -> bool {
        let email = email.trim().to_ascii_lowercase();
        if self.allow_emails.iter().any(|a| a.trim().eq_ignore_ascii_case(&email)) {
            return true;
        }
        // The domain is what follows the last `@`; an address with none is
        // not an address.
        match email.rsplit_once('@') {
            Some((_, domain)) => {
                self.allow_domains.iter().any(|d| d.trim().eq_ignore_ascii_case(domain))
            }
            None => false,
        }
    }

    /// Whether this address, with these claimed roles, is an administrator.
    pub fn is_admin(&self, email: &str, roles: &[String]) -> bool {
        if self.admin_emails.iter().any(|a| a.trim().eq_ignore_ascii_case(email.trim())) {
            return true;
        }
        self.role_claim.is_some()
            && roles.iter().any(|r| self.admin_roles.iter().any(|a| a.trim() == r.trim()))
    }
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
        Server { bind: SocketAddr::from(([127, 0, 0, 1], 8080)), autoload: true }
    }
}

impl Default for Database {
    fn default() -> Self {
        Database { path: kvad::weights::data_dir().join("kvad.db") }
    }
}

impl Default for Auth {
    fn default() -> Self {
        Auth { mode: Mode::None, oidc: Oidc::default() }
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
        // Which modes exist is `auth::provider`'s to say, not this file's: it
        // is the one that has to build them.
        crate::auth::provider(self.auth.mode, &self.auth.oidc)?;
        if self.auth.mode == Mode::Oidc {
            self.auth.oidc.check()?;
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
        assert!(cfg.server.autoload, "a service should come back ready by default");

        let broken = std::env::temp_dir().join(format!("kvad-broken-{}.toml", std::process::id()));
        std::fs::write(&broken, "[server]\nbind = \"not an address\"\n").unwrap();
        let err = Config::load(&broken).unwrap_err().to_string();
        assert!(err.contains(broken.to_str().unwrap()), "the error does not say which file: {err}");
        std::fs::remove_file(&broken).unwrap();
    }

    /// Turning the autoload off has to be possible without restating the
    /// bind address, which is the whole point of `default` on the section.
    #[test]
    fn the_autoload_can_be_turned_off_on_its_own() {
        let path = std::env::temp_dir().join(format!("kvad-autoload-{}.toml", std::process::id()));
        std::fs::write(&path, "[server]\nautoload = false\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert!(!cfg.server.autoload);
        assert_eq!(cfg.server.bind.port(), 8080, "the bind default was lost");
        std::fs::remove_file(&path).unwrap();
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

    /// The one configuration mistake that would let the whole internet in.
    #[test]
    fn oidc_without_an_allow_list_is_refused() {
        let mut o = Oidc {
            issuer: "https://accounts.example.com".into(),
            client_id: "abc".into(),
            redirect_url: "https://kvad.example/api/auth/oidc/callback".into(),
            ..Oidc::default()
        };
        let err = o.check().unwrap_err();
        assert!(err.contains("allow_emails"), "{err}");

        o.allow_domains = vec!["example.com".into()];
        assert!(o.check().is_ok());

        // And each piece it cannot work without is named on its own.
        for missing in ["issuer", "client_id", "redirect_url"] {
            let mut o = o.clone();
            match missing {
                "issuer" => o.issuer.clear(),
                "client_id" => o.client_id.clear(),
                _ => o.redirect_url.clear(),
            }
            assert!(o.check().unwrap_err().contains(missing));
        }

        // A role claim that decides nothing is a mistake worth naming too.
        let mut o = o.clone();
        o.role_claim = Some("groups".into());
        assert!(o.check().unwrap_err().contains("admin_roles"));
    }

    #[test]
    fn the_allow_list_matches_addresses_and_domains_without_case() {
        let o = Oidc {
            allow_emails: vec!["Ada@Example.COM".into()],
            allow_domains: vec!["Kvad.test".into()],
            admin_emails: vec!["ada@example.com".into()],
            role_claim: Some("groups".into()),
            admin_roles: vec!["kvad-admins".into()],
            ..Oidc::default()
        };
        assert!(o.allows("ada@example.com"));
        assert!(o.allows("ADA@EXAMPLE.COM"));
        assert!(o.allows("anyone@kvad.test"));
        assert!(!o.allows("bob@example.com"), "only ada is listed at example.com");
        assert!(!o.allows("ada@example.com.evil.test"));
        assert!(!o.allows("not-an-address"));
        assert!(!o.allows(""));

        // Admin by address, or by a claimed role, and neither by accident.
        assert!(o.is_admin("ada@example.com", &[]));
        assert!(o.is_admin("bob@kvad.test", &["kvad-admins".into()]));
        assert!(!o.is_admin("bob@kvad.test", &["everyone".into()]));
        assert!(!o.is_admin("bob@kvad.test", &[]));

        // With no role_claim configured, a claimed role means nothing.
        let emails_only = Oidc { role_claim: None, ..o.clone() };
        assert!(!emails_only.is_admin("bob@kvad.test", &["kvad-admins".into()]));
    }

    /// A mode is checked for what it needs before the socket opens, not when
    /// somebody first tries to use it.
    #[test]
    fn an_auth_mode_that_cannot_work_stops_the_server() {
        let path = std::env::temp_dir().join(format!("kvad-oidc-{}.toml", std::process::id()));
        std::fs::write(&path, "[auth]\nmode = \"oidc\"\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.auth.mode, Mode::Oidc);
        // Named but with nothing under it: refused, and the message says
        // which key is missing rather than "misconfigured".
        let err = cfg.check(false).unwrap_err().to_string();
        assert!(err.contains("issuer"), "{err}");

        // Filled in properly, it starts — and may face the network, because
        // unlike `none` it has something to check.
        std::fs::write(
            &path,
            r#"
[server]
bind = "0.0.0.0:8080"
[auth]
mode = "oidc"
[auth.oidc]
issuer = "https://accounts.example.com"
client_id = "abc"
redirect_url = "https://kvad.example/api/auth/oidc/callback"
allow_domains = ["example.com"]
"#,
        )
        .unwrap();
        Config::load(&path).unwrap().check(false).unwrap();
        std::fs::remove_file(&path).unwrap();
    }

    /// The modes that *are* built start, and a mode with accounts may bind
    /// somewhere other than loopback without being insisted on: the reason
    /// `none` may not is that it has nothing to check.
    #[test]
    fn a_mode_with_accounts_may_face_the_network() {
        let mut cfg = Config::default();
        cfg.server.bind = "0.0.0.0:8080".parse().unwrap();
        for mode in [Mode::Local, Mode::Basic] {
            cfg.auth.mode = mode;
            assert!(cfg.check(false).is_ok(), "{mode} was refused");
        }
        cfg.auth.mode = Mode::None;
        assert!(cfg.check(false).is_err());
    }
}
