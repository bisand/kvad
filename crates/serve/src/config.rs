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

/// The port this server listens on when nobody says otherwise.
///
/// `KVAD` on a phone keypad, which is the only reason it is memorable and a
/// good enough one. What matters more is what it is not: 8080 was the old
/// default and is the most contended port on a developer's machine — Tomcat,
/// llama.cpp, LocalAI, Open WebUI, and every reverse proxy and ssh tunnel
/// anybody has ever left running. The machine this was changed on already
/// had something answering there.
///
/// Checked rather than guessed: unassigned in IANA's registry — 5820 is the
/// nearest neighbour — and in `/etc/services`, absent from Chrome's blocked
/// port list, since a web UI on a blocked port is a page that will not load,
/// and well below the ephemeral range, so the operating system will not hand
/// it to somebody's outgoing connection first. It collides with none of
/// Ollama (11434), LM Studio (1234), vLLM (8000), SGLang (30000), ComfyUI
/// (8188), text-generation-webui (7860) or GPT4All (4891).
///
/// The number itself lives in `kvad::client`, because the CLI looks for a
/// running server here before it does anything itself, and two copies of a
/// port number are one change away from a CLI that never finds its server.
pub const DEFAULT_PORT: u16 = kvad::client::DEFAULT_PORT;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    pub server: Server,
    pub data: Data,
    pub database: Database,
    pub auth: Auth,
    pub videos: Videos,
    pub cluster: Cluster,
    pub worker: Worker,
    pub remote: Remote,
    /// The command line's, not the server's. It is here so that a file with
    /// it in is not refused as having a key nobody knows.
    pub client: Client,
}

/// Where `kvad` sends its commands; see `kvad::client`. Read by the CLI and
/// ignored here: a server has no business knowing which server its own
/// machine's CLI talks to.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Client {
    pub url: Option<String>,
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
    /// Off by default. The model that was active when the server stopped
    /// would be in memory again a minute later whether or not anybody turns
    /// up, and a restart is not the moment to decide to spend that memory —
    /// or to start a load that has to be waited out before anything else on
    /// the queue runs. With it off, the first request loads, as it would
    /// with `load_on_request`. A machine that should come back ready sets
    /// this to true.
    ///
    /// It is here rather than in the database because it is read once, at
    /// startup, before there is a request to change it.
    pub autoload: bool,
    /// Load a model that a request names and that is not in memory, if it
    /// fits beside what is.
    ///
    /// Nothing is ever unloaded to make room: a model that does not fit is
    /// refused, with the models holding the memory named. On by default,
    /// because a client like OpenCode picks its model per request and would
    /// otherwise need somebody to load each one first. Off for a machine
    /// where loading should only ever be something a person did — a
    /// benchmark run is one.
    pub load_on_request: bool,
    /// Gigabytes the models in memory may take between them. Unset, this
    /// machine's usable memory; see `kvad::machine::usable_memory`.
    pub memory_gb: Option<f64>,
    /// Tokens of KV cache each model in memory is charged for, when deciding
    /// whether another fits. See `memory::DEFAULT_CONTEXT` for the default
    /// and why.
    pub context: Option<usize>,
    /// How many generations may wait for the engine before the next request
    /// is answered 503 with a `Retry-After`. Unset,
    /// `scheduler::DEFAULT_LIMIT`; zero lets any number wait.
    pub queue: Option<usize>,
    /// How many of those may be one account's, before that account's next
    /// request is answered 429. Unset, `scheduler::DEFAULT_SHARE`; zero
    /// lets one account have the whole queue. Not applied with
    /// `auth.mode = "none"`, where there is nobody to share with.
    pub queue_per_account: Option<usize>,
    /// How many generations one account may ask for in a minute, before
    /// its next request is answered 429 with the wait in `Retry-After`.
    /// Unset or zero, any number: an agent asks many times a minute as a
    /// matter of course, and what is too many depends on who shares the
    /// machine. Not applied with `auth.mode = "none"`.
    pub generations_per_minute: Option<usize>,
    /// How many replies run at once on one model, each a token further
    /// every step. Unset, `scheduler::DEFAULT_BATCH`; one is one at a time.
    ///
    /// More at once is more tokens a second in total and fewer for each
    /// reply, and a reply that begins makes the others wait while its
    /// prompt is read.
    pub batch: Option<usize>,
}

/// Other machines to put a model's layers on: see `docs/cluster-plan.md`.
///
/// With workers named, a language model loaded at a CPU backend is placed
/// across this machine and them: as many layers as fit on the fastest,
/// then the next. A model that fits here stays here whole, and its load is
/// the load it always was, so naming workers costs the models that do not
/// need them nothing but the asking.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Cluster {
    /// Each by the name `kvad worker` prints, its id, or an address.
    pub workers: Vec<String>,
    /// What to place layers for: `reply`, `prompt`, or `N:M` for N tokens
    /// read to M written.
    pub place: String,
    /// The layers each machine holds, this one first, for every model
    /// loaded: `0..12,12..24`. For a rig with one model on it, where the
    /// placing wanted is not the one worked out.
    pub split: Option<String>,
}

impl Default for Cluster {
    fn default() -> Self {
        Cluster { workers: Vec::new(), place: "reply".into(), split: None }
    }
}

/// This server as a worker for another machine's models.
///
/// It answers a coordinator that holds the same cluster key, which is the
/// file `cluster.key` in the data directory: made on the first start with
/// this on, and copied to the other machines.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Worker {
    pub enabled: bool,
    pub port: u16,
}

impl Default for Worker {
    fn default() -> Self {
        Worker { enabled: false, port: kvad::cluster::wire::PORT }
    }
}

/// Reaching this server from anywhere, through the tunnel of
/// `docs/relay-plan.md`: end-to-end encrypted, directly when a path can be
/// made and through the relays here when it cannot.
///
/// The server's identity is `remote.key` in the data directory, and its knock
/// `remote.knock` beside it, both made on the first start with this on. The
/// address it prints at startup, `kvad://…`, is what `kvad --remote` dials.
///
/// A tunnel is a way to reach the server, not a way around signing in: a
/// tunnelled request needs a key or a session like any other, so this is
/// refused with `auth.mode = "none"`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Remote {
    pub enabled: bool,
    /// `https://` relays, tried in order.
    pub relays: Vec<String>,
    /// `auto`, every path; `off`, the relay only, with no UDP socket open.
    pub direct: String,
    /// The UDP port for direct connections, fixed so that it can be
    /// forwarded by hand. Kvad never asks the router to open it.
    pub port: u16,
    /// Devices let in besides the paired ones (`kvad devices`): each the id
    /// `kvad remote id` prints on it. Pairing is the usual way; this is for a
    /// device that is to be let in without an account's key.
    pub allow: Vec<String>,
}

impl Default for Remote {
    fn default() -> Self {
        Remote { enabled: false, relays: Vec::new(), direct: "auto".into(), port: REMOTE_PORT, allow: Vec::new() }
    }
}

/// UDP, beside the HTTP port.
pub const REMOTE_PORT: u16 = 5824;

impl Remote {
    /// What the tunnel is started with, from this section and the data
    /// directory its secrets are kept in.
    pub fn tunnel(&self, data_dir: &Path) -> Result<kvad_tunnel::ServerConfig, String> {
        let direct = match self.direct.as_str() {
            "auto" => kvad_tunnel::Direct::Auto,
            "off" => kvad_tunnel::Direct::Off,
            "lan" => {
                return Err("remote.direct = \"lan\" is planned but not built yet: use \"auto\" or \"off\"".into())
            }
            other => return Err(format!("remote.direct = \"{other}\": use \"auto\" or \"off\"")),
        };
        let allowed = kvad_tunnel::Allowed::new(self.ids()?);
        if self.relays.is_empty() && direct == kvad_tunnel::Direct::Off {
            return Err("remote.direct = \"off\" needs remote.relays: with neither, nothing could reach this server".into());
        }
        Ok(kvad_tunnel::ServerConfig {
            key: data_dir.join("remote.key"),
            knock: data_dir.join("remote.knock"),
            relays: self.relays.clone(),
            direct,
            port: self.port,
            allowed,
        })
    }

    /// `allow`, read: devices let in whatever the `devices` table says.
    pub fn ids(&self) -> Result<Vec<kvad_tunnel::NodeId>, String> {
        self.allow.iter().map(|id| id.parse()).collect::<Result<Vec<_>, _>>().map_err(|e| format!("remote.allow: {e}"))
    }
}

/// What happens to a video once it is made.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Videos {
    /// The `ffmpeg` that compresses each video: `auto` to look for one,
    /// `off` to keep the uncompressed file, or a path. See
    /// `videos::find_ffmpeg`.
    pub ffmpeg: String,
}

impl Default for Videos {
    fn default() -> Self {
        Videos { ffmpeg: "auto".into() }
    }
}

/// Where the database, images, datasets and trained models go.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Data {
    /// As written: `~/` and a path relative to the file are resolved by
    /// [`Config::data_dir`].
    pub dir: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Database {
    /// `kvad.db` in the data directory when not given. Resolved late, by
    /// [`Config::database_path`], because the data directory is only known
    /// once the whole file has been read.
    pub path: Option<PathBuf>,
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
        Server {
            bind: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            autoload: false,
            load_on_request: true,
            memory_gb: None,
            context: None,
            queue: None,
            queue_per_account: None,
            generations_per_minute: None,
            batch: None,
        }
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

    /// The data directory this server uses: `KVAD_DATA_DIR`, else `[data]
    /// dir` in this file, else the default. `path` is the file this was read
    /// from, which a relative `dir` is taken from.
    ///
    /// This file's word rather than whatever [`kvad::weights::data_dir`]
    /// would find, because that reads the default `kvad.toml` and a server
    /// started with `--config` may have been given another.
    pub fn data_dir(&self, path: &Path) -> PathBuf {
        self.chosen_data_dir(path).unwrap_or_else(kvad::weights::default_data_dir)
    }

    /// The same, but nothing when neither names one: what
    /// [`kvad::weights::set_data_dir`] wants, since only a data directory
    /// somebody chose takes the Hub's cache with it.
    pub fn chosen_data_dir(&self, path: &Path) -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os("KVAD_DATA_DIR").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(dir));
        }
        let dir = self.data.dir.as_deref().filter(|d| !d.trim().is_empty())?;
        Some(kvad::weights::configured_dir(dir, path))
    }

    /// `[database] path`, else `kvad.db` in the data directory.
    pub fn database_path(&self) -> PathBuf {
        self.database.path.clone().unwrap_or_else(|| kvad::weights::data_dir().join("kvad.db"))
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
        // Not even with --insecure: that is for something in front of this
        // server that authenticates, and nothing stands in front of a tunnel.
        if self.remote.enabled && self.auth.mode == Mode::None {
            return Err("refusing [remote] with auth.mode = \"none\": every device that reached this \
                        server through the tunnel would be an administrator. Set an auth mode, and \
                        give each device a key (kvad auth login --remote kvad://…)."
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

    fn remote(toml: &str) -> Config {
        toml::from_str(toml).unwrap()
    }

    const ID: &str = "14713dcae14786ed6e5c20f250a2c67ab31ee028a72d1d12ec457ca850a2a310";

    /// A tunnelled request has to sign in like any other, so there is no
    /// `--insecure` way round it: nothing stands in front of a tunnel.
    #[test]
    fn a_tunnel_is_refused_without_accounts() {
        let open = remote(&format!("[remote]\nenabled = true\nallow = [\"{ID}\"]\nrelays = [\"https://relay.kvad.eu\"]\n"));
        assert_eq!(open.auth.mode, Mode::None);
        let e = open.check(false).unwrap_err().to_string();
        assert!(e.contains("[remote]"), "{e}");
        assert!(open.check(true).is_err(), "--insecure must not let a tunnel run without accounts");
        let signed = remote(&format!("[auth]\nmode = \"local\"\n[remote]\nenabled = true\nallow = [\"{ID}\"]\n"));
        assert!(signed.check(false).is_ok());
        // Off, it is nobody's business what the rest says.
        assert!(Config::default().check(false).is_ok());
    }

    #[test]
    fn a_tunnel_says_what_is_wrong_with_its_section() {
        let dir = Path::new("/tmp/kvad");
        let ok = |t: &str| remote(t).remote.tunnel(dir);
        let base = format!("allow = [\"{ID}\"]\n");
        let t = ok(&format!("[remote]\n{base}")).unwrap();
        assert_eq!((t.port, t.direct), (REMOTE_PORT, kvad_tunnel::Direct::Auto));
        assert_eq!(t.key, dir.join("remote.key"));
        assert!(ok("[remote]\n").is_ok(), "no allow is fine: devices are paired");
        assert!(ok("[remote]\nallow = [\"nope\"]\n").unwrap_err().contains("not a node id"));
        assert!(ok(&format!("[remote]\n{base}direct = \"lan\"\n")).unwrap_err().contains("not built yet"));
        assert!(ok(&format!("[remote]\n{base}direct = \"sometimes\"\n")).is_err());
        assert!(ok(&format!("[remote]\n{base}direct = \"off\"\n")).unwrap_err().contains("needs remote.relays"));
        let relayed = ok(&format!("[remote]\n{base}direct = \"off\"\nrelays = [\"https://relay.kvad.eu\"]\n")).unwrap();
        assert_eq!(relayed.direct, kvad_tunnel::Direct::Off);
    }

    /// `[data] dir` is where the database goes too, unless `[database] path`
    /// says otherwise, and a relative one is read from the file's directory.
    #[test]
    fn the_data_directory_comes_from_the_file_that_was_read() {
        if std::env::var_os("KVAD_DATA_DIR").is_some() {
            return; // the environment outranks any file, which is the point of it
        }
        let dir = std::env::temp_dir().join(format!("kvad-config-data-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kvad.toml");

        std::fs::write(&path, "[data]\ndir = \"store\"\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.data_dir(&path), dir.join("store"));

        std::fs::write(&path, "[data]\ndir = \"/srv/kvad\"\n").unwrap();
        assert_eq!(Config::load(&path).unwrap().data_dir(&path), PathBuf::from("/srv/kvad"));

        std::fs::write(&path, "[server]\nautoload = false\n").unwrap();
        assert_eq!(Config::load(&path).unwrap().data_dir(&path), kvad::weights::default_data_dir());

        std::fs::write(&path, "[database]\npath = \"/srv/other.db\"\n").unwrap();
        assert_eq!(Config::load(&path).unwrap().database_path(), PathBuf::from("/srv/other.db"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_absent_file_is_the_defaults_and_a_broken_one_is_an_error() {
        let missing = std::env::temp_dir().join("kvad-no-such-config.toml");
        let _ = std::fs::remove_file(&missing);
        let cfg = Config::load(&missing).unwrap();
        assert_eq!(cfg.server.bind.port(), DEFAULT_PORT);
        assert_eq!(cfg.auth.mode, Mode::None);
        assert!(is_loopback(&cfg.server.bind));
        assert!(!cfg.server.autoload, "a restart should not load anything unless asked to");

        let broken = std::env::temp_dir().join(format!("kvad-broken-{}.toml", std::process::id()));
        std::fs::write(&broken, "[server]\nbind = \"not an address\"\n").unwrap();
        let err = Config::load(&broken).unwrap_err().to_string();
        assert!(err.contains(broken.to_str().unwrap()), "the error does not say which file: {err}");
        std::fs::remove_file(&broken).unwrap();
    }

    /// Turning the autoload on has to be possible without restating the
    /// bind address, which is the whole point of `default` on the section.
    #[test]
    fn the_autoload_can_be_turned_on_on_its_own() {
        let path = std::env::temp_dir().join(format!("kvad-autoload-{}.toml", std::process::id()));
        std::fs::write(&path, "[server]\nautoload = true\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert!(cfg.server.autoload);
        assert_eq!(cfg.server.bind.port(), DEFAULT_PORT, "the bind default was lost");
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

        for address in ["0.0.0.0:5823", "[::]:5823", "192.168.1.10:5823"] {
            cfg.server.bind = address.parse().unwrap();
            let err = cfg.check(false).unwrap_err().to_string();
            assert!(err.contains("--insecure"), "{address}: {err}");
            assert!(cfg.check(true).is_ok(), "{address} should be allowed once insisted on");
        }

        // Loopback in either family is still loopback.
        for address in ["127.0.0.1:5823", "[::1]:5823"] {
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
bind = "0.0.0.0:5823"
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
        cfg.server.bind = "0.0.0.0:5823".parse().unwrap();
        for mode in [Mode::Local, Mode::Basic] {
            cfg.auth.mode = mode;
            assert!(cfg.check(false).is_ok(), "{mode} was refused");
        }
        cfg.auth.mode = Mode::None;
        assert!(cfg.check(false).is_err());
    }

    /// The file the release bundles has to be a file the server starts with,
    /// including the section that is the CLI's and not its own — the server
    /// refuses unknown keys, so a `[client]` it did not know would stop it.
    #[test]
    fn the_bundled_example_is_a_config_the_server_takes() {
        let text = include_str!("../../../docs/kvad.example.toml");
        let cfg: Config = toml::from_str(text).expect("kvad.example.toml does not parse");
        assert_eq!(cfg.server.bind.port(), DEFAULT_PORT);

        let with_client = format!("{text}\nurl = \"http://gpu-box.local:5823\"\n");
        let cfg: Config = toml::from_str(&with_client).expect("a [client] url was refused");
        assert_eq!(cfg.client.url.as_deref(), Some("http://gpu-box.local:5823"));
    }
}
