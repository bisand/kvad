//! The command line as a client of a running server.
//!
//! Without this, `kvad` on a machine where the service runs is a second,
//! parallel installation that happens to share a model directory: `kvad
//! chat` loads a second copy of weights the service already holds, which on
//! a 17 GB model is the difference between answering and swapping. So every
//! command that can be answered by a server is, when there is one, and the
//! question this module exists to answer first is *which* one.
//!
//! # Which kvad am I talking to?
//!
//! Resolved in one order, and reported on the first line of every command
//! rather than guessed silently:
//!
//! 1. `--remote URL` or `--local` on the command line.
//! 2. `KVAD_URL` in the environment.
//! 3. `url` under `[client]` in `kvad.toml`, the file the server reads its
//!    own configuration from.
//! 4. A kvad answering `/api/health` where this machine's service listens:
//!    the address its unit file gives it, else `server.bind` from the same
//!    file, else the default port.
//! 5. Nothing answering: this process, as the CLI always worked.
//!
//! The first three are things somebody said, and a server they name that
//! does not answer is an error, not a reason to quietly do the work here
//! instead. The fourth is a guess, which is why it is printed: `kvad chat`
//! uses a different machine's memory depending on whether something is
//! running, and that belongs on the screen, not in a manual.
//!
//! # Credentials
//!
//! None, in the common case: the default server binds loopback with no
//! authentication. Otherwise an API key, sent as `Authorization: Bearer`,
//! from `KVAD_API_KEY` or from the file `kvad auth login` writes —
//! `credentials.json` in the config directory, readable by its owner only,
//! one key per server.
//!
//! # Streams
//!
//! Loads, completions and job updates arrive as server-sent events. [`Events`]
//! reads them; what each command does with them is its own business, and
//! `--json` hands them over raw so that a command composes with `jq` rather
//! than only with eyes.

use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The port `kvad-serve` listens on when nobody says otherwise. Why this
/// number is written up where the server declares it, in its `config`
/// module, which uses this constant so that there is one of it.
pub const DEFAULT_PORT: u16 = 5823;

/// The environment variable that names a server.
pub const URL_VAR: &str = "KVAD_URL";

/// The environment variable that carries an API key, ahead of any stored.
pub const KEY_VAR: &str = "KVAD_API_KEY";

/// What the command line said about where to run.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Choice {
    /// Nothing: work it out.
    #[default]
    Unset,
    /// `--local`: this process, whatever is running.
    Local,
    /// `--remote URL`.
    Remote(String),
}

/// Where a command runs.
pub enum Target {
    /// In this process. `why` is the sentence for the first line.
    Local { why: String },
    Remote(Remote),
}

/// How a server came to be the one talked to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Why {
    Flag,
    Env,
    /// `[client] url` in this config file.
    Config(PathBuf),
    /// The service found answering at this machine's service address.
    Service,
}

impl Target {
    /// The first line of a command's output: where it is running, and why
    /// there. Written to stderr by the caller, so it never lands in a pipe.
    pub fn line(&self) -> String {
        match self {
            Target::Local { why } => format!("kvad: in this process ({why})"),
            Target::Remote(r) => format!("kvad: {} ({})", r.base, r.why_text()),
        }
    }
}

/// Resolve the target, in the order at the top of this module.
///
/// An error means somebody named a server (or a config file that cannot be
/// read) and it could not be used. It is never "nothing is running": that
/// answer is [`Target::Local`].
pub fn resolve(choice: &Choice) -> Result<Target, String> {
    match choice {
        Choice::Local => return Ok(Target::Local { why: "--local".into() }),
        Choice::Remote(url) => return Ok(Target::Remote(Remote::new(url, Why::Flag)?)),
        Choice::Unset => {}
    }
    if let Some(url) = std::env::var(URL_VAR).ok().filter(|v| !v.trim().is_empty()) {
        return Ok(Target::Remote(Remote::new(&url, Why::Env)?));
    }
    let settings = Settings::read()?;
    if let Some(url) = &settings.client_url {
        return Ok(Target::Remote(Remote::new(url, Why::Config(settings.path.clone()))?));
    }

    let (address, _) = service_address(&settings);
    let remote = Remote::new(&address, Why::Service)?;
    match remote.probe() {
        Probe::Kvad => Ok(Target::Remote(remote)),
        Probe::Other(what) => Ok(Target::Local {
            why: format!("{} answers {address}, and it is not kvad", what),
        }),
        Probe::Nothing => Ok(Target::Local { why: format!("nothing is answering on {address}") }),
    }
}

/// Where this machine's service listens, and what said so.
///
/// The unit file first, because it passes `--bind` on the command line and
/// that overrides the config file. Then `server.bind` from the config file,
/// then the default. An address bound to every interface is reached on
/// loopback: `0.0.0.0` is somewhere to listen, not somewhere to connect.
pub fn service_address(settings: &Settings) -> (String, &'static str) {
    let (bind, from) = match crate::daemon::installed_bind() {
        Some(bind) => (bind, "the installed service"),
        None => match &settings.server_bind {
            Some(bind) => (bind.clone(), "server.bind in kvad.toml"),
            None => (format!("127.0.0.1:{DEFAULT_PORT}"), "the default"),
        },
    };
    (reachable(&bind), from)
}

/// `0.0.0.0:P` as `127.0.0.1:P`, and `[::]:P` as `[::1]:P`. Anything else
/// as it was.
fn reachable(bind: &str) -> String {
    match bind.parse::<std::net::SocketAddr>() {
        Ok(addr) if addr.ip().is_unspecified() => {
            let ip: std::net::IpAddr = match addr {
                std::net::SocketAddr::V4(_) => std::net::Ipv4Addr::LOCALHOST.into(),
                std::net::SocketAddr::V6(_) => std::net::Ipv6Addr::LOCALHOST.into(),
            };
            std::net::SocketAddr::new(ip, addr.port()).to_string()
        }
        _ => bind.to_string(),
    }
}

// ---------------------------------------------------------------------------
// kvad.toml, as the client reads it
// ---------------------------------------------------------------------------

/// What the client wants from `kvad.toml`.
///
/// Read as a plain table rather than as the server's own `Config`, which
/// lives in the server's crate and would take the server's dependencies with
/// it. The server refuses a file it cannot parse; so does this, because a
/// file somebody wrote and that is being ignored is how a command ends up
/// talking to the wrong machine.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    pub path: PathBuf,
    pub client_url: Option<String>,
    pub server_bind: Option<String>,
    /// `[data] dir`, resolved; see [`crate::weights::data_dir`].
    pub data_dir: Option<PathBuf>,
}

impl Settings {
    pub fn path() -> PathBuf {
        crate::hub::config_dir().join("kvad.toml")
    }

    pub fn read() -> Result<Settings, String> {
        let path = Settings::path();
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Settings { path, ..Settings::default() })
            }
            Err(e) => return Err(format!("could not read {}: {e}", path.display())),
        };
        Settings::from_text(&text, path)
    }

    /// `text` as the file at `path` would be read.
    fn from_text(text: &str, path: PathBuf) -> Result<Settings, String> {
        let table: toml::Table =
            text.parse().map_err(|e| format!("could not parse {}: {e}", path.display()))?;
        let string = |section: &str, key: &str| {
            table.get(section)?.get(key)?.as_str().map(str::to_string)
        };
        Ok(Settings {
            client_url: string("client", "url").filter(|u| !u.trim().is_empty()),
            server_bind: string("server", "bind"),
            data_dir: string("data", "dir")
                .filter(|d| !d.trim().is_empty())
                .map(|d| crate::weights::configured_dir(&d, &path)),
            path,
        })
    }
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// An API key for one server, as `kvad auth login` stores it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Credential {
    pub key: String,
    /// The key's id on the server, so that `kvad auth logout` can revoke it
    /// rather than only forget it. Absent for a key pasted in, which the
    /// server can say nothing about until it is used.
    #[serde(default)]
    pub id: Option<i64>,
    /// Whose it is, for `kvad auth status` to say without asking.
    #[serde(default)]
    pub user: Option<String>,
    /// For a `kvad://` server: its whole address, knock and hints, so that
    /// `--remote kvad://<node id>` is enough once a device is paired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
}

/// `credentials.json` in the config directory. Beside `kvad.toml` rather
/// than in it: that file is the server's, is often shared, and is edited by
/// hand; this one holds secrets and is written only by `kvad auth`.
pub fn credentials_path() -> PathBuf {
    crate::hub::config_dir().join("credentials.json")
}

fn read_credentials() -> serde_json::Map<String, Value> {
    std::fs::read(credentials_path())
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// The stored key for a server, if there is one.
pub fn credential(base: &str) -> Option<Credential> {
    read_credentials().get(base).and_then(|v| serde_json::from_value(v.clone()).ok())
}

/// Store a key for a server, replacing any before it.
pub fn remember(base: &str, credential: &Credential) -> Res<PathBuf> {
    let mut all = read_credentials();
    all.insert(base.to_string(), serde_json::to_value(credential)?);
    write_credentials(&all)
}

/// Forget a server's key, returning what was stored.
pub fn forget(base: &str) -> Res<Option<Credential>> {
    let mut all = read_credentials();
    let gone = all.remove(base).and_then(|v| serde_json::from_value(v).ok());
    if gone.is_some() {
        write_credentials(&all)?;
    }
    Ok(gone)
}

/// Written to a new file and renamed over the old, and created readable by
/// its owner only: a key is as good as a password, and a window in which it
/// sat in a file anybody could read is a window too many.
fn write_credentials(all: &serde_json::Map<String, Value>) -> Res<PathBuf> {
    let path = credentials_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.new");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        use std::io::Write;
        let mut file = options.open(&tmp).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
        file.write_all(&serde_json::to_vec_pretty(all)?)?;
        file.write_all(b"\n")?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// The transport
// ---------------------------------------------------------------------------

/// How a request proves who it is from.
#[derive(Debug, Clone)]
pub enum Auth {
    None,
    Bearer(String),
    /// A session cookie, which `kvad auth login` holds for the few requests
    /// it takes to make a key and then gives up.
    Cookie(String),
    Basic { name: String, password: String },
}

/// A server, and how to talk to it.
#[derive(Clone)]
pub struct Remote {
    /// `http://host:port`, with no trailing slash, or `kvad://<node id>` for
    /// a server reached through the tunnel: what is shown, and what a key is
    /// kept under.
    pub base: String,
    pub why: Why,
    pub auth: Auth,
    agent: ureq::Agent,
    /// What request URLs start with: `base`, or for the tunnel a loopback
    /// stand-in that its connector never dials.
    http: String,
    /// The connection to a `kvad://` server, which every agent of this
    /// remote's opens its streams through.
    tunnel: Option<std::sync::Arc<kvad_tunnel::blocking::Tunnel>>,
}

/// The stand-in a tunnelled request's URL starts with. An address, so that no
/// name is looked up, and never connected to: the tunnel's connector takes
/// every connection.
const TUNNELLED: &str = "http://127.0.0.1:5823";

/// This machine's identity as a device of a server reached through the
/// tunnel; its id is what the server's `[remote] allow` lists.
pub fn device_key_path() -> PathBuf {
    crate::hub::config_dir().join("device.key")
}

/// The id of this machine's device key, made on first use, and whether it was
/// made just now.
pub fn device_id() -> Result<(String, bool), String> {
    let (key, made) = kvad_tunnel::Key::load(&device_key_path())?;
    Ok((key.id().to_string(), made))
}

/// What a request came back with when it was not a success.
#[derive(Debug)]
pub struct Failure {
    pub status: u16,
    /// What was asked for, which decides whether "sign in" is the advice: a
    /// wrong password at `/api/auth/login` is also a 401.
    pub path: String,
    /// The server's own sentence, from `{"error": "..."}`, or the status
    /// line when there was none.
    pub message: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)?;
        match self.status == 401 && !self.path.starts_with("/api/auth") {
            true => f.write_str("\n  this server wants a credential: kvad auth login"),
            false => Ok(()),
        }
    }
}

impl std::error::Error for Failure {}

/// What a server said when asked whether it is kvad.
enum Probe {
    Kvad,
    /// Something answered, and it was not a kvad. The status, for the note.
    Other(String),
    Nothing,
}

/// A request body.
pub enum Body<'a> {
    Json(&'a Value),
    Text(String),
    /// A file as it is: a picture going into a dataset.
    Bytes(Vec<u8>),
    /// `multipart/form-data`: a recording and the fields beside it, which is
    /// the only way OpenAI's audio endpoints take one.
    Form(Vec<Part>),
}

/// One field of a [`Body::Form`]: text, or a file with its name.
pub struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

impl Part {
    pub fn text(name: &str, value: impl ToString) -> Part {
        Part { name: name.into(), filename: None, data: value.to_string().into_bytes() }
    }

    pub fn file(name: &str, filename: &str, data: Vec<u8>) -> Part {
        Part { name: name.into(), filename: Some(filename.into()), data }
    }
}

/// A form's bytes, and the boundary between its parts (RFC 7578).
///
/// The boundary is chosen so that it is in none of the parts, which a file
/// could otherwise contain by chance and cut itself short at.
pub fn form(parts: &[Part]) -> (Vec<u8>, String) {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut boundary = format!("kvad-{stamp:x}");
    while parts.iter().any(|p| p.data.windows(boundary.len()).any(|w| w == boundary.as_bytes())) {
        boundary.push('x');
    }
    let mut body = Vec::new();
    for p in parts {
        // A quote or a line break in a name would end the header early.
        let quoted = |s: &str| s.replace(['"', '\r', '\n'], "_");
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{}\"", quoted(&p.name)).as_bytes());
        match &p.filename {
            Some(f) => body.extend_from_slice(
                format!("; filename=\"{}\"\r\nContent-Type: application/octet-stream\r\n\r\n", quoted(f)).as_bytes(),
            ),
            None => body.extend_from_slice(b"\r\n\r\n"),
        }
        body.extend_from_slice(&p.data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (body, boundary)
}

impl Remote {
    /// A server by URL, with whatever key is on hand for it.
    ///
    /// `host:port` is accepted for `http://host:port`, because that is what
    /// people type, and a trailing slash is dropped so that paths join.
    pub fn new(url: &str, why: Why) -> Result<Remote, String> {
        let url = url.trim();
        if url.starts_with("kvad://") {
            return Remote::tunnelled(url, why);
        }
        let base = match url.contains("://") {
            true => url.to_string(),
            false => format!("http://{url}"),
        };
        let base = base.trim_end_matches('/').to_string();
        let parsed = url::Url::parse(&base).map_err(|e| format!("`{url}` is not a URL: {e}"))?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(format!("`{url}` is not an http:// or https:// address"));
        }
        let auth = match std::env::var(KEY_VAR).ok().filter(|k| !k.trim().is_empty()) {
            Some(key) => Auth::Bearer(key.trim().to_string()),
            None => credential(&base).map_or(Auth::None, |c| Auth::Bearer(c.key)),
        };
        let loopback = parsed.host().is_some_and(|h| match h {
            url::Host::Domain(d) => d == "localhost",
            url::Host::Ipv4(ip) => ip.is_loopback(),
            url::Host::Ipv6(ip) => ip.is_loopback(),
        });
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                .user_agent(concat!("kvad/", env!("CARGO_PKG_VERSION")))
                .timeout_connect(Some(Duration::from_secs(5)))
                // Statuses are answers; `call` reads the server's sentence out
                // of them rather than ureq's.
                .http_status_as_error(false)
                // A proxy is for leaving the machine. Loopback through one is
                // a request that goes somewhere else to come back, or does not.
                .proxy(if loopback { None } else { ureq::Proxy::try_from_env() })
                .build(),
        );
        Ok(Remote { http: base.clone(), base, why, auth, agent, tunnel: None })
    }

    /// A server by its `kvad://` address, through the tunnel: dialled now, as
    /// this machine's device key, and held until this remote is dropped.
    fn tunnelled(url: &str, why: Why) -> Result<Remote, String> {
        // `kvad://<node id>` alone is a paired server's name: its address is
        // the one kept when this device was paired.
        let url = match url.contains('?') {
            true => url.to_string(),
            false => credential(url.trim_end_matches('/'))
                .and_then(|c| c.address)
                .ok_or_else(|| format!("{url} is not a server this device is paired with: give its whole address, with k= in it"))?,
        };
        let address: kvad_tunnel::Address = url.parse()?;
        let tunnel = std::sync::Arc::new(kvad_tunnel::blocking::Tunnel::connect(&device_key_path(), &address)?);
        let base = address.label();
        let auth = match std::env::var(KEY_VAR).ok().filter(|k| !k.trim().is_empty()) {
            Some(key) => Auth::Bearer(key.trim().to_string()),
            None => credential(&base).map_or(Auth::None, |c| Auth::Bearer(c.key)),
        };
        let tunnel = Some(tunnel);
        let agent = agent(&tunnel, ureq::Agent::config_builder()
            .user_agent(concat!("kvad/", env!("CARGO_PKG_VERSION")))
            .http_status_as_error(false)
            .proxy(None)
            .build());
        Ok(Remote { base, why, auth, agent, http: TUNNELLED.into(), tunnel })
    }

    /// The same server, proving itself a different way.
    pub fn with_auth(&self, auth: Auth) -> Remote {
        let mut other = self.clone();
        other.auth = auth;
        other
    }

    /// A `kvad://` server's whole address, knock and hints, to keep with its
    /// key.
    pub fn dial_address(&self) -> Option<String> {
        Some(self.tunnel.as_ref()?.address().to_string())
    }

    /// The path the tunnel is on, for a `kvad://` server.
    pub fn path(&self) -> Option<String> {
        self.tunnel.as_ref()?.path().map(|p| p.to_string())
    }

    /// The same server, asked by somebody who will not wait: half a second
    /// to connect and two for the whole answer.
    ///
    /// For shell completion, which runs between a Tab and the next keypress.
    /// A listing that is late is worth less than no listing, and the agent
    /// every other request uses has no limit on the answer at all, because a
    /// generation may take an hour.
    pub fn hurried(mut self) -> Remote {
        let loopback = self.http.contains("://127.0.0.1") || self.http.contains("://localhost") || self.http.contains("://[::1]");
        self.agent = agent(
            &self.tunnel,
            ureq::Agent::config_builder()
                .user_agent(concat!("kvad/", env!("CARGO_PKG_VERSION")))
                .timeout_connect(Some(Duration::from_millis(500)))
                .timeout_global(Some(Duration::from_secs(2)))
                .http_status_as_error(false)
                .proxy(if loopback { None } else { ureq::Proxy::try_from_env() })
                .build(),
        );
        self
    }

    pub fn why_text(&self) -> String {
        match &self.why {
            Why::Flag => "--remote".into(),
            Why::Env => URL_VAR.into(),
            Why::Config(path) => format!("client.url in {}", path.display()),
            Why::Service => "the kvad service; --local to run in this process instead".into(),
        }
    }

    /// Is a kvad answering here?
    ///
    /// `/api/health` first, with a short leash: this runs before every
    /// command that could go either way, and a connection refused on
    /// loopback is instant but a host that swallows packets is not. A 401 is
    /// a kvad that wants a credential — but so is any server with a login,
    /// so that case is confirmed at `/api/auth`, the one route that answers
    /// anybody, and whose `mode` field nothing else would have.
    fn probe(&self) -> Probe {
        let quick = agent(
            &self.tunnel,
            ureq::Agent::config_builder()
                .timeout_connect(Some(Duration::from_millis(400)))
                .timeout_global(Some(Duration::from_secs(3)))
                .http_status_as_error(false)
                .proxy(None)
                .build(),
        );
        let ask = |path: &str, auth: bool| -> Option<(u16, Value)> {
            let mut req = quick.get(format!("{}{path}", self.http));
            if let (true, Some(value)) = (auth, self.authorization()) {
                req = req.header("authorization", value);
            }
            let mut response = req.call().ok()?;
            let status = response.status().as_u16();
            let json = response.body_mut().read_json::<Value>().unwrap_or(Value::Null);
            Some((status, json))
        };
        match ask("/api/health", true) {
            None => Probe::Nothing,
            Some((200, health)) if health["status"] == "ok" => Probe::Kvad,
            Some((401 | 403, _)) => match ask("/api/auth", false) {
                Some((200, auth)) if auth["mode"].is_string() => Probe::Kvad,
                _ => Probe::Other("a server".into()),
            },
            Some((status, _)) => Probe::Other(format!("a server saying {status}")),
        }
    }

    fn authorization(&self) -> Option<String> {
        match &self.auth {
            Auth::None | Auth::Cookie(_) => None,
            Auth::Bearer(key) => Some(format!("Bearer {key}")),
            Auth::Basic { name, password } => {
                use base64_lite::encode;
                Some(format!("Basic {}", encode(format!("{name}:{password}").as_bytes())))
            }
        }
    }

    /// Send one request and hand back the response, whatever its status.
    fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<Body>,
        accept: &str,
    ) -> Res<ureq::http::Response<ureq::Body>> {
        let mut req = ureq::http::Request::builder()
            .method(method.to_ascii_uppercase().as_str())
            .uri(format!("{}{path}", self.http))
            .header("accept", accept);
        if let Some(value) = self.authorization() {
            req = req.header("authorization", value);
        }
        if let Auth::Cookie(cookie) = &self.auth {
            // A cookie on a request that changes something has to say where
            // it came from, or the server takes it for a forgery from some
            // other site's page. It came from here: this is the client, not
            // a page in a browser somebody else controls.
            req = req.header("cookie", cookie.as_str()).header("origin", self.http.as_str());
        }
        let sent = match body {
            None => self.agent.run(req.body(())?),
            Some(Body::Json(value)) => self
                .agent
                .run(req.header("content-type", "application/json").body(serde_json::to_vec(value)?)?),
            Some(Body::Text(text)) => self
                .agent
                .run(req.header("content-type", "text/plain; charset=utf-8").body(text.into_bytes())?),
            Some(Body::Bytes(bytes)) => self.agent.run(req.header("content-type", "application/octet-stream").body(bytes)?),
            Some(Body::Form(parts)) => {
                let (bytes, boundary) = form(&parts);
                self.agent.run(req.header("content-type", format!("multipart/form-data; boundary={boundary}")).body(bytes)?)
            }
        };
        sent.map_err(|e| match e {
            ureq::Error::Io(io) if io.kind() == std::io::ErrorKind::ConnectionRefused => {
                format!("nothing is answering at {}", self.base).into()
            }
            other => format!("could not reach {}: {other}", self.base).into(),
        })
    }

    /// A request whose answer is one JSON document.
    ///
    /// A status over 399 becomes a [`Failure`] carrying the server's own
    /// sentence, which is written for a person to read.
    pub fn call(&self, method: &str, path: &str, body: Option<Body>) -> Res<Value> {
        let response = self.send(method, path, body, "application/json")?;
        Ok(read_reply(response, path)?.0)
    }

    /// A request whose answer is a document to keep as it was sent: a
    /// transcript, whose formats are the server's to write. JSON read and
    /// written out again is not the same bytes — a float can come back a
    /// digit shorter — and `kvad transcribe` promises the same file from a
    /// server as from this process.
    pub fn call_text(&self, method: &str, path: &str, body: Option<Body>) -> Res<String> {
        let mut response = self.send(method, path, body, "*/*")?;
        if response.status().as_u16() >= 400 {
            read_reply(response, path)?;
            return Err(format!("{path} failed").into());
        }
        Ok(response.body_mut().with_config().limit(256 * 1024 * 1024).read_to_string()?)
    }

    /// The same, keeping the session cookie a sign-in sets.
    pub fn call_for_cookie(&self, method: &str, path: &str, body: Option<Body>) -> Res<(Value, Option<String>)> {
        let response = self.send(method, path, body, "application/json")?;
        read_reply(response, path)
    }

    pub fn get(&self, path: &str) -> Res<Value> {
        self.call("get", path, None)
    }

    pub fn post(&self, path: &str, body: &Value) -> Res<Value> {
        self.call("post", path, Some(Body::Json(body)))
    }

    pub fn patch(&self, path: &str, body: &Value) -> Res<Value> {
        self.call("patch", path, Some(Body::Json(body)))
    }

    pub fn delete(&self, path: &str) -> Res<Value> {
        self.call("delete", path, None)
    }

    /// A file, written to `to` as it arrives: `kvad videos get`, whose files
    /// run to hundreds of megabytes. Written aside and renamed, so that a
    /// download cut short leaves no file that looks whole. The bytes
    /// written come back.
    pub fn download(&self, path: &str, to: &std::path::Path) -> Res<u64> {
        let response = self.send("get", path, None, "*/*")?;
        if response.status().as_u16() >= 400 {
            read_reply(response, path)?;
            return Err(format!("{path} failed").into());
        }
        let aside = to.with_extension("part");
        let written = (|| {
            let mut file = std::fs::File::create(&aside)?;
            let n = std::io::copy(&mut response.into_body().into_reader(), &mut file)?;
            std::fs::rename(&aside, to)?;
            Ok::<u64, std::io::Error>(n)
        })();
        written.map_err(|e| {
            let _ = std::fs::remove_file(&aside);
            format!("could not write {}: {e}", to.display()).into()
        })
    }

    /// A request whose answer is a stream of server-sent events.
    pub fn stream(&self, method: &str, path: &str, body: Option<Body>) -> Res<Events<Stream>> {
        match self.exchange_with(method, path, body, "text/event-stream")? {
            Reply::Events(events) => Ok(events),
            Reply::Json(_) => Err(format!("expected a stream of events from {path}, and got a document").into()),
        }
    }

    /// A request that may answer either way: `kvad api`, which does not know
    /// in advance which route it is sending to.
    pub fn exchange(&self, method: &str, path: &str, body: Option<Body>) -> Res<Reply> {
        self.exchange_with(method, path, body, "application/json, text/event-stream")
    }

    fn exchange_with(&self, method: &str, path: &str, body: Option<Body>, accept: &str) -> Res<Reply> {
        let response = self.send(method, path, body, accept)?;
        let streams = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|k| k.starts_with("text/event-stream"));
        if streams && response.status().as_u16() < 400 {
            return Ok(Reply::Events(Events::new(BufReader::new(response.into_body().into_reader()))));
        }
        Ok(Reply::Json(read_reply(response, path)?.0))
    }
}

/// A response body being read as it arrives.
pub type Stream = BufReader<ureq::BodyReader<'static>>;

/// What a request came back as.
pub enum Reply {
    Json(Value),
    Events(Events<Stream>),
}

/// The body of a finished response, or its failure.
fn read_reply(mut response: ureq::http::Response<ureq::Body>, path: &str) -> Res<(Value, Option<String>)> {
    let status = response.status().as_u16();
    let cookie = response
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(str::to_string);
    let text = response
        .body_mut()
        .with_config()
        // Well past anything a listing is, and a bound all the same.
        .limit(256 * 1024 * 1024)
        .read_to_string()
        .unwrap_or_default();
    let json = match text.trim() {
        "" => Value::Null,
        t => serde_json::from_str(t).unwrap_or_else(|_| Value::String(text.clone())),
    };
    if status >= 400 {
        let message = match &json {
            Value::Object(o) => o.get("error").and_then(Value::as_str).map(str::to_string),
            Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
            _ => None,
        };
        let message = message.unwrap_or_else(|| {
            let reason = response.status().canonical_reason().unwrap_or("");
            format!("the server said {status} {reason}").trim_end().to_string()
        });
        return Err(Box::new(Failure { status, path: path.to_string(), message }));
    }
    Ok((json, cookie))
}

/// Base64 to bytes: how `/v1/images/generations` sends a picture.
pub fn decode_base64(text: &str) -> Res<Vec<u8>> {
    base64_lite::decode(text).ok_or_else(|| "the server sent an image that is not base64".into())
}

/// Bytes to base64: how `kvad videos make --image` sends a picture, as a
/// `data:` URL.
pub fn encode_base64(bytes: &[u8]) -> String {
    base64_lite::encode(bytes)
}

/// A tiny base64 encoder, for HTTP Basic, and its inverse, for the pictures
/// `kvad images make` is sent. Forty lines is less than a dependency.
mod base64_lite {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
            let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    /// The inverse of [`encode`]; `None` for anything that is not base64.
    pub fn decode(text: &str) -> Option<Vec<u8>> {
        let text = text.trim_end_matches('=');
        let mut out = Vec::with_capacity(text.len() * 3 / 4);
        let (mut acc, mut bits) = (0u32, 0);
        for c in text.bytes() {
            let v = ALPHABET.iter().position(|&a| a == c)? as u32;
            acc = acc << 6 | v;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
                acc &= (1 << bits) - 1;
            }
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Server-sent events
// ---------------------------------------------------------------------------

/// One event: its name (`message` when it had none) and its data.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub name: String,
    pub data: String,
}

impl Event {
    pub fn json(&self) -> Res<Value> {
        serde_json::from_str(&self.data)
            .map_err(|e| format!("a `{}` event that is not JSON ({e}): {}", self.name, self.data).into())
    }

    /// The event as one line of JSON, for `--json`: `{"event": …, "data": …}`,
    /// with the data parsed when it is JSON and kept as a string when it is
    /// not — OpenAI's stream ends with a bare `[DONE]`.
    pub fn to_json(&self) -> Value {
        let data = serde_json::from_str(&self.data).unwrap_or_else(|_| Value::String(self.data.clone()));
        serde_json::json!({ "event": self.name, "data": data })
    }
}

/// The events in a response, in order.
///
/// The format is lines: `event:` names the next event, each `data:` line adds
/// a line to its data, a line starting `:` is a comment — the server's
/// keep-alive, every fifteen seconds — and a blank line ends the event. An
/// event the stream ends in the middle of is dropped, as the specification
/// says: it was never finished, so it was never sent.
pub struct Events<R: BufRead> {
    reader: R,
}

impl<R: BufRead> Events<R> {
    pub fn new(reader: R) -> Self {
        Events { reader }
    }
}

impl<R: BufRead> Iterator for Events<R> {
    type Item = Res<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut name = String::new();
        let mut data = String::new();
        let mut any = false;
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => return Some(Err(format!("the stream broke off: {e}").into())),
            }
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !any {
                    continue;
                }
                if data.ends_with('\n') {
                    data.pop();
                }
                let name = if name.is_empty() { "message".to_string() } else { name };
                return Some(Ok(Event { name, data }));
            }
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" => {
                    name = value.to_string();
                    any = true;
                }
                "data" => {
                    data.push_str(value);
                    data.push('\n');
                    any = true;
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// What the command line covers
// ---------------------------------------------------------------------------

/// Every route the server has, and the command that reaches it.
///
/// `(method, path, command)`. Checked from both ends, so that "the CLI can
/// do everything the API can" stays true after the day it is written:
/// `kvad-serve`'s tests fail when a route it serves is in neither this table
/// nor [`NOT_COMMANDS`], and the CLI's own tests fail when a path here is
/// never requested by its source, or its source requests one that is not
/// here.
pub const COMMANDS: &[(&str, &str, &str)] = &[
    ("get", "/api/health", "kvad service status"),
    ("get", "/api/remote", "kvad remote status"),
    ("post", "/api/remote/pair", "kvad remote pair"),
    ("get", "/api/devices", "kvad devices"),
    ("delete", "/api/devices/{id}", "kvad devices rm"),
    ("get", "/api/openapi.json", "kvad api"),
    ("get", "/api/models", "kvad ls, kvad ps, kvad cache"),
    ("delete", "/api/models", "kvad rm"),
    ("get", "/api/models/search", "kvad search"),
    ("get", "/api/models/detail", "kvad info"),
    ("post", "/api/models/load", "kvad load"),
    ("post", "/api/models/unload", "kvad unload"),
    ("post", "/api/models/active", "kvad use"),
    ("post", "/api/models/pull", "kvad pull"),
    ("delete", "/api/qcache", "kvad cache"),
    ("post", "/v1/chat/completions", "kvad chat, kvad run"),
    ("get", "/v1/models", "kvad ls"),
    ("delete", "/api/generation", "kvad cancel"),
    ("delete", "/api/generation/{id}", "kvad cancel"),
    ("get", "/api/queue", "kvad queue"),
    ("get", "/api/conversations", "kvad conversations"),
    ("post", "/api/conversations", "kvad chat"),
    ("get", "/api/conversations/{id}", "kvad conversations show, kvad chat"),
    ("patch", "/api/conversations/{id}", "kvad conversations edit"),
    ("delete", "/api/conversations/{id}", "kvad conversations rm"),
    ("post", "/api/conversations/{id}/messages", "kvad chat"),
    ("post", "/api/playground/complete", "kvad run"),
    ("post", "/api/playground/tokenize", "kvad tokenize"),
    ("post", "/api/train", "kvad train"),
    ("get", "/api/train/options", "kvad train options"),
    ("post", "/api/tune", "kvad tune"),
    ("get", "/api/tune/options", "kvad tune options"),
    ("get", "/api/jobs", "kvad jobs"),
    ("get", "/api/jobs/{id}", "kvad jobs show"),
    ("delete", "/api/jobs/{id}", "kvad jobs cancel"),
    ("get", "/api/jobs/{id}/events", "kvad jobs watch"),
    ("get", "/api/jobs/{id}/pictures/{file}", "kvad jobs pictures"),
    ("get", "/api/datasets", "kvad datasets"),
    ("post", "/api/datasets", "kvad datasets add"),
    ("post", "/api/datasets/crawl", "kvad datasets crawl"),
    ("get", "/api/datasets/{id}", "kvad datasets show"),
    ("delete", "/api/datasets/{id}", "kvad datasets rm"),
    ("get", "/api/datasets/{id}/check", "kvad datasets check"),
    ("get", "/api/datasets/{id}/search", "kvad datasets search"),
    ("post", "/api/datasets/pictures", "kvad datasets add, kvad tune"),
    ("get", "/api/datasets/{id}/pictures", "kvad datasets show"),
    ("put", "/api/datasets/{id}/files/{file}", "kvad datasets add, kvad datasets put, kvad tune"),
    ("get", "/api/datasets/{id}/files/{file}", "kvad datasets get"),
    ("delete", "/api/datasets/{id}/files/{file}", "kvad datasets rm"),
    ("get", "/api/evals/suites", "kvad evals suites"),
    ("post", "/api/evals/suites", "kvad evals add"),
    ("patch", "/api/evals/suites/{id}", "kvad evals edit"),
    ("delete", "/api/evals/suites/{id}", "kvad evals rm"),
    ("post", "/api/evals/run", "kvad evals run"),
    ("post", "/api/evals/perplexity", "kvad evals perplexity"),
    ("get", "/api/evals/runs", "kvad evals"),
    ("get", "/api/evals/runs/{id}", "kvad evals show"),
    ("post", "/api/bench/run", "kvad bench run"),
    ("get", "/api/bench/runs", "kvad bench"),
    ("get", "/api/bench/runs/{id}", "kvad bench show"),
    ("get", "/api/metrics", "kvad metrics"),
    ("get", "/api/metrics/requests", "kvad metrics requests"),
    ("get", "/api/metrics/log", "kvad metrics log"),
    ("get", "/api/metrics/usage", "kvad metrics usage"),
    ("get", "/api/cluster", "kvad cluster status"),
    ("get", "/api/auth", "kvad auth status"),
    ("post", "/api/auth/login", "kvad auth login"),
    ("post", "/api/auth/logout", "kvad auth login"),
    ("post", "/api/auth/setup", "kvad auth setup"),
    ("post", "/api/auth/password", "kvad auth password"),
    ("get", "/api/users", "kvad users"),
    ("post", "/api/users", "kvad users add"),
    ("patch", "/api/users/{id}", "kvad users edit"),
    ("delete", "/api/users/{id}", "kvad users rm"),
    ("get", "/api/sessions", "kvad sessions"),
    ("delete", "/api/sessions/{hash}", "kvad sessions rm"),
    ("get", "/api/keys", "kvad keys"),
    ("post", "/api/keys", "kvad keys add, kvad auth login"),
    ("delete", "/api/keys/{id}", "kvad keys rm, kvad auth logout"),
    ("post", "/v1/images/generations", "kvad images make"),
    ("post", "/v1/images/edits", "kvad images edit"),
    ("get", "/api/images", "kvad images"),
    ("delete", "/api/images/{id}", "kvad images rm"),
    ("post", "/v1/videos", "kvad videos make"),
    ("get", "/v1/videos", "kvad videos"),
    ("get", "/v1/videos/{id}", "kvad videos show, kvad videos get"),
    ("get", "/v1/videos/{id}/events", "kvad videos make, kvad videos watch"),
    ("delete", "/v1/videos/{id}", "kvad videos rm"),
    ("get", "/v1/videos/{id}/content", "kvad videos get, kvad videos make"),
    ("post", "/v1/audio/transcriptions", "kvad transcribe"),
    ("post", "/v1/audio/translations", "kvad transcribe --translate"),
];

/// Routes that deliberately have no command, and why not.
pub const NOT_COMMANDS: &[(&str, &str, &str)] = &[
    (
        "get",
        "/api/auth/oidc/start",
        "a redirect to an identity provider, for a browser to follow. Sign in \
         there, make a key on the Account page, and give it to `kvad auth login --key`",
    ),
    ("get", "/api/auth/oidc/callback", "where the identity provider sends the browser back"),
    (
        "get",
        "/api/images/{id}",
        "the PNG itself, for the web UI's <img>. `kvad images make` writes each picture \
         to a file from the same bytes as it arrives",
    ),
    ("get", "/api/images/{id}/input", "the picture an edit was made from, for the web UI's <img>; whoever ran `kvad images edit` has the file"),
    ("get", "/api/images/{id}/mask", "an edit's mask, likewise"),
    (
        "get",
        "/api/settings",
        "the server's kvad.toml, for the web UI's Settings page. On the server's own \
         machine it is a file to read, at the path kvad-serve prints as it starts \
         (`kvad service logs`)",
    ),
    ("put", "/api/settings", "the same file, written; on the server's machine, an editor does it"),
    (
        "post",
        "/api/restart",
        "for the web UI, which has no other way to apply a settings change. On the \
         server's machine, `kvad service restart` restarts the service through launchd \
         or systemd",
    ),
    ("get", "/api/data", "the Settings page's view of the data directory; on the server's machine, `du` sees it"),
    ("get", "/api/data/plan", "the Settings page asking before it moves the data"),
    (
        "post",
        "/api/data/move",
        "moving the data from the Settings page, which then restarts the server; by hand it \
         is `install.sh --data-dir`, which prints the moves to make",
    ),
    ("delete", "/api/data/move", "cancelling a move the Settings page started"),
    ("delete", "/api/data/previous", "deleting what a move from the Settings page left behind"),
];

/// An agent with `config`, which for a tunnelled server opens every
/// connection through the tunnel.
fn agent(tunnel: &Option<std::sync::Arc<kvad_tunnel::blocking::Tunnel>>, config: ureq::config::Config) -> ureq::Agent {
    match tunnel {
        None => ureq::Agent::new_with_config(config),
        Some(t) => ureq::Agent::with_parts(
            config,
            through_tunnel::Connector(t.clone()),
            ureq::unversioned::resolver::DefaultResolver::default(),
        ),
    }
}

/// ureq's connections, made of streams through the tunnel: each one end of a
/// Unix socket pair the tunnel copies to a QUIC stream, so that no port is
/// opened anywhere for the CLI to talk through.
mod through_tunnel {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use ureq::unversioned::transport::{Buffers, ConnectionDetails, LazyBuffers, NextTimeout, Transport};

    pub struct Connector(pub Arc<kvad_tunnel::blocking::Tunnel>);

    impl std::fmt::Debug for Connector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Connector({})", self.0.address().label())
        }
    }

    impl ureq::unversioned::transport::Connector<()> for Connector {
        type Out = Socket;

        fn connect(&self, details: &ConnectionDetails, _: Option<()>) -> Result<Option<Socket>, ureq::Error> {
            let stream = self.0.open().map_err(|e| ureq::Error::Io(std::io::Error::other(e)))?;
            let buffers = LazyBuffers::new(details.config.input_buffer_size(), details.config.output_buffer_size());
            Ok(Some(Socket { stream, buffers }))
        }
    }

    pub struct Socket {
        stream: UnixStream,
        buffers: LazyBuffers,
    }

    impl std::fmt::Debug for Socket {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Socket(tunnel)")
        }
    }

    fn timed_out(e: &std::io::Error) -> bool {
        matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
    }

    impl Transport for Socket {
        fn buffers(&mut self) -> &mut dyn Buffers {
            &mut self.buffers
        }

        fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
            self.stream.set_write_timeout(timeout.not_zero().map(|t| *t))?;
            let output = &self.buffers.output()[..amount];
            match self.stream.write_all(output) {
                Err(e) if timed_out(&e) => Err(ureq::Error::Timeout(timeout.reason)),
                other => Ok(other?),
            }
        }

        fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
            self.stream.set_read_timeout(timeout.not_zero().map(|t| *t))?;
            let input = self.buffers.input_append_buf();
            let amount = match self.stream.read(input) {
                Err(e) if timed_out(&e) => return Err(ureq::Error::Timeout(timeout.reason)),
                other => other?,
            };
            self.buffers.input_appended(amount);
            Ok(amount > 0)
        }

        fn is_open(&mut self) -> bool {
            // A peek would say whether the other end has gone; a pooled
            // stream the server closed is found out at the next request
            // instead, which ureq retries.
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// `[data] dir` reaches the CLI resolved the way the server resolves
    /// it, so that the two agree on where the data is.
    #[test]
    fn the_data_directory_is_read_from_kvad_toml() {
        let path = PathBuf::from("/etc/kvad/kvad.toml");
        let dir = |text: &str| Settings::from_text(text, path.clone()).unwrap().data_dir;
        assert_eq!(dir("[data]\ndir = \"/srv/kvad\"\n"), Some(PathBuf::from("/srv/kvad")));
        assert_eq!(dir("[data]\ndir = \"store\"\n"), Some(PathBuf::from("/etc/kvad/store")));
        assert_eq!(dir("[data]\ndir = \"\"\n"), None);
        assert_eq!(dir("[server]\nbind = \"127.0.0.1:5823\"\n"), None);
    }

    fn events(text: &str) -> Vec<Event> {
        Events::new(Cursor::new(text.to_string())).map(|e| e.unwrap()).collect()
    }

    #[test]
    fn events_are_named_or_messages_and_comments_are_skipped() {
        let got = events(
            ": keep-alive\n\n\
             event: progress\ndata: {\"kind\":\"status\"}\n\n\
             data: [DONE]\n\n",
        );
        assert_eq!(
            got,
            [
                Event { name: "progress".into(), data: "{\"kind\":\"status\"}".into() },
                Event { name: "message".into(), data: "[DONE]".into() },
            ]
        );
    }

    #[test]
    fn data_lines_join_and_crlf_and_missing_spaces_are_taken() {
        let got = events("event:update\r\ndata:one\r\ndata: two\r\n\r\n");
        assert_eq!(got, [Event { name: "update".into(), data: "one\ntwo".into() }]);
    }

    /// Half an event is not an event.
    #[test]
    fn an_event_the_stream_ends_inside_is_dropped() {
        assert_eq!(events("event: loaded\ndata: {}\n"), []);
    }

    #[test]
    fn an_event_as_json_keeps_data_that_is_not_json() {
        let done = Event { name: "message".into(), data: "[DONE]".into() };
        assert_eq!(done.to_json(), serde_json::json!({ "event": "message", "data": "[DONE]" }));
        let some = Event { name: "token".into(), data: "{\"text\":\"hi\"}".into() };
        assert_eq!(some.to_json()["data"]["text"], "hi");
    }

    #[test]
    fn an_address_to_listen_on_is_turned_into_one_to_connect_to() {
        assert_eq!(reachable("0.0.0.0:5823"), "127.0.0.1:5823");
        assert_eq!(reachable("[::]:9000"), "[::1]:9000");
        assert_eq!(reachable("192.168.1.4:5823"), "192.168.1.4:5823");
        assert_eq!(reachable("box.local:5823"), "box.local:5823");
    }

    #[test]
    fn a_bare_address_is_http_and_a_trailing_slash_goes() {
        let r = Remote::new("127.0.0.1:5823", Why::Flag).unwrap();
        assert_eq!(r.base, "http://127.0.0.1:5823");
        let r = Remote::new("https://kvad.example.com/", Why::Flag).unwrap();
        assert_eq!(r.base, "https://kvad.example.com");
        assert!(Remote::new("ftp://x", Why::Flag).is_err());
    }

    #[test]
    fn base64_decodes_what_it_encodes_at_every_padding() {
        for n in 0..10 {
            let bytes: Vec<u8> = (0..n).map(|i| (i * 37 + 200) as u8).collect();
            assert_eq!(base64_lite::decode(&base64_lite::encode(&bytes)).unwrap(), bytes, "{n} bytes");
        }
        assert!(base64_lite::decode("not*base64").is_none());
    }

    #[test]
    fn basic_auth_is_encoded_as_base64() {
        assert_eq!(base64_lite::encode(b"ada:lovelace"), "YWRhOmxvdmVsYWNl");
        assert_eq!(base64_lite::encode(b"a"), "YQ==");
        assert_eq!(base64_lite::encode(b"ab"), "YWI=");
        assert_eq!(base64_lite::encode(b""), "");
    }

    /// A route is in one table or the other, never both, and never twice.
    #[test]
    fn every_route_is_listed_once() {
        let mut seen = std::collections::BTreeSet::new();
        for (method, path, _) in COMMANDS.iter().chain(NOT_COMMANDS) {
            assert!(seen.insert((method, path)), "{method} {path} is listed twice");
        }
    }
}
