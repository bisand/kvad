//! Reaching a kvad-serve from anywhere: Phase 1 of `docs/relay-plan.md`.
//!
//! The server side ([`Server`]) waits for paired devices and hands each
//! stream they open to kvad-serve's router, as if it had come in on its TCP
//! port. The client side ([`Client`], and [`blocking::Tunnel`] for the CLI)
//! dials a server by its [`Address`], `kvad://<node id>?k=…&relay=…`, and
//! opens a stream per HTTP connection.
//!
//! The transport is iroh: QUIC between the two machines, directly when a path
//! can be made, through a relay when it cannot, and moving between the two
//! while a connection is open. Nothing outside this crate names it, so that
//! replacing it is replacing this crate's insides (the plan's *build on iroh,
//! or write it*).
//!
//! What it holds to, from the plan's *Security*:
//!
//! - *Both ends are pinned.* A client checks that it reached the key the
//!   address names; the server refuses every key not in its allow list, at
//!   the handshake, before a byte of HTTP is read.
//! - *A knock.* The ALPN carries a secret from the server's knock file, so a
//!   connection that does not know it gets no answer, or one TLS alert before
//!   the server's key would have been sent.
//! - *Post-quantum only.* X25519MLKEM768 is the only key exchange offered, to
//!   peers and to relays alike.
//! - *No ports opened in the router.* iroh's port mapper is not compiled in.
//! - *Keys in owner-only files*, made with those permissions, never logged.
//!
//! What a tunnel does not do is decide who may do what. That is kvad-serve's
//! sign-in, which applies to a tunnelled request exactly as to any other; see
//! `kvad-serve`'s `config::Remote`.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{ErrorKind, Read, Write as _};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path as FsPath, PathBuf};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use iroh::endpoint::{
    presets, AfterHandshakeOutcome, Connection, EndpointHooks, IncomingAddr, PortmapperConfig, RecvStream, SendStream,
};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl, SecretKey, TransportAddr};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

pub mod blocking;

pub type Res<T> = Result<T, String>;

/// The ALPN, before the knock: the protocol and its version, so that a change
/// of protocol is refused at the handshake rather than misread after it.
const ALPN_PREFIX: &str = "kvad-tunnel/1/";

/// How many connections a server holds at once: a household's devices, each
/// with a command or two open, and then some.
const MAX_LIVE: usize = 32;

/// How long an address whose handshake was refused is ignored afterwards.
const BACKOFF: Duration = Duration::from_secs(30);

/// How long a client waits for a connection before saying so.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// What a server says first, on a stream of its own, once it has let a
/// device in. A client waits for it, so that a refusal is an error at
/// connect rather than a silence later, and a different version is named.
const HELLO: &str = "kvad-tunnel/1";

// ---------------------------------------------------------------------------
// Identities and secrets on disk
// ---------------------------------------------------------------------------

/// An Ed25519 identity, kept in a file only its owner can read.
pub struct Key {
    secret: SecretKey,
}

/// Shows whose key it is, never the key.
impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key({})", self.id().short())
    }
}

impl Key {
    /// The key in `path`, made on first use. The second value says whether it
    /// was made just now.
    pub fn load(path: &FsPath) -> Res<(Key, bool)> {
        let (bytes, made) = secret_file::<32>(path, || Ok(SecretKey::generate().to_bytes()))?;
        let mut bytes = bytes;
        let secret = SecretKey::from_bytes(&bytes);
        wipe(&mut bytes);
        Ok((Key { secret }, made))
    }

    pub fn id(&self) -> NodeId {
        NodeId(self.secret.public())
    }
}

/// The public half of a [`Key`]: what an [`Address`] names and what a
/// server's allow list holds. Written as 64 hex digits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(EndpointId);

impl NodeId {
    /// The first ten digits, for logs and status lines.
    pub fn short(&self) -> String {
        self.to_string()[..10].to_string()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.short())
    }
}

impl FromStr for NodeId {
    type Err = String;
    fn from_str(s: &str) -> Res<NodeId> {
        let s = s.trim();
        if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("`{s}` is not a node id: 64 hex digits"));
        }
        s.parse().map(NodeId).map_err(|_| format!("`{s}` is not a node id"))
    }
}

/// The server's knock: 32 hex digits from a file only its owner can read,
/// made on first use.
fn knock(path: &FsPath) -> Res<String> {
    let (bytes, _) = secret_file::<16>(path, random::<16>)?;
    Ok(hex(&bytes))
}

/// `N` bytes kept in `path`, made by `make` on first use: a directory only its
/// owner may write to, a file only its owner may read, created so that it
/// never exists with looser permissions even for a moment. A symlink is
/// refused, and the file checked is the file read.
fn secret_file<const N: usize>(path: &FsPath, make: impl FnOnce() -> Res<[u8; N]>) -> Res<([u8; N], bool)> {
    let shown = path.display();
    let me = unsafe { libc::geteuid() };
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).ok_or_else(|| format!("{shown}: needs a directory"))?;
    fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let d = fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if d.permissions().mode() & 0o022 != 0 || (d.uid() != me && d.uid() != 0) {
        return Err(format!("{}: others can write to it, so a secret in it cannot be trusted", dir.display()));
    }
    match fs::symlink_metadata(path) {
        Ok(m) => {
            if !m.file_type().is_file() {
                return Err(format!("{shown}: not a plain file"));
            }
            let mut f = fs::File::open(path).map_err(|e| format!("{shown}: {e}"))?;
            let opened = f.metadata().map_err(|e| format!("{shown}: {e}"))?;
            if (opened.dev(), opened.ino()) != (m.dev(), m.ino()) {
                return Err(format!("{shown}: changed while it was being opened"));
            }
            let mode = opened.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                return Err(format!("{shown}: others can read it (mode {mode:o}); chmod 600 it, or delete it to make a new one"));
            }
            if opened.uid() != me {
                return Err(format!("{shown}: not owned by this user"));
            }
            let mut bytes = [0u8; N];
            f.read_exact(&mut bytes).map_err(|e| format!("{shown}: {e}"))?;
            if f.read(&mut [0u8; 1]).map_err(|e| format!("{shown}: {e}"))? != 0 {
                return Err(format!("{shown}: longer than it should be"));
            }
            Ok((bytes, false))
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let mut bytes = make()?;
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)
                .map_err(|e| format!("{shown}: {e}"))?;
            let written = f.write_all(&bytes).and_then(|_| f.sync_all());
            if let Err(e) = written {
                wipe_n(&mut bytes);
                return Err(format!("{shown}: {e}"));
            }
            Ok((bytes, true))
        }
        Err(e) => Err(format!("{shown}: {e}")),
    }
}

/// Zeroes a copy of a key, in a way the optimiser may not drop.
fn wipe(b: &mut [u8; 32]) {
    wipe_n(b)
}

fn wipe_n<const N: usize>(b: &mut [u8; N]) {
    for x in b.iter_mut() {
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

fn random<const N: usize>() -> Res<[u8; N]> {
    let mut b = [0u8; N];
    fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(b)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Addresses and paths
// ---------------------------------------------------------------------------

/// Where a server is: its node id, its knock, its relays, and the direct
/// addresses it knows, as `kvad://<node id>?k=<knock>&relay=<url>&addr=<ip:port>`.
///
/// The knock makes an address worth keeping to oneself, as a server's URL
/// with a token in it would be. It is not what lets a device in: that is the
/// server's allow list, and then its sign-in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Address {
    pub node: NodeId,
    pub knock: String,
    pub relays: Vec<String>,
    pub addrs: Vec<SocketAddr>,
}

impl Address {
    /// `kvad://<node id>`, without the knock or the hints: what a credential
    /// is kept under, and what a status line shows.
    pub fn label(&self) -> String {
        format!("kvad://{}", self.node)
    }

    fn alpn(&self) -> Vec<u8> {
        alpn(&self.knock)
    }

    fn endpoint_addr(&self) -> Res<EndpointAddr> {
        let mut addrs = Vec::new();
        for r in &self.relays {
            addrs.push(TransportAddr::Relay(relay_url(r)?));
        }
        addrs.extend(self.addrs.iter().map(|a| TransportAddr::Ip(*a)));
        Ok(EndpointAddr::from_parts(self.node.0, addrs))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kvad://{}?k={}", self.node, self.knock)?;
        for r in &self.relays {
            write!(f, "&relay={r}")?;
        }
        for a in &self.addrs {
            write!(f, "&addr={a}")?;
        }
        Ok(())
    }
}

impl FromStr for Address {
    type Err = String;
    fn from_str(s: &str) -> Res<Address> {
        let s = s.trim();
        let rest = s.strip_prefix("kvad://").ok_or_else(|| format!("`{s}` is not a kvad:// address"))?;
        let (node, query) = rest.split_once('?').unwrap_or((rest, ""));
        let node: NodeId = node.trim_end_matches('/').parse()?;
        let (mut knock, mut relays, mut addrs) = (None, Vec::new(), Vec::new());
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            match pair.split_once('=') {
                Some(("k", k)) if k.len() == 32 && k.bytes().all(|c| c.is_ascii_hexdigit()) => knock = Some(k.to_ascii_lowercase()),
                Some(("k", _)) => return Err("the address's knock (k=) is not 32 hex digits".into()),
                Some(("relay", r)) => {
                    relay_url(r)?;
                    relays.push(r.to_string());
                }
                Some(("addr", a)) => addrs.push(a.parse().map_err(|_| format!("addr={a}: not an address and port"))?),
                _ => return Err(format!("`{pair}` is not part of a kvad:// address")),
            }
        }
        let knock = knock.ok_or("the address has no knock (k=): copy the whole address kvad-serve printed")?;
        if relays.is_empty() && addrs.is_empty() {
            return Err("the address names no relay and no address, so there is nowhere to dial".into());
        }
        Ok(Address { node, knock, relays, addrs })
    }
}

/// A relay must be HTTPS. Plain HTTP only to this machine, for tests.
fn relay_url(s: &str) -> Res<RelayUrl> {
    if let Some(rest) = s.strip_prefix("http://") {
        let authority = rest.split('/').next().unwrap_or("");
        let host = authority.rsplit_once(':').map_or(authority, |(h, _)| h);
        if !matches!(host, "localhost" | "127.0.0.1" | "[::1]") {
            return Err(format!("{s}: a relay must be https://; plain http only to this machine"));
        }
    } else if !s.starts_with("https://") {
        return Err(format!("{s}: a relay must be https://"));
    }
    s.parse().map_err(|e| format!("{s}: {e}"))
}

fn alpn(knock: &str) -> Vec<u8> {
    format!("{ALPN_PREFIX}{knock}").into_bytes()
}

/// Which of the plan's paths a connection is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Path {
    /// The same network: a private, link-local or loopback address.
    Lan(SocketAddr),
    /// Across the internet, straight to the other machine.
    Direct(SocketAddr),
    /// Through a relay.
    Relay(String),
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Path::Lan(a) => write!(f, "lan {a}"),
            Path::Direct(a) => write!(f, "direct {a}"),
            Path::Relay(u) => write!(f, "relay {u}"),
        }
    }
}

fn selected(conn: &Connection) -> Option<Path> {
    let paths = conn.paths();
    let found = paths.iter().find(|p| p.is_selected()).and_then(|p| describe(p.remote_addr()));
    found
}

fn describe(a: &TransportAddr) -> Option<Path> {
    match a {
        TransportAddr::Relay(u) => Some(Path::Relay(u.to_string().trim_end_matches('/').to_string())),
        TransportAddr::Ip(s) if is_lan(s.ip()) => Some(Path::Lan(*s)),
        TransportAddr::Ip(s) => Some(Path::Direct(*s)),
        _ => None,
    }
}

fn is_lan(ip: IpAddr) -> bool {
    match ip.to_canonical() {
        IpAddr::V4(a) => a.is_private() || a.is_link_local() || a.is_loopback(),
        IpAddr::V6(a) => {
            let s = a.segments()[0];
            a.is_loopback() || s & 0xfe00 == 0xfc00 || s & 0xffc0 == 0xfe80
        }
    }
}

// ---------------------------------------------------------------------------
// The endpoint
// ---------------------------------------------------------------------------

/// Hybrid post-quantum key exchange, and nothing else: both ends of a peer
/// connection are this code, and kvad-relay speaks it too, so nothing needs a
/// classical fallback that an attacker could steer towards.
fn crypto() -> Arc<rustls::crypto::CryptoProvider> {
    use rustls::crypto::aws_lc_rs::{self, kx_group};
    let mut p = aws_lc_rs::default_provider();
    p.kx_groups = vec![kx_group::X25519MLKEM768];
    Arc::new(p)
}

/// Refuses every peer not on the list, once the handshake has proved which
/// key it holds and before any stream.
#[derive(Debug)]
struct Allow(Arc<Vec<NodeId>>);

impl EndpointHooks for Allow {
    fn after_handshake<'a>(&'a self, conn: &'a Connection) -> impl std::future::Future<Output = AfterHandshakeOutcome> + Send + 'a {
        let ok = self.0.contains(&NodeId(conn.remote_id()));
        async move {
            match ok {
                true => AfterHandshakeOutcome::accept(),
                false => AfterHandshakeOutcome::Reject { error_code: 403u32.into(), reason: Vec::new() },
            }
        }
    }
}

/// Whether a server takes direct connections, or only relayed ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direct {
    /// Every path: the LAN, a direct path across the internet, the relay.
    Auto,
    /// The relay only. No UDP socket is opened, so nothing on this machine
    /// answers the internet except through the relay.
    Off,
}

async fn endpoint(secret: SecretKey, relays: &[String], alpns: Vec<Vec<u8>>, direct: Direct, port: u16, allow: Option<Allow>) -> Res<Endpoint> {
    let relays = relays.iter().map(|r| relay_url(r)).collect::<Res<Vec<_>>>()?;
    let mode = match relays.is_empty() {
        true => RelayMode::Disabled,
        false => RelayMode::custom(relays),
    };
    if direct == Direct::Off && matches!(mode, RelayMode::Disabled) {
        return Err("direct = \"off\" needs a relay: with neither, nothing can reach this machine".into());
    }
    let mut b = Endpoint::builder(presets::Empty)
        .crypto_provider(crypto())
        .secret_key(secret)
        .alpns(alpns)
        .relay_mode(mode)
        .portmapper_config(PortmapperConfig::Disabled);
    match direct {
        Direct::Off => b = b.clear_ip_transports(),
        Direct::Auto if port != 0 => {
            let v4 = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
            let v6 = SocketAddr::from((Ipv6Addr::UNSPECIFIED, port));
            b = b.clear_ip_transports().bind_addr(v4).map_err(|e| e.to_string())?.bind_addr(v6).map_err(|e| e.to_string())?;
        }
        Direct::Auto => {}
    }
    if let Some(allow) = allow {
        b = b.hooks(allow);
    }
    b.bind().await.map_err(|e| format!("starting the tunnel: {e}"))
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// One stream through the tunnel: what one HTTP connection runs over.
pub struct Stream {
    recv: RecvStream,
    send: SendStream,
}

impl AsyncRead for Stream {
    fn poll_read(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<std::io::Result<()>> {
        AsyncRead::poll_read(Pin::new(&mut self.recv), cx, buf)
    }
}

impl AsyncWrite for Stream {
    fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<std::io::Result<usize>> {
        AsyncWrite::poll_write(Pin::new(&mut self.send), cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        AsyncWrite::poll_flush(Pin::new(&mut self.send), cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        AsyncWrite::poll_shutdown(Pin::new(&mut self.send), cx)
    }
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

/// What a server is told by kvad-serve's `[remote]` section.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// `remote.key`, in kvad-serve's data directory.
    pub key: PathBuf,
    /// `remote.knock`, beside it.
    pub knock: PathBuf,
    pub relays: Vec<String>,
    pub direct: Direct,
    /// The UDP port for direct connections; fixed, so that it can be
    /// forwarded by hand. 0 picks one.
    pub port: u16,
    /// The devices that may connect. Phase 2's pairing will keep these in
    /// kvad-serve's database; until then they are written in the config.
    pub allow: Vec<NodeId>,
}

/// A server's side of the tunnel.
pub struct Server {
    endpoint: Endpoint,
    address: Address,
    streams: tokio::sync::mpsc::Receiver<(NodeId, Stream)>,
    accepting: tokio::task::JoinHandle<()>,
}

impl Server {
    pub async fn start(cfg: &ServerConfig) -> Res<Server> {
        if cfg.allow.is_empty() {
            return Err("[remote] allow is empty, so no device could connect: add the id `kvad remote id` prints on each".into());
        }
        let (key, _) = Key::load(&cfg.key)?;
        let node = key.id();
        let knock = knock(&cfg.knock)?;
        let alpn = alpn(&knock);
        let allow = Arc::new(cfg.allow.clone());
        let endpoint = endpoint(key.secret, &cfg.relays, vec![alpn.clone()], cfg.direct, cfg.port, Some(Allow(allow))).await?;
        if !cfg.relays.is_empty() && tokio::time::timeout(Duration::from_secs(10), endpoint.online()).await.is_err() {
            tracing::warn!("no relay answered within 10 s; the address carries direct addresses only until one does");
        }
        let addrs = match cfg.direct {
            Direct::Off => Vec::new(),
            Direct::Auto => endpoint.addr().ip_addrs().copied().collect(),
        };
        let address = Address { node, knock, relays: cfg.relays.clone(), addrs };
        let (tx, streams) = tokio::sync::mpsc::channel(64);
        let accepting = tokio::spawn(accept_loop(endpoint.clone(), alpn, tx));
        Ok(Server { endpoint, address, streams, accepting })
    }

    /// What a client dials.
    pub fn address(&self) -> &Address {
        &self.address
    }

    /// The next stream a device opened, with the device it came from. `None`
    /// once the server is closed.
    pub async fn accept(&mut self) -> Option<(NodeId, Stream)> {
        self.streams.recv().await
    }

    pub async fn close(self) {
        self.closer().close().await
    }

    /// A handle that closes the server after it has been handed to
    /// `axum::serve`, which keeps the server itself. Closing gives the UDP
    /// port back, which a restart in the same process needs.
    pub fn closer(&self) -> Closer {
        Closer { endpoint: self.endpoint.clone(), accepting: self.accepting.abort_handle() }
    }
}

/// See [`Server::closer`].
pub struct Closer {
    endpoint: Endpoint,
    accepting: tokio::task::AbortHandle,
}

impl Closer {
    pub async fn close(self) {
        self.accepting.abort();
        self.endpoint.close().await;
    }
}

/// The tunnel as one more listener for an axum router: `axum::serve(server,
/// app)` serves it as it serves a TCP listener. What a connection is "from"
/// is the device's node id; nothing in it is an IP address, so nothing can
/// mistake a tunnelled request for one from loopback.
#[cfg(feature = "axum")]
impl axum::serve::Listener for Server {
    type Io = Stream;
    type Addr = NodeId;

    async fn accept(&mut self) -> (Stream, NodeId) {
        match self.streams.recv().await {
            Some((peer, stream)) => (stream, peer),
            // Closed: there will be no more, and axum's loop has no way to be
            // told so other than its own shutdown signal.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> std::io::Result<NodeId> {
        Ok(self.address.node)
    }
}

async fn accept_loop(endpoint: Endpoint, alpn: Vec<u8>, tx: tokio::sync::mpsc::Sender<(NodeId, Stream)>) {
    let live = Arc::new(AtomicUsize::new(0));
    let refused: Arc<Mutex<HashMap<IpAddr, Instant>>> = Arc::default();
    while let Some(incoming) = endpoint.accept().await {
        let ip = match incoming.remote_addr() {
            IncomingAddr::Ip(a) => Some(a.ip().to_canonical()),
            _ => None,
        };
        // The knock. When the first packet shows the ALPNs and ours is not
        // among them, nothing is sent back. When it does not, which is usual
        // with the post-quantum key share, rustls refuses a wrong ALPN while
        // reading the ClientHello, before the server's key would be sent.
        let wrong = incoming.decrypt().and_then(|d| d.alpns()).is_some_and(|mut it| !it.any(|a| a.is_ok_and(|a| a[..] == alpn[..])));
        let backing_off = ip.is_some_and(|ip| refused.lock().unwrap().get(&ip).is_some_and(|t| t.elapsed() < BACKOFF));
        if wrong || backing_off || live.load(Ordering::SeqCst) >= MAX_LIVE {
            incoming.ignore();
            continue;
        }
        let (live, refused, tx) = (live.clone(), refused.clone(), tx.clone());
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    if let Some(ip) = ip {
                        refused.lock().unwrap().insert(ip, Instant::now());
                    }
                    tracing::info!("remote: refused a connection: {e}");
                    return;
                }
            };
            live.fetch_add(1, Ordering::SeqCst);
            let peer = NodeId(conn.remote_id());
            tracing::info!("remote: {} connected", peer.short());
            if let Ok(mut hello) = conn.open_uni().await {
                let _ = hello.write_all(format!("{HELLO}\n").as_bytes()).await;
                let _ = hello.finish();
            }
            let watching = tokio::spawn(watch(conn.clone(), peer));
            loop {
                match conn.accept_bi().await {
                    Ok((send, recv)) => {
                        if tx.send((peer, Stream { recv, send })).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            watching.abort();
            live.fetch_sub(1, Ordering::SeqCst);
            tracing::info!("remote: {} gone", peer.short());
        });
    }
}

/// Says in the log which path a device is on, whenever it changes.
async fn watch(conn: Connection, peer: NodeId) {
    let mut last = None;
    loop {
        let now = selected(&conn);
        if now != last {
            if let Some(path) = &now {
                tracing::info!("remote: {} on {path}", peer.short());
            }
            last = now;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

/// A device's side of the tunnel: one connection to one server.
pub struct Client {
    endpoint: Endpoint,
    conn: Connection,
}

impl Client {
    /// Dials `to` as `key`, and checks that the far end holds the key the
    /// address names.
    pub async fn connect(key: &Key, to: &Address) -> Res<Client> {
        // No ALPN of its own: a client accepts nothing.
        let endpoint = endpoint(key.secret.clone(), &to.relays, Vec::new(), Direct::Auto, 0, None).await?;
        let addr = to.endpoint_addr()?;
        let conn = match tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(addr, &to.alpn())).await {
            Err(_) => {
                return Err(format!(
                    "{} did not answer within {} s: it may be off or asleep, or the address may be out of date",
                    to.label(),
                    CONNECT_TIMEOUT.as_secs()
                ))
            }
            Ok(Err(e)) => return Err(format!("could not reach {}: {e}", to.label())),
            Ok(Ok(c)) => c,
        };
        if NodeId(conn.remote_id()) != to.node {
            return Err(format!("reached a key other than {}'s", to.label()));
        }
        let hello = tokio::time::timeout(CONNECT_TIMEOUT, async {
            let mut s = conn.accept_uni().await.map_err(|e| refused(&e.to_string()))?;
            let said = s.read_to_end(64).await.map_err(|e| refused(&e.to_string()))?;
            Ok::<_, String>(String::from_utf8_lossy(&said).trim().to_string())
        });
        match hello.await {
            Err(_) => return Err(format!("{} let the connection in but said nothing", to.label())),
            Ok(Err(e)) => return Err(e),
            Ok(Ok(said)) if said != HELLO => {
                return Err(format!("{} speaks {said}, and this kvad speaks {HELLO}", to.label()));
            }
            Ok(Ok(_)) => {}
        }
        Ok(Client { endpoint, conn })
    }

    /// A new stream, for one HTTP connection. The server refuses streams from
    /// a key it does not know by closing the connection, which shows here.
    pub async fn open(&self) -> Res<Stream> {
        let (send, recv) = self.conn.open_bi().await.map_err(|e| refused(&e.to_string()))?;
        Ok(Stream { recv, send })
    }

    /// The path the connection is on now.
    pub fn path(&self) -> Option<Path> {
        selected(&self.conn)
    }

    pub async fn close(&self) {
        self.conn.close(0u32.into(), b"");
        self.endpoint.close().await;
    }
}

fn refused(e: &str) -> String {
    match e.contains("403") || e.contains("rejected") {
        true => "the server refused this device: add its id (`kvad remote id`) to the server's [remote] allow".into(),
        false => format!("the tunnel closed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node() -> NodeId {
        Key { secret: SecretKey::from_bytes(&[7; 32]) }.id()
    }

    #[test]
    fn an_address_reads_back_as_it_was_written() {
        let a = Address {
            node: node(),
            knock: "0123456789abcdef0123456789abcdef".into(),
            relays: vec!["https://relay.kvad.eu".into()],
            addrs: vec!["192.168.1.20:5824".parse().unwrap(), "[2001:db8::20]:5824".parse().unwrap()],
        };
        let back: Address = a.to_string().parse().unwrap();
        assert_eq!(back, a);
        assert_eq!(a.label(), format!("kvad://{}", node()));
    }

    #[test]
    fn an_address_is_refused_when_it_cannot_be_dialled() {
        let n = node();
        let k = "0123456789abcdef0123456789abcdef";
        assert!(format!("kvad://{n}").parse::<Address>().is_err(), "no knock");
        assert!(format!("kvad://{n}?k={k}").parse::<Address>().is_err(), "nowhere to dial");
        assert!(format!("kvad://{n}?k=abc&relay=https://r.example").parse::<Address>().is_err(), "short knock");
        assert!(format!("kvad://{n}?k={k}&relay=http://r.example").parse::<Address>().is_err(), "plain http off loopback");
        assert!(format!("kvad://{n}?k={k}&relay=http://localhost.evil.example:80").parse::<Address>().is_err());
        assert!(format!("kvad://{n}?k={k}&nope=1&relay=https://r.example").parse::<Address>().is_err(), "unknown key");
        assert!(format!("http://{n}?k={k}&relay=https://r.example").parse::<Address>().is_err(), "scheme");
        assert!(format!("kvad://{n}?k={k}&relay=https://r.example").parse::<Address>().is_ok());
    }

    #[test]
    fn a_node_id_is_64_hex_digits() {
        assert!("abc".parse::<NodeId>().is_err());
        assert_eq!(node().to_string().parse::<NodeId>().unwrap(), node());
        assert_eq!(node().short().len(), 10);
    }

    #[test]
    fn paths_are_named_by_where_they_go() {
        assert!(matches!(describe(&TransportAddr::Ip("192.168.1.2:1".parse().unwrap())), Some(Path::Lan(_))));
        assert!(matches!(describe(&TransportAddr::Ip("[fe80::1]:1".parse().unwrap())), Some(Path::Lan(_))));
        assert!(matches!(describe(&TransportAddr::Ip("85.137.228.199:1".parse().unwrap())), Some(Path::Direct(_))));
    }

    #[test]
    fn secrets_are_kept_owner_only_and_checked() {
        let dir = std::env::temp_dir().join(format!("kvad-tunnel-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("remote.key");
        let (a, made) = Key::load(&path).unwrap();
        assert!(made);
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        let (b, made) = Key::load(&path).unwrap();
        assert!(!made);
        assert_eq!(a.id(), b.id());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(Key::load(&path).unwrap_err().contains("others can read"));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let link = dir.join("link.key");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(Key::load(&link).unwrap_err().contains("not a plain file"));
        let k1 = knock(&dir.join("remote.knock")).unwrap();
        assert_eq!(k1.len(), 32);
        assert_eq!(knock(&dir.join("remote.knock")).unwrap(), k1);
        fs::remove_dir_all(&dir).unwrap();
    }
}
