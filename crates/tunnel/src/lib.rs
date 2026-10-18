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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::{ErrorKind, Read, Write as _};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path as FsPath, PathBuf};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use iroh::endpoint::{
    presets, AfterHandshakeOutcome, Connection, EndpointHooks, IncomingAddr, PortmapperConfig, RecvStream, SendStream,
};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMap, RelayMode, RelayUrl, SecretKey, TransportAddr};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

pub mod blocking;

pub type Res<T> = Result<T, String>;

/// The ALPN, before the knock: the protocol and its version, so that a change
/// of protocol is refused at the handshake rather than misread after it.
const ALPN_PREFIX: &str = "kvad-tunnel/1/";

/// How many connections a server holds at once: a household's devices, each
/// with a command or two open, and then some.
const MAX_LIVE: usize = 32;

/// An address refused this many times within [`BACKOFF`] is ignored until
/// the window has passed: a device that tried before its pairing was open
/// gets its retry, and something trying keys does not get many.
const REFUSALS: u32 = 3;
const BACKOFF: Duration = Duration::from_secs(60);

/// How long a client waits for a connection before saying so.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// What a server says first, on a stream of its own, once it has let a
/// device in. A client waits for it, so that a refusal is an error at
/// connect rather than a silence later, and a different version is named.
const HELLO: &str = "kvad-tunnel/1";

/// What a server says first to a key it does not know, while a pairing is
/// open: the connection is for one pairing request and its answer, and for
/// nothing else, HTTP least of all.
const PAIRING_HELLO: &str = "kvad-tunnel/1 pairing";

/// How large a pairing request or answer may be.
const PAIRING_LIMIT: usize = 16 * 1024;

/// How long a pairing connection may take, start to finish.
const PAIRING_TIMEOUT: Duration = Duration::from_secs(30);

/// How many pairing connections a server holds at once.
const MAX_PAIRING: usize = 4;

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

    /// Vouchers signed with this key.
    pub fn issuer(&self) -> Issuer {
        Issuer(Arc::new(self.secret.clone()))
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
// Vouchers
// ---------------------------------------------------------------------------

/// How long a paired device's voucher is good for. A device renews it
/// through the tunnel once less than half is left, so one in use never runs
/// out; one that is not, and was unpaired, stops at the relay within this.
pub const VOUCHER_LIFE: Duration = Duration::from_secs(30 * 24 * 3600);

/// The longest a voucher for any device may live: the one a pairing link
/// carries, so that a device can pair through the relay. Past this a relay
/// refuses it, however it was signed.
pub const ANY_DEVICE_LIFE: Duration = Duration::from_secs(15 * 60);

/// A server's word that a device may use the relays that trust the server:
/// what a device shows a relay, which has no list of devices of its own.
///
/// Signed with the server's node key, so a relay needs only the node ids it
/// serves to check one, keeps no state, and cannot be told anything by a
/// device that the server did not say. `kv1.<node>.<device or *>.<expires,
/// unix seconds>.<signature>`; `*` is any device, and only for a pairing
/// link's few minutes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Voucher {
    pub node: NodeId,
    pub device: Option<NodeId>,
    pub expires: u64,
    signature: [u8; 64],
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Voucher {
    fn message(node: &NodeId, device: Option<&NodeId>, expires: u64) -> Vec<u8> {
        let device = device.map_or("*".to_string(), |d| d.to_string());
        format!("kvad-voucher/1\n{node}\n{device}\n{expires}\n").into_bytes()
    }

    /// Whether this voucher lets `device` in, for a relay that trusts the
    /// nodes `trusted` says it does.
    pub fn check(&self, device: &NodeId, trusted: impl Fn(&NodeId) -> bool) -> Res<()> {
        let now = unix_now();
        if !trusted(&self.node) {
            return Err(format!("signed by {}, which this relay does not serve", self.node.short()));
        }
        if self.expires <= now {
            return Err("expired".into());
        }
        match &self.device {
            Some(d) if d != device => return Err("for another device".into()),
            None if self.expires > now + ANY_DEVICE_LIFE.as_secs() => {
                return Err("for any device and for too long".into());
            }
            _ => {}
        }
        let message = Voucher::message(&self.node, self.device.as_ref(), self.expires);
        let signature = iroh::Signature::from_bytes(&self.signature);
        self.node.0.verify(&message, &signature).map_err(|_| "not signed by the node it names".to_string())
    }

    /// How long until it expires.
    pub fn left(&self) -> Duration {
        Duration::from_secs(self.expires.saturating_sub(unix_now()))
    }
}

impl fmt::Display for Voucher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = self.device.map_or("*".to_string(), |d| d.to_string());
        write!(f, "kv1.{}.{device}.{}.{}", self.node, self.expires, hex(&self.signature))
    }
}

impl FromStr for Voucher {
    type Err = String;
    fn from_str(s: &str) -> Res<Voucher> {
        let bad = || "not a voucher".to_string();
        let [version, node, device, expires, signature] = s.trim().split('.').collect::<Vec<_>>()[..] else { return Err(bad()) };
        if version != "kv1" || signature.len() != 128 || !signature.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(bad());
        }
        let mut sig = [0u8; 64];
        for (i, b) in sig.iter_mut().enumerate() {
            *b = u8::from_str_radix(&signature[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        Ok(Voucher {
            node: node.parse()?,
            device: match device {
                "*" => None,
                d => Some(d.parse()?),
            },
            expires: expires.parse().map_err(|_| bad())?,
            signature: sig,
        })
    }
}

/// What a server signs vouchers with: its node key, kept out of every log.
#[derive(Clone)]
pub struct Issuer(Arc<SecretKey>);

impl Issuer {
    /// A voucher for `device`, or for any device with `None`, good for
    /// `life` (at most [`ANY_DEVICE_LIFE`] for any device).
    pub fn issue(&self, device: Option<NodeId>, life: Duration) -> Voucher {
        let life = match device {
            None => life.min(ANY_DEVICE_LIFE),
            Some(_) => life,
        };
        let node = NodeId(self.0.public());
        let expires = unix_now() + life.as_secs();
        let signature = self.0.sign(&Voucher::message(&node, device.as_ref(), expires)).to_bytes();
        Voucher { node, device, expires, signature }
    }
}

impl fmt::Debug for Issuer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Issuer({})", NodeId(self.0.public()).short())
    }
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

/// A pairing link: a server's [`Address`] and a single-use pairing token,
/// `kvad://<node id>?k=…&relay=…&addr=…&pair=<token>`.
///
/// It is shown on the server's own screen, as a QR code or as text, and is
/// worth as much as a password for the few minutes it lives: whoever presents
/// the token first is paired. (A link sent somewhere less trusted wants the
/// code and the PAKE of `docs/relay-plan.md`, which are not built yet.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairLink {
    pub address: Address,
    pub token: String,
    /// A voucher for any device, good for the link's few minutes, so that a
    /// device can pair through the relay as well as on the LAN.
    pub voucher: Option<Voucher>,
}

impl fmt::Display for PairLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.address)?;
        if let Some(v) = &self.voucher {
            write!(f, "&rv={v}")?;
        }
        write!(f, "&pair={}", self.token)
    }
}

impl FromStr for PairLink {
    type Err = String;
    fn from_str(s: &str) -> Res<PairLink> {
        let s = s.trim();
        let (rest, token) = match s.rsplit_once("&pair=").or_else(|| s.rsplit_once("?pair=")) {
            Some((rest, token)) => (rest, token),
            None => return Err("that is a server's address, not a pairing link: it has no pair= token".into()),
        };
        if token.len() < 32 || token.len() > 128 || !token.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_') {
            return Err("the link's pairing token is not one a server makes".into());
        }
        let (rest, voucher) = match rest.rsplit_once("&rv=") {
            Some((rest, v)) => (rest, Some(v.parse()?)),
            None => (rest, None),
        };
        Ok(PairLink { address: rest.parse()?, token: token.to_string(), voucher })
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

/// The devices a server lets in, shared between its tunnel and whatever
/// keeps the list (kvad-serve's `devices` table, and `[remote] allow`), so
/// that a device paired or removed is let in or shut out at once.
///
/// While a pairing is open, a key not on the list is let in too, but only to
/// the pairing lane: one request, one answer, then the connection is closed.
#[derive(Clone, Default, Debug)]
pub struct Allowed(Arc<AllowedInner>);

#[derive(Default, Debug)]
struct AllowedInner {
    ids: RwLock<HashSet<NodeId>>,
    pairing: AtomicUsize,
}

impl Allowed {
    pub fn new(ids: impl IntoIterator<Item = NodeId>) -> Allowed {
        let a = Allowed::default();
        a.set(ids);
        a
    }

    pub fn set(&self, ids: impl IntoIterator<Item = NodeId>) {
        *self.0.ids.write().unwrap() = ids.into_iter().collect();
    }

    pub fn insert(&self, id: NodeId) {
        self.0.ids.write().unwrap().insert(id);
    }

    /// Takes a device off the list. Its live connection, if it has one, is
    /// closed within a quarter of a second.
    pub fn remove(&self, id: &NodeId) -> bool {
        self.0.ids.write().unwrap().remove(id)
    }

    pub fn contains(&self, id: &NodeId) -> bool {
        self.0.ids.read().unwrap().contains(id)
    }

    pub fn len(&self) -> usize {
        self.0.ids.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Says that a pairing slot is open, so that an unknown key may reach the
    /// pairing lane. Each call is undone by one [`Allowed::close_pairing`].
    pub fn open_pairing(&self) {
        self.0.pairing.fetch_add(1, Ordering::SeqCst);
    }

    pub fn close_pairing(&self) {
        let _ = self.0.pairing.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
    }

    pub fn pairing(&self) -> bool {
        self.0.pairing.load(Ordering::SeqCst) > 0
    }
}

/// Refuses every peer not on the list, once the handshake has proved which
/// key it holds and before any stream; an unknown one gets in only while a
/// pairing is open, and then only to the pairing lane.
#[derive(Debug)]
struct Allow(Allowed);

impl EndpointHooks for Allow {
    fn after_handshake<'a>(&'a self, conn: &'a Connection) -> impl std::future::Future<Output = AfterHandshakeOutcome> + Send + 'a {
        let ok = self.0.contains(&NodeId(conn.remote_id())) || self.0.pairing();
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

async fn endpoint(
    secret: SecretKey,
    relays: &[String],
    alpns: Vec<Vec<u8>>,
    direct: Direct,
    port: u16,
    allow: Option<Allow>,
    voucher: Option<&Voucher>,
) -> Res<Endpoint> {
    let relays = relays.iter().map(|r| relay_url(r)).collect::<Res<Vec<_>>>()?;
    let mode = match (relays.is_empty(), voucher) {
        (true, _) => RelayMode::Disabled,
        (false, None) => RelayMode::custom(relays),
        // Shown to every relay as it is connected to: what lets in a device
        // the relay has no list of.
        (false, Some(v)) => RelayMode::Custom(RelayMap::from_iter(relays).with_auth_token(v.to_string())),
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
    /// The devices that may connect, kept up to date by the caller.
    pub allowed: Allowed,
}

/// A pairing request from a key the server does not know, delivered by
/// [`Server::take_pairings`]. Whoever answers it decides whether the device
/// is paired; the tunnel only carries the bytes.
pub struct Pairing {
    /// The key that asked, proved by the handshake.
    pub device: NodeId,
    pub request: Vec<u8>,
    reply: tokio::sync::oneshot::Sender<Vec<u8>>,
}

impl Pairing {
    pub fn answer(self, reply: Vec<u8>) {
        let _ = self.reply.send(reply);
    }
}

/// A server's side of the tunnel.
pub struct Server {
    endpoint: Endpoint,
    address: Address,
    streams: tokio::sync::mpsc::Receiver<(NodeId, Stream)>,
    pairings: Option<tokio::sync::mpsc::Receiver<Pairing>>,
    accepting: tokio::task::JoinHandle<()>,
    issuer: Issuer,
}

impl Server {
    pub async fn start(cfg: &ServerConfig) -> Res<Server> {
        let (key, _) = Key::load(&cfg.key)?;
        let node = key.id();
        let knock = knock(&cfg.knock)?;
        let alpn = alpn(&knock);
        let allow = Allow(cfg.allowed.clone());
        let issuer = Issuer(Arc::new(key.secret.clone()));
        let endpoint = endpoint(key.secret, &cfg.relays, vec![alpn.clone()], cfg.direct, cfg.port, Some(allow), None).await?;
        if !cfg.relays.is_empty() && tokio::time::timeout(Duration::from_secs(10), endpoint.online()).await.is_err() {
            tracing::warn!("no relay answered within 10 s; the address carries direct addresses only until one does");
        }
        let addrs = match cfg.direct {
            Direct::Off => Vec::new(),
            Direct::Auto => endpoint.addr().ip_addrs().copied().collect(),
        };
        let address = Address { node, knock, relays: cfg.relays.clone(), addrs };
        let (tx, streams) = tokio::sync::mpsc::channel(64);
        let (pair_tx, pairings) = tokio::sync::mpsc::channel(MAX_PAIRING);
        let accepting = tokio::spawn(accept_loop(endpoint.clone(), alpn, cfg.allowed.clone(), tx, pair_tx));
        Ok(Server { endpoint, address, streams, pairings: Some(pairings), accepting, issuer })
    }

    /// The pairing requests that unknown keys send while a pairing is open.
    /// Taken once, by whoever decides them; a request nobody is taking is
    /// answered with a refusal.
    pub fn take_pairings(&mut self) -> Option<tokio::sync::mpsc::Receiver<Pairing>> {
        self.pairings.take()
    }

    /// What this server signs vouchers with.
    pub fn issuer(&self) -> Issuer {
        self.issuer.clone()
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

async fn accept_loop(
    endpoint: Endpoint,
    alpn: Vec<u8>,
    allowed: Allowed,
    tx: tokio::sync::mpsc::Sender<(NodeId, Stream)>,
    pairings: tokio::sync::mpsc::Sender<Pairing>,
) {
    let live = Arc::new(AtomicUsize::new(0));
    let pairing_live = Arc::new(AtomicUsize::new(0));
    let refused: Arc<Mutex<HashMap<IpAddr, (u32, Instant)>>> = Arc::default();
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
        let backing_off = ip.is_some_and(|ip| refused.lock().unwrap().get(&ip).is_some_and(|(n, since)| *n >= REFUSALS && since.elapsed() < BACKOFF));
        if wrong || backing_off || live.load(Ordering::SeqCst) >= MAX_LIVE {
            incoming.ignore();
            continue;
        }
        let (live, pairing_live, refused, tx, pairings, allowed) =
            (live.clone(), pairing_live.clone(), refused.clone(), tx.clone(), pairings.clone(), allowed.clone());
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    if let Some(ip) = ip {
                        refuse(&refused, ip);
                    }
                    tracing::info!("remote: refused a connection: {e}");
                    return;
                }
            };
            let peer = NodeId(conn.remote_id());
            if !allowed.contains(&peer) {
                // Let in by the hook only because a pairing is open.
                if pairing_live.fetch_add(1, Ordering::SeqCst) >= MAX_PAIRING {
                    conn.close(429u32.into(), b"");
                } else {
                    tracing::info!("remote: {} asks to pair", peer.short());
                    let outcome = tokio::time::timeout(PAIRING_TIMEOUT, pair_lane(&conn, peer, &pairings)).await;
                    if !matches!(outcome, Ok(Ok(()))) {
                        if let Some(ip) = ip {
                            refuse(&refused, ip);
                        }
                    }
                    // Long enough for the answer to arrive before the close.
                    let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
                    conn.close(0u32.into(), b"");
                }
                pairing_live.fetch_sub(1, Ordering::SeqCst);
                return;
            }
            live.fetch_add(1, Ordering::SeqCst);
            tracing::info!("remote: {} connected", peer.short());
            if let Ok(mut hello) = conn.open_uni().await {
                let _ = hello.write_all(format!("{HELLO}\n").as_bytes()).await;
                let _ = hello.finish();
            }
            let watching = tokio::spawn(watch(conn.clone(), peer, allowed));
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

/// Counts a refusal from `ip`, in a window that starts with the first.
fn refuse(refused: &Mutex<HashMap<IpAddr, (u32, Instant)>>, ip: IpAddr) {
    let mut all = refused.lock().unwrap();
    let entry = all.entry(ip).or_insert((0, Instant::now()));
    if entry.1.elapsed() >= BACKOFF {
        *entry = (0, Instant::now());
    }
    entry.0 += 1;
}

/// One pairing request from a key the server does not know, and its answer:
/// all an unpaired device can ever do.
async fn pair_lane(conn: &Connection, peer: NodeId, pairings: &tokio::sync::mpsc::Sender<Pairing>) -> Res<()> {
    let mut hello = conn.open_uni().await.map_err(|e| e.to_string())?;
    hello.write_all(format!("{PAIRING_HELLO}\n").as_bytes()).await.map_err(|e| e.to_string())?;
    let _ = hello.finish();
    let (mut send, mut recv) = conn.accept_bi().await.map_err(|e| e.to_string())?;
    let request = recv.read_to_end(PAIRING_LIMIT).await.map_err(|e| e.to_string())?;
    let (reply, answer) = tokio::sync::oneshot::channel();
    let refusal = br#"{"error":"this server is not taking pairings now"}"#.to_vec();
    let answer = match pairings.try_send(Pairing { device: peer, request, reply }) {
        Ok(()) => answer.await.unwrap_or(refusal),
        Err(_) => refusal,
    };
    send.write_all(&answer).await.map_err(|e| e.to_string())?;
    send.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Says in the log which path a device is on, whenever it changes, and
/// closes the connection once the device is no longer allowed.
async fn watch(conn: Connection, peer: NodeId, allowed: Allowed) {
    let mut last = None;
    loop {
        if !allowed.contains(&peer) {
            tracing::info!("remote: {} removed, closing its connection", peer.short());
            conn.close(403u32.into(), b"revoked");
            return;
        }
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
    /// address names and has let this device in.
    pub async fn connect(key: &Key, to: &Address) -> Res<Client> {
        Client::connect_vouched(key, to, None).await
    }

    /// The same, showing the relays `voucher`: how a paired device that is
    /// on no relay's list gets through one.
    pub async fn connect_vouched(key: &Key, to: &Address, voucher: Option<&Voucher>) -> Res<Client> {
        let (endpoint, conn, hello) = dial(key, to, voucher).await?;
        match hello.as_str() {
            HELLO => Ok(Client { endpoint, conn }),
            PAIRING_HELLO => {
                endpoint.close().await;
                Err(format!("this device is not paired with {}: `kvad remote join` with a pairing link from it", to.label()))
            }
            other => {
                endpoint.close().await;
                Err(format!("{} speaks {other}, and this kvad speaks {HELLO}", to.label()))
            }
        }
    }

    /// Sends one pairing request to `to` as `key`, which the server does not
    /// know yet, and returns its answer. The server lets an unknown key this
    /// far only while one of its pairings is open.
    pub async fn pair(key: &Key, to: &Address, request: &[u8], voucher: Option<&Voucher>) -> Res<Vec<u8>> {
        let (endpoint, conn, hello) = dial(key, to, voucher).await?;
        let answer = async {
            match hello.as_str() {
                PAIRING_HELLO => {}
                HELLO => return Err(format!("this device is already paired with {}", to.label())),
                other => return Err(format!("{} speaks {other}, and this kvad speaks {HELLO}", to.label())),
            }
            let (mut send, mut recv) = conn.open_bi().await.map_err(|e| refused(&e.to_string()))?;
            send.write_all(request).await.map_err(|e| e.to_string())?;
            send.finish().map_err(|e| e.to_string())?;
            recv.read_to_end(PAIRING_LIMIT).await.map_err(|e| format!("the pairing broke off: {e}"))
        }
        .await;
        conn.close(0u32.into(), b"");
        endpoint.close().await;
        answer
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

/// Connects, checks the far end's key, and reads what it says first.
async fn dial(key: &Key, to: &Address, voucher: Option<&Voucher>) -> Res<(Endpoint, Connection, String)> {
    // No ALPN of its own: a client accepts nothing.
    let endpoint = endpoint(key.secret.clone(), &to.relays, Vec::new(), Direct::Auto, 0, None, voucher).await?;
    let addr = to.endpoint_addr()?;
    let conn = match tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(addr, &to.alpn())).await {
        Err(_) => {
            // A relay that turned this device away says so, which is the
            // more useful thing to tell somebody than that nothing answered.
            let denied = iroh::Watcher::get(&mut endpoint.home_relay_status()).iter().find_map(|r| r.auth_denied_reason().map(|why| (r.url().to_string(), why.to_string())));
            if let Some((relay, _)) = denied {
                return Err(format!(
                    "{} refused this device: it has no voucher from {}, or it has expired. Pair again, or reach the \
                     server once on its LAN to renew it",
                    relay.trim_end_matches('/'),
                    to.label()
                ));
            }
            return Err(format!(
                "{} did not answer within {} s: it may be off or asleep, or the address may be out of date",
                to.label(),
                CONNECT_TIMEOUT.as_secs()
            ));
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
        Err(_) => Err(format!("{} let the connection in but said nothing", to.label())),
        Ok(Err(e)) => Err(e),
        Ok(Ok(said)) => Ok((endpoint, conn, said)),
    }
}

fn refused(e: &str) -> String {
    match e.contains("403") || e.contains("rejected") {
        true => "the server refused this device: pair it (`kvad remote pair` on the server, `kvad remote join` here), or add its id to the server's [remote] allow".into(),
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
    fn a_voucher_lets_in_the_device_it_names_while_it_lives() {
        let issuer = Issuer(Arc::new(SecretKey::from_bytes(&[7; 32])));
        let (phone, other) = (Key { secret: SecretKey::from_bytes(&[8; 32]) }.id(), Key { secret: SecretKey::from_bytes(&[9; 32]) }.id());
        let trusts = |n: &NodeId| *n == node();
        let v = issuer.issue(Some(phone), VOUCHER_LIFE);
        let back: Voucher = v.to_string().parse().unwrap();
        assert_eq!(back, v);
        assert!(back.check(&phone, trusts).is_ok());
        assert!(back.check(&other, trusts).unwrap_err().contains("another device"));
        assert!(back.check(&phone, |_| false).unwrap_err().contains("does not serve"));
        let mut forged = back.clone();
        forged.expires += 1;
        assert!(forged.check(&phone, trusts).unwrap_err().contains("not signed"));
        let mut old = issuer.issue(Some(phone), Duration::ZERO);
        old.expires = 1;
        assert!(old.check(&phone, trusts).unwrap_err().contains("expired"));
        // Any device, but only for a pairing's few minutes.
        let any = issuer.issue(None, VOUCHER_LIFE);
        assert!(any.left() <= ANY_DEVICE_LIFE);
        assert!(any.check(&other, trusts).is_ok());
        let mut long = any.clone();
        long.expires += 3600;
        assert!(long.check(&other, trusts).unwrap_err().contains("too long"));
    }

    #[test]
    fn a_pairing_link_is_an_address_and_a_token() {
        let address = Address { node: node(), knock: "0123456789abcdef0123456789abcdef".into(), relays: vec!["https://relay.kvad.eu".into()], addrs: Vec::new() };
        let issuer = Issuer(Arc::new(SecretKey::from_bytes(&[7; 32])));
        let link = PairLink { address: address.clone(), token: "a".repeat(64), voucher: Some(issuer.issue(None, ANY_DEVICE_LIFE)) };
        let back: PairLink = link.to_string().parse().unwrap();
        assert_eq!(back, link);
        let bare = PairLink { voucher: None, ..link };
        assert_eq!(bare.to_string().parse::<PairLink>().unwrap(), bare);
        assert!(address.to_string().parse::<PairLink>().unwrap_err().contains("not a pairing link"));
        assert!(format!("{address}&pair=short").parse::<PairLink>().is_err());
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
