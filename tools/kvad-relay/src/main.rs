//! kvad-relay: the relay of docs/relay-plan.md, for the Phase 0 field runs
//! and the start of Phase 3.
//!
//! It is iroh's relay server, used as a library so that its TLS is Kvad's
//! choice and not the stock binary's. The stock `iroh-relay` is built on
//! ring, which has no post-quantum key exchange; this one offers
//! X25519MLKEM768 first, and the probe and the app offer nothing else.
//!
//!     kvad-relay --hostname relay.kvad.eu --bind 85.137.228.199 \
//!                --allow /etc/kvad-relay/allow --cache /var/lib/kvad-relay
//!     kvad-relay --dev --allow FILE        # plain HTTP on 127.0.0.1:3340
//!
//! What it does and does not do:
//!
//! - *Only the keys in `--allow`* (one endpoint id a line, `#` comments) may
//!   use it; every other is refused before it can send a byte through. It is
//!   not an open relay. The file is read again on SIGHUP.
//! - *TLS from Let's Encrypt*, by the TLS-ALPN challenge on 443, so port 80
//!   need not be open: the plain-HTTP side listens on loopback only.
//! - *X25519MLKEM768 first, X25519 second.* The second is there because Let's
//!   Encrypt's validator and its ACME API speak only classical key exchange
//!   today. A Kvad client offers only the first, so it gets only the first.
//! - *QUIC address discovery* on UDP 7842, which tells each device the
//!   address its packets arrive from, so that two can punch through to each
//!   other.
//! - *Rate limits* on accepting connections and on what each client sends.
//! - *No metrics endpoint* and no logs beyond its own start and errors.

use iroh_base::EndpointId;
use iroh_relay::server::{
    Access, AccessControl, AcmeConfig, CertConfig, ClientRateLimit, ClientRequest, Limits, QuicConfig, RelayConfig, Server,
    ServerConfig, TlsConfig,
};
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

type Res<T> = Result<T, String>;

const USAGE: &str = "\
usage: kvad-relay --hostname NAME --bind IP --allow FILE --cache DIR [--staging]
                  [--https-port 443] [--quic-port 7842] [--http-port 80]
       kvad-relay --dev --allow FILE";

struct Opts {
    hostname: Option<String>,
    bind: Option<IpAddr>,
    allow: PathBuf,
    cache: Option<PathBuf>,
    staging: bool,
    dev: bool,
    /// Where HTTPS, QUIC address discovery and the loopback HTTP listen. Not
    /// the defaults when another server holds 443 and passes the relay's
    /// name through to it, as Traefik does on a k3s that serves websites.
    https_port: u16,
    quic_port: u16,
    http_port: u16,
}

fn parse() -> Res<Opts> {
    let mut o = Opts {
        hostname: None,
        bind: None,
        allow: PathBuf::new(),
        cache: None,
        staging: false,
        dev: false,
        https_port: 443,
        quic_port: 7842,
        http_port: 80,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--hostname" => o.hostname = Some(value()?),
            "--bind" => o.bind = Some(value()?.parse().map_err(|_| "--bind takes an IP address")?),
            "--allow" => o.allow = PathBuf::from(value()?),
            "--cache" => o.cache = Some(PathBuf::from(value()?)),
            "--staging" => o.staging = true,
            "--dev" => o.dev = true,
            "--https-port" => o.https_port = value()?.parse().map_err(|_| "--https-port takes a port")?,
            "--quic-port" => o.quic_port = value()?.parse().map_err(|_| "--quic-port takes a port")?,
            "--http-port" => o.http_port = value()?.parse().map_err(|_| "--http-port takes a port")?,
            _ => return Err(format!("unknown argument {a}\n{USAGE}")),
        }
    }
    if o.allow.as_os_str().is_empty() {
        return Err(format!("--allow is required: a relay serves only the keys it is told about\n{USAGE}"));
    }
    if !o.dev && (o.hostname.is_none() || o.bind.is_none() || o.cache.is_none()) {
        return Err(format!("--hostname, --bind and --cache are required outside --dev\n{USAGE}"));
    }
    Ok(o)
}

/// The keys this relay serves, from a file of endpoint ids.
#[derive(Debug)]
struct Allow {
    path: PathBuf,
    ids: RwLock<HashSet<EndpointId>>,
}

impl Allow {
    fn load(path: PathBuf) -> Res<Allow> {
        let ids = read_ids(&path)?;
        Ok(Allow { path, ids: RwLock::new(ids) })
    }

    fn reload(&self) {
        match read_ids(&self.path) {
            Ok(ids) => {
                eprintln!("kvad-relay: {} allowed keys", ids.len());
                *self.ids.write().unwrap() = ids;
            }
            Err(e) => eprintln!("kvad-relay: kept the old allow list: {e}"),
        }
    }
}

fn read_ids(path: &PathBuf) -> Res<HashSet<EndpointId>> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut ids = HashSet::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let id = line.parse().map_err(|_| format!("{}:{}: not an endpoint id", path.display(), n + 1))?;
        ids.insert(id);
    }
    Ok(ids)
}

/// The relay's door, which asks the allow list.
#[derive(Debug)]
struct Gate(Arc<Allow>);

impl AccessControl for Gate {
    async fn on_connect(&self, request: &ClientRequest) -> Access {
        if self.0.ids.read().unwrap().contains(&request.endpoint_id()) {
            Access::Allow
        } else {
            Access::Deny { reason: None }
        }
    }
}

/// Post-quantum first; classical only for what cannot do better, which today
/// is Let's Encrypt.
fn crypto() -> Arc<rustls::crypto::CryptoProvider> {
    use rustls::crypto::aws_lc_rs::{self, kx_group};
    let mut p = aws_lc_rs::default_provider();
    p.kx_groups = vec![kx_group::X25519MLKEM768, kx_group::X25519];
    Arc::new(p)
}

fn limits() -> Limits {
    let mut l = Limits::default();
    // A few new connections a second is a household of devices reconnecting;
    // more is someone else.
    l.accept_conn_limit = Some(5.0);
    l.accept_conn_burst = Some(20);
    // What one client may push through: enough for a voice note and a chat,
    // not enough to make the VPS someone's file transfer.
    let mut rx = ClientRateLimit::new(NonZeroU32::new(4 * 1024 * 1024).unwrap());
    rx.max_burst_bytes = NonZeroU32::new(16 * 1024 * 1024);
    l.client_rx = Some(rx);
    l
}

async fn config(o: &Opts, allow: Arc<Allow>) -> Res<ServerConfig> {
    if o.dev {
        let mut relay = RelayConfig::new(SocketAddr::from((Ipv4Addr::LOCALHOST, 3340)));
        relay.access = Arc::new(Gate(allow));
        relay.limits = limits();
        let mut c = ServerConfig::default();
        c.relay = Some(relay);
        return Ok(c);
    }
    let (hostname, bind, cache) = (o.hostname.clone().unwrap(), o.bind.unwrap(), o.cache.clone().unwrap());
    let builder = rustls::ServerConfig::builder_with_provider(crypto())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .with_no_client_auth();
    let acme = AcmeConfig::letsencrypt(!o.staging).domains(vec![hostname]).cache_path(cache);
    let tls = TlsConfig::new(SocketAddr::new(bind, o.https_port), CertConfig::LetsEncrypt { acme_config: acme, server_config_builder: builder });
    // Plain HTTP only where nothing outside the machine can reach it.
    let mut relay = RelayConfig::new(SocketAddr::from((Ipv4Addr::LOCALHOST, o.http_port)));
    relay.tls = Some(tls);
    relay.access = Arc::new(Gate(allow));
    relay.limits = limits();
    let mut c = ServerConfig::default();
    c.relay = Some(relay);
    c.quic = Some(QuicConfig::new(SocketAddr::new(bind, o.quic_port)));
    c.metrics_addr = None;
    Ok(c)
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("kvad-relay: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Res<()> {
    let o = parse()?;
    // Ours is the process's provider too: parts of the relay ask rustls for
    // the default, and with ring and aws-lc-rs both compiled in (iroh-relay's
    // server feature brings ring) it cannot choose one by itself.
    let _ = rustls::crypto::CryptoProvider::install_default((*crypto()).clone());
    let allow = Arc::new(Allow::load(o.allow.clone())?);
    eprintln!("kvad-relay: {} allowed keys", allow.ids.read().unwrap().len());
    let server = Server::spawn(config(&o, allow.clone()).await?).await.map_err(|e| format!("starting: {e}"))?;
    if let Some(a) = server.https_addr() {
        eprintln!("kvad-relay: https on {a}");
    }
    if let Some(a) = server.quic_addr() {
        eprintln!("kvad-relay: address discovery on udp {a}");
    }
    let mut hup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()).map_err(|e| e.to_string())?;
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).map_err(|e| e.to_string())?;
    loop {
        tokio::select! {
            _ = hup.recv() => allow.reload(),
            _ = term.recv() => break,
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    server.shutdown().await.map_err(|e| e.to_string())
}
