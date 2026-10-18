//! Phase 0 of `docs/relay-plan.md`: which path two devices reach each other
//! by, how long the relay is needed before a direct path takes over, and what
//! each path costs in round-trip time.
//!
//! On the machine that dials (a laptop on another network):
//!
//!     relay-probe id
//!
//! On the machine to be reached (the Mac), with the id that printed:
//!
//!     relay-probe listen --allow <id> [--relay https://relay.example.org]
//!
//! It prints a ticket. Back on the dialling machine, paste it in:
//!
//!     relay-probe dial [--secs 60]
//!
//! Safe by construction, not by a flag that could be left off:
//!
//! - *No port mapping.* iroh's `portmapper` is not compiled in. The probe
//!   never asks a router for anything.
//! - *No third party.* No relay and no address lookup service is used unless
//!   one is named: `--relay URL`, or `--n0` for n0's public relays.
//! - *Both ends pinned.* The dialler checks the listener's key, which the
//!   ticket carries. The listener refuses every key not given with `--allow`
//!   at the handshake, before a byte of the probe's own protocol is read.
//! - *Nothing for scanners.* The ALPN is random for each run and travels only
//!   in the ticket. A connection attempt without it gets either no answer, or
//!   one TLS alert sent before the listener's key would have been.
//! - *Post-quantum.* The only key exchange offered is X25519MLKEM768, for the
//!   peer connection and for the relay's HTTPS alike. A relay that cannot do
//!   it is refused, not fallen back from.
//! - *Keys in owner-only files.* Never on the command line, never printed. A
//!   key file that others can read is refused.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{ErrorKind, Read, Write as _};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use iroh::endpoint::{presets, AfterHandshakeOutcome, Connection, EndpointHooks, IncomingAddr, PortmapperConfig};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl, SecretKey, TransportAddr, Watcher as _};

type Res<T> = Result<T, String>;

/// How many paired connections the listener serves at once. A probe needs one.
const MAX_LIVE: usize = 4;
/// How long an address whose handshake was refused is then ignored.
const BACKOFF: Duration = Duration::from_secs(30);

const USAGE: &str = "\
usage: relay-probe id     [--key FILE]
       relay-probe listen --allow ID... [--relay URL]... [--n0] [--relay-only] [--port N] [--key FILE]
       relay-probe dial   [--secs N] [--every MS] [--relay-only] [--key FILE]   (ticket on stdin)";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("a tokio runtime");
    if let Err(e) = rt.block_on(run(args)) {
        eprintln!("relay-probe: {e}");
        std::process::exit(1);
    }
}

struct Opts {
    key: PathBuf,
    relays: Vec<RelayUrl>,
    n0: bool,
    relay_only: bool,
    port: u16,
    allow: Vec<EndpointId>,
    secs: u64,
    every: u64,
}

fn parse(args: &[String]) -> Res<Opts> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
    let mut o = Opts {
        key: PathBuf::from(home).join(".kvad-probe").join("secret.key"),
        relays: Vec::new(),
        n0: false,
        relay_only: false,
        port: 5824,
        allow: Vec::new(),
        secs: 60,
        every: 250,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--key" => o.key = PathBuf::from(value()?),
            "--relay" => o.relays.push(relay_url(value()?)?),
            "--n0" => o.n0 = true,
            "--relay-only" => o.relay_only = true,
            "--port" => o.port = value()?.parse().map_err(|_| "--port takes a number")?,
            "--allow" => {
                let v = value()?;
                o.allow.push(v.parse().map_err(|_| format!("--allow {v}: not an id from `relay-probe id`"))?);
            }
            "--secs" => o.secs = value()?.parse().map_err(|_| "--secs takes a number")?,
            "--every" => o.every = value()?.parse().map_err(|_| "--every takes milliseconds")?,
            _ => return Err(format!("unknown argument {a}\n{USAGE}")),
        }
    }
    if o.every == 0 {
        return Err("--every must be at least 1 ms".into());
    }
    Ok(o)
}

async fn run(args: Vec<String>) -> Res<()> {
    let Some((verb, rest)) = args.split_first() else { return Err(USAGE.into()) };
    let o = parse(rest)?;
    match verb.as_str() {
        "id" => {
            println!("{}", key(&o.key)?.public());
            Ok(())
        }
        "listen" => listen(o).await,
        "dial" => dial(o).await,
        _ => Err(USAGE.into()),
    }
}

/// A relay must be HTTPS. Plain HTTP is let through only to this machine, so
/// that a relay started with `iroh-relay --dev` can be tested against.
fn relay_url(s: &str) -> Res<RelayUrl> {
    if let Some(rest) = s.strip_prefix("http://") {
        let authority = rest.split('/').next().unwrap_or("");
        let host = authority.rsplit_once(':').map_or(authority, |(h, _)| h);
        if !matches!(host, "localhost" | "127.0.0.1" | "[::1]") {
            return Err(format!("{s}: a relay must be https://; plain http only to this machine, with a port"));
        }
    } else if !s.starts_with("https://") {
        return Err(format!("{s}: a relay must be https://"));
    }
    s.parse().map_err(|e| format!("{s}: {e}"))
}

/// The identity in `path`, made on first use: a directory only its owner may
/// write to, a file only its owner may read, created so that it never exists
/// with looser permissions even for a moment. A symlink is refused, and the
/// file checked is the file read.
fn key(path: &PathBuf) -> Res<SecretKey> {
    let shown = path.display();
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).ok_or_else(|| format!("{shown}: a key needs a directory"))?;
    fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let d = fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let me = fs::symlink_metadata(std::env::var_os("HOME").ok_or("HOME is not set")?).map(|m| m.uid()).ok();
    if d.permissions().mode() & 0o022 != 0 || Some(d.uid()) != me && d.uid() != 0 {
        return Err(format!("{}: others can write to it, so a key in it cannot be trusted", dir.display()));
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
            if Some(opened.uid()) != me {
                return Err(format!("{shown}: not owned by this user"));
            }
            let mut bytes = [0u8; 32];
            f.read_exact(&mut bytes).map_err(|e| format!("{shown}: {e}"))?;
            if f.read(&mut [0u8; 1]).map_err(|e| format!("{shown}: {e}"))? != 0 {
                return Err(format!("{shown}: longer than a key"));
            }
            let k = SecretKey::from_bytes(&bytes);
            wipe(&mut bytes);
            Ok(k)
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let k = SecretKey::generate();
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)
                .map_err(|e| format!("{shown}: {e}"))?;
            let mut bytes = k.to_bytes();
            let r = f.write_all(&bytes).and_then(|_| f.sync_all());
            wipe(&mut bytes);
            r.map_err(|e| format!("{shown}: {e}"))?;
            eprintln!("made a new identity in {shown}");
            Ok(k)
        }
        Err(e) => Err(format!("{shown}: {e}")),
    }
}

/// Zeroes a copy of a key, in a way the optimiser may not drop.
fn wipe(b: &mut [u8; 32]) {
    for x in b.iter_mut() {
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

/// Bytes from the system's generator.
fn random<const N: usize>() -> Res<[u8; N]> {
    let mut b = [0u8; N];
    fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(b)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Hybrid post-quantum key exchange, and nothing else. Both ends of a peer
/// connection are this binary, so nothing needs a classical fallback, and a
/// fallback that exists is one an attacker can try to steer towards.
fn crypto() -> Arc<rustls::crypto::CryptoProvider> {
    use rustls::crypto::aws_lc_rs::{self, kx_group};
    let mut p = aws_lc_rs::default_provider();
    p.kx_groups = vec![kx_group::X25519MLKEM768];
    Arc::new(p)
}

/// Refuses every peer whose key was not given with `--allow`. Runs once the
/// TLS handshake has proved the peer holds that key, before any stream.
#[derive(Debug)]
struct Allow(Vec<EndpointId>);

impl EndpointHooks for Allow {
    fn after_handshake<'a>(&'a self, conn: &'a Connection) -> impl std::future::Future<Output = AfterHandshakeOutcome> + Send + 'a {
        let ok = self.0.contains(&conn.remote_id());
        async move {
            if ok {
                AfterHandshakeOutcome::accept()
            } else {
                AfterHandshakeOutcome::Reject { error_code: 403u32.into(), reason: Vec::new() }
            }
        }
    }
}

async fn endpoint(secret: SecretKey, o: &Opts, relays: Vec<RelayUrl>, alpns: Vec<Vec<u8>>, port: Option<u16>, allow: Option<Allow>) -> Res<Endpoint> {
    let mode = if o.n0 {
        RelayMode::Default
    } else if relays.is_empty() {
        RelayMode::Disabled
    } else {
        RelayMode::custom(relays)
    };
    if o.relay_only && matches!(mode, RelayMode::Disabled) {
        return Err("--relay-only needs a relay".into());
    }
    let mut b = Endpoint::builder(presets::Empty)
        .crypto_provider(crypto())
        .secret_key(secret)
        .alpns(alpns)
        .relay_mode(mode)
        .portmapper_config(PortmapperConfig::Disabled);
    if o.relay_only {
        // No UDP socket at all: nothing on this machine answers the internet.
        b = b.clear_ip_transports();
    } else if let Some(port) = port {
        let v4 = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
        let v6 = SocketAddr::from((Ipv6Addr::UNSPECIFIED, port));
        b = b.clear_ip_transports().bind_addr(v4).map_err(|e| e.to_string())?.bind_addr(v6).map_err(|e| e.to_string())?;
    }
    if let Some(allow) = allow {
        b = b.hooks(allow);
    }
    b.bind().await.map_err(|e| format!("binding the endpoint: {e}"))
}

/// What the ticket carries: who to dial, by which ALPN, and where.
struct Ticket {
    addr: EndpointAddr,
    alpn: Vec<u8>,
}

impl Ticket {
    fn write(&self) -> String {
        let knock = std::str::from_utf8(&self.alpn).unwrap_or_default().rsplit('/').next().unwrap_or_default();
        let mut s = format!("kvadprobe:1:{}:{knock}", self.addr.id);
        let mut sep = '?';
        for a in &self.addr.addrs {
            match a {
                TransportAddr::Relay(u) => s += &format!("{sep}relay={u}"),
                TransportAddr::Ip(ip) => s += &format!("{sep}addr={ip}"),
                _ => continue,
            }
            sep = '&';
        }
        s
    }

    fn read(s: &str) -> Res<Ticket> {
        let bad = || "not a relay-probe ticket".to_string();
        let (head, query) = s.trim().split_once('?').unwrap_or((s.trim(), ""));
        let [magic, version, id, knock] = head.split(':').collect::<Vec<_>>()[..] else { return Err(bad()) };
        if magic != "kvadprobe" {
            return Err(bad());
        }
        if version != "1" {
            return Err(format!("ticket version {version}; this probe reads version 1"));
        }
        if knock.len() != 32 || !knock.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(bad());
        }
        let id: EndpointId = id.parse().map_err(|_| bad())?;
        let mut addrs = Vec::new();
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            match pair.split_once('=') {
                Some(("relay", u)) => addrs.push(TransportAddr::Relay(relay_url(u)?)),
                Some(("addr", a)) => addrs.push(TransportAddr::Ip(a.parse().map_err(|_| format!("addr={a}: not an address"))?)),
                _ => return Err(format!("{pair}: unknown in a ticket")),
            }
        }
        Ok(Ticket { addr: EndpointAddr::from_parts(id, addrs), alpn: alpn(knock) })
    }
}

fn alpn(knock: &str) -> Vec<u8> {
    format!("kvad-probe/1/{knock}").into_bytes()
}

/// Which of the plan's paths a connection is on now, and to where.
fn selected(conn: &Connection) -> Option<(&'static str, String)> {
    let paths = conn.paths();
    let found = paths.iter().find(|p| p.is_selected()).map(|p| describe(p.remote_addr()));
    found
}

fn describe(a: &TransportAddr) -> (&'static str, String) {
    match a {
        TransportAddr::Relay(u) => ("relay", u.to_string()),
        TransportAddr::Ip(s) if is_lan(s.ip()) => ("lan", s.to_string()),
        // 100.64.0.0/10: a carrier's NAT, or a tailnet such as Tailscale's.
        TransportAddr::Ip(s) if matches!(s.ip().to_canonical(), IpAddr::V4(a) if a.octets()[0] == 100 && a.octets()[1] & 0xc0 == 64) => ("cgnat", s.to_string()),
        TransportAddr::Ip(s) if s.is_ipv6() && s.ip().to_canonical().is_ipv6() => ("direct-v6", s.to_string()),
        TransportAddr::Ip(s) => ("direct-v4", s.to_string()),
        other => ("other", format!("{other:?}")),
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

fn since(t0: Instant) -> String {
    format!("{:>7.2}s", t0.elapsed().as_secs_f64())
}

/// What the endpoint learnt about its network from its relays, if it has any.
async fn report(ep: &Endpoint) {
    let mut w = ep.net_report();
    let Ok(r) = tokio::time::timeout(Duration::from_secs(8), w.initialized()).await else {
        println!("network: no report from a relay within 8 s");
        return;
    };
    let hard = match r.mapping_varies_by_dest() {
        Some(true) => "yes: the NAT gives each destination its own port, so punching will mostly fail",
        Some(false) => "no",
        None => "unknown",
    };
    println!("network: udp v4 {}, udp v6 {}", r.udp_v4, r.udp_v6);
    println!("network: public v4 {}", r.global_v4.map_or("none seen".into(), |a| a.to_string()));
    println!("network: public v6 {}", r.global_v6.map_or("none seen".into(), |a| a.to_string()));
    println!("network: NAT mapping varies by destination: {hard}");
    if r.captive_portal == Some(true) {
        println!("network: behind a captive portal");
    }
}

async fn listen(o: Opts) -> Res<()> {
    if o.allow.is_empty() {
        return Err("listen needs --allow ID; run `relay-probe id` on the dialling machine for it".into());
    }
    let secret = key(&o.key)?;
    let knock = hex(&random::<16>()?);
    let alpn = alpn(&knock);
    let relays = o.relays.clone();
    let has_relay = o.n0 || !relays.is_empty();
    let ep = endpoint(secret, &o, relays, vec![alpn.clone()], Some(o.port), Some(Allow(o.allow.clone()))).await?;

    if has_relay {
        if tokio::time::timeout(Duration::from_secs(10), ep.online()).await.is_err() {
            println!("warning: no relay answered within 10 s; the ticket has direct addresses only");
        }
        report(&ep).await;
    } else {
        // With no relay, the addresses come from the interfaces alone.
        let t = Instant::now();
        while ep.addr().ip_addrs().next().is_none() && t.elapsed() < Duration::from_secs(3) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    let addr = ep.addr();
    println!("listening as {} on udp {}", addr.id, if o.relay_only { "nothing (relay only)".to_string() } else { o.port.to_string() });
    println!("allowed: {}", o.allow.iter().map(|i| i.fmt_short().to_string()).collect::<Vec<_>>().join(", "));
    println!("\nticket (paste into `relay-probe dial`; good for this run only):\n\n{}\n", Ticket { addr, alpn: alpn.clone() }.write());

    let t0 = Instant::now();
    let live = Arc::new(AtomicUsize::new(0));
    let refused: Arc<Mutex<HashMap<IpAddr, Instant>>> = Arc::default();
    loop {
        let incoming = tokio::select! {
            i = ep.accept() => match i { Some(i) => i, None => break },
            _ = tokio::signal::ctrl_c() => break,
        };
        let from = incoming.remote_addr();
        let ip = match &from {
            IncomingAddr::Ip(a) => Some(a.ip().to_canonical()),
            _ => None,
        };
        // The knock. When the first packet shows the ALPNs and ours is not
        // among them, nothing is sent back at all. When it does not show them,
        // which is usual, since the post-quantum key share pushes the
        // ClientHello past one packet, rustls refuses a wrong ALPN while
        // reading the ClientHello: one alert, before a ServerHello, so the key
        // is never sent.
        let wrong = incoming.decrypt().and_then(|d| d.alpns()).is_some_and(|mut it| !it.any(|a| a.is_ok_and(|a| a[..] == alpn[..])));
        if wrong {
            incoming.ignore();
            continue;
        }
        if let Some(ip) = ip {
            if refused.lock().unwrap().get(&ip).is_some_and(|t| t.elapsed() < BACKOFF) {
                incoming.ignore();
                continue;
            }
        }
        if live.load(Ordering::SeqCst) >= MAX_LIVE {
            incoming.ignore();
            continue;
        }
        let (live, refused) = (live.clone(), refused.clone());
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    if let Some(ip) = ip {
                        refused.lock().unwrap().insert(ip, Instant::now());
                    }
                    println!("{} refused {}: {e}", since(t0), show_from(&from));
                    return;
                }
            };
            live.fetch_add(1, Ordering::SeqCst);
            let who = conn.remote_id().fmt_short().to_string();
            println!("{} {who} connected", since(t0));
            let watch = tokio::spawn(watch_paths(conn.clone(), who.clone(), t0));
            let r = echo(&conn).await;
            watch.abort();
            live.fetch_sub(1, Ordering::SeqCst);
            println!("{} {who} gone: {}", since(t0), r.err().unwrap_or_else(|| "closed".into()));
        });
    }
    ep.close().await;
    Ok(())
}

fn show_from(a: &IncomingAddr) -> String {
    match a {
        IncomingAddr::Ip(s) => s.to_string(),
        IncomingAddr::Relay { url, .. } => format!("via {url}"),
        other => format!("{other:?}"),
    }
}

/// Sends back every 16-byte ping on the first stream, and opens no other.
async fn echo(conn: &Connection) -> Res<()> {
    let (mut send, mut recv) = conn.accept_bi().await.map_err(|e| e.to_string())?;
    let mut buf = [0u8; 16];
    loop {
        recv.read_exact(&mut buf).await.map_err(|e| e.to_string())?;
        send.write_all(&buf).await.map_err(|e| e.to_string())?;
    }
}

async fn watch_paths(conn: Connection, who: String, t0: Instant) {
    let mut last = None;
    loop {
        let now = selected(&conn);
        if now != last {
            if let Some((kind, to)) = &now {
                println!("{} {who} now on {kind} {to}", since(t0));
            }
            last = now;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn dial(o: Opts) -> Res<()> {
    eprintln!("paste the ticket from `relay-probe listen`, then press return:");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
    let ticket = Ticket::read(&line)?;
    let secret = key(&o.key)?;
    let mut relays: Vec<RelayUrl> = ticket.addr.relay_urls().cloned().collect();
    relays.extend(o.relays.iter().cloned());
    // No ALPN of its own: the dialler accepts nothing.
    let ep = endpoint(secret, &o, relays.clone(), Vec::new(), None, None).await?;
    if !relays.is_empty() || o.n0 {
        report(&ep).await;
    }

    let want = ticket.addr.id;
    let t0 = Instant::now();
    let conn = match tokio::time::timeout(Duration::from_secs(30), ep.connect(ticket.addr, &ticket.alpn)).await {
        Err(_) => return Err("no connection within 30 s".into()),
        Ok(Err(e)) => return Err(format!("connecting: {e}")),
        Ok(Ok(c)) => c,
    };
    if conn.remote_id() != want {
        return Err("connected to a key other than the ticket's".into());
    }
    let opened = t0.elapsed();
    let first = selected(&conn);
    println!("{} connected in {} ms on {}", since(t0), opened.as_millis(), first.as_ref().map_or("?".into(), |(k, a)| format!("{k} {a}")));

    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
    let mut rtts: BTreeMap<&'static str, Vec<Duration>> = BTreeMap::new();
    let mut last = first;
    let mut changes = 0;
    let mut first_direct: Option<Duration> = last.as_ref().filter(|(k, _)| *k != "relay").map(|_| opened);
    let mut longest = Duration::ZERO;
    let mut failure = None;
    let end = t0 + Duration::from_secs(o.secs);
    let every = Duration::from_millis(o.every);
    let mut seq: u64 = 0;
    while Instant::now() < end {
        let tick = Instant::now();
        let mut frame = [0u8; 16];
        frame[..8].copy_from_slice(&seq.to_be_bytes());
        frame[8..].copy_from_slice(&(t0.elapsed().as_nanos() as u64).to_be_bytes());
        let ping = async {
            send.write_all(&frame).await.map_err(|e| e.to_string())?;
            let mut back = [0u8; 16];
            recv.read_exact(&mut back).await.map_err(|e| e.to_string())?;
            if back != frame {
                return Err("the echo came back changed".to_string());
            }
            Ok(())
        };
        let r = tokio::select! {
            r = tokio::time::timeout(Duration::from_secs(30), ping) => r,
            _ = tokio::signal::ctrl_c() => break,
        };
        match r {
            Err(_) => {
                failure = Some("a ping went unanswered for 30 s".to_string());
                break;
            }
            Ok(Err(e)) => {
                failure = Some(e);
                break;
            }
            Ok(Ok(())) => {}
        }
        let rtt = tick.elapsed();
        longest = longest.max(rtt);
        let now = selected(&conn);
        if now != last {
            changes += 1;
            if let Some((kind, to)) = &now {
                println!("{} now on {kind} {to}", since(t0));
                if *kind != "relay" && first_direct.is_none() {
                    first_direct = Some(t0.elapsed());
                }
            }
            last = now.clone();
        }
        if let Some((kind, _)) = now {
            rtts.entry(kind).or_default().push(rtt);
        }
        seq += 1;
        if let Some(rest) = every.checked_sub(tick.elapsed()) {
            tokio::time::sleep(rest).await;
        }
    }
    conn.close(0u32.into(), b"");
    ep.close().await;

    println!("\npath        pings     median        p90        max");
    for (kind, v) in &mut rtts {
        v.sort();
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        println!("{kind:<10} {:>6} {:>10} {:>10} {:>10}", v.len(), ms(at(0.5)), ms(at(0.9)), ms(at(1.0)));
    }
    println!(
        "\n{seq} pings over {:.1} s; {changes} path changes; longest ping {}; first direct path {}",
        t0.elapsed().as_secs_f64(),
        ms(longest),
        first_direct.map_or("never".into(), |d| format!("after {:.2} s", d.as_secs_f64())),
    );
    match failure {
        Some(e) => Err(format!("stopped early: {e}")),
        None => Ok(()),
    }
}

fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1e3)
}
