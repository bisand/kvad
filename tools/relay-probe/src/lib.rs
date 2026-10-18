//! The probe's core, shared by the `relay-probe` command (`main.rs`) and the
//! Network test screen of Kvad for iPhone (`apps/phone`): tickets, the
//! post-quantum endpoint with no port mapping, the dialler's ping loop, and
//! the path classification and summary it prints.
//!
//! What it promises is in `main.rs`; this is where those promises are kept,
//! so the phone keeps them too.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::future::Future;
use std::io::{ErrorKind, Read, Write as _};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use iroh::endpoint::{presets, AfterHandshakeOutcome, Connection, EndpointHooks, PortmapperConfig};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl, SecretKey, TransportAddr, Watcher as _};

pub type Res<T> = Result<T, String>;

/// A relay must be HTTPS. Plain HTTP is let through only to this machine, so
/// that a relay started with `iroh-relay --dev` can be tested against.
pub fn relay_url(s: &str) -> Res<RelayUrl> {
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
pub fn key(path: &Path) -> Res<SecretKey> {
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
pub fn random<const N: usize>() -> Res<[u8; N]> {
    let mut b = [0u8; N];
    fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(b)
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Hybrid post-quantum key exchange, and nothing else. Both ends of a peer
/// connection are this code, so nothing needs a classical fallback, and a
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
pub struct Allow(pub Vec<EndpointId>);

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

/// Which networks an endpoint may use: the command line's `--relay`, `--n0`,
/// `--relay-only` and `--bind`.
#[derive(Clone, Debug, Default)]
pub struct Net {
    pub relays: Vec<RelayUrl>,
    pub n0: bool,
    pub relay_only: bool,
    /// Local addresses to bind the UDP socket to, instead of every interface:
    /// how a run is held to one network when a machine is on several.
    pub bind: Vec<IpAddr>,
}

pub async fn endpoint(secret: SecretKey, net: &Net, relays: Vec<RelayUrl>, alpns: Vec<Vec<u8>>, port: Option<u16>, allow: Option<Allow>) -> Res<Endpoint> {
    let mode = if net.n0 {
        RelayMode::Default
    } else if relays.is_empty() {
        RelayMode::Disabled
    } else {
        RelayMode::custom(relays)
    };
    if net.relay_only && matches!(mode, RelayMode::Disabled) {
        return Err("--relay-only needs a relay".into());
    }
    let mut b = Endpoint::builder(presets::Empty)
        .crypto_provider(crypto())
        .secret_key(secret)
        .alpns(alpns)
        .relay_mode(mode)
        .portmapper_config(PortmapperConfig::Disabled);
    if net.relay_only {
        // No UDP socket at all: nothing on this machine answers the internet.
        b = b.clear_ip_transports();
    } else if !net.bind.is_empty() {
        // Only these addresses. Packets leave from them alone, and an answer
        // to any other of this machine's addresses (a cable, a LAN, a
        // tailnet) has no socket to arrive at, so that path cannot form.
        b = b.clear_ip_transports();
        for ip in &net.bind {
            b = b.bind_addr(SocketAddr::new(*ip, port.unwrap_or(0))).map_err(|e| e.to_string())?;
        }
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
pub struct Ticket {
    pub addr: EndpointAddr,
    pub alpn: Vec<u8>,
}

impl Ticket {
    pub fn write(&self) -> String {
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

    pub fn read(s: &str) -> Res<Ticket> {
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

pub fn alpn(knock: &str) -> Vec<u8> {
    format!("kvad-probe/1/{knock}").into_bytes()
}

/// A path as the plan names it (relay, lan, cgnat, direct-v4, direct-v6),
/// and the address it goes to.
pub type Route = (&'static str, String);

/// Which of the plan's paths a connection is on now, and to where.
pub fn selected(conn: &Connection) -> Option<Route> {
    let paths = conn.paths();
    let found = paths.iter().find(|p| p.is_selected()).map(|p| describe(p.remote_addr()));
    found
}

fn describe(a: &TransportAddr) -> Route {
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

/// A time since the start of a run, as every line of the log begins.
pub fn at(d: Duration) -> String {
    format!("{:>7.2}s", d.as_secs_f64())
}

pub fn since(t0: Instant) -> String {
    at(t0.elapsed())
}

pub fn ms(d: Duration) -> String {
    format!("{:.1} ms", d.as_secs_f64() * 1e3)
}

/// What the endpoint learnt about its network from its relays, if it has
/// any: one `network:` line each.
pub async fn report(ep: &Endpoint, mut line: impl FnMut(String)) {
    let mut w = ep.net_report();
    let Ok(r) = tokio::time::timeout(Duration::from_secs(8), w.initialized()).await else {
        line("network: no report from a relay within 8 s".into());
        return;
    };
    let hard = match r.mapping_varies_by_dest() {
        Some(true) => "yes: the NAT gives each destination its own port, so punching will mostly fail",
        Some(false) => "no",
        None => "unknown",
    };
    line(format!("network: udp v4 {}, udp v6 {}", r.udp_v4, r.udp_v6));
    line(format!("network: public v4 {}", r.global_v4.map_or("none seen".into(), |a| a.to_string())));
    line(format!("network: public v6 {}", r.global_v6.map_or("none seen".into(), |a| a.to_string())));
    line(format!("network: NAT mapping varies by destination: {hard}"));
    if r.captive_portal == Some(true) {
        line("network: behind a captive portal".into());
    }
}

/// What a dial says as it goes. Each but `Ping` is a line the command prints,
/// which `Display` gives.
#[derive(Clone, Debug)]
pub enum Event {
    /// A line of the network report, `network:` and all.
    Network(String),
    /// The connection is up, after `ms`, on `path` if one is selected yet.
    Connected { at: Duration, ms: u128, path: Option<Route> },
    /// The connection moved to another path.
    Moved { at: Duration, path: Route },
    /// A ping came back, on the path the connection is on.
    Ping { at: Duration, rtt: Duration },
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::Network(s) => f.write_str(s),
            Event::Connected { at: t, ms, path } => {
                write!(f, "{} connected in {ms} ms on {}", at(*t), path.as_ref().map_or("?".into(), |(k, a)| format!("{k} {a}")))
            }
            Event::Moved { at: t, path: (kind, to) } => write!(f, "{} now on {kind} {to}", at(*t)),
            Event::Ping { at: t, rtt } => write!(f, "{} ping {}", at(*t), ms(*rtt)),
        }
    }
}

/// One row of the summary's table: a path and its round trips.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub kind: &'static str,
    pub pings: usize,
    pub median: Duration,
    pub p90: Duration,
    pub max: Duration,
}

/// The run so far, or once it is over: what the command prints at the end.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    rtts: BTreeMap<&'static str, Vec<Duration>>,
    pub pings: u64,
    pub elapsed: Duration,
    pub changes: u32,
    pub longest: Duration,
    pub first_direct: Option<Duration>,
    /// Why the run ended before its time, if it did.
    pub failure: Option<String>,
}

impl Summary {
    /// Each path's pings, in the plan's order of names.
    pub fn rows(&self) -> Vec<Row> {
        self.rtts
            .iter()
            .map(|(kind, v)| {
                let mut v = v.clone();
                v.sort();
                let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
                Row { kind, pings: v.len(), median: at(0.5), p90: at(0.9), max: at(1.0) }
            })
            .collect()
    }

    /// The line under the table.
    pub fn totals(&self) -> String {
        format!(
            "{} pings over {:.1} s; {} path changes; longest ping {}; first direct path {}",
            self.pings,
            self.elapsed.as_secs_f64(),
            self.changes,
            ms(self.longest),
            self.first_direct.map_or("never".into(), |d| format!("after {:.2} s", d.as_secs_f64())),
        )
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "path        pings     median        p90        max")?;
        for r in self.rows() {
            writeln!(f, "{:<10} {:>6} {:>10} {:>10} {:>10}", r.kind, r.pings, ms(r.median), ms(r.p90), ms(r.max))?;
        }
        write!(f, "\n{}", self.totals())
    }
}

/// How long a dial pings for, and how often.
#[derive(Clone, Copy, Debug)]
pub struct Pace {
    pub secs: u64,
    pub every: u64,
}

/// Dials the ticket's listener with `secret`, and pings it every `every` ms
/// for `secs` s, or until `stop` is ready. Tells `on` what happens as it
/// happens, with the summary so far. An error is a run that never began; a
/// run that broke off has its reason in the summary's `failure`.
pub async fn dial(secret: SecretKey, net: &Net, ticket: Ticket, pace: Pace, stop: impl Future<Output = ()>, mut on: impl FnMut(&Event, &Summary)) -> Res<Summary> {
    let mut s = Summary::default();
    let mut relays: Vec<RelayUrl> = ticket.addr.relay_urls().cloned().collect();
    relays.extend(net.relays.iter().cloned());
    // No ALPN of its own: the dialler accepts nothing.
    let ep = endpoint(secret, net, relays.clone(), Vec::new(), None, None).await?;
    if !relays.is_empty() || net.n0 {
        let mut lines = Vec::new();
        report(&ep, |l| lines.push(l)).await;
        for l in lines {
            on(&Event::Network(l), &s);
        }
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
    s.first_direct = first.as_ref().filter(|(k, _)| *k != "relay").map(|_| opened);
    on(&Event::Connected { at: t0.elapsed(), ms: opened.as_millis(), path: first.clone() }, &s);

    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
    let mut last = first;
    let end = t0 + Duration::from_secs(pace.secs);
    let every = Duration::from_millis(pace.every);
    let mut stop = std::pin::pin!(stop);
    while Instant::now() < end {
        let tick = Instant::now();
        let mut frame = [0u8; 16];
        frame[..8].copy_from_slice(&s.pings.to_be_bytes());
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
            _ = &mut stop => break,
        };
        match r {
            Err(_) => {
                s.failure = Some("a ping went unanswered for 30 s".to_string());
                break;
            }
            Ok(Err(e)) => {
                s.failure = Some(e);
                break;
            }
            Ok(Ok(())) => {}
        }
        let rtt = tick.elapsed();
        s.longest = s.longest.max(rtt);
        let now = selected(&conn);
        if now != last {
            s.changes += 1;
            if let Some(path) = &now {
                if path.0 != "relay" && s.first_direct.is_none() {
                    s.first_direct = Some(t0.elapsed());
                }
                on(&Event::Moved { at: t0.elapsed(), path: path.clone() }, &s);
            }
            last = now.clone();
        }
        if let Some((kind, _)) = now {
            s.rtts.entry(kind).or_default().push(rtt);
        }
        s.pings += 1;
        s.elapsed = t0.elapsed();
        on(&Event::Ping { at: s.elapsed, rtt }, &s);
        if let Some(rest) = every.checked_sub(tick.elapsed()) {
            tokio::time::sleep(rest).await;
        }
    }
    conn.close(0u32.into(), b"");
    ep.close().await;
    s.elapsed = t0.elapsed();
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ticket_reads_back_as_written() {
        let id = SecretKey::generate().public();
        let addrs = vec![TransportAddr::Relay(relay_url("https://relay2.kvad.eu").unwrap()), TransportAddr::Ip("192.0.2.7:5824".parse().unwrap())];
        let t = Ticket { addr: EndpointAddr::from_parts(id, addrs), alpn: alpn(&hex(&[7u8; 16])) };
        let s = t.write();
        let back = Ticket::read(&format!("  {s}\n")).unwrap();
        assert_eq!(back.write(), s);
        assert_eq!(back.alpn, t.alpn);
    }

    #[test]
    fn a_ticket_must_be_whole() {
        assert!(Ticket::read("kvadprobe:1:abc:def").is_err());
        assert!(Ticket::read("hello").is_err());
        let id = SecretKey::generate().public();
        let knock = hex(&[1u8; 16]);
        assert!(Ticket::read(&format!("kvadprobe:2:{id}:{knock}")).err().unwrap().contains("version 2"));
        assert!(Ticket::read(&format!("kvadprobe:1:{id}:{knock}?relay=http://relay.example.org")).is_err());
    }

    #[test]
    fn paths_are_named_as_the_plan_names_them() {
        let ip = |s: &str| describe(&TransportAddr::Ip(s.parse().unwrap())).0;
        assert_eq!(ip("192.168.1.20:1"), "lan");
        assert_eq!(ip("[fe80::1]:1"), "lan");
        assert_eq!(ip("100.70.1.2:1"), "cgnat");
        assert_eq!(ip("100.128.1.2:1"), "direct-v4");
        assert_eq!(ip("[2001:db8::1]:1"), "direct-v6");
        assert_eq!(ip("[::ffff:203.0.113.5]:1"), "direct-v4");
    }

    #[test]
    fn the_table_is_the_commands() {
        let mut s = Summary { pings: 3, elapsed: Duration::from_millis(1500), changes: 1, longest: Duration::from_millis(60), ..Summary::default() };
        s.rtts.insert("relay", vec![Duration::from_millis(60), Duration::from_millis(40)]);
        s.rtts.insert("lan", vec![Duration::from_millis(2)]);
        s.first_direct = Some(Duration::from_millis(1250));
        let want = "\
path        pings     median        p90        max
lan             1     2.0 ms     2.0 ms     2.0 ms
relay           2    60.0 ms    60.0 ms    60.0 ms

3 pings over 1.5 s; 1 path changes; longest ping 60.0 ms; first direct path after 1.25 s";
        assert_eq!(s.to_string(), want);
    }
}
