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

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use iroh::endpoint::{Connection, IncomingAddr};
use iroh::EndpointId;
use relay_probe::{alpn, dial, endpoint, hex, key, random, relay_url, report, selected, since, Allow, Net, Pace, Res, Ticket};

/// How many paired connections the listener serves at once. A probe needs one.
const MAX_LIVE: usize = 4;
/// How long an address whose handshake was refused is then ignored.
const BACKOFF: Duration = Duration::from_secs(30);

const USAGE: &str = "\
usage: relay-probe id     [--key FILE]
       relay-probe listen --allow ID... [--relay URL]... [--n0] [--relay-only] [--port N] [--bind IP]... [--key FILE]
       relay-probe dial   [--secs N] [--every MS] [--relay-only] [--bind IP]... [--key FILE]   (ticket on stdin)";

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
    net: Net,
    port: u16,
    allow: Vec<EndpointId>,
    secs: u64,
    every: u64,
}

fn parse(args: &[String]) -> Res<Opts> {
    let home = std::env::var_os("HOME").ok_or("HOME is not set")?;
    let mut o = Opts {
        key: PathBuf::from(home).join(".kvad-probe").join("secret.key"),
        net: Net::default(),
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
            "--relay" => o.net.relays.push(relay_url(value()?)?),
            "--n0" => o.net.n0 = true,
            "--relay-only" => o.net.relay_only = true,
            "--port" => o.port = value()?.parse().map_err(|_| "--port takes a number")?,
            "--allow" => {
                let v = value()?;
                o.allow.push(v.parse().map_err(|_| format!("--allow {v}: not an id from `relay-probe id`"))?);
            }
            "--secs" => o.secs = value()?.parse().map_err(|_| "--secs takes a number")?,
            "--every" => o.every = value()?.parse().map_err(|_| "--every takes milliseconds")?,
            "--bind" => {
                let v = value()?;
                o.net.bind.push(v.parse().map_err(|_| format!("--bind {v}: not an IP address"))?);
            }
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
        "dial" => dial_ticket(o).await,
        _ => Err(USAGE.into()),
    }
}

async fn listen(o: Opts) -> Res<()> {
    if o.allow.is_empty() {
        return Err("listen needs --allow ID; run `relay-probe id` on the dialling machine for it".into());
    }
    let secret = key(&o.key)?;
    let knock = hex(&random::<16>()?);
    let alpn = alpn(&knock);
    let relays = o.net.relays.clone();
    let has_relay = o.net.n0 || !relays.is_empty();
    let ep = endpoint(secret, &o.net, relays, vec![alpn.clone()], Some(o.port), Some(Allow(o.allow.clone()))).await?;

    if has_relay {
        if tokio::time::timeout(Duration::from_secs(10), ep.online()).await.is_err() {
            println!("warning: no relay answered within 10 s; the ticket has direct addresses only");
        }
        report(&ep, |l| println!("{l}")).await;
    } else {
        // With no relay, the addresses come from the interfaces alone.
        let t = Instant::now();
        while ep.addr().ip_addrs().next().is_none() && t.elapsed() < Duration::from_secs(3) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    let addr = ep.addr();
    println!("listening as {} on udp {}", addr.id, if o.net.relay_only { "nothing (relay only)".to_string() } else { o.port.to_string() });
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

async fn dial_ticket(o: Opts) -> Res<()> {
    eprintln!("paste the ticket from `relay-probe listen`, then press return:");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(|e| e.to_string())?;
    let ticket = Ticket::read(&line)?;
    let secret = key(&o.key)?;
    let stop = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let pace = Pace { secs: o.secs, every: o.every };
    let s = dial(secret, &o.net, ticket, pace, stop, |e, _| {
        if !matches!(e, relay_probe::Event::Ping { .. }) {
            println!("{e}");
        }
    })
    .await?;
    println!("\n{s}");
    match s.failure {
        Some(e) => Err(format!("stopped early: {e}")),
        None => Ok(()),
    }
}
