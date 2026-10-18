//! `kvad tunnel`: a port on this machine that is a paired `kvad://` server,
//! for the programs that can only speak HTTP to a port (`docs/relay-plan.md`,
//! Phase 4).
//!
//! It listens on `127.0.0.1` and forwards each TCP connection, byte for
//! byte, as one stream through the tunnel, the same streams the CLI's own
//! commands open. So OpenCode, curl or the openai SDK work against
//! `http://127.0.0.1:5824/v1` unmodified, from wherever the device is.
//!
//! What the port gives away is the part to be careful with. The CLI's own
//! connections are Unix socket pairs that nothing else can reach; a port can
//! be reached by every process on the machine, and by every other user of
//! it. Through it they all speak as this device to the server's handshake.
//! What they do not get is the device's sign-in: the server still wants a
//! key with each request, so each program brings one of its own (`kvad keys
//! add`) and can be revoked on its own.
//!
//! `--with-key` is the shortcut: the forwarder adds the device's key to
//! every request, and anything that can reach the port acts as the device,
//! with its rights. It is never on unless asked for, and says so when it
//! starts. With it on, a request a browser sent from some other page is
//! refused here (`Origin`, `Sec-Fetch-Site`, and a `Host` that is not this
//! port), since a web page can make requests to loopback too, and each
//! request gets a connection of its own, so that every request on it is one
//! this forwarder has read the head of.

use super::Args;
use kvad::client::{self, Choice};
use kvad_tunnel::{Client, Key, Path, Voucher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Where it listens unless told: beside kvad-serve's 5823, and not on it,
/// so that the two can run on one machine.
pub const DEFAULT_PORT: u16 = 5824;

/// How long a connection waits for the tunnel while it is being made again,
/// before it is told the server cannot be reached.
const WAIT_FOR_LINK: Duration = Duration::from_secs(20);

/// The longest pause between attempts to reach a server that is not
/// answering.
const MOST_WAIT: Duration = Duration::from_secs(30);

/// How often a long-lived tunnel looks at its relay voucher, which is renewed
/// once half spent.
const RENEW_EVERY: Duration = Duration::from_secs(3600);

/// The most a request's head may be before it is refused, with `--with-key`,
/// which reads it.
const HEAD_LIMIT: usize = 64 * 1024;

pub fn run(args: &Args) -> Res<()> {
    let url = server(args)?;
    let label = client::tunnel_address(&url)?.label();
    let key = match args.with_key {
        true => Some(device_key(&label)?),
        false => None,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("kvad-tunnel")
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let port = args.port.unwrap_or(DEFAULT_PORT);
        let listener = TcpListener::bind(("127.0.0.1", port)).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::AddrInUse => format!("127.0.0.1:{port} is in use: pick another with --port N"),
            _ => format!("listening on 127.0.0.1:{port}: {e}"),
        })?;
        let port = listener.local_addr()?.port();
        let link = Link { key: client::device_key_path(), url, label: label.clone() };
        // Dialled before anything is said to be listening: a device that is
        // not paired, or a server that refuses it, is an error now rather
        // than a 503 to whatever connects first.
        let first = link.dial().await?;
        let (tx, live) = watch::channel(None);
        println!("forwarding http://127.0.0.1:{port} to {label}");
        match &key {
            None => {
                println!("  each program sends a key of its own, made with");
                println!("    kvad keys add NAME --remote {label}");
            }
            Some(_) => {
                eprintln!("warning: --with-key adds this device's key to every request through 127.0.0.1:{port}.");
                eprintln!("  Every program on this machine, and every user of it, acts as this device on");
                eprintln!("    {label}");
                eprintln!("  with its rights, until this stops. Requests from browser pages are refused.");
            }
        }
        println!("  OpenAI clients: base URL http://127.0.0.1:{port}/v1 · Ctrl-C to stop");
        tokio::spawn(keep(link, Some(first), tx));
        let forward = Forward { live: live.clone(), key, port, label };
        tokio::select! {
            _ = forward.serve(listener) => {}
            _ = tokio::signal::ctrl_c() => {}
        }
        // Says goodbye rather than leave the server to time the connection
        // out; bounded, so that a dead link cannot hold the exit.
        let last = live.borrow().clone();
        if let Some(client) = last {
            let _ = tokio::time::timeout(Duration::from_secs(1), client.close()).await;
        }
        Ok(())
    })
}

/// The server to forward to: `--remote`, then `KVAD_URL` or `client.url`
/// where either is a `kvad://` one, then the one server this device is
/// paired with, if there is only one.
fn server(args: &Args) -> Res<String> {
    match &args.choice {
        Choice::Remote(url) if url.trim().starts_with("kvad://") => return Ok(url.trim().to_string()),
        Choice::Remote(url) => {
            return Err(format!("{url} is not a kvad:// server. A tunnel is to a server this device is paired with; an http:// one is a port already").into())
        }
        Choice::Local => return Err("--local is this process, and a tunnel is to a server: --remote kvad://<node id>".into()),
        Choice::Unset => {}
    }
    let configured = std::env::var(client::URL_VAR).ok().or_else(|| client::Settings::read().ok()?.client_url);
    if let Some(url) = configured.filter(|u| u.trim().starts_with("kvad://")) {
        return Ok(url.trim().to_string());
    }
    match client::paired().as_slice() {
        [one] => Ok(one.clone()),
        [] => Err("this device is paired with no server: `kvad remote join LINK`, or give --remote kvad://<node id>".into()),
        many => Err(format!("this device is paired with {} servers; say which with --remote: {}", many.len(), many.join(", ")).into()),
    }
}

/// The key `--with-key` adds: `KVAD_API_KEY` if it is set, as for every
/// other command, else the one kept when this device was paired.
fn device_key(label: &str) -> Res<String> {
    if let Some(key) = std::env::var(client::KEY_VAR).ok().filter(|k| !k.trim().is_empty()) {
        return Ok(key.trim().to_string());
    }
    match client::credential(label) {
        Some(c) => Ok(c.key),
        None => Err(format!("--with-key: no key is kept for {label}; pair with `kvad remote join LINK` first").into()),
    }
}

// ---------------------------------------------------------------------------
// Keeping a connection up
// ---------------------------------------------------------------------------

/// What a connection is made from, read again for each one: the address and
/// the voucher are the ones kept now, which another kvad may have renewed.
struct Link {
    key: PathBuf,
    url: String,
    label: String,
}

impl Link {
    async fn dial(&self) -> Result<Client, String> {
        let address = client::tunnel_address(&self.url)?;
        let voucher = client::credential(&self.label).and_then(|c| c.voucher).and_then(|v| v.parse::<Voucher>().ok());
        let (key, _) = Key::load(&self.key)?;
        Client::connect_vouched(&key, &address, voucher.as_ref()).await
    }
}

/// Keeps a connection to the server up for as long as the process runs,
/// published on `live` while it is up: says which path it is on, and again
/// whenever that changes, and when it drops makes another, waiting longer
/// between attempts the longer the server does not answer.
async fn keep(link: Link, first: Option<Client>, live: watch::Sender<Option<Arc<Client>>>) {
    let mut next = first;
    let mut wait = Duration::from_secs(1);
    loop {
        let client = match next.take() {
            Some(c) => c,
            None => match link.dial().await {
                Ok(c) => c,
                Err(e) => {
                    println!("{} {} cannot be reached: {e}; trying again in {} s", clock(), link.label, wait.as_secs());
                    tokio::time::sleep(wait).await;
                    wait = (wait * 2).min(MOST_WAIT);
                    continue;
                }
            },
        };
        wait = Duration::from_secs(1);
        let client = Arc::new(client);
        let _ = live.send(Some(client.clone()));
        let why = watch_path(&client, &link).await;
        let _ = live.send(None);
        println!("{} the tunnel to {} dropped ({why}); connecting again", clock(), link.label);
    }
}

/// Says the path whenever it changes, renews the voucher when it is due,
/// and returns why the connection ended.
async fn watch_path(client: &Client, link: &Link) -> String {
    let closed = client.closed();
    tokio::pin!(closed);
    let mut shown: Option<Path> = None;
    let mut renewed: Option<Instant> = None;
    loop {
        if let Some(path) = client.path().filter(|p| shown.as_ref() != Some(p)) {
            println!("{} path: {path}", clock());
            shown = Some(path);
        }
        if renewed.is_none_or(|t| t.elapsed() > RENEW_EVERY) {
            renew(client, &link.label).await;
            renewed = Some(Instant::now());
        }
        tokio::select! {
            why = &mut closed => return why,
            _ = tokio::time::sleep(Duration::from_millis(500)) => {}
        }
    }
}

/// Swaps a half-spent relay voucher for a fresh one, as `Remote::renew`
/// does for the CLI's commands. Best effort: a device whose voucher runs out
/// still reaches its server on the LAN, and gets a new one there.
async fn renew(client: &Client, label: &str) {
    let Some(mut kept) = client::credential(label) else { return };
    let half_spent = kept.voucher.as_deref().and_then(|v| v.parse::<Voucher>().ok()).is_some_and(|v| v.left() < kvad_tunnel::VOUCHER_LIFE / 2);
    if !half_spent {
        return;
    }
    let asked = tokio::time::timeout(Duration::from_secs(15), get(client, "/api/remote/voucher", &kept.key)).await;
    let Some(answer) = asked.ok().and_then(Result::ok) else { return };
    let Some(fresh) = answer["voucher"].as_str() else { return };
    kept.voucher = Some(fresh.to_string());
    if client::remember(label, &kept).is_ok() {
        println!("{} renewed this device's relay voucher", clock());
    }
}

/// One GET through the tunnel, and its JSON. Enough HTTP for the one route
/// this asks for, whose answer has a length.
async fn get(client: &Client, path: &str, key: &str) -> Result<serde_json::Value, String> {
    let mut stream = client.open().await?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: kvad\r\nAuthorization: Bearer {key}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.map_err(|e| e.to_string())?;
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.map_err(|e| e.to_string())?;
    let end = find(&reply, b"\r\n\r\n").ok_or("no reply")?;
    if !reply.starts_with(b"HTTP/1.1 200") {
        return Err(String::from_utf8_lossy(&reply[..end]).into_owned());
    }
    serde_json::from_slice(&reply[end + 4..]).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Forwarding
// ---------------------------------------------------------------------------

/// The listener's side: each connection it accepts, put through whichever
/// connection to the server is up.
#[derive(Clone)]
struct Forward {
    live: watch::Receiver<Option<Arc<Client>>>,
    /// `--with-key`: the key added to every request.
    key: Option<String>,
    /// The port listened on, which a request's `Host` must name.
    port: u16,
    label: String,
}

impl Forward {
    async fn serve(self, listener: TcpListener) {
        loop {
            match listener.accept().await {
                Ok((tcp, _)) => {
                    let forward = self.clone();
                    tokio::spawn(async move {
                        let _ = forward.one(tcp).await;
                    });
                }
                // Out of descriptors, most likely: a pause, not a spin.
                Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    }

    async fn one(&self, mut tcp: TcpStream) -> std::io::Result<()> {
        let _ = tcp.set_nodelay(true);
        // With a key, the head is read and checked before the server is asked
        // for anything, so that a refused request never reaches it.
        let head = match &self.key {
            None => None,
            Some(key) => {
                let (head, rest) = match read_head(&mut tcp).await? {
                    Head::Whole(head, rest) => (head, rest),
                    Head::TooLarge => return reply(&mut tcp, 431, "the request's head is too large").await,
                    Head::Closed => return Ok(()),
                };
                match inject(&head, key, self.port) {
                    Ok(mut head) => {
                        head.extend_from_slice(&rest);
                        Some(head)
                    }
                    Err((status, why)) => return reply(&mut tcp, status, why).await,
                }
            }
        };
        let mut live = self.live.clone();
        let up = tokio::time::timeout(WAIT_FOR_LINK, async { live.wait_for(Option::is_some).await.ok().and_then(|c| c.clone()) }).await;
        let Some(client) = up.ok().flatten() else {
            let why = format!("{} cannot be reached just now; kvad tunnel keeps trying", self.label);
            return reply(&mut tcp, 503, &why).await;
        };
        let mut stream = match client.open().await {
            Ok(s) => s,
            Err(e) => return reply(&mut tcp, 502, &e).await,
        };
        if let Some(head) = head {
            stream.write_all(&head).await?;
        }
        tokio::io::copy_bidirectional(&mut tcp, &mut stream).await?;
        Ok(())
    }
}

/// What [`read_head`] found.
enum Head {
    /// The head, up to and with its blank line, and whatever was read after it.
    Whole(Vec<u8>, Vec<u8>),
    /// Over [`HEAD_LIMIT`] with no end in sight.
    TooLarge,
    /// The connection closed before the head was whole.
    Closed,
}

async fn read_head(tcp: &mut TcpStream) -> std::io::Result<Head> {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    loop {
        if let Some(end) = find(&buf, b"\r\n\r\n") {
            let rest = buf.split_off(end + 4);
            return Ok(Head::Whole(buf, rest));
        }
        if buf.len() > HEAD_LIMIT {
            return Ok(Head::TooLarge);
        }
        let n = tcp.read(&mut chunk).await?;
        if n == 0 {
            return Ok(Head::Closed);
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// A request's head with the device's key in it, or the status and the
/// sentence it is refused with.
///
/// Whatever `Authorization` the client sent is dropped for the device's. The
/// connection is closed after the reply (`Connection: close`), so that the
/// next request on it cannot go through unread; a WebSocket's upgrade is
/// kept, after which there is no next request. A request a browser sent
/// from another page is refused: by its `Origin` or `Sec-Fetch-Site`, and by
/// a `Host` that is not this port, which is what a page that has had its
/// name pointed at 127.0.0.1 sends.
fn inject(head: &[u8], key: &str, port: u16) -> Result<Vec<u8>, (u16, &'static str)> {
    let text = std::str::from_utf8(head).map_err(|_| (400, "the request's head is not text"))?;
    let mut lines = text.strip_suffix("\r\n\r\n").ok_or((400, "the request's head does not end"))?.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let ours = [format!("127.0.0.1:{port}"), format!("localhost:{port}"), format!("[::1]:{port}")];
    let mut out = format!("{request_line}\r\n");
    let mut upgrade = false;
    for line in lines {
        if line.contains(['\r', '\n']) {
            return Err((400, "a header line has a stray line break in it"));
        }
        let (name, value) = line.split_once(':').ok_or((400, "a header line has no colon"))?;
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "host" if !ours.iter().any(|h| h.eq_ignore_ascii_case(value)) => {
                return Err((403, "the request is for another host, and this port adds the device's key: refused"))
            }
            "origin" if !ours.iter().any(|h| value.eq_ignore_ascii_case(&format!("http://{h}"))) => {
                return Err((403, "a web page sent this, and this port adds the device's key: refused"))
            }
            "sec-fetch-site" if !matches!(value, "same-origin" | "none") => {
                return Err((403, "a web page sent this, and this port adds the device's key: refused"))
            }
            "authorization" | "connection" | "keep-alive" | "proxy-connection" => continue,
            "upgrade" => upgrade = true,
            _ => {}
        }
        out.push_str(line);
        out.push_str("\r\n");
    }
    out.push_str(if upgrade { "Connection: upgrade\r\n" } else { "Connection: close\r\n" });
    out.push_str(&format!("Authorization: Bearer {key}\r\n\r\n"));
    Ok(out.into_bytes())
}

/// An answer from the forwarder itself, when the server is not asked.
async fn reply(tcp: &mut TcpStream, status: u16, why: &str) -> std::io::Result<()> {
    let reason = match status {
        400 => "Bad Request",
        403 => "Forbidden",
        431 => "Request Header Fields Too Large",
        502 => "Bad Gateway",
        _ => "Service Unavailable",
    };
    let body = serde_json::json!({ "error": format!("kvad tunnel: {why}") }).to_string();
    let answer = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tcp.write_all(answer.as_bytes()).await?;
    tcp.shutdown().await?;
    // The request may not have been read, and a socket closed with bytes
    // unread is reset, which can lose the answer before the client reads it.
    let drain = async {
        let mut sink = [0u8; 4096];
        while tcp.read(&mut sink).await? > 0 {}
        std::io::Result::Ok(())
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), drain).await;
    Ok(())
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// The time of day, for lines a long-running command prints.
fn clock() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as libc::time_t;
    // SAFETY: `localtime_r` writes only into `tm`, which outlives the call.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&now, &mut tm) }.is_null() {
        return String::new();
    }
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad_tunnel::{Allowed, Direct, Server, ServerConfig};
    use std::io::{Read, Write};

    fn head(s: &str) -> Result<String, (u16, &'static str)> {
        inject(s.as_bytes(), "dev-key", 5824).map(|h| String::from_utf8(h).unwrap())
    }

    #[test]
    fn the_device_key_replaces_the_clients_and_the_connection_ends_with_the_reply() {
        let h = head("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:5824\r\nAuthorization: Bearer theirs\r\nConnection: keep-alive\r\nKeep-Alive: timeout=5\r\nContent-Length: 2\r\n\r\n").unwrap();
        assert_eq!(
            h,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:5824\r\nContent-Length: 2\r\nConnection: close\r\nAuthorization: Bearer dev-key\r\n\r\n"
        );
        // Any spelling of the name.
        assert!(!head("GET / HTTP/1.1\r\nhost: localhost:5824\r\nAUTHORIZATION: Basic eA==\r\n\r\n").unwrap().contains("eA=="));
        // A WebSocket keeps its upgrade.
        let ws = head("GET /v1/realtime HTTP/1.1\r\nHost: [::1]:5824\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n").unwrap();
        assert!(ws.contains("Upgrade: websocket\r\nConnection: upgrade\r\n"), "{ws}");
    }

    #[test]
    fn requests_a_browser_sent_from_elsewhere_are_refused() {
        let refused = |extra: &str| head(&format!("POST /v1/x HTTP/1.1\r\nHost: 127.0.0.1:5824\r\n{extra}\r\n")).map_err(|e| e.0);
        assert_eq!(refused("Origin: https://example.com\r\n"), Err(403));
        assert_eq!(refused("Origin: null\r\n"), Err(403));
        assert_eq!(refused("Sec-Fetch-Site: cross-site\r\n"), Err(403));
        // A page that had its name pointed at 127.0.0.1.
        assert_eq!(head("GET / HTTP/1.1\r\nHost: evil.example:5824\r\n\r\n").map_err(|e| e.0), Err(403));
        assert_eq!(head("GET / HTTP/1.1\r\nHost: 127.0.0.1:9999\r\n\r\n").map_err(|e| e.0), Err(403));
        // The server's own pages, served through this port, and a person at
        // the address bar.
        assert!(refused("Origin: http://127.0.0.1:5824\r\nSec-Fetch-Site: same-origin\r\n").is_ok());
        assert!(refused("Sec-Fetch-Site: none\r\n").is_ok());
        // A line break inside a header would be a second header to the server.
        assert_eq!(refused("X-A: 1\nAuthorization: Bearer theirs\r\n").map_err(|_| 400), Err(400));
    }

    /// A directory of its own for the test's keys, removed at the end.
    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A server with no router behind it: each stream is answered with the
    /// `Authorization` it carried, so that a test can see what arrived.
    fn echo(rt: &tokio::runtime::Runtime, dir: &std::path::Path, allowed: Allowed) -> kvad_tunnel::Address {
        let cfg = ServerConfig { key: dir.join("remote.key"), knock: dir.join("remote.knock"), relays: Vec::new(), direct: Direct::Auto, port: 0, allowed };
        let mut server = rt.block_on(Server::start(&cfg)).unwrap();
        let mut address = server.address().clone();
        // Loopback, which a firewall leaves alone: macOS's holds back a
        // freshly built binary's LAN addresses until somebody allows it.
        address.addrs = address.addrs.iter().filter(|a| a.is_ipv4()).map(|a| std::net::SocketAddr::from(([127, 0, 0, 1], a.port()))).take(1).collect();
        rt.spawn(async move {
            while let Some((_, mut stream)) = server.accept().await {
                tokio::spawn(async move {
                    let mut got = Vec::new();
                    let mut chunk = [0u8; 1024];
                    while find(&got, b"\r\n\r\n").is_none() {
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => got.extend_from_slice(&chunk[..n]),
                        }
                    }
                    let text = String::from_utf8_lossy(&got).into_owned();
                    let auth = text.lines().find_map(|l| l.strip_prefix("Authorization: ")).unwrap_or("none").to_string();
                    let answer = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{auth}", auth.len());
                    let _ = stream.write_all(answer.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        address
    }

    /// One request to the forwarder, and the whole reply.
    fn ask(port: u16, extra: &str) -> String {
        let mut tcp = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        write!(tcp, "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer theirs\r\n{extra}\r\n").unwrap();
        let mut reply = String::new();
        tcp.read_to_string(&mut reply).unwrap();
        reply
    }

    fn wait_until(within: Duration, ok: impl Fn() -> bool) -> bool {
        let t = Instant::now();
        while t.elapsed() < within {
            if ok() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    #[test]
    fn a_request_goes_through_the_tunnel_with_its_key_or_the_devices_and_again_after_a_drop() {
        let dir = std::env::temp_dir().join(format!("kvad-cli-tunnel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _scratch = Scratch(dir.clone());
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let device = dir.join("device.key");
        let (key, _) = Key::load(&device).unwrap();
        let allowed = Allowed::new([key.id()]);
        let address = echo(&rt, &dir, allowed.clone());

        let link = Link { key: device, url: address.to_string(), label: address.label() };
        let (tx, live) = watch::channel(None);
        rt.spawn(keep(link, None, tx));
        let listen = |key: Option<&str>| {
            let listener = rt.block_on(TcpListener::bind(("127.0.0.1", 0))).unwrap();
            let port = listener.local_addr().unwrap().port();
            let forward = Forward { live: live.clone(), key: key.map(str::to_string), port, label: address.label() };
            rt.spawn(forward.serve(listener));
            port
        };
        let (plain, keyed) = (listen(None), listen(Some("dev-key")));
        assert!(wait_until(Duration::from_secs(30), || live.borrow().is_some()), "never connected");

        // As it is: the client's own key, untouched.
        let reply = ask(plain, "");
        assert!(reply.starts_with("HTTP/1.1 200") && reply.ends_with("Bearer theirs"), "{reply}");
        // --with-key: the device's, in place of the client's.
        let reply = ask(keyed, "");
        assert!(reply.starts_with("HTTP/1.1 200") && reply.ends_with("Bearer dev-key"), "{reply}");
        // And a browser page's request never reaches the server.
        let reply = ask(keyed, "Origin: https://example.com\r\n");
        assert!(reply.starts_with("HTTP/1.1 403") && reply.contains("web page"), "{reply}");

        // The server drops the device, then lets it back: the forwarder
        // notices, and is through again without being restarted.
        assert!(allowed.remove(&key.id()));
        assert!(wait_until(Duration::from_secs(10), || live.borrow().is_none()), "the drop went unnoticed");
        allowed.insert(key.id());
        assert!(wait_until(Duration::from_secs(20), || live.borrow().is_some()), "never connected again");
        let reply = ask(plain, "");
        assert!(reply.starts_with("HTTP/1.1 200") && reply.ends_with("Bearer theirs"), "{reply}");
    }
}
