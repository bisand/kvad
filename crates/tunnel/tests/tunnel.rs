//! The tunnel end to end, on this machine: an axum router behind a
//! [`Server`], reached through [`blocking::Tunnel`] as the CLI reaches it,
//! directly, through a relay in this process, and across the relay's loss.

use kvad_tunnel::{blocking, Address, Allowed, Direct, Key, NodeId, Path, Server, ServerConfig, Voucher};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// A directory of its own for each test's keys, removed at the end.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("kvad-tunnel-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Scratch(dir)
    }
    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap()
}

/// `/hello` says hi; `/slow` writes a numbered line every 100 ms for 4 s, so
/// that something can happen to the network while a reply is under way.
fn router() -> axum::Router {
    use axum::routing::get;
    axum::Router::new().route("/hello", get(|| async { "hi" })).route(
        "/slow",
        get(|| async {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(4);
            tokio::spawn(async move {
                for i in 1..=40 {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    if tx.send(Ok(format!("line {i}\n"))).await.is_err() {
                        return;
                    }
                }
            });
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx))
        }),
    )
}

/// A relay in this process, over plain HTTP on loopback.
async fn relay() -> (iroh_relay::server::Server, String) {
    let mut cfg = iroh_relay::server::ServerConfig::default();
    cfg.relay = Some(iroh_relay::server::RelayConfig::new(std::net::SocketAddr::from(([127, 0, 0, 1], 0))));
    let server = iroh_relay::server::Server::spawn(cfg).await.unwrap();
    let url = format!("http://127.0.0.1:{}", server.http_addr().unwrap().port());
    (server, url)
}

/// A server for `allow`, serving the router, and its address.
fn serve(rt: &tokio::runtime::Runtime, dir: &Scratch, relays: Vec<String>, direct: Direct, allow: Vec<NodeId>) -> Address {
    serve_with(rt, dir, relays, direct, Allowed::new(allow), |_| {}).0
}

/// The same, with the list shared, and the pairing requests handed to
/// `pairs` on a task of their own.
fn serve_with(
    rt: &tokio::runtime::Runtime,
    dir: &Scratch,
    relays: Vec<String>,
    direct: Direct,
    allowed: Allowed,
    pairs: impl Fn(kvad_tunnel::Pairing) + Send + 'static,
) -> (Address, Allowed) {
    let cfg = ServerConfig { key: dir.path("remote.key"), knock: dir.path("remote.knock"), relays, direct, port: 0, allowed: allowed.clone() };
    let mut server = rt.block_on(Server::start(&cfg)).unwrap();
    let address = server.address().clone();
    let mut pairings = server.take_pairings().unwrap();
    rt.spawn(async move {
        while let Some(p) = pairings.recv().await {
            pairs(p);
        }
    });
    rt.spawn(async move { axum::serve(server, router()).await });
    (address, allowed)
}

fn client_key(dir: &Scratch) -> (PathBuf, NodeId) {
    let path = dir.path("client.key");
    let (key, _) = Key::load(&path).unwrap();
    (path, key.id())
}

/// A request over one tunnel stream, and the whole reply.
fn get(tunnel: &blocking::Tunnel, path: &str) -> Result<String, String> {
    let mut s = tunnel.open()?;
    s.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: tunnel\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
    let mut reply = String::new();
    s.read_to_string(&mut reply).map_err(|e| e.to_string())?;
    Ok(reply)
}

fn wait_for(tunnel: &blocking::Tunnel, ok: impl Fn(&Path) -> bool, within: Duration) -> Option<Path> {
    let t = Instant::now();
    while t.elapsed() < within {
        if let Some(p) = tunnel.path().filter(|p| ok(p)) {
            return Some(p);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    tunnel.path()
}

#[test]
fn http_goes_through_directly_with_no_relay() {
    let dir = Scratch::new("direct");
    let rt = runtime();
    let (key, id) = client_key(&dir);
    let address = serve(&rt, &dir, Vec::new(), Direct::Auto, vec![id]);
    assert!(address.relays.is_empty() && !address.addrs.is_empty());
    let tunnel = blocking::Tunnel::connect(&key, &address, None).unwrap_or_else(|e| panic!("{e}; the address was {address}"));
    let reply = get(&tunnel, "/hello").unwrap();
    assert!(reply.starts_with("HTTP/1.1 200") && reply.ends_with("hi"), "{reply}");
    // The LAN, or a tailnet's address if the machine is on one: either way
    // not a relay, since there is none.
    assert!(matches!(tunnel.path(), Some(Path::Lan(_) | Path::Direct(_))), "{:?}", tunnel.path());
    // Several streams over the one connection, as a pooled client opens them.
    for _ in 0..3 {
        assert!(get(&tunnel, "/hello").unwrap().ends_with("hi"));
    }
}

#[test]
fn http_goes_through_a_relay_when_direct_is_off() {
    let dir = Scratch::new("relayed");
    let rt = runtime();
    let (_relay, url) = rt.block_on(relay());
    let (key, id) = client_key(&dir);
    let address = serve(&rt, &dir, vec![url.clone()], Direct::Off, vec![id]);
    assert!(address.addrs.is_empty(), "direct = off advertises no address");
    let tunnel = blocking::Tunnel::connect(&key, &address, None).unwrap();
    assert!(get(&tunnel, "/hello").unwrap().ends_with("hi"));
    assert_eq!(tunnel.path(), Some(Path::Relay(url)));
}

#[test]
fn a_device_not_on_the_list_is_refused() {
    let dir = Scratch::new("refused");
    let rt = runtime();
    let (key, _) = client_key(&dir);
    let (other, _) = Key::load(&dir.path("other.key")).unwrap();
    let address = serve(&rt, &dir, Vec::new(), Direct::Auto, vec![other.id()]);
    let outcome = blocking::Tunnel::connect(&key, &address, None).map(|_| ());
    let e = outcome.expect_err("an unlisted key was let in");
    assert!(e.contains("refused this device"), "{e}");
}

#[test]
fn a_wrong_knock_is_refused() {
    let dir = Scratch::new("knock");
    let rt = runtime();
    let (key, id) = client_key(&dir);
    let mut address = serve(&rt, &dir, Vec::new(), Direct::Auto, vec![id]);
    address.knock = "00000000000000000000000000000000".into();
    let outcome = blocking::Tunnel::connect(&key, &address, None).and_then(|t| get(&t, "/hello"));
    assert!(outcome.is_err(), "a wrong knock got: {outcome:?}");
}

#[test]
fn a_reply_carries_on_when_the_relay_goes() {
    let dir = Scratch::new("relay-loss");
    let rt = runtime();
    let (relay, url) = rt.block_on(relay());
    let (key, id) = client_key(&dir);
    let mut address = serve(&rt, &dir, vec![url], Direct::Auto, vec![id]);
    // Only the relay, as a device that has not yet learnt a direct address
    // would have it: the connection opens there and must find the rest.
    address.addrs.clear();
    let tunnel = blocking::Tunnel::connect(&key, &address, None).unwrap();
    let direct = wait_for(&tunnel, |p| !matches!(p, Path::Relay(_)), Duration::from_secs(10));
    assert!(matches!(direct, Some(Path::Lan(_) | Path::Direct(_))), "never left the relay: {direct:?}");

    let tunnel = std::sync::Arc::new(tunnel);
    let reading = std::thread::spawn({
        let tunnel = tunnel.clone();
        move || get(&tunnel, "/slow")
    });
    std::thread::sleep(Duration::from_millis(800));
    rt.block_on(relay.shutdown()).unwrap();
    let reply = reading.join().unwrap().unwrap();
    assert!(reply.contains("line 1\n") && reply.contains("line 40\n"), "the reply broke off: {reply}");
    // And a new request after it, with no relay left at all.
    assert!(get(&tunnel, "/hello").unwrap().ends_with("hi"));
}

#[test]
fn an_unknown_key_reaches_only_the_pairing_lane_and_only_while_it_is_open() {
    let dir = Scratch::new("pairing");
    let rt = runtime();
    let (key, id) = client_key(&dir);
    let (address, allowed) = serve_with(&rt, &dir, Vec::new(), Direct::Auto, Allowed::default(), |p| {
        let answer = format!("{{\"device\":\"{}\",\"asked\":{}}}", p.device, String::from_utf8_lossy(&p.request));
        p.answer(answer.into_bytes());
    });
    // Closed: refused at the handshake, as any unknown key is.
    let e = blocking::pair(&key, &address, b"\"hello\"", None).unwrap_err();
    assert!(e.contains("refused"), "{e}");

    allowed.open_pairing();
    let answer = String::from_utf8(blocking::pair(&key, &address, b"\"hello\"", None).unwrap()).unwrap();
    assert_eq!(answer, format!("{{\"device\":\"{id}\",\"asked\":\"hello\"}}"), "the answer names the key the handshake proved");
    // Open, but still not paired: the lane is all it can reach.
    let e = blocking::Tunnel::connect(&key, &address, None).map(|_| ()).unwrap_err();
    assert!(e.contains("not paired"), "{e}");

    // Paired: a tunnel now, and no second pairing.
    allowed.insert(id);
    let tunnel = blocking::Tunnel::connect(&key, &address, None).unwrap();
    assert!(get(&tunnel, "/hello").unwrap().ends_with("hi"));
    let e = blocking::pair(&key, &address, b"\"again\"", None).unwrap_err();
    assert!(e.contains("already paired"), "{e}");
}

#[test]
fn a_removed_device_is_cut_off_mid_reply() {
    let dir = Scratch::new("revoked");
    let rt = runtime();
    let (key, id) = client_key(&dir);
    let (address, allowed) = serve_with(&rt, &dir, Vec::new(), Direct::Auto, Allowed::new([id]), |_| {});
    let tunnel = std::sync::Arc::new(blocking::Tunnel::connect(&key, &address, None).unwrap());
    let reading = std::thread::spawn({
        let tunnel = tunnel.clone();
        move || get(&tunnel, "/slow")
    });
    std::thread::sleep(Duration::from_millis(600));
    assert!(allowed.remove(&id));
    let reply = reading.join().unwrap();
    assert!(reply.as_ref().map_or(true, |r| !r.contains("line 40\n")), "a removed device read the whole reply: {reply:?}");
    let e = blocking::Tunnel::connect(&key, &address, None).map(|_| ()).unwrap_err();
    assert!(e.contains("refused"), "{e}");
}

/// kvad-relay's rule, in a relay in this process: the server it serves, and
/// a device only with that server's voucher for it.
#[derive(Debug)]
struct Vouched(NodeId);

impl iroh_relay::server::AccessControl for Vouched {
    async fn on_connect(&self, r: &iroh_relay::server::ClientRequest) -> iroh_relay::server::Access {
        let device: NodeId = r.endpoint_id().to_string().parse().unwrap();
        let vouched = r.auth_token().and_then(|t| t.parse::<Voucher>().ok()).is_some_and(|v| v.check(&device, |n| *n == self.0).is_ok());
        match device == self.0 || vouched {
            true => iroh_relay::server::Access::Allow,
            false => iroh_relay::server::Access::Deny { reason: None },
        }
    }
}

#[test]
fn a_device_on_no_relays_list_gets_through_with_a_voucher_and_not_without() {
    let dir = Scratch::new("vouched");
    let rt = runtime();
    let server_key = dir.path("remote.key");
    let (key, _) = Key::load(&server_key).unwrap();
    let mut cfg = iroh_relay::server::ServerConfig::default();
    let mut relay_cfg = iroh_relay::server::RelayConfig::new(std::net::SocketAddr::from(([127, 0, 0, 1], 0)));
    relay_cfg.access = std::sync::Arc::new(Vouched(key.id()));
    cfg.relay = Some(relay_cfg);
    let relay = rt.block_on(iroh_relay::server::Server::spawn(cfg)).unwrap();
    let url = format!("http://127.0.0.1:{}", relay.http_addr().unwrap().port());

    let (phone_key, phone) = client_key(&dir);
    let (newcomer_key, _) = (dir.path("newcomer.key"), Key::load(&dir.path("newcomer.key")).unwrap().0.id());
    let allowed = Allowed::new([phone]);
    let cfg = ServerConfig { key: server_key, knock: dir.path("remote.knock"), relays: vec![url], direct: Direct::Off, port: 0, allowed: allowed.clone() };
    let mut server = rt.block_on(Server::start(&cfg)).unwrap();
    let (address, issuer) = (server.address().clone(), server.issuer());
    let mut pairings = server.take_pairings().unwrap();
    rt.spawn(async move {
        while let Some(p) = pairings.recv().await {
            p.answer(b"{\"paired\":1}".to_vec());
        }
    });
    rt.spawn(async move { axum::serve(server, router()).await });

    // Vouched: through the relay, which has never heard of it.
    let voucher = issuer.issue(Some(phone), kvad_tunnel::VOUCHER_LIFE);
    let tunnel = blocking::Tunnel::connect(&phone_key, &address, Some(&voucher)).unwrap();
    assert!(get(&tunnel, "/hello").unwrap().ends_with("hi"));
    assert!(matches!(tunnel.path(), Some(Path::Relay(_))));

    // A pairing link's voucher, for any device, gets a newcomer to the lane.
    allowed.open_pairing();
    let any = issuer.issue(None, kvad_tunnel::ANY_DEVICE_LIFE);
    assert_eq!(blocking::pair(&newcomer_key, &address, b"{}", Some(&any)).unwrap(), b"{\"paired\":1}");

    // Without one, the relay does not carry it at all.
    let refused = blocking::Tunnel::connect(&phone_key, &address, None).map(|_| ());
    assert!(refused.is_err(), "a device with no voucher got through the relay");
}

/// The plan's rule, held to in the lockfile: nothing that asks a router for
/// a port is anywhere in the build.
#[test]
fn nothing_in_the_build_maps_router_ports() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock")).unwrap();
    for name in ["portmapper", "igd", "igd-next", "natpmp", "rupnp", "easy-upnp"] {
        assert!(!lock.contains(&format!("name = \"{name}\"")), "{name} is in Cargo.lock");
    }
}
