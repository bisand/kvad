//! Reaching this server from anywhere, and pairing the devices that may:
//! Phase 2 of `docs/relay-plan.md`.
//!
//! The tunnel itself is `kvad-tunnel`'s; this is what kvad-serve decides
//! about it:
//!
//! - **Who is let in.** [`Remote::allowed`] is every paired device in the
//!   `devices` table, and the ids in `[remote] allow`. A device removed is
//!   shut out at once, its live connection closed.
//! - **Pairing.** `POST /api/remote/pair` opens a slot: a single-use token
//!   that lives ten minutes, shown as a link and a QR code. While one is
//!   open, the tunnel lets an unknown key reach its pairing lane, one
//!   request and one answer; a request that presents the token pairs that
//!   key to the slot's account. The token is burned on first use, right or
//!   wrong, and slots are kept in memory only, so a restart burns them too.
//! - **What a device may do.** Its key works only from that device, through
//!   the tunnel ([`Tunnelled`]), and as `user` unless the slot granted
//!   `admin`. A device cannot open a slot for another unless it has `admin`.

use crate::api::{blocking, Fail};
use crate::auth::{Identity, State};
use crate::devices::{self, Device};
use crate::secret;
use axum::extract::{Path, State as St};
use axum::http::StatusCode;
use axum::Json;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a pairing slot is open.
pub const SLOT_LIFE: Duration = Duration::from_secs(10 * 60);

/// How long a used token is remembered, so that presenting it again is told
/// "already used" rather than "unknown".
const BURNT_FOR: Duration = Duration::from_secs(60 * 60);

/// What a request that came through the tunnel came from: the device's
/// node id, as its handshake proved it. Absent on a request that came in on
/// the TCP port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tunnelled(pub kvad_tunnel::NodeId);

impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, kvad_tunnel::Server>> for Tunnelled {
    fn connect_info(stream: axum::serve::IncomingStream<'_, kvad_tunnel::Server>) -> Self {
        Tunnelled(*stream.remote_addr())
    }
}

struct Slot {
    user: i64,
    admin: bool,
    opened: Instant,
}

/// The tunnel's state, as kvad-serve holds it.
pub struct Remote {
    pub allowed: kvad_tunnel::Allowed,
    /// `[remote] allow`, which stays allowed whatever the table says.
    configured: Vec<kvad_tunnel::NodeId>,
    address: kvad_tunnel::Address,
    /// The node key, for the vouchers that get devices through relays.
    issuer: kvad_tunnel::Issuer,
    /// Open slots by their token's fingerprint; the token itself is never kept.
    slots: Mutex<HashMap<String, Slot>>,
    burnt: Mutex<HashMap<String, Instant>>,
}

/// Why a pairing was refused, in the words the device is shown.
#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    Unknown,
    Used,
    Expired,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Refusal::Unknown => "this server did not make that pairing link",
            Refusal::Used => "that pairing link has been used: make a new one with `kvad remote pair`",
            Refusal::Expired => "that pairing link has expired: make a new one with `kvad remote pair`",
        })
    }
}

impl Remote {
    pub fn new(configured: Vec<kvad_tunnel::NodeId>, address: kvad_tunnel::Address, allowed: kvad_tunnel::Allowed, issuer: kvad_tunnel::Issuer) -> Remote {
        Remote { allowed, configured, address, issuer, slots: Mutex::default(), burnt: Mutex::default() }
    }

    /// A voucher for a paired device: what gets it through the relays that
    /// serve this server, from wherever it is.
    pub fn voucher(&self, device: kvad_tunnel::NodeId) -> kvad_tunnel::Voucher {
        self.issuer.issue(Some(device), kvad_tunnel::VOUCHER_LIFE)
    }

    pub fn address(&self) -> &kvad_tunnel::Address {
        &self.address
    }

    /// Lets in exactly the paired devices and the configured ids.
    pub fn reload(&self, db: &crate::db::Db) -> Result<(), Box<dyn std::error::Error>> {
        let mut ids = self.configured.clone();
        for node in devices::nodes(db)? {
            if let Ok(id) = node.parse() {
                ids.push(id);
            }
        }
        self.allowed.set(ids);
        Ok(())
    }

    /// Opens a slot for `user`; the token is returned once and only its
    /// fingerprint is kept.
    pub fn open(&self, user: i64, admin: bool) -> String {
        self.sweep();
        let token = secret::token();
        self.slots.lock().unwrap().insert(secret::fingerprint(&token), Slot { user, admin, opened: Instant::now() });
        self.allowed.open_pairing();
        token
    }

    /// Closes the slots that have run out, so that the tunnel stops letting
    /// unknown keys in once none is left.
    pub fn sweep(&self) {
        let mut slots = self.slots.lock().unwrap();
        let before = slots.len();
        slots.retain(|_, s| s.opened.elapsed() < SLOT_LIFE);
        for _ in slots.len()..before {
            self.allowed.close_pairing();
        }
        self.burnt.lock().unwrap().retain(|_, at| at.elapsed() < BURNT_FOR);
    }

    /// Takes the slot `token` opened, burning the token whether or not it is
    /// still good.
    fn take(&self, token: &str) -> Result<Slot, Refusal> {
        let fingerprint = secret::fingerprint(token);
        let slot = self.slots.lock().unwrap().remove(&fingerprint);
        if self.burnt.lock().unwrap().insert(fingerprint, Instant::now()).is_some() {
            return Err(Refusal::Used);
        }
        let Some(slot) = slot else { return Err(Refusal::Unknown) };
        self.allowed.close_pairing();
        match slot.opened.elapsed() < SLOT_LIFE {
            true => Ok(slot),
            false => Err(Refusal::Expired),
        }
    }

    /// Answers one pairing request from `node`: pairs it, or says why not.
    pub fn answer(&self, db: &crate::db::Db, node: kvad_tunnel::NodeId, request: &[u8]) -> serde_json::Value {
        #[derive(serde::Deserialize)]
        struct Request {
            v: u32,
            token: String,
            #[serde(default)]
            name: String,
        }
        let Ok(request) = serde_json::from_slice::<Request>(request) else {
            return json!({ "error": "that was not a pairing request" });
        };
        if request.v != 1 {
            return json!({ "error": format!("pairing version {} is not one this server speaks (1)", request.v) });
        }
        let slot = match self.take(&request.token) {
            Ok(slot) => slot,
            Err(why) => {
                tracing::info!("remote: refused to pair {}: {why}", node.short());
                return json!({ "error": why.to_string() });
            }
        };
        match devices::pair(db, &node.to_string(), &request.name, slot.user, slot.admin) {
            Ok((device, key)) => {
                self.allowed.insert(node);
                let voucher = self.voucher(node);
                tracing::info!("remote: paired {} as \"{}\" for {}", node.short(), device.name, device.owner);
                json!({
                    "paired": device.id,
                    "name": device.name,
                    "account": device.owner,
                    "role": if device.admin { "admin" } else { "user" },
                    "key": key,
                    "voucher": voucher.to_string(),
                })
            }
            Err(e) => json!({ "error": e.to_string() }),
        }
    }

    /// Answers pairing requests as the tunnel hands them over, until it
    /// stops.
    pub async fn serve_pairings(self: Arc<Self>, db: crate::db::Db, mut pairings: tokio::sync::mpsc::Receiver<kvad_tunnel::Pairing>) {
        let mut sweep = tokio::time::interval(Duration::from_secs(15));
        loop {
            tokio::select! {
                p = pairings.recv() => {
                    let Some(p) = p else { return };
                    let (remote, db) = (self.clone(), db.clone());
                    let answer = tokio::task::spawn_blocking(move || {
                        let answer = remote.answer(&db, p.device, &p.request);
                        (p, answer)
                    })
                    .await;
                    if let Ok((p, answer)) = answer {
                        p.answer(answer.to_string().into_bytes());
                    }
                }
                _ = sweep.tick() => self.sweep(),
            }
        }
    }
}

fn remote(state: &State) -> Result<Arc<Remote>, Fail> {
    state.remote.clone().ok_or_else(|| Fail(StatusCode::CONFLICT, "this server has no [remote] section, so nothing can pair with it".into(), None))
}

#[derive(serde::Deserialize, Default)]
pub struct PairAsk {
    /// Grant the device the account's administrator role. Only an
    /// administrator can ask for it.
    #[serde(default)]
    admin: bool,
}

/// `POST /api/remote/pair`: a slot, and the link that uses it.
pub async fn pair(who: Identity, St(state): St<State>, body: Option<Json<PairAsk>>) -> Result<Json<serde_json::Value>, Fail> {
    let remote = remote(&state)?;
    let Json(ask) = body.unwrap_or_default();
    let Some(user) = who.id else {
        return Err(Fail::bad("a device belongs to an account, and this server has none"));
    };
    if who.device.is_some() && !who.is_admin() {
        return Err(Fail(StatusCode::FORBIDDEN, "a device cannot pair other devices unless it was paired with admin".into(), None));
    }
    if ask.admin && !who.is_admin() {
        return Err(Fail(StatusCode::FORBIDDEN, "only an administrator can pair a device as one".into(), None));
    }
    let token = remote.open(user, ask.admin);
    // For any device, for the slot's life: a device that is not on the LAN
    // pairs through the relay with it.
    let voucher = remote.issuer.issue(None, SLOT_LIFE);
    let link = kvad_tunnel::PairLink { address: remote.address().clone(), token, voucher: Some(voucher) };
    Ok(Json(json!({
        "link": link.to_string(),
        "server": remote.address().label(),
        "expires_in": SLOT_LIFE.as_secs(),
        "role": if ask.admin { "admin" } else { "user" },
    })))
}

/// `GET /api/remote/voucher`: a fresh voucher for the device asking, which
/// the CLI fetches on its own when the one it holds is half spent.
pub async fn voucher(who: Identity, St(state): St<State>) -> Result<Json<serde_json::Value>, Fail> {
    let remote = remote(&state)?;
    let id = who.device.ok_or_else(|| Fail::bad("a voucher is for a paired device, and this request did not come from one"))?;
    let db = state.db.clone();
    let device = blocking(move || devices::get(&db, id)).await?.ok_or_else(|| Fail::missing("no such device"))?;
    let node = device.node.parse().map_err(|_| Fail::bad("the device's node id is not one"))?;
    let v = remote.voucher(node);
    Ok(Json(json!({ "voucher": v.to_string(), "expires_in": v.left().as_secs() })))
}

/// `GET /api/remote`: whether the tunnel is on, and as what.
pub async fn status(_: Identity, St(state): St<State>) -> Json<serde_json::Value> {
    match &state.remote {
        None => Json(json!({ "enabled": false })),
        Some(r) => Json(json!({ "enabled": true, "server": r.address().label(), "allowed": r.allowed.len() })),
    }
}

/// `GET /api/devices`: your devices, or with `?all=true` an administrator's
/// view of everyone's.
pub async fn list(who: Identity, St(state): St<State>, axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>) -> Result<Json<Vec<Device>>, Fail> {
    let all = q.get("all").is_some_and(|v| v == "true" || v == "1");
    if all && !who.is_admin() {
        return Err(Fail(StatusCode::FORBIDDEN, "only an administrator can see every account's devices".into(), None));
    }
    let user = match all {
        true => None,
        false => Some(who.id.ok_or_else(|| Fail::bad("this server has no accounts, so no devices"))?),
    };
    let db = state.db.clone();
    blocking(move || devices::list(&db, user)).await.map(Json)
}

/// `DELETE /api/devices/{id}`: unpairs a device, revokes its key, and closes
/// its connection.
pub async fn remove(who: Identity, St(state): St<State>, Path(id): Path<i64>) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let device = blocking(move || devices::get(&db, id)).await?.ok_or_else(|| Fail::missing("no such device"))?;
    if Some(device.user) != who.id && !who.is_admin() {
        return Err(Fail::missing("no such device"));
    }
    if who.device == Some(device.id) {
        // Allowed: a device can unpair itself. Said so in the log.
        tracing::info!("remote: {} unpairs itself", device.name);
    } else if who.device.is_some() && !who.is_admin() {
        return Err(Fail(StatusCode::FORBIDDEN, "a device cannot unpair other devices unless it was paired with admin".into(), None));
    }
    let db = state.db.clone();
    blocking(move || devices::remove(&db, id)).await?;
    if let (Some(r), Ok(node)) = (&state.remote, device.node.parse()) {
        r.allowed.remove(&node);
    }
    tracing::info!("remote: unpaired \"{}\" ({})", device.name, &device.node[..10]);
    Ok(Json(json!({ "removed": id })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use crate::users;

    /// Where this test keeps its keys. libtest names each test's thread after the test, so tests running in
    /// parallel never share a key file, while two helper calls inside one test find the same keys.
    fn keys() -> std::path::PathBuf {
        let name = std::thread::current().name().unwrap_or("unnamed").replace(':', "_");
        std::env::temp_dir().join(format!("kvad-remote-test-{}", std::process::id())).join(name)
    }

    fn remote() -> Remote {
        let (key, _) = kvad_tunnel::Key::load(&keys().join("k")).unwrap();
        let address = kvad_tunnel::Address { node: key.id(), knock: "0".repeat(32), relays: vec!["https://relay.kvad.eu".into()], addrs: Vec::new() };
        Remote::new(Vec::new(), address, kvad_tunnel::Allowed::default(), key.issuer())
    }

    fn node(n: u8) -> kvad_tunnel::NodeId {
        kvad_tunnel::Key::load(&keys().join(format!("d{n}"))).unwrap().0.id()
    }

    fn ask(token: &str, name: &str) -> Vec<u8> {
        json!({ "v": 1, "token": token, "name": name }).to_string().into_bytes()
    }

    #[test]
    fn a_good_pairing_makes_a_device_the_tunnel_lets_in() {
        let db = Db::in_memory().unwrap();
        let ada = users::create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
        let r = remote();
        let token = r.open(ada.id, false);
        assert!(r.allowed.pairing(), "an open slot lets unknown keys reach the lane");
        let answer = r.answer(&db, node(1), &ask(&token, "phone"));
        assert_eq!(answer["role"], "user", "{answer}");
        assert_eq!(answer["account"], "ada");
        assert!(answer["key"].as_str().is_some_and(|k| k.len() == 64));
        let voucher: kvad_tunnel::Voucher = answer["voucher"].as_str().unwrap().parse().unwrap();
        assert!(voucher.check(&node(1), |n| *n == r.address().node).is_ok(), "the device gets a voucher for itself");
        assert!(voucher.check(&node(2), |n| *n == r.address().node).is_err());
        assert!(r.allowed.contains(&node(1)));
        assert!(!r.allowed.pairing(), "the slot is spent, and the lane closed with it");
    }

    #[test]
    fn a_token_is_good_once_and_only_while_it_lives() {
        let db = Db::in_memory().unwrap();
        let ada = users::create(&db, "ada", Some("lovelace-1843"), "user", None).unwrap();
        let r = remote();
        let token = r.open(ada.id, false);
        assert!(r.answer(&db, node(2), &ask(&token, "phone")).get("key").is_some());
        let again = r.answer(&db, node(3), &ask(&token, "other"));
        assert!(again["error"].as_str().unwrap().contains("has been used"), "{again}");
        assert!(!r.allowed.contains(&node(3)));

        let wrong = r.answer(&db, node(3), &ask(&secret::token(), "other"));
        assert!(wrong["error"].as_str().unwrap().contains("did not make"), "{wrong}");

        let token = r.open(ada.id, false);
        r.slots.lock().unwrap().values_mut().for_each(|s| s.opened -= SLOT_LIFE);
        let late = r.answer(&db, node(4), &ask(&token, "late"));
        assert!(late["error"].as_str().unwrap().contains("expired"), "{late}");
        assert!(!r.allowed.pairing());

        // A token tried and refused is burnt too: a second try with it says so.
        let token = r.open(ada.id, false);
        r.slots.lock().unwrap().values_mut().for_each(|s| s.opened -= SLOT_LIFE);
        r.answer(&db, node(5), &ask(&token, "x"));
        assert!(r.answer(&db, node(5), &ask(&token, "x"))["error"].as_str().unwrap().contains("has been used"));
    }

    #[test]
    fn an_expired_slot_closes_the_lane_on_its_own() {
        let r = remote();
        r.open(1, false);
        assert!(r.allowed.pairing());
        r.slots.lock().unwrap().values_mut().for_each(|s| s.opened -= SLOT_LIFE);
        r.sweep();
        assert!(!r.allowed.pairing());
    }

    #[test]
    fn an_admin_slot_grants_admin_and_a_plain_one_does_not() {
        let db = Db::in_memory().unwrap();
        let ada = users::create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
        let r = remote();
        let granted = r.answer(&db, node(6), &ask(&r.open(ada.id, true), "laptop"));
        assert_eq!(granted["role"], "admin");
        let plain = r.answer(&db, node(7), &ask(&r.open(ada.id, false), "phone"));
        assert_eq!(plain["role"], "user");
    }

    #[test]
    fn garbage_and_other_versions_are_refused() {
        let db = Db::in_memory().unwrap();
        let r = remote();
        assert!(r.answer(&db, node(8), b"nonsense")["error"].is_string());
        let v2 = json!({ "v": 2, "token": "x", "name": "y" }).to_string();
        assert!(r.answer(&db, node(8), v2.as_bytes())["error"].as_str().unwrap().contains("version 2"));
    }
}
