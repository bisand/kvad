//! The servers this phone is paired with, and its device key: kept in the
//! Keychain (`secrets`), read when the app starts, written when they change.
//!
//! For each server, by its `kvad://<node id>`: its whole address, the API key
//! it made for this phone, the relay voucher that gets the phone through its
//! relays, the account, and the device's id there, for unpairing.

use serde_json::{json, Map, Value};

pub type Res<T> = Result<T, String>;

#[derive(Clone, Debug, PartialEq)]
pub struct Pairing {
    pub address: String,
    pub key: String,
    pub voucher: Option<String>,
    pub account: String,
    pub role: String,
    pub device: Option<i64>,
}

pub struct Store {
    device: Option<[u8; 32]>,
    servers: Map<String, Value>,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

impl Store {
    pub fn load() -> Store {
        let v: Value = crate::secrets::get().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        Store {
            device: v.get("device").and_then(Value::as_str).and_then(unhex32),
            servers: v.get("servers").and_then(Value::as_object).cloned().unwrap_or_default(),
        }
    }

    fn save(&self) -> Res<()> {
        let v = json!({ "device": self.device.map(|d| hex(&d)), "servers": self.servers });
        crate::secrets::set(v.to_string().as_bytes())
    }

    /// This phone's key, made and kept on first use.
    pub fn device_key(&mut self) -> Res<kvad_tunnel::Key> {
        if let Some(bytes) = &self.device {
            return Ok(kvad_tunnel::Key::from_bytes(bytes));
        }
        let key = kvad_tunnel::Key::generate();
        self.device = Some(key.to_bytes());
        self.save()?;
        Ok(key)
    }

    pub fn get(&self, server: &str) -> Option<Pairing> {
        let p = self.servers.get(server)?;
        let text = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
        Some(Pairing {
            address: text("address")?,
            key: text("key")?,
            voucher: text("voucher"),
            account: text("account").unwrap_or_default(),
            role: text("role").unwrap_or_default(),
            device: p.get("device").and_then(Value::as_i64),
        })
    }

    pub fn put(&mut self, server: &str, p: &Pairing) -> Res<()> {
        self.servers.insert(
            server.to_string(),
            json!({
                "address": p.address, "key": p.key, "voucher": p.voucher,
                "account": p.account, "role": p.role, "device": p.device,
            }),
        );
        self.save()
    }

    pub fn forget(&mut self, server: &str) -> Res<()> {
        self.servers.remove(server);
        self.save()
    }

    /// The first server paired with, for the Connect screen to offer.
    pub fn first(&self) -> Option<String> {
        self.servers.keys().next().cloned()
    }
}
