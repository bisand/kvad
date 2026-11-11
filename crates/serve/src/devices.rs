//! The devices paired with this server through the tunnel: the `devices`
//! table (migration 022) and the rules for a device's key.
//!
//! A device has a node id, which the tunnel lets in, and an API key, which
//! works only when that same device presents it through the tunnel. What it
//! may do is its account's role, lowered to `user` unless it was paired with
//! `admin` granted.
//!
//! The same goes for every key made through the device (`kvad keys add` on
//! it): `api_keys.device` says whose it is, and such a key is the device's in
//! all three ways, so that asking for a key is not a way out of the limits.

use crate::db::Db;
use crate::secret;
use crate::users;
use rusqlite::{params, OptionalExtension};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct Device {
    pub id: i64,
    /// The device's public key, 64 hex digits.
    pub node: String,
    pub name: String,
    pub user: i64,
    /// The owning account's name, for an administrator's listing.
    pub owner: String,
    pub api_key: Option<i64>,
    /// Whether pairing granted it the account's administrator role.
    pub admin: bool,
    pub paired_at: String,
    pub last_seen: Option<String>,
}

const COLUMNS: &str = "d.id, d.node, d.name, d.user, u.name AS owner, d.api_key, d.admin, d.paired_at, d.last_seen";

fn row(r: &rusqlite::Row) -> rusqlite::Result<Device> {
    Ok(Device {
        id: r.get("id")?,
        node: r.get("node")?,
        name: r.get("name")?,
        user: r.get("user")?,
        owner: r.get("owner")?,
        api_key: r.get("api_key")?,
        admin: r.get::<_, i64>("admin")? == 1,
        paired_at: r.get("paired_at")?,
        last_seen: r.get("last_seen")?,
    })
}

/// A device's name as it is kept: trimmed, at most 64 characters, no control
/// characters, since it is shown in terminals and pages.
pub fn clean_name(name: &str) -> String {
    let name: String = name.chars().filter(|c| !c.is_control()).take(64).collect();
    match name.trim() {
        "" => "a device".into(),
        n => n.to_string(),
    }
}

/// Pairs `node` to `user`: the row, and an API key of the device's. One
/// transaction in effect, since a device with no key is a device that cannot
/// sign in, and the row is removed again if the key cannot be made.
pub fn pair(db: &Db, node: &str, name: &str, user: i64, admin: bool) -> Res<(Device, String)> {
    if get_by_node(db, node)?.is_some() {
        return Err("this device is already paired; remove it first (kvad devices rm)".into());
    }
    let name = clean_name(name);
    let id = db.with(|c| {
        c.execute(
            "INSERT INTO devices (node, name, user, admin) VALUES (?1, ?2, ?3, ?4)",
            params![node, name, user, admin as i64],
        )?;
        Ok(c.last_insert_rowid())
    })?;
    let keyed = users::create_key(db, user, Some(id), &format!("device: {name}")).and_then(|(key, token)| {
        db.with(|c| c.execute("UPDATE devices SET api_key = ?1 WHERE id = ?2", [key.id, id]))?;
        Ok(token)
    });
    let token = match keyed {
        Ok(token) => token,
        Err(e) => {
            let _ = remove(db, id);
            return Err(e);
        }
    };
    let device = get(db, id)?.ok_or("the device vanished as it was paired")?;
    Ok((device, token))
}

pub fn get(db: &Db, id: i64) -> Res<Option<Device>> {
    db.with(|c| {
        c.query_row(&format!("SELECT {COLUMNS} FROM devices d JOIN users u ON u.id = d.user WHERE d.id = ?1"), [id], row)
            .optional()
    })
}

pub fn get_by_node(db: &Db, node: &str) -> Res<Option<Device>> {
    db.with(|c| {
        c.query_row(&format!("SELECT {COLUMNS} FROM devices d JOIN users u ON u.id = d.user WHERE d.node = ?1"), [node], row)
            .optional()
    })
}

/// The device whose key `token` is, if it is a device's: the one pairing
/// gave it, or one made through it.
pub fn by_token(db: &Db, token: &str) -> Res<Option<Device>> {
    let fingerprint = secret::fingerprint(token);
    db.with(|c| {
        c.query_row(
            &format!(
                "SELECT {COLUMNS} FROM devices d JOIN users u ON u.id = d.user
                 JOIN api_keys k ON k.device = d.id WHERE k.token_hash = ?1"
            ),
            [&fingerprint],
            row,
        )
        .optional()
    })
}

/// One account's devices, or every device for `None`.
pub fn list(db: &Db, user: Option<i64>) -> Res<Vec<Device>> {
    db.with(|c| {
        let all = format!("SELECT {COLUMNS} FROM devices d JOIN users u ON u.id = d.user ORDER BY d.id");
        let one = format!("SELECT {COLUMNS} FROM devices d JOIN users u ON u.id = d.user WHERE d.user = ?1 ORDER BY d.id");
        let rows = match user {
            None => c.prepare(&all)?.query_map([], row)?.collect(),
            Some(u) => c.prepare(&one)?.query_map([u], row)?.collect(),
        };
        rows
    })
}

/// Every paired device's node id: what the tunnel lets in.
pub fn nodes(db: &Db) -> Res<Vec<String>> {
    db.with(|c| c.prepare("SELECT node FROM devices")?.query_map([], |r| r.get(0))?.collect())
}

/// Removes a device and its keys together, so that it can neither reach the
/// server nor use a key anywhere else. The device it was, if there was one.
pub fn remove(db: &Db, id: i64) -> Res<Option<Device>> {
    let Some(device) = get(db, id)? else { return Ok(None) };
    db.with(|c| {
        // The foreign key cascades too; said here, since it is the point.
        c.execute("DELETE FROM api_keys WHERE device = ?1", [id])?;
        c.execute("DELETE FROM devices WHERE id = ?1", [id])
    })?;
    Ok(Some(device))
}

pub fn touch(db: &Db, id: i64) {
    let _ = db.with(|c| c.execute("UPDATE devices SET last_seen = datetime('now') WHERE id = ?1", [id]));
}

#[cfg(test)]
mod tests {
    use super::*;

    const NODE: &str = "14713dcae14786ed6e5c20f250a2c67ab31ee028a72d1d12ec457ca850a2a310";

    fn seeded() -> (Db, i64) {
        let db = Db::in_memory().unwrap();
        let ada = users::create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
        (db, ada.id)
    }

    #[test]
    fn a_pairing_makes_a_device_and_a_key_that_finds_it() {
        let (db, ada) = seeded();
        let (device, token) = pair(&db, NODE, "  Ada's\u{7}phone  ", ada, false).unwrap();
        assert_eq!(device.name, "Ada'sphone");
        assert!(!device.admin);
        assert_eq!(by_token(&db, &token).unwrap().unwrap().id, device.id);
        assert_eq!(nodes(&db).unwrap(), vec![NODE.to_string()]);
        assert!(pair(&db, NODE, "again", ada, false).unwrap_err().to_string().contains("already paired"));
        // The key is an ordinary key of the account's, and goes with the device.
        assert!(users::from_key(&db, &token).unwrap().is_some());
        remove(&db, device.id).unwrap().unwrap();
        assert!(users::from_key(&db, &token).unwrap().is_none(), "a removed device's key still works");
        assert!(nodes(&db).unwrap().is_empty());
    }

    #[test]
    fn an_ordinary_key_is_not_a_device() {
        let (db, ada) = seeded();
        let (_, token) = users::create_key(&db, ada, None, "laptop").unwrap();
        assert!(by_token(&db, &token).unwrap().is_none());
    }
}
