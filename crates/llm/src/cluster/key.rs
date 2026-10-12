//! The cluster key, and proving you have it without sending it.
//!
//! A worker downloads what it is told to and gives whoever asks its memory
//! and its processor, so it does not answer just anybody. Every machine of a
//! cluster holds the same 32 random bytes, and a connection begins with
//! each end showing the other that it does:
//!
//! ```text
//!   coordinator                              worker
//!   Hello { nonce: C }            →
//!                                 ←          Hello { nonce: W }
//!   Auth { HMAC(key, "coordinator" C W) } →
//!                                 ←          Auth { HMAC(key, "worker" C W) }
//! ```
//!
//! The key never crosses. A nonce from each end means neither answer can be
//! a recording of an earlier one, and the two labels mean a worker's answer
//! is no use to somebody pretending to be a coordinator.
//!
//! What this does not do is hide anything. The hidden states that follow
//! cross in the clear: they are not the prompt, and they are not nothing.
//!
//! HMAC is written out here — it is eight lines over a hash — and the hash
//! is `sha2`, which the Hub client already brings.

use sha2::{Digest, Sha256};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Where this machine keeps the key unless told otherwise.
pub fn path() -> PathBuf {
    crate::weights::data_dir().join("cluster.key")
}

/// The variable that names a key directly, for a coordinator that has been
/// handed one and has no file.
pub const ENV: &str = "KVAD_CLUSTER_KEY";

#[derive(Clone, PartialEq, Eq)]
pub struct Key([u8; 32]);

/// Never printed by accident: a key in a log is a key no longer.
impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Key(…)")
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    let text = text.trim();
    if text.len() % 2 != 0 || !text.is_ascii() {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).ok()).collect()
}

/// Bytes from the operating system's generator.
pub fn random<const N: usize>() -> io::Result<[u8; N]> {
    let mut bytes = [0u8; N];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// A nonce for one handshake, as it is written in a `Hello`.
pub fn nonce() -> io::Result<String> {
    Ok(hex(&random::<16>()?))
}

impl Key {
    pub fn generate() -> io::Result<Key> {
        Ok(Key(random()?))
    }

    /// As it is written down: 64 hex digits.
    pub fn to_hex(&self) -> String {
        hex(&self.0)
    }

    pub fn from_hex(text: &str) -> Result<Key, String> {
        match unhex(text).map(<[u8; 32]>::try_from) {
            Some(Ok(bytes)) => Ok(Key(bytes)),
            _ => Err("a cluster key is 64 hex digits".into()),
        }
    }

    /// The key in `path`.
    pub fn read(path: &Path) -> Result<Key, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Key::from_hex(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The key in `path`, or a new one written there for its owner alone.
    /// Says whether it was made just now, which is the one time to show it.
    pub fn read_or_create(path: &Path) -> Result<(Key, bool), String> {
        if path.exists() {
            return Key::read(path).map(|key| (key, false));
        }
        let key = Key::generate().map_err(|e| format!("no random bytes for a key: {e}"))?;
        let wrote = || -> io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            use std::io::Write;
            options.open(path)?.write_all(format!("{}\n", key.to_hex()).as_bytes())
        };
        wrote().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((key, true))
    }

    /// The key a coordinator uses: the variable if it is set, else the file.
    pub fn ours() -> Result<Key, String> {
        match std::env::var(ENV) {
            Ok(text) if !text.trim().is_empty() => Key::from_hex(&text).map_err(|e| format!("{ENV}: {e}")),
            _ => Key::read(&path()).map_err(|e| {
                format!("no cluster key: {e}\nA worker prints its key the first time it starts. Put it in {ENV}, or in that file.")
            }),
        }
    }

    /// HMAC-SHA256, as RFC 2104 writes it: the hash of the key and the
    /// message, hashed again with the key, the key padded and flipped
    /// differently each time.
    pub fn mac(&self, parts: &[&[u8]]) -> [u8; 32] {
        // A key no longer than the hash's block is used as it is.
        let mut block = [0u8; 64];
        block[..32].copy_from_slice(&self.0);
        let mut inner = Sha256::new();
        inner.update(block.map(|b| b ^ 0x36));
        for part in parts {
            inner.update(part);
        }
        let mut outer = Sha256::new();
        outer.update(block.map(|b| b ^ 0x5c));
        outer.update(inner.finalize());
        outer.finalize().into()
    }

    /// What one end of a handshake sends to show it holds the key.
    ///
    /// The nonces are hex and so can hold no separator, which is what
    /// makes joining them with one unambiguous.
    pub fn proof(&self, who: Role, coordinator_nonce: &str, worker_nonce: &str) -> String {
        let label: &[u8] = match who {
            Role::Coordinator => b"kvad coordinator",
            Role::Worker => b"kvad worker",
        };
        hex(&self.mac(&[label, b"|", coordinator_nonce.as_bytes(), b"|", worker_nonce.as_bytes()]))
    }

    /// Whether `proof` is the one `who` would have sent.
    pub fn accepts(&self, who: Role, coordinator_nonce: &str, worker_nonce: &str, proof: &str) -> bool {
        let want = self.proof(who, coordinator_nonce, worker_nonce);
        // Every byte is looked at whatever the first difference: how long a
        // comparison took must not say how much of a guess was right.
        want.len() == proof.len() && want.bytes().zip(proof.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Coordinator,
    Worker,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4231, test case 1, with its 20-byte key padded to this type's 32
    /// — which HMAC does to a short key anyway, so the answer is the RFC's.
    #[test]
    fn the_mac_is_hmac_sha256() {
        let mut key = [0u8; 32];
        key[..20].fill(0x0b);
        let mac = Key(key).mac(&[b"Hi ", b"There"]);
        assert_eq!(hex(&mac), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    }

    #[test]
    fn a_proof_is_good_for_one_role_and_one_pair_of_nonces() {
        let key = Key::from_hex(&"ab".repeat(32)).unwrap();
        let other = Key::from_hex(&"cd".repeat(32)).unwrap();
        let proof = key.proof(Role::Coordinator, "aa11", "bb22");
        assert!(key.accepts(Role::Coordinator, "aa11", "bb22", &proof));
        assert!(!key.accepts(Role::Worker, "aa11", "bb22", &proof), "a coordinator's proof is not a worker's");
        assert!(!key.accepts(Role::Coordinator, "aa11", "bb23", &proof), "nor good for another handshake");
        assert!(!key.accepts(Role::Coordinator, "bb22", "aa11", &proof), "nor with the nonces changed over");
        assert!(!other.accepts(Role::Coordinator, "aa11", "bb22", &proof), "nor under another key");
        assert!(!key.accepts(Role::Coordinator, "aa11", "bb22", ""));
        assert!(!key.accepts(Role::Coordinator, "aa11", "bb22", &proof[..62]));
    }

    #[test]
    fn a_key_is_written_once_and_read_back() {
        let path = std::env::temp_dir().join(format!("kvad-key-{}", std::process::id())).join("cluster.key");
        let _ = std::fs::remove_file(&path);
        let (made, new) = Key::read_or_create(&path).unwrap();
        assert!(new);
        let (again, new) = Key::read_or_create(&path).unwrap();
        assert!(!new && again == made, "the second start reads what the first wrote");
        assert_eq!(Key::read(&path).unwrap().to_hex().len(), 64);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert_eq!(format!("{made:?}"), "Key(…)");
        assert_ne!(Key::generate().unwrap(), made);

        std::fs::write(&path, "not a key").unwrap();
        assert!(Key::read(&path).unwrap_err().contains("64 hex digits"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(Key::from_hex(&"zz".repeat(32)).is_err());
        assert!(Key::from_hex(&"ab".repeat(31)).is_err());
        assert_ne!(nonce().unwrap(), nonce().unwrap());
    }
}
