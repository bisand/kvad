//! Passwords, tokens, and the two different ways to hash them.
//!
//! The two are not interchangeable and the difference is the whole of this
//! module:
//!
//! * **A password** is short, chosen by a person, and guessable. It is hashed
//!   with argon2id, which is deliberately slow and memory-hungry so that a
//!   stolen database is expensive to attack offline.
//! * **A token** — a session cookie, an API key — is 32 bytes this program
//!   drew from the operating system. There is nothing to guess, so there is
//!   nothing for a slow hash to buy: SHA-256 is right, and it is fast enough
//!   to run on every request.
//!
//! Using argon2 for tokens would make every request cost a tenth of a second.
//! Using SHA-256 for passwords would make a stolen database a list of
//! passwords. Both mistakes are easy and this file exists to make them hard.
//!
//! # What is stored
//!
//! Never the token. The database holds `sha256(token)`, so somebody who reads
//! the database cannot sign in with what they find. A token is shown to its
//! owner exactly once, when it is made.

use argon2::password_hash::{PasswordHasher, PasswordVerifier};
use argon2::Argon2;
use sha2::{Digest, Sha256};

/// A fresh secret: 32 bytes from the operating system, in hex.
///
/// 256 bits, because these are guessed online against a server that will
/// answer forever. Hex rather than base64 so that a token can be pasted into
/// anything — a URL, a shell, a config file — without an encoding argument.
pub fn token() -> String {
    let mut bytes = [0u8; 32];
    // A failure here means the OS has no randomness, which is not a condition
    // to paper over with a weaker token.
    getrandom::fill(&mut bytes).expect("the operating system has no randomness");
    hex(&bytes)
}

/// The hash a token is stored and looked up by.
pub fn fingerprint(token: &str) -> String {
    hex(&Sha256::digest(token.as_bytes()))
}

/// The first few characters of a token, kept so that a key can be recognised
/// in a list without the list being a list of keys.
pub fn prefix(token: &str) -> String {
    token.chars().take(8).collect()
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).expect("a nibble is a hex digit"));
        out.push(char::from_digit((b & 0xf) as u32, 16).expect("a nibble is a hex digit"));
    }
    out
}

/// Hash a password for storage: argon2id, with a random salt, as a PHC string
/// that carries its own parameters.
///
/// The parameters travel inside the hash, so raising them later does not
/// invalidate what is already stored: an old hash still verifies against the
/// settings it was made with.
pub fn hash_password(password: &str) -> Result<String, String> {
    Argon2::default()
        .hash_password(password.as_bytes())
        .map(|h| h.to_string())
        .map_err(|e| format!("could not hash the password: {e}"))
}

/// Whether `password` is the one behind `stored`.
///
/// False for anything that is not a hash we can read, including the empty
/// string a user with no password has: a user who cannot sign in with a
/// password must not be signed in by supplying none.
pub fn verify_password(password: &str, stored: &str) -> bool {
    // Belt as well as braces: argon2 refuses an empty string too, because it
    // is not a PHC hash. This says out loud what would otherwise depend on
    // that, and would still hold if the verifier were ever swapped.
    if stored.is_empty() {
        return false;
    }
    Argon2::default().verify_password(password.as_bytes(), stored).is_ok()
}

/// The least a password may be.
///
/// Length and nothing else. Composition rules — a digit, a symbol, a capital
/// — make passwords harder to remember and not much harder to guess, and this
/// server is usually one person's own machine.
pub const MIN_PASSWORD: usize = 8;

pub fn check_password(password: &str) -> Result<(), String> {
    match password.chars().count() >= MIN_PASSWORD {
        true => Ok(()),
        false => Err(format!("a password needs at least {MIN_PASSWORD} characters")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_password_verifies_against_its_own_hash_and_nothing_else() {
        let stored = hash_password("correct horse battery").unwrap();
        assert!(verify_password("correct horse battery", &stored));
        assert!(!verify_password("correct horse batteries", &stored));
        assert!(!verify_password("", &stored));

        // argon2id, and the parameters travel with the hash so that raising
        // them later does not lock anybody out.
        assert!(stored.starts_with("$argon2id$"), "{stored}");

        // A random salt, so two people with the same password do not have the
        // same row — and a leak of one does not confirm the other.
        let twice = hash_password("correct horse battery").unwrap();
        assert_ne!(stored, twice);
        assert!(verify_password("correct horse battery", &twice));
    }

    /// A user with no password — one who signs in another way, or a row left
    /// half-made — must not be signed in by supplying nothing.
    #[test]
    fn nothing_verifies_against_an_empty_hash() {
        assert!(!verify_password("", ""));
        assert!(!verify_password("anything", ""));
        assert!(!verify_password("anything", "not a phc string"));
    }

    #[test]
    fn tokens_are_long_random_and_stored_only_as_their_hash() {
        let a = token();
        let b = token();
        assert_ne!(a, b);
        assert_eq!(a.len(), 64, "256 bits, in hex");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));

        // The fingerprint is what the database holds, and it is not the token.
        let f = fingerprint(&a);
        assert_eq!(f.len(), 64);
        assert_ne!(f, a);
        assert_eq!(f, fingerprint(&a), "the same token must hash the same way twice");
        assert_ne!(f, fingerprint(&b));

        assert_eq!(prefix(&a), a[..8]);
    }

    #[test]
    fn hex_is_lowercase_and_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff, 0xa5]), "000fffa5");
    }

    #[test]
    fn a_short_password_is_refused_with_a_reason() {
        assert!(check_password("longenough").is_ok());
        let err = check_password("short").unwrap_err();
        assert!(err.contains("8"), "{err}");
    }
}
