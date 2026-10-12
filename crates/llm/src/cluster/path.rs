//! Which way to a worker, when there is more than one.
//!
//! One machine honestly has several addresses — a link-local one on each
//! cable, another on Wi-Fi, an IPv4 beside each — and they are not the same
//! road. So before a worker is given layers, every address it was found on
//! is measured the way [`super::ping`] measures, over the connection a
//! stage would use, and the quickest is the one kept.
//!
//! It is also what catches the charging cable. A USB 2 link between two
//! Macs presents as a working link and is told from Thunderbolt by nothing
//! a socket can see; a path that is named for what it is, with a number
//! beside it, cannot be mistaken for the other one quietly. So this runs
//! for a worker named by its address as well, where there is nothing to
//! choose between: what a model's hidden states crossed goes on the record
//! either way.

use super::key::Key;
use super::link::{self, Link};
use super::ping::{self, DECODE_HOP, PREFILL_CHUNK};
use super::stage::Fault;
use super::wire::{self, Hello};
use std::net::TcpStream;
use std::time::Duration;

/// One way to a worker, and what it measured.
pub struct Path {
    /// As `--workers` would take it.
    pub address: String,
    /// Who answered there.
    pub hello: Hello,
    /// What this machine's end of it is, where that can be told.
    pub link: Option<Link>,
    /// Milliseconds there and back for a decode hop of 32 KB, and for a
    /// prefill chunk of 2 MB: medians of three after a round thrown away.
    pub hop_ms: f64,
    pub chunk_ms: f64,
}

impl Path {
    /// One line: where, what, and how fast.
    pub fn line(&self) -> String {
        let what = match &self.link {
            Some(link) => format!("{} on {}", link.title(), link.iface),
            None => "a link not identified".to_string(),
        };
        format!("{}: {what}; {:.2} ms for 32 KB there and back, {:.1} ms for 2 MB", self.address, self.hop_ms, self.chunk_ms)
    }
}

/// An address and a port as one word: bracketed where the address has
/// colons of its own, and bare where the port is the usual one.
pub fn with_port(address: &str, port: u16) -> String {
    match (port == wire::PORT, address.contains(':')) {
        (true, _) => address.to_string(),
        (false, true) => format!("[{address}]:{port}"),
        (false, false) => format!("{address}:{port}"),
    }
}

/// Measure the way a connection to a worker goes.
fn measured(address: &str, mut stream: TcpStream, hello: Hello) -> Result<Path, Fault> {
    let link = match (stream.local_addr(), stream.peer_addr()) {
        (Ok(local), Ok(peer)) => link::between(&local.ip(), &peer.ip()),
        _ => None,
    };
    let options = ping::Options { rounds: 3, sizes: vec![DECODE_HOP, PREFILL_CHUNK], stream: 0 };
    let got = ping::measure(&mut stream, &options, &mut |_| {}).map_err(|e| format!("{address}: {e}"))?;
    Ok(Path { address: address.to_string(), hello, link, hop_ms: got.median(0), chunk_ms: got.median(1) })
}

/// Connect to a worker at `address`, measure the way there, and hang up.
pub fn measure(address: &str, key: &Key) -> Result<Path, Fault> {
    let (stream, hello) = super::remote::connect(address, key)?;
    measured(address, stream, hello)
}

/// Measure every address a worker was found on and keep the quickest for a
/// decode hop, which is what every token pays. Each is reported, the ones
/// that failed among them: a cable that is plugged in and does not answer
/// is worth a line.
///
/// They are greeted all at once and measured one after another. An
/// address that leads nowhere costs a few seconds of waiting, and five of
/// them should cost that once and not five times; but two measurements
/// sharing a worker would each record the other.
///
/// `id` is who ought to answer, when that is known: an address that turns
/// out to be somebody else's is not a path to this worker.
pub fn quickest(addresses: &[String], id: &str, key: &Key, said: &mut dyn FnMut(&str)) -> Result<Path, Fault> {
    let (reach, greeting) = match addresses.len() {
        // The only way there is worth the usual patience.
        1 => (Duration::from_secs(5), Duration::from_secs(10)),
        _ => (Duration::from_secs(2), Duration::from_secs(3)),
    };
    let greeted: Vec<Result<(TcpStream, Hello), Fault>> = std::thread::scope(|threads| {
        let greeting_each: Vec<_> =
            addresses.iter().map(|address| threads.spawn(move || super::remote::connect_within(address, key, reach, greeting))).collect();
        greeting_each.into_iter().map(|thread| thread.join().unwrap_or_else(|_| Err("the greeting panicked".into()))).collect()
    });

    let mut best: Option<Path> = None;
    let mut last: Option<Fault> = None;
    for (address, greeted) in addresses.iter().zip(greeted) {
        match greeted.and_then(|(stream, hello)| measured(address, stream, hello)) {
            Ok(path) if !id.is_empty() && path.hello.id != id => {
                said(&format!("  {address}: answered by {} and not by the worker asked for", path.hello.name));
            }
            Ok(path) => {
                said(&format!("  {}", path.line()));
                if best.as_ref().is_none_or(|b| path.hop_ms < b.hop_ms) {
                    best = Some(path);
                }
            }
            Err(e) => {
                said(&format!("  {}", e.to_string().lines().next().unwrap_or("")));
                last = Some(e);
            }
        }
    }
    best.ok_or_else(|| last.unwrap_or_else(|| "it was found, and on no address".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_takes_its_port_the_way_it_can() {
        assert_eq!(with_port("fe80::1%en13", wire::PORT), "fe80::1%en13");
        assert_eq!(with_port("fe80::1%en13", 7000), "[fe80::1%en13]:7000");
        assert_eq!(with_port("10.0.0.7", 7000), "10.0.0.7:7000");
        for (address, port) in [("fe80::1%en13", 7000), ("10.0.0.7", 7000), ("fe80::1%en13", wire::PORT)] {
            assert_eq!(wire::host_port(&with_port(address, port)), (address.to_string(), port));
        }
    }
}
