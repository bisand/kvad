//! Finding workers without being told where they are.
//!
//! Over a cable a worker is `fe80::ca1:57c7:6946:68a8%en13`. Nobody types
//! that twice; it changes when the cable moves to another port; and the
//! interface it is scoped to has a different name at each end. So a worker
//! says it is one — `_kvad._tcp`, over multicast DNS, with its id and its
//! build beside the name — and a coordinator asks who is out there and is
//! told every address each of them can be reached on.
//!
//! Nothing here is a resolver. Every Mac runs `mDNSResponder` already; a
//! worker registers with it through the system's own `dns_sd` calls, so
//! the advertisement lasts exactly as long as the process does, and a
//! coordinator reads what `dns-sd` prints. On anything else nothing is
//! advertised or found, and a worker is named by its address.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::Duration;

/// What a worker advertises as.
pub const SERVICE: &str = "_kvad._tcp";

/// A worker somebody is advertising.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    /// What it calls itself: its host name, unless two share one.
    pub name: String,
    /// Its machine id, from the advertisement.
    pub id: String,
    /// Its build, likewise.
    pub build: String,
    pub port: u16,
    /// Every way to reach it, as `--workers` would take them: link-local
    /// addresses scoped to this machine's interface, and the rest.
    pub addresses: Vec<String>,
}

impl Found {
    /// Whether `word` means this worker: its name, whatever the case, or
    /// the start of its id.
    pub fn answers_to(&self, word: &str) -> bool {
        self.name.eq_ignore_ascii_case(word) || (word.len() >= 4 && self.id.starts_with(&word.to_ascii_lowercase()))
    }
}

/// Whether a word of `--workers` could be a worker's name, and is not
/// plainly somewhere to connect to: no colon, so no port and no IPv6; no
/// dot, so no domain and no IPv4.
pub fn is_a_name(word: &str) -> bool {
    !word.is_empty() && !word.contains([':', '.', '[', '%']) && word != "localhost"
}

// ---------------------------------------------------------------------------
// Saying so
// ---------------------------------------------------------------------------

/// An advertisement, for as long as this is kept.
pub struct Advert {
    #[cfg(target_os = "macos")]
    service: dns_sd::Ref,
}

/// A TXT record: each `key=value` behind a byte of its length.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn txt(pairs: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (key, value) in pairs {
        let entry = format!("{key}={value}");
        let entry = &entry.as_bytes()[..entry.len().min(255)];
        out.push(entry.len() as u8);
        out.extend_from_slice(entry);
    }
    out
}

#[cfg(target_os = "macos")]
mod dns_sd {
    //! The three calls of `<dns_sd.h>` an advertisement takes. They are in
    //! libSystem, which everything links.
    use std::ffi::{c_char, c_void};

    pub type Ref = *mut c_void;

    extern "C" {
        pub fn DNSServiceRegister(
            sd: *mut Ref,
            flags: u32,
            interface: u32,
            name: *const c_char,
            regtype: *const c_char,
            domain: *const c_char,
            host: *const c_char,
            port: u16,
            txt_len: u16,
            txt: *const c_void,
            callback: *const c_void,
            context: *mut c_void,
        ) -> i32;
        pub fn DNSServiceRefDeallocate(sd: Ref);
    }
}

/// Tell the link there is a worker here, on `port`.
///
/// Registered with the system's responder and not by a helper process, so
/// that it is withdrawn when this process ends however it ends: an
/// advertisement that outlived its worker would send every coordinator to
/// a port nobody answers.
#[cfg(target_os = "macos")]
pub fn advertise(name: &str, port: u16, id: &str, build: &str) -> Result<Advert, String> {
    let name = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
    let kind = std::ffi::CString::new(SERVICE).expect("no NUL in a constant");
    let record = txt(&[("id", id), ("build", build)]);
    let mut service: dns_sd::Ref = std::ptr::null_mut();
    // SAFETY: every pointer is to something that outlives the call, the
    // record's length is the length given, and no callback is asked for.
    let status = unsafe {
        dns_sd::DNSServiceRegister(
            &mut service,
            0,
            0,
            name.as_ptr(),
            kind.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            port.to_be(),
            record.len() as u16,
            record.as_ptr().cast(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    match status {
        0 => Ok(Advert { service }),
        code => Err(format!("the system would not advertise this worker (dns_sd error {code})")),
    }
}

#[cfg(not(target_os = "macos"))]
pub fn advertise(_name: &str, _port: u16, _id: &str, _build: &str) -> Result<Advert, String> {
    Err("not advertised: finding workers is written for macOS so far, and elsewhere a worker is named by its address".into())
}

#[cfg(target_os = "macos")]
impl Drop for Advert {
    fn drop(&mut self) {
        // SAFETY: the reference came from `DNSServiceRegister` and is
        // released once.
        unsafe { dns_sd::DNSServiceRefDeallocate(self.service) }
    }
}

// The reference is a handle on a connection to the responder, used from
// whichever thread drops it and from no two at once.
#[cfg(target_os = "macos")]
unsafe impl Send for Advert {}

// ---------------------------------------------------------------------------
// Asking
// ---------------------------------------------------------------------------

/// What `dns-sd` printed in `wait`. It answers for as long as it is left
/// running, so it is left running that long and no longer.
fn dns_sd_for(args: &[&str], wait: Duration) -> String {
    let Ok(mut child) = Command::new("dns-sd").args(args).stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null()).spawn() else {
        return String::new();
    };
    std::thread::sleep(wait);
    let _ = child.kill();
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        let _ = stdout.read_to_string(&mut out);
    }
    let _ = child.wait();
    out
}

/// The instances a browse found, each once.
///
/// ```text
/// Timestamp     A/R    Flags  if Domain               Service Type         Instance Name
///  8:58:12.318  Add        3  29 local.               _kvad._tcp.          kvad test
/// ```
fn instances(browse: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // The name is the last column and may hold spaces, so it is found by
    // where the header says it starts.
    let Some(at) = browse.lines().find_map(|line| line.find("Instance Name")) else { return out };
    for line in browse.lines() {
        let mut words = line.split_whitespace();
        if words.nth(1) != Some("Add") {
            continue;
        }
        if let Some(name) = line.get(at..).map(str::trim).filter(|n| !n.is_empty()) {
            if !out.iter().any(|seen| seen == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// Where a lookup says an instance is, and what its TXT record says.
///
/// ```text
///  8:58:13.825  kvad\032test._kvad._tcp.local. can be reached at host.local.:7499 (interface 15)
///  id=0011aabb build=0.18.2+abc1234
/// ```
fn located(lookup: &str) -> Option<(String, u16, Vec<(String, String)>)> {
    let mut lines = lookup.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.split_once(" can be reached at ").map(|(_, rest)| rest) else { continue };
        let target = rest.split_whitespace().next()?;
        let (host, port) = target.rsplit_once(':')?;
        let pairs = lines
            .next()
            .unwrap_or("")
            .split_whitespace()
            .filter_map(|pair| pair.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .collect();
        return Some((host.to_string(), port.parse().ok()?, pairs));
    }
    None
}

/// The addresses a host has, as something to connect to.
///
/// ```text
/// Timestamp     A/R  Flags         IF  Hostname                  Address                                      TTL
///  8:58:15.357  Add  40000003      30  host.local.               FE80:0000:0000:0000:1836:AD27:BDB8:58C3%en13 4500
///  8:58:15.358  Add  40000003      15  host.local.               10.10.9.177                                  4500
/// ```
///
/// Kept: addresses on Ethernet-like interfaces and bridges, and plain IPv4.
/// Dropped: the loopback, records that say there is none, and the
/// interfaces macOS keeps for itself (`awdl`, `llw`, `anpi`, `utun`),
/// which answer to a name and carry nothing a worker could use.
fn reachable(resolved: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in resolved.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.get(1) != Some(&"Add") || line.contains("No Such Record") {
            continue;
        }
        let Some(raw) = words.get(5) else { continue };
        let address = match raw.split_once('%') {
            Some((ip, scope)) => {
                let Ok(ip) = ip.parse::<std::net::Ipv6Addr>() else { continue };
                if ip.is_loopback() || !(scope.starts_with("en") || scope.starts_with("bridge")) {
                    continue;
                }
                format!("{ip}%{scope}")
            }
            None => match raw.parse::<std::net::IpAddr>() {
                Ok(ip) if !ip.is_loopback() && !ip.is_unspecified() => ip.to_string(),
                _ => continue,
            },
        };
        if !out.contains(&address) {
            out.push(address);
        }
    }
    out
}

/// Ask the link who is a worker. Takes a couple of seconds: an answer to
/// multicast is waited for, not fetched.
pub fn workers() -> Vec<Found> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let wait = Duration::from_millis(900);
    let mut found = Vec::new();
    for name in instances(&dns_sd_for(&["-B", SERVICE, "local."], wait)) {
        let Some((host, port, pairs)) = located(&dns_sd_for(&["-L", &name, SERVICE, "local."], wait)) else { continue };
        let value = |key: &str| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone()).unwrap_or_default();
        let addresses = reachable(&dns_sd_for(&["-G", "v4v6", &host], wait));
        found.push(Found { name, id: value("id"), build: value("build"), port, addresses });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    // What `dns-sd` printed on an M5 MacBook Pro on 2026-10-12, with a test
    // service registered on it.
    const BROWSE: &str = "\
Browsing for _kvad._tcp.local.
DATE: ---Mon 12 Oct 2026---
 8:58:12.317  ...STARTING...
Timestamp     A/R    Flags  if Domain               Service Type         Instance Name
 8:58:12.318  Add        3  29 local.               _kvad._tcp.          kvad test
 8:58:12.318  Add        3   1 local.               _kvad._tcp.          kvad test
 8:58:12.318  Add        2  15 local.               _kvad._tcp.          Mac
 8:58:12.900  Rmv        0  15 local.               _kvad._tcp.          gone
";

    const LOOKUP: &str = "\
Lookup kvad test._kvad._tcp.local.
DATE: ---Mon 12 Oct 2026---
 8:58:13.824  ...STARTING...
 8:58:13.825  kvad\\032test._kvad._tcp.local. can be reached at bisand-macbook-pro.local.:7499 (interface 15) Flags: 1
 id=0011aabb build=0.18.2+abc1234
 8:58:13.825  kvad\\032test._kvad._tcp.local. can be reached at bisand-macbook-pro.local.:7499 (interface 15)
 id=0011aabb build=0.18.2+abc1234
";

    const RESOLVED: &str = "\
DATE: ---Mon 12 Oct 2026---
 8:58:15.356  ...STARTING...
Timestamp     A/R  Flags         IF  Hostname                               Address                                      TTL
 8:58:15.357  Add  40000003      29  bisand-macbook-pro.local.              FE80:0000:0000:0000:7C7A:B0FF:FED3:E5F2%anri0 4500
 8:58:15.357  Add  40000003       1  bisand-macbook-pro.local.              0000:0000:0000:0000:0000:0000:0000:0001%<0>  4500
 8:58:15.357  Add  40000003       1  bisand-macbook-pro.local.              FE80:0000:0000:0000:0000:0000:0000:0001%lo0  4500
 8:58:15.357  Add  40000003      15  bisand-macbook-pro.local.              FE80:0000:0000:0000:0435:4ED8:EAA4:734F%en0  4500
 8:58:15.357  Add  40000003      30  bisand-macbook-pro.local.              FE80:0000:0000:0000:1836:AD27:BDB8:58C3%en13 4500
 8:58:15.358  Add  40000003      29  bisand-macbook-pro.local.              0.0.0.0                                      1   No Such Record
 8:58:15.358  Add  40000003       1  bisand-macbook-pro.local.              127.0.0.1                                    4500
 8:58:15.358  Add  40000003      15  bisand-macbook-pro.local.              10.10.9.177                                  4500
 8:58:15.358  Add  40000003      30  bisand-macbook-pro.local.              169.254.4.162                                4500
 8:58:15.358  Add  40000002      15  bisand-macbook-pro.local.              10.10.9.177                                  4500
";

    #[test]
    fn a_browse_is_read_for_its_instances() {
        assert_eq!(instances(BROWSE), ["kvad test", "Mac"], "each once, spaces and all, and none that left");
        assert!(instances("").is_empty());
        assert!(instances("Browsing for _kvad._tcp.local.\n").is_empty());
    }

    #[test]
    fn a_lookup_is_read_for_where_and_what() {
        let (host, port, pairs) = located(LOOKUP).unwrap();
        assert_eq!((host.as_str(), port), ("bisand-macbook-pro.local.", 7499));
        assert_eq!(pairs, [("id".to_string(), "0011aabb".to_string()), ("build".to_string(), "0.18.2+abc1234".to_string())]);
        assert_eq!(located("Lookup x._kvad._tcp.local.\n"), None);
    }

    #[test]
    fn addresses_are_kept_where_a_worker_could_be_reached() {
        assert_eq!(
            reachable(RESOLVED),
            ["fe80::435:4ed8:eaa4:734f%en0", "fe80::1836:ad27:bdb8:58c3%en13", "10.10.9.177", "169.254.4.162"],
            "scoped and compressed; no loopback, no interface of the system's own, nothing twice"
        );
    }

    #[test]
    fn a_txt_record_is_lengths_and_pairs() {
        assert_eq!(txt(&[("id", "ab"), ("build", "1")]), b"\x05id=ab\x07build=1");
        assert!(txt(&[]).is_empty());
        assert_eq!(txt(&[("k", &"v".repeat(400))]).len(), 256, "an entry is at most what a byte can count");
    }

    #[test]
    fn a_worker_is_named_or_addressed() {
        for name in ["Mac", "air", "bisand-macbook-pro", "0011aabb"] {
            assert!(is_a_name(name), "{name}");
        }
        for address in ["air.local", "10.0.0.7", "air:7420", "fe80::1%en13", "[::1]:7420", "localhost", ""] {
            assert!(!is_a_name(address), "{address}");
        }
        let mac = Found { name: "Mac".into(), id: "0011aabbccddeeff".into(), build: "0.18.2".into(), port: 7420, addresses: Vec::new() };
        assert!(mac.answers_to("mac") && mac.answers_to("0011AABB") && mac.answers_to("0011aabbccddeeff"));
        assert!(!mac.answers_to("ma") && !mac.answers_to("001") && !mac.answers_to("air"));
    }
}
