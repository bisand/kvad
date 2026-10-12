//! What a link is, asked of the machine and not of the socket.
//!
//! This exists because of how the first rig started. A USB-C charging cable
//! carries USB 2. Join two Macs with one and the Thunderbolt Bridge never
//! comes up — and macOS falls back to USB NCM without saying so. An ordinary
//! interface appears, takes a link-local address, answers `ping6`, carries
//! TCP and accepts SSH, at about a hundredth of what the two ports could do.
//! Everything a connectivity test would check passes. A worker would run on
//! it and simply be slow.
//!
//! Nothing a socket can observe tells that apart from Thunderbolt, so a
//! number recorded without identifying the link is a number about nothing.
//! `ifconfig` does not help either: both are up, both `RUNNING`. Three
//! things do, and the registry is the one that cannot be misread — the
//! driver directly above the interface is `AppleThunderboltIP…` or
//! `AppleUSBNCM…`, and the port a USB link hangs off is named `…-port-hs`
//! for USB 2.0 High Speed, whatever the connector's shape implies.
//!
//! Everything here that decides takes text, so that it can be tested
//! against what these machines printed; the functions that run the
//! commands only fetch it. On anything but macOS nothing is fetched and a
//! link is honestly unidentified.

use std::net::IpAddr;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum What {
    /// This machine talking to itself.
    Loopback,
    /// Thunderbolt carrying IP: the link the plan is written for.
    Thunderbolt,
    /// USB carrying Ethernet frames, at whatever the cable allows.
    UsbNcm(Usb),
    /// A port macOS has a name for: `Wi-Fi`, `Ethernet`.
    Port(String),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Usb {
    /// USB 2.0: 480 Mb/s nominal, which is what a charging cable carries.
    High,
    /// USB 3 or later.
    Super,
    Unsaid,
}

#[derive(Clone, Debug)]
pub struct Link {
    pub iface: String,
    pub what: What,
    /// What was looked at, and what it said.
    pub evidence: Vec<(&'static str, String)>,
}

impl Link {
    pub fn title(&self) -> String {
        match &self.what {
            What::Loopback => "loopback: this machine, and no link at all".into(),
            What::Thunderbolt => "Thunderbolt carrying IP".into(),
            What::UsbNcm(Usb::High) => "USB NCM, USB 2.0 High Speed: 480 Mb/s".into(),
            What::UsbNcm(Usb::Super) => "USB NCM, USB SuperSpeed".into(),
            What::UsbNcm(Usb::Unsaid) => "USB NCM".into(),
            What::Port(name) => name.clone(),
            What::Unknown => "not identified".into(),
        }
    }

    /// What to know before writing a number from this link down.
    pub fn caution(&self) -> Option<&'static str> {
        match self.what {
            What::Loopback => Some("no cable was crossed. The tool is measured here, and no link."),
            What::UsbNcm(_) => Some("not Thunderbolt. Fine to build on, wrong to quote."),
            What::Unknown => Some("a number from a link nobody identified describes nothing."),
            What::Thunderbolt | What::Port(_) => None,
        }
    }
}

/// A command's output, or nothing if it could not be run. Every use is a
/// diagnostic, and a machine that answers none of them is still worth
/// measuring.
fn sh(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
        Err(_) => String::new(),
    }
}

/// Which interface holds an address of this machine's.
pub fn interface_of(ip: &IpAddr) -> Option<String> {
    owner_of(&sh("ifconfig", &[]), ip)
}

/// As [`interface_of`], over `ifconfig`'s text.
fn owner_of(ifconfig: &str, ip: &IpAddr) -> Option<String> {
    let wanted = ip.to_string();
    let mut iface = None;
    for line in ifconfig.lines() {
        if !line.starts_with(char::is_whitespace) {
            iface = line.split(':').next().filter(|name| !name.is_empty());
            continue;
        }
        let mut words = line.split_whitespace();
        if let (Some("inet" | "inet6"), Some(addr)) = (words.next(), words.next()) {
            // A link-local address is written with its scope: fe80::1%lo0.
            if addr.split('%').next() == Some(wanted.as_str()) {
                return iface.map(str::to_string);
            }
        }
    }
    None
}

/// The registry's path to a node, root first: each entry's name and class.
///
/// `ioreg -t` draws the tree as `+-o NAME  <class CLASS, id …>`, a line an
/// entry, with the node asked for at the end of its ancestry and its
/// children after it.
fn ancestry(ioreg: &str) -> Vec<(String, String)> {
    ioreg
        .lines()
        .filter_map(|line| {
            let entry = line.split_once("+-o ")?.1;
            let (name, rest) = entry.split_once("<class ")?;
            let class = rest.split([',', '>']).next()?;
            // `Mac@00100000`, `usb-drd0-port-hs@00100000`: the name, then
            // where on its bus.
            Some((name.trim().split('@').next().unwrap_or("").to_string(), class.trim().to_string()))
        })
        .collect()
}

/// macOS's name for a device, from `networksetup -listallhardwareports`.
fn port_name(ports: &str, iface: &str) -> Option<String> {
    let mut name = None;
    for line in ports.lines() {
        if let Some(port) = line.strip_prefix("Hardware Port: ") {
            name = Some(port.trim());
        } else if line.strip_prefix("Device: ").map(str::trim) == Some(iface) {
            return name.map(str::to_string);
        }
    }
    None
}

/// What `ifconfig IFACE` says after `key: `.
fn field<'a>(ifconfig: &'a str, key: &str) -> Option<&'a str> {
    ifconfig.lines().find_map(|line| line.trim().strip_prefix(key)?.strip_prefix(": ")).map(str::trim)
}

/// The texts [`classify`] decides from.
#[derive(Default)]
struct Seen<'a> {
    /// `ioreg -w0 -p IOService -n IFACE -r -t`
    ioreg: &'a str,
    /// `ifconfig IFACE`
    iface: &'a str,
    /// `ifconfig bridge0`
    bridge: &'a str,
    /// `system_profiler SPThunderboltDataType`
    thunderbolt: &'a str,
    /// `networksetup -listallhardwareports`
    ports: &'a str,
}

fn classify(iface: &str, seen: &Seen) -> Link {
    let mut evidence = Vec::new();
    if iface.starts_with("lo") {
        return Link { iface: iface.into(), what: What::Loopback, evidence };
    }

    // The node directly above the interface is the driver that owns it,
    // which is most of the answer.
    let chain = ancestry(seen.ioreg);
    let at = chain.iter().position(|(name, _)| name == iface);
    let above = &chain[..at.unwrap_or(0)];
    let is_bridge = iface.starts_with("bridge");
    // A bridge is the kernel's own and hangs off no driver.
    if !(is_bridge && above.is_empty()) {
        evidence.push((
            "ioreg",
            match above.last() {
                Some((name, class)) if name == class => name.clone(),
                Some((name, class)) => format!("{name} ({class})"),
                None => "no such interface in the registry; is the cable in?".into(),
            },
        ));
    }

    // "Link Status: 0x100" ends in "Status:" as well, so the line is matched
    // and not the word: counting the word finds six ports on a machine that
    // has three.
    // And only this machine's own ports: a dock on one of them lists
    // ports of its own further in, and they are not ways to another Mac.
    let own = seen.thunderbolt.lines().filter(|l| l.trim_start().starts_with("Status:")).map(|l| l.len() - l.trim_start().len()).min();
    let status: Vec<&str> =
        seen.thunderbolt.lines().filter(|l| Some(l.len() - l.trim_start().len()) == own).map(str::trim).filter(|l| l.starts_with("Status:")).collect();
    let live = status.iter().filter(|l| !l.contains("No device connected")).count();
    let member = seen.bridge.contains(&format!("member: {iface} "));
    let bridge_up = field(seen.bridge, "status") == Some("active");

    let what = if above.iter().any(|(_, class)| class.starts_with("AppleThunderboltIP")) {
        What::Thunderbolt
    } else if above.iter().any(|(_, class)| class.starts_with("AppleUSB") && class.contains("NCM")) {
        // `AppleUSBNCM…` on the Mac that is the cable's host, and
        // `AppleUSBDeviceNCM…` on the one that is its device: which is
        // which depends on the order they were plugged in, and it is the
        // same link from either end. The device's end hangs off no port
        // that says its speed.
        // `usb-drdN-port-hs` is USB 2.0 High Speed and `-ss` is SuperSpeed.
        // The connector is the same shape either way, which is the whole
        // problem.
        What::UsbNcm(match above.iter().rev().find_map(|(name, _)| name.rsplit_once("-port-").map(|(_, speed)| speed)) {
            Some("hs") => Usb::High,
            Some(speed) if speed.starts_with("ss") => Usb::Super,
            _ => Usb::Unsaid,
        })
    } else if is_bridge && bridge_up && live > 0 {
        // The Thunderbolt Bridge holds the address, and its members carry
        // it: with a device on a Thunderbolt port and the bridge up, that
        // is what this is.
        What::Thunderbolt
    } else if let Some(name) = port_name(seen.ports, iface) {
        What::Port(name)
    } else {
        What::Unknown
    };

    if is_bridge {
        let members: Vec<&str> = seen.bridge.lines().filter_map(|l| l.trim().strip_prefix("member: ")?.split(' ').next()).collect();
        evidence.push(("bridge0", format!("{}, of {}", if bridge_up { "active" } else { "inactive" }, members.join(" "))));
    } else if !seen.bridge.is_empty() {
        evidence.push(("bridge0", format!("{}, {}", if member { "a member" } else { "not a member" }, if bridge_up { "active" } else { "inactive" })));
    }
    if !status.is_empty() {
        evidence.push(("Thunderbolt", format!("{live} of {} ports with a device", status.len())));
    }
    if let Some(media) = field(seen.iface, "media").filter(|m| *m != "autoselect") {
        evidence.push(("media", media.to_string()));
    }
    Link { iface: iface.into(), what, evidence }
}

/// The link a connection is on, from its two ends. `None` where this
/// machine cannot say which of its interfaces the connection left by.
///
/// A connection to one of this machine's own addresses never leaves it,
/// whichever interface the address belongs to: the kernel hands the
/// packets straight back. Named for the interface, that would be a cable
/// measured at memory's speed.
pub fn between(local: &IpAddr, peer: &IpAddr) -> Option<Link> {
    if peer.is_loopback() || interface_of(peer).is_some() {
        return Some(Link { iface: "lo0".into(), what: What::Loopback, evidence: Vec::new() });
    }
    interface_of(local).map(|iface| identify(&iface))
}

/// Name the physical link behind an interface of this machine, and keep
/// what was looked at.
pub fn identify(iface: &str) -> Link {
    if !cfg!(target_os = "macos") || iface.starts_with("lo") {
        return classify(iface, &Seen::default());
    }
    classify(
        iface,
        &Seen {
            ioreg: &sh("ioreg", &["-w0", "-p", "IOService", "-n", iface, "-r", "-t"]),
            iface: &sh("ifconfig", &[iface]),
            bridge: &sh("ifconfig", &["bridge0"]),
            thunderbolt: &sh("system_profiler", &["SPThunderboltDataType"]),
            ports: &sh("networksetup", &["-listallhardwareports"]),
        },
    )
}

// ---------------------------------------------------------------------------
// Who is on the cable
// ---------------------------------------------------------------------------

/// A machine that answered, and the interface it answered on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peer {
    pub iface: String,
    /// Its link-local address, scoped: what `kvad cluster ping` takes.
    pub addr: String,
}

/// Does this interface look like a cable between two machines?
///
/// The sign is no routable IPv4 address — none at all, or a self-assigned
/// 169.254 — because nothing on the far end of a cable hands out leases. A
/// Wi-Fi or LAN interface has a real address and twenty neighbours, none of
/// which is what anybody is looking for here.
fn is_cable(ifconfig: &str) -> bool {
    ifconfig.lines().filter_map(|l| l.trim().strip_prefix("inet ")?.split(' ').next()).all(|v4| v4.starts_with("169.254."))
}

/// This machine's own link-local addresses on an interface: a multicast
/// probe hears itself first, every time.
fn own(ifconfig: &str) -> Vec<&str> {
    ifconfig.lines().filter_map(|l| l.trim().strip_prefix("inet6 ")?.split(['%', ' ']).next()).filter(|a| a.starts_with("fe80:")).collect()
}

/// Who answered a `ping6` of all-nodes, each once.
fn answered(ping: &str, mine: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in ping.lines() {
        let Some(from) = line.split_once(" from ").map(|(_, rest)| rest) else { continue };
        let addr = from.split(['%', ',', ' ']).next().unwrap_or("");
        if addr.starts_with("fe80:") && !mine.contains(&addr) && !out.iter().any(|a| a == addr) {
            out.push(addr.to_string());
        }
    }
    out
}

/// Ask every cable who else is on it. Returns the interfaces asked, and
/// who answered.
///
/// `ff02::1` is all-nodes: on a link between two machines exactly one thing
/// answers that is not us, and it answers whether or not either side has
/// been configured, because a link-local address needs no configuring. It
/// is the discovery the plan wants done properly by mDNS, done crudely
/// enough to need nothing running at the other end.
pub fn peers(everything: bool) -> (Vec<String>, Vec<Peer>) {
    let (mut asked, mut found) = (Vec::new(), Vec::new());
    if !cfg!(target_os = "macos") {
        return (asked, found);
    }
    for iface in sh("ifconfig", &["-l"]).split_whitespace().filter(|n| n.starts_with("en") || n.starts_with("bridge")) {
        let config = sh("ifconfig", &[iface]);
        let mine = own(&config);
        // Nothing to send from, or not a cable.
        if mine.is_empty() || !(everything || is_cable(&config)) {
            continue;
        }
        asked.push(iface.to_string());
        let ping = sh("ping6", &["-c", "2", "-i", "0.3", &format!("ff02::1%{iface}")]);
        found.extend(answered(&ping, &mine).into_iter().map(|addr| Peer { iface: iface.to_string(), addr: format!("{addr}%{iface}") }));
    }
    (asked, found)
}

#[cfg(test)]
mod tests {
    use super::*;

    // What an M5 MacBook Pro printed on 2026-10-12 with an M2 MacBook Air on
    // a USB-C charging cable, cut to the lines that are read.
    const IOREG_CHARGING_CABLE: &str = "\
+-o Root  <class IORegistryEntry, id 0x100000100, retain 46>
  +-o J714sAP  <class IOPlatformExpertDevice, id 0x10000037c, registered, matched, active, busy 0 (18093 ms), retain 45>
    +-o AppleARMPE  <class AppleARMPE, id 0x100000382, registered, matched, active, busy 0 (9199 ms), retain 59>
      +-o arm-io@10F00000  <class IOPlatformDevice, id 0x1000003be, registered, matched, active, busy 0 (8917 ms), retain 252>
        +-o AppleSoCIO  <class AppleSoCIO, id 0x1000003fa, registered, matched, active, busy 0 (8906 ms), retain 250>
          +-o usb-drd0@A2280000  <class AppleARMIODevice, id 0x1000004a8, registered, matched, active, busy 0 (615 ms), retain 14>
            +-o AppleT8142USBXHCI@00000000  <class AppleT8142USBXHCI, id 0x1000004aa, registered, matched, active, busy 0 (119 ms), retain 99>
              +-o usb-drd0-port-hs@00100000  <class AppleUSB20XHCIARMPort, id 0x100000c16, registered, matched, active, busy 0 (110 ms), retain 17>
                +-o Mac@00100000  <class IOUSBHostDevice, id 0x10000b837, registered, matched, active, busy 0 (92 ms), retain 46>
                  +-o NCM Control@0  <class IOUSBHostInterface, id 0x10000b83c, registered, matched, active, busy 0 (45 ms), retain 9>
                    +-o AppleUSBNCM11Control  <class AppleUSBNCM11Control, id 0x10000b849, registered, matched, active, busy 0 (29 ms), retain 7>
                      +-o AppleUSBNCM11Data  <class AppleUSBNCM11Data, id 0x10000b850, registered, matched, active, busy 0 (22 ms), retain 7>
                        +-o en13  <class IOEthernetInterface, id 0x10000b857, registered, matched, active, busy 0 (70 ms), retain 11>
                          +-o IONetworkStack  <class IONetworkStack, id 0x10000094a, registered, matched, active, busy 0 (18 ms), retain 30>
";

    const BRIDGE_IDLE: &str = "\
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	ether 36:42:63:78:44:c0
	member: en1 flags=3<LEARNING,DISCOVER>
	        ifmaxaddr 0 port 10 priority 0 path cost 0
	member: en6 flags=3<LEARNING,DISCOVER>
	member: en2 flags=3<LEARNING,DISCOVER>
	media: <unknown type>
	status: inactive
";

    const THUNDERBOLT_EMPTY: &str = "\
    Thunderbolt/USB4 Bus 2:
      Port:
          Status: No device connected
          Link Status: 0x100
    Thunderbolt/USB4 Bus 1:
      Port:
          Status: No device connected
          Link Status: 0x100
    Thunderbolt/USB4 Bus 0:
      Port:
          Status: No device connected
          Link Status: 0x100
";

    const PORTS: &str = "\
Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 36:42:63:78:44:c0

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: fc:b2:14:b8:15:37

Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 36:42:63:78:44:c0
";

    const IFCONFIG: &str = "\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
	inet 127.0.0.1 netmask 0xff000000
	inet6 ::1 prefixlen 128
	inet6 fe80::1%lo0 prefixlen 64 scopeid 0x1
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	ether fc:b2:14:b8:15:37
	inet6 fe80::801:686c:7639:cd37%en0 prefixlen 64 secured scopeid 0xf
	inet 192.168.174.180 netmask 0xffffff00 broadcast 192.168.174.255
	media: autoselect
	status: active
en13: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	ether 7e:7a:b0:d3:e5:12
	inet6 fe80::1836:ad27:bdb8:58c3%en13 prefixlen 64 secured scopeid 0x1e
	media: autoselect
	status: active
";

    fn seen<'a>(ioreg: &'a str, bridge: &'a str, thunderbolt: &'a str) -> Seen<'a> {
        Seen { ioreg, iface: "", bridge, thunderbolt, ports: PORTS }
    }

    #[test]
    fn a_charging_cable_is_named_for_what_it_is() {
        let link = classify("en13", &seen(IOREG_CHARGING_CABLE, BRIDGE_IDLE, THUNDERBOLT_EMPTY));
        assert_eq!(link.what, What::UsbNcm(Usb::High));
        assert_eq!(link.title(), "USB NCM, USB 2.0 High Speed: 480 Mb/s");
        assert!(link.caution().unwrap().contains("not Thunderbolt"));
        assert_eq!(
            link.evidence,
            [("ioreg", "AppleUSBNCM11Data".to_string()), ("bridge0", "not a member, inactive".to_string()), ("Thunderbolt", "0 of 3 ports with a device".to_string())]
        );
    }

    // The same pair on the same day with the cable the other way round:
    // this Mac is the USB device, and what it printed for its end.
    const IOREG_DEVICE_END: &str = "\
+-o Root  <class IORegistryEntry, id 0x100000100, retain 46>
  +-o J714sAP  <class IOPlatformExpertDevice, id 0x10000037c, registered, matched, active, busy 0 (18093 ms), retain 45>
    +-o AppleARMPE  <class AppleARMPE, id 0x100000382, registered, matched, active, busy 0 (9199 ms), retain 59>
      +-o arm-io@10F00000  <class IOPlatformDevice, id 0x1000003be, registered, matched, active, busy 0 (8917 ms), retain 252>
        +-o AppleSoCIO  <class AppleSoCIO, id 0x1000003fa, registered, matched, active, busy 0 (8906 ms), retain 250>
          +-o usb-drd1@AA280000  <class AppleARMIODevice, id 0x1000004b0, registered, matched, active, busy 0 (615 ms), retain 14>
            +-o AppleT8142USBXDCI@1  <class AppleT8142USBXDCI, id 0x1000004b2, registered, matched, active, busy 0 (119 ms), retain 99>
              +-o AppleUSBNCMData@1  <class IOUSBDeviceInterface, id 0x10000c001, registered, matched, active, busy 0 (45 ms), retain 9>
                +-o AppleUSBDeviceNCM11Data  <class AppleUSBDeviceNCM11Data, id 0x10000c002, registered, matched, active, busy 0 (22 ms), retain 7>
                  +-o en4  <class IOEthernetInterface, id 0x10000c003, registered, matched, active, busy 0 (70 ms), retain 11>
";

    /// A Mac with a dock on one port and another Mac on a second, as
    /// `system_profiler` lays it out: the dock's own ports are further in.
    const THUNDERBOLT_WITH_A_DOCK: &str = "\
    Thunderbolt/USB4 Bus 2:
      Port:
          Status: No device connected
          Speed: Up to 120 Gb/s
    Thunderbolt/USB4 Bus 1:
      Port:
          Status: Device connected
          Speed: 20 Gb/s
        MacBook Air:
          Vendor Name: Apple Inc.
    Thunderbolt/USB4 Bus 0:
      Port:
          Status: Device connected
          Speed: 40 Gb/s
        ThinkPad Thunderbolt 4 Dock:
          Port (Upstream):
              Status: Device connected
              Speed: 40 Gb/s
          Port:
              Status: No device connected
              Speed: Up to 40 Gb/s
";

    /// The cable looks the same from the Mac that is its device as from
    /// the one that is its host, and is not taken for an Ethernet adapter
    /// because macOS has a port name for that end.
    #[test]
    fn a_charging_cable_is_named_from_its_device_end_too() {
        let ports = format!("{PORTS}\nHardware Port: Ethernet Adapter (en4)\nDevice: en4\nEthernet Address: a2:19:80:e5:b2:f9\n");
        let link = classify("en4", &Seen { ioreg: IOREG_DEVICE_END, iface: "\tmedia: 100baseTX <full-duplex>\n", bridge: BRIDGE_IDLE, thunderbolt: THUNDERBOLT_WITH_A_DOCK, ports: &ports });
        assert_eq!(link.what, What::UsbNcm(Usb::Unsaid), "the device's end names no port, and so no speed");
        assert!(link.caution().unwrap().contains("not Thunderbolt"));
        assert!(link.evidence.contains(&("ioreg", "AppleUSBDeviceNCM11Data".to_string())));
        // Three ports on this machine, two with something on them: the
        // dock's own two ports are the dock's.
        assert!(link.evidence.contains(&("Thunderbolt", "2 of 3 ports with a device".to_string())), "{:?}", link.evidence);
    }

    #[test]
    fn the_same_driver_on_a_faster_port_is_not_called_slow() {
        let ioreg = IOREG_CHARGING_CABLE.replace("usb-drd0-port-hs", "usb-drd0-port-ss");
        assert_eq!(classify("en13", &seen(&ioreg, BRIDGE_IDLE, THUNDERBOLT_EMPTY)).what, What::UsbNcm(Usb::Super));
        let ioreg = IOREG_CHARGING_CABLE.replace("usb-drd0-port-hs", "somewhere");
        assert_eq!(classify("en13", &seen(&ioreg, BRIDGE_IDLE, THUNDERBOLT_EMPTY)).what, What::UsbNcm(Usb::Unsaid));
    }

    /// Not what a machine printed: there has been no Thunderbolt cable to
    /// print it. The shapes are the ones `docs/cluster-plan.md` describes.
    #[test]
    fn thunderbolt_is_known_by_its_driver_or_by_its_bridge() {
        let ioreg = "\
+-o Root  <class IORegistryEntry, id 0x100000100, retain 46>
  +-o AppleThunderboltIPService  <class AppleThunderboltIPService, id 0x1, retain 7>
    +-o AppleThunderboltIPPort  <class AppleThunderboltIPPort, id 0x2, retain 7>
      +-o en1  <class IOEthernetInterface, id 0x3, retain 11>
";
        let up = BRIDGE_IDLE.replace("status: inactive", "status: active");
        let peer = THUNDERBOLT_EMPTY.replacen("No device connected", "Device connected", 1);
        let link = classify("en1", &seen(ioreg, &up, &peer));
        assert_eq!(link.what, What::Thunderbolt);
        assert_eq!(link.caution(), None);
        assert!(link.evidence.contains(&("bridge0", "a member, active".to_string())));
        assert!(link.evidence.contains(&("Thunderbolt", "1 of 3 ports with a device".to_string())));

        // The bridge itself, which is where the address lives.
        let link = classify("bridge0", &seen("", &up, &peer));
        assert_eq!(link.what, What::Thunderbolt);
        assert!(link.evidence.contains(&("bridge0", "active, of en1 en6 en2".to_string())));
        // An idle bridge is only a port with a name.
        assert_eq!(classify("bridge0", &seen("", BRIDGE_IDLE, THUNDERBOLT_EMPTY)).what, What::Port("Thunderbolt Bridge".into()));
    }

    #[test]
    fn a_named_port_keeps_its_name_and_anything_else_is_unidentified() {
        let wifi = classify("en0", &Seen { iface: "\tmedia: autoselect (1000baseT <full-duplex>)\n", ..seen("", BRIDGE_IDLE, THUNDERBOLT_EMPTY) });
        assert_eq!(wifi.what, What::Port("Wi-Fi".into()));
        assert_eq!(wifi.caution(), None);
        assert!(wifi.evidence.contains(&("media", "autoselect (1000baseT <full-duplex>)".to_string())));

        let nothing = classify("en13", &Seen::default());
        assert_eq!(nothing.what, What::Unknown);
        assert!(nothing.evidence[0].1.contains("is the cable in?"));
        assert!(nothing.caution().is_some());

        let lo = classify("lo0", &Seen::default());
        assert_eq!(lo.what, What::Loopback);
        assert!(lo.caution().unwrap().contains("no cable"));
    }

    #[test]
    fn the_registry_path_is_read_root_first() {
        let chain = ancestry(IOREG_CHARGING_CABLE);
        assert_eq!(chain.len(), 14);
        assert_eq!(chain[0], ("Root".to_string(), "IORegistryEntry".to_string()));
        assert_eq!(chain[7], ("usb-drd0-port-hs".to_string(), "AppleUSB20XHCIARMPort".to_string()));
        assert_eq!(chain[9].0, "NCM Control", "a name with a space in it");
        assert_eq!(chain[12].0, "en13");
        assert!(ancestry("").is_empty());
    }

    #[test]
    fn an_address_is_traced_to_its_interface() {
        let at = |ip: &str| owner_of(IFCONFIG, &ip.parse().unwrap());
        assert_eq!(at("127.0.0.1").as_deref(), Some("lo0"));
        assert_eq!(at("::1").as_deref(), Some("lo0"));
        assert_eq!(at("192.168.174.180").as_deref(), Some("en0"));
        assert_eq!(at("fe80::1836:ad27:bdb8:58c3").as_deref(), Some("en13"));
        assert_eq!(at("10.0.0.1"), None);
    }

    #[test]
    fn a_cable_is_an_interface_nobody_gave_an_address() {
        let block = |name: &str| {
            let from = IFCONFIG.find(&format!("{name}: ")).unwrap();
            let rest = &IFCONFIG[from..];
            let to = rest[1..].find("\nen").map_or(rest.len(), |i| i + 2);
            &rest[..to]
        };
        assert!(is_cable(block("en13")));
        assert!(!is_cable(block("en0")));
        assert!(is_cable("en12: flags\n\tinet 169.254.12.7 netmask 0xffff0000\n"));
        assert_eq!(own(block("en13")), ["fe80::1836:ad27:bdb8:58c3"]);
    }

    #[test]
    fn whoever_answers_all_nodes_but_us_is_a_peer_once() {
        let ping = "\
PING6(56=40+8+8 bytes) fe80::1836:ad27:bdb8:58c3%en13 --> ff02::1%en13
16 bytes from fe80::1836:ad27:bdb8:58c3%en13, icmp_seq=0 hlim=64 time=0.094 ms
16 bytes from fe80::ca1:57c7:6946:68a8%en13, icmp_seq=0 hlim=64 time=1.385 ms(DUP!)
16 bytes from fe80::1836:ad27:bdb8:58c3%en13, icmp_seq=1 hlim=64 time=0.122 ms
16 bytes from fe80::ca1:57c7:6946:68a8%en13, icmp_seq=1 hlim=64 time=1.201 ms(DUP!)
";
        assert_eq!(answered(ping, &["fe80::1836:ad27:bdb8:58c3"]), ["fe80::ca1:57c7:6946:68a8"]);
        assert!(answered("", &[]).is_empty());
    }
}
