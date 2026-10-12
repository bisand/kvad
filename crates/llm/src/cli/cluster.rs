//! `kvad cluster`: what there is so far of one model across several
//! machines, which is the measuring of the links between them.
//!
//! The link is identified before any number is printed, and on every run,
//! because of the one way this goes wrong without anybody noticing: a
//! charging cable presents as a working link. See `kvad::cluster::link`.

use super::Res;
use crate::Args;
use kvad::cluster::{key, link, ping, wire, worker as far};
use std::net::TcpListener;

use super::spec::CLUSTER as USAGE;

pub fn run(args: &Args) -> Res<()> {
    let word = |i: usize| args.words.get(i).map(String::as_str);
    match word(0) {
        Some("find") => find(args.all),
        Some("listen") => listen(args.port.unwrap_or(wire::PORT)),
        Some("ping") => measure(args, word(1)),
        Some("link") => match word(1) {
            Some(iface) => {
                show(&link::identify(iface));
                Ok(())
            }
            None => Err("which interface? `kvad cluster link en12`; `kvad cluster find` lists the cables.".into()),
        },
        Some(other) => {
            eprintln!("`kvad cluster {other}` is not a command.\n\n{USAGE}");
            std::process::exit(2);
        }
        None => {
            println!("{USAGE}");
            Ok(())
        }
    }
}

fn show(link: &link::Link) {
    println!("link on {}: {}", link.iface, link.title());
    for (what, said) in &link.evidence {
        println!("  {what:<15} {said}");
    }
    if let Some(caution) = link.caution() {
        println!("  -> {caution}");
    }
}

fn find(everything: bool) -> Res<()> {
    let (asked, found) = link::peers(everything);
    for peer in &found {
        println!("{}\t{}", peer.iface, peer.addr);
    }
    eprintln!(
        "asked {}{}",
        if asked.is_empty() { "no interface: none looks like a cable".to_string() } else { asked.join(", ") },
        if everything { "" } else { "; --all to ask Wi-Fi and the LAN too" }
    );
    if found.is_empty() {
        eprintln!(
            "Nothing answered. The other machine has to be awake and the cable in: an\n\
             unplugged USB link does not go down, it stops existing."
        );
        std::process::exit(1);
    }
    Ok(())
}

fn listen(port: u16) -> Res<()> {
    // `::` and not `0.0.0.0`: a cable's addresses are link-local IPv6, and
    // an IPv6 socket takes IPv4 as well.
    let listener = TcpListener::bind(("::", port)).or_else(|_| TcpListener::bind(("0.0.0.0", port))).map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    println!("listening on port {port}; ^C to stop");
    println!("It sends back what it is sent, to anybody who can reach this port, and nothing else.");
    ping::listen(&listener, &mut |event| match event {
        ping::Event::Came(from, hello) => println!("  <- {} ({}, kvad {})", from.ip(), hello.name, hello.version),
        ping::Event::Went(_, how) => println!("     {how}"),
    })?;
    Ok(())
}

fn measure(args: &Args, host: Option<&str>) -> Res<()> {
    let found;
    let host = match host {
        Some(host) => host,
        None => {
            let (asked, mut peers) = link::peers(false);
            match peers.len() {
                1 => {
                    found = peers.remove(0).addr;
                    eprintln!("found {found}");
                    &found
                }
                0 => return Err(format!("no machine named, and none found on a cable ({}).\n`kvad cluster ping HOST` names one.", if asked.is_empty() { "there is no cable".into() } else { format!("asked {}", asked.join(", ")) }).into()),
                _ => return Err(format!("no machine named, and {} found: {}.\n`kvad cluster ping HOST` names one.", peers.len(), peers.iter().map(|p| p.addr.as_str()).collect::<Vec<_>>().join(", ")).into()),
            }
        }
    };
    let mut options = ping::Options::default();
    if let Some(rounds) = args.rounds {
        options.rounds = rounds.max(1);
    }
    if let Some(sizes) = &args.sizes {
        options.sizes = ping::parse_sizes(sizes)?;
    }
    if let Some(mb) = args.stream_mb {
        options.stream = mb.min(4095) << 20;
    }

    let port = args.port.unwrap_or(wire::PORT);
    let (mut stream, peer) = ping::connect(host, port).map_err(|e| {
        format!("cannot reach a listener at {host}, port {port}: {e}\nOn that machine: kvad cluster listen")
    })?;

    // What the link is, before a single number.
    let local = stream.local_addr()?;
    let link = match link::interface_of(&local.ip()) {
        Some(iface) => link::identify(&iface),
        None => link::Link { iface: local.ip().to_string(), what: link::What::Unknown, evidence: Vec::new() },
    };
    show(&link);
    let memory = peer.memory.map(|b| format!(", {:.0} GB for a model", b as f64 / 1e9)).unwrap_or_default();
    println!("  peer            {} (kvad {}, {} threads{memory})", peer.name, peer.version, peer.threads);
    if peer.version != env!("CARGO_PKG_VERSION") {
        println!("  -> this is kvad {}. The link does not mind; a split model will.", env!("CARGO_PKG_VERSION"));
    }
    println!();

    let rounds = options.rounds;
    let got = ping::measure(&mut stream, &options, &mut |round| {
        eprintln!("  round {round}/{rounds}{}", if round == 0 { " (warm-up, not kept)" } else { "" });
    })?;

    println!();
    println!("| bytes | round trip, median | least | link rate, 2x bytes / round trip | median / least |");
    println!("|---|---|---|---|---|");
    let mut noisy = false;
    for (i, &n) in got.sizes.iter().enumerate() {
        let (median, least) = (got.median(i), got.least(i));
        let note = match n {
            ping::DECODE_HOP => " (decode hop)",
            ping::PREFILL_CHUNK => " (prefill chunk)",
            _ => "",
        };
        // A median well off its own minimum is a busy machine, not a link.
        let spread = match median > least * 1.5 {
            true => {
                noisy = true;
                format!(" {:.1}x", median / least)
            }
            false => String::new(),
        };
        println!("| {}{note} | {median:.3} ms | {least:.3} ms | {:.0} Mb/s |{spread} |", ping::size_label(n), got.rate(i));
    }
    let (empty, bandwidth) = got.fit();
    println!();
    println!("empty-hop latency   {empty:.3} ms      (the fitted intercept)");
    println!("fitted bandwidth    {bandwidth:.0} Mb/s   (from the slope)");
    if got.stream_bytes > 0 {
        println!("one-way ceiling     {:.0} Mb/s   ({} in one direction)", got.ceiling(), ping::size_label(got.stream_bytes));
    }
    println!();
    match noisy {
        true => println!("Some medians sit well off their minimums: a machine was busy, and this run is\nworth repeating rather than quoting."),
        false => println!("Every median sits on its minimum, which is what a quiet run looks like."),
    }
    Ok(())
}

/// `kvad worker`: hold layers for a coordinator that has the key.
pub fn worker(args: &Args) -> Res<()> {
    let path = key::path();
    let (key, new) = key::Key::read_or_create(&path)?;
    match new {
        // Once, and to the terminal that started it: after this it is a
        // file only its owner can read.
        true => println!("cluster key: {}\n  new, and kept in {}. Every machine of the cluster needs the same one:\n  copy that file to them, or set {} there.", key.to_hex(), path.display(), key::ENV),
        false => println!("cluster key: the one in {}", path.display()),
    }
    let port = args.port.unwrap_or(wire::PORT);
    // `::` and not `0.0.0.0`: a cable's addresses are link-local IPv6, and
    // an IPv6 socket takes IPv4 as well.
    let listener = match &args.host {
        Some(host) => TcpListener::bind((host.as_str(), port)),
        None => TcpListener::bind(("::", port)).or_else(|_| TcpListener::bind(("0.0.0.0", port))),
    }
    .map_err(|e| format!("cannot listen on port {port}: {e}"))?;
    println!("{} is a worker on port {port}; ^C to stop", kvad::cluster::hostname());

    let started = std::time::Instant::now();
    far::serve(&listener, &key, &mut far::from_disk(), &mut |event| {
        let at = started.elapsed().as_secs_f32();
        match event {
            far::Event::Came(from, hello) => println!("[{at:>9.3}s] <- {} ({}, kvad {})", from.ip(), hello.name, hello.version),
            far::Event::Progress(message) => println!("[{at:>9.3}s]    {message}"),
            far::Event::Loaded(load, ready) => println!(
                "[{at:>9.3}s]    holding layers {}..{} of {} at {}: {:.1}M parameters, {:.0} MB",
                load.layers[0],
                load.layers[1],
                load.repo,
                load.precision,
                ready.params as f64 / 1e6,
                ready.bytes as f64 / 1e6
            ),
            far::Event::Went(from, how) => println!("[{at:>9.3}s] -> {} gone: {how}", from.ip()),
        }
    })?;
    Ok(())
}
