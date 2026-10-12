//! `kvad cluster`: what there is so far of one model across several
//! machines, which is the measuring of the links between them.
//!
//! The link is identified before any number is printed, and on every run,
//! because of the one way this goes wrong without anybody noticing: a
//! charging cable presents as a working link. See `kvad::cluster::link`.

use super::Res;
use crate::Args;
use kvad::cluster::{discover, key, link, ping, session, wire, worker as far};
use std::net::TcpListener;

use super::spec::CLUSTER as USAGE;

pub fn run(args: &Args) -> Res<()> {
    let word = |i: usize| args.words.get(i).map(String::as_str);
    match word(0) {
        Some("find") => find(args.all),
        Some("plan") => plan(args, word(1)),
        Some("status") => status(args),
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
    // Workers, who say so; then machines on a cable, who need not be.
    let workers = discover::workers();
    for worker in &workers {
        let ours = if worker.build == kvad::cluster::build() { "" } else { ", which is not this build" };
        println!("worker {} ({}, kvad {}{ours})", worker.name, worker.id, worker.build);
        for address in &worker.addresses {
            println!("  {}", kvad::cluster::path::with_port(address, worker.port));
        }
    }
    let (asked, found) = link::peers(everything);
    for peer in &found {
        println!("{}\t{}", peer.iface, peer.addr);
    }
    if !workers.is_empty() && found.is_empty() {
        return Ok(());
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
    if let Some(ms) = args.gap {
        options.gap = std::time::Duration::from_secs_f64(ms.clamp(0.0, 10_000.0) / 1e3);
    }
    if let Some(mb) = args.stream_mb {
        options.stream = mb.min(4095) << 20;
    }

    let port = args.port.unwrap_or(wire::PORT);
    let (mut stream, peer) = ping::connect(host, port).map_err(|e| {
        format!("cannot reach a listener at {host}, port {port}: {e}\nOn that machine: kvad cluster listen")
    })?;

    // What the link is, before a single number.
    let (local, there) = (stream.local_addr()?, stream.peer_addr()?);
    let link = link::between(&local.ip(), &there.ip())
        .unwrap_or_else(|| link::Link { iface: local.ip().to_string(), what: link::What::Unknown, evidence: Vec::new() });
    show(&link);
    let memory = peer.memory.map(|b| format!(", {:.0} GB for a model", b as f64 / 1e9)).unwrap_or_default();
    println!("  peer            {} (kvad {}, {} threads{memory})", peer.name, peer.version, peer.threads);
    if peer.version != kvad::cluster::build() {
        println!("  -> this is kvad {}. The link does not mind; a split model will.", kvad::cluster::build());
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
    let (name, id) = (kvad::cluster::hostname(), kvad::cluster::make_id()?);
    println!("{name} ({id}, kvad {}) is a worker on port {port}; ^C to stop", kvad::cluster::build());
    // Kept until the worker ends, and withdrawn when it does.
    let _advert = match discover::advertise(&name, port, &id, &kvad::cluster::build()) {
        Ok(advert) => {
            println!("found on the link as `{name}`: --workers {name}");
            Some(advert)
        }
        Err(why) => {
            println!("{why}");
            None
        }
    };

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

/// The longest conversation a plan leaves room for unless told, where the
/// model allows it: 8192 tokens.
pub const CONTEXT: usize = 8192;

/// A model's shape and its size, from this machine's disk if it is there
/// and from the Hub's few kilobytes about it if not. Nothing is fetched
/// that is a weight.
fn shape_of(model: &str) -> Res<(kvad::model::Spec, Option<u64>)> {
    use kvad::model::{Json, Spec};
    if let Some(local) = kvad::hub::find_local(model).filter(|m| m.complete) {
        let files = kvad::weights::fetch_with(model, &mut |_| {})?;
        return Ok((Spec::from_json(&files.config)?, local.params));
    }
    if let Some(dir) = kvad::hub::find_trained(model) {
        return Ok((Spec::from_json(&dir.path.join("config.json"))?, dir.params));
    }
    let config = kvad::hub::remote_config(model).ok_or_else(|| format!("{model} is not on this machine, and the Hub did not answer for its config"))?;
    let params = kvad::hub::search(model, 10).ok().and_then(|found| found.into_iter().find(|m| m.id.eq_ignore_ascii_case(model))).and_then(|m| m.params);
    Ok((Spec::from_config(Json::new(config))?, params))
}

/// `kvad cluster plan MODEL --workers ...`: which machine would hold
/// which layers, before anything is downloaded or loaded.
fn plan(args: &Args, model: Option<&str>) -> Res<()> {
    let model = model.or(args.model.as_deref()).ok_or("which model? `kvad cluster plan MODEL --workers NAME`")?;
    let workers: Vec<String> = args.workers.as_deref().unwrap_or("").split(',').map(|w| w.trim().to_string()).filter(|w| !w.is_empty()).collect();
    let split = args.split.as_deref().map(session::parse_split).transpose()?;
    let (spec, params) = shape_of(model)?;
    let context = args.context.unwrap_or(spec.n_ctx.min(CONTEXT));
    println!("{model} at {}: {}", args.quant, spec.summary());
    match params {
        Some(params) => {
            let sizes = kvad::cluster::plan::Sizes::of(&spec, params, args.quant, context);
            println!(
                "{:.1} GB of weights and {:.1} GB of cache for a conversation of {context} tokens",
                (sizes.ends + sizes.n_layer as u64 * sizes.layer) as f64 / 1e9,
                (sizes.n_layer as u64 * sizes.cache) as f64 / 1e9
            );
        }
        None => println!("its size is not known here, so memory cannot be checked"),
    }

    // A plan for this machine alone needs nobody's key.
    let key = match workers.is_empty() {
        true => None,
        false => Some(key::Key::ours()?),
    };
    let nobody = key::Key::from_hex(&"00".repeat(32))?;
    let key = key.as_ref().unwrap_or(&nobody);
    let (nodes, _) = session::survey(&spec, args.quant, &workers, key, &mut |line| eprintln!("{line}"))?;
    // The files are not wanted, only what is known about the model.
    let files = kvad::weights::ModelFiles { weights: Vec::new(), tokenizer: Default::default(), config: Default::default(), tokenizer_config: None, generation_config: None };
    let mix = args.place.as_deref().map(kvad::cluster::plan::Mix::parse).transpose()?.unwrap_or(kvad::cluster::plan::Mix::REPLY);
    let want = session::Across { repo: model, files: &files, spec: &spec, precision: args.quant, workers: &workers, split: split.as_deref(), key, params, context, mix };
    let plan = session::planned(&want, &nodes)?;
    println!();
    for line in kvad::cluster::plan::table(&nodes, &plan) {
        println!("{line}");
    }
    let used: Vec<&str> = workers.iter().zip(&plan.shares[1..]).filter(|(_, share)| !share.layers.is_empty()).map(|(w, _)| w.as_str()).collect();
    println!();
    match used.is_empty() {
        true => println!("It fits on this machine, where a layer costs no round trip: kvad run --model {model}"),
        false => println!("kvad run --model {model} --workers {} --split {}", workers.join(","), plan.split()),
    }
    Ok(())
}

/// `kvad cluster status`: a server's part in a cluster, asked of the server.
fn status(args: &Args) -> Res<()> {
    let remote = match super::target(args)? {
        kvad::client::Target::Remote(remote) => remote,
        kvad::client::Target::Local { .. } => super::no_server("cluster"),
    };
    let cluster = remote.get("/api/cluster")?;
    if args.json {
        super::out::json(&cluster);
        return Ok(());
    }
    let s = super::out::s;
    println!("kvad {}", s(&cluster["build"]));
    match cluster["worker"].is_object() {
        true => println!(
            "a worker for other machines: {} ({}) on port {}{}",
            s(&cluster["worker"]["name"]),
            s(&cluster["worker"]["id"]),
            s(&cluster["worker"]["port"]),
            if cluster["worker"]["advertised"] == true { "" } else { ", not advertised: name it by its address" }
        ),
        false => println!("not a worker for other machines: [worker] enabled = true in kvad.toml makes it one"),
    }
    let named: Vec<String> = super::out::items(&cluster["workers"]).iter().map(s).collect();
    match named.is_empty() {
        true => println!("spreads its models over nobody: [cluster] workers names the machines"),
        false => println!(
            "spreads a model too large for it over {}, placed for {}{}",
            named.join(", "),
            s(&cluster["place"]),
            if cluster["split"].is_string() { format!(", split {} by hand", s(&cluster["split"])) } else { String::new() }
        ),
    }
    if cluster["has_key"] != true {
        println!("no cluster key here: a worker makes one the first time it starts, and every machine needs the same");
    }
    let found = super::out::items(&cluster["found"]);
    if found.is_empty() {
        println!("\nNo worker says it is on the network.");
        return Ok(());
    }
    println!();
    let rows: Vec<Vec<String>> = found
        .iter()
        .map(|w| {
            let mut notes = Vec::new();
            if w["this_machine"] == true {
                notes.push("this server");
            }
            if w["named"] == true {
                notes.push("named in [cluster]");
            }
            if w["same_build"] != true {
                notes.push("another build, and would be refused");
            }
            let addresses: Vec<String> = super::out::items(&w["addresses"]).iter().map(s).collect();
            vec![s(&w["name"]), s(&w["id"]), s(&w["build"]), addresses.len().to_string(), notes.join("; ")]
        })
        .collect();
    super::out::table(&["WORKER", "ID", "BUILD", "ADDRESSES", ""], &rows);
    Ok(())
}
