//! `kvad remote` and `kvad devices`: this machine as a device of a server
//! reached through the tunnel (`docs/relay-plan.md`), and the devices a
//! server has paired.
//!
//! On the server's machine, `kvad remote pair` opens a pairing slot and
//! shows its link as a QR code and as text. On the device, `kvad remote join
//! LINK` pairs, and keeps the key the server makes for it, so that every
//! command takes `--remote kvad://<node id>` after that.

use super::{out, Args};
use kvad::client::{self, Remote, Target};
use serde_json::{json, Value};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn run(args: &Args) -> Res<()> {
    let word = |i: usize| args.words.get(i).map(String::as_str);
    match word(0) {
        Some("id") | None => id(),
        Some("status") => status(args),
        Some("pair") => pair(args),
        Some("join") => match word(1) {
            Some(link) => join(link, args),
            None => Err("join what? `kvad remote join 'kvad://…&pair=…'`, with the link `kvad remote pair` shows".into()),
        },
        Some(other) => Err(format!("`kvad remote {other}` is not a thing: id, status, pair or join").into()),
    }
}

/// The id a server's `[remote] allow` lists for this machine, made on first
/// use. Only the id: the key it is the public half of never leaves its file.
fn id() -> Res<()> {
    let (id, made) = client::device_id()?;
    if made {
        eprintln!("made a device key in {}, readable by you only", client::device_key_path().display());
    }
    println!("{id}");
    Ok(())
}

fn server(args: &Args) -> Res<Remote> {
    match client::resolve(&args.choice)? {
        Target::Remote(remote) => Ok(remote),
        Target::Local { why } => Err(format!("this needs a running kvad-serve, and there is none here ({why})").into()),
    }
}

/// Which path the tunnel to a `kvad://` server is on, and what answers.
fn status(args: &Args) -> Res<()> {
    let remote = server(args)?;
    let health = remote.get("/api/health")?;
    let version = health["version"].as_str().unwrap_or("?");
    println!("server    {} (kvad {version})", remote.base);
    match remote.path() {
        Some(path) => println!("path      {path}"),
        None => println!("path      not through the tunnel"),
    }
    if let Ok(r) = remote.get("/api/remote") {
        match r["enabled"].as_bool() {
            Some(true) => println!("remote    {} · {} allowed", out::s(&r["server"]), r["allowed"]),
            _ => println!("remote    off: no [remote] in its kvad.toml"),
        }
    }
    Ok(())
}

/// A pairing slot on the server, shown as a QR code and as a link.
fn pair(args: &Args) -> Res<()> {
    let remote = server(args)?;
    let made = remote.post("/api/remote/pair", &json!({ "admin": args.admin }))?;
    let link = made["link"].as_str().ok_or("the server opened a slot and sent no link")?;
    let minutes = made["expires_in"].as_u64().unwrap_or(600) / 60;
    if args.json {
        out::json(&made);
        return Ok(());
    }
    println!("{}", qr(link)?);
    println!("Scan that with Kvad on the phone, or on another machine run:\n");
    println!("    kvad remote join '{link}'\n");
    println!(
        "Good once, for {minutes} minutes. The device will act as {} on {}.",
        out::s(&made["role"]),
        out::s(&made["server"])
    );
    println!("Anyone who has this link in that time can pair, so keep it on this screen.");
    Ok(())
}

/// A QR code in the terminal: two rows of modules to a line, dark on light.
fn qr(text: &str) -> Res<String> {
    let code = qrcode::QrCode::with_error_correction_level(text.as_bytes(), qrcode::EcLevel::L)?;
    Ok(code
        .render::<qrcode::render::unicode::Dense1x2>()
        .dark_color(qrcode::render::unicode::Dense1x2::Light)
        .light_color(qrcode::render::unicode::Dense1x2::Dark)
        .quiet_zone(true)
        .build())
}

/// Pairs this machine with the server a link names, and keeps the key it
/// makes for this device.
fn join(link: &str, args: &Args) -> Res<()> {
    let link: kvad_tunnel::PairLink = link.parse()?;
    let name = args.name.clone().unwrap_or_else(super::api::hostname);
    let request = json!({ "v": 1, "token": link.token, "name": name }).to_string();
    let answer = kvad_tunnel::blocking::pair(&client::device_key_path(), &link.address, request.as_bytes())?;
    let answer: Value = serde_json::from_slice(&answer).map_err(|_| "the server's answer was not one a kvad gives")?;
    if let Some(e) = answer["error"].as_str() {
        return Err(e.into());
    }
    let key = answer["key"].as_str().ok_or("the server paired this device and sent no key")?;
    let label = link.address.label();
    let credential = client::Credential {
        key: key.to_string(),
        id: None,
        user: answer["account"].as_str().map(str::to_string),
        address: Some(link.address.to_string()),
    };
    let path = client::remember(&label, &credential)?;
    println!("paired with {label} as \"{}\", acting as {} of {}", out::s(&answer["name"]), out::s(&answer["role"]), out::s(&answer["account"]));
    println!("  key kept in {}, readable by you only; it works from this device alone", path.display());
    println!("  try:  kvad ls --remote {label}");
    Ok(())
}

/// `kvad devices`, `kvad devices rm ID`: the devices paired with a server.
pub fn devices(remote: &Remote, args: &Args) -> Res<()> {
    let word = |i: usize| args.words.get(i).map(String::as_str);
    match word(0) {
        None | Some("ls") => {
            let list = match args.all {
                true => remote.get("/api/devices?all=true")?,
                false => remote.get("/api/devices")?,
            };
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows = list.as_array().cloned().unwrap_or_default();
            if rows.is_empty() {
                println!("no devices paired with {}: kvad remote pair", remote.base);
                return Ok(());
            }
            println!("{:<5} {:<24} {:<6} {:<12} {:<20} LAST SEEN", "ID", "NAME", "ROLE", "ACCOUNT", "NODE");
            for d in rows {
                let role = if d["admin"].as_bool() == Some(true) { "admin" } else { "user" };
                let node = out::s(&d["node"]);
                println!(
                    "{:<5} {:<24} {:<6} {:<12} {:<20} {}",
                    d["id"].to_string(),
                    out::s(&d["name"]),
                    role,
                    out::s(&d["owner"]),
                    format!("{}…", &node[..node.len().min(16)]),
                    d["last_seen"].as_str().unwrap_or("never")
                );
            }
            Ok(())
        }
        Some("rm") => {
            let id = word(1).and_then(|w| w.parse::<i64>().ok()).ok_or("which device? `kvad devices rm ID`, with the ID `kvad devices` shows")?;
            remote.delete(&format!("/api/devices/{id}"))?;
            println!("unpaired device {id} from {}: its key is revoked and its connection closed", remote.base);
            Ok(())
        }
        Some(other) => Err(format!("`kvad devices {other}` is not a thing: ls or rm").into()),
    }
}
