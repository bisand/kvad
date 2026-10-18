//! `kvad remote`: this machine as a device of a server reached through the
//! tunnel (`docs/relay-plan.md`), with `--remote kvad://…`.

use super::Args;
use kvad::client::{self, Target};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn run(args: &Args) -> Res<()> {
    match args.words.first().map(String::as_str) {
        Some("id") | None => id(),
        Some("status") => status(args),
        Some(other) => Err(format!("`kvad remote {other}` is not a thing: `id` or `status`").into()),
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

/// Which path the tunnel to a `kvad://` server is on, and what answers.
fn status(args: &Args) -> Res<()> {
    let Target::Remote(remote) = client::resolve(&args.choice)? else {
        return Err("which server? `kvad remote status --remote kvad://…`".into());
    };
    let health = remote.get("/api/health")?;
    let version = health["version"].as_str().unwrap_or("?");
    println!("server    {} (kvad {version})", remote.base);
    match remote.path() {
        Some(path) => println!("path      {path}"),
        None => println!("path      not through the tunnel"),
    }
    Ok(())
}
