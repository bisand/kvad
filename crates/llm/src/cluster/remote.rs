//! A stage on another machine: a socket where [`super::stage::LocalStage`]
//! has a model.
//!
//! Rows go out in a `Forward` and come back in a `Hidden`, and that round
//! trip is all a token costs a cluster beyond what its layers cost. Nothing
//! else is here because nothing else crosses: the stage's cache stays with
//! the stage.
//!
//! When the socket fails, the stage is gone and says which machine it was.
//! Nothing is retried and nothing re-planned: a reply that lost a stage
//! half way through has lost that stage's cache with it, and the honest
//! thing to tell whoever asked is that it failed and where.

use super::key::{self, Key, Role};
use super::stage::{Fault, Stage};
use super::wire::{self, Hello, Kind, Load, Ready, Rows, Truncate};
use crate::model::Shard;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// How long to wait for a worker to be there at all.
const CONNECT: Duration = Duration::from_secs(5);
/// How long a worker may take to answer a greeting. One that is there
/// answers in a millisecond.
const GREETING: Duration = Duration::from_secs(10);
/// How long a rewind or a pass may take. A pass is a stage's
/// layers over a chunk of prompt, which on a slow machine with a large
/// model is not quick; a worker that has gone is noticed by this.
const PATIENCE: Duration = Duration::from_secs(300);

pub struct RemoteStage {
    stream: TcpStream,
    /// The worker's name for itself and where it was reached, for errors.
    name: String,
    shard: Shard,
    held: Ready,
    out: Vec<u8>,
    back: Vec<u8>,
}

/// The coordinator's half of the handshake in [`super::key`].
fn greet(stream: &mut TcpStream, key: &Key, at: &str, waited: Duration) -> Result<Hello, Fault> {
    let ours = Hello { nonce: key::nonce()?, ..Hello::ours() };
    ours.send(stream)?;
    let mut buf = Vec::new();
    let answer = wire::read(stream, 1 << 16, &mut buf).map_err(|e| match e.kind() {
        // Found across a cable: macOS's firewall lets the connection be
        // made, holds every byte of it, and asks on that machine's screen
        // whether to allow the program. From here that is a worker that
        // took the call and never spoke.
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => format!(
            "{at} took the connection and said nothing for {} seconds.\nA firewall on that machine that has not been told to allow kvad looks exactly like this:\nlook for its question on that machine's screen.",
            waited.as_secs()
        ),
        _ => format!("{at}: {e}"),
    })?;
    let theirs = match answer {
        Kind::Hello => Hello::parse(&buf)?,
        Kind::Error => return Err(format!("{at} refused: {}", wire::error_text(&buf)).into()),
        kind => return Err(format!("{at} sent a {kind:?} where a greeting was due; is that a kvad worker?").into()),
    };
    if theirs.nonce.is_empty() {
        return Err(format!("{at} is `kvad cluster listen`, which measures a link and runs nothing; a worker is `kvad worker`").into());
    }
    // Before anything is asked of it: two builds with different kernels can
    // both be right and not agree to the bit, and a model split across
    // them is then a bug report nobody can act on.
    if theirs.version != ours.version {
        return Err(format!("{} at {at} is kvad {} and this is kvad {}; a split model wants one build on every machine", theirs.name, theirs.version, ours.version).into());
    }
    wire::send(stream, Kind::Auth, &serde_json::json!({ "proof": key.proof(Role::Coordinator, &ours.nonce, &theirs.nonce) }))?;
    let proof = match wire::read(stream, 1 << 16, &mut buf)? {
        Kind::Auth => wire::parse::<serde_json::Value>(Kind::Auth, &buf)?["proof"].as_str().unwrap_or("").to_string(),
        Kind::Error => return Err(format!("{} at {at} refused: {}", theirs.name, wire::error_text(&buf)).into()),
        kind => return Err(format!("{at} sent a {kind:?} where proof of the cluster key was due").into()),
    };
    if !key.accepts(Role::Worker, &ours.nonce, &theirs.nonce, &proof) {
        return Err(format!("{} at {at} does not hold this cluster's key", theirs.name).into());
    }
    Ok(theirs)
}

/// Reach the worker at `address` and show each other the key. What comes
/// back is a connection a worker will take frames on, and who it is.
pub fn connect(address: &str, key: &Key) -> Result<(TcpStream, Hello), Fault> {
    connect_within(address, key, CONNECT, GREETING)
}

/// [`connect`], giving up sooner: for an address that is one of several
/// and may well lead nowhere.
pub fn connect_within(address: &str, key: &Key, reach: Duration, greeting: Duration) -> Result<(TcpStream, Hello), Fault> {
    let (host, port) = wire::host_port(address);
    let mut last: Fault = format!("{host} names no address").into();
    let mut stream = None;
    for addr in (host.as_str(), port).to_socket_addrs().map_err(|e| format!("{address}: {e}"))? {
        match TcpStream::connect_timeout(&addr, reach) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(e) => last = format!("no worker at {address}: {e}\nOn that machine: kvad worker").into(),
        }
    }
    let Some(mut stream) = stream else { return Err(last) };
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(greeting))?;
    stream.set_write_timeout(Some(PATIENCE))?;
    let hello = greet(&mut stream, key, address, greeting)?;
    stream.set_read_timeout(Some(PATIENCE))?;
    Ok((stream, hello))
}

/// Ask the worker at `address` how long a layer of this shape takes it,
/// in milliseconds: [`super::plan::bench`], run there.
pub fn bench(address: &str, key: &Key, n_embd: usize, intermediate: usize, precision: crate::quant::Precision) -> Result<f64, Fault> {
    let (mut stream, hello) = connect(address, key)?;
    wire::send(&mut stream, Kind::Bench, &wire::Bench { n_embd, intermediate, precision: precision.to_string() })?;
    let mut buf = Vec::new();
    match wire::read(&mut stream, 1 << 16, &mut buf).map_err(|e| format!("{} ({address}): {e}", hello.name))? {
        Kind::Benched => wire::parse::<serde_json::Value>(Kind::Benched, &buf)?["layer_ms"]
            .as_f64()
            .filter(|ms| ms.is_finite() && *ms > 0.0)
            .ok_or_else(|| format!("{} timed a layer and gave no time", hello.name).into()),
        Kind::Error => Err(format!("{} ({address}): {}", hello.name, wire::error_text(&buf)).into()),
        kind => Err(format!("{} answered a timing with a {kind:?}", hello.name).into()),
    }
}

impl RemoteStage {
    /// [`connect`], and then have the worker load `layers` of `repo`.
    /// Returns when the stage is ready to run, which for a model the worker
    /// has to fetch or quantise is a while; what it says in the meantime
    /// goes to `progress`.
    pub fn open(address: &str, key: &Key, load: &Load, progress: &mut dyn FnMut(&str)) -> Result<RemoteStage, Fault> {
        let (mut stream, hello) = connect(address, key)?;
        let name = format!("{} ({address})", hello.name);

        // A load is as long as it is: a download, or the quantising of
        // forty layers, with a line now and then.
        stream.set_read_timeout(None)?;
        wire::send(&mut stream, Kind::Load, load)?;
        let mut buf = Vec::new();
        let held = loop {
            match wire::read(&mut stream, 1 << 16, &mut buf).map_err(|e| format!("{name} went away while loading: {e}"))? {
                Kind::Progress => {
                    let line = wire::parse::<serde_json::Value>(Kind::Progress, &buf)?;
                    progress(&format!("{}: {}", hello.name, line["message"].as_str().unwrap_or("")));
                }
                Kind::Ready => break wire::parse::<Ready>(Kind::Ready, &buf)?,
                Kind::Error => return Err(format!("{name} could not load {}: {}", load.repo, wire::error_text(&buf)).into()),
                kind => return Err(format!("{name} sent a {kind:?} in the middle of a load").into()),
            }
        };
        stream.set_read_timeout(Some(PATIENCE))?;
        let shard = Shard { layers: load.layers[0]..load.layers[1], ends: false };
        Ok(RemoteStage { stream, name, shard, held, out: Vec::new(), back: Vec::new() })
    }

    /// One frame out and the answer to it, or why there was none.
    fn ask(&mut self, kind: Kind, most: usize) -> Result<Kind, Fault> {
        let gone = |name: &str, shard: &Shard, e: std::io::Error| -> Fault { format!("{name}, which holds {shard}, went away: {e}").into() };
        wire::write(&mut self.stream, kind, &self.out).map_err(|e| gone(&self.name, &self.shard, e))?;
        match wire::read(&mut self.stream, most, &mut self.back).map_err(|e| gone(&self.name, &self.shard, e))? {
            Kind::Error => Err(format!("{}: {}", self.name, wire::error_text(&self.back)).into()),
            kind => Ok(kind),
        }
    }
}

impl Stage for RemoteStage {
    fn shard(&self) -> Shard {
        self.shard.clone()
    }

    fn run(&mut self, seq: u64, pos0: usize, m: usize, xs: Vec<f32>) -> Result<Vec<f32>, Fault> {
        if m == 0 {
            return Ok(xs);
        }
        let sent = Rows { seq, pos0: pos0 as u32, m: m as u32, n_embd: (xs.len() / m) as u32, xs };
        sent.encode(&mut self.out);
        match self.ask(Kind::Forward, self.out.len())? {
            Kind::Hidden => {}
            kind => return Err(format!("{} answered rows with a {kind:?}", self.name).into()),
        }
        let back = Rows::decode(&self.back)?;
        // The rows that were asked for, and not some other conversation's
        // or some other position's: a worker that answered the wrong
        // question is found out here and not in the reply.
        if (back.seq, back.pos0, back.m, back.n_embd) != (sent.seq, sent.pos0, sent.m, sent.n_embd) {
            return Err(format!("{} answered for other rows than it was sent", self.name).into());
        }
        Ok(back.xs)
    }

    fn truncate(&mut self, seq: u64, len: usize) -> Result<usize, Fault> {
        self.out = serde_json::to_vec(&Truncate { seq, len })?;
        match self.ask(Kind::Truncate, 1 << 16)? {
            Kind::Truncated => Ok(wire::parse::<Truncate>(Kind::Truncated, &self.back)?.len),
            kind => Err(format!("{} answered a rewind with a {kind:?}", self.name).into()),
        }
    }

    fn holds(&self) -> (usize, usize) {
        (self.held.params, self.held.bytes)
    }

    fn elsewhere(&self) -> bool {
        true
    }

    fn inside(&mut self) -> Option<wire::Spent> {
        self.out.clear();
        match self.ask(Kind::Spend, 1 << 16) {
            Ok(Kind::Spent) => wire::parse(Kind::Spent, &self.back).ok(),
            _ => None,
        }
    }
}
