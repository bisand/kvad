//! A worker: some of a model's blocks, for a coordinator somewhere else.
//!
//! It listens, and to a coordinator that holds the cluster key it is a
//! stage: told which layers of which model to hold, it loads them, and then
//! takes rows of the residual stream and gives rows back until told
//! otherwise. It has no tokens, no tokenizer, no sampler, no HTTP and no
//! database; it never sees a prompt, only what a prompt has become some
//! layers in.
//!
//! One model at a time. A stage does one thing at a time, and a worker
//! holding layers for two coordinators would be two models in the memory
//! of one, so the second to ask is told it is busy. Asking is all it is
//! refused: any number of coordinators may greet a worker and measure the
//! way to it while it works, which is how one that was found on five
//! addresses is tried on all five at once.
//!
//! When the coordinator goes — hangs up, or stops answering the kernel's
//! probes — the stage and its caches are dropped and the worker goes back
//! to listening. Nothing is kept for a coordinator that might return.

use super::key::{self, Key, Role};
use super::ping::MOST_ECHOED;
use super::stage::{Fault, LocalStage, Stage};
use super::wire::{self, Hello, Kind, Load, Ready, Rows, Truncate};
use crate::model::{Shard, Spec};
use crate::quant::Precision;
use crate::weights::Watcher;
use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How often a coordinator is told where one file's download has got to.
const OFTEN: Duration = Duration::from_millis(100);

/// The longest tensor frame taken: a prefill chunk of the widest model
/// there is comes to a few megabytes, and this is not a place to find out
/// what a stranger with the key can make a machine allocate.
const MOST_ROWS: usize = 256 << 20;

/// How long a handshake may take. A coordinator that is there answers at
/// once.
const GREETING: Duration = Duration::from_secs(30);

/// What turns a [`Load`] into something that runs. It comes from whoever
/// starts the worker, as a session's backend does, so that a binary with a
/// GPU in it can make GPU stages and this crate need not know.
///
/// It is given two places to say how it is getting on, as every load in
/// this crate is: lines for a person, and a download's bytes for a bar.
/// Both cross to the coordinator.
pub type Factory = Box<dyn FnMut(&Load, &mut dyn FnMut(&str), &Watcher) -> Result<Box<dyn Stage>, Fault> + Send>;

/// The stages this crate can make by itself: the model's files from this
/// machine's disk, or the Hub if it has not got them, and the shard read
/// from them onto the CPU.
pub fn from_disk() -> Factory {
    Box::new(|load, progress, watch| {
        let precision = Precision::parse(&load.precision).ok_or_else(|| format!("`{}` is not a precision", load.precision))?;
        let shard = Shard { layers: load.layers[0]..load.layers[1], ends: false };
        // Its own layers' files, and none of the rest of the checkpoint.
        let files = crate::weights::fetch_for(&load.repo, &|tensor| shard.wants(tensor), progress, watch).map_err(|e| e.to_string())?;
        let spec = Spec::from_json(&files.config).map_err(|e| e.to_string())?;
        progress(&format!("reading {shard} of {} at {precision}", load.repo));
        let model = crate::qcache::load_shard(&load.repo, &files, &spec, precision, &shard, progress).map_err(|e| e.to_string())?;
        Ok(Box::new(LocalStage::new(Arc::from(model))) as Box<dyn Stage>)
    })
}

/// Whether a `Load` names a model and not a place: a Hub repo id or one of
/// this machine's own names, and never a path a coordinator chose.
pub fn names_a_model(repo: &str) -> bool {
    let plain = |part: &str| {
        !part.is_empty() && !part.starts_with('.') && part.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    let parts: Vec<&str> = repo.split('/').collect();
    parts.len() <= 2 && parts.iter().all(|part| plain(part))
}

/// What a worker has to say about a connection.
pub enum Event<'a> {
    Came(SocketAddr, &'a Hello),
    Progress(&'a str),
    Loaded(&'a Load, &'a Ready),
    Went(SocketAddr, String),
}

/// Keep a connection that has gone quiet honest: the kernel asks after a
/// minute's silence whether the other end is still there, and a coordinator
/// whose machine slept or lost its cable stops holding the worker.
fn keepalive(stream: &TcpStream) {
    use std::os::fd::AsRawFd;
    let (fd, on, idle) = (stream.as_raw_fd(), 1 as libc::c_int, 60 as libc::c_int);
    let size = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    #[cfg(target_os = "macos")]
    let idle_option = libc::TCP_KEEPALIVE;
    #[cfg(not(target_os = "macos"))]
    let idle_option = libc::TCP_KEEPIDLE;
    // SAFETY: the descriptor is open for as long as `stream` is borrowed,
    // and each value is a `c_int` of the size given.
    unsafe {
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_KEEPALIVE, (&on as *const libc::c_int).cast(), size);
        libc::setsockopt(fd, libc::IPPROTO_TCP, idle_option, (&idle as *const libc::c_int).cast(), size);
    }
}

/// What the connections of one worker share.
struct Shared<'f, 's> {
    /// Whether some connection holds a stage.
    busy: std::sync::atomic::AtomicBool,
    factory: std::sync::Mutex<&'f mut Factory>,
    said: std::sync::Mutex<&'s mut (dyn FnMut(Event) + Send)>,
    room: Room<'s>,
    bench: Bencher<'s>,
}

/// How long a layer takes this worker some other way than on its CPU: for
/// a `Bench` whose precision is not a CPU's, such as `gpu-q8`. `None` is
/// "not a way this worker runs layers", and the coordinator is told so.
pub type Bencher<'b> = &'b (dyn Fn(&wire::Bench) -> Option<Result<super::plan::Speed, String>> + Sync);

/// What this worker has for a coordinator's layers, in bytes, asked at
/// every greeting: a worker that is also a server has what its own models
/// have left, and that changes. `None` is the machine's usable memory.
pub type Room<'r> = &'r (dyn Fn() -> Option<u64> + Sync);

impl Shared<'_, '_> {
    fn say(&self, event: Event) {
        if let Ok(mut said) = self.said.lock() {
            said(event)
        }
    }
}

/// The stage a connection holds, and the worker's one place for it: given
/// back when the connection ends, however it ends.
struct Held<'h, 'f, 's> {
    shared: &'h Shared<'f, 's>,
    stage: Option<Box<dyn Stage>>,
    ours: bool,
}

impl Held<'_, '_, '_> {
    /// Take the worker's place for a stage, unless another connection has it.
    fn claim(&mut self) -> bool {
        use std::sync::atomic::Ordering;
        self.ours = self.ours || !self.shared.busy.swap(true, Ordering::SeqCst);
        self.ours
    }

    fn release(&mut self) {
        self.stage = None;
        if std::mem::take(&mut self.ours) {
            self.shared.busy.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

impl Drop for Held<'_, '_, '_> {
    fn drop(&mut self) {
        self.release()
    }
}

/// Answer coordinators for ever, each on a thread of its own.
pub fn serve(listener: &TcpListener, key: &Key, factory: &mut Factory, said: &mut (dyn FnMut(Event) + Send)) -> io::Result<()> {
    serve_with(listener, key, factory, said, &|| None, &|_| None)
}

/// [`serve`], by a worker that keeps count of its memory: see [`Room`].
///
/// And one that runs layers some way this crate does not: see [`Bencher`].
pub fn serve_with<'a>(listener: &TcpListener, key: &Key, factory: &mut Factory, said: &'a mut (dyn FnMut(Event) + Send), room: Room<'a>, bench: Bencher<'a>) -> io::Result<()> {
    let shared = Shared { busy: false.into(), factory: factory.into(), said: said.into(), room, bench };
    std::thread::scope(|threads| {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let Ok(from) = stream.peer_addr() else { continue };
            let shared = &shared;
            threads.spawn(move || {
                let ended = match one(&mut stream, key, shared) {
                    Ok(()) => "done".to_string(),
                    Err(e) => {
                        // As the ping listener does: say why, then take what
                        // was still coming, so the reason is not overtaken
                        // by a reset.
                        let _ = wire::send_error(&mut stream, &e.to_string());
                        let _ = stream.shutdown(std::net::Shutdown::Write);
                        let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
                        let _ = io::copy(&mut io::Read::take(&stream, 1 << 20), &mut io::sink());
                        e.to_string()
                    }
                };
                shared.say(Event::Went(from, ended));
            });
        }
    });
    Ok(())
}

/// One coordinator, until it goes: [`serve`] for a single connection.
pub fn serve_one(stream: &mut TcpStream, key: &Key, factory: &mut Factory, said: &mut (dyn FnMut(Event) + Send)) -> io::Result<()> {
    one(stream, key, &Shared { busy: false.into(), factory: factory.into(), said: said.into(), room: &|| None, bench: &|_| None })
}

fn other(what: impl Into<String>) -> io::Error {
    io::Error::other(what.into())
}

/// The worker's half of the handshake in [`super::key`].
fn greet(stream: &mut TcpStream, key: &Key, room: Room, buf: &mut Vec<u8>) -> io::Result<Hello> {
    let theirs = match wire::read(stream, 1 << 16, buf)? {
        Kind::Hello => Hello::parse(buf)?,
        kind => return Err(other(format!("a {kind:?} before any greeting"))),
    };
    if theirs.nonce.is_empty() {
        return Err(other("a greeting with no nonce: this is a worker, and `kvad cluster listen` is the one that only measures"));
    }
    let mut ours = Hello { nonce: key::nonce()?, ..Hello::ours() };
    ours.memory = room().or(ours.memory);
    ours.send(stream)?;
    let proof = match wire::read(stream, 1 << 16, buf)? {
        Kind::Auth => wire::parse::<serde_json::Value>(Kind::Auth, buf)?["proof"].as_str().unwrap_or("").to_string(),
        kind => return Err(other(format!("a {kind:?} where proof of the cluster key was due"))),
    };
    if !key.accepts(Role::Coordinator, &theirs.nonce, &ours.nonce, &proof) {
        return Err(other("that is not this cluster's key"));
    }
    wire::send(stream, Kind::Auth, &serde_json::json!({ "proof": key.proof(Role::Worker, &theirs.nonce, &ours.nonce) }))?;
    Ok(theirs)
}

fn one(stream: &mut TcpStream, key: &Key, shared: &Shared) -> io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(GREETING))?;
    stream.set_write_timeout(Some(GREETING))?;
    let (mut buf, mut out) = (Vec::new(), Vec::new());
    let from = stream.peer_addr()?;
    shared.say(Event::Came(from, &greet(stream, key, shared.room, &mut buf)?));

    // From here a silence is a coordinator with nothing to say, which is
    // most of a coordinator's life; the kernel's probes say when it is gone.
    stream.set_read_timeout(None)?;
    keepalive(stream);

    // Threads for a stage's matmuls, made when there is a stage: a
    // coordinator that only measures the way here needs none.
    let mut pool: Option<rayon::ThreadPool> = None;
    let mut held = Held { shared, stage: None, ours: false };
    let mut spent = wire::Spent::default();
    // This machine, kept awake between one token's rows and the next.
    let mut awake: Option<crate::model::Lingering> = None;
    loop {
        let (kind, len) = match wire::read_header(stream) {
            Ok(header) => header,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        // Whatever has arrived, the wait for it is over.
        drop(awake.take());
        match kind {
            Kind::Forward if len <= MOST_ROWS => {
                wire::read_payload(stream, len, &mut buf)?;
                let rows = Rows::decode(&buf)?;
                let (seq, pos0, m, n_embd) = (rows.seq, rows.pos0, rows.m, rows.n_embd);
                let ran = match (held.stage.as_mut(), pool.as_ref()) {
                    // A panic in a kernel is this stage's fault and one
                    // reply's: said, and the worker goes on listening.
                    (Some(stage), Some(pool)) => {
                        let started = std::time::Instant::now();
                        let ran = pool.install(|| {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stage.run(seq, pos0 as usize, m as usize, rows.xs)))
                            .map_err(|panic| panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "the stage panicked".into()))
                            .and_then(|ran| ran.map_err(|fault| fault.to_string()))
                        });
                        let took = started.elapsed().as_secs_f64() * 1e3;
                        spent.runs += 1;
                        spent.ms += took;
                        if m == 1 {
                            spent.one_runs += 1;
                            spent.one_ms += took;
                        }
                        awake = crate::model::linger(Some(pool));
                        ran
                    }
                    _ => Err("rows for a stage that has not been loaded".to_string()),
                };
                match ran {
                    Ok(xs) => {
                        Rows { seq, pos0, m, n_embd, xs }.encode(&mut out);
                        wire::write(stream, Kind::Hidden, &out)?;
                    }
                    Err(why) => wire::send_error(stream, &why)?,
                }
            }
            Kind::Truncate if len <= 1 << 16 => {
                wire::read_payload(stream, len, &mut buf)?;
                let asked: Truncate = wire::parse(Kind::Truncate, &buf)?;
                let got = match held.stage.as_mut() {
                    Some(stage) => stage.truncate(asked.seq, asked.len).map_err(|fault| fault.to_string()),
                    None => Ok(0),
                };
                match got {
                    Ok(len) => wire::send(stream, Kind::Truncated, &Truncate { seq: asked.seq, len })?,
                    Err(why) => wire::send_error(stream, &why)?,
                }
            }
            Kind::Load if len <= 1 << 16 => {
                wire::read_payload(stream, len, &mut buf)?;
                let load: Load = wire::parse(Kind::Load, &buf)?;
                // The one held goes first: two shards at once is the
                // memory of two.
                held.stage = None;
                let named = names_a_model(&load.repo) && load.layers[0] < load.layers[1];
                let made = match named && held.claim() {
                    false if named => Err("this worker is holding layers for another coordinator".to_string()),
                    false => Err(format!("`{}`, layers {}..{}: a worker loads a model by its name, and at least one layer of it", load.repo, load.layers[0], load.layers[1])),
                    true => {
                        // A download is watched from threads that are not
                        // this one, so the frames about it are written
                        // through a second handle to the socket, one at a
                        // time, no oftener than a bar can show, and not at
                        // all while the bytes stand where they stood.
                        let out = Arc::new(Mutex::new((stream.try_clone()?, None::<io::Error>, Instant::now(), String::new(), 0u64)));
                        let tell = |out: &Mutex<(TcpStream, Option<io::Error>, Instant, String, u64)>, step: wire::Progress| {
                            let Ok(mut out) = out.lock() else { return };
                            let moving = !step.file.is_empty() && !step.done && step.bytes < step.total;
                            if moving && out.3 == step.file && (out.2.elapsed() < OFTEN || out.4 == step.bytes) {
                                return;
                            }
                            if moving {
                                (out.2, out.3, out.4) = (Instant::now(), step.file.clone(), step.bytes);
                            }
                            if let Err(e) = wire::send(&mut out.0, Kind::Progress, &step) {
                                out.1.get_or_insert(e);
                            }
                        };
                        let watch = Watcher::new({
                            let out = out.clone();
                            move |fetch| {
                                if let Some(step) = wire::Progress::of(&fetch) {
                                    tell(&out, step)
                                }
                            }
                        });
                        let mut factory = shared.factory.lock().map_err(|_| other("a load on this worker panicked"))?;
                        let made = factory(
                            &load,
                            &mut |message| {
                                shared.say(Event::Progress(message));
                                tell(&out, wire::Progress::said(message));
                            },
                            &watch,
                        );
                        if let Some(e) = out.lock().ok().and_then(|mut out| out.1.take()) {
                            return Err(e);
                        }
                        made.map_err(|fault| fault.to_string())
                    }
                };
                match made {
                    Ok(made) => {
                        let (params, bytes) = made.holds();
                        let ready = Ready { params, bytes };
                        shared.say(Event::Loaded(&load, &ready));
                        pool.get_or_insert_with(crate::model::pool);
                        held.stage = Some(made);
                        wire::send(stream, Kind::Ready, &ready)?;
                    }
                    Err(why) => {
                        // Nothing loaded is nothing held.
                        held.release();
                        wire::send_error(stream, &why)?
                    }
                }
            }
            Kind::Unload => {
                wire::skip(stream, len)?;
                held.release();
                wire::write(stream, Kind::Ack, &[])?;
            }
            Kind::Spend => {
                wire::skip(stream, len)?;
                wire::send(stream, Kind::Spent, &std::mem::take(&mut spent))?;
            }
            Kind::Bench if len <= 1 << 16 => {
                wire::read_payload(stream, len, &mut buf)?;
                let asked: wire::Bench = wire::parse(Kind::Bench, &buf)?;
                // Widths a stranger with the key chose: bounded, so that
                // being asked how fast is not a way to be asked for memory.
                match (Precision::parse(&asked.precision), asked.n_embd <= 1 << 16 && asked.intermediate <= 1 << 20) {
                    (Some(precision), true) => {
                        wire::send(stream, Kind::Benched, &super::plan::bench(asked.n_embd, asked.intermediate, precision))?
                    }
                    (None, true) => match (shared.bench)(&asked) {
                        Some(Ok(speed)) => wire::send(stream, Kind::Benched, &speed)?,
                        Some(Err(why)) => wire::send_error(stream, &why)?,
                        None => wire::send_error(stream, &format!("this worker does not run layers at `{}`", asked.precision))?,
                    },
                    _ => wire::send_error(stream, "that is not the shape of a layer")?,
                }
            }
            // A worker is a place to measure the link to, as well.
            Kind::Ping if len <= MOST_ECHOED => {
                wire::read_payload(stream, len, &mut buf)?;
                wire::write(stream, Kind::Pong, &buf)?;
            }
            Kind::Stream => {
                wire::skip(stream, len)?;
                wire::write(stream, Kind::Ack, &[])?;
            }
            kind => return Err(other(format!("a {kind:?} frame of {len} bytes, which a worker has no answer to"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_is_named_and_a_path_is_not() {
        for name in ["Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "my-model", "gpt2", "a_b.c/d-e"] {
            assert!(names_a_model(name), "{name}");
        }
        for path in ["", "/etc/passwd", "../models/x", "a/../b", "~/models/x", "./x", "a/b/c", "a//b", "C:\\models", "a b", "a/.hidden", "x/"] {
            assert!(!names_a_model(path), "{path}");
        }
    }
}
