//! A worker on loopback: a stage at the far end of a socket.
//!
//! Everything a cable would carry is carried here, to `127.0.0.1` and back:
//! the handshake and the key, the load, rows out and rows home. What is
//! held to the bit is the same as when every stage was in one process —
//! the model whole — because a socket moves floats and must not change
//! one. The rest is what goes wrong: the wrong key, another build, a
//! worker that dies in the middle of a reply, and a split that is not one.

#![cfg(all(feature = "cluster", feature = "arch-llama"))]

mod common;

use common::{llama, Fixture};
use kvad::cluster::key::{Key, Role};
use kvad::cluster::remote::RemoteStage;
use kvad::cluster::session::{self, PipelineSession};
use kvad::cluster::stage::{Fault, LocalStage, Stage};
use kvad::cluster::wire::{self, Hello, Kind, Load};
use kvad::cluster::{path, ping, remote, worker};
use kvad::model::{CpuSession, Session, Shard, Transformer};
use kvad::quant::Precision;
use kvad::weights::Checkpoint;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

fn key() -> Key {
    Key::from_hex(&"5a".repeat(32)).unwrap()
}

/// What a worker makes of a `Load` here: the shard, read from the test's
/// own checkpoint. The repo's name is the checkpoint's.
fn factory(fixture: &Fixture) -> worker::Factory {
    let (path, spec) = (fixture.path.clone(), fixture.spec.clone());
    Box::new(move |load, progress| {
        progress("reading the test's checkpoint");
        let precision = Precision::parse(&load.precision).ok_or("no such precision")?;
        let ckpt = Checkpoint::open(std::slice::from_ref(&path)).map_err(|e| e.to_string())?;
        let shard = Shard { layers: load.layers[0]..load.layers[1], ends: false };
        let model = common::read_shard(&ckpt, &spec, precision, &shard);
        Ok(Box::new(LocalStage::new(Arc::from(model))) as Box<dyn Stage>)
    })
}

/// A worker on a free port of this machine, in a thread, for as long as
/// the test runs. Returns where it is.
fn a_worker(fixture: &Fixture) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let at = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    let mut factory = factory(fixture);
    std::thread::spawn(move || worker::serve(&listener, &key(), &mut factory, &mut |_| {}));
    at
}

fn asked(precision: Precision, layers: std::ops::Range<usize>) -> Load {
    Load { repo: "tiny".into(), precision: precision.to_string(), layers: [layers.start, layers.end] }
}

fn remote(at: &str, precision: Precision, layers: std::ops::Range<usize>) -> Box<dyn Stage> {
    Box::new(RemoteStage::open(at, &key(), &asked(precision, layers), &mut |_| {}).unwrap())
}

/// The claim of the phase before, again with sockets in it: a model whose
/// layers are at the far end of a connection answers to the bit as the
/// model whole. One worker, with some layers kept here; and two workers,
/// with none kept here at all.
#[test]
fn a_model_across_workers_answers_to_the_bit() {
    for model_type in ["llama", "qwen3"] {
        for precision in [Precision::F32, Precision::Q8, Precision::Q4] {
            let fixture = llama(model_type, &format!("worker-{model_type}-{precision}"));
            let what = format!("{model_type} {precision}");
            let whole = || CpuSession::new(fixture.load(precision, &Shard::whole(&fixture.spec)), precision).reading(8);

            // Layer 0 here, 1 to 3 on a worker.
            let at = a_worker(&fixture);
            let ends: Arc<dyn Transformer> = Arc::from(fixture.load(precision, &Shard { layers: 0..1, ends: true }));
            let stages: Vec<Box<dyn Stage>> = vec![Box::new(LocalStage::new(ends.clone())), remote(&at, precision, 1..4)];
            let mut split = PipelineSession::new(ends, stages, precision).unwrap().reading(8);
            assert_eq!(split.param_count(), whole().param_count(), "{what}: a worker says what it holds");
            common::same_answers(&mut whole(), &mut split, &format!("{what}, one worker"));

            // Only the ends here; the layers on two workers.
            let (first, second) = (a_worker(&fixture), a_worker(&fixture));
            let ends: Arc<dyn Transformer> = Arc::from(fixture.load(precision, &Shard { layers: 0..0, ends: true }));
            let stages = vec![remote(&first, precision, 0..2), remote(&second, precision, 2..4)];
            let mut split = PipelineSession::new(ends, stages, precision).unwrap().reading(8);
            common::same_answers(&mut whole(), &mut split, &format!("{what}, two workers"));
        }
    }
}

fn refusal(at: &str, key: &Key, load: &Load) -> String {
    RemoteStage::open(at, key, load, &mut |_| {}).err().expect("refused").to_string()
}

/// A worker does not answer a coordinator with another key, and a
/// coordinator does not use a worker with one. Neither learns the other's.
#[test]
fn the_wrong_key_is_refused_by_whichever_end_notices() {
    let fixture = llama("llama", "worker-key");
    let at = a_worker(&fixture);
    let other = Key::from_hex(&"a5".repeat(32)).unwrap();
    let why = refusal(&at, &other, &asked(Precision::F32, 1..4));
    assert!(why.contains("refused: that is not this cluster's key"), "{why}");
    // And it is still there for one that has the key.
    assert!(RemoteStage::open(&at, &key(), &asked(Precision::F32, 1..4), &mut |_| {}).is_ok());

    // A worker that has the wrong key and says yes anyway: it cannot make
    // the proof, and the coordinator will not take its word.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let at = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        wire::read(&mut stream, 1 << 16, &mut buf).unwrap();
        let theirs = Hello::parse(&buf).unwrap();
        let ours = Hello { nonce: "00ff".into(), name: "impostor".into(), ..Hello::ours() };
        ours.send(&mut stream).unwrap();
        wire::read(&mut stream, 1 << 16, &mut buf).unwrap();
        wire::send(&mut stream, Kind::Auth, &serde_json::json!({ "proof": other.proof(Role::Worker, &theirs.nonce, &ours.nonce) })).unwrap();
        let _ = wire::read(&mut stream, 1 << 16, &mut buf);
    });
    let why = refusal(&at, &key(), &asked(Precision::F32, 1..4));
    assert!(why.contains("impostor at") && why.contains("does not hold this cluster's key"), "{why}");
}

/// Two builds can both be right and not agree to the bit, so a worker of
/// another build is turned down before it is asked for anything.
#[test]
fn a_worker_of_another_build_is_turned_down() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let at = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        wire::read(&mut stream, 1 << 16, &mut buf).unwrap();
        Hello { nonce: "00ff".into(), name: "older".into(), version: "0.1.0".into(), ..Hello::ours() }.send(&mut stream).unwrap();
        let _ = wire::read(&mut stream, 1 << 16, &mut buf);
    });
    let why = refusal(&at, &key(), &asked(Precision::F32, 1..4));
    assert!(why.contains("older at") && why.contains("is kvad 0.1.0") && why.contains("one build on every machine"), "{why}");
}

/// The listener that only measures is not a worker, and each says so to
/// somebody who took it for the other.
#[test]
fn a_ping_listener_and_a_worker_are_not_mistaken_for_each_other() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || ping::listen(&listener, &mut |_| {}));
    let why = refusal(&format!("127.0.0.1:{port}"), &key(), &asked(Precision::F32, 1..4));
    assert!(why.contains("a worker is `kvad worker`"), "{why}");

    let fixture = llama("llama", "worker-ping");
    let at = a_worker(&fixture);
    let (host, port) = wire::host_port(&at);
    let why = ping::connect(&host, port).unwrap_err().to_string();
    assert!(why.contains("this is a worker"), "{why}");

    // Something there that takes the connection and never speaks, which
    // is what a firewall holding a worker's traffic looks like from here.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let started = std::time::Instant::now();
    let why = refusal(&format!("127.0.0.1:{}", silent.local_addr().unwrap().port()), &key(), &asked(Precision::F32, 1..4));
    assert!(why.contains("took the connection and said nothing") && why.contains("firewall"), "{why}");
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "and it does not wait five minutes to say so");

    // Nobody there at all: port 1, which nothing here could have bound.
    let why = refusal("127.0.0.1:1", &key(), &asked(Precision::F32, 1..4));
    assert!(why.contains("no worker at") && why.contains("kvad worker"), "{why}");
}

/// What a worker will not load, and what it says of a load that fails.
#[test]
fn a_worker_loads_a_model_by_name_and_says_why_it_could_not() {
    let fixture = llama("llama", "worker-load");
    let at = a_worker(&fixture);
    for (repo, layers) in [("../../etc", [1, 4]), ("/tmp/model", [1, 4]), ("tiny", [2, 2])] {
        let load = Load { repo: repo.into(), precision: "f32".into(), layers };
        let why = refusal(&at, &key(), &load);
        assert!(why.contains("could not load") && why.contains("by its name"), "{repo}: {why}");
    }
    let why = refusal(&at, &key(), &Load { repo: "tiny".into(), precision: "q7".into(), layers: [1, 4] });
    assert!(why.contains("could not load tiny: no such precision"), "{why}");
    // None of which has cost the next coordinator anything.
    let mut seen = Vec::new();
    assert!(RemoteStage::open(&at, &key(), &asked(Precision::F32, 1..4), &mut |line| seen.push(line.to_string())).is_ok());
    assert!(seen.iter().any(|line| line.contains("reading the test's checkpoint")), "what the worker says while loading is passed on: {seen:?}");
}

/// A stage that is handed the wrong rows says so and stays in step, and a
/// worker that dies takes the reply with it and is named.
#[test]
fn a_worker_that_goes_away_fails_the_reply_and_is_named() {
    let fixture = llama("llama", "worker-gone");
    let e = fixture.spec.n_embd;

    // By hand, so that the test holds the worker's end of the socket.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let at = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    let mut factory = factory(&fixture);
    let (handle_out, handle_in) = std::sync::mpsc::channel::<TcpStream>();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        handle_out.send(stream.try_clone().unwrap()).unwrap();
        let _ = worker::serve_one(&mut stream, &key(), &mut factory, &mut |_| {});
    });

    let ends: Arc<dyn Transformer> = Arc::from(fixture.load(Precision::F32, &Shard { layers: 0..2, ends: true }));
    let mut far = RemoteStage::open(&at, &key(), &asked(Precision::F32, 2..4), &mut |_| {}).unwrap();

    // Rows from the wrong place are refused across the wire as they are in
    // one process, and the connection outlives the refusal.
    let why = far.run(0, 3, 1, vec![0.5; e]).unwrap_err().to_string();
    assert!(why.contains("handed position 3 of a conversation it holds 0 positions of"), "{why}");
    assert_eq!(far.truncate(0, 0).unwrap(), 0);

    let stages: Vec<Box<dyn Stage>> = vec![Box::new(LocalStage::new(ends.clone())), Box::new(far)];
    let mut split = PipelineSession::new(ends, stages, Precision::F32).unwrap().reading(8);
    let prompt: Vec<u32> = (0..12).collect();
    assert_eq!(split.forward(&prompt).unwrap().len(), common::VOCAB);

    // The worker's machine goes: its socket closes under it.
    handle_in.recv().unwrap().shutdown(std::net::Shutdown::Both).unwrap();
    let why = split.forward(&[5]).unwrap_err().to_string();
    assert!(why.contains(&at) && why.contains("which holds layers 2..4, went away"), "{why}");
    assert_eq!(split.cached(), 12, "a pass that failed did not happen");
    // And it stays failed: nothing reconnects, and nothing pretends.
    assert!(split.forward(&[5]).is_err());
    assert!(split.truncate(0).is_err());
}

/// A stage that fails once, as a worker handed a bad frame would.
struct Flaky {
    inner: LocalStage,
    fail_at: usize,
    runs: usize,
}

impl Stage for Flaky {
    fn shard(&self) -> Shard {
        self.inner.shard()
    }
    fn run(&mut self, seq: u64, pos0: usize, m: usize, xs: Vec<f32>) -> Result<Vec<f32>, Fault> {
        self.runs += 1;
        match self.runs == self.fail_at {
            true => Err("a cosmic ray".into()),
            false => self.inner.run(seq, pos0, m, xs),
        }
    }
    fn truncate(&mut self, seq: u64, len: usize) -> Result<usize, Fault> {
        self.inner.truncate(seq, len)
    }
    fn holds(&self) -> (usize, usize) {
        self.inner.holds()
    }
}

/// A pass that fails half way down the pipeline leaves the stages before
/// the failure a chunk ahead of the ones after it. The session knows, and
/// puts them back in step before it runs anything else — so a prompt
/// abandoned in the middle, or a stage that stumbled once, costs that pass
/// and not the conversation.
#[test]
fn a_pass_that_fails_part_way_is_undone_before_the_next() {
    let fixture = llama("llama", "worker-torn");
    let precision = Precision::Q8;
    let ends: Arc<dyn Transformer> = Arc::from(fixture.load(precision, &Shard { layers: 0..2, ends: true }));
    let later = LocalStage::new(Arc::from(fixture.load(precision, &Shard { layers: 2..4, ends: false })));
    // The second chunk of the prompt is the one that fails, in the second
    // stage: the first stage has by then run it.
    let stages: Vec<Box<dyn Stage>> = vec![Box::new(LocalStage::new(ends.clone())), Box::new(Flaky { inner: later, fail_at: 2, runs: 0 })];
    let mut split = PipelineSession::new(ends, stages, precision).unwrap().reading(8);
    let mut whole = CpuSession::new(fixture.load(precision, &Shard::whole(&fixture.spec)), precision).reading(8);

    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 5 + 2) % common::VOCAB as u32).collect();
    let why = split.forward(&prompt).unwrap_err().to_string();
    assert!(why.contains("a cosmic ray"), "{why}");
    assert_eq!(split.cached(), 8, "the chunk that went through every stage is kept");

    // Carry on from where it got to, and it is the model again.
    let bits = |xs: &[f32]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&split.forward(&prompt[8..]).unwrap()), bits(&whole.forward(&prompt).unwrap()));
    // Or drop it all and start another: a prompt cancelled in the middle.
    assert_eq!(split.truncate(0).unwrap(), 0);
    assert_eq!(whole.truncate(0).unwrap(), 0);
    common::same_answers(&mut whole, &mut split, "after a failed pass");
}

/// `--split` is checked before anything is read or anybody is asked.
#[test]
fn a_split_that_is_not_one_is_refused_before_anything_is_loaded() {
    let fixture = llama("llama", "worker-split");
    let files = kvad::weights::ModelFiles {
        weights: vec![fixture.path.clone()],
        tokenizer: "none".into(),
        config: "none".into(),
        tokenizer_config: None,
        generation_config: None,
    };
    let want = |workers: &[String], split: Option<&[std::ops::Range<usize>]>, params: Option<u64>| {
        let want = session::Across { repo: "tiny", files: &files, spec: &fixture.spec, precision: Precision::F32, workers, split, key: &key(), params, context: 64 };
        session::across(&want, &mut |_| {}).map_err(|e| e.to_string())
    };
    let why = |workers: &[&str], split: &str| {
        let workers: Vec<String> = workers.iter().map(|w| w.to_string()).collect();
        want(&workers, Some(&session::parse_split(split).unwrap()), None).err().expect("refused")
    };
    // Nobody is listening at these, and nobody is asked.
    assert!(why(&["a:1"], "0..4").contains("1 run(s) of layers and there are 2 machine(s)"));
    assert!(why(&["a:1"], "0..1,1..2,2..4").contains("3 run(s) of layers and there are 2 machine(s)"));
    assert!(why(&["a:1"], "0..1,2..4").contains("layer 1 is nowhere"));
    assert!(why(&["a:1"], "0..3,2..4").contains("layer 2 is twice"));
    assert!(why(&["a:1"], "0..2,2..3").contains("3 layers of 4"));
    assert!(why(&["a:1", "b:1"], "0..4,4..4,4..4").contains("a:1 is given no layers"));

    assert_eq!(session::parse_split("0..12, 12..24").unwrap(), [0..12, 12..24]);
    assert_eq!(session::parse_split("0..0,0..30").unwrap(), [0..0, 0..30]);
    for bad in ["", "0-12", "12..0", "a..b", "0..12,", "0..12;12..24"] {
        assert!(session::parse_split(bad).is_err(), "{bad}");
    }

    // With a worker really there, the same call makes a session that runs.
    let at = a_worker(&fixture);
    let split = session::parse_split("0..1,1..4").unwrap();
    let mut session = want(&[at.clone()], Some(&split), None).unwrap();
    assert_eq!(session.forward(&[1, 2, 3]).unwrap().len(), common::VOCAB);
    assert_eq!(session.label(), "cpu f32 in 2 stages");
    drop(session);

    // A split by hand is held to each machine's memory, when the model's
    // size is known: a model of a million gigabytes is nobody's to hold.
    let huge = want(&[at.clone()], Some(&split), Some(250_000_000_000_000)).err().expect("refused");
    assert!(huge.contains("is given layers") && huge.contains("for a model"), "{huge}");

    // With no split the layers are placed, and a model that fits on the
    // machine with the tokens stays there: a worker is a hop, and here a
    // hop buys nothing. The session that results asks the worker for
    // nothing and runs in one stage.
    let mut lines = Vec::new();
    let params = fixture.load(Precision::F32, &Shard::whole(&fixture.spec)).param_count() as u64;
    let placed = session::Across { repo: "tiny", files: &files, spec: &fixture.spec, precision: Precision::F32, workers: &[at], split: None, key: &key(), params: Some(params), context: 64 };
    let mut session = session::across(&placed, &mut |line| lines.push(line.to_string())).unwrap();
    assert_eq!(session.label(), "cpu f32 in 1 stages");
    assert!(lines.iter().any(|l| l.starts_with("MACHINE")) && lines.iter().any(|l| l.contains("the workers hold nothing")), "{lines:?}");
    assert_eq!(session.forward(&[1, 2, 3]).unwrap().len(), common::VOCAB);
    // And with nothing known of its size, nothing is placed.
    let unknown = session::Across { params: None, ..placed };
    assert!(session::across(&unknown, &mut |_| {}).err().expect("refused").to_string().contains("could not be found out"));
}

/// A worker holds one model, and says so to the second coordinator that
/// asks — at once, and without making the first wait. What it does not
/// refuse is a greeting or a measurement, from anybody with the key.
#[test]
fn a_worker_holds_one_model_and_greets_everybody() {
    let fixture = llama("llama", "worker-busy");
    let at = a_worker(&fixture);
    let mut first = RemoteStage::open(&at, &key(), &asked(Precision::F32, 1..4), &mut |_| {}).unwrap();

    let why = refusal(&at, &key(), &asked(Precision::F32, 2..4));
    assert!(why.contains("holding layers for another coordinator"), "{why}");
    // The way there can still be measured, while the first one works.
    let way = path::measure(&at, &key()).unwrap();
    assert!(way.hop_ms > 0.0 && way.chunk_ms > 0.0);
    assert!(way.line().contains("loopback") && way.line().contains("ms for 32 KB"), "{}", way.line());
    assert_eq!(first.run(0, 0, 2, vec![0.25; 2 * fixture.spec.n_embd]).unwrap().len(), 2 * fixture.spec.n_embd);

    // When the first has gone, the place is free. Its going is noticed on
    // the worker's own thread, so the next to ask may be a moment early.
    drop(first);
    let again = (0..50).find_map(|_| {
        std::thread::sleep(std::time::Duration::from_millis(20));
        RemoteStage::open(&at, &key(), &asked(Precision::F32, 2..4), &mut |_| {}).ok()
    });
    assert!(again.is_some(), "the worker never let go of the first coordinator's place");
}

/// A worker found on several addresses is tried on all of them, and the
/// ones that lead nowhere cost a line each and a few seconds between them.
#[test]
fn the_quickest_way_to_a_worker_is_one_that_answers() {
    let fixture = llama("llama", "worker-paths");
    let at = a_worker(&fixture);
    let (_, port) = wire::host_port(&at);
    // The worker, by a second spelling of the same place; a port where
    // nothing listens; and something that takes a connection and says
    // nothing.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let addresses = vec![
        "127.0.0.1:1".to_string(),
        format!("127.0.0.1:{}", silent.local_addr().unwrap().port()),
        at.clone(),
        format!("localhost:{port}"),
    ];
    let mut lines = Vec::new();
    let started = std::time::Instant::now();
    let way = path::quickest(&addresses, "", &key(), &mut |line| lines.push(line.to_string())).unwrap();
    assert!(started.elapsed() < std::time::Duration::from_secs(8), "the dead ends are waited for together");
    assert!(way.address == at || way.address == format!("localhost:{port}"), "{}", way.address);
    assert_eq!(lines.len(), 4, "a line for each, the dead ends too: {lines:?}");
    assert!(lines[0].contains("no worker at 127.0.0.1:1"), "{}", lines[0]);
    assert!(lines[1].contains("said nothing for 3 seconds"), "{}", lines[1]);
    assert!(lines[2].contains("ms for 32 KB") && lines[3].contains("ms for 32 KB"), "{lines:?}");

    // An address that somebody else answers on is not a way to the worker
    // that was asked for.
    let mut lines = Vec::new();
    assert!(path::quickest(&[at.clone()], "ffffffffffffffff", &key(), &mut |line| lines.push(line.to_string())).is_err());
    assert!(lines[0].contains("and not by the worker asked for"), "{}", lines[0]);
    // And nowhere at all is the error of the last place tried.
    let why = path::quickest(&["127.0.0.1:1".to_string()], "", &key(), &mut |_| {}).err().unwrap().to_string();
    assert!(why.contains("no worker at"), "{why}");

    // The greeting says which build and which machine, whoever asks.
    let (_, hello) = remote::connect(&at, &key()).unwrap();
    assert_eq!(hello.version, kvad::cluster::build());
}
