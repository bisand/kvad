//! Speak against a scripted realtime server, in this process, that speaks
//! `/v1/realtime` as kvad-serve does but hears a word in every second of
//! audio: `w1` once it has 16 000 samples, `w2` at 32 000, and so on. It
//! holds its one session across sockets, as kvad-serve holds one for a
//! minute, and can cut a socket mid-word, refuse connections for a while,
//! or forget the session. `POST /v1/audio/transcriptions` answers with how
//! many samples the WAV it was sent holds.
//!
//! The audio is a known ramp, sample `i` being `i % 1000`, with silence
//! where a test wants a pause, and the server checks every sample it takes
//! against it: a sample sent twice, lost or out of place fails the test. A
//! session begun again after one was let go of goes on from where the
//! phone's settled words ended, and so do its words' numbers.

use kvad_phone_core::api::Server;
use kvad_phone_core::session::Emit;
use kvad_phone_core::speak::{Speak, RATE};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};
use tungstenite::Message;

#[derive(Default, Clone)]
struct Script {
    /// Cut the first socket, with no close, once the session has this many
    /// samples.
    cut_at: Option<u64>,
    /// After the cut, close every connection at once for this long.
    down_for: Duration,
    /// After the cut, the session is let go of: a resume is not found.
    forget: bool,
    /// No `/v1/realtime`: the upgrade is answered 404, as a server from
    /// before it answers.
    no_live: bool,
    /// Samples of the take that are silence.
    quiet: Option<(u64, u64)>,
}

/// The session, as the server holds it between sockets.
#[derive(Default)]
struct Held {
    id: String,
    /// The take's sample that is this session's first.
    base: u64,
    /// The words of the take before this session's.
    first: u64,
    /// This session's words whose deltas went out.
    sent_words: u64,
    /// Samples taken.
    offset: u64,
    /// Where the next frame begins.
    next: u64,
    words: u64,
    said: String,
    item: u64,
    events: u64,
    /// Settled events sent, with the offset each was heard to.
    log: Vec<(u64, String)>,
}

impl Held {
    fn event(&mut self, kind: &str, mut body: Value) -> String {
        self.events += 1;
        body["type"] = kind.into();
        body["event_id"] = format!("event_{}", self.events).into();
        body.to_string()
    }
}

struct Fake {
    script: Script,
    held: Mutex<Option<Held>>,
    connections: AtomicUsize,
    /// While refusing connections, until this many ms after the epoch.
    down_until: AtomicU64,
    /// The samples each file sent to the file endpoint held.
    files: Mutex<Vec<u64>>,
    /// What went wrong in the server, which the test reports.
    wrong: Mutex<Vec<String>>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

fn ramp(i: u64) -> i16 {
    (i % 1000) as i16
}

/// The take's sample `i`.
fn signal(i: u64, quiet: Option<(u64, u64)>) -> i16 {
    match quiet {
        Some((a, b)) if (a..b).contains(&i) => 0,
        _ => ramp(i),
    }
}

impl Fake {
    fn start(script: Script) -> (Arc<Fake>, String) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let fake = Arc::new(Fake {
            script,
            held: Mutex::new(None),
            connections: AtomicUsize::new(0),
            down_until: AtomicU64::new(0),
            files: Mutex::new(Vec::new()),
            wrong: Mutex::new(Vec::new()),
        });
        let f = fake.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let f = f.clone();
                std::thread::spawn(move || f.serve(stream));
            }
        });
        (fake, addr)
    }

    fn wrong(&self, what: String) {
        self.wrong.lock().unwrap().push(what);
    }

    fn serve(&self, stream: TcpStream) {
        if now_ms() < self.down_until.load(Ordering::Relaxed) {
            return;
        }
        let mut first = [0u8; 4];
        let mut got = 0;
        while got < 4 {
            match stream.peek(&mut first) {
                Ok(n) if n > 0 => got = n,
                _ => return,
            }
        }
        if &first == b"POST" {
            self.file(stream);
        } else if self.script.no_live {
            let mut r = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while r.read_line(&mut line).is_ok_and(|n| n > 0) && line != "\r\n" {
                line.clear();
            }
            let _ = (&stream).write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
        } else {
            self.socket(stream);
        }
    }

    /// `POST /v1/audio/transcriptions`, streamed: one delta, then the text,
    /// which says how many samples the WAV held.
    fn file(&self, stream: TcpStream) {
        let mut r = BufReader::new(stream.try_clone().unwrap());
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = v.trim().parse().unwrap();
            }
        }
        let mut body = vec![0u8; length];
        r.read_exact(&mut body).unwrap();
        let at = body.windows(4).position(|w| w == b"RIFF").expect("a WAV in the form");
        let wav = &body[at..];
        let data = u32::from_le_bytes(wav[40..44].try_into().unwrap()) as u64;
        let end = body.windows(2).rposition(|w| w == b"\r\n").unwrap();
        let end = body[..end].windows(2).rposition(|w| w == b"\r\n").unwrap_or(end);
        let samples = ((end - at) as u64).saturating_sub(44) / 2;
        if data != samples * 2 {
            self.wrong(format!("the WAV's header says {data} bytes, and it holds {}", samples * 2));
        }
        self.files.lock().unwrap().push(samples);
        let text = format!("file {samples}");
        let secs = |n: u64| n as f64 / RATE as f64;
        // Whisper's segments as kvad-serve sends them. Without timestamps a
        // segment is a whole 30-second window, and the windows meet. With
        // them it is a sentence, but its times are a guess: across the
        // silence each side is stretched in, to well under a pause apart.
        let asked = b"name=\"timestamps\"\r\n\r\ntrue\r\n";
        let timed = body.windows(asked.len()).any(|w| w == asked);
        let segments = match (self.script.quiet, timed) {
            (Some((a, b)), true) => json!([{ "start": 0.0, "end": secs(a) + 1.2, "text": " file a" }, { "start": secs(b) - 1.2, "end": secs(samples), "text": " file b" }]),
            (Some(_), false) => json!([{ "start": 0.0, "end": secs(samples), "text": " file a file b" }]),
            _ => json!([{ "start": 0.0, "end": secs(samples), "text": format!(" {text}") }]),
        };
        let mut w = stream;
        let delta = json!({ "type": "transcript.text.delta", "delta": text, "kvad": { "segments": segments } });
        let done = json!({ "type": "transcript.text.done", "text": text });
        let _ = write!(
            w,
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\nevent: transcript.text.delta\ndata: {delta}\n\nevent: transcript.text.done\ndata: {done}\n\n"
        );
    }

    // tungstenite's handshake callback answers with its whole response.
    #[allow(clippy::result_large_err)]
    fn socket(&self, stream: TcpStream) {
        let mut query = String::new();
        let mut ws = match tungstenite::accept_hdr(stream, |req: &tungstenite::handshake::server::Request, res| {
            query = req.uri().query().unwrap_or_default().to_string();
            if req.headers().get("authorization").is_some() {
                self.wrong("a key was sent to a server that wants none".into());
            }
            Ok(res)
        }) {
            Ok(ws) => ws,
            Err(_) => return,
        };
        let n = self.connections.fetch_add(1, Ordering::Relaxed);
        let arg = |k: &str| query.split('&').find_map(|p| p.strip_prefix(&format!("{k}="))).map(str::to_string);
        let mut hello = Vec::new();
        {
            let mut held = self.held.lock().unwrap();
            match arg("resume") {
                None => {
                    // A take's session let go of before its end is followed
                    // by one from where the phone's words were settled to.
                    let (base, first) = match held.as_ref().filter(|h| h.item == 1) {
                        Some(h) => (h.base + h.sent_words * RATE, h.first + h.sent_words),
                        None => (0, 0),
                    };
                    let mut h = Held { id: format!("sess_{}", n), item: 1, base, first, ..Held::default() };
                    hello.push(h.event("session.created", json!({ "session": { "id": h.id } })));
                    *held = Some(h);
                }
                Some(id) => match held.as_mut().filter(|h| h.id == id && !self.script.forget) {
                    None => {
                        let m = json!({ "type": "error", "event_id": "event_1", "error": { "code": "session_not_found", "message": "not held" } });
                        let _ = ws.send(Message::Text(m.to_string().into()));
                        let _ = ws.close(None);
                        return;
                    }
                    Some(h) => {
                        let from: u64 = arg("offset").and_then(|o| o.parse().ok()).unwrap_or(0);
                        h.next = h.offset;
                        let body = json!({ "session": { "id": h.id }, "kvad": { "resumed": true, "offset": h.offset, "item_id": format!("item_{}", h.item), "transcript": h.said } });
                        hello.push(h.event("session.created", body));
                        hello.extend(h.log.iter().filter(|(at, _)| *at >= from).map(|(_, m)| m.clone()));
                    }
                },
            }
        }
        for m in hello {
            if ws.send(Message::Text(m.into())).is_err() {
                return;
            }
        }
        loop {
            let Ok(msg) = ws.read() else { return };
            let mut out: Vec<String> = Vec::new();
            {
                let mut guard = self.held.lock().unwrap();
                let h = guard.as_mut().unwrap();
                match msg {
                    Message::Binary(pcm) => {
                        let at = h.next;
                        let count = pcm.len() as u64 / 2;
                        h.next = at + count;
                        if at > h.offset {
                            out.push(h.event("error", json!({ "error": { "code": "audio_gap", "message": format!("send from {}", h.offset) } })));
                        } else {
                            for (k, c) in pcm.as_chunks::<2>().0.iter().enumerate() {
                                let i = at + k as u64;
                                if i < h.offset {
                                    continue;
                                }
                                let s = i16::from_le_bytes(*c);
                                let want = signal(h.base + i, self.script.quiet);
                                if s != want {
                                    self.wrong(format!("sample {} of the take was {s}, not {want}", h.base + i));
                                }
                                h.offset = i + 1;
                            }
                            while h.offset >= (h.words + 1) * RATE {
                                h.words += 1;
                                let delta = format!(" w{}", h.first + h.words);
                                h.said.push_str(&delta);
                                let at = h.words * RATE;
                                let item = format!("item_{}", h.item);
                                let m = h.event("conversation.item.input_audio_transcription.delta", json!({ "item_id": item, "delta": delta, "kvad": { "lag": 0.1, "offset": at } }));
                                h.log.push((at, m.clone()));
                                out.push(m);
                                out.push(h.event("kvad.transcription.partial", json!({ "item_id": item, "text": " w…", "kvad": { "lag": 0.1, "offset": at } })));
                            }
                        }
                    }
                    Message::Text(text) => {
                        let v: Value = serde_json::from_str(&text).unwrap();
                        match v["type"].as_str().unwrap() {
                            "session.update" => {
                                if v["session"]["audio"]["input"]["format"]["rate"] != RATE {
                                    self.wrong(format!("the session was set up as {v}"));
                                }
                                out.push(h.event("session.updated", json!({})));
                            }
                            "kvad.input_audio_buffer.offset" => h.next = v["offset"].as_u64().unwrap(),
                            "input_audio_buffer.commit" => {
                                let item = format!("item_{}", h.item);
                                let at = h.offset;
                                let c = h.event("input_audio_buffer.committed", json!({ "item_id": item, "kvad": { "offset": at } }));
                                let said = std::mem::take(&mut h.said);
                                let d = h.event("conversation.item.input_audio_transcription.completed", json!({ "item_id": item, "transcript": said.trim(), "kvad": { "offset": at } }));
                                h.log.push((at, c.clone()));
                                h.log.push((at, d.clone()));
                                h.item += 1;
                                out.push(c);
                                out.push(d);
                            }
                            other => self.wrong(format!("the client sent {other}")),
                        }
                    }
                    Message::Close(_) => {
                        drop(guard);
                        while ws.read().is_ok() {}
                        return;
                    }
                    _ => {}
                }
                if n == 0 && self.script.cut_at.is_some_and(|c| h.offset >= c) {
                    // The link goes, mid-word, with nothing said.
                    self.down_until.store(now_ms() + self.script.down_for.as_millis() as u64, Ordering::Relaxed);
                    let _ = ws.get_ref().shutdown(Shutdown::Both);
                    return;
                }
            }
            for m in out {
                let delta = m.contains("input_audio_transcription.delta");
                if ws.send(Message::Text(m.into())).is_err() {
                    return;
                }
                if delta {
                    self.held.lock().unwrap().as_mut().unwrap().sent_words += 1;
                }
            }
        }
    }
}

/// A scratch folder for a test's takes.
fn folder(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kvad-speak-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn speak() -> (Speak, mpsc::Receiver<Value>) {
    let (tx, rx) = mpsc::channel();
    let tx = Mutex::new(tx);
    let emit: Emit = Arc::new(move |e| {
        let _ = tx.lock().unwrap().send(e);
    });
    (Speak::new(emit), rx)
}

fn name(e: &Value) -> &str {
    e.as_object().and_then(|o| o.keys().next()).map(String::as_str).unwrap_or_default()
}

/// `seconds` of the ramp, a tenth of a second at a time as the app hands
/// it over, from sample `from`.
fn play(s: &Speak, from: u64, seconds: f64) -> u64 {
    play_with(s, from, seconds, None)
}

/// The same, silent from one sample to another.
fn play_with(s: &Speak, from: u64, seconds: f64, quiet: Option<(u64, u64)>) -> u64 {
    let end = from + (seconds * RATE as f64) as u64;
    let mut at = from;
    while at < end {
        let n = (end - at).min(RATE / 10);
        let chunk: Vec<i16> = (at..at + n).map(|i| signal(i, quiet)).collect();
        s.audio(&chunk);
        at += n;
        std::thread::sleep(Duration::from_millis(2));
    }
    end
}

/// Every event until `last`, which is returned with them.
fn until(rx: &mpsc::Receiver<Value>, last: &str, secs: u64) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut all = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let e = rx.recv_timeout(left).unwrap_or_else(|_| panic!("no {last} in {secs} s; had {all:?}"));
        let done = name(&e) == last || name(&e) == "speak_failed";
        all.push(e);
        if done {
            return all;
        }
    }
}

fn settled(events: &[Value]) -> String {
    events.iter().filter(|e| name(e) == "speak_settled").map(|e| e["speak_settled"]["text"].as_str().unwrap()).collect()
}

fn links(events: &[Value]) -> Vec<String> {
    events.iter().filter(|e| name(e) == "speak_link").map(|e| e["speak_link"]["state"].as_str().unwrap().to_string()).collect()
}

fn words(n: u64) -> String {
    (1..=n).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ")
}

fn no_wrong(fake: &Fake) {
    let wrong = fake.wrong.lock().unwrap();
    assert!(wrong.is_empty(), "the server saw: {wrong:?}");
}

/// The take on the phone: a WAV with the samples played and a header that
/// says so, and its JSON beside it.
fn kept(dir: &std::path::Path, take: &Value, samples: u64) {
    let id = take["id"].as_str().unwrap();
    let wav = std::fs::read(dir.join(format!("{id}.wav"))).unwrap();
    assert_eq!(wav.len() as u64, 44 + 2 * samples);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()) as u64, 2 * samples);
    let meta: Value = serde_json::from_slice(&std::fs::read(dir.join(format!("{id}.json"))).unwrap()).unwrap();
    assert_eq!(&meta, take);
    assert_eq!(meta["seconds"].as_f64().unwrap(), samples as f64 / RATE as f64);
}

#[test]
fn a_take_gives_its_words_each_once() {
    let (fake, addr) = Fake::start(Script::default());
    let dir = folder("whole");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "NbAiLab/nb-whisper-small", "no", dir.to_str().unwrap(), false);
    assert_eq!(name(&rx.recv_timeout(Duration::from_secs(1)).unwrap()), "speak_started");
    let samples = play(&s, 0, 10.0);
    std::thread::sleep(Duration::from_millis(300));
    s.stop();
    let events = until(&rx, "speak_done", 20);
    let take = &events.last().unwrap()["speak_done"]["take"];
    assert_eq!(take["text"], words(10), "{events:?}");
    assert_eq!(take["how"], "live");
    assert_eq!(settled(&events).trim(), words(10));
    assert_eq!(links(&events), ["live"]);
    kept(&dir, take, samples);
    no_wrong(&fake);
    assert_eq!(fake.connections.load(Ordering::Relaxed), 1);

    // Listed, and then deleted.
    s.takes(dir.to_str().unwrap());
    let e = rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(e["takes"]["takes"], json!([take]));
    s.delete(dir.to_str().unwrap(), take["id"].as_str().unwrap());
    assert_eq!(name(&rx.recv_timeout(Duration::from_secs(1)).unwrap()), "take_deleted");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
}

#[test]
fn a_socket_cut_mid_word_and_resumed_gives_the_take_of_one_never_cut() {
    let (fake, addr) = Fake::start(Script { cut_at: Some(RATE * 9 / 2), down_for: Duration::from_millis(300), ..Script::default() });
    let dir = folder("cut");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let samples = play(&s, 0, 10.0);
    std::thread::sleep(Duration::from_millis(1500));
    s.stop();
    let events = until(&rx, "speak_done", 30);
    let take = &events.last().unwrap()["speak_done"]["take"];
    assert_eq!(take["text"], words(10), "{events:?}");
    assert_eq!(settled(&events).trim(), words(10), "a word was settled twice, or lost");
    assert!(links(&events).contains(&"reconnecting".to_string()), "{events:?}");
    kept(&dir, take, samples);
    no_wrong(&fake);
    assert!(fake.connections.load(Ordering::Relaxed) >= 2);
    assert_eq!(fake.held.lock().unwrap().as_ref().unwrap().offset, samples, "the server did not hear every sample once");
}

/// As a server restarted mid-take does: the resume is not found, and a new
/// session goes on from the last settled word, live, each word once.
#[test]
fn a_session_let_go_of_goes_on_live_in_a_new_one() {
    let (fake, addr) = Fake::start(Script { cut_at: Some(RATE * 7 / 2), forget: true, ..Script::default() });
    let dir = folder("forgot");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let samples = play(&s, 0, 10.0);
    std::thread::sleep(Duration::from_millis(1500));
    s.stop();
    let events = until(&rx, "speak_done", 30);
    let take = &events.last().unwrap()["speak_done"]["take"];
    assert_eq!(take["how"], "live", "{events:?}");
    assert_eq!(take["text"], words(10));
    assert_eq!(settled(&events).trim(), words(10), "a word was settled twice, or lost");
    let l = links(&events);
    let back = l.iter().position(|s| s == "reconnecting").and_then(|r| l[r..].iter().position(|s| s == "live"));
    assert!(back.is_some(), "never live again: {l:?}");
    assert!(fake.files.lock().unwrap().is_empty(), "the take was sent whole as well");
    kept(&dir, take, samples);
    no_wrong(&fake);
    let h = fake.held.lock().unwrap();
    let h = h.as_ref().unwrap();
    assert!(h.base > 0 && h.base + h.offset == samples, "the new session heard {} from {}", h.offset, h.base);
}

/// A server from before `/v1/realtime`: the take is sent whole when it
/// ends, and its paragraphs come from the gaps between segments.
#[test]
fn a_server_without_live_gets_the_take_whole_in_paragraphs() {
    let quiet = Some((RATE * 3, RATE * 6));
    let (fake, addr) = Fake::start(Script { no_live: true, quiet, ..Script::default() });
    let dir = folder("nolive");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let samples = play_with(&s, 0, 8.0, quiet);
    s.stop();
    let events = until(&rx, "speak_done", 30);
    let take = &events.last().unwrap()["speak_done"]["take"];
    assert_eq!(take["how"], "file", "{events:?}");
    assert_eq!(take["text"], "file a\n\nfile b");
    assert_eq!(*fake.files.lock().unwrap(), [samples]);
    assert_eq!(links(&events), ["offline", "sending"]);
    kept(&dir, take, samples);
    no_wrong(&fake);
}

/// A take transcribed again begins its paragraphs where the recording
/// paused, though Whisper's segments nearly meet across the pause: from the
/// pauses kept with the take, and for a take kept before they were, from
/// its file.
#[test]
fn a_take_transcribed_again_keeps_its_paragraphs() {
    let quiet = Some((RATE * 3, RATE * 6));
    let (fake, addr) = Fake::start(Script { quiet, ..Script::default() });
    let dir = folder("again");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let samples = play_with(&s, 0, 8.0, quiet);
    std::thread::sleep(Duration::from_millis(300));
    s.stop();
    let events = until(&rx, "speak_done", 30);
    let take = events.last().unwrap()["speak_done"]["take"].clone();
    let id = take["id"].as_str().unwrap().to_string();
    let pauses = take["pauses"].as_array().expect("the take keeps its pauses");
    assert_eq!(pauses.len(), 1, "{pauses:?}");
    let (from, to) = (pauses[0][0].as_f64().unwrap(), pauses[0][1].as_f64().unwrap());
    assert!((from - 3.0).abs() <= 0.1 && (to - 6.0).abs() <= 0.1, "the pause kept was {from}–{to} s");

    let again = |rx: &mpsc::Receiver<Value>| {
        s.retranscribe(Some(Server::new(&addr, None).unwrap()), dir.to_str().unwrap(), &id, "", "");
        let events = until(rx, "retranscribe_done", 30);
        events.last().unwrap()["retranscribe_done"]["take"]["text"].as_str().unwrap().to_string()
    };
    assert_eq!(again(&rx), "file a\n\nfile b");

    // The same take as one kept before pauses were.
    let meta = dir.join(format!("{id}.json"));
    let mut v: Value = serde_json::from_slice(&std::fs::read(&meta).unwrap()).unwrap();
    v.as_object_mut().unwrap().remove("pauses");
    std::fs::write(&meta, v.to_string()).unwrap();
    assert_eq!(again(&rx), "file a\n\nfile b");
    assert_eq!(*fake.files.lock().unwrap(), [samples, samples]);
    no_wrong(&fake);
}

/// Three seconds of silence: the word heard after it begins a paragraph.
#[test]
fn a_pause_begins_a_paragraph() {
    let quiet = Some((RATE * 4, RATE * 7));
    let (fake, addr) = Fake::start(Script { quiet, ..Script::default() });
    let dir = folder("pause");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let samples = play_with(&s, 0, 11.0, quiet);
    std::thread::sleep(Duration::from_millis(300));
    s.stop();
    let events = until(&rx, "speak_done", 30);
    let take = &events.last().unwrap()["speak_done"]["take"];
    let want = "w1 w2 w3 w4 w5 w6\n\nw7 w8 w9 w10 w11";
    assert_eq!(take["text"], want, "{events:?}");
    assert_eq!(settled(&events).trim(), want, "the app is told the paragraph as it comes");
    kept(&dir, take, samples);
    no_wrong(&fake);
}

#[test]
fn a_stop_during_a_reconnect_loses_nothing() {
    let (fake, addr) = Fake::start(Script { cut_at: Some(RATE * 6), down_for: Duration::from_millis(1500), ..Script::default() });
    let dir = folder("stop");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let samples = play(&s, 0, 8.0);
    // Stopped while the server is down.
    std::thread::sleep(Duration::from_millis(300));
    s.stop();
    let events = until(&rx, "speak_done", 30);
    let take = &events.last().unwrap()["speak_done"]["take"];
    assert_eq!(take["text"], words(8), "{events:?}");
    assert_eq!(settled(&events).trim(), words(8));
    kept(&dir, take, samples);
    no_wrong(&fake);
    assert_eq!(fake.held.lock().unwrap().as_ref().unwrap().offset, samples);
}

#[test]
fn a_cancelled_take_leaves_nothing() {
    let (_fake, addr) = Fake::start(Script::default());
    let dir = folder("cancel");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let id = rx.recv_timeout(Duration::from_secs(1)).unwrap()["speak_started"]["id"].as_str().unwrap().to_string();
    play(&s, 0, 1.0);
    s.cancel();
    let events = until(&rx, "speak_cancelled", 5);
    assert_eq!(events.last().unwrap()["speak_cancelled"]["id"], id.as_str());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    // Nothing more comes of it, and a new take can begin.
    std::thread::sleep(Duration::from_millis(500));
    assert!(rx.try_iter().all(|e| !matches!(name(&e), "speak_done" | "speak_failed")));
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    assert_eq!(name(&rx.recv_timeout(Duration::from_secs(1)).unwrap()), "speak_started");
    s.cancel();
}

#[test]
fn a_take_cut_short_by_the_app_is_listed_untranscribed() {
    let dir = folder("orphan");
    std::fs::create_dir_all(&dir).unwrap();
    // A WAV with a header that still says empty, as the app leaves one it
    // was stopped in the middle of.
    let mut wav = vec![0u8; 44];
    wav[..4].copy_from_slice(b"RIFF");
    wav.extend((0..RATE).flat_map(|i| ramp(i).to_le_bytes()));
    std::fs::write(dir.join("1760870000000.wav"), &wav).unwrap();
    let (s, rx) = speak();
    s.takes(dir.to_str().unwrap());
    let e = rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let take = &e["takes"]["takes"][0];
    assert_eq!(take["id"], "1760870000000");
    assert_eq!(take["started"], 1760870000);
    assert_eq!(take["how"], "");
    kept(&dir, take, RATE);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A take of a room keeps its far voice brought up in its file, which is
/// what is sent.
#[test]
fn a_take_of_a_room_is_kept_levelled() {
    let (_fake, addr) = Fake::start(Script::default());
    let dir = folder("level");
    let (s, rx) = speak();
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), true);
    assert_eq!(name(&rx.recv_timeout(Duration::from_secs(1)).unwrap()), "speak_started");
    // Words at -40 dBFS, 250 ms with 100 ms between, for 4 s.
    let tau = std::f64::consts::TAU;
    let far: Vec<i16> = (0..4 * RATE).map(|i| if i % 5600 < 4000 { (330.0 * (i as f64 * 220.0 * tau / RATE as f64).sin()) as i16 } else { 0 }).collect();
    for chunk in far.chunks(RATE as usize / 10) {
        s.audio(chunk);
    }
    s.stop();
    let events = until(&rx, "speak_done", 20);
    let take = &events.last().unwrap()["speak_done"]["take"];
    let wav = std::fs::read(dir.join(format!("{}.wav", take["id"].as_str().unwrap()))).unwrap();
    let kept: Vec<i16> = wav[44..].chunks(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    let peak = |s: &[i16]| s.iter().map(|x| x.unsigned_abs()).max().unwrap();
    let last = 3 * RATE as usize;
    assert!(peak(&kept[last..]) > 6 * peak(&far[last..]), "{} against {}", peak(&kept[last..]), peak(&far[last..]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_take_wants_a_server_and_one_at_a_time() {
    let (s, rx) = speak();
    let dir = folder("one");
    s.start(None, "", "", dir.to_str().unwrap(), false);
    assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), json!({ "speak_failed": { "error": "Not connected to a server.", "kept": false } }));
    let (_fake, addr) = Fake::start(Script::default());
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    assert_eq!(name(&rx.recv_timeout(Duration::from_secs(1)).unwrap()), "speak_started");
    s.start(Some(Server::new(&addr, None).unwrap()), "", "", dir.to_str().unwrap(), false);
    let e = rx.try_iter().find(|e| name(e) == "speak_failed").expect("a second take refused");
    assert_eq!(e["speak_failed"]["kept"], false);
    s.cancel();
}

/// The samples of a 16 kHz mono 16-bit WAV, wherever its `data` chunk is.
fn samples_of(wav: &[u8]) -> Vec<i16> {
    let mut at = 12;
    while at + 8 <= wav.len() {
        let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().unwrap()) as usize;
        if &wav[at..at + 4] == b"data" {
            return wav[at + 8..(at + 8 + size).min(wav.len())].as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect();
        }
        at += 8 + size + size % 2;
    }
    panic!("no data chunk in the WAV");
}

/// Against a real kvad-serve, which this machine may not have: a recording
/// played in at real time, settled live, then the same take sent whole to
/// the file endpoint, and the two transcripts mostly the same words. With
/// `KVAD_PHONE_TEST_AUDIO=speech.wav` (16 kHz mono 16-bit),
/// `KVAD_PHONE_TEST_SERVER` (127.0.0.1:5823) and `KVAD_PHONE_TEST_MODEL`
/// (NbAiLab/nb-whisper-small), and `-- --ignored`.
#[test]
#[ignore]
fn a_take_with_a_running_server() {
    let env = |k: &str, or: &str| std::env::var(k).unwrap_or_else(|_| or.to_string());
    let audio = samples_of(&std::fs::read(env("KVAD_PHONE_TEST_AUDIO", "speech.wav")).expect("KVAD_PHONE_TEST_AUDIO"));
    let (url, model) = (env("KVAD_PHONE_TEST_SERVER", "127.0.0.1:5823"), env("KVAD_PHONE_TEST_MODEL", "NbAiLab/nb-whisper-small"));
    let dir = folder("live");
    let (s, rx) = speak();
    let server = || Some(Server::new(&url, None).unwrap());
    s.start(server(), &model, "no", dir.to_str().unwrap(), false);
    // `KVAD_PHONE_TEST_BACKLOG` seconds of it at once first, as a session
    // begun again after a restart is sent what was said meanwhile.
    let backlog: u64 = env("KVAD_PHONE_TEST_BACKLOG", "0").parse().unwrap();
    let begun = Instant::now();
    for (i, chunk) in audio.chunks(RATE as usize / 10).enumerate() {
        s.audio(chunk);
        // At the pace it was spoken, after the backlog.
        let due = begun + Duration::from_millis((100 * (i as u64 + 1)).saturating_sub(1000 * backlog));
        std::thread::sleep(due.saturating_duration_since(Instant::now()));
    }
    s.stop();
    let stopped = Instant::now();
    let events = until(&rx, "speak_done", 120);
    let take = events.last().unwrap()["speak_done"]["take"].clone();
    let live = take["text"].as_str().unwrap().to_string();
    let first = events.iter().position(|e| name(e) == "speak_settled");
    eprintln!("live, done {:.1} s after the stop: {live}", stopped.elapsed().as_secs_f64());
    eprintln!("{} partials, {} settled events, the first after {:?} events", events.iter().filter(|e| name(e) == "speak_partial").count(), events.iter().filter(|e| name(e) == "speak_settled").count(), first);
    assert_eq!(take["how"], "live", "{events:?}");
    assert!(first.is_some(), "nothing was settled before the end");
    kept(&dir, &take, audio.len() as u64);

    s.retranscribe(server(), dir.to_str().unwrap(), take["id"].as_str().unwrap(), &model, "no");
    let events = until(&rx, "retranscribe_done", 120);
    let file = events.last().unwrap()["retranscribe_done"]["take"]["text"].as_str().unwrap().to_string();
    eprintln!("file: {file}");
    let words = |t: &str| t.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).filter(|w| !w.is_empty()).collect::<Vec<_>>();
    let (l, f) = (words(&live), words(&file));
    let shared = l.iter().filter(|w| f.contains(w)).count();
    eprintln!("{shared} of the live take's {} words are in the file's {}", l.len(), f.len());
    // `KVAD_PHONE_TEST_KEEP` keeps the take, and its link's log, to read.
    if std::env::var_os("KVAD_PHONE_TEST_KEEP").is_some() {
        eprintln!("kept in {}", dir.display());
    } else {
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(shared * 10 >= f.len() * 8, "live and file differ by more than a fifth");
}
