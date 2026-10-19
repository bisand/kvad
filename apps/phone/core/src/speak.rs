//! Speak: a take recorded on the phone and transcribed by the Mac while it
//! is being recorded, over `/v1/realtime` (docs/phone-speak-plan.md, and
//! the server's end in docs/speech-plan.md, *Live, from the phone*).
//!
//! The app's microphone hands 16 kHz mono 16-bit samples to [`Speak::audio`]
//! a tenth of a second at a time, on its own thread. They go into the take's
//! file before anything else, and a thread of the take's own sends them on
//! from the file. So the file is always ahead of the socket, and what the
//! link loses the file still has:
//!
//! - **A socket that fails** is opened again with the session's id and the
//!   offset of the last settled word, for as long as the server holds the
//!   session ([`HOLD`]). The server says how much audio it has, and the rest
//!   is sent again from the file. Settled events it sends again are known by
//!   their `event_id` and dropped.
//! - **A session the server has let go of**, a server with no live
//!   transcription, or one that fell too far behind: the take goes on into
//!   its file alone, and when it ends it is sent whole to
//!   `POST /v1/audio/transcriptions`.
//!
//! Every take is a WAV and a JSON file beside it, named by the millisecond
//! it began, in the folder the app gives. The app makes that folder, with
//! its protection class and its backups setting; the files take the
//! folder's class.

use crate::api::{Refused, Server, Socket};
use crate::session::{event, Emit};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::File;
use std::io::{ErrorKind, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the app records at, and what Whisper hears.
pub const RATE: u64 = 16_000;

/// How long the server holds a session after its socket goes
/// (`realtime::HOLD` in kvad-serve): after this the phone stops trying.
pub const HOLD: Duration = Duration::from_secs(60);

/// How long the socket may go without a word from the server while it owes
/// one (audio sent, or a commit) before it is taken for dead. The server
/// answers about once a second while audio comes.
const STALLED: Duration = Duration::from_secs(15);

/// How long a read waits: the loop's tick, between sending and reading.
const TICK: Duration = Duration::from_millis(20);

/// The most audio a frame carries, in samples: a tenth of a second, what
/// the app hands over at a time.
const FRAME: u64 = RATE / 10;

/// The size of a WAV's header, before its samples.
const HEADER: u64 = 44;

pub struct Speak {
    emit: Emit,
    /// The take being recorded, if one is.
    take: Current,
}

/// The take being recorded, shared with its thread, which clears it at the
/// end.
type Current = Arc<Mutex<Option<Arc<Take>>>>;

/// A take while it is being recorded.
struct Take {
    id: String,
    wav: PathBuf,
    meta: PathBuf,
    started: u64,
    model: String,
    language: String,
    server: String,
    /// Where the samples go. `None` once the take has ended, when audio
    /// that still comes is dropped.
    writer: Mutex<Option<File>>,
    /// Samples written so far.
    recorded: AtomicU64,
    stop: AtomicBool,
    cancel: AtomicBool,
    /// Why the file could not be written, which ends the take.
    broken: Mutex<Option<String>>,
}

impl Speak {
    pub fn new(emit: Emit) -> Speak {
        Speak { emit, take: Current::default() }
    }

    /// Begins a take in `folder`, live through `server`. The app starts its
    /// microphone once this returns: the take's file is there by then.
    pub fn start(&self, server: Option<Server>, model: &str, language: &str, folder: &str) {
        let fail = |e: String| (self.emit)(event("speak_failed", json!({ "error": e, "kept": false })));
        let Some(server) = server else { return fail("Not connected to a server.".into()) };
        let mut current = self.take.lock().unwrap();
        if current.is_some() {
            return fail("A take is being recorded already. Stop it first.".into());
        }
        let folder = Path::new(folder);
        if let Err(e) = std::fs::create_dir_all(folder) {
            return fail(format!("There is nowhere to keep the take: {e}"));
        }
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let id = now.as_millis().to_string();
        let wav = folder.join(format!("{id}.wav"));
        let opened = std::fs::OpenOptions::new().write(true).create_new(true).open(&wav).and_then(|mut w| {
            w.write_all(&header(0))?;
            // Opened now, while the phone is unlocked: a file of class
            // `completeUnlessOpen` cannot be opened again once it is locked.
            let r = File::open(&wav)?;
            Ok((w, r))
        });
        let (writer, reader) = match opened {
            Ok(both) => both,
            Err(e) => return fail(format!("The take could not be made: {e}")),
        };
        let take = Arc::new(Take {
            meta: folder.join(format!("{id}.json")),
            id: id.clone(),
            wav,
            started: now.as_secs(),
            model: model.to_string(),
            language: language.to_string(),
            server: server.name().to_string(),
            writer: Mutex::new(Some(writer)),
            recorded: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            broken: Mutex::new(None),
        });
        *current = Some(take.clone());
        drop(current);
        (self.emit)(event("speak_started", json!({ "id": id })));
        let (emit, current) = (self.emit.clone(), self.take.clone());
        std::thread::spawn(move || Live::new(emit, current, take, server, reader).run());
    }

    /// Samples from the microphone, 16 kHz mono, into the take's file.
    /// Dropped when no take is being recorded.
    pub fn audio(&self, samples: &[i16]) {
        let Some(take) = self.take.lock().unwrap().clone() else { return };
        let mut writer = take.writer.lock().unwrap();
        let Some(w) = writer.as_mut() else { return };
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        match w.write_all(&bytes) {
            Ok(()) => {
                take.recorded.fetch_add(samples.len() as u64, Ordering::Release);
            }
            Err(e) => {
                *writer = None;
                *take.broken.lock().unwrap() = Some(format!("The take could not be written: {e}"));
            }
        }
    }

    /// Ends the take: what is left is sent, and the rest settled.
    pub fn stop(&self) {
        if let Some(take) = self.take.lock().unwrap().as_ref() {
            take.stop.store(true, Ordering::Relaxed);
        }
    }

    /// Ends the take and throws it away.
    pub fn cancel(&self) {
        let Some(take) = self.take.lock().unwrap().take() else { return };
        take.cancel.store(true, Ordering::Relaxed);
        *take.writer.lock().unwrap() = None;
        let _ = std::fs::remove_file(&take.wav);
        let _ = std::fs::remove_file(&take.meta);
        (self.emit)(event("speak_cancelled", json!({ "id": take.id })));
    }

    /// Every take in `folder`, newest first. A take with no JSON beside its
    /// WAV is one the app was stopped in the middle of: its header is put
    /// right and it is listed as not transcribed.
    pub fn takes(&self, folder: &str) {
        let recording = self.take.lock().unwrap().as_ref().map(|t| t.id.clone());
        let mut list: Vec<Value> = Vec::new();
        for entry in std::fs::read_dir(folder).into_iter().flatten().flatten() {
            let path = entry.path();
            let Some(id) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else { continue };
            if path.extension().and_then(|e| e.to_str()) != Some("wav") || Some(&id) == recording.as_ref() {
                continue;
            }
            let meta = path.with_extension("json");
            let kept = std::fs::read(&meta).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            list.push(match kept {
                Some(v) => v,
                None => {
                    let samples = std::fs::metadata(&path).map(|m| m.len().saturating_sub(HEADER) / 2).unwrap_or(0);
                    if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&path) {
                        let _ = f.write_all_at(&header(samples), 0);
                    }
                    let started = id.parse::<u64>().map(|ms| ms / 1000).unwrap_or(0);
                    let v = record(&id, started, samples, "", "", "", "", "");
                    let _ = std::fs::write(&meta, v.to_string());
                    v
                }
            });
        }
        list.sort_by(|a, b| b["id"].as_str().unwrap_or("").cmp(a["id"].as_str().unwrap_or("")).then(b["started"].as_u64().cmp(&a["started"].as_u64())));
        (self.emit)(event("takes", json!({ "takes": list })));
    }

    /// Deletes take `id` in `folder`, its audio and its transcript.
    pub fn delete(&self, folder: &str, id: &str) {
        if !plain(id) || self.take.lock().unwrap().as_ref().is_some_and(|t| t.id == id) {
            return (self.emit)(event("error", json!({ "error": "That take cannot be deleted now." })));
        }
        let folder = Path::new(folder);
        let _ = std::fs::remove_file(folder.join(format!("{id}.wav")));
        let _ = std::fs::remove_file(folder.join(format!("{id}.json")));
        (self.emit)(event("take_deleted", json!({ "id": id })));
    }

    /// Take `id` transcribed again, whole, by `model` on the file endpoint:
    /// a better model than live could keep up with, or a take that never
    /// reached a Mac. Its transcript is replaced once the new one is in.
    pub fn retranscribe(&self, server: Option<Server>, folder: &str, id: &str, model: &str, language: &str) {
        let fail = |e: &str| (self.emit)(event("retranscribe_failed", json!({ "id": id, "error": e })));
        let Some(server) = server else { return fail("Not connected to a server.") };
        if !plain(id) || self.take.lock().unwrap().as_ref().is_some_and(|t| t.id == id) {
            return fail("That take cannot be sent now.");
        }
        let wav = Path::new(folder).join(format!("{id}.wav"));
        if !wav.exists() {
            return fail("That take is not on this phone.");
        }
        let (emit, id, model, language) = (self.emit.clone(), id.to_string(), model.to_string(), language.to_string());
        std::thread::spawn(move || {
            let stop = AtomicBool::new(false);
            let result = server.transcribe(&wav, &model, &language, &stop, |piece| emit(event("retranscribe_piece", json!({ "id": id, "text": piece }))));
            match result {
                Ok(text) => {
                    let meta = wav.with_extension("json");
                    let mut v = std::fs::read(&meta).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).unwrap_or_else(|| json!({ "id": id }));
                    v["text"] = text.into();
                    v["how"] = "file".into();
                    v["model"] = model.into();
                    v["language"] = language.into();
                    v["server"] = server.name().into();
                    match std::fs::write(&meta, v.to_string()) {
                        Ok(()) => emit(event("retranscribe_done", json!({ "take": v }))),
                        Err(e) => emit(event("retranscribe_failed", json!({ "id": id, "error": format!("The transcript could not be kept: {e}") }))),
                    }
                }
                Err(e) => emit(event("retranscribe_failed", json!({ "id": id, "error": e }))),
            }
        });
    }

}

/// A take's id is the millisecond it began, so nothing in it can climb out
/// of the folder.
fn plain(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit())
}

/// What a take's JSON holds, and what the app is told of it. `how` is
/// `live` or `file`, for how its text was written, or empty for a take
/// with no transcript yet.
#[allow(clippy::too_many_arguments)]
fn record(id: &str, started: u64, samples: u64, model: &str, language: &str, text: &str, how: &str, server: &str) -> Value {
    json!({
        "id": id,
        "started": started,
        "seconds": samples as f64 / RATE as f64,
        "model": model,
        "language": language,
        "text": text,
        "how": how,
        "server": server,
    })
}

/// A WAV header for `samples` of 16 kHz mono 16-bit PCM.
fn header(samples: u64) -> [u8; HEADER as usize] {
    let data = (samples * 2).min(u32::MAX as u64 - 36) as u32;
    let mut h = [0u8; HEADER as usize];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36 + data).to_le_bytes());
    h[8..16].copy_from_slice(b"WAVEfmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&1u16.to_le_bytes());
    h[24..28].copy_from_slice(&(RATE as u32).to_le_bytes());
    h[28..32].copy_from_slice(&(RATE as u32 * 2).to_le_bytes());
    h[32..34].copy_from_slice(&2u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data.to_le_bytes());
    h
}

/// How a take's thread ended its live session.
enum End {
    /// `completed` came: the transcript.
    Done(String),
    /// Live is over, and the take goes on into its file. The app has been
    /// told why.
    Offline,
    /// The take cannot go on: why.
    Failed(String),
    Cancelled,
}

/// A take's thread: the socket, what has been sent on it, and what has
/// come back.
struct Live {
    emit: Emit,
    current: Current,
    take: Arc<Take>,
    server: Server,
    /// The take's file, opened for reading when it was made.
    reader: File,
    /// The session's id, once the server has given one: what a reconnect
    /// names.
    id: Option<String>,
    /// The offset of the last settled event, what a reconnect resumes from.
    settled_to: u64,
    /// The samples the server has.
    sent: u64,
    /// Settled events had, by `event_id`, so that those sent again are
    /// dropped.
    seen: HashSet<String>,
    settled: String,
    /// The commit is in: the server said `committed`, or a resumed session
    /// is on an item after the take's first.
    committed: bool,
    /// Since when the server owes an answer, if it does.
    owed: Option<Instant>,
    /// What the app was last told of the link.
    said: &'static str,
    /// Events `heard` has for the server, which `on` sends: it holds the
    /// socket.
    outbox: Vec<Value>,
}

impl Live {
    fn new(emit: Emit, current: Current, take: Arc<Take>, server: Server, reader: File) -> Live {
        Live {
            emit,
            current,
            take,
            server,
            reader,
            id: None,
            settled_to: 0,
            sent: 0,
            seen: HashSet::new(),
            settled: String::new(),
            committed: false,
            owed: None,
            said: "",
            outbox: Vec::new(),
        }
    }

    fn run(mut self) {
        let end = self.live();
        if let End::Cancelled = end {
            return;
        }
        // Wait for the end of the take, whatever happened to the link.
        while !self.take.stop.load(Ordering::Relaxed) && !self.take.cancel.load(Ordering::Relaxed) && self.broken().is_none() && !matches!(end, End::Done(_) | End::Failed(_)) {
            std::thread::sleep(Duration::from_millis(50));
        }
        if self.take.cancel.load(Ordering::Relaxed) {
            return;
        }
        let samples = self.close_file();
        let (text, how, failed) = match end {
            End::Done(text) => (text, "live", None),
            End::Failed(why) => (self.settled.trim().to_string(), "", Some(why)),
            End::Offline => match self.broken() {
                Some(broken) => (self.settled.trim().to_string(), "", Some(broken)),
                None => {
                    self.link("sending");
                    match self.send_whole() {
                        Ok(text) => (text, "file", None),
                        Err(e) => (self.settled.trim().to_string(), "", Some(format!("The take could not be transcribed: {e} It is kept on this phone, to send again."))),
                    }
                }
            },
            End::Cancelled => unreachable!("returned above"),
        };
        let t = &self.take;
        let v = record(&t.id, t.started, samples, &t.model, &t.language, &text, how, &t.server);
        let kept = std::fs::write(&t.meta, v.to_string());
        // The take has ended: no more audio goes into it.
        let mut current = self.current.lock().unwrap();
        if current.as_ref().is_some_and(|c| c.id == t.id) {
            *current = None;
        }
        drop(current);
        match (failed, kept) {
            (None, Ok(())) => (self.emit)(event("speak_done", json!({ "take": v }))),
            (None, Err(e)) => (self.emit)(event("speak_failed", json!({ "error": format!("The transcript could not be kept: {e}"), "kept": true }))),
            (Some(why), _) => (self.emit)(event("speak_failed", json!({ "error": why, "kept": true }))),
        }
    }

    fn broken(&self) -> Option<String> {
        self.take.broken.lock().unwrap().clone()
    }

    /// No more audio into the take, and its header made right: the samples
    /// it holds.
    fn close_file(&self) -> u64 {
        let writer = self.take.writer.lock().unwrap().take();
        let samples = self.take.recorded.load(Ordering::Acquire);
        if let Some(w) = writer {
            let _ = w.write_all_at(&header(samples), 0);
            let _ = w.sync_all();
        }
        samples
    }

    /// The take whole to the file endpoint, a few times over a minute, for a
    /// Mac that is waking or a network coming back.
    fn send_whole(&self) -> Result<String, String> {
        let stop = AtomicBool::new(false);
        let mut last = String::new();
        for wait in [0, 5, 15, 30] {
            std::thread::sleep(Duration::from_secs(wait));
            if self.take.cancel.load(Ordering::Relaxed) {
                return Err("Cancelled.".into());
            }
            match self.server.transcribe(&self.take.wav, &self.take.model, &self.take.language, &stop, |_| {}) {
                Ok(text) => return Ok(text),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    /// Tells the app how the link is, when that changes: `live`,
    /// `reconnecting`, `offline` (with why) or `sending`.
    fn link(&mut self, state: &'static str) {
        if self.said != state {
            self.said = state;
            (self.emit)(event("speak_link", json!({ "state": state })));
        }
    }

    fn offline(&mut self, why: String) -> End {
        self.said = "offline";
        (self.emit)(event("speak_link", json!({ "state": "offline", "why": why })));
        End::Offline
    }

    /// The live session, from the first connect to `completed`, or to why
    /// it could not go on.
    fn live(&mut self) -> End {
        let mut down = Instant::now();
        let mut tries = 0u32;
        loop {
            if self.take.cancel.load(Ordering::Relaxed) {
                return End::Cancelled;
            }
            if let Some(why) = self.broken() {
                return End::Failed(why);
            }
            if down.elapsed() > HOLD {
                return self.offline("The Mac could not be reached for a minute. The take goes on, and is sent whole when it ends.".into());
            }
            if tries > 0 {
                self.link("reconnecting");
                std::thread::sleep(Duration::from_millis(250 << tries.min(3)));
            }
            tries += 1;
            let path = match &self.id {
                None => "/v1/realtime?intent=transcription".to_string(),
                Some(id) => format!("/v1/realtime?intent=transcription&resume={id}&offset={}", self.settled_to),
            };
            let socket = match self.server.socket(&path) {
                Ok(s) => s,
                Err(Refused { status: Some(401 | 403), why }) => return End::Failed(why),
                // Busy, or starting: again in a moment.
                Err(Refused { status: Some(429 | 500..), .. }) | Err(Refused { status: None, .. }) => continue,
                // Any other answer but the upgrade: a server from before
                // `/v1/realtime`, which may answer an unknown path with its
                // web page.
                Err(Refused { status: Some(_), .. }) => return self.offline("This server has no live transcription. The take is sent whole when it ends.".into()),
            };
            match self.on(socket) {
                Some(end) => return end,
                None => {
                    // The link went: try again, for as long as the server
                    // holds the session.
                    down = Instant::now();
                    tries = 1;
                }
            }
        }
    }

    /// One socket, until it goes (`None`) or the session ends.
    fn on(&mut self, mut ws: Socket) -> Option<End> {
        let _ = ws.get_ref().set_read_timeout(Some(TICK));
        self.owed = Some(Instant::now());
        let mut ready = false;
        let mut commit_sent = false;
        loop {
            if self.take.cancel.load(Ordering::Relaxed) {
                let _ = ws.close(None);
                return Some(End::Cancelled);
            }
            if let Some(why) = self.broken() {
                return Some(End::Failed(why));
            }
            // What has come.
            loop {
                match ws.read() {
                    Ok(tungstenite::Message::Text(body)) => {
                        self.owed = None;
                        let Ok(v) = serde_json::from_str::<Value>(&body) else { continue };
                        let heard = self.heard(&v, &mut ready);
                        for m in std::mem::take(&mut self.outbox) {
                            if ws.send(text(m)).is_err() {
                                return None;
                            }
                        }
                        match heard {
                            Heard::On => {}
                            Heard::Gone => return None,
                            Heard::End(end) => {
                                if let End::Done(_) = end {
                                    closed(&mut ws);
                                }
                                return Some(end);
                            }
                        }
                    }
                    Ok(_) => self.owed = None,
                    Err(tungstenite::Error::Io(e)) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
                    Err(_) => return None,
                }
            }
            if !ready {
                if self.owed.is_some_and(|t| t.elapsed() > STALLED) {
                    return None;
                }
                continue;
            }
            // What is to go.
            self.link("live");
            let recorded = self.take.recorded.load(Ordering::Acquire);
            let mut sent_now = 0;
            while self.sent < recorded && sent_now < 10 * FRAME {
                let n = (recorded - self.sent).min(FRAME);
                let mut pcm = vec![0u8; 2 * n as usize];
                if self.reader.read_exact_at(&mut pcm, HEADER + 2 * self.sent).is_err() {
                    return Some(End::Failed("The take could not be read back.".into()));
                }
                if ws.send(tungstenite::Message::Binary(pcm.into())).is_err() {
                    return None;
                }
                self.sent += n;
                sent_now += n;
                self.owed.get_or_insert_with(Instant::now);
            }
            if self.take.stop.load(Ordering::Relaxed) && self.sent >= recorded && !commit_sent && !self.committed {
                if ws.send(text(json!({ "type": "input_audio_buffer.commit" }))).is_err() {
                    return None;
                }
                commit_sent = true;
                self.owed.get_or_insert_with(Instant::now);
            }
            if self.owed.is_some_and(|t| t.elapsed() > STALLED) {
                return None;
            }
        }
    }

    /// One event from the server.
    fn heard(&mut self, v: &Value, ready: &mut bool) -> Heard {
        let kind = v["type"].as_str().unwrap_or_default();
        let lag = v["kvad"]["lag"].as_f64().unwrap_or(0.0);
        // A settled event sent again after a reconnect is one had already.
        let again = |seen: &mut HashSet<String>| !seen.insert(v["event_id"].as_str().unwrap_or_default().to_string());
        match kind {
            "session.created" => {
                let s = &v["session"];
                let resumed = v["kvad"]["resumed"].as_bool().unwrap_or(false);
                if resumed {
                    self.sent = v["kvad"]["offset"].as_u64().unwrap_or(0);
                    // The take is one item; on a later one, its commit is in.
                    self.committed |= v["kvad"]["item_id"].as_str().is_some_and(|i| i != "item_1");
                } else {
                    self.id = s["id"].as_str().map(str::to_string);
                    self.sent = 0;
                    let t = &self.take;
                    let mut transcription = json!({});
                    if !t.model.is_empty() {
                        transcription["model"] = t.model.clone().into();
                    }
                    if !t.language.is_empty() {
                        transcription["language"] = t.language.clone().into();
                    }
                    let update = json!({ "type": "session.update", "session": { "type": "transcription", "audio": { "input": {
                        "format": { "type": "audio/pcm", "rate": RATE },
                        "transcription": transcription,
                    }}}});
                    self.outbox.push(update);
                }
                // Where the frames that follow begin.
                self.outbox.push(json!({ "type": "kvad.input_audio_buffer.offset", "offset": self.sent }));
                *ready = true;
                Heard::On
            }
            "conversation.item.input_audio_transcription.delta" => {
                if !again(&mut self.seen) {
                    let delta = v["delta"].as_str().unwrap_or_default();
                    self.settled.push_str(delta);
                    self.settled_to = v["kvad"]["offset"].as_u64().unwrap_or(self.settled_to);
                    (self.emit)(event("speak_settled", json!({ "text": delta, "lag": lag })));
                }
                Heard::On
            }
            "kvad.transcription.partial" => {
                (self.emit)(event("speak_partial", json!({ "text": v["text"].as_str().unwrap_or_default(), "lag": lag })));
                Heard::On
            }
            "input_audio_buffer.committed" => {
                if !again(&mut self.seen) {
                    self.committed = true;
                    self.settled_to = v["kvad"]["offset"].as_u64().unwrap_or(self.settled_to);
                }
                Heard::On
            }
            "conversation.item.input_audio_transcription.completed" => {
                if again(&mut self.seen) {
                    return Heard::On;
                }
                Heard::End(End::Done(v["transcript"].as_str().unwrap_or_default().to_string()))
            }
            "error" => {
                let e = &v["error"];
                let why = e["message"].as_str().unwrap_or("The server refused the live session.").to_string();
                match e["code"].as_str().unwrap_or_default() {
                    "session_not_found" => Heard::End(self.offline("The Mac let go of the live session while the link was down. The take goes on, and is sent whole when it ends.".into())),
                    "too_far_behind" => Heard::End(self.offline("The Mac fell too far behind to keep up live. The take goes on, and is sent whole when it ends.".into())),
                    // A gap, or a socket taken over: a reconnect puts it right.
                    "audio_gap" | "session_taken" => Heard::Gone,
                    _ => Heard::End(End::Failed(why)),
                }
            }
            _ => Heard::On,
        }
    }
}

enum Heard {
    On,
    /// The socket is no good: open another.
    Gone,
    End(End),
}

fn text(v: Value) -> tungstenite::Message {
    tungstenite::Message::Text(v.to_string().into())
}

/// The clean end of a socket: a close, and its answer waited for a moment.
fn closed(ws: &mut Socket) {
    let _ = ws.close(None);
    let until = Instant::now() + Duration::from_secs(2);
    while Instant::now() < until {
        match ws.read() {
            Ok(_) => {}
            Err(tungstenite::Error::Io(e)) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => return,
        }
    }
}
