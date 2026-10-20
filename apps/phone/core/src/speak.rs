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
//! - **A session the server has let go of**, as a server that was restarted
//!   has: a new one is begun from the last settled word, and what was said
//!   since is sent from the file, so the words go on coming live. A Mac out
//!   of reach for longer than [`HOLD`] is tried on and on, and goes live
//!   again the same way once it answers.
//! - **A server with no live transcription**, one that fell too far behind,
//!   or a Mac still out of reach when the take is stopped: the take goes on
//!   into its file alone, and when it ends it is sent whole to
//!   `POST /v1/audio/transcriptions`.
//!
//! **Paragraphs.** A pause of [`PAUSE`] or more begins a new paragraph.
//! The pauses are found in the audio as it comes ([`Quiet`]), and a settled
//! word heard after one ends goes down after a blank line. A take sent whole
//! has its paragraphs from the gaps between Whisper's segments instead.
//! Who is speaking would begin one too, once kvad can tell speakers apart
//! (docs/diarization-plan.md).
//!
//! Every take is a WAV and a JSON file beside it, named by the millisecond
//! it began, in the folder the app gives. The app makes that folder, with
//! its protection class and its backups setting; the files take the
//! folder's class.

use crate::api::{Refused, Server, Socket};
use crate::level::Level;
use crate::session::{event, Emit};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the app records at, and what Whisper hears.
pub const RATE: u64 = 16_000;

/// How long the server holds a session after its socket goes
/// (`realtime::HOLD` in kvad-serve): after this the app is told the Mac is
/// out of reach, and a take stopped meanwhile is sent whole.
pub const HOLD: Duration = Duration::from_secs(60);

/// The shortest quiet that begins a new paragraph, in samples: a second
/// and a half, longer than a breath or a comma, shorter than a turn.
pub const PAUSE: u64 = RATE * 3 / 2;

/// The blocks loudness is judged in, in samples: 50 ms.
const BLOCK: u64 = RATE / 20;

/// Below this amplitude a block is quiet in any room: about -66 dBFS.
const QUIETEST: f32 = 0.0005;

/// Where the noise floor starts, before the room has been heard: a quiet
/// room through a phone's microphone, about -54 dBFS.
const FLOOR: f32 = 0.002;

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
    /// The pauses in it so far.
    quiet: Mutex<Quiet>,
    /// The gain the samples go through first, for a take of a room
    /// (`level.rs`).
    level: Option<Mutex<Level>>,
}

/// The pauses in a take, found from its loudness 50 ms at a time against
/// the room's own noise: a block is quiet when it is under 2.5 times the
/// floor's amplitude (8 dB). The floor falls at once to a quieter block, and
/// rises only toward a block that was quiet, over a second or so: a room
/// that gets a little noisier is followed, and a voice, however steady,
/// never raises it. A room that gets very much noisier at once has no
/// pauses until it settles, which is the safe way to be wrong.
#[derive(Default)]
pub struct Quiet {
    floor: Option<f32>,
    /// Where the quiet under way began.
    since: Option<u64>,
    /// The pauses over, as the samples they began and ended at.
    pauses: Vec<(u64, u64)>,
    /// Samples fed so far.
    at: u64,
    /// The block being filled: its samples and their sum of squares.
    block: (u64, f64),
}

impl Quiet {
    pub fn feed(&mut self, samples: &[i16]) {
        for &s in samples {
            let x = s as f64 / 32768.0;
            self.block.0 += 1;
            self.block.1 += x * x;
            self.at += 1;
            if self.block.0 == BLOCK {
                let rms = (self.block.1 / BLOCK as f64).sqrt() as f32;
                self.block = (0, 0.0);
                self.judge(rms);
            }
        }
    }

    fn judge(&mut self, rms: f32) {
        let floor = self.floor.get_or_insert(FLOOR);
        let quiet = rms < (*floor * 2.5).max(QUIETEST);
        if rms < *floor {
            *floor = rms;
        } else if quiet {
            *floor += (rms - *floor) * 0.05;
        }
        let begun = self.at - BLOCK;
        match (quiet, self.since) {
            (true, None) => self.since = Some(begun),
            (false, Some(since)) => {
                if begun - since >= PAUSE {
                    self.pauses.push((since, begun));
                }
                self.since = None;
            }
            _ => {}
        }
    }

    /// Whether a pause ended after sample `after` and by sample `to`.
    pub fn ended_between(&self, after: u64, to: u64) -> bool {
        self.pauses.iter().any(|&(_, end)| end > after && end <= to)
    }

    pub fn pauses(&self) -> &[(u64, u64)] {
        &self.pauses
    }

    /// The pauses, in seconds.
    pub fn seconds(&self) -> Vec<(f64, f64)> {
        self.pauses.iter().map(|&(a, b)| (a as f64 / RATE as f64, b as f64 / RATE as f64)).collect()
    }

    /// The pauses in a take's WAV, for a take kept before its pauses were.
    /// Found in what was saved, which for a take of a room is after its gain
    /// (`level.rs`): the gain holds between voices, so a pause after a quiet
    /// one is raised with it and can be missed, never invented.
    pub fn of_file(wav: &Path) -> Vec<(f64, f64)> {
        let mut quiet = Quiet::default();
        let Ok(mut file) = File::open(wav) else { return Vec::new() };
        let mut bytes = vec![0u8; 64 * 1024];
        let mut skip = HEADER as usize;
        let mut odd: Option<u8> = None;
        while let Ok(n) = file.read(&mut bytes) {
            if n == 0 {
                break;
            }
            let mut chunk: Vec<u8> = odd.take().into_iter().collect();
            chunk.extend_from_slice(&bytes[skip.min(n)..n]);
            skip = skip.saturating_sub(n);
            if chunk.len() % 2 == 1 {
                odd = chunk.pop();
            }
            let samples: Vec<i16> = chunk.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
            quiet.feed(&samples);
        }
        quiet.seconds()
    }
}

impl Speak {
    pub fn new(emit: Emit) -> Speak {
        Speak { emit, take: Current::default() }
    }

    /// Begins a take in `folder`, live through `server`. The app starts its
    /// microphone once this returns: the take's file is there by then.
    pub fn start(&self, server: Option<Server>, model: &str, language: &str, folder: &str, level: bool) {
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
            quiet: Mutex::new(Quiet::default()),
            level: level.then(|| Mutex::new(Level::default())),
        });
        *current = Some(take.clone());
        drop(current);
        (self.emit)(event("speak_started", json!({ "id": id })));
        let (emit, current) = (self.emit.clone(), self.take.clone());
        std::thread::spawn(move || Live::new(emit, current, take, server, reader).run());
    }

    /// Samples from the microphone, 16 kHz mono, into the take's file,
    /// through its gain if it has one. Dropped when no take is being
    /// recorded.
    pub fn audio(&self, samples: &[i16]) {
        let Some(take) = self.take.lock().unwrap().clone() else { return };
        let mut writer = take.writer.lock().unwrap();
        let Some(w) = writer.as_mut() else { return };
        // The pauses are found in the room as it is, before any gain.
        take.quiet.lock().unwrap().feed(samples);
        let mut levelled = Vec::new();
        let samples = match &take.level {
            Some(level) => {
                levelled.extend_from_slice(samples);
                level.lock().unwrap().feed(&mut levelled);
                &levelled[..]
            }
            None => samples,
        };
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
        let _ = std::fs::remove_file(take.wav.with_extension("log"));
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
                    let v = record(&id, started, samples, "", "", "", "", "", None);
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
        let _ = std::fs::remove_file(folder.join(format!("{id}.log")));
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
            // The pauses kept with the take, heard before any gain; a take
            // from before they were kept has them found in its file.
            let kept = std::fs::read(wav.with_extension("json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            let pauses = match kept.as_ref().and_then(|v| v["pauses"].as_array()) {
                Some(p) => p.iter().filter_map(|p| Some((p[0].as_f64()?, p[1].as_f64()?))).collect(),
                None => Quiet::of_file(&wav),
            };
            let result = server.transcribe(&wav, &model, &language, &pauses, &stop, |piece| emit(event("retranscribe_piece", json!({ "id": id, "text": piece }))));
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
/// What is kept of a take beside its WAV. `pauses` are where it was quiet
/// for [`PAUSE`] or more, in seconds, heard before any gain, so that the take
/// transcribed again has its paragraphs where live had them. A take found
/// without its record has none: they are found in its file if it is sent.
#[allow(clippy::too_many_arguments)]
fn record(id: &str, started: u64, samples: u64, model: &str, language: &str, text: &str, how: &str, server: &str, pauses: Option<&[(f64, f64)]>) -> Value {
    let mut v = json!({
        "id": id,
        "started": started,
        "seconds": samples as f64 / RATE as f64,
        "model": model,
        "language": language,
        "text": text,
        "how": how,
        "server": server,
    });
    if let Some(pauses) = pauses {
        let hundredths = |x: f64| (x * 100.0).round() / 100.0;
        v["pauses"] = pauses.iter().map(|&(a, b)| [hundredths(a), hundredths(b)]).collect::<Vec<_>>().into();
    }
    v
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
    /// `completed` came: the take's text is all settled.
    Done,
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
    /// The take's sample that is the session's first. A session the server
    /// let go of is followed by one that begins where its words were
    /// settled to, and the offsets the server speaks of are the session's.
    origin: u64,
    /// The offset of the last settled event, what a reconnect resumes from.
    settled_to: u64,
    /// The samples the server has.
    sent: u64,
    /// Settled events had, by `event_id`, so that those sent again are
    /// dropped. A session's ids are its own.
    seen: HashSet<String>,
    /// The take's text so far, every session's, with its paragraphs.
    text: String,
    /// The take's sample up to which pauses have made their paragraphs.
    broke_to: u64,
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
    /// In debug builds, a log of the take's link beside it (`<id>.log`):
    /// connects, sessions, offsets, errors and why each socket ended, and
    /// none of the words. For field runs, where nothing else can be seen.
    trace: Option<File>,
    began: Instant,
}

impl Live {
    fn new(emit: Emit, current: Current, take: Arc<Take>, server: Server, reader: File) -> Live {
        Live {
            emit,
            current,
            server,
            reader,
            id: None,
            origin: 0,
            settled_to: 0,
            sent: 0,
            seen: HashSet::new(),
            text: String::new(),
            broke_to: 0,
            committed: false,
            owed: None,
            said: "",
            outbox: Vec::new(),
            trace: cfg!(debug_assertions).then(|| File::create(take.wav.with_extension("log")).ok()).flatten(),
            began: Instant::now(),
            take,
        }
    }

    fn note(&mut self, what: impl AsRef<str>) {
        let at = self.began.elapsed().as_secs_f64();
        if let Some(f) = &mut self.trace {
            let _ = writeln!(f, "{at:8.3} {}", what.as_ref());
        }
    }

    /// The socket is done with, for `why`: try another.
    fn gone(&mut self, why: impl AsRef<str>) -> Option<End> {
        self.note(format!("socket gone: {}", why.as_ref()));
        None
    }

    fn run(mut self) {
        let end = self.live();
        if let End::Cancelled = end {
            return;
        }
        // Wait for the end of the take, whatever happened to the link.
        while !self.take.stop.load(Ordering::Relaxed) && !self.take.cancel.load(Ordering::Relaxed) && self.broken().is_none() && !matches!(end, End::Done | End::Failed(_)) {
            std::thread::sleep(Duration::from_millis(50));
        }
        if self.take.cancel.load(Ordering::Relaxed) {
            return;
        }
        let samples = self.close_file();
        let so_far = self.text.trim().to_string();
        let (text, how, failed) = match end {
            End::Done => (so_far, "live", None),
            End::Failed(why) => (so_far, "", Some(why)),
            End::Offline => match self.broken() {
                Some(broken) => (so_far, "", Some(broken)),
                None => {
                    self.link("sending");
                    match self.send_whole() {
                        Ok(text) => (text, "file", None),
                        Err(e) => (so_far, "", Some(format!("The take could not be transcribed: {e} It is kept on this phone, to send again."))),
                    }
                }
            },
            End::Cancelled => unreachable!("returned above"),
        };
        let t = &self.take;
        let pauses = t.quiet.lock().unwrap().seconds();
        let v = record(&t.id, t.started, samples, &t.model, &t.language, &text, how, &t.server, Some(&pauses));
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
            let pauses = self.take.quiet.lock().unwrap().seconds();
            match self.server.transcribe(&self.take.wav, &self.take.model, &self.take.language, &pauses, &stop, |_| {}) {
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

    /// The server has let go of the session, as a server restarted has: a
    /// new one, beginning at the last word settled, for the rest of the
    /// take. What was said after it is sent from the file.
    fn anew(&mut self) {
        self.note(format!("session lost; a new one from {}", self.origin + self.settled_to));
        self.origin += self.settled_to;
        self.settled_to = 0;
        self.sent = 0;
        self.id = None;
        self.seen.clear();
        self.committed = false;
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
            let out_of_reach = down.elapsed() > HOLD;
            if out_of_reach && self.take.stop.load(Ordering::Relaxed) {
                return End::Offline;
            }
            if out_of_reach && self.said != "offline" {
                self.said = "offline";
                let why = "The Mac cannot be reached. Recording goes on, and the words come back live once it answers.";
                (self.emit)(event("speak_link", json!({ "state": "offline", "why": why })));
            }
            if tries > 0 {
                if !out_of_reach {
                    self.link("reconnecting");
                }
                std::thread::sleep(Duration::from_millis(250 << tries.min(3)));
            }
            tries += 1;
            let path = match &self.id {
                None => "/v1/realtime?intent=transcription".to_string(),
                Some(id) => format!("/v1/realtime?intent=transcription&resume={id}&offset={}", self.settled_to),
            };
            self.note(format!("connect {}", if self.id.is_some() { format!("resume from {}", self.settled_to) } else { format!("fresh at {}", self.origin) }));
            let socket = match self.server.socket(&path) {
                Ok(s) => s,
                Err(r) => {
                    self.note(format!("refused {:?}: {}", r.status, r.why));
                    match r {
                        Refused { status: Some(401 | 403), why } => return End::Failed(why),
                        // Busy, or starting: again in a moment.
                        Refused { status: Some(429 | 500..) | None, .. } => continue,
                        // Any other answer but the upgrade: a server from
                        // before `/v1/realtime`, which may answer an unknown
                        // path with its web page.
                        Refused { status: Some(_), .. } => return self.offline("This server has no live transcription. The take is sent whole when it ends.".into()),
                    }
                }
            };
            match self.on(socket) {
                Some(end) => return end,
                // The link went, or the session did: try again. A session
                // let go of has been begun again (`anew`), so the next
                // socket asks for a new one.
                None => {
                    if self.said == "live" {
                        down = Instant::now();
                    }
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
                            if let Err(e) = ws.send(text(m)) {
                                return self.gone(format!("sending an event: {e}"));
                            }
                        }
                        match heard {
                            Heard::On => {}
                            Heard::Gone => return self.gone("the server said so"),
                            Heard::End(end) => {
                                if let End::Done = end {
                                    closed(&mut ws);
                                }
                                return Some(end);
                            }
                        }
                    }
                    Ok(_) => self.owed = None,
                    Err(tungstenite::Error::Io(e)) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
                    Err(e) => return self.gone(format!("reading: {e}")),
                }
            }
            if !ready {
                if self.owed.is_some_and(|t| t.elapsed() > STALLED) {
                    return self.gone("no session.created in time");
                }
                continue;
            }
            // What is to go.
            self.link("live");
            let recorded = self.take.recorded.load(Ordering::Acquire).saturating_sub(self.origin);
            let mut sent_now = 0;
            while self.sent < recorded && sent_now < 10 * FRAME {
                let n = (recorded - self.sent).min(FRAME);
                let mut pcm = vec![0u8; 2 * n as usize];
                if self.reader.read_exact_at(&mut pcm, HEADER + 2 * (self.origin + self.sent)).is_err() {
                    return Some(End::Failed("The take could not be read back.".into()));
                }
                if let Err(e) = ws.send(tungstenite::Message::Binary(pcm.into())) {
                    return self.gone(format!("sending audio at {}: {e}", self.sent));
                }
                if (self.sent + n) / (5 * RATE) > self.sent / (5 * RATE) {
                    self.note(format!("sent to {} of {recorded}", self.sent + n));
                }
                self.sent += n;
                sent_now += n;
                self.owed.get_or_insert_with(Instant::now);
            }
            if self.take.stop.load(Ordering::Relaxed) && self.sent >= recorded && !commit_sent && !self.committed {
                if let Err(e) = ws.send(text(json!({ "type": "input_audio_buffer.commit" }))) {
                    return self.gone(format!("sending the commit: {e}"));
                }
                self.note(format!("commit at {}", self.sent));
                commit_sent = true;
                self.owed.get_or_insert_with(Instant::now);
            }
            if self.owed.is_some_and(|t| t.elapsed() > STALLED) {
                return self.gone(format!("nothing heard for {} s", STALLED.as_secs()));
            }
        }
    }

    /// One event from the server.
    fn heard(&mut self, v: &Value, ready: &mut bool) -> Heard {
        let kind = v["type"].as_str().unwrap_or_default();
        let lag = v["kvad"]["lag"].as_f64().unwrap_or(0.0);
        // A settled event sent again after a reconnect is one had already.
        let again = |seen: &mut HashSet<String>| !seen.insert(v["event_id"].as_str().unwrap_or_default().to_string());
        let k = &v["kvad"];
        let line = match kind {
            "kvad.transcription.partial" => format!("partial {} chars, heard {}, lag {}", v["text"].as_str().map_or(0, |t| t.chars().count()), k["heard"], k["lag"]),
            "conversation.item.input_audio_transcription.delta" => format!("delta {} {} chars to {}", v["event_id"], v["delta"].as_str().map_or(0, |t| t.chars().count()), k["offset"]),
            "session.created" => format!("session {} resumed {} offset {} item {}", v["session"]["id"], k["resumed"], k["offset"], k["item_id"]),
            "error" => format!("error {}: {}", v["error"]["code"], v["error"]["message"]),
            other => format!("{other} {}", v["event_id"]),
        };
        self.note(line);
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
                    self.settled_to = v["kvad"]["offset"].as_u64().unwrap_or(self.settled_to);
                    // Heard after a pause ended: a new paragraph.
                    let heard = self.origin + self.settled_to;
                    let paragraph = !self.text.trim().is_empty() && self.take.quiet.lock().unwrap().ended_between(self.broke_to, heard);
                    self.broke_to = self.broke_to.max(heard);
                    let piece = if paragraph { format!("\n\n{}", delta.trim_start()) } else { delta.to_string() };
                    self.text.push_str(&piece);
                    (self.emit)(event("speak_settled", json!({ "text": piece, "lag": lag })));
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
                Heard::End(End::Done)
            }
            "error" => {
                let e = &v["error"];
                let why = e["message"].as_str().unwrap_or("The server refused the live session.").to_string();
                match e["code"].as_str().unwrap_or_default() {
                    "session_not_found" => {
                        self.anew();
                        Heard::Gone
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `secs` of a 220 Hz tone at amplitude `a`, with a noise of amplitude
    /// `noise` under it, from a fixed seed.
    fn sound(secs: f64, a: f64, noise: f64, seed: &mut u64) -> Vec<i16> {
        (0..(secs * RATE as f64) as usize)
            .map(|i| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let n = ((*seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0 * noise;
                let x = a * (i as f64 * 220.0 * std::f64::consts::TAU / RATE as f64).sin() + n;
                (x * 32767.0) as i16
            })
            .collect()
    }

    fn pauses(parts: &[(f64, f64)], noise: f64) -> Vec<(f64, f64)> {
        let (mut q, mut seed) = (Quiet::default(), 7u64);
        for &(secs, a) in parts {
            // Fed as the app feeds it, a tenth of a second at a time.
            for chunk in sound(secs, a, noise, &mut seed).chunks(1600) {
                q.feed(chunk);
            }
        }
        q.pauses().iter().map(|&(a, b)| (a as f64 / RATE as f64, b as f64 / RATE as f64)).collect()
    }

    #[test]
    fn a_pause_is_found_where_it_is() {
        assert_eq!(pauses(&[(3.0, 0.1), (2.0, 0.0), (3.0, 0.1)], 0.0), [(3.0, 5.0)]);
    }

    #[test]
    fn a_breath_is_not_a_pause() {
        assert_eq!(pauses(&[(3.0, 0.1), (0.8, 0.0), (3.0, 0.1), (1.2, 0.0), (2.0, 0.1)], 0.0), []);
    }

    /// In a noisy room the quiet between sentences is the room's noise, not
    /// silence; and a voice held steady for long is still a voice.
    #[test]
    fn a_pause_in_a_noisy_room_is_found_and_steady_speech_is_not_one() {
        assert_eq!(pauses(&[(20.0, 0.05), (2.0, 0.0), (3.0, 0.05)], 0.004), [(20.0, 22.0)]);
    }
}
