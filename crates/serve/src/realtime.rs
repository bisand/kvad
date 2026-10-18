//! `/v1/realtime?intent=transcription`: live transcription over a
//! WebSocket, audio up and words down while somebody is speaking, from a
//! Whisper's [`kvad::speech::Live`] session.
//!
//! OpenAI's Realtime API, its transcription sessions only, for the reason
//! the other `/v1` routes are OpenAI's: a client written for theirs speaks
//! to this one by changing a URL. In:
//!
//! - `session.update`, `{session: {type: "transcription", audio: {input:
//!   {format: {type: "audio/pcm", rate}, transcription: {model, language,
//!   prompt}}}}}`, before the first audio. `languages`, a list, is taken
//!   for `language`, its first. The beta's `transcription_session.update`,
//!   with `input_audio_format` and `input_audio_transcription`, is read
//!   too. Answered by `session.updated` (or `transcription_session.updated`).
//! - `input_audio_buffer.append`, `{audio}`: base64 of 16-bit little-endian
//!   mono PCM at the session's rate.
//! - `input_audio_buffer.commit`: the end of a turn. Everything heard is
//!   settled, and the next audio begins a new item, read after this one.
//!
//! Out, besides `session.created` on connecting, whose `session.id` names
//! the session for a reconnect: `input_audio_buffer.committed`;
//! `conversation.item.input_audio_transcription.delta`, `{item_id,
//! content_index, delta}`, text settled, which never changes once sent; and
//! `conversation.item.input_audio_transcription.completed`, `{item_id,
//! content_index, transcript}`, after a commit. A failure is `error`,
//! `{error: {type, code, message}}`.
//!
//! Kvad's own, as `docs/speech-plan.md` (*Live, from the phone*) planned:
//!
//! - **Binary frames are audio**, the same PCM as `append` carries without
//!   the base64, which is a third more bytes for nothing on a phone's
//!   uplink.
//! - **`kvad.transcription.partial`**, `{item_id, text}`: the current guess
//!   at the words after the settled ones. Each replaces the last.
//! - **`kvad` on the delta and the partial**: `heard`, how many seconds of
//!   audio the update had heard, and `lag`, how far the audio received is
//!   ahead of that as it is sent: how far behind the voice the text is.
//! - **The rate is 16 kHz** unless the session says otherwise. That is
//!   what Whisper hears, so nothing is resampled; OpenAI's is 24 kHz, which
//!   their clients name. Any other rate from 8 to 48 kHz is resampled here,
//!   linearly, which is enough for speech.
//!
//! An update runs about once a second, while there is new audio, as one job
//! on the model's engine. When one is still running as the next falls due,
//! the next is skipped: audio is never dropped, partials are. The clean end
//! is a commit, its `completed`, and then the client's close.
//!
//! **When the link drops.** A phone's link goes as it walks out of the
//! Wi-Fi, and the plan is that it keeps recording and loses nothing. So a
//! socket that goes without a close leaves its session held, for [`HOLD`]:
//! the [`Live`], the audio not yet decoded, and what was settled, in memory
//! and nowhere else. The update under way when it went is finished first.
//!
//! - **Every piece of audio has a sample offset**: samples at the session's
//!   rate since the session began. `input_audio_buffer.append` carries it
//!   as Kvad's own `kvad: {offset}`. A binary frame has no room for it, and
//!   a header would make it something other than the PCM a browser sends,
//!   so an event goes before it: `kvad.input_audio_buffer.offset`,
//!   `{offset}`, says where the next frame begins. Audio without an offset
//!   follows on from the piece before, so a client says it once after
//!   connecting and its frames follow on. Audio the session has already is
//!   passed over, and audio after a gap is refused (`audio_gap`, which says
//!   where to send from): placed, never doubled.
//! - **Every settled event says where it was heard to**: `kvad.offset` on
//!   the delta, the partial, `committed` and `completed`.
//! - **A reconnect** is `GET /v1/realtime?resume=<session.id>&offset=<n>`,
//!   `n` the `kvad.offset` of the last settled event the client has. Its
//!   `session.created` is the same session, with `kvad: {resumed: true,
//!   offset, item_id, transcript}`: the samples the server has, which the
//!   client sends on from, the item under way and its text settled so far.
//!   Then every settled event from `n` on comes again as it was, its
//!   `event_id` too, so a client drops the ones it had. A commit or an
//!   update that ran while the link was down is among them.
//! - **A socket the server still thinks is open**, as a phone that changed
//!   networks leaves behind until TCP gives up on it, is not in the way: a
//!   reconnect takes the session off it, and that socket is told
//!   `session_taken` and closed.
//! - **After the minute, or for an id not held**, the answer is `error`,
//!   `session_not_found`, and a close. The client has its recording, and
//!   `POST /v1/audio/transcriptions` takes it whole. A session is taken up
//!   only by the account and device that began it; another's is not found
//!   either.
//!
//! A client's close ends a session: nothing of it is held.

use crate::api::Fail;
use crate::auth::{Identity, State};
use crate::scheduler::{Key, Kind, Listened};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State as St};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use base64::Engine as _;
use kvad::speech::mel::SAMPLE_RATE;
use kvad::speech::{Live, Request, Update};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub fn routes() -> Router<State> {
    Router::new().route("/v1/realtime", get(realtime))
}

/// How often an update falls due: what `live_speed` measured at.
const CADENCE: Duration = Duration::from_secs(1);

/// How far behind the audio a session may fall before it is ended: a
/// minute of audio waiting for an engine that is busy with something else.
const MOST_BEHIND: f32 = 60.0;

/// The most audio one update hears, in samples at 16 kHz. A live window is
/// cut at a segment once it holds more than `CUT` (20 s); handed more than
/// the 10 s between that and Whisper's 30 s at once, it begins again at its
/// newest 30 s and what came before was never decoded. A backlog, from a
/// busy server or a slow decoder, is heard this much at a time instead, an
/// update after another until it is gone.
const MOST_AT_ONCE: usize = 5 * SAMPLE_RATE;

/// How long a session is held after its socket goes without a close, for
/// the client to reconnect and go on with it.
pub const HOLD: Duration = Duration::from_secs(60);

#[derive(serde::Deserialize)]
pub struct Asked {
    intent: Option<String>,
    model: Option<String>,
    /// A session to go on with, by the id its `session.created` gave.
    resume: Option<String>,
    /// The sample the reconnecting client has everything settled before:
    /// what was settled from there on is sent again.
    offset: Option<u64>,
}

async fn realtime(who: Identity, headers: HeaderMap, Query(asked): Query<Asked>, St(state): St<State>, ws: WebSocketUpgrade) -> Result<Response, Fail> {
    if !crate::auth::socket_may_use(&headers) {
        return Err(Fail(StatusCode::FORBIDDEN, "a WebSocket opened on a session cookie must come from this server's own pages".into(), None));
    }
    match asked.intent.as_deref() {
        None | Some("transcription") => {}
        Some(other) => return Err(Fail::bad(format!("intent={other}: this server's realtime sessions are transcription only"))),
    }
    // A session is taken up again by whoever began it, from the device it
    // began on.
    let owner = format!("{:?}/{:?}", who.id, who.device);
    let resume = asked.resume.map(|id| (id, asked.offset.unwrap_or(0)));
    let fresh = move || Session::new(Box::new(Engine { state, on: None }), asked.model);
    Ok(ws.on_upgrade(move |socket| connected(HELD.clone(), owner, resume, fresh, socket)))
}

/// 16-bit little-endian PCM at any rate, as 16 kHz samples, a piece at a
/// time: linear interpolation, carried over from one piece to the next.
struct Rate {
    from: usize,
    /// Where the next sample falls, in input samples after `last`.
    at: f64,
    /// The previous piece's last sample.
    last: f32,
}

impl Rate {
    fn new(from: usize) -> Rate {
        Rate { from, at: 1.0, last: 0.0 }
    }

    fn feed(&mut self, pcm: &[u8], out: &mut Vec<f32>) {
        let x: Vec<f32> = pcm.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
        if self.from == SAMPLE_RATE {
            out.extend_from_slice(&x);
            return;
        }
        let Some(&end) = x.last() else { return };
        let step = self.from as f64 / SAMPLE_RATE as f64;
        // The piece as read here: the last one's last sample, then this one.
        let at = |i: usize| if i == 0 { self.last } else { x[i - 1] };
        while self.at < x.len() as f64 {
            let i = self.at.floor() as usize;
            let f = (self.at - i as f64) as f32;
            out.push(at(i) * (1.0 - f) + at(i + 1) * f);
            self.at += step;
        }
        self.at -= x.len() as f64;
        self.last = end;
    }
}

type Pending = Pin<Box<dyn Future<Output = Listened> + Send>>;

/// What a session is set up to do, as `session.update` says it.
#[derive(Debug, Clone, PartialEq)]
struct Setup {
    model: Option<String>,
    request: Request,
    /// The audio's sample rate.
    rate: usize,
}

impl Setup {
    /// Read a `session.update`'s `session`, OpenAI's shape or the beta's,
    /// into this. Nothing changes when it is refused.
    fn read(&mut self, s: &Value) -> Result<(), String> {
        if let Some(t) = s["type"].as_str().filter(|&t| t != "transcription") {
            return Err(format!("session.type is `transcription` here, not `{t}`"));
        }
        let mut next = self.clone();
        let input = &s["audio"]["input"];
        let format = &input["format"];
        if let Some(t) = format["type"].as_str().or_else(|| s["input_audio_format"].as_str()) {
            if t != "audio/pcm" && t != "pcm16" {
                return Err(format!("the audio is `audio/pcm`, 16-bit little-endian, not `{t}`"));
            }
        }
        if let Some(rate) = format["rate"].as_u64() {
            if !(8_000..=48_000).contains(&rate) {
                return Err(format!("a rate from 8000 to 48000, not {rate}"));
            }
            next.rate = rate as usize;
        } else if s["input_audio_format"].as_str() == Some("pcm16") {
            // The beta's pcm16 was 24 kHz, and said no rate.
            next.rate = 24_000;
        }
        let t = if input["transcription"].is_object() { &input["transcription"] } else { &s["input_audio_transcription"] };
        if let Some(m) = t["model"].as_str() {
            next.model = Some(m.to_string());
        }
        if let Some(l) = t["language"].as_str().or_else(|| t["languages"][0].as_str()) {
            next.request.language = Some(l.trim().to_ascii_lowercase());
        }
        if let Some(p) = t["prompt"].as_str() {
            next.request.prompt = Some(p.to_string());
        }
        *self = next;
        Ok(())
    }
}

/// A session's socket: axum's WebSocket, or a test's pair of channels.
trait Link: Send {
    fn recv(&mut self) -> impl Future<Output = Option<Result<Message, axum::Error>>> + Send;
    fn send(&mut self, m: Message) -> impl Future<Output = Result<(), axum::Error>> + Send;
}

impl Link for WebSocket {
    fn recv(&mut self) -> impl Future<Output = Option<Result<Message, axum::Error>>> + Send {
        WebSocket::recv(self)
    }
    fn send(&mut self, m: Message) -> impl Future<Output = Result<(), axum::Error>> + Send {
        WebSocket::send(self, m)
    }
}

/// The server's end of a session that cannot go on: a close, said, and
/// the client's answer waited for a moment. Closed with what the client is
/// still sending unread, the connection would be reset instead.
async fn end<L: Link>(link: &mut L) -> Result<(), axum::Error> {
    link.send(Message::Close(Some(axum::extract::ws::CloseFrame { code: 1000, reason: "".into() }))).await?;
    let _ = tokio::time::timeout(Duration::from_secs(2), async { while let Some(Ok(_)) = link.recv().await {} }).await;
    Ok(())
}

/// Where a session's updates run: the model's engine, or a test's
/// stand-in for it.
trait Ears: Send {
    /// Find the model the session names, the first time: why there is
    /// none.
    fn find<'a>(&'a mut self, model: Option<&'a str>) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;
    /// One update: `audio` pushed to `live`, a new session when there is
    /// none, and the window decoded again; with `finish`, all of it settled.
    fn listen(&self, live: Option<Box<Live>>, request: Request, audio: Vec<f32>, finish: bool) -> Pending;
}

/// A session's updates as jobs on the engine of the resident that has its
/// model, so other work runs between them.
struct Engine {
    state: State,
    on: Option<Key>,
}

impl Ears for Engine {
    fn find<'a>(&'a mut self, model: Option<&'a str>) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>> {
        Box::pin(async move {
            if self.on.is_none() {
                let r = crate::openai::resident_for(&self.state, model, Kind::Speech).await.map_err(|Fail(_, why, _)| why)?;
                self.on = Some(r.key);
            }
            Ok(())
        })
    }

    fn listen(&self, live: Option<Box<Live>>, request: Request, audio: Vec<f32>, finish: bool) -> Pending {
        let (engine, on) = (self.state.engine.clone(), self.on.clone().expect("found before the first update"));
        Box::pin(async move { engine.listen(&on, live, request, audio, finish).await })
    }
}

/// A reconnect's request for a session that is on a socket: where to send
/// it.
type Handover = oneshot::Sender<Box<Session>>;

/// How long a reconnect waits for a session to come off the socket it is
/// on: the update under way finishes first.
const HANDOVER: Duration = Duration::from_secs(10);

/// The sessions there are, by id: on a socket, or held after their socket
/// went, for [`HOLD`], for a reconnect to take up.
pub struct Held {
    hold: Duration,
    slots: Mutex<HashMap<String, Slot>>,
    /// Tells one hold of a session from a later one, so that the first's
    /// timer does not end the second.
    holds: AtomicU64,
}

enum Slot {
    /// On a socket, which may have gone without the server hearing it: a
    /// phone that changes networks leaves its old connection open here
    /// until TCP gives up on it. A reconnect asks for the session here.
    On { owner: String, ask: mpsc::Sender<Handover> },
    /// Its socket gone, since `since`.
    Off { owner: String, session: Box<Session>, since: Instant, n: u64 },
}

impl Held {
    fn new(hold: Duration) -> Held {
        Held { hold, slots: Mutex::new(HashMap::new()), holds: AtomicU64::new(0) }
    }

    /// Session `id` is on a socket, and is asked for through `ask`.
    fn on(&self, id: &str, owner: &str, ask: mpsc::Sender<Handover>) {
        self.slots.lock().unwrap().insert(id.to_string(), Slot::On { owner: owner.to_string(), ask });
    }

    /// Session `id`, for `owner` reconnecting: out of its hold, or off the
    /// socket it is still on. Nothing for an id that is not held, another
    /// owner's, or one held longer than the hold.
    async fn take(&self, id: &str, owner: &str) -> Option<Box<Session>> {
        let handed = {
            let mut slots = self.slots.lock().unwrap();
            match slots.get(id)? {
                Slot::Off { owner: o, .. } if o == owner => {
                    let Some(Slot::Off { session, since, .. }) = slots.remove(id) else { unreachable!("just matched") };
                    return (since.elapsed() <= self.hold).then_some(session);
                }
                Slot::On { owner: o, ask } if o == owner => {
                    let (tx, rx) = oneshot::channel();
                    ask.try_send(tx).ok()?;
                    rx
                }
                _ => return None,
            }
        };
        tokio::time::timeout(HANDOVER, handed).await.ok()?.ok()
    }

    /// Hold `s`, its socket gone, or hand it to a reconnect that asked for
    /// it meanwhile. A reconnect asks with the slots locked, so it either
    /// asked already or finds the session held.
    fn park(self: &Arc<Self>, s: Box<Session>, owner: String, ask: &mut mpsc::Receiver<Handover>) {
        let mut slots = self.slots.lock().unwrap();
        let mut s = s;
        while let Ok(reply) = ask.try_recv() {
            match reply.send(s) {
                Ok(()) => return,
                Err(back) => s = back,
            }
        }
        let (id, n) = (s.id.clone(), self.holds.fetch_add(1, Ordering::Relaxed));
        tracing::debug!("realtime session {id} held for {} s", self.hold.as_secs());
        slots.insert(id.clone(), Slot::Off { owner, session: s, since: Instant::now(), n });
        let held = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(held.hold).await;
            let mut slots = held.slots.lock().unwrap();
            if matches!(slots.get(&id), Some(Slot::Off { n: m, .. }) if *m == n) {
                slots.remove(&id);
                tracing::debug!("realtime session {id} let go of: nobody came back for it");
            }
        });
    }

    /// Session `id` has ended: nothing of it is kept, and a reconnect that
    /// asked for it meanwhile is told so.
    fn forget(&self, id: &str, ask: mpsc::Receiver<Handover>) {
        let mut slots = self.slots.lock().unwrap();
        slots.remove(id);
        drop(ask);
    }
}

/// Every session in the server.
static HELD: LazyLock<Arc<Held>> = LazyLock::new(|| Arc::new(Held::new(HOLD)));

/// How a session's time on a socket ended.
enum Gone {
    /// The client closed it, or it could not go on: nothing is held.
    Ended,
    /// The socket went without a close: held for a reconnect.
    Dropped,
    /// A reconnect asked for it.
    Taken(Handover),
}

/// A socket, from its upgrade to its end: a new session, or the one a
/// reconnect names, picked up where it was.
async fn connected<L: Link>(held: Arc<Held>, owner: String, resume: Option<(String, u64)>, fresh: impl FnOnce() -> Box<Session>, mut link: L) {
    let (mut s, hello) = match resume {
        None => {
            let mut s = fresh();
            let hello = s.hello(None);
            (s, hello)
        }
        Some((id, from)) => match held.take(&id, &owner).await {
            Some(mut s) => {
                tracing::debug!("realtime session {id} taken up again from sample {from}");
                let hello = s.hello(Some(from));
                (s, hello)
            }
            None => {
                let why = format!(
                    "no session `{id}` is held here: it ended, or its socket went more than {} s ago, or there never was one. \
                     What was recorded can be sent whole to POST /v1/audio/transcriptions",
                    held.hold.as_secs()
                );
                let m = json!({ "type": "error", "event_id": "event_1", "error": { "type": "invalid_request_error", "code": "session_not_found", "message": why, "param": "resume" } });
                if link.send(Message::Text(m.to_string().into())).await.is_ok() {
                    let _ = end(&mut link).await;
                }
                return;
            }
        },
    };
    let (tell, mut ask) = mpsc::channel(1);
    held.on(&s.id, &owner, tell);
    match s.run(&mut link, &mut ask, hello).await {
        Gone::Ended => held.forget(&s.id, ask),
        Gone::Dropped => match s.settle().await {
            true => held.park(s, owner, &mut ask),
            false => held.forget(&s.id, ask),
        },
        Gone::Taken(reply) => {
            let goes_on = s.settle().await;
            let m = s.error("session_taken", "a reconnect has taken this session over");
            match goes_on {
                true => {
                    if let Err(s) = reply.send(s) {
                        // The reconnect stopped waiting.
                        held.park(s, owner, &mut ask);
                    }
                }
                false => held.forget(&s.id, ask),
            }
            let _ = tokio::time::timeout(Duration::from_secs(2), async {
                if link.send(m).await.is_ok() {
                    let _ = end(&mut link).await;
                }
            })
            .await;
        }
    }
}

struct Session {
    /// What a reconnect names it by.
    id: String,
    ears: Box<dyn Ears>,
    /// How often an update falls due.
    cadence: Duration,
    setup: Setup,
    rate: Rate,
    /// The live session, between updates; with the engine during one.
    live: Option<Box<Live>>,
    /// Audio come since the last update, at 16 kHz.
    audio: Vec<f32>,
    /// Seconds of audio received since the session began, at 16 kHz.
    received: f32,
    /// Samples received since the session began, at the session's rate:
    /// where the audio that follows on goes.
    offset: u64,
    /// Where the next piece of audio begins, as the client says it, or
    /// following on from the last.
    next: u64,
    /// Whether any audio has come: the session can no longer be changed.
    begun: bool,
    /// The update under way, and whether it is a commit's.
    pending: Option<(Pending, bool)>,
    /// A commit asked for while an update was under way.
    commit: bool,
    item: u64,
    events: u64,
    /// This item's settled text.
    said: String,
    /// Every settled event sent, with the offset its update had heard to,
    /// to be sent again to a reconnect that may not have had it.
    log: Vec<(u64, Message)>,
    beta: bool,
}

impl Session {
    fn new(ears: Box<dyn Ears>, model: Option<String>) -> Box<Session> {
        Box::new(Session {
            id: format!("sess_{}", &crate::secret::token()[..32]),
            ears,
            cadence: CADENCE,
            setup: Setup { model, request: Request::default(), rate: SAMPLE_RATE },
            rate: Rate::new(SAMPLE_RATE),
            live: None,
            audio: Vec::new(),
            received: 0.0,
            offset: 0,
            next: 0,
            begun: false,
            pending: None,
            commit: false,
            item: 1,
            events: 0,
            said: String::new(),
            log: Vec::new(),
            beta: false,
        })
    }

    fn event(&mut self, kind: &str, mut body: Value) -> Message {
        self.events += 1;
        body["type"] = kind.into();
        body["event_id"] = format!("event_{}", self.events).into();
        Message::Text(body.to_string().into())
    }

    fn item_id(&self) -> String {
        format!("item_{}", self.item)
    }

    fn described(&self) -> Value {
        json!({
            "id": self.id,
            "type": "transcription",
            "audio": { "input": {
                "format": { "type": "audio/pcm", "rate": self.setup.rate },
                "transcription": { "model": self.setup.model, "language": self.setup.request.language, "prompt": self.setup.request.prompt },
                "turn_detection": null,
            }},
        })
    }

    fn error(&mut self, code: &str, message: impl Into<String>) -> Message {
        self.event("error", json!({ "error": { "type": "invalid_request_error", "code": code, "message": message.into(), "param": null } }))
    }

    /// What a socket is told on connecting: `session.created`. A reconnect
    /// that has everything settled before sample `from` is told where the
    /// session stands, and then sent everything settled from there on
    /// again, each event as it was, its `event_id` too.
    fn hello(&mut self, resumed: Option<u64>) -> Vec<Message> {
        let mut body = json!({ "session": self.described() });
        let Some(from) = resumed else { return vec![self.event("session.created", body)] };
        body["kvad"] = json!({ "resumed": true, "offset": self.offset, "item_id": self.item_id(), "transcript": self.said });
        // Audio that comes without an offset follows on from what is here.
        self.next = self.offset;
        self.log.retain(|(at, _)| *at >= from);
        let mut out = vec![self.event("session.created", body)];
        out.extend(self.log.iter().map(|(_, m)| m.clone()));
        out
    }

    async fn run<L: Link>(&mut self, link: &mut L, ask: &mut mpsc::Receiver<Handover>, hello: Vec<Message>) -> Gone {
        for m in hello {
            if link.send(m).await.is_err() {
                return Gone::Dropped;
            }
        }
        let mut tick = tokio::time::interval(self.cadence);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let mut out = Vec::new();
            let mut ended = false;
            tokio::select! {
                msg = link.recv() => {
                    let Some(Ok(msg)) = msg else { return Gone::Dropped };
                    out = match msg {
                        Message::Binary(pcm) => self.heard(&pcm, None),
                        Message::Text(text) => self.told(&text),
                        Message::Close(_) => {
                            // Read on until the socket ends: that is what
                            // sends the answer to the client's close.
                            while let Some(Ok(_)) = link.recv().await {}
                            return Gone::Ended;
                        }
                        _ => Vec::new(),
                    };
                }
                Some(reply) = ask.recv() => return Gone::Taken(reply),
                _ = tick.tick() => {
                    if self.pending.is_none() && (!self.audio.is_empty() || self.commit) {
                        if let Err(m) = self.begin().await {
                            (out, ended) = (vec![m], true);
                        }
                    }
                }
                listened = async { (&mut self.pending.as_mut().expect("guarded").0).await }, if self.pending.is_some() => {
                    let (_, finished) = self.pending.take().expect("guarded");
                    (out, ended) = self.listened(listened, finished);
                    // A commit that came during an update is run at once,
                    // and so is a backlog's next piece.
                    if !ended && (self.commit || self.audio.len() >= MOST_AT_ONCE) {
                        if let Err(m) = self.begin().await {
                            out.push(m);
                            ended = true;
                        }
                    }
                }
            }
            if !ended && self.audio.len() as f32 / SAMPLE_RATE as f32 > MOST_BEHIND {
                out.push(self.error("too_far_behind", format!("this session is more than {MOST_BEHIND} s of audio behind; the server is busy")));
                ended = true;
            }
            for m in out {
                if link.send(m).await.is_err() {
                    return if ended { Gone::Ended } else { Gone::Dropped };
                }
            }
            if ended {
                let _ = end(link).await;
                return Gone::Ended;
            }
        }
    }

    /// Wait out the update under way, the socket gone: what it settles is
    /// kept for the reconnect. Whether the session can go on.
    async fn settle(&mut self) -> bool {
        let Some((pending, finished)) = self.pending.take() else { return true };
        let listened = pending.await;
        !self.listened(listened, finished).1
    }

    /// Audio: `pcm`, beginning at sample `at` of the session, or where the
    /// last piece ended. What the session already has, as resent after a
    /// reconnect, is passed over.
    fn heard(&mut self, pcm: &[u8], at: Option<u64>) -> Vec<Message> {
        if pcm.len() % 2 != 0 {
            return vec![self.error("invalid_audio", "audio is 16-bit PCM: a frame of an odd number of bytes is not")];
        }
        let at = at.unwrap_or(self.next);
        let n = pcm.len() as u64 / 2;
        self.next = at + n;
        if at > self.offset {
            let has = self.offset;
            return vec![self.error("audio_gap", format!("audio from sample {at}, but this session has {has}: what comes between is missing; send from {has}"))];
        }
        let pcm = &pcm[2 * (self.offset - at).min(n) as usize..];
        if pcm.is_empty() {
            return Vec::new();
        }
        self.offset += pcm.len() as u64 / 2;
        let before = self.audio.len();
        self.rate.feed(pcm, &mut self.audio);
        self.received += (self.audio.len() - before) as f32 / SAMPLE_RATE as f32;
        self.begun = true;
        Vec::new()
    }

    /// A text frame: an event.
    fn told(&mut self, text: &str) -> Vec<Message> {
        let Ok(v) = serde_json::from_str::<Value>(text) else {
            return vec![self.error("invalid_json", "an event is a JSON object")];
        };
        match v["type"].as_str().unwrap_or("") {
            "input_audio_buffer.append" => match v["audio"].as_str().map(|a| base64::engine::general_purpose::STANDARD.decode(a)) {
                Some(Ok(pcm)) => self.heard(&pcm, v["kvad"]["offset"].as_u64()),
                _ => vec![self.error("invalid_audio", "input_audio_buffer.append carries `audio`, base64 of 16-bit PCM")],
            },
            "kvad.input_audio_buffer.offset" => match v["offset"].as_u64() {
                Some(at) => {
                    self.next = at;
                    Vec::new()
                }
                None => vec![self.error("invalid_offset", "kvad.input_audio_buffer.offset carries `offset`, the sample the next binary frame begins at")],
            },
            "input_audio_buffer.commit" => {
                self.commit = true;
                Vec::new()
            }
            kind @ ("session.update" | "transcription_session.update") => {
                self.beta = kind.starts_with("transcription_session");
                match self.configure(&v["session"]) {
                    Ok(()) => {
                        let described = self.described();
                        vec![self.event(if self.beta { "transcription_session.updated" } else { "session.updated" }, json!({ "session": described }))]
                    }
                    Err(why) => vec![self.error("invalid_session", why)],
                }
            }
            other => vec![self.error(
                "unknown_event",
                format!(
                    "`{other}` is not an event this server takes; a transcription session takes session.update, \
                     input_audio_buffer.append, input_audio_buffer.commit and kvad.input_audio_buffer.offset"
                ),
            )],
        }
    }

    fn configure(&mut self, s: &Value) -> Result<(), String> {
        if self.begun {
            return Err("the session is changed before its first audio, not after".into());
        }
        self.setup.read(s)?;
        self.rate = Rate::new(self.setup.rate);
        Ok(())
    }

    /// Send the audio come since the last update to the engine, up to
    /// `MOST_AT_ONCE` of it, and the window decoded again; with a commit and
    /// nothing left waiting, everything settled. An error, which ends the
    /// session, if there is no model to do it.
    async fn begin(&mut self) -> Result<(), Message> {
        if let Err(why) = self.ears.find(self.setup.model.as_deref()).await {
            return Err(self.error("model_not_available", why));
        }
        let audio: Vec<f32> = self.audio.drain(..self.audio.len().min(MOST_AT_ONCE)).collect();
        // A commit finishes the item once the audio before it is all heard.
        let finish = self.commit && self.audio.is_empty();
        self.commit &= !finish;
        let pending = self.ears.listen(self.live.take(), self.setup.request.clone(), audio, finish);
        self.pending = Some((pending, finish));
        Ok(())
    }

    /// What an update said, as events, and whether the session must end.
    /// The settled ones are logged, for a reconnect.
    fn listened(&mut self, listened: Listened, finished: bool) -> (Vec<Message>, bool) {
        let mut out = Vec::new();
        let (live, u): (Box<Live>, Update) = match listened {
            Listened::Heard(live, u) => (live, u),
            Listened::Refused(why) => return (vec![self.error("invalid_request", why)], true),
            Listened::Failed(why) => return (vec![self.event("error", json!({ "error": { "type": "server_error", "code": "failed", "message": why } }))], true),
        };
        self.live = Some(live);
        let lag = (self.received - u.heard).max(0.0);
        // The sample the update had heard to, at the session's rate.
        let offset = ((u.heard as f64 * self.setup.rate as f64) as u64).min(self.offset);
        let item = self.item_id();
        let kvad = json!({ "heard": u.heard, "lag": lag, "offset": offset });
        if !u.settled.is_empty() {
            self.said.push_str(&u.settled);
            out.push(self.event(
                "conversation.item.input_audio_transcription.delta",
                json!({ "item_id": item, "content_index": 0, "delta": u.settled, "kvad": kvad }),
            ));
        }
        if !finished {
            self.log.extend(out.iter().map(|m| (offset, m.clone())));
            out.push(self.event("kvad.transcription.partial", json!({ "item_id": item, "text": u.partial, "kvad": kvad })));
            return (out, false);
        }
        let transcript = std::mem::take(&mut self.said).trim().to_string();
        let language = self.live.as_ref().and_then(|l| l.language()).map(str::to_string);
        let previous = (self.item > 1).then(|| format!("item_{}", self.item - 1));
        out.insert(0, self.event("input_audio_buffer.committed", json!({ "item_id": item, "previous_item_id": previous, "kvad": { "offset": offset } })));
        out.push(self.event(
            "conversation.item.input_audio_transcription.completed",
            json!({ "item_id": item, "content_index": 0, "transcript": transcript, "kvad": { "language": language, "heard": u.heard, "offset": offset } }),
        ));
        self.log.extend(out.iter().map(|m| (offset, m.clone())));
        self.item += 1;
        (out, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pcm(x: &[f32]) -> Vec<u8> {
        x.iter().flat_map(|&v| ((v * 32767.0) as i16).to_le_bytes()).collect()
    }

    fn setup() -> Setup {
        Setup { model: None, request: Request::default(), rate: SAMPLE_RATE }
    }

    /// OpenAI's session, and its beta's, set the same things.
    #[test]
    fn a_session_is_set_up_in_either_shape() {
        let mut ga = setup();
        let update = json!({ "type": "transcription", "audio": { "input": {
            "format": { "type": "audio/pcm", "rate": 24000 },
            "transcription": { "model": "NbAiLab/nb-whisper-small", "languages": ["NO", "en"], "prompt": "Kygo" },
        }}});
        ga.read(&update).unwrap();
        assert_eq!(ga.model.as_deref(), Some("NbAiLab/nb-whisper-small"));
        assert_eq!(ga.request.language.as_deref(), Some("no"));
        assert_eq!(ga.request.prompt.as_deref(), Some("Kygo"));
        assert_eq!(ga.rate, 24_000);

        let mut beta = setup();
        let update = json!({ "input_audio_format": "pcm16", "input_audio_transcription": { "model": "NbAiLab/nb-whisper-small", "language": "no", "prompt": "Kygo" } });
        beta.read(&update).unwrap();
        assert_eq!(beta, ga);

        // Nothing said, nothing changed: 16 kHz, the server's own model.
        let mut bare = setup();
        bare.read(&json!({})).unwrap();
        assert_eq!(bare, setup());
    }

    #[test]
    fn what_cannot_be_heard_is_refused_and_changes_nothing() {
        let mut s = setup();
        for (update, says) in [
            (json!({ "type": "realtime" }), "transcription"),
            (json!({ "audio": { "input": { "format": { "type": "audio/pcmu" } } } }), "audio/pcmu"),
            (json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 96000 }, "transcription": { "model": "x" } } } }), "96000"),
        ] {
            assert!(s.read(&update).unwrap_err().contains(says), "{update}");
            assert_eq!(s, setup());
        }
    }

    /// 16 kHz goes through as it came; 24 kHz comes out two samples for
    /// three, the same whether it arrives whole or in odd pieces.
    #[test]
    fn audio_is_read_at_its_rate_in_any_pieces() {
        let x: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let mut same = Vec::new();
        Rate::new(16_000).feed(&pcm(&x), &mut same);
        assert_eq!(same.len(), x.len());
        assert!(same.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-4));

        let mut whole = Vec::new();
        Rate::new(24_000).feed(&pcm(&x), &mut whole);
        assert_eq!(whole.len(), 1600);
        let mut pieces = Vec::new();
        let mut r = Rate::new(24_000);
        for chunk in pcm(&x).chunks(2 * 37) {
            r.feed(chunk, &mut pieces);
        }
        assert_eq!(pieces, whole);
        // The sample at 16 kHz index k is the 24 kHz one at 1.5 k.
        assert!((whole[100] - x[150]).abs() < 1e-3);
        assert!((whole[101] - 0.5 * (x[151] + x[152])).abs() < 1e-3);
    }

    use kvad::speech::live::script::{script_text, sound, Script};
    use kvad::speech::Transcriber;

    /// The scripted Whisper as a session's engine, each update `slow` late.
    struct Scripted {
        tr: Arc<Transcriber>,
        slow: Duration,
    }

    impl Ears for Scripted {
        fn find<'a>(&'a mut self, _: Option<&'a str>) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }

        fn listen(&self, live: Option<Box<Live>>, request: Request, audio: Vec<f32>, finish: bool) -> Pending {
            let (tr, slow) = (self.tr.clone(), self.slow);
            Box::pin(async move {
                tokio::time::sleep(slow).await;
                let heard = tokio::task::spawn_blocking(move || {
                    let mut live = match live {
                        Some(l) => l,
                        None => Box::new(Live::new(&tr, &request).map_err(|e| e.to_string())?),
                    };
                    live.push(&audio);
                    let u = if finish { live.finish(&tr) } else { live.update(&tr) };
                    Ok::<_, String>((live, u.map_err(|e| e.to_string())?))
                });
                match heard.await.expect("the update ran") {
                    Ok((live, u)) => Listened::Heard(live, u),
                    Err(why) => Listened::Failed(why),
                }
            })
        }
    }

    /// A socket that is a pair of channels.
    struct Pipe {
        from: mpsc::UnboundedReceiver<Message>,
        to: mpsc::UnboundedSender<Message>,
    }

    impl Link for Pipe {
        async fn recv(&mut self) -> Option<Result<Message, axum::Error>> {
            self.from.recv().await.map(Ok)
        }
        async fn send(&mut self, m: Message) -> Result<(), axum::Error> {
            self.to.send(m).map_err(|_| axum::Error::new("the client has gone"))
        }
    }

    /// The phone's end: a recording on the phone, and what came back.
    struct Client {
        to: mpsc::UnboundedSender<Message>,
        from: mpsc::UnboundedReceiver<Message>,
        events: Vec<Value>,
    }

    /// 40 s of the script, as the phone recorded it.
    fn recording() -> Vec<u8> {
        pcm(&sound(40.0))
    }

    fn connect(held: &Arc<Held>, tr: &Arc<Transcriber>, slow: Duration, resume: Option<(String, u64)>) -> Client {
        let (to, from_client) = mpsc::unbounded_channel();
        let (to_client, from) = mpsc::unbounded_channel();
        let ears = Scripted { tr: tr.clone(), slow };
        let fresh = move || {
            let mut s = Session::new(Box::new(ears), None);
            s.cadence = Duration::from_millis(5);
            s
        };
        tokio::spawn(connected(held.clone(), "me".into(), resume, fresh, Pipe { from: from_client, to: to_client }));
        Client { to, from, events: Vec::new() }
    }

    impl Client {
        /// Samples `a` to `b` of the recording, in 100 ms frames.
        fn play(&self, rec: &[u8], a: usize, b: usize) {
            for chunk in rec[2 * a..2 * b].chunks(2 * SAMPLE_RATE / 10) {
                self.to.send(Message::Binary(chunk.to_vec().into())).unwrap();
            }
        }

        fn say(&self, event: Value) {
            self.to.send(Message::Text(event.to_string().into())).unwrap();
        }

        /// The next event, kept.
        async fn next(&mut self) -> Option<Value> {
            loop {
                match tokio::time::timeout(Duration::from_secs(20), self.from.recv()).await.expect("an event in time")? {
                    Message::Text(t) => {
                        let v: Value = serde_json::from_str(&t).unwrap();
                        self.events.push(v.clone());
                        return Some(v);
                    }
                    Message::Close(_) => return None,
                    _ => {}
                }
            }
        }

        /// Events until one of `kind`.
        async fn until(&mut self, kind: &str) -> Value {
            loop {
                let v = self.next().await.unwrap_or_else(|| panic!("closed before {kind}"));
                assert_ne!(v["type"], "error", "{v}");
                if v["type"] == kind {
                    return v;
                }
            }
        }

        /// Events until an update has heard `seconds`.
        async fn heard(&mut self, seconds: f32) {
            loop {
                let v = self.next().await.expect("open");
                assert_ne!(v["type"], "error", "{v}");
                if v["kvad"]["heard"].as_f64().is_some_and(|h| h as f32 >= seconds) {
                    return;
                }
            }
        }

        /// The sample of the last settled event this end has had.
        fn settled_to(&self) -> u64 {
            self.events.iter().filter(|v| v["type"] != "kvad.transcription.partial").filter_map(|v| v["kvad"]["offset"].as_u64()).last().unwrap_or(0)
        }
    }

    /// The deltas the phone has, each event once however often it came,
    /// put together; and the item's `completed` transcript.
    fn transcript(events: &[Value]) -> (String, String) {
        let mut seen = std::collections::HashSet::new();
        let (mut deltas, mut completed) = (String::new(), String::new());
        for v in events {
            if !seen.insert(v["event_id"].as_str().unwrap().to_string()) {
                continue;
            }
            match v["type"].as_str().unwrap() {
                "conversation.item.input_audio_transcription.delta" => deltas.push_str(v["delta"].as_str().unwrap()),
                "conversation.item.input_audio_transcription.completed" => completed = v["transcript"].as_str().unwrap().to_string(),
                _ => {}
            }
        }
        (deltas, completed)
    }

    const S: usize = SAMPLE_RATE;

    /// 40 s played a second at a time, each waited for, never cut.
    async fn never_cut(held: &Arc<Held>, tr: &Arc<Transcriber>, rec: &[u8]) -> Vec<Value> {
        let mut c = connect(held, tr, Duration::ZERO, None);
        c.until("session.created").await;
        for second in 1..=40 {
            c.play(rec, (second - 1) * S, second * S);
            c.heard(second as f32).await;
        }
        c.say(json!({ "type": "input_audio_buffer.commit" }));
        c.until("conversation.item.input_audio_transcription.completed").await;
        c.events
    }

    /// The link cut mid-sentence, at 12.5 s, and what was sent after the
    /// last settled event lost; the phone reconnects, names the session and
    /// where it had text to, and plays its recording from there. What it
    /// ends with is what a session never cut gives, each word once.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_link_cut_mid_sentence_and_resumed_gives_the_same_text() {
        let (held, tr, rec) = (Arc::new(Held::new(HOLD)), Arc::new(Script::transcriber()), recording());
        let (deltas, whole) = transcript(&never_cut(&held, &tr, &rec).await);
        assert_eq!(whole, script_text(40).trim());
        assert_eq!(deltas.trim(), whole);

        let mut c = connect(&held, &tr, Duration::ZERO, None);
        let id = c.until("session.created").await["session"]["id"].as_str().unwrap().to_string();
        for second in 1..=12 {
            c.play(&rec, (second - 1) * S, second * S);
            c.heard(second as f32).await;
        }
        // Half a word more, and the link goes: what the server had read of
        // it is kept, the rest is lost in flight, and the phone hears
        // nothing of what it settles.
        c.play(&rec, 12 * S, 12 * S + S / 2);
        let (from, before) = (c.settled_to(), c.events);
        assert!(from > 0 && from <= 12 * S as u64, "settled to {from}");
        drop((c.to, c.from));

        let mut c = connect(&held, &tr, Duration::ZERO, Some((id.clone(), from)));
        let created = c.until("session.created").await;
        assert_eq!(created["session"]["id"], id.as_str());
        assert_eq!(created["kvad"]["resumed"], true);
        let has = created["kvad"]["offset"].as_u64().unwrap();
        assert!(has >= 12 * S as u64 && has <= (12 * S + S / 2) as u64, "the server says what it has: {has}");
        // The phone resends from where it had text: what the server has is
        // passed over, not heard twice.
        c.say(json!({ "type": "kvad.input_audio_buffer.offset", "offset": from }));
        c.play(&rec, from as usize, 14 * S);
        c.heard(14.0).await;
        for second in 15..=40 {
            c.play(&rec, (second - 1) * S, second * S);
            c.heard(second as f32).await;
        }
        c.say(json!({ "type": "input_audio_buffer.commit" }));
        let completed = c.until("conversation.item.input_audio_transcription.completed").await;
        assert_eq!(completed["kvad"]["heard"], 40.0, "every sample heard once");

        // What settled at the offset the phone named came again, as it was.
        let had: std::collections::HashSet<&Value> = before.iter().map(|v| &v["event_id"]).collect();
        assert!(c.events.iter().any(|v| v["type"] != "session.created" && had.contains(&v["event_id"])), "nothing sent again");
        let events: Vec<Value> = before.into_iter().chain(c.events).collect();
        let (resumed_deltas, resumed) = transcript(&events);
        assert_eq!(resumed, whole);
        assert_eq!(resumed_deltas.trim(), whole, "the deltas, each once, are the transcript");
    }

    /// An update that takes longer than the cadence: the ones falling due
    /// meanwhile are skipped, but every second of audio is heard, and the
    /// text is what a decoder that keeps up writes. A second of audio comes
    /// every 40 ms, and an update takes 100 ms and more, so each hears 3 s
    /// or so, and on a slow machine up to `MOST_AT_ONCE`, never more: a
    /// window handed 30 s at once would begin again, and the script hears a
    /// window with no padding as silence. On a busy CI runner, before
    /// `MOST_AT_ONCE`, this one kept 10 words of 40.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_slow_decoder_skips_partials_but_never_audio() {
        let (held, tr, rec) = (Arc::new(Held::new(HOLD)), Arc::new(Script::transcriber()), recording());
        let mut c = connect(&held, &tr, Duration::from_millis(100), None);
        c.until("session.created").await;
        for second in 1..=40 {
            c.play(&rec, (second - 1) * S, second * S);
            tokio::time::sleep(Duration::from_millis(40)).await;
        }
        c.say(json!({ "type": "input_audio_buffer.commit" }));
        let completed = c.until("conversation.item.input_audio_transcription.completed").await;
        assert_eq!(completed["kvad"]["heard"], 40.0);
        let partials = c.events.iter().filter(|v| v["type"] == "kvad.transcription.partial").count();
        assert!(partials < 20, "{partials} updates in 40 s of audio; most were skipped");
        let (deltas, whole) = transcript(&c.events);
        assert_eq!(whole, script_text(40).trim());
        assert_eq!(deltas.trim(), whole);
    }

    /// A backlog, as a busy server or a slow decoder makes: all 40 s come
    /// while the first update is under way. The next updates hear it a few
    /// seconds at a time, and every word is in the transcript.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_backlog_is_heard_a_few_seconds_at_a_time() {
        let (held, tr, rec) = (Arc::new(Held::new(HOLD)), Arc::new(Script::transcriber()), recording());
        let mut c = connect(&held, &tr, Duration::from_millis(300), None);
        c.until("session.created").await;
        c.play(&rec, 0, 40 * S);
        c.say(json!({ "type": "input_audio_buffer.commit" }));
        let completed = c.until("conversation.item.input_audio_transcription.completed").await;
        assert_eq!(completed["kvad"]["heard"], 40.0);
        let (deltas, whole) = transcript(&c.events);
        assert_eq!(whole, script_text(40).trim());
        assert_eq!(deltas.trim(), whole);
    }

    /// A reconnect while the server still thinks the old socket is open
    /// takes the session over; the old socket is told and closed.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_reconnect_takes_a_session_from_a_socket_not_yet_gone() {
        let (held, tr, rec) = (Arc::new(Held::new(HOLD)), Arc::new(Script::transcriber()), recording());
        let mut old = connect(&held, &tr, Duration::ZERO, None);
        let id = old.until("session.created").await["session"]["id"].as_str().unwrap().to_string();
        old.play(&rec, 0, 3 * S);
        old.heard(3.0).await;

        let mut c = connect(&held, &tr, Duration::ZERO, Some((id, 0)));
        let created = c.until("session.created").await;
        assert_eq!(created["kvad"]["offset"], 3 * S);
        // After whatever of the last update it had not read yet.
        let told = loop {
            let v = old.next().await.expect("told before the close");
            if v["type"] == "error" {
                break v;
            }
        };
        assert_eq!(told["error"]["code"], "session_taken", "{told}");
        assert!(old.next().await.is_none(), "the old socket is closed");

        c.play(&rec, 3 * S, 5 * S);
        c.say(json!({ "type": "input_audio_buffer.commit" }));
        let completed = c.until("conversation.item.input_audio_transcription.completed").await;
        assert_eq!(completed["transcript"], script_text(5).trim());
    }

    /// After the hold, or for an id never given, or for another's session,
    /// the answer is `session_not_found`, and the socket closes: the phone
    /// sends its file instead.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_session_not_held_is_refused() {
        let (held, tr, rec) = (Arc::new(Held::new(Duration::from_millis(100))), Arc::new(Script::transcriber()), recording());
        let refused = |mut c: Client| async move {
            let v = c.next().await.unwrap();
            assert_eq!(v["error"]["code"], "session_not_found", "{v}");
            assert!(v["error"]["message"].as_str().unwrap().contains("/v1/audio/transcriptions"));
            assert!(c.next().await.is_none(), "closed");
        };
        refused(connect(&held, &tr, Duration::ZERO, Some(("sess_nobody".into(), 0)))).await;

        let mut c = connect(&held, &tr, Duration::ZERO, None);
        let id = c.until("session.created").await["session"]["id"].as_str().unwrap().to_string();
        c.play(&rec, 0, 2 * S);
        c.heard(2.0).await;
        drop((c.to, c.from));
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Another's, within the hold: as if there were none; and it is
        // still held for its owner.
        let (to, from_client) = mpsc::unbounded_channel();
        let (to_client, from) = mpsc::unbounded_channel();
        let fresh = || -> Box<Session> { unreachable!("a resume makes no session") };
        tokio::spawn(connected(held.clone(), "someone else".into(), Some((id.clone(), 0)), fresh, Pipe { from: from_client, to: to_client }));
        refused(Client { to, from, events: Vec::new() }).await;

        tokio::time::sleep(Duration::from_millis(200)).await;
        refused(connect(&held, &tr, Duration::ZERO, Some((id, 0)))).await;
        assert!(held.slots.lock().unwrap().is_empty(), "nothing is kept after the hold");
    }

    /// Audio placed by its offset: what is had already is passed over, a
    /// gap is refused, and audio without an offset follows on.
    #[test]
    fn audio_is_placed_by_its_offset() {
        let ears = Scripted { tr: Arc::new(Script::transcriber()), slow: Duration::ZERO };
        let mut s = Session::new(Box::new(ears), None);
        let x: Vec<u8> = pcm(&(0..100).map(|i| i as f32 / 200.0).collect::<Vec<_>>());
        assert!(s.heard(&x[..120], None).is_empty());
        assert_eq!((s.offset, s.audio.len()), (60, 60));
        // Samples 40 to 100: 40 to 60 are had.
        assert!(s.heard(&x[80..], Some(40)).is_empty());
        assert_eq!((s.offset, s.audio.len()), (100, 100));
        let want: Vec<f32> = x.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
        assert_eq!(s.audio, want, "each sample once, in order");
        // All of it again: nothing new.
        assert!(s.heard(&x, Some(0)).is_empty());
        assert_eq!(s.offset, 100);
        // A frame from 120, with 100 to 120 missing.
        let gap = s.heard(&x[..20], Some(120));
        let Message::Text(t) = &gap[0] else { panic!() };
        assert!(t.contains("audio_gap") && t.contains("send from 100"), "{t}");
        assert_eq!(s.offset, 100);
        s.told(&json!({ "type": "kvad.input_audio_buffer.offset", "offset": 100 }).to_string());
        assert!(s.heard(&x[..20], None).is_empty());
        assert_eq!(s.offset, 110);
    }
}
