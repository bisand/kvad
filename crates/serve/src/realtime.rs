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
//! Out, besides `session.created` on connecting: `input_audio_buffer.committed`;
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
//! is a commit, its `completed`, and then the client's close. A client that
//! leaves without a commit leaves: nothing more is decoded, and nothing is
//! kept, neither the audio nor the text.

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
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

pub fn routes() -> Router<State> {
    Router::new().route("/v1/realtime", get(realtime))
}

/// How often an update falls due: what `live_speed` measured at.
const CADENCE: Duration = Duration::from_secs(1);

/// How far behind the audio a session may fall before it is ended: a
/// minute of audio waiting for an engine that is busy with something else.
const MOST_BEHIND: f32 = 60.0;

#[derive(serde::Deserialize)]
pub struct Asked {
    intent: Option<String>,
    model: Option<String>,
}

async fn realtime(_: Identity, headers: HeaderMap, Query(asked): Query<Asked>, St(state): St<State>, ws: WebSocketUpgrade) -> Result<Response, Fail> {
    if !crate::auth::socket_may_use(&headers) {
        return Err(Fail(StatusCode::FORBIDDEN, "a WebSocket opened on a session cookie must come from this server's own pages".into(), None));
    }
    match asked.intent.as_deref() {
        None | Some("transcription") => {}
        Some(other) => return Err(Fail::bad(format!("intent={other}: this server's realtime sessions are transcription only"))),
    }
    Ok(ws.on_upgrade(move |socket| async move {
        let mut s = Session::new(state, asked.model);
        if let Err(e) = s.run(socket).await {
            tracing::debug!("a realtime session ended: {e}");
        }
    }))
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

/// The server's end of a session that cannot go on: a close, said, and
/// the client's answer waited for a moment. Closed with what the client is
/// still sending unread, the connection would be reset instead.
async fn end(mut socket: WebSocket) -> Result<(), axum::Error> {
    socket.send(Message::Close(Some(axum::extract::ws::CloseFrame { code: 1000, reason: "".into() }))).await?;
    let _ = tokio::time::timeout(Duration::from_secs(2), async { while let Some(Ok(_)) = socket.recv().await {} }).await;
    Ok(())
}

struct Session {
    state: State,
    setup: Setup,
    rate: Rate,
    /// The resident the session runs on, once the first update has found it.
    on: Option<Key>,
    /// The live session, between updates; with the engine during one.
    live: Option<Box<Live>>,
    /// Audio come since the last update, at 16 kHz.
    audio: Vec<f32>,
    /// Seconds of audio received since the session began.
    received: f32,
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
    beta: bool,
}

impl Session {
    fn new(state: State, model: Option<String>) -> Session {
        Session {
            state,
            setup: Setup { model, request: Request::default(), rate: SAMPLE_RATE },
            rate: Rate::new(SAMPLE_RATE),
            on: None,
            live: None,
            audio: Vec::new(),
            received: 0.0,
            begun: false,
            pending: None,
            commit: false,
            item: 1,
            events: 0,
            said: String::new(),
            beta: false,
        }
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

    async fn run(&mut self, mut socket: WebSocket) -> Result<(), axum::Error> {
        let created = self.event("session.created", json!({ "session": self.described() }));
        socket.send(created).await?;
        let mut tick = tokio::time::interval(CADENCE);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                msg = socket.recv() => {
                    let Some(msg) = msg else { return Ok(()) };
                    let out = match msg? {
                        Message::Binary(pcm) => self.heard(&pcm),
                        Message::Text(text) => self.told(&text),
                        Message::Close(_) => {
                            // Read on until the socket ends: that is what
                            // sends the answer to the client's close.
                            while let Some(Ok(_)) = socket.recv().await {}
                            return Ok(());
                        }
                        _ => Vec::new(),
                    };
                    for m in out {
                        socket.send(m).await?;
                    }
                }
                _ = tick.tick() => {
                    if self.pending.is_none() && (!self.audio.is_empty() || self.commit) {
                        if let Err(m) = self.begin().await {
                            socket.send(m).await?;
                            return end(socket).await;
                        }
                    }
                }
                listened = async { (&mut self.pending.as_mut().expect("guarded").0).await }, if self.pending.is_some() => {
                    let (_, finished) = self.pending.take().expect("guarded");
                    let (out, fatal) = self.listened(listened, finished);
                    for m in out {
                        socket.send(m).await?;
                    }
                    if fatal {
                        return end(socket).await;
                    }
                    // A commit that came during an update is run at once.
                    if self.commit {
                        if let Err(m) = self.begin().await {
                            socket.send(m).await?;
                            return end(socket).await;
                        }
                    }
                }
            }
            if self.audio.len() as f32 / SAMPLE_RATE as f32 > MOST_BEHIND {
                let m = self.error("too_far_behind", format!("this session is more than {MOST_BEHIND} s of audio behind; the server is busy"));
                socket.send(m).await?;
                return end(socket).await;
            }
        }
    }

    /// A binary frame: audio.
    fn heard(&mut self, pcm: &[u8]) -> Vec<Message> {
        if pcm.len() % 2 != 0 {
            return vec![self.error("invalid_audio", "audio is 16-bit PCM: a frame of an odd number of bytes is not")];
        }
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
                Some(Ok(pcm)) => self.heard(&pcm),
                _ => vec![self.error("invalid_audio", "input_audio_buffer.append carries `audio`, base64 of 16-bit PCM")],
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
            other => vec![self.error("unknown_event", format!("`{other}` is not an event this server takes; a transcription session takes session.update, input_audio_buffer.append and input_audio_buffer.commit"))],
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

    /// Send the audio come since the last update to the engine, and the
    /// window decoded again, or with a commit, everything settled. An
    /// error, which ends the session, if there is no model to do it.
    async fn begin(&mut self) -> Result<(), Message> {
        let on = match &self.on {
            Some(on) => on.clone(),
            None => match crate::openai::resident_for(&self.state, self.setup.model.as_deref(), Kind::Speech).await {
                Ok(r) => {
                    self.on = Some(r.key.clone());
                    r.key
                }
                Err(Fail(_, why, _)) => return Err(self.error("model_not_available", why)),
            },
        };
        let finish = std::mem::take(&mut self.commit);
        let audio = std::mem::take(&mut self.audio);
        let (engine, live, request) = (self.state.engine.clone(), self.live.take(), self.setup.request.clone());
        self.pending = Some((Box::pin(async move { engine.listen(&on, live, request, audio, finish).await }), finish));
        Ok(())
    }

    /// What an update said, as events, and whether the session must end.
    fn listened(&mut self, listened: Listened, finished: bool) -> (Vec<Message>, bool) {
        let mut out = Vec::new();
        let (live, u): (Box<Live>, Update) = match listened {
            Listened::Heard(live, u) => (live, u),
            Listened::Refused(why) => return (vec![self.error("invalid_request", why)], true),
            Listened::Failed(why) => return (vec![self.event("error", json!({ "error": { "type": "server_error", "code": "failed", "message": why } }))], true),
        };
        self.live = Some(live);
        let lag = (self.received - u.heard).max(0.0);
        let item = self.item_id();
        let kvad = json!({ "heard": u.heard, "lag": lag });
        if !u.settled.is_empty() {
            self.said.push_str(&u.settled);
            out.push(self.event(
                "conversation.item.input_audio_transcription.delta",
                json!({ "item_id": item, "content_index": 0, "delta": u.settled, "kvad": kvad }),
            ));
        }
        if !finished {
            out.push(self.event("kvad.transcription.partial", json!({ "item_id": item, "text": u.partial, "kvad": kvad })));
            return (out, false);
        }
        let transcript = std::mem::take(&mut self.said).trim().to_string();
        let language = self.live.as_ref().and_then(|l| l.language()).map(str::to_string);
        let previous = (self.item > 1).then(|| format!("item_{}", self.item - 1));
        out.insert(0, self.event("input_audio_buffer.committed", json!({ "item_id": item, "previous_item_id": previous })));
        out.push(self.event(
            "conversation.item.input_audio_transcription.completed",
            json!({ "item_id": item, "content_index": 0, "transcript": transcript, "kvad": { "language": language, "heard": u.heard } }),
        ));
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
}
