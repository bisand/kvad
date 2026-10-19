//! The phone as a client of a kvad-serve.
//!
//! Blocking, on purpose: every call here runs on a worker thread that the UI
//! started, and hands what it learnt back to the UI's thread. No async
//! runtime is needed until the tunnel brings one (docs/phone-plan.md).
//!
//! The shapes are the server's own. `/api/health` says who the server is and
//! whether it wants a key; `/v1/models` lists what can be asked for; a chat
//! is OpenAI's `/v1/chat/completions` with `stream: true`, read as
//! server-sent events a line at a time.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub type Res<T> = Result<T, String>;

/// Where the server is, and the key to show it, if it wants one.
#[derive(Clone)]
pub struct Server {
    /// What request URLs start with.
    base: String,
    /// What the server is called on screen: its address, or for one reached
    /// through the tunnel its `kvad://<node id>`.
    name: String,
    key: Option<String>,
    agent: ureq::Agent,
    tunnel: Option<std::sync::Arc<kvad_tunnel::blocking::Tunnel>>,
}

/// The stand-in a tunnelled request's URL starts with: an address, so no name
/// is looked up, and never dialled, since every connection is a stream
/// through the tunnel.
const TUNNELLED: &str = "http://127.0.0.1:5823";

/// What `/api/health` says.
#[derive(Clone, Debug, PartialEq)]
pub struct Health {
    pub version: String,
    /// `none` when the server takes anyone on loopback; otherwise the kind of
    /// sign-in it wants.
    pub auth: String,
    /// Who the server takes this client to be, and with what role.
    pub you: String,
    pub role: String,
}

/// One model the server can be asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub id: String,
    /// chat, image, video, speech, embedding…
    pub kind: String,
    /// In memory now, so a request is answered without a load first.
    pub resident: bool,
}

/// One turn of a conversation, as OpenAI's API has it.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub role: &'static str,
    pub content: String,
}

impl Server {
    /// A server at `url`, which is checked to be one the phone may talk to.
    pub fn new(url: &str, key: Option<String>) -> Res<Server> {
        let base = normalise(url)?;
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                .user_agent(concat!("kvad-phone/", env!("CARGO_PKG_VERSION")))
                .timeout_connect(Some(Duration::from_secs(5)))
                // Statuses are answers: the server's sentence is read out of
                // them rather than ureq's.
                .http_status_as_error(false)
                .build(),
        );
        let key = key.filter(|k| !k.trim().is_empty()).map(|k| k.trim().to_string());
        Ok(Server { name: base.clone(), base, key, agent, tunnel: None })
    }

    /// A paired server, through `tunnel`, with the key it made for this phone.
    pub fn through(tunnel: kvad_tunnel::blocking::Tunnel, key: String) -> Server {
        let name = tunnel.address().label();
        let tunnel = std::sync::Arc::new(tunnel);
        let config = ureq::Agent::config_builder()
            .user_agent(concat!("kvad-phone/", env!("CARGO_PKG_VERSION")))
            .http_status_as_error(false)
            .build();
        let agent = kvad_tunnel::for_ureq::agent(tunnel.clone(), config);
        Server { base: TUNNELLED.into(), name, key: Some(key), agent, tunnel: Some(tunnel) }
    }

    /// What the server is called on screen.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The path the tunnel is on now: `lan …`, `direct …` or `relay …`.
    pub fn path(&self) -> Option<String> {
        self.tunnel.as_ref()?.path().map(|p| p.to_string())
    }

    /// Says the phone's network may have changed; see
    /// `kvad_tunnel::Client::network_changed`.
    pub fn network_changed(&self) {
        if let Some(t) = &self.tunnel {
            t.network_changed();
        }
    }

    /// A fresh relay voucher for this phone, from a server it is paired with.
    pub fn voucher(&self) -> Res<String> {
        let v = self.get("/api/remote/voucher")?;
        v.get("voucher").and_then(Value::as_str).map(str::to_string).ok_or_else(|| "The server sent no voucher.".to_string())
    }

    /// Unpairs device `id`: this phone, when it is its own id.
    pub fn unpair(&self, id: i64) -> Res<()> {
        let req = self.request("DELETE", &format!("/api/devices/{id}"), "application/json").body(Vec::new()).map_err(|e| e.to_string())?;
        read(self.send(req)?).map(|_| ())
    }

    fn request(&self, method: &str, path: &str, accept: &str) -> ureq::http::request::Builder {
        let mut req = ureq::http::Request::builder().method(method).uri(format!("{}{path}", self.base)).header("accept", accept);
        if let Some(key) = &self.key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        req
    }

    fn send(&self, req: ureq::http::Request<impl ureq::AsSendBody>) -> Res<ureq::http::Response<ureq::Body>> {
        self.agent.run(req).map_err(|e| match e {
            ureq::Error::Io(io) if io.kind() == std::io::ErrorKind::ConnectionRefused => {
                format!("Nothing is answering at {}.", self.name)
            }
            ureq::Error::Timeout(_) => format!("{} did not answer in time.", self.name),
            other => format!("Could not reach {}: {other}", self.name),
        })
    }

    fn get(&self, path: &str) -> Res<Value> {
        let req = self.request("GET", path, "application/json").body(Vec::new()).map_err(|e| e.to_string())?;
        read(self.send(req)?)
    }

    pub fn health(&self) -> Res<Health> {
        let v = self.get("/api/health")?;
        if v.get("status").and_then(Value::as_str) != Some("ok") {
            return Err(format!("{} answered, but not as a kvad server would.", self.name));
        }
        let text = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        let you = v.get("you").cloned().unwrap_or(Value::Null);
        Ok(Health { version: text(&v, "version"), auth: text(&v, "auth"), you: text(&you, "name"), role: text(&you, "role") })
    }

    pub fn models(&self) -> Res<Vec<Model>> {
        let v = self.get("/v1/models")?;
        let list = v.get("data").and_then(Value::as_array).ok_or("The server's model list was not a list.")?;
        Ok(list
            .iter()
            .filter_map(|m| {
                let kvad = m.get("kvad").cloned().unwrap_or(Value::Null);
                Some(Model {
                    id: m.get("id")?.as_str()?.to_string(),
                    kind: kvad.get("kind").and_then(Value::as_str).unwrap_or("chat").to_string(),
                    resident: kvad.get("resident").and_then(Value::as_bool).unwrap_or(false),
                })
            })
            .collect())
    }

    /// A reply to `turns` from `model`, a piece at a time to `piece` as the
    /// server writes it. Stops early, between pieces, once `stop` is set.
    pub fn chat(&self, model: &str, turns: &[Turn], stop: &AtomicBool, mut piece: impl FnMut(&str)) -> Res<()> {
        let messages: Vec<Value> = turns.iter().map(|t| json!({ "role": t.role, "content": t.content })).collect();
        let body = json!({ "model": model, "messages": messages, "stream": true });
        let req = self
            .request("POST", "/v1/chat/completions", "text/event-stream")
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&body).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let response = self.send(req)?;
        if response.status().as_u16() >= 400 {
            return read(response).map(|_| ());
        }
        let mut lines = BufReader::new(response.into_body().into_reader());
        let mut line = String::new();
        loop {
            if stop.load(Ordering::Relaxed) {
                // Dropping the reader hangs up, and the server stops writing.
                return Ok(());
            }
            line.clear();
            if lines.read_line(&mut line).map_err(|e| format!("The reply broke off: {e}"))? == 0 {
                return Ok(());
            }
            match event(&line)? {
                Event::Piece(text) => piece(&text),
                Event::Done => return Ok(()),
                Event::Nothing => {}
            }
        }
    }
}

impl Server {
    /// A WebSocket to `path` (with its query), for the live transcription
    /// of `docs/speech-plan.md`: a stream through the tunnel, or a TCP
    /// connection on the LAN, upgraded with the same key as every request.
    /// Its timeouts are 10 s each way until the caller sets its own.
    pub fn socket(&self, path: &str) -> Result<Socket, Refused> {
        let not = |why: String| Refused { status: None, why };
        let (scheme, host) = self.base.split_once("://").expect("a base has a scheme");
        let stream = match &self.tunnel {
            Some(t) => Stream::Tunnel(t.open().map_err(not)?),
            None if scheme == "https" => {
                return Err(not("Live words need a paired server or an http:// address. This recording is kept, and sent whole when it ends.".into()))
            }
            None => {
                let addr = host.to_socket_addrs().ok().and_then(|mut a| a.next()).ok_or_else(|| not(format!("{} is not an address this phone can find.", self.name)))?;
                Stream::Tcp(TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| not(format!("Could not reach {}: {e}", self.name)))?)
            }
        };
        stream.set_timeouts(Some(Duration::from_secs(10))).map_err(|e| not(e.to_string()))?;
        let mut req = tungstenite::client::IntoClientRequest::into_client_request(format!("ws://{host}{path}")).map_err(|e| not(e.to_string()))?;
        let headers = req.headers_mut();
        headers.insert("user-agent", concat!("kvad-phone/", env!("CARGO_PKG_VERSION")).parse().expect("a header value"));
        if let Some(key) = &self.key {
            headers.insert("authorization", format!("Bearer {key}").parse().map_err(|_| not("The key is not one a header can carry.".into()))?);
        }
        match tungstenite::client(req, stream) {
            Ok((socket, _)) => Ok(socket),
            Err(tungstenite::HandshakeError::Failure(tungstenite::Error::Http(answer))) => {
                let status = answer.status().as_u16();
                let json: Value = answer.body().as_deref().and_then(|b| serde_json::from_slice(b).ok()).unwrap_or(Value::Null);
                Err(Refused { status: Some(status), why: error_message(&json).unwrap_or_else(|| format!("The server answered {status}.")) })
            }
            Err(e) => Err(not(format!("The live connection to {} failed: {e}", self.name))),
        }
    }

    /// `wav` transcribed whole by `POST /v1/audio/transcriptions`, the text
    /// a 30-second window at a time to `piece` as the server writes it
    /// (`stream: true`), and all of it at the end, a new paragraph wherever
    /// Whisper's segments leave a gap of `speak::PAUSE` or more. The file is
    /// read as it is sent, never held whole. Stops between windows once
    /// `stop` is set.
    pub fn transcribe(&self, wav: &std::path::Path, model: &str, language: &str, stop: &AtomicBool, mut piece: impl FnMut(&str)) -> Res<String> {
        let boundary = format!("kvad-{:016x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));
        let mut head = String::new();
        let mut field = |name: &str, value: &str| {
            if !value.is_empty() {
                head += &format!("--{boundary}\r\ncontent-disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n");
            }
        };
        field("model", model);
        field("language", language);
        field("stream", "true");
        head += &format!("--{boundary}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"take.wav\"\r\ncontent-type: audio/wav\r\n\r\n");
        let tail = format!("\r\n--{boundary}--\r\n");
        let file = std::fs::File::open(wav).map_err(|e| format!("The recording could not be read: {e}"))?;
        let length = head.len() as u64 + file.metadata().map_err(|e| e.to_string())?.len() + tail.len() as u64;
        let body = std::io::Cursor::new(head.into_bytes()).chain(file).chain(std::io::Cursor::new(tail.into_bytes()));
        let req = self
            .request("POST", "/v1/audio/transcriptions", "text/event-stream")
            .header("content-type", format!("multipart/form-data; boundary={boundary}"))
            .header("content-length", length.to_string())
            .body(ureq::SendBody::from_owned_reader(body))
            .map_err(|e| e.to_string())?;
        let response = self.send(req)?;
        if response.status().as_u16() >= 400 {
            return read(response).and(Err("The server sent no transcript.".into()));
        }
        let mut lines = BufReader::new(response.into_body().into_reader());
        let mut line = String::new();
        let mut segments: Vec<(f64, f64, String)> = Vec::new();
        loop {
            if stop.load(Ordering::Relaxed) {
                return Err("Stopped.".into());
            }
            line.clear();
            if lines.read_line(&mut line).map_err(|e| format!("The transcript broke off: {e}"))? == 0 {
                return Err("The transcript broke off before its end.".into());
            }
            let Some(data) = line.trim_end().strip_prefix("data:") else { continue };
            let v: Value = serde_json::from_str(data.trim_start()).map_err(|_| "The server sent a piece of transcript that was not JSON.".to_string())?;
            if let Some(message) = error_message(&v) {
                return Err(message);
            }
            match v["type"].as_str() {
                Some("transcript.text.delta") => {
                    piece(v["delta"].as_str().unwrap_or_default());
                    for s in v["kvad"]["segments"].as_array().into_iter().flatten() {
                        let time = |k: &str| s[k].as_f64().unwrap_or(0.0);
                        segments.push((time("start"), time("end"), s["text"].as_str().unwrap_or_default().to_string()));
                    }
                }
                Some("transcript.text.done") => {
                    let text = v["text"].as_str().unwrap_or_default();
                    return Ok(if segments.is_empty() { text.to_string() } else { paragraphs(&segments) });
                }
                _ => {}
            }
        }
    }
}

/// A connection to the server: TCP on the LAN, or a stream through the
/// tunnel.
pub enum Stream {
    Tcp(TcpStream),
    Tunnel(UnixStream),
}

/// A live transcription's WebSocket.
pub type Socket = tungstenite::WebSocket<Stream>;

/// Why a WebSocket was not opened, with the server's status when it
/// answered at all.
#[derive(Debug)]
pub struct Refused {
    pub status: Option<u16>,
    pub why: String,
}

impl Stream {
    /// How long a read or a write may wait before it gives up with
    /// `WouldBlock` or `TimedOut`.
    pub fn set_timeouts(&self, wait: Option<Duration>) -> std::io::Result<()> {
        match self {
            Stream::Tcp(s) => s.set_read_timeout(wait).and_then(|_| s.set_write_timeout(wait)),
            Stream::Tunnel(s) => s.set_read_timeout(wait).and_then(|_| s.set_write_timeout(wait)),
        }
    }

    /// Only reads, for `read`; a write keeps its own.
    pub fn set_read_timeout(&self, wait: Option<Duration>) -> std::io::Result<()> {
        match self {
            Stream::Tcp(s) => s.set_read_timeout(wait),
            Stream::Tunnel(s) => s.set_read_timeout(wait),
        }
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.read(buf),
            Stream::Tunnel(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.write(buf),
            Stream::Tunnel(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Stream::Tcp(s) => s.flush(),
            Stream::Tunnel(s) => s.flush(),
        }
    }
}

/// Segments' text, with a blank line where one begins a pause or more after
/// the last ended.
pub fn paragraphs(segments: &[(f64, f64, String)]) -> String {
    let pause = crate::speak::PAUSE as f64 / crate::speak::RATE as f64;
    let mut text = String::new();
    let mut last_end: Option<f64> = None;
    for (start, end, words) in segments {
        let words = words.trim();
        if words.is_empty() {
            continue;
        }
        match last_end {
            Some(e) if start - e >= pause => text.push_str("\n\n"),
            Some(_) => text.push(' '),
            None => {}
        }
        text.push_str(words);
        last_end = Some(*end);
    }
    text
}

/// What one line of a streamed chat is.
#[derive(Debug, PartialEq)]
enum Event {
    Piece(String),
    Done,
    Nothing,
}

fn event(line: &str) -> Res<Event> {
    let Some(data) = line.trim_end().strip_prefix("data:") else { return Ok(Event::Nothing) };
    let data = data.trim_start();
    if data == "[DONE]" {
        return Ok(Event::Done);
    }
    let v: Value = serde_json::from_str(data).map_err(|_| "The server sent a piece of reply that was not JSON.".to_string())?;
    if let Some(message) = error_message(&v) {
        return Err(message);
    }
    let text = v.pointer("/choices/0/delta/content").and_then(Value::as_str).unwrap_or_default();
    Ok(if text.is_empty() { Event::Nothing } else { Event::Piece(text.to_string()) })
}

/// The server's own sentence out of an error, in either of the shapes it
/// writes them: kvad's `{"error": "…"}` or OpenAI's `{"error": {"message"}}`.
fn error_message(v: &Value) -> Option<String> {
    match v.get("error")? {
        Value::String(s) => Some(s.clone()),
        e => e.get("message").and_then(Value::as_str).map(str::to_string),
    }
}

fn read(mut response: ureq::http::Response<ureq::Body>) -> Res<Value> {
    let status = response.status().as_u16();
    let text = response.body_mut().with_config().limit(64 * 1024 * 1024).read_to_string().unwrap_or_default();
    let json: Value = serde_json::from_str(text.trim()).unwrap_or(Value::Null);
    if status == 401 || status == 403 {
        return Err(error_message(&json).unwrap_or_else(|| "The server wants a key, or a different one.".into()));
    }
    if status >= 400 {
        return Err(error_message(&json).unwrap_or_else(|| format!("The server answered {status}.")));
    }
    Ok(json)
}

/// A base URL the phone may use: http or https, no path, no trailing slash.
/// Plain http is left to the phone's own network by App Transport Security
/// (NSAllowsLocalNetworking), which is the stricter check; this one is for
/// what a person types.
fn normalise(url: &str) -> Res<String> {
    let url = url.trim().trim_end_matches('/');
    let with_scheme = if url.contains("://") { url.to_string() } else { format!("http://{url}") };
    let Some((scheme, rest)) = with_scheme.split_once("://") else { unreachable!() };
    if scheme != "http" && scheme != "https" {
        return Err(format!("{scheme}:// is not a way to reach a server; use http:// or https://."));
    }
    if rest.is_empty() || rest.contains('/') || rest.contains('@') {
        return Err("A server is a host and a port, like 192.168.1.20:5823.".into());
    }
    Ok(format!("{scheme}://{rest}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gap_between_segments_begins_a_paragraph() {
        let seg = |a: f64, b: f64, t: &str| (a, b, t.to_string());
        let text = paragraphs(&[seg(0.0, 2.0, " Hei."), seg(2.4, 4.0, " Hvordan går det?"), seg(6.0, 7.0, " Bra."), seg(7.1, 7.5, " ")]);
        assert_eq!(text, "Hei. Hvordan går det?\n\nBra.");
        assert_eq!(paragraphs(&[]), "");
    }

    #[test]
    fn a_typed_address_becomes_a_base() {
        assert_eq!(normalise("192.168.1.20:5823").unwrap(), "http://192.168.1.20:5823");
        assert_eq!(normalise(" https://mac.local:5823/ ").unwrap(), "https://mac.local:5823");
        assert!(normalise("ftp://x").is_err());
        assert!(normalise("http://x/api").is_err());
        assert!(normalise("http://user@x").is_err());
        assert!(normalise("").is_err());
    }

    #[test]
    fn a_stream_is_read_a_piece_at_a_time() {
        let piece = r#"data: {"choices":[{"delta":{"content":"Hei"}}]}"#;
        assert_eq!(event(piece).unwrap(), Event::Piece("Hei".into()));
        assert_eq!(event("data: [DONE]\n").unwrap(), Event::Done);
        assert_eq!(event(": keep-alive\n").unwrap(), Event::Nothing);
        assert_eq!(event(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#).unwrap(), Event::Nothing);
    }

    #[test]
    fn an_error_in_a_stream_is_the_servers_sentence() {
        assert_eq!(event(r#"data: {"error":"no model loaded"}"#).unwrap_err(), "no model loaded");
        assert_eq!(event(r#"data: {"error":{"message":"too long"}}"#).unwrap_err(), "too long");
    }
}
