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
use std::io::{BufRead, BufReader};
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

    fn send(&self, req: ureq::http::Request<Vec<u8>>) -> Res<ureq::http::Response<ureq::Body>> {
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
