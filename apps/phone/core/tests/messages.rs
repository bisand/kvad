//! The core, driven by its messages alone, as the SwiftUI app drives it:
//! every command in messages.json is one the core knows, and every event it
//! emits is one messages.json lists, with no field the example lacks.
//!
//! No test reaches a server. A connect goes to a closed port on this
//! machine, and the rest are answered without the network.

use kvad_phone_core::session::{Emit, Session};
use serde_json::{json, Value};
use std::sync::mpsc;
use std::time::Duration;

fn messages() -> Value {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../messages.json")).expect("messages.json");
    serde_json::from_str(&text).expect("messages.json is JSON")
}

/// A session whose events arrive on a channel.
fn session() -> (Session, mpsc::Receiver<Value>) {
    let (tx, rx) = mpsc::channel();
    let tx = std::sync::Mutex::new(tx);
    let emit: Emit = std::sync::Arc::new(move |e| {
        let _ = tx.lock().unwrap().send(e);
    });
    (Session::new(emit), rx)
}

/// The next event, within `secs`.
fn next(rx: &mpsc::Receiver<Value>, secs: u64) -> Value {
    rx.recv_timeout(Duration::from_secs(secs)).expect("an event")
}

fn name(e: &Value) -> &str {
    e.as_object().and_then(|o| o.keys().next()).map(String::as_str).unwrap_or_default()
}

/// The event is one messages.json lists, and has no field its example lacks.
fn listed(e: &Value) {
    let all = messages();
    let n = name(e);
    let example = &all["events"][n];
    assert!(example.is_object(), "{n} is not in messages.json's events: {e}");
    for key in e[n].as_object().expect("an event's body is an object").keys() {
        assert!(example.get(key).is_some(), "{n}.{key} is not in messages.json's example: {e}");
    }
}

/// A valid `kvad://` address for a server that is not there.
fn address() -> String {
    format!("kvad://{}?k=0123456789abcdef0123456789abcdef&addr=127.0.0.1:9", kvad_tunnel::Key::generate().id())
}

#[test]
fn starts_ready() {
    let (_s, rx) = session();
    let e = next(&rx, 5);
    assert_eq!(name(&e), "ready");
    listed(&e);
}

#[test]
fn every_command_is_known() {
    // probe_open is left out: it makes the probe key under $HOME.
    for (command, example) in messages()["commands"].as_object().unwrap() {
        if command == "probe_open" {
            continue;
        }
        let (s, rx) = session();
        next(&rx, 5);
        s.handle(&json!({ command: example }));
        // A command that answers at all answers within a second, except a
        // connect, which is a connect_failed after a moment.
        while let Ok(e) = rx.recv_timeout(Duration::from_secs(if command == "connect" { 20 } else { 1 })) {
            listed(&e);
            assert_ne!(name(&e), "error", "{command}: {e}");
            if command == "connect" {
                break;
            }
        }
    }
}

#[test]
fn an_unknown_command_or_text_that_is_not_json_is_an_error() {
    let (s, rx) = session();
    next(&rx, 5);
    s.handle(&json!({ "dance": {} }));
    let e = next(&rx, 1);
    listed(&e);
    assert_eq!(e["error"]["error"], "No command is called dance.");
    s.handle(&json!({ "chat": {}, "stop": {} }));
    assert_eq!(name(&next(&rx, 1)), "error");
}

#[test]
fn chat_models_and_unpair_want_a_connection() {
    let (s, rx) = session();
    next(&rx, 5);
    s.handle(&json!({ "chat": { "model": "m", "text": "Hi" } }));
    let e = next(&rx, 1);
    listed(&e);
    assert_eq!(e, json!({ "chat_failed": { "error": "Not connected to a server.", "nothing": true } }));
    s.handle(&json!({ "models": {} }));
    assert_eq!(name(&next(&rx, 1)), "models_failed");
    s.handle(&json!({ "unpair": {} }));
    assert_eq!(name(&next(&rx, 1)), "unpair_failed");
}

#[test]
fn a_connect_to_nothing_fails() {
    let (s, rx) = session();
    next(&rx, 5);
    s.handle(&json!({ "connect": { "url": "127.0.0.1:9", "key": "", "device": "test" } }));
    let e = next(&rx, 20);
    listed(&e);
    assert_eq!(name(&e), "connect_failed", "{e}");
}

#[test]
fn a_disconnect_drops_the_answer_to_an_earlier_connect() {
    let (s, rx) = session();
    next(&rx, 5);
    s.handle(&json!({ "connect": { "url": "127.0.0.1:9", "key": "", "device": "test" } }));
    s.handle(&json!({ "disconnect": {} }));
    assert_eq!(name(&next(&rx, 1)), "disconnected");
    assert!(rx.recv_timeout(Duration::from_secs(3)).is_err(), "the failed connect was not dropped");
}

#[test]
fn an_opened_link_is_named_and_never_followed() {
    let (s, rx) = session();
    next(&rx, 5);
    let link = address();
    s.handle(&json!({ "opened": { "link": link } }));
    let e = next(&rx, 1);
    listed(&e);
    assert_eq!(e["link"]["server"], link.as_str());
    assert!(e["link"]["notice"].as_str().unwrap().starts_with("Opened kvad://"), "{e}");

    let pairing = format!("{link}&pair={}", "a".repeat(43));
    s.handle(&json!({ "opened": { "link": pairing } }));
    let e = next(&rx, 1);
    assert!(e["link"]["notice"].as_str().unwrap().starts_with("Pair with kvad://"), "{e}");

    for bad in ["https://example.com", "kvad://nothing", &"x".repeat(5000)] {
        s.handle(&json!({ "opened": { "link": bad } }));
        assert_eq!(name(&next(&rx, 1)), "link_refused", "{bad}");
    }
    // Nothing else came of it: no connect was tried.
    assert!(rx.recv_timeout(Duration::from_millis(500)).is_err());
}

#[test]
fn a_probe_without_a_ticket_fails_and_a_stop_is_harmless() {
    let (s, rx) = session();
    next(&rx, 5);
    s.handle(&json!({ "probe_start": { "ticket": "nonsense", "secs": 30 } }));
    let e = next(&rx, 1);
    listed(&e);
    assert_eq!(name(&e), "probe_failed");
    s.handle(&json!({ "probe_stop": {} }));
    assert!(rx.recv_timeout(Duration::from_millis(500)).is_err());
}

/// Against a real kvad-serve, which this machine may not have: connected,
/// the reply in pieces, then reply_done, each one listed. With
/// `KVAD_PHONE_TEST_SERVER=127.0.0.1:5823 cargo test -p kvad-phone-core -- --ignored`
/// and `KVAD_PHONE_TEST_MODEL` for a model other than the server's default.
#[test]
#[ignore]
fn a_chat_with_a_running_server() {
    let url = std::env::var("KVAD_PHONE_TEST_SERVER").unwrap_or_else(|_| "127.0.0.1:5823".into());
    let (s, rx) = session();
    next(&rx, 5);
    s.handle(&json!({ "connect": { "url": url, "key": "", "device": "test" } }));
    let e = next(&rx, 30);
    listed(&e);
    assert_eq!(name(&e), "connected", "{e}");
    let model = std::env::var("KVAD_PHONE_TEST_MODEL").unwrap_or_else(|_| e["connected"]["default"].as_str().unwrap().to_string());
    s.handle(&json!({ "chat": { "model": model, "text": "Say hello in three words." } }));
    let mut reply = String::new();
    loop {
        let e = next(&rx, 120);
        listed(&e);
        match name(&e) {
            "piece" => reply.push_str(e["piece"]["text"].as_str().unwrap()),
            "reply_done" => break,
            _ => panic!("{e}"),
        }
    }
    assert!(!reply.trim().is_empty());
    s.handle(&json!({ "models": {} }));
    let e = next(&rx, 30);
    listed(&e);
    assert_eq!(name(&e), "models");
}
