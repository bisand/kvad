//! The Rust under Kvad for iPhone, for the SwiftUI app to call: three C
//! functions, and one JSON object a message each way. The commands and
//! events are in `session` and `probe`, and every one of them is in
//! `messages.json`, which the tests on both sides read
//! (docs/phone-swiftui-plan.md).
//!
//! ```c
//! void kvad_core_start(void *context, void (*on_event)(void *context, const char *json));
//! void kvad_core_send(const char *json);
//! const char *kvad_core_version(void);
//! ```
//!
//! `on_event` is called on any of the core's threads, with a JSON object
//! the core frees once the call returns: copy it, then hop to the main
//! thread. `kvad_core_send` returns at once; what comes of it are events.

pub mod api;
pub mod pairings;
pub mod probe;
pub mod secrets;
pub mod session;

use session::{Emit, Session};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::{Arc, OnceLock};

static SESSION: OnceLock<Session> = OnceLock::new();

/// Starts the core, once: later calls change nothing. Its first event is
/// `ready`.
pub fn start(emit: Emit) {
    SESSION.get_or_init(|| Session::new(emit));
}

/// One command, as JSON text. Text that is not JSON is answered with an
/// `error` event; before `start` there is no one to answer.
pub fn send(command: &str) {
    let Some(session) = SESSION.get() else { return };
    match serde_json::from_str(command) {
        Ok(command) => session.handle(&command),
        Err(e) => session.handle(&serde_json::json!({ "not json": e.to_string() })),
    }
}

/// The C callback and the pointer it is given back, which belong to the
/// app. It promises that both may be used from any thread.
struct Callback {
    context: *mut c_void,
    on_event: extern "C" fn(*mut c_void, *const c_char),
}

unsafe impl Send for Callback {}
unsafe impl Sync for Callback {}

impl Callback {
    /// Through a method, so that a closure takes the whole `Callback`, which
    /// is `Send`, and not its pointer alone, which is not.
    fn call(&self, json: &CStr) {
        (self.on_event)(self.context, json.as_ptr());
    }
}

/// # Safety
/// `on_event` must be safe to call from any thread with `context`, for as
/// long as the app runs.
#[no_mangle]
pub unsafe extern "C" fn kvad_core_start(context: *mut c_void, on_event: extern "C" fn(*mut c_void, *const c_char)) {
    let callback = Callback { context, on_event };
    start(Arc::new(move |event| {
        // JSON escapes any NUL inside a string, so the text has none.
        if let Ok(text) = CString::new(event.to_string()) {
            callback.call(&text);
        }
    }));
}

/// # Safety
/// `json` must be a NUL-terminated string, or null, which is ignored.
#[no_mangle]
pub unsafe extern "C" fn kvad_core_send(json: *const c_char) {
    if json.is_null() {
        return;
    }
    send(&CStr::from_ptr(json).to_string_lossy());
}

#[no_mangle]
pub extern "C" fn kvad_core_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}
