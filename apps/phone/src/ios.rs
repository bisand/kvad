//! The few UIKit calls the app makes itself, through the Objective-C runtime
//! that UIKit is already linked with: the pasteboard, which Slint does not
//! expose, and the idle timer, which would lock the phone mid-run.
//!
//! Each must be called on the main thread, which is the UI's: Slint's
//! callbacks and `upgrade_in_event_loop` both run there.
//!
//! Off the phone (the Mac, while designing) they do nothing.

#[cfg(target_os = "ios")]
mod objc {
    use std::ffi::{c_char, c_void, CStr};

    pub type Id = *mut c_void;

    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Id;
        fn objc_msgSend();
    }

    /// `[target sel]`, returning an object.
    pub unsafe fn send(target: Id, sel: &CStr) -> Id {
        let f: unsafe extern "C" fn(Id, Id) -> Id = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, sel_registerName(sel.as_ptr()))
    }

    /// `[target sel:arg]`, with an object argument.
    pub unsafe fn send_id(target: Id, sel: &CStr, arg: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Id, Id) -> Id = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, sel_registerName(sel.as_ptr()), arg)
    }

    /// `[target sel:flag]`.
    pub unsafe fn send_bool(target: Id, sel: &CStr, flag: bool) {
        let f: unsafe extern "C" fn(Id, Id, bool) = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, sel_registerName(sel.as_ptr()), flag)
    }

    pub unsafe fn class(name: &CStr) -> Id {
        objc_getClass(name.as_ptr())
    }

    /// An `NSString`'s text.
    pub unsafe fn string(s: Id) -> Option<String> {
        if s.is_null() {
            return None;
        }
        let utf8 = send(s, c"UTF8String") as *const c_char;
        (!utf8.is_null()).then(|| CStr::from_ptr(utf8).to_string_lossy().into_owned())
    }
}

/// Keeps the screen from locking, which would suspend the app and its
/// sockets, while a run is on. `false` gives the phone its lock back.
pub fn keep_awake(on: bool) {
    #[cfg(target_os = "ios")]
    unsafe {
        let app = objc::send(objc::class(c"UIApplication"), c"sharedApplication");
        if !app.is_null() {
            objc::send_bool(app, c"setIdleTimerDisabled:", on);
        }
    }
    #[cfg(not(target_os = "ios"))]
    let _ = on;
}

/// What the phone is called, for the server's list of devices: the name iOS
/// gives apps, which is the model unless the app is entitled to more.
pub fn device_name() -> String {
    #[cfg(target_os = "ios")]
    unsafe {
        let device = objc::send(objc::class(c"UIDevice"), c"currentDevice");
        if !device.is_null() {
            if let Some(name) = objc::string(objc::send(device, c"name")) {
                return name;
            }
        }
    }
    "Kvad for iPhone".into()
}

/// The text on the pasteboard, which Universal Clipboard fills from the Mac.
/// iOS asks the user before it hands it over.
pub fn paste() -> Option<String> {
    #[cfg(target_os = "ios")]
    unsafe {
        let board = objc::send(objc::class(c"UIPasteboard"), c"generalPasteboard");
        if board.is_null() {
            return None;
        }
        objc::string(objc::send(board, c"string"))
    }
    #[cfg(not(target_os = "ios"))]
    None
}

/// Puts `text` on the pasteboard, from where the Mac can paste it too.
pub fn copy(text: &str) {
    #[cfg(target_os = "ios")]
    unsafe {
        let Ok(c) = std::ffi::CString::new(text) else { return };
        let board = objc::send(objc::class(c"UIPasteboard"), c"generalPasteboard");
        let s = objc::send_id(objc::class(c"NSString"), c"stringWithUTF8String:", c.as_ptr() as objc::Id);
        if !board.is_null() && !s.is_null() {
            objc::send_id(board, c"setString:", s);
        }
    }
    #[cfg(not(target_os = "ios"))]
    let _ = text;
}
