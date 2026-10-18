//! The few UIKit calls the app makes itself, through the Objective-C runtime
//! that UIKit is already linked with: the pasteboard, which Slint does not
//! expose; the idle timer, which would lock the phone mid-run; and the scene
//! delegate's methods for a `kvad://` link opened in the app, and for its
//! coming back to the foreground ([`listen`]).
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
        fn class_addMethod(class: Id, sel: Id, imp: *const c_void, types: *const c_char) -> bool;
    }

    /// `[target sel]`, returning an integer.
    pub unsafe fn send_usize(target: Id, sel: &CStr) -> usize {
        let f: unsafe extern "C" fn(Id, Id) -> usize = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, sel_registerName(sel.as_ptr()))
    }

    /// `[target sel:i]`, with an integer argument, returning an object.
    pub unsafe fn send_index(target: Id, sel: &CStr, i: usize) -> Id {
        let f: unsafe extern "C" fn(Id, Id, usize) -> Id = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(target, sel_registerName(sel.as_ptr()), i)
    }

    /// Adds a method to a class, unless it has one by that name already.
    pub unsafe fn add_method(class: Id, sel: &CStr, imp: *const c_void, types: &CStr) -> bool {
        !class.is_null() && class_addMethod(class, sel_registerName(sel.as_ptr()), imp, types.as_ptr())
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

/// What iOS tells the app about itself, once [`listen`] has been called.
#[cfg_attr(not(target_os = "ios"), allow(dead_code))]
pub struct Events {
    /// A `kvad://` link opened in the app: the system's Camera scanned the QR
    /// code `kvad remote pair` shows, or a link was tapped. It is only ever
    /// put in front of the person, never acted on by itself.
    pub opened: Box<dyn Fn(String) + Send + Sync>,
    /// The app came back to the foreground, maybe on another network.
    pub active: Box<dyn Fn() + Send + Sync>,
}

#[cfg(target_os = "ios")]
static EVENTS: std::sync::OnceLock<Events> = std::sync::OnceLock::new();

/// Adds to Slint's scene delegate the methods that UIKit calls with an
/// opened URL (at launch, and while running) and when the app becomes
/// active. Slint's delegate implements none of them; it attaches its windows
/// from notifications. Call it after the window is made, which registers the
/// class, and before `run`, which connects the scene.
pub fn listen(events: Events) {
    #[cfg(target_os = "ios")]
    unsafe {
        use std::ffi::c_void;
        if EVENTS.set(events).is_err() {
            return;
        }
        let class = objc::class(c"SlintWindowSceneDelegate");
        let open: extern "C" fn(objc::Id, objc::Id, objc::Id, objc::Id) = opened_urls;
        let connect: extern "C" fn(objc::Id, objc::Id, objc::Id, objc::Id, objc::Id) = will_connect;
        let active: extern "C" fn(objc::Id, objc::Id, objc::Id) = became_active;
        let added = [
            objc::add_method(class, c"scene:openURLContexts:", open as *const c_void, c"v@:@@"),
            objc::add_method(class, c"scene:willConnectToSession:options:", connect as *const c_void, c"v@:@@@"),
            objc::add_method(class, c"sceneDidBecomeActive:", active as *const c_void, c"v@:@"),
        ];
        if added.contains(&false) {
            eprintln!("kvad-phone: the scene delegate already has some of its methods; links may not open");
        }
    }
    #[cfg(not(target_os = "ios"))]
    let _ = events;
}

#[cfg(target_os = "ios")]
extern "C" fn opened_urls(_this: objc::Id, _cmd: objc::Id, _scene: objc::Id, contexts: objc::Id) {
    unsafe { deliver(contexts) }
}

#[cfg(target_os = "ios")]
extern "C" fn will_connect(_this: objc::Id, _cmd: objc::Id, _scene: objc::Id, _session: objc::Id, options: objc::Id) {
    if !options.is_null() {
        unsafe { deliver(objc::send(options, c"URLContexts")) }
    }
}

#[cfg(target_os = "ios")]
extern "C" fn became_active(_this: objc::Id, _cmd: objc::Id, _scene: objc::Id) {
    if let Some(e) = EVENTS.get() {
        (e.active)();
    }
}

/// Each `UIOpenURLContext` in a set, as its URL's text.
#[cfg(target_os = "ios")]
unsafe fn deliver(contexts: objc::Id) {
    let Some(e) = EVENTS.get() else { return };
    if contexts.is_null() {
        return;
    }
    let all = objc::send(contexts, c"allObjects");
    for i in 0..objc::send_usize(all, c"count") {
        let url = objc::send(objc::send_index(all, c"objectAtIndex:", i), c"URL");
        if let Some(text) = objc::string(objc::send(url, c"absoluteString")) {
            (e.opened)(text);
        }
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
