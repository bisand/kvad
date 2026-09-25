//! Where a model's time goes, a piece at a time.
//!
//! Off, [`span`] is one atomic load and a call. On ([`start`]), it
//! synchronises the device before and after its piece, so the time it
//! records is the GPU's and not the queue's, and adds it to a row named by
//! the enclosing [`scope`] and its own label. [`stop`] hands the rows back.
//!
//! The synchronising changes what it measures a little. candle's Metal pool
//! frees every dropped buffer at a synchronise, so a profiled run allocates
//! afresh where an ordinary one would reuse, and a profiled run is slower
//! for it. Compare its total with an unprofiled run's before trusting the
//! parts.

use candle_core::{Device, Result};
use std::borrow::Cow;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);

/// `(scope, label)` → seconds and calls, in first-seen order.
static ROWS: Mutex<Vec<Row>> = Mutex::new(Vec::new());

thread_local! {
    static SCOPE: Cell<&'static str> = const { Cell::new("") };
}

/// One piece's total.
#[derive(Clone, Debug)]
pub struct Row {
    pub scope: &'static str,
    pub label: Cow<'static, str>,
    pub seconds: f64,
    pub calls: usize,
}

/// Start recording, with no rows.
pub fn start() {
    ROWS.lock().unwrap().clear();
    ON.store(true, Ordering::Relaxed);
}

/// Stop recording and take the rows.
pub fn stop() -> Vec<Row> {
    ON.store(false, Ordering::Relaxed);
    std::mem::take(&mut *ROWS.lock().unwrap())
}

/// Run `f` with its spans labelled `name`. Scopes do not nest: the inner
/// one's name replaces the outer's until it returns.
pub(crate) fn scope<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !ON.load(Ordering::Relaxed) {
        return f();
    }
    let outer = SCOPE.with(|s| s.replace(name));
    let r = f();
    SCOPE.with(|s| s.set(outer));
    r
}

/// Run `f`, timing it on `device` when recording. Spans should not nest:
/// the outer one would count the inner one's synchronises too. A label that
/// has to be built is built only when recording: pass a closure.
pub(crate) fn span<T, L: Into<Cow<'static, str>>>(label: impl FnOnce() -> L, device: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !ON.load(Ordering::Relaxed) {
        return f();
    }
    device.synchronize()?;
    let t = Instant::now();
    let r = f()?;
    device.synchronize()?;
    let seconds = t.elapsed().as_secs_f64();
    let (scope, label) = (SCOPE.with(|s| s.get()), label().into());
    let mut rows = ROWS.lock().unwrap();
    match rows.iter_mut().find(|r| r.scope == scope && r.label == label) {
        Some(r) => (r.seconds, r.calls) = (r.seconds + seconds, r.calls + 1),
        None => rows.push(Row { scope, label, seconds, calls: 1 }),
    }
    Ok(r)
}
