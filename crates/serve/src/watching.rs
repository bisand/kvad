//! The two things that have to be wired into the server itself: a middleware
//! that times every request, and a writer that keeps the last few hundred log
//! lines where a browser can read them.

use crate::auth::State;
use crate::metrics::Metrics;
use axum::extract::{MatchedPath, Request, State as St};
use axum::middleware::Next;
use axum::response::Response;
use std::io::Write;
use std::sync::Arc;

/// Time a request and remember it.
///
/// The path recorded is the *route*, `/api/jobs/{id}` rather than
/// `/api/jobs/17`, because a histogram grouped by the second is a histogram
/// with one row per id. `MatchedPath` is axum's own answer to that and is
/// present for anything the router matched; the fallback — the embedded UI —
/// is recorded under one name for the same reason.
pub async fn timed(St(state): St<State>, request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = match request.extensions().get::<MatchedPath>() {
        Some(matched) => matched.as_str().to_string(),
        // Everything the API did not claim is the SPA; recording one row per
        // asset filename would bury the routes that matter.
        None => "(web ui)".to_string(),
    };
    let started = std::time::Instant::now();
    let response = next.run(request).await;
    // A handler that timed itself says so; see `RecordedItself`. Recording it
    // again here would double the row and halve the number.
    if response.extensions().get::<RecordedItself>().is_none() {
        state.metrics.record(&method, &path, response.status().as_u16(), started.elapsed());
    }
    response
}

/// A marker a handler puts in its response to say it has already recorded
/// itself.
///
/// The completions handler does, because only it knows what the generation
/// cost and because for a streamed reply this middleware would time the
/// headers rather than the work.
#[derive(Clone, Copy)]
pub struct RecordedItself;

/// A `tracing` writer that keeps the tail in memory as well as printing it.
///
/// So that "what did the server say" is answerable from the browser on a
/// machine somebody is not sitting at. Bounded, in memory, and gone on
/// restart: a log that mattered longer than that belongs in whatever is
/// running the process.
#[derive(Clone)]
pub struct LogTail(pub Arc<Metrics>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogTail {
    type Writer = TailWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TailWriter { metrics: Arc::clone(&self.0), buffer: Vec::new() }
    }
}

pub struct TailWriter {
    metrics: Arc<Metrics>,
    buffer: Vec<u8>,
}

impl Write for TailWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        // Still to stderr, because somebody watching a terminal should not
        // have to open a browser.
        std::io::stderr().write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

impl Drop for TailWriter {
    /// `tracing` makes a writer per event and drops it when the event is
    /// written, so this is where a line is complete.
    fn drop(&mut self) {
        let text = String::from_utf8_lossy(&self.buffer);
        for line in text.lines() {
            if !line.trim().is_empty() {
                self.metrics.log_line(strip_ansi(line));
            }
        }
    }
}

/// Remove the colour escapes `tracing` writes for a terminal.
///
/// The browser is not a terminal and would show them as `[2m[32m`. Only the
/// CSI sequences `tracing_subscriber` emits are handled, which is all there
/// are to handle.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        // `ESC [ … m` — skip to the terminating letter.
        for next in chars.by_ref() {
            if next.is_ascii_alphabetic() {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_escapes_do_not_reach_the_browser() {
        let coloured = "\u{1b}[2m2026-09-19T21:00:00Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m starting";
        assert_eq!(strip_ansi(coloured), "2026-09-19T21:00:00Z  INFO starting");
        assert_eq!(strip_ansi("plain text"), "plain text");
        // An escape with nothing after it must not hang or panic.
        assert_eq!(strip_ansi("trailing \u{1b}["), "trailing ");
    }

    /// A writer collects a whole event and files it when it is dropped, which
    /// is what `tracing` does with one.
    #[test]
    fn a_log_line_reaches_the_tail_when_its_writer_is_dropped() {
        let metrics = Arc::new(Metrics::new());
        {
            let mut w = TailWriter { metrics: Arc::clone(&metrics), buffer: Vec::new() };
            w.write_all("\u{1b}[32m INFO\u{1b}[0m one\nsecond line\n".as_bytes()).unwrap();
            assert!(metrics.log(10).is_empty(), "the line was filed before it was complete");
        }
        assert_eq!(metrics.log(10), [" INFO one", "second line"]);
    }
}
