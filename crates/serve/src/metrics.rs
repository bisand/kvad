//! What the server has been doing lately.
//!
//! Two stores, because two questions:
//!
//! * **A ring buffer in memory** answers "what is happening now" — the last
//!   few thousand requests, the last few hundred generations, the last few
//!   hundred log lines. Reading it costs a mutex and no disk, which is what a
//!   dashboard polling every few seconds needs.
//! * **A table in SQLite** answers "what did yesterday look like". It is
//!   written in batches rather than per request, because a lock on the
//!   database in the middle of every response is a lock on the server, and
//!   it is pruned, because a row a second is nothing for SQLite and
//!   unreadable for a person.
//!
//! Neither is a metrics system. There is no histogram bucketing, no
//! percentile estimator and no exporter: the numbers here are the ones this
//! server can measure honestly about itself, and the percentiles are computed
//! from the actual samples because there are few enough to sort.

use crate::db::Db;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Requests kept in memory. A few thousand is a few hundred kilobytes and
/// about an hour of a busy server.
const REQUESTS: usize = 2000;
/// Generations kept. Fewer, because each is seconds rather than milliseconds.
const GENERATIONS: usize = 500;
/// Log lines kept for the tail.
const LOG_LINES: usize = 500;
/// How often the ring buffer is emptied into the database.
pub const FLUSH_EVERY: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, serde::Serialize)]
pub struct Request {
    /// Milliseconds since the server started, so the client can draw a
    /// timeline without every row carrying a timestamp string.
    pub at_millis: u64,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub millis: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<Generation>,
}

/// What one reply cost the engine.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Generation {
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub generated_tokens: u32,
    pub decode_per_sec: f64,
    /// Time to the first token: prefill, which is what somebody waiting for a
    /// reply actually experiences before anything appears.
    pub ttft_millis: f64,
}

pub struct Metrics {
    started: Instant,
    requests: Mutex<VecDeque<Request>>,
    /// Requests written to memory and not yet to the database.
    pending: Mutex<Vec<Request>>,
    generations: Mutex<VecDeque<Generation>>,
    log: Mutex<VecDeque<String>>,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            started: Instant::now(),
            requests: Mutex::new(VecDeque::with_capacity(REQUESTS)),
            pending: Mutex::new(Vec::new()),
            generations: Mutex::new(VecDeque::with_capacity(GENERATIONS)),
            log: Mutex::new(VecDeque::with_capacity(LOG_LINES)),
        }
    }

    fn now(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// Record a finished request. Called from the middleware, on the response
    /// path, so it must not do anything slow — which is why the database is
    /// somebody else's problem.
    pub fn record(&self, method: &str, path: &str, status: u16, took: Duration) {
        let request = Request {
            at_millis: self.now(),
            method: method.to_string(),
            path: path.to_string(),
            status,
            millis: took.as_secs_f64() * 1000.0,
            generation: None,
        };
        push(&self.requests, request.clone(), REQUESTS);
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).push(request);
    }

    /// Record a request that carries what it cost the engine.
    ///
    /// Called by the completions handler rather than by the middleware, for
    /// two reasons. The numbers are only known inside the handler; and for a
    /// streamed reply the middleware's timing would be time-to-first-*byte* —
    /// the moment the headers went out, which for a chat is the moment
    /// before all the work. Timing it here measures the generation.
    pub fn record_generation(
        &self,
        method: &str,
        path: &str,
        status: u16,
        took: Duration,
        g: Generation,
    ) {
        push(&self.generations, g, GENERATIONS);
        let request = Request {
            at_millis: self.now(),
            method: method.to_string(),
            path: path.to_string(),
            status,
            millis: took.as_secs_f64() * 1000.0,
            generation: Some(g),
        };
        push(&self.requests, request.clone(), REQUESTS);
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).push(request);
    }

    pub fn log_line(&self, line: String) {
        push(&self.log, line, LOG_LINES);
    }

    pub fn recent(&self, limit: usize) -> Vec<Request> {
        let held = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        held.iter().rev().take(limit).cloned().collect()
    }

    pub fn generations(&self) -> Vec<Generation> {
        self.generations.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().collect()
    }

    pub fn log(&self, limit: usize) -> Vec<String> {
        let held = self.log.lock().unwrap_or_else(|e| e.into_inner());
        held.iter().rev().take(limit).rev().cloned().collect()
    }

    /// Move everything not yet written into the database. Blocking.
    pub fn flush(&self, db: &Db) -> Res<usize> {
        let batch: Vec<Request> = std::mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()));
        if batch.is_empty() {
            return Ok(0);
        }
        // One transaction for the batch: a hundred separate inserts would be
        // a hundred fsyncs, which is the whole reason for batching.
        db.with(|c| {
            c.execute("BEGIN", [])?;
            for r in &batch {
                c.execute(
                    "INSERT INTO requests (method, path, status, millis, tokens, decode_per_sec, ttft_millis)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![
                        r.method,
                        r.path,
                        r.status as i64,
                        r.millis,
                        r.generation.map(|g| g.generated_tokens as i64),
                        r.generation.map(|g| g.decode_per_sec),
                        r.generation.map(|g| g.ttft_millis),
                    ],
                )?;
            }
            c.execute("COMMIT", [])
        })?;
        Ok(batch.len())
    }

    /// Throw away rows older than `days`.
    pub fn prune(db: &Db, days: u32) -> Res<usize> {
        db.with(|c| {
            c.execute("DELETE FROM requests WHERE at < datetime('now', ?1)", [format!("-{days} days")])
        })
    }
}

fn push<T>(store: &Mutex<VecDeque<T>>, value: T, cap: usize) {
    let mut held = store.lock().unwrap_or_else(|e| e.into_inner());
    if held.len() == cap {
        held.pop_front();
    }
    held.push_back(value);
}

/// What a set of timings looked like.
///
/// Percentiles from the samples themselves rather than from buckets: there
/// are at most a few thousand, sorting them is microseconds, and an estimate
/// would be a worse number for no saving.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Summary {
    pub count: usize,
    pub median: f64,
    pub p95: f64,
    pub worst: f64,
}

pub fn summarise(mut values: Vec<f64>) -> Summary {
    if values.is_empty() {
        return Summary::default();
    }
    values.sort_by(f64::total_cmp);
    Summary {
        count: values.len(),
        median: at(&values, 0.5),
        p95: at(&values, 0.95),
        worst: *values.last().expect("not empty"),
    }
}

/// The value at a quantile, by nearest rank. No interpolation: with a handful
/// of samples an interpolated p95 is a number that never happened.
fn at(sorted: &[f64], q: f64) -> f64 {
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Requests grouped by route, slowest first. The Monitoring page's histogram.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ByRoute {
    pub path: String,
    pub method: String,
    #[serde(flatten)]
    pub latency: Summary,
    pub errors: usize,
}

pub fn by_route(requests: &[Request]) -> Vec<ByRoute> {
    let mut groups: std::collections::HashMap<(String, String), (Vec<f64>, usize)> =
        std::collections::HashMap::new();
    for r in requests {
        let entry = groups.entry((r.method.clone(), r.path.clone())).or_default();
        entry.0.push(r.millis);
        // A 4xx is usually the client's doing and a 5xx is always ours, but
        // both are a request that did not get what it asked for.
        if r.status >= 400 {
            entry.1 += 1;
        }
    }
    let mut out: Vec<ByRoute> = groups
        .into_iter()
        .map(|((method, path), (times, errors))| ByRoute {
            path,
            method,
            latency: summarise(times),
            errors,
        })
        .collect();
    out.sort_by(|a, b| b.latency.median.total_cmp(&a.latency.median));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(path: &str, status: u16, millis: f64) -> Request {
        Request {
            at_millis: 0,
            method: "GET".into(),
            path: path.into(),
            status,
            millis,
            generation: None,
        }
    }

    /// The buffer is a window, not a log: the oldest goes when it is full.
    #[test]
    fn the_ring_buffer_keeps_the_newest_and_drops_the_oldest() {
        let m = Metrics::new();
        for i in 0..(REQUESTS + 50) {
            m.record("GET", &format!("/api/{i}"), 200, Duration::from_millis(1));
        }
        let recent = m.recent(REQUESTS * 2);
        assert_eq!(recent.len(), REQUESTS, "the buffer grew past its cap");
        // Newest first, and the fifty oldest are gone.
        assert_eq!(recent[0].path, format!("/api/{}", REQUESTS + 49));
        assert!(!recent.iter().any(|r| r.path == "/api/0"));
    }

    /// Percentiles are real samples, not interpolations between them.
    #[test]
    fn a_summary_is_made_of_numbers_that_actually_happened() {
        let s = summarise(vec![10.0, 20.0, 30.0, 40.0, 100.0]);
        assert_eq!(s.count, 5);
        assert_eq!(s.median, 30.0);
        assert_eq!(s.p95, 100.0);
        assert_eq!(s.worst, 100.0);

        // One sample is its own median and its own p95.
        let one = summarise(vec![7.0]);
        assert_eq!((one.median, one.p95, one.worst), (7.0, 7.0, 7.0));

        // None is zeroes rather than a panic or a NaN.
        let none = summarise(Vec::new());
        assert_eq!(none.count, 0);
        assert_eq!(none.median, 0.0);
    }

    #[test]
    fn routes_are_grouped_and_the_slowest_comes_first() {
        let requests = vec![
            request("/api/health", 200, 1.0),
            request("/api/health", 200, 3.0),
            request("/v1/chat/completions", 200, 900.0),
            request("/api/models", 500, 5.0),
            request("/api/models", 404, 2.0),
        ];
        let grouped = by_route(&requests);
        assert_eq!(grouped.len(), 3);
        assert_eq!(grouped[0].path, "/v1/chat/completions");
        assert_eq!(grouped[0].latency.median, 900.0);

        let models = grouped.iter().find(|g| g.path == "/api/models").unwrap();
        assert_eq!(models.errors, 2, "a 404 and a 500 are both requests that failed");
        let health = grouped.iter().find(|g| g.path == "/api/health").unwrap();
        assert_eq!(health.errors, 0);
        assert_eq!(health.latency.count, 2);
    }

    /// A generation's numbers belong to the request that produced them, and
    /// to no other.
    ///
    /// An earlier version attached them to "the most recent completion",
    /// which for a non-streamed reply was the *previous* one — the handler
    /// knows the numbers before the middleware records the row. Recording
    /// both together is the fix; this is the test that would have caught it.
    #[test]
    fn generation_numbers_belong_to_their_own_request() {
        let m = Metrics::new();
        let g = |tokens| Generation {
            prompt_tokens: 40,
            cached_tokens: 10,
            generated_tokens: tokens,
            decode_per_sec: 70.0,
            ttft_millis: 120.0,
        };
        m.record("GET", "/api/health", 200, Duration::from_millis(1));
        m.record_generation("POST", "/v1/chat/completions", 200, Duration::from_millis(900), g(64));
        m.record_generation("POST", "/v1/chat/completions", 200, Duration::from_millis(400), g(32));

        let recent = m.recent(10); // newest first
        assert_eq!(recent[0].generation.unwrap().generated_tokens, 32);
        assert_eq!(recent[0].millis, 400.0);
        assert_eq!(recent[1].generation.unwrap().generated_tokens, 64);
        assert!(recent[2].generation.is_none(), "health picked up a generation's numbers");
        assert_eq!(m.generations().len(), 2);

        // And the batch waiting for the database carries them too.
        let db = Db::in_memory().unwrap();
        assert_eq!(m.flush(&db).unwrap(), 3);
        let tokens: Vec<Option<i64>> = db
            .with(|c| {
                let mut q = c.prepare("SELECT tokens FROM requests ORDER BY id")?;
                let rows = q.query_map([], |r| r.get(0))?.collect();
                rows
            })
            .unwrap();
        assert_eq!(tokens, [None, Some(64), Some(32)]);
    }

    /// Flushing empties the pending batch, and does it in one transaction.
    #[test]
    fn flushing_writes_what_is_waiting_and_then_waits_again() {
        let db = Db::in_memory().unwrap();
        let m = Metrics::new();
        for _ in 0..10 {
            m.record("GET", "/api/health", 200, Duration::from_millis(2));
        }
        assert_eq!(m.flush(&db).unwrap(), 10);
        assert_eq!(m.flush(&db).unwrap(), 0, "the same rows were written twice");

        let rows: i64 =
            db.with(|c| c.query_row("SELECT count(*) FROM requests", [], |r| r.get(0))).unwrap();
        assert_eq!(rows, 10);

        // The ring buffer keeps them; the flush is about the database.
        assert_eq!(m.recent(100).len(), 10);
    }

    #[test]
    fn pruning_takes_the_old_rows_and_leaves_the_rest() {
        let db = Db::in_memory().unwrap();
        db.with(|c| {
            c.execute(
                "INSERT INTO requests (at, method, path, status, millis)
                 VALUES (datetime('now', '-30 days'), 'GET', '/old', 200, 1.0)",
                [],
            )?;
            c.execute(
                "INSERT INTO requests (method, path, status, millis)
                 VALUES ('GET', '/new', 200, 1.0)",
                [],
            )
        })
        .unwrap();

        assert_eq!(Metrics::prune(&db, 7).unwrap(), 1);
        let left: String =
            db.with(|c| c.query_row("SELECT path FROM requests", [], |r| r.get(0))).unwrap();
        assert_eq!(left, "/new");
    }

    /// The log tail reads oldest-first, the way a log does.
    #[test]
    fn the_log_tail_is_in_the_order_it_happened() {
        let m = Metrics::new();
        for i in 0..5 {
            m.log_line(format!("line {i}"));
        }
        assert_eq!(m.log(3), ["line 2", "line 3", "line 4"]);
        assert_eq!(m.log(100).len(), 5);
    }
}
