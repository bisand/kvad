//! Measure a link the way a layer split will use it.
//!
//! A stage boundary is a round trip: the coordinator hands a stage its rows
//! and waits for them to come back. So that is what is timed — a frame out
//! and the same frame back — and not one-way bandwidth, which is this
//! link's ceiling and says nothing about what a 32 KB hop pays before its
//! first byte moves. For decode that is the whole cost.
//!
//! Two sizes are the ones the plan has to report: `n_embd` floats for a
//! decode hop, and a prefill chunk of them. They are measured inside a
//! sweep, because every "X is slower than Y" in this repository has turned
//! out to be a crossover, and the one that matters here is where a frame
//! stops paying latency and starts paying bandwidth.

use super::wire::{self, Hello, Kind};
use std::io::{self, Read};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// A decode hop for a 70B: 8192 floats.
pub const DECODE_HOP: usize = 32 << 10;
/// A prefill chunk of them: `KVAD_PREFILL_CHUNK`'s default of 64 rows.
pub const PREFILL_CHUNK: usize = 2 << 20;

pub const SIZES: [usize; 8] = [1 << 10, 1 << 12, 1 << 14, DECODE_HOP, 1 << 17, 1 << 19, PREFILL_CHUNK, 1 << 23];

/// The longest frame the listener sends back. It holds what it echoes, and
/// it listens to anybody.
pub const MOST_ECHOED: usize = 64 << 20;

/// How long either end waits on a silent socket before giving it up.
const PATIENCE: Duration = Duration::from_secs(30);

fn other(what: impl Into<String>) -> io::Error {
    io::Error::other(what.into())
}

// ---------------------------------------------------------------------------
// The far end
// ---------------------------------------------------------------------------

/// What the listener has to say about a connection.
pub enum Event<'a> {
    Came(SocketAddr, &'a Hello),
    Went(SocketAddr, String),
}

/// Answer one coordinator at a time, for ever.
///
/// One at a time because that is the shape of a stage, and because two
/// measurements sharing a link would each record the other.
pub fn listen(listener: &TcpListener, said: &mut dyn FnMut(Event)) -> io::Result<()> {
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let Ok(from) = stream.peer_addr() else { continue };
        let ended = match answer(&mut stream, &mut |hello| said(Event::Came(from, hello))) {
            Ok(()) => "done".to_string(),
            Err(e) => {
                // Best effort: the socket may be the thing that failed.
                let _ = wire::send_error(&mut stream, &e.to_string());
                // Then take what it was still sending, briefly. Closing on
                // unread bytes resets the connection, and a reset can
                // overtake the reason and arrive instead of it.
                let _ = stream.shutdown(std::net::Shutdown::Write);
                let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
                let _ = io::copy(&mut (&stream).take(1 << 20), &mut io::sink());
                e.to_string()
            }
        };
        said(Event::Went(from, ended));
    }
    Ok(())
}

/// One connection, until its other end hangs up.
pub fn answer(stream: &mut TcpStream, greeted: &mut dyn FnMut(&Hello)) -> io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(PATIENCE))?;
    stream.set_write_timeout(Some(PATIENCE))?;
    let mut buf = Vec::new();

    match wire::read(stream, 1 << 16, &mut buf)? {
        Kind::Hello => greeted(&Hello::parse(&buf)?),
        kind => return Err(other(format!("a {kind:?} before any greeting"))),
    }
    Hello::ours().send(stream)?;

    loop {
        let (kind, len) = match wire::read_header(stream) {
            Ok(header) => header,
            // Hanging up between frames is how a measurement ends.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        match kind {
            Kind::Ping if len <= MOST_ECHOED => {
                wire::read_payload(stream, len, &mut buf)?;
                wire::write(stream, Kind::Pong, &buf)?;
            }
            Kind::Ping => return Err(other(format!("a ping of {len} bytes; {MOST_ECHOED} is the most sent back"))),
            Kind::Stream => {
                wire::skip(stream, len)?;
                wire::write(stream, Kind::Ack, &[])?;
            }
            kind => return Err(other(format!("a {kind:?} frame, which a ping listener has no answer to"))),
        }
    }
}

// ---------------------------------------------------------------------------
// The near end
// ---------------------------------------------------------------------------

/// Connect, and exchange greetings.
///
/// `host` is whatever `getaddrinfo` takes, so a link-local address scoped
/// to its interface — `fe80::1c%en12` — works as it is written.
pub fn connect(host: &str, port: u16) -> io::Result<(TcpStream, Hello)> {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let mut last = other(format!("{host} names no address"));
    for addr in (host, port).to_socket_addrs()? {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(5)) {
            Ok(mut stream) => {
                stream.set_nodelay(true)?;
                stream.set_read_timeout(Some(PATIENCE))?;
                stream.set_write_timeout(Some(PATIENCE))?;
                Hello::ours().send(&mut stream)?;
                let mut buf = Vec::new();
                return match wire::read(&mut stream, 1 << 16, &mut buf)? {
                    Kind::Hello => Ok((stream, Hello::parse(&buf)?)),
                    Kind::Error => Err(other(format!("the listener refused: {}", wire::error_text(&buf)))),
                    kind => Err(other(format!("a {kind:?} where a greeting was due; is that a kvad listener?"))),
                };
            }
            Err(e) => last = e,
        }
    }
    Err(last)
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Rounds that count, after one that does not.
    pub rounds: usize,
    pub sizes: Vec<usize>,
    /// Bytes sent one way, for the ceiling.
    pub stream: usize,
    /// How long to leave the link idle before each frame. Nothing, for a
    /// sweep back to back; a token's worth of milliseconds, to measure a
    /// hop as a model takes one, which is a frame, a wait while somebody
    /// computes, and a frame.
    pub gap: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Options { rounds: 7, sizes: SIZES.to_vec(), stream: 64 << 20, gap: Duration::ZERO }
    }
}

/// What a sweep found.
#[derive(Clone, Debug)]
pub struct Measured {
    pub sizes: Vec<usize>,
    /// Milliseconds there and back, a list for each size.
    pub samples: Vec<Vec<f64>>,
    pub stream_bytes: usize,
    pub stream_ms: f64,
}

/// Bytes nothing can compress, in case anything on the path tries.
fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        // xorshift64: nothing here needs to be random, only incompressible.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

/// Walk every size once a round, not every round once a size.
///
/// Load on a machine swings enough to move a timing by a quarter, so a size
/// measured all in one stretch records that stretch and not the link.
/// Interleaving spreads each size over the whole run. The first round is
/// thrown away: it pays for neighbour discovery, for the window opening and
/// for the buffers growing to fit.
pub fn measure(stream: &mut TcpStream, options: &Options, round_done: &mut dyn FnMut(usize)) -> io::Result<Measured> {
    if let Some(big) = options.sizes.iter().find(|&&n| n > MOST_ECHOED) {
        return Err(other(format!("{big} bytes is more than a listener sends back ({MOST_ECHOED})")));
    }
    let blobs: Vec<Vec<u8>> = options.sizes.iter().map(|&n| noise(n, 0x6b76_6164 + n as u64)).collect();
    let mut samples = vec![Vec::with_capacity(options.rounds); options.sizes.len()];
    let mut back = Vec::new();
    for round in 0..=options.rounds {
        for (blob, samples) in blobs.iter().zip(&mut samples) {
            if !options.gap.is_zero() {
                std::thread::sleep(options.gap);
            }
            let start = Instant::now();
            wire::write(stream, Kind::Ping, blob)?;
            let kind = wire::read(stream, MOST_ECHOED, &mut back)?;
            let ms = start.elapsed().as_secs_f64() * 1e3;
            match kind {
                Kind::Pong if back == *blob => {}
                Kind::Pong => return Err(other(format!("{} bytes came back changed", blob.len()))),
                Kind::Error => return Err(other(format!("the listener gave up: {}", wire::error_text(&back)))),
                kind => return Err(other(format!("a {kind:?} where the ping was due back"))),
            }
            if round > 0 {
                samples.push(ms);
            }
        }
        round_done(round);
    }

    let blob = noise(options.stream, 0x73_7472);
    let start = Instant::now();
    wire::write(stream, Kind::Stream, &blob)?;
    match wire::read(stream, 1 << 16, &mut back)? {
        Kind::Ack => {}
        Kind::Error => return Err(other(format!("the listener gave up: {}", wire::error_text(&back)))),
        kind => return Err(other(format!("a {kind:?} where the stream's acknowledgement was due"))),
    }
    let stream_ms = start.elapsed().as_secs_f64() * 1e3;
    Ok(Measured { sizes: options.sizes.clone(), samples, stream_bytes: options.stream, stream_ms })
}

pub fn median(xs: &[f64]) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

impl Measured {
    pub fn median(&self, i: usize) -> f64 {
        median(&self.samples[i])
    }

    pub fn least(&self, i: usize) -> f64 {
        self.samples[i].iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// What the link carried for size `i`, in Mb/s: the bytes cross twice.
    pub fn rate(&self, i: usize) -> f64 {
        2.0 * self.sizes[i] as f64 * 8.0 / (self.median(i) / 1e3) / 1e6
    }

    /// Least squares of the median round trip against size: `rtt = a +
    /// 2·bytes/B`. Returns `a` in milliseconds and `B` in Mb/s.
    ///
    /// `a` is what a hop costs carrying nothing, which is the number decode
    /// lives on. `B` is what the link does once a frame is big enough to
    /// hide `a`, which is the number prefill lives on. Either alone
    /// describes the link wrongly.
    pub fn fit(&self) -> (f64, f64) {
        let xs: Vec<f64> = self.sizes.iter().map(|&n| n as f64).collect();
        let ys: Vec<f64> = (0..self.sizes.len()).map(|i| self.median(i)).collect();
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 }; // ms a byte, both ways
        let mbit = if slope > 0.0 { 2.0 * 8.0 / slope / 1e3 } else { f64::INFINITY };
        (my - slope * mx, mbit)
    }

    /// One way, in Mb/s.
    pub fn ceiling(&self) -> f64 {
        self.stream_bytes as f64 * 8.0 / (self.stream_ms / 1e3) / 1e6
    }
}

/// `16k,32k,2m` as bytes, for looking closely at one part of the sweep.
///
/// The default sweep is spaced widely enough to find a crossover and too
/// widely to describe one. When a size misbehaves the answer is more sizes
/// around it, not more repetitions of the same one.
pub fn parse_sizes(text: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for word in text.split(',').map(str::trim).filter(|w| !w.is_empty()) {
        let lower = word.to_ascii_lowercase();
        let (digits, scale) = match lower.as_bytes().last() {
            Some(b'k') => (&lower[..lower.len() - 1], 1usize << 10),
            Some(b'm') => (&lower[..lower.len() - 1], 1 << 20),
            _ => (lower.as_str(), 1),
        };
        let n: f64 = digits.parse().map_err(|_| format!("`{word}` is not a size; write 32k, 2m or a number of bytes"))?;
        if !(n >= 0.0) || n * scale as f64 > MOST_ECHOED as f64 {
            return Err(format!("`{word}` is outside what a listener sends back, 0 to {} MB", MOST_ECHOED >> 20));
        }
        out.push((n * scale as f64) as usize);
    }
    match out.is_empty() {
        true => Err("no sizes; write them as 16k,32k,2m".into()),
        false => Ok(out),
    }
}

/// `32 KB`, `2 MB`, and bytes below a kilobyte.
pub fn size_label(n: usize) -> String {
    match n {
        n if n >= 1 << 20 && n % (1 << 20) == 0 => format!("{} MB", n >> 20),
        n if n >= 1 << 20 => format!("{:.1} MB", n as f64 / (1 << 20) as f64),
        n if n >= 1 << 10 && n % (1 << 10) == 0 => format!("{} KB", n >> 10),
        n if n >= 1 << 10 => format!("{:.1} KB", n as f64 / 1024.0),
        n => format!("{n} B"),
    }
}

/// A listener on any free port of this machine, answering in a thread: what
/// the tests measure against, and nothing else's business.
#[cfg(test)]
fn local() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || listen(&listener, &mut |_| {}));
    port
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_sweep_over_loopback_times_every_size_every_round() {
        let (mut stream, peer) = connect("127.0.0.1", local()).unwrap();
        assert_eq!(peer, Hello::ours());
        let options = Options { rounds: 3, sizes: vec![0, 1, 1 << 10, DECODE_HOP, 300_000], stream: 1 << 20, gap: std::time::Duration::ZERO };
        let mut rounds = Vec::new();
        let got = measure(&mut stream, &options, &mut |r| rounds.push(r)).unwrap();
        assert_eq!(rounds, [0, 1, 2, 3], "the warm-up is a round too, and is said");
        assert_eq!(got.sizes, options.sizes);
        for (i, samples) in got.samples.iter().enumerate() {
            assert_eq!(samples.len(), 3, "the warm-up round is not kept");
            assert!(samples.iter().all(|&ms| ms > 0.0 && ms < 5_000.0));
            assert!(got.least(i) <= got.median(i));
        }
        assert!(got.stream_ms > 0.0 && got.ceiling() > 0.0);
        // The same connection measures again: nothing was left on the wire.
        assert!(measure(&mut stream, &Options { rounds: 1, sizes: vec![64], stream: 0, gap: std::time::Duration::ZERO }, &mut |_| {}).is_ok());
    }

    #[test]
    fn a_listener_refuses_what_it_will_not_hold_and_what_it_does_not_know() {
        let port = local();
        // Too long to send back: said in a frame, and the line closed.
        let (mut stream, _) = connect("127.0.0.1", port).unwrap();
        let mut head = ((MOST_ECHOED + 1) as u32).to_le_bytes().to_vec();
        head.push(Kind::Ping as u8);
        stream.write_all(&head).unwrap();
        let mut buf = Vec::new();
        assert_eq!(wire::read(&mut stream, 1 << 16, &mut buf).unwrap(), Kind::Error);
        assert!(wire::error_text(&buf).contains("the most sent back"), "{}", wire::error_text(&buf));
        assert_eq!(stream.read(&mut [0; 1]).unwrap(), 0, "and it hangs up");

        // A frame that is a stage's, sent to a listener that has no stage.
        let (mut stream, _) = connect("127.0.0.1", port).unwrap();
        wire::write(&mut stream, Kind::Forward, &[0; 20]).unwrap();
        assert_eq!(wire::read(&mut stream, 1 << 16, &mut buf).unwrap(), Kind::Error);

        // Something that is not kvad at all.
        let mut stranger = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stranger.write_all(b"GET / HTTP/1.1\r\n\r\n").unwrap();
        assert_eq!(wire::read(&mut stranger, 1 << 16, &mut buf).unwrap(), Kind::Error);

        // And after all that it still answers the next one.
        let (mut stream, _) = connect("127.0.0.1", port).unwrap();
        assert!(measure(&mut stream, &Options { rounds: 1, sizes: vec![1024], stream: 1024, gap: std::time::Duration::ZERO }, &mut |_| {}).is_ok());
    }

    #[test]
    fn a_peer_of_another_protocol_is_told_so_by_both_ends() {
        // The listener's side.
        let mut stream = TcpStream::connect(("127.0.0.1", local())).unwrap();
        wire::write(&mut stream, Kind::Hello, br#"{"protocol": 99, "name": "later"}"#).unwrap();
        let mut buf = Vec::new();
        assert_eq!(wire::read(&mut stream, 1 << 16, &mut buf).unwrap(), Kind::Error);
        assert!(wire::error_text(&buf).contains("later speaks protocol 99"));

        // The connecting side, against a listener from another time.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            wire::read(&mut stream, 1 << 16, &mut buf).unwrap();
            wire::write(&mut stream, Kind::Hello, br#"{"protocol": 0, "name": "earlier"}"#).unwrap();
        });
        let err = connect("127.0.0.1", port).unwrap_err();
        assert!(err.to_string().contains("earlier speaks protocol 0"), "{err}");
    }

    #[test]
    fn nothing_listening_is_an_error_and_not_a_wait() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let start = Instant::now();
        assert!(connect("127.0.0.1", port).is_err());
        assert!(start.elapsed() < Duration::from_secs(4));
    }

    #[test]
    fn the_fit_finds_the_latency_and_the_bandwidth_it_was_given() {
        // 1.1 ms empty and 300 Mb/s, the charging cable's own numbers.
        let sizes = SIZES.to_vec();
        let rtt = |n: usize| 1.1 + 2.0 * n as f64 * 8.0 / 300e6 * 1e3;
        let samples = sizes.iter().map(|&n| vec![rtt(n) * 1.5, rtt(n), rtt(n)]).collect();
        let got = Measured { sizes, samples, stream_bytes: 64 << 20, stream_ms: (64 << 20) as f64 * 8.0 / 300e6 * 1e3 };
        let (a, b) = got.fit();
        assert!((a - 1.1).abs() < 1e-6 && (b - 300.0).abs() < 1e-3, "{a} ms, {b} Mb/s");
        assert!((got.ceiling() - 300.0).abs() < 1e-6);
        // A 32 KB hop on that link moves at well under its bandwidth.
        let hop = got.sizes.iter().position(|&n| n == DECODE_HOP).unwrap();
        assert!(got.rate(hop) < 200.0 && got.rate(got.sizes.len() - 1) > 295.0);
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
    }

    #[test]
    fn sizes_are_read_as_they_are_written() {
        assert_eq!(parse_sizes("16k, 32K,2m,100").unwrap(), [16 << 10, 32 << 10, 2 << 20, 100]);
        assert_eq!(parse_sizes("1.5k").unwrap(), [1536]);
        for bad in ["", "big", "65m", "-1k", "12q"] {
            assert!(parse_sizes(bad).is_err(), "{bad}");
        }
        assert_eq!([size_label(512), size_label(32 << 10), size_label(1536), size_label(2 << 20)], ["512 B", "32 KB", "1.5 KB", "2 MB"]);
    }

    #[test]
    fn noise_is_as_long_as_asked_and_does_not_repeat() {
        assert!(noise(0, 1).is_empty());
        let a = noise(4099, 7);
        assert_eq!(a.len(), 4099);
        assert_ne!(a[..2048], a[2048..4096]);
        assert_ne!(a, noise(4099, 9));
    }
}
