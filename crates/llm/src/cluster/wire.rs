//! The framing: what two machines running one model say to each other.
//!
//! ```text
//!   u32 length   u8 kind   payload
//!   little-endian          `length` bytes
//! ```
//!
//! Nine bytes of protocol would be overstating it. A frame is its length, a
//! kind, and that many bytes; what the bytes are depends on the kind, and
//! there are two sorts:
//!
//! * **Control** frames carry JSON. They are rare — a handful a model load —
//!   and worth being able to read in a packet capture.
//! * **Tensor** frames carry a fixed header, `seq, pos0, m, n_embd`, and then
//!   `m × n_embd` little-endian `f32`. They are every token, so nothing in
//!   them is parsed.
//!
//! `seq` is which conversation the rows belong to. One conversation at a
//! time has no use for it; several at once, each at a different stage of the
//! pipeline in the same moment, cannot do without it — and a field added to
//! a wire format later is a second wire format.
//!
//! Hidden states cross as `f32` and not `f16`. Halving 32 KB that was never
//! the problem would cost the one property the split is tested on: a model
//! cut across machines answers to the bit as it does on one.
//!
//! It is `std::net` and blocking calls. A stage does one thing at a time —
//! take rows, run its layers, give rows back — and a thread that waits on a
//! socket says that more plainly than a future would.

use std::io::{self, IoSlice, Read, Write};

/// Bumped when a frame changes shape. Two ends that disagree refuse each
/// other at `Hello`, before either has sent anything it would misread.
pub const PROTOCOL: u32 = 1;

/// Where a worker listens unless told otherwise.
pub const PORT: u16 = 7420;

/// The length and the kind.
pub const HEADER: usize = 5;

/// What a tensor frame says before its floats: `seq u64, pos0 u32, m u32,
/// n_embd u32`.
pub const TENSOR_HEADER: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// Who this is: JSON, [`Hello`]. The first frame each way.
    Hello = 1,
    /// Why this end is hanging up: JSON, `{"error": "..."}`.
    Error = 2,
    /// Bytes to send straight back, as a [`Kind::Pong`]: a stage boundary
    /// with no stage behind it, which is how a link is measured.
    Ping = 3,
    Pong = 4,
    /// Bytes to count and drop, answered by one [`Kind::Ack`]: the link's
    /// ceiling in one direction.
    Stream = 5,
    Ack = 6,
    /// Rows for a stage's layers, and the rows that come back.
    Forward = 7,
    Hidden = 8,
    /// Proof of the cluster key: JSON, `{"proof": "..."}`. See
    /// [`super::key`].
    Auth = 9,
    /// Which part of which model to hold: JSON, [`Load`]. Answered by any
    /// number of [`Kind::Progress`] and then a [`Kind::Ready`].
    Load = 10,
    /// A line about a load in progress: JSON, `{"message": "..."}`.
    Progress = 11,
    /// The stage is loaded: JSON, [`Ready`].
    Ready = 12,
    /// Drop a conversation's cache past a length: JSON, [`Truncate`]. The
    /// answer is a [`Kind::Truncated`] of the same shape, with how far back
    /// it got.
    Truncate = 13,
    Truncated = 14,
    /// Let the stage go. Answered by an [`Kind::Ack`].
    Unload = 15,
}

impl Kind {
    pub fn from_u8(byte: u8) -> Option<Kind> {
        Some(match byte {
            1 => Kind::Hello,
            2 => Kind::Error,
            3 => Kind::Ping,
            4 => Kind::Pong,
            5 => Kind::Stream,
            6 => Kind::Ack,
            7 => Kind::Forward,
            8 => Kind::Hidden,
            9 => Kind::Auth,
            10 => Kind::Load,
            11 => Kind::Progress,
            12 => Kind::Ready,
            13 => Kind::Truncate,
            14 => Kind::Truncated,
            15 => Kind::Unload,
            _ => return None,
        })
    }
}

fn invalid(what: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.into())
}

/// Write one frame, header and payload in a single call where the socket
/// will take them.
///
/// Two writes would be two packets: with `TCP_NODELAY` set — and it has to
/// be, or Nagle holds a 32 KB hop's tail back for an acknowledgement — five
/// bytes of header would cross alone and the payload after them.
pub fn write(w: &mut impl Write, kind: Kind, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len()).map_err(|_| invalid("a frame holds less than 4 GB"))?;
    let mut head = [0u8; HEADER];
    head[..4].copy_from_slice(&len.to_le_bytes());
    head[4] = kind as u8;
    let sent = w.write_vectored(&[IoSlice::new(&head), IoSlice::new(payload)])?;
    if sent < HEADER {
        w.write_all(&head[sent..])?;
        w.write_all(payload)?;
    } else {
        w.write_all(&payload[sent - HEADER..])?;
    }
    w.flush()
}

/// Read a frame's header: its kind, and how many bytes follow.
///
/// The payload is left on the socket, because what to do with it depends on
/// the kind: a `Stream` is counted and dropped a piece at a time, and never
/// held.
pub fn read_header(r: &mut impl Read) -> io::Result<(Kind, usize)> {
    let mut head = [0u8; HEADER];
    r.read_exact(&mut head)?;
    let len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let kind = Kind::from_u8(head[4]).ok_or_else(|| invalid(format!("frame kind {} is none of ours", head[4])))?;
    Ok((kind, len))
}

/// Read a payload into `buf`, which is reused from one frame to the next.
pub fn read_payload(r: &mut impl Read, len: usize, buf: &mut Vec<u8>) -> io::Result<()> {
    buf.resize(len, 0);
    r.read_exact(buf)
}

/// Read and drop `len` bytes.
pub fn skip(r: &mut impl Read, len: usize) -> io::Result<()> {
    let copied = io::copy(&mut r.take(len as u64), &mut io::sink())?;
    match copied == len as u64 {
        true => Ok(()),
        false => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the peer hung up in the middle of a frame")),
    }
}

/// A header and a payload no longer than `most`, which is how a listener
/// declines to allocate whatever a stranger asks for.
pub fn read(r: &mut impl Read, most: usize, buf: &mut Vec<u8>) -> io::Result<Kind> {
    let (kind, len) = read_header(r)?;
    if len > most {
        return Err(invalid(format!("a {kind:?} frame of {len} bytes, where {most} is the most taken")));
    }
    read_payload(r, len, buf)?;
    Ok(kind)
}

// ---------------------------------------------------------------------------
// Control frames
// ---------------------------------------------------------------------------

/// The first frame each way: who is on the other end.
///
/// Every field but the first has a default, so that a later build can say
/// more without an earlier one refusing to listen.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Hello {
    pub protocol: u32,
    /// The build: the version and, where there is one, the commit. Two
    /// builds with different kernels can both be right and not agree to
    /// the bit, so a coordinator wants these equal. See [`super::build`].
    #[serde(default)]
    pub version: String,
    /// The machine, whatever address it answered on: see
    /// [`super::make_id`]. Empty from one that has never been a worker.
    #[serde(default)]
    pub id: String,
    /// The machine's own name for itself.
    #[serde(default)]
    pub name: String,
    /// What it could give a model's weights, in bytes.
    #[serde(default)]
    pub memory: Option<u64>,
    #[serde(default)]
    pub threads: usize,
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Random, and new for each connection: what the other end's proof of
    /// the key is made over. Empty from an end that is only measuring.
    #[serde(default)]
    pub nonce: String,
}

impl Hello {
    /// This process, on this machine.
    pub fn ours() -> Hello {
        Hello {
            protocol: PROTOCOL,
            version: super::build(),
            id: super::id().unwrap_or_default(),
            name: super::hostname(),
            memory: crate::machine::usable_memory_cached(),
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            architectures: crate::model::arch::registry().iter().map(|a| a.id().to_string()).collect(),
            nonce: String::new(),
        }
    }

    pub fn send(&self, w: &mut impl Write) -> io::Result<()> {
        write(w, Kind::Hello, &serde_json::to_vec(self).map_err(|e| invalid(e.to_string()))?)
    }

    pub fn parse(payload: &[u8]) -> io::Result<Hello> {
        let hello: Hello = serde_json::from_slice(payload).map_err(|e| invalid(format!("not a greeting: {e}")))?;
        match hello.protocol == PROTOCOL {
            true => Ok(hello),
            false => Err(invalid(format!(
                "{} speaks protocol {} and this build speaks {PROTOCOL}; both ends want the same kvad",
                if hello.name.is_empty() { "the peer" } else { &hello.name },
                hello.protocol
            ))),
        }
    }
}

/// Say why, as the last frame before hanging up.
pub fn send_error(w: &mut impl Write, why: &str) -> io::Result<()> {
    write(w, Kind::Error, serde_json::json!({ "error": why }).to_string().as_bytes())
}

/// What an `Error` frame said.
pub fn error_text(payload: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| v["error"].as_str().map(str::to_string))
        .unwrap_or_else(|| String::from_utf8_lossy(payload).into_owned())
}

/// What a worker is asked to hold.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Load {
    /// A Hub repo id, or the worker's own name for a model. Never a path.
    pub repo: String,
    /// `f32`, `q8` or `q4`.
    pub precision: String,
    /// The layers, first and one past the last. A worker never holds the
    /// ends: those stay with the tokens.
    pub layers: [usize; 2],
}

/// A loaded stage, and what it holds.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Ready {
    pub params: usize,
    pub bytes: usize,
}

/// A conversation, and the length to cut its cache to — or, coming back,
/// the length it was cut to.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Truncate {
    pub seq: u64,
    pub len: usize,
}

/// Send a control frame.
pub fn send<T: serde::Serialize>(w: &mut impl Write, kind: Kind, body: &T) -> io::Result<()> {
    write(w, kind, &serde_json::to_vec(body).map_err(|e| invalid(e.to_string()))?)
}

/// Read a control frame's body.
pub fn parse<T: serde::de::DeserializeOwned>(kind: Kind, payload: &[u8]) -> io::Result<T> {
    serde_json::from_slice(payload).map_err(|e| invalid(format!("a {kind:?} frame that does not read: {e}")))
}

/// `host:port`, `[v6]:port`, or a bare host — which for an IPv6 address
/// scoped to its interface is the only spelling without brackets, since it
/// is full of colons already.
pub fn host_port(text: &str) -> (String, u16) {
    let text = text.trim();
    if let Some((host, port)) = text.strip_prefix('[').and_then(|rest| rest.split_once("]:")) {
        if let Ok(port) = port.parse() {
            return (host.to_string(), port);
        }
    }
    match text.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => match port.parse() {
            Ok(port) => (host.to_string(), port),
            Err(_) => (text.to_string(), PORT),
        },
        _ => (text.trim_start_matches('[').trim_end_matches(']').to_string(), PORT),
    }
}

// ---------------------------------------------------------------------------
// Tensor frames
// ---------------------------------------------------------------------------

/// Rows of the residual stream, and whose they are.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    /// Which conversation.
    pub seq: u64,
    /// The position of the first row in it.
    pub pos0: u32,
    /// How many rows, each `n_embd` wide.
    pub m: u32,
    pub n_embd: u32,
    pub xs: Vec<f32>,
}

impl Rows {
    /// The payload of a `Forward` or a `Hidden`, written into `out`.
    pub fn encode(&self, out: &mut Vec<u8>) {
        assert_eq!(self.xs.len(), self.m as usize * self.n_embd as usize, "rows and their shape disagree");
        out.clear();
        out.reserve(TENSOR_HEADER + self.xs.len() * 4);
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.pos0.to_le_bytes());
        out.extend_from_slice(&self.m.to_le_bytes());
        out.extend_from_slice(&self.n_embd.to_le_bytes());
        for x in &self.xs {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// Read a payload back. The floats come out as the bits that went in:
    /// a NaN keeps its payload and a negative zero its sign.
    pub fn decode(payload: &[u8]) -> io::Result<Rows> {
        if payload.len() < TENSOR_HEADER {
            return Err(invalid(format!("a tensor frame of {} bytes has no room for its header", payload.len())));
        }
        let u32_at = |i: usize| u32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]);
        let seq = u64::from_le_bytes(payload[..8].try_into().expect("eight bytes"));
        let (pos0, m, n_embd) = (u32_at(8), u32_at(12), u32_at(16));
        let body = &payload[TENSOR_HEADER..];
        let want = (m as usize).checked_mul(n_embd as usize).and_then(|n| n.checked_mul(4));
        if want != Some(body.len()) {
            return Err(invalid(format!("{m} rows of {n_embd} floats, and {} bytes of them", body.len())));
        }
        let xs = body.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        Ok(Rows { seq, pos0, m, n_embd, xs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn a_frame_comes_back_as_it_went() {
        let mut wire = Vec::new();
        write(&mut wire, Kind::Ping, b"hidden state").unwrap();
        write(&mut wire, Kind::Ack, b"").unwrap();
        assert_eq!(wire.len(), HEADER + 12 + HEADER);
        assert_eq!(&wire[..HEADER], &[12, 0, 0, 0, Kind::Ping as u8]);

        let (mut r, mut buf) = (Cursor::new(wire), Vec::new());
        assert_eq!(read(&mut r, 64, &mut buf).unwrap(), Kind::Ping);
        assert_eq!(buf, b"hidden state");
        assert_eq!(read(&mut r, 64, &mut buf).unwrap(), Kind::Ack);
        assert!(buf.is_empty());
        assert_eq!(read(&mut r, 64, &mut buf).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    /// A writer that takes a few bytes a call, as a socket with a full
    /// buffer does.
    struct Trickle(Vec<u8>, usize);

    impl Write for Trickle {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let n = buf.len().min(self.1);
            self.0.extend_from_slice(&buf[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_frame_survives_a_writer_that_takes_it_in_pieces() {
        let payload: Vec<u8> = (0..=255).collect();
        // Less than the header, exactly the header, and into the payload.
        for step in [1, 3, 5, 7, 300] {
            let mut w = Trickle(Vec::new(), step);
            write(&mut w, Kind::Pong, &payload).unwrap();
            let mut buf = Vec::new();
            assert_eq!(read(&mut Cursor::new(w.0), 256, &mut buf).unwrap(), Kind::Pong, "{step} bytes a write");
            assert_eq!(buf, payload, "{step} bytes a write");
        }
    }

    #[test]
    fn a_frame_too_long_or_of_no_kind_is_refused_before_it_is_read() {
        let mut wire = Vec::new();
        write(&mut wire, Kind::Ping, &[0; 100]).unwrap();
        let mut buf = Vec::new();
        let err = read(&mut Cursor::new(&wire), 99, &mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(buf.is_empty(), "nothing is allocated for a frame that is refused");

        // An HTTP request, which is what a stranger on the port sends.
        let err = read(&mut Cursor::new(b"GET / HTTP/1.1\r\n"), 1 << 20, &mut buf).unwrap_err();
        assert!(err.to_string().contains("none of ours"), "{err}");
    }

    #[test]
    fn a_stream_is_dropped_and_a_short_one_noticed() {
        let mut r = Cursor::new(vec![7u8; 1000]);
        skip(&mut r, 600).unwrap();
        assert_eq!(skip(&mut r, 600).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn a_greeting_is_read_by_a_build_that_knows_less_and_one_that_knows_more() {
        let ours = Hello::ours();
        let mut wire = Vec::new();
        ours.send(&mut wire).unwrap();
        let mut buf = Vec::new();
        assert_eq!(read(&mut Cursor::new(wire), 1 << 16, &mut buf).unwrap(), Kind::Hello);
        assert_eq!(Hello::parse(&buf).unwrap(), ours);
        assert!(!ours.name.is_empty() && ours.threads > 0);

        // Fewer fields than we send, and more.
        let less = Hello::parse(br#"{"protocol": 1}"#).unwrap();
        assert_eq!((less.name.as_str(), less.memory, less.threads), ("", None, 0));
        assert!(Hello::parse(br#"{"protocol": 1, "name": "air", "backends": ["gpu-q8"]}"#).is_ok());

        let err = Hello::parse(br#"{"protocol": 2, "name": "air"}"#).unwrap_err();
        assert!(err.to_string().contains("air speaks protocol 2"), "{err}");
        assert!(Hello::parse(b"hello").is_err());
    }

    #[test]
    fn an_address_is_read_with_its_port_or_without() {
        assert_eq!(host_port("air.local:7421"), ("air.local".to_string(), 7421));
        assert_eq!(host_port("10.0.0.7"), ("10.0.0.7".to_string(), PORT));
        assert_eq!(host_port("fe80::ca1:57c7%en13"), ("fe80::ca1:57c7%en13".to_string(), PORT));
        assert_eq!(host_port("[fe80::ca1:57c7%en13]:7000"), ("fe80::ca1:57c7%en13".to_string(), 7000));
        assert_eq!(host_port("[::1]"), ("::1".to_string(), PORT));
        assert_eq!(host_port(" air:x "), ("air:x".to_string(), PORT));
    }

    #[test]
    fn control_frames_keep_their_shape() {
        let load = Load { repo: "Qwen/Qwen2.5-1.5B-Instruct".into(), precision: "q8".into(), layers: [14, 28] };
        let mut wire = Vec::new();
        send(&mut wire, Kind::Load, &load).unwrap();
        let mut buf = Vec::new();
        assert_eq!(read(&mut Cursor::new(wire), 1 << 16, &mut buf).unwrap(), Kind::Load);
        assert_eq!(parse::<Load>(Kind::Load, &buf).unwrap(), load);
        assert!(parse::<Ready>(Kind::Ready, &buf).is_err());
        for byte in 1..=15u8 {
            assert_eq!(Kind::from_u8(byte).map(|k| k as u8), Some(byte));
        }
        assert_eq!(Kind::from_u8(16), None);
    }

    #[test]
    fn an_error_says_why() {
        let mut wire = Vec::new();
        send_error(&mut wire, "a \"quoted\" reason").unwrap();
        let mut buf = Vec::new();
        assert_eq!(read(&mut Cursor::new(wire), 1 << 16, &mut buf).unwrap(), Kind::Error);
        assert_eq!(error_text(&buf), "a \"quoted\" reason");
        assert_eq!(error_text(b"not json"), "not json");
    }

    #[test]
    fn rows_cross_to_the_bit() {
        // The values a conversion would lose: a NaN's payload, the sign of
        // zero, a subnormal, and the two infinities.
        let odd = [f32::from_bits(0x7fc0_1234), -0.0, f32::from_bits(1), f32::INFINITY, f32::NEG_INFINITY, 1.0 / 3.0];
        let rows = Rows { seq: u64::MAX - 1, pos0: 4096, m: 2, n_embd: 3, xs: odd.to_vec() };
        let mut payload = Vec::new();
        rows.encode(&mut payload);
        assert_eq!(payload.len(), TENSOR_HEADER + 24);
        let back = Rows::decode(&payload).unwrap();
        assert_eq!((back.seq, back.pos0, back.m, back.n_embd), (rows.seq, 4096, 2, 3));
        let bits = |xs: &[f32]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&back.xs), bits(&rows.xs));

        // No rows is a frame too: a stage asked to run nothing.
        let none = Rows { seq: 0, pos0: 0, m: 0, n_embd: 8192, xs: Vec::new() };
        none.encode(&mut payload);
        assert_eq!(Rows::decode(&payload).unwrap(), none);
    }

    #[test]
    fn rows_that_disagree_with_their_shape_are_refused() {
        let rows = Rows { seq: 1, pos0: 0, m: 2, n_embd: 4, xs: vec![0.5; 8] };
        let mut payload = Vec::new();
        rows.encode(&mut payload);
        assert!(Rows::decode(&payload[..payload.len() - 1]).is_err());
        assert!(Rows::decode(&payload[..TENSOR_HEADER - 1]).is_err());
        // A shape whose size does not fit in a usize's worth of bytes.
        payload[12..20].copy_from_slice(&[0xff; 8]);
        assert!(Rows::decode(&payload).is_err());
    }
}
