//! One model across several machines. See `docs/cluster-plan.md`, which is
//! the design; this is as much of it as exists.
//!
//! * [`stage`] — some of a model's blocks, wherever they are, and
//!   [`session`], the pipeline of them that the rest of the engine sees as
//!   one model.
//! * [`wire`] — the framing two machines speak.
//! * [`ping`] — a listener that sends frames straight back, and the
//!   measurement of a link made against it: what a stage boundary costs
//!   before there is a stage.
//! * [`link`] — what a link *is*, asked of the machine and not of the
//!   socket, because a charging cable and a Thunderbolt cable are told apart
//!   by nothing a socket can observe.

pub mod link;
pub mod ping;
pub mod session;
pub mod stage;
pub mod wire;

/// This machine's name for itself, without its domain.
pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is as long as we say it is.
    let ok = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } == 0;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(0);
    match ok && end > 0 {
        true => String::from_utf8_lossy(&buf[..end]).split('.').next().unwrap_or("").to_string(),
        false => "this machine".into(),
    }
}
