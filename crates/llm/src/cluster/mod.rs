//! One model across several machines. See `docs/cluster-plan.md`, which is
//! the design; this is as much of it as exists.
//!
//! * [`stage`] — some of a model's blocks, wherever they are, and
//!   [`session`], the pipeline of them that the rest of the engine sees as
//!   one model.
//! * [`worker`] — the far end: a stage for a coordinator that holds the
//!   key. [`remote`] is the near end of the same socket, and [`key`] is how
//!   each shows the other it belongs.
//! * [`discover`] — a worker saying it is one, and a coordinator asking who
//!   is; [`path`] — which of a worker's addresses to use, by measuring.
//! * [`wire`] — the framing two machines speak.
//! * [`ping`] — a listener that sends frames straight back, and the
//!   measurement of a link made against it: what a stage boundary costs
//!   before there is a stage.
//! * [`link`] — what a link *is*, asked of the machine and not of the
//!   socket, because a charging cable and a Thunderbolt cable are told apart
//!   by nothing a socket can observe.

pub mod discover;
pub mod key;
pub mod link;
pub mod path;
pub mod ping;
pub mod remote;
pub mod session;
pub mod stage;
pub mod wire;
pub mod worker;

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

/// This build, as a worker and a coordinator compare it: the version, and
/// the commit where there is one. `0.18.2+8f21d2d1c4`, or `0.18.2` for a
/// build made from no repository.
pub fn build() -> String {
    match env!("KVAD_REVISION") {
        "" => env!("CARGO_PKG_VERSION").to_string(),
        revision => format!("{}+{revision}", env!("CARGO_PKG_VERSION")),
    }
}

/// Where a worker keeps the name it gives itself.
pub fn id_path() -> std::path::PathBuf {
    crate::weights::data_dir().join("machine.id")
}

/// This machine's id, if it has been given one: see [`make_id`].
pub fn id() -> Option<String> {
    id_in(&id_path())
}

fn id_in(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let id = text.trim();
    (id.len() == 16 && id.bytes().all(|b| b.is_ascii_hexdigit())).then(|| id.to_ascii_lowercase())
}

/// This machine's id, made now if it had none.
///
/// A machine is an identity and not an address, because it honestly has
/// several: a link-local one on each cable, another on Wi-Fi, a tailnet's.
/// The same id answering on four of them is one machine with four paths,
/// not four workers, and its memory must not be counted four times. So a
/// worker picks sixteen hex digits once, keeps them, and says them in
/// every greeting and every advertisement.
pub fn make_id() -> Result<String, String> {
    make_id_in(&id_path())
}

fn make_id_in(path: &std::path::Path) -> Result<String, String> {
    if let Some(id) = id_in(path) {
        return Ok(id);
    }
    let bytes = key::random::<8>().map_err(|e| format!("no random bytes for a machine id: {e}"))?;
    let id: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, format!("{id}\n")).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_build_is_its_version_and_its_commit() {
        let build = build();
        assert!(build.starts_with(env!("CARGO_PKG_VERSION")));
        // Built from a repository, as a test run is: the commit is there.
        match build.split_once('+') {
            Some((_, revision)) => assert!(revision.len() >= 10 && revision.trim_end_matches("-dirty").bytes().all(|b| b.is_ascii_hexdigit()), "{build}"),
            None => assert_eq!(build, env!("CARGO_PKG_VERSION")),
        }
    }

    #[test]
    fn a_machine_names_itself_once() {
        let path = std::env::temp_dir().join(format!("kvad-id-{}", std::process::id())).join("machine.id");
        assert_eq!(id_in(&path), None);
        let made = make_id_in(&path).unwrap();
        assert_eq!(made.len(), 16);
        assert_eq!(make_id_in(&path).unwrap(), made, "the second start reads what the first wrote");
        assert_eq!(id_in(&path), Some(made));
        std::fs::write(&path, "not an id").unwrap();
        assert_eq!(id_in(&path), None);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
