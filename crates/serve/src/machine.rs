//! What this process and this disk are using.
//!
//! Small and per-platform, because "how much memory am I using" has no
//! portable answer and the ones that look portable are usually the wrong
//! number. `ru_maxrss` from `getrusage`, for instance, is the *peak* — it
//! never goes down, so a dashboard drawn from it would show a model that had
//! been unloaded as still resident.

/// Bytes this process currently has resident, or `None` where we cannot ask.
pub fn resident_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `proc_pidinfo` writes at most `size_of::<proc_taskinfo>()`
        // bytes into the buffer, which is exactly what is given, and returns
        // how many it wrote.
        unsafe {
            let mut info: libc::proc_taskinfo = std::mem::zeroed();
            let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
            let written = libc::proc_pidinfo(
                std::process::id() as libc::c_int,
                libc::PROC_PIDTASKINFO,
                0,
                &mut info as *mut _ as *mut libc::c_void,
                size,
            );
            (written == size).then_some(info.pti_resident_size)
        }
    }
    #[cfg(target_os = "linux")]
    {
        // `/proc/self/statm` is pages: total, resident, shared, …
        let text = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = text.split_whitespace().nth(1)?.parse().ok()?;
        // SAFETY: `sysconf` reads a constant and cannot fail in a way that
        // matters; a negative answer is filtered below.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (page > 0).then(|| pages * page as u64)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// What the models and their quantised weights are costing on disk.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Disk {
    /// Downloaded from the Hub. Can always be fetched again.
    pub downloaded: u64,
    /// Trained here. Cannot.
    pub trained: u64,
    /// Pre-quantised weights, which are derived and always safe to delete.
    pub quantised: u64,
    pub datasets: u64,
    /// The sum, so that a client showing only one number does not have to
    /// know what the parts are — and cannot get the arithmetic wrong when a
    /// part is added.
    pub total: u64,
}

impl Disk {
    fn summed(self) -> Disk {
        Disk {
            total: self.downloaded + self.trained + self.quantised + self.datasets,
            ..self
        }
    }
}

/// Walk everything kvad keeps. Blocking — several directory trees.
pub fn disk() -> Disk {
    let sum = |models: Vec<kvad::hub::LocalModel>| models.iter().map(|m| m.bytes).sum();
    Disk {
        downloaded: sum(kvad::hub::local_models()),
        trained: sum(kvad::hub::trained_models()),
        quantised: kvad::qcache::entries().iter().map(|(_, _, _, bytes)| bytes).sum(),
        datasets: dir_bytes(&crate::datasets::dir()),
        total: 0,
    }
    .summed()
}

fn dir_bytes(path: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number has to be current rather than peak, and plausible: this
    /// test binary is more than a megabyte and less than the machine.
    #[test]
    fn resident_memory_is_a_believable_number() {
        let Some(bytes) = resident_bytes() else {
            // A platform we cannot ask. Saying nothing is the right answer;
            // inventing a number is not.
            return;
        };
        assert!(bytes > 1 << 20, "{bytes} bytes resident is too little to be true");
        assert!(bytes < 1 << 40, "{bytes} bytes resident is too much to be true");
    }

    #[test]
    fn disk_totals_are_the_sum_of_their_parts() {
        let d = Disk { downloaded: 1, trained: 2, quantised: 4, datasets: 8, total: 0 }.summed();
        assert_eq!(d.total, 15);
        assert_eq!(Disk::default().summed().total, 0);
        // A directory that is not there is nothing, not an error.
        assert_eq!(dir_bytes(std::path::Path::new("/no/such/directory")), 0);
    }
}
