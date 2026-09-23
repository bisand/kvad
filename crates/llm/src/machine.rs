//! What this machine has, for the questions that need an answer before a
//! download rather than after one.
//!
//! One number so far: how much memory there is. A search result that says a
//! model is 16 GB is useful; one that says it will not fit is the answer
//! somebody was actually looking for.

/// Total physical memory, or `None` where we do not know how to ask.
///
/// *Total*, not free. Free memory on a machine with a page cache is a number
/// that means very little — most of it is reclaimable — and a model's weights
/// are a long-lived allocation that the operating system will make room for.
/// What matters is whether the machine is big enough at all, which is a
/// question about the total.
pub fn total_memory() -> Option<u64> {
    imp::total_memory()
}

/// How much of it is worth promising to a model's weights.
///
/// Everything else on the machine needs some, the KV cache needs more as a
/// conversation grows, and an allocation that just fits is an allocation that
/// swaps. Three quarters is a guess, and it is a stated one rather than a
/// hidden one.
pub const USABLE_FRACTION: f64 = 0.75;

pub fn usable_memory() -> Option<u64> {
    total_memory().map(|b| (b as f64 * USABLE_FRACTION) as u64)
}

/// As [`usable_memory`], answered once.
///
/// How much memory a machine has does not change while the process runs,
/// and reading it costs a `sysctl` — which is a process spawn, and was
/// being paid once per row of a forty-row search.
pub fn usable_memory_cached() -> Option<u64> {
    static ONCE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *ONCE.get_or_init(usable_memory)
}

#[cfg(target_os = "macos")]
mod imp {
    pub fn total_memory() -> Option<u64> {
        // `sysctl hw.memsize`, without the dependency: the CLI prints the same
        // number this call returns.
        let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        String::from_utf8(out.stdout).ok()?.trim().parse().ok()
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn total_memory() -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        // "MemTotal:       16316948 kB"
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    pub fn total_memory() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever this machine is, the answer has to be plausible — and on a
    /// platform we cannot ask, honestly absent rather than zero.
    #[test]
    fn memory_is_a_believable_number_or_nothing() {
        match total_memory() {
            Some(bytes) => {
                assert!(bytes >= 1 << 30, "{bytes} bytes of RAM is not a machine that runs this");
                assert!(bytes < 1 << 44, "{bytes} bytes is 16 TB; something parsed wrong");
                assert!(usable_memory().unwrap() < bytes);
            }
            // Only a platform we have no way to ask may answer nothing.
            None => {
                let askable = cfg!(any(target_os = "macos", target_os = "linux"));
                assert!(!askable, "this platform can be asked and did not answer");
            }
        }
    }
}
