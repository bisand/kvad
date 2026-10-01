//! A ceiling on this process's memory, for anything that measures what a
//! large computation costs.
//!
//! On macOS a Metal allocation past physical memory is not refused. The
//! system compresses, then swaps, then stops answering, and takes the
//! user's other work with it: twice a probe from this repository did that
//! (`conv3d_cost` at a whole decoder stage, and one training step at
//! 512×512 before gradient checkpointing). So a probe sets its own ceiling
//! well under the machine's memory, and is killed at it by itself, while
//! the machine can still do it.
//!
//! The figure watched is the *physical footprint*, the one Activity Monitor
//! shows and `/usr/bin/time -l` reports the peak of. It counts the GPU's
//! buffers, which the resident set size does not.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The largest footprint the watching thread has seen since [`peak`] last
/// took it.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The largest footprint seen since this was last called, in bytes, by the
/// thread [`at`] starts: what a stretch of work reached while it ran, where
/// [`footprint`] afterwards only says what it left.
pub fn peak() -> u64 {
    PEAK.swap(0, Ordering::Relaxed).max(footprint())
}

/// This process's physical footprint in bytes; 0 where it cannot be read.
pub fn footprint() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
        // SAFETY: `proc_pid_rusage` fills a `rusage_info_v2` for flavour
        // `RUSAGE_INFO_V2`, and `info` is one, zeroed, that outlives the call.
        let ok = unsafe { libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V2, info.as_mut_ptr().cast()) } == 0;
        if ok {
            // SAFETY: the call succeeded, so the struct is filled.
            return unsafe { info.assume_init() }.ri_phys_footprint;
        }
    }
    0
}

/// End the process, with exit code 137 and a line saying why, as soon as
/// its footprint passes `gigabytes`. Checked fifty times a second from a
/// thread of its own, so it holds whatever the rest of the process is in
/// the middle of.
///
/// It is not exact: one allocation larger than the room left goes over
/// before it is seen. Leave that much room under the machine's memory.
pub fn at(gigabytes: f64) {
    let cap = (gigabytes * 1e9) as u64;
    std::thread::spawn(move || loop {
        let now = footprint();
        PEAK.fetch_max(now, Ordering::Relaxed);
        if now > cap {
            eprintln!("stopped: the memory footprint reached {:.1} GB, over the {gigabytes:.1} GB this run allowed itself", now as f64 / 1e9);
            std::process::exit(137);
        }
        std::thread::sleep(Duration::from_millis(20));
    });
}

#[cfg(test)]
mod tests {
    /// The footprint is read, and grows by about what is allocated and
    /// touched.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_footprint_counts_what_is_touched() {
        let before = super::footprint();
        assert!(before > 1 << 20, "the footprint reads {before}");
        // Kept from the optimiser, which would otherwise not make it.
        let block = std::hint::black_box(vec![1u8; 256 << 20]);
        let after = super::footprint();
        assert!(after >= before + (200 << 20), "256 MB touched moved the footprint from {before} to {after}");
        assert_eq!(std::hint::black_box(&block)[block.len() - 1], 1);
    }
}
