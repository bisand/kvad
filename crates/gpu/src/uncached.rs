//! Reading weights past the page cache.
//!
//! A model on the GPU is read from disk once and kept whole on the device.
//! Read through a memory map, every page it touches stays in the page cache
//! after the bytes have been copied out, so a load holds two copies: the
//! weights, and the file pages they came from. A model more than half the
//! machine's memory does not fit twice, and macOS does not make room by
//! dropping the clean file pages. It compresses and swaps the weights already
//! loaded, which bf16 and Q8_0 blocks barely repay, and nothing looks wrong
//! until the GPU first touches them:
//!
//! - FLUX.1-schnell at bf16 (33.7 GB) on a 48 GB M5: its first image took 29 s
//!   to encode and 31 s for the first step; the second took 0.17 s and 1.8 s.
//! - Qwen-Image at q8 (29.5 GB, read from [`crate::qcache`]): the load
//!   compressed 36 GB, and the first image's encode and first step took 7.8 s
//!   and 11.4 s.
//!
//! So a file here is read with `pread` into a buffer of its own, which the
//! caller drops once the bytes are on the device. On macOS the file is opened
//! with `F_NOCACHE`, and two things make that flag work:
//!
//! - **The read is page-aligned at both ends, in the file and in memory.** An
//!   uncached read that is not falls back to the cache. Neither a safetensors
//!   tensor nor a cache blob starts on a page.
//! - **What the cache already holds of the file is dropped when it is
//!   opened.** An uncached read of a page the cache has is served from the
//!   cache, and marks the page used, so a checkpoint just downloaded, or read
//!   by any earlier load, stays through the whole load. With 12.5 GB of
//!   FLUX's files cached that way, the load compressed 35 GB of weights and
//!   the first image was as slow as with no `F_NOCACHE` at all.
//!   `msync(MS_INVALIDATE)` over a mapping of the file drops its clean pages,
//!   and a weight file has no other kind.
//!
//! Elsewhere a file is read the same way and cached as the system sees fit.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// The largest page size a read has to be aligned to: Apple silicon's.
const PAGE: usize = 16384;

/// `path`, opened for reading past the page cache.
pub(crate) fn open(path: &Path) -> io::Result<File> {
    let file = File::open(path)?;
    #[cfg(target_os = "macos")]
    uncache(&file).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    Ok(file)
}

/// `len` bytes at `at` in `file`.
///
/// Reads the pages that hold them into a page-aligned buffer, and derefs to
/// just the bytes asked for.
pub(crate) fn read(file: &File, at: u64, len: usize) -> io::Result<Pages> {
    let page = PAGE as u64;
    let from = at / page * page;
    let end = at.checked_add(len as u64).ok_or_else(|| io::Error::other("a span that overflows"))?;
    let size = file.metadata()?.len();
    if end > size {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("{at}..{end} runs past the end of the file")));
    }
    // The last page may be the file's, which ends where it ends.
    let to = (end.div_ceil(page) * page).min(size);
    let span = (to - from) as usize;
    let mut buf = vec![0u8; span + PAGE];
    let off = buf.as_ptr().align_offset(PAGE);
    file.read_exact_at(&mut buf[off..off + span], from)?;
    Ok(Pages { buf, at: off + (at - from) as usize, len })
}

/// Bytes read by [`read`], in the buffer they were read into.
pub(crate) struct Pages {
    buf: Vec<u8>,
    at: usize,
    len: usize,
}

impl std::ops::Deref for Pages {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.buf[self.at..self.at + self.len]
    }
}

/// Read `file` past the page cache from now on, and drop what the cache
/// already holds of it.
#[cfg(target_os = "macos")]
fn uncache(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    let len = file.metadata()?.len() as usize;
    // SAFETY: calls on a descriptor this function borrows, and a read-only
    // mapping that nothing reads and that is unmapped before returning.
    unsafe {
        if libc::fcntl(fd, libc::F_NOCACHE, 1) == -1 {
            return Err(io::Error::last_os_error());
        }
        if len == 0 {
            return Ok(());
        }
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, fd, 0);
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let r = libc::msync(p, len, libc::MS_INVALIDATE);
        let e = io::Error::last_os_error();
        libc::munmap(p, len);
        if r == -1 {
            return Err(e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every span comes back as written: at the start and the end of the
    /// file, across a page boundary, longer than a page, and empty; and a
    /// span past the end is refused.
    #[test]
    fn a_read_gives_back_the_bytes_at_any_offset() {
        let path = std::env::temp_dir().join(format!("kvad-gpu-pages-{}", std::process::id()));
        let data: Vec<u8> = (0..3 * PAGE + 123).map(|i| (i * 7 + i / 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let file = open(&path).unwrap();
        let n = data.len();
        for (at, len) in [(0, 10), (5, PAGE), (PAGE - 3, 7), (PAGE + 1, 2 * PAGE), (n - 9, 9), (0, n), (n, 0), (77, 0)] {
            assert_eq!(&*read(&file, at as u64, len).unwrap(), &data[at..at + len], "{at}..+{len}");
        }
        assert!(read(&file, (n - 4) as u64, 5).is_err(), "a span past the end is refused");
        let _ = std::fs::remove_file(&path);
    }
}
