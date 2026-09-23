//! Experts as storage: where each one lives, and what it costs to read back.
//!
//! [`crate::residency`] established that a mixture can keep a fraction of
//! its experts in memory and read the rest from disk, and how much of what
//! a token asks for would already be resident. It could not say what the
//! misses *cost*, because it never touched a disk: every figure it prints
//! divides bytes by a bandwidth measured on a different file in a different
//! access pattern. This module is the other half — the reads themselves.
//!
//! # There is no new file format here
//!
//! The expectation was an expert store: experts laid out one after another,
//! each aligned, each readable in a single request. [`crate::qcache`]
//! already is one, by accident of how it is written. `Moe::load` asks for
//! `gate`, `up` and `down` in that order, the writer appends each array as
//! it is produced, and the result — checked across all 6144 experts of
//! Qwen3-30B-A3B — is that every expert occupies one contiguous extent with
//! no internal gaps, and 99% of them sit immediately after the one before.
//! The exceptions are layer boundaries.
//!
//! So this indexes what is there rather than rewriting it.
//!
//! # What the disk wants
//!
//! Measured on an M5 Pro, scattered reads against a 32 GB cache file:
//!
//! ```text
//!            queue depth 1    2      4      8
//!   2 MB          5.87     11.75  13.36  13.52   GB/s
//!   5.3 MB        9.70     13.72  13.79  13.59
//! ```
//!
//! Block size matters far less than having more than one request in
//! flight. A lone 2 MB read gets 44% of the device; four of them get all of
//! it. That is the whole reason [`Fetcher`] takes a slice of experts rather
//! than one at a time — and it costs nothing to arrange, because a token
//! wants `top_k` of them at once anyway.
//!
//! # Alignment
//!
//! `F_NOCACHE` wants page-aligned offsets, and the cache file aligns its
//! arrays to 64 bytes. So a read rounds down to the page below and asks for
//! a little more than it needs: at most 4 KB of over-read against a 5.3 MB
//! expert, which is under a tenth of a per cent and not worth a new format
//! to avoid.
//!
//! The flag is not only for measuring honestly. A residency cache that
//! holds its own slabs does not want the kernel holding a second copy of
//! the same bytes, so the real implementation wants `F_NOCACHE` too — and
//! it is what stops a benchmark on a 32 GB file quietly reporting the speed
//! of a 48 GB machine's page cache.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Where one expert's weights sit in the cache file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    pub off: u64,
    pub len: u32,
}

impl Extent {
    /// The page-aligned read that contains this extent, as `(offset, len)`.
    ///
    /// `F_NOCACHE` reads want page alignment, and the cache file only
    /// promises 64 bytes, so the request is widened at both ends and the
    /// caller takes the middle.
    pub fn aligned(&self) -> (u64, usize, usize) {
        const PAGE: u64 = 4096;
        let base = self.off & !(PAGE - 1);
        let skip = (self.off - base) as usize;
        let len = ((skip as u64 + self.len as u64 + PAGE - 1) & !(PAGE - 1)) as usize;
        (base, len, skip)
    }
}

/// Every routed expert in a quantised cache file, and where it lives.
pub struct ExpertStore {
    path: PathBuf,
    /// Layer-major, `layer * n_experts + expert`. `None` for a dense layer.
    extents: Vec<Option<Extent>>,
    n_layers: usize,
    n_experts: usize,
}

impl ExpertStore {
    /// Index the experts of a `.nq` cache, or `None` if it holds no mixture.
    pub fn open(path: &Path) -> Res<Option<ExpertStore>> {
        let Some(container) = crate::qcache::Container::open(path)? else { return Ok(None) };
        let header = container.header();
        let Some(tensors) = header.get("tensors").and_then(|t| t.as_object()) else {
            return Err("quant cache has no tensor table".into());
        };

        // The header names every array, so the shape of the mixture can be
        // read off the names rather than guessed from the config — which
        // matters because a family with a dense prefix has layers that are
        // simply absent from this table.
        let (mut n_layers, mut n_experts) = (0usize, 0usize);
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for name in tensors.keys() {
            let Some((layer, expert)) = parse_expert(name) else { continue };
            n_layers = n_layers.max(layer + 1);
            n_experts = n_experts.max(expert + 1);
            seen.push((layer, expert));
        }
        if seen.is_empty() {
            return Ok(None);
        }
        seen.sort_unstable();
        seen.dedup();

        let mut extents = vec![None; n_layers * n_experts];
        for (layer, expert) in seen {
            let mut spans: Vec<(u64, u64)> = Vec::with_capacity(6);
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let name = format!("layers.{layer}.mlp.experts.{expert}.{proj}.weight");
                let Some(entry) = tensors.get(&name) else {
                    return Err(format!("expert {expert} of layer {layer} has no {proj}").into());
                };
                spans.extend(arrays(entry)?);
            }
            spans.sort_unstable();
            let lo = spans[0].0;
            let hi = spans.last().map(|(o, l)| o + l).unwrap_or(lo);
            // A gap would mean one read no longer fetches one expert, and
            // every figure downstream assumes it does. Say so rather than
            // silently reading somebody else's weights into the slab.
            let held: u64 = spans.iter().map(|(_, l)| l).sum();
            if hi - lo != held {
                return Err(format!(
                    "expert {expert} of layer {layer} is not contiguous: \
                     {held} bytes spread over {} — this wants a store that packs them",
                    hi - lo
                )
                .into());
            }
            extents[layer * n_experts + expert] = Some(Extent { off: lo, len: (hi - lo) as u32 });
        }
        Ok(Some(ExpertStore {
            path: path.to_path_buf(),
            extents,
            n_layers,
            n_experts,
        }))
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn get(&self, layer: usize, expert: usize) -> Option<Extent> {
        self.extents.get(layer * self.n_experts + expert).copied().flatten()
    }

    /// What one expert costs, if they are all the same size.
    ///
    /// They are, within a model: every routed expert of a mixture has the
    /// same shape. Returns the largest, so a slab sized by it always fits.
    pub fn expert_bytes(&self) -> u32 {
        self.extents.iter().flatten().map(|e| e.len).max().unwrap_or(0)
    }

    /// How many experts this holds, counting only layers that route.
    pub fn len(&self) -> usize {
        self.extents.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Open a descriptor that will not be cached by the kernel.
    pub fn open_uncached(&self) -> Res<File> {
        let file = File::open(&self.path)?;
        // Best effort: a kernel that declines still returns correct bytes,
        // just slower to measure and heavier on memory.
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
            libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0);
        }
        Ok(file)
    }
}

/// The `(offset, bytes)` of every array backing one tensor.
///
/// `scales` is counted in `f32`s and the code arrays in bytes, which is the
/// one place the cache's header is not self-describing.
fn arrays(entry: &serde_json::Value) -> Res<Vec<(u64, u64)>> {
    let mut out = Vec::new();
    for (key, width) in [("scales", 4u64), ("qs", 1), ("data", 4)] {
        let Some(pair) = entry.get(key).and_then(|v| v.as_array()) else { continue };
        let (Some(off), Some(len)) = (pair.first().and_then(|v| v.as_u64()),
                                      pair.get(1).and_then(|v| v.as_u64()))
        else {
            return Err(format!("malformed `{key}` in the cache header").into());
        };
        out.push((off, len * width));
    }
    if out.is_empty() {
        return Err("a tensor entry names no arrays at all".into());
    }
    Ok(out)
}

/// `layers.3.mlp.experts.7.gate_proj.weight` -> `(3, 7)`.
fn parse_expert(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, rest) = rest.split_once(".mlp.experts.")?;
    let (expert, _) = rest.split_once('.')?;
    Some((layer.parse().ok()?, expert.parse().ok()?))
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Reads experts off the disk, several at once.
///
/// The concurrency is the point. One 2 MB request gets 5.9 GB/s out of this
/// machine's NVMe and four get 13.4, so a fetcher that reads one expert at
/// a time would leave more than half the device unused — and a token
/// already wants `top_k` experts at the same moment, so there is nothing to
/// arrange.
pub struct Fetcher {
    files: Vec<File>,
}

impl Fetcher {
    /// `threads` descriptors, each of which will read independently.
    ///
    /// Four is enough on the machine this was written for; past that the
    /// device is saturated and the extra threads only add latency to each
    /// individual read.
    pub fn new(store: &ExpertStore, threads: usize) -> Res<Fetcher> {
        let threads = threads.clamp(1, 32);
        let files = (0..threads).map(|_| store.open_uncached()).collect::<Res<Vec<_>>>()?;
        Ok(Fetcher { files })
    }

    pub fn threads(&self) -> usize {
        self.files.len()
    }

    /// Read every extent into its own slab, returning the bytes moved.
    ///
    /// Each slab must hold the *aligned* read, which is up to a page larger
    /// than the extent itself; [`Extent::aligned`] says how much. The
    /// expert's own bytes begin at the returned skip within the slab.
    pub fn fetch(&self, want: &[Extent], slabs: &mut [Vec<u8>]) -> Res<u64> {
        if want.len() != slabs.len() {
            return Err("fetch: one slab per extent, please".into());
        }
        let moved = std::sync::atomic::AtomicU64::new(0);
        let failed: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

        // Round robin rather than a contiguous split: the reads are the
        // same size, so interleaving keeps every descriptor busy for the
        // same length of time.
        std::thread::scope(|scope| {
            let (moved, failed) = (&moved, &failed);
            for (t, file) in self.files.iter().enumerate() {
                let mine: Vec<(usize, Extent)> = want
                    .iter()
                    .enumerate()
                    .skip(t)
                    .step_by(self.files.len())
                    .map(|(i, e)| (i, *e))
                    .collect();
                if mine.is_empty() {
                    continue;
                }
                // Each thread owns a disjoint set of slabs, which is what
                // makes handing out raw pointers to them sound.
                let slabs = SlabsPtr(slabs.as_mut_ptr());
                scope.spawn(move || {
                    let _ = &slabs;
                    for (i, extent) in mine {
                        let (base, len, _) = extent.aligned();
                        // SAFETY: `mine` is disjoint across threads, so no
                        // two of these ever name the same slab.
                        let slab = unsafe { &mut *slabs.0.add(i) };
                        if slab.len() < len {
                            slab.resize(len, 0);
                        }
                        match file.read_at(&mut slab[..len], base) {
                            Ok(n) => {
                                moved.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(e) => {
                                *failed.lock().unwrap_or_else(|p| p.into_inner()) =
                                    Some(format!("reading at {base}: {e}"));
                                return;
                            }
                        }
                    }
                });
            }
        });

        match failed.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(e) => Err(e.into()),
            None => Ok(moved.into_inner()),
        }
    }
}

/// A pointer to the slab array, carried into the scope threads.
///
/// `&mut [Vec<u8>]` cannot be shared, and each thread touches a disjoint
/// subset, which the borrow checker has no way to be told.
#[derive(Clone, Copy)]
struct SlabsPtr(*mut Vec<u8>);
// SAFETY: the threads that receive this only ever index slabs from their
// own stride of the round robin, so no two alias.
unsafe impl Send for SlabsPtr {}
unsafe impl Sync for SlabsPtr {}

// ---------------------------------------------------------------------------
// Residency
// ---------------------------------------------------------------------------

/// A fixed set of slabs holding the experts most recently asked for.
///
/// Least-recently-used, because that is what [`crate::residency`] measured
/// and what it recommends: on two real mixtures a static hot set chosen
/// from other traffic lost to this by eleven points on one and up to
/// thirty-six on the other.
pub struct Cache {
    slabs: Vec<Vec<u8>>,
    /// Slot holding each resident key, and the clock reading when it was
    /// last wanted.
    at: HashMap<u32, usize>,
    used: Vec<u64>,
    key: Vec<u32>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

impl Cache {
    pub fn new(capacity: usize) -> Cache {
        Cache {
            slabs: vec![Vec::new(); capacity],
            at: HashMap::with_capacity(capacity * 2),
            used: vec![0; capacity],
            key: vec![u32::MAX; capacity],
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.slabs.len()
    }

    /// Note that `key` was wanted. `true` if it was already here.
    ///
    /// A miss reserves a slot but does not fill it — the caller gathers the
    /// misses and reads them together, because one read at a time is worth
    /// less than half the device.
    pub fn touch(&mut self, key: u32) -> bool {
        self.clock += 1;
        if let Some(&slot) = self.at.get(&key) {
            self.used[slot] = self.clock;
            self.hits += 1;
            return true;
        }
        self.misses += 1;
        let slot = match self.key.iter().position(|k| *k == u32::MAX) {
            Some(free) => free,
            // The victim is the slot wanted longest ago. Linear, which is
            // fine for the thousands of slots a cache this size has and
            // would not be for millions.
            None => {
                let victim = (0..self.used.len()).min_by_key(|s| self.used[*s]).unwrap_or(0);
                self.at.remove(&self.key[victim]);
                victim
            }
        };
        self.key[slot] = key;
        self.used[slot] = self.clock;
        self.at.insert(key, slot);
        false
    }

    /// The slab a resident key occupies, for a caller about to fill it.
    pub fn slab_mut(&mut self, key: u32) -> Option<&mut Vec<u8>> {
        let slot = *self.at.get(&key)?;
        Some(&mut self.slabs[slot])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_extent_widens_to_whole_pages() {
        // 64-byte aligned, as the cache file writes them.
        let e = Extent { off: 4096 * 3 + 64, len: 5_308_416 };
        let (base, len, skip) = e.aligned();
        assert_eq!(base, 4096 * 3);
        assert_eq!(skip, 64);
        assert_eq!(base % 4096, 0);
        assert_eq!(len % 4096, 0);
        assert!(len >= skip + e.len as usize, "the read must cover the extent");
        // Never more than a page of waste at each end.
        assert!(len - e.len as usize <= 2 * 4096);
    }

    #[test]
    fn an_already_aligned_extent_is_not_moved() {
        let e = Extent { off: 8192, len: 4096 };
        assert_eq!(e.aligned(), (8192, 4096, 0));
    }

    #[test]
    fn names_are_read_the_way_the_cache_writes_them() {
        assert_eq!(parse_expert("layers.3.mlp.experts.7.gate_proj.weight"), Some((3, 7)));
        assert_eq!(parse_expert("layers.0.mlp.experts.127.down_proj.weight"), Some((0, 127)));
        assert_eq!(parse_expert("layers.3.self_attn.q_proj.weight"), None);
        assert_eq!(parse_expert("layers.3.mlp.gate.weight"), None);
    }

    #[test]
    fn the_cache_evicts_what_was_wanted_longest_ago() {
        let mut c = Cache::new(2);
        assert!(!c.touch(1));
        assert!(!c.touch(2));
        assert!(c.touch(1)); // 1 is now newer than 2
        assert!(!c.touch(3)); // so 2 is the victim
        assert!(c.touch(1), "1 should have survived");
        assert!(!c.touch(2), "2 should have been evicted");
        assert_eq!(c.hits, 2);
        assert_eq!(c.misses, 4);
    }

    /// The pattern a forward pass makes: every layer in order, over and
    /// over. A cache smaller than one pass keeps nothing at all.
    #[test]
    fn a_cache_under_the_working_set_never_hits() {
        let mut c = Cache::new(3);
        for _ in 0..5 {
            for key in 0..4 {
                c.touch(key);
            }
        }
        assert_eq!(c.hits, 0);
        let mut roomy = Cache::new(4);
        for _ in 0..5 {
            for key in 0..4 {
                roomy.touch(key);
            }
        }
        assert_eq!(roomy.hits, 16, "four slots for four keys: only the first pass misses");
    }
}
