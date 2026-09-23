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
        ExpertStore::index(&container, path)
    }

    /// As [`ExpertStore::open`], for a container already open at `path`.
    pub fn index(container: &crate::qcache::Container, path: &Path) -> Res<Option<ExpertStore>> {
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

    /// Room one expert needs in memory: the widest page-aligned read, so any
    /// expert fits in any slot.
    pub fn slot_bytes(&self) -> usize {
        self.extents.iter().flatten().map(|e| e.aligned().1).max().unwrap_or(0)
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
        for (extent, slab) in want.iter().zip(slabs.iter_mut()) {
            let (_, len, _) = extent.aligned();
            if slab.len() < len {
                slab.resize(len, 0);
            }
        }
        let mut jobs: Vec<(Extent, &mut [u8])> =
            want.iter().copied().zip(slabs.iter_mut().map(|s| &mut s[..])).collect();
        self.fetch_into(&mut jobs)
    }

    /// Read each extent's aligned span into the buffer beside it, which must
    /// already be large enough for it.
    ///
    /// One request is read on the calling thread: a thread spawned to make
    /// a single read wins nothing, and a mixture whose cache is working
    /// misses one expert a layer as often as it misses several.
    pub fn fetch_into(&self, jobs: &mut [(Extent, &mut [u8])]) -> Res<u64> {
        // The page-rounded read can run past the end of the file, and needs
        // only to cover the expert itself, so a short read that does is fine.
        let read = |file: &File, extent: &Extent, buf: &mut [u8]| -> Res<u64> {
            let (base, len, skip) = extent.aligned();
            let buf = buf.get_mut(..len).ok_or("fetch: a buffer is smaller than its read")?;
            let need = skip + extent.len as usize;
            let mut got = 0;
            while got < need {
                match file.read_at(&mut buf[got..], base + got as u64) {
                    Ok(0) => return Err(format!("reading at {base}: the file ends early").into()),
                    Ok(n) => got += n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(format!("reading at {base}: {e}").into()),
                }
            }
            Ok(got as u64)
        };
        if jobs.len() <= 1 {
            return jobs.iter_mut().map(|(e, b)| read(&self.files[0], e, b)).sum::<Res<u64>>();
        }
        let moved = std::sync::atomic::AtomicU64::new(0);
        let failed: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        // As `fetch`: round robin, so every descriptor gets the same share.
        let mut shares: Vec<Vec<&mut (Extent, &mut [u8])>> =
            (0..self.files.len()).map(|_| Vec::new()).collect();
        for (i, job) in jobs.iter_mut().enumerate() {
            shares[i % self.files.len()].push(job);
        }
        std::thread::scope(|scope| {
            for (file, share) in self.files.iter().zip(shares) {
                if share.is_empty() {
                    continue;
                }
                let (moved, failed) = (&moved, &failed);
                scope.spawn(move || {
                    for job in share {
                        match read(file, &job.0, &mut job.1[..]) {
                            Ok(n) => {
                                moved.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(e) => {
                                *failed.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
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

    /// The slot a resident key occupies.
    pub fn slot(&self, key: u32) -> Option<usize> {
        self.at.get(&key).copied()
    }
}

// ---------------------------------------------------------------------------
// In the forward pass
// ---------------------------------------------------------------------------

/// Routed experts held in memory of our own, and read from the disk on a
/// miss, for a mixture larger than memory.
///
/// The alternative is what a mapping does unasked: page an expert in when a
/// matmul touches it, 16 KB at a time, one fault after another, and evict
/// by whatever the kernel's idea of recent is. This reads a whole expert in
/// one request, the misses of a layer all at once, and evicts the expert
/// wanted longest ago -- which is the policy [`crate::residency`] measured.
///
/// On by default for a model whose weights are larger than the memory this
/// machine gives them -- see [`default_budget`] -- and off otherwise.
/// `KVAD_EXPERT_CACHE` overrides both ways: a number of gigabytes to spend,
/// or `0` for none, which keeps it a one-flag A/B.
pub struct Resident {
    store: ExpertStore,
    fetcher: Fetcher,
    slots: std::sync::Mutex<Slots>,
    /// Experts that were fetched and then run from the mapping anyway,
    /// because their weights could not be pointed at the slab. Zero for any
    /// cache this engine writes; counted so that a test can say so.
    fallbacks: std::sync::atomic::AtomicU64,
}

struct Slots {
    lru: Cache,
    /// One per slot of `lru`. Refilled through `Arc::get_mut`, which is what
    /// guarantees no weight still reading an evicted expert sees it change.
    slabs: Vec<std::sync::Arc<crate::qcache::Slab>>,
    bytes: u64,
}

impl Resident {
    /// The cache this model should read its experts through, over the cache
    /// file at `path`: what `KVAD_EXPERT_CACHE` asks for, or else
    /// [`default_budget`].
    ///
    /// `None` when none is wanted, when the file holds no mixture, or when it
    /// cannot be set up -- the last said on stderr, since the mapping still
    /// runs the model and nothing here is worth failing a load over.
    pub fn wanted(container: &crate::qcache::Container, path: &Path) -> Option<std::sync::Arc<Resident>> {
        let bytes = container.bytes() as u64;
        let gb: f64 = match std::env::var("KVAD_EXPERT_CACHE") {
            Ok(v) => v.trim().parse().ok()?,
            Err(_) => {
                let usable = crate::machine::usable_memory_cached()?;
                let budget = default_budget(bytes, usable)?;
                eprintln!(
                    "expert cache: on, because {} of weights is over the {} this machine \
                     gives them (KVAD_EXPERT_CACHE=0 turns it off)",
                    crate::hub::human_bytes(bytes),
                    crate::hub::human_bytes(usable)
                );
                budget as f64 / 1e9
            }
        };
        if gb <= 0.0 {
            return None;
        }
        let store = match ExpertStore::index(container, path) {
            Ok(Some(store)) => store,
            Ok(None) => return None,
            Err(e) => {
                eprintln!("expert cache: not used: {e}");
                return None;
            }
        };
        let slot = store.slot_bytes();
        let capacity = (gb * 1e9) as usize / slot.max(1);
        if capacity == 0 {
            eprintln!("expert cache: {gb} GB holds no experts of {} MB", slot / 1_000_000);
            return None;
        }
        let threads = std::env::var("KVAD_EXPERT_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(4);
        match Resident::new(store, capacity, threads) {
            Ok(r) => {
                eprintln!(
                    "expert cache: {} of {} experts ({}), {} reader(s)",
                    r.capacity(),
                    r.store.len(),
                    crate::hub::human_bytes((r.capacity() * slot) as u64),
                    r.fetcher.threads()
                );
                Some(r)
            }
            Err(e) => {
                eprintln!("expert cache: not used: {e}");
                None
            }
        }
    }

    /// A cache of `slots` experts over `store`, read with `threads`
    /// descriptors. Never more slots than there are experts.
    ///
    /// Fewer than one layer wants is allowed, and slow: [`Resident::with`]
    /// runs a layer's experts in groups that fit, so it is never wrong.
    pub fn new(store: ExpertStore, slots: usize, threads: usize) -> Res<std::sync::Arc<Resident>> {
        let slots = slots.clamp(1, store.len().max(1));
        let fetcher = Fetcher::new(&store, threads)?;
        Ok(std::sync::Arc::new(Resident {
            slots: std::sync::Mutex::new(Slots {
                lru: Cache::new(slots),
                slabs: (0..slots).map(|_| std::sync::Arc::new(crate::qcache::Slab::new())).collect(),
                bytes: 0,
            }),
            store,
            fetcher,
            fallbacks: std::sync::atomic::AtomicU64::new(0),
        }))
    }

    /// Note an expert that ran from the mapping after all.
    pub fn fell_back(&self) {
        self.fallbacks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn fallbacks(&self) -> u64 {
        self.fallbacks.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.slots.lock().unwrap_or_else(|p| p.into_inner()).lru.capacity()
    }

    /// Hits and misses so far.
    pub fn stats(&self) -> (u64, u64) {
        let slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        (slots.lru.hits, slots.lru.misses)
    }

    /// Make `experts` of `layer` resident, and hand them to `run` a group at
    /// a time: each with the slab holding it and the file offset that slab
    /// begins at. A group together, because a token's experts run together.
    ///
    /// In groups no larger than the cache, so that fetching the end of a
    /// group never evicts its beginning before it has run: within a group
    /// every expert is the most recently wanted, and LRU evicts the rest
    /// first. Decode wants `top_k` at a time and is always one group; a long
    /// prefill can want every expert of the layer.
    pub fn with(
        &self,
        layer: usize,
        experts: &[usize],
        mut run: impl FnMut(&[(usize, &std::sync::Arc<crate::qcache::Slab>, u64)]),
    ) -> Res<()> {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        let group = slots.lru.capacity();
        for chunk in experts.chunks(group) {
            let mut misses: Vec<(usize, Extent)> = Vec::new();
            let mut found: Vec<(usize, usize, Extent)> = Vec::with_capacity(chunk.len());
            for &expert in chunk {
                let extent = self
                    .store
                    .get(layer, expert)
                    .ok_or_else(|| format!("layer {layer} has no expert {expert} in the cache file"))?;
                let key = (layer * self.store.n_experts + expert) as u32;
                let hit = slots.lru.touch(key);
                let slot = slots.lru.slot(key).ok_or("expert cache: a key has no slot")?;
                if !hit {
                    misses.push((slot, extent));
                }
                found.push((expert, slot, extent));
            }
            if !misses.is_empty() {
                let mut want: Vec<Option<Extent>> = vec![None; group];
                for &(slot, extent) in &misses {
                    want[slot] = Some(extent);
                }
                let Slots { slabs, bytes, .. } = &mut *slots;
                let mut jobs: Vec<(Extent, &mut [u8])> = Vec::with_capacity(misses.len());
                for (slab, want) in slabs.iter_mut().zip(want) {
                    let Some(extent) = want else { continue };
                    // A view outliving its expert would be a bug in `Moe::run`;
                    // a fresh slab keeps it from also being a wrong answer.
                    if std::sync::Arc::get_mut(slab).is_none() {
                        *slab = std::sync::Arc::new(crate::qcache::Slab::new());
                    }
                    let slab = std::sync::Arc::get_mut(slab).ok_or("expert cache: slab is shared")?;
                    slab.reserve(extent.aligned().1);
                    jobs.push((extent, slab.bytes_mut()));
                }
                *bytes += self.fetcher.fetch_into(&mut jobs)?;
            }
            let group: Vec<_> = found
                .iter()
                .map(|&(expert, slot, extent)| (expert, &slots.slabs[slot], extent.aligned().0))
                .collect();
            run(&group);
        }
        Ok(())
    }
}

/// The bytes to spend on an expert cache by default: half of `usable`, for
/// a model whose `bytes` of weights are more than `usable`. `None` for one
/// that fits, where the mapping holds everything and a cache would only be
/// a second copy.
///
/// Over memory, because that is where it was measured to pay: interleaved,
/// on Qwen3-Next-80B at q4 (50 GB against 36 usable), the batched decode
/// ran 11.2-12.3 tok/s with a 16 GB cache against 6.7-9.1 without, and on
/// Qwen3-30B at q8 15.4 and 16.1 against 7.5-7.9. Half, because that is
/// the size measured -- 16 GB of 36 -- and 26 GB was no better, while
/// every byte of it is memory the page cache, the KV cache and everything
/// else on the machine no longer has.
pub fn default_budget(bytes: u64, usable: u64) -> Option<u64> {
    (bytes > usable).then_some(usable / 2)
}

impl Drop for Resident {
    /// What the cache did, because the point of it is a number.
    fn drop(&mut self) {
        let slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        let (h, m) = (slots.lru.hits, slots.lru.misses);
        if h + m > 0 {
            eprintln!(
                "expert cache: {:.1}% hits ({h} of {}), {:.2} GB read",
                100.0 * h as f64 / (h + m) as f64,
                h + m,
                slots.bytes as f64 / 1e9
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Over memory it is on, at half of what the machine gives weights;
    /// otherwise off. The two models it was measured on, at the sizes they
    /// are, against this machine's 36 GB.
    #[test]
    fn the_cache_is_on_by_default_only_for_a_model_over_memory() {
        let usable = 36_000_000_000;
        assert_eq!(default_budget(49_814_745_728, usable), Some(18_000_000_000), "Qwen3-Next-80B q4");
        // Fits on paper, if only just -- and a cache would be a second copy.
        assert_eq!(default_budget(34_352_646_288, usable), None, "Qwen3-30B q8");
        assert_eq!(default_budget(usable, usable), None, "exactly full is not over");
    }

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
