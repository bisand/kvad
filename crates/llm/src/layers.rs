//! A dense model larger than memory, read off the disk a layer at a time.
//!
//! [`crate::experts`] does this for a mixture's routed experts, where a token
//! touches a handful of them. A dense model touches every weight every token,
//! so there is nothing to cache: what helps is reading each layer in as few,
//! large, concurrent requests as the disk likes, *before* the layer is wanted,
//! into memory of our own that is reused rather than left to the page cache.
//!
//! The mapping already runs such a model, and keeps it out of the footprint:
//! its pages are the file's, and the kernel drops them. What it does not do
//! is read well. A token walks the weights in order and the mapping faults
//! them in 16 KB at a time as the matmuls reach them, and a file bigger than
//! memory evicts each layer before the next token comes back for it.
//!
//! # What this reuses
//!
//! No new format: [`crate::qcache`] writes a layer's arrays one after another,
//! because a loader asks for them a layer at a time, so a layer is one extent
//! of the cache file, as an expert is. The reads are [`Fetcher`]'s, uncached
//! and several at once; the memory is [`Slab`]s; and a layer's weights are
//! pointed at its slab with [`crate::quant::Weight::rebased`], as an expert's
//! are. The arithmetic does not change by a bit.
//!
//! # Read-ahead
//!
//! One thread reads; the forward pass asks for layer `l`, and that asks for
//! the next [`ahead`] layers too, wrapping round to the first, since the next
//! token starts there. A layer is waited for only if its read has not
//! finished. The slabs are a pool of at most `ahead + 1` in use, so the
//! footprint is a few layers whatever the model's size.

use crate::experts::{Extent, Fetcher};
use crate::qcache::{Container, Slab};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{mpsc, Arc, Condvar, Mutex};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One request's size. Several in flight is what the disk wants (see
/// [`crate::experts`]); a page multiple keeps every one aligned.
const CHUNK: usize = 8 << 20;
const PAGE: u64 = 4096;

/// How many layers to read ahead of the one running. `KVAD_LAYER_AHEAD`.
fn ahead() -> usize {
    std::env::var("KVAD_LAYER_AHEAD").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}

/// The layer a cache entry belongs to: `backbone.layers.3.mixer.in_proj.weight`
/// and `layers.3.self_attn.q_proj.weight` are 3.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("backbone.").or_else(|| name.strip_prefix("model.")).unwrap_or(name);
    rest.strip_prefix("layers.")?.split('.').next()?.parse().ok()
}

/// Each layer's page-aligned span of the cache file, `(base, len)`, where its
/// arrays are one run with nobody else's between them.
fn spans(container: &Container) -> Res<Vec<Option<(u64, usize)>>> {
    let tensors = container.header().get("tensors").and_then(|t| t.as_object()).ok_or("quant cache has no tensor table")?;
    let mut all: Vec<(u64, u64, Option<usize>)> = Vec::new();
    for (name, entry) in tensors {
        let layer = layer_of(name);
        all.extend(crate::experts::arrays(entry)?.into_iter().map(|(off, len)| (off, len, layer)));
    }
    all.sort_unstable();
    let n = all.iter().filter_map(|a| a.2).max().map_or(0, |m| m + 1);
    let mut out = vec![None; n];
    let mut i = 0;
    while i < all.len() {
        let Some(layer) = all[i].2 else {
            i += 1;
            continue;
        };
        let start = all[i].0;
        let mut end = start + all[i].1;
        while i + 1 < all.len() && all[i + 1].2 == Some(layer) {
            i += 1;
            end = end.max(all[i].0 + all[i].1);
        }
        // A layer seen twice is not one run: leave it to the mapping.
        out[layer] = match out[layer] {
            None => {
                let base = start & !(PAGE - 1);
                Some((base, ((end - base + PAGE - 1) & !(PAGE - 1)) as usize))
            }
            Some(_) => return Err(format!("layer {layer} is not one run of the cache file").into()),
        };
        i += 1;
    }
    Ok(out)
}

struct State {
    /// Layers read, or why one could not be.
    have: HashMap<usize, Result<Arc<Slab>, String>>,
    asked: HashSet<usize>,
    spare: Vec<Slab>,
    read: u64,
    fell_back: u64,
}

struct Shared {
    fetcher: Fetcher,
    spans: Vec<Option<(u64, usize)>>,
    state: Mutex<State>,
    ready: Condvar,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Read `layer` into a spare slab, on the reading thread.
    fn read(&self, layer: usize) {
        let Some((base, len)) = self.spans.get(layer).copied().flatten() else { return };
        let mut slab = self.lock().spare.pop().unwrap_or_default();
        slab.reserve(len);
        let bytes = &mut slab.bytes_mut()[..len];
        let mut jobs: Vec<(Extent, &mut [u8])> = bytes
            .chunks_mut(CHUNK)
            .enumerate()
            .map(|(i, buf)| (Extent { off: base + (i * CHUNK) as u64, len: buf.len() as u32 }, buf))
            .collect();
        let got = self.fetcher.fetch_into(&mut jobs).map_err(|e| e.to_string());
        let mut state = self.lock();
        match got {
            Ok(n) => {
                state.read += n;
                state.have.insert(layer, Ok(Arc::new(slab)));
            }
            Err(e) => {
                state.spare.push(slab);
                state.have.insert(layer, Err(e));
            }
        }
        self.ready.notify_all();
    }
}

/// The layers of one model, streamed. See the module documentation.
pub struct Layers {
    shared: Arc<Shared>,
    ask: mpsc::Sender<usize>,
    /// The layers to stream, in the order a token runs them.
    order: Vec<usize>,
    ahead: usize,
}

impl Layers {
    /// Stream `which` of the cache file at `path`: what `KVAD_LAYER_STREAM`
    /// asks for, or by default when the file is larger than the memory this
    /// machine gives it. `None` when not wanted, or when the file cannot be
    /// streamed (said on stderr: the mapping still runs the model).
    pub fn wanted(container: &Container, path: &Path, which: &[usize]) -> Option<Arc<Layers>> {
        let on = match std::env::var("KVAD_LAYER_STREAM").as_deref() {
            Ok("0") | Ok("false") => false,
            Ok("1") | Ok("true") => true,
            _ => {
                let usable = crate::machine::usable_memory_cached()?;
                let big = container.bytes() as u64 > usable;
                if big && !which.is_empty() {
                    eprintln!(
                        "layer stream: on, because {} of weights is over the {} this machine gives \
                         them (KVAD_LAYER_STREAM=0 turns it off)",
                        crate::hub::human_bytes(container.bytes() as u64),
                        crate::hub::human_bytes(usable)
                    );
                }
                big
            }
        };
        if !on || which.is_empty() {
            return None;
        }
        match Layers::new(container, path, which, ahead()) {
            Ok(layers) => Some(Arc::new(layers)),
            Err(e) => {
                eprintln!("layer stream: not used: {e}");
                None
            }
        }
    }

    pub fn new(container: &Container, path: &Path, which: &[usize], ahead: usize) -> Res<Layers> {
        let spans = spans(container)?;
        if let Some(&l) = which.iter().find(|&&l| spans.get(l).copied().flatten().is_none()) {
            return Err(format!("layer {l} has no weights in the cache file").into());
        }
        let shared = Arc::new(Shared {
            fetcher: Fetcher::of(path, 4)?,
            spans,
            state: Mutex::new(State {
                have: HashMap::new(),
                asked: HashSet::new(),
                spare: Vec::new(),
                read: 0,
                fell_back: 0,
            }),
            ready: Condvar::new(),
        });
        let (ask, asked) = mpsc::channel::<usize>();
        let reader = Arc::clone(&shared);
        std::thread::Builder::new().name("kvad-layers".into()).spawn(move || {
            // Ends when the `Layers` is dropped, and its sender with it.
            while let Ok(layer) = asked.recv() {
                reader.read(layer);
            }
        })?;
        Ok(Layers { shared, ask, order: which.to_vec(), ahead })
    }

    fn request(&self, state: &mut State, layer: usize) {
        if state.asked.insert(layer) && self.ask.send(layer).is_err() {
            state.have.insert(layer, Err("the reading thread is gone".into()));
        }
    }

    /// Layer `layer`'s bytes, as `(file offset the slab starts at, slab)`,
    /// once read; and the next layers asked for while it is used. `None` if
    /// it could not be read, and the caller runs it from the mapping.
    pub fn get(&self, layer: usize) -> Option<(u64, Arc<Slab>)> {
        let mut state = self.shared.lock();
        self.request(&mut state, layer);
        if let Some(at) = self.order.iter().position(|&l| l == layer) {
            for k in 1..=self.ahead.min(self.order.len() - 1) {
                self.request(&mut state, self.order[(at + k) % self.order.len()]);
            }
        }
        loop {
            match state.have.get(&layer) {
                Some(Ok(slab)) => return Some((self.shared.spans[layer]?.0, Arc::clone(slab))),
                Some(Err(_)) => return None,
                None => state = self.shared.ready.wait(state).unwrap_or_else(|p| p.into_inner()),
            }
        }
    }

    /// Done with `layer` for this pass: its slab goes back to the pool, to be
    /// read into again. Every weight pointed at it must be dropped by now,
    /// or the slab is let go instead.
    pub fn done(&self, layer: usize) {
        let mut state = self.shared.lock();
        state.asked.remove(&layer);
        if let Some(Ok(slab)) = state.have.remove(&layer) {
            if let Ok(slab) = Arc::try_unwrap(slab) {
                state.spare.push(slab);
            }
        }
    }

    /// A weight that could not be pointed at its slab, and ran mapped.
    pub fn fell_back(&self) {
        self.shared.lock().fell_back += 1;
    }

    /// Bytes read so far, and weights that ran mapped instead.
    pub fn counts(&self) -> (u64, u64) {
        let state = self.shared.lock();
        (state.read, state.fell_back)
    }
}

impl Drop for Layers {
    fn drop(&mut self) {
        let (read, fell) = self.counts();
        if read > 0 {
            eprintln!("layer stream: {:.2} GB read, {fell} weight(s) ran mapped instead", read as f64 / 1e9);
        }
    }
}

#[cfg(test)]
#[test]
fn layer_names_are_read_in_either_spelling() {
    assert_eq!(layer_of("backbone.layers.3.mixer.in_proj.weight"), Some(3));
    assert_eq!(layer_of("layers.12.self_attn.q_proj.weight"), Some(12));
    assert_eq!(layer_of("backbone.embeddings.weight"), None);
    assert_eq!(layer_of("mtp.layers.0.mixer.q_proj.weight"), None);
}
