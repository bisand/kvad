//! A stage: some of a model's blocks, wherever they are.
//!
//! The pipeline hands a stage rows of the residual stream and gets rows
//! back. Whether the blocks in between are in this process or at the far
//! end of a cable is the stage's own business, which is the point of the
//! trait: [`super::session::PipelineSession`] is written once, and a model
//! on four machines is that code with a different list.
//!
//! A stage keeps the caches for its own layers and nobody else's. Nothing
//! about a cache ever crosses between stages but how long it should be.

use crate::model::{KvCache, Shard, Transformer};
use std::sync::Arc;

/// What goes wrong in a stage, which has to be able to leave the thread it
/// went wrong on: a pass runs inside a pool.
pub type Fault = Box<dyn std::error::Error + Send + Sync>;

type Res<T> = Result<T, Fault>;

pub trait Stage: Send {
    /// The layers this stage runs.
    fn shard(&self) -> Shard;

    /// Run this stage's blocks over `m` rows of conversation `seq`, the
    /// first of them at position `pos0`, and give the rows back.
    ///
    /// The position is said and not assumed. A stage counts its own, from
    /// its cache; the two disagreeing means a token was lost or run twice
    /// somewhere between here and the coordinator, and a model that carries
    /// on from there is wrong from that token to the end of the reply.
    fn run(&mut self, seq: u64, pos0: usize, m: usize, xs: Vec<f32>) -> Res<Vec<f32>>;

    /// Drop everything of conversation `seq` after `len` positions, and say
    /// how far back that got: see [`KvCache::truncate`].
    fn truncate(&mut self, seq: u64, len: usize) -> Res<usize>;

    /// Parameters held, and the bytes they take.
    fn holds(&self) -> (usize, usize);

    /// For a stage somewhere else: the passes it has run and the time it
    /// was inside its layers for them, since this was last asked. `None`
    /// from a stage in this process, whose every millisecond is already
    /// the caller's to see.
    ///
    /// A round trip to a worker is its layers' time and the hop's, and
    /// only the worker can say which is which.
    fn inside(&mut self) -> Option<super::wire::Spent> {
        None
    }

    /// Where this stage's layers are, for whoever is shown the pipeline.
    fn place(&self) -> Place {
        Place { machine: super::hostname(), address: None, link: None, hop_ms: None, chunk_ms: None }
    }

    /// Whether this stage's layers are on another machine, so that this
    /// one waits while they run.
    fn elsewhere(&self) -> bool {
        false
    }
}

/// Where a stage is: the machine, and for one that is not this machine,
/// how it is reached and what the way there measured.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Place {
    pub machine: String,
    /// As it was connected to. `None` for a stage in this process.
    pub address: Option<String>,
    /// What the link is, in the words `kvad cluster link` uses.
    pub link: Option<String>,
    /// Milliseconds there and back for 32 KB and for 2 MB, as measured
    /// when the stage was opened.
    pub hop_ms: Option<f64>,
    pub chunk_ms: Option<f64>,
}

/// A stage in this process: a shard of the model and the caches for it.
pub struct LocalStage {
    model: Arc<dyn Transformer>,
    /// A cache a conversation. A list and not a map: there are a handful.
    caches: Vec<(u64, KvCache)>,
}

impl LocalStage {
    pub fn new(model: Arc<dyn Transformer>) -> Self {
        LocalStage { model, caches: Vec::new() }
    }

    fn cache(&mut self, seq: u64) -> &mut KvCache {
        let at = match self.caches.iter().position(|(s, _)| *s == seq) {
            Some(at) => at,
            None => {
                self.caches.push((seq, KvCache::for_shard(self.model.spec(), &self.model.shard())));
                self.caches.len() - 1
            }
        };
        &mut self.caches[at].1
    }
}

impl Stage for LocalStage {
    fn shard(&self) -> Shard {
        self.model.shard()
    }

    fn run(&mut self, seq: u64, pos0: usize, m: usize, mut xs: Vec<f32>) -> Res<Vec<f32>> {
        let (e, shard) = (self.model.spec().n_embd, self.model.shard());
        if xs.len() != m * e {
            return Err(format!("{shard}: {m} rows of {e} were announced and {} numbers arrived", xs.len()).into());
        }
        let model = self.model.clone();
        let cache = self.cache(seq);
        if cache.len != pos0 {
            return Err(format!("{shard}: handed position {pos0} of a conversation it holds {} positions of", cache.len).into());
        }
        model.blocks(&mut xs, m, cache);
        Ok(xs)
    }

    fn truncate(&mut self, seq: u64, len: usize) -> Res<usize> {
        Ok(self.cache(seq).truncate(len))
    }

    fn holds(&self) -> (usize, usize) {
        (self.model.param_count(), self.model.memory_bytes())
    }
}
