//! A model in several pieces, as one [`Session`].
//!
//! ```text
//!   tokens → embed → stage → stage → … → head → logits
//!            └ ends ┘                    └ ends ┘
//! ```
//!
//! The ends — the embedding and the output head — are one shard, here.
//! Between them the residual stream goes through every stage in order and
//! comes back. That is the whole of it, and everything above a session —
//! sampling, prefix reuse, the scheduler, the server — is as indifferent to
//! it as to whether the weights are quantised.
//!
//! It is the same arithmetic as the model run whole, in the same order, and
//! not merely close to it: a whole model's pass *is* `embed`, `blocks`,
//! `head` (see [`Transformer`]), and this is `embed`, one `blocks` per
//! stage, `head`. The test that holds it to that compares bits.

use super::stage::{Fault, LocalStage, Stage};
use crate::model::{pool, prefill_chunk, Session, Shard, Spec, Transformer};
use crate::quant::Precision;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A stage's error, as the rest of the engine carries errors.
fn plain(fault: Fault) -> Box<dyn std::error::Error> {
    fault
}

/// The one conversation a session holds. A stage keeps a cache for each it
/// is told of, and a session tells it of this one.
const SEQ: u64 = 0;

pub struct PipelineSession {
    /// The shard with the embedding and the head.
    ends: Arc<dyn Transformer>,
    stages: Vec<Box<dyn Stage>>,
    /// Positions every stage holds.
    len: usize,
    chunk: usize,
    precision: Precision,
    pool: rayon::ThreadPool,
}

impl PipelineSession {
    /// A pipeline over `stages`, in the order given, with `ends` at both
    /// ends of it. The stages have to be one model between them: every
    /// layer once, in order.
    pub fn new(ends: Arc<dyn Transformer>, stages: Vec<Box<dyn Stage>>, precision: Precision) -> Res<Self> {
        let spec = ends.spec();
        if !ends.splits() {
            return Err(format!("the `{}` architecture runs whole and does not split yet", spec.arch).into());
        }
        if !ends.shard().ends {
            return Err(format!("{} cannot be a pipeline's ends: it has neither the embedding nor the head", ends.shard()).into());
        }
        // The ends are checked above and counted once here, whichever stage
        // happens to hold them as well.
        let mut layers: Vec<Shard> = stages.iter().map(|s| Shard { ends: false, ..s.shard() }).collect();
        if let Some(empty) = layers.iter().find(|s| s.layers.is_empty()) {
            return Err(format!("a stage with no layers ({empty}) is a hop that does nothing").into());
        }
        layers.insert(0, Shard { layers: 0..0, ends: true });
        Shard::check(&layers, spec.n_layer)?;
        Ok(PipelineSession { ends, stages, len: 0, chunk: prefill_chunk(), precision, pool: pool() })
    }

    /// Every stage in this process: the shards of one model, in any order.
    /// Nothing crosses a wire. This is the pipeline with the network taken
    /// out, which is where to find out whether a split model is the same
    /// model before finding out whether a socket is the same socket.
    pub fn local(shards: Vec<Box<dyn Transformer>>, precision: Precision) -> Res<Self> {
        let mut shards: Vec<Arc<dyn Transformer>> = shards.into_iter().map(Arc::from).collect();
        shards.sort_by_key(|s| (s.shard().layers.start, s.shard().layers.end));
        let n_layer = shards.first().ok_or("a pipeline of no shards")?.spec().n_layer;
        Shard::check(&shards.iter().map(|s| s.shard()).collect::<Vec<_>>(), n_layer)?;
        let ends = shards.iter().find(|s| s.shard().ends).expect("checked: exactly one").clone();
        let stages = shards
            .into_iter()
            .filter(|s| !s.shard().layers.is_empty())
            .map(|s| Box::new(LocalStage::new(s)) as Box<dyn Stage>)
            .collect();
        PipelineSession::new(ends, stages, precision)
    }

    /// Read a prompt `chunk` tokens a pass: as [`crate::model::CpuSession::reading`].
    pub fn reading(mut self, chunk: usize) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    /// One pass: `tokens` through the embedding and every stage. What comes
    /// back is the residual stream, for the head to be applied to or not.
    fn pass(ends: &dyn Transformer, stages: &mut [Box<dyn Stage>], len: &mut usize, tokens: &[u32]) -> Result<Vec<f32>, Fault> {
        let m = tokens.len();
        let mut xs = ends.embed(tokens, *len);
        for stage in stages.iter_mut() {
            xs = stage.run(SEQ, *len, m, xs)?;
        }
        *len += m;
        Ok(xs)
    }
}

impl Session for PipelineSession {
    fn spec(&self) -> &Spec {
        self.ends.spec()
    }

    /// Inside the pool, as [`crate::model::CpuSession::forward`] is and for
    /// its reason: a stage in this process runs its matmuls from a pool
    /// thread, and the cold path is paid once a pass.
    ///
    /// A prompt longer than a chunk is several passes, and only the last
    /// of them ends in the head: the logits of a position in the middle of
    /// a prompt are nobody's to read.
    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let PipelineSession { ends, stages, len, chunk, pool, .. } = self;
        let logits = pool.install(|| -> Result<Vec<f32>, Fault> {
            let mut logits = Vec::new();
            let mut parts = tokens.chunks(*chunk).peekable();
            while let Some(part) = parts.next() {
                let xs = Self::pass(ends.as_ref(), stages, len, part)?;
                if parts.peek().is_none() {
                    logits = ends.head(&xs, part.len(), false);
                }
            }
            Ok(logits)
        });
        logits.map_err(plain)
    }

    fn forward_all(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let PipelineSession { ends, stages, len, chunk, pool, .. } = self;
        let all = pool.install(|| -> Result<Vec<f32>, Fault> {
            let mut all = Vec::new();
            for part in tokens.chunks(*chunk) {
                let xs = Self::pass(ends.as_ref(), stages, len, part)?;
                all.extend(ends.head(&xs, part.len(), true));
            }
            Ok(all)
        });
        all.map_err(plain)
    }

    fn cached(&self) -> usize {
        self.len
    }

    fn prefill_pass(&self, left: usize) -> usize {
        left.min(self.chunk)
    }

    /// Every stage is told, and every stage has to give the same answer: a
    /// pipeline whose stages hold different lengths of one conversation has
    /// no position to carry on from.
    fn truncate(&mut self, len: usize) -> Res<usize> {
        if len >= self.len {
            return Ok(self.len);
        }
        let mut got = None;
        for stage in self.stages.iter_mut() {
            let back = stage.truncate(SEQ, len).map_err(plain)?;
            if got.is_some_and(|g| g != back) {
                return Err(format!("{} rewound to {back} positions where the stage before it rewound to {}", stage.shard(), got.unwrap_or(0)).into());
            }
            got = Some(back);
        }
        self.len = got.unwrap_or(len);
        Ok(self.len)
    }

    fn label(&self) -> String {
        format!("cpu {} in {} stages", self.precision, self.stages.len())
    }

    fn param_count(&self) -> usize {
        self.held().0
    }

    fn weight_bytes(&self) -> usize {
        self.held().1
    }
}

impl PipelineSession {
    /// Parameters and bytes across the stages, and the ends where no stage
    /// has them: a stage that is also the ends counts them as its own.
    fn held(&self) -> (usize, usize) {
        let ends_apart = self.ends.shard().layers.is_empty();
        let start = if ends_apart { (self.ends.param_count(), self.ends.memory_bytes()) } else { (0, 0) };
        self.stages.iter().map(|s| s.holds()).fold(start, |a, b| (a.0 + b.0, a.1 + b.1))
    }
}
