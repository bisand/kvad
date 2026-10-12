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

use super::key::Key;
use super::{discover, path, plan};
use super::remote::RemoteStage;
use super::stage::{Fault, LocalStage, Stage};
use super::wire::Load;
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
    /// Passes each stage has run and the milliseconds they took as seen
    /// from here, and the same for the two ends: see
    /// [`PipelineSession::timing`].
    took: Vec<(u64, f64)>,
    ends_took: (u64, f64),
    /// The same, since the model was loaded and never emptied, and what
    /// each worker has said of its own time: what [`live`] shows.
    life: Vec<Life>,
    /// What the workers have said they spent inside their layers since
    /// [`PipelineSession::timing`] last asked.
    inside: Vec<Option<f64>>,
    /// When the workers were last asked, which is not every token.
    asked: std::time::Instant,
    /// Where this pipeline's report is kept for [`live`] to read.
    board: Option<Arc<std::sync::Mutex<Report>>>,
    /// A pass failed part way down the pipeline, so the stages before the
    /// failure hold rows the ones after it never saw.
    torn: bool,
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
        let (took, life, inside) = (vec![(0, 0.0); stages.len()], vec![Life::default(); stages.len()], vec![None; stages.len()]);
        Ok(PipelineSession {
            ends,
            stages,
            len: 0,
            took,
            ends_took: (0, 0.0),
            life,
            inside,
            asked: std::time::Instant::now(),
            board: None,
            torn: false,
            chunk: prefill_chunk(),
            precision,
            pool: pool(),
        })
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
    fn pass(ends: &dyn Transformer, stages: &mut [Box<dyn Stage>], took: &mut [(u64, f64)], life: &mut [Life], len: &mut usize, tokens: &[u32]) -> Result<Vec<f32>, Fault> {
        let m = tokens.len();
        let mut xs = ends.embed(tokens, *len);
        for ((stage, took), life) in stages.iter_mut().zip(took.iter_mut()).zip(life.iter_mut()) {
            let started = std::time::Instant::now();
            // While another machine has the rows this one has nothing to
            // do, and would be slow to start again: see `linger`.
            let awake = stage.elsewhere().then(|| crate::model::linger(None)).flatten();
            xs = stage.run(SEQ, *len, m, xs)?;
            drop(awake);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            *took = (took.0 + 1, took.1 + ms);
            life.saw(m, ms);
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
        self.mend()?;
        if self.pipes(tokens) {
            return self.piped(tokens, false);
        }
        let PipelineSession { ends, stages, len, chunk, pool, torn, took, ends_took, life, .. } = self;
        let logits = pool.install(|| -> Result<Vec<f32>, Fault> {
            let mut logits = Vec::new();
            let mut parts = tokens.chunks(*chunk).peekable();
            while let Some(part) = parts.next() {
                let xs = Self::pass(ends.as_ref(), stages, took, life, len, part)?;
                if parts.peek().is_none() {
                    let started = std::time::Instant::now();
                    logits = ends.head(&xs, part.len(), false);
                    *ends_took = (ends_took.0 + 1, ends_took.1 + started.elapsed().as_secs_f64() * 1e3);
                }
            }
            Ok(logits)
        });
        *torn = logits.is_err();
        self.post();
        logits.map_err(plain)
    }

    fn forward_all(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        self.mend()?;
        if self.pipes(tokens) {
            return self.piped(tokens, true);
        }
        let PipelineSession { ends, stages, len, chunk, pool, torn, took, life, .. } = self;
        let all = pool.install(|| -> Result<Vec<f32>, Fault> {
            let mut all = Vec::new();
            for part in tokens.chunks(*chunk) {
                let xs = Self::pass(ends.as_ref(), stages, took, life, len, part)?;
                all.extend(ends.head(&xs, part.len(), true));
            }
            Ok(all)
        });
        *torn = all.is_err();
        self.post();
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
        self.mend()?;
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

/// What a stage has done since its model was loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Life {
    /// Passes of one row — tokens written — and their milliseconds as the
    /// pipeline saw them: layers and round trip both.
    pub one_runs: u64,
    pub one_ms: f64,
    /// Passes of several rows — chunks of a prompt — the tokens in them,
    /// and their milliseconds.
    pub many_runs: u64,
    pub many_tokens: u64,
    pub many_ms: f64,
    /// Of each, what a worker says it was inside its layers. `None` for a
    /// stage here, and for a worker that has not been asked yet.
    pub inside_one_ms: Option<f64>,
    pub inside_many_ms: Option<f64>,
    /// The first three numbers as they stood when the worker was last
    /// asked. A worker is asked now and then and not every token, so what
    /// it has said is behind what the pipeline has seen; the hop is the
    /// difference of two numbers taken at the same moment, and these are
    /// the ones taken with the worker's.
    pub asked_one_runs: u64,
    pub asked_one_ms: f64,
    pub asked_many_ms: f64,
}

impl Life {
    fn saw(&mut self, rows: usize, ms: f64) {
        match rows {
            1 => (self.one_runs, self.one_ms) = (self.one_runs + 1, self.one_ms + ms),
            _ => (self.many_runs, self.many_tokens, self.many_ms) = (self.many_runs + 1, self.many_tokens + rows as u64, self.many_ms + ms),
        }
    }
}

/// One stage of a model that is loaded across machines.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Staged {
    /// The layers, first and one past the last.
    pub layers: [usize; 2],
    /// Whether they are on the machine that is showing this.
    pub here: bool,
    #[serde(flatten)]
    pub place: super::stage::Place,
    pub params: usize,
    pub bytes: usize,
    #[serde(flatten)]
    pub life: Life,
}

/// A model loaded across machines, as it is now.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Report {
    pub repo: String,
    pub precision: String,
    /// Positions of the conversation every stage holds.
    pub positions: usize,
    pub stages: Vec<Staged>,
}

/// Every pipeline in this process that asked to be shown.
static BOARDS: std::sync::Mutex<Vec<std::sync::Weak<std::sync::Mutex<Report>>>> = std::sync::Mutex::new(Vec::new());

/// The models loaded across machines in this process, as they are now: a
/// pipeline's own account of itself, kept where something that is not the
/// thread running it can read it.
pub fn live() -> Vec<Report> {
    let Ok(boards) = BOARDS.lock() else { return Vec::new() };
    boards.iter().filter_map(|shown| shown.upgrade()).filter_map(|shown| shown.lock().ok().map(|report| report.clone())).collect()
}

/// Where a pipeline's time went, a stage at a time.
#[derive(Clone, Debug)]
pub struct Timing {
    pub shard: Shard,
    /// Passes, and the milliseconds they took as the pipeline saw them.
    pub runs: u64,
    pub ms: f64,
    /// For a stage on another machine: the milliseconds of those that the
    /// machine says it was inside its layers. The rest was the hop.
    pub inside_ms: Option<f64>,
}

/// A chunk of a prompt on its way down the pipeline: which chunk, where
/// in the conversation its first row is, how many rows, and the rows.
type Chunk = (usize, usize, usize, Vec<f32>);

impl PipelineSession {
    /// Whether a prompt is worth reading as a pipeline: more than one
    /// chunk of it, more than one stage, and a stage somewhere else to do
    /// one chunk while this machine does another. With every stage in
    /// this process they would only be taking turns at the same cores.
    ///
    /// `KVAD_PIPELINE=0` reads every prompt a chunk at a time, for telling
    /// what this is worth.
    fn pipes(&self, tokens: &[u32]) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("KVAD_PIPELINE").as_deref() == Ok("0"));
        !off && tokens.len() > self.chunk && self.stages.len() > 1 && self.stages.iter().any(|stage| stage.elsewhere())
    }

    /// Read a prompt with every stage at work at once.
    ///
    /// One token cannot be hurried by a second machine: its rows go through
    /// the stages one after another, and the others wait. A prompt can. It
    /// is read in chunks, and a chunk that has left the first stage leaves
    /// that stage free, so the first stage takes the second chunk while the
    /// second stage has the first. Each stage has a thread here that takes
    /// chunks in order from the stage before and hands them on, and with
    /// stages enough and prompt enough every machine is busy the whole way
    /// through — which is the one place a cluster is faster than its parts
    /// in a row, and not only larger.
    ///
    /// It is the same arithmetic in the same order as reading the chunks
    /// one at a time. A stage still sees them in order, each after the one
    /// before it is in that stage's cache; what changes is only that stage
    /// two need not have finished a chunk for stage one to start the next.
    ///
    /// Between stages there is room for one chunk and no more. A prompt of
    /// thirty thousand tokens is five hundred chunks, and embedded all at
    /// once and left in a queue they would be a gigabyte waiting.
    fn piped(&mut self, tokens: &[u32], all: bool) -> Res<Vec<f32>> {
        use std::sync::mpsc::sync_channel;
        let PipelineSession { ends, stages, len, chunk, pool, torn, took, ends_took, life, .. } = self;
        let (ends, pool): (&dyn Transformer, &rayon::ThreadPool) = (ends.as_ref(), pool);
        let parts: Vec<&[u32]> = tokens.chunks(*chunk).collect();
        let (start, last) = (*len, parts.len() - 1);

        let (read, out, fault) = std::thread::scope(|scope| {
            let (first, mut from_before) = sync_channel::<Chunk>(1);
            let mut working = Vec::with_capacity(stages.len());
            for ((stage, took), life) in stages.iter_mut().zip(took.iter_mut()).zip(life.iter_mut()) {
                let (to_next, from_this) = sync_channel::<Chunk>(1);
                let arriving = std::mem::replace(&mut from_before, from_this);
                working.push(scope.spawn(move || -> Result<(), Fault> {
                    for (i, pos0, m, xs) in arriving {
                        let started = std::time::Instant::now();
                        let xs = match stage.elsewhere() {
                            // Somebody else's cores: this thread waits on a
                            // socket, and keeps this machine awake to do it.
                            true => {
                                let _awake = crate::model::linger(Some(pool));
                                stage.run(SEQ, pos0, m, xs)?
                            }
                            false => pool.install(|| stage.run(SEQ, pos0, m, xs))?,
                        };
                        let ms = started.elapsed().as_secs_f64() * 1e3;
                        *took = (took.0 + 1, took.1 + ms);
                        life.saw(m, ms);
                        // Nobody left to hand it to: a later stage failed.
                        if to_next.send((i, pos0, m, xs)).is_err() {
                            break;
                        }
                    }
                    Ok(())
                }));
            }
            // The chunks, embedded as the first stage has room for them.
            scope.spawn(move || {
                let mut pos = start;
                for (i, part) in parts.iter().enumerate() {
                    if first.send((i, pos, part.len(), ends.embed(part, pos))).is_err() {
                        break;
                    }
                    pos += part.len();
                }
            });
            // And here, what comes out of the last stage: in order, since
            // every stage kept the order it was given.
            let (mut read, mut out) = (0, Vec::new());
            for (i, _, m, xs) in from_before {
                read += m;
                if all || i == last {
                    let started = std::time::Instant::now();
                    let logits = pool.install(|| ends.head(&xs, m, all));
                    *ends_took = (ends_took.0 + 1, ends_took.1 + started.elapsed().as_secs_f64() * 1e3);
                    match all {
                        true => out.extend(logits),
                        false => out = logits,
                    }
                }
            }
            let mut fault: Option<Fault> = None;
            for stage in working {
                match stage.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => fault = fault.or(Some(e)),
                    Err(_) => fault = fault.or(Some("a stage of the pipeline panicked".into())),
                }
            }
            (read, out, fault)
        });

        // What came out of the last stage is held by every stage. Past it,
        // the stages before a failure hold chunks the ones after never saw.
        *len += read;
        *torn = fault.is_some();
        self.post();
        match fault {
            Some(fault) => Err(plain(fault)),
            None => Ok(out),
        }
    }

    /// Where the time has gone since this was last asked: each stage's
    /// passes and their milliseconds, and last the output head's.
    ///
    /// A token through a split model costs each stage's layers and, for a
    /// stage on another machine, a round trip. Which of those a slow token
    /// was slow in is the first thing to know about it, and neither end
    /// can say alone: the pipeline sees a round trip, and the worker sees
    /// its layers.
    pub fn timing(&mut self) -> (Vec<Timing>, (u64, f64)) {
        self.ask();
        let stages = self
            .stages
            .iter()
            .zip(self.took.iter_mut())
            .zip(self.inside.iter_mut())
            .map(|((stage, took), inside)| {
                let (runs, ms) = std::mem::take(took);
                // Taken, and so nothing to say until it is asked again.
                Timing { shard: stage.shard(), runs, ms, inside_ms: inside.take() }
            })
            .collect();
        (stages, std::mem::take(&mut self.ends_took))
    }

    /// Ask every worker what it has spent inside its layers since it was
    /// last asked. A round trip each, which is why it is not every token.
    fn ask(&mut self) {
        self.asked = std::time::Instant::now();
        for ((stage, life), inside) in self.stages.iter_mut().zip(self.life.iter_mut()).zip(self.inside.iter_mut()) {
            if let Some(spent) = stage.inside() {
                *inside = Some(inside.unwrap_or(0.0) + spent.ms);
                life.inside_one_ms = Some(life.inside_one_ms.unwrap_or(0.0) + spent.one_ms);
                life.inside_many_ms = Some(life.inside_many_ms.unwrap_or(0.0) + spent.ms - spent.one_ms);
                (life.asked_one_runs, life.asked_one_ms, life.asked_many_ms) = (life.one_runs, life.one_ms, life.many_ms);
            }
        }
    }

    /// Have this pipeline's stages and their times shown to whoever asks
    /// [`live`], under the model's name, for as long as it is loaded.
    pub fn shown_as(mut self, repo: &str) -> Self {
        let board = Arc::new(std::sync::Mutex::new(self.report(repo)));
        if let Ok(mut boards) = BOARDS.lock() {
            boards.retain(|shown| shown.strong_count() > 0);
            boards.push(Arc::downgrade(&board));
        }
        self.board = Some(board);
        self
    }

    /// Bring what is shown up to date. After every pass, with what this
    /// machine saw; and with what the workers say, every couple of seconds.
    fn post(&mut self) {
        let Some(board) = self.board.clone() else { return };
        // Every couple of seconds, and the first time there is anything
        // to ask about: a short reply should not end with nothing known.
        // And every fifty tokens, so that a reply shorter than that wait
        // is known about as well: one round trip in fifty.
        let due = self.stages.iter().zip(&self.life).any(|(stage, life)| {
            stage.elsewhere() && (life.inside_one_ms.is_none() || life.one_runs >= life.asked_one_runs + 50)
        });
        if due || self.asked.elapsed() > std::time::Duration::from_secs(2) {
            self.ask();
        }
        let Ok(mut shown) = board.lock() else { return };
        let repo = std::mem::take(&mut shown.repo);
        *shown = self.report(&repo);
    }

    fn report(&self, repo: &str) -> Report {
        Report {
            repo: repo.to_string(),
            precision: self.precision.to_string(),
            positions: self.len,
            stages: self
                .stages
                .iter()
                .zip(&self.life)
                .map(|(stage, life)| {
                    let (shard, (params, bytes)) = (stage.shard(), stage.holds());
                    Staged { layers: [shard.layers.start, shard.layers.end], here: !stage.elsewhere(), place: stage.place(), params, bytes, life: *life }
                })
                .collect(),
        }
    }

    /// Put the stages back in step after a pass that failed part way.
    ///
    /// The stages before the failure ran the pass's rows and the ones
    /// after did not, and `len` was never moved: so every stage is told to
    /// hold `len` positions, which is a rewind for some and nothing for the
    /// rest. A stage that cannot — it has gone — fails this as it failed
    /// the pass, and the session stays torn.
    fn mend(&mut self) -> Res<()> {
        if !self.torn {
            return Ok(());
        }
        for stage in self.stages.iter_mut() {
            let held = stage.truncate(SEQ, self.len).map_err(plain)?;
            if held != self.len {
                return Err(format!("{} holds {held} positions of a conversation that is {} long", stage.shard(), self.len).into());
            }
        }
        self.torn = false;
        Ok(())
    }

    /// Parameters and bytes across the stages, and the ends where no stage
    /// has them: a stage that is also the ends counts them as its own.
    fn held(&self) -> (usize, usize) {
        let ends_apart = self.ends.shard().layers.is_empty();
        let start = if ends_apart { (self.ends.param_count(), self.ends.memory_bytes()) } else { (0, 0) };
        self.stages.iter().map(|s| s.holds()).fold(start, |a, b| (a.0 + b.0, a.1 + b.1))
    }
}

/// The way to one worker of `--workers`: found by its name or its id and
/// reached by whichever of its addresses measures quickest, or reached at
/// the address given. Either way the path is measured and said, so that
/// what a model's hidden states crossed is on the record.
fn way_to(worker: &str, found: &[discover::Found], key: &Key, progress: &mut dyn FnMut(&str)) -> Res<path::Path> {
    if !discover::is_a_name(worker) {
        progress(&format!("the way to {worker}:"));
        return path::quickest(&[worker.to_string()], "", key, progress).map_err(plain);
    }
    let named: Vec<&discover::Found> = found.iter().filter(|f| f.answers_to(worker)).collect();
    let there = match named.as_slice() {
        [one] => *one,
        [] => {
            let seen = match found.is_empty() {
                true => "Nobody on the link says they are one".to_string(),
                false => format!("The workers on the link are {}", found.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", ")),
            };
            return Err(format!("no worker called `{worker}` was found. {seen}.\nIs `kvad worker` running there? A worker can also be named by its address.").into());
        }
        several => {
            let ids: Vec<String> = several.iter().map(|f| format!("{} ({})", f.name, f.id)).collect();
            return Err(format!("`{worker}` is {} workers: {}. Name one by its id.", several.len(), ids.join(", ")).into());
        }
    };
    // Said before any address is tried, since no address would get past it.
    if there.build != super::build() {
        return Err(format!("{} is kvad {} and this is kvad {}; a split model wants one build on every machine", there.name, there.build, super::build()).into());
    }
    progress(&format!("{} is at {} address(es):", there.name, there.addresses.len()));
    let addresses: Vec<String> = there.addresses.iter().map(|a| path::with_port(a, there.port)).collect();
    let way = path::quickest(&addresses, &there.id, key, progress).map_err(plain)?;
    progress(&format!("  using {}", way.address));
    Ok(way)
}

/// `0..12,12..24` as layer ranges: what `--split` takes.
pub fn parse_split(text: &str) -> Res<Vec<std::ops::Range<usize>>> {
    let range = |word: &str| -> Option<std::ops::Range<usize>> {
        let (from, to) = word.trim().split_once("..")?;
        let (from, to) = (from.trim().parse().ok()?, to.trim().parse().ok()?);
        (from <= to).then_some(from..to)
    };
    text.split(',').map(|word| range(word).ok_or_else(|| format!("`{}` is not a run of layers; --split is written 0..12,12..24", word.trim()).into())).collect()
}

/// What `--workers` asks for: a model, the machines to put it on, and
/// what is known about how.
pub struct Across<'a> {
    /// The model, as a worker will be asked for it.
    pub repo: &'a str,
    pub files: &'a crate::weights::ModelFiles,
    pub spec: &'a Spec,
    pub precision: Precision,
    /// As `--workers` gives them: names, ids or addresses.
    pub workers: &'a [String],
    /// The layers each machine holds, this one first, when the operator
    /// says; and when not, they are placed: see [`plan`].
    pub split: Option<&'a [std::ops::Range<usize>]>,
    pub key: &'a Key,
    /// The model's parameters, where anything knows. Without them nothing
    /// can be said about memory, and nothing is placed.
    pub params: Option<u64>,
    /// The longest conversation to leave room for, in tokens.
    pub context: usize,
    /// What the layers are placed for, when they are placed here: so much
    /// prompt read to so much reply written.
    pub mix: plan::Mix,
    /// Who is shown the load a machine at a time: every machine's steps
    /// go here under its name, as [`crate::weights::Fetch::Part`].
    pub watch: &'a crate::weights::Watcher,
}

/// The machines of a plan: this one, timed, and each worker reached by its
/// quickest path and timed there.
pub fn survey(spec: &Spec, precision: Precision, workers: &[String], key: &Key, progress: &mut dyn FnMut(&str)) -> Res<(Vec<plan::Node>, Vec<path::Path>)> {
    // Asked once, and only if somebody was named rather than addressed:
    // an answer to multicast is waited for, and that is a couple of seconds.
    let found = match workers.iter().any(|w| discover::is_a_name(w)) {
        true => {
            progress("asking who on the link is a worker");
            discover::workers()
        }
        false => Vec::new(),
    };
    let here = plan::bench(spec.n_embd, spec.intermediate, precision);
    let mut nodes = vec![plan::Node {
        name: super::hostname(),
        memory: crate::machine::usable_memory_cached().unwrap_or(u64::MAX),
        layer_ms: here.layer_ms,
        hop_ms: 0.0,
        prompt_layer_ms: here.prompt_layer_ms,
        prompt_hop_ms: 0.0,
    }];
    // What crosses: a row of the residual stream for a token written, and
    // a chunk of them for a prompt.
    let row = spec.n_embd * std::mem::size_of::<f32>();
    let mut ways = Vec::with_capacity(workers.len());
    for worker in workers {
        let way = way_to(worker, &found, key, progress)?;
        let there = super::remote::bench(&way.address, key, spec.n_embd, spec.intermediate, precision).map_err(plain)?;
        nodes.push(plan::Node {
            name: way.hello.name.clone(),
            // A worker that does not say is taken at its word about
            // nothing: it is given what it is given, and finds out.
            memory: way.hello.memory.unwrap_or(u64::MAX),
            layer_ms: there.layer_ms,
            hop_ms: way.round_trip(row),
            prompt_layer_ms: there.prompt_layer_ms,
            prompt_hop_ms: way.round_trip(row * prefill_chunk()),
        });
        ways.push(way);
    }
    Ok((nodes, ways))
}

/// The plan for `want` over `nodes`: the operator's split, checked against
/// each machine's memory, or one made here.
pub fn planned(want: &Across, nodes: &[plan::Node]) -> Res<plan::Plan> {
    checked(want)?;
    let sizes = want.params.map(|params| plan::Sizes::of(want.spec, params, want.precision, want.context));
    match (want.split, sizes) {
        (Some(split), Some(sizes)) => Ok(plan::check(nodes, &sizes, split)?),
        // Nothing is known of the model's size, so nothing is checked: the
        // operator's word is the plan.
        (Some(split), None) => {
            let nothing = plan::Sizes { n_layer: want.spec.n_layer, ends: 0, layer: 0, cache: 0 };
            Ok(plan::check(nodes, &nothing, split)?)
        }
        (None, Some(sizes)) => Ok(plan::place(nodes, &sizes, want.mix)?),
        (None, None) => Err("how large this model is could not be found out, so its layers cannot be placed; say where they go with --split".into()),
    }
}

/// What `--split` has to be before anything is asked of anybody: a run for
/// each machine, covering the model, and no worker with none.
fn checked(want: &Across) -> Res<()> {
    let Some(split) = want.split else { return Ok(()) };
    if split.len() != want.workers.len() + 1 {
        return Err(format!(
            "--split names {} run(s) of layers and there are {} machine(s): this one, then each of --workers, a run apiece",
            split.len(),
            want.workers.len() + 1
        )
        .into());
    }
    let shards: Vec<Shard> = split.iter().enumerate().map(|(i, layers)| Shard { layers: layers.clone(), ends: i == 0 }).collect();
    Shard::check(&shards, want.spec.n_layer)?;
    if let Some(i) = (1..shards.len()).find(|&i| shards[i].layers.is_empty()) {
        return Err(format!("{} is given no layers; leave it out of --workers", want.workers[i - 1]).into());
    }
    Ok(())
}

/// A model across this machine and others: the session `--workers` asks
/// for.
///
/// This machine always holds the ends, and may hold no layers. With
/// `--split` the layers go where the operator says, a run for each machine
/// in order, this one first; without it they are placed, fastest machine
/// first, and a worker the plan has no use for is left out. Either way
/// every machine's share is held to its memory before anything is loaded,
/// and the plan is said.
pub fn across(want: &Across, progress: &mut dyn FnMut(&str)) -> Res<Box<dyn Session>> {
    checked(want)?;
    let (nodes, ways) = survey(want.spec, want.precision, want.workers, want.key, progress)?;
    let plan = planned(want, &nodes)?;
    for line in plan::table(&nodes, &plan) {
        progress(&line);
    }
    // Nothing for any worker: then this is a model on one machine, and is
    // loaded as one. Not as a pipeline of one stage, which would be the
    // same arithmetic without what a model whole has that a pipeline has
    // not yet — its cache file, several conversations, replies in a batch.
    if plan.shares.iter().skip(1).all(|share| share.layers.is_empty()) {
        if want.split.is_none() {
            progress("it fits on this machine, and a layer here costs no round trip: the workers hold nothing. --split overrules.");
        }
        let model = crate::qcache::load(want.repo, want.files, want.spec, want.precision, progress)?;
        return Ok(Box::new(crate::model::CpuSession::new(model, want.precision)));
    }
    Ok(Box::new(staged(want, &plan, &ways, progress)?))
}

/// [`across`], as the pipeline it is: for a caller that wants to ask it
/// where its time went.
pub fn pipeline(want: &Across, progress: &mut dyn FnMut(&str)) -> Res<PipelineSession> {
    checked(want)?;
    let (nodes, ways) = survey(want.spec, want.precision, want.workers, want.key, progress)?;
    let plan = planned(want, &nodes)?;
    for line in plan::table(&nodes, &plan) {
        progress(&line);
    }
    staged(want, &plan, &ways, progress)
}

/// The pipeline a plan describes: this machine's shard loaded, and each
/// worker with layers asked for them.
///
/// All at once. A machine's load is its own disk and its own processor,
/// and often its own download, so nothing is gained by one waiting for
/// another, and a model across three machines is ready when the slowest of
/// them is and not after the sum of them. What each says arrives in the
/// order it was said, which is no longer a machine at a time: a worker's
/// lines carry its name, and [`Across::watch`] is told whose every step is.
fn staged(want: &Across, plan: &plan::Plan, ways: &[path::Path], progress: &mut dyn FnMut(&str)) -> Res<PipelineSession> {
    use crate::weights::{Fetch, Part};
    let here = Shard { layers: plan.shares[0].layers.clone(), ends: true };
    let me = super::hostname();
    let mine = |step: Part| want.watch.emit(Fetch::Part { machine: me.clone(), layers: [here.layers.start, here.layers.end], step });
    let asked: Vec<(&path::Path, Load)> = ways
        .iter()
        .zip(&plan.shares[1..])
        .filter(|(_, share)| !share.layers.is_empty())
        .map(|(way, share)| (way, Load { repo: want.repo.to_string(), precision: want.precision.to_string(), layers: [share.layers.start, share.layers.end] }))
        .collect();
    progress(&format!("reading {here} here"));
    for (way, load) in &asked {
        progress(&format!("asking {} for layers {}..{}", way.hello.name, load.layers[0], load.layers[1]));
    }

    let (said, lines) = std::sync::mpsc::channel::<String>();
    let (ends, far) = std::thread::scope(|scope| {
        let local = scope.spawn({
            let (said, here, mine) = (said.clone(), &here, &mine);
            move || {
                mine(Part::Asked);
                let mut line = |line: &str| {
                    mine(Part::Said(line.to_string()));
                    let _ = said.send(line.to_string());
                };
                let read = crate::qcache::load_shard(want.repo, want.files, want.spec, want.precision, here, &mut line).map_err(|e| e.to_string());
                match &read {
                    Ok(model) => mine(Part::Ready { bytes: model.memory_bytes() as u64 }),
                    Err(why) => mine(Part::Failed(why.clone())),
                }
                read
            }
        });
        let far: Vec<_> = asked
            .iter()
            .map(|(way, load)| {
                let said = said.clone();
                scope.spawn(move || {
                    let mut line = |line: &str| {
                        let _ = said.send(line.to_string());
                    };
                    RemoteStage::open(&way.address, want.key, load, &mut line, want.watch).map(|stage| stage.reached_by(way)).map_err(|e| e.to_string())
                })
            })
            .collect();
        // Every sender is a loader's now: the lines end when the loads do.
        drop(said);
        for line in lines {
            progress(&line);
        }
        let gone = |_| "a load panicked".to_string();
        (local.join().unwrap_or_else(|e| Err(gone(e))), far.into_iter().map(|load| load.join().unwrap_or_else(|e| Err(gone(e)))).collect::<Vec<_>>())
    });
    let ends: Arc<dyn Transformer> = Arc::from(ends?);
    let mut stages: Vec<Box<dyn Stage>> = Vec::new();
    if !here.layers.is_empty() {
        stages.push(Box::new(LocalStage::new(ends.clone())));
    }
    for stage in far {
        stages.push(Box::new(stage?));
    }
    Ok(PipelineSession::new(ends, stages, want.precision)?.shown_as(want.repo))
}
