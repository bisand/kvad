//! Some of a Llama's blocks on the GPU, for a model that is across several
//! machines: [`kvad::cluster::stage::Stage`], as [`super::GpuLlama`] is
//! [`kvad::model::Session`].
//!
//! A stage has no tokens. Rows of the residual stream come in from the
//! host, go through this stage's blocks on the device and come back, and
//! that is one upload and one readback of `n_embd` floats a token: the
//! price of a stage boundary on a GPU, paid whether the next stage is in
//! this process or across a cable. The embedding and the head stay with
//! whoever has the tokens.
//!
//! The blocks, the kernels and the cache are [`super::GpuLlama`]'s own,
//! through the same [`super::Pass`]. What differs is the last add: a model
//! whole fuses a block's MLP output into the next block's norm, and the
//! last block's into the final norm. A stage's last block has neither, so
//! its output is added plainly and the sum is what crosses.

use super::{read_blocks, rotations, settled, Block, Pass};
use crate::common::{alone, check_block, kv_store, pool_only, pooled, report_failures, KvCache, Loader, Reader};
use crate::fused;
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kvad::cluster::stage::{Fault, Stage};
use kvad::model::ffn::{Layout, Router};
use kvad::model::{Shard, Spec};
use kvad::serde_json;
use std::ops::Range;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A conversation's keys and values in this stage's blocks, and how many
/// positions of it they hold.
struct Held {
    seq: u64,
    kv: Vec<KvCache>,
    pos: usize,
}

pub struct GpuStage {
    spec: Spec,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    layers: Range<usize>,
    blocks: Vec<Block>,
    cos: Tensor,
    sin: Tensor,
    /// A cache a conversation. A list and not a map: there are a handful.
    held: Vec<Held>,
    fuse: bool,
    /// The device kept busy between one pass and the next: see [`Warm`].
    warm: Option<Warm>,
}

/// A GPU kept awake for a token that is on its way.
///
/// A stage works in bursts: a token's rows arrive, its layers run, and
/// then nothing until the token has been round the other machines. A GPU
/// that has had nothing to do for three milliseconds does its next burst
/// two and a half times slower, and after ten, four times: fifteen layers
/// of SmolLM2-135M on an M5 Pro take 0.85 ms back to back, 2.1 ms after
/// 3 ms idle and 3.7 ms after 10. A spinning CPU thread, which is what
/// keeps a CPU stage quick, recovers some of it and not most: it is the
/// GPU that has stepped down. `examples/stage_gaps` measures all of it.
///
/// So between passes a thread gives the device something small to do,
/// over and over, until the next pass begins or [`kvad::model::linger_ms`]
/// has gone by: the limit is for the token that never comes. What it runs
/// is [`fused::metal::Warmer`], which touches nothing of candle's. That
/// matters twice. candle's Metal backend is not safe for two threads, and
/// a thread running candle's own ops beside a model is a segfault waiting.
/// And each of candle's readbacks sweeps its pool of buffers, so a thread
/// doing them between passes threw away the buffers the next pass was
/// about to reuse.
struct Warm {
    state: std::sync::Arc<Resting>,
    thread: std::thread::Thread,
}

struct Resting {
    /// Between passes, since when.
    since: std::sync::Mutex<Option<std::time::Instant>>,
    gone: std::sync::atomic::AtomicBool,
}

impl Warm {
    fn start(device: &Device) -> Option<Warm> {
        let limit = std::time::Duration::from_millis(kvad::model::linger_ms());
        if limit.is_zero() || !device.is_metal() {
            return None;
        }
        let state = std::sync::Arc::new(Resting { since: std::sync::Mutex::new(None), gone: false.into() });
        let (theirs, device) = (state.clone(), device.clone());
        let thread = std::thread::Builder::new()
            .name("kvad-gpu-warm".into())
            .spawn(move || {
                let Some(warmer) = fused::metal::Warmer::new(&device) else { return };
                while !theirs.gone.load(std::sync::atomic::Ordering::Acquire) {
                    let resting = theirs.since.lock().map(|since| since.is_some_and(|since| since.elapsed() < limit)).unwrap_or(false);
                    match resting {
                        // Waited for, so that one is done before the next
                        // is asked for; in a pool, as Metal on a thread of
                        // its own has to be.
                        true if pool_only(|| warmer.once()).is_some() => {}
                        true => return,
                        // A pass is running, or no token came: until told.
                        false => std::thread::park(),
                    }
                }
            })
            .ok()?;
        Some(Warm { state, thread: thread.thread().clone() })
    }

    /// A pass is over, and the wait for the next has begun.
    fn rest(&self) {
        if let Ok(mut since) = self.state.since.lock() {
            *since = Some(std::time::Instant::now());
        }
        self.thread.unpark();
    }

    /// A pass is beginning: the device is the model's.
    fn work(&self) {
        if let Ok(mut since) = self.state.since.lock() {
            *since = None;
        }
    }
}

impl Drop for Warm {
    fn drop(&mut self) {
        self.state.gone.store(true, std::sync::atomic::Ordering::Release);
        self.thread.unpark();
    }
}

/// Whether a model of this architecture can be held a stage at a time on
/// the GPU: the Llama family, whose blocks these are.
pub fn supports(arch: kvad::model::Arch) -> bool {
    arch.is("llama")
}

impl GpuStage {
    /// Layers `layers` of the model in `paths`, which need hold no other
    /// layer's tensors: a worker fetches its own shards of a checkpoint and
    /// no more.
    ///
    /// Quantised weights are kept in a cache file of this stage's own,
    /// named for the model and the layers. The model's whole cache is not
    /// read: it is identified by every file of the checkpoint, and a
    /// worker may have only some of them.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        repo: &str,
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
        layers: Range<usize>,
        progress: &mut dyn FnMut(&str),
    ) -> Res<Self> {
        // A turn at the device for the whole of it: the weights go up as
        // they are read.
        alone(|| Self::read(repo, paths, spec, dtype, quant, device, layers, progress))
    }

    #[allow(clippy::too_many_arguments)]
    fn read(
        repo: &str,
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
        layers: Range<usize>,
        progress: &mut dyn FnMut(&str),
    ) -> Res<Self> {
        if !supports(spec.arch) {
            return Err(format!("`{}` is not held a stage at a time on the GPU; the Llama family is", spec.arch).into());
        }
        if layers.is_empty() || layers.end > spec.n_layer {
            return Err(format!("layers {}..{} of a model with {}", layers.start, layers.end, spec.n_layer).into());
        }
        // As `GpuLlama::load`, which says why of each line.
        let load_dtype = if quant.is_some() { DType::F32 } else { dtype };
        let routed = match Router::count(&spec.config) {
            None => None,
            Some(_) => Some((Router::read(&spec.config)?, Layout::read(&spec.config))),
        };
        let expert_width = spec.config.num(&["moe_intermediate_size"]).unwrap_or(spec.intermediate);
        let mut widths = vec![
            ("hidden size", spec.n_embd),
            ("MLP width", spec.intermediate),
            ("attention output", spec.n_head * spec.head_dim),
            ("KV width", spec.kv_dim()),
        ];
        if routed.is_some() {
            widths.push(("expert width", expert_width));
        }
        check_block(quant, &widths)?;

        let shape = serde_json::json!({
            "arch": spec.arch.to_string(),
            "n_layer": spec.n_layer,
            "n_embd": spec.n_embd,
            "layers": [layers.start, layers.end],
        });
        let mut cache = Vault::open_as(&format!("{repo}/layers-{}-{}", layers.start, layers.end), paths, shape, quant, progress);

        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, load_dtype, &Device::Cpu)? };
        let vb = Reader::new(vb);
        let ld = Loader::new(quant, device.clone(), &cache);
        let blocks = read_blocks(&ld, &vb.pp("model"), &spec, routed.as_ref(), expert_width, load_dtype, layers.clone())?;
        drop(ld);
        cache.finish(progress);

        let (cos, sin) = rotations(&spec, &device, load_dtype)?;
        settled(&device)?;
        Ok(GpuStage { fuse: fused::available(&device), held: Vec::new(), warm: Warm::start(&device), blocks, cos, sin, device, dtype: load_dtype, quant, layers, spec })
    }

    fn held(&mut self, seq: u64) -> usize {
        match self.held.iter().position(|held| held.seq == seq) {
            Some(at) => at,
            None => {
                let kv = (0..self.blocks.len()).map(|_| KvCache::new(2).stored_as(kv_store(&self.device, self.quant))).collect();
                self.held.push(Held { seq, kv, pos: 0 });
                self.held.len() - 1
            }
        }
    }

    fn pass(&mut self, at: usize, pos0: usize, m: usize, xs: Vec<f32>) -> Res<Vec<f32>> {
        let x = Tensor::from_vec(xs, (m, self.spec.n_embd), &self.device)?.to_dtype(self.dtype)?;
        let pass = Pass { spec: &self.spec, cos: &self.cos, sin: &self.sin, fuse: self.fuse, dtype: self.dtype, device: &self.device };
        let (x, pending) = pass.through(&self.blocks, x, pos0, m, &mut self.held[at].kv)?;
        let x = match pending {
            Some(f) => (x + f)?,
            None => x,
        };
        let out = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        report_failures(&self.device)?;
        Ok(out)
    }
}

impl Stage for GpuStage {
    fn shard(&self) -> Shard {
        Shard { layers: self.layers.clone(), ends: false }
    }

    fn run(&mut self, seq: u64, pos0: usize, m: usize, xs: Vec<f32>) -> Result<Vec<f32>, Fault> {
        let (e, shard) = (self.spec.n_embd, self.shard());
        if xs.len() != m * e {
            return Err(format!("{shard}: {m} rows of {e} were announced and {} numbers arrived", xs.len()).into());
        }
        let at = self.held(seq);
        if self.held[at].pos != pos0 {
            return Err(format!("{shard}: handed position {pos0} of a conversation it holds {} positions of", self.held[at].pos).into());
        }
        if pos0 + m > self.spec.n_ctx {
            return Err(format!("{shard}: position {} is past this model's {}", pos0 + m, self.spec.n_ctx).into());
        }
        if let Some(warm) = &self.warm {
            warm.work();
        }
        // A pool a pass: this runs on whatever thread a pipeline or a
        // worker gives it, and none of them has one. See `pooled`.
        match pooled(|| self.pass(at, pos0, m, xs)) {
            Ok(out) => {
                self.held[at].pos += m;
                if let Some(warm) = &self.warm {
                    warm.rest();
                }
                Ok(out)
            }
            Err(why) => {
                // Some blocks wrote their keys and some did not: back to
                // where every block agrees, so the next pass is not wrong.
                self.held[at].kv.iter_mut().for_each(|c| c.truncate(pos0));
                Err(format!("{shard}: {why}").into())
            }
        }
    }

    fn truncate(&mut self, seq: u64, len: usize) -> Result<usize, Fault> {
        let at = self.held(seq);
        let held = &mut self.held[at];
        if len < held.pos {
            held.kv.iter_mut().for_each(|c| c.truncate(len));
            held.pos = len;
        }
        Ok(held.pos)
    }

    fn holds(&self) -> (usize, usize) {
        (self.blocks.iter().map(Block::params).sum(), self.blocks.iter().map(Block::bytes).sum())
    }

    fn keeps_itself_awake(&self) -> bool {
        self.device.is_metal()
    }
}

/// How long one layer of a model this shape takes this device: three
/// blocks of the model's widths, run as a stage runs them, for one token
/// and for a chunk of a prompt's.
///
/// What [`kvad::cluster::plan::bench`] is to the CPU, asked before there is
/// a model: the blocks are zeros, which a matrix product costs the same to
/// multiply. Three and not one, so that what a stage pays once a pass —
/// the rows up to the device and back — is a third of the answer and not
/// all of it; the planner is told a layer's time and multiplies.
pub fn bench(n_embd: usize, intermediate: usize, dtype: DType, quant: Option<GgmlDType>, device: Device) -> Res<kvad::cluster::plan::Speed> {
    alone(|| timed(n_embd, intermediate, dtype, quant, device))
}

fn timed(n_embd: usize, intermediate: usize, dtype: DType, quant: Option<GgmlDType>, device: Device) -> Res<kvad::cluster::plan::Speed> {
    const BLOCKS: usize = 3;
    let chunk = kvad::model::prefill_chunk();
    // Heads sixty-four wide where the model's width allows, as most are.
    let n_head = if n_embd % 64 == 0 { n_embd / 64 } else { 1 };
    let head_dim = n_embd / n_head;
    let spec = Spec {
        arch: kvad::model::Arch::require("llama"),
        n_layer: BLOCKS,
        n_head,
        n_kv_head: n_head,
        n_embd,
        head_dim,
        n_ctx: chunk + 64,
        vocab_size: 1,
        intermediate,
        eps: 1e-5,
        rope_theta: 10000.0,
        tie_embeddings: true,
        cache: kvad::model::CacheLayout::uniform(kvad::model::CacheShape::kv(n_embd, n_embd)),
        config: kvad::model::Json::default(),
    };
    let load_dtype = if quant.is_some() { DType::F32 } else { dtype };
    check_block(quant, &[("hidden size", n_embd), ("MLP width", intermediate)])?;

    let host = Device::Cpu;
    let zeros = |rows: usize, cols: usize| Tensor::zeros((rows, cols), load_dtype, &host);
    let ones = Tensor::ones(n_embd, load_dtype, &host)?;
    let (wide, tall, down) = (zeros(n_embd, n_embd)?, zeros(intermediate, n_embd)?, zeros(n_embd, intermediate)?);
    let mut tensors = std::collections::HashMap::new();
    for layer in 0..BLOCKS {
        let p = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            tensors.insert(format!("{p}.{norm}.weight"), ones.clone());
        }
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.insert(format!("{p}.self_attn.{proj}.weight"), wide.clone());
        }
        tensors.insert(format!("{p}.mlp.gate_proj.weight"), tall.clone());
        tensors.insert(format!("{p}.mlp.up_proj.weight"), tall.clone());
        tensors.insert(format!("{p}.mlp.down_proj.weight"), down.clone());
    }
    let vb = Reader::new(VarBuilder::from_tensors(tensors, load_dtype, &host));
    let cache = Vault::off();
    let ld = Loader::new(quant, device.clone(), &cache);
    let blocks = read_blocks(&ld, &vb.pp("model"), &spec, None, intermediate, load_dtype, 0..BLOCKS)?;
    drop(ld);
    let (cos, sin) = rotations(&spec, &device, load_dtype)?;
    settled(&device)?;
    let mut stage = GpuStage { fuse: fused::available(&device), held: Vec::new(), warm: None, blocks, cos, sin, device, dtype: load_dtype, quant, layers: 0..BLOCKS, spec };

    let time = |stage: &mut GpuStage, pos0: usize, m: usize| -> Res<f64> {
        let rows = vec![0.01f32; m * n_embd];
        let start = std::time::Instant::now();
        stage.run(0, pos0, m, rows).map_err(|e| e.to_string())?;
        Ok(start.elapsed().as_secs_f64() * 1e3)
    };
    // The first passes compile the kernels and make the buffers.
    for pos in 0..4 {
        time(&mut stage, pos, 1)?;
    }
    let mut ones: Vec<f64> = (4..20).map(|pos| time(&mut stage, pos, 1)).collect::<Res<_>>()?;
    ones.sort_by(f64::total_cmp);
    let mut chunks = Vec::new();
    for _ in 0..4 {
        stage.truncate(0, 0).map_err(|e| e.to_string())?;
        chunks.push(time(&mut stage, 0, chunk)?);
    }
    chunks.sort_by(f64::total_cmp);
    Ok(kvad::cluster::plan::Speed {
        layer_ms: ones[ones.len() / 2] / BLOCKS as f64,
        // The first chunk made the buffers: the second quickest of four.
        prompt_layer_ms: chunks[1] / (BLOCKS * chunk) as f64,
    })
}
