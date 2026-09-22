//! GPT-2 on the GPU.
//!
//! Read this next to [`kvad::model::gpt2`], which is the same block written by
//! hand, and next to [`crate::model`], which is the same *backend* written for
//! a different block. Almost everything that differs from the Llama file here
//! is a difference GPT-2 has from Llama, not a difference candle has from us:
//!
//! | | Llama | GPT-2 |
//! |---|---|---|
//! | position | RoPE, per layer | one learned vector added at the bottom |
//! | normalisation | RMSNorm, no bias | LayerNorm, centred, with a bias |
//! | attention | three projections, grouped KV heads | one fused projection, every head its own |
//! | MLP | SwiGLU, three matrices | GELU, two matrices |
//! | biases | mostly absent | everywhere |
//!
//! # The one that is a real trap
//!
//! GPT-2 was written before `nn.Linear` was the obvious choice, and its
//! projections are `Conv1D`, which stores its weight `[in, out]` — the
//! transpose of what every checkpoint since does. The hand-written engine
//! deals with this by reading them through `matrix_t`.
//!
//! Here it cancels out, twice, in opposite directions. A dense [`Proj`] wants
//! `[in, out]`, which is what GPT-2 already has, so the dense path transposes
//! *nothing* where the Llama loader transposes everything. A quantised `Proj`
//! wants `[out, in]`, so that path transposes what the Llama loader leaves
//! alone. One flag, read by both loaders, rather than two spellings of the same
//! sentence.
//!
//! Nothing announces which convention a file uses, and the shapes are square
//! in the two places it would be caught cheaply (`c_proj` in attention is
//! `[e, e]`). So this is exactly the class of mistake the cross-engine tests
//! exist for: a transposed weight is not an error, it is a model that runs at
//! full speed and says something else.

use crate::common::{
    causal_mask, check_block, embedding, label, linear, unread, unread_error, Embed, Loader, Proj,
    Reader, Stored,
};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops, VarBuilder};
use kvad::model::{Session, Spec};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// LayerNorm's two vectors, which travel together everywhere.
struct Norm {
    gain: Tensor,
    bias: Tensor,
}

impl Norm {
    fn forward(&self, x: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
        ops::layer_norm(&x.contiguous()?, &self.gain, &self.bias, eps)
    }

    fn params(&self) -> usize {
        self.gain.elem_count() + self.bias.elem_count()
    }
}

struct Block {
    ln1: Norm,
    /// `[n_embd, 3 * n_embd]`: query, key and value in one matrix, because
    /// three matmuls that all read the same input is wasteful when one will do.
    attn: Proj,
    attn_b: Tensor,
    o: Proj,
    o_b: Tensor,
    ln2: Norm,
    fc: Proj,
    fc_b: Tensor,
    down: Proj,
    down_b: Tensor,
}

pub struct GpuGpt2 {
    spec: Spec,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    /// The token table, `[vocab, n_embd]`.
    wte: Embed,
    /// The position table, `[n_ctx, n_embd]`. Left dense whatever the mode:
    /// GPT-2's context is 1024 slots, so this is under 3 MB even at n_embd
    /// 1600, and it is read by row like the token table rather than multiplied.
    wpe: Tensor,
    head: Proj,
    /// GPT-2 itself has none — its head *is* the embedding table applied
    /// backwards, and a table has no bias to apply with it. A model trained
    /// here with an untied head may.
    head_b: Option<Tensor>,
    /// Whether `head` is the same allocation as `wte`.
    tied: bool,
    blocks: Vec<Block>,
    ln_f: Norm,
    /// Per layer, `[1, n_head, seq, head_dim]` for keys and values. No grouping:
    /// GPT-2 predates it, so `n_kv_head == n_head`.
    kv: Vec<Option<(Tensor, Tensor)>>,
    pos: usize,
}

impl GpuGpt2 {
    pub fn load(
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
        cache: &Vault,
    ) -> Res<Self> {
        let load_dtype = if quant.is_some() { DType::F32 } else { dtype };
        let compute = load_dtype;
        let (e, hd) = (spec.n_embd, spec.head_dim);
        let inter = spec.intermediate;

        check_block(
            quant,
            &[("hidden size", e), ("MLP width", inter), ("fused QKV width", 3 * e)],
        )?;

        // The checkpoint is mapped on the host in both modes, and each
        // tensor is moved to the device once it is in its final form —
        // quantised into blocks, or transposed. Reading straight onto the
        // device instead costs a second full copy of every dense weight
        // while its transpose is built, which is what put a 7B over this
        // machine's GPU budget; see `Loader::proj`.
        let load_dev = Device::Cpu;
        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, load_dtype, &load_dev)? };
        let vb = Reader::new(vb);

        let to_dev = |t: Tensor| -> Res<Tensor> { Ok(t.to_device(&device)?) };
        let norm = |vb: &Reader<'_>, name: &str| -> Res<Norm> {
            Ok(Norm {
                gain: to_dev(vb.get(e, &format!("{name}.weight"))?)?,
                bias: to_dev(vb.get(e, &format!("{name}.bias"))?)?,
            })
        };

        // `Conv1D` stores `[in, out]`, which is the transpose of what every
        // other loader here reads. That used to be a paragraph in this file
        // about transposing in the quantised path and not in the dense one;
        // it is `Stored::InOut` now, and the rule lives in one place.
        let ld = Loader::new(quant, device.clone(), cache);
        let load_c = |vb: &Reader<'_>, name: &str, inp: usize, out: usize| -> Res<Proj> {
            ld.proj(vb, name, out, inp, Stored::InOut)
        };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            let l = vb.pp(format!("h.{i}"));
            let attn = l.pp("attn");
            let mlp = l.pp("mlp");
            blocks.push(Block {
                ln1: norm(&l, "ln_1")?,
                attn: load_c(&attn, "c_attn.weight", e, 3 * e)?,
                attn_b: to_dev(attn.get(3 * e, "c_attn.bias")?)?,
                o: load_c(&attn, "c_proj.weight", e, e)?,
                o_b: to_dev(attn.get(e, "c_proj.bias")?)?,
                ln2: norm(&l, "ln_2")?,
                fc: load_c(&mlp, "c_fc.weight", e, inter)?,
                fc_b: to_dev(mlp.get(inter, "c_fc.bias")?)?,
                down: load_c(&mlp, "c_proj.weight", inter, e)?,
                down_b: to_dev(mlp.get(e, "c_proj.bias")?)?,
            });
        }

        let (wte, head, tied) = embedding(&ld, &vb, &vb, "wte.weight", &spec)?;
        let wpe = to_dev(vb.get((spec.n_ctx, e), "wpe.weight")?.to_dtype(compute)?)?;

        // GPT-2 ties and ships no head at all; a model trained here may untie
        // and ship one with a bias beside it, which `embedding` has no opinion
        // about because no other architecture here has one.
        let head_b = match spec.tie_embeddings {
            true => {
                vb.record("lm_head.bias");
                None
            }
            false => vb.try_get(spec.vocab_size, "lm_head.bias").map(&to_dev).transpose()?,
        };

        let ln_f = norm(&vb, "ln_f")?;

        let left = unread(paths, &vb.seen(), &vb.skipped())?;
        if !left.is_empty() {
            return Err(unread_error("gpt2", &left).into());
        }

        debug_assert_eq!(spec.n_head * hd, e, "GPT-2 splits n_embd exactly across its heads");

        Ok(GpuGpt2 {
            kv: (0..spec.n_layer).map(|_| None).collect(),
            blocks,
            wte,
            wpe,
            head,
            head_b,
            tied,
            ln_f,
            device,
            dtype: compute,
            quant,
            pos: 0,
            spec,
        })
    }

    pub fn device_label(&self) -> String {
        label(&self.device, self.dtype, self.quant)
    }

    fn rewind(&mut self, len: usize) -> Res<()> {
        if len >= self.pos {
            return Ok(());
        }
        if len == 0 {
            self.kv.iter_mut().for_each(|s| *s = None);
        } else {
            for slot in self.kv.iter_mut() {
                if let Some((k, v)) = slot.take() {
                    *slot = Some((k.narrow(2, 0, len)?, v.narrow(2, 0, len)?));
                }
            }
        }
        self.pos = len;
        Ok(())
    }

    /// Run `tokens` and return logits for the last one.
    fn run(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let m = tokens.len();
        let spec = &self.spec;
        let (e, hd, heads) = (spec.n_embd, spec.head_dim, spec.n_head);
        let pos0 = self.pos;
        let scale = 1.0 / (hd as f64).sqrt();

        if pos0 + m > spec.n_ctx {
            return Err(format!(
                "GPT-2 has {} position vectors and nothing to say about slot {}: its context \
                 is learned, not extrapolated.",
                spec.n_ctx,
                pos0 + m
            )
            .into());
        }

        // Position is added once, at the bottom, and never mentioned again —
        // where RoPE rotates Q and K inside every layer.
        let ids = Tensor::from_slice(tokens, (m,), &self.device)?;
        let tok = self.wte.rows(&ids)?.to_dtype(self.dtype)?;
        let mut x = (tok + self.wpe.narrow(0, pos0, m)?)?;

        let mask = match m > 1 {
            true => Some(causal_mask(m, pos0, &self.device, self.dtype)?),
            false => None,
        };

        for (i, blk) in self.blocks.iter().enumerate() {
            let h = blk.ln1.forward(&x, spec.eps)?;

            // One matmul, then three views of its output. The halves are
            // contiguous rows of `[m, 3e]`, so this is a narrow, not a gather.
            let qkv = linear(&h, &blk.attn, Some(&blk.attn_b))?;
            let split = |at: usize| -> candle_core::Result<Tensor> {
                qkv.narrow(1, at * e, e)?
                    .reshape((1, m, heads, hd))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            let (q, k, v) = (split(0)?, split(1)?, split(2)?);

            let (k, v) = match self.kv[i].take() {
                None => (k, v),
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            };
            self.kv[i] = Some((k.clone(), v.clone()));

            let mut att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
            if let Some(msk) = &mask {
                att = att.broadcast_add(msk)?;
            }
            let att = ops::softmax_last_dim(&att)?;

            let out = att.matmul(&v)?.transpose(1, 2)?.reshape((m, e))?;
            x = (x + linear(&out, &blk.o, Some(&blk.o_b))?)?;

            let h = blk.ln2.forward(&x, spec.eps)?;
            // `gelu` is candle's tanh approximation, which is the one GPT-2 was
            // trained with and the one `tensor.rs` implements. `gelu_erf` is the
            // exact form and would be a different model.
            let h = linear(&h, &blk.fc, Some(&blk.fc_b))?.gelu()?;
            x = (x + linear(&h, &blk.down, Some(&blk.down_b))?)?;
        }

        self.pos += m;

        let last = x.i(m - 1)?.unsqueeze(0)?;
        let last = self.ln_f.forward(&last, spec.eps)?;
        let logits = match &self.head_b {
            None => self.head.forward(&last)?,
            Some(b) => self.head.forward(&last)?.broadcast_add(b)?,
        };
        Ok(logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn params(&self) -> usize {
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.attn.params()
                    + b.o.params()
                    + b.fc.params()
                    + b.down.params()
                    + b.attn_b.elem_count()
                    + b.o_b.elem_count()
                    + b.fc_b.elem_count()
                    + b.down_b.elem_count()
                    + b.ln1.params()
                    + b.ln2.params()
            })
            .sum();
        // Counted when the model does not tie, whatever this backend does about
        // allocations — see the same rule in `model.rs`.
        let head = match self.spec.tie_embeddings {
            true => 0,
            false => self.head.params() + self.head_b.as_ref().map_or(0, |b| b.elem_count()),
        };
        blocks + self.wte.params() + self.wpe.elem_count() + head + self.ln_f.params()
    }

    fn memory_bytes(&self) -> usize {
        let per = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.attn.bytes()
                    + b.o.bytes()
                    + b.fc.bytes()
                    + b.down.bytes()
                    + per(&b.attn_b)
                    + per(&b.o_b)
                    + per(&b.fc_b)
                    + per(&b.down_b)
                    + per(&b.ln1.gain)
                    + per(&b.ln1.bias)
                    + per(&b.ln2.gain)
                    + per(&b.ln2.bias)
            })
            .sum();
        let head = if self.tied { 0 } else { self.head.bytes() };
        blocks + self.wte.bytes() + per(&self.wpe) + head + per(&self.ln_f.gain)
            + per(&self.ln_f.bias)
    }
}

/// Prompt tokens per prefill pass. GPT-2's whole context is 1024, so this is
/// less about memory than about the Llama side's reason: one matmul per layer
/// beats `m` of them, up to the point where the attention matrix stops fitting.
const PREFILL_CHUNK: usize = 512;

impl Session for GpuGpt2 {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let mut last = Vec::new();
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            last = self.run(chunk)?;
        }
        Ok(last)
    }

    fn cached(&self) -> usize {
        self.pos
    }

    fn truncate(&mut self, len: usize) -> Res<usize> {
        self.rewind(len)?;
        // Keys and values rewind exactly, and `rewind` clamps a request for
        // more than is held — so the position afterwards is the answer. No
        // backend here carries a recurrent state that would refuse; see
        // `kvad::model::KvCache::truncate`.
        Ok(self.pos)
    }

    fn label(&self) -> String {
        self.device_label()
    }

    fn param_count(&self) -> usize {
        self.params()
    }

    fn weight_bytes(&self) -> usize {
        self.memory_bytes()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use kvad::model::{Arch, Transformer};
    use std::collections::HashMap;

    /// Small enough to build from random numbers, and every dimension a
    /// multiple of 32 so the quantisers will take it.
    pub(crate) fn tiny_spec(tie: bool) -> Spec {
        Spec {
            arch: Arch::require("gpt2"),
            n_layer: 2,
            n_head: 2,
            n_kv_head: 2,
            n_embd: 64,
            head_dim: 32,
            n_ctx: 16,
            vocab_size: 64,
            intermediate: 256,
            eps: 1e-5,
            rope_theta: 10000.0,
            tie_embeddings: tie,
            cache: kvad::model::CacheLayout::uniform(kvad::model::CacheShape::kv(32, 32)),
            config: kvad::model::Json::default(),
        }
    }

    /// A GPT-2 checkpoint in the layout the real ones use: `Conv1D` weights
    /// stored `[in, out]`, a bias beside every projection, and the causal mask
    /// `torch` had nowhere else to put.
    pub(crate) fn write_tensors(
        spec: &Spec,
        extra: &[(String, Tensor)],
        tag: &str,
    ) -> std::path::PathBuf {
        let d = Device::Cpu;
        let (e, i) = (spec.n_embd, spec.intermediate);
        let scaled =
            |sd: f32, r: usize, c: usize| Tensor::randn(0f32, sd, (r, c), &d).unwrap();
        let rand = |r: usize, c: usize| scaled(0.02, r, c);
        // Norm gains and biases are random too. All-ones and all-zeros is the
        // trap: it makes LayerNorm's affine step the identity, so a backend
        // that dropped the gain or the bias would still look right.
        let vec1 = |n: usize| Tensor::randn(1f32, 0.1f32, n, &d).unwrap();

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("wte.weight".into(), rand(spec.vocab_size, e));
        t.insert("wpe.weight".into(), rand(spec.n_ctx, e));
        t.insert("ln_f.weight".into(), vec1(e));
        t.insert("ln_f.bias".into(), vec1(e));
        for l in 0..spec.n_layer {
            let p = format!("h.{l}");
            for (name, tensor) in [
                ("ln_1.weight", vec1(e)),
                ("ln_1.bias", vec1(e)),
                ("ln_2.weight", vec1(e)),
                ("ln_2.bias", vec1(e)),
                ("attn.c_attn.weight", rand(e, 3 * e)),
                ("attn.c_attn.bias", vec1(3 * e)),
                ("attn.c_proj.weight", rand(e, e)),
                ("attn.c_proj.bias", vec1(e)),
                // Larger than the rest on purpose, so the MLP's
                // pre-activations land near +/-2.7, which is where GELU's two
                // forms are furthest apart. At 0.02 they agree to a thousandth
                // and the wrong one passes; much larger and they converge
                // again, because the gap closes at both tails. `c_proj` below
                // stays small, so the residual stream and the logits do not.
                ("mlp.c_fc.weight", scaled(0.25, e, i)),
                ("mlp.c_fc.bias", vec1(i)),
                ("mlp.c_proj.weight", rand(i, e)),
                ("mlp.c_proj.bias", vec1(e)),
            ] {
                t.insert(format!("{p}.{name}"), tensor);
            }
            // Lower-triangular, as `torch` saves it. Nothing reads it.
            let mask: Vec<f32> = (0..spec.n_ctx * spec.n_ctx)
                .map(|k| ((k % spec.n_ctx) <= (k / spec.n_ctx)) as u8 as f32)
                .collect();
            t.insert(
                format!("{p}.attn.bias"),
                Tensor::from_vec(mask, (1, 1, spec.n_ctx, spec.n_ctx), &d).unwrap(),
            );
        }
        if !spec.tie_embeddings {
            t.insert("lm_head.weight".into(), rand(spec.vocab_size, e));
            t.insert("lm_head.bias".into(), vec1(spec.vocab_size));
        }
        for (name, tensor) in extra {
            t.insert(name.clone(), tensor.clone());
        }

        let path = std::env::temp_dir()
            .join(format!("gpu-gpt2-{}-{tag}.safetensors", std::process::id()));
        candle_core::safetensors::save(&t, &path).unwrap();
        path
    }

    fn cpu_model(path: &std::path::PathBuf, spec: &Spec) -> kvad::model::gpt2::Model {
        let ckpt = kvad::weights::Checkpoint::open(std::slice::from_ref(path)).unwrap();
        let src = kvad::qcache::Live::new(&ckpt, kvad::quant::Precision::F32);
        kvad::model::gpt2::Model::load(&src, spec.clone()).unwrap()
    }

    /// The largest disagreement between the two engines on one checkpoint.
    ///
    /// Both in f32, so what is left is the order the sums happen in. Anything
    /// above a rounding error means they are running different models, which
    /// for a backend written from a paper rather than from weights is the only
    /// check that can catch a transposed matrix or a dropped bias.
    fn engines_differ_by(path: &std::path::PathBuf, spec: &Spec, tokens: &[u32]) -> f32 {
        let mut gpu =
            GpuGpt2::load(
                std::slice::from_ref(path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap();
        let mine = gpu.forward(tokens).unwrap();

        let cpu = cpu_model(path, spec);
        let mut cache = kvad::model::KvCache::new(spec);
        let theirs = cpu.forward_batch(tokens, &mut cache);

        assert_eq!(mine.len(), theirs.len());
        mine.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
    }

    /// The whole block, against the engine that was written by hand first.
    ///
    /// Every difference GPT-2 has from Llama is in here at once: the learned
    /// positions, LayerNorm's centring and bias, the fused QKV split, GELU, and
    /// the `Conv1D` transpose that this backend has to undo in one mode and not
    /// the other. A backend that got any of them wrong would still run.
    #[test]
    fn the_block_agrees_with_the_cpu_engine() {
        for tie in [true, false] {
            let spec = tiny_spec(tie);
            let path = write_tensors(&spec, &[], &format!("agree-{tie}"));

            // More than one token, so the causal mask and the batched attention
            // are exercised rather than the single-token path.
            // Tighter than the Llama side's 1e-4, and deliberately: GELU's
            // two forms differ by at most 0.00047, and at this fixture's scale
            // that reaches the logits as 3e-5. Measured across a dozen runs the
            // agreeing case stays under 2e-6 and the disagreeing one never drops
            // below 2.7e-5, so 1e-5 separates them with room on both sides.
            let worst = engines_differ_by(&path, &spec, &[1, 2, 3, 4, 5]);
            assert!(worst < 1e-5, "logits disagree by {worst} (tie = {tie})");

            std::fs::remove_file(&path).unwrap();
        }
    }

    /// Decoding one token at a time through the cache must land where running
    /// the whole prompt at once does.
    #[test]
    fn the_cache_agrees_with_a_single_pass() {
        let spec = tiny_spec(true);
        let path = write_tensors(&spec, &[], "cache");
        let load = || {
            GpuGpt2::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap()
        };

        let mut at_once = load();
        let whole = at_once.forward(&[1, 2, 3, 4]).unwrap();

        let mut stepped = load();
        let mut last = Vec::new();
        for t in [1u32, 2, 3, 4] {
            last = stepped.forward(&[t]).unwrap();
        }

        let worst =
            whole.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-4, "the cache changes the answer by {worst}");
        assert_eq!(stepped.cached(), 4);

        std::fs::remove_file(&path).unwrap();
    }

    /// Both engines have to count the same model, including the biases that are
    /// on every projection here and on almost none in the Llama family.
    #[test]
    fn both_engines_count_the_same_parameters() {
        for tie in [true, false] {
            let spec = tiny_spec(tie);
            let path = write_tensors(&spec, &[], &format!("params-{tie}"));
            let gpu = GpuGpt2::load(
                std::slice::from_ref(&path),
                spec.clone(),
                DType::F32,
                None,
                Device::Cpu,
                &Vault::off(),
            )
            .unwrap();
            assert_eq!(gpu.param_count(), cpu_model(&path, &spec).param_count());
            std::fs::remove_file(&path).unwrap();
        }
    }

    /// `gelu` is the tanh approximation and `gelu_erf` is the exact form, and
    /// GPT-2 was trained with the first.
    ///
    /// The cross-engine test catches a backend that picks the wrong one, but
    /// only because its fixture is scaled to make them separate. This says the
    /// fact on its own, so that the scale over there is a decision with a
    /// reason rather than a number that happens to work.
    #[test]
    fn the_gelu_is_the_one_gpt2_was_trained_with() {
        let d = Device::Cpu;
        let xs: Vec<f32> = (-40..=40).map(|i| i as f32 / 10.0).collect();
        let t = Tensor::from_vec(xs.clone(), xs.len(), &d).unwrap();

        // The hand-written engine's, which is OpenAI's.
        let mut want = xs.clone();
        kvad::tensor::gelu_inplace(&mut want);

        let got = t.gelu().unwrap().to_vec1::<f32>().unwrap();
        let worst =
            got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-6, "candle's `gelu` is not the tanh approximation: off by {worst}");

        // And the exact form is a different function, so this is a choice
        // rather than a coincidence.
        let erf = t.gelu_erf().unwrap().to_vec1::<f32>().unwrap();
        let apart =
            erf.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(apart > 1e-4, "the two GELUs are indistinguishable here, so this proves nothing");
    }

    /// GPT-2's stored causal mask is not an omission, and a check that refused
    /// it would do the harm it exists to prevent. Every fixture above carries
    /// one, so this only has to say out loud that they load.
    #[test]
    fn the_stored_causal_mask_is_not_an_omission() {
        let spec = tiny_spec(true);
        let path = write_tensors(&spec, &[], "mask");
        assert!(GpuGpt2::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        )
        .is_ok());
        std::fs::remove_file(&path).unwrap();
    }

    /// And a weight this backend has never heard of stops the load, because
    /// running a GPT-2 forward pass over a model that is not quite GPT-2 is the
    /// mistake that started all of this.
    #[test]
    fn a_tensor_this_backend_never_reads_refuses_to_load() {
        let spec = tiny_spec(true);
        let extra =
            [("h.0.attn.q_norm.weight".to_string(), Tensor::ones(32, DType::F32, &Device::Cpu).unwrap())];
        let path = write_tensors(&spec, &extra, "unknown");

        let error = match GpuGpt2::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a weight nobody reads must not load quietly"),
        };
        assert!(error.contains("q_norm"), "it should name the tensor: {error}");

        std::fs::remove_file(&path).unwrap();
    }

    /// An untied model with no head of its own is a broken checkpoint, and must
    /// not fall through to the embedding table.
    #[test]
    fn an_untied_model_without_a_head_says_so() {
        let mut spec = tiny_spec(true);
        let path = write_tensors(&spec, &[], "headless");
        spec.tie_embeddings = false;

        let error = match GpuGpt2::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an untied model with no `lm_head.weight` must not load"),
        };
        assert!(error.contains("lm_head.weight"), "unhelpful message: {error}");
        assert!(error.contains("tie_word_embeddings"), "{error}");

        std::fs::remove_file(&path).unwrap();
    }
}
