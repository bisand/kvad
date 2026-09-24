//! Qwen3.5, Qwen3.8 and Qwen3-Next on the GPU: a state per layer, not a cache.
//!
//! Read this next to [`kvad::model::qwen3_5`], which explains what a gated
//! delta net is and why three layers in four have one. The arithmetic is the
//! same; what changes is that the recurrence is written with whole-tensor
//! operations instead of a loop over heads.
//!
//! ```text
//!     S <- S · e^g ;  δ = (v − Sᵀk)·β ;  S <- S + k⊗δ ;  y = Sᵀq
//! ```
//!
//! `S` is `[v_heads, key_dim, value_dim]` and every line above is one
//! broadcast against it — six kernels a layer rather than six per head. What
//! is *not* batched is the sequence: each token's state depends on the last
//! one's, so a prompt is a loop here in a way it is not for attention. The
//! reference implementation has a chunked formulation that fixes this, and it
//! wants its own session with a model to check against.
//!
//! The convolution keeps its window as a `[conv_dim, kernel − 1]` tensor and
//! slides it with `cat` and `narrow`, which is the same two lines the CPU
//! engine writes as an index shift.

use crate::common::{
    check_block, embedding, label, linear, unread, unread_error, Embed, KvCache, kv_store, Loader, Proj, Reader,
    Stored,
};
use crate::ffn::{Ffn, Mlp, Moe};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, rotary_emb, VarBuilder};
use kvad::model::ffn::{Layout, Router};
use kvad::model::qwen3_5::{Delta, Family, Kind};
use kvad::model::{Session, Spec};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// How a checkpoint spells the delta net's input projections. The mirror of
/// [`kvad::model::qwen3_5`]'s: Qwen3.5 writes four matrices in head order,
/// Qwen3-Next writes two with the heads interleaved.
enum Inputs {
    Split { qkv: Proj, z: Proj, b: Proj, a: Proj },
    Fused { qkvz: Proj, ba: Proj },
}

impl Inputs {
    fn load(ld: &Loader, vb: &Reader<'_>, e: usize, d: &Delta, family: Family) -> Res<Self> {
        let p = |name: &str, out: usize| ld.proj(vb, name, out, e, Stored::OutIn);
        Ok(match family {
            Family::Qwen35 => Inputs::Split {
                qkv: p("in_proj_qkv.weight", d.conv_dim)?,
                z: p("in_proj_z.weight", d.value_dim)?,
                b: p("in_proj_b.weight", d.n_v_head)?,
                a: p("in_proj_a.weight", d.n_v_head)?,
            },
            Family::Next => Inputs::Fused {
                qkvz: p("in_proj_qkvz.weight", d.key_dim * 2 + d.value_dim * 2)?,
                ba: p("in_proj_ba.weight", d.n_v_head * 2)?,
            },
        })
    }

    /// `(qkv, z, b, a)` for one token: the convolution's input as
    /// `[conv_dim, 1]`, the output gate as `[v_heads, v_head]`, and the write
    /// strength and decay as `[v_heads]` in f32.
    ///
    /// Qwen3-Next's two matrices are laid out one *key* head at a time, which
    /// is a `[n_k_head, stride]` view and four `narrow`s — the reshape the
    /// split spelling has to do by hand falls out of the layout here.
    fn project(&self, h: &Tensor, d: &Delta) -> Res<(Tensor, Tensor, Tensor, Tensor)> {
        let (nv, kh, vh, group) = (d.n_v_head, d.k_head, d.v_head, d.group());
        Ok(match self {
            Inputs::Split { qkv, z, b, a } => (
                qkv.forward(h)?.reshape((d.conv_dim, 1))?,
                z.forward(h)?.reshape((nv, vh))?,
                b.forward(h)?.reshape(nv)?.to_dtype(DType::F32)?,
                a.forward(h)?.reshape(nv)?.to_dtype(DType::F32)?,
            ),
            Inputs::Fused { qkvz, ba } => {
                let wide = group * vh;
                let m = qkvz.forward(h)?.reshape((d.n_k_head, 2 * kh + 2 * wide))?;
                let part = |at: usize, n: usize| m.narrow(1, at, n)?.contiguous();
                let qkv = Tensor::cat(
                    &[
                        part(0, kh)?.flatten_all()?,
                        part(kh, kh)?.flatten_all()?,
                        part(2 * kh, wide)?.flatten_all()?,
                    ],
                    0,
                )?
                .reshape((d.conv_dim, 1))?;
                let z = part(2 * kh + wide, wide)?.reshape((nv, vh))?;

                let m = ba.forward(h)?.reshape((d.n_k_head, 2 * group))?;
                let part = |at: usize| m.narrow(1, at, group)?.contiguous()?.reshape(nv);
                (qkv, z, part(0)?.to_dtype(DType::F32)?, part(group)?.to_dtype(DType::F32)?)
            }
        })
    }

    fn params(&self) -> usize {
        match self {
            Inputs::Split { qkv, z, b, a } => {
                qkv.params() + z.params() + b.params() + a.params()
            }
            Inputs::Fused { qkvz, ba } => qkvz.params() + ba.params(),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Inputs::Split { qkv, z, b, a } => qkv.bytes() + z.bytes() + b.bytes() + a.bytes(),
            Inputs::Fused { qkvz, ba } => qkvz.bytes() + ba.bytes(),
        }
    }
}

struct DeltaNet {
    inputs: Inputs,
    /// `[conv_dim, kernel]`, one filter per channel.
    conv: Tensor,
    dt_bias: Tensor,
    /// `-exp(A_log)`, folded once at load: the decay is always used this way
    /// and the exponential does not depend on the token.
    neg_a: Tensor,
    norm: Tensor,
    out: Proj,
}

struct Attn {
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    q_norm: Tensor,
    k_norm: Tensor,
}

enum Mixer {
    Linear(Box<DeltaNet>),
    Full(Box<Attn>),
}

struct Block {
    attn_norm: Tensor,
    mixer: Mixer,
    mlp_norm: Tensor,
    mlp: Mlp,
}

/// What one linear layer carries between tokens.
struct State {
    /// `[conv_dim, kernel - 1]`
    conv: Tensor,
    /// `[v_heads, key_head, value_head]`
    s: Tensor,
}

pub struct GpuQwen35 {
    spec: Spec,
    delta: Delta,
    kinds: Vec<Kind>,
    device: Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    embed: Embed,
    head: Proj,
    tied: bool,
    blocks: Vec<Block>,
    final_norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    rope_dim: usize,
    /// Per full-attention layer, `[1, n_kv, seq, head_dim]`.
    kv: Vec<KvCache>,
    /// Per linear layer.
    state: Vec<Option<State>>,
    pos: usize,
}

impl GpuQwen35 {
    pub fn load(
        paths: &[std::path::PathBuf],
        spec: Spec,
        dtype: DType,
        quant: Option<GgmlDType>,
        device: Device,
        cache: &Vault,
    ) -> Res<Self> {
        let delta = Delta::read(&spec.config)?;
        let kinds = kvad::model::qwen3_5::layer_kinds(&spec.config, spec.n_layer)?;
        let family = Family::of(&spec);
        // The same two questions the CPU loader asks, asked the same way, so
        // the two backends cannot disagree about which layers route.
        let routed = match Router::count(&spec.config) {
            None => None,
            Some(_) => Some((Router::read(&spec.config)?, Layout::read(&spec.config))),
        };
        let expert = spec.config.num(&["moe_intermediate_size"]).unwrap_or(spec.intermediate);
        let load_dtype = if quant.is_some() { DType::F32 } else { dtype };
        let compute = load_dtype;
        let (e, hd) = (spec.n_embd, spec.head_dim);
        let (qd, kvd) = (spec.n_head * hd, spec.kv_dim());

        let mut widths = vec![
            ("hidden size", e),
            ("MLP width", spec.intermediate),
            ("attention output", qd),
            ("KV width", kvd),
            ("linear key width", delta.key_dim),
            ("linear value width", delta.value_dim),
        ];
        if let Some((_, layout)) = &routed {
            // An expert is far narrower than a dense layer — 512 against 5120
            // on Qwen3-Next — and it is the expert's width the quantiser's
            // block size has to divide.
            widths.push(("expert width", expert));
            let shared = layout.shared.width(expert);
            if shared > 0 {
                widths.push(("shared expert width", shared));
            }
        }
        check_block(quant, &widths)?;

        let load_dev = Device::Cpu;
        // SAFETY: candle memory-maps the checkpoints; they are read-only cache
        // entries that nothing else writes while we hold them.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, load_dtype, &load_dev)? };
        let vb = Reader::new(vb);
        // Qwen3.5 is one level deeper than every other family — its text
        // model is part of a multimodal wrapper — and Qwen3-Next is not.
        let model = match family.prefix() {
            "" => vb.pp("model"),
            _ => vb.pp("model").pp("language_model"),
        };

        let to_dev = |t: Tensor| -> Res<Tensor> { Ok(t.to_device(&device)?) };
        // This family's plain RMSNorm scales by `1 + w`: the stored vector is
        // an offset from one, initialised to zeros, which is Gemma's
        // convention and not the one `ops::rms_norm` implements. Folding the
        // one in at load keeps the hot path the ordinary kernel — and matches
        // what the CPU loader does, so the two cannot drift apart.
        //
        // `linear_attn.norm` is deliberately not read this way: the gated norm
        // in the same checkpoint scales by `w` itself.
        let offset = |t: Tensor| -> Res<Tensor> { Ok((t + 1.0)?.to_device(&device)?) };
        let ld = Loader::new(quant, device.clone(), cache);
        let load_t = |vb: &Reader<'_>, name: &str, out: usize, inp: usize| -> Res<Proj> {
            ld.proj(vb, name, out, inp, Stored::OutIn)
        };

        let mut blocks = Vec::with_capacity(spec.n_layer);
        for (i, kind) in kinds.iter().enumerate() {
            let l = model.pp(format!("layers.{i}"));
            let mlp = l.pp("mlp");
            let mixer = match kind {
                Kind::Full => {
                    let sa = l.pp("self_attn");
                    Mixer::Full(Box::new(Attn {
                        // Twice the queries: the second half of each head is
                        // the output gate.
                        q: load_t(&sa, "q_proj.weight", qd * 2, e)?,
                        k: load_t(&sa, "k_proj.weight", kvd, e)?,
                        v: load_t(&sa, "v_proj.weight", kvd, e)?,
                        o: load_t(&sa, "o_proj.weight", e, qd)?,
                        q_norm: offset(sa.get(hd, "q_norm.weight")?)?,
                        k_norm: offset(sa.get(hd, "k_norm.weight")?)?,
                    }))
                }
                Kind::Linear => {
                    let la = l.pp("linear_attn");
                    let a_log = la.get(delta.n_v_head, "A_log")?;
                    Mixer::Linear(Box::new(DeltaNet {
                        inputs: Inputs::load(&ld, &la, e, &delta, family)?,
                        conv: to_dev(
                            la.conv(delta.conv_dim, delta.conv, "conv1d.weight")?
                                .to_dtype(compute)?,
                        )?,
                        dt_bias: to_dev(la.get(delta.n_v_head, "dt_bias")?.to_dtype(DType::F32)?)?,
                        neg_a: to_dev(a_log.to_dtype(DType::F32)?.exp()?.neg()?)?,
                        norm: to_dev(la.get(delta.v_head, "norm.weight")?)?,
                        out: load_t(&la, "out_proj.weight", e, delta.value_dim)?,
                    }))
                }
            };
            blocks.push(Block {
                attn_norm: offset(l.get(e, "input_layernorm.weight")?)?,
                mixer,
                mlp_norm: offset(l.get(e, "post_attention_layernorm.weight")?)?,
                mlp: match &routed {
                    Some((router, layout)) if layout.is_moe(i, router.n_experts) => {
                        Mlp::Moe(Box::new(Moe::load(&ld, &mlp, e, expert, router, layout.shared)?))
                    }
                    _ => Mlp::Dense(Ffn::load(&ld, &mlp, e, spec.intermediate)?),
                },
            });
        }

        // The parts this does not run. Named rather than ignored, so the check
        // below can tell a decision from an omission. Both families ship a
        // multi-token-prediction head; the vision tower is Qwen3.5's alone.
        vb.skip_under("mtp.");
        if family == Family::Qwen35 {
            vb.skip_under("model.visual.");
        }

        let (embed, head, tied) = embedding(&ld, &vb, &model, "embed_tokens.weight", &spec)?;
        let final_norm = offset(model.get(e, "norm.weight")?)?;

        let left = unread(paths, &vb.seen(), &vb.skipped())?;
        if !left.is_empty() {
            return Err(unread_error(spec.arch.id(), &left).into());
        }

        // Partial RoPE: only the first `rope_dim` of each head rotates.
        let rope_dim = kvad::model::qwen3_5::rope_dim(&spec);
        let half = rope_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / spec.rope_theta.powf(2.0 * i as f32 / rope_dim as f32))
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), &device)?;
        let positions: Vec<f32> = (0..spec.n_ctx).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (spec.n_ctx, 1), &device)?;
        let angles = positions.matmul(&inv)?;
        let cos = angles.cos()?.to_dtype(compute)?;
        let sin = angles.sin()?.to_dtype(compute)?;

        crate::model::settled(&device)?;

        Ok(GpuQwen35 {
            kv: (0..spec.n_layer).map(|_| KvCache::new(2).stored_as(kv_store(&device, quant))).collect(),
            state: (0..spec.n_layer).map(|_| None).collect(),
            blocks,
            embed,
            head,
            tied,
            final_norm,
            cos,
            sin,
            rope_dim,
            delta,
            kinds,
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
}

impl GpuQwen35 {
    /// `x / ‖x‖` along the last axis, which the delta rule wants on its
    /// queries and keys so that `Sᵀk` reads the state rather than rescaling it.
    fn l2(x: &Tensor) -> candle_core::Result<Tensor> {
        let norm = x.sqr()?.sum_keepdim(x.rank() - 1)?.sqrt()?.clamp(1e-6, f64::INFINITY)?;
        x.broadcast_div(&norm)
    }

    /// One token through a gated delta net.
    fn linear_step(&self, net: &DeltaNet, h: &Tensor, layer: usize) -> Res<(Tensor, State)> {
        let d = &self.delta;
        let (nv, kh, vh) = (d.n_v_head, d.k_head, d.v_head);

        // ---- the depthwise causal convolution ---------------------------
        // The window already holds the previous `kernel − 1` inputs, so
        // appending this one makes the whole filter's view, and dropping the
        // oldest column afterwards is the slide.
        let (qkv, z, b, a) = net.inputs.project(h, d)?;
        let state = self.state[layer].as_ref().ok_or("linear layer has no state")?;
        let window = Tensor::cat(&[&state.conv, &qkv], 1)?;
        let conved = window.mul(&net.conv)?.sum(1)?;
        let conved = ops::silu(&conved)?;
        let next_conv = window.narrow(1, 1, d.conv - 1)?.contiguous()?;

        let q = conved.narrow(0, 0, d.key_dim)?.reshape((d.n_k_head, kh))?;
        let k = conved.narrow(0, d.key_dim, d.key_dim)?.reshape((d.n_k_head, kh))?;
        let v = conved.narrow(0, 2 * d.key_dim, d.value_dim)?.reshape((nv, vh))?;

        // Query and key heads are shared: `group` value heads read each one.
        let group = d.group();
        let spread = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.unsqueeze(1)?.expand((d.n_k_head, group, kh))?.reshape((nv, kh))
        };
        let q = Self::l2(&spread(&q)?)?;
        let q = (q * (1.0 / (kh as f64).sqrt()))?;
        let k = Self::l2(&spread(&k)?)?;

        let beta = ops::sigmoid(&b)?;
        // `softplus`, and then the decay. Both in f32 whatever the activations
        // are: the reference is explicit that `A` becomes `-inf` in fp16.
        let sp = ((a + &net.dt_bias)?.exp()? + 1.0)?.log()?;
        let decay = net.neg_a.mul(&sp)?.exp()?.to_dtype(self.dtype)?;
        let beta = beta.to_dtype(self.dtype)?;

        // S <- S·e^g ; δ = (v − Sᵀk)·β ; S <- S + k⊗δ ; y = Sᵀq
        let s = state.s.broadcast_mul(&decay.reshape((nv, 1, 1))?)?;
        let k3 = k.reshape((nv, kh, 1))?;
        let mem = s.broadcast_mul(&k3)?.sum(1)?;
        let delta = (v - mem)?.broadcast_mul(&beta.reshape((nv, 1))?)?;
        let s = (s + k3.broadcast_mul(&delta.reshape((nv, 1, vh))?)?)?;
        let out = s.broadcast_mul(&q.reshape((nv, kh, 1))?)?.sum(1)?;

        // The gated norm scales by `w` itself — not by `1 + w`, which is what
        // every *other* norm in this architecture does.
        let out = ops::rms_norm(&out.contiguous()?, &net.norm, self.spec.eps)?;
        let out = out.mul(&ops::silu(&z)?)?;

        let out = net.out.forward(&out.reshape((1, d.value_dim))?)?;
        Ok((out, State { conv: next_conv, s }))
    }

    /// One token through a full-attention layer.
    fn full_step(&self, attn: &Attn, h: &Tensor, cache: &mut KvCache) -> Res<Tensor> {
        let spec = &self.spec;
        let (hd, nh, nkv) = (spec.head_dim, spec.n_head, spec.n_kv_head);
        let pos = self.pos;

        // `q_proj` is twice as wide as the queries: per head, the query then
        // its output gate.
        let qg = linear(h, &attn.q, None)?.reshape((1, nh, hd * 2))?;
        let q = qg.narrow(2, 0, hd)?.contiguous()?;
        let gate = qg.narrow(2, hd, hd)?.contiguous()?.reshape((1, nh * hd))?;

        let q = ops::rms_norm(&q, &attn.q_norm, spec.eps)?.reshape((1, 1, nh, hd))?;
        let k = linear(h, &attn.k, None)?.reshape((1, 1, nkv, hd))?;
        let k = ops::rms_norm(&k.contiguous()?, &attn.k_norm, spec.eps)?;
        let v = linear(h, &attn.v, None)?.reshape((1, 1, nkv, hd))?;

        // Partial RoPE: the first `rope_dim` of each head rotates, the rest
        // carries no position at all.
        let cos = self.cos.narrow(0, pos, 1)?.contiguous()?;
        let sin = self.sin.narrow(0, pos, 1)?.contiguous()?;
        let part = |t: &Tensor| -> candle_core::Result<Tensor> {
            let t = t.transpose(1, 2)?.contiguous()?;
            let rot = t.narrow(3, 0, self.rope_dim)?.contiguous()?;
            let rot = rotary_emb::rope(&rot, &cos, &sin)?;
            match self.rope_dim == hd {
                true => Ok(rot),
                false => {
                    let pass = t.narrow(3, self.rope_dim, hd - self.rope_dim)?.contiguous()?;
                    Tensor::cat(&[rot, pass], 3)?.contiguous()
                }
            }
        };
        let q = part(&q)?;
        let k = part(&k)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (k, v) = cache.push(&k, &v)?;

        // `attention` folds the query heads onto their KV head rather than
        // copying the KV head out per query head, and takes the fused kernel
        // where it is trustworthy. One token, so no mask: the cache holds
        // only earlier positions and there is nothing ahead to hide.
        let scale = 1.0 / (hd as f64).sqrt();
        let out = crate::model::attention(&q, &k, &v, None, scale)?;
        let out = out.transpose(1, 2)?.reshape((1, nh * hd))?;

        // The gate, which is what `attn_output_gate` names.
        let out = out.mul(&ops::sigmoid(&gate)?)?;
        Ok(linear(&out, &attn.o, None)?)
    }
}

impl GpuQwen35 {
    /// Run `tokens` and return logits for the last one.
    ///
    /// A loop, and not a batched pass. A recurrent layer's state at position
    /// `t` depends on its state at `t − 1`, so a prompt cannot be one matmul
    /// the way attention's can. The reference implementation has a chunked
    /// formulation that recovers most of it; writing that wants a model to
    /// check against, and this is correct in the meantime.
    fn run(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        let spec = self.spec.clone();
        let mut last = Vec::new();
        for &token in tokens {
            let ids = Tensor::from_slice(&[token], (1,), &self.device)?;
            let mut x = self.embed.rows(&ids)?.to_dtype(self.dtype)?;

            for l in 0..self.blocks.len() {
                let h = ops::rms_norm(&x, &self.blocks[l].attn_norm, spec.eps)?;
                let mixed = match &self.blocks[l].mixer {
                    Mixer::Full(attn) => {
                        // Lent out for the step: `full_step` reads the rest
                        // of `self` while it writes this.
                        let mut cache = std::mem::replace(&mut self.kv[l], KvCache::new(2));
                        let out = self.full_step(attn, &h, &mut cache);
                        self.kv[l] = cache;
                        out?
                    }
                    Mixer::Linear(net) => {
                        let (out, state) = self.linear_step(net, &h, l)?;
                        self.state[l] = Some(state);
                        out
                    }
                };
                x = (x + mixed)?;

                let b = &self.blocks[l];
                let h = ops::rms_norm(&x, &b.mlp_norm, spec.eps)?;
                x = (x + b.mlp.forward(&h, 1, spec.n_embd)?)?;
            }
            self.pos += 1;

            let h = ops::rms_norm(&x, &self.final_norm, spec.eps)?;
            let logits = self.head.forward(&h)?.to_dtype(DType::F32)?;
            last = logits.flatten_all()?.to_vec1::<f32>()?;
        }
        Ok(last)
    }

    /// Start every layer's state from nothing.
    fn reset_state(&mut self) -> Res<()> {
        let d = &self.delta;
        for l in 0..self.blocks.len() {
            self.kv[l].truncate(0);
            self.state[l] = match self.kinds[l] {
                Kind::Full => None,
                Kind::Linear => Some(State {
                    conv: Tensor::zeros(
                        (d.conv_dim, d.conv - 1),
                        self.dtype,
                        &self.device,
                    )?,
                    s: Tensor::zeros(
                        (d.n_v_head, d.k_head, d.v_head),
                        self.dtype,
                        &self.device,
                    )?,
                }),
            };
        }
        self.pos = 0;
        Ok(())
    }

    fn params(&self) -> usize {
        let n = |t: &Tensor| t.elem_count();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let mixer = match &b.mixer {
                    Mixer::Full(a) => {
                        a.q.params()
                            + a.k.params()
                            + a.v.params()
                            + a.o.params()
                            + n(&a.q_norm)
                            + n(&a.k_norm)
                    }
                    Mixer::Linear(d) => {
                        d.inputs.params()
                            + n(&d.conv)
                            + n(&d.dt_bias)
                            + n(&d.neg_a)
                            + n(&d.norm)
                            + d.out.params()
                    }
                };
                mixer
                    + b.mlp.params()
                    + n(&b.attn_norm)
                    + n(&b.mlp_norm)
            })
            .sum();
        let head = match self.spec.tie_embeddings {
            true => 0,
            false => self.head.params(),
        };
        blocks + self.embed.params() + head + n(&self.final_norm)
    }

    fn memory_bytes(&self) -> usize {
        let per = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let blocks: usize = self
            .blocks
            .iter()
            .map(|b| {
                let mixer = match &b.mixer {
                    Mixer::Full(a) => a.q.bytes() + a.k.bytes() + a.v.bytes() + a.o.bytes(),
                    Mixer::Linear(d) => d.inputs.bytes() + per(&d.conv) + d.out.bytes(),
                };
                mixer + b.mlp.bytes()
            })
            .sum();
        let head = if self.tied { 0 } else { self.head.bytes() };
        blocks + self.embed.bytes() + head
    }
}

impl Session for GpuQwen35 {
    fn spec(&self) -> &Spec {
        &self.spec
    }

    fn forward(&mut self, tokens: &[u32]) -> Res<Vec<f32>> {
        if self.pos == 0 {
            self.reset_state()?;
        }
        self.run(tokens)
    }

    fn cached(&self) -> usize {
        self.pos
    }

    /// Rewind, or say that it could not.
    ///
    /// This is the first backend here that ever answers anything but `len`.
    /// Three layers in four hold a recurrent state that has already absorbed
    /// the tokens being dropped, and no subtraction takes them back out — so
    /// a partial rewind clears everything and reports `0`, and the caller
    /// re-prefills from the start. See
    /// [`kvad::model::KvCache::truncate`], which is where this engine's
    /// answer to that is recorded, and the CPU side of the same decision.
    fn truncate(&mut self, len: usize) -> Res<usize> {
        if len >= self.pos {
            return Ok(self.pos);
        }
        self.reset_state()?;
        Ok(0)
    }

    fn label(&self) -> String {
        self.device_label()
    }

    fn param_count(&self) -> usize {
        self.params()
    }

    fn kv_number_bytes(&self) -> usize {
        self.kv.first().map_or(self.dtype.size_in_bytes(), |c| c.number_bytes(self.dtype))
    }

    /// Read off a state when there is one, and otherwise what `reset_state`
    /// will make: the compute dtype.
    fn state_number_bytes(&self) -> usize {
        let held = self.state.iter().flatten().next();
        held.map_or(self.dtype.size_in_bytes(), |s| s.s.dtype().size_in_bytes())
    }

    fn weight_bytes(&self) -> usize {
        self.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::model::{Json, Transformer};
    use kvad::serde_json;
    use std::collections::HashMap;

    /// A model small enough to build from random numbers, with both kinds of
    /// layer and every width a multiple of 32 so the quantisers will take it.
    ///
    /// `next` picks the other family in this module: the same mixer with its
    /// inputs fused one key head at a time, a mixture instead of an MLP, and
    /// its decoder where every Llama keeps one. `tag` because these tests run
    /// in parallel and would otherwise delete each other's checkpoints.
    fn tiny_of(next: bool, tag: &str) -> (Spec, std::path::PathBuf) {
        let (e, hd, nh, nkv) = (64usize, 32usize, 4usize, 2usize);
        let (nk, nv, kh, vh, conv) = (2usize, 4usize, 32usize, 32usize, 4usize);
        let (n_layer, vocab, inter) = (4usize, 64usize, 128usize);
        let (n_experts, top_k, expert, shared) = (8usize, 2usize, 32usize, 32usize);
        let (key_dim, value_dim) = (nk * kh, nv * vh);
        let conv_dim = key_dim * 2 + value_dim;
        let kinds: Vec<&str> = (0..n_layer)
            .map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" })
            .collect();

        let config = match next {
            true => serde_json::json!({
                "model_type": "qwen3_next",
                "num_hidden_layers": n_layer,
                "num_attention_heads": nh,
                "num_key_value_heads": nkv,
                "head_dim": hd,
                "hidden_size": e,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
                "tie_word_embeddings": false,
                "full_attention_interval": 4,
                "linear_num_key_heads": nk,
                "linear_num_value_heads": nv,
                "linear_key_head_dim": kh,
                "linear_value_head_dim": vh,
                "linear_conv_kernel_dim": conv,
                "num_experts": n_experts,
                "num_experts_per_tok": top_k,
                "moe_intermediate_size": expert,
                "shared_expert_intermediate_size": shared,
                "norm_topk_prob": true,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            }),
            false => serde_json::json!({
                "model_type": "qwen3_5",
                "text_config": {
                    "num_hidden_layers": n_layer,
                    "num_attention_heads": nh,
                    "num_key_value_heads": nkv,
                    "head_dim": hd,
                    "hidden_size": e,
                    "intermediate_size": inter,
                    "vocab_size": vocab,
                    "max_position_embeddings": 64,
                    "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": false,
                    "layer_types": kinds,
                    "linear_num_key_heads": nk,
                    "linear_num_value_heads": nv,
                    "linear_key_head_dim": kh,
                    "linear_value_head_dim": vh,
                    "linear_conv_kernel_dim": conv,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.5,
                    },
                },
            }),
        };
        let spec = Spec::from_config(Json::new(config)).unwrap();

        let d = Device::Cpu;
        let rand = |r: usize, c: usize| Tensor::randn(0f32, 0.05f32, (r, c), &d).unwrap();
        let vec1 = |n: usize| Tensor::randn(0f32, 0.05f32, n, &d).unwrap();
        let root = match next {
            true => "model",
            false => "model.language_model",
        };

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(format!("{root}.embed_tokens.weight"), rand(vocab, e));
        t.insert(format!("{root}.norm.weight"), vec1(e));
        t.insert("lm_head.weight".into(), rand(vocab, e));
        for (i, kind) in kinds.iter().enumerate() {
            let p = format!("{root}.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), vec1(e));
            t.insert(format!("{p}.post_attention_layernorm.weight"), vec1(e));
            match next {
                false => {
                    t.insert(format!("{p}.mlp.gate_proj.weight"), rand(inter, e));
                    t.insert(format!("{p}.mlp.up_proj.weight"), rand(inter, e));
                    t.insert(format!("{p}.mlp.down_proj.weight"), rand(e, inter));
                }
                true => {
                    t.insert(format!("{p}.mlp.gate.weight"), rand(n_experts, e));
                    for x in 0..n_experts {
                        let m = format!("{p}.mlp.experts.{x}");
                        t.insert(format!("{m}.gate_proj.weight"), rand(expert, e));
                        t.insert(format!("{m}.up_proj.weight"), rand(expert, e));
                        t.insert(format!("{m}.down_proj.weight"), rand(e, expert));
                    }
                    let m = format!("{p}.mlp.shared_expert");
                    t.insert(format!("{m}.gate_proj.weight"), rand(shared, e));
                    t.insert(format!("{m}.up_proj.weight"), rand(shared, e));
                    t.insert(format!("{m}.down_proj.weight"), rand(e, shared));
                    t.insert(format!("{p}.mlp.shared_expert_gate.weight"), rand(1, e));
                }
            }
            match *kind {
                "full_attention" => {
                    let q = format!("{p}.self_attn");
                    t.insert(format!("{q}.q_proj.weight"), rand(nh * hd * 2, e));
                    t.insert(format!("{q}.k_proj.weight"), rand(nkv * hd, e));
                    t.insert(format!("{q}.v_proj.weight"), rand(nkv * hd, e));
                    t.insert(format!("{q}.o_proj.weight"), rand(e, nh * hd));
                    t.insert(format!("{q}.q_norm.weight"), vec1(hd));
                    t.insert(format!("{q}.k_norm.weight"), vec1(hd));
                }
                _ => {
                    let q = format!("{p}.linear_attn");
                    match next {
                        true => {
                            t.insert(
                                format!("{q}.in_proj_qkvz.weight"),
                                rand(key_dim * 2 + value_dim * 2, e),
                            );
                            t.insert(format!("{q}.in_proj_ba.weight"), rand(nv * 2, e));
                        }
                        false => {
                            t.insert(format!("{q}.in_proj_qkv.weight"), rand(conv_dim, e));
                            t.insert(format!("{q}.in_proj_z.weight"), rand(value_dim, e));
                            t.insert(format!("{q}.in_proj_b.weight"), rand(nv, e));
                            t.insert(format!("{q}.in_proj_a.weight"), rand(nv, e));
                        }
                    }
                    t.insert(format!("{q}.conv1d.weight"), rand(conv_dim, conv));
                    t.insert(format!("{q}.dt_bias"), vec1(nv));
                    t.insert(format!("{q}.A_log"), vec1(nv));
                    t.insert(format!("{q}.norm.weight"), Tensor::ones(vh, DType::F32, &d).unwrap());
                    t.insert(format!("{q}.out_proj.weight"), rand(e, value_dim));
                }
            }
        }

        let path = std::env::temp_dir()
            .join(format!("gpu-q35-{}-{tag}.safetensors", std::process::id()));
        candle_core::safetensors::save(&t, &path).unwrap();
        (spec, path)
    }

    fn tiny(tag: &str) -> (Spec, std::path::PathBuf) {
        tiny_of(false, tag)
    }

    /// The same widths, run against the CPU engine on every device there is.
    fn both_backends_agree(spec: &Spec, path: &std::path::Path, what: &str) {
        let tokens = [3u32, 17, 8, 31, 4];
        let mut cache = kvad::model::KvCache::new(spec);
        let ckpt = kvad::weights::Checkpoint::open(std::slice::from_ref(&path.to_path_buf()))
            .unwrap();
        let src = kvad::qcache::Live::new(&ckpt, kvad::quant::Precision::F32);
        let ours = kvad::model::qwen3_5::Model::load(&src, spec.clone())
            .unwrap()
            .forward_batch(&tokens, &mut cache);

        let devices = match Device::new_metal(0) {
            Ok(gpu) => vec![("cpu device", Device::Cpu), ("metal", gpu)],
            Err(_) => vec![("cpu device", Device::Cpu)],
        };
        for (where_, dev) in devices {
            let mut gpu = GpuQwen35::load(
                std::slice::from_ref(&path.to_path_buf()),
                spec.clone(),
                DType::F32,
                None,
                dev,
                &Vault::off(),
            )
            .unwrap();
            let theirs = gpu.forward(&tokens).unwrap();
            assert_eq!(theirs.len(), ours.len(), "on {where_}");
            let worst = theirs.iter().zip(&ours).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-3, "{what} on {where_} differs from the CPU engine by {worst}");
        }
    }

    /// The gated delta net on this backend against the hand-written one.
    ///
    /// Both in f32, so what is left is the order the sums happen in. The two
    /// implementations share the layer layout and nothing else: the
    /// recurrence is a loop over heads there and six broadcasts here, and a
    /// disagreement means one of them is running a different model.
    #[test]
    fn the_hybrid_agrees_with_the_cpu_engine() {
        let (spec, path) = tiny("agree");
        both_backends_agree(&spec, &path, "the hybrid");
        std::fs::remove_file(&path).unwrap();
    }

    /// Qwen3-Next's spelling of the same mixer, with a mixture behind it.
    ///
    /// The fused input projections are four `narrow`s here and a loop of
    /// `copy_from_slice` on the CPU, and the router runs on the host in both —
    /// so a disagreement is in the deinterleaving, which is the one thing this
    /// family adds that is easy to get silently backwards.
    #[test]
    fn the_next_spelling_agrees_with_the_cpu_engine() {
        let (spec, path) = tiny_of(true, "next");
        both_backends_agree(&spec, &path, "Qwen3-Next");
        std::fs::remove_file(&path).unwrap();
    }

    /// The refusal, on the backend that is the first to need it.
    #[test]
    fn a_partial_rewind_is_refused_and_says_zero() {
        let (spec, path) = tiny("rewind");
        let mut gpu = GpuQwen35::load(
            std::slice::from_ref(&path),
            spec.clone(),
            DType::F32,
            None,
            Device::Cpu,
            &Vault::off(),
        )
        .unwrap();
        gpu.forward(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(gpu.cached(), 5);

        // Asking for everything held is not a rewind, so it is not refused.
        assert_eq!(gpu.truncate(5).unwrap(), 5);
        // Asking for less is, and there is no subtraction that does it.
        assert_eq!(gpu.truncate(3).unwrap(), 0);
        assert_eq!(gpu.cached(), 0);
        std::fs::remove_file(&path).unwrap();
    }

    /// The recurrent state is as wide as admission charges for it before
    /// the model loads, and as big: the bytes the linear layers hold after a
    /// step, against the spec's layout at that width with no positions.
    /// Pinned as well, so the rule and the session cannot go wrong together:
    /// f32 and q8 keep it in f32, bf16 in bf16, the attention cache aside.
    #[test]
    fn the_state_is_as_wide_as_admission_charges() {
        let mut devices = vec![Device::Cpu];
        if let Ok(m) = Device::new_metal(0) {
            devices.push(m);
        }
        for next in [false, true] {
            let (spec, path) = tiny_of(next, &format!("state-width-{next}"));
            for dev in &devices {
                for (dtype, quant, pinned) in
                    [(DType::F32, None, 4), (DType::BF16, None, 2), (DType::F32, Some(GgmlDType::Q8_0), 4)]
                {
                    // candle has no bf16 matmul on the CPU, so no step to
                    // take there.
                    if dtype == DType::BF16 && dev.is_cpu() {
                        continue;
                    }
                    let what = format!("{} {dtype:?} {quant:?} on {dev:?}", spec.arch);
                    let mut gpu =
                        GpuQwen35::load(std::slice::from_ref(&path), spec.clone(), dtype, quant, dev.clone(), &Vault::off())
                            .unwrap();
                    gpu.forward(&[1, 2, 3]).unwrap();
                    let charged = crate::model::state_number_bytes(dtype, quant);
                    assert_eq!(gpu.state_number_bytes(), charged, "{what}: kept, against charged");
                    assert_eq!(charged, pinned, "{what}");
                    let held: usize = gpu
                        .state
                        .iter()
                        .flatten()
                        .map(|st| [&st.conv, &st.s].iter().map(|t| t.elem_count() * t.dtype().size_in_bytes()).sum::<usize>())
                        .sum();
                    assert!(held > 0, "{what}: no state");
                    assert_eq!(held, spec.cache.bytes_as(spec.n_layer, 0, 0, charged), "{what}: held, against charged");
                }
            }
            std::fs::remove_file(&path).unwrap();
        }
    }

    /// Qwen3.5's full-attention cache is f16 at q8 on Metal, as the Llama
    /// family's is: see `model::tests::check_kv_width`.
    #[test]
    fn the_cache_is_as_wide_as_admission_charges() {
        let (spec, path) = tiny_of(false, "kv-width");
        crate::model::tests::check_kv_width(spec.arch, 2, &|dtype, quant, dev| {
            Box::new(GpuQwen35::load(std::slice::from_ref(&path), spec.clone(), dtype, quant, dev, &Vault::off()).unwrap())
        });
        std::fs::remove_file(&path).unwrap();
    }

}
