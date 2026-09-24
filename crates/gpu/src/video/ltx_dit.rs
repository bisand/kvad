//! LTX-2.5's DiT: 48 blocks that denoise a video and its sound together.
//!
//! ```text
//! video latent [128, F, h, w] ─ tokens (f, h, w) ─ patchify ─ 4096 wide ─┐
//! audio latent [8, T, 16]     ─ tokens (t)       ─ patchify ─ 2048 wide ─┤
//!                                                                         │
//!   48 × { video: self-attention, attention to its text context         │
//!          audio: the same, at half the width                            │
//!          audio → video and video → audio attention                     │
//!          video and audio feed-forwards }                                │
//!                                                                         │
//! video ─ layer norm, scale and shift ─ 128 ─ velocity ◀──────────────────┘
//! audio ─ the same, 2048 → 128
//! ```
//!
//! Every norm is modulated by the noise level σ. Eight "adaLN single" modules
//! turn σ into rows of shifts, scales and gates once a step, and each block
//! adds its own learned table to them. `docs/video-plan.md` has the table of
//! rows, the positions, and where the reference was read for each.

use super::ltx_nn::{gelu, rms, GatedAttention, Rope};
use super::ltx_text::Contexts;
use super::metadata;
use crate::common::{Loader, Reader};
use crate::image::nn::{layer_norm_plain, Ctx, Linear};
use crate::image::{finish, open};
use crate::qcache::Vault;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use kvad::serde_json::{json, Value};
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A clip's size in pixels and frames, and what that comes to in latents.
///
/// The video VAE packs 32×32 pixels and 8 frames into one latent cell, except
/// that the first latent frame holds the first pixel frame alone: so a clip is
/// 8k + 1 frames. The audio VAE makes 25 latents a second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: f64,
}

impl Shape {
    pub fn new(width: usize, height: usize, frames: usize, fps: f64) -> Res<Self> {
        if width % 32 != 0 || height % 32 != 0 || width == 0 || height == 0 {
            return Err(format!("{width}×{height}: LTX wants both sides a multiple of 32").into());
        }
        if frames % 8 != 1 {
            return Err(format!("{frames} frames: LTX wants 8k + 1 (for example {} or {})", frames / 8 * 8 + 1, frames.div_ceil(8) * 8 + 1).into());
        }
        if !(fps > 0.0) {
            return Err(format!("{fps} frames a second").into());
        }
        Ok(Shape { width, height, frames, fps })
    }

    pub fn latent_frames(&self) -> usize {
        (self.frames - 1) / 8 + 1
    }

    /// Latent rows and columns.
    pub fn grid(&self) -> (usize, usize) {
        (self.height / 32, self.width / 32)
    }

    /// Video tokens a latent frame holds.
    pub fn frame_tokens(&self) -> usize {
        let (h, w) = self.grid();
        h * w
    }

    pub fn video_tokens(&self) -> usize {
        self.latent_frames() * self.frame_tokens()
    }

    /// Audio latents: the clip's length at 25 a second, rounded half to even
    /// as Python's `round` does. 121 frames at 24 fps is 126.
    pub fn audio_latents(&self) -> usize {
        (self.frames as f64 / self.fps * 25.0).round_ties_even() as usize
    }

    /// Where each video token is, on (time, row, column): the midpoint, in
    /// seconds and pixels, of what its latent cell covers.
    ///
    /// Latent frame f covers pixel frames 8f − 7 to 8f + 1, clamped at 0,
    /// because the first covers one frame and the rest eight; rows and
    /// columns cover 32 pixels each. Computed in f32 in the reference's
    /// order: the bounds divided by the frame rate, then averaged.
    pub fn video_positions(&self) -> [Vec<f32>; 3] {
        let (rows, cols) = self.grid();
        let fps = self.fps as f32;
        let n = self.video_tokens();
        let (mut t, mut h, mut w) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for f in 0..self.latent_frames() {
            let bound = |p: usize| (p as f32 + 1.0 - 8.0).max(0.0) / fps;
            let mid = (bound(8 * f) + bound(8 * f + 8)) / 2.0;
            for r in 0..rows {
                for c in 0..cols {
                    t.push(mid);
                    h.push((32 * r + 32 * r + 32) as f32 / 2.0);
                    w.push((32 * c + 32 * c + 32) as f32 / 2.0);
                }
            }
        }
        [t, h, w]
    }

    /// Where each audio latent is in time: the midpoint of what it covers,
    /// in seconds. Latent i covers mel frames 4i − 3 to 4i + 1, clamped at 0,
    /// at 100 mel frames a second.
    pub fn audio_positions(&self) -> Vec<f32> {
        let bound = |i: usize| (i as f32 * 4.0 + 1.0 - 4.0).max(0.0) * 160.0 / 16000.0;
        (0..self.audio_latents()).map(|i| (bound(i) + bound(i + 1)) / 2.0).collect()
    }
}

/// A shape's rotary tables, built once per generation: they depend on
/// nothing but the shape and the frame rate.
pub struct Grid {
    shape: Shape,
    /// Video self-attention: time, rows and columns across 4096.
    video: Rope,
    /// Audio self-attention, and audio's side of both audio–video
    /// attentions: time across 2048.
    audio: Rope,
    /// Video's side of the audio–video attentions: its time alone, across
    /// the audio width those attentions work at.
    video_time: Rope,
}

impl Grid {
    pub fn shape(&self) -> Shape {
        self.shape
    }
}

/// What the DiT's config says, and the parts of it this code relies on.
struct Config {
    layers: usize,
    heads: usize,
    head_dim: usize,
    audio_heads: usize,
    audio_head_dim: usize,
    channels: usize,
    audio_channels: usize,
    theta: f64,
    max_pos: Vec<f32>,
    audio_max_pos: f32,
    /// The gate modules' σ multiplier over the others': 1 for LTX-2.5.
    gate_factor: f32,
    ff_bias: bool,
    audio_ff_bias: bool,
    keyframes: bool,
}

impl Config {
    fn read(t: &Value) -> Res<Self> {
        let num = |k: &str| t[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("DiT config: no `{k}`"));
        let float = |k: &str| t[k].as_f64().ok_or_else(|| format!("DiT config: no `{k}`"));
        // What the code below implements; anything else is a different model.
        let want = |k: &str, v: Value| match t.get(k).unwrap_or(&Value::Null) == &v {
            true => Ok(()),
            false => Err(format!("DiT config: `{k}` is {}, and only {v} is implemented", t[k])),
        };
        want("cross_attention_adaln", json!(true))?;
        want("apply_gated_attention", json!(true))?;
        want("rope_type", json!("split"))?;
        want("frequencies_precision", json!("float64"))?;
        want("use_middle_indices_grid", json!(true))?;
        want("caption_proj_before_connector", json!(true))?;
        want("activation_fn", json!("gelu-approximate"))?;
        if t.get("use_prompt_adaln_single") == Some(&json!(false)) {
            return Err("DiT config: a model without prompt adaLN is not implemented".into());
        }
        let max_pos: Vec<f32> = t["positional_embedding_max_pos"].as_array().ok_or("DiT config: no max_pos")?.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
        if max_pos.len() != 3 {
            return Err(format!("DiT config: {} position axes, not 3", max_pos.len()).into());
        }
        let audio_max_pos = t["audio_positional_embedding_max_pos"][0].as_f64().ok_or("DiT config: no audio max_pos")? as f32;
        let flag = |k: &str, default: bool| t[k].as_bool().unwrap_or(default);
        Ok(Config {
            layers: num("num_layers")?,
            heads: num("num_attention_heads")?,
            head_dim: num("attention_head_dim")?,
            audio_heads: num("audio_num_attention_heads")?,
            audio_head_dim: num("audio_attention_head_dim")?,
            channels: num("in_channels")?,
            audio_channels: t["audio_in_channels"].as_u64().unwrap_or(128) as usize,
            theta: float("positional_embedding_theta")?,
            max_pos,
            audio_max_pos,
            gate_factor: (float("av_ca_timestep_scale_multiplier")? / float("timestep_scale_multiplier")?) as f32,
            ff_bias: flag("ff_bias", true),
            audio_ff_bias: flag("audio_ff_bias", true),
            keyframes: flag("use_keyframes_abs_pos_embedding", false),
        })
    }
}

/// σ as the 256 cosines and sines an adaLN module reads, cosines first.
///
/// In f32, in the reference's order of operations. The angles reach a
/// thousand radians, where an f32 is good to a ten-thousandth of one.
fn sinusoid(t: f32, device: &Device) -> Res<Tensor> {
    let half = 128;
    let angle = |i: usize| t * (-(10000f64.ln() as f32) * i as f32 / half as f32).exp();
    let cos = (0..half).map(|i| angle(i).cos());
    let sin = (0..half).map(|i| angle(i).sin());
    Ok(Tensor::from_vec(cos.chain(sin).collect::<Vec<f32>>(), (1, 2 * half), device)?)
}

/// An "adaLN single" module: σ to `rows` modulation rows.
///
/// `σ·1000 ─ sinusoid ─ linear ─ silu ─ linear` is the embedded timestep,
/// which the output head uses as it is; `silu ─ linear` of that is the rows.
struct AdaLn {
    l1: Linear,
    l2: Linear,
    out: Linear,
    rows: usize,
    width: usize,
}

impl AdaLn {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, width: usize, rows: usize) -> Res<Self> {
        let e = r.pp("emb.timestep_embedder");
        Ok(AdaLn {
            l1: Linear::load(cx, &e, "linear_1", 256, width, true)?,
            l2: Linear::load(cx, &e, "linear_2", width, width, true)?,
            out: Linear::load(cx, r, "linear", width, rows * width, true)?,
            rows,
            width,
        })
    }

    /// `t` is σ·1000. The rows `[rows, width]`, and the embedded timestep
    /// `[1, width]`.
    fn forward(&self, t: f32, device: &Device, dtype: DType) -> Res<(Tensor, Tensor)> {
        let x = sinusoid(t, device)?.to_dtype(dtype)?;
        let e = self.l2.forward(&self.l1.forward(&x)?.to_dtype(dtype)?.silu()?)?.to_dtype(dtype)?;
        let m = self.out.forward(&e.silu()?)?.to_dtype(dtype)?.reshape((self.rows, self.width))?;
        Ok((m, e))
    }
}

/// One step's modulation rows, before each block adds its own tables.
struct Mods {
    video: Tensor,
    audio: Tensor,
    video_prompt: Tensor,
    audio_prompt: Tensor,
    video_av: Tensor,
    audio_av: Tensor,
    /// The audio → video gate, driven by the *audio* σ.
    a2v_gate: Tensor,
    /// The video → audio gate, driven by the *video* σ.
    v2a_gate: Tensor,
    video_embedded: Tensor,
    audio_embedded: Tensor,
}

fn row(t: &Tensor, i: usize) -> candle_core::Result<Tensor> {
    t.narrow(0, i, 1)
}

/// `rms(x)·(1 + scale) + shift`: how every norm in a block is modulated.
fn ada(x: &Tensor, scale: &Tensor, shift: &Tensor) -> candle_core::Result<Tensor> {
    rms(x, 1e-6)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

/// A feed-forward: up, tanh-GELU, down.
struct Ff {
    up: Linear,
    down: Linear,
}

impl Ff {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, width: usize, bias: bool) -> Res<Self> {
        Ok(Ff {
            up: Linear::load(cx, &r.pp("net.0"), "proj", width, 4 * width, bias)?,
            down: Linear::load(cx, &r.pp("net"), "2", 4 * width, width, bias)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let h = gelu(&self.up.forward(x)?.to_dtype(dtype)?)?;
        self.down.forward(&h)?.to_dtype(dtype)
    }
}

/// A block's tables for one stream: nine rows beside the step's (shift,
/// scale and gate for self-attention, for the feed-forward, and for the text
/// query), two for the text keys and values, and five for the audio–video
/// attentions, which are (scale, shift) rather than (shift, scale).
struct Tables {
    main: Tensor,
    prompt: Tensor,
    av: Tensor,
}

struct Stream {
    attn1: GatedAttention,
    attn2: GatedAttention,
    ff: Ff,
    tables: Tables,
}

impl Stream {
    fn load(cx: &Ctx<'_>, r: &Reader<'_>, prefix: &str, width: usize, heads: usize, head_dim: usize, ff_bias: bool) -> Res<Self> {
        let n = |s: &str| format!("{prefix}{s}");
        Ok(Stream {
            attn1: GatedAttention::load(cx, &r.pp(n("attn1")), (width, width), heads, head_dim, true)?,
            attn2: GatedAttention::load(cx, &r.pp(n("attn2")), (width, width), heads, head_dim, true)?,
            ff: Ff::load(cx, &r.pp(n("ff")), width, ff_bias)?,
            tables: Tables {
                main: cx.get(r, (9, width), &n("scale_shift_table"))?,
                prompt: cx.get(r, (2, width), &n("prompt_scale_shift_table"))?,
                av: cx.get(r, (5, width), &format!("scale_shift_table_a2v_ca_{}", if prefix.is_empty() { "video" } else { "audio" }))?,
            },
        })
    }

    /// Self-attention, then attention to the text: the first half of a
    /// block, for one stream. `m` is the nine rows with the table added, `p`
    /// the two for the text.
    fn attend(&self, x: &Tensor, m: &Tensor, p: &Tensor, context: &Tensor, rope: &Rope) -> candle_core::Result<Tensor> {
        let y = self.attn1.forward(&ada(x, &row(m, 1)?, &row(m, 0)?)?, None, Some(rope), Some(rope))?;
        let x = (x + y.broadcast_mul(&row(m, 2)?)?)?;
        // The text is modulated but not normalised, and σ moves the
        // modulation, so its keys and values change every step.
        let c = context.broadcast_mul(&(row(p, 1)? + 1.0)?)?.broadcast_add(&row(p, 0)?)?;
        let y = self.attn2.forward(&ada(&x, &row(m, 7)?, &row(m, 6)?)?, Some(&c), None, None)?;
        &x + y.broadcast_mul(&row(m, 8)?)?
    }

    fn feed(&self, x: &Tensor, m: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.ff.forward(&ada(x, &row(m, 4)?, &row(m, 3)?)?)?;
        x + y.broadcast_mul(&row(m, 5)?)?
    }
}

struct Block {
    video: Stream,
    audio: Stream,
    /// Queries from video, keys and values from audio, at audio's head layout.
    a2v: GatedAttention,
    v2a: GatedAttention,
}

impl Block {
    fn forward(&self, vx: &Tensor, ax: &Tensor, m: &Mods, ctx: &Contexts, g: &Grid) -> candle_core::Result<(Tensor, Tensor)> {
        let (v, a) = (&self.video.tables, &self.audio.tables);
        let vm = (&v.main + &m.video)?;
        let am = (&a.main + &m.audio)?;
        let vx = self.video.attend(vx, &vm, &(&v.prompt + &m.video_prompt)?, &ctx.video, &g.video)?;
        let ax = self.audio.attend(ax, &am, &(&a.prompt + &m.audio_prompt)?, &ctx.audio, &g.audio)?;

        // Each direction reads both streams as they were before either
        // update, so the order of the two does not matter.
        let vav = (v.av.narrow(0, 0, 4)? + &m.video_av)?;
        let aav = (a.av.narrow(0, 0, 4)? + &m.audio_av)?;
        let a2v = self.a2v.forward(
            &ada(&vx, &row(&vav, 0)?, &row(&vav, 1)?)?,
            Some(&ada(&ax, &row(&aav, 0)?, &row(&aav, 1)?)?),
            Some(&g.video_time),
            Some(&g.audio),
        )?;
        let v2a = self.v2a.forward(
            &ada(&ax, &row(&aav, 2)?, &row(&aav, 3)?)?,
            Some(&ada(&vx, &row(&vav, 2)?, &row(&vav, 3)?)?),
            Some(&g.audio),
            Some(&g.video_time),
        )?;
        let vx = (&vx + a2v.broadcast_mul(&(row(&v.av, 4)? + &m.a2v_gate)?)?)?;
        let ax = (&ax + v2a.broadcast_mul(&(row(&a.av, 4)? + &m.v2a_gate)?)?)?;

        Ok((self.video.feed(&vx, &vm)?, self.audio.feed(&ax, &am)?))
    }
}

/// An output head: a plain layer norm, scaled and shifted by the embedded
/// timestep plus a table, projected down to the latent's channels.
struct Head {
    table: Tensor,
    proj: Linear,
}

impl Head {
    fn forward(&self, x: &Tensor, embedded: &Tensor) -> candle_core::Result<Tensor> {
        let t = self.table.broadcast_add(embedded)?;
        let x = layer_norm_plain(x, 1e-6)?.broadcast_mul(&(row(&t, 1)? + 1.0)?)?.broadcast_add(&row(&t, 0)?)?;
        self.proj.forward(&x)?.to_dtype(x.dtype())
    }
}

pub struct Dit {
    cfg: Config,
    patchify: Linear,
    audio_patchify: Linear,
    /// Added to the tokens of latent frame 0, which holds one pixel frame
    /// where the others hold eight.
    keyframe: Option<Tensor>,
    adaln: AdaLn,
    audio_adaln: AdaLn,
    prompt_adaln: AdaLn,
    audio_prompt_adaln: AdaLn,
    video_av: AdaLn,
    audio_av: AdaLn,
    a2v_gate: AdaLn,
    v2a_gate: AdaLn,
    blocks: Vec<Block>,
    head: Head,
    audio_head: Head,
    device: Device,
    dtype: DType,
    params: usize,
}

impl Dit {
    /// The DiT in the file at `path`, computing in `dtype` on `device`.
    ///
    /// `layers` loads only the first so many blocks, for a check against a
    /// reference that cannot afford all of them. `quant` quantises every
    /// matrix and, for the whole DiT, caches the result, so the next load
    /// maps it instead.
    pub fn load(path: &Path, device: &Device, dtype: DType, layers: Option<usize>, quant: Option<GgmlDType>, progress: &mut dyn FnMut(&str)) -> Res<Self> {
        let cfg = Config::read(&metadata(path, "config")?["transformer"])?;
        let n = layers.unwrap_or(cfg.layers).min(cfg.layers);
        let paths = [path.to_path_buf()];
        // Only the whole DiT is cached: a check that loads two blocks would
        // otherwise find the whole one's cache stale and replace it with its
        // own, and the next generation would quantise all 20 GB again.
        let mut vault = match n == cfg.layers {
            true => Vault::open_as(&format!("{}/transformer", super::LTX_REPO), &paths, json!({ "component": "transformer", "layers": n }), quant, progress),
            false => Vault::off(),
        };
        let dit = {
            let cx = Ctx { ld: Loader::new(quant, device.clone(), &vault).accelerated(), dtype };
            // The tables are f32 in the file; read in f32 when computing in
            // it, so that they stay exact.
            let r = open(&paths, if dtype == DType::F32 { DType::F32 } else { DType::BF16 })?;
            let m = r.pp("model.diffusion_model");
            // The connectors are the text path's; see `ltx_text`.
            m.skip_under("video_embeddings_connector");
            m.skip_under("audio_embeddings_connector");
            for i in n..cfg.layers {
                m.skip_under(&format!("transformer_blocks.{i}"));
            }
            let (vw, aw) = (cfg.heads * cfg.head_dim, cfg.audio_heads * cfg.audio_head_dim);
            let mut blocks = Vec::with_capacity(n);
            for i in 0..n {
                let b = m.pp(format!("transformer_blocks.{i}"));
                blocks.push(Block {
                    video: Stream::load(&cx, &b, "", vw, cfg.heads, cfg.head_dim, cfg.ff_bias)?,
                    audio: Stream::load(&cx, &b, "audio_", aw, cfg.audio_heads, cfg.audio_head_dim, cfg.audio_ff_bias)?,
                    a2v: GatedAttention::load(&cx, &b.pp("audio_to_video_attn"), (vw, aw), cfg.audio_heads, cfg.audio_head_dim, true)?,
                    v2a: GatedAttention::load(&cx, &b.pp("video_to_audio_attn"), (aw, vw), cfg.audio_heads, cfg.audio_head_dim, true)?,
                });
                if (i + 1) % 8 == 0 {
                    progress(&format!("DiT: {} of {n} blocks", i + 1));
                }
            }
            let dit = Dit {
                patchify: Linear::load(&cx, &m, "patchify_proj", cfg.channels, vw, true)?,
                audio_patchify: Linear::load(&cx, &m, "audio_patchify_proj", cfg.audio_channels, aw, true)?,
                keyframe: match cfg.keyframes {
                    true => Some(cx.get(&m, (1, vw), "keyframes_abs_pos_embedding")?),
                    false => None,
                },
                adaln: AdaLn::load(&cx, &m.pp("adaln_single"), vw, 9)?,
                audio_adaln: AdaLn::load(&cx, &m.pp("audio_adaln_single"), aw, 9)?,
                prompt_adaln: AdaLn::load(&cx, &m.pp("prompt_adaln_single"), vw, 2)?,
                audio_prompt_adaln: AdaLn::load(&cx, &m.pp("audio_prompt_adaln_single"), aw, 2)?,
                video_av: AdaLn::load(&cx, &m.pp("av_ca_video_scale_shift_adaln_single"), vw, 4)?,
                audio_av: AdaLn::load(&cx, &m.pp("av_ca_audio_scale_shift_adaln_single"), aw, 4)?,
                a2v_gate: AdaLn::load(&cx, &m.pp("av_ca_a2v_gate_adaln_single"), vw, 1)?,
                v2a_gate: AdaLn::load(&cx, &m.pp("av_ca_v2a_gate_adaln_single"), aw, 1)?,
                blocks,
                head: Head { table: cx.get(&m, (2, vw), "scale_shift_table")?, proj: Linear::load(&cx, &m, "proj_out", vw, cfg.channels, true)? },
                audio_head: Head {
                    table: cx.get(&m, (2, aw), "audio_scale_shift_table")?,
                    proj: Linear::load(&cx, &m, "audio_proj_out", aw, cfg.audio_channels, true)?,
                },
                params: finish("LTX DiT", &paths, &r)?,
                cfg,
                device: device.clone(),
                dtype,
            };
            dit
        };
        vault.finish(progress);
        Ok(dit)
    }

    pub fn params(&self) -> usize {
        self.params
    }

    pub fn layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The rotary tables for a shape.
    ///
    /// Their angles reach about 15 700 radians, so they are built on the
    /// host in f32 (`Rope::split`) and uploaded, never computed on the GPU.
    pub fn grid(&self, shape: Shape) -> Res<Grid> {
        let c = &self.cfg;
        let (vw, aw) = (c.heads * c.head_dim, c.audio_heads * c.audio_head_dim);
        let (dev, dt) = (&self.device, self.dtype);
        let video = shape.video_positions();
        let audio = vec![shape.audio_positions()];
        // The audio–video attentions measure time on one scale for both.
        let cross = c.max_pos[0].max(c.audio_max_pos);
        Ok(Grid {
            shape,
            video: Rope::split(&video, &c.max_pos, vw, c.heads, c.theta, dev, dt)?,
            audio: Rope::split(&audio, &[c.audio_max_pos], aw, c.audio_heads, c.theta, dev, dt)?,
            video_time: Rope::split(&video[..1], &[cross], aw, c.heads, c.theta, dev, dt)?,
        })
    }

    fn mods(&self, video_sigma: f32, audio_sigma: f32) -> Res<Mods> {
        let (dev, dt) = (&self.device, self.dtype);
        let (tv, ta) = (video_sigma * 1000.0, audio_sigma * 1000.0);
        let (video, video_embedded) = self.adaln.forward(tv, dev, dt)?;
        let (audio, audio_embedded) = self.audio_adaln.forward(ta, dev, dt)?;
        Ok(Mods {
            video,
            audio,
            video_prompt: self.prompt_adaln.forward(tv, dev, dt)?.0,
            audio_prompt: self.audio_prompt_adaln.forward(ta, dev, dt)?.0,
            video_av: self.video_av.forward(tv, dev, dt)?.0,
            audio_av: self.audio_av.forward(ta, dev, dt)?.0,
            a2v_gate: self.a2v_gate.forward(ta * self.cfg.gate_factor, dev, dt)?.0,
            v2a_gate: self.v2a_gate.forward(tv * self.cfg.gate_factor, dev, dt)?.0,
            video_embedded,
            audio_embedded,
        })
    }

    /// The velocities `ε − x₀` for video tokens `[F·h·w, 128]` and audio
    /// tokens `[T, 128]` at noise levels σ, in the DiT's dtype.
    pub fn forward(&self, video: &Tensor, audio: &Tensor, sigma: (f32, f32), ctx: &Contexts, grid: &Grid) -> Res<(Tensor, Tensor)> {
        self.forward_watched(video, audio, sigma, ctx, grid, &mut |_, _, _| Ok(()))
    }

    /// [`Dit::forward`], showing `watch` both streams after every block.
    pub fn forward_watched(
        &self,
        video: &Tensor,
        audio: &Tensor,
        sigma: (f32, f32),
        ctx: &Contexts,
        grid: &Grid,
        watch: &mut dyn FnMut(usize, &Tensor, &Tensor) -> Res<()>,
    ) -> Res<(Tensor, Tensor)> {
        let dt = self.dtype;
        let m = self.mods(sigma.0, sigma.1)?;
        let mut vx = self.patchify.forward(&video.to_dtype(dt)?)?.to_dtype(dt)?;
        if let Some(k) = &self.keyframe {
            let n = grid.shape.frame_tokens();
            let first = vx.narrow(0, 0, n)?.broadcast_add(k)?;
            vx = Tensor::cat(&[&first, &vx.narrow(0, n, vx.dim(0)? - n)?], 0)?;
        }
        let mut ax = self.audio_patchify.forward(&audio.to_dtype(dt)?)?.to_dtype(dt)?;
        let ctx = Contexts { video: ctx.video.to_dtype(dt)?, audio: ctx.audio.to_dtype(dt)? };
        for (i, b) in self.blocks.iter().enumerate() {
            (vx, ax) = b.forward(&vx, &ax, &m, &ctx, grid)?;
            watch(i, &vx, &ax)?;
        }
        Ok((self.head.forward(&vx, &m.video_embedded)?, self.audio_head.forward(&ax, &m.audio_embedded)?))
    }
}

/// A video latent `[C, F, h, w]` as tokens `[F·h·w, C]`, frame by frame and
/// row by row.
pub fn video_tokens(latent: &Tensor) -> candle_core::Result<Tensor> {
    let (c, f, h, w) = latent.dims4()?;
    latent.reshape((c, f * h * w))?.t()?.contiguous()
}

/// Tokens back to a video latent of `shape`.
pub fn video_latent(tokens: &Tensor, shape: Shape) -> candle_core::Result<Tensor> {
    let (h, w) = shape.grid();
    let c = tokens.dim(1)?;
    tokens.t()?.contiguous()?.reshape((c, shape.latent_frames(), h, w))
}

/// An audio latent `[C, T, bins]` as tokens `[T, C·bins]`.
pub fn audio_tokens(latent: &Tensor) -> candle_core::Result<Tensor> {
    let (c, t, f) = latent.dims3()?;
    latent.permute((1, 0, 2))?.contiguous()?.reshape((t, c * f))
}

/// Tokens back to an audio latent of `channels` channels.
pub fn audio_latent(tokens: &Tensor, channels: usize) -> candle_core::Result<Tensor> {
    let (t, cf) = tokens.dims2()?;
    tokens.reshape((t, channels, cf / channels))?.permute((1, 0, 2))?.contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_follow_the_vae() {
        let s = Shape::new(768, 512, 121, 24.0).unwrap();
        assert_eq!((s.latent_frames(), s.grid(), s.video_tokens(), s.audio_latents()), (16, (16, 24), 6144, 126));
        let s = Shape::new(512, 320, 25, 24.0).unwrap();
        assert_eq!((s.latent_frames(), s.video_tokens(), s.audio_latents()), (4, 640, 26));
        assert!(Shape::new(500, 320, 25, 24.0).is_err());
        assert!(Shape::new(512, 320, 24, 24.0).is_err());
    }

    #[test]
    fn positions_are_midpoints_in_seconds_and_pixels() {
        let s = Shape::new(64, 32, 17, 24.0).unwrap();
        let [t, h, w] = s.video_positions();
        // Frame 0 covers pixel frame 0 alone; frame 1 covers 1 to 9.
        assert_eq!(t[0], 0.5 / 24.0);
        assert!((t[2] - 5.0 / 24.0).abs() < 1e-7);
        assert_eq!((h[0], w[0], w[1]), (16.0, 16.0, 48.0));
        let a = s.audio_positions();
        assert_eq!(a.len(), 18);
        assert!((a[0] - 0.005).abs() < 1e-7 && (a[1] - 0.03).abs() < 1e-7);
    }

    #[test]
    fn tokens_round_trip() {
        let dev = Device::Cpu;
        let s = Shape::new(64, 32, 9, 24.0).unwrap();
        let v = Tensor::arange(0f32, (128 * 2 * 1 * 2) as f32, &dev).unwrap().reshape((128, 2, 1, 2)).unwrap();
        let t = video_tokens(&v).unwrap();
        // Token (f, h, w) = (1, 0, 0) is the third; channel 5 of it is
        // latent[5, 1, 0, 0].
        assert_eq!(t.get(2).unwrap().get(5).unwrap().to_scalar::<f32>().unwrap(), (5 * 4 + 2) as f32);
        assert_eq!(video_latent(&t, s).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), v.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let a = Tensor::arange(0f32, (8 * 3 * 16) as f32, &dev).unwrap().reshape((8, 3, 16)).unwrap();
        let t = audio_tokens(&a).unwrap();
        // Token 1, column c·16 + f, is latent[c, 1, f].
        assert_eq!(t.get(1).unwrap().get(2 * 16 + 3).unwrap().to_scalar::<f32>().unwrap(), (2 * 48 + 16 + 3) as f32);
        assert_eq!(audio_latent(&t, 8).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), a.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
}
