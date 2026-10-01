//! LoRAs, applied at run time.
//!
//! A LoRA adapts a set of linear layers `W` with a pair of thin matrices
//! each, `A` (r × in) and `B` (out × r), and a scale, so that the layer
//! computes `W·x + scale·B·(A·x)`. Nothing here merges that into `W`: the
//! product is a side path added to each adapted layer's answer, as PEFT
//! adds it and diffusers leaves it unless told to fuse. So a quantised base,
//! Kvad's q8 or a community GGUF, takes a LoRA as it is, a request chooses
//! its LoRAs and their strengths without a reload, and nothing is
//! re-quantised or cached per LoRA. The price is two thin multiplies per
//! adapted layer per step. `docs/lora-plan.md` has why, and what it costs.
//!
//! Two halves:
//!
//! - [`File`] reads a LoRA's pairs from its header, in the spellings people
//!   publish: PEFT's `lora_A`/`lora_B`, kohya's `lora_down`/`lora_up` with
//!   an `alpha` beside each pair, and diffusers' two older ones.
//!   Anything else in the file is refused by name, LyCORIS' kinds and DoRA
//!   among it.
//! - [`Adapters`], a registry a [`crate::common::Reader`] can carry, for one
//!   [`Part`] of a model: every [`super::nn::Linear`] and
//!   [`super::nn::Conv2d`] loaded through that reader registers its name and
//!   shape and keeps a [`Slot`]. A LoRA's names say which part by their
//!   prefix (`lora_te2_`, `unet.`, `transformer.`), and in kohya's names by
//!   underscores for dots. [`Adapters::set`] fills the slots a request's
//!   LoRAs name, and refuses a LoRA any of whose pairs names no layer: a
//!   LoRA that silently did not apply would look like one that does nothing.

use super::Uncached;
use kvad::lora::SPELLINGS;
use crate::common::Proj;
use candle_core::{DType, Device, Tensor};
use kvad::image::ImageRequest;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One LoRA file: its pairs, by the name of the layer each adapts as the
/// file spells it, and the file to read them from.
pub struct File {
    path: PathBuf,
    /// What its messages call it: the name it was asked for by, or its path.
    name: String,
    file: Arc<Uncached>,
    pairs: Vec<Pair>,
}

/// One layer's `A` and `B`, by their names in the file, and its `alpha`.
struct Pair {
    module: String,
    down: String,
    up: String,
    alpha: Option<String>,
}


impl File {
    pub fn open(path: &Path) -> Res<Self> {
        let file = super::open_file(path)?;
        let names: Vec<String> = file.names().map(str::to_string).collect();
        let shown = path.display();
        // Kinds of adapter that are not a pair of factors, refused as what
        // they are rather than as names nobody reads.
        let other = |n: &str| -> Option<&str> {
            match n {
                _ if n.contains("hada_") => Some("a LoHa"),
                _ if n.contains("lokr_") => Some("a LoKr"),
                _ if n.ends_with(".dora_scale") => Some("a DoRA"),
                _ if n.ends_with(".diff") || n.ends_with(".diff_b") => Some("a set of whole-weight differences"),
                _ => None,
            }
        };
        if let Some(what) = names.iter().find_map(|n| other(n)) {
            return Err(format!("{shown} is {what}, not a LoRA, and only LoRAs are implemented here").into());
        }
        let mut pairs: HashMap<String, Pair> = HashMap::new();
        let mut unknown = Vec::new();
        fn pair<'p>(pairs: &'p mut HashMap<String, Pair>, module: &str) -> &'p mut Pair {
            pairs.entry(module.to_string()).or_insert_with(|| Pair { module: module.to_string(), down: String::new(), up: String::new(), alpha: None })
        }
        for n in &names {
            let mut placed = false;
            for (a, b) in SPELLINGS {
                if let Some(m) = n.strip_suffix(a) {
                    pair(&mut pairs, m).down = n.clone();
                    placed = true;
                } else if let Some(m) = n.strip_suffix(b) {
                    pair(&mut pairs, m).up = n.clone();
                    placed = true;
                }
            }
            if let Some(m) = n.strip_suffix(".alpha") {
                pair(&mut pairs, m).alpha = Some(n.clone());
                placed = true;
            }
            if !placed {
                unknown.push(n.clone());
            }
        }
        if !unknown.is_empty() {
            unknown.sort();
            return Err(format!("{shown}: {} tensor(s) are no LoRA factor:\n  {}", unknown.len(), kvad::weights::collapsed(&unknown).join("\n  ")).into());
        }
        let mut pairs: Vec<Pair> = pairs.into_values().collect();
        pairs.sort_by(|a, b| a.module.cmp(&b.module));
        let half: Vec<&str> = pairs.iter().filter(|p| p.down.is_empty() || p.up.is_empty()).map(|p| p.module.as_str()).collect();
        if !half.is_empty() {
            return Err(format!("{shown}: {} layer(s) have one factor of the two, or an alpha alone: {}", half.len(), half.join(", ")).into());
        }
        if pairs.is_empty() {
            return Err(format!("{shown} holds no LoRA factors").into());
        }
        Ok(File { path: path.to_path_buf(), name: path.display().to_string(), file, pairs })
    }

    /// The same LoRA, called `name` in what it says: the name a request
    /// gave, rather than a path in the Hub's cache.
    pub fn named(self, name: &str) -> Self {
        File { name: name.to_string(), ..self }
    }

    /// Only the pairs whose layer `keep` keeps: for a check that builds
    /// some of a model's blocks.
    #[cfg(test)]
    pub(crate) fn only(mut self, keep: impl Fn(&str) -> bool) -> Self {
        self.pairs.retain(|p| keep(&p.module));
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The layers it adapts.
    pub fn layers(&self) -> usize {
        self.pairs.len()
    }

    /// Its factors' bytes as stored: what it adds on the device, in the
    /// dtype it is stored in.
    pub fn bytes(&self) -> u64 {
        self.pairs.iter().map(|p| (self.file.bytes_of(&p.down) + self.file.bytes_of(&p.up)) as u64).sum()
    }

    /// `A` and `B` as the file has them, `[r, in]` and `[out, r]` for a
    /// linear layer, `[r, in, k, k]` and `[out, r, 1, 1]` for a convolution;
    /// and `alpha / r`, or one where the file gives no `alpha`.
    fn factors(&self, p: &Pair) -> Res<(Tensor, Tensor, f64)> {
        let (down, up) = (self.file.load(&p.down)?, self.file.load(&p.up)?);
        if !matches!((down.rank(), up.rank()), (2, 2) | (4, 4)) {
            return Err(format!("{}: `{}` is {:?} and `{}` {:?}, not a pair of factors", self.name, p.down, down.dims(), p.up, up.dims()).into());
        }
        let scale = match &p.alpha {
            Some(a) => self.file.load(a)?.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?[0] / down.dim(0)? as f64,
            None => 1.0,
        };
        Ok((down, up, scale))
    }
}

/// One LoRA's side path on one layer: `(x·A)·B`, the scale folded into `B`
/// once, in f32, as it is set; each factor a dense matrix, so a half
/// precision one runs on the M5's matrix units. On a convolution, `A` is a
/// convolution of the layer's own kernel, stride and padding into `r`
/// channels, and `B` a 1×1 one out of them.
struct Low {
    down: Down,
    b: Proj,
    dtype: DType,
}

enum Down {
    Linear(Proj),
    Conv { w: Tensor, stride: usize, pad: usize },
}

/// The layers a model's loaders registered, and the LoRAs set on them.
///
/// Shared by every layer's [`Slot`], and set between requests: a layer reads
/// its own entry on each forward pass, which is one uncontended lock.
#[derive(Clone)]
pub(crate) struct Adapters(Arc<RwLock<Registry>>);

struct Registry {
    /// What a file's names may start with, and the part of the model each
    /// such name is in: `lora_te2_` for SDXL's second text encoder, `""` for
    /// a model of one part whose LoRAs name its layers as they are.
    prefixes: Vec<(&'static str, &'static str)>,
    /// Each layer by `part/name`, and by any other name it has there.
    index: HashMap<String, usize>,
    layers: Vec<Layer>,
    /// What each part has and does not compute, by prefix, `part/prefix`: a
    /// LoRA's pairs under one are known, and change nothing.
    inert: Vec<String>,
    /// Layers a checkpoint's other layout fuses, `part/name` there, each to
    /// the layers it is here and the rows of its answer each takes: FLUX's
    /// `qkv` and `linear1` in Black Forest Labs' layout. A LoRA's pair for
    /// one is each layer's, `A` whole, since they read the same input, and
    /// `B` those rows.
    fused: HashMap<String, Vec<(usize, Rows)>>,
}

/// The rows of a fused layer's answer one layer here takes, in order.
type Rows = Vec<Range<usize>>;

/// A pair placed on a layer: the layer, the rows of `B` it takes if the
/// pair is a fused layer's, the file, the pair, and its strength.
type Placed<'f> = (usize, Option<Rows>, &'f File, &'f Pair, f64);

/// Where a LoRA's pair lands.
enum Target {
    Layer(usize),
    /// A fused layer's, split: each layer and the rows of `B` it takes.
    Fused(Vec<(usize, Rows)>),
    /// A layer the model has and does not compute.
    Inert,
}

/// What a layer is, and so what a LoRA's factors for it must be.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    Linear { inp: usize, out: usize },
    Conv { inp: usize, out: usize, k: usize, stride: usize, pad: usize },
}

struct Layer {
    name: String,
    shape: Shape,
    set: Option<Arc<Vec<Low>>>,
}

/// The registry, as one part of a model sees it: what a reader carries, so
/// that each layer loaded through it registers under that part.
#[derive(Clone)]
pub(crate) struct Part {
    adapters: Adapters,
    part: &'static str,
    /// What the checkpoint names this part's layers under and a LoRA does
    /// not: LTX-2.5's `model.diffusion_model.`, which its LoRAs call
    /// `diffusion_model.`.
    strip: &'static str,
}

/// A layer's place in its model's [`Adapters`].
#[derive(Clone)]
pub(crate) struct Slot {
    adapters: Adapters,
    i: usize,
}

impl Adapters {
    pub(crate) fn new(prefixes: &[(&'static str, &'static str)]) -> Self {
        Adapters(Arc::new(RwLock::new(Registry { prefixes: prefixes.to_vec(), index: HashMap::new(), layers: Vec::new(), inert: Vec::new(), fused: HashMap::new() })))
    }

    /// The registry for the layers of `part`.
    pub(crate) fn part(&self, part: &'static str) -> Part {
        Part { adapters: self.clone(), part, strip: "" }
    }

    /// [`Adapters::part`], for a checkpoint that names the part's layers
    /// under `strip`, which is left off the names they register under.
    pub(crate) fn part_under(&self, part: &'static str, strip: &'static str) -> Part {
        Part { adapters: self.clone(), part, strip }
    }

    fn register(&self, key: String, shape: Shape) -> Slot {
        let mut r = self.0.write().expect("the adapters' lock");
        let i = match r.index.get(&key) {
            Some(&i) => i,
            None => {
                let i = r.layers.len();
                r.layers.push(Layer { name: key.clone(), shape, set: None });
                r.index.insert(key, i);
                i
            }
        };
        Slot { adapters: self.clone(), i }
    }

    /// The names the layers of `part` registered under.
    pub(crate) fn names(&self, part: &str) -> Vec<String> {
        let r = self.0.read().expect("the adapters' lock");
        let pre = format!("{part}/");
        r.layers.iter().filter_map(|l| l.name.strip_prefix(&pre).map(str::to_string)).collect()
    }

    /// `from`, a layer of `part` fused in a checkpoint's other layout, as
    /// the layer registered as `to` here, whose answer is `rows` of `from`'s,
    /// in that order. Called once for each layer `from` is.
    pub(crate) fn fused(&self, part: &str, from: &str, to: &str, rows: Rows) {
        let mut r = self.0.write().expect("the adapters' lock");
        if let Some(&i) = r.index.get(&format!("{part}/{to}")) {
            r.fused.entry(format!("{part}/{from}")).or_default().push((i, rows));
        }
    }

    /// Another name for the layer of `part` registered as `name`: its name in
    /// a checkpoint's other layout, which LoRAs made for that layout use.
    pub(crate) fn alias(&self, part: &str, alias: &str, name: &str) {
        let mut r = self.0.write().expect("the adapters' lock");
        if let Some(&i) = r.index.get(&format!("{part}/{name}")) {
            r.index.entry(format!("{part}/{alias}")).or_insert(i);
        }
    }

    /// Set `loras`, each at its strength, in place of whatever was set: on
    /// `device`, their factors in `dtype`, which the side path computes in.
    /// Every pair of every file must name a layer, or nothing is set. The
    /// layers adapted.
    pub(crate) fn set(&self, loras: &[(&File, f64)], device: &Device, dtype: DType) -> Res<usize> {
        self.clear();
        let mut r = self.0.write().expect("the adapters' lock");
        // kohya writes a layer's name with underscores for its dots. Two
        // names the same that way would be ambiguous, and are found by
        // their dotted names only.
        let mut under: HashMap<String, Option<usize>> = HashMap::new();
        for (key, &i) in r.index.iter() {
            under.entry(key.replace('.', "_")).and_modify(|e| *e = if *e == Some(i) { Some(i) } else { None }).or_insert(Some(i));
        }
        let fused_flat: HashMap<String, &String> = r.fused.keys().map(|k| (k.replace('.', "_"), k)).collect();
        let find = |m: &str| -> Option<Target> {
            let m = old_names(m);
            r.prefixes.iter().find_map(|(p, part)| {
                let k = format!("{part}/{}", m.strip_prefix(p)?);
                let (flat, dot) = (k.replace('.', "_"), format!("{k}."));
                if let Some(i) = r.index.get(&k).copied().or_else(|| under.get(&flat).copied().flatten()) {
                    return Some(Target::Layer(i));
                }
                if let Some(t) = r.fused.get(&k).or_else(|| fused_flat.get(&flat).and_then(|k| r.fused.get(*k))) {
                    return Some(Target::Fused(t.clone()));
                }
                r.inert.iter().any(|i| dot.starts_with(i.as_str()) || format!("{flat}_").starts_with(&i.replace('.', "_"))).then_some(Target::Inert)
            })
        };
        let mut placed: Vec<Placed> = Vec::new();
        for (f, strength) in loras {
            let mut missing = Vec::new();
            for p in &f.pairs {
                match find(&p.module) {
                    Some(Target::Inert) => {}
                    Some(Target::Layer(i)) => placed.push((i, None, *f, p, *strength)),
                    Some(Target::Fused(to)) => placed.extend(to.into_iter().map(|(i, rows)| (i, Some(rows), *f, p, *strength))),
                    None => missing.push(p.module.clone()),
                }
            }
            if !missing.is_empty() {
                missing.sort();
                // kohya's underscores hide the numbers `collapsed` folds,
                // so a LoRA for another model can be hundreds of lines.
                let lines = kvad::weights::collapsed(&missing);
                let shown = lines.iter().take(6).cloned().collect::<Vec<_>>().join("\n  ");
                let more = match lines.len() > 6 {
                    true => format!("\n  and {} more", lines.len() - 6),
                    false => String::new(),
                };
                let whole = match missing.len() == f.pairs.len() {
                    true => ", so it is a LoRA for another model",
                    false => "",
                };
                return Err(format!("{}: {} of its {} layers are none of this model's{whole}:\n  {shown}{more}", f.name, missing.len(), f.pairs.len()).into());
            }
        }
        let mut sets: HashMap<usize, Vec<Low>> = HashMap::new();
        for (i, rows, f, p, strength) in placed {
            let (down, up, scale) = f.factors(p)?;
            // A fused layer's `B`, as this layer's rows of it.
            let up = match rows {
                Some(rows) => {
                    let pieces = rows.iter().map(|r| up.narrow(0, r.start, r.len())).collect::<candle_core::Result<Vec<_>>>()?;
                    Tensor::cat(&pieces, 0)?
                }
                None => up,
            };
            let l = &r.layers[i];
            let on = |t: Tensor| -> Res<Tensor> { Ok(t.to_dtype(dtype)?.contiguous()?.to_device(device)?) };
            let unfit = |what: String| -> Box<dyn std::error::Error> { format!("{}: `{}` is {what}, and the layer `{}` is {:?}", f.name, p.module, l.name, l.shape).into() };
            // `B` as `[r, out]`, the scale folded in; a 1×1 kernel is a matrix.
            let (out, r_) = (up.dim(0)?, up.dim(1)?);
            if up.dims()[2..].iter().any(|&d| d != 1) || down.dim(0)? != r_ {
                return Err(unfit(format!("{:?} over {:?}, not a pair of factors of one rank", down.dims(), up.dims())));
            }
            let b = (up.reshape((out, r_))?.t()?.to_dtype(DType::F32)? * (scale * strength))?;
            let low = match l.shape {
                Shape::Linear { inp, out: lout } => {
                    // A 1×1 convolution's factors, for a layer Kvad runs as
                    // the linear one it is (SD 1.5's projections).
                    if down.dims()[2..].iter().any(|&d| d != 1) || down.elem_count() != r_ * inp || out != lout {
                        return Err(unfit(format!("{:?} over {:?}", down.dims(), up.dims())));
                    }
                    Low { down: Down::Linear(Proj::Dense(on(down.reshape((r_, inp))?.t()?)?)), b: Proj::Dense(on(b)?), dtype }
                }
                Shape::Conv { inp, out: lout, k, stride, pad } => {
                    // LoCon's: `A` a convolution of the layer's own kernel.
                    let w = match down.rank() {
                        4 if down.dims()[1..] == [inp, k, k] => down,
                        2 if k == 1 && down.dim(1)? == inp => down.reshape((r_, inp, 1, 1))?,
                        _ => return Err(unfit(format!("{:?} over {:?}", down.dims(), up.dims()))),
                    };
                    if out != lout {
                        return Err(unfit(format!("{:?} over {:?}", w.dims(), up.dims())));
                    }
                    Low { down: Down::Conv { w: on(w)?, stride, pad }, b: Proj::Dense(on(b)?), dtype }
                }
            };
            sets.entry(i).or_default().push(low);
        }
        let n = sets.len();
        for (i, lows) in sets {
            r.layers[i].set = Some(Arc::new(lows));
        }
        Ok(n)
    }

    /// Put the factors `a` (`[in, r]`) and `b` (`[r, out]`) on the linear
    /// layer `name` of `part` as its one LoRA, as they are: the layer then
    /// answers `W·x + (x·a)·b`. Nothing is copied, scaled or cast, so that
    /// factors that are variables stay the tensors `backward` reports on:
    /// what training (#75) sets, where [`Adapters::set`] is for a file's.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn place(&self, part: &str, name: &str, a: &Tensor, b: &Tensor) -> Res<()> {
        let mut r = self.0.write().expect("the adapters' lock");
        let i = *r.index.get(&format!("{part}/{name}")).ok_or_else(|| format!("`{name}` is no layer of `{part}`"))?;
        let Shape::Linear { inp, out } = r.layers[i].shape else { return Err(format!("`{name}` is not a linear layer").into()) };
        let rank = a.dim(1)?;
        if a.dims() != [inp, rank] || b.dims() != [rank, out] {
            return Err(format!("`{name}` is {inp} to {out}, and the factors are {:?} and {:?}", a.dims(), b.dims()).into());
        }
        r.layers[i].set = Some(Arc::new(vec![Low { down: Down::Linear(Proj::Dense(a.clone())), b: Proj::Dense(b.clone()), dtype: a.dtype() }]));
        Ok(())
    }

    /// No LoRA on any layer.
    pub(crate) fn clear(&self) {
        for l in &mut self.0.write().expect("the adapters' lock").layers {
            l.set = None;
        }
    }
}

/// `draw`, with `req`'s LoRAs set on `adapters` for it and taken off after,
/// whatever happens; their factors in `dtype`, on `device`. Each is found on
/// this machine by its name, never fetched. What the request asked for and
/// the model cannot do (a LoRA not here, one for another model) is refused
/// as [`kvad::image::Refused`], the asker's to change.
///
/// Without LoRAs it only draws, and leaves the adapters as they stand.
pub(crate) fn painting<T>(adapters: &Adapters, req: &ImageRequest, device: &Device, dtype: DType, draw: impl FnOnce() -> Res<T>) -> Res<T> {
    if req.loras.is_empty() {
        return draw();
    }
    let refused = |e: Box<dyn std::error::Error>| -> Box<dyn std::error::Error> { Box::new(kvad::image::Refused(e.to_string())) };
    let files = req
        .loras
        .iter()
        .map(|l| -> Res<(File, f64)> {
            let here = kvad::lora::local(&l.name).ok_or_else(|| format!("the LoRA {} is not on this machine; `kvad pull {}` fetches it", l.name, l.name))?;
            Ok((File::open(&here.file)?.named(&l.name), l.scale))
        })
        .collect::<Res<Vec<_>>>()
        .map_err(refused)?;
    let set: Vec<(&File, f64)> = files.iter().map(|(f, s)| (f, *s)).collect();
    adapters.set(&set, device, dtype).map_err(refused)?;
    let drawn = draw();
    adapters.clear();
    drawn
}

impl Adapters {
    /// Every layer of the UNet `part` also by its names in Stability's
    /// layout, which kohya's SDXL LoRAs use (`lora_unet_input_blocks_4_1_…`).
    pub(crate) fn alias_ldm(&self, part: &str) {
        for name in self.names(part) {
            for alias in super::single::ldm_of(&name) {
                self.alias(part, &alias, &name);
            }
        }
    }
}

/// diffusers' older names for an attention's projections, as its
/// `LoRAAttnProcessor` saved them, `attn1.processor.to_q` and
/// `attn1.processor.to_out`, as the layers are named: `attn1.to_q`,
/// `attn1.to_out.0`.
fn old_names(m: &str) -> std::borrow::Cow<'_, str> {
    if !m.contains(".processor.") {
        return m.into();
    }
    let m = m.replace(".processor.", ".");
    match m.strip_suffix(".to_out") {
        Some(base) => format!("{base}.to_out.0").into(),
        None => m.into(),
    }
}

impl Part {
    /// What this part has and does not compute, everything under `prefix`:
    /// a LoRA's pairs there are the model's, and change nothing.
    pub(crate) fn inert(&self, prefix: &str) {
        let mut r = self.adapters.0.write().expect("the adapters' lock");
        let prefix = prefix.strip_prefix(self.strip).unwrap_or(prefix);
        let prefix = match prefix.ends_with('.') {
            true => prefix.to_string(),
            false => format!("{prefix}."),
        };
        r.inert.push(format!("{}/{prefix}", self.part));
    }

    /// A linear layer, by its full name in the checkpoint, taking `inp` and
    /// giving `out`.
    pub(crate) fn linear(&self, name: &str, inp: usize, out: usize) -> Slot {
        let name = name.strip_prefix(self.strip).unwrap_or(name);
        self.adapters.register(format!("{}/{name}", self.part), Shape::Linear { inp, out })
    }

    /// A convolution, `k × k` from `inp` channels to `out`.
    pub(crate) fn conv(&self, name: &str, (inp, out, k): (usize, usize, usize), stride: usize, pad: usize) -> Slot {
        let name = name.strip_prefix(self.strip).unwrap_or(name);
        self.adapters.register(format!("{}/{name}", self.part), Shape::Conv { inp, out, k, stride, pad })
    }
}

impl Slot {
    /// Whether a LoRA is set on this layer.
    pub(crate) fn active(&self) -> bool {
        self.adapters.0.read().expect("the adapters' lock").layers[self.i].set.is_some()
    }

    fn set(&self) -> Option<Arc<Vec<Low>>> {
        self.adapters.0.read().expect("the adapters' lock").layers[self.i].set.clone()
    }

    /// `y`, a linear layer's answer to `x`, with each LoRA's side path
    /// added: `y + (x·A)·B`, `x·A` in the factors' dtype, as PEFT computes
    /// it. Any axes before the last are the rows.
    ///
    /// On the M5's matrix units the second product is added into `y` in
    /// place, rounded once into `y`'s dtype (`mpp::dense_acc`), where PEFT
    /// rounds it, scales it and rounds the sum: one pass over the answer
    /// where there were four, which were most of a side path's time.
    pub(crate) fn add(&self, x: &Tensor, y: Tensor) -> candle_core::Result<Tensor> {
        let Some(set) = self.set() else { return Ok(y) };
        let (dims, inp) = (y.dims().to_vec(), x.dim(candle_core::D::Minus1)?);
        let x = x.reshape((x.elem_count() / inp, inp))?;
        let mut y = y.reshape((x.dim(0)?, dims[dims.len() - 1]))?;
        for l in set.iter() {
            let Down::Linear(a) = &l.down else { candle_core::bail!("a convolution's LoRA on a linear layer") };
            let h = a.forward(&x.to_dtype(l.dtype)?)?;
            y = accumulate(y, &h, &l.b)?;
        }
        y.reshape(dims)
    }

    /// `y`, a convolution's answer to `x` (`[B, C, H, W]`), with each LoRA's
    /// side path added: `A` convolved over `x` into `r` channels, then `B`
    /// across them at each pixel, as a row of the same product a linear
    /// layer's takes.
    pub(crate) fn add_conv(&self, x: &Tensor, y: Tensor) -> candle_core::Result<Tensor> {
        let Some(set) = self.set() else { return Ok(y) };
        let (b, out, h, w) = y.dims4()?;
        // Pixels as rows, channels last, for the product and the sum.
        let mut rows = y.permute((0, 2, 3, 1))?.reshape((b * h * w, out))?;
        for l in set.iter() {
            let Down::Conv { w: a, stride, pad } = &l.down else { candle_core::bail!("a linear layer's LoRA on a convolution") };
            let hid = x.to_dtype(l.dtype)?.conv2d(a, *pad, *stride, 1, 1)?;
            let r = hid.dim(1)?;
            let hid = hid.permute((0, 2, 3, 1))?.reshape((b * h * w, r))?;
            rows = accumulate(rows, &hid, &l.b)?;
        }
        rows.reshape((b, h, w, out))?.permute((0, 3, 1, 2))?.contiguous()
    }
}

/// `y + h·B`, `h` and `B` in the factors' dtype: into `y` in place where the
/// M5's dense kernel takes it, and by a product and a sum where not.
fn accumulate(y: Tensor, h: &Tensor, b: &Proj) -> candle_core::Result<Tensor> {
    #[cfg(target_os = "macos")]
    if let Proj::Dense(bw) = b {
        // Summed into in place, so it must be its own buffer, from its
        // start: `affine(1, 0)` writes one, where `contiguous` and `copy`
        // can hand back storage another tensor shares.
        let y = if y.is_contiguous() && y.layout().start_offset() == 0 { y } else { y.affine(1.0, 0.0)? };
        if crate::mpp::dense_acc(&y, h, bw)? {
            return Ok(y);
        }
        let dt = y.dtype();
        return y + b.forward(h)?.to_dtype(dt)?;
    }
    let dt = y.dtype();
    y + b.forward(h)?.to_dtype(dt)?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A LoRA file of `tensors` in the temporary directory, by `name`.
    fn write(name: &str, tensors: &[(&str, Tensor)]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("kvad-lora-{}-{name}.safetensors", std::process::id()));
        let map: HashMap<String, Tensor> = tensors.iter().map(|(n, t)| (n.to_string(), t.clone())).collect();
        candle_core::safetensors::save(&map, &path).unwrap();
        path
    }

    fn t(v: &[f32], shape: (usize, usize)) -> Tensor {
        Tensor::from_slice(v, shape, &Device::Cpu).unwrap()
    }

    /// #41's check: `y + (alpha / r)·strength·B·(A·x)` for known numbers,
    /// in each of the three spellings, with and without an `alpha`, and
    /// found through a prefix and through kohya's underscores.
    #[test]
    fn a_side_path_adds_its_scaled_product() {
        // A layer 3 → 2; a rank-1 pair, A = [1, 2, 3], B = [1, -1]ᵀ.
        let (a, b) = (t(&[1., 2., 3.], (1, 3)), t(&[1., -1.], (2, 1)));
        let x = t(&[1., 0., 2., 0., 1., 1.], (2, 3));
        let y = t(&[10., 20., 30., 40.], (2, 2));
        // A·x per row: 7, 5.
        // A file's name, its tensors, and the `alpha / r` it should give.
        type Case<'n> = (&'n str, Vec<(&'n str, Tensor)>, f64);
        let cases: [Case; 4] = [
            ("peft", vec![("transformer.blocks.0.proj.lora_A.weight", a.clone()), ("transformer.blocks.0.proj.lora_B.weight", b.clone())], 1.0),
            ("kohya", vec![("lora_unet_blocks_0_proj.lora_down.weight", a.clone()), ("lora_unet_blocks_0_proj.lora_up.weight", b.clone()), ("lora_unet_blocks_0_proj.alpha", Tensor::new(0.5f32, &Device::Cpu).unwrap())], 0.5),
            ("old", vec![("blocks.0.proj.lora.down.weight", a.clone()), ("blocks.0.proj.lora.up.weight", b.clone())], 1.0),
            ("comfy", vec![("diffusion_model.blocks.0.proj.lora_A.weight", a.clone()), ("diffusion_model.blocks.0.proj.lora_B.weight", b.clone())], 1.0),
        ];
        for (name, tensors, scale) in cases {
            let file = File::open(&write(name, &tensors)).unwrap();
            let adapters = Adapters::new(&[("transformer.", "m"), ("diffusion_model.", "m"), ("lora_unet_", "m"), ("", "m")]);
            let slot = adapters.part("m").linear("blocks.0.proj", 3, 2);
            adapters.part("m").linear("blocks.1.proj", 3, 2);
            assert_eq!(adapters.set(&[(&file, 2.0)], &Device::Cpu, DType::F32).unwrap(), 1, "{name}");
            let got = slot.add(&x, y.clone()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let s = (2.0 * scale) as f32;
            assert_eq!(got, [10. + 7. * s, 20. - 7. * s, 30. + 5. * s, 40. - 5. * s], "{name}");
            adapters.clear();
            assert!(!slot.active());
            assert_eq!(slot.add(&x, y.clone()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), [10., 20., 30., 40.]);
        }
    }

    /// LoCon's side path on a 3×3 convolution: `A` convolved over the
    /// input with the layer's own padding, `B` across its channels, scaled,
    /// and added; the same numbers as the product merged into the kernel,
    /// `W + s·B·A`, convolved at once.
    #[test]
    fn a_convolution_takes_its_lora_as_a_merged_kernel_would() {
        let dev = Device::Cpu;
        let (cin, cout, r, k) = (3, 4, 2, 3);
        let x = Tensor::randn(0f32, 1.0, (1, cin, 5, 6), &dev).unwrap();
        let w = Tensor::randn(0f32, 1.0, (cout, cin, k, k), &dev).unwrap();
        let down = Tensor::randn(0f32, 1.0, (r, cin, k, k), &dev).unwrap();
        let up = Tensor::randn(0f32, 1.0, (cout, r, 1, 1), &dev).unwrap();
        let alpha = Tensor::new(1f32, &dev).unwrap();
        let file = File::open(&write("locon", &[("lora_unet_c.lora_down.weight", down.clone()), ("lora_unet_c.lora_up.weight", up.clone()), ("lora_unet_c.alpha", alpha)])).unwrap();
        let adapters = Adapters::new(&[("lora_unet_", "unet")]);
        let slot = adapters.part("unet").conv("c", (cin, cout, k), 1, 1);
        adapters.set(&[(&file, 0.5)], &dev, DType::F32).unwrap();
        let y = x.conv2d(&w, 1, 1, 1, 1).unwrap();
        let got = slot.add_conv(&x, y).unwrap();
        // alpha / r = 1/2, times 0.5.
        let delta = up.reshape((cout, r)).unwrap().matmul(&down.reshape((r, cin * k * k)).unwrap()).unwrap().reshape((cout, cin, k, k)).unwrap();
        let merged = (&w + (delta * 0.25).unwrap()).unwrap();
        let want = x.conv2d(&merged, 1, 1, 1, 1).unwrap();
        let apart = (got - want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(apart < 1e-4, "{apart}");
    }

    #[test]
    fn a_lora_that_does_not_fit_is_refused() {
        let (a, b) = (t(&[1., 2., 3.], (1, 3)), t(&[1., -1.], (2, 1)));
        let adapters = Adapters::new(&[("transformer.", "m"), ("", "m")]);
        adapters.part("m").linear("blocks.0.proj", 3, 2);
        let set = |name: &str, tensors: &[(&str, Tensor)]| -> String {
            match File::open(&write(name, tensors)) {
                Err(e) => e.to_string(),
                Ok(f) => adapters.set(&[(&f, 1.0)], &Device::Cpu, DType::F32).err().map(|e| e.to_string()).unwrap_or_default(),
            }
        };
        let e = set("elsewhere", &[("blocks.9.proj.lora_A.weight", a.clone()), ("blocks.9.proj.lora_B.weight", b.clone())]);
        assert!(e.contains("none of this model's") && e.contains("blocks.*.proj"), "{e}");
        let e = set("wide", &[("blocks.0.proj.lora_A.weight", t(&[1., 2.], (1, 2))), ("blocks.0.proj.lora_B.weight", b.clone())]);
        assert!(e.contains("is [1, 2] over [2, 1]") && e.contains("inp: 3"), "{e}");
        let e = set("half", &[("blocks.0.proj.lora_A.weight", a.clone())]);
        assert!(e.contains("one factor of the two"), "{e}");
        let e = set("loha", &[("blocks.0.proj.hada_w1_a", a.clone())]);
        assert!(e.contains("a LoHa"), "{e}");
        let e = set("stray", &[("blocks.0.proj.lora_A.weight", a.clone()), ("blocks.0.proj.lora_B.weight", b.clone()), ("text_encoder.x.weight", a)]);
        assert!(e.contains("no LoRA factor") && e.contains("text_encoder.x.weight"), "{e}");
        // Nothing is left set by a refusal.
        assert!(!adapters.0.read().unwrap().layers[0].set.is_some());
    }
}

#[cfg(all(test, target_os = "macos"))]
mod cost {
    use super::*;

    /// Where a side path's time goes, piece by piece, at Qwen-Image's shapes
    /// on Metal: 4096 rows of 3072 in, rank 64, to 3072 and to 12288 out.
    ///
    ///     cargo test --release -p kvad-gpu lora::cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn where_a_side_path_spends() {
        let dev = Device::new_metal(0).unwrap();
        let time = |what: &str, f: &dyn Fn() -> Tensor| {
            for _ in 0..3 {
                f();
            }
            dev.synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            dev.synchronize().unwrap();
            eprintln!("  {what}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
        };
        for out in [3072, 12288] {
            eprintln!("3072 → {out}, rank 64:");
            let x = Tensor::randn(0f32, 1.0, (4096, 3072), &dev).unwrap();
            let y = Tensor::randn(0f32, 1.0, (4096, out), &dev).unwrap();
            let a = Proj::Dense(Tensor::randn(0f32, 0.02, (3072, 64), &dev).unwrap().to_dtype(DType::BF16).unwrap());
            let b = Proj::Dense(Tensor::randn(0f32, 0.02, (64, out), &dev).unwrap().to_dtype(DType::BF16).unwrap());
            let xb = x.to_dtype(DType::BF16).unwrap();
            let h = a.forward(&xb).unwrap();
            let side = b.forward(&h).unwrap();
            time("x to bf16", &|| x.to_dtype(DType::BF16).unwrap());
            time("x·A", &|| a.forward(&xb).unwrap());
            time("·B", &|| b.forward(&h).unwrap());
            time("× scale", &|| (&side * 0.125).unwrap());
            time("to f32", &|| side.to_dtype(DType::F32).unwrap());
            let sf = side.to_dtype(DType::F32).unwrap();
            time("y + side", &|| (&y + &sf).unwrap());
            let Proj::Dense(bw) = &b else { unreachable!() };
            time("y += h·B, in place", &|| {
                assert!(crate::mpp::dense_acc(&y, &h, bw).unwrap());
                y.clone()
            });
        }
    }
}

#[cfg(test)]
mod sd_tests {
    use super::*;
    use crate::common::Loader;
    use crate::image::clip::{self, Clip, ClipConfig, Pooled};
    use crate::image::nn::Ctx;
    use crate::image::unet::{Unet, UnetConfig};
    use crate::image::{open, read_json, sd15, sdxl};
    use crate::qcache::Vault;
    use kvad::weights::{fetch_file, Watcher};

    /// SDXL's and SD 1.5's LoRAs against diffusers with PEFT
    /// (`scripts/lora-fixtures.py --pipeline sdxl|sd15`), one fixture a
    /// LoRA, each found by the name in its file's: what the text encoders
    /// make of the prompt, and one UNet call on the reference's own hidden
    /// states, without the LoRA, with it, and at half its strength; and the
    /// LoRA's own part of each, the adapted less the plain. On the CPU in
    /// f32, and on Metal in f16 as the pipelines run.
    ///
    ///     KVAD_LORA_FIXTURES=/tmp/lora-fx cargo test --release -p kvad-gpu sd_loras_agree -- --ignored --nocapture
    #[test]
    #[ignore]
    fn sd_loras_agree_with_peft() {
        let dir = std::env::var("KVAD_LORA_FIXTURES").expect("KVAD_LORA_FIXTURES, from scripts/lora-fixtures.py");
        let db = |want: &Tensor, got: &Tensor| -> f64 {
            let (w, g) = (want.to_dtype(DType::F32).unwrap(), got.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap());
            let e = (&w - &g).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
            let s = w.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;
            10.0 * (s / e.max(1e-30)).log10()
        };
        let w = Watcher::none();
        let tok = tokenizers::Tokenizer::from_file(fetch_file(sdxl::TOKENIZER_REPO, "tokenizer.json", &w).unwrap()).unwrap();
        let prompt = "a red fox sitting in fresh snow, photograph";
        let mut fixtures: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
        fixtures.sort();
        let mut seen = 0;
        for path in fixtures {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let Some((pipeline, name)) = stem.split_once('_').filter(|(p, _)| *p == "sdxl" || *p == "sd15") else { continue };
            // A file's name keeps its ending under the fixture's own.
            let name = name.replace("--", "/").replace("@@", ":");
            let name = if name.contains(':') && !name.ends_with(".safetensors") { format!("{name}.safetensors") } else { name };
            let xl = pipeline == "sdxl";
            let fx = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
            let get = |k: &str| fx.get(k).unwrap_or_else(|| panic!("{stem}: no `{k}`")).clone();
            let here = kvad::lora::local(&name).unwrap_or_else(|| panic!("{name} is not on this machine"));
            let file = File::open(&here.file).unwrap().named(&name);
            let repo = if xl { sdxl::REPO } else { sd15::REPO };
            let config = |d: &str| read_json(&fetch_file(repo, &format!("{d}/config.json"), &w).unwrap()).unwrap();
            let weights = |d: &str, stem: &str| vec![sdxl::weights(repo, d, stem, &w).unwrap()];
            let metal = Device::new_metal(0).ok();
            for (device, dtype) in [(Device::Cpu, DType::F32)].into_iter().chain(metal.map(|m| (m, DType::F16))) {
                let vault = Vault::off();
                let cx = Ctx { ld: Loader::new(None, device.clone(), &vault), dtype };
                let adapters = Adapters::new(if xl { &sdxl::PREFIXES } else { &sd15::PREFIXES });
                let clip = |d: &str, part: &'static str, pooled: Pooled| {
                    let r = open(&weights(d, "model"), dtype).unwrap().with_adapters(adapters.part(part));
                    Clip::load(&cx, &r, ClipConfig::from_json(&config(d)).unwrap(), pooled).unwrap()
                };
                let r = open(&weights("unet", "diffusion_pytorch_model"), dtype).unwrap().with_adapters(adapters.part("unet"));
                let unet = Unet::load(&cx, &r, UnetConfig::from_json(&config("unet")).unwrap()).unwrap();
                adapters.alias_ldm("unet");
                let clips = match xl {
                    true => vec![clip("text_encoder", "te1", Pooled::No), clip("text_encoder_2", "te2", Pooled::Projected)],
                    false => vec![clip("text_encoder", "te1", Pooled::Last)],
                };
                // What the pipelines make of the prompt: SDXL's two encoders'
                // states side by side, bigG's padded with `!`, and its pooled.
                let encode = || -> (Tensor, Option<Tensor>) {
                    let (ids, _) = clip::tokenize(&tok, prompt, clip::END).unwrap();
                    let (l, _) = clips[0].encode(&ids, 0).unwrap();
                    if !xl {
                        return (l, None);
                    }
                    let (ids, end) = clip::tokenize(&tok, prompt, 0).unwrap();
                    let (g, pooled) = clips[1].encode(&ids, end).unwrap();
                    (Tensor::cat(&[&l, &g], 2).unwrap(), pooled)
                };
                let on = |t: Tensor| t.to_device(&device).unwrap().to_dtype(dtype).unwrap();
                let (x, t) = (on(get("x")), get("t").to_vec1::<f32>().unwrap()[0] as f64);
                let time_ids = [512.0, 512.0, 0.0, 0.0, 512.0, 512.0];
                let run = |which: &str| -> (Tensor, Option<Tensor>, Tensor) {
                    let (hidden, pooled) = encode();
                    let (ctx, p) = (on(get(&format!("{which}_hidden"))), xl.then(|| on(get(&format!("{which}_pooled")))));
                    let eps = unet.forward(&x, t, &ctx, p.as_ref().map(|p| (p, &time_ids))).unwrap();
                    (hidden, pooled, eps)
                };
                let plain = run("plain");
                adapters.set(&[(&file, 1.0)], &device, dtype).unwrap();
                let adapted = run("adapted");
                adapters.set(&[(&file, 0.5)], &device, dtype).unwrap();
                let half = run("half");
                let mut line = format!("{name}, {dtype:?}:");
                let mut worst_part = f64::INFINITY;
                for (what, i) in [("hidden", 0), ("eps", 2)] {
                    let pick = |r: &(Tensor, Option<Tensor>, Tensor)| if i == 0 { r.0.clone() } else { r.2.clone() };
                    let (p, a, h) = (db(&get(&format!("plain_{what}")), &pick(&plain)), db(&get(&format!("adapted_{what}")), &pick(&adapted)), db(&get(&format!("half_{what}")), &pick(&half)));
                    let part_ref = (get(&format!("adapted_{what}")) - get(&format!("plain_{what}"))).unwrap();
                    let moved = part_ref.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() > 0.0;
                    let part = match moved {
                        true => db(&part_ref, &(pick(&adapted).to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap() - pick(&plain).to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap()).unwrap()),
                        false => f64::INFINITY,
                    };
                    line += &format!(" {what} {p:.1} / {a:.1} / {h:.1} dB, part {part:.1};");
                    let floor = if dtype == DType::F32 { 80.0 } else { 30.0 };
                    assert!(a > floor && h > floor, "{line}");
                    worst_part = worst_part.min(part);
                }
                eprintln!("{line}");
                if dtype == DType::F32 {
                    assert!(worst_part > 60.0, "{line}: a LoRA's part further from the reference's than f32 explains");
                }
            }
            seen += 1;
        }
        assert!(seen > 0, "no sdxl_ or sd15_ fixtures in {dir}");
    }
}
