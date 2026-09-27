//! Checkpoints quantised by someone else: GGUF files.
//!
//! Every quantised model here is otherwise quantised by Kvad, from the
//! maker's bf16, at load, and kept in [`crate::qcache`]. That only ever
//! makes Q8_0: nothing here chooses which matrices can take fewer bits. The
//! community's GGUFs have made that choice already, per matrix. A "Q4_K_M"
//! file of Qwen-Image is Q4_K for most of the transformer, Q5_K or Q6_K for
//! the matrices that suffer at four bits, and bf16 for the six small ones at
//! each end. A file like that is read here as it is: each matrix goes to the
//! device in the blocks its maker wrote, and nothing is quantised again.
//!
//! [`Gguf`] parses the header with candle's reader, which also turns
//! GGML's innermost-first dimensions into candle's order: a matrix stored
//! `[in, out]` in GGML's spelling is `[out, in]` here, which is
//! HuggingFace's layout and the one `QMatMul` wants. The data is read past
//! the page cache, as every other weight is ([`crate::uncached`]).
//!
//! It is a [`SimpleBackend`], so a [`crate::common::Reader`] reads it like
//! a safetensors file: a norm or a bias comes back as a tensor, in the dtype
//! asked for. A quantised matrix asked for that way is dequantised, which is
//! right but wasteful; the loader asks [`Gguf::blocks`] first.
//!
//! **Names.** Some files keep the model's original layout rather than
//! diffusers': city96's FLUX has one `qkv` matrix where diffusers has
//! `to_q`, `to_k` and `to_v`. [`Gguf::mapped`] presents such a file under
//! the names the loader asks for, each one some rows of the file's
//! tensors. A quantised matrix's rows are whole blocks, so a slice of rows
//! is a slice of bytes, and nothing is dequantised to split it.

use crate::uncached::{self, Pages};
use std::ops::{Deref, Range};
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One GGUF file, its header parsed and its data read on demand.
pub struct Gguf {
    path: PathBuf,
    file: File,
    metadata: HashMap<String, Value>,
    tensors: HashMap<String, Stored>,
    /// How many tensors of each type the file holds, however it is named.
    made: Vec<(GgmlDType, usize)>,
}

/// Where one tensor's bytes are, and what they are.
#[derive(Clone)]
pub struct Stored {
    pub dtype: GgmlDType,
    /// In candle's order: outermost first.
    pub shape: Vec<usize>,
    /// Its bytes: one span in the file, or several joined, for a tensor
    /// [`Gguf::mapped`] makes from rows of others.
    spans: Vec<(u64, usize)>,
    pub len: usize,
}

/// Some rows of one of the file's tensors: what a mapped name is made of.
pub struct Part {
    pub name: String,
    pub rows: Range<usize>,
}

impl Part {
    pub fn all(name: impl Into<String>, rows: usize) -> Self {
        Part { name: name.into(), rows: 0..rows }
    }

    pub fn rows(name: impl Into<String>, rows: Range<usize>) -> Self {
        Part { name: name.into(), rows }
    }
}

/// A tensor's bytes: read from the file in one span, or joined from several.
pub(crate) enum Bytes {
    Read(Pages),
    Joined(Vec<u8>),
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Read(p) => p,
            Bytes::Joined(v) => v,
        }
    }
}

impl Stored {
    /// Whether this is blocks rather than plain numbers.
    pub fn quantised(&self) -> bool {
        !matches!(self.dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
    }

    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A type as GGUF's makers name it, as in "Q4_K_S".
pub fn type_name(d: GgmlDType) -> &'static str {
    match d {
        GgmlDType::F32 => "F32",
        GgmlDType::F16 => "F16",
        GgmlDType::BF16 => "BF16",
        GgmlDType::Q4_0 => "Q4_0",
        GgmlDType::Q4_1 => "Q4_1",
        GgmlDType::Q5_0 => "Q5_0",
        GgmlDType::Q5_1 => "Q5_1",
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q8_1 => "Q8_1",
        GgmlDType::Q2K => "Q2_K",
        GgmlDType::Q3K => "Q3_K",
        GgmlDType::Q4K => "Q4_K",
        GgmlDType::Q5K => "Q5_K",
        GgmlDType::Q6K => "Q6_K",
        GgmlDType::Q8K => "Q8_K",
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Res<Self> {
        let bad = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
        // The header through the cache: it is a few hundred kilobytes, read
        // once, and `uncached::open` drops what the cache holds of the file.
        let content = Content::read(&mut File::open(path)?).map_err(|e| bad(&e))?;
        let file = uncached::open(path)?;
        let size = file.metadata()?.len();
        let mut tensors = HashMap::with_capacity(content.tensor_infos.len());
        for (name, info) in &content.tensor_infos {
            let (dtype, shape) = (info.ggml_dtype, info.shape.dims().to_vec());
            let elems: usize = shape.iter().product();
            if elems % dtype.block_size() != 0 {
                return Err(bad(&format!("`{name}` has {elems} numbers, not whole blocks of {:?}", dtype)).into());
            }
            let len = elems / dtype.block_size() * dtype.type_size();
            let at = content.tensor_data_offset + info.offset;
            if at + len as u64 > size {
                return Err(bad(&format!("`{name}` runs past the end of the file")).into());
            }
            tensors.insert(name.clone(), Stored { dtype, shape, spans: vec![(at, len)], len });
        }
        let mut made: Vec<(GgmlDType, usize)> = Vec::new();
        for t in tensors.values() {
            match made.iter_mut().find(|(d, _)| *d == t.dtype) {
                Some((_, c)) => *c += 1,
                None => made.push((t.dtype, 1)),
            }
        }
        made.sort_by_key(|&(d, c)| (std::cmp::Reverse(c), format!("{d:?}")));
        Ok(Gguf { path: path.to_path_buf(), file, metadata: content.metadata, tensors, made })
    }

    /// The same file under the names a loader asks for: each of `map`'s
    /// names is the rows its parts list, in order, of the file's tensors.
    ///
    /// Every row of a tensor a part names must be in exactly one part, so
    /// that a weight left out, or used twice, is an error here rather than
    /// a picture that is subtly wrong. A tensor no part names keeps its own
    /// name, for the unread-weights guard to find.
    pub fn mapped(self, map: Vec<(String, Vec<Part>)>) -> Res<Self> {
        let bad = |e: String| format!("{}: {e}", self.path.display());
        let mut used: HashMap<&str, Vec<Range<usize>>> = HashMap::new();
        let mut tensors = HashMap::with_capacity(map.len());
        for (to, parts) in &map {
            let mut made: Option<Stored> = None;
            for p in parts {
                let t = self.tensors.get(&p.name).ok_or_else(|| bad(format!("no `{}` for `{to}`", p.name)))?;
                let rows = t.shape.first().copied().unwrap_or(1);
                if p.rows.start >= p.rows.end || p.rows.end > rows || t.len % rows != 0 {
                    return Err(bad(format!("rows {:?} of `{}`, which has {rows}", p.rows, p.name)).into());
                }
                let row = t.len / rows;
                let (at, _) = t.spans[0];
                let span = (at + (p.rows.start * row) as u64, p.rows.len() * row);
                let mut shape = t.shape.clone();
                if let Some(first) = shape.first_mut() {
                    *first = p.rows.len();
                }
                made = Some(match made {
                    None => Stored { dtype: t.dtype, shape, spans: vec![span], len: span.1 },
                    Some(m) if m.dtype == t.dtype && m.shape[1..] == shape[1..] => {
                        let mut s = m.shape.clone();
                        s[0] += p.rows.len();
                        let mut spans = m.spans;
                        spans.push(span);
                        Stored { dtype: m.dtype, shape: s, spans, len: m.len + span.1 }
                    }
                    Some(_) => return Err(bad(format!("`{to}` joins tensors of different types or widths")).into()),
                });
                used.entry(p.name.as_str()).or_default().push(p.rows.clone());
            }
            tensors.insert(to.clone(), made.ok_or_else(|| bad(format!("`{to}` is made of nothing")))?);
        }
        for (name, mut ranges) in used {
            let rows = self.tensors[name].shape.first().copied().unwrap_or(1);
            ranges.sort_by_key(|r| r.start);
            let mut next = 0;
            for r in &ranges {
                if r.start != next {
                    return Err(bad(format!("rows {next}..{} of `{name}` are {}", r.start, if r.start > next { "in no name" } else { "in two" })).into());
                }
                next = r.end;
            }
            if next != rows {
                return Err(bad(format!("rows {next}..{rows} of `{name}` are in no name")).into());
            }
        }
        for (name, t) in &self.tensors {
            if !map.iter().any(|(_, parts)| parts.iter().any(|p| &p.name == name)) {
                tensors.insert(name.clone(), t.clone());
            }
        }
        Ok(Gguf { tensors, ..self })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn stored(&self, name: &str) -> Option<&Stored> {
        self.tensors.get(name)
    }

    /// The same file with every name under `prefix`: LTX-2.5's DiT file
    /// files its tensors under `model.diffusion_model.`, and a GGUF of it
    /// does not.
    pub fn prefixed(self, prefix: &str) -> Res<Self> {
        let map = self.tensors.iter().map(|(n, t)| (format!("{prefix}{n}"), vec![Part::all(n.clone(), t.shape.first().copied().unwrap_or(1))])).collect();
        self.mapped(map)
    }

    /// A string from the header, such as `general.architecture`.
    pub fn text(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.to_string().ok().map(String::as_str)
    }

    /// How many tensors of each type the file holds, most first: a
    /// "Q4_K_M" file's real make-up, counted as the file has them.
    pub fn types(&self) -> Vec<(GgmlDType, usize)> {
        self.made.clone()
    }

    /// The quantised matrices' types, most first, as "716 Q4_K, 124 Q5_K".
    pub fn make_up(&self) -> String {
        let q: Vec<String> = self.types().into_iter().filter(|(d, _)| !matches!(d, GgmlDType::F32)).map(|(d, n)| format!("{n} {}", type_name(d))).collect();
        q.join(", ")
    }

    /// Bytes the file's tensors take on the device, with its plain ones
    /// widened to `plain`, as a load in that dtype holds them.
    pub fn device_bytes(&self, plain: DType) -> usize {
        self.tensors.values().map(|t| if t.quantised() { t.len } else { t.elems() * plain.size_in_bytes() }).sum()
    }

    /// A tensor's bytes, as the file has them.
    pub(crate) fn bytes(&self, name: &str) -> candle_core::Result<Bytes> {
        let t = self.tensors.get(name).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        if let [(at, len)] = t.spans[..] {
            return Ok(Bytes::Read(uncached::read(&self.file, at, len)?));
        }
        let mut joined = Vec::with_capacity(t.len);
        for &(at, len) in &t.spans {
            joined.extend_from_slice(&uncached::read(&self.file, at, len)?);
        }
        Ok(Bytes::Joined(joined))
    }

    /// A quantised matrix's blocks, if `name` is one, checked against the
    /// shape the model expects. `None` for a tensor of plain numbers,
    /// which the caller reads as a tensor instead.
    pub(crate) fn blocks(&self, name: &str, shape: &[usize]) -> candle_core::Result<Option<(GgmlDType, Bytes)>> {
        let Some(t) = self.tensors.get(name).filter(|t| t.quantised()) else { return Ok(None) };
        if t.shape != shape {
            let msg = format!("shape mismatch for {name}");
            return Err(candle_core::Error::UnexpectedShape { msg, expected: shape.into(), got: t.shape.as_slice().into() }.bt());
        }
        Ok(Some((t.dtype, self.bytes(name)?)))
    }

    /// A quantised matrix as a `QTensor` on `device`, as its maker wrote it.
    pub fn qtensor(&self, name: &str, device: &Device) -> candle_core::Result<QTensor> {
        let t = self.tensors.get(name).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        let bytes = self.bytes(name)?;
        QTensor::new(QStorage::from_data(Cow::Borrowed(&bytes), device, t.dtype)?, t.shape.as_slice())
    }

    /// Any tensor as numbers on the host: plain ones as they are stored,
    /// quantised ones dequantised to f32.
    pub fn tensor(&self, name: &str) -> candle_core::Result<Tensor> {
        let t = self.tensors.get(name).ok_or_else(|| candle_core::Error::CannotFindTensor { path: name.to_string() }.bt())?;
        let plain = match t.dtype {
            GgmlDType::F32 => Some(DType::F32),
            GgmlDType::F16 => Some(DType::F16),
            GgmlDType::BF16 => Some(DType::BF16),
            _ => None,
        };
        match plain {
            Some(dtype) => Tensor::from_raw_buffer(&self.bytes(name)?, dtype, &t.shape, &Device::Cpu),
            None => self.qtensor(name, &Device::Cpu)?.dequantize(&Device::Cpu),
        }
    }
}

impl SimpleBackend for Gguf {
    fn get(&self, s: Shape, name: &str, _: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &s {
            let msg = format!("shape mismatch for {name}");
            return Err(candle_core::Error::UnexpectedShape { msg, expected: s, got: t.shape().clone() }.bt());
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.tensor(name)?.to_dtype(dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// A [`Gguf`] shared between the [`crate::common::Reader`] that offers its
/// blocks and the `VarBuilder` that reads its plain tensors.
pub(crate) struct Shared(pub(crate) std::sync::Arc<Gguf>);

impl SimpleBackend for Shared {
    fn get(&self, s: Shape, name: &str, init: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.0.get(s, name, init, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.0.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tensor this reader gives is the one candle's own GGUF reader
    /// gives, blocks and numbers alike, in a file with a quantised matrix of
    /// each kind the community's image files use and the plain ones beside
    /// them.
    #[test]
    fn reads_what_candles_reader_does() {
        let dir = std::env::temp_dir().join(format!("kvad-gpu-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        let ramp = |n: usize| -> Tensor {
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
            Tensor::from_vec(v, n, &Device::Cpu).unwrap()
        };
        let kinds = [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q4_0];
        let mut qs: Vec<(String, QTensor)> = kinds
            .iter()
            .enumerate()
            .map(|(i, &k)| (format!("blocks.{i}.weight"), QTensor::quantize(&ramp(3 * 512).reshape((3, 512)).unwrap(), k).unwrap()))
            .collect();
        let plain = [("norm.weight", DType::F32), ("proj.weight", DType::BF16), ("half.weight", DType::F16)];
        for (name, dtype) in plain {
            let t = ramp(64).reshape((2, 32)).unwrap().to_dtype(dtype).unwrap();
            let q = match dtype {
                DType::F32 => GgmlDType::F32,
                DType::F16 => GgmlDType::F16,
                _ => GgmlDType::BF16,
            };
            qs.push((name.to_string(), QTensor::quantize(&t, q).unwrap()));
        }
        let arch = Value::String("test".to_string());
        let refs: Vec<(&str, &QTensor)> = qs.iter().map(|(n, q)| (n.as_str(), q)).collect();
        candle_core::quantized::gguf_file::write(&mut File::create(&path).unwrap(), &[("general.architecture", &arch)], &refs).unwrap();

        let ours = Gguf::open(&path).unwrap();
        assert_eq!(ours.text("general.architecture"), Some("test"));
        let mut file = File::open(&path).unwrap();
        let content = Content::read(&mut file).unwrap();
        for (name, q) in &qs {
            let theirs = content.tensor(&mut file, name, &Device::Cpu).unwrap();
            let stored = ours.stored(name).unwrap();
            assert_eq!((stored.dtype, stored.shape.as_slice()), (q.dtype(), q.shape().dims()), "{name}");
            assert_eq!(&*ours.bytes(name).unwrap(), &*theirs.data().unwrap(), "{name}'s bytes");
            let bits = |t: &Tensor| -> Vec<u32> {
                t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|f| f.to_bits()).collect()
            };
            assert_eq!(bits(&ours.tensor(name).unwrap()), bits(&theirs.dequantize(&Device::Cpu).unwrap()), "{name}'s numbers");
            let blocks = ours.blocks(name, q.shape().dims()).unwrap();
            assert_eq!(blocks.is_some(), stored.quantised(), "{name}");
        }
        assert!(ours.blocks("blocks.0.weight", &[512, 3]).is_err(), "a transposed shape is refused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fused matrix read as three, by rows, blocks and all; two halves of
    /// another read in the other order; and a map that leaves rows out, or
    /// uses them twice, refused.
    #[test]
    fn a_mapped_file_is_rows_of_its_tensors() {
        let dir = std::env::temp_dir().join(format!("kvad-gpu-gguf-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        let ramp = |n: usize| -> Tensor {
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.29).cos() * 2.0 + i as f32 * 1e-3).collect();
            Tensor::from_vec(v, n, &Device::Cpu).unwrap()
        };
        let qkv = QTensor::quantize(&ramp(6 * 256).reshape((6, 256)).unwrap(), GgmlDType::Q4K).unwrap();
        let bias = QTensor::quantize(&ramp(6), GgmlDType::F32).unwrap();
        let halves = QTensor::quantize(&ramp(4 * 32).reshape((4, 32)).unwrap().to_dtype(DType::F16).unwrap(), GgmlDType::F16).unwrap();
        let arch = Value::String("test".to_string());
        let tensors = [("qkv.weight", &qkv), ("qkv.bias", &bias), ("mod.weight", &halves)];
        candle_core::quantized::gguf_file::write(&mut File::create(&path).unwrap(), &[("general.architecture", &arch)], &tensors).unwrap();

        let thirds = |n: &str, of: &str, rows: usize| -> Vec<(String, Vec<Part>)> {
            ["q", "k", "v"].iter().enumerate().map(|(i, x)| (format!("{x}.{n}"), vec![Part::rows(of, i * rows..(i + 1) * rows)])).collect()
        };
        let mut map = thirds("weight", "qkv.weight", 2);
        map.extend(thirds("bias", "qkv.bias", 2));
        map.push(("swapped.weight".to_string(), vec![Part::rows("mod.weight", 2..4), Part::rows("mod.weight", 0..2)]));
        let file = Gguf::open(&path).unwrap().mapped(map).unwrap();

        let whole = qkv.dequantize(&Device::Cpu).unwrap();
        let data = qkv.data().unwrap();
        for (i, x) in ["q", "k", "v"].iter().enumerate() {
            let (dtype, blocks) = file.blocks(&format!("{x}.weight"), &[2, 256]).unwrap().unwrap();
            assert_eq!(dtype, GgmlDType::Q4K);
            let row = data.len() / 6;
            assert_eq!(&*blocks, &data[i * 2 * row..(i + 1) * 2 * row], "{x}'s blocks are its rows'");
            let got = file.tensor(&format!("{x}.weight")).unwrap();
            let want = whole.narrow(0, i * 2, 2).unwrap();
            assert_eq!(got.to_vec2::<f32>().unwrap(), want.to_vec2::<f32>().unwrap(), "{x}'s numbers");
            let b = file.tensor(&format!("{x}.bias")).unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(b, ramp(6).narrow(0, i * 2, 2).unwrap().to_vec1::<f32>().unwrap());
        }
        let m = halves.dequantize(&Device::Cpu).unwrap();
        let want = Tensor::cat(&[m.narrow(0, 2, 2).unwrap(), m.narrow(0, 0, 2).unwrap()], 0).unwrap();
        let got = file.tensor("swapped.weight").unwrap().to_dtype(DType::F32).unwrap();
        assert_eq!(got.to_vec2::<f32>().unwrap(), want.to_vec2::<f32>().unwrap(), "the halves, swapped");
        let mut names: Vec<&str> = file.names().collect();
        names.sort();
        assert_eq!(names, ["k.bias", "k.weight", "q.bias", "q.weight", "swapped.weight", "v.bias", "v.weight"]);
        assert_eq!(file.types().iter().map(|(_, n)| n).sum::<usize>(), 3, "the make-up is the file's");

        let short = vec![("q.weight".to_string(), vec![Part::rows("qkv.weight", 0..2)])];
        assert!(Gguf::open(&path).unwrap().mapped(short).is_err(), "rows in no name are refused");
        let twice = vec![
            ("q.weight".to_string(), vec![Part::rows("qkv.weight", 0..4)]),
            ("k.weight".to_string(), vec![Part::rows("qkv.weight", 2..6)]),
        ];
        assert!(Gguf::open(&path).unwrap().mapped(twice).is_err(), "rows in two names are refused");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
