//! What a community GGUF holds, and whether it runs right here.
//!
//!     cargo run --release -p kvad-gpu --example gguf -- FILE [--base REPO] [--every N]
//!
//! 1. **The make-up:** the architecture the file says, and how many
//!    matrices of each type. A "Q4_K_S" file is not all Q4_K.
//! 2. **The kernels:** one matrix of each quantised type, multiplied on
//!    Metal by candle's quantised kernels, against the same matrix
//!    dequantised and multiplied in f32 on the CPU: 4096 rows, a 1024²
//!    image's tokens, then one row, then 4096 rows narrowed out of a larger
//!    tensor, the case whose offset candle's Q8_0 kernel once ignored. With
//!    each, its time on Metal, as TFLOP/s.
//! 3. **With `--base`**, the diffusers repo the file was made from: its
//!    Q8_0 matrices against Kvad's own quantisation of the base's weights,
//!    block by block, for every `N`th matrix (all with `--every 1`).
//! 4. **Also with `--base`**: every `N`th quantised matrix of each type
//!    against the base's bf16, as dB, beside Kvad's own Q8_0 of the same
//!    matrices, which is what the file's types cost in the weights.
//!
//! With `--base`, the file is read under the base's names, as its loader
//! reads it: FLUX's `qkv` as `to_q`, `to_k` and `to_v`, and so on.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use kvad::weights::{fetch_file, Watcher};
use kvad_gpu::gguf::{type_name, Gguf};
use std::path::PathBuf;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Signal-to-error ratio in dB of `got` against `want`.
fn db(got: &Tensor, want: &Tensor) -> Res<f32> {
    let cpu = |t: &Tensor| -> candle_core::Result<Tensor> { t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all() };
    let (g, w) = (cpu(got)?, cpu(want)?);
    let err = (&g - &w)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let sig = w.sqr()?.sum_all()?.to_scalar::<f32>()?;
    if !(err + sig).is_finite() {
        return Err("a comparison met a NaN or an infinity".into());
    }
    Ok(10.0 * (sig / err.max(1e-30)).log10())
}

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let value = |f: &str| argv.iter().position(|a| a == f).and_then(|i| argv.get(i + 1)).cloned();
    let path = PathBuf::from(argv.first().filter(|a| !a.starts_with("--")).ok_or("a GGUF file is required")?);
    let every: usize = value("--every").map(|v| v.parse()).transpose()?.unwrap_or(20);
    // Under the base's names when there is a base: FLUX's are mapped.
    let file = match value("--base") {
        Some(base) => kvad_gpu::image::open_gguf(&path, &base, &Watcher::none())?,
        None => Gguf::open(&path)?,
    };

    // 1. The make-up.
    let mut names: Vec<&str> = file.names().collect();
    names.sort();
    eprintln!("1. {}: {} tensors of {}", path.display(), names.len(), file.text("general.architecture").unwrap_or("?"));
    eprintln!("   {}", file.make_up());

    // 2. The kernels, one matrix of each quantised type.
    let metal = Device::new_metal(0)?;
    eprintln!("2. candle's Metal kernels against f32 on the CPU:");
    let mut kinds: Vec<(GgmlDType, &str)> = Vec::new();
    for n in &names {
        let t = file.stored(n).ok_or("a name with no tensor")?;
        if t.quantised() && t.shape.len() == 2 && !kinds.iter().any(|(d, _)| *d == t.dtype) {
            kinds.push((t.dtype, n));
        }
    }
    let rows = 4096;
    for (dtype, name) in kinds {
        let q = file.qtensor(name, &metal)?;
        let (out, inp) = q.shape().dims2()?;
        let w = file.tensor(name)?;
        let x = Tensor::randn(0f32, 1., (1, rows + 8, inp), &Device::Cpu)?;
        let want = |x: &Tensor| -> Res<Tensor> { Ok(x.broadcast_matmul(&w.t()?)?) };
        let mm = QMatMul::from_qtensor(q)?;
        let run = |x: &Tensor| -> Res<(Tensor, f64)> {
            let x = x.to_device(&metal)?;
            let y = mm.forward(&x)?;
            metal.synchronize()?;
            let t = Instant::now();
            let y2 = mm.forward(&x)?;
            metal.synchronize()?;
            let s = t.elapsed().as_secs_f64();
            drop(y2);
            Ok((y, s))
        };
        let full = x.narrow(1, 0, rows)?.contiguous()?;
        let (y, s) = run(&full)?;
        let tflops = 2.0 * (rows * out * inp) as f64 / s / 1e12;
        let one = x.narrow(1, 0, 1)?.contiguous()?;
        let (y1, _) = run(&one)?;
        // Narrowed on the device, so the kernel sees an offset; forced
        // contiguous first, as `Proj::forward` does.
        let xm = x.to_device(&metal)?.narrow(1, 8, rows)?.force_contiguous()?;
        let yn = mm.forward(&xm)?;
        eprintln!(
            "   {:5} {name} [{out}, {inp}]: {rows} rows {:5.1} dB at {tflops:4.1} TFLOP/s; one row {:5.1} dB; narrowed {:5.1} dB",
            type_name(dtype),
            db(&y, &want(&full)?)?,
            db(&y1, &want(&one)?)?,
            db(&yn, &want(&x.narrow(1, 8, rows)?)?)?
        );
    }

    // 3. Its Q8_0 blocks against Kvad's own, from the base's weights.
    let Some(base) = value("--base") else { return Ok(()) };
    // LTX-2.5's DiT is one file of its own; a diffusers pipeline's is a
    // directory of shards.
    let paths = match file.text("general.architecture") {
        Some("ltxv") => vec![fetch_file(&base, kvad::video::LTX_DENOISER, &Watcher::none())?],
        _ => {
            let index = fetch_file(&base, "transformer/diffusion_pytorch_model.safetensors.index.json", &Watcher::none())?;
            let map: kvad::serde_json::Value = kvad::serde_json::from_str(&std::fs::read_to_string(index)?)?;
            let mut shards: Vec<String> = map["weight_map"].as_object().ok_or("no weight_map")?.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
            shards.sort();
            shards.dedup();
            shards.iter().map(|s| fetch_file(&base, &format!("transformer/{s}"), &Watcher::none())).collect::<Result<Vec<_>, _>>()?
        }
    };
    // SAFETY: read-only files in the Hub's cache.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&paths)? };
    let q8: Vec<&str> = names.iter().copied().filter(|n| file.stored(n).is_some_and(|t| t.dtype == GgmlDType::Q8_0)).collect();
    let (mut same, mut blocks, mut matrices, mut worst) = (0usize, 0usize, 0usize, f32::INFINITY);
    let t = Instant::now();
    for n in q8.iter().step_by(every) {
        let w = st.load(n, &Device::Cpu)?;
        let ours = QTensor::quantize(&w.to_dtype(DType::F32)?, GgmlDType::Q8_0)?;
        let theirs = file.qtensor(n, &Device::Cpu)?;
        let (a, b) = (ours.data()?, theirs.data()?);
        if a.len() != b.len() {
            return Err(format!("{n}: {} bytes here, {} in the file", a.len(), b.len()).into());
        }
        // 34-byte blocks: an f16 scale, then 32 signed bytes.
        let n_blocks = a.len() / 34;
        same += a.chunks(34).zip(b.chunks(34)).filter(|(x, y)| x == y).count();
        blocks += n_blocks;
        matrices += 1;
        if a != b {
            worst = worst.min(db(&theirs.dequantize(&Device::Cpu)?, &ours.dequantize(&Device::Cpu)?)?);
        }
    }
    let apart = match worst.is_finite() {
        true => format!("the least alike {worst:.1} dB apart"),
        false => "every matrix byte for byte".to_string(),
    };
    if !q8.is_empty() {
        eprintln!(
            "3. {matrices} of its {} Q8_0 matrices against Kvad's own Q8_0 of {base}: {same} of {blocks} blocks identical ({:.4}%), \
             {apart}; {:.0} s",
            q8.len(),
            100.0 * same as f64 / blocks.max(1) as f64,
            t.elapsed().as_secs_f64()
        );
    }

    // 4. Each type's matrices against the base's bf16, and Kvad's own
    //    Q8_0 of the same matrices for scale.
    let which = if every == 1 { "every matrix".to_string() } else { format!("every {every}th matrix") };
    eprintln!("4. {which} of each type against {base}'s bf16 (Kvad's own Q8_0 of the same):");
    let mut by: Vec<(GgmlDType, Vec<f32>, Vec<f32>)> = Vec::new();
    // Sampled within each type, since a type can fall in a stride with the
    // names: a Q4_K_S file's Q5_K matrices are its value projections.
    let mut quantised: Vec<&str> = Vec::new();
    for (dtype, _) in file.types() {
        let of: Vec<&str> = names.iter().copied().filter(|n| file.stored(n).is_some_and(|t| t.quantised() && t.dtype == dtype)).collect();
        quantised.extend(of.into_iter().step_by(every));
    }
    for n in &quantised {
        let dtype = file.stored(n).ok_or("a name with no tensor")?.dtype;
        let w = st.load(n, &Device::Cpu)?.to_dtype(DType::F32)?;
        let theirs = db(&file.tensor(n)?, &w)?;
        let ours = db(&QTensor::quantize(&w, GgmlDType::Q8_0)?.dequantize(&Device::Cpu)?, &w)?;
        match by.iter_mut().find(|(d, _, _)| *d == dtype) {
            Some((_, a, b)) => {
                a.push(theirs);
                b.push(ours);
            }
            None => by.push((dtype, vec![theirs], vec![ours])),
        }
    }
    for (dtype, theirs, ours) in by {
        let range = |v: &[f32]| (v.iter().copied().fold(f32::INFINITY, f32::min), v.iter().sum::<f32>() / v.len() as f32, v.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let (a, b) = (range(&theirs), range(&ours));
        eprintln!(
            "   {:5} {:3} matrices: {:4.1} to {:4.1} dB, mean {:4.1} (Q8_0: {:4.1} to {:4.1}, mean {:4.1})",
            type_name(dtype), theirs.len(), a.0, a.2, a.1, b.0, b.2, b.1
        );
    }

    // 5. Every plain tensor against the base's: exact where the file widened
    //    bf16 to f32, close where it narrowed to f16, and far off wherever
    //    a name was mapped to the wrong rows.
    let plain: Vec<&str> = names.iter().copied().filter(|n| file.stored(n).is_some_and(|t| !t.quantised())).collect();
    let (mut exact, mut worst, mut off) = (0, f32::INFINITY, Vec::new());
    for n in &plain {
        let w = st.load(n, &Device::Cpu)?.to_dtype(DType::F32)?;
        let got = file.tensor(n)?.to_dtype(DType::F32)?;
        if got.dims() != w.dims() {
            return Err(format!("{n}: {:?} in the file, {:?} in {base}", got.dims(), w.dims()).into());
        }
        match (&got - &w)?.abs()?.max_all()?.to_scalar::<f32>()? {
            0.0 => exact += 1,
            _ => {
                let d = db(&got, &w)?;
                worst = worst.min(d);
                if d < 40.0 {
                    off.push(format!("{n} {d:.1} dB"));
                }
            }
        }
    }
    let rest = match exact == plain.len() {
        true => String::new(),
        false => format!(", the rest {worst:.1} dB or closer"),
    };
    eprintln!("5. its {} plain tensors against {base}'s: {exact} exact{rest}", plain.len());
    for o in &off {
        eprintln!("   far off: {o}");
    }
    Ok(())
}
