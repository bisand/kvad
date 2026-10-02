//! A folder of captioned pictures, and each as a denoiser is trained on it.
//!
//! The folder is the kohya and diffusers convention: pictures, and beside
//! each a `.txt` of the same name with its caption.
//!
//! ```text
//! my-photos/
//!   one.jpg   one.txt     "a photo of sks dog on a beach"
//!   two.png   two.txt     "sks dog asleep on a sofa"
//! ```
//!
//! Training never looks at a picture or a caption again after it has read
//! them once. The denoiser is trained on *latents*, what the VAE's encoder
//! makes of a picture, and on what the text encoders make of its caption,
//! and neither changes while a LoRA on the denoiser is trained. So each is
//! read once, before the first step ([`read_all`]), and kept on disk
//! beside the weights' cache for the next run; and the encoders that read
//! them, 1.6 GB of them for SDXL and 16 GB for Qwen-Image, are gone before
//! the denoiser is loaded.
//!
//! What is kept of a picture is not one latent. The encoder answers with a
//! Gaussian, a mean and a spread for every number, and each step draws its
//! own latent from it, which is what the VAE was trained to make its
//! decoder robust to. The spread is small, and it is the only thing in a
//! picture that differs from one epoch to the next.

use super::sdxl::Readers;
use candle_core::{DType, Device, Tensor};
use kvad::image::Image;
use kvad::weights::Watcher;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What a picture's file may end in. `ffmpeg` decodes them, as it does a
/// video's first frame (`kvad::video::picture_from_file`).
pub const KINDS: [&str; 5] = ["jpg", "jpeg", "png", "webp", "bmp"];

/// One picture and its caption.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub path: PathBuf,
    pub caption: String,
}

/// The pictures in `dir`, by name, each with the caption in the `.txt`
/// beside it, on one line; or with `fallback` where it has none.
///
/// Without a fallback, a picture with no caption is refused, by name: one
/// trained with an empty caption by accident teaches the model that the
/// subject is what it draws when told nothing.
pub fn folder(dir: &Path, fallback: Option<&str>) -> Res<Vec<Entry>> {
    let kind = |p: &Path| p.extension().and_then(|e| e.to_str()).is_some_and(|e| KINDS.contains(&e.to_ascii_lowercase().as_str()));
    let hidden = |p: &Path| p.file_name().and_then(|n| n.to_str()).is_none_or(|n| n.starts_with('.'));
    let mut pictures: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("could not read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && kind(p) && !hidden(p))
        .collect();
    pictures.sort();
    if pictures.is_empty() {
        return Err(format!("{} holds no pictures: none of its files end in {}", dir.display(), KINDS.join(", ")).into());
    }
    let (mut entries, mut bare) = (Vec::new(), Vec::new());
    for path in pictures {
        let caption = match std::fs::read_to_string(path.with_extension("txt")) {
            Ok(text) => Some(text.split_whitespace().collect::<Vec<_>>().join(" ")),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => fallback.map(str::to_string),
            Err(e) => return Err(format!("could not read {}: {e}", path.with_extension("txt").display()).into()),
        };
        match caption {
            Some(caption) => entries.push(Entry { path, caption }),
            None => bare.push(path.file_name().unwrap_or_default().to_string_lossy().into_owned()),
        }
    }
    if !bare.is_empty() {
        let more = match bare.len() > 6 {
            true => format!(" and {} more", bare.len() - 6),
            false => String::new(),
        };
        return Err(format!(
            "{} of the pictures in {} have no caption: {}{more}.\n  A caption is a `.txt` of the same name beside the picture; `--caption TEXT` gives one to every picture without.",
            bare.len(),
            dir.display(),
            bare.iter().take(6).cloned().collect::<Vec<_>>().join(", "),
        )
        .into());
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// One size
// ---------------------------------------------------------------------------

/// For each of `to` pixels along one side, the pixels of the `from` it is
/// made of and how much of each.
///
/// **Smaller**: a new pixel covers `from / to` old ones, and is their
/// mean, each weighed by how much of it lies under the new pixel, the two
/// at the edges being covered in part. Nothing is skipped, so fine detail
/// is averaged and does not turn into the false patterns that taking
/// every nth pixel makes of it. It is what kohya's scripts shrink with
/// (OpenCV's `INTER_AREA`).
///
/// **Larger**: each new pixel's centre falls between two old ones, and is
/// their mix by how near each is.
fn taps(from: usize, to: usize) -> Vec<Vec<(usize, f32)>> {
    let s = from as f64 / to as f64;
    (0..to)
        .map(|i| match s >= 1.0 {
            true => {
                let (a, b) = (i as f64 * s, (i + 1) as f64 * s);
                let covered = |j: usize| (b.min(j as f64 + 1.0) - a.max(j as f64)) / s;
                (a.floor() as usize..(b.ceil() as usize).min(from)).map(|j| (j, covered(j) as f32)).filter(|t| t.1 > 0.0).collect()
            }
            false => {
                let c = ((i as f64 + 0.5) * s - 0.5).clamp(0.0, (from - 1) as f64);
                let (j, f) = (c.floor() as usize, c.fract() as f32);
                match j + 1 < from {
                    true => vec![(j, 1.0 - f), (j + 1, f)],
                    false => vec![(j, 1.0)],
                }
            }
        })
        .collect()
}

/// `image` at `width` × `height`: along each row, then down each column
/// ([`taps`]).
fn resized(image: &Image, width: usize, height: usize) -> Image {
    if (image.width, image.height) == (width, height) {
        return image.clone();
    }
    let (across, down) = (taps(image.width, width), taps(image.height, height));
    let mut rows = vec![0f32; image.height * width * 3];
    for y in 0..image.height {
        let from = &image.rgb[y * image.width * 3..(y + 1) * image.width * 3];
        for (x, taps) in across.iter().enumerate() {
            for c in 0..3 {
                rows[(y * width + x) * 3 + c] = taps.iter().map(|&(j, w)| from[j * 3 + c] as f32 * w).sum();
            }
        }
    }
    let mut rgb = vec![0u8; width * height * 3];
    for (y, taps) in down.iter().enumerate() {
        for i in 0..width * 3 {
            let v: f32 = taps.iter().map(|&(j, w)| rows[j * width * 3 + i] * w).sum();
            rgb[y * width * 3 + i] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Image { width, height, rgb }
}

/// `image` as a square `side` pixels wide: made smaller or larger until its
/// short side is `side`, and its middle cut out.
///
/// With it, SDXL's six sizes for the picture, as diffusers' training script
/// gives them: the height and width it had, where the cut starts in the
/// resized picture, top then left, and the height and width it is now.
/// SDXL was told these for every picture it was trained on, so that it
/// could be asked afterwards for one that is neither small nor cut; a LoRA
/// is trained telling it the truth as well.
pub(crate) fn fitted(image: &Image, side: usize) -> (Image, [f64; 6]) {
    let scale = side as f64 / image.width.min(image.height) as f64;
    let (w, h) = (((image.width as f64 * scale).round() as usize).max(side), ((image.height as f64 * scale).round() as usize).max(side));
    let whole = resized(image, w, h);
    let (left, top) = ((w - side) / 2, (h - side) / 2);
    let mut rgb = Vec::with_capacity(side * side * 3);
    for y in top..top + side {
        rgb.extend_from_slice(&whole.rgb[(y * w + left) * 3..(y * w + left + side) * 3]);
    }
    let ids = [image.height as f64, image.width as f64, top as f64, left as f64, side as f64, side as f64];
    (Image { width: side, height: side, rgb }, ids)
}

/// A picture as a VAE reads one: `[1, 3, H, W]` in `[−1, 1]`, on the host.
fn pixels(image: &Image) -> Res<Tensor> {
    let n = image.width * image.height;
    let mut planes = vec![0f32; n * 3];
    for (i, &v) in image.rgb.iter().enumerate() {
        planes[(i % 3) * n + i / 3] = v as f32 / 127.5 - 1.0;
    }
    Ok(Tensor::from_vec(planes, (1, 3, image.height, image.width), &Device::Cpu)?)
}

// ---------------------------------------------------------------------------
// Read once
// ---------------------------------------------------------------------------

/// A picture and its caption as SDXL's UNet is trained on them, on the
/// host.
pub(crate) struct Seen {
    /// The picture's file name, for what is said about it.
    pub(crate) name: String,
    /// The Gaussian over its latents, in the UNet's units and in f32,
    /// `[1, 4, side/8, side/8]` each: the mean, and the spread.
    pub(crate) mean: Tensor,
    pub(crate) spread: Tensor,
    /// Its caption: `[1, 77, 2048]` a token and `[1, 1280]` pooled, in the
    /// precision the text encoders ran in.
    pub(crate) ctx: Tensor,
    pub(crate) pooled: Tensor,
    /// Its six sizes ([`fitted`]).
    pub(crate) ids: [f64; 6],
}

impl Seen {
    fn save(&self, path: &Path) -> Res<()> {
        let ids = Tensor::from_vec(self.ids.to_vec(), 6, &Device::Cpu)?;
        let tensors: HashMap<&str, Tensor> = [("mean", &self.mean), ("spread", &self.spread), ("ctx", &self.ctx), ("pooled", &self.pooled), ("ids", &ids)].into_iter().map(|(k, t)| (k, t.clone())).collect();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // Written aside and moved into place, so that a run that is stopped
        // while it writes leaves no half of a file for the next to read.
        let aside = path.with_extension(format!("{}.part", std::process::id()));
        candle_core::safetensors::save(&tensors, &aside)?;
        Ok(std::fs::rename(&aside, path)?)
    }

    fn load(path: &Path, name: &str) -> Option<Self> {
        let mut t = candle_core::safetensors::load(path, &Device::Cpu).ok()?;
        let ids: [f64; 6] = t.remove("ids")?.to_vec1::<f64>().ok()?.try_into().ok()?;
        Some(Seen { name: name.to_string(), mean: t.remove("mean")?, spread: t.remove("spread")?, ctx: t.remove("ctx")?, pooled: t.remove("pooled")?, ids })
    }
}

/// Where what [`read_all`] made of each picture is kept: beside the models
/// trained here.
pub fn cache_dir() -> PathBuf {
    kvad::weights::data_dir().join("tune-cache")
}

/// A name for everything that decides what a picture is read as: FNV-1a
/// over `parts`, each closed by a byte no part holds.
fn key(parts: &[&str]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in parts.iter().flat_map(|p| p.bytes().chain([0xff])) {
        h = (h ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Every entry as the UNet of `repo` is trained on it at `side` pixels,
/// read from `cache` where it was read before and by the text encoders and
/// the VAE where not; `said` is told as each is done. The encoders are
/// loaded only if something is to be read, and are gone when this returns.
///
/// A picture is read again if its file, its caption, the model or the size
/// has changed: all of them are in the name its reading is kept under.
///
/// `prompts` are read with them, each as the two text encoders read it,
/// `[1, 77, 2048]` a token and `[1, 1280]` pooled, on the host: what a run
/// draws its samples from, which it cannot read once the encoders are gone.
pub(crate) fn read_all(entries: &[Entry], prompts: &[String], repo: &str, side: usize, ffmpeg: &Path, device: &Device, cache: &Path, said: &mut dyn FnMut(&str)) -> Res<(Vec<Seen>, Vec<(Tensor, Tensor)>)> {
    let mut readers: Option<Readers> = None;
    let mut seen = Vec::with_capacity(entries.len());
    let (mut fresh, mut enlarged) = (0, Vec::new());
    for (i, e) in entries.iter().enumerate() {
        let name = e.path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let meta = std::fs::metadata(&e.path)?;
        let changed = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        let whole = std::fs::canonicalize(&e.path)?;
        let file = cache.join(format!(
            "{}.safetensors",
            key(&["1", repo, super::sdxl::VAE_REPO, &side.to_string(), &whole.to_string_lossy(), &meta.len().to_string(), &changed.to_string(), &e.caption])
        ));
        if let Some(s) = Seen::load(&file, &name) {
            seen.push(s);
            continue;
        }
        if readers.is_none() {
            said("loading the text encoders and the VAE's encoder");
            readers = Some(Readers::load(repo, device, &Watcher::none())?);
        }
        let readers = readers.as_ref().expect("just loaded");
        let picture = kvad::video::picture_from_file(ffmpeg, &e.path, 0).map_err(|err| format!("{name}: {err}"))?;
        if picture.width.min(picture.height) < side {
            enlarged.push(name.clone());
        }
        let (square, ids) = fitted(&picture, side);
        // A pool a picture: see `pooled`.
        let s = crate::common::pooled(|| -> Res<Seen> {
            let (mean, spread) = readers.picture(&pixels(&square)?)?;
            let (ctx, pooled) = readers.caption(&e.caption)?;
            let host = |t: Tensor| t.to_device(&Device::Cpu);
            Ok(Seen { name, mean: host(mean)?, spread: host(spread)?, ctx: host(ctx)?, pooled: host(pooled)?, ids })
        })?;
        if s.mean.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().any(|v| !v.is_finite()) {
            return Err(format!("{}: the VAE's reading of it is not a number", s.name).into());
        }
        s.save(&file)?;
        fresh += 1;
        said(&format!("read {} of {}: {} ({}×{})", i + 1, entries.len(), s.name, picture.width, picture.height));
        seen.push(s);
    }
    if fresh < entries.len() {
        said(&format!("{} of {} pictures were read before, and are as {} kept them", entries.len() - fresh, entries.len(), cache.display()));
    }
    if !enlarged.is_empty() {
        said(&format!("{} picture(s) are smaller than {side} pixels on their short side and were enlarged, which teaches blur: {}", enlarged.len(), enlarged.iter().take(6).cloned().collect::<Vec<_>>().join(", ")));
    }
    let mut read = Vec::with_capacity(prompts.len());
    for text in prompts {
        let file = cache.join(format!("{}.safetensors", key(&["prompt 1", repo, text])));
        let kept = candle_core::safetensors::load(&file, &Device::Cpu).ok().and_then(|mut t| Some((t.remove("ctx")?, t.remove("pooled")?)));
        if let Some(pair) = kept {
            read.push(pair);
            continue;
        }
        if readers.is_none() {
            said("loading the text encoders and the VAE's encoder");
            readers = Some(Readers::load(repo, device, &Watcher::none())?);
        }
        let readers = readers.as_ref().expect("just loaded");
        let (ctx, pooled) = crate::common::pooled(|| -> Res<(Tensor, Tensor)> {
            let (ctx, pooled) = readers.caption(text)?;
            Ok((ctx.to_device(&Device::Cpu)?, pooled.to_device(&Device::Cpu)?))
        })?;
        std::fs::create_dir_all(cache)?;
        let aside = file.with_extension(format!("{}.part", std::process::id()));
        candle_core::safetensors::save(&HashMap::from([("ctx", ctx.clone()), ("pooled", pooled.clone())]), &aside)?;
        std::fs::rename(&aside, &file)?;
        read.push((ctx, pooled));
    }
    drop(readers);
    crate::common::settle(device)?;
    Ok((seen, read))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvad-dataset-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_folder_is_its_pictures_each_with_the_caption_beside_it() {
        let d = dir("folder");
        for (name, text) in [("b.PNG", ""), ("a.jpg", ""), ("a.txt", " a photo\nof a dog \n"), ("b.txt", "a cat"), ("notes.md", ""), (".hidden.png", "")] {
            std::fs::write(d.join(name), text).unwrap();
        }
        let got = folder(&d, None).unwrap();
        assert_eq!(got.iter().map(|e| (e.path.file_name().unwrap().to_str().unwrap(), e.caption.as_str())).collect::<Vec<_>>(), [("a.jpg", "a photo of a dog"), ("b.PNG", "a cat")]);

        // One without a caption is refused by name, or takes the fallback.
        std::fs::write(d.join("c.webp"), "").unwrap();
        let refused = folder(&d, None).unwrap_err().to_string();
        assert!(refused.contains("c.webp") && refused.contains("--caption"), "{refused}");
        assert_eq!(folder(&d, Some("a thing")).unwrap()[2].caption, "a thing");
        assert!(folder(&dir("empty"), None).unwrap_err().to_string().contains("no pictures"));
    }

    #[test]
    fn shrinking_averages_and_enlarging_mixes() {
        // Every new pixel's shares are of the whole of it.
        for (from, to) in [(7, 3), (1000, 512), (512, 512), (3, 7), (1, 4)] {
            for t in taps(from, to) {
                assert!((t.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5, "{from} to {to}: {t:?}");
                assert!(t.iter().all(|x| x.0 < from));
            }
        }
        // 3 from 6 is each pair's mean; 2 from 3 takes a pixel and half the
        // middle one, two to one.
        assert_eq!(taps(6, 3)[1], [(2, 0.5), (3, 0.5)]);
        let t = taps(3, 2);
        assert!((t[0][0].1 - 2.0 / 3.0).abs() < 1e-6 && (t[0][1].1 - 1.0 / 3.0).abs() < 1e-6 && t[1][0].0 == 1);

        // A checkerboard of single pixels, halved, is grey, where every
        // other pixel would be all black or all white.
        let board = Image { width: 4, height: 4, rgb: (0..16).flat_map(|i| [if (i % 4 + i / 4) % 2 == 0 { 255u8 } else { 0 }; 3]).collect() };
        assert!(resized(&board, 2, 2).rgb.iter().all(|&v| v == 128));
    }

    #[test]
    fn a_picture_is_fitted_by_its_short_side_and_cut_from_its_middle() {
        // 8 wide and 4 high, each column its own grey: to 2×2, it is
        // halved to 4×2 and the middle two columns are kept.
        let wide = Image { width: 8, height: 4, rgb: (0..32).flat_map(|i| [(i % 8) as u8 * 10; 3]).collect() };
        let (cut, ids) = fitted(&wide, 2);
        assert_eq!((cut.width, cut.height), (2, 2));
        assert_eq!(ids, [4.0, 8.0, 0.0, 1.0, 2.0, 2.0]);
        // Columns 2,3 and 4,5 of the original, averaged.
        assert_eq!(cut.rgb[..6], [25, 25, 25, 45, 45, 45]);

        let p = pixels(&cut).unwrap();
        assert_eq!(p.dims(), [1, 3, 2, 2]);
        let v = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - (25.0 / 127.5 - 1.0)).abs() < 1e-6 && (v[1] - (45.0 / 127.5 - 1.0)).abs() < 1e-6);
    }

    #[test]
    fn a_reading_is_named_by_everything_it_depends_on() {
        assert_eq!(key(&["a", "b"]), key(&["a", "b"]));
        assert_ne!(key(&["ab", ""]), key(&["a", "b"]));
        assert_ne!(key(&["a", "b"]), key(&["a", "c"]));
    }
}
