//! Images: what is asked for, what comes back, and the file it is saved as.
//!
//! Everything that makes an image lives in `kvad-gpu`, because every model
//! that does it is a convolution stack and this crate has no convolutions (see
//! `docs/image-plan.md` for why that is decided rather than missing). What is
//! here is the part the rest of the engine has to be able to *name* without
//! depending on candle: the request, the result, the progress report, and the
//! [`Painter`] trait a pipeline implements so that the service layer can hold
//! one the way it holds an [`Llm`](crate::runtime::Llm).
//!
//! It is a peer of the text path, not a mode of it. [`ImageRequest`] is not
//! [`Sampling`](crate::service::Sampling) with more fields: a denoiser has no
//! temperature and a language model has no guidance scale, and one type
//! stretched over both would be wrong for each.
//!
//! # PNG, by hand
//!
//! A PNG is a signature and a list of chunks, and the only hard part of
//! writing one is that the pixels have to be zlib-compressed. Except they do
//! not, quite: deflate has a *stored* block type that holds bytes as they are,
//! and a stream of stored blocks is a valid zlib stream. So [`Image::png`]
//! writes the format exactly, compresses nothing, and needs no dependency —
//! three checksums and some framing. A 1024×1024 image is 3 MB this way
//! rather than 1.5 MB or so compressed, which is a fair price for a file
//! format that fits on one screen.

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// What a caller asks a [`Painter`] for.
///
/// Every knob but the prompt is optional, because the right default is the
/// model's and not this crate's: SDXL was trained at 1024 pixels and wants
/// guidance near 5, Qwen-Image at 1328 and near 4. [`Painter::defaults`] fills
/// in what was left out, and [`ImageRequest::resolved`] is where the two meet.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageRequest {
    pub prompt: String,
    /// What to steer *away* from. Guidance moves along the difference between
    /// the prompt and this; left out, the difference is taken from nothing.
    pub negative_prompt: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Denoising steps: forward passes of the denoiser, twice each when
    /// guidance is on.
    pub steps: Option<usize>,
    /// How hard to push towards the prompt. 1 turns guidance off and halves
    /// the work.
    pub guidance: Option<f32>,
    /// The noise the image starts from. The same seed and settings give the
    /// same image.
    pub seed: Option<u64>,
    /// Whether each [`Step`] should carry a rough preview of the image so far.
    pub preview: bool,
    /// LoRAs to apply for this image, each at its strength.
    pub loras: Vec<Lora>,
    /// A picture to start from, where the image is an edit of one.
    pub edit: Option<Edit>,
}

/// A picture an image is made *from*, where it is not made from noise alone.
///
/// The picture is noised part of the way and denoised from there with the
/// prompt, which is SDEdit, and what diffusers calls image-to-image: the
/// prompt describes the picture wanted, not the change to make. `strength`
/// says how far up the noise the picture is taken, and so how much of it is
/// left: near 0 it comes back as it was, at 1 nothing of it remains but
/// where a mask kept it.
///
/// With a mask it is inpainting. Outside the mask the picture is put back
/// after every step, at that step's noise, so the model draws the masked
/// part in the knowledge of the rest and the rest comes out as it went in.
#[derive(Debug, Clone, PartialEq)]
pub struct Edit {
    pub image: Image,
    pub mask: Option<Mask>,
    /// 0 to 1. Without one, [`STRENGTH`], or 1 where there is a mask.
    pub strength: Option<f32>,
}

/// Where an [`Edit`] may change its picture: a byte a pixel, row by row,
/// 255 where the model draws anew and 0 where the picture is kept, and
/// between them a mix. The picture's own size.
#[derive(Clone, PartialEq, Eq)]
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub repaint: Vec<u8>,
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mask({}×{})", self.width, self.height)
    }
}

/// How far an edit noises its picture when the request does not say:
/// three quarters of the way, where the picture's layout and colours
/// survive and its detail is the prompt's. diffusers' default is 0.8.
pub const STRENGTH: f32 = 0.75;

/// A LoRA a request applies: its name, as a checkpoint's is (`repo`,
/// `repo:file.safetensors` or a path; [`crate::lora`]), and how strongly.
///
/// Applied for the one request and taken off after it, at run time beside
/// the model's own layers, so the model stays as it was loaded and the next
/// request chooses its own. `docs/lora-plan.md` says why.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Lora {
    pub name: String,
    /// The LoRA's product is multiplied by this: 1 as it was trained, 0.5
    /// half as strongly. Its own `alpha / rank` scale comes on top.
    #[serde(default = "full_strength")]
    pub scale: f64,
}

fn full_strength() -> f64 {
    1.0
}

impl Lora {
    /// `NAME` or `NAME:SCALE`, as the CLI takes one. A name holds colons of
    /// its own (`repo:file.safetensors`), so only a last part that is a
    /// number is a scale.
    pub fn parse(s: &str) -> Lora {
        match s.rsplit_once(':').map(|(n, x)| (n, x.parse::<f64>())) {
            Some((name, Ok(scale))) => Lora { name: name.to_string(), scale },
            _ => Lora { name: s.to_string(), scale: 1.0 },
        }
    }
}

/// A request a [`Painter`] found it cannot honour only once it tried: a
/// LoRA that does not fit the model, above all. The asker's to change, not
/// a fault of the server's, and answered as such.
#[derive(Debug)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// The most LoRAs one request may apply.
pub const MAX_LORAS: usize = 4;

/// A request's LoRAs, for a model that `takes` them or does not: none for
/// one that does not; at most [`MAX_LORAS`], each named, each once, each at
/// a scale between −10 and 10. The same for a picture and a video.
pub fn check_loras(loras: &[Lora], takes: bool) -> Res<()> {
    if !loras.is_empty() && !takes {
        return Err("this model takes no LoRAs; so far Qwen-Image, FLUX, SDXL, SD 1.5 and LTX-2.5 do".into());
    }
    if loras.len() > MAX_LORAS {
        return Err(format!("at most {MAX_LORAS} LoRAs, not {}", loras.len()).into());
    }
    for (i, l) in loras.iter().enumerate() {
        if l.name.trim().is_empty() {
            return Err("a LoRA has no name".into());
        }
        if !(l.scale.is_finite() && (-10.0..=10.0).contains(&l.scale)) {
            return Err(format!("{}'s scale must be between -10 and 10, not {}", l.name, l.scale).into());
        }
        if loras[..i].iter().any(|o| o.name == l.name) {
            return Err(format!("{} is asked for twice; give it one scale", l.name).into());
        }
    }
    Ok(())
}

/// A model's own answers for everything an [`ImageRequest`] may leave out.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Defaults {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    /// Width and height must both be a multiple of this: the VAE's
    /// downsampling, times the denoiser's patch size where it has one.
    pub multiple: usize,
    /// Whether the model does anything with guidance at all. A distilled
    /// model — FLUX.1-schnell — was trained to make an image without it, and
    /// has no use for a guidance scale or a negative prompt either.
    pub takes_guidance: bool,
    /// Whether it has a use for a negative prompt: only a model that is
    /// guided by running it twice, with the prompt and with what to steer
    /// away from. FLUX.1-dev takes guidance and no negative prompt: its
    /// guidance is a number the model reads, in one pass.
    pub takes_negative: bool,
    /// Whether a request may apply LoRAs to it ([`Lora`]).
    pub takes_loras: bool,
    /// Whether it makes an image from a picture ([`Edit`]).
    pub edits: bool,
}

/// An [`ImageRequest`] with every blank filled and checked.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    pub preview: bool,
    pub loras: Vec<Lora>,
    /// How far an edit noised its picture; `None` for an image that is not
    /// an edit. `steps` is still what was asked for: an edit runs the last
    /// `strength` of them ([`edit_steps`]).
    pub strength: Option<f32>,
    /// Whether an edit kept the picture outside a mask.
    pub masked: bool,
}

/// The steps an edit of `strength` runs out of `steps`: the bottom of the
/// schedule, from where the picture was noised to. At least one.
pub fn edit_steps(steps: usize, strength: f32) -> usize {
    ((steps as f32 * strength).round() as usize).clamp(1, steps)
}

/// The size an edit is made at when the request names none: the picture's
/// own shape, at about as many pixels as the model's own size has, each
/// side a multiple of what the model needs.
pub fn edit_size(picture: (usize, usize), d: &Defaults) -> (usize, usize) {
    let (w, h) = (picture.0.max(1) as f64, picture.1.max(1) as f64);
    let scale = ((d.width * d.height) as f64 / (w * h)).sqrt();
    let side = |n: f64| (((n * scale / d.multiple as f64).round() as usize).max(1) * d.multiple).min(MAX_SIDE / d.multiple * d.multiple);
    (side(w), side(h))
}

/// The largest side a request may ask for. Past this the VAE decode alone
/// wants tens of gigabytes, and a typo is more likely than an intent.
pub const MAX_SIDE: usize = 2048;

/// The most steps a request may ask for. Schedulers are trained on a thousand
/// noise levels; nobody gets a better image from more than a couple of
/// hundred of them.
pub const MAX_STEPS: usize = 200;

impl ImageRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        ImageRequest { prompt: prompt.into(), ..Default::default() }
    }

    /// Fill the blanks from `d` and refuse what cannot be drawn.
    ///
    /// A seed left out is chosen here, from the clock, so that the result can
    /// report which seed made it. An image that cannot be made again is a
    /// worse answer than one that can.
    pub fn resolved(&self, d: &Defaults) -> Res<Resolved> {
        if self.edit.is_some() && !d.edits {
            return Err("this model makes images from a prompt alone, not from a picture; SDXL and SD 1.5 edit one".into());
        }
        // An edit with no size is its picture's shape, and with one side
        // only, that side and the model's own for the other, as an image is.
        let fitted = match (&self.edit, self.width, self.height) {
            (Some(e), None, None) => Some(edit_size((e.image.width, e.image.height), d)),
            _ => None,
        };
        let width = self.width.or(fitted.map(|f| f.0)).unwrap_or(d.width);
        let height = self.height.or(fitted.map(|f| f.1)).unwrap_or(d.height);
        let steps = self.steps.unwrap_or(d.steps);
        let guidance = self.guidance.unwrap_or(d.guidance);
        for (what, n) in [("width", width), ("height", height)] {
            if n == 0 || n > MAX_SIDE {
                return Err(format!("{what} must be between {} and {MAX_SIDE}, not {n}", d.multiple).into());
            }
            if n % d.multiple != 0 {
                return Err(format!(
                    "{what} must be a multiple of {} for this model, and {n} is not; try {}",
                    d.multiple,
                    (n / d.multiple).max(1) * d.multiple
                )
                .into());
            }
        }
        if steps == 0 || steps > MAX_STEPS {
            return Err(format!("steps must be between 1 and {MAX_STEPS}, not {steps}").into());
        }
        if !(guidance.is_finite() && (0.0..=30.0).contains(&guidance)) {
            return Err(format!("guidance must be between 0 and 30, not {guidance}").into());
        }
        if self.prompt.trim().is_empty() {
            return Err("the prompt is empty".into());
        }
        // Refused rather than ignored: an image made without the guidance
        // somebody asked for should not be filed as if it had been.
        if !d.takes_guidance {
            if self.guidance.is_some_and(|g| g != 0.0) {
                return Err("this model makes images without guidance; leave guidance_scale out, or set it to 0".into());
            }
        }
        if !d.takes_negative && self.negative_prompt.as_deref().is_some_and(|n| !n.is_empty()) {
            return Err(match d.takes_guidance {
                false => "this model makes images without guidance, so it has no use for a negative prompt",
                true => "this model reads its guidance as a number and is run once a step, so it has no use for a negative prompt",
            }
            .into());
        }
        // Kept below 2³² so that it survives a round trip through a browser,
        // where every number is a double and a 64-bit seed would come back
        // as a different one.
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64 % (1 << 32))
                .unwrap_or(0)
        });
        check_loras(&self.loras, d.takes_loras)?;
        let strength = match &self.edit {
            None => None,
            Some(e) => {
                if e.image.width == 0 || e.image.height == 0 || e.image.rgb.len() != e.image.width * e.image.height * 3 {
                    return Err("the picture to edit is empty".into());
                }
                if let Some(m) = &e.mask {
                    if (m.width, m.height) != (e.image.width, e.image.height) || m.repaint.len() != m.width * m.height {
                        return Err(format!(
                            "the mask is {}×{} and the picture {}×{}; a mask is its picture's size",
                            m.width, m.height, e.image.width, e.image.height
                        )
                        .into());
                    }
                    if m.repaint.iter().all(|&b| b == 0) {
                        return Err("the mask keeps the whole picture, so there is nothing to draw".into());
                    }
                }
                // Inside a mask the usual wish is something new; without
                // one, the picture changed and still itself.
                let strength = e.strength.unwrap_or(if e.mask.is_some() { 1.0 } else { STRENGTH });
                if !(strength.is_finite() && strength > 0.0 && strength <= 1.0) {
                    return Err(format!("strength is above 0 and at most 1, not {strength}").into());
                }
                Some(strength)
            }
        };
        let negative_prompt = self.negative_prompt.clone().filter(|n| !n.is_empty());
        Ok(Resolved {
            prompt: self.prompt.clone(),
            negative_prompt,
            width,
            height,
            steps,
            guidance,
            seed,
            preview: self.preview,
            loras: self.loras.clone(),
            strength,
            masked: self.edit.as_ref().is_some_and(|e| e.mask.is_some()),
        })
    }
}

/// One finished denoising step.
#[derive(Debug, Clone)]
pub struct Step {
    /// Steps finished, counting this one.
    pub done: usize,
    pub total: usize,
    /// A cheap look at the image so far, when the request asked for one.
    ///
    /// Not a VAE decode — that costs as much as several steps. It is the
    /// latent's channels mixed straight into RGB by a fixed matrix, at the
    /// latent's resolution: blurry, slightly wrong in colour, and free.
    pub preview: Option<Image>,
}

/// An 8-bit RGB image, row by row from the top left.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Image({}×{})", self.width, self.height)
    }
}

/// What a finished generation hands back: the picture and how it was made.
#[derive(Debug, Clone)]
pub struct Painted {
    pub image: Image,
    /// The request as it was run, blanks filled — the seed above all.
    pub request: Resolved,
    /// Seconds spent in the text encoder, the denoising loop and the decoder.
    pub encode_secs: f64,
    pub denoise_secs: f64,
    pub decode_secs: f64,
}

/// A loaded text-to-image pipeline.
///
/// `Send` because it lives on the engine's thread, the way an `Llm` does, and
/// is built on whichever thread loaded it.
pub trait Painter: Send {
    /// Make one image.
    ///
    /// `on_step` is called after every denoising step. Returning `false` stops
    /// the generation, which then fails with "cancelled" rather than return a
    /// half-denoised image as if it were an answer.
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted>;

    fn defaults(&self) -> Defaults;

    /// One line for a person: architecture, size, where it runs.
    fn summary(&self) -> String;

    fn params(&self) -> usize;

    /// Bytes the weights take where they live, for memory accounting.
    fn weight_bytes(&self) -> usize;

    /// `metal f16`, `metal q8`: where and how this pipeline runs.
    fn backend(&self) -> String;
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------

impl Image {
    /// The image as a PNG file.
    pub fn png(&self) -> Vec<u8> {
        // Every row is prefixed by its filter type, and filter 0 is "none".
        let row = self.width * 3;
        let mut raw = Vec::with_capacity((row + 1) * self.height);
        for y in 0..self.height {
            raw.push(0);
            raw.extend_from_slice(&self.rgb[y * row..(y + 1) * row]);
        }

        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&(self.width as u32).to_be_bytes());
        ihdr.extend_from_slice(&(self.height as u32).to_be_bytes());
        // Bit depth 8, colour type 2 (RGB), then compression, filter and
        // interlace methods, each of which has exactly one legal value.
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &zlib_stored(&raw));
        chunk(&mut out, b"IEND", &[]);
        out
    }
}

/// A PNG chunk: length, type, data, and a CRC over the type and data.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// `data` as a zlib stream of uncompressed deflate blocks.
///
/// Two header bytes (`0x78 0x01`: deflate, 32 KB window, no dictionary, and a
/// check that makes the pair a multiple of 31), then blocks of at most 65535
/// bytes, each a header byte — `1` on the last block, `0` before it, with
/// block type 00 — and its length twice, the second time inverted. Then the
/// Adler-32 of the uncompressed bytes.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    const MAX: usize = 65535;
    let mut out = Vec::with_capacity(data.len() + data.len() / MAX * 5 + 16);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut blocks = data.chunks(MAX).peekable();
    if blocks.peek().is_none() {
        // An empty stream still needs one (empty, final) block.
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        let last = blocks.peek().is_none();
        let len = block.len() as u16;
        out.push(last as u8);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// CRC-32 as PNG and zip use it: reflected, polynomial `0xEDB88320`.
///
/// Bit by bit rather than from a 256-entry table. A 3 MB image is 25 million
/// iterations of this loop, which is a few tens of milliseconds against a
/// generation measured in seconds — not worth a table to read.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Adler-32: two running sums modulo the largest prime below 2¹⁶.
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 is the most bytes that can be summed before `b` could overflow a
    // u32, so the modulo is taken once per run of them rather than per byte.
    for run in data.chunks(5552) {
        for &x in run {
            a += x as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checksums_give_their_published_check_values() {
        // The check value every CRC-32 catalogue lists for this input.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // Wikipedia's worked example of Adler-32.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn adler_agrees_with_itself_across_the_run_boundary() {
        // The deferred modulo is the one place this could go wrong: a run of
        // 0xff bytes is the largest `b` can grow per byte.
        let data = vec![0xffu8; 20_000];
        let (mut a, mut b) = (1u64, 0u64);
        for &x in &data {
            a = (a + x as u64) % 65521;
            b = (b + a) % 65521;
        }
        assert_eq!(adler32(&data), ((b << 16) | a) as u32);
    }

    /// Undo [`zlib_stored`], checking every length and checksum on the way.
    fn inflate_stored(z: &[u8]) -> Vec<u8> {
        assert_eq!(&z[..2], &[0x78, 0x01]);
        assert_eq!(u16::from_be_bytes([z[0], z[1]]) % 31, 0, "zlib header check bits");
        let mut out = Vec::new();
        let mut i = 2;
        loop {
            let last = z[i] & 1 == 1;
            assert_eq!(z[i] >> 1, 0, "block type must be stored");
            let len = u16::from_le_bytes([z[i + 1], z[i + 2]]);
            let nlen = u16::from_le_bytes([z[i + 3], z[i + 4]]);
            assert_eq!(len, !nlen);
            out.extend_from_slice(&z[i + 5..i + 5 + len as usize]);
            i += 5 + len as usize;
            if last {
                break;
            }
        }
        assert_eq!(u32::from_be_bytes(z[i..i + 4].try_into().unwrap()), adler32(&out));
        assert_eq!(i + 4, z.len(), "nothing may follow the checksum");
        out
    }

    #[test]
    fn stored_deflate_round_trips_across_block_boundaries() {
        for n in [0, 1, 65535, 65536, 200_000] {
            let data: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
            assert_eq!(inflate_stored(&zlib_stored(&data)), data, "{n} bytes");
        }
    }

    #[test]
    fn a_png_is_its_chunks_with_valid_crcs_and_its_pixels_inside() {
        let (w, h) = (3, 2);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| i as u8 * 10).collect();
        let png = Image { width: w, height: h, rgb: rgb.clone() }.png();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");

        let mut i = 8;
        let mut kinds = Vec::new();
        let mut idat = Vec::new();
        while i < png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            let body = &png[i + 4..i + 8 + len];
            let crc = u32::from_be_bytes(png[i + 8 + len..i + 12 + len].try_into().unwrap());
            assert_eq!(crc32(body), crc);
            let kind = std::str::from_utf8(&body[..4]).unwrap().to_string();
            if kind == "IHDR" {
                assert_eq!(&body[4..12], &[0, 0, 0, 3, 0, 0, 0, 2]);
            }
            if kind == "IDAT" {
                idat.extend_from_slice(&body[4..]);
            }
            kinds.push(kind);
            i += 12 + len;
        }
        assert_eq!(kinds, ["IHDR", "IDAT", "IEND"]);

        let raw = inflate_stored(&idat);
        let mut expect = Vec::new();
        for y in 0..h {
            expect.push(0);
            expect.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(raw, expect);
    }

    fn sdxl() -> Defaults {
        Defaults { width: 1024, height: 1024, steps: 30, guidance: 5.0, multiple: 8, takes_guidance: true, takes_negative: true, takes_loras: false, edits: true }
    }

    /// An edit with no size is its picture's shape at the model's own
    /// number of pixels, and runs the last `strength` of its steps.
    #[test]
    fn an_edit_is_sized_from_its_picture_and_runs_part_of_its_steps() {
        assert_eq!(edit_size((512, 512), &sdxl()), (1024, 1024));
        assert_eq!(edit_size((1000, 500), &sdxl()), (1448, 728));
        assert_eq!(edit_size((3000, 4000), &sdxl()), (888, 1184));
        assert_eq!(edit_size((10_000, 10), &sdxl()).0, MAX_SIDE, "held to the largest side there is");

        assert_eq!((edit_steps(20, 0.2), edit_steps(30, 0.75), edit_steps(30, 1.0), edit_steps(4, 0.01)), (4, 23, 30, 1));
    }

    #[test]
    fn an_edit_is_checked_before_it_is_run() {
        let picture = |w: usize, h: usize| Image { width: w, height: h, rgb: vec![0; w * h * 3] };
        let edit = |mask: Option<Mask>, strength: Option<f32>| ImageRequest {
            edit: Some(Edit { image: picture(640, 480), mask, strength }),
            ..ImageRequest::new("a cat")
        };
        let whole = |w: usize, h: usize, v: u8| Mask { width: w, height: h, repaint: vec![v; w * h] };

        // Three quarters of the way without a mask, and all of it with one.
        let r = edit(None, None).resolved(&sdxl()).unwrap();
        assert_eq!((r.width, r.height, r.strength, r.masked), (1184, 888, Some(STRENGTH), false));
        let r = edit(Some(whole(640, 480, 255)), None).resolved(&sdxl()).unwrap();
        assert_eq!((r.strength, r.masked), (Some(1.0), true));
        // A size that was asked for is the size.
        let sized = ImageRequest { width: Some(512), height: Some(512), ..edit(None, Some(0.3)) }.resolved(&sdxl()).unwrap();
        assert_eq!((sized.width, sized.height, sized.strength), (512, 512, Some(0.3)));
        // An image that is not an edit says so by having no strength.
        assert_eq!(ImageRequest::new("a cat").resolved(&sdxl()).unwrap().strength, None);

        for (request, wrong) in [
            (edit(None, Some(0.0)), "strength"),
            (edit(None, Some(1.5)), "strength"),
            (edit(Some(whole(64, 64, 255)), None), "a mask is its picture's size"),
            (edit(Some(whole(640, 480, 0)), None), "nothing to draw"),
            (ImageRequest { edit: Some(Edit { image: picture(0, 0), mask: None, strength: None }), width: Some(512), height: Some(512), ..ImageRequest::new("a cat") }, "empty"),
        ] {
            let said = request.resolved(&sdxl()).unwrap_err().to_string();
            assert!(said.contains(wrong), "{said}");
        }
        let cannot = Defaults { edits: false, ..sdxl() };
        assert!(edit(None, None).resolved(&cannot).unwrap_err().to_string().contains("not from a picture"));
    }

    /// A LoRA is refused by a model that takes none, and checked by one that
    /// does: its name, its scale, how many, and each once. A CLI name keeps
    /// its colons and gives up only a last part that is a number.
    #[test]
    fn loras_are_checked_against_the_model() {
        let lora = |name: &str, scale: f64| Lora { name: name.into(), scale };
        let mut req = ImageRequest::new("a fox");
        req.loras = vec![lora("o/r:l.safetensors", 1.0)];
        let e = req.resolved(&sdxl()).unwrap_err().to_string();
        assert!(e.contains("takes no LoRAs"), "{e}");
        let takes = Defaults { takes_loras: true, ..sdxl() };
        assert_eq!(req.resolved(&takes).unwrap().loras, req.loras);
        for (loras, why) in [
            (vec![lora("a/b", f64::NAN)], "between -10 and 10"),
            (vec![lora(" ", 1.0)], "no name"),
            (vec![lora("a/b", 1.0), lora("a/b", 0.5)], "twice"),
            ((0..=MAX_LORAS).map(|i| lora(&format!("a/b{i}"), 1.0)).collect(), "at most"),
        ] {
            req.loras = loras;
            let e = req.resolved(&takes).unwrap_err().to_string();
            assert!(e.contains(why), "{e}");
        }
        assert_eq!(Lora::parse("o/r:l.safetensors:0.8"), lora("o/r:l.safetensors", 0.8));
        assert_eq!(Lora::parse("o/r:l.safetensors"), lora("o/r:l.safetensors", 1.0));
        assert_eq!(Lora::parse("o/r"), lora("o/r", 1.0));
        assert_eq!(Lora::parse("./style.safetensors:-0.5"), lora("./style.safetensors", -0.5));
    }

    #[test]
    fn a_request_takes_the_models_defaults_for_what_it_leaves_out() {
        let r = ImageRequest { seed: Some(3), ..ImageRequest::new("a cat") }.resolved(&sdxl()).unwrap();
        assert_eq!((r.width, r.height, r.steps, r.guidance, r.seed), (1024, 1024, 30, 5.0, 3));
        assert_eq!(r.negative_prompt, None);
    }

    #[test]
    fn a_request_the_model_cannot_draw_is_refused_with_the_nearest_it_can() {
        let bad = |f: fn(&mut ImageRequest)| {
            let mut r = ImageRequest::new("a cat");
            f(&mut r);
            r.resolved(&sdxl()).unwrap_err().to_string()
        };
        assert!(bad(|r| r.width = Some(1020)).contains("try 1016"), "{}", bad(|r| r.width = Some(1020)));
        assert!(bad(|r| r.height = Some(4096)).contains("between"));
        assert!(bad(|r| r.steps = Some(0)).contains("steps"));
        assert!(bad(|r| r.guidance = Some(f32::NAN)).contains("guidance"));
        assert!(bad(|r| r.prompt = "  ".into()).contains("empty"));
    }

    #[test]
    fn a_model_without_guidance_refuses_a_guidance_scale_and_a_negative_prompt() {
        let schnell = Defaults { width: 1024, height: 1024, steps: 4, guidance: 0.0, multiple: 16, takes_guidance: false, takes_negative: false, takes_loras: false, edits: false };
        let ask = |g: Option<f32>, n: Option<&str>| {
            ImageRequest { guidance: g, negative_prompt: n.map(str::to_string), ..ImageRequest::new("a cat") }.resolved(&schnell)
        };
        assert!(ask(None, None).is_ok());
        assert!(ask(Some(0.0), Some("")).is_ok(), "zero and empty are the same as leaving them out");
        assert!(ask(Some(3.5), None).unwrap_err().to_string().contains("guidance_scale"));
        assert!(ask(None, Some("blurry")).unwrap_err().to_string().contains("negative prompt"));
    }

    /// FLUX.1-dev: guided by a number it reads, in one pass, so with
    /// nothing for a negative prompt to do.
    #[test]
    fn a_model_that_reads_its_guidance_takes_a_scale_and_refuses_a_negative_prompt() {
        let dev = Defaults { width: 1024, height: 1024, steps: 28, guidance: 3.5, multiple: 16, takes_guidance: true, takes_negative: false, takes_loras: true, edits: false };
        let ask = |g: Option<f32>, n: Option<&str>| {
            ImageRequest { guidance: g, negative_prompt: n.map(str::to_string), ..ImageRequest::new("a cat") }.resolved(&dev)
        };
        assert_eq!(ask(None, None).unwrap().guidance, 3.5);
        assert_eq!(ask(Some(1.0), Some("")).unwrap().guidance, 1.0);
        let why = ask(Some(2.0), Some("blurry")).unwrap_err().to_string();
        assert!(why.contains("negative prompt") && !why.contains("without guidance"), "{why}");
    }

    #[test]
    fn an_empty_negative_prompt_is_the_same_as_none() {
        let r = ImageRequest { negative_prompt: Some(String::new()), ..ImageRequest::new("a cat") };
        assert_eq!(r.resolved(&sdxl()).unwrap().negative_prompt, None);
    }
}
