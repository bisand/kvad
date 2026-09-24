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
        let width = self.width.unwrap_or(d.width);
        let height = self.height.unwrap_or(d.height);
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
        // Kept below 2³² so that it survives a round trip through a browser,
        // where every number is a double and a 64-bit seed would come back
        // as a different one.
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64 % (1 << 32))
                .unwrap_or(0)
        });
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
        Defaults { width: 1024, height: 1024, steps: 30, guidance: 5.0, multiple: 8 }
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
    fn an_empty_negative_prompt_is_the_same_as_none() {
        let r = ImageRequest { negative_prompt: Some(String::new()), ..ImageRequest::new("a cat") };
        assert_eq!(r.resolved(&sdxl()).unwrap().negative_prompt, None);
    }
}
