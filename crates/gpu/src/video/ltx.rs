//! LTX-2.5 as a [`Director`]: what `examples/ltx.rs` does, for the server.
//!
//! The pipeline is the example's, phase for phase (`docs/video-plan.md`,
//! steps 5 and 6): the text path encodes the prompt and is dropped; the DiT
//! runs stage 1 at half size, the upsampler doubles the latent, the DiT
//! refines it at full size, and is dropped; the decoders make the frames and
//! the sound.
//!
//! # Nothing is kept between generations
//!
//! The plan was to keep the DiT resident and reload the text path for each
//! request. Measured on an M5 Pro, from the q8 caches, the text path loads in
//! 8.9 s and the DiT in 3.5 s, against 75 s for a 768×512 clip all told. So
//! [`Ltx`] holds paths and nothing else, and loads each phase when it gets
//! to it. Kept, the DiT's 20 GB would sit in memory between requests and buy
//! 3.5 s of each; and it could not stay through the next text phase anyway,
//! because the two together are 36 GB before either has done any work.
//!
//! What admission charges is therefore not what is held but what a
//! generation will need at its largest: [`Director::weight_bytes`] is the
//! peak of the largest clip [`Defaults::max_volume`] allows.

use super::ltx_dit::{video_tokens, Dit, Shape};
use super::ltx_sample::{dev_sigmas, guided, one_stage, refine, Guide, Latents, AUDIO_GUIDE, NEGATIVE_PROMPT, STAGE_1, STAGE_2, VIDEO_GUIDE};
use super::ltx_text::{Contexts, TextEncoder, DEV_FILE, DISTILLED_LORA, DIT_FILE, TEXT_FILE};
use super::{ltx_audio, ltx_duration, ltx_upsample, ltx_vae};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use kvad::image::Image;
use kvad::video::{Defaults, Director, Filmed, Step, VideoRequest};
use kvad::weights::{fetch_file, Watcher};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The pipelines this module implements, by the names
/// [`kvad::hub::pipeline`] gives them.
pub const PIPELINES: [&str; 1] = [kvad::video::LTX_PIPELINE];

/// Every file a generation reads, in the order it reads them.
const FILES: [&str; 5] = [TEXT_FILE, DIT_FILE, ltx_upsample::FILE, ltx_vae::FILE, ltx_audio::FILE];

/// The largest clip measured: 1536×1024 × 121 frames. Nothing bigger has
/// been run, and on this hardware the price of finding out the hard way is
/// a reboot (a Metal allocation past the machine's memory takes the whole
/// Mac down), so nothing bigger is offered.
const MEASURED_VOLUME: usize = 1536 * 1024 * 121;

/// A generation's peak memory footprint, `FIXED + PER_VOXEL × w·h·frames`.
///
/// A line through the two sizes measured at q8 on an M5 Pro: 768×512 × 121
/// at 27.0 GB, and 1536×1024 × 121 at 35.3 GB, the higher of the two peaks
/// measured there (#88; #89's was 33.0, and a peak moves by a GB or two
/// between runs). The peak is stage 2's: the DiT's 20.2 GB and the
/// upsampler, plus activations that grow with the tokens. The text phase
/// and the decode both peak lower.
const PEAK_FIXED: f64 = 24.2e9;
const PEAK_PER_VOXEL: f64 = 58.2;

/// What a clip of `volume` pixels × frames will take at its peak.
pub fn peak_bytes(volume: usize) -> u64 {
    (PEAK_FIXED + PEAK_PER_VOXEL * volume as f64) as u64
}

/// The largest volume whose peak fits in `memory` bytes, and never more than
/// was measured.
fn max_volume(memory: u64) -> usize {
    let fits = ((memory as f64 - PEAK_FIXED) / PEAK_PER_VOXEL).max(0.0) as usize;
    fits.min(MEASURED_VOLUME)
}

/// Whether `repo` is LTX-2.5, asking the disk and never the Hub.
///
/// Lightricks' own repo is LTX-2.5 whether it is here yet or not, so that a
/// load of it can fetch the files it needs, and only those: the repo is
/// 71 GB, and a generation reads about 66 GB of it.
pub fn is_pipeline(repo: &str) -> bool {
    if repo.eq_ignore_ascii_case(super::LTX_REPO) {
        return true;
    }
    // By the file `kvad::hub::pipeline` knows it by, asked directly: the
    // listing would size every model in the cache to answer.
    crate::image::local_file(repo, kvad::video::LTX_DENOISER).is_some()
}

/// Whether `repo`'s guided pipeline can run: its dev DiT and distilled LoRA
/// on this machine, asked of the disk. `Err` says how to fetch them.
pub fn guided_ready(repo: &str) -> Result<(), String> {
    let missing: Vec<&str> = kvad::video::LTX_DEV_FILES.iter().copied().filter(|f| crate::image::local_file(repo, f).is_none()).collect();
    match missing.is_empty() {
        true => Ok(()),
        false => Err(format!(
            "guidance runs {repo}'s dev model, whose files are not on this machine ({}); \
             `kvad pull {repo} --dev` fetches them, 51 GB",
            missing.join(", ")
        )),
    }
}

/// What loading `repo` will need admitted, when every file it reads is on
/// this machine; `None` otherwise. See the module docs for why this is a
/// generation's peak and not the weights.
pub fn weight_bytes(repo: &str, quant: Option<GgmlDType>) -> Option<u64> {
    if quant != Some(GgmlDType::Q8_0) || !is_pipeline(repo) {
        return None;
    }
    FILES.iter().all(|f| crate::image::local_file(repo, f).is_some()).then(|| peak_bytes(volume_for_this_machine()))
}

fn volume_for_this_machine() -> usize {
    match kvad::machine::usable_memory_cached() {
        Some(m) => max_volume(m),
        // A machine that cannot say gets what was measured, as admission
        // gives it everything when it cannot say.
        None => MEASURED_VOLUME,
    }
}

/// LTX-2.5's distilled model in two stages, ready to make a clip.
pub struct Ltx {
    repo: String,
    paths: [PathBuf; 5],
    /// The duration head, which chooses a clip's length when a request
    /// does not say. Optional: 4 MB fetched at load, and without it a clip
    /// is [`Defaults::frames`] long, as before there was one.
    head: Option<PathBuf>,
    quant: GgmlDType,
    device: Device,
    defaults: Defaults,
    params: usize,
}

impl Ltx {
    /// Find the files, fetching what is missing, and make sure the q8 caches
    /// exist, so that the first generation does not spend two minutes
    /// quantising.
    pub fn load(repo: &str, quant: Option<GgmlDType>, progress: &mut dyn FnMut(&str), watch: &Watcher) -> Res<Self> {
        // bf16 is 42 GB of DiT on its own, and q4 has never been run.
        let quant = match quant {
            Some(GgmlDType::Q8_0) => GgmlDType::Q8_0,
            _ => return Err("LTX-2.5 runs at gpu-q8 only here: its DiT is 42 GB in bf16, and no other quantisation has been measured".into()),
        };
        let device = crate::model::pick_device(None)?;
        if !device.is_metal() {
            return Err("LTX-2.5 runs on Metal only here: its attention, projections and convolutions are Metal kernels".into());
        }
        let mut paths = Vec::with_capacity(FILES.len());
        for f in FILES {
            progress(&format!("finding {f}"));
            paths.push(fetch_file(repo, f, watch)?);
        }
        let paths: [PathBuf; 5] = paths.try_into().map_err(|_| "five files")?;
        let head = match fetch_file(repo, ltx_duration::FILE, watch) {
            Ok(p) => Some(p),
            Err(e) => {
                progress(&format!("no duration head, so a clip is 121 frames unless asked otherwise: {e}"));
                None
            }
        };

        let tag = crate::qcache::tag(quant);
        let cached = |component: &str| kvad::qcache::path_for_tag(&format!("{}/{component}", super::LTX_REPO), &tag).is_file();
        if kvad::qcache::enabled() && !cached("text_encoder") {
            progress("quantising the text path to q8, once");
            drop(TextEncoder::load(&paths[0], &paths[1], &device, DType::BF16, Some(quant), progress)?);
            device.synchronize()?;
        }
        if kvad::qcache::enabled() && !cached("transformer") {
            progress("quantising the DiT to q8, once");
            drop(Dit::load(&paths[1], &device, DType::BF16, None, Some(quant), progress)?);
            device.synchronize()?;
        }

        let params = [
            header_params(&paths[0], &["vision_model.", "multi_modal_projector.", "audio_projector."])?,
            header_params(&paths[1], &[])?,
            header_params(&paths[2], &[])?,
            header_params(&paths[3], &["encoder."])?,
            header_params(&paths[4], &["audio_vae.encoder"])?,
            head.as_deref().map(|h| header_params(h, &[])).transpose()?.unwrap_or(0),
        ]
        .iter()
        .sum();
        let defaults = Defaults {
            width: 768,
            height: 512,
            frames: 121,
            fps: 24,
            // Stage 1 runs at half size, and the DiT's patches are 32 pixels.
            multiple: 64,
            frame_step: 8,
            max_frames: 121,
            max_volume: volume_for_this_machine(),
            image: true,
            duration: head.is_some(),
            // The dev model, whose files `kvad pull … --dev` fetches.
            guided: Some(kvad::video::Guided { steps: super::ltx_sample::DEV_STEPS, max_steps: 60, guidance: VIDEO_GUIDE.cfg }),
        };
        Ok(Ltx { repo: repo.to_string(), paths, head, quant, device, defaults, params })
    }
}

/// Parameters in a safetensors file, from its header, leaving out tensors
/// under `skip`: the parts of a file a generation never reads.
fn header_params(path: &Path, skip: &[&str]) -> Res<usize> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let header: kvad::serde_json::Value = kvad::serde_json::from_slice(&header)?;
    let tensors = header.as_object().ok_or("a safetensors header that is not an object")?;
    Ok(tensors
        .iter()
        .filter(|(name, _)| *name != "__metadata__" && !skip.iter().any(|s| name.contains(s)))
        .filter_map(|(_, t)| Some(t["shape"].as_array()?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).product::<usize>()))
        .sum())
}

/// Seconds each part of a generation is expected to take, for
/// [`Step::progress`].
///
/// Measured on an M5 Pro at q8, and only their proportions matter: a slower
/// machine is slower at all of them. A DiT step's time is linear in its
/// tokens plus attention's quadratic part, fitted through 2.2 s at 1536
/// tokens, 9.6 s at 6144 and 52 s at 24576.
struct Plan {
    /// Encoding the picture a video starts from, when there is one.
    picture: f64,
    /// Stage 1's steps, and the DiT calls each makes: 8 and 1 for the
    /// distilled model, the request's steps and 4 for the dev model.
    steps_1: usize,
    text: f64,
    load: f64,
    /// Loading stage 2's DiT: only the dev model has a second one.
    load_2: f64,
    stage_1: f64,
    upsample: f64,
    stage_2: f64,
    decode: f64,
}

impl Plan {
    fn new(first: Shape, full: Shape, picture: bool, guided: Option<usize>) -> Self {
        let step = |tokens: usize| tokens as f64 * (1.386e-3 + 2.86e-8 * tokens as f64);
        let at_768 = (768 * 512 * 121) as f64;
        let volume = (full.width * full.height * full.frames) as f64;
        let (steps_1, calls) = match guided {
            Some(n) => (n, 4.0),
            None => (STAGE_1.len() - 1, 1.0),
        };
        Plan {
            picture: if picture { 1.0 } else { 0.0 },
            steps_1,
            // The dev model encodes a negative prompt too, and loads a
            // second DiT for stage 2.
            text: if guided.is_some() { 10.0 } else { 9.4 },
            load: 3.5,
            load_2: if guided.is_some() { 3.5 } else { 0.0 },
            stage_1: step(first.video_tokens()) * calls,
            upsample: 2.1 * full.video_tokens() as f64 / 6144.0,
            stage_2: step(full.video_tokens()),
            decode: 13.0 * volume / at_768,
        }
    }

    fn total(&self) -> f64 {
        self.picture + self.text + self.load + self.load_2 + self.stage_1 * self.steps_1 as f64 + self.upsample + self.stage_2 * (STAGE_2.len() - 1) as f64 + self.decode
    }
}

impl Ltx {
    /// [`Director::film`], before the synchronise that frees what it held.
    fn run(&self, req: &VideoRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Filmed> {
        let r = req.resolved(&self.defaults)?;
        let mut r = r;
        let fps = r.fps as f64;
        // Until the head has chosen a length, the shapes and the plan are
        // at the most it may choose; the size is all the picture needs.
        let shapes = |frames: usize| -> Res<(Shape, Shape)> {
            Ok((Shape::new(r.width / 2, r.height / 2, frames, fps)?, Shape::new(r.width, r.height, frames, fps)?))
        };
        let (mut first, mut full) = shapes(r.frames)?;
        let (device, dtype, quant) = (&self.device, DType::BF16, Some(self.quant));
        let guided_steps = r.guided.as_ref().map(|g| g.steps);
        let mut plan = Plan::new(first, full, req.image.is_some(), guided_steps);
        let total = std::cell::Cell::new(plan.total());
        let chosen = std::cell::Cell::new(None);
        let started = Instant::now();
        // What the plan says is done by the time `phase` reaches step `done`.
        let mut report = |phase: &'static str, done: usize, of: usize, before: f64, preview: Option<Image>| -> Res<()> {
            let progress = (before / total.get()).clamp(0.0, 1.0) as f32;
            let elapsed = started.elapsed().as_secs_f64();
            match on_step(Step { phase, frames: chosen.get(), done, total: of, progress, elapsed, preview }) {
                true => Ok(()),
                false => Err("cancelled".into()),
            }
        };
        let mut quiet = |_: &str| {};
        let s2 = STAGE_2.len() - 1;
        // The dev model's files, when the request wants guidance: asked again
        // here, since they could have gone since the request was checked.
        let dev = match &r.guided {
            Some(_) => {
                guided_ready(&self.repo)?;
                let find = |f: &str| crate::image::local_file(&self.repo, f).ok_or_else(|| format!("{f} is not on this machine"));
                Some((find(DEV_FILE)?, find(DISTILLED_LORA)?))
            }
            None => None,
        };

        // 0. The picture, when the video starts from one: encoded at each
        // stage's size, from the picture itself each time, as the reference
        // does. The encoder is 0.6 GB and gone before the text path loads.
        let t = Instant::now();
        let stills = match &req.image {
            None => None,
            Some(p) => {
                report("picture", 0, 1, 0.0, None)?;
                let enc = ltx_vae::ImageEncoder::load(&self.paths[3], device, dtype)?;
                let at = |s: Shape| -> Res<Tensor> {
                    let z = enc.encode(&ltx_vae::picture(&p.rgb, p.width, p.height, s.width, s.height)?.to_device(device)?)?;
                    Ok(video_tokens(&z)?.to_dtype(dtype)?)
                };
                let stills = (at(first)?, at(full)?);
                drop(enc);
                device.synchronize()?;
                Some(stills)
            }
        };
        let (still_1, still_2) = match &stills {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        };

        // 1. The prompt.
        report("text", 0, 1, plan.picture, None)?;
        // The connectors are read from the distilled DiT's file whichever
        // model runs: the dev file's are the same, byte for byte (258 of 258
        // tensors, checked), so the text path has one cache.
        let (ctx, neg) = {
            let enc = TextEncoder::load(&self.paths[0], &self.paths[1], device, dtype, quant, &mut quiet)?;
            let neg = match &r.guided {
                Some(g) => Some(enc.encode(g.negative_prompt.as_deref().unwrap_or(NEGATIVE_PROMPT))?),
                None => None,
            };
            (enc.encode(&r.prompt)?, neg)
        };
        // The length, when the model chooses it: the head reads the two
        // contexts just made, and chooses within the reference's 1–20 s and
        // under what this size allows here, which `r.frames` holds.
        if let (true, Some(path)) = (r.chosen, &self.head) {
            let seconds = ltx_duration::DurationHead::load(path, device)?.seconds(&ctx.video, &ctx.audio)?;
            let most = ((ltx_duration::MAX_SECONDS * fps).round() as usize).min(r.frames);
            r.frames = ltx_duration::frames_for(seconds, fps, (ltx_duration::MIN_SECONDS * fps).round() as usize, most);
            (first, full) = shapes(r.frames)?;
            plan = Plan::new(first, full, req.image.is_some(), guided_steps);
            total.set(plan.total());
            chosen.set(Some(r.frames));
        }
        // candle's Metal pool lets a dropped tensor's buffer go only at the
        // next synchronise; without this the text path's 13 GB would sit
        // beside the DiT's 20.
        device.synchronize()?;
        let encode_secs = t.elapsed().as_secs_f64();

        // 2. The latents.
        let t = Instant::now();
        let mut done = plan.picture + plan.text;
        let s1 = plan.steps_1;
        report("stage 1", 0, s1, done, None)?;
        let latents = {
            let dit = match &dev {
                Some((d, _)) => Dit::load_as(d, None, "transformer-dev", device, dtype, None, quant, &mut quiet)?,
                None => Dit::load(&self.paths[1], device, dtype, None, quant, &mut quiet)?,
            };
            done += plan.load;
            let ctx = Contexts { video: ctx.video.clone(), audio: ctx.audio.clone() };
            let l = {
                let mut step = |i: usize, _sigma: f32, clean: &Tensor| -> Res<()> {
                    let look = preview(clean, first)?;
                    report("stage 1", i + 1, s1, done + plan.stage_1 * (i + 1) as f64, Some(look))
                };
                match (&r.guided, &neg) {
                    // The dev model: the request's steps, guided as its
                    // reference's pipelines guide it, the video's CFG as asked.
                    (Some(g), Some(neg)) => {
                        let video = Guide { cfg: g.guidance, ..VIDEO_GUIDE };
                        guided(&dit, &ctx, neg, &dit.grid(first)?, r.seed, &dev_sigmas(g.steps), (video, AUDIO_GUIDE), still_1, &mut step)?
                    }
                    _ => one_stage(&dit, &ctx, &dit.grid(first)?, r.seed, still_1, &mut step)?,
                }
            };
            done += plan.stage_1 * s1 as f64;
            report("upsample", 0, 1, done, None)?;
            let video = {
                let up = ltx_upsample::Upsampler::load(&self.paths[2], &self.paths[3], device, DType::F32)?;
                up.forward(&l.video)?
            };
            device.synchronize()?;
            done += plan.upsample;
            report("stage 2", 0, s2, done, None)?;
            // The dev model's second stage is its DiT with the distilled
            // LoRA fused in, cached on its own; the first goes before it
            // loads, so that the two are never resident together.
            let fused;
            let second = match &dev {
                Some((d, lora)) => {
                    drop(dit);
                    device.synchronize()?;
                    fused = Dit::load_as(d, Some((lora, 1.0)), "transformer-dev-distilled", device, dtype, None, quant, &mut quiet)?;
                    &fused
                }
                None => &dit,
            };
            done += plan.load_2;
            let mut step = |i: usize, _sigma: f32, clean: &Tensor| -> Res<()> {
                let look = preview(clean, full)?;
                report("stage 2", i + 1, s2, done + plan.stage_2 * (i + 1) as f64, Some(look))
            };
            let stage_1_audio = l.audio.clone();
            let l = refine(second, &ctx, &second.grid(full)?, &Latents { video, audio: l.audio }, r.seed, still_2, &mut step)?;
            match dev {
                // The reference keeps stage 1's sound: its stage 2 refines
                // the video only.
                Some(_) => Latents { video: l.video, audio: stage_1_audio },
                None => l,
            }
        };
        drop(neg);
        drop(ctx);
        // And the DiT's 20 GB, before the decoders need room for frames.
        device.synchronize()?;
        let denoise_secs = t.elapsed().as_secs_f64();

        // 3. Pictures and sound.
        let t = Instant::now();
        report("decode", 0, 1, plan.total() - plan.decode, None)?;
        let frames = ltx_vae::VideoDecoder::load(&self.paths[3], device, dtype)?.decode(&latents.video)?.to_device(&Device::Cpu)?;
        let audio = match r.audio {
            true => Some(ltx_audio::AudioPath::load(&self.paths[4], device)?.decode(&latents.audio)?),
            false => None,
        };
        drop(latents);
        device.synchronize()?;
        let video = ltx_vae::to_video(&frames, r.fps)?;
        let decode_secs = t.elapsed().as_secs_f64();
        report("decode", 1, 1, total.get(), None)?;
        Ok(Filmed { video, audio, request: r, encode_secs, denoise_secs, decode_secs })
    }
}

impl Director for Ltx {
    fn film(&mut self, req: &VideoRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Filmed> {
        let filmed = self.run(req, on_step);
        // Whatever `run` held is dropped by now, and candle's Metal pool
        // lets it go only here. A generation that was cancelled, or failed,
        // mid-stage returned before its own synchronise, and would leave the
        // DiT's 20 GB in the pool for the next one to load beside.
        self.device.synchronize()?;
        filmed
    }

    fn defaults(&self) -> Defaults {
        self.defaults
    }

    fn summary(&self) -> String {
        format!(
            "LTX-2.5 distilled ({}), {:.1} B parameters: Gemma 4 text path, 48-block DiT in two stages, video and audio decoders",
            self.repo,
            self.params as f64 / 1e9
        )
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        peak_bytes(self.defaults.max_volume) as usize
    }

    fn backend(&self) -> String {
        "metal q8".into()
    }
}

/// LTX-2.5's 128 latent channels as colour, roughly, and the bias after
/// them: fitted by least squares from the final video latents of three
/// 768×512 × 121 clips (a fox in snow, a city street at night in the rain, a
/// dog on a beach at sunset; seeds 1 to 3) to their decoded frames, each
/// averaged over the 32×32 pixels and 8 frames a latent cell covers.
///
/// Fitted on two clips and scored on the third, it explains 84–96% of the
/// variance in each of red, green and blue, 22–26 dB from the pooled frames;
/// on all three, 98%. Ridge regression did no better on the clip left out.
const PREVIEW: [[f32; 3]; 128] = [
    [0.0049, -0.0039, 0.0008],
    [-0.0026, -0.0020, -0.0022],
    [0.0021, 0.0038, 0.0047],
    [0.0050, 0.0012, -0.0004],
    [-0.0001, 0.0009, 0.0032],
    [0.0033, 0.0012, 0.0033],
    [-0.0066, -0.0085, -0.0110],
    [-0.0097, 0.0112, 0.0155],
    [-0.0037, 0.0029, 0.0030],
    [-0.0010, -0.0020, -0.0083],
    [-0.0002, 0.0014, 0.0016],
    [0.0014, 0.0036, 0.0042],
    [0.0110, 0.0092, 0.0029],
    [0.0019, -0.0047, -0.0077],
    [0.0010, 0.0029, 0.0033],
    [-0.0030, -0.0037, -0.0030],
    [0.0022, 0.0028, 0.0062],
    [-0.0022, -0.0012, -0.0010],
    [-0.0018, -0.0024, -0.0002],
    [0.0011, -0.0007, 0.0024],
    [0.0065, 0.0013, -0.0022],
    [-0.0015, -0.0025, -0.0042],
    [0.0038, 0.0016, 0.0046],
    [0.0029, -0.0014, 0.0058],
    [0.0064, 0.0053, 0.0081],
    [-0.0016, -0.0022, 0.0009],
    [-0.0079, -0.0046, -0.0053],
    [-0.0053, -0.0014, 0.0001],
    [0.0008, -0.0010, -0.0017],
    [-0.0006, -0.0003, -0.0001],
    [-0.0037, 0.0010, 0.0019],
    [-0.0014, -0.0014, -0.0009],
    [0.0037, 0.0020, 0.0024],
    [-0.0007, -0.0016, -0.0017],
    [0.0044, 0.0030, 0.0047],
    [-0.0016, -0.0000, 0.0013],
    [-0.0016, -0.0002, 0.0012],
    [-0.0036, -0.0040, -0.0038],
    [0.0019, -0.0011, -0.0020],
    [0.0015, -0.0014, -0.0021],
    [0.0018, 0.0014, 0.0008],
    [-0.0073, -0.0032, 0.0002],
    [0.0004, 0.0012, 0.0030],
    [-0.0009, 0.0005, -0.0014],
    [0.0042, 0.0030, 0.0028],
    [0.0112, 0.0093, 0.0122],
    [-0.0039, -0.0041, -0.0051],
    [-0.0013, -0.0008, -0.0006],
    [-0.0001, -0.0010, 0.0008],
    [-0.0009, -0.0014, -0.0015],
    [-0.0004, -0.0020, 0.0015],
    [-0.0007, 0.0017, 0.0010],
    [0.0043, 0.0029, 0.0032],
    [0.0025, 0.0027, 0.0029],
    [-0.0015, -0.0006, -0.0023],
    [0.0060, 0.0072, 0.0117],
    [-0.0007, -0.0008, -0.0020],
    [0.0290, 0.0259, 0.0173],
    [0.0033, 0.0028, 0.0033],
    [-0.0031, -0.0024, -0.0019],
    [0.0027, 0.0018, -0.0001],
    [0.0002, 0.0006, 0.0007],
    [-0.0008, 0.0028, 0.0036],
    [0.0229, 0.0146, 0.0136],
    [-0.0028, 0.0062, 0.0091],
    [-0.0076, -0.0042, -0.0094],
    [-0.0063, -0.0021, -0.0001],
    [-0.0033, -0.0030, -0.0010],
    [0.0015, -0.0008, -0.0027],
    [0.0013, 0.0010, 0.0018],
    [0.0010, 0.0003, -0.0016],
    [0.0027, 0.0023, 0.0014],
    [-0.0142, -0.0071, -0.0064],
    [-0.0007, -0.0006, -0.0009],
    [-0.0037, 0.0000, 0.0011],
    [-0.0361, -0.0437, -0.0396],
    [0.0037, -0.0011, -0.0012],
    [-0.0005, 0.0006, -0.0004],
    [0.0011, -0.0014, -0.0029],
    [0.0021, 0.0009, 0.0037],
    [0.0047, -0.0026, -0.0015],
    [0.0065, 0.0057, 0.0027],
    [0.0059, -0.0022, -0.0030],
    [-0.0041, -0.0060, -0.0050],
    [-0.0023, 0.0002, 0.0014],
    [0.0007, -0.0002, -0.0005],
    [0.0049, 0.0032, 0.0034],
    [0.0054, -0.0010, -0.0031],
    [0.0021, 0.0019, 0.0013],
    [-0.0043, 0.0001, 0.0011],
    [-0.0006, -0.0020, -0.0016],
    [-0.0005, 0.0023, 0.0021],
    [0.0020, -0.0007, 0.0047],
    [0.0024, 0.0011, 0.0026],
    [0.0031, 0.0035, 0.0018],
    [-0.0011, -0.0003, -0.0020],
    [-0.0093, -0.0052, -0.0116],
    [-0.0329, -0.0512, -0.0597],
    [-0.0021, 0.0053, 0.0026],
    [0.0012, 0.0016, 0.0022],
    [-0.0031, 0.0004, 0.0010],
    [0.0095, 0.0016, 0.0015],
    [-0.0009, -0.0023, -0.0001],
    [0.0374, 0.0453, 0.0448],
    [-0.0011, -0.0027, -0.0023],
    [0.0020, 0.0009, 0.0052],
    [-0.0016, -0.0010, -0.0011],
    [0.0020, 0.0064, 0.0057],
    [-0.0133, -0.0136, -0.0081],
    [0.0045, 0.0035, 0.0048],
    [-0.0044, 0.0005, -0.0139],
    [-0.0020, 0.0024, 0.0025],
    [0.0004, 0.0012, 0.0007],
    [-0.0047, 0.0016, -0.0003],
    [-0.0023, -0.0080, -0.0060],
    [0.0089, 0.0065, 0.0058],
    [-0.0390, -0.0399, -0.0397],
    [0.0014, -0.0002, -0.0022],
    [0.0014, 0.0024, 0.0011],
    [0.0002, -0.0031, -0.0048],
    [0.0045, 0.0015, 0.0012],
    [0.0013, -0.0010, -0.0000],
    [0.0002, 0.0011, 0.0019],
    [-0.0043, -0.0032, -0.0040],
    [0.0004, 0.0018, 0.0032],
    [-0.0027, 0.0009, 0.0007],
    [-0.0019, -0.0048, -0.0088],
    [0.0002, -0.0010, -0.0010],
];
const PREVIEW_BIAS: [f32; 3] = [0.3907, 0.3650, 0.3519];

/// The middle latent frame of `clean`, the DiT's prediction of the clean
/// video as tokens `[n, 128]`, mixed straight into colour by [`PREVIEW`]:
/// an image at the latent's size, one pixel a token.
///
/// Reading it back is what synchronises the step, which the progress report
/// needs anyway: the time a step is reported at is the time it finished.
fn preview(clean: &Tensor, shape: Shape) -> Res<Image> {
    let (rows, cols) = shape.grid();
    let middle = shape.latent_frames() / 2;
    let z = clean.narrow(0, middle * rows * cols, rows * cols)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let z = z.to_vec2::<f32>()?;
    let rgb = z
        .iter()
        .flat_map(|t| {
            (0..3).map(move |c| {
                let v = PREVIEW_BIAS[c] + t.iter().zip(&PREVIEW).map(|(x, w)| x * w[c]).sum::<f32>();
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            })
        })
        .collect();
    Ok(Image { width: cols, height: rows, rgb })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The middle latent frame, one pixel a token, row by row: here the
    /// only frame of three with nothing in it, so every pixel is the bias.
    #[test]
    fn a_preview_is_the_middle_frame_mixed_into_colour() {
        let shape = Shape::new(96, 64, 17, 24.0).unwrap();
        let (rows, cols) = shape.grid();
        let n = rows * cols;
        let mut z = vec![10f32; shape.video_tokens() * 128];
        z[n * 128..2 * n * 128].fill(0.0);
        let clean = Tensor::from_vec(z, (shape.video_tokens(), 128), &Device::Cpu).unwrap().to_dtype(DType::BF16).unwrap();
        let look = preview(&clean, shape).unwrap();
        assert_eq!((look.width, look.height), (3, 2));
        let bias = PREVIEW_BIAS.map(|b| (b * 255.0).round() as u8);
        assert!(look.rgb.chunks(3).all(|p| p == bias), "{:?}", look.rgb);
    }

    #[test]
    fn the_peak_model_goes_through_both_measurements() {
        let gb = |v: usize| peak_bytes(v) as f64 / 1e9;
        assert!((gb(768 * 512 * 121) - 27.0).abs() < 0.1, "{}", gb(768 * 512 * 121));
        assert!((gb(1536 * 1024 * 121) - 35.3).abs() < 0.1, "{}", gb(1536 * 1024 * 121));
    }

    #[test]
    fn a_machine_gets_what_fits_and_never_more_than_was_measured() {
        // 48 GiB at three quarters: everything measured.
        assert_eq!(max_volume(38_654_705_664), MEASURED_VOLUME);
        // 32 GiB: 768×512 × 121 does not fit, and something smaller does.
        let small = max_volume(25_769_803_776);
        assert!(small < 768 * 512 * 121 && small > 512 * 320 * 121, "{small}");
        assert_eq!(max_volume(10_000_000_000), 0);
    }

    #[test]
    fn guidance_without_the_dev_files_says_how_to_get_them() {
        let why = guided_ready("nobody/nothing-here").unwrap_err();
        assert!(why.contains("kvad pull nobody/nothing-here --dev"), "{why}");
        assert!(why.contains(kvad::video::LTX_DEV_FILES[0]) && why.contains(kvad::video::LTX_DEV_FILES[1]), "{why}");
    }

    #[test]
    fn a_guided_plan_charges_four_calls_a_step_and_a_second_load() {
        let full = Shape::new(768, 512, 121, 24.0).unwrap();
        let first = Shape::new(384, 256, 121, 24.0).unwrap();
        let (p, g) = (Plan::new(first, full, false, None), Plan::new(first, full, false, Some(30)));
        assert!((g.stage_1 - 4.0 * p.stage_1).abs() < 1e-9 && g.steps_1 == 30 && p.steps_1 == 8);
        assert_eq!((p.load_2, g.load_2), (0.0, 3.5));
        // Measured: 283.8 s for stage 1's thirty steps at 768×512.
        assert!((g.stage_1 * 30.0 - 283.8).abs() / 283.8 < 0.1, "{}", g.stage_1 * 30.0);
    }

    #[test]
    fn the_plans_proportions_are_the_measured_ones() {
        let full = Shape::new(768, 512, 121, 24.0).unwrap();
        let first = Shape::new(384, 256, 121, 24.0).unwrap();
        let p = Plan::new(first, full, false, None);
        // Stage 1's eight steps took 17.8 s and stage 2's three 28.8 s.
        assert!((p.stage_1 * 8.0 - 17.8).abs() < 0.5, "{}", p.stage_1 * 8.0);
        assert!((p.stage_2 * 3.0 - 28.8).abs() < 0.5, "{}", p.stage_2 * 3.0);
        assert!((p.total() - 75.0).abs() < 2.0, "{}", p.total());
        // A picture adds its encoding, and nothing else.
        let q = Plan::new(first, full, true, None);
        assert!((q.total() - p.total() - q.picture).abs() < 1e-9 && q.picture > 0.0);
    }
}
