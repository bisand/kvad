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

use super::ltx_dit::{Dit, Shape};
use super::ltx_sample::{one_stage, refine, Latents, STAGE_1, STAGE_2};
use super::ltx_text::{Contexts, TextEncoder, DIT_FILE, TEXT_FILE};
use super::{ltx_audio, ltx_upsample, ltx_vae};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
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
        };
        Ok(Ltx { repo: repo.to_string(), paths, quant, device, defaults, params })
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
    text: f64,
    load: f64,
    stage_1: f64,
    upsample: f64,
    stage_2: f64,
    decode: f64,
}

impl Plan {
    fn new(first: Shape, full: Shape) -> Self {
        let step = |tokens: usize| tokens as f64 * (1.386e-3 + 2.86e-8 * tokens as f64);
        let at_768 = (768 * 512 * 121) as f64;
        let volume = (full.width * full.height * full.frames) as f64;
        Plan {
            text: 9.4,
            load: 3.5,
            stage_1: step(first.video_tokens()),
            upsample: 2.1 * full.video_tokens() as f64 / 6144.0,
            stage_2: step(full.video_tokens()),
            decode: 13.0 * volume / at_768,
        }
    }

    fn total(&self) -> f64 {
        self.text + self.load + self.stage_1 * (STAGE_1.len() - 1) as f64 + self.upsample + self.stage_2 * (STAGE_2.len() - 1) as f64 + self.decode
    }
}

impl Ltx {
    /// [`Director::film`], before the synchronise that frees what it held.
    fn run(&self, req: &VideoRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Filmed> {
        let r = req.resolved(&self.defaults)?;
        let fps = r.fps as f64;
        let full = Shape::new(r.width, r.height, r.frames, fps)?;
        let first = Shape::new(r.width / 2, r.height / 2, r.frames, fps)?;
        let (device, dtype, quant) = (&self.device, DType::BF16, Some(self.quant));
        let plan = Plan::new(first, full);
        let total = plan.total();
        let started = Instant::now();
        // What the plan says is done by the time `phase` reaches step `done`.
        let mut report = |phase: &'static str, done: usize, of: usize, before: f64| -> Res<()> {
            let progress = (before / total).clamp(0.0, 1.0) as f32;
            match on_step(Step { phase, done, total: of, progress, elapsed: started.elapsed().as_secs_f64() }) {
                true => Ok(()),
                false => Err("cancelled".into()),
            }
        };
        let mut quiet = |_: &str| {};
        let (s1, s2) = (STAGE_1.len() - 1, STAGE_2.len() - 1);

        // 1. The prompt.
        let t = Instant::now();
        report("text", 0, 1, 0.0)?;
        let ctx = {
            let enc = TextEncoder::load(&self.paths[0], &self.paths[1], device, dtype, quant, &mut quiet)?;
            enc.encode(&r.prompt)?
        };
        // candle's Metal pool lets a dropped tensor's buffer go only at the
        // next synchronise; without this the text path's 13 GB would sit
        // beside the DiT's 20.
        device.synchronize()?;
        let encode_secs = t.elapsed().as_secs_f64();

        // 2. The latents.
        let t = Instant::now();
        let mut done = plan.text;
        report("stage 1", 0, s1, done)?;
        let latents = {
            let dit = Dit::load(&self.paths[1], device, dtype, None, quant, &mut quiet)?;
            done += plan.load;
            let ctx = Contexts { video: ctx.video.clone(), audio: ctx.audio.clone() };
            let l = {
                let mut step = |i: usize, _sigma: f32| -> Res<()> {
                    device.synchronize()?;
                    report("stage 1", i + 1, s1, done + plan.stage_1 * (i + 1) as f64)
                };
                one_stage(&dit, &ctx, &dit.grid(first)?, r.seed, &mut step)?
            };
            done += plan.stage_1 * s1 as f64;
            report("upsample", 0, 1, done)?;
            let video = {
                let up = ltx_upsample::Upsampler::load(&self.paths[2], &self.paths[3], device, DType::F32)?;
                up.forward(&l.video)?
            };
            device.synchronize()?;
            done += plan.upsample;
            report("stage 2", 0, s2, done)?;
            let mut step = |i: usize, _sigma: f32| -> Res<()> {
                device.synchronize()?;
                report("stage 2", i + 1, s2, done + plan.stage_2 * (i + 1) as f64)
            };
            refine(&dit, &ctx, &dit.grid(full)?, &Latents { video, audio: l.audio }, r.seed, &mut step)?
        };
        drop(ctx);
        // And the DiT's 20 GB, before the decoders need room for frames.
        device.synchronize()?;
        let denoise_secs = t.elapsed().as_secs_f64();

        // 3. Pictures and sound.
        let t = Instant::now();
        report("decode", 0, 1, plan.total() - plan.decode)?;
        let frames = ltx_vae::VideoDecoder::load(&self.paths[3], device, dtype)?.decode(&latents.video)?.to_device(&Device::Cpu)?;
        let audio = match r.audio {
            true => Some(ltx_audio::AudioPath::load(&self.paths[4], device)?.decode(&latents.audio)?),
            false => None,
        };
        drop(latents);
        device.synchronize()?;
        let video = ltx_vae::to_video(&frames, r.fps)?;
        let decode_secs = t.elapsed().as_secs_f64();
        report("decode", 1, 1, total)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_plans_proportions_are_the_measured_ones() {
        let full = Shape::new(768, 512, 121, 24.0).unwrap();
        let first = Shape::new(384, 256, 121, 24.0).unwrap();
        let p = Plan::new(first, full);
        // Stage 1's eight steps took 17.8 s and stage 2's three 28.8 s.
        assert!((p.stage_1 * 8.0 - 17.8).abs() < 0.5, "{}", p.stage_1 * 8.0);
        assert!((p.stage_2 * 3.0 - 28.8).abs() < 0.5, "{}", p.stage_2 * 3.0);
        assert!((p.total() - 75.0).abs() < 2.0, "{}", p.total());
    }
}
