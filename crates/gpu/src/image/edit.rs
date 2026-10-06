//! An image made from a picture: image-to-image, and inpainting.
//!
//! Drawing starts from noise and walks down the schedule to a picture. An
//! edit starts part of the way down, from a picture somebody gave: the
//! picture is encoded, noised to the level a `strength` names, and the walk
//! goes on from there with the prompt. It is SDEdit (Meng et al., 2021), and
//! what diffusers' image-to-image pipelines do. The model is the one that
//! draws, unchanged; the prompt says what the picture should be, and how
//! much of the old one shows through is how far up it was taken.
//!
//! ```text
//! noise prediction (SDXL, SD 1.5)   x = x₀ + σ·ε
//! flow matching                     x = (1 − σ)·x₀ + σ·ε
//! ```
//!
//! With a mask, the part of the picture to keep is put back after every
//! step, noised to the level the step came down to, with the same ε. So at
//! each step the model sees the kept part exactly as noisy as the part it
//! is drawing, and draws that part to go with it; at the last step the
//! level is 0 and the kept part is the picture's own latent. It is
//! diffusers' inpainting for a model that was not trained for it, the
//! four-channel UNets here, where an inpainting model would be shown the
//! mask as well.
//!
//! The kept part is then put back once more, as pixels: a VAE does not
//! return what it was given, to the bit, and outside the mask nothing was
//! asked to change.

use super::schedule::{Kind, Schedule};
use candle_core::{DType, Device, Tensor};
use kvad::image::{Edit, Image, Mask, Resolved};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `width` × `height` cut from the middle of `image`, in its own pixels:
/// the largest piece of that shape there is.
fn middle(of: (usize, usize), width: usize, height: usize) -> (usize, usize, usize, usize) {
    let (w, h) = of;
    // The shape wanted is wider than the picture's, or it is taller.
    let (cw, ch) = match width * h >= height * w {
        true => (w, (w * height / width).clamp(1, h)),
        false => ((h * width / height).clamp(1, w), h),
    };
    ((w - cw) / 2, (h - ch) / 2, cw, ch)
}

/// The picture as the request's size wants it: its middle, in that shape,
/// made larger or smaller to fit. A picture already that shape is only
/// resized, and one already that size is as it was.
pub(crate) fn fit(image: &Image, width: usize, height: usize) -> Image {
    let (left, top, cw, ch) = middle((image.width, image.height), width, height);
    let mut rgb = Vec::with_capacity(cw * ch * 3);
    for y in top..top + ch {
        rgb.extend_from_slice(&image.rgb[(y * image.width + left) * 3..(y * image.width + left + cw) * 3]);
    }
    super::dataset::resized(&Image { width: cw, height: ch, rgb }, width, height)
}

/// The mask, cut and resized as [`fit`] does its picture.
pub(crate) fn fit_mask(mask: &Mask, width: usize, height: usize) -> Mask {
    // As a grey picture, so that one resize serves both.
    let grey = Image { width: mask.width, height: mask.height, rgb: mask.repaint.iter().flat_map(|&v| [v, v, v]).collect() };
    let fitted = fit(&grey, width, height);
    Mask { width, height, repaint: fitted.rgb.chunks_exact(3).map(|p| p[0]).collect() }
}

/// What an edit carries through the steps.
pub(crate) struct Edited {
    /// The picture's latent, in the denoiser's units and f32, in the shape
    /// the loop keeps its latent in.
    x0: Tensor,
    /// The noise it was started with, and is noised with again where a
    /// mask keeps it.
    eps: Tensor,
    /// Where the picture is kept, 1, and where it is drawn, 0, a number a
    /// latent pixel and broadcast over its channels; with no mask, none.
    keep: Option<Tensor>,
    /// The step the walk starts at.
    pub(crate) first: usize,
    /// The picture and its mask at the request's size, for [`Edited::paste`].
    fitted: Image,
    mask: Option<Mask>,
}

/// A mask for a latent `factor` times smaller than its picture: the share
/// of each latent pixel's patch that is kept, `[1, 1, h, w]`.
///
/// A mean and not the nearest pixel, so that an edge that crosses a latent
/// pixel mixes there and does not step.
fn kept(mask: &Mask, factor: usize, device: &Device) -> Res<Tensor> {
    let (w, h) = (mask.width / factor, mask.height / factor);
    let mut keep = vec![0f32; w * h];
    for (i, k) in keep.iter_mut().enumerate() {
        let (lx, ly) = (i % w, i / w);
        let mut repaint = 0u32;
        for y in ly * factor..(ly + 1) * factor {
            repaint += mask.repaint[y * mask.width + lx * factor..y * mask.width + (lx + 1) * factor].iter().map(|&v| v as u32).sum::<u32>();
        }
        *k = 1.0 - repaint as f32 / (255 * factor * factor) as f32;
    }
    Ok(Tensor::from_vec(keep, (1, 1, h, w), device)?)
}

/// `x₀` at noise level `sigma`, with the noise `eps`: the two lines at the
/// top of this file.
fn noised(kind: Kind, x0: &Tensor, eps: &Tensor, sigma: f64) -> candle_core::Result<Tensor> {
    match kind {
        Kind::Epsilon => x0 + (eps * sigma)?,
        Kind::Flow => (x0 * (1.0 - sigma))? + (eps * sigma)?,
    }
}

impl Edited {
    /// Start an edit: the picture at the request's size, encoded by
    /// `encode` (pixels `[1, 3, H, W]` in `[−1, 1]` on the host, to a
    /// latent in the denoiser's units and f32 on the device), and the
    /// latent the first step is shown.
    ///
    /// `factor` is how many pixels a latent pixel covers along a side, and
    /// `eps` the noise, in the latent's shape: the seed's, as drawing from
    /// noise would start with, so that one seed is one picture here too.
    pub(crate) fn begin(edit: &Edit, req: &Resolved, sched: &Schedule, factor: usize, eps: Tensor, encode: impl FnOnce(&Tensor) -> Res<Tensor>) -> Res<(Self, Tensor)> {
        let fitted = fit(&edit.image, req.width, req.height);
        let mask = edit.mask.as_ref().map(|m| fit_mask(m, req.width, req.height));
        let x0 = encode(&super::dataset::pixels(&fitted)?)?.to_dtype(DType::F32)?;
        if x0.dims() != eps.dims() {
            return Err(format!("the picture's latent is {:?} and the noise {:?}", x0.dims(), eps.dims()).into());
        }
        let keep = mask.as_ref().map(|m| kept(m, factor, x0.device())).transpose()?;
        let steps = sched.steps();
        let first = steps - kvad::image::edit_steps(steps, req.strength.unwrap_or(1.0));
        let x = noised(sched.kind, &x0, &eps, sched.sigmas[first])?;
        Ok((Edited { x0, eps, keep, first, fitted, mask }, x))
    }

    /// After step `i`: the latent with the kept part put back, at the level
    /// the step came down to. As it was, where there is no mask.
    pub(crate) fn hold(&self, x: Tensor, sched: &Schedule, i: usize) -> candle_core::Result<Tensor> {
        let Some(keep) = &self.keep else { return Ok(x) };
        let was = noised(sched.kind, &self.x0, &self.eps, sched.sigmas[i + 1])?;
        // x + keep · (was − x)
        &x + (was - &x)?.broadcast_mul(keep)?
    }

    /// The decoded image with the kept pixels the picture's own again.
    pub(crate) fn paste(&self, mut image: Image) -> Image {
        let Some(mask) = &self.mask else { return image };
        for (i, &m) in mask.repaint.iter().enumerate() {
            let m = m as u32;
            for c in 0..3 {
                let (new, old) = (image.rgb[i * 3 + c] as u32, self.fitted.rgb[i * 3 + c] as u32);
                image.rgb[i * 3 + c] = ((new * m + old * (255 - m) + 127) / 255) as u8;
            }
        }
        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::schedule;
    use kvad::serde_json::json;

    fn picture(width: usize, height: usize) -> Image {
        Image { width, height, rgb: (0..width * height * 3).map(|i| (i % 251) as u8).collect() }
    }

    /// The largest piece of the shape asked for, from the middle.
    #[test]
    fn a_picture_is_cut_from_its_middle_to_the_shape_asked_for() {
        assert_eq!(middle((1000, 500), 512, 512), (250, 0, 500, 500));
        assert_eq!(middle((500, 1000), 512, 512), (0, 250, 500, 500));
        assert_eq!(middle((1000, 500), 1024, 512), (0, 0, 1000, 500));
        assert_eq!(middle((640, 480), 768, 512), (0, 27, 640, 426));
        // The size it already is: itself, to the byte.
        let p = picture(16, 8);
        assert_eq!(fit(&p, 16, 8), p);
        let f = fit(&p, 8, 8);
        assert_eq!((f.width, f.height), (8, 8));
        assert_eq!(&f.rgb[..3], &p.rgb[4 * 3..5 * 3], "the middle eight columns");
    }

    /// A latent pixel's share of its patch that is kept.
    #[test]
    fn a_mask_is_averaged_down_to_the_latent() {
        // 16×8, the right half repainted, and one pixel of the left.
        let mut repaint = vec![0u8; 16 * 8];
        for y in 0..8 {
            for x in 8..16 {
                repaint[y * 16 + x] = 255;
            }
        }
        repaint[0] = 255;
        let keep = kept(&Mask { width: 16, height: 8, repaint }, 8, &Device::Cpu).unwrap();
        assert_eq!(keep.dims(), [1, 1, 1, 2]);
        let v = keep.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 63.0 / 64.0).abs() < 1e-6 && v[1] == 0.0, "{v:?}");
    }

    /// An edit starts where its strength says, from the picture noised to
    /// there; a mask puts the picture back at each level, and at the last
    /// it is the picture.
    #[test]
    fn an_edit_starts_part_of_the_way_down_and_a_mask_keeps_its_side() {
        let cfg = json!({ "_class_name": "EulerDiscreteScheduler", "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear", "num_train_timesteps": 1000, "prediction_type": "epsilon", "steps_offset": 1, "timestep_spacing": "leading" });
        let sched = schedule::euler(&cfg, 20).unwrap();
        let image = picture(16, 8);
        // The left latent pixel is kept and the right one drawn.
        let mut repaint = vec![0u8; 16 * 8];
        (0..8).for_each(|y| (8..16).for_each(|x| repaint[y * 16 + x] = 255));
        let edit = Edit { image: image.clone(), mask: Some(Mask { width: 16, height: 8, repaint }), strength: Some(0.5) };
        let req = kvad::image::ImageRequest { edit: Some(edit.clone()), width: Some(16), height: Some(8), ..kvad::image::ImageRequest::new("x") }
            .resolved(&kvad::image::Defaults { width: 16, height: 8, steps: 20, guidance: 5.0, multiple: 8, takes_guidance: true, takes_negative: true, takes_loras: false, edits: true })
            .unwrap();
        let eps = Tensor::from_vec(vec![1f32, -1.0], (1, 1, 1, 2), &Device::Cpu).unwrap();
        // An "encoder" that says 3 everywhere.
        let (e, x) = Edited::begin(&edit, &req, &sched, 8, eps, |p| {
            assert_eq!(p.dims(), [1, 3, 8, 16]);
            Ok(Tensor::full(3f32, (1, 1, 1, 2), &Device::Cpu)?)
        })
        .unwrap();
        assert_eq!(e.first, 10, "half of twenty steps are left to run");
        let s = sched.sigmas[10] as f32;
        assert_eq!(x.flatten_all().unwrap().to_vec1::<f32>().unwrap(), [3.0 + s, 3.0 - s]);

        // A step that answered 100 everywhere: kept on the left, at the
        // next level's noise; the model's own on the right.
        let drawn = Tensor::full(100f32, (1, 1, 1, 2), &Device::Cpu).unwrap();
        let held = e.hold(drawn.clone(), &sched, 10).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((held[0] - (3.0 + sched.sigmas[11] as f32)).abs() < 1e-5 && held[1] == 100.0, "{held:?}");
        let last = e.hold(drawn, &sched, 19).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(last, [3.0, 100.0], "at the last level the kept side is the picture's latent");

        // And as pixels: the kept half is the picture's own bytes.
        let white = Image { width: 16, height: 8, rgb: vec![255; 16 * 8 * 3] };
        let pasted = e.paste(white);
        assert_eq!(&pasted.rgb[..8 * 3], &image.rgb[..8 * 3]);
        assert!(pasted.rgb[8 * 3..16 * 3].iter().all(|&v| v == 255));
    }

    /// SD 1.5 itself, on a picture it is given. A little strength returns
    /// nearly the picture and all of it something else; a mask's kept half
    /// comes back to the byte, and its other half is drawn. Needs SD 1.5 on
    /// this machine:
    ///
    ///     cargo test --release -p kvad-gpu edit::tests::sd15 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn sd15_edits_a_picture_and_keeps_what_a_mask_keeps() {
        use super::super::sd15::{Sd15, REPO};
        use kvad::image::{ImageRequest, Painter};
        crate::cap::at(20.0);
        let device = Device::new_metal(0).expect("a Metal device");
        let mut sd = Sd15::load(REPO, device, &mut |_| {}, &kvad::weights::Watcher::none()).unwrap();

        // Something to recognise: the model's own drawing.
        let first = sd.paint(&ImageRequest { seed: Some(7), steps: Some(20), ..ImageRequest::new("a red apple on a wooden table, photograph") }, &mut |_| true).unwrap().image;
        assert_eq!((first.width, first.height), (512, 512));
        // Mean absolute difference, in levels of 255.
        let apart = |a: &Image, b: &Image| a.rgb.iter().zip(&b.rgb).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.rgb.len() as f64;

        let mut edit = |strength: f32, mask: Option<Mask>, steps: &mut Vec<usize>| {
            let req = ImageRequest {
                seed: Some(11),
                steps: Some(20),
                edit: Some(Edit { image: first.clone(), mask, strength: Some(strength) }),
                ..ImageRequest::new("a green apple on a wooden table, photograph")
            };
            sd.paint(&req, &mut |s| {
                steps.push(s.total);
                true
            })
            .unwrap()
        };
        let mut ran = Vec::new();
        let little = edit(0.2, None, &mut ran);
        assert_eq!((ran.len(), ran[0]), (4, 4), "a fifth of twenty steps");
        assert_eq!(little.request.strength, Some(0.2));
        let mut ran = Vec::new();
        let all = edit(1.0, None, &mut ran);
        assert_eq!(ran.len(), 20);
        let (near, far) = (apart(&little.image, &first), apart(&all.image, &first));
        eprintln!("strength 0.2 is {near:.1} levels from the picture, and 1.0 is {far:.1}");
        assert!(near < 12.0 && far > 2.0 * near, "{near} and {far}");

        // The right half drawn anew, the left kept.
        let mut repaint = vec![0u8; 512 * 512];
        (0..512).for_each(|y| (256..512).for_each(|x| repaint[y * 512 + x] = 255));
        let masked = edit(1.0, Some(Mask { width: 512, height: 512, repaint }), &mut Vec::new());
        assert!(masked.request.masked);
        let half = |im: &Image, right: bool| Image {
            width: 256,
            height: 512,
            rgb: (0..512).flat_map(|y| { let at = (y * 512 + if right { 256 } else { 0 }) * 3; im.rgb[at..at + 256 * 3].to_vec() }).collect(),
        };
        assert_eq!(half(&masked.image, false), half(&first, false), "the kept half is the picture's own");
        let drawn = apart(&half(&masked.image, true), &half(&first, true));
        eprintln!("the drawn half is {drawn:.1} levels from what was there");
        assert!(drawn > 5.0, "{drawn}");
        for (name, im) in [("first", &first), ("little", &little.image), ("all", &all.image), ("masked", &masked.image)] {
            if let Ok(dir) = std::env::var("KVAD_EDIT_OUT") {
                std::fs::write(std::path::Path::new(&dir).join(format!("{name}.png")), im.png()).unwrap();
            }
        }
    }

    /// Flow matching mixes the picture and the noise; it does not add.
    #[test]
    fn a_flow_model_s_picture_is_mixed_with_its_noise() {
        let (x0, eps) = (Tensor::full(2f32, (1, 2), &Device::Cpu).unwrap(), Tensor::full(-1f32, (1, 2), &Device::Cpu).unwrap());
        let at = |kind, s| noised(kind, &x0, &eps, s).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert_eq!((at(Kind::Flow, 0.0), at(Kind::Flow, 1.0), at(Kind::Flow, 0.25)), (2.0, -1.0, 1.25));
        assert_eq!((at(Kind::Epsilon, 0.0), at(Kind::Epsilon, 2.0)), (2.0, 0.0));
    }
}
