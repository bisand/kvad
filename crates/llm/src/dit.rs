//! Image models trained here: a diffusion transformer from `nervus`, drawn
//! on the CPU.
//!
//! Every other image model `kvad` runs is a GPU pipeline in `kvad-gpu`,
//! because each is billions of parameters behind a VAE. `nervus`'s
//! `train_digits` makes a DiT of about a million, drawing pixels directly,
//! and the code that trained it is the code that draws with it: the same
//! [`nervus::dit::Dit`], the same [`nervus::flow::sample_watched`]. So it
//! runs here, in this crate, on the CPU, at whatever backend it was asked
//! for — the way a text model trained by `kvad train` is loaded by the code
//! that loads GPT-2.
//!
//! It is a [`Painter`] like any other, found by the `_class_name` in its
//! `model_index.json` ([`nervus::dit::PIPELINE`]), so the server lists it,
//! loads it and hands it image requests by the same rules as FLUX. What the
//! prompt means is the one thing that differs: this model knows a handful of
//! labels, not language, so the prompt has to be one of them.

use crate::image::{Defaults, Image, ImageRequest, Painted, Painter, Refused, Step};
use nervus::dit::{self, Dit};
use nervus::rng::Rng;
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub use nervus::dit::PIPELINE;

/// The prompt that asks for no label in particular: whatever the model
/// draws when it is not told what to draw.
pub const ANY: &str = "any";

/// The directory `repo` names, if it is a model of this kind on this machine.
///
/// Asks the disk only. These models are made here and have no Hub repo, so
/// anything that is not already a directory is not one.
pub fn dir_of(repo: &str) -> Option<PathBuf> {
    let dir = crate::weights::local_dir(repo)?;
    is_pipeline_dir(&dir).then_some(dir)
}

fn is_pipeline_dir(dir: &Path) -> bool {
    crate::weights::read_json(&dir.join("model_index.json"))
        .ok()
        .and_then(|v| v.get("_class_name")?.as_str().map(str::to_string))
        .is_some_and(|c| c == dit::PIPELINE)
}

/// Whether `repo` is a model of this kind on this machine.
pub fn is_one(repo: &str) -> bool {
    dir_of(repo).is_some()
}

/// What the model will hold in memory: every parameter, in f32. Not the
/// file's size, which holds each block's copy of the conditioning's
/// embedders where the model keeps one.
pub fn weight_bytes(repo: &str) -> Option<u64> {
    let (mut model, _) = dit::load(&dir_of(repo)?).ok()?;
    Some(4 * model.param_count() as u64)
}

pub struct Drawer {
    model: Dit,
    labels: Vec<String>,
    params: usize,
}

impl Drawer {
    pub fn load(repo: &str, progress: &mut dyn FnMut(&str)) -> Res<Drawer> {
        let dir = dir_of(repo).ok_or_else(|| format!("{repo} is not a model trained by nervus on this machine"))?;
        progress(&format!("reading {}", dir.display()));
        let (mut model, labels) = dit::load(&dir)?;
        let params = model.param_count();
        Ok(Drawer { model, labels, params })
    }

    /// The label a prompt names: one of the model's own, matched without
    /// regard to case or the spaces around it, or [`ANY`] for none.
    pub fn label(&self, prompt: &str) -> Result<usize, Refused> {
        let asked = prompt.trim();
        if asked.eq_ignore_ascii_case(ANY) {
            return Ok(self.model.config().unconditional());
        }
        self.labels.iter().position(|l| l.eq_ignore_ascii_case(asked)).ok_or_else(|| {
            Refused(format!(
                "this model draws one of {}, or \"{ANY}\" for whichever it likes; it knows labels, not language, and \"{asked}\" is not one",
                self.labels.join(", ")
            ))
        })
    }
}

/// A picture in [-1, 1], one channel, as grey RGB.
fn to_image(x: &[f32], side: usize) -> Image {
    let rgb = x.iter().flat_map(|&v| [(((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8; 3]).collect();
    Image { width: side, height: side, rgb }
}

impl Painter for Drawer {
    fn paint(&mut self, req: &ImageRequest, on_step: &mut dyn FnMut(Step) -> bool) -> Res<Painted> {
        let resolved = req.resolved(&self.defaults())?;
        let side = self.model.config().image;
        if (resolved.width, resolved.height) != (side, side) {
            return Err(Refused(format!("this model draws {side}×{side} and nothing else")).into());
        }
        if self.model.config().channels != 1 {
            return Err(format!("a model with {} channels; only greyscale is drawn here", self.model.config().channels).into());
        }
        if resolved.negative_prompt.is_some() {
            return Err(Refused("this model steers away from no label in particular, so it has no use for a negative prompt".into()).into());
        }
        let label = self.label(&resolved.prompt)?;

        let started = std::time::Instant::now();
        let total = resolved.steps;
        let drawn = nervus::flow::sample_watched(
            &mut self.model,
            &[label],
            total,
            resolved.guidance,
            &mut Rng::new(resolved.seed),
            &mut |done, x| on_step(Step { done, total, preview: resolved.preview.then(|| to_image(x, side)) }),
        )
        .ok_or("cancelled")?;
        Ok(Painted {
            image: to_image(&drawn, side),
            request: resolved,
            encode_secs: 0.0,
            denoise_secs: started.elapsed().as_secs_f64(),
            decode_secs: 0.0,
        })
    }

    fn defaults(&self) -> Defaults {
        let side = self.model.config().image;
        // `train_digits` draws its checkpoints at 20 steps and guidance 2.
        Defaults { width: side, height: side, steps: 20, guidance: 2.0, multiple: side, takes_guidance: true, takes_loras: false }
    }

    fn summary(&self) -> String {
        let c = self.model.config();
        format!(
            "DiT trained by nervus, {} layers, d_model {}, {:.2} M parameters, draws {}×{} ({})",
            c.n_layers,
            c.d_model,
            self.params as f64 / 1e6,
            c.image,
            c.image,
            self.labels.join(", ")
        )
    }

    fn params(&self) -> usize {
        self.params
    }

    fn weight_bytes(&self) -> usize {
        4 * self.params
    }

    fn backend(&self) -> String {
        "cpu f32".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nervus::dit::{Attention, DitConfig};

    const CONFIG: DitConfig = DitConfig { image: 8, frames: 1, attention: Attention::Full, channels: 1, patch: 4, classes: 3, d_model: 16, n_heads: 2, n_layers: 2 };

    fn saved(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kvad-dit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dit::save(&dir, &mut Dit::new(CONFIG, &mut Rng::new(1)), &["cat", "dog", "Fish"]).unwrap();
        dir
    }

    fn drawer(dir: &Path) -> Drawer {
        Drawer::load(dir.to_str().unwrap(), &mut |_| {}).unwrap()
    }

    #[test]
    fn it_is_found_by_its_pipeline_and_nothing_else_is() {
        let dir = saved("found");
        assert!(is_one(dir.to_str().unwrap()));
        assert_eq!(weight_bytes(dir.to_str().unwrap()), Some(4 * Dit::new(CONFIG, &mut Rng::new(1)).param_count() as u64));
        std::fs::write(dir.join("model_index.json"), r#"{"_class_name": "FluxPipeline"}"#).unwrap();
        assert!(!is_one(dir.to_str().unwrap()));
        assert!(!is_one("owner/some-repo"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_prompt_is_a_label_or_it_is_refused_with_the_list() {
        let dir = saved("labels");
        let d = drawer(&dir);
        assert_eq!(d.label("dog").unwrap(), 1);
        assert_eq!(d.label("  fish ").unwrap(), 2, "case and spaces are not the point");
        assert_eq!(d.label("ANY").unwrap(), CONFIG.unconditional());
        let err = d.label("a dog on a beach").unwrap_err().to_string();
        assert!(err.contains("cat, dog, Fish"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Drawn through the `Painter`, it is the image `nervus` draws for the
    /// same seed, one report per step, and a stop is a refusal, not a picture.
    #[test]
    fn it_paints_what_nervus_draws_and_stops_when_asked() {
        let dir = saved("paint");
        let mut d = drawer(&dir);
        let req = ImageRequest { steps: Some(5), seed: Some(9), preview: true, ..ImageRequest::new("dog") };

        let mut steps = Vec::new();
        let painted = d.paint(&req, &mut |s| {
            steps.push((s.done, s.total, s.preview.is_some()));
            true
        })
        .unwrap();
        assert_eq!(steps, (1..=5).map(|n| (n, 5, true)).collect::<Vec<_>>());
        assert_eq!((painted.image.width, painted.image.height), (8, 8));

        let (mut model, _) = dit::load(&dir).unwrap();
        let expected = to_image(&nervus::flow::sample(&mut model, 1, 5, 2.0, &mut Rng::new(9)), 8);
        assert_eq!(painted.image, expected);

        let stopped = d.paint(&req, &mut |s| s.done < 2);
        assert_eq!(stopped.err().map(|e| e.to_string()).as_deref(), Some("cancelled"));

        let wrong_size = d.paint(&ImageRequest { width: Some(16), height: Some(16), ..req.clone() }, &mut |_| true);
        assert!(wrong_size.err().unwrap().to_string().contains("8×8"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
