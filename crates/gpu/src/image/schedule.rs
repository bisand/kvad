//! Schedulers: the arithmetic between one denoiser call and the next.
//!
//! A scheduler has no weights. It decides which noise levels the steps visit,
//! what the denoiser is told about each one, and how a prediction turns into
//! the next latent. The two here are the two families that matter now.
//!
//! **Noise prediction** (SDXL). Training added Gaussian noise of standard
//! deviation σ to a clean latent, and the model learned to predict that noise,
//! `ε`. Karras et al. showed that sampling is then an ODE in σ, and the
//! simplest way to integrate it is Euler's: the direction of travel at σ *is*
//! the predicted noise, so `x ← x + ε̂ · (σ_next − σ)` walks from σ = 14.6
//! down to 0.
//!
//! **Flow matching** (Qwen-Image, FLUX, SD3). The noisy latent is a straight
//! line between image and noise, `x_σ = (1 − σ)·image + σ·noise` with σ from
//! 1 to 0, and the model predicts the line's direction, `v = noise − image`.
//! Euler along it is `x ← x + v̂ · (σ_next − σ)`: the same update, with a
//! different meaning for σ and for what the model predicts. A straight path is
//! why these models get away with fewer steps.

use kvad::serde_json::Value;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The noise levels a run visits, top down, with a final 0 appended, and what
/// the denoiser is told at each.
#[derive(Debug, Clone)]
pub(crate) struct Schedule {
    pub(crate) sigmas: Vec<f64>,
    /// What the denoiser's time embedding is fed at each step: a training
    /// timestep for SDXL, σ itself for flow matching.
    pub(crate) timesteps: Vec<f64>,
    /// The latent starts as unit noise times this.
    pub(crate) init_scale: f64,
    pub(crate) kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    /// The model sees `x / √(σ² + 1)` and predicts noise.
    Epsilon,
    /// The model sees `x` as it is and predicts velocity.
    Flow,
}

impl Schedule {
    /// What the denoiser should be shown at step `i`.
    pub(crate) fn input_scale(&self, i: usize) -> f64 {
        match self.kind {
            Kind::Epsilon => 1.0 / (self.sigmas[i].powi(2) + 1.0).sqrt(),
            Kind::Flow => 1.0,
        }
    }

    /// `dt` for step `i`: what the prediction is multiplied by and added.
    pub(crate) fn dt(&self, i: usize) -> f64 {
        self.sigmas[i + 1] - self.sigmas[i]
    }

    pub(crate) fn steps(&self) -> usize {
        self.timesteps.len()
    }
}

/// `EulerDiscreteScheduler` as SDXL configures it.
///
/// Only the settings SDXL ships are implemented, and anything else in the
/// config is refused rather than ignored: a scheduler that silently runs the
/// wrong schedule draws a worse picture and says nothing.
///
/// A setting the config leaves out means what diffusers' class for it
/// defaults to, and the class is the config's own even though Euler runs:
/// SD 1.5's PNDM config says neither `prediction_type` nor
/// `timestep_spacing`, which for PNDM are `epsilon` and `leading`.
pub(crate) fn euler(config: &Value, steps: usize) -> Res<Schedule> {
    // The noise levels first: they refuse a prediction or a β schedule that
    // is not SDXL's.
    let noising = Noising::ddpm(config)?;
    // diffusers' default spacing is the class's own.
    let spacing = config.get("timestep_spacing").and_then(Value::as_str).unwrap_or(match config.get("_class_name").and_then(Value::as_str) {
        Some("PNDMScheduler" | "DDIMScheduler" | "DDPMScheduler") => "leading",
        _ => "linspace",
    });
    if spacing != "leading" {
        return Err(format!("scheduler `timestep_spacing` is {spacing:?}; only \"leading\" is implemented").into());
    }
    if config.get("use_karras_sigmas").and_then(Value::as_bool) == Some(true) {
        return Err("Karras sigmas are not implemented".into());
    }
    let train = config.get("num_train_timesteps").and_then(Value::as_u64).unwrap_or(1000) as usize;
    let offset = config.get("steps_offset").and_then(Value::as_u64).unwrap_or(0) as usize;

    // σ for every training timestep: the noise-to-signal ratio after that
    // much cumulative noising.
    let table: Vec<f64> = noising.kept.iter().map(|&kept| ((1.0 - kept) / kept).sqrt()).collect();

    // `leading`: every ⌊1000/n⌋-th timestep counted up from 0, moved up by
    // the offset, then run from the top.
    let ratio = train / steps;
    let timesteps: Vec<f64> = (0..steps).rev().map(|i| (i * ratio + offset) as f64).collect();
    // Timesteps are whole here, so "interpolating" the table is reading it.
    let mut sigmas: Vec<f64> = timesteps.iter().map(|&t| table[t as usize]).collect();
    let top = sigmas[0];
    sigmas.push(0.0);
    Ok(Schedule {
        sigmas,
        timesteps,
        // `leading` spacing starts from the top σ of the run, with the unit
        // variance of the clean latent added in quadrature.
        init_scale: (top * top + 1.0).sqrt(),
        kind: Kind::Epsilon,
    })
}

/// What training does to a clean latent before the denoiser sees it: the
/// other direction along the road [`euler`] comes down.
///
/// A noise-prediction model (SDXL) was trained on a thousand noise levels.
/// Level `t` keeps a share `ᾱ_t` of the clean latent's variance and makes
/// up the rest with noise:
///
/// ```text
/// x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε
/// ```
///
/// and the model, shown `x_t` and told `t`, is asked for `ε`. Each level
/// multiplies what the one before kept by `1 − β_t`, and `scaled_linear`
/// spaces the *square roots* of β evenly from `beta_start` to `beta_end`.
/// At `t = 0` nearly all of the latent is kept, 0.99915; at 999, 0.0047.
///
/// It is the same picture [`euler`] draws from: there the latent is
/// `x₀ + σ·ε` and the model is shown it divided by `√(σ² + 1)`, and with
/// `σ² = (1 − ᾱ)/ᾱ` that is this line exactly. So a LoRA trained on these
/// inputs is trained on what drawing shows the model.
pub(crate) struct Noising {
    /// `ᾱ_t` for each training timestep.
    kept: Vec<f64>,
}

impl Noising {
    /// From a scheduler config, as [`euler`] reads it: only what SDXL
    /// ships, and anything else refused by name.
    pub(crate) fn ddpm(config: &Value) -> Res<Self> {
        let f = |k: &str| config.get(k).and_then(Value::as_f64).ok_or_else(|| format!("scheduler config has no `{k}`"));
        for (k, default, want) in [("beta_schedule", "linear", "scaled_linear"), ("prediction_type", "epsilon", "epsilon")] {
            let is = config.get(k).and_then(Value::as_str).unwrap_or(default);
            if is != want {
                return Err(format!("scheduler `{k}` is {is:?}; only {want:?} is implemented").into());
            }
        }
        let train = config.get("num_train_timesteps").and_then(Value::as_u64).unwrap_or(1000) as usize;
        let (b0, b1) = (f("beta_start")?.sqrt(), f("beta_end")?.sqrt());
        let mut alpha_bar = 1.0;
        let kept = (0..train)
            .map(|i| {
                let beta = (b0 + (b1 - b0) * i as f64 / (train - 1) as f64).powi(2);
                alpha_bar *= 1.0 - beta;
                alpha_bar
            })
            .collect();
        Ok(Noising { kept })
    }

    /// How many noise levels there are: a timestep is one of `0..levels`.
    pub(crate) fn levels(&self) -> usize {
        self.kept.len()
    }

    /// What level `t` multiplies the clean latent by, and the noise.
    pub(crate) fn mix(&self, t: usize) -> (f64, f64) {
        (self.kept[t].sqrt(), (1.0 - self.kept[t]).sqrt())
    }
}

/// `FlowMatchEulerDiscreteScheduler` with resolution-dependent shifting, as
/// Qwen-Image configures it.
///
/// `patches` is the number of image tokens the denoiser sees. Larger images
/// have more redundant pixels, so the same σ destroys less of what matters;
/// the shift spends more of the run at high noise to compensate.
pub(crate) fn flow(config: &Value, steps: usize, patches: usize) -> Res<Schedule> {
    let f = |k: &str, d: f64| config.get(k).and_then(Value::as_f64).unwrap_or(d);
    let n = steps as f64;
    let linear = (0..steps).map(|i| 1.0 - i as f64 * (1.0 - 1.0 / n) / (n - 1.0).max(1.0));
    let mut sigmas: Vec<f64> = match config.get("use_dynamic_shifting").and_then(Value::as_bool) {
        Some(true) => {
            if config.get("time_shift_type").and_then(Value::as_str).unwrap_or("exponential") != "exponential" {
                return Err("only the exponential time shift is implemented".into());
            }
            let (base_len, max_len) = (f("base_image_seq_len", 256.0), f("max_image_seq_len", 4096.0));
            let (base_shift, max_shift) = (f("base_shift", 0.5), f("max_shift", 1.15));
            let m = (max_shift - base_shift) / (max_len - base_len);
            let mu = patches as f64 * m + (base_shift - m * base_len);
            linear.map(|s| mu.exp() / (mu.exp() + (1.0 / s - 1.0))).collect()
        }
        // A fixed shift, the same at every size. FLUX.1-schnell's is 1,
        // which leaves the straight line from 1 down to 1/n as it is.
        _ => {
            let shift = f("shift", 1.0);
            linear.map(|s| shift * s / (1.0 + (shift - 1.0) * s)).collect()
        }
    };
    // Stretch so the run ends at `shift_terminal` rather than wherever the
    // shift left it.
    if let Some(end) = config.get("shift_terminal").and_then(Value::as_f64).filter(|&e| e > 0.0) {
        let last = 1.0 - sigmas[steps - 1];
        let scale = last / (1.0 - end);
        for s in &mut sigmas {
            *s = 1.0 - (1.0 - *s) / scale;
        }
    }
    let timesteps = sigmas.clone();
    sigmas.push(0.0);
    Ok(Schedule { sigmas, timesteps, init_scale: 1.0, kind: Kind::Flow })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvad::serde_json::json;

    fn sdxl() -> Value {
        json!({
            "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
            "num_train_timesteps": 1000, "prediction_type": "epsilon", "steps_offset": 1,
            "timestep_spacing": "leading", "use_karras_sigmas": false
        })
    }

    /// The timesteps diffusers' `EulerDiscreteScheduler.set_timesteps(30)`
    /// picks with SDXL's config, and σ at two of them as its β table gives
    /// them (recomputed independently, in double precision, from the same
    /// formula).
    #[test]
    fn sdxl_s_thirty_steps_are_the_ones_diffusers_picks() {
        let s = euler(&sdxl(), 30).unwrap();
        assert_eq!(&s.timesteps[..3], &[958.0, 925.0, 892.0]);
        assert_eq!(&s.timesteps[27..], &[67.0, 34.0, 1.0]);
        assert_eq!(s.sigmas.len(), 31);
        assert_eq!(s.sigmas[30], 0.0);
        // σ at timestep 1 and at 958, from the β table.
        assert!((s.sigmas[29] - 0.041_314).abs() < 1e-5, "{}", s.sigmas[29]);
        assert!((s.sigmas[0] - 11.476_85).abs() < 1e-4, "{}", s.sigmas[0]);
        assert!((s.init_scale - (s.sigmas[0].powi(2) + 1.0).sqrt()).abs() < 1e-12);
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]), "σ must fall every step");
    }

    /// Training's noised latent is what drawing shows the model: at every
    /// step of a run, `x₀ + σ·ε` scaled as [`Schedule::input_scale`] scales
    /// it is `√ᾱ·x₀ + √(1 − ᾱ)·ε`.
    #[test]
    fn training_noises_a_latent_as_drawing_shows_it() {
        let (noising, s) = (Noising::ddpm(&sdxl()).unwrap(), euler(&sdxl(), 30).unwrap());
        assert_eq!(noising.levels(), 1000);
        for i in 0..s.steps() {
            let (signal, noise) = noising.mix(s.timesteps[i] as usize);
            assert!((signal - s.input_scale(i)).abs() < 1e-12 && (noise - s.sigmas[i] * s.input_scale(i)).abs() < 1e-12);
            assert!((signal * signal + noise * noise - 1.0).abs() < 1e-12, "the variance is kept");
        }
        let (first, last) = (noising.mix(0).0.powi(2), noising.mix(999).0.powi(2));
        assert!((first - 0.99915).abs() < 1e-5 && (last - 0.00466).abs() < 1e-5, "{first} {last}");
    }

    #[test]
    fn a_setting_this_does_not_implement_is_refused_by_name() {
        let mut c = sdxl();
        c["prediction_type"] = json!("v_prediction");
        assert!(euler(&c, 30).unwrap_err().to_string().contains("prediction_type"));

        // Left out, a setting is the class's default: SD 1.5's PNDM config
        // runs as SDXL's Euler does, and an Euler config without a spacing
        // is `linspace`, which is refused.
        let pndm = json!({
            "_class_name": "PNDMScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
            "num_train_timesteps": 1000, "set_alpha_to_one": false, "skip_prk_steps": true, "steps_offset": 1,
        });
        let mut sdxl = pndm.clone();
        sdxl["prediction_type"] = json!("epsilon");
        sdxl["timestep_spacing"] = json!("leading");
        assert_eq!(euler(&pndm, 25).unwrap().sigmas, euler(&sdxl, 25).unwrap().sigmas);
        let mut plain = pndm.clone();
        plain["_class_name"] = json!("EulerDiscreteScheduler");
        assert!(euler(&plain, 25).unwrap_err().to_string().contains("linspace"));
    }

    fn qwen() -> Value {
        json!({
            "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 8192,
            "max_shift": 0.9, "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02,
            "time_shift_type": "exponential", "use_dynamic_shifting": true
        })
    }

    /// FLUX.1-schnell's: no dynamic shift, a shift of 1, four steps.
    #[test]
    fn schnell_s_four_steps_are_a_straight_line() {
        let c = json!({ "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096,
                        "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 1.0,
                        "use_dynamic_shifting": false });
        let s = flow(&c, 4, 4096).unwrap();
        assert_eq!(s.sigmas, [1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn a_flow_schedule_runs_from_one_to_its_terminal_then_zero() {
        // 1328² is 83² patches after the VAE and the 2×2 patching.
        let s = flow(&qwen(), 50, 83 * 83).unwrap();
        assert_eq!(s.sigmas.len(), 51);
        assert!((s.sigmas[0] - 1.0).abs() < 1e-12);
        assert!((s.sigmas[49] - 0.02).abs() < 1e-12, "{}", s.sigmas[49]);
        assert_eq!(s.sigmas[50], 0.0);
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
        // The shift keeps the run near the noisy end longer than a straight
        // line from 1 would: halfway through, σ is still well above a half.
        assert!(s.sigmas[25] > 0.6, "{}", s.sigmas[25]);
    }
}
