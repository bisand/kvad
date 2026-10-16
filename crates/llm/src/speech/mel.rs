//! From sound to the picture of it Whisper reads: a log-mel spectrogram.
//!
//! The encoder never sees a waveform. It sees 3000 columns of 80 numbers (128
//! on large-v3), one column for every 10 ms of a 30-second window, each
//! saying how much energy there was in each of 80 frequency bands in the
//! 25 ms around that moment. Making that picture is four steps, and every
//! one of them has to be the reference's exactly, because the model learned
//! to read *that* picture and nothing near it:
//!
//! 1. **Frames.** Slide a 400-sample (25 ms) window along the audio, 160
//!    samples (10 ms) at a time, after padding each end by 200 samples
//!    mirrored from the inside, so frame `t` is centred on sample `160 t`.
//!    Multiply each frame by a Hann window, so its edges fade to zero and
//!    the cut does not show up as energy at every frequency.
//! 2. **Spectrum.** A discrete Fourier transform of each frame: 201
//!    frequencies from 0 to 8 kHz, 40 Hz apart. Keep the power, `re² + im²`.
//! 3. **Mel bands.** Hearing is logarithmic in pitch, so 201 evenly spaced
//!    frequencies are folded into 80 bands that are narrow at the bottom and
//!    wide at the top, each a triangle over the frequencies it covers.
//! 4. **Log, floor, scale.** `log10` of each band, everything more than 8
//!    (80 dB) below the loudest value raised to that floor, then
//!    `(x + 4) / 4` to put the result roughly in −1..1.
//!
//! The floor in step 4 is taken over the *whole* recording, not per window,
//! which is why [`log_mel`] is computed once for a file rather than once for
//! each 30 seconds of it.
//!
//! # The FFT
//!
//! Step 2 is the only new mathematics in speech recognition, and the one
//! place this repository needed a Fourier transform. A direct DFT of 400
//! samples is 400 × 201 multiply-adds per frame; [`fft`] gets the same
//! answer by halving: the transform of `n` samples is two transforms of
//! `n/2` — the even samples and the odd ones — recombined with one rotation
//! each. 400 halves to 25 and then stops being even, and 25 is done directly.
//! That is Cooley–Tukey, radix 2, with a plain DFT at the bottom, which is
//! all a frame length that is not a power of two needs.

/// Samples per second Whisper was trained on. Everything is resampled to this.
pub const SAMPLE_RATE: usize = 16_000;
/// Samples per frame: 25 ms.
pub const N_FFT: usize = 400;
/// Samples between frames: 10 ms.
pub const HOP: usize = 160;
/// Seconds in one window the encoder reads.
pub const CHUNK_SECONDS: usize = 30;
/// Samples in one window.
pub const N_SAMPLES: usize = CHUNK_SECONDS * SAMPLE_RATE;
/// Spectrogram columns in one window.
pub const N_FRAMES: usize = N_SAMPLES / HOP;

/// One complex number, as a pair. Nothing else here needs complex arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    fn mul(self, o: Complex) -> Complex {
        Complex { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn add(self, o: Complex) -> Complex {
        Complex { re: self.re + o.re, im: self.im + o.im }
    }
    fn sub(self, o: Complex) -> Complex {
        Complex { re: self.re - o.re, im: self.im - o.im }
    }
}

/// `e^{-2πi k/n}`: the rotation by `k` steps of `n` around the circle.
///
/// Computed in f64 and rounded once, so a long transform does not collect
/// the error of an f32 `sin` at every level.
fn twiddle(k: usize, n: usize) -> Complex {
    let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
    Complex { re: a.cos() as f32, im: a.sin() as f32 }
}

/// The discrete Fourier transform of `x`, the slow way: `X[k] = Σ x[t] e^{-2πi kt/n}`.
///
/// What [`fft`] falls back to when a length stops being even, and what the
/// tests check it against.
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex::default();
            for (t, &v) in x.iter().enumerate() {
                // `k t` reduced mod `n` first: the rotation only depends on
                // where around the circle it lands, and a smaller angle is a
                // more accurate one.
                acc = acc.add(v.mul(twiddle(k * t % n, n)));
            }
            acc
        })
        .collect()
}

/// The same transform as [`dft`], by halving while the length is even.
///
/// Split `x` into its even and odd samples, `E` and `O`, and transform each.
/// Then for `k < n/2`:
///
/// ```text
///   X[k]       = E[k] + w^k O[k]
///   X[k + n/2] = E[k] − w^k O[k]        w = e^{-2πi/n}
/// ```
///
/// because the odd samples are the even ones shifted by one, a shift is a
/// rotation in frequency, and `w^{k + n/2} = −w^k`. Each level does `n` of
/// these, and there are `log₂ n` levels.
pub fn fft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    if n <= 1 || n % 2 == 1 {
        return dft(x);
    }
    let even: Vec<Complex> = x.iter().step_by(2).copied().collect();
    let odd: Vec<Complex> = x.iter().skip(1).step_by(2).copied().collect();
    let (e, o) = (fft(&even), fft(&odd));
    let mut out = vec![Complex::default(); n];
    for k in 0..n / 2 {
        let t = twiddle(k, n).mul(o[k]);
        out[k] = e[k].add(t);
        out[k + n / 2] = e[k].sub(t);
    }
    out
}

/// The periodic Hann window: `0.5 − 0.5 cos(2π t / n)`.
///
/// *Periodic* rather than symmetric — the window is one period of a cosine
/// that would repeat, so it is zero at `t = 0` and not quite at `t = n − 1`.
/// It is what `torch.hann_window` gives by default and what Whisper was
/// trained with; the symmetric one is a different set of numbers.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|t| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * t as f64 / n as f64).cos()) as f32).collect()
}

/// Hertz to mels, on the Slaney scale: linear below 1 kHz, logarithmic above.
///
/// There are two mel scales in circulation. HTK's is `2595 log10(1 + f/700)`
/// everywhere; Slaney's, which librosa defaults to and Whisper was built on,
/// is a straight line up to 1 kHz and a logarithm after it. They agree in
/// spirit and not in numbers.
pub fn hz_to_mel(f: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0; // 1000 Hz on the linear part: 1000 / (200 / 3)
    let logstep = 6.4f64.ln() / 27.0;
    if f >= MIN_LOG_HZ {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / logstep
    } else {
        3.0 * f / 200.0
    }
}

pub fn mel_to_hz(m: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (m - MIN_LOG_MEL)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// The mel filterbank: `n_mels` triangles over the `N_FFT / 2 + 1` frequencies
/// of a frame's spectrum, as `[n_mels][n_freqs]`.
///
/// Band `m` rises from zero at its left edge to one at its centre and falls
/// back to zero at its right edge, where the edges are its neighbours'
/// centres, all evenly spaced in mels from 0 to 8 kHz. Each triangle is then
/// scaled by `2 / width in Hz` ("Slaney normalisation"), so that a wide band
/// at the top does not count for more just because it covers more
/// frequencies.
///
/// openai-whisper ships these as a file of numbers; transformers computes
/// them as here, and the two agree.
pub fn filterbank(n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let top = SAMPLE_RATE as f64 / 2.0;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(top));
    // n_mels + 2 edges: every band's left edge, centre and right edge.
    let edges: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64)).collect();
    let freqs: Vec<f64> = (0..n_freqs).map(|i| top * i as f64 / (n_freqs - 1) as f64).collect();
    (0..n_mels)
        .map(|m| {
            let (left, centre, right) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (right - left);
            freqs
                .iter()
                .map(|&f| {
                    let up = (f - left) / (centre - left);
                    let down = (right - f) / (right - centre);
                    (up.min(down).max(0.0) * norm) as f32
                })
                .collect()
        })
        .collect()
}

/// The log-mel spectrogram of 16 kHz mono `audio`, followed by `pad` samples
/// of silence, as `[n_mels][frames]`: one row per band, one column per 10 ms.
///
/// `pad` is there for the window loop: the last stretch of a recording is
/// read as a whole 30-second window like every other, so the spectrogram is
/// taken with 30 seconds of silence after it and the loop never runs off
/// the end. Silence before the floor is taken, as the reference does, so
/// what the floor is does not depend on how the audio is cut up.
///
/// There are `(audio + pad) / HOP` columns. Centring gives one more, over
/// the last half-frame, and the reference drops it, so this does too.
pub fn log_mel(audio: &[f32], pad: usize, n_mels: usize) -> Vec<Vec<f32>> {
    use rayon::prelude::*;

    let len = audio.len() + pad;
    let sample = |i: usize| if i < audio.len() { audio[i] } else { 0.0 };
    // Reflect-padding by half a frame at each end: sample `-1` is sample `1`,
    // and one past the end is one before it.
    let half = N_FFT / 2;
    let reflected = |i: isize| -> f32 {
        let n = len as isize;
        let j = if i < 0 { -i } else if i >= n { 2 * (n - 1) - i } else { i };
        sample(j as usize)
    };

    let frames = len / HOP;
    let window = hann(N_FFT);
    let bank = filterbank(n_mels);
    let n_freqs = N_FFT / 2 + 1;

    // One column per frame, in parallel: each is independent of the others.
    let columns: Vec<Vec<f32>> = (0..frames)
        .into_par_iter()
        .map(|t| {
            let start = (t * HOP) as isize - half as isize;
            let frame: Vec<Complex> =
                (0..N_FFT).map(|i| Complex { re: reflected(start + i as isize) * window[i], im: 0.0 }).collect();
            let spectrum = fft(&frame);
            let power: Vec<f32> = spectrum[..n_freqs].iter().map(|c| c.re * c.re + c.im * c.im).collect();
            bank.iter()
                .map(|band| {
                    let energy: f32 = band.iter().zip(&power).map(|(w, p)| w * p).sum();
                    energy.max(1e-10).log10()
                })
                .collect()
        })
        .collect();

    let loudest = columns.iter().flatten().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = loudest - 8.0;
    let mut mel = vec![vec![0.0f32; frames]; n_mels];
    for (t, col) in columns.iter().enumerate() {
        for (m, &v) in col.iter().enumerate() {
            mel[m][t] = (v.max(floor) + 4.0) / 4.0;
        }
    }
    mel
}

/// Columns `start..start + N_FRAMES` of a spectrogram, as one window the
/// encoder reads, laid out `[frame][band]` — time-major, the way the
/// encoder's first convolution slides along it.
///
/// The spectrogram was taken with silence after the audio (see [`log_mel`]),
/// so a window that runs past the end of the recording ends in what silence
/// looks like, which is what a short clip looked like in training: it was
/// padded as *sound* and then turned into a picture. transformers does the
/// same for a clip under 30 seconds. openai-whisper pads the *picture* with
/// zeros instead, and a zero there is −4 in `log10`, which is not silence;
/// the two give different last windows, and this follows the one that
/// matches training. Past the end of `mel` altogether, the last column is
/// repeated.
pub fn window(mel: &[Vec<f32>], start: usize) -> Vec<f32> {
    let n_mels = mel.len();
    let have = mel.first().map_or(0, Vec::len);
    let mut out = vec![0.0f32; N_FRAMES * n_mels];
    for t in 0..N_FRAMES {
        let col = start + t;
        for (m, row) in mel.iter().enumerate() {
            out[t * n_mels + m] = if col < have { row[col] } else { row[have - 1] };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Complex, b: Complex, tol: f32) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    #[test]
    fn the_fft_is_the_dft_for_every_length_that_matters() {
        // 400 is the one that is used: 2⁴ × 25, so four halvings and a
        // direct transform of 25 at the bottom. The others cover the base
        // cases: a power of two, an odd length, one and two.
        for n in [1, 2, 7, 16, 50, 400] {
            let x: Vec<Complex> =
                (0..n).map(|i| Complex { re: ((i * 37 % 11) as f32 - 5.0) / 3.0, im: ((i * 13 % 7) as f32 - 3.0) / 5.0 }).collect();
            let (fast, slow) = (fft(&x), dft(&x));
            for k in 0..n {
                assert!(close(fast[k], slow[k], 1e-3), "n={n} k={k}: {:?} vs {:?}", fast[k], slow[k]);
            }
        }
    }

    #[test]
    fn a_pure_tone_lands_in_its_own_bin() {
        // 10 cycles across 400 samples is bin 10: all the energy there and
        // at its mirror image, 390, and nothing anywhere else.
        let x: Vec<Complex> =
            (0..400).map(|t| Complex { re: (2.0 * std::f32::consts::PI * 10.0 * t as f32 / 400.0).cos(), im: 0.0 }).collect();
        let s = fft(&x);
        for (k, c) in s.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            let want = if k == 10 || k == 390 { 200.0 } else { 0.0 };
            assert!((mag - want).abs() < 1e-2, "bin {k}: {mag}");
        }
    }

    #[test]
    fn the_mel_scale_is_linear_then_logarithmic_and_inverts() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12);
        assert!((hz_to_mel(500.0) - 7.5).abs() < 1e-12);
        for f in [0.0, 300.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6, "{f}");
        }
    }

    #[test]
    fn every_band_is_a_triangle_that_sees_some_frequency() {
        for n_mels in [80, 128] {
            let bank = filterbank(n_mels);
            assert_eq!(bank.len(), n_mels);
            for (m, band) in bank.iter().enumerate() {
                assert_eq!(band.len(), N_FFT / 2 + 1);
                // Rises to one peak and falls: never up again after going down.
                let peak = band.iter().enumerate().fold(0, |b, (i, &v)| if v > band[b] { i } else { b });
                assert!(band[peak] > 0.0, "band {m} of {n_mels} is empty");
                assert!(band[..peak].windows(2).all(|w| w[0] <= w[1]), "band {m} not rising");
                assert!(band[peak..].windows(2).all(|w| w[0] >= w[1]), "band {m} not falling");
            }
        }
    }

    #[test]
    fn silence_is_the_floor_everywhere() {
        // All zeros: every band is log10(1e-10) = -10, which is also the
        // loudest value, so nothing is floored and every cell is (−10+4)/4.
        let mel = log_mel(&[], N_SAMPLES, 80);
        assert_eq!(mel.len(), 80);
        assert_eq!(mel[0].len(), N_FRAMES);
        assert!(mel.iter().flatten().all(|&v| (v + 1.5).abs() < 1e-6));
    }
}
