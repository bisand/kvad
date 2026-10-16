//! Getting from a file to 16 kHz mono samples.
//!
//! A WAV file is read here by hand: it is a few chunks of framing around
//! samples stored as they are, and every recorder and converter can write
//! one. Anything else — MP3, M4A, FLAC, Opus, the audio of a video — is
//! handed to `ffmpeg` when there is one on the `PATH`, which is asked for
//! exactly what Whisper wants and writes it back as raw samples. That keeps
//! every codec out of this crate's dependencies, and leaves a machine with
//! no `ffmpeg` able to transcribe the one format that needs no codec.
//!
//! Whatever comes in at another rate is resampled by [`resample`], a
//! windowed-sinc filter: what the samples would have been, had the sound been
//! recorded at 16 kHz in the first place.

use super::mel::SAMPLE_RATE;
use std::path::Path;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A file, as Whisper hears it: mono, 16 kHz, `f32` in −1..1.
///
/// `ffmpeg` is the one to run for anything that is not a WAV, found by the
/// caller — [`crate::video::ffmpeg_on`] looks where a service's bare `PATH`
/// does not.
pub fn load(path: &Path, ffmpeg: Option<&Path>) -> Res<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if is_wav(&bytes) {
        return from_wav(&bytes).map_err(|e| format!("{}: {e}", path.display()).into());
    }
    with_ffmpeg(ffmpeg, path)
}

/// Samples from the bytes of an upload, which may be any format `ffmpeg`
/// reads. A WAV is read in memory; anything else is written to a temporary
/// file for `ffmpeg` to open, because some containers (MP4 above all) keep
/// their index at the end and cannot be read from a pipe.
pub fn from_bytes(bytes: &[u8], name_hint: &str, ffmpeg: Option<&Path>) -> Res<Vec<f32>> {
    if is_wav(bytes) {
        return from_wav(bytes);
    }
    let ext = Path::new(name_hint).extension().and_then(|e| e.to_str()).filter(|e| e.chars().all(|c| c.is_ascii_alphanumeric()));
    let path = std::env::temp_dir().join(format!("kvad-audio-{}-{}.{}", std::process::id(), unique(), ext.unwrap_or("bin")));
    std::fs::write(&path, bytes)?;
    let out = with_ffmpeg(ffmpeg, &path);
    let _ = std::fs::remove_file(&path);
    out
}

fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}

fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// Ask `ffmpeg` for 16 kHz mono 32-bit float, little-endian, on its stdout.
fn with_ffmpeg(ffmpeg: Option<&Path>, path: &Path) -> Res<Vec<f32>> {
    let Some(ffmpeg) = ffmpeg else {
        return Err(format!(
            "{} is not a WAV file, and reading anything else takes ffmpeg, which is not installed. \
             Install it, or convert the file first: ffmpeg -i IN -ar 16000 -ac 1 OUT.wav",
            path.display()
        )
        .into());
    };
    let input = path.to_str().ok_or("an audio file's path that is not UTF-8")?;
    let rate = SAMPLE_RATE.to_string();
    let out = crate::video::ffmpeg_out(ffmpeg, &["-i", input, "-vn", "-f", "f32le", "-ac", "1", "-ar", &rate, "pipe:1"], None)
        .map_err(|e| format!("ffmpeg could not read {}: {e}", path.display()))?;
    Ok(out.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// A WAV file's samples, mixed to mono and resampled to 16 kHz.
///
/// A WAV is a `RIFF` header and a list of chunks, each a four-letter name, a
/// length, and that many bytes. Two matter: `fmt `, which says how the
/// samples are stored, and `data`, which is them. Anything else — `LIST`
/// metadata, `fact`, cue points — is skipped by its length.
pub fn from_wav(bytes: &[u8]) -> Res<Vec<f32>> {
    if !is_wav(bytes) {
        return Err("not a WAV file: no RIFF/WAVE header".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);

    let mut format = None;
    let mut data = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32_at(at + 4) as usize;
        let body = at + 8;
        // A recorder that is stopped mid-write can leave a `data` length
        // that runs past the end of the file, or 0xFFFFFFFF; take what is there.
        let end = body.saturating_add(len).min(bytes.len());
        match id {
            b"fmt " if end - body >= 16 => {
                let mut tag = u16_at(body);
                let channels = u16_at(body + 2) as usize;
                let rate = u32_at(body + 4) as usize;
                let bits = u16_at(body + 14) as usize;
                // WAVE_FORMAT_EXTENSIBLE keeps the real format in the first two
                // bytes of a GUID further in.
                if tag == 0xFFFE && end - body >= 26 {
                    tag = u16_at(body + 24);
                }
                format = Some((tag, channels, rate, bits));
            }
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }
        // Chunks are padded to an even length.
        at = body.saturating_add(len).saturating_add(len & 1);
    }
    let (tag, channels, rate, bits) = format.ok_or("WAV file has no fmt chunk")?;
    let data = data.ok_or("WAV file has no data chunk")?;
    if channels == 0 || rate == 0 {
        return Err(format!("WAV file says it has {channels} channels at {rate} Hz").into());
    }

    let samples: Vec<f32> = match (tag, bits) {
        (1, 8) => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        (1, 16) => data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        (1, 24) => data
            .chunks_exact(3)
            .map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0)
            .collect(),
        (1, 32) => data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0).collect(),
        (3, 32) => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        (3, 64) => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        _ => return Err(format!("WAV format {tag} at {bits} bits is not one this reads (PCM 8/16/24/32, float 32/64)").into()),
    };

    // Mono is the average of the channels, frame by frame.
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect()
    };
    Ok(resample(&mono, rate, SAMPLE_RATE))
}

/// `x`, recorded at `from` Hz, as it would have been recorded at `to`.
///
/// Each output sample is a weighted sum of the input samples near the same
/// moment, weighted by a sinc — the ideal low-pass filter — cut off at the
/// lower of the two Nyquist frequencies, so that going down in rate drops
/// what the new rate cannot hold rather than folding it back as noise. The
/// sinc is infinitely long, so it is cut to `ZEROS` crossings each side and
/// tapered with a Hann window, which is what keeps the cut from ringing.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    use rayon::prelude::*;
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    const ZEROS: f64 = 16.0;
    let ratio = to as f64 / from as f64;
    // Cutoff as a fraction of the input rate's Nyquist.
    let cutoff = ratio.min(1.0) * 0.97;
    let half = ZEROS / cutoff; // filter half-width, in input samples
    let n_out = ((x.len() as f64) * ratio).round() as usize;
    (0..n_out)
        .into_par_iter()
        .map(|j| {
            let centre = j as f64 / ratio;
            let lo = (centre - half).ceil().max(0.0) as usize;
            let hi = ((centre + half).floor() as usize).min(x.len() - 1);
            let mut acc = 0.0f64;
            for (i, &v) in x.iter().enumerate().take(hi + 1).skip(lo) {
                let t = i as f64 - centre;
                let arg = std::f64::consts::PI * t * cutoff;
                let sinc = if arg.abs() < 1e-9 { 1.0 } else { arg.sin() / arg };
                let taper = 0.5 + 0.5 * (std::f64::consts::PI * t / half).cos();
                acc += v as f64 * cutoff * sinc * taper;
            }
            acc as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WAV file of `samples`, written the way any recorder would.
    pub(crate) fn wav(samples: &[i16], channels: u16, rate: u32) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32 + 12).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        // A chunk nobody asked about, to be skipped by its length.
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(b"abc\0"); // three bytes and a pad byte
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn a_wav_is_read_past_the_chunks_it_does_not_know() {
        let got = from_wav(&wav(&[0, 16384, -32768, 32767], 1, 16_000)).unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], 0.5);
        assert_eq!(got[2], -1.0);
        assert!((got[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let got = from_wav(&wav(&[16384, 0, -16384, -16384], 2, 16_000)).unwrap();
        assert_eq!(got, vec![0.25, -0.5]);
    }

    #[test]
    fn resampling_keeps_a_tone_a_tone() {
        // 440 Hz at 44.1 kHz, to 16 kHz: the same sine at the new rate, to
        // within what the filter's taper leaves, away from the two ends.
        let from = 44_100;
        let x: Vec<f32> = (0..from).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin() * 0.5).collect();
        let y = resample(&x, from, 16_000);
        assert_eq!(y.len(), 16_000);
        for (j, &v) in y.iter().enumerate().take(15_000).skip(1000) {
            let want = (2.0 * std::f32::consts::PI * 440.0 * j as f32 / 16_000.0).sin() * 0.5;
            assert!((v - want).abs() < 5e-3, "sample {j}: {v} vs {want}");
        }
    }

    #[test]
    fn resampling_down_drops_what_the_new_rate_cannot_hold() {
        // 12 kHz at 48 kHz is above 16 kHz's Nyquist (8 kHz). Kept, it would
        // fold back as a 4 kHz tone; filtered, it is close to nothing.
        let from = 48_000;
        let x: Vec<f32> = (0..from).map(|i| (2.0 * std::f32::consts::PI * 12_000.0 * i as f32 / from as f32).sin()).collect();
        let y = resample(&x, from, 16_000);
        let rms = (y[1000..15_000].iter().map(|v| v * v).sum::<f32>() / 14_000.0).sqrt();
        assert!(rms < 0.01, "{rms}");
    }
}
