//! A take's loudness, evened out before Whisper hears it: a slow gain that
//! brings a quiet voice up toward the level of a near one, and never turns a
//! loud one down.
//!
//! A phone on a meeting table hears the person beside it 20 dB or more above
//! the one across the room, and Whisper, which reads loudness as it is
//! (its log-mel has no gain of its own), wrote nothing while the far one
//! spoke (docs/phone-speak-plan.md, *Hearing the far end of the table*).
//!
//! The level is measured 50 ms at a time, and only a block that stands out
//! of the room's noise counts as voice. The noise is the quietest block of
//! the last three seconds: speech has gaps between its words, and a room
//! that gets noisier is followed within the three. The gain follows voices,
//! and holds between them, so the room's own hiss is raised no more than
//! the last voice was. It rises toward a quieter voice
//! over about a second and falls at once for a louder one, and a block's
//! peak caps it, so a near voice or a laugh never clips. With no file of its
//! own, so that `crates/gpu/examples/live_session` hears takes through the
//! very same code.

/// 50 ms at 16 kHz.
const BLOCK: usize = 800;

/// The level a voice is brought up toward, as RMS of full scale: -26 dBFS,
/// where a near voice is on the phone.
pub const TARGET: f32 = 0.05;

/// The most it raises a voice: 18 dB.
pub const MOST: f32 = 8.0;

/// A block is voice when it is this many times the room's noise (8 dB), as
/// `speak::Quiet` has it.
const ABOVE: f32 = 2.5;

/// The blocks the room's noise is the quietest of: 3 s.
const HEARD: usize = 60;

/// Quieter than this is never voice: -66 dBFS.
const QUIETEST: f32 = 0.0005;

/// The highest a block's peak is let reach.
const CEILING: f32 = 0.9;

/// The gain stage: fed the microphone's samples as they come, which it
/// changes in place.
pub struct Level {
    target: f32,
    most: f32,
    /// The last [`HEARD`] blocks' RMS, the room's noise being the least.
    recent: Vec<f32>,
    /// The voice's level, as log2 of RMS, once one is heard.
    voice: Option<f32>,
    /// The gain at the end of the last block.
    gain: f32,
}

impl Default for Level {
    fn default() -> Level {
        Level::new(TARGET, MOST)
    }
}

impl Level {
    /// A gain bringing voices up toward `target` RMS, by at most `most`.
    pub fn new(target: f32, most: f32) -> Level {
        Level { target, most, recent: Vec::with_capacity(HEARD), voice: None, gain: 1.0 }
    }

    /// The gain now, as a factor.
    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn feed(&mut self, samples: &mut [i16]) {
        for block in samples.chunks_mut(BLOCK) {
            self.block(block);
        }
    }

    fn block(&mut self, block: &mut [i16]) {
        let (mut sum, mut peak) = (0.0f64, 0.0f32);
        for &s in block.iter() {
            let x = s as f32 / 32768.0;
            sum += (x * x) as f64;
            peak = peak.max(x.abs());
        }
        let rms = (sum / block.len() as f64).sqrt() as f32;

        if self.recent.len() == HEARD {
            self.recent.remove(0);
        }
        self.recent.push(rms);
        let floor = self.recent.iter().copied().fold(f32::MAX, f32::min);
        let voice = rms > (floor * ABOVE).max(QUIETEST);

        let mut want = self.gain;
        if voice {
            // In decibels: up to a louder voice in a block or two, down to
            // a quieter one over a second or so.
            let db = rms.log2();
            let level = self.voice.get_or_insert(db);
            *level += (db - *level) * if db > *level { 0.5 } else { 0.1 };
            want = (self.target / level.exp2()).clamp(1.0, self.most);
        }
        if peak > 0.0 {
            want = want.min((CEILING / peak).max(1.0));
        }

        // Up from the last block's gain across the block, so that it never
        // steps; down at once, so that a sudden loud block is not clipped.
        let (from, n) = (self.gain.min(want), block.len() as f32);
        for (i, s) in block.iter_mut().enumerate() {
            let g = from + (want - from) * (i + 1) as f32 / n;
            *s = (*s as f32 * g).round().clamp(-32768.0, 32767.0) as i16;
        }
        self.gain = want;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `secs` of words, as a 220 Hz tone at amplitude `a` 250 ms at a time
    /// with 100 ms between, over a noise of `noise`.
    fn sound(secs: f32, a: f32, noise: f32, seed: &mut u64) -> Vec<i16> {
        (0..(secs * 16_000.0) as usize)
            .map(|i| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let n = ((*seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * noise;
                let word = if i % 5600 < 4000 { a } else { 0.0 };
                ((word * (i as f32 * 220.0 * std::f32::consts::TAU / 16_000.0).sin() + n) * 32767.0) as i16
            })
            .collect()
    }

    fn rms(s: &[i16]) -> f32 {
        (s.iter().map(|&x| (x as f64 / 32768.0).powi(2)).sum::<f64>() / s.len() as f64).sqrt() as f32
    }

    /// Fed as the app feeds it, a tenth of a second at a time.
    fn through(level: &mut Level, mut s: Vec<i16>) -> Vec<i16> {
        for chunk in s.chunks_mut(1600) {
            level.feed(chunk);
        }
        s
    }

    #[test]
    fn a_far_voice_is_brought_up_and_a_near_one_is_left_alone() {
        let (mut level, mut seed) = (Level::default(), 3u64);
        let near = sound(3.0, 0.2, 0.002, &mut seed);
        assert_eq!(through(&mut level, near.clone()), near);
        // A voice at a tenth of it, -20 dB, comes up to within 3 dB of the
        // target after a couple of seconds.
        let far = through(&mut level, sound(4.0, 0.01, 0.002, &mut seed));
        let r = rms(&far[2 * 16_000..]);
        assert!(r > TARGET / 1.42 && r < TARGET * 1.42, "{r}");
    }

    #[test]
    fn the_room_is_not_raised_with_no_one_speaking() {
        let (mut level, mut seed) = (Level::default(), 5u64);
        through(&mut level, sound(5.0, 0.0, 0.003, &mut seed));
        assert_eq!(level.gain(), 1.0);
    }

    #[test]
    fn a_loud_block_after_a_quiet_voice_does_not_clip() {
        let (mut level, mut seed) = (Level::default(), 9u64);
        through(&mut level, sound(4.0, 0.01, 0.001, &mut seed));
        assert!(level.gain() > 4.0);
        let loud = through(&mut level, sound(0.5, 0.6, 0.001, &mut seed));
        let peak = loud.iter().map(|&x| (x as i32).abs()).max().unwrap();
        assert!(peak < 32_000, "{peak}");
    }
}
