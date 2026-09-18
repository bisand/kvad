//! A tiny deterministic RNG.
//!
//! We need random numbers to initialise weights and shuffle data, but pulling
//! in `rand` would break the "zero dependencies" rule. xorshift64* is about
//! ten lines and is more than good enough for this purpose.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state is a fixed point for xorshift, so avoid it.
        Rng { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        // Use the top 24 bits: f32 has a 24-bit significand, so this gives
        // us every representable value in [0,1) with equal probability.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal, via the Box-Muller transform.
    pub fn normal(&mut self) -> f32 {
        // u1 must not be 0, or ln(u1) is -inf.
        let u1 = self.uniform().max(f32::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            slice.swap(i, self.below(i + 1));
        }
    }
}
