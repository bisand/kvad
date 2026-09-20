//! Hand-written SIMD, for the two kernels the compiler will not reach on its
//! own.
//!
//! Both are here for the same reason and it is not the usual one. Neither
//! needs an instruction Rust cannot name — `SDOT` it emits happily, and
//! `SMMLA` is four lines of assembly away. What it will not do is *choose*
//! them: for `SMMLA` there is no source shape that implies it, and for the
//! 4-bit kernel every shape that should imply `SDOT` makes LLVM pick
//! something else instead. See [`q4_row_dots`].
//!
//! # SMMLA
//!
//! ARMv8.6's `i8mm` extension adds `SMMLA`, which multiplies a 2x8 block of
//! `i8` by an 8x2 block and accumulates a 2x2 `i32` result:
//!
//! ```text
//!   a = [ A0 (8 bytes) | A1 (8 bytes) ]
//!   b = [ B0 (8 bytes) | B1 (8 bytes) ]
//!   result lanes = [ A0·B0, A0·B1, A1·B0, A1·B1 ]
//! ```
//!
//! That is 4 dot products of length 8 — **32 multiply-accumulates in one
//! instruction**, twice what `SDOT` manages.
//!
//! # Why it cannot help token generation
//!
//! Look at the shape it wants: *two* independent A rows and *two* independent
//! B rows. Generating one token at a time gives you exactly one activation
//! vector, so `B0` and `B1` would have to be the same data and half the
//! result lanes would be duplicates. The useful throughput collapses back to
//! `SDOT`'s.
//!
//! It only pays off where there is a real matrix on both sides — which, in a
//! decoder, means **prefill**: the prompt's tokens can all go through a layer
//! together. That is why [`crate::quant::Weight::matmul_bt`] exists.
//!
//! # Why inline assembly
//!
//! The ACLE intrinsic (`vmmlaq_s32`) is still unstable in Rust, and putting
//! the whole project on nightly for one instruction is a bad trade. Inline
//! assembly is stable, and this is four lines of it.

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;
    use std::arch::asm;
    use std::sync::OnceLock;

    /// Does this CPU have `i8mm`? Checked once, then cached.
    ///
    /// Detected at runtime rather than gated at compile time, so one binary
    /// runs everywhere: Apple Silicon from M2 on and recent server ARM have
    /// it, older cores do not.
    pub fn has_i8mm() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            // Setting KVAD_NO_I8MM forces the SDOT path, which is how the
            // contribution of this one instruction gets measured rather than
            // assumed.
            if std::env::var_os("KVAD_NO_I8MM").is_some() {
                return false;
            }
            std::arch::is_aarch64_feature_detected!("i8mm")
        })
    }

    /// One `SMMLA`.
    ///
    /// # Safety
    /// The caller must have checked [`has_i8mm`].
    #[inline]
    #[target_feature(enable = "i8mm")]
    pub unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut out = acc;
        asm!(
            "smmla {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack)
        );
        out
    }

    /// Build the 16-byte operand from two rows that are far apart in memory.
    ///
    /// Weight rows stay in their ordinary row-major layout; only the small
    /// activation block is pre-packed. Measured, this costs about 12% against
    /// a fully pre-packed version and saves repacking the weights on every
    /// call.
    ///
    /// # Safety
    /// `lo` and `hi` must each be readable for 8 bytes.
    #[inline]
    pub unsafe fn combine_rows(lo: *const i8, hi: *const i8) -> int8x16_t {
        vcombine_s8(vld1_s8(lo), vld1_s8(hi))
    }

    #[inline]
    pub unsafe fn zero() -> int32x4_t {
        vdupq_n_s32(0)
    }

    #[inline]
    pub unsafe fn add4(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        vaddq_s32(a, b)
    }

    #[inline]
    pub unsafe fn lanes(v: int32x4_t) -> [i32; 4] {
        let mut out = [0i32; 4];
        vst1q_s32(out.as_mut_ptr(), v);
        out
    }

    #[inline]
    pub unsafe fn load16(p: *const i8) -> int8x16_t {
        vld1q_s8(p)
    }

    /// Does this CPU have `dotprod`, and so `SDOT`? Checked once, then cached.
    ///
    /// Armv8.2 and universal on Apple Silicon, but not on every 64-bit ARM —
    /// and `KVAD_NO_DOTPROD` forces the portable path, which is how the
    /// kernel below gets measured against what it replaced rather than
    /// assumed to beat it.
    pub fn has_dotprod() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            if std::env::var_os("KVAD_NO_DOTPROD").is_some() {
                return false;
            }
            std::arch::is_aarch64_feature_detected!("dotprod")
        })
    }

    /// One block: sixteen packed bytes against thirty-two activations.
    ///
    /// The nibbles come out unsigned, 0 to 15, and the `-8` that makes them
    /// real weights is applied here — two vector ops for thirty-two of them.
    /// That is the whole trick, and it is why this function exists at all:
    /// `SDOT` needs both operands signed, and a masked nibble widened from
    /// `u8` is a zero-extend.
    ///
    /// # Safety
    /// `w` must be readable for 16 bytes and `x` for 32.
    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn block_dot(w: *const u8, x: *const i8) -> int32x4_t {
        let packed = vld1q_u8(w);
        let bias = vdupq_n_s8(-8);
        let lo = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f))), bias);
        let hi = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), bias);
        // Low nibbles against the first half of the block, high against the
        // second — the split packing, which exists so that this is two whole
        // vectors and not an interleave.
        let acc = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x));
        vdotq_s32(acc, hi, vld1q_s8(x.add(16)))
    }

    /// Per-block dots for one 4-bit weight row, signed and already unbiased.
    ///
    /// # Why this is written and not inferred
    ///
    /// The portable kernel multiplies *unsigned* nibbles and takes the bias
    /// off once per block afterwards, because that is the only formulation
    /// that vectorises at all. Every attempt to make the inner loop signed —
    /// which is what `SDOT` needs — made LLVM abandon the reduction and
    /// vectorise *across* blocks instead, loading weights one byte at a time
    /// into lanes with `LD1.B`. Measured on one core over a 896-wide row:
    ///
    /// ```text
    ///   unsigned nibbles, bias per block     24 GMAC/s   (the portable path)
    ///   signed nibbles, same iterator form    4 GMAC/s
    ///   signed, one fused accumulator         7 GMAC/s
    ///   signed, block sizes as array types    2 GMAC/s
    ///   this kernel                          80 GMAC/s
    ///   q8, for scale                        88 GMAC/s
    /// ```
    ///
    /// Written out, unpacking nibbles costs 9% against not having to — which
    /// is the number the 4-bit format deserves, and about a fifth of what the
    /// autovectoriser was charging for it.
    ///
    /// Four blocks per iteration: `VPADD` reduces four accumulators in three
    /// instructions where four `ADDV`s cost one apiece, and `ADDV` is the
    /// longest-latency thing in the loop.
    ///
    /// # Safety
    /// The caller must have checked [`has_dotprod`]. `packed` must be
    /// `16 * out.len()` bytes and `acts` `32 * out.len()`.
    #[target_feature(enable = "dotprod")]
    pub unsafe fn q4_row_dots(packed: &[u8], acts: &[i8], out: &mut [i32]) {
        debug_assert_eq!(packed.len(), 16 * out.len());
        debug_assert_eq!(acts.len(), 32 * out.len());
        let (w, x, n) = (packed.as_ptr(), acts.as_ptr(), out.len());

        let mut b = 0;
        while b + 4 <= n {
            let a0 = block_dot(w.add(16 * b), x.add(32 * b));
            let a1 = block_dot(w.add(16 * b + 16), x.add(32 * b + 32));
            let a2 = block_dot(w.add(16 * b + 32), x.add(32 * b + 64));
            let a3 = block_dot(w.add(16 * b + 48), x.add(32 * b + 96));
            // [sum(a0), sum(a1)] then [sum(a2), sum(a3)], then all four.
            let ab = vpaddq_s32(a0, a1);
            let cd = vpaddq_s32(a2, a3);
            vst1q_s32(out.as_mut_ptr().add(b), vpaddq_s32(ab, cd));
            b += 4;
        }
        while b < n {
            out[b] = vaddvq_s32(block_dot(w.add(16 * b), x.add(32 * b)));
            b += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(not(target_arch = "aarch64"))]
pub fn has_i8mm() -> bool {
    false
}

#[cfg(not(target_arch = "aarch64"))]
pub fn has_dotprod() -> bool {
    false
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn smmla_computes_a_two_by_two_of_eight_long_dots() {
        if !has_i8mm() {
            eprintln!("i8mm not available on this CPU; skipping");
            return;
        }
        // A0 = 1..8, A1 = all ones, B0 = all ones, B1 = 0,...,0,10
        let mut a = [0i8; 16];
        for i in 0..8 {
            a[i] = (i + 1) as i8;
        }
        for i in 8..16 {
            a[i] = 1;
        }
        let mut b = [0i8; 16];
        for i in 0..8 {
            b[i] = 1;
        }
        b[15] = 10;

        let out = unsafe {
            let r = smmla(zero(), load16(a.as_ptr()), load16(b.as_ptr()));
            lanes(r)
        };
        // [A0·B0, A0·B1, A1·B0, A1·B1]
        assert_eq!(out, [36, 80, 8, 10]);
    }

    /// The 4-bit kernel against longhand, on a row long enough to exercise
    /// both of its loops.
    ///
    /// Nine blocks: two passes of the four-at-a-time body and a tail of one.
    /// A kernel that handled only whole groups of four would pass a test with
    /// eight blocks and silently drop the ninth, and a wrong `VPADD` order
    /// would put block 1's dot in block 2's slot — which no aggregate check
    /// downstream would notice, because every block's scale is close to every
    /// other's.
    #[test]
    fn the_nibble_kernel_matches_longhand() {
        if !has_dotprod() {
            eprintln!("dotprod not available or disabled; skipping");
            return;
        }
        const BLOCK: usize = 32;
        let blocks = 9;
        // Nothing random: a pattern that puts every nibble value somewhere,
        // and activations that differ block to block.
        let packed: Vec<u8> = (0..blocks * BLOCK / 2).map(|i| (i * 37 + 11) as u8).collect();
        let acts: Vec<i8> = (0..blocks * BLOCK).map(|i| ((i * 13) % 251) as i8 - 125).collect();

        let mut want = vec![0i32; blocks];
        for (b, d) in want.iter_mut().enumerate() {
            for j in 0..BLOCK / 2 {
                let byte = packed[b * BLOCK / 2 + j];
                let x = &acts[b * BLOCK..];
                *d += ((byte & 0x0f) as i32 - 8) * x[j] as i32;
                *d += ((byte >> 4) as i32 - 8) * x[BLOCK / 2 + j] as i32;
            }
        }

        let mut got = vec![0i32; blocks];
        unsafe { q4_row_dots(&packed, &acts, &mut got) };
        assert_eq!(got, want);
    }
}
