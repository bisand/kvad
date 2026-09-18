//! Hand-written SIMD, for the one instruction the compiler will not reach on
//! its own.
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
            // Setting LLM_NO_I8MM forces the SDOT path, which is how the
            // contribution of this one instruction gets measured rather than
            // assumed.
            if std::env::var_os("LLM_NO_I8MM").is_some() {
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
}

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(not(target_arch = "aarch64"))]
pub fn has_i8mm() -> bool {
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
}
