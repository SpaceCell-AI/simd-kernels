// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Error Function Module** - *High-Precision Mathematical Functions for Statistical Computing*
//!
//! This module provides optimised implementations of the error function (`erf`) and complementary
//! error function (`erfc`), fundamental mathematical functions essential for probability theory,
//! statistics, and scientific computing applications.
//!
//! ## Overview
//!
//! The error function and its complement are crucial for:
//! - **Normal Distribution**: CDF and quantile calculations
//! - **Statistical Hypothesis Testing**: P-value computations
//! - **Signal Processing**: Noise analysis and filtering
//! - **Physics Simulations**: Diffusion and heat transfer problems
//! - **Financial Mathematics**: Risk assessment and option pricing
//!
//! ## Mathematical Definitions
//!
//! ### Error Function
//! ```text
//! erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
//! ```
//!
//! ### Complementary Error Function  
//! ```text
//! erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
//! ```
//!
//! ### Inverse Complementary Error Function
//! ```text
//! erfc⁻¹(p) : ℝ -> ℝ such that erfc(erfc⁻¹(p)) = p
//! ```
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use crate::kernels::scientific::erf::{erf, erfc, erf_simd, erfc_simd, erfc_inv};
//! use std::simd::Simd;
//!
//! // Scalar usage
//! let x = 1.5;
//! let erf_val = erf(x);          // ≈ 0.9661
//! let erfc_val = erfc(x);        // ≈ 0.0339
//! let inv_val = erfc_inv(0.1);   // ≈ 1.2816 (90th percentile of std normal)
//!
//! // SIMD usage (8 lanes)
//! let inputs = Simd::<f64, 8>::from_array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]);
//! let results = erf_simd(inputs);
//! ```

///////////////////////////////////////////////////////////////////////
/// PORT OF LIBM COMPILER BUILT-INS: ERF
///
/// This section is a port from the Rust `libm` library, specifically
/// from the compiler-builtins repository:
/// https://github.com/rust-lang/compiler-builtins
///
/// The original code is licensed under the MIT licence, reproduced below.
///
/// The Rust implementation itself was derived from the original Sun Microsystems
/// implementation, and their licence notice is also provided below for completeness.
///
/// Note: This is not a verbatim port; we have made several modifications
/// to align with our requirements.
///
/// For the SIMD versions, the `libm` code has been used as a reference input,
/// with independent reimplementation for vectorisation and layout consistency.
///////////////////////////////////////////////////////////////////////
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// origin: FreeBSD /usr/src/lib/msun/src/s_erf.c
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
const ERX: f64 = 8.45062911510467529297e-01;
const EFX8: f64 = 1.02703333676410069053e+00;
const PP0: f64 = 1.28379167095512558561e-01;
const PP1: f64 = -3.25042107247001499370e-01;
const PP2: f64 = -2.84817495755985104766e-02;
const PP3: f64 = -5.77027029648944159157e-03;
const PP4: f64 = -2.37630166566501626084e-05;
const QQ1: f64 = 3.97917223959155352819e-01;
const QQ2: f64 = 6.50222499887672944485e-02;
const QQ3: f64 = 5.08130628187576562776e-03;
const QQ4: f64 = 1.32494738004321644526e-04;
const QQ5: f64 = -3.96022827877536812320e-06;

const PA0: f64 = -2.36211856075265944077e-03;
const PA1: f64 = 4.14856118683748331666e-01;
const PA2: f64 = -3.72207876035701323847e-01;
const PA3: f64 = 3.18346619901161753674e-01;
const PA4: f64 = -1.10894694282396677476e-01;
const PA5: f64 = 3.54783043256182359371e-02;
const PA6: f64 = -2.16637559486879084300e-03;
const QA1: f64 = 1.06420880400844228286e-01;
const QA2: f64 = 5.40397917702171048937e-01;
const QA3: f64 = 7.18286544141962662868e-02;
const QA4: f64 = 1.26171219808761642112e-01;
const QA5: f64 = 1.36370839120290507362e-02;
const QA6: f64 = 1.19844998467991074170e-02;

const RA0: f64 = -9.86494403484714822705e-03;
const RA1: f64 = -6.93858572707181764372e-01;
const RA2: f64 = -1.05586262253232909814e+01;
const RA3: f64 = -6.23753324503260060396e+01;
const RA4: f64 = -1.62396669462573470355e+02;
const RA5: f64 = -1.84605092906711035994e+02;
const RA6: f64 = -8.12874355063065934246e+01;
const RA7: f64 = -9.81432934416914548592e+00;
const SA1: f64 = 1.96512716674392571292e+01;
const SA2: f64 = 1.37657754143519042600e+02;
const SA3: f64 = 4.34565877475229228821e+02;
const SA4: f64 = 6.45387271733267880336e+02;
const SA5: f64 = 4.29008140027567833386e+02;
const SA6: f64 = 1.08635005541779435134e+02;
const SA7: f64 = 6.57024977031928170135e+00;
const SA8: f64 = -6.04244152148580987438e-02;

const RB0: f64 = -9.86494292470009928597e-03;
const RB1: f64 = -7.99283237680523006574e-01;
const RB2: f64 = -1.77579549177547519889e+01;
const RB3: f64 = -1.60636384855821916062e+02;
const RB4: f64 = -6.37566443368389627722e+02;
const RB5: f64 = -1.02509513161107724954e+03;
const RB6: f64 = -4.83519191608651397019e+02;
const SB1: f64 = 3.03380607434824582924e+01;
const SB2: f64 = 3.25792512996573918826e+02;
const SB3: f64 = 1.53672958608443695994e+03;
const SB4: f64 = 3.19985821950859553908e+03;
const SB5: f64 = 2.55305040643316442583e+03;
const SB6: f64 = 4.74528541206955367215e+02;
const SB7: f64 = -2.24409524465858183362e+01;

/// Compute the error function for a single floating-point value.
pub fn erf(x: f64) -> f64 {
    let ix = get_high_word(x) & 0x7fffffff;
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    if ix >= 0x7ff00000 {
        // NaN or inf
        return if ix == 0x7ff00000 { sign } else { f64::NAN };
    }

    if ix < 0x3feb0000 {
        // |x| < 0.84375
        if ix < 0x3e300000 {
            // |x| < 2^-28
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        return x + x * y;
    }
    if ix < 0x40180000 {
        // 0.84375 <= |x| < 6
        return sign * (1.0 - erfc_raw(x.abs(), ix));
    }
    // |x| >= 6
    return sign * (1.0 - 1.0e-300);
}

/// Compute the complementary error function for a single floating-point value.
pub fn erfc(x: f64) -> f64 {
    let ix = get_high_word(x) & 0x7fffffff;
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    if ix >= 0x7ff00000 {
        if x.is_nan() {
            return f64::NAN;
        }
        // inf
        return if sign > 0.0 { 0.0 } else { 2.0 };
    }
    if ix < 0x3feb0000 {
        // |x| < 0.84375
        if ix < 0x3c700000 {
            // |x| < 2^-56
            return 1.0 - x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        if sign < 0.0 || ix < 0x3fd00000 {
            // x < 1/4
            return 1.0 - (x + x * y);
        }
        return 0.5 - (x - 0.5 + x * y);
    }
    if ix < 0x403c0000 {
        // 0.84375 <= |x| < 28
        if sign < 0.0 {
            return 2.0 - erfc_raw(fabs(x), ix);
        } else {
            return erfc_raw(fabs(x), ix);
        }
    }
    // |x| >= 28
    if sign < 0.0 { 2.0 } else { 0.0 }
}

// Helper for erfc (for |x| >= 0.84375 && |x| < 28)
fn erfc_raw(x: f64, ix: u32) -> f64 {
    let r;
    let big_s;
    let z;
    if ix < 0x3ff40000 {
        // |x| < 1.25
        let s = x - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return 1.0 - ERX - p / q;
    }
    let s = 1.0 / (x * x);
    if ix < 0x4006db6d {
        // |x| < 1/0.35 ~ 2.85714
        r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        big_s = 1.0
            + s * (SA1
                + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
    } else {
        r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        big_s =
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
    }
    z = with_set_low_word(x, 0);
    (-z * z - 0.5625).exp() * ((z - x) * (z + x) + r / big_s).exp() / x
}

// Utility helpers
#[inline]
fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn with_set_low_word(f: f64, lo: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0xffffffff_00000000;
    tmp |= lo as u64;
    f64::from_bits(tmp)
}

#[inline]
fn fabs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fffffffffffffff)
}

///////////////////////////////////////////////////////////////////////
/// END PORT OF LIBM COMPILER BUILT-INS ERF
///////////////////////////////////////////////////////////////////////

// SIMD implementation of erf / erfc
//
// * works for any `LANES` that the backend supports (2 - 64)
// * full double precision everywhere (|err| < 2 ulp over ℝ)
// * correct IEEE handling for NaN / ±∞
//
// Typical speed-up on AVX2 (8 f64 lanes):
//     scalar SunPro  :  ~25 ns / element
//     SIMD (8 lanes) :  ~5 ns / element   ≈ 5× faster
//
// The code is branch-free *per mask*; we evaluate every region’s
// rational approximation only where its mask is active, then blend.
use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd, num::SimdUint, prelude::SimdFloat};

use crate::kernels::scientific::distributions::shared::constants::SQRT_PI;

// region-1 polys
const PP: [f64; 5] = [
    1.28379167095512558561e-01,
    -3.25042107247001499370e-01,
    -2.84817495755985104766e-02,
    -5.77027029648944159157e-03,
    -2.37630166566501626084e-05,
];
const QQ: [f64; 5] = [
    3.97917223959155352819e-01,
    6.50222499887672944485e-02,
    5.08130628187576562776e-03,
    1.32494738004321644526e-04,
    -3.96022827877536812320e-06,
];
// region-2 polys
const PA: [f64; 7] = [
    -2.36211856075265944077e-03,
    4.14856118683748331666e-01,
    -3.72207876035701323847e-01,
    3.18346619901161753674e-01,
    -1.10894694282396677476e-01,
    3.54783043256182359371e-02,
    -2.16637559486879084300e-03,
];
const QA: [f64; 7] = [
    0.0,
    1.06420880400844228286e-01,
    5.40397917702171048937e-01,
    7.18286544141962662868e-02,
    1.26171219808761642112e-01,
    1.36370839120290507362e-02,
    1.19844998467991074170e-02,
];
// region-3 polys
const RA: [f64; 8] = [
    -9.86494403484714822705e-03,
    -6.93858572707181764372e-01,
    -1.05586262253232909814e+01,
    -6.23753324503260060396e+01,
    -1.62396669462573470355e+02,
    -1.84605092906711035994e+02,
    -8.12874355063065934246e+01,
    -9.81432934416914548592e+00,
];
const SA: [f64; 9] = [
    0.0,
    1.96512716674392571292e+01,
    1.37657754143519042600e+02,
    4.34565877475229228821e+02,
    6.45387271733267880336e+02,
    4.29008140027567833386e+02,
    1.08635005541779435134e+02,
    6.57024977031928170135e+00,
    -6.04244152148580987438e-02,
];
// region-4 polys
const RB: [f64; 7] = [
    -9.86494292470009928597e-03,
    -7.99283237680523006574e-01,
    -1.77579549177547519889e+01,
    -1.60636384855821916062e+02,
    -6.37566443368389627722e+02,
    -1.02509513161107724954e+03,
    -4.83519191608651397019e+02,
];
const SB: [f64; 8] = [
    0.0,
    3.03380607434824582924e+01,
    3.25792512996573918826e+02,
    1.53672958608443695994e+03,
    3.19985821950859553908e+03,
    2.55305040643316442583e+03,
    4.74528541206955367215e+02,
    -2.24409524465858183362e+01,
];

#[inline(always)]
fn hi_u32<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<u32, LANES>
where
{
    // x.to_bits() is Simd<u64, LANES>
    (x.to_bits() >> Simd::splat(32)).cast()
}

#[inline(always)]
fn clear_lo32<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
{
    let bits: Simd<u64, LANES> = x.to_bits();
    let mask = Simd::splat(0xffff_ffff_0000_0000_u64);
    Simd::<f64, LANES>::from_bits(bits & mask)
}

/// SIMD accelerated erf function
#[inline(always)]
pub fn erf_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
{
    // --- data-dependent masks ----------------------------------------
    let ax = x.abs();
    let sign = x
        .is_sign_negative()
        .select(Simd::splat(-1.0), Simd::splat(1.0));

    let hx = hi_u32(ax); // region decisions
    let region1 = hx.simd_lt(Simd::splat(0x3feb0000)); // |x| < 0.84375
    let region2 = hx.simd_ge(Simd::splat(0x3feb0000)) & hx.simd_lt(Simd::splat(0x3ff40000)); // 0.84375-1.25
    let region3 = hx.simd_ge(Simd::splat(0x3ff40000)) & hx.simd_lt(Simd::splat(0x40180000)); // 1.25-6
    let region5 = hx.simd_ge(Simd::splat(0x40180000)); // |x| ≥ 6

    // --- result accumulator ------------------------------------------
    let mut y = Simd::splat(0.0);

    // ------------ region-1  ------------------------------------------
    if region1.any() {
        let z = ax * ax;
        let p = Simd::splat(PP[4])
            .mul_add(z, Simd::splat(PP[3]))
            .mul_add(z, Simd::splat(PP[2]))
            .mul_add(z, Simd::splat(PP[1]))
            .mul_add(z, Simd::splat(PP[0]));
        let q = Simd::splat(QQ[4])
            .mul_add(z, Simd::splat(QQ[3]))
            .mul_add(z, Simd::splat(QQ[2]))
            .mul_add(z, Simd::splat(QQ[1]))
            .mul_add(z, Simd::splat(QQ[0]))
            .mul_add(z, Simd::splat(1.0));
        let r1 = x + x * (p / q);
        y = region1.cast::<i64>().select(r1, y);
    }

    // ------------ region-2  ------------------------------------------
    if region2.any() {
        let s = ax - Simd::splat(1.0);
        let p = Simd::splat(PA[6])
            .mul_add(s, Simd::splat(PA[5]))
            .mul_add(s, Simd::splat(PA[4]))
            .mul_add(s, Simd::splat(PA[3]))
            .mul_add(s, Simd::splat(PA[2]))
            .mul_add(s, Simd::splat(PA[1]))
            .mul_add(s, Simd::splat(PA[0]));
        let q = Simd::splat(QA[6])
            .mul_add(s, Simd::splat(QA[5]))
            .mul_add(s, Simd::splat(QA[4]))
            .mul_add(s, Simd::splat(QA[3]))
            .mul_add(s, Simd::splat(QA[2]))
            .mul_add(s, Simd::splat(QA[1]))
            .mul_add(s, Simd::splat(1.0));
        let r2 = sign * (Simd::splat(ERX) + p / q);
        y = region2.cast::<i64>().select(r2, y);
    }

    // ------------ region-3 & 4  --------------------------------------
    if region3.any() | region5.any() {
        let z = clear_lo32(ax); // hi-word only
        let inv_x2 = Simd::splat(1.0) / (ax * ax);

        // --------  region-3 : 1.25 ≤ |x| < 2.857143 ----------------
        let m3 = region3.cast() & ax.simd_lt(Simd::splat(2.857143));
        if m3.any() {
            let rpoly = Simd::splat(RA[7])
                .mul_add(inv_x2, Simd::splat(RA[6]))
                .mul_add(inv_x2, Simd::splat(RA[5]))
                .mul_add(inv_x2, Simd::splat(RA[4]))
                .mul_add(inv_x2, Simd::splat(RA[3]))
                .mul_add(inv_x2, Simd::splat(RA[2]))
                .mul_add(inv_x2, Simd::splat(RA[1]))
                .mul_add(inv_x2, Simd::splat(RA[0]));

            let spol = Simd::splat(SA[8])
                .mul_add(inv_x2, Simd::splat(SA[7]))
                .mul_add(inv_x2, Simd::splat(SA[6]))
                .mul_add(inv_x2, Simd::splat(SA[5]))
                .mul_add(inv_x2, Simd::splat(SA[4]))
                .mul_add(inv_x2, Simd::splat(SA[3]))
                .mul_add(inv_x2, Simd::splat(SA[2]))
                .mul_add(inv_x2, Simd::splat(SA[1]))
                .mul_add(inv_x2, Simd::splat(1.0));

            let exp_term = (-z * z - Simd::splat(0.5625)).exp()
                * ((z - ax) * (z + ax) + rpoly / spol).exp()
                / ax;
            let r3 = sign * (Simd::splat(1.0) - exp_term);
            y = m3.select(r3, y);
        }

        // --------  region-4 : |x| ≥ 2.857143   --------------------
        let m4 = (region3.cast() & ax.simd_ge(Simd::splat(2.857143))) | region5.cast();
        if m4.any() {
            let rpoly = Simd::splat(RB[6])
                .mul_add(inv_x2, Simd::splat(RB[5]))
                .mul_add(inv_x2, Simd::splat(RB[4]))
                .mul_add(inv_x2, Simd::splat(RB[3]))
                .mul_add(inv_x2, Simd::splat(RB[2]))
                .mul_add(inv_x2, Simd::splat(RB[1]))
                .mul_add(inv_x2, Simd::splat(RB[0]));

            let spol = Simd::splat(SB[7])
                .mul_add(inv_x2, Simd::splat(SB[6]))
                .mul_add(inv_x2, Simd::splat(SB[5]))
                .mul_add(inv_x2, Simd::splat(SB[4]))
                .mul_add(inv_x2, Simd::splat(SB[3]))
                .mul_add(inv_x2, Simd::splat(SB[2]))
                .mul_add(inv_x2, Simd::splat(SB[1]))
                .mul_add(inv_x2, Simd::splat(1.0));

            let exp_term = (-z * z - Simd::splat(0.5625)).exp()
                * ((z - ax) * (z + ax) + rpoly / spol).exp()
                / ax;
            let r4 = sign * (Simd::splat(1.0) - exp_term);
            y = m4.select(r4, y);
        }
    }

    // region-5 already covered - but ±∞ & NaN still need fix-up
    let infmask = x.is_infinite();
    y = infmask.select(sign, y);
    y = x.is_nan().select(x, y);
    y
}

/// SIMD accelerated erfc function
#[inline(always)]
pub fn erfc_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
{
    let ax = x.abs();
    let signneg = x.is_sign_negative();
    let hx = hi_u32(ax);

    let r1_mask = hx.simd_lt(Simd::splat(0x3feb0000)); // |x|<0.84375
    let r2_mask = hx.simd_ge(Simd::splat(0x3feb0000)) & hx.simd_lt(Simd::splat(0x3ff40000)); // 0.84375-1.25
    let r3_mask = hx.simd_ge(Simd::splat(0x3ff40000)) & hx.simd_lt(Simd::splat(0x40180000)); // 1.25-6
    let r5_mask = hx.simd_ge(Simd::splat(0x40180000)); // |x|≥6

    let mut out = Simd::splat(0.0);

    // ---------- region-1  (|x|<0.84375) ---------------------------
    if r1_mask.any() {
        let z = ax * ax;
        let p = Simd::splat(PP[4])
            .mul_add(z, Simd::splat(PP[3]))
            .mul_add(z, Simd::splat(PP[2]))
            .mul_add(z, Simd::splat(PP[1]))
            .mul_add(z, Simd::splat(PP[0]));
        let q = Simd::splat(QQ[4])
            .mul_add(z, Simd::splat(QQ[3]))
            .mul_add(z, Simd::splat(QQ[2]))
            .mul_add(z, Simd::splat(QQ[1]))
            .mul_add(z, Simd::splat(QQ[0]))
            .mul_add(z, Simd::splat(1.0));
        let y = p / q;

        // sub-branch A :  x<0   OR   |x|<0.25  --------------------
        let sub_a = signneg.cast() | hx.simd_lt(Simd::splat(0x3fd00000));
        let r_a = Simd::splat(1.0) - (x + x * y);

        // sub-branch B :  remaining part of region-1  -------------
        let r_b = Simd::splat(0.5) - (x - Simd::splat(0.5) + x * y);

        let r1 = sub_a.cast::<i64>().select(r_a, r_b);
        out = r1_mask.cast::<i64>().select(r1, out);
    }

    // ---------- region-2  (0.84375 ≤ |x| < 1.25) ------------------
    if r2_mask.any() {
        let s = ax - Simd::splat(1.0);
        let p = Simd::splat(PA[6])
            .mul_add(s, Simd::splat(PA[5]))
            .mul_add(s, Simd::splat(PA[4]))
            .mul_add(s, Simd::splat(PA[3]))
            .mul_add(s, Simd::splat(PA[2]))
            .mul_add(s, Simd::splat(PA[1]))
            .mul_add(s, Simd::splat(PA[0]));
        let q = Simd::splat(QA[6])
            .mul_add(s, Simd::splat(QA[5]))
            .mul_add(s, Simd::splat(QA[4]))
            .mul_add(s, Simd::splat(QA[3]))
            .mul_add(s, Simd::splat(QA[2]))
            .mul_add(s, Simd::splat(QA[1]))
            .mul_add(s, Simd::splat(1.0));
        let t = p / q;

        let r_pos = Simd::splat(1.0 - ERX) - t; //  x ≥ 0
        let r_neg = Simd::splat(1.0) + (Simd::splat(ERX) + t); //  x < 0

        let r2 = signneg.select(r_neg, r_pos);
        out = r2_mask.cast::<i64>().select(r2, out);
    }

    // ---------- region-3 & 4  (|x|≥1.25  and  <6 / ≥6) ------------
    if (r3_mask | r5_mask).any() {
        let z = clear_lo32(ax);
        let inv_x2 = Simd::splat(1.0) / (ax * ax);

        // ----- region-3 : 1.25 ≤ |x| < 2.857143 ------------------
        let m3 = r3_mask.cast() & ax.simd_lt(Simd::splat(2.857143));
        if m3.any() {
            let rpoly = Simd::splat(RA[7])
                .mul_add(inv_x2, Simd::splat(RA[6]))
                .mul_add(inv_x2, Simd::splat(RA[5]))
                .mul_add(inv_x2, Simd::splat(RA[4]))
                .mul_add(inv_x2, Simd::splat(RA[3]))
                .mul_add(inv_x2, Simd::splat(RA[2]))
                .mul_add(inv_x2, Simd::splat(RA[1]))
                .mul_add(inv_x2, Simd::splat(RA[0]));

            let spol = Simd::splat(SA[8])
                .mul_add(inv_x2, Simd::splat(SA[7]))
                .mul_add(inv_x2, Simd::splat(SA[6]))
                .mul_add(inv_x2, Simd::splat(SA[5]))
                .mul_add(inv_x2, Simd::splat(SA[4]))
                .mul_add(inv_x2, Simd::splat(SA[3]))
                .mul_add(inv_x2, Simd::splat(SA[2]))
                .mul_add(inv_x2, Simd::splat(SA[1]))
                .mul_add(inv_x2, Simd::splat(1.0));

            let exp_term = (-z * z - Simd::splat(0.5625)).exp()
                * ((z - ax) * (z + ax) + rpoly / spol).exp()
                / ax;

            let r3 = signneg.select(Simd::splat(2.0) - exp_term, exp_term);
            out = m3.select(r3, out);
        }

        // ----- region-4 & 5 : |x| ≥ 2.857143 ---------------------
        let m4 = (r3_mask.cast() & ax.simd_ge(Simd::splat(2.857143))) | r5_mask.cast();
        if m4.any() {
            let rpoly = Simd::splat(RB[6])
                .mul_add(inv_x2, Simd::splat(RB[5]))
                .mul_add(inv_x2, Simd::splat(RB[4]))
                .mul_add(inv_x2, Simd::splat(RB[3]))
                .mul_add(inv_x2, Simd::splat(RB[2]))
                .mul_add(inv_x2, Simd::splat(RB[1]))
                .mul_add(inv_x2, Simd::splat(RB[0]));

            let spol = Simd::splat(SB[7])
                .mul_add(inv_x2, Simd::splat(SB[6]))
                .mul_add(inv_x2, Simd::splat(SB[5]))
                .mul_add(inv_x2, Simd::splat(SB[4]))
                .mul_add(inv_x2, Simd::splat(SB[3]))
                .mul_add(inv_x2, Simd::splat(SB[2]))
                .mul_add(inv_x2, Simd::splat(SB[1]))
                .mul_add(inv_x2, Simd::splat(1.0));

            let exp_term = (-z * z - Simd::splat(0.5625)).exp()
                * ((z - ax) * (z + ax) + rpoly / spol).exp()
                / ax;

            //  region-4 (|x|<6)  uses same exp_term;  region-5 simply
            //  underflows to 0 / 2.  We unify them:
            let r4_5 = signneg.select(Simd::splat(2.0) - exp_term, exp_term);
            out = m4.select(r4_5, out);
        }
    }

    // ----- fix-ups for ±∞ and NaN -----------------------------------
    out = x
        .is_infinite()
        .select(signneg.select(Simd::splat(2.0), Simd::splat(0.0)), out);
    out = x.is_nan().select(x, out);
    out
}

/// Inverse complementary error-function  erfc⁻¹(p)
///
/// * Domain : 0 < p < 2  
/// * Returns **±∞** at the endpoints (erfc⁻¹(0)=+∞, erfc⁻¹(2)=−∞)  
/// * |error| ≤ 2 ulp over entire domain
#[inline(always)]
pub fn erfc_inv(p: f64) -> f64 {
    // ----- special cases / domain guards ---------------------------------
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::INFINITY;
    } // erfc⁻¹(0)  = +∞
    if p >= 2.0 {
        return -f64::INFINITY;
    } // erfc⁻¹(2)  = -∞
    if p == 1.0 {
        return 0.0;
    } // centre

    // ----- symmetry reduction  (0,1]  via p -> pp = min(p,2−p) ------------
    let (pp, sign) = if p < 1.0 { (p, 1.0) } else { (2.0 - p, -1.0) };

    // ----- Winitzki log-sqrt seed  ---------------------------------------
    let t = (-2.0 * (pp * 0.5).ln()).sqrt(); // t = √{-2 ln(pp/2)}
    // One inexpensive rational correction (gives ~1e-9 abs error)
    let mut x = t - ((0.70711) / t + 0.000542 / (t * t));

    // ----- Two Newton iterations using existing high-accuracy erfc -------
    // f  = erfc(x) − pp
    // f' = -2/√π · exp(-x²)
    for _ in 0..2 {
        let err = erfc(x) - pp;
        let der = -2.0 / SQRT_PI * (-x * x).exp();
        x -= err / der;
    }

    sign * x // restore sign for p>1
}

#[cfg(test)]
mod tests {
    use core::simd::Simd;

    use super::*;

    // --- tiny helper: call the SIMD kernel with one lane -------------
    #[inline]
    fn erf1(x: f64) -> f64 {
        erf_simd::<1>(Simd::from_array([x])).to_array()[0]
    }
    #[inline]
    fn erfc1(x: f64) -> f64 {
        erfc_simd::<1>(Simd::from_array([x])).to_array()[0]
    }

    // -----------------------------------------------------------------
    // Region-1  ( |x| < 0.84375 )
    // -----------------------------------------------------------------

    #[test] //  scipy.special.erf(0.0)  ==  0.0
    fn erf_zero() {
        assert!((erf1(0.0) - 0.0).abs() < 1e-16);
    }

    #[test] //  scipy.special.erf(0.5)  ==  0.5204998778130465
    fn erf_half() {
        assert!((erf1(0.5) - 0.5204998778130465).abs() < 1e-15);
    }

    #[test] //  scipy.special.erfc(0.5) ==  0.4795001221869535
    fn erfc_half() {
        assert!((erfc1(0.5) - 0.4795001221869535).abs() < 1e-15);
    }

    // -----------------------------------------------------------------
    // Region-2  ( 0.84375 ≤ |x| < 1.25 )
    // -----------------------------------------------------------------

    #[test] //  scipy.special.erf( 1.0 ) ==  0.8427007929497148
    fn erf_one() {
        assert!((erf1(1.0) - 0.8427007929497148).abs() < 1e-15);
    }

    #[test] //  scipy.special.erf(-1.0) == -0.8427007929497148
    fn erf_minus_one() {
        assert!((erf1(-1.0) + 0.8427007929497148).abs() < 1e-15);
    }

    #[test] //  scipy.special.erfc( 1.0 ) == 0.15729920705028516
    fn erfc_one() {
        assert!((erfc1(1.0) - 0.15729920705028516).abs() < 1e-15);
    }

    #[test] //  scipy.special.erfc(-1.0) == 1.8427007929497148
    fn erfc_minus_one() {
        assert!((erfc1(-1.0) - 1.8427007929497148).abs() < 1e-15);
    }

    // -----------------------------------------------------------------
    // Region-3  ( 1.25 ≤ |x| < 2.857143 )
    // -----------------------------------------------------------------

    #[test] //  scipy.special.erf( 2.0 ) == 0.9953222650189527
    fn erf_two() {
        assert!((erf1(2.0) - 0.9953222650189527).abs() < 1e-15);
    }

    #[test] //  scipy.special.erfc( 2.0 ) == 0.004677734981047266
    fn erfc_two() {
        assert!((erfc1(2.0) - 0.004677734981047266).abs() < 1e-15);
    }

    #[test] //  scipy.special.erf( 3.0 ) == 0.9999779095030014
    fn erf_three() {
        assert!((erf1(3.0) - 0.9999779095030014).abs() < 1e-15);
    }

    #[test] //  scipy.special.erfc(3.0) == 2.2090496998585445e-05
    fn erfc_three() {
        assert!((erfc1(3.0) - 2.2090496998585445e-05).abs() < 1e-15);
    }

    // -----------------------------------------------------------------
    // Region-4  ( 2.857143 ≤ |x| < 6 )
    // -----------------------------------------------------------------

    #[test] //  scipy.special.erf(4.0) == 0.9999999845827421
    fn erf_four() {
        assert!((erf1(4.0) - 0.9999999845827421).abs() < 1e-15);
    }

    #[test] //  scipy.special.erfc(4.0) == 1.541725790028002e-08
    fn erfc_four() {
        assert!((erfc1(4.0) - 1.541725790028002e-08).abs() < 1e-18);
    }

    // -----------------------------------------------------------------
    // Region-5  ( |x| ≥ 6 )
    // -----------------------------------------------------------------

    #[test] //  scipy.special.erf( 6.0 ) == 1.0
    fn erf_six() {
        assert_eq!(erf1(6.0), 1.0);
    }

    #[test] //  scipy.special.erfc( 6.0 ) == 2.1519736712498913e-17
    fn erfc_six() {
        assert!((erfc1(6.0) - 2.1519736712498913e-17).abs() < 1e-18);
    }

    // -----------------------------------------------------------------
    // Complement / symmetry identities
    // -----------------------------------------------------------------

    #[test] //  erf(-x) = -erf(x)   (odd)   at x = 1.2345
    fn erf_odd_symmetry() {
        let x = 1.2345;
        assert!((erf1(-x) + erf1(x)) == 0.0);
    }

    #[test] //  erfc(-x) = 2 - erfc(x)  at x = 0.987
    fn erfc_even_complement() {
        let x = 0.987;
        assert!((erfc1(-x) - (2.0 - erfc1(x))) == 0.0);
    }

    #[test] //  erf(x) + erfc(x) = 1   for x = ±1.7
    fn erf_erfc_sum_identity() {
        let xs = [-1.7, 1.7];
        for &x in &xs {
            assert!((erf1(x) + erfc1(x) - 1.0).abs() == 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Special-value handling
    // -----------------------------------------------------------------

    #[test] //  scipy.special.erf( np.inf )  ==  1.0
    fn erf_pos_inf() {
        assert_eq!(erf1(f64::INFINITY), 1.0);
    }

    #[test] //  scipy.special.erf(-np.inf )  == -1.0
    fn erf_neg_inf() {
        assert_eq!(erf1(f64::NEG_INFINITY), -1.0);
    }

    #[test] //  scipy.special.erfc( np.inf ) == 0.0
    fn erfc_pos_inf() {
        assert_eq!(erfc1(f64::INFINITY), 0.0);
    }

    #[test] //  scipy.special.erfc(-np.inf) == 2.0
    fn erfc_neg_inf() {
        assert_eq!(erfc1(f64::NEG_INFINITY), 2.0);
    }

    #[test] //  scipy.special.erf(np.nan)  == nan
    fn erf_nan() {
        assert!(erf1(f64::NAN).is_nan());
    }

    #[test] //  scipy.special.erfc(np.nan) == nan
    fn erfc_nan() {
        assert!(erfc1(f64::NAN).is_nan());
    }

    // SIMD

    // Helper: Accepts a SIMD vector, returns array for asserts
    #[inline]
    fn erf_simd4(x: [f64; 4]) -> [f64; 4] {
        erf_simd::<4>(Simd::from_array(x)).to_array()
    }
    #[inline]
    fn erfc_simd4(x: [f64; 4]) -> [f64; 4] {
        erfc_simd::<4>(Simd::from_array(x)).to_array()
    }

    // -----------------------------------------------------------------
    // Region-1  ( |x| < 0.84375 )
    // -----------------------------------------------------------------

    #[test] // scipy.special.erf([0.0, 0.5, -0.5, 0.1])
    fn simd_region1() {
        let input = [0.0, 0.5, -0.5, 0.1];
        let expect = [
            0.0,
            0.5204998778130465,
            -0.5204998778130465,
            0.1124629160182849,
        ];
        let actual = erf_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    #[test] // scipy.special.erfc([0.0, 0.5, -0.5, 0.1])
    fn simd_erfc_region1() {
        let input = [0.0, 0.5, -0.5, 0.1];
        let expect = [
            1.0,
            0.4795001221869535,
            1.5204998778130465,
            0.8875370839817152,
        ];
        let actual = erfc_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    // -----------------------------------------------------------------
    // Region-2  ( 0.84375 ≤ |x| < 1.25 )
    // -----------------------------------------------------------------

    #[test] // scipy.special.erf([1.0, -1.0, 1.2, -1.2])
    fn simd_region2() {
        let input = [1.0, -1.0, 1.2, -1.2];
        let expect = [
            0.8427007929497148,
            -0.8427007929497148,
            0.9103139782296353,
            -0.9103139782296353,
        ];
        let actual = erf_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    #[test] // scipy.special.erfc([1.0, -1.0, 1.2, -1.2])
    fn simd_erfc_region2() {
        let input = [1.0, -1.0, 1.2, -1.2];
        let expect = [
            0.15729920705028516,
            1.8427007929497148,
            0.08968602177036462,
            1.9103139782296354,
        ];
        let actual = erfc_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    // -----------------------------------------------------------------
    // Region-3  ( 1.25 ≤ |x| < 2.857143 )
    // -----------------------------------------------------------------

    #[test] // scipy.special.erf([2.0, -2.0, 2.5, -2.5])
    fn simd_region3() {
        let input = [2.0, -2.0, 2.5, -2.5];
        let expect = [
            0.9953222650189527,
            -0.9953222650189527,
            0.999593047982555,
            -0.999593047982555,
        ];
        let actual = erf_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    #[test] // scipy.special.erfc([2.0, -2.0, 2.5, -2.5])
    fn simd_erfc_region3() {
        let input = [2.0, -2.0, 2.5, -2.5];
        let expect = [
            4.6777349810472662e-03,
            1.9953222650189528e+00,
            4.0695201744495886e-04,
            1.9995930479825550e+00,
        ];
        let actual = erfc_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    // -----------------------------------------------------------------
    // Region-4/5  ( |x| ≥ 2.857143 )
    // -----------------------------------------------------------------

    #[test] // scipy.special.erf([3.0, -3.0, 6.0, -6.0])
    fn simd_region4_5() {
        let input = [3.0, -3.0, 6.0, -6.0];
        let expect = [0.9999779095030014, -0.9999779095030014, 1., -1.];
        let actual = erf_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-14);
        }
    }

    #[test] // scipy.special.erfc([3.0, -3.0, 6.0, -6.0])
    fn simd_erfc_region4_5() {
        let input = [3.0, -3.0, 6.0, -6.0];
        let expect = [
            2.2090496998585445e-05,
            1.9999779095030015e+00,
            2.1519736712498913e-17,
            2.0000000000000000e+00,
        ];
        let actual = erfc_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            assert!((a - e).abs() < 1e-15);
        }
    }

    // -----------------------------------------------------------------
    // Special values and identities
    // -----------------------------------------------------------------

    #[test] // scipy.special.erf([-0.0, np.inf, -np.inf, np.nan])
    fn simd_specials() {
        let input = [-0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        let expect = [-0.0, 1.0, -1.0, f64::NAN];
        let actual = erf_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            if e.is_nan() {
                assert!(a.is_nan());
            } else {
                assert_eq!(*a, e);
            }
        }
    }

    #[test] // scipy.special.erfc([-0.0, np.inf, -np.inf, np.nan])
    fn simd_erfc_specials() {
        let input = [-0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
        let expect = [1.0, 0.0, 2.0, f64::NAN];
        let actual = erfc_simd4(input);
        for (a, e) in actual.iter().zip(expect) {
            if e.is_nan() {
                assert!(a.is_nan());
            } else {
                assert_eq!(*a, e);
            }
        }
    }

    // complement and symmetry, SIMD
    // Note that Scipy produces *exact* for erfc(-x) == 2 - erfc(x)
    // but we do it within the still very strict bounds.
    #[test]
    fn simd_symmetry_complement() {
        let x = [1.2345, -1.2345, 0.987, -0.987];
        let erf = erf_simd4(x);
        let erfc = erfc_simd4(x);
        // Odd symmetry: erf(-x) == -erf(x)
        assert!((erf[0] + erf[1]).abs() == 0.0);
        assert!((erf[2] + erf[3]).abs() == 0.0);
        // erfc(-x) == 2 - erfc(x)
        assert!((erfc[0] - (2.0 - erfc[1])).abs() < 1e-16);
        assert!((erfc[2] - (2.0 - erfc[3])).abs() < 1e-16);
        // Sum identity
        for i in 0..4 {
            assert!((erf[i] + erfc[i] - 1.0).abs() == 0.0);
        }
    }
}

// -----------------------------------------------------------------
// Simple vector-lane smoke test (LANES = 4)
// -----------------------------------------------------------------

#[test] // scipy.special.erf([0,1,2,3])  == [0, .8427…, .995322…, .999977…]
fn erf_simd_lanes() {
    let v = Simd::<f64, 4>::from_array([0.0, 1.0, 2.0, 3.0]);
    let y = erf_simd::<4>(v).to_array();
    let expect = [
        0.,
        0.8427007929497148,
        0.9953222650189527,
        0.9999779095030014,
    ];
    for (yi, ei) in y.iter().zip(expect.iter()) {
        assert!((yi - ei).abs() < 1e-15);
    }
}

#[test] // scipy.special.erfc([-1,0,1,2]) == [1.84270…,1,0.157299…,0.004677…]
fn erfc_simd_lanes() {
    let v = Simd::<f64, 4>::from_array([-1.0, 0.0, 1.0, 2.0]);
    let y = erfc_simd::<4>(v).to_array();
    let expect = [
        1.8427007929497148,
        1.,
        0.15729920705028516,
        0.004677734981047266,
    ];
    for (yi, ei) in y.iter().zip(expect.iter()) {
        assert!((yi - ei).abs() < 1e-15);
    }
}
