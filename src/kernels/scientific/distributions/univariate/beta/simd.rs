// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Beta Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of beta distribution functions
//! utilising vectorised operations for bulk computations on 64-byte aligned data.

use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray, Vec64};
use std::simd::num::SimdFloat;
use std::simd::prelude::{SimdPartialEq, SimdPartialOrd};
use std::simd::{Simd, StdFloat};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// SIMD-accelerated implementation of beta distribution PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn beta_pdf_simd_to(
    x: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if alpha <= 0.0 || beta <= 0.0 || !alpha.is_finite() || !beta.is_finite() {
        return Err(KernelError::InvalidArguments(
            "beta_pdf: invalid alpha or beta".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let log_norm = -ln_gamma(alpha) - ln_gamma(beta) + ln_gamma(alpha + beta);

    let alpha_v = Simd::<f64, N>::splat(alpha);
    let beta_v = Simd::<f64, N>::splat(beta);
    let log_norm_v = Simd::<f64, N>::splat(log_norm);

    let scalar_body = |xi: f64| -> f64 {
        if xi < 0.0 || xi > 1.0 {
            0.0
        } else if xi == 0.0 || xi == 1.0 {
            // Handle boundary cases
            if (alpha < 1.0 && xi == 0.0) || (beta < 1.0 && xi == 1.0) {
                f64::INFINITY
            } else if (alpha > 1.0 && xi == 0.0) || (beta > 1.0 && xi == 1.0) {
                0.0
            } else if alpha == 1.0 && xi == 0.0 {
                beta
            } else if beta == 1.0 && xi == 1.0 {
                alpha
            } else {
                // alpha == 1.0 && beta == 1.0, uniform distribution
                1.0
            }
        } else {
            ((alpha - 1.0) * xi.ln() + (beta - 1.0) * (1.0 - xi).ln() + log_norm).exp()
        }
    };

    let simd_body = |x_v: Simd<f64, N>| {
        let zero = Simd::<f64, N>::splat(0.0);
        let one = Simd::<f64, N>::splat(1.0);
        let nan = Simd::<f64, N>::splat(f64::NAN);
        let inf = Simd::<f64, N>::splat(f64::INFINITY);

        // Check for NaN inputs (from masked values)
        let is_nan = x_v.simd_ne(x_v); // NaN != NaN

        // Compute the main PDF for interior points
        let ln_x = x_v.ln();
        let ln_1mx = (one - x_v).ln();
        let pow = (alpha_v - Simd::splat(1.0)) * ln_x + (beta_v - Simd::splat(1.0)) * ln_1mx;
        let log_pdf = log_norm_v + pow;
        let pdf = log_pdf.exp();

        // Masks for different regions
        let in_bounds = (x_v.simd_gt(zero)) & (x_v.simd_lt(one));
        let at_zero = x_v.simd_eq(zero);
        let at_one = x_v.simd_eq(one);

        // Handle boundary cases at x=0
        let alpha_lt_1 = alpha_v.simd_lt(Simd::splat(1.0));
        let alpha_eq_1 = alpha_v.simd_eq(Simd::splat(1.0));
        let at_zero_result = alpha_lt_1.select(inf, alpha_eq_1.select(beta_v, zero));

        // Handle boundary cases at x=1
        let beta_lt_1 = beta_v.simd_lt(Simd::splat(1.0));
        let beta_eq_1 = beta_v.simd_eq(Simd::splat(1.0));
        let at_one_result = beta_lt_1.select(inf, beta_eq_1.select(alpha_v, zero));

        // Combine all cases
        let mut result = in_bounds.select(pdf, zero);
        result = at_zero.select(at_zero_result, result);
        result = at_one.select(at_one_result, result);
        result = is_nan.select(nan, result);
        result
    };

    // Dense fast path - no nulls
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }

        // Scalar fallback - alignment check failed
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();

    // Check if arrays are 64-byte aligned for SIMD
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }

    // Scalar fallback - alignment check failed
    masked_univariate_kernel_f64_std_to(x, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// SIMD-accelerated implementation of beta distribution probability density function.
///
/// Processes multiple PDF evaluations simultaneously using vectorised operations
/// for improved performance on large datasets with 64-byte memory alignment.
#[inline(always)]
pub fn beta_pdf_simd(
    x: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    beta_pdf_simd_to(x, alpha, beta, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated implementation of beta distribution CDF (zero-allocation variant).
use core::simd::{Mask, Select};

/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn beta_cdf_simd_to(
    x: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if alpha <= 0.0 || beta <= 0.0 || !alpha.is_finite() || !beta.is_finite() {
        return Err(KernelError::InvalidArguments(
            "beta_cdf: invalid alpha or beta".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;

    #[inline]
    fn scalar_body(xi: f64, alpha: f64, beta: f64) -> f64 {
        if xi <= 0.0 {
            0.0
        } else if xi >= 1.0 {
            1.0
        } else {
            incomplete_beta(alpha, beta, xi)
        }
    }

    // Fast path with no nulls
    if !has_nulls(null_count, null_mask) {
        for (i, &xi) in x.iter().enumerate() {
            output[i] = scalar_body(xi, alpha, beta);
        }
        return Ok(());
    }

    // Null-aware masked SIMD path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();

    masked_univariate_kernel_f64_simd_to::<N, _, _>(
        x,
        mask,
        output,
        &mut out_mask,
        |x_v| {
            // Vectorised regularised incomplete beta with boundary handling.
            incomplete_beta_simd::<N>(alpha, beta, x_v)
        },
        |xi| scalar_body(xi, alpha, beta),
    );

    Ok(())
}

/// SIMD-accelerated implementation of beta distribution cumulative distribution function.
#[inline(always)]
pub fn beta_cdf_simd(
    x: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    beta_cdf_simd_to(x, alpha, beta, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Vectorised regularised incomplete beta I_x(a,b).
/// Handles x<=0 -> 0 and x>=1 -> 1 via masking; computes middle region with Lentz CF.
#[inline(always)]
fn incomplete_beta_simd<const N: usize>(a: f64, b: f64, x: Simd<f64, N>) -> Simd<f64, N>
where
{
    let zero = Simd::splat(0.0);
    let one = Simd::splat(1.0);

    // Masks for edge lanes
    let m_lo = x.simd_le(zero);
    let m_hi = x.simd_ge(one);
    let m_mid = !(m_lo | m_hi);

    // Clamp for safe logs on middle lanes only; dummy 0.5 elsewhere (masked out later).
    let eps_low = Simd::splat(1.0e-300);
    let eps_high = Simd::splat(1.0 - 1.0e-16);
    let x_clamped = {
        let xc = x.simd_max(eps_low).simd_min(eps_high);
        m_mid.select(xc, Simd::splat(0.5))
    };

    // Symmetry decision (Numerical Recipes threshold)
    let thresh = (a + 1.0) / (a + b + 2.0);
    let use_direct = x_clamped.simd_lt(Simd::splat(thresh));

    // Precompute scalar ln Beta(a,b)
    let ln_beta_ab = ln_beta_scalar(a, b);

    // Direct branch: I_x(a,b) = front(a,b,x) * betacf(a,b,x)
    let res_direct = {
        let front = front_factor::<N>(a, b, x_clamped, ln_beta_ab, /*div=*/ a);
        let cf = betacf_simd::<N>(a, b, x_clamped);
        front * cf
    };

    // Complementary branch: 1 - front(b,a,1-x) * betacf(b,a,1-x)
    let res_compl = {
        let xm = one - x_clamped;
        let front = front_factor::<N>(b, a, xm, ln_beta_ab, /*div=*/ b);
        let cf = betacf_simd::<N>(b, a, xm);
        one - front * cf
    };

    // Select branch per-lane, then apply boundary masks.
    let mid = use_direct.select(res_direct, res_compl);
    let mut y = Simd::splat(0.0);
    y = m_mid.select(mid, y);
    y = m_hi.select(one, y);
    y
}

/// front = exp(a*ln(x) + b*ln(1-x) - lnB(a,b)) / div
#[inline(always)]
fn front_factor<const N: usize>(
    a: f64,
    b: f64,
    x: Simd<f64, N>,
    ln_beta_ab: f64,
    div: f64,
) -> Simd<f64, N>
where
{
    // Implement ln/exp using elementwise ops via std::simd intrinsic methods.
    // If your toolchain lacks ln/exp on Simd, provide them via a SIMD math crate or polynomial approximations.
    let one = Simd::splat(1.0);
    let t = Simd::splat(a) * x.ln() + Simd::splat(b) * (one - x).ln() - Simd::splat(ln_beta_ab);
    t.exp() / Simd::splat(div)
}

/// Vectorised Modified Lentz continued fraction for the incomplete beta.
/// Returns the continued fraction value; caller multiplies by the front factor.
#[inline(always)]
fn betacf_simd<const N: usize>(a: f64, b: f64, x: Simd<f64, N>) -> Simd<f64, N>
where
{
    // Constants tuned for f64
    const MAX_ITERS: i32 = 200;
    let eps = Simd::splat(3.0e-14);
    let one = Simd::splat(1.0);
    let zero = Simd::splat(0.0);
    let fpmin = Simd::splat(1.0e-300);

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    // Initialisations
    let mut c = one;
    let mut d = one - Simd::splat(qab) * x / Simd::splat(qap);
    // Safeguard against underflow
    d = d.simd_lt(fpmin).select(fpmin, d);
    d = one / d;
    let mut h = d;

    // Track convergence per lane
    let mut converged: Mask<i64, N> = Mask::splat(false);

    let mut m = 1;
    while m <= MAX_ITERS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let aa_even = m_f * (b - m_f);
        let denom_even = (qam + m2) * (a + m2);
        let aa_even_v = Simd::splat(aa_even / denom_even) * x;

        let mut d_new = one + aa_even_v * d;
        d_new = d_new.simd_lt(fpmin).select(fpmin, d_new);
        d_new = one / d_new;

        let mut c_new = one + aa_even_v / c;
        c_new = c_new.simd_lt(fpmin).select(fpmin, c_new);

        let h_new = h * d_new * c_new;

        // Odd step
        let aa_odd = -((a + m_f) * (qab + m_f));
        let denom_odd = (a + m2) * (qap + m2);
        let aa_odd_v = Simd::splat(aa_odd / denom_odd) * x;

        let mut d2 = one + aa_odd_v * d_new;
        d2 = d2.simd_lt(fpmin).select(fpmin, d2);
        d2 = one / d2;

        let mut c2 = one + aa_odd_v / c_new;
        c2 = c2.simd_lt(fpmin).select(fpmin, c2);

        let del = d2 * c2;
        let h2 = h_new * del;

        // Convergence check: |del - 1| < eps
        let e = del - one;
        let abs_e = e.simd_lt(zero).select(-e, e);
        let just_converged = abs_e.simd_lt(eps);

        // Update lanes that haven't converged yet
        let upd_mask = !converged;
        d = upd_mask.select(d2, d);
        c = upd_mask.select(c2, c);
        h = upd_mask.select(h2, h);

        converged |= just_converged;
        if converged.all() {
            break;
        }

        m += 1;
    }

    h
}

/// ln(Beta(a,b)) using scalar Lanczos log-gamma; computed once since a,b are scalars.
#[inline(always)]
fn ln_beta_scalar(a: f64, b: f64) -> f64 {
    ln_gamma_scalar(a) + ln_gamma_scalar(b) - ln_gamma_scalar(a + b)
}

/// Scalar Lanczos log-gamma with reflection for stability.
#[inline(always)]
fn ln_gamma_scalar(z: f64) -> f64 {
    // Coefficients for g=7, n=9 (from Numerical Recipes)
    const COF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        0.000009984369578019572,
        0.00000015056327351493116,
    ];
    const LOG_SQRT_2PI: f64 = 0.91893853320467274178032973640562; // ln(sqrt(2π))

    if z < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        return std::f64::consts::PI.ln() - (pi * z).sin().ln() - ln_gamma_scalar(1.0 - z);
    }

    let z1 = z - 1.0;
    let mut x = COF[0];
    for (i, &c) in COF.iter().enumerate().skip(1) {
        x += c / (z1 + i as f64);
    }
    let t = z1 + 7.5;
    LOG_SQRT_2PI + (z1 + 0.5) * t.ln() - t + x.ln()
}
