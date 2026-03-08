// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Uniform Distribution SIMD Implementations**
//!
//! High-performance SIMD-accelerated implementations of uniform distribution functions optimised
//! for Monte Carlo simulation, random sampling, and statistical modelling applications requiring
//! flat probability distributions over bounded intervals.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Select, Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use minarrow::{Bitmask, FloatArray, Vec64, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;

/// SIMD-accelerated uniform PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_pdf_simd_to(
    x: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_pdf: a must be < b and finite".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let inv_width = 1.0 / (b - a);

    let simd_body = move |x_v: Simd<f64, N>| -> Simd<f64, N> {
        let a_v = Simd::<f64, N>::splat(a);
        let b_v = Simd::<f64, N>::splat(b);
        let inv_v = Simd::<f64, N>::splat(inv_width);
        let zero = Simd::<f64, N>::splat(0.0);
        let is_nan = x_v.simd_ne(x_v);
        let in_range = x_v.simd_ge(a_v) & x_v.simd_le(b_v);
        let result = in_range.select(inv_v, zero);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
    };

    let scalar_body = move |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= a && xi <= b {
            inv_width
        } else {
            0.0
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("uniform_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// **Uniform Distribution Probability Density Function**
///
/// Computes the probability density function of the continuous uniform distribution using vectorised
/// SIMD operations where possible, with scalar fallback for compatibility.
#[inline(always)]
pub fn uniform_pdf_simd(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    uniform_pdf_simd_to(x, a, b, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated uniform CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_cdf_simd_to(
    x: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_cdf: a must be < b and both finite".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let inv_width = 1.0 / (b - a);

    let scalar_body = |xi: f64| -> f64 {
        if xi.is_nan() {
            f64::NAN
        } else if xi < a {
            0.0
        } else if xi > b {
            1.0
        } else {
            (xi - a) * inv_width
        }
    };

    let simd_body = {
        let a_v = Simd::<f64, N>::splat(a);
        let b_v = Simd::<f64, N>::splat(b);
        let inv_v = Simd::<f64, N>::splat(inv_width);
        let one = Simd::<f64, N>::splat(1.0);
        let zero = Simd::<f64, N>::splat(0.0);
        move |x_v: Simd<f64, N>| -> Simd<f64, N> {
            let is_nan = x_v.simd_ne(x_v);
            let below = x_v.simd_lt(a_v);
            let above = x_v.simd_gt(b_v);
            let linear = (x_v - a_v) * inv_v;
            let result = below.select(zero, above.select(one, linear));
            is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("uniform_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// **Uniform Distribution Cumulative Distribution Function** - *SIMD-Accelerated Flat CDF*
///
/// Computes the cumulative distribution function of the continuous uniform distribution using vectorised
/// SIMD operations where possible, providing high-precision probability computation for bounded intervals.
#[inline(always)]
pub fn uniform_cdf_simd(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    uniform_cdf_simd_to(x, a, b, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated uniform quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_quantile_simd_to(
    p: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(a < b) || !a.is_finite() || !b.is_finite() {
        return Err(KernelError::InvalidArguments(
            "uniform_quantile: a must be < b and finite".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let width = b - a;

    let scalar_body = move |pi: f64| -> f64 {
        if (0.0..=1.0).contains(&pi) && pi.is_finite() {
            a + pi * width
        } else {
            f64::NAN
        }
    };

    let simd_body = {
        let a_v = Simd::<f64, N>::splat(a);
        let width_v = Simd::<f64, N>::splat(width);
        let zero = Simd::<f64, N>::splat(0.0);
        let one = Simd::<f64, N>::splat(1.0);
        move |p_v: Simd<f64, N>| {
            let valid = p_v.simd_ge(zero) & p_v.simd_le(one) & p_v.is_finite();
            let q = a_v + p_v * width_v;
            valid.select(q, Simd::splat(f64::NAN))
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(p) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(p, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("uniform_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(p) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            p,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// **Uniform Distribution Quantile Function** - *SIMD-Accelerated Inverse CDF*
///
/// Computes the quantile function (inverse CDF) of the continuous uniform distribution using vectorised
/// SIMD operations where possible, providing efficient random variate generation and percentile computation.
#[inline(always)]
pub fn uniform_quantile_simd(
    p: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    uniform_quantile_simd_to(p, a, b, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
