// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Laplace Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of Laplace (double exponential)
//! distribution functions utilising vectorised operations for bulk computations on
//! symmetric exponential data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the Laplace distribution, characterised by its symmetric double-exponential
//! shape. The implementations automatically fall back to scalar versions when data
//! alignment requirements are not met.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};

use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

const N: usize = W64;

/// Laplace PDF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn laplace_pdf_simd_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "laplace_pdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let inv_b = 1.0 / scale;
    let norm = 0.5 * inv_b; // 1/(2b)

    // SIMD constants
    let loc_v = Simd::<f64, N>::splat(location);
    let inv_b_v = Simd::<f64, N>::splat(inv_b);
    let norm_v = Simd::<f64, N>::splat(norm);

    // Scalar fallback for one lane
    let scalar_body = |xi: f64| {
        let z = (xi - location).abs() * inv_b;
        norm * (-z).exp()
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v).abs() * inv_b_v;
        norm_v * (-z).exp()
    };

    // Dense fast path (no nulls)
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
    let mask_ref = null_mask.expect("laplace_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();

    // Check if arrays are SIMD 64-byte aligned
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

    // Scalar fallback - alignment check failed
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// SIMD-accelerated implementation of Laplace distribution probability density function.
///
/// Computes the probability density function (PDF) of the Laplace (double exponential)
/// distribution using vectorised SIMD operations for enhanced performance on
/// symmetric exponential data analysis.
///
/// ## Parameters
/// - `x`: Input values where PDF should be evaluated (domain: all real numbers)
/// - `location`: Location parameter μ (median and centre of distribution)
/// - `scale`: Scale parameter b > 0 (controls distribution spread)
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid location or scale parameters
///
/// ## Special Cases and Boundary Conditions
/// - **All finite x**: Standard PDF computation using absolute value and exponential
/// - **x = μ**: Maximum density value of 1/(2b) at the location parameter
/// - **Symmetric shape**: PDF is symmetric around μ with exponential decay
/// - **Invalid parameters**: Returns error for b ≤ 0 or non-finite parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When scale ≤ 0 or parameters are non-finite
///
/// ## Applications
/// Commonly used in:
/// - Robust statistical estimation (L1 regression)
/// - Signal processing and noise modelling
/// - Bayesian analysis with sparse priors
/// - Financial risk modelling
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
/// let location = 0.0;  // μ = 0 (centred)
/// let scale = 1.0;     // b = 1 (standard scale)
/// let result = laplace_pdf_simd(&x, location, scale, None, None)?;
/// // Returns PDF values for standard Laplace distribution
/// ```
#[inline(always)]
pub fn laplace_pdf_simd(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    laplace_pdf_simd_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Laplace CDF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn laplace_cdf_simd_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "laplace_cdf: invalid location or scale".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let inv_b = 1.0 / scale;

    // SIMD splats
    let loc_v = Simd::<f64, N>::splat(location);
    let inv_b_v = Simd::<f64, N>::splat(inv_b);

    // Scalar fallback
    let scalar_body = |xi: f64| {
        let z = (xi - location) * inv_b;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let z = (x_v - loc_v) * inv_b_v;
        let below = z.simd_lt(Simd::splat(0.0));
        let expm = Simd::splat(0.5) * z.exp(); // 0.5·exp(z)
        let expnm = Simd::splat(0.5) * (-z).exp(); // 0.5·exp(−z)
        below.select(expm, Simd::splat(1.0) - expnm)
    };

    // Dense fast path (no nulls)
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
    let mask_ref = null_mask.expect("laplace_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();

    // Check if arrays are SIMD 64-byte aligned
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

    // Scalar fallback - alignment check failed
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// SIMD-accelerated implementation of Laplace distribution cumulative distribution function.
///
/// Computes the cumulative distribution function (CDF) of the Laplace (double exponential)
/// distribution using vectorised SIMD operations for enhanced performance on
/// symmetric exponential probability calculations.
#[inline(always)]
pub fn laplace_cdf_simd(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    laplace_cdf_simd_to(
        x,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Laplace quantile SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn laplace_quantile_simd_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if !location.is_finite() || !(scale > 0.0 && scale.is_finite()) {
        return Err(KernelError::InvalidArguments(
            "laplace_quantile: invalid location or scale".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    // SIMD constants
    let loc_v = Simd::<f64, N>::splat(location);
    let scale_v = Simd::<f64, N>::splat(scale);
    let two_v = Simd::<f64, N>::splat(2.0);
    let one_v = Simd::<f64, N>::splat(1.0);
    let half_v = Simd::<f64, N>::splat(0.5);

    // scalar fallback
    let scalar_body = move |pi: f64| -> f64 {
        if pi > 0.0 && pi < 1.0 && pi.is_finite() {
            if pi < 0.5 {
                location + scale * (2.0 * pi).ln()
            } else {
                location - scale * (2.0 * (1.0 - pi)).ln()
            }
        } else if pi == 0.0 {
            f64::NEG_INFINITY
        } else if pi == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    };

    // SIMD body for both dense & masked paths
    let simd_body = move |p_v: Simd<f64, N>| {
        let low = loc_v + scale_v * (two_v * p_v).ln();
        let high = loc_v - scale_v * (two_v * (one_v - p_v)).ln();
        p_v.simd_lt(half_v).select(low, high)
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        // Check if arrays are SIMD 64-byte aligned
        if is_simd_aligned(p) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(p, output, simd_body, scalar_body);
            return Ok(());
        }

        // Scalar fallback - alignment check failed
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    // Null-aware masked path
    let mask_ref = null_mask.expect("laplace_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();

    // Check if arrays are SIMD 64-byte aligned
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

    // Scalar fallback - alignment check failed
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);

    Ok(())
}

/// SIMD-accelerated implementation of Laplace distribution quantile function.
///
/// Computes the quantile function (inverse CDF) of the Laplace (double exponential)
/// distribution using vectorised SIMD operations for enhanced performance on
/// symmetric exponential quantile calculations.
#[inline(always)]
pub fn laplace_quantile_simd(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    laplace_quantile_simd_to(
        p,
        location,
        scale,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
