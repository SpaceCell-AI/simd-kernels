// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Log-Normal Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of log-normal distribution functions
//! utilising vectorised operations for bulk computations on logarithmically-transformed
//! normally-distributed data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the log-normal distribution, arising when the logarithm of a random variable
//! follows a normal distribution. The implementations automatically fall back to scalar
//! versions when data alignment requirements are not met.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

use minarrow::{Bitmask, FloatArray, Vec64, enums::error::KernelError};

use crate::kernels::scientific::distributions::shared::constants::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::erf::{erf, erf_simd};
use crate::utils::has_nulls;

// TODO: Add alignment check

/// Lognormal PDF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|μ,σ) = 1/(xσ√2π) * exp(-½[(ln(x)-μ)/σ]^2)
#[inline(always)]
pub fn lognormal_pdf_simd_to(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_pdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let denom = sdlog * SQRT_2PI;

    // SIMD splats
    let meanlog_v = Simd::<f64, N>::splat(meanlog);
    let sdlog_v = Simd::<f64, N>::splat(sdlog);
    let denom_v = Simd::<f64, N>::splat(denom);
    let zero_v = Simd::<f64, N>::splat(0.0);
    let half_neg = Simd::<f64, N>::splat(-0.5);

    // scalar fallback
    let scalar_body = |xi: f64| -> f64 {
        if xi > 0.0 {
            let z = (xi.ln() - meanlog) / sdlog;
            (-0.5 * z * z).exp() / (denom * xi)
        } else {
            0.0
        }
    };

    // vectorised body
    let simd_body = move |x_v: Simd<f64, N>| {
        let positive = x_v.simd_gt(zero_v);
        let logx = x_v.ln();
        let z = (logx - meanlog_v) / sdlog_v;
        let pdf = (half_neg * z * z).exp() / (denom_v * x_v);
        positive.select(pdf, zero_v)
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
        return Ok(());
    }

    // Null-aware path
    let mask_ref = null_mask.expect("lognormal_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_simd_to::<N, _, _>(
        x,
        mask_ref,
        output,
        &mut out_mask,
        simd_body,
        scalar_body,
    );
    Ok(())
}

/// SIMD-accelerated implementation of log-normal distribution probability density function.
///
/// Computes the probability density function (PDF) of the log-normal distribution
/// using vectorised SIMD operations for enhanced performance on logarithmically
/// transformed normally-distributed data.
///
/// ## Parameters
/// - `x`: Input values where PDF should be evaluated (domain: x > 0)
/// - `meanlog`: Mean of underlying normal distribution μ
/// - `sdlog`: Standard deviation of underlying normal distribution σ > 0
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid sdlog or meanlog parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When sdlog ≤ 0 or parameters are non-finite
///
/// ## Use cases
/// - Financial modelling - asset prices, portfolio returns
/// - Reliability engineering - failure times
/// - Environmental science - pollutant concentrations
/// - Biology - multiplicative growth processes
/// - Economics - income and wealth distributions
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [0.5, 1.0, 2.0, 3.0];  // positive values only
/// let meanlog = 0.0;             // μ = 0 (median = 1.0)
/// let sdlog = 1.0;               // σ = 1 (standard log-normal)
/// let result = lognormal_pdf_simd(&x, meanlog, sdlog, None, None)?;
/// // Returns PDF values for standard log-normal distribution
/// ```
#[inline(always)]
pub fn lognormal_pdf_simd(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    lognormal_pdf_simd_to(x, meanlog, sdlog, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

// TODO: Add alignment check

/// Lognormal CDF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|μ,σ) = Φ((ln(x) - μ)/σ)
#[inline(always)]
pub fn lognormal_cdf_simd_to(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // 1) Parameter checks
    if sdlog <= 0.0 || !sdlog.is_finite() || !meanlog.is_finite() {
        return Err(KernelError::InvalidArguments(
            "lognormal_cdf: invalid parameters".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    // Constants
    const N: usize = W64;
    let sqrt2 = SQRT_2;

    let meanlog_v = Simd::<f64, N>::splat(meanlog);
    let sdlog_v = Simd::<f64, N>::splat(sdlog);
    let sqrt2_v = Simd::<f64, N>::splat(sqrt2);
    let half_v = Simd::<f64, N>::splat(0.5);
    let one_v = Simd::<f64, N>::splat(1.0);
    let zero_v = Simd::<f64, N>::splat(0.0);

    // scalar fallback
    let scalar_body = move |xi: f64| -> f64 {
        if xi > 0.0 {
            let z = (xi.ln() - meanlog) / sdlog / sqrt2;
            0.5 * (1.0 + erf(z))
        } else {
            0.0
        }
    };

    // for x>0, compute ½[1+erf((ln x−μ)/(σ√2))], else 0, NaN for NaN inputs
    let simd_body = move |x_v: Simd<f64, N>| {
        let is_nan = x_v.simd_ne(x_v); // NaN != NaN is true
        let positive = x_v.simd_gt(zero_v);
        let logx = x_v.ln();
        let z = (logx - meanlog_v) / (sdlog_v * sqrt2_v);
        let phi = half_v * (one_v + erf_simd::<N>(z));
        // Return NaN for NaN inputs, zero for non-positive, computed value for positive
        is_nan.select(
            Simd::<f64, N>::splat(f64::NAN),
            positive.select(phi, zero_v),
        )
    };

    // Dense fast path
    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
        return Ok(());
    }

    // Null‐aware path
    let mask_ref = null_mask.expect("lognormal_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_simd_to::<N, _, _>(
        x,
        mask_ref,
        output,
        &mut out_mask,
        simd_body,
        scalar_body,
    );

    Ok(())
}

/// SIMD-accelerated implementation of log-normal distribution cumulative distribution function.
///
/// Computes the cumulative distribution function (CDF) of the log-normal distribution
/// using vectorised SIMD operations for enhanced performance on logarithmically-
/// transformed normal probability calculations.
///
/// ## Parameters
/// - `x`: Input values where CDF should be evaluated (domain: x > 0)
/// - `meanlog`: Mean of underlying normal distribution μ
/// - `sdlog`: Standard deviation of underlying normal distribution σ > 0
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed CDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid sdlog or meanlog parameters
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When sdlog ≤ 0 or parameters are non-finite
///
/// ## Use cases
/// The log-normal CDF represents the probability of observing a value less than or
/// equal to x, essential for:
/// - Risk assessment in financial modelling
/// - Reliability analysis - probability of failure before time x
/// - Environmental impact studies - probability of concentration below threshold
/// - Quality control - probability of measurement below specification
///
/// ## Example Usage
/// ```rust,ignore
/// let x = [0.5, 1.0, 2.0, 5.0];  // positive values only
/// let meanlog = 0.0;             // μ = 0 (median = 1.0)
/// let sdlog = 1.0;               // σ = 1 (standard log-normal)
/// let result = lognormal_cdf_simd(&x, meanlog, sdlog, None, None)?;
/// // Returns cumulative probabilities for standard log-normal
/// ```
#[inline(always)]
pub fn lognormal_cdf_simd(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    lognormal_cdf_simd_to(x, meanlog, sdlog, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
