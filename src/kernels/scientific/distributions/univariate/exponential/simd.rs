// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Exponential Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of exponential distribution functions
//! utilising vectorised transcendental operations for bulk computations on aligned data.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

use minarrow::{Bitmask, FloatArray, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};

use crate::utils::has_nulls;

/// SIMD-accelerated exponential PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_pdf_simd_to(
    x: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "exponential_pdf: λ must be positive and finite".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let lambda_v = Simd::<f64, N>::splat(lambda);
    let zero_v = Simd::<f64, N>::splat(0.0);

    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            lambda * (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let is_nan = x_v.simd_ne(x_v);
        let valid = x_v.simd_ge(zero_v);
        let pdf = lambda_v * (-lambda_v * x_v).exp();
        let result = valid.select(pdf, zero_v);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("exponential_pdf: null_count > 0 requires null_mask");
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

/// SIMD-accelerated implementation of exponential distribution probability density function.
#[inline(always)]
pub fn exponential_pdf_simd(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    use minarrow::Vec64;

    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    exponential_pdf_simd_to(x, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated exponential CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_cdf_simd_to(
    x: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "exponential_cdf: λ must be positive and finite".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let lambda_v = Simd::<f64, N>::splat(lambda);
    let zero_v = Simd::<f64, N>::splat(0.0);
    let one_v = Simd::<f64, N>::splat(1.0);

    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            1.0 - (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let is_nan = x_v.simd_ne(x_v);
        let valid = x_v.simd_ge(zero_v);
        let y = one_v - (-lambda_v * x_v).exp();
        let result = valid.select(y, zero_v);
        is_nan.select(Simd::<f64, N>::splat(f64::NAN), result)
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("exponential_cdf: null_count > 0 requires null_mask");
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

/// SIMD-accelerated implementation of exponential distribution cumulative distribution function.
#[inline(always)]
pub fn exponential_cdf_simd(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    use minarrow::Vec64;

    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    exponential_cdf_simd_to(x, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated exponential quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_quantile_simd_to(
    p: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "exponential_quantile: λ must be positive and finite".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let lambda_v = Simd::<f64, N>::splat(lambda);
    let scalar_body = |pi: f64| -((1.0 - pi).ln()) / lambda;
    let simd_body = move |p_v: Simd<f64, N>| -((Simd::splat(1.0) - p_v).ln()) / lambda_v;

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(p) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(p, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("exponential_quantile: null_count > 0 requires null_mask");
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

/// SIMD-accelerated implementation of exponential distribution quantile function.
#[inline(always)]
pub fn exponential_quantile_simd(
    p: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    use minarrow::Vec64;

    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    exponential_quantile_simd_to(p, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
