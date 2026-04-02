// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Exponential PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_pdf_std_to(
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

    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            lambda * (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("exponential_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Exponential PDF: f(x|λ) = λ·exp(-λ·x) for x ≥ 0, 0 otherwise.
/// Returns error if λ ≤ 0 or non-finite.
#[inline(always)]
pub fn exponential_pdf_std(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    exponential_pdf_std_to(x, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Exponential CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_cdf_std_to(
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

    let scalar_body = move |xi: f64| {
        if xi.is_nan() {
            f64::NAN
        } else if xi >= 0.0 {
            1.0 - (-lambda * xi).exp()
        } else {
            0.0
        }
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("exponential_cdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Exponential CDF: F(x|λ) = 1 - exp(-λ·x) for x ≥ 0, 0 otherwise.
/// Error if λ ≤ 0 or non‐finite.
#[inline(always)]
pub fn exponential_cdf_std(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    exponential_cdf_std_to(x, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Exponential quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_quantile_std_to(
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

    let scalar_body = |pi: f64| -((1.0 - pi).ln()) / lambda;

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(p, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("exponential_quantile: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(p, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Exponential quantile (inverse CDF): Q(p|λ) = -ln(1-p)/λ, pure math
#[inline(always)]
pub fn exponential_quantile_std(
    p: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    exponential_quantile_std_to(p, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
