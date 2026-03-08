// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Student's t-Distribution Scalar Implementations** - *Small-Sample Statistical Foundation*
//!
//! Scalar implementations of Student's t-distribution functions providing the computational
//! foundation for statistical inference with unknown population variance.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Student-t PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn student_t_pdf_std_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_pdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let coeff =
        (ln_gamma((df + 1.0) * 0.5) - ln_gamma(df * 0.5) - 0.5 * (df * std::f64::consts::PI).ln())
            .exp();

    let scalar_body =
        move |xi: f64| -> f64 { coeff * (1.0 + xi * xi / df).powf(-(df + 1.0) * 0.5) };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("student_t_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Student‐t PDF
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn student_t_pdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    student_t_pdf_std_to(x, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Student-t CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn student_t_cdf_std_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_cdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    let len = x.len();
    let half_df = 0.5 * df;
    let t = df;
    let a = half_df;
    let b = 0.5;

    let eval = |xi: f64| -> f64 {
        let xb = t / (t + xi * xi);
        let ib = incomplete_beta(a, b, xb);
        let sign = if xi >= 0.0 { 1.0 } else { -1.0 };
        0.5 + sign * 0.5 * (1.0 - ib)
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &xi) in x.iter().enumerate() {
            output[i] = eval(xi);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null path requires a mask");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = eval(x[i]);
        }
    }
    Ok(())
}

/// Student-t CDF
/// - Algorithm AS 3 (Hill, 1970) / Algorithm 395 (G. W. Hill, 1970)
/// - https://www.jstor.org/stable/2346841
///
/// Scalar implementation uses the incomplete beta via continued fraction.
/// Returns error if df < 1.
#[inline(always)]
pub fn student_t_cdf_std(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    student_t_cdf_std_to(x, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Student-t quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
pub fn student_t_quantile_std_to(
    p: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "student_t_quantile: invalid df".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    let len = p.len();

    let compute_quantile = |prob: f64| -> f64 {
        if !(prob >= 0.0 && prob <= 1.0) || !prob.is_finite() {
            f64::NAN
        } else if prob == 0.0 {
            f64::NEG_INFINITY
        } else if prob == 1.0 {
            f64::INFINITY
        } else if df > 1e7 {
            normal_quantile_scalar(prob, 0.0, 1.0)
        } else {
            let a = 0.5 * df;
            let b = 0.5;
            let x = if prob < 0.5 {
                incomplete_beta_inv(a, b, 2.0 * prob)
            } else {
                incomplete_beta_inv(a, b, 2.0 * (1.0 - prob))
            };
            let t = ((df * (1.0 - x)) / x).sqrt();
            if prob < 0.5 { -t } else { t }
        }
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &prob) in p.iter().enumerate() {
            output[i] = compute_quantile(prob);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null path requires a mask");
    for i in 0..len {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = compute_quantile(p[i]);
        }
    }
    Ok(())
}

/// Student-t quantile (inverse CDF), using the AS 241 algorithm
/// Reference: Michael J. Wichura, Algorithm AS 241: The Percentage Points of the Normal Distribution
pub fn student_t_quantile_std(
    p: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    student_t_quantile_std_to(p, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
