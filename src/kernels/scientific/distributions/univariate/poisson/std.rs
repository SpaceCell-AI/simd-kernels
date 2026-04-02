// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Poisson Distribution Scalar Implementations** - *Discrete Event Computation*
//!
//! Scalar (non-SIMD) implementations of Poisson distribution functions optimised for numerical
//! stability and computational efficiency.

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::*;
#[cfg(not(feature = "simd"))]
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std, dense_univariate_kernel_u64_std_to,
    masked_univariate_kernel_u64_std, masked_univariate_kernel_u64_std_to,
};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Poisson PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn poisson_pmf_std_to(
    k: &[u64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // λ may be zero (degenerate at k=0); only forbid negatives / non-finite
    if lambda < 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "poisson_pmf: λ must be non-negative and finite".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    // Degenerate distribution: λ == 0 -> PMF(k) = 1_{k==0}
    if lambda == 0.0 {
        if !has_nulls(null_count, null_mask) {
            for (i, &ki) in k.iter().enumerate() {
                output[i] = if ki == 0 { 1.0 } else { 0.0 };
            }
            return Ok(());
        }
        let mask = null_mask.expect("poisson_pmf: null_count > 0 requires null_mask");
        for i in 0..k.len() {
            if !unsafe { mask.get_unchecked(i) } {
                output[i] = f64::NAN;
            } else {
                output[i] = if k[i] == 0 { 1.0 } else { 0.0 };
            }
        }
        return Ok(());
    }

    // Regular path (λ > 0)
    let log_lambda = lambda.ln();
    let neg_lambda = -lambda;

    let scalar_body = move |ki: u64| -> f64 {
        let kf = ki as f64;
        (neg_lambda + kf * log_lambda - ln_gamma_plus1(kf)).exp()
    };

    if !has_nulls(null_count, null_mask) {
        dense_univariate_kernel_u64_std_to(k, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("poisson_pmf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    masked_univariate_kernel_u64_std_to(k, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Poisson PMF: P(K=k|λ) = e^{-λ} · λ^k / k!
/// k: observed event counts (all ≥ 0)
/// λ: event rate (λ > 0, finite)
#[cfg(not(feature = "simd"))]
#[inline(always)]
pub fn poisson_pmf_std(
    k: &[u64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    poisson_pmf_std_to(k, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Poisson CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn poisson_cdf_std_to(
    k: &[u64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if lambda < 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "poisson_cdf: λ must be non-negative and finite".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    // Dense path: no nulls
    if !has_nulls(null_count, null_mask) {
        for (i, &ki) in k.iter().enumerate() {
            output[i] = if ki == u64::MAX {
                1.0
            } else if lambda == 0.0 {
                1.0
            } else {
                1.0 - reg_lower_gamma((ki as f64) + 1.0, lambda)
            };
        }
        return Ok(());
    }

    // Null-aware path: propagate input nulls
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    for idx in 0..k.len() {
        if !unsafe { mask.get_unchecked(idx) } {
            output[idx] = f64::NAN;
        } else {
            let ki = k[idx];
            output[idx] = if ki == u64::MAX {
                1.0
            } else if lambda == 0.0 {
                1.0
            } else {
                1.0 - reg_lower_gamma((ki as f64) + 1.0, lambda)
            };
        }
    }
    Ok(())
}

/// Poisson CDF: F(K=k|λ) = ∑_{i=0}^k Poisson_pmf_std(i, λ)
/// Efficient and robust using the lower regularised incomplete gamma:
/// F(K=k|λ) = γ(⌊k+1⌋, λ) / Γ(⌊k+1⌋)
#[inline(always)]
pub fn poisson_cdf_std(
    k: &[u64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    poisson_cdf_std_to(k, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Poisson quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn poisson_quantile_std_to(
    p: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if lambda < 0.0 || !lambda.is_finite() {
        return Err(KernelError::InvalidArguments(
            "poisson_quantile: λ must be non-negative and finite".into(),
        ));
    }
    if p.is_empty() {
        return Ok(());
    }

    // absolute tolerance to avoid off-by-one from tiny FP underestimation near 1
    const ABS_TOL: f64 = 1e-12;

    let compute_quantile = |pi: f64| -> f64 {
        if !(pi >= 0.0 && pi <= 1.0) || !pi.is_finite() {
            f64::NAN
        } else if pi == 0.0 {
            -1.0
        } else if pi == 1.0 {
            f64::INFINITY
        } else if lambda == 0.0 {
            0.0
        } else {
            // Cornish-Fisher start
            let mu = lambda;
            let sigma = lambda.sqrt();
            let g1 = 1.0 / sigma;
            let z = normal_quantile_scalar(pi, 0.0, 1.0);
            let mut k_est = mu + sigma * (z + g1 * (z * z - 1.0) / 6.0);
            if k_est < 0.0 {
                k_est = 0.0;
            }
            let mut k = k_est.floor() as u64;

            // increase until CDF(k) ≥ p within tolerance
            let max_k = (lambda * 10.0).ceil() as u64 + 1000;
            // forward search: ensure CDF(k) ≥ p within tolerance
            let cdf_at = |kk: u64| -> f64 { 1.0 - reg_lower_gamma((kk as f64) + 1.0, lambda) };
            let cdf_before = |kk: u64| -> f64 {
                if kk == 0 {
                    0.0
                } else {
                    1.0 - reg_lower_gamma(kk as f64, lambda)
                }
            };

            let mut cdf = cdf_at(k);
            while cdf + ABS_TOL < pi && k < max_k {
                k += 1;
                cdf = cdf_at(k);
            }

            // step down to ensure *minimal* k
            while k > 0 {
                let prev = cdf_before(k);
                if prev >= pi - ABS_TOL {
                    k -= 1;
                } else {
                    break;
                }
            }

            k as f64
        }
    };

    if !has_nulls(null_count, null_mask) {
        for (i, &pi) in p.iter().enumerate() {
            output[i] = compute_quantile(pi);
        }
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    for i in 0..p.len() {
        if !unsafe { mask.get_unchecked(i) } {
            output[i] = f64::NAN;
        } else {
            output[i] = compute_quantile(p[i]);
        }
    }
    Ok(())
}

/// Poisson quantile function (inverse CDF).
///
/// For probability `p` ∈ (0,1), returns the smallest integer `k` such that
///     Pr[X ≤ k] ≥ p, where X ~ Poisson(λ).
#[inline(always)]
pub fn poisson_quantile_std(
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

    poisson_quantile_std_to(p, lambda, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
