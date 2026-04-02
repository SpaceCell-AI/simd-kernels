// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Beta Distribution Module** - *Continuous Probability Distribution (0, 1)*
//!
//! High-performance implementation of the beta distribution with SIMD-accelerated kernels
//! for probability density function (PDF), cumulative distribution function (CDF), and
//! quantile (inverse CDF) calculations.
//!
//! ## Overview
//! The beta distribution Beta(α, β) is a continuous probability distribution defined on the
//! interval (0, 1), parameterised by two positive shape parameters α (alpha) and β (beta).
//! It is extensively used in Bayesian statistics, quality control, and modelling proportions.
//!
//! ## Mathematical Definition
//! - **PDF**: f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α, β)
//! - **CDF**: F(x; α, β) = I_x(α, β) (regularised incomplete beta function)
//! - **Support**: x ∈ (0, 1)
//! - **Parameters**: α > 0, β > 0
//!
//! Where B(α, β) is the beta function and I_x(α, β) is the regularised incomplete beta function.
//!
//! ## Use Cases
//! - Bayesian conjugate prior for binomial distributions
//! - Modelling proportions, percentages, and probabilities
//! - Quality control and reliability analysis
//! - Finance: modelling recovery rates and default probabilities
//! - Machine learning: beta-distributed latent variables
//!
//! ## Implementation Strategy
//! - **SIMD Path**: Utilises vectorised operations for bulk computations when data is
//!   64-byte aligned, falling back gracefully to scalar implementations
//! - **Scalar Path**: High-precision reference implementations using continued fractions
//!   for CDF and Newton-Raphson iteration for quantile calculations
//! - **Numerical Stability**: Special handling for boundary cases (x = 0, x = 1) and
//!   parameter edge cases (α = 1, β = 1)
//! - Memory: Zero-copy compatible with **Minarrow**'s 64-byte aligned `FloatArray`
//!
//! ## Accuracy and Validation
//! - Numerical accuracy verified against SciPy reference implementations
//! - Relative error typically < 1e-14 or 1e-15 for PDF/CDF. See `./tests` for your specific function,
//! and keep in mind that floating point accuracy can diverge across platforms.
//! - Comprehensive edge case handling for domain boundaries
//! - Round-trip validation: quantile(CDF(x)) ≈ x within numerical tolerance
//!
//! ## Thread Safety
//! All functions are thread-safe and good for parallel processing with libraries like Rayon.

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Beta PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// ## Parameters
/// - `x`: Array of values to evaluate, must be in [0, 1] for non-zero results
/// - `alpha`: Shape parameter α > 0
/// - `beta`: Shape parameter β > 0
/// - `output`: Pre-allocated output buffer (must match input length)
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
#[inline(always)]
pub fn beta_pdf_to(
    x: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::beta_pdf_simd_to(x, alpha, beta, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::beta_pdf_std_to(x, alpha, beta, output, null_mask, null_count)
    }
}

/// Computes the probability density function (PDF) of the beta distribution.
///
/// Calculates f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α, β) for each element
/// in the input array, where B(α, β) is the beta function.
///
/// ## Parameters
/// - `x`: Array of values to evaluate, must be in [0, 1] for non-zero results
/// - `alpha`: Shape parameter α > 0
/// - `beta`: Shape parameter β > 0
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `FloatArray<f64>` containing PDF values, with nulls propagated from input mask.
///
/// ## Behaviour
/// - Values outside [0, 1] return 0.0
/// - Boundary cases (x=0, x=1) handled according to parameter values:
///   - If α < 1 and x=0, or β < 1 and x=1: returns `f64::INFINITY`
///   - If α > 1 and x=0, or β > 1 and x=1: returns 0.0
///   - Special cases for α=1 or β=1 return finite values
/// - Input nulls propagate to output nulls
/// - SIMD-accelerated when `simd` feature enabled and data is 64-byte aligned
/// - Falls back to optimised scalar implementation otherwise
///
/// ## Errors
/// Returns `KernelError::InvalidArguments` if α ≤ 0, β ≤ 0, or parameters are non-finite.
///
/// ## Example
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::beta::beta_pdf;
/// use minarrow::vec64;
///
/// let x = vec64![0.1, 0.5, 0.9];
/// let result = beta_pdf(&x, 2.0, 3.0, None, None).unwrap();
/// ```
#[inline(always)]
pub fn beta_pdf(
    x: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::beta_pdf_simd(x, alpha, beta, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::beta_pdf_std(x, alpha, beta, null_mask, null_count)
    }
}

/// Beta CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// ## Parameters
/// - `x`: Array of values to evaluate, domain [0, 1]
/// - `alpha`: Shape parameter α > 0
/// - `beta`: Shape parameter β > 0
/// - `output`: Pre-allocated output buffer (must match input length)
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
#[inline(always)]
pub fn beta_cdf_to(
    x: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::beta_cdf_simd_to(x, alpha, beta, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::beta_cdf_std_to(x, alpha, beta, output, null_mask, null_count)
    }
}

/// Computes the cumulative distribution function (CDF) of the beta distribution.
///
/// Calculates F(x; α, β) = I_x(α, β), the regularised incomplete beta function,
/// representing P(X ≤ x) where X ~ Beta(α, β).
///
/// ## Parameters
/// - `x`: Array of values to evaluate, domain [0, 1]
/// - `alpha`: Shape parameter α > 0
/// - `beta`: Shape parameter β > 0
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `FloatArray<f64>` containing CDF values in [0, 1], with nulls propagated from input mask.
///
/// ## Behaviour
/// - Values x ≤ 0 return 0.0
/// - Values x ≥ 1 return 1.0
/// - Interior values computed using continued fraction expansion of incomplete beta
/// - Monotonically non-decreasing: F(x₁) ≤ F(x₂) for x₁ ≤ x₂
/// - Input nulls propagate to output nulls
///
/// ## Errors
/// Returns `KernelError::InvalidArguments` if α ≤ 0, β ≤ 0, or parameters are non-finite.
///
/// ## Example
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::beta::beta_cdf;
/// use minarrow::vec64;
///
/// let x = vec64![0.0, 0.25, 0.5, 0.75, 1.0];
/// let result = beta_cdf(&x, 2.0, 3.0, None, None).unwrap();
/// ```
#[inline(always)]
pub fn beta_cdf(
    x: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::beta_cdf_simd(x, alpha, beta, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::beta_cdf_std(x, alpha, beta, null_mask, null_count)
    }
}

/// Beta quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// ## Parameters
/// - `p`: Array of probability values to evaluate, must be in [0, 1]
/// - `alpha`: Shape parameter α > 0
/// - `beta`: Shape parameter β > 0
/// - `output`: Pre-allocated output buffer (must match input length)
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
#[inline(always)]
pub fn beta_quantile_to(
    p: &[f64],
    alpha: f64,
    beta: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::beta_quantile_std_to(p, alpha, beta, output, null_mask, null_count)
}

/// Computes the quantile function (inverse CDF) of the beta distribution.
///
/// Calculates F⁻¹(p; α, β), the value x such that F(x; α, β) = p,
/// where F is the CDF of Beta(α, β). Also known as the percent-point function (PPF).
///
/// ## Parameters
/// - `p`: Array of probability values to evaluate, must be in [0, 1]
/// - `alpha`: Shape parameter α > 0
/// - `beta`: Shape parameter β > 0
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `FloatArray<f64>` containing quantile values in [0, 1], with nulls propagated from input mask.
///
/// ## Behaviour
/// - p = 0.0 returns 0.0
/// - p = 1.0 returns 1.0
/// - Values outside [0, 1] return `f64::NAN`
/// - Interior values computed using AS 109 algorithm with Newton-Raphson refinement
/// - Monotonically non-decreasing: Q(p₁) ≤ Q(p₂) for p₁ ≤ p₂
/// - Input nulls propagate to output nulls
///
/// ## Errors
/// Returns `KernelError::InvalidArguments` if α ≤ 0, β ≤ 0, or parameters are non-finite.
///
/// ## Example
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::beta::beta_quantile;
/// use minarrow::vec64;
///
/// let p = vec64![0.1, 0.25, 0.5, 0.75, 0.9];
/// let result = beta_quantile(&p, 2.0, 3.0, None, None).unwrap();
/// // Returns quantiles for Beta(2, 3) distribution
/// ```
#[inline(always)]
pub fn beta_quantile(
    p: &[f64],
    alpha: f64,
    beta: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::beta_quantile_std(p, alpha, beta, null_mask, null_count)
}

#[cfg(test)]
mod beta_tests {

    // See `./tests` for the scipy test suite

    use super::*;
    use crate::kernels::scientific::distributions::univariate::common::dense_data;
    use minarrow::{Bitmask, vec64};

    // helpers
    fn mask_vec(m: &Bitmask) -> Vec<bool> {
        (0..m.len()).map(|i| m.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // beta_pdf  -  correctness

    #[test]
    fn beta_pdf_exact_values() {
        // α = 2, β = 5  ->  f(x) = 30·x·(1−x)^4
        let x = vec64![0.1, 0.2, 0.5];
        let expect = vec64![1.9683000000000004, 2.4576, 0.9375]; // 30*x*(1-x)^4
        let arr = dense_data(beta_pdf(&x, 2.0, 5.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    #[test]
    fn beta_pdf_bulk_vs_scalar() {
        let x = vec64![0.01, 0.25, 0.6, 0.9];
        let all = dense_data(beta_pdf(&x, 3.0, 4.0, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let s = dense_data(beta_pdf(xi_aligned.as_slice(), 3.0, 4.0, None, None).unwrap())[0];
            assert_close(all[i], s, 1e-14);
        }
    }

    #[test]
    fn beta_pdf_out_of_domain_zero() {
        let x = vec64![-0.1, 0.5, 1.2];
        let arr = dense_data(beta_pdf(&x, 2.0, 2.0, None, None).unwrap());
        assert_eq!(arr[0], 0.0);
        assert!(arr[1].is_finite() && arr[1] > 0.0);
        assert_eq!(arr[2], 0.0);
    }

    // beta_pdf  -  mask propagation

    #[test]
    fn beta_pdf_mask_and_nulls() {
        let x = vec64![0.2, 0.4, 0.6];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(1, false);
        } // null the middle value
        let out = beta_pdf(&x, 2.0, 5.0, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(out.null_mask.as_ref().unwrap());
        assert!(nulls == [true, false, true]);
        assert!(out.data[1].is_nan());
    }

    // beta_cdf  -  correctness

    #[test]
    fn beta_cdf_exact_values() {
        // α = 2, β = 5  ->  F(x)=1-(1+5x)(1-x)^5
        fn exact(x: f64) -> f64 {
            1.0 - (1.0 + 5.0 * x) * (1.0 - x).powi(5)
        }
        let x = [0.1, 0.2, 0.5];
        let expect = x.map(exact);
        let arr = dense_data(beta_cdf(&x, 2.0, 5.0, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    #[test]
    fn beta_cdf_edges() {
        let x = vec64![-1.0, 0.0, 1.0, 2.0];
        let arr = dense_data(beta_cdf(&x, 2.0, 2.0, None, None).unwrap());
        assert_close(arr[1], 0.0, 1e-15); // x ≤ 0  -> 0
        assert_close(arr[2], 1.0, 1e-15); // x ≥ 1  -> 1
        // out-of-domain negatives/positives are clamped, never NaN
        assert_close(arr[0], 0.0, 1e-15);
        assert_close(arr[3], 1.0, 1e-15);
    }

    // beta_quantile  -  correctness & round-trip

    #[test]
    fn beta_quantile_round_trip() {
        let p = vec64![0.0001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.9999];
        let q = dense_data(beta_quantile(&p, 2.0, 5.0, None, None).unwrap());
        let p2 = dense_data(beta_cdf(&q, 2.0, 5.0, None, None).unwrap());
        for (orig, back) in p.iter().zip(p2.iter()) {
            assert_close(*orig, *back, 5e-12); // beta_inv accuracy ~1e-12
        }
    }

    #[test]
    fn beta_quantile_bounds() {
        let p = vec64![0.0, 1.0];
        let arr = dense_data(beta_quantile(&p, 3.0, 4.0, None, None).unwrap());
        assert_close(arr[0], 0.0, 1e-15);
        assert_close(arr[1], 1.0, 1e-15);
    }

    #[test]
    fn beta_quantile_domain_violations_nan() {
        let p = vec64![-0.1, 1.1, f64::NAN];
        let arr = dense_data(beta_quantile(&p, 2.0, 2.0, None, None).unwrap());
        assert!(arr.iter().all(|v| v.is_nan()));
    }

    // parameter validation

    #[test]
    fn beta_invalid_parameters_error() {
        assert!(beta_pdf(&[0.5], -1.0, 2.0, None, None).is_err());
        assert!(beta_cdf(&[0.5], 0.0, 1.0, None, None).is_err());
        assert!(beta_quantile(&[0.5], 2.0, f64::INFINITY, None, None).is_err());
    }

    // empty input / null-mask round-trip

    #[test]
    fn beta_empty_input() {
        assert!(beta_pdf(&[], 2.0, 3.0, None, None).unwrap().data.is_empty());
        assert!(beta_cdf(&[], 2.0, 3.0, None, None).unwrap().data.is_empty());
        assert!(
            beta_quantile(&[], 2.0, 3.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
    }

    #[test]
    fn beta_quantile_mask_propagation() {
        let p = vec64![0.2, 0.4, 0.6];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(0, false);
        }
        let arr = beta_quantile(&p, 2.0, 5.0, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert!(!nulls[0] && nulls[1] && nulls[2]);
        assert!(arr.data[0].is_nan());
    }
}
