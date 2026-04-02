//! # Laplace Distribution (Double Exponential)
//!
//! The Laplace distribution, also known as the double exponential distribution, is a continuous
//! probability distribution characterised by its sharp peak at the location parameter and
//! exponential decay on both sides. It is commonly used in robust statistics, signal processing,
//! and Bayesian analysis as a heavy-tailed alternative to the normal distribution.
//!
//! ## Mathematical Definition
//!
//! The Laplace distribution is parameterised by location μ and scale b > 0:
//! - **PDF**: f(x; μ, b) = (1/2b) exp(-|x - μ|/b)
//! - **CDF**: F(x; μ, b) = (1/2) exp((x - μ)/b) for x < μ, 1 - (1/2) exp(-(x - μ)/b) for x ≥ μ
//! - **Quantile**: Q(p; μ, b) = μ + b sign(p - 1/2) ln(2|p - 1/2|)
//!
//! ## Common Applications
//!
//! - **Robust statistics**: L1 regression and median-based estimation
//! - **Signal processing**: Noise modelling with impulsive characteristics  
//! - **Economics**: Financial return modelling with fat tails
//! - **Image processing**: Edge-preserving filters and compressed sensing
//! - **Machine learning**: Lasso regularisation (L1 penalty)
//! - **Bayesian analysis**: Prior distributions for sparse parameters
//!
//! ## Implementation Features
//!
//! - **SIMD acceleration** for all functions with ~2-4x speedup
//! - **Numerical stability** through careful absolute value handling
//! - **Robust quantile inversion** with special case handling
//! - **High accuracy** with extensive validation against reference implementations

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray};

/// Laplace PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn laplace_pdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::laplace_pdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::laplace_pdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Laplace CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn laplace_cdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::laplace_cdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::laplace_cdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Laplace quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn laplace_quantile_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::laplace_quantile_simd_to(p, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::laplace_quantile_std_to(p, location, scale, output, null_mask, null_count)
    }
}

/// Compute the probability density function (PDF) for the Laplace distribution.
///
/// The Laplace distribution, also known as the double exponential distribution, is a continuous
/// distribution characterised by its symmetric shape with exponential decay on both sides of the
/// location parameter. It is widely used in robust statistics and signal processing applications.
///
/// ## Mathematical Definition
///
/// The PDF of the Laplace distribution is defined as:
///
/// ```text
/// f(x; μ, b) = (1/2b) × exp(-|x - μ|/b)
/// ```
///
/// where:
/// - `μ` (location) is the location parameter (mean and median)
/// - `b` (scale) is the scale parameter, with `b > 0`
///
/// ## Parameters
///
/// * `x` - Input values to evaluate the PDF at
/// * `location` - Location parameter μ (mean and median of the distribution)
/// * `scale` - Scale parameter b, must be positive (b > 0)
/// * `null_mask` - Optional bitmask for null value handling in Arrow format
/// * `null_count` - Optional count of null values for optimisation
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing the PDF values, or a `KernelError` if:
/// - Scale parameter is non-positive (b ≤ 0)
/// - Location parameter is not finite (NaN or infinite)
/// - Scale parameter is not finite (NaN or infinite)
///
/// ## Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::laplace::laplace_pdf;
/// use minarrow::vec64;
/// ```
///
/// // Standard Laplace distribution (μ=0, b=1)
/// let x = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let pdf = laplace_pdf(&x, 0.0, 1.0, None, None).unwrap();
///
/// # Applications
///
/// - **Robust regression**: L1 (Lasso) regression uses Laplace prior
/// - **Signal processing**: Modelling impulsive noise
/// - **Finance**: Heavy-tailed return distributions
/// - **Image processing**: Edge-preserving denoising
/// - **Bayesian analysis**: Sparse parameter priors
#[inline(always)]
pub fn laplace_pdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::laplace_pdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::laplace_pdf_std(x, location, scale, null_mask, null_count)
    }
}

/// Laplace (double-exponential) distribution CDF, null-aware and SIMD-accelerated.
/// F(x; μ, b) = 0.5·exp((x−μ)/b)          if x < μ
///           = 1 − 0.5·exp(−(x−μ)/b)     if x ≥ μ
#[inline(always)]
pub fn laplace_cdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::laplace_cdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::laplace_cdf_std(x, location, scale, null_mask, null_count)
    }
}

// TODO: Test whether simd returns the correct edge values

/// Laplace quantile (inverse CDF), null-aware and SIMD-accelerated.
/// Q(p; μ, b) = μ + b·ln(2p)           if p < 0.5
///           = μ − b·ln(2(1−p))       if p ≥ 0.5
#[inline(always)]
pub fn laplace_quantile(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::laplace_quantile_simd(p, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::laplace_quantile_std(p, location, scale, null_mask, null_count)
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // see "./tests" for scipy test suite

    // Helpers reused across tests

    fn mask_vec(bm: &Bitmask) -> Vec<bool> {
        (0..bm.len()).map(|i| bm.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        // Handle infinity cases: two infinities of the same sign are equal
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            return;
        }
        // Handle NaN cases: both NaN should be considered equal
        if a.is_nan() && b.is_nan() {
            return;
        }
        assert!(
            (a - b).abs() < tol,
            "assert_close FAILED: {a} vs {b} (tol={tol})"
        );
    }

    // Distribution parameters used throughout
    const MU: f64 = 1.2; // location μ
    const B: f64 = 0.8; // scale    b  ( > 0 )

    // Reference cases used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"

    fn ref_pdf(x: f64) -> f64 {
        let z = (x - MU).abs() / B;
        (0.5 / B) * (-z).exp()
    }
    fn ref_cdf(x: f64) -> f64 {
        let z = (x - MU) / B;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }
    fn ref_quantile(p: f64) -> f64 {
        if p == 0.0 {
            f64::NEG_INFINITY
        } else if p == 1.0 {
            f64::INFINITY
        } else if 0.0 < p && p < 0.5 {
            MU + B * (2.0 * p).ln()
        } else if 0.5 <= p && p < 1.0 {
            MU - B * (2.0 * (1.0 - p)).ln()
        } else {
            f64::NAN
        }
    }

    // PDF - correctness
    #[test]
    fn pdf_reference_values() {
        let x = vec64![-2.0, 0.5, 1.2, 3.0];
        let expect: Vec<f64> = x.iter().copied().map(ref_pdf).collect();
        let out = dense_data(laplace_pdf(&x, MU, B, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn pdf_bulk_vs_scalar() {
        let x = vec64![-1.0, 0.0, 1.5, 4.4];
        let bulk = dense_data(laplace_pdf(&x, MU, B, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let single =
                dense_data(laplace_pdf(xi_aligned.as_slice(), MU, B, None, None).unwrap())[0];
            assert_close(bulk[i], single, 1e-15);
        }
    }

    // CDF - correctness & tails
    #[test]
    fn cdf_reference_values() {
        let x = vec64![-3.0, 0.0, 1.2, 2.5, 10.0];
        let expect: Vec<f64> = x.iter().copied().map(ref_cdf).collect();
        let out = dense_data(laplace_cdf(&x, MU, B, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn cdf_tails() {
        let x = vec64![-1e308, 1e308];
        let out = dense_data(laplace_cdf(&x, MU, B, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15);
        assert_close(out[1], 1.0, 1e-15);
    }

    // Quantile - reference, round-trip, domain edges
    #[test]
    fn quantile_reference_values() {
        let p = vec64![0.0, 0.1, 0.5, 0.9, 1.0];
        let expect: Vec<f64> = p.iter().copied().map(ref_quantile).collect();
        let out = dense_data(laplace_quantile(&p, MU, B, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            // tails are ~1e-14, others tighter
            assert_close(*a, *e, 2e-14);
        }
    }

    #[test]
    fn quantile_cdf_roundtrip() {
        let x = vec64![-1.0, 0.3, 1.2, 2.8];
        let p = dense_data(laplace_cdf(&x, MU, B, None, None).unwrap());
        let x2 = dense_data(laplace_quantile(&p, MU, B, None, None).unwrap());
        for (orig, back) in x.iter().zip(x2.iter()) {
            assert_close(*orig, *back, 2e-13);
        }
    }

    #[test]
    fn quantile_domain_and_edges() {
        let p = vec64![-0.1, 0.0, 1.0, 1.1, f64::NAN];
        let q = dense_data(laplace_quantile(&p, MU, B, None, None).unwrap());
        assert!(q[0].is_nan());
        assert!(q[1].is_infinite() && q[1].is_sign_negative());
        assert!(q[2].is_infinite() && q[2].is_sign_positive());
        assert!(q[3].is_nan());
        assert!(q[4].is_nan());
    }

    // Mask propagation
    #[test]
    fn pdf_mask_propagation() {
        let x = vec64![-1.0, 0.0, 3.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(1, false) };
        let arr = laplace_pdf(&x, MU, B, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    #[test]
    fn quantile_mask_propagation() {
        let p = vec64![0.2, 0.6, 0.8];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(0, false) };
        let arr = laplace_quantile(&p, MU, B, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![false, true, true]);
        assert!(arr.data[0].is_nan());
    }

    // Error cases & empty slice
    #[test]
    fn invalid_params_error() {
        assert!(laplace_pdf(&[0.0], 0.0, -1.0, None, None).is_err());
        assert!(laplace_cdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
        assert!(laplace_quantile(&[0.5], 0.0, 0.0, None, None).is_err());
    }

    #[test]
    fn empty_input() {
        let arr = laplace_pdf(&[], MU, B, None, None).unwrap();
        assert!(arr.data.is_empty() && arr.null_mask.is_none());
    }
}
