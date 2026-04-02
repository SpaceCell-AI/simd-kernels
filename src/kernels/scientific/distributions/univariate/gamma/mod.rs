// Copyright (c) 2025 Peter Bower
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # Gamma Distribution
//!
//! The gamma distribution is a two-parameter continuous probability distribution widely used in
//! statistical modelling, particularly for positive-valued random variables. It generalises the
//! exponential distribution and serves as a conjugate prior for precision parameters in Bayesian
//! statistics.
//!
//! ## Mathematical Definition
//!
//! The gamma distribution is parameterised by a shape parameter α (alpha) and a rate parameter β
//! (beta), both strictly positive:
//!
//! - **PDF**: f(x; α, θ) = x^(α-1) exp(-x/θ) / (Γ(α) θ^α) for x ≥ 0
//! - **CDF**: F(x; α, θ) = γ(α, x/θ) / Γ(α)
//! - **Quantile**: x such that F(x) = p
//!
//! Where Γ(α) is the gamma function and γ(α, z) is the lower incomplete gamma function.
//!
//! ## Common Applications
//!
//! - **Reliability engineering**: Time-to-failure analysis
//! - **Queuing theory**: Inter-arrival times in Poisson processes
//! - **Bayesian statistics**: Conjugate prior for precision parameters
//! - **Economics**: Modelling income distributions and waiting times
//! - **Meteorology**: Rainfall and precipitation modelling

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::Bitmask;
use minarrow::FloatArray;
use minarrow::enums::error::KernelError;

/// Gamma distribution PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn gamma_pdf_to(
    x: &[f64],
    shape: f64,
    rate: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gamma_pdf_simd_to(x, shape, rate, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gamma_pdf_std_to(x, shape, rate, output, null_mask, null_count)
    }
}

/// Computes the probability density function (PDF) of the gamma distribution.
///
/// Evaluates f(x; α, β) = β^α x^(α-1) exp(-βx) / Γ(α) for each element in the input array,
/// where α is the shape parameter and β is the rate parameter.
///
/// # Mathematical Definition
///
/// The gamma PDF in rate parameterisation is:
/// ```text
/// f(x; α, β) = β^α x^(α-1) exp(-βx) / Γ(α)
/// ```
///
/// For x ≥ 0, α > 0 (shape), β > 0 (rate). The function returns 0 for x < 0.
///
/// # Special Cases
///
/// - When x = 0:
///   - If α > 1: f(0) = 0
///   - If α < 1: f(0) = +∞
///   - If α = 1: f(0) = β (reduces to exponential distribution)
/// - When x < 0: f(x) = 0 (gamma distribution has support [0, ∞))
/// - When x or parameters are non-finite: returns NaN
///
/// # Parameters
///
/// * `x` - Input values where the PDF is evaluated
/// * `shape` - Shape parameter α > 0
/// * `rate` - Rate parameter β > 0, the reciprocal of the scale parameter θ
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the PDF values, or a `KernelError` if parameters
/// are invalid (non-positive or non-finite).
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::gamma::gamma_pdf;
/// use minarrow::vec64;
///
/// let x = vec64![0.5, 1.0, 2.0];
/// let result = gamma_pdf(&x, 2.0, 1.0, None, None).unwrap();
/// // For shape=2, rate=1: PDF values for exponential squared distribution
/// ```
#[inline(always)]
pub fn gamma_pdf(
    x: &[f64],
    shape: f64,
    rate: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gamma_pdf_simd(x, shape, rate, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gamma_pdf_std(x, shape, rate, null_mask, null_count)
    }
}

/// Gamma distribution CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn gamma_cdf_to(
    x: &[f64],
    shape: f64,
    rate: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::gamma_cdf_std_to(x, shape, rate, output, null_mask, null_count)
}

/// Computes the cumulative distribution function (CDF) of the gamma distribution.
///
/// Evaluates F(x; α, β) = P(α, βx) = γ(α, βx) / Γ(α) for each element in the input array,
/// where P is the regularised lower incomplete gamma function.
///
/// # Mathematical Definition
///
/// The gamma CDF in rate parameterisation is:
/// ```text
/// F(x; α, β) = P(α, βx) = γ(α, βx) / Γ(α)
/// ```
///
/// Where γ(α, z) is the lower incomplete gamma function and Γ(α) is the gamma function.
///
/// # Domain and Range
///
/// - **Domain**: x ∈ [0, ∞) (returns 0 for x ≤ 0)
/// - **Range**: [0, 1]
/// - **Monotonicity**: Strictly increasing for x > 0
///
/// # Parameters
///
/// * `x` - Input values where the CDF is evaluated
/// * `shape` - Shape parameter α > 0
/// * `rate` - Rate parameter β > 0, the reciprocal of the scale parameter θ
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the CDF values, or a `KernelError` if parameters
/// are invalid (non-positive or non-finite).
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::gamma::gamma_cdf;
/// use minarrow::vec64;
///
/// let x = vec64![0.5, 1.0, 2.0];
/// let result = gamma_cdf(&x, 2.0, 1.0, None, None).unwrap();
/// // Returns cumulative probabilities for each x value
/// ```
#[inline(always)]
pub fn gamma_cdf(
    x: &[f64],
    shape: f64,
    rate: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::gamma_cdf_std(x, shape, rate, null_mask, null_count)
}

/// Gamma distribution quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn gamma_quantile_to(
    p: &[f64],
    shape: f64,
    rate: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::gamma_quantile_std_to(p, shape, rate, output, null_mask, null_count)
}

/// Computes the quantile function (inverse CDF) of the gamma distribution.
///
/// Finds x such that F(x; α, β) = p for each probability p in the input array.
/// This is the inverse of the gamma CDF function.
///
/// # Mathematical Definition
///
/// The gamma quantile function finds x such that:
/// ```text
/// P(α, βx) = p
/// ```
///
/// Where P is the regularised lower incomplete gamma function.
///
/// # Algorithm Details
///
/// Uses robust numerical inversion with region-specific algorithms:
/// - **Central region** (p ∈ [1e-10, 1-1e-10]): Lower incomplete gamma inversion
/// - **Right tail** (p > 1-1e-10): Upper incomplete gamma inversion for stability
/// - **Left tail** (p < 1e-10): Direct lower tail algorithm
///
/// # Domain and Range
///
/// - **Domain**: p ∈ [0, 1] (returns NaN for p outside this range)
/// - **Range**: x ∈ [0, ∞)
/// - **Boundary conditions**: Q(0) = 0, Q(1) = +∞
///
/// # Parameters
///
/// * `p` - Probability values in [0, 1] for which quantiles are computed
/// * `shape` - Shape parameter α > 0
/// * `rate` - Rate parameter β > 0, the reciprocal of the scale parameter θ
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the quantile values, or a `KernelError` if parameters
/// are invalid (non-positive or non-finite).
///
/// # Numerical Accuracy
///
/// - Round-trip accuracy: CDF(quantile(p)) ≈ p with tolerance ~1e-12
/// - Maintains precision across the full parameter space
/// - Robust handling of extreme probabilities (near 0 and 1)
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::gamma::gamma_quantile;
/// use minarrow::vec64;
///
/// let p = vec64![0.25, 0.5, 0.75];
/// let result = gamma_quantile(&p, 2.0, 1.0, None, None).unwrap();
/// // Returns x values such that CDF(x) equals each probability
/// ```
#[inline(always)]
pub fn gamma_quantile(
    p: &[f64],
    shape: f64,
    rate: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::gamma_quantile_std(p, shape, rate, null_mask, null_count)
}

#[cfg(test)]
mod gamma_tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // See `./tests` for the scipy test suite

    // ---------- helpers ---------------------------------------------------

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: {a} vs {b} (tol {tol})"
        );
    }

    // gamma_pdf  - correctness

    #[test]
    fn gamma_pdf_shape2_rate1_values() {
        // For k=2, θ=1   ⇒   f(x)=x e^{-x}
        let x = vec64![0.0, 0.5, 1.0, 2.0];
        let expect = vec64![
            0.0,                   // 0 * e^0
            0.5 * (-0.5f64).exp(), // 0.3032653298563167
            1.0 * (-1.0f64).exp(), // 0.3678794411714423
            2.0 * (-2.0f64).exp(), // 0.2706705664732254
        ];
        let out = dense_data(gamma_pdf(&x, 2.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-15);
        }
    }

    #[test]
    fn gamma_pdf_shape3_rate_half_example() {
        // k=3, β=0.5 (rate parameterization)  ⇒  f(x)=β^k x^{k-1} e^{-βx} / Γ(k) = 0.5^3 x^{2} e^{-0.5x} / Γ(3) = x^{2} e^{-x/2} / 16
        let x = vec64![0.0, 2.0, 4.0];
        let expect = vec64![
            0.0,
            (2.0_f64.powi(2) * (-1.0f64).exp()) / 16.0, // x=2
            (4.0_f64.powi(2) * (-2.0f64).exp()) / 16.0, // x=4
        ];
        let out = dense_data(gamma_pdf(&x, 3.0, 0.5, None, None).unwrap()); // rate = 0.5, not scale = 2.0
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-15);
        }
    }

    #[test]
    fn gamma_pdf_negative_x_zero() {
        let out = dense_data(gamma_pdf(&[-3.0, -0.1], 2.0, 1.0, None, None).unwrap());
        assert_eq!(out, vec64![0.0, 0.0]);
    }

    #[test]
    fn gamma_pdf_bulk_vs_scalar_consistency() {
        let x = vec64![0.1, 0.7, 3.2, 5.0];
        let k = 4.5;
        let θ = 1.3;
        let bulk = dense_data(gamma_pdf(&x, k, θ, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let scalar = dense_data(gamma_pdf(xi_aligned.as_slice(), k, θ, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    // gamma_cdf  - correctness

    #[test]
    fn gamma_cdf_shape2_rate1_values() {
        // For k=2, θ=1   ⇒   F(x)=1-(x+1)e^{-x}
        let x = vec64![0.0, 0.5, 1.0, 2.0];
        let expect = vec64![
            0.0,
            1.0 - (1.5) * (-0.5f64).exp(), // 0.090204010432...
            1.0 - 2.0 * (-1.0f64).exp(),   // 0.264241117657...
            1.0 - 3.0 * (-2.0f64).exp(),   // 0.593994150519...
        ];
        let out = dense_data(gamma_cdf(&x, 2.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-14);
        }
    }

    #[test]
    fn gamma_cdf_tail_behaviour() {
        let k = 5.0;
        let θ = 2.0;
        let out = dense_data(gamma_cdf(&[1e-12, 1e5], k, θ, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15); // tiny x
        assert_close(out[1], 1.0, 1e-12); // huge x
    }

    // gamma_quantile  - correctness & round-trip

    #[test]
    fn gamma_quantile_bounds_and_domain() {
        // 0 -> 0 , 1 -> +∞ , outside [0,1] -> NaN
        let k = 3.3;
        let θ = 0.8;
        let arr = dense_data(gamma_quantile(&[0.0, 1.0, -0.1, 1.1], k, θ, None, None).unwrap());
        assert_close(arr[0], 0.0, 0.0);
        assert!(arr[1].is_infinite() && arr[1].is_sign_positive());
        assert!(arr[2].is_nan() && arr[3].is_nan());
    }

    #[test]
    fn gamma_quantile_roundtrip_shape2_rate1() {
        let k = 2.0;
        let θ = 1.0;
        let x = vec64![0.2, 0.5, 1.3, 4.0];
        let p = dense_data(gamma_cdf(&x, k, θ, None, None).unwrap());
        let x2 = dense_data(gamma_quantile(&p, k, θ, None, None).unwrap());
        for (a, b) in x.iter().zip(x2.iter()) {
            assert_close(*a, *b, 5e-13);
        }
    }

    // mask propagation

    #[test]
    fn gamma_pdf_null_mask() {
        let x = vec64![0.0, 1.0, 2.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(1, false);
        }
        let arr = gamma_pdf(&x, 2.0, 1.0, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    #[test]
    fn gamma_cdf_null_mask() {
        let x = vec64![1.0, 2.0, 3.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(0, false);
        }
        let arr = gamma_cdf(&x, 2.0, 1.0, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![false, true, true]);
        assert!(arr.data[0].is_nan());
    }

    #[test]
    fn gamma_quantile_null_mask() {
        let p = vec64![0.2, 0.4, 0.6];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(2, false);
        }
        let arr = gamma_quantile(&p, 2.0, 1.0, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, true, false]);
        assert!(arr.data[2].is_nan());
    }

    // parameter validation

    #[test]
    fn gamma_invalid_parameters() {
        assert!(gamma_pdf(&[1.0], -1.0, 1.0, None, None).is_err());
        assert!(gamma_cdf(&[1.0], 1.0, 0.0, None, None).is_err());
        assert!(gamma_quantile(&[0.5], f64::NAN, 1.0, None, None).is_err());
    }

    // empty-input paths

    #[test]
    fn gamma_empty_inputs() {
        let pdf = gamma_pdf(&[], 2.0, 1.0, None, None).unwrap();
        let cdf = gamma_cdf(&[], 2.0, 1.0, None, None).unwrap();
        let qtl = gamma_quantile(&[], 2.0, 1.0, None, None).unwrap();
        assert!(pdf.data.is_empty() && cdf.data.is_empty() && qtl.data.is_empty());
    }
}
