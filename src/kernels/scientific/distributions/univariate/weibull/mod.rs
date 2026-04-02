// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Weibull Distribution Module** - *SIMD Accelerated Survival Analysis*
//!
//! High-performance implementation of the Weibull distribution, fundamental to reliability
//! engineering, survival analysis, and extreme value theory. This distribution also has
//! broad applications in engineering and life sciences.
//!
//! ## Overview
//! - **Domain**: `x ≥ 0` (non-negative real numbers)
//! - **Parameters**: `k > 0` (shape), `λ > 0` (scale)
//! - **PDF**: `f(x) = (k/λ) × (x/λ)^{k-1} × exp(-(x/λ)^k)`
//! - **CDF**: `F(x) = 1 - exp(-(x/λ)^k)`
//! - **Mean**: `E[X] = λ × Γ(1 + 1/k)`
//! - **Variance**: `Var[X] = λ² × [Γ(1 + 2/k) - Γ²(1 + 1/k)]`
//!
//! ## Statistical Applications
//! The Weibull distribution is essential across multiple domains:
//! - **Reliability engineering**: component failure analysis and lifetime modelling
//! - **Survival analysis**: time-to-event data in medical and biological studies
//! - **Extreme value theory**: modelling maximum and minimum values
//! - **Materials science**: fatigue analysis and strength characterisation
//! - **Weather modelling**: wind speed distributions and extreme weather events
//! - **Quality control**: defect analysis and process capability studies
//!
//! ## Shape Parameter Interpretation
//! The shape parameter `k` characterises failure behaviour:
//! - **k < 1**: Decreasing hazard rate
//! - **k = 1**: Constant hazard rate
//! - **k > 1**: Increasing hazard rate
//! - **k = 2**: Rayleigh distribution
//! - **k -> ∞**: Approaches normal distribution
//!
//! ## Usage Examples
//! ```rust,ignore
//! use minarrow::{vec64, FloatArray};
//! use simd_kernels::kernels::scientific::distributions::univariate::weibull::*;
//!
//! // Component reliability analysis
//! let times = vec64![100.0, 500.0, 1000.0, 2000.0, 5000.0]; // operating hours
//! let shape = 2.0;   // increasing hazard rate (wear-out)
//! let scale = 1000.0; // characteristic life
//!
//! let failure_pdf = weibull_pdf(&times, shape, scale, None, None).unwrap();
//! let reliability = weibull_cdf(&times, shape, scale, None, None).unwrap();
//!
//! // Find design life for 90% reliability
//! let reliability_targets = vec64![0.10]; // 10% failure rate
//! let design_life = weibull_quantile(&reliability_targets, shape, scale, None, None).unwrap();
//!
//! // Wind speed modelling (Rayleigh case: k=2)
//! let wind_speeds = vec64![5.0, 10.0, 15.0, 20.0, 25.0]; // m/s
//! let wind_pdf = weibull_pdf(&wind_speeds, 2.0, 12.0, None, None).unwrap();
//! ```
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Weibull PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|k,λ) = (k/λ) × (x/λ)^(k-1) × exp(-(x/λ)^k) for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_pdf_to(
    x: &[f64],
    shape: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::weibull_pdf_simd_to(x, shape, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::weibull_pdf_std_to(x, shape, scale, output, null_mask, null_count)
    }
}

/// Weibull CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|k,λ) = 1 − exp[−(x/λ)^k] for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_cdf_to(
    x: &[f64],
    shape: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::weibull_cdf_simd_to(x, shape, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::weibull_cdf_std_to(x, shape, scale, output, null_mask, null_count)
    }
}

/// Weibull quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p|k,λ) = λ × [−ln(1−p)]^(1/k), p ∈ (0,1)
#[inline(always)]
pub fn weibull_quantile_to(
    p: &[f64],
    shape: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::weibull_quantile_simd_to(p, shape, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::weibull_quantile_std_to(p, shape, scale, output, null_mask, null_count)
    }
}

/// Compute the probability density function (PDF) for the Weibull distribution.
///
/// The Weibull distribution is a continuous probability distribution widely used in reliability
/// engineering, survival analysis, and extreme value theory. It is particularly valuable for
/// modelling time-to-failure data and characterising the life distribution of materials,
/// components, and systems.
///
/// ## Mathematical Definition
///
/// The PDF of the Weibull distribution is defined as:
///
/// ```text
/// f(x; k, λ) = (k/λ) × (x/λ)^(k-1) × exp(-(x/λ)^k)   for x ≥ 0
/// f(x; k, λ) = 0                                       for x < 0
/// ```
///
/// where:
/// - `k` (shape) is the shape parameter, with `k > 0`
/// - `λ` (scale) is the scale parameter, with `λ > 0`
///
/// ## Parameters
///
/// * `x` - Input values to evaluate the PDF at (non-negative for non-zero PDF)
/// * `shape` - Shape parameter k, must be positive (k > 0)
/// * `scale` - Scale parameter λ, must be positive (λ > 0)
/// * `null_mask` - Optional bitmask for null value handling in Arrow format
/// * `null_count` - Optional count of null values for optimisation
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing the PDF values, or a `KernelError` if:
/// - Shape parameter is non-positive (k ≤ 0)
/// - Scale parameter is non-positive (λ ≤ 0)
/// - Either parameter is not finite (NaN or infinite)
///
/// ## Examples
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::weibull::weibull_pdf;
/// use minarrow::vec64;
///
/// // Component reliability analysis
/// let lifetimes = vec64![100.0, 500.0, 1000.0, 2000.0]; // operating hours
/// let shape = 2.5;   // increasing hazard rate
/// let scale = 1000.0; // characteristic life (63.2% failure point)
/// let pdf = weibull_pdf(&lifetimes, shape, scale, None, None).unwrap();
/// ```
/// ## Applications
///
/// - **Reliability engineering**: Component failure analysis, MTBF calculations
/// - **Survival analysis**: Time-to-event modelling in medical and biological research
/// - **Extreme value theory**: Maximum wind speeds, flood levels, earthquake magnitudes
/// - **Materials science**: Fatigue life, fracture strength, durability testing
/// - **Quality control**: Process capability analysis, defect occurrence modelling
/// - **Finance**: Risk assessment, extreme loss modelling
/// - **Environmental science**: Precipitation extremes, pollutant concentrations
#[inline(always)]
pub fn weibull_pdf(
    x: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::weibull_pdf_simd(x, shape, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::weibull_pdf_std(x, shape, scale, null_mask, null_count)
    }
}

/// Weibull CDF: F(x; k, λ) = 1 − exp[−(x/λ)^k]  for x ≥ 0, else 0
#[inline(always)]
pub fn weibull_cdf(
    x: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::weibull_cdf_simd(x, shape, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::weibull_cdf_std(x, shape, scale, null_mask, null_count)
    }
}

/// Weibull quantile (inverse CDF):
/// Q(p; k, λ) = λ · [−ln(1−p)]^(1/k),  p ∈ (0,1)
#[inline(always)]
pub fn weibull_quantile(
    p: &[f64],
    shape: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::weibull_quantile_simd(p, shape, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::weibull_quantile_std(p, shape, scale, null_mask, null_count)
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // see "./tests" for scipy test suite

    // Helpers

    fn mask_vec(bm: &Bitmask) -> Vec<bool> {
        (0..bm.len()).map(|i| bm.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        // Handle special cases for infinity
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            return; // Both are the same infinity
        }
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: {a} vs {b} (tol = {tol})"
        );
    }

    // Parameters used throughout
    const K: f64 = 1.5; // shape k
    const L: f64 = 2.0; // scale λ

    // Reference cases used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"

    fn ref_pdf(x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            let t = x / L;
            (K / L) * t.powf(K - 1.0) * (-(t.powf(K))).exp()
        }
    }
    fn ref_cdf(x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-(x / L).powf(K)).exp()
        }
    }
    fn ref_quantile(p: f64) -> f64 {
        if p == 0.0 {
            0.0
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            L * (-(1.0 - p).ln()).powf(1.0 / K)
        }
    }

    // PDF - core correctness
    #[test]
    fn pdf_reference_values() {
        let x = vec64![-1.0, 0.0, 0.5, 1.2, 4.0];
        let expect: Vec<f64> = x.iter().map(|&v| ref_pdf(v)).collect();
        let out = dense_data(weibull_pdf(&x, K, L, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn pdf_bulk_vs_scalar() {
        let x = vec64![0.0, 0.3, 1.1, 5.3];
        let bulk = dense_data(weibull_pdf(&x, K, L, None, None).unwrap());
        for (i, &v) in x.iter().enumerate() {
            let v_aligned = vec64![v];
            let sc = dense_data(weibull_pdf(v_aligned.as_slice(), K, L, None, None).unwrap())[0];
            assert_close(bulk[i], sc, 1e-15);
        }
    }

    // CDF - correctness and tails
    #[test]
    fn cdf_reference_values() {
        let x = vec64![-2.0, 0.0, 0.8, 2.5, 10.0];
        let expect: Vec<f64> = x.iter().map(|&v| ref_cdf(v)).collect();
        let out = dense_data(weibull_cdf(&x, K, L, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn cdf_tail_extremes() {
        // Very small and very large x
        let x = vec64![-1e6, 0.0, 1e6];
        let out = dense_data(weibull_cdf(&x, K, L, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15);
        assert_close(out[1], 0.0, 1e-15);
        assert_close(out[2], 1.0, 1e-15);
    }

    // Quantile - reference, round-trip, edge & domain
    #[test]
    fn quantile_reference_values() {
        let p = vec64![0.0, 0.1, 0.5, 0.9, 1.0];
        let expect: Vec<f64> = p.iter().map(|&q| ref_quantile(q)).collect();
        let out = dense_data(weibull_quantile(&p, K, L, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-13);
        }
    }

    #[test]
    fn quantile_cdf_roundtrip() {
        let x = vec64![0.1, 0.8, 3.5, 6.0];
        let p = dense_data(weibull_cdf(&x, K, L, None, None).unwrap());
        let x2 = dense_data(weibull_quantile(&p, K, L, None, None).unwrap());
        for (orig, back) in x.iter().zip(x2.iter()) {
            assert_close(*orig, *back, 3e-13);
        }
    }

    #[test]
    fn quantile_domain_and_edges() {
        let p = vec64![-0.2, 0.0, 1.0, 1.2, f64::NAN];
        let q = dense_data(weibull_quantile(&p, K, L, None, None).unwrap());
        assert!(q[0].is_nan()); // <0
        assert_close(q[1], 0.0, 0.0); // ==0
        assert!(q[2].is_infinite()); // ==1
        assert!(q[3].is_nan()); // >1
        assert!(q[4].is_nan()); // NaN in -> NaN out
    }

    // Mask propagation tests
    #[test]
    fn pdf_mask_propagation() {
        let x = vec64![-1.0, 0.0, 2.0, 5.0];
        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(2, false) }; // mask-out index 2
        let arr = weibull_pdf(&x, K, L, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, true, false, true]);
        assert!(arr.data[2].is_nan());
    }

    #[test]
    fn quantile_mask_propagation() {
        let p = vec64![0.2, 0.5, 0.7];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(0, false) };
        let arr = weibull_quantile(&p, K, L, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert!(!m[0] && m[1] && m[2]);
        assert!(arr.data[0].is_nan());
    }

    // Error handling & empty input
    #[test]
    fn invalid_parameter_errors() {
        assert!(weibull_pdf(&[0.0], -1.0, 1.0, None, None).is_err());
        assert!(weibull_cdf(&[0.0], 2.0, 0.0, None, None).is_err());
        assert!(weibull_quantile(&[0.5], f64::NAN, 1.0, None, None).is_err());
    }

    #[test]
    fn empty_input() {
        let arr = weibull_pdf(&[], K, L, None, None).unwrap();
        assert!(arr.data.is_empty() && arr.null_mask.is_none());
    }
}
