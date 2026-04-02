// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Cauchy Distribution Module** - *Heavy-Tailed Continuous Distribution*
//!
//! High-performance implementation of the Cauchy distribution with SIMD-accelerated kernels
//! for probability density function (PDF), cumulative distribution function (CDF), and
//! quantile (inverse CDF) calculations.
//!
//! ## Overview
//! The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability
//! distribution with heavy tails and undefined mean and variance. It is characterised by its
//! location parameter (median) and scale parameter (half-width at half-maximum).
//!
//! ## Mathematical Definition
//! - **PDF**: f(x; x₀, γ) = (1/π) × [γ / ((x - x₀)² + γ²)]
//! - **CDF**: F(x; x₀, γ) = (1/2) + (1/π) × arctan((x - x₀)/γ)  
//! - **Support**: x ∈ (-∞, +∞)
//! - **Parameters**: x₀ ∈ ℝ (location), γ > 0 (scale)
//!
//! ## Use Cases
//! - Physics: Lorentzian line shapes in spectroscopy and resonance phenomena
//! - Finance: Modelling extreme market movements and fat-tail risk
//! - Statistics: Robust statistics and outlier-resistant methods
//! - Signal processing: Natural line widths and frequency responses
//! - Machine learning: Heavy-tailed priors and Bayesian inference
//! - Geology: Modelling fracture orientations and directional data

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Computes the probability density function (PDF) of the Cauchy distribution.
///
/// Calculates f(x; x₀, γ) = (1/π) × [γ / ((x - x₀)² + γ²)] for each element
/// in the input array, where x₀ is the location parameter and γ is the scale parameter.
///
/// ## Parameters
/// - `x`: Array of values to evaluate
/// - `location`: Location parameter x₀ (median of the distribution)
/// - `scale`: Scale parameter γ > 0 (related to the full width at half maximum)
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `FloatArray<f64>` containing PDF values, with nulls propagated from input mask.
///
/// ## Behaviour
/// - All real values are valid inputs (infinite support)
/// - Maximum value occurs at x = location with f(location) = 1/(π × scale)
/// - Heavy tails: f(x) ~ γ/(π × (x - x₀)²) for large |x - x₀|
/// - Symmetric about the location parameter
/// - Input nulls propagate to output nulls
///
/// ## Errors
/// Returns `KernelError::InvalidArguments` if scale ≤ 0 or parameters are non-finite.
///
/// ## Example
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::cauchy::cauchy_pdf;
/// use minarrow::vec64;
///
/// let x = vec64![-2.0, 0.0, 2.0];
/// let result = cauchy_pdf(&x, 0.0, 1.0, None, None).unwrap();
/// // Returns PDF values for standard Cauchy distribution
/// ```
#[inline(always)]
pub fn cauchy_pdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::cauchy_pdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::cauchy_pdf_std(x, location, scale, null_mask, null_count)
    }
}

/// Computes the cumulative distribution function (CDF) of the Cauchy distribution.
///
/// Calculates F(x; x₀, γ) = (1/2) + (1/π) × arctan((x - x₀)/γ) for each element
/// in the input array, representing P(X ≤ x) where X ~ Cauchy(x₀, γ).
///
/// ## Parameters
/// - `x`: Array of values to evaluate
/// - `location`: Location parameter x₀ (median of the distribution)
/// - `scale`: Scale parameter γ > 0 (related to the interquartile range)
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `FloatArray<f64>` containing CDF values in (0, 1), with nulls propagated from input mask.
///
/// ## Behaviour
/// - All real values are valid inputs (infinite support)
/// - F(-∞) = 0, F(+∞) = 1, F(location) = 0.5 (median)
/// - Strictly increasing and continuous
/// - Symmetric about location: F(location + t) = 1 - F(location - t)
/// - Input nulls propagate to output nulls
///
/// ## Errors
/// Returns `KernelError::InvalidArguments` if scale ≤ 0 or parameters are non-finite.
///
/// ## Example
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::cauchy::cauchy_cdf;
/// use minarrow::vec64;
///
/// let x = vec64![-1.0, 0.0, 1.0];
/// let result = cauchy_cdf(&x, 0.0, 1.0, None, None).unwrap();
/// // Returns [0.25, 0.5, 0.75] for standard Cauchy distribution
/// ```
#[inline(always)]
pub fn cauchy_cdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::cauchy_cdf_std(x, location, scale, null_mask, null_count)
}

/// Computes the quantile function (inverse CDF) of the Cauchy distribution.
///
/// Calculates F⁻¹(p; x₀, γ) = x₀ + γ × tan(π × (p - 1/2)) for each element
/// in the input array, where F⁻¹ is the inverse CDF of Cauchy(x₀, γ).
///
/// ## Parameters
/// - `p`: Array of probability values to evaluate, must be in (0, 1)
/// - `location`: Location parameter x₀ (median of the distribution)
/// - `scale`: Scale parameter γ > 0 (related to the interquartile range)
/// - `null_mask`: Optional validity mask (Arrow-style: 1=valid, 0=null)
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `FloatArray<f64>` containing quantile values, with nulls propagated from input mask.
///
/// ## Errors
/// Returns `KernelError::InvalidArguments` if scale ≤ 0 or parameters are non-finite.
///
/// ## Example
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::cauchy::cauchy_quantile;
/// use minarrow::vec64;
///
/// let p = vec64![0.1, 0.25, 0.5, 0.75, 0.9];
/// let result = cauchy_quantile(&p, 0.0, 1.0, None, None).unwrap();
/// // Returns quantiles for standard Cauchy distribution
/// ```
#[inline(always)]
pub fn cauchy_quantile(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::cauchy_quantile_std(p, location, scale, null_mask, null_count)
}

// Zero-allocation variants

/// Cauchy PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x; x₀, γ) = (1/π) × [γ / ((x - x₀)² + γ²)]
#[inline(always)]
pub fn cauchy_pdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::cauchy_pdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::cauchy_pdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Cauchy CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x; x₀, γ) = (1/2) + (1/π) × arctan((x - x₀)/γ)
#[inline(always)]
pub fn cauchy_cdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::cauchy_cdf_std_to(x, location, scale, output, null_mask, null_count)
}

/// Cauchy quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p; x₀, γ) = x₀ + γ × tan(π × (p - 1/2))
#[inline(always)]
pub fn cauchy_quantile_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::cauchy_quantile_std_to(p, location, scale, output, null_mask, null_count)
}

#[cfg(test)]
mod cauchy_tests {
    use ::std::f64::consts::PI;

    use crate::kernels::scientific::distributions::{
        shared::constants::INV_PI, univariate::common::dense_data,
    };

    use super::*;
    use minarrow::{Bitmask, Vec64, vec64};

    // See `./tests` for the scipy test suite

    // helpers

    fn mask_vec(m: &Bitmask) -> Vec64<bool> {
        (0..m.len()).map(|i| m.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // Reference cases used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"
    fn cauchy_pdf_ref(x: f64, loc: f64, scale: f64) -> f64 {
        INV_PI * scale / ((x - loc).powi(2) + scale.powi(2))
    }
    fn cauchy_cdf_ref(x: f64, loc: f64, scale: f64) -> f64 {
        0.5 + INV_PI * ((x - loc) / scale).atan()
    }
    fn cauchy_quantile_ref(p: f64, loc: f64, scale: f64) -> f64 {
        loc + scale * ((p - 0.5) * PI).tan()
    }

    // PDF - correctness & consistency
    #[test]
    fn cauchy_pdf_basic_values() {
        let xs = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expect: Vec<f64> = xs.iter().map(|&x| cauchy_pdf_ref(x, 0.0, 1.0)).collect();
        let out = dense_data(cauchy_pdf(&xs, 0.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn cauchy_pdf_location_scale() {
        let xs = vec64![0.0, 2.0, 5.0];
        let (loc, sc) = (1.5, 3.0);
        let expect: Vec<f64> = xs.iter().map(|&x| cauchy_pdf_ref(x, loc, sc)).collect();
        let out = dense_data(cauchy_pdf(&xs, loc, sc, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn cauchy_pdf_bulk_vs_scalar() {
        let xs = vec64![-5.0, -1.0, 0.3, 2.2, 7.0];
        let bulk = dense_data(cauchy_pdf(&xs, 0.0, 1.0, None, None).unwrap());
        for (i, &x) in xs.iter().enumerate() {
            let xi_aligned = vec64![x];
            let scalar =
                dense_data(cauchy_pdf(xi_aligned.as_slice(), 0.0, 1.0, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    #[test]
    fn cauchy_pdf_inf_inputs_zero() {
        let xs = vec64![f64::INFINITY, f64::NEG_INFINITY];
        let out = dense_data(cauchy_pdf(&xs, 0.0, 1.0, None, None).unwrap());
        assert_eq!(out, vec64![0.0, 0.0]);
    }

    // PDF - mask propagation
    #[test]
    fn cauchy_pdf_mask_propagation() {
        let xs = vec64![0.0, 1.0, 2.0];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(1, false);
        }
        let arr = cauchy_pdf(&xs, 0.0, 1.0, Some(&m), Some(1)).unwrap();
        assert_eq!(
            mask_vec(arr.null_mask.as_ref().unwrap()),
            vec64![true, false, true]
        );
        assert!(arr.data[1].is_nan());
    }

    // CDF - correctness & symmetry
    #[test]
    fn cauchy_cdf_standard_values() {
        let xs = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expect: Vec<f64> = xs.iter().map(|&x| cauchy_cdf_ref(x, 0.0, 1.0)).collect();
        let out = dense_data(cauchy_cdf(&xs, 0.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn cauchy_cdf_symmetry() {
        let xs = vec64![-5.0, -1.3, 0.0, 0.7, 4.2];
        let arr = dense_data(cauchy_cdf(&xs, 0.0, 1.0, None, None).unwrap());
        let arr_reflected = dense_data(
            cauchy_cdf(
                &xs.iter().map(|x| -x).collect::<Vec<_>>(),
                0.0,
                1.0,
                None,
                None,
            )
            .unwrap(),
        );
        for (f, b) in arr.iter().zip(arr_reflected.iter()) {
            assert_close(*f + *b, 1.0, 1e-15); // F(x) + F(-x) == 1
        }
    }

    #[test]
    fn cauchy_cdf_tails() {
        let xs = vec64![-1e308, 1e308];
        let out = dense_data(cauchy_cdf(&xs, 0.0, 1.0, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15);
        assert_close(out[1], 1.0, 1e-15);
    }

    // CDF - mask / error handling
    #[test]
    fn cauchy_cdf_mask_nan_propagation() {
        let xs = vec64![0.0, f64::NAN, 3.0];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(2, false);
        }
        let arr = cauchy_cdf(&xs, 0.0, 1.0, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec64![true, true, false]);
        assert!(arr.data[2].is_nan());
    }

    // Quantile - correctness & round-trip
    #[test]
    fn cauchy_quantile_known_probs() {
        let ps = vec64![0.01, 0.25, 0.5, 0.75, 0.99];
        let expect: Vec<f64> = ps
            .iter()
            .map(|&p| cauchy_quantile_ref(p, 0.0, 1.0))
            .collect();
        let out = dense_data(cauchy_quantile(&ps, 0.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    #[test]
    fn cauchy_quantile_location_scale() {
        let ps = vec64![0.1, 0.9];
        let (loc, sc) = (2.0, 4.0);
        let exp: Vec<f64> = ps
            .iter()
            .map(|&p| cauchy_quantile_ref(p, loc, sc))
            .collect();
        let out = dense_data(cauchy_quantile(&ps, loc, sc, None, None).unwrap());
        for (a, e) in out.iter().zip(exp.iter()) {
            assert_close(*a, *e, 1e-13);
        }
    }

    #[test]
    fn cauchy_quantile_reflection_roundtrip() {
        // symmetry: Q(p) == -Q(1-p) for standard cauchy
        let ps = vec64![1e-6, 0.1, 0.3, 0.7, 0.9, 1.0 - 1e-6];
        let q = dense_data(cauchy_quantile(&ps, 0.0, 1.0, None, None).unwrap());
        let q_r = dense_data(
            cauchy_quantile(
                &ps.iter().map(|p| 1.0 - p).collect::<Vec<_>>(),
                0.0,
                1.0,
                None,
                None,
            )
            .unwrap(),
        );
        for (u, v) in q.iter().zip(q_r.iter()) {
            assert_close(*u, -*v, 1e-13);
        }

        // round-trip with CDF
        let xs = vec64![-10.0, -1.0, 0.0, 1.0, 8.0];
        let pvals = dense_data(cauchy_cdf(&xs, 0.0, 1.0, None, None).unwrap());
        let back = dense_data(cauchy_quantile(&pvals, 0.0, 1.0, None, None).unwrap());
        for (x, b) in xs.iter().zip(back.iter()) {
            assert_close(*x, *b, 1e-13);
        }
    }

    #[test]
    fn cauchy_quantile_domain_edges() {
        let ps = vec64![0.0, 1.0, -0.1, 1.1, f64::NAN];
        let out = dense_data(cauchy_quantile(&ps, 0.0, 1.0, None, None).unwrap());
        assert!(out[0].is_infinite() && out[0].is_sign_negative());
        assert!(out[1].is_infinite() && out[1].is_sign_positive());
        assert!(out[2].is_nan() && out[3].is_nan() && out[4].is_nan());
    }

    // Quantile - mask propagation
    #[test]
    fn cauchy_quantile_mask_propagation() {
        let ps = vec64![0.2, 0.5, 0.8];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(0, false);
        }
        let arr = cauchy_quantile(&ps, 0.0, 1.0, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec64![false, true, true]);
        assert!(arr.data[0].is_nan());
    }

    // Parameter validation & empty input
    #[test]
    fn cauchy_invalid_param_errors() {
        assert!(cauchy_pdf(&[0.0], 0.0, -1.0, None, None).is_err());
        assert!(cauchy_cdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
        assert!(cauchy_quantile(&[0.5], 0.0, 0.0, None, None).is_err());
    }

    #[test]
    fn cauchy_empty_inputs() {
        assert!(
            cauchy_pdf(&[], 0.0, 1.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            cauchy_cdf(&[], 0.0, 1.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            cauchy_quantile(&[], 0.0, 1.0, None, None)
                .unwrap()
                .data
                .is_empty()
        );
    }
}
