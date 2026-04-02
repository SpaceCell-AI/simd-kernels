// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # Gumbel Distribution - **Type I Extreme Value**
//!
//! The Gumbel distribution, also known as the Type I extreme value distribution, is used to
//! model the maximum (or minimum) of a large number of independent, identically distributed
//! random variableIt is widely applied in fields such as hydrology, meteorology, and
//! reliability engineering for extreme event analysis.
//!
//! ## Mathematical Definition
//!
//! The Gumbel distribution is parameterised by a location parameter μ and a scale parameter β > 0:
//!
//! - **PDF**: f(x; μ, β) = (1/β) exp(-z - exp(-z)) where z = (x - μ)/β
//! - **CDF**: F(x; μ, β) = exp(-exp(-z)) where z = (x - μ)/β  
//! - **Quantile**: Q(p; μ, β) = μ - β ln(-ln(p)) for p ∈ (0, 1)
//!
//! ## Common Applications
//!
//! - **Hydrology**: Flood frequency analysis and extreme rainfall events
//! - **Meteorology**: Extreme wind speeds and temperature analysis
//! - **Engineering**: Reliability analysis and fatigue life modelling
//! - **Finance**: Value-at-Risk calculations for extreme market movements
//! - **Seismology**: Earthquake magnitude and frequency analysis
//! - **Quality control**: Extreme defect rates and process capability studies

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Computes the probability density function (PDF) of the Gumbel distribution.
///
/// Evaluates f(x; μ, β) = (1/β) exp(-z - exp(-z)) where z = (x - μ)/β for each
/// element in the input array.
///
/// # Mathematical Definition
///
/// The Gumbel PDF is:
/// ```text
/// f(x; μ, β) = (1/β) exp(-z - exp(-z))
/// where z = (x - μ)/β
/// ```
///
/// This represents the probability density for extreme values in the maximum domain.
///
/// # Parameters
///
/// * `x` - Input values where the PDF is evaluated
/// * `location` - Location parameter μ (real number, determines the mode)
/// * `scale` - Scale parameter β > 0 (determines the spread)
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the PDF values, or a `KernelError` if
/// the scale parameter is non-positive or non-finite, or if location is non-finite.
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::gumbell::gumbel_pdf;
/// use minarrow::vec64;
///
/// let x = vec64![-2.0, 0.0, 2.0];
/// let result = gumbel_pdf(&x, 0.0, 1.0, None, None).unwrap();
/// // Returns PDF values for standard Gumbel distribution
/// ```
#[inline(always)]
pub fn gumbel_pdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gumbel_pdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gumbel_pdf_std(x, location, scale, null_mask, null_count)
    }
}

/// Computes the cumulative distribution function (CDF) of the Gumbel distribution.
///
/// Evaluates F(x; μ, β) = exp(-exp(-z)) where z = (x - μ)/β for each element
/// in the input array.
///
/// # Mathematical Definition
///
/// The Gumbel CDF is:
/// ```text
/// F(x; μ, β) = exp(-exp(-z))
/// where z = (x - μ)/β
/// ```
///
/// This double exponential form gives the Gumbel distribution its characteristic
/// asymmetric shape with a heavy right tail.
///
/// # Parameters
///
/// * `x` - Input values where the CDF is evaluated
/// * `location` - Location parameter μ (real number)
/// * `scale` - Scale parameter β > 0
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the CDF values in [0, 1], or a
/// `KernelError` if parameters are invalid.
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::gumbell::gumbel_cdf;
/// use minarrow::vec64;
///
/// let x = vec64![-5.0, 0.0, 5.0];
/// let result = gumbel_cdf(&x, 0.0, 1.0, None, None).unwrap();
/// // Returns cumulative probabilities
/// ```
#[inline(always)]
pub fn gumbel_cdf(
    x: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gumbel_cdf_simd(x, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gumbel_cdf_std(x, location, scale, null_mask, null_count)
    }
}

/// Computes the quantile function (inverse CDF) of the Gumbel distribution.
///
/// Finds x such that F(x; μ, β) = p for each probability p in the input array.
/// Uses the closed-form inverse: Q(p; μ, β) = μ - β ln(-ln(p)).
///
/// # Mathematical Definition
///
/// The Gumbel quantile function is:
/// ```text
/// Q(p; μ, β) = μ - β ln(-ln(p))  for p ∈ (0, 1)
/// ```
///
/// This inverse transformation allows direct computation of extreme value quantiles.
///
/// # Parameters
///
/// * `p` - Probability values in (0, 1) for which quantiles are computed
/// * `location` - Location parameter μ (real number)
/// * `scale` - Scale parameter β > 0
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the quantile values, or a `KernelError`
/// if parameters are invalid.
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::gumbell::gumbel_quantile;
/// use minarrow::vec64;
///
/// let p = vec64![0.1, 0.5, 0.9];
/// let result = gumbel_quantile(&p, 0.0, 1.0, None, None).unwrap();
/// // Returns x values such that CDF(x) equals each probability
/// ```
#[inline(always)]
pub fn gumbel_quantile(
    p: &[f64],
    location: f64,
    scale: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gumbel_quantile_simd(p, location, scale, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gumbel_quantile_std(p, location, scale, null_mask, null_count)
    }
}

/// Zero-allocation variant of [`gumbel_pdf`].
///
/// Writes directly to caller-provided output buffer.
/// f(x; μ, β) = (1/β) exp(-z - exp(-z)) where z = (x - μ)/β
#[inline(always)]
pub fn gumbel_pdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gumbel_pdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gumbel_pdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Zero-allocation variant of [`gumbel_cdf`].
///
/// Writes directly to caller-provided output buffer.
/// F(x; μ, β) = exp(-exp(-z)) where z = (x - μ)/β
#[inline(always)]
pub fn gumbel_cdf_to(
    x: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gumbel_cdf_simd_to(x, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gumbel_cdf_std_to(x, location, scale, output, null_mask, null_count)
    }
}

/// Zero-allocation variant of [`gumbel_quantile`].
///
/// Writes directly to caller-provided output buffer.
/// Q(p; μ, β) = μ - β ln(-ln(p))
#[inline(always)]
pub fn gumbel_quantile_to(
    p: &[f64],
    location: f64,
    scale: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::gumbel_quantile_simd_to(p, location, scale, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::gumbel_quantile_std_to(p, location, scale, output, null_mask, null_count)
    }
}

#[cfg(test)]
mod gumbel_tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // See `./tests` for the scipy test suite

    // ---------- helper utilities -----------------------------------------

    fn mask_vec(m: &Bitmask) -> Vec<bool> {
        (0..m.len()).map(|i| m.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // Reference cases used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"

    // Standard Gumbel (μ = 0, β = 1)
    fn ref_pdf(x: f64) -> f64 {
        (-(x + (-x).exp())).exp()
    }
    fn ref_cdf(x: f64) -> f64 {
        (-(-x).exp()).exp()
    }
    fn ref_quantile(p: f64) -> f64 {
        -(-p.ln()).ln()
    }

    // gumbel_pdf  - correctness & scalar-bulk consistency

    // TODO: Double check SIMD vec64 align
    #[test]
    fn gumbel_pdf_reference_values() {
        // x = [-2, -1, 0, 1, 2]  (standard μ=0, β=1)
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let expect = x.map(ref_pdf);
        let out = dense_data(gumbel_pdf(&x, 0.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 3e-15);
        }
    }

    #[test]
    fn gumbel_pdf_scalar_vs_bulk() {
        let xs = vec64![-5.0, -0.3, 0.7, 5.0];
        let bulk = dense_data(gumbel_pdf(&xs, 1.5, 2.2, None, None).unwrap());
        for (i, &xi) in xs.iter().enumerate() {
            let sc = dense_data(gumbel_pdf(&[xi], 1.5, 2.2, None, None).unwrap())[0];
            assert_close(bulk[i], sc, 1e-15);
        }
    }

    // gumbel_cdf  - correctness & tail behaviour

    // TODO: Double check SIMD vec64 align
    #[test]
    fn gumbel_cdf_reference_values() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let expect = x.map(ref_cdf);
        let out = dense_data(gumbel_cdf(&x, 0.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-12);
        }
    }

    #[test]
    fn gumbel_cdf_tail_extremes() {
        let x = vec64![-1.0e6, 1.0e6];
        let out = dense_data(gumbel_cdf(&x, 0.0, 1.0, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15);
        assert_close(out[1], 1.0, 1e-15);
    }

    // gumbel_quantile  - correctness & round-trip

    // TODO: Double check SIMD vec64 align
    #[test]
    fn gumbel_quantile_reference_values() {
        let p = [0.01, 0.5, 0.9];
        let expect = p.map(ref_quantile);
        let out = dense_data(gumbel_quantile(&p, 0.0, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-12);
        }
    }

    #[test]
    fn gumbel_quantile_roundtrip() {
        let x = vec64![-3.0, -0.5, 0.2, 2.0, 6.0];
        let cdf = dense_data(gumbel_cdf(&x, 0.0, 1.0, None, None).unwrap());
        let x2 = dense_data(gumbel_quantile(&cdf, 0.0, 1.0, None, None).unwrap());
        for (orig, back) in x.iter().zip(x2.iter()) {
            assert_close(*orig, *back, 3e-12);
        }
    }

    #[test]
    fn gumbel_quantile_extremes() {
        let p = vec64![0.0, 1.0];
        let out = dense_data(gumbel_quantile(&p, 0.0, 1.0, None, None).unwrap());
        assert!(out[0].is_infinite() && out[0].is_sign_negative());
        assert!(out[1].is_infinite() && out[1].is_sign_positive());
    }

    // mask propagation

    #[test]
    fn gumbel_pdf_mask() {
        let x = vec64![-1.0, 0.0, 1.0];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(1, false);
        }
        let arr = gumbel_pdf(&x, 0.0, 1.0, Some(&m), Some(1)).unwrap();
        let mv = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(mv, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    #[test]
    fn gumbel_cdf_mask() {
        let x = vec64![-10.0, 0.0, 10.0];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(0, false);
        }
        let arr = gumbel_cdf(&x, 0.0, 1.0, Some(&m), Some(1)).unwrap();
        let mv = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(mv, vec![false, true, true]);
        assert!(arr.data[0].is_nan());
    }

    #[test]
    fn gumbel_quantile_mask() {
        let p = vec64![0.2, 0.8, 0.9];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(2, false);
        }
        let arr = gumbel_quantile(&p, 0.0, 1.0, Some(&m), Some(1)).unwrap();
        let mv = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(mv, vec![true, true, false]);
        assert!(arr.data[2].is_nan());
    }

    // parameter validation

    #[test]
    fn gumbel_invalid_params_error() {
        assert!(gumbel_pdf(&[0.0], 0.0, 0.0, None, None).is_err());
        assert!(gumbel_cdf(&[0.0], f64::NAN, 1.0, None, None).is_err());
        assert!(gumbel_quantile(&[0.5], 0.0, f64::NEG_INFINITY, None, None).is_err());
    }

    // empty inputs

    #[test]
    fn gumbel_empty_fast_paths() {
        let pdf = gumbel_pdf(&[], 0.0, 1.0, None, None).unwrap();
        let cdf = gumbel_cdf(&[], 0.0, 1.0, None, None).unwrap();
        let qtl = gumbel_quantile(&[], 0.0, 1.0, None, None).unwrap();
        assert!(pdf.data.is_empty() && cdf.data.is_empty() && qtl.data.is_empty());
    }
}
