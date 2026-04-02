// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Uniform Distribution Module** - *Constant Density, Maximum Entropy*
//!
//! High-performance implementation of the continuous uniform distribution, providing the foundation
//! for random number generation, Monte Carlo methods, and maximum entropy probability modelling.
//!
//! ## Statistical Applications
//! The uniform distribution serves as the foundation for numerous statistical applications:
//! - **Random number generation**: basis for inverse transform sampling
//! - **Monte Carlo methods**: fundamental building block for simulation techniques
//! - **Maximum entropy principle**: represents complete ignorance within bounded support
//! - **Hypothesis testing**: null distribution for various non-parametric tests
//! - **Calibration**: reference distribution for probability integral transforms
//!
//! ## Usage Examples
//! ```rust,ignore
//! use minarrow::{vec64, FloatArray};
//! use simd_kernels::kernels::scientific::distributions::univariate::uniform::*;
//!
//! // Standard uniform distribution on [0, 1]
//! let x = vec64![0.0, 0.25, 0.5, 0.75, 1.0];
//! let pdf = uniform_pdf(&x, 0.0, 1.0, None, None).unwrap();
//! let cdf = uniform_cdf(&x, 0.0, 1.0, None, None).unwrap();
//!
//! // Custom interval [-5, 10]
//! let samples = vec64![-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0];
//! let probabilities = uniform_cdf(&samples, -5.0, 10.0, None, None).unwrap();
//!
//! // Quantile function for inverse sampling
//! let p_values = vec64![0.1, 0.25, 0.5, 0.75, 0.9];
//! let quantiles = uniform_quantile(&p_values, 0.0, 100.0, None, None).unwrap();
//! ```
#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray};

/// Uniform PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_pdf_to(
    x: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::uniform_pdf_simd_to(x, a, b, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::uniform_pdf_std_to(x, a, b, output, null_mask, null_count)
    }
}

/// Uniform PDF: f(x|a,b) = 1/(b-a) for x in [a, b], 0 otherwise.
#[inline(always)]
pub fn uniform_pdf(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::uniform_pdf_simd(x, a, b, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::uniform_pdf_std(x, a, b, null_mask, null_count)
    }
}

/// Uniform CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_cdf_to(
    x: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::uniform_cdf_simd_to(x, a, b, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::uniform_cdf_std_to(x, a, b, output, null_mask, null_count)
    }
}

/// Uniform CDF: F(x|a,b) = 0 if x < a, (x-a)/(b-a) if x in [a,b], 1 if x > b.
#[inline(always)]
pub fn uniform_cdf(
    x: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::uniform_cdf_simd(x, a, b, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::uniform_cdf_std(x, a, b, null_mask, null_count)
    }
}

/// Uniform quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn uniform_quantile_to(
    p: &[f64],
    a: f64,
    b: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::uniform_quantile_simd_to(p, a, b, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::uniform_quantile_std_to(p, a, b, output, null_mask, null_count)
    }
}

/// Continuous-uniform quantile
#[inline(always)]
pub fn uniform_quantile(
    p: &[f64],
    a: f64,
    b: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::uniform_quantile_simd(p, a, b, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::uniform_quantile_std(p, a, b, null_mask, null_count)
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // see "./tests" for scipy test suite

    // Helper utilities

    fn mask_vec(m: &Bitmask) -> Vec<bool> {
        (0..m.len()).map(|i| m.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // Test parameters used repeatedly
    const A: f64 = -1.0;
    const B: f64 = 2.0; // width = 3.0, inv_width = 1/3

    // PDF correctness & behaviour
    #[test]
    fn pdf_reference_values() {
        // Verified against scipy.stats.uniform.pdf(x, loc=-1.0, scale=3.0) on 2025-08-14
        // Full precision - np.set_printoptions(precision=17, floatmode='fixed')
        let x = vec64![-2.0, -1.0, 0.0, 2.0, 3.0];
        let out = dense_data(uniform_pdf(&x, A, B, None, None).unwrap());
        let expect = vec64![
            0.00000000000000000,
            0.33333333333333331,
            0.33333333333333331,
            0.33333333333333331,
            0.00000000000000000
        ];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-16);
        }
    }

    #[test]
    fn pdf_bulk_vs_scalar() {
        let x = vec64![-1.5, -1.0, -0.2, 1.7, 2.5];
        let bulk = dense_data(uniform_pdf(&x, A, B, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let sc = dense_data(uniform_pdf(xi_aligned.as_slice(), A, B, None, None).unwrap())[0];
            assert_close(bulk[i], sc, 1e-15);
        }
    }

    // CDF correctness & symmetry
    #[test]
    fn cdf_reference_values() {
        let x = vec64![-2.0, -1.0, 0.0, 1.5, 2.0, 4.0];
        let out = dense_data(uniform_cdf(&x, A, B, None, None).unwrap());
        let expect = vec64![
            0.0,
            0.0,
            (0.0 - A) / (B - A), // 0.333333…
            (1.5 - A) / (B - A), // 0.833333…
            1.0,
            1.0,
        ];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    // Quantile correctness & round-trip
    #[test]
    fn quantile_reference_values() {
        let p = vec64![0.0, 0.25, 0.5, 1.0];
        let out = dense_data(uniform_quantile(&p, A, B, None, None).unwrap());
        let expect = vec64![A, A + 0.75, A + 1.5, B];
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn quantile_cdf_roundtrip() {
        // Points strictly inside the interval to avoid double-endpoint ambiguity
        let x = vec64![-0.8, -0.1, 0.4, 1.3];
        let c = dense_data(uniform_cdf(&x, A, B, None, None).unwrap());
        let qx = dense_data(uniform_quantile(&c, A, B, None, None).unwrap());
        for (xi, qi) in x.iter().zip(qx.iter()) {
            assert_close(*xi, *qi, 1e-14);
        }
    }

    // Mask propagation tests
    #[test]
    fn pdf_mask_propagation() {
        // Null handling: null inputs should produce NaN outputs
        // Verified against scipy expectation on 2025-08-14:
        // scipy.stats.uniform.pdf([-2.0, -1.0, 2.0, 3.0], loc=-1.0, scale=3.0)
        // = [0.00000000000000000, 0.33333333333333331, 0.33333333333333331, 0.00000000000000000]
        // with null at index 2 -> [0.00000000000000000, 0.33333333333333331, NaN, 0.00000000000000000]
        let x = vec64![A - 1.0, A, B, B + 1.0];
        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(2, false) }; // mask‐out index 2
        let arr = uniform_pdf(&x, A, B, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());

        // Mask replicated, NaN sentinel
        assert_eq!(m, vec![true, true, false, true]);
        assert!(arr.data[2].is_nan());
        // Verify non-null values match scipy
        assert_close(arr.data[0], 0.00000000000000000, 1e-16);
        assert_close(arr.data[1], 0.33333333333333331, 1e-16);
        assert_close(arr.data[3], 0.00000000000000000, 1e-16);
    }

    #[test]
    fn quantile_mask_and_domain_handling() {
        // include out-of-range probs and an null
        let p = vec64![-0.1, 0.0, 0.3, 1.0, 1.2];
        let mut mask = Bitmask::new_set_all(5, true);
        unsafe { mask.set_unchecked(1, false) }; // null index 1

        let arr = uniform_quantile(&p, A, B, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());

        // Index-by-index expectations
        assert!(arr.data[0].is_nan() && m[0]); // out-of-range prob
        assert!(arr.data[1].is_nan() && !m[1]); // masked-out input
        assert_close(arr.data[2], A + 0.9, 1e-15); // 0.3 inside
        assert_close(arr.data[3], B, 1e-15); // p == 1
        assert!(arr.data[4].is_nan() && m[4]); // >1 -> NaN
    }

    // Error handling & empty input
    #[test]
    fn invalid_parameter_errors() {
        assert!(uniform_pdf(&[0.0], 1.0, 1.0, None, None).is_err()); // a==b
        assert!(uniform_cdf(&[0.0], 2.0, 1.0, None, None).is_err()); // a>b
        assert!(uniform_quantile(&[0.5], f64::NAN, 2.0, None, None).is_err());
    }

    #[test]
    fn empty_input() {
        let arr = uniform_cdf(&[], A, B, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }
}
