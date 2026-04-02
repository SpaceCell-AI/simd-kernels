// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Student's t-Distribution Module** - *Heavy-Tailed Statistical Foundation*
//!
//! High-performance implementation of Student's t-distribution, essential for small-sample
//! statistical inference and robust statistical modelling. This implementation provides the
//! computational foundation for t-tests, confidence intervals, and regression analysis.
//!
//! ## Use cases
//! The Student's t-distribution is fundamental to inferential statistics, particularly for:
//! - **Small sample inference**: t-tests when population variance is unknown
//! - **Confidence intervals**: robust interval estimation with heavy tails
//! - **Regression analysis**: coefficient testing and parameter uncertainty
//! - **Robustness**: alternative to normal distribution with heavier tails
//! - **Degrees of freedom modelling**: captures uncertainty reduction with sample size
//!
//! ## Usage Examples
//! ```rust,ignore
//! use minarrow::{vec64, FloatArray};
//! use simd_kernels::kernels::scientific::distributions::univariate::student_t::*;
//!
//! // t-test critical values (two-tailed, α=0.05)
//! let df_values = vec![1.0, 5.0, 10.0, 30.0]; // degrees of freedom
//! let alpha = 0.025;  // two-tailed significance level
//!
//! for df in df_values {
//!     let critical_value = student_t_quantile(&[1.0 - alpha], df, None, None).unwrap();
//!     println!("df={}: t_crit = {:.4}", df, critical_value.data[0]);
//! }
//!
//! // Heavy-tail comparison with normal distribution
//! let x = vec64![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
//! let t_pdf = student_t_pdf(&x, 3.0, None, None).unwrap();  // df=3
//! ```
//!
#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray};

/// Student-t PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn student_t_pdf_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::student_t_pdf_simd_to(x, df, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::student_t_pdf_std_to(x, df, output, null_mask, null_count)
    }
}

/// Student‐t PDF
#[inline(always)]
pub fn student_t_pdf(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::student_t_pdf_simd(x, df, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::student_t_pdf_std(x, df, null_mask, null_count)
    }
}

/// Student-t CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn student_t_cdf_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::student_t_cdf_std_to(x, df, output, null_mask, null_count)
}

/// Student-t CDF: see
/// - Algorithm AS 3 (Hill, 1970) / Algorithm 395 (G. W. Hill, 1970)
/// - https://www.jstor.org/stable/2346841
///
/// Scalar implementation uses the incomplete beta via continued fraction.
/// Returns error if df < 1.
#[inline(always)]
pub fn student_t_cdf(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::student_t_cdf_std(x, df, null_mask, null_count)
}

/// Student-t quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn student_t_quantile_to(
    p: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::student_t_quantile_std_to(p, df, output, null_mask, null_count)
}

/// Student-t quantile (inverse CDF), using the AS 241 algorithm
/// Reference: Michael J. Wichura, Algorithm AS 241: The Percentage Points of the Normal Distribution
#[inline(always)]
pub fn student_t_quantile(
    p: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::student_t_quantile_std(p, df, null_mask, null_count)
}

#[cfg(test)]
mod tests {

    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, Vec64, vec64};

    // see "./tests" for scipy test suite

    // Helpers

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // PDF - reference values & properties
    #[test]
    fn pdf_reference_values_df1() {
        // df = 1  (Cauchy)  SciPy: stats.t.pdf([0,1,2],1) == [0.318309886, 0.159154943, 0.063661977]
        let x = vec64![0.0, 1.0, 2.0];
        let expect = vec64![0.3183098861837907, 0.15915494309189535, 0.06366197723675814];
        let out = dense_data(student_t_pdf(&x, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn pdf_even_function() {
        let df = 7.0;
        let x = vec64![-3.2, -1.1, 0.5, 2.4];
        let mut xp = Vec64::with_capacity(x.len());
        for &v in &x {
            xp.push(-v);
        }
        let f_neg = dense_data(student_t_pdf(&x, df, None, None).unwrap());
        let f_pos = dense_data(student_t_pdf(&xp, df, None, None).unwrap());
        for (a, b) in f_neg.iter().zip(f_pos.iter()) {
            assert_close(*a, *b, 1e-15);
        }
    }

    #[test]
    fn pdf_bulk_vs_scalar() {
        let df = 3.5;
        let x = vec64![-2.5, -0.8, 0.0, 1.3, 4.7];
        let bulk = dense_data(student_t_pdf(&x, df, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let sc = dense_data(student_t_pdf(xi_aligned.as_slice(), df, None, None).unwrap())[0];
            assert_close(bulk[i], sc, 1e-15);
        }
    }

    // CDF - symmetry & limits
    #[test]
    fn cdf_symmetry() {
        let df = 5.0;
        let x = vec64![-4.0, -1.0, -0.3, 0.3, 1.0, 4.0];
        let cdf = dense_data(student_t_cdf(&x, df, None, None).unwrap());
        for (xi, fi) in x.iter().zip(cdf.iter()) {
            let refl = dense_data(student_t_cdf(&[-xi], df, None, None).unwrap())[0];
            assert_close(*fi, 1.0 - refl, 1e-14);
        }
    }

    #[test]
    fn cdf_tail_extremes() {
        let df = 2.0;
        let x = vec64![-1e308, 1e308];
        let out = dense_data(student_t_cdf(&x, df, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15);
        assert_close(out[1], 1.0, 1e-15);
    }

    // Quantile - round-trip & special points
    #[test]
    fn quantile_end_points() {
        let df = 4.0;
        let probs = vec64![0.0, 1.0];
        let q = dense_data(student_t_quantile(&probs, df, None, None).unwrap());
        assert!(q[0].is_infinite() && q[0].is_sign_negative());
        assert!(q[1].is_infinite() && q[1].is_sign_positive());
    }

    #[test]
    fn quantile_cdf_roundtrip() {
        let df = 9.0;
        let x = vec64![-3.0, -0.7, 0.0, 0.9, 5.0];
        let p = dense_data(student_t_cdf(&x, df, None, None).unwrap());
        let qx = dense_data(student_t_quantile(&p, df, None, None).unwrap());
        for (xi, qi) in x.iter().zip(qx.iter()) {
            assert_close(*xi, *qi, 2e-13);
        }
    }

    // Mask propagation
    #[test]
    fn pdf_mask_propagation() {
        let x = vec64![-1.0, 0.0, 2.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(1, false) };
        let arr = student_t_pdf(&x, 3.0, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    // Error handling & empty
    #[test]
    fn invalid_df_errors() {
        assert!(student_t_pdf(&[0.0], 0.0, None, None).is_err());
        assert!(student_t_cdf(&[0.0], f64::NAN, None, None).is_err());
        assert!(student_t_quantile(&[0.5], -2.0, None, None).is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        let arr = student_t_pdf(&[], 5.0, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }
}
