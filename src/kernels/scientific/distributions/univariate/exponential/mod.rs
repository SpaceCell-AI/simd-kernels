// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::Bitmask;
use minarrow::FloatArray;
use minarrow::enums::error::KernelError;

/// Exponential PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_pdf_to(
    x: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::exponential_pdf_simd_to(x, lambda, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::exponential_pdf_std_to(x, lambda, output, null_mask, null_count)
    }
}

/// Exponential PDF: f(x|λ) = λ·exp(-λ·x) for x ≥ 0, 0 otherwise.
/// Returns error if λ ≤ 0 or non-finite.
#[inline(always)]
pub fn exponential_pdf(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::exponential_pdf_simd(x, lambda, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::exponential_pdf_std(x, lambda, null_mask, null_count)
    }
}

/// Exponential CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_cdf_to(
    x: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::exponential_cdf_simd_to(x, lambda, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::exponential_cdf_std_to(x, lambda, output, null_mask, null_count)
    }
}

/// Exponential CDF: F(x|λ) = 1 - exp(-λ·x) for x ≥ 0, 0 otherwise.
/// Error if λ ≤ 0 or non‐finite.
#[inline(always)]
pub fn exponential_cdf(
    x: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::exponential_cdf_simd(x, lambda, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::exponential_cdf_std(x, lambda, null_mask, null_count)
    }
}

/// Exponential quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn exponential_quantile_to(
    p: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::exponential_quantile_simd_to(p, lambda, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::exponential_quantile_std_to(p, lambda, output, null_mask, null_count)
    }
}

/// Exponential quantile (inverse CDF): Q(p|λ) = -ln(1-p)/λ, pure math
#[inline(always)]
pub fn exponential_quantile(
    p: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::exponential_quantile_simd(p, lambda, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::exponential_quantile_std(p, lambda, null_mask, null_count)
    }
}

#[cfg(test)]
mod exponential_tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // See `./tests` for the scipy test suite

    // helpers

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: {a} vs {b} (tol {tol})"
        );
    }

    // exponential_pdf  - correctness

    #[test]
    fn exp_pdf_basic_values_lambda1() {
        // f(x) = e^{-x}  for λ = 1
        // Verified against scipy.stats.expon.pdf(x, scale=1.0) on 2025-08-14
        // Full precision: np.set_printoptions(precision=17, floatmode='fixed')
        let x = vec64![0.0, 0.5, 1.0, 2.0];
        let expect = vec64![
            1.00000000000000000,
            0.60653065971263342,
            0.36787944117144233,
            0.13533528323661270
        ];
        let out = dense_data(exponential_pdf(&x, 1.0, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-17);
        }
    }

    #[test]
    fn exp_pdf_basic_values_lambda2() {
        // f(x) = 2 e^{-2x}
        // Verified against scipy.stats.expon.pdf(x, scale=0.5) on 2025-08-14
        // Full precision: np.set_printoptions(precision=17, floatmode='fixed')
        let λ = 2.0;
        let x = vec64![0.0, 0.25, 1.0];
        let expect = vec64![
            2.00000000000000000,
            1.21306131942526685,
            0.27067056647322540
        ];
        let out = dense_data(exponential_pdf(&x, λ, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-17);
        }
    }

    #[test]
    fn exp_pdf_negative_x_zero() {
        let x = vec64![-5.0, -0.1];
        let out = dense_data(exponential_pdf(&x, 1.3, None, None).unwrap());
        assert_eq!(out, vec64![0.0, 0.0]);
    }

    #[test]
    fn exp_pdf_bulk_vs_scalar_consistency() {
        let λ = 0.7;
        let x = vec64![0.0, 0.2, 0.5, 1.0, 4.2];
        let bulk = dense_data(exponential_pdf(&x, λ, None, None).unwrap());
        for (i, &xi) in x.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let scalar =
                dense_data(exponential_pdf(xi_aligned.as_slice(), λ, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    // exponential_cdf  - correctness

    #[test]
    fn exp_cdf_basic_values() {
        // Verified against scipy.stats.expon.cdf(x, scale=1/3) on 2025-08-14
        // Full precision: np.set_printoptions(precision=17, floatmode='fixed')
        let λ = 3.0;
        let x = vec64![0.0, 0.1, 0.5, 2.0];
        let expect = vec64![
            0.00000000000000000,
            0.25918177931828218,
            0.77686983985157021,
            0.99752124782333362
        ];
        let out = dense_data(exponential_cdf(&x, λ, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-16);
        }
    }

    #[test]
    fn exp_cdf_negative_x_zero() {
        let out = dense_data(exponential_cdf(&[-3.0, -1e-9], 2.0, None, None).unwrap());
        assert_eq!(out, vec64![0.0, 0.0]);
    }

    #[test]
    fn exp_cdf_tail_extremes() {
        let out = dense_data(exponential_cdf(&[1e308], 1.0, None, None).unwrap());
        assert_close(out[0], 1.0, 1e-15);
    }

    // exponential_quantile  - correctness & round-trip

    #[test]
    fn exp_quantile_basic_values() {
        // Verified against scipy.stats.expon.ppf(p, scale=1/0.8) on 2025-08-14
        // Full precision: np.set_printoptions(precision=17, floatmode='fixed')
        let λ = 0.8;
        let p = vec64![0.01, 0.25, 0.5, 0.9];
        let expect = vec64![
            0.01256291981687680,
            0.35960259056472610,
            0.86643397569993164,
            2.87823136624255760
        ];
        let out = dense_data(exponential_quantile(&p, λ, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-15);
        }
    }

    #[test]
    fn exp_quantile_bounds() {
        let arr = dense_data(exponential_quantile(&[0.0, 1.0], 5.0, None, None).unwrap());
        assert_close(arr[0], 0.0, 0.0);
        assert!(arr[1].is_infinite() && arr[1].is_sign_positive());
    }

    #[test]
    fn exp_quantile_roundtrip() {
        let λ = 1.7;
        let p = vec64![0.001, 0.1, 0.5, 0.8, 0.999];
        let q = dense_data(exponential_quantile(&p, λ, None, None).unwrap());
        let c = dense_data(exponential_cdf(&q, λ, None, None).unwrap());
        for (pi, ci) in p.iter().zip(c.iter()) {
            assert_close(*pi, *ci, 1e-12);
        }
    }

    // mask & null propagation

    #[test]
    fn exp_pdf_null_mask() {
        // Null handling: null inputs should produce NaN outputs
        // Verified against scipy expectation on 2025-08-14:
        // scipy.stats.expon.pdf([0.0, 1.0, 2.0], scale=1.0) with full precision
        // = [1.00000000000000000, 0.36787944117144233, 0.13533528323661270]
        // with null at index 1 -> [1.00000000000000000, NaN, 0.13533528323661270]
        let x = vec64![0.0, 1.0, 2.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(1, false);
        } // null the middle
        let arr = exponential_pdf(&x, 1.0, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
        // Verify non-null values match scipy
        assert_close(arr.data[0], 1.00000000000000000, 1e-17);
        assert_close(arr.data[2], 0.13533528323661270, 1e-17);
    }

    #[test]
    fn exp_cdf_null_mask() {
        // Null handling: null inputs should produce NaN outputs
        // Verified against scipy expectation on 2025-08-14:
        // scipy.stats.expon.cdf([-1.0, 0.5, 1.0], scale=1.0/0.9) with full precision
        // = [0.00000000000000000, 0.36237184837822667, 0.59343034025940078]
        // with null at index 0 -> [NaN, 0.36237184837822667, 0.59343034025940078]
        let x = vec64![-1.0, 0.5, 1.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(0, false);
        }
        let arr = exponential_cdf(&x, 0.9, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![false, true, true]);
        assert!(arr.data[0].is_nan());
        // Verify non-null values match scipy
        assert_close(arr.data[1], 0.36237184837822667, 1e-16);
        assert_close(arr.data[2], 0.59343034025940078, 1e-15);
    }

    #[test]
    fn exp_quantile_null_mask() {
        let p = vec64![0.2, 0.4, 0.6];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe {
            mask.set_unchecked(2, false);
        }
        let arr = exponential_quantile(&p, 2.3, Some(&mask), Some(1)).unwrap();
        let m = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(m, vec![true, true, false]);
        assert!(arr.data[2].is_nan());
    }

    // parameter validation errors

    #[test]
    fn exp_invalid_lambda() {
        assert!(exponential_pdf(&[0.0], 0.0, None, None).is_err());
        assert!(exponential_cdf(&[0.0], -1.0, None, None).is_err());
        assert!(exponential_quantile(&[0.5], f64::NAN, None, None).is_err());
    }

    // empty-input behaviour

    #[test]
    fn exp_empty_inputs() {
        let pdf = exponential_pdf(&[], 1.0, None, None).unwrap();
        let cdf = exponential_cdf(&[], 1.0, None, None).unwrap();
        let qtl = exponential_quantile(&[], 1.0, None, None).unwrap();
        assert!(pdf.data.is_empty() && cdf.data.is_empty() && qtl.data.is_empty());
    }
}
