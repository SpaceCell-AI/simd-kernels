// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Negative Binomial Distribution Module** - *Pascal Distribution, Discrete Failures*
//!
//! High-performance implementation of the negative binomial distribution (also known as the Pascal
//! distribution), representing the number of failures before achieving `r` successes in a sequence
//! of independent Bernoulli trials.
//!
//! ## Usage Examples
//! ```rust,ignore
//! use minarrow::{vec64, FloatArray};
//! use simd_kernels::kernels::scientific::distributions::univariate::neg_binomial::*;
//!
//! // 20 failures before 5 successes, with 30% success probability
//! let failures = vec64![0, 5, 10, 15, 20];
//! let r = 5;       // target successes
//! let p = 0.3;     // success probability
//!
//! // Compute probabilities
//! let pmf = neg_binomial_pmf(&failures, r, p, None, None).unwrap();
//! let cdf = neg_binomial_cdf(&failures, r, p, None, None).unwrap();
//!
//! // Find critical values
//! let quantiles = vec64![0.1, 0.5, 0.9];
//! let critical_points = neg_binomial_quantile(&quantiles, r, p, None, None).unwrap();
//! ```

#[cfg(feature = "simd")]
pub mod simd;
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Negative Binomial PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn neg_binomial_pmf_to(
    k: &[u64],
    r: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::neg_binomial_pmf_simd_to(k, r, p, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::neg_binomial_pmf_std_to(k, r, p, output, null_mask, null_count)
    }
}

/// Negative Binomial CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn neg_binomial_cdf_to(
    k: &[u64],
    r: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::neg_binomial_cdf_std_to(k, r, p, output, null_mask, null_count)
}

/// Negative Binomial quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn neg_binomial_quantile_to(
    q: &[f64],
    r: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::neg_binomial_quantile_std_to(q, r, p, output, null_mask, null_count)
}

/// Negative Binomial PMF (Pascal distribution, number of failures before r-th success)
/// PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k, for k=0,1,...
#[inline(always)]
pub fn neg_binomial_pmf(
    k: &[u64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::neg_binomial_pmf_simd(k, r, p, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::neg_binomial_pmf_std(k, r, p, null_mask, null_count)
    }
}

/// Negative Binomial CDF (sum of PMFs to k): F(X ≤ k) = I_p(r, k+1)
#[inline(always)]
pub fn neg_binomial_cdf(
    k: &[u64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::neg_binomial_cdf_std(k, r, p, null_mask, null_count)
}

/// Negative Binomial Quantile (inverse CDF): returns minimal k such that CDF(k) >= q
#[inline(always)]
pub fn neg_binomial_quantile(
    q: &[f64],
    r: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::neg_binomial_quantile_std(q, r, p, null_mask, null_count)
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::{
        shared::scalar::ln_choose, univariate::common::dense_data,
    };

    use super::*;
    use minarrow::{Bitmask, Vec64, vec64};

    //  Helpers

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // Reference cases used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"

    fn choose(n: u64, k: u64) -> f64 {
        ln_choose(n, k).exp()
    }
    fn nb_pmf_scalar(k: u64, r: u64, p: f64) -> f64 {
        choose(k + r - 1, k) * p.powi(r as i32) * (1.0 - p).powi(k as i32)
    }

    //  PMF - correctness
    #[test]
    fn pmf_reference_values() {
        let r = 3;
        let p = 0.25;
        let ks = vec64![0, 1, 2, 5, 10];
        let exp: Vec<f64> = ks.iter().map(|&k| nb_pmf_scalar(k, r, p)).collect();

        let arr = dense_data(neg_binomial_pmf(&ks, r, p, None, None).unwrap());
        for (a, e) in arr.iter().zip(exp.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn pmf_sums_to_one_truncated_tail() {
        // Sum pmf from k=0..60  (more than enough for convergence at p=0.3, r=4)
        let r = 4;
        let p = 0.3;
        let ks: Vec64<u64> = (0..61).collect();
        let arr = dense_data(neg_binomial_pmf(&ks, r, p, None, None).unwrap());

        let sum: f64 = arr.iter().sum();
        assert_close(sum, 1.0, 5e-6);
    }

    //  CDF - correctness & relationship with PMF
    #[test]
    fn cdf_matches_cumulative_pmf() {
        let r = 2;
        let p = 0.4;
        let ks: Vec<u64> = (0..20).collect();

        // cumulative manually
        let pmf = dense_data(neg_binomial_pmf(&ks, r, p, None, None).unwrap());
        let mut cumsum = Vec64::with_capacity(ks.len());
        let mut acc = 0.0;
        for &v in &pmf {
            acc += v;
            cumsum.push(acc);
        }

        // kernel cdf
        let cdf = dense_data(neg_binomial_cdf(&ks, r, p, None, None).unwrap());
        for (a, e) in cdf.iter().zip(cumsum.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    //  Quantile - basic values & round-trip
    #[test]
    fn quantile_basic_cases() {
        let r = 5;
        let p = 0.2;
        // hand-picked probabilities
        let q = vec64![0.0, 0.1, 0.5, 0.9, 1.0];
        let arr = dense_data(neg_binomial_quantile(&q, r, p, None, None).unwrap());

        // 0 ➜ -1 ; 1 ➜ +inf ; others integer ≥0
        assert_eq!(arr[0], -1.0);
        assert!(arr[4].is_infinite());

        for &v in &arr[1..4] {
            assert!(v >= 0.0 && v.fract() == 0.0); // integer, non-negative
        }
    }

    #[test]
    fn quantile_cdf_roundtrip() {
        let r = 3;
        let p = 0.35;
        let ks: Vec<u64> = (0..15).collect();
        let cdf = dense_data(neg_binomial_cdf(&ks, r, p, None, None).unwrap());
        let quant = dense_data(neg_binomial_quantile(&cdf, r, p, None, None).unwrap());

        for (k, qk) in ks.iter().zip(quant.iter()) {
            assert_eq!(*k as f64, *qk); // perfect integer round-trip
        }
    }

    //  Mask propagation
    #[test]
    fn pmf_mask_propagation() {
        let ks = vec64![0, 1, 2, 3];
        let r = 2;
        let p = 0.5;

        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(1, false) }; // mask out index 1

        let arr = neg_binomial_pmf(&ks, r, p, Some(&mask), Some(1)).unwrap();
        let mvec = mask_vec(arr.null_mask.as_ref().unwrap());

        assert_eq!(mvec, vec![true, false, true, true]);
        assert!(arr.data[1].is_nan());
    }

    //  Error handling
    #[test]
    fn pmf_invalid_parameters() {
        let r = 0;
        let ks = vec64![1, 2, 3];
        let badp = 1.2;
        assert!(neg_binomial_pmf(&ks, r, 0.3, None, None).is_err());
        assert!(neg_binomial_pmf(&ks, 2, badp, None, None).is_err());
    }

    #[test]
    fn cdf_invalid_parameters() {
        assert!(neg_binomial_cdf(&[1], 0, 0.5, None, None).is_err());
        assert!(neg_binomial_cdf(&[1], 3, -0.1, None, None).is_err());
    }

    #[test]
    fn quantile_invalid_parameters() {
        assert!(neg_binomial_quantile(&[0.5], 0, 0.4, None, None).is_err());
        assert!(neg_binomial_quantile(&[0.5], 4, 1.1, None, None).is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        let arr = neg_binomial_pmf(&[], 2, 0.3, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }
}
