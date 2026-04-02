// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Poisson Distribution Module** - *Discrete Events, Counting Processes*
//!
//! High-performance implementation of the Poisson distribution, modelling the number of independent
//! events occurring within a fixed time interval or spatial region. This implementation provides
//! numerically stable computation with optimised performance for large-scale statistical applications.
//!
//! Due to the nature of the distribution, only the pmf case is SIMD-accelerated.
//!
//! ## Overview
//! - **Domain**: `k ∈ {0, 1, 2, ...}` (discrete, non-negative integers)
//! - **Parameter**: `λ > 0` (rate parameter, average number of events)
//! - **PMF**: `P(X = k) = e^(-λ) × λ^k / k!`
//! - **Mean**: `E[X] = λ`
//! - **Variance**: `Var[X] = λ`
//!
//! ## Use cases
//! The Poisson distribution is fundamental in modelling discrete counting processes where events
//! occur independently at a constant average rate:
//! - **Telecommunications**: packet arrivals in network queues
//! - **Epidemiology**: disease outbreak modelling and infection counts
//! - **Manufacturing**: defect rates and quality control systems
//! - **Astronomy**: photon detection and stellar event counting
//! - **Finance**: rare events like market crashes or high-frequency trading
//! - **Biology**: mutation rates and cell division processes
//!
//! ## Numerical Accuracy
//!
//! ### Validation Approach
//! All implementations are validated against SciPy's `scipy.stats.poisson` using:
//! - **Reference values**: hand-calculated cases for small λ and k
//! - **Distributional properties**: moment validation and tail behaviour verification
//! - **Cross-validation**: PMF/CDF consistency and quantile round-trip testing
//! - **Parameter robustness**: extensive testing across wide λ ranges
//!
//! See `./tests` for coverage, and confirm the results on your target platform if you
//! have specific requirements. We make no guarantees.
//!
//! ## Usage Examples
//! ```rust,ignore
//! use minarrow::{vec64, FloatArray};
//! use simd_kernels::kernels::scientific::distributions::univariate::poisson::*;
//!
//! // Call center receiving average 4.5 calls per minute
//! let call_counts = vec64![0, 2, 4, 6, 8, 10];
//! let lambda = 4.5;  // average rate
//!
//! // Compute probabilities
//! let pmf = poisson_pmf(&call_counts, lambda, None, None).unwrap();
//! let cdf = poisson_cdf(&call_counts, lambda, None, None).unwrap();
//!
//! // Find capacity requirements (95th percentile)
//! let percentiles = vec64![0.90, 0.95, 0.99];
//! let capacity_levels = poisson_quantile(&percentiles, lambda, None, None).unwrap();
//! ```
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray};

/// Poisson PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn poisson_pmf_to(
    k: &[u64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::poisson_pmf_simd_to(k, lambda, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::poisson_pmf_std_to(k, lambda, output, null_mask, null_count)
    }
}

/// Poisson PMF: P(K=k|λ) = e^{-λ} · λ^k / k!
/// k: observed event counts (all ≥ 0)
/// λ: event rate (λ > 0, finite)
#[inline(always)]
pub fn poisson_pmf(
    k: &[u64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::poisson_pmf_simd(k, lambda, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::poisson_pmf_std(k, lambda, null_mask, null_count)
    }
}

/// Poisson CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn poisson_cdf_to(
    k: &[u64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::poisson_cdf_std_to(k, lambda, output, null_mask, null_count)
}

/// Poisson CDF: F(K=k|λ) = ∑_{i=0}^k Poisson_PMF(i, λ)
/// Efficient and robust using the lower regularised incomplete gamma:
/// F(K=k|λ) = γ(⌊k+1⌋, λ) / Γ(⌊k+1⌋)
#[inline(always)]
pub fn poisson_cdf(
    k: &[u64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::poisson_cdf_std(k, lambda, null_mask, null_count)
}

/// Poisson quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn poisson_quantile_to(
    p: &[f64],
    lambda: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::poisson_quantile_std_to(p, lambda, output, null_mask, null_count)
}

/// Poisson quantile function (inverse CDF).
///
/// For probability `p` ∈ (0,1), returns the smallest integer `k` such that
///     Pr[X ≤ k] ≥ p, where X ~ Poisson(λ).
#[inline(always)]
pub fn poisson_quantile(
    p: &[f64],
    lambda: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::poisson_quantile_std(p, lambda, null_mask, null_count)
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::{
        shared::scalar::ln_gamma_plus1, univariate::common::dense_data,
    };

    // see "./tests" for scipy test suite

    use super::*;
    use minarrow::{Bitmask, Vec64, vec64};

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

    // Reference case used during early testing without the library / SIMD code
    // Later scipy test suite was implemented under "./tests"
    fn scalar_pmf(k: u64, lambda: f64) -> f64 {
        let kf = k as f64;
        ((-lambda) + kf * lambda.ln() - ln_gamma_plus1(kf)).exp()
    }

    // PMF - numerical checks
    #[test]
    fn pmf_reference_values() {
        let lambda = 3.5;
        let ks = vec64![0, 1, 2, 3, 5, 10];
        let expected: Vec<f64> = ks.iter().copied().map(|k| scalar_pmf(k, lambda)).collect();

        let arr = dense_data(poisson_pmf(&ks, lambda, None, None).unwrap());
        for (a, e) in arr.iter().zip(expected.iter()) {
            assert_close(*a, *e, 1e-15);
        }
    }

    #[test]
    fn pmf_sums_to_one_reasonably() {
        let lambda = 12.0;
        let ks: Vec64<u64> = (0..200).collect();
        let arr = dense_data(poisson_pmf(&ks, lambda, None, None).unwrap());
        let sum: f64 = arr.iter().sum();
        assert_close(sum, 1.0, 1e-10);
    }

    // CDF - comparison with cumulative PMF
    #[test]
    fn cdf_matches_manual_cumulative() {
        let lambda = 4.2;
        let ks: Vec64<u64> = (0..25).collect();

        let mut pmf = dense_data(poisson_pmf(&ks, lambda, None, None).unwrap());
        let mut cumsum = Vec64::with_capacity(ks.len());
        let mut acc = 0.0;
        for v in &mut pmf {
            acc += *v;
            cumsum.push(acc);
        }

        let cdf = dense_data(poisson_cdf(&ks, lambda, None, None).unwrap());
        for (a, e) in cdf.iter().zip(cumsum.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    // Quantile - sanity & round-trip
    #[test]
    fn quantile_basic_cases() {
        let lambda = 5.0;
        let p = vec64![0.0, 0.25, 0.5, 0.9, 1.0];
        let arr = dense_data(poisson_quantile(&p, lambda, None, None).unwrap());

        assert_eq!(arr[0], -1.0);
        assert!(arr[4].is_infinite());
        assert!(arr[1].fract() == 0.0 && arr[2].fract() == 0.0 && arr[3].fract() == 0.0);
    }

    #[test]
    fn quantile_cdf_roundtrip() {
        let lambda = 7.0;
        let ks: Vec64<u64> = (0..25).collect(); // Reduce range to avoid extreme boundary cases
        let mut cdf = dense_data(poisson_cdf(&ks, lambda, None, None).unwrap());

        // Reduce each cdf slightly to stay strictly below the boundary
        // Use a smaller reduction factor for better precision at high k values
        for p in &mut cdf {
            *p = (*p * 0.9999999999999).max(0.0);
        }

        let qs = dense_data(poisson_quantile(&cdf, lambda, None, None).unwrap());
        for (k, q) in ks.iter().zip(qs.iter()) {
            let diff = (*k as f64 - *q).abs();
            assert!(
                diff <= 1.0,
                "Roundtrip failed: k={}, q={}, diff={}",
                k,
                q,
                diff
            );
        }
    }

    // Mask propagation
    #[test]
    fn pmf_mask_propagation() {
        let k = vec64![0, 1, 2, 3];
        let lambda = 2.0;

        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(2, false) }; // make index 2 null

        let arr = poisson_pmf(&k, lambda, Some(&mask), Some(1)).unwrap();
        let out_mask = mask_vec(arr.null_mask.as_ref().unwrap());

        assert_eq!(out_mask, vec![true, true, false, true]);
        assert!(arr.data[2].is_nan());
    }

    // Error handling & edge behaviour
    #[test]
    fn invalid_lambda_errors() {
        // λ = 0 is degenerate-at-0: PMF(0)=1, PMF(k>0)=0 (no error)
        let pmf0 = dense_data(poisson_pmf(&[0, 1, 2], 0.0, None, None).unwrap());
        assert_eq!(pmf0, vec64![1.0, 0.0, 0.0]);

        // still error for negative or non-finite λ
        assert!(poisson_cdf(&[1], -3.0, None, None).is_err());
        assert!(poisson_quantile(&[0.5], f64::NAN, None, None).is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        let arr = poisson_pmf(&[], 1.0, None, None).unwrap();
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());
    }

    // Scalar vs bulk consistency (spot-check)
    #[test]
    fn pmf_bulk_vs_scalar() {
        let lambda = 6.3;
        let kvals = vec64![0, 2, 5, 8, 12];
        let bulk = dense_data(poisson_pmf(&kvals, lambda, None, None).unwrap());
        for (i, &k) in kvals.iter().enumerate() {
            let scalar = dense_data(poisson_pmf(&[k], lambda, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }
}
