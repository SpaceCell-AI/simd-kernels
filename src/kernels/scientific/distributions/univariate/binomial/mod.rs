// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Binomial Distribution Module** - *Discrete Probability Distribution*
//!
//! High-performance implementation of the binomial distribution with SIMD-accelerated kernels
//! for probability mass function (PMF), cumulative distribution function (CDF), and
//! quantile (inverse CDF) calculations.
//!
//! ## Overview
//! The binomial distribution Binomial(n, p) is a discrete probability distribution that
//! models the number of successes in n independent Bernoulli trials, each with success
//! probability p. It is fundamental in statistics, quality control, and scientific computing.
//!
//! ## Mathematical Definition
//! - **PMF**: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
//! - **CDF**: P(X ≤ k) = Σᵢ₌₀ᵏ C(n,i) × pⁱ × (1-p)^(n-i)
//! - **Support**: k ∈ {0, 1, 2, ..., n}
//! - **Parameters**: n ∈ ℕ₀ (number of trials), p ∈ [0, 1] (success probability)
//!
//! Where C(n,k) = n!/(k!(n-k)!) is the binomial coefficient.
//!
//! ## Use Cases
//! - Quality control: defect rates in manufacturing
//! - Clinical trials: success rates in medical treatments  
//! - A/B testing: conversion rate analysis
//! - Survey sampling: proportion estimation
//! - Machine learning: binary classification metrics
//! - Finance: default probability modelling

#[cfg(feature = "simd")]
mod simd;
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Binomial PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn binomial_pmf_to(
    k: &[u64],
    n: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::binomial_pmf_simd_to(k, n, p, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::binomial_pmf_std_to(k, n, p, output, null_mask, null_count)
    }
}

/// Computes the probability mass function (PMF) of the binomial distribution.
#[inline(always)]
pub fn binomial_pmf(
    k: &[u64],
    n: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::binomial_pmf_simd(k, n, p, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::binomial_pmf_std(k, n, p, null_mask, null_count)
    }
}

/// Binomial CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn binomial_cdf_to(
    k: &[u64],
    n: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::binomial_cdf_simd_to(k, n, p, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::binomial_cdf_std_to(k, n, p, output, null_mask, null_count)
    }
}

/// Computes the cumulative distribution function (CDF) of the binomial distribution.
#[inline(always)]
pub fn binomial_cdf(
    k: &[u64],
    n: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::binomial_cdf_simd(k, n, p, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::binomial_cdf_std(k, n, p, null_mask, null_count)
    }
}

/// Binomial quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn binomial_quantile_to(
    p: &[f64],
    n: u64,
    p_: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::binomial_quantile_std_to(p, n, p_, output, null_mask, null_count)
}

/// Computes the quantile function (inverse CDF) of the binomial distribution.
#[inline(always)]
pub fn binomial_quantile(
    p: &[f64],
    n: u64,
    p_: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::binomial_quantile_std(p, n, p_, null_mask, null_count)
}

#[cfg(test)]
mod binomial_tests {
    use crate::kernels::scientific::distributions::{
        shared::scalar::ln_gamma, univariate::common::dense_data,
    };

    use super::*;
    use minarrow::{Bitmask, vec64};

    // See `./tests` for the scipy test suite

    // helpers - identical to those in the Normal/Beta suites

    fn mask_vec(m: &Bitmask) -> Vec<bool> {
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
    fn ln_choose(n: u64, k: u64) -> f64 {
        ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
    }
    fn pmf_ref(n: u64, p: f64, k: u64) -> f64 {
        if k > n {
            return 0.0;
        }
        (ln_choose(n, k) + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln()).exp()
    }
    fn cdf_ref(n: u64, p: f64, k: u64) -> f64 {
        (0..=k).map(|j| pmf_ref(n, p, j)).sum()
    }
    fn quantile_ref(n: u64, p_success: f64, prob: f64) -> u64 {
        // Small-n exact search (only used for tests with n≤20)
        let mut cdf = 0.0;
        for k in 0..=n {
            cdf += pmf_ref(n, p_success, k);
            if cdf >= prob {
                return k;
            }
        }
        n
    }

    // parameters used in exact tests
    const N_REF: u64 = 10;
    const P_REF: f64 = 0.3;

    // binomial_pmf - correctness

    #[test]
    fn binomial_pmf_exact_values() {
        // k = 0..N_REF
        let ks: Vec<u64> = (0..=N_REF).collect();
        let expect: Vec<f64> = ks.iter().map(|&k| pmf_ref(N_REF, P_REF, k)).collect();
        let arr = dense_data(binomial_pmf(&ks, N_REF, P_REF, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-14);
        }
        // PMF must sum to 1
        assert_close(arr.iter().sum::<f64>(), 1.0, 1e-14);
    }

    #[test]
    fn binomial_pmf_bulk_vs_scalar() {
        let ks = vec64![0, 3, 5, 10];
        let bulk = dense_data(binomial_pmf(&ks, N_REF, P_REF, None, None).unwrap());
        for (i, &k) in ks.iter().enumerate() {
            let s = dense_data(binomial_pmf(&[k], N_REF, P_REF, None, None).unwrap())[0];
            assert_close(bulk[i], s, 1e-14);
        }
    }

    #[test]
    fn binomial_pmf_out_of_range_zero() {
        // k > n  -> 0
        let ks = vec64![N_REF + 1, N_REF + 5];
        let arr = dense_data(binomial_pmf(&ks, N_REF, P_REF, None, None).unwrap());
        assert!(arr.iter().all(|&v| v == 0.0));
    }

    // binomial_pmf - mask propagation

    #[test]
    fn binomial_pmf_mask_propagation() {
        let ks = vec64![1, 2, 3];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(1, false);
        } // null middle lane
        let out = binomial_pmf(&ks, 5, 0.4, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(out.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![true, false, true]);
        assert!(out.data[1].is_nan());
    }

    // binomial_cdf - correctness & monotonicity

    #[test]
    fn binomial_cdf_exact_values() {
        let ks = vec64![0, 3, 7, 10];
        let expect: Vec<f64> = ks.iter().map(|&k| cdf_ref(N_REF, P_REF, k)).collect();
        let arr = dense_data(binomial_cdf(&ks, N_REF, P_REF, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_close(*a, *e, 1e-14);
        }
    }

    #[test]
    fn binomial_cdf_monotone() {
        let ks: Vec<u64> = (0..=N_REF).collect();
        let arr = dense_data(binomial_cdf(&ks, N_REF, P_REF, None, None).unwrap());
        for pair in arr.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
        assert_close(arr[0], pmf_ref(N_REF, P_REF, 0), 1e-14); // first CDF = PMF(0)
        assert_close(arr[N_REF as usize], 1.0, 1e-14); // last  = 1
    }

    // binomial_quantile - correctness & round-trip (small n)

    #[test]
    fn binomial_quantile_exact_small_n() {
        let probs = vec64![0.05, 0.2, 0.5, 0.8, 0.95];
        let expect: Vec<f64> = probs
            .iter()
            .map(|&p| quantile_ref(N_REF, P_REF, p) as f64)
            .collect();
        let arr = dense_data(binomial_quantile(&probs, N_REF, P_REF, None, None).unwrap());
        for (a, e) in arr.iter().zip(expect.iter()) {
            assert_eq!(*a, *e);
        } // integers match exactly
    }

    #[test]
    fn binomial_round_trip_small_n() {
        // Confirm that for p strictly between 0 and 1:
        //    k = Q(p)   ⇒   F(k) ≥ p  and  F(k-1) < p   (definition of quantile)
        let probs = vec64![0.01, 0.1, 0.33, 0.6, 0.99];
        let k_arr = dense_data(binomial_quantile(&probs, N_REF, P_REF, None, None).unwrap());
        for (&p, &k_f) in probs.iter().zip(k_arr.iter()) {
            let k = k_f as u64;
            let cdf_k = cdf_ref(N_REF, P_REF, k);
            let cdf_km1 = if k == 0 {
                0.0
            } else {
                cdf_ref(N_REF, P_REF, k - 1)
            };
            assert!(cdf_k >= p && cdf_km1 < p);
        }
    }

    // domain violations & NaN behaviour

    #[test]
    fn binomial_quantile_domain_violations_nan() {
        let probs = vec64![-0.1, 1.1, f64::NAN];
        let arr = dense_data(binomial_quantile(&probs, N_REF, P_REF, None, None).unwrap());
        assert!(arr.iter().all(|v| v.is_nan()));
    }

    // parameter validation

    #[test]
    fn binomial_invalid_parameter_errors() {
        assert!(binomial_pmf(&[1], 5, -0.1, None, None).is_err());
        assert!(binomial_cdf(&[1], 5, 1.1, None, None).is_err());
        assert!(
            binomial_quantile(&[0.5], 0, 0.3, None, None)
                .unwrap()
                .data
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(binomial_quantile(&[0.5], 10, 0.0, None, None).is_err());
    }

    // empty input + mask round-trip

    #[test]
    fn binomial_empty_inputs() {
        assert!(
            binomial_pmf(&[], 10, 0.4, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            binomial_cdf(&[], 10, 0.4, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            binomial_quantile(&[], 10, 0.4, None, None)
                .unwrap()
                .data
                .is_empty()
        );
    }

    #[test]
    fn binomial_quantile_mask_propagation() {
        let p = vec64![0.2, 0.4, 0.6];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(2, false);
        }
        let arr = binomial_quantile(&p, 10, 0.3, Some(&m), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(nulls, vec![true, true, false]);
        assert!(arr.data[2].is_nan());
    }
}
