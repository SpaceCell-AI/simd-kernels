// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # Geometric Distribution
//!
//! The geometric distribution models the number of trials needed to achieve the first success
//! in a sequence of independent Bernoulli trials, each with success probability p. This module
//! implements both the "number of failures" (support on {0, 1, 2, ...}) and "number of trials"
//! (support on {1, 2, 3, ...}) parameterisations, following SciPy conventions.
//!
//! ## Mathematical Definition
//!
//! The geometric distribution is parameterised by a single probability p ∈ (0, 1]:
//!
//! ### SciPy Convention (Number of Trials)
//! - **PMF**: P(X = k) = (1-p)^(k-1) p for k = 1, 2, 3, ... and P(X = 0) = 0
//! - **CDF**: F(k) = 1 - (1-p)^k for k ≥ 1, and F(0) = 0
//! - **Quantile**: Q(q) = ⌈ln(1-q) / ln(1-p)⌉ for q ∈ (0, 1)
//!
//! ## Common Applications
//!
//! - **Quality control**: Number of items inspected until finding a defect
//! - **Telecommunications**: Packet retransmission analysis
//! - **Reliability engineering**: Time to first failure (discrete version)
//! - **Biology**: Genetic inheritance patterns and mutation detection
//! - **Economics**: Search models and customer behaviour analysis
//!
//! ## Parameterisation Details
//!
//! This implementation follows **SciPy's convention**:
//! - Support on {1, 2, 3, ...} (number of trials to first success)
//! - P(X = 0) = 0 and F(0) = 0 by definition
//! - Consistent with `scipy.stats.geom` behaviour

#[cfg(feature = "simd")]
mod simd;
#[cfg(not(feature = "simd"))]
mod std;

use minarrow::{Bitmask, FloatArray};

use minarrow::enums::error::KernelError;

/// Geometric PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_pmf_to(
    k: &[u64],
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::geometric_pmf_simd_to(k, p, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::geometric_pmf_std_to(k, p, output, null_mask, null_count)
    }
}

/// Geometric CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_cdf_to(
    k: &[u64],
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::geometric_cdf_simd_to(k, p, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::geometric_cdf_std_to(k, p, output, null_mask, null_count)
    }
}

/// Geometric quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_quantile_to(
    pv: &[f64],
    p_succ: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::geometric_quantile_simd_to(pv, p_succ, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::geometric_quantile_std_to(pv, p_succ, output, null_mask, null_count)
    }
}
/// Compute the probability mass function (PMF) for the geometric distribution.
///
/// The geometric distribution models the number of trials needed to achieve the first success
/// in a sequence of independent Bernoulli trials, each with success probability p. This
/// implementation follows the SciPy convention with support on {1, 2, 3, ...} representing
/// the number of trials to first success.
///
/// ## Mathematical Definition
///
/// The PMF of the geometric distribution is defined as:
///
/// ```text
/// P(X = k) = (1-p)^(k-1) × p   for k = 1, 2, 3, ...
/// P(X = 0) = 0                 by SciPy convention
/// ```
///
/// where:
/// - `p` is the success probability for each trial, with `p ∈ (0, 1]`
/// - `k` is the number of trials to achieve the first success
///
/// ## Parameters
///
/// * `k` - Number of trials to first success (non-negative integers, but P(X=0) = 0)
/// * `p` - Success probability for each trial, must be in (0, 1]
/// * `null_mask` - Optional bitmask for null value handling in Arrow format
/// * `null_count` - Optional count of null values for optimisation
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing the PMF values, or a `KernelError` if:
/// - Success probability is not in (0, 1] (p ≤ 0 or p > 1)
/// - Success probability is not finite (NaN or infinite)
///
/// ## Applications
///
/// - **Quality control**: Number of inspections until finding a defective item
/// - **Telecommunications**: Packet retransmission analysis and protocol design
/// - **Reliability engineering**: Time to first failure in discrete-time systems
/// - **Biology**: Genetic studies and mutation detection timing
/// - **Economics**: Search models and customer acquisition analysis
/// - **Games and simulation**: Modelling random events and wait times
#[inline(always)]
pub fn geometric_pmf(
    k: &[u64],
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::geometric_pmf_simd(k, p, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::geometric_pmf_std(k, p, null_mask, null_count)
    }
}

/// Geometric CDF: F(X ≤ k) = 1 - (1-p)^k for k ≥ 1, 0 for k < 1 (scipy convention)
#[inline(always)]
pub fn geometric_cdf(
    k: &[u64],
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::geometric_cdf_simd(k, p, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::geometric_cdf_std(k, p, null_mask, null_count)
    }
}

/// Computes the quantile function (inverse CDF) of the geometric distribution.
///
/// Finds the smallest integer k such that F(k) ≥ q for each cumulative probability q
/// in the input array. This follows the SciPy convention for discrete distributions.
///
/// # Mathematical Definition
///
/// The geometric quantile function finds k such that:
/// ```text
/// k = Q(q) = ⌈ln(1-q) / ln(1-p)⌉  for q ∈ (0, 1)
/// Q(0) = 1                        (by SciPy convention)
/// Q(1) = +∞                      (theoretical limit)
/// ```
///
/// Where ⌈·⌉ denotes the ceiling function, ensuring integer outputs appropriate
/// for a discrete distribution.
///
/// # Parameters
///
/// * `pv` - Cumulative probability values in [0, 1] for which quantiles are computed
/// * `p_succ` - Success probability ∈ (0, 1] for each trial
/// * `null_mask` - Optional bitmask indicating null values in the input
/// * `null_count` - Optional count of null values for optimisation
///
/// # Returns
///
/// Returns a `FloatArray<f64>` containing the quantile values (as floating-point
/// representations of integers), or a `KernelError` if p_succ is outside (0, 1]
/// or non-finite.
///
/// # Domain and Range
///
/// - **Domain**: q ∈ [0, 1] (returns NaN for q outside this range)
/// - **Range**: {1, 2, 3, ...} ∪ {+∞} (discrete values ≥ 1)
/// - **Boundary conditions**: Q(0) = 1, Q(1) = +∞
///
/// # Algorithm Details
///
/// Uses direct logarithmic inversion with ceiling operation:
/// - Efficient for all practical parameter ranges
/// - Robust handling of boundary cases
/// - Maintains integer semantics appropriate for discrete distributions
///
/// # Example
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::geometric::geometric_quantile;
///
/// let probabilities = [0.0, 0.3, 0.6, 0.9]; // Cumulative probabilities
/// let result = geometric_quantile(&probabilities, 0.3, None, None).unwrap();
/// // Returns minimum number of trials needed to achieve each cumulative probability
/// ```
#[inline(always)]
pub fn geometric_quantile(
    pv: &[f64],
    p_succ: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::geometric_quantile_simd(pv, p_succ, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::geometric_quantile_std(pv, p_succ, null_mask, null_count)
    }
}

#[cfg(test)]
mod geometric_tests {
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

    // geometric_pmf  - correctness

    #[test]
    fn geom_pmf_basic_values() {
        // SciPy convention:
        // P(X=k) = (1-p)^(k-1) * p  for k=1,2,... and P(0)=0
        let k = vec64![0_u64, 1, 2, 3];
        let expect = vec64![0.0, 0.3, 0.21, 0.147];
        let out = dense_data(geometric_pmf(&k, 0.3, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-15);
        }
    }

    #[test]
    fn geom_pmf_scalar_vs_bulk() {
        // includes k=0; PMF(0)=0 under SciPy
        let k = vec64![0_u64, 5, 10, 20];
        let p = 0.42;
        let bulk = dense_data(geometric_pmf(&k, p, None, None).unwrap());
        for (i, &ki) in k.iter().enumerate() {
            let scalar = dense_data(geometric_pmf(&[ki], p, None, None).unwrap())[0];
            assert_close(bulk[i], scalar, 1e-15);
        }
    }

    // geometric_cdf  - correctness & tails (SciPy: CDF(0)=0)

    #[test]
    fn geom_cdf_basic_values() {
        // Verified against scipy.stats.geom.cdf
        let k = vec64![0_u64, 1, 2, 3];
        let p = 0.3;
        let expect = vec64![
            0.0,   // CDF(0) = 0
            0.3,   // CDF(1) = p
            0.51,  // 1 - (1-p)^2
            0.657  // 1 - (1-p)^3
        ];
        let out = dense_data(geometric_cdf(&k, p, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-15);
        }
    }

    #[test]
    fn geom_cdf_tail_behaviour() {
        let p = 0.25;
        let out = dense_data(geometric_cdf(&[0_u64, 1_000_000], p, None, None).unwrap());
        assert_close(out[0], 0.0, 1e-15); // k = 0 (SciPy convention)
        assert_close(out[1], 1.0, 1e-12); // very large k -> 1
    }

    // geometric_quantile  - correctness & round-trip

    #[test]
    fn geom_quantile_basic() {
        let p_succ = 0.3;
        let probs = [0.0, 0.3, 0.51, 0.7599, 0.9];
        // SciPy geom.ppf: PPF(0)=1, PPF(0.3)=1, PPF(0.51)=2, ...
        let expect = [1.0, 1.0, 2.0, 4.0, 7.0];
        let out = dense_data(geometric_quantile(&probs, p_succ, None, None).unwrap());
        for (a, e) in out.iter().zip(expect) {
            assert_close(*a, e, 1e-12);
        }
    }

    #[test]
    fn geom_quantile_roundtrip() {
        let p_succ = 0.42;
        // Use support k≥1 for round-trip with SciPy convention
        let k = vec64![1_u64, 3, 7, 15];
        let cdf = dense_data(geometric_cdf(&k, p_succ, None, None).unwrap());
        let k2 = dense_data(geometric_quantile(&cdf, p_succ, None, None).unwrap());
        for (orig, back) in k.iter().zip(k2.iter()) {
            assert_close(*orig as f64, *back, 1e-6);
        }
    }

    #[test]
    fn geom_quantile_domain_extremes() {
        let p_succ = 0.5;
        let probs = vec64![-0.1, 0.0, 1.0, 1.1, f64::NAN];
        let out = dense_data(geometric_quantile(&probs, p_succ, None, None).unwrap());
        assert!(out[0].is_nan());
        assert_close(out[1], 1.0, 0.0); // SciPy: PPF(0) = 1
        assert!(out[2].is_infinite() && out[2].is_sign_positive());
        assert!(out[3].is_nan());
        assert!(out[4].is_nan());
    }

    // mask propagation

    #[test]
    fn geom_pmf_mask_propagation() {
        let k = vec64![0_u64, 1, 2];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(1, false);
        }
        let arr = geometric_pmf(&k, 0.3, Some(&m), Some(1)).unwrap();
        let mv = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(mv, vec![true, false, true]);
        assert!(arr.data[1].is_nan());
    }

    #[test]
    fn geom_cdf_mask_propagation() {
        let k = vec64![0_u64, 5, 10];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(0, false);
        }
        let arr = geometric_cdf(&k, 0.3, Some(&m), Some(1)).unwrap();
        let mv = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(mv, vec![false, true, true]);
        assert!(arr.data[0].is_nan());
    }

    #[test]
    fn geom_quantile_mask_propagation() {
        let p = vec64![0.2, 0.6, 0.8];
        let mut m = Bitmask::new_set_all(3, true);
        unsafe {
            m.set_unchecked(2, false);
        }
        let arr = geometric_quantile(&p, 0.4, Some(&m), Some(1)).unwrap();
        let mv = mask_vec(arr.null_mask.as_ref().unwrap());
        assert_eq!(mv, vec![true, true, false]);
        assert!(arr.data[2].is_nan());
    }

    // parameter validation & error paths

    #[test]
    fn geom_invalid_parameters() {
        // p must be finite and within [0,1]; p=1 is allowed (degenerate), p=0 is not
        assert!(geometric_pmf(&[1], f64::NAN, None, None).is_err());
        assert!(geometric_pmf(&[1], -0.1, None, None).is_err());
        assert!(geometric_pmf(&[1], 1.1, None, None).is_err());
        assert!(geometric_pmf(&[1], 0.0, None, None).is_err());

        assert!(geometric_cdf(&[1], f64::INFINITY, None, None).is_err());
        assert!(geometric_cdf(&[1], -0.5, None, None).is_err());
        assert!(geometric_cdf(&[1], 1.0000000001, None, None).is_err());
        assert!(geometric_cdf(&[1], 0.0, None, None).is_err());

        assert!(geometric_quantile(&[0.5], f64::NEG_INFINITY, None, None).is_err());
        assert!(geometric_quantile(&[0.5], -0.3, None, None).is_err());
        assert!(geometric_quantile(&[0.5], 0.0, None, None).is_err());
        assert!(geometric_quantile(&[0.5], 1.0000000001, None, None).is_err());
    }

    // empty-input fast paths

    #[test]
    fn geom_empty_inputs() {
        let pmf = geometric_pmf(&[], 0.3, None, None).unwrap();
        let cdf = geometric_cdf(&[], 0.3, None, None).unwrap();
        let qtl = geometric_quantile(&[], 0.3, None, None).unwrap();
        assert!(pmf.data.is_empty() && cdf.data.is_empty() && qtl.data.is_empty());
        assert!(pmf.null_mask.is_none() && cdf.null_mask.is_none() && qtl.null_mask.is_none());
    }
}
