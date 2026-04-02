// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # Multinomial Distribution
//!
//! **************************************************************************************************
//! ⚠️ Warning: This module has not been fully tested, and is not ready for production use.
//! This warning applies to all multivariate kernels in *SIMD-kernels*, which are to be finalised
//! in an upcoming release.
//! **************************************************************************************************
//!
//! The multinomial distribution generalises the binomial distribution to multiple categories,
//! modelling the probability of observing specific counts across k mutually exclusive outcomes
//! in n independent trials. Each trial results in exactly one of the k possible outcomes.
//!
//! ## Mathematical Definition
//!
//! For k categories with probabilities p₁, p₂, ..., pₖ (where Σpᵢ = 1) and n trials:
//! ```text
//! P(X₁=x₁, X₂=x₂, ..., Xₖ=xₖ) = n! / (x₁!x₂!...xₖ!) × p₁^x₁ × p₂^x₂ × ... × pₖ^xₖ
//! ```
//!
//! Where xᵢ ≥ 0 and Σxᵢ = n (total count constraint).
//!
//! ## Applications
//!
//! - **Market research**: Customer preference across multiple brands
//! - **Genetics**: Allele frequencies in population studies
//! - **Text analysis**: Word frequencies and topic modelling
//! - **Quality control**: Defect classification in manufacturing
//! - **Survey analysis**: Multiple-choice response patterns
//! - **Machine learning**: Multi-class classification outcomes
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::kernels::scientific::distributions::shared::scalar::ln_gamma;
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Computes the probability mass function of the multinomial distribution.
///
/// Calculates the probability of observing specific count combinations across multiple
/// categories in n independent trials, where each trial results in exactly one of k
/// mutually exclusive outcomes with known probabilities.
///
/// ## Input Format
/// The function processes data in a flattened format where multiple observations (rows)
/// are concatenated:
/// - **Single observation**: [x₁, x₂, ..., xₖ] (length k)
/// - **Multiple observations**: [x₁₁, x₁₂, ..., x₁ₖ, x₂₁, x₂₂, ..., x₂ₖ, ...] (length n_rows × k)
///
/// Each row represents one multinomial observation with counts across all k categories.
///
/// ## Parameters
/// - `x`: Flattened array of count observations (length = n_rows × k)
/// - `n`: Number of trials per observation (must be > 0)
/// - `p`: Category probabilities (length k ≥ 2, sum = 1, each ∈ (0,1))
/// - `null_mask`: Optional bitmask for null observations
/// - `null_count`: Optional count of null observations for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: Array with one PMF value per observation row
/// - **Error**: KernelError::InvalidArguments for constraint violations
///
/// ## Example Usage
/// ```rust,ignore
/// // 3-category multinomial with 5 trials
/// let x = [2, 1, 2, 1, 0, 4];  // Two observations: (2,1,2) and (1,0,4)
/// let n = 5;                    // 5 trials each
/// let p = [0.3, 0.2, 0.5];     // Category probabilities
/// let result = multinomial_pmf(&x, n, &p, None, None)?;
/// // Returns PMF for each observation row
/// ```
#[inline(always)]
pub fn multinomial_pmf(
    x: &[u64],
    n: u64,
    p: &[f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let k = p.len();
    if k < 2 {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: need at least 2 categories".into(),
        ));
    }
    if x.is_empty() {
        return Ok(FloatArray::from_slice(&[]));
    }
    if x.len() % k != 0 {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: x length not multiple of p length".into(),
        ));
    }
    if n == 0 {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: n=0 not allowed".into(),
        ));
    }
    for &pi in p {
        if !(pi > 0.0 && pi < 1.0) || !pi.is_finite() {
            return Err(KernelError::InvalidArguments(
                "multinomial_pmf: all p in (0,1)".into(),
            ));
        }
    }
    let sum_p: f64 = p.iter().sum();
    if (sum_p - 1.0).abs() > 1e-12 {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: probabilities must sum to 1".into(),
        ));
    }
    let rows = x.len() / k;

    // Fast dense path
    if !has_nulls(null_count, null_mask) {
        let mut out = Vec64::with_capacity(rows);
        for row in 0..rows {
            let xi = &x[row * k..(row + 1) * k];
            let sum_x: u64 = xi.iter().sum();
            if sum_x != n {
                out.push(0.0);
                continue;
            }
            let mut log_pmf = ln_gamma(n as f64 + 1.0);
            for j in 0..k {
                log_pmf -= ln_gamma(xi[j] as f64 + 1.0);
                if xi[j] > 0 {
                    log_pmf += (xi[j] as f64) * p[j].ln();
                }
            }
            out.push(log_pmf.exp());
        }
        return Ok(FloatArray::from_vec64(out, null_mask.cloned()));
    }

    // Null-aware path. We propagate input nulls,
    // and fill a NaN sentinel.
    let mask = null_mask.expect("null path requires a mask");
    let mut out = Vec64::with_capacity(rows);

    for row in 0..rows {
        if !unsafe { mask.get_unchecked(row) } {
            out.push(f64::NAN);
            continue;
        }
        let xi = &x[row * k..(row + 1) * k];
        let sum_x: u64 = xi.iter().sum();
        if sum_x != n {
            out.push(0.0);
            continue;
        }
        let mut log_pmf = ln_gamma(n as f64 + 1.0);
        for j in 0..k {
            log_pmf -= ln_gamma(xi[j] as f64 + 1.0);
            if xi[j] > 0 {
                log_pmf += (xi[j] as f64) * p[j].ln();
            }
        }
        let pmf = log_pmf.exp();
        out.push(pmf);
    }
    Ok(FloatArray {
        data: out.into(),
        null_mask: Some(mask.clone()),
    })
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    //  Common helpers

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol={tol})"
        );
    }

    // Simple factorial/combinations helpers for reference numbers
    fn fact(n: u64) -> f64 {
        (1..=n).fold(1.0_f64, |acc, v| acc * v as f64)
    }
    fn choose(n: u64, k: u64) -> f64 {
        fact(n) / (fact(k) * fact(n - k))
    }

    //  multinomial_pmf - correctness
    #[test]
    fn pmf_reference_single_row() {
        // n = 4;  p = (0.2, 0.5, 0.3);  x = (1,2,1)
        let n = 4;
        let p = vec64![0.2, 0.5, 0.3];
        let x = vec64![1, 2, 1]; // one row
        //
        // Expected:  4!/(1!2!1!) * 0.2¹ * 0.5² * 0.3¹
        let expect = (fact(4) / (fact(1) * fact(2) * fact(1)))
            * 0.2_f64.powi(1)
            * 0.5_f64.powi(2)
            * 0.3_f64.powi(1);

        let arr = dense_data(multinomial_pmf(&x, n, &p, None, None).unwrap());
        assert_close(arr[0], expect, 1e-15);
    }

    #[test]
    fn pmf_multiple_rows() {
        // 2 rows, n = 5
        let n = 5;
        let p = vec64![0.1, 0.3, 0.6];
        // rows: (2,2,1)  and (0,3,2)
        let x = vec64![2, 2, 1, 0, 3, 2];
        let arr = dense_data(multinomial_pmf(&x, n, &p, None, None).unwrap());

        // manual reference
        let r0 = choose(5, 2)                     // choose slots for first cat
               * choose(3, 2)                     // choose 2 of remaining for second
               * 0.1_f64.powi(2) * 0.3_f64.powi(2) * 0.6_f64.powi(1);
        let r1 = choose(5, 0) * choose(5, 3)      // (0,3,2)
               * 0.1_f64.powi(0) * 0.3_f64.powi(3) * 0.6_f64.powi(2);

        assert_close(arr[0], r0, 1e-14);
        assert_close(arr[1], r1, 1e-14);
    }

    #[test]
    fn pmf_sum_x_not_equal_n_returns_zero() {
        // Second row does *not* sum to n
        let n = 4;
        let p = vec64![0.25, 0.25, 0.5];
        let x = vec64![
            1, 1, 2, // ok
            2, 2, 2
        ]; // sums to 6 ≠ 4
        let arr = dense_data(multinomial_pmf(&x, n, &p, None, None).unwrap());
        assert!(arr[0] > 0.0);
        assert_eq!(arr[1], 0.0);
    }

    //  Mask / null propagation
    #[test]
    fn pmf_null_mask_propagation() {
        let n = 3;
        let p = vec64![0.2, 0.3, 0.5];
        let x = vec64![
            1, 1, 1, // row 0
            0, 0, 0, // row 1 - will be masked
            3, 0, 0
        ]; // row 2
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(1, false) }; // mask out row 1

        let arr = multinomial_pmf(&x, n, &p, Some(&mask), Some(1)).unwrap();
        let mvec = mask_vec(arr.null_mask.as_ref().unwrap());

        assert!(mvec == [true, false, true]);
        assert!(arr.data[1].is_nan()); // sentinel for masked lane
    }

    //  Validation / error handling
    #[test]
    fn invalid_input_dimension_errors() {
        // x length not a multiple of k
        let p = vec64![0.5, 0.5];
        let res = multinomial_pmf(&[1, 1, 1], 3, &p, None, None);
        assert!(res.is_err());
    }

    #[test]
    fn invalid_probabilities_error() {
        // p outside (0,1)
        let p = vec64![0.2, 0.85]; // sum > 1 & a prob >1
        let res = multinomial_pmf(&[1, 2, 1, 0], 4, &p, None, None);
        assert!(res.is_err());
    }

    #[test]
    fn probabilities_do_not_sum_to_one_error() {
        let p = vec64![0.1, 0.2, 0.3]; // sum 0.6
        let res = multinomial_pmf(&[1, 1, 2], 4, &p, None, None);
        assert!(res.is_err());
    }

    #[test]
    fn zero_trials_error() {
        let p = vec64![0.5, 0.5];
        let res = multinomial_pmf(&[0, 0], 0, &p, None, None);
        assert!(res.is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        let p = vec64![0.4, 0.6];
        let arr = multinomial_pmf(&[], 0, &p, None, None).unwrap();
        assert!(arr.data.is_empty());
    }
}
