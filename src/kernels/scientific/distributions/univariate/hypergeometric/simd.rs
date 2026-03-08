// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Hypergeometric Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of hypergeometric distribution functions
//! utilising vectorised operations for bulk computations on discrete sampling scenarios.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the hypergeometric distribution, which models sampling without replacement
//! from finite populations. The implementations automatically fall back to scalar
//! versions when data alignment requirements are not met.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Mask, Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::{SimdFloat, SimdUint},
};

use minarrow::{Bitmask, FloatArray, Vec64, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_u64_simd_to, masked_univariate_kernel_u64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_u64_std_to, masked_univariate_kernel_u64_std_to,
};
use crate::utils::{bitmask_to_simd_mask, has_nulls};

/// SIMD-accelerated Hypergeometric PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// P(X = k) = C(K,k) × C(N-K, n-k) / C(N,n)
#[inline(always)]
pub fn hypergeometric_pmf_simd_to(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // Parameter checks
    if population == 0 || success > population || draws > population {
        return Err(KernelError::InvalidArguments(
            "hypergeometric_pmf: invalid parameters".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    // ----- Exact degenerate case: drawing the entire population -----
    if draws == population {
        if !has_nulls(null_count, null_mask) {
            // Use SIMD for simple comparison even in degenerate case
            if is_simd_aligned(k) {
                const N: usize = W64;
                let success_v = Simd::<u64, N>::splat(success);
                let one_f = Simd::<f64, N>::splat(1.0);
                let zero_f = Simd::<f64, N>::splat(0.0);
                let mut i = 0;
                while i + N <= k.len() {
                    let k_v = Simd::<u64, N>::from_slice(&k[i..i + N]);
                    let eq_mask = k_v.simd_eq(success_v);
                    let result = eq_mask.select(one_f, zero_f);
                    output[i..i + N].copy_from_slice(result.as_array());
                    i += N;
                }
                // Handle remainder with scalar
                for idx in i..k.len() {
                    output[idx] = if k[idx] == success { 1.0 } else { 0.0 };
                }
            } else {
                for (idx, &ki) in k.iter().enumerate() {
                    output[idx] = if ki == success { 1.0 } else { 0.0 };
                }
            }
            return Ok(());
        } else {
            let mask = null_mask.expect("null_count > 0 requires null_mask");
            // Use SIMD for masked case too
            if is_simd_aligned(k) {
                const N: usize = W64;
                let success_v = Simd::<u64, N>::splat(success);
                let one_f = Simd::<f64, N>::splat(1.0);
                let zero_f = Simd::<f64, N>::splat(0.0);
                let nan_f = Simd::<f64, N>::splat(f64::NAN);
                let mask_bytes = mask.as_bytes();
                let mut i = 0;
                while i + N <= k.len() {
                    let k_v = Simd::<u64, N>::from_slice(&k[i..i + N]);
                    let lane_mask: Mask<i64, N> =
                        bitmask_to_simd_mask::<N, i64>(mask_bytes, i, k.len());
                    let eq_mask = k_v.simd_eq(success_v);
                    let vals = eq_mask.select(one_f, zero_f);
                    let result = lane_mask.select(vals, nan_f);
                    output[i..i + N].copy_from_slice(result.as_array());
                    i += N;
                }
                // Handle remainder with scalar
                for idx in i..k.len() {
                    if unsafe { mask.get_unchecked(idx) } {
                        output[idx] = if k[idx] == success { 1.0 } else { 0.0 };
                    } else {
                        output[idx] = f64::NAN;
                    }
                }
            } else {
                for (idx, &ki) in k.iter().enumerate() {
                    if unsafe { mask.get_unchecked(idx) } {
                        output[idx] = if ki == success { 1.0 } else { 0.0 };
                    } else {
                        output[idx] = f64::NAN;
                    }
                }
            }
            return Ok(());
        }
    }

    // ----- Common pre-computations -----
    const N: usize = W64;
    let ln_denom = ln_choose(population, draws);
    let min_k = success.min(draws);

    // SIMD constants
    let n_suc_v = Simd::<f64, N>::splat(success as f64);
    let n_draw_v = Simd::<f64, N>::splat(draws as f64);
    let n_pop_v = Simd::<f64, N>::splat(population as f64);
    let ln_den_v = Simd::<f64, N>::splat(ln_denom);
    let zero_f = Simd::<f64, N>::splat(0.0);
    let one_f = Simd::<f64, N>::splat(1.0);

    // Scalar lane
    let scalar_body = |ki: u64| -> f64 {
        if ki <= min_k && draws >= ki && draws - ki <= population - success {
            let v = (ln_choose(success, ki) + ln_choose(population - success, draws - ki)
                - ln_denom)
                .exp();
            v.max(0.0).min(1.0) // clamp to [0,1]
        } else {
            0.0
        }
    };

    // SIMD lane
    let simd_body = |ku: Simd<u64, N>| -> Simd<f64, N> {
        let kf = ku.cast::<f64>();
        let ln1 = ln_choose_simd(n_suc_v, kf);
        let ln2 = ln_choose_simd(n_pop_v - n_suc_v, n_draw_v - kf);
        let pmf = (ln1 + ln2 - ln_den_v).exp();

        // validity: 0 ≤ k ≤ min(draws, success) and draws-k ≤ population-success
        let valid = kf.simd_ge(zero_f)
            & kf.simd_le(n_draw_v.simd_min(n_suc_v))
            & (n_draw_v - kf).simd_le(n_pop_v - n_suc_v);

        // zero invalid lanes, clamp valid lanes to [0,1]
        let pmf = valid.select(pmf, zero_f);
        pmf.simd_min(one_f).simd_max(zero_f)
    };

    // ----- Dense vs masked, with SIMD alignment check -----
    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(k) {
            dense_univariate_kernel_u64_simd_to::<N, _, _>(k, output, simd_body, scalar_body);
        } else {
            // scalar fallback when not 64B-aligned
            dense_univariate_kernel_u64_std_to(k, output, scalar_body);
        }
    } else {
        let in_mask = null_mask.expect("hypergeometric_pmf: null_count > 0 requires null_mask");
        let mut out_mask = in_mask.clone();
        if is_simd_aligned(k) {
            masked_univariate_kernel_u64_simd_to::<N, _, _>(
                k,
                in_mask,
                output,
                &mut out_mask,
                simd_body,
                scalar_body,
            );
        } else {
            // scalar fallback when not 64B-aligned
            masked_univariate_kernel_u64_std_to(k, in_mask, output, &mut out_mask, scalar_body);
        }
    }

    Ok(())
}

/// SIMD-accelerated implementation of hypergeometric distribution probability mass function.
///
/// Computes the probability mass function (PMF) of the hypergeometric distribution
/// using vectorised SIMD operations for enhanced performance on discrete sampling
/// without replacement scenarios.
///
/// ## Parameters
/// - `k`: Input success counts where PMF should be evaluated (domain: non-negative integers)
/// - `population`: Total population size N > 0
/// - `success`: Number of success states in population K ≤ N
/// - `draws`: Number of draws (sample size) n ≤ N
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PMF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid population parameters
///
/// ## Domain Constraints
/// Valid parameter relationships:
/// - N ≥ 1 - non-empty population
/// - 0 ≤ K ≤ N - success states cannot exceed population
/// - 0 ≤ n ≤ N - cannot draw more than population
/// - max(0, n+K-N) ≤ k ≤ min(n, K) - logical sampling constraints
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When population = 0, success > population, or draws > population
///
/// ## Example Usage
/// ```rust,ignore
/// // Urn problem: 10 balls total, 4 red, drawing 3 balls
/// let k = [0, 1, 2, 3];        // number of red balls drawn
/// let population = 10;         // N = 10 total balls
/// let success = 4;             // K = 4 red balls
/// let draws = 3;               // n = 3 draws
/// let result = hypergeometric_pmf_simd(&k, population, success, draws, None, None)?;
/// // Returns probabilities for drawing 0, 1, 2, or 3 red balls
/// ```
#[inline(always)]
pub fn hypergeometric_pmf_simd(
    k: &[u64],
    population: u64,
    success: u64,
    draws: u64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    hypergeometric_pmf_simd_to(
        k,
        population,
        success,
        draws,
        out.as_mut_slice(),
        null_mask,
        null_count,
    )?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
