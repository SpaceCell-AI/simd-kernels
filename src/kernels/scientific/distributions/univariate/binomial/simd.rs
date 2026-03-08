// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Binomial Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of binomial distribution functions
//! utilising vectorised operations for bulk computations on 64-byte aligned data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! that process multiple binomial distribution evaluations simultaneously using
//! CPU vector instructions. The implementations automatically fall back to scalar
//! versions when data alignment requirements are not met.

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray, Vec64};
use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat};

use crate::kernels::scientific::distributions::shared::scalar::{ln_gamma_simd, *};
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;
use minarrow::utils::is_simd_aligned;

/// SIMD-accelerated binomial PMF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn binomial_pmf_simd_to(
    k: &[u64],
    n: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(0.0 <= p && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "binomial_pmf: invalid p".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let ln_choose_n = ln_gamma((n as f64) + 1.0);

    let ln_choose_n_v = Simd::<f64, N>::splat(ln_choose_n);
    let ln_p_v = Simd::<f64, N>::splat(p.max(f64::MIN_POSITIVE).ln());
    let ln_q_v = Simd::<f64, N>::splat((1.0 - p).max(f64::MIN_POSITIVE).ln());
    let n_f = Simd::<f64, N>::splat(n as f64);
    let n_i = Simd::<u64, N>::splat(n);

    let scalar_body = |ki_f: f64| -> f64 {
        let ki = ki_f as u64;
        if ki <= n {
            (ln_choose_n - ln_gamma((ki as f64) + 1.0) - ln_gamma(((n - ki) as f64) + 1.0)
                + (ki as f64) * p.ln()
                + ((n - ki) as f64) * (1.0 - p).ln())
            .exp()
        } else {
            0.0
        }
    };

    let simd_body = |k_v: Simd<f64, N>| {
        let k_u = k_v.cast::<u64>();
        let valid = k_u.simd_le(n_i);

        let ln_g_k1 = ln_gamma_simd(k_v + Simd::splat(1.0));
        let ln_g_nk1 = ln_gamma_simd(n_f - k_v + Simd::splat(1.0));

        let ln_pmf = ln_choose_n_v - ln_g_k1 - ln_g_nk1 + k_v * ln_p_v + (n_f - k_v) * ln_q_v;
        let pmf = ln_pmf.exp();

        valid.select(pmf, Simd::splat(0.0))
    };

    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(&k_f64) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(&k_f64, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(&k_f64, output, scalar_body);
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    if is_simd_aligned(&k_f64) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            &k_f64,
            mask,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(&k_f64, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// SIMD-accelerated implementation of binomial distribution probability mass function.
#[inline(always)]
pub fn binomial_pmf_simd(
    k: &[u64],
    n: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    binomial_pmf_simd_to(k, n, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// SIMD-accelerated binomial CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn binomial_cdf_simd_to(
    k: &[u64],
    n: u64,
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(p >= 0.0 && p <= 1.0) || n > 1_000_000_000 {
        return Err(KernelError::InvalidArguments(
            "binomial_cdf: invalid p or n".into(),
        ));
    }
    if k.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let n_f = n as f64;

    let scalar_body = |ki_f: f64| -> f64 {
        let ki = ki_f as u64;
        if ki >= n {
            1.0
        } else if p == 0.0 {
            1.0
        } else if p == 1.0 {
            if ki >= n { 1.0 } else { 0.0 }
        } else {
            1.0 - incomplete_beta((ki + 1) as f64, n_f - ki as f64, p)
        }
    };

    let simd_body = |k_v: Simd<f64, N>| {
        let k_u = k_v.cast::<u64>();
        let n_simd = Simd::<u64, N>::splat(n);
        let one_simd = Simd::<f64, N>::splat(1.0);
        let zero_simd = Simd::<f64, N>::splat(0.0);

        let ki_ge_n = k_u.simd_ge(n_simd);
        let mut result = one_simd;

        if p == 1.0 {
            result = ki_ge_n.select(one_simd, zero_simd);
        } else if p != 0.0 {
            let mut out_arr = [0.0f64; N];
            let k_arr = k_v.to_array();
            for i in 0..N {
                let ki = k_arr[i] as u64;
                out_arr[i] = if ki >= n {
                    1.0
                } else {
                    1.0 - incomplete_beta((ki + 1) as f64, n_f - ki as f64, p)
                };
            }
            return Simd::<f64, N>::from_array(out_arr);
        }

        result
    };

    let k_f64: Vec<f64> = k.iter().map(|&ki| ki as f64).collect();

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(&k_f64) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(&k_f64, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(&k_f64, output, scalar_body);
        return Ok(());
    }

    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();
    if is_simd_aligned(&k_f64) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            &k_f64,
            mask,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(&k_f64, mask, output, &mut out_mask, scalar_body);
    Ok(())
}

/// Partially SIMD-accelerated implementation of binomial distribution cumulative distribution function.
#[inline(always)]
pub fn binomial_cdf_simd(
    k: &[u64],
    n: u64,
    p: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    binomial_cdf_simd_to(k, n, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
