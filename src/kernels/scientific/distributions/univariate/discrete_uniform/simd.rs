// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Discrete Uniform Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of discrete uniform distribution functions
//! utilising vectorised integer operations for bulk computations on aligned data arrays.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Mask, Select, Simd,
    cmp::{SimdOrd, SimdPartialOrd},
    prelude::{SimdFloat, SimdInt},
};

use minarrow::{Bitmask, FloatArray, Vec64};

use crate::utils::bitmask_to_simd_mask;
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

/// Discrete uniform PMF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn discrete_uniform_pmf_simd_to(
    k: &[i64],
    low: i64,
    high: i64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if low >= high {
        return Err(KernelError::InvalidArguments(
            "discrete_uniform_pmf: require low < high (upper exclusive)".into(),
        ));
    }

    // span = N = high - low  (≥ 1)
    let span = high.checked_sub(low).ok_or_else(|| {
        KernelError::InvalidArguments("discrete_uniform_pmf: range overflow".into())
    })?;
    let p = 1.0 / (span as f64);

    let len = k.len();
    if len == 0 {
        return Ok(());
    }

    const N: usize = W64;
    let low_v = Simd::<i64, N>::splat(low);
    let high_v = Simd::<i64, N>::splat(high);
    let p_v = Simd::<f64, N>::splat(p);
    let zero_v = Simd::<f64, N>::splat(0.0);

    if !has_nulls(null_count, null_mask) {
        let mut i = 0;
        while i + N <= len {
            let kv = Simd::<i64, N>::from_slice(&k[i..i + N]);
            let in_range = kv.simd_ge(low_v) & kv.simd_lt(high_v); // [low, high)
            let y = in_range.select(p_v, zero_v);
            output[i..i + N].copy_from_slice(&y.to_array());
            i += N;
        }
        for (idx, &ki) in k[i..].iter().enumerate() {
            output[i + idx] = if (low..high).contains(&ki) { p } else { 0.0 };
        }
        return Ok(());
    }

    let mask_ref = null_mask.expect("discrete_uniform_pmf: null_count > 0 requires null_mask");

    let mask_bytes = mask_ref.as_bytes();
    let mut i = 0;
    while i + N <= len {
        let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);

        let kv = Simd::<i64, N>::from_slice(&k[i..i + N]);

        let in_range = kv.simd_ge(low_v) & kv.simd_lt(high_v);
        let y = in_range.select(p_v, zero_v);

        let res = lane_mask.select(y, Simd::<f64, N>::splat(f64::NAN));
        output[i..i + N].copy_from_slice(res.as_array());
        i += N;
    }
    for idx in i..len {
        if !unsafe { mask_ref.get_unchecked(idx) } {
            output[idx] = f64::NAN;
        } else {
            let ki = k[idx];
            output[idx] = if (low..high).contains(&ki) { p } else { 0.0 };
        }
    }

    Ok(())
}

/// SIMD-accelerated implementation of discrete uniform distribution probability mass function.
///
/// Processes multiple PMF evaluations simultaneously using vectorised integer comparisons
/// for high performance on large datasets with integer-valued inputs.
///
/// ## Mathematical Definition
/// ```text
/// P(X = k) = 1/(high - low)  for k ∈ [low, high)
///          = 0               otherwise
/// ```
/// where the support is the integer interval [low, high) (upper-exclusive).
///
/// ## Parameters
/// - `k`: Integer values to evaluate for probability mass
/// - `low`: Lower bound of support (inclusive)
/// - `high`: Upper bound of support (exclusive, must satisfy `high > low`)
/// - `null_mask`: Optional bitmask for null value handling
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing PMF values, or error for invalid parameters.
#[inline(always)]
pub fn discrete_uniform_pmf_simd(
    k: &[i64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    discrete_uniform_pmf_simd_to(k, low, high, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Discrete uniform CDF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn discrete_uniform_cdf_simd_to(
    k: &[i64],
    low: i64,
    high: i64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if low >= high {
        return Err(KernelError::InvalidArguments(
            "discrete_uniform_cdf: require low < high (upper exclusive)".into(),
        ));
    }

    // span = N = high - low
    let span = high.checked_sub(low).ok_or_else(|| {
        KernelError::InvalidArguments("discrete_uniform_cdf: range overflow".into())
    })?;
    let inv_n = 1.0 / (span as f64);

    let len = k.len();
    if len == 0 {
        return Ok(());
    }

    const N: usize = W64;
    let low_v = Simd::<i64, N>::splat(low);
    let highm1_v = Simd::<i64, N>::splat(high - 1);
    let one_i = Simd::<i64, N>::splat(1);
    let zero_f = Simd::<f64, N>::splat(0.0);
    let one_f = Simd::<f64, N>::splat(1.0);
    let invn_v = Simd::<f64, N>::splat(inv_n);

    if !has_nulls(null_count, null_mask) {
        let mut i = 0;
        while i + N <= len {
            let kv = Simd::<i64, N>::from_slice(&k[i..i + N]);

            let below = kv.simd_lt(low_v);
            let at_or_above_max = kv.simd_ge(highm1_v); // ≥ high-1 ⇒ CDF = 1

            let delta = (kv - low_v + one_i).cast::<f64>(); // valid in middle region
            let mid = delta * invn_v;

            let tmp = below.select(zero_f, mid);
            let y = at_or_above_max.select(one_f, tmp);

            output[i..i + N].copy_from_slice(&y.to_array());
            i += N;
        }
        for (idx, &ki) in k[i..].iter().enumerate() {
            output[i + idx] = if ki < low {
                0.0
            } else if ki >= high - 1 {
                1.0
            } else {
                ((ki - low + 1) as f64) * inv_n
            };
        }
        return Ok(());
    }

    let mask_ref = null_mask.expect("discrete_uniform_cdf: null_count > 0 requires null_mask");

    let mask_bytes = mask_ref.as_bytes();
    let mut i = 0;
    while i + N <= len {
        let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);

        let mut tmpi = [0i64; N];
        for j in 0..N {
            tmpi[j] = k[i + j];
        }
        let kv: Simd<i64, N> = Simd::from_array(tmpi);

        let below = kv.simd_lt(low_v);
        let at_or_above_max = kv.simd_ge(highm1_v);

        let delta = (kv - low_v + one_i).cast::<f64>();
        let mid = delta * invn_v;

        let tmp = below.select(zero_f, mid);
        let y = at_or_above_max.select(one_f, tmp);

        let res = lane_mask.select(y, Simd::<f64, N>::splat(f64::NAN));
        output[i..i + N].copy_from_slice(res.as_array());
        i += N;
    }
    for idx in i..len {
        if !unsafe { mask_ref.get_unchecked(idx) } {
            output[idx] = f64::NAN;
        } else {
            let ki = k[idx];
            output[idx] = if ki < low {
                0.0
            } else if ki >= high - 1 {
                1.0
            } else {
                ((ki - low + 1) as f64) * inv_n
            };
        }
    }

    Ok(())
}

/// Discrete uniform CDF (lower tail, inclusive; upper-exclusive support).
/// F(k) = 0 for k < low; F(k) = 1 for k ≥ high−1; else F(k) = (k − low + 1)/(high − low).
#[inline(always)]
pub fn discrete_uniform_cdf_simd(
    k: &[i64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = k.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    discrete_uniform_cdf_simd_to(k, low, high, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Discrete uniform quantile SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn discrete_uniform_quantile_simd_to(
    p: &[f64],
    low: i64,
    high: i64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if low >= high {
        return Err(KernelError::InvalidArguments(
            "discrete_uniform_quantile: require low < high (upper exclusive)".into(),
        ));
    }

    // span = N
    let span = high.checked_sub(low).ok_or_else(|| {
        KernelError::InvalidArguments("discrete_uniform_quantile: range overflow".into())
    })?;
    let n_f = span as f64;

    let len = p.len();
    if len == 0 {
        return Ok(());
    }

    const NLANES: usize = W64;
    let zero_f = Simd::<f64, NLANES>::splat(0.0);
    let one_f = Simd::<f64, NLANES>::splat(1.0);
    let low_i = Simd::<i64, NLANES>::splat(low);
    let n_v = Simd::<f64, NLANES>::splat(n_f);
    let one_i = Simd::<i64, NLANES>::splat(1);
    let neg1_i = Simd::<i64, NLANES>::splat(-1);
    let n_1_i = Simd::<i64, NLANES>::splat((span as i64) - 1);

    if !has_nulls(null_count, null_mask) {
        let mut i = 0;
        while i + NLANES <= len {
            let pv = Simd::<f64, NLANES>::from_slice(&p[i..i + NLANES]);
            let pc = pv.simd_clamp(zero_f, one_f);
            let t = pc * n_v; // in [0, N]
            let tfi = t.cast::<i64>();
            let tff = tfi.cast::<f64>();
            let frac_nonzero = t.simd_gt(tff);
            let ceil_i = tfi + frac_nonzero.select(one_i, Simd::splat(0));
            let idx = (ceil_i - one_i).simd_clamp(neg1_i, n_1_i); // [-1, N-1]
            let k_i = low_i + idx; // low-1 .. high-1
            output[i..i + NLANES].copy_from_slice(&k_i.cast::<f64>().to_array());
            i += NLANES;
        }
        for (j, &pi) in p[i..].iter().enumerate() {
            if pi.is_nan() {
                output[i + j] = f64::NAN;
            } else {
                let pc = pi.clamp(0.0, 1.0);
                let t = (pc * n_f).ceil();
                let mut idx = (t as i64).saturating_sub(1);
                let n_minus1 = (span as i64).saturating_sub(1);
                if idx < -1 {
                    idx = -1;
                }
                if idx > n_minus1 {
                    idx = n_minus1;
                }
                let k = (low as i64).saturating_add(idx);
                output[i + j] = k as f64;
            }
        }
        return Ok(());
    }

    let mask_ref = null_mask.expect("discrete_uniform_quantile: null_count > 0 requires null_mask");

    let mask_bytes = mask_ref.as_bytes();
    let mut i = 0;
    while i + NLANES <= len {
        let lane_mask: Mask<i64, NLANES> = bitmask_to_simd_mask::<NLANES, i64>(mask_bytes, i, len);

        let pv = Simd::<f64, NLANES>::from_slice(&p[i..i + NLANES]);
        let pc = pv.simd_clamp(zero_f, one_f);
        let t = pc * n_v;
        let tfi = t.cast::<i64>();
        let tff = tfi.cast::<f64>();
        let frac_nonzero = t.simd_gt(tff);
        let ceil_i = tfi + frac_nonzero.select(one_i, Simd::splat(0));
        let idx = (ceil_i - one_i).simd_clamp(neg1_i, n_1_i);
        let k_i = low_i + idx;

        let res = lane_mask.select(k_i.cast::<f64>(), Simd::<f64, NLANES>::splat(f64::NAN));
        output[i..i + NLANES].copy_from_slice(res.as_array());
        i += NLANES;
    }
    for idx in i..len {
        if !unsafe { mask_ref.get_unchecked(idx) } {
            output[idx] = f64::NAN;
        } else {
            let pi = p[idx];
            if pi.is_nan() {
                output[idx] = f64::NAN;
            } else {
                let pc = pi.clamp(0.0, 1.0);
                let t = (pc * n_f).ceil();
                let mut id = (t as i64).saturating_sub(1);
                let n_minus1 = (span as i64).saturating_sub(1);
                if id < -1 {
                    id = -1;
                }
                if id > n_minus1 {
                    id = n_minus1;
                }
                let k = (low as i64).saturating_add(id);
                output[idx] = k as f64;
            }
        }
    }

    Ok(())
}

/// Discrete uniform quantile (lower tail; upper-exclusive support).
/// Support = {low, …, high−1}, N = high−low.
/// Q(p) = low + clamp(ceil(p*N)−1, −1, N−1).
/// p is clamped to [0,1]; NaN -> NaN (valid).
#[inline(always)]
pub fn discrete_uniform_quantile_simd(
    p: &[f64],
    low: i64,
    high: i64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = p.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    discrete_uniform_quantile_simd_to(p, low, high, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
