// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! SIMD-accelerated implementation of geometric distribution functions.
//!
//! This module provides vectorised implementations of the geometric distribution's probability
//! mass function (PMF), cumulative distribution function (CDF), and quantile function using
//! SIMD instructions. The implementation automatically falls back to scalar computation for
//! unaligned data or when SIMD is not available.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

const N: usize = W64;

use std::simd::{
    Mask, Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::{SimdFloat, SimdUint},
};

use minarrow::{Bitmask, FloatArray, Vec64, enums::error::KernelError, utils::is_simd_aligned};

use crate::utils::{bitmask_to_simd_mask, simd_mask_to_bitmask};
use crate::utils::{has_nulls, write_global_bitmask_block};

/// Geometric PMF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_pmf_simd_to(
    k: &[u64],
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    // 1) Parameter check (allow p == 1.0 for the degenerate case)
    if !(p > 0.0 && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "geometric_pmf: p must be in (0,1] and finite".into(),
        ));
    }
    // 2) Empty input
    let len = k.len();
    if len == 0 {
        return Ok(());
    }

    // ----- Degenerate fast path: p == 1  ->  PMF(k) = 1_{k==1} -----
    if p == 1.0 {
        if !has_nulls(null_count, null_mask) {
            if is_simd_aligned(k) {
                let one = Simd::<u64, N>::splat(1);
                let mut i = 0;
                while i + N <= len {
                    let ku = Simd::<u64, N>::from_slice(&k[i..i + N]);
                    let is_one = ku.simd_eq(one);
                    let vals =
                        is_one.select(Simd::<f64, N>::splat(1.0), Simd::<f64, N>::splat(0.0));
                    output[i..i + N].copy_from_slice(vals.as_array());
                    i += N;
                }
                for idx in i..len {
                    output[idx] = if k[idx] == 1 { 1.0 } else { 0.0 };
                }
            } else {
                for (i, &ki) in k.iter().enumerate() {
                    output[i] = if ki == 1 { 1.0 } else { 0.0 };
                }
            }
            return Ok(());
        }

        // masked path
        let mask = null_mask.expect("null_count > 0 requires null_mask");
        let mut out_mask = mask.clone();
        if is_simd_aligned(k) {
            let mask_bytes = mask.as_bytes();
            let one = Simd::<u64, N>::splat(1);
            let mut i = 0;
            while i + N <= len {
                let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);
                let ku = Simd::<u64, N>::from_slice(&k[i..i + N]);
                let masked_ku = lane_mask.select(ku, Simd::<u64, N>::splat(0));
                let is_one = masked_ku.simd_eq(one);
                let vals = is_one.select(Simd::<f64, N>::splat(1.0), Simd::<f64, N>::splat(0.0));
                let res = lane_mask.select(vals, Simd::<f64, N>::splat(f64::NAN));
                output[i..i + N].copy_from_slice(res.as_array());

                let bits = simd_mask_to_bitmask::<N, i64>(lane_mask, N);
                write_global_bitmask_block(&mut out_mask, &bits, i, N);
                i += N;
            }
            for idx in i..len {
                if !unsafe { mask.get_unchecked(idx) } {
                    output[idx] = f64::NAN;
                    unsafe { out_mask.set_unchecked(idx, false) }
                } else {
                    output[idx] = if k[idx] == 1 { 1.0 } else { 0.0 };
                    unsafe { out_mask.set_unchecked(idx, true) }
                }
            }
        } else {
            for idx in 0..len {
                if !mask.get(idx) {
                    output[idx] = f64::NAN;
                    unsafe { out_mask.set_unchecked(idx, false) }
                } else {
                    output[idx] = if k[idx] == 1 { 1.0 } else { 0.0 };
                    unsafe { out_mask.set_unchecked(idx, true) }
                }
            }
        }
        return Ok(());
    }

    // ----- Regular path: 0 < p < 1 -----
    let log1mp = (1.0 - p).ln();
    let p_v = Simd::<f64, N>::splat(p);
    let log1mp_v = Simd::<f64, N>::splat(log1mp);

    let scalar_body = |ki: u64| -> f64 {
        if ki == 0 {
            0.0
        } else {
            (((ki as f64) - 1.0) * log1mp).exp() * p
        }
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(k) {
            let zero_u = Simd::<u64, N>::splat(0);
            let one_f = Simd::<f64, N>::splat(1.0);
            let zero_f = Simd::<f64, N>::splat(0.0);
            let mut i = 0;
            while i + N <= len {
                let k_u = Simd::<u64, N>::from_slice(&k[i..i + N]);
                let is_zero = k_u.simd_eq(zero_u);
                let kf = k_u.cast::<f64>();
                let pmf_nonzero = ((kf - one_f) * log1mp_v).exp() * p_v;
                let pmf = is_zero.select(zero_f, pmf_nonzero);
                output[i..i + N].copy_from_slice(pmf.as_array());
                i += N;
            }
            for idx in i..len {
                output[idx] = scalar_body(k[idx]);
            }
            return Ok(());
        }
        for (i, &ki) in k.iter().enumerate() {
            output[i] = scalar_body(ki);
        }
        return Ok(());
    }

    // masked
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();

    if is_simd_aligned(k) {
        let one_f = Simd::<f64, N>::splat(1.0);
        let zero_f = Simd::<f64, N>::splat(0.0);
        let mask_bytes = mask.as_bytes();
        let mut i = 0;
        while i + N <= len {
            let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);
            let mut k_arr = [0u64; N];
            for j in 0..N {
                k_arr[j] = if unsafe { lane_mask.test_unchecked(j) } {
                    k[i + j]
                } else {
                    0
                };
            }
            let k_u = Simd::<u64, N>::from_array(k_arr);
            let is_zero = k_u.simd_eq(Simd::<u64, N>::splat(0));
            let kf = k_u.cast::<f64>();
            let pmf_nonzero = ((kf - one_f) * log1mp_v).exp() * p_v;
            let pmf = is_zero.select(zero_f, pmf_nonzero);
            let res_v = lane_mask.select(pmf, Simd::<f64, N>::splat(f64::NAN));
            output[i..i + N].copy_from_slice(res_v.as_array());

            let bits = simd_mask_to_bitmask::<N, i64>(lane_mask, N);
            write_global_bitmask_block(&mut out_mask, &bits, i, N);
            i += N;
        }
        for idx in i..len {
            if !unsafe { mask.get_unchecked(idx) } {
                output[idx] = f64::NAN;
                unsafe { out_mask.set_unchecked(idx, false) }
            } else {
                output[idx] = scalar_body(k[idx]);
                unsafe { out_mask.set_unchecked(idx, true) }
            }
        }
        return Ok(());
    }

    for idx in 0..len {
        if !mask.get(idx) {
            output[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) }
        } else {
            output[idx] = scalar_body(k[idx]);
            unsafe { out_mask.set_unchecked(idx, true) }
        }
    }
    Ok(())
}

/// SIMD-accelerated implementation of geometric distribution probability mass function.
///
/// Computes the probability mass function (PMF) of the geometric distribution following
/// SciPy's convention using vectorised SIMD operations for enhanced performance on
/// discrete probability computations.
///
/// ## Parameters
/// - `k`: Input trial numbers where PMF should be evaluated (domain: k ∈ {0, 1, 2, ...})
/// - `p`: Success probability parameter ∈ (0, 1] controlling distribution
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed PMF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid probability parameter
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When p ∉ (0, 1] or p is non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// let k = [0, 1, 2, 3, 4];  // trial numbers
/// let p = 0.3;             // success probability
/// let result = geometric_pmf_simd(&k, p, None, None)?;
/// ```
#[inline(always)]
pub fn geometric_pmf_simd(
    k: &[u64],
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

    geometric_pmf_simd_to(k, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Geometric CDF SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_cdf_simd_to(
    k: &[u64],
    p: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(p > 0.0 && p <= 1.0) || !p.is_finite() {
        return Err(KernelError::InvalidArguments(
            "geometric_cdf: p must be in (0,1] and finite".into(),
        ));
    }
    let len = k.len();
    if len == 0 {
        return Ok(());
    }

    // Degenerate p == 1 -> CDF(k) = 0 if k==0 else 1
    if p == 1.0 {
        if !has_nulls(null_count, null_mask) {
            if is_simd_aligned(k) {
                let zero_u = Simd::<u64, N>::splat(0);
                let mut i = 0;
                while i + N <= len {
                    let ku = Simd::<u64, N>::from_slice(&k[i..i + N]);
                    let is_zero = ku.simd_eq(zero_u);
                    let vals =
                        is_zero.select(Simd::<f64, N>::splat(0.0), Simd::<f64, N>::splat(1.0));
                    output[i..i + N].copy_from_slice(vals.as_array());
                    i += N;
                }
                for idx in i..len {
                    output[idx] = if k[idx] == 0 { 0.0 } else { 1.0 };
                }
            } else {
                for (i, &ki) in k.iter().enumerate() {
                    output[i] = if ki == 0 { 0.0 } else { 1.0 };
                }
            }
            return Ok(());
        }

        // masked
        let mask = null_mask.expect("null_count > 0 requires null_mask");
        let mut out_mask = mask.clone();
        if is_simd_aligned(k) {
            let mask_bytes = mask.as_bytes();
            let zero_u = Simd::<u64, N>::splat(0);
            let mut i = 0;
            while i + N <= len {
                let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);
                let ku = Simd::<u64, N>::from_slice(&k[i..i + N]);
                let masked_ku = lane_mask.select(ku, Simd::<u64, N>::splat(0));
                let is_zero = masked_ku.simd_eq(zero_u);
                let vals = is_zero.select(Simd::<f64, N>::splat(0.0), Simd::<f64, N>::splat(1.0));
                let res = lane_mask.select(vals, Simd::<f64, N>::splat(f64::NAN));
                output[i..i + N].copy_from_slice(res.as_array());

                let bits = simd_mask_to_bitmask::<N, i64>(lane_mask, N);
                write_global_bitmask_block(&mut out_mask, &bits, i, N);
                i += N;
            }
            for idx in i..len {
                if !unsafe { mask.get_unchecked(idx) } {
                    output[idx] = f64::NAN;
                    unsafe { out_mask.set_unchecked(idx, false) };
                } else {
                    output[idx] = if k[idx] == 0 { 0.0 } else { 1.0 };
                    unsafe { out_mask.set_unchecked(idx, true) };
                }
            }
        } else {
            for idx in 0..len {
                if !mask.get(idx) {
                    output[idx] = f64::NAN;
                    unsafe { out_mask.set_unchecked(idx, false) };
                } else {
                    output[idx] = if k[idx] == 0 { 0.0 } else { 1.0 };
                    unsafe { out_mask.set_unchecked(idx, true) };
                }
            }
        }
        return Ok(());
    }

    // 0 < p < 1
    let log1mp = (1.0 - p).ln();

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(k) {
            let log1mp_v = Simd::<f64, N>::splat(log1mp);
            let one_v = Simd::<f64, N>::splat(1.0);
            let zero_u = Simd::<u64, N>::splat(0);
            let mut i = 0;
            while i + N <= len {
                let k_u = Simd::<u64, N>::from_slice(&k[i..i + N]);
                let kf = k_u.cast::<f64>();
                let is_zero = k_u.simd_eq(zero_u);
                let cdf_nonzero = one_v - (kf * log1mp_v).exp();
                let cdf_v = is_zero.select(Simd::<f64, N>::splat(0.0), cdf_nonzero);
                output[i..i + N].copy_from_slice(cdf_v.as_array());
                i += N;
            }
            for idx in i..len {
                let ki = k[idx];
                output[idx] = if ki == 0 {
                    0.0
                } else {
                    1.0 - ((ki as f64) * log1mp).exp()
                };
            }
            return Ok(());
        }
        for (i, &ki) in k.iter().enumerate() {
            output[i] = if ki == 0 {
                0.0
            } else {
                1.0 - ((ki as f64) * log1mp).exp()
            };
        }
        return Ok(());
    }

    // masked path
    let mask = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask.clone();

    if is_simd_aligned(k) {
        let log1mp_v = Simd::<f64, N>::splat(log1mp);
        let one_v = Simd::<f64, N>::splat(1.0);
        let mask_bytes = mask.as_bytes();
        let zero_u = Simd::<u64, N>::splat(0);

        let mut i = 0;
        while i + N <= len {
            let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);
            let mut k_arr = [0u64; N];
            for j in 0..N {
                let idx = i + j;
                k_arr[j] = if unsafe { lane_mask.test_unchecked(j) } {
                    k[idx]
                } else {
                    0
                };
            }
            let k_u = Simd::<u64, N>::from_array(k_arr);
            let kf = k_u.cast::<f64>();
            let is_zero = k_u.simd_eq(zero_u);
            let cdf_nonzero = one_v - (kf * log1mp_v).exp();
            let cdf_v = is_zero.select(Simd::<f64, N>::splat(0.0), cdf_nonzero);

            let result = lane_mask.select(cdf_v, Simd::<f64, N>::splat(f64::NAN));
            output[i..i + N].copy_from_slice(result.as_array());

            let bits = simd_mask_to_bitmask::<N, i64>(lane_mask, N);
            write_global_bitmask_block(&mut out_mask, &bits, i, N);

            i += N;
        }
        for idx in i..len {
            if !unsafe { mask.get_unchecked(idx) } {
                output[idx] = f64::NAN;
                unsafe { out_mask.set_unchecked(idx, false) };
            } else {
                let ki = k[idx];
                output[idx] = if ki == 0 {
                    0.0
                } else {
                    1.0 - ((ki as f64) * log1mp).exp()
                };
                unsafe { out_mask.set_unchecked(idx, true) };
            }
        }
        return Ok(());
    }

    for idx in 0..len {
        if !mask.get(idx) {
            output[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let ki = k[idx];
            output[idx] = if ki == 0 {
                0.0
            } else {
                1.0 - ((ki as f64) * log1mp).exp()
            };
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }

    Ok(())
}

/// SIMD-accelerated implementation of geometric distribution cumulative distribution function.
///
/// Computes the cumulative distribution function (CDF) of the geometric distribution
/// following SciPy's convention using vectorised SIMD operations for enhanced
/// performance on discrete probability computations.
///
/// ## Parameters
/// - `k`: Input trial numbers where CDF should be evaluated (domain: k ∈ {0, 1, 2, ...})
/// - `p`: Success probability parameter ∈ (0, 1] controlling distribution
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed CDF values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid probability parameter
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When p ∉ (0, 1] or p is non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// let k = [0, 1, 2, 3, 4];  // trial numbers
/// let p = 0.3;             // success probability
/// let result = geometric_cdf_simd(&k, p, None, None)?;
/// ```
#[inline(always)]
pub fn geometric_cdf_simd(
    k: &[u64],
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

    geometric_cdf_simd_to(k, p, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}

/// Geometric quantile SIMD (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn geometric_quantile_simd_to(
    pv: &[f64],
    p_succ: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if !(p_succ > 0.0 && p_succ.is_finite() && p_succ <= 1.0) {
        return Err(KernelError::InvalidArguments(
            "geometric_quantile: p_succ must be in (0,1] and finite".into(),
        ));
    }
    let len = pv.len();
    if len == 0 {
        return Ok(());
    }

    // scalar kernel for tails & unaligned
    let scalar_body = |q: f64| -> f64 {
        if !q.is_finite() || q < 0.0 || q > 1.0 {
            return f64::NAN;
        }
        if p_succ == 1.0 {
            return if q == 1.0 { f64::INFINITY } else { 1.0 };
        }
        if q == 0.0 {
            return 1.0;
        }
        if q == 1.0 {
            return f64::INFINITY;
        }
        let log1mp = (1.0 - p_succ).ln(); // < 0
        ((1.0 - q).ln() / log1mp).ceil()
    };

    // dense path - no nulls
    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(pv) {
            let one = Simd::<f64, N>::splat(1.0);
            let zero = Simd::<f64, N>::splat(0.0);

            // constants depending on p_succ
            let log1mp_v = if p_succ == 1.0 {
                Simd::<f64, N>::splat(0.0) // unused
            } else {
                Simd::<f64, N>::splat((1.0 - p_succ).ln())
            };

            let mut i = 0;
            while i + N <= len {
                let qv = Simd::<f64, N>::from_slice(&pv[i..i + N]);

                let oob = qv.simd_lt(zero) | qv.simd_gt(one) | !qv.is_finite();
                let at0 = qv.simd_eq(zero);
                let at1 = qv.simd_eq(one);

                // base result
                let mut rv = if p_succ == 1.0 {
                    // degenerate-at-1 - q<1 -> 1, q==1 -> +∞
                    let ones = Simd::<f64, N>::splat(1.0);
                    let infs = Simd::<f64, N>::splat(f64::INFINITY);
                    at1.select(infs, ones)
                } else {
                    // regular - ceil(ln(1-q)/ln(1-p))
                    let base = ((one - qv).ln() / log1mp_v).ceil();
                    // apply edges
                    let with0 = at0.select(Simd::<f64, N>::splat(1.0), base);
                    at1.select(Simd::<f64, N>::splat(f64::INFINITY), with0)
                };

                // mask invalid domain to NaN
                rv = oob.select(Simd::<f64, N>::splat(f64::NAN), rv);

                output[i..i + N].copy_from_slice(rv.as_array());
                i += N;
            }
            // scalar tail
            for idx in i..len {
                output[idx] = scalar_body(pv[idx]);
            }
            return Ok(());
        }

        // unaligned -> scalar
        for (i, &q) in pv.iter().enumerate() {
            output[i] = scalar_body(q);
        }
        return Ok(());
    }

    // masked path
    let mask_ref = null_mask.expect("null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();

    if is_simd_aligned(pv) {
        let one = Simd::<f64, N>::splat(1.0);
        let zero = Simd::<f64, N>::splat(0.0);
        let mask_bytes = mask_ref.as_bytes();
        let log1mp_v = if p_succ == 1.0 {
            Simd::<f64, N>::splat(0.0) // unused when degenerate
        } else {
            Simd::<f64, N>::splat((1.0 - p_succ).ln())
        };

        let mut i = 0;
        while i + N <= len {
            // lane validity from input null mask
            let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);

            // gather q (value unused when lane is null)
            let mut buf = [0.0f64; N];
            for j in 0..N {
                buf[j] = if unsafe { lane_mask.test_unchecked(j) } {
                    pv[i + j]
                } else {
                    0.0
                };
            }
            let qv = Simd::<f64, N>::from_array(buf);

            let is_nan = qv.simd_ne(qv);
            let oob = qv.simd_lt(zero) | qv.simd_gt(one) | !qv.is_finite();
            let at0 = qv.simd_eq(zero);
            let at1 = qv.simd_eq(one);

            // compute result (without null overlay yet)
            let mut rv = if p_succ == 1.0 {
                let ones = Simd::<f64, N>::splat(1.0);
                let infs = Simd::<f64, N>::splat(f64::INFINITY);
                at1.select(infs, ones)
            } else {
                let base = ((one - qv).ln() / log1mp_v).ceil();
                let with0 = at0.select(Simd::<f64, N>::splat(1.0), base);
                at1.select(Simd::<f64, N>::splat(f64::INFINITY), with0)
            };

            // domain invalid -> NaN
            let invalid = is_nan | oob;
            rv = invalid.select(Simd::<f64, N>::splat(f64::NAN), rv);

            // apply null mask: null lanes -> NaN, propagate mask unchanged
            let res_v = lane_mask.select(rv, Simd::<f64, N>::splat(f64::NAN));
            output[i..i + N].copy_from_slice(res_v.as_array());

            // keep the input null mask bits verbatim
            let bits = simd_mask_to_bitmask::<N, i64>(lane_mask, N);
            write_global_bitmask_block(&mut out_mask, &bits, i, N);

            i += N;
        }
        // scalar tail
        for idx in i..len {
            if !unsafe { mask_ref.get_unchecked(idx) } {
                output[idx] = f64::NAN;
                unsafe { out_mask.set_unchecked(idx, false) };
            } else {
                output[idx] = scalar_body(pv[idx]);
                unsafe { out_mask.set_unchecked(idx, true) };
            }
        }

        return Ok(());
    }

    // unaligned masked -> scalar per element
    for idx in 0..len {
        if !mask_ref.get(idx) {
            output[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            output[idx] = scalar_body(pv[idx]);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }

    Ok(())
}

/// SIMD-accelerated implementation of geometric distribution quantile function.
///
/// Computes the quantile function (inverse CDF) of the geometric distribution
/// following SciPy's convention using vectorised SIMD operations for enhanced
/// performance on inverse probability computations.
///
/// ## Parameters
/// - `pv`: Input probability values where quantiles should be evaluated (domain: q ∈ [0, 1])
/// - `p_succ`: Success probability parameter ∈ (0, 1] controlling distribution
/// - `null_mask`: Optional bitmask indicating null values in input
/// - `null_count`: Optional count of null values for optimisation
///
/// ## Returns
/// `Result<FloatArray<f64>, KernelError>` containing:
/// - **Success**: FloatArray with computed quantile values and appropriate null mask
/// - **Error**: KernelError::InvalidArguments for invalid success probability parameter
///
/// ## Special Cases and Boundary Conditions
/// - **q = 0**: Returns 1 (minimum trials for any success probability)
/// - **q = 1**: Returns +∞ (infinite trials needed for certainty)
/// - **p = 1.0**: Degenerate distribution (Q(q<1) = 1, Q(1) = +∞)
/// - **Invalid q**: Returns NaN for q ∉ [0, 1] or non-finite q
/// - **Invalid parameters**: Returns error for p ∉ (0, 1] or non-finite p
///
/// ## Errors
/// - `KernelError::InvalidArguments`: When p_succ ∉ (0, 1] or p_succ is non-finite
///
/// ## Example Usage
/// ```rust,ignore
/// let q = [0.0, 0.25, 0.5, 0.75, 1.0];  // probability values
/// let p_succ = 0.3;                      // success probability
/// let result = geometric_quantile_simd(&q, p_succ, None, None)?;
/// // Returns quantiles: [1.0, 1.0, 2.0, 5.0, ∞]
/// ```
#[inline(always)]
pub fn geometric_quantile_simd(
    pv: &[f64],
    p_succ: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let len = pv.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    geometric_quantile_simd_to(pv, p_succ, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
