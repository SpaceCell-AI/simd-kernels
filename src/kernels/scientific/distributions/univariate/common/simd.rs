// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Common SIMD Distribution Utilities**
//!
//! Shared SIMD kernel infrastructure for univariate statistical distribution implementations.
//! Provides high-performance, reusable compute kernels that abstract the complexities
//! of SIMD vectorisation, memory alignment, and null value handling.

use std::simd::{Mask, Select, Simd};

use minarrow::{Bitmask, Vec64, utils::is_simd_aligned};

use crate::utils::bitmask_to_simd_mask;

/// High-performance SIMD kernel for dense f64->f64 univariate distribution computations
/// (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// ## Parameters
/// - `x`: Input array slice (requires 64-byte alignment for SIMD activation)
/// - `out`: Output buffer (must match input length)
/// - `simd_body`: Vectorised computation function: `Simd<f64, N> -> Simd<f64, N>`
/// - `scalar_body`: Scalar computation function for tail elements: `f64 -> f64`
///
/// ## Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn dense_univariate_kernel_f64_simd_to<const N: usize, FSimd, FScalar>(
    x: &[f64],
    out: &mut [f64],
    simd_body: FSimd,
    scalar_body: FScalar,
) where
    FSimd: Fn(Simd<f64, N>) -> Simd<f64, N>,
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    assert_eq!(
        len,
        out.len(),
        "dense_univariate_kernel_f64_simd_to: input/output length mismatch"
    );

    // Check if input array is 64-byte aligned for SIMD
    if is_simd_aligned(x) {
        let mut i = 0;
        while i + N <= len {
            let x_v = Simd::<f64, N>::from_slice(&x[i..i + N]);
            let y_v = simd_body(x_v);
            out[i..i + N].copy_from_slice(y_v.as_array());
            i += N;
        }
        // Scalar tail
        for j in i..len {
            out[j] = scalar_body(x[j]);
        }
        return;
    }

    // Scalar fallback - alignment check failed
    for (i, &xi) in x.iter().enumerate() {
        out[i] = scalar_body(xi);
    }
}

/// High-performance SIMD kernel for dense f64->f64 univariate distribution computations.
///
/// Processes arrays without null values using vectorised operations for maximum performance.
/// This kernel provides the fastest execution path when input data is guaranteed to be complete.
///
/// ## Parameters
/// - `x`: Input array slice (requires 64-byte alignment for SIMD activation)
/// - `has_mask`: Whether to return an all-true validity mask for consistency
/// - `simd_body`: Vectorised computation function: `Simd<f64, N> -> Simd<f64, N>`
/// - `scalar_body`: Scalar computation function for tail elements: `f64 -> f64`
///
/// ## Returns
/// `(Vec64<f64>, Option<Bitmask>)` containing computed results and optional validity mask.
#[inline(always)]
pub fn dense_univariate_kernel_f64_simd<const N: usize, FSimd, FScalar>(
    x: &[f64],
    has_mask: bool,
    simd_body: FSimd,
    scalar_body: FScalar,
) -> (Vec64<f64>, Option<Bitmask>)
where
    FSimd: Fn(Simd<f64, N>) -> Simd<f64, N>,
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    dense_univariate_kernel_f64_simd_to::<N, _, _>(x, out.as_mut_slice(), simd_body, scalar_body);

    let out_mask = if has_mask {
        Some(Bitmask::new_set_all(len, true))
    } else {
        None
    };
    (out, out_mask)
}

/// High-performance SIMD kernel for null-aware f64->f64 univariate distribution computations
/// (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer and mask.
///
/// ## Parameters
/// - `x`: Input array slice (requires 64-byte alignment for SIMD activation)
/// - `mask`: Arrow bitmask defining valid/null elements (required)
/// - `out`: Output buffer (must match input length)
/// - `out_mask`: Output mask (must be pre-initialised, typically cloned from input mask)
/// - `simd_body`: Vectorised computation function: `Simd<f64, N> -> Simd<f64, N>`
/// - `scalar_body`: Scalar computation function: `f64 -> f64`
///
/// ## Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn masked_univariate_kernel_f64_simd_to<const N: usize, FSimd, FScalar>(
    x: &[f64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
    simd_body: FSimd,
    scalar_body: FScalar,
) where
    FSimd: Fn(Simd<f64, N>) -> Simd<f64, N>,
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    assert_eq!(
        len,
        out.len(),
        "masked_univariate_kernel_f64_simd_to: input/output length mismatch"
    );
    let mask_bytes = mask.as_bytes();

    // Check if input array is 64-byte aligned for SIMD
    if is_simd_aligned(x) {
        let mut i = 0;
        while i + N <= len {
            // Load SIMD mask
            let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);

            // Load inputs
            let mut x_arr = [0.0f64; N];
            for j in 0..N {
                x_arr[j] = unsafe { *x.get_unchecked(i + j) };
            }
            let x_v_raw = Simd::<f64, N>::from_array(x_arr);

            // Replace null lanes with NaN
            let nan_v = Simd::<f64, N>::splat(f64::NAN);
            let x_v_in = lane_mask.select(x_v_raw, nan_v);

            // SIMD kernel
            let y_v = simd_body(x_v_in);
            out[i..i + N].copy_from_slice(y_v.as_array());

            i += N;
        }

        // Scalar tail
        for idx in i..len {
            if !unsafe { mask.get_unchecked(idx) } {
                out[idx] = f64::NAN;
                unsafe { out_mask.set_unchecked(idx, false) };
            } else {
                let xi = unsafe { *x.get_unchecked(idx) };
                out[idx] = scalar_body(xi);
                unsafe { out_mask.set_unchecked(idx, true) };
            }
        }
        return;
    }

    // Scalar fallback - alignment check failed
    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let xi = unsafe { *x.get_unchecked(idx) };
            out[idx] = scalar_body(xi);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}

/// High-performance SIMD kernel for null-aware f64->f64 univariate distribution computations.
///
/// Processes arrays with null values using vectorised operations combined with efficient
/// bitmask-based null propagation.
///
/// ## SIMD Processing Strategy
/// - **Alignment Check**: Verifies 64-byte memory alignment for optimal SIMD performance
/// - **Bitmask Conversion**: Efficiently converts Arrow bitmasks to SIMD lane masks
/// - **Null Lane Injection**: Replaces null input lanes with NaN for safe SIMD processing
/// - **Result Masking**: Propagates null status through SIMD conditional operations
/// - **Scalar Tail**: Handles remaining elements using optimised scalar null-aware processing
///
/// ## Parameters
/// - `x`: Input array slice (requires 64-byte alignment for SIMD activation)
/// - `mask`: Arrow bitmask defining valid/null elements (required)
/// - `simd_body`: Vectorised computation function: `Simd<f64, N> -> Simd<f64, N>`
/// - `scalar_body`: Scalar computation function: `f64 -> f64`
///
/// ## Returns
/// `(Vec64<f64>, Bitmask)` containing computed results and propagated null mask.
#[inline(always)]
pub fn masked_univariate_kernel_f64_simd<const N: usize, FSimd, FScalar>(
    x: &[f64],
    mask: &Bitmask,
    simd_body: FSimd,
    scalar_body: FScalar,
) -> (Vec64<f64>, Bitmask)
where
    FSimd: Fn(Simd<f64, N>) -> Simd<f64, N>,
    FScalar: Fn(f64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };
    let mut out_mask = mask.clone();

    masked_univariate_kernel_f64_simd_to::<N, _, _>(
        x,
        mask,
        out.as_mut_slice(),
        &mut out_mask,
        simd_body,
        scalar_body,
    );

    (out, out_mask)
}

/// High-performance SIMD kernel for dense u64->f64 univariate distribution computations
/// (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
///
/// ## Parameters
/// - `x`: Input u64 array slice (requires 64-byte alignment for SIMD activation)
/// - `out`: Output buffer (must match input length)
/// - `simd_body`: Vectorised computation function: `Simd<u64, N> -> Simd<f64, N>`
/// - `scalar_body`: Scalar computation function: `u64 -> f64`
///
/// ## Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn dense_univariate_kernel_u64_simd_to<const N: usize, FSimd, FScalar>(
    x: &[u64],
    out: &mut [f64],
    simd_body: FSimd,
    scalar_body: FScalar,
) where
    FSimd: Fn(Simd<u64, N>) -> Simd<f64, N>,
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    assert_eq!(
        len,
        out.len(),
        "dense_univariate_kernel_u64_simd_to: input/output length mismatch"
    );

    // Check if input array is 64-byte aligned for SIMD
    if is_simd_aligned(x) {
        let mut i = 0;
        while i + N <= len {
            // load u64 lanes
            let k_u = Simd::<u64, N>::from_slice(&x[i..i + N]);
            // run the SIMD body
            let y_v = simd_body(k_u);
            out[i..i + N].copy_from_slice(y_v.as_array());
            i += N;
        }
        // scalar tail
        for j in i..len {
            out[j] = scalar_body(x[j]);
        }
        return;
    }

    // Scalar fallback - alignment check failed
    for (i, &ki) in x.iter().enumerate() {
        out[i] = scalar_body(ki);
    }
}

/// High-performance SIMD kernel for dense u64->f64 univariate distribution computations.
///
/// It's for discrete distributions that process integer input values and produce
/// floating-point probability results.
///
/// Library functions *(and you can)* use this when you know that your data is dense, to
/// avoid null-mask related conditional checks in the hot loop. We still take a 'has_mask'
/// for when this is used in a micro-batching context, and it will then ensure the output
/// receives a mask back with all bits set to *true*.
///
/// ## SIMD Processing Strategy
/// - **Integer Loading**: Efficiently loads u64 values into SIMD registers
/// - **Type Conversion**: Handles u64->f64 casting within vectorised operations where needed
/// - **Alignment Optimisation**: Leverages 64-byte alignment for optimal memory throughput
/// - **Hybrid Processing**: SIMD for bulk operations, scalar for tail elements
///
/// ## Parameters
/// - `x`: Input u64 array slice (requires 64-byte alignment for SIMD activation)
/// - `has_mask`: Whether to return an all-true validity mask for consistency
/// - `simd_body`: Vectorised computation function: `Simd<u64, N> -> Simd<f64, N>`
/// - `scalar_body`: Scalar computation function: `u64 -> f64`
///
/// ## Returns
/// `(Vec64<f64>, Option<Bitmask>)` containing computed probability values and optional validity mask.
#[inline(always)]
pub fn dense_univariate_kernel_u64_simd<const N: usize, FSimd, FScalar>(
    x: &[u64],
    has_mask: bool,
    simd_body: FSimd,
    scalar_body: FScalar,
) -> (Vec64<f64>, Option<Bitmask>)
where
    FSimd: Fn(Simd<u64, N>) -> Simd<f64, N>,
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    dense_univariate_kernel_u64_simd_to::<N, _, _>(x, out.as_mut_slice(), simd_body, scalar_body);

    let out_mask = if has_mask {
        Some(Bitmask::new_set_all(len, true))
    } else {
        None
    };
    (out, out_mask)
}

/// Null‐aware masked kernel helper for u64->f64 kernels (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer and mask.
///
/// ## Panics
/// Panics if `x.len() != out.len()`.
#[inline(always)]
pub fn masked_univariate_kernel_u64_simd_to<const N: usize, FSimd, FScalar>(
    x: &[u64],
    mask: &Bitmask,
    out: &mut [f64],
    out_mask: &mut Bitmask,
    simd_body: FSimd,
    scalar_body: FScalar,
) where
    FSimd: Fn(Simd<u64, N>) -> Simd<f64, N>,
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    assert_eq!(
        len,
        out.len(),
        "masked_univariate_kernel_u64_simd_to: input/output length mismatch"
    );
    let mask_bytes = mask.as_bytes();

    if is_simd_aligned(x) {
        let mut i = 0;
        while i + N <= len {
            // 1) turn Arrow bitmask -> SIMD lane mask
            let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);

            // 2) gather inputs (null -> 0)
            let mut k_arr = [0u64; N];
            for j in 0..N {
                k_arr[j] = unsafe { *x.get_unchecked(i + j) };
            }
            let k_u = Simd::<u64, N>::from_array(k_arr);

            // Replace null lanes with 0
            let zero_v = Simd::<u64, N>::splat(0u64);
            let x_v_in = lane_mask.select(k_u, zero_v);

            // 3) run the SIMD body
            let y_v = simd_body(x_v_in);

            // 4) replace null lanes with NaN in the result
            let nan_v = Simd::<f64, N>::splat(f64::NAN);
            let result = lane_mask.select(y_v, nan_v);

            // 5) write results
            out[i..i + N].copy_from_slice(result.as_array());

            i += N;
        }

        // scalar tail
        for idx in i..len {
            if !unsafe { mask.get_unchecked(idx) } {
                out[idx] = f64::NAN;
                unsafe { out_mask.set_unchecked(idx, false) };
            } else {
                let ki = unsafe { *x.get_unchecked(idx) };
                out[idx] = scalar_body(ki);
                unsafe { out_mask.set_unchecked(idx, true) };
            }
        }
        return;
    }

    // Scalar fallback - alignment check failed
    for idx in 0..len {
        if !unsafe { mask.get_unchecked(idx) } {
            out[idx] = f64::NAN;
            unsafe { out_mask.set_unchecked(idx, false) };
        } else {
            let ki = unsafe { *x.get_unchecked(idx) };
            out[idx] = scalar_body(ki);
            unsafe { out_mask.set_unchecked(idx, true) };
        }
    }
}

/// Null‐aware masked kernel helper for u64->f64 kernels.
///
/// Propagates the input mask and leaves any "invalid" (null) lanes as NULL;
/// lanes that are in the mask but produce a non‐finite result in the scalar
/// or SIMD body also become NULL.
#[inline(always)]
pub fn masked_univariate_kernel_u64_simd<const N: usize, FSimd, FScalar>(
    x: &[u64],
    mask: &Bitmask,
    simd_body: FSimd,
    scalar_body: FScalar,
) -> (Vec64<f64>, Bitmask)
where
    FSimd: Fn(Simd<u64, N>) -> Simd<f64, N>,
    FScalar: Fn(u64) -> f64,
{
    let len = x.len();
    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };
    let mut out_mask = mask.clone();

    masked_univariate_kernel_u64_simd_to::<N, _, _>(
        x,
        mask,
        out.as_mut_slice(),
        &mut out_mask,
        simd_body,
        scalar_body,
    );

    (out, out_mask)
}
