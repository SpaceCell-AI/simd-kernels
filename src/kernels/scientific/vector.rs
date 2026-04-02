// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Vector Mathematics Module** - *High-Performance BLAS-Compatible Vector Operations*
//!
//! This module provides a streamlined set of vector operations without the need for BLAS/LAPACK linking.
//! It includes SIMD-accelerated arms where appropriate.
//!
//! ## Usage Examples
//!
//! ### Basic Linear Algebra
//! ```rust,ignore
//! use minarrow::vec64;
//! use crate::kernels::scientific::vector::{dot, scale, axpy, l2_norm};
//!
//! let x = vec64![1.0, 2.0, 3.0, 4.0];
//! let mut y = vec64![5.0, 6.0, 7.0, 8.0];
//!
//! // Dot product computation
//! let result = dot(&x, &y, None, None);
//!
//! // In-place vector scaling
//! scale(&mut y, 2.0, None, None);
//!
//! // Scaled addition: y ← 2.0 * x + y
//! axpy(4, 2.0, &x, 1, &mut y, 1, None, None)?;
//!
//! // Euclidean norm
//! let norm = l2_norm(&x, None, None);
//! ```

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

#[cfg(feature = "simd")]
use std::simd::{Mask, Select, Simd, num::SimdFloat};

use minarrow::{Bitmask, FloatArray, Vec64, vec64};

#[cfg(feature = "simd")]
use crate::kernels::aggregate::neumaier_simd_add;
use crate::kernels::aggregate::{neumaier_add, sum_squares};
use crate::utils::has_nulls;
use minarrow::enums::error::KernelError;

#[cfg(feature = "simd")]
use crate::utils::bitmask_to_simd_mask;

#[cfg(feature = "simd")]
use minarrow::utils::is_simd_aligned;

// --- Vector analytics ---

/// Computes the dot product with another vector, propagating nulls via an optional Bitmask.
#[inline(always)]
pub fn dot(v1: &[f64], v2: &[f64], null_mask: Option<&Bitmask>, null_count: Option<usize>) -> f64 {
    assert_eq!(v1.len(), v2.len(), "dot: input length mismatch");

    let len = v1.len();

    #[cfg(feature = "simd")]
    if is_simd_aligned(v1) && is_simd_aligned(v2) {
        let needs_nulls = has_nulls(null_count, null_mask);
        const N: usize = W64;
        let mut sum_v = Simd::<f64, N>::splat(0.0);
        let mut comp_v = Simd::<f64, N>::splat(0.0);
        let mut i = 0;
        if !needs_nulls {
            // fast path: no nulls - compensated per-lane accumulation
            while i + N <= len {
                let a = Simd::<f64, N>::from_slice(&v1[i..i + N]);
                let b = Simd::<f64, N>::from_slice(&v2[i..i + N]);
                neumaier_simd_add(&mut sum_v, &mut comp_v, a * b);
                i += N;
            }
        } else {
            // null-aware path: compensated per-lane with masked products
            let mask_bytes = null_mask.unwrap().as_bytes();
            while i + N <= len {
                let a = Simd::<f64, N>::from_slice(&v1[i..i + N]);
                let b = Simd::<f64, N>::from_slice(&v2[i..i + N]);
                let prod = a * b;
                let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);
                neumaier_simd_add(
                    &mut sum_v,
                    &mut comp_v,
                    lane_mask.select(prod, Simd::splat(0.0)),
                );
                i += N;
            }
        }
        // Horizontal reduction + scalar tail
        let mut acc = (sum_v + comp_v).reduce_sum();
        for idx in i..len {
            if !needs_nulls || unsafe { null_mask.unwrap().get_unchecked(idx) } {
                acc += v1[idx] * v2[idx];
            }
        }
        return acc;
    }

    // Scalar fallback - alignment check failed or no simd flag
    let needs_nulls = has_nulls(null_count, null_mask);
    let mut acc = 0.0_f64;
    let mut comp = 0.0_f64;
    if !needs_nulls {
        for i in 0..len {
            neumaier_add(&mut acc, &mut comp, v1[i] * v2[i]);
        }
    } else {
        let mb = null_mask.unwrap();
        for i in 0..len {
            if unsafe { mb.get_unchecked(i) } {
                neumaier_add(&mut acc, &mut comp, v1[i] * v2[i]);
            }
        }
    }
    acc + comp
}

/// Computes a stable key-based argsort of indices
#[inline(always)]
pub fn argsort(
    v: &[f64],
    mask: Option<&Bitmask>,
    null_count: Option<usize>,
    descending: bool,
) -> Vec64<usize> {
    let mut idx: Vec64<usize> = (0..v.len()).collect();
    if !has_nulls(null_count, mask) {
        if descending {
            idx.sort_unstable_by(|&i, &j| {
                v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            idx.sort_unstable_by(|&i, &j| {
                v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    } else {
        let mb = mask.expect("argsort: mask required when nulls present");
        idx.sort_unstable_by(|&i, &j| {
            let vi_null = !unsafe { mb.get_unchecked(i) };
            let vj_null = !unsafe { mb.get_unchecked(j) };
            match (vi_null, vj_null) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // nulls last
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => {
                    if descending {
                        v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal)
                    }
                }
            }
        });
    }
    idx
}

/// Bins values into histogram buckets.
#[inline(always)]
pub fn histogram(
    v: &[f64],
    bins: &[f64],
    mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Vec<usize> {
    assert!(!bins.is_empty(), "histogram: bins must be non-empty");
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => mask.is_some(),
    };

    let bins_sorted = bins.windows(2).all(|w| w[0] <= w[1]);
    let mut counts = vec![0; bins.len() + 1];
    if !has_nulls {
        if bins_sorted {
            for &x in v {
                match bins.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
                    Ok(pos) | Err(pos) => counts[pos] += 1,
                }
            }
        } else {
            for &x in v {
                let mut placed = false;
                for (i, &b) in bins.iter().enumerate() {
                    if x < b {
                        counts[i] += 1;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    counts[bins.len()] += 1;
                }
            }
        }
    } else {
        let mb = mask.expect("histogram: mask required when nulls present");
        if bins_sorted {
            for (i, &x) in v.iter().enumerate() {
                if unsafe { mb.get_unchecked(i) } {
                    match bins.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
                        Ok(pos) | Err(pos) => counts[pos] += 1,
                    }
                }
            }
        } else {
            for (i, &x) in v.iter().enumerate() {
                if unsafe { mb.get_unchecked(i) } {
                    let mut placed = false;
                    for (j, &b) in bins.iter().enumerate() {
                        if x < b {
                            counts[j] += 1;
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        counts[bins.len()] += 1;
                    }
                }
            }
        }
    }
    counts
}

/// Fixed-bin histogram with `[left, right)` boundary semantics.
///
/// `edges` must be sorted and have at least 2 elements. Returns a count vector
/// of length `edges.len() - 1` where `counts[i]` holds the number of values
/// satisfying `edges[i] <= v < edges[i+1]`. Values outside the range are excluded.
#[inline(always)]
pub fn histogram_edges(
    v: &[f64],
    edges: &[f64],
    mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Vec64<u64> {
    debug_assert!(edges.len() >= 2, "histogram_edges: need at least 2 edges");
    let n_bins = edges.len() - 1;
    let mut counts = vec64![0u64; n_bins];
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => mask.is_some(),
    };

    if !has_nulls {
        for &x in v {
            if x < edges[0] || x >= edges[n_bins] {
                continue;
            }
            let mut lo = 0usize;
            let mut hi = edges.len() - 1;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if x < edges[mid] {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            counts[lo] += 1;
        }
    } else {
        let mb = mask.expect("histogram_edges: mask required when nulls present");
        for (i, &x) in v.iter().enumerate() {
            if !unsafe { mb.get_unchecked(i) } {
                continue;
            }
            if x < edges[0] || x >= edges[n_bins] {
                continue;
            }
            let mut lo = 0usize;
            let mut hi = edges.len() - 1;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if x < edges[mid] {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            counts[lo] += 1;
        }
    }
    counts
}

/// Uniform reservoir sampling of size k.
#[inline(always)]
pub fn reservoir_sample(
    v: &[f64],
    k: usize,
    mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> FloatArray<f64> {
    use rand::prelude::*;
    assert!(k > 0, "reservoir_sample: k must be positive");
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => mask.is_some(),
    };

    let mut rng = rand::rng();
    let mut out = Vec64::with_capacity(k);
    let mut seen = 0usize;

    for (idx, &val) in v.iter().enumerate() {
        if has_nulls && !unsafe { mask.unwrap().get_unchecked(idx) } {
            continue;
        }
        if out.len() < k {
            out.push(val);
        } else {
            let j = rng.random_range(0..=seen);
            if j < k {
                out[j] = val;
            }
        }
        seen += 1;
    }
    assert_eq!(out.len(), k, "reservoir_sample: not enough valid values");
    FloatArray::from_vec64(out, None)
}

// --- Vector transformations ---

/// Scales the vector in place: x ← αx
/// Kernel: SCAL (L2-2)
#[inline(always)]
pub fn scale(v: &mut [f64], alpha: f64, null_mask: Option<&Bitmask>, null_count: Option<usize>) {
    #[cfg(feature = "simd")]
    if is_simd_aligned(v) {
        const N: usize = W64;
        let len = v.len();
        let alpha_v = Simd::<f64, N>::splat(alpha);
        let mut i = 0;

        if !has_nulls(null_count, null_mask) {
            // Dense path: just scale every lane
            while i + N <= len {
                let chunk = Simd::from_slice(&v[i..i + N]);
                let scaled = chunk * alpha_v;
                v[i..i + N].copy_from_slice(&scaled.to_array());
                i += N;
            }
            // Remainder
            for x in &mut v[i..] {
                *x *= alpha;
            }
        } else {
            // Null‐aware path
            let mb = null_mask.expect("scale: mask required when nulls present");
            let mask_bytes = mb.as_bytes();

            // Vectorized blocks
            while i + N <= len {
                let chunk = Simd::from_slice(&v[i..i + N]);
                // Build SIMD mask<M,N> from the global bitmask
                let lane_mask: Mask<i64, N> = bitmask_to_simd_mask::<N, i64>(mask_bytes, i, len);
                // Only scale the valid lanes, leave the rest untouched
                let result = lane_mask.select(chunk * alpha_v, chunk);
                v[i..i + N].copy_from_slice(&result.to_array());
                i += N;
            }
            // Scalar remainder
            for idx in i..len {
                if unsafe { mb.get_unchecked(idx) } {
                    v[idx] *= alpha;
                }
            }
        }
        return;
    }

    // Scalar fallback - alignment check failed or no SIMD flag
    if !has_nulls(null_count, null_mask) {
        for x in v {
            *x *= alpha;
        }
    } else {
        let mb = null_mask.expect("scale: mask required when nulls present");
        for (i, x) in v.iter_mut().enumerate() {
            if unsafe { mb.get_unchecked(i) } {
                *x *= alpha;
            }
        }
    }
}

/// In-place vector scaling: x ← α·x
///
/// This is the version for a matrix in a flat strided buffer,
/// and therefore tackles incx ("increment x") the gap between values.
///
/// Multiplies each element of the input vector `x` by the scalar `alpha`.
/// Equivalent to `x[i] ← alpha * x[i]` for all valid `i`.
#[inline(always)]
pub fn scale_vec(
    n: i32,
    alpha: f64,
    x: &mut [f64],
    incx: i32,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if n < 0 {
        return Err(KernelError::InvalidArguments(
            "n must be non-negative".into(),
        ));
    }
    let n = n as usize;
    let incx = incx as usize;
    if incx == 0 {
        return Err(KernelError::InvalidArguments(
            "incx must be positive".into(),
        ));
    }
    if n == 0 {
        return Ok(());
    }
    if (n - 1).saturating_mul(incx) >= x.len() {
        return Err(KernelError::InvalidArguments(
            "indexing out of bounds".into(),
        ));
    }

    let mask_present = has_nulls(null_count, null_mask);

    #[cfg(feature = "simd")]
    if is_simd_aligned(x) {
        const LANES: usize = W64;
        let alpha_v = Simd::<f64, LANES>::splat(alpha);

        // only vectorize when truly contiguous
        if incx == 1 && !mask_present {
            // dense contiguous path
            let mut i = 0;
            while i + LANES <= n {
                let chunk = Simd::from_slice(&x[i..i + LANES]);
                let out_v = chunk * alpha_v;
                x[i..i + LANES].copy_from_slice(&out_v.to_array());
                i += LANES;
            }
            for xi in &mut x[i..n] {
                *xi *= alpha;
            }
            return Ok(());
        }
        if incx == 1 && mask_present {
            // contiguous + null‐aware
            let mb = null_mask.unwrap();
            let bytes = mb.as_bytes();
            let mut i = 0;
            while i + LANES <= n {
                // build a SIMD mask from the arrow bitmap
                let lane_mask: Mask<i64, LANES> = bitmask_to_simd_mask::<LANES, i64>(bytes, i, n);
                let chunk = Simd::from_slice(&x[i..i + LANES]);
                let scaled = chunk * alpha_v;
                // only scale where mask==true, else leave original
                let out_v = lane_mask.select(scaled, chunk);
                x[i..i + LANES].copy_from_slice(&out_v.to_array());
                i += LANES;
            }
            // tail
            for idx in i..n {
                if unsafe { mb.get_unchecked(idx) } {
                    x[idx] *= alpha;
                }
            }
            return Ok(());
        }
    }

    // scalar (strided) fallback
    if !mask_present {
        let mut idx = 0;
        for _ in 0..n {
            x[idx] *= alpha;
            idx += incx;
        }
    } else {
        let mb = null_mask.unwrap();
        let mut idx = 0;
        for _ in 0..n {
            if unsafe { mb.get_unchecked(idx) } {
                x[idx] *= alpha;
            }
            idx += incx;
        }
    }
    Ok(())
}

/// Scaled vector addition: y ← α·x + y
///
/// Computes a linear combination of two vectors, scaling `x` by `alpha` and accumulating into `y`.
/// Equivalent to `y[i] ← alpha * x[i] + y[i]` for all valid `i`.
#[inline(always)]
pub fn axpy(
    n: i32,
    alpha: f64,
    x: &[f64],
    incx: i32,
    y: &mut [f64],
    incy: i32,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if n < 0 {
        return Err(KernelError::InvalidArguments(
            "n must be non-negative".into(),
        ));
    }
    let n = n as usize;
    let incx = incx as usize;
    let incy = incy as usize;
    if incx == 0 || incy == 0 {
        return Err(KernelError::InvalidArguments(
            "increments must be positive".into(),
        ));
    }
    if n == 0 {
        return Ok(());
    }
    if (n - 1).saturating_mul(incx) >= x.len() {
        return Err(KernelError::InvalidArguments("x out of bounds".into()));
    }
    if (n - 1).saturating_mul(incy) >= y.len() {
        return Err(KernelError::InvalidArguments("y out of bounds".into()));
    }

    let mask_present = has_nulls(null_count, null_mask);

    #[cfg(feature = "simd")]
    if is_simd_aligned(x) {
        const LANES: usize = W64;
        let alpha_v = Simd::<f64, LANES>::splat(alpha);

        // only when both x and y are contiguous
        if incx == 1 && incy == 1 && !mask_present {
            // dense contiguous
            let mut i = 0;
            while i + LANES <= n {
                let xv = Simd::from_slice(&x[i..i + LANES]);
                let yv = Simd::from_slice(&y[i..i + LANES]);
                let out_v = alpha_v * xv + yv;
                y[i..i + LANES].copy_from_slice(&out_v.to_array());
                i += LANES;
            }
            for j in i..n {
                y[j] += alpha * x[j];
            }
            return Ok(());
        }
        if incx == 1 && incy == 1 && mask_present {
            // contiguous + null‐aware
            let mb = null_mask.unwrap();
            let bytes = mb.as_bytes();
            let mut i = 0;
            while i + LANES <= n {
                let lane_mask: Mask<i64, LANES> = bitmask_to_simd_mask::<LANES, i64>(bytes, i, n);
                let xv = Simd::from_slice(&x[i..i + LANES]);
                let yv = Simd::from_slice(&y[i..i + LANES]);
                let computed = alpha_v * xv + yv;
                let out_v = lane_mask.select(computed, yv);
                y[i..i + LANES].copy_from_slice(&out_v.to_array());
                i += LANES;
            }
            for idx in i..n {
                if unsafe { mb.get_unchecked(idx) } {
                    y[idx] += alpha * x[idx];
                }
            }
            return Ok(());
        }
    }

    // scalar (potentially strided) fallback
    if !mask_present {
        let mut ix = 0;
        let mut iy = 0;
        for _ in 0..n {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    } else {
        let mb = null_mask.unwrap();
        let mut ix = 0;
        let mut iy = 0;
        for _ in 0..n {
            if unsafe { mb.get_unchecked(ix) } {
                y[iy] += alpha * x[ix];
            }
            ix += incx;
            iy += incy;
        }
    }
    Ok(())
}

/// Computes the L2 norm (‖x‖₂)
#[inline(always)]
pub fn l2_norm(v: &[f64], mask: Option<&Bitmask>, null_count: Option<usize>) -> f64 {
    sum_squares(v, mask, null_count).sqrt()
}

/// Computes the Euclidean (ℓ₂) norm of a vector: ‖x‖₂ = sqrt(Σ xᵢ²)
///
/// Returns the magnitude (Euclidean norm) of the vector `x`.
/// For an input vector `x`, computes `sqrt(sum(x_i^2))`.
#[inline(always)]
pub fn vector_norm(
    n: i32,
    x: &[f64],
    incx: i32,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<f64, KernelError> {
    if n < 0 {
        return Err(KernelError::InvalidArguments(
            "n must be non-negative".into(),
        ));
    }
    if incx <= 0 {
        return Err(KernelError::InvalidArguments(
            "incx must be positive".into(),
        ));
    }
    let n = n as usize;
    let incx = incx as usize;
    if n == 0 {
        return Ok(0.0);
    }
    if (n - 1).saturating_mul(incx) >= x.len() {
        return Err(KernelError::InvalidArguments(
            "indexing out of bounds for x".into(),
        ));
    }

    let mask_present = has_nulls(null_count, null_mask);

    #[cfg(feature = "simd")]
    if is_simd_aligned(x) {
        use crate::utils::bitmask_to_simd_mask;
        use core::simd::Simd;

        const LANES: usize = W64;

        // only vectorise when contiguous
        if incx == 1 && !mask_present {
            // dense contiguous path - compensated per-lane accumulation
            let mut sum_v = Simd::<f64, LANES>::splat(0.0);
            let mut comp_v = Simd::<f64, LANES>::splat(0.0);
            let mut i = 0;
            while i + LANES <= n {
                let v = Simd::from_slice(&x[i..i + LANES]);
                neumaier_simd_add(&mut sum_v, &mut comp_v, v * v);
                i += LANES;
            }
            // Horizontal reduction + scalar tail
            let mut sumsq = (sum_v + comp_v).reduce_sum();
            for &xi in &x[i..n] {
                sumsq += xi * xi;
            }
            return Ok(sumsq.sqrt());
        }

        if incx == 1 && mask_present {
            // contiguous + null-aware - compensated per-lane accumulation
            let mb = null_mask.unwrap();
            let bytes = mb.as_bytes();
            let mut sum_v = Simd::<f64, LANES>::splat(0.0);
            let mut comp_v = Simd::<f64, LANES>::splat(0.0);
            let mut i = 0;
            while i + LANES <= n {
                let lane_mask: Mask<i64, LANES> = bitmask_to_simd_mask::<LANES, i64>(bytes, i, n);
                let v = Simd::from_slice(&x[i..i + LANES]);
                let v = lane_mask.select(v, Simd::splat(0.0));
                neumaier_simd_add(&mut sum_v, &mut comp_v, v * v);
                i += LANES;
            }
            // Horizontal reduction + scalar tail
            let mut sumsq = (sum_v + comp_v).reduce_sum();
            for idx in i..n {
                if unsafe { mb.get_unchecked(idx) } {
                    sumsq += x[idx] * x[idx];
                }
            }
            return Ok(sumsq.sqrt());
        }
    }

    // scalar (possibly strided) fallback - compensated
    let mut sumsq = 0.0_f64;
    let mut comp = 0.0_f64;
    if !mask_present {
        let mut idx = 0;
        for _ in 0..n {
            let xi = x[idx];
            neumaier_add(&mut sumsq, &mut comp, xi * xi);
            idx += incx;
        }
    } else {
        let mb = null_mask.unwrap();
        let mut idx = 0;
        for _ in 0..n {
            if unsafe { mb.get_unchecked(idx) } {
                let xi = x[idx];
                neumaier_add(&mut sumsq, &mut comp, xi * xi);
            }
            idx += incx;
        }
    }
    Ok((sumsq + comp).sqrt())
}

/// Compute Huber weights in place: w[i] = min(1, c / |r_i|)
/// where r_i = (vals[i] - location) / scale.
///
/// Elements where |r_i| <= c get weight 1.0 (inlier), larger residuals
/// are downweighted proportional to c / |r|.
#[inline]
pub fn huber_weights(
    vals: &[f64],
    weights: &mut [f64],
    location: f64,
    scale: f64,
    c: f64,
) {
    debug_assert_eq!(vals.len(), weights.len());
    for (w, &x) in weights.iter_mut().zip(vals.iter()) {
        let r = ((x - location) / scale).abs();
        *w = if r <= c { 1.0 } else { c / r };
    }
}

/// Compute Tukey biweight weights in place: w[i] = (1 - u^2)^2 if |r_i| < c, else 0
/// where u = r_i / c and r_i = (vals[i] - location) / scale.
///
/// Completely rejects points with standardised residuals beyond c.
#[inline]
pub fn biweight_weights(
    vals: &[f64],
    weights: &mut [f64],
    location: f64,
    scale: f64,
    c: f64,
) {
    debug_assert_eq!(vals.len(), weights.len());
    for (w, &x) in weights.iter_mut().zip(vals.iter()) {
        let r = ((x - location) / scale).abs();
        *w = if r < c {
            let u = r / c;
            let t = 1.0 - u * u;
            t * t
        } else {
            0.0
        };
    }
}

/// Compute weighted mean: sum(w * x) / sum(w).
/// Returns NaN if sum of weights is zero.
#[inline]
pub fn weighted_mean(vals: &[f64], weights: &[f64]) -> f64 {
    debug_assert_eq!(vals.len(), weights.len());
    let mut sum_wx = 0.0_f64;
    let mut sum_w = 0.0_f64;
    for (&x, &w) in vals.iter().zip(weights.iter()) {
        sum_wx += w * x;
        sum_w += w;
    }
    if sum_w == 0.0 { f64::NAN } else { sum_wx / sum_w }
}

#[cfg(test)]
mod tests {
    use minarrow::{Bitmask, vec64};

    use super::*;

    fn make_mask(bits: &[bool]) -> Bitmask {
        Bitmask::from_bools(bits)
    }

    #[test]
    fn test_dot_no_nulls() {
        let v1 = vec64![1.0, 2.0, 3.0, 4.0];
        let v2 = vec64![2.0, 0.5, 1.0, -1.0];
        let expected = 1.0 * 2.0 + 2.0 * 0.5 + 3.0 * 1.0 + 4.0 * (-1.0);
        assert_eq!(dot(v1.as_slice(), v2.as_slice(), None, None), expected);

        // Null count is Some(0) should still use fast path
        assert_eq!(dot(&v1, &v2, None, Some(0)), expected);
    }

    #[test]
    fn test_dot_with_nulls() {
        let v1 = vec64![1.0, 2.0, 3.0, 4.0];
        let v2 = vec64![2.0, 0.5, 1.0, -1.0];
        let mask = make_mask(&[true, false, true, false]);
        let null_count = 2;
        // Only indices 0 and 2 are valid
        let expected = 1.0 * 2.0 + 3.0 * 1.0;
        assert_eq!(dot(&v1, &v2, Some(&mask), Some(null_count)), expected);

        // Null count None but mask provided, should also work
        assert_eq!(dot(&v1, &v2, Some(&mask), None), expected);
    }

    #[test]
    fn test_dot_all_nulls() {
        let v1 = vec64![1.0, 2.0, 3.0, 4.0];
        let v2 = vec64![2.0, 0.5, 1.0, -1.0];
        let mask = make_mask(&[false, false, false, false]);
        let expected = 0.0;
        assert_eq!(dot(&v1, &v2, Some(&mask), Some(4)), expected);
    }

    #[test]
    fn test_argsort_no_nulls() {
        let v = vec64![5.0, 2.0, 1.0, 4.0];
        assert_eq!(argsort(&v, None, None, false), vec64![2, 1, 3, 0]);
        assert_eq!(argsort(&v, None, None, true), vec64![0, 3, 1, 2]);
    }

    #[test]
    fn test_argsort_with_nulls() {
        let v = vec64![5.0, 2.0, 1.0, 4.0, 3.0];
        let mask = make_mask(&[true, false, true, true, false]);
        // valid indices: 0,2,3; so sorted: [2,3,0] (by value) then nulls [1,4]
        assert_eq!(
            argsort(&v, Some(&mask), Some(2), false),
            vec64![2, 3, 0, 1, 4]
        );
        assert_eq!(
            argsort(&v, Some(&mask), Some(2), true),
            vec64![0, 3, 2, 1, 4]
        );
    }

    #[test]
    fn test_histogram_no_nulls() {
        let v = vec64![1.0, 2.0, 4.0, 6.0];
        let bins = [2.5, 5.0];
        // Buckets: [<2.5],[2.5,5.0),[>=5.0] => [2,1,1]
        // values: [1.0,2.0] in bucket 0, [4.0] in bucket 1, [6.0] in bucket 2
        assert_eq!(histogram(&v, &bins, None, None), vec![2, 1, 1]);
    }

    #[test]
    fn test_histogram_with_nulls() {
        let v = vec64![1.0, 2.0, 4.0, 6.0];
        let bins = [2.5, 5.0];
        let mask = make_mask(&[false, true, true, false]);
        // Only v[1]=2.0 and v[2]=4.0: bins [<2.5]=1, [2.5,5.0)=1, [>=5.0]=0
        assert_eq!(histogram(&v, &bins, Some(&mask), Some(2)), vec![1, 1, 0]);
    }

    #[test]
    fn test_histogram_all_nulls() {
        let v = vec64![1.0, 2.0, 4.0, 6.0];
        let bins = [2.5, 5.0];
        let mask = make_mask(&[false, false, false, false]);
        assert_eq!(histogram(&v, &bins, Some(&mask), Some(4)), vec![0, 0, 0]);
    }

    #[test]
    fn test_reservoir_sample_basic() {
        let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = reservoir_sample(&v, 3, None, None);
        assert_eq!(arr.data.len(), 3);
        // Should be a subset of original input (since all valid)
        for &x in arr.data.iter() {
            assert!(v.contains(&x));
        }
    }

    #[test]
    fn test_reservoir_sample_with_nulls() {
        let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = make_mask(&[false, true, false, true, false, true]);
        let arr = reservoir_sample(&v, 2, Some(&mask), Some(3));
        // Only indices 1,3,5 are valid. Sampled values must come from v[1], v[3], v[5].
        for &x in arr.data.iter() {
            assert!(x == 2.0 || x == 4.0 || x == 6.0);
        }
    }

    #[test]
    fn test_scale_no_nulls() {
        let mut v = vec64![1.0, 2.0, 3.0];
        scale(&mut v, 3.0, None, None);
        assert_eq!(*v, [3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_scale_with_nulls() {
        let mut v = vec64![1.0, 2.0, 3.0];
        let mask = make_mask(&[false, true, true]);
        scale(&mut v, 4.0, Some(&mask), Some(1));
        // Only v[1] and v[2] are scaled
        assert_eq!(*v, [1.0, 8.0, 12.0]);
    }

    #[test]
    fn test_scale_vec_no_nulls() {
        let mut v = vec64![2.0, 4.0, 8.0, 16.0];
        let r = scale_vec(2, 0.5, &mut v, 2, None, None);
        assert!(r.is_ok());
        assert_eq!(*v, [1.0, 4.0, 4.0, 16.0]);
    }

    #[test]
    fn test_scale_vec_with_nulls() {
        let mut v = vec64![2.0, 4.0, 8.0, 16.0];
        let mask = make_mask(&[false, true, true, false]);
        let r = scale_vec(2, 0.5, &mut v, 2, Some(&mask), Some(2));
        assert!(r.is_ok());
        // Only index 0 and 2 are processed but only 2 is valid
        assert_eq!(*v, [2.0, 4.0, 4.0, 16.0]);
    }

    #[test]
    fn test_axpy_no_nulls() {
        let mut y = vec64![10.0, 20.0, 30.0, 40.0];
        let x = vec64![1.0, 2.0, 3.0, 4.0];
        let r = axpy(4, 2.0, &x, 1, &mut y, 1, None, None);
        assert!(r.is_ok());
        assert_eq!(*y, [12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_axpy_with_nulls() {
        let mut y = vec64![10.0, 20.0, 30.0, 40.0];
        let x = vec64![1.0, 2.0, 3.0, 4.0];
        let mask = make_mask(&[false, true, false, true]);
        let r = axpy(4, 2.0, &x, 1, &mut y, 1, Some(&mask), Some(2));
        assert!(r.is_ok());
        // Only indices 1 and 3 are updated
        assert_eq!(*y, [10.0, 24.0, 30.0, 48.0]);
    }

    #[test]
    fn test_l2_norm_no_nulls() {
        let v = vec64![3.0, 4.0];
        let n = l2_norm(&v, None, None);
        assert!((n - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_l2_norm_with_nulls() {
        let v = vec64![3.0, 4.0];
        let mask = make_mask(&[false, true]);
        let n = l2_norm(&v, Some(&mask), Some(1));
        assert!((n - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_vector_norm_no_nulls() {
        let v = vec64![3.0, 0.0, 4.0];
        let n = vector_norm(3, &v, 1, None, None).unwrap();
        assert!((n - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_vector_norm_with_nulls() {
        let v = vec64![3.0, 0.0, 4.0];
        let mask = make_mask(&[true, false, true]);
        let n = vector_norm(3, &v, 1, Some(&mask), Some(1)).unwrap();
        // Only indices 0 and 2 are used: sqrt(9+16)=5
        assert!((n - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_vector_norm_all_nulls() {
        let v = vec64![3.0, 0.0, 4.0];
        let mask = make_mask(&[false, false, false]);
        let n = vector_norm(3, &v, 1, Some(&mask), Some(3)).unwrap();
        assert_eq!(n, 0.0);
    }

    /// Verify Neumaier compensation on a large dot product where naive
    /// summation would accumulate significant floating-point error.
    ///
    /// Strategy: construct two 1M-element vectors where each product term
    /// equals 1.0, so the exact dot product is 1_000_000.0. Add a large
    /// offset to one vector to create catastrophic cancellation in naive
    /// summation, then verify the compensated result stays within 1 ULP.
    #[test]
    fn test_dot_large_compensated_accuracy() {
        let n = 1_000_000;
        let offset = 1e12;

        // v1[i] = offset + 1.0, v2[i] = 1.0 for all i
        // Each product = offset + 1.0. Sum = n * (offset + 1.0).
        // We compare against the known exact answer.
        let v1: Vec64<f64> = (0..n).map(|_| offset + 1.0).collect();
        let v2: Vec64<f64> = (0..n).map(|_| 1.0).collect();

        let result = dot(&v1, &v2, None, None);
        let exact = n as f64 * (offset + 1.0);

        // With Neumaier compensation, error should be within a few ULPs.
        // Without it, naive summation on 1M terms accumulates O(n) ULPs.
        let ulps = ((result - exact) / exact).abs() / f64::EPSILON;
        assert!(
            ulps < 4.0,
            "dot product off by {ulps:.1} ULPs: got {result}, expected {exact}",
        );
    }

    /// Same principle but with alternating signs to stress cancellation
    /// across SIMD lanes. Exact result should be 0.0 since pairs cancel.
    #[test]
    fn test_dot_large_cancellation() {
        let n = 1_000_000;

        // v1 = [1, -1, 1, -1, ...], v2 = [1, 1, 1, 1, ...]
        // Exact dot = 0 (pairs cancel). Naive summation drifts.
        let v1: Vec64<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1e8 } else { -1e8 })
            .collect();
        let v2: Vec64<f64> = (0..n).map(|_| 1.0).collect();

        let result = dot(&v1, &v2, None, None);
        // Exact answer is 0.0 since n is even
        assert!(
            result.abs() < 1e-4,
            "dot cancellation failed: got {result}, expected 0.0",
        );
    }

    /// Verify compensated l2_norm on a large vector where naive summation
    /// would lose precision in the sum-of-squares accumulation.
    #[test]
    fn test_l2_norm_large_compensated_accuracy() {
        let n = 1_000_000;

        // All elements = 1.0, so ||v||_2 = sqrt(n) = 1000.0
        let v: Vec64<f64> = (0..n).map(|_| 1.0).collect();

        let result = l2_norm(&v, None, None);
        let exact = (n as f64).sqrt();

        let ulps = ((result - exact) / exact).abs() / f64::EPSILON;
        assert!(
            ulps < 4.0,
            "l2_norm off by {ulps:.1} ULPs: got {result}, expected {exact}",
        );
    }

    // ── Robust weight kernels ──

    #[test]
    fn huber_weights_inliers_get_one() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut w = [0.0; 5];
        // location=3, scale=1, c=1.345 (standard Huber constant)
        huber_weights(&vals, &mut w, 3.0, 1.0, 1.345);
        // |r| for vals: [2, 1, 0, 1, 2]. c=1.345
        // indices 1,2,3 are inliers (|r| <= 1.345)
        assert!((w[2] - 1.0).abs() < 1e-15, "center should be 1.0");
        assert!((w[1] - 1.0).abs() < 1e-15, "r=1 inlier");
        assert!((w[3] - 1.0).abs() < 1e-15, "r=1 inlier");
        // indices 0,4 are outliers (|r|=2 > 1.345)
        assert!((w[0] - 1.345 / 2.0).abs() < 1e-15, "r=2 outlier weight");
        assert!((w[4] - 1.345 / 2.0).abs() < 1e-15, "r=2 outlier weight");
    }

    #[test]
    fn biweight_weights_rejects_outliers() {
        let vals = [0.0, 3.0, 100.0];
        let mut w = [0.0; 3];
        // location=3, scale=1, c=4.685 (standard biweight constant)
        biweight_weights(&vals, &mut w, 3.0, 1.0, 4.685);
        // r for vals: [3, 0, 97]
        // index 0: |r|=3 < 4.685, gets nonzero weight
        assert!(w[0] > 0.0, "r=3 should get nonzero weight");
        // index 1: r=0, weight = (1-0)^2 = 1.0
        assert!((w[1] - 1.0).abs() < 1e-15, "center should be 1.0");
        // index 2: |r|=97 >> c, weight = 0.0
        assert!((w[2] - 0.0).abs() < 1e-15, "extreme outlier should be 0.0");
    }

    #[test]
    fn weighted_mean_basic() {
        let vals = [1.0, 2.0, 3.0];
        let w = [1.0, 1.0, 1.0];
        assert!((weighted_mean(&vals, &w) - 2.0).abs() < 1e-15);

        let w2 = [0.0, 0.0, 1.0];
        assert!((weighted_mean(&vals, &w2) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn weighted_mean_zero_weights_nan() {
        let vals = [1.0, 2.0];
        let w = [0.0, 0.0];
        assert!(weighted_mean(&vals, &w).is_nan());
    }
}
