// Copyright (c) 2025 Peter Bower
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Aggregation Kernels Module** - *High-Performance Statistical Reduction Operations*
//!
//! Null-aware aggregation kernels implementing statistical reductions with SIMD acceleration.
//! Core building blocks for analytical workloads requiring high-throughput data summarisation.
//!
//! ## Core Operations
//! - **Sum aggregation**: Optimised summation with overflow handling and numerical stability
//! - **Min/Max operations**: Efficient extrema detection with proper null handling
//! - **Count operations**: Fast counting with distinct value detection capabilities
//! - **Statistical moments**: Mean, variance, and standard deviation calculations
//! - **Percentile calculations**: Quantile computation with interpolation support
//! - **String aggregation**: Concatenation and string-based reduction operations
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

#[cfg(not(feature = "fast_hash"))]
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::{Mul, Sub};
#[cfg(feature = "simd")]
use std::simd::{
    Mask, Select, Simd, SimdElement,
    cmp::{SimdOrd, SimdPartialOrd},
    num::{SimdFloat, SimdInt, SimdUint},
};

#[cfg(feature = "fast_hash")]
use ahash::AHashMap;
use minarrow::{Bitmask, Vec64};
use num_traits::{Float, NumCast, One, ToPrimitive, Zero};

use crate::kernels::sort::{sort_float, total_cmp_f};
use crate::traits::dense_iter::collect_valid;
use crate::traits::to_bits::ToBits;
use crate::utils::has_nulls;
use minarrow::utils::is_simd_aligned;

#[cfg(feature = "simd")]
use crate::utils::bitmask_to_simd_mask;

// --- Neumaier compensated summation helper ------------------------------------

/// Applies one step of Neumaier compensated summation.
///
/// Neumaier is a refinement of Kahan summation that also handles the case
/// where the addend is larger in magnitude than the running sum, reducing
/// floating-point accumulation error from O(n) to O(1) for well-conditioned inputs.
///
/// When infinities are involved, compensation is skipped to avoid NaN propagation
/// from `inf - inf` in the error term.
#[inline(always)]
pub(crate) fn neumaier_add(sum: &mut f64, comp: &mut f64, value: f64) {
    let t = *sum + value;
    // Skip compensation when infinities are involved to avoid NaN from inf - inf
    if t.is_finite() {
        if sum.abs() >= value.abs() {
            *comp += (*sum - t) + value;
        } else {
            *comp += (value - t) + *sum;
        }
    }
    *sum = t;
}

/// SIMD-width Neumaier compensated summation step.
///
/// Same algorithm as the scalar `neumaier_add` but operating on all SIMD lanes
/// in parallel. Each lane independently tracks its own sum and compensation term.
#[cfg(feature = "simd")]
#[inline(always)]
pub(crate) fn neumaier_simd_add<const N: usize>(
    sum: &mut Simd<f64, N>,
    comp: &mut Simd<f64, N>,
    value: Simd<f64, N>,
) where
{
    let t = *sum + value;
    let sum_dominant = sum.abs().simd_ge(value.abs());
    let comp_a = (*sum - t) + value;
    let comp_b = (value - t) + *sum;
    let delta = sum_dominant.select(comp_a, comp_b);
    // Skip compensation for non-finite lanes to avoid NaN from inf - inf
    let finite = t.is_finite();
    *comp += finite.select(delta, Simd::splat(0.0));
    *sum = t;
}

// --- SIMD/stat-moments helpers -----------------------------------------------

/// Generates stat‐moment aggregation (sum, sum², count)
/// for float types using SIMD or scalar fallback.
/// Uses Neumaier compensated summation for improved numerical accuracy.
macro_rules! impl_stat_moments_float {
    ($name:ident, $ty:ty, $LANES:expr) => {
        #[doc = concat!("Computes sum, sum-of-squares, and count for a `", stringify!($ty), "` slice.")]
        #[doc = "Skips nulls if present. Uses SIMD if available."]
        #[doc = "Uses Neumaier compensated summation for numerical accuracy."]
        #[inline(always)]
        pub fn $name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> (f64, f64, usize) {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    type V<const L: usize> = Simd<$ty, L>;
                    // The mask‐type for `Simd<$ty, N>` is `<$ty as SimdElement>::Mask`.
                    type M<const L: usize> = <$ty as SimdElement>::Mask;

                    const N: usize = $LANES;
                    let len = data.len();
                    let mut i = 0;
                    let (mut sum, mut sum2) = (0.0_f64, 0.0_f64);
                    let (mut comp, mut comp2) = (0.0_f64, 0.0_f64);
                    let mut cnt = 0_usize;

                    if !has_nulls {
                        // dense path
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            let dv = v.cast::<f64>();
                            cnt += N;
                            let chunk_sum = dv.reduce_sum();
                            neumaier_add(&mut sum, &mut comp, chunk_sum);
                            let chunk_sum2 = (dv * dv).reduce_sum();
                            neumaier_add(&mut sum2, &mut comp2, chunk_sum2);
                            i += N;
                        }
                        // tail
                        for &v in &data[i..] {
                            let x = v as f64;
                            neumaier_add(&mut sum, &mut comp, x);
                            neumaier_add(&mut sum2, &mut comp2, x * x);
                            cnt += 1;
                        }
                    } else {
                        // null‐aware SIMD path
                        let mb = mask.expect("Mask must be Some if nulls are present");
                        let mask_bytes = mb.as_bytes();

                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            // extract a SIMD mask of valid lanes
                            let lane_mask: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                            // zero‐out invalid lanes
                            let zeros = V::<N>::splat(0.0 as $ty);
                            let vv = lane_mask.select(v, zeros);

                            // count valid lanes
                            cnt += lane_mask.to_bitmask().count_ones() as usize;

                            let dv = vv.cast::<f64>();
                            let chunk_sum = dv.reduce_sum();
                            neumaier_add(&mut sum, &mut comp, chunk_sum);
                            let chunk_sum2 = (dv * dv).reduce_sum();
                            neumaier_add(&mut sum2, &mut comp2, chunk_sum2);

                            i += N;
                        }
                        // scalar tail
                        for j in i..len {
                            if unsafe { mb.get_unchecked(j) } {
                                let x = data[j] as f64;
                                neumaier_add(&mut sum, &mut comp, x);
                                neumaier_add(&mut sum2, &mut comp2, x * x);
                                cnt += 1;
                            }
                        }
                    }
                    return (sum + comp, sum2 + comp2, cnt);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let (mut sum, mut sum2, mut cnt) = (0.0_f64, 0.0_f64, 0_usize);
            let (mut comp, mut comp2) = (0.0_f64, 0.0_f64);
            if !has_nulls {
                for &v in data {
                    let x = v as f64;
                    neumaier_add(&mut sum, &mut comp, x);
                    neumaier_add(&mut sum2, &mut comp2, x * x);
                    cnt += 1;
                }
            } else {
                let mb = mask.expect("Mask must be Some if nulls are present");
                for (i, &v) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(i) } {
                        let x = v as f64;
                        neumaier_add(&mut sum, &mut comp, x);
                        neumaier_add(&mut sum2, &mut comp2, x * x);
                        cnt += 1;
                    }
                }
            }
            (sum + comp, sum2 + comp2, cnt)
        }
    };
}

/// Generates stat-moment aggregation (sum, sum², count)
/// for integer types using SIMD or scalar fallback.
macro_rules! impl_stat_moments_int {
    ($name:ident, $ty:ty, $acc:ty, $LANES:expr, $mask_ty:ty) => {
        #[doc = concat!("Computes sum, sum-of-squares, and count for `", stringify!($ty), "` integers.")]
        #[doc = concat!(" Uses `", stringify!($acc), "` for accumulation. Skips nulls if present.")]
        #[inline(always)]
        pub fn $name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>
        ) -> (f64, f64, usize) {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                const N: usize = $LANES;
                let mut i = 0;
                let mut sum: $acc = 0;
                let mut sum2: $acc = 0;
                let mut cnt = 0usize;

                if !has_nulls {
                    while i + N <= data.len() {
                        let v = Simd::<$ty, N>::from_slice(&data[i..i + N]);
                        sum += v.reduce_sum() as $acc;
                        sum2 += (v * v).reduce_sum() as $acc;
                        cnt += N;
                        i += N;
                    }
                    for &x in &data[i..] {
                        let x = x as $acc;
                        sum += x;
                        sum2 += x * x;
                        cnt += 1;
                    }
                } else {
                    let mb = mask.expect("Mask must be Some if nulls are present or mask was supplied");
                    let mask_bytes = &mb.bits;
                    while i + N <= data.len() {
                        let v = Simd::<$ty, N>::from_slice(&data[i..i + N]);
                        // use signed type for mask to satisfy trait bound
                        let lane_mask = bitmask_to_simd_mask::<N, $mask_ty>(mask_bytes, i, data.len());
                        let vv = lane_mask.select(v, Simd::<$ty, N>::splat(0 as $ty));
                        sum += vv.reduce_sum() as $acc;
                        sum2 += (vv * vv).reduce_sum() as $acc;
                        cnt += lane_mask.to_bitmask().count_ones() as usize;
                        i += N;
                    }
                    for j in i..data.len() {
                        if unsafe { mb.get_unchecked(j) } {
                            let x = data[j] as $acc;
                            sum += x;
                            sum2 += x * x;
                            cnt += 1;
                        }
                    }
                }
                    return (sum as f64, sum2 as f64, cnt);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut sum: $acc = 0;
            let mut sum2: $acc = 0;
            let mut cnt = 0usize;
            if !has_nulls {
                for &x in data.iter() {
                    let xi = x as $acc;
                    sum += xi;
                    sum2 += xi * xi;
                    cnt += 1;
                }
            } else {
                let mb = mask.expect("Mask must be Some if nulls are present or mask was supplied");
                for (i, &x) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(i) } {
                        let xi = x as $acc;
                        sum += xi;
                        sum2 += xi * xi;
                        cnt += 1;
                    }
                }
            }
            (sum as f64, sum2 as f64, cnt)
        }
    };
}

/// Implements SIMD-enabled min/max reducers
/// for numeric slices with or without nulls.
macro_rules! impl_reduce_min_max {
    ($name:ident, $ty:ty, $LANES:expr, $minval:expr, $maxval:expr) => {
        /// Finds the minimum and maximum values in a numeric slice.
        ///
        /// Uses SIMD operations when available for optimal performance.
        /// Handles null values through optional bitmask.
        ///
        /// # Arguments
        ///
        /// * `data` - The slice of numeric values to process
        /// * `mask` - Optional bitmask indicating valid/invalid elements
        /// * `null_count` - Optional count of null values for optimisation
        ///
        /// # Returns
        ///
        /// `Some((min, max))` if any valid values exist, `None` if all values are null
        #[inline(always)]
        pub fn $name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Option<($ty, $ty)> {
            let has_nulls = has_nulls(null_count, mask);
            let len = data.len();
            if len == 0 {
                return None;
            }

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    type V<const N: usize> = Simd<$ty, N>;
                    type M<const N: usize> = <$ty as SimdElement>::Mask;
                    const N: usize = $LANES;

                    let mut first_valid = None;
                    if !has_nulls {
                        // Find first value
                        if len == 0 {
                            return None;
                        }
                        first_valid = Some((0, data[0]));
                    } else {
                        let mb = mask.expect("Mask must be Some if nulls are present");
                        for (idx, &x) in data.iter().enumerate() {
                            if unsafe { mb.get_unchecked(idx) } {
                                first_valid = Some((idx, x));
                                break;
                            }
                        }
                    }
                    let (mut i, x0) = match first_valid {
                        Some(v) => v,
                        None => return None,
                    };
                    let mut min_s = x0;
                    let mut max_s = x0;
                    i += 1;

                    // SIMD alignment: process until next SIMD-aligned index
                    while i < len && (i % N != 0) {
                        let x = data[i];
                        if !has_nulls
                            || (has_nulls && {
                                let mb = mask.unwrap();
                                unsafe { mb.get_unchecked(i) }
                            })
                        {
                            min_s = min_s.min(x);
                            max_s = max_s.max(x);
                        }
                        i += 1;
                    }

                    let mut min_v = V::<N>::splat(min_s);
                    let mut max_v = V::<N>::splat(max_s);

                    if !has_nulls {
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            min_v = min_v.simd_min(v);
                            max_v = max_v.simd_max(v);
                            i += N;
                        }
                    } else {
                        let mb = mask.unwrap();
                        let mask_bytes = mb.as_bytes();
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                            // Null entries are replaced by previous running min/max
                            let v_min = lane_m.select(v, V::<N>::splat(min_s));
                            let v_max = lane_m.select(v, V::<N>::splat(max_s));
                            min_v = min_v.simd_min(v_min);
                            max_v = max_v.simd_max(v_max);
                            i += N;
                        }
                    }

                    for idx in i..len {
                        let x = data[idx];
                        if !has_nulls
                            || (has_nulls && {
                                let mb = mask.unwrap();
                                unsafe { mb.get_unchecked(idx) }
                            })
                        {
                            min_s = min_s.min(x);
                            max_s = max_s.max(x);
                        }
                    }

                    min_s = min_s.min(min_v.reduce_min());
                    max_s = max_s.max(max_v.reduce_max());
                    return Some((min_s, max_s));
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut min = None;
            let mut max = None;
            if !has_nulls {
                for &x in data {
                    match min {
                        None => {
                            min = Some(x);
                            max = Some(x);
                        }
                        Some(m) => {
                            min = Some(m.min(x));
                            max = Some(max.unwrap().max(x));
                        }
                    }
                }
            } else {
                let mb = mask.expect("Mask must be Some if nulls are present");
                for (i, &x) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(i) } {
                        match min {
                            None => {
                                min = Some(x);
                                max = Some(x);
                            }
                            Some(m) => {
                                min = Some(m.min(x));
                                max = Some(max.unwrap().max(x));
                            }
                        }
                    }
                }
            }
            min.zip(max)
        }
    };
}

/// Implements SIMD-enabled floating-point min/max reducers
/// for numeric slices with or without nulls, handling NaN values.
macro_rules! impl_reduce_min_max_float {
    ($fn_name:ident, $ty:ty, $LANES:ident, $SIMD:ident, $MASK:ident) => {
        /// Finds the minimum and maximum values in a floating-point slice.
        ///
        /// Handles NaN values correctly by treating them as missing data.
        /// Uses SIMD operations when available for optimal performance.
        ///
        /// # Arguments
        ///
        /// * `data` - The slice of floating-point values to process
        /// * `mask` - Optional bitmask indicating valid/invalid elements
        /// * `null_count` - Optional count of null values for optimisation
        ///
        /// # Returns
        ///
        /// `Some((min, max))` if any valid values exist, `None` if all values are null/NaN
        #[inline(always)]
        pub fn $fn_name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Option<($ty, $ty)> {
            let has_nulls = has_nulls(null_count, mask);
            let len = data.len();
            if len == 0 {
                return None;
            }

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    const LANES: usize = $LANES;
                    type V = $SIMD<$ty, LANES>;
                    type M = <$ty as SimdElement>::Mask;

                    // Find the first valid value (not null and not NaN)
                    let mut first_valid = None;
                    if !has_nulls {
                        for (idx, &x) in data.iter().enumerate() {
                            if !x.is_nan() {
                                first_valid = Some((idx, x));
                                break;
                            }
                        }
                    } else {
                        let mb = mask.expect("Bitmask must be Some if nulls are present");
                        for (idx, &x) in data.iter().enumerate() {
                            if unsafe { mb.get_unchecked(idx) } && !x.is_nan() {
                                first_valid = Some((idx, x));
                                break;
                            }
                        }
                    }
                    let (mut i, x0) = match first_valid {
                        Some(v) => v,
                        None => return None,
                    };
                    let mut min_s = x0;
                    let mut max_s = x0;
                    i += 1;

                    // SIMD alignment for tail
                    while i < len && (i % LANES != 0) {
                        let x = data[i];
                        if (!has_nulls && !x.is_nan())
                            || (has_nulls && {
                                let mb = mask.unwrap();
                                unsafe { mb.get_unchecked(i) && !x.is_nan() }
                            })
                        {
                            min_s = min_s.min(x);
                            max_s = max_s.max(x);
                        }
                        i += 1;
                    }

                    let mut min_v = V::splat(min_s);
                    let mut max_v = V::splat(max_s);

                    if !has_nulls {
                        while i + LANES <= len {
                            let v = V::from_slice(&data[i..i + LANES]);
                            let valid = !v.is_nan();
                            let v_min = valid.select(v, V::splat(<$ty>::INFINITY));
                            let v_max = valid.select(v, V::splat(<$ty>::NEG_INFINITY));
                            min_v = min_v.simd_min(v_min);
                            max_v = max_v.simd_max(v_max);
                            i += LANES;
                        }
                    } else {
                        let mb = mask.unwrap();
                        let mask_bytes = mb.as_bytes();
                        while i + LANES <= len {
                            let v = V::from_slice(&data[i..i + LANES]);
                            let valid_mask =
                                bitmask_to_simd_mask::<LANES, M>(mask_bytes, i, len) & !v.is_nan();
                            let v_min = valid_mask.select(v, V::splat(<$ty>::INFINITY));
                            let v_max = valid_mask.select(v, V::splat(<$ty>::NEG_INFINITY));
                            min_v = min_v.simd_min(v_min);
                            max_v = max_v.simd_max(v_max);
                            i += LANES;
                        }
                    }

                    for idx in i..len {
                        let x = data[idx];
                        if (!has_nulls && !x.is_nan())
                            || (has_nulls && {
                                let mb = mask.unwrap();
                                unsafe { mb.get_unchecked(idx) && !x.is_nan() }
                            })
                        {
                            min_s = min_s.min(x);
                            max_s = max_s.max(x);
                        }
                    }

                    min_s = min_s.min(min_v.reduce_min());
                    max_s = max_s.max(max_v.reduce_max());
                    return Some((min_s, max_s));
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut min = None;
            let mut max = None;
            if !has_nulls {
                for &x in data {
                    if !x.is_nan() {
                        match min {
                            None => {
                                min = Some(x);
                                max = Some(x);
                            }
                            Some(m) => {
                                min = Some(m.min(x));
                                max = Some(max.unwrap().max(x));
                            }
                        }
                    }
                }
            } else {
                let mb = mask.expect("Bitmask must be Some if nulls are present");
                for (i, &x) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(i) } && !x.is_nan() {
                        match min {
                            None => {
                                min = Some(x);
                                max = Some(x);
                            }
                            Some(m) => {
                                min = Some(m.min(x));
                                max = Some(max.unwrap().max(x));
                            }
                        }
                    }
                }
            }
            min.zip(max)
        }
    };
}

/// Dispatches to the right stat-moment function by type.
macro_rules! stat_moments {
    (f64 => $d:expr, $m:expr, $hn:expr) => {
        stat_moments_f64($d, $m, $hn)
    };
    (f32 => $d:expr, $m:expr, $hn:expr) => {
        stat_moments_f32($d, $m, $hn)
    };
    (i64 => $d:expr, $m:expr, $hn:expr) => {
        stat_moments_i64($d, $m, $hn)
    };
    (u64 => $d:expr, $m:expr, $hn:expr) => {
        stat_moments_u64($d, $m, $hn)
    };
    (i32 => $d:expr, $m:expr, $hn:expr) => {
        stat_moments_i32($d, $m, $hn)
    };
    (u32 => $d:expr, $m:expr, $hn:expr) => {
        stat_moments_u32($d, $m, $hn)
    };
}

/// Generates a variance function using stat moments.
/// Generates both population and sample variance for a numeric type.
macro_rules! impl_variance {
    ($ty:ident, $fn_var:ident) => {
        #[doc = concat!("Returns the variance for `", stringify!($ty), "`; if `sample` is true, uses Bessel's correction (n-1).")]
        #[inline(always)]
        pub fn $fn_var(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
            sample: bool,
        ) -> Option<f64> {
            let (s, s2, n) = stat_moments!($ty => data, mask, null_count);
            if n == 0 || (sample && n < 2) {
                None
            } else if sample {
                Some((s2 - s * s / n as f64) / (n as f64 - 1.0))
            } else {
                Some((s2 - s * s / n as f64) / n as f64)
            }
        }
    };
}

/// Generates min, max, and (min, max) range extractors from a reduce.
macro_rules! impl_min_max_range {
    ($ty:ty, $reduce_fn:ident, $min_fn:ident, $max_fn:ident, $range_fn:ident) => {
        #[doc = concat!("Returns the minimum of a `", stringify!($ty), "` slice.")]
        #[inline(always)]
        pub fn $min_fn(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Option<$ty> {
            $reduce_fn(data, mask, null_count).map(|(min, _)| min)
        }

        #[doc = concat!("Returns the maximum of a `", stringify!($ty), "` slice.")]
        #[inline(always)]
        pub fn $max_fn(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Option<$ty> {
            $reduce_fn(data, mask, null_count).map(|(_, max)| max)
        }

        #[doc = concat!("Returns the (min, max) of a `", stringify!($ty), "` slice.")]
        #[inline(always)]
        pub fn $range_fn(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Option<($ty, $ty)> {
            $reduce_fn(data, mask, null_count)
        }
    };
}

// ---------- 1. SIMD helpers: SUM / COUNT / SUM² ------------------------

/// Implements raw (sum, count) aggregation for float types.
macro_rules! impl_sum_float {
    ($name:ident, $ty:ty, $LANES:expr) => {
        #[doc = concat!("Returns (sum, count) for non-null `", stringify!($ty), "` values.")]
        #[doc = " Uses SIMD where available. Uses Neumaier compensated summation."]
        #[inline(always)]
        pub fn $name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> (f64, usize) {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    const N: usize = $LANES;
                    type V<const L: usize> = Simd<$ty, L>;
                    type M<const L: usize> = <$ty as SimdElement>::Mask;

                    let len = data.len();
                    let mut i = 0;
                    let mut sum = 0.0f64;
                    let mut comp = 0.0f64;
                    let mut cnt = 0usize;

                    if !has_nulls {
                        // Dense path
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            let chunk_sum = v.cast::<f64>().reduce_sum();
                            neumaier_add(&mut sum, &mut comp, chunk_sum);
                            cnt += N;
                            i += N;
                        }
                        for &x in &data[i..] {
                            neumaier_add(&mut sum, &mut comp, x as f64);
                            cnt += 1;
                        }
                    } else {
                        // Null‐aware path
                        let mb = mask.expect(
                            "Bitmask must be Some if nulls are present or mask is supplied",
                        );
                        let bytes = mb.as_bytes();

                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            // extract a SIMD‐mask for these N lanes
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(bytes, i, len);
                            // zero out the null lanes
                            let vv = lane_m.select(v, V::<N>::splat(0.0 as $ty));
                            let chunk_sum = vv.cast::<f64>().reduce_sum();
                            neumaier_add(&mut sum, &mut comp, chunk_sum);
                            // count the 1‐bits in lane_m
                            cnt += lane_m.to_bitmask().count_ones() as usize;
                            i += N;
                        }
                        for idx in i..len {
                            if unsafe { mb.get_unchecked(idx) } {
                                neumaier_add(&mut sum, &mut comp, data[idx] as f64);
                                cnt += 1;
                            }
                        }
                    }

                    return (sum + comp, cnt);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut sum = 0.0f64;
            let mut comp = 0.0f64;
            let mut cnt = 0usize;
            let len = data.len();

            if !has_nulls {
                for &x in data {
                    neumaier_add(&mut sum, &mut comp, x as f64);
                    cnt += 1;
                }
            } else {
                let mb =
                    mask.expect("Bitmask must be Some if nulls are present or mask is supplied");
                for i in 0..len {
                    if unsafe { mb.get_unchecked(i) } {
                        neumaier_add(&mut sum, &mut comp, data[i] as f64);
                        cnt += 1;
                    }
                }
            }

            (sum + comp, cnt)
        }
    };
}

/// Macro to implement raw sum/count aggregation for integer types.
macro_rules! impl_sum_int {
    ($name:ident, $ty:ty, $acc:ty, $LANES:expr) => {
        #[doc = concat!("Returns (sum, count) for non-null `", stringify!($ty), "` values.")]
        #[doc = concat!(" Accumulates using `", stringify!($acc), "`. Uses SIMD if available.")]
        #[inline(always)]
        pub fn $name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> ($acc, usize) {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    const N: usize = $LANES;
                    type V<const L: usize> = Simd<$ty, L>;
                    type M<const L: usize> = <$ty as SimdElement>::Mask;

                    let len = data.len();
                    let mut i = 0;
                    let mut sum: $acc = 0;
                    let mut cnt = 0_usize;

                    if !has_nulls {
                        // dense path
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            sum += v.reduce_sum() as $acc;
                            cnt += N;
                            i += N;
                        }
                        // tail
                        for &x in &data[i..] {
                            sum += x as $acc;
                            cnt += 1;
                        }
                    } else {
                        // null‐aware path
                        let mb = mask.expect(
                            "Bitmask must be Some if nulls are present or mask is supplied",
                        );
                        let bytes = mb.as_bytes();

                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            // extract the SIMD validity mask
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(bytes, i, len);
                            // zero‐out invalid lanes
                            let vv = lane_m.select(v, V::<N>::splat(0 as $ty));
                            sum += vv.reduce_sum() as $acc;
                            // count ones in the mask
                            cnt += lane_m.to_bitmask().count_ones() as usize;
                            i += N;
                        }
                        // scalar tail
                        for idx in i..len {
                            if unsafe { mb.get_unchecked(idx) } {
                                sum += data[idx] as $acc;
                                cnt += 1;
                            }
                        }
                    }

                    return (sum, cnt);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut sum: $acc = 0;
            let mut cnt = 0_usize;
            let len = data.len();

            if !has_nulls {
                for &x in data {
                    sum += x as $acc;
                    cnt += 1;
                }
            } else {
                let mb =
                    mask.expect("Bitmask must be Some if nulls are present or mask is supplied");
                for i in 0..len {
                    if unsafe { mb.get_unchecked(i) } {
                        sum += data[i] as $acc;
                        cnt += 1;
                    }
                }
            }

            (sum, cnt)
        }
    };
}

/// Computes Σx² for `$ty` values, skipping nulls.
/// SIMD‐accelerated when available. Uses Neumaier compensated summation.
macro_rules! impl_sum2_float {
    ($fn_name:ident, $ty:ty, $LANES:expr) => {
        #[doc = concat!("Computes sum-of-squares for `", stringify!($ty), "` (Σx²).")]
        #[doc = " Skips nulls if present. SIMD‐enabled. Neumaier compensated."]
        #[inline(always)]
        pub fn $fn_name(data: &[$ty], mask: Option<&Bitmask>, null_count: Option<usize>) -> f64 {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    // Vector type for the original element type
                    type V<const N: usize> = Simd<$ty, N>;
                    // Mask type comes from the element's SimdElement impl
                    type M<const N: usize> = <$ty as SimdElement>::Mask;

                    const N: usize = $LANES;
                    let len = data.len();
                    let mut i = 0;
                    let mut s2 = 0.0f64;
                    let mut comp = 0.0f64;

                    if !has_nulls {
                        // Dense path: no nulls
                        while i + N <= len {
                            let dv = V::<N>::from_slice(&data[i..i + N]).cast::<f64>();
                            let chunk = (dv * dv).reduce_sum();
                            neumaier_add(&mut s2, &mut comp, chunk);
                            i += N;
                        }
                        // tail
                        for &x in &data[i..] {
                            let y = x as f64;
                            neumaier_add(&mut s2, &mut comp, y * y);
                        }
                    } else {
                        // Null‐aware path
                        let mb = mask.expect("mask must be Some if nulls present");
                        let bytes = mb.as_bytes();
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]);
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(bytes, i, len);
                            let vv = lane_m.select(v, V::<N>::splat(0 as $ty));
                            let dv = vv.cast::<f64>();
                            let chunk = (dv * dv).reduce_sum();
                            neumaier_add(&mut s2, &mut comp, chunk);
                            i += N;
                        }
                        // tail
                        for idx in i..len {
                            if unsafe { mb.get_unchecked(idx) } {
                                let y = data[idx] as f64;
                                neumaier_add(&mut s2, &mut comp, y * y);
                            }
                        }
                    }
                    return s2 + comp;
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut s2 = 0.0f64;
            let mut comp = 0.0f64;
            if !has_nulls {
                for &x in data {
                    let y = x as f64;
                    neumaier_add(&mut s2, &mut comp, y * y);
                }
            } else {
                let mb = mask.expect("mask must be Some if nulls present");
                for (i, &x) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(i) } {
                        let y = x as f64;
                        neumaier_add(&mut s2, &mut comp, y * y);
                    }
                }
            }
            s2 + comp
        }
    };
}

/// Macro to implement null-aware element counting.
macro_rules! impl_count {
    ($fn_name:ident) => {
        #[doc = "Returns the number of valid (non-null) elements."]
        #[inline(always)]
        pub fn $fn_name(len: usize, mask: Option<&Bitmask>, null_count: Option<usize>) -> usize {
            match null_count {
                Some(n) => len - n,
                None => match mask {
                    Some(m) => m.count_ones(),
                    None => len,
                },
            }
        }
    };
}

/// Macro to implement sum and average aggregations for float/int types.
/// Also handles raw (sum, count) cases internally.
macro_rules! impl_sum_avg {
    // floats ---------------------------------------------------
    (float $ty:ident, $sum_fn:ident, $avg_fn:ident, $raw:ident) => {
        #[doc = concat!("Returns the sum of `", stringify!($ty), "` values, or None if all-null.")]
        #[inline(always)]
        pub fn $sum_fn(d: &[$ty], m: Option<&Bitmask>, null_count: Option<usize>) -> Option<f64> {
            let (s, n) = $raw(d, m, null_count);
            if n == 0 { None } else { Some(s) }
        }

        #[doc = concat!("Returns the average of `", stringify!($ty), "` values, or None if all-null.")]
        #[inline(always)]
        pub fn $avg_fn(d: &[$ty], m: Option<&Bitmask>, null_count: Option<usize>) -> Option<f64> {
            let (s, n) = $raw(d, m, null_count);
            if n == 0 { None } else { Some(s / n as f64) }
        }
    };

    // ints -----------------------------------------------------
    (int $ty:ident, $sum_fn:ident, $avg_fn:ident, $acc:ty, $raw:ident) => {
        #[doc = concat!("Returns the sum of `", stringify!($ty), "` values, or None if all-null.")]
        #[inline(always)]
        pub fn $sum_fn(d: &[$ty], m: Option<&Bitmask>, null_count: Option<usize>) -> Option<$acc> {
            let (s, n) = $raw(d, m, null_count);
            if n == 0 { None } else { Some(s) }
        }

        #[doc = concat!("Returns the average of `", stringify!($ty), "` values, or None if all-null.")]
        #[inline(always)]
        pub fn $avg_fn(d: &[$ty], m: Option<&Bitmask>, null_count: Option<usize>) -> Option<f64> {
            let (s, n) = $raw(d, m, null_count);
            if n == 0 { None } else { Some(s as f64 / n as f64) }
        }
    };
}

macro_rules! impl_stat_moments4_float {
    ($fn_name:ident, $ty:ty, $LANES:expr) => {
        #[doc = concat!(
                                    "SIMD-accelerated (∑x … ∑x⁴) for `", stringify!($ty), "`.\n",
                                    "Returns `(s1,s2,s3,s4,n)` as `(f64,f64,f64,f64,usize)`.\n",
                                    "Uses Neumaier compensated summation for all 4 accumulators."
                                )]
        #[inline(always)]
        pub fn $fn_name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> (f64, f64, f64, f64, usize) {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                    type V<const L: usize> = Simd<$ty, L>;
                    type M<const L: usize> = <$ty as SimdElement>::Mask;

                    const N: usize = $LANES;
                    let len = data.len();
                    let mut i = 0usize;

                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    let mut s4 = 0.0;
                    let mut c1 = 0.0;
                    let mut c2 = 0.0;
                    let mut c3 = 0.0;
                    let mut c4 = 0.0;
                    let mut n = 0usize;

                    if !has_nulls {
                        // Dense, no nulls
                        while i + N <= len {
                            let v = V::<N>::from_slice(&data[i..i + N]).cast::<f64>();
                            let v2 = v * v;
                            neumaier_add(&mut s1, &mut c1, v.reduce_sum());
                            neumaier_add(&mut s2, &mut c2, v2.reduce_sum());
                            neumaier_add(&mut s3, &mut c3, (v2 * v).reduce_sum());
                            neumaier_add(&mut s4, &mut c4, (v2 * v2).reduce_sum());
                            n += N;
                            i += N;
                        }
                        for &x in &data[i..] {
                            let xf = x as f64;
                            let x2 = xf * xf;
                            neumaier_add(&mut s1, &mut c1, xf);
                            neumaier_add(&mut s2, &mut c2, x2);
                            neumaier_add(&mut s3, &mut c3, x2 * xf);
                            neumaier_add(&mut s4, &mut c4, x2 * x2);
                            n += 1;
                        }
                    } else {
                        // Null-aware SIMD
                        let mb = mask.expect("mask must be Some when nulls present");
                        let mask_bytes = mb.as_bytes();
                        while i + N <= len {
                            let v_raw = V::<N>::from_slice(&data[i..i + N]);
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                            if !lane_m.any() {
                                i += N;
                                continue;
                            }
                            // Zero out nulls
                            let v = lane_m.select(v_raw, V::<N>::splat(0 as $ty)).cast::<f64>();
                            let v2 = v * v;
                            neumaier_add(&mut s1, &mut c1, v.reduce_sum());
                            neumaier_add(&mut s2, &mut c2, v2.reduce_sum());
                            neumaier_add(&mut s3, &mut c3, (v2 * v).reduce_sum());
                            neumaier_add(&mut s4, &mut c4, (v2 * v2).reduce_sum());
                            n += lane_m.to_bitmask().count_ones() as usize;
                            i += N;
                        }
                        for j in i..len {
                            if unsafe { mb.get_unchecked(j) } {
                                let xf = data[j] as f64;
                                let x2 = xf * xf;
                                neumaier_add(&mut s1, &mut c1, xf);
                                neumaier_add(&mut s2, &mut c2, x2);
                                neumaier_add(&mut s3, &mut c3, x2 * xf);
                                neumaier_add(&mut s4, &mut c4, x2 * x2);
                                n += 1;
                            }
                        }
                    }
                    return (s1 + c1, s2 + c2, s3 + c3, s4 + c4, n);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut s3 = 0.0;
            let mut s4 = 0.0;
            let mut c1 = 0.0;
            let mut c2 = 0.0;
            let mut c3 = 0.0;
            let mut c4 = 0.0;
            let mut n = 0usize;

            if !has_nulls {
                for &x in data {
                    let xf = x as f64;
                    let x2 = xf * xf;
                    neumaier_add(&mut s1, &mut c1, xf);
                    neumaier_add(&mut s2, &mut c2, x2);
                    neumaier_add(&mut s3, &mut c3, x2 * xf);
                    neumaier_add(&mut s4, &mut c4, x2 * x2);
                    n += 1;
                }
            } else {
                let mb = mask.expect("mask must be Some when nulls present");
                for (idx, &x) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(idx) } {
                        let xf = x as f64;
                        let x2 = xf * xf;
                        neumaier_add(&mut s1, &mut c1, xf);
                        neumaier_add(&mut s2, &mut c2, x2);
                        neumaier_add(&mut s3, &mut c3, x2 * xf);
                        neumaier_add(&mut s4, &mut c4, x2 * x2);
                        n += 1;
                    }
                }
            }
            (s1 + c1, s2 + c2, s3 + c3, s4 + c4, n)
        }
    };
}

macro_rules! impl_stat_moments4_int {
    ($fn_name:ident, $ty:ty, $acc:ty, $LANES:expr) => {
        #[doc = concat!(
                    "SIMD-accelerated (∑x … ∑x⁴) for `", stringify!($ty), "` (promoted to f64).\n",
                    "Accumulator type for scalar paths: `", stringify!($acc), "`."
                )]
        #[inline(always)]
        pub fn $fn_name(
            data: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>
        ) -> (f64, f64, f64, f64, usize) {
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(data) {
                type V<const L: usize> = Simd<$ty, L>;
                type M<const L: usize> = <$ty as SimdElement>::Mask;

                const N: usize = $LANES;
                let len = data.len();
                let mut i = 0usize;

                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                let mut s4 = 0.0;
                let mut n = 0usize;

                if !has_nulls {
                    while i + N <= len {
                        let v_i = V::<N>::from_slice(&data[i..i + N]);
                        let v_f = v_i.cast::<f64>();
                        let v2 = v_f * v_f;
                        s1 += v_f.reduce_sum();
                        s2 += v2.reduce_sum();
                        s3 += (v2 * v_f).reduce_sum();
                        s4 += (v2 * v2).reduce_sum();
                        n += N;
                        i += N;
                    }
                    for &x in &data[i..] {
                        let xf = x as f64;
                        s1 += xf;
                        let x2 = xf * xf;
                        s2 += x2;
                        s3 += x2 * xf;
                        s4 += x2 * x2;
                        n += 1;
                    }
                } else {
                    let mb = mask.expect("mask must be Some when nulls present");
                    let mask_bytes = mb.as_bytes();
                    while i + N <= len {
                        let v_i = V::<N>::from_slice(&data[i..i + N]);
                        let lane_m: Mask<M<N>, N> =
                            bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                        if !lane_m.any() {
                            i += N;
                            continue;
                        }
                        let v_masked = lane_m.select(v_i, V::<N>::splat(0));
                        let v_f = v_masked.cast::<f64>();
                        let v2 = v_f * v_f;
                        s1 += v_f.reduce_sum();
                        s2 += v2.reduce_sum();
                        s3 += (v2 * v_f).reduce_sum();
                        s4 += (v2 * v2).reduce_sum();
                        n += lane_m.to_bitmask().count_ones() as usize;
                        i += N;
                    }
                    for j in i..len {
                        if unsafe { mb.get_unchecked(j) } {
                            let xf = data[j] as f64;
                            let x2 = xf * xf;
                            s1 += xf;
                            s2 += x2;
                            s3 += x2 * xf;
                            s4 += x2 * x2;
                            n += 1;
                        }
                    }
                }
                return (s1, s2, s3, s4, n);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            // For int types, accumulate in a wide accumulator, then cast
            let mut s1: $acc = 0;
            let mut s2: $acc = 0;
            let mut s3: $acc = 0;
            let mut s4: $acc = 0;
            let mut n = 0usize;

            if !has_nulls {
                for &x in data {
                    let xi: $acc = x as $acc;
                    let x2 = xi * xi;
                    s1 += xi;
                    s2 += x2;
                    s3 += x2 * xi;
                    s4 += x2 * x2;
                    n += 1;
                }
            } else {
                let mb = mask.expect("mask must be Some when nulls present");
                for (idx, &x) in data.iter().enumerate() {
                    if unsafe { mb.get_unchecked(idx) } {
                        let xi: $acc = x as $acc;
                        let x2 = xi * xi;
                        s1 += xi;
                        s2 += x2;
                        s3 += x2 * xi;
                        s4 += x2 * x2;
                        n += 1;
                    }
                }
            }
            (s1 as f64, s2 as f64, s3 as f64, s4 as f64, n)
        }
    };
}

// Weighted-Sum-2  &  Pair-Stats kernels
macro_rules! impl_weighted_sum2_float {
    ($fn:ident, $ty:ty, $LANES:expr) => {
        #[doc = concat!(
                                    "Returns `(∑w·x, ∑w, ∑w·x², rows)` for `", stringify!($ty),
                                    "` slices `vals`, `wts` with optional null mask.\n",
                                    "Uses Neumaier compensated summation for all 3 accumulators."
                                )]
        #[inline(always)]
        pub fn $fn(
            vals: &[$ty],
            wts: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> (f64, f64, f64, usize) {
            debug_assert_eq!(vals.len(), wts.len(), "weighted_sum2x: len mismatch");
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(vals) && is_simd_aligned(wts) {
                    type V<const L: usize> = Simd<$ty, L>;
                    type M<const L: usize> = <$ty as SimdElement>::Mask;
                    const N: usize = $LANES;

                    let len = vals.len();
                    let mut i = 0usize;
                    let mut swx = 0.0;
                    let mut sw = 0.0;
                    let mut swx2 = 0.0;
                    let mut cwx = 0.0;
                    let mut cw = 0.0;
                    let mut cwx2 = 0.0;
                    let mut n = 0usize;

                    if !has_nulls {
                        while i + N <= len {
                            let v = V::<N>::from_slice(&vals[i..i + N]).cast::<f64>();
                            let w = V::<N>::from_slice(&wts[i..i + N]).cast::<f64>();
                            neumaier_add(&mut swx, &mut cwx, (v * w).reduce_sum());
                            neumaier_add(&mut sw, &mut cw, w.reduce_sum());
                            neumaier_add(&mut swx2, &mut cwx2, (w * v * v).reduce_sum());
                            n += N;
                            i += N;
                        }
                        for j in i..len {
                            let vf = vals[j] as f64;
                            let wf = wts[j] as f64;
                            neumaier_add(&mut swx, &mut cwx, wf * vf);
                            neumaier_add(&mut sw, &mut cw, wf);
                            neumaier_add(&mut swx2, &mut cwx2, wf * vf * vf);
                            n += 1;
                        }
                    } else {
                        let mb = mask.expect("mask = Some when nulls present");
                        let mask_bytes = mb.as_bytes();
                        while i + N <= len {
                            let v_raw = V::<N>::from_slice(&vals[i..i + N]);
                            let w_raw = V::<N>::from_slice(&wts[i..i + N]);
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                            if !lane_m.any() {
                                i += N;
                                continue;
                            }
                            let v = lane_m.select(v_raw, V::<N>::splat(0.0)).cast::<f64>();
                            let w = lane_m.select(w_raw, V::<N>::splat(0.0)).cast::<f64>();
                            neumaier_add(&mut swx, &mut cwx, (v * w).reduce_sum());
                            neumaier_add(&mut sw, &mut cw, w.reduce_sum());
                            neumaier_add(&mut swx2, &mut cwx2, (w * v * v).reduce_sum());
                            n += lane_m.to_bitmask().count_ones() as usize;
                            i += N;
                        }
                        for j in i..len {
                            if unsafe { mb.get_unchecked(j) } {
                                let vf = vals[j] as f64;
                                let wf = wts[j] as f64;
                                neumaier_add(&mut swx, &mut cwx, wf * vf);
                                neumaier_add(&mut sw, &mut cw, wf);
                                neumaier_add(&mut swx2, &mut cwx2, wf * vf * vf);
                                n += 1;
                            }
                        }
                    }
                    return (swx + cwx, sw + cw, swx2 + cwx2, n);
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut swx = 0.0;
            let mut sw = 0.0;
            let mut swx2 = 0.0;
            let mut cwx = 0.0;
            let mut cw = 0.0;
            let mut cwx2 = 0.0;
            let mut n = 0usize;

            if !has_nulls {
                for i in 0..vals.len() {
                    let vf = vals[i] as f64;
                    let wf = wts[i] as f64;
                    neumaier_add(&mut swx, &mut cwx, wf * vf);
                    neumaier_add(&mut sw, &mut cw, wf);
                    neumaier_add(&mut swx2, &mut cwx2, wf * vf * vf);
                    n += 1;
                }
            } else {
                let mb = mask.expect("mask = Some when nulls present");
                for i in 0..vals.len() {
                    if unsafe { mb.get_unchecked(i) } {
                        let vf = vals[i] as f64;
                        let wf = wts[i] as f64;
                        neumaier_add(&mut swx, &mut cwx, wf * vf);
                        neumaier_add(&mut sw, &mut cw, wf);
                        neumaier_add(&mut swx2, &mut cwx2, wf * vf * vf);
                        n += 1;
                    }
                }
            }
            (swx + cwx, sw + cw, swx2 + cwx2, n)
        }
    };
}

/// Statistical summary for paired data analysis.
///
/// Accumulates fundamental statistical quantities for bivariate data pairs,
/// providing the building blocks for correlation analysis, regression coefficients,
/// and covariance calculations with SIMD-accelerated computation.
///
/// # Fields
/// - **`n`**: Count of valid (non-null) data pairs
/// - **`sx`**: Sum of x-values (Σx)
/// - **`sy`**: Sum of y-values (Σy)
/// - **`sxy`**: Sum of products (Σxy) for covariance calculation
/// - **`sx2`**: Sum of x-squared values (Σx²) for variance calculation
/// - **`sy2`**: Sum of y-squared values (Σy²) for variance calculation
///
/// # Applications
/// These statistics enable efficient calculation of:
/// - **Pearson correlation coefficient**: r = (n⋅Σxy - Σx⋅Σy) / √[(n⋅Σx² - (Σx)²)(n⋅Σy² - (Σy)²)]
/// - **Linear regression slope**: β₁ = (n⋅Σxy - Σx⋅Σy) / (n⋅Σx² - (Σx)²)
/// - **Sample covariance**: cov(x,y) = (Σxy - n⋅x̄⋅ȳ) / (n-1)
/// - **Coefficient of determination**: R² for regression analysis
#[derive(Default)]
pub struct PairStats {
    /// Count of valid (non-null) paired observations
    pub n: usize,
    /// Sum of x-values (Σx)
    pub sx: f64,
    /// Sum of y-values (Σy)
    pub sy: f64,
    /// Sum of cross-products (Σxy) for covariance computation
    pub sxy: f64,
    /// Sum of squared x-values (Σx²) for variance computation
    pub sx2: f64,
    /// Sum of squared y-values (Σy²) for variance computation
    pub sy2: f64,
}

// PAIR - Σ stats (x,y)  (float)
macro_rules! impl_pair_stats_float {
    ($fn:ident, $ty:ty, $LANES:expr) => {
        /// Returns PairStats (n, sum_x, sum_y, sum_xy, sum_x2, sum_y2) for floating point types.
        /// Uses Neumaier compensated summation for all 5 accumulators.
        #[inline(always)]
        pub fn $fn(
            xs: &[$ty],
            ys: &[$ty],
            mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> PairStats {
            debug_assert_eq!(xs.len(), ys.len(), "pair_stats: len mismatch ");
            let has_nulls = has_nulls(null_count, mask);

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(xs) && is_simd_aligned(ys) {
                    type V<const L: usize> = Simd<$ty, L>;
                    type M<const L: usize> = <$ty as SimdElement>::Mask;
                    const N: usize = $LANES;
                    let len = xs.len();

                    let mut i = 0usize;
                    let mut sx = 0.0;
                    let mut sy = 0.0;
                    let mut sxy = 0.0;
                    let mut sx2 = 0.0;
                    let mut sy2 = 0.0;
                    let mut cx = 0.0;
                    let mut cy = 0.0;
                    let mut cxy = 0.0;
                    let mut cx2 = 0.0;
                    let mut cy2 = 0.0;
                    let mut n = 0usize;

                    if !has_nulls {
                        while i + N <= len {
                            let vx = V::<N>::from_slice(&xs[i..i + N]).cast::<f64>();
                            let vy = V::<N>::from_slice(&ys[i..i + N]).cast::<f64>();

                            neumaier_add(&mut sx, &mut cx, vx.reduce_sum());
                            neumaier_add(&mut sy, &mut cy, vy.reduce_sum());
                            neumaier_add(&mut sxy, &mut cxy, (vx * vy).reduce_sum());
                            neumaier_add(&mut sx2, &mut cx2, (vx * vx).reduce_sum());
                            neumaier_add(&mut sy2, &mut cy2, (vy * vy).reduce_sum());
                            n += N;
                            i += N;
                        }
                        for j in i..len {
                            let xf = xs[j] as f64;
                            let yf = ys[j] as f64;
                            neumaier_add(&mut sx, &mut cx, xf);
                            neumaier_add(&mut sy, &mut cy, yf);
                            neumaier_add(&mut sxy, &mut cxy, xf * yf);
                            neumaier_add(&mut sx2, &mut cx2, xf * xf);
                            neumaier_add(&mut sy2, &mut cy2, yf * yf);
                            n += 1;
                        }
                    } else {
                        let mb = mask.expect("mask Some when nulls present");
                        let mask_bytes = mb.as_bytes();

                        while i + N <= len {
                            let vx_raw = V::<N>::from_slice(&xs[i..i + N]);
                            let vy_raw = V::<N>::from_slice(&ys[i..i + N]);
                            let lane_m: Mask<M<N>, N> =
                                bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                            if !lane_m.any() {
                                i += N;
                                continue;
                            }

                            let vx = lane_m.select(vx_raw, V::<N>::splat(0.0)).cast::<f64>();
                            let vy = lane_m.select(vy_raw, V::<N>::splat(0.0)).cast::<f64>();

                            neumaier_add(&mut sx, &mut cx, vx.reduce_sum());
                            neumaier_add(&mut sy, &mut cy, vy.reduce_sum());
                            neumaier_add(&mut sxy, &mut cxy, (vx * vy).reduce_sum());
                            neumaier_add(&mut sx2, &mut cx2, (vx * vx).reduce_sum());
                            neumaier_add(&mut sy2, &mut cy2, (vy * vy).reduce_sum());
                            n += lane_m.to_bitmask().count_ones() as usize;
                            i += N;
                        }
                        for j in i..len {
                            if unsafe { mb.get_unchecked(j) } {
                                let xf = xs[j] as f64;
                                let yf = ys[j] as f64;
                                neumaier_add(&mut sx, &mut cx, xf);
                                neumaier_add(&mut sy, &mut cy, yf);
                                neumaier_add(&mut sxy, &mut cxy, xf * yf);
                                neumaier_add(&mut sx2, &mut cx2, xf * xf);
                                neumaier_add(&mut sy2, &mut cy2, yf * yf);
                                n += 1;
                            }
                        }
                    }
                    return PairStats {
                        n,
                        sx: sx + cx,
                        sy: sy + cy,
                        sxy: sxy + cxy,
                        sx2: sx2 + cx2,
                        sy2: sy2 + cy2,
                    };
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut sxy = 0.0;
            let mut sx2 = 0.0;
            let mut sy2 = 0.0;
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut cxy = 0.0;
            let mut cx2 = 0.0;
            let mut cy2 = 0.0;
            let mut n = 0usize;

            if !has_nulls {
                for i in 0..xs.len() {
                    let xf = xs[i] as f64;
                    let yf = ys[i] as f64;
                    neumaier_add(&mut sx, &mut cx, xf);
                    neumaier_add(&mut sy, &mut cy, yf);
                    neumaier_add(&mut sxy, &mut cxy, xf * yf);
                    neumaier_add(&mut sx2, &mut cx2, xf * xf);
                    neumaier_add(&mut sy2, &mut cy2, yf * yf);
                    n += 1;
                }
            } else {
                let mb = mask.expect("mask Some when nulls present");
                for i in 0..xs.len() {
                    if unsafe { mb.get_unchecked(i) } {
                        let xf = xs[i] as f64;
                        let yf = ys[i] as f64;
                        neumaier_add(&mut sx, &mut cx, xf);
                        neumaier_add(&mut sy, &mut cy, yf);
                        neumaier_add(&mut sxy, &mut cxy, xf * yf);
                        neumaier_add(&mut sx2, &mut cx2, xf * xf);
                        neumaier_add(&mut sy2, &mut cy2, yf * yf);
                        n += 1;
                    }
                }
            }
            PairStats {
                n,
                sx: sx + cx,
                sy: sy + cy,
                sxy: sxy + cxy,
                sx2: sx2 + cx2,
                sy2: sy2 + cy2,
            }
        }
    };
}

// 4-moment running-update helper (scalar). Works for any Into<f64> value.
#[inline(always)]
fn moments4_scalar<I>(iter: I) -> (usize, f64, f64, f64, f64)
where
    I: IntoIterator<Item = f64>,
{
    let mut n = 0usize;
    let mut m1 = 0.0; // mean
    let mut m2 = 0.0; // Σ(x-μ)²
    let mut m3 = 0.0; // Σ(x-μ)³
    let mut m4 = 0.0; // Σ(x-μ)⁴

    for x in iter {
        n += 1;
        let n_f = n as f64;
        let delta = x - m1;
        let delta_n = delta / n_f;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n_f - 1.0);

        m4 += term1 * delta_n2 * (n_f * n_f - 3.0 * n_f + 3.0) + 6.0 * delta_n2 * m2
            - 4.0 * delta_n * m3;

        m3 += term1 * delta_n * (n_f - 2.0) - 3.0 * delta_n * m2;
        m2 += term1;
        m1 += delta_n;
    }
    (n, m1, m2, m3, m4)
}

//  Skewness & Kurtosis — float
/// Implements skewness and kurtosis calculation functions for floating-point types.
macro_rules! impl_skew_kurt_float {
    ($ty:ident, $skew_fn:ident, $kurt_fn:ident) => {
        /// Calculates the skewness of a floating-point dataset.
        ///
        /// Skewness measures the asymmetry of the probability distribution.
        /// Positive skewness indicates a longer tail on the right side,
        /// negative skewness indicates a longer tail on the left side.
        ///
        /// # Arguments
        ///
        /// * `d` - The slice of floating-point values
        /// * `m` - Optional bitmask for null value handling
        /// * `null_count` - Optional count of null values
        /// * `adjust_sample_bias` - Whether to apply sample bias correction
        ///
        /// # Returns
        ///
        /// `Some(skewness)` if calculation is possible, `None` if insufficient data
        #[inline(always)]
        pub fn $skew_fn(
            d: &[$ty],
            m: Option<&Bitmask>,
            null_count: Option<usize>,
            adjust_sample_bias: bool,
        ) -> Option<f64> {
            let vals: Vec<f64> = if has_nulls(null_count, m) {
                collect_valid(d, m).into_iter().map(|x| x as f64).collect()
            } else {
                d.iter().map(|&x| x as f64).collect()
            };

            let (n, _m1, m2, m3, _) = moments4_scalar(vals);
            if n < 3 || m2 == 0.0 {
                return None;
            }
            let n_f = n as f64;

            // population skewness (fisher-style)
            let pop = n_f.sqrt() * m3 / m2.powf(1.5);
            if !adjust_sample_bias {
                return Some(pop);
            }

            // unbiased sample skewness (Joanes & Gill)
            Some(n_f * (n_f - 1.0).sqrt() * m3 / ((n_f - 2.0) * m2.powf(1.5)))
        }

        /// Calculates the kurtosis of a floating-point dataset.
        ///
        /// Kurtosis measures the "tailedness" of the probability distribution.
        /// Higher kurtosis indicates more extreme outliers, lower kurtosis
        /// indicates a distribution with lighter tails.
        ///
        /// # Arguments
        ///
        /// * `d` - The slice of floating-point values
        /// * `m` - Optional bitmask for null value handling
        /// * `null_count` - Optional count of null values
        /// * `adjust_sample_bias` - Whether to apply sample bias correction
        ///
        /// # Returns
        ///
        /// `Some(kurtosis)` if calculation is possible, `None` if insufficient data
        #[inline(always)]
        pub fn $kurt_fn(
            d: &[$ty],
            m: Option<&Bitmask>,
            null_count: Option<usize>,
            adjust_sample_bias: bool,
        ) -> Option<f64> {
            let vals: Vec<f64> = if has_nulls(null_count, m) {
                collect_valid(d, m).into_iter().map(|x| x as f64).collect()
            } else {
                d.iter().map(|&x| x as f64).collect()
            };

            let (n, _m1, m2, _m3, m4) = moments4_scalar(vals);
            if n < 2 || m2 == 0.0 {
                return None;
            }
            let n_f = n as f64;
            let pop_excess = n_f * m4 / (m2 * m2) - 3.0;
            if !adjust_sample_bias {
                return Some(pop_excess);
            }

            // n < 4 -> unbiased formula undefined; fall back to population value
            if n < 4 {
                return Some(pop_excess);
            }

            // unbiased sample excess kurtosis (Joanes & Gill)
            let numerator = (n_f + 1.0) * n_f * (m4 / (m2 * m2)) - 3.0 * (n_f - 1.0);
            let denominator = (n_f - 2.0) * (n_f - 3.0);
            Some(numerator * (n_f - 1.0) / denominator)
        }
    };
}

//  Skewness & Kurtosis — Integer
/// Implements skewness and kurtosis calculation functions for integer types.
macro_rules! impl_skew_kurt_int {
    ($ty:ident, $skew_fn:ident, $kurt_fn:ident) => {
        /// Calculates the skewness of an integer dataset.
        ///
        /// Skewness measures the asymmetry of the probability distribution.
        /// Values are converted to f64 for calculation to maintain precision.
        ///
        /// # Arguments
        ///
        /// * `d` - The slice of integer values
        /// * `m` - Optional bitmask for null value handling
        /// * `null_count` - Optional count of null values
        /// * `adjust_sample_bias` - Whether to apply sample bias correction
        ///
        /// # Returns
        ///
        /// `Some(skewness)` if calculation is possible, `None` if insufficient data
        #[inline(always)]
        pub fn $skew_fn(
            d: &[$ty],
            m: Option<&Bitmask>,
            null_count: Option<usize>,
            adjust_sample_bias: bool,
        ) -> Option<f64> {
            let vals: Vec<f64> = if has_nulls(null_count, m) {
                collect_valid(d, m).into_iter().map(|x| x as f64).collect()
            } else {
                d.iter().map(|&x| x as f64).collect()
            };

            let (n, _m1, m2, m3, _) = moments4_scalar(vals);
            if n < 3 || m2 == 0.0 {
                return None;
            }
            let n_f = n as f64;

            let pop = n_f.sqrt() * m3 / m2.powf(1.5);
            if !adjust_sample_bias {
                return Some(pop);
            }

            Some(n_f * (n_f - 1.0).sqrt() * m3 / ((n_f - 2.0) * m2.powf(1.5)))
        }

        /// Calculates the kurtosis of an integer dataset.
        ///
        /// Kurtosis measures the "tailedness" of the probability distribution.
        /// Values are converted to f64 for calculation to maintain precision.
        ///
        /// # Arguments
        ///
        /// * `d` - The slice of integer values
        /// * `m` - Optional bitmask for null value handling
        /// * `null_count` - Optional count of null values
        /// * `adjust_sample_bias` - Whether to apply sample bias correction
        ///
        /// # Returns
        ///
        /// `Some(kurtosis)` if calculation is possible, `None` if insufficient data
        #[inline(always)]
        pub fn $kurt_fn(
            d: &[$ty],
            m: Option<&Bitmask>,
            null_count: Option<usize>,
            adjust_sample_bias: bool,
        ) -> Option<f64> {
            let vals: Vec<f64> = if has_nulls(null_count, m) {
                collect_valid(d, m).into_iter().map(|x| x as f64).collect()
            } else {
                d.iter().map(|&x| x as f64).collect()
            };

            let (n, _m1, m2, _m3, m4) = moments4_scalar(vals);
            if n < 2 || m2 == 0.0 {
                return None;
            }
            let n_f = n as f64;

            let pop_excess = n_f * m4 / (m2 * m2) - 3.0;
            if !adjust_sample_bias {
                return Some(pop_excess);
            }
            if n < 4 {
                return Some(pop_excess);
            }

            let numerator = (n_f + 1.0) * n_f * (m4 / (m2 * m2)) - 3.0 * (n_f - 1.0);
            let denominator = (n_f - 2.0) * (n_f - 3.0);
            Some(numerator * (n_f - 1.0) / denominator)
        }
    };
}

/// Computes the sum of squares (Σx²).
/// Kernel: SumP2 (L1-2)
/// Uses Neumaier compensated summation for numerical accuracy.
#[inline(always)]
pub fn sum_squares(v: &[f64], mask: Option<&Bitmask>, null_count: Option<usize>) -> f64 {
    let has_nulls = has_nulls(null_count, mask);

    #[cfg(feature = "simd")]
    if is_simd_aligned(v) {
        type V<const L: usize> = Simd<f64, L>;
        type M<const L: usize> = <f64 as SimdElement>::Mask;
        const N: usize = W64;
        let len = v.len();
        let mut i = 0;
        let mut acc = 0.0f64;
        let mut comp = 0.0f64;
        if !has_nulls {
            while i + N <= len {
                let simd_v = V::<N>::from_slice(&v[i..i + N]);
                let chunk = (simd_v * simd_v).reduce_sum();
                neumaier_add(&mut acc, &mut comp, chunk);
                i += N;
            }
            for &x in &v[i..] {
                neumaier_add(&mut acc, &mut comp, x * x);
            }
        } else {
            let mb = mask.expect("sum_squares: mask must be Some when nulls are present");
            let mask_bytes = mb.as_bytes();
            while i + N <= len {
                let simd_v = V::<N>::from_slice(&v[i..i + N]);
                let lane_m: Mask<M<N>, N> = bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                let simd_v_masked = lane_m.select(simd_v, V::<N>::splat(0.0));
                let chunk = (simd_v_masked * simd_v_masked).reduce_sum();
                neumaier_add(&mut acc, &mut comp, chunk);
                i += N;
            }
            for j in i..len {
                if unsafe { mb.get_unchecked(j) } {
                    neumaier_add(&mut acc, &mut comp, v[j] * v[j]);
                }
            }
        }
        return acc + comp;
    }

    // Scalar fallback - alignment check failed or no simd flag
    let mut acc = 0.0f64;
    let mut comp = 0.0f64;
    if !has_nulls {
        for &x in v {
            neumaier_add(&mut acc, &mut comp, x * x);
        }
    } else {
        let mb = mask.expect("sum_squares: mask must be Some when nulls are present");
        for (j, &x) in v.iter().enumerate() {
            if unsafe { mb.get_unchecked(j) } {
                neumaier_add(&mut acc, &mut comp, x * x);
            }
        }
    }
    acc + comp
}

/// Computes the sum of cubes (Σx³).
/// Kernel: SumP3 (L1-3)
/// Uses Neumaier compensated summation for numerical accuracy.
#[inline(always)]
pub fn sum_cubes(v: &[f64], mask: Option<&Bitmask>, null_count: Option<usize>) -> f64 {
    let has_nulls = has_nulls(null_count, mask);

    #[cfg(feature = "simd")]
    if is_simd_aligned(v) {
        type V<const L: usize> = Simd<f64, L>;
        type M<const L: usize> = <f64 as SimdElement>::Mask;
        const N: usize = W64;
        let len = v.len();
        let mut i = 0;
        let mut acc = 0.0f64;
        let mut comp = 0.0f64;
        let mut found_infinite = false;

        if !has_nulls {
            while i + N <= len {
                let simd_v = V::<N>::from_slice(&v[i..i + N]);
                let simd_v2 = simd_v * simd_v;
                let simd_cubes = simd_v2 * simd_v;
                // SIMD horizontal check for infinities
                let infinite_mask = simd_cubes.is_infinite();
                if infinite_mask.any() {
                    found_infinite = true;
                }
                neumaier_add(&mut acc, &mut comp, simd_cubes.reduce_sum());
                i += N;
            }
            for &x in &v[i..] {
                let cube = x * x * x;
                if cube.is_infinite() {
                    found_infinite = true;
                }
                neumaier_add(&mut acc, &mut comp, cube);
            }
        } else {
            let mb = mask.expect("sum_cubes: mask must be Some when nulls are present");
            let mask_bytes = mb.as_bytes();
            while i + N <= len {
                let simd_v = V::<N>::from_slice(&v[i..i + N]);
                let lane_m: Mask<M<N>, N> = bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                let simd_v_masked = lane_m.select(simd_v, V::<N>::splat(0.0));
                let simd_v2 = simd_v_masked * simd_v_masked;
                let simd_cubes = simd_v2 * simd_v_masked;
                let infinite_mask = simd_cubes.is_infinite();
                if infinite_mask.any() {
                    found_infinite = true;
                }
                neumaier_add(&mut acc, &mut comp, simd_cubes.reduce_sum());
                i += N;
            }
            for j in i..len {
                if unsafe { mb.get_unchecked(j) } {
                    let cube = v[j] * v[j] * v[j];
                    if cube.is_infinite() {
                        found_infinite = true;
                    }
                    neumaier_add(&mut acc, &mut comp, cube);
                }
            }
        }
        let result = acc + comp;
        return if result.is_nan() && found_infinite {
            f64::INFINITY
        } else {
            result
        };
    }

    // Scalar fallback - alignment check failed or no simd flag
    let mut acc = 0.0f64;
    let mut comp = 0.0f64;
    let mut found_infinite = false;
    if !has_nulls {
        for &x in v {
            let cube = x * x * x;
            if cube.is_infinite() {
                found_infinite = true;
            }
            neumaier_add(&mut acc, &mut comp, cube);
        }
    } else {
        let mb = mask.expect("sum_cubes: mask must be Some when nulls are present");
        for (j, &x) in v.iter().enumerate() {
            if unsafe { mb.get_unchecked(j) } {
                let cube = x * x * x;
                if cube.is_infinite() {
                    found_infinite = true;
                }
                neumaier_add(&mut acc, &mut comp, cube);
            }
        }
    }
    let result = acc + comp;
    if result.is_nan() && found_infinite {
        f64::INFINITY
    } else {
        result
    }
}

/// Computes the sum of fourth powers (Σx⁴).
/// Kernel: SumP4 (L1-4)
/// Uses Neumaier compensated summation for numerical accuracy.
#[inline(always)]
pub fn sum_quartics(v: &[f64], mask: Option<&Bitmask>, null_count: Option<usize>) -> f64 {
    let has_nulls = has_nulls(null_count, mask);

    #[cfg(feature = "simd")]
    if is_simd_aligned(v) {
        type V<const L: usize> = Simd<f64, L>;
        type M<const L: usize> = <f64 as SimdElement>::Mask;
        const N: usize = W64;
        let len = v.len();
        let mut i = 0;
        let mut acc = 0.0f64;
        let mut comp = 0.0f64;
        if !has_nulls {
            while i + N <= len {
                let simd_v = V::<N>::from_slice(&v[i..i + N]);
                let simd_v2 = simd_v * simd_v;
                let chunk = (simd_v2 * simd_v2).reduce_sum();
                neumaier_add(&mut acc, &mut comp, chunk);
                i += N;
            }
            for &x in &v[i..] {
                let x2 = x * x;
                neumaier_add(&mut acc, &mut comp, x2 * x2);
            }
        } else {
            let mb = mask.expect("sum_quartics: mask must be Some when nulls are present");
            let mask_bytes = mb.as_bytes();
            while i + N <= len {
                let simd_v = V::<N>::from_slice(&v[i..i + N]);
                let lane_m: Mask<M<N>, N> = bitmask_to_simd_mask::<N, M<N>>(mask_bytes, i, len);
                let simd_v_masked = lane_m.select(simd_v, V::<N>::splat(0.0));
                let simd_v2 = simd_v_masked * simd_v_masked;
                let chunk = (simd_v2 * simd_v2).reduce_sum();
                neumaier_add(&mut acc, &mut comp, chunk);
                i += N;
            }
            for j in i..len {
                if unsafe { mb.get_unchecked(j) } {
                    let x2 = v[j] * v[j];
                    neumaier_add(&mut acc, &mut comp, x2 * x2);
                }
            }
        }
        return acc + comp;
    }

    // Scalar fallback - alignment check failed or no simd flag
    let mut acc = 0.0f64;
    let mut comp = 0.0f64;
    if !has_nulls {
        for &x in v {
            let x2 = x * x;
            neumaier_add(&mut acc, &mut comp, x2 * x2);
        }
    } else {
        let mb = mask.expect("sum_quartics: mask must be Some when nulls are present");
        for (j, &x) in v.iter().enumerate() {
            if unsafe { mb.get_unchecked(j) } {
                let x2 = x * x;
                neumaier_add(&mut acc, &mut comp, x2 * x2);
            }
        }
    }
    acc + comp
}

// Macro impls per type

impl_weighted_sum2_float!(weighted_sum2_f64, f64, W64);
impl_weighted_sum2_float!(weighted_sum2_f32, f32, W32);

impl_pair_stats_float!(pair_stats_f64, f64, W64);
impl_pair_stats_float!(pair_stats_f32, f32, W32);

impl_stat_moments4_float!(stat_moments4_f64, f64, W64);
impl_stat_moments4_float!(stat_moments4_f32, f32, W32);

impl_stat_moments4_int!(stat_moments4_i64, i64, i128, W64);
impl_stat_moments4_int!(stat_moments4_u64, u64, u128, W64);
impl_stat_moments4_int!(stat_moments4_i32, i32, i128, W32);
impl_stat_moments4_int!(stat_moments4_u32, u32, u128, W32);

impl_stat_moments_float!(stat_moments_f64, f64, W64);
impl_stat_moments_float!(stat_moments_f32, f32, W32);
impl_stat_moments_int!(stat_moments_i64, i64, i64, W64, i64);
impl_stat_moments_int!(stat_moments_u64, u64, u64, W64, i64);
impl_stat_moments_int!(stat_moments_i32, i32, i64, W32, i32);
impl_stat_moments_int!(stat_moments_u32, u32, u64, W32, i32);
impl_reduce_min_max!(reduce_min_max_i64, i64, W64, i64::MAX, i64::MIN);
impl_reduce_min_max!(reduce_min_max_u64, u64, W64, u64::MAX, u64::MIN);
impl_reduce_min_max!(reduce_min_max_i32, i32, W32, i32::MAX, i32::MIN);
impl_reduce_min_max!(reduce_min_max_u32, u32, W32, u32::MAX, u32::MIN);
impl_reduce_min_max_float!(reduce_min_max_f64, f64, W64, Simd, Mask);
impl_reduce_min_max_float!(reduce_min_max_f32, f32, W32, Simd, Mask);

impl_min_max_range!(f64, reduce_min_max_f64, min_f64, max_f64, range_f64);
impl_min_max_range!(f32, reduce_min_max_f32, min_f32, max_f32, range_f32);
impl_min_max_range!(i64, reduce_min_max_i64, min_i64, max_i64, range_i64);
impl_min_max_range!(u64, reduce_min_max_u64, min_u64, max_u64, range_u64);
impl_min_max_range!(i32, reduce_min_max_i32, min_i32, max_i32, range_i32);
impl_min_max_range!(u32, reduce_min_max_u32, min_u32, max_u32, range_u32);

impl_sum_avg!(float f64, sum_f64, average_f64, sum_f64_raw);
impl_sum_avg!(float f32, sum_f32, average_f32, sum_f32_raw);

impl_sum_avg!(int  i64, sum_i64, average_i64, i64, sum_i64_raw);
impl_sum_avg!(int  u64, sum_u64, average_u64, u64, sum_u64_raw);
impl_sum_avg!(int  i32, sum_i32, average_i32, i64 , sum_i32_raw);
impl_sum_avg!(int  u32, sum_u32, average_u32, u64 , sum_u32_raw);

impl_variance!(f64, variance_f64);
impl_variance!(f32, variance_f32);
impl_variance!(i64, variance_i64);
impl_variance!(u64, variance_u64);
impl_variance!(i32, variance_i32);
impl_variance!(u32, variance_u32);

impl_sum_float!(sum_f64_raw, f64, W64);
impl_sum_float!(sum_f32_raw, f32, W32);
impl_sum_int!(sum_i64_raw, i64, i64, W64);
impl_sum_int!(sum_u64_raw, u64, u64, W64);
impl_sum_int!(sum_i32_raw, i32, i64, W32);
impl_sum_int!(sum_u32_raw, u32, u64, W32);

impl_sum2_float!(sum2_f64_raw, f64, W64);
impl_sum2_float!(sum2_f32_raw, f32, W32);

impl_count!(count_generic_raw);

impl_skew_kurt_float!(f64, skewness_f64, kurtosis_f64);
impl_skew_kurt_float!(f32, skewness_f32, kurtosis_f32);

impl_skew_kurt_int!(i64, skewness_i64, kurtosis_i64);
impl_skew_kurt_int!(u64, skewness_u64, kurtosis_u64);
impl_skew_kurt_int!(i32, skewness_i32, kurtosis_i32);
impl_skew_kurt_int!(u32, skewness_u32, kurtosis_u32);

/// Computes median of a sorted/unsorted slice with optional nulls.
#[inline(always)]
pub fn median<T: Ord + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    sort: bool,
) -> Option<T> {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    // collect valid values
    let mut v = if !has_nulls {
        Vec64::from_slice(d)
    } else {
        collect_valid(d, m)
    };

    if v.is_empty() {
        return None;
    }

    let len = v.len();
    let mid = len / 2;

    if len % 2 == 1 {
        // odd: middle element
        if sort {
            let (_, nth, _) = v.select_nth_unstable(mid);
            Some(*nth)
        } else {
            Some(v[mid])
        }
    } else {
        // even: keep the *lower* median when sorted,
        // but for the unsorted case pick the element at index mid
        if sort {
            let (_, nth, _) = v.select_nth_unstable(mid - 1);
            Some(*nth)
        } else {
            Some(v[mid])
        }
    }
}

/// Computes median for float types, with optional nulls and sorting.
#[inline(always)]
pub fn median_f<T: Float + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    sort: bool,
) -> Option<T> {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let mut v = if !has_nulls {
        Vec64::from_slice(d)
    } else {
        collect_valid(d, m)
    };

    if v.is_empty() {
        return None;
    }

    let len = v.len();
    let mid = len / 2;

    if !sort {
        return Some(if len & 1 == 0 {
            (v[mid - 1] + v[mid]) / T::from(2.0).unwrap()
        } else {
            v[mid]
        });
    }

    if len & 1 == 1 {
        let (_, nth, _) = v.select_nth_unstable_by(mid, total_cmp_f);
        Some(*nth)
    } else {
        let (_, nth_hi, _) = v.select_nth_unstable_by(mid, total_cmp_f);
        let hi = *nth_hi;
        let (_, nth_lo, _) = v.select_nth_unstable_by(mid - 1, total_cmp_f);
        let lo = *nth_lo;
        Some((lo + hi) / T::from(2.0).unwrap())
    }
}
/// Computes percentile for ordinal types, optionally sorted.
#[inline(always)]
pub fn percentile_ord<T: Ord + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    p: f64,
    sort: bool,
) -> Option<T> {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let mut v = if !has_nulls {
        Vec64::from_slice(d)
    } else {
        collect_valid(d, m)
    };

    if v.is_empty() {
        return None;
    }

    let idx = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;

    if !sort {
        return v.get(idx).copied();
    }

    let (_, nth, _) = v.select_nth_unstable(idx);
    Some(*nth)
}

/// Computes percentile for float types, optionally sorted.
#[inline(always)]
pub fn percentile_f<T: Float + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    p: f64,
    sort: bool,
) -> Option<T> {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let mut v = if !has_nulls {
        Vec64::from_slice(d)
    } else {
        collect_valid(d, m)
    };

    if v.is_empty() {
        return None;
    }

    let idx = (p.clamp(0.0, 1.0) * (v.len() - 1) as f64).round() as usize;

    if !sort {
        return v.get(idx).copied();
    }

    let (_, nth, _) = v.select_nth_unstable_by(idx, total_cmp_f);
    Some(*nth)
}

/// Computes `q` quantiles for ordinal types, optionally sorted.
#[inline(always)]
pub fn quantile<T: Ord + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    q: usize,
    sort: bool,
) -> Option<Vec64<T>> {
    if q < 2 {
        return None;
    }

    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let mut v = if !has_nulls {
        Vec64::from_slice(d)
    } else {
        collect_valid(d, m)
    };

    let n = v.len();
    if n == 0 {
        return None;
    }

    if sort {
        v.sort_unstable();
    }

    let mut out = Vec64::with_capacity(q - 1);
    unsafe {
        out.set_len(q - 1);
    }

    for k in 1..q {
        let p = k as f64 / q as f64;
        let h = 1.0 + (n as f64 - 1.0) * p;
        let idx = h.floor() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);
        unsafe {
            *out.get_unchecked_mut(k - 1) = v[idx];
        }
    }

    Some(out)
}

/// Computes `q` quantiles for float types, optionally sorted.
#[inline(always)]
pub fn quantile_f<T: Float + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    q: usize,
    sort: bool,
) -> Option<Vec64<T>> {
    if q < 2 {
        return None;
    }

    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let mut v = if !has_nulls {
        Vec64::from_slice(d)
    } else {
        collect_valid(d, m)
    };

    let n = v.len();
    if n == 0 {
        return None;
    }

    if sort {
        sort_float(&mut v);
    }

    let mut out = Vec64::with_capacity(q - 1);
    unsafe {
        out.set_len(q - 1);
    }

    for k in 1..q {
        let p = k as f64 / q as f64;
        let h = 1.0 + (n as f64 - 1.0) * p;
        let hf = h.floor();
        let hc = h.ceil();

        let idx_lo = (hf as usize).saturating_sub(1).min(n - 1);
        let idx_hi = (hc as usize).saturating_sub(1).min(n - 1);

        let value = if idx_lo == idx_hi {
            v[idx_lo]
        } else {
            let weight = T::from(h - hf).unwrap();
            let v_lo = v[idx_lo];
            let v_hi = v[idx_hi];
            v_lo + (v_hi - v_lo) * weight
        };

        unsafe {
            *out.get_unchecked_mut(k - 1) = value;
        }
    }

    Some(out)
}

/// Computes interquartile range for ordinal types.
#[inline(always)]
pub fn iqr<T>(d: &[T], m: Option<&Bitmask>, null_count: Option<usize>, s: bool) -> Option<T>
where
    T: Ord + Copy + Sub<Output = T>,
{
    Some(percentile_ord(d, m, null_count, 75.0, s)? - percentile_ord(d, m, null_count, 25.0, s)?)
}

/// Computes interquartile range for float types.
#[inline(always)]
pub fn iqr_f<T: Float + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
    s: bool,
) -> Option<T> {
    Some(percentile_f(d, m, null_count, 0.75, s)? - percentile_f(d, m, null_count, 0.25, s)?)
}

/// Computes mode (most frequent value) for ordinal types.
#[inline(always)]
pub fn mode<T: Eq + Hash + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Option<T> {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    #[cfg(feature = "fast_hash")]
    let mut f = AHashMap::with_capacity(d.len());
    #[cfg(not(feature = "fast_hash"))]
    let mut f = HashMap::with_capacity(d.len());

    if !has_nulls {
        for &v in d {
            *f.entry(v).or_insert(0usize) += 1;
        }
    } else {
        let mb = m.expect("mode: nulls present but mask is None");
        for (i, &v) in d.iter().enumerate() {
            if unsafe { mb.get_unchecked(i) } {
                *f.entry(v).or_insert(0usize) += 1;
            }
        }
    }

    f.into_iter().max_by_key(|e| e.1).map(|e| e.0)
}

/// Computes mode for float types via bit-level equivalence.
#[inline(always)]
pub fn mode_f<T: ToBits + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Option<T> {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    #[cfg(feature = "fast_hash")]
    let mut f: AHashMap<T::Bits, (usize, T)> = AHashMap::with_capacity(d.len());
    #[cfg(not(feature = "fast_hash"))]
    let mut f: HashMap<T::Bits, (usize, T)> = HashMap::with_capacity(d.len());

    if !has_nulls {
        for &v in d {
            let e = f.entry(v.to_bits()).or_insert((0, v));
            e.0 += 1;
        }
    } else {
        let mb = m.expect("mode_f: nulls present but mask is None");
        for (i, &v) in d.iter().enumerate() {
            if unsafe { mb.get_unchecked(i) } {
                let e = f.entry(v.to_bits()).or_insert((0, v));
                e.0 += 1;
            }
        }
    }

    f.into_values().max_by_key(|x| x.0).map(|x| x.1)
}

/// Counts distinct values in an ordinal slice, with optional nulls.
#[inline(always)]
pub fn count_distinct<T: Eq + Hash + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> usize {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    if !has_nulls {
        d.iter().collect::<HashSet<_>>().len()
    } else {
        let mb = m.expect("count_distinct: nulls present but mask is None");
        d.iter()
            .enumerate()
            .filter(|(i, _)| unsafe { mb.get_unchecked(*i) })
            .map(|(_, v)| v)
            .collect::<HashSet<_>>()
            .len()
    }
}

/// Counts distinct values in a float slice using bitwise equality.
#[inline(always)]
pub fn count_distinct_f<T: Float + ToBits + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> usize {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    if !has_nulls {
        d.iter().map(|v| v.to_bits()).collect::<HashSet<_>>().len()
    } else {
        let mb = m.expect("count_distinct_f: nulls present but mask is None");
        d.iter()
            .enumerate()
            .filter(|(i, _)| unsafe { mb.get_unchecked(*i) })
            .map(|(_, v)| v.to_bits())
            .collect::<HashSet<_>>()
            .len()
    }
}

/// Computes harmonic mean for integer types.
#[inline(always)]
pub fn harmonic_mean_int<T: NumCast + Copy + PartialOrd>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> f64 {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let v: Vec<f64> = if !has_nulls {
        d.iter().map(|&x| NumCast::from(x).unwrap()).collect()
    } else {
        collect_valid(d, m)
            .into_iter()
            .map(|x| NumCast::from(x).unwrap())
            .collect()
    };

    if v.is_empty() {
        panic!("harmonic_mean_int: input data is empty");
    }
    if v.iter().any(|&x| x <= 0.0) {
        panic!("harmonic_mean_int: non-positive values are invalid");
    }

    let n = v.len() as f64;
    let denom: f64 = v.iter().map(|&x| 1.0 / x).sum();

    if denom == 0.0 {
        panic!("harmonic_mean_int: denominator is zero");
    }

    n / denom
}

/// Computes harmonic mean for unsigned integers.
#[inline(always)]
pub fn harmonic_mean_uint<T: NumCast + Copy + PartialOrd>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> f64 {
    harmonic_mean_int(d, m, null_count)
}

/// Computes harmonic mean for float types.
#[inline(always)]
pub fn harmonic_mean_f<T: Float + Into<f64> + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> f64 {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let (mut s, mut n) = (0.0, 0usize);

    if !has_nulls {
        for &v in d {
            let x = v.into();
            if x <= 0.0 {
                panic!("harmonic_mean_f: non-positive values are invalid");
            }
            s += 1.0 / x;
            n += 1;
        }
    } else {
        let mb = m.expect("harmonic_mean_f: nulls present but mask is None");
        for (i, &v) in d.iter().enumerate() {
            if unsafe { mb.get_unchecked(i) } {
                let x = v.into();
                if x <= 0.0 {
                    panic!("harmonic_mean_f: non-positive values are invalid");
                }
                s += 1.0 / x;
                n += 1;
            }
        }
    }

    if n == 0 {
        panic!("harmonic_mean_f: input data is empty");
    }
    if s == 0.0 {
        panic!("harmonic_mean_f: denominator is zero");
    }

    n as f64 / s
}

/// Computes geometric mean for integer types.
#[inline(always)]
pub fn geometric_mean_int<T: ToPrimitive + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> f64 {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let (mut s, mut n) = (0.0, 0usize);
    if !has_nulls {
        for &v in d.iter() {
            let val = v.to_f64().unwrap();
            if val <= 0.0 {
                panic!("geometric_mean_int: non-positive values are invalid");
            }
            s += val.ln();
            n += 1;
        }
    } else {
        let mb = m.expect("geometric_mean_int: nulls present but mask is None");
        for (i, &v) in d.iter().enumerate() {
            if unsafe { mb.get_unchecked(i) } {
                let val = v.to_f64().unwrap();
                if val <= 0.0 {
                    panic!("geometric_mean_int: non-positive values are invalid");
                }
                s += val.ln();
                n += 1;
            }
        }
    }

    if n == 0 {
        panic!("geometric_mean_int: input data is empty");
    }

    (s / n as f64).exp()
}

/// Computes geometric mean for unsigned integer types.
#[inline(always)]
pub fn geometric_mean_uint<T: ToPrimitive + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> f64 {
    geometric_mean_int(d, m, null_count)
}

/// Computes geometric mean for float types.
#[inline(always)]
pub fn geometric_mean_f<T: Float + Into<f64> + Copy>(
    d: &[T],
    m: Option<&Bitmask>,
    null_count: Option<usize>,
) -> f64 {
    let has_nulls = match null_count {
        Some(n) => n > 0,
        None => m.is_some(),
    };

    let (mut s, mut n) = (0.0, 0usize);
    if !has_nulls {
        for &v in d.iter() {
            let val = v.into();
            if val <= 0.0 {
                panic!("geometric_mean_f: non-positive values are invalid");
            }
            s += val.ln();
            n += 1;
        }
    } else {
        let mb = m.expect("geometric_mean_f: nulls present but mask is None");
        for (i, &v) in d.iter().enumerate() {
            if unsafe { mb.get_unchecked(i) } {
                let val = v.into();
                if val <= 0.0 {
                    panic!("geometric_mean_f: non-positive values are invalid");
                }
                s += val.ln();
                n += 1;
            }
        }
    }

    if n == 0 {
        panic!("geometric_mean_f: input data is empty");
    }

    (s / n as f64).exp()
}

/// Returns the accumulating product of all values
#[inline(always)]
pub fn agg_product<T: Copy + One + Mul<Output = T> + Zero + PartialEq + 'static>(
    data: &[T],
    mask: Option<&Bitmask>,
    offset: usize,
    len: usize,
) -> T {
    let mut prod = T::one();
    match mask {
        Some(mask) => {
            let mask = mask.slice_clone(offset, len);
            for (i, &x) in data[offset..offset + len].iter().enumerate() {
                if unsafe { mask.get_unchecked(i) } {
                    prod = prod * x;
                    if prod == T::zero() {
                        break;
                    }
                }
            }
        }
        None => {
            for &x in &data[offset..offset + len] {
                prod = prod * x;
                if prod == T::zero() {
                    break;
                }
            }
        }
    }
    prod
}

#[cfg(test)]
mod tests {
    use minarrow::{Vec64, vec64};
    use num_traits::Float;

    use super::*;

    // Build Arrow-style validity mask from bools (least significant bit is index 0)
    fn mask_from_bools(bits: &[bool]) -> Bitmask {
        Bitmask::from_bools(bits)
    }

    // Compare floats robustly (for variance etc)
    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn test_stat_moments_f64_dense() {
        let data = vec64![1.0, 2.0, 3.0, 4.0];
        let (sum, sum2, count) = stat_moments_f64(&data, None, None);
        assert_eq!(sum, 10.0);
        assert_eq!(sum2, 30.0);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_stat_moments_f64_masked() {
        let data = vec64![1.0, 2.0, 3.0, 4.0];
        let mask = mask_from_bools(&[true, false, true, false]);
        let (sum, sum2, count) = stat_moments_f64(&data, Some(&mask), Some(2));
        assert_eq!(sum, 4.0);
        assert_eq!(sum2, 10.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_stat_moments_i32_dense() {
        let data = vec64![1i32, 2, 3, 4];
        let (sum, sum2, count) = stat_moments_i32(&data, None, None);
        assert_eq!(sum, 10.0);
        assert_eq!(sum2, 30.0);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_stat_moments_i32_masked() {
        let data = vec64![1i32, 2, 3, 4];
        let mask = mask_from_bools(&[true, false, false, true]);
        let (sum, sum2, count) = stat_moments_i32(&data, Some(&mask), Some(2));
        assert_eq!(sum, 5.0);
        assert_eq!(sum2, 17.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_min_max_range_f64_dense() {
        let data = vec64![5.0, 3.0, 1.0, 7.0];
        assert_eq!(min_f64(&data, None, None), Some(1.0));
        assert_eq!(max_f64(&data, None, None), Some(7.0));
        assert_eq!(range_f64(&data, None, None), Some((1.0, 7.0)));
    }

    #[test]
    fn test_min_max_range_f64_masked() {
        let data = vec64![5.0, 3.0, 1.0, 7.0];
        let mask = mask_from_bools(&[false, true, true, false]);
        assert_eq!(min_f64(&data, Some(&mask), Some(2)), Some(1.0));
        assert_eq!(max_f64(&data, Some(&mask), Some(2)), Some(3.0));
        assert_eq!(range_f64(&data, Some(&mask), Some(2)), Some((1.0, 3.0)));
    }

    #[test]
    fn test_sum_and_avg_i32() {
        let data = vec64![1i32, 2, 3, 4];
        assert_eq!(sum_i32(&data, None, None), Some(10));
        assert_eq!(average_i32(&data, None, None), Some(2.5));
        let mask = mask_from_bools(&[true, false, true, true]);
        assert_eq!(sum_i32(&data, Some(&mask), Some(1)), Some(8));
        assert_eq!(average_i32(&data, Some(&mask), Some(1)), Some(8.0 / 3.0));
    }

    #[test]
    fn test_sum_and_avg_f32() {
        let data = vec64![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(sum_f32(&data, None, None), Some(10.0));
        assert_eq!(average_f32(&data, None, None), Some(2.5));
        let mask = mask_from_bools(&[false, true, true, false]);
        assert_eq!(sum_f32(&data, Some(&mask), Some(2)), Some(5.0));
        assert_eq!(average_f32(&data, Some(&mask), Some(2)), Some(2.5));
    }

    #[test]
    fn test_variance_f64() {
        let data = vec64![1.0, 2.0, 3.0, 4.0];
        let v = variance_f64(&data, None, None, false);
        assert!(approx_eq(v.unwrap(), 1.25, 1e-9));
        let mask = mask_from_bools(&[true, true, false, false]);
        let v2 = variance_f64(&data, Some(&mask), Some(2), false);
        assert!(approx_eq(v2.unwrap(), 0.25, 1e-9));
    }

    #[test]
    fn test_count_generic_raw() {
        let n = 100;
        let mask = mask_from_bools(&[true; 100]);
        assert_eq!(count_generic_raw(n, None, None), 100);
        assert_eq!(count_generic_raw(n, Some(&mask), Some(0)), 100);

        let mask = mask_from_bools(&(0..n).map(|i| i % 2 == 0).collect::<Vec<_>>());
        assert_eq!(count_generic_raw(n, Some(&mask), Some(50)), 50);
    }

    #[test]
    fn test_median_ord_sorted() {
        let data = vec64![1i32, 4, 2, 3];
        assert_eq!(median(&data, None, None, true), Some(2));
    }

    #[test]
    fn test_median_ord_unsorted() {
        let data = vec64![1i32, 4, 2, 3];
        // unsorted, even-length ⇒ index mid = 2 ⇒ value 2
        assert_eq!(median(&data, None, None, false), Some(2));
    }

    #[test]
    fn test_median_ord_masked() {
        let data = vec64![1i32, 4, 2, 3];
        let mask = mask_from_bools(&[true, true, false, false]);
        assert_eq!(median(&data, Some(&mask), Some(2), true), Some(1));
    }

    #[test]
    fn test_median_f_sorted() {
        let dataf = vec64![1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(median_f(&dataf, None, None, true), Some(2.5));
    }

    #[test]
    fn test_median_f_masked() {
        let dataf = vec64![1.0f64, 2.0, 3.0, 4.0];
        let mask = mask_from_bools(&[true, false, true, false]);
        assert_eq!(median_f(&dataf, Some(&mask), Some(2), true), Some(2.0));
    }

    #[test]
    fn test_percentile_ord_and_f() {
        let data = vec64![10i32, 20, 30, 40, 50];
        assert_eq!(percentile_ord(&data, None, None, 0.0, true), Some(10));
        assert_eq!(percentile_ord(&data, None, None, 100.0, true), Some(50));
        assert_eq!(percentile_ord(&data, None, None, 50.0, true), Some(30));
        let mask = mask_from_bools(&[false, true, true, false, true]);
        assert_eq!(
            percentile_ord(&data, Some(&mask), Some(2), 50.0, true),
            Some(30)
        );

        let dataf = vec64![1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_f(&dataf, None, None, 0.0, true), Some(1.0));
        assert_eq!(percentile_f(&dataf, None, None, 1.0, true), Some(5.0));
        assert_eq!(percentile_f(&dataf, None, None, 0.5, true), Some(3.0));
    }

    #[test]
    fn test_quantile_ord_and_f() {
        let data = vec64![1i32, 2, 3, 4, 5];
        let out = quantile(&data, None, None, 4, true).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out, Vec64::from_slice(&[2, 3, 4]));

        let mask = mask_from_bools(&[true, false, true, true, false]);
        let out2 = quantile(&data, Some(&mask), Some(2), 3, true).unwrap();
        assert_eq!(out2, Vec64::from_slice(&[1, 3]));

        let dataf = vec64![10.0f64, 20.0, 30.0, 40.0, 50.0];
        let out = quantile_f(&dataf, None, None, 4, true).unwrap();
        assert_eq!(out.len(), 3);
        assert!(approx_eq(out[1], 30.0, 1e-9));
    }

    #[test]
    fn test_iqr_ord() {
        let data = vec64![1i32, 2, 3, 4, 5];
        assert_eq!(iqr(&data, None, None, true), Some(2));
    }

    #[test]
    fn test_iqr_f64() {
        let dataf = vec64![10.0f64, 20.0, 30.0, 40.0, 50.0];
        // p*(n−1) rounding definition, Q3=40, Q1=20 -> IQR=20
        assert_eq!(iqr_f(&dataf, None, None, true), Some(20.0));
    }
    #[test]
    fn test_mode_ord_and_f() {
        let data = vec64![1i32, 2, 3, 2, 2, 4, 5, 1];
        assert_eq!(mode(&data, None, None), Some(2));
        let dataf = vec64![1.0f32, 2.0, 2.0, 3.0, 3.0, 3.0];
        assert_eq!(mode_f(&dataf, None, None), Some(3.0));
        let data = vec64![1i32, 2, 3, 3, 2, 1];
        let m = mode(&data, None, None);
        assert!(m == Some(1) || m == Some(2) || m == Some(3));
    }

    #[test]
    fn test_count_distinct_and_f() {
        let data = vec64![1i32, 2, 2, 3, 3, 4];
        assert_eq!(count_distinct(&data, None, None), 4);
        let dataf = vec64![1.0f32, 2.0, 2.0, 3.0, 3.0, 4.0];
        assert_eq!(count_distinct_f(&dataf, None, None), 4);
    }

    #[test]
    fn test_harmonic_and_geometric_mean() {
        let data = vec64![1.0f64, 2.0, 4.0];
        let hm = harmonic_mean_f(&data, None, None);
        assert!(approx_eq(hm, 1.714285714, 1e-6));

        let gm = geometric_mean_f(&data, None, None);
        assert!(approx_eq(gm, 2.0, 1e-6));

        let data = vec64![1i32, 2, 4];
        let hm = harmonic_mean_int(&data, None, None);
        assert!(approx_eq(hm, 1.714285714, 1e-6));

        let gm = geometric_mean_int(&data, None, None);
        assert!(approx_eq(gm, 2.0, 1e-6));
    }

    #[test]
    fn test_empty_arrays() {
        assert_eq!(stat_moments_f64(&[], None, None), (0.0, 0.0, 0));
        assert_eq!(stat_moments_i32(&[], None, None), (0.0, 0.0, 0));
        assert_eq!(sum_f64(&[], None, None), None);
        assert_eq!(average_i32(&[], None, None), None);
        assert_eq!(variance_f64(&[], None, None, false), None);
        assert_eq!(median::<i32>(&[], None, None, true), None);
        assert_eq!(median_f::<f64>(&[], None, None, true), None);
        assert_eq!(mode::<i32>(&[], None, None), None);
        assert_eq!(mode_f::<f32>(&[], None, None), None);
        assert_eq!(quantile::<i32>(&[], None, None, 3, true), None);
        assert_eq!(quantile_f::<f64>(&[], None, None, 3, true), None);
        assert_eq!(iqr::<i32>(&[], None, None, true), None);
        assert_eq!(iqr_f::<f64>(&[], None, None, true), None);
        assert_eq!(count_distinct::<i32>(&[], None, None), 0);
        assert_eq!(count_distinct_f::<f32>(&[], None, None), 0);
    }

    // --- i64, u64, f32, u32: moments and basic stats (dense/masked) ---
    #[test]
    fn test_stat_moments_all_types() {
        let i64_data = vec64![-8i64, 2, 8, -2, 3];
        let mask = mask_from_bools(&[true, false, true, true, true]);
        let (sum, sum2, count) = stat_moments_i64(&i64_data, Some(&mask), Some(1));
        assert_eq!(sum, (-8.0) + 8.0 + (-2.0) + 3.0);
        assert_eq!(sum2, 64.0 + 64.0 + 4.0 + 9.0);
        assert_eq!(count, 4);

        let u64_data = vec64![1u64, 2, 3, 4, 5];
        let mask = mask_from_bools(&[false, true, true, true, false]);
        let (sum, sum2, count) = stat_moments_u64(&u64_data, Some(&mask), Some(2));
        assert_eq!(sum, 2.0 + 3.0 + 4.0);
        assert_eq!(sum2, 4.0 + 9.0 + 16.0);
        assert_eq!(count, 3);

        let f32_data = vec64![0.5f32, 2.0, 4.5, -2.0, 3.0];
        let mask = mask_from_bools(&[true, true, false, false, true]);
        let (sum, sum2, count) = stat_moments_f32(&f32_data, Some(&mask), Some(2));
        assert!(approx_eq(sum, 0.5 + 2.0 + 3.0, 1e-6));
        assert!(approx_eq(sum2, 0.25 + 4.0 + 9.0, 1e-6));
        assert_eq!(count, 3);

        let u32_data = vec64![100u32, 200, 300, 400, 500];
        let mask = mask_from_bools(&[true, false, false, true, true]);
        let (sum, sum2, count) = stat_moments_u32(&u32_data, Some(&mask), Some(2));
        assert_eq!(sum, 100.0 + 400.0 + 500.0);
        assert_eq!(sum2, 10000.0 + 160000.0 + 250000.0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_reduce_min_max_for_all_types() {
        let i64_data = vec64![-8i64, 2, 8, -2, 3];
        assert_eq!(reduce_min_max_i64(&i64_data, None, None), Some((-8, 8)));
        let mask = mask_from_bools(&[true, false, false, false, true]);
        assert_eq!(
            reduce_min_max_i64(&i64_data, Some(&mask), Some(3)),
            Some((-8, 3))
        );

        let u64_data = vec64![1u64, 2, 3, 4, 5];
        assert_eq!(reduce_min_max_u64(&u64_data, None, None), Some((1, 5)));
        let mask = mask_from_bools(&[false, false, true, true, false]);
        assert_eq!(
            reduce_min_max_u64(&u64_data, Some(&mask), Some(3)),
            Some((3, 4))
        );

        let i32_data = vec64![9i32, -2, 5, -2];
        assert_eq!(reduce_min_max_i32(&i32_data, None, None), Some((-2, 9)));
        let mask = mask_from_bools(&[false, false, true, false]);
        assert_eq!(
            reduce_min_max_i32(&i32_data, Some(&mask), Some(3)),
            Some((5, 5))
        );

        let u32_data = vec64![7u32, 2, 5, 11];
        assert_eq!(reduce_min_max_u32(&u32_data, None, None), Some((2, 11)));
        let mask = mask_from_bools(&[true, false, false, true]);
        assert_eq!(
            reduce_min_max_u32(&u32_data, Some(&mask), Some(2)),
            Some((7, 11))
        );

        let f64_data = vec64![f64::NAN, 3.0, 1.0, 7.0];
        assert_eq!(reduce_min_max_f64(&f64_data, None, None), Some((1.0, 7.0))); // NaN ignored
    }

    #[test]
    fn test_empty_and_all_null_min_max() {
        let empty: [i32; 0] = [];
        assert_eq!(reduce_min_max_i32(&empty, None, None), None);

        let all_nulls = vec64![5i32, 10, 15];
        let mask = mask_from_bools(&[false, false, false]);
        assert_eq!(reduce_min_max_i32(&all_nulls, Some(&mask), Some(3)), None);
    }

    // --- sum/avg: all types, empty and all-null ---
    #[test]
    fn test_sum_avg_and_variance_all_types() {
        // u32
        let data = vec64![100u32, 200, 0, 400];
        assert_eq!(sum_u32(&data, None, None), Some(700));
        assert_eq!(average_u32(&data, None, None), Some(175.0));
        let mask = mask_from_bools(&[false, true, false, true]);
        assert_eq!(sum_u32(&data, Some(&mask), Some(2)), Some(600));
        assert_eq!(average_u32(&data, Some(&mask), Some(2)), Some(300.0));
        assert_eq!(
            variance_u32(&data, None, None, false).unwrap(),
            ((100f64.powi(2) + 200f64.powi(2) + 400f64.powi(2)) / 4.0 - (700.0 / 4.0).powi(2))
        );

        // i64
        let data = vec64![5i64, -10, 25, 5];
        assert_eq!(sum_i64(&data, None, None), Some(25));
        assert_eq!(average_i64(&data, None, None), Some(6.25));
        let mask = mask_from_bools(&[true, false, false, false]);
        assert_eq!(sum_i64(&data, Some(&mask), Some(3)), Some(5));
        assert_eq!(average_i64(&data, Some(&mask), Some(3)), Some(5.0));
        assert!(variance_i64(&data, None, None, false).unwrap() > 0.0);

        // u64
        let data = vec64![10u64, 0, 10, 10];
        assert_eq!(sum_u64(&data, None, None), Some(30));
        assert_eq!(average_u64(&data, None, None), Some(7.5));
        let mask = mask_from_bools(&[false, false, false, false]);
        assert_eq!(sum_u64(&data, Some(&mask), Some(4)), None);
        assert_eq!(average_u64(&data, Some(&mask), Some(4)), None);

        // f32
        let data = vec64![2.0f32, 4.0, 6.0];
        assert_eq!(sum_f32(&data, None, None), Some(12.0));
        assert_eq!(average_f32(&data, None, None), Some(4.0));
        let mask = mask_from_bools(&[true, false, true]);
        assert_eq!(sum_f32(&data, Some(&mask), Some(1)), Some(8.0));
        assert_eq!(average_f32(&data, Some(&mask), Some(1)), Some(4.0));
        assert!(variance_f32(&data, None, None, false).unwrap() > 0.0);

        // f64
        let data = vec64![1.0f64, f64::INFINITY, f64::NAN];
        assert!(
            sum_f64(&data, None, None).unwrap().is_nan()
                || sum_f64(&data, None, None).unwrap().is_infinite()
        );
    }

    // --- min/max/range: singleton, duplicates, negative, NaN, INF ---
    #[test]
    fn test_min_max_range_i32_singleton() {
        let d = vec64![1i32];
        assert_eq!(min_i32(&d, None, None), Some(1));
        assert_eq!(max_i32(&d, None, None), Some(1));
        assert_eq!(range_i32(&d, None, None), Some((1, 1)));
    }

    #[test]
    fn test_min_f64_nan() {
        let d = vec64![5.0f64, f64::NAN];
        // we ignore NaNs, so the min of [5.0, NaN] is 5.0
        assert_eq!(min_f64(&d, None, None), Some(5.0));
    }

    #[test]
    fn test_max_f64_nan() {
        let d = vec64![5.0f64, f64::NAN];
        // we ignore NaNs, so the max of [5.0, NaN] is 5.0
        assert_eq!(max_f64(&d, None, None), Some(5.0));
    }

    #[test]
    fn test_min_f32_neg_inf() {
        let d = vec64![f32::NEG_INFINITY, 0.0f32, 5.0f32];
        assert_eq!(min_f32(&d, None, None), Some(f32::NEG_INFINITY));
    }

    #[test]
    fn test_max_f32_regular() {
        let d = vec64![f32::NEG_INFINITY, 0.0f32, 5.0f32];
        assert_eq!(max_f32(&d, None, None), Some(5.0));
    }

    #[test]
    fn test_median_ord_even() {
        let d = vec64![2i64, 4, 6, 8];
        // sorted ⇒ [2,4,6,8], lower median is 4
        // Keep integers as integers - with users able
        // to cast to floats as needed
        assert_eq!(median(&d, None, None, true), Some(4));
    }

    #[test]
    fn test_median_ord_odd() {
        let d = vec64![7i32, 3, 5];
        assert_eq!(median(&d, None, None, true), Some(5));
    }

    #[test]
    fn test_median_ord_with_mask() {
        let d = vec64![7i32, 3, 5];
        let mask = mask_from_bools(&[false, true, false]);
        assert_eq!(median(&d, Some(&mask), Some(2), true), Some(3));
    }

    #[test]
    fn test_quantile_ord() {
        let d = vec64![10u32, 20, 30, 40, 50, 60];
        let q = quantile(&d, None, None, 3, true).unwrap();
        assert_eq!(q.len(), 2);
        assert!(q[0] >= 20 && q[1] >= 40);
    }

    #[test]
    fn test_median_f64() {
        let df = vec64![1.0f64, 5.0, 3.0, 2.0];
        assert!(approx_eq(
            median_f(&df, None, None, true).unwrap(),
            2.5,
            1e-9
        ));
    }

    #[test]
    fn test_quantile_f64() {
        let df = vec64![1.0f64, 5.0, 3.0, 2.0];
        let qf = quantile_f(&df, None, None, 3, true).unwrap();
        // we produce exactly q−1 = 2 interior tertile values
        assert_eq!(qf.len(), 2);
        assert!(qf.iter().all(|&v| v >= 1.0 && v <= 5.0));
    }

    #[test]
    fn test_iqr_i32() {
        let d = vec64![1i32, 3, 2, 8, 4, 5];
        assert!(iqr(&d, None, None, true).unwrap() >= 2);
    }

    #[test]
    fn test_iqr_f32() {
        let d = vec64![10.0f32, 100.0, 30.0, 50.0];
        assert!(iqr_f(&d, None, None, true).unwrap() > 0.0);
    }

    // --- mode & count_distinct: ties, all unique, all equal, NaN/Inf ---
    #[test]
    fn test_mode_and_count_distinct_exhaustive() {
        // Mode: all unique
        let d = vec64![7u64, 5, 3, 1];
        assert!(d.contains(&mode(&d, None, None).unwrap()));

        // Mode: all equal
        let d = vec64![2i64; 6];
        assert_eq!(mode(&d, None, None), Some(2));

        // Mode: tie
        let d = vec64![3i32, 5, 3, 5];
        let m = mode(&d, None, None);
        assert!(m == Some(3) || m == Some(5));

        // Mode: NaN in floats, inf, repeated
        let d = vec64![f32::NAN, f32::INFINITY, 5.0, 5.0, f32::NAN];
        let m = mode_f(&d, None, None);
        assert!(m == Some(5.0) || m.unwrap().is_nan() || m.unwrap().is_infinite());

        // Count distinct: all equal, all unique, mask
        let d = vec64![1u64, 1, 1, 1];
        assert_eq!(count_distinct(&d, None, None), 1);

        let d = vec64![1i32, 2, 3, 4, 5];
        assert_eq!(count_distinct(&d, None, None), 5);

        let mask = mask_from_bools(&[true, false, false, true, true]);
        assert_eq!(count_distinct(&d, Some(&mask), Some(2)), 3);

        let d = vec64![f64::NAN, f64::INFINITY, 0.0];
        assert_eq!(count_distinct_f(&d, None, None), 3);
    }

    // --- harmonic & geometric means: all types, zero/negative/NaN ---
    #[test]
    fn test_harmonic_mean_f64() {
        let data = vec64![0.5f64, 2.0, 1.0];
        // the true harmonic mean is 3/(2 + 0.5 + 1) ≈ 0.8571
        let hm = harmonic_mean_f(&data, None, None);
        assert!(approx_eq(hm, 0.857142857, 1e-2));
    }

    #[test]
    fn test_geometric_mean_f64() {
        let data = vec64![0.5f64, 2.0, 1.0];
        let gm = geometric_mean_f(&data, None, None);
        assert!(approx_eq(gm, 1.0, 1e-9));
    }

    #[test]
    #[should_panic(expected = "harmonic_mean_int: non-positive values are invalid")]
    fn test_harmonic_mean_int_all_zeros() {
        let d = vec64![0i32, 0, 0];
        // Panics due to non-positive values.
        let _ = harmonic_mean_int(&d, None, None);
    }

    #[test]
    fn test_geometric_mean_uint() {
        let data = vec64![2u64, 8, 4];
        let gm = geometric_mean_uint(&data, None, None);
        assert!(approx_eq(gm, 4.0, 1e-9));
    }

    #[test]
    #[should_panic(expected = "geometric_mean_int: non-positive values are invalid")]
    fn test_geometric_mean_int_negative() {
        // Should panic on negative input, even if result is complex
        let d = vec64![-2i64, -4, -8];
        let _ = geometric_mean_int(&d, None, None);
    }

    // --- percentile: boundary and empty, floats with mask, p out of range ---
    #[test]
    fn test_percentile_ord_and_f_all_cases() {
        let d = vec64![1i32, 4, 6, 8, 10];
        assert_eq!(percentile_ord(&d, None, None, 0.0, true), Some(1));
        assert_eq!(percentile_ord(&d, None, None, 100.0, true), Some(10));
        let mask = mask_from_bools(&[false, false, false, false, false]);
        assert_eq!(percentile_ord(&d, Some(&mask), Some(5), 50.0, true), None);

        let df = vec64![1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_f(&df, None, None, 0.0, true), Some(1.0));
        assert_eq!(percentile_f(&df, None, None, 1.0, true), Some(5.0));
        let mask = mask_from_bools(&[false, false, false, false, false]);
        assert_eq!(percentile_f(&df, Some(&mask), Some(5), 0.5, true), None);

        // Out of range p (should clamp)
        assert_eq!(percentile_f(&df, None, None, 10.0, true), Some(5.0));
        assert_eq!(percentile_f(&df, None, None, -5.0, true), Some(1.0));
    }

    // --- quantile: small q, empty, all-null, mask with only one valid ---
    #[test]
    fn test_quantile_f_and_ord_special_cases() {
        let d = vec64![2.0f32, 4.0, 6.0, 8.0];
        // q < 2 (should return None)
        assert_eq!(quantile_f(&d, None, None, 1, true), None);
        // all nulls
        let mask = mask_from_bools(&[false, false, false, false]);
        assert_eq!(quantile_f(&d, Some(&mask), Some(4), 2, true), None);

        // mask: only one valid
        let mask = mask_from_bools(&[false, false, true, false]);
        let out = quantile_f(&d, Some(&mask), Some(3), 2, true).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out.iter().all(|&x| x == 6.0));
    }

    // --- Test for proper handling of singletons, all-nulls, degenerate input for all aggregation APIs ---
    #[test]
    fn test_all_apis_singleton_and_nulls() {
        let d = vec64![7u32];
        assert_eq!(sum_u32(&d, None, None), Some(7));
        assert_eq!(average_u32(&d, None, None), Some(7.0));
        assert_eq!(variance_u32(&d, None, None, false).unwrap(), 0.0);
        assert_eq!(mode(&d, None, None), Some(7));
        assert_eq!(median(&d, None, None, true), Some(7));
        assert_eq!(
            quantile(&d, None, None, 3, true).unwrap(),
            Vec64::from_slice(&[7, 7])
        );
        assert_eq!(iqr(&d, None, None, true), Some(0));

        let mask = mask_from_bools(&[false]);
        assert_eq!(sum_u32(&d, Some(&mask), Some(1)), None);
        assert_eq!(median(&d, Some(&mask), Some(1), true), None);
        assert_eq!(quantile(&d, Some(&mask), Some(1), 3, true), None);
        assert_eq!(mode(&d, Some(&mask), Some(1)), None);
        assert_eq!(iqr(&d, Some(&mask), Some(1), true), None);
    }

    #[test]
    fn test_skewness_kurtosis_dense() {
        // Data 1..=5 -> mean 3, symmetric
        // population skewness = 0, population excess kurtosis = -1.3
        // sample skewness = 0, sample excess kurtosis = -1.2
        let d_f64 = vec64![1.0f64, 2.0, 3.0, 4.0, 5.0];
        // population
        assert!(approx_eq(
            skewness_f64(&d_f64, None, None, false).unwrap(),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_f64(&d_f64, None, None, false).unwrap(),
            -1.3,
            1e-12
        ));
        // sample
        assert!(approx_eq(
            skewness_f64(&d_f64, None, None, true).unwrap(),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_f64(&d_f64, None, None, true).unwrap(),
            -1.2,
            1e-12
        ));

        let d_f32 = vec64![1.0f32, 2.0, 3.0, 4.0, 5.0];
        // population
        assert!(approx_eq(
            skewness_f32(&d_f32, None, None, false).unwrap(),
            0.0,
            1e-6
        ));
        assert!(approx_eq(
            kurtosis_f32(&d_f32, None, None, false).unwrap(),
            -1.3,
            1e-6
        ));
        // sample
        assert!(approx_eq(
            skewness_f32(&d_f32, None, None, true).unwrap(),
            0.0,
            1e-6
        ));
        assert!(approx_eq(
            kurtosis_f32(&d_f32, None, None, true).unwrap(),
            -1.2,
            1e-6
        ));

        let d_i32 = vec64![1i32, 2, 3, 4, 5];
        assert!(approx_eq(
            skewness_i32(&d_i32, None, None, false).unwrap(),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_i32(&d_i32, None, None, false).unwrap(),
            -1.3,
            1e-12
        ));
        assert!(approx_eq(
            skewness_i32(&d_i32, None, None, true).unwrap(),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_i32(&d_i32, None, None, true).unwrap(),
            -1.2,
            1e-12
        ));
    }

    #[test]
    fn test_skewness_kurtosis_masked() {
        // Mask out 1 & 5 -> [2,3,4]
        // population skewness = 0, population & sample excess kurtosis = -1.5
        let data = vec64![1i32, 2, 3, 4, 5];
        let mask = mask_from_bools(&[false, true, true, true, false]);
        // population
        assert!(approx_eq(
            skewness_i32(&data, Some(&mask), Some(2), false).unwrap(),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_i32(&data, Some(&mask), Some(2), false).unwrap(),
            -1.5,
            1e-12
        ));
        // sample
        assert!(approx_eq(
            skewness_i32(&data, Some(&mask), Some(2), true).unwrap(),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_i32(&data, Some(&mask), Some(2), true).unwrap(),
            -1.5,
            1e-12
        ));
    }

    #[test]
    fn test_skewness_kurtosis_constant_none() {
        // Variance zero -> expect None
        let d = vec64![7u32; 5];
        assert_eq!(skewness_u32(&d, None, None, false), None);
        assert_eq!(kurtosis_u32(&d, None, None, false), None);
        assert_eq!(skewness_u32(&d, None, None, true), None);
        assert_eq!(kurtosis_u32(&d, None, None, true), None);
    }

    #[test]
    fn test_kurtosis_skewness_large_f32() {
        // Mixed negative/positive of length 11:
        // population: skew ≈ -0.20081265, kurt ≈ -1.28414694
        // sample:     skew ≈ -0.23401565, kurt ≈ -1.30691156
        let data = vec64![-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0];
        // population
        assert!(approx_eq(
            skewness_f32(&data, None, None, false).unwrap(),
            -0.20081265422848577_f64,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_f32(&data, None, None, false).unwrap(),
            -1.2841469381610144_f64,
            1e-12
        ));
        // sample
        assert!(approx_eq(
            skewness_f32(&data, None, None, true).unwrap(),
            -0.23401565397707677_f64,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_f32(&data, None, None, true).unwrap(),
            -1.3069115636016906_f64,
            1e-12
        ));
    }

    fn bitmask_from_bools(bs: &[bool]) -> Bitmask {
        let mut mask = Bitmask::with_capacity(bs.len());
        for (i, &b) in bs.iter().enumerate() {
            mask.set(i, b);
        }
        mask
    }

    #[test]
    fn test_kurtosis_skewness_large_masked_i32() {
        // Drop 4th element (4), leaving 9 values:
        // population: skew ≈ 2.43067998, kurt ≈ 3.99120344
        // sample:     skew ≈ 2.94642909, kurt ≈ 8.74514940
        let data = vec64![1, 2, 3, 4, 5, 6, 7, 100, 1000, 10000];
        let mask = mask_from_bools(&[true, true, true, false, true, true, true, true, true, true]);
        // population
        assert!(approx_eq(
            skewness_i32(&data, Some(&mask), None, false).unwrap(),
            2.4306799843134046_f64,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_i32(&data, Some(&mask), None, false).unwrap(),
            3.991203437112363_f64,
            1e-12
        ));
        // sample
        assert!(approx_eq(
            skewness_i32(&data, Some(&mask), None, true).unwrap(),
            2.946429085375575_f64,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_i32(&data, Some(&mask), None, true).unwrap(),
            8.745149404023547_f64,
            1e-12
        ));
    }

    #[test]
    fn test_kurtosis_skewness_no_valids() {
        let data: [f64; 3] = [0.0, 0.0, 0.0];
        let mask = bitmask_from_bools(&[false, false, false]);
        assert_eq!(kurtosis_f64(&data, Some(&mask), None, false), None);
        assert_eq!(skewness_f64(&data, Some(&mask), None, false), None);
        assert_eq!(kurtosis_f64(&data, Some(&mask), None, true), None);
        assert_eq!(skewness_f64(&data, Some(&mask), None, true), None);
    }

    #[test]
    fn test_kurtosis_skewness_single_value() {
        let data = vec64![1u32];
        assert_eq!(kurtosis_u32(&data, None, None, false), None);
        assert_eq!(skewness_u32(&data, None, None, false), None);
        assert_eq!(kurtosis_u32(&data, None, None, true), None);
        assert_eq!(skewness_u32(&data, None, None, true), None);
    }

    #[test]
    fn test_kurtosis_skewness_all_equal() {
        let data = vec64![7.0, 7.0, 7.0, 7.0];
        assert_eq!(kurtosis_f64(&data, None, None, false), None);
        assert_eq!(skewness_f64(&data, None, None, false), None);
        assert_eq!(kurtosis_f64(&data, None, None, true), None);
        assert_eq!(skewness_f64(&data, None, None, true), None);
    }

    #[test]
    fn test_kurtosis_skewness_mixed_nulls() {
        // Valid [2.0,7.0,10.0] (length 3):
        // population: skew ≈ -0.29479962, kurt = -1.5
        // sample:     skew ≈ -0.72210865, kurt = -1.5
        let data = vec64![2.0, 5.0, 7.0, 3.0, 10.0];
        let mask = mask_from_bools(&[true, false, true, false, true]);
        // population
        assert!(approx_eq(
            skewness_f64(&data, Some(&mask), None, false).unwrap(),
            -0.29479962014482897_f64,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_f64(&data, Some(&mask), None, false).unwrap(),
            -1.5_f64,
            1e-12
        ));
        // sample
        assert!(approx_eq(
            skewness_f64(&data, Some(&mask), None, true).unwrap(),
            -0.7221086457211346_f64,
            1e-12
        ));
        assert!(approx_eq(
            kurtosis_f64(&data, Some(&mask), None, true).unwrap(),
            -1.5_f64,
            1e-12
        ));
    }

    #[test]
    fn test_kurtosis_skewness_integers_large() {
        let data = vec64![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
        assert!(kurtosis_i32(&data, None, None, false).unwrap().abs() < 1.5);
        assert!(skewness_i32(&data, None, None, false).unwrap().abs() < 1e-6);
        // Sample
        assert!(kurtosis_i32(&data, None, None, true).unwrap().abs() < 2.0);
        assert!(skewness_i32(&data, None, None, true).unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_agg_product_dense() {
        // No mask, all values included
        let data = vec64![2, 3, 4, 5];
        let prod = agg_product(&data, None, 0, data.len());
        assert_eq!(prod, 120);

        // Offset/len window
        let prod = agg_product(&data, None, 1, 2);
        assert_eq!(prod, 12); // 3*4
    }

    #[test]
    fn test_agg_product_with_mask_all_valid() {
        let data = vec64![2, 3, 4, 5];
        let mask = Bitmask::new_set_all(4, true);
        let prod = agg_product(&data, Some(&mask), 0, 4);
        assert_eq!(prod, 120);
    }

    #[test]
    fn test_agg_product_with_mask_some_nulls() {
        let data = vec64![2, 3, 4, 5];
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set(0, true);
        mask.set(2, true);
        // Only data[0] and data[2] included: 2*4=8
        let prod = agg_product(&data, Some(&mask), 0, 4);
        assert_eq!(prod, 8);
    }

    #[test]
    fn test_agg_product_zero_short_circuit() {
        let data = vec64![2, 0, 4, 5];
        let prod = agg_product(&data, None, 0, 4);
        assert_eq!(prod, 0);
    }

    #[test]
    fn test_agg_product_empty() {
        let data: [i32; 0] = [];
        let prod = agg_product(&data, None, 0, 0);
        assert_eq!(prod, 1); // One by convention
    }

    #[test]
    fn test_agg_product_mask_all_null() {
        let data = vec64![2, 3, 4, 5];
        let mask = Bitmask::new_set_all(4, false);
        let prod = agg_product(&data, Some(&mask), 0, 4);
        assert_eq!(prod, 1); // No values included: returns one
    }

    #[test]
    fn test_agg_product_offset_and_mask() {
        let data = vec64![2, 3, 4, 5, 6];
        let mut mask = Bitmask::new_set_all(5, false);
        mask.set(1, true);
        mask.set(3, true);
        let prod = agg_product(&data, Some(&mask), 1, 3);
        assert_eq!(prod, 15);
    }

    #[test]
    fn test_sum_squares_empty() {
        let v: [f64; 0] = [];
        assert_eq!(sum_squares(&v, None, None), 0.0);
    }

    #[test]
    fn test_sum_squares_basic() {
        let v = vec64![1.0, 2.0, 3.0];
        assert_eq!(sum_squares(&v, None, None), 1.0 + 4.0 + 9.0);
    }

    #[test]
    fn test_sum_squares_neg() {
        let v = vec64![-1.0, -2.0, -3.0];
        assert_eq!(sum_squares(&v, None, None), 1.0 + 4.0 + 9.0);
    }

    #[test]
    fn test_sum_squares_large() {
        let v: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let expect: f64 = v.iter().map(|x| x * x).sum();
        assert!((sum_squares(&v, None, None) - expect).abs() < 1e-9);
    }

    #[test]
    fn test_sum_squares_masked() {
        let v = vec64![1.0, 2.0, 3.0, 4.0];
        let mask = Bitmask::from_bools(&[true, false, true, false]);
        assert_eq!(sum_squares(&v, Some(&mask), Some(2)), 1.0 + 9.0);
    }

    #[test]
    fn test_sum_cubes_empty() {
        let v: [f64; 0] = [];
        assert_eq!(sum_cubes(&v, None, None), 0.0);
    }

    #[test]
    fn test_sum_cubes_basic() {
        let v = vec64![1.0, 2.0, 3.0];
        assert_eq!(sum_cubes(&v, None, None), 1.0 + 8.0 + 27.0);
    }

    #[test]
    fn test_sum_cubes_neg() {
        let v = vec64![-1.0, -2.0, -3.0];
        assert_eq!(sum_cubes(&v, None, None), -1.0 - 8.0 - 27.0);
    }

    #[test]
    fn test_sum_cubes_large() {
        let v: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let expect: f64 = v.iter().map(|x| x * x * x).sum();
        assert!((sum_cubes(&v, None, None) - expect).abs() < 1e-6);
    }

    #[test]
    fn test_sum_cubes_masked() {
        let v = vec64![1.0, 2.0, 3.0, 4.0];
        let mask = Bitmask::from_bools(&[true, true, false, true]);
        assert_eq!(sum_cubes(&v, Some(&mask), Some(1)), 1.0 + 8.0 + 64.0);
    }

    #[test]
    fn test_sum_quartics_empty() {
        let v: [f64; 0] = [];
        assert_eq!(sum_quartics(&v, None, None), 0.0);
    }

    #[test]
    fn test_sum_quartics_basic() {
        let v = vec64![1.0, 2.0, 3.0];
        assert_eq!(sum_quartics(&v, None, None), 1.0 + 16.0 + 81.0);
    }

    #[test]
    fn test_sum_quartics_neg() {
        let v = vec64![-1.0, -2.0, -3.0];
        assert_eq!(sum_quartics(&v, None, None), 1.0 + 16.0 + 81.0);
    }

    #[test]
    fn test_sum_quartics_large() {
        let v: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let expect: f64 = v
            .iter()
            .map(|x| {
                let x2 = x * x;
                x2 * x2
            })
            .sum();
        assert!((sum_quartics(&v, None, None) - expect).abs() < 1e-3);
    }

    #[test]
    fn test_sum_quartics_masked() {
        let v = vec64![1.0, 2.0, 3.0, 4.0];
        let mask = Bitmask::from_bools(&[false, true, false, true]);
        assert_eq!(sum_quartics(&v, Some(&mask), Some(2)), 16.0 + 256.0);
    }

    #[test]
    fn test_sum_powers_singleton() {
        let v = vec64![4.2];
        assert!((sum_squares(&v, None, None) - 4.2 * 4.2).abs() < 1e-12);
        assert!((sum_cubes(&v, None, None) - 4.2 * 4.2 * 4.2).abs() < 1e-12);
        assert!((sum_quartics(&v, None, None) - (4.2 * 4.2 * 4.2 * 4.2)).abs() < 1e-12);
    }

    #[test]
    fn test_sum_powers_f64_extremes() {
        let v = vec64![f64::MAX, -f64::MAX, 0.0, f64::MIN_POSITIVE];
        let ss = sum_squares(&v, None, None).is_infinite();
        let sc = sum_cubes(&v, None, None).is_infinite();
        let sq = sum_quartics(&v, None, None).is_infinite();
        println!("{:?}", ss);
        println!("{:?}", sc);
        println!("{:?}", sq);
        // f64::MAX * f64::MAX is inf, so expect inf for all powers >1
        assert!(sum_squares(&v, None, None).is_infinite());
        assert!(sum_cubes(&v, None, None).is_infinite());
        assert!(sum_quartics(&v, None, None).is_infinite());
    }
}
