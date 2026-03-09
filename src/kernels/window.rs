// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Window Functions Kernels Module** - *High-Performance Analytical Window Operations*
//!
//! Advanced window function kernels for sliding window computations,
//! ranking operations, and positional analytics with SIMD acceleration and null-aware semantics.
//! Backbone of time series analysis, analytical SQL window functions, and chunked streaming computations.
//!
//! ## Core Operations
//! - **Moving averages**: Rolling mean calculations with configurable window sizes
//! - **Cumulative functions**: Running sums, products, and statistical aggregations  
//! - **Ranking functions**: ROW_NUMBER, RANK, DENSE_RANK with tie-handling strategies
//! - **Lead/lag operations**: Positional value access with configurable offsets
//! - **Percentile functions**: Moving quantile calculations with interpolation support
//! - **Window aggregates**: MIN, MAX, SUM operations over sliding windows

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::marker::PhantomData;

use minarrow::{
    Bitmask, BooleanAVT, BooleanArray, FloatArray, Integer, IntegerArray, Length, MaskedArray,
    Offset, StringArray, Vec64,
    aliases::{FloatAVT, IntegerAVT},
    enums::error::KernelError,
    vec64,
};
use num_traits::{Float, Num, NumCast, One, Zero};

use minarrow::StringAVT;
use minarrow::utils::confirm_mask_capacity;

use crate::kernels::sort::total_cmp_f;

// Helpers
#[inline(always)]
fn new_null_mask(len: usize) -> Bitmask {
    Bitmask::new_set_all(len, false)
}

#[inline(always)]
fn prealloc_vec<T: Copy>(len: usize) -> Vec64<T> {
    let mut v = Vec64::<T>::with_capacity(len);
    unsafe { v.set_len(len) };
    v
}

// Rolling kernels (sum, product, min, max, mean, count)

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Generic sliding window aggregator for kernels that allow an
/// incremental push and pop update (sum, product, etc.).
/// Always emits the running aggregate, even when the subwindow has nulls.
/// Only flags "valid" once the full subwindow has been seen.
///
/// Panics if `out.len() != data.len()` or `out_mask.capacity() < data.len()`.
#[inline(always)]
fn rolling_push_pop_to<T, FAdd, FRem>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    mut add: FAdd,
    mut remove: FRem,
    zero: T,
    out: &mut [T],
    out_mask: &mut Bitmask,
) where
    T: Copy,
    FAdd: FnMut(T, T) -> T,
    FRem: FnMut(T, T) -> T,
{
    let n = data.len();
    assert_eq!(
        n,
        out.len(),
        "rolling_push_pop_to: input/output length mismatch"
    );

    if subwindow == 0 {
        for slot in out.iter_mut() {
            *slot = zero;
        }
        return;
    }

    let mut agg = zero;
    let mut invalids = 0usize;
    for i in 0..n {
        if mask.map_or(true, |m| unsafe { m.get_unchecked(i) }) {
            agg = add(agg, data[i]);
        } else {
            invalids += 1;
        }
        if i + 1 > subwindow {
            let j = i + 1 - subwindow - 1;
            if mask.map_or(true, |m| unsafe { m.get_unchecked(j) }) {
                agg = remove(agg, data[j]);
            } else {
                invalids -= 1;
            }
        }
        if i + 1 < subwindow {
            unsafe { out_mask.set_unchecked(i, false) };
            out[i] = zero;
        } else {
            let ok = invalids == 0;
            unsafe { out_mask.set_unchecked(i, ok) };
            out[i] = agg;
        }
    }
}

/// Allocating variant: creates new output buffers internally.
#[inline(always)]
pub fn rolling_push_pop<T, FAdd, FRem>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    add: FAdd,
    remove: FRem,
    zero: T,
) -> (Vec64<T>, Bitmask)
where
    T: Copy,
    FAdd: FnMut(T, T) -> T,
    FRem: FnMut(T, T) -> T,
{
    let n = data.len();
    let mut out = prealloc_vec::<T>(n);
    let mut out_mask = new_null_mask(n);
    rolling_push_pop_to(
        data,
        mask,
        subwindow,
        add,
        remove,
        zero,
        &mut out,
        &mut out_mask,
    );
    (out, out_mask)
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Generic rolling extreme aggregator (min/max) for a subwindow over a slice.
///
/// Panics if `out.len() != data.len()`.
#[inline(always)]
pub fn rolling_extreme_to<T, F>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    mut better: F,
    zero: T,
    out: &mut [T],
    out_mask: &mut Bitmask,
) where
    T: Copy,
    F: FnMut(&T, &T) -> bool,
{
    let n = data.len();
    assert_eq!(
        n,
        out.len(),
        "rolling_extreme_to: input/output length mismatch"
    );

    if subwindow == 0 {
        return;
    }

    for i in 0..n {
        if i + 1 < subwindow {
            unsafe { out_mask.set_unchecked(i, false) };
            out[i] = zero;
            continue;
        }
        let start = i + 1 - subwindow;
        let mut found = false;
        let mut extreme = zero;
        for j in start..=i {
            if mask.map_or(true, |m| unsafe { m.get_unchecked(j) }) {
                if !found {
                    extreme = data[j];
                    found = true;
                } else if better(&data[j], &extreme) {
                    extreme = data[j];
                }
            } else {
                found = false;
                break;
            }
        }
        unsafe { out_mask.set_unchecked(i, found) };
        out[i] = if found { extreme } else { zero };
    }
}

/// Allocating variant: creates new output buffers internally.
#[inline(always)]
pub fn rolling_extreme<T, F>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    better: F,
    zero: T,
) -> (Vec64<T>, Bitmask)
where
    T: Copy,
    F: FnMut(&T, &T) -> bool,
{
    let n = data.len();
    let mut out = prealloc_vec::<T>(n);
    let mut out_mask = new_null_mask(n);
    rolling_extreme_to(data, mask, subwindow, better, zero, &mut out, &mut out_mask);
    (out, out_mask)
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Computes rolling sums over a sliding window for integer data with null-aware semantics.
/// Panics if `out.len() != data.len()`.
#[inline]
pub fn rolling_sum_int_to<T: Num + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_push_pop_to(
        data,
        mask,
        subwindow,
        |a, b| a + b,
        |a, b| a - b,
        T::zero(),
        out,
        out_mask,
    );
    if mask.is_some() && subwindow > 0 && subwindow - 1 < out.len() {
        unsafe { out_mask.set_unchecked(subwindow - 1, false) };
        out[subwindow - 1] = T::zero();
    }
}

/// Computes rolling sums over a sliding window for integer data with null-aware semantics.
///
/// Applies a sliding window of configurable size to compute cumulative sums, employing
/// incremental computation to avoid O(n²) complexity through efficient push-pop operations.
/// Each position in the output represents the sum of values within the preceding window.
///
/// ## Parameters
/// * `window` - Integer array view containing the data, offset, and length information
/// * `subwindow` - Size of the sliding window (number of elements to sum)
///
/// ## Returns
/// Returns an `IntegerArray<T>` containing:
/// - Rolling sums for each position where a complete window exists
/// - Zero values for positions before the window is complete
/// - Null mask indicating validity (false for incomplete windows or null-contaminated windows)
#[inline]
pub fn rolling_sum_int<T: Num + Copy + Zero>(
    window: IntegerAVT<'_, T>,
    subwindow: usize,
) -> IntegerArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_sum_int_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Computes rolling sums over a sliding window for floating-point data.
/// Panics if `out.len() != data.len()`.
#[inline]
pub fn rolling_sum_float_to<T: Float + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_push_pop_to(
        data,
        mask,
        subwindow,
        |a, b| a + b,
        |a, b| a - b,
        T::zero(),
        out,
        out_mask,
    );
    if subwindow > 0 && subwindow - 1 < out.len() {
        out_mask.set(subwindow - 1, false);
        out[subwindow - 1] = T::zero();
    }
}

/// Computes rolling sums over a sliding window for floating-point data with IEEE 754 compliance.
///
/// Applies incremental computation to calculate cumulative sums across sliding windows,
/// maintaining numerical stability through careful accumulation strategies. Handles
/// special floating-point values (infinity, NaN) according to IEEE 754 semantics.
///
/// ## Parameters
/// * `window` - Float array view containing the data, offset, and length information
/// * `subwindow` - Size of the sliding window for summation
///
/// ## Returns
/// Returns a `FloatArray<T>` containing:
/// - Rolling sums computed incrementally for efficiency
/// - Zero values for positions with incomplete windows
/// - Proper null mask for window validity tracking
#[inline]
pub fn rolling_sum_float<T: Float + Copy + Zero>(
    window: FloatAVT<'_, T>,
    subwindow: usize,
) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_sum_float_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Computes rolling sums over i32 data (pre-converted from booleans).
/// Panics if `out.len() != data.len()`.
#[inline]
pub fn rolling_sum_bool_to(
    data: &[i32],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) {
    rolling_push_pop_to(
        data,
        mask,
        subwindow,
        |a, b| a + b,
        |a, b| a - b,
        0,
        out,
        out_mask,
    );
    if subwindow > 0 && subwindow - 1 < out.len() {
        out_mask.set(subwindow - 1, false);
        out[subwindow - 1] = 0;
    }
}

/// Computes rolling sums over boolean data, counting true values within sliding windows.
///
/// Treats boolean values as integers (true=1, false=0) and applies sliding window
/// summation to count true occurrences within each window position. Essential for
/// constructing conditional aggregations and boolean pattern analysis.
///
/// ## Parameters
/// * `window` - Boolean array view with offset and length specifications
/// * `subwindow` - Number of boolean values to consider in each sliding window
///
/// ## Returns
/// Returns an `IntegerArray<i32>` containing:
/// - Count of true values within each complete window
/// - Zero for positions with incomplete windows
/// - Null mask indicating window completeness and null contamination
#[inline]
pub fn rolling_sum_bool(window: BooleanAVT<'_, ()>, subwindow: usize) -> IntegerArray<i32> {
    let (arr, offset, len) = window;
    let bools: Vec<i32> = arr.iter_range(offset, len).map(|b| b as i32).collect();
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<i32>(len);
    let mut out_mask = new_null_mask(len);
    rolling_sum_bool_to(&bools, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_product_int_to<T: Num + Copy + One + Zero + PartialEq>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = data.len();
    assert_eq!(
        n,
        out.len(),
        "rolling_product_int_to: input/output length mismatch"
    );

    if subwindow == 0 {
        for slot in out.iter_mut() {
            *slot = T::zero();
        }
        return;
    }

    // Track the product of non-zero elements and a separate zero count
    // to avoid integer division-by-zero when a zero leaves the window.
    let mut nz_product = T::one();
    let mut zero_count = 0usize;
    let mut invalids = 0usize;

    for i in 0..n {
        if mask.map_or(true, |m| unsafe { m.get_unchecked(i) }) {
            if data[i] == T::zero() {
                zero_count += 1;
            } else {
                nz_product = nz_product * data[i];
            }
        } else {
            invalids += 1;
        }

        if i + 1 > subwindow {
            let j = i + 1 - subwindow - 1;
            if mask.map_or(true, |m| unsafe { m.get_unchecked(j) }) {
                if data[j] == T::zero() {
                    zero_count -= 1;
                } else {
                    nz_product = nz_product / data[j];
                }
            } else {
                invalids -= 1;
            }
        }

        if i + 1 < subwindow {
            unsafe { out_mask.set_unchecked(i, false) };
            out[i] = T::zero();
        } else {
            let ok = invalids == 0;
            unsafe { out_mask.set_unchecked(i, ok) };
            out[i] = if zero_count > 0 {
                T::zero()
            } else {
                nz_product
            };
        }
    }
}

/// Computes rolling products over a sliding window for integer data with overflow protection.
///
/// Applies multiplicative aggregation across sliding windows using incremental computation
/// through division operations. Maintains numerical stability through careful handling of
/// zero values and potential overflow conditions in integer arithmetic.
#[inline]
pub fn rolling_product_int<T: Num + Copy + One + Zero>(
    window: IntegerAVT<'_, T>,
    subwindow: usize,
) -> IntegerArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_product_int_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_product_float_to<T: Float + Copy + One + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_push_pop_to(
        data,
        mask,
        subwindow,
        |a, b| a * b,
        |a, b| a / b,
        T::one(),
        out,
        out_mask,
    );
}

/// Computes rolling products over floating-point data with IEEE 754 mathematical semantics.
///
/// Performs multiplicative aggregation using incremental computation strategies that
/// maintain numerical precision through careful handling of special values (infinity,
/// NaN, zero) according to IEEE 754 standards.
#[inline]
pub fn rolling_product_float<T: Float + Copy + One + Zero>(
    window: FloatAVT<'_, T>,
    subwindow: usize,
) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_product_float_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Computes rolling logical AND over boolean data (pre-converted to i32: 1=true, 0=false).
#[inline]
pub fn rolling_product_bool_to(
    data: &[i32],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut Bitmask,
    out_mask: &mut Bitmask,
) {
    let n = data.len();
    for i in 0..n {
        let start = if i + 1 >= subwindow {
            i + 1 - subwindow
        } else {
            0
        };
        let mut acc = true;
        let mut valid = subwindow > 0 && i + 1 >= subwindow;
        for j in start..=i {
            let is_valid = mask.map_or(true, |m| unsafe { m.get_unchecked(j) });
            if is_valid {
                acc &= data[j] != 0;
            } else {
                valid = false;
                break;
            }
        }
        unsafe { out_mask.set_unchecked(i, valid) };
        out.set(i, valid && acc);
    }
}

/// Computes rolling logical AND operations over boolean data within sliding windows.
///
/// Treats boolean multiplication as logical AND operations, computing the conjunction
/// of all boolean values within each sliding window. Essential for constructing
/// compound logical conditions and boolean pattern validation.
#[inline]
pub fn rolling_product_bool(window: BooleanAVT<'_, ()>, subwindow: usize) -> BooleanArray<()> {
    let (arr, offset, len) = window;
    let n = len;
    let mut out_mask = new_null_mask(n);
    let mut out = Bitmask::new_set_all(n, false);

    for i in 0..n {
        let start = if i + 1 >= subwindow {
            i + 1 - subwindow
        } else {
            0
        };
        let mut acc = true;
        let mut valid = subwindow > 0 && i + 1 >= subwindow;
        for j in start..=i {
            match unsafe { arr.get_unchecked(offset + j) } {
                Some(val) => acc &= val,
                None => {
                    valid = false;
                    break;
                }
            }
        }
        unsafe { out_mask.set_unchecked(i, valid) };
        out.set(i, valid && acc);
    }

    BooleanArray {
        data: out.into(),
        null_mask: Some(out_mask),
        len: n,
        _phantom: PhantomData,
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_mean_int_to<T: NumCast + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [f64],
    out_mask: &mut Bitmask,
) {
    let n = data.len();
    if subwindow == 0 {
        return;
    }
    for i in 0..n {
        if i + 1 < subwindow {
            unsafe { out_mask.set_unchecked(i, false) };
            out[i] = 0.0;
            continue;
        }
        let start = i + 1 - subwindow;
        let mut sum = 0.0;
        let mut valid = true;
        for j in start..=i {
            if mask.map_or(true, |m| unsafe { m.get_unchecked(j) }) {
                sum += num_traits::cast(data[j]).unwrap_or(0.0);
            } else {
                valid = false;
                break;
            }
        }
        unsafe { out_mask.set_unchecked(i, valid) };
        out[i] = if valid { sum / subwindow as f64 } else { 0.0 };
    }
    if subwindow > 0 && subwindow - 1 < out.len() {
        unsafe { out_mask.set_unchecked(subwindow - 1, false) };
        out[subwindow - 1] = 0.0;
    }
}

/// Computes rolling arithmetic means over integer data with high-precision floating-point output.
#[inline]
pub fn rolling_mean_int<T: NumCast + Copy + Zero>(
    window: IntegerAVT<'_, T>,
    subwindow: usize,
) -> FloatArray<f64> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<f64>(len);
    let mut out_mask = new_null_mask(len);
    rolling_mean_int_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_mean_float_to<T: Float + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let n = data.len();
    if subwindow == 0 {
        return;
    }
    for i in 0..n {
        if i + 1 < subwindow {
            unsafe { out_mask.set_unchecked(i, false) };
            out[i] = T::zero();
            continue;
        }
        let start = i + 1 - subwindow;
        let mut sum = T::zero();
        let mut valid = true;
        for j in start..=i {
            if mask.map_or(true, |m| unsafe { m.get_unchecked(j) }) {
                sum = sum + data[j];
            } else {
                valid = false;
                break;
            }
        }
        unsafe { out_mask.set_unchecked(i, valid) };
        out[i] = if valid {
            sum / T::from(subwindow as u32).unwrap()
        } else {
            T::zero()
        };
    }
    if subwindow > 0 && subwindow - 1 < out.len() {
        unsafe { out_mask.set_unchecked(subwindow - 1, false) };
        out[subwindow - 1] = T::zero();
    }
}

/// Computes rolling arithmetic means over floating-point data with IEEE 754 compliance.
#[inline]
pub fn rolling_mean_float<T: Float + Copy + Zero>(
    window: FloatAVT<'_, T>,
    subwindow: usize,
) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_mean_float_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_min_int_to<T: Ord + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_extreme_to(
        data,
        mask,
        subwindow,
        |a, b| a < b,
        T::zero(),
        out,
        out_mask,
    );
    if subwindow > 0 && subwindow - 1 < out.len() {
        out_mask.set(subwindow - 1, false);
        out[subwindow - 1] = T::zero();
    }
}

/// Computes rolling minimum values over integer data within sliding windows.
#[inline]
pub fn rolling_min_int<T: Ord + Copy + Zero>(
    window: IntegerAVT<'_, T>,
    subwindow: usize,
) -> IntegerArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_min_int_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_max_int_to<T: Ord + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_extreme_to(
        data,
        mask,
        subwindow,
        |a, b| a > b,
        T::zero(),
        out,
        out_mask,
    );
}

/// Computes rolling maximum values over integer data within sliding windows.
#[inline]
pub fn rolling_max_int<T: Ord + Copy + Zero>(
    window: IntegerAVT<'_, T>,
    subwindow: usize,
) -> IntegerArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_max_int_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_min_float_to<T: Float + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_extreme_to(
        data,
        mask,
        subwindow,
        |a, b| a < b,
        T::zero(),
        out,
        out_mask,
    );
    if subwindow > 0 && subwindow - 1 < out.len() {
        out_mask.set(subwindow - 1, false);
        out[subwindow - 1] = T::zero();
    }
}

/// Computes rolling minimum values over floating-point data with IEEE 754 compliance.
#[inline]
pub fn rolling_min_float<T: Float + Copy + Zero>(
    window: FloatAVT<'_, T>,
    subwindow: usize,
) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_min_float_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_max_float_to<T: Float + Copy + Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    subwindow: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    rolling_extreme_to(
        data,
        mask,
        subwindow,
        |a, b| a > b,
        T::zero(),
        out,
        out_mask,
    );
}

/// Computes rolling maximum values over floating-point data with IEEE 754 compliance.
#[inline]
pub fn rolling_max_float<T: Float + Copy + Zero>(
    window: FloatAVT<'_, T>,
    subwindow: usize,
) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = new_null_mask(len);
    rolling_max_float_to(data, mask.as_ref(), subwindow, &mut out, &mut out_mask);
    FloatArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Computes rolling counts of elements within sliding windows for positional analysis.
///
/// Counts the number of elements present within each sliding window position, providing
/// fundamental cardinality information essential for statistical analysis and data validation.
/// Unlike other rolling functions, operates on position information rather than data values.
///
/// ## Parameters
/// * `window` - Tuple containing offset and length defining the data window scope
/// * `subwindow` - Size of sliding window for element counting
///
/// ## Returns
/// Returns an `IntegerArray<i32>` containing:
/// - Count of elements in each complete window (always equals subwindow size)
/// - Zero values for positions with incomplete windows
/// - Null mask indicating where complete windows exist
///
/// ## Examples
/// ```rust,ignore
/// use simd_kernels::kernels::window::rolling_count;
///
/// let result = rolling_count((0, 5), 3); // 5 elements, window size 3
/// ```
#[inline]
/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn rolling_count_to(len: usize, subwindow: usize, out: &mut [i32], out_mask: &mut Bitmask) {
    for i in 0..len {
        let start = if i + 1 >= subwindow {
            i + 1 - subwindow
        } else {
            0
        };
        let count = (i - start + 1) as i32;
        let valid_row = subwindow > 0 && i + 1 >= subwindow;
        unsafe { out_mask.set_unchecked(i, valid_row) };
        out[i] = if valid_row { count } else { 0 };
    }
}

pub fn rolling_count(window: (Offset, Length), subwindow: usize) -> IntegerArray<i32> {
    let (_offset, len) = window;
    let mut out = prealloc_vec::<i32>(len);
    let mut out_mask = new_null_mask(len);
    for i in 0..len {
        let start = if i + 1 >= subwindow {
            i + 1 - subwindow
        } else {
            0
        };
        let count = (i - start + 1) as i32;
        let valid_row = subwindow > 0 && i + 1 >= subwindow;
        unsafe { out_mask.set_unchecked(i, valid_row) };
        out[i] = if valid_row { count } else { 0 };
    }
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

// Rank and Dense-rank kernels

#[inline(always)]
fn rank_numeric_to<T, F>(
    data: &[T],
    mask: Option<&Bitmask>,
    mut cmp: F,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) where
    T: Copy,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| cmp(&data[i], &data[j]));

    for (rank, &i) in indices.iter().enumerate() {
        if mask.map_or(true, |m| unsafe { m.get_unchecked(i) }) {
            out[i] = (rank + 1) as i32;
            unsafe { out_mask.set_unchecked(i, true) };
        }
    }
}

#[inline(always)]
fn rank_numeric<T, F>(data: &[T], mask: Option<&Bitmask>, cmp: F) -> IntegerArray<i32>
where
    T: Copy,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    let n = data.len();
    let mut out = vec64![0i32; n];
    let mut out_mask = Bitmask::new_set_all(n, false);
    rank_numeric_to(data, mask, cmp, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Computes standard SQL ROW_NUMBER() ranking for integer data with tie handling.
///
/// Assigns sequential rank values to elements based on their sorted order, providing
/// standard SQL ROW_NUMBER() semantics where tied values receive different ranks.
/// Essential for analytical queries requiring unique positional ranking.
///
/// ## Parameters
/// * `window` - Integer array view containing values for ranking
///
/// ## Returns
/// Returns an `IntegerArray<i32>` containing:
/// - Rank values from 1 to n for valid elements
/// - Zero values for null elements
/// - Null mask indicating which positions have valid ranks
///
/// ## Ranking Semantics
/// - **ROW_NUMBER() behaviour**: Each element receives a unique rank (1, 2, 3, ...)
/// - **Tie breaking**: Tied values receive different ranks based on their position
/// - **Ascending order**: Smaller values receive lower (better) ranks
/// - **Null exclusion**: Null values are excluded from ranking and receive rank 0
///
/// ## Use Cases
/// - **Analytical queries**: SQL ROW_NUMBER() window function implementation
/// - **Leaderboards**: Creating ordered rankings with unique positions
/// - **Percentile calculation**: Basis for percentile and quartile computations
/// - **Data analysis**: Establishing ordinality in integer datasets
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::IntegerArray;
/// use simd_kernels::kernels::window::rank_int;
///
/// let arr = IntegerArray::<i32>::from_slice(&[30, 10, 20, 10]);
/// let result = rank_int((&arr, 0, arr.len()));
/// // Output: [4, 1, 3, 2] - ROW_NUMBER() style ranking
/// ```
/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline(always)]
pub fn rank_int_to<T: Ord + Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) {
    rank_numeric_to(data, mask, T::cmp, out, out_mask);
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline(always)]
pub fn rank_float_to<T: Float + Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) {
    rank_numeric_to(data, mask, total_cmp_f, out, out_mask);
}

#[inline(always)]
pub fn rank_int<T: Ord + Copy>(window: IntegerAVT<T>) -> IntegerArray<i32> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let null_mask = if len != arr.data.len() {
        &arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len))
    } else {
        &arr.null_mask
    };
    rank_numeric(data, null_mask.as_ref(), |a, b| a.cmp(b))
}

/// Computes standard SQL ROW_NUMBER() ranking for floating-point data with IEEE 754 compliance.
///
/// Assigns sequential rank values based on sorted floating-point order, implementing
/// ROW_NUMBER() semantics with proper handling of special floating-point values (NaN,
/// infinity) according to IEEE 754 comparison standards.
///
/// ## Parameters
/// * `window` - Float array view containing values for ranking
///
/// ## Returns
/// Returns an `IntegerArray<i32>` containing:
/// - Rank values from 1 to n for valid, non-NaN elements
/// - Zero values for null or NaN elements
/// - Null mask indicating positions with valid ranks
///
/// ## Floating-Point Ranking
/// - **IEEE 754 ordering**: Uses IEEE 754 compliant comparison operations
/// - **NaN handling**: NaN values are excluded from ranking (receive rank 0)
/// - **Infinity treatment**: Positive/negative infinity participate in ranking
/// - **Precision preservation**: Maintains full floating-point comparison precision
///
/// ## Ranking Semantics
/// - **ROW_NUMBER() style**: Each non-NaN element receives unique sequential rank
/// - **Ascending order**: Smaller floating-point values receive lower ranks
/// - **Tie breaking**: Floating-point ties broken by original array position
/// - **Special value exclusion**: NaN and null values excluded from rank assignment
///
/// ## Applications
/// - **Statistical ranking**: Ranking continuous numerical data
/// - **Scientific analysis**: Ordered ranking of experimental measurements
/// - **Financial analysis**: Ranking performance metrics and indicators
/// - **Data preprocessing**: Establishing ordinality for regression analysis
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::FloatArray;
/// use simd_kernels::kernels::window::rank_float;
///
/// let arr = FloatArray::<f64>::from_slice(&[3.14, 2.71, 1.41, f64::NAN]);
/// let result = rank_float((&arr, 0, arr.len()));
/// // Output: [3, 2, 1, 0] - NaN excluded, others ranked by value
/// ```
#[inline(always)]
pub fn rank_float<T: Float + Copy>(window: FloatAVT<T>) -> IntegerArray<i32> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let null_mask = if len != arr.data.len() {
        &arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len))
    } else {
        &arr.null_mask
    };
    rank_numeric(data, null_mask.as_ref(), total_cmp_f)
}

/// Computes standard SQL ROW_NUMBER() ranking for string data with lexicographic ordering.
///
/// Assigns sequential rank values based on lexicographic string comparison, implementing
/// ROW_NUMBER() semantics for textual data. Essential for alphabetical ranking and
/// string-based analytical operations.
///
/// ## Parameters
/// * `arr` - String array view containing textual values for ranking
///
/// ## Returns
/// Returns `Result<IntegerArray<i32>, KernelError>` containing:
/// - **Success**: Rank values from 1 to n for valid string elements
/// - **Error**: KernelError if capacity validation fails
/// - Zero values for null string elements
/// - Null mask indicating positions with valid ranks
///
/// ## String Ranking Semantics
/// - **Lexicographic order**: Uses standard string comparison (dictionary order)
/// - **Case sensitivity**: Comparisons are case-sensitive ("A" < "a")
/// - **Unicode support**: Proper handling of UTF-8 encoded string data
/// - **ROW_NUMBER() behaviour**: Tied strings receive different ranks by position
///
/// ## Error Conditions
/// - **Capacity errors**: Returns KernelError if mask capacity validation fails
/// - **Memory allocation**: May fail with insufficient memory for large datasets
///
/// ## Use Cases
/// - **Alphabetical ranking**: Creating alphabetically ordered rankings
/// - **Text analysis**: Establishing lexicographic ordinality in textual data
/// - **Database operations**: SQL ROW_NUMBER() implementation for string columns
/// - **Sorting applications**: Providing ranking information for string sorting
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::StringArray;
/// use simd_kernels::kernels::window::rank_str;
///
/// let arr = StringArray::<u32>::from_slice(&["zebra", "apple", "banana"]);
/// let result = rank_str((&arr, 0, arr.len())).unwrap();
/// // Output: [3, 1, 2] - lexicographic ranking
/// ```
#[inline(always)]
pub fn rank_str<T: Integer>(arr: StringAVT<T>) -> Result<IntegerArray<i32>, KernelError> {
    let (array, offset, len) = arr;
    let mask = array.null_mask.as_ref();
    let _ = confirm_mask_capacity(array.len(), mask)?;

    // Gather (local_idx, string) pairs for valid elements in the window
    let mut tuples: Vec<(usize, &str)> = (0..len)
        .filter(|&i| mask.map_or(true, |m| unsafe { m.get_unchecked(offset + i) }))
        .map(|i| (i, unsafe { array.get_unchecked(offset + i) }.unwrap_or("")))
        .collect();

    // Sort by string value
    tuples.sort_by(|a, b| a.1.cmp(&b.1));

    let mut out = vec64![0i32; len];
    let mut out_mask = new_null_mask(len);

    // Assign ranks (1-based), using local output indices
    for (rank, (i, _)) in tuples.iter().enumerate() {
        out[*i] = (rank + 1) as i32;
        unsafe { out_mask.set_unchecked(*i, true) };
    }

    Ok(IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

#[inline(always)]
fn dense_rank_numeric_to<T, F, G>(
    data: &[T],
    mask: Option<&Bitmask>,
    mut sort: F,
    mut eq: G,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) -> Result<(), KernelError>
where
    T: Copy,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
    G: FnMut(&T, &T) -> bool,
{
    let n = data.len();
    let _ = confirm_mask_capacity(n, mask)?;
    let mut uniqs: Vec<T> = (0..n)
        .filter(|&i| mask.map_or(true, |m| unsafe { m.get_unchecked(i) }))
        .map(|i| data[i])
        .collect();

    uniqs.sort_by(&mut sort);
    uniqs.dedup_by(|a, b| eq(&*a, &*b));

    for i in 0..n {
        if mask.map_or(true, |m| unsafe { m.get_unchecked(i) }) {
            let rank = uniqs.binary_search_by(|x| sort(x, &data[i])).unwrap() + 1;
            out[i] = rank as i32;
            unsafe { out_mask.set_unchecked(i, true) };
        } else {
            out[i] = 0;
        }
    }

    Ok(())
}

#[inline(always)]
fn dense_rank_numeric<T, F, G>(
    data: &[T],
    mask: Option<&Bitmask>,
    sort: F,
    eq: G,
) -> Result<IntegerArray<i32>, KernelError>
where
    T: Copy,
    F: FnMut(&T, &T) -> std::cmp::Ordering,
    G: FnMut(&T, &T) -> bool,
{
    let n = data.len();
    let mut out = prealloc_vec::<i32>(n);
    let mut out_mask = Bitmask::new_set_all(n, false);
    dense_rank_numeric_to(data, mask, sort, eq, &mut out, &mut out_mask)?;
    Ok(IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

/// Computes SQL DENSE_RANK() ranking for integer data with consecutive rank assignment.
///
/// Assigns consecutive rank values where tied values receive identical ranks, implementing
/// SQL DENSE_RANK() semantics. Unlike standard ranking, eliminates gaps in rank sequence
/// when ties occur, providing compact rank numbering for analytical applications.
///
/// ## Parameters
/// * `window` - Integer array view containing values for dense ranking
///
/// ## Returns
/// Returns `Result<IntegerArray<i32>, KernelError>` containing:
/// - **Success**: Dense rank values with no gaps in sequence
/// - **Error**: KernelError if capacity validation fails
/// - Zero values for null elements
/// - Null mask indicating positions with valid ranks
///
/// ## Dense Ranking Semantics
/// - **DENSE_RANK() behaviour**: Tied values receive same rank, next rank is consecutive
/// - **No rank gaps**: Eliminates gaps that occur in standard RANK() function
/// - **Unique value counting**: Essentially counts distinct values in sorted order
/// - **Ascending order**: Smaller integer values receive lower (better) ranks
///
/// ## Comparison with RANK()
/// - **RANK()**: [1, 2, 2, 4] for values [10, 20, 20, 30]
/// - **DENSE_RANK()**: [1, 2, 2, 3] for values [10, 20, 20, 30]
///
/// ## Use Cases
/// - **Analytical queries**: SQL DENSE_RANK() window function implementation
/// - **Categorical ranking**: Creating compact categorical orderings
/// - **Percentile calculation**: Foundation for percentile computations without gaps
/// - **Data binning**: Assigning data points to consecutive bins based on value
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::IntegerArray;
/// use simd_kernels::kernels::window::dense_rank_int;
///
/// let arr = IntegerArray::<i32>::from_slice(&[10, 30, 20, 30]);
/// let result = dense_rank_int((&arr, 0, arr.len())).unwrap();
/// // Output: [1, 3, 2, 3] - dense ranking with tied values
/// ```
#[inline(always)]
/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline(always)]
pub fn dense_rank_int_to<T: Ord + Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) -> Result<(), KernelError> {
    dense_rank_numeric_to(data, mask, T::cmp, |a, b| a == b, out, out_mask)
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline(always)]
pub fn dense_rank_float_to<T: Float + Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
    out: &mut [i32],
    out_mask: &mut Bitmask,
) -> Result<(), KernelError> {
    dense_rank_numeric_to(data, mask, total_cmp_f, |a, b| a == b, out, out_mask)
}

pub fn dense_rank_int<T: Ord + Copy>(
    window: IntegerAVT<T>,
) -> Result<IntegerArray<i32>, KernelError> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let null_mask = if len != arr.data.len() {
        &arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len))
    } else {
        &arr.null_mask
    };
    dense_rank_numeric(data, null_mask.as_ref(), |a, b| a.cmp(b), |a, b| a == b)
}

/// Computes SQL DENSE_RANK() ranking for floating-point data with IEEE 754 compliance.
///
/// Implements dense ranking for floating-point values where tied values receive identical
/// consecutive ranks. Handles special floating-point values (NaN, infinity) according
/// to IEEE 754 standards while maintaining dense rank sequence properties.
///
/// ## Parameters
/// * `window` - Float array view containing values for dense ranking
///
/// ## Returns
/// Returns `Result<IntegerArray<i32>, KernelError>` containing:
/// - **Success**: Dense rank values with consecutive numbering
/// - **Error**: KernelError if capacity validation fails
/// - Zero values for null or NaN elements
/// - Null mask indicating positions with valid ranks
///
/// ## Applications
/// - **Scientific ranking**: Dense ranking of experimental measurements
/// - **Statistical analysis**: Percentile calculations without rank gaps
/// - **Financial modeling**: Dense ranking of performance metrics
/// - **Data preprocessing**: Creating ordinal encodings for continuous variables
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::FloatArray;
/// use simd_kernels::kernels::window::dense_rank_float;
///
/// let arr = FloatArray::<f64>::from_slice(&[1.5, 3.14, 2.71, 3.14]);
/// let result = dense_rank_float((&arr, 0, arr.len())).unwrap();
/// // Output: [1, 3, 2, 3] - dense ranking with tied 3.14 values
/// ```
#[inline(always)]
pub fn dense_rank_float<T: Float + Copy>(
    window: FloatAVT<T>,
) -> Result<IntegerArray<i32>, KernelError> {
    let (arr, offset, len) = window;
    let data = &arr.data[offset..offset + len];
    let null_mask = if len != arr.data.len() {
        &arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len))
    } else {
        &arr.null_mask
    };
    dense_rank_numeric(data, null_mask.as_ref(), total_cmp_f, |a, b| a == b)
}

/// Computes SQL DENSE_RANK() ranking for string data with lexicographic dense ordering.
///
/// Implements dense ranking for string values using lexicographic comparison, where
/// identical strings receive the same rank and subsequent ranks remain consecutive.
/// Essential for alphabetical dense ranking and textual categorical analysis.
///
/// ## Parameters
/// * `arr` - String array view containing textual values for dense ranking
///
/// ## Returns
/// Returns `Result<IntegerArray<i32>, KernelError>` containing:
/// - **Success**: Dense rank values with consecutive sequence
/// - **Error**: KernelError if capacity validation fails
/// - Zero values for null string elements
/// - Null mask indicating positions with valid ranks
///
/// ## Dense String Ranking
/// - **DENSE_RANK() semantics**: Identical strings receive same rank, no rank gaps
/// - **Lexicographic ordering**: Standard dictionary-style string comparison
/// - **Case sensitivity**: Maintains case-sensitive comparison ("Apple" ≠ "apple")
/// - **UTF-8 support**: Proper handling of Unicode string sequences
///
/// ## Use Cases
/// - **Alphabetical dense ranking**: Creating compact alphabetical orderings
/// - **Categorical encoding**: Converting string categories to dense integer codes
/// - **Text analytics**: Establishing lexicographic ordinality for text processing
/// - **Database operations**: SQL DENSE_RANK() for string-valued columns
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::StringArray;
/// use simd_kernels::kernels::window::dense_rank_str;
///
/// let arr = StringArray::<u32>::from_slice(&["banana", "apple", "cherry", "apple"]);
/// let result = dense_rank_str((&arr, 0, arr.len())).unwrap();
/// // Output: [2, 1, 3, 1] - dense ranking with tied "apple" values
/// ```
#[inline(always)]
pub fn dense_rank_str<T: Integer>(arr: StringAVT<T>) -> Result<IntegerArray<i32>, KernelError> {
    let (array, offset, len) = arr;
    let mask = array.null_mask.as_ref();
    let _ = confirm_mask_capacity(array.len(), mask)?;

    // Collect all unique valid values in window
    let mut vals: Vec<&str> = (0..len)
        .filter(|&i| mask.map_or(true, |m| unsafe { m.get_unchecked(offset + i) }))
        .map(|i| unsafe { array.get_unchecked(offset + i) }.unwrap_or(""))
        .collect();
    vals.sort();
    vals.dedup();

    let mut out = prealloc_vec::<i32>(len);
    let mut out_mask = Bitmask::new_set_all(len, false);

    for i in 0..len {
        let valid = mask.map_or(true, |m| unsafe { m.get_unchecked(offset + i) });
        if valid {
            let rank = vals
                .binary_search(&unsafe { array.get_unchecked(offset + i) }.unwrap_or(""))
                .unwrap()
                + 1;
            out[i] = rank as i32;
            unsafe { out_mask.set_unchecked(i, true) };
        } else {
            out[i] = 0;
        }
    }

    Ok(IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    })
}

// Lag / Lead / Shift kernels

/// Zero-allocation variant that writes directly to caller-provided output buffers.
///
/// Panics if `out.len() != len`.
#[inline(always)]
fn shift_with_bounds_to<T: Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
    len: usize,
    offset_fn: impl Fn(usize) -> Option<usize>,
    default: T,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    assert_eq!(
        len,
        out.len(),
        "shift_with_bounds_to: input/output length mismatch"
    );
    for i in 0..len {
        if let Some(j) = offset_fn(i) {
            out[i] = data[j];
            let is_valid = mask.map_or(true, |m| unsafe { m.get_unchecked(j) });
            unsafe { out_mask.set_unchecked(i, is_valid) };
        } else {
            out[i] = default;
        }
    }
}

/// Allocating variant: creates new output buffers internally.
#[inline(always)]
fn shift_with_bounds<T: Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
    len: usize,
    offset_fn: impl Fn(usize) -> Option<usize>,
    default: T,
) -> (Vec64<T>, Bitmask) {
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = Bitmask::new_set_all(len, false);
    shift_with_bounds_to(data, mask, len, offset_fn, default, &mut out, &mut out_mask);
    (out, out_mask)
}

#[inline(always)]
fn shift_str_with_bounds<T: Integer>(
    arr: StringAVT<T>,
    offset_fn: impl Fn(usize) -> Option<usize>,
) -> Result<StringArray<T>, KernelError> {
    let (array, offset, len) = arr;
    let src_mask = array.null_mask.as_ref();
    let _ = confirm_mask_capacity(array.len(), src_mask)?;

    // Determine offsets and total bytes required
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe {
        offsets.set_len(len + 1);
    }
    offsets[0] = T::zero();

    let mut total_bytes = 0;
    let mut string_lengths = vec![0usize; len];

    for i in 0..len {
        let byte_len = if let Some(j) = offset_fn(i) {
            let src_idx = offset + j;
            let valid = src_mask.map_or(true, |m| unsafe { m.get_unchecked(src_idx) });
            if valid {
                unsafe { array.get_unchecked(src_idx).unwrap_or("").len() }
            } else {
                0
            }
        } else {
            0
        };
        total_bytes += byte_len;
        string_lengths[i] = byte_len;
        offsets[i + 1] = T::from_usize(total_bytes);
    }

    // Allocate data buffer
    let mut data = Vec64::<u8>::with_capacity(total_bytes);
    let mut out_mask = Bitmask::new_set_all(len, false);

    // Write string content
    for i in 0..len {
        if let Some(j) = offset_fn(i) {
            let src_idx = offset + j;
            let valid = src_mask.map_or(true, |m| unsafe { m.get_unchecked(src_idx) });
            if valid {
                let s = unsafe { array.get_unchecked(src_idx).unwrap_or("") };
                data.extend_from_slice(s.as_bytes());
                unsafe { out_mask.set_unchecked(i, true) };
                continue;
            }
        }
        // Not valid or OOB write nothing
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: data.into(),
        null_mask: Some(out_mask),
    })
}

// Integer

/// Accesses values from previous positions in integer arrays with configurable offset.
///
/// Implements SQL LAG() window function semantics, retrieving values from earlier positions
/// in the array sequence. Essential for time series analysis, trend detection, and
/// comparative analytics requiring access to historical data points.
///
/// ## Parameters
/// * `window` - Integer array view containing sequential data for lag access
/// * `n` - Lag offset specifying how many positions to look backward
///
/// ## Returns
/// Returns an `IntegerArray<T>` containing:
/// - Values from n positions earlier in the sequence
/// - Default values for positions where lag source is unavailable
/// - Null mask indicating validity of lagged values
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::IntegerArray;
/// use simd_kernels::kernels::window::lag_int;
///
/// let arr = IntegerArray::<i32>::from_slice(&[10, 20, 30, 40]);
/// let result = lag_int((&arr, 0, arr.len()), 1);
/// ```
/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn lag_int_to<T: Copy + Default>(
    data: &[T],
    mask: Option<&Bitmask>,
    n: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let len = data.len();
    shift_with_bounds_to(
        data,
        mask,
        len,
        |i| if i >= n { Some(i - n) } else { None },
        T::default(),
        out,
        out_mask,
    );
}

#[inline]
pub fn lag_int<T: Copy + Default>(window: IntegerAVT<T>, n: usize) -> IntegerArray<T> {
    let (arr, offset, len) = window;
    let data_window = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = Bitmask::new_set_all(len, false);
    lag_int_to(data_window, mask.as_ref(), n, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Accesses values from future positions in integer arrays with configurable offset.
///
/// Implements SQL LEAD() window function semantics, retrieving values from later positions
/// in the array sequence. Essential for predictive analytics, forward-looking comparisons,
/// and temporal analysis requiring access to future data points.
///
/// ## Parameters
/// * `window` - Integer array view containing sequential data for lead access
/// * `n` - Lead offset specifying how many positions to look forward
///
/// ## Returns
/// Returns an `IntegerArray<T>` containing:
/// - Values from n positions later in the sequence
/// - Default values for positions where lead source is unavailable
/// - Null mask indicating validity of lead values
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::IntegerArray;
/// use simd_kernels::kernels::window::lead_int;
///
/// let arr = IntegerArray::<i32>::from_slice(&[10, 20, 30, 40]);
/// let result = lead_int((&arr, 0, arr.len()), 2);
/// ```
/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn lead_int_to<T: Copy + Default>(
    data: &[T],
    mask: Option<&Bitmask>,
    n: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let len = data.len();
    shift_with_bounds_to(
        data,
        mask,
        len,
        |i| if i + n < len { Some(i + n) } else { None },
        T::default(),
        out,
        out_mask,
    );
}

#[inline]
pub fn lead_int<T: Copy + Default>(window: IntegerAVT<T>, n: usize) -> IntegerArray<T> {
    let (arr, offset, len) = window;
    let data_window = &arr.data[offset..offset + len];
    let mask = arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len));
    let mut out = prealloc_vec::<T>(len);
    let mut out_mask = Bitmask::new_set_all(len, false);
    lead_int_to(data_window, mask.as_ref(), n, &mut out, &mut out_mask);
    IntegerArray {
        data: out.into(),
        null_mask: Some(out_mask),
    }
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn lag_float_to<T: Copy + num_traits::Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    n: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let len = data.len();
    shift_with_bounds_to(
        data,
        mask,
        len,
        |i| if i >= n { Some(i - n) } else { None },
        T::zero(),
        out,
        out_mask,
    );
}

/// Zero-allocation variant that writes directly to caller-provided output buffers.
#[inline]
pub fn lead_float_to<T: Copy + num_traits::Zero>(
    data: &[T],
    mask: Option<&Bitmask>,
    n: usize,
    out: &mut [T],
    out_mask: &mut Bitmask,
) {
    let len = data.len();
    shift_with_bounds_to(
        data,
        mask,
        len,
        |i| if i + n < len { Some(i + n) } else { None },
        T::zero(),
        out,
        out_mask,
    );
}

/// Accesses values from previous positions in floating-point arrays with IEEE 754 compliance.
#[inline]
pub fn lag_float<T: Copy + num_traits::Zero>(window: FloatAVT<T>, n: usize) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data_window = &arr.data[offset..offset + len];
    let mask_window = if len != arr.data.len() {
        arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len))
    } else {
        arr.null_mask.clone()
    };
    let (data, null_mask) = shift_with_bounds(
        data_window,
        mask_window.as_ref(),
        len,
        |i| if i >= n { Some(i - n) } else { None },
        T::zero(),
    );
    FloatArray {
        data: data.into(),
        null_mask: Some(null_mask),
    }
}

/// Accesses values from future positions in floating-point arrays with IEEE 754 compliance.
///
/// Implements SQL LEAD() function for floating-point data, retrieving values from later
/// positions while preserving IEEE 754 semantics. Essential for forward-looking analysis
/// and predictive modeling with continuous numerical data.
///
/// ## Parameters
/// * `window` - Float array view containing sequential floating-point data
/// * `n` - Lead offset specifying forward position distance
///
/// ## Returns
/// Returns a `FloatArray<T>` containing:
/// - Floating-point values from n positions later
/// - Zero values for positions beyond available future
/// - Null mask indicating lead validity and special value propagation
///
/// ## Use Cases
/// - **Predictive analytics**: Accessing future values for comparison and modeling
/// - **Signal analysis**: Forward-looking operations in digital signal processing
/// - **Financial modeling**: Computing forward returns and future value analysis
/// - **Scientific computing**: Implementing forward difference schemes
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::FloatArray;
/// use simd_kernels::kernels::window::lead_float;
///
/// let arr = FloatArray::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4]);
/// let result = lead_float((&arr, 0, arr.len()), 2);
/// // Output: [3.3, 4.4, 0.0, 0.0] - lead by 2 positions
/// ```
#[inline]
pub fn lead_float<T: Copy + num_traits::Zero>(window: FloatAVT<T>, n: usize) -> FloatArray<T> {
    let (arr, offset, len) = window;
    let data_window = &arr.data[offset..offset + len];
    let mask_window = if len != arr.data.len() {
        arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len))
    } else {
        arr.null_mask.clone()
    };
    let (data, null_mask) = shift_with_bounds(
        data_window,
        mask_window.as_ref(),
        len,
        |i| if i + n < len { Some(i + n) } else { None },
        T::zero(),
    );
    FloatArray {
        data: data.into(),
        null_mask: Some(null_mask),
    }
}

// String

/// Accesses string values from previous positions with UTF-8 string handling.
///
/// Implements SQL LAG() function for string data, retrieving textual values from earlier
/// positions in the array sequence. Essential for textual analysis, sequential string
/// processing, and comparative text analytics.
///
/// ## Parameters
/// * `arr` - String array view containing sequential textual data
/// * `n` - Lag offset specifying backward position distance
///
/// ## Returns
/// Returns `Result<StringArray<T>, KernelError>` containing:
/// - **Success**: String values from n positions earlier
/// - **Error**: KernelError if string processing fails
/// - Empty strings for positions with insufficient history
/// - Null mask indicating lag validity and source availability
///
/// ## String Lag Semantics
/// - **UTF-8 preservation**: Maintains proper UTF-8 encoding throughout operation
/// - **Null propagation**: Null strings in source positions result in null outputs
/// - **Memory management**: Efficient string copying and allocation strategies
/// - **Boundary handling**: Positions without history receive empty string defaults
///
/// ## Applications
/// - **Text analysis**: Comparing current text with previous entries
/// - **Sequential processing**: Analysing patterns in ordered textual data
/// - **Log analysis**: Accessing previous log entries for context
/// - **Natural language processing**: Context-aware text processing with history
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::StringArray;
/// use simd_kernels::kernels::window::lag_str;
///
/// let arr = StringArray::<u32>::from_slice(&["first", "second", "third"]);
/// let result = lag_str((&arr, 0, arr.len()), 1).unwrap();
/// // Output: ["", "first", "second"] - strings lagged by 1 position
/// ```
#[inline]
pub fn lag_str<T: Integer>(arr: StringAVT<T>, n: usize) -> Result<StringArray<T>, KernelError> {
    shift_str_with_bounds(arr, |i| if i >= n { Some(i - n) } else { None })
}

/// Accesses string values from future positions with efficient UTF-8 processing.
///
/// Implements SQL LEAD() function for string data, retrieving textual values from later
/// positions in the array sequence. Critical for forward-looking text analysis and
/// sequential string pattern recognition.
///
/// ## Parameters
/// * `arr` - String array view containing sequential textual data
/// * `n` - Lead offset specifying forward position distance
///
/// ## Returns
/// Returns `Result<StringArray<T>, KernelError>` containing:
/// - **Success**: String values from n positions later
/// - **Error**: KernelError if string processing encounters issues
/// - Empty strings for positions beyond available future
/// - Null mask indicating lead validity and source availability
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::StringArray;
/// use simd_kernels::kernels::window::lead_str;
///
/// let arr = StringArray::<u32>::from_slice(&["alpha", "beta", "gamma"]);
/// let result = lead_str((&arr, 0, arr.len()), 1).unwrap();
/// ```
#[inline]
pub fn lead_str<T: Integer>(arr: StringAVT<T>, n: usize) -> Result<StringArray<T>, KernelError> {
    let (_array, _offset, len) = arr;
    shift_str_with_bounds(arr, |i| if i + n < len { Some(i + n) } else { None })
}

// Shift variants
/// Shifts integer array elements by specified offset with bidirectional support.
///
/// Provides unified interface for both LAG and LEAD operations through signed offset
/// parameter. Positive offsets implement LEAD semantics (forward shift), negative
/// offsets implement LAG semantics (backward shift), enabling flexible positional access.
///
/// ## Parameters
/// * `window` - Integer array view containing data for shifting
/// * `offset` - Signed offset: positive for LEAD (forward), negative for LAG (backward), zero for identity
///
/// ## Returns
/// Returns an `IntegerArray<T>` containing:
/// - Shifted integer values according to offset direction
/// - Default values for positions beyond available data
/// - Null mask reflecting validity of shifted positions
///
/// ## Shift Semantics
/// - **Positive offset**: LEAD operation (shift left, access future values)
/// - **Negative offset**: LAG operation (shift right, access past values)
/// - **Zero offset**: Identity operation (returns original array)
/// - **Boundary handling**: Out-of-bounds positions receive default values
///
/// ## Applications
/// - **Time series analysis**: Flexible temporal shifting for comparison operations
/// - **Sequence processing**: Bidirectional access in ordered integer sequences
/// - **Algorithm implementation**: Building blocks for complex windowing operations
/// - **Data transformation**: Positional transformations in numerical datasets
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::IntegerArray;
/// use simd_kernels::kernels::window::shift_int;
///
/// let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]);
/// let lag = shift_int((&arr, 0, arr.len()), -1);  // LAG by 1
/// let lead = shift_int((&arr, 0, arr.len()), 2);  // LEAD by 2
/// // lag:  [0, 1, 2, 3] - previous values
/// // lead: [3, 4, 0, 0] - future values
/// ```
#[inline(always)]
pub fn shift_int<T: Copy + Default>(window: IntegerAVT<T>, offset: isize) -> IntegerArray<T> {
    let (arr, win_offset, win_len) = window;
    if offset == 0 {
        return IntegerArray {
            data: Vec64::from_slice(&arr.data[win_offset..win_offset + win_len]).into(),
            null_mask: if win_len != arr.data.len() {
                arr.null_mask
                    .as_ref()
                    .map(|m| m.slice_clone(win_offset, win_len))
            } else {
                arr.null_mask.clone()
            },
        };
    } else if offset > 0 {
        lead_int((arr, win_offset, win_len), offset as usize)
    } else {
        lag_int((arr, win_offset, win_len), offset.unsigned_abs())
    }
}

/// Shifts floating-point array elements with IEEE 754 compliance and bidirectional support.
///
/// Unified shifting interface for floating-point data supporting both LAG and LEAD semantics
/// through signed offset parameter. Maintains IEEE 754 standards for special value handling
/// while providing efficient bidirectional positional access.
///
/// ## Parameters
/// * `window` - Float array view containing data for shifting
/// * `offset` - Signed offset: positive for LEAD (forward), negative for LAG (backward), zero for identity
///
/// ## Returns
/// Returns a `FloatArray<T>` containing:
/// - Shifted floating-point values preserving IEEE 754 semantics
/// - Zero values for positions beyond data boundaries
/// - Null mask indicating validity of shifted positions
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::FloatArray;
/// use simd_kernels::kernels::window::shift_float;
///
/// let arr = FloatArray::<f64>::from_slice(&[1.1, 2.2, 3.3, 4.4]);
/// let backward = shift_float((&arr, 0, arr.len()), -2); // LAG by 2
/// let forward = shift_float((&arr, 0, arr.len()), 1);   // LEAD by 1
/// ```
#[inline(always)]
pub fn shift_float<T: Copy + num_traits::Zero>(
    window: FloatAVT<T>,
    offset: isize,
) -> FloatArray<T> {
    let (arr, win_offset, win_len) = window;
    if offset == 0 {
        return FloatArray {
            data: Vec64::from_slice(&arr.data[win_offset..win_offset + win_len]).into(),
            null_mask: if win_len != arr.data.len() {
                arr.null_mask
                    .as_ref()
                    .map(|m| m.slice_clone(win_offset, win_len))
            } else {
                arr.null_mask.clone()
            },
        };
    } else if offset > 0 {
        lead_float((arr, win_offset, win_len), offset as usize)
    } else {
        lag_float((arr, win_offset, win_len), offset.unsigned_abs())
    }
}

/// Shifts string array elements with UTF-8 integrity and bidirectional offset support.
///
/// String shifting function supporting both LAG and LEAD operations through
/// signed offset parameter. Maintains UTF-8 encoding integrity while providing flexible
/// positional access for textual sequence analysis.
///
/// ## Parameters
/// * `arr` - String array view containing textual data for shifting
/// * `shift_offset` - Signed offset: positive for LEAD (forward), negative for LAG (backward), zero for identity
///
/// ## Returns
/// Returns `Result<StringArray<T>, KernelError>` containing:
/// - **Success**: Shifted string values maintaining UTF-8 integrity
/// - **Error**: KernelError if string processing encounters issues
/// - Empty strings for positions beyond data boundaries
/// - Null mask reflecting validity of shifted string positions
///
/// ## Shift Semantics
/// - **Positive offset**: LEAD operation accessing future string values
/// - **Negative offset**: LAG operation accessing historical string values
/// - **Zero offset**: Identity operation (returns cloned array slice)
/// - **Boundary conditions**: Out-of-range positions produce empty strings
///
/// ## Examples
/// ```rust,ignore
/// use minarrow::StringArray;
/// use simd_kernels::kernels::window::shift_str;
///
/// let arr = StringArray::<u32>::from_slice(&["one", "two", "three"]);
/// let back = shift_str((&arr, 0, arr.len()), -1).unwrap(); // LAG
/// let forward = shift_str((&arr, 0, arr.len()), 1).unwrap(); // LEAD
/// // back:    ["", "one", "two"]
/// // forward: ["two", "three", ""]
/// ```
#[inline(always)]
pub fn shift_str<T: Integer>(
    arr: StringAVT<T>,
    shift_offset: isize,
) -> Result<StringArray<T>, KernelError> {
    if shift_offset == 0 {
        // Return this slice's window as a cloned StringArray
        let (array, off, len) = arr;
        Ok(array.slice_clone(off, len))
    } else if shift_offset > 0 {
        lead_str(arr, shift_offset as usize)
    } else {
        lag_str(arr, shift_offset.unsigned_abs())
    }
}

#[cfg(test)]
mod tests {
    use minarrow::structs::variants::float::FloatArray;
    use minarrow::structs::variants::integer::IntegerArray;
    use minarrow::structs::variants::string::StringArray;
    use minarrow::{Bitmask, BooleanArray};

    use super::*;

    // ─────────────────────────── Helpers ───────────────────────────

    /// Build a `Bitmask` from booleans.
    fn bm(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            m.set(i, b);
        }
        m
    }

    /// Simple equality for `IntegerArray<T>`
    fn expect_int<T: PartialEq + std::fmt::Debug + Clone>(
        arr: &IntegerArray<T>,
        values: &[T],
        valid: &[bool],
    ) {
        assert_eq!(arr.data.as_slice(), values);
        let mask = arr.null_mask.as_ref().expect("mask missing");
        for (i, &v) in valid.iter().enumerate() {
            assert_eq!(mask.get(i), v, "mask bit {}", i);
        }
    }

    /// Simple equality for `FloatArray<T>`
    fn expect_float<T: num_traits::Float + std::fmt::Debug>(
        arr: &FloatArray<T>,
        values: &[T],
        valid: &[bool],
        eps: T,
    ) {
        let data = arr.data.as_slice();
        assert_eq!(data.len(), values.len());
        for (a, b) in data.iter().zip(values.iter()) {
            assert!((*a - *b).abs() <= eps, "value mismatch {:?} vs {:?}", a, b);
        }
        let mask = arr.null_mask.as_ref().expect("mask missing");
        for (i, &v) in valid.iter().enumerate() {
            assert_eq!(mask.get(i), v);
        }
    }

    // ───────────────────────── Rolling kernels ─────────────────────────

    #[test]
    fn test_rolling_sum_int_basic() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4, 5]);
        let res = rolling_sum_int((&arr, 0, arr.len()), 3);
        expect_int(&res, &[0, 0, 6, 9, 12], &[false, false, true, true, true]);
    }

    #[test]
    fn test_rolling_sum_int_masked() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]);
        arr.null_mask = Some(bm(&[true, false, true, true]));
        let res = rolling_sum_int((&arr, 0, arr.len()), 2);
        expect_int(
            &res,
            &[0, 0, 3, 7],
            &[false, false, false, true], // window valid only when no nulls in window
        );
    }

    #[test]
    fn test_rolling_sum_float() {
        let arr = FloatArray::<f32>::from_slice(&[1.0, 2.0, 3.0]);
        let res = rolling_sum_float((&arr, 0, arr.len()), 2);
        expect_float(&res, &[0.0, 0.0, 5.0], &[false, false, true], 1e-6f32);
    }

    #[test]
    fn test_rolling_sum_bool() {
        let bools = BooleanArray::from_slice(&[true, true, false, true]);
        let res = rolling_sum_bool((&bools, 0, bools.len()), 2); // counts trues over window
        expect_int(&res, &[0, 0, 1, 1], &[false, false, true, true]);
    }

    #[test]
    fn test_rolling_min_max_mean_count() {
        let arr = IntegerArray::<i32>::from_slice(&[3, 1, 4, 1, 5]);
        // min
        let rmin = rolling_min_int((&arr, 0, arr.len()), 2);
        expect_int(&rmin, &[0, 0, 1, 1, 1], &[false, false, true, true, true]);

        // max
        let rmax = rolling_max_int((&arr, 0, arr.len()), 3);
        expect_int(&rmax, &[0, 0, 4, 4, 5], &[false, false, true, true, true]);

        // mean
        let rmean = rolling_mean_int((&arr, 0, arr.len()), 2);
        expect_float(
            &rmean,
            &[0.0, 0.0, 2.5, 2.5, 3.0],
            &[false, false, true, true, true],
            1e-12,
        );

        // count
        let cnt = rolling_count((0, 5), 3);
        expect_int(&cnt, &[0, 0, 3, 3, 3], &[false, false, true, true, true]);
    }

    // ───────────────────────── Rank / Dense-rank ─────────────────────────

    #[test]
    fn test_rank_int_basic() {
        let arr = IntegerArray::<i32>::from_slice(&[30, 10, 20]);
        let res = rank_int((&arr, 0, arr.len()));
        expect_int(&res, &[3, 1, 2], &[true, true, true]);
    }

    #[test]
    fn test_rank_float_with_nulls() {
        let mut arr = FloatArray::<f64>::from_slice(&[2.0, 1.0, 3.0]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let res = rank_float((&arr, 0, arr.len()));
        expect_int(&res, &[2, 0, 3], &[true, false, true]);
    }

    #[test]
    fn test_dense_rank_str_duplicates() {
        let arr = StringArray::<u32>::from_slice(&["b", "a", "b", "c"]);
        let res = dense_rank_str((&arr, 0, arr.len())).unwrap();
        expect_int(&res, &[2, 1, 2, 3], &[true, true, true, true]);
    }

    #[test]
    fn test_dense_rank_str_duplicates_chunk() {
        // Windowed over ["x", "b", "a", "b", "c", "y"]
        let arr = StringArray::<u32>::from_slice(&["x", "b", "a", "b", "c", "y"]);
        let res = dense_rank_str((&arr, 1, 4)).unwrap(); // window is "b", "a", "b", "c"
        expect_int(&res, &[2, 1, 2, 3], &[true, true, true, true]);
    }

    // ───────────────────────── Lag / Lead / Shift ─────────────────────────

    #[test]
    fn test_lag_lead_int() {
        let arr = IntegerArray::<i32>::from_slice(&[10, 20, 30, 40]);
        let lag1 = lag_int((&arr, 0, arr.len()), 1);
        expect_int(&lag1, &[0, 10, 20, 30], &[false, true, true, true]);

        let lead2 = lead_int((&arr, 0, arr.len()), 2);
        expect_int(&lead2, &[30, 40, 0, 0], &[true, true, false, false]);
    }

    #[test]
    fn test_shift_float_positive_negative_zero() {
        let arr = FloatArray::<f32>::from_slice(&[1.0, 2.0, 3.0]);
        let s0 = shift_float((&arr, 0, arr.len()), 0);
        assert_eq!(s0.data, arr.data); // exact copy

        let s1 = shift_float((&arr, 0, arr.len()), 1);
        expect_float(&s1, &[2.0, 3.0, 0.0], &[true, true, false], 1e-6f32);

        let s_neg = shift_float((&arr, 0, arr.len()), -1);
        expect_float(&s_neg, &[0.0, 1.0, 2.0], &[false, true, true], 1e-6f32);
    }

    #[test]
    fn test_lag_lead_str() {
        let arr = StringArray::<u32>::from_slice(&["a", "b", "c"]);
        let l1 = lag_str((&arr, 0, arr.len()), 1).unwrap();
        assert_eq!(l1.get(0), None);
        assert_eq!(l1.get(1), Some("a"));
        assert_eq!(l1.get(2), Some("b"));

        let d1 = lead_str((&arr, 0, arr.len()), 1).unwrap();
        assert_eq!(d1.get(0), Some("b"));
        assert_eq!(d1.get(1), Some("c"));
        assert_eq!(d1.get(2), None);
    }

    #[test]
    fn test_lag_lead_str_chunk() {
        // Window is ["x", "a", "b", "c", "y"], test on chunk "a", "b", "c"
        let arr = StringArray::<u32>::from_slice(&["x", "a", "b", "c", "y"]);
        let l1 = lag_str((&arr, 1, 3), 1).unwrap();
        assert_eq!(l1.get(0), None);
        assert_eq!(l1.get(1), Some("a"));
        assert_eq!(l1.get(2), Some("b"));

        let d1 = lead_str((&arr, 1, 3), 1).unwrap();
        assert_eq!(d1.get(0), Some("b"));
        assert_eq!(d1.get(1), Some("c"));
        assert_eq!(d1.get(2), None);
    }

    #[test]
    fn test_rolling_sum_int_edge_windows() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4, 5]);

        // window = 0 -> all zeros + mask all false
        let r0 = rolling_sum_int((&arr, 0, arr.len()), 0);
        assert_eq!(r0.data.as_slice(), &[0, 0, 0, 0, 0]);
        assert_eq!(r0.null_mask.as_ref().unwrap().all_unset(), true);

        // window = 1 -> identity
        let r1 = rolling_sum_int((&arr, 0, arr.len()), 1);
        assert_eq!(r1.data.as_slice(), &[1, 2, 3, 4, 5]);
        assert!(r1.null_mask.as_ref().unwrap().all_set());

        // window > len -> all zero + all false
        let r_large = rolling_sum_int((&arr, 0, arr.len()), 10);
        assert_eq!(r_large.data.as_slice(), &[0, 0, 0, 0, 0]);
        assert_eq!(r_large.null_mask.as_ref().unwrap().all_unset(), true);
    }

    #[test]
    fn test_rolling_sum_float_masked_nulls_propagate() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        // null in the middle
        arr.null_mask = Some(bm(&[true, true, false, true]));
        let r = rolling_sum_float((&arr, 0, arr.len()), 2);
        //   i=0: <full-window, 0.0, false>
        //   i=1: first full-window -> cleared -> 0.0, false
        //   i=2: window contains null -> mask=false, but value = 2.0
        //   i=3: window contains null -> mask=false, but value = 4.0
        expect_float(
            &r,
            &[0.0, 0.0, 2.0, 4.0],
            &[false, false, false, false],
            1e-6,
        );
    }

    #[test]
    fn test_rolling_sum_bool_with_nulls() {
        let mut b = BooleanArray::from_slice(&[true, false, true, true]);
        b.null_mask = Some(bm(&[true, false, true, true]));
        let r = rolling_sum_bool((&b, 0, b.len()), 2);
        // windows [t,f], [f,t], [t,t] -> only last window is all non-null
        expect_int(&r, &[0, 0, 1, 2], &[false, false, false, true]);
    }

    #[test]
    fn test_lag_str_null_propagation() {
        let mut arr = StringArray::<u32>::from_slice(&["x", "y", "z"]);
        arr.null_mask = Some(bm(&[true, false, true])); // y is null
        let lag1 = lag_str((&arr, 0, arr.len()), 1).unwrap();
        assert_eq!(lag1.get(0), None); // no source -> null
        assert_eq!(lag1.get(1), Some("x"));
        assert_eq!(lag1.get(2), None); // source was null
        let m = lag1.null_mask.unwrap();
        assert_eq!(m.get(0), false);
        assert_eq!(m.get(1), true);
        assert_eq!(m.get(2), false);
    }

    #[test]
    fn test_lag_str_null_propagation_chunk() {
        // Window ["w", "x", "y", "z", "q"], test on chunk "x", "y", "z"
        let mut arr = StringArray::<u32>::from_slice(&["w", "x", "y", "z", "q"]);
        arr.null_mask = Some(bm(&[true, true, false, true, true]));
        let lag1 = lag_str((&arr, 1, 3), 1).unwrap();
        assert_eq!(lag1.get(0), None); // "x", index 0 in chunk, no source
        assert_eq!(lag1.get(1), Some("x")); // "y", index 1 pulls "x" (valid)
        assert_eq!(lag1.get(2), None); // "z", index 2 pulls "y" (invalid)
        let m = lag1.null_mask.unwrap();
        assert_eq!(m.get(0), false);
        assert_eq!(m.get(1), true);
        assert_eq!(m.get(2), false);
    }

    #[test]
    fn test_shift_str_large_offset() {
        let arr = StringArray::<u32>::from_slice(&["a", "b", "c"]);
        let shifted = shift_str((&arr, 0, arr.len()), 10).unwrap(); // > len
        assert_eq!(shifted.len(), 3);
        for i in 0..3 {
            assert_eq!(shifted.get(i), None);
            assert_eq!(shifted.null_mask.as_ref().unwrap().get(i), false);
        }
    }

    #[test]
    fn test_shift_str_large_offset_chunk() {
        // Window ["w", "a", "b", "c", "x"]
        let arr = StringArray::<u32>::from_slice(&["w", "a", "b", "c", "x"]);
        let shifted = shift_str((&arr, 1, 3), 10).unwrap(); // window is "a","b","c"
        assert_eq!(shifted.len(), 3);
        for i in 0..3 {
            assert_eq!(shifted.get(i), None);
            assert_eq!(shifted.null_mask.as_ref().unwrap().get(i), false);
        }
    }

    #[test]
    fn test_rank_str_ties_and_nulls() {
        let mut arr = StringArray::<u32>::from_slice(&["a", "b", "a", "c"]);
        arr.null_mask = Some(bm(&[true, true, false, true]));
        let r = rank_str((&arr, 0, arr.len())).unwrap();
        // valid positions: 0="a"(rank1),1="b"(rank3),2=null,3="c"(rank4)
        expect_int(&r, &[1, 2, 0, 3], &[true, true, false, true]);
    }

    #[test]
    fn test_rank_str_ties_and_nulls_chunk() {
        // Window ["w", "a", "b", "a", "c"]
        let mut arr = StringArray::<u32>::from_slice(&["w", "a", "b", "a", "c"]);
        arr.null_mask = Some(bm(&[true, true, true, false, true]));
        let r = rank_str((&arr, 1, 4)).unwrap(); // "a","b","a","c"
        expect_int(&r, &[1, 2, 0, 3], &[true, true, false, true]);
    }
}
