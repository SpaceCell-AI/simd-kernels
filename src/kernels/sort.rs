// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Sorting Algorithms Kernels Module** - *Array Sorting and Ordering Operations*
//!
//! Sorting kernels for ordering operations across Arrow-compatible
//! data types with null-aware semantics and optimised comparison strategies. Essential foundation
//! for analytical operations requiring ordered data access and ranking computations.
//!
//! Regular sorts here alter the actual data.
//! The argsort variants return the indices.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::cmp::Ordering;

#[cfg(feature = "simd")]
use std::simd::{
    Mask, Select, Simd,
    cmp::SimdPartialOrd,
    num::{SimdFloat, SimdUint},
};

use minarrow::{Bitmask, BooleanArray, CategoricalArray, Integer, MaskedArray, Vec64};
use num_traits::{Float, NumCast, ToPrimitive, Zero};

/// Sort algorithm selection for argsort operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortAlgorithm {
    /// Automatically select the best algorithm for the data type.
    ///
    /// When `config.stable = true`, this falls back to comparison sort
    /// which preserves relative order of equal elements. When `stable = false`,
    /// radix sort may be selected for eligible types.
    #[default]
    Auto,
    /// Standard comparison sort, O(n log n).
    ///
    /// Respects `config.stable`: when true, uses a stable sort that preserves
    /// relative order of equal elements; when false, uses an unstable sort.
    Comparison,
    /// LSD radix sort for integers, O(n*k).
    ///
    /// Always unstable - the `config.stable` flag is ignored when this
    /// algorithm is selected.
    Radix,
    /// SIMD-accelerated radix sort (requires `simd_sort` feature).
    ///
    /// Always unstable - the `config.stable` flag is ignored when this
    /// algorithm is selected.
    #[cfg(feature = "simd_sort")]
    Simd,
}

/// Configuration for argsort with algorithm selection
#[derive(Debug, Clone)]
pub struct ArgsortConfig {
    pub descending: bool,
    pub algorithm: SortAlgorithm,
    /// Use stable sort to preserve relative order of equal elements. Default: true.
    pub stable: bool,
    /// Enable parallel sorting for large arrays. Default: false.
    pub parallel: bool,
}

impl Default for ArgsortConfig {
    fn default() -> Self {
        Self {
            descending: false,
            algorithm: SortAlgorithm::default(),
            stable: true,
            parallel: false,
        }
    }
}

impl ArgsortConfig {
    /// Create a new config with default settings (ascending, auto algorithm)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set descending order
    pub fn descending(mut self, descending: bool) -> Self {
        self.descending = descending;
        self
    }

    /// Set algorithm
    pub fn algorithm(mut self, algorithm: SortAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set stable sorting to preserve relative order of equal elements
    pub fn stable(mut self, stable: bool) -> Self {
        self.stable = stable;
        self
    }

    /// Enable parallel sorting for large arrays
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Total ordering for f32/f64 as per IEEE 754
///
/// - NaN sorts greater than all numbers, including +inf.
/// - -0.0 and +0.0 are distinct (we can change in future if needed - to filter first).
/// - Preserves all edge case ordering.

#[inline]
pub fn sort_float<T: Float>(slice: &mut [T]) {
    slice.sort_unstable_by(total_cmp_f);
}
/// TLDR: Instead of comparing the raw bit patterns directly (which places all negative‐sign bit values after the positives),
/// we transform each bit pattern into a “sort key”:
///     => If the sign bit is 1 (negative or negative‐NaN), we invert all bits: !bits.
///     => Otherwise (sign bit 0), we flip only the sign bit: bits ^ 0x80…0.
#[inline(always)]
pub fn total_cmp_f<T: Float>(a: &T, b: &T) -> Ordering {
    // We reinterpret the bits of `T` as either u32 or u64:
    if std::mem::size_of::<T>() == 4 {
        // f32 path
        let ua = unsafe { *(a as *const T as *const u32) };
        let ub = unsafe { *(b as *const T as *const u32) };
        // Negative values get inverted; non-negatives get their sign bit flipped.
        let ka = if ua & 0x8000_0000 != 0 {
            !ua
        } else {
            ua ^ 0x8000_0000
        };
        let kb = if ub & 0x8000_0000 != 0 {
            !ub
        } else {
            ub ^ 0x8000_0000
        };
        ka.cmp(&kb)
    } else if std::mem::size_of::<T>() == 8 {
        // f64 path
        let ua = unsafe { *(a as *const T as *const u64) };
        let ub = unsafe { *(b as *const T as *const u64) };
        let ka = if ua & 0x8000_0000_0000_0000 != 0 {
            !ua
        } else {
            ua ^ 0x8000_0000_0000_0000
        };
        let kb = if ub & 0x8000_0000_0000_0000 != 0 {
            !ub
        } else {
            ub ^ 0x8000_0000_0000_0000
        };
        ka.cmp(&kb)
    } else {
        unreachable!("Only f32 and f64 are valid Float types.")
    }
}

/// Returns a newly sorted Vec64, leaving the original slice untouched.
/// Zero-allocation variant: writes sorted data to caller's output buffer.
/// Panics if `out.len() != data.len()`.
#[inline]
pub fn sorted_float_to<T: Float>(data: &[T], out: &mut [T]) {
    assert_eq!(
        data.len(),
        out.len(),
        "sorted_float_to: input/output length mismatch"
    );
    out.copy_from_slice(data);
    sort_float(out);
}

pub fn sorted_float<T: Float>(data: &[T]) -> Vec64<T> {
    let mut v = Vec64::from_slice(data);
    sort_float(&mut v);
    v
}

/// Performs in-place unstable sorting of integer slices with optimal performance.
///
/// High-performance sorting function optimised for integer types using unstable sort algorithms
/// that prioritise speed over preserving the relative order of equal elements.
///
/// # Type Parameters
/// - `T`: Integer type implementing `Ord + Copy` (i8, i16, i32, i64, u8, u16, u32, u64, etc.)
///
/// # Parameters
/// - `slice`: Mutable slice to be sorted in-place
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sort_int;
///
/// let mut data = [64i32, 34, 25, 12, 22, 11, 90];
/// sort_int(&mut data);
/// // data is now [11, 12, 22, 25, 34, 64, 90]
/// ```
#[inline]
pub fn sort_int<T: Ord + Copy>(slice: &mut [T]) {
    slice.sort_unstable();
}

/// Creates a new sorted copy of integer data in a Vec64 container.
///
/// Clones input data into a Vec64 and sorts it using the optimised
/// integer sorting algorithm. Returns a new sorted container while leaving the original data
/// unchanged.
///
/// # Type Parameters
/// - `T`: Integer type implementing `Ord + Copy` (i8, i16, i32, i64, u8, u16, u32, u64, etc.)
///
/// # Parameters
/// - `data`: Source slice to be copied and sorted
///
/// # Returns
/// A new `Vec64<T>` containing the sorted elements from the input slice.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sorted_int;
///
/// let data = [64i32, 34, 25, 12, 22, 11, 90];
/// let sorted = sorted_int(&data);
/// // sorted contains [11, 12, 22, 25, 34, 64, 90]
/// // original data unchanged
/// ```
/// Zero-allocation variant: writes sorted data to caller's output buffer.
/// Panics if `out.len() != data.len()`.
#[inline]
pub fn sorted_int_to<T: Ord + Copy>(data: &[T], out: &mut [T]) {
    assert_eq!(
        data.len(),
        out.len(),
        "sorted_int_to: input/output length mismatch"
    );
    out.copy_from_slice(data);
    sort_int(out);
}

pub fn sorted_int<T: Ord + Copy>(data: &[T]) -> Vec64<T> {
    let mut v = Vec64::from_slice(data);
    sort_int(&mut v);
    v
}

/// Performs in-place unstable sorting of string slice references with lexicographic ordering.
///
/// High-performance sorting function for string references using unstable sort algorithms.
/// Efficient lexicographic ordering for string processing, text analysis, and data organisation tasks.
///
/// # Parameters
/// - `slice`: Mutable slice of string references to be sorted in-place
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sort_str;
///
/// let mut words = ["zebra", "apple", "banana", "cherry"];
/// sort_str(&mut words);
/// // words is now ["apple", "banana", "cherry", "zebra"]
/// ```
#[inline]
pub fn sort_str(slice: &mut [&str]) {
    slice.sort_unstable();
}

/// Creates a new sorted collection of owned strings from string references.
///
/// Copies string references into owned String objects within a Vec64
/// container and sorts them lexicographically. Returns a new sorted collection while preserving
/// the original string references for scenarios requiring owned string data.
///
/// # Parameters
/// - `data`: Source slice of string references to be copied and sorted
///
/// # Returns
/// A new `Vec64<String>` containing owned, sorted string copies from the input references.
///
/// # Memory Allocation
/// - String allocation: Each string reference copied into a new String object
/// - Container allocation: Vec64 provides 64-byte aligned storage for optimal performance
/// - Heap usage: Total memory proportional to sum of string lengths plus container overhead
///
/// # Performance Characteristics
/// - O(n) string allocation and copying (proportional to total string length)
/// - O(n log n) sorting time complexity with string comparison overhead
/// - Memory overhead: Requires additional heap space for all string content
///
/// # String Ownership
/// - Input references: Original string references remain unchanged
/// - Output strings: New owned String objects independent of input lifetime
/// - Memory management: Vec64<String> handles automatic cleanup of owned strings
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::sort::sorted_str;
///
/// let words = ["zebra", "apple", "banana", "cherry"];
/// let sorted = sorted_str(&words);
/// // sorted contains owned Strings: ["apple", "banana", "cherry", "zebra"]
/// // original words slice unchanged
/// ```
pub fn sorted_str(data: &[&str]) -> Vec64<String> {
    let mut v = Vec64::from_slice(data);
    sort_str(&mut v);
    v.iter().map(|s| s.to_string()).collect()
}

/// For StringArray as contiguous utf8 buffer plus offsets.
/// Assumes offsets + values as in minarrow StringArray.
pub fn sort_string_array(offsets: &[usize], values: &str) -> (Vec64<usize>, String) {
    let n = offsets.len() - 1;
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_unstable_by(|&i, &j| {
        let a = &values[offsets[i]..offsets[i + 1]];
        let b = &values[offsets[j]..offsets[j + 1]];
        a.cmp(b)
    });

    // Precompute total length for the output string buffer
    let total_bytes: usize = indices
        .iter()
        .map(|&idx| offsets[idx + 1] - offsets[idx])
        .sum();

    // Preallocate buffers
    let mut new_values = String::with_capacity(total_bytes);
    let mut new_offsets = Vec64::<usize>::with_capacity(n + 1);

    // Pre-extend buffers for unchecked writes
    unsafe {
        new_offsets.set_len(n + 1);
    }
    unsafe {
        new_values.as_mut_vec().set_len(total_bytes);
    }

    let values_bytes = values.as_bytes();
    let out_bytes = unsafe { new_values.as_mut_vec() };

    // First offset is always 0
    unsafe {
        *new_offsets.get_unchecked_mut(0) = 0;
    }
    let mut current_offset = 0;
    let mut out_ptr = 0;
    for (i, &idx) in indices.iter().enumerate() {
        let start = offsets[idx];
        let end = offsets[idx + 1];
        let len = end - start;
        // Copy string bytes
        unsafe {
            std::ptr::copy_nonoverlapping(
                values_bytes.as_ptr().add(start),
                out_bytes.as_mut_ptr().add(out_ptr),
                len,
            );
            current_offset += len;
            *new_offsets.get_unchecked_mut(i + 1) = current_offset;
        }
        out_ptr += len;
    }
    // SAFETY: We filled up to `total_bytes`
    unsafe {
        new_values.as_mut_vec().set_len(current_offset);
    }

    (new_offsets, new_values)
}

/// Sorts a CategoricalArray lexically by its unique values, returning new indices and mask.
/// The category dictionary is preserved. Nulls sort first.
pub fn sort_categorical_lexical<T: Ord + Copy + Zero + NumCast + Integer>(
    cat: &CategoricalArray<T>,
) -> (Vec64<T>, Option<Bitmask>) {
    let len = cat.data.len();
    let mut indices: Vec<usize> = (0..len).collect();

    // Sort indices: nulls first, then by value, stable.
    indices.sort_by(|&i, &j| {
        let a_is_null = cat.is_null(i);
        let b_is_null = cat.is_null(j);
        match (a_is_null, b_is_null) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                // compare by the string value of the keys
                let a_key = &cat.unique_values[cat.data[i].to_usize()];
                let b_key = &cat.unique_values[cat.data[j].to_usize()];
                a_key.cmp(b_key)
            }
        }
    });

    // Build output data and mask
    let mut sorted_data = Vec64::<T>::with_capacity(len);
    let mut sorted_mask = cat
        .null_mask
        .as_ref()
        .map(|_| Bitmask::new_set_all(len, false));

    for (out_i, &orig_i) in indices.iter().enumerate() {
        sorted_data.push(cat.data[orig_i]);
        if let Some(ref mask) = cat.null_mask {
            if let Some(ref mut sm) = sorted_mask {
                if unsafe { mask.get_unchecked(orig_i) } {
                    unsafe { sm.set_unchecked(out_i, true) };
                }
            }
        }
    }

    (sorted_data, sorted_mask)
}

/// Unpacks a Bitmask into a Vec<bool>
#[inline]
fn unpack_bitmask(mask: &Bitmask) -> Vec64<bool> {
    let mut out = Vec64::with_capacity(mask.len);
    for i in 0..mask.len {
        out.push(unsafe { mask.get_unchecked(i) });
    }
    out
}

/// Packs a Vec<bool> into a Bitmask
#[inline]
fn pack_bitmask(bits: &[bool]) -> Bitmask {
    let mut mask = Bitmask::new_set_all(bits.len(), false);
    for (i, &b) in bits.iter().enumerate() {
        if b {
            unsafe { mask.set_unchecked(i, true) };
        }
    }
    mask
}

/// Sorts a BooleanArray in-place by value: all false first, then true.
/// Nulls sort first if present.
pub fn sort_boolean_array(arr: &mut BooleanArray<()>) {
    let bits: Vec64<bool> = unpack_bitmask(&arr.data);
    let nulls: Option<Vec64<bool>> = arr.null_mask.as_ref().map(unpack_bitmask);

    let mut indices: Vec<usize> = (0..arr.len).collect();

    // Nulls sort first (is_null = !is_valid)
    indices.sort_unstable_by(|&i, &j| {
        let a_null = !nulls.as_ref().map_or(true, |n| n[i]);
        let b_null = !nulls.as_ref().map_or(true, |n| n[j]);

        match (a_null, b_null) {
            (true, true) => Ordering::Equal,    // both null
            (true, false) => Ordering::Less,    // null < non-null
            (false, true) => Ordering::Greater, // non-null > null
            (false, false) => {
                // Both non-null: false < true
                match (bits[i], bits[j]) {
                    (false, false) => Ordering::Equal,
                    (false, true) => Ordering::Less,
                    (true, false) => Ordering::Greater,
                    (true, true) => Ordering::Equal,
                }
            }
        }
    });

    // Re-pack sorted logical values, forcing value=false for null slots
    let sorted_bits: Vec<bool> = indices
        .iter()
        .map(|&idx| {
            let is_null = !nulls.as_ref().map_or(true, |n| n[idx]);
            if is_null { false } else { bits[idx] }
        })
        .collect();
    arr.data = pack_bitmask(&sorted_bits);

    // Re-pack the (optional) null mask.
    if let Some(null_mask) = arr.null_mask.as_mut() {
        let sorted_nulls: Vec<bool> = indices
            .iter()
            .map(|&idx| nulls.as_ref().unwrap()[idx])
            .collect();
        *null_mask = pack_bitmask(&sorted_nulls);
    }
}

/// Sorts array data and applies the same permutation to the null mask.
pub fn sort_slice_with_mask<T: Ord + Copy>(
    data: &[T],
    mask: Option<&Bitmask>,
) -> (Vec64<T>, Option<Bitmask>) {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&i, &j| data[i].cmp(&data[j]));

    let mut sorted_data = Vec64::<T>::with_capacity(n);
    unsafe {
        sorted_data.set_len(n);
    }
    for (i, &idx) in indices.iter().enumerate() {
        unsafe {
            *sorted_data.get_unchecked_mut(i) = data[idx];
        }
    }

    let sorted_mask = mask.map(|m| {
        let mut out = Bitmask::new_set_all(n, false);
        for (i, &idx) in indices.iter().enumerate() {
            if unsafe { m.get_unchecked(idx) } {
                unsafe { out.set_unchecked(i, true) };
            }
        }
        out
    });

    (sorted_data, sorted_mask)
}

// Argsort - Returns sorted indices (O(n log n) comparison sort)

/// Comparison-based argsort for any Ord type - returns indices that would sort the data
#[inline]
pub fn argsort<T: Ord>(data: &[T], descending: bool) -> Vec<usize> {
    argsort_with_stability(data, descending, false)
}

/// Comparison-based argsort with stability control
#[inline]
pub fn argsort_with_stability<T: Ord>(data: &[T], descending: bool, stable: bool) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    if stable {
        if descending {
            indices.sort_by(|&i, &j| data[j].cmp(&data[i]));
        } else {
            indices.sort_by(|&i, &j| data[i].cmp(&data[j]));
        }
    } else {
        if descending {
            indices.sort_unstable_by(|&i, &j| data[j].cmp(&data[i]));
        } else {
            indices.sort_unstable_by(|&i, &j| data[i].cmp(&data[j]));
        }
    }
    indices
}

/// Argsort for floats with proper NaN handling
#[inline]
pub fn argsort_float<T: Float>(data: &[T], descending: bool) -> Vec<usize> {
    argsort_float_with_stability(data, descending, false)
}

/// Argsort for floats with proper NaN handling and stability control
#[inline]
pub fn argsort_float_with_stability<T: Float>(
    data: &[T],
    descending: bool,
    stable: bool,
) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    if stable {
        if descending {
            indices.sort_by(|&i, &j| total_cmp_f(&data[j], &data[i]));
        } else {
            indices.sort_by(|&i, &j| total_cmp_f(&data[i], &data[j]));
        }
    } else {
        if descending {
            indices.sort_unstable_by(|&i, &j| total_cmp_f(&data[j], &data[i]));
        } else {
            indices.sort_unstable_by(|&i, &j| total_cmp_f(&data[i], &data[j]));
        }
    }
    indices
}

/// Argsort for string slices - returns indices that would sort the data lexicographically
#[inline]
pub fn argsort_str(data: &[&str], descending: bool) -> Vec<usize> {
    argsort_str_with_stability(data, descending, false)
}

/// Argsort for string slices with stability control
#[inline]
pub fn argsort_str_with_stability(data: &[&str], descending: bool, stable: bool) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    if stable {
        if descending {
            indices.sort_by(|&i, &j| data[j].cmp(data[i]));
        } else {
            indices.sort_by(|&i, &j| data[i].cmp(data[j]));
        }
    } else {
        if descending {
            indices.sort_unstable_by(|&i, &j| data[j].cmp(data[i]));
        } else {
            indices.sort_unstable_by(|&i, &j| data[i].cmp(data[j]));
        }
    }
    indices
}

/// Argsort for StringArray (offset-based string storage)
///
/// Returns indices that would sort the strings lexicographically
pub fn argsort_string_array(offsets: &[usize], values: &str, descending: bool) -> Vec<usize> {
    argsort_string_array_with_stability(offsets, values, descending, false)
}

/// Argsort for StringArray with stability control
pub fn argsort_string_array_with_stability(
    offsets: &[usize],
    values: &str,
    descending: bool,
    stable: bool,
) -> Vec<usize> {
    let n = if offsets.is_empty() {
        0
    } else {
        offsets.len() - 1
    };
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let cmp = |i: &usize, j: &usize| {
        let a = &values[offsets[*i]..offsets[*i + 1]];
        let b = &values[offsets[*j]..offsets[*j + 1]];
        if descending { b.cmp(a) } else { a.cmp(b) }
    };
    if stable {
        indices.sort_by(cmp);
    } else {
        indices.sort_unstable_by(cmp);
    }
    indices
}

/// Argsort for CategoricalArray - lexical ordering by string values
///
/// Sorts by the string representation of each category value.
/// Nulls sort first (ascending) or last (descending).
pub fn argsort_categorical_lexical<T: Ord + Copy + Zero + NumCast + Integer>(
    cat: &CategoricalArray<T>,
    descending: bool,
) -> Vec<usize> {
    let len = cat.data.len();
    if len == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..len).collect();

    indices.sort_by(|&i, &j| {
        let a_is_null = cat.is_null(i);
        let b_is_null = cat.is_null(j);

        let ordering = match (a_is_null, b_is_null) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less, // nulls first
            (false, true) => Ordering::Greater,
            (false, false) => {
                let a_key = &cat.unique_values[cat.data[i].to_usize()];
                let b_key = &cat.unique_values[cat.data[j].to_usize()];
                a_key.cmp(b_key)
            }
        };

        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });

    indices
}

/// Argsort for CategoricalArray with custom category order
///
/// Sorts by a user-defined category order. Categories not in the order list
/// are sorted after those in the list, in lexical order.
/// Nulls sort first (ascending) or last (descending).
pub fn argsort_categorical_custom<T: Ord + Copy + Zero + NumCast + Integer>(
    cat: &CategoricalArray<T>,
    category_order: &[&str],
    descending: bool,
) -> Vec<usize> {
    let len = cat.data.len();
    if len == 0 {
        return vec![];
    }

    // Build a lookup map: category string -> order position
    use std::collections::HashMap;
    let order_map: HashMap<&str, usize> = category_order
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();

    let mut indices: Vec<usize> = (0..len).collect();

    indices.sort_by(|&i, &j| {
        let a_is_null = cat.is_null(i);
        let b_is_null = cat.is_null(j);

        let ordering = match (a_is_null, b_is_null) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                let a_key = &cat.unique_values[cat.data[i].to_usize()];
                let b_key = &cat.unique_values[cat.data[j].to_usize()];

                // Get order positions (None means not in custom order)
                let a_pos = order_map.get(a_key.as_str());
                let b_pos = order_map.get(b_key.as_str());

                match (a_pos, b_pos) {
                    (Some(ap), Some(bp)) => ap.cmp(bp),
                    (Some(_), None) => Ordering::Less, // defined order comes first
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => a_key.cmp(b_key), // fall back to lexical
                }
            }
        };

        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });

    indices
}

/// Argsort for boolean arrays - false sorts before true
///
/// Nulls sort first (ascending) or last (descending).
pub fn argsort_boolean(
    data: &Bitmask,
    null_mask: Option<&Bitmask>,
    descending: bool,
) -> Vec<usize> {
    let n = data.len;
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_unstable_by(|&i, &j| {
        let a_null = null_mask.map_or(false, |m| !m.get(i));
        let b_null = null_mask.map_or(false, |m| !m.get(j));

        let ordering = match (a_null, b_null) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => {
                let a_val = data.get(i);
                let b_val = data.get(j);
                a_val.cmp(&b_val)
            }
        };

        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });

    indices
}

// Radix Sort Argsort - O(n·k) for integers

/// LSD Radix argsort for u32 - O(n·k) where k=4 (bytes)
///
/// Significantly faster than comparison sort for integer data.
/// Uses 8-bit radix (256 buckets) with 4 passes.
pub fn argsort_radix_u32(data: &[u32], descending: bool) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut temp_indices = vec![0usize; n];

    // Process 8 bits at a time (4 passes for u32)
    for shift in (0..32).step_by(8) {
        let mut counts = [0usize; 256];
        for &idx in &indices {
            let byte = ((data[idx] >> shift) & 0xFF) as usize;
            counts[byte] += 1;
        }

        let mut positions = [0usize; 256];
        if descending {
            let mut sum = 0;
            for i in (0..256).rev() {
                positions[i] = sum;
                sum += counts[i];
            }
        } else {
            let mut sum = 0;
            for i in 0..256 {
                positions[i] = sum;
                sum += counts[i];
            }
        }

        for &idx in &indices {
            let byte = ((data[idx] >> shift) & 0xFF) as usize;
            temp_indices[positions[byte]] = idx;
            positions[byte] += 1;
        }

        std::mem::swap(&mut indices, &mut temp_indices);
    }

    indices
}

/// LSD Radix argsort for u64 - O(n·k) where k=8 (bytes)
pub fn argsort_radix_u64(data: &[u64], descending: bool) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut temp_indices = vec![0usize; n];

    for shift in (0..64).step_by(8) {
        let mut counts = [0usize; 256];
        for &idx in &indices {
            let byte = ((data[idx] >> shift) & 0xFF) as usize;
            counts[byte] += 1;
        }

        let mut positions = [0usize; 256];
        if descending {
            let mut sum = 0;
            for i in (0..256).rev() {
                positions[i] = sum;
                sum += counts[i];
            }
        } else {
            let mut sum = 0;
            for i in 0..256 {
                positions[i] = sum;
                sum += counts[i];
            }
        }

        for &idx in &indices {
            let byte = ((data[idx] >> shift) & 0xFF) as usize;
            temp_indices[positions[byte]] = idx;
            positions[byte] += 1;
        }

        std::mem::swap(&mut indices, &mut temp_indices);
    }

    indices
}

/// LSD Radix argsort for i32 - uses sign-bit flipping for proper signed ordering
pub fn argsort_radix_i32(data: &[i32], descending: bool) -> Vec<usize> {
    // Convert to unsigned with sign-bit flip for proper ordering
    let unsigned: Vec<u32> = data.iter().map(|&x| (x as u32) ^ 0x8000_0000).collect();
    argsort_radix_u32(&unsigned, descending)
}

/// LSD Radix argsort for i64 - uses sign-bit flipping for proper signed ordering
pub fn argsort_radix_i64(data: &[i64], descending: bool) -> Vec<usize> {
    let unsigned: Vec<u64> = data
        .iter()
        .map(|&x| (x as u64) ^ 0x8000_0000_0000_0000)
        .collect();
    argsort_radix_u64(&unsigned, descending)
}

// SIMD Radix Sort (feature-gated)

#[cfg(feature = "simd_sort")]
pub mod simd_argsort {
    use std::cmp::Ordering;
    use voracious_radix_sort::{RadixSort, Radixable};

    /// u32 value with index for SIMD argsort
    #[derive(Copy, Clone, Debug)]
    struct IndexedU32 {
        value: u32,
        index: usize,
    }

    impl PartialOrd for IndexedU32 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }

    impl PartialEq for IndexedU32 {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Radixable<u32> for IndexedU32 {
        type Key = u32;
        #[inline]
        fn key(&self) -> Self::Key {
            self.value
        }
    }

    /// u64 value with index for SIMD argsort
    #[derive(Copy, Clone, Debug)]
    struct IndexedU64 {
        value: u64,
        index: usize,
    }

    impl PartialOrd for IndexedU64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }

    impl PartialEq for IndexedU64 {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Radixable<u64> for IndexedU64 {
        type Key = u64;
        #[inline]
        fn key(&self) -> Self::Key {
            self.value
        }
    }

    /// i32 value with index for SIMD argsort
    #[derive(Copy, Clone, Debug)]
    struct IndexedI32 {
        value: i32,
        index: usize,
    }

    impl PartialOrd for IndexedI32 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }

    impl PartialEq for IndexedI32 {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Radixable<i32> for IndexedI32 {
        type Key = i32;
        #[inline]
        fn key(&self) -> Self::Key {
            self.value
        }
    }

    /// i64 value with index for SIMD argsort
    #[derive(Copy, Clone, Debug)]
    struct IndexedI64 {
        value: i64,
        index: usize,
    }

    impl PartialOrd for IndexedI64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }

    impl PartialEq for IndexedI64 {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Radixable<i64> for IndexedI64 {
        type Key = i64;
        #[inline]
        fn key(&self) -> Self::Key {
            self.value
        }
    }

    /// SIMD-accelerated radix argsort for u32
    pub fn argsort_simd_u32(data: &[u32], descending: bool) -> Vec<usize> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut indexed: Vec<IndexedU32> = data
            .iter()
            .enumerate()
            .map(|(index, &value)| IndexedU32 { value, index })
            .collect();

        indexed.voracious_sort();

        if descending {
            indexed.iter().rev().map(|x| x.index).collect()
        } else {
            indexed.iter().map(|x| x.index).collect()
        }
    }

    /// SIMD-accelerated radix argsort for u64
    pub fn argsort_simd_u64(data: &[u64], descending: bool) -> Vec<usize> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut indexed: Vec<IndexedU64> = data
            .iter()
            .enumerate()
            .map(|(index, &value)| IndexedU64 { value, index })
            .collect();

        indexed.voracious_sort();

        if descending {
            indexed.iter().rev().map(|x| x.index).collect()
        } else {
            indexed.iter().map(|x| x.index).collect()
        }
    }

    /// SIMD-accelerated radix argsort for i32
    pub fn argsort_simd_i32(data: &[i32], descending: bool) -> Vec<usize> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut indexed: Vec<IndexedI32> = data
            .iter()
            .enumerate()
            .map(|(index, &value)| IndexedI32 { value, index })
            .collect();

        indexed.voracious_sort();

        if descending {
            indexed.iter().rev().map(|x| x.index).collect()
        } else {
            indexed.iter().map(|x| x.index).collect()
        }
    }

    /// SIMD-accelerated radix argsort for i64
    pub fn argsort_simd_i64(data: &[i64], descending: bool) -> Vec<usize> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut indexed: Vec<IndexedI64> = data
            .iter()
            .enumerate()
            .map(|(index, &value)| IndexedI64 { value, index })
            .collect();

        indexed.voracious_sort();

        if descending {
            indexed.iter().rev().map(|x| x.index).collect()
        } else {
            indexed.iter().map(|x| x.index).collect()
        }
    }
}

/// Argsort for i32 with automatic algorithm selection
///
/// Selects the optimal algorithm based on configuration:
/// - Auto: SIMD (if available) > Radix > Comparison
/// - Radix: LSD radix sort
/// - Comparison: Standard comparison sort
/// - Simd: SIMD-accelerated radix sort (requires feature)
pub fn argsort_auto_i32(data: &[i32], config: &ArgsortConfig) -> Vec<usize> {
    match config.algorithm {
        SortAlgorithm::Comparison => argsort_with_stability(data, config.descending, config.stable),
        SortAlgorithm::Radix => argsort_radix_i32(data, config.descending),
        #[cfg(feature = "simd_sort")]
        SortAlgorithm::Simd => simd_argsort::argsort_simd_i32(data, config.descending),
        SortAlgorithm::Auto => {
            if config.stable {
                return argsort_with_stability(data, config.descending, true);
            }
            #[cfg(feature = "simd_sort")]
            {
                simd_argsort::argsort_simd_i32(data, config.descending)
            }
            #[cfg(not(feature = "simd_sort"))]
            {
                argsort_radix_i32(data, config.descending)
            }
        }
    }
}

/// Argsort for i64 with automatic algorithm selection
pub fn argsort_auto_i64(data: &[i64], config: &ArgsortConfig) -> Vec<usize> {
    match config.algorithm {
        SortAlgorithm::Comparison => argsort_with_stability(data, config.descending, config.stable),
        SortAlgorithm::Radix => argsort_radix_i64(data, config.descending),
        #[cfg(feature = "simd_sort")]
        SortAlgorithm::Simd => simd_argsort::argsort_simd_i64(data, config.descending),
        SortAlgorithm::Auto => {
            if config.stable {
                return argsort_with_stability(data, config.descending, true);
            }
            #[cfg(feature = "simd_sort")]
            {
                simd_argsort::argsort_simd_i64(data, config.descending)
            }
            #[cfg(not(feature = "simd_sort"))]
            {
                argsort_radix_i64(data, config.descending)
            }
        }
    }
}

/// Argsort for u32 with automatic algorithm selection
pub fn argsort_auto_u32(data: &[u32], config: &ArgsortConfig) -> Vec<usize> {
    match config.algorithm {
        SortAlgorithm::Comparison => argsort_with_stability(data, config.descending, config.stable),
        SortAlgorithm::Radix => argsort_radix_u32(data, config.descending),
        #[cfg(feature = "simd_sort")]
        SortAlgorithm::Simd => simd_argsort::argsort_simd_u32(data, config.descending),
        SortAlgorithm::Auto => {
            if config.stable {
                return argsort_with_stability(data, config.descending, true);
            }
            #[cfg(feature = "simd_sort")]
            {
                simd_argsort::argsort_simd_u32(data, config.descending)
            }
            #[cfg(not(feature = "simd_sort"))]
            {
                argsort_radix_u32(data, config.descending)
            }
        }
    }
}

/// Argsort for u64 with automatic algorithm selection
pub fn argsort_auto_u64(data: &[u64], config: &ArgsortConfig) -> Vec<usize> {
    match config.algorithm {
        SortAlgorithm::Comparison => argsort_with_stability(data, config.descending, config.stable),
        SortAlgorithm::Radix => argsort_radix_u64(data, config.descending),
        #[cfg(feature = "simd_sort")]
        SortAlgorithm::Simd => simd_argsort::argsort_simd_u64(data, config.descending),
        SortAlgorithm::Auto => {
            if config.stable {
                return argsort_with_stability(data, config.descending, true);
            }
            #[cfg(feature = "simd_sort")]
            {
                simd_argsort::argsort_simd_u64(data, config.descending)
            }
            #[cfg(not(feature = "simd_sort"))]
            {
                argsort_radix_u64(data, config.descending)
            }
        }
    }
}

// Pre-sorted detection
//
// O(n) scan to check whether a slice is already sorted, avoiding the O(n log n) sort
// when data arrives pre-ordered. The integer variants use explicit std::simd to compare
// N adjacent pairs per iteration. Float variants use total_cmp_f for NaN-consistent
// ordering which prevents SIMD vectorisation, so they use scalar loops with early exit.

// SIMD sortedness check for integer types.
// Loads `data[i..i+N]` and `data[i+1..i+1+N]` into SIMD vectors, then checks
// `simd_le` (ascending) or `simd_ge` (descending) across all lanes at once.
#[cfg(feature = "simd")]
macro_rules! impl_is_sorted_simd {
    ($fn_asc:ident, $fn_desc:ident, $fn_dir:ident, $ty:ty, $lanes:expr, $mask_elem:ty) => {
        /// SIMD-accelerated ascending sortedness check.
        ///
        /// Compares N adjacent pairs per iteration by loading overlapping
        /// SIMD vectors from `data[i..i+N]` and `data[i+1..i+1+N]`.
        #[inline]
        pub fn $fn_asc(data: &[$ty]) -> bool {
            let len = data.len();
            if len <= 1 {
                return true;
            }
            const N: usize = $lanes;
            let pairs = len - 1;
            let mut i = 0;

            // Process N pairs at a time
            while i + N < len {
                let a = Simd::<$ty, N>::from_slice(&data[i..i + N]);
                let b = Simd::<$ty, N>::from_slice(&data[i + 1..i + 1 + N]);
                let ok: std::simd::Mask<$mask_elem, N> = a.simd_le(b);
                if !ok.all() {
                    return false;
                }
                i += N;
            }
            // Scalar tail
            while i < pairs {
                if data[i] > data[i + 1] {
                    return false;
                }
                i += 1;
            }
            true
        }

        /// SIMD-accelerated descending sortedness check.
        #[inline]
        pub fn $fn_desc(data: &[$ty]) -> bool {
            let len = data.len();
            if len <= 1 {
                return true;
            }
            const N: usize = $lanes;
            let pairs = len - 1;
            let mut i = 0;

            while i + N < len {
                let a = Simd::<$ty, N>::from_slice(&data[i..i + N]);
                let b = Simd::<$ty, N>::from_slice(&data[i + 1..i + 1 + N]);
                let ok: std::simd::Mask<$mask_elem, N> = a.simd_ge(b);
                if !ok.all() {
                    return false;
                }
                i += N;
            }
            while i < pairs {
                if data[i] < data[i + 1] {
                    return false;
                }
                i += 1;
            }
            true
        }

        /// Check if a slice is sorted in the given direction using SIMD.
        #[inline]
        pub fn $fn_dir(data: &[$ty], descending: bool) -> bool {
            if descending {
                $fn_desc(data)
            } else {
                $fn_asc(data)
            }
        }
    };
}

#[cfg(feature = "simd")]
impl_is_sorted_simd!(is_sorted_i32_asc, is_sorted_i32_desc, is_sorted_i32, i32, W32, i32);
#[cfg(feature = "simd")]
impl_is_sorted_simd!(is_sorted_i64_asc, is_sorted_i64_desc, is_sorted_i64, i64, W64, i64);
#[cfg(feature = "simd")]
impl_is_sorted_simd!(is_sorted_u32_asc, is_sorted_u32_desc, is_sorted_u32, u32, W32, i32);
#[cfg(feature = "simd")]
impl_is_sorted_simd!(is_sorted_u64_asc, is_sorted_u64_desc, is_sorted_u64, u64, W64, i64);

// Scalar fallbacks when SIMD is disabled
#[cfg(not(feature = "simd"))]
macro_rules! impl_is_sorted_scalar {
    ($fn_asc:ident, $fn_desc:ident, $fn_dir:ident, $ty:ty) => {
        /// Scalar ascending sortedness check.
        #[inline]
        pub fn $fn_asc(data: &[$ty]) -> bool {
            data.windows(2).all(|w| w[0] <= w[1])
        }

        /// Scalar descending sortedness check.
        #[inline]
        pub fn $fn_desc(data: &[$ty]) -> bool {
            data.windows(2).all(|w| w[0] >= w[1])
        }

        /// Check if a slice is sorted in the given direction.
        #[inline]
        pub fn $fn_dir(data: &[$ty], descending: bool) -> bool {
            if descending { $fn_desc(data) } else { $fn_asc(data) }
        }
    };
}

#[cfg(not(feature = "simd"))]
impl_is_sorted_scalar!(is_sorted_i32_asc, is_sorted_i32_desc, is_sorted_i32, i32);
#[cfg(not(feature = "simd"))]
impl_is_sorted_scalar!(is_sorted_i64_asc, is_sorted_i64_desc, is_sorted_i64, i64);
#[cfg(not(feature = "simd"))]
impl_is_sorted_scalar!(is_sorted_u32_asc, is_sorted_u32_desc, is_sorted_u32, u32);
#[cfg(not(feature = "simd"))]
impl_is_sorted_scalar!(is_sorted_u64_asc, is_sorted_u64_desc, is_sorted_u64, u64);

/// Generic ascending sortedness check for any Ord type.
///
/// Scalar-only. Use the typed variants (is_sorted_i32, etc.) for SIMD acceleration.
#[inline]
pub fn is_sorted_int_asc<T: Ord>(data: &[T]) -> bool {
    if data.len() <= 1 {
        return true;
    }
    data.windows(2).all(|w| w[0] <= w[1])
}

/// Generic descending sortedness check for any Ord type.
#[inline]
pub fn is_sorted_int_desc<T: Ord>(data: &[T]) -> bool {
    if data.len() <= 1 {
        return true;
    }
    data.windows(2).all(|w| w[0] >= w[1])
}

/// Generic sortedness check for any Ord type in the given direction.
#[inline]
pub fn is_sorted_int<T: Ord>(data: &[T], descending: bool) -> bool {
    if descending {
        is_sorted_int_desc(data)
    } else {
        is_sorted_int_asc(data)
    }
}

// SIMD float is_sorted using the same total-ordering key transform as total_cmp_f.
//
// The trick: reinterpret float bits as unsigned integer, then apply a branchless
// key transform so that unsigned comparison gives IEEE 754 total ordering:
//   - negative floats (sign bit set): invert all bits
//   - positive floats (sign bit clear): flip only the sign bit
// This maps the float number line onto the unsigned integer number line,
// so simd_le on the keys == total_cmp_f <= on the floats, with NaN after +inf.

#[cfg(feature = "simd")]
macro_rules! impl_is_sorted_float_simd {
    ($fn_asc:ident, $fn_desc:ident, $fn_dir:ident,
     $fty:ty, $uty:ty, $ity:ty, $lanes:expr, $sign_mask:expr) => {
        /// SIMD-accelerated ascending sortedness check for floats.
        ///
        /// Uses the same total-ordering key transform as `total_cmp_f`:
        /// reinterpret as unsigned, conditionally negate based on sign bit,
        /// then compare N adjacent pairs at once with `simd_le`.
        #[inline]
        pub fn $fn_asc(data: &[$fty]) -> bool {
            let len = data.len();
            if len <= 1 {
                return true;
            }
            const N: usize = $lanes;
            let sign_mask = Simd::<$uty, N>::splat($sign_mask);
            let pairs = len - 1;
            let mut i = 0;

            while i + N < len {
                let a = Simd::<$fty, N>::from_slice(&data[i..i + N]);
                let b = Simd::<$fty, N>::from_slice(&data[i + 1..i + 1 + N]);

                // Transform to sort keys via the total_cmp_f bit trick
                let bits_a: Simd<$uty, N> = a.to_bits();
                let bits_b: Simd<$uty, N> = b.to_bits();

                // Sign check: cast to signed and compare < 0
                let neg_a: Mask<$ity, N> = bits_a.cast::<$ity>().simd_lt(Simd::splat(0));
                let neg_b: Mask<$ity, N> = bits_b.cast::<$ity>().simd_lt(Simd::splat(0));

                // Branchless key: negative -> !bits, positive -> bits ^ sign_mask
                let key_a: Simd<$uty, N> = neg_a.select(!bits_a, bits_a ^ sign_mask);
                let key_b: Simd<$uty, N> = neg_b.select(!bits_b, bits_b ^ sign_mask);

                if !key_a.simd_le(key_b).all() {
                    return false;
                }
                i += N;
            }
            // Scalar tail using the existing total_cmp_f
            while i < pairs {
                if total_cmp_f(&data[i], &data[i + 1]) == Ordering::Greater {
                    return false;
                }
                i += 1;
            }
            true
        }

        /// SIMD-accelerated descending sortedness check for floats.
        #[inline]
        pub fn $fn_desc(data: &[$fty]) -> bool {
            let len = data.len();
            if len <= 1 {
                return true;
            }
            const N: usize = $lanes;
            let sign_mask = Simd::<$uty, N>::splat($sign_mask);
            let pairs = len - 1;
            let mut i = 0;

            while i + N < len {
                let a = Simd::<$fty, N>::from_slice(&data[i..i + N]);
                let b = Simd::<$fty, N>::from_slice(&data[i + 1..i + 1 + N]);

                let bits_a: Simd<$uty, N> = a.to_bits();
                let bits_b: Simd<$uty, N> = b.to_bits();

                let neg_a: Mask<$ity, N> = bits_a.cast::<$ity>().simd_lt(Simd::splat(0));
                let neg_b: Mask<$ity, N> = bits_b.cast::<$ity>().simd_lt(Simd::splat(0));

                let key_a: Simd<$uty, N> = neg_a.select(!bits_a, bits_a ^ sign_mask);
                let key_b: Simd<$uty, N> = neg_b.select(!bits_b, bits_b ^ sign_mask);

                if !key_a.simd_ge(key_b).all() {
                    return false;
                }
                i += N;
            }
            while i < pairs {
                if total_cmp_f(&data[i], &data[i + 1]) == Ordering::Less {
                    return false;
                }
                i += 1;
            }
            true
        }

        /// Check if a float slice is sorted in the given direction using SIMD.
        #[inline]
        pub fn $fn_dir(data: &[$fty], descending: bool) -> bool {
            if descending { $fn_desc(data) } else { $fn_asc(data) }
        }
    };
}

#[cfg(feature = "simd")]
impl_is_sorted_float_simd!(
    is_sorted_f32_asc, is_sorted_f32_desc, is_sorted_f32,
    f32, u32, i32, W32, 0x8000_0000u32
);
#[cfg(feature = "simd")]
impl_is_sorted_float_simd!(
    is_sorted_f64_asc, is_sorted_f64_desc, is_sorted_f64,
    f64, u64, i64, W64, 0x8000_0000_0000_0000u64
);

// Scalar fallbacks when SIMD is disabled
#[cfg(not(feature = "simd"))]
macro_rules! impl_is_sorted_float_scalar {
    ($fn_asc:ident, $fn_desc:ident, $fn_dir:ident, $fty:ty) => {
        /// Scalar ascending sortedness check for floats using total_cmp_f.
        #[inline]
        pub fn $fn_asc(data: &[$fty]) -> bool {
            data.windows(2)
                .all(|w| total_cmp_f(&w[0], &w[1]) != Ordering::Greater)
        }

        /// Scalar descending sortedness check for floats using total_cmp_f.
        #[inline]
        pub fn $fn_desc(data: &[$fty]) -> bool {
            data.windows(2)
                .all(|w| total_cmp_f(&w[0], &w[1]) != Ordering::Less)
        }

        /// Check if a float slice is sorted in the given direction.
        #[inline]
        pub fn $fn_dir(data: &[$fty], descending: bool) -> bool {
            if descending { $fn_desc(data) } else { $fn_asc(data) }
        }
    };
}

#[cfg(not(feature = "simd"))]
impl_is_sorted_float_scalar!(is_sorted_f32_asc, is_sorted_f32_desc, is_sorted_f32, f32);
#[cfg(not(feature = "simd"))]
impl_is_sorted_float_scalar!(is_sorted_f64_asc, is_sorted_f64_desc, is_sorted_f64, f64);

/// Generic ascending sortedness check for any Float type using total ordering.
///
/// Scalar-only. Use the typed variants (is_sorted_f32, is_sorted_f64) for SIMD acceleration.
#[inline]
pub fn is_sorted_float_asc<T: Float>(data: &[T]) -> bool {
    if data.len() <= 1 {
        return true;
    }
    data.windows(2)
        .all(|w| total_cmp_f(&w[0], &w[1]) != Ordering::Greater)
}

/// Generic descending sortedness check for any Float type using total ordering.
#[inline]
pub fn is_sorted_float_desc<T: Float>(data: &[T]) -> bool {
    if data.len() <= 1 {
        return true;
    }
    data.windows(2)
        .all(|w| total_cmp_f(&w[0], &w[1]) != Ordering::Less)
}

/// Generic sortedness check for any Float type in the given direction.
#[inline]
pub fn is_sorted_float<T: Float>(data: &[T], descending: bool) -> bool {
    if descending {
        is_sorted_float_desc(data)
    } else {
        is_sorted_float_asc(data)
    }
}

/// Check if a string slice is sorted in the given direction.
#[inline]
pub fn is_sorted_str(data: &[&str], descending: bool) -> bool {
    if data.len() <= 1 {
        return true;
    }
    if descending {
        data.windows(2).all(|w| w[0] >= w[1])
    } else {
        data.windows(2).all(|w| w[0] <= w[1])
    }
}

/// Check if a StringArray (offset-based) is sorted in the given direction.
///
/// Generic over the offset type T to support both String32 (u32 offsets)
/// and String64 (u64 offsets) without conversion.
pub fn is_sorted_string_array<T>(offsets: &[T], values: &[u8], descending: bool) -> bool
where
    T: Copy + ToPrimitive,
{
    let n = if offsets.is_empty() {
        0
    } else {
        offsets.len() - 1
    };
    if n <= 1 {
        return true;
    }

    for i in 0..n - 1 {
        let a_start = offsets[i].to_usize().unwrap();
        let a_end = offsets[i + 1].to_usize().unwrap();
        let b_end = offsets[i + 2].to_usize().unwrap();
        let a = &values[a_start..a_end];
        let b = &values[a_end..b_end];
        // Byte comparison is equivalent to UTF-8 lexicographic comparison
        let cmp = a.cmp(b);
        if (descending && cmp == Ordering::Less) || (!descending && cmp == Ordering::Greater) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use minarrow::vec64;

    use super::*;

    #[test]
    fn test_total_cmp_f32_ordering_basic() {
        let a = 1.0f32;
        let b = 2.0f32;
        assert_eq!(total_cmp_f(&a, &b), std::cmp::Ordering::Less);
        assert_eq!(total_cmp_f(&b, &a), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&a, &a), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_f64_ordering_basic() {
        let a = 1.0f64;
        let b = 2.0f64;
        assert_eq!(total_cmp_f(&a, &b), std::cmp::Ordering::Less);
        assert_eq!(total_cmp_f(&b, &a), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&a, &a), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_zero_and_negzero() {
        let pz = 0.0f32;
        let nz = -0.0f32;
        // -0.0 should not equal 0.0 in bitwise comparison
        assert_ne!(f32::to_bits(pz), f32::to_bits(nz));
        assert_ne!(total_cmp_f(&pz, &nz), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_nan_ordering_f32() {
        let nan = f32::NAN;
        let pos_inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let one = 1.0f32;

        // NaN is greater than everything in this ordering
        assert_eq!(total_cmp_f(&nan, &pos_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &neg_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &one), std::cmp::Ordering::Greater);
        // Self-equality
        assert_eq!(total_cmp_f(&nan, &nan), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_total_cmp_nan_ordering_f64() {
        let nan = f64::NAN;
        let pos_inf = f64::INFINITY;
        let neg_inf = f64::NEG_INFINITY;
        let one = 1.0f64;

        assert_eq!(total_cmp_f(&nan, &pos_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &neg_inf), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &one), std::cmp::Ordering::Greater);
        assert_eq!(total_cmp_f(&nan, &nan), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_sort_float_f32_all_edge_cases() {
        let mut v = vec64![
            3.0f32,
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            f32::NAN,
            2.0,
            -2.0,
        ];
        sort_float(&mut v);
        // Sorted by bit-pattern, not by value
        // -2.0 < -1.0 < -0.0 < 0.0 < 1.0 < 2.0 < 3.0 < INF < NAN
        assert_eq!(v[0], f32::NEG_INFINITY);
        assert_eq!(v[1], -2.0);
        assert_eq!(v[2], -1.0);
        assert_eq!(v[3], -0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 1.0);
        assert_eq!(v[6], 2.0);
        assert_eq!(v[7], 3.0);
        assert_eq!(v[8], f32::INFINITY);
        assert!(v[9].is_nan());
    }

    #[test]
    fn test_sort_float_f64_all_edge_cases() {
        let mut v = vec64![
            3.0f64,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            -1.0,
            f64::NAN,
            2.0,
            -2.0,
        ];
        sort_float(&mut v);
        assert_eq!(v[0], f64::NEG_INFINITY);
        assert_eq!(v[1], -2.0);
        assert_eq!(v[2], -1.0);
        assert_eq!(v[3], -0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 1.0);
        assert_eq!(v[6], 2.0);
        assert_eq!(v[7], 3.0);
        assert_eq!(v[8], f64::INFINITY);
        assert!(v[9].is_nan());
    }

    #[test]
    fn test_sorted_float_immutability_and_return_type() {
        let v = vec64![1.0f32, 2.0, 3.0];
        let out = sorted_float(&v);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0]);
        // Ensure original is unchanged
        assert_eq!(*v, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sorted_float_correct_for_f64() {
        let v = vec64![3.0f64, 2.0, 1.0];
        let out = sorted_float(&v);
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sort_float_empty_and_single() {
        let mut v: [f32; 0] = [];
        sort_float(&mut v);
        let mut v2 = [42.0f32];
        sort_float(&mut v2);
        assert_eq!(v2, [42.0]);
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use minarrow::{Vec64, vec64};

        #[test]
        fn test_sort_int_empty_and_single() {
            let mut arr: [i32; 0] = [];
            sort_int(&mut arr);
            assert_eq!(arr, [] as [i32; 0]);
            let mut arr2 = vec64![42];
            sort_int(&mut arr2);
            assert_eq!(*arr2, [42]);
        }

        #[test]
        fn test_sort_int_order() {
            let mut arr = vec64![4, 2, 7, 1, 1, 6, 0, -5];
            sort_int(&mut arr);
            assert_eq!(*arr, [-5, 0, 1, 1, 2, 4, 6, 7]);
        }

        #[test]
        fn test_sorted_int_immutability_and_output() {
            let arr = vec64![5, 3, 7, 1, 2];
            let sorted = sorted_int(&arr);
            assert_eq!(sorted.as_slice(), &[1, 2, 3, 5, 7]);
            // ensure input not changed
            assert_eq!(*arr, [5, 3, 7, 1, 2]);
        }

        #[test]
        fn test_sort_str_basic() {
            let mut arr = vec64!["z", "b", "a", "d"];
            sort_str(&mut arr);
            assert_eq!(*arr, ["a", "b", "d", "z"]);
        }

        #[test]
        fn test_sorted_str_and_non_ascii() {
            let arr = vec64!["z", "ä", "b", "a", "d"];
            let sorted = sorted_str(&arr);
            assert_eq!(sorted.as_slice(), &["a", "b", "d", "z", "ä"]); // note: utf8, ä > z in Rust
            // input is untouched
            assert_eq!(*arr, ["z", "ä", "b", "a", "d"]);
        }

        #[test]
        fn test_sort_string_array_basic() {
            let offsets = vec![0, 1, 3, 5, 5, 6]; // ["a", "bc", "de", "", "f"]
            let values = "abcde".to_string() + "f";
            let (new_offsets, new_values) = sort_string_array(&offsets, &values);
            // Sorted order: "", "a", "bc", "de", "f"
            let slices: Vec<_> = new_offsets
                .windows(2)
                .map(|w| &new_values[w[0]..w[1]])
                .collect();
            assert_eq!(slices, &["", "a", "bc", "de", "f"]);
        }

        #[test]
        fn test_sort_categorical_lexical_basic_and_nulls() {
            // Categories: 0 = "apple", 1 = "banana", 2 = "pear"
            let unique = Vec64::from_slice_clone(&[
                "apple".to_string(),
                "banana".to_string(),
                "pear".to_string(),
            ]);
            let data = Vec64::from_slice(&[2u8, 0, 1, 1, 2, 0, 1]);
            let mask = Bitmask::from_bools(&[true, false, true, true, true, false, true]);
            let cat = CategoricalArray {
                data: data.into(),
                unique_values: unique,
                null_mask: Some(mask.clone()),
            };
            let (sorted_data, sorted_mask) = sort_categorical_lexical(&cat);

            // Nulls first
            let mask_out = sorted_mask.unwrap();
            let null_pos: Vec<_> = (0..mask_out.len()).filter(|&i| !mask_out.get(i)).collect();
            assert_eq!(null_pos, &[0, 1]);

            // Remaining valid values
            let sorted_as_str: Vec<_> = sorted_data
                .iter()
                .map(|&k| &cat.unique_values[k.to_usize()][..])
                .collect();
            let vals = &sorted_as_str[null_pos.len()..];
            assert_eq!(vals, &["banana", "banana", "banana", "pear", "pear"]);
        }

        #[test]
        fn test_sort_categorical_all_nulls_and_no_nulls() {
            // All null
            let unique = Vec64::from_slice_clone(&["x".to_string()]);
            let data = Vec64::from_slice(&[0u8, 0, 0]);
            let mask = Bitmask::from_bools(&[false, false, false]);
            let cat = CategoricalArray {
                data: data.clone().into(),
                unique_values: unique.clone(),
                null_mask: Some(mask.clone()),
            };
            let (_, sorted_mask) = sort_categorical_lexical(&cat);
            assert_eq!(
                unpack_bitmask(&sorted_mask.unwrap()),
                vec64![false, false, false]
            );
            // No nulls
            let mask2 = Bitmask::from_bools(&[true, true, true]);
            let cat2 = CategoricalArray {
                data: data.clone().into(),
                unique_values: unique,
                null_mask: Some(mask2.clone()),
            };
            let (_, sorted_mask2) = sort_categorical_lexical(&cat2);
            assert_eq!(
                unpack_bitmask(&sorted_mask2.unwrap()),
                vec64![true, true, true]
            );
        }
        #[test]
        fn test_sort_boolean_array_with_nulls() {
            let mut arr = BooleanArray {
                data: pack_bitmask(&[false, true, false, true, true, false]),
                null_mask: Some(pack_bitmask(&[true, false, true, true, false, true])),
                len: 6,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr);
            // Nulls first (mask false)
            let bits = unpack_bitmask(&arr.data);
            let nulls = unpack_bitmask(arr.null_mask.as_ref().unwrap());
            assert_eq!(nulls, vec64![false, false, true, true, true, true]);
            // Sorted data for valid (true): all false first, then true
            assert_eq!(&bits[2..], [false, false, false, true]);
        }

        #[test]
        fn test_sort_boolean_array_all_true_and_all_false() {
            let mut arr = BooleanArray {
                data: pack_bitmask(&[true, true, true]),
                null_mask: None,
                len: 3,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr);
            assert_eq!(unpack_bitmask(&arr.data), vec64![true, true, true]);

            let mut arr2 = BooleanArray {
                data: pack_bitmask(&[false, false, false]),
                null_mask: None,
                len: 3,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr2);
            assert_eq!(unpack_bitmask(&arr2.data), vec64![false, false, false]);
        }

        #[test]
        fn test_sort_boolean_array_all_nulls_and_none() {
            let mut arr = BooleanArray {
                data: pack_bitmask(&[true, false, true]),
                null_mask: Some(pack_bitmask(&[false, false, false])),
                len: 3,
                _phantom: std::marker::PhantomData,
            };
            sort_boolean_array(&mut arr);
            assert_eq!(
                unpack_bitmask(arr.null_mask.as_ref().unwrap()),
                vec64![false, false, false]
            );
        }

        #[test]
        fn test_sort_slice_with_mask_basic() {
            let data = vec64![3, 1, 2, 1];
            let mask = pack_bitmask(&[true, false, true, true]);
            let (sorted, mask_out) = sort_slice_with_mask(&data, Some(&mask));
            assert_eq!(sorted.as_slice(), &[1, 1, 2, 3]);
            assert_eq!(
                unpack_bitmask(&mask_out.unwrap()),
                vec64![false, true, true, true]
            );
        }

        #[test]
        fn test_sort_slice_with_mask_no_mask() {
            let data = vec64![3, 2, 1, 1, 0];
            let (sorted, mask_out) = sort_slice_with_mask(&data, None);
            assert_eq!(sorted.as_slice(), &[0, 1, 1, 2, 3]);
            assert!(mask_out.is_none());
        }

        #[test]
        fn test_sort_slice_with_mask_all_true_and_all_false() {
            let data = vec64![3, 2, 1, 0];
            let mask_true = pack_bitmask(&[true; 4]);
            let mask_false = pack_bitmask(&[false; 4]);
            let (_, mask_true_out) = sort_slice_with_mask(&data, Some(&mask_true));
            let (_, mask_false_out) = sort_slice_with_mask(&data, Some(&mask_false));
            assert_eq!(
                unpack_bitmask(&mask_true_out.unwrap()),
                vec64![true, true, true, true]
            );
            assert_eq!(
                unpack_bitmask(&mask_false_out.unwrap()),
                vec64![false, false, false, false]
            );
        }

        #[test]
        fn test_sort_int_with_duplicates_and_negatives() {
            let mut arr = vec64![-10, -1, 5, 0, 5, -10];
            sort_int(&mut arr);
            assert_eq!(*arr, [-10, -10, -1, 0, 5, 5]);
        }

        #[test]
        fn test_sort_str_empty_and_duplicate() {
            let mut arr = vec64!["", "a", "b", "a", ""];
            sort_str(&mut arr);
            assert_eq!(*arr, ["", "", "a", "a", "b"]);
        }

        // ====================================================================
        // Argsort Tests
        // ====================================================================

        #[test]
        fn test_argsort_basic() {
            let data = [5i32, 2, 8, 1, 9];
            let indices = argsort(&data, false);
            assert_eq!(indices, vec![3, 1, 0, 2, 4]); // 1, 2, 5, 8, 9

            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![1, 2, 5, 8, 9]);
        }

        #[test]
        fn test_argsort_descending() {
            let data = [5i32, 2, 8, 1, 9];
            let indices = argsort(&data, true);
            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![9, 8, 5, 2, 1]);
        }

        #[test]
        fn test_argsort_empty() {
            let data: [i32; 0] = [];
            let indices = argsort(&data, false);
            assert!(indices.is_empty());
        }

        #[test]
        fn test_argsort_float_basic() {
            let data = [3.0f64, 1.0, 4.0, 1.5, 2.0];
            let indices = argsort_float(&data, false);
            let sorted: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_argsort_float_with_nan() {
            let data = [3.0f64, f64::NAN, 1.0, f64::NEG_INFINITY];
            let indices = argsort_float(&data, false);
            let sorted: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
            // NaN should sort last (greatest)
            assert_eq!(sorted[0], f64::NEG_INFINITY);
            assert_eq!(sorted[1], 1.0);
            assert_eq!(sorted[2], 3.0);
            assert!(sorted[3].is_nan());
        }

        #[test]
        fn test_argsort_str_basic() {
            let data = ["cherry", "apple", "banana"];
            let indices = argsort_str(&data, false);
            let sorted: Vec<&str> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec!["apple", "banana", "cherry"]);
        }

        #[test]
        fn test_argsort_str_descending() {
            let data = ["cherry", "apple", "banana"];
            let indices = argsort_str(&data, true);
            let sorted: Vec<&str> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec!["cherry", "banana", "apple"]);
        }

        #[test]
        fn test_argsort_string_array_basic() {
            // Simulating StringArray: "apple", "cherry", "banana"
            let values = "applecherrybanana";
            let offsets = [0usize, 5, 11, 17];
            let indices = argsort_string_array(&offsets, values, false);
            // Expected order: apple(0), banana(2), cherry(1)
            assert_eq!(indices, vec![0, 2, 1]);
        }

        #[test]
        fn test_argsort_radix_u32_basic() {
            let data = vec![5u32, 2, 8, 1, 9, 3, 7, 4, 6, 0];
            let indices = argsort_radix_u32(&data, false);
            let sorted: Vec<u32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        }

        #[test]
        fn test_argsort_radix_u32_descending() {
            let data = vec![5u32, 2, 8, 1, 9, 3, 7, 4, 6, 0];
            let indices = argsort_radix_u32(&data, true);
            let sorted: Vec<u32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        }

        #[test]
        fn test_argsort_radix_i32_with_negatives() {
            let data = vec![5i32, -2, 8, -1, 9, 3, -7, 4, 6, 0];
            let indices = argsort_radix_i32(&data, false);
            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![-7, -2, -1, 0, 3, 4, 5, 6, 8, 9]);
        }

        #[test]
        fn test_argsort_radix_u64_basic() {
            let data = vec![100u64, 50, 200, 25];
            let indices = argsort_radix_u64(&data, false);
            let sorted: Vec<u64> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![25, 50, 100, 200]);
        }

        #[test]
        fn test_argsort_radix_i64_with_negatives() {
            let data = vec![5i64, -2, 8, -1, 0];
            let indices = argsort_radix_i64(&data, false);
            let sorted: Vec<i64> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![-2, -1, 0, 5, 8]);
        }

        #[test]
        fn test_argsort_boolean_basic() {
            use minarrow::Bitmask;
            // [true, false, true, false]
            let mut data = Bitmask::new_set_all(4, false);
            data.set(0, true);
            data.set(2, true);

            let indices = argsort_boolean(&data, None, false);
            // false(1), false(3), true(0), true(2)
            let sorted: Vec<bool> = indices.iter().map(|&i| data.get(i)).collect();
            assert_eq!(sorted, vec![false, false, true, true]);
        }

        #[test]
        fn test_argsort_boolean_descending() {
            use minarrow::Bitmask;
            let mut data = Bitmask::new_set_all(4, false);
            data.set(0, true);
            data.set(2, true);

            let indices = argsort_boolean(&data, None, true);
            let sorted: Vec<bool> = indices.iter().map(|&i| data.get(i)).collect();
            assert_eq!(sorted, vec![true, true, false, false]);
        }

        #[test]
        fn test_argsort_boolean_with_nulls() {
            use minarrow::Bitmask;
            // data: [true, false, true, false]
            // mask: [valid, null, valid, valid] (1=valid, 0=null)
            let mut data = Bitmask::new_set_all(4, false);
            data.set(0, true);
            data.set(2, true);

            let mut null_mask = Bitmask::new_set_all(4, true);
            null_mask.set(1, false); // index 1 is null

            let indices = argsort_boolean(&data, Some(&null_mask), false);
            // nulls first: null(1), then false(3), true(0), true(2)
            assert_eq!(indices[0], 1); // null first
        }

        #[test]
        fn test_argsort_auto_i32_basic() {
            let data = [5i32, 2, 8, 1, 9];
            let config = ArgsortConfig::default();
            let indices = argsort_auto_i32(&data, &config);
            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![1, 2, 5, 8, 9]);
        }

        #[test]
        fn test_argsort_auto_with_comparison() {
            let data = [5i32, 2, 8, 1, 9];
            let config = ArgsortConfig::new().algorithm(SortAlgorithm::Comparison);
            let indices = argsort_auto_i32(&data, &config);
            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![1, 2, 5, 8, 9]);
        }

        #[test]
        fn test_argsort_auto_descending() {
            let data = [5i32, 2, 8, 1, 9];
            let config = ArgsortConfig::new().descending(true);
            let indices = argsort_auto_i32(&data, &config);
            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![9, 8, 5, 2, 1]);
        }

        // ====================================================================
        // Algorithm Consistency Tests
        // These tests verify that all sort algorithm variants produce identical
        // sorted output (though indices may differ for equal elements).
        // ====================================================================

        /// Helper: Apply indices to data and return sorted values
        fn apply_indices<T: Copy>(data: &[T], indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&i| data[i]).collect()
        }

        #[test]
        fn test_i32_sorts_produce_same_results() {
            // Test data with duplicates, negatives, and edge cases
            let data = vec![
                5i32,
                -2,
                8,
                -1,
                9,
                3,
                -7,
                4,
                6,
                0,
                i32::MAX,
                i32::MIN,
                0,
                -100,
                100,
                1,
                1,
                1,
                -1,
                -1, // duplicates
            ];

            // Get indices from each algorithm
            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_i32(&data, false);
            let auto_comparison = argsort_auto_i32(
                &data,
                &ArgsortConfig::new().algorithm(SortAlgorithm::Comparison),
            );
            let auto_radix =
                argsort_auto_i32(&data, &ArgsortConfig::new().algorithm(SortAlgorithm::Radix));
            let auto_default = argsort_auto_i32(&data, &ArgsortConfig::default());

            // Convert to sorted values
            let sorted_comparison: Vec<i32> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<i32> = apply_indices(&data, &radix_asc);
            let sorted_auto_comparison: Vec<i32> = apply_indices(&data, &auto_comparison);
            let sorted_auto_radix: Vec<i32> = apply_indices(&data, &auto_radix);
            let sorted_auto_default: Vec<i32> = apply_indices(&data, &auto_default);

            // All should produce the same sorted order
            assert_eq!(
                sorted_comparison, sorted_radix,
                "comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_comparison,
                "comparison vs auto_comparison mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_radix,
                "comparison vs auto_radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_default,
                "comparison vs auto_default mismatch"
            );

            // Verify it's actually sorted
            for i in 1..sorted_comparison.len() {
                assert!(
                    sorted_comparison[i - 1] <= sorted_comparison[i],
                    "not sorted at index {}",
                    i
                );
            }

            // Test descending too
            let comparison_desc = argsort(&data, true);
            let radix_desc = argsort_radix_i32(&data, true);

            let sorted_comparison_desc: Vec<i32> = apply_indices(&data, &comparison_desc);
            let sorted_radix_desc: Vec<i32> = apply_indices(&data, &radix_desc);

            assert_eq!(
                sorted_comparison_desc, sorted_radix_desc,
                "descending comparison vs radix mismatch"
            );

            // Verify descending order
            for i in 1..sorted_comparison_desc.len() {
                assert!(
                    sorted_comparison_desc[i - 1] >= sorted_comparison_desc[i],
                    "not sorted descending at index {}",
                    i
                );
            }
        }

        #[test]
        fn test_i64_sorts_produce_same_results() {
            let data = vec![
                5i64,
                -2,
                8,
                -1,
                9,
                3,
                -7,
                4,
                6,
                0,
                i64::MAX,
                i64::MIN,
                0,
                -100,
                100,
                i64::MAX - 1,
                i64::MIN + 1,
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_i64(&data, false);
            let auto_comparison = argsort_auto_i64(
                &data,
                &ArgsortConfig::new().algorithm(SortAlgorithm::Comparison),
            );
            let auto_radix =
                argsort_auto_i64(&data, &ArgsortConfig::new().algorithm(SortAlgorithm::Radix));

            let sorted_comparison: Vec<i64> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<i64> = apply_indices(&data, &radix_asc);
            let sorted_auto_comparison: Vec<i64> = apply_indices(&data, &auto_comparison);
            let sorted_auto_radix: Vec<i64> = apply_indices(&data, &auto_radix);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "i64: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_comparison,
                "i64: comparison vs auto_comparison mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_radix,
                "i64: comparison vs auto_radix mismatch"
            );

            // Verify sorted
            for i in 1..sorted_comparison.len() {
                assert!(
                    sorted_comparison[i - 1] <= sorted_comparison[i],
                    "i64: not sorted at index {}",
                    i
                );
            }
        }

        #[test]
        fn test_u32_sorts_produce_same_results() {
            let data = vec![
                5u32,
                2,
                8,
                1,
                9,
                3,
                7,
                4,
                6,
                0,
                u32::MAX,
                0,
                100,
                u32::MAX - 1,
                1,
                1,
                1, // duplicates
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_u32(&data, false);
            let auto_comparison = argsort_auto_u32(
                &data,
                &ArgsortConfig::new().algorithm(SortAlgorithm::Comparison),
            );
            let auto_radix =
                argsort_auto_u32(&data, &ArgsortConfig::new().algorithm(SortAlgorithm::Radix));

            let sorted_comparison: Vec<u32> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<u32> = apply_indices(&data, &radix_asc);
            let sorted_auto_comparison: Vec<u32> = apply_indices(&data, &auto_comparison);
            let sorted_auto_radix: Vec<u32> = apply_indices(&data, &auto_radix);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "u32: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_comparison,
                "u32: comparison vs auto_comparison mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_radix,
                "u32: comparison vs auto_radix mismatch"
            );

            // Verify sorted
            for i in 1..sorted_comparison.len() {
                assert!(
                    sorted_comparison[i - 1] <= sorted_comparison[i],
                    "u32: not sorted at index {}",
                    i
                );
            }
        }

        #[test]
        fn test_u64_sorts_produce_same_results() {
            let data = vec![
                5u64,
                2,
                8,
                1,
                9,
                3,
                7,
                4,
                6,
                0,
                u64::MAX,
                0,
                100,
                u64::MAX - 1,
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_u64(&data, false);
            let auto_comparison = argsort_auto_u64(
                &data,
                &ArgsortConfig::new().algorithm(SortAlgorithm::Comparison),
            );
            let auto_radix =
                argsort_auto_u64(&data, &ArgsortConfig::new().algorithm(SortAlgorithm::Radix));

            let sorted_comparison: Vec<u64> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<u64> = apply_indices(&data, &radix_asc);
            let sorted_auto_comparison: Vec<u64> = apply_indices(&data, &auto_comparison);
            let sorted_auto_radix: Vec<u64> = apply_indices(&data, &auto_radix);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "u64: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_comparison,
                "u64: comparison vs auto_comparison mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_radix,
                "u64: comparison vs auto_radix mismatch"
            );

            // Verify sorted
            for i in 1..sorted_comparison.len() {
                assert!(
                    sorted_comparison[i - 1] <= sorted_comparison[i],
                    "u64: not sorted at index {}",
                    i
                );
            }
        }

        #[cfg(feature = "simd_sort")]
        #[test]
        fn test_i32_sorts_produce_same_results_with_simd() {
            use super::simd_argsort;

            let data = vec![
                5i32,
                -2,
                8,
                -1,
                9,
                3,
                -7,
                4,
                6,
                0,
                i32::MAX,
                i32::MIN,
                0,
                -100,
                100,
                1,
                1,
                1,
                -1,
                -1,
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_i32(&data, false);
            let simd_asc = simd_argsort::argsort_simd_i32(&data, false);
            let auto_simd =
                argsort_auto_i32(&data, &ArgsortConfig::new().algorithm(SortAlgorithm::Simd));

            let sorted_comparison: Vec<i32> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<i32> = apply_indices(&data, &radix_asc);
            let sorted_simd: Vec<i32> = apply_indices(&data, &simd_asc);
            let sorted_auto_simd: Vec<i32> = apply_indices(&data, &auto_simd);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "simd test: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_simd,
                "simd test: comparison vs simd mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_auto_simd,
                "simd test: comparison vs auto_simd mismatch"
            );
        }

        #[cfg(feature = "simd_sort")]
        #[test]
        fn test_i64_sorts_produce_same_results_with_simd() {
            use super::simd_argsort;

            let data = vec![
                5i64,
                -2,
                8,
                -1,
                9,
                3,
                -7,
                4,
                6,
                0,
                i64::MAX,
                i64::MIN,
                0,
                -100,
                100,
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_i64(&data, false);
            let simd_asc = simd_argsort::argsort_simd_i64(&data, false);

            let sorted_comparison: Vec<i64> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<i64> = apply_indices(&data, &radix_asc);
            let sorted_simd: Vec<i64> = apply_indices(&data, &simd_asc);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "simd i64: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_simd,
                "simd i64: comparison vs simd mismatch"
            );
        }

        #[cfg(feature = "simd_sort")]
        #[test]
        fn test_u32_sorts_produce_same_results_with_simd() {
            use super::simd_argsort;

            let data = vec![
                5u32,
                2,
                8,
                1,
                9,
                3,
                7,
                4,
                6,
                0,
                u32::MAX,
                0,
                100,
                u32::MAX - 1,
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_u32(&data, false);
            let simd_asc = simd_argsort::argsort_simd_u32(&data, false);

            let sorted_comparison: Vec<u32> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<u32> = apply_indices(&data, &radix_asc);
            let sorted_simd: Vec<u32> = apply_indices(&data, &simd_asc);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "simd u32: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_simd,
                "simd u32: comparison vs simd mismatch"
            );
        }

        #[cfg(feature = "simd_sort")]
        #[test]
        fn test_u64_sorts_produce_same_results_with_simd() {
            use super::simd_argsort;

            let data = vec![
                5u64,
                2,
                8,
                1,
                9,
                3,
                7,
                4,
                6,
                0,
                u64::MAX,
                0,
                100,
                u64::MAX - 1,
            ];

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_u64(&data, false);
            let simd_asc = simd_argsort::argsort_simd_u64(&data, false);

            let sorted_comparison: Vec<u64> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<u64> = apply_indices(&data, &radix_asc);
            let sorted_simd: Vec<u64> = apply_indices(&data, &simd_asc);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "simd u64: comparison vs radix mismatch"
            );
            assert_eq!(
                sorted_comparison, sorted_simd,
                "simd u64: comparison vs simd mismatch"
            );
        }

        /// Test with larger dataset to stress-test algorithm correctness
        #[test]
        fn test_numeric_sorts_produce_same_results_large_dataset() {
            // Generate deterministic "random" data using LCG
            fn lcg_next(state: &mut u64) -> i32 {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (*state >> 33) as i32
            }

            let mut state = 12345u64;
            let data: Vec<i32> = (0..10_000).map(|_| lcg_next(&mut state)).collect();

            let comparison_asc = argsort(&data, false);
            let radix_asc = argsort_radix_i32(&data, false);

            let sorted_comparison: Vec<i32> = apply_indices(&data, &comparison_asc);
            let sorted_radix: Vec<i32> = apply_indices(&data, &radix_asc);

            assert_eq!(
                sorted_comparison, sorted_radix,
                "large dataset: comparison vs radix mismatch"
            );

            // Verify it's actually sorted
            for i in 1..sorted_comparison.len() {
                assert!(
                    sorted_comparison[i - 1] <= sorted_comparison[i],
                    "large dataset: not sorted at index {}",
                    i
                );
            }
        }

        // ====================================================================
        // Stable sort tests
        // ====================================================================

        #[test]
        fn test_stable_sort_preserves_relative_order() {
            // [3, 1, 2, 1] -- two 1s at indices 1 and 3
            let data = [3i32, 1, 2, 1];
            let indices = argsort_with_stability(&data, false, true);
            // Stable: the 1 at index 1 should come before the 1 at index 3
            assert_eq!(indices[0], 1); // first 1
            assert_eq!(indices[1], 3); // second 1
            assert_eq!(indices[2], 2); // the 2
            assert_eq!(indices[3], 0); // the 3
        }

        #[test]
        fn test_stable_float_preserves_relative_order() {
            // Two equal 1.0 values at indices 1 and 3
            let data = [3.0f64, 1.0, 2.0, 1.0];
            let indices = argsort_float_with_stability(&data, false, true);
            assert_eq!(indices[0], 1); // first 1.0
            assert_eq!(indices[1], 3); // second 1.0
            assert_eq!(indices[2], 2); // 2.0
            assert_eq!(indices[3], 0); // 3.0
        }

        #[test]
        fn test_argsort_config_stable_default_true() {
            let config = ArgsortConfig::default();
            assert!(config.stable);
        }

        #[test]
        fn test_argsort_auto_stable_preserves_order() {
            let data = [3i32, 1, 2, 1];
            let config = ArgsortConfig::new(); // stable=true by default
            let indices = argsort_auto_i32(&data, &config);
            // With stable=true and Auto algorithm, should preserve relative order
            assert_eq!(indices[0], 1); // first 1
            assert_eq!(indices[1], 3); // second 1
        }

        #[test]
        fn test_argsort_auto_unstable_still_sorts_correctly() {
            let data = [5i32, 2, 8, 1, 9];
            let config = ArgsortConfig::new().stable(false);
            let indices = argsort_auto_i32(&data, &config);
            let sorted: Vec<i32> = indices.iter().map(|&i| data[i]).collect();
            assert_eq!(sorted, vec![1, 2, 5, 8, 9]);
        }

        // is_sorted tests

        #[test]
        fn test_is_sorted_i32_ascending() {
            assert!(is_sorted_i32(&[1i32, 2, 3, 4, 5], false));
            assert!(!is_sorted_i32(&[1i32, 3, 2, 4, 5], false));
            assert!(is_sorted_i32(&[1i32, 1, 2, 2, 3], false));
        }

        #[test]
        fn test_is_sorted_i32_descending() {
            assert!(is_sorted_i32(&[5i32, 4, 3, 2, 1], true));
            assert!(!is_sorted_i32(&[5i32, 3, 4, 2, 1], true));
            assert!(is_sorted_i32(&[3i32, 3, 2, 2, 1], true));
        }

        #[test]
        fn test_is_sorted_i64_ascending() {
            assert!(is_sorted_i64(&[10i64, 20, 30, 40, 50], false));
            assert!(!is_sorted_i64(&[10i64, 30, 20, 40, 50], false));
        }

        #[test]
        fn test_is_sorted_u32_ascending() {
            assert!(is_sorted_u32(&[0u32, 1, 2, 3, 100], false));
            assert!(!is_sorted_u32(&[0u32, 2, 1, 3, 100], false));
        }

        #[test]
        fn test_is_sorted_u64_descending() {
            assert!(is_sorted_u64(&[100u64, 50, 25, 10, 1], true));
            assert!(!is_sorted_u64(&[100u64, 50, 60, 10, 1], true));
        }

        #[test]
        fn test_is_sorted_edge_cases() {
            // Empty and single element
            assert!(is_sorted_i32(&[], false));
            assert!(is_sorted_i32(&[42], false));
            assert!(is_sorted_i32(&[], true));
            assert!(is_sorted_i32(&[42], true));

            // Two elements
            assert!(is_sorted_i32(&[1, 2], false));
            assert!(!is_sorted_i32(&[2, 1], false));
            assert!(is_sorted_i32(&[2, 1], true));
            assert!(!is_sorted_i32(&[1, 2], true));

            // All equal
            assert!(is_sorted_i32(&[5, 5, 5, 5], false));
            assert!(is_sorted_i32(&[5, 5, 5, 5], true));
        }

        #[test]
        fn test_is_sorted_i32_large_array() {
            // Larger than one SIMD width to exercise the vectorised path + tail
            let ascending: Vec<i32> = (0..200).collect();
            assert!(is_sorted_i32(&ascending, false));
            assert!(!is_sorted_i32(&ascending, true));

            let descending: Vec<i32> = (0..200).rev().collect();
            assert!(is_sorted_i32(&descending, true));
            assert!(!is_sorted_i32(&descending, false));

            // Sorted except last element
            let mut almost: Vec<i32> = (0..200).collect();
            almost[199] = 0;
            assert!(!is_sorted_i32(&almost, false));
        }

        #[test]
        fn test_is_sorted_f64_ascending() {
            assert!(is_sorted_f64(&[1.0f64, 2.0, 3.0, 4.0, 5.0], false));
            assert!(!is_sorted_f64(&[1.0f64, 3.0, 2.0, 4.0, 5.0], false));
        }

        #[test]
        fn test_is_sorted_f64_descending() {
            assert!(is_sorted_f64(&[5.0f64, 4.0, 3.0, 2.0, 1.0], true));
            assert!(!is_sorted_f64(&[5.0f64, 3.0, 4.0, 2.0, 1.0], true));
        }

        #[test]
        fn test_is_sorted_f64_nan() {
            // NaN sorts after +inf in total ordering
            let with_nan = [1.0f64, 2.0, f64::INFINITY, f64::NAN];
            assert!(is_sorted_f64(&with_nan, false));

            // NaN first is not ascending
            let nan_first = [f64::NAN, 1.0, 2.0];
            assert!(!is_sorted_f64(&nan_first, false));

            // NaN then descending values is descending
            let nan_desc = [f64::NAN, f64::INFINITY, 2.0, 1.0];
            assert!(is_sorted_f64(&nan_desc, true));
        }

        #[test]
        fn test_is_sorted_f64_negatives() {
            assert!(is_sorted_f64(&[-3.0f64, -2.0, -1.0, 0.0, 1.0], false));
            assert!(!is_sorted_f64(&[-3.0f64, -1.0, -2.0, 0.0, 1.0], false));
            assert!(is_sorted_f64(&[1.0f64, 0.0, -1.0, -2.0, -3.0], true));
        }

        #[test]
        fn test_is_sorted_f64_special_values() {
            // -inf < negatives < -0 < +0 < positives < +inf < NaN
            let special = [
                f64::NEG_INFINITY,
                -1.0,
                -0.0f64,
                0.0,
                1.0,
                f64::INFINITY,
                f64::NAN,
            ];
            assert!(is_sorted_f64(&special, false));
        }

        #[test]
        fn test_is_sorted_f32_ascending() {
            assert!(is_sorted_f32(&[1.0f32, 2.0, 3.0], false));
            assert!(!is_sorted_f32(&[1.0f32, 3.0, 2.0], false));
        }

        #[test]
        fn test_is_sorted_f64_large_array() {
            // Larger than SIMD width to exercise the vectorised path
            let ascending: Vec<f64> = (0..200).map(|i| i as f64).collect();
            assert!(is_sorted_f64(&ascending, false));
            assert!(!is_sorted_f64(&ascending, true));

            let descending: Vec<f64> = (0..200).rev().map(|i| i as f64).collect();
            assert!(is_sorted_f64(&descending, true));
            assert!(!is_sorted_f64(&descending, false));
        }

        #[test]
        fn test_is_sorted_str() {
            assert!(is_sorted_str(&["a", "b", "c"], false));
            assert!(!is_sorted_str(&["a", "c", "b"], false));
            assert!(is_sorted_str(&["c", "b", "a"], true));
            assert!(is_sorted_str(&[], false));
            assert!(is_sorted_str(&["x"], true));
        }

        #[test]
        fn test_is_sorted_string_array() {
            // "apple" "banana" "cherry" - ascending
            let values = b"applebananacherry";
            let offsets: &[u32] = &[0, 5, 11, 17];
            assert!(is_sorted_string_array(offsets, values, false));
            assert!(!is_sorted_string_array(offsets, values, true));

            // "cherry" "banana" "apple" - descending
            let values2 = b"cherrybananaapple";
            let offsets2: &[u32] = &[0, 6, 12, 17];
            assert!(is_sorted_string_array(offsets2, values2, true));
            assert!(!is_sorted_string_array(offsets2, values2, false));

            // Works with u64 offsets too (String64)
            let offsets_u64: &[u64] = &[0, 5, 11, 17];
            assert!(is_sorted_string_array(offsets_u64, values, false));
        }

        #[test]
        fn test_is_sorted_generic_int() {
            // Generic version (scalar-only, any Ord type)
            assert!(is_sorted_int(&[1i8, 2, 3], false));
            assert!(!is_sorted_int(&[3i8, 1, 2], false));
            assert!(is_sorted_int::<i8>(&[], false));
            assert!(is_sorted_int(&[42i8], true));
        }

        #[test]
        fn test_is_sorted_generic_float() {
            // Generic version (scalar-only, any Float type)
            assert!(is_sorted_float(&[1.0f64, 2.0, 3.0], false));
            assert!(!is_sorted_float(&[3.0f64, 1.0, 2.0], false));
            assert!(is_sorted_float::<f64>(&[], false));
            assert!(is_sorted_float(&[f64::NAN], true));
        }
    }
}
