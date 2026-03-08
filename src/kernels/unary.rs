// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Unary Operations Kernels Module** - *Single-Array Transformations*
//!
//! Unary operation kernels for single-array transformations
//! with SIMD acceleration and null-aware semantics. Essential building blocks for data
//! preprocessing, mathematical transformations, and analytical computations.
//!
//! ## Core Operations
//! - **Mathematical functions**: Absolute value, negation, square root, logarithmic, and trigonometric functions
//! - **Type casting**: Safe and unsafe type conversions with overflow detection
//! - **Boolean operations**: Logical NOT operations on boolean arrays with bitmask optimisation
//! - **Null handling**: NULL coalescing and null indicator operations  
//! - **Statistical transforms**: Standardisation, normalisation, and ranking operations
//! - **String transformations**: Length calculation, case conversion, and format operations

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::hash::Hash;
#[cfg(feature = "simd")]
use std::simd::num::SimdUint;

#[cfg(feature = "fast_hash")]
use ahash::AHashMap;
use minarrow::{
    BooleanAVT, BooleanArray, CategoricalAVT, CategoricalArray, FloatAVT, FloatArray, Integer,
    IntegerAVT, IntegerArray, StringAVT, StringArray, Vec64,
};
#[cfg(feature = "datetime")]
use minarrow::{DatetimeAVT, DatetimeArray};
use num_traits::{Float, Signed};
#[cfg(not(feature = "fast_hash"))]
use std::collections::HashMap;

use crate::kernels::logical::not_bool;
use minarrow::enums::error::KernelError;

// Helper

#[inline(always)]
fn prealloc_vec<T: Copy>(len: usize) -> Vec64<T> {
    let mut v = Vec64::<T>::with_capacity(len);
    // SAFETY: we will write every slot beforee reading.
    unsafe { v.set_len(len) };
    v
}

// SIMD helpers

#[cfg(feature = "simd")]
mod simd_impl {
    use core::simd::{Simd, SimdElement};

    use num_traits::Zero;

    /// `out = -a`  (works even if `Simd` lacks a direct `Neg`)
    #[inline(always)]
    pub fn negate_dense<T, const LANES: usize>(a: &[T], out: &mut [T])
    where
        T: SimdElement + core::ops::Neg<Output = T> + Copy + Zero,
        Simd<T, LANES>: core::ops::Sub<Output = Simd<T, LANES>>,
    {
        let mut i = 0;
        while i + LANES <= a.len() {
            let v = Simd::<T, LANES>::from_slice(&a[i..i + LANES]);
            let res = Simd::<T, LANES>::splat(T::zero()) - v;
            res.copy_to_slice(&mut out[i..i + LANES]);
            i += LANES;
        }
        // Tail often caused by `n % LANES != 0`; uses scalar fallback.
        for j in i..a.len() {
            out[j] = -a[j];
        }
    }
}

// Integer negate

/// Generates integer negation functions with SIMD optimisation.
macro_rules! impl_unary_neg_int {
    ($fn_name:ident, $fn_name_to:ident, $ty:ty, $lanes:expr) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Canonical implementation with SIMD acceleration.
        /// Panics if input.len() != output.len().
        #[inline(always)]
        pub fn $fn_name_to(input: &[$ty], output: &mut [$ty]) {
            assert_eq!(
                input.len(),
                output.len(),
                concat!(stringify!($fn_name_to), ": input/output length mismatch")
            );

            #[cfg(feature = "simd")]
            simd_impl::negate_dense::<$ty, $lanes>(input, output);

            #[cfg(not(feature = "simd"))]
            for i in 0..input.len() {
                output[i] = -input[i];
            }
        }

        /// Negates all elements in an integer array.
        ///
        /// Applies the unary negation operator to each element in the array,
        /// using SIMD operations when available for optimal performance.
        ///
        /// # Arguments
        ///
        /// * `window` - Integer array view tuple (array, offset, length)
        ///
        /// # Returns
        ///
        /// New integer array containing negated values
        #[inline(always)]
        pub fn $fn_name(window: IntegerAVT<$ty>) -> IntegerArray<$ty> {
            let (arr, offset, len) = window;
            let src = &arr.data[offset..offset + len];
            let mut data = prealloc_vec::<$ty>(len);

            $fn_name_to(src, &mut data);

            IntegerArray {
                data: data.into(),
                null_mask: arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len)),
            }
        }
    };
}

#[cfg(feature = "datetime")]
/// Generates datetime negation functions with SIMD optimisation.
macro_rules! impl_unary_neg_datetime {
    ($fn_name:ident, $fn_name_to:ident, $ty:ty, $lanes:expr) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Canonical implementation with SIMD acceleration.
        /// Panics if input.len() != output.len().
        #[inline(always)]
        pub fn $fn_name_to(input: &[$ty], output: &mut [$ty]) {
            assert_eq!(
                input.len(),
                output.len(),
                concat!(stringify!($fn_name_to), ": input/output length mismatch")
            );

            #[cfg(feature = "simd")]
            simd_impl::negate_dense::<$ty, $lanes>(input, output);

            #[cfg(not(feature = "simd"))]
            for i in 0..input.len() {
                output[i] = -input[i];
            }
        }

        /// Negates all elements in a datetime array.
        ///
        /// Applies the unary negation operator to each element in the datetime array,
        /// using SIMD operations when available for optimal performance.
        ///
        /// # Arguments
        ///
        /// * `window` - Datetime array view tuple (array, offset, length)
        ///
        /// # Returns
        ///
        /// New datetime array containing negated values
        #[inline(always)]
        pub fn $fn_name(window: DatetimeAVT<$ty>) -> DatetimeArray<$ty> {
            let (arr, offset, len) = window;
            let src = &arr.data[offset..offset + len];
            let mut data = prealloc_vec::<$ty>(len);

            $fn_name_to(src, &mut data);

            DatetimeArray {
                data: data.into(),
                null_mask: arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len)),
                time_unit: arr.time_unit.clone(),
            }
        }
    };
}

#[cfg(feature = "datetime")]
/// Generic datetime negation dispatcher.
///
/// Dispatches to the appropriate type-specific datetime negation function
/// based on the concrete integer type at runtime.
///
/// # Type Parameters
///
/// * `T` - Signed integer type for datetime values
///
/// # Arguments
///
/// * `window` - Datetime array view tuple (array, offset, length)
///
/// # Returns
///
/// New datetime array containing negated values
#[inline(always)]
pub fn unary_negate_int_datetime<T>(window: DatetimeAVT<T>) -> DatetimeArray<T>
where
    T: Signed + Copy + 'static,
{
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        return unsafe { std::mem::transmute(unary_neg_datetime_i32(std::mem::transmute(window))) };
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        return unsafe { std::mem::transmute(unary_neg_datetime_i64(std::mem::transmute(window))) };
    }
    unreachable!("unsupported datetime type for negation")
}

#[cfg(feature = "datetime")]
impl_unary_neg_datetime!(unary_neg_datetime_i32, unary_neg_datetime_i32_to, i32, W32);
#[cfg(feature = "datetime")]
impl_unary_neg_datetime!(unary_neg_datetime_i64, unary_neg_datetime_i64_to, i64, W64);
#[cfg(feature = "extended_numeric_types")]
impl_unary_neg_int!(unary_neg_i8, unary_neg_i8_to, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_unary_neg_int!(unary_neg_i16, unary_neg_i16_to, i16, W16);
impl_unary_neg_int!(unary_neg_i32, unary_neg_i32_to, i32, W32);
impl_unary_neg_int!(unary_neg_i64, unary_neg_i64_to, i64, W64);

// Unified entry point

/// Generic integer negation dispatcher.
///
/// Dispatches to the appropriate type-specific integer negation function
/// based on the concrete integer type at runtime.
///
/// # Type Parameters
///
/// * `T` - Signed integer type
///
/// # Arguments
///
/// * `window` - Integer array view tuple (array, offset, length)
///
/// # Returns
///
/// New integer array containing negated values
#[inline(always)]
pub fn unary_negate_int<T>(window: IntegerAVT<T>) -> IntegerArray<T>
where
    T: Signed + Copy + 'static,
{
    macro_rules! dispatch {
        ($t:ty, $f:ident) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$t>() {
                return unsafe { std::mem::transmute($f(std::mem::transmute(window))) };
            }
        };
    }
    #[cfg(feature = "extended_numeric_types")]
    dispatch!(i8, unary_neg_i8);
    #[cfg(feature = "extended_numeric_types")]
    dispatch!(i16, unary_neg_i16);
    dispatch!(i32, unary_neg_i32);
    dispatch!(i64, unary_neg_i64);

    unreachable!("unsupported integer type")
}

/// Zero-allocation variant: writes directly to caller's output buffer.
///
/// Panics if input.len() != output.len().
#[inline(always)]
pub fn unary_negate_u32_to_i32_to(input: &[u32], output: &mut [i32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "unary_negate_u32_to_i32_to: input/output length mismatch"
    );

    for (dst, &v) in output.iter_mut().zip(input) {
        *dst = -(v as i32);
    }
}

/// Negates u32 values and converts them to i32.
///
/// Applies unary negation to unsigned 32-bit integers and returns
/// the result as signed 32-bit integers.
///
/// # Arguments
///
/// * `window` - u32 integer array view tuple (array, offset, length)
///
/// # Returns
///
/// New i32 integer array containing negated values
pub fn unary_negate_u32_to_i32(window: IntegerAVT<u32>) -> IntegerArray<i32> {
    let (arr, offset, len) = window;
    let src = &arr.data[offset..offset + len];
    let mut data = prealloc_vec::<i32>(len);

    unary_negate_u32_to_i32_to(src, &mut data);

    IntegerArray {
        data: data.into(),
        null_mask: arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len)),
    }
}

/// Zero-allocation variant: writes directly to caller's output buffer.
///
/// Panics if input.len() != output.len().
#[inline(always)]
pub fn unary_negate_u64_to_i64_to(input: &[u64], output: &mut [i64]) {
    let len = input.len();
    assert_eq!(
        len,
        output.len(),
        "unary_negate_u64_to_i64_to: input/output length mismatch"
    );

    #[cfg(feature = "simd")]
    {
        use core::simd::Simd;
        const LANES: usize = W64;
        let mut i = 0;
        while i + LANES <= len {
            let v = Simd::<u64, LANES>::from_slice(&input[i..i + LANES]).cast::<i64>();
            (Simd::<i64, LANES>::splat(0) - v).copy_to_slice(&mut output[i..i + LANES]);
            i += LANES;
        }
        for j in i..len {
            output[j] = -(input[j] as i64);
        }
    }
    #[cfg(not(feature = "simd"))]
    for (dst, &v) in output.iter_mut().zip(input) {
        *dst = -(v as i64);
    }
}

/// Negates u64 values and converts them to i64.
///
/// Applies unary negation to unsigned 64-bit integers and returns
/// the result as signed 64-bit integers.
///
/// # Arguments
///
/// * `window` - u64 integer array view tuple (array, offset, length)
///
/// # Returns
///
/// New i64 integer array containing negated values
pub fn unary_negate_u64_to_i64(window: IntegerAVT<u64>) -> IntegerArray<i64> {
    let (arr, offset, len) = window;
    let src = &arr.data[offset..offset + len];
    let mut data = prealloc_vec::<i64>(len);

    unary_negate_u64_to_i64_to(src, &mut data);

    IntegerArray {
        data: data.into(),
        null_mask: arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len)),
    }
}

// Float negate

/// Generates floating-point negation functions with SIMD optimisation.
macro_rules! impl_unary_neg_float {
    ($fname:ident, $fname_to:ident, $ty:ty, $lanes:expr) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Canonical implementation with SIMD acceleration.
        /// Panics if input.len() != output.len().
        #[inline(always)]
        pub fn $fname_to(input: &[$ty], output: &mut [$ty]) {
            assert_eq!(
                input.len(),
                output.len(),
                concat!(stringify!($fname_to), ": input/output length mismatch")
            );

            #[cfg(feature = "simd")]
            simd_impl::negate_dense::<$ty, $lanes>(input, output);

            #[cfg(not(feature = "simd"))]
            for i in 0..input.len() {
                output[i] = -input[i];
            }
        }

        /// Negates all elements in a floating-point array.
        ///
        /// Applies the unary negation operator to each element in the array,
        /// using SIMD operations when available for optimal performance.
        ///
        /// # Arguments
        ///
        /// * `window` - Float array view tuple (array, offset, length)
        ///
        /// # Returns
        ///
        /// New float array containing negated values
        #[inline(always)]
        pub fn $fname(window: FloatAVT<$ty>) -> FloatArray<$ty> {
            let (arr, offset, len) = window;
            let src = &arr.data[offset..offset + len];
            let mut data = prealloc_vec::<$ty>(len);

            $fname_to(src, &mut data);

            FloatArray {
                data: data.into(),
                null_mask: arr.null_mask.as_ref().map(|m| m.slice_clone(offset, len)),
            }
        }
    };
}

impl_unary_neg_float!(unary_neg_f32, unary_neg_f32_to, f32, W32);
impl_unary_neg_float!(unary_neg_f64, unary_neg_f64_to, f64, W64);

/// Generic floating-point negation dispatcher.
///
/// Dispatches to the appropriate type-specific negation function
/// based on the concrete float type at runtime.
///
/// # Type Parameters
///
/// * `T` - Floating-point type implementing Float trait
///
/// # Arguments
///
/// * `window` - Float array view tuple (array, offset, length)
///
/// # Returns
///
/// New float array containing negated values
#[inline(always)]
pub fn unary_negate_float<T>(window: FloatAVT<T>) -> FloatArray<T>
where
    T: Float + Copy + 'static,
{
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        return unsafe { std::mem::transmute(unary_neg_f32(std::mem::transmute(window))) };
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        return unsafe { std::mem::transmute(unary_neg_f64(std::mem::transmute(window))) };
    }
    unreachable!("unsupported float type")
}

/// Applies logical NOT to boolean values element-wise with SIMD acceleration.\n///\n/// Performs bitwise logical negation on boolean arrays using vectorised operations\n/// with configurable lane width for optimal performance.\n///\n/// # Parameters\n/// - `arr_window`: Boolean array view tuple `(BooleanArray, offset, length)`\n///\n/// # Const Generics\n/// - `LANES`: SIMD lane count for vectorised operations (typically W64)\n///\n/// # Returns\n/// `Result<BooleanArray<()>, KernelError>` with logically negated boolean values.\n///\n/// # Errors\n/// May return `KernelError` for invalid array configurations or memory issues.\n///\n/// # Performance\n/// Uses SIMD bitwise NOT operations with lane-parallel processing for bulk negation.
pub fn unary_not_bool<const LANES: usize>(
    arr_window: BooleanAVT<()>,
) -> Result<BooleanArray<()>, KernelError>
where
{
    not_bool::<LANES>(arr_window)
}

/// Reverses the character order within each string in a string array.
///
/// Creates a new string array where each string has its characters reversed,
/// preserving UTF-8 encoding and null mask patterns.
///
/// # Type Parameters
///
/// * `T` - Integer type for string array offsets
///
/// # Arguments
///
/// * `arr` - String array view tuple (array, offset, length)
///
/// # Returns
///
/// New string array with character-reversed strings
pub fn unary_reverse_str<T: Integer>(arr: StringAVT<T>) -> StringArray<T> {
    let (array, offset, len) = arr;
    let offsets = &array.offsets;
    let data_buf = &array.data;
    let mask = array.null_mask.as_ref();

    // Estimate output buffer size: sum of windowed input string lengths
    let total_bytes = if len == 0 {
        0
    } else {
        let start = offsets[offset].to_usize();
        let end = offsets[offset + len].to_usize();
        end - start
    };

    // Prepare output buffers
    let mut out_offsets = Vec64::<T>::with_capacity(len + 1);
    let mut out_data = Vec64::<u8>::with_capacity(total_bytes);
    unsafe {
        out_offsets.set_len(len + 1);
    }
    out_offsets[0] = T::zero();

    for i in 0..len {
        // Preserve payload (set empty string), do not write reversed string.
        if mask.map_or(true, |m| m.get(offset + i)) {
            let start = offsets[offset + i].to_usize();
            let end = offsets[offset + i + 1].to_usize();
            let s = unsafe { std::str::from_utf8_unchecked(&data_buf[start..end]) };
            for ch in s.chars().rev() {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                out_data.extend_from_slice(encoded.as_bytes());
            }
        }
        // For nulls-  do not append any bytes.
        out_offsets[i + 1] = T::from_usize(out_data.len());
    }

    let out_null_mask = mask.map(|m| m.slice_clone(offset, len));

    StringArray {
        offsets: out_offsets.into(),
        data: out_data.into(),
        null_mask: out_null_mask,
    }
}

/// Reverses the character order within each string in a categorical array dictionary.
///
/// Creates a new categorical array where the dictionary strings have their
/// characters reversed, while preserving the codes and null mask patterns.
///
/// # Type Parameters
///
/// * `T` - Integer type implementing Hash for categorical codes
///
/// # Arguments
///
/// * `arr` - Categorical array view tuple (array, offset, length)
///
/// # Returns
///
/// New categorical array with character-reversed dictionary strings
pub fn unary_reverse_dict<T: Integer + Hash>(arr: CategoricalAVT<T>) -> CategoricalArray<T> {
    let (array, offset, len) = arr;
    let mask = array.null_mask.as_ref();

    // Window the data codes
    let windowed_codes = array.data[offset..offset + len].to_vec();

    // Build the set of codes actually used in this window, remap to new indices.
    #[cfg(feature = "fast_hash")]
    let mut remap: AHashMap<T, T> = AHashMap::new();
    #[cfg(not(feature = "fast_hash"))]
    let mut remap: HashMap<T, T> = HashMap::new();
    let mut new_uniques = Vec64::<String>::new();
    let mut new_codes = Vec64::<T>::with_capacity(len);

    for &code in &windowed_codes {
        if !remap.contains_key(&code) {
            let reversed = array.unique_values[code.to_usize()]
                .chars()
                .rev()
                .collect::<String>();
            remap.insert(code, T::from_usize(new_uniques.len()));
            new_uniques.push(reversed);
        }
        new_codes.push(remap[&code]);
    }

    let out_null_mask = mask.map(|m| m.slice_clone(offset, len));

    CategoricalArray {
        data: new_codes.into(),
        unique_values: new_uniques,
        null_mask: out_null_mask,
    }
}

#[cfg(test)]
mod tests {
    use minarrow::structs::variants::categorical::CategoricalArray;
    use minarrow::structs::variants::float::FloatArray;
    use minarrow::structs::variants::integer::IntegerArray;
    use minarrow::structs::variants::string::StringArray;
    use minarrow::{Bitmask, BooleanArray, MaskedArray};

    use super::*;

    // Helpers

    fn bm(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            m.set(i, b);
        }
        m
    }

    fn expect_int<T: PartialEq + std::fmt::Debug>(
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

    fn expect_float<T: num_traits::Float + std::fmt::Debug>(
        arr: &FloatArray<T>,
        values: &[T],
        valid: &[bool],
        eps: T,
    ) {
        assert_eq!(arr.data.len(), values.len());
        for (a, b) in arr.data.iter().zip(values.iter()) {
            assert!((*a - *b).abs() <= eps, "value mismatch {:?} vs {:?}", a, b);
        }
        let mask = arr.null_mask.as_ref().expect("mask missing");
        for (i, &v) in valid.iter().enumerate() {
            assert_eq!(mask.get(i), v, "mask bit {}", i);
        }
    }

    // Integer Negation

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn neg_i8_dense() {
        let arr = IntegerArray::<i8>::from_slice(&[1, -2, 127]);
        let out = unary_neg_i8((&arr, 0, arr.len()));
        assert_eq!(out.data.as_slice(), &[-1, 2, -127]);
        assert!(out.null_mask.is_none());
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn neg_i16_masked() {
        let mut arr = IntegerArray::<i16>::from_slice(&[-4, 12, 8, 0]);
        arr.null_mask = Some(bm(&[true, false, true, true]));
        let out = unary_neg_i16((&arr, 0, arr.len()));
        expect_int(&out, &[4, -12, -8, 0], &[true, false, true, true]);
    }

    #[test]
    fn neg_i32_empty() {
        let arr = IntegerArray::<i32>::from_slice(&[]);
        let out = unary_neg_i32((&arr, 0, arr.len()));
        assert_eq!(out.data.len(), 0);
    }

    #[test]
    fn neg_i64_all_nulls() {
        let mut arr = IntegerArray::<i64>::from_slice(&[5, 10]);
        arr.null_mask = Some(bm(&[false, false]));
        let out = unary_neg_i64((&arr, 0, arr.len()));
        expect_int(&out, &[-5, -10], &[false, false]);
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn neg_dispatch_i16() {
        let mut arr = IntegerArray::<i16>::from_slice(&[-2, 4]);
        arr.null_mask = Some(bm(&[true, true]));
        let out = unary_negate_int((&arr, 0, arr.len()));
        expect_int(&out, &[2, -4], &[true, true]);
    }

    // Unsigned to Signed Negation

    #[test]
    fn neg_u32_to_i32() {
        let mut arr = IntegerArray::<u32>::from_slice(&[1, 2, 100]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let out = unary_negate_u32_to_i32((&arr, 0, arr.len()));
        expect_int(&out, &[-1, -2, -100], &[true, false, true]);
    }

    #[test]
    fn neg_u64_to_i64() {
        let arr = IntegerArray::<u64>::from_slice(&[3, 4, 0]);
        let out = unary_negate_u64_to_i64((&arr, 0, arr.len()));
        assert_eq!(out.data.as_slice(), &[-3, -4, 0]);
    }

    // Float Negation

    #[test]
    fn neg_f32_dense() {
        let arr = FloatArray::<f32>::from_slice(&[0.5, -1.5, 2.0]);
        let out = unary_neg_f32((&arr, 0, arr.len()));
        assert_eq!(out.data.as_slice(), &[-0.5, 1.5, -2.0]);
        assert!(out.null_mask.is_none());
    }

    #[test]
    fn neg_f64_masked() {
        let mut arr = FloatArray::<f64>::from_slice(&[1.1, -2.2, 3.3]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let out = unary_neg_f64((&arr, 0, arr.len()));
        expect_float(&out, &[-1.1, 2.2, -3.3], &[true, false, true], 1e-12);
    }

    #[test]
    fn neg_dispatch_f64() {
        let arr = FloatArray::<f64>::from_slice(&[2.2, -4.4]);
        let out = unary_negate_float((&arr, 0, arr.len()));
        assert_eq!(out.data.as_slice(), &[-2.2, 4.4]);
    }

    // Boolean NOT

    #[test]
    fn not_bool_basic() {
        let arr = BooleanArray::from_slice(&[true, false, true, false]);
        let out = unary_not_bool::<W64>((&arr, 0, arr.len())).unwrap();
        assert_eq!(out.data.as_slice(), &[0b00001010]);
        assert!(out.null_mask.is_none());
    }

    #[test]
    fn not_bool_masked() {
        let mut arr = BooleanArray::from_slice(&[false, false, true, true]);
        arr.null_mask = Some(bm(&[true, false, true, true]));
        let out = unary_not_bool::<W64>((&arr, 0, arr.len())).unwrap();
        assert_eq!(out.data.as_slice(), &[0b00000011]);
        assert_eq!(out.null_mask, arr.null_mask);
    }

    // String Reverse

    #[test]
    fn reverse_str_basic() {
        let arr = StringArray::<u32>::from_slice(&["ab", "xyz", ""]);
        let out = unary_reverse_str((&arr, 0, arr.len()));
        assert_eq!(out.get(0), Some("ba"));
        assert_eq!(out.get(1), Some("zyx"));
        assert_eq!(out.get(2), Some(""));
    }

    #[test]
    fn reverse_str_basic_chunk() {
        let arr = StringArray::<u32>::from_slice(&["xxx", "ab", "xyz", ""]);
        let out = unary_reverse_str((&arr, 1, 3)); // "ab", "xyz", ""
        assert_eq!(out.get(0), Some("ba"));
        assert_eq!(out.get(1), Some("zyx"));
        assert_eq!(out.get(2), Some(""));
    }

    #[test]
    fn reverse_str_with_nulls() {
        let mut arr = StringArray::<u32>::from_slice(&["apple", "banana", "carrot"]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let out = unary_reverse_str((&arr, 0, arr.len()));
        assert_eq!(out.get(0), Some("elppa"));
        assert_eq!(out.get(1), None);
        assert_eq!(out.get(2), Some("torrac"));
        assert_eq!(out.null_mask, arr.null_mask);
    }

    #[test]
    fn reverse_str_with_nulls_chunk() {
        let mut arr = StringArray::<u32>::from_slice(&["zero", "apple", "banana", "carrot"]);
        arr.null_mask = Some(bm(&[true, true, false, true]));
        let out = unary_reverse_str((&arr, 1, 3)); // "apple", "banana", "carrot"
        assert_eq!(out.get(0), Some("elppa"));
        assert_eq!(out.get(1), None);
        assert_eq!(out.get(2), Some("torrac"));
        assert_eq!(
            out.null_mask.as_ref().unwrap().as_slice(),
            bm(&[true, false, true]).as_slice()
        );
    }

    // Categorical Reverse

    #[test]
    fn reverse_dict_basic() {
        let arr = CategoricalArray::<u32>::from_values(["cat", "dog", "bird", "cat"]);
        let out = unary_reverse_dict((&arr, 0, arr.data.len()));
        let uniq: Vec<String> = out.unique_values.iter().map(|s| s.clone()).collect();
        assert!(uniq.contains(&"tac".to_string()));
        assert!(uniq.contains(&"god".to_string()));
        assert!(uniq.contains(&"drib".to_string()));
        assert_eq!(out.data, arr.data);
    }

    #[test]
    fn reverse_dict_basic_chunk() {
        let arr = CategoricalArray::<u32>::from_values(["z", "cat", "dog", "bird"]);
        let out = unary_reverse_dict((&arr, 1, 3));
        let uniq: Vec<String> = out.unique_values.iter().map(|s| s.clone()).collect();
        assert!(uniq.contains(&"tac".to_string()));
        assert!(uniq.contains(&"god".to_string()));
        assert!(uniq.contains(&"drib".to_string()));
        assert_eq!(&out.data[..], &[0, 1, 2]);
    }

    #[test]
    fn test_unary_reverse_str_empty_and_all_nulls() {
        // empty array
        let arr0 = StringArray::<u32>::from_slice(&[]);
        let out0 = unary_reverse_str((&arr0, 0, arr0.len()));
        assert_eq!(out0.len(), 0);

        // all-null array
        let mut arr1 = StringArray::<u32>::from_slice(&["a", "b"]);
        arr1.null_mask = Some(bm(&[false, false]));
        let out1 = unary_reverse_str((&arr1, 0, arr1.len()));
        // get() should return None for each, and mask preserved
        assert_eq!(out1.get(0), None);
        assert_eq!(out1.get(1), None);
        assert_eq!(out1.null_mask, arr1.null_mask);
    }

    #[test]
    fn test_unary_reverse_str_empty_and_all_nulls_chunk() {
        // chunk of all-null
        let mut arr = StringArray::<u32>::from_slice(&["x", "a", "b", "y"]);
        arr.null_mask = Some(bm(&[true, false, false, true]));
        let out = unary_reverse_str((&arr, 1, 2)); // "a", "b"
        assert_eq!(out.get(0), None);
        assert_eq!(out.get(1), None);
        assert_eq!(out.null_mask, Some(bm(&[false, false])));
    }
}
