// Copyright (c) 2025 Peter Bower
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Conditional Logic Kernels Module** - *High-Performance Conditional Operations and Data Selection*
//!
//! Advanced conditional logic kernels providing efficient data selection, filtering, and transformation
//! operations with comprehensive null handling and SIMD acceleration. Essential infrastructure
//! for implementing complex analytical workflows and query execution.
//!
//! ## Core Operations
//! - **Conditional selection**: IF-THEN-ELSE operations with three-valued logic support
//! - **Array filtering**: Efficient boolean mask-based filtering with zero-copy optimisation
//! - **Coalescing operations**: Null-aware value selection with fallback hierarchies
//! - **Case-when logic**: Multi-condition branching with optimised evaluation strategies
//! - **Null propagation**: Comprehensive null handling following Apache Arrow semantics
//! - **Type preservation**: Maintains input data types through conditional transformations

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::marker::PhantomData;

#[cfg(feature = "fast_hash")]
use ahash::AHashMap;
#[cfg(not(feature = "fast_hash"))]
use std::collections::HashMap;

use minarrow::{
    Bitmask, BooleanAVT, BooleanArray, CategoricalAVT, CategoricalArray, FloatArray, Integer,
    IntegerArray, StringAVT, StringArray, Vec64, enums::error::KernelError,
    utils::confirm_equal_len,
};

#[cfg(feature = "simd")]
use core::simd::{Mask, Select, Simd};

#[cfg(feature = "simd")]
use minarrow::utils::is_simd_aligned;

use minarrow::utils::confirm_capacity;
#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, TimeUnit};

#[inline(always)]
fn prealloc_vec<T: Copy>(len: usize) -> Vec64<T> {
    let mut v = Vec64::<T>::with_capacity(len);
    // SAFETY: every slot is written before any read
    unsafe { v.set_len(len) };
    v
}

// Numeric Int float
macro_rules! impl_conditional_copy_numeric {
    ($fn_name:ident, $fn_name_to:ident, $ty:ty, $mask_elem:ty, $lanes:expr, $array_ty:ident) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Conditional copy operation: select elements from `then_data` or `else_data` based on boolean mask.
        /// Panics if output.len() != mask.len.
        #[inline(always)]
        pub fn $fn_name_to(
            mask: &BooleanArray<()>,
            then_data: &[$ty],
            else_data: &[$ty],
            output: &mut [$ty],
        ) {
            let len = mask.len;
            assert_eq!(
                len,
                output.len(),
                concat!(stringify!($fn_name_to), ": output length mismatch")
            );
            let mask_data = &mask.data;

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(then_data) && is_simd_aligned(else_data) {
                    const N: usize = $lanes;
                    let mut i = 0;
                    while i + N <= len {
                        let mut bits = [false; N];
                        for l in 0..N {
                            bits[l] = unsafe { mask_data.get_unchecked(i + l) };
                        }
                        let cond = Mask::<$mask_elem, N>::from_array(bits);
                        let t = Simd::<$ty, N>::from_slice(&then_data[i..i + N]);
                        let e = Simd::<$ty, N>::from_slice(&else_data[i..i + N]);
                        cond.select(t, e).copy_to_slice(&mut output[i..i + N]);
                        i += N;
                    }
                    // Tail often caused by `n % LANES != 0`; uses scalar fallback.
                    for j in i..len {
                        output[j] = if unsafe { mask_data.get_unchecked(j) } {
                            then_data[j]
                        } else {
                            else_data[j]
                        };
                    }
                    return;
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            for i in 0..len {
                output[i] = if unsafe { mask_data.get_unchecked(i) } {
                    then_data[i]
                } else {
                    else_data[i]
                };
            }
        }

        /// Conditional copy operation: select elements from `then_data` or `else_data` based on boolean mask.
        #[inline(always)]
        pub fn $fn_name(
            mask: &BooleanArray<()>,
            then_data: &[$ty],
            else_data: &[$ty],
        ) -> $array_ty<$ty> {
            let len = mask.len;
            let mut data = prealloc_vec::<$ty>(len);
            $fn_name_to(mask, then_data, else_data, &mut data);
            $array_ty {
                data: data.into(),
                null_mask: mask.null_mask.clone(),
            }
        }
    };
}

// Conditional datetime
#[cfg(feature = "datetime")]
macro_rules! impl_conditional_copy_datetime {
    ($fn_name:ident, $fn_name_to:ident, $ty:ty, $mask_elem:ty, $lanes:expr) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Panics if output.len() != mask.len.
        #[inline(always)]
        pub fn $fn_name_to(
            mask: &BooleanArray<()>,
            then_data: &[$ty],
            else_data: &[$ty],
            output: &mut [$ty],
        ) {
            let len = mask.len;
            assert_eq!(
                len,
                output.len(),
                concat!(stringify!($fn_name_to), ": output length mismatch")
            );
            let mask_data = &mask.data;

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(then_data) && is_simd_aligned(else_data) {
                    use core::simd::{Mask, Select, Simd};

                    const N: usize = $lanes;
                    let mut i = 0;
                    while i + N <= len {
                        let mut bits = [false; N];
                        for l in 0..N {
                            bits[l] = unsafe { mask_data.get_unchecked(i + l) };
                        }
                        let cond = Mask::<$mask_elem, N>::from_array(bits);
                        let t = Simd::<$ty, N>::from_slice(&then_data[i..i + N]);
                        let e = Simd::<$ty, N>::from_slice(&else_data[i..i + N]);
                        cond.select(t, e).copy_to_slice(&mut output[i..i + N]);
                        i += N;
                    }
                    // Scalar tail
                    for j in i..len {
                        output[j] = if unsafe { mask_data.get_unchecked(j) } {
                            then_data[j]
                        } else {
                            else_data[j]
                        };
                    }
                    return;
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            for i in 0..len {
                output[i] = unsafe {
                    if mask_data.get_unchecked(i) {
                        then_data[i]
                    } else {
                        else_data[i]
                    }
                };
            }
        }

        #[inline(always)]
        pub fn $fn_name(
            mask: &BooleanArray<()>,
            then_data: &[$ty],
            else_data: &[$ty],
            time_unit: TimeUnit,
        ) -> DatetimeArray<$ty> {
            let len = mask.len;
            let mut data = prealloc_vec::<$ty>(len);
            $fn_name_to(mask, then_data, else_data, &mut data);
            DatetimeArray {
                data: data.into(),
                null_mask: mask.null_mask.clone(),
                time_unit,
            }
        }
    };
}

#[cfg(feature = "extended_numeric_types")]
impl_conditional_copy_numeric!(
    conditional_copy_i8,
    conditional_copy_i8_to,
    i8,
    i8,
    W8,
    IntegerArray
);
#[cfg(feature = "extended_numeric_types")]
impl_conditional_copy_numeric!(
    conditional_copy_u8,
    conditional_copy_u8_to,
    u8,
    i8,
    W8,
    IntegerArray
);
#[cfg(feature = "extended_numeric_types")]
impl_conditional_copy_numeric!(
    conditional_copy_i16,
    conditional_copy_i16_to,
    i16,
    i16,
    W16,
    IntegerArray
);
#[cfg(feature = "extended_numeric_types")]
impl_conditional_copy_numeric!(
    conditional_copy_u16,
    conditional_copy_u16_to,
    u16,
    i16,
    W16,
    IntegerArray
);
impl_conditional_copy_numeric!(
    conditional_copy_i32,
    conditional_copy_i32_to,
    i32,
    i32,
    W32,
    IntegerArray
);
impl_conditional_copy_numeric!(
    conditional_copy_u32,
    conditional_copy_u32_to,
    u32,
    i32,
    W32,
    IntegerArray
);
impl_conditional_copy_numeric!(
    conditional_copy_i64,
    conditional_copy_i64_to,
    i64,
    i64,
    W64,
    IntegerArray
);
impl_conditional_copy_numeric!(
    conditional_copy_u64,
    conditional_copy_u64_to,
    u64,
    i64,
    W64,
    IntegerArray
);
impl_conditional_copy_numeric!(
    conditional_copy_f32,
    conditional_copy_f32_to,
    f32,
    i32,
    W32,
    FloatArray
);
impl_conditional_copy_numeric!(
    conditional_copy_f64,
    conditional_copy_f64_to,
    f64,
    i64,
    W64,
    FloatArray
);

#[cfg(feature = "datetime")]
impl_conditional_copy_datetime!(
    conditional_copy_datetime32,
    conditional_copy_datetime32_to,
    i32,
    i32,
    W32
);
#[cfg(feature = "datetime")]
impl_conditional_copy_datetime!(
    conditional_copy_datetime64,
    conditional_copy_datetime64_to,
    i64,
    i64,
    W64
);

/// Zero-allocation variant: writes directly to caller's output buffer.
///
/// Conditional copy for floating-point arrays with runtime type dispatch.
/// Panics if output.len() != mask.len.
#[inline(always)]
pub fn conditional_copy_float_to<T: Copy + 'static>(
    mask: &BooleanArray<()>,
    then_data: &[T],
    else_data: &[T],
    output: &mut [T],
) {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        unsafe {
            conditional_copy_f32_to(
                mask,
                std::mem::transmute(then_data),
                std::mem::transmute(else_data),
                std::mem::transmute(output),
            )
        };
        return;
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        unsafe {
            conditional_copy_f64_to(
                mask,
                std::mem::transmute(then_data),
                std::mem::transmute(else_data),
                std::mem::transmute(output),
            )
        };
        return;
    }
    unreachable!("unsupported float type")
}

/// Conditional copy for floating-point arrays with runtime type dispatch.
#[inline(always)]
pub fn conditional_copy_float<T: Copy + 'static>(
    mask: &BooleanArray<()>,
    then_data: &[T],
    else_data: &[T],
) -> FloatArray<T> {
    let len = mask.len;
    let mut data = prealloc_vec::<T>(len);
    conditional_copy_float_to(mask, then_data, else_data, &mut data);
    FloatArray {
        data: data.into(),
        null_mask: mask.null_mask.clone(),
    }
}

// Bit-packed Boolean
/// Zero-allocation variant: writes directly to caller's output buffer.
///
/// Conditional copy operation for boolean bitmask arrays.
/// The output Bitmask must have capacity >= mask.len.
pub fn conditional_copy_bool_to(
    mask: &BooleanArray<()>,
    then_data: &Bitmask,
    else_data: &Bitmask,
    output: &mut Bitmask,
) -> Result<(), KernelError> {
    let len_bits = mask.len;
    confirm_capacity("if_then_else: then_data", then_data.capacity(), len_bits)?;
    confirm_capacity("if_then_else: else_data", else_data.capacity(), len_bits)?;
    confirm_capacity("if_then_else: output", output.capacity(), len_bits)?;

    // Dense fast-path
    if mask.null_mask.is_none() {
        for i in 0..len_bits {
            let m = unsafe { mask.data.get_unchecked(i) };
            let v = if m {
                unsafe { then_data.get_unchecked(i) }
            } else {
                unsafe { else_data.get_unchecked(i) }
            };
            output.set(i, v);
        }
        return Ok(());
    }

    // Null-aware path
    for i in 0..len_bits {
        if !mask
            .null_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) })
        {
            continue;
        }
        let choose_then = unsafe { mask.data.get_unchecked(i) };
        let v = if choose_then {
            unsafe { then_data.get_unchecked(i) }
        } else {
            unsafe { else_data.get_unchecked(i) }
        };
        output.set(i, v);
    }

    Ok(())
}

/// Conditional copy operation for boolean bitmask arrays.
pub fn conditional_copy_bool(
    mask: &BooleanArray<()>,
    then_data: &Bitmask,
    else_data: &Bitmask,
) -> Result<BooleanArray<()>, KernelError> {
    let len_bits = mask.len;
    let mut out = Bitmask::with_capacity(len_bits);
    conditional_copy_bool_to(mask, then_data, else_data, &mut out)?;

    // Dense fast-path
    if mask.null_mask.is_none() {
        return Ok(BooleanArray {
            data: out.into(),
            null_mask: None,
            len: len_bits,
            _phantom: PhantomData,
        });
    }

    // Null-aware path: build output mask
    let mut out_mask = Bitmask::new_set_all(len_bits, false);
    for i in 0..len_bits {
        if mask
            .null_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) })
        {
            out_mask.set(i, true);
        }
    }

    Ok(BooleanArray {
        data: out.into(),
        null_mask: Some(out_mask),
        len: len_bits,
        _phantom: PhantomData,
    })
}

// Strings
#[inline(always)]
/// Conditional copy operation for UTF-8 string arrays.
pub fn conditional_copy_str<'a, T: Integer>(
    mask: BooleanAVT<'a, ()>,
    then_arr: StringAVT<'a, T>,
    else_arr: StringAVT<'a, T>,
) -> Result<StringArray<T>, KernelError> {
    let (mask_arr, mask_off, mask_len) = mask;
    let (then_arr, then_off, then_len) = then_arr;
    let (else_arr, else_off, else_len) = else_arr;

    confirm_equal_len(
        "if_then_else: then_arr.len() != mask_len",
        then_len,
        mask_len,
    )?;
    confirm_equal_len(
        "if_then_else: else_arr.len() != mask_len",
        else_len,
        mask_len,
    )?;

    // First pass: compute total bytes required
    let mut total_bytes = 0;
    for i in 0..mask_len {
        let idx = mask_off + i;
        let valid = mask_arr
            .null_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(idx) });
        if valid {
            let use_then = unsafe { mask_arr.data.get_unchecked(idx) };
            let s = unsafe {
                if use_then {
                    then_arr.get_str_unchecked(then_off + i)
                } else {
                    else_arr.get_str_unchecked(else_off + i)
                }
            };
            total_bytes += s.len();
        }
    }

    // Allocate output
    let mut offsets = Vec64::<T>::with_capacity(mask_len + 1);
    let mut values = Vec64::<u8>::with_capacity(total_bytes);
    let mut out_mask = Bitmask::new_set_all(mask_len, false);
    unsafe {
        offsets.set_len(mask_len + 1);
    }

    // Fill
    offsets[0] = T::zero();
    let mut cur = 0;

    for i in 0..mask_len {
        let idx = mask_off + i;
        let mask_valid = mask_arr
            .null_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(idx) });
        if !mask_valid {
            offsets[i + 1] = T::from_usize(cur);
            continue;
        }

        let use_then = unsafe { mask_arr.data.get_unchecked(idx) };
        let s = unsafe {
            if use_then {
                then_arr.get_str_unchecked(then_off + i)
            } else {
                else_arr.get_str_unchecked(else_off + i)
            }
        }
        .as_bytes();

        values.extend_from_slice(s);
        cur += s.len();
        offsets[i + 1] = T::from_usize(cur);
        unsafe {
            out_mask.set_unchecked(i, true);
        }
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: values.into(),
        null_mask: Some(out_mask),
    })
}

// Dictionary

/// Conditional copy operation for dictionary/categorical arrays.
pub fn conditional_copy_dict32<'a, T: Integer>(
    mask: BooleanAVT<'a, ()>,
    then_arr: CategoricalAVT<'a, T>,
    else_arr: CategoricalAVT<'a, T>,
) -> Result<CategoricalArray<T>, KernelError> {
    let (mask_arr, mask_off, mask_len) = mask;
    let (then_arr, then_off, then_len) = then_arr;
    let (else_arr, else_off, else_len) = else_arr;

    confirm_equal_len(
        "if_then_else: then_arr.len() != mask_len",
        then_len,
        mask_len,
    )?;
    confirm_equal_len(
        "if_then_else: else_arr.len() != mask_len",
        else_len,
        mask_len,
    )?;

    if mask_len == 0 {
        return Ok(CategoricalArray {
            data: Vec64::new().into(),
            unique_values: Vec64::new(),
            null_mask: Some(Bitmask::new_set_all(0, false)),
        });
    }

    // Merge unique values
    let mut uniques = then_arr.unique_values.clone();
    for v in &else_arr.unique_values {
        if !uniques.contains(v) {
            uniques.push(v.clone());
        }
    }

    #[cfg(feature = "fast_hash")]
    let lookup: AHashMap<&str, T> = uniques
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), T::from_usize(i)))
        .collect();
    #[cfg(not(feature = "fast_hash"))]
    let lookup: HashMap<&str, T> = uniques
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), T::from_usize(i)))
        .collect();

    let mut data = Vec64::<T>::with_capacity(mask_len);
    unsafe {
        data.set_len(mask_len);
    }

    let mut out_mask = Bitmask::new_set_all(mask_len, false);

    for i in 0..mask_len {
        let mask_idx = mask_off + i;
        let then_idx = then_off + i;
        let else_idx = else_off + i;

        let mask_valid = mask_arr
            .null_mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(mask_idx) });
        if !mask_valid {
            data[i] = T::zero();
            continue;
        }

        let choose_then = unsafe { mask_arr.data.get_unchecked(mask_idx) };
        let (src_arr, src_idx, valid_mask) = if choose_then {
            (then_arr, then_idx, then_arr.null_mask.as_ref())
        } else {
            (else_arr, else_idx, else_arr.null_mask.as_ref())
        };

        if valid_mask.map_or(true, |m| unsafe { m.get_unchecked(src_idx) }) {
            let idx = unsafe { *src_arr.data.get_unchecked(src_idx) }.to_usize();
            let val = unsafe { src_arr.unique_values.get_unchecked(idx) };
            data[i] = *lookup.get(val.as_str()).unwrap();
            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            data[i] = T::zero();
        }
    }

    Ok(CategoricalArray {
        data: data.into(),
        unique_values: uniques,
        null_mask: Some(out_mask),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use minarrow::{
        Bitmask, BooleanArray, MaskedArray,
        structs::variants::{categorical::CategoricalArray, string::StringArray},
    };

    fn bm(bools: &[bool]) -> Bitmask {
        Bitmask::from_bools(bools)
    }

    fn bool_arr(bools: &[bool]) -> BooleanArray<()> {
        BooleanArray::from_slice(bools)
    }

    #[test]
    fn test_conditional_copy_numeric_no_null() {
        // i32
        let mask = bool_arr(&[true, false, true, false, false, true]);
        let then = vec![10, 20, 30, 40, 50, 60];
        let els = vec![1, 2, 3, 4, 5, 6];
        let arr = conditional_copy_i32(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[10, 2, 30, 4, 5, 60]);
        assert!(arr.null_mask.is_none());

        // f64
        let mask = bool_arr(&[true, false, false]);
        let then = vec![2.0, 4.0, 6.0];
        let els = vec![1.0, 3.0, 5.0];
        let arr = conditional_copy_f64(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[2.0, 3.0, 5.0]);
        assert!(arr.null_mask.is_none());
    }

    #[test]
    fn test_conditional_copy_numeric_with_null() {
        let mut mask = bool_arr(&[true, false, true]);
        mask.null_mask = Some(bm(&[true, false, true]));
        let then = vec![10i64, 20, 30];
        let els = vec![1i64, 2, 3];
        let arr = conditional_copy_i64(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[10, 2, 30]);
        let null_mask = arr.null_mask.as_ref().unwrap();
        assert_eq!(null_mask.get(0), true);
        assert_eq!(null_mask.get(1), false);
        assert_eq!(null_mask.get(2), true);
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn test_conditional_copy_numeric_edge_cases() {
        // Empty
        let mask = bool_arr(&[]);
        let then: Vec<u32> = vec![];
        let els: Vec<u32> = vec![];
        let arr = conditional_copy_u32(&mask, &then, &els);
        assert_eq!(arr.data.len(), 0);
        assert!(arr.null_mask.is_none());
        // 1-element, mask true/false/null
        let mask = bool_arr(&[true]);
        let then = vec![42];
        let els = vec![1];
        let arr = conditional_copy_u8(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[42]);
        let mask = bool_arr(&[false]);
        let arr = conditional_copy_u8(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[1]);
        let mut mask = bool_arr(&[true]);
        mask.null_mask = Some(bm(&[false]));
        let arr = conditional_copy_u8(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[42]);
        assert_eq!(arr.null_mask.as_ref().unwrap().get(0), false);
    }

    #[test]
    fn test_conditional_copy_bool_no_null() {
        let mask = bool_arr(&[true, false, true, false]);
        let then = bm(&[true, true, false, false]);
        let els = bm(&[false, true, true, false]);
        let out = conditional_copy_bool(&mask, &then, &els).unwrap();
        assert_eq!(out.data.get(0), true);
        assert_eq!(out.data.get(1), true);
        assert_eq!(out.data.get(2), false);
        assert_eq!(out.data.get(3), false);
        assert!(out.null_mask.is_none());
        assert_eq!(out.len, 4);
    }

    #[test]
    fn test_conditional_copy_bool_with_null() {
        let mut mask = bool_arr(&[true, false, true, false, true]);
        mask.null_mask = Some(bm(&[true, false, true, true, false]));
        let then = bm(&[false, true, false, true, true]);
        let els = bm(&[true, false, true, false, false]);
        let out = conditional_copy_bool(&mask, &then, &els).unwrap();
        // Only positions 0,2,3 should be valid (others null)
        let null_mask = out.null_mask.as_ref().unwrap();
        assert_eq!(null_mask.get(0), true);
        assert_eq!(null_mask.get(1), false);
        assert_eq!(null_mask.get(2), true);
        assert_eq!(null_mask.get(3), true);
        assert_eq!(null_mask.get(4), false);
        assert_eq!(out.data.get(0), false);
        assert_eq!(out.data.get(2), false);
        assert_eq!(out.data.get(3), false);
    }

    #[test]
    fn test_conditional_copy_bool_edge_cases() {
        // Empty
        let mask = bool_arr(&[]);
        let then = bm(&[]);
        let els = bm(&[]);
        let out = conditional_copy_bool(&mask, &then, &els).unwrap();
        assert_eq!(out.len, 0);
        assert!(out.data.is_empty());
        // All nulls in mask
        let mut mask = bool_arr(&[true, false, true]);
        mask.null_mask = Some(bm(&[false, false, false]));
        let then = bm(&[true, true, true]);
        let els = bm(&[false, false, false]);
        let out = conditional_copy_bool(&mask, &then, &els).unwrap();
        assert_eq!(out.len, 3);
        let null_mask = out.null_mask.as_ref().unwrap();
        assert!(!null_mask.get(0));
        assert!(!null_mask.get(1));
        assert!(!null_mask.get(2));
    }

    #[test]
    fn test_conditional_copy_float_type_dispatch() {
        // f32
        let mask = bool_arr(&[true, false]);
        let then: Vec<f32> = vec![1.0, 2.0];
        let els: Vec<f32> = vec![3.0, 4.0];
        let arr = conditional_copy_float(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[1.0, 4.0]);
        // f64
        let mask = bool_arr(&[false, true]);
        let then: Vec<f64> = vec![7.0, 8.0];
        let els: Vec<f64> = vec![9.0, 10.0];
        let arr = conditional_copy_float(&mask, &then, &els);
        assert_eq!(&arr.data[..], &[9.0, 8.0]);
    }

    #[test]
    fn test_conditional_copy_str_basic() {
        // mask of length 4
        let mask = bool_arr(&[true, false, false, true]);

        // then_arr and else_arr must also be length 4
        let a = StringArray::<u32>::from_slice(&["foo", "bar", "baz", "qux"]);
        let b = StringArray::<u32>::from_slice(&["AAA", "Y", "Z", "BBB"]);

        // Wrap as slices
        let mask_slice = (&mask, 0, mask.len());
        let a_slice = (&a, 0, a.len());
        let b_slice = (&b, 0, b.len());

        let arr = conditional_copy_str(mask_slice, a_slice, b_slice).unwrap();

        // mask picks a[0], b[1], b[2], a[3]
        assert_eq!(arr.get(0), Some("foo"));
        assert_eq!(arr.get(1), Some("Y"));
        assert_eq!(arr.get(2), Some("Z"));
        assert_eq!(arr.get(3), Some("qux"));

        assert_eq!(arr.len(), 4);
        let nm = arr.null_mask.as_ref().unwrap();
        assert!(nm.all_set());
    }

    #[test]
    fn test_conditional_copy_str_with_null() {
        let mut mask = bool_arr(&[true, false, true]);
        mask.null_mask = Some(bm(&[true, false, true]));
        let mut a = StringArray::<u32>::from_slice(&["one", "two", "three"]);
        let mut b = StringArray::<u32>::from_slice(&["uno", "dos", "tres"]);
        a.set_null(2);
        b.set_null(0);
        let mask_slice = (&mask, 0, mask.len());
        let a_slice = (&a, 0, a.len());
        let b_slice = (&b, 0, b.len());

        let arr = conditional_copy_str(mask_slice, a_slice, b_slice).unwrap();

        assert_eq!(arr.get(0), Some("one"));
        assert!(arr.get(1).is_none());
        assert_eq!(arr.get(2), Some(""));
        let nm = arr.null_mask.as_ref().unwrap();
        assert!(nm.get(0));
        assert!(!nm.get(1));
        assert!(nm.get(2));
    }

    #[test]
    fn test_conditional_copy_str_with_null_chunk() {
        // pad to allow offset
        let mut mask = bool_arr(&[false, true, false, true, false]);
        mask.null_mask = Some(bm(&[false, true, false, true, false]));
        let mut a = StringArray::<u32>::from_slice(&["", "one", "two", "three", ""]);
        let mut b = StringArray::<u32>::from_slice(&["", "uno", "dos", "tres", ""]);
        a.set_null(3);
        b.set_null(1);
        let mask_slice = (&mask, 1, 3);
        let a_slice = (&a, 1, 3);
        let b_slice = (&b, 1, 3);

        let arr = conditional_copy_str(mask_slice, a_slice, b_slice).unwrap();

        assert_eq!(arr.get(0), Some("one"));
        assert!(arr.get(1).is_none());
        assert_eq!(arr.get(2), Some(""));
        let nm = arr.null_mask.as_ref().unwrap();
        assert!(nm.get(0));
        assert!(!nm.get(1));
        assert!(nm.get(2));
    }

    #[test]
    fn test_conditional_copy_str_edge_cases() {
        let mut mask = bool_arr(&[true, false]);
        mask.null_mask = Some(bm(&[false, false]));
        let a = StringArray::<u32>::from_slice(&["foo", "bar"]);
        let b = StringArray::<u32>::from_slice(&["baz", "qux"]);
        let mask_slice = (&mask, 0, mask.len());
        let a_slice = (&a, 0, a.len());
        let b_slice = (&b, 0, b.len());
        let arr = conditional_copy_str(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.len(), 2);
        assert!(!arr.null_mask.as_ref().unwrap().get(0));
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
        // Empty arrays
        let mask = bool_arr(&[]);
        let a = StringArray::<u32>::from_slice(&[]);
        let b = StringArray::<u32>::from_slice(&[]);
        let mask_slice = (&mask, 0, 0);
        let a_slice = (&a, 0, 0);
        let b_slice = (&b, 0, 0);
        let arr = conditional_copy_str(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn test_conditional_copy_str_edge_cases_chunk() {
        // chunked window: use 0-length window
        let mask = bool_arr(&[false, true, false]);
        let a = StringArray::<u32>::from_slice(&["foo", "bar", "baz"]);
        let b = StringArray::<u32>::from_slice(&["qux", "quux", "quuz"]);
        let mask_slice = (&mask, 1, 0);
        let a_slice = (&a, 1, 0);
        let b_slice = (&b, 1, 0);
        let arr = conditional_copy_str(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn test_conditional_copy_dict32_basic() {
        let mask = bool_arr(&[true, false, false, true]);
        let a = CategoricalArray::<u32>::from_slices(
            &[0, 1, 1, 0],
            &["dog".to_string(), "cat".to_string()],
        );
        let b = CategoricalArray::<u32>::from_slices(
            &[0, 0, 1, 1],
            &["fish".to_string(), "cat".to_string()],
        );
        let mask_slice = (&mask, 0, mask.len());
        let a_slice = (&a, 0, a.data.len());
        let b_slice = (&b, 0, b.data.len());

        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();

        let vals: Vec<Option<&str>> = (0..4).map(|i| arr.get(i)).collect();
        assert_eq!(vals[0], Some("dog"));
        assert_eq!(vals[1], Some("fish"));
        assert_eq!(vals[2], Some("cat"));
        assert_eq!(vals[3], Some("dog"));

        let mut all = arr
            .unique_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        all.sort();
        let mut ref_all = vec!["cat", "dog", "fish"];
        ref_all.sort();
        assert_eq!(all, ref_all);

        assert!(arr.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_conditional_copy_dict32_basic_chunk() {
        let mask = bool_arr(&[false, true, false, true, false]);
        let a = CategoricalArray::<u32>::from_slices(
            &[0, 1, 1, 0, 0],
            &["dog".to_string(), "cat".to_string()],
        );
        let b = CategoricalArray::<u32>::from_slices(
            &[0, 0, 1, 1, 0],
            &["fish".to_string(), "cat".to_string()],
        );
        let mask_slice = (&mask, 1, 3);
        let a_slice = (&a, 1, 3);
        let b_slice = (&b, 1, 3);

        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();

        let vals: Vec<Option<&str>> = (0..3).map(|i| arr.get(i)).collect();
        assert_eq!(vals[0], Some("cat"));
        assert_eq!(vals[1], Some("cat"));
        assert_eq!(vals[2], Some("dog"));

        let mut all = arr
            .unique_values
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        all.sort();
        let mut ref_all = vec!["cat", "dog", "fish"];
        ref_all.sort();
        assert_eq!(all, ref_all);

        assert!(arr.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_conditional_copy_dict32_with_null() {
        let mut mask = bool_arr(&[true, false, true]);
        mask.null_mask = Some(bm(&[true, false, true]));
        let mut a = CategoricalArray::<u32>::from_slices(
            &[0, 1, 0],
            &["dog".to_string(), "cat".to_string()],
        );
        let mut b = CategoricalArray::<u32>::from_slices(
            &[1, 0, 1],
            &["cat".to_string(), "fish".to_string()],
        );
        a.set_null(2);
        b.set_null(0);
        let mask_slice = (&mask, 0, mask.len());
        let a_slice = (&a, 0, a.data.len());
        let b_slice = (&b, 0, b.data.len());
        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.get(0), Some("dog"));
        assert!(arr.get(1).is_none());
        assert!(arr.get(2).is_none());
        assert!(arr.null_mask.as_ref().unwrap().get(0));
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
        assert!(!arr.null_mask.as_ref().unwrap().get(2));
    }

    #[test]
    fn test_conditional_copy_dict32_with_null_chunk() {
        let mut mask = bool_arr(&[false, true, false, true]);
        mask.null_mask = Some(bm(&[false, true, false, true]));
        let mut a = CategoricalArray::<u32>::from_slices(
            &[0, 1, 0, 1],
            &["dog".to_string(), "cat".to_string()],
        );
        let mut b = CategoricalArray::<u32>::from_slices(
            &[1, 0, 1, 0],
            &["cat".to_string(), "fish".to_string()],
        );
        a.set_null(3);
        b.set_null(1);
        let mask_slice = (&mask, 1, 2);
        let a_slice = (&a, 1, 2);
        let b_slice = (&b, 1, 2);
        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.get(0), Some("cat"));
        assert!(arr.get(1).is_none());
        assert!(arr.null_mask.as_ref().unwrap().get(0));
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
    }

    #[test]
    fn test_conditional_copy_dict32_edge_cases() {
        let mask = bool_arr(&[]);
        let a = CategoricalArray::<u32>::from_slices(&[], &[]);
        let b = CategoricalArray::<u32>::from_slices(&[], &[]);
        let mask_slice = (&mask, 0, 0);
        let a_slice = (&a, 0, 0);
        let b_slice = (&b, 0, 0);
        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.data.len(), 0);
        assert_eq!(arr.unique_values.len(), 0);

        let mut mask = bool_arr(&[true, false]);
        mask.null_mask = Some(bm(&[false, false]));
        let a = CategoricalArray::<u32>::from_slices(&[0, 0], &["foo".to_string()]);
        let b = CategoricalArray::<u32>::from_slices(&[0, 0], &["bar".to_string()]);
        let mask_slice = (&mask, 0, mask.len());
        let a_slice = (&a, 0, a.data.len());
        let b_slice = (&b, 0, b.data.len());
        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();
        assert!(!arr.null_mask.as_ref().unwrap().get(0));
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
    }

    #[test]
    fn test_conditional_copy_dict32_edge_cases_chunk() {
        let mask = bool_arr(&[false, false, false]);
        let a = CategoricalArray::<u32>::from_slices(&[0, 0, 0], &["foo".to_string()]);
        let b = CategoricalArray::<u32>::from_slices(&[0, 0, 0], &["bar".to_string()]);
        let mask_slice = (&mask, 1, 0);
        let a_slice = (&a, 1, 0);
        let b_slice = (&b, 1, 0);
        let arr = conditional_copy_dict32(mask_slice, a_slice, b_slice).unwrap();
        assert_eq!(arr.data.len(), 0);
        assert_eq!(arr.unique_values.len(), 0);
    }
}
