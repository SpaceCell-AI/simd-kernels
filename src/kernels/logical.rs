// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Logical Operations Kernels Module** - *Boolean Logic and Set Operations*
//!
//! Logical operation kernels providing efficient boolean algebra, set membership testing,
//! and range operations with SIMD acceleration and null-aware semantics. Critical foundation
//! for query execution, filtering predicates, and analytical data processing workflows.
//!
//! ## Core Operations
//! - **Boolean algebra**: AND, OR, XOR, NOT operations on boolean arrays with bitmask optimisation
//! - **Set membership**: IN and NOT IN operations with hash-based lookup optimisation
//! - **Range operations**: BETWEEN predicates for numeric and string data types
//! - **Pattern matching**: String pattern matching with optimised prefix/suffix detection
//! - **Null-aware logic**: Three-valued logic implementation following SQL semantics
//! - **Compound predicates**: Efficient evaluation of complex multi-condition expressions

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::collections::HashSet;
use std::hash::Hash;
use std::marker::PhantomData;
#[cfg(feature = "simd")]
use std::simd::{Mask, Simd, cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};

use minarrow::kernels::arithmetic::string::MAX_DICT_CHECK;
use minarrow::traits::type_unions::Float;
use minarrow::{
    Array, Bitmask, BooleanAVT, BooleanArray, CategoricalAVT, Integer, MaskedArray, Numeric,
    NumericArray, StringAVT, TextArray, Vec64,
};
#[cfg(feature = "datetime")]
use minarrow::TemporalArray;

#[cfg(not(feature = "simd"))]
use crate::kernels::bitmask::dispatch::{and_masks, or_masks, xor_masks};
use crate::operators::LogicalOperator;
use minarrow::enums::error::KernelError;
#[cfg(feature = "simd")]
use minarrow::kernels::bitmask::simd::{and_masks_simd, or_masks_simd, xor_masks_simd};
use minarrow::utils::confirm_mask_capacity;

#[cfg(feature = "simd")]
use minarrow::utils::is_simd_aligned;
use std::any::TypeId;

/// Builds the Boolean result buffer.
/// `len` - number of rows that will be written.
#[inline(always)]
fn new_bool_buffer(len: usize) -> Bitmask {
    Bitmask::new_set_all(len, false)
}

// Between

macro_rules! impl_between_numeric {
    ($name:ident, $name_to:ident, $ty:ty, $mask_elem:ty, $lanes:expr) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Test if LHS values fall between RHS min/max bounds.
        /// The output Bitmask must have capacity >= lhs.len().
        #[inline(always)]
        pub fn $name_to(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            has_nulls: bool,
            output: &mut Bitmask,
        ) -> Result<(), KernelError> {
            let len = lhs.len();
            if rhs.len() != 2 && rhs.len() != 2 * len {
                return Err(KernelError::InvalidArguments(
                    format!("between: RHS must have len 2 or 2×LHS (got lhs: {}, rhs: {})", len, rhs.len())
                ));
            }

            if let Some(m) = mask {
                if m.capacity() < len {
                    return Err(KernelError::InvalidArguments(
                        format!("between: mask (Bitmask) capacity must be ≥ len (got capacity: {}, len: {})", m.capacity(), len)
                    ));
                }
            }
            assert!(output.capacity() >= len, concat!(stringify!($name_to), ": output capacity too small"));

            // SIMD fast-path
            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    const N: usize = $lanes;
                    type V = Simd<$ty, N>;
                    type M = Mask<$mask_elem, N>;

                    if !has_nulls && rhs.len() == 2 {
                        let min_v = V::splat(rhs[0]);
                        let max_v = V::splat(rhs[1]);

                        let mut i = 0usize;
                        while i + N <= len {
                            let x = V::from_slice(&lhs[i..i + N]);
                            let m: M = x.simd_ge(min_v) & x.simd_le(max_v);
                            let bm = m.to_bitmask();

                            for l in 0..N {
                                if ((bm >> l) & 1) == 1 {
                                    output.set(i + l, true);
                                }
                            }
                            i += N;
                        }
                        // fall back to scalar for tail
                        for j in i..len {
                            if lhs[j] >= rhs[0] && lhs[j] <= rhs[1] {
                                output.set(j, true);
                            }
                        }

                        return Ok(());
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar / null-aware path
            if rhs.len() == 2 {
                let (min, max) = (rhs[0], rhs[1]);
                for i in 0..len {
                    if (!has_nulls || mask.map_or(true, |m| unsafe { m.get_unchecked(i) }))
                        && lhs[i] >= min
                        && lhs[i] <= max
                    {
                        output.set(i, true);
                    }
                }
            } else {
                // per-row min / max
                for i in 0..len {
                    let min = rhs[i * 2];
                    let max = rhs[i * 2 + 1];
                    if (!has_nulls || mask.map_or(true, |m| unsafe { m.get_unchecked(i) }))
                        && lhs[i] >= min
                        && lhs[i] <= max
                    {
                        output.set(i, true);
                    }
                }
            }

            Ok(())
        }

        /// Test if LHS values fall between RHS min/max bounds, producing boolean result array.
        #[inline(always)]
        pub fn $name(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            has_nulls: bool
        ) -> Result<BooleanArray<()>, KernelError> {
            let len = lhs.len();
            let mut out_data = new_bool_buffer(len);
            $name_to(lhs, rhs, mask, has_nulls, &mut out_data)?;
            Ok(BooleanArray {
                data: out_data.into(),
                null_mask: mask.cloned(),
                len,
                _phantom: PhantomData
            })
        }
    };
}

// floats

#[inline(always)]
fn between_generic<T: Numeric + Copy + std::cmp::PartialOrd>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
    has_nulls: bool,
) -> Result<BooleanArray<()>, KernelError> {
    let len = lhs.len();
    let mut out = new_bool_buffer(len);
    let _ = confirm_mask_capacity(len, mask)?;
    if rhs.len() == 2 {
        let (min, max) = (rhs[0], rhs[1]);
        for i in 0..len {
            if (!has_nulls || mask.map_or(true, |m| unsafe { m.get_unchecked(i) }))
                && lhs[i] >= min
                && lhs[i] <= max
            {
                out.set(i, true);
            }
        }
    } else {
        for i in 0..len {
            let min = rhs[i * 2];
            let max = rhs[i * 2 + 1];
            if (!has_nulls || mask.map_or(true, |m| unsafe { m.get_unchecked(i) }))
                && lhs[i] >= min
                && lhs[i] <= max
            {
                out.set(i, true);
            }
        }
    }

    Ok(BooleanArray {
        data: out.into(),
        null_mask: mask.cloned(),
        len,
        _phantom: PhantomData,
    })
}

// In and Not In

macro_rules! impl_in_int {
    ($name:ident, $name_to:ident, $ty:ty, $lanes:expr, $mask_elem:ty) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Test membership of LHS integer values in RHS set.
        /// The output Bitmask must have capacity >= lhs.len().
        #[inline(always)]
        pub fn $name_to(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            has_nulls: bool,
            output: &mut Bitmask,
        ) -> Result<(), KernelError> {
            let len = lhs.len();
            let _ = confirm_mask_capacity(len, mask)?;
            assert!(
                output.capacity() >= len,
                concat!(stringify!($name_to), ": output capacity too small")
            );

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    use crate::utils::bitmask_to_simd_mask;
                    use core::simd::{Mask, Simd};

                    if rhs.len() <= 16 {
                        let mut i = 0;
                        let rhs_simd = rhs;
                        if !has_nulls {
                            while i + $lanes <= len {
                                let x = Simd::<$ty, $lanes>::from_slice(&lhs[i..i + $lanes]);
                                let mut m = Mask::<$mask_elem, $lanes>::splat(false);
                                for &v in rhs_simd {
                                    m |= x.simd_eq(Simd::<$ty, $lanes>::splat(v));
                                }
                                let bm = m.to_bitmask();
                                for l in 0..$lanes {
                                    if ((bm >> l) & 1) == 1 {
                                        output.set(i + l, true);
                                    }
                                }
                                i += $lanes;
                            }
                            for j in i..len {
                                if rhs_simd.contains(&lhs[j]) {
                                    output.set(j, true);
                                }
                            }
                            return Ok(());
                        } else {
                            // ---- SIMD + nulls: use bitmask_to_simd_mask
                            let mb = mask.expect("Bitmask must be Some if has_nulls is set");
                            let mask_bytes = mb.as_bytes();
                            while i + $lanes <= len {
                                let x = Simd::<$ty, $lanes>::from_slice(&lhs[i..i + $lanes]);
                                // valid lanes
                                let lane_mask =
                                    bitmask_to_simd_mask::<$lanes, $mask_elem>(mask_bytes, i, len);
                                let mut in_mask = Mask::<$mask_elem, $lanes>::splat(false);
                                for &v in rhs_simd {
                                    in_mask |= x.simd_eq(Simd::<$ty, $lanes>::splat(v));
                                }
                                // Only set bits for lanes that are both valid and match RHS
                                let valid_in = lane_mask & in_mask;
                                let bm = valid_in.to_bitmask();
                                for l in 0..$lanes {
                                    if ((bm >> l) & 1) == 1 {
                                        output.set(i + l, true);
                                    }
                                }
                                i += $lanes;
                            }
                            for j in i..len {
                                if unsafe { mb.get_unchecked(j) } && rhs_simd.contains(&lhs[j]) {
                                    output.set(j, true);
                                }
                            }
                            return Ok(());
                        }
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback (null-aware and large-RHS)
            let set: std::collections::HashSet<$ty> = rhs.iter().copied().collect();
            for i in 0..len {
                if (!has_nulls || mask.map_or(true, |m| unsafe { m.get_unchecked(i) }))
                    && set.contains(&lhs[i])
                {
                    output.set(i, true);
                }
            }
            Ok(())
        }

        /// Test membership of LHS integer values in RHS set, producing boolean result array.
        #[inline(always)]
        pub fn $name(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            has_nulls: bool,
        ) -> Result<BooleanArray<()>, KernelError> {
            let len = lhs.len();
            let mut out = new_bool_buffer(len);
            $name_to(lhs, rhs, mask, has_nulls, &mut out)?;
            Ok(BooleanArray {
                data: out.into(),
                null_mask: mask.cloned(),
                len,
                _phantom: PhantomData,
            })
        }
    };
}

/// Implements SIMD/Scalar IN kernel for floats, handling NaN semantics and optional null mask.
macro_rules! impl_in_float {
    (
        $fn_name:ident, $fn_name_to:ident, $ty:ty, $lanes:expr, $mask_elem:ty
    ) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Test membership of LHS floating-point values in RHS set with NaN handling.
        /// The output Bitmask must have capacity >= lhs.len().
        #[inline(always)]
        pub fn $fn_name_to(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            has_nulls: bool,
            output: &mut Bitmask,
        ) -> Result<(), KernelError> {
            let len = lhs.len();
            let _ = confirm_mask_capacity(len, mask)?;
            assert!(
                output.capacity() >= len,
                concat!(stringify!($fn_name_to), ": output capacity too small")
            );

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    use crate::utils::bitmask_to_simd_mask;
                    use core::simd::{Mask, Simd};
                    if rhs.len() <= 16 {
                        let mut i = 0;
                        if !has_nulls {
                            while i + $lanes <= len {
                                let x = Simd::<$ty, $lanes>::from_slice(&lhs[i..i + $lanes]);
                                let mut m = Mask::<$mask_elem, $lanes>::splat(false);
                                for &v in rhs {
                                    let vmask = x.simd_eq(Simd::<$ty, $lanes>::splat(v))
                                        | (x.is_nan() & Simd::<$ty, $lanes>::splat(v).is_nan());
                                    m |= vmask;
                                }
                                let bm = m.to_bitmask();
                                for l in 0..$lanes {
                                    if ((bm >> l) & 1) == 1 {
                                        output.set(i + l, true);
                                    }
                                }
                                i += $lanes;
                            }
                            for j in i..len {
                                let x = lhs[j];
                                if rhs.iter().any(|&v| x == v || (x.is_nan() && v.is_nan())) {
                                    output.set(j, true);
                                }
                            }
                            return Ok(());
                        } else {
                            let mb = mask.expect("Bitmask must be Some if nulls are present");
                            let mask_bytes = mb.as_bytes();
                            while i + $lanes <= len {
                                let x = Simd::<$ty, $lanes>::from_slice(&lhs[i..i + $lanes]);
                                let lane_mask =
                                    bitmask_to_simd_mask::<$lanes, $mask_elem>(mask_bytes, i, len);
                                let mut m = Mask::<$mask_elem, $lanes>::splat(false);
                                for &v in rhs {
                                    let vmask = x.simd_eq(Simd::<$ty, $lanes>::splat(v))
                                        | (x.is_nan() & Simd::<$ty, $lanes>::splat(v).is_nan());
                                    m |= vmask;
                                }
                                let m = m & lane_mask;
                                let bm = m.to_bitmask();
                                for l in 0..$lanes {
                                    if ((bm >> l) & 1) == 1 {
                                        output.set(i + l, true);
                                    }
                                }
                                i += $lanes;
                            }
                            for j in i..len {
                                if mask.map_or(true, |m| unsafe { m.get_unchecked(j) }) {
                                    let x = lhs[j];
                                    if rhs.iter().any(|&v| x == v || (x.is_nan() && v.is_nan())) {
                                        output.set(j, true);
                                    }
                                }
                            }
                            return Ok(());
                        }
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback
            for i in 0..len {
                if has_nulls && !mask.map_or(true, |m| unsafe { m.get_unchecked(i) }) {
                    continue;
                }
                let x = lhs[i];
                if rhs.iter().any(|&v| x == v || (x.is_nan() && v.is_nan())) {
                    output.set(i, true);
                }
            }
            Ok(())
        }

        /// Test membership of LHS floating-point values in RHS set with NaN handling.
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            has_nulls: bool,
        ) -> Result<BooleanArray<()>, KernelError> {
            let len = lhs.len();
            let mut out = new_bool_buffer(len);
            $fn_name_to(lhs, rhs, mask, has_nulls, &mut out)?;
            Ok(BooleanArray {
                data: out.into(),
                null_mask: mask.cloned(),
                len,
                _phantom: PhantomData,
            })
        }
    };
}

// Correct MaskElement types per std::simd
#[cfg(feature = "extended_numeric_types")]
impl_in_int!(in_i8, in_i8_to, i8, W8, i8);
#[cfg(feature = "extended_numeric_types")]
impl_in_int!(in_u8, in_u8_to, u8, W8, i8);
#[cfg(feature = "extended_numeric_types")]
impl_in_int!(in_i16, in_i16_to, i16, W16, i16);
#[cfg(feature = "extended_numeric_types")]
impl_in_int!(in_u16, in_u16_to, u16, W16, i16);
impl_in_int!(in_i32, in_i32_to, i32, W32, i32);
impl_in_int!(in_u32, in_u32_to, u32, W32, i32);
impl_in_int!(in_i64, in_i64_to, i64, W64, i64);
impl_in_int!(in_u64, in_u64_to, u64, W64, i64);
impl_in_float!(in_f32, in_f32_to, f32, W32, i32);
impl_in_float!(in_f64, in_f64_to, f64, W64, i64);

#[cfg(feature = "extended_numeric_types")]
impl_between_numeric!(between_i8, between_i8_to, i8, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_between_numeric!(between_u8, between_u8_to, u8, i8, W8);
#[cfg(feature = "extended_numeric_types")]
impl_between_numeric!(between_i16, between_i16_to, i16, i16, W16);
#[cfg(feature = "extended_numeric_types")]
impl_between_numeric!(between_u16, between_u16_to, u16, i16, W16);

impl_between_numeric!(between_i32, between_i32_to, i32, i32, W32);
impl_between_numeric!(between_u32, between_u32_to, u32, i32, W32);
impl_between_numeric!(between_i64, between_i64_to, i64, i64, W64);
impl_between_numeric!(between_u64, between_u64_to, u64, i64, W64);
impl_between_numeric!(between_f32, between_f32_to, f32, i32, W32);
impl_between_numeric!(between_f64, between_f64_to, f64, i64, W64);

// String and dictionary

/// Test if LHS string values fall lexicographically between RHS min/max bounds.
#[inline(always)]
pub fn cmp_str_between<'a, T: Integer>(
    lhs: StringAVT<'a, T>,
    rhs: StringAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;

    if rlen < 2 {
        return Err(KernelError::InvalidArguments(format!(
            "str_between: RHS must contain at least two values (got {})",
            rlen
        )));
    }
    let min = rarr.get(roff).unwrap_or("");
    let max = rarr.get(roff + 1).unwrap_or("");
    let mask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let _ = confirm_mask_capacity(llen, mask.as_ref())?;

    let mut out = new_bool_buffer(llen);

    for i in 0..llen {
        if mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) })
        {
            let s = unsafe { larr.get_str_unchecked(loff + i) };
            if s >= min && s <= max {
                unsafe { out.set_unchecked(i, true) };
            }
        }
    }

    Ok(BooleanArray {
        data: out.into(),
        null_mask: mask,
        len: llen,
        _phantom: PhantomData,
    })
}

#[inline(always)]
/// Test membership of LHS string values in RHS string set.
pub fn cmp_str_in<'a, T: Integer>(
    lhs: StringAVT<'a, T>,
    rhs: StringAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;

    let set: HashSet<&str> = (0..rlen)
        .map(|i| unsafe { rarr.get_str_unchecked(roff + i) })
        .collect();

    let mask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let _ = confirm_mask_capacity(llen, mask.as_ref())?;

    let mut out = new_bool_buffer(llen);

    for i in 0..llen {
        if mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) })
        {
            let s = unsafe { larr.get_str_unchecked(loff + i) };
            if set.contains(s) {
                unsafe { out.set_unchecked(i, true) };
            }
        }
    }
    Ok(BooleanArray {
        data: out.into(),
        null_mask: mask,
        len: llen,
        _phantom: PhantomData,
    })
}

// Public functions

/// Test if values fall between min/max bounds for comparable numeric types.
pub fn cmp_between<T: PartialOrd + Copy + Numeric>(
    lhs: &[T],
    rhs: &[T],
) -> Result<BooleanArray<()>, KernelError> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        return between_i32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        return between_u32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        return between_i64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        return between_u64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // Fallback - floats or any other PartialOrd type
    between_generic(lhs, rhs, None, false)
}

/// Mask-aware variant
#[inline(always)]
pub fn cmp_between_mask<T: PartialOrd + Copy + Numeric>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
) -> Result<BooleanArray<()>, KernelError> {
    let has_nulls = mask.is_some();
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        return between_i32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            has_nulls,
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        return between_u32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            has_nulls,
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        return between_i64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            has_nulls,
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        return between_u64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            has_nulls,
        );
    }
    between_generic(lhs, rhs, mask, has_nulls)
}

// In and Not In

/// Test membership in set for hashable types using hash-based lookup.
pub fn cmp_in<T: Eq + Hash + Copy + 'static>(
    lhs: &[T],
    rhs: &[T],
) -> Result<BooleanArray<()>, KernelError> {
    // i32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        return in_i32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // u32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        return in_u32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // i64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        return in_i64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // u64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        return in_u64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // i16
    #[cfg(feature = "extended_numeric_types")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i16>() {
        return in_i16(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // u16
    #[cfg(feature = "extended_numeric_types")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u16>() {
        return in_u16(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // i8
    #[cfg(feature = "extended_numeric_types")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        return in_i8(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    // u8
    #[cfg(feature = "extended_numeric_types")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
        return in_u8(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            None,
            false,
        );
    }
    return Err(KernelError::UnsupportedType(
        "cmp_in: unsupported type for SIMD in".into(),
    ));
}

/// Mask-aware variant
#[inline(always)]
pub fn cmp_in_mask<T: Eq + Hash + Copy + 'static>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
) -> Result<BooleanArray<()>, KernelError> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        return in_i32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            mask.is_some(),
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        return in_u32(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            mask.is_some(),
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        return in_i64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            mask.is_some(),
        );
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        return in_u64(
            unsafe { std::mem::transmute(lhs) },
            unsafe { std::mem::transmute(rhs) },
            mask,
            mask.is_some(),
        );
    }
    return Err(KernelError::UnsupportedType(
        "cmp_in_mask: unsupported type (expected integer type)".into(),
    ));
}

/// SIMD-aware, type-specific dispatch for cmp_in_f_mask and cmp_in_f
#[inline(always)]
pub fn cmp_in_f_mask<T: Float + Copy>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
) -> Result<BooleanArray<()>, KernelError> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let lhs = unsafe { &*(lhs as *const [T] as *const [f32]) };
        let rhs = unsafe { &*(rhs as *const [T] as *const [f32]) };
        in_f32(lhs, rhs, mask, mask.is_some())
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let lhs = unsafe { &*(lhs as *const [T] as *const [f64]) };
        let rhs = unsafe { &*(rhs as *const [T] as *const [f64]) };
        in_f64(lhs, rhs, mask, mask.is_some())
    } else {
        unreachable!("cmp_in_f_mask: Only f32/f64 supported for Float kernels")
    }
}

#[inline(always)]
/// Test membership in set for floating-point types with NaN handling.
pub fn cmp_in_f<T: Float + Copy>(lhs: &[T], rhs: &[T]) -> Result<BooleanArray<()>, KernelError> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let lhs = unsafe { &*(lhs as *const [T] as *const [f32]) };
        let rhs = unsafe { &*(rhs as *const [T] as *const [f32]) };
        in_f32(lhs, rhs, None, false)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let lhs = unsafe { &*(lhs as *const [T] as *const [f64]) };
        let rhs = unsafe { &*(rhs as *const [T] as *const [f64]) };
        in_f64(lhs, rhs, None, false)
    } else {
        unreachable!("cmp_in_f: Only f32/f64 supported for Float kernels")
    }
}

// String and dictionary

/// Test if floating-point values fall between bounds with NaN handling.
pub fn cmp_between_f<T: PartialOrd + Copy + Float + Numeric>(
    lhs: &[T],
    rhs: &[T],
) -> Result<BooleanArray<()>, KernelError> {
    between_generic(lhs, rhs, None, false)
}

/// Test if dictionary/categorical values fall between lexicographic bounds.
pub fn cmp_dict_between<'a, T: Integer>(
    lhs: CategoricalAVT<'a, T>,
    rhs: CategoricalAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, _rlen) = rhs;

    let min = rarr.get(roff).unwrap_or("");
    let max = rarr.get(roff + 1).unwrap_or("");
    let mask = larr.null_mask.as_ref();
    let _ = confirm_mask_capacity(larr.data.len(), mask)?;
    let has_nulls = mask.is_some();

    let mut out = new_bool_buffer(llen);
    for i in 0..llen {
        let li = loff + i;
        if !has_nulls || mask.map_or(true, |m| unsafe { m.get_unchecked(li) }) {
            let s = unsafe { larr.get_str_unchecked(li) };
            if s >= min && s <= max {
                unsafe { out.set_unchecked(i, true) };
            }
        }
    }
    Ok(BooleanArray {
        data: out.into(),
        null_mask: mask.map(|m| m.slice_clone(loff, llen)),
        len: llen,
        _phantom: PhantomData,
    })
}

/// Returns `true` for each row in `lhs` whose string value also appears
/// anywhere in `rhs`, respecting null masks on both sides.
/// Returns `true` for each row in `lhs` whose string value also appears
/// anywhere in `rhs`, respecting null masks on both sides.
pub fn cmp_dict_in<'a, T: Integer + Hash>(
    lhs: CategoricalAVT<'a, T>,
    rhs: CategoricalAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let mask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let _ = confirm_mask_capacity(llen, mask.as_ref())?;

    let mut out = Bitmask::new_set_all(llen, false);

    if (larr.unique_values.len() == rarr.unique_values.len())
        && (larr.unique_values.len() <= MAX_DICT_CHECK)
    {
        let mut same_dict = true;
        for (a, b) in larr.unique_values.iter().zip(rarr.unique_values.iter()) {
            if a != b {
                same_dict = false;
                break;
            }
        }

        if same_dict {
            let rhs_codes: HashSet<T> = rarr.data[roff..roff + rlen].iter().copied().collect();
            for i in 0..llen {
                if mask
                    .as_ref()
                    .map_or(true, |m| unsafe { m.get_unchecked(i) })
                {
                    let code = larr.data[loff + i];
                    if rhs_codes.contains(&code) {
                        unsafe { out.set_unchecked(i, true) };
                    }
                }
            }
            return Ok(BooleanArray {
                data: out.into(),
                null_mask: mask,
                len: llen,
                _phantom: PhantomData,
            });
        }
    }

    let rhs_strings: HashSet<&str> = (0..rlen)
        .filter(|&i| {
            rarr.null_mask
                .as_ref()
                .map_or(true, |m| unsafe { m.get_unchecked(roff + i) })
        })
        .map(|i| unsafe { rarr.get_str_unchecked(roff + i) })
        .collect();

    for i in 0..llen {
        if mask
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) })
        {
            let s = unsafe { larr.get_str_unchecked(loff + i) };
            if rhs_strings.contains(s) {
                unsafe { out.set_unchecked(i, true) };
            }
        }
    }

    Ok(BooleanArray {
        data: out.into(),
        null_mask: mask,
        len: llen,
        _phantom: PhantomData,
    })
}

// Is Null and Not null predicates

/// Generate boolean mask indicating null elements in any array type.
pub fn is_null_array(arr: &Array) -> Result<BooleanArray<()>, KernelError> {
    let not_null = is_not_null_array(arr)?;
    Ok(!not_null)
}
/// Generate boolean mask indicating non-null elements in any array type.
pub fn is_not_null_array(arr: &Array) -> Result<BooleanArray<()>, KernelError> {
    let len = arr.len();
    let mut data = Bitmask::new_set_all(len, false);

    if let Some(mask) = arr.null_mask() {
        data = mask.clone();
    } else {
        data.fill(true);
    }
    Ok(BooleanArray {
        data,
        null_mask: None,
        len,
        _phantom: PhantomData,
    })
}

// Array in, between , not in
/// Test membership of array elements in values set, dispatching by array type.
pub fn in_array(input: &Array, values: &Array) -> Result<BooleanArray<()>, KernelError> {
    match (input, values) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (
            Array::NumericArray(NumericArray::Int64(a)),
            Array::NumericArray(NumericArray::Int64(b)),
        ) => cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (
            Array::NumericArray(NumericArray::UInt32(a)),
            Array::NumericArray(NumericArray::UInt32(b)),
        ) => cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (
            Array::NumericArray(NumericArray::UInt64(a)),
            Array::NumericArray(NumericArray::UInt64(b)),
        ) => cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (
            Array::NumericArray(NumericArray::Float32(a)),
            Array::NumericArray(NumericArray::Float32(b)),
        ) => cmp_in_f_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (
            Array::NumericArray(NumericArray::Float64(a)),
            Array::NumericArray(NumericArray::Float64(b)),
        ) => cmp_in_f_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (Array::TextArray(TextArray::String32(a)), Array::TextArray(TextArray::String32(b))) => {
            cmp_str_in((**a).tuple_ref(0, a.len()), (**b).tuple_ref(0, b.len()))
        }
        (Array::BooleanArray(a), Array::BooleanArray(b)) => {
            cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref())
        }
        #[cfg(feature = "default_categorical_8")]
        (
            Array::TextArray(TextArray::Categorical8(a)),
            Array::TextArray(TextArray::Categorical8(b)),
        ) => cmp_dict_in((**a).tuple_ref(0, a.len()), (**b).tuple_ref(0, b.len())),
        #[cfg(any(not(feature = "default_categorical_8"), feature = "extended_categorical"))]
        (
            Array::TextArray(TextArray::Categorical32(a)),
            Array::TextArray(TextArray::Categorical32(b)),
        ) => cmp_dict_in((**a).tuple_ref(0, a.len()), (**b).tuple_ref(0, b.len())),
        #[cfg(feature = "large_string")]
        (Array::TextArray(TextArray::String64(a)), Array::TextArray(TextArray::String64(b))) => {
            cmp_str_in((**a).tuple_ref(0, a.len()), (**b).tuple_ref(0, b.len()))
        }
        #[cfg(feature = "datetime")]
        (
            Array::TemporalArray(TemporalArray::Datetime32(a)),
            Array::TemporalArray(TemporalArray::Datetime32(b)),
        ) => cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref()),
        #[cfg(feature = "datetime")]
        (
            Array::TemporalArray(TemporalArray::Datetime64(a)),
            Array::TemporalArray(TemporalArray::Datetime64(b)),
        ) => cmp_in_mask(&a.data, &b.data, a.null_mask.as_ref()),
        (input, values) => Err(KernelError::TypeMismatch(format!(
            "in_array: unsupported type combination ({:?}, {:?})",
            std::mem::discriminant(input), std::mem::discriminant(values),
        ))),
    }
}

/// Test non-membership of array elements in values set, dispatching by array type.
#[inline(always)]
pub fn not_in_array(input: &Array, values: &Array) -> Result<BooleanArray<()>, KernelError> {
    let result = in_array(input, values)?;
    Ok(!result)
}

/// Test if array elements fall between min/max bounds, dispatching by array type.
pub fn between_array(input: &Array, min: &Array, max: &Array) -> Result<Array, KernelError> {
    macro_rules! between_case {
        ($variant:ident, $cmp:ident) => {{
            let arr = match input {
                Array::NumericArray(NumericArray::$variant(arr)) => arr,
                _ => unreachable!(),
            };
            let mins = match min {
                Array::NumericArray(NumericArray::$variant(arr)) => arr,
                _ => unreachable!(),
            };
            let maxs = match max {
                Array::NumericArray(NumericArray::$variant(arr)) => arr,
                _ => unreachable!(),
            };
            let rhs: Vec64<_> = mins
                .data
                .iter()
                .zip(&maxs.data)
                .flat_map(|(&lo, &hi)| [lo, hi])
                .collect();
            Ok(Array::BooleanArray(
                $cmp(
                    &arr.data,
                    &rhs,
                    arr.null_mask.as_ref(),
                    arr.null_mask.is_some(),
                )?
                .into(),
            ))
        }};
    }

    match (input, min, max) {
        (
            Array::NumericArray(NumericArray::Int32(..)),
            Array::NumericArray(NumericArray::Int32(..)),
            Array::NumericArray(NumericArray::Int32(..)),
        ) => between_case!(Int32, between_i32),
        (
            Array::NumericArray(NumericArray::Int64(..)),
            Array::NumericArray(NumericArray::Int64(..)),
            Array::NumericArray(NumericArray::Int64(..)),
        ) => between_case!(Int64, between_i64),
        (
            Array::NumericArray(NumericArray::UInt32(..)),
            Array::NumericArray(NumericArray::UInt32(..)),
            Array::NumericArray(NumericArray::UInt32(..)),
        ) => between_case!(UInt32, between_u32),
        (
            Array::NumericArray(NumericArray::UInt64(..)),
            Array::NumericArray(NumericArray::UInt64(..)),
            Array::NumericArray(NumericArray::UInt64(..)),
        ) => between_case!(UInt64, between_u64),
        (
            Array::NumericArray(NumericArray::Float32(..)),
            Array::NumericArray(NumericArray::Float32(..)),
            Array::NumericArray(NumericArray::Float32(..)),
        ) => between_case!(Float32, between_generic),
        (
            Array::NumericArray(NumericArray::Float64(..)),
            Array::NumericArray(NumericArray::Float64(..)),
            Array::NumericArray(NumericArray::Float64(..)),
        ) => between_case!(Float64, between_generic),
        _ => Err(KernelError::UnsupportedType(
            "Unsupported Type Error.".to_string(),
        )),
    }
}

/// Bitwise NOT of a bit-packed boolean mask window.
/// Offset is a bit offset; len is in bits.
/// Requires offset % 64 == 0 for word-level SIMD processing.
#[inline]
pub fn not_bool<const LANES: usize>(
    src: BooleanAVT<'_, ()>,
) -> Result<BooleanArray<()>, KernelError>
where
{
    let (arr, offset, len) = src;

    if offset % 64 != 0 {
        return Err(KernelError::InvalidArguments(format!(
            "not_bool: offset must be 64-bit aligned (got offset={})",
            offset
        )));
    }

    let null_mask = arr.null_mask.as_ref().map(|nm| nm.slice_clone(offset, len));

    let data = {
        #[cfg(feature = "simd")]
        {
            minarrow::kernels::bitmask::simd::not_mask_simd::<LANES>((&arr.data, offset, len))
        }
        #[cfg(not(feature = "simd"))]
        {
            minarrow::kernels::bitmask::std::not_mask((&arr.data, offset, len))
        }
    };

    Ok(BooleanArray {
        data,
        null_mask,
        len,
        _phantom: core::marker::PhantomData,
    })
}

/// Logical AND/OR/XOR of two bit-packed boolean masks over a window.
/// Offsets are bit offsets. Length is in bits.
/// Panics if offsets are not 64-bit aligned.
pub fn apply_logical_bool<const LANES: usize>(
    lhs: BooleanAVT<'_, ()>,
    rhs: BooleanAVT<'_, ()>,
    op: LogicalOperator,
) -> Result<BooleanArray<()>, KernelError>
where
{
    let (lhs_arr, lhs_off, len) = lhs;
    let (rhs_arr, rhs_off, rlen) = rhs;

    if len != rlen {
        return Err(KernelError::LengthMismatch(format!(
            "logical_bool: window length mismatch (lhs: {}, rhs: {})",
            len, rlen
        )));
    }
    if lhs_off % 64 != 0 || rhs_off % 64 != 0 {
        return Err(KernelError::InvalidArguments(format!(
            "logical_bool: offsets must be 64-bit aligned (lhs: {}, rhs: {})",
            lhs_off, rhs_off
        )));
    }

    // Apply bitmask kernel for the logical operation.

    #[cfg(feature = "simd")]
    let data = match op {
        LogicalOperator::And => {
            and_masks_simd::<LANES>((&lhs_arr.data, lhs_off, len), (&rhs_arr.data, rhs_off, len))
        }
        LogicalOperator::Or => {
            or_masks_simd::<LANES>((&lhs_arr.data, lhs_off, len), (&rhs_arr.data, rhs_off, len))
        }
        LogicalOperator::Xor => {
            xor_masks_simd::<LANES>((&lhs_arr.data, lhs_off, len), (&rhs_arr.data, rhs_off, len))
        }
    };

    // Merge validity (null) masks using AND
    #[cfg(feature = "simd")]
    let null_mask = match (lhs_arr.null_mask.as_ref(), rhs_arr.null_mask.as_ref()) {
        (None, None) => None,
        (Some(a), None) | (None, Some(a)) => Some(a.slice_clone(lhs_off, len)),
        (Some(a), Some(b)) => Some(and_masks_simd::<LANES>(
            (a, lhs_off, len),
            (b, rhs_off, len),
        )),
    };

    #[cfg(not(feature = "simd"))]
    let data = match op {
        LogicalOperator::And => {
            and_masks((&lhs_arr.data, lhs_off, len), (&rhs_arr.data, rhs_off, len))
        }
        LogicalOperator::Or => {
            or_masks((&lhs_arr.data, lhs_off, len), (&rhs_arr.data, rhs_off, len))
        }
        LogicalOperator::Xor => {
            xor_masks((&lhs_arr.data, lhs_off, len), (&rhs_arr.data, rhs_off, len))
        }
    };

    #[cfg(not(feature = "simd"))]
    let null_mask = match (lhs_arr.null_mask.as_ref(), rhs_arr.null_mask.as_ref()) {
        (None, None) => None,
        (Some(a), None) | (None, Some(a)) => Some(a.slice_clone(lhs_off, len)),
        (Some(a), Some(b)) => Some(and_masks((a, lhs_off, len), (b, rhs_off, len))),
    };

    Ok(BooleanArray {
        data,
        null_mask,
        len,
        _phantom: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use minarrow::structs::variants::categorical::CategoricalArray;
    use minarrow::structs::variants::float::FloatArray;
    use minarrow::structs::variants::integer::IntegerArray;
    use minarrow::structs::variants::string::StringArray;
    use minarrow::{Array, Bitmask, BooleanArray, vec64};

    use super::*;

    // --- helpers ---

    fn bm(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            m.set(i, b);
        }
        m
    }

    fn assert_bool(arr: &BooleanArray<()>, expect: &[bool], expect_mask: Option<&[bool]>) {
        assert_eq!(arr.len, expect.len(), "length mismatch");
        for i in 0..expect.len() {
            assert_eq!(arr.data.get(i), expect[i], "val @ {i}");
        }
        match (expect_mask, &arr.null_mask) {
            (None, None) => {}
            (Some(exp), Some(mask)) => {
                for (i, &b) in exp.iter().enumerate() {
                    assert_eq!(mask.get(i), b, "mask @ {i}");
                }
            }
            (None, Some(mask)) => {
                // all mask bits should be true
                for i in 0..arr.len {
                    assert!(mask.get(i), "unexpected false mask @ {i}");
                }
            }
            (Some(_), None) => panic!("expected null mask"),
        }
    }

    fn i32_arr(data: &[i32]) -> IntegerArray<i32> {
        IntegerArray::from_slice(data)
    }
    fn f32_arr(data: &[f32]) -> FloatArray<f32> {
        FloatArray::from_slice(data)
    }
    fn str_arr<T: Integer>(vals: &[&str]) -> StringArray<T> {
        StringArray::<T>::from_slice(vals)
    }
    fn dict_arr<T: Integer>(vals: &[&str]) -> CategoricalArray<T> {
        let owned: Vec<&str> = vals.to_vec();
        CategoricalArray::<T>::from_values(owned)
    }
    //  BETWEEN

    #[test]
    fn between_i32_scalar_rhs() {
        let lhs = vec64![1, 3, 5, 7];
        let rhs = vec64![2, 6];
        let out = between_i32(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, true, false], None);
    }

    #[test]
    fn between_i32_per_row_rhs() {
        let lhs = vec64![5, 9, 2, 8];
        let rhs = vec64![0, 10, 0, 4, 2, 2, 8, 9]; // min/max for each row
        let out = between_i32(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[true, false, true, true], None);
    }

    #[test]
    fn between_i32_nulls_propagate() {
        let lhs = vec64![5, 9, 2, 8];
        let rhs = vec64![0, 10, 0, 4, 2, 2, 8, 9];
        let mask = bm(&[true, false, true, true]);
        let out = between_i32(&lhs, &rhs, Some(&mask), true).unwrap();
        assert_bool(
            &out,
            &[true, false, true, true],
            Some(&[true, false, true, true]),
        );
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn between_i16_works() {
        let lhs = vec64![10i16, 12, 99];
        let rhs = vec64![10i16, 12];
        let out = in_i16(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[true, true, false], None);
    }

    #[test]
    fn between_f64_scalar_and_nulls() {
        let lhs = vec64![1.0, 5.0, 8.0, 20.0];
        let rhs = vec64![4.0, 10.0];
        let mask = bm(&[true, false, true, true]);
        let out = between_f64(&lhs, &rhs, Some(&mask), true).unwrap();
        assert_bool(
            &out,
            &[false, false, true, false],
            Some(&[true, false, true, true]),
        );
    }

    #[test]
    fn between_f32_generic_dispatch() {
        let lhs = vec64![0.1f32, 0.5, 1.2, -1.0];
        let rhs = vec64![0.0, 1.0];
        let out = cmp_between(&lhs, &rhs).unwrap();
        assert_bool(&out, &[true, true, false, false], None);
    }

    #[test]
    fn between_masked_dispatch() {
        let lhs = vec64![1i32, 2, 3];
        let rhs = vec64![0, 2];
        let mask = bm(&[true, false, true]);
        let out = cmp_between_mask(&lhs, &rhs, Some(&mask)).unwrap();
        assert_bool(&out, &[true, false, false], Some(&[true, false, true]));
    }

    // IN

    #[test]
    fn in_i32_small_rhs() {
        let lhs = vec64![1, 2, 3, 4, 5];
        let rhs = vec64![2, 4];
        let out = in_i32(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, false, true, false], None);
    }

    #[test]
    fn in_i32_with_nulls() {
        let lhs = vec64![7, 8, 9];
        let rhs = vec64![8];
        let mask = bm(&[true, false, true]);
        let out = in_i32(&lhs, &rhs, Some(&mask), true).unwrap();
        assert_bool(&out, &[false, false, false], Some(&[true, false, true]));
    }

    #[test]
    fn in_i64_large_rhs() {
        let lhs = vec64![1i64, 2, 3, 7, 8, 15];
        let rhs: Vec<i64> = (2..10).collect();
        let out = in_i64(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, true, true, true, false], None);
    }

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn in_u8_small_rhs() {
        let lhs = vec64![1u8, 2, 3, 4];
        let rhs = vec64![2u8, 3];
        let out = in_u8(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, true, false], None);
    }

    #[test]
    fn in_float_nan_and_normal() {
        let lhs = vec64![1.0f32, f32::NAN, 7.0];
        let rhs = vec64![f32::NAN, 7.0];
        let out = in_f32(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, true], None);
    }

    // BETWEEN / IN

    #[test]
    fn string_between() {
        let lhs = str_arr::<u32>(&["aa", "bb", "zz"]);
        let rhs = str_arr::<u32>(&["b", "y"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_str_between(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn string_between_chunk() {
        let lhs = str_arr::<u32>(&["0", "aa", "bb", "zz", "9"]);
        let rhs = str_arr::<u32>(&["a", "b", "y", "z"]);
        // Windowed: skip first/last for lhs; use a window for rhs
        let lhs_slice = (&lhs, 1, 3); // ["aa", "bb", "zz"]
        let rhs_slice = (&rhs, 1, 2); // ["b", "y"]
        let out = cmp_str_between(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn string_in_basic() {
        let lhs = str_arr::<u32>(&["x", "y", "z"]);
        let rhs = str_arr::<u32>(&["y", "a"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_str_in(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn string_in_basic_chunk() {
        let lhs = str_arr::<u32>(&["0", "x", "y", "z", "9"]);
        let rhs = str_arr::<u32>(&["b", "y", "a", "c"]);
        let lhs_slice = (&lhs, 1, 3); // ["x", "y", "z"]
        let rhs_slice = (&rhs, 1, 2); // ["y", "a"]
        let out = cmp_str_in(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn dict_between() {
        let lhs = dict_arr::<u32>(&["cat", "dog", "emu"]);
        let rhs = dict_arr::<u32>(&["cobra", "dove"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_dict_between(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn dict_between_chunk() {
        let lhs = dict_arr::<u32>(&["a", "cat", "dog", "emu", "z"]);
        let rhs = dict_arr::<u32>(&["a", "cobra", "dove", "zz"]);
        let lhs_slice = (&lhs, 1, 3); // ["cat", "dog", "emu"]
        let rhs_slice = (&rhs, 1, 2); // ["cobra", "dove"]
        let out = cmp_dict_between(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn dict_in_membership() {
        let lhs = dict_arr::<u32>(&["aa", "bb", "cc"]);
        let rhs = dict_arr::<u32>(&["bb", "dd"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_dict_in(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn dict_in_membership_chunk() {
        let lhs = dict_arr::<u32>(&["0", "aa", "bb", "cc", "9"]);
        let rhs = dict_arr::<u32>(&["a", "bb", "dd", "zz"]);
        let lhs_slice = (&lhs, 1, 3); // ["aa", "bb", "cc"]
        let rhs_slice = (&rhs, 1, 2); // ["bb", "dd"]
        let out = cmp_dict_in(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[test]
    fn string_between_nulls() {
        let mut lhs = str_arr::<u32>(&["foo", "bar", "baz"]);
        lhs.null_mask = Some(bm(&[true, false, true]));
        let rhs = str_arr::<u32>(&["a", "zzz"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_str_between(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[true, false, true], Some(&[true, false, true]));
    }

    #[test]
    fn string_between_nulls_chunk() {
        let mut lhs = str_arr::<u32>(&["0", "foo", "bar", "baz", "z"]);
        lhs.null_mask = Some(bm(&[true, true, false, true, true]));
        let rhs = str_arr::<u32>(&["0", "a", "zzz", "9"]);
        let lhs_slice = (&lhs, 1, 3); // ["foo", "bar", "baz"]
        let rhs_slice = (&rhs, 1, 2); // ["a", "zzz"]
        let out = cmp_str_between(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[true, false, true], Some(&[true, false, true]));
    }

    #[test]
    fn dict_in_nulls() {
        let mut lhs = dict_arr::<u32>(&["one", "two", "three"]);
        lhs.null_mask = Some(bm(&[false, true, true]));
        let rhs = dict_arr::<u32>(&["two", "four"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_dict_in(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], Some(&[false, true, true]));
    }

    #[test]
    fn dict_in_nulls_chunk() {
        let mut lhs = dict_arr::<u32>(&["x", "one", "two", "three", "z"]);
        lhs.null_mask = Some(bm(&[true, false, true, true, true]));
        let rhs = dict_arr::<u32>(&["a", "two", "four", "b"]);
        let lhs_slice = (&lhs, 1, 3); // ["one", "two", "three"]
        let rhs_slice = (&rhs, 1, 2); // ["two", "four"]
        let out = cmp_dict_in(lhs_slice, rhs_slice).unwrap();
        assert_bool(&out, &[false, true, false], Some(&[false, true, true]));
    }

    // Boolean/Null

    #[test]
    fn is_null_and_is_not_null() {
        let mut arr = i32_arr(&[1, 2, 0]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let array = Array::from_int32(arr.clone());

        let not_null = is_not_null_array(&array).unwrap();
        let is_null = is_null_array(&array).unwrap();

        assert_bool(&not_null, &[true, false, true], None);
        assert_bool(&is_null, &[false, true, false], None);
    }

    #[test]
    fn is_null_not_null_dense() {
        let arr = i32_arr(&[1, 2, 3]);
        let array = Array::from_int32(arr.clone());
        let is_null = is_null_array(&array).unwrap();
        assert_bool(&is_null, &[false, false, false], None);
        let not_null = is_not_null_array(&array).unwrap();
        assert_bool(&not_null, &[true, true, true], None);
    }

    //  Array dispatch in_array, not_in_array, between_array ----

    #[test]
    fn in_array_int32_dispatch() {
        let inp = Array::from_int32(i32_arr(&[10, 20, 30]));
        let vals = Array::from_int32(i32_arr(&[20, 40]));
        let out = in_array(&inp, &vals).unwrap();
        assert_bool(&out, &[false, true, false], None);

        let out_not = not_in_array(&inp, &vals).unwrap();
        assert_bool(&out_not, &[true, false, true], None);
    }

    #[test]
    fn in_array_f32_dispatch() {
        let inp = Array::from_float32(f32_arr(&[1.0, f32::NAN, 7.0]));
        let vals = Array::from_float32(f32_arr(&[f32::NAN, 7.0]));
        let out = in_array(&inp, &vals).unwrap();
        assert_bool(&out, &[false, true, true], None);
    }

    #[test]
    fn in_array_string_dispatch() {
        let inp = Array::from_string32(str_arr::<u32>(&["a", "b", "c"]));
        let vals = Array::from_string32(str_arr::<u32>(&["b", "d"]));
        let out = in_array(&inp, &vals).unwrap();
        assert_bool(&out, &[false, true, false], None);
    }

    #[cfg(feature = "default_categorical_8")]
    #[test]
    fn in_array_dictionary_dispatch() {
        let inp = Array::from_categorical8(dict_arr::<u8>(&["aa", "bb", "cc"]));
        let vals = Array::from_categorical8(dict_arr::<u8>(&["bb", "cc"]));
        let out = in_array(&inp, &vals).unwrap();
        assert_bool(&out, &[false, true, true], None);
    }

    #[cfg(not(feature = "default_categorical_8"))]
    #[test]
    fn in_array_dictionary_dispatch() {
        let inp = Array::from_categorical32(dict_arr::<u32>(&["aa", "bb", "cc"]));
        let vals = Array::from_categorical32(dict_arr::<u32>(&["bb", "cc"]));
        let out = in_array(&inp, &vals).unwrap();
        assert_bool(&out, &[false, true, true], None);
    }

    #[test]
    fn between_array_int32_rows() {
        let inp = Array::from_int32(i32_arr(&[5, 15, 25]));
        let min = Array::from_int32(i32_arr(&[0, 10, 20]));
        let max = Array::from_int32(i32_arr(&[10, 20, 30]));

        let out = between_array(&inp, &min, &max).unwrap();
        match out {
            Array::BooleanArray(b) => assert_bool(&b, &[true, true, true], None),
            _ => panic!("expected Bool array"),
        }
    }

    #[test]
    fn between_array_float_generic() {
        let inp = Array::from_float32(f32_arr(&[0.5, 1.5, 2.5]));
        let min = Array::from_float32(f32_arr(&[0.0, 1.0, 2.0]));
        let max = Array::from_float32(f32_arr(&[1.0, 2.0, 3.0]));

        let out = between_array(&inp, &min, &max).unwrap();
        match out {
            Array::BooleanArray(b) => assert_bool(&b, &[true, true, true], None),
            _ => panic!("expected Bool"),
        }
    }

    #[test]
    fn between_array_type_mismatch() {
        let inp = Array::from_int32(i32_arr(&[1, 2, 3]));
        let min = Array::from_float32(f32_arr(&[0.0, 0.0, 0.0]));
        let max = Array::from_float32(f32_arr(&[5.0, 5.0, 5.0]));
        let err = between_array(&inp, &min, &max).unwrap_err();
        match err {
            KernelError::UnsupportedType(_) => {}
            _ => panic!("Expected UnsupportedType error"),
        }
    }

    // all integer types, short and long

    #[test]
    fn in_integers_various_types() {
        #[cfg(feature = "extended_numeric_types")]
        {
            let u8_lhs = vec64![1u8, 2, 3, 5];
            let u8_rhs = vec64![3u8, 5, 8];
            let out = in_u8(&u8_lhs, &u8_rhs, None, false).unwrap();
            assert_bool(&out, &[false, false, true, true], None);

            let u16_lhs = vec64![100u16, 200, 300];
            let u16_rhs = vec64![200u16, 500];
            let out = in_u16(&u16_lhs, &u16_rhs, None, false).unwrap();
            assert_bool(&out, &[false, true, false], None);

            let i16_lhs = vec64![10i16, 15, 42];
            let i16_rhs = vec64![15i16, 42, 77];
            let out = in_i16(&i16_lhs, &i16_rhs, None, false).unwrap();
            assert_bool(&out, &[false, true, true], None);
        }

        let u32_lhs = vec64![0u32, 1, 2, 9];
        let u32_rhs = vec64![9u32, 1];
        let out = in_u32(&u32_lhs, &u32_rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, false, true], None);

        let i64_lhs = vec64![1i64, 9, 10];
        let i64_rhs = vec64![2i64, 10, 20];
        let out = in_i64(&i64_lhs, &i64_rhs, None, false).unwrap();
        assert_bool(&out, &[false, false, true], None);

        let u64_lhs = vec64![1u64, 2, 3, 4];
        let u64_rhs = vec64![2u64, 4, 8];
        let out = in_u64(&u64_lhs, &u64_rhs, None, false).unwrap();
        assert_bool(&out, &[false, true, false, true], None);
    }

    // empty input edge

    #[test]
    fn between_and_in_empty_inputs() {
        // Between, scalar rhs (for numeric arrays, no slice tuple needed)
        let lhs: [i32; 0] = [];
        let rhs = vec64![0, 1];
        let out = between_i32(&lhs, &rhs, None, false).unwrap();
        assert_eq!(out.len, 0);

        // In, any rhs (for numeric arrays, no slice tuple needed)
        let lhs: [i32; 0] = [];
        let rhs = vec64![1, 2, 3];
        let out = in_i32(&lhs, &rhs, None, false).unwrap();
        assert_eq!(out.len, 0);

        // String, in (slice API)
        let lhs = str_arr::<u32>(&[]);
        let rhs = str_arr::<u32>(&["a", "b"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_str_in(lhs_slice, rhs_slice).unwrap();
        assert_eq!(out.len, 0);
    }

    #[test]
    fn between_and_in_empty_inputs_chunk() {
        // Only applies to the string in version
        let lhs = str_arr::<u32>(&["x", "y"]);
        let rhs = str_arr::<u32>(&["a", "b", "c"]);
        let lhs_slice = (&lhs, 1, 0); // zero-length window
        let rhs_slice = (&rhs, 1, 2); // window ["b", "c"]
        let out = cmp_str_in(lhs_slice, rhs_slice).unwrap();
        assert_eq!(out.len, 0);
    }

    #[test]
    fn between_per_row_bounds_on_last_row() {
        // Coverage: last row per-row
        let lhs = vec64![0i32, 10, 20, 30];
        let rhs = vec64![0, 5, 5, 15, 15, 25, 25, 35];
        let out = between_i32(&lhs, &rhs, None, false).unwrap();
        assert_bool(&out, &[true, true, true, true], None);
    }

    #[test]
    fn test_cmp_dict_in_force_fallback() {
        // lhs and rhs have different unique_values lengths
        let mut lhs = dict_arr::<u32>(&["a", "b", "c", "a"]);
        lhs.unique_values = vec64!["a".to_string(), "b".to_string(), "c".to_string()]; // len=3
        let mut rhs = dict_arr::<u32>(&["b", "x", "y", "z"]);
        rhs.unique_values = vec64![
            "b".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string()
        ]; // len=4
        lhs.null_mask = Some(bm(&[true, true, true, true]));
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let out = cmp_dict_in(lhs_slice, rhs_slice).unwrap();
        // should fall back to string-matching: only "b" matches
        assert_bool(
            &out,
            &[false, true, false, false],
            Some(&[true, true, true, true]),
        );
    }

    #[test]
    fn test_cmp_dict_in_force_fallback_chunk() {
        let mut lhs = dict_arr::<u32>(&["z", "a", "b", "c", "a", "q"]);
        lhs.unique_values = vec64![
            "z".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "q".to_string()
        ];
        let mut rhs = dict_arr::<u32>(&["x", "b", "x", "y", "z"]);
        rhs.unique_values = vec64![
            "x".to_string(),
            "b".to_string(),
            "y".to_string(),
            "z".to_string()
        ];
        lhs.null_mask = Some(bm(&[true, true, true, true, true, true]));
        // Window: pick ["a", "b", "c", "a"] and ["b", "x", "y", "z"]
        let lhs_slice = (&lhs, 1, 4);
        let rhs_slice = (&rhs, 1, 4);
        let out = cmp_dict_in(lhs_slice, rhs_slice).unwrap();
        // Only "b" matches (index 1 of window)
        assert_bool(
            &out,
            &[false, true, false, false],
            Some(&[true, true, true, true]),
        );
    }

    #[test]
    fn test_in_array_empty_rhs() {
        let arr = Array::from_int32(i32_arr(&[1, 2, 3]));
        let empty = Array::from_int32(i32_arr(&[]));
        let out = in_array(&arr, &empty).unwrap();
        // must be all false, and mask preserved (no mask => all bits true)
        assert_bool(&out, &[false, false, false], None);
    }
}
