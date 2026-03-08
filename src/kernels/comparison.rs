// Copyright (c) 2025 Peter Bower
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Comparison Operations Kernels Module** - *High-Performance Element-wise Comparison Operations*
//!
//! Optimised comparison kernels providing comprehensive element-wise comparison operations
//! across numeric, string, and categorical data types with SIMD acceleration and null-aware semantics.
//! Foundation for filtering, conditional logic, and analytical query processing.
//!
//! ## Core Operations
//! - **Numeric comparisons**: Equal, not equal, greater than, less than, greater/less than or equal
//! - **String comparisons**: UTF-8 aware lexicographic ordering with efficient prefix matching
//! - **Categorical comparisons**: Dictionary-encoded comparisons avoiding string materialisation
//! - **Null-aware semantics**: Proper three-valued logic handling (true/false/null)
//! - **SIMD vectorisation**: Hardware-accelerated bulk comparison operations
//! - **Bitmask operations**: Efficient boolean result representation using bit manipulation

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

// SIMD
use std::marker::PhantomData;
#[cfg(feature = "simd")]
use std::simd::{Mask, Simd};

use minarrow::{Bitmask, BooleanArray, Integer, Numeric};

#[cfg(not(feature = "simd"))]
use crate::kernels::bitmask::std::{and_masks, in_mask, not_in_mask, not_mask};
use crate::operators::ComparisonOperator;
use minarrow::enums::error::KernelError;
#[cfg(feature = "simd")]
use minarrow::kernels::bitmask::simd::{
    and_masks_simd, in_mask_simd, not_in_mask_simd, not_mask_simd,
};
use minarrow::utils::confirm_equal_len;
#[cfg(feature = "simd")]
use minarrow::utils::is_simd_aligned;
use minarrow::{BitmaskVT, BooleanAVT, CategoricalAVT, StringAVT};

/// Returns a new Bitmask for boolean buffers, all bits cleared (false).
#[inline(always)]
fn new_bool_bitmask(len: usize) -> Bitmask {
    Bitmask::new_set_all(len, false)
}

/// Merge two Bitmasks using bitwise AND, or propagate one if only one is present.
fn merge_bitmasks_to_new(a: Option<&Bitmask>, b: Option<&Bitmask>, len: usize) -> Option<Bitmask> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x.slice_clone(0, len)),
        (Some(x), Some(y)) => {
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                unsafe { out.set_unchecked(i, x.get_unchecked(i) && y.get_unchecked(i)) };
            }
            Some(out)
        }
    }
}

// Int and float

macro_rules! impl_cmp_numeric {
    ($fn_name:ident, $fn_name_to:ident, $ty:ty, $lanes:expr, $mask_elem:ty) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Type-specific SIMD-accelerated comparison function with vectorised operations.
        /// Panics if output capacity < lhs.len().
        #[inline(always)]
        pub fn $fn_name_to(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            op: ComparisonOperator,
            output: &mut Bitmask,
        ) -> Result<(), KernelError> {
            let len = lhs.len();
            confirm_equal_len("compare numeric length mismatch", len, rhs.len())?;
            assert!(
                output.capacity() >= len,
                concat!(stringify!($fn_name_to), ": output capacity too small")
            );
            let has_nulls = mask.is_some();

            #[cfg(feature = "simd")]
            {
                // Check if both arrays are 64-byte aligned for SIMD
                if is_simd_aligned(lhs) && is_simd_aligned(rhs) {
                    use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
                    const N: usize = $lanes;
                    if !has_nulls {
                        let mut i = 0;
                        while i + N <= len {
                            let a = Simd::<$ty, N>::from_slice(&lhs[i..i + N]);
                            let b = Simd::<$ty, N>::from_slice(&rhs[i..i + N]);
                            let m: Mask<$mask_elem, N> = match op {
                                ComparisonOperator::Equals => a.simd_eq(b),
                                ComparisonOperator::NotEquals => a.simd_ne(b),
                                ComparisonOperator::LessThan => a.simd_lt(b),
                                ComparisonOperator::LessThanOrEqualTo => a.simd_le(b),
                                ComparisonOperator::GreaterThan => a.simd_gt(b),
                                ComparisonOperator::GreaterThanOrEqualTo => a.simd_ge(b),
                                _ => Mask::splat(false),
                            };
                            let bits = m.to_bitmask();
                            for l in 0..N {
                                if ((bits >> l) & 1) == 1 {
                                    unsafe { output.set_unchecked(i + l, true) };
                                }
                            }
                            i += N;
                        }
                        // Tail often caused by `n % LANES != 0`; uses scalar fallback.
                        for j in i..len {
                            let res = match op {
                                ComparisonOperator::Equals => lhs[j] == rhs[j],
                                ComparisonOperator::NotEquals => lhs[j] != rhs[j],
                                ComparisonOperator::LessThan => lhs[j] < rhs[j],
                                ComparisonOperator::LessThanOrEqualTo => lhs[j] <= rhs[j],
                                ComparisonOperator::GreaterThan => lhs[j] > rhs[j],
                                ComparisonOperator::GreaterThanOrEqualTo => lhs[j] >= rhs[j],
                                _ => false,
                            };
                            if res {
                                unsafe { output.set_unchecked(j, true) };
                            }
                        }

                        return Ok(());
                    }
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            for i in 0..len {
                if has_nulls && !mask.map_or(true, |m| unsafe { m.get_unchecked(i) }) {
                    continue;
                }
                let res = match op {
                    ComparisonOperator::Equals => lhs[i] == rhs[i],
                    ComparisonOperator::NotEquals => lhs[i] != rhs[i],
                    ComparisonOperator::LessThan => lhs[i] < rhs[i],
                    ComparisonOperator::LessThanOrEqualTo => lhs[i] <= rhs[i],
                    ComparisonOperator::GreaterThan => lhs[i] > rhs[i],
                    ComparisonOperator::GreaterThanOrEqualTo => lhs[i] >= rhs[i],
                    _ => false,
                };
                if res {
                    unsafe { output.set_unchecked(i, true) };
                }
            }
            Ok(())
        }

        /// Type-specific SIMD-accelerated comparison function with vectorised operations.
        ///
        /// Specialised comparison implementation optimised for the specific numeric type with
        /// architecture-appropriate lane configuration. Features memory alignment checking,
        /// SIMD vectorisation, and optional null mask support for maximum performance.
        ///
        /// # Parameters
        /// - `lhs`: Left-hand side slice for comparison
        /// - `rhs`: Right-hand side slice for comparison
        /// - `mask`: Optional validity mask applied after comparison
        /// - `op`: Comparison operator to apply
        ///
        /// # Returns
        /// `Result<BooleanArray<()>, KernelError>` containing comparison results.
        ///
        /// # SIMD Optimisations
        /// - Memory alignment: Checks 64-byte alignment for optimal SIMD operations
        /// - Vectorised comparisons: Uses SIMD compare operations for parallel processing
        /// - Scalar fallback: Efficient scalar path for unaligned or remainder elements
        #[inline(always)]
        pub fn $fn_name(
            lhs: &[$ty],
            rhs: &[$ty],
            mask: Option<&Bitmask>,
            op: ComparisonOperator,
        ) -> Result<BooleanArray<()>, KernelError> {
            let len = lhs.len();
            let mut out = new_bool_bitmask(len);
            $fn_name_to(lhs, rhs, mask, op, &mut out)?;
            Ok(BooleanArray {
                data: out.into(),
                null_mask: mask.cloned(),
                len,
                _phantom: PhantomData,
            })
        }
    };
}

/// Zero-allocation variant: writes directly to caller's output buffer.
///
/// Unified numeric comparison dispatch with optional null mask support.
/// The output Bitmask must have capacity >= lhs.len().
#[inline(always)]
pub fn cmp_numeric_to<T: Numeric + Copy + 'static>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
    op: ComparisonOperator,
    output: &mut Bitmask,
) -> Result<(), KernelError> {
    macro_rules! dispatch {
        ($t:ty, $f:ident) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$t>() {
                return $f(
                    unsafe { std::mem::transmute(lhs) },
                    unsafe { std::mem::transmute(rhs) },
                    mask,
                    op,
                    output,
                );
            }
        };
    }
    dispatch!(i32, cmp_i32_to);
    dispatch!(i64, cmp_i64_to);
    dispatch!(u32, cmp_u32_to);
    dispatch!(u64, cmp_u64_to);
    dispatch!(f32, cmp_f32_to);
    dispatch!(f64, cmp_f64_to);

    unreachable!("Unsupported numeric type for compare_numeric");
}

/// Unified numeric comparison dispatch with optional null mask support.
///
/// High-performance generic comparison function that dispatches to type-specific SIMD implementations
/// based on runtime type identification. Supports all numeric types with optional null mask filtering
/// and comprehensive error handling for mismatched lengths and unsupported types.
///
/// # Type Parameters
/// - `T`: Numeric type implementing `Numeric + Copy + 'static` (i32, i64, u32, u64, f32, f64)
///
/// # Parameters
/// - `lhs`: Left-hand side numeric slice for comparison
/// - `rhs`: Right-hand side numeric slice for comparison
/// - `mask`: Optional validity mask applied after comparison (AND operation)
/// - `op`: Comparison operator to apply (Equals, NotEquals, LessThan, etc.)
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` containing comparison results or error details.
///
/// # Dispatch Strategy
/// Uses `TypeId` runtime checking to dispatch to optimal type-specific implementations:
/// - `i32`/`u32`: 32-bit integer SIMD kernels with W32 lane configuration
/// - `i64`/`u64`: 64-bit integer SIMD kernels with W64 lane configuration
/// - `f32`/`f64`: IEEE 754 floating-point SIMD kernels with specialised NaN handling
///
/// # Error Conditions
/// - `KernelError::LengthMismatch`: Input slices have different lengths
/// - `KernelError::InvalidArguments`: Unsupported numeric type (unreachable in practice)
///
/// # Performance Benefits
/// - Zero-cost dispatch: Type resolution optimised away at compile time for monomorphic usage
/// - SIMD acceleration: Delegates to vectorised implementations for maximum throughput
/// - Memory efficiency: Optional mask processing avoids unnecessary allocations
///
/// # Example Usage
/// ```rust,ignore
/// use simd_kernels::kernels::comparison::{cmp_numeric, ComparisonOperator};
///
/// let lhs = &[1i32, 2, 3, 4];
/// let rhs = &[1i32, 1, 4, 3];
/// let result = cmp_numeric(lhs, rhs, None, ComparisonOperator::Equals)?;
/// // Result: [true, false, false, false]
/// ```
#[inline(always)]
pub fn cmp_numeric<T: Numeric + Copy + 'static>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError> {
    let len = lhs.len();
    let mut out = new_bool_bitmask(len);
    cmp_numeric_to(lhs, rhs, mask, op, &mut out)?;
    Ok(BooleanArray {
        data: out.into(),
        null_mask: mask.cloned(),
        len,
        _phantom: PhantomData,
    })
}

/// SIMD-accelerated compare bitmask
///
/// Compare two packed bool bitmask slices over a window, using the given operator.
/// The offsets are bit offsets into each mask.
/// The mask, if provided, is ANDed after the comparison.
/// Requires that all offsets are 64-bit aligned (i.e., offset % 64 == 0).
///
/// This lower level kernel can be orchestrated by apply_cmp_bool which
/// wraps it into a BoolWindow with null-aware semantics.
#[cfg(feature = "simd")]
pub fn cmp_bitmask_simd<const LANES: usize>(
    lhs: BitmaskVT<'_>,
    rhs: BitmaskVT<'_>,
    mask: Option<BitmaskVT<'_>>,
    op: ComparisonOperator,
) -> Result<Bitmask, KernelError>
where
{
    // We have some code duplication here with the `std` version,
    // but unifying then means a const LANE generic on the non-simd path,
    // and adding a higher level dispatch layer creates additional indirection
    // and 9 args instead of 4, hence why it's this way.

    confirm_equal_len("compare bool length mismatch", lhs.2, rhs.2)?;
    let (lhs_mask, lhs_offset, len) = lhs;
    let (rhs_mask, rhs_offset, _) = rhs;

    // Handle 'In' and 'NotIn' early

    if matches!(op, ComparisonOperator::In | ComparisonOperator::NotIn) {
        let mut out = match op {
            ComparisonOperator::In => in_mask_simd::<LANES>(lhs, rhs),
            ComparisonOperator::NotIn => not_in_mask_simd::<LANES>(lhs, rhs),
            _ => unreachable!(),
        };
        if let Some(mask_slice) = mask {
            out = and_masks_simd::<LANES>((&out, 0, out.len), mask_slice);
        }
        return Ok(out);
    }

    // Word-aligned offsets
    if lhs_offset % 64 != 0
        || rhs_offset % 64 != 0
        || mask.as_ref().map_or(false, |(_, mo, _)| mo % 64 != 0)
    {
        return Err(KernelError::InvalidArguments(format!(
            "cmp_bitmask: all offsets must be 64-bit aligned (lhs: {}, rhs: {}, mask offset: {:?})",
            lhs_offset,
            rhs_offset,
            mask.as_ref().map(|(_, mo, _)| mo)
        )));
    }

    // Precompute word indices/counts
    let lhs_word_start = lhs_offset / 64;
    let rhs_word_start = rhs_offset / 64;
    let n_words = (len + 63) / 64;

    // Allocate output
    let mut out = Bitmask::new_set_all(len, false);

    type Word = u64;
    let lane_words = LANES;
    let simd_chunks = n_words / lane_words;

    let tail_words = n_words % lane_words;
    let mut word_idx = 0;

    // SIMD main path
    for chunk in 0..simd_chunks {
        let base_lhs = lhs_word_start + chunk * lane_words;
        let base_rhs = rhs_word_start + chunk * lane_words;
        let base_mask = mask
            .as_ref()
            .map(|(m, mask_word_start, _)| (m, mask_word_start + chunk * lane_words));

        let mut lhs_arr = [0u64; LANES];
        let mut rhs_arr = [0u64; LANES];
        let mut mask_arr = [!0u64; LANES];

        for lane in 0..LANES {
            lhs_arr[lane] = unsafe { lhs_mask.word_unchecked(base_lhs + lane) };
            rhs_arr[lane] = unsafe { rhs_mask.word_unchecked(base_rhs + lane) };
            if let Some((m, mask_word_start)) = base_mask {
                mask_arr[lane] = unsafe { m.word_unchecked(mask_word_start + lane) };
            }
        }
        let lhs_v = Simd::<Word, LANES>::from_array(lhs_arr);
        let rhs_v = Simd::<Word, LANES>::from_array(rhs_arr);
        let mask_v = Simd::<Word, LANES>::from_array(mask_arr);

        let cmp_v = match op {
            ComparisonOperator::Equals => !(lhs_v ^ rhs_v),
            ComparisonOperator::NotEquals => lhs_v ^ rhs_v,
            ComparisonOperator::GreaterThan => lhs_v & (!rhs_v),
            ComparisonOperator::LessThan => (!lhs_v) & rhs_v,
            ComparisonOperator::GreaterThanOrEqualTo => lhs_v | (!rhs_v),
            ComparisonOperator::LessThanOrEqualTo => (!lhs_v) | rhs_v,
            _ => Simd::splat(0),
        };
        let result_v = cmp_v & mask_v;

        for lane in 0..LANES {
            unsafe {
                out.set_word_unchecked(word_idx, result_v[lane]);
            }
            word_idx += 1;
        }
    }

    // Tail often caused by `n % LANES != 0`; uses scalar fallback.
    let base_lhs = lhs_word_start + simd_chunks * lane_words;
    let base_rhs = rhs_word_start + simd_chunks * lane_words;
    let base_mask: Option<(&Bitmask, usize)> = mask
        .as_ref()
        .map(|(m, mo, _)| (*m, mo + simd_chunks * lane_words));

    for tail in 0..tail_words {
        let a = unsafe { lhs_mask.word_unchecked(base_lhs + tail) };
        let b = unsafe { rhs_mask.word_unchecked(base_rhs + tail) };
        let m = if let Some((m, mask_word_start)) = base_mask {
            unsafe { m.word_unchecked(mask_word_start + tail) }
        } else {
            !0u64
        };
        let cmp = match op {
            ComparisonOperator::Equals => !(a ^ b),
            ComparisonOperator::NotEquals => a ^ b,
            ComparisonOperator::GreaterThan => a & (!b),
            ComparisonOperator::LessThan => (!a) & b,
            ComparisonOperator::GreaterThanOrEqualTo => a | (!b),
            ComparisonOperator::LessThanOrEqualTo => (!a) | b,
            _ => 0,
        } & m;
        unsafe {
            out.set_word_unchecked(word_idx, cmp);
        }
        word_idx += 1;
    }

    out.mask_trailing_bits();
    Ok(out)
}

/// Performs vectorised boolean array comparisons with null mask handling.
///
/// High-performance SIMD-accelerated comparison function for boolean arrays with automatic null
/// mask merging and operator-specific optimisations. Supports all comparison operators through
/// efficient bitmask operations with configurable lane counts for architecture-specific tuning.
///
/// # Type Parameters
/// - `LANES`: Number of SIMD lanes to process simultaneously
///
/// # Parameters
/// - `lhs`: Left-hand side boolean array view as `(array, offset, length)` tuple
/// - `rhs`: Right-hand side boolean array view as `(array, offset, length)` tuple  
/// - `op`: Comparison operator (Equals, NotEquals, In, NotIn, IsNull, IsNotNull, etc.)
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` containing comparison results with merged null semantics.
pub fn cmp_bool<const LANES: usize>(
    lhs: BooleanAVT<'_, ()>,
    rhs: BooleanAVT<'_, ()>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError>
where
{
    let (lhs_arr, lhs_off, len) = lhs;
    let (rhs_arr, rhs_off, rlen) = rhs;
    debug_assert_eq!(len, rlen, "cmp_bool: window length mismatch");

    #[cfg(feature = "simd")]
    let merged_null_mask: Option<Bitmask> =
        match (lhs_arr.null_mask.as_ref(), rhs_arr.null_mask.as_ref()) {
            (None, None) => None,
            (Some(m), None) | (None, Some(m)) => Some(m.slice_clone(lhs_off, len)),
            (Some(a), Some(b)) => {
                let am = (a, lhs_off, len);
                let bm = (b, rhs_off, len);
                Some(and_masks_simd::<LANES>(am, bm))
            }
        };

    #[cfg(not(feature = "simd"))]
    let merged_null_mask: Option<Bitmask> =
        match (lhs_arr.null_mask.as_ref(), rhs_arr.null_mask.as_ref()) {
            (None, None) => None,
            (Some(m), None) | (None, Some(m)) => Some(m.slice_clone(lhs_off, len)),
            (Some(a), Some(b)) => {
                let am = (a, lhs_off, len);
                let bm = (b, rhs_off, len);
                Some(and_masks(am, bm))
            }
        };

    let mask_slice = merged_null_mask.as_ref().map(|m| (m, 0, len));

    let data = match op {
        ComparisonOperator::Equals
        | ComparisonOperator::NotEquals
        | ComparisonOperator::LessThan
        | ComparisonOperator::LessThanOrEqualTo
        | ComparisonOperator::GreaterThan
        | ComparisonOperator::GreaterThanOrEqualTo
        | ComparisonOperator::In
        | ComparisonOperator::NotIn => {
            #[cfg(feature = "simd")]
            let res = cmp_bitmask_simd::<LANES>(
                (&lhs_arr.data, lhs_off, len),
                (&rhs_arr.data, rhs_off, len),
                mask_slice,
                op,
            )?;
            #[cfg(not(feature = "simd"))]
            let res = cmp_bitmask_std(
                (&lhs_arr.data, lhs_off, len),
                (&rhs_arr.data, rhs_off, len),
                mask_slice,
                op,
            )?;
            res
        }
        ComparisonOperator::IsNull => {
            #[cfg(feature = "simd")]
            let data = match merged_null_mask.as_ref() {
                Some(mask) => not_mask_simd::<LANES>((mask, 0, len)),
                None => Bitmask::new_set_all(len, false),
            };
            #[cfg(not(feature = "simd"))]
            let data = match merged_null_mask.as_ref() {
                Some(mask) => not_mask((mask, 0, len)),
                None => Bitmask::new_set_all(len, false),
            };
            return Ok(BooleanArray {
                data,
                null_mask: None,
                len,
                _phantom: PhantomData,
            });
        }
        ComparisonOperator::IsNotNull => {
            let data = match merged_null_mask.as_ref() {
                Some(mask) => mask.slice_clone(0, len),
                None => Bitmask::new_set_all(len, true),
            };
            return Ok(BooleanArray {
                data,
                null_mask: None,
                len,
                _phantom: PhantomData,
            });
        }
        ComparisonOperator::Between => {
            return Err(KernelError::InvalidArguments(
                "Set operations are not defined for Bool arrays".to_owned(),
            ));
        }
    };

    Ok(BooleanArray {
        data,
        null_mask: merged_null_mask,
        len,
        _phantom: PhantomData,
    })
}

/// Compare two packed bool bitmask slices over a window, using the given operator.
/// The offsets are bit offsets into each mask.
/// The mask, if provided, is ANDed after the comparison.
/// Requires that all offsets are 64-bit aligned (i.e., offset % 64 == 0).
///
/// This lower level kernel can be orchestrated by apply_cmp_bool which
/// wraps it into a BoolWindow with null-aware semantics.
#[cfg(not(feature = "simd"))]
pub fn cmp_bitmask_std(
    lhs: BitmaskVT<'_>,
    rhs: BitmaskVT<'_>,
    mask: Option<BitmaskVT<'_>>,
    op: ComparisonOperator,
) -> Result<Bitmask, KernelError> {
    // We have some code duplication here with the `simd` version,
    // but unifying then means a const LANE generic on the non-simd path,
    // and adding a higher level dispatch layer create additional indirection
    // and 9 args instead of 4, hence why it's this way.

    confirm_equal_len("compare bool length mismatch", lhs.2, rhs.2)?;
    let (lhs_mask, lhs_offset, len) = lhs;
    let (rhs_mask, rhs_offset, _) = rhs;

    // Handle 'In' and 'NotIn' early

    if matches!(op, ComparisonOperator::In | ComparisonOperator::NotIn) {
        let mut out = match op {
            ComparisonOperator::In => in_mask(lhs, rhs),
            ComparisonOperator::NotIn => not_in_mask(lhs, rhs),
            _ => unreachable!(),
        };
        if let Some(mask_slice) = mask {
            out = and_masks((&out, 0, out.len), mask_slice);
        }
        return Ok(out);
    }

    // Word-aligned offsets
    if lhs_offset % 64 != 0
        || rhs_offset % 64 != 0
        || mask.as_ref().map_or(false, |(_, mo, _)| mo % 64 != 0)
    {
        return Err(KernelError::InvalidArguments(format!(
            "cmp_bitmask: all offsets must be 64-bit aligned (lhs: {}, rhs: {}, mask offset: {:?})",
            lhs_offset,
            rhs_offset,
            mask.as_ref().map(|(_, mo, _)| mo)
        )));
    }

    // Precompute word indices/counts
    let lhs_word_start = lhs_offset / 64;
    let rhs_word_start = rhs_offset / 64;
    let n_words = (len + 63) / 64;

    // Allocate output
    let mut out = Bitmask::new_set_all(len, false);

    let words = n_words;
    let tail = len % 64;
    let mask_mask_opt = mask;

    // Word-aligned loop
    for w in 0..words {
        let a = unsafe { lhs_mask.word_unchecked(lhs_word_start + w) };
        let b = unsafe { rhs_mask.word_unchecked(rhs_word_start + w) };
        let valid_bits =
            mask_mask_opt
                .as_ref()
                .map_or(!0u64, |(mask_mask, mask_word_start, _)| unsafe {
                    mask_mask.word_unchecked(mask_word_start + w)
                });
        let word_cmp = match op {
            ComparisonOperator::Equals => !(a ^ b),
            ComparisonOperator::NotEquals => a ^ b,
            ComparisonOperator::GreaterThan => a & (!b),
            ComparisonOperator::LessThan => (!a) & b,
            ComparisonOperator::GreaterThanOrEqualTo => a | (!b),
            ComparisonOperator::LessThanOrEqualTo => (!a) | b,
            _ => 0,
        };
        let final_bits = word_cmp & valid_bits;
        unsafe {
            out.set_word_unchecked(w, final_bits);
        }
    }

    // Tail often caused by `n % LANES != 0`; uses scalar fallback.

    let base = words * 64;
    for i in 0..tail {
        let idx_lhs = lhs_offset + base + i;
        let idx_rhs = rhs_offset + base + i;
        let mask_valid =
            mask_mask_opt
                .as_ref()
                .map_or(true, |(mask_mask, mask_word_start, mask_len)| unsafe {
                    let mask_idx = mask_word_start * 64 + base + i;
                    if mask_idx < *mask_len {
                        mask_mask.get_unchecked(mask_idx)
                    } else {
                        false
                    }
                });
        if !mask_valid {
            continue;
        }
        if idx_lhs >= lhs_mask.len() || idx_rhs >= rhs_mask.len() {
            continue;
        }
        let a = unsafe { lhs_mask.get_unchecked(idx_lhs) };
        let b = unsafe { rhs_mask.get_unchecked(idx_rhs) };
        let res = match op {
            ComparisonOperator::Equals => a == b,
            ComparisonOperator::NotEquals => a != b,
            ComparisonOperator::GreaterThan => a & !b,
            ComparisonOperator::LessThan => !a & b,
            ComparisonOperator::GreaterThanOrEqualTo => a | !b,
            ComparisonOperator::LessThanOrEqualTo => !a | b,
            _ => false,
        };
        if res {
            out.set(base + i, true)
        }
    }
    out.mask_trailing_bits();
    Ok(out)
}

// String and dictionary

macro_rules! impl_cmp_utf8_slice {
    ($fn_name:ident, $fn_name_to:ident, $lhs_slice:ty, $rhs_slice:ty, [$($gen:tt)+]) => {
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Compare UTF-8 string or dictionary arrays using the specified comparison operator.
        /// The output Bitmask must have capacity >= llen.
        #[inline(always)]
        pub fn $fn_name_to<$($gen)+>(
            lhs: $lhs_slice,
            rhs: $rhs_slice,
            op: ComparisonOperator,
            output: &mut Bitmask,
        ) -> Result<(), KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            confirm_equal_len("compare string/dict length mismatch (slice contract)", llen, rlen)?;
            assert!(output.capacity() >= llen, concat!(stringify!($fn_name_to), ": output capacity too small"));

            let lhs_mask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
            let rhs_mask = rarr.null_mask.as_ref().map(|m| m.slice_clone(roff, rlen));

            if let Some(m) = larr.null_mask.as_ref() {
                if m.capacity() < loff + llen {
                    return Err(KernelError::InvalidArguments(
                        format!(
                            "lhs mask capacity too small (expected ≥ {}, got {})",
                            loff + llen,
                            m.capacity()
                        ),
                    ));
                }
            }
            if let Some(m) = rarr.null_mask.as_ref() {
                if m.capacity() < roff + rlen {
                    return Err(KernelError::InvalidArguments(
                        format!(
                            "rhs mask capacity too small (expected ≥ {}, got {})",
                            roff + rlen,
                            m.capacity()
                        ),
                    ));
                }
            }

            let has_nulls = lhs_mask.is_some() || rhs_mask.is_some();
            for i in 0..llen {
                if has_nulls
                    && !(lhs_mask.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) })
                        && rhs_mask.as_ref().map_or(true, |m| unsafe { m.get_unchecked(i) }))
                {
                    continue;
                }
                let l = unsafe { larr.get_str_unchecked(loff + i) };
                let r = unsafe { rarr.get_str_unchecked(roff + i) };
                let res = match op {
                    ComparisonOperator::Equals => l == r,
                    ComparisonOperator::NotEquals => l != r,
                    ComparisonOperator::GreaterThan => l > r,
                    ComparisonOperator::LessThan => l < r,
                    ComparisonOperator::GreaterThanOrEqualTo => l >= r,
                    ComparisonOperator::LessThanOrEqualTo => l <= r,
                    _ => false,
                };
                if res {
                    output.set(i, true);
                }
            }
            Ok(())
        }

        /// Compare UTF-8 string or dictionary arrays using the specified comparison operator.
        #[inline(always)]
        pub fn $fn_name<$($gen)+>(
            lhs: $lhs_slice,
            rhs: $rhs_slice,
            op: ComparisonOperator,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, _) = rhs;
            let lhs_mask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
            let rhs_mask = rarr.null_mask.as_ref().map(|m| m.slice_clone(roff, llen));
            let mut out = new_bool_bitmask(llen);
            $fn_name_to((larr, loff, llen), (rarr, roff, llen), op, &mut out)?;
            let null_mask = merge_bitmasks_to_new(lhs_mask.as_ref(), rhs_mask.as_ref(), llen);
            Ok(BooleanArray { data: out.into(), null_mask, len: llen, _phantom: PhantomData })
        }
    };
}

impl_cmp_numeric!(cmp_i32, cmp_i32_to, i32, W32, i32);
impl_cmp_numeric!(cmp_u32, cmp_u32_to, u32, W32, i32);
impl_cmp_numeric!(cmp_i64, cmp_i64_to, i64, W64, i64);
impl_cmp_numeric!(cmp_u64, cmp_u64_to, u64, W64, i64);
impl_cmp_numeric!(cmp_f32, cmp_f32_to, f32, W32, i32);
impl_cmp_numeric!(cmp_f64, cmp_f64_to, f64, W64, i64);
impl_cmp_utf8_slice!(cmp_str_str,   cmp_str_str_to,   StringAVT<'a, T>,     StringAVT<'a, T>,      [ 'a, T: Integer ]);
impl_cmp_utf8_slice!(cmp_str_dict,  cmp_str_dict_to,  StringAVT<'a, T>,     CategoricalAVT<'a, U>,      [ 'a, T: Integer, U: Integer ]);
impl_cmp_utf8_slice!(cmp_dict_str,  cmp_dict_str_to,  CategoricalAVT<'a, T>,     StringAVT<'a, U>,      [ 'a, T: Integer, U: Integer ]);
impl_cmp_utf8_slice!(cmp_dict_dict, cmp_dict_dict_to, CategoricalAVT<'a, T>,     CategoricalAVT<'a, T>,      [ 'a, T: Integer ]);

#[cfg(test)]
mod tests {
    use minarrow::{Bitmask, BooleanArray, CategoricalArray, Integer, StringArray, vec64};

    use crate::kernels::comparison::{
        cmp_dict_dict, cmp_dict_str, cmp_i32, cmp_numeric, cmp_str_dict,
    };

    #[cfg(feature = "simd")]
    use crate::kernels::comparison::{W64, cmp_bitmask_simd};

    use crate::operators::ComparisonOperator;

    /// --- helpers --------------------------------------------------------------

    fn bm(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            m.set(i, b);
        }
        m
    }

    /// Assert BooleanArray ⇢ expected value bits & expected null bits.
    fn assert_bool(arr: &BooleanArray<()>, expect: &[bool], expect_mask: Option<&[bool]>) {
        assert_eq!(arr.len, expect.len());
        for i in 0..expect.len() {
            assert_eq!(arr.data.get(i), expect[i], "value bit {i}");
        }
        match (arr.null_mask.as_ref(), expect_mask) {
            (None, None) => {}
            (Some(m), Some(exp)) => {
                for (i, &b) in exp.iter().enumerate() {
                    assert_eq!(m.get(i), b, "null-bit {i}");
                }
            }
            _ => panic!("mask mismatch"),
        }
    }

    /// Tiny helpers to build test String / Dict arrays.
    fn str_arr<T: Integer>(v: &[&str]) -> StringArray<T> {
        StringArray::<T>::from_slice(v)
    }

    fn dict_arr<T: Integer>(vals: &[&str]) -> CategoricalArray<T> {
        let owned: Vec<&str> = vals.to_vec();
        CategoricalArray::<T>::from_values(owned)
    }

    // NUMERIC

    #[test]
    fn numeric_compare_no_nulls() {
        let a = vec64![1i32, 2, 3, 4];
        let b = vec64![1i32, 1, 4, 4];

        let eq = cmp_i32(&a, &b, None, ComparisonOperator::Equals).unwrap();
        let neq = cmp_i32(&a, &b, None, ComparisonOperator::NotEquals).unwrap();
        let lt = cmp_i32(&a, &b, None, ComparisonOperator::LessThan).unwrap();
        let le = cmp_i32(&a, &b, None, ComparisonOperator::LessThanOrEqualTo).unwrap();
        let gt = cmp_i32(&a, &b, None, ComparisonOperator::GreaterThan).unwrap();
        let ge = cmp_i32(&a, &b, None, ComparisonOperator::GreaterThanOrEqualTo).unwrap();

        assert_bool(&eq, &[true, false, false, true], None);
        assert_bool(&neq, &[false, true, true, false], None);
        assert_bool(&lt, &[false, false, true, false], None);
        assert_bool(&le, &[true, false, true, true], None);
        assert_bool(&gt, &[false, true, false, false], None);
        assert_bool(&ge, &[true, true, false, true], None);
    }

    #[test]
    fn numeric_compare_with_nulls_generic_dispatch() {
        // last element masked-out
        let a = vec64![1u64, 5, 9, 10];
        let b = vec64![0u64, 5, 8, 11];
        let mask = bm(&[true, true, true, false]);

        let out = cmp_numeric(&a, &b, Some(&mask), ComparisonOperator::GreaterThan).unwrap();
        // result bits for valid rows only
        assert_bool(
            &out,
            &[true, false, true, false],
            Some(&[true, true, true, false]),
        );
    }

    // BOOLEAN

    #[cfg(feature = "simd")]
    #[test]
    fn bool_compare_all_ops() {
        let a = bm(&[true, false, true, false]);
        let b = bm(&[true, true, false, false]);
        let eq = cmp_bitmask_simd::<W64>(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            None,
            ComparisonOperator::Equals,
        )
        .unwrap();
        let lt = cmp_bitmask_simd::<W64>(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            None,
            ComparisonOperator::LessThan,
        )
        .unwrap();
        let gt = cmp_bitmask_simd::<W64>(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            None,
            ComparisonOperator::GreaterThan,
        )
        .unwrap();

        assert_bool(
            &BooleanArray::from_bitmask(eq, None),
            &[true, false, false, true],
            None,
        );
        assert_bool(
            &BooleanArray::from_bitmask(lt, None),
            &[false, true, false, false],
            None,
        );
        assert_bool(
            &BooleanArray::from_bitmask(gt, None),
            &[false, false, true, false],
            None,
        );
    }

    // UTF & DICTIONARY

    #[test]
    fn string_vs_dict_compare_with_nulls() {
        let mut lhs = str_arr::<u32>(&["x", "y", "z"]);
        lhs.null_mask = Some(bm(&[true, false, true]));
        let rhs = dict_arr::<u32>(&["x", "w", "zz"]);
        let lhs_slice = (&lhs, 0, lhs.len());
        let rhs_slice = (&rhs, 0, rhs.data.len());
        let res = cmp_str_dict(lhs_slice, rhs_slice, ComparisonOperator::Equals).unwrap();
        assert_bool(&res, &[true, false, false], Some(&[true, false, true]));
    }

    #[test]
    fn string_vs_dict_compare_with_nulls_chunk() {
        let mut lhs = str_arr::<u32>(&["pad", "x", "y", "z", "pad"]);
        lhs.null_mask = Some(bm(&[true, true, false, true, true]));
        let rhs = dict_arr::<u32>(&["pad", "x", "w", "zz", "pad"]);
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let res = cmp_str_dict(lhs_slice, rhs_slice, ComparisonOperator::Equals).unwrap();
        assert_bool(&res, &[true, false, false], Some(&[true, false, true]));
    }

    #[test]
    fn dict_vs_dict_compare_gt() {
        let lhs = dict_arr::<u32>(&["apple", "pear", "banana"]);
        let rhs = dict_arr::<u32>(&["ant", "pear", "apricot"]);
        let lhs_slice = (&lhs, 0, lhs.data.len());
        let rhs_slice = (&rhs, 0, rhs.data.len());
        let res = cmp_dict_dict(lhs_slice, rhs_slice, ComparisonOperator::GreaterThan).unwrap();
        assert_bool(&res, &[true, false, true], None);
    }

    #[test]
    fn dict_vs_dict_compare_gt_chunk() {
        let lhs = dict_arr::<u32>(&["pad", "apple", "pear", "banana", "pad"]);
        let rhs = dict_arr::<u32>(&["pad", "ant", "pear", "apricot", "pad"]);
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let res = cmp_dict_dict(lhs_slice, rhs_slice, ComparisonOperator::GreaterThan).unwrap();
        assert_bool(&res, &[true, false, true], None);
    }

    #[test]
    fn dict_vs_string_compare_le() {
        let lhs = dict_arr::<u32>(&["a", "b", "c"]);
        let rhs = str_arr::<u32>(&["b", "b", "d"]);
        let lhs_slice = (&lhs, 0, lhs.data.len());
        let rhs_slice = (&rhs, 0, rhs.len());
        let res =
            cmp_dict_str(lhs_slice, rhs_slice, ComparisonOperator::LessThanOrEqualTo).unwrap();
        assert_bool(&res, &[true, true, true], None);
    }

    #[test]
    fn dict_vs_string_compare_le_chunk() {
        let lhs = dict_arr::<u32>(&["pad", "a", "b", "c", "pad"]);
        let rhs = str_arr::<u32>(&["pad", "b", "b", "d", "pad"]);
        let lhs_slice = (&lhs, 1, 3);
        let rhs_slice = (&rhs, 1, 3);
        let res =
            cmp_dict_str(lhs_slice, rhs_slice, ComparisonOperator::LessThanOrEqualTo).unwrap();
        assert_bool(&res, &[true, true, true], None);
    }
}
