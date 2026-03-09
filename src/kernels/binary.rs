// Copyright (c) 2025 Peter Bower
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Binary Operations Kernels Module** - *High-Performance Element-wise Binary Operations*
//!
//! Comprehensive binary operation kernels providing element-wise operations between array pairs
//! with null-aware semantics and SIMD acceleration. Critical foundation for analytical computing
//! requiring efficient pairwise data transformations.
//!
//! ## Core Operations
//! - **Numeric comparisons**: Greater than, less than, equal operations across all numeric types
//! - **String operations**: String comparison with UTF-8 aware lexicographic ordering
//! - **Categorical operations**: Dictionary-encoded string comparisons with optimised lookups
//! - **Set operations**: Membership testing with efficient hash-based implementations
//! - **Range operations**: Between operations for numeric and string data types
//! - **Logical combinations**: AND, OR, XOR operations on boolean arrays with bitmask optimisation
//!
//! ## Architecture Overview
//! The module provides a unified interface for binary operations across heterogeneous data types:
//!
//! - **Type-aware dispatch**: Automatic selection of optimised kernels based on input types
//! - **Memory layout optimisation**: Direct array-to-array operations minimising intermediate allocations
//! - **Null propagation**: Proper handling of null values following Apache Arrow semantics
//! - **SIMD vectorisation**: Hardware-accelerated operations on compatible data types

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::hash::Hash;
use std::marker::PhantomData;

use minarrow::traits::type_unions::Float;
use minarrow::utils::confirm_equal_len;
use minarrow::{Bitmask, BooleanAVT, BooleanArray, CategoricalAVT, Integer, Numeric, StringAVT};

#[cfg(not(feature = "simd"))]
use crate::kernels::bitmask::std::{and_masks, not_mask};
#[cfg(not(feature = "simd"))]
use crate::kernels::comparison::cmp_bitmask_std;
use crate::kernels::comparison::{
    cmp_dict_dict, cmp_dict_str, cmp_numeric, cmp_str_dict, cmp_str_str,
};
use crate::kernels::logical::{
    cmp_between, cmp_dict_between, cmp_dict_in, cmp_in, cmp_in_f, cmp_str_between, cmp_str_in,
};
use crate::operators::ComparisonOperator;
use minarrow::enums::error::KernelError;

/// Returns a new Bitmask for boolean buffers, all bits cleared (false).
#[inline(always)]
fn new_bool_bitmask(len_bits: usize) -> Bitmask {
    Bitmask::new_set_all(len_bits, false)
}

/// Returns a new Bitmask for boolean buffers, all bits set (true).
#[inline(always)]
fn full_bool_bitmask(len_bits: usize) -> Bitmask {
    Bitmask::new_set_all(len_bits, true)
}

/// Merge two optional Bitmasks into a new output mask, computing per-row AND.
/// Returns None if both inputs are None (output is dense).
#[inline]
fn merge_bitmasks_to_new(
    lhs: Option<&Bitmask>,
    rhs: Option<&Bitmask>,
    len: usize,
) -> Option<Bitmask> {
    if let Some(m) = lhs {
        debug_assert!(
            m.capacity() >= len,
            "lhs null mask too small: capacity {} < len {}",
            m.capacity(),
            len
        );
    }
    if let Some(m) = rhs {
        debug_assert!(
            m.capacity() >= len,
            "rhs null mask too small: capacity {} < len {}",
            m.capacity(),
            len
        );
    }

    match (lhs, rhs) {
        (None, None) => None,

        (Some(l), None) | (None, Some(l)) => {
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                unsafe {
                    out.set_unchecked(i, l.get_unchecked(i));
                }
            }
            Some(out)
        }

        (Some(l), Some(r)) => {
            let mut out = Bitmask::new_set_all(len, true);
            for i in 0..len {
                unsafe {
                    out.set_unchecked(i, l.get_unchecked(i) && r.get_unchecked(i));
                }
            }
            Some(out)
        }
    }
}

// Numeric and Float

/// Applies comparison operations between numeric arrays with comprehensive operator support.
///
/// Performs element-wise comparison operations between two numeric arrays using the specified
/// comparison operator. Supports the full range of SQL comparison semantics including
/// set membership operations and null-aware comparisons.
///
/// ## Parameters
/// * `lhs` - Left-hand side numeric array for comparison
/// * `rhs` - Right-hand side numeric array for comparison  
/// * `mask` - Optional bitmask indicating valid elements in input arrays
/// * `op` - Comparison operator defining the comparison semantics to apply
///
/// ## Returns
/// Returns `Result<BooleanArray<()>, KernelError>` containing:
/// - **Success**: Boolean array with comparison results
/// - **Error**: KernelError if comparison operation fails
///
/// ## Supported Operations
/// - **Basic comparisons**: `<`, `<=`, `>`, `>=`, `==`, `!=`
/// - **Set operations**: `IN`, `NOT IN` for membership testing
/// - **Range operations**: `BETWEEN` for range inclusion testing
/// - **Null operations**: `IS NULL`, `IS NOT NULL` for null checking
///
/// ## Examples
/// ```rust,ignore
/// use simd_kernels::kernels::binary::apply_cmp;
/// use simd_kernels::operators::ComparisonOperator;
///
/// let lhs = [1, 2, 3, 4];
/// let rhs = [2, 2, 2, 2];
/// let result = apply_cmp(&lhs, &rhs, None, ComparisonOperator::LessThan).unwrap();
/// // Result: [true, false, false, false]
/// ```
pub fn apply_cmp<T>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError>
where
    T: Numeric + Copy + Hash + Eq + PartialOrd + 'static,
{
    let len = lhs.len();
    match op {
        ComparisonOperator::Between => {
            let mut out = cmp_between(lhs, rhs)?;
            out.null_mask = mask.cloned();
            Ok(out)
        }
        ComparisonOperator::In => {
            let mut out = cmp_in(lhs, rhs)?;
            out.null_mask = mask.cloned();
            Ok(out)
        }
        ComparisonOperator::NotIn => {
            let mut out = cmp_in(lhs, rhs)?;
            for i in 0..len {
                unsafe { out.data.set_unchecked(i, !out.data.get_unchecked(i)) };
            }
            out.null_mask = mask.cloned();
            Ok(out)
        }
        ComparisonOperator::IsNull => {
            let data = match mask {
                Some(m) => {
                    #[cfg(feature = "simd")]
                    {
                        minarrow::kernels::bitmask::simd::not_mask_simd::<W8>((m, 0, len))
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        not_mask((m, 0, len))
                    }
                }
                None => new_bool_bitmask(len),
            };
            Ok(BooleanArray {
                data,
                null_mask: None,
                len,
                _phantom: std::marker::PhantomData,
            })
        }
        ComparisonOperator::IsNotNull => {
            let data = match mask {
                Some(m) => m.slice_clone(0, len),
                None => full_bool_bitmask(len),
            };
            Ok(BooleanArray {
                data,
                null_mask: None,
                len,
                _phantom: std::marker::PhantomData,
            })
        }
        _ => {
            let mut out = cmp_numeric(lhs, rhs, mask, op)?;
            out.null_mask = mask.cloned();
            Ok(out)
        }
    }
}

/// Applies comparison operations between floating-point arrays with IEEE 754 compliance.
///
/// Performs element-wise floating-point comparisons with proper handling of IEEE 754
/// special values (NaN, infinity). Implements comprehensive comparison semantics for
/// floating-point data including set operations and null-aware processing.
///
/// ## Parameters
/// * `lhs` - Left-hand side floating-point array for comparison
/// * `rhs` - Right-hand side floating-point array for comparison
/// * `mask` - Optional bitmask indicating valid elements in arrays
/// * `op` - Comparison operator specifying the comparison type to perform
///
/// ## Returns
/// Returns `Result<BooleanArray<()>, KernelError>` containing:
/// - **Success**: Boolean array with IEEE 754 compliant comparison results
/// - **Error**: KernelError if floating-point comparison fails
/// ```
pub fn apply_cmp_f<T>(
    lhs: &[T],
    rhs: &[T],
    mask: Option<&Bitmask>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError>
where
    T: Float + Numeric + Copy + 'static,
{
    let len = lhs.len();
    match op {
        ComparisonOperator::Between => {
            let mut out = cmp_between(lhs, rhs)?;
            out.null_mask = mask.cloned();
            Ok(out)
        }
        ComparisonOperator::In => {
            let mut out = cmp_in_f(lhs, rhs)?;
            out.null_mask = mask.cloned();
            Ok(out)
        }
        ComparisonOperator::NotIn => {
            let mut out = cmp_in_f(lhs, rhs)?;
            for i in 0..len {
                unsafe { out.data.set_unchecked(i, !out.data.get_unchecked(i)) };
            }
            out.null_mask = mask.cloned();
            Ok(out)
        }
        ComparisonOperator::IsNull => {
            let data = match mask {
                Some(m) => {
                    #[cfg(feature = "simd")]
                    {
                        minarrow::kernels::bitmask::simd::not_mask_simd::<W8>((m, 0, len))
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        not_mask((m, 0, len))
                    }
                }
                None => new_bool_bitmask(len),
            };
            Ok(BooleanArray {
                data,
                null_mask: None,
                len,
                _phantom: std::marker::PhantomData,
            })
        }
        ComparisonOperator::IsNotNull => {
            let data = match mask {
                Some(m) => m.slice_clone(0, len),
                None => full_bool_bitmask(len),
            };
            Ok(BooleanArray {
                data,
                null_mask: None,
                len,
                _phantom: std::marker::PhantomData,
            })
        }
        ComparisonOperator::Equals | ComparisonOperator::NotEquals => {
            let mut out = cmp_numeric(lhs, rhs, mask, op)?;
            // Patch NaN-pairs for legacy semantics
            for i in 0..len {
                let is_valid = mask.map_or(true, |m| unsafe { m.get_unchecked(i) });
                if is_valid && lhs[i].is_nan() && rhs[i].is_nan() {
                    match op {
                        ComparisonOperator::Equals => unsafe { out.data.set_unchecked(i, true) },
                        ComparisonOperator::NotEquals => unsafe {
                            out.data.set_unchecked(i, false)
                        },
                        _ => {}
                    }
                }
            }
            out.null_mask = mask.cloned();
            Ok(out)
        }
        _ => {
            let mut out = cmp_numeric(lhs, rhs, mask, op)?;
            out.null_mask = mask.cloned();
            Ok(out)
        }
    }
}

/// Boolean Bit packed
///
/// Note that this function delegates to SIMD or not SIMD within the inner cmp_bool
/// module, given bool is self-contained as a datatype.
/// "Elementwise boolean bitwise SIMD comparison falling back to scalar if simd not enabled.
/// Returns `BooleanArray<()>`."
#[inline(always)]
pub fn apply_cmp_bool(
    lhs: BooleanAVT<'_, ()>,
    rhs: BooleanAVT<'_, ()>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError> {
    let (lhs_arr, lhs_off, len) = lhs;
    let (rhs_arr, rhs_off, rlen) = rhs;
    confirm_equal_len("apply_cmp_bool_windowed: window length mismatch", len, rlen)?;

    // Merge windowed null masks
    #[cfg(feature = "simd")]
    let merged_null_mask: Option<Bitmask> =
        match (lhs_arr.null_mask.as_ref(), rhs_arr.null_mask.as_ref()) {
            (None, None) => None,
            (Some(m), None) => Some(m.slice_clone(lhs_off, len)),
            (None, Some(m)) => Some(m.slice_clone(rhs_off, len)),
            (Some(a), Some(b)) => {
                use minarrow::kernels::bitmask::simd::and_masks_simd;
                let am = (a, lhs_off, len);
                let bm = (b, rhs_off, len);
                Some(and_masks_simd::<W8>(am, bm))
            }
        };

    #[cfg(not(feature = "simd"))]
    let merged_null_mask: Option<Bitmask> =
        match (lhs_arr.null_mask.as_ref(), rhs_arr.null_mask.as_ref()) {
            (None, None) => None,
            (Some(m), None) => Some(m.slice_clone(lhs_off, len)),
            (None, Some(m)) => Some(m.slice_clone(rhs_off, len)),
            (Some(a), Some(b)) => {
                let am = (a, lhs_off, len);
                let bm = (b, rhs_off, len);
                Some(and_masks(am, bm))
            }
        };

    let mask_slice = merged_null_mask.as_ref().map(|m| (m, 0, len));

    #[cfg(feature = "simd")]
    let data = match op {
        ComparisonOperator::Equals
        | ComparisonOperator::NotEquals
        | ComparisonOperator::LessThan
        | ComparisonOperator::LessThanOrEqualTo
        | ComparisonOperator::GreaterThan
        | ComparisonOperator::GreaterThanOrEqualTo
        | ComparisonOperator::In
        | ComparisonOperator::NotIn => crate::kernels::comparison::cmp_bitmask_simd::<W8>(
            (&lhs_arr.data, lhs_off, len),
            (&rhs_arr.data, rhs_off, len),
            mask_slice,
            op,
        )?,
        ComparisonOperator::IsNull => {
            let data = match merged_null_mask.as_ref() {
                Some(mask) => minarrow::kernels::bitmask::simd::not_mask_simd::<W8>((mask, 0, len)),
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

    #[cfg(not(feature = "simd"))]
    let data = match op {
        ComparisonOperator::Equals
        | ComparisonOperator::NotEquals
        | ComparisonOperator::LessThan
        | ComparisonOperator::LessThanOrEqualTo
        | ComparisonOperator::GreaterThan
        | ComparisonOperator::GreaterThanOrEqualTo
        | ComparisonOperator::In
        | ComparisonOperator::NotIn => cmp_bitmask_std(
            (&lhs_arr.data, lhs_off, len),
            (&rhs_arr.data, rhs_off, len),
            mask_slice,
            op,
        )?,
        ComparisonOperator::IsNull => {
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

// Utf8 Dictionary

/// Applies comparison operations between corresponding string elements from string arrays.
///
/// Performs element-wise string comparison using lexicographic ordering with
/// UTF-8 awareness and efficient null handling.
///
/// # Parameters
/// - `lhs`: Left-hand string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Right-hand string array view tuple `(StringArray, offset, length)`
/// - `op`: Comparison operator (Eq, Ne, Lt, Le, Gt, Ge, In, NotIn)
///
/// # String Comparison
/// - Uses Rust's standard UTF-8 aware lexicographic ordering
/// - Null strings handled consistently across all operations
/// - Set operations support string membership testing
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true elements satisfy the comparison.
///
/// # Performance
/// - Optimised string comparison avoiding unnecessary allocations
/// - Efficient null mask processing with bitwise operations
/// - Dictionary-style operations for set membership testing
pub fn apply_cmp_str<T: Integer>(
    lhs: StringAVT<T>,
    rhs: StringAVT<T>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError> {
    // Destructure slice windows
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;

    assert_eq!(llen, rlen, "apply_cmp_str: slice lengths must match");

    let lmask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let rmask = rarr.null_mask.as_ref().map(|m| m.slice_clone(roff, rlen));
    let null_mask = merge_bitmasks_to_new(lmask.as_ref(), rmask.as_ref(), llen);

    let mut out = match op {
        ComparisonOperator::Between => cmp_str_between((larr, loff, llen), (rarr, roff, rlen)),
        ComparisonOperator::In => cmp_str_in((larr, loff, llen), (rarr, roff, rlen)),
        ComparisonOperator::NotIn => {
            let mut b = cmp_str_in((larr, loff, llen), (rarr, roff, rlen))?;
            debug_assert!(
                b.data.capacity() >= llen,
                "bitmask capacity {} < needed len {}",
                b.data.capacity(),
                llen
            );
            for i in 0..llen {
                unsafe { b.data.set_unchecked(i, !b.data.get_unchecked(i)) };
            }
            Ok(b)
        }
        ComparisonOperator::IsNull => {
            let data = match null_mask.as_ref() {
                Some(m) => {
                    #[cfg(feature = "simd")]
                    {
                        minarrow::kernels::bitmask::simd::not_mask_simd::<W8>((m, 0, llen))
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        not_mask((m, 0, llen))
                    }
                }
                None => new_bool_bitmask(llen),
            };
            return Ok(BooleanArray {
                data,
                null_mask: None,
                len: llen,
                _phantom: std::marker::PhantomData,
            });
        }
        ComparisonOperator::IsNotNull => {
            let data = match null_mask.as_ref() {
                Some(m) => m.slice_clone(0, llen),
                None => full_bool_bitmask(llen),
            };
            return Ok(BooleanArray {
                data,
                null_mask: None,
                len: llen,
                _phantom: std::marker::PhantomData,
            });
        }
        _ => cmp_str_str((larr, loff, llen), (rarr, roff, rlen), op),
    }?;
    out.null_mask = null_mask;
    out.len = llen;
    Ok(out)
}

/// Applies comparison operations between string array elements and categorical dictionary values.
///
/// Performs element-wise comparison where left operands are strings and right operands
/// are resolved from a categorical array's dictionary.
///
/// # Parameters
/// - `lhs`: String array view tuple `(StringArray, offset, length)`
/// - `rhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
/// - `op`: Comparison operator (Eq, Ne, Lt, Le, Gt, Ge, In, NotIn)
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true elements satisfy the comparison.
///
/// # Performance
/// Dictionary lookups amortised across categorical comparisons with caching opportunities.
pub fn apply_cmp_str_dict<T: Integer, U: Integer>(
    lhs: StringAVT<T>,
    rhs: CategoricalAVT<U>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    assert_eq!(llen, rlen, "apply_cmp_str_dict: slice lengths must match");

    // TODO: Avoid double clone - merge/slice bitmasks in one go
    let lmask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let rmask = rarr.null_mask.as_ref().map(|m| m.slice_clone(roff, rlen));
    let null_mask = merge_bitmasks_to_new(lmask.as_ref(), rmask.as_ref(), llen);

    let mut out = cmp_str_dict((larr, loff, llen), (rarr, roff, rlen), op)?;
    out.null_mask = null_mask;
    out.len = llen;
    Ok(out)
}

/// Applies comparison operations between categorical dictionary values and string array elements.
///
/// Performs element-wise comparison where left operands are resolved from a categorical
/// array's dictionary and right operands are strings.
///
/// # Parameters
/// - `lhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: String array view tuple `(StringArray, offset, length)`
/// - `op`: Comparison operator (Eq, Ne, Lt, Le, Gt, Ge, In, NotIn)
///
/// # Type Parameters
/// - `T`: Integer type for categorical array indices
/// - `U`: Integer type for string array offsets
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true elements satisfy the comparison.
///
/// # Performance
/// Dictionary lookups optimised with categorical encoding efficiency.
pub fn apply_cmp_dict_str<T: Integer, U: Integer>(
    lhs: CategoricalAVT<T>,
    rhs: StringAVT<U>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    assert_eq!(llen, rlen, "apply_cmp_dict_str: slice lengths must match");

    // TODO: Avoid double clone - merge/slice bitmasks in one go
    let lmask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let rmask = rarr.null_mask.as_ref().map(|m| m.slice_clone(roff, rlen));
    let null_mask = merge_bitmasks_to_new(lmask.as_ref(), rmask.as_ref(), llen);

    let mut out = cmp_dict_str((larr, loff, llen), (rarr, roff, rlen), op)?;
    out.null_mask = null_mask;
    out.len = llen;
    Ok(out)
}

/// Applies comparison operations between corresponding categorical dictionary values.
///
/// Performs element-wise comparison by resolving both operands from their respective
/// categorical dictionaries and comparing the resulting string values.
///
/// # Parameters
/// - `lhs`: Left categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Right categorical array view tuple `(CategoricalArray, offset, length)`
/// - `op`: Comparison operator (Eq, Ne, Lt, Le, Gt, Ge, In, NotIn)
///
/// # Type Parameters
/// - `T`: Integer type for categorical array indices (must implement `Hash`)
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true elements satisfy the comparison.
///
/// # Performance
/// - Dictionary lookups amortised across bulk categorical operations
/// - Hash-based optimisations for set membership operations
/// - Efficient categorical code comparison where possible
pub fn apply_cmp_dict<T: Integer + Hash>(
    lhs: CategoricalAVT<T>,
    rhs: CategoricalAVT<T>,
    op: ComparisonOperator,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    assert_eq!(llen, rlen, "apply_cmp_dict: slice lengths must match");
    let lmask = larr.null_mask.as_ref().map(|m| m.slice_clone(loff, llen));
    let rmask = rarr.null_mask.as_ref().map(|m| m.slice_clone(roff, rlen));
    let null_mask = merge_bitmasks_to_new(lmask.as_ref(), rmask.as_ref(), llen);
    let mut out = match op {
        ComparisonOperator::Between => cmp_dict_between((larr, loff, llen), (rarr, roff, rlen)),
        ComparisonOperator::In => cmp_dict_in((larr, loff, llen), (rarr, roff, rlen)),
        ComparisonOperator::NotIn => {
            let mut b = cmp_dict_in((larr, loff, llen), (rarr, roff, rlen))?;
            for i in 0..llen {
                unsafe {
                    b.data.set_unchecked(i, !b.data.get_unchecked(i));
                }
            }
            Ok(b)
        }
        ComparisonOperator::IsNull => {
            let data = match null_mask.as_ref() {
                Some(m) => {
                    #[cfg(feature = "simd")]
                    {
                        minarrow::kernels::bitmask::simd::not_mask_simd::<W8>((m, 0, llen))
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        not_mask((m, 0, llen))
                    }
                }
                None => new_bool_bitmask(llen),
            };
            return Ok(BooleanArray {
                data,
                null_mask: None,
                len: llen,
                _phantom: std::marker::PhantomData,
            });
        }
        ComparisonOperator::IsNotNull => {
            let data = match null_mask.as_ref() {
                Some(m) => m.slice_clone(0, llen),
                None => full_bool_bitmask(llen),
            };
            return Ok(BooleanArray {
                data,
                null_mask: None,
                len: llen,
                _phantom: std::marker::PhantomData,
            });
        }
        _ => cmp_dict_dict((larr, loff, llen), (rarr, roff, rlen), op),
    }?;
    out.null_mask = null_mask;
    out.len = llen;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use minarrow::structs::variants::categorical::CategoricalArray;
    use minarrow::structs::variants::string::StringArray;
    use minarrow::{Bitmask, BooleanArray, MaskedArray, vec64};

    use super::*;

    // --- Helpers ---
    fn bm(bools: &[bool]) -> Bitmask {
        Bitmask::from_bools(bools)
    }
    fn bool_arr(bools: &[bool]) -> BooleanArray<()> {
        BooleanArray::from_slice(bools)
    }

    // ----------- Numeric & Float -----------
    #[test]
    fn test_apply_cmp_numeric_all_ops() {
        let a = vec64![1, 2, 3, 4, 5, 6];
        let b = vec64![3, 2, 1, 4, 5, 0];
        let mask = bm(&[true, false, true, true, true, true]);

        // Standard operators
        for &op in &[
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqualTo,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqualTo,
        ] {
            let arr = apply_cmp(&a, &b, Some(&mask), op).unwrap();
            for i in 0..a.len() {
                let expect = match op {
                    ComparisonOperator::Equals => a[i] == b[i],
                    ComparisonOperator::NotEquals => a[i] != b[i],
                    ComparisonOperator::LessThan => a[i] < b[i],
                    ComparisonOperator::LessThanOrEqualTo => a[i] <= b[i],
                    ComparisonOperator::GreaterThan => a[i] > b[i],
                    ComparisonOperator::GreaterThanOrEqualTo => a[i] >= b[i],
                    _ => unreachable!(),
                };
                if mask.get(i) {
                    assert_eq!(arr.data.get(i), expect);
                } else {
                    assert_eq!(arr.get(i), None);
                }
            }
            assert_eq!(arr.null_mask, Some(mask.clone()));
        }
    }

    #[test]
    fn test_apply_cmp_numeric_between_in_notin() {
        let a = vec64![4, 2, 3, 5];
        let mask = bm(&[true, true, false, true]);
        // Between [2, 4] (all lhs compared to range 2..=4)
        let rhs = vec64![2, 4];
        let arr = apply_cmp(&a, &rhs, Some(&mask), ComparisonOperator::Between).unwrap();
        assert_eq!(arr.data.get(0), true); // 4 in [2,4]
        assert_eq!(arr.data.get(1), true); // 2 in [2,4]
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.data.get(3), false); // 5 in [2,4]
        // In: a in b
        let rhs = vec64![2, 3, 4];
        let arr = apply_cmp(&a, &rhs, Some(&mask), ComparisonOperator::In).unwrap();
        assert_eq!(arr.data.get(0), true); // 4 in [2,3,4]
        assert_eq!(arr.data.get(1), true); // 2 in [2,3,4]
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.data.get(3), false); // 5 not in [2,3,4]
        // NotIn: inverted
        let arr = apply_cmp(&a, &rhs, Some(&mask), ComparisonOperator::NotIn).unwrap();
        assert_eq!(arr.data.get(0), false);
        assert_eq!(arr.data.get(1), false);
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.data.get(3), true);
    }

    #[test]
    fn test_apply_cmp_numeric_isnull_isnotnull() {
        let a = vec64![1, 2, 3];
        let mask = bm(&[true, false, true]); // position 1 is null
        let arr = apply_cmp(&a, &a, Some(&mask), ComparisonOperator::IsNull).unwrap();
        assert_eq!(arr.data.get(0), false); // present -> not null
        assert_eq!(arr.data.get(1), true); // absent -> is null
        assert_eq!(arr.data.get(2), false); // present -> not null
        assert_eq!(arr.null_mask, None); // result is always valid
        let arr = apply_cmp(&a, &a, Some(&mask), ComparisonOperator::IsNotNull).unwrap();
        assert_eq!(arr.data.get(0), true); // present -> is not null
        assert_eq!(arr.data.get(1), false); // absent -> is null
        assert_eq!(arr.data.get(2), true); // present -> is not null
        assert_eq!(arr.null_mask, None); // result is always valid
    }

    #[test]
    fn test_apply_cmp_numeric_edge_cases() {
        // Empty
        let a: [i32; 0] = [];
        let arr = apply_cmp(&a, &a, None, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.len, 0);
        assert!(arr.null_mask.is_none());
        // All mask None
        let a = vec64![7];
        let arr = apply_cmp(&a, &a, None, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.data.get(0), true);
        assert!(arr.null_mask.is_none());
    }

    #[test]
    fn test_apply_cmp_f_all_ops_nan_patch() {
        let a = vec64![1.0, 2.0, f32::NAN, f32::NAN];
        let b = vec64![1.0, 3.0, f32::NAN, 0.0];
        let mask = bm(&[true, true, true, false]);
        // Equals/NotEquals patches NaN==NaN to true/false
        for &op in &[ComparisonOperator::Equals, ComparisonOperator::NotEquals] {
            let arr = apply_cmp_f(&a, &b, Some(&mask), op).unwrap();
            assert_eq!(arr.data.get(2), matches!(op, ComparisonOperator::Equals)) // true for ==, false for !=
        }
        // In/NotIn
        let arr = apply_cmp_f(&a, &b, Some(&mask), ComparisonOperator::In).unwrap();
        assert_eq!(arr.data.get(0), true); // 1.0 in 1.0
        assert_eq!(arr.data.get(1), false);
    }

    #[test]
    fn test_cmp_bool_w8() {
        let a = bool_arr(&[true, false, true]);
        let b = bool_arr(&[false, false, true]);
        let op = ComparisonOperator::Equals;
        let arr = apply_cmp_bool((&a, 0, a.len()), (&b, 0, b.len()), op).unwrap();
        assert!(!arr.data.get(0));
        assert!(arr.data.get(1));
        assert!(arr.data.get(2));
        println!("mask bytes: {:02x?}", arr.data.bits);
        println!("get(0): {}", arr.data.get(0));
        println!("get(1): {}", arr.data.get(1));
        println!("get(2): {}", arr.data.get(2));
        println!("lhs: {:?}", a);
        println!("rhs: {:?}", b);
        println!(
            "{}: mask bytes: {:?} get(0): {} get(1): {} get(2): {}",
            stringify!($test_name),
            arr.data.as_slice(),
            arr.data.get(0),
            arr.data.get(1),
            arr.data.get(2)
        );

        // NotEquals
        let arr = apply_cmp_bool(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            ComparisonOperator::NotEquals,
        )
        .unwrap();
        assert!(arr.data.get(0));
        assert!(!arr.data.get(1));
        assert!(!arr.data.get(2));

        // LessThan
        let arr = apply_cmp_bool(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            ComparisonOperator::LessThan,
        )
        .unwrap();
        assert!(!arr.data.get(0));
        assert!(!arr.data.get(1));
        assert!(!arr.data.get(2));

        // All null masks
        let mut a = bool_arr(&[true, false]);
        a.null_mask = Some(bm(&[true, false]));
        let mut b = bool_arr(&[true, false]);
        b.null_mask = Some(bm(&[true, true]));
        let arr = apply_cmp_bool(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            ComparisonOperator::Equals,
        )
        .unwrap();
        assert!(arr.null_mask.as_ref().unwrap().get(0));
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
    }

    #[test]
    fn test_bool_is_null() {
        let a = bool_arr(&[true, false]);
        let b = bool_arr(&[false, true]);
        let arr = apply_cmp_bool(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            ComparisonOperator::IsNull,
        )
        .unwrap();
        assert!(!arr.data.get(0));
        assert!(!arr.data.get(1));
        let arr = apply_cmp_bool(
            (&a, 0, a.len()),
            (&b, 0, b.len()),
            ComparisonOperator::IsNotNull,
        )
        .unwrap();
        assert!(arr.data.get(0));
        assert!(arr.data.get(1));
    }

    // ----------- String/Utf8 -----------

    #[test]
    fn test_apply_cmp_str_all_ops() {
        let a = StringArray::<u32>::from_slice(&["foo", "bar", "baz", "qux"]);
        let b = StringArray::<u32>::from_slice(&["foo", "baz", "baz", "quux"]);
        let mut a2 = a.clone();
        a2.set_null(2);
        let a_slice = (&a, 0, a.len());
        let b_slice = (&b, 0, b.len());
        let a2_slice = (&a2, 0, a2.len());

        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.data.get(0), true); // "foo" == "foo"
        assert_eq!(arr.data.get(1), false); // "bar" != "baz"
        assert_eq!(arr.data.get(2), true); // "baz" == "baz"
        assert_eq!(arr.data.get(3), false);

        // NotEquals
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::NotEquals).unwrap();
        assert_eq!(arr.data.get(0), false);
        assert_eq!(arr.data.get(1), true);
        assert_eq!(arr.data.get(2), false);
        assert_eq!(arr.data.get(3), true);

        // LessThan
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::LessThan).unwrap();
        assert_eq!(arr.data.get(0), false); // "foo" < "foo"
        assert_eq!(arr.data.get(1), true); // "bar" < "baz"
        assert_eq!(arr.data.get(2), false);
        assert_eq!(arr.data.get(3), false);

        // Null merging
        let mut b2 = b.clone();
        b2.set_null(1);
        let b2_slice = (&b2, 0, b2.len());
        let arr = apply_cmp_str(a2_slice, b2_slice, ComparisonOperator::Equals).unwrap();
        assert!(!arr.null_mask.as_ref().unwrap().get(2));
        assert!(!arr.null_mask.as_ref().unwrap().get(1));
        assert!(arr.null_mask.as_ref().unwrap().get(0));
        assert!(arr.null_mask.as_ref().unwrap().get(3));
    }

    #[test]
    fn test_apply_cmp_str_all_ops_chunk() {
        let a = StringArray::<u32>::from_slice(&["x", "foo", "bar", "baz", "qux", "y"]);
        let b = StringArray::<u32>::from_slice(&["q", "foo", "baz", "baz", "quux", "z"]);
        // Chunk [1,2,3,4]
        let a_slice = (&a, 1, 4);
        let b_slice = (&b, 1, 4);
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.data.get(0), true); // "foo" == "foo"
        assert_eq!(arr.data.get(1), false); // "bar" != "baz"
        assert_eq!(arr.data.get(2), true); // "baz" == "baz"
        assert_eq!(arr.data.get(3), false);
    }

    #[test]
    fn test_apply_cmp_str_set_ops() {
        let a = StringArray::<u32>::from_slice(&["foo", "bar", "baz"]);
        let b = StringArray::<u32>::from_slice(&["foo", "qux", "baz"]);
        let a_slice = (&a, 0, a.len());
        let b_slice = (&b, 0, b.len());
        // Between
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::Between).unwrap();
        assert_eq!(arr.len, 3);
        // In/NotIn
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::In).unwrap();
        let arr2 = apply_cmp_str(a_slice, b_slice, ComparisonOperator::NotIn).unwrap();
        for i in 0..a.len() {
            assert_eq!(arr.data.get(i), !arr2.data.get(i));
        }
    }

    #[test]
    fn test_apply_cmp_str_set_ops_chunk() {
        let a = StringArray::<u32>::from_slice(&["foo", "bar", "baz", "w"]);
        let b = StringArray::<u32>::from_slice(&["foo", "qux", "baz", "w"]);
        // Chunk [1,2]
        let a_slice = (&a, 1, 2);
        let b_slice = (&b, 1, 2);
        // Between
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::Between).unwrap();
        assert_eq!(arr.len, 2);
        // In/NotIn
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::In).unwrap();
        let arr2 = apply_cmp_str(a_slice, b_slice, ComparisonOperator::NotIn).unwrap();
        for i in 0..2 {
            assert_eq!(arr.data.get(i), !arr2.data.get(i));
        }
    }

    #[test]
    fn test_apply_cmp_str_isnull_isnotnull() {
        let a = StringArray::<u32>::from_slice(&["foo"]);
        let b = StringArray::<u32>::from_slice(&["bar"]);
        let a_slice = (&a, 0, a.len());
        let b_slice = (&b, 0, b.len());
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::IsNull).unwrap();
        assert_eq!(arr.data.get(0), false);
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::IsNotNull).unwrap();
        assert_eq!(arr.data.get(0), true);
    }

    #[test]
    fn test_apply_cmp_str_isnull_isnotnull_chunk() {
        let a = StringArray::<u32>::from_slice(&["pad", "foo"]);
        let b = StringArray::<u32>::from_slice(&["pad", "bar"]);
        let a_slice = (&a, 1, 1);
        let b_slice = (&b, 1, 1);
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::IsNull).unwrap();
        assert_eq!(arr.data.get(0), false);
        let arr = apply_cmp_str(a_slice, b_slice, ComparisonOperator::IsNotNull).unwrap();
        assert_eq!(arr.data.get(0), true);
    }

    // ----------- String/Dict -----------

    #[test]
    fn test_apply_cmp_str_dict() {
        let s = StringArray::<u32>::from_slice(&["a", "b", "c"]);
        let dict = CategoricalArray::<u32>::from_slices(&[0, 1, 0], &["a".into(), "b".into()]);

        let s_slice = (&s, 0, s.len());
        let dict_slice = (&dict, 0, dict.data.len());
        let arr = apply_cmp_str_dict(s_slice, dict_slice, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.len, 3);

        // Null merging
        let mut s2 = s.clone();
        s2.set_null(0);
        let mut d2 = dict.clone();
        d2.set_null(1);

        let s2_slice = (&s2, 0, s2.len());
        let d2_slice = (&d2, 0, d2.data.len());
        let arr = apply_cmp_str_dict(s2_slice, d2_slice, ComparisonOperator::Equals).unwrap();

        let mask = arr.null_mask.as_ref().unwrap();
        assert!(!mask.get(0));
        assert!(!mask.get(1));
        assert!(mask.get(2));
    }

    #[test]
    fn test_apply_cmp_str_dict_chunk() {
        let s = StringArray::<u32>::from_slice(&["pad", "a", "b", "c", "pad2"]);
        let dict = CategoricalArray::<u32>::from_slices(
            &[2, 0, 1, 0, 2], // All indices valid for 3 unique values
            &["z".into(), "a".into(), "b".into()],
        );
        // Slice window ["a", "b", "c"] and ["a", "b", "a"]
        let s_slice = (&s, 1, 3);
        let dict_slice = (&dict, 1, 3);
        let arr = apply_cmp_str_dict(s_slice, dict_slice, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.len, 3);
    }

    // ----------- Dict/Str -----------

    #[test]
    fn test_apply_cmp_dict_str() {
        let dict = CategoricalArray::<u32>::from_slices(&[0, 1, 0], &["a".into(), "b".into()]);
        let s = StringArray::<u32>::from_slice(&["a", "b", "c"]);
        let dict_slice = (&dict, 0, dict.data.len());
        let s_slice = (&s, 0, s.len());
        let arr = apply_cmp_dict_str(dict_slice, s_slice, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.len, 3);
    }

    #[test]
    fn test_apply_cmp_dict_str_chunk() {
        let dict = CategoricalArray::<u32>::from_slices(
            &[2, 0, 1, 0, 2], // Use only indices 0, 1, 2
            &["z".into(), "a".into(), "b".into()],
        );
        let s = StringArray::<u32>::from_slice(&["pad", "a", "b", "c", "pad2"]);
        let dict_slice = (&dict, 1, 3);
        let s_slice = (&s, 1, 3);
        let arr = apply_cmp_dict_str(dict_slice, s_slice, ComparisonOperator::Equals).unwrap();
        assert_eq!(arr.len, 3);
    }

    // ----------- Dict/Dict -----------

    #[test]
    fn test_apply_cmp_dict_all_ops() {
        let a = CategoricalArray::<u32>::from_slices(
            &[0, 1, 2],
            &["dog".into(), "cat".into(), "fish".into()],
        );
        let b = CategoricalArray::<u32>::from_slices(
            &[2, 1, 0],
            &["fish".into(), "cat".into(), "dog".into()],
        );

        let a_slice = (&a, 0, a.data.len());
        let b_slice = (&b, 0, b.data.len());

        // Equals, NotEquals, etc.
        for &op in &[
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqualTo,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqualTo,
        ] {
            let arr = apply_cmp_dict(a_slice, b_slice, op).unwrap();
            assert_eq!(arr.len, 3);
        }
        // Between, In, NotIn
        let arr = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::Between).unwrap();
        assert_eq!(arr.len, 3);
        let arr2 = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::In).unwrap();
        let arr3 = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::NotIn).unwrap();
        for i in 0..3 {
            assert_eq!(arr2.data.get(i), !arr3.data.get(i));
        }
    }

    #[test]
    fn test_apply_cmp_dict_all_ops_chunk() {
        let a = CategoricalArray::<u32>::from_slices(
            &[0, 1, 2, 3, 1], // All indices in 0..4 for 4 unique values
            &["pad".into(), "dog".into(), "cat".into(), "fish".into()],
        );
        let b = CategoricalArray::<u32>::from_slices(
            &[3, 2, 1, 0, 2], // All indices in 0..4 for 4 unique values
            &["foo".into(), "fish".into(), "cat".into(), "dog".into()],
        );
        // Slice window [1, 2, 3] and [2, 1, 0]
        let a_slice = (&a, 1, 3);
        let b_slice = (&b, 1, 3);

        for &op in &[
            ComparisonOperator::Equals,
            ComparisonOperator::NotEquals,
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqualTo,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::GreaterThanOrEqualTo,
        ] {
            let arr = apply_cmp_dict(a_slice, b_slice, op).unwrap();
            assert_eq!(arr.len, 3);
        }
        let arr = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::Between).unwrap();
        assert_eq!(arr.len, 3);
        let arr2 = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::In).unwrap();
        let arr3 = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::NotIn).unwrap();
        for i in 0..3 {
            assert_eq!(arr2.data.get(i), !arr3.data.get(i));
        }
    }

    #[test]
    fn test_apply_cmp_dict_isnull_isnotnull() {
        let a = CategoricalArray::<u32>::from_slices(&[0, 1], &["x".into(), "y".into()]);
        let b = CategoricalArray::<u32>::from_slices(&[1, 0], &["y".into(), "x".into()]);
        let a_slice = (&a, 0, a.data.len());
        let b_slice = (&b, 0, b.data.len());
        let arr = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::IsNull).unwrap();
        assert_eq!(arr.data.get(0), false);
        let arr = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::IsNotNull).unwrap();
        assert_eq!(arr.data.get(0), true);
    }

    #[test]
    fn test_apply_cmp_dict_isnull_isnotnull_chunk() {
        let a = CategoricalArray::<u32>::from_slices(
            &[2, 0, 1, 2],
            &["z".into(), "x".into(), "y".into()],
        );
        let b = CategoricalArray::<u32>::from_slices(
            &[2, 1, 0, 1],
            &["w".into(), "y".into(), "x".into(), "z".into()],
        );
        let a_slice = (&a, 1, 2);
        let b_slice = (&b, 1, 2);
        let arr = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::IsNull).unwrap();
        assert_eq!(arr.data.get(0), false);
        let arr = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::IsNotNull).unwrap();
        assert_eq!(arr.data.get(0), true);
    }

    #[test]
    #[should_panic(expected = "All indices must be valid for unique_values")]
    fn test_apply_cmp_dict_isnull_isnotnull_chunk_invalid_indices() {
        let a = CategoricalArray::<u32>::from_slices(
            &[9, 0, 1, 9], // 9 is out-of-bounds for 3 unique values
            &["z".into(), "x".into(), "y".into()],
        );
        let b = CategoricalArray::<u32>::from_slices(
            &[2, 1, 0, 3], /* 3 is out-of-bounds for 4 unique values (0..3 is valid, so 3 is valid here) */
            &["w".into(), "y".into(), "x".into(), "z".into()],
        );
        let a_slice = (&a, 1, 2);
        let b_slice = (&b, 1, 2);
        let _ = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::IsNull).unwrap();
        let _ = apply_cmp_dict(a_slice, b_slice, ComparisonOperator::IsNotNull).unwrap();
    }

    // ----------- merge_bitmasks_to_new -----------
    #[test]
    fn test_merge_bitmasks_to_new_none_none() {
        assert!(merge_bitmasks_to_new(None, None, 5).is_none());
    }
    #[test]
    fn test_merge_bitmasks_to_new_some_none() {
        let m = bm(&[true, false, true]);
        let out = merge_bitmasks_to_new(Some(&m), None, 3).unwrap();
        for i in 0..3 {
            assert_eq!(out.get(i), m.get(i));
        }
        let out2 = merge_bitmasks_to_new(None, Some(&m), 3).unwrap();
        for i in 0..3 {
            assert_eq!(out2.get(i), m.get(i));
        }
    }
    #[test]
    fn test_merge_bitmasks_to_new_both_some_and() {
        let a = bm(&[true, false, true, true]);
        let b = bm(&[true, true, false, true]);
        let out = merge_bitmasks_to_new(Some(&a), Some(&b), 4).unwrap();
        assert_eq!(out.get(0), true);
        assert_eq!(out.get(1), false);
        assert_eq!(out.get(2), false);
        assert_eq!(out.get(3), true);
    }
}
