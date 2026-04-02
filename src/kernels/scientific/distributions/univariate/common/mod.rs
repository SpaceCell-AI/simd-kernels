// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Common Distribution Utilities** - *Shared Testing and Helper Infrastructure*
//!
//! Common testing utilities, helper functions, and macros shared across all univariate
//! distribution implementations to ensure consistency and reduce code duplication.
//!
//! ## Testing Infrastructure
//! This module provides standardised testing patterns that validate:
//! - **Numerical accuracy**: Comparison against reference implementations
//! - **Null handling**: Proper propagation of missing values through calculations
//! - **Edge cases**: Boundary conditions and special value handling
//! - **Performance**: Bulk vs scalar operation consistency
//!
//! ## Helper Functions
//! - **Array extraction**: Safe unwrapping of dense arrays without null masks
//! - **Scalar testing**: Single-value operation testing utilities
//! - **Mask creation**: Null mask generation for testing scenarios
//! - **Tolerance checking**: Numerical comparison with configurable precision
//!
//! ## Test Macros
//! The `common_tests!` macro generates standard test suites for distribution functions,
//! ensuring consistent validation across all statistical implementations.

#[cfg(feature = "simd")]
pub mod simd;
/// Scalar implementations of common distribution utilities.
pub mod std;

use minarrow::{Bitmask, Buffer, FloatArray};

// Common test helpers

/// Test Helper: unwrap `FloatArray`, assert *no* null mask, return data.
pub fn dense_data(arr: FloatArray<f64>) -> Buffer<f64> {
    assert!(arr.null_mask.is_none(), "unexpected mask on dense path");
    arr.data
}

/// Build a 1-lane slice (`&[T]`) on the fly, call `kernel`,
/// and return the single f64 result for *scalar* comparison.
pub fn scalar_call<F>(kernel: F, x: f64) -> f64
where
    F: Fn(&[f64]) -> FloatArray<f64>,
{
    dense_data(kernel(&[x])).into_iter().next().unwrap()
}

/// Create a mask of given length with exactly the lane `idx` null.
pub fn single_null_mask(len: usize, idx: usize) -> Bitmask {
    let mut m = Bitmask::new_set_all(len, true);
    unsafe { m.set_unchecked(idx, false) };
    m
}

/// Assert absolute difference ≤ `tol`.
pub fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() < tol,
        "assert_close failed: {} vs {} (tol={})",
        a,
        b,
        tol
    );
}

/// Generate the three most-common tests (empty-input, mask propagation,
/// bulk-vs-scalar) for a `fn kernel(&[f64]) -> FloatArray<f64>`.
///
/// Usage:
/// ```ignore
/// common_tests!(normal_pdf, |x| normal_pdf(x, 0.0, 1.0, None, None).unwrap());
/// ```
#[macro_export]
macro_rules! common_tests {
    // $name    - a unique test-group prefix
    // $call:expr - *closure* that gets &[f64] and returns FloatArray<f64>
    ($name:ident, $call:expr) => {
        mod $name {
            use super::*;
            use crate::tests::common::*;

            #[test]
            fn empty_input() {
                let arr = ($call)(&[]);
                assert!(arr.data.is_empty());
                assert!(arr.null_mask.is_none());
            }

            #[test]
            fn bulk_vs_scalar_consistency() {
                let xs = vec64![-3.0, -1.0, 0.0, 1.0, 2.0];
                let bulk = dense(($call)(&xs));
                for (i, &x) in xs.iter().enumerate() {
                    let scalar = scalar_call($call, x);
                    assert_close(bulk[i], scalar, 1e-14);
                }
            }

            #[test]
            fn mask_propagation() {
                let xs = vec64![1.0, 2.0, 3.0];
                let mask = single_null_mask(3, 1); // middle lane null
                let arr = ($call)(&xs);
                // lane 1 -> NaN + null
                assert!(!arr.null_mask.as_ref().unwrap().get(1));
                assert!(arr.data[1].is_nan());
            }
        }
    };
}
