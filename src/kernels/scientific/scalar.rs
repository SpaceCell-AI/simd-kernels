// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Universal Scalar Function Module** - *High-Performance Element-Wise Mathematical Operations*
//!
//! Full suite of vectorised mathematical functions that operate
//! element-wise on arrays of floating-point values. It serves as the computational backbone for
//! mathematical operations across the simd-kernels crate, for both scalar and SIMD-accelerated
//! implementations with opt-in Arrow-compatible null masking.
//!
//! These are the semantic equivalent of *numpy ufuncs* in Python.
//!
//! ## Overview
//!
//! Universal scalar functions are fundamental building blocks for:
//! - **Data Preprocessing**: Normalisations, transformations, and scaling operations
//! - **Scientific Computing**: Mathematical transformations and special function evaluation
//! - **Machine Learning**: Activation functions, feature engineering, and data preparation
//! - **Signal Processing**: Filtering, transforms, and spectral analysis
//! - **Statistics**: Data transformations and statistical preprocessing
//! - **Financial Mathematics**: Risk calculations and price transformations

use crate::kernels::scientific::erf::erf as erf_fn;
use crate::utils::bitmask_to_simd_mask;
use minarrow::utils::is_simd_aligned;
use minarrow::{Bitmask, FloatArray, Vec64};

/// Generates a mapping kernel in three variants:
///
/// 1. `$name_to` - Zero-allocation canonical implementation, writes to caller's buffer
/// 2. `$name` - Allocates internally, delegates to `$name_to`
/// 3. `$name_elem` - Element-wise `fn(f64) -> f64` for kernel fusion
///
/// The `_to` variant is for pre-allocated parallel execution where each chunk
/// writes directly to its slice of a shared output buffer.
///
/// The `_elem` variant is for kernel fusion where multiple operations are composed
/// into a single loop, keeping intermediate values in registers instead of memory.
///
/// `$name`      – allocating function name
/// `$name_to`   – zero-allocation function name
/// `$name_elem` – element-wise function for fusion
/// `$expr`      – expression mapping a scalar `f64 -> f64`
#[macro_export]
macro_rules! impl_vecmap {
    ($name:ident, $name_to:ident, $name_elem:ident, $expr:expr) => {
        /// Element-wise variant for kernel fusion.
        ///
        /// # Example
        /// ```ignore
        /// let ops = &[neg_elem, exp_elem, sin_elem];
        /// execute_fused::<8>(input, output, ops);
        /// // Equivalent to neg -> exp -> sin but with ONE memory read/write
        /// ```
        #[inline(always)]
        pub fn $name_elem(x: f64) -> f64 {
            $expr(x)
        }
        /// Zero-allocation variant: writes directly to caller's output buffer.
        ///
        /// Canonical implementation with full SIMD acceleration and null handling.
        /// For parallel execution with pre-allocated output.
        /// Panics if input.len() != output.len().
        #[inline(always)]
        pub fn $name_to<const LANES: usize>(
            input: &[f64],
            output: &mut [f64],
            null_mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Result<(), &'static str>
        where {
            let len = input.len();
            assert_eq!(
                len,
                output.len(),
                concat!(stringify!($name_to), ": input/output length mismatch")
            );

            if len == 0 {
                return Ok(());
            }
            // decide if we need the null‐aware path
            let has_nulls = match null_count {
                Some(n) => n > 0,
                None => null_mask.is_some(),
            };
            // dense (no nulls) path
            if !has_nulls {
                #[cfg(feature = "simd")]
                {
                    if is_simd_aligned(input) {
                        use core::simd::Simd;
                        let mut i = 0;
                        while i + LANES <= len {
                            let v = Simd::<f64, LANES>::from_slice(&input[i..i + LANES]);
                            let mut r = Simd::<f64, LANES>::splat(0.0);
                            for lane in 0..LANES {
                                r[lane] = $expr(v[lane]);
                            }
                            output[i..i + LANES].copy_from_slice(r.as_array());
                            i += LANES;
                        }
                        // scalar tail
                        for j in i..len {
                            output[j] = $expr(input[j]);
                        }
                        return Ok(());
                    }
                }

                // Scalar fallback
                for j in 0..len {
                    output[j] = $expr(input[j]);
                }
                return Ok(());
            }
            // null‐aware path
            let mb = null_mask.ok_or(concat!(
                stringify!($name_to),
                ": input mask required when nulls present"
            ))?;

            #[cfg(feature = "simd")]
            {
                // Check if input array is properly aligned for SIMD (cheap runtime check)
                if is_simd_aligned(input) {
                    use core::simd::{Mask, Simd};
                    let mask_bytes = mb.as_bytes();
                    let mut i = 0;
                    while i + LANES <= len {
                        // pull in the Arrow validity into a SIMD mask
                        let lane_valid: Mask<i8, LANES> =
                            bitmask_to_simd_mask::<LANES, i8>(mask_bytes, i, len);

                        // Gather inputs (nulls -> NaN)
                        let mut arr = [0.0f64; LANES];
                        for j in 0..LANES {
                            let idx = i + j;
                            arr[j] = if unsafe { lane_valid.test_unchecked(j) } {
                                input[idx]
                            } else {
                                f64::NAN
                            };
                        }
                        let v = Simd::<f64, LANES>::from_array(arr);

                        // Apply your scalar expr in SIMD form
                        let mut r = Simd::<f64, LANES>::splat(0.0);
                        for lane in 0..LANES {
                            r[lane] = $expr(v[lane]);
                        }
                        let r_arr = r.as_array();
                        output[i..i + LANES].copy_from_slice(r_arr);

                        i += LANES;
                    }
                    // scalar tail
                    for idx in i..len {
                        if !unsafe { mb.get_unchecked(idx) } {
                            output[idx] = f64::NAN;
                        } else {
                            output[idx] = $expr(input[idx]);
                        }
                    }

                    return Ok(());
                }
                // Fall through to scalar path if alignment check failed
            }

            // Scalar fallback - alignment check failed
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..len {
                    if !unsafe { mb.get_unchecked(idx) } {
                        output[idx] = f64::NAN;
                    } else {
                        output[idx] = $expr(input[idx]);
                    }
                }
            }
            #[cfg(feature = "simd")]
            {
                for idx in 0..len {
                    if !unsafe { mb.get_unchecked(idx) } {
                        output[idx] = f64::NAN;
                    } else {
                        output[idx] = $expr(input[idx]);
                    }
                }
            }

            Ok(())
        }

        /// Returns a new `FloatArray<f64>` with the function applied element-wise.
        /// Propagates any input nulls (null lanes are not touched).
        #[inline(always)]
        pub fn $name<const LANES: usize>(
            input: &[f64],
            null_mask: Option<&Bitmask>,
            null_count: Option<usize>,
        ) -> Result<FloatArray<f64>, &'static str>
        where {
            let len = input.len();
            // fast length‐0 case
            if len == 0 {
                return Ok(FloatArray::from_slice(&[]));
            }

            let mut out = Vec64::with_capacity(len);
            // SAFETY: we just allocated capacity, extend len to match
            unsafe {
                out.set_len(len);
            }

            $name_to::<LANES>(input, out.as_mut_slice(), null_mask, null_count)?;

            Ok(FloatArray::from_vec64(out, null_mask.cloned()))
        }
    };
}

// Basic operations
impl_vecmap!(abs, abs_to, abs_elem, |x: f64| x.abs());
impl_vecmap!(neg, neg_to, neg_elem, |x: f64| -x);
impl_vecmap!(recip, recip_to, recip_elem, |x: f64| 1.0 / x);
impl_vecmap!(sqrt, sqrt_to, sqrt_elem, |x: f64| x.sqrt());
impl_vecmap!(cbrt, cbrt_to, cbrt_elem, |x: f64| x.cbrt());

// Exponential and logarithmic
impl_vecmap!(exp, exp_to, exp_elem, |x: f64| x.exp());
impl_vecmap!(exp2, exp2_to, exp2_elem, |x: f64| x.exp2());
impl_vecmap!(ln, ln_to, ln_elem, |x: f64| x.ln());
impl_vecmap!(log2, log2_to, log2_elem, |x: f64| x.log2());
impl_vecmap!(log10, log10_to, log10_elem, |x: f64| x.log10());

// Trigonometric
impl_vecmap!(sin, sin_to, sin_elem, |x: f64| x.sin());
impl_vecmap!(cos, cos_to, cos_elem, |x: f64| x.cos());
impl_vecmap!(tan, tan_to, tan_elem, |x: f64| x.tan());
impl_vecmap!(asin, asin_to, asin_elem, |x: f64| x.asin());
impl_vecmap!(acos, acos_to, acos_elem, |x: f64| x.acos());
impl_vecmap!(atan, atan_to, atan_elem, |x: f64| x.atan());

// Hyperbolic
impl_vecmap!(sinh, sinh_to, sinh_elem, |x: f64| x.sinh());
impl_vecmap!(cosh, cosh_to, cosh_elem, |x: f64| x.cosh());
impl_vecmap!(tanh, tanh_to, tanh_elem, |x: f64| x.tanh());
impl_vecmap!(asinh, asinh_to, asinh_elem, |x: f64| x.asinh());
impl_vecmap!(acosh, acosh_to, acosh_elem, |x: f64| x.acosh());
impl_vecmap!(atanh, atanh_to, atanh_elem, |x: f64| x.atanh());

// Error functions
impl_vecmap!(erf, erf_to, erf_elem, |x: f64| erf_fn(x));
impl_vecmap!(erfc, erfc_to, erfc_elem, |x: f64| 1.0 - erf_fn(x));

// Rounding
impl_vecmap!(ceil, ceil_to, ceil_elem, |x: f64| x.ceil());
impl_vecmap!(floor, floor_to, floor_elem, |x: f64| x.floor());
impl_vecmap!(trunc, trunc_to, trunc_elem, |x: f64| x.trunc());
impl_vecmap!(round, round_to, round_elem, |x: f64| x.round());
impl_vecmap!(sign, sign_to, sign_elem, |x: f64| x.signum());

// Activation functions
impl_vecmap!(sigmoid, sigmoid_to, sigmoid_elem, |x: f64| 1.0
    / (1.0 + (-x).exp()));
impl_vecmap!(softplus, softplus_to, softplus_elem, |x: f64| if x > 20.0 {
    x
} else {
    (1.0 + x.exp()).ln()
});
impl_vecmap!(relu, relu_to, relu_elem, |x: f64| if x > 0.0 {
    x
} else {
    0.0
});
impl_vecmap!(gelu, gelu_to, gelu_elem, |x: f64| {
    0.5 * x * (1.0 + erf_fn(x / std::f64::consts::SQRT_2))
});
