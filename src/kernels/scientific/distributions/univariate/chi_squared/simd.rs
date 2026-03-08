// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Chi-Squared Distribution SIMD Implementation**
//!
//! High-performance SIMD-accelerated implementations of chi-squared distribution functions
//! utilising vectorised operations for bulk computations on 64-byte aligned data.
//!
//! ## Overview
//! This module provides SIMD (Single Instruction, Multiple Data) implementations
//! for the chi-squared distribution that process multiple values simultaneously
//! using CPU vector instructions. The implementations automatically fall back to
//! scalar versions when data alignment requirements are not met.
include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::simd::{
    Select, Simd, StdFloat,
    cmp::{SimdPartialEq, SimdPartialOrd},
    num::SimdFloat,
};

use minarrow::{Bitmask, FloatArray, enums::error::KernelError, utils::is_simd_aligned};

use crate::kernels::scientific::distributions::shared::scalar::*;
use crate::kernels::scientific::distributions::univariate::common::simd::{
    dense_univariate_kernel_f64_simd_to, masked_univariate_kernel_f64_simd_to,
};
use crate::kernels::scientific::distributions::univariate::common::std::{
    dense_univariate_kernel_f64_std_to, masked_univariate_kernel_f64_std_to,
};
use crate::utils::has_nulls;

/// SIMD-accelerated chi-squared PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
#[inline(always)]
pub fn chi_square_pdf_simd_to(
    x: &[f64],
    df: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "chi_square_pdf: invalid df".into(),
        ));
    }
    if x.is_empty() {
        return Ok(());
    }

    const N: usize = W64;
    let k2 = 0.5 * df;
    let log_norm = -k2 * std::f64::consts::LN_2 - ln_gamma(k2);

    let k2m1_v = Simd::<f64, N>::splat(k2 - 1.0);
    let log_norm_v = Simd::<f64, N>::splat(log_norm);
    let zero_v = Simd::<f64, N>::splat(0.0);
    let neg_half_v = Simd::<f64, N>::splat(-0.5);

    let scalar_body = move |xi: f64| {
        if xi < 0.0 {
            0.0
        } else if xi == 0.0 && k2 < 1.0 {
            f64::INFINITY
        } else if xi == 0.0 && k2 == 1.0 {
            log_norm.exp()
        } else if xi == 0.0 {
            0.0
        } else if xi.is_infinite() {
            0.0
        } else if xi.is_nan() {
            f64::NAN
        } else {
            (log_norm + (k2 - 1.0) * xi.ln() - 0.5 * xi).exp()
        }
    };

    let simd_body = move |x_v: Simd<f64, N>| {
        let is_negative = x_v.simd_lt(zero_v);
        let is_nan = x_v.simd_ne(x_v);
        let is_infinite = x_v.is_infinite();
        let is_zero = x_v.simd_eq(zero_v);

        let lnx = x_v.ln();
        let expv = neg_half_v * x_v;
        let y = (log_norm_v + k2m1_v * lnx + expv).exp();

        let zero_result = if k2 == 1.0 {
            Simd::<f64, N>::splat(log_norm.exp())
        } else if k2 < 1.0 {
            Simd::<f64, N>::splat(f64::INFINITY)
        } else {
            zero_v
        };

        let nan_v = Simd::<f64, N>::splat(f64::NAN);
        is_nan.select(
            nan_v,
            is_negative.select(
                zero_v,
                is_infinite.select(zero_v, is_zero.select(zero_result, y)),
            ),
        )
    };

    if !has_nulls(null_count, null_mask) {
        if is_simd_aligned(x) {
            dense_univariate_kernel_f64_simd_to::<N, _, _>(x, output, simd_body, scalar_body);
            return Ok(());
        }
        dense_univariate_kernel_f64_std_to(x, output, scalar_body);
        return Ok(());
    }

    let mask_ref = null_mask.expect("chi_square_pdf: null_count > 0 requires null_mask");
    let mut out_mask = mask_ref.clone();
    if is_simd_aligned(x) {
        masked_univariate_kernel_f64_simd_to::<N, _, _>(
            x,
            mask_ref,
            output,
            &mut out_mask,
            simd_body,
            scalar_body,
        );
        return Ok(());
    }
    masked_univariate_kernel_f64_std_to(x, mask_ref, output, &mut out_mask, scalar_body);
    Ok(())
}

/// SIMD-accelerated implementation of chi-squared distribution probability density function.
#[inline(always)]
pub fn chi_square_pdf_simd(
    x: &[f64],
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    use minarrow::Vec64;

    let len = x.len();
    if len == 0 {
        return Ok(FloatArray::from_slice(&[]));
    }

    let mut out = Vec64::with_capacity(len);
    unsafe { out.set_len(len) };

    chi_square_pdf_simd_to(x, df, out.as_mut_slice(), null_mask, null_count)?;

    Ok(FloatArray::from_vec64(out, null_mask.cloned()))
}
