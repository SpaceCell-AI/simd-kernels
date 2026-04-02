//! # Lognormal Distribution
//!
//! The lognormal distribution models positive random variables whose logarithm follows a
//! normal distribution. It is widely used for modelling multiplicative processes,
//! financial data, and natural phenomena with right-skewed distributions.
//!
//! ## Mathematical Definition
//!
//! Parameterised by meanlog μ and sdlog σ > 0:
//! - **PDF**: f(x; μ, σ) = 1/(xσ√2π) exp(-½[(ln(x)-μ)/σ]²) for x > 0
//! - **CDF**: F(x; μ, σ) = Φ((ln(x)-μ)/σ) where Φ is the standard normal CDF
//! - **Quantile**: Q(p; μ, σ) = exp(μ + σΦ⁻¹(p))
//!
//! ## Applications
//!
//! - **Finance**: Stock prices, income distributions, insurance claims
//! - **Reliability**: Component lifetimes and failure analysis
//! - **Environmental**: Pollutant concentrations, species abundances
//! - **Economics**: Firm sizes, city populations, wealth distributions

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

#[cfg(feature = "simd")]
pub mod simd;
mod std;

use minarrow::enums::error::KernelError;
use minarrow::{Bitmask, FloatArray};

/// Lognormal PDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// f(x|μ,σ) = 1/(xσ√2π) * exp(-½[(ln(x)-μ)/σ]^2)
#[inline(always)]
pub fn lognormal_pdf_to(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::lognormal_pdf_simd_to(x, meanlog, sdlog, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::lognormal_pdf_std_to(x, meanlog, sdlog, output, null_mask, null_count)
    }
}

/// Lognormal CDF (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// F(x|μ,σ) = Φ((ln(x) - μ)/σ)
#[inline(always)]
pub fn lognormal_cdf_to(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::lognormal_cdf_simd_to(x, meanlog, sdlog, output, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::lognormal_cdf_std_to(x, meanlog, sdlog, output, null_mask, null_count)
    }
}

/// Lognormal quantile (zero-allocation variant).
///
/// Writes directly to caller-provided output buffer.
/// Q(p|μ,σ) = exp(μ + σ * Φ⁻¹(p)), p in (0,1)
#[inline(always)]
pub fn lognormal_quantile_to(
    p: &[f64],
    meanlog: f64,
    sdlog: f64,
    output: &mut [f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<(), KernelError> {
    std::lognormal_quantile_std_to(p, meanlog, sdlog, output, null_mask, null_count)
}

/// Compute the probability density function (PDF) for the lognormal distribution.
///
/// The lognormal distribution is a continuous probability distribution of a random variable
/// whose logarithm follows a normal distribution. It is characterised by its positive support
/// and right-skewed shape, making it ideal for modelling multiplicative processes, financial
/// data, and many natural phenomena.
///
/// ## Mathematical Definition
///
/// The PDF of the lognormal distribution is defined as:
///
/// ```text
/// f(x; μ, σ) = 1/(x × σ × √(2π)) × exp(-½[(ln(x) - μ)/σ]²)   for x > 0
/// f(x; μ, σ) = 0                                                for x ≤ 0
/// ```
///
/// where:
/// - `μ` (meanlog) is the mean of the underlying normal distribution (log scale)
/// - `σ` (sdlog) is the standard deviation of the underlying normal distribution, with `σ > 0`
///
/// ## Parameters
///
/// * `x` - Input values to evaluate the PDF at (must be positive for non-zero PDF)
/// * `meanlog` - Mean of the underlying normal distribution (μ, log scale parameter)
/// * `sdlog` - Standard deviation of the underlying normal distribution (σ > 0, log scale)
/// * `null_mask` - Optional bitmask for null value handling in Arrow format
/// * `null_count` - Optional count of null values for optimisation
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing the PDF values, or a `KernelError` if:
/// - Standard deviation parameter is non-positive (σ ≤ 0)
/// - Mean parameter is not finite (NaN or infinite)
/// - Standard deviation parameter is not finite (NaN or infinite)
///
/// ## Examples
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::univariate::lognormal::lognormal_pdf;
/// use minarrow::vec64;
///
/// // Standard lognormal distribution (μ=0, σ=1)
/// let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
/// let pdf = lognormal_pdf(&x, 0.0, 1.0, None, None).unwrap();
/// ```
///
/// ## Applications
///
/// - **Finance**: Stock prices, asset returns, income distributions
/// - **Reliability engineering**: Component lifetimes, time-to-failure analysis
/// - **Environmental science**: Pollutant concentrations, species abundances
/// - **Economics**: Firm sizes, city populations, wealth distributions
/// - **Biology**: Cell sizes, organism masses, reaction times
/// - **Insurance**: Claim amounts, loss distributions
#[inline(always)]
pub fn lognormal_pdf(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::lognormal_pdf_simd(x, meanlog, sdlog, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::lognormal_pdf_std(x, meanlog, sdlog, null_mask, null_count)
    }
}

/// Lognormal CDF: F(x|μ,σ) = Φ((ln(x) - μ)/σ)
#[inline(always)]
pub fn lognormal_cdf(
    x: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    #[cfg(feature = "simd")]
    {
        simd::lognormal_cdf_simd(x, meanlog, sdlog, null_mask, null_count)
    }

    #[cfg(not(feature = "simd"))]
    {
        std::lognormal_cdf_std(x, meanlog, sdlog, null_mask, null_count)
    }
}

/// Lognormal quantile: Q(p|μ,σ) = exp(μ + σ * Φ⁻¹(p)), p in (0,1)
#[inline(always)]
pub fn lognormal_quantile(
    p: &[f64],
    meanlog: f64,
    sdlog: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    std::lognormal_quantile_std(p, meanlog, sdlog, null_mask, null_count)
}

#[cfg(test)]
mod tests {
    use crate::kernels::scientific::distributions::univariate::common::dense_data;

    use super::*;
    use minarrow::{Bitmask, vec64};

    // see "./tests" for scipy test suite

    // Helper utilities shared by the tests

    fn mask_vec(mask: &Bitmask) -> Vec<bool> {
        (0..mask.len()).map(|i| mask.get(i)).collect()
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "assert_close failed: {a} vs {b} (tol = {tol})"
        );
    }

    // Constants
    const MU0: f64 = 0.0;
    const SIG1: f64 = 1.0;

    //  lognormal_pdf - correctness

    #[test]
    fn pdf_reference_values() {
        let x = vec64![0.5, 1.0, 2.0, 4.0];
        // Verified against scipy on 2025-08-14: stats.lognorm.pdf([0.5, 1.0, 2.0, 4.0], s=1.0, scale=exp(0.0))
        let expect = vec64![
            0.627496077115924477,
            0.398942280401432703,
            0.156874019278981117,
            0.038153456511886449,
        ];
        let got = dense_data(lognormal_pdf(&x, MU0, SIG1, None, None).unwrap());
        for (g, e) in got.iter().zip(expect.iter()) {
            assert_close(*g, *e, 1e-15);
        }
    }

    #[test]
    fn pdf_location_scale() {
        // μ = 1, σ = 0.5
        // Verified against scipy on 2025-08-14: stats.lognorm.pdf([1,2,5], s=0.5, scale=exp(1.0))
        let x = vec64![1.0, 2.0, 5.0];
        let expect = vec64![
            0.107981933026376127,
            0.330464565983483946,
            0.075921269497598846,
        ];
        let got = dense_data(lognormal_pdf(&x, 1.0, 0.5, None, None).unwrap());
        for (g, e) in got.iter().zip(expect.iter()) {
            assert_close(*g, *e, 1e-15);
        }
    }

    #[test]
    fn pdf_bulk_vs_scalar() {
        let xs = vec64![0.2, 0.7, 1.5, 5.0];
        let bulk = dense_data(lognormal_pdf(&xs, MU0, SIG1, None, None).unwrap());
        for (idx, &xi) in xs.iter().enumerate() {
            let xi_aligned = vec64![xi];
            let scalar =
                dense_data(lognormal_pdf(xi_aligned.as_slice(), MU0, SIG1, None, None).unwrap())[0];
            assert_close(bulk[idx], scalar, 1e-15);
        }
    }

    #[test]
    fn pdf_negative_and_zero_x() {
        let xs = vec64![-3.0, 0.0, 1.0];
        let got = dense_data(lognormal_pdf(&xs, MU0, SIG1, None, None).unwrap());
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], 0.0);
        assert!(got[2] > 0.0);
    }

    //  lognormal_cdf - correctness
    #[test]
    fn cdf_reference_values() {
        let x = vec64![0.5, 1.0, 2.0, 4.0];
        // Verified against scipy on 2025-08-14: stats.lognorm.cdf([0.5, 1.0, 2.0, 4.0], s=1.0, scale=exp(0.0))
        let expect = vec64![
            0.244108595785582753,
            0.500000000000000000,
            0.755891404214417247,
            0.917171480998301590,
        ];
        let got = dense_data(lognormal_cdf(&x, MU0, SIG1, None, None).unwrap());
        for (g, e) in got.iter().zip(expect.iter()) {
            assert_close(*g, *e, 2e-15);
        }
    }

    #[test]
    fn cdf_left_and_right_tails() {
        let x = vec64![1e-50, 1e+50];
        let got = dense_data(lognormal_cdf(&x, MU0, SIG1, None, None).unwrap());
        assert_close(got[0], 0.0, 1e-308);
        assert_close(got[1], 1.0, 1e-15);
    }

    #[test]
    fn cdf_mask_nan_propagation() {
        let x = vec64![0.3, f64::NAN, 2.0];
        let mut mask = Bitmask::new_set_all(3, true);
        unsafe { mask.set_unchecked(1, false) };
        let arr = lognormal_cdf(&x, MU0, SIG1, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());
        assert!(!nulls[1] && nulls[0] && nulls[2]);
        assert!(arr.data[1].is_nan());
    }

    //  lognormal_quantile - correctness

    #[test]
    fn quantile_reference_values() {
        // SciPy: st.lognorm.ppf([0.001, 0.5, 0.999], 1.0, scale=np.exp(0))
        let p = vec64![0.001, 0.5, 0.999];
        let expect = vec64![0.045_491_385_247_653_51, 1.0, 21.982_183_979_582_84];
        let got = dense_data(lognormal_quantile(&p, MU0, SIG1, None, None).unwrap());
        for (g, e) in got.iter().zip(expect.iter()) {
            assert_close(*g, *e, 2e-14);
        }
    }

    #[test]
    fn quantile_bulk_vs_scalar() {
        let ps = vec64![0.02, 0.2, 0.8, 0.98];
        let bulk = dense_data(lognormal_quantile(&ps, MU0, SIG1, None, None).unwrap());
        for (idx, &pi) in ps.iter().enumerate() {
            let scalar = dense_data(lognormal_quantile(&[pi], MU0, SIG1, None, None).unwrap())[0];
            assert_close(bulk[idx], scalar, 2e-14);
        }
    }

    #[test]
    fn quantile_reflection_roundtrip() {
        // Round-trip:  Q(Φ(ln x)) ≈ x
        let xs = vec64![0.3, 1.0, 5.0, 30.0];
        let cdf = dense_data(lognormal_cdf(&xs, MU0, SIG1, None, None).unwrap());
        let round = dense_data(lognormal_quantile(&cdf, MU0, SIG1, None, None).unwrap());
        for (x, r) in xs.iter().zip(round.iter()) {
            assert_close(*x, *r, 6e-13);
        }
    }

    //  Parameter validation & edge-cases
    #[test]
    fn invalid_parameters_error() {
        assert!(lognormal_pdf(&[1.0], 0.0, 0.0, None, None).is_err());
        assert!(lognormal_cdf(&[1.0], 0.0, -1.0, None, None).is_err());
        assert!(lognormal_quantile(&[0.5], f64::NAN, 1.0, None, None).is_err());
        assert!(lognormal_pdf(&[1.0], f64::NAN, 1.0, None, None).is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(
            lognormal_pdf(&[], MU0, SIG1, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            lognormal_cdf(&[], MU0, SIG1, None, None)
                .unwrap()
                .data
                .is_empty()
        );
        assert!(
            lognormal_quantile(&[], MU0, SIG1, None, None)
                .unwrap()
                .data
                .is_empty()
        );
    }

    //  Mask propagation + NaN / ∞ behaviour
    #[test]
    fn mask_propagation_and_nan_output() {
        let p = vec64![0.3, f64::NAN, 1.2, 0.8]; // 1.2 -> out-of-range -> NaN
        let mut mask = Bitmask::new_set_all(4, true);
        unsafe { mask.set_unchecked(1, false) }; // second element masked

        let arr = lognormal_quantile(&p, MU0, SIG1, Some(&mask), Some(1)).unwrap();
        let nulls = mask_vec(arr.null_mask.as_ref().unwrap());

        // Masked-out lane remains invalid
        assert!(!nulls[1]);
        // Other lanes valid irrespective of NaN value
        assert!(nulls[0] && nulls[2] && nulls[3]);
        assert!(arr.data[1].is_nan()); // propagated
        assert!(arr.data[2].is_nan()); // p = 1.2 -> NaN
    }
}
