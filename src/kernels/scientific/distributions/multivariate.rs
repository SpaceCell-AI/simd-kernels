// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Multivariate Distributions Module** - *Multivariate Statistical Computing*
//!
//! Multivariate probability distribution kernels providing high-performance
//! evaluation of probability density functions, cumulative distributions, and random sampling
//! for complex multivariate statistical models with full BLAS/LAPACK integration.
//!
//! ## Supported Multivariate Distributions
//! - **Multivariate Normal (MVN)**: Gaussian distributions with arbitrary covariance structures  
//! - **Multivariate Student-t**: Heavy-tailed multivariate distributions with configurable degrees of freedom
//! - **Wishart distribution**: Matrix-variate distributions for covariance matrix modelling
//! - **Inverse Wishart**: Conjugate prior distributions for Bayesian covariance estimation
//! - **Multivariate Beta**: Dirichlet and related multivariate beta family distributions
//! - **Matrix-normal distributions**: Matrix-variate Gaussian distributions with Kronecker covariance

use std::f64::consts::PI;

use lapack::{dpotrf, dpotrs};
use minarrow::{Bitmask, FloatArray, Vec64};
use rand::{RngExt, rng};

use crate::kernels::scientific::distributions::{
    shared::sampler::{Sampler, sample_gamma},
    shared::{constants::LN_PI, scalar::ln_gamma},
};
use minarrow::enums::error::KernelError;

/// Standard multivariate probability distributions: PDF, log-PDF, and sampling.
/// Covers all distributions needed for statistical analysis and Bayesian modelling.
/// These are kernel-level, and 2d matrix-level broadcasting must be
/// handled at a higher layer.

/// Evaluates the multivariate normal log-PDF (and PDF) and draws samples,
/// using LAPACK’s dpotrf + dpotrs for the SPD covariance matrix.
///
/// - `x` is a flat slice of length `m * d` (m points in ℝᵈ).
/// - `mean` is length `d`.
/// - `cov` is a `d×d` slice‐of‐slices, row major.
///
/// Returns a FloatArray of length m for `mvn_logpdf` and `mvn_pdf`, or
/// `n_samples` FloatArrays of length d for `mvn_sample`.
fn ensure_square(mean: &[f64], cov: &[&[f64]]) -> Result<usize, KernelError> {
    let d = mean.len();
    if cov.len() != d {
        Err(KernelError::InvalidArguments(
            "mean.len() ≠ cov.len()".into(),
        ))
    } else {
        for row in cov {
            if row.len() != d {
                return Err(KernelError::InvalidArguments("cov must be square".into()));
            }
        }
        Ok(d)
    }
}

/// Multivariate gamma function: Γ_d(a) = π^{d(d-1)/4} ∏_{j=1}^d Γ(a + (1−j)/2)
fn ln_multivariate_gamma(d: usize, a: f64) -> f64 {
    let mut sum = (d * (d - 1)) as f64 / 4.0 * LN_PI;
    for j in 0..d {
        sum += ln_gamma(a + (1.0 - (j as f64 + 1.0)) * 0.5);
    }
    sum
}

/// Compute the multivariate normal log-probability density function (log-PDF).
///
/// Evaluates the logarithm of the multivariate normal probability density function for
/// observations given mean vector and covariance matrix. This function provides numerically
/// stable computation by working directly in log-space, avoiding overflow issues common
/// with high-dimensional distributions.
///
/// ## Mathematical Definition
///
/// The log-PDF of the multivariate normal distribution is:
///
/// ```text
/// log f(x; μ, Σ) = -½[d ln(2π) + ln|Σ| + (x-μ)ᵀ Σ⁻¹ (x-μ)]
/// ```
///
/// where:
/// - `d` is the dimensionality of the distribution
/// - `μ` is the mean vector (d × 1)
/// - `Σ` is the covariance matrix (d × d, positive definite)
/// - `|Σ|` is the determinant of the covariance matrix
///
/// ## Parameters
///
/// * `x` - Flattened observations (m × d elements, row-major)
/// * `mean` - Mean vector of length d
/// * `cov` - Covariance matrix as vector of row slices (d rows, each of length d)
/// * `null_mask` - Currently unsupported, must be None
/// * `null_count` - Currently unsupported, must be None  
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing m log-PDF values, or a `KernelError` if:
/// - Covariance matrix is not positive definite
/// - Dimensions are inconsistent (x.len() not multiple of mean.len())
/// - Null mask parameters are provided (not yet supported)
///
/// ## Examples
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::multivariate::mvn_logpdf;
///
/// // 2D multivariate normal
/// let x = vec64![0.0, 0.0, 1.0, 1.0]; // 2 observations of 2D vectors
/// let mean = vec64![0.0, 0.0];
/// let cov = vec![
///     vec64![1.0, 0.0].as_slice(),
///     vec64![0.0, 1.0].as_slice(),
/// ];
/// let logpdf = mvn_logpdf(&x, &mean, cov, None, None).unwrap();
/// ```
pub fn mvn_logpdf(
    x: &[f64],
    mean: &[f64],
    cov: Vec<&[f64]>,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = ensure_square(mean, &cov)?;
    if x.len() % d != 0 {
        return Err(KernelError::InvalidArguments(
            "x.len() must be a multiple of mean.len()".into(),
        ));
    }
    let m = x.len() / d;

    // Build a column-major copy of Σ
    let mut a = Vec64::with_capacity(d * d);
    for j in 0..d {
        for i in 0..d {
            a.push(cov[i][j]);
        }
    }

    // Cholesky factorization: Σ = L·Lᵀ, lower triangle stored in a
    let mut info = 0;
    unsafe { dpotrf(b'L', d as i32, &mut a, d as i32, &mut info) };
    if info > 0 {
        return Err(KernelError::InvalidArguments(
            "mvn_logpdf: covariance not positive-definite".into(),
        ));
    } else if info < 0 {
        return Err(KernelError::InvalidArguments(
            format!("mvn_logpdf: LAPACK dpotrf arg {} was invalid", -info).into(),
        ));
    }

    // compute log_det = 2 * Σ ln L_ii
    let mut log_det = 0.0;
    for i in 0..d {
        let idx = i + i * d; // (i,i) in column-major a
        log_det += a[idx].ln();
    }
    log_det *= 2.0;

    let ln2pi = (2.0 * PI).ln();
    let mut out = Vec64::with_capacity(m);

    let mut b = vec![0.0f64; d]; // workspace for each solve

    for sample in 0..m {
        // diff = x - mean
        let base = sample * d;
        for i in 0..d {
            b[i] = x[base + i] - mean[i];
        }
        // solve Σ · y = diff  ->  y = Σ⁻¹ diff
        let mut info = 0;
        unsafe { dpotrs(b'L', d as i32, 1, &a, d as i32, &mut b, d as i32, &mut info) };
        if info < 0 {
            return Err(KernelError::InvalidArguments(
                format!("mvn_logpdf: LAPACK dpotrs arg {}", -info).into(),
            ));
        }
        // Mahalanobis = diffᵀ · y
        let mut quad = 0.0;
        for i in 0..d {
            quad += (x[base + i] - mean[i]) * b[i];
        }
        let logpdf = -0.5 * ((d as f64) * ln2pi + log_det + quad);
        out.push(logpdf);
    }

    Ok(FloatArray::from_vec64(out, None))
}

/// Compute the multivariate normal probability density function (PDF).
///
/// Evaluates the multivariate normal probability density function by exponentiating
/// the log-PDF result. For high-dimensional distributions or extreme parameter values,
/// consider using `mvn_logpdf` directly to avoid numerical overflow.
///
/// ## Mathematical Definition
///
/// ```text
/// f(x; μ, Σ) = exp(-½[(x-μ)ᵀ Σ⁻¹ (x-μ)]) / √((2π)ᵈ |Σ|)
/// ```
///
/// This function computes `exp(mvn_logpdf(x, mean, cov))`.
///
/// ## Parameters
///
/// * `x` - Flattened observations (m × d elements, row-major)
/// * `mean` - Mean vector of length d
/// * `cov` - Covariance matrix as vector of row slices
/// * `null_mask` - Currently unsupported, must be None
/// * `null_count` - Currently unsupported, must be None
///
/// ## Returns
///
/// Returns a `FloatArray<f64>` containing m PDF values.
///
/// ## Warning
///
/// For high-dimensional distributions, PDF values may underflow to zero.
/// Use `mvn_logpdf` for numerical stability in such cases.
pub fn mvn_pdf(
    x: &[f64],
    mean: &[f64],
    cov: Vec<&[f64]>,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let logs = mvn_logpdf(x, mean, cov, null_mask, null_count)?;
    let mut data = Vec64::with_capacity(logs.data.len());
    for &v in logs.data.as_slice() {
        data.push(v.exp());
    }
    Ok(FloatArray::from_vec64(data, None))
}

/// Generate samples from a multivariate normal distribution.
///
/// Draws independent samples from the multivariate normal distribution with
/// specified mean vector and covariance matrix using Cholesky decomposition
/// for transformation of independent standard normal random variables.
///
/// ## Parameters
///
/// * `mean` - Mean vector of length d
/// * `cov` - Covariance matrix as vector of row slices (d × d, positive definite)
/// * `n_samples` - Number of samples to generate
/// * `null_mask` - Currently unsupported, must be None
/// * `null_count` - Currently unsupported, must be None
///
/// ## Returns
///
/// Returns a `Vec<FloatArray<f64>>` where each element is a d-dimensional sample.
///
/// ## Examples
///
/// ```rust,ignore
/// use simd_kernels::kernels::scientific::distributions::multivariate::mvn_sample;
///
/// let mean = vec![0.0, 0.0];
/// let cov = vec![
///     vec![1.0, 0.5].as_slice(),
///     vec![0.5, 1.0].as_slice(),
/// ];
/// let samples = mvn_sample(&mean, cov, 100, None, None).unwrap();
/// // Returns 100 samples, each containing 2 elements
/// ```
pub fn mvn_sample(
    mean: &[f64],
    cov: Vec<&[f64]>,
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = ensure_square(mean, &cov)?;
    // Build column-major Σ again
    let mut a = Vec64::with_capacity(d * d);
    for j in 0..d {
        for i in 0..d {
            a.push(cov[i][j]);
        }
    }
    // Cholesky: Σ = L·Lᵀ
    let mut info = 0;
    unsafe { dpotrf(b'L', d as i32, &mut a, d as i32, &mut info) };
    if info > 0 {
        return Err(KernelError::InvalidArguments(
            "mvn_sample: covariance not SPD".into(),
        ));
    } else if info < 0 {
        return Err(KernelError::InvalidArguments(
            format!("mvn_sample: LAPACK dpotrf arg {}", -info).into(),
        ));
    }

    let mut rng = rng();
    let mut outputs = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // draw z ∼ N(0,I)
        let mut z = Vec64::with_capacity(d);
        while z.len() < d {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            z.push(r * theta.cos());
            if z.len() < d {
                z.push(r * theta.sin());
            }
        }
        // compute L · z, where L is the lower triangle in `a`
        let mut out = Vec64::with_capacity(d);
        for i in 0..d {
            let mut s = 0.0;
            for j in 0..=i {
                // a[i + j*d] = L[i,j]
                s += a[i + j * d] * z[j];
            }
            out.push(mean[i] + s);
        }
        outputs.push(FloatArray::from_vec64(out, None));
    }

    Ok(outputs)
}

/// Multivariate Student-t log-pdf:
#[inline]
pub fn mvt_logpdf(
    x: &[f64],
    mean: &[f64],
    scale: Vec<&[f64]>,
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = mean.len();
    // check args
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "mvt_logpdf: df must be positive & finite".into(),
        ));
    }
    if x.len() != d {
        return Err(KernelError::InvalidArguments(
            "mvt_logpdf: x/mean length mismatch".into(),
        ));
    }
    if scale.len() != d {
        return Err(KernelError::InvalidArguments(
            "mvt_logpdf: scale dimension mismatch".into(),
        ));
    }
    for row in &scale {
        if row.len() != d {
            return Err(KernelError::InvalidArguments(
                "mvt_logpdf: scale must be d×d".into(),
            ));
        }
    }

    // 1) flatten Σ into column-major `a`
    let mut a = Vec64::with_capacity(d * d);
    for col in 0..d {
        for row in 0..d {
            a.push(scale[row][col]);
        }
    }
    // 2) Cholesky factor a = L·Lᵀ
    let mut info = 0;
    unsafe { dpotrf(b'L', d as i32, &mut a, d as i32, &mut info) };
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "mvt_logpdf: scale matrix not PD".into(),
        ));
    }
    // 3) log|Σ| = 2·sum ln(diag(L))
    let mut log_det_sigma = 0.0;
    for i in 0..d {
        log_det_sigma += a[i * d + i].ln();
    }
    log_det_sigma *= 2.0;

    // 4) form Σ⁻¹ by solving L·Y = I  then Lᵀ·X = Y
    // build identity on `id`
    let mut id = vec![0.0; d * d];
    for i in 0..d {
        id[i * d + i] = 1.0;
    }
    let mut info2 = 0;
    unsafe {
        dpotrs(
            b'L', d as i32, d as i32, &a, d as i32, &mut id, d as i32, &mut info2,
        )
    };
    if info2 != 0 {
        return Err(KernelError::InvalidArguments(
            "mvt_logpdf: failed to invert scale".into(),
        ));
    }
    // now `id` holds Σ⁻¹ in its lower triangle (and symmetric upper)

    // 5) compute δ = x - μ
    let mut delta = Vec64::with_capacity(d);
    for i in 0..d {
        delta.push(x[i] - mean[i]);
    }
    // 6) compute q = δᵀ Σ⁻¹ δ
    //    first y = Σ⁻¹·δ
    let mut y = vec![0.0; d];
    // matrix-vector multiply (Σ⁻¹ is column-major in id)
    for i in 0..d {
        let mut sum = 0.0;
        for j in 0..d {
            // id[i,j] is at id[j*d + i]
            sum += id[j * d + i] * delta[j];
        }
        y[i] = sum;
    }
    let mut q = 0.0;
    for i in 0..d {
        q += delta[i] * y[i];
    }

    // 7) assemble log-PDF
    let half_df = df * 0.5;
    let half_nu_plus_d = 0.5 * (df + d as f64);
    let ln_coeff = ln_gamma(half_df + d as f64 * 0.5)
        - ln_gamma(half_df)
        - (d as f64) * 0.5 * (df * std::f64::consts::PI).ln()
        - 0.5 * log_det_sigma;
    let ln_pdf = ln_coeff - half_nu_plus_d * (1.0 + q / df).ln();

    Ok(FloatArray::from_vec64(Vec64::from(vec![ln_pdf]), None))
}

/// Multivariate Student-t pdf = exp(logpdf)
#[inline]
pub fn mvt_pdf(
    x: &[f64],
    mean: &[f64],
    scale: Vec<&[f64]>,
    df: f64,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    let logarr = mvt_logpdf(x, mean, scale, df, null_mask, null_count)?;
    // exponentiate the single element
    let v = logarr.data[0].exp();
    Ok(FloatArray::from_vec64(Vec64::from(vec![v]), None))
}

/// Multivariate Student-t sampling:
///   x = μ + (L · z) · sqrt(df / w),  
/// where z ~ N(0,I), w ~ χ²(df), and L·Lᵀ = Σ (scale matrix).
#[cfg(feature = "linear_algebra")]
pub fn mvt_sample(
    mean: &[f64],
    scale: Vec<&[f64]>,
    df: f64,
    n_samples: usize,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    let d = mean.len();
    // --- validate inputs ---
    if d == 0 {
        return Err(KernelError::InvalidArguments(
            "mvt_sample: mean must be non-empty".into(),
        ));
    }
    if scale.len() != d {
        return Err(KernelError::InvalidArguments(
            "mvt_sample: scale must be a square matrix matching mean.len()".into(),
        ));
    }
    for row in &scale {
        if row.len() != d {
            return Err(KernelError::InvalidArguments(
                "mvt_sample: scale must be square".into(),
            ));
        }
    }
    if df <= 0.0 || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "mvt_sample: df must be > 0 and finite".into(),
        ));
    }
    if n_samples == 0 {
        return Ok(Vec::new());
    }

    // --- pack scale matrix into column-major Vec<f64> for LAPACK ---
    let mut a = Vec64::with_capacity(d * d);
    for j in 0..d {
        for i in 0..d {
            a.push(scale[i][j]);
        }
    }

    // --- cholesky factorization in-place: A = L · Lᵀ (lower-triangular) ---
    let mut info = 0;
    unsafe {
        lapack::dpotrf(b'L', d as i32, &mut a, d as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(format!(
            "mvt_sample: Cholesky failed (info={})",
            info
        )));
    }

    let mut sampler = Sampler::new();
    let mut out = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // 1) z ~ N(0, I_d)
        let mut z = Vec64::with_capacity(d);
        for _ in 0..d {
            z.push(sampler.sample_standard_normal());
        }

        // 2) y = L · z  (L is lower-triangular in `a`, col-major)
        let mut y = Vec64::with_capacity(d);
        for i in 0..d {
            let mut sum = 0.0;
            // L[i][j] is stored at a[i + j*d] for j <= i
            for j in 0..=i {
                sum += a[i + j * d] * z[j];
            }
            y.push(sum);
        }

        // 3) w ~ χ²(df)
        let w = sampler.chi2(df);
        let scale_factor = (df / w).sqrt();

        // 4) assemble x = μ + y · scale_factor
        let mut sample = Vec64::with_capacity(d);
        for i in 0..d {
            sample.push(mean[i] + y[i] * scale_factor);
        }
        out.push(FloatArray::from_vec64(sample, None));
    }

    Ok(out)
}

/// Wishart PDF
pub fn wishart_pdf(
    x: Vec<&[f64]>, // each &[f64] is one row of the d×d matrix W
    df: f64,
    scale: Vec<&[f64]>, // Σ rows
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = scale.len();
    if df < d as f64 {
        return Err(KernelError::InvalidArguments(
            "wishart_pdf: df must be ≥ d".into(),
        ));
    }
    // flatten Σ column-major
    let mut a = Vec64::with_capacity(d * d);
    for col in 0..d {
        for row in 0..d {
            a.push(scale[row][col]);
        }
    }
    // factor Σ = L Lᵀ
    let mut info = 0;
    unsafe {
        dpotrf(b'L', d as i32, &mut a, d as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "wishart_pdf: scale not PD".into(),
        ));
    }
    // log|Σ| = 2∑ ln L_{ii}
    let mut log_det_sigma = 0.0;
    for i in 0..d {
        log_det_sigma += a[i * d + i].ln();
    }
    log_det_sigma *= 2.0;

    // precompute Σ⁻¹ via dpotrs on identity
    // invert via multiple right‐hands: I_d
    let mut id = vec![0.0; d * d];
    for i in 0..d {
        id[i * d + i] = 1.0
    }
    let mut info2 = 0;
    unsafe {
        dpotrs(
            b'L', d as i32, d as i32, &a, d as i32, &mut id, d as i32, &mut info2,
        );
    }
    if info2 != 0 {
        return Err(KernelError::InvalidArguments(
            "wishart_pdf: inversion failed".into(),
        ));
    }
    // id now holds Σ⁻¹ in its lower triangle

    // normalisation constant
    let half_df = df * 0.5;
    let ln_norm = half_df * d as f64 * 2.0_f64.ln()  // 2^{νd/2} in denom -> −(νd/2) ln2
        + ln_multivariate_gamma(d, half_df)
        + half_df * log_det_sigma;

    // now for each observation W:
    //   ln|W|  and  tr(Σ⁻¹ W)
    let mut out = Vec64::with_capacity(x.len());
    let mut w_flat = Vec64::with_capacity(d * d);

    for row_slice in x {
        // row_slice is &[f64] of length d*d in row-major
        // compute |W| via det of cholesky
        // flatten into col-major for LAPACK
        w_flat.clear();
        for col in 0..d {
            for row in 0..d {
                w_flat.push(row_slice[row * d + col]);
            }
        }
        // cholesky factor of W
        let mut chol = w_flat.clone();
        let mut info3 = 0;
        unsafe {
            dpotrf(b'L', d as i32, &mut chol, d as i32, &mut info3);
        }
        if info3 != 0 {
            out.push(0.0);
            continue;
        }
        // ln|W| = 2∑ ln diag(chol)
        let mut ln_w = 0.0;
        for i in 0..d {
            ln_w += chol[i * d + i].ln();
        }
        ln_w *= 2.0;

        // trace Σ⁻¹ W = sum_{i,j} (Σ⁻¹)_{ij} W_{ji}
        let mut tr = 0.0;
        for i in 0..d {
            for j in 0..d {
                tr += id[i * d + j] * row_slice[j * d + i];
            }
        }

        // ln f = ½( (ν−d−1) ln|W| − tr(Σ⁻¹W) ) − ln_norm
        let ln_f = 0.5 * ((df - d as f64 - 1.0) * ln_w - tr) - ln_norm;
        out.push(ln_f.exp());
    }

    Ok(FloatArray::from_vec64(out, None))
}

/// Wishart sampler via Bartlett’s decomposition
pub fn wishart_sample(
    df: f64,
    scale: Vec<&[f64]>,
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<Vec64<FloatArray<f64>>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = scale.len();
    if df < d as f64 {
        return Err(KernelError::InvalidArguments(
            "wishart_sample: df must be ≥ d".into(),
        ));
    }
    // Cholesky Σ = L Lᵀ (in place column-major)
    let mut a = Vec64::with_capacity(d * d);
    for col in 0..d {
        for row in 0..d {
            a.push(scale[row][col]);
        }
    }
    let mut info = 0;
    unsafe {
        dpotrf(b'L', d as i32, &mut a, d as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "wishart_sample: scale not PD".into(),
        ));
    }

    let mut samp = Sampler::new();
    let mut out = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // 1) form B
        let mut b = vec![0.0; d * d];
        for i in 0..d {
            // diagonal
            let df_i = df - i as f64;
            b[i * d + i] = samp.chi2(df_i).sqrt();
            // below diag
            for j in 0..i {
                b[i * d + j] = samp.gamma(0.5, 1.0); // actually N(0,1)
                // sampler.gamma isn't normal -> use standard_normal_vec
                // but for speed, just:
                b[i * d + j] = samp.standard_normal_vec(1)[0];
            }
        }
        // 2) compute M = L·B  (both lower triangular)
        //    then W = M·Mᵀ
        let mut m = vec![0.0; d * d];
        // m_ij = ∑_{k=0..i} L_{ik} B_{kj}
        for i in 0..d {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..=j {
                    sum += a[k * d + i] * b[k * d + j];
                }
                m[i * d + j] = sum;
            }
        }
        // W = m·mᵀ
        let mut w = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut sum = 0.0;
                // row i of m, row j of m
                for k in 0..=i.min(j) {
                    sum += m[i * d + k] * m[j * d + k];
                }
                w[i * d + j] = sum;
            }
        }
        // push as FloatArray rows
        let mut rows = Vec64::with_capacity(d);
        for i in 0..d {
            rows.push(FloatArray::from_slice(&w[i * d..(i + 1) * d]));
        }
        out.push(rows);
    }

    Ok(out)
}

/// Inverse-Wishart PDF:
///   f(X) ∝ |Ψ|^{ν/2} / (2^{νp/2} Γₚ(ν/2)) · |X|^{-(ν+p+1)/2} · exp(–½ tr(Ψ X⁻¹))
#[inline(always)]
pub fn inv_wishart_pdf(
    x: Vec<&[f64]>,
    df: f64,
    scale: Vec<&[f64]>,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let p = scale.len();
    if p == 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_pdf: scale must be non-empty".into(),
        ));
    }
    if x.len() != p {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_pdf: x must be same dimension as scale".into(),
        ));
    }
    // each row must have length p
    for row in x.iter().chain(scale.iter()) {
        if row.len() != p {
            return Err(KernelError::InvalidArguments(
                "inv_wishart_pdf: all rows must have length p".into(),
            ));
        }
    }
    // df must exceed p–1
    if df <= (p as f64 - 1.0) || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_pdf: df must be > p-1 and finite".into(),
        ));
    }

    // 1) pack Ψ into column-major for LAPACK
    let mut psi = Vec64::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            psi.push(scale[i][j]);
        }
    }
    // 2) Cholesky Ψ = L·Lᵀ
    let mut info = 0;
    unsafe {
        lapack::dpotrf(b'L', p as i32, &mut psi, p as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_pdf: scale not SPD".into(),
        ));
    }
    // 3) log|Ψ| = 2 ∑ ln L_{ii}
    let mut ln_det_psi = 0.0;
    for i in 0..p {
        ln_det_psi += 2.0 * psi[i + i * p].ln();
    }

    // 4) pack X
    let mut xa = Vec64::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            xa.push(x[i][j]);
        }
    }
    // 5) Cholesky X = Lx·Lxᵀ
    unsafe {
        lapack::dpotrf(b'L', p as i32, &mut xa, p as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_pdf: x not SPD".into(),
        ));
    }
    // 6) log|X| = 2 ∑ ln (Lx_{ii})
    let mut ln_det_x = 0.0;
    for i in 0..p {
        ln_det_x += 2.0 * xa[i + i * p].ln();
    }

    // 7) invert X in place: solve X·Y = I -> Y = X⁻¹
    let mut invx = Vec64::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            invx.push(if i == j { 1.0 } else { 0.0 });
        }
    }
    unsafe {
        lapack::dpotrs(
            b'L', p as i32, p as i32, &mut xa, p as i32, &mut invx, p as i32, &mut info,
        );
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_pdf: failed solving X⁻¹".into(),
        ));
    }

    // 8) tr(Ψ X⁻¹) = ∑_{i,k} Ψ_{i,k} · (X⁻¹)_{k,i}
    let mut trace = 0.0;
    for i in 0..p {
        for k in 0..p {
            // invx is col-major: (k,i) at invx[k + i*p]
            trace += scale[i][k] * invx[k + i * p];
        }
    }

    // 9) form the normalising constant in log-space
    let a = df * 0.5;
    let ln_gamma_p = ln_multivariate_gamma(p, a);
    //   ln_norm = a·ln|Ψ| – (a·p)·ln 2 – ln Γₚ(a)
    let ln_norm = a * ln_det_psi - (a * p as f64) * 2.0f64.ln() - ln_gamma_p;

    // 10) log-pdf = ln_norm – ((df+p+1)/2)·ln|X| – 0.5·tr(Ψ X⁻¹)
    let log_pdf = ln_norm - ((df + p as f64 + 1.0) * 0.5) * ln_det_x - 0.5 * trace;
    let pdf = log_pdf.exp();

    // wrap into a length-1 FloatArray
    let mut out = Vec64::with_capacity(1);
    out.push(pdf);
    Ok(FloatArray::from_vec64(out, None))
}

/// Inverse-Wishart sampling via Bartlett + inversion:
///   1) Compute Ψ⁻¹ = inv(Ψ) via cholesky+solve.
///   2) L = chol(Ψ⁻¹),  
///   3) Bartlett: A_{ii} = √χ²(df – i + 1), A_{ij< i} ~ N(0,1).  
///      Then W = L·A·Aᵀ·Lᵀ ~ Wishart(df, Ψ⁻¹).  
///   4) X = W⁻¹ is an Inv-Wishart(df, Ψ).
pub fn inv_wishart_sample(
    df: f64,
    scale: Vec<&[f64]>,
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<Vec64<FloatArray<f64>>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let p = scale.len();
    if p == 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_sample: scale must be non-empty".into(),
        ));
    }
    for row in &scale {
        if row.len() != p {
            return Err(KernelError::InvalidArguments(
                "inv_wishart_sample: scale must be square".into(),
            ));
        }
    }
    if df <= (p as f64 - 1.0) || !df.is_finite() {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_sample: df must be > p-1 and finite".into(),
        ));
    }
    if n_samples == 0 {
        return Ok(Vec::new());
    }

    // 1) pack Ψ and invert it
    let mut psi = Vec64::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            psi.push(scale[i][j]);
        }
    }
    let mut info = 0;
    unsafe {
        lapack::dpotrf(b'L', p as i32, &mut psi, p as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_sample: scale not SPD".into(),
        ));
    }
    // solve Ψ · Y = I -> Y = Ψ⁻¹
    let mut psi_inv = Vec64::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            psi_inv.push(if i == j { 1.0 } else { 0.0 });
        }
    }
    unsafe {
        lapack::dpotrs(
            b'L',
            p as i32,
            p as i32,
            &mut psi,
            p as i32,
            &mut psi_inv,
            p as i32,
            &mut info,
        );
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_sample: failed to invert scale".into(),
        ));
    }

    // 2) cholesky Ψ⁻¹ = L_inv·L_invᵀ
    unsafe {
        lapack::dpotrf(b'L', p as i32, &mut psi_inv, p as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "inv_wishart_sample: Ψ⁻¹ not SPD?".into(),
        ));
    }

    let mut sampler = Sampler::new();
    let mut all = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // 3) build Bartlett A (lower triangular)
        let mut mat_a = vec![0.0; p * p];
        for i in 0..p {
            // diagonal
            let df_i = df - i as f64;
            let chi = sampler.chi2(df_i);
            mat_a[i + i * p] = chi.sqrt();
            // below diagonal
            for j in 0..i {
                mat_a[i + j * p] = sampler.sample_standard_normal();
            }
        }

        // 4) M = L_inv · A  (both lower triangular)
        let mut mat_m = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..=i {
                // row i of L_inv times col j of A
                let mut sum = 0.0;
                for k in j..=i {
                    sum += psi_inv[i + k * p] * mat_a[k + j * p];
                }
                mat_m[i + j * p] = sum;
            }
        }

        // 5) W = M·Mᵀ
        let mut mat_w = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..=i {
                let mut s = 0.0;
                for k in 0..=i.min(j) {
                    s += mat_m[i + k * p] * mat_m[j + k * p];
                }
                mat_w[i + j * p] = s;
                mat_w[j + i * p] = s;
            }
        }

        // 6) invert W -> X = W⁻¹
        //   factor W = Lw·Lwᵀ
        unsafe {
            lapack::dpotrf(b'L', p as i32, &mut mat_w, p as i32, &mut info);
        }
        if info != 0 {
            return Err(KernelError::InvalidArguments(
                "inv_wishart_sample: W sample not SPD".into(),
            ));
        }
        // prepare eye
        let mut eye = Vec64::with_capacity(p * p);
        for j in 0..p {
            for i in 0..p {
                eye.push(if i == j { 1.0 } else { 0.0 });
            }
        }
        unsafe {
            lapack::dpotrs(
                b'L', p as i32, p as i32, &mut mat_w, p as i32, &mut eye, p as i32, &mut info,
            );
        }
        if info != 0 {
            return Err(KernelError::InvalidArguments(
                "inv_wishart_sample: failed to invert W".into(),
            ));
        }

        // 7) collect rows into FloatArray
        let mut mat = Vec64::with_capacity(p);
        for i in 0..p {
            let mut row = Vec64::with_capacity(p);
            for j in 0..p {
                // eye is X⁻¹ col-major
                row.push(eye[i + j * p]);
            }
            mat.push(FloatArray::from_vec64(row, None));
        }
        all.push(mat);
    }

    Ok(all)
}

// Matrix Normal
/// Compute probability density function of matrix normal distribution.
///
/// Matrix normal distribution MN(M, U, V) where X is m×n, M is mean matrix,
/// U is row covariance, and V is column covariance.
///
/// Currently unimplemented - reserved for future release.
pub fn matrix_normal_pdf(
    _x: Vec<&[f64]>,
    _mean: Vec<&[f64]>,
    _row_cov: Vec<&[f64]>,
    _col_cov: Vec<&[f64]>,
    _null_mask: Option<&Bitmask>,
    _null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    unimplemented!("This function will be implemented in a future release")
}

/// Generate samples from matrix normal distribution.
///
/// Samples from matrix normal distribution MN(M, U, V) where M is mean matrix,
/// U is row covariance, and V is column covariance.
///
/// Currently unimplemented - reserved for future release.
pub fn matrix_normal_sample(
    _mean: Vec<&[f64]>,
    _row_cov: Vec<&[f64]>,
    _col_cov: Vec<&[f64]>,
    _n_samples: usize,
    _null_mask: Option<&Bitmask>,
    _null_count: Option<usize>,
) -> Result<Vec<Vec64<FloatArray<f64>>>, KernelError> {
    unimplemented!("This function will be implemented in a future release")
}

/// Dirichlet PDF
pub fn dirichlet_pdf(
    x: &[f64],
    alpha: &[f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    if x.len() != alpha.len() {
        return Err(KernelError::InvalidArguments(
            "dirichlet_pdf: dimension mismatch".into(),
        ));
    }
    // check support
    let mut sum_x = 0.0;
    for &xi in x {
        if xi < 0.0 || xi > 1.0 {
            return Err(KernelError::InvalidArguments(
                "dirichlet_pdf: x out of [0,1]".into(),
            ));
        }
        sum_x += xi;
    }
    if (sum_x - 1.0).abs() > 1e-12 {
        return Err(KernelError::InvalidArguments(
            "dirichlet_pdf: x must sum to 1".into(),
        ));
    }
    // normalisation
    let mut sum_alpha = 0.0;
    for &a in alpha {
        sum_alpha += a;
    }
    let ln_norm = ln_gamma(sum_alpha) - alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>();

    let mut log_pdf = 0.0;
    for (&xi, &a) in x.iter().zip(alpha.iter()) {
        log_pdf += (a - 1.0) * xi.ln();
    }
    let pdf = (ln_norm + log_pdf).exp();
    Ok(FloatArray::from_vec64(vec![pdf].into(), None))
}

/// Dirichlet sampler
pub fn dirichlet_sample(
    alpha: &[f64],
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let mut samp = Sampler::new();
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let draw = samp.dirichlet(alpha);
        out.push(FloatArray::from_vec64(Vec64::from(draw), None));
    }
    Ok(out)
}

/// Multinomial PMF:
///   P(X = x) = n! / (x₁!⋯xₖ!) · ∏ pᵢˣⁱ
/// where `x.len() == p.len()`, ∑ xᵢ = n, xᵢ ≥ 0 integer, pᵢ ≥ 0, ∑ pᵢ = 1.
#[inline(always)]
pub fn multinomial_pmf(
    x: &[f64],
    n: u64,
    p: &[f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let k = p.len();
    if x.len() != k {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: x and p must have same length".into(),
        ));
    }
    // Validate probabilities
    let sum_p: f64 = p.iter().copied().sum();
    if !(sum_p > 0.0) || (sum_p - 1.0).abs() > 1e-8 {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: p must sum to 1".into(),
        ));
    }
    // Validate counts
    let sum_x: f64 = x.iter().copied().sum();
    if (sum_x - n as f64).abs() > 1e-8 {
        return Err(KernelError::InvalidArguments(
            "multinomial_pmf: ∑ xᵢ must equal n".into(),
        ));
    }
    // Ensure each xᵢ is a non-negative integer
    for &xi in x {
        if xi < 0.0 || (xi.round() - xi).abs() > 1e-8 {
            return Err(KernelError::InvalidArguments(
                "multinomial_pmf: each xᵢ must be a non-negative integer".into(),
            ));
        }
    }
    // Compute log PMF:
    //   ln(n!) - ∑ ln(xᵢ!) + ∑ xᵢ ln(pᵢ)
    let ln_n_fact = ln_gamma(n as f64 + 1.0);
    let mut ln_denom = 0.0;
    let mut sum_lp = 0.0;
    for (&xi, &pi) in x.iter().zip(p.iter()) {
        let xi_f = xi;
        ln_denom += ln_gamma(xi_f + 1.0);
        // pᵢ > 0 required to avoid ln(0)
        if pi <= 0.0 {
            if xi_f > 0.0 {
                // if pᵢ == 0 but xᵢ > 0 then pmf = 0
                return Ok(FloatArray::from_vec64(Vec64::from(vec![0.0_f64]), None));
            }
        } else {
            sum_lp += xi_f * pi.ln();
        }
    }
    let log_pmf = ln_n_fact - ln_denom + sum_lp;
    let pmf = log_pmf.exp();

    // Return a length-1 array containing this PMF
    let mut out = Vec64::with_capacity(1);
    out.push(pmf);
    Ok(FloatArray::from_vec64(out, None))
}

/// Multinomial sampling by repeated categorical draws.
/// Each trial i = 1..n picks one category ∼ Categorical(p), counts up xᵢ.
pub fn multinomial_sample(
    n: u64,
    p: &[f64],
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }
    let k = p.len();
    if k == 0 {
        return Err(KernelError::InvalidArguments(
            "multinomial_sample: p must be non-empty".into(),
        ));
    }
    // Validate probabilities
    let sum_p: f64 = p.iter().copied().sum();
    if !(sum_p > 0.0) || (sum_p - 1.0).abs() > 1e-8 {
        return Err(KernelError::InvalidArguments(
            "multinomial_sample: p must sum to 1".into(),
        ));
    }

    let mut rng = rng();
    let mut out = Vec::with_capacity(n_samples);
    // Precompute cumulative probabilities
    let mut cum: Vec64<f64> = Vec64::with_capacity(k);
    let mut acc = 0.0;
    for &pi in p {
        acc += pi;
        cum.push(acc);
    }
    // Ensure the last entry is exactly 1.0
    cum[k - 1] = 1.0;

    for _ in 0..n_samples {
        // initialise counts
        let mut counts = vec![0u64; k];
        for _ in 0..n {
            let u: f64 = rng.random::<f64>();
            // binary search cum to find category
            let idx = match cum.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
                Ok(i) => i,
                Err(i) => i,
            };
            counts[idx] += 1;
        }
        // convert to FloatArray row
        let mut row = Vec64::with_capacity(k);
        for c in counts {
            row.push(c as f64);
        }
        out.push(FloatArray::from_vec64(row, None));
    }
    Ok(out)
}

/// Categorical PMF:
///   P(X = k) = p[k]
/// where `0 ≤ k < p.len()`, pᵢ ≥ 0, ∑ pᵢ = 1.
#[inline(always)]
pub fn categorical_pmf(
    k: usize,
    p: &[f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    if p.is_empty() {
        return Err(KernelError::InvalidArguments(
            "categorical_pmf: p must be non-empty".into(),
        ));
    }
    if k >= p.len() {
        return Err(KernelError::InvalidArguments(format!(
            "categorical_pmf: category index k={} out of range 0..{}",
            k,
            p.len() - 1
        )));
    }
    // Validate probabilities sum to 1 (within tolerance) and non-negative
    let sum_p: f64 = p.iter().copied().sum();
    if !(sum_p > 0.0) || (sum_p - 1.0).abs() > 1e-8 {
        return Err(KernelError::InvalidArguments(
            "categorical_pmf: p must sum to 1".into(),
        ));
    }
    for (i, &pi) in p.iter().enumerate() {
        if pi < 0.0 {
            return Err(KernelError::InvalidArguments(format!(
                "categorical_pmf: p[{}]={} is negative",
                i, pi
            )));
        }
    }
    // Probability of category k
    let pk = p[k];
    let mut out = Vec64::with_capacity(1);
    out.push(pk);
    Ok(FloatArray::from_vec64(out, None))
}

/// Draw `n_samples` independent samples from a categorical distribution with probabilities `p`.
/// Returns a `Vec<usize>` of category indices.
pub fn categorical_sample(
    p: &[f64],
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<usize>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let k = p.len();
    if k == 0 {
        return Err(KernelError::InvalidArguments(
            "categorical_sample: p must be non-empty".into(),
        ));
    }
    // Validate probabilities sum to 1 (within tolerance) and non-negative
    let sum_p: f64 = p.iter().copied().sum();
    if !(sum_p > 0.0) || (sum_p - 1.0).abs() > 1e-8 {
        return Err(KernelError::InvalidArguments(
            "categorical_sample: p must sum to 1".into(),
        ));
    }
    for (i, &pi) in p.iter().enumerate() {
        if pi < 0.0 {
            return Err(KernelError::InvalidArguments(format!(
                "categorical_sample: p[{}]={} is negative",
                i, pi
            )));
        }
    }

    // Build cumulative distribution
    let mut cum = Vec64::with_capacity(k);
    let mut acc = 0.0;
    for &pi in p {
        acc += pi;
        cum.push(acc);
    }
    // Ensure final value is exactly 1.0
    cum[k - 1] = 1.0;

    let mut rng = rng();
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let u: f64 = rng.random();
        // binary search to find first index i where cum[i] >= u
        let idx = match cum.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
            Ok(i) => i,
            Err(i) => i,
        };
        samples.push(idx);
    }
    Ok(samples)
}

/// Compute probability density function of multivariate Beta (Dirichlet) distribution.
///
/// For input vectors on the simplex (each component ≥ 0, sum = 1),
/// computes f(x; α) = Γ(∑α_i) / ∏Γ(α_i) · ∏ x_i^(α_i - 1).
///
/// Currently unimplemented - reserved for future release.
pub fn multivariate_beta_pdf(
    _x: &[f64],
    _alpha: &[f64],
    _null_mask: Option<&Bitmask>,
    _null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    unimplemented!("This function will be implemented in a future release")
}
/// Generate samples from multivariate Beta (Dirichlet) distribution.
///
/// Generates random samples from the Dirichlet distribution with concentration
/// parameters α, producing vectors on the probability simplex.
///
/// Currently unimplemented - reserved for future release.
pub fn multivariate_beta_sample(
    _alpha: &[f64],
    _n_samples: usize,
    _null_mask: Option<&Bitmask>,
    _null_count: Option<usize>,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    unimplemented!("This function will be implemented in a future release")
}

/// Multivariate log-normal PDF:
/// For each d-dim point in `x` (flattened, length = n_obs * d),
///    f_X(x) = (2π)^{-d/2} |Σ|^{-1/2} exp{ -½ (ln x – μ)^T Σ⁻¹ (ln x – μ) }
///             × ∏_{i=1}^d 1/x_i
pub fn mv_lognormal_pdf(
    x: &[f64],
    mean: &[f64],
    cov: Vec<&[f64]>,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = mean.len();
    if d == 0 {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_pdf: mean must be non-empty".into(),
        ));
    }
    // length of x must be a multiple of d
    if x.len() % d != 0 {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_pdf: x.len() must be multiple of mean.len()".into(),
        ));
    }
    let n_obs = x.len() / d;
    // validate cov is d × d
    if cov.len() != d || cov.iter().any(|row| row.len() != d) {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_pdf: cov must be d×d".into(),
        ));
    }

    // Build a column-major copy of cov into `a`; we'll Cholesky-factor it
    let mut a = Vec64::with_capacity(d * d);
    for j in 0..d {
        for i in 0..d {
            a.push(cov[i][j]);
        }
    }

    // Cholesky: a = L·Lᵀ, lower stored
    let mut info = 0;
    unsafe {
        lapack::dpotrf(b'L', d as i32, &mut a, d as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_pdf: cov not positive-definite".into(),
        ));
    }

    // Compute log-determinant = 2 * ∑ ln(L_ii)
    let mut logdet = 0.0;
    for i in 0..d {
        logdet += (a[i + i * d]).ln();
    }
    logdet *= 2.0;

    // Invert Σ by solving Σ·X = I
    // Prepare B = I (col-major), size d×d
    let mut b = vec![0.0; d * d];
    for i in 0..d {
        b[i + i * d] = 1.0;
    }
    unsafe {
        lapack::dpotrs(
            b'L', d as i32, d as i32, &mut a, d as i32, &mut b, d as i32, &mut info,
        );
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_pdf: failed matrix solve".into(),
        ));
    }
    // Now `b` holds Σ⁻¹ in column-major

    // Precompute constant: lognorm = -½ d ln(2π) - ½ logdet
    let lognorm = -0.5 * (d as f64) * (2.0 * PI).ln() - 0.5 * logdet;

    let mut out = Vec64::with_capacity(n_obs);
    // For each observation
    for obs in 0..n_obs {
        let base = obs * d;
        // compute y = ln(x_obs) - mean; also accumulate ∑ ln(x)
        let mut y = Vec64::with_capacity(d);
        let mut sum_ln_x = 0.0;
        let mut skip = false;
        for i in 0..d {
            let xi = x[base + i];
            if xi <= 0.0 || !xi.is_finite() {
                // PDF zero if any coordinate non-positive or non-finite
                skip = true;
                break;
            }
            sum_ln_x += xi.ln();
            y.push(xi.ln() - mean[i]);
        }
        if skip {
            out.push(0.0);
            continue;
        }
        // z = Σ⁻¹ * y  (col-major b: b[col*d + row] = Σ⁻¹[row, col])
        let mut md2 = 0.0;
        for row in 0..d {
            // compute z_row = ∑_j Σ⁻¹[row,j] * y[j]
            let mut z_row = 0.0;
            for j in 0..d {
                z_row += b[j * d + row] * y[j];
            }
            md2 += y[row] * z_row;
        }
        // logpdf = lognorm - ½ md2 - sum_ln_x
        let logpdf = lognorm - 0.5 * md2 - sum_ln_x;
        out.push(logpdf.exp());
    }

    Ok(FloatArray::from_vec64(out, None))
}

/// Multivariate log-normal sampling:
///   1) draw y ~ MVN( mean, cov ) via Cholesky + standard normals
///   2) x = exp(y)
pub fn mv_lognormal_sample(
    mean: &[f64],
    cov: Vec<&[f64]>,
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = mean.len();
    if d == 0 {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_sample: mean must be non-empty".into(),
        ));
    }
    if cov.len() != d || cov.iter().any(|row| row.len() != d) {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_sample: cov must be d×d".into(),
        ));
    }

    // Build column-major copy of cov
    let mut a = Vec64::with_capacity(d * d);
    for j in 0..d {
        for i in 0..d {
            a.push(cov[i][j]);
        }
    }
    // Cholesky
    let mut info = 0;
    unsafe {
        lapack::dpotrf(b'L', d as i32, &mut a, d as i32, &mut info);
    }
    if info != 0 {
        return Err(KernelError::InvalidArguments(
            "mv_lognormal_sample: cov not positive-definite".into(),
        ));
    }

    let mut rng = rand::rng();
    let mut samples = Vec::with_capacity(n_samples);

    // simple Box–Muller generator for standard normals
    let mut have_extra = false;
    let mut extra = 0.0_f64;

    for _ in 0..n_samples {
        // generate d iid N(0,1)
        let mut z = Vec64::with_capacity(d);
        let mut i = 0;
        while i < d {
            if have_extra {
                z.push(extra);
                have_extra = false;
                i += 1;
                continue;
            }
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            let z0 = r * theta.cos();
            let z1 = r * theta.sin();
            z.push(z0);
            if i + 1 < d {
                z.push(z1);
                i += 2;
            } else {
                // odd last
                extra = z1;
                have_extra = true;
                i += 1;
            }
        }

        // y = mean + L * z, where L is lower-triangular in `a` (col-major):
        // L[i,j] at a[j*d + i] for j<=i
        let mut y = vec![0.0; d];
        for i in 0..d {
            let mut yi = mean[i];
            for j in 0..=i {
                yi += a[j * d + i] * z[j];
            }
            y[i] = yi;
        }
        // x = exp(y)
        let mut x = Vec64::with_capacity(d);
        for yi in y {
            x.push(yi.exp());
        }
        samples.push(FloatArray::from_vec64(x, None));
    }

    Ok(samples)
}

/// Multivariate Γ(shapeᵢ, scaleᵢ) PDF:
/// f( x⃗ ) = ∏ᵢ [ xᵢ^(shapeᵢ−1) exp(−xᵢ/scaleᵢ) / (Γ(shapeᵢ) scaleᵢ^(shapeᵢ)) ].
pub fn multivariate_gamma_pdf(
    x: &[f64],
    shape: &[f64],
    scale: &[f64],
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<FloatArray<f64>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = shape.len();
    if d == 0 {
        return Err(KernelError::InvalidArguments(
            "multivariate_gamma_pdf: empty shape".into(),
        ));
    }
    if scale.len() != d {
        return Err(KernelError::InvalidArguments(
            "multivariate_gamma_pdf: scale.len() != shape.len()".into(),
        ));
    }
    if x.len() % d != 0 {
        return Err(KernelError::InvalidArguments(
            "multivariate_gamma_pdf: x.len() must be multiple of shape.len()".into(),
        ));
    }
    let n_obs = x.len() / d;
    let mut out = Vec64::with_capacity(n_obs);

    for obs in 0..n_obs {
        let mut prod = 1.0;
        let base = obs * d;
        for i in 0..d {
            let xi = x[base + i];
            let alpha = shape[i];
            let theta = scale[i];
            if xi <= 0.0 || !alpha.is_finite() || alpha <= 0.0 || !theta.is_finite() || theta <= 0.0
            {
                prod = 0.0;
                break;
            }
            // univariate gamma PDF
            // = xi^(α−1) * exp(−xi/θ) / (Γ(α) * θ^α)
            let log_pdf =
                (alpha - 1.0) * xi.ln() - xi / theta - ln_gamma(alpha) - alpha * theta.ln();
            prod *= log_pdf.exp();
        }
        out.push(prod);
    }

    Ok(FloatArray::from_vec64(out, None))
}

/// Multivariate Γ(shapeᵢ, scaleᵢ) sampling: independent draws.
pub fn multivariate_gamma_sample(
    shape: &[f64],
    scale: &[f64],
    n_samples: usize,
    null_mask: Option<&Bitmask>,
    null_count: Option<usize>,
) -> Result<Vec<FloatArray<f64>>, KernelError> {
    if null_mask.is_some() || null_count.is_some() {
        return Err(KernelError::InvalidArguments(
            "Null mask support is not yet implemented for multivariate distributions".into(),
        ));
    }

    let d = shape.len();
    if d == 0 {
        return Err(KernelError::InvalidArguments(
            "multivariate_gamma_sample: empty shape".into(),
        ));
    }
    if scale.len() != d {
        return Err(KernelError::InvalidArguments(
            "multivariate_gamma_sample: scale.len() != shape.len()".into(),
        ));
    }
    let mut rng = rng();
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut v = Vec64::with_capacity(d);
        for i in 0..d {
            let alpha = shape[i];
            let theta = scale[i];
            if !(alpha > 0.0 && alpha.is_finite() && theta > 0.0 && theta.is_finite()) {
                return Err(KernelError::InvalidArguments(
                    "multivariate_gamma_sample: invalid parameters".into(),
                ));
            }
            let g = sample_gamma(&mut rng, alpha, theta);
            v.push(g);
        }
        samples.push(FloatArray::from_vec64(v, None));
    }
    Ok(samples)
}
