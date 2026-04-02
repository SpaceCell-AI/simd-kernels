// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # Statistical Sampling Module - High-Performance Pseudorandom Distribution Sampling
//!
//! Pseudorandom number generation kernels providing sampling from statistical
//! distributions with strong performance and distributional accuracy.

use minarrow::Vec64;
use rand::rngs::ThreadRng;
use rand::{Rng, RngExt, rng};
use std::f64::consts::PI;

/// Thread-local statistical distribution sampler backed by a high-quality PRNG.
pub struct Sampler {
    rng: ThreadRng,
}

impl Sampler {
    /// Creates a new sampler instance with a thread-local pseudorandom number generator.
    #[inline]
    pub fn new() -> Self {
        Sampler { rng: rng() }
    }

    /// Generates a single sample from the standard normal distribution N(0, 1).
    #[inline]
    pub fn sample_standard_normal(&mut self) -> f64 {
        sample_standard_normal(&mut self.rng)
    }

    /// Marsaglia-Tsang for Γ(shape, scale). Preconditions: shape > 0, scale > 0.
    #[inline]
    fn sample_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        sample_gamma(&mut self.rng, shape, scale)
    }

    /// Gamma(shape, scale). Preconditions: shape > 0, scale > 0.
    #[inline]
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        self.sample_gamma(shape, scale)
    }

    /// Chi-square(df) == Gamma(df/2, 2). Preconditions: df > 0.
    #[inline]
    pub fn chi2(&mut self, df: f64) -> f64 {
        assert!(df.is_finite() && df > 0.0, "df must be finite and > 0");
        self.gamma(df * 0.5, 2.0)
    }

    /// Vector of iid N(0,1) samples of length `dim`.
    #[inline]
    pub fn standard_normal_vec(&mut self, dim: usize) -> Vec64<f64> {
        let mut v = Vec64::with_capacity(dim);
        for _ in 0..dim {
            v.push(self.sample_standard_normal());
        }
        v
    }

    /// Dirichlet-distributed probability vector via normalised gamma sampling.
    ///
    /// Preconditions: `alpha` non-empty; all entries finite and > 0.
    #[inline]
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec64<f64> {
        assert!(!alpha.is_empty(), "alpha must be non-empty");
        assert!(
            alpha.iter().all(|&a| a.is_finite() && a > 0.0),
            "all alpha entries must be finite and > 0"
        );

        let mut draw = Vec64::with_capacity(alpha.len());
        let mut sum = 0.0;
        for &a in alpha {
            let x = self.gamma(a, 1.0);
            sum += x;
            draw.push(x);
        }
        // With the preconditions above, sum > 0 with probability 1
        draw.iter_mut().for_each(|v| *v /= sum);
        draw
    }
}

// Box-Muller to get one N(0,1)
/// Generates a single sample from the standard normal distribution N(0,1).
#[inline]
pub fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    // U1 ∈ (0,1], U2 ∈ [0,1)
    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE); // avoid log(0)
    let u2: f64 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * PI * u2).cos()
}

/// Generates a single sample from the Gamma distribution using the Marsaglia-Tsang algorithm.
/// Preconditions: shape > 0, scale > 0.
#[inline]
pub fn sample_gamma<R: Rng + ?Sized>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    assert!(
        shape.is_finite() && shape > 0.0,
        "shape must be finite and > 0"
    );
    assert!(
        scale.is_finite() && scale > 0.0,
        "scale must be finite and > 0"
    );

    // Handle 0 < shape < 1 by boosting to shape+1, then apply a power-law correction.
    if shape < 1.0 {
        let u: f64 = rng.random::<f64>();
        return sample_gamma(rng, shape + 1.0, scale) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt(); // Correct: c = 1 / sqrt(9d)

    loop {
        let x = sample_standard_normal(rng);
        let one_plus_cx = 1.0 + c * x;
        if one_plus_cx <= 0.0 {
            continue;
        }
        let v = one_plus_cx * one_plus_cx * one_plus_cx; // (1 + c x)^3
        let u: f64 = rng.random::<f64>();

        // Squeeze step
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v * scale;
        }
        // Log acceptance step
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}
