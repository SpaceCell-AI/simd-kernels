//! # **Scalar Distribution Utilities Module** - *High-Precision Scalar Statistical Functions*
//!
//! Fundamental scalar mathematical functions providing the computational building blocks for
//! statistical distribution calculations with optimal numerical precision and SIMD acceleration.
//! These utilities form the foundation for all distribution PDF, CDF, and quantile computations.

// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

use std::{
    f64::consts::PI,
    simd::{Simd, StdFloat},
};

use crate::kernels::scientific::{
    distributions::shared::constants::*,
    erf::{erfc, erfc_inv},
};

/// Natural log of the absolute value of the Gamma function, ln|Γ(x)|.
///
/// * Aims to match `scipy.special.gammaln` for all real inputs.
/// * Lanczos approximation (g = 7, n = 9) for x ≥ 0.5.
/// * Reflection formula for x < 0.5 using `ln(|sin(πx)|)`.
/// * Poles at non-positive integers return **+∞**.
/// * Propagates NaN.
#[inline(always)]
pub fn ln_gamma(x: f64) -> f64 {
    // Propagate NaN
    if x.is_nan() {
        return f64::NAN;
    }

    // Infinity input: ln_gamma(inf) == inf
    if x.is_infinite() && x.is_sign_positive() {
        return f64::INFINITY;
    }

    // Poles: Γ(x) has simple poles at 0, −1, −2, …  ⇒  ln|Γ| -> +∞
    if x <= 0.0 && (x.fract().abs() < 1e-14) {
        return f64::INFINITY;
    }

    // Reflection branch for  x < 0.5
    //
    // SciPy’s gammaln returns ln|Γ(x)|, hence the absolute value on sin(πx).
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - ln_gamma(1.0 - x);
    }

    // Lanczos approximation for  x ≥ 0.5
    let z = x - 1.0; // shift to minimise cancellation
    let mut a = COF[0];
    for (i, &c) in COF.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    let t = z + 7.5; // g + ½  with g = 7
    HALF_LOG_TWO_PI + (z + 0.5) * t.ln() - t + a.ln()
}

/// ln(k!) = ln_gamma(k+1)
#[inline(always)]
pub fn ln_gamma_plus1(k: f64) -> f64 {
    ln_gamma(k + 1.0)
}

/// Vectorised Lanczos ln Γ for x >= 1.0  (reflection not needed in binomial)
/// Helper due to missing simd helpers in std_lib
#[inline(always)]
pub fn ln_gamma_simd<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N>
where
{
    let z = x - Simd::splat(1.0); // x‐1
    let mut a = Simd::splat(COF[0]); // Σ c₀
    for (i, &c) in COF.iter().enumerate().skip(1) {
        a += Simd::splat(c) / (z + Simd::splat(i as f64));
    }
    let t = z + Simd::splat(7.5); // x-1 + g + 0.5
    let half_ln_two_pi = Simd::splat(0.9189385332046727_f64); // ½ ln(2π)
    half_ln_two_pi + (z + Simd::splat(0.5)) * t.ln() - t + a.ln()
}

/// Inverse of the regularised lower incomplete gamma:
/// finds `x` such that  P(a, x) = p  (a>0, 0≤p≤1).
#[inline(always)]
pub fn inv_reg_lower_gamma(a: f64, p: f64) -> f64 {
    if !(a.is_finite() && p.is_finite()) || a <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Use correct small-x asymptotic:  P(a,x) ~ x^a / Γ(a+1)
    // This works for any a and is essential for tiny p (e.g. 1e-300).
    let mut x = if p < 1e-8 {
        (p * gamma_func(a + 1.0)).powf(1.0 / a).max(1e-300)
    } else if a > 1.0 {
        // Wilson–Hilferty style guess (no normal inverse needed).
        let t = (-2.0 * p.ln()).sqrt();
        let s = (t - (2.0 * a - 1.0).sqrt()) / (2.0 * (a - 1.0).sqrt());
        (a * (1.0 - s + (s * s) / 3.0)).max(1e-300)
    } else {
        (p * gamma_func(a + 1.0)).powf(1.0 / a).max(1e-300)
    };

    // ---- Newton refinement ----------------------------------------------
    const MAX_ITERS: usize = 80;
    const REL_TOL: f64 = 1e-15;
    const ABS_FLOOR: f64 = 5e-16;

    for _ in 0..MAX_ITERS {
        let f = reg_lower_gamma(a, x) - p; // F(x)=P(a,x)-p
        let fp = gamma_pdf_scalar(a, x); // F'(x)=f(a,x)
        if !fp.is_finite() || fp.abs() < 1e-300 {
            break;
        }
        let mut dx = f / fp;

        let max_step = x.max(1.0); // a bit looser than 0.5*x
        if dx.abs() > max_step {
            dx = max_step * dx.signum();
        }

        let x_new = (x - dx).max(1e-300);
        let tol = (REL_TOL * (1.0 + x_new.abs())).max(ABS_FLOOR);
        if (x_new - x).abs() <= tol {
            x = x_new;
            break;
        }
        x = x_new;
    }

    let f = reg_lower_gamma(a, x) - p;
    let fp = gamma_pdf_scalar(a, x);
    if fp.is_finite() && fp.abs() > 0.0 && x.is_finite() && x > 0.0 {
        let fpp = fp * ((a - 1.0) / x - 1.0); // derivative of pdf
        let h = f / fp;
        let denom = 1.0 - 0.5 * h * (fpp / fp);
        if denom.is_finite() && denom != 0.0 {
            let xh = (x - h / denom).max(1e-300);
            // only accept if it actually improves:
            if (reg_lower_gamma(a, xh) - p).abs() <= f.abs() {
                x = xh;
            }
        }
    }

    x
}

/// Inverse of the regularised upper incomplete gamma:
/// finds `x` such that  Q(a, x) = q  (a>0, 0≤q≤1), Q=1-P.
#[inline(always)]
pub fn inv_reg_upper_gamma(a: f64, q: f64) -> f64 {
    if !(a.is_finite() && q.is_finite()) || a <= 0.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        return f64::INFINITY;
    }
    if q >= 1.0 {
        return 0.0;
    }

    let p = 1.0 - q;

    // ---- Initial guess ---------------------------------------------------
    // For extreme right tail, this bias works well; for moderate q Newton fixes it.
    let mut x = if q > 1e-8 && a > 1.0 {
        let t = (-2.0 * q.ln()).sqrt();
        let s = (t - (2.0 * a - 1.0).sqrt()) / (2.0 * (a - 1.0).sqrt());
        (a * (1.0 - s + (s * s) / 3.0)).max(1e-300)
    } else {
        // When 1-p is tiny (i.e., p≈1), start from tiny-x for the *complement*.
        (p * gamma_func(a + 1.0)).powf(1.0 / a).max(1e-300)
    };

    // ---- Newton refinement on F(x)=Q(a,x)-q = 1-P(a,x)-q ----------------
    const MAX_ITERS: usize = 80;
    const REL_TOL: f64 = 1e-15;
    const ABS_FLOOR: f64 = 5e-16;

    for _ in 0..MAX_ITERS {
        let f = (1.0 - reg_lower_gamma(a, x)) - q; // F(x)=Q-q
        let fp = -gamma_pdf_scalar(a, x); // F'(x)=-pdf
        if !fp.is_finite() || fp.abs() < 1e-300 {
            break;
        }
        let mut dx = f / fp;

        let max_step = x.max(1.0);
        if dx.abs() > max_step {
            dx = max_step * dx.signum();
        }

        let x_new = (x - dx).max(1e-300);
        let tol = (REL_TOL * (1.0 + x_new.abs())).max(ABS_FLOOR);
        if (x_new - x).abs() <= tol {
            x = x_new;
            break;
        }
        x = x_new;
    }

    // Halley polish
    let f = (1.0 - reg_lower_gamma(a, x)) - q;
    let fp = -gamma_pdf_scalar(a, x);
    if fp.is_finite() && fp.abs() > 0.0 && x.is_finite() && x > 0.0 {
        let fpp = -fp * ((a - 1.0) / x - 1.0); // derivative of -pdf
        let h = f / fp;
        let denom = 1.0 - 0.5 * h * (fpp / fp);
        if denom.is_finite() && denom != 0.0 {
            let xh = (x - h / denom).max(1e-300);
            if ((1.0 - reg_lower_gamma(a, xh)) - q).abs() <= f.abs() {
                x = xh;
            }
        }
    }

    x
}

/// Regularised lower incomplete gamma P(a, x)
///
/// Edge cases:
/// * `x < 0`              -> NaN
/// * `a  < 0`             -> NaN
/// * `a == 0` & x ≥ 0     -> 1.0
/// * `x == 0` & a  > 0    -> 0.0
/// * any NaN argument     -> NaN
#[inline(always)]
pub fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    // Propagate NaNs first
    if !(a.is_finite() && x.is_finite()) {
        return f64::NAN;
    }
    // Domain-error branches -----------------------------------------------
    if x < 0.0 {
        return f64::NAN;
    }
    if a < 0.0 {
        return f64::NAN;
    }
    if a == 0.0 {
        return 1.0;
    } // gammainc(0, x) == 1 for x ≥ 0
    if x == 0.0 {
        return 0.0;
    } // positive a, zero x

    if x <= 0.0 {
        return 0.0;
    }
    if a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series representation
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..100 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        (-(x) + a * x.ln() - ln_gamma(a)).exp() * sum
    } else {
        // Continued fraction (Lentz's method)
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / f64::MIN_POSITIVE;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100 {
            let an = -i as f64 * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-14 {
                break;
            }
        }
        1.0 - (-(x) + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Scalar gamma PDF for Newton refinement
#[inline(always)]
pub fn gamma_pdf_scalar(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - ln_gamma(a)).exp()
}

//// Computes the Gamma function Γ(x).
///  
/// Special cases:
/// * `x = 0`           -> `+∞`
/// * `x ∈ ℤ⁻` (negative integer) -> `NaN`
/// * `x = n`   (1 ≤ n ≤ 170, integer) -> exact `(n-1)!` via lookup
/// * `x = n+½` (0 ≤ n ≤ 170)         -> closed-form half-integer Gamma
/// * otherwise  
///   * if `x > 0`  -> Lanczos via `exp(ln_gamma(x))`
///   * if `x < 0`  -> reflection  Γ(x)=π / [sin(πx) Γ(1−x)]
#[inline(always)]
pub fn gamma_func(x: f64) -> f64 {
    // NaN propagates
    if x.is_nan() {
        return f64::NAN;
    }

    // 0 -> +∞    (SciPy)
    if x == 0.0 {
        return f64::INFINITY;
    }

    // Negative integers (simple poles) -> NaN   (SciPy)
    if x < 0.0 && (x.fract().abs() < 1e-14) {
        return f64::NAN;
    }

    // ----------------  Positive region fast paths  ---------------- //
    if x > 0.0 {
        // Exact factorials  Γ(n) = (n-1)!   for 1 ≤ n ≤ 171
        if x.fract().abs() < 1e-14 {
            let n = x as usize;
            if n >= 1 && n <= 171 {
                // (n-1)!  ;  lookup table holds 0! … 170!
                return factorial_lookup((n - 1) as u64);
            }
        }

        // Positive half-integers  Γ(n+½)
        if (x - 0.5).fract().abs() < 1e-14 {
            let n = (x - 0.5).round() as u64; // n ≥ 0
            if n <= 170 {
                return half_integer_gamma(n);
            }
        }

        // General positive x  — Lanczos via lnΓ
        return ln_gamma(x).exp();
    }

    // ----------------  x < 0  (non-integer)  ---------------- //
    //
    // Reflection formula:
    //     Γ(x) = π / [ sin(πx) · Γ(1 − x) ]
    let sin_pi_x = (std::f64::consts::PI * x).sin();

    // If sin(πx) == 0 the point would be a pole; but we already ruled
    // out integer arguments above, so a tiny guard is sufficient.
    if sin_pi_x.abs() < f64::EPSILON {
        return f64::NAN; // should never be hit, keeps us safe
    }

    // Γ(1−x) is called with a positive argument (because x<0 ⇒ 1−x>1),
    // so the recursion bottoms out in the positive fast-paths.
    let gamma_1_minus_x = gamma_func(1.0 - x);

    std::f64::consts::PI / (sin_pi_x * gamma_1_minus_x)
}

/// Evaluates gamma function at half-integer arguments using closed-form expression.
///
/// Computes Γ(n + 1/2) for non-negative integer n using the exact closed-form
/// formula involving factorials and powers.
#[inline(always)]
pub fn half_integer_gamma(n: u64) -> f64 {
    let two_n_fact = factorial_lookup(2 * n);
    let n_fact = factorial_lookup(n);
    let pow4n = 2f64.powi(2 * n as i32); // 4^n = 2^(2n)
    (PI.sqrt() * two_n_fact) / (pow4n * n_fact)
}

/// Regularised incomplete beta I_x(a, b).
///
///   * `a == 0`  ->  1.0  (mass entirely to the right of x)
///   * `b == 0`  ->  0.0  (mass entirely at the left of x)
///   * non-finite inputs propagate `NaN`
///   * x ≤ 0 -> 0 ··· x ≥ 1 -> 1
#[inline(always)]
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    // Handle NaNs first
    if !(a.is_finite() && b.is_finite() && x.is_finite()) {
        return f64::NAN;
    }

    // Domain edges -----------------------------------------------------------
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // a = 0 or b = 0  (SciPy conventions)
    if a == 0.0 {
        return 1.0; // all probability mass to the right of any x > 0
    }
    if b == 0.0 {
        return 0.0; // all mass at x = 0, so CDF is identically 0
    }

    // -----------------------------------------------------------------------
    //    Regular computation
    // -----------------------------------------------------------------------
    // Use symmetry for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        return 1.0 - incomplete_beta(b, a, 1.0 - x);
    };

    // front factor :  x^a (1-x)^b / (a * B(a,b))
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Lentz's continued fraction
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-300;
    const MAX_ITS: usize = 200;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITS {
        let m2 = 2 * m;

        // -------- even step
        let aa = m as f64 * (b - m as f64) * x / ((a + m2 as f64 - 1.0) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // -------- odd step
        let aa =
            -(a + m as f64) * (a + b + m as f64) * x / ((a + m2 as f64) * (a + m2 as f64 + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    front * h
}

/// Inverse regularised incomplete beta I_x(a, b)
///
/// Special cases:
///   * Any non-finite input -> NaN
///   * a == 0 -> returns 1.0  (if p < 1)  or NaN if p out of [0,1]
///   * b == 0 -> returns 0.0  (if p > 0)  or NaN if p out of [0,1]
/// Inverse regularised incomplete beta I_x(a, b)
///
/// * Any non-finite input -> NaN
/// * a == 0 -> 1.0   (if 0 ≤ p ≤ 1)
/// * b == 0 -> 0.0   (if 0 ≤ p ≤ 1)
#[inline(always)]
pub fn incomplete_beta_inv(a: f64, b: f64, p: f64) -> f64 {
    // Fast parameter & domain checks
    if !(a.is_finite() && b.is_finite() && p.is_finite()) {
        return f64::NAN;
    }
    if p < 0.0 || p > 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }

    if a == 0.0 {
        return 1.0;
    } // mass jumps to the right of 0
    if b == 0.0 {
        return 0.0;
    } // mass concentrated at 0

    // Initial Cornish-Fisher / Wilson–Hilferty seed
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let mut x: f64;

    if a > 1.0 && b > 1.0 {
        // central region (both shape parameters > 1)
        let num = 2.30753 + 0.27061 * t;
        let den = 1.0 + (0.99229 + 0.04481 * t) * t;
        let mut xp = t - num / den;
        if p < 0.5 {
            xp = -xp;
        }

        let al = (xp * xp - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = xp * (al + h).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        x = a / (a + b * w.exp());
    } else {
        // one of the shapes ≤ 1 : use power-law seed
        let r = a / (a + b);
        let y = if p < r { p / r } else { (1.0 - p) / (1.0 - r) };
        x = if p < r {
            y.powf(1.0 / a)
        } else {
            1.0 - y.powf(1.0 / b)
        };
    }

    // Clamp to open interval (0,1) – avoids log/derivative blow-ups
    const EPS_X: f64 = 1e-15;
    if x <= 0.0 {
        x = EPS_X;
    }
    if x >= 1.0 {
        x = 1.0 - EPS_X;
    }

    // Newton iterations
    const NEWTON_EPS: f64 = 1e-14;
    const MAX_NEWTON: usize = 20;

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);

    for _ in 0..MAX_NEWTON {
        let f = incomplete_beta(a, b, x) - p;

        // derivative:  x^(a-1) (1-x)^(b-1) / B(a,b)
        let df = ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta).exp();

        // if derivative underflows, break to bisection
        if df == 0.0 || !df.is_finite() {
            break;
        }

        let dx = f / df;
        let mut x_new = x - dx;

        // Keep within (0,1)
        if x_new <= 0.0 {
            x_new = 0.5 * x;
        }
        if x_new >= 1.0 {
            x_new = 0.5 * (1.0 + x);
        }

        if (dx).abs() < NEWTON_EPS * x_new.max(EPS_X) {
            return x_new;
        }
        x = x_new;
    }

    // Bracketed bisection – fallback
    let mut lo = 0.0;
    let mut hi = 1.0;

    if incomplete_beta(a, b, x) > p {
        hi = x;
    } else {
        lo = x;
    }

    for _ in 0..64 {
        // 2⁻⁶⁴  <  5e-20  ⇒ double precision satisfied
        let mid = 0.5 * (lo + hi);
        let f_mid = incomplete_beta(a, b, mid);

        if (f_mid - p).abs() < 1e-14 {
            // SciPy-level tolerance
            return mid;
        }
        if f_mid < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // final midpoint
    0.5 * (lo + hi)
}

/// Regularised lower incomplete gamma (series + continued fraction).
#[inline(always)]
pub fn regularised_gamma_p(s: f64, x: f64) -> f64 {
    // s > 0, x >= 0
    if x <= 0.0 {
        return 0.0;
    }
    if x < s + 1.0 {
        // Series representation
        let mut sum = 1.0 / s;
        let mut value = sum;
        let mut n = 1.0;
        while value.abs() > 1e-15 * sum {
            value *= x / (s + n);
            sum += value;
            n += 1.0;
            if n > 200.0 {
                break;
            }
        }
        (-(x) + s * x.ln() - ln_gamma(s)).exp() * sum
    } else {
        // Continued fraction representation
        let mut a = 1.0 - s;
        let mut b = a + x + 1.0;
        let mut c = 0.0;
        let mut p1 = 1.0;
        let mut q1 = x;
        let mut p2 = x + 1.0;
        let mut q2 = b * x;
        let mut ans = p2 / q2;
        let mut n = 1.0;
        while (p2 - p1).abs() > 1e-15 * ans.abs() {
            a += 1.0;
            b += 2.0;
            c += 1.0;
            let an = a * c;
            let p3 = b * p2 - an * p1;
            let q3 = b * q2 - an * q1;
            if q3.abs() < 1e-30 {
                break;
            }
            p1 = p2;
            q1 = q2;
            p2 = p3;
            q2 = q3;
            if q2 != 0.0 {
                ans = p2 / q2;
            }
            n += 1.0;
            if n > 200.0 {
                break;
            }
        }
        1.0 - (-(x) + s * x.ln() - ln_gamma(s)).exp() * ans
    }
}

/// High-performance vectorised logarithmic binomial coefficient computation.
///
/// Computes ln(C(n,k)) = ln(n! / (k!(n-k)!)) for vectors of n and k values
/// using SIMD vectorisation and optimised gamma function evaluation.
#[inline(always)]
pub fn ln_choose_v(
    n: core::simd::Simd<f64, W64>,
    k: core::simd::Simd<f64, W64>,
) -> core::simd::Simd<f64, W64> {
    ln_gamma_simd(n + core::simd::Simd::<f64, W64>::splat(1.0))
        - ln_gamma_simd(k + core::simd::Simd::<f64, W64>::splat(1.0))
        - ln_gamma_simd(n - k + core::simd::Simd::<f64, W64>::splat(1.0))
}

/// Computes logarithmic binomial coefficient for integer arguments with validation.
///
/// Evaluates ln(C(n,k)) = ln(n! / (k!(n-k)!)) for non-negative integer arguments
/// using gamma function evaluation.
#[inline(always)]
pub fn ln_choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    } // log(0) for invalid (impossible) binomials
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

/// Generic SIMD logarithmic binomial coefficient with compile-time lane count.
///
/// Computes ln(C(n,k)) for vectors of n and k values using SIMD vectorisation
/// with arbitrary lane counts determined at compile time.
#[inline(always)]
pub fn ln_choose_simd<const N: usize>(n: Simd<f64, N>, k: Simd<f64, N>) -> Simd<f64, N>
where
{
    ln_gamma_simd(n + Simd::splat(1.0))
        - ln_gamma_simd(k + Simd::splat(1.0))
        - ln_gamma_simd(n - k + Simd::splat(1.0))
}

/// Compute the Cornish–Fisher-based binomial quantile for a single `pi`.
/// Returns `f64::NAN` for any out-of-range or non-finite input.
/// Does not handle nulls; caller is responsible.
#[inline(always)]
pub fn binomial_quantile_cornish_fisher(pi: f64, n: u64, p_: f64) -> f64 {
    // Edge cases per scipy convention
    if !pi.is_finite() || !p_.is_finite() || n > (i64::MAX as u64) {
        return f64::NAN;
    }
    if pi <= 0.0 {
        return -1.0;
    }
    if pi >= 1.0 {
        return n as f64;
    }
    let mu = n as f64 * p_;
    let sigma = (n as f64 * p_ * (1.0 - p_)).sqrt();
    let z = normal_quantile_scalar(pi, 0.0, 1.0);
    let skew = (1.0 - 2.0 * p_) / sigma;
    let cf = z + skew * (z * z - 1.0) / 6.0;
    let mut k = (mu + sigma * cf + 0.5).floor();
    if k < 0.0 {
        k = 0.0;
    }
    if k > n as f64 {
        k = n as f64;
    }
    let mut k_int = k as i64;
    let cdf = binomial_cdf_scalar(k_int, n, p_);
    if cdf < pi {
        while k_int < (n as i64) && binomial_cdf_scalar(k_int, n, p_) < pi {
            k_int += 1;
        }
    } else {
        while k_int > 0 && binomial_cdf_scalar(k_int - 1, n, p_) >= pi {
            k_int -= 1;
        }
    }
    k_int as f64
}

/// Scalar binomial CDF
///
/// * k < 0  -> 0  
/// * k ≥ n  -> 1  
/// * p ≤ 0  -> 1 for k ≥ 0  
/// * p ≥ 1  -> 0 for k < n, 1 for k ≥ n
///
/// Arguments:
///     k: i64 - Number of observed successes (can be negative).
///     n: u64 - Number of trials.
///     p: f64 - Probability of success (0 <= p <= 1).
///
/// Accuracy:
/// - This function aims to match the output of `scipy.stats.binom`
/// with a maximum absolute error of `1e-12` for all reasonable arguments.
/// - For most cases tested, that error is within < `1e-14`.
#[inline(always)]
pub fn binomial_cdf_scalar(k: i64, n: u64, p: f64) -> f64 {
    // ---- Domain short-cuts ------------------------------------------------
    if k < 0 {
        return 0.0;
    } // left of support
    if k as u64 >= n {
        return 1.0;
    } // CDF reaches 1 at n
    if p <= 0.0 {
        return 1.0;
    } // all mass at 0
    if p >= 1.0 {
        return 0.0;
    } // all mass at n (but k < n here)

    // ---- Regular summation (k < n, 0 < p < 1) -----------------------------
    let k = k as u64;
    let mut cdf = 0.0_f64;
    let mut prob = (1.0 - p).powf(n as f64); // P(X = 0)
    cdf += prob;

    for i in 1..=k as usize {
        // recurrence:  P(X = i) = P(X = i-1) * (n-i+1)/i * p/(1-p)
        prob *= ((n - i as u64 + 1) as f64) * p / ((i as f64) * (1.0 - p));
        cdf += prob;
    }
    cdf
}

/// Core inverse standard normal function for left tail probabilities.
///
/// Computes Φ⁻¹(p) for probabilities p ∈ (0, 0.5] using Acklam's rational
/// approximation optimised for the left tail region.
#[inline(always)]
pub fn inv_std_normal_core(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p <= 0.5);

    if p > P_LOW {
        // ---------------- central region ----------------
        let r = p - 0.5;
        let s = r * r;
        let num = (((((A[0] * s + A[1]) * s + A[2]) * s + A[3]) * s + A[4]) * s + A[5]) * r;
        let den = ((((B[0] * s + B[1]) * s + B[2]) * s + B[3]) * s + B[4]) * s + 1.0;
        num / den
    } else {
        // ---------------- lower tail --------------------
        let r = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5];
        let den = (((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0;
        //  NOTE:  `num` is already negative here, so we do *not*
        //  apply an extra minus sign.
        num / den // ⇒  negative z-score
    }
}

/// Computes the inverse standard normal cumulative distribution function (quantile function).
///
/// Calculates the z-score corresponding to a given probability p such that Φ(z) = p,
/// where Φ is the standard normal CDF. Uses high-precision rational approximations
/// for numerical accuracy across the entire probability range.
///
/// # Parameters
/// - `p`: Probability value in the range [0, 1] for which to compute the quantile
///
/// # Returns
/// The z-score (quantile) such that the standard normal CDF evaluated at z equals p.
///
/// # Domain and Range
/// - **Domain**: p ∈ [0, 1]
/// - **Range**: z ∈ (-∞, ∞)
/// - **Special cases**:
///   - `p = 0.0` returns `-∞`
///   - `p = 1.0` returns `+∞`
///   - `p = 0.5` returns `0.0`
#[inline(always)]
pub fn inv_std_normal(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let (q, sign) = if p < 0.5 { (p, 1.0) } else { (1.0 - p, -1.0) };
    let x = if q < 0.02425 {
        let t = (-2.0 * q.ln()).sqrt();
        (((((C[0] * t + C[1]) * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5])
            / ((((D[0] * t + D[1]) * t + D[2]) * t + D[3]) * t + 1.0)
    } else {
        let t = q - 0.5;
        let r = t * t;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * t
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    sign * x
}

/// Specialised Newton refinement for extreme chi-squared quantile computation.
///
/// High-precision iterative refinement method specifically optimised for
/// extreme chi-squared quantiles where standard methods may suffer from
/// numerical instability. Uses extended iteration counts and tighter
/// convergence tolerances to achieve maximum accuracy in challenging regions.
#[inline(always)]
pub fn chi2_newton_refine_extreme(mut x: f64, a: f64, p: f64) -> f64 {
    // Specialised Newton refinement for extreme chi-squared quantiles
    // Uses more iterations and tighter tolerance for extreme probabilities
    let ln_norm = -a * std::f64::consts::LN_2 - ln_gamma(a);
    let mut lo = 0.0_f64;
    let mut hi = f64::INFINITY;
    for _ in 0..16 {
        if !x.is_finite() {
            break;
        }
        let t = 0.5 * x;
        let fx = reg_lower_gamma(a, t) - p;
        if fx.abs() < 1e-15 {
            break;
        }

        let log_pdf = ln_norm + (a - 1.0) * x.ln() - 0.5 * x;
        let pdf = log_pdf.exp();

        if pdf <= 0.0 || !pdf.is_finite() {
            if fx > 0.0 {
                hi = hi.min(x);
            } else {
                lo = lo.max(x);
            }
            x = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                (x + lo.max(0.0)) * 0.5
            };
            continue;
        }

        let mut step = fx / pdf;
        let max_step = 0.5 * (x.max(1.0));
        if step.abs() > max_step {
            step = step.signum() * max_step;
        }

        let x_new = (x - step).max(0.0);
        if fx > 0.0 {
            hi = hi.min(x);
        } else {
            lo = lo.max(x);
        }
        if x_new < lo || x_new > hi {
            x = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                (x + lo) * 0.5
            };
        } else {
            x = x_new;
        }
    }
    x
}

/// Standard Newton refinement for chi-squared quantile computation.
///
/// Efficient iterative refinement method for chi-squared distribution quantiles
/// using safeguarded Newton's method with adaptive step damping. Provides
/// optimal balance between computational efficiency and numerical accuracy
/// for most parameter combinations encountered in statistical applications.
#[inline(always)]
pub fn chi2_newton_refine(mut x: f64, a: f64, p: f64) -> f64 {
    // Safeguarded Newton with damping; works from a decent seed
    let ln_norm = -a * std::f64::consts::LN_2 - ln_gamma(a); // for chi2 pdf log-constant
    let mut lo = 0.0_f64;
    let mut hi = f64::INFINITY;
    for _ in 0..8 {
        if !x.is_finite() {
            break;
        }
        let t = 0.5 * x;
        let fx = reg_lower_gamma(a, t) - p; // target function
        if fx.abs() < 1e-14 {
            break;
        }

        // log pdf: ln f'(x) = ln pdf_{chi2}(x)
        // pdf(x) = exp(ln_norm + (a-1)*ln(x) - x/2)
        let log_pdf = ln_norm + (a - 1.0) * x.ln() - 0.5 * x;
        let pdf = log_pdf.exp();

        if pdf <= 0.0 || !pdf.is_finite() {
            // fall back to bisection-like step if derivative unusable
            if fx > 0.0 {
                hi = hi.min(x);
            } else {
                lo = lo.max(x);
            }
            x = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                (x + lo.max(0.0)) * 0.5
            };
            continue;
        }

        // Newton step with damping
        let mut step = fx / pdf;
        // limit overly large relative steps to keep monotone progress
        let max_step = 0.5 * (x.max(1.0));
        if step.abs() > max_step {
            step = step.signum() * max_step;
        }

        let x_new = (x - step).max(0.0); // enforce domain
        // maintain bracket based on sign of f
        if fx > 0.0 {
            hi = hi.min(x);
        } else {
            lo = lo.max(x);
        }
        x = x_new;
    }
    x
}

/// Evaluates standard normal cumulative distribution function with high accuracy.
///
/// Computes the cumulative distribution function Φ(z) of the standard normal
/// distribution N(0,1) at the specified point z using the complementary error
/// function for optimal numerical precision. This implementation provides
/// superior accuracy compared to direct integration methods.
#[inline(always)]
pub fn normal_cdf_scalar(z: f64) -> f64 {
    // high-accuracy CDF:  0.5·erfc(–z/√2) on the left, 1 – 0.5·erfc(z/√2) on the right
    if z < 0.0 {
        0.5 * erfc(-z / SQRT_2)
    } else {
        1.0 - 0.5 * erfc(z / SQRT_2)
    }
}

/// Evaluates standard normal probability density function at given point.
///
/// Computes the probability density function φ(z) of the standard normal
/// distribution N(0,1) at the specified point z. This function provides
/// numerically stable evaluation across the entire real domain with
/// optimal computational efficiency for statistical applications.
#[inline(always)]
pub fn normal_pdf_scalar(z: f64) -> f64 {
    // Standard normal PDF
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Inverse CDF Φ⁻¹(q) for the normal distribution.
///
/// Accuracy:
/// - Centre and bulk (e.g. 0.025 ≤ q ≤ 0.975): |err| < 1e-14 (equivalent to scipy.stats.norm.ppf, confirmed by unit tests).
/// - Extreme tails (q ≲ 1e-10 or q ≳ 1–1e-10): |err| < 1e-12 compared to SciPy reference values.
/// - **Reciprocal symmetry:** |Φ⁻¹(q) + Φ⁻¹(1–q)| is only guaranteed < 1e-7 in the extreme tails,  
///   due to inherent limitations of the underlying algorithms and double-precision arithmetic.
///   This limitation is observed in SciPy as well as this implementation.
/// ```
pub fn normal_quantile_scalar(q: f64, mean: f64, std: f64) -> f64 {
    // Early exit edge cases
    if !q.is_finite() || !mean.is_finite() || !std.is_finite() || std <= 0.0 {
        return f64::NAN;
    }
    if q < 0.0 || q > 1.0 {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::NEG_INFINITY;
    }
    if q == 1.0 {
        return f64::INFINITY;
    }
    if q == 0.5 {
        return mean;
    }

    // symmetry reduction
    let (p_left, sign) = if q < 0.5 { (q, -1.0) } else { (1.0 - q, 1.0) };

    // extreme-tail shortcut via erfc⁻¹
    const EPS_DBL: f64 = 1.110_223_024_625_156_5e-16;
    if p_left < EPS_DBL {
        // Φ⁻¹(p) = −√2 · erfc⁻¹(2p)   (for p ≤ 0.5)
        let z_tail = -SQRT_2 * erfc_inv(2.0 * p_left);
        return mean + std * sign * -z_tail; // mirror if q > 0.5
    }

    // Acklam initial approximation
    let mut z = inv_std_normal_core(p_left); // negative

    // one Halley refinement step
    // Halley:  z_{n+1} = z_n − f/f' · (1 + ½ f · f'' / f'^2)
    // Here f = Φ(z) − p,  f' = φ(z),  f'' = −z φ(z)
    let pdf = normal_pdf_scalar(z);
    let cdf = normal_cdf_scalar(z);
    let f = cdf - p_left;
    let u = f / pdf;
    z -= u * (1.0 + 0.5 * z * u); // ≤ 1 ulp after this step

    // reflect to right tail if necessary
    let z_final = sign * -z;

    mean + std * z_final
}

/// Student's t-distribution scalar CDF.
///
/// Computes P(T ≤ x) for a Student's t-distribution with `df` degrees of freedom.
/// Uses the regularised incomplete beta function directly, matching the vectorised
/// `student_t_cdf` kernel.
///
/// Returns NaN for non-finite inputs or df < 1.
#[inline(always)]
pub fn student_t_cdf_scalar(x: f64, df: f64) -> f64 {
    if !x.is_finite() || !df.is_finite() || df < 1.0 {
        return f64::NAN;
    }
    let xb = df / (df + x * x);
    let ib = incomplete_beta(df / 2.0, 0.5, xb);
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    0.5 + sign * 0.5 * (1.0 - ib)
}

/// Chi-squared distribution scalar CDF.
///
/// Computes P(X ≤ x) for a chi-squared distribution with `df` degrees of freedom.
/// Delegates to `reg_lower_gamma(df/2, x/2)`.
///
/// Returns NaN for non-finite inputs or df < 1. Returns 0 for x ≤ 0.
#[inline(always)]
pub fn chi_square_cdf_scalar(x: f64, df: f64) -> f64 {
    if !x.is_finite() || !df.is_finite() || df < 1.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    reg_lower_gamma(df / 2.0, x / 2.0)
}

/// F-distribution scalar CDF.
///
/// Computes P(F ≤ x) for an F-distribution with `d1` and `d2` degrees of freedom.
/// Uses the regularised incomplete beta function.
///
/// Returns NaN for non-finite inputs or d1/d2 < 1. Returns 0 for x ≤ 0.
#[inline(always)]
pub fn f_cdf_scalar(x: f64, d1: f64, d2: f64) -> f64 {
    if !x.is_finite() || !d1.is_finite() || !d2.is_finite() || d1 < 1.0 || d2 < 1.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let xt = d1 * x / (d1 * x + d2);
    incomplete_beta(d1 / 2.0, d2 / 2.0, xt)
}

/// Student's t-distribution scalar quantile (inverse CDF).
///
/// Computes the value t such that P(T ≤ t) = q for a t-distribution with `df`
/// degrees of freedom. Uses the inverse regularised incomplete beta function,
/// transforming via t = sqrt(df * (1 - x) / x) where x = I^{-1}(df/2, 1/2, p).
///
/// Returns NaN for non-finite inputs, q outside [0, 1], or df < 1.
#[inline(always)]
pub fn student_t_quantile_scalar(q: f64, df: f64) -> f64 {
    if !q.is_finite() || !df.is_finite() || df < 1.0 {
        return f64::NAN;
    }
    if q < 0.0 || q > 1.0 {
        return f64::NAN;
    }
    if q == 0.0 {
        return f64::NEG_INFINITY;
    }
    if q == 1.0 {
        return f64::INFINITY;
    }
    if df > 1e7 {
        return normal_quantile_scalar(q, 0.0, 1.0);
    }
    // Use symmetry: work with the tail probability so incomplete_beta_inv
    // always receives a value in (0, 1). For q < 0.5 the result is negative.
    let x = if q < 0.5 {
        incomplete_beta_inv(0.5 * df, 0.5, 2.0 * q)
    } else {
        incomplete_beta_inv(0.5 * df, 0.5, 2.0 * (1.0 - q))
    };
    let t = ((df * (1.0 - x)) / x).sqrt();
    if q < 0.5 { -t } else { t }
}

/// Standard normal quantile (inverse CDF) for N(0, 1).
///
/// Shorthand for `normal_quantile_scalar(q, 0.0, 1.0)`.
#[inline(always)]
pub fn normal_quantile_scalar_std(q: f64) -> f64 {
    normal_quantile_scalar(q, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // All tests below were ran with Scipy v1.16.0, with the code responsible for producing
    // them saved under /python/tests/univariate.py.
    // The tests annotate the expected result we have matched with the functions.

    #[test]
    fn test_ln_gamma() {
        // scipy.special.gammaln(1.0) == 0.0
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-14);
        // scipy.special.gammaln(5.0) == 3.1780538303479458
        assert!((ln_gamma(5.0) - 3.1780538303479458).abs() < 1e-14);
        // scipy.special.gammaln(0.5) == 0.5723649429247
        assert!((ln_gamma(0.5) - 0.5723649429247).abs() < 1e-14);
        // scipy.special.gammaln(10.1) == 13.027526738633238
        assert!((ln_gamma(10.1) - 13.027526738633238).abs() < 1e-10);
        // scipy.special.gammaln(0.0) == inf
        assert!(ln_gamma(0.0).is_infinite() && ln_gamma(0.0).is_sign_positive());
        // scipy.special.gammaln(-1.0) == inf
        assert!(ln_gamma(-1.0).is_infinite() && ln_gamma(-1.0).is_sign_positive());
        // scipy.special.gammaln(-0.5) == 1.2655121234846454
        assert!((ln_gamma(-0.5) - 1.2655121234846454).abs() < 1e-14);
        // scipy.special.gammaln(-10.1) == -13.020973271011497
        assert!((ln_gamma(-10.1) - -13.020973271011497).abs() < 1e-12);
        // scipy.special.gammaln(np.nan) == nan
        assert!(ln_gamma(f64::NAN).is_nan());
        // scipy.special.gammaln(1e-10) == 23.025850929882733
        assert!((ln_gamma(1e-10) - 23.025850929882733).abs() < 1e-12);
        // scipy.special.gammaln(171.624) == 709.7807744366991
        assert!((ln_gamma(171.624) - 709.7807744366991).abs() < 1e-9);
    }

    #[test]
    fn test_ln_gamma_plus1() {
        // scipy.special.gammaln(6.0) == 4.787491742782046
        assert!((ln_gamma_plus1(5.0) - 4.787491742782046).abs() < 1e-14);
    }

    #[test]
    fn test_gamma_func() {
        // scipy.special.gamma(1.0) == 1.0
        assert!((gamma_func(1.0) - 1.0).abs() < 1e-14);
        // scipy.special.gamma(5.0) == 24.0
        assert!((gamma_func(5.0) - 24.0).abs() < 1e-14);
        // scipy.special.gamma(0.5) == 1.7724538509055159
        assert!((gamma_func(0.5) - 1.7724538509055159).abs() < 1e-14);
        // scipy.special.gamma(10.1) == 454760.7514415855
        assert!((gamma_func(10.1) - 454760.7514415855).abs() < 1e-7);
        // scipy.special.gamma(0.0) == inf
        assert!(gamma_func(0.0).is_infinite() && gamma_func(0.0).is_sign_positive());
        // scipy.special.gamma(-1.0) == nan
        assert!(gamma_func(-1.0).is_nan());
        // scipy.special.gamma(-0.5) == -3.5449077018110318
        assert!((gamma_func(-0.5) + 3.5449077018110318).abs() < 1e-14);
        // scipy.special.gamma(-10.1) == -2.213416583085619e-06
        assert!((gamma_func(-10.1) + 2.213416583085619e-6).abs() < 1e-14);
        // scipy.special.gamma(171.0) == 7.257415615308e+306
        assert!((gamma_func(171.0) - 7.257415615308e+306).abs() < 1e292); // tolerance due to magnitude
        // scipy.special.gamma(np.nan) == nan
        assert!(gamma_func(f64::NAN).is_nan());
        // Large argument, should not panic and should return inf.
        assert!(ln_gamma(1e308).is_infinite());
        // Negative infinity should return NaN by convention.
        assert!(ln_gamma(f64::NEG_INFINITY).is_nan());
        // Positive infinity should return inf
        assert!(ln_gamma(f64::INFINITY).is_infinite());
    }

    #[test]
    fn test_ln_choose() {
        // np.log(scipy.special.comb(5, 2, exact=False)) == 2.302585092994046
        assert!((ln_choose(5, 2) - 2.302585092994046).abs() < 1e-14);
        // k > n, should be -inf
        assert!(ln_choose(2, 3).is_infinite() && ln_choose(2, 3).is_sign_negative());
        // symmetry: C(n,k) == C(n, n-k)
        assert!((ln_choose(100, 3) - ln_choose(100, 97)).abs() < 1e-14);
        // np.log(scipy.special.comb(1000, 10, exact=False)) == 53.927997037888275
        assert!((ln_choose(1000, 10) - 53.927997037888275).abs() < 1e-10);
        // Edge combinatorics
        assert!((ln_choose(10, 0) - 0.0).abs() < 1e-14);
        assert!((ln_choose(10, 10) - 0.0).abs() < 1e-14);
        assert!((ln_choose(0, 0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_incomplete_beta() {
        // scipy.special.betainc(2.0, 2.0, 0.5) == 0.5
        assert!((incomplete_beta(2.0, 2.0, 0.5) - 0.5).abs() < 1e-14);
        // scipy.special.betainc(2.5, 1.5, 0.7) == 0.5843121477019746
        assert!((incomplete_beta(2.5, 1.5, 0.7) - 0.5843121477019746).abs() < 1e-12);
        // scipy.special.betainc(2.0, 2.0, 0.0) == 0.0
        assert!((incomplete_beta(2.0, 2.0, 0.0) - 0.0).abs() < 1e-14);
        // scipy.special.betainc(2.0, 2.0, 1.0) == 1.0
        assert!((incomplete_beta(2.0, 2.0, 1.0) - 1.0).abs() < 1e-14);
        // scipy.special.betainc(0.0, 2.0, 0.5) == 1.0
        assert!((incomplete_beta(0.0, 2.0, 0.5) - 1.0).abs() < 1e-14);
        // scipy.special.betainc(2.0, 0.0, 0.5) == 0.0
        assert!((incomplete_beta(2.0, 0.0, 0.5) - 0.0).abs() < 1e-14);
        // scipy.special.betainc(2.0, 2.0, np.nan) == nan
        assert!(incomplete_beta(2.0, 2.0, f64::NAN).is_nan());

        // Symmetry property Iₓ(a,b)+I_{1-x}(b,a)=1:
        let a = 2.7;
        let b = 5.3;
        let x = 0.4;
        let ix = incomplete_beta(a, b, x);
        let ix2 = incomplete_beta(b, a, 1.0 - x);
        assert!((ix + ix2 - 1.0).abs() < 1e-16);
        // scipy.special.betainc(50.0, 0.5, 0.01) == 7.998227417904836e-102
        assert!((incomplete_beta(50.0, 0.5, 0.01) - 7.998227417904836e-102).abs() < 1e-90);

        // Large a, b
        assert!(incomplete_beta(1e8, 2.0, 0.5).is_finite());
        // x extremely close to 0
        assert!((incomplete_beta(2.0, 3.0, 1e-20) - 0.0).abs() < 1e-20);
        // x extremely close to 1
        assert!((incomplete_beta(2.0, 3.0, 1.0 - 1e-20) - 1.0).abs() < 1e-14);
        // Smallest positive subnormal x
        assert!(incomplete_beta(2.0, 3.0, f64::MIN_POSITIVE).is_finite());

        // Incomplete beta round-trip
        for &a in &[2.0, 5.0, 10.0] {
            for &b in &[2.0, 5.0, 10.0] {
                for &p in &[1e-15, 1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-8, 1.0 - 1e-15] {
                    let x = incomplete_beta_inv(a, b, p);
                    let p2 = incomplete_beta(a, b, x);
                    assert!((p - p2).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_incomplete_beta_inv() {
        // scipy.special.betaincinv(2.0, 2.0, 0.5) == 0.5
        assert!((incomplete_beta_inv(2.0, 2.0, 0.5) - 0.5).abs() < 1e-14);
        // scipy.special.betaincinv(2.5, 1.5, 0.8433681004114891) == 0.8595961302031141
        assert!(
            (incomplete_beta_inv(2.5, 1.5, 0.8433681004114891) - 0.8595961302031141).abs() < 1e-12
        );
        // scipy.special.betaincinv(2.0, 2.0, 0.0) == 0.0
        assert!((incomplete_beta_inv(2.0, 2.0, 0.0) - 0.0).abs() < 1e-14);
        // scipy.special.betaincinv(2.0, 2.0, 1.0) == 1.0
        assert!((incomplete_beta_inv(2.0, 2.0, 1.0) - 1.0).abs() < 1e-14);

        // incomplete_beta_inv edge cases
        // scipy.special.betaincinv(2.0, 2.0, np.nan) == nan
        assert!(incomplete_beta_inv(2.0, 2.0, f64::NAN).is_nan());
        // scipy.special.betaincinv(np.nan, 2.0, 0.5) == nan
        assert!(incomplete_beta_inv(f64::NAN, 2.0, 0.5).is_nan());

        // incomplete_beta_inv
        let a = 7.0;
        let b = 3.0;
        let p = 1e-6;
        let x = incomplete_beta_inv(a, b, p);
        let p2 = incomplete_beta(a, b, x);

        println!("{}", (p - p2).abs());
        assert!((p - p2).abs() < 1e-16);
    }

    #[test]
    fn test_reg_lower_gamma() {
        // scipy.special.gammainc(2.0, 2.0) == 0.5939941502901616
        assert!((reg_lower_gamma(2.0, 2.0) - 0.5939941502901616).abs() < 1e-12);
        // scipy.special.gammainc(5.0, 1.0) == 0.003659846827343713
        assert!((reg_lower_gamma(5.0, 1.0) - 0.003659846827343713).abs() < 1e-14);
        // scipy.special.gammainc(2.0, 0.0) == 0.0
        assert!((reg_lower_gamma(2.0, 0.0) - 0.0).abs() < 1e-14);
        // scipy.special.gammainc(0.0, 2.0) == 1.0
        assert!((reg_lower_gamma(0.0, 2.0) - 1.0).abs() < 1e-14);
        // scipy.special.gammainc(2.0, -1.0) == nan
        assert!(reg_lower_gamma(2.0, -1.0).is_nan());
        // scipy.special.gammainc(-1.0, 2.0) == nan
        assert!(reg_lower_gamma(-1.0, 2.0).is_nan());
        // scipy.special.gammainc(np.nan, 2.0) == nan
        assert!(reg_lower_gamma(f64::NAN, 2.0).is_nan());

        // Roundtrip
        let a = 3.5;
        let p = 0.123456;
        let x = inv_reg_lower_gamma(a, p);
        let p2 = reg_lower_gamma(a, x);
        assert!((p - p2).abs() < 1e-10);

        assert!(reg_lower_gamma(1e8, 1e8).is_finite());
        assert!(reg_lower_gamma(1e-20, 1e20).is_finite());
    }

    #[test]
    fn test_inv_reg_lower_gamma() {
        // scipy.special.gammaincinv(2.0, 0.5939941502901616) == 2.0
        assert!((inv_reg_lower_gamma(2.0, 0.5939941502901616) - 2.0).abs() < 1e-10);
        // scipy.special.gammaincinv(5.0, 0.003659846827343713) == 1.0000000000000002
        assert!(
            (inv_reg_lower_gamma(5.0, 0.003659846827343713) - 1.0000000000000002).abs() < 1e-10
        );
        // scipy.special.gammaincinv(2.0, np.nan) == nan
        assert!(inv_reg_lower_gamma(2.0, f64::NAN).is_nan());
        // scipy.special.gammaincinv(np.nan, 2.0) == nan
        assert!(inv_reg_lower_gamma(f64::NAN, 2.0).is_nan());
    }

    #[test]
    fn test_binomial_cdf_scalar() {
        // scipy.stats.binom.cdf(2, 4, 0.5) == 0.6875
        assert!((binomial_cdf_scalar(2, 4, 0.5) - 0.6875).abs() < 1e-14);
        // scipy.stats.binom.cdf(0, 4, 0.5) == 0.0625
        assert!((binomial_cdf_scalar(0, 4, 0.5) - 0.0625).abs() < 1e-14);
        // scipy.stats.binom.cdf(0, 4, 0.0) == 1.0
        assert!((binomial_cdf_scalar(0, 4, 0.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(2, 4, 0.0) == 1.0
        assert!((binomial_cdf_scalar(2, 4, 0.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(4, 4, 1.0) == 1.0
        assert!((binomial_cdf_scalar(4, 4, 1.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(2, 4, 1.0) == 0.0
        assert!((binomial_cdf_scalar(2, 4, 1.0) - 0.0).abs() < 1e-14);

        // For any k >= 0 and p == 0.0, result is 1.0
        // scipy.stats.binom.cdf(10, 4, 0.0) == 1.0
        assert!((binomial_cdf_scalar(10, 4, 0.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(0, 0, 0.0) == 1.0
        assert!((binomial_cdf_scalar(0, 0, 0.0) - 1.0).abs() < 1e-14);

        // For any k < n and p == 1.0, result is 0.0
        // scipy.stats.binom.cdf(0, 4, 1.0) == 0.0
        assert!((binomial_cdf_scalar(0, 4, 1.0) - 0.0).abs() < 1e-14);

        // For any k >= n and p == 1.0, result is 1.0
        // scipy.stats.binom.cdf(5, 4, 1.0) == 1.0
        assert!((binomial_cdf_scalar(5, 4, 1.0) - 1.0).abs() < 1e-14);

        // Degenerate n=0, all probability at 0
        // scipy.stats.binom.cdf(0, 0, 0.5) == 1.0
        assert!((binomial_cdf_scalar(0, 0, 0.5) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(1, 0, 0.5) == 1.0
        assert!((binomial_cdf_scalar(1, 0, 0.5) - 1.0).abs() < 1e-14);

        // scipy.stats.binom.cdf(0, 0, 0.5) == 1.0
        assert!((binomial_cdf_scalar(0, 0, 0.5) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(-1, 0, 0.5) == 0.0
        assert!((binomial_cdf_scalar(-1i64, 0, 0.5) - 0.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(0, 0, 0.0) == 1.0
        assert!((binomial_cdf_scalar(0, 0, 0.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(0, 0, 1.0) == 1.0
        assert!((binomial_cdf_scalar(0, 0, 1.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(10, 0, 0.0) == 1.0
        assert!((binomial_cdf_scalar(10, 0, 0.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(10, 0, 1.0) == 1.0
        assert!((binomial_cdf_scalar(10, 0, 1.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(10, 10, 0.0) == 1.0
        assert!((binomial_cdf_scalar(10, 10, 0.0) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(10, 10, 1.0) == 1.0
        assert!((binomial_cdf_scalar(10, 10, 1.0) - 1.0).abs() < 1e-14);

        // scipy.stats.binom.cdf(5, 10000, 1e-4) == 0.999406428148761
        assert!((binomial_cdf_scalar(5, 10000, 1e-4) - 0.999406428148761).abs() < 1e-12);

        // Monotonicity & limits:
        for n in [0_i64, 1, 10, 100] {
            for k in 0..=n {
                assert!(
                    binomial_cdf_scalar(k, n.try_into().unwrap(), 0.3)
                        <= binomial_cdf_scalar(k + 1, n.try_into().unwrap(), 0.3)
                );
            }
        }
        assert!((binomial_cdf_scalar(0, 1000000, 0.5) - (0.5f64).powi(1000000)).abs() < 1e-14);
        assert!((binomial_cdf_scalar(1000000, 1000000, 0.5) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(10, 100, sys.float_info.min() == 1.0
        assert!((binomial_cdf_scalar(10, 100, f64::MIN_POSITIVE) - 1.0).abs() < 1e-14);
        // scipy.stats.binom.cdf(10, 100, 1.0 - np.finfo(float).eps) == 0.0
        assert!((binomial_cdf_scalar(10, 100, 1.0 - f64::EPSILON) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_binomial_quantile_cornish_fisher() {
        // scipy.stats.binom.ppf(0.5, 10, 0.5) == 5.0
        assert!((binomial_quantile_cornish_fisher(0.5, 10, 0.5) - 5.0).abs() < 1e-14);
        // scipy.stats.binom.ppf(1e-10, 10, 0.5) == 0.0
        assert!((binomial_quantile_cornish_fisher(1e-10, 10, 0.5) - 0.0).abs() < 1e-8);
        // scipy.stats.binom.ppf(1.0 - 1e-10, 10, 0.5) == 10.0
        assert!((binomial_quantile_cornish_fisher(1.0 - 1e-10, 10, 0.5) - 10.0).abs() < 1e-8);

        // scipy.stats.binom.ppf(0.0, 20, 0.3) == -1.0
        assert_eq!(binomial_quantile_cornish_fisher(0.0, 20, 0.3), -1.0);
        // scipy.stats.binom.ppf(1.0, 20, 0.3) == 20
        assert_eq!(binomial_quantile_cornish_fisher(1.0, 20, 0.3), 20.0);
    }

    #[test]
    fn test_ln_gamma_additional_edges() {
        // Poles at all non-positive integers
        for i in 0..100 {
            let x = -(i as f64);
            assert!(ln_gamma(x).is_infinite());
        }

        // Just above/below a pole (should not be infinite or NaN)
        for i in 0..10 {
            let x = -(i as f64);
            let just_above = x + 1e-12;
            let just_below = x - 1e-12;
            assert!(
                ln_gamma(just_above).is_finite(),
                "ln_gamma({}) not finite",
                just_above
            );
            assert!(
                ln_gamma(just_below).is_finite(),
                "ln_gamma({}) not finite",
                just_below
            );
        }
    }

    #[test]
    fn test_student_t_cdf_scalar() {
        // scipy.stats.t.cdf(0.0, 10) == 0.5
        assert!((student_t_cdf_scalar(0.0, 10.0) - 0.5).abs() < 1e-15);
        // scipy.stats.t.cdf(1.0, 10) == 0.82955343384897
        assert!((student_t_cdf_scalar(1.0, 10.0) - 0.82955343384897).abs() < 1e-14);
        // scipy.stats.t.cdf(-2.0, 5) == 0.05096973941492914
        assert!((student_t_cdf_scalar(-2.0, 5.0) - 0.05096973941492914).abs() < 1e-14);
        // scipy.stats.t.cdf(3.0, 30) == 0.997305017967174
        assert!((student_t_cdf_scalar(3.0, 30.0) - 0.997305017967174).abs() < 1e-14);
        // scipy.stats.t.cdf(0.5, 1) == 0.6475836176504333 (Cauchy)
        assert!((student_t_cdf_scalar(0.5, 1.0) - 0.6475836176504333).abs() < 1e-14);
        // Symmetry: CDF(x, df) + CDF(-x, df) == 1
        let cdf_pos = student_t_cdf_scalar(1.0, 10.0);
        let cdf_neg = student_t_cdf_scalar(-1.0, 10.0);
        assert!((cdf_pos + cdf_neg - 1.0).abs() < 1e-15);
        // Edge: NaN propagation
        assert!(student_t_cdf_scalar(f64::NAN, 10.0).is_nan());
        assert!(student_t_cdf_scalar(1.0, f64::NAN).is_nan());
        assert!(student_t_cdf_scalar(1.0, 0.5).is_nan()); // df < 1
    }

    #[test]
    fn test_chi_square_cdf_scalar() {
        // scipy.stats.chi2.cdf(3.841459148, 1) == 0.9500000097600837
        assert!((chi_square_cdf_scalar(3.841459148, 1.0) - 0.9500000097600837).abs() < 1e-14);
        // scipy.stats.chi2.cdf(5.991464547, 2) == 0.9499999999973004
        assert!((chi_square_cdf_scalar(5.991464547, 2.0) - 0.9499999999973004).abs() < 1e-14);
        // scipy.stats.chi2.cdf(0.0, 5) == 0.0
        assert!((chi_square_cdf_scalar(0.0, 5.0) - 0.0).abs() < 1e-15);
        // scipy.stats.chi2.cdf(10.0, 5) == 0.9247647538534878
        assert!((chi_square_cdf_scalar(10.0, 5.0) - 0.9247647538534878).abs() < 1e-14);
        // scipy.stats.chi2.cdf(1.0, 1) == 0.6826894921370859
        assert!((chi_square_cdf_scalar(1.0, 1.0) - 0.6826894921370859).abs() < 1e-14);
        // scipy.stats.chi2.cdf(20.0, 10) == 0.9707473119230389
        assert!((chi_square_cdf_scalar(20.0, 10.0) - 0.9707473119230389).abs() < 1e-14);
        // Edge: negative x
        assert!((chi_square_cdf_scalar(-1.0, 5.0) - 0.0).abs() < 1e-15);
        // Edge: NaN
        assert!(chi_square_cdf_scalar(f64::NAN, 5.0).is_nan());
        assert!(chi_square_cdf_scalar(5.0, 0.5).is_nan()); // df < 1
    }

    #[test]
    fn test_f_cdf_scalar() {
        // scipy.stats.f.cdf(1.0, 5, 10) == 0.5348805734622
        assert!((f_cdf_scalar(1.0, 5.0, 10.0) - 0.5348805734622).abs() < 1e-13);
        // scipy.stats.f.cdf(3.0, 2, 20) == 0.9274618497135944
        assert!((f_cdf_scalar(3.0, 2.0, 20.0) - 0.9274618497135944).abs() < 1e-14);
        // scipy.stats.f.cdf(0.0, 5, 10) == 0.0
        assert!((f_cdf_scalar(0.0, 5.0, 10.0) - 0.0).abs() < 1e-15);
        // scipy.stats.f.cdf(4.0, 3, 15) == 0.9718629601291221
        assert!((f_cdf_scalar(4.0, 3.0, 15.0) - 0.9718629601291221).abs() < 1e-14);
        // scipy.stats.f.cdf(10.0, 1, 1) == 0.805017770957863
        assert!((f_cdf_scalar(10.0, 1.0, 1.0) - 0.805017770957863).abs() < 1e-14);
        // Edge: NaN
        assert!(f_cdf_scalar(f64::NAN, 5.0, 10.0).is_nan());
        assert!(f_cdf_scalar(1.0, 0.5, 10.0).is_nan()); // d1 < 1
        assert!(f_cdf_scalar(1.0, 5.0, 0.5).is_nan()); // d2 < 1
    }

    #[test]
    fn test_student_t_quantile_scalar() {
        // Reference values from mpmath (v1.3, 50-digit precision) via Newton iteration on
        // mpmath.betainc. SciPy's t.ppf has ~1e-11 error vs the true quantile, so mpmath
        // was used instead. Each value was independently verified by checking that
        // mpmath's CDF at the nearest f64 gives residual < 1e-17.
        // mpmath t.ppf(0.975, 10) = 2.22813885198627481543
        assert!((student_t_quantile_scalar(0.975, 10.0) - 2.22813885198627481543e+00).abs() < 1e-14);
        // mpmath t.ppf(0.025, 10) = -2.22813885198627481543
        assert!((student_t_quantile_scalar(0.025, 10.0) - -2.22813885198627481543e+00).abs() < 1e-14);
        // mpmath t.ppf(0.99, 5) = 3.36492999890721877776
        assert!((student_t_quantile_scalar(0.99, 5.0) - 3.36492999890721877776e+00).abs() < 1e-14);
        // mpmath t.ppf(0.01, 5) = -3.36492999890721877776
        assert!((student_t_quantile_scalar(0.01, 5.0) - -3.36492999890721877776e+00).abs() < 1e-14);
        // mpmath t.ppf(0.5, 10) = 0.0
        assert!(student_t_quantile_scalar(0.5, 10.0).abs() < 1e-14);
        // mpmath t.ppf(0.9, 15) = 1.34060560785045557175
        assert!((student_t_quantile_scalar(0.9, 15.0) - 1.34060560785045557175e+00).abs() < 1e-14);
        // mpmath t.ppf(0.1, 15) = -1.34060560785045557175
        assert!((student_t_quantile_scalar(0.1, 15.0) - -1.34060560785045557175e+00).abs() < 1e-14);
        // mpmath t.ppf(0.75, 20) = 0.686954496448803464403
        assert!((student_t_quantile_scalar(0.75, 20.0) - 6.86954496448803464403e-01).abs() < 1e-14);
        // mpmath t.ppf(0.999, 3) = 10.2145318524073864808
        assert!((student_t_quantile_scalar(0.999, 3.0) - 1.02145318524073864808e+01).abs() < 1e-14);
        // mpmath t.ppf(0.001, 3) = -10.2145318524073864808
        assert!((student_t_quantile_scalar(0.001, 3.0) - -1.02145318524073864808e+01).abs() < 1e-14);
        // mpmath t.ppf(0.95, 1) = 6.31375151467504291958
        assert!((student_t_quantile_scalar(0.95, 1.0) - 6.31375151467504291958e+00).abs() < 1e-14);
        // mpmath t.ppf(0.99, 100) = 2.36421736623848222081
        assert!((student_t_quantile_scalar(0.99, 100.0) - 2.36421736623848222081e+00).abs() < 1e-14);
        // mpmath t.ppf(0.9, 2) = 1.88561808316412671260
        assert!((student_t_quantile_scalar(0.9, 2.0) - 1.88561808316412671260e+00).abs() < 1e-14);
        // mpmath t.ppf(0.999, 30) = 3.38518486682930497267
        assert!((student_t_quantile_scalar(0.999, 30.0) - 3.38518486682930497267e+00).abs() < 1e-14);

        // Symmetry: ppf(q, df) == -ppf(1-q, df)
        for &df in &[1.0, 3.0, 10.0, 30.0, 100.0] {
            for &q in &[0.01, 0.1, 0.25] {
                let lo = student_t_quantile_scalar(q, df);
                let hi = student_t_quantile_scalar(1.0 - q, df);
                assert!((lo + hi).abs() < 1e-12,
                    "symmetry failed for q={}, df={}: lo={}, hi={}", q, df, lo, hi);
            }
        }

        // Edge cases
        assert!(student_t_quantile_scalar(0.0, 10.0).is_infinite()
            && student_t_quantile_scalar(0.0, 10.0).is_sign_negative());
        assert!(student_t_quantile_scalar(1.0, 10.0).is_infinite()
            && student_t_quantile_scalar(1.0, 10.0).is_sign_positive());
        assert!(student_t_quantile_scalar(f64::NAN, 10.0).is_nan());
        assert!(student_t_quantile_scalar(0.5, 0.5).is_nan()); // df < 1

        // Roundtrip: CDF(quantile(q, df), df) ≈ q across varied df and q
        for &df in &[1.0, 2.0, 5.0, 10.0, 30.0, 50.0, 100.0, 500.0] {
            for &q in &[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999] {
                let x = student_t_quantile_scalar(q, df);
                let p = student_t_cdf_scalar(x, df);
                assert!((p - q).abs() < 1e-14,
                    "roundtrip failed for q={}, df={}: got p={}, err={:.2e}",
                    q, df, p, (p - q).abs());
            }
        }
    }

    #[test]
    fn test_normal_quantile_scalar_std() {
        // scipy.stats.norm.ppf(0.5) == 0.0
        assert!((normal_quantile_scalar_std(0.5) - 0.0).abs() < 1e-15);
        // scipy.stats.norm.ppf(0.975) == 1.959963984540054
        assert!((normal_quantile_scalar_std(0.975) - 1.959963984540054).abs() < 1e-14);
        // scipy.stats.norm.ppf(0.025) == -1.9599639845400545
        assert!((normal_quantile_scalar_std(0.025) - -1.9599639845400545).abs() < 1e-14);
        // scipy.stats.norm.ppf(0.01) == -2.3263478740408408
        assert!((normal_quantile_scalar_std(0.01) - -2.3263478740408408).abs() < 1e-14);
        // Edge: NaN
        assert!(normal_quantile_scalar_std(f64::NAN).is_nan());
    }
}
