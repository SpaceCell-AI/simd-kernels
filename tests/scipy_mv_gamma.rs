// SciPy reference tests for Multivariate Gamma PDF (independent marginals definition).
//
// Reference values generated from SciPy 1.16.1:
//   product of scipy.stats.gamma.pdf(x_i, a=shape_i, scale=scale_i)
//
// Uses the independent marginals definition:
//   f(x) = prod_i Gamma(x_i; shape_i, scale_i)
// with no cross-dimensional correlation.

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_mv_gamma_tests {
    use super::util::assert_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::multivariate_gamma_pdf;

    fn pdf_one(x: &[f64], shape: &[f64], scale: &[f64]) -> f64 {
        let result = multivariate_gamma_pdf(x, shape, scale, None, None).unwrap();
        result.data[0]
    }

    // shapes=[2.0, 3.0], scales=[1.0, 0.5]

    #[test]
    fn mv_gamma_pdf_moderate() {
        // scipy: prod(gamma.pdf(1.0, a=2, scale=1), gamma.pdf(0.5, a=3, scale=0.5))
        //      = 1.353352832366126746e-01
        let got = pdf_one(&[1.0, 0.5], &[2.0, 3.0], &[1.0, 0.5]);
        assert_close(got, 1.353352832366126746e-01, 1e-15);
    }

    #[test]
    fn mv_gamma_pdf_near_mode() {
        // scipy: 1.465251111098734571e-01
        let got = pdf_one(&[2.0, 1.0], &[2.0, 3.0], &[1.0, 0.5]);
        assert_close(got, 1.465251111098734571e-01, 1e-15);
    }

    #[test]
    fn mv_gamma_pdf_small_x() {
        // scipy: 2.963272882726872802e-03
        let got = pdf_one(&[0.1, 0.1], &[2.0, 3.0], &[1.0, 0.5]);
        assert_close(got, 2.963272882726872802e-03, 1e-15);
    }

    #[test]
    fn mv_gamma_pdf_tail() {
        // scipy: 3.006306142244217963e-03
        let got = pdf_one(&[5.0, 3.0], &[2.0, 3.0], &[1.0, 0.5]);
        assert_close(got, 3.006306142244217963e-03, 1e-14);
    }
}
