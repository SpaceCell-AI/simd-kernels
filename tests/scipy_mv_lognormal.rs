// SciPy reference tests for Multivariate Lognormal PDF.
//
// Reference values generated from SciPy 1.16.1.
// No direct scipy.stats.multivariate_lognormal exists; computed as:
//   f(x) = multivariate_normal.pdf(ln(x); mu, Sigma) * prod(1/x_i)
//
// Parameters used: mu=[0,0], Sigma=[[1,0],[0,1]] (independent standard lognormal)

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_mv_lognormal_tests {
    use super::util::assert_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::mv_lognormal_pdf;

    fn pdf_one(x: &[f64], mean: &[f64], cov: &[&[f64]]) -> f64 {
        let cov_v: Vec<&[f64]> = cov.to_vec();
        let result = mv_lognormal_pdf(x, mean, cov_v, None, None).unwrap();
        result.data[0]
    }

    const MEAN: &[f64] = &[0.0, 0.0];
    const COV: &[&[f64]] = &[&[1.0, 0.0], &[0.0, 1.0]];

    #[test]
    fn mv_lognormal_pdf_at_one_one() {
        // scipy: mvn.pdf([0,0]; [0,0], I) * 1/(1*1) = (2pi)^{-1} = 1.591549430918953456e-01
        let got = pdf_one(&[1.0, 1.0], MEAN, COV);
        assert_close(got, 1.591549430918953456e-01, 1e-15);
    }

    #[test]
    fn mv_lognormal_pdf_at_two_three() {
        // scipy: 1.140917385583220534e-02
        let got = pdf_one(&[2.0, 3.0], MEAN, COV);
        assert_close(got, 1.140917385583220534e-02, 1e-15);
    }

    #[test]
    fn mv_lognormal_pdf_at_half_half() {
        // scipy: 3.937513267958741014e-01
        let got = pdf_one(&[0.5, 0.5], MEAN, COV);
        assert_close(got, 3.937513267958741014e-01, 1e-15);
    }

    #[test]
    fn mv_lognormal_pdf_extreme_spread() {
        // x=[0.1, 10.0] - one small, one large
        // scipy: 7.929303454965448521e-04
        let got = pdf_one(&[0.1, 10.0], MEAN, COV);
        assert_close(got, 7.929303454965448521e-04, 1e-14);
    }
}
