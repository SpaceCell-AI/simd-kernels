// SciPy + mpmath reference tests for Dirichlet PDF.
//
// Reference values generated from SciPy 1.16.1 and cross-checked with
// mpmath (50-digit precision).

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_dirichlet_tests {
    use super::util::assert_slice_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::dirichlet_pdf;

    fn pdf_one(x: &[f64], alpha: &[f64]) -> f64 {
        let result = dirichlet_pdf(x, alpha, None, None).unwrap();
        result[0]
    }

    // ---- 3-component ----

    #[test]
    fn dirichlet_pdf_3d_uniform_alpha() {
        // alpha = [1,1,1] -> Dirichlet is uniform on simplex, PDF = (3-1)! = 2.0
        // mpmath: 2.0
        let got = pdf_one(&[0.2, 0.3, 0.5], &[1.0, 1.0, 1.0]);
        assert_slice_close(&[got], &[2.0], 1e-15);
    }

    #[test]
    fn dirichlet_pdf_3d_varied_alpha() {
        // scipy: 8.505000000000003
        // mpmath: 8.5049999999999998426
        let got = pdf_one(&[0.2, 0.3, 0.5], &[2.0, 3.0, 5.0]);
        assert_slice_close(&[got], &[8.505000000000003], 1e-14);
    }

    #[test]
    fn dirichlet_pdf_3d_skewed_x() {
        // scipy: 7.260623999999999
        // mpmath: 7.2606239999999993666
        let got = pdf_one(&[0.1, 0.2, 0.7], &[2.0, 3.0, 5.0]);
        assert_slice_close(&[got], &[7.260623999999999], 1e-14);
    }

    #[test]
    fn dirichlet_pdf_3d_alpha_lt_1() {
        // alpha < 1 gives U-shaped marginals
        // scipy: 0.9188814923696538
        // mpmath: 0.91888149236965340735
        let got = pdf_one(&[0.5, 0.3, 0.2], &[0.5, 0.5, 0.5]);
        assert_slice_close(&[got], &[0.9188814923696538], 1e-14);
    }

    #[test]
    fn dirichlet_pdf_3d_near_uniform_x() {
        // x near centre of simplex, uniform alpha
        // scipy: 2.0
        let got = pdf_one(&[0.333333333, 0.333333333, 0.333333334], &[1.0, 1.0, 1.0]);
        assert_slice_close(&[got], &[2.0], 1e-15);
    }

    #[test]
    fn dirichlet_pdf_3d_extreme_x() {
        // x near a vertex
        // scipy: 0.011760000000000008
        let got = pdf_one(&[0.01, 0.01, 0.98], &[2.0, 2.0, 2.0]);
        assert_slice_close(&[got], &[0.011760000000000008], 1e-14);
    }

    // ---- 2-component (Beta equivalent) ----

    #[test]
    fn dirichlet_pdf_2d_uniform() {
        // alpha = [1,1] -> uniform on [0,1], PDF = 1.0
        let got = pdf_one(&[0.5, 0.5], &[1.0, 1.0]);
        assert_slice_close(&[got], &[1.0], 1e-15);
    }

    #[test]
    fn dirichlet_pdf_2d_varied() {
        // scipy: 2.1609
        let got = pdf_one(&[0.3, 0.7], &[2.0, 5.0]);
        assert_slice_close(&[got], &[2.1609], 1e-14);
    }

    #[test]
    fn dirichlet_pdf_2d_alpha_lt_1() {
        // scipy: 1.061032953945969
        let got = pdf_one(&[0.1, 0.9], &[0.5, 0.5]);
        assert_slice_close(&[got], &[1.061032953945969], 1e-14);
    }

    // ---- 4-component ----

    #[test]
    fn dirichlet_pdf_4d_uniform() {
        // alpha = [1,1,1,1], PDF = (4-1)! = 6.0
        let got = pdf_one(&[0.25, 0.25, 0.25, 0.25], &[1.0, 1.0, 1.0, 1.0]);
        assert_slice_close(&[got], &[6.0], 1e-15);
    }

    #[test]
    fn dirichlet_pdf_4d_varied() {
        // scipy: 59.779399679999855
        let got = pdf_one(&[0.1, 0.2, 0.3, 0.4], &[2.0, 3.0, 4.0, 5.0]);
        assert_slice_close(&[got], &[59.779399679999855], 1e-13);
    }

    // ---- parameter validation ----

    #[test]
    fn dirichlet_pdf_rejects_x_not_summing_to_1() {
        let result = dirichlet_pdf(&[0.2, 0.3, 0.4], &[1.0, 1.0, 1.0], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn dirichlet_pdf_rejects_negative_x() {
        let result = dirichlet_pdf(&[-0.1, 0.5, 0.6], &[1.0, 1.0, 1.0], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn dirichlet_pdf_rejects_dimension_mismatch() {
        let result = dirichlet_pdf(&[0.5, 0.5], &[1.0, 1.0, 1.0], None, None);
        assert!(result.is_err());
    }
}
