// SciPy reference tests for Inverse-Wishart PDF.
//
// Reference values generated from SciPy 1.16.1 (scipy.stats.invwishart).

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_inv_wishart_tests {
    use super::util::assert_slice_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::inv_wishart_pdf;

    fn pdf_one(x_rows: &[&[f64]], df: f64, scale_rows: &[&[f64]]) -> f64 {
        let x_vec: Vec<&[f64]> = x_rows.to_vec();
        let s_vec: Vec<&[f64]> = scale_rows.to_vec();
        let result = inv_wishart_pdf(x_vec, df, s_vec, None, None).unwrap();
        result[0]
    }

    // ---- 2x2, identity scale (Psi=I), df=5 ----

    #[test]
    fn inv_wishart_pdf_2x2_identity_df5_spd1() {
        // X = [[2, 0.5], [0.5, 3]]
        let got = pdf_one(
            &[&[2.0, 0.5], &[0.5, 3.0]],
            5.0,
            &[&[1.0, 0.0], &[0.0, 1.0]],
        );
        assert_slice_close(&[got], &[7.854957128521874e-06], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_identity_df5_spd2() {
        // X = [[5, 1], [1, 5]]
        let got = pdf_one(
            &[&[5.0, 1.0], &[1.0, 5.0]],
            5.0,
            &[&[1.0, 0.0], &[0.0, 1.0]],
        );
        assert_slice_close(&[got], &[3.245756247333693e-08], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_identity_df5_eye() {
        // X = I
        let got = pdf_one(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            5.0,
            &[&[1.0, 0.0], &[0.0, 1.0]],
        );
        assert_slice_close(&[got], &[0.004879152627026595], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_identity_df5_large() {
        // X = [[10, 2], [2, 8]]
        let got = pdf_one(
            &[&[10.0, 2.0], &[2.0, 8.0]],
            5.0,
            &[&[1.0, 0.0], &[0.0, 1.0]],
        );
        assert_slice_close(&[got], &[3.5314650033593246e-10], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_identity_df5_small() {
        // X = [[0.5, 0.1], [0.1, 0.5]]
        let got = pdf_one(
            &[&[0.5, 0.1], &[0.1, 0.5]],
            5.0,
            &[&[1.0, 0.0], &[0.0, 1.0]],
        );
        assert_slice_close(&[got], &[0.497752841696577], 1e-14);
    }

    // ---- 2x2, non-identity Psi = [[3, 0.5], [0.5, 2]], df=5 ----

    #[test]
    fn inv_wishart_pdf_2x2_nonid_df5_spd1() {
        let got = pdf_one(
            &[&[2.0, 0.5], &[0.5, 3.0]],
            5.0,
            &[&[3.0, 0.5], &[0.5, 2.0]],
        );
        assert_slice_close(&[got], &[0.00032439779913923436], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_nonid_df5_spd2() {
        let got = pdf_one(
            &[&[5.0, 1.0], &[1.0, 5.0]],
            5.0,
            &[&[3.0, 0.5], &[0.5, 2.0]],
        );
        assert_slice_close(&[got], &[1.922278415809418e-06], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_nonid_df5_eye() {
        let got = pdf_one(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            5.0,
            &[&[3.0, 0.5], &[0.5, 2.0]],
        );
        assert_slice_close(&[got], &[0.08631222109355495], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_nonid_df5_large() {
        let got = pdf_one(
            &[&[10.0, 2.0], &[2.0, 8.0]],
            5.0,
            &[&[3.0, 0.5], &[0.5, 2.0]],
        );
        assert_slice_close(&[got], &[2.3908465641313583e-08], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_2x2_nonid_df5_small() {
        let got = pdf_one(
            &[&[0.5, 0.1], &[0.1, 0.5]],
            5.0,
            &[&[3.0, 0.5], &[0.5, 2.0]],
        );
        assert_slice_close(&[got], &[2.135458426676462], 1e-14);
    }

    // ---- 3x3, identity scale, df=6 ----

    #[test]
    fn inv_wishart_pdf_3x3_identity_df6_spd1() {
        let got = pdf_one(
            &[&[3.0, 0.5, 0.2], &[0.5, 4.0, 0.3], &[0.2, 0.3, 2.0]],
            6.0,
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
        );
        assert_slice_close(&[got], &[1.145347331312696e-11], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_3x3_identity_df6_spd2() {
        let got = pdf_one(
            &[&[5.0, 1.0, 0.0], &[1.0, 5.0, 1.0], &[0.0, 1.0, 5.0]],
            6.0,
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
        );
        assert_slice_close(&[got], &[4.775386647247757e-15], 1e-14);
    }

    #[test]
    fn inv_wishart_pdf_3x3_identity_df6_eye() {
        let got = pdf_one(
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
            6.0,
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
        );
        assert_slice_close(&[got], &[2.9437255120499094e-05], 1e-14);
    }

    // ---- parameter validation ----

    #[test]
    fn inv_wishart_pdf_rejects_df_too_small() {
        // df must be > p-1; for 2x2 with df=1, should fail
        let result = inv_wishart_pdf(
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            1.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn inv_wishart_pdf_rejects_non_spd_scale() {
        let result = inv_wishart_pdf(
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            5.0,
            vec![&[1.0, 3.0], &[3.0, 1.0]],
            None,
            None,
        );
        assert!(result.is_err());
    }
}
