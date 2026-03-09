// SciPy + mpmath reference tests for Wishart PDF.
//
// Reference values generated from SciPy 1.16.1 and cross-checked with
// mpmath (50-digit precision) where noted.

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_wishart_tests {
    use super::util::assert_slice_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::wishart_pdf;

    // ---- 2x2, identity scale, df=5 ----
    // x: each entry is a flat d*d row-major observation
    // scale: d rows of the scale matrix

    #[test]
    fn wishart_pdf_2x2_identity_df5_spd1() {
        // W = [[2, 0.5], [0.5, 3]], scipy = 0.006259945115578187
        // mpmath: 0.0062599451155781906863
        let result = wishart_pdf(
            vec![&[2.0, 0.5, 0.5, 3.0]],
            5.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.006259945115578187], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_identity_df5_spd2() {
        // W = [[5, 1], [1, 5]], scipy = 0.0021447551423913074
        // mpmath: 0.0021447551423913089395
        let result = wishart_pdf(
            vec![&[5.0, 1.0, 1.0, 5.0]],
            5.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.0021447551423913074], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_identity_df5_eye() {
        // W = I, scipy = 0.004879152627026595
        let result = wishart_pdf(
            vec![&[1.0, 0.0, 0.0, 1.0]],
            5.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.004879152627026595], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_identity_df5_large() {
        // W = [[10, 2], [2, 8]], scipy = 0.0001243947755271837
        let result = wishart_pdf(
            vec![&[10.0, 2.0, 2.0, 8.0]],
            5.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.0001243947755271837], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_identity_df5_small() {
        // W = [[0.5, 0.1], [0.1, 0.5]], scipy = 0.0019306470526010778
        let result = wishart_pdf(
            vec![&[0.5, 0.1, 0.1, 0.5]],
            5.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.0019306470526010778], 1e-14);
    }

    // ---- 2x2, non-identity scale [[2, 0.5], [0.5, 1]], df=4 ----

    #[test]
    fn wishart_pdf_2x2_nonid_df4_spd1() {
        // W = [[2, 0.5], [0.5, 3]], scipy = 0.0036549962241230593
        let result = wishart_pdf(
            vec![&[2.0, 0.5, 0.5, 3.0]],
            4.0,
            vec![&[2.0, 0.5], &[0.5, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.0036549962241230593], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_nonid_df4_spd2() {
        // W = [[5, 1], [1, 5]], scipy = 0.001165766943191605
        let result = wishart_pdf(
            vec![&[5.0, 1.0, 1.0, 5.0]],
            4.0,
            vec![&[2.0, 0.5], &[0.5, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.001165766943191605], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_nonid_df4_eye() {
        // W = I, scipy = 0.005513553967629443
        let result = wishart_pdf(
            vec![&[1.0, 0.0, 0.0, 1.0]],
            4.0,
            vec![&[2.0, 0.5], &[0.5, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.005513553967629443], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_nonid_df4_large() {
        // W = [[10, 2], [2, 8]], scipy = 0.00011914382390302654
        let result = wishart_pdf(
            vec![&[10.0, 2.0, 2.0, 8.0]],
            4.0,
            vec![&[2.0, 0.5], &[0.5, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.00011914382390302654], 1e-14);
    }

    #[test]
    fn wishart_pdf_2x2_nonid_df4_small() {
        // W = [[0.5, 0.1], [0.1, 0.5]], scipy = 0.004266501189361621
        let result = wishart_pdf(
            vec![&[0.5, 0.1, 0.1, 0.5]],
            4.0,
            vec![&[2.0, 0.5], &[0.5, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.004266501189361621], 1e-14);
    }

    // ---- 3x3, identity scale, df=5 ----

    #[test]
    fn wishart_pdf_3x3_identity_df5_spd1() {
        // W = [[3, 0.5, 0.2], [0.5, 4, 0.3], [0.2, 0.3, 2]], scipy = 4.499163035113307e-05
        let result = wishart_pdf(
            vec![&[3.0, 0.5, 0.2, 0.5, 4.0, 0.3, 0.2, 0.3, 2.0]],
            5.0,
            vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[4.499163035113307e-05], 1e-14);
    }

    #[test]
    fn wishart_pdf_3x3_identity_df5_spd2() {
        // W = [[5, 1, 0], [1, 5, 1], [0, 1, 5]], scipy = 4.994699780721471e-06
        let result = wishart_pdf(
            vec![&[5.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 5.0]],
            5.0,
            vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[4.994699780721471e-06], 1e-14);
    }

    #[test]
    fn wishart_pdf_3x3_identity_df5_eye() {
        // W = I, scipy = 0.00018790025098449037
        let result = wishart_pdf(
            vec![&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            5.0,
            vec![&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[0.00018790025098449037], 1e-14);
    }

    // ---- 3x3, non-identity scale [[2, 0.5, 0.1], [0.5, 3, 0.2], [0.1, 0.2, 1]], df=6 ----

    #[test]
    fn wishart_pdf_3x3_nonid_df6_spd1() {
        // W = [[3, 0.5, 0.2], [0.5, 4, 0.3], [0.2, 0.3, 2]], scipy = 1.4734513193540317e-06
        let result = wishart_pdf(
            vec![&[3.0, 0.5, 0.2, 0.5, 4.0, 0.3, 0.2, 0.3, 2.0]],
            6.0,
            vec![&[2.0, 0.5, 0.1], &[0.5, 3.0, 0.2], &[0.1, 0.2, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[1.4734513193540317e-06], 1e-14);
    }

    #[test]
    fn wishart_pdf_3x3_nonid_df6_spd2() {
        // W = [[5, 1, 0], [1, 5, 1], [0, 1, 5]], scipy = 8.58486953583685e-07
        let result = wishart_pdf(
            vec![&[5.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 5.0]],
            6.0,
            vec![&[2.0, 0.5, 0.1], &[0.5, 3.0, 0.2], &[0.1, 0.2, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[8.58486953583685e-07], 1e-14);
    }

    #[test]
    fn wishart_pdf_3x3_nonid_df6_eye() {
        // W = I, scipy = 2.827363883127632e-07
        let result = wishart_pdf(
            vec![&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            6.0,
            vec![&[2.0, 0.5, 0.1], &[0.5, 3.0, 0.2], &[0.1, 0.2, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(&result, &[2.827363883127632e-07], 1e-14);
    }

    // ---- batched: two observations in one call ----

    #[test]
    fn wishart_pdf_2x2_batched() {
        // W1 = [[2, 0.5], [0.5, 3]], W2 = [[5, 1], [1, 5]]
        // scipy = [0.006259945115578187, 0.0021447551423913074]
        let result = wishart_pdf(
            vec![&[2.0, 0.5, 0.5, 3.0], &[5.0, 1.0, 1.0, 5.0]],
            5.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        )
        .unwrap();
        assert_slice_close(
            &result,
            &[0.006259945115578187, 0.0021447551423913074],
            1e-14,
        );
    }

    // ---- parameter validation ----

    #[test]
    fn wishart_pdf_rejects_df_too_small() {
        // df must be >= d; for 2x2 with df=1, should fail
        let result = wishart_pdf(
            vec![&[1.0, 0.0, 0.0, 1.0]],
            1.0,
            vec![&[1.0, 0.0], &[0.0, 1.0]],
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn wishart_pdf_rejects_non_spd() {
        // scale is not positive definite
        let result = wishart_pdf(
            vec![&[1.0, 0.0, 0.0, 1.0]],
            5.0,
            vec![&[1.0, 2.0], &[2.0, 1.0]],
            None,
            None,
        );
        assert!(result.is_err());
    }
}
