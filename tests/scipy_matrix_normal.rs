// SciPy reference tests for Matrix Normal PDF.
//
// Reference values generated from SciPy 1.16.1:
//   scipy.stats.matrix_normal(mean=M, rowcov=U, colcov=V).pdf(X)
//   scipy.stats.matrix_normal(mean=M, rowcov=U, colcov=V).logpdf(X)
//
// Parameters used:
//   M = [[1,2],[3,4]], U = [[1,0.2],[0.2,1]], V = [[1,0.5],[0.5,1]]

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_matrix_normal_tests {
    use super::util::assert_close;
    use minarrow::Vec64;
    use simd_kernels::kernels::scientific::distributions::multivariate::matrix_normal_pdf;

    fn pdf_one(x: &[&[f64]], mean: &[&[f64]], row_cov: &[&[f64]], col_cov: &[&[f64]]) -> f64 {
        let result = matrix_normal_pdf(
            x.iter().copied().collect::<Vec64<_>>(),
            mean.iter().copied().collect::<Vec64<_>>(),
            row_cov.iter().copied().collect::<Vec64<_>>(),
            col_cov.iter().copied().collect::<Vec64<_>>(),
            None, None,
        ).unwrap();
        result.data[0]
    }

    const MEAN: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0]];
    const ROW_COV: &[&[f64]] = &[&[1.0, 0.2], &[0.2, 1.0]];
    const COL_COV: &[&[f64]] = &[&[1.0, 0.5], &[0.5, 1.0]];

    #[test]
    fn matrix_normal_pdf_at_mean() {
        // scipy: 3.518096654247840349e-02
        let got = pdf_one(MEAN, MEAN, ROW_COV, COL_COV);
        assert_close(got, 3.518096654247840349e-02, 1e-15);
    }

    #[test]
    fn matrix_normal_pdf_near_mean() {
        // scipy: 2.664835533922778679e-02
        let x: &[&[f64]] = &[&[1.5, 2.5], &[3.5, 4.5]];
        let got = pdf_one(x, MEAN, ROW_COV, COL_COV);
        assert_close(got, 2.664835533922778679e-02, 1e-15);
    }

    #[test]
    fn matrix_normal_pdf_at_zero() {
        // scipy: 2.783797625035480192e-06
        let x: &[&[f64]] = &[&[0.0, 0.0], &[0.0, 0.0]];
        let got = pdf_one(x, MEAN, ROW_COV, COL_COV);
        assert_close(got, 2.783797625035480192e-06, 1e-14);
    }

    #[test]
    fn matrix_normal_pdf_far_from_mean() {
        // scipy: 2.783797625035480192e-06 (symmetric about mean for this case)
        let x: &[&[f64]] = &[&[5.0, 5.0], &[5.0, 5.0]];
        let got = pdf_one(x, MEAN, ROW_COV, COL_COV);
        assert_close(got, 2.783797625035480192e-06, 1e-14);
    }
}
