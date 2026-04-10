// SciPy reference tests for Multivariate Student-t PDF/logPDF.
//
// Reference values generated from SciPy 1.16.1:
//   scipy.stats.multivariate_t(loc=mean, shape=scale, df=df).pdf(x)
//   scipy.stats.multivariate_t(loc=mean, shape=scale, df=df).logpdf(x)

#![allow(clippy::excessive_precision)]

mod util;

#[cfg(feature = "probability_distributions")]
#[cfg(feature = "linear_algebra")]
mod scipy_mvt_tests {
    use super::util::assert_close;
    use simd_kernels::kernels::scientific::distributions::multivariate::{mvt_logpdf, mvt_pdf};

    fn pdf_one(x: &[f64], mean: &[f64], scale: &[&[f64]], df: f64) -> f64 {
        let scale_v: Vec<&[f64]> = scale.to_vec();
        let result = mvt_pdf(x, mean, scale_v, df, None, None).unwrap();
        result.data[0]
    }

    fn logpdf_one(x: &[f64], mean: &[f64], scale: &[&[f64]], df: f64) -> f64 {
        let scale_v: Vec<&[f64]> = scale.to_vec();
        let result = mvt_logpdf(x, mean, scale_v, df, None, None).unwrap();
        result.data[0]
    }

    // mean=[0,0], scale=[[1,0.3],[0.3,1]], df=5

    #[test]
    fn mvt_pdf_2d_at_origin() {
        // scipy: 1.668397135325737080e-01
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = pdf_one(&[0.0, 0.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, 1.668397135325737080e-01, 1e-15);
    }

    #[test]
    fn mvt_logpdf_2d_at_origin() {
        // scipy: -1.790721726673724756e+00
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = logpdf_one(&[0.0, 0.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, -1.790721726673724756e+00, 1e-15);
    }

    #[test]
    fn mvt_pdf_2d_unit_x() {
        // scipy: 8.323651389846004056e-02
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = pdf_one(&[1.0, 0.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, 8.323651389846004056e-02, 1e-15);
    }

    #[test]
    fn mvt_logpdf_2d_unit_x() {
        // scipy: -2.486069158457918871e+00
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = logpdf_one(&[1.0, 0.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, -2.486069158457918871e+00, 1e-15);
    }

    #[test]
    fn mvt_pdf_2d_symmetric_unit_y() {
        // scipy: 8.323651389846004056e-02 (same as (1,0) by symmetry of identity-ish scale)
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = pdf_one(&[0.0, 1.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, 8.323651389846004056e-02, 1e-15);
    }

    #[test]
    fn mvt_pdf_2d_negative_quadrant() {
        // scipy: 6.524240227720562446e-02
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = pdf_one(&[-1.0, -1.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, 6.524240227720562446e-02, 1e-15);
    }

    #[test]
    fn mvt_logpdf_2d_negative_quadrant() {
        // scipy: -2.729645679755102528e+00
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = logpdf_one(&[-1.0, -1.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, -2.729645679755102528e+00, 1e-15);
    }

    #[test]
    fn mvt_pdf_2d_off_axis() {
        // scipy: 1.045508627760821097e-01
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = pdf_one(&[0.5, -0.5], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, 1.045508627760821097e-01, 1e-15);
    }

    #[test]
    fn mvt_pdf_2d_tail() {
        // scipy: 1.604777728436940698e-03
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = pdf_one(&[3.0, 3.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, 1.604777728436940698e-03, 1e-14);
    }

    #[test]
    fn mvt_logpdf_2d_tail() {
        // scipy: -6.434770018945538794e+00
        let scale = [[1.0, 0.3].as_slice(), [0.3, 1.0].as_slice()];
        let got = logpdf_one(&[3.0, 3.0], &[0.0, 0.0], &scale, 5.0);
        assert_close(got, -6.434770018945538794e+00, 1e-14);
    }
}
