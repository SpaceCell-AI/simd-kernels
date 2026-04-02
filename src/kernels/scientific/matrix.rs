// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Matrix Operations Kernels Module** - *Linear Algebra and Numerical Computing*
//!
//! **************************************************************************************************
//! ⚠️ Warning: This module has not been fully tested, and is not ready for production use.
//! This warning applies to all multivariate kernels in *SIMD-kernels*, which are to be finalised
//! in an upcoming release.
//! **************************************************************************************************
//!
//! All matrices are stored in **column-major** order (Fortran/BLAS convention).
//! Leading dimension parameters (`lda`, `ldb`, etc.) specify the physical stride
//! between consecutive columns.

use crate::kernels::aggregate::{neumaier_add, reduce_min_max_f64};
use crate::kernels::scientific::blas_lapack;

use minarrow::enums::error::KernelError;

// **************************************************
// Matrix-vector and matrix-matrix products
// **************************************************

/// Matrix-vector product: y <- alpha*A*x + beta*y (column-major).
///
/// Computes a general matrix-vector multiplication (GEMV) with scaling and accumulation:
///     y <- alpha * op(A) * x + beta * y
/// where:
///     - `op(A)` is either A or its transpose, controlled by `trans`.
///     - `alpha` scales the matrix-vector product. Typical value: 1.0.
///     - `beta` scales the initial contents of y. Typical value: 0.0.
///
/// Storage and Layout
/// - All data is in column-major order (BLAS/Fortran-compatible).
/// - The matrix A is passed as a contiguous buffer of length >= m * n, with leading dimension `lda = m`.
/// - Strides for x and y are unit (`incx = incy = 1`).
///
/// Arguments
/// - `m`:    Number of rows of A (and y if not transposed; x if transposed).
/// - `n`:    Number of columns of A (and x if not transposed; y if transposed).
/// - `a`:    Matrix buffer for A, column-major, of length >= m * n.
/// - `x`:    Input vector x, length: n if not transposed; m if transposed.
/// - `y`:    Output vector y, length: m if not transposed; n if transposed. Mutated in place.
/// - `alpha`: Scalar multiplier for the matrix-vector product. Default in high-level APIs: 1.0.
/// - `beta`:  Scalar multiplier for the initial y. Default in high-level APIs: 0.0.
/// - `trans`: If true, use the transpose of A; if false, use A as-is.
///
/// Returns
/// - `Ok(())` on success.
/// - `Err(KernelError::InvalidArguments)` if any buffer is too small or arguments are invalid.
#[inline(always)]
pub fn matrix_vector_product(
    m: i32,
    n: i32,
    a: &[f64],
    x: &[f64],
    y: &mut [f64],
    alpha: f64,
    beta: f64,
    trans: bool,
) -> Result<(), KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }

    let (rows_a, cols_a, rows_y, rows_x) = if trans {
        (m, n, n, m) // At: result len = n
    } else {
        (m, n, m, n) // standard GEMV
    };

    // Size checks -----------------------------------------------------------
    if a.len() < (rows_a * cols_a) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    if x.len() < rows_x as usize {
        return Err(KernelError::InvalidArguments("x buffer too small".into()));
    }
    if y.len() < rows_y as usize {
        return Err(KernelError::InvalidArguments("y buffer too small".into()));
    }

    blas_lapack::gemv(
        m, n, a, m, // lda = rows of A (column-major)
        x, 1, // incx
        y, 1, // incy
        alpha, beta, trans,
    )
    .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Matrix-matrix product: C <- alpha*A*B + beta*C (column-major).
///
/// Performs a general matrix-matrix multiplication with optional scaling and accumulation.
///
/// This routine computes:
///     C <- alpha * op(A) * op(B) + beta * C
/// where:
///     - `op(A)` is either A or its transpose, controlled by `trans_a`
///     - `op(B)` is either B or its transpose, controlled by `trans_b`
///     - `alpha` scales the matrix product; default in high-level APIs is typically 1.0
///     - `beta`  scales the initial contents of C; default in high-level APIs is typically 0.0
///
/// Storage:
///     - All matrices are stored in column-major order (Fortran/BLAS-compatible).
///     - Leading dimension (`lda`, `ldb`, `ldc`) specifies the physical stride between columns.
///     - For a contiguous matrix with `m` rows, `ld* = m`.
///
/// Arguments:
/// - `m`:      Number of rows of the output matrix C and of op(A).
/// - `n`:      Number of columns of the output matrix C and of op(B).
/// - `k`:      Shared inner dimension (`op(A).cols`, `op(B).rows`).
/// - `alpha`:  Scalar multiplier for op(A) * op(B).
/// - `a`:      Input buffer for matrix A.
/// - `lda`:    Leading dimension (column stride) of A. Must be >= rows in op(A).
/// - `b`:      Input buffer for matrix B.
/// - `ldb`:    Leading dimension (column stride) of B. Must be >= rows in op(B).
/// - `beta`:   Scalar multiplier for the existing data in C.
/// - `c`:      Output buffer for matrix C (mutated in place).
/// - `ldc`:    Leading dimension (column stride) of C. Must be >= m.
/// - `trans_a`: If true, A is transposed before multiplication; otherwise, not transposed.
/// - `trans_b`: If true, B is transposed before multiplication; otherwise, not transposed.
///
/// Returns:
///     - `Ok(())` on success.
///     - `Err(KernelError::InvalidArguments)` if any buffer is too small or dimensions are invalid.
///
/// Notes:
/// - This API is identical to BLAS `dgemm` in both semantics and memory layout.
/// - For typical users, set `alpha=1.0` and `beta=0.0` for the usual matrix product.
#[inline(always)]
pub fn matmul(
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
    trans_a: bool,
    trans_b: bool,
) -> Result<(), KernelError> {
    // Sanity checks
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n, k must be positive".into(),
        ));
    }
    if lda < if trans_a { k } else { m } {
        return Err(KernelError::InvalidArguments("lda is too small".into()));
    }
    if ldb < if trans_b { n } else { k } {
        return Err(KernelError::InvalidArguments("ldb is too small".into()));
    }
    if ldc < m {
        return Err(KernelError::InvalidArguments("ldc is too small".into()));
    }

    // Buffer length checks
    let (_rows_a, cols_a) = if trans_a { (k, m) } else { (m, k) };
    let (_rows_b, cols_b) = if trans_b { (n, k) } else { (k, n) };

    if a.len() < (lda * cols_a) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    if b.len() < (ldb * cols_b) as usize {
        return Err(KernelError::InvalidArguments("B buffer too small".into()));
    }
    if c.len() < (ldc * n) as usize {
        return Err(KernelError::InvalidArguments("C buffer too small".into()));
    }

    blas_lapack::blocked_gemm(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, trans_a, trans_b,
    )
    .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Computes the minimum and maximum element in a matrix.
#[inline(always)]
pub fn matrix_min_max(
    data: &[f64],
    rows: usize,
    cols: usize,
    lda: usize, // usually same as 'rows'
) -> (f64, f64) {
    let n_elems = rows * cols;
    if lda == rows && data.len() >= n_elems {
        reduce_min_max_f64(&data[..n_elems], None, Some(0))
            .unwrap_or((f64::INFINITY, f64::NEG_INFINITY))
    } else {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for col in 0..cols {
            let offset = col * lda;
            let col_slice = &data[offset..offset + rows];
            if let Some((cmin, cmax)) = reduce_min_max_f64(col_slice, None, Some(0)) {
                min = min.min(cmin);
                max = max.max(cmax);
            }
        }
        (min, max)
    }
}

// **************************************************
// Decompositions
// **************************************************

/// LU decomposition with partial pivoting: PA = LU.
///
/// Factors the m-by-n matrix A into:
///   P * A = L * U
/// where P is a permutation matrix encoded as pivot indices, L is unit
/// lower-triangular, and U is upper-triangular.
///
/// On exit, A is overwritten with the L and U factors. The unit diagonal of L
/// is not stored. `ipiv` is populated with 1-based pivot indices (LAPACK convention):
/// row i was swapped with row ipiv[i].
///
/// Column-major layout. `lda` is the leading dimension of A (>= m).
///
/// Arguments
/// - `m`:    Number of rows of A.
/// - `n`:    Number of columns of A.
/// - `a`:    Matrix buffer, length >= lda * n. Overwritten with L and U factors.
/// - `lda`:  Leading dimension of A (>= m).
/// - `ipiv`: Pivot index buffer, length >= min(m, n). Populated on exit.
pub fn _lu(m: i32, n: i32, a: &mut [f64], lda: i32, ipiv: &mut [i32]) -> Result<(), KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    let k = m.min(n) as usize;
    if ipiv.len() < k {
        return Err(KernelError::InvalidArguments(
            "ipiv buffer too small".into(),
        ));
    }

    blas_lapack::lu_with_piv(m, n, a, lda, ipiv)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// QR decomposition via Householder reflections.
///
/// Factors the m-by-n matrix A (m >= n) into:
///   A = Q * R
/// where Q is an m-by-m orthogonal matrix represented implicitly by Householder
/// vectors stored below the diagonal of A, and R is stored in the upper triangle.
///
/// On exit, `a` is overwritten: the upper triangle contains R, and the lower
/// portion (below the diagonal) contains the Householder vectors. `taus` is
/// populated with the scalar factors of each Householder reflector.
///
/// To form Q explicitly, call `qr_q` with the output of this function.
///
/// Column-major layout. `lda` is the leading dimension of A (>= m).
///
/// Arguments
/// - `m`:    Number of rows of A.
/// - `n`:    Number of columns of A.
/// - `a`:    Matrix buffer, length >= lda * n. Overwritten on exit.
/// - `lda`:  Leading dimension of A (>= m).
/// - `taus`: Householder scalar factors, length >= min(m, n).
pub fn qr(m: i32, n: i32, a: &mut [f64], lda: i32, taus: &mut [f64]) -> Result<(), KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    let k = m.min(n) as usize;
    if taus.len() < k {
        return Err(KernelError::InvalidArguments(
            "taus buffer too small".into(),
        ));
    }

    blas_lapack::qr_block(m, n, a, lda, taus)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Forms the orthogonal matrix Q from a prior QR decomposition.
///
/// Given the Householder vectors and tau scalars produced by `qr`, this
/// function explicitly forms the first `k` columns of Q. On exit, `a` is
/// overwritten with Q.
///
/// Column-major layout. `lda` is the leading dimension of A (>= m).
///
/// Arguments
/// - `m`:    Number of rows of Q.
/// - `n`:    Number of columns of Q to form.
/// - `k`:    Number of Householder reflectors (from the original QR call).
/// - `a`:    On entry: output of `qr`. On exit: columns of Q. Length >= lda * n.
/// - `lda`:  Leading dimension of A (>= m).
/// - `taus`: Householder scalar factors from `qr`, length >= k.
pub fn qr_q(
    m: i32,
    n: i32,
    k: i32,
    a: &mut [f64],
    lda: i32,
    taus: &[f64],
) -> Result<(), KernelError> {
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n, k must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    if taus.len() < k as usize {
        return Err(KernelError::InvalidArguments(
            "taus buffer too small".into(),
        ));
    }

    blas_lapack::qr_form_q(m, n, k, a, lda, taus)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Cholesky decomposition for symmetric positive-definite matrices: A = L * Lt.
///
/// Computes the lower-triangular Cholesky factor L such that A = L * Lt.
/// On exit, the lower triangle of `a` is overwritten with L. The upper
/// triangle is not modified.
///
/// Column-major layout. `lda` is the leading dimension of A (>= n).
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `a`:    Symmetric positive-definite matrix buffer, length >= lda * n.
///           Lower triangle overwritten with L on exit.
/// - `lda`:  Leading dimension of A (>= n).
pub fn cholesky(n: i32, a: &mut [f64], lda: i32) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    blas_lapack::spd_cholesky(n, a, lda).map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Singular value decomposition: A = U * diag(S) * Vt.
///
/// Computes the SVD of an m-by-n matrix A. When `economy` is true, only the
/// first min(m,n) columns of U and rows of Vt are computed (the "thin" SVD).
/// When false, full U (m-by-m) and Vt (n-by-n) are computed.
///
/// On exit, `a` is destroyed. `s` contains the singular values in descending
/// order. `u` contains U, `vt` contains Vt.
///
/// Column-major layout.
///
/// Arguments
/// - `m`:      Number of rows of A.
/// - `n`:      Number of columns of A.
/// - `a`:      Matrix buffer, length >= lda * n. Destroyed on exit.
/// - `lda`:    Leading dimension of A (>= m).
/// - `s`:      Singular values buffer, length >= min(m, n).
/// - `u`:      Left singular vectors buffer.
///             Economy: length >= ldu * min(m,n). Full: length >= ldu * m.
/// - `ldu`:    Leading dimension of U (>= m).
/// - `vt`:     Right singular vectors buffer (transposed).
///             Economy: length >= ldvt * n. Full: length >= ldvt * n.
/// - `ldvt`:   Leading dimension of Vt. Economy: >= min(m,n). Full: >= n.
/// - `economy`: If true, compute thin SVD ('S'). If false, compute full SVD ('A').
pub fn svd(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    s: &mut [f64],
    u: &mut [f64],
    ldu: i32,
    vt: &mut [f64],
    ldvt: i32,
    economy: bool,
) -> Result<(), KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    let k = m.min(n);
    if s.len() < k as usize {
        return Err(KernelError::InvalidArguments("s buffer too small".into()));
    }
    if ldu < m {
        return Err(KernelError::InvalidArguments("ldu must be >= m".into()));
    }

    let (jobu, jobvt) = if economy { (b'S', b'S') } else { (b'A', b'A') };

    // Validate U buffer size
    let u_cols = if economy { k } else { m };
    if u.len() < (ldu * u_cols) as usize {
        return Err(KernelError::InvalidArguments("U buffer too small".into()));
    }

    // Validate Vt buffer size
    let min_ldvt = if economy { k } else { n };
    if ldvt < min_ldvt {
        return Err(KernelError::InvalidArguments("ldvt too small".into()));
    }
    if vt.len() < (ldvt * n) as usize {
        return Err(KernelError::InvalidArguments("Vt buffer too small".into()));
    }

    blas_lapack::svd_block(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Symmetric eigendecomposition: A = Q * diag(eigenvalues) * Qt.
///
/// Computes all eigenvalues and eigenvectors of a symmetric n-by-n matrix.
/// On exit, `a` is overwritten with the orthogonal eigenvector matrix Q
/// (column j is the eigenvector for eigenvalue j). `eigenvalues` is populated
/// in ascending order.
///
/// The upper triangle of A is read on entry (LAPACK 'U' convention).
///
/// Column-major layout. `lda` is the leading dimension of A (>= n).
///
/// Arguments
/// - `n`:           Order of the matrix (n-by-n).
/// - `a`:           Symmetric matrix buffer, length >= lda * n.
///                  Overwritten with eigenvectors on exit.
/// - `lda`:         Leading dimension of A (>= n).
/// - `eigenvalues`: Eigenvalue buffer, length >= n. Ascending order on exit.
pub fn eig_symmetric(
    n: i32,
    a: &mut [f64],
    lda: i32,
    eigenvalues: &mut [f64],
) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    if eigenvalues.len() < n as usize {
        return Err(KernelError::InvalidArguments(
            "eigenvalues buffer too small".into(),
        ));
    }

    blas_lapack::symeig_full(n, a, lda, eigenvalues)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

// **************************************************
// Solvers
// **************************************************

/// General linear system solver: A * X = B using LU factorisation.
///
/// Factors A via LU with partial pivoting, then solves for X.
/// On exit, `a` is overwritten with the LU factors and `b` is overwritten
/// with the solution X. `ipiv` receives the pivot indices.
///
/// Column-major layout. Both A (n-by-n) and B (n-by-nrhs) must be square-compatible.
///
/// Arguments
/// - `n`:    Order of the matrix A (n-by-n).
/// - `nrhs`: Number of right-hand-side columns in B.
/// - `a`:    Matrix buffer, length >= lda * n. Overwritten with LU factors.
/// - `lda`:  Leading dimension of A (>= n).
/// - `ipiv`: Pivot index buffer, length >= n.
/// - `b`:    Right-hand-side matrix, length >= ldb * nrhs. Overwritten with solution.
/// - `ldb`:  Leading dimension of B (>= n).
pub fn solve(
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f64],
    ldb: i32,
) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if nrhs <= 0 {
        return Err(KernelError::InvalidArguments(
            "nrhs must be positive".into(),
        ));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if ldb < n {
        return Err(KernelError::InvalidArguments("ldb must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    if ipiv.len() < n as usize {
        return Err(KernelError::InvalidArguments(
            "ipiv buffer too small".into(),
        ));
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err(KernelError::InvalidArguments("B buffer too small".into()));
    }

    // Factor A = P * L * U
    _lu(n, n, a, lda, ipiv)?;

    // Solve using the LU factorisation
    blas_lapack::getrs(n, nrhs, a, lda, ipiv, b, ldb)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// SPD linear system solver: Sigma * X = B using Cholesky factor L.
///
/// Solves the system where `l` is the pre-computed Cholesky factor (lower-triangular)
/// of a symmetric positive-definite matrix. On exit, `b` is overwritten with the solution X.
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `nrhs`: Number of right-hand-side columns in B.
/// - `l`:    Cholesky factor L (lower-triangular), length >= ldl * n.
/// - `ldl`:  Leading dimension of L (>= n).
/// - `b`:    Right-hand-side matrix, length >= ldb * nrhs. Overwritten with solution.
/// - `ldb`:  Leading dimension of B (>= n).
pub fn spd_solve(
    n: i32,
    nrhs: i32,
    l: &[f64],
    ldl: i32,
    b: &mut [f64],
    ldb: i32,
) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if nrhs <= 0 {
        return Err(KernelError::InvalidArguments(
            "nrhs must be positive".into(),
        ));
    }
    if ldl < n {
        return Err(KernelError::InvalidArguments("ldl must be >= n".into()));
    }
    if ldb < n {
        return Err(KernelError::InvalidArguments("ldb must be >= n".into()));
    }
    if l.len() < (ldl * n) as usize {
        return Err(KernelError::InvalidArguments("L buffer too small".into()));
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err(KernelError::InvalidArguments("B buffer too small".into()));
    }

    blas_lapack::spd_solve(n, nrhs, l, ldl, b, ldb)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Triangular solve (upper): U * X = B.
///
/// Solves for X where U is upper-triangular. On exit, `b` is overwritten with the solution.
///
/// Arguments
/// - `n`:    Order of U (n-by-n).
/// - `nrhs`: Number of right-hand-side columns.
/// - `u`:    Upper-triangular matrix, length >= ldu * n.
/// - `ldu`:  Leading dimension of U (>= n).
/// - `b`:    Right-hand-side matrix, length >= ldb * nrhs. Overwritten with solution.
/// - `ldb`:  Leading dimension of B (>= n).
pub fn solve_triangular_upper(
    n: i32,
    nrhs: i32,
    u: &[f64],
    ldu: i32,
    b: &mut [f64],
    ldb: i32,
) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if ldu < n {
        return Err(KernelError::InvalidArguments("ldu must be >= n".into()));
    }
    if ldb < n {
        return Err(KernelError::InvalidArguments("ldb must be >= n".into()));
    }
    if u.len() < (ldu * n) as usize {
        return Err(KernelError::InvalidArguments("U buffer too small".into()));
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err(KernelError::InvalidArguments("B buffer too small".into()));
    }

    blas_lapack::trisolve_upper(n, nrhs, u, ldu, b, ldb)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Triangular solve (lower): L * X = B.
///
/// Solves for X where L is lower-triangular. On exit, `b` is overwritten with the solution.
///
/// Arguments
/// - `n`:    Order of L (n-by-n).
/// - `nrhs`: Number of right-hand-side columns.
/// - `l`:    Lower-triangular matrix, length >= ldl * n.
/// - `ldl`:  Leading dimension of L (>= n).
/// - `b`:    Right-hand-side matrix, length >= ldb * nrhs. Overwritten with solution.
/// - `ldb`:  Leading dimension of B (>= n).
pub fn solve_triangular_lower(
    n: i32,
    nrhs: i32,
    l: &[f64],
    ldl: i32,
    b: &mut [f64],
    ldb: i32,
) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if ldl < n {
        return Err(KernelError::InvalidArguments("ldl must be >= n".into()));
    }
    if ldb < n {
        return Err(KernelError::InvalidArguments("ldb must be >= n".into()));
    }
    if l.len() < (ldl * n) as usize {
        return Err(KernelError::InvalidArguments("L buffer too small".into()));
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err(KernelError::InvalidArguments("B buffer too small".into()));
    }

    blas_lapack::trisolve_lower(n, nrhs, l, ldl, b, ldb)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// General matrix inverse: A -> A_inv in-place.
///
/// Computes the inverse of a general n-by-n matrix using LU factorisation
/// followed by back-substitution (dgetrf + dgetri). On exit, `a` is
/// overwritten with A_inv.
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `a`:    Matrix buffer, length >= lda * n. Overwritten with inverse.
/// - `lda`:  Leading dimension of A (>= n).
pub fn inverse(n: i32, a: &mut [f64], lda: i32) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    let mut ipiv = vec![0i32; n as usize];
    _lu(n, n, a, lda, &mut ipiv)?;

    blas_lapack::getri(n, a, lda, &ipiv).map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// SPD matrix inverse in-place via Cholesky factorisation.
///
/// Computes the inverse of a symmetric positive-definite n-by-n matrix
/// using Cholesky factorisation (dpotrf + dpotri). On exit, `a` is
/// overwritten with A_inv. The result is symmetrised (lower triangle
/// copied to upper triangle).
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `a`:    SPD matrix buffer, length >= lda * n. Overwritten with inverse.
/// - `lda`:  Leading dimension of A (>= n).
pub fn spd_inverse(n: i32, a: &mut [f64], lda: i32) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    // Cholesky factor A = L*Lt
    blas_lapack::spd_cholesky(n, a, lda)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))?;

    // Compute inverse from Cholesky factor
    blas_lapack::spd_inverse(n, a, lda)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))?;

    // Symmetrise: copy lower to upper
    let nu = n as usize;
    let ldau = lda as usize;
    for col in 0..nu {
        for row in 0..col {
            a[row + col * ldau] = a[col + row * ldau];
        }
    }

    Ok(())
}

// **************************************************
// Diagnostics
// **************************************************

/// Determinant via LU factorisation.
///
/// Computes det(A) = product of diagonal(U) * sign(permutation).
/// `a` is destroyed on exit (overwritten with LU factors).
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `a`:    Matrix buffer, length >= lda * n. Destroyed on exit.
/// - `lda`:  Leading dimension of A (>= n).
pub fn determinant(n: i32, a: &mut [f64], lda: i32) -> Result<f64, KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    let mut ipiv = vec![0i32; n as usize];
    _lu(n, n, a, lda, &mut ipiv)?;

    let nu = n as usize;
    let ldau = lda as usize;
    let mut det = 1.0_f64;
    let mut swaps = 0;
    for i in 0..nu {
        det *= a[i + i * ldau]; // diagonal of U
        // LAPACK uses 1-based pivot indices
        if ipiv[i] != (i as i32 + 1) {
            swaps += 1;
        }
    }
    if swaps % 2 == 1 {
        det = -det;
    }

    Ok(det)
}

/// Matrix rank via SVD.
///
/// Computes the numerical rank by counting singular values above `tol`.
/// `a` is destroyed on exit.
///
/// Arguments
/// - `m`:    Number of rows of A.
/// - `n`:    Number of columns of A.
/// - `a`:    Matrix buffer, length >= lda * n. Destroyed on exit.
/// - `lda`:  Leading dimension of A (>= m).
/// - `tol`:  Tolerance for singular value cutoff.
pub fn rank(m: i32, n: i32, a: &mut [f64], lda: i32, tol: f64) -> Result<usize, KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    let k = m.min(n) as usize;
    let mut s = vec![0.0_f64; k];

    // We only need singular values, not vectors. Use 'N' for both.
    let mut u_dummy = [0.0_f64; 1];
    let mut vt_dummy = [0.0_f64; 1];
    blas_lapack::svd_block(
        b'N',
        b'N',
        m,
        n,
        a,
        lda,
        &mut s,
        &mut u_dummy,
        1,
        &mut vt_dummy,
        1,
    )
    .map_err(|e| KernelError::InvalidArguments(e.to_string()))?;

    let r = s.iter().filter(|&&sv| sv > tol).count();
    Ok(r)
}

/// Least-squares solver: minimise ||A*x - b||_2 using QR.
///
/// Solves the overdetermined (m >= n) or underdetermined (m < n) least-squares
/// problem via QR decomposition (LAPACK dgels).
///
/// On exit, `a` is destroyed. The first `n` rows of each column of `b` contain
/// the solution vectors.
///
/// Arguments
/// - `m`:    Number of rows of A.
/// - `n`:    Number of columns of A.
/// - `nrhs`: Number of right-hand-side columns.
/// - `a`:    Matrix buffer, length >= lda * n. Destroyed on exit.
/// - `lda`:  Leading dimension of A (>= m).
/// - `b`:    Right-hand-side matrix, length >= ldb * nrhs. First n rows overwritten with solution.
/// - `ldb`:  Leading dimension of B (>= max(m, n)).
pub fn least_squares(
    m: i32,
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    b: &mut [f64],
    ldb: i32,
) -> Result<(), KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if nrhs <= 0 {
        return Err(KernelError::InvalidArguments(
            "nrhs must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if ldb < m.max(n) {
        return Err(KernelError::InvalidArguments(
            "ldb must be >= max(m, n)".into(),
        ));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err(KernelError::InvalidArguments("B buffer too small".into()));
    }

    blas_lapack::least_squares_qr(m, n, nrhs, a, lda, b, ldb)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

/// Condition number via SVD: kappa(A) = sigma_max / sigma_min.
///
/// Computes the 2-norm condition number of the matrix. `a` is destroyed on exit.
///
/// Arguments
/// - `m`:    Number of rows of A.
/// - `n`:    Number of columns of A.
/// - `a`:    Matrix buffer, length >= lda * n. Destroyed on exit.
/// - `lda`:  Leading dimension of A (>= m).
pub fn condition_number(m: i32, n: i32, a: &mut [f64], lda: i32) -> Result<f64, KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    let k = m.min(n) as usize;
    let mut s = vec![0.0_f64; k];

    let mut u_dummy = [0.0_f64; 1];
    let mut vt_dummy = [0.0_f64; 1];
    blas_lapack::svd_block(
        b'N',
        b'N',
        m,
        n,
        a,
        lda,
        &mut s,
        &mut u_dummy,
        1,
        &mut vt_dummy,
        1,
    )
    .map_err(|e| KernelError::InvalidArguments(e.to_string()))?;

    let s_max = s[0]; // singular values are in descending order
    let s_min = s[k - 1];

    if s_min == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(s_max / s_min)
    }
}

/// Matrix norm.
///
/// Computes the matrix norm of an m-by-n matrix. Pure Rust, no BLAS needed.
///
/// Supported `kind` values:
/// - `"fro"`: Frobenius norm = sqrt(sum of squares of all elements)
/// - `"l1"`:  1-norm = max column sum of absolute values
/// - `"linf"`: Infinity norm = max row sum of absolute values
///
/// Arguments
/// - `m`:    Number of rows of A.
/// - `n`:    Number of columns of A.
/// - `a`:    Matrix buffer, length >= lda * n.
/// - `lda`:  Leading dimension of A (>= m).
/// - `kind`: Norm type: "fro", "l1", or "linf".
pub fn norm(m: i32, n: i32, a: &[f64], lda: i32, kind: &str) -> Result<f64, KernelError> {
    if m <= 0 || n <= 0 {
        return Err(KernelError::InvalidArguments(
            "m, n must be positive".into(),
        ));
    }
    if lda < m {
        return Err(KernelError::InvalidArguments("lda must be >= m".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    let mu = m as usize;
    let nu = n as usize;
    let ldau = lda as usize;

    match kind {
        "fro" => {
            let mut sum = 0.0_f64;
            let mut comp = 0.0_f64;
            for col in 0..nu {
                for row in 0..mu {
                    let v = a[row + col * ldau];
                    neumaier_add(&mut sum, &mut comp, v * v);
                }
            }
            Ok((sum + comp).sqrt())
        }
        "l1" => {
            // Max column sum of absolute values
            let mut max_col_sum = 0.0_f64;
            for col in 0..nu {
                let mut col_sum = 0.0_f64;
                let mut col_comp = 0.0_f64;
                for row in 0..mu {
                    neumaier_add(&mut col_sum, &mut col_comp, a[row + col * ldau].abs());
                }
                let total = col_sum + col_comp;
                if total > max_col_sum {
                    max_col_sum = total;
                }
            }
            Ok(max_col_sum)
        }
        "linf" => {
            // Max row sum of absolute values
            let mut row_sums = vec![0.0_f64; mu];
            let mut row_comps = vec![0.0_f64; mu];
            for col in 0..nu {
                for row in 0..mu {
                    neumaier_add(
                        &mut row_sums[row],
                        &mut row_comps[row],
                        a[row + col * ldau].abs(),
                    );
                }
            }
            let max_row_sum = row_sums
                .iter()
                .zip(row_comps.iter())
                .map(|(s, c)| s + c)
                .fold(0.0_f64, f64::max);
            Ok(max_row_sum)
        }
        _ => Err(KernelError::InvalidArguments(format!(
            "Unknown norm kind '{}'. Use \"fro\", \"l1\", or \"linf\".",
            kind
        ))),
    }
}

// **************************************************
// Tier 2: Covariance, log-determinant, Mahalanobis, triangular inverse
// **************************************************

/// Covariance matrix from an n_obs-by-n_feat observation matrix X (column-major).
///
/// Computes the d x d sample covariance matrix:
///   C = X_c^T * X_c / (n_obs - ddof)
/// where X_c is the mean-centred copy of X.
///
/// Algorithm:
///   1. Compute column means: mean_j = sum(X[:,j]) / n_obs
///   2. Create mean-centred copy X_c = X - means
///   3. Compute C = X_c^T * X_c via blocked_gemm (trans_a=true)
///   4. Scale C /= (n_obs - ddof)
///   5. Symmetrise: copy lower triangle to upper triangle
///
/// Arguments
/// - `n_obs`:  Number of observations (rows of X).
/// - `n_feat`: Number of features (columns of X).
/// - `x`:      Observation matrix buffer, column-major, length >= ldx * n_feat.
/// - `ldx`:    Leading dimension of X (>= n_obs).
/// - `cov`:    Output covariance matrix buffer, column-major, length >= ldc * n_feat.
/// - `ldc`:    Leading dimension of cov (>= n_feat).
/// - `ddof`:   Delta degrees of freedom. Typically 1 for sample covariance.
pub fn covariance(
    n_obs: i32,
    n_feat: i32,
    x: &[f64],
    ldx: i32,
    cov: &mut [f64],
    ldc: i32,
    ddof: usize,
) -> Result<(), KernelError> {
    if n_obs <= 0 || n_feat <= 0 {
        return Err(KernelError::InvalidArguments(
            "n_obs, n_feat must be positive".into(),
        ));
    }
    if ldx < n_obs {
        return Err(KernelError::InvalidArguments("ldx must be >= n_obs".into()));
    }
    if ldc < n_feat {
        return Err(KernelError::InvalidArguments(
            "ldc must be >= n_feat".into(),
        ));
    }
    if x.len() < (ldx * n_feat) as usize {
        return Err(KernelError::InvalidArguments("X buffer too small".into()));
    }
    if cov.len() < (ldc * n_feat) as usize {
        return Err(KernelError::InvalidArguments("cov buffer too small".into()));
    }
    let n_obs_u = n_obs as usize;
    let n_feat_u = n_feat as usize;
    let ldx_u = ldx as usize;
    let denom = n_obs_u
        .checked_sub(ddof)
        .ok_or_else(|| KernelError::InvalidArguments("ddof >= n_obs: division by zero".into()))?;
    if denom == 0 {
        return Err(KernelError::InvalidArguments(
            "n_obs - ddof is zero: division by zero".into(),
        ));
    }

    // Step 1: compute column means with Neumaier compensated summation
    let mut means = vec![0.0_f64; n_feat_u];
    for j in 0..n_feat_u {
        let col_offset = j * ldx_u;
        let mut s = 0.0_f64;
        let mut comp = 0.0_f64;
        for i in 0..n_obs_u {
            neumaier_add(&mut s, &mut comp, x[col_offset + i]);
        }
        means[j] = (s + comp) / n_obs_u as f64;
    }

    // Step 2: create mean-centred copy X_c (contiguous, lda = n_obs)
    let mut xc = vec![0.0_f64; n_obs_u * n_feat_u];
    for j in 0..n_feat_u {
        let src_offset = j * ldx_u;
        let dst_offset = j * n_obs_u;
        let m = means[j];
        for i in 0..n_obs_u {
            xc[dst_offset + i] = x[src_offset + i] - m;
        }
    }

    // Step 3: C = X_c^T * X_c via blocked_gemm
    // trans_a=true, trans_b=false: (n_feat x n_obs) * (n_obs x n_feat) = n_feat x n_feat
    // m=n_feat, n=n_feat, k=n_obs
    blas_lapack::blocked_gemm(
        n_feat, // m
        n_feat, // n
        n_obs,  // k
        1.0,    // alpha
        &xc,    // A (used as A^T)
        n_obs,  // lda (rows of the untransposed A = n_obs)
        &xc,    // B
        n_obs,  // ldb
        0.0,    // beta
        cov,    // C
        ldc,    // ldc
        true,   // trans_a
        false,  // trans_b
    )
    .map_err(|e| KernelError::InvalidArguments(e.to_string()))?;

    // Step 4: scale C /= (n_obs - ddof)
    let ldc_u = ldc as usize;
    let scale = 1.0 / denom as f64;
    for j in 0..n_feat_u {
        for i in 0..n_feat_u {
            cov[i + j * ldc_u] *= scale;
        }
    }

    // Step 5: symmetrise -- copy lower to upper
    for col in 0..n_feat_u {
        for row in 0..col {
            cov[row + col * ldc_u] = cov[col + row * ldc_u];
        }
    }

    Ok(())
}

/// Log-determinant of a symmetric positive-definite matrix: log|A|.
///
/// Uses Cholesky factorisation for numerical stability:
///   A = L * L^T  =>  log|A| = 2 * sum(ln(L[i,i]))
///
/// `a` is modified in-place (overwritten with Cholesky factor in lower triangle).
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `a`:    SPD matrix buffer, length >= lda * n. Overwritten with Cholesky factor.
/// - `lda`:  Leading dimension of A (>= n).
pub fn log_det_spd(n: i32, a: &mut [f64], lda: i32) -> Result<f64, KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if lda < n {
        return Err(KernelError::InvalidArguments("lda must be >= n".into()));
    }
    if a.len() < (lda * n) as usize {
        return Err(KernelError::InvalidArguments("A buffer too small".into()));
    }

    // Factor A = L * L^T
    cholesky(n, a, lda)?;

    // log|A| = 2 * sum(ln(L[i,i]))
    let nu = n as usize;
    let ldau = lda as usize;
    let mut log_det = 0.0_f64;
    let mut comp = 0.0_f64;
    for i in 0..nu {
        let diag = a[i + i * ldau];
        if diag <= 0.0 {
            return Err(KernelError::InvalidArguments(
                "Cholesky diagonal is non-positive: matrix is not SPD".into(),
            ));
        }
        neumaier_add(&mut log_det, &mut comp, diag.ln());
    }
    Ok(2.0 * (log_det + comp))
}

/// Mahalanobis distance of each observation from a mean vector.
///
/// For each observation row i, computes:
///   d_i = sqrt((x_i - mean)^T * Sigma_inv * (x_i - mean))
///
/// Uses forward substitution on the Cholesky factor for numerical stability
/// rather than forming Sigma_inv explicitly. For each row:
///   1. Form diff = x_i - mean
///   2. Solve L * z = diff via forward substitution
///   3. distance = ||z||_2
///
/// The caller provides the Cholesky factor L of Sigma, not Sigma itself.
///
/// Arguments
/// - `n_obs`:      Number of observations (rows of X).
/// - `n_feat`:     Number of features (columns of X).
/// - `x`:          Observation matrix, column-major, length >= ldx * n_feat.
/// - `ldx`:        Leading dimension of X (>= n_obs).
/// - `mean`:       Mean vector, length >= n_feat.
/// - `sigma_chol`: Cholesky factor L of Sigma, column-major, length >= lds * n_feat.
/// - `lds`:        Leading dimension of sigma_chol (>= n_feat).
/// - `distances`:  Output buffer for distances, length >= n_obs.
pub fn mahalanobis(
    n_obs: i32,
    n_feat: i32,
    x: &[f64],
    ldx: i32,
    mean: &[f64],
    sigma_chol: &[f64],
    lds: i32,
    distances: &mut [f64],
) -> Result<(), KernelError> {
    if n_obs <= 0 || n_feat <= 0 {
        return Err(KernelError::InvalidArguments(
            "n_obs, n_feat must be positive".into(),
        ));
    }
    if ldx < n_obs {
        return Err(KernelError::InvalidArguments("ldx must be >= n_obs".into()));
    }
    if lds < n_feat {
        return Err(KernelError::InvalidArguments(
            "lds must be >= n_feat".into(),
        ));
    }
    if x.len() < (ldx * n_feat) as usize {
        return Err(KernelError::InvalidArguments("X buffer too small".into()));
    }
    if mean.len() < n_feat as usize {
        return Err(KernelError::InvalidArguments(
            "mean buffer too small".into(),
        ));
    }
    if sigma_chol.len() < (lds * n_feat) as usize {
        return Err(KernelError::InvalidArguments(
            "sigma_chol buffer too small".into(),
        ));
    }
    if distances.len() < n_obs as usize {
        return Err(KernelError::InvalidArguments(
            "distances buffer too small".into(),
        ));
    }

    let n_obs_u = n_obs as usize;
    let n_feat_u = n_feat as usize;
    let ldx_u = ldx as usize;

    // For each observation row, form diff then solve L*z = diff
    let mut diff = vec![0.0_f64; n_feat_u];
    for i in 0..n_obs_u {
        // Extract row i from column-major X and subtract mean
        for j in 0..n_feat_u {
            diff[j] = x[i + j * ldx_u] - mean[j];
        }

        // Solve L * z = diff via trisolve_lower (nrhs=1)
        // trisolve_lower expects column-major B with ldb; for a single column ldb = n_feat
        blas_lapack::trisolve_lower(n_feat, 1, sigma_chol, lds, &mut diff, n_feat)
            .map_err(|e| KernelError::InvalidArguments(e.to_string()))?;

        // distance = ||z||_2
        let mut sum_sq = 0.0_f64;
        let mut comp = 0.0_f64;
        for &v in &diff {
            neumaier_add(&mut sum_sq, &mut comp, v * v);
        }
        distances[i] = (sum_sq + comp).sqrt();
    }

    Ok(())
}

/// Triangular matrix inverse in-place.
///
/// Computes T_inv from a triangular matrix T, overwriting T in-place.
/// Delegates to LAPACK dtrtri via `blas_lapack::tri_inverse`.
///
/// Arguments
/// - `n`:     Order of the matrix (n-by-n).
/// - `t`:     Triangular matrix buffer, length >= ldt * n. Overwritten with inverse.
/// - `ldt`:   Leading dimension of T (>= n).
/// - `upper`: If true, T is upper-triangular. If false, T is lower-triangular.
pub fn triangular_inverse(n: i32, t: &mut [f64], ldt: i32, upper: bool) -> Result<(), KernelError> {
    if n <= 0 {
        return Err(KernelError::InvalidArguments("n must be positive".into()));
    }
    if ldt < n {
        return Err(KernelError::InvalidArguments("ldt must be >= n".into()));
    }
    if t.len() < (ldt * n) as usize {
        return Err(KernelError::InvalidArguments("T buffer too small".into()));
    }

    blas_lapack::tri_inverse(n, t, ldt, upper)
        .map_err(|e| KernelError::InvalidArguments(e.to_string()))
}

// **************************************************
// Tests
// **************************************************

#[cfg(test)]
#[cfg(feature = "linear_algebra")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Helper: C = A * B via matmul (no transpose, alpha=1, beta=0).
    /// A is m x k, B is k x n, result is m x n. All column-major.
    fn mm(m: i32, n: i32, k: i32, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; (m * n) as usize];
        matmul(m, n, k, 1.0, a, m, b, k, 0.0, &mut c, m, false, false).unwrap();
        c
    }

    /// Helper: C = A^T * B via matmul.
    /// A is k x m (physical), so op(A) = A^T is m x k. B is k x n. Result is m x n.
    fn mm_at_b(m: i32, n: i32, k: i32, a: &[f64], lda: i32, b: &[f64], ldb: i32) -> Vec<f64> {
        let mut c = vec![0.0; (m * n) as usize];
        matmul(m, n, k, 1.0, a, lda, b, ldb, 0.0, &mut c, m, true, false).unwrap();
        c
    }

    // Decompositions

    // ---- LU ----

    #[test]
    fn test_lu_3x3_reconstruct() {
        // A = [[2, 1, 1],
        //      [4, 3, 3],
        //      [8, 7, 9]]
        // Column-major
        let mut a = vec![2.0, 4.0, 8.0, 1.0, 3.0, 7.0, 1.0, 3.0, 9.0];
        let a_orig = a.clone();
        let mut ipiv = vec![0i32; 3];

        _lu(3, 3, &mut a, 3, &mut ipiv).unwrap();

        // Extract L (unit lower) and U (upper) from the packed result
        let mut l = vec![0.0_f64; 9];
        let mut u = vec![0.0_f64; 9];
        for col in 0..3_usize {
            for row in 0..3_usize {
                let idx = row + col * 3;
                if row > col {
                    l[idx] = a[idx]; // strictly lower
                } else if row == col {
                    l[idx] = 1.0; // unit diagonal
                    u[idx] = a[idx];
                } else {
                    u[idx] = a[idx]; // upper
                }
            }
        }

        // Compute L * U
        let lu = mm(3, 3, 3, &l, &u);

        // Apply pivots to reconstruct P * A, then compare with L * U
        // LAPACK ipiv is 1-based: row i was swapped with row ipiv[i]
        let mut pa = a_orig.clone();
        // Apply pivots in forward order (same order LAPACK applies them)
        for i in 0..3_usize {
            let swap_row = (ipiv[i] - 1) as usize; // convert to 0-based
            if swap_row != i {
                // Swap rows i and swap_row across all columns
                for col in 0..3_usize {
                    pa.swap(i + col * 3, swap_row + col * 3);
                }
            }
        }

        for idx in 0..9 {
            assert!(
                approx_eq(lu[idx], pa[idx], TOL),
                "LU[{idx}] = {}, PA[{idx}] = {}, diff = {}",
                lu[idx],
                pa[idx],
                (lu[idx] - pa[idx]).abs()
            );
        }
    }

    #[test]
    fn test_lu_singular_matrix() {
        // Singular matrix: row 2 = row 0 + row 1
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [5, 7, 9]]
        let mut a = vec![1.0, 4.0, 5.0, 2.0, 5.0, 7.0, 3.0, 6.0, 9.0];
        let mut ipiv = vec![0i32; 3];

        // LU factorisation itself should succeed (info >= 0) but the U factor
        // will have a zero diagonal, which downstream solvers would detect.
        // LAPACK returns info > 0 for singular matrices, but our wrapper maps
        // info > 0 to an error.
        let result = _lu(3, 3, &mut a, 3, &mut ipiv);
        // Either it succeeds with a zero on U diagonal, or returns an error.
        // Both are acceptable for singular matrices.
        if result.is_ok() {
            // Check that a diagonal element is zero/near-zero (singular U)
            let has_zero_diag = (0..3).any(|i| a[i + i * 3].abs() < 1e-12);
            assert!(
                has_zero_diag,
                "Singular matrix should have zero diagonal in U"
            );
        }
        // If Err, the LAPACK detected singularity -- also correct
    }

    // ---- QR ----

    #[test]
    fn test_qr_4x3_orthogonal_and_reconstruct() {
        // A = [[1, 2, 3],     column-major: rows then columns
        //      [4, 5, 6],
        //      [7, 8, 9],
        //      [10, 11, 12]]
        // 4x3 matrix
        let a_orig = vec![
            1.0, 4.0, 7.0, 10.0, // col 0
            2.0, 5.0, 8.0, 11.0, // col 1
            3.0, 6.0, 9.0, 12.0, // col 2
        ];

        // Step 1: QR factorisation
        let mut a_qr = a_orig.clone();
        let mut taus = vec![0.0_f64; 3]; // min(4,3) = 3
        qr(4, 3, &mut a_qr, 4, &mut taus).unwrap();

        // Extract R from upper triangle of a_qr (3x3 upper part)
        let mut r = vec![0.0_f64; 3 * 3]; // 3x3 column-major
        for col in 0..3_usize {
            for row in 0..=col.min(2) {
                r[row + col * 3] = a_qr[row + col * 4]; // lda=4 in source
            }
        }

        // Step 2: Form Q explicitly (4x3)
        let mut q = a_qr.clone(); // reuse Householder vectors
        qr_q(4, 3, 3, &mut q, 4, &taus).unwrap();

        // Verify Q^T * Q = I_3 (orthonormal columns)
        let qtq = mm_at_b(3, 3, 4, &q, 4, &q, 4);
        for col in 0..3_usize {
            for row in 0..3_usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                let idx = row + col * 3;
                assert!(
                    approx_eq(qtq[idx], expected, TOL),
                    "Q^T*Q[{row},{col}] = {}, expected {expected}",
                    qtq[idx]
                );
            }
        }

        // Verify Q * R = A (reconstruction)
        // Q is 4x3 (lda=4), R is 3x3 (lda=3), result should be 4x3
        let qr_product = mm(4, 3, 3, &q, &r);
        for idx in 0..12 {
            assert!(
                approx_eq(qr_product[idx], a_orig[idx], TOL),
                "QR[{idx}] = {}, A[{idx}] = {}, diff = {}",
                qr_product[idx],
                a_orig[idx],
                (qr_product[idx] - a_orig[idx]).abs()
            );
        }
    }

    // ---- Cholesky ----

    #[test]
    fn test_cholesky_3x3_reconstruct() {
        // A = [[4, 2, 1],
        //      [2, 5, 3],
        //      [1, 3, 6]]
        // Column-major
        let mut a = vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let a_orig = a.clone();

        cholesky(3, &mut a, 3).unwrap();

        // Extract L from lower triangle
        let mut l = vec![0.0_f64; 9];
        for col in 0..3_usize {
            for row in col..3_usize {
                l[row + col * 3] = a[row + col * 3];
            }
        }

        // Verify L * L^T = A
        // L^T is upper triangular. L * L^T via matmul with trans_b=true.
        let mut llt = vec![0.0_f64; 9];
        matmul(3, 3, 3, 1.0, &l, 3, &l, 3, 0.0, &mut llt, 3, false, true).unwrap();

        for idx in 0..9 {
            assert!(
                approx_eq(llt[idx], a_orig[idx], TOL),
                "L*Lt[{idx}] = {}, A[{idx}] = {}, diff = {}",
                llt[idx],
                a_orig[idx],
                (llt[idx] - a_orig[idx]).abs()
            );
        }
    }

    #[test]
    fn test_cholesky_non_spd_errors() {
        // Not SPD: has negative eigenvalue
        // [[1, 2], [2, 1]] has eigenvalues -1 and 3
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        let result = cholesky(2, &mut a, 2);
        assert!(result.is_err(), "Non-SPD matrix should fail Cholesky");
    }

    // ---- SVD ----

    #[test]
    fn test_svd_3x2_economy() {
        // A = [[1, 2],    column-major: [1, 3, 5, 2, 4, 6]
        //      [3, 4],
        //      [5, 6]]
        // 3x2 matrix, economy SVD produces U(3x2), S(2), Vt(2x2)
        let mut a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut s = vec![0.0_f64; 2];
        let mut u = vec![0.0_f64; 3 * 2]; // ldu=3, cols=2
        let mut vt = vec![0.0_f64; 2 * 2]; // ldvt=2, cols=2

        svd(3, 2, &mut a, 3, &mut s, &mut u, 3, &mut vt, 2, true).unwrap();

        // Verify singular values are non-negative and descending
        assert!(s[0] >= 0.0, "s[0] should be non-negative");
        assert!(s[1] >= 0.0, "s[1] should be non-negative");
        assert!(s[0] >= s[1], "singular values should be descending");

        // Verify U * diag(S) * Vt = A
        // U is 3x2, S is [s0, s1], Vt is 2x2
        // First multiply U * diag(S): scale column j of U by s[j]
        let mut us = vec![0.0_f64; 3 * 2];
        for col in 0..2_usize {
            for row in 0..3_usize {
                us[row + col * 3] = u[row + col * 3] * s[col];
            }
        }
        // Then US * Vt (3x2 * 2x2 = 3x2)
        let a_reconstructed = mm(3, 2, 2, &us, &vt);

        let a_orig = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        for idx in 0..6 {
            assert!(
                approx_eq(a_reconstructed[idx], a_orig[idx], TOL),
                "USVt[{idx}] = {}, A[{idx}] = {}, diff = {}",
                a_reconstructed[idx],
                a_orig[idx],
                (a_reconstructed[idx] - a_orig[idx]).abs()
            );
        }
    }

    // ---- Eigendecomposition ----

    #[test]
    fn test_eig_symmetric_3x3() {
        // A = [[2, 1, 0],
        //      [1, 3, 1],
        //      [0, 1, 2]]
        // Symmetric. trace = 7, so eigenvalues should sum to 7.
        let mut a = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut eigenvalues = vec![0.0_f64; 3];

        eig_symmetric(3, &mut a, 3, &mut eigenvalues).unwrap();

        // Eigenvalues sum = trace
        let eig_sum: f64 = eigenvalues.iter().sum();
        assert!(
            approx_eq(eig_sum, 7.0, TOL),
            "Eigenvalue sum = {eig_sum}, expected 7.0"
        );

        // Eigenvalues are in ascending order (LAPACK convention)
        for i in 1..3 {
            assert!(
                eigenvalues[i] >= eigenvalues[i - 1] - TOL,
                "Eigenvalues should be ascending: {} >= {}",
                eigenvalues[i],
                eigenvalues[i - 1]
            );
        }

        // Eigenvectors (stored in a) should be orthogonal: V^T * V = I
        let vtv = mm_at_b(3, 3, 3, &a, 3, &a, 3);
        for col in 0..3_usize {
            for row in 0..3_usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                let idx = row + col * 3;
                assert!(
                    approx_eq(vtv[idx], expected, TOL),
                    "V^T*V[{row},{col}] = {}, expected {expected}",
                    vtv[idx]
                );
            }
        }
    }

    // Solvers

    #[test]
    fn test_solve_3x3() {
        // A = [[1, 2, 3],
        //      [4, 5, 6],
        //      [7, 8, 10]]
        // x = [1, 2, 3]
        // b = A*x = [1+4+9, 4+10+18, 7+16+30] = [14, 32, 53]
        let mut a = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0];
        let mut b = vec![14.0, 32.0, 53.0];
        let mut ipiv = vec![0i32; 3];

        solve(3, 1, &mut a, 3, &mut ipiv, &mut b, 3).unwrap();

        assert!(approx_eq(b[0], 1.0, TOL), "x[0] = {}, expected 1.0", b[0]);
        assert!(approx_eq(b[1], 2.0, TOL), "x[1] = {}, expected 2.0", b[1]);
        assert!(approx_eq(b[2], 3.0, TOL), "x[2] = {}, expected 3.0", b[2]);
    }

    #[test]
    fn test_spd_solve_2x2() {
        // Sigma = [[4, 1], [1, 2]] (SPD)
        // Cholesky first: L from Sigma
        // x = [1, 1]
        // b = Sigma * x = [5, 3]
        let mut sigma = vec![4.0, 1.0, 1.0, 2.0];
        cholesky(2, &mut sigma, 2).unwrap();

        let mut b = vec![5.0, 3.0];
        spd_solve(2, 1, &sigma, 2, &mut b, 2).unwrap();

        assert!(approx_eq(b[0], 1.0, TOL), "x[0] = {}, expected 1.0", b[0]);
        assert!(approx_eq(b[1], 1.0, TOL), "x[1] = {}, expected 1.0", b[1]);
    }

    #[test]
    fn test_solve_triangular_upper_3x3() {
        // U = [[2, 1, 3],   column-major: [2, 0, 0, 1, 4, 0, 3, 2, 5]
        //      [0, 4, 2],
        //      [0, 0, 5]]
        // x = [1, 2, 3]
        // b = U*x = [2+2+9, 0+8+6, 0+0+15] = [13, 14, 15]
        let u = vec![2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 2.0, 5.0];
        let mut b = vec![13.0, 14.0, 15.0];

        solve_triangular_upper(3, 1, &u, 3, &mut b, 3).unwrap();

        assert!(approx_eq(b[0], 1.0, TOL), "x[0] = {}, expected 1.0", b[0]);
        assert!(approx_eq(b[1], 2.0, TOL), "x[1] = {}, expected 2.0", b[1]);
        assert!(approx_eq(b[2], 3.0, TOL), "x[2] = {}, expected 3.0", b[2]);
    }

    #[test]
    fn test_solve_triangular_lower_3x3() {
        // L = [[3, 0, 0],   column-major: [3, 1, 2, 0, 5, 4, 0, 0, 6]
        //      [1, 5, 0],
        //      [2, 4, 6]]
        // x = [1, 2, 3]
        // b = L*x = [3, 1+10, 2+8+18] = [3, 11, 28]
        let l = vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 0.0, 0.0, 6.0];
        let mut b = vec![3.0, 11.0, 28.0];

        solve_triangular_lower(3, 1, &l, 3, &mut b, 3).unwrap();

        assert!(approx_eq(b[0], 1.0, TOL), "x[0] = {}, expected 1.0", b[0]);
        assert!(approx_eq(b[1], 2.0, TOL), "x[1] = {}, expected 2.0", b[1]);
        assert!(approx_eq(b[2], 3.0, TOL), "x[2] = {}, expected 3.0", b[2]);
    }

    #[test]
    fn test_least_squares_overdetermined() {
        // A = [[1, 1],    column-major: [1, 1, 1, 1, 1, 2, 3, 4]
        //      [1, 2],
        //      [1, 3],
        //      [1, 4]]
        // b = [1, 2, 4, 4]
        //
        // Normal equations: A^T*A = [[4,10],[10,30]], A^T*b = [11, 33]
        // inv(A^T*A) = [[1.5,-0.5],[-0.5,0.2]]
        // x = inv(A^T*A) * A^T*b = [0.0, 1.1]
        let mut a = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        // ldb must be >= max(m, n) = max(4, 2) = 4
        let mut b = vec![1.0, 2.0, 4.0, 4.0];

        least_squares(4, 2, 1, &mut a, 4, &mut b, 4).unwrap();

        // Solution in first n=2 entries of b
        assert!(
            approx_eq(b[0], 0.0, TOL),
            "intercept = {}, expected 0.0",
            b[0]
        );
        assert!(approx_eq(b[1], 1.1, TOL), "slope = {}, expected 1.1", b[1]);
    }

    // Diagnostics

    #[test]
    fn test_inverse_3x3() {
        // A = [[1, 2, 3],
        //      [0, 4, 5],
        //      [1, 0, 6]]
        // det(A) = 22
        let mut a = vec![1.0, 0.0, 1.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0];
        let a_orig = a.clone();

        inverse(3, &mut a, 3).unwrap();

        // Verify A_orig * A_inv = I
        let product = mm(3, 3, 3, &a_orig, &a);
        for col in 0..3_usize {
            for row in 0..3_usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                let idx = row + col * 3;
                assert!(
                    approx_eq(product[idx], expected, TOL),
                    "A*Ainv[{row},{col}] = {}, expected {expected}",
                    product[idx]
                );
            }
        }
    }

    #[test]
    fn test_spd_inverse_2x2() {
        // A = [[4, 1], [1, 2]] (SPD)
        let mut a = vec![4.0, 1.0, 1.0, 2.0];
        let a_orig = a.clone();

        spd_inverse(2, &mut a, 2).unwrap();

        // Verify A_orig * A_inv = I
        let product = mm(2, 2, 2, &a_orig, &a);
        for col in 0..2_usize {
            for row in 0..2_usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                let idx = row + col * 2;
                assert!(
                    approx_eq(product[idx], expected, TOL),
                    "A*Ainv[{row},{col}] = {}, expected {expected}",
                    product[idx]
                );
            }
        }
    }

    #[test]
    fn test_determinant_3x3() {
        // A = [[1, 2, 3],
        //      [0, 4, 5],
        //      [1, 0, 6]]
        // det = 1*(4*6-5*0) - 2*(0*6-5*1) + 3*(0*0-4*1) = 24 - (-10) + (-12) = 22
        let mut a = vec![1.0, 0.0, 1.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0];

        let det = determinant(3, &mut a, 3).unwrap();
        assert!(approx_eq(det, 22.0, TOL), "det = {det}, expected 22.0");
    }

    #[test]
    fn test_rank_full() {
        // Full rank 3x3 identity
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = rank(3, 3, &mut a, 3, 1e-12).unwrap();
        assert_eq!(r, 3, "Identity should have rank 3");
    }

    #[test]
    fn test_rank_deficient() {
        // Rank 2: row 2 = row 0 + row 1
        // [[1, 0, 1],
        //  [0, 1, 1],
        //  [1, 1, 2]]
        let mut a = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0];
        let r = rank(3, 3, &mut a, 3, 1e-12).unwrap();
        assert_eq!(r, 2, "Rank-deficient matrix should have rank 2");
    }

    #[test]
    fn test_condition_number_identity() {
        // Identity matrix: condition number = 1.0
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let kappa = condition_number(3, 3, &mut a, 3).unwrap();
        assert!(
            approx_eq(kappa, 1.0, TOL),
            "Identity condition number = {kappa}, expected 1.0"
        );
    }

    #[test]
    fn test_condition_number_ill_conditioned() {
        // Near-singular Hilbert-like 2x2: [[1, 0.5], [0.5, 0.3333]]
        // Condition number should be large
        let mut a = vec![1.0, 0.5, 0.5, 1.0 / 3.0];
        let kappa = condition_number(2, 2, &mut a, 2).unwrap();
        assert!(
            kappa > 10.0,
            "Ill-conditioned matrix should have kappa > 10, got {kappa}"
        );
    }

    #[test]
    fn test_norm_frobenius() {
        // A = [[1, 2],
        //      [3, 4]]
        // Column-major: [1, 3, 2, 4]
        // Frobenius = sqrt(1+9+4+16) = sqrt(30)
        let a = vec![1.0, 3.0, 2.0, 4.0];
        let n = norm(2, 2, &a, 2, "fro").unwrap();
        assert!(
            approx_eq(n, 30.0_f64.sqrt(), TOL),
            "Frobenius norm = {n}, expected {}",
            30.0_f64.sqrt()
        );
    }

    #[test]
    fn test_norm_l1() {
        // A = [[1, -2],
        //      [3, 4]]
        // Column-major: [1, 3, -2, 4]
        // L1 = max col sum of abs: max(|1|+|3|, |-2|+|4|) = max(4, 6) = 6
        let a = vec![1.0, 3.0, -2.0, 4.0];
        let n = norm(2, 2, &a, 2, "l1").unwrap();
        assert!(approx_eq(n, 6.0, TOL), "L1 norm = {n}, expected 6.0");
    }

    #[test]
    fn test_norm_linf() {
        // A = [[1, -2],
        //      [3, 4]]
        // Column-major: [1, 3, -2, 4]
        // Linf = max row sum of abs: max(|1|+|-2|, |3|+|4|) = max(3, 7) = 7
        let a = vec![1.0, 3.0, -2.0, 4.0];
        let n = norm(2, 2, &a, 2, "linf").unwrap();
        assert!(approx_eq(n, 7.0, TOL), "Linf norm = {n}, expected 7.0");
    }

    // Tier 2: Covariance, log_det_spd, Mahalanobis, triangular_inverse

    #[test]
    fn test_covariance_perfectly_correlated_ddof1() {
        // 4 observations x 3 features, perfectly correlated:
        // X = [[1,  2,  3],
        //      [4,  5,  6],
        //      [7,  8,  9],
        //      [10, 11, 12]]
        // Column-major (4 rows per column):
        let x = vec![
            1.0, 4.0, 7.0, 10.0, // feature 0
            2.0, 5.0, 8.0, 11.0, // feature 1
            3.0, 6.0, 9.0, 12.0, // feature 2
        ];

        // Column means: [5.5, 6.5, 7.5]
        // Each feature has the same centred values: [-4.5, -1.5, 1.5, 4.5]
        // var = (20.25+2.25+2.25+20.25)/3 = 45/3 = 15.0
        // cov(i,j) = 15.0 for all pairs (perfectly correlated)
        let mut cov = vec![0.0_f64; 9]; // 3x3
        covariance(4, 3, &x, 4, &mut cov, 3, 1).unwrap();

        for col in 0..3_usize {
            for row in 0..3_usize {
                let idx = row + col * 3;
                assert!(
                    approx_eq(cov[idx], 15.0, TOL),
                    "cov[{row},{col}] = {}, expected 15.0",
                    cov[idx]
                );
            }
        }

        // Verify symmetry
        for col in 0..3_usize {
            for row in 0..col {
                assert!(
                    cov[row + col * 3] == cov[col + row * 3],
                    "cov should be symmetric: cov[{row},{col}] != cov[{col},{row}]"
                );
            }
        }
    }

    #[test]
    fn test_covariance_ddof0() {
        // Same data, ddof=0 (population covariance)
        // var = 45/4 = 11.25
        let x = vec![
            1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0,
        ];

        let mut cov = vec![0.0_f64; 9];
        covariance(4, 3, &x, 4, &mut cov, 3, 0).unwrap();

        for col in 0..3_usize {
            for row in 0..3_usize {
                let idx = row + col * 3;
                assert!(
                    approx_eq(cov[idx], 11.25, TOL),
                    "cov_pop[{row},{col}] = {}, expected 11.25",
                    cov[idx]
                );
            }
        }
    }

    #[test]
    fn test_covariance_uncorrelated() {
        // 4 obs x 2 features, uncorrelated:
        // f0 = [1, -1, 1, -1], f1 = [1, 1, -1, -1]
        // means: [0, 0]
        // cov(f0,f0) = (1+1+1+1)/3 = 4/3
        // cov(f0,f1) = (1*1 + (-1)*1 + 1*(-1) + (-1)*(-1))/3 = 0/3 = 0
        // cov(f1,f1) = 4/3
        let x = vec![
            1.0, -1.0, 1.0, -1.0, // feature 0
            1.0, 1.0, -1.0, -1.0, // feature 1
        ];

        let mut cov = vec![0.0_f64; 4];
        covariance(4, 2, &x, 4, &mut cov, 2, 1).unwrap();

        let expected_var = 4.0 / 3.0;
        assert!(
            approx_eq(cov[0], expected_var, TOL),
            "cov[0,0] = {}, expected {}",
            cov[0],
            expected_var
        );
        assert!(
            approx_eq(cov[1], 0.0, TOL),
            "cov[1,0] = {}, expected 0.0",
            cov[1]
        );
        assert!(
            approx_eq(cov[2], 0.0, TOL),
            "cov[0,1] = {}, expected 0.0",
            cov[2]
        );
        assert!(
            approx_eq(cov[3], expected_var, TOL),
            "cov[1,1] = {}, expected {}",
            cov[3],
            expected_var
        );
    }

    #[test]
    fn test_log_det_spd_2x2() {
        // A = [[4, 2], [2, 3]] (SPD, eigenvalues 1 and 6... let's verify: det = 12-4 = 8)
        // log|A| = ln(8) = 2.0794415...
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let ld = log_det_spd(2, &mut a, 2).unwrap();
        assert!(
            approx_eq(ld, 8.0_f64.ln(), TOL),
            "log_det = {ld}, expected {}",
            8.0_f64.ln()
        );
    }

    #[test]
    fn test_log_det_spd_3x3() {
        // A = [[4, 2, 1],
        //      [2, 5, 3],
        //      [1, 3, 6]]
        // det = 4*(30-9) - 2*(12-3) + 1*(6-5) = 84 - 18 + 1 = 67
        // log|A| = ln(67) = 4.2047...
        let mut a = vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let ld = log_det_spd(3, &mut a, 3).unwrap();
        assert!(
            approx_eq(ld, 67.0_f64.ln(), TOL),
            "log_det = {ld}, expected {}",
            67.0_f64.ln()
        );
    }

    #[test]
    fn test_mahalanobis_identity_covariance() {
        // Sigma = I_2, L = I_2 (Cholesky of identity)
        // mean = [0, 0]
        // point = [3, 4]
        // Mahalanobis distance = Euclidean distance = sqrt(9+16) = 5.0
        let x = vec![3.0, 4.0]; // 1 obs x 2 features, column-major: [3, 4] for col0=[3], col1=[4]
        // Wait -- 1 obs x 2 feat column-major: col0 = [3.0], col1 = [4.0]
        // So x = [3.0, 4.0] with ldx=1
        let mean = vec![0.0, 0.0];
        let sigma_chol = vec![1.0, 0.0, 0.0, 1.0]; // I_2, lds=2
        let mut distances = vec![0.0_f64; 1];

        mahalanobis(1, 2, &x, 1, &mean, &sigma_chol, 2, &mut distances).unwrap();

        assert!(
            approx_eq(distances[0], 5.0, TOL),
            "distance = {}, expected 5.0",
            distances[0]
        );
    }

    #[test]
    fn test_mahalanobis_non_identity() {
        // Sigma = [[4, 0], [0, 1]]
        // L = chol(Sigma) = [[2, 0], [0, 1]]
        // mean = [0, 0], point = [2, 1]
        // d = sqrt((x-mu)^T Sigma_inv (x-mu))
        // Sigma_inv = [[0.25, 0], [0, 1]]
        // (x-mu)^T Sigma_inv (x-mu) = 2^2*0.25 + 1^2*1 = 1 + 1 = 2
        // d = sqrt(2)
        let x = vec![2.0, 1.0]; // 1 obs x 2 feat, ldx=1
        let mean = vec![0.0, 0.0];
        let sigma_chol = vec![2.0, 0.0, 0.0, 1.0]; // L = [[2,0],[0,1]], lds=2
        let mut distances = vec![0.0_f64; 1];

        mahalanobis(1, 2, &x, 1, &mean, &sigma_chol, 2, &mut distances).unwrap();

        assert!(
            approx_eq(distances[0], 2.0_f64.sqrt(), TOL),
            "distance = {}, expected {}",
            distances[0],
            2.0_f64.sqrt()
        );
    }

    #[test]
    fn test_mahalanobis_multiple_observations() {
        // Sigma = I_2, mean = [1, 1]
        // 3 observations:
        //   [1, 1] -> distance = 0.0
        //   [4, 1] -> distance = 3.0
        //   [1, 5] -> distance = 4.0
        // Column-major 3 obs x 2 feat:
        //   col0 = [1, 4, 1], col1 = [1, 1, 5]
        let x = vec![1.0, 4.0, 1.0, 1.0, 1.0, 5.0]; // ldx=3
        let mean = vec![1.0, 1.0];
        let sigma_chol = vec![1.0, 0.0, 0.0, 1.0]; // I_2
        let mut distances = vec![0.0_f64; 3];

        mahalanobis(3, 2, &x, 3, &mean, &sigma_chol, 2, &mut distances).unwrap();

        assert!(
            approx_eq(distances[0], 0.0, TOL),
            "d[0] = {}, expected 0.0",
            distances[0]
        );
        assert!(
            approx_eq(distances[1], 3.0, TOL),
            "d[1] = {}, expected 3.0",
            distances[1]
        );
        assert!(
            approx_eq(distances[2], 4.0, TOL),
            "d[2] = {}, expected 4.0",
            distances[2]
        );
    }

    #[test]
    fn test_triangular_inverse_lower() {
        // L = [[2, 0, 0],
        //      [1, 3, 0],
        //      [4, 5, 6]]
        // Column-major: [2, 1, 4, 0, 3, 5, 0, 0, 6]
        let mut l = vec![2.0, 1.0, 4.0, 0.0, 3.0, 5.0, 0.0, 0.0, 6.0];
        let l_orig = l.clone();

        triangular_inverse(3, &mut l, 3, false).unwrap();

        // Verify L_orig * L_inv = I
        let product = mm(3, 3, 3, &l_orig, &l);
        for col in 0..3_usize {
            for row in 0..3_usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                let idx = row + col * 3;
                assert!(
                    approx_eq(product[idx], expected, TOL),
                    "L*Linv[{row},{col}] = {}, expected {expected}",
                    product[idx]
                );
            }
        }
    }

    #[test]
    fn test_triangular_inverse_upper() {
        // U = [[2, 1, 3],
        //      [0, 4, 2],
        //      [0, 0, 5]]
        // Column-major: [2, 0, 0, 1, 4, 0, 3, 2, 5]
        let mut u = vec![2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 2.0, 5.0];
        let u_orig = u.clone();

        triangular_inverse(3, &mut u, 3, true).unwrap();

        // Verify U_orig * U_inv = I
        let product = mm(3, 3, 3, &u_orig, &u);
        for col in 0..3_usize {
            for row in 0..3_usize {
                let expected = if row == col { 1.0 } else { 0.0 };
                let idx = row + col * 3;
                assert!(
                    approx_eq(product[idx], expected, TOL),
                    "U*Uinv[{row},{col}] = {}, expected {expected}",
                    product[idx]
                );
            }
        }
    }
}
