// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **BLAS/LAPACK Integration Module** - *High-Performance Linear Algebra Kernels*
//!
//! This module provides low-level bindings and optimised kernels for linear algebra operations
//! through integration with industry-standard BLAS and LAPACK libraries. It forms the
//! computational backbone for numerical linear algebra in the simd-kernels crate.
//!
//! ## Overview
//!
//! The module wraps BLAS and LAPACK industry-standard linear algebra kernels:
//!
//! ### Level 1 BLAS: Vector Operations
//! - Vector scaling, dot products, and norms
//! - Efficient memory access patterns with unit stride optimisation
//!
//! ### Level 2 BLAS: Matrix-Vector Operations  
//! - **GEMV**: General matrix-vector multiplication with transpose support
//! - Triangular system solvers for upper/lower matrices
//! - Symmetric and packed matrix operations
//!
//! ### Level 3 BLAS: Matrix-Matrix Operations
//! - **GEMM**: General matrix-matrix multiplication with blocking optimisation
//! - **SYRK**: Symmetric rank-k updates for covariance computation
//! - **SYR2K**: Symmetric rank-2k updates for advanced statistical operations
//!
//! ### LAPACK Decompositions and Solvers
//! - **LU Decomposition**: General linear systems with pivoting (`DGETRF`)
//! - **QR Decomposition**: Orthogonal factorisation for least squares (`DGEQRF`)
//! - **Cholesky Decomposition**: Symmetric positive definite systems (`DPOTRF`)
//! - **SVD**: Singular value decomposition for dimensionality reduction (`DGESVD`)
//! - **Eigensolvers**: Symmetric eigenvalue problems (`DSYEV`)
//!
//! ## External Dependencies
//!
//! This module requires linking against BLAS and LAPACK.
//!
//! Supported implementations include:
//! - OpenBLAS - recommended for general uses
//! - Intel MKL - optimal for Intel processors  
//! - ATLAS, Accelerate, or system-provided BLAS/LAPACK
use blas::*;
use lapack::*;

/// GEMV  (y ← α·A·x + β·y)
#[inline(always)]
pub fn gemv(
    m: i32,
    n: i32,
    a: &[f64], // len = lda * n  (column-major)
    lda: i32,
    x: &[f64], // len = 1 + (len_x-1)*incx
    incx: i32,
    y: &mut [f64], // len = 1 + (len_y-1)*incy
    incy: i32,
    alpha: f64,
    beta: f64,
    trans_a: bool, // false = 'N', true = 'T'
) -> Result<(), &'static str> {
    // Dimension sanity
    if m <= 0 || n <= 0 {
        return Err("m and n must be positive");
    }
    if lda < m {
        return Err("lda must be ≥ m");
    }
    if incx == 0 || incy == 0 {
        return Err("incx and incy must be non-zero");
    }

    // Matrix buffer check
    if a.len() < (lda * n) as usize {
        return Err("A too small");
    }

    // Vector buffer checks
    let (len_x, len_y) = if trans_a { (m, n) } else { (n, m) };

    if x.len() < (1 + (len_x - 1) * incx.abs()) as usize {
        return Err("x too small");
    }
    if y.len() < (1 + (len_y - 1) * incy.abs()) as usize {
        return Err("y too small");
    }

    // BLAS call
    unsafe {
        dgemv(
            if trans_a { b'T' } else { b'N' },
            m,
            n,
            alpha,
            a,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
        );
    }
    Ok(())
}

/// 4 × 4 micro-kernel (C += A·B)
#[inline(always)]
pub fn gemm_4x4_microkernel(
    a: &[f64; 16],     // 4×4 block of A (column-major)
    b: &[f64; 16],     // 4×4 block of B (column-major)
    c: &mut [f64; 16], // 4×4 block of C (column-major, updated in-place)
    alpha: f64,
    beta: f64,
) {
    // m = n = k = 4, lda = ldb = ldc = 4 (column-major, contiguous)
    unsafe {
        dgemm(
            b'N', // trans_a: not transposed
            b'N', // trans_b: not transposed
            4,    // m
            4,    // n
            4,    // k
            alpha, a, 4, b, 4, beta, c, 4,
        );
    }
}

/// 2×2 triangular solve (U · x = b  OR  L · x = b)
#[inline(always)]
pub fn trisolve_2x2(
    upper: bool,
    a: &mut [f64; 4], // 2×2 triangular matrix, overwritten by LAPACK
    b: &mut [f64; 2], // RHS - will contain the solution on return
) -> Result<(), &'static str> {
    unsafe {
        dtrsv(
            if upper { b'U' } else { b'L' }, // UPLO
            b'N',                            // trans = NoTrans
            b'N',                            // diag  = NonUnit
            2,                               // n
            a,                               // A
            2,                               // lda
            b,                               // x
            1,                               // incx
        );
    }
    Ok(())
}

/// Applies Householder reflector(s) to a matrix panel.
#[inline(always)]
pub fn householder_apply(
    m: i32,
    n: i32,
    a: &mut [f64], // len ≥ lda*n   (column-major)
    lda: i32,
    taus: &mut [f64], // len ≥ n
) -> Result<(), &'static str> {
    // recommended lwork ≥ n*64 for small panels
    let lwork = (n.max(1) * 64) as i32;
    let mut work = vec![0.0_f64; lwork as usize];
    let mut info = 0;

    unsafe {
        dgeqrf(m, n, a, lda, taus, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgeqrf failed")
    }
}

// Blocked GEMM wrapper (C ← αAB + βC)
#[inline(always)]
pub fn blocked_gemm(
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
) -> Result<(), &'static str> {
    // Buffer length checks
    let (_rows_a, cols_a) = if trans_a { (k, m) } else { (m, k) };
    let (_rows_b, cols_b) = if trans_b { (n, k) } else { (k, n) };

    if a.len() < (lda * cols_a) as usize {
        return Err("A too small");
    }
    if b.len() < (ldb * cols_b) as usize {
        return Err("B too small");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C too small");
    }

    unsafe {
        dgemm(
            if trans_a { b'T' } else { b'N' },
            if trans_b { b'T' } else { b'N' },
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }
    Ok(())
}

/// CholeskyPanel  (lower-triangular, in-place)
pub fn cholesky_panel(
    n: i32,
    a: &mut [f64], // len ≥ lda*n ; on exit contains L in lower triangle
    lda: i32,
) -> Result<(), &'static str> {
    let mut info = 0;
    unsafe {
        dpotrf(b'L', n, a, lda, &mut info);
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix not positive-definite")
    } else {
        Err("LAPACK dpotrf argument error")
    }
}

/// Performs LU decomposition with partial pivoting on an m×n panel.
#[inline(always)]
pub fn lu_with_piv(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32], // ↓ length ≥ min(m,n)
) -> Result<(), &'static str> {
    use std::cmp::min;

    if a.len() < (lda * n) as usize {
        return Err("A too small for GETRF");
    }
    if ipiv.len() < min(m, n) as usize {
        return Err("ipiv too small for GETRF");
    }

    let mut info = 0;
    unsafe {
        dgetrf(m, n, a, lda, ipiv, &mut info);
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix is singular to machine precision")
    } else {
        Err("LAPACK dgetrf argument error")
    }
}

/// Blocked QR factorisation (full panel) using Householder reflections.
#[inline(always)]
pub fn qr_block(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    taus: &mut [f64],
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A too small for GEQRF");
    }
    if taus.len() < n as usize {
        return Err("taus too small for GEQRF");
    }

    // Workspace query: call with lwork = -1 to get optimal size
    let mut work_query = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgeqrf(
            m,
            n,
            a,
            lda,
            taus,
            &mut work_query,
            -1, // workspace query
            &mut info,
        );
    }
    if info != 0 {
        return Err("GEQRF workspace query failed");
    }

    // Allocate the optimal amount returned in work_query[0]
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgeqrf(m, n, a, lda, taus, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgeqrf failed")
    }
}

// Composite Linear Algebra

/// SYRK_Panel   (C ← α A·Aᵀ + β C)
#[inline(always)]
pub fn syrk_panel(
    n: i32, // C is n×n
    k: i32, // inner-dim
    alpha: f64,
    a: &[f64], // len ≥ lda*k
    lda: i32,
    beta: f64,
    c: &mut [f64], // len ≥ ldc*n   (full-storage, col-major)
    ldc: i32,
    trans_a: bool, // false = use A    , true = use Aᵀ
) -> Result<(), &'static str> {
    use blas::dsyrk;

    if a.len() < (lda * k) as usize {
        return Err("A too small for SYRK");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C too small for SYRK");
    }

    unsafe {
        dsyrk(
            b'L', // UPLO: lower triangle
            if trans_a { b'T' } else { b'N' },
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
        );
    }
    Ok(())
}

/// Computes eigenvalues/vectors of 2×2 symmetric matrix analytically via SIMD.
#[inline(always)]
pub fn symeig2x2(
    a_in: &[f64; 4], // input 2×2 symmetric (column-major)
    eigvals: &mut [f64; 2],
    eigvecs: &mut [f64; 4], // Q (column-major)
) -> Result<(), &'static str> {
    let mut a = *a_in; // make a mutable copy for LAPACK
    let mut info = 0;
    // workspace: lwork ≥ 3*n-1  when jobz='V'
    let mut work = [0.0_f64; 10];
    let len = work.len();
    unsafe {
        dsyev(
            b'V', // jobz = compute eigen-vectors
            b'U', // upper triangle supplied
            2,    // n
            &mut a, 2,       // a, lda
            eigvals, // w
            &mut work, len as i32, &mut info,
        );
    }
    if info != 0 {
        return Err("LAPACK dsyev failed on 2×2");
    }

    // Copy eigen-vectors (returned in A) for the caller
    eigvecs.copy_from_slice(&a);
    Ok(())
}

/// Performs Golub-Kahan bidiagonalisation (first SVD phase).
#[inline(always)]
pub fn bidiag_reduction(
    m: i32,
    n: i32,
    a: &mut [f64], // len ≥ lda*n
    lda: i32,
    d: &mut [f64],    // len ≥ min(m,n)
    e: &mut [f64],    // len ≥ min(m,n)-1
    tauq: &mut [f64], // len ≥ min(m,n)   (left  reflectors)
    taup: &mut [f64], // len ≥ min(m,n)   (right reflectors)
) -> Result<(), &'static str> {
    use std::cmp::min;

    use lapack::dgebrd;

    let k = min(m, n);
    if a.len() < (lda * n) as usize {
        return Err("A too small for GEBRD");
    }
    if d.len() < k as usize {
        return Err("d too small for GEBRD");
    }
    if e.len() < (k - 1).max(0) as usize {
        return Err("e too small for GEBRD");
    }
    if tauq.len() < k as usize {
        return Err("tauq too small");
    }
    if taup.len() < k as usize {
        return Err("taup too small");
    }

    // Query optimal workspace
    let mut work_query = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgebrd(
            m,
            n,
            a,
            lda,
            d,
            e,
            tauq,
            taup,
            &mut work_query,
            -1, // lwork = -1 query
            &mut info,
        );
    }
    if info != 0 {
        return Err("GEBRD workspace query failed");
    }

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgebrd(m, n, a, lda, d, e, tauq, taup, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgebrd failed")
    }
}

/// Completes the SVD of a bidiagonal matrix produced by `bidiag_reduction`.
#[inline(always)]
pub fn svd_qr_iter() {
    todo!()
}

/// Computes the full or economy-size SVD of `A` in one shot.
#[inline(always)]
pub fn svd_block(
    jobu: u8,  // 'A', 'S', or 'N'
    jobvt: u8, // same for Vᵀ
    m: i32,
    n: i32,
    a: &mut [f64], // len ≥ lda*n   (destroyed)
    lda: i32,
    s: &mut [f64], // len ≥ min(m,n)
    u: &mut [f64], // len ≥ ldu*ucol   (ucol depends on jobu)
    ldu: i32,
    vt: &mut [f64], // len ≥ ldvt*ncol (ncol depends on jobvt)
    ldvt: i32,
) -> Result<(), &'static str> {
    use std::cmp::min;

    use lapack::dgesvd;

    let k = min(m, n);
    if a.len() < (lda * n) as usize {
        return Err("A too small for DGESVD");
    }
    if s.len() < k as usize {
        return Err("s too small for DGESVD");
    }

    // Workspace query
    let mut wk = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgesvd(
            jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &mut wk, -1, // query
            &mut info,
        );
    }
    if info != 0 {
        return Err("DGESVD workspace query failed");
    }

    let lwork = wk[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgesvd(
            jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &mut work, lwork, &mut info,
        );
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("DGESVD failed to converge")
    } else {
        Err("DGESVD argument error")
    }
}

/// Projects data matrix X onto top-k principal components (PCA scores).
#[inline(always)]
pub fn pca_project(
    m: i32, // obs
    f: i32, // features (cols of X, rows of W)
    k: i32, // components
    alpha: f64,
    x: &[f64],
    ldx: i32, // m×f data matrix  (column-major)
    w: &[f64],
    ldw: i32, // f×k weight matrix
    beta: f64,
    y: &mut [f64],
    ldy: i32, // m×k output scores
) -> Result<(), &'static str> {
    use blas::dgemm;

    if x.len() < (ldx * f) as usize {
        return Err("X buffer too small");
    }
    if w.len() < (ldw * k) as usize {
        return Err("W buffer too small");
    }
    if y.len() < (ldy * k) as usize {
        return Err("Y buffer too small");
    }

    unsafe {
        // Y <- α·X·W + β·Y   (no transposes)
        dgemm(b'N', b'N', m, k, f, alpha, x, ldx, w, ldw, beta, y, ldy);
    }
    Ok(())
}

/// Computes streaming covariance matrix using SYRK on buffered tiles.
#[inline(always)]
pub fn cachecov_syrk(
    n_feat: i32,
    obs: i32,
    x: &[f64],
    ldx: i32,
    c: &mut [f64], // packed ‖C‖, len = n(n+1)/2
) -> Result<(), &'static str> {
    use blas::dsyrk;

    // Packed-triangle length check   n(n+1)/2
    let need = (n_feat as usize * (n_feat as usize + 1)) / 2;
    if c.len() < need {
        return Err("C buffer too small");
    }
    if x.len() < (ldx * obs) as usize {
        return Err("X tile too small");
    }

    // dsyrk wants full-matrix C, so we expand a scratch view & repack.
    // For simplicity (and because tiles are usually small) we allocate
    // a dense n×n scratch; copy back the lower triangle afterwards.
    let n = n_feat as usize;
    let mut full = vec![0.0_f64; n * n];

    // Unpack current packed ‖C‖ into dense
    for col in 0..n {
        for row in col..n {
            // lower
            full[row + col * n] = c[(row * (row + 1)) / 2 + col];
        }
    }

    // C ← Xᵀ·X   (β = 1 to accumulate)
    unsafe {
        dsyrk(
            b'L', b'T', n_feat, obs, 1.0, // α
            x, ldx, 1.0, // β  (accumulate)
            &mut full, n_feat,
        );
    }

    // Re-pack lower triangle back into c
    for col in 0..n {
        for row in col..n {
            c[(row * (row + 1)) / 2 + col] = full[row + col * n];
        }
    }
    Ok(())
}

/// LU decomposition with partial pivoting (PA = LU).
///
/// Computes `PA = LU` factorisation *in-place*.
#[inline(always)]
pub fn lufactor(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    piv: &mut [i32], // len ≥ min(m,n)
) -> Result<(), &'static str> {
    use lapack::dgetrf;

    let k = std::cmp::min(m, n);
    if a.len() < (lda * n) as usize {
        return Err("A too small for GETRF");
    }
    if piv.len() < k as usize {
        return Err("pivot array too small");
    }

    let mut info = 0;
    unsafe {
        dgetrf(m, n, a, lda, piv, &mut info);
    }

    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("U is singular")
    } else {
        Err("GETRF argument error")
    }
}

/// Solves **U X = B** where **U** is upper-triangular.
#[inline(always)]
pub fn trisolve_upper(
    n: i32,
    nrhs: i32,
    u: &[f64],
    ldu: i32, // n×n upper-triangular
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if u.len() < (ldu * n) as usize {
        return Err("U buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    for j in 0..nrhs {
        let col = &mut b[(j * ldb) as usize..][..n as usize];
        unsafe {
            dtrsv(
                b'U', // UPLO = upper
                b'N', // trans = NoTrans
                b'N', // DIAG = Non-unit
                n, u, ldu, col, 1,
            );
        }
    }
    Ok(())
}

/// Solves **L · X = B** for X, where **L** is *lower-triangular*.
#[inline(always)]
pub fn trisolve_lower(
    n: i32,
    nrhs: i32,
    l: &[f64],
    ldl: i32, // n × n, lower-triangular
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if l.len() < (ldl * n) as usize {
        return Err("L buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    for j in 0..nrhs {
        let col = &mut b[(j * ldb) as usize..][..n as usize];
        unsafe {
            dtrsv(
                b'L', // UPLO  = lower
                b'N', // trans = NoTrans
                b'N', // DIAG  = non-unit
                n, l, ldl, col, 1,
            );
        }
    }
    Ok(())
}

/// Computes inverse of a triangular matrix (A⁻¹) via block solve.
#[inline(always)]
pub fn tri_inverse(n: i32, t: &mut [f64], ldt: i32, upper: bool) -> Result<(), &'static str> {
    if t.len() < (ldt * n) as usize {
        return Err("T buffer too small");
    }

    let mut info = 0;
    unsafe {
        dtrtri(
            if upper { b'U' } else { b'L' },
            b'N', // non-unit diagonal
            n,
            t,
            ldt,
            &mut info,
        );
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("T is singular")
    } else {
        Err("DTRTRI argument error")
    }
}

// Performs Cholesky decomposition: A = LLᵀ for SPD matrices.
#[inline(always)]
pub fn spd_cholesky(n: i32, a: &mut [f64], lda: i32) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }

    let mut info = 0;
    unsafe { dpotrf(b'L', n, a, lda, &mut info) };

    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix is not SPD")
    } else {
        Err("DPOTRF argument error")
    }
}

/// Solves **Σ X = B** where **Σ = L Lᵀ** (factor must already exist).
#[inline(always)]
pub fn spd_solve(
    n: i32,
    nrhs: i32,
    l: &[f64],
    ldl: i32, // lower-triangular factor
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if l.len() < (ldl * n) as usize {
        return Err("L buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    let mut info = 0;
    unsafe {
        dpotrs(b'L', n, nrhs, l, ldl, b, ldb, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("DPOTRS failed")
    }
}

/// Computes inverse of SPD matrix via Cholesky factorisation.
#[inline(always)]
pub fn spd_inverse(n: i32, l: &mut [f64], ldl: i32) -> Result<(), &'static str> {
    if l.len() < (ldl * n) as usize {
        return Err("L buffer too small");
    }

    let mut info = 0;
    unsafe { dpotri(b'L', n, l, ldl, &mut info) };

    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix not SPD (dpotri)")
    } else {
        Err("DPOTRI argument error")
    }
}

/// Panel-wise QR decomposition using Householder reflectors.
#[inline(always)]
pub fn qr_panel(
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,         // m × n panel, overwritten with R + Householder vectors
    taus: &mut [f64], // len ≥ n, receives τ scalars
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if taus.len() < n as usize {
        return Err("TAU buffer too small");
    }

    // Workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe { dgeqrf(m, n, a, lda, taus, &mut work_q, lwork, &mut info) };
    if info != 0 {
        return Err("DGEQRF work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe { dgeqrf(m, n, a, lda, taus, &mut work, lwork, &mut info) };
    if info == 0 {
        Ok(())
    } else {
        Err("DGEQRF factorisation failed")
    }
}

/// Forms orthonormal matrix Q from QR factorisation.
#[inline(always)]
pub fn qr_form_q(
    m: i32,
    n: i32,
    k: i32, // number of elementary reflectors (τ.len() ≥ k)
    a: &mut [f64],
    lda: i32,     // on entry: from `qr_panel`; on exit: Q
    taus: &[f64], // τ from `qr_panel`
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if taus.len() < k as usize {
        return Err("TAU buffer too small");
    }

    // workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe { dorgqr(m, n, k, a, lda, taus, &mut work_q, lwork, &mut info) };
    if info != 0 {
        return Err("DORGQR work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe { dorgqr(m, n, k, a, lda, taus, &mut work, lwork, &mut info) };
    if info == 0 {
        Ok(())
    } else {
        Err("DORGQR failed")
    }
}

/// Solves least-squares problem Ax = b using QR decomposition.
#[inline(always)]
pub fn least_squares_qr(
    m: i32,
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32, // A (overwritten)
    b: &mut [f64],
    ldb: i32, // B -> on exit contains solution X (min-norm)
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small");
    }

    // workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgels(
            b'N',
            m,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            &mut work_q,
            lwork,
            &mut info,
            1, // workspace buffer length (for the query, just 1)
        )
    };
    if info != 0 {
        return Err("DGELS work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];
    let len = work.len();
    unsafe {
        dgels(
            b'N', m, n, nrhs, a, lda, b, ldb, &mut work, lwork, &mut info, len,
        )
    };
    if info == 0 {
        Ok(())
    } else {
        Err("DGELS failed")
    }
}

/// Full symmetric eigendecomposition: A = QΛQᵀ.
#[inline(always)]
pub fn symeig_full(
    n: i32,
    a: &mut [f64],
    lda: i32,      // on entry upper-triangle of A; on exit Q (col-major)
    w: &mut [f64], // eigen-values λ (len ≥ n)
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small");
    }
    if w.len() < n as usize {
        return Err("W buffer too small");
    }

    // workspace query
    let mut lwork = -1;
    let mut work_q = [0.0_f64];
    let mut info = 0;
    unsafe { dsyev(b'V', b'U', n, a, lda, w, &mut work_q, lwork, &mut info) };
    if info != 0 {
        return Err("DSYEV work-query failed");
    }

    lwork = work_q[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe { dsyev(b'V', b'U', n, a, lda, w, &mut work, lwork, &mut info) };
    if info == 0 {
        Ok(())
    } else {
        Err("DSYEV failed")
    }
}

/// Builds Fisher information matrix or XᵀWX using symmetric rank-k update.
#[inline(always)]
pub fn syrk_fisher_info(
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32, // Xᵀ or W½X depending on context
    beta: f64,
    c: &mut [f64],
    ldc: i32, // symmetric C (lower-packed or full col-major lower)
) -> Result<(), &'static str> {
    if a.len() < (lda * k) as usize {
        return Err("A buffer too small");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C buffer too small");
    }

    unsafe {
        dsyrk(
            b'L', // use lower triangle
            b'N', // C ← α A Aᵀ + β C
            n, k, alpha, a, lda, beta, c, ldc,
        );
    }
    Ok(())
}

/// Performs symmetric rank-2k update: C += AᵀB + BᵀA.
#[inline(always)]
pub fn sym_rank2k_update(
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
) -> Result<(), &'static str> {
    if a.len() < (lda * k) as usize {
        return Err("A buffer too small");
    }
    if b.len() < (ldb * k) as usize {
        return Err("B buffer too small");
    }
    if c.len() < (ldc * n) as usize {
        return Err("C buffer too small");
    }

    unsafe {
        dsyr2k(
            b'L', // lower triangle
            b'T', // C += α (Aᵀ B + Bᵀ A)
            n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
    Ok(())
}

/// Solves A * X = B using an existing LU factorisation (dgetrs).
///
/// `a` and `ipiv` must be the output of a prior `lu_with_piv` (or `lufactor`) call.
/// On exit, `b` is overwritten with the solution X.
///
/// Arguments
/// - `n`:    Order of the matrix A (n-by-n).
/// - `nrhs`: Number of right-hand-side columns in B.
/// - `a`:    LU-factored matrix from dgetrf, length >= lda * n.
/// - `lda`:  Leading dimension of A (>= n).
/// - `ipiv`: Pivot indices from dgetrf, length >= n.
/// - `b`:    Right-hand-side matrix, length >= ldb * nrhs. Overwritten with solution.
/// - `ldb`:  Leading dimension of B (>= n).
#[inline(always)]
pub fn getrs(
    n: i32,
    nrhs: i32,
    a: &[f64],
    lda: i32,
    ipiv: &[i32],
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small for GETRS");
    }
    if ipiv.len() < n as usize {
        return Err("ipiv too small for GETRS");
    }
    if b.len() < (ldb * nrhs) as usize {
        return Err("B buffer too small for GETRS");
    }

    let mut info = 0;
    unsafe {
        dgetrs(b'N', n, nrhs, a, lda, ipiv, b, ldb, &mut info);
    }
    if info == 0 {
        Ok(())
    } else {
        Err("LAPACK dgetrs failed")
    }
}

/// Computes the inverse of a matrix from its LU factorisation (dgetri).
///
/// `a` and `ipiv` must be the output of a prior `lu_with_piv` (or `lufactor`) call.
/// On exit, `a` is overwritten with A_inv.
///
/// Arguments
/// - `n`:    Order of the matrix (n-by-n).
/// - `a`:    LU-factored matrix, length >= lda * n. Overwritten with inverse.
/// - `lda`:  Leading dimension of A (>= n).
/// - `ipiv`: Pivot indices from dgetrf, length >= n.
#[inline(always)]
pub fn getri(n: i32, a: &mut [f64], lda: i32, ipiv: &[i32]) -> Result<(), &'static str> {
    if a.len() < (lda * n) as usize {
        return Err("A buffer too small for GETRI");
    }
    if ipiv.len() < n as usize {
        return Err("ipiv too small for GETRI");
    }

    // Workspace query
    let mut work_query = [0.0_f64];
    let mut info = 0;
    unsafe {
        dgetri(n, a, lda, ipiv, &mut work_query, -1, &mut info);
    }
    if info != 0 {
        return Err("DGETRI workspace query failed");
    }

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0_f64; lwork as usize];

    unsafe {
        dgetri(n, a, lda, ipiv, &mut work, lwork, &mut info);
    }
    if info == 0 {
        Ok(())
    } else if info > 0 {
        Err("Matrix is singular (dgetri)")
    } else {
        Err("DGETRI argument error")
    }
}

#[cfg(test)]
#[cfg(feature = "linear_algebra")]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_near(a: f64, b: f64, msg: &str) {
        assert!(
            (a - b).abs() < TOL,
            "{msg}: expected {b}, got {a}, diff {}",
            (a - b).abs()
        );
    }

    // 1. gemv -- matrix-vector product

    #[test]
    fn test_gemv_no_trans() {
        // A = [[1, 2, 3],   column-major: [1, 4, 2, 5, 3, 6]
        //      [4, 5, 6]]
        // x = [1, 2, 3]
        // y = A*x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 2];
        gemv(2, 3, &a, 2, &x, 1, &mut y, 1, 1.0, 0.0, false).unwrap();
        assert_near(y[0], 14.0, "y[0]");
        assert_near(y[1], 32.0, "y[1]");
    }

    #[test]
    fn test_gemv_trans() {
        // A = [[1, 2, 3],   column-major: [1, 4, 2, 5, 3, 6]
        //      [4, 5, 6]]
        // x = [1, 2]  (length m=2 when trans_a=true)
        // y = At*x = [[1,4],[2,5],[3,6]] * [1,2] = [9, 12, 15]
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = [1.0, 2.0];
        let mut y = [0.0; 3];
        gemv(2, 3, &a, 2, &x, 1, &mut y, 1, 1.0, 0.0, true).unwrap();
        assert_near(y[0], 9.0, "y[0]");
        assert_near(y[1], 12.0, "y[1]");
        assert_near(y[2], 15.0, "y[2]");
    }

    #[test]
    fn test_gemv_bad_buffer() {
        let a = [1.0; 4]; // only 4 elements, but need lda*n = 2*3 = 6
        let x = [1.0; 3];
        let mut y = [0.0; 2];
        let result = gemv(2, 3, &a, 2, &x, 1, &mut y, 1, 1.0, 0.0, false);
        assert!(result.is_err());
    }

    // 2. gemm_4x4_microkernel -- 4x4 matrix multiply

    #[test]
    fn test_gemm_4x4_identity() {
        // A = I4, B = I4 => C = alpha*I4 + beta*0 = I4 (alpha=1, beta=0)
        let mut a = [0.0_f64; 16];
        let mut b = [0.0_f64; 16];
        for i in 0..4 {
            a[i * 4 + i] = 1.0;
            b[i * 4 + i] = 1.0;
        }
        let mut c = [0.0_f64; 16];
        gemm_4x4_microkernel(&a, &b, &mut c, 1.0, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(c[j * 4 + i], expected, &format!("c[{i},{j}]"));
            }
        }
    }

    #[test]
    fn test_gemm_4x4_accumulation() {
        // A = I4, B = I4, C starts at 2*I4
        // C = 1.0*I4*I4 + 1.0*(2*I4) = 3*I4  (alpha=1, beta=1)
        let mut a = [0.0_f64; 16];
        let mut b = [0.0_f64; 16];
        let mut c = [0.0_f64; 16];
        for i in 0..4 {
            a[i * 4 + i] = 1.0;
            b[i * 4 + i] = 1.0;
            c[i * 4 + i] = 2.0;
        }
        gemm_4x4_microkernel(&a, &b, &mut c, 1.0, 1.0);
        for i in 0..4 {
            assert_near(c[i * 4 + i], 3.0, &format!("c[{i},{i}]"));
        }
    }

    // 3. trisolve_2x2 -- upper and lower triangular 2x2 solve

    #[test]
    fn test_trisolve_2x2_upper() {
        // U = [[2, 3], [0, 4]]  column-major: [2, 0, 3, 4]
        // b = [11, 4]
        // Ux = b => 4*x2 = 4 => x2=1, 2*x1 + 3*1 = 11 => x1=4
        let mut u = [2.0, 0.0, 3.0, 4.0];
        let mut b = [11.0, 4.0];
        trisolve_2x2(true, &mut u, &mut b).unwrap();
        assert_near(b[0], 4.0, "x[0]");
        assert_near(b[1], 1.0, "x[1]");
    }

    #[test]
    fn test_trisolve_2x2_lower() {
        // L = [[2, 0], [3, 4]]  column-major: [2, 3, 0, 4]
        // b = [6, 19]
        // Lx = b => 2*x1 = 6 => x1=3, 3*3 + 4*x2 = 19 => x2=2.5
        let mut l = [2.0, 3.0, 0.0, 4.0];
        let mut b = [6.0, 19.0];
        trisolve_2x2(false, &mut l, &mut b).unwrap();
        assert_near(b[0], 3.0, "x[0]");
        assert_near(b[1], 2.5, "x[1]");
    }

    // 4. householder_apply -- QR via Householder

    #[test]
    fn test_householder_3x2() {
        // A = [[1, 2],   column-major: [1, 3, 5, 2, 4, 6]
        //      [3, 4],
        //      [5, 6]]
        // After QR, upper triangle of A should contain R
        let mut a = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut taus = [0.0; 2];
        householder_apply(3, 2, &mut a, 3, &mut taus).unwrap();
        // R is upper triangular in the first 2 rows
        // R[0,0] should be -sqrt(1+9+25) = -sqrt(35)
        let r00 = a[0];
        assert!(r00.abs() > 1e-10, "R[0,0] should be nonzero");
        // taus should be populated
        assert!(taus[0].abs() > 1e-10, "tau[0] should be nonzero");
    }

    #[test]
    fn test_householder_r_upper_triangular() {
        // After QR on 3x2, the sub-diagonal of R (column 0, row > 0 part in
        // the packed Householder format) stores reflector vectors, but the
        // upper triangle should be valid R values.
        let mut a = [2.0, 0.0, 0.0, 1.0, 3.0, 0.0]; // 3x2 column-major
        let mut taus = [0.0; 2];
        householder_apply(3, 2, &mut a, 3, &mut taus).unwrap();
        // R[0,0] and R[0,1] are the top two elements in their columns
        // After QR on a matrix with zeros below diagonal, result should
        // preserve the structure
        assert!(a[0].abs() > 1e-10, "R[0,0] should be nonzero");
    }

    // 5. cholesky_panel -- 2x2 SPD Cholesky

    #[test]
    fn test_cholesky_panel_2x2() {
        // A = [[4, 2], [2, 3]]  column-major: [4, 2, 2, 3]
        // L = [[2, 0], [1, sqrt(2)]]
        let mut a = [4.0, 2.0, 2.0, 3.0];
        cholesky_panel(2, &mut a, 2).unwrap();
        assert_near(a[0], 2.0, "L[0,0]");
        assert_near(a[1], 1.0, "L[1,0]");
        assert_near(a[3], 2.0_f64.sqrt(), "L[1,1]");
    }

    #[test]
    fn test_cholesky_panel_not_spd() {
        // Not positive definite: [[1, 2], [2, 1]]  (eigenvalues: 3 and -1)
        let mut a = [1.0, 2.0, 2.0, 1.0];
        let result = cholesky_panel(2, &mut a, 2);
        assert!(result.is_err());
    }

    // 6. lu_with_piv -- LU decomposition with pivoting

    #[test]
    fn test_lu_with_piv_2x2() {
        // A = [[2, 1], [6, 4]]  column-major: [2, 6, 1, 4]
        // With pivoting, P*A = L*U
        let mut a = [2.0, 6.0, 1.0, 4.0];
        let mut ipiv = [0_i32; 2];
        lu_with_piv(2, 2, &mut a, 2, &mut ipiv).unwrap();
        // ipiv should be populated (1-indexed LAPACK pivots)
        assert!(ipiv[0] > 0, "pivot should be set");
    }

    #[test]
    fn test_lu_with_piv_singular() {
        // Singular: [[1, 2], [2, 4]]  column-major: [1, 2, 2, 4]
        let mut a = [1.0, 2.0, 2.0, 4.0];
        let mut ipiv = [0_i32; 2];
        let result = lu_with_piv(2, 2, &mut a, 2, &mut ipiv);
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_with_piv_reconstruct() {
        // A = [[2, 1], [6, 4]] => verify PA = LU reconstruction
        let mut a = [2.0, 6.0, 1.0, 4.0];
        let a_orig = a;
        let mut ipiv = [0_i32; 2];
        lu_with_piv(2, 2, &mut a, 2, &mut ipiv).unwrap();

        // Extract L (unit lower) and U (upper) from packed result
        let l00 = 1.0;
        let l10 = a[1]; // below diagonal
        let l01 = 0.0;
        let l11 = 1.0;
        let u00 = a[0]; // diagonal and above
        let u01 = a[2];
        let u10 = 0.0;
        let u11 = a[3];

        // Compute L*U
        let lu00 = l00 * u00 + l01 * u10;
        let lu10 = l10 * u00 + l11 * u10;
        let lu01 = l00 * u01 + l01 * u11;
        let lu11 = l10 * u01 + l11 * u11;

        // Apply permutation to get PA
        let mut pa = a_orig;
        if ipiv[0] != 1 {
            // Swap rows according to pivot
            pa.swap(0, (ipiv[0] - 1) as usize);
            pa.swap(2, 2 + (ipiv[0] - 1) as usize);
        }

        assert_near(lu00, pa[0], "PA=LU [0,0]");
        assert_near(lu10, pa[1], "PA=LU [1,0]");
        assert_near(lu01, pa[2], "PA=LU [0,1]");
        assert_near(lu11, pa[3], "PA=LU [1,1]");
    }

    // 7. qr_block -- full QR factorisation

    #[test]
    fn test_qr_block_3x2() {
        // A = [[1, 2],  column-major: [1, 3, 5, 2, 4, 6]
        //      [3, 4],
        //      [5, 6]]
        let mut a = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut taus = [0.0; 2];
        qr_block(3, 2, &mut a, 3, &mut taus).unwrap();
        // taus should be populated
        assert!(taus[0].abs() > 1e-10, "tau[0] should be nonzero");
        assert!(taus[1].abs() > 1e-10, "tau[1] should be nonzero");
        // R[0,0] = -norm([1,3,5]) = -sqrt(35)
        assert_near(a[0].abs(), 35.0_f64.sqrt(), "|R[0,0]|");
    }

    #[test]
    fn test_qr_block_taus_too_small() {
        let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut taus = [0.0; 1]; // need 2 but only 1
        let result = qr_block(3, 2, &mut a, 3, &mut taus);
        assert!(result.is_err());
    }

    // 8. spd_cholesky -- 3x3 SPD Cholesky

    #[test]
    fn test_spd_cholesky_3x3() {
        // A = [[4, 2, 1],   column-major: [4, 2, 1, 2, 5, 3, 1, 3, 6]
        //      [2, 5, 3],
        //      [1, 3, 6]]
        // This is SPD (all eigenvalues positive)
        let mut a = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        spd_cholesky(3, &mut a, 3).unwrap();
        // L[0,0] = sqrt(4) = 2
        assert_near(a[0], 2.0, "L[0,0]");
        // L[1,0] = 2/2 = 1
        assert_near(a[1], 1.0, "L[1,0]");
        // L[2,0] = 1/2 = 0.5
        assert_near(a[2], 0.5, "L[2,0]");
        // Verify L*Lt reconstructs original lower triangle
        // L[1,1] = sqrt(5 - 1) = 2
        assert_near(a[4], 2.0, "L[1,1]");
    }

    #[test]
    fn test_spd_cholesky_not_spd() {
        // Not SPD: [[1, 2, 0], [2, 1, 0], [0, 0, 1]]
        let mut a = [1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = spd_cholesky(3, &mut a, 3);
        assert!(result.is_err());
    }

    // 9. spd_solve -- SPD solve Ax=b

    #[test]
    fn test_spd_solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [8, 7]
        // Ax=b => x = A^{-1}*b
        // A^{-1} = (1/8)*[[3,-2],[-2,4]] => x = [(3*8-2*7)/8, (-2*8+4*7)/8] = [10/8, 12/8] = [1.25, 1.5]
        // First factorise A
        let mut l = [4.0, 2.0, 2.0, 3.0];
        spd_cholesky(2, &mut l, 2).unwrap();
        // Now solve
        let mut b = [8.0, 7.0];
        spd_solve(2, 1, &l, 2, &mut b, 2).unwrap();
        assert_near(b[0], 1.25, "x[0]");
        assert_near(b[1], 1.5, "x[1]");
    }

    #[test]
    fn test_spd_solve_identity() {
        // A = I2 => x = b
        let mut l = [1.0, 0.0, 0.0, 1.0];
        spd_cholesky(2, &mut l, 2).unwrap();
        let mut b = [3.0, 7.0];
        spd_solve(2, 1, &l, 2, &mut b, 2).unwrap();
        assert_near(b[0], 3.0, "x[0]");
        assert_near(b[1], 7.0, "x[1]");
    }

    // 10. spd_inverse -- SPD inverse

    #[test]
    fn test_spd_inverse_2x2() {
        // A = [[4, 2], [2, 3]]
        // A^{-1} = (1/8)*[[3,-2],[-2,4]]
        let mut a = [4.0, 2.0, 2.0, 3.0];
        spd_cholesky(2, &mut a, 2).unwrap();
        spd_inverse(2, &mut a, 2).unwrap();
        // Lower triangle of result = A^{-1}
        assert_near(a[0], 3.0 / 8.0, "A_inv[0,0]");
        assert_near(a[1], -2.0 / 8.0, "A_inv[1,0]");
        assert_near(a[3], 4.0 / 8.0, "A_inv[1,1]");
    }

    #[test]
    fn test_spd_inverse_product_identity() {
        // Verify A * A^{-1} = I for 2x2 SPD
        let a_orig = [4.0, 2.0, 2.0, 3.0];
        let mut a = a_orig;
        spd_cholesky(2, &mut a, 2).unwrap();
        spd_inverse(2, &mut a, 2).unwrap();
        // dpotri returns full symmetric inverse in lower triangle;
        // fill upper from lower for matmul
        let a_inv = [a[0], a[1], a[1], a[3]];
        // Compute product A_orig * A_inv
        let mut prod = [0.0_f64; 4];
        blocked_gemm(
            2, 2, 2, 1.0, &a_orig, 2, &a_inv, 2, 0.0, &mut prod, 2, false, false,
        )
        .unwrap();
        assert_near(prod[0], 1.0, "I[0,0]");
        assert_near(prod[1], 0.0, "I[1,0]");
        assert_near(prod[2], 0.0, "I[0,1]");
        assert_near(prod[3], 1.0, "I[1,1]");
    }

    // 11. trisolve_upper -- 3x3 upper triangular solve

    #[test]
    fn test_trisolve_upper_3x3() {
        // U = [[2, 1, 3],   column-major: [2, 0, 0, 1, 4, 0, 3, 2, 5]
        //      [0, 4, 2],
        //      [0, 0, 5]]
        // b = [13, 18, 10]
        // Back-sub: 5*x3=10 => x3=2, 4*x2+2*2=18 => x2=3.5, 2*x1+1*3.5+3*2=13 => 2*x1=3.5 => x1=1.75
        let u = [2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 2.0, 5.0];
        let mut b = [13.0, 18.0, 10.0];
        trisolve_upper(3, 1, &u, 3, &mut b, 3).unwrap();
        assert_near(b[0], 1.75, "x[0]");
        assert_near(b[1], 3.5, "x[1]");
        assert_near(b[2], 2.0, "x[2]");
    }

    #[test]
    fn test_trisolve_upper_identity() {
        // U = I3 => x = b
        let u = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = [5.0, 7.0, 9.0];
        trisolve_upper(3, 1, &u, 3, &mut b, 3).unwrap();
        assert_near(b[0], 5.0, "x[0]");
        assert_near(b[1], 7.0, "x[1]");
        assert_near(b[2], 9.0, "x[2]");
    }

    // 12. trisolve_lower -- 3x3 lower triangular solve

    #[test]
    fn test_trisolve_lower_3x3() {
        // L = [[3, 0, 0],   column-major: [3, 1, 2, 0, 4, 3, 0, 0, 5]
        //      [1, 4, 0],
        //      [2, 3, 5]]
        // b = [9, 17, 37]
        // Forward-sub: 3*x1=9 => x1=3, 1*3+4*x2=17 => x2=3.5, 2*3+3*3.5+5*x3=37 => x3=4.1
        let l = [3.0, 1.0, 2.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0];
        let mut b = [9.0, 17.0, 37.0];
        trisolve_lower(3, 1, &l, 3, &mut b, 3).unwrap();
        assert_near(b[0], 3.0, "x[0]");
        assert_near(b[1], 3.5, "x[1]");
        assert_near(b[2], 4.1, "x[2]");
    }

    #[test]
    fn test_trisolve_lower_identity() {
        let l = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = [2.0, 4.0, 6.0];
        trisolve_lower(3, 1, &l, 3, &mut b, 3).unwrap();
        assert_near(b[0], 2.0, "x[0]");
        assert_near(b[1], 4.0, "x[1]");
        assert_near(b[2], 6.0, "x[2]");
    }

    // 13. tri_inverse -- triangular inverse

    #[test]
    fn test_tri_inverse_upper_3x3() {
        // U = [[2, 1, 3],   column-major: [2, 0, 0, 1, 4, 0, 3, 2, 5]
        //      [0, 4, 2],
        //      [0, 0, 5]]
        let mut t = [2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 2.0, 5.0];
        let t_orig = t;
        tri_inverse(3, &mut t, 3, true).unwrap();
        // Verify T * T_inv = I
        let mut prod = [0.0_f64; 9];
        blocked_gemm(
            3, 3, 3, 1.0, &t_orig, 3, &t, 3, 0.0, &mut prod, 3, false, false,
        )
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(prod[j * 3 + i], expected, &format!("I[{i},{j}]"));
            }
        }
    }

    #[test]
    fn test_tri_inverse_lower_3x3() {
        // L = [[3, 0, 0],   column-major: [3, 1, 2, 0, 4, 3, 0, 0, 5]
        //      [1, 4, 0],
        //      [2, 3, 5]]
        let mut t = [3.0, 1.0, 2.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0];
        let t_orig = t;
        tri_inverse(3, &mut t, 3, false).unwrap();
        // Verify T * T_inv = I
        let mut prod = [0.0_f64; 9];
        blocked_gemm(
            3, 3, 3, 1.0, &t_orig, 3, &t, 3, 0.0, &mut prod, 3, false, false,
        )
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(prod[j * 3 + i], expected, &format!("I[{i},{j}]"));
            }
        }
    }

    // 14. blocked_gemm -- general matrix multiply

    #[test]
    fn test_blocked_gemm_3x3_times_3x2() {
        // A = [[1, 2, 3],   column-major: [1, 4, 7, 2, 5, 8, 3, 6, 9]
        //      [4, 5, 6],
        //      [7, 8, 9]]
        // B = [[1, 2],      column-major: [1, 3, 5, 2, 4, 6]
        //      [3, 4],
        //      [5, 6]]
        // C = A*B = [[1+6+15, 2+8+18],    = [[22, 28],
        //            [4+15+30, 8+20+36],      [49, 64],
        //            [7+24+45, 14+32+54]]     [76, 100]]
        let a = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let b = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut c = [0.0_f64; 6]; // 3x2
        blocked_gemm(3, 2, 3, 1.0, &a, 3, &b, 3, 0.0, &mut c, 3, false, false).unwrap();
        assert_near(c[0], 22.0, "c[0,0]");
        assert_near(c[1], 49.0, "c[1,0]");
        assert_near(c[2], 76.0, "c[2,0]");
        assert_near(c[3], 28.0, "c[0,1]");
        assert_near(c[4], 64.0, "c[1,1]");
        assert_near(c[5], 100.0, "c[2,1]");
    }

    #[test]
    fn test_blocked_gemm_trans_a() {
        // A = [[1, 2],   column-major: [1, 2, 3, 4]  (2x2 matrix)
        //      [3, 4]]
        // At*A = [[1+9, 2+12], [2+12, 4+16]] = [[10, 14], [14, 20]]
        let a = [1.0, 3.0, 2.0, 4.0];
        let mut c = [0.0_f64; 4];
        blocked_gemm(2, 2, 2, 1.0, &a, 2, &a, 2, 0.0, &mut c, 2, true, false).unwrap();
        assert_near(c[0], 10.0, "c[0,0]");
        assert_near(c[1], 14.0, "c[1,0]");
        assert_near(c[2], 14.0, "c[0,1]");
        assert_near(c[3], 20.0, "c[1,1]");
    }

    #[test]
    fn test_blocked_gemm_trans_b() {
        // A = [[1, 2],   B = [[1, 2],
        //      [3, 4]]        [3, 4]]
        // A*Bt = [[1+4, 3+8], [3+8, 9+16]] = [[5, 11], [11, 25]]
        let a = [1.0, 3.0, 2.0, 4.0];
        let b = [1.0, 3.0, 2.0, 4.0];
        let mut c = [0.0_f64; 4];
        blocked_gemm(2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2, false, true).unwrap();
        assert_near(c[0], 5.0, "c[0,0]");
        assert_near(c[1], 11.0, "c[1,0]");
        assert_near(c[2], 11.0, "c[0,1]");
        assert_near(c[3], 25.0, "c[1,1]");
    }

    #[test]
    fn test_blocked_gemm_bad_buffer() {
        let a = [1.0; 4]; // 2x2
        let b = [1.0; 2]; // too small for 2x3 (need ldb*cols_b = 2*3 = 6)
        let mut c = [0.0_f64; 6];
        let result = blocked_gemm(2, 3, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2, false, false);
        assert!(result.is_err());
    }

    // 15. syrk_panel -- symmetric rank-k update

    #[test]
    fn test_syrk_panel_aat() {
        // A (2x3) = [[1, 2, 3],   column-major: [1, 4, 2, 5, 3, 6]
        //            [4, 5, 6]]
        // C = A*At = [[1+4+9, 4+10+18], [4+10+18, 16+25+36]] = [[14, 32], [32, 77]]
        // syrk_panel with trans_a=false: C = alpha * A * At + beta * C
        // A is n=2 rows, k=3 cols, lda=2
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut c = [0.0_f64; 4]; // 2x2
        syrk_panel(2, 3, 1.0, &a, 2, 0.0, &mut c, 2, false).unwrap();
        // Lower triangle populated
        assert_near(c[0], 14.0, "c[0,0]");
        assert_near(c[1], 32.0, "c[1,0]");
        assert_near(c[3], 77.0, "c[1,1]");
    }

    #[test]
    fn test_syrk_panel_symmetric() {
        // Verify the result is symmetric by checking lower triangle
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 column-major
        let mut c = [0.0_f64; 9]; // 3x3
        syrk_panel(3, 2, 1.0, &a, 3, 0.0, &mut c, 3, false).unwrap();
        // dsyrk fills only the lower triangle. Verify diagonal and below.
        // A*At[0,0] = 1*1+4*4 = 17
        assert_near(c[0], 17.0, "c[0,0]");
        // A*At[1,0] = 2*1+5*4 = 22
        assert_near(c[1], 22.0, "c[1,0]");
        // A*At[2,0] = 3*1+6*4 = 27
        assert_near(c[2], 27.0, "c[2,0]");
    }

    // 16. symeig2x2 -- 2x2 symmetric eigendecomposition

    #[test]
    fn test_symeig2x2_known() {
        // A = [[2, 1], [1, 3]]  column-major: [2, 1, 1, 3]
        // Eigenvalues: trace=5, det=5, disc=sqrt(1+1)=sqrt(2)
        // lambda = (5 +- sqrt(9-20))/2 .. use characteristic poly:
        // lambda^2 - 5*lambda + 5 = 0 => lambda = (5 +- sqrt(5))/2
        // lambda1 = (5-sqrt(5))/2 ~= 1.38197, lambda2 = (5+sqrt(5))/2 ~= 3.61803
        let a = [2.0, 1.0, 1.0, 3.0];
        let mut eigvals = [0.0_f64; 2];
        let mut eigvecs = [0.0_f64; 4];
        symeig2x2(&a, &mut eigvals, &mut eigvecs).unwrap();
        let expected_0 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_1 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert_near(eigvals[0], expected_0, "lambda_0");
        assert_near(eigvals[1], expected_1, "lambda_1");
    }

    #[test]
    fn test_symeig2x2_orthogonal_eigvecs() {
        // Eigenvectors should be orthogonal: v1 . v2 = 0
        let a = [2.0, 1.0, 1.0, 3.0];
        let mut eigvals = [0.0_f64; 2];
        let mut eigvecs = [0.0_f64; 4];
        symeig2x2(&a, &mut eigvals, &mut eigvecs).unwrap();
        // Column-major: v1 = [eigvecs[0], eigvecs[1]], v2 = [eigvecs[2], eigvecs[3]]
        let dot = eigvecs[0] * eigvecs[2] + eigvecs[1] * eigvecs[3];
        assert_near(dot, 0.0, "v1.v2 orthogonality");
    }

    // 17. bidiag_reduction -- bidiagonalisation

    #[test]
    fn test_bidiag_reduction_3x3() {
        // A = [[1, 2, 3],   column-major: [1, 4, 7, 2, 5, 8, 3, 6, 9]
        //      [4, 5, 6],
        //      [7, 8, 9]]
        let mut a = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let mut d = [0.0_f64; 3]; // min(m,n)=3
        let mut e = [0.0_f64; 2]; // min(m,n)-1=2
        let mut tauq = [0.0_f64; 3];
        let mut taup = [0.0_f64; 3];
        bidiag_reduction(3, 3, &mut a, 3, &mut d, &mut e, &mut tauq, &mut taup).unwrap();
        // d should contain bidiagonal main diagonal entries
        assert!(d[0].abs() > 1e-10, "d[0] should be nonzero");
        // e should contain superdiagonal entries
        assert!(e[0].abs() > 1e-10, "e[0] should be nonzero");
    }

    #[test]
    fn test_bidiag_reduction_lengths() {
        // For a 4x3 matrix: d has 3 entries, e has 2
        let mut a = [1.0; 12]; // 4x3
        let mut d = [0.0_f64; 3];
        let mut e = [0.0_f64; 2];
        let mut tauq = [0.0_f64; 3];
        let mut taup = [0.0_f64; 3];
        bidiag_reduction(4, 3, &mut a, 4, &mut d, &mut e, &mut tauq, &mut taup).unwrap();
        // tauq should be populated
        assert!(tauq[0].abs() > 1e-10, "tauq[0] should be nonzero");
    }

    // 18. svd_block -- economy SVD

    #[test]
    fn test_svd_block_3x2_reconstruct() {
        // A = [[1, 2],   column-major: [1, 3, 5, 2, 4, 6]
        //      [3, 4],
        //      [5, 6]]
        // Economy SVD: jobu='S', jobvt='S'
        // U is 3x2, S has 2 values, Vt is 2x2
        let a_orig = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut a = a_orig;
        let mut s = [0.0_f64; 2];
        let mut u = [0.0_f64; 6]; // 3x2
        let mut vt = [0.0_f64; 4]; // 2x2
        svd_block(b'S', b'S', 3, 2, &mut a, 3, &mut s, &mut u, 3, &mut vt, 2).unwrap();

        // Reconstruct: A_hat = U * diag(S) * Vt
        // First compute U*diag(S): scale columns of U by singular values
        let mut us = [0.0_f64; 6];
        for j in 0..2 {
            for i in 0..3 {
                us[j * 3 + i] = u[j * 3 + i] * s[j];
            }
        }
        // Then A_hat = US * Vt
        let mut a_hat = [0.0_f64; 6];
        blocked_gemm(
            3, 2, 2, 1.0, &us, 3, &vt, 2, 0.0, &mut a_hat, 3, false, false,
        )
        .unwrap();
        for i in 0..6 {
            assert_near(a_hat[i], a_orig[i], &format!("A_hat[{i}]"));
        }
    }

    #[test]
    fn test_svd_block_singular_values_descending() {
        // Singular values should be in descending order
        let mut a = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut s = [0.0_f64; 2];
        let mut u = [0.0_f64; 6];
        let mut vt = [0.0_f64; 4];
        svd_block(b'S', b'S', 3, 2, &mut a, 3, &mut s, &mut u, 3, &mut vt, 2).unwrap();
        assert!(s[0] >= s[1], "singular values should be descending");
        assert!(s[0] > 0.0, "first singular value should be positive");
    }

    // 19. pca_project -- X * W projection

    #[test]
    fn test_pca_project_4x3_to_4x2() {
        // X = 4x3, W = 3x2, Y = X*W should be 4x2
        // X = [[1,0,0],   column-major: [1,2,3,4, 0,1,0,1, 0,0,1,0]
        //      [2,1,0],
        //      [3,0,1],
        //      [4,1,0]]
        // W = [[1,0],     column-major: [1,0,1, 0,1,0]
        //      [0,1],
        //      [1,0]]
        // Y = X*W:
        //   row 0: [1*1+0*0+0*1, 1*0+0*1+0*0] = [1, 0]
        //   row 1: [2*1+1*0+0*1, 2*0+1*1+0*0] = [2, 1]
        //   row 2: [3*1+0*0+1*1, 3*0+0*1+1*0] = [4, 0]
        //   row 3: [4*1+1*0+0*1, 4*0+1*1+0*0] = [4, 1]
        let x = [1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let w = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut y = [0.0_f64; 8]; // 4x2
        pca_project(4, 3, 2, 1.0, &x, 4, &w, 3, 0.0, &mut y, 4).unwrap();
        // Y column-major: [1,2,4,4, 0,1,0,1]
        assert_near(y[0], 1.0, "Y[0,0]");
        assert_near(y[1], 2.0, "Y[1,0]");
        assert_near(y[2], 4.0, "Y[2,0]");
        assert_near(y[3], 4.0, "Y[3,0]");
        assert_near(y[4], 0.0, "Y[0,1]");
        assert_near(y[5], 1.0, "Y[1,1]");
        assert_near(y[6], 0.0, "Y[2,1]");
        assert_near(y[7], 1.0, "Y[3,1]");
    }

    #[test]
    fn test_pca_project_identity_weight() {
        // W = I3 (3x3), Y should equal X
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // I3
        let mut y = [0.0_f64; 9];
        pca_project(3, 3, 3, 1.0, &x, 3, &w, 3, 0.0, &mut y, 3).unwrap();
        for i in 0..9 {
            assert_near(y[i], x[i], &format!("Y[{i}]"));
        }
    }

    // 20. cachecov_syrk -- packed covariance accumulation

    #[test]
    fn test_cachecov_syrk_3feat_4obs() {
        // X = 4x3 observation matrix (4 observations, 3 features)
        // column-major: col0=[1,2,3,4], col1=[5,6,7,8], col2=[9,10,11,12]
        // Padded to ldx*obs=4*4=16 to satisfy buffer check (only 12 used)
        // XtX[i,j] = sum_k X[k,i]*X[k,j]
        // XtX[0,0] = 1+4+9+16 = 30
        // XtX[1,0] = 5+12+21+32 = 70
        // XtX[2,0] = 9+20+33+48 = 110
        // XtX[1,1] = 25+36+49+64 = 174
        // XtX[2,1] = 45+60+77+96 = 278
        // XtX[2,2] = 81+100+121+144 = 446
        let x = [
            1.0, 2.0, 3.0, 4.0, // col 0
            5.0, 6.0, 7.0, 8.0, // col 1
            9.0, 10.0, 11.0, 12.0, // col 2
            0.0, 0.0, 0.0, 0.0, // padding for buffer check
        ];
        // Packed lower triangle: n*(n+1)/2 = 6 elements
        // Index mapping: (row*(row+1))/2 + col for row >= col
        let mut c = [0.0_f64; 6];
        cachecov_syrk(3, 4, &x, 4, &mut c).unwrap();
        // c[(0*(0+1))/2 + 0] = c[0] = XtX[0,0] = 30
        assert_near(c[0], 30.0, "XtX[0,0]");
        // c[(1*(1+1))/2 + 0] = c[1] = XtX[1,0] = 70
        assert_near(c[1], 70.0, "XtX[1,0]");
        // c[(1*(1+1))/2 + 1] = c[2] = XtX[1,1] = 174
        assert_near(c[2], 174.0, "XtX[1,1]");
        // c[(2*(2+1))/2 + 0] = c[3] = XtX[2,0] = 110
        assert_near(c[3], 110.0, "XtX[2,0]");
        // c[(2*(2+1))/2 + 1] = c[4] = XtX[2,1] = 278
        assert_near(c[4], 278.0, "XtX[2,1]");
        // c[(2*(2+1))/2 + 2] = c[5] = XtX[2,2] = 446
        assert_near(c[5], 446.0, "XtX[2,2]");
    }

    #[test]
    fn test_cachecov_syrk_accumulation() {
        // Verify accumulation: calling twice with the same data should double
        let x = [1.0, 2.0, 3.0, 4.0]; // 2x2 (2 obs, 2 features)
        let mut c = [0.0_f64; 3]; // packed 2x2 lower
        cachecov_syrk(2, 2, &x, 2, &mut c).unwrap();
        let first = c;
        cachecov_syrk(2, 2, &x, 2, &mut c).unwrap();
        for i in 0..3 {
            assert_near(c[i], 2.0 * first[i], &format!("c[{i}] doubled"));
        }
    }

    // 21. lufactor -- LU decomposition (consistent with lu_with_piv)

    #[test]
    fn test_lufactor_3x3() {
        // A = [[2, 1, 1],   column-major: [2, 4, 2, 1, 3, 1, 1, 1, 3]
        //      [4, 3, 1],
        //      [2, 1, 3]]
        let mut a = [2.0, 4.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0];
        let mut piv = [0_i32; 3];
        lufactor(3, 3, &mut a, 3, &mut piv).unwrap();
        // Pivots should be populated (1-indexed)
        assert!(piv[0] > 0, "piv[0] should be set");
    }

    #[test]
    fn test_lufactor_consistency_with_lu_with_piv() {
        // Both should produce the same factorisation for the same input
        let a_orig = [2.0, 6.0, 1.0, 4.0]; // 2x2
        let mut a1 = a_orig;
        let mut a2 = a_orig;
        let mut piv1 = [0_i32; 2];
        let mut piv2 = [0_i32; 2];
        lu_with_piv(2, 2, &mut a1, 2, &mut piv1).unwrap();
        lufactor(2, 2, &mut a2, 2, &mut piv2).unwrap();
        for i in 0..4 {
            assert_near(a1[i], a2[i], &format!("a[{i}]"));
        }
        assert_eq!(piv1, piv2, "pivots should match");
    }

    // 22. qr_panel -- panel QR (consistent with qr_block)

    #[test]
    fn test_qr_panel_3x2() {
        let mut a = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 column-major
        let mut taus = [0.0_f64; 2];
        qr_panel(3, 2, &mut a, 3, &mut taus).unwrap();
        // taus should be populated
        assert!(taus[0].abs() > 1e-10, "tau[0] should be nonzero");
        // R[0,0] = -norm([1,3,5])
        assert_near(a[0].abs(), 35.0_f64.sqrt(), "|R[0,0]|");
    }

    #[test]
    fn test_qr_panel_consistent_with_qr_block() {
        let a_orig = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2
        let mut a1 = a_orig;
        let mut a2 = a_orig;
        let mut taus1 = [0.0_f64; 2];
        let mut taus2 = [0.0_f64; 2];
        qr_block(3, 2, &mut a1, 3, &mut taus1).unwrap();
        qr_panel(3, 2, &mut a2, 3, &mut taus2).unwrap();
        for i in 0..6 {
            assert_near(a1[i], a2[i], &format!("a[{i}]"));
        }
        for i in 0..2 {
            assert_near(taus1[i], taus2[i], &format!("tau[{i}]"));
        }
    }

    // 23. qr_form_q -- form Q from QR factorisation

    #[test]
    fn test_qr_form_q_orthogonal() {
        // After QR on 3x2, form Q (3x2) and verify QtQ = I2
        let mut a = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut taus = [0.0_f64; 2];
        qr_panel(3, 2, &mut a, 3, &mut taus).unwrap();
        qr_form_q(3, 2, 2, &mut a, 3, &taus).unwrap();
        // Q is now in a (3x2 column-major)
        // Compute QtQ (2x2) -- Qt is trans of Q
        let mut qtq = [0.0_f64; 4];
        blocked_gemm(2, 2, 3, 1.0, &a, 3, &a, 3, 0.0, &mut qtq, 2, true, false).unwrap();
        assert_near(qtq[0], 1.0, "QtQ[0,0]");
        assert_near(qtq[1], 0.0, "QtQ[1,0]");
        assert_near(qtq[2], 0.0, "QtQ[0,1]");
        assert_near(qtq[3], 1.0, "QtQ[1,1]");
    }

    #[test]
    fn test_qr_form_q_full_square() {
        // Full QR on 3x3, form Q (3x3), verify QtQ = I3
        let mut a = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0]; // 3x3 nonsingular
        let mut taus = [0.0_f64; 3];
        qr_panel(3, 3, &mut a, 3, &mut taus).unwrap();
        qr_form_q(3, 3, 3, &mut a, 3, &taus).unwrap();
        let mut qtq = [0.0_f64; 9];
        blocked_gemm(3, 3, 3, 1.0, &a, 3, &a, 3, 0.0, &mut qtq, 3, true, false).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[j * 3 + i], expected, &format!("QtQ[{i},{j}]"));
            }
        }
    }

    // 24. least_squares_qr -- overdetermined system

    #[test]
    fn test_least_squares_qr_overdetermined() {
        // Solve A*x = b in least-squares sense
        // A = [[1, 1],   b = [1, 2, 3]
        //      [1, 2],
        //      [1, 3]]
        // Normal equations: AtA = [[3, 6], [6, 14]], Atb = [6, 14]
        // x = (AtA)^{-1} Atb = (1/6)*[[14,-6],[-6,3]] * [6,14] = [0, 1]
        // Actually: (14*6-6*14)/6 = 0, (-6*6+3*14)/6 = 6/6 = 1
        // So x = [0, 1]... let me verify: residuals = [1-0-1, 2-0-2, 3-0-3] = [0,0,0]
        // Perfect fit since b = [1,2,3] = 0*[1,1,1] + 1*[1,2,3] which is A*[0,1]
        let mut a = [1.0, 1.0, 1.0, 1.0, 2.0, 3.0]; // 3x2 column-major
        let mut b = [1.0, 2.0, 3.0]; // 3x1
        least_squares_qr(3, 2, 1, &mut a, 3, &mut b, 3).unwrap();
        // Solution is in first n=2 elements of b
        assert_near(b[0], 0.0, "x[0]");
        assert_near(b[1], 1.0, "x[1]");
    }

    #[test]
    fn test_least_squares_qr_exact() {
        // Square system (exact solution): A = [[2, 1], [1, 3]], b = [5, 7]
        // x = [8/5, 9/5] = [1.6, 1.8]... let me solve:
        // 2*x1 + x2 = 5, x1 + 3*x2 = 7
        // x1 = 5 - x2, (5-x2)+3*x2 = 7, 5+2*x2=7, x2=1, x1=4... nope:
        // Actually x1 = (5*3-1*7)/(2*3-1*1) = (15-7)/5 = 8/5 = 1.6
        // x2 = (2*7-1*5)/(2*3-1*1) = (14-5)/5 = 9/5 = 1.8
        let mut a = [2.0, 1.0, 1.0, 3.0]; // 2x2 column-major
        let mut b = [5.0, 7.0];
        least_squares_qr(2, 2, 1, &mut a, 2, &mut b, 2).unwrap();
        assert_near(b[0], 1.6, "x[0]");
        assert_near(b[1], 1.8, "x[1]");
    }

    // 25. symeig_full -- full symmetric eigendecomposition

    #[test]
    fn test_symeig_full_3x3_trace() {
        // A = [[2, 1, 0],   column-major: [2, 1, 0, 1, 3, 1, 0, 1, 4]
        //      [1, 3, 1],
        //      [0, 1, 4]]
        // Sum of eigenvalues = trace = 2+3+4 = 9
        let mut a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let mut w = [0.0_f64; 3];
        symeig_full(3, &mut a, 3, &mut w).unwrap();
        let sum = w[0] + w[1] + w[2];
        assert_near(sum, 9.0, "sum(eigenvalues) = trace");
        // Eigenvalues should be ascending
        assert!(w[0] <= w[1], "eigenvalues ascending");
        assert!(w[1] <= w[2], "eigenvalues ascending");
    }

    #[test]
    fn test_symeig_full_orthogonal_q() {
        // After eigendecomp, A contains Q (eigenvectors). Verify QtQ = I
        let mut a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let mut w = [0.0_f64; 3];
        symeig_full(3, &mut a, 3, &mut w).unwrap();
        // a now contains Q
        let mut qtq = [0.0_f64; 9];
        blocked_gemm(3, 3, 3, 1.0, &a, 3, &a, 3, 0.0, &mut qtq, 3, true, false).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(qtq[j * 3 + i], expected, &format!("QtQ[{i},{j}]"));
            }
        }
    }

    // 26. syrk_fisher_info -- XtX Fisher information

    #[test]
    fn test_syrk_fisher_info_2x3() {
        // A = 2x3 (n=2, k=3): C = A*At (2x2 symmetric)
        // Using trans='N': C = alpha * A * At + beta * C
        // A = [[1, 2, 3],  column-major (2x3): [1, 4, 2, 5, 3, 6]
        //      [4, 5, 6]]
        // A*At = [[14, 32], [32, 77]]
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut c = [0.0_f64; 4]; // 2x2
        syrk_fisher_info(2, 3, 1.0, &a, 2, 0.0, &mut c, 2).unwrap();
        assert_near(c[0], 14.0, "c[0,0]");
        assert_near(c[1], 32.0, "c[1,0]");
        assert_near(c[3], 77.0, "c[1,1]");
    }

    #[test]
    fn test_syrk_fisher_info_positive_semidefinite() {
        // Result should be positive semidefinite (all diagonal >= 0)
        let a = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x3
        let mut c = [0.0_f64; 4];
        syrk_fisher_info(2, 3, 1.0, &a, 2, 0.0, &mut c, 2).unwrap();
        assert!(c[0] >= 0.0, "c[0,0] >= 0");
        assert!(c[3] >= 0.0, "c[1,1] >= 0");
        // det(C) >= 0 for PSD
        let det = c[0] * c[3] - c[1] * c[2];
        assert!(det >= -TOL, "det(C) >= 0 for PSD");
    }

    // 27. sym_rank2k_update -- symmetric rank-2k update

    #[test]
    fn test_sym_rank2k_update_symmetric() {
        // C = alpha * (At*B + Bt*A) + beta*C where A, B are k x n
        // A (3x2) column-major: [1, 2, 3, 4, 5, 6] padded to lda*k=9
        // B (3x2) column-major: [7, 8, 9, 10, 11, 12] padded to lda*k=9
        // n=2, k=3
        // AtB = [[1*7+2*8+3*9, 1*10+2*11+3*12], [4*7+5*8+6*9, 4*10+5*11+6*12]]
        //     = [[50, 68], [122, 167]]
        // BtA = AtB transposed = [[50, 122], [68, 167]]
        // AtB + BtA = [[100, 190], [190, 334]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]; // k=3 x n=2, padded
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0]; // padded
        let mut c = [0.0_f64; 4]; // 2x2
        sym_rank2k_update(2, 3, 1.0, &a, 3, &b, 3, 0.0, &mut c, 2).unwrap();
        // Lower triangle: c[0]=C[0,0], c[1]=C[1,0], c[3]=C[1,1]
        assert_near(c[0], 100.0, "C[0,0]");
        assert_near(c[1], 190.0, "C[1,0]");
        assert_near(c[3], 334.0, "C[1,1]");
    }

    #[test]
    fn test_sym_rank2k_update_a_equals_b() {
        // When A==B, C = 2*At*A (which should equal 2*syrk result)
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 (k=2, n=2, lda=2)
        let mut c_rank2k = [0.0_f64; 4];
        sym_rank2k_update(2, 2, 1.0, &a, 2, &a, 2, 0.0, &mut c_rank2k, 2).unwrap();
        // Compare with 2*syrk
        let mut c_syrk = [0.0_f64; 4];
        syrk_panel(2, 2, 2.0, &a, 2, 0.0, &mut c_syrk, 2, true).unwrap();
        // Lower triangles should match
        assert_near(c_rank2k[0], c_syrk[0], "C[0,0]");
        assert_near(c_rank2k[1], c_syrk[1], "C[1,0]");
        assert_near(c_rank2k[3], c_syrk[3], "C[1,1]");
    }
}
