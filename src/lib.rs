// Copyright (c) 2025 SpaceCell Enterprises
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

#![feature(portable_simd)]

// Link OpenBLAS when linear_algebra feature is enabled.
// This forces the linker to include the OpenBLAS symbols.
#[cfg(feature = "linear_algebra")]
extern crate openblas_src;

// compile with RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features portable_simd

pub mod operators;

pub mod kernels {
    pub mod aggregate;
    pub mod binary;
    pub mod comparison;
    pub mod conditional;
    pub mod logical;
    pub mod sort;
    pub mod unary;
    pub mod window;
    pub mod scientific {
        #[cfg(feature = "linear_algebra")]
        pub mod blas_lapack;
        #[cfg(feature = "probability_distributions")]
        pub mod distributions;
        #[cfg(feature = "probability_distributions")]
        pub mod erf;
        #[cfg(feature = "fourier_transforms")]
        pub mod fft;
        #[cfg(feature = "linear_algebra")]
        pub mod matrix;
        #[cfg(feature = "universal_functions")]
        pub mod scalar;
        pub mod vector;
    }
}

pub mod traits {
    pub mod dense_iter;
    pub mod to_bits;
}

pub mod utils;

// The bitmask, arithmetic and string kernels are contained in the upstream `Minarrow` crate,
// and are available in the namespace.
pub mod minarrow_kernels {
    pub use minarrow::kernels::arithmetic;
    pub use minarrow::kernels::bitmask;
}