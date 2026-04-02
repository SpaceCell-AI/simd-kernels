# SIMD-Kernels

**High-performance compute kernels for Rust, built on `std::simd`.**

SIMD-Kernels provides vectorised arithmetic, statistics, scientific functions, and sorting over typed slices. Built on Rust's [`std::simd`](https://doc.rust-lang.org/std/simd/) portable SIMD, every kernel runs hardware-accelerated on x86, ARM, and WASM without a single platform-specific intrinsic. Null-aware by default, feature-gated for minimal compile footprint, and built for real-time and HPC workloads.

## Why SIMD-Kernels?

Writing SIMD by hand means maintaining separate code paths for SSE2, AVX2, AVX-512, and NEON. Most libraries either avoid SIMD entirely, rely on auto-vectorisation hints that may or may not fire, or use trait-object dispatch that defeats the optimiser at the call boundary.

SIMD-Kernels is built directly on [`std::simd`](https://doc.rust-lang.org/std/simd/), Rust's portable SIMD module. You write one kernel - the compiler lowers it to the best available ISA on each target. Every kernel operates on concrete typed slices (`&[f64]`, `&[i32]`) with null-mask support following Apache Arrow semantics. No `dyn`, no `Any`, no runtime downcasting.

## Quick Start

```rust
use simd_kernels::kernels::scientific::distributions::univariate::gamma::gamma_pdf_simd;

let x = arr_f64![0.5, 1.0, 2.0, 3.0, 4.0];
let shape = 2.0; // α - control distribution shape
let scale = 1.5; // β rate parameter

let result = gamma_pdf_simd(x, shape, scale, None, None)?;
// PDF values for Gamma(α=2, β=1.5) evaluated at each x
// Minarrow ensures 64-byte SIMD-alignment ensuring speed and consistency
```

## What's Included

### Core Kernels

| Module | Description |
|--------|-------------|
| `aggregate` | Sum, mean, variance, min/max, count distinct |
| `binary` | Bitwise operations |
| `comparison` | SIMD mask comparisons across all numeric types |
| `conditional` | Lane-parallel if-then-else |
| `logical` | Boolean logic with bitmap kernels |
| `sort` | SIMD radix sort |
| `unary` | Element-wise transforms |
| `vector` | Dot product, norms, weighted stats |
| `window` | Sliding window aggregations |

### Scientific Computing

| Module | Description |
|--------|-------------|
| `scientific/distributions` | 19 univariate families, 60+ functions - see below |
| `scientific/erf` | Error functions |
| `scientific/fft` | Radix-2/4/8 FFT pipelines with SIMD complex arithmetic |
| `scientific/matrix` | Dense matrix kernels |
| `scientific/scalar` | exp, ln, log10, gamma, and friends |
| `scientific/vector` | SIMD vector operations |
| `scientific/blas_lapack` | Optional BLAS/LAPACK bindings |

### Probability Distributions

19 univariate families, 60+ SIMD-accelerated kernels - PDF, CDF, and quantile functions - each validated against SciPy to high precision.

| Family | | | |
|--------|---|---|---|
| Normal | Beta | Gamma | Student's t |
| Exponential | Weibull | Cauchy | Logistic |
| Lognormal | Laplace | Chi-squared | Gumbel |
| Poisson | Binomial | Geometric | Negative Binomial |
| Hypergeometric | Uniform | Discrete Uniform | |

Each family has a scalar fallback path and a SIMD-vectorised path, selected automatically when the `simd` feature is enabled.

```rust
use simd_kernels::kernels::scientific::distributions::univariate::normal::*;

let x = &[-2.0, -1.0, 0.0, 1.0, 2.0];
let pdf = normal_pdf(x, 0.0, 1.0, None, None).unwrap();
let cdf = normal_cdf(x, 0.0, 1.0, None, None).unwrap();
```

SciPy's `scipy.stats` has been the gold standard for statistical distributions in Python for over a decade. SIMD-Kernels matches that family coverage and numerical rigour, running natively in Rust with SIMD vectorisation. Functions are tested against SciPy reference outputs across standard domains, tail regions, and known edge cases.

## Working with Arrays
[Minarrow](https://github.com/pbow/minarrow) is the recommended array library for use with SIMD-Kernels. Its `Vec64` and `FloatArray` types allocate on 64-byte aligned boundaries, which matches the alignment requirements of AVX-512 processors and ensures that SIMD-accelerated kernel paths are taken without fallback to scalar. Passing non-aligned slices from other sources is supported but will silently route to the scalar path, which may be unexpected in performance-sensitive workloads. If you are integrating with an existing Arrow pipeline, Minarrow offers "zero-copy" data movement out of the box. 

## Null-Mask Handling

Kernels support Apache Arrow-compatible null masks via [Minarrow](https://github.com/pbower/minarrow):

- Null masks are opt-in - omitting them routes directly to dense kernel paths
- Supplying `null_count = 0` skips mask checks identically
- Null propagation, masking, and early exits are SIMD-accelerated where possible

*SIMD-kernels builds on Minarrow, a focused Rust implementation of the core Arrow memory specification. Apache Arrow is a trademark of the Apache Software Foundation.*

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `simd` | SIMD acceleration via `std::simd` | ✓ |
| `probability_distributions` | PDFs, CDFs, quantiles | ✓ |
| `fourier_transforms` | FFT operations | ✓ |
| `universal_functions` | Scalar maths: exp, ln, sin, etc. | ✓ |
| `linear_algebra` | BLAS/LAPACK via OpenBLAS | |
| `simd_sort` | SIMD-accelerated radix sort for integers | |
| `fast_hash` | ahash for count distinct and categorical ops | |

Sub 2-second compile times with defaults.

```toml
[dependencies]
simd-kernels = { version = "0.2", features = ["linear_algebra"] }
```

## Numerical Accuracy

Distribution, special function, and scientific kernel functions are validated against SciPy reference outputs across standard domains, tail regions, and known difficult cases. Reference values are hardcoded from a validated x86_64 baseline and embedded directly in the test suite.

| Domain | Relative Error |
|--------|---------------|
| Core functions (`normal_pdf`, `gamma`, `erf`) | < 1e-15 |
| Distributions across standard mean ranges | < 1e-14 |
| Heavy-tail and extreme domains | < 1e-12 |
| Boundary cases where SciPy itself becomes unstable | < 1e-10 |

**This library is provided as-is, without warranties or guarantees of accuracy, correctness, or fitness for any purpose. Any reliance on it in critical, safety-related, or production systems is entirely at the user's own risk. Users must independently verify all outputs.**

## SIMD Configuration

`std::simd` is portable - no configuration required for correct vectorisation. The options below are for squeezing out extra performance or testing specific ISA widths.

### Compiling

```bash
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --features simd
```

### Overriding Lane Widths

```bash
# Format: "W8,W16,W32,W64"
SIMD_LANES_OVERRIDE="64,32,16,8" \
RUSTFLAGS="-C target-cpu=native" \
cargo +nightly build --features simd
```

### SIMD Widths by Architecture

| Feature | Register Width | f64 lanes | f32 lanes | i16 lanes |
|---------|---------------|-----------|-----------|-----------|
| SSE2 | 128-bit | 2 | 4 | 8 |
| AVX/AVX2 | 256-bit | 4 | 8 | 16 |
| AVX-512 | 512-bit | 8 | 16 | 32 |
| NEON | 128-bit | 2 | 4 | 8 |
| WASM SIMD128 | 128-bit | 2 | 4 | 8 |

Check ISA support with `lscpu | grep Flags` and look for `avx`, `avx2`, `avx512f`.

## Threading

SIMD gives you parallelism within a single thread. This library deliberately stops there - thread-level parallelism is use-case specific and the orchestration overhead of getting it wrong can dwarf the gains.

Pair SIMD-Kernels with a threading layer of your choice. [Rayon](https://github.com/rayon-rs/rayon) is the natural fit for batch workloads.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md). 

## Licence

Dual-licensed under [AGPL-3.0](LICENSE) and a [commercial licence](LICENSE-COMMERCIAL.md).

AGPL-3.0 is available for open-source use. If you are integrating SIMD-Kernels into a proprietary product or commercial service, a commercial licence is required. Contact [licensing@spacecell.com] to purchase a license.