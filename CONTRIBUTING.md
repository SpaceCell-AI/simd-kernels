# Contributing to SIMD-Kernels

Thank you for your interest in contributing to SIMD-Kernels! We welcome contributions from the community and appreciate your help!

## Getting Started

### Prerequisites

- Rust 1.89.0-nightly or later
- Git
- Familiarity with Apache Arrow and low-level memory concepts (helpful but not required)

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/simd-kernels.git
   cd simd_kernels
   ```
3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/originaluser/simd-kernels.git
   ```
4. Install dependencies and run tests:
   ```bash
   cargo test
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, following the coding standards below
3. Add or update tests as appropriate
4. Run the test suite:
   ```bash
   cargo test --all-features
   ```
or, for exhaustive feature-flag checks (requires cargo install cargo-all-features):
   ```base
   cargo test-all-features
   ```
The second form is recommended for changes that may affect multiple features.
It tests combinations of up to two feature flags and takes about five minutes.

5. Run clippy for linting:
   ```bash
   cargo clippy --all-features -- -D warnings
   ```
6. Format your code:
   ```bash
   cargo fmt
   ```

### Commit Messages

Please use clear, descriptive commit messages following conventional commit format:

- `feat: add new array type support`
- `fix: resolve memory alignment issue in Vec64`
- `docs: update API documentation for Table`
- `perf: optimise SIMD operations for integer arrays`
- `test: add benchmarks for categorical arrays`

## Coding Standards

### Code Style

- Follow standard Rust formatting (cargo fmt)
- Use meaningful variable and function names
- Keep functions focused and reasonably sized
- Avoid traits and superfluous abstractions unless they genuinely add value - this is particularly important as LLMs will often suggest these unnecessarily
- Ensure modules, structs and functions contain exactly what they say on the tin
- Brief and informative comments are great. Well-named objects are even better!
- Single-responsibility principle - each function should have one clear purpose
- Avoid code duplication through helper functions and macros where appropriate
- At enum match-arms - it is often preferable for maintability to call into dedicated functions rather than inlining logic per type. 
- Avoid overengineering if it is not adding genuine value to the codebase

### Testing

- Write unit tests for all new functionality
- Run Scipy or Numpy in Python, output the hardcoded results, and test against
that in the repository. 
- Include edge cases and error conditions
- Use property-based testing where appropriate
- Benchmark performance-critical code changes
- Test with both default and all features enabled

**Important** Please do not ever trust figures from an AI model
that says it has outputted Scipy or Numpy results. These will almost always be wrong!
Instead, find the Python function that outputs the required results, then use that.
Similarly, they are great at 'Fake SIMD' - promising you that it's SIMD. It's not.
Instead, probably it has copied data into SIMD buffers, run scalar operations over it,
then claimed it is 'faster'.

### Error Handling

- Use `Result<T, E>` for fallible operations
- Create specific error types rather than using strings
- Provide helpful error messages
- Document error conditions in function documentation

## Types of Contributions

### Priority Areas

We particularly welcome contributions in these areas:

* Kernel Coverage - Add kernels for missing primitives
* Numerical Accuracy - Validate against SciPy, write regression tests
* SIMD Optimisation - Improve SIMD coverage, improve speed

### Breaking changes

**Please do not introduce breaking changes**. This is particularly important given
today's developer community reliance on AI models, where users impacted by seemingly small changes means the AI gets it wrong for the next 2+ years until the next major release.

### Bug Fixes

When reporting bugs:
- Use the GitHub issue template
- Include minimal reproduction cases
- Specify Rust version and target platform
- Include relevant feature flags

When fixing bugs:
- Add regression tests
- Update documentation if the fix changes behaviour
- Reference the issue number in your commit message

## Code Review Process

### Pull Request Guidelines

1. **Before submitting:**
   - Ensure all tests pass
   - Update documentation
   - Add changelog entry if applicable
   - Rebase on latest main branch

2. **Pull request description should include:**
   - Clear description of changes
   - Motivation for the changes
   - Testing performed
   - Breaking changes (if any)
   - Related issue numbers

3. **Review process:**
   - All PRs currently require approval from Peter Bower.
   - PR's will be reviewed within 7 days (usually sooner).
   - Address feedback promptly
   - Maintain discussion in PR comments

4. **Licensing**
    - Please ensure no code is copied or pulled from other repositories,
    including MIT-licensed ones. This is to help ensure the codebase
    remains license-compliant, as even Apache-Licensed projects can
    experience legal challenges.

### Review Criteria

Reviewers will evaluate:

- **Correctness**: Does the code work as intended?
- **Performance**: Are there performance implications?
- **API Design**: Is the API intuitive and consistent?
- **Safety**: Does the code follow Rust safety principles?
- **Testing**: Are tests adequate and comprehensive?
- **Documentation**: Is the code well-documented?

### Maintainer status
Regular high-quality contributions is likely to result in you being
granted maintainer status to also approve PR's, and drive the crate's direction.

## Performance Considerations

### SIMD Optimisation

- Ensure algorithms work with aligned data
- Test on multiple CPU architectures when possible
- Provide fallback implementations for unsupported features
- Document SIMD requirements clearly

## Feature Flags

When adding new features:

- Use feature flags for optional functionality
- Document feature dependencies clearly
- Ensure core functionality works without optional features
- Update CI to test relevant feature combinations

## Release Process

### Versioning

We follow Semantic Versioning (SemVer):
- Major version: Breaking changes
- Minor version: New features, backwards compatible
- Patch version: Bug fixes, backwards compatible

### Changelog

Update `CHANGELOG.md` with:
- New features
- Bug fixes
- Breaking changes
- Performance improvements
- Deprecations

## Community Guidelines

### Code of Conduct

Please:

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Assume good intentions

### Communication

- Use GitHub issues for bug reports and feature requests
- Join discussions in pull request comments
- Ask questions if you're unsure about anything
- Share knowledge and help others

See `CODE_OF_CONDUCT.md`.

## Getting Help

If you need assistance:

1. Check existing documentation and issues
2. Ask questions in GitHub discussions
3. Reach out to maintainers for guidance
4. Join community channels (if available)

## Licensing
This project is dual-licensed under AGPL-3.0 and a commercial licence. By contributing, you agree that your contributions are subject to the Contributor Licence Agreement, which assigns copyright to SpaceCell Enterprises Ltd and permits distribution under both licences. Do not submit contributions that you are not authorised to assign under those terms.

## Recognition

Contributors will be recognised in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Project documentation where appropriate

Thank you for contributing to SIMD-Kernels! Your efforts help make high-performance data processing more accessible to the Rust community.
