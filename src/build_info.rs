//! Build metadata embedded in generated conversion reports.

/// Placeholder used when build-time provenance cannot be discovered.
const UNKNOWN: &str = "unknown";

/// Converter crate package version.
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the git commit captured when this binary was built.
#[inline]
#[must_use]
pub const fn git_commit() -> &'static str {
    match option_env!("GEMS_A10_CONVERTER_GIT_COMMIT") {
        Some(commit) => commit,
        None => UNKNOWN,
    }
}

/// Returns whether the source tree was dirty when this binary was built.
#[inline]
#[must_use]
pub const fn git_dirty() -> &'static str {
    match option_env!("GEMS_A10_CONVERTER_GIT_DIRTY") {
        Some(dirty) => dirty,
        None => UNKNOWN,
    }
}
