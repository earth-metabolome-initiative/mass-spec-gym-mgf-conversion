//! Build script for embedding converter provenance metadata.

use std::process::Command;

/// Compile-time environment variable containing the current git commit.
const GIT_COMMIT_ENV: &str = "GEMS_A10_CONVERTER_GIT_COMMIT";
/// Compile-time environment variable containing the current git dirty state.
const GIT_DIRTY_ENV: &str = "GEMS_A10_CONVERTER_GIT_DIRTY";

/// Captures optional git metadata for conversion reports.
fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    if let Some(commit) = git_output(["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env={GIT_COMMIT_ENV}={commit}");
    }
    if let Some(dirty) = git_dirty() {
        println!("cargo:rustc-env={GIT_DIRTY_ENV}={dirty}");
    }
}

/// Runs a git command and returns non-empty trimmed standard output.
fn git_output<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Returns whether the repository had uncommitted changes at build time.
fn git_dirty() -> Option<&'static str> {
    let output = Command::new("git")
        .args(["status", "--short"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    if output.stdout.is_empty() {
        Some("false")
    } else {
        Some("true")
    }
}
