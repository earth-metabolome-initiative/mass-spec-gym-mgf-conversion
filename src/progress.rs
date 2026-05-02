//! Terminal progress reporting with `indicatif`.

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Shared progress-bar factory for the conversion pipeline.
#[derive(Debug, Clone)]
pub struct ProgressReporter {
    /// Underlying multi-progress renderer.
    multi: MultiProgress,
}

impl ProgressReporter {
    /// Creates a progress reporter that renders to standard error.
    #[must_use]
    pub fn new() -> Self {
        Self::with_draw_target(ProgressDrawTarget::stderr_with_hz(4))
    }

    /// Creates a hidden progress reporter for tests and non-interactive calls.
    #[must_use]
    pub fn hidden() -> Self {
        Self::with_draw_target(ProgressDrawTarget::hidden())
    }

    /// Creates a progress reporter with the provided draw target.
    #[must_use]
    fn with_draw_target(draw_target: ProgressDrawTarget) -> Self {
        let multi = MultiProgress::with_draw_target(draw_target);
        Self { multi }
    }

    /// Prints a stable status line above active progress bars.
    ///
    /// # Errors
    ///
    /// Returns an error if the progress renderer cannot write the line.
    pub fn println<S: AsRef<str>>(&self, message: S) -> Result<()> {
        self.multi
            .println(message.as_ref())
            .context("failed to write progress message")
    }

    /// Creates a spinner for a pipeline step without known length.
    ///
    /// # Errors
    ///
    /// Returns an error if the spinner template is invalid.
    pub fn spinner<S: Into<String>>(&self, message: S) -> Result<ProgressBar> {
        let bar = self.multi.add(ProgressBar::new_spinner());
        bar.set_style(spinner_style()?);
        bar.set_message(message.into());
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        Ok(bar)
    }

    /// Creates a row-count progress bar.
    ///
    /// # Errors
    ///
    /// Returns an error if the row count cannot fit `u64` or the bar template
    /// is invalid.
    pub(crate) fn row_bar(&self, rows: usize, message: impl Into<String>) -> Result<ProgressBar> {
        let rows = u64::try_from(rows).context("row count does not fit u64")?;
        let bar = self.multi.add(ProgressBar::new(rows));
        bar.set_style(row_style()?);
        bar.set_message(message.into());
        Ok(bar)
    }

    /// Creates a byte-count progress bar.
    ///
    /// # Errors
    ///
    /// Returns an error if the bar template is invalid.
    pub(crate) fn byte_bar(&self, bytes: u64, message: impl Into<String>) -> Result<ProgressBar> {
        let bar = self.multi.add(ProgressBar::new(bytes));
        bar.set_style(byte_style()?);
        bar.set_message(message.into());
        Ok(bar)
    }
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the spinner style used for metadata and network steps.
fn spinner_style() -> Result<ProgressStyle> {
    ProgressStyle::with_template("{spinner:.green} {msg} [{elapsed_precise}]")
        .context("invalid indicatif spinner template")
}

/// Builds the row progress style used during HDF5 conversion and MGF validation.
fn row_style() -> Result<ProgressStyle> {
    ProgressStyle::with_template(
        "{wide_bar:.cyan/blue} {pos}/{len} rows {per_sec} eta {eta} | {msg}",
    )
    .context("invalid indicatif row template")
}

/// Builds the byte progress style used during checksum generation.
fn byte_style() -> Result<ProgressStyle> {
    ProgressStyle::with_template(
        "{wide_bar:.cyan/blue} {bytes}/{total_bytes} {bytes_per_sec} eta {eta} | {msg}",
    )
    .context("invalid indicatif byte template")
}
