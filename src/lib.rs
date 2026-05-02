//! GeMS-A10 HDF5 to MGF conversion pipeline.

/// Runtime configuration loaded from environment variables.
mod config;
/// HDF5 to MGF conversion logic.
mod conversion;
/// Terminal progress reporting.
mod progress;
/// Zenodo publication support.
mod publication;

pub use config::Config;
pub use conversion::{
    ConversionReport, ManifestRow, convert_gems_a10, convert_gems_a10_with_progress,
    finite_positive, write_dataset_readme, write_sha256sums, write_sha256sums_with_progress,
};
pub use progress::ProgressReporter;
pub use publication::publish_to_zenodo;
