//! Environment-backed runtime configuration.

use std::env;
use std::path::PathBuf;

/// Fixed source HDF5 path used by the full conversion run.
const INPUT_HDF5: &str = "data/data/GeMS_A/GeMS_A10.hdf5";
/// Fixed output directory for the converted MGF document and metadata.
const OUTPUT_DIR: &str = "converted/GeMS_A10";
/// Fixed HDF5 row chunk size for the conversion.
const CHUNK_SIZE: usize = 250_000;
/// Default maximum number of fragment peaks retained per spectrum.
const DEFAULT_MAX_FRAGMENT_PEAKS: usize = 60;

/// Runtime configuration for the deterministic full conversion run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Config {
    /// Source GeMS-A10 HDF5 file.
    pub input_hdf5: PathBuf,
    /// Output directory for the MGF document and metadata.
    pub output_dir: PathBuf,
    /// Number of HDF5 rows read per HDF5 chunk.
    pub chunk_size: usize,
    /// Maximum number of highest-intensity fragment peaks retained per spectrum.
    pub max_fragment_peaks: usize,
    /// First zero-based HDF5 row to visit.
    pub start_row: usize,
    /// Optional row limit for sample conversion.
    pub limit: Option<usize>,
    /// Parse the written MGF document with `mascot-rs`.
    pub validate_output: bool,
    /// Whether production Zenodo publication should run after validation.
    pub publish_to_zenodo: bool,
}

impl Config {
    /// Builds the fixed runtime configuration.
    ///
    /// # Errors
    ///
    /// This currently cannot fail, but returns `Result` to keep the caller's
    /// setup path explicit if a future fixed preflight check is added.
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            input_hdf5: PathBuf::from(INPUT_HDF5),
            output_dir: PathBuf::from(OUTPUT_DIR),
            chunk_size: CHUNK_SIZE,
            max_fragment_peaks: DEFAULT_MAX_FRAGMENT_PEAKS,
            start_row: 0,
            limit: None,
            validate_output: true,
            publish_to_zenodo: zenodo_token_present(),
        })
    }
}

/// Returns whether a production Zenodo token is present and non-empty.
fn zenodo_token_present() -> bool {
    env::var_os("ZENODO_TOKEN").is_some_and(|token| !token.is_empty())
}
