//! Binary entrypoint for the configured GeMS-A10 conversion run.

use anyhow::Context;
use mass_spec_gym_mgf_conversion::{
    Config, ProgressReporter, convert_gems_a10_with_progress, expected_configured_artifact_paths,
    mgf_part_rows, publish_to_zenodo_with_progress, write_sha256sums_with_progress,
};

/// Runs conversion, checksum generation, and optional Zenodo publication.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let progress = ProgressReporter::new();
    let configuration = progress.spinner("loading .env and runtime configuration")?;
    dotenvy::dotenv().ok();

    let config = Config::from_env().context("failed to read conversion configuration")?;
    configuration.finish_with_message("configuration loaded");
    progress.println(format!("input: {}", config.input_hdf5.display()))?;
    progress.println(format!("output: {}", config.output_dir.display()))?;
    progress.println(format!("HDF5 read chunk size: {}", config.chunk_size))?;
    progress.println(format!("MGF part rows: {}", mgf_part_rows()))?;
    progress.println(format!(
        "maximum fragment peaks: {}",
        config.max_fragment_peaks
    ))?;

    if expected_configured_artifact_paths(&config, true).is_ok() {
        progress
            .println("complete converted artifacts found; skipping conversion and checksums")?;
    } else if expected_configured_artifact_paths(&config, false).is_ok() {
        progress.println("converted artifacts found; regenerating SHA256SUMS")?;
        let checksum_path = write_sha256sums_with_progress(&config.output_dir, &progress)
            .context("failed to write SHA256SUMS")?;
        progress.println(format!("checksums: {}", checksum_path.display()))?;
    } else {
        let report = convert_gems_a10_with_progress(&config, &progress)
            .context("GeMS-A10 conversion failed")?;
        let checksum_path = write_sha256sums_with_progress(&config.output_dir, &progress)
            .context("failed to write SHA256SUMS")?;
        progress.println(format!("checksums: {}", checksum_path.display()))?;
        progress.println(format!(
            "converted rows {}-{}: written={}, skipped={}",
            report.start_row,
            report.end_row.unwrap_or(report.start_row),
            report.spectra_written,
            report.spectra_skipped
        ))?;
    }

    if config.publish_to_zenodo {
        let record = Box::pin(publish_to_zenodo_with_progress(&config, &progress))
            .await
            .context("Zenodo publication failed")?;
        progress.println(format!("published Zenodo record: {}", record.record.id))?;
    } else {
        progress.println("Zenodo publication skipped because ZENODO_TOKEN is not set.")?;
    }

    Ok(())
}
