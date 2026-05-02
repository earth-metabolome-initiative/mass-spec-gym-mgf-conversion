//! Binary entrypoint for the configured GeMS-A10 conversion run.

use anyhow::Context;
use mass_spec_gym_mgf_conversion::{
    Config, ProgressReporter, convert_gems_a10_with_progress, publish_to_zenodo,
    write_sha256sums_with_progress,
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
    progress.println(format!("chunk size: {}", config.chunk_size))?;

    let report =
        convert_gems_a10_with_progress(&config, &progress).context("GeMS-A10 conversion failed")?;
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

    if config.publish_to_zenodo {
        let publication = progress.spinner("publishing artifacts to Zenodo")?;
        let record = Box::pin(publish_to_zenodo(&config))
            .await
            .context("Zenodo publication failed")?;
        publication.finish_with_message(format!("published Zenodo record: {}", record.record.id));
    } else {
        progress.println("Zenodo publication skipped because ZENODO_TOKEN is not set.")?;
    }

    Ok(())
}
