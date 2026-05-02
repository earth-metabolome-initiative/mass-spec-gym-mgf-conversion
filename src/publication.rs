//! Zenodo metadata and upload integration.

use anyhow::Context;
use zenodo_rs::{
    AccessRight, Auth, Creator, DepositMetadataUpdate, PublishedRecord, UploadSpec, UploadType,
    ZenodoClient,
};

use crate::{Config, conversion::expected_artifact_paths};

/// Zenodo creator display name.
const ZENODO_CREATOR_NAME: &str = "Cappelletti, Luca";
/// Zenodo creator ORCID identifier.
const ZENODO_CREATOR_ORCID: &str = "0000-0002-1269-2038";
/// Target Zenodo community slug.
const ZENODO_COMMUNITY: &str = "earth-metabolome";

/// Publishes generated artifacts to Zenodo using `zenodo-rs`.
///
/// Publication is intentionally only called when `ZENODO_TOKEN` is present.
///
/// # Errors
///
/// Returns an error if metadata, upload specs, authentication, or publication
/// fails.
pub async fn publish_to_zenodo(config: &Config) -> anyhow::Result<PublishedRecord> {
    let metadata = publication_metadata()?;
    let files = upload_specs(config).context("failed to build Zenodo upload file list")?;
    let client = ZenodoClient::new(Auth::from_env()?)?;

    client
        .create_and_publish_dataset(&metadata, files)
        .await
        .context("Zenodo create-and-publish workflow failed")
}

/// Builds the fixed Zenodo metadata for the converted dataset.
fn publication_metadata() -> anyhow::Result<DepositMetadataUpdate> {
    Ok(DepositMetadataUpdate::builder()
        .title("GeMS-A10 converted to Mascot Generic Format")
        .upload_type(UploadType::Dataset)
        .description_html(
            "<p>MGF conversion of the GeMS-A10 unlabeled MS/MS spectral collection.</p>",
        )
        .creator(
            Creator::builder()
                .name(ZENODO_CREATOR_NAME)
                .orcid(ZENODO_CREATOR_ORCID)
                .build()?,
        )
        .community_identifier(ZENODO_COMMUNITY)
        .access_right(AccessRight::Open)
        .build()?)
}

/// Builds upload specifications for the generated document and metadata files.
fn upload_specs(config: &Config) -> anyhow::Result<Vec<UploadSpec>> {
    expected_artifact_paths(&config.output_dir, true)?
        .into_iter()
        .map(UploadSpec::from_path)
        .collect::<Result<Vec<_>, _>>()
        .context("failed to create path upload specs")
}

#[cfg(test)]
/// Tests for Zenodo publication metadata.
mod tests {
    use anyhow::ensure;

    use super::*;

    /// Confirms the publication metadata targets the requested creator and community.
    #[test]
    fn metadata_targets_luca_and_earth_metabolome() -> anyhow::Result<()> {
        let metadata = publication_metadata()?;
        ensure!(metadata.creators.len() == 1, "unexpected creator count");
        let creator = metadata.creators.first().context("missing creator")?;
        ensure!(
            creator.name == ZENODO_CREATOR_NAME,
            "unexpected creator name"
        );
        ensure!(
            creator.orcid.as_deref() == Some(ZENODO_CREATOR_ORCID),
            "unexpected creator ORCID"
        );
        ensure!(
            metadata.communities.len() == 1,
            "unexpected community count"
        );
        let community = metadata.communities.first().context("missing community")?;
        ensure!(
            community.identifier == ZENODO_COMMUNITY,
            "unexpected community"
        );
        Ok(())
    }
}
