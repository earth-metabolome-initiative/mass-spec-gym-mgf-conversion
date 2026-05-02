//! Zenodo metadata and upload integration.

use anyhow::Context;
use chrono::{NaiveDate, Utc};
use zenodo_rs::{
    AccessRight, Auth, Creator, DepositMetadataUpdate, PublishedRecord, RelatedIdentifier,
    UploadSpec, UploadType, ZenodoClient,
};

use crate::{Config, conversion::expected_artifact_paths};

/// Zenodo creator display name.
const ZENODO_CREATOR_NAME: &str = "Cappelletti, Luca";
/// Zenodo creator ORCID identifier.
const ZENODO_CREATOR_ORCID: &str = "0000-0002-1269-2038";
/// Zenodo creator affiliation.
const ZENODO_CREATOR_AFFILIATION: &str = "University of Fribourg";
/// Target Zenodo community slug.
const ZENODO_COMMUNITY: &str = "earth-metabolome";
/// Zenodo license identifier for the converted dataset.
const ZENODO_LICENSE: &str = "mit";
/// Source Hugging Face dataset URL.
const SOURCE_DATASET_URL: &str = "https://huggingface.co/datasets/roman-bushuiev/GeMS";
/// Source GeMS-A10 HDF5 file URL.
const SOURCE_FILE_URL: &str =
    "https://huggingface.co/datasets/roman-bushuiev/GeMS/blob/main/data/GeMS_A/GeMS_A10.hdf5";
/// `MassSpecGym` repository URL.
const MASS_SPEC_GYM_REPOSITORY_URL: &str = "https://github.com/pluskal-lab/MassSpecGym";
/// Conversion crate repository URL.
const CONVERTER_REPOSITORY_URL: &str =
    "https://github.com/earth-metabolome-initiative/mass-spec-gym-mgf-conversion";
/// `DreaMS` paper DOI.
const DREAMS_PAPER_DOI: &str = "10.1038/s41587-025-02663-3";
/// Zenodo description for the converted dataset.
const ZENODO_DESCRIPTION_HTML: &str = "\
<p>This record contains a Mascot Generic Format (MGF) conversion of the GeMS-A10 unlabeled MS/MS spectral collection.</p>
<p>GeMS-A10 is distributed through the DreaMS/GeMS Hugging Face dataset and is the MassSpecGym auxiliary unlabeled spectral collection, not the smaller labeled MassSpecGym TSV dataset.</p>
<p>The conversion writes one compressed MGF document, preserves GeMS row metadata, removes invalid spectra and padded peaks, keeps spectra with at least two valid fragment peaks, caps spectra to the 100 highest-intensity fragment peaks, and removes SPLASH duplicates. The record also includes a manifest, conversion report, duplicate report, dataset README, and SHA256 checksums.</p>";

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
    publication_metadata_for_date(Utc::now().date_naive())
}

/// Builds the fixed Zenodo metadata for a specific publication date.
fn publication_metadata_for_date(
    publication_date: NaiveDate,
) -> anyhow::Result<DepositMetadataUpdate> {
    Ok(DepositMetadataUpdate::builder()
        .title("GeMS-A10 converted to Mascot Generic Format")
        .upload_type(UploadType::Dataset)
        .publication_date(publication_date)
        .description_html(ZENODO_DESCRIPTION_HTML)
        .creator(
            Creator::builder()
                .name(ZENODO_CREATOR_NAME)
                .affiliation(ZENODO_CREATOR_AFFILIATION)
                .orcid(ZENODO_CREATOR_ORCID)
                .build()?,
        )
        .community_identifier(ZENODO_COMMUNITY)
        .access_right(AccessRight::Open)
        .license(ZENODO_LICENSE)
        .keywords(publication_keywords())
        .related_identifiers(publication_related_identifiers()?)
        .build()?)
}

/// Builds the Zenodo keyword list for discovery.
fn publication_keywords() -> Vec<String> {
    [
        "mass spectrometry",
        "MS/MS",
        "MGF",
        "Mascot Generic Format",
        "GeMS-A10",
        "GeMS",
        "MassSpecGym",
        "DreaMS",
        "SPLASH",
        "spectral library",
        "unlabeled spectra",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Builds related identifiers for source data, software, and literature.
fn publication_related_identifiers() -> anyhow::Result<Vec<RelatedIdentifier>> {
    Ok(vec![
        RelatedIdentifier::builder()
            .identifier(SOURCE_DATASET_URL)
            .relation("isDerivedFrom")
            .scheme("url")
            .resource_type("dataset")
            .build()?,
        RelatedIdentifier::builder()
            .identifier(SOURCE_FILE_URL)
            .relation("isDerivedFrom")
            .scheme("url")
            .resource_type("dataset")
            .build()?,
        RelatedIdentifier::builder()
            .identifier(MASS_SPEC_GYM_REPOSITORY_URL)
            .relation("references")
            .scheme("url")
            .resource_type("software")
            .build()?,
        RelatedIdentifier::builder()
            .identifier(CONVERTER_REPOSITORY_URL)
            .relation("isCompiledBy")
            .scheme("url")
            .resource_type("software")
            .build()?,
        RelatedIdentifier::builder()
            .identifier(DREAMS_PAPER_DOI)
            .relation("cites")
            .scheme("doi")
            .resource_type("publication-article")
            .build()?,
    ])
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
        let publication_date = chrono::NaiveDate::from_ymd_opt(2026, 5, 2)
            .context("failed to build test publication date")?;
        let metadata = publication_metadata_for_date(publication_date)?;
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
            creator.affiliation.as_deref() == Some(ZENODO_CREATOR_AFFILIATION),
            "unexpected creator affiliation"
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
        ensure!(
            metadata.publication_date == Some(publication_date),
            "unexpected publication date"
        );
        ensure!(
            metadata.license.as_deref() == Some(ZENODO_LICENSE),
            "unexpected license"
        );
        ensure!(
            metadata.description_html.contains("SPLASH"),
            "description should describe deduplication"
        );
        ensure!(
            metadata.keywords == publication_keywords(),
            "unexpected keywords"
        );
        ensure!(
            metadata.related_identifiers.len() == 5,
            "unexpected related identifier count"
        );
        ensure!(
            metadata.related_identifiers.iter().any(|identifier| {
                identifier.identifier == SOURCE_DATASET_URL
                    && identifier.relation == "isDerivedFrom"
            }),
            "missing source dataset related identifier"
        );
        ensure!(
            metadata.related_identifiers.iter().any(|identifier| {
                identifier.identifier == DREAMS_PAPER_DOI && identifier.relation == "cites"
            }),
            "missing DreaMS paper related identifier"
        );
        Ok(())
    }
}
