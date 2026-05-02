//! Zenodo metadata and upload integration.

use std::{fs, io::ErrorKind, time::Duration};

use anyhow::{Context, bail};
use chrono::{NaiveDate, Utc};
use indicatif::ProgressBar;
use tokio::time::sleep;
use zenodo_rs::{
    AccessRight, Auth, Creator, DepositMetadataUpdate, Deposition, DepositionId, FileReplacePolicy,
    PublishedRecord, Record, RelatedIdentifier, TransferProgress, UploadSpec, UploadType,
    ZenodoClient, ZenodoError,
};

use crate::{Config, ProgressReporter, conversion::expected_artifact_paths};

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
/// Number of times to poll Zenodo after requesting publication.
const ZENODO_PUBLICATION_POLLS: usize = 60;
/// Delay between publication-state polls.
const ZENODO_PUBLICATION_POLL_DELAY: Duration = Duration::from_secs(5);
/// Number of attempts for the full Zenodo file reconciliation step.
const ZENODO_UPLOAD_ATTEMPTS: usize = 5;
/// Delay between retryable Zenodo upload failures.
const ZENODO_UPLOAD_RETRY_DELAY: Duration = Duration::from_secs(30);
/// Local state file containing the Zenodo deposition id for resumable uploads.
const ZENODO_DEPOSITION_ID_FILE: &str = "zenodo_deposition_id.txt";
/// Zenodo description for the converted dataset.
const ZENODO_DESCRIPTION_HTML: &str = "\
<p>This record contains a Mascot Generic Format (MGF) conversion of the GeMS-A10 unlabeled MS/MS spectral collection.</p>
<p>GeMS-A10 is distributed through the DreaMS/GeMS Hugging Face dataset and is the MassSpecGym auxiliary unlabeled spectral collection, not the smaller labeled MassSpecGym TSV dataset.</p>
<p>The conversion writes compressed MGF part documents, preserves GeMS row metadata, removes invalid spectra and padded peaks, keeps spectra with at least two valid fragment peaks, caps spectra to the 100 highest-intensity fragment peaks, and removes SPLASH duplicates. The record also includes a manifest, conversion report, duplicate report, dataset README, and SHA256 checksums.</p>";

/// `indicatif` progress bar adapter for `zenodo-rs` transfer hooks.
#[derive(Debug, Clone)]
struct ZenodoUploadProgress {
    /// Aggregate upload progress bar.
    bar: ProgressBar,
}

impl TransferProgress for ZenodoUploadProgress {
    fn begin(&self, total_bytes: Option<u64>) {
        self.bar.set_position(0);
        if let Some(total_bytes) = total_bytes {
            self.bar.set_length(total_bytes);
        }
    }

    fn advance(&self, delta: u64) {
        self.bar.inc(delta);
    }

    fn finish(&self) {
        self.bar.finish_with_message("Zenodo upload complete");
    }
}

/// Publishes generated artifacts to Zenodo using `zenodo-rs`.
///
/// Publication is intentionally only called when `ZENODO_TOKEN` is present.
///
/// # Errors
///
/// Returns an error if metadata, upload specs, authentication, or publication
/// fails.
pub async fn publish_to_zenodo(config: &Config) -> anyhow::Result<PublishedRecord> {
    publish_to_zenodo_with_progress(config, &ProgressReporter::hidden()).await
}

/// Publishes generated artifacts to Zenodo while reporting upload progress.
///
/// Publication is intentionally only called when `ZENODO_TOKEN` is present.
///
/// # Errors
///
/// Returns an error if metadata, upload specs, authentication, upload, or
/// publication fails.
pub async fn publish_to_zenodo_with_progress(
    config: &Config,
    progress: &ProgressReporter,
) -> anyhow::Result<PublishedRecord> {
    let metadata = publication_metadata()?;
    let client = ZenodoClient::new(Auth::from_env()?)?;
    let draft = load_or_create_draft(config, &client, progress).await?;
    if draft.is_published() {
        return published_record_from_deposition(&client, draft).await;
    }

    let metadata_step = progress.spinner(format!("updating Zenodo metadata for {}", draft.id))?;
    let draft = client
        .update_metadata(draft.id, &metadata)
        .await
        .context("failed to update Zenodo deposition metadata")?;
    metadata_step.finish_with_message(format!("Zenodo metadata updated: {}", draft.id));

    upload_files_with_retries(config, &client, &draft, progress).await?;

    let publication = progress.spinner(format!("publishing Zenodo draft {}", draft.id))?;
    let published = publish_or_recover(&client, &draft, progress).await?;
    let published = published_record_from_deposition(&client, published).await?;
    publication.finish_with_message(format!("published Zenodo record: {}", published.record.id));

    Ok(published)
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

/// Loads a persisted Zenodo draft id or creates a new draft.
async fn load_or_create_draft(
    config: &Config,
    client: &ZenodoClient,
    progress: &ProgressReporter,
) -> anyhow::Result<Deposition> {
    if let Some(id) = read_deposition_id(config)? {
        let draft_step = progress.spinner(format!("loading Zenodo deposition {id}"))?;
        let draft = client
            .get_deposition(id)
            .await
            .with_context(|| format!("failed to load Zenodo deposition {id}"))?;
        draft_step.finish_with_message(format!("Zenodo deposition loaded: {}", draft.id));
        progress.println(format!("Zenodo deposition id: {}", draft.id))?;
        return Ok(draft);
    }

    let draft_step = progress.spinner("creating Zenodo deposition draft")?;
    let draft = client
        .create_deposition()
        .await
        .context("failed to create Zenodo deposition draft")?;
    write_deposition_id(config, draft.id)?;
    draft_step.finish_with_message(format!("Zenodo draft created: {}", draft.id));
    progress.println(format!("Zenodo deposition id: {}", draft.id))?;
    Ok(draft)
}

/// Reads the persisted Zenodo deposition id, when present.
fn read_deposition_id(config: &Config) -> anyhow::Result<Option<DepositionId>> {
    let path = config.output_dir.join(ZENODO_DEPOSITION_ID_FILE);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let id = text
        .trim()
        .parse::<u64>()
        .with_context(|| format!("failed to parse Zenodo deposition id in {}", path.display()))?;
    Ok(Some(DepositionId::from(id)))
}

/// Persists the Zenodo deposition id for resumable uploads.
fn write_deposition_id(config: &Config, id: DepositionId) -> anyhow::Result<()> {
    let path = config.output_dir.join(ZENODO_DEPOSITION_ID_FILE);
    fs::write(&path, format!("{id}\n"))
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Uploads all files, retrying transient Zenodo upload failures.
async fn upload_files_with_retries(
    config: &Config,
    client: &ZenodoClient,
    draft: &Deposition,
    progress: &ProgressReporter,
) -> anyhow::Result<()> {
    for attempt in 1..=ZENODO_UPLOAD_ATTEMPTS {
        match upload_files_once(config, client, draft, progress, attempt).await {
            Ok(()) => return Ok(()),
            Err(error) if attempt < ZENODO_UPLOAD_ATTEMPTS && is_retryable_anyhow(&error) => {
                progress.println(format!(
                    "Zenodo upload attempt {attempt} failed with a retryable error; retrying in {}s",
                    ZENODO_UPLOAD_RETRY_DELAY.as_secs()
                ))?;
                sleep(ZENODO_UPLOAD_RETRY_DELAY).await;
            }
            Err(error) => return Err(error).context("failed to upload Zenodo files"),
        }
    }

    bail!("Zenodo upload attempts exhausted")
}

/// Runs one full file reconciliation attempt.
async fn upload_files_once(
    config: &Config,
    client: &ZenodoClient,
    draft: &Deposition,
    progress: &ProgressReporter,
    attempt: usize,
) -> anyhow::Result<()> {
    let files = upload_specs(config).context("failed to build Zenodo upload file list")?;
    let upload_count = files.len();
    let upload_bytes = files
        .iter()
        .try_fold(0u64, |total, spec| {
            Ok::<u64, std::io::Error>(total + spec.content_length()?)
        })
        .context("failed to compute Zenodo upload size")?;
    let upload_bar = progress.byte_bar(
        upload_bytes,
        format!(
            "uploading {upload_count} files to Zenodo draft {} | attempt {attempt}/{}",
            draft.id, ZENODO_UPLOAD_ATTEMPTS
        ),
    )?;
    let result = client
        .reconcile_files_with_progress(
            draft,
            FileReplacePolicy::ReplaceAll,
            files,
            ZenodoUploadProgress {
                bar: upload_bar.clone(),
            },
        )
        .await;

    match result {
        Ok(uploads) => {
            progress.println(format!("Zenodo uploaded {} files", uploads.len()))?;
            Ok(())
        }
        Err(error) => {
            upload_bar.abandon_with_message(format!("Zenodo upload attempt {attempt} failed"));
            Err(error).context("Zenodo file reconciliation failed")
        }
    }
}

/// Publishes a draft and recovers if Zenodo reports a transient error after accepting the request.
async fn publish_or_recover(
    client: &ZenodoClient,
    draft: &Deposition,
    progress: &ProgressReporter,
) -> anyhow::Result<Deposition> {
    match client.publish(draft.id).await {
        Ok(deposition) => Ok(deposition),
        Err(error) if retryable_zenodo_error(&error) => {
            progress.println(format!(
                "Zenodo publish returned a retryable error; checking draft {}",
                draft.id
            ))?;
            wait_for_published_deposition(client, draft.clone())
                .await
                .with_context(|| format!("Zenodo publish returned a retryable error: {error}"))
        }
        Err(error) => Err(error).context("failed to publish Zenodo deposition"),
    }
}

/// Polls Zenodo until a deposition is published and has a record identifier.
async fn wait_for_published_deposition(
    client: &ZenodoClient,
    mut deposition: Deposition,
) -> anyhow::Result<Deposition> {
    for _ in 0..ZENODO_PUBLICATION_POLLS {
        if deposition.is_published() && deposition.record_id.is_some() {
            return Ok(deposition);
        }

        sleep(ZENODO_PUBLICATION_POLL_DELAY).await;
        match client.get_deposition(deposition.id).await {
            Ok(refreshed) => deposition = refreshed,
            Err(error) if retryable_zenodo_error(&error) => {}
            Err(error) => return Err(error).context("failed to refresh Zenodo deposition"),
        }
    }

    bail!(
        "timed out waiting for Zenodo deposition {} to publish",
        deposition.id
    )
}

/// Resolves a published deposition into the final published record payload.
async fn published_record_from_deposition(
    client: &ZenodoClient,
    deposition: Deposition,
) -> anyhow::Result<PublishedRecord> {
    let published = wait_for_published_deposition(client, deposition).await?;
    let record_id = published
        .record_id
        .context("published Zenodo deposition is missing record_id")?;
    let record = get_record_with_retries(client, record_id)
        .await
        .context("failed to fetch published Zenodo record")?;
    Ok(PublishedRecord {
        deposition: published,
        record,
    })
}

/// Fetches the published record, retrying transient Zenodo failures.
async fn get_record_with_retries(
    client: &ZenodoClient,
    record_id: zenodo_rs::RecordId,
) -> anyhow::Result<Record> {
    for _ in 0..ZENODO_PUBLICATION_POLLS {
        match client.get_record(record_id).await {
            Ok(record) => return Ok(record),
            Err(error) if retryable_zenodo_error(&error) => {
                sleep(ZENODO_PUBLICATION_POLL_DELAY).await;
            }
            Err(error) => return Err(error).context("failed to fetch Zenodo record"),
        }
    }

    bail!("timed out waiting for Zenodo record {record_id}")
}

/// Returns whether a Zenodo error can plausibly be a transient server-side failure.
fn retryable_zenodo_error(error: &ZenodoError) -> bool {
    match error {
        ZenodoError::Http { status, .. } => {
            matches!(status.as_u16(), 409 | 429) || status.is_server_error()
        }
        ZenodoError::Transport(_) => true,
        ZenodoError::Json(_)
        | ZenodoError::Io(_)
        | ZenodoError::Url(_)
        | ZenodoError::EnvVar { .. }
        | ZenodoError::InvalidState(_)
        | ZenodoError::MissingLink(_)
        | ZenodoError::MissingFile { .. }
        | ZenodoError::DuplicateUploadFilename { .. }
        | ZenodoError::ConflictingDraftFile { .. }
        | ZenodoError::UnsupportedSelector(_)
        | ZenodoError::ChecksumMismatch { .. }
        | ZenodoError::Timeout(_) => false,
    }
}

/// Returns whether an error chain contains a retryable Zenodo failure.
fn is_retryable_anyhow(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|source| source.downcast_ref::<ZenodoError>())
        .is_some_and(retryable_zenodo_error)
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
