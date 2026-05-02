//! Conversion from DreaMS/GeMS HDF5 tensors to Mascot Generic Format.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use csv::Writer;
use hdf5::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{Dataset, Dataspace, Datatype, File as H5File, Selection};
use hdf5_sys::h5d::H5Dread;
use hdf5_sys::h5p::H5P_DEFAULT;
use indicatif::ProgressBar;
use mascot_rs::mascot_generic_format::MGFPathIter;
use mascot_rs::prelude::{
    IonMode, MGFIter, MascotGenericFormat, MascotGenericFormatMetadata, Spectrum, SpectrumAlloc,
};
use mass_spectrometry::prelude::{ELECTRON_MASS, MAX_MZ, SpectrumSplash};
use ndarray::{Array1, Array3, s};
use sha2::{Digest, Sha256};

use crate::{Config, ProgressReporter};

/// Dataset label written into `GeMS` metadata fields.
const DATASET_NAME: &str = "GeMS-A10";
/// HDF5 spectrum tensor dataset name.
const SPECTRUM: &str = "spectrum";
/// HDF5 precursor m/z dataset name.
const PRECURSOR_MZ: &str = "precursor_mz";
/// HDF5 precursor charge dataset name.
const CHARGE: &str = "charge";
/// HDF5 retention-time dataset name.
const RETENTION_TIME: &str = "RT";
/// Preferred HDF5 source file-name dataset name.
const FILE_NAME: &str = "file_name";
/// Actual GeMS-A10 HDF5 source file-name dataset name.
const NAME: &str = "name";
/// HDF5 locality-sensitive-hashing cluster dataset name.
const LSH: &str = "lsh";
/// HDF5 source-run accuracy estimate dataset name.
const ACCURACY: &str = "instrument accuracy est.";
/// Name of the single compressed MGF output document.
const OUTPUT_MGF: &str = "GeMS_A10.mgf.zst";
/// Name of the conversion summary report.
const CONVERSION_REPORT: &str = "conversion_report.csv";
/// Name of the row-level SPLASH duplicate report.
const DUPLICATE_REPORT: &str = "splash_duplicates.csv";
/// Exact GeMS-A10 fragment peak capacity in the HDF5 spectrum tensor.
const EXPECTED_FRAGMENT_PEAKS: usize = 128;
/// Minimum number of valid fragment peaks required to write a spectrum.
const MIN_FRAGMENT_PEAKS: usize = 2;
/// Maximum number of fragment peaks retained in each written spectrum.
const MAX_FRAGMENT_PEAKS: usize = 100;
/// Number of row visits batched before updating the progress bar.
const PROGRESS_UPDATE_ROWS: u64 = 1024;
/// Read buffer size used for checksum generation.
const CHECKSUM_BUFFER_BYTES: usize = 0x0010_0000;

/// Manifest row for the generated MGF document.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ManifestRow {
    /// Dataset name.
    pub dataset: String,
    /// Relative MGF path.
    pub path: String,
    /// First HDF5 row covered by this document.
    pub start_row: usize,
    /// Last HDF5 row covered by this document.
    pub end_row: usize,
    /// Number of MGF records written.
    pub spectra_written: usize,
    /// Number of HDF5 rows skipped.
    pub spectra_skipped: usize,
    /// Number of skipped rows with invalid precursor m/z.
    pub skipped_invalid_precursor_mz: usize,
    /// Number of skipped rows with zero precursor charge.
    pub skipped_invalid_charge: usize,
    /// Number of skipped rows below the minimum valid fragment peak count.
    pub skipped_low_peak_spectra: usize,
    /// Number of skipped rows removed because their SPLASH was already seen.
    pub duplicates_removed: usize,
}

/// Summary for one conversion run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConversionReport {
    /// Manifest row for the generated document, if any rows were visited.
    pub manifest: Option<ManifestRow>,
    /// First HDF5 row visited.
    pub start_row: usize,
    /// Last HDF5 row visited, if any row was visited.
    pub end_row: Option<usize>,
    /// Total MGF records written.
    pub spectra_written: usize,
    /// Total HDF5 rows skipped.
    pub spectra_skipped: usize,
    /// Number of skipped rows with invalid precursor m/z.
    pub skipped_invalid_precursor_mz: usize,
    /// Number of skipped rows with zero precursor charge.
    pub skipped_invalid_charge: usize,
    /// Number of skipped rows below the minimum valid fragment peak count.
    pub skipped_low_peak_spectra: usize,
    /// Number of skipped rows removed because their SPLASH was already seen.
    pub duplicates_removed: usize,
}

/// Returns whether `value` is finite and strictly positive.
#[inline]
#[must_use]
pub fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Returns whether `value` is inside the physical m/z range accepted downstream.
#[inline]
#[must_use]
pub fn valid_mz(value: f64) -> bool {
    value.is_finite() && (ELECTRON_MASS..=MAX_MZ).contains(&value)
}

/// Returns whether `charge` is usable as a precursor charge.
#[inline]
#[must_use]
pub const fn valid_charge(charge: i8) -> bool {
    charge != 0
}

/// Converts the configured GeMS-A10 HDF5 file into one zstd-compressed MGF document.
///
/// # Errors
///
/// Returns an error if the HDF5 file is malformed, output cannot be written, or
/// optional output validation fails.
pub fn convert_gems_a10(config: &Config) -> anyhow::Result<ConversionReport> {
    convert_gems_a10_with_progress(config, &ProgressReporter::hidden())
}

/// Converts the configured GeMS-A10 HDF5 file while reporting progress.
///
/// # Errors
///
/// Returns an error if the HDF5 file is malformed, output cannot be written, or
/// optional output validation fails.
pub fn convert_gems_a10_with_progress(
    config: &Config,
    progress: &ProgressReporter,
) -> anyhow::Result<ConversionReport> {
    let setup = progress.spinner("preparing output directory and HDF5 datasets")?;
    if config.chunk_size == 0 {
        bail!("configured chunk size must be positive");
    }
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("failed to create {}", config.output_dir.display()))?;

    let h5 = H5File::open(&config.input_hdf5)
        .with_context(|| format!("failed to open {}", config.input_hdf5.display()))?;
    let datasets = Hdf5Datasets::open(&h5)?;
    let row_count = datasets.validate()?;
    setup.finish_with_message(format!(
        "HDF5 ready | rows={row_count} | input={}",
        config.input_hdf5.display()
    ));

    if config.start_row > row_count {
        bail!(
            "configured start row {} exceeds HDF5 row count {}",
            config.start_row,
            row_count
        );
    }

    let stop_row = config
        .limit
        .map_or(row_count, |limit| row_count.min(config.start_row + limit));
    let manifest_path = config.output_dir.join("manifest.csv");
    let mut manifest = Writer::from_path(&manifest_path)
        .with_context(|| format!("failed to create {}", manifest_path.display()))?;
    manifest.write_record([
        "dataset",
        "path",
        "start_row",
        "end_row",
        "spectra_written",
        "spectra_skipped",
        "skipped_invalid_precursor_mz",
        "skipped_invalid_charge",
        "skipped_low_peak_spectra",
        "duplicates_removed",
    ])?;

    let manifest_row = (stop_row > config.start_row)
        .then(|| write_document(config, &datasets, config.start_row, stop_row, progress))
        .transpose()?;
    if manifest_row.is_none() {
        write_empty_duplicate_report(&config.output_dir)?;
    }

    if let Some(row) = &manifest_row {
        manifest.serialize((
            &row.dataset,
            &row.path,
            row.start_row,
            row.end_row,
            row.spectra_written,
            row.spectra_skipped,
            row.skipped_invalid_precursor_mz,
            row.skipped_invalid_charge,
            row.skipped_low_peak_spectra,
            row.duplicates_removed,
        ))?;
        manifest.flush()?;

        if config.validate_output {
            validate_output_document(
                &config.output_dir.join(&row.path),
                row.spectra_written,
                progress,
            )?;
        }
        progress.println(format!(
            "{} rows {}-{} written={} skipped={}",
            row.path, row.start_row, row.end_row, row.spectra_written, row.spectra_skipped
        ))?;
    }

    let metadata = progress.spinner("writing README and conversion reports")?;
    write_dataset_readme(&config.output_dir, &config.input_hdf5)?;
    write_conversion_report(
        &config.output_dir,
        &config.input_hdf5,
        manifest_row.as_ref(),
    )?;
    metadata.finish_with_message(format!(
        "metadata reports written | output={}",
        config.output_dir.display()
    ));

    Ok(conversion_report_from_manifest(
        config.start_row,
        stop_row,
        manifest_row,
    ))
}

/// Builds the public conversion report from the optional manifest row.
fn conversion_report_from_manifest(
    start_row: usize,
    stop_row: usize,
    manifest_row: Option<ManifestRow>,
) -> ConversionReport {
    let spectra_written = manifest_row.as_ref().map_or(0, |row| row.spectra_written);
    let spectra_skipped = manifest_row.as_ref().map_or(0, |row| row.spectra_skipped);
    let skipped_invalid_precursor_mz = manifest_row
        .as_ref()
        .map_or(0, |row| row.skipped_invalid_precursor_mz);
    let skipped_invalid_charge = manifest_row
        .as_ref()
        .map_or(0, |row| row.skipped_invalid_charge);
    let skipped_low_peak_spectra = manifest_row
        .as_ref()
        .map_or(0, |row| row.skipped_low_peak_spectra);
    let duplicates_removed = manifest_row
        .as_ref()
        .map_or(0, |row| row.duplicates_removed);
    ConversionReport {
        manifest: manifest_row,
        start_row,
        end_row: (stop_row > start_row).then_some(stop_row - 1),
        spectra_written,
        spectra_skipped,
        skipped_invalid_precursor_mz,
        skipped_invalid_charge,
        skipped_low_peak_spectra,
        duplicates_removed,
    }
}

/// Writes the dataset README intended to travel with Zenodo artifacts.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_dataset_readme(output_dir: &Path, source_hdf5: &Path) -> anyhow::Result<PathBuf> {
    let readme_path = output_dir.join("README.txt");
    let source_name = source_hdf5
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("GeMS_A10.hdf5");
    let today = chrono::Utc::now().date_naive();
    fs::write(
        &readme_path,
        format!(
            "\
GeMS-A10 converted to Mascot Generic Format

Source HDF5: {source_name}
Source dataset: https://huggingface.co/datasets/roman-bushuiev/GeMS
Source file: data/GeMS_A/GeMS_A10.hdf5
Conversion date: {today}
Validator: mascot-rs via this Rust conversion crate

Conversion policy:
- one HDF5 row maps to one MGF BEGIN IONS block when valid and SPLASH-unique
- FEATURE_ID, SCANS, and GEMS_ROW_INDEX use the zero-based HDF5 row
- PEPMASS comes from precursor_mz and must be within the physical m/z range
- RTINSECONDS is written only for finite positive RT values
- MSLEVEL is written as 2
- rows with zero charge are skipped
- fragment peaks outside the physical m/z range or with non-positive/non-finite intensity are removed
- rows with fewer than {MIN_FRAGMENT_PEAKS} valid fragment peaks after m/z merging are skipped
- at most {MAX_FRAGMENT_PEAKS} highest-intensity fragment peaks are retained
- fragment intensities are not renormalized
- SPLASH is computed from the filtered and capped fragment peaks and written as metadata
- rows with duplicate SPLASH values are removed after the first retained row
- SOURCE_INSTRUMENT is not written because instrument accuracy est. is not an instrument identity
- IONMODE is omitted because GeMS-A10 does not expose polarity

Files:
- GeMS_A10.mgf.zst: compressed MGF document
- manifest.csv: row range and skipped/written counts
- conversion_report.csv: conversion summary and duplicate counts
- splash_duplicates.csv: row-level SPLASH duplicate report
- SHA256SUMS: checksums
"
        ),
    )
    .with_context(|| format!("failed to write {}", readme_path.display()))?;
    Ok(readme_path)
}

/// Writes `SHA256SUMS` for the MGF document and metadata.
///
/// # Errors
///
/// Returns an error if the output directory cannot be read or the checksum file
/// cannot be written.
pub fn write_sha256sums(output_dir: &Path) -> anyhow::Result<PathBuf> {
    write_sha256sums_with_progress(output_dir, &ProgressReporter::hidden())
}

/// Writes `SHA256SUMS` while reporting checksum progress.
///
/// # Errors
///
/// Returns an error if the output directory cannot be read or the checksum file
/// cannot be written.
pub fn write_sha256sums_with_progress(
    output_dir: &Path,
    progress: &ProgressReporter,
) -> anyhow::Result<PathBuf> {
    let paths = expected_artifact_paths(output_dir, false)?;
    let total_bytes = paths.iter().try_fold(0u64, |total, path| {
        Ok::<u64, anyhow::Error>(total + fs::metadata(path)?.len())
    })?;
    let checksums =
        progress.byte_bar(total_bytes, format!("checksumming {} files", paths.len()))?;

    let checksum_path = output_dir.join("SHA256SUMS");
    let mut writer = BufWriter::new(
        File::create(&checksum_path)
            .with_context(|| format!("failed to create {}", checksum_path.display()))?,
    );
    for path in paths {
        checksums.set_message(format!(
            "checksumming {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("artifact")
        ));
        let digest = sha256_file(&path, Some(&checksums))?;
        let digest = format_digest_hex(digest.as_ref())?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("checksum path has no UTF-8 filename")?;
        writeln!(writer, "{digest}  {name}")?;
    }
    writer.flush()?;
    checksums.finish_with_message(format!("checksums written | {}", checksum_path.display()));
    Ok(checksum_path)
}

/// Returns the exact expected artifact paths, failing if any are missing.
///
/// When `include_checksums` is true, `SHA256SUMS` is included in the required
/// artifact list.
pub fn expected_artifact_paths(
    output_dir: &Path,
    include_checksums: bool,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut names = vec![
        OUTPUT_MGF,
        "manifest.csv",
        "README.txt",
        CONVERSION_REPORT,
        DUPLICATE_REPORT,
    ];
    if include_checksums {
        names.push("SHA256SUMS");
    }

    let paths = names
        .into_iter()
        .map(|name| {
            let path = output_dir.join(name);
            if !path.is_file() {
                bail!("expected artifact is missing: {}", path.display());
            }
            Ok(path)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(paths)
}

/// Open HDF5 datasets required by the conversion.
struct Hdf5Datasets {
    /// Fragment spectrum tensor dataset.
    spectrum: Dataset,
    /// Precursor m/z values.
    precursor_mz: Dataset,
    /// Precursor charge values.
    charge: Dataset,
    /// Retention-time values.
    retention_time: Dataset,
    /// Source run/file names.
    file_name: Dataset,
    /// `GeMS` LSH cluster keys.
    lsh: Dataset,
    /// Instrument accuracy estimates.
    accuracy: Dataset,
}

impl Hdf5Datasets {
    /// Opens all datasets required by the conversion.
    fn open(h5: &H5File) -> anyhow::Result<Self> {
        Ok(Self {
            spectrum: h5.dataset(SPECTRUM)?,
            precursor_mz: h5.dataset(PRECURSOR_MZ)?,
            charge: h5.dataset(CHARGE)?,
            retention_time: h5.dataset(RETENTION_TIME)?,
            file_name: h5.dataset(FILE_NAME).or_else(|_| h5.dataset(NAME))?,
            lsh: h5.dataset(LSH)?,
            accuracy: h5.dataset(ACCURACY)?,
        })
    }

    /// Validates required row alignment and returns the spectrum row count.
    fn validate(&self) -> anyhow::Result<usize> {
        let shape = self.spectrum.shape();
        let row_count = match shape.as_slice() {
            [row_count, 2, peak_count] if *peak_count == EXPECTED_FRAGMENT_PEAKS => *row_count,
            _ => {
                bail!("spectrum must have shape (N, 2, {EXPECTED_FRAGMENT_PEAKS}), got {shape:?}");
            }
        };
        if row_count == 0 {
            return Ok(row_count);
        }
        let file_name_dataset_name = self.file_name.name();
        for (name, dataset) in [
            (PRECURSOR_MZ, &self.precursor_mz),
            (CHARGE, &self.charge),
            (RETENTION_TIME, &self.retention_time),
            (file_name_dataset_name.as_str(), &self.file_name),
            (LSH, &self.lsh),
            (ACCURACY, &self.accuracy),
        ] {
            let shape = dataset.shape();
            if shape.first().copied() != Some(row_count) {
                bail!(
                    "{name} row count {:?} does not match spectrum row count {row_count}",
                    shape.first()
                );
            }
        }
        Ok(row_count)
    }
}

/// Chunk of HDF5 rows read into memory for conversion.
#[derive(Debug)]
struct Chunk {
    /// Fragment spectrum tensor for the chunk.
    spectra: Array3<f64>,
    /// Precursor m/z values for the chunk.
    precursor_mz: Array1<f64>,
    /// Precursor charges for the chunk.
    charge: Array1<i8>,
    /// Retention times for the chunk.
    retention_time: Array1<f64>,
    /// Source file names for the chunk.
    file_name: Vec<String>,
    /// LSH cluster keys for the chunk.
    lsh: Vec<String>,
    /// Instrument accuracy estimates for the chunk.
    accuracy: Array1<f64>,
}

/// Summary for the first retained spectrum for a SPLASH.
#[derive(Debug, Clone, PartialEq)]
struct RetainedSpectrum {
    /// Zero-based HDF5 row retained in the MGF document.
    row: usize,
    /// Precursor m/z for the retained spectrum.
    precursor_mz: f64,
    /// Precursor charge for the retained spectrum.
    charge: i8,
    /// Positive finite retention time for the retained spectrum.
    retention_time: Option<f64>,
    /// Number of retained fragment peaks.
    peak_count: usize,
}

/// Report information for the current spectrum row.
#[derive(Debug, Clone, PartialEq)]
struct SpectrumReportInfo {
    /// Zero-based HDF5 row.
    row: usize,
    /// Precursor m/z.
    precursor_mz: f64,
    /// Precursor charge.
    charge: i8,
    /// Positive finite retention time.
    retention_time: Option<f64>,
    /// Source run or file name.
    file_name: Option<String>,
    /// `GeMS` LSH cluster key.
    lsh: String,
    /// Finite source-run accuracy estimate.
    accuracy: Option<f64>,
    /// Number of retained fragment peaks.
    peak_count: usize,
}

/// MGF record prepared for SPLASH deduplication.
#[derive(Debug)]
struct PreparedRecord {
    /// MGF record produced by `mascot-rs`.
    record: MascotGenericFormat<u32, f64>,
    /// SPLASH code computed from filtered fragment peaks.
    splash: String,
    /// Row-level information for reports and final metadata.
    report: SpectrumReportInfo,
}

impl PreparedRecord {
    /// Builds the retained-spectrum summary for this row.
    #[inline]
    const fn retained(&self) -> RetainedSpectrum {
        RetainedSpectrum {
            row: self.report.row,
            precursor_mz: self.report.precursor_mz,
            charge: self.report.charge,
            retention_time: self.report.retention_time,
            peak_count: self.report.peak_count,
        }
    }

    /// Adds GeMS-specific metadata before writing a unique record.
    fn add_output_metadata(&mut self) {
        let metadata = self.record.metadata_mut();
        metadata.insert_arbitrary_metadata("GEMS_DATASET", DATASET_NAME);
        metadata.insert_arbitrary_metadata("GEMS_ROW_INDEX", self.report.row.to_string());
        metadata.insert_arbitrary_metadata("GEMS_LSH", &self.report.lsh);
        if let Some(accuracy) = self.report.accuracy {
            metadata
                .insert_arbitrary_metadata("GEMS_INSTRUMENT_ACCURACY_EST", format_float(accuracy));
        }
        metadata.insert_arbitrary_metadata("SPLASH", &self.splash);
    }
}

/// Writes the configured row range into the single compressed MGF document.
fn write_document(
    config: &Config,
    datasets: &Hdf5Datasets,
    start: usize,
    end: usize,
    progress: &ProgressReporter,
) -> anyhow::Result<ManifestRow> {
    let document_path = config.output_dir.join(OUTPUT_MGF);
    let mut encoder = create_mgf_encoder(&document_path)?;
    let duplicate_report_path = config.output_dir.join(DUPLICATE_REPORT);
    let mut duplicate_report = Writer::from_path(&duplicate_report_path)
        .with_context(|| format!("failed to create {}", duplicate_report_path.display()))?;
    write_duplicate_report_header(&mut duplicate_report)?;

    let mut written = 0;
    let mut skipped_invalid_precursor_mz = 0;
    let mut skipped_invalid_charge = 0;
    let mut skipped_low_peak_spectra = 0;
    let mut duplicates_removed = 0;
    let rows_to_visit = end - start;
    let mut seen_splashes = HashMap::<String, RetainedSpectrum>::with_capacity(rows_to_visit);
    let rows = progress.row_bar(
        rows_to_visit,
        format!(
            "converting rows {start}-{} | written=0 skipped=0 duplicates=0",
            end - 1
        ),
    )?;
    let mut pending_progress = 0;

    for chunk_start in (start..end).step_by(config.chunk_size) {
        let chunk_end = (chunk_start + config.chunk_size).min(end);
        rows.set_message(format!(
            "reading chunk {chunk_start}-{} | written={written} skipped={} duplicates={duplicates_removed}",
            chunk_end - 1,
                skipped_invalid_precursor_mz
                    + skipped_invalid_charge
                    + skipped_low_peak_spectra
                    + duplicates_removed
        ));
        let chunk = read_chunk(datasets, chunk_start, chunk_end)?;
        for offset in 0..(chunk_end - chunk_start) {
            let precursor_mz = array_value(&chunk.precursor_mz, offset, PRECURSOR_MZ)?;
            if !valid_mz(precursor_mz) {
                skipped_invalid_precursor_mz += 1;
                advance_progress(&rows, &mut pending_progress);
                continue;
            }
            let charge = array_value(&chunk.charge, offset, CHARGE)?;
            if !valid_charge(charge) {
                skipped_invalid_charge += 1;
                advance_progress(&rows, &mut pending_progress);
                continue;
            }

            let row = chunk_start + offset;
            let Some(prepared) = prepare_record(&chunk, offset, row, precursor_mz, charge)? else {
                skipped_low_peak_spectra += 1;
                advance_progress(&rows, &mut pending_progress);
                continue;
            };
            write_prepared_record(
                prepared,
                &mut seen_splashes,
                &mut duplicate_report,
                &mut encoder,
                &mut written,
                &mut duplicates_removed,
            )?;
            advance_progress(&rows, &mut pending_progress);
        }
    }
    flush_progress(&rows, &mut pending_progress);

    encoder.finish()?;
    duplicate_report.flush()?;
    let spectra_skipped = skipped_invalid_precursor_mz
        + skipped_invalid_charge
        + skipped_low_peak_spectra
        + duplicates_removed;
    rows.finish_with_message(format!(
        "conversion complete | rows={rows_to_visit} written={written} skipped={spectra_skipped} duplicates={duplicates_removed}"
    ));
    Ok(ManifestRow {
        dataset: DATASET_NAME.to_owned(),
        path: OUTPUT_MGF.to_owned(),
        start_row: start,
        end_row: end - 1,
        spectra_written: written,
        spectra_skipped,
        skipped_invalid_precursor_mz,
        skipped_invalid_charge,
        skipped_low_peak_spectra,
        duplicates_removed,
    })
}

/// Creates the compressed MGF writer.
fn create_mgf_encoder(
    document_path: &Path,
) -> anyhow::Result<zstd::stream::write::Encoder<'static, BufWriter<File>>> {
    let file = File::create(document_path)
        .with_context(|| format!("failed to create {}", document_path.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = zstd::stream::write::Encoder::new(writer, 3).with_context(|| {
        format!(
            "failed to create zstd encoder for {}",
            document_path.display()
        )
    })?;
    encoder
        .multithread(zstd_worker_count())
        .context("failed to enable multithreaded zstd compression")?;
    Ok(encoder)
}

/// Writes or reports a prepared MGF record after SPLASH deduplication.
fn write_prepared_record<D: Write, Z: Write>(
    prepared: PreparedRecord,
    seen_splashes: &mut HashMap<String, RetainedSpectrum>,
    duplicate_report: &mut Writer<D>,
    encoder: &mut zstd::stream::write::Encoder<'_, Z>,
    written: &mut usize,
    duplicates_removed: &mut usize,
) -> anyhow::Result<()> {
    match seen_splashes.entry(prepared.splash.clone()) {
        Entry::Occupied(entry) => {
            write_duplicate_report_row(
                duplicate_report,
                &prepared.splash,
                entry.get(),
                &prepared.report,
            )?;
            *duplicates_removed += 1;
        }
        Entry::Vacant(entry) => {
            entry.insert(prepared.retained());
            let mut prepared = prepared;
            prepared.add_output_metadata();
            prepared.record.write_to(&mut *encoder)?;
            writeln!(encoder)?;
            *written += 1;
        }
    }
    Ok(())
}

/// Writes an empty duplicate report for an empty conversion range.
fn write_empty_duplicate_report(output_dir: &Path) -> anyhow::Result<PathBuf> {
    let duplicate_report_path = output_dir.join(DUPLICATE_REPORT);
    let mut duplicate_report = Writer::from_path(&duplicate_report_path)
        .with_context(|| format!("failed to create {}", duplicate_report_path.display()))?;
    write_duplicate_report_header(&mut duplicate_report)?;
    duplicate_report.flush()?;
    Ok(duplicate_report_path)
}

/// Returns the zstd compression worker count for this machine.
fn zstd_worker_count() -> u32 {
    let workers = std::thread::available_parallelism()
        .map_or(1, |available| available.get().saturating_sub(1).max(1));
    u32::try_from(workers).map_or(u32::MAX, |worker_count| worker_count)
}

/// Records one processed row and updates the progress bar in batches.
#[inline]
fn advance_progress(progress: &ProgressBar, pending: &mut u64) {
    *pending += 1;
    if *pending >= PROGRESS_UPDATE_ROWS {
        flush_progress(progress, pending);
    }
}

/// Flushes any pending batched progress updates.
#[inline]
fn flush_progress(progress: &ProgressBar, pending: &mut u64) {
    if *pending > 0 {
        progress.inc(*pending);
        *pending = 0;
    }
}

/// Writes the header for the row-level SPLASH duplicate report.
fn write_duplicate_report_header<W: Write>(writer: &mut Writer<W>) -> anyhow::Result<()> {
    writer.write_record([
        "dataset",
        "splash",
        "retained_row",
        "duplicate_row",
        "duplicate_precursor_mz",
        "duplicate_charge",
        "duplicate_rt",
        "duplicate_file_name",
        "duplicate_lsh",
        "duplicate_instrument_accuracy_est",
        "duplicate_peak_count",
        "retained_precursor_mz",
        "retained_charge",
        "retained_rt",
        "retained_peak_count",
    ])?;
    Ok(())
}

/// Writes one skipped duplicate spectrum to the duplicate report.
fn write_duplicate_report_row<W: Write>(
    writer: &mut Writer<W>,
    splash: &str,
    retained: &RetainedSpectrum,
    duplicate: &SpectrumReportInfo,
) -> anyhow::Result<()> {
    writer.write_record([
        DATASET_NAME.to_owned(),
        splash.to_owned(),
        retained.row.to_string(),
        duplicate.row.to_string(),
        format_float(duplicate.precursor_mz),
        duplicate.charge.to_string(),
        format_optional_float(duplicate.retention_time),
        duplicate.file_name.clone().unwrap_or_default(),
        duplicate.lsh.clone(),
        format_optional_float(duplicate.accuracy),
        duplicate.peak_count.to_string(),
        format_float(retained.precursor_mz),
        retained.charge.to_string(),
        format_optional_float(retained.retention_time),
        retained.peak_count.to_string(),
    ])?;
    Ok(())
}

/// Writes a summary report for the conversion and deduplication run.
fn write_conversion_report(
    output_dir: &Path,
    source_hdf5: &Path,
    manifest: Option<&ManifestRow>,
) -> anyhow::Result<PathBuf> {
    let report_path = output_dir.join(CONVERSION_REPORT);
    let mut writer = Writer::from_path(&report_path)
        .with_context(|| format!("failed to create {}", report_path.display()))?;
    writer.write_record(["field", "value"])?;
    write_report_field(&mut writer, "dataset", DATASET_NAME)?;
    write_report_field(
        &mut writer,
        "source_hdf5",
        &source_hdf5.display().to_string(),
    )?;
    write_report_field(&mut writer, "output_mgf", OUTPUT_MGF)?;
    write_report_field(&mut writer, "duplicate_report", DUPLICATE_REPORT)?;
    write_report_field(
        &mut writer,
        "max_fragment_peaks",
        &MAX_FRAGMENT_PEAKS.to_string(),
    )?;
    write_report_field(
        &mut writer,
        "min_fragment_peaks",
        &MIN_FRAGMENT_PEAKS.to_string(),
    )?;
    if let Some(row) = manifest {
        let rows_visited = row.end_row - row.start_row + 1;
        write_report_field(&mut writer, "start_row", &row.start_row.to_string())?;
        write_report_field(&mut writer, "end_row", &row.end_row.to_string())?;
        write_report_field(&mut writer, "rows_visited", &rows_visited.to_string())?;
        write_report_field(
            &mut writer,
            "spectra_written",
            &row.spectra_written.to_string(),
        )?;
        write_report_field(
            &mut writer,
            "spectra_skipped",
            &row.spectra_skipped.to_string(),
        )?;
        write_report_field(
            &mut writer,
            "skipped_invalid_precursor_mz",
            &row.skipped_invalid_precursor_mz.to_string(),
        )?;
        write_report_field(
            &mut writer,
            "skipped_invalid_charge",
            &row.skipped_invalid_charge.to_string(),
        )?;
        write_report_field(
            &mut writer,
            "skipped_low_peak_spectra",
            &row.skipped_low_peak_spectra.to_string(),
        )?;
        write_report_field(
            &mut writer,
            "duplicates_removed",
            &row.duplicates_removed.to_string(),
        )?;
        write_report_field(
            &mut writer,
            "unique_splash_count",
            &row.spectra_written.to_string(),
        )?;
    } else {
        for field in [
            "start_row",
            "end_row",
            "rows_visited",
            "spectra_written",
            "spectra_skipped",
            "skipped_invalid_precursor_mz",
            "skipped_invalid_charge",
            "skipped_low_peak_spectra",
            "duplicates_removed",
            "unique_splash_count",
        ] {
            write_report_field(&mut writer, field, "0")?;
        }
    }
    writer.flush()?;
    Ok(report_path)
}

/// Writes one key-value conversion report field.
fn write_report_field<W: Write>(
    writer: &mut Writer<W>,
    field: &str,
    value: &str,
) -> anyhow::Result<()> {
    writer.write_record([field, value])?;
    Ok(())
}

/// Reads one row chunk from all required HDF5 datasets.
fn read_chunk(datasets: &Hdf5Datasets, start: usize, end: usize) -> anyhow::Result<Chunk> {
    let selection = s![start..end];
    Ok(Chunk {
        spectra: datasets.spectrum.read_slice(s![start..end, .., ..])?,
        precursor_mz: datasets.precursor_mz.read_slice(selection)?,
        charge: datasets.charge.read_slice(selection)?,
        retention_time: datasets.retention_time.read_slice(selection)?,
        file_name: read_string_slice(&datasets.file_name, start, end)
            .context("failed to read file_name string chunk")?,
        lsh: read_lsh_slice(&datasets.lsh, start, end).context("failed to read lsh chunk")?,
        accuracy: datasets.accuracy.read_slice(selection)?,
    })
}

/// Reads an LSH chunk as text from string or integer HDF5 datasets.
fn read_lsh_slice(dataset: &Dataset, start: usize, end: usize) -> anyhow::Result<Vec<String>> {
    if let Ok(values) = read_string_slice(dataset, start, end) {
        return Ok(values);
    }
    if let Ok(values) = dataset.read_slice_1d::<i64, _>(s![start..end]) {
        return Ok(values.iter().map(ToString::to_string).collect());
    }
    if let Ok(values) = dataset.read_slice_1d::<u64, _>(s![start..end]) {
        return Ok(values.iter().map(ToString::to_string).collect());
    }

    bail!(
        "unsupported HDF5 LSH dtype for '{}': {:?}",
        dataset.name(),
        dataset.dtype()?.to_descriptor()?
    );
}

/// Reads a UTF-8 string slice from variable- or fixed-width HDF5 string datasets.
fn read_string_slice(dataset: &Dataset, start: usize, end: usize) -> anyhow::Result<Vec<String>> {
    if let Ok(values) = dataset.read_slice_1d::<VarLenUnicode, _>(s![start..end]) {
        return Ok(values.iter().map(ToString::to_string).collect());
    }
    if let Ok(values) = dataset.read_slice_1d::<VarLenAscii, _>(s![start..end]) {
        return Ok(values.iter().map(ToString::to_string).collect());
    }

    let dtype = dataset.dtype()?.to_descriptor()?;
    match dtype {
        TypeDescriptor::FixedAscii(width) | TypeDescriptor::FixedUnicode(width) => {
            read_fixed_string_slice(dataset, start, end, width)
        }
        other @ (TypeDescriptor::Integer(_)
        | TypeDescriptor::Unsigned(_)
        | TypeDescriptor::Float(_)
        | TypeDescriptor::Boolean
        | TypeDescriptor::Enum(_)
        | TypeDescriptor::Compound(_)
        | TypeDescriptor::FixedArray(..)
        | TypeDescriptor::VarLenArray(_)
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode
        | TypeDescriptor::Reference(_)) => bail!(
            "unsupported HDF5 string dtype for '{}': {other:?}",
            dataset.name()
        ),
    }
}

/// Reads fixed-width HDF5 strings into owned Rust strings.
fn read_fixed_string_slice(
    dataset: &Dataset,
    start: usize,
    end: usize,
    width: usize,
) -> anyhow::Result<Vec<String>> {
    let count = end - start;
    if width == 0 {
        return Ok(vec![String::new(); count]);
    }

    let selection: Selection = s![start..end].try_into()?;
    let file_space = dataset.space()?.select(selection)?;
    let memory_space = Dataspace::try_new([count])?;
    let memory_type = Datatype::from_descriptor(&dataset.dtype()?.to_descriptor()?)?;
    let mut bytes = vec![0u8; count * width];
    // Safety: `bytes` has room for `count * width` bytes, and the memory
    // dataspace/type describe exactly `count` fixed-width string elements.
    let status = unsafe {
        H5Dread(
            dataset.id(),
            memory_type.id(),
            memory_space.id(),
            file_space.id(),
            H5P_DEFAULT,
            bytes.as_mut_ptr().cast(),
        )
    };
    if status < 0i32 {
        bail!(
            "failed to read fixed-width HDF5 string dataset '{}'",
            dataset.name()
        );
    }

    Ok(bytes.chunks_exact(width).map(decode_fixed_string).collect())
}

/// Decodes one fixed-width string cell, trimming null padding.
fn decode_fixed_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    let trimmed = bytes.get(..end).unwrap_or(bytes);
    String::from_utf8_lossy(trimmed).into_owned()
}

/// Prepares one MGF record for SPLASH deduplication.
fn prepare_record(
    chunk: &Chunk,
    offset: usize,
    row: usize,
    precursor_mz: f64,
    charge: i8,
) -> anyhow::Result<Option<PreparedRecord>> {
    let retention_time = array_value(&chunk.retention_time, offset, RETENTION_TIME)?;
    let rt = finite_positive(retention_time).then_some(retention_time);
    let filename = non_empty_string(slice_value(&chunk.file_name, offset, FILE_NAME)?.clone());
    let lsh = slice_value(&chunk.lsh, offset, LSH)?.clone();
    let accuracy = array_value(&chunk.accuracy, offset, ACCURACY)?;
    let row_id = u32::try_from(row).context("HDF5 row index does not fit u32")?;
    let peak_count = chunk
        .spectra
        .shape()
        .get(2)
        .copied()
        .context("spectrum chunk is missing peak dimension")?;
    let mz_values = chunk.spectra.slice(s![offset, 0usize, ..]);
    let intensity_values = chunk.spectra.slice(s![offset, 1usize, ..]);
    let mut mzs = Vec::with_capacity(peak_count);
    let mut intensities = Vec::with_capacity(peak_count);
    for (mz, intensity) in mz_values.iter().zip(intensity_values.iter()) {
        if valid_mz(*mz) && finite_positive(*intensity) {
            mzs.push(*mz);
            intensities.push(*intensity);
        }
    }
    if mzs.is_empty() {
        return Ok(None);
    }

    let metadata = MascotGenericFormatMetadata::<u32>::new_with_smiles_and_ion_mode(
        Some(row_id),
        2,
        rt,
        charge,
        filename.clone(),
        None,
        None::<IonMode>,
    )?;
    let spectrum = MascotGenericFormat::new(metadata, precursor_mz, mzs, intensities)?
        .top_k_peaks(MAX_FRAGMENT_PEAKS)
        .context("failed to retain top fragment peaks")?;
    let retained_peak_count = spectrum.len();
    if retained_peak_count < MIN_FRAGMENT_PEAKS {
        return Ok(None);
    }

    let splash = spectrum.splash().context("failed to compute SPLASH")?;
    let accuracy = accuracy.is_finite().then_some(accuracy);
    Ok(Some(PreparedRecord {
        record: spectrum,
        splash,
        report: SpectrumReportInfo {
            row,
            precursor_mz,
            charge,
            retention_time: rt,
            file_name: filename,
            lsh,
            accuracy,
            peak_count: retained_peak_count,
        },
    }))
}

/// Returns a copied array value with contextual bounds errors.
#[inline]
fn array_value<T: Copy>(values: &Array1<T>, offset: usize, field: &str) -> anyhow::Result<T> {
    values
        .get(offset)
        .copied()
        .with_context(|| format!("missing {field} value at chunk offset {offset}"))
}

/// Returns a slice value with contextual bounds errors.
#[inline]
fn slice_value<'a, T>(values: &'a [T], offset: usize, field: &str) -> anyhow::Result<&'a T> {
    values
        .get(offset)
        .with_context(|| format!("missing {field} value at chunk offset {offset}"))
}

/// Streams the written MGF document and checks the expected record count.
fn validate_output_document(
    path: &Path,
    expected_records: usize,
    progress: &ProgressReporter,
) -> anyhow::Result<()> {
    let validation = progress.row_bar(
        expected_records,
        format!("validating MGF parse-back | {}", path.display()),
    )?;
    let mut records: MGFPathIter<usize, f64> = MGFIter::from_path(path)
        .with_context(|| format!("failed to open written MGF document {}", path.display()))?;
    let mut seen_splashes = HashMap::<String, usize>::with_capacity(expected_records);
    let mut pending_progress = 0;
    let observed = records.try_fold(0usize, |count, record| {
        let record = record
            .with_context(|| format!("failed to parse written MGF document {}", path.display()))?;
        validate_parsed_record(&record, path, count, &mut seen_splashes)?;
        advance_progress(&validation, &mut pending_progress);
        Ok::<usize, anyhow::Error>(count + 1)
    })?;
    flush_progress(&validation, &mut pending_progress);
    if observed != expected_records {
        bail!(
            "written MGF document {} parsed to {observed} records, expected {expected_records}",
            path.display()
        );
    }
    validation.finish_with_message(format!(
        "validation complete | parsed={observed} records | {}",
        path.display()
    ));
    Ok(())
}

/// Validates one parsed output MGF record against the conversion policy.
fn validate_parsed_record(
    record: &MascotGenericFormat<usize, f64>,
    path: &Path,
    record_index: usize,
    seen_splashes: &mut HashMap<String, usize>,
) -> anyhow::Result<()> {
    let peak_count = record.len();
    if !(MIN_FRAGMENT_PEAKS..=MAX_FRAGMENT_PEAKS).contains(&peak_count) {
        bail!(
            "record {record_index} in {} has {peak_count} peaks, expected {}..={}",
            path.display(),
            MIN_FRAGMENT_PEAKS,
            MAX_FRAGMENT_PEAKS
        );
    }
    if record.level() != 2 {
        bail!(
            "record {record_index} in {} has MSLEVEL={}, expected 2",
            path.display(),
            record.level()
        );
    }
    if !valid_charge(record.charge()) {
        bail!(
            "record {record_index} in {} has invalid zero charge",
            path.display()
        );
    }
    if !valid_mz(record.precursor_mz()) {
        bail!(
            "record {record_index} in {} has invalid precursor m/z {}",
            path.display(),
            record.precursor_mz()
        );
    }
    for (mz, intensity) in record.peaks() {
        if !valid_mz(mz) || !finite_positive(intensity) {
            bail!(
                "record {record_index} in {} has invalid peak {mz} {intensity}",
                path.display()
            );
        }
    }

    let feature_id = record.feature_id().with_context(|| {
        format!(
            "record {record_index} in {} has no FEATURE_ID",
            path.display()
        )
    })?;
    let row_index = record
        .metadata()
        .arbitrary_metadata_value("GEMS_ROW_INDEX")
        .with_context(|| {
            format!(
                "record {record_index} in {} has no GEMS_ROW_INDEX",
                path.display()
            )
        })?
        .parse::<usize>()
        .with_context(|| {
            format!(
                "record {record_index} in {} has non-numeric GEMS_ROW_INDEX",
                path.display()
            )
        })?;
    if feature_id != row_index {
        bail!(
            "record {record_index} in {} has FEATURE_ID={feature_id} but GEMS_ROW_INDEX={row_index}",
            path.display()
        );
    }

    let observed_splash = record
        .metadata()
        .arbitrary_metadata_value("SPLASH")
        .with_context(|| format!("record {record_index} in {} has no SPLASH", path.display()))?;
    let computed_splash = record
        .splash()
        .context("failed to recompute parsed SPLASH")?;
    if observed_splash != computed_splash {
        bail!(
            "record {record_index} in {} has SPLASH={observed_splash}, recomputed {computed_splash}",
            path.display()
        );
    }
    match seen_splashes.entry(observed_splash.to_owned()) {
        Entry::Occupied(entry) => {
            bail!(
                "record {record_index} in {} repeats SPLASH from record {}",
                path.display(),
                entry.get()
            );
        }
        Entry::Vacant(entry) => {
            entry.insert(record_index);
        }
    }
    Ok(())
}

/// Computes a SHA-256 digest for one file.
fn sha256_file(
    path: &Path,
    progress: Option<&ProgressBar>,
) -> anyhow::Result<sha2::digest::Output<Sha256>> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; CHECKSUM_BUFFER_BYTES];
    loop {
        let bytes_read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        let chunk = buffer
            .get(..bytes_read)
            .with_context(|| format!("invalid read length for {}", path.display()))?;
        digest.update(chunk);
        if let Some(progress) = progress {
            progress.inc(u64::try_from(bytes_read).context("read length does not fit u64")?);
        }
    }
    Ok(digest.finalize())
}

/// Formats digest bytes as lowercase hexadecimal text.
fn format_digest_hex(bytes: &[u8]) -> anyhow::Result<String> {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut hex, "{byte:02x}").context("failed to format digest as hex")?;
    }
    Ok(hex)
}

/// Converts an empty string into `None`.
#[inline]
fn non_empty_string(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

/// Formats a floating point value without trailing zero padding.
fn format_float(value: f64) -> String {
    format!("{value:.10}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

/// Formats an optional floating point value, leaving missing values empty.
fn format_optional_float(value: Option<f64>) -> String {
    value.map(format_float).unwrap_or_default()
}

#[cfg(test)]
/// Tests for HDF5 conversion, fixed-string decoding, and metadata outputs.
mod tests {
    use std::str::FromStr;

    use anyhow::ensure;
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenUnicode};
    use ndarray::Array3;
    use tempfile::TempDir;

    use super::*;

    /// Converts a realistic fixture and checks filtering, metadata, and artifacts.
    #[test]
    fn convert_fixture_filters_and_writes_manifest() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let input = temp.path().join("GeMS_A10.hdf5");
        write_fixture(&input)?;

        let config = Config {
            input_hdf5: input,
            output_dir: temp.path().join("converted"),
            chunk_size: 1,
            start_row: 0,
            limit: None,
            validate_output: true,
            publish_to_zenodo: false,
        };

        let report = convert_gems_a10(&config)?;
        ensure!(report.spectra_written == 1, "unexpected written count");
        ensure!(report.spectra_skipped == 6, "unexpected skipped count");
        ensure!(
            report.skipped_invalid_precursor_mz == 1,
            "unexpected invalid precursor count"
        );
        ensure!(
            report.skipped_invalid_charge == 1,
            "unexpected invalid charge count"
        );
        ensure!(
            report.skipped_low_peak_spectra == 3,
            "unexpected low-peak spectrum count"
        );
        ensure!(report.duplicates_removed == 1, "unexpected duplicate count");
        ensure!(
            report.manifest.as_ref().map(|row| (
                row.spectra_written,
                row.spectra_skipped,
                row.duplicates_removed
            )) == Some((1, 6, 1)),
            "unexpected manifest counts"
        );

        let output_path = config.output_dir.join("GeMS_A10.mgf.zst");
        let mut parsed: MGFPathIter<usize, f64> = MGFIter::from_path(output_path)?;
        let first = parsed
            .next()
            .transpose()?
            .context("missing first parsed MGF")?;
        ensure!(first.feature_id() == Some(0), "unexpected feature id");
        ensure!(
            first.metadata().arbitrary_metadata_value("GEMS_ROW_INDEX") == Some("0"),
            "unexpected row index metadata"
        );
        ensure!(
            first.metadata().arbitrary_metadata_value("GEMS_LSH") == Some("101"),
            "unexpected LSH metadata"
        );
        ensure!(
            first
                .metadata()
                .arbitrary_metadata_value("SPLASH")
                .is_some_and(|splash| splash.starts_with("splash10-")),
            "missing SPLASH metadata"
        );
        ensure!(
            config.output_dir.join("manifest.csv").exists(),
            "manifest was not written"
        );
        ensure!(
            config.output_dir.join("README.txt").exists(),
            "dataset README was not written"
        );
        let duplicate_report = fs::read_to_string(config.output_dir.join("splash_duplicates.csv"))?;
        ensure!(
            duplicate_report.contains(",0,3,"),
            "duplicate report did not link retained and duplicate rows"
        );
        let conversion_report =
            fs::read_to_string(config.output_dir.join("conversion_report.csv"))?;
        ensure!(
            conversion_report.contains("duplicates_removed,1"),
            "conversion report did not include duplicate count"
        );
        ensure!(
            conversion_report.contains("min_fragment_peaks,2"),
            "conversion report did not include the minimum fragment peak filter"
        );
        ensure!(
            conversion_report.contains("skipped_invalid_charge,1"),
            "conversion report did not include invalid charge skipped count"
        );
        ensure!(
            conversion_report.contains("skipped_low_peak_spectra,3"),
            "conversion report did not include low-peak skipped count"
        );
        Ok(())
    }

    /// Confirms spectra with more than 100 fragments are capped through `mascot-rs`.
    #[test]
    fn conversion_caps_fragment_peaks_to_top_100() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let input = temp.path().join("GeMS_A10.hdf5");
        write_many_peaks_fixture(&input)?;

        let config = Config {
            input_hdf5: input,
            output_dir: temp.path().join("converted"),
            chunk_size: 1,
            start_row: 0,
            limit: None,
            validate_output: true,
            publish_to_zenodo: false,
        };

        let report = convert_gems_a10(&config)?;
        ensure!(report.spectra_written == 1, "unexpected written count");
        ensure!(report.spectra_skipped == 0, "unexpected skipped count");

        let output_path = config.output_dir.join("GeMS_A10.mgf.zst");
        let mut parsed: MGFPathIter<usize, f64> = MGFIter::from_path(output_path)?;
        let first = parsed
            .next()
            .transpose()?
            .context("missing first parsed MGF")?;
        ensure!(
            first.len() == MAX_FRAGMENT_PEAKS,
            "spectrum was not capped to the configured peak limit"
        );

        let first_mz = first
            .peaks()
            .next()
            .map(|peak| peak.0)
            .context("missing first retained peak")?;
        let last_mz = first
            .peaks()
            .last()
            .map(|peak| peak.0)
            .context("missing last retained peak")?;
        ensure!(
            (first_mz - 78.0f64).abs() < f64::EPSILON,
            "lowest retained m/z was not from the top-intensity set"
        );
        ensure!(
            (last_mz - 177.0f64).abs() < f64::EPSILON,
            "highest retained m/z was not from the top-intensity set"
        );
        Ok(())
    }

    /// Converts a bounded row range without changing row identifiers.
    #[test]
    fn limit_and_start_row_bound_conversion() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let input = temp.path().join("GeMS_A10.hdf5");
        write_fixture(&input)?;

        let config = Config {
            input_hdf5: input,
            output_dir: temp.path().join("converted"),
            chunk_size: 2,
            start_row: 3,
            limit: Some(1),
            validate_output: true,
            publish_to_zenodo: false,
        };

        let report = convert_gems_a10(&config)?;
        ensure!(report.spectra_written == 1, "unexpected written count");
        ensure!(report.spectra_skipped == 0, "unexpected skipped count");
        let manifest = report.manifest.as_ref().context("missing manifest row")?;
        ensure!(manifest.start_row == 3, "unexpected start row");
        ensure!(manifest.end_row == 3, "unexpected end row");
        Ok(())
    }

    /// Confirms checksum generation covers the document and metadata files.
    #[test]
    fn checksums_include_document_and_metadata() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let document = temp.path().join("GeMS_A10.mgf.zst");
        fs::write(&document, b"example")?;
        fs::write(temp.path().join("manifest.csv"), b"dataset,path\n")?;
        fs::write(temp.path().join("README.txt"), b"readme\n")?;
        fs::write(temp.path().join("conversion_report.csv"), b"field,value\n")?;
        fs::write(
            temp.path().join("splash_duplicates.csv"),
            b"dataset,splash\n",
        )?;

        let checksum_path = write_sha256sums(temp.path())?;
        let checksums = fs::read_to_string(checksum_path)?;
        ensure!(
            checksums.contains("GeMS_A10.mgf.zst"),
            "document checksum missing"
        );
        ensure!(
            checksums.contains("manifest.csv"),
            "manifest checksum missing"
        );
        ensure!(checksums.contains("README.txt"), "README checksum missing");
        ensure!(
            checksums.contains("conversion_report.csv"),
            "conversion report checksum missing"
        );
        ensure!(
            checksums.contains("splash_duplicates.csv"),
            "duplicate report checksum missing"
        );
        Ok(())
    }

    /// Confirms checksum generation fails when an expected artifact is absent.
    #[test]
    fn checksums_require_all_expected_artifacts() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        fs::write(temp.path().join("GeMS_A10.mgf.zst"), b"example")?;
        fs::write(temp.path().join("manifest.csv"), b"dataset,path\n")?;
        fs::write(temp.path().join("README.txt"), b"readme\n")?;
        fs::write(temp.path().join("conversion_report.csv"), b"field,value\n")?;

        ensure!(
            write_sha256sums(temp.path()).is_err(),
            "checksums should fail when the duplicate report is missing"
        );
        Ok(())
    }

    /// Confirms direct API calls reject a zero chunk size before conversion.
    #[test]
    fn conversion_rejects_zero_chunk_size() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let config = Config {
            input_hdf5: temp.path().join("missing.hdf5"),
            output_dir: temp.path().join("converted"),
            chunk_size: 0,
            start_row: 0,
            limit: None,
            validate_output: false,
            publish_to_zenodo: false,
        };

        ensure!(
            convert_gems_a10(&config).is_err(),
            "zero chunk size should be rejected"
        );
        Ok(())
    }

    /// Confirms the converter rejects non-GeMS-A10 spectrum tensor widths.
    #[test]
    fn conversion_rejects_wrong_peak_dimension() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let input = temp.path().join("GeMS_A10.hdf5");
        write_wrong_peak_dimension_fixture(&input)?;

        let config = Config {
            input_hdf5: input,
            output_dir: temp.path().join("converted"),
            chunk_size: 1,
            start_row: 0,
            limit: None,
            validate_output: false,
            publish_to_zenodo: false,
        };

        ensure!(
            convert_gems_a10(&config).is_err(),
            "wrong spectrum peak dimension should be rejected"
        );
        Ok(())
    }

    /// Confirms physically invalid positive m/z values are filtered, not fatal.
    #[test]
    fn conversion_filters_physical_mz_bounds() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let input = temp.path().join("GeMS_A10.hdf5");
        write_invalid_mz_fixture(&input)?;

        let config = Config {
            input_hdf5: input,
            output_dir: temp.path().join("converted"),
            chunk_size: 1,
            start_row: 0,
            limit: None,
            validate_output: true,
            publish_to_zenodo: false,
        };

        let report = convert_gems_a10(&config)?;
        ensure!(report.spectra_written == 1, "unexpected written count");
        ensure!(report.spectra_skipped == 1, "unexpected skipped count");
        ensure!(
            report.skipped_invalid_precursor_mz == 1,
            "unexpected invalid precursor count"
        );

        let output_path = config.output_dir.join("GeMS_A10.mgf.zst");
        let mut parsed: MGFPathIter<usize, f64> = MGFIter::from_path(output_path)?;
        let first = parsed
            .next()
            .transpose()?
            .context("missing first parsed MGF")?;
        ensure!(
            first.len() == 2,
            "invalid fragment m/z values were not removed"
        );
        ensure!(
            first
                .peaks()
                .all(|(mz, intensity)| valid_mz(mz) && finite_positive(intensity)),
            "written spectrum contains a physically invalid peak"
        );
        Ok(())
    }

    /// Confirms fixed-width ASCII and UTF-8 HDF5 strings decode correctly.
    #[test]
    fn reads_fixed_width_string_columns() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let input = temp.path().join("strings.hdf5");
        let h5 = H5File::create(&input)?;
        h5.new_dataset_builder()
            .with_data(&[
                FixedAscii::<8>::from_ascii(b"file-a")?,
                FixedAscii::<8>::from_ascii(b"file-b")?,
            ])
            .create(FILE_NAME)?;
        h5.new_dataset_builder()
            .with_data(&[
                FixedUnicode::<8>::from_str("abc")?,
                FixedUnicode::<8>::from_str("def")?,
            ])
            .create(LSH)?;

        ensure!(
            read_string_slice(&h5.dataset(FILE_NAME)?, 0, 2)?
                == vec!["file-a".to_owned(), "file-b".to_owned()],
            "fixed ASCII strings were not decoded correctly"
        );
        ensure!(
            read_string_slice(&h5.dataset(LSH)?, 0, 2)? == vec!["abc".to_owned(), "def".to_owned()],
            "fixed UTF-8 strings were not decoded correctly"
        );
        Ok(())
    }

    /// Writes a synthetic GeMS-like HDF5 fixture.
    fn write_fixture(path: &Path) -> anyhow::Result<()> {
        let h5 = H5File::create(path)?;
        let mut spectra = Array3::<f64>::zeros((7, 2, EXPECTED_FRAGMENT_PEAKS));
        set_spectrum_value(&mut spectra, (0, 0, 0), 50.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (0, 0, 1), 0.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 1), 1.0f64)?;
        set_spectrum_value(&mut spectra, (0, 0, 2), 75.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 2), 2.5f64)?;
        set_spectrum_value(&mut spectra, (1, 0, 0), 25.0f64)?;
        set_spectrum_value(&mut spectra, (1, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (3, 0, 0), 50.0f64)?;
        set_spectrum_value(&mut spectra, (3, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (3, 0, 1), 75.0f64)?;
        set_spectrum_value(&mut spectra, (3, 1, 1), 2.5f64)?;
        set_spectrum_value(&mut spectra, (4, 0, 0), 90.0f64)?;
        set_spectrum_value(&mut spectra, (4, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (5, 0, 0), 110.0f64)?;
        set_spectrum_value(&mut spectra, (5, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (5, 0, 1), 125.0f64)?;
        set_spectrum_value(&mut spectra, (5, 1, 1), 1.5f64)?;
        set_spectrum_value(&mut spectra, (6, 0, 0), 140.0f64)?;
        set_spectrum_value(&mut spectra, (6, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (6, 0, 1), 140.0f64)?;
        set_spectrum_value(&mut spectra, (6, 1, 1), 2.0f64)?;
        h5.new_dataset_builder()
            .with_data(&spectra)
            .create(SPECTRUM)?;

        h5.new_dataset_builder()
            .with_data(&[
                100.125f64,
                f64::NAN,
                300.0f64,
                400.0f64,
                500.0f64,
                600.0f64,
                700.0f64,
            ])
            .create(PRECURSOR_MZ)?;
        h5.new_dataset_builder()
            .with_data(&[1i8, 2i8, 1i8, -1i8, 1i8, 0i8, 1i8])
            .create(CHARGE)?;
        h5.new_dataset_builder()
            .with_data(&[
                12.5f64, 20.0f64, 30.0f64, -4.0f64, 40.0f64, 50.0f64, 60.0f64,
            ])
            .create(RETENTION_TIME)?;
        h5.new_dataset_builder()
            .with_data(&[
                VarLenUnicode::from_str("file-a")?,
                VarLenUnicode::from_str("file-b")?,
                VarLenUnicode::from_str("file-c")?,
                VarLenUnicode::from_str("file-d")?,
                VarLenUnicode::from_str("file-e")?,
                VarLenUnicode::from_str("file-f")?,
                VarLenUnicode::from_str("file-g")?,
            ])
            .create(NAME)?;
        h5.new_dataset_builder()
            .with_data(&[101i64, 202i64, 303i64, 404i64, 505i64, 606i64, 707i64])
            .create(LSH)?;
        h5.new_dataset_builder()
            .with_data(&[
                0.001_25f64,
                0.002_5f64,
                0.003_75f64,
                f64::NAN,
                0.005f64,
                0.006f64,
                0.007f64,
            ])
            .create(ACCURACY)?;
        Ok(())
    }

    /// Writes a one-row HDF5 fixture with more than `MAX_FRAGMENT_PEAKS` peaks.
    fn write_many_peaks_fixture(path: &Path) -> anyhow::Result<()> {
        let h5 = H5File::create(path)?;
        let peak_total = EXPECTED_FRAGMENT_PEAKS;
        let mut spectra = Array3::<f64>::zeros((1, 2, peak_total));
        for peak_index in 0..peak_total {
            let peak_index_f64 = f64::from(u32::try_from(peak_index)?);
            set_spectrum_value(&mut spectra, (0, 0, peak_index), 50.0f64 + peak_index_f64)?;
            set_spectrum_value(&mut spectra, (0, 1, peak_index), 1.0f64 + peak_index_f64)?;
        }
        h5.new_dataset_builder()
            .with_data(&spectra)
            .create(SPECTRUM)?;

        h5.new_dataset_builder()
            .with_data(&[100.125f64])
            .create(PRECURSOR_MZ)?;
        h5.new_dataset_builder().with_data(&[1i8]).create(CHARGE)?;
        h5.new_dataset_builder()
            .with_data(&[12.5f64])
            .create(RETENTION_TIME)?;
        h5.new_dataset_builder()
            .with_data(&[VarLenUnicode::from_str("file-a")?])
            .create(NAME)?;
        h5.new_dataset_builder().with_data(&[101i64]).create(LSH)?;
        h5.new_dataset_builder()
            .with_data(&[0.001_25f64])
            .create(ACCURACY)?;
        Ok(())
    }

    /// Writes a fixture with the wrong HDF5 spectrum peak dimension.
    fn write_wrong_peak_dimension_fixture(path: &Path) -> anyhow::Result<()> {
        let h5 = H5File::create(path)?;
        let mut spectra = Array3::<f64>::zeros((1, 2, 4));
        set_spectrum_value(&mut spectra, (0, 0, 0), 50.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (0, 0, 1), 75.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 1), 2.5f64)?;
        h5.new_dataset_builder()
            .with_data(&spectra)
            .create(SPECTRUM)?;
        write_one_row_metadata(&h5, 100.125f64, 1i8, "file-a", 101i64)?;
        Ok(())
    }

    /// Writes a fixture containing positive but physically invalid m/z values.
    fn write_invalid_mz_fixture(path: &Path) -> anyhow::Result<()> {
        let h5 = H5File::create(path)?;
        let mut spectra = Array3::<f64>::zeros((2, 2, EXPECTED_FRAGMENT_PEAKS));
        set_spectrum_value(&mut spectra, (0, 0, 0), 50.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 0), 1.0f64)?;
        set_spectrum_value(&mut spectra, (0, 0, 1), 75.0f64)?;
        set_spectrum_value(&mut spectra, (0, 1, 1), 2.5f64)?;
        set_spectrum_value(&mut spectra, (1, 0, 0), ELECTRON_MASS / 2.0f64)?;
        set_spectrum_value(&mut spectra, (1, 1, 0), 10.0f64)?;
        set_spectrum_value(&mut spectra, (1, 0, 1), MAX_MZ + 1.0f64)?;
        set_spectrum_value(&mut spectra, (1, 1, 1), 11.0f64)?;
        set_spectrum_value(&mut spectra, (1, 0, 2), 100.0f64)?;
        set_spectrum_value(&mut spectra, (1, 1, 2), 1.0f64)?;
        set_spectrum_value(&mut spectra, (1, 0, 3), 125.0f64)?;
        set_spectrum_value(&mut spectra, (1, 1, 3), 2.0f64)?;
        h5.new_dataset_builder()
            .with_data(&spectra)
            .create(SPECTRUM)?;

        h5.new_dataset_builder()
            .with_data(&[MAX_MZ + 1.0f64, 200.0f64])
            .create(PRECURSOR_MZ)?;
        h5.new_dataset_builder()
            .with_data(&[1i8, 1i8])
            .create(CHARGE)?;
        h5.new_dataset_builder()
            .with_data(&[12.5f64, 20.0f64])
            .create(RETENTION_TIME)?;
        h5.new_dataset_builder()
            .with_data(&[
                VarLenUnicode::from_str("file-a")?,
                VarLenUnicode::from_str("file-b")?,
            ])
            .create(NAME)?;
        h5.new_dataset_builder()
            .with_data(&[101i64, 202i64])
            .create(LSH)?;
        h5.new_dataset_builder()
            .with_data(&[0.001_25f64, 0.002_5f64])
            .create(ACCURACY)?;
        Ok(())
    }

    /// Writes the one-row metadata columns shared by small HDF5 fixtures.
    fn write_one_row_metadata(
        h5: &H5File,
        precursor_mz: f64,
        charge: i8,
        file_name: &str,
        lsh: i64,
    ) -> anyhow::Result<()> {
        h5.new_dataset_builder()
            .with_data(&[precursor_mz])
            .create(PRECURSOR_MZ)?;
        h5.new_dataset_builder()
            .with_data(&[charge])
            .create(CHARGE)?;
        h5.new_dataset_builder()
            .with_data(&[12.5f64])
            .create(RETENTION_TIME)?;
        h5.new_dataset_builder()
            .with_data(&[VarLenUnicode::from_str(file_name)?])
            .create(NAME)?;
        h5.new_dataset_builder().with_data(&[lsh]).create(LSH)?;
        h5.new_dataset_builder()
            .with_data(&[0.001_25f64])
            .create(ACCURACY)?;
        Ok(())
    }

    /// Sets one spectrum fixture value through checked indexing.
    fn set_spectrum_value(
        spectra: &mut Array3<f64>,
        index: (usize, usize, usize),
        value: f64,
    ) -> anyhow::Result<()> {
        let element = spectra
            .get_mut(index)
            .context("fixture index out of bounds")?;
        *element = value;
        Ok(())
    }
}
