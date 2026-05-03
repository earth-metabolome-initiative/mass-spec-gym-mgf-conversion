//! Shared dataset and publication metadata constants.

/// Dataset label written into `GeMS` metadata fields.
pub const DATASET_NAME: &str = "GeMS-A10";
/// Source Hugging Face dataset URL.
pub const SOURCE_DATASET_URL: &str = "https://huggingface.co/datasets/roman-bushuiev/GeMS";
/// Source GeMS-A10 HDF5 path inside the Hugging Face dataset.
pub const SOURCE_FILE_PATH: &str = "data/GeMS_A/GeMS_A10.hdf5";
/// Source GeMS-A10 HDF5 file page URL.
pub const SOURCE_FILE_URL: &str =
    "https://huggingface.co/datasets/roman-bushuiev/GeMS/blob/main/data/GeMS_A/GeMS_A10.hdf5";
/// Direct source GeMS-A10 HDF5 download URL.
pub const SOURCE_DIRECT_DOWNLOAD_URL: &str = "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/GeMS_A/GeMS_A10.hdf5?download=true";
/// `MassSpecGym` repository URL.
pub const MASS_SPEC_GYM_REPOSITORY_URL: &str = "https://github.com/pluskal-lab/MassSpecGym";
/// Conversion crate repository URL.
pub const CONVERTER_REPOSITORY_URL: &str =
    "https://github.com/earth-metabolome-initiative/mass-spec-gym-mgf-conversion";
/// `DreaMS` paper DOI.
pub const DREAMS_PAPER_DOI: &str = "10.1038/s41587-025-02663-3";
