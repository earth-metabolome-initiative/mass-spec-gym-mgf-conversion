# mass-spec-gym-mgf-conversion

[![CI](https://github.com/earth-metabolome-initiative/mass-spec-gym-mgf-conversion/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/mass-spec-gym-mgf-conversion/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19980668.svg)](https://doi.org/10.5281/zenodo.19980668)

Rust converter for the MassSpecGym auxiliary GeMS-A10 spectral collection. It
turns the GeMS-A10 HDF5 file into compressed Mascot Generic Format documents
and can publish the generated artifacts to Zenodo.

## Source

GeMS-A10 is the unlabeled 24M-spectrum auxiliary collection from the DreaMS/GeMS
Hugging Face dataset:

- Dataset: <https://huggingface.co/datasets/roman-bushuiev/GeMS>
- File: `data/GeMS_A/GeMS_A10.hdf5`
- Direct download:
  <https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/GeMS_A/GeMS_A10.hdf5?download=true>

Expected local input: `data/data/GeMS_A/GeMS_A10.hdf5`.

## Run

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

Set `ZENODO_TOKEN` to publish to Zenodo. Without it, the run only writes local
artifacts.

## Conversion

The output is `converted/GeMS_A10/GeMS_A10.mgf.part-*.mgf.zst`, with 1,000,000
input rows per part. The conversion filters invalid spectra, caps each spectrum
to 100 peaks, and removes SPLASH duplicates.
