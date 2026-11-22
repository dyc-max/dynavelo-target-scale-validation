# DynaVelo target-scale diagnosis and validation

Code and documentation accompanying **Target scale shapes which gene relationships
are prioritized by single-cell kinetic sensitivity**. Yuchen Duan is listed as
DuanYuChen in the archive. Code release 1.1.0.

This project diagnoses target-scale dominance in existing DynaVelo sensitivity
scores, compares raw and column-normalized rankings with GRNBoost2, and relates
representative gene pairs to context-specific perturbation responses. It builds
on [DynaVelo](https://github.com/karbalayghareh/DynaVelo), rather than introducing
a replacement for the underlying model.

## Contents

- `tools/`: portable, explicitly invoked artifact verification and figure rendering.
- `archive/project/`: archived study training, derivative, screening, GRNBoost2,
  normalization and external-analysis code, in its original relative layout.
- `archive/read_only_sign_audit.py`: the existing seed-sign audit implementation.
- `environments/`: archived CPU, XPU and CUDA dependency records.
- `docs/`: data dictionary, stage map, checkpoint/source bindings and replay limits.
- `LICENSE`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`: authorship, rights and citation.

[Zenodo data archive 1.1.0](https://doi.org/10.5281/zenodo.24437365) provides the frozen model weights, matrices and numerical results. The [Section S11 archive](https://github.com/dyc-max/dynavelo-target-scale-validation/releases/download/v1.1.0/DynaVelo_Section_S11_Reproducibility.zip) provides scripts and machine-readable Tables S11a–S11c for the perturbation benchmark, overlap permutations and cell-cycle sensitivity analysis. It is also available from Zenodo.

The Zenodo archive `07_source_tables_and_scripts.zip` contains the frozen figure source tables, supplementary results, and numerical provenance scripts. See [Reproducibility scope](docs/REPRODUCIBILITY.md) for the distinction between the archived study files and the standalone S11 replay.

## Inspect the frozen data

Download the Zenodo archives and checksum manifests into a directory. Follow the [transport instructions](https://zenodo.org/records/24437365/files/EXTERNAL_VALIDATION_TRANSPORT.md?download=1) to reconstruct `06_external_validation.zip` from its six parts before running the verifier. Extract the S11 archive separately and follow its own README. Verify the numbered archive payloads and file hashes with this command.

```bash
python tools/verify_deposit.py --deposit /path/to/downloads
```

This checks the recorded SHA-256 of every archived payload by streaming bytes.
It does not import the model, unpickle weights, fit statistics or calculate new
scientific results. It also checks for missing archives, duplicate and unsafe
archive members. Extract the archives into one chosen data directory; they all
share the top-level directory `DynaVelo_Research`.

## Render the four archived figures from frozen tables

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements-figures.txt
python tools/render_frozen_figures.py --source-data /path/to/data/DynaVelo_Research/manuscript/Source_Data --output /path/to/new/figure-output
```

The output directory must be new. This is figure rendering, not model inference
or re-fitting differential expression. Fonts can affect PNG pixels across systems;
compare the regenerated figures against the manuscript figures.
The portable wrapper changes output/input handling and omits the original
machine-specific authoring QA helper; scientific plotting values are unchanged.

## Scope of the data and code

The complete primary matrices have **1,191 input regulators × 3,006 targets for
each of three seeds**, aggregated over 192 sampled cells. The separate full-cell
readouts cover **1,884 cells for the existing 20 RNA and eight motif inputs**,
with all targets and 13 original groups retained. They are not a 1,191-input,
1,884-cell derivative census. GRNBoost2 outputs cover all 3,006 targets and
three seeds (9,018 fits), with all predictor scores, including zeros.

The archival scientific scripts preserve their original bytes and may reference
the original Windows project root, manifests, authorization records and hardware
environments. They are supplied for method inspection and provenance; this
release does **not** assert that the original distributed workflow is a turnkey
cross-platform training pipeline. See `docs/REPRODUCIBILITY.md` before attempting
any upstream replay. The packaging operation did not execute these scripts.

Representative perturbation agreement is context-dependent. CEBPA–MPO has mixed
seed signs (two positive, one negative) and belongs to a 56-way highest-score tie,
with deterministic index position 49. Additional DC target-set comparisons and the verified T-cell source coordinates are described in the Supplementary Information.
No claim of one-to-one candidate exclusivity is required or made.

## License and citation

Study code: MIT, copyright 2026 DuanYuChen. Study-generated data and weights:
CC BY 4.0 in the Zenodo deposit. Original DynaVelo code and reused public data
retain their respective attribution and source terms. Cite both this release
and Karbalayghareh et al., Nature Biotechnology (2026),
https://doi.org/10.1038/s41587-026-03348-1. The data archive with the Section S11 files is https://doi.org/10.5281/zenodo.24437365.
