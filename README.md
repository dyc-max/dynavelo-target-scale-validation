# DynaVelo target-scale diagnosis and validation

Code and documentation accompanying **Target-scale bias diagnosis and orthogonal
validation for directional gene regulatory inference from single-cell kinetic
sensitivity**. Author: **DuanYuChen**. Release: **1.0.0**.

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

Weights and full numerical data belong in the associated Zenodo deposit. The code repository is https://github.com/dyc-max/dynavelo-target-scale-validation. The data deposit DOI is
https://doi.org/10.5281/zenodo.24305323; DOI registration completes when the Zenodo record is published.

## Inspect the frozen data

Download the Zenodo archives and `FILE_MANIFEST.tsv` into a directory, then:

```bash
python tools/verify_deposit.py --deposit /path/to/downloads
```

This checks the recorded SHA-256 of every archived payload by streaming bytes.
It does not import the model, unpickle weights, fit statistics or calculate new
scientific results. It also checks for missing archives, duplicate and unsafe
archive members. Extract the archives into one chosen data directory; they all
share the top-level directory `DynaVelo_Research`.

## Render the four figures from frozen tables

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements-figures.txt
python tools/render_frozen_figures.py --source-data /path/to/data/DynaVelo_Research/manuscript/Source_Data --output /path/to/new/figure-output
```

The output directory must be new. This is figure rendering, not model inference
or re-fitting differential expression. Fonts can affect PNG pixels across systems;
the deposited PDF/SVG/PNG files are the authoritative manuscript figures.
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
with deterministic index position 49. DC target-deletion rankings and the T-cell
numerical contrast retain the source-detail limits described in the manuscript.
No claim of one-to-one candidate exclusivity is required or made.

## License and citation

Study code: MIT, copyright 2026 DuanYuChen. Study-generated data and weights:
CC BY 4.0 in the Zenodo deposit. Original DynaVelo code and reused public data
retain their respective attribution and source terms. Cite both this release
and Karbalayghareh et al., Nature Biotechnology (2026),
https://doi.org/10.1038/s41587-026-03348-1. The associated dataset DOI is https://doi.org/10.5281/zenodo.24305323.
