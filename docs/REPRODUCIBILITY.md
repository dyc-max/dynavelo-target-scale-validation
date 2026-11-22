# Reproducibility scope

## What can be checked without scientific re-analysis

1. Check the numbered archive payloads against `FILE_MANIFEST.tsv` and the deposited files against `SHA256SUMS.tsv`.
2. Inspect all complete source tables and archived model/score parameters.
3. Verify checkpoint identities against S3 and `model_index.json`.
4. Inspect all 9,018 completed GRNBoost2 outputs and committed fit records.
5. Re-render figures from fixed source tables in a new output directory.

The packaging validation covers byte integrity, file inventory, Python syntax,
model/axis metadata, archive safety and published-source attribution. It does
not independently rerun training, derivatives, normalization statistics,
GRNBoost2, the seed-sign audit or external differential models.

## Historical scientific workflow

Archival code is byte-identical to the selected local source files. The original
relative `repo/`, `work/` and `results/` structure is preserved. Data archives
restore the same relative paths beneath `DynaVelo_Research`. To inspect the
original layout, overlay `archive/project` onto a separate extracted copy.
Several scripts hard-code the original project root and source-bound manifests;
relocation changes source hashes and requires deliberate path/manifest adaptation.
No upstream runner is invoked by quick-start commands or CI.

Training used different original runtimes: CUDA for seed 0 and XPU for seeds 1/2.
Per-seed package versions and selected epochs are in `model_index.json` and the
original configurations. CPU/XPU locks and CUDA requirements are separate
historical environments; they should not be merged into one install command.
Environment locks may contain OS-specific or historical package builds.

All five central training source files matched hashes retained in the training
configurations. Some historical compiled helper/runner hashes remain recorded,
but matching compiled executables are not provided. Final model checkpoints are
preserved; exact training-process replay across devices is not certified.

## Deliberate archive boundaries

- Only the three factual checkpoints used in the manuscript are included;
  control-model runs, intermediate checkpoints and failed/retry job directories
  are not presented as additional final models.
- Primary sensitivity matrices retain every input and target; full-population
  candidate readouts retain all measured targets, routes and groups. Per-cell
  derivative tensors from the historical 19-GB numerical run are not included.
- The numerical qualification archive is included, but its completed checks do
  not establish CPU-to-A100 equivalence or transfer full-population qualification
  to the separate primary matrix.
- GRNBoost2 archives contain full predictor-level importance outputs and fit
  metadata, not serialized estimator/tree objects.
- Historical DC target-set deletion summaries have incomplete per-candidate tabulations. The later manuscript Supplementary Information (S4.6) documents the verified source coordinates for the T-cell effects.
- No raw FASTQ/BAM, clinical identifiers, cloud credentials, job-control tokens or
  account configuration files are intentionally included.

## Publication links

The [frozen code release](https://github.com/dyc-max/dynavelo-target-scale-validation/releases/tag/v1.1.0) accompanies [Zenodo data archive 1.1.0](https://doi.org/10.5281/zenodo.24437365). The version includes the [Section S11 scripts and tables](https://github.com/dyc-max/dynavelo-target-scale-validation/releases/download/v1.1.0/DynaVelo_Section_S11_Reproducibility.zip). Its README describes a replay using the packaged inputs for the perturbation benchmark, overlap test and cell-cycle analysis.

The Zenodo file `07_source_tables_and_scripts.zip` preserves the frozen figure-source tables and numerical provenance scripts. The `06_external_validation.zip` payload is available as six transport parts. Follow the [reassembly instructions](https://zenodo.org/records/24437365/files/EXTERNAL_VALIDATION_TRANSPORT.md?download=1) before invoking the original deposit verifier.
