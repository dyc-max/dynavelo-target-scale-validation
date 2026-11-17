# Data dictionary and completeness

## Complete primary sensitivity matrices

`weighted_signed_mean_seed0.npy`, `seed1.npy`, `seed2.npy` are float64 arrays,
shape (1191, 3006). Row = RNA input regulator; column = RNA target. `axes.json`
provides zero-based indices, Ensembl stable identifiers and symbols. All three
matrices use the same axes. Their aggregation domain is the primary 192-cell
sample, weighted to the 1,884-cell evaluation population.

Values are signed finite-difference RNA-velocity sensitivities per stored RNA
input unit. They are not expression log2 fold changes. Per-cell signed values
were weighted before seed aggregation; opposite cell signs may cancel.
The raw ranking magnitude is the median across seeds of the absolute weighted
signed means. Column L1 divides by the total over 1,191 inputs; column percentile
uses ascending tie-average rank (r − 0.5)/1191, with all-zero columns set to zero.

## Full-population and primary numerical readouts

The 112 descriptive NPZ files retain both sampling scopes, 28 inputs (20 RNA,
8 motif), both output channels and all 13 original groups. `unique_weighted_means`
is organized as seed × numerical route × group × target; see
`ARRAY_HEADER_CATALOG.json` for exact stored shapes. RNA and motif units are
distinct. The 448 numerical-support NPZ files retain the source route-comparison
arrays and qualification fields. These are complete for that candidate scope,
not a full 1,191-input derivative evaluation over all 1,884 cells.

Group indices, target axes, input indices, weight sums and route definitions are
in the archived plan and selector records in 08_provenance. Original status
codes remain in source data for traceability; they are not new biological claims.

## Discovery inputs

`RNA_X.npy`: float32, shape (6035, 3006), exact stored discovery RNA expression
without a new normalization step. `adata_rna.h5ad` and `adata_atac.h5ad` are the
model-ready RNA and auxiliary inputs. The latter contains 365 motif features.
Axes include cell identifiers, original broad labels, training/validation split,
RNA stable identifiers, symbols and predictor/motif mappings. Identifiers refer
to public processed library 8489-MV-1 (GSE284047), not newly collected subjects.

## GRNBoost2

9,018 completed seed–target fits cover 3 seeds × 3,006 targets. Individual
`all_predictors.tsv` tables retain complete predictor-level scores and zeros;
the per-fit metadata record the target/seed identity and fitted-tree count.
Official importance and tree-count-normalized importance are separate quantities.
The `family.json`, completion record and commit census establish the fitted
universe. There are no newly fitted models in this deposit.

## External validation and manuscript tables

External matrices retain gene-by-sample expression/counts, sample/control mapping,
the full tested-gene estimates and adjusted P values. See prepared sample tables
and S4 for independent cultures, animals or pools. Columns named log2_effect or
logFC represent perturbation minus control; signed sensitivity uses model units.
Confidence intervals and P values are inherited from frozen analyses. Missing
fields remain empty/NA; zero scores are valid numeric values, not missing data.

Manuscript `Source_Data` contains S1–S6, the four-figure data, complete 20-candidate
target scores, numerical source provenance and source-detail limitations.
`S5_Alternative_DC_contexts.csv` and `S5_DC_cross_study_gene_details.csv` are
identical historical aliases, not independent datasets or deletion-score tables.
The 0.0857 permutation P refers to the unremoved 27-target primary DC comparison.
The T-cell point estimate is an archived author-result summary; no original
author export row or independently recoverable exact donor aggregation is supplied.

## Axis and precision conventions

No axes are re-sorted and no numerical values are rounded during packaging.
CSV/TSV retain their original delimiters/encodings and NPY/NPZ/H5AD their bytes.
All included payloads have SHA-256 checksums in FILE_MANIFEST.tsv. Archive-member
names preserve the original project-relative structure under DynaVelo_Research.
