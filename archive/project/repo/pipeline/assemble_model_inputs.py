"""Align real RNA/velocity and chromVAR inputs; no model fitting or discovery.

Run after activate_env.ps1. All explicit paths must be absolute inside the
authorized D-drive root. The fixed cell split is an internal, transductive
single-library check, not independent biological validation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

from dynavelo_cpu_adapter import PROJECT_ROOT, project_path

BASE = PROJECT_ROOT / "data/processed/GSE284047/8489-MV-1"


def absolute_project_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"An absolute D-drive project path is required: {path}")
    return project_path(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = absolute_project_path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_h5ad_atomic(adata, path):
    path = absolute_project_path(path)
    temporary = path.with_name(path.stem + ".partial.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", type=absolute_project_path, default=BASE / "adata_rna_velocity.h5ad")
    parser.add_argument("--zscores", type=absolute_project_path, default=BASE / "chromvar_zscores.tsv.gz")
    parser.add_argument("--motif-metadata", type=absolute_project_path, default=BASE / "chromvar_motif_metadata.tsv")
    parser.add_argument("--out-dir", type=absolute_project_path, default=BASE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.rna, args.zscores, args.motif_metadata):
        if not absolute_project_path(path).is_file():
            raise FileNotFoundError(f"Required real input is absent: {path}")
    out_dir = absolute_project_path(args.out_dir)
    output_names = ("adata_rna.h5ad", "adata_atac.h5ad", "motif_selection_mapping.tsv", "model_cell_alignment.tsv", "model_split.tsv", "model_TF_coverage.tsv", "model_input_assembly.json")
    if not args.overwrite:
        existing = [str(out_dir / name) for name in output_names if (out_dir / name).exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite assembled outputs: {existing}")

    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    rna = ad.read_h5ad(args.rna)
    if not rna.obs_names.is_unique or not rna.var_names.is_unique:
        raise ValueError("RNA cell IDs and human gene symbols must be unique")
    if any(not str(x) for x in rna.obs_names) or any(not str(x) for x in rna.var_names):
        raise ValueError("RNA IDs cannot be empty")
    if "cell_id" in rna.obs and not np.array_equal(rna.obs["cell_id"].astype(str), rna.obs_names.astype(str)):
        raise ValueError("RNA cell_id column disagrees with obs_names")
    for column in ("celltype", "latent_time"):
        if column not in rna.obs or rna.obs[column].isna().any():
            raise ValueError(f"Required complete RNA obs column: {column}")
    if "velocity" not in rna.layers:
        raise ValueError("scVelo velocity layer is required; counts cannot replace it")
    if not np.isfinite(np.asarray(rna.obs["latent_time"], dtype=float)).all():
        raise ValueError("RNA latent_time must be finite")
    x_values = rna.X.data if sparse.issparse(rna.X) else np.asarray(rna.X)
    if not np.isfinite(x_values).all() or (x_values < 0).any():
        raise ValueError("RNA X must be finite nonnegative log-normalized expression")

    opener = gzip.open if args.zscores.suffix == ".gz" else open
    with opener(args.zscores, "rt", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"))
    score_ids = header[1:]
    if len(score_ids) != len(set(score_ids)) or any(not value for value in score_ids):
        raise ValueError("chromVAR header has duplicate or empty cell IDs")
    scores = pd.read_csv(args.zscores, sep="\t", index_col=0)
    scores.index = scores.index.astype(str)
    if not scores.index.is_unique or list(scores.columns) != score_ids:
        raise ValueError("chromVAR motif rows must be unique and header must be unchanged")
    meta = pd.read_csv(args.motif_metadata, sep="\t", dtype=str, keep_default_na=False)
    required = {"TF", "motif_id", "output_motif_name"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing motif metadata fields: {required - set(meta.columns)}")
    if not meta["output_motif_name"].is_unique or (meta["output_motif_name"] == "").any():
        raise ValueError("Motif metadata output_motif_name must be unique and nonempty")

    rna_ids = list(rna.obs_names.astype(str))
    score_set = set(score_ids)
    missing_rna_ids = set(rna_ids) - score_set
    if missing_rna_ids:
        raise ValueError(f"Every RNA cell must have an exact chromVAR ID; missing {len(missing_rna_ids)} (examples: {sorted(missing_rna_ids)[:5]})")
    common = [cell for cell in rna_ids if cell in score_set]
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} exact paired cell IDs; at least 10 required")
    rna = rna[common].copy()
    rna.obs_names.name = "cell_id"
    rna.obs["cell_id"] = rna.obs_names.astype(str)
    rna.obs["celltype"] = pd.Categorical(rna.obs["celltype"]).remove_unused_categories()
    rna.X = rna.X.astype(np.float32)
    velocity = rna.layers["velocity"]
    velocity = velocity.toarray() if sparse.issparse(velocity) else np.asarray(velocity)
    velocity = velocity.astype(np.float32)
    if np.isinf(velocity).any() or not np.any(np.isfinite(velocity) & (velocity != 0)):
        raise ValueError("Velocity must contain finite nonzero values, no infinities; unknown NaNs are retained")
    rna.layers["velocity"] = velocity

    # Sort before sampling so file row order cannot silently alter the frozen split.
    sorted_ids = np.array(sorted(common), dtype=str)
    permutation = np.random.default_rng(0).permutation(len(common))
    train_set = set(sorted_ids[permutation[:int(np.floor(0.8 * len(common)))]] )
    train = np.array([cell in train_set for cell in common], dtype=bool)
    rna.obs["split"] = pd.Categorical(np.where(train, "train", "validation"), categories=["train", "validation"])
    values = scores.loc[:, common].to_numpy(dtype=np.float32)
    score_row = {name: row for row, name in enumerate(scores.index)}
    gene_symbols = set(rna.var_names)
    mapping = meta.copy()
    mapping["train_variance"] = np.nan
    mapping["finite_fraction_aligned"] = np.nan
    mapping["selection_status"] = "not_evaluated"
    mapping["selected"] = False
    mapping["selected_output_motif_for_TF"] = ""
    for index, row in mapping.iterrows():
        tf, name = row["TF"], row["output_motif_name"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tf):
            reason = "excluded_composite_or_noncanonical_TF_label"
        elif tf not in gene_symbols:
            reason = "excluded_no_exact_RNA_symbol"
        elif name not in score_row:
            reason = "excluded_no_score_row"
        else:
            vector = values[score_row[name]]
            mapping.at[index, "finite_fraction_aligned"] = float(np.isfinite(vector).mean())
            if not np.isfinite(vector).all():
                reason = "excluded_nonfinite_aligned_scores"
            else:
                variance = float(np.var(vector[train], dtype=np.float64, ddof=0))
                mapping.at[index, "train_variance"] = variance
                reason = "eligible" if variance > 0 else "excluded_zero_training_variance"
        mapping.at[index, "selection_status"] = reason
    eligible = mapping[mapping["selection_status"] == "eligible"]
    chosen = eligible.sort_values(["TF", "train_variance", "motif_id", "output_motif_name"], ascending=[True, False, True, True], kind="stable").drop_duplicates("TF").sort_values("TF", kind="stable")
    for index, row in chosen.iterrows():
        tf_rows = mapping["TF"] == row["TF"]
        mapping.loc[tf_rows, "selected_output_motif_for_TF"] = row["output_motif_name"]
        mapping.loc[tf_rows & (mapping["selection_status"] == "eligible"), "selection_status"] = "excluded_lower_train_variance_or_lexical_tiebreak"
        mapping.at[index, "selection_status"] = "selected"
        mapping.at[index, "selected"] = True
    orphan_names = sorted(set(scores.index) - set(meta["output_motif_name"]))
    if orphan_names:
        mapping = pd.concat([mapping, pd.DataFrame({"output_motif_name": orphan_names, "TF": "", "motif_id": "", "selection_status": "excluded_no_metadata", "selected": False, "selected_output_motif_for_TF": ""})], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out_dir / "motif_selection_mapping.tsv", sep="\t", index=False)
    if len(chosen) < 2:
        raise ValueError("Fewer than two unique TF motifs survive fixed input rules; see mapping")
    var = chosen[["TF", "motif_id", "output_motif_name", "train_variance"]].copy()
    var.index = pd.Index(var["TF"].to_numpy(), name="TF_id")
    atac = ad.AnnData(X=values[[score_row[name] for name in chosen["output_motif_name"]]].T.copy(), obs=rna.obs.copy(), var=var)
    # RNA-only TFs remain in the full RNA feature space. Motif availability
    # changes the later perturbation route, never the RNA inclusion rule.
    rna.var["has_selected_chromvar_motif"] = rna.var_names.isin(atac.var["TF"])
    rna.var["dynavelo_perturbation_modality"] = np.where(rna.var["has_selected_chromvar_motif"], "joint_RNA_motif", "RNA_only")
    coverage_columns = [name for name in ("is_curated_tf", "has_jaspar_motif", "perturbation_eligible_tf", "has_selected_chromvar_motif", "dynavelo_perturbation_modality") if name in rna.var]
    rna.var[coverage_columns].to_csv(out_dir / "model_TF_coverage.tsv", sep="\t", index_label="RNA_symbol")
    if not rna.obs_names.equals(atac.obs_names) or not atac.var["TF"].is_unique:
        raise AssertionError("Exact paired-cell / unique-TF contract failed")
    rna_set = set(rna_ids)
    pd.DataFrame([{"cell_id": cell, "in_RNA": cell in rna_set, "in_chromVAR": cell in score_set, "included": cell in rna_set and cell in score_set} for cell in sorted(rna_set | score_set)]).to_csv(out_dir / "model_cell_alignment.tsv", sep="\t", index=False)
    rna.obs[["split"]].to_csv(out_dir / "model_split.tsv", sep="\t", index_label="cell_id")
    report = {
        "status": "ASSEMBLED_INPUTS_QC_REVIEW_PENDING", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {str(path): sha256(path) for path in (args.rna, args.zscores, args.motif_metadata)},
        "script_sha256": sha256(Path(__file__)), "n_cells": rna.n_obs, "n_RNA_genes": rna.n_vars, "n_unique_TF": atac.n_vars,
        "RNA_cells_without_chromVAR": len(rna_set - score_set), "chromVAR_cells_without_RNA": len(score_set - rna_set),
        "split_seed": 0, "split_rule": "sorted exact cell IDs; numpy default_rng(0); floor(0.8*N) training; remainder validation",
        "n_train": int(train.sum()), "n_validation": int((~train).sum()),
        "motif_selector": "highest population variance in fixed training cells; tie by motif_id then output_motif_name ascending",
        "motif_status_counts": {str(key): int(value) for key, value in mapping["selection_status"].value_counts().items()},
        "gene_symbol_case_changed": False, "barcode_transform_applied": False,
        "RNA_feature_filter_applied": False, "RNA_only_TF_features_retained": True,
        "velocity_nan_entries_preserved": int(np.isnan(velocity).sum()),
        "scope": "Single-library internal cell split; upstream preprocessing may use all cells. Transductive check, not independent donor validation.",
        "external_validation": "NOT_PERFORMED", "model_training": "NOT_PERFORMED",
    }
    for obj in (rna, atac):
        obj.uns["model_input_assembly_json"] = json.dumps(report, ensure_ascii=False, allow_nan=False)
    write_h5ad_atomic(rna, out_dir / "adata_rna.h5ad")
    write_h5ad_atomic(atac, out_dir / "adata_atac.h5ad")
    report["output_h5ad_sha256"] = {str(out_dir / name): sha256(out_dir / name) for name in ("adata_rna.h5ad", "adata_atac.h5ad")}
    write_json(out_dir / "model_input_assembly.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
