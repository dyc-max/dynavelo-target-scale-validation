"""Independent Linux/CUDA HSPC runner; original upstream Jacobians and loss.

Generated from the reviewed Windows trainer. Only transport, storage and CUDA
placement differ. No analytic acceleration or automatic biological PASS.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import random
import shutil
import sys
import time
import traceback

from colab_runtime import BASE, absolute_project_path, sha256, write_h5ad_atomic, write_json, configure_cloud_storage, verify_transfer_manifest, verify_frozen_environment
from dynavelo_colab_adapter import PROJECT_ROOT, load_dynavelo
from colab_training import fit_resumable, preserve_resume_config, verify_checkpoint_restore_manifest, benchmark, make_download_manifest
import atexit


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", type=absolute_project_path, default=BASE / "adata_rna.h5ad")
    parser.add_argument("--atac", type=absolute_project_path, default=BASE / "adata_atac.h5ad")
    parser.add_argument("--prediction-root", type=absolute_project_path, default=BASE / "model_runs")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--zx-dim", type=int, default=50)
    parser.add_argument("--zy-dim", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--evaluation-chunk-cells", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--transfer-manifest", type=absolute_project_path, default=PROJECT_ROOT / "data/colab_transfer_manifest.json")
    parser.add_argument("--resume-checkpoint", type=absolute_project_path, default=None)
    parser.add_argument("--resume-manifest", type=absolute_project_path, default=None)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=2)
    args = parser.parse_args()
    if bool(args.resume_checkpoint) != bool(args.resume_manifest):
        parser.error("resume-checkpoint and resume-manifest must be provided together")
    if args.benchmark_batches < 1 or (args.benchmark_only and args.resume_checkpoint):
        parser.error("benchmark-batches must be positive; a benchmark cannot resume a research checkpoint")
    for field in ("max_epochs", "hidden", "zx_dim", "zy_dim", "n_samples", "threads"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.batch_size < 3 or not 3 <= args.evaluation_chunk_cells <= 256:
        parser.error("batch-size must be >=3; evaluation-chunk-cells must be 3..256 (avoid singleton squeeze)")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    for path in (args.rna, args.atac):
        if not absolute_project_path(path).is_file():
            raise FileNotFoundError(f"Required assembled real input is absent: {path}")
    return args


def batches_without_singleton(indices, size):
    """All cells exactly once; shift one cell if the last batch would be 1."""
    if len(indices) < 2 or size < 3:
        raise ValueError("At least two cells and a maximum batch size >=3 are required")
    batches = [indices[start:start + size] for start in range(0, len(indices), size)]
    if len(batches[-1]) == 1:
        batches[-1] = [batches[-2].pop()] + batches[-1]
    return batches


def dense32(value):
    import numpy as np
    from scipy import sparse
    return np.asarray(value.toarray() if sparse.issparse(value) else value, dtype=np.float32)


def subset_for_dataset(adata, indices):
    import pandas as pd
    result = adata[indices].copy()
    result.obs["celltype"] = pd.Categorical(result.obs["celltype"]).remove_unused_categories()
    return result


def validate_inputs(rna, atac):
    import numpy as np
    if not rna.obs_names.is_unique or not rna.var_names.is_unique or not atac.var_names.is_unique:
        raise ValueError("RNA/ATAC IDs must be unique")
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and motif cell IDs/order must match exactly; no automatic reordering")
    if rna.n_obs < 10 or min(rna.n_vars, atac.n_vars) < 2:
        raise ValueError("Insufficient paired cells or features")
    if "TF" not in atac.var or not atac.var["TF"].is_unique or not set(atac.var["TF"]).issubset(rna.var_names):
        raise ValueError("Motif TFs must be unique exact human RNA symbols")
    if list(atac.var["TF"].astype(str)) != list(atac.var_names.astype(str)):
        raise ValueError("ATAC feature IDs must be the exact unique TF symbols")
    for obj in (rna, atac):
        for column in ("celltype", "latent_time", "split"):
            if column not in obj.obs or obj.obs[column].isna().any():
                raise ValueError(f"Missing/incomplete obs field: {column}")
        if not np.isfinite(np.asarray(obj.obs["latent_time"], dtype=float)).all():
            raise ValueError("Input latent_time must be finite")
        obj.X = dense32(obj.X)
        if not np.isfinite(obj.X).all():
            raise ValueError("Model expression/activity inputs must be finite")
    if np.any(rna.X < 0):
        raise ValueError("RNA input must be nonnegative log-normalized expression")
    if "velocity" not in rna.layers:
        raise ValueError("RNA velocity prior layer is absent")
    rna.layers["velocity"] = dense32(rna.layers["velocity"])
    velocity = rna.layers["velocity"]
    if np.isinf(velocity).any() or not np.any(np.isfinite(velocity) & (velocity != 0)):
        raise ValueError("Velocity requires finite nonzero entries and no infinity; NaN is an unknown prior")
    split = np.asarray(rna.obs["split"].astype(str))
    if not np.array_equal(split, np.asarray(atac.obs["split"].astype(str))) or set(split) != {"train", "validation"}:
        raise ValueError("Frozen train/validation split differs between modalities")
    train, valid = np.flatnonzero(split == "train"), np.flatnonzero(split == "validation")
    if min(len(train), len(valid)) < 2:
        raise ValueError("Both fixed split groups need at least two cells")
    assembly = json.loads(rna.uns.get("model_input_assembly_json", "{}"))
    atac_assembly = json.loads(atac.uns.get("model_input_assembly_json", "{}"))
    if assembly.get("split_seed") != 0 or assembly != atac_assembly:
        raise ValueError("Matching input-assembly provenance and split_seed=0 are required")
    sorted_ids = np.array(sorted(rna.obs_names.astype(str)), dtype=str)
    permutation = np.random.default_rng(0).permutation(rna.n_obs)
    expected = set(sorted_ids[permutation[:int(np.floor(0.8 * rna.n_obs))]])
    if set(rna.obs_names[train]) != expected:
        raise ValueError("Input split no longer matches the frozen seed-0 rule")
    for indices in (train, valid):
        subset = velocity[indices]
        if not np.any(np.isfinite(subset) & (subset != 0)):
            raise ValueError("Each split needs some nonzero finite velocity prior")
    return train, valid, assembly


def reconstruction_metrics(observed, predicted, train, valid, chunk_cells):
    import numpy as np
    mean = np.zeros(observed.shape[1], dtype=np.float64)
    for start in range(0, len(train), chunk_cells):
        mean += np.sum(observed[train[start:start + chunk_cells]], axis=0, dtype=np.float64)
    mean /= len(train)
    model_sse, baseline_sse = 0.0, 0.0
    for start in range(0, len(valid), chunk_cells):
        indices = valid[start:start + chunk_cells]
        truth = np.asarray(observed[indices], dtype=np.float64)
        error = truth - np.asarray(predicted[indices], dtype=np.float64)
        model_sse += float(np.square(error).sum())
        baseline_sse += float(np.square(truth - mean).sum())
    entries = int(len(valid) * observed.shape[1])
    return {
        "n_validation_cells": len(valid), "n_features": observed.shape[1], "n_entries": entries,
        "model_MSE": model_sse / entries, "training_mean_baseline_MSE": baseline_sse / entries,
        "MSE_ratio_model_over_baseline": model_sse / baseline_sse if baseline_sse > 0 else None,
        "better_than_training_mean": bool(model_sse < baseline_sse) if baseline_sse > 0 else None,
        "baseline": "Per-feature arithmetic mean of fixed training cells only",
        "interpretation": "Internal validation-cell reconstruction, not independent generalization or biological convergence",
    }


def velocity_metrics(velocity, predicted, valid, cell_ids, chunk_cells, out_path):
    import numpy as np
    import pandas as pd
    rows = []
    for start in range(0, len(valid), chunk_cells):
        indices = valid[start:start + chunk_cells]
        truth = np.asarray(velocity[indices], dtype=np.float64)
        prediction = np.asarray(predicted[indices], dtype=np.float64)
        mask = np.isfinite(truth)
        truth = np.where(mask, truth, 0.0)
        prediction = np.where(mask, prediction, 0.0)
        denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
        for offset, cell in enumerate(indices):
            eligible = bool(denominator[offset] > 0 and mask[offset].any())
            cosine = float(np.dot(truth[offset], prediction[offset]) / denominator[offset]) if eligible else None
            rows.append({"cell_id": str(cell_ids[cell]), "n_finite_prior_genes": int(mask[offset].sum()), "eligible_nonzero_norm": eligible, "masked_cosine": cosine})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    values = np.array([row["masked_cosine"] for row in rows if row["masked_cosine"] is not None], dtype=float)
    return {
        "n_validation_cells": len(valid), "n_eligible_cells": len(values),
        "mean_masked_cosine": float(values.mean()) if len(values) else None,
        "median_masked_cosine": float(np.median(values)) if len(values) else None,
        "cell_statistics_file": str(out_path), "P_value": None,
        "interpretation": "Internal scVelo-prior agreement only. Prior/preprocessing can share cells; neither external validation nor a causal test.",
    }


def chunked_evaluate(model, Dataset, rna, atac, args, cache_dir, run_dir):
    """Bound upstream Monte Carlo arrays to a cell chunk, persist summaries."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    n_cells = rna.n_obs
    arrays = {}
    times = {name: np.zeros(n_cells, dtype=np.float64) for name in ("latent_time_mean", "latent_time_std")}
    blocks = batches_without_singleton(list(range(n_cells)), args.evaluation_chunk_cells)
    model.mode = "evaluation-sample"
    progress = []
    for block_number, indices in enumerate(blocks, start=1):
        before = time.monotonic()
        # Keep original annotation/graphs in final objects, not repeated per chunk.
        obs = rna.obs.iloc[indices].copy()
        obs["celltype"] = pd.Categorical(obs["celltype"]).remove_unused_categories()
        block_rna = ad.AnnData(X=rna.X[indices].copy(), obs=obs.copy(), var=rna.var.copy(), layers={"velocity": rna.layers["velocity"][indices].copy()})
        block_atac = ad.AnnData(X=atac.X[indices].copy(), obs=obs.copy(), var=atac.var.copy())
        loader = DataLoader(Dataset(block_rna, block_atac, use_weights=False), batch_size=len(indices), shuffle=False, drop_last=False, num_workers=0)
        pred_rna, pred_atac = model.evaluate(block_rna, block_atac, loader, n_samples=args.n_samples)
        for modality, prediction in (("rna", pred_rna), ("atac", pred_atac)):
            layer_names = ("x_pred_mean", "x_pred_std", "vx_pred_mean", "vx_pred_std") if modality == "rna" else ("y_pred_mean", "y_pred_std", "vy_pred_mean", "vy_pred_std")
            for field, keys in (("layers", layer_names), ("obsm", list(prediction.obsm.keys()))):
                container = getattr(prediction, field)
                for key in keys:
                    values = np.asarray(container[key])
                    if values.ndim != 2 or values.shape[0] != len(indices) or not np.isfinite(values).all():
                        raise FloatingPointError(f"Invalid {modality}/{field}/{key} prediction in chunk {block_number}")
                    identity = (modality, field, key)
                    if identity not in arrays:
                        path = cache_dir / ("__".join(identity) + ".npy")
                        arrays[identity] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_cells, values.shape[1]))
                    arrays[identity][indices] = values.astype(np.float32)
                    if not np.isfinite(arrays[identity][indices]).all():
                        raise FloatingPointError("Prediction overflow converting summary storage to float32")
                    arrays[identity].flush()
        for key in times:
            values = np.asarray(pred_rna.obs[key], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite {key}")
            if not np.array_equal(values, np.asarray(pred_atac.obs[key], dtype=np.float64)):
                raise ValueError("RNA/motif posterior times disagree within a chunk")
            times[key][indices] = values
        progress.append({"chunk": block_number, "start": indices[0], "stop_exclusive": indices[-1] + 1, "n_cells": len(indices), "elapsed_seconds": time.monotonic() - before})
        write_json(run_dir / "evaluation_progress.json", {"completed_chunks": progress, "total_chunks": len(blocks), "complete": block_number == len(blocks)})
        print(f"Evaluation chunk {block_number}/{len(blocks)} persisted ({len(indices)} cells)", flush=True)
        del pred_rna, pred_atac, block_rna, block_atac, loader

    # Upstream evaluate ranks each call. Those chunk-local ranks are discarded.
    order = np.argsort(times["latent_time_mean"], kind="stable")
    global_rank = np.empty(n_cells, dtype=np.float64)
    global_rank[order] = np.arange(n_cells, dtype=np.float64) / (n_cells - 1)
    for modality, obj in (("rna", rna), ("atac", atac)):
        obj.obs["latent_time_scvelo"] = obj.obs["latent_time"].copy()
        for key, values in times.items():
            obj.obs[key] = values.copy()
        obj.obs["latent_time"] = global_rank.copy()
        for (array_modality, field, key), values in arrays.items():
            if array_modality == modality:
                # ndarray view keeps disk backing but uses AnnData's ndarray
                # writer registration rather than relying on memmap dispatch.
                getattr(obj, field)[key] = np.asarray(values)
    return rna, atac, progress


def run(args, run_id, run_dir, prediction_dir, cache_dir, transfer_check):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU fallback")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    gpu = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gpu_info = {"device": "cuda:0", "name": gpu.name, "total_bytes": int(total_bytes),
                "free_bytes_before_training": int(free_bytes), "torch_cuda_version": torch.version.cuda,
                "torch_version": torch.__version__, "cudnn_version": torch.backends.cudnn.version(),
                "allow_tf32": False, "precision": "float32; no AMP; no analytic Jacobian replacement"}
    rna, atac = ad.read_h5ad(args.rna), ad.read_h5ad(args.atac)
    train, valid, assembly = validate_inputs(rna, atac)
    Dataset, DynaVelo, adapter_manifest = load_dynavelo(analytic_velocity=False)
    if adapter_manifest.get("analytic_velocity") is not False:
        raise RuntimeError("Research script requires the original upstream Jacobian path")
    config = {
        "run_id": run_id, "status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": {key: str(value) if hasattr(value, "drive") else value for key, value in vars(args).items()},
        "input_files": {str(path): sha256(path) for path in (args.rna, args.atac)},
        "script_sha256": sha256(absolute_project_path(__file__)), "adapter_manifest": adapter_manifest,
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "anndata", "torchdiffeq", "torchmetrics")},
        "cuda_runtime": gpu_info, "transfer_verification": transfer_check,
        "training_implementation_sha256": sha256(PROJECT_ROOT / "repo/colab/colab_training.py"),
        "source_Windows_trainer_sha256": "41950d69ad83e4496eff3fa1d3b64d7308807345f10049b963b1f2d183346163",
        "input_assembly": assembly, "n_cells": rna.n_obs, "n_genes": rna.n_vars, "n_TF": atac.n_vars,
        "n_train": len(train), "n_validation": len(valid), "split_seed": 0, "training_seed": args.seed,
        "analytic_velocity": False, "velocity_computation": "Original upstream vmap(jacrev(decoder)) and batched matrix product",
        "analytic_hold_reason": "Representative batch128 gradient equivalence failed; small-batch PASS cannot authorize this optimization",
        "batching": "CUDA; default 64; all cells retained; singleton tail rebalanced without changing loss/solver",
        "model_defaults": {"num_hidden": args.hidden, "zx_dim": args.zx_dim, "zy_dim": args.zy_dim, "k_t": 1000, "k_z0": 1000, "k_velocity": 10000, "k_consistency": 10000, "sigma_x": 0.1, "sigma_y": 0.1, "mu_pz0": 0, "sigma_pz0": 1, "alpha_pt": 2, "beta_pt": 2, "time_variant": False},
        "fit": "Upstream fit, Adam, ReduceLROnPlateau patience5 factor0.5 min_lr1e-6, EarlyStopping patience10 min_delta0",
        "prediction_storage": "Float32 disk-backed posterior summary matrices; upstream chunk accumulation uses float64",
        "time_rank": "All posterior means concatenated, then a single stable ascending ordinal rank / (N-1)",
        "chunking_scope": "Bounded evaluation reuses the same fitted model. Posterior draw order and adaptive ODE integration can depend on chunk partition; no bitwise whole-dataset equivalence claim.",
        "interpretation": "Transductive single-library internal QC; no independent external validation; no biological convergence PASS",
        "prediction_dir": str(prediction_dir), "cache_dir": str(cache_dir), "checkpoint_dir": str(run_dir),
    }
    preserve_resume_config(run_dir, config, args.resume_checkpoint,
                           prior_config=transfer_check["resume_verification"]["prior_config"] if args.resume_checkpoint else None)
    write_json(run_dir / "config.json", config)
    estimate = (2 * sum(path.stat().st_size for path in (args.rna, args.atac))
                + 4 * rna.n_obs * (8 * rna.n_vars + 8 * atac.n_vars + 22 * (args.zx_dim + args.zy_dim)))
    if shutil.disk_usage(PROJECT_ROOT).free < estimate + 2 * 1024**3:
        raise OSError(f"Insufficient D-drive space for projected summaries plus 2 GiB reserve ({estimate} bytes estimated)")

    class BatchSampler:
        def __init__(self, n_cells, shuffle):
            self.n_cells, self.shuffle = n_cells, shuffle
            self.generator = torch.Generator().manual_seed(args.seed)

        def __iter__(self):
            indices = torch.randperm(self.n_cells, generator=self.generator).tolist() if self.shuffle else list(range(self.n_cells))
            yield from batches_without_singleton(indices, args.batch_size)

        def __len__(self):
            return math.ceil(self.n_cells / args.batch_size)

    class FiniteAdam(torch.optim.Adam):
        def step(self, closure=None):
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("Nonfinite gradient before optimizer step")
            result = super().step(closure=closure)
            for group in self.param_groups:
                for parameter in group["params"]:
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError("Nonfinite parameter after optimizer step")
            return result

    train_dataset = Dataset(subset_for_dataset(rna, train), subset_for_dataset(atac, train))
    valid_dataset = Dataset(subset_for_dataset(rna, valid), subset_for_dataset(atac, valid))
    train_loader = DataLoader(train_dataset, batch_sampler=BatchSampler(len(train), True), num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_sampler=BatchSampler(len(valid), False), num_workers=0)
    model = DynaVelo(x_dim=rna.n_vars, y_dim=atac.n_vars, zx_dim=args.zx_dim, zy_dim=args.zy_dim, num_hidden=args.hidden, seed=args.seed, device="cuda:0", mode="train", dataset_name="GSE284047", sample_name=run_id)
    if absolute_project_path(model.train_dir) != run_dir:
        raise ValueError("Model checkpoint directory differs from recorded run directory")
    optimizer = FiniteAdam(model.parameters(), lr=args.learning_rate)
    rows, fit_started = [], time.monotonic()
    original_add = model.logger.add

    def finite_logger_add(row):
        numeric = np.asarray(row, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"Nonfinite epoch loss/metric: {row}")
        original_add(row)
        record = {name: float(value) for name, value in zip(model.logger.names, row)}
        record["Epoch"] = int(record["Epoch"])
        record["learning_rate_before_scheduler"] = float(optimizer.param_groups[0]["lr"])
        record["elapsed_seconds"] = (time.monotonic() - fit_started + getattr(model, "_resume_elapsed_seconds", 0.0))
        rows.append(record)
        pd.DataFrame(rows).to_csv(run_dir / "training_metrics.tsv", sep="\t", index=False)
        print(json.dumps(record, allow_nan=False), flush=True)

    model.logger.add = finite_logger_add
    print("Training uses original upstream Jacobians; analytic_velocity=False; biological QC remains pending", flush=True)
    if args.benchmark_only:
        benchmark(model, train_loader, optimizer, args, run_dir)
        config["status"] = "BENCHMARK_ONLY_NO_RESEARCH_MODEL"
        write_json(run_dir / "config.json", config)
        return
    fit_resumable(model, train_loader, valid_loader, optimizer, args, run_dir, rows, config)
    if not rows:
        raise RuntimeError("No completed finite training epoch")
    checkpoint = absolute_project_path(model.ckpt_path)
    if not checkpoint.is_file():
        raise FileNotFoundError("Upstream fit did not save a best checkpoint")
    model.load(optimizer)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("Reloaded best checkpoint has nonfinite parameters")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    if not all(torch.equal(value.detach().cpu(), checkpoint_state[key]) for key, value in model.state_dict().items()):
        raise RuntimeError("Best checkpoint reload did not reproduce saved model state")
    del checkpoint_state, train_loader, valid_loader, train_dataset, valid_dataset
    losses = np.array([row["loss_test"] for row in rows])
    best_epoch = rows[int(np.flatnonzero(losses == losses.min())[-1])]["Epoch"]
    metrics = {
        "run_id": run_id, "status": "QC_REVIEW_PENDING", "numerical_training_gate": "PASS_FINITE_LOSSES_GRADIENTS_PARAMETERS",
        "biological_convergence": "NOT_ESTABLISHED_REQUIRES_QC_REVIEW", "external_validation": "NOT_PERFORMED", "perturbation": "NOT_PERFORMED",
        "epochs_completed": len(rows), "best_validation_loss_epoch": best_epoch, "best_validation_loss": float(losses.min()),
        "training_seconds": (time.monotonic() - fit_started + getattr(model, "_resume_elapsed_seconds", 0.0)), "best_checkpoint_reloaded": True,
        "best_checkpoint": str(checkpoint), "best_checkpoint_sha256": sha256(checkpoint),
        "analytic_velocity": False, "config_file": str(run_dir / "config.json"),
        "heldout_scope": "Fixed seed0 80/20 cell split in one library, upstream shared preprocessing, no independent donor replication",
    }
    metrics["last_checkpoint"] = str(run_dir / "last_checkpoint.pth")
    write_json(run_dir / "training_gate.json", metrics)
    if args.training_only:
        metrics["status"] = "TRAINING_CHECKPOINT_READY_QC_REVIEW_PENDING"
        metrics["posterior_inference"] = "DEFERRED_BY_TRAINING_ONLY"
        config["status"] = metrics["status"]
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "config.json", config)
        print(json.dumps(metrics, indent=2), flush=True)
        return
    rna_pred, atac_pred, progress = chunked_evaluate(model, Dataset, rna, atac, args, cache_dir, run_dir)
    metrics["RNA_reconstruction"] = reconstruction_metrics(rna.X, rna_pred.layers["x_pred_mean"], train, valid, args.evaluation_chunk_cells)
    metrics["motif_reconstruction"] = reconstruction_metrics(atac.X, atac_pred.layers["y_pred_mean"], train, valid, args.evaluation_chunk_cells)
    metrics["velocity_prior_agreement"] = velocity_metrics(rna.layers["velocity"], rna_pred.layers["vx_pred_mean"], valid, rna.obs_names, args.evaluation_chunk_cells, run_dir / "validation_velocity_cosines.tsv")
    metrics["evaluation_chunks"] = len(progress)
    metrics["posterior_samples_per_cell"] = args.n_samples
    metrics["latent_time_rank_scope"] = "All aligned cells globally; chunk-local ranks discarded"
    metrics["velocity_nan_entries_preserved"] = int(np.isnan(rna_pred.layers["velocity"]).sum())
    config["status"] = "QC_REVIEW_PENDING"
    for obj in (rna_pred, atac_pred):
        obj.uns["dynavelo_training_config_json"] = json.dumps(config, ensure_ascii=False, allow_nan=False)
        obj.uns["dynavelo_internal_qc_json"] = json.dumps(metrics, ensure_ascii=False, allow_nan=False)
    rna_output, atac_output = prediction_dir / "adata_rna_pred.h5ad", prediction_dir / "adata_atac_pred.h5ad"
    write_h5ad_atomic(rna_pred, rna_output)
    write_h5ad_atomic(atac_pred, atac_output)
    metrics["prediction_files"] = {str(path): sha256(path) for path in (rna_output, atac_output)}
    metrics["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "metrics.json", metrics)
    write_json(PROJECT_ROOT / "results/reports" / (run_id + "_internal_qc.json"), metrics)
    write_json(run_dir / "config.json", config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def main():
    configure_cloud_storage()
    args = parse_args()
    required_paths = [args.rna, args.atac, PROJECT_ROOT / "repo/DynaVelo/dynavelo/models.py",
                      PROJECT_ROOT / "repo/DynaVelo/dynavelo/utils.py", PROJECT_ROOT / "repo/colab/colab_runtime.py",
                      PROJECT_ROOT / "repo/colab/dynavelo_colab_adapter.py", PROJECT_ROOT / "repo/colab/colab_training.py", absolute_project_path(__file__)]
    resume_check = None
    if args.resume_checkpoint:
        resume_check = verify_checkpoint_restore_manifest(args.resume_manifest, args.resume_checkpoint)
        required_paths += [args.resume_manifest] + [absolute_project_path(resume_check[role])
                           for role in ("checkpoint", "prior_config", "best_snapshot")]
    transfer_check = verify_transfer_manifest(args.transfer_manifest, required_paths)
    if resume_check is not None:
        transfer_check["resume_verification"] = resume_check
    transfer_check["environment_check"] = verify_frozen_environment()
    run_id = f"8489-MV-1_cuda_seed{args.seed}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.resume_checkpoint:
        if not args.resume_checkpoint.is_file() or args.resume_checkpoint.name != "last_checkpoint.pth":
            raise ValueError("Resume requires a completed-epoch last_checkpoint.pth")
        if args.resume_checkpoint.parent.parent != PROJECT_ROOT / "checkpoints/GSE284047":
            raise ValueError("Resume checkpoint is outside the expected cloud run tree")
        run_id = args.resume_checkpoint.parent.name
    run_dir = absolute_project_path(PROJECT_ROOT / "checkpoints/GSE284047" / run_id)
    prediction_dir = absolute_project_path(args.prediction_root / run_id)
    cache_dir = absolute_project_path(PROJECT_ROOT / "tmp_cache/prediction_arrays" / run_id)
    for directory in (run_dir, prediction_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=bool(args.resume_checkpoint))
    atexit.register(make_download_manifest, run_dir, prediction_dir, run_id)
    with (run_dir / "console.log").open("a" if args.resume_checkpoint else "w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, run_id, run_dir, prediction_dir, cache_dir, transfer_check)
            except Exception as error:
                failure = {"run_id": run_id, "status": "FAILED_ENGINEERING_OR_NUMERICAL_GATE", "error": repr(error), "traceback": traceback.format_exc(), "biological_convergence": "NOT_ESTABLISHED", "external_validation": "NOT_PERFORMED", "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                write_json(run_dir / "failure.json", failure)
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
