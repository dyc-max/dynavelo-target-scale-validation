"""Nine completed own-best models, one newly registered CPU inference recipe.

The frozen training sources are never edited. Completion provenance is checked
separately from the engineering optimization qualification. No fitting here.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import types

ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
SELF = Path(__file__).resolve()
OPT = ROOT / "repo/posttraining_optimization_20261114"
COHORT = ROOT / "results/reports/posttraining_optimization_20261114/nine_model_cpu_qualification.json"
COHORT_SHA = "1d29f3fd6aded99f1dcfb0b7d49a18ce06631ec1ac433c3793880c72b6a88a10"
REPORT = ROOT / "results/reports/common_cpu_v1_20261114"
RECIPE = REPORT / "recipe.json"
OWNER_SHA = "14f9b61c72ab2cde44a859288904ca34a61d7e6a8114a79177ae20d23e29995b"
PINNED = {
    "repo/pipeline/local_common_inference_core.py": "f106b4dcaac7c3d2efa28e3067741aeb595d45d31d3629a866cf81e997c48493",
    "repo/pipeline/dynavelo_cpu_adapter.py": "0f29d3bd4c919d743820c6a022266aba2358a72348aa84385461d57c1aebc05f",
    "repo/pipeline/supervise_local_resume.py": OWNER_SHA,
    "repo/posttraining_optimization_20261114/exact_inference.py": "dd3bceceb63cbc7279f02730e20508d23ca2deb9922377bf0d84772e3aada18a",
    "repo/posttraining_optimization_20261114/qualified_runtime.py": "9862d7e3dea0866950948b217655003541d6231fae2a594ca7d325e518a85fcf",
}
LABELS = tuple(f"{kind}_seed{seed}" for kind in ("F", "V", "M") for seed in range(3))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(yes, message):
    if not yes:
        raise ValueError(message)


for _rel, _digest in PINNED.items():
    require(sha(ROOT / _rel) == _digest, "Frozen source drift: " + _rel)
import local_common_inference_core as q
sys.path.insert(0, str(OPT))
import qualified_runtime


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record(path):
    return q.file_record(ROOT, Path(path))


def checked(rec):
    return q.checked(ROOT, rec)


def sources():
    q.source_check()
    pins = {**q.PINS, **PINNED,
            "repo/pipeline/common_cpu_v1.py": sha(SELF),
            "repo/pipeline/run_common_cpu_v1.py": sha(SELF.with_name("run_common_cpu_v1.py")),
            "repo/pipeline/supervise_common_cpu_v1.py": sha(SELF.with_name("supervise_common_cpu_v1.py"))}
    for rel, digest in pins.items():
        require(sha(ROOT / rel) == digest, "Runtime source differs")
    return pins


def runtime():
    import torch
    require(str(torch.__version__) == "2.5.1+cpu" and not torch.cuda.is_available(), "CPU2.5.1 only; no backend fallback")
    require(torch.get_num_threads() == 4 and torch.get_float32_matmul_precision() == "highest"
            and not torch.are_deterministic_algorithms_enabled(), "Registered CPU arithmetic policy differs")
    versions = {k: importlib.metadata.version(k) for k in ("torch", "numpy", "scipy", "pandas", "anndata", "h5py", "torchmetrics", "torchdiffeq")}
    require(versions == {"torch": "2.5.1+cpu", "numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.3",
                        "anndata": "0.10.9", "h5py": "3.12.1", "torchmetrics": "1.6.1", "torchdiffeq": "0.2.5"}, "Library versions differ")
    return {"device": "cpu", "threads": 4, "float32_matmul_precision": "highest", "deterministic_algorithms": False,
            "scientific_packages": versions, "implementation": "structured decoder Jacobian; original bmm; exact indexed copies",
            "training_backend_equivalence_claim": False}


# The original CPU-compatible conditional RNG wrapper is used byte for byte.
_rng_module = types.ModuleType("_unchanged_common_cpu_rng")
_rng_module.__dict__.update(q.__dict__)
exec(compile(q.function(q._original, "evaluate_with_chunk_rng"), "<unchanged_original_common_rng>", "exec"), _rng_module.__dict__)
evaluate_with_chunk_rng = _rng_module.evaluate_with_chunk_rng


def factual_evidence(seed, binding):
    if seed == 2:
        import native_seed2_recovery_bridge as bridge
        entry = bridge.adapt("evaluate")
        path = ROOT / "results/reports/local_completion_evidence/seed2_recovery_completion_20261114_v1/evidence.json"
    else:
        import evaluate_own_best_local as entry
        path = ROOT / (entry.SEED0_AUDIT if seed == 0 else
                      "results/reports/local_completion_evidence/" + binding["run_id"] + "_completion_v1/evidence.json")
    evidence, census = entry.completed_evidence({"training_seed": seed, "training_run_id": binding["run_id"], "completion_evidence": record(path)})
    return evidence, census, record(path)


def audit_training(label, qualification, controls):
    import torch
    binding = qualification["binding"]
    binding_path = OPT / (label + "_binding.json")
    require(sha(binding_path) == qualification["binding_sha256"] and read(binding_path) == binding, "Qualification binding differs")
    run, seed, kind = binding["run_id"], int(label[-1]), label[0]
    cp = Path(binding["checkpoint"]["path"])
    rd = Path(binding["config"]["path"]).parent
    for key in ("checkpoint", "config", "rna", "atac"):
        p = Path(binding[key]["path"])
        require(p.is_absolute() and p.is_relative_to(ROOT) and sha(p) == binding[key]["sha256"], "Bound file differs: " + key)
    require({k: binding[k]["sha256"] for k in ("rna", "atac")} == q.INPUT_SHA, "Common factual inputs differ")
    paths = {"config": rd / "config.json", "metrics": rd / "metrics.json", "history": rd / "training_metrics.tsv",
             "restore_manifest": rd / "checkpoint_restore_manifest.json", "last_checkpoint": rd / "last_checkpoint.pth",
             "prior_config": rd / "resume_config.json", "best_checkpoint": cp}
    config, metrics = read(paths["config"]), read(paths["metrics"])
    restore = read(paths["restore_manifest"])
    require(config["run_id"] == metrics["run_id"] == restore["run_id"] == run and config["training_seed"] == seed, "Own run/seed differs")
    if kind == "F":
        evidence, census, provenance = factual_evidence(seed, binding)
        require(config["status"] == metrics["status"] == "QC_REVIEW_PENDING" and "training_control" not in config["input_assembly"], "Factual completion differs")
        require(sorted(config["input_files"].values()) == sorted(q.INPUT_SHA.values()), "Factual training input differs")
    else:
        receipt_path = Path(binding["provenance"]["receipt"])
        require(sha(receipt_path) == binding["provenance"]["receipt_sha256"], "Final receipt changed")
        receipt = read(receipt_path)
        require(receipt["status"] == "PASS_VERIFIED_RETURN_BYTES" and receipt["label"] == "final" and receipt["run_id"] == run, "Real final return required")
        census = {}
        for item in receipt["files"]:
            p = receipt_path.parent / item["relative_path"]
            require(p.resolve().is_relative_to(receipt_path.parent.resolve()) and p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], "Returned final bytes differ")
            census[p.relative_to(ROOT).as_posix()] = record(p)
        terminal = read(rd / "terminal.json")
        require(terminal["run_id"] == run and terminal["exit_code"] == 0 and terminal["stop_reason"] is None
                and terminal["process_reaped"] is True, "Actual clean cloud training terminal required")
        require(config["status"] == metrics["status"] == "TRAINING_CHECKPOINT_READY_QC_REVIEW_PENDING"
                and config["command_arguments"]["training_only"] is True
                and metrics["posterior_inference"] == "DEFERRED_BY_TRAINING_ONLY", "Cloud training-only scope differs")
        control = controls[kind]
        require(config["input_assembly"]["training_control"] == control["lineage"]
                and config["input_files"] == {"/content/DynaVelo_Research/" + v["relative_path"]: v["sha256"] for v in control["outputs"].values()}, "Fixed control inputs/lineage differ")
        require(config["script_sha256"] == q.SOURCES["repo/colab/train_hspc_colab.py"]
                and config["training_implementation_sha256"] == q.SOURCES["repo/colab/colab_training.py"], "Frozen cloud training source differs")
        provenance = record(receipt_path)
    require(config["analytic_velocity"] is False and metrics["analytic_velocity"] is False and metrics["best_checkpoint_reloaded"] is True, "Original training source or own selection differs")
    require((config["n_cells"], config["n_genes"], config["n_TF"], config["n_train"], config["n_validation"], config["split_seed"]) == (6035, 3006, 365, 4828, 1207, 0), "Fixed dimensions differ")
    for key, value in {"seed": seed, "max_epochs": 200, "batch_size": 64, "learning_rate": .001, "hidden": 200, "zx_dim": 50, "zy_dim": 50}.items():
        require(config["command_arguments"][key] == value, "Fit recipe differs: " + key)
    selection = q.own_history_selection(metrics, paths["history"].read_bytes())
    require(all(selection[k] == v for k, v in binding["selection"].items()), "Qualified checkpoint selection differs")
    require(selection["completed_epochs"] - selection["own_best_epoch"] == 10, "Original early stopping sequence differs")
    for item in restore["required_files"]:
        # Preserve original restore indexes; only resolve them inside the actual returned run directory.
        p = rd / Path(item["relative_path"]).name
        require(p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], "Required recovery bytes differ")
        if item["role"] == "best_snapshot":
            paths["best_snapshot"] = p
    require(sha(cp) == metrics["best_checkpoint_sha256"] == sha(paths["best_snapshot"]), "Own best weights differ")
    for p in paths.values():
        r = record(p)
        require(census.get(r["relative_path"]) == r, "Training artifact absent from verified completion census")
    last = torch.load(paths["last_checkpoint"], map_location="cpu", weights_only=True)
    helper = q.training_helpers(historical_cloud_root=kind != "F" or seed == 0)
    require(last["schema"] == "dynavelo_epoch_resume_v1" and last["completed_epoch"] == metrics["epochs_completed"] == restore["completed_epoch"]
            and last["best_loss"] == metrics["best_validation_loss"] and last["best_checkpoint_sha256"] == sha(cp)
            and last["best_snapshot_filename"] == paths["best_snapshot"].name
            and last["immutable_config"] == helper.immutable_config(config) == helper.immutable_config(read(paths["prior_config"])), "Completed checkpoint state/metadata differs")
    with paths["history"].open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    require([(int(float(r["Epoch"])), float(r["loss_test"])) for r in rows]
            == [(int(r["Epoch"]), float(r["loss_test"])) for r in last["history"]], "Checkpoint history differs")
    best = torch.load(cp, map_location="cpu", weights_only=True)["model_state_dict"]
    require(len(best) == 32 and all(torch.isfinite(v).all().item() for v in best.values()), "Nonfinite own best")
    migration = None
    if label == "M_seed2":
        migration = read(rd / "native_migration.json")
        require(migration["source_completed_epoch"] == 9 and migration["cloud_segment_run_id"] == run
                and migration["cross_device_bitwise_continuation_claim"] is False and migration["cuda_stream_seed"] == 2,
                "M2 mixed-segment provenance absent")
        native = ROOT / "checkpoints/GSE284047" / migration["source_run_id"]
        require(sha(native / "last_checkpoint.pth") == migration["source_checkpoint_sha256"], "Native M2 parent differs")
        extra = [rd / "native_migration.json", rd / "native_parent_config.raw.json", rd / "native_initialization.raw.json"]
    else:
        extra = []
    return {"label": label, "run_id": run, "training_seed": seed, "condition": "factual" if kind == "F" else kind,
            "binding": record(binding_path), "completion_evidence": provenance, "selection": selection,
            "training_artifacts": {k: record(p) for k, p in paths.items()}, "additional_provenance": [record(p) for p in extra],
            "architecture": config["model_defaults"], "training_versions": config["package_versions"],
            "migration": migration, "scientific_pass": False, "posterior_inference_needed": True}


def freeze():
    import torch
    import numpy as np
    import anndata as ad
    import train_hspc_local_control as control
    require(not REPORT.exists(), "Common CPU freeze already attempted; no overwrite")
    observed_runtime = runtime()
    _, reports = qualified_runtime.read_cohort(COHORT, COHORT_SHA)
    controls = {k: control.verified_control_input(k) for k in ("V", "M")}
    models = {}
    for label in LABELS:
        models[label] = audit_training(label, reports[label], controls)
        print(json.dumps({"completed_training_audited": label, "selection": models[label]["selection"]}), flush=True)
    inputs = {k: ROOT / "data/processed/colab_minimal/HSPC" / ("adata_" + k + ".h5ad") for k in q.INPUT_SHA}
    require(all(sha(p) == q.INPUT_SHA[k] for k, p in inputs.items()), "Original factual input changed")
    rna, atac = [ad.read_h5ad(inputs[k]) for k in ("rna", "atac")]
    train, valid, assembly = q.numerical_helpers().validate_inputs(rna, atac)
    require("training_control" not in assembly and (rna.n_obs, rna.n_vars, atac.n_vars, len(train), len(valid)) == (6035, 3006, 365, 4828, 1207), "Actual common population differs")
    baselines = {}
    for name, obj in (("rna", rna), ("atac", atac)):
        mean = np.zeros(obj.n_vars, dtype=np.float64)
        for start in range(0, len(train), 64):
            mean += np.sum(obj.X[train[start:start + 64]], axis=0, dtype=np.float64)
        baselines[name] = mean / len(train)
    REPORT.mkdir(parents=True, exist_ok=False)
    baseline_path = REPORT / "fresh_common_training_means.npz"
    np.savez(baseline_path, **baselines)
    recipe = {"schema": "dynavelo_common_cpu_recipe_v1", "implementation_version": "common_cpu_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "/root", "execution_authorized": True,
              "training_authorized": False, "scientific_pass": False, "sources": sources(), "runtime": observed_runtime,
              "cohort": record(COHORT), "master_seed_registration": record(ROOT / q.REGISTRATION),
              "inference_master_seed": q.MASTER_SEED, "seed_policy": q.SEED_POLICY, "chunk_cells": 64, "n_samples": 8,
              "axis_sha256": q.axes(rna, atac), "inputs": {k: record(p) for k, p in inputs.items()},
              "train_cells_sha256": q.digest(list(rna.obs_names[train].astype(str))),
              "valid_cells_sha256": q.digest(list(rna.obs_names[valid].astype(str))), "training_mean_baseline": record(baseline_path),
              "models": models, "original_forward_audit_chunks": [0, 47, 94], "audit_draws_per_chunk": 8,
              "comparison_scope": "New common CPU outputs only. Seeds/cells are not donor replicates. Mixed training backend provenance retained.",
              "old_GPU_prediction_or_graph_reuse_authorized": False, "external_results_opened": False}
    q.save_new(RECIPE, recipe)
    return {"recipe": record(RECIPE), "models": len(models), "new_baseline": record(baseline_path), "scientific_pass": False}


def verify_recipe(digest):
    require(sha(RECIPE) == digest, "Exact common recipe required")
    value = read(RECIPE)
    require(value["schema"] == "dynavelo_common_cpu_recipe_v1" and value["sources"] == sources()
            and value["runtime"] == runtime() and value["inference_master_seed"] == 2026111201
            and value["chunk_cells"] == 64 and value["n_samples"] == 8
            and set(value["models"]) == set(LABELS) and value["scientific_pass"] is False,
            "Registered common CPU contract differs")
    checked(value["cohort"]);checked(value["master_seed_registration"]);checked(value["training_mean_baseline"])
    for r in value["inputs"].values():
        checked(r)
    return value


def verify_model(record_value):
    checked(record_value["binding"]);checked(record_value["completion_evidence"])
    for r in [*record_value["training_artifacts"].values(), *record_value["additional_provenance"]]:
        checked(r)
    return {k: checked(r) for k, r in record_value["training_artifacts"].items()}
