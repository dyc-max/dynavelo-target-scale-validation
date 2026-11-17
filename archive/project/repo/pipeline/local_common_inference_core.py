"""Native XPU numerical primitives for common own-best factual reevaluation.

This module has no production execution action. Completed training provenance,
exclusive job ownership and an approved common runtime recipe belong to the
caller. Import/inspect never loads human data, model tensors or a GPU context.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import re
import time
import types
from pathlib import Path

ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
SELF = Path(__file__).resolve()
BASE = "repo/colab/evaluate_own_best_on_factual.py"
BASE_SHA = "d8d040ccc985b112f82b69b04071508f95236473dde3199f47d1cd95eb2c342b"
LOCAL = "repo/pipeline/train_hspc_local_resume.py"
LOCAL_SHA = "e6b810b1da4448135646a90db3d9e4dffeafb5382c43e6648e9e9893a56e71a5"
CPU = "repo/pipeline/dynavelo_cpu_adapter.py"
CPU_SHA = "0f29d3bd4c919d743820c6a022266aba2358a72348aa84385461d57c1aebc05f"
ASSEMBLY = "repo/pipeline/assemble_model_inputs.py"
ASSEMBLY_SHA = "e3d76669b4740e81f7760dae922543f1af67627e648bbac76dceed36e13ee3d3"
REGISTRATION = "results/reports/common_inference_master_seed_registration_20261112.json"
REGISTRATION_SHA = "dd2d7514fbe6180949689a02ce2f026bff5f1915e9800b54c863d33ea0cad815"
MASTER_SEED = 2026111201


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source(path, digest):
    p = ROOT / path
    if sha(p) != digest:
        raise ValueError("Frozen source drift: " + path)
    return p.read_text(encoding="utf-8-sig")


def function(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError("Exactly one original function required: " + name)
    return ast.get_source_segment(text, nodes[0])


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Frozen adaptation anchor differs: " + old[:100])
    return text.replace(old, new, 1)


_original = source(BASE, BASE_SHA)
for _node in ast.parse(_original).body:
    if isinstance(_node, ast.Assign) and len(_node.targets) == 1 and isinstance(_node.targets[0], ast.Name):
        if _node.targets[0].id in ("SOURCES", "INPUT_SHA", "SEED_POLICY", "PAIRS"):
            globals()[_node.targets[0].id] = ast.literal_eval(_node.value)
PINS = {**SOURCES, BASE: BASE_SHA, LOCAL: LOCAL_SHA, CPU: CPU_SHA, ASSEMBLY: ASSEMBLY_SHA}
_unchanged = ("canonical", "digest", "safe", "checked", "json_bytes", "save_new", "file_record", "chunk_seed", "time_description", "axes")
exec(compile("\n\n".join(function(_original, n) for n in _unchanged), "<pinned_common_inference_utilities>", "exec"), globals())


def source_check():
    for rel, expected in PINS.items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Frozen source drift: " + rel)
    if sha(ROOT / REGISTRATION) != REGISTRATION_SHA:
        raise ValueError("Pre-outcome master seed registration changed")
    reg, _ = json_bytes(ROOT / REGISTRATION, REGISTRATION_SHA)
    if (reg.get("inference_master_seed"), reg.get("n_samples"), reg.get("chunk_cells"), reg.get("seed_policy")) != (MASTER_SEED, 8, 64, SEED_POLICY):
        raise ValueError("Master seed registration differs")
    return {**PINS, REGISTRATION: REGISTRATION_SHA}


def runtime_contract():
    """Call only in the exclusive inference worker, never alongside training."""
    source_check()
    import train_hspc_local_resume as local
    versions = local.source_check("xpu")
    if versions["torch"] != "2.10.0+xpu":
        raise RuntimeError("Qualified torch 2.10.0+xpu is required")
    import torch
    if not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable; no automatic backend substitution")
    if torch.get_num_threads() != 4 or torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Frozen four-thread runtime policy differs")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Original full FP32 precision is required")
    return {"device": "xpu", "device_name": torch.xpu.get_device_name(0),
            "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
            "deterministic_algorithms": False, "float32_matmul_precision": "highest",
            "scientific_packages": versions,
            "execution_boundary": "XPU primary; qualified operator CPU fallback; no CUDA bitwise equivalence"}


def validate_device_contract(device):
    keys = {"device", "device_name", "torch_version", "threads", "deterministic_algorithms", "float32_matmul_precision", "scientific_packages", "execution_boundary"}
    packages = {"torch": "2.10.0+xpu", "numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.3", "anndata": "0.10.9", "h5py": "3.12.1", "torchmetrics": "1.6.1", "torchdiffeq": "0.2.5"}
    if (set(device) != keys or device.get("device") != "xpu" or device.get("torch_version") != "2.10.0+xpu"
            or device.get("threads") != 4 or device.get("deterministic_algorithms") is not False
            or device.get("float32_matmul_precision") != "highest" or device.get("scientific_packages") != packages
            or not isinstance(device.get("device_name"), str) or not device["device_name"]
            or device.get("execution_boundary") != "XPU primary; qualified operator CPU fallback; no CUDA bitwise equivalence"):
        raise ValueError("Complete frozen native XPU device/library contract required")


_recipe = function(_original, "validate_recipe")
_recipe = replace_once(_recipe, "'dynavelo_common_inference_recipe_v1'", "'dynavelo_local_common_inference_recipe_v1'")
_device_start = _recipe.index(" d=recipe.get('device_contract',{})")
_device_end = _recipe.index(" return recipe", _device_start)
_recipe = _recipe[:_device_start] + " validate_device_contract(recipe.get('device_contract',{}))\n" + _recipe[_device_end:]
_recipe = replace_once(_recipe, " return recipe", " if recipe.get('inference_master_seed')!=MASTER_SEED or recipe.get('master_seed_registration_sha256')!=REGISTRATION_SHA or recipe.get('local_sources')!=PINS:raise ValueError('Registered native inference identity differs')\n return recipe")
exec(compile(_recipe, "<same_common_recipe_with_native_contract>", "exec"), globals())

_rng_original = function(_original, "evaluate_with_chunk_rng")
if _rng_original.count("torch.cuda.") != 7:
    raise ValueError("Unexpected original RNG interface count")
_rng = _rng_original.replace("torch.cuda.", "torch.xpu.")
_rng = replace_once(_rng, "initial_torch_cuda0_rng_sha256", "initial_torch_xpu0_rng_sha256")
# Preserve the exact original operation order, reseeding and finally restoration.
# Availability is a production caller gate; CPU fixtures emulate only this API.
exec(compile(_rng, "<same_common_chunk_rng_with_xpu_api>", "exec"), globals())

_history = function(_original, "validate_training_metadata")
_history = "def own_history_selection(metrics,history_raw):\n" + _history[_history.index(" rows=list(csv.DictReader"):]
exec(compile(_history, "<unchanged_own_mapped_validation_selection>", "exec"), globals())

_load = function(_original, "load_own_best")
_old_adapter = "adapter['adapter_sha256']!=SOURCES['repo/colab/dynavelo_colab_adapter.py']"
_new_adapter = "adapter.get('local_resume_source_sha256')!=LOCAL_SHA or adapter.get('base_cpu_adapter_sha256')!=CPU_SHA or adapter.get('adapter_sha256')!=LOCAL_SHA or adapter.get('device')!='xpu'"
_load = replace_once(_load, _old_adapter, _new_adapter)
_load = replace_once(_load, "device='cuda:0'", "device='xpu'")
exec(compile(_load, "<same_own_best_loading_with_qualified_local_adapter>", "exec"), globals())


def numerical_helpers():
    """Extract only inference/data/metric helpers, never run/fit/optimizer code."""
    source_check()
    from assemble_model_inputs import write_json
    names = ["batches_without_singleton", "dense32", "subset_for_dataset", "validate_inputs", "reconstruction_metrics", "velocity_metrics", "chunked_evaluate"]
    text = source("repo/colab/train_hspc_colab.py", SOURCES["repo/colab/train_hspc_colab.py"])
    helper = types.ModuleType("_native_common_numerical_helpers")
    helper.__dict__.update(json=json, math=math, time=time, write_json=write_json)
    code = "\n\n".join(function(text, n) for n in names)
    exec(compile(code, "<unchanged_inference_helpers>", "exec"), helper.__dict__)
    helper.extracted_source_sha256 = hashlib.sha256(code.encode()).hexdigest()
    return helper


def training_helpers(historical_cloud_root=False):
    """Verify a completed recovery index with local paths, without rewriting it.

    The caller must separately establish actual quiescent completion/provenance.
    A valid recovery index alone is never accepted as completed training evidence.
    The historical root exception is only for an explicitly attested seed0.
    """
    source_check()
    from assemble_model_inputs import absolute_project_path
    text = source("repo/colab/colab_training.py", SOURCES["repo/colab/colab_training.py"])
    code = function(text, "immutable_config") + "\n\n" + function(text, "verify_checkpoint_restore_manifest")
    if historical_cloud_root:
        code = replace_once(code, 'manifest.get("cloud_root") != str(PROJECT_ROOT)', 'manifest.get("cloud_root") != "/content/DynaVelo_Research"')
    helper = types.ModuleType("_native_completed_checkpoint_helpers")
    helper.__dict__.update(PROJECT_ROOT=ROOT, absolute_project_path=absolute_project_path, sha256=sha, json=json)
    exec(compile(code, "<same_restore_index_local_paths_explicit_historical_root>", "exec"), helper.__dict__)
    return helper


def qualified_loader():
    source_check()
    import train_hspc_local_resume as local
    return local.local_adapter("xpu")


def interface_manifest():
    return {"schema": "dynavelo_native_common_inference_core_v1", "status": "SOURCE_INTERFACE_ONLY_NO_EXECUTION",
            "sources": source_check(), "core_sha256": sha(SELF),
            "extracted_function_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in
                {"original_rng": _rng_original, "native_rng": _rng, "native_recipe": _recipe, "native_own_best_loader": _load, "own_history_selection": _history}.items()},
            "unchanged_functions": list(_unchanged), "master_seed": MASTER_SEED, "n_samples": 8, "chunk_cells": 64,
            "production_entry_implemented": False, "actual_XPU_inference_tested": False,
            "completion_provenance_authority": "Caller must verify real completed source-specific training evidence before loading any tensor",
            "scientific_pass": False, "training_started": False}


if __name__ == "__main__":
    print(json.dumps(interface_manifest(), ensure_ascii=False, allow_nan=False))
