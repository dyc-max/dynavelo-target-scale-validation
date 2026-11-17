"""One manifest-bound fresh local XPU seed2, using the pinned original fit loop.

Default inspect does not construct a model. Short-test updates are discarded.
This module makes no cloud calls and never edits an earlier trained run.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
SELF = Path(__file__).resolve()
LOCAL_SHA = "e6b810b1da4448135646a90db3d9e4dffeafb5382c43e6648e9e9893a56e71a5"
REVIEW_SHA = "4ff1f830c89aad5b0088097887e35b0ddeecd7947700076bca2ac4cca41f084f"

def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

if sha(ROOT / "repo/pipeline/train_hspc_local_resume.py") != LOCAL_SHA:
    raise RuntimeError("Pinned local device adapter changed")
import train_hspc_local_resume as local

FIT = {**local.FIT, "seed": 2}
RECIPE = {**FIT, "max_epochs": 200, "n_samples": 8, "evaluation_chunk_cells": 64, "threads": 4}
RUN_RE = re.compile(r"8489-MV-1_xpu_seed2_\d{8}T\d{12}Z")

class ProbeComplete(Exception):
    pass

def inspect_manifest(path, digest):
    raw, plan = local.pinned_json(path, digest)
    if plan.get("schema") != "dynavelo_local_fresh_seed2_recipe_v1" or plan.get("recipe") != RECIPE:
        raise ValueError("Wrong fresh seed2 recipe")
    if plan.get("entry_sha256") != sha(SELF) or plan.get("local_adapter_source_sha256") != LOCAL_SHA:
        raise ValueError("Source recipe differs")
    review = ROOT / "results/reports/local_seed1_completed_stage_review_v1.json"
    if sha(review) != REVIEW_SHA or plan.get("seed1_review_sha256") != REVIEW_SHA:
        raise ValueError("Seed1 stage review differs")
    if json.loads(review.read_text(encoding="utf-8"))["decision"] != "ADVANCE_ADAPTIVE_DC_SCOPE_TO_FRESH_SEED2_AND_SEPARATELY_PLANNED_CONTROLS":
        raise ValueError("Seed1 does not support this stage")
    versions = local.source_check("xpu")
    if versions["torch"] != "2.10.0+xpu":
        raise ValueError("Requires the qualified local XPU environment")
    if plan.get("frozen_sources") != local.FROZEN:
        raise ValueError("Frozen original sources differ")
    inputs = plan.get("input_files", {})
    if set(inputs) != set(local.INPUTS):
        raise ValueError("Exact original paired input files required")
    for relative, rec in inputs.items():
        p = local.absolute_project_path(ROOT / relative)
        if p.stat().st_size != rec.get("bytes") or sha(p) != rec.get("sha256"):
            raise ValueError("Actual input bytes differ: " + relative)
    helper, runner, compiled = build_modules()
    return {"status": "INSPECTED_NO_MODEL_CONSTRUCTED", "manifest_path": str(path), "manifest_sha256": digest,
            "recipe": RECIPE, "input_files": inputs, "source_sha256": sha(SELF), "versions": versions,
            "compiled_sources": compiled, "model_constructed": False, "seed1_review_sha256": REVIEW_SHA}

def initial_state(model, config, directory):
    import torch
    if tuple(config[k] for k in ("n_cells", "n_genes", "n_TF", "n_train", "n_validation", "split_seed")) != (6035, 3006, 365, 4828, 1207, 0):
        raise ValueError("Actual population/split differs")
    if config["model_defaults"] != local.ARCH or config["training_seed"] != 2:
        raise ValueError("Actual fresh model recipe differs")
    if any(p.device.type != "xpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Model must be on XPU in original float32")
    # This reads state only: no forward, RNG draws or optimizer updates.
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    h = hashlib.sha256()
    descriptions = []
    for name in sorted(state):
        value = state[name].contiguous()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite initial model state")
        meta = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
        h.update(value.numpy().tobytes(order="C"))
        descriptions.append(meta)
    checkpoint = directory / "initial_model_state.pth"
    with checkpoint.open("xb") as f:
        torch.save({"model_state_dict": state}, f)
    local.write_json(directory / "initialization.json", {"training_seed": 2, "logical_state_sha256": h.hexdigest(),
        "archive_sha256": sha(checkpoint), "tensors": descriptions, "forward_or_updates_before_capture": False,
        "semantics": "Within-library initialization evidence; not donor replication or CPU/CUDA numerical equivalence."})

def probe_fit(model, train_loader, valid_loader, optimizer, args, directory, rows, config):
    import numpy as np
    import torch
    class Limited:
        def __init__(self, loader, count):
            self.batches = list(itertools.islice(loader, count))
            self.dataset = range(sum(len(b[0]) for b in self.batches))
        def __iter__(self):
            return iter(self.batches)
    train, valid = Limited(train_loader, 2), Limited(valid_loader, 1)
    model.mode = "train"
    started = time.monotonic()
    loss = float(model.train_step(train, optimizer))
    values = list(map(float, model.test_step(valid)))
    torch.xpu.synchronize()
    if not np.isfinite([loss, *values]).all():
        raise FloatingPointError("Nonfinite fresh short test")
    local.write_json(directory / "short_test.json", {"status": "SHORT_TEST_ONLY_NOT_RESUMABLE", "device": "xpu",
        "source_sha256": sha(SELF), "recipe_manifest_sha256": args.recipe_manifest_sha256,
        "training_seed": 2, "train_batches": 2, "validation_batches": 1, "batch_size": 64,
        "seconds": time.monotonic() - started, "training_loss": loss, "validation_components": values,
        "original_jacobian_forward_backward_adam": True, "research_checkpoint_saved": False,
        "model_disposition": "Discard updates; actual fresh run starts seed2 initialization again."})
    raise ProbeComplete()

def build_modules(probe=False):
    common = dict(PROJECT_ROOT=ROOT, absolute_project_path=local.absolute_project_path, sha256=sha,
        write_json=local.write_json, write_h5ad_atomic=local.write_h5ad_atomic, datetime=datetime, timezone=timezone,
        json=json, random=random, shutil=shutil, time=time, math=math, sys=sys, importlib=importlib,
        __file__=str(SELF), _LOCAL_SOURCE=SELF, _LOCAL_DEVICE="xpu", configure_output_paths=local.configure_output_paths)
    names = ["immutable_config", "preserve_resume_config", "_save_atomic", "_publish_best", "_manifest_record",
             "write_checkpoint_restore_manifest", "verify_checkpoint_restore_manifest", "fit_resumable"]
    hs = local.original_functions("repo/colab/colab_training.py", names)
    helper = types.ModuleType("_dynavelo_local_fresh_helper")
    helper.__dict__.update(common)
    capture, restore = local.rng_functions("xpu")
    helper.__dict__.update(_rng_state=capture, _restore_rng=restore)
    exec(compile(hs, "<unchanged_original_training_helpers>", "exec"), helper.__dict__)
    functions = ["batches_without_singleton", "dense32", "subset_for_dataset", "validate_inputs", "reconstruction_metrics",
                 "velocity_metrics", "chunked_evaluate", "run"]
    ts = local.original_functions("repo/colab/train_hspc_colab.py", functions)
    start = ts.index("    if not torch.cuda.is_available():")
    end = ts.index("    rna, atac =", start)
    ts = ts[:start] + '''    if not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable; no silent CPU fallback")
    torch.xpu.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    gpu_info = {"device": "xpu", "torch_version": torch.__version__, "threads": args.threads,
                "precision": "float32; original Jacobians; no AMP", "bitwise_CUDA_equivalence_claim": False,
                "execution": "XPU-primary; torchdiffeq adjoint dot may fall back to CPU"}
''' + ts[end:]
    ts = local.once(ts, '"cuda_runtime": gpu_info', '"local_runtime": gpu_info')
    ts = local.once(ts, 'sha256(PROJECT_ROOT / "repo/colab/colab_training.py")', 'sha256(_LOCAL_SOURCE)')
    ts = local.once(ts, '"batching": "CUDA; default 64;', '"batching": "Explicit local XPU; default 64;')
    ts = local.once(ts, 'device="cuda:0", mode="train"', 'device=_LOCAL_DEVICE, mode="train"')
    ts = local.once(ts, '    optimizer = FiniteAdam(model.parameters(), lr=args.learning_rate)',
        '    configure_output_paths(model, run_dir, config)\n    optimizer = FiniteAdam(model.parameters(), lr=args.learning_rate)')
    runner = types.ModuleType("_dynavelo_local_fresh_runner")
    runner.__dict__.update(common)
    load = local.local_adapter("xpu")
    def load_fresh(*args, **kwargs):
        Dataset, Model, meta = load(*args, **kwargs)
        return Dataset, Model, {**meta, "local_fresh_source_sha256": sha(SELF), "training_seed": 2}
    def fit(model, train_loader, valid_loader, optimizer, args, directory, rows, config):
        initial_state(model, config, directory)
        fn = probe_fit if probe else helper.fit_resumable
        return fn(model, train_loader, valid_loader, optimizer, args, directory, rows, config)
    runner.__dict__.update(load_dynavelo=load_fresh, preserve_resume_config=helper.preserve_resume_config,
        fit_resumable=fit, benchmark=lambda *a: (_ for _ in ()).throw(ValueError("Use the explicit bounded short-test")))
    exec(compile(ts, "<local_fresh_runner_from_pinned_original>", "exec"), runner.__dict__)
    return helper, runner, {"helper_sha256": hashlib.sha256(hs.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(ts.encode()).hexdigest(), "helper_source": hs, "runner_source": ts}

def execute(a, binding):
    if not a.execute or not RUN_RE.fullmatch(a.run_id or ""):
        raise ValueError("Explicit --execute and unique seed2 run ID required")
    if a.action == "train":
        _, e = local.pinned_json(a.device_evidence, a.device_evidence_sha256)
        if (e.get("status"), e.get("source_sha256"), e.get("recipe_manifest_sha256"), e.get("train_batches"), e.get("validation_batches")) != (
                "SHORT_TEST_ONLY_NOT_RESUMABLE", sha(SELF), a.manifest_sha256, 2, 1):
            raise ValueError("Same-entry/same-recipe short test required")
    cache_root = local.setup_storage()
    run_dir = ROOT / "checkpoints/GSE284047" / a.run_id
    prediction = ROOT / "data/processed/GSE284047/8489-MV-1/model_runs" / a.run_id
    cache = ROOT / "tmp_cache/prediction_arrays" / a.run_id
    if any(p.exists() for p in (run_dir, prediction, cache)):
        raise FileExistsError("Fresh run ID already used; no silent overwrite/resume")
    for p in (run_dir, prediction, cache):
        local.absolute_project_path(p).mkdir(parents=True, exist_ok=False)
    helper, runner, compiled = build_modules(probe=a.action == "short-test")
    for kind in ("helper", "runner"):
        (run_dir / (kind + "_compiled.py.txt")).write_text(compiled.pop(kind + "_source"), encoding="utf-8")
    local.write_json(run_dir / "source_adaptation.json", {**compiled, "entry_sha256": sha(SELF),
        "unchanged_fit_loop": True, "local_adapter_source_sha256": LOCAL_SHA,
        "changes": ["Explicit XPU seeding/device placement and local runtime metadata", "Actual local entry identity",
                    "Short Windows filenames via qualified existing adapter", "Read-only initial-state capture before original fit",
                    "XPU RNG state capture in original completed-epoch checkpoints"],
        "scientific_recipe_changed": False})
    local.write_json(run_dir / "fresh_input_binding.json", {k: v for k, v in binding.items() if k != "compiled_sources"})
    args = argparse.Namespace(**RECIPE, rna=ROOT / local.INPUTS[0], atac=ROOT / local.INPUTS[1],
        prediction_root=prediction.parent, resume_checkpoint=None, resume_manifest=None,
        training_only=False, benchmark_only=False, benchmark_batches=2,
        recipe_manifest_sha256=a.manifest_sha256, local_action=a.action)
    source = (ROOT / "repo/colab/train_hspc_colab.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Tee")
    ns = {}
    exec(compile(ast.get_source_segment(source, node), "<pinned_Tee>", "exec"), ns)
    try:
        with (run_dir / "console.log").open("x", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(ns["Tee"](sys.stdout, log)), redirect_stderr(ns["Tee"](sys.stderr, log)):
                runner.run(args, a.run_id, run_dir, prediction, cache,
                    {"status": "LOCAL_INPUT_SHA_VERIFIED_NO_CLOUD", "recipe_manifest_sha256": a.manifest_sha256})
    except ProbeComplete:
        local.write_json(run_dir / "short_test_completion.json", {"status": "SHORT_TEST_ONLY_NOT_RESUMABLE", "run_id": a.run_id})
    except BaseException as error:
        local.write_json(run_dir / "failure.json", {"status": "FAILED_LOCAL_EXECUTION", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        for rel, rec in binding["input_files"].items():
            if sha(ROOT / rel) != rec["sha256"]:
                raise RuntimeError("Protected input bytes changed")
    return {"run_id": a.run_id, "action": a.action, "run_dir": str(run_dir), "scientific_pass": False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=local.absolute_project_path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--action", choices=("inspect", "short-test", "train"), default="inspect")
    p.add_argument("--run-id")
    p.add_argument("--device-evidence", type=local.absolute_project_path)
    p.add_argument("--device-evidence-sha256")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--report", type=local.absolute_project_path)
    a = p.parse_args()
    binding = inspect_manifest(a.manifest, a.manifest_sha256)
    output = execute(a, binding) if a.action != "inspect" else {
        **binding, "compiled_sources": {k: v for k, v in binding["compiled_sources"].items() if k.endswith("sha256")}}
    if a.report:
        if a.report.exists():
            raise FileExistsError("Report already exists")
        local.write_json(a.report, output)
    print(json.dumps(output, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
