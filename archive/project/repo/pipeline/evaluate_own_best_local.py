"""Owned, completed-factual-model XPU reevaluation; default source inspection.

Only attested seed0/1/2 factual runs are currently supported. Control runs need
their own completed-provenance adapter; they are explicitly rejected here.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

CORE = Path(__file__).resolve().with_name("local_common_inference_core.py")
CORE_SHA = "f106b4dcaac7c3d2efa28e3067741aeb595d45d31d3629a866cf81e997c48493"
import hashlib
if hashlib.sha256(CORE.read_bytes()).hexdigest() != CORE_SHA:
    raise RuntimeError("Native inference core changed")
import local_common_inference_core as q

ROOT = q.ROOT
SELF = Path(__file__).resolve()
SEED0_RUN = "8489-MV-1_cuda_seed0_20261112T131938278584Z"
SEED0_AUDIT = "results/reports/epoch90_terminal_posterior_audit_20261112.json"
SEED0_SHA = "9cc265a708935ed7e7c5381b143d376ea797990e28645f1e7cecf050fda15cea"
ADAPTERS = {1: ("repo/pipeline/audit_local_factual_readout.py", "523a9f5e4b2fb6b6dc85c7f0fa83fb716f2071b92562ae2779ba68a4263f7f23"),
            2: ("repo/pipeline/audit_local_fresh_readout.py", "047bc3646e619f4336bccf0b1ab55351ab12f9c5fbb5d810b75afb927c6e1567")}
OWNER_CORE_SHA = "14f9b61c72ab2cde44a859288904ca34a61d7e6a8114a79177ae20d23e29995b"
PINS = {"repo/pipeline/local_common_inference_core.py": CORE_SHA, **q.PINS,
        **{rel: value for rel, value in ADAPTERS.values()}, "repo/pipeline/supervise_local_resume.py": OWNER_CORE_SHA}
ROLES = {"config", "metrics", "history", "restore_manifest", "last_checkpoint", "prior_config", "best_snapshot", "best_checkpoint"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(value):
    p = Path(value)
    require(p.is_absolute() and p.resolve() == p and p.is_relative_to(ROOT), "Exact absolute D-project path required")
    return p


def plan_shape(plan):
    require(plan.get("schema") == "dynavelo_native_own_best_factual_evaluation_v1"
            and plan.get("reviewer") == "/root" and plan.get("execution_authorized") is True
            and plan.get("training_authorized") is False and plan.get("scientific_pass") is False,
            "Explicit reviewed inference-only plan required")
    require(plan.get("condition") == "factual", "Native control completion adapter is not implemented; no fallback")
    seed, run = plan.get("training_seed"), plan.get("training_run_id")
    require(type(seed) is int and seed in (0, 1, 2), "Explicit factual seed required")
    require((seed == 0 and run == SEED0_RUN) or (seed in (1, 2) and isinstance(run, str)
            and re.fullmatch(r"8489-MV-1_xpu_seed" + str(seed) + r"_\d{8}T\d{12}Z", run)), "Actual native or attested historical run required")
    require(re.fullmatch(r"[A-Za-z0-9_-]{8,80}", plan.get("evaluation_id", "")) is not None, "Unique evaluation namespace required")
    require(type(plan.get("wall_seconds")) is int and 1 <= plan["wall_seconds"] <= 7200, "One evaluation <=2h explicit wall required")
    require(plan.get("evaluator_sha256") == q.sha(SELF) and plan.get("sources") == PINS, "Evaluation source identity differs")
    require(set(plan.get("training_artifacts", {})) == ROLES, "Exact own-training file set required")


def completed_evidence(plan):
    seed, run = plan["training_seed"], plan["training_run_id"]
    rec = plan["completion_evidence"]
    path = q.checked(ROOT, rec)
    if seed == 0:
        require(rec["relative_path"] == SEED0_AUDIT and rec["sha256"] == SEED0_SHA, "Exact prior audited seed0 completion required")
        evidence, _ = q.json_bytes(path, SEED0_SHA)
        require(evidence.get("schema") == "dynavelo_epoch90_terminal_posterior_audit_v1"
                and evidence.get("status") == "PASS_TERMINAL_ARTIFACTS_AXES_FINITE_AND_SEMANTICS_ONLY"
                and evidence.get("terminal_status") == "PROCESS_EXITED_0_SCIENTIFIC_QC_PENDING"
                and evidence.get("run_id") == run and evidence.get("scientific_pass") is False,
                "Historical terminal is not verified completion")
        records = [{"relative_path": x["source_relative_path"], "bytes": x["bytes"], "sha256": x["sha256"]}
                   for x in evidence["source_manifest_records"]]
    else:
        rel, expected = ADAPTERS[seed]
        require(q.sha(ROOT / rel) == expected, "Completed-run adapter source changed")
        if seed == 1:
            import audit_local_factual_readout as adapter
        else:
            import audit_local_fresh_readout as adapter
        evidence = adapter.validate_capture(path, rec["sha256"], run)
        records = evidence["source_records"]
    by_path = {}
    for row in records:
        key = row["relative_path"]
        normalized = {k: row[k] for k in ("relative_path", "bytes", "sha256")}
        require(key not in by_path or by_path[key] == normalized, "Ambiguous completed artifact census")
        by_path[key] = normalized
    return evidence, by_path


def check_metadata(plan, config, metrics, paths, evidence):
    seed, run = plan["training_seed"], plan["training_run_id"]
    require(config.get("run_id") == metrics.get("run_id") == run and config.get("training_seed") == seed, "Own run/seed differs")
    fixed = {"n_cells": 6035, "n_genes": 3006, "n_TF": 365, "n_train": 4828, "n_validation": 1207, "split_seed": 0, "analytic_velocity": False}
    require(all(config.get(k) == v for k, v in fixed.items()), "Fixed factual population/model differs")
    require(config.get("status") == metrics.get("status") == "QC_REVIEW_PENDING" and metrics.get("best_checkpoint_reloaded") is True
            and metrics.get("analytic_velocity") is False, "Completed own-best original-Jacobian model required")
    require("training_control" not in config.get("input_assembly", {}), "Control model cannot use factual completion route")
    prefix = "/content/DynaVelo_Research" if seed == 0 else str(ROOT)
    expected_inputs = {str(Path(prefix) / "data/processed/colab_minimal/HSPC" / ("adata_" + k + ".h5ad")): v for k, v in q.INPUT_SHA.items()}
    actual_inputs = {str(Path(k)): v for k, v in config.get("input_files", {}).items()}
    require(actual_inputs == expected_inputs, "Own training input identity differs")
    args = config["command_arguments"]
    fixed_args = {"seed": seed, "batch_size": 64, "hidden": 200, "zx_dim": 50, "zy_dim": 50, "learning_rate": .001, "max_epochs": 200}
    require(all(args.get(k) == v for k, v in fixed_args.items()) and not args.get("benchmark_only"), "Original fit recipe differs")
    selection = q.own_history_selection(metrics, paths["history"].read_bytes())
    require(selection["own_best_epoch"] == evidence["best_epoch"], "Selection differs from actual completed audit")
    return selection


def preflight(plan_path, plan_sha):
    require(os.name == "nt" and ROOT.drive.upper() == "D:", "Native Windows D-project only")
    plan_path = local_path(plan_path)
    plan, raw = q.json_bytes(plan_path, plan_sha)
    plan_shape(plan)
    q.source_check()
    for rel, expected in PINS.items():
        require(q.sha(ROOT / rel) == expected, "Frozen evaluation dependency drift: " + rel)
    recipe_path = q.checked(ROOT, plan["recipe"])
    recipe = q.validate_recipe(q.json_bytes(recipe_path, plan["recipe"]["sha256"])[0])
    evidence, census = completed_evidence(plan)  # Quiescent completion first, before any checkpoint content.
    for rec in plan["training_artifacts"].values():
        require(census.get(rec["relative_path"]) == rec, "Planned weights/metadata absent from verified completed source census")
    paths = {name: q.checked(ROOT, rec) for name, rec in plan["training_artifacts"].items()}
    run_dir = ROOT / "checkpoints/GSE284047" / plan["training_run_id"]
    names = {"config": "config.json", "metrics": "metrics.json", "history": "training_metrics.tsv", "restore_manifest": "checkpoint_restore_manifest.json",
             "last_checkpoint": "last_checkpoint.pth", "prior_config": "resume_config.json"}
    require(all(paths[role] == run_dir / name for role, name in names.items()), "Training files outside own canonical run")
    require(paths["best_snapshot"].parent == run_dir and paths["best_snapshot"].name in ("best_snapshot_0.pth", "best_snapshot_1.pth")
            and paths["best_checkpoint"].parent == run_dir, "Own snapshot/canonical path differs")
    config, _ = q.json_bytes(paths["config"])
    metrics, _ = q.json_bytes(paths["metrics"])
    historical_best = metrics["best_checkpoint"]
    expected_best = ROOT / historical_best[len("/content/DynaVelo_Research/"):] if plan["training_seed"] == 0 and historical_best.startswith("/content/DynaVelo_Research/") else Path(historical_best)
    require(paths["best_checkpoint"] == expected_best and q.sha(paths["best_checkpoint"]) == metrics["best_checkpoint_sha256"] == q.sha(paths["best_snapshot"]), "Actual own canonical best identity differs")
    selection = check_metadata(plan, config, metrics, paths, evidence)
    return plan, raw, recipe, paths, config, metrics, selection


def owned_worker(job_id, plan_path, plan_sha, plan):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id or "") is not None, "Owned job ID required")
    directory = ROOT / "tmp_cache/local_owned_jobs" / job_id
    job, _ = q.json_bytes(directory / "job.json")
    expected_command = [str(ROOT / "repo/.venv-xpu/Scripts/python.exe"), "-B", str(SELF), "--plan", str(plan_path), "--plan-sha256", plan_sha,
                        "--job-id", job_id, "--execute"]
    require(job.get("schema") == "dynavelo_native_factual_evaluation_job_v1" and job.get("command") == expected_command
            and job.get("job_id") == job_id and job.get("run_id") == plan["evaluation_id"]
            and job.get("manifest_sha256") == plan_sha and job.get("entry_sha256") == q.sha(SELF)
            and job.get("ownership_core_sha256") == OWNER_CORE_SHA and job.get("wall_seconds") == plan["wall_seconds"]
            and job.get("max_rss_bytes") == 13 * 1024**3 and job.get("min_free_bytes") == 3 * 1024**3
            and job.get("supervisor_sha256") == plan.get("supervisor_sha256")
            and q.sha(ROOT / "repo/pipeline/supervise_local_evaluation.py") == plan.get("supervisor_sha256"), "Actual owned job differs from reviewed inference plan")
    require(Path(sys.executable) == Path(expected_command[0]) and q.sha(sys.executable) == job["python_sha256"], "Actual worker interpreter differs")
    in_job = ctypes.c_int()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    require(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)) and in_job.value, "Actual worker must be inside a kernel job")
    require(not (directory / "STOP").exists() and not (directory / "terminal.json").exists(), "Owned job stopped or already terminal")
    return q.file_record(ROOT, directory / "job.json")


def execute(plan_path, plan_sha, job_id):
    started = time.monotonic()
    # Check the owned job before expensive evidence validation or tensor loading.
    plan, _ = q.json_bytes(local_path(plan_path), plan_sha)
    plan_shape(plan)
    job_record = owned_worker(job_id, local_path(plan_path), plan_sha, plan)
    plan, raw, recipe, paths, config, metrics, selection = preflight(plan_path, plan_sha)
    import train_hspc_local_resume as local
    local.setup_storage()
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = q.runtime_contract()
    require(device == recipe["device_contract"], "Actual shared device/library contract differs")
    import anndata as ad
    helpers = q.numerical_helpers()
    helpers.load_dynavelo = q.qualified_loader()
    inputs = {k: ROOT / "data/processed/colab_minimal/HSPC" / ("adata_" + k + ".h5ad") for k in q.INPUT_SHA}
    for k, path in inputs.items():
        require(q.sha(path) == q.INPUT_SHA[k], "Original factual input changed")
    rna, atac = [ad.read_h5ad(inputs[k]) for k in ("rna", "atac")]
    train, valid, assembly = helpers.validate_inputs(rna, atac)
    require("training_control" not in assembly and (rna.n_obs, rna.n_vars, atac.n_vars, len(train), len(valid)) == (6035, 3006, 365, 4828, 1207)
            and q.axes(rna, atac) == recipe["axis_sha256"], "Exact common factual population/axes differs")
    run_dir = ROOT / "checkpoints/factual_evaluations" / plan["evaluation_id"]
    prediction = ROOT / "data/processed/factual_evaluations" / plan["evaluation_id"]
    cache = ROOT / "tmp_cache/factual_evaluations" / plan["evaluation_id"]
    require(not any(p.exists() for p in (run_dir, prediction, cache)), "Evaluation ID already used; explicit new attempt required")
    estimate = 2 * sum(p.stat().st_size for p in inputs.values()) + 4 * rna.n_obs * (8 * rna.n_vars + 8 * atac.n_vars + 22 * 100)
    require(shutil.disk_usage(ROOT).free >= estimate + 2 * 1024**3, "Insufficient D space plus2GiB reserve")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        prediction.mkdir(parents=True, exist_ok=False)
        cache.mkdir(parents=True, exist_ok=False)
        (run_dir / "approved_plan.json").write_bytes(raw)
        (run_dir / "approved_recipe.json").write_bytes(q.checked(ROOT, plan["recipe"]).read_bytes())
        q.save_new(run_dir / "claim.json", {"evaluation_id": plan["evaluation_id"], "pid": os.getpid(), "job": job_record, "plan_sha256": plan_sha,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "training_authorized": False, "scientific_pass": False})
        training = q.training_helpers(historical_cloud_root=plan["training_seed"] == 0)
        model, Dataset, adapter = q.load_own_best(ROOT, plan, paths, config, metrics, helpers, training)
        remaining = plan["wall_seconds"] - math.ceil(time.monotonic() - started)
        require(remaining > 0, "Evaluation budget expired during preparation")
        (rna, atac, progress), events = q.evaluate_with_chunk_rng(model, Dataset, rna, atac,
             types.SimpleNamespace(n_samples=8, evaluation_chunk_cells=64), cache, run_dir, helpers.chunked_evaluate,
             master_seed=recipe["inference_master_seed"], wall_seconds=remaining)
        require(len(progress) == len(events) == 95 and q.axes(rna, atac) == recipe["axis_sha256"], "Incomplete original95chunks/axes")
        velocity = helpers.velocity_metrics(rna.layers["velocity"], rna.layers["vx_pred_mean"], valid, rna.obs_names, 64, run_dir / "original_prior_cell_cosines.tsv")
        velocity.pop("P_value", None)
        result = {"schema": "dynavelo_native_own_best_factual_evaluation_result_v1", "status": "COMPLETE_DESCRIPTIVE_MODEL_EVALUATION_ONLY",
                  "evaluation_id": plan["evaluation_id"], "plan_sha256": plan_sha, "recipe_sha256": plan["recipe"]["sha256"], "job": job_record,
                  "condition": "factual", "training_seed": plan["training_seed"], "training_run_id": plan["training_run_id"], "completion_evidence": plan["completion_evidence"],
                  "own_checkpoint_selection": selection, "own_best_checkpoint": plan["training_artifacts"]["best_checkpoint"],
                  "factual_inputs_sha256": q.INPUT_SHA, "axis_sha256": q.axes(rna, atac), "n_samples": 8, "chunk_cells": 64, "completed_chunks": 95,
                  "RNA_reconstruction": helpers.reconstruction_metrics(rna.X, rna.layers["x_pred_mean"], train, valid, 64),
                  "motif_reconstruction": helpers.reconstruction_metrics(atac.X, atac.layers["y_pred_mean"], train, valid, 64),
                  "original_velocity_prior_agreement": velocity, "raw_time": q.time_description(rna.obs), "random_stream_events": events,
                  "sources": PINS, "evaluator_sha256": q.sha(SELF), "adapter": adapter, "device": device,
                  "common_randomness": "Same chunk starting streams; parameter-dependent rejection sampling does not guarantee latent-wise noise matching.",
                  "comparison_scope": "Require same native recipe hash. Common inference does not remove historical CUDA/XPU/version/RNG training differences; seeds/cells are not donor replicates.",
                  "retained_annotations": "Input graphs/latent_time/embeddings retain factual provenance; no fresh trajectory graph is produced here.",
                  "actual_control_inference_performed": False, "scientific_pass": False, "P_values_generated": False, "causal_pass": False, "training_started": False}
        for obj in (rna, atac):
            obj.uns["common_factual_evaluation_json"] = q.canonical({k: result[k] for k in ("schema", "evaluation_id", "plan_sha256", "recipe_sha256", "training_run_id", "own_best_checkpoint", "scientific_pass")}).decode()
        from assemble_model_inputs import write_h5ad_atomic
        write_h5ad_atomic(rna, prediction / "rna_posterior.h5ad")
        write_h5ad_atomic(atac, prediction / "atac_posterior.h5ad")
        require(preflight(plan_path, plan_sha)[1] == raw, "Immutable evidence changed during inference")
        for k, path in inputs.items():
            require(q.sha(path) == q.INPUT_SHA[k], "Input changed during inference")
        owned_worker(job_id, local_path(plan_path), plan_sha, plan)
        require(not (run_dir / "stop_requested.json").exists() and time.monotonic() - started < plan["wall_seconds"], "Stopped or expired before completion")
        result["elapsed_seconds"] = time.monotonic() - started
        q.save_new(run_dir / "evaluation_metrics.json", result)
        files = [p for directory in (run_dir, prediction) for p in sorted(directory.rglob("*")) if p.is_file()]
        q.save_new(run_dir / "output_manifest.json", {"schema": "dynavelo_native_factual_evaluation_manifest_v1", "evaluation_id": plan["evaluation_id"],
                   "plan_sha256": plan_sha, "files": [q.file_record(ROOT, p) for p in files], "scientific_pass": False})
        q.save_new(run_dir / "COMPLETE.json", {"status": result["status"], "manifest": q.file_record(ROOT, run_dir / "output_manifest.json"),
                   "owned_terminal_still_required": True, "scientific_pass": False})
        return {k: result[k] for k in ("status", "evaluation_id", "recipe_sha256", "scientific_pass")}
    except BaseException as exc:
        q.save_new(run_dir / "FAILED.json", {"status": "INCOMPLETE_NO_RESULT_PASS", "error_type": type(exc).__name__, "detail": str(exc)[:2000], "scientific_pass": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--job-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "SOURCE_INSPECTION_ONLY", "evaluator_sha256": q.sha(SELF), "core_sha256": CORE_SHA,
                          "conditions_supported": ["factual"], "n_samples": 8, "chunk_cells": 64, "human_inputs_or_tensors_read": False,
                          "actual_XPU_inference_tested": False, "scientific_pass": False}))
        return
    if not args.plan or not args.plan_sha256 or not args.job_id:
        parser.error("Execution needs an exact plan/SHA and kernel-owned job")
    print(json.dumps(execute(args.plan, args.plan_sha256, args.job_id), allow_nan=False))


if __name__ == "__main__":
    main()
