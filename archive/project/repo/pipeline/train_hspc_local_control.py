"""Native fresh V/M control fitting, using the unchanged qualified fit loop.

Default is source inspection. Production requires complete factual-scope review,
the fixed real export, an exact recipe and an exclusive kernel-owned GPU job.
"""
from __future__ import annotations
import argparse
import ast
import ctypes
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
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
SELF = Path(__file__).resolve()
FRESH_REL = "repo/pipeline/train_hspc_local_fresh.py"
FRESH_SHA = "c89a55d92bd3410a386514af88bc8b6457fbf70bfffd518b3187c92fef9c1474"
LOCAL_SHA = "e6b810b1da4448135646a90db3d9e4dffeafb5382c43e6648e9e9893a56e71a5"
EXPORTER_SHA = "fcb72cd30c5aec7ee18b6d88daa022f6a5fcb307bf5b384cd48cd4b1337c0f1b"
AUDIT_REL = "results/reports/native_control_export_audit_v1/audit.json"
AUDIT_SHA = "9acb5736e9690d91c8c8b3ed6b1e308aa96df5cfa9bd2795afd15c03ba46c120"
OWNER_SHA = "14f9b61c72ab2cde44a859288904ca34a61d7e6a8114a79177ae20d23e29995b"
SEED2_INITIAL_SHA = "eb70d59041ecdc364eda271f0d638c3da365d3d2970c8f1cfef30130fac85c9b"
CONTROL_MANIFEST_SHA = {"V": "a83040c95321605815408acb0c9c9a37f8a7184428c670051bf608b6e6417739",
                        "M": "f3dd494a2024bc851dcd4a89291e9cbb8069fa0e827596ba988e479cce1bfbc4"}
RUN_RE = re.compile(r"8489-MV-1_xpu_control([VM])_seed([012])_\d{8}T\d{12}Z\Z")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


for _relative, _expected in (("repo/pipeline/train_hspc_local_resume.py", LOCAL_SHA), (FRESH_REL, FRESH_SHA),
                             ("repo/pipeline/export_training_control_inputs_v2.py", EXPORTER_SHA)):
    if sha(ROOT / _relative) != _expected:
        raise RuntimeError("Frozen local source changed: " + _relative)
import train_hspc_local_resume as local
import export_training_control_inputs_v2 as exporter

PINS = {**local.FROZEN, FRESH_REL: FRESH_SHA, "repo/pipeline/train_hspc_local_resume.py": LOCAL_SHA,
        "repo/pipeline/export_training_control_inputs_v2.py": EXPORTER_SHA,
        "repo/pipeline/supervise_local_resume.py": OWNER_SHA,
        "repo/pipeline/audit_native_control_exports.py": "eddfb437fe6d317959c2e2863f0ca400eda18c45b6c7c9a82df8586c4725e230"}


class ProbeComplete(Exception):
    pass


def require(yes, message):
    if not yes:
        raise ValueError(message)


def recipe(seed):
    require(type(seed) is int and seed in (0, 1, 2), "Original three initialization seeds required")
    return {**local.FIT, "seed": seed, "max_epochs": 200, "n_samples": 8, "evaluation_chunk_cells": 64, "threads": 4}


def source_check():
    for rel, expected in PINS.items():
        require(sha(ROOT / rel) == expected, "Frozen source differs: " + rel)


def verified_control_input(condition):
    require(condition in CONTROL_MANIFEST_SHA, "Only fixed V/M inputs")
    source_check()
    _, audit = local.pinned_json(ROOT / AUDIT_REL, AUDIT_SHA)
    require(audit.get("status") == "PASS_SERIALIZED_INPUTS_AND_DESCRIPTIVE_DIAGNOSTICS_ONLY" and audit.get("map_resampled") is False,
            "Completed actual input audit required")
    rows = [x for x in audit["audits"] if x["control"] == condition]
    require(len(rows) == 1, "Exactly one fixed export per condition")
    row = rows[0]
    manifest_path = exporter.verify_record(ROOT, row["export_manifest"])
    require(row["export_manifest"]["sha256"] == CONTROL_MANIFEST_SHA[condition], "Original fixed control manifest differs")
    manifest = exporter.read(manifest_path)
    require(manifest.get("status") == "PASS_EXPORTED_INPUT_INTEGRITY_ONLY" and manifest.get("fixture_only") is False
            and manifest.get("control") == condition, "Completed real export required")
    lineage = manifest["lineage"]
    require(lineage.get("schema") == "dynavelo_training_control_lineage_v2" and lineage.get("control") == condition
            and lineage.get("factual_inputs") == exporter.FACTUAL and lineage.get("coded_subject_provenance") == exporter.CODED_CONTEXT
            and lineage.get("shared_initialization_seeds") == [0, 1, 2] and lineage.get("permutation_seed") == exporter.SEEDS[condition]
            and lineage.get("exporter_sha256") == EXPORTER_SHA, "Registered control lineage differs")
    complete = exporter.read(manifest_path.parent / "COMPLETE.json")
    require(complete.get("manifest") == row["export_manifest"] and complete.get("plan_sha256") == manifest["plan_sha256"], "Export completion marker differs")
    terminal = exporter.read(exporter.verify_record(ROOT, row["owned_terminal"]))
    job = exporter.read(exporter.verify_record(ROOT, row["owned_job"]))
    require(terminal.get("exit_code") == 0 and terminal.get("stop_reason") == [] and terminal.get("errors") == []
            and terminal.get("owned_job_confirmed_empty_before_close") is True, "Export did not terminate cleanly")
    require(job.get("entry_sha256") == EXPORTER_SHA and job.get("manifest_sha256") == manifest["plan_sha256"], "Export job differs")
    for rec in [*manifest["outputs"].values(), *exporter.FACTUAL.values()]:
        exporter.verify_record(ROOT, rec)
    return {"manifest": row["export_manifest"], "outputs": manifest["outputs"], "lineage": lineage,
            "audit": {"relative_path": AUDIT_REL, "bytes": (ROOT / AUDIT_REL).stat().st_size, "sha256": AUDIT_SHA}}


def validate_recipe_shape(plan):
    require(plan.get("schema") == "dynavelo_native_control_training_recipe_v1" and plan.get("reviewer") == "/root", "Root control recipe required")
    require(plan.get("condition") in ("V", "M") and plan.get("recipe") == recipe(plan.get("training_seed")), "Fixed control/fit recipe differs")
    require(plan.get("entry_sha256") == sha(SELF) and plan.get("sources") == PINS and plan.get("model_defaults") == local.ARCH, "Source/architecture differs")
    require(plan.get("control_input_audit_sha256") == AUDIT_SHA and plan.get("candidate_access_authorized") is False
            and plan.get("external_expression_access_authorized") is False and plan.get("scientific_pass") is False,
            "Control-only scope required")
    require(plan.get("initialization_seeds") == [0, 1, 2] and plan.get("planned_total_control_fits") == 6
            and plan.get("seed2_initial_logical_sha256") == SEED2_INITIAL_SHA, "Control family or initial-state rule differs")
    require(type(plan.get("execution_authorized")) is bool and type(plan.get("training_authorized")) is bool, "Explicit authority fields required")
    require(plan.get("resource_policy") == {"threads": 4, "max_rss_bytes": 13 * 1024**3, "min_free_bytes": 3 * 1024**3,
            "short_test_wall_seconds": 300, "training_wall_seconds": 86400, "automatic_extension_or_restart": False}, "Original bounded local resource policy required")


def validate_factual_scope(plan):
    """Future actual completion plus root scientific review; never infer from maps."""
    rec = plan["factual_scope_review"]
    report = exporter.read(exporter.verify_record(ROOT, rec))
    require(report.get("schema") == "dynavelo_completed_factual_scope_review_v1" and report.get("reviewer") == "/root"
            and report.get("decision") == "ALLOW_FROZEN_DC_CONTROLS" and report.get("scientific_pass") is False
            and report.get("target_cells") == 1884 and report.get("context_cells") == 3453
            and report.get("training_cells") == 6035 and report.get("full_HSPC_pass_claim") is False,
            "Explicit actual completed factual DC scope review required")
    evidence = report.get("completed_factual_evidence", {})
    require(set(evidence) == {"0", "1", "2"}, "All three actual factual completions required")
    for value in evidence.values():
        exporter.verify_record(ROOT, value)
    require(evidence["0"]["sha256"] == "9cc265a708935ed7e7c5381b143d376ea797990e28645f1e7cecf050fda15cea"
            and evidence["1"]["sha256"] == "2a946c80ef5ea5ad27add9fc7edafcdda840c76a19ce397558ad6f387cc6dc98", "Verified earlier factual completions differ")
    require(sha(ROOT / "repo/pipeline/audit_local_fresh_readout.py") == "047bc3646e619f4336bccf0b1ab55351ab12f9c5fbb5d810b75afb927c6e1567", "Actual fresh completion adapter differs")
    import audit_local_fresh_readout as fresh_audit
    actual = exporter.read(ROOT / evidence["2"]["relative_path"])
    fresh_audit.validate_capture(ROOT / evidence["2"]["relative_path"], evidence["2"]["sha256"], actual["run_id"])
    return rec


def inspect_manifest(path, digest, *, require_execution=False):
    raw, plan = local.pinned_json(path, digest)
    validate_recipe_shape(plan)
    verified = verified_control_input(plan["condition"])
    require(plan.get("control_export_manifest") == verified["manifest"] and plan.get("inputs") == verified["outputs"], "Recipe not bound to exact fixed control export")
    if require_execution:
        require(plan["execution_authorized"] is True and plan["training_authorized"] is True, "Reviewed model execution authority absent")
        validate_factual_scope(plan)
        versions = local.source_check("xpu")
        require(versions["torch"] == "2.10.0+xpu", "Qualified native XPU environment required")
    else:
        versions = None
    _, _, compiled, _ = construct(plan["training_seed"], plan["condition"], verified)
    return {"status": "INSPECTED_CONTROL_INPUT_BYTES_NO_MODEL", "manifest_path": str(path), "manifest_sha256": digest,
            "plan": plan, "control": verified, "recipe": recipe(plan["training_seed"]), "input_files": {x["relative_path"]: x for x in verified["outputs"].values()},
            "compiled_sources": compiled, "versions": versions, "model_constructed": False, "scientific_pass": False}


def function_source(name):
    text = (ROOT / FRESH_REL).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


def construct(seed, condition, verified, probe=False):
    """Compile frozen fresh-run functions; device adapter creation is deferred."""
    source_check()
    require(condition in ("V", "M"), "V/M condition required")
    recipe(seed)
    ns = {**globals(), "_CONTROL_SEED": seed, "_CONTROL_CONDITION": condition, "_CONTROL_INPUTS": verified["outputs"],
          "RECIPE": recipe(seed), "FIT": {**local.FIT, "seed": seed}}
    initial = function_source("initial_state")
    initial = local.once(initial, 'config["training_seed"] != 2', 'config["training_seed"] != _CONTROL_SEED')
    initial = local.once(initial, '"training_seed": 2', '"training_seed": _CONTROL_SEED, "control": _CONTROL_CONDITION')
    initial = local.once(initial, '    checkpoint = directory / "initial_model_state.pth"',
        '    if _CONTROL_SEED == 2 and h.hexdigest() != SEED2_INITIAL_SHA:\n        raise ValueError("Native controlseed2 initialization differs from actual factualseed2")\n    checkpoint = directory / "initial_model_state.pth"')
    probe_code = local.once(function_source("probe_fit"), '"training_seed": 2', '"training_seed": _CONTROL_SEED, "control": _CONTROL_CONDITION')
    probe_code = local.once(probe_code, 'actual fresh run starts seed2 initialization again.', 'actual control run starts its registered initialization again.')
    build = function_source("build_modules")
    build = local.once(build, 'load = local.local_adapter("xpu")', 'load = _deferred_loader')
    build = local.once(build, '"local_fresh_source_sha256": sha(SELF), "training_seed": 2',
                       '"local_control_source_sha256": sha(SELF), "training_seed": _CONTROL_SEED, "control": _CONTROL_CONDITION')
    build = local.once(build, '        initial_state(model, config, directory)', '        _validate_runtime_control(config, args)\n        initial_state(model, config, directory)')
    def deferred_loader(*args, **kwargs):
        return local.local_adapter("xpu")(*args, **kwargs)
    def runtime_control(config, args):
        require(config.get("input_assembly", {}).get("training_control") == verified["lineage"], "Actually loaded control lineage differs")
        require(config.get("training_seed") == seed and vars(args)["seed"] == seed, "Actually loaded control seed differs")
        require(config["input_files"] == {str(ROOT / x["relative_path"]): x["sha256"] for x in verified["outputs"].values()}, "Actually loaded controlled input bytes differ")
    ns.update(_deferred_loader=deferred_loader, _validate_runtime_control=runtime_control)
    exec(compile(initial + "\n\n" + probe_code + "\n\n" + build, "<fresh_original_fit_control_identity_adapter>", "exec"), ns)
    helper, runner, compiled = ns["build_modules"](probe=probe)
    compiled["control_binding_source_sha256"] = hashlib.sha256((initial + "\n\n" + probe_code + "\n\n" + build).encode()).hexdigest()
    # The actual fit loop and compiled numerical runner MUST be identical to the
    # currently qualified factual fresh entry, before any model is instantiated.
    require(compiled["helper_sha256"] == "96f4e552a192762632f976f7c136a6328756fd92dafe243f0c6a345037f0d9ac"
            and compiled["runner_sha256"] == "03b118bbf1ee0263adcd54cc02dc8c5ed3033b908af2da0c80c9577987d28ca3", "Original numerical training source differs")
    ns["build_modules"] = lambda probe=False: construct(seed, condition, verified, probe)[:3]
    execute_source = function_source("execute")
    execute_source = local.once(execute_source, 'unique seed2 run ID required', 'unique native control run ID required')
    execute_source = local.once(execute_source, '"fresh_input_binding.json"', '"control_input_binding.json"')
    execute_source = local.once(execute_source, '"scientific_recipe_changed": False',
        '"numerical_fit_recipe_changed": False, "control_condition": _CONTROL_CONDITION,\n        "training_seed": _CONTROL_SEED, "base_fresh_source_sha256": FRESH_SHA')
    execute_source = local.once(execute_source, 'rna=ROOT / local.INPUTS[0], atac=ROOT / local.INPUTS[1]',
                                'rna=ROOT / _CONTROL_INPUTS["rna"]["relative_path"], atac=ROOT / _CONTROL_INPUTS["atac"]["relative_path"]')
    exec(compile(execute_source, "<same_fresh_executor_with_fixed_control_inputs>", "exec"), ns)
    compiled["control_executor_source_sha256"] = hashlib.sha256(execute_source.encode()).hexdigest()
    return helper, runner, compiled, ns["execute"]


def validate_probe(args, binding):
    require(args.device_evidence and args.device_evidence_sha256 and args.probe_terminal and args.probe_terminal_sha256, "Actual completed same-recipe short test required")
    _, probe = local.pinned_json(args.device_evidence, args.device_evidence_sha256)
    _, terminal = local.pinned_json(args.probe_terminal, args.probe_terminal_sha256)
    job = exporter.read(Path(args.probe_terminal).parent / "job.json")
    probe_run = Path(args.device_evidence).parent.name
    require(RUN_RE.fullmatch(probe_run) and job.get("run_id") == probe_run and job.get("job_id") == terminal.get("job_id"), "Own short-test run identity differs")
    require(job.get("manifest_sha256") == binding["manifest_sha256"] and job.get("entry_sha256") == sha(SELF)
            and job.get("supervisor_sha256") == binding["plan"]["supervisor_sha256"]
            and job["command"][job["command"].index("--action") + 1] == "short-test", "Actual probe source/recipe/command differs")
    require(terminal.get("exit_code") == 0 and terminal.get("stop_reason") == [] and terminal.get("errors") == []
            and terminal.get("owned_job_confirmed_empty_before_close") is True, "Probe owned job did not terminate cleanly")
    expected = {"status": "SHORT_TEST_ONLY_NOT_RESUMABLE", "source_sha256": sha(SELF), "recipe_manifest_sha256": binding["manifest_sha256"],
                "control": binding["plan"]["condition"], "training_seed": binding["plan"]["training_seed"], "train_batches": 2, "validation_batches": 1,
                "original_jacobian_forward_backward_adam": True, "research_checkpoint_saved": False}
    require(all(probe.get(k) == v for k, v in expected.items()), "Probe is not this control/initialization recipe")


def owned_worker(args, plan):
    require(os.name == "nt" and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.job_id or ""), "Explicit Windows-owned job required")
    directory = ROOT / "tmp_cache/local_owned_jobs" / args.job_id
    job = exporter.read(directory / "job.json")
    policy = plan["resource_policy"]
    expected_command = [str(ROOT / "repo/.venv-xpu/Scripts/python.exe"), "-B", str(SELF), *sys.argv[1:]]
    require(job.get("schema") == "dynavelo_native_control_training_job_v1" and job.get("command") == expected_command
            and job.get("manifest_sha256") == args.manifest_sha256 and job.get("entry_sha256") == sha(SELF)
            and job.get("run_id") == args.run_id and job.get("job_id") == args.job_id
            and job.get("ownership_core_sha256") == OWNER_SHA and job.get("supervisor_sha256") == plan["supervisor_sha256"]
            and job.get("wall_seconds") == policy["short_test_wall_seconds" if args.action == "short-test" else "training_wall_seconds"]
            and job.get("max_rss_bytes") == policy["max_rss_bytes"] and job.get("min_free_bytes") == policy["min_free_bytes"], "Actual control ownership/command differs")
    require(sha(ROOT / "repo/pipeline/supervise_local_control.py") == plan["supervisor_sha256"] and Path(sys.executable) == Path(expected_command[0])
            and sha(sys.executable) == job["python_sha256"], "Actual supervisor or interpreter differs")
    value = ctypes.c_int()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    require(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(value)) and value.value, "Actual worker lacks kernel ownership")
    require(not (directory / "STOP").exists() and not (directory / "terminal.json").exists(), "Job stopped or already terminal")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=local.absolute_project_path)
    p.add_argument("--manifest-sha256")
    p.add_argument("--action", choices=("inspect", "short-test", "train"), default="inspect")
    p.add_argument("--run-id")
    p.add_argument("--job-id")
    p.add_argument("--device-evidence", type=local.absolute_project_path)
    p.add_argument("--device-evidence-sha256")
    p.add_argument("--probe-terminal", type=local.absolute_project_path)
    p.add_argument("--probe-terminal-sha256")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.action == "inspect" and not a.manifest:
        source_check()
        print(json.dumps({"status": "SOURCE_INSPECTION_ONLY_NO_INPUT_OR_GPU_READ", "source_sha256": sha(SELF), "conditions": ["V", "M"],
                          "seeds": [0, 1, 2], "original_fit_source_sha256": FRESH_SHA, "scientific_pass": False}))
        return
    if not a.manifest or not a.manifest_sha256:
        p.error("Exact recipe manifest/SHA required")
    if a.action == "inspect":
        binding = inspect_manifest(a.manifest, a.manifest_sha256)
        print(json.dumps({k: v for k, v in binding.items() if k not in ("compiled_sources", "plan")}, allow_nan=False))
        return
    require(a.execute, "Explicit execution flag required")
    _, plan = local.pinned_json(a.manifest, a.manifest_sha256)
    validate_recipe_shape(plan)
    match = RUN_RE.fullmatch(a.run_id or "")
    require(match and match[1] == plan["condition"] and int(match[2]) == plan["training_seed"], "Run does not name its actual control/seed")
    owned_worker(a, plan)
    binding = inspect_manifest(a.manifest, a.manifest_sha256, require_execution=True)
    if a.action == "train":
        validate_probe(a, binding)
    _, _, _, run = construct(plan["training_seed"], plan["condition"], binding["control"], probe=a.action == "short-test")
    try:
        result = run(a, binding)
    finally:
        verified_control_input(plan["condition"])
    print(json.dumps({**result, "control": plan["condition"], "seed": plan["training_seed"]}, allow_nan=False))


if __name__ == "__main__":
    main()
