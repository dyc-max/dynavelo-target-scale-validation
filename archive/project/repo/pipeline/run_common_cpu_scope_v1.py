"""Owned common-CPU entry for the frozen, independently reviewed scope screen.

Original scope authorization, sampling, registry and numerical streaming are
retained. Only an explicit qualified CPU inference adapter is installed.
"""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ExitStack
import argparse
import ctypes
import json
import os
import random
import re
import sys
import common_cpu_v1 as c

ROOT, SELF = c.ROOT, Path(__file__).resolve()
SUPERVISOR = ROOT / "repo/pipeline/supervise_common_cpu_scope_v1.py"
RECIPE_SHA = "6f6489e595ebf8fc1eac9a76a6ef6c50250552910ca0efbc2075193f736c24f3"
PINS = {
    "repo/colab/screen_dispatch.py": "8b064677b3ddbcdf0929774e82fa63f2f8719d30d987a1421fc1847751290db8",
    "repo/colab/dynavelo_streaming_core.py": "94a7b832e06c1f0404fd20790e8968f93724d01b732029fd91e22a06a3ae6368",
    "repo/colab/screen_scope_dispatch_v2.py": "72d0891c63f5e8fc6fe76a052f5623b67ca91545f95c69aaaef129d6351a0614",
    "repo/colab/screen_scope_execution_contract_v2.py": "8ad3f8795706ce99a6780b12f1e13658d66d1a66559e8f2597bf2ef775737b2f",
    "repo/colab/screen_scope_plan_runtime_v2.py": "29d375be2914add37846b6318e3949f8578b704496417d0ddd54a9b9d4f295e5",
    "repo/colab/screen_scope_registry_v2.py": "f6592182e275e4939478848c0c15bae2076d7eafc6f9aafccd0ec86fc563b594",
    "repo/colab/screen_scope_shortlist_v2.py": "32929ff949ee84cfd5fb3c613707602f686045113e00761c18567c9ec332b533",
    "repo/colab/screen_scope_followup_v2.py": "06dfbd016778459c2af813378688e54ed2f476090152b8daa8fdc57611f0de59",
    "repo/colab/screen_scope_followup_numerics_v2.py": "ba3d387adc939bc0b644bade8efec12e8a48fb05a70060b538ee45ddc1c05e03",
}
PRIMARY = {"baseline", "matched-baseline", "screen"}
FOLLOWUP = {"epsilon-baseline", "epsilon", "grn", "full-cell-rna", "full-cell-joint", "full-cell-grn"}
RESOURCE = {"cpu_threads": 4, "rss_bytes": 13 * 1024**3, "system_available_bytes": 3 * 1024**3,
            "forward_seconds": 120, "min_free_disk_bytes": 2 * 1024**3}


def source_check():
    for relative, expected in PINS.items():
        c.require(c.sha(ROOT / relative) == expected, "Frozen scope source differs: " + relative)
    c.require(c.sha(c.RECIPE) == RECIPE_SHA, "Common CPU recipe differs")


def request_shape(request):
    c.require(request.get("schema") == "dynavelo_common_cpu_scope_invocation_v1"
              and request.get("execution_authorized") is True and request.get("reviewer") == "/root"
              and request.get("training_authorized") is False and request.get("scientific_pass") is False
              and request.get("external_expression_access_authorized") is False, "Explicit bounded invocation required")
    c.require(re.fullmatch(r"[A-Za-z0-9_-]{8,80}", request.get("invocation_id", "")) is not None,
              "Unique invocation identity required")
    c.require(request["entry_sha256"] == c.sha(SELF) and request["supervisor_sha256"] == c.sha(SUPERVISOR)
              and request["sources"] == PINS and request["recipe_sha256"] == RECIPE_SHA, "Actual source identities required")
    c.require(type(request["training_seed"]) is int and request["training_seed"] in (0, 1, 2)
              and request["stage"] in PRIMARY | FOLLOWUP and type(request["max_tasks"]) is int
              and 1 <= request["max_tasks"] <= 256 and type(request["wall_seconds"]) is int
              and 1 <= request["wall_seconds"] <= 7200, "One factual seed and bounded stage required")
    c.require(request["trusted_reviewers"] and "/root" not in request["trusted_reviewers"], "Genuine independent existing reviewer required")
    needed = {"scope_plan", "review"}
    if request["stage"] != "baseline":
        needed.add("frame")
    if request["stage"] in FOLLOWUP:
        needed |= {"shortlist", "screen_review"}
    c.require(set(request["records"]) == needed, "Exact stage-specific records required")


def owned(request_path, request_sha, request, job_id):
    c.require(os.name == "nt" and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id or ""), "Windows-owned invocation required")
    directory = ROOT / "tmp_cache/local_owned_jobs" / job_id
    job = c.read(directory / "job.json")
    expected = [str(ROOT / "repo/.venv/Scripts/python.exe"), "-B", str(SELF), "--request", str(request_path),
                "--request-sha256", request_sha, "--job-id", job_id, "--execute"]
    c.require(job["schema"] == "dynavelo_common_cpu_scope_job_v1" and job["command"] == expected
              and job["entry_sha256"] == c.sha(SELF) and job["manifest_sha256"] == request_sha
              and job["supervisor_sha256"] == c.sha(SUPERVISOR) and job["ownership_core_sha256"] == c.OWNER_SHA
              and job["max_rss_bytes"] == RESOURCE["rss_bytes"] and job["min_free_bytes"] == RESOURCE["system_available_bytes"]
              and job["threads"] == 4 and job["wall_seconds"] == request["wall_seconds"]
              and Path(sys.executable) == Path(expected[0]) and c.sha(sys.executable) == job["python_sha256"]
              and not (directory / "STOP").exists() and not (directory / "terminal.json").exists(), "Owned command/resource binding differs")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True);inside = ctypes.c_int()
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    c.require(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)) and inside.value, "Worker lacks kernel ownership")
    return directory, c.record(directory / "job.json")


def install_runtime(stack, plan, output, audits, counters):
    import numpy as np
    import torch
    import psutil
    import screen_dispatch as base
    from qualified_runtime import use_qualified_cpu
    recipe = c.read(c.RECIPE)
    original_loader, original_contract, original_check = base.load_model, base.validate_contract, base.ResourceGuard.check
    expected_runtime = c.runtime()

    def contract_checked(context, root, review_record):
        contract = original_contract(context, root, review_record)
        cpu = contract.get("common_cpu_inference", {})
        c.require(Path(root) == ROOT and contract["device"] == "cpu" and contract.get("one_training_seed_per_invocation") is True
                  and context["spec"]["chunk_geometry"]["cells"] == 64
                  and cpu == {"schema": "dynavelo_qualified_common_cpu_fixed_v1", "recipe_sha256": RECIPE_SHA,
                              "cohort_sha256": c.COHORT_SHA, "runtime": expected_runtime,
                              "entry": c.record(SELF), "supervisor": c.record(SUPERVISOR),
                              "original_forward_audits": "call0,47,94,95,142,189 and every127th call; exact9outputs andRNG",
                              "scope_authorization_implementation_changed": False}, "Registered qualified fixed-mode CPU contract differs")
        limits = contract["resource_limits"]
        c.require(all(limits.get(key) == value for key, value in RESOURCE.items()), "Common CPU resource policy differs")
        return contract

    def guard_checked(self):
        original_check(self)
        if psutil.virtual_memory().available < RESOURCE["system_available_bytes"]:
            raise base.ResourceStop("System free-memory floor")

    def load_checked(root, context, model_record, population, contract):
        label = "F_seed" + str(model_record["seed"])
        registered = recipe["models"][label]
        c.verify_model(registered)
        c.require(model_record["run_id"] == registered["run_id"]
                  and all(model_record[key] == registered["training_artifacts"][role]
                          for key, role in (("checkpoint", "best_checkpoint"), ("config", "config"), ("metrics", "metrics"))),
                  "Screen must use the actual completed factual own-best model")
        model = original_loader(root, context, model_record, population, contract)
        original_forward = model.forward
        implementation = stack.enter_context(use_qualified_cpu(model, label, c.COHORT, c.COHORT_SHA))
        optimized_forward = model.forward
        c.q.save_new(output / "inference_implementation.json", implementation)
        def rng():
            return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        def restore(state):
            random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        def same_rng(left, right):
            return left[0] == right[0] and left[1][0] == right[1][0] and np.array_equal(left[1][1], right[1][1]) and left[1][2:] == right[1][2:] and torch.equal(left[2], right[2])
        def audited(x, y):
            index = counters["forward_calls"];counters["forward_calls"] += 1
            before = rng()
            result = optimized_forward(x, y)
            if index in {0, 47, 94, 95, 142, 189} or index % 127 == 0:
                after = rng()
                try:
                    restore(before)
                    reference = original_forward(x, y)
                    original_after = rng()
                finally:
                    restore(after)
                equal = (len(reference) == len(result) == 9 and all(a.dtype == b.dtype and a.shape == b.shape
                          and a.detach().cpu().contiguous().numpy().tobytes() == b.detach().cpu().contiguous().numpy().tobytes()
                          for a, b in zip(reference, result)))
                record = {"label": label, "forward_call_index": index, "batch_cells": len(x),
                          "nine_outputs_bitwise_equal": equal, "rng_equal": same_rng(after, original_after)}
                audits.append(record);c.q.save_new(output / ("original_forward_audit_" + str(index) + ".json"), record)
                c.require(equal and record["rng_equal"], "Qualified fixed-mode original-forward audit failed")
            return result
        model.forward = audited
        return model

    base.validate_contract, base.ResourceGuard.check, base.load_model = contract_checked, guard_checked, load_checked
    return base


def execute(a):
    path = Path(a.request).resolve()
    c.require(path.is_relative_to(ROOT) and c.sha(path) == a.request_sha256, "Exact D-project request required")
    request = c.read(path);request_shape(request);source_check()
    jd, job_record = owned(path, a.request_sha256, request, a.job_id)
    import train_hspc_local_resume as local
    local.setup_storage()
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    c.verify_recipe(RECIPE_SHA)
    sys.path.insert(0, str(ROOT / "repo/colab"))
    from screen_scope_plan_runtime_v2 import validate_scope_plan, validate_scope_frame
    import screen_scope_dispatch_v2 as primary
    import screen_scope_followup_v2 as followup
    docs = {key: c.read(c.checked(record)) for key, record in request["records"].items()}
    plan = docs["scope_plan"];validate_scope_plan(plan)
    c.require(plan["purpose"] == "human_exploratory_input_sensitivity"
              and [model["seed"] for model in sorted(plan["context_plan"]["spec"]["models"], key=lambda item: item["seed"])] == [0, 1, 2],
              "Full original factual model family required")
    if "frame" in docs:
        validate_scope_frame(plan, docs["frame"])
    output = ROOT / "checkpoints/common_cpu_scope_invocations" / request["invocation_id"]
    output.mkdir(parents=True, exist_ok=False)
    c.q.save_new(output / "STARTED.json", {"job": job_record, "request": c.record(path), "scientific_pass": False})
    audits, counters = [], {"forward_calls": 0}
    try:
        with ExitStack() as stack:
            install_runtime(stack, plan, output, audits, counters)
            kwargs = {"trusted_reviewers": request["trusted_reviewers"], "seed": request["training_seed"],
                      "max_tasks": request["max_tasks"], "shard_index": request.get("shard_index"), "stop_file": jd / "STOP"}
            root_output = ROOT / "results/screen_runs_v2" / plan["scope_plan_sha256"]
            if request["stage"] in PRIMARY:
                result = primary.dispatch(plan, request["records"]["review"], ROOT, request["stage"], root_output,
                                          frame=docs.get("frame"), **kwargs)
            else:
                result = followup.dispatch_followup(plan, docs["frame"], docs["shortlist"], request["records"]["screen_review"],
                                                   request["records"]["review"], ROOT, request["stage"], root_output, **kwargs)
        c.require(counters["forward_calls"] == 0 or audits, "Actual original-algorithm spot checks required")
        source_check();c.require(c.sha(path) == a.request_sha256, "Invocation changed during execution")
        for record in request["records"].values():
            c.checked(record)
        receipt = {"schema": "dynavelo_common_cpu_scope_receipt_v1", "job": job_record, "request": c.record(path),
                   "runtime": c.runtime(), "original_dispatch_result": result, "original_forward_audits": audits,
                   **counters, "scientific_pass": False, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "scope_authorization_implementation_changed": False, "owned_terminal_still_required": True}
        c.q.save_new(output / "worker_receipt.json", receipt)
        c.require(not result.get("error"), "Original scope dispatcher reported a failed/stopped task")
        print(json.dumps({"status": result["status"], "receipt": c.record(output / "worker_receipt.json"), **counters}), flush=True)
    except BaseException as error:
        c.q.save_new(output / "FAILED.json", {"error_type": type(error).__name__, "message": str(error),
                                             "original_forward_audits": audits, **counters, "scientific_pass": False})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request");p.add_argument("--request-sha256");p.add_argument("--job-id")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.execute:
        execute(a)
    else:
        source_check()
        print(json.dumps({"status": "SOURCE_ONLY_NO_HUMAN_EFFECTS_OR_BASELINES", "source_sha256": c.sha(SELF),
                          "original_scientific_authorization_unchanged": True, "scientific_pass": False}))
