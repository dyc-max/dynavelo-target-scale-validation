"""Recover the captured seed2 epoch92 on the same XPU runtime, never fresh.

The qualified resume implementation is compiled with identity-only substitutions.
The original optimizer, scheduler, early-stop rule, losses and Jacobians remain.
Default inspection loads small checkpoints on CPU and constructs no model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import types
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
SELF = Path(__file__).resolve()
BASE = ROOT / "repo/pipeline/train_hspc_local_resume.py"
BASE_SHA = "e6b810b1da4448135646a90db3d9e4dffeafb5382c43e6648e9e9893a56e71a5"
FRESH_SHA = "c89a55d92bd3410a386514af88bc8b6457fbf70bfffd518b3187c92fef9c1474"
PARENT_ID = "8489-MV-1_xpu_seed2_20261113T115753541561Z"
PARENT_MANIFEST = ROOT / "checkpoints/GSE284047" / PARENT_ID / "checkpoint_restore_manifest.json"
PARENT_SHA = "0ac7195062f0889bccff31c764ba16f80eebe4b2e001d16f3ffd3e73ff6c7875"
INTERRUPTION = ROOT / "results/reports/local_seed2_resource_stop_20261114_v1.json"
INTERRUPTION_SHA = "b64cf3af8a8a8386b56b65765662ad3eecddb4bcf5eaf8760b8a6e3cd050e21e"
DEADLINE = datetime.fromisoformat("2026-11-14T11:58:16.030804+00:00")
MIN_LAUNCH_FREE = int(14.25 * 1024**3)
_resource_preflight = None

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def check_launch_budget():
    """Estimate includes Torch/import memory; check before loading checkpoints."""
    global _resource_preflight
    available = psutil.virtual_memory().available
    if available < MIN_LAUNCH_FREE:
        raise RuntimeError("Recovery startup requires at least14.25GiB free system RAM")
    _resource_preflight = {"checked_utc": datetime.now(timezone.utc).isoformat(), "available_bytes": available,
                           "required_bytes": MIN_LAUNCH_FREE, "runtime_min_free_bytes": 3 * 1024**3}

def check_fresh_source(prior, historic):
    if historic != str(ROOT) or prior.get("script_sha256") != FRESH_SHA or prior.get("training_implementation_sha256") != FRESH_SHA:
        raise ValueError("Only the captured original fresh seed2 source is allowed")
    adapter = prior.get("adapter_manifest", {})
    if adapter.get("local_resume_source_sha256") != BASE_SHA or adapter.get("local_fresh_source_sha256") != FRESH_SHA or adapter.get("training_seed") != 2:
        raise ValueError("Fresh seed2 adapter identity differs")
    runtime = prior.get("local_runtime", {})
    if runtime.get("device") != "xpu" or runtime.get("torch_version") != "2.10.0+xpu" or runtime.get("threads") != 4:
        raise ValueError("Parent is not the qualified same-XPU runtime")
    for k, v in {"max_epochs": 200, "n_samples": 8, "evaluation_chunk_cells": 64, "threads": 4}.items():
        if prior["command_arguments"].get(k) != v:
            raise ValueError("Original fit or posterior recipe differs")

def adapt_source():
    if sha(BASE) != BASE_SHA:
        raise ValueError("Qualified base resume source drift")
    source = BASE.read_text(encoding="utf-8")
    edits = [
        ("FIT={'seed':1,", "FIT={'seed':2,"),
        ("RUN_RE=re.compile(r'8489-MV-1_(?:cuda|cpu|xpu)_seed1_\\d{8}T\\d{12}Z')", "RUN_RE=re.compile(r'8489-MV-1_xpu_seed2_\\d{8}T\\d{12}Z')"),
        ("elif prior.get('script_sha256')!=sha256(SELF) or prior.get('training_implementation_sha256')!=sha256(SELF):raise ValueError('Local resume requires exact execution source')", "else:_check_fresh_source(prior,historic)"),
        ("'_seed1_'+datetime.now", "'_seed2_'+datetime.now"),
        ('CUDA RNG archived, not converted; new XPU stream seeded 1 if needed', 'Same XPU runtime; saved XPU, CPU, sampler, Python and NumPy RNG required'),
        ('Restore prior XPU state if present; otherwise explicitly seed new stream with 1', 'Restore captured XPU state; missing state is fatal; no fresh stream'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("Pinned seed2 adaptation anchor differs: " + before[:70])
        source = source.replace(before, after, 1)
    return source

_source = adapt_source()
local = types.ModuleType("_qualified_seed2_same_xpu_recovery")
local.__dict__.update(__file__=str(SELF), _check_fresh_source=check_fresh_source)
exec(compile(_source, "<pinned_resume_seed2_identity_adaptation>", "exec"), local.__dict__)
local.FROZEN.update({"repo/pipeline/train_hspc_local_resume.py": BASE_SHA,
                     "repo/pipeline/train_hspc_local_fresh.py": FRESH_SHA})
_base_audit = local.audit_state
_base_inspect = local.inspect_parent
_base_execute = local.execute
_base_rng = local.rng_functions

def audit_state(state, prior, manifest):
    import numpy as np
    import torch
    result = _base_audit(state, prior, manifest)
    rng = state["rng_state"]
    values = rng.get("torch_xpu")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Exactly one saved XPU generator required; never reseed")
    if rng["torch_cuda"] != []:
        raise ValueError("Fresh XPU parent unexpectedly has CUDA RNG")
    for value in values:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0 or value.device.type != "cpu":
            raise ValueError("Malformed saved XPU RNG tensor")
    # Validate host generators using independent objects, without changing
    # process-global state or constructing a model/device context.
    random.Random().setstate(rng["python"])
    v = rng["numpy"]
    np.random.RandomState(0).set_state((v[0], np.asarray(v[1], dtype=np.uint32), v[2], v[3], v[4]))
    for key in ("torch_cpu", "train_sampler"):
        g = torch.Generator(device="cpu"); g.set_state(rng[key])
        if not torch.equal(g.get_state(), rng[key]):
            raise ValueError("Host RNG state roundtrip differs")
    result.update(saved_xpu_generators=len(values), saved_xpu_rng_sha256=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in values],
                  host_rng_roundtrip_checked=True, xpu_rng_application_pending=True)
    return result

def quiescent_capture():
    _, audit = local.pinned_json(INTERRUPTION, INTERRUPTION_SHA)
    for rec in [audit["terminal"], audit["capture"], audit["original_manifest"], *audit["required_files"].values()]:
        p = local.absolute_project_path(rec["path"])
        if p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
            raise ValueError("Quiescent capture artifact changed")
    terminal = audit["terminal_data"]
    if terminal.get("stop_reason") != ["SYSTEM_FREE_LIMIT"] or terminal.get("errors") != [] or not terminal.get("owned_job_confirmed_empty_before_close"):
        raise ValueError("Resource interruption was not cleanly owned and closed")
    for rec in terminal["observed_processes"]:
        try:
            if psutil.Process(rec["pid"]).create_time() == rec["create_time"]:
                raise ValueError("Original parent process identity still exists")
        except psutil.NoSuchProcess:
            pass
    return audit

def inspect_parent(path, digest, device="cpu"):
    if Path(path).resolve() != PARENT_MANIFEST.resolve() or digest != PARENT_SHA:
        raise ValueError("Only the immutable captured seed2 epoch92 parent is allowed")
    audit = quiescent_capture()
    binding = _base_inspect(path, digest, device)
    if device == "xpu" and binding["actual_package_versions"]["torch"] != "2.10.0+xpu":
        raise ValueError("Same XPU PyTorch2.10.0 runtime required")
    if binding["manifest"]["completed_epoch"] != 92 or binding["prior_config"].get("training_seed") != 2 or binding["state_summary"]["already_stopped"]:
        raise ValueError("Expected incomplete seed2 epoch92 checkpoint")
    binding["recovery_identity"] = {"schema": "dynavelo_seed2_same_xpu_recovery_v1", "interruption_audit_sha256": INTERRUPTION_SHA,
        "capture_manifest_sha256": audit["capture"]["sha256"], "base_source_sha256": BASE_SHA,
        "compiled_identity_adaptation_sha256": hashlib.sha256(_source.encode()).hexdigest(),
        "original_deadline_utc": DEADLINE.isoformat(), "resume_next_epoch": 93, "fresh_restart_allowed": False,
        "XPU_bitwise_numerical_equivalence_claim": False}
    return binding

def rng_functions(device):
    capture, _ = _base_rng(device)
    def restore(state, loader):
        import numpy as np
        import torch
        if device != "xpu" or not isinstance(state.get("torch_xpu"), list) or len(state["torch_xpu"]) != torch.xpu.device_count():
            raise ValueError("Missing or mismatched saved XPU RNG; fresh stream forbidden")
        random.setstate(state["python"])
        v = state["numpy"]; np.random.set_state((v[0], np.asarray(v[1], dtype=np.uint32), v[2], v[3], v[4]))
        torch.set_rng_state(state["torch_cpu"].cpu())
        loader.batch_sampler.generator.set_state(state["train_sampler"].cpu())
        torch.xpu.set_rng_state_all([v.cpu() for v in state["torch_xpu"]])
        checks = [(torch.get_rng_state(), state["torch_cpu"]), (loader.batch_sampler.generator.get_state(), state["train_sampler"])]
        checks.extend(zip(torch.xpu.get_rng_state_all(), state["torch_xpu"]))
        if any(not torch.equal(a.cpu(), b.cpu()) for a, b in checks):
            raise ValueError("Applied device/host/sampler RNG bytes differ")
    return capture, restore

def execute(binding, action, threads, run_id=None, probe_batches=2, device_evidence=None, device_evidence_sha=None):
    if binding["target_device"] != "xpu" or threads != 4 or probe_batches != 2:
        raise ValueError("Recovery requires same XPU, four threads and two discarded probe batches")
    if datetime.now(timezone.utc) >= DEADLINE:
        raise ValueError("Original deadline reached; no automatic extension")
    if _resource_preflight is None or _resource_preflight["required_bytes"] != MIN_LAUNCH_FREE:
        raise RuntimeError("Pre-import resource check is required before recovery execution")
    if psutil.virtual_memory().available < 3 * 1024**3:
        raise RuntimeError("Runtime3GiB memory floor already crossed")
    return _base_execute(binding, action, threads, run_id, probe_batches, device_evidence, device_evidence_sha)

local.audit_state = audit_state
local.inspect_parent = inspect_parent
local.rng_functions = rng_functions
local.execute = execute

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=local.absolute_project_path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    p.add_argument("--action", choices=("inspect", "short-test", "train"), default="inspect")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--probe-batches", type=int, default=2)
    p.add_argument("--run-id")
    p.add_argument("--report", type=local.absolute_project_path)
    p.add_argument("--device-evidence")
    p.add_argument("--device-evidence-sha256")
    a = p.parse_args()
    local.setup_storage()
    if a.action != "inspect" and not a.execute:
        p.error("Execution must be explicit and supervised")
    if a.action != "inspect":
        check_launch_budget()
    binding = inspect_parent(a.manifest, a.manifest_sha256, a.device)
    if a.report:
        if a.report.exists(): raise FileExistsError("Inspection report must be new")
        a.report.parent.mkdir(parents=True, exist_ok=True); local.write_json(a.report, binding)
    result = {k: binding[k] for k in ("status", "state_summary", "actual_package_versions", "local_source_sha256", "model_constructed", "recovery_identity")} if a.action == "inspect" else execute(binding, a.action, a.threads, a.run_id, a.probe_batches, a.device_evidence, a.device_evidence_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
