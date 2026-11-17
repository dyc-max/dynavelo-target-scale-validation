"""Completed fresh-XPU seed2 evidence and reused factual reconstruction readout.

Inspect by default. No live checkpoint/prediction reads, model execution or
cloud evidence substitution. Pairwise seed comparison is a separate interface.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import audit_local_factual_readout as old

q = old.q
ROOT = q.ROOT
SELF = Path(__file__).resolve()
RUN = re.compile(r"8489-MV-1_xpu_seed2_\d{8}T\d{12}Z\Z")
ENTRY = "repo/pipeline/train_hspc_local_fresh.py"
ENTRY_SHA = "c89a55d92bd3410a386514af88bc8b6457fbf70bfffd518b3187c92fef9c1474"
SUPERVISOR = "repo/pipeline/supervise_local_fresh.py"
SUPERVISOR_SHA = "1d08365a99002f9165362c80cef06a24755f5f25ce423ba698c7ff43a7eb6414"
CORE_SHA = "14f9b61c72ab2cde44a859288904ca34a61d7e6a8114a79177ae20d23e29995b"
PINS = {**old.PINS, old.SUPERVISOR: CORE_SHA, ENTRY: ENTRY_SHA, SUPERVISOR: SUPERVISOR_SHA,
        "repo/pipeline/audit_local_factual_readout.py": "523a9f5e4b2fb6b6dc85c7f0fa83fb716f2071b92562ae2779ba68a4263f7f23"}
MIGRATION_NOTE = ("Seed2 is initialized afresh on local XPU. Seed0 used CUDA and seed1 migrated CUDA to XPU; "
    "comparisons mix initialization, backend/software and RNG history. Seeds/cells are not donor replicates. "
    "XPU operator fallback to CPU is retained; no pure-GPU, bitwise-equivalence or calibrated-time claim.")
require, local, raw, pinned = old.require, old.local, old.raw, old.pinned
function_source, once, check_output_id = old.function_source, old.once, old.check_output_id

def validate_terminal(value, job_id):
    require(value.get("job_id") == job_id and value.get("status") == "PROCESS_TERMINATED_NO_SCIENTIFIC_PASS",
            "Wrong terminal identity")
    require(type(value.get("exit_code")) is int and value["exit_code"] == 0 and value.get("errors") == []
            and value.get("stop_reason") == [] and value.get("owned_job_confirmed_empty_before_close") is True,
            "HOLD: clean normal terminal and empty owned job required")

def final_summary(text):
    found = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"^\{", text, re.M):
        try:
            value, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"run_id", "action", "run_dir", "scientific_pass"}:
            found.append((value, match.start() + end))
    require(len(found) == 1 and not text[found[0][1]:].strip(), "One exact final entry summary must end owned console")
    return found[0][0]

def completed_recipe(config, metrics, progress, run):
    source = function_source(old, "validate_local_terminal")
    source = once(source, "config.get('training_seed')==1", "config.get('training_seed')==2")
    source = once(source, "recipe={'seed':1", "recipe={'seed':2")
    source = once(source, "get('bitwise_continuation_claim')", "get('bitwise_CUDA_equivalence_claim')")
    ns = {"require": require}
    exec(compile(source, "<same_terminal_checks_fresh_seed2_identity>", "exec"), ns)
    return ns["validate_local_terminal"](config, metrics, progress, run, "xpu")

def build_capture(root, job_id, run, job_sha, terminal_sha):
    require(RUN.fullmatch(run or "") and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id or ""), "Exact fresh job/run required")
    directory = q.safe(root, "tmp_cache/local_owned_jobs/" + job_id)
    # Termination is checked before any training artifact is opened.
    terminal = pinned(directory / "terminal.json", terminal_sha)
    validate_terminal(terminal, job_id)
    job = pinned(directory / "job.json", job_sha)
    require(job.get("schema") == "dynavelo_local_fresh_owned_job_v1" and job.get("job_id") == job_id
            and job.get("run_id") == run and local(root, job["job_dir"]) == directory, "Wrong owned fresh job")
    require(job.get("entry_sha256") == ENTRY_SHA and job.get("supervisor_sha256") == SUPERVISOR_SHA
            and job.get("ownership_core_sha256") == CORE_SHA, "Owned job source changed")
    fixed = old.constants(root)
    records = []
    for rel, digest in {**PINS, **fixed["FROZEN"]}.items():
        p = q.safe(root, rel)
        require(q.sha(p) == digest, "Frozen source drift: " + rel)
        records.append(q.record(p, root))
    def use(p):
        p = local(root, p)
        records.append(q.record(p, root))
        return q.read(p)
    launch = use(directory / "launcher.json")
    require(launch.get("kernel_job_assigned_before_execution") is True and any(
        r.get("pid") == launch.get("pid") and r.get("create_time") == launch.get("create_time")
        and r.get("command") == job.get("command") for r in terminal.get("observed_processes", [])), "Owned launcher identity missing")
    command = job["command"]
    require(command[1] == "-B" and local(root, command[2]) == q.safe(root, ENTRY), "Wrong actual entry")
    options = {}; i = 3
    while i < len(command):
        key = command[i]
        require(key.startswith("--") and key not in options, "Duplicate command option")
        options[key] = True if key == "--execute" else command[i + 1]
        i += 1 if key == "--execute" else 2
    require(set(options) == {"--manifest", "--manifest-sha256", "--action", "--run-id", "--execute", "--device-evidence", "--device-evidence-sha256"}
            and options["--action"] == "train" and options["--run-id"] == run, "Wrong fresh command scope")
    interpreter = local(root, command[0])
    require(q.sha(interpreter) == job["python_sha256"], "Interpreter differs")
    recipe_path = local(root, options["--manifest"])
    recipe = pinned(recipe_path, options["--manifest-sha256"])
    require(job["manifest_sha256"] == options["--manifest-sha256"] and recipe.get("entry_sha256") == ENTRY_SHA
            and recipe.get("schema") == "dynavelo_local_fresh_seed2_recipe_v1", "Wrong actual recipe")
    base = q.safe(root, "checkpoints/GSE284047/" + run)
    pred = q.safe(root, "data/processed/GSE284047/8489-MV-1/model_runs/" + run)
    require(not (base / "failure.json").exists() and not (base / "short_test_completion.json").exists(), "Failed/probe run is not factual completion")
    summary = final_summary(raw(directory / "console.log").decode("utf-8-sig"))
    require(summary == {"run_id": run, "action": "train", "run_dir": str(base), "scientific_pass": False}, "Final fresh summary differs")
    config, metrics, progress = [use(base / name) for name in ("config.json", "metrics.json", "evaluation_progress.json")]
    best = completed_recipe(config, metrics, progress, run)
    require(config["script_sha256"] == config["training_implementation_sha256"] == ENTRY_SHA, "Fresh source mislabeled")
    require(config["model_defaults"] == fixed["ARCH"] and all(config["command_arguments"][k] == v for k, v in recipe["recipe"].items()), "Fresh recipe differs")
    require(config["command_arguments"]["resume_checkpoint"] is None and config["command_arguments"]["resume_manifest"] is None, "Unexpected resume")
    require(config["command_arguments"]["recipe_manifest_sha256"] == job["manifest_sha256"], "Recipe not bound in actual config")
    adapter = config["adapter_manifest"]
    require(adapter.get("local_fresh_source_sha256") == ENTRY_SHA and adapter.get("local_resume_source_sha256") == old.ENTRY_SHA
            and adapter.get("device") == "xpu" and adapter.get("analytic_velocity") is False, "Device adapter differs")
    require(config["package_versions"]["torch"] == config["local_runtime"]["torch_version"] == "2.10.0+xpu", "Actual Torch differs")
    adaptation = use(base / "source_adaptation.json")
    for kind in ("helper", "runner"):
        p = base / (kind + "_compiled.py.txt")
        require(hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() == adaptation[kind + "_sha256"], "Compiled source text differs")
        records.append(q.record(p, root))
    expected_inputs = {}
    for key, rel in zip(("rna", "atac"), fixed["INPUTS"]):
        p = q.safe(root, rel); rec = q.record(p, root); records.append(rec)
        require(rec["sha256"] == q.INPUT_SHA[key] == recipe["input_files"][rel]["sha256"] and rec["bytes"] == recipe["input_files"][rel]["bytes"], "Original input differs")
        require(local(root, config["command_arguments"][key]) == p, "Input path differs")
        expected_inputs[str(p)] = rec["sha256"]
    require(config["input_files"] == expected_inputs and "training_control" not in config.get("input_assembly", {}), "Not original factual inputs")
    require(local(root, config["prediction_dir"]) == pred and local(root, config["checkpoint_dir"]) == base, "Output path differs")
    restore = use(base / "checkpoint_restore_manifest.json")
    require(restore.get("schema") == "dynavelo_colab_checkpoint_restore_v1" and restore.get("run_id") == run
            and local(root, restore["cloud_root"]) == root and restore.get("completed_epoch") == metrics["epochs_completed"], "Final local recovery identity differs")
    roles = {r["role"]: r for r in restore["required_files"]}
    names = {"checkpoint": {"last_checkpoint.pth"}, "prior_config": {"resume_config.json"}, "best_snapshot": {"best_snapshot_0.pth", "best_snapshot_1.pth"}}
    require(len(restore["required_files"]) == 3 and set(roles) == set(names), "Recovery role census differs")
    for role, rec in roles.items():
        p = q.checked(root, rec)
        require(p.parent == base and p.name in names[role], "Recovery outside this run")
        records.append(q.record(p, root))
    require(local(root, metrics["best_checkpoint"]) == base / "model_best.pth" and q.sha(base / "model_best.pth")
            == metrics["best_checkpoint_sha256"] == roles["best_snapshot"]["sha256"], "Own best weight mismatch")
    require(set(metrics["prediction_files"]) == {str(pred / ("adata_" + k + "_pred.h5ad")) for k in ("rna", "atac")}, "Prediction census differs")
    for name, digest in metrics["prediction_files"].items():
        rec = q.record(local(root, name), root); require(rec["sha256"] == digest, "Final prediction digest differs"); records.append(rec)
    gate = use(base / "training_gate.json")
    require(all(metrics.get(k) == v for k, v in gate.items()), "Training gate differs")
    require([int(v) for v in re.findall(r"^n:\s+(\d+)\s*$", raw(base / "console.log").decode("utf-8-sig"), re.M)] == list(range(8)) * 95, "Posterior draw log incomplete")
    short_path = local(root, options["--device-evidence"])
    short = pinned(short_path, options["--device-evidence-sha256"])
    require(short.get("status") == "SHORT_TEST_ONLY_NOT_RESUMABLE" and short.get("source_sha256") == ENTRY_SHA
            and short.get("recipe_manifest_sha256") == job["manifest_sha256"], "Fresh short-test evidence differs")
    initialization = use(base / "initialization.json")
    require(initialization["archive_sha256"] == q.sha(base / "initial_model_state.pth")
            and initialization["logical_state_sha256"] == use(short_path.parent / "initialization.json")["logical_state_sha256"], "Fresh initialization evidence differs")
    for p in [recipe_path, interpreter, short_path, *(directory / n for n in ("job.json", "terminal.json", "status.json", "console.log")),
              *(base / n for n in ("initial_model_state.pth", "fresh_input_binding.json", "model_best.pth", "training_metrics.tsv", "validation_velocity_cosines.tsv", "console.log"))]:
        records.append(q.record(p, root))
    unique = {r["relative_path"]: r for r in records}
    for r in records:
        require(unique[r["relative_path"]] == r, "Source changed during capture"); q.checked(root, r)
    return {"schema": "dynavelo_local_fresh_completion_v1", "status": "LOCAL_OWNED_TERMINAL_AND_ARTIFACTS_VERIFIED_NO_SCIENTIFIC_PASS",
        "run_id": run, "seed": 2, "job_id": job_id, "job_sha256": job_sha, "terminal_sha256": terminal_sha, "best_epoch": best,
        "execution": {"local_runtime": config["local_runtime"], "recipe_sha256": job["manifest_sha256"], "fresh_initialization": True},
        "migration_note": MIGRATION_NOTE, "source_records": list(unique.values()), "script_sha256": q.sha(SELF),
        "scientific_pass": False, "prediction_matrices_decoded": False, "created_utc": datetime.now(timezone.utc).isoformat()}

def validate_capture(path, digest, run):
    e = pinned(local(ROOT, path), digest)
    require(e.get("schema") == "dynavelo_local_fresh_completion_v1" and e.get("seed") == 2 and e.get("run_id") == run
            and e.get("script_sha256") == q.sha(SELF), "Fresh completion identity differs")
    actual = build_capture(ROOT, e["job_id"], run, e["job_sha256"], e["terminal_sha256"])
    require({k: v for k, v in e.items() if k != "created_utc"} == {k: v for k, v in actual.items() if k != "created_utc"}, "Completion inventory drift")
    return e

def readout(path, digest, run, output_id):
    # Numerical tail and reconstruction code stay identical. Only seed identity
    # and the fresh-run evidence adapter replace seed1 migration provenance.
    ns = {**vars(old), **globals()}
    finish = function_source(old, "adapted_finish")
    reader = once(function_source(old, "readout"), "'seed':1", "'seed':2")
    exec(compile(finish + "\n\n" + reader, "<same_factual_readout_with_fresh_seed2_evidence>", "exec"), ns)
    return ns["readout"](path, digest, run, output_id)

def validate_readout(path, digest, run):
    source = once(function_source(old, "validate_readout"), "a.get('seed')==1", "a.get('seed')==2")
    ns = {**vars(old), **globals()}
    exec(compile(source, "<same_readout_completion_check_fresh_seed2>", "exec"), ns)
    return ns["validate_readout"](path, digest, run)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("inspect", "capture", "readout"), default="inspect")
    p.add_argument("--execute", action="store_true")
    for key in ("run-id", "job-id", "job-sha256", "terminal-sha256", "evidence-sha256", "output-id"):
        p.add_argument("--" + key)
    p.add_argument("--evidence", type=Path)
    a = p.parse_args()
    if a.action == "inspect":
        print(json.dumps({"status": "INSPECT_ONLY_NO_TRAINING_ARTIFACT_OPENED", "scope": "fresh local seed2 capture/readout; original statistics", "scientific_pass": False})); return
    require(a.execute and RUN.fullmatch(a.run_id or ""), "Explicit fresh seed2 execution required")
    check_output_id(a.output_id)
    if a.action == "capture":
        value = build_capture(ROOT, a.job_id, a.run_id, a.job_sha256, a.terminal_sha256)
        out = q.safe(ROOT, "results/reports/local_completion_evidence/" + a.output_id)
        out.mkdir(parents=True, exist_ok=False); path = out / "evidence.json"; q.save(path, value)
        print(json.dumps({"evidence": str(path), "sha256": q.sha(path)}))
    else:
        value = readout(a.evidence, a.evidence_sha256, a.run_id, a.output_id)
        print(json.dumps({"status": value["status"], "run_id": value["run_id"], "seed": value["seed"]}))

if __name__ == "__main__":
    main()
