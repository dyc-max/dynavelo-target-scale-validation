"""Resumable training bookkeeping and bounded CUDA benchmark; no new loss."""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
import random
import shutil
import time

from colab_runtime import PROJECT_ROOT, absolute_project_path, sha256, write_json


def immutable_config(config):
    args = config["command_arguments"]
    return {"inputs": config["input_files"], "adapter": config["adapter_manifest"],
            "runner_sha256": config["script_sha256"], "training_helper_sha256": config["training_implementation_sha256"],
            "architecture": config["model_defaults"],
            "fit_arguments": {key: args[key] for key in ("seed", "batch_size", "learning_rate", "hidden", "zx_dim", "zy_dim")}}


def preserve_resume_config(run_dir, config, resume_checkpoint, prior_config=None):
    if not resume_checkpoint:
        return
    if prior_config is None:
        raise ValueError("Resume requires the hash-verified prior configuration snapshot")
    old_path = absolute_project_path(prior_config)
    if old_path.parent != run_dir or old_path.name != "resume_config.json":
        raise ValueError("Unexpected verified prior configuration snapshot")
    if not old_path.is_file():
        raise FileNotFoundError("Resume run has no prior resume_config.json snapshot")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if immutable_config(old) != immutable_config(config):
        raise ValueError("Resume cannot change inputs, model source, architecture, seed, batch size or initial learning rate")
    archive = run_dir / ("config_before_resume_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    archive.write_bytes(old_path.read_bytes())
    config["resume_parent_config_sha256"] = sha256(archive)
    config["resume_parent_config"] = str(archive)


def _rng_state(train_loader):
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "train_sampler": train_loader.batch_sampler.generator.get_state()}


def _restore_rng(state, train_loader):
    import numpy as np
    import torch
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), values[2], values[3], values[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    train_loader.batch_sampler.generator.set_state(state["train_sampler"].cpu())


def _save_atomic(path, state):
    import torch
    path = absolute_project_path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def _publish_best(snapshot, canonical):
    snapshot, canonical = absolute_project_path(snapshot), absolute_project_path(canonical)
    temporary = canonical.with_suffix(canonical.suffix + ".partial")
    shutil.copyfile(snapshot, temporary)
    temporary.replace(canonical)


def _manifest_record(path, role):
    path = absolute_project_path(path)
    return {"role": role, "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_checkpoint_restore_manifest(run_dir, config, checkpoint, canonical):
    """Atomic point-in-time recovery index; it is not a remote backup."""
    run_dir = absolute_project_path(run_dir)
    prior_config = run_dir / "resume_config.json"
    # Unlike config.json, this snapshot is not changed by final evaluation/status
    # writes. It is replaced only when a later epoch is committed.
    write_json(prior_config, config)
    required = [
        _manifest_record(run_dir / "last_checkpoint.pth", "checkpoint"),
        _manifest_record(prior_config, "prior_config"),
        _manifest_record(run_dir / checkpoint["best_snapshot_filename"], "best_snapshot"),
    ]
    observed = []
    for path in sorted(set([absolute_project_path(canonical), run_dir / "console.log",
                            run_dir / "training_metrics.tsv", *run_dir.glob("*.log")])):
        if path.is_file():
            observed.append(_manifest_record(path, "observed_at_epoch_commit"))
    destination = run_dir / "checkpoint_restore_manifest.json"
    write_json(destination, {
        "schema": "dynavelo_colab_checkpoint_restore_v1", "run_id": run_dir.name,
        "completed_epoch": checkpoint["completed_epoch"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "cloud_root": str(PROJECT_ROOT),
        "required_files": required, "observed_files": observed,
        "scope": "Point-in-time completed-epoch index. Required filenames can change at the next commit; logs can continue changing. Verify bytes/SHA after download and retry a torn capture.",
        "durability": "No remote backup is implied. VM loss destroys files not already transferred and verified on D:.",
    })
    return destination


def verify_checkpoint_restore_manifest(manifest_path, checkpoint_path):
    """Verify all three recovery inputs before reading checkpoint contents."""
    manifest_path, checkpoint_path = [absolute_project_path(value) for value in (manifest_path, checkpoint_path)]
    run_dir = checkpoint_path.parent
    if checkpoint_path.name != "last_checkpoint.pth" or manifest_path != run_dir / "checkpoint_restore_manifest.json":
        raise ValueError("Resume needs last_checkpoint.pth and its same-run checkpoint_restore_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") != "dynavelo_colab_checkpoint_restore_v1"
            or manifest.get("cloud_root") != str(PROJECT_ROOT) or manifest.get("run_id") != run_dir.name
            or type(manifest.get("completed_epoch")) is not int or manifest["completed_epoch"] < 1):
        raise ValueError("Invalid completed-epoch restore manifest identity/schema")
    expected_names = {"checkpoint": {"last_checkpoint.pth"}, "prior_config": {"resume_config.json"},
                      "best_snapshot": {"best_snapshot_0.pth", "best_snapshot_1.pth"}}
    checked = {}
    for entry in manifest.get("required_files", []):
        role, relative = entry.get("role"), entry.get("relative_path", "")
        if role not in expected_names or role in checked or not isinstance(relative, str):
            raise ValueError("Duplicate or unknown restore input role")
        relative_path = type(run_dir)(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in relative:
            raise ValueError("Unsafe restore manifest relative path")
        path = absolute_project_path(PROJECT_ROOT / relative_path)
        if path.parent != run_dir or path.name not in expected_names[role]:
            raise ValueError("Restore inputs must be the expected files in one run directory")
        if type(entry.get("bytes")) is not int or entry["bytes"] < 1:
            raise ValueError("Restore input needs an exact positive byte count")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry.get("sha256"):
            raise ValueError(f"Restore input bytes/SHA mismatch: {role}")
        checked[role] = {"path": str(path), "sha256": entry["sha256"]}
    if set(checked) != set(expected_names) or checked["checkpoint"]["path"] != str(checkpoint_path):
        raise ValueError("Restore manifest omits checkpoint, prior configuration or best snapshot")
    return {"status": "PASS_RESTORE_INPUT_BYTES_AND_SHA256", "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path), "completed_epoch": manifest["completed_epoch"],
            **{role: entry["path"] for role, entry in checked.items()},
            "best_snapshot_sha256": checked["best_snapshot"]["sha256"],
            "n_verified_required_files": len(checked),
            "observed_logs": "Not required for resume; recorded hashes describe only the original capture time"}


def fit_resumable(model, train_loader, valid_loader, optimizer, args, run_dir, rows, config):
    """Upstream train_step/test_step, scheduler and exact early-stop rules.

Checkpoint and recovery index at each completed epoch. Resume requires a
matching verified package; partial epochs are not accepted. The index is
atomic, but the rolling files are not a multi-file atomic transaction.
"""
    import numpy as np
    import torch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)
    start_epoch, best_loss, counter = 1, None, 0
    best_snapshot = None
    model.logger.start()
    model._resume_elapsed_seconds = 0.0
    session_started = time.monotonic()
    if args.resume_checkpoint:
        verified = verify_checkpoint_restore_manifest(args.resume_manifest, args.resume_checkpoint)
        state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
        if state.get("schema") != "dynavelo_epoch_resume_v1" or state["immutable_config"] != immutable_config(config):
            raise ValueError("Resume checkpoint schema or immutable configuration mismatch")
        if (state["completed_epoch"] != verified["completed_epoch"]
                or state["best_snapshot_filename"] != absolute_project_path(verified["best_snapshot"]).name
                or state["best_checkpoint_sha256"] != verified["best_snapshot_sha256"]):
            raise ValueError("Checkpoint contents differ from the verified recovery manifest")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["completed_epoch"]) + 1
        best_loss, counter = float(state["best_loss"]), int(state["early_stopping_counter"])
        rows.extend(state["history"])
        model.logger.data = {name: [row[name] for row in rows] for name in model.logger.names}
        model._resume_elapsed_seconds = float(state["elapsed_training_seconds"])
        best_snapshot = absolute_project_path(run_dir / state["best_snapshot_filename"])
        if best_snapshot.parent != run_dir or sha256(best_snapshot) != state["best_checkpoint_sha256"]:
            raise ValueError("Committed best-weight snapshot is absent or differs from resume metadata")
        # Repair a canonical best file if interruption happened between the
        # epoch transaction commit and publishing the convenience filename.
        _publish_best(best_snapshot, model.ckpt_path)
        _restore_rng(state["rng_state"], train_loader)
    for epoch in range(start_epoch, args.max_epochs + 1):
        if counter >= 10:
            break
        model.mode = "train"
        loss_train = model.train_step(train_loader, optimizer)
        test_values = model.test_step(valid_loader)
        row = [epoch, loss_train, *test_values]
        if not np.isfinite(np.asarray(row, dtype=np.float64)).all():
            raise FloatingPointError("Nonfinite epoch loss or validation component")
        model.logger.add(row)
        model.logger.save(str(run_dir / (model.model_suffix + ".log")))
        loss_test = float(test_values[0])
        scheduler.step(loss_test)
        # Preserve upstream strict comparisons; ties do not reset/increment.
        if best_loss is None:
            best_loss = loss_test
        elif best_loss - loss_test > 0:
            best_loss, counter = loss_test, 0
        elif best_loss - loss_test < 0:
            counter += 1
        best_changed = loss_test <= best_loss
        if best_changed:
            next_name = "best_snapshot_1.pth" if best_snapshot is not None and best_snapshot.name == "best_snapshot_0.pth" else "best_snapshot_0.pth"
            best_snapshot = run_dir / next_name
            _save_atomic(best_snapshot, {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()})
        checkpoint = {"schema": "dynavelo_epoch_resume_v1", "completed_epoch": epoch,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(), "best_loss": best_loss,
                      "early_stopping_counter": counter, "history": rows,
                      "rng_state": _rng_state(train_loader), "immutable_config": immutable_config(config),
                      "elapsed_training_seconds": model._resume_elapsed_seconds + time.monotonic() - session_started,
                      "best_snapshot_filename": best_snapshot.name,
                      "best_checkpoint_sha256": sha256(best_snapshot)}
        _save_atomic(run_dir / "last_checkpoint.pth", checkpoint)
        if best_changed:
            _publish_best(best_snapshot, model.ckpt_path)
        write_checkpoint_restore_manifest(run_dir, config, checkpoint, model.ckpt_path)
        print(f"Epoch {epoch} committed; early-stopping counter {counter}/10", flush=True)


def benchmark(model, loader, optimizer, args, run_dir):
    import numpy as np
    import torch
    batches = list(itertools.islice(loader, args.benchmark_batches))
    if not batches:
        raise ValueError("No real batches available for the CUDA benchmark")

    class BoundedLoader:
        dataset = range(sum(len(batch[0]) for batch in batches))

        def __iter__(self):
            return iter(batches)

    model.mode = "train"
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    loss = float(model.train_step(BoundedLoader(), optimizer))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if not np.isfinite(loss):
        raise FloatingPointError("Nonfinite benchmark training loss")
    result = {"status": "BENCHMARK_ONLY_NO_RESEARCH_MODEL", "n_batches": len(batches),
              "n_real_cells": sum(len(batch[0]) for batch in batches), "batch_size": args.batch_size,
              "training_seconds": elapsed, "seconds_per_batch": elapsed / len(batches),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "finite_training_loss": loss,
              "rough_training_epoch_seconds": elapsed / len(batches) * len(loader),
              "timing_limit": "Includes first-batch overhead; estimate excludes validation, posterior inference and transfer",
              "model_disposition": "Discard benchmark model; never resume it for research", "external_validation": "NOT_PERFORMED"}
    write_json(run_dir / "benchmark.json", result)
    print(json.dumps(result, indent=2), flush=True)


def make_download_manifest(run_dir, prediction_dir, run_id):
    """Called at normal interpreter shutdown, after the console log is closed."""
    report = PROJECT_ROOT / "results/reports" / (run_id + "_internal_qc.json")
    paths = [path for root in (run_dir, prediction_dir) for path in root.rglob("*") if path.is_file() and ".partial" not in path.name]
    if report.is_file():
        paths.append(report)
    environment_freeze = PROJECT_ROOT / "results/reports/colab_environment_freeze.txt"
    if environment_freeze.is_file():
        paths.append(environment_freeze)
    entries = []
    for path in sorted(set(paths)):
        path = absolute_project_path(path)
        entries.append({"relative_path": path.relative_to(PROJECT_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    destination = PROJECT_ROOT / "results/reports" / (run_id + "_download_manifest.json")
    write_json(destination, {"schema": "dynavelo_colab_download_v1", "run_id": run_id,
                            "cloud_root": str(PROJECT_ROOT), "local_root": r"D:\dyc_7\DynaVelo_Research",
                            "files": entries, "total_bytes": sum(row["bytes"] for row in entries)})
    print(f"DOWNLOAD_MANIFEST={destination}", flush=True)
