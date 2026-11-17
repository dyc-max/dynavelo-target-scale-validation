"""Storage and transfer guards for the isolated Colab CUDA runner.

This file is authored on D: and executed only after upload to Linux Colab.
It never mounts Drive or authenticates a user; transport belongs to the caller.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path("/content/DynaVelo_Research")
BASE = PROJECT_ROOT / "data/processed/GSE284047/8489-MV-1"
FROZEN_VERSIONS = {"torch": "2.5.1", "numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.3",
                   "anndata": "0.10.9", "h5py": "3.12.1", "torchmetrics": "1.6.1", "torchdiffeq": "0.2.5"}


def verify_frozen_environment():
    actual = {name: importlib.metadata.version(name) for name in FROZEN_VERSIONS}
    mismatch = {name: version for name, version in actual.items() if version.split("+")[0] != FROZEN_VERSIONS[name]}
    if mismatch:
        raise RuntimeError(f"Cloud core package versions differ from the verified local baseline: {mismatch}")
    return {"status": "PASS_PINNED_CORE_PACKAGE_VERSIONS", "actual": actual,
            "CUDA_note": "CUDA availability and device runtime are verified separately; CPU wheel is not sufficient"}


def absolute_project_path(value):
    if os.name != "posix":
        raise RuntimeError("Colab paths are Linux-only; do not execute this runner on Windows")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"An absolute cloud project path is required: {path}")
    root = PROJECT_ROOT.resolve()
    if str(root) != "/content/DynaVelo_Research":
        raise ValueError("Cloud project root must not redirect outside /content/DynaVelo_Research")
    result = path.resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Cloud artifact must remain within {root}: {result}")
    return result


def configure_cloud_storage():
    root = absolute_project_path(PROJECT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("repo", "data/raw", "data/processed", "checkpoints", "results/figures", "results/reports", "tmp_cache"):
        absolute_project_path(root / relative).mkdir(parents=True, exist_ok=True)
    settings = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "PIP_CACHE_DIR": "pip",
        "HF_HOME": "hf", "HUGGINGFACE_HUB_CACHE": "hf/hub", "HF_DATASETS_CACHE": "hf/datasets",
        "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
        "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "xdg_config", "XDG_DATA_HOME": "xdg_data",
        "JOBLIB_TEMP_FOLDER": "joblib", "PYTHONPYCACHEPREFIX": "pycache",
        "IPYTHONDIR": "ipython", "JUPYTER_CONFIG_DIR": "jupyter",
    }
    for variable, suffix in settings.items():
        path = absolute_project_path(root / "tmp_cache" / suffix)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    os.environ["MPLBACKEND"] = "Agg"


def sha256(path):
    digest = hashlib.sha256()
    with absolute_project_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = absolute_project_path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_h5ad_atomic(adata, path):
    path = absolute_project_path(path)
    temporary = path.with_name(path.stem + ".partial.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(path)


def verify_transfer_manifest(manifest_path, required_paths):
    """Verify uploaded bytes against a manifest frozen on D: before transport."""
    manifest_path = absolute_project_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "dynavelo_colab_transfer_v1":
        raise ValueError("Missing/unsupported transfer-manifest schema")
    if manifest.get("target_project_root") != str(PROJECT_ROOT):
        raise ValueError("Transfer manifest target root differs from the frozen cloud root")
    checked = {}
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Manifest must use safe project-relative POSIX paths")
        path = absolute_project_path(PROJECT_ROOT / relative)
        if path in checked:
            raise ValueError(f"Duplicate manifest path: {path}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Uploaded file differs from D-drive manifest: {path}")
        checked[path] = entry["sha256"]
    missing = {absolute_project_path(path) for path in required_paths} - set(checked)
    if missing:
        raise ValueError(f"Transfer manifest omits required inputs/code: {sorted(map(str, missing))}")
    return {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "n_verified_files": len(checked), "status": "PASS_UPLOADED_BYTES_MATCH_FROZEN_MANIFEST"}
