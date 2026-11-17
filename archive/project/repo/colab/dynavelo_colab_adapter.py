"""Linux/CUDA compatibility wrapper; upstream model/loss/Jacobians unchanged."""
from __future__ import annotations

import inspect
import re
import sys
import types
import hashlib

from colab_runtime import PROJECT_ROOT, absolute_project_path, sha256


def load_dynavelo(project_root=PROJECT_ROOT, analytic_velocity=False, require_analytic_validation=True):
    import torch

    if analytic_velocity:
        raise ValueError("Colab research runner uses original Jacobians; analytic optimization is disabled")
    root = absolute_project_path(project_root)
    if root != PROJECT_ROOT:
        raise ValueError("Unexpected cloud project root")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime before training. No silent CPU fallback.")
    models_path, utils_path = [root / "repo/DynaVelo/dynavelo" / name for name in ("models.py", "utils.py")]
    manifest = {
        "project_root": str(root), "upstream_models_sha256": sha256(models_path),
        "upstream_utils_sha256": sha256(utils_path), "adapter_sha256": sha256(__file__),
        "analytic_velocity": False, "device": "cuda:0", "patches": [],
        "upstream_model_loss_solver_and_jacobian_unchanged": True,
    }
    utils_name = "_dynavelo_colab_compat_utils"
    utils_module = types.ModuleType(utils_name)
    utils_module.__file__ = str(utils_path)
    exec(compile(utils_path.read_text(encoding="utf-8"), str(utils_path), "exec"), utils_module.__dict__)

    class CompatibleLRScheduler:
        def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
            options = dict(mode="min", patience=patience, factor=factor, min_lr=min_lr)
            if "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
                options["verbose"] = True
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **options)

        def __call__(self, val_loss):
            self.lr_scheduler.step(val_loss)

    utils_module.LRScheduler = CompatibleLRScheduler
    sys.modules[utils_name] = utils_module
    manifest["patches"].append("scheduler: omit removed verbose keyword when unsupported")
    source = models_path.read_text(encoding="utf-8")

    def replace_once(old, new, label):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Upstream drift: expected exactly one occurrence for {label}")
        source = source.replace(old, new, 1)
        manifest["patches"].append(label)

    replace_once("from utils import EarlyStopping, LRScheduler", f"from {utils_name} import EarlyStopping, LRScheduler", "scoped stopping/scheduler import")
    replace_once("from utils import Logger", f"from {utils_name} import Logger", "scoped logger import")
    replace_once("self.rna_obs['celltype'][idx]", "self.rna_obs['celltype'].iloc[idx]", "explicit pandas positional indexing")
    replace_once("self.train_dir = f'../checkpoints/{self.dataset_name}/{self.sample_name}/'", "self.train_dir = os.path.join(_PROJECT_ROOT, 'checkpoints', self.dataset_name, self.sample_name) + os.sep", "absolute cloud checkpoint path")
    replace_once('self.logger.save(f"../log/{self.model_suffix}.log")', "self.logger.save(os.path.join(self.train_dir, self.model_suffix + '.log'))", "absolute cloud log path")
    replace_once("checkpoint = torch.load(self.ckpt_path)", "checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=True)", "device-aware own-checkpoint loading")
    manifest["adapted_source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    models_name = "_dynavelo_colab_compat_models"
    models_module = types.ModuleType(models_name)
    models_module.__file__ = str(models_path)
    models_module._PROJECT_ROOT = str(root)
    sys.modules[models_name] = models_module
    exec(compile(source, str(models_path), "exec"), models_module.__dict__)

    class DynaVeloCUDA(models_module.DynaVelo):
        def __init__(self, *args, **kwargs):
            if kwargs.get("device", "cuda:0") != "cuda:0":
                raise ValueError("CUDA runner requires explicit cuda:0")
            for name in ("dataset_name", "sample_name"):
                if name in kwargs and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(kwargs[name])):
                    raise ValueError(f"Unsafe output directory component: {name}")
            kwargs["device"] = "cuda:0"
            super().__init__(*args, **kwargs)
            self.to(torch.device("cuda:0"))

    return models_module.MultiomeDataset, DynaVeloCUDA, manifest
