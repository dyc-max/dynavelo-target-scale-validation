"""Auditable runtime compatibility patches; the upstream checkout stays untouched.

Compatibility changes do not alter the model, loss, solver or inference rule.
An optional validated analytic decoder derivative replaces Jacobian allocation
only for the exact upstream Linear+ReLU / Linear decoder architecture.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import types


PROJECT_ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
ANALYTIC_FORMULA_VERSION = "linear_relu_v1"


def project_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    root = project_root.resolve()
    candidate = Path(path).resolve()
    if root.drive.upper() != "D:" or not candidate.is_relative_to(root):
        raise ValueError(f"Path must remain inside the D-drive project: {candidate}")
    return candidate


def _guard_decoder_architecture(model):
    import torch.nn as nn
    x, y = model.decoder_net_x, model.decoder_net_y
    if not (type(x) is nn.Sequential and len(x) == 2 and type(x[0]) is nn.Linear and type(x[1]) is nn.ReLU
            and type(y) is nn.Sequential and len(y) == 1 and type(y[0]) is nn.Linear):
        raise ValueError("Analytic velocity requires exactly Sequential(Linear, ReLU) and Sequential(Linear)")
    if x[1].inplace or x[0].bias is None or y[0].bias is None:
        raise ValueError("Analytic velocity requires the upstream non-inplace ReLU and biased linear layers")
    if (x[0].in_features, x[0].out_features, y[0].in_features, y[0].out_features) != (model.zx_dim, model.x_dim, model.zy_dim, model.y_dim):
        raise ValueError("Analytic decoder dimensions disagree with model dimensions")
    for module in (x, x[0], x[1], y, y[0]):
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError("Decoder hooks are not supported by the analytic equivalence contract")


def _analytic_decoder_velocity(model, zx, zy, vzx, vzy):
    import torch.nn.functional as F
    _guard_decoder_architecture(model)
    if zx.ndim != 2 or zy.ndim != 2 or zx.shape != vzx.shape or zy.shape != vzy.shape or zx.shape[0] != zy.shape[0]:
        raise ValueError("Analytic velocity expects matched 2D latent states/velocities")
    linear_x, linear_y = model.decoder_net_x[0], model.decoder_net_y[0]
    preactivation = F.linear(zx, linear_x.weight, linear_x.bias)
    vx = F.linear(vzx, linear_x.weight, bias=None) * (preactivation > 0)
    vy = F.linear(vzy, linear_y.weight, bias=None)
    # Preserve upstream squeeze semantics, including its singleton dimensions.
    return vx.squeeze(), vy.squeeze()


def load_dynavelo(project_root: Path = PROJECT_ROOT, analytic_velocity=False, require_analytic_validation=True):
    """Return (MultiomeDataset, CPU-default DynaVelo, patch manifest)."""
    import torch

    root = project_path(project_root, project_root)
    upstream = root / "repo" / "DynaVelo" / "dynavelo"
    models_path = upstream / "models.py"
    utils_path = upstream / "utils.py"
    utils_name = "_dynavelo_project_compat_utils"
    models_name = "_dynavelo_project_compat_models_analytic" if analytic_velocity else "_dynavelo_project_compat_models_jacobian"
    manifest = {
        "project_root": str(root),
        "upstream_models_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
        "upstream_utils_sha256": hashlib.sha256(utils_path.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analytic_velocity": bool(analytic_velocity),
        "analytic_formula_version": ANALYTIC_FORMULA_VERSION,
        "patches": [],
    }

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

    def replace_once(old: str, new: str, label: str) -> None:
        nonlocal source
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"Upstream drift for {label}: expected one match, found {count}")
        source = source.replace(old, new, 1)
        manifest["patches"].append(label)

    replace_once("from utils import EarlyStopping, LRScheduler", f"from {utils_name} import EarlyStopping, LRScheduler", "scoped utils import (scheduler and stopping)")
    replace_once("from utils import Logger", f"from {utils_name} import Logger", "scoped utils import (logger)")
    replace_once("device='cuda:0'", "device='cpu'", "default device is CPU")
    replace_once("self.rna_obs['celltype'][idx]", "self.rna_obs['celltype'].iloc[idx]", "explicit pandas positional indexing")
    replace_once(
        "self.train_dir = f'../checkpoints/{self.dataset_name}/{self.sample_name}/'",
        "self.train_dir = os.path.join(_PROJECT_ROOT, 'checkpoints', self.dataset_name, self.sample_name) + os.sep",
        "absolute D-drive checkpoint directory",
    )
    replace_once(
        'self.logger.save(f"../log/{self.model_suffix}.log")',
        "self.logger.save(os.path.join(self.train_dir, self.model_suffix + '.log'))",
        "absolute D-drive training log directory",
    )
    replace_once("checkpoint = torch.load(self.ckpt_path)", "checkpoint = torch.load(self.ckpt_path, map_location=self.device, weights_only=True)", "CPU-safe own-checkpoint loading with weights_only")
    if analytic_velocity:
        block = (
            r"(?m)^(?P<indent>[ \t]*)Jx = vmap\(jacrev\(self\.decoder_net_x\)\)\(zx\)\n"
            r"(?P=indent)Jy = vmap\(jacrev\(self\.decoder_net_y\)\)\(zy\)\n[ \t]*\n"
            r"(?P=indent)vx_pred = torch\.bmm\(Jx, vzx\.unsqueeze\(2\)\)\.squeeze\(\)\n"
            r"(?P=indent)vy_pred = torch\.bmm\(Jy, vzy\.unsqueeze\(2\)\)\.squeeze\(\)"
        )
        source, count = re.subn(block, lambda match: match.group("indent") + "vx_pred, vy_pred = _analytic_decoder_velocity(self, zx, zy, vzx, vzy)", source)
        if count != 4:
            raise RuntimeError(f"Upstream decoder drift: expected 4 Jacobian blocks, found {count}")
        manifest["patches"].append("four exact Linear+ReLU / Linear Jacobian-vector products replaced by guarded analytic formula")
    manifest["adapted_source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if analytic_velocity and require_analytic_validation:
        representative_path = root / "results/reports/batch128_equivalence_resource.json"
        if representative_path.is_file():
            representative = json.loads(representative_path.read_text(encoding="utf-8"))
            if representative.get("status") != "PASS_BATCH128_EQUIVALENCE_RESOURCE_CHECK":
                raise RuntimeError(
                    "Analytic velocity is on HOLD: the representative batch128/full-feature "
                    "equivalence gate did not pass. Use analytic_velocity=False; do not "
                    "generalize the earlier small-batch PASS or relax its tolerances."
                )
        validation_path = root / "results/reports/analytic_velocity_equivalence.json"
        if not validation_path.is_file():
            raise RuntimeError("Run the real-data analytic equivalence test before enabling this optimization")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not (validation.get("status") == "PASS_NUMERICAL_EQUIVALENCE"
                and validation.get("validated_adapter_sha256") == manifest["adapter_sha256"]
                and validation.get("validated_analytic_source_sha256") == manifest["adapted_source_sha256"]):
            raise RuntimeError("Analytic equivalence PASS is absent or does not match this adapter/source version")
        manifest["analytic_validation_report"] = str(validation_path)
    elif analytic_velocity:
        manifest["analytic_validation_bypass"] = "explicit diagnostic equivalence-test mode; not authorized for scientific use"
    models_module = types.ModuleType(models_name)
    models_module.__file__ = str(models_path)
    models_module._PROJECT_ROOT = str(root)
    models_module._analytic_decoder_velocity = _analytic_decoder_velocity
    sys.modules[models_name] = models_module
    exec(compile(source, str(models_path), "exec"), models_module.__dict__)

    class DynaVeloCPU(models_module.DynaVelo):
        def __init__(self, *args, **kwargs):
            if str(kwargs.get("device", "cpu")) != "cpu":
                raise ValueError("This adapter is the CPU smoke-test route")
            for field in ("dataset_name", "sample_name"):
                if field in kwargs and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(kwargs[field])):
                    raise ValueError(f"Unsafe directory component: {field}")
            kwargs["device"] = "cpu"
            super().__init__(*args, **kwargs)
            if analytic_velocity:
                _guard_decoder_architecture(self)

    return models_module.MultiomeDataset, DynaVeloCPU, manifest
