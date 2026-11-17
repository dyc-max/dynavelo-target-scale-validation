"""Separate FP64 specialization of the source-pinned streaming primitives.

This module does not change a model forward, ODE solver, weight or original
source file. Its two compiled core classes differ only at explicit dtype
attributes and explanatory text. Additional wrappers enforce a distinct result
namespace and FP64/default-dtype operation. This is not original FP32 output.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re

import numpy as np
import torch

import dynavelo_streaming_core as _core

ORIGINAL_CORE_SHA256 = '94a7b832e06c1f0404fd20790e8968f93724d01b732029fd91e22a06a3ae6368'
BASELINE_SCHEMA = 'dynavelo_precision_stream_baseline_v1'
FP64_ARITHMETIC = ('Separate FP64 precision reference: float64 inputs, parameters, forward, '
                   'baseline, subtraction and finite-difference scalar multiplication; '
                   'same original forward/ODE code, no original FP32 equivalence claim')
_TEXT_REPLACEMENTS = {
    'Full frozen inference population, with owned finite float32 inputs.':
        'Full frozen population of owned finite float64 precision-reference inputs.',
    'Expected aligned float32 dense/model matrix: ': 'Expected aligned float64 dense/model matrix: ',
    'Only upstream float32 parameter arithmetic is supported':
        'Separate FP64 precision-reference parameter arithmetic required',
    'Exact upstream one-sided finite-difference order; outputs stream by channel.':
        'FP64 specialization of the upstream one-sided difference order; channels stream.',
    'float32 forward; perturb subtract after float64 cast; GRN float32 subtract and scalar multiply then float64 storage':
        FP64_ARITHMETIC,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


class _DtypesOnly(ast.NodeTransformer):
    def __init__(self):
        self.numpy = self.torch = self.text = 0

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.attr == 'float32':
            if node.value.id == 'np':
                node.attr = 'float64'; self.numpy += 1
            elif node.value.id == 'torch':
                node.attr = 'float64'; self.torch += 1
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in _TEXT_REPLACEMENTS:
            node.value = _TEXT_REPLACEMENTS[node.value]; self.text += 1
        return node


def _specialize():
    path = Path(_core.__file__).resolve()
    if _sha(path) != ORIGINAL_CORE_SHA256:
        raise ValueError('Original streaming primitive source drift')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    original = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in ('Population', 'StreamingSession')}
    if set(original) != {'Population', 'StreamingSession'}:
        raise ValueError('Original primitive class inventory changed')
    # Verify the imported classes came from the exact pinned bodies, not a
    # preinstalled patch at the same filename.
    for name, node in original.items():
        live = ast.parse(inspect.getsource(getattr(_core, name))).body[0]
        if ast.dump(live, include_attributes=False) != ast.dump(node, include_attributes=False):
            raise ValueError('Imported original primitive differs from pinned class')
    audit = {}
    classes = []
    for name in ('Population', 'StreamingSession'):
        transform = _DtypesOnly()
        node = transform.visit(copy.deepcopy(original[name]))
        expected = (2, 0, 2) if name == 'Population' else (2, 6, 3)
        counts = (transform.numpy, transform.torch, transform.text)
        if counts != expected:
            raise ValueError('Unexpected dtype specialization count: ' + name)
        audit[name] = {'np_float32_to_float64': counts[0],
                       'torch_float32_to_float64': counts[1],
                       'explanatory_text_replacements': counts[2]}
        classes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    namespace = dict(vars(_core))
    namespace['__name__'] = __name__
    exec(compile(module, '<pinned_streaming_explicit_fp64_dtype_specialization>', 'exec'), namespace)
    return namespace['Population'], namespace['StreamingSession'], audit


_Population64, _Session64, _COUNTS = _specialize()
SPECIALIZATION_AUDIT = {
    'schema': 'dynavelo_streaming_fp64_specialization_v1',
    'original_core_sha256': ORIGINAL_CORE_SHA256,
    'dtype_and_text_replacements': _COUNTS,
    'core_mathematical_statements_changed': False,
    'model_forward_or_ODE_solver_changed': False,
    'different_precision_from_original_FP32': True,
}
digest_array = _core.digest_array
model_fingerprint = _core.model_fingerprint
batches_without_singleton = _core.batches_without_singleton


def specialization_audit():
    """Return an owned copy suitable for immutable manifests."""
    return copy.deepcopy(SPECIALIZATION_AUDIT)


def _precision_directory(value):
    directory = Path(value).absolute()
    roots = [Path(r'D:\dyc_7\DynaVelo_Research'), Path('/content/DynaVelo_Research')]
    if directory.resolve() != directory:
        raise ValueError('Redirected precision baseline directory')
    if not any(directory.is_relative_to(root / 'results/precision_runs_v1') for root in roots):
        raise ValueError('FP64 baselines require the separate results/precision_runs_v1 namespace')
    return directory


class Population64(_Population64):
    """Owned immutable FP64 input population; constructors require FP64 arrays."""

    @classmethod
    def from_float32(cls, population):
        """Exactly promote a caller-verified original Population; never renormalize."""
        for name in ('x', 'y', 'x_min', 'y_min'):
            value = getattr(population, name)
            if value.dtype != np.float32 or digest_array(value) != population.identity[name + '_sha256']:
                raise ValueError('Original float32 population identity differs')
        result = cls(population.x.astype(np.float64), population.y.astype(np.float64),
                     population.cell_ids, population.genes, population.motif_tfs)
        for name in ('x', 'y', 'x_min', 'y_min'):
            if not np.array_equal(getattr(result, name), getattr(population, name).astype(np.float64)):
                raise ValueError('FP64 population is not an exact float32 value promotion')
        result.identity['precision_origin'] = {
            'rule': 'Exact promotion of verified float32 values; no new scaling/filtering/minima',
            'original_float32_identity': copy.deepcopy(population.identity),
            'original_core_sha256': ORIGINAL_CORE_SHA256,
        }
        return result


def promote_population64(population):
    return Population64.from_float32(population)


class StreamingSession64(_Session64):
    def __init__(self, model, population, cell_indices, batch_size, baseline_directory, *, checkpoint_sha256):
        directory = _precision_directory(baseline_directory)
        super().__init__(model, population, cell_indices, batch_size, directory,
                         checkpoint_sha256=checkpoint_sha256)
        self.identity['schema'] = BASELINE_SCHEMA
        self.identity['precision_specialization'] = specialization_audit()

    def _check_mode(self):
        super()._check_mode()
        if torch.get_default_dtype() != torch.float64:
            raise ValueError('Scoped torch default float64 required for original forward allocations')
        if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
            raise ValueError('Autocast is outside the FP64 precision contract')
        if any(module.training for module in self.model.modules()):
            raise ValueError('Every FP64 model module must be in eval mode')
        if any(value.is_floating_point() and value.dtype != torch.float64
               for value in self.model.buffers()):
            raise ValueError('Floating model buffers must also be float64')


def attach_baseline64(model, population, indices, task, directory, guard=None):
    """Restore only an exact, fully hashed FP64 cache without writing any file.

    The caller remains responsible for verifying the actual source checkpoint,
    full frozen plan and parent task COMMIT before invoking this constructor.
    """
    if guard is not None:
        guard.check()
    directory = _precision_directory(directory)
    if not directory.is_dir():
        raise ValueError('FP64 baseline directory absent')
    checkpoint = task.get('checkpoint_sha256')
    if not isinstance(checkpoint, str) or re.fullmatch('[0-9a-f]{64}', checkpoint) is None:
        raise ValueError('Exact checkpoint SHA required')
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) < 2
            or len(np.unique(indices)) != len(indices) or indices.min() < 0
            or indices.max() >= len(population.cell_ids)):
        raise ValueError('Invalid restored FP64 indices')
    if task.get('n_cells') != len(indices):
        raise ValueError('Task/restored FP64 cell count differs')
    if 'sample_cell_ids' in task and list(task['sample_cell_ids']) != [population.cell_ids[int(i)] for i in indices]:
        raise ValueError('Task/restored FP64 cell IDs differ')
    batch_size = task['chunk_geometry']['cells']
    batches = batches_without_singleton(len(indices), batch_size)
    if 'batches' in task and _canonical(task['batches']) != _canonical(batches):
        raise ValueError('Task/restored FP64 batch geometry differs')
    manifest = json.loads((directory / 'baseline_manifest.json').read_text(encoding='utf-8'))
    fingerprint = model_fingerprint(model)
    expected = {'schema': BASELINE_SCHEMA, 'mode': 'evaluation-fixed',
                'population': population.identity, 'selected_indices': indices.tolist(),
                'batches': batches, 'checkpoint_sha256': checkpoint,
                'model_state_sha256': fingerprint, 'torch': torch.__version__, 'numpy': np.__version__,
                'arithmetic': FP64_ARITHMETIC,
                'minima_scope': 'full supplied Population, never cell subset or batch',
                'precision_specialization': specialization_audit(),
                'scientific_gate': 'NOT_EVALUATED_BY_NUMERICAL_PRIMITIVE'}
    for key, value in expected.items():
        if _canonical(manifest.get(key)) != _canonical(value):
            raise ValueError('FP64 matched baseline differs: ' + key)
    if set(manifest.get('baseline_arrays', {})) != {'vx', 'vy', 'vz', 'time'}:
        raise ValueError('FP64 baseline channel inventory differs')
    obj = object.__new__(StreamingSession64)
    obj.model, obj.population = model, population
    obj.indices = indices.astype(np.int64, copy=True); obj.indices.flags.writeable = False
    obj.batches, obj.directory = batches, directory
    obj.identity, obj.fingerprint, obj.baseline = manifest, fingerprint, {}
    obj._check_mode()
    shapes = {'vx': (len(indices), model.x_dim), 'vy': (len(indices), model.y_dim),
              'vz': (len(indices), model.zx_dim + model.zy_dim), 'time': (len(indices),)}
    try:
        for name, shape in shapes.items():
            if guard is not None:
                guard.check()
            metadata = manifest['baseline_arrays'][name]
            if _canonical(metadata.get('shape')) != _canonical(shape):
                raise ValueError('FP64 baseline manifest shape differs: ' + name)
            value = np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            obj.baseline[name] = value
            if value.shape != shape or value.dtype != np.float64:
                raise ValueError('FP64 baseline shape/dtype differs: ' + name)
            for start in range(0, len(value), 256):
                if not np.isfinite(value[start:start + 256]).all():
                    raise ValueError('Nonfinite FP64 baseline cache: ' + name)
        obj._check_ready()
        if guard is not None:
            guard.check()
    except BaseException:
        for value in obj.baseline.values():
            value._mmap.close()
        raise
    return obj
