"""Bounded DynaVelo input-sensitivity primitives, not a screening CLI.

No model/loss/solver modifications. Callers must separately enforce scientific
and preregistration gates. Every baseline and intervention uses one frozen
cell order and batch geometry. No N x output x regulator cube is allocated.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

CHANNELS = {'vx': 2, 'vy': 3, 'vz': 6, 'time': 7}

def digest_array(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
    if value.ndim == 0:
        digest.update(value.tobytes())
        return digest.hexdigest()
    for start in range(0, len(value), 256):
        digest.update(np.ascontiguousarray(value[start:start + 256]).tobytes())
    return digest.hexdigest()

def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(digest_array(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()

def unique_names(values, label):
    values = tuple(str(value) for value in values)
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError('Empty/duplicate ' + label)
    return values

def batches_without_singleton(n_cells, batch_size):
    if type(n_cells) is not int or type(batch_size) is not int or n_cells < 2 or batch_size < 2:
        raise ValueError('At least two cells and batch size >=2 required')
    edges = list(range(0, n_cells, batch_size)) + [n_cells]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        edges.pop(-2)
    return tuple((a, b) for a, b in zip(edges[:-1], edges[1:]))

class Population:
    """Full frozen inference population, with owned finite float32 inputs."""
    def __init__(self, x, y, cell_ids, genes, motif_tfs):
        self.cell_ids = unique_names(cell_ids, 'cell IDs')
        self.genes = unique_names(genes, 'RNA feature IDs')
        self.motif_tfs = unique_names(motif_tfs, 'motif TF mappings')
        for name, value, width in [('x', x, len(self.genes)), ('y', y, len(self.motif_tfs))]:
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (len(self.cell_ids), width):
                raise ValueError('Expected aligned float32 dense/model matrix: ' + name)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite model input: ' + name)
            owned = np.array(value, dtype=np.float32, order='C', copy=True)
            owned.flags.writeable = False
            setattr(self, name, owned)
        self.x_min = self.x.min(axis=0); self.y_min = self.y.min(axis=0)
        self.x_min.flags.writeable = False; self.y_min.flags.writeable = False
        self.gene_index = {value: i for i, value in enumerate(self.genes)}
        self.motif_index = {value: i for i, value in enumerate(self.motif_tfs)}
        self.identity = {'x_sha256': digest_array(self.x), 'y_sha256': digest_array(self.y),
                         'x_min_sha256': digest_array(self.x_min), 'y_min_sha256': digest_array(self.y_min),
                         'cell_ids': self.cell_ids, 'genes': self.genes, 'motif_tfs': self.motif_tfs}

class StreamingSession:
    def __init__(self, model, population, cell_indices, batch_size, baseline_directory, *, checkpoint_sha256):
        import re
        self.model, self.population = model, population
        self._check_mode()
        if not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256):
            raise ValueError('Exact checkpoint SHA required')
        indices = np.asarray(cell_indices)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != len(indices)
                or len(indices) < 2 or indices.min() < 0 or indices.max() >= len(population.cell_ids)):
            raise ValueError('Invalid/duplicate selected cell positions')
        self.indices = indices.astype(np.int64, copy=True); self.indices.flags.writeable = False
        self.batches = batches_without_singleton(len(indices), batch_size)
        self.directory = Path(baseline_directory).absolute()
        roots = [Path(r'D:\dyc_7\DynaVelo_Research'), Path('/content/DynaVelo_Research')]
        if (self.directory.resolve() != self.directory or not any(self.directory.is_relative_to(root) for root in roots)):
            raise ValueError('Baseline directory must remain inside the project')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fingerprint = model_fingerprint(model)
        self.identity = {'schema': 'dynavelo_stream_baseline_v1', 'mode': 'evaluation-fixed',
                         'population': population.identity, 'selected_indices': self.indices.tolist(),
                         'batches': self.batches, 'checkpoint_sha256': checkpoint_sha256,
                         'model_state_sha256': self.fingerprint, 'torch': torch.__version__, 'numpy': np.__version__,
                         'arithmetic': 'float32 forward; perturb subtract after float64 cast; GRN float32 subtract and scalar multiply then float64 storage',
                         'minima_scope': 'full supplied Population, never cell subset or batch',
                         'scientific_gate': 'NOT_EVALUATED_BY_NUMERICAL_PRIMITIVE'}
        self.baseline = {}

    def _check_mode(self):
        if self.model.training or self.model.mode != 'evaluation-fixed':
            raise ValueError('Explicit model.eval() and evaluation-fixed required')
        if any(value.dtype != torch.float32 for value in self.model.parameters()):
            raise ValueError('Only upstream float32 parameter arithmetic is supported')

    def _check_ready(self):
        self._check_mode()
        if not self.baseline or model_fingerprint(self.model) != self.fingerprint:
            raise ValueError('Baseline absent or model changed since baseline')
        if any(digest_array(value) != self.identity['baseline_arrays'][name]['sha256'] for name, value in self.baseline.items()):
            raise ValueError('Baseline bytes changed after matched capture')
        for name in ('x', 'y', 'x_min', 'y_min'):
            if digest_array(getattr(self.population, name)) != self.population.identity[name + '_sha256']:
                raise ValueError('Frozen population or global minimum changed')

    def full_population_time_ranks(self):
        self._check_ready()
        if not np.array_equal(self.indices, np.arange(len(self.population.cell_ids))):
            raise ValueError('Global time ranks require the entire population in its frozen order')
        return global_time_ranks(self.baseline['time'])

    def inputs(self, start, stop):
        selected = self.indices[start:stop]
        # Advanced indexing owns new arrays; interventions never mutate Population.
        return (torch.tensor(self.population.x[selected], dtype=torch.float32, device=self.model.device),
                torch.tensor(self.population.y[selected], dtype=torch.float32, device=self.model.device))

    def forward(self, x, y):
        with torch.no_grad():
            output = self.model(x, y)
        expected = {'vx': (len(x), self.model.x_dim), 'vy': (len(x), self.model.y_dim),
                    'vz': (len(x), self.model.zx_dim + self.model.zy_dim), 'time': (len(x),)}
        result = {}
        for name, position in CHANNELS.items():
            value = output[position].detach().cpu().numpy()
            if value.shape != expected[name] or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Unexpected/nonfinite forward channel: ' + name)
            result[name] = value
        return result

    def build_baseline(self):
        self._check_mode()
        if self.baseline:
            raise ValueError('Baseline already built')
        n = len(self.indices)
        shapes = {'vx': (n, self.model.x_dim), 'vy': (n, self.model.y_dim),
                  'vz': (n, self.model.zx_dim + self.model.zy_dim), 'time': (n,)}
        arrays = {name: np.lib.format.open_memmap(self.directory / (name + '.npy'), mode='w+', dtype=np.float32, shape=shape)
                  for name, shape in shapes.items()}
        for start, stop in self.batches:
            values = self.forward(*self.inputs(start, stop))
            for name, value in values.items():
                arrays[name][start:stop] = value
        for value in arrays.values():
            value.flush()
        if model_fingerprint(self.model) != self.fingerprint:
            raise ValueError('Model changed during baseline')
        self.identity['baseline_arrays'] = {name: {'shape': value.shape, 'sha256': digest_array(value)} for name, value in arrays.items()}
        del arrays
        self.baseline = {name: np.load(self.directory / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in shapes}
        (self.directory / 'baseline_manifest.json').write_text(json.dumps(self.identity, indent=2) + '\n', encoding='utf-8')
        return self.identity

    def perturbations(self, regulators, *, mode='rna-only'):
        """Yield one regulator/cell-block. Official joint falls back to RNA-only when unmapped."""
        self._check_ready()
        regulators = unique_names(regulators, 'regulators')
        if mode not in {'rna-only', 'official-joint', 'no-op'}:
            raise ValueError('Unsupported intervention mode')
        if any(name not in self.population.gene_index for name in regulators):
            raise ValueError('RNA regulator absent')
        for name in regulators:
            g = self.population.gene_index[name]
            motif = self.population.motif_index.get(name) if mode == 'official-joint' else None
            for start, stop in self.batches:
                x, y = self.inputs(start, stop)
                if mode != 'no-op':
                    x[:, g] = torch.tensor(self.population.x_min[g], dtype=torch.float32, device=self.model.device)
                    if motif is not None:
                        y[:, motif] = torch.tensor(self.population.y_min[motif], dtype=torch.float32, device=self.model.device)
                values = self.forward(x, y)
                # Official perturbation fills float64 buffers before subtracting.
                deltas = {key: value.astype(np.float64) - np.asarray(self.baseline[key][start:stop], dtype=np.float64)
                          for key, value in values.items()}
                yield {'regulator': name, 'mode': mode, 'motif_changed': motif is not None,
                       'start': start, 'stop': stop, 'population_indices': self.indices[start:stop], 'delta': deltas}

    def jacobians(self, regulators, *, input_space='rna', rna_targets=None, target_chunk=256, epsilon=1e-4):
        """Exact upstream one-sided finite-difference order; outputs stream by channel."""
        self._check_ready()
        regulators = unique_names(regulators, 'regulators')
        if input_space not in {'rna', 'motif'} or target_chunk < 1 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Invalid finite difference settings')
        mapping = self.population.gene_index if input_space == 'rna' else self.population.motif_index
        if any(name not in mapping for name in regulators):
            raise ValueError('Regulator missing from requested input space')
        targets = self.population.genes if rna_targets is None else unique_names(rna_targets, 'RNA targets')
        if any(name not in self.population.gene_index for name in targets):
            raise ValueError('RNA target absent')
        target_indices = np.array([self.population.gene_index[name] for name in targets], dtype=np.int64)
        for name in regulators:
            for start, stop in self.batches:
                x, y = self.inputs(start, stop)
                e = torch.zeros(x.shape[1] if input_space == 'rna' else y.shape[1], dtype=torch.float32, device=self.model.device)
                e[mapping[name]] = 1.
                values = self.forward(x + epsilon * e, y) if input_space == 'rna' else self.forward(x, y + epsilon * e)
                for channel, output_names, indices in [('vx', targets, target_indices),
                        ('vy', self.population.motif_tfs, np.arange(len(self.population.motif_tfs)))]:
                    for left in range(0, len(indices), target_chunk):
                        right = min(left + target_chunk, len(indices)); selected = indices[left:right]
                        # Upstream performs this NumPy expression in float32, then assigns to float64 arrays.
                        value = (1 / epsilon) * (values[channel][:, selected] - self.baseline[channel][start:stop, selected])
                        yield {'regulator': name, 'input_space': input_space, 'channel': channel,
                               'start': start, 'stop': stop, 'target_start': left, 'target_stop': right,
                               'targets': output_names[left:right], 'population_indices': self.indices[start:stop],
                               'values': value.astype(np.float64)}

def global_time_ranks(full_population_times):
    """Global stable ordinal ranks, called once on a full-population baseline."""
    times = np.asarray(full_population_times)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('Full finite time vector required')
    ranks = np.empty(len(times), dtype=np.float64)
    ranks[np.argsort(times, kind='stable')] = np.arange(len(times)) / (len(times) - 1)
    return ranks

class EdgeAccumulator:
    """One regulator/channel at a time; finite counts and float64 sufficient sums."""
    def __init__(self, n_groups, n_targets):
        self.sum = np.zeros((n_groups, n_targets), dtype=np.float64)
        self.abs_sum = np.zeros_like(self.sum)
        self.count = np.zeros_like(self.sum, dtype=np.int64)
        self.positive = np.zeros_like(self.count); self.negative = np.zeros_like(self.count)
        self.zero = np.zeros_like(self.count); self.nonfinite = np.zeros_like(self.count)

    def update(self, values, group_ids, target_start):
        values, group_ids = np.asarray(values, dtype=np.float64), np.asarray(group_ids)
        if values.ndim != 2 or group_ids.shape != (len(values),) or group_ids.dtype.kind not in 'iu':
            raise ValueError('Accumulator shape/group type mismatch')
        stop = target_start + values.shape[1]
        if target_start < 0 or stop > self.sum.shape[1] or group_ids.min() < 0 or group_ids.max() >= len(self.sum):
            raise ValueError('Accumulator index out of bounds')
        for group in np.unique(group_ids):
            block = values[group_ids == group]; finite = np.isfinite(block)
            valid = np.where(finite, block, 0)
            self.sum[group, target_start:stop] += valid.sum(axis=0, dtype=np.float64)
            self.abs_sum[group, target_start:stop] += np.abs(valid).sum(axis=0, dtype=np.float64)
            self.count[group, target_start:stop] += finite.sum(axis=0)
            self.nonfinite[group, target_start:stop] += (~finite).sum(axis=0)
            self.positive[group, target_start:stop] += ((block > 0) & finite).sum(axis=0)
            self.negative[group, target_start:stop] += ((block < 0) & finite).sum(axis=0)
            self.zero[group, target_start:stop] += ((block == 0) & finite).sum(axis=0)
