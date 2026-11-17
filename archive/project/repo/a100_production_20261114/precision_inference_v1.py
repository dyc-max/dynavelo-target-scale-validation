"""Opt-in FP64 evaluation-fixed candidate; qualification is external to this file.

Compile the pinned original forward with only structured decoder Jacobians and
indexed row copies. Preserve original ODE calls/tolerances, bmm and batching.
Never converts model weights, changes the global dtype, or patches source files.
"""
from __future__ import annotations
import ast
import contextlib
import copy
import hashlib
from pathlib import Path
import types

UPSTREAM_SHA = 'ff9ad39be46382a731d03b8899e4a0ccb7e82c93dfecb6ff1fd572f251e118ca'
_HELPER = '_precision64_structured_jacobians'
_ROWS = '_precision64_rows'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard64(model):
    import torch
    from torch import nn
    if model.training or model.mode != 'evaluation-fixed':
        raise ValueError('FP64 candidate requires eval and evaluation-fixed only')
    if any(module.training for module in model.modules()):
        raise ValueError('All modules must be in eval mode')
    if torch.get_default_dtype() != torch.float64:
        raise ValueError('Caller must explicitly set default dtype to float64')
    parameters = tuple(model.parameters())
    if not parameters or any(value.dtype != torch.float64 for value in parameters):
        raise ValueError('Only FP64 model parameters are supported')
    if any(value.is_floating_point() and value.dtype != torch.float64 for value in model.buffers()):
        raise ValueError('All floating-point buffers must be FP64')
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        raise ValueError('Autocast is outside the FP64 candidate contract')
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError('Model hooks are outside the candidate contract')
    x, y = model.decoder_net_x, model.decoder_net_y
    if not (type(x) is nn.Sequential and len(x) == 2 and type(x[0]) is nn.Linear
            and type(x[1]) is nn.ReLU and not x[1].inplace
            and type(y) is nn.Sequential and len(y) == 1 and type(y[0]) is nn.Linear):
        raise ValueError('Original Linear/ReLU and Linear decoders required')
    if (x[0].in_features, x[0].out_features, y[0].in_features, y[0].out_features) != (
            model.zx_dim, model.x_dim, model.zy_dim, model.y_dim):
        raise ValueError('Decoder feature axes differ')


def structured_jacobians64(model, x_pred, zy):
    """Original-size contiguous double Jacobians; original downstream bmm stays."""
    import torch
    wx, wy = model.decoder_net_x[0].weight, model.decoder_net_y[0].weight
    if any(value.dtype != torch.float64 for value in (wx, wy, x_pred, zy)):
        raise ValueError('Structured Jacobian inputs and parameters must be FP64')
    jx = torch.where(x_pred.unsqueeze(-1) > 0, wx.unsqueeze(0), 0.0).contiguous()
    jy = wy.unsqueeze(0).expand(zy.shape[0], -1, -1).contiguous()
    return jx, jy


class _Transform64(ast.NodeTransformer):
    def __init__(self):
        self.copies = self.rna_jacobians = self.motif_jacobians = 0

    def visit_For(self, node):
        if ast.unparse(node.target) == '(idx, cell)' and ast.unparse(node.iter) == 'enumerate(indices)':
            expected = ast.parse('for idx, cell in enumerate(indices):\n z[cell,:] = z_all_times[idx+1,cell,:]\n vz[cell,:] = vz_all_times[idx+1,cell,:]').body[0]
            if ast.dump(node) != ast.dump(expected):
                raise ValueError('Pinned copy-loop changed')
            self.copies += 1
            return ast.parse(f'{_ROWS} = torch.arange(1, len(indices)+1, device=indices.device)\nz[indices, :] = z_all_times[{_ROWS}, indices, :]\nvz[indices, :] = vz_all_times[{_ROWS}, indices, :]').body
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            return node
        name = ast.unparse(node.targets[0])
        if name == 'Jx':
            if ast.unparse(node.value) != 'vmap(jacrev(self.decoder_net_x))(zx)':
                raise ValueError('Pinned RNA Jacobian changed')
            self.rna_jacobians += 1
            return ast.parse(f'Jx, Jy = {_HELPER}(self, x_pred, zy)').body[0]
        if name == 'Jy':
            if ast.unparse(node.value) != 'vmap(jacrev(self.decoder_net_y))(zy)':
                raise ValueError('Pinned motif Jacobian changed')
            self.motif_jacobians += 1
            return None
        return node


def _pinned_forward(upstream_path):
    path = Path(upstream_path)
    if sha(path) != UPSTREAM_SHA:
        raise ValueError('Pinned upstream source hash differs')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DynaVelo')
    original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    names = {node.id for node in ast.walk(original) if isinstance(node, ast.Name)}
    if names.intersection({_HELPER, _ROWS}):
        raise ValueError('Reserved precision helper name occurs in original forward')
    # Preserve module/class compiler context for installed-code identity checks;
    # compilation itself executes no import, class body, constructor or model.
    module_code = compile(tree, '<precision64_original_identity>', 'exec')
    class_code = next(c for c in module_code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'DynaVelo')
    expected = next(c for c in class_code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'forward')
    return original, expected


def _code_identity(code):
    fields = ('co_code', 'co_names', 'co_consts', 'co_argcount', 'co_posonlyargcount',
              'co_kwonlyargcount', 'co_flags', 'co_varnames', 'co_freevars', 'co_cellvars')
    return tuple(getattr(code, key) for key in fields)


def make_forward64(model, upstream_path):
    """Return a bound FP64-only candidate. Caller must qualify all9 outputs/RNG.

    Requires an unmodified installed original forward and already promoted model.
    It does not install itself, convert weights, alter ODE tolerance or qualify a
    backend. The caller retains original forward for same-A100 bitwise checks.
    """
    import torch
    guard64(model)
    original, expected = _pinned_forward(upstream_path)
    if not isinstance(model.forward, types.MethodType):
        raise ValueError('Original bound forward required')
    function = model.forward.__func__
    if _code_identity(function.__code__) != _code_identity(expected):
        raise ValueError('Installed forward differs from pinned original; adapter stacking forbidden')
    changed = copy.deepcopy(original)
    transform = _Transform64()
    changed = transform.visit(changed)
    if (transform.copies, transform.rna_jacobians, transform.motif_jacobians) != (3, 3, 3):
        raise ValueError('Unexpected pinned forward transformation counts')
    # Exactly preserve all odeint expressions, including atol/rtol/method.
    calls = lambda node: [ast.dump(n, include_attributes=False) for n in ast.walk(node)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'odeint']
    if calls(original) != calls(changed):
        raise ValueError('ODE calls or tolerances changed')
    module = ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[]))
    env = dict(function.__globals__)
    env[_HELPER] = structured_jacobians64
    exec(compile(module, '<precision64_structured_candidate>', 'exec'), env)
    compiled = env['forward']

    def candidate(self, x, y):
        guard64(self)
        if torch.is_grad_enabled():
            raise ValueError('FP64 candidate requires torch.no_grad()')
        if x.dtype != torch.float64 or y.dtype != torch.float64:
            raise ValueError('Both input tensors must be FP64')
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError('Original non-singleton aligned 2D batches required')
        if x.shape[1] != self.x_dim or y.shape[1] != self.y_dim:
            raise ValueError('Input feature axes differ from model')
        if x.device != next(self.parameters()).device or y.device != x.device:
            raise ValueError('Inputs and model must share the same device')
        return compiled(self, x, y)

    candidate.manifest = {'schema': 'dynavelo_fp64_structured_inference_candidate_v1',
        'upstream_sha256': UPSTREAM_SHA, 'adapter_sha256': sha(__file__),
        'generated_forward_sha256': hashlib.sha256(ast.unparse(module).encode()).hexdigest(),
        'required_mode': 'evaluation-fixed', 'required_parameter_input_default_dtype': 'float64',
        'structured_original_size_contiguous_jacobians': True, 'original_bmm_unchanged': True,
        'copy_loops_replaced': transform.copies, 'ODE_calls_tolerances_method_unchanged': True,
        'model_conversion_performed': False, 'global_dtype_changed_by_adapter': False,
        'training_supported': False, 'autocast_supported': False,
        'qualification_required': 'Same owned A100 originalFP64 vs candidate all9outputs bitwise and CPU/CUDA/Python/NumPy RNG exact on representative frozen baselines/perturbations and ODEbranches',
        'gpu_qualified': False, 'production_enabled': False, 'scientific_pass': False}
    return types.MethodType(candidate, model)


@contextlib.contextmanager
def installed64(model, forward):
    """Temporary instance-local installation; original restored even on failure."""
    if not isinstance(forward, types.MethodType) or forward.__self__ is not model:
        raise ValueError('Candidate must be bound to this model')
    if getattr(forward, 'manifest', {}).get('schema') != 'dynavelo_fp64_structured_inference_candidate_v1':
        raise ValueError('Explicit FP64 candidate required')
    prior = model.__dict__.get('forward'); had_instance = 'forward' in model.__dict__
    model.forward = forward
    try:
        yield model
    finally:
        if had_instance: model.forward = prior
        else: del model.forward
