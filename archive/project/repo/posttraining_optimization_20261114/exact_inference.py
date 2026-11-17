"""Opt-in evaluation candidate. Does not patch any training or production file.

Keep ODE, FP32, posterior draws, batch geometry, Jacobian tensors and bmm.
Only construct the exact Linear/ReLU decoder Jacobian directly and replace
per-cell copies with indexed copies. Deployment requires backend/model tests.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import types
from pathlib import Path

UPSTREAM_SHA = "ff9ad39be46382a731d03b8899e4a0ccb7e82c93dfecb6ff1fd572f251e118ca"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard(model):
    import torch
    from torch import nn
    if model.training or model.mode not in {"evaluation-fixed", "evaluation-sample"}:
        raise ValueError("Evaluation only; training optimization is prohibited")
    if any(m.training for m in model.modules()):
        raise ValueError("All modules must be in eval mode")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 parameters required")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Autocast is outside this candidate's contract")
    for m in model.modules():
        if m._forward_hooks or m._forward_pre_hooks or m._backward_hooks:
            raise ValueError("Hooks are not qualified")
    x, y = model.decoder_net_x, model.decoder_net_y
    if not (type(x) is nn.Sequential and len(x) == 2 and type(x[0]) is nn.Linear
            and type(x[1]) is nn.ReLU and not x[1].inplace
            and type(y) is nn.Sequential and len(y) == 1 and type(y[0]) is nn.Linear):
        raise ValueError("Only original Linear/ReLU and Linear decoders supported")
    if (x[0].in_features, x[0].out_features, y[0].in_features, y[0].out_features) != (
            model.zx_dim, model.x_dim, model.zy_dim, model.y_dim):
        raise ValueError("Decoder axes differ")


def structured_jacobians(model, x_pred, zy):
    """Materialize original-size J, with original contiguous layout, then bmm.

    Unlike the earlier F.linear(v, W) shortcut, this does not change the
    velocity dot-product operation or its reduction dimensions.
    """
    import torch
    wx = model.decoder_net_x[0].weight
    wy = model.decoder_net_y[0].weight
    jx = torch.where(x_pred.unsqueeze(-1) > 0, wx.unsqueeze(0), 0.0).contiguous()
    jy = wy.unsqueeze(0).expand(zy.shape[0], -1, -1).contiguous()
    return jx, jy


class _Transform(ast.NodeTransformer):
    def __init__(self, structured):
        self.structured = structured
        self.copies = self.jacobians = 0

    def visit_For(self, node):
        if (ast.unparse(node.target) == "(idx, cell)"
                and ast.unparse(node.iter) == "enumerate(indices)"):
            expected = ast.parse("for idx, cell in enumerate(indices):\n z[cell,:] = z_all_times[idx+1,cell,:]\n vz[cell,:] = vz_all_times[idx+1,cell,:]").body[0]
            if ast.dump(node) != ast.dump(expected):
                raise ValueError("Original copy-loop drift")
            self.copies += 1
            return ast.parse("_rows = torch.arange(1, len(indices)+1, device=indices.device)\nz[indices, :] = z_all_times[_rows, indices, :]\nvz[indices, :] = vz_all_times[_rows, indices, :]").body
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if not self.structured or len(node.targets) != 1:
            return node
        name = ast.unparse(node.targets[0])
        if name == "Jx":
            if ast.unparse(node.value) != "vmap(jacrev(self.decoder_net_x))(zx)":
                raise ValueError("Original RNA Jacobian drift")
            self.jacobians += 1
            return ast.parse("Jx, Jy = _structured_jacobians(self, x_pred, zy)").body[0]
        if name == "Jy":
            if ast.unparse(node.value) != "vmap(jacrev(self.decoder_net_y))(zy)":
                raise ValueError("Original motif Jacobian drift")
            return None
        return node


def make_forward(model, upstream_path, *, structured=True):
    """Compile an independent, guarded forward from the pinned original AST."""
    import torch
    guard(model)
    if sha(upstream_path) != UPSTREAM_SHA:
        raise ValueError("Pinned upstream source changed")
    tree = ast.parse(Path(upstream_path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DynaVelo")
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    # Verify the installed loader really preserves the original forward.
    env = dict(model.forward.__func__.__globals__)
    current = model.forward.__func__.__code__
    # Compile in its original module/class context: CPython optimizes imported
    # module attributes differently from a detached function. Never execute it.
    compiled_source = compile(tree, "<original_forward_identity>", "exec")
    class_code = next(c for c in compiled_source.co_consts if isinstance(c, types.CodeType) and c.co_name == "DynaVelo")
    expected = next(c for c in class_code.co_consts if isinstance(c, types.CodeType) and c.co_name == "forward")
    if (current.co_code, current.co_names, current.co_consts) != (expected.co_code, expected.co_names, expected.co_consts):
        raise ValueError("Installed forward differs from original; cannot stack unknown adapters")
    transform = _Transform(structured)
    changed = transform.visit(copy.deepcopy(original))
    if transform.copies != 3 or (structured and transform.jacobians != 3):
        raise ValueError("Unexpected number of original forward blocks")
    # Although the copied function contains the original train branch, wrapper
    # rejects train mode and autograd. No training call can reach it.
    module = ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[]))
    env["_structured_jacobians"] = structured_jacobians
    exec(compile(module, "<posttraining_exact_candidate>", "exec"), env)
    compiled = env["forward"]

    def candidate(self, x, y):
        guard(self)
        if torch.is_grad_enabled():
            raise ValueError("Use torch.no_grad(); candidate is inference only")
        if x.dtype != torch.float32 or y.dtype != torch.float32 or len(x) < 2:
            raise ValueError("FP32 input and original non-singleton batches required")
        return compiled(self, x, y)

    candidate.manifest = {
        "upstream_sha256": UPSTREAM_SHA,
        "adapter_sha256": sha(__file__),
        "generated_forward_sha256": hashlib.sha256(ast.unparse(module).encode()).hexdigest(),
        "structured_jacobian_then_original_bmm": structured,
        "vectorized_index_copies": True,
        "ode_precision_batching_rng_unchanged": True,
        "training_supported": False,
        "production_enabled": False,
    }
    return types.MethodType(candidate, model)


@contextlib.contextmanager
def installed(model, forward):
    """Temporary instance-local substitution; restores on success or failure."""
    prior = model.__dict__.get("forward")
    had_instance_forward = "forward" in model.__dict__
    model.forward = forward
    try:
        yield model
    finally:
        if had_instance_forward:
            model.forward = prior
        else:
            del model.forward


def state_pair_aggregates(graph, codes, n_states):
    """Exact original row-major np.add.at order, without a Python row loop.

    Caller retains provenance, sign/finite/axis checks and CSR duplicate/zero
    normalization from summarize_model_graph_directions.py.
    """
    import numpy as np
    from scipy import sparse
    codes = np.asarray(codes)
    if (not sparse.isspmatrix_csr(graph) or codes.shape != (graph.shape[0],)
            or graph.shape[0] != graph.shape[1] or codes.dtype.kind not in "iu"
            or n_states < 1 or np.any(codes < 0) or np.any(codes >= n_states)
            or np.any(graph.indices < 0) or np.any(graph.indices >= len(codes))
            or not np.isfinite(graph.data).all()):
        raise ValueError("Invalid graph/cell/state axes")
    source = np.repeat(np.arange(len(codes), dtype=np.int64), np.diff(graph.indptr))
    target_state = codes[graph.indices]
    pair = codes[source] * n_states + target_state
    sums = np.zeros(n_states * n_states, dtype=np.float64)
    edges = np.zeros_like(sums, dtype=np.int64)
    sources = np.zeros_like(edges)
    np.add.at(sums, pair, graph.data)
    np.add.at(edges, pair, 1)
    unique_source_target = np.unique(source * n_states + target_state)
    source_pair = codes[unique_source_target // n_states] * n_states + unique_source_target % n_states
    np.add.at(sources, source_pair, 1)
    return tuple(a.reshape(n_states, n_states) for a in (sums, edges, sources))
