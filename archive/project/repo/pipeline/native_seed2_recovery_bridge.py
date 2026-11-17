"""Bind recovered seed2 completion to otherwise unchanged native postfit tools.

All original files remain immutable. No model, data, or job is opened by default.
The existing scientific reviews, resource guards and full-screen scope remain.
"""
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(r"D:\dyc_7\DynaVelo_Research")
SELF = Path(__file__).resolve()
RELATIVE = "repo/pipeline/native_seed2_recovery_bridge.py"
RECOVERY = "repo/pipeline/audit_local_seed2_recovery_readout.py"
RECOVERY_SHA = "8c91c621cd553f0d4e399aea22e6278bad5c284fec29f9d9b885b07244057e03"
SOURCES = {
    "evaluate": ("repo/pipeline/evaluate_own_best_local.py", "2f76eafdf8c857d35565b3e4056a436a93ef0ed2af55beb0e53c09ea9e88dc39"),
    "backend": ("repo/pipeline/native_screen_backend.py", "a28c3146988b5805fbd4883ced5c615cdcd3010be6a2d1d4f70f8a1d10cc9150"),
    "screen": ("repo/pipeline/run_native_scope_screen.py", "0b6a6af2ce2f40cdcbf3000d373079081230ec994069b23928fa972d7d625c46"),
    "supervise-screen": ("repo/pipeline/supervise_native_scope_screen.py", "bf32628eb57388a2265e051e4ccc3091cc9ae86dbfb964b55de77f4ddc2c9d7e"),
    "supervise-evaluation": ("repo/pipeline/supervise_local_evaluation.py", "4a4f21f61a7efc3cde9659ff4675c64a66e948b7666ecf4d58a88e0e71bc1031"),
    "audit-screen": ("repo/pipeline/audit_native_scope_invocation.py", "ec7849f7facad66189cb5a7b16616589f8e0a166ac51ece2742d262fb1f834d3"),
}
_CACHE = {}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def binding_pins():
    pins = {path: digest for path, digest in SOURCES.values()}
    pins[RECOVERY] = RECOVERY_SHA
    for relative, digest in pins.items():
        if sha(ROOT / relative) != digest:
            raise ValueError("Frozen recovery bridge source differs: " + relative)
    pins[RELATIVE] = sha(SELF)
    return pins


def edits(role):
    if role == "evaluate":
        return [
            ('2: ("repo/pipeline/audit_local_fresh_readout.py", "047bc3646e619f4336bccf0b1ab55351ab12f9c5fbb5d810b75afb927c6e1567")',
             '2: ("' + RECOVERY + '", "' + RECOVERY_SHA + '")'),
            ("import audit_local_fresh_readout as adapter", "import audit_local_seed2_recovery_readout as adapter"),
            ('str(SELF), "--plan",', 'str(SELF), "--role", "evaluate", "--plan",'),
            ('and q.sha(ROOT / "repo/pipeline/supervise_local_evaluation.py") == plan.get("supervisor_sha256")',
             'and q.sha(SELF) == plan.get("supervisor_sha256")'),
        ]
    if role == "backend":
        return [("import evaluate_own_best_local as completed", 'completed = _adapt("evaluate")')]
    if role == "screen":
        return [
            ("import native_screen_backend as n", 'n = _adapt("backend")'),
            ('SUPERVISOR = "repo/pipeline/supervise_native_scope_screen.py"', 'SUPERVISOR = "' + RELATIVE + '"'),
            ('str(SELF), "--request",', 'str(SELF), "--role", "screen", "--request",'),
        ]
    if role == "supervise-screen":
        return [
            ('ENTRY = ROOT / "repo/pipeline/run_native_scope_screen.py"', 'ENTRY = ROOT / "' + RELATIVE + '"'),
            ("import run_native_scope_screen as entry", 'entry = _adapt("screen")'),
            ('str(ENTRY), "--request",', 'str(ENTRY), "--role", "screen", "--request",'),
        ]
    if role == "supervise-evaluation":
        return [
            ('ENTRY = ROOT / "repo/pipeline/evaluate_own_best_local.py"', 'ENTRY = ROOT / "' + RELATIVE + '"'),
            ("import evaluate_own_best_local as evaluator", 'evaluator = _adapt("evaluate")'),
            ('str(ENTRY), "--plan",', 'str(ENTRY), "--role", "evaluate", "--plan",'),
        ]
    if role == "audit-screen":
        return [
            ('ENTRY = "repo/pipeline/run_native_scope_screen.py"', 'ENTRY = "' + RELATIVE + '"'),
            ('ENTRY_SHA = "0b6a6af2ce2f40cdcbf3000d373079081230ec994069b23928fa972d7d625c46"',
             'ENTRY_SHA = hashlib.sha256(SELF.read_bytes()).hexdigest()'),
            ('SUPERVISOR_SHA = "bf32628eb57388a2265e051e4ccc3091cc9ae86dbfb964b55de77f4ddc2c9d7e"',
             'SUPERVISOR_SHA = hashlib.sha256(SELF.read_bytes()).hexdigest()'),
            ("import run_native_scope_screen as entry", 'entry = _adapt("screen")'),
            ('str(ROOT / ENTRY), "--request",', 'str(ROOT / ENTRY), "--role", "screen", "--request",'),
        ]
    raise ValueError("Unsupported bridge role")


def adapted_source(role):
    binding_pins()
    source = (ROOT / SOURCES[role][0]).read_text(encoding="utf-8")
    for before, after in edits(role):
        if source.count(before) != 1:
            raise ValueError("Recovery bridge source anchor differs: " + before)
        source = source.replace(before, after, 1)
    return source


def adapt(role):
    binding_pins()
    if role in _CACHE:
        return _CACHE[role]
    module = types.ModuleType("_recovered_seed2_" + role.replace("-", "_"))
    module.__dict__.update(__file__=str(SELF), _adapt=adapt)
    _CACHE[role] = module
    try:
        exec(compile(adapted_source(role), "<pinned_native_source_recovery_provenance_only>", "exec"), module.__dict__)
        if hasattr(module, "PINS"):
            module.PINS.update(binding_pins())
    except BaseException:
        del _CACHE[role]
        raise
    return module


def inspect():
    compiled = {}
    for role in SOURCES:
        source = adapted_source(role)
        original = (ROOT / SOURCES[role][0]).read_text(encoding="utf-8")
        reverse = source
        for before, after in reversed(edits(role)):
            if reverse.count(after) != 1:
                raise ValueError("Recovery bridge inverse differs")
            reverse = reverse.replace(after, before, 1)
        if reverse != original:
            raise ValueError("Undeclared source difference")
        compiled[role] = hashlib.sha256(source.encode()).hexdigest()
    return {"status": "PROVENANCE_BRIDGE_SOURCE_ONLY", "source_sha256": sha(SELF),
            "pins": binding_pins(), "compiled_sha256": compiled,
            "numerical_changes": False, "scientific_review_requirements_changed": False,
            "all_original_sources_recovered_by_inverse_edits": True,
            "torch_imported": "torch" in sys.modules, "human_results_opened": False,
            "scientific_pass": False, "production_invocation_authorized": False}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", choices=["inspect", *[r for r in SOURCES if r != "backend"]], default="inspect")
    args, remaining = parser.parse_known_args()
    if args.role == "inspect":
        if remaining:
            raise ValueError("Inspection accepts no execution arguments")
        print(json.dumps(inspect())); return
    module = adapt(args.role)
    sys.argv = [str(SELF), *remaining]
    module.main()


if __name__ == "__main__":
    main()
