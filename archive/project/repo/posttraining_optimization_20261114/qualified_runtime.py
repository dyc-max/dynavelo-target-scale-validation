"""Opt-in gate for a complete nine-model CPU inference cohort.

This does not launch work or waive scientific/ownership gates. Production caller
must register the changed inference implementation and common CPU runtime.
"""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import types
from exact_inference import make_forward,installed,sha,UPSTREAM_SHA

ROOT=Path(r"D:\dyc_7\DynaVelo_Research")
LABELS={f"{kind}_seed{seed}" for kind in ("F","V","M") for seed in range(3)}


def checked(path,digest):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT) or sha(p)!=digest:raise ValueError("Qualified artifact drift")
    return p


def read_cohort(path,digest):
    import torch
    manifest=json.loads(checked(path,digest).read_text(encoding="utf-8"))
    if (manifest.get("schema")!="posttraining_nine_cpu_qualification_v1"
            or manifest.get("device")!="cpu" or set(manifest.get("reports",{}))!=LABELS
            or manifest.get("chunk_cells")!=64 or manifest.get("posterior_draws")!=8
            or manifest.get("master_seed")!=2026111201):raise ValueError("Complete common nine-model recipe required")
    if sha(ROOT/"repo/DynaVelo/dynavelo/models.py")!=UPSTREAM_SHA:raise ValueError("Upstream drift")
    reports={};input_sets=set()
    for label,entry in manifest["reports"].items():
        report=json.loads(checked(entry["path"],entry["sha256"]).read_text(encoding="utf-8"))
        opt=report.get("optimization",{})
        if (report.get("status")!="PASS_BOUNDED_OWN_CHECKPOINT_NUMERICS" or report.get("device")!="cpu"
                or report.get("torch")!=torch.__version__ or report.get("threads")!=4
                or report.get("all_9_outputs_bitwise_equal") is not True
                or report.get("posterior_mean_std_equal") is not True
                or report.get("finite_differences_equal") is not True
                or report.get("perturbation_deltas_equal") is not True
                or report.get("binding",{}).get("label")!=label
                or opt.get("adapter_sha256")!=sha(Path(__file__).with_name("exact_inference.py"))
                or opt.get("upstream_sha256")!=UPSTREAM_SHA):raise ValueError("Unqualified model/backend/source: "+label)
        if len(report.get("cases",[]))!=48 or not all(c["pass"] for c in report["cases"]):raise ValueError("Incomplete case family")
        b=report["binding"]
        input_sets.add(tuple(b[k]["sha256"] for k in ("rna","atac")))
        reports[label]=report
    if len(input_sets)!=1:raise ValueError("Models were qualified on different common inputs")
    return manifest,reports


@contextlib.contextmanager
def use_qualified_cpu(model,label,cohort_path,cohort_sha256):
    """Install only on matching frozen weights; restores on any exit.

    Qualifies a bounded numerical test, not every future intervention. Preserve
    independent science gates and compare original outputs on audit cases.
    """
    import torch
    _,reports=read_cohort(cohort_path,cohort_sha256)
    if (label not in LABELS or str(model.device)!="cpu" or torch.get_num_threads()!=4
            or any(p.device.type!="cpu" for p in model.parameters())):
        raise ValueError("Qualified CPU/threads/label required")
    if torch.get_float32_matmul_precision()!="highest":raise ValueError("Highest FP32 precision required")
    record=reports[label]["binding"]["checkpoint"]
    checkpoint=checked(record["path"],record["sha256"])
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)["model_state_dict"]
    def check_weights():
        current=model.state_dict()
        if current.keys()!=state.keys() or any(not torch.equal(current[k].detach().cpu(),v) for k,v in state.items()):
            raise ValueError("Loaded model differs from qualified own-best checkpoint")
    check_weights()
    candidate=make_forward(model,ROOT/"repo/DynaVelo/dynavelo/models.py")
    def guarded(self,x,y):
        if torch.is_autocast_enabled(torch.device(self.device).type):raise ValueError("Autocast prohibited")
        if not 2<=len(x)<=64 or len(y)!=len(x):raise ValueError("Only qualified 2-64 cell inference blocks")
        return candidate(x,y)
    try:
        with installed(model,types.MethodType(guarded,model)):
            yield {"cohort_sha256":cohort_sha256,"label":label,"checkpoint_sha256":record["sha256"],
                   "optimization":candidate.manifest,"runtime_gate_sha256":sha(__file__),
                   "inference_implementation_changed":True,"scientific_pass":False}
    finally:
        check_weights()
