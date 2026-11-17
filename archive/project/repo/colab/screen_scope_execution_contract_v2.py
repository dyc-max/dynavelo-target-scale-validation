"""Source/artifact/science preflight for scoped primary execution; no approvals."""
from pathlib import Path
import screen_dispatch as base
from screen_plan import canonical, identity, verify_artifact
from screen_scope_plan_runtime_v2 import validate_scope_plan, validate_scope_frame
from human_return_common import sha256, safe_target

MODULES={'scope_plan_module':'screen_scope_plan_runtime_v2.py',
         'scope_contract_module':'screen_scope_execution_contract_v2.py',
         'scope_registry_module':'screen_scope_registry_v2.py',
         'scope_dispatch_module':'screen_scope_dispatch_v2.py',
         'scope_shortlist_module':'screen_scope_shortlist_v2.py'}
STAGES={'baseline','matched-baseline','screen'}


def validate_namespace(plan,root,output):
    root=base.checked_root(root);output=base.checked_root(output)
    relative=output.relative_to(root).as_posix();safe_target(root,relative)
    if relative!='results/screen_runs_v2/'+plan['scope_plan_sha256']:
        raise ValueError('Exactly one scoped registry namespace per execution root')
    contract=plan['context_plan']['spec'].get('scope_execution_contract',{})
    if contract.get('schema')!='dynavelo_scope_execution_contract_v2' or contract.get('registry_execution_root')!=str(root):
        raise PermissionError('Exact registered execution root required; migration unsupported')
    return output


def authorize_scope_stage(plan,review_record,root,stage,*,trusted_reviewers=(),frame=None,guard=None):
    validate_scope_plan(plan);root=base.checked_root(root);context=plan['context_plan']
    def verify(record):
        if guard:guard.check()
        path=verify_artifact(root,record)
        if guard:guard.check()
        return path
    if stage not in STAGES or not review_record or not trusted_reviewers:
        raise PermissionError('Explicit primary stage and independent existing review required')
    if not review_record['relative_path'].startswith('results/reports/'):
        raise PermissionError('Review must be a report artifact')
    review=base.read_json(verify(review_record))
    expected={'schema':'dynavelo_scope_science_review_v2','decision':'APPROVED_FOR_EXPLICIT_EXPLORATORY_SCOPE',
        'scope_plan_sha256':plan['scope_plan_sha256'],'scope_sha256':plan['scope_sha256'],
        'context_plan_sha256':context['plan_sha256'],'checkpoint_set_sha256':context['checkpoint_set_sha256'],
        'population_id':context['spec']['population_id'],'purpose':plan['purpose'],
        'independent_of_plan_preparer':True,'model_quality_gate':'PASS',
        'claim_scope':'exploratory_model_input_sensitivity_only','external_confirmation_or_causal_rescue_authorized':False}
    if any(type(review.get(k))!=type(v) or review.get(k)!=v for k,v in expected.items()):
        raise PermissionError('Review missing/held or belongs to another trained context/effect scope')
    if (review.get('reviewer_id') not in trusted_reviewers or review.get('reviewer_id')==context['spec']['prepared_by']
        or stage not in review.get('allowed_stages',[]) or not set(review.get('allowed_stages',[]))<=STAGES):
        raise PermissionError('Reviewer/stage not independently authorized')
    for requirement in ('model_quality','direction','controls','resource_budget','claim_scope'):
        verify(review['evidence'][requirement])
    inputs=context['spec']['input_artifacts']
    for rec in inputs.values():verify(rec)
    for role,name in MODULES.items():
        path=verify(inputs[role])
        if sha256(Path(__file__).with_name(name))!=inputs[role]['sha256'] or path.read_bytes()!=Path(__file__).with_name(name).read_bytes():
            raise ValueError('Actually imported v2 source differs: '+role)
    for model in context['spec']['models']:
        for key in ('checkpoint','config','metrics'):verify(model[key])
        if plan['purpose']=='human_exploratory_input_sensitivity':
            cfg=base.read_json(verify(model['config']));metrics=base.read_json(verify(model['metrics']))
            if (cfg.get('run_id')!=model['run_id'] or cfg.get('training_seed')!=model['seed'] or cfg.get('split_seed')!=0
                or metrics.get('run_id')!=model['run_id'] or metrics.get('best_checkpoint_sha256')!=model['checkpoint']['sha256']
                or metrics.get('best_checkpoint_reloaded') is not True or cfg.get('analytic_velocity') is not False
                or metrics.get('analytic_velocity') is not False or 'BENCHMARK' in cfg.get('status','')
                or 'BENCHMARK' in metrics.get('status','') or (cfg.get('n_cells'),cfg.get('n_genes'),cfg.get('n_TF'))!=(6035,3006,365)
                or sorted(cfg.get('input_files',{}).values())!=sorted(inputs[k]['sha256'] for k in ('rna','atac'))):
                raise PermissionError('All three actual canonical-best full-context model bindings required before any task')
    if guard:guard.check()
    contract=base.validate_contract(context,root,review_record)
    if guard:guard.check()
    execution=context['spec'].get('scope_execution_contract',{})
    if execution!={'schema':'dynavelo_scope_execution_contract_v2','registry_execution_root':str(root)}:
        raise ValueError('Unknown scoped execution contract or cross-root migration')
    if stage=='baseline':
        if frame is not None:raise ValueError('Full-context baseline precedes frame')
    else:
        validate_scope_frame(plan,frame)
        if frame['status']!='READY_FOR_SEPARATE_SCOPE_FRAME_REVIEW' or review.get('scope_frame_sha256')!=frame['scope_frame_sha256']:
            raise PermissionError('Exact common/local frame must be independently reviewed')
        summaries=[]
        for summary in frame['baseline_summaries']:
            body=base.read_json(verify(summary['artifact']))
            if 'artifact' in body or canonical({**body,'artifact':summary['artifact']})!=canonical(summary):
                raise PermissionError('Frame summary differs from verified complete full baseline')
            summaries.append(body)
    return {'schema':'dynavelo_scope_authorization_v2','stage':stage,'review_sha256':review_record['sha256'],
            'scope_plan_sha256':plan['scope_plan_sha256'],'scope_sha256':plan['scope_sha256'],
            'scope_frame_sha256':frame['scope_frame_sha256'] if frame else None,
            'context_plan_sha256':context['plan_sha256'],'external_confirmation_authorized':False,
            'status':'SCOPED_PREFLIGHT_ONLY_NOT_SCIENTIFIC_PASS','dispatch_contract':contract}
