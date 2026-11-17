"""Complete-family scoped ranking using unchanged percentile/top20 arithmetic."""
import json
import numpy as np
import screen_dispatch as base
from screen_plan import canonical,identity,seal,check_seal
from screen_shortlist import percentile,_rank_boundary
from screen_scope_plan_runtime_v2 import validate_scope_plan,validate_scope_frame
from screen_scope_execution_contract_v2 import authorize_scope_stage


def _derive_shortlist(plan,frame,registry,checkpoint):
    validate_scope_plan(plan);validate_scope_frame(plan,frame)
    if (registry.book['scope_plan_sha256']!=plan['scope_plan_sha256']
        or registry.book['scope_frame_sha256']!=frame['scope_frame_sha256']):raise ValueError('Registry scope differs')
    coverage=registry.require_full_screen_coverage(checkpoint)
    universe=plan['context_plan']['tf_universe'];ids=[r['stable_id'] for r in universe];positions={v:i for i,v in enumerate(ids)}
    scores={sample:np.full((3,len(ids)),np.nan) for sample in ('primary','stability')}
    for task in registry.book['tasks']:
        if checkpoint:checkpoint()
        saved=registry.resume(task['task_id']);path=registry.path(saved['manifest_relative_path']).parent/'result.json'
        value=json.loads(path.read_text(encoding='utf-8'))
        scores[task['sample_id']][task['seed'],positions[task['tf_stable_id']]]=value['primary_effect']
    if any(not np.isfinite(x).all() for x in scores.values()):raise ValueError('Incomplete finite ranking family')
    ranks={sample:np.stack([percentile(np.abs(a[seed])) for seed in (0,1,2)]) for sample,a in scores.items()}
    selected,median,boundary=_rank_boundary(ids,scores['primary']);rows=[]
    for i in selected:
        rows.append({**universe[i],'primary_median_percentile':float(median[i]),
            'primary_effect_by_seed':scores['primary'][:,i].tolist(),'stability_effect_by_seed':scores['stability'][:,i].tolist(),
            'primary_seeds_in_top_5_percent':int((ranks['primary'][:,i]>=.95).sum()),
            'stability_seeds_in_top_5_percent':int((ranks['stability'][:,i]>=.95).sum()),
            'joint_followup':'not_implemented' if universe[i]['has_selected_motif'] else 'not_applicable','claim_status':'NOT_CONFIRMED'})
    return seal({'schema':'dynavelo_scope_computational_shortlist_v2','scope_plan_sha256':plan['scope_plan_sha256'],
        'scope_sha256':plan['scope_sha256'],'context_plan_sha256':plan['context_plan_sha256'],
        'scope_frame_sha256':frame['scope_frame_sha256'],'full_context_frame_sha256':frame['full_context_frame_sha256'],
        'calibration_sha256':frame['calibration_sha256'],'book_sha256':registry.book['book_sha256'],
        'coverage_sha256':coverage['coverage_sha256'],'full_tf_denominator':len(ids),'maximum':20,'selected':rows,
        'boundary_tied_stable_ids':sorted(boundary),'boundary_unselected_tied_stable_ids':sorted(set(boundary)-{ids[i] for i in selected}),
        'ranking_family_sha256':identity({sample:a.tolist() for sample,a in scores.items()}),
        'stability_policy':plan['stability_policy'],'no_replacement_after_failure':True,
        'status':'COMPUTATIONAL_TRUNCATION_ONLY_REVIEW_REQUIRED','external_confirmation_or_rescue_claim':False},'shortlist_sha256')


def derive_shortlist(plan,frame,registry,checkpoint=None):
    with registry.dependency_audit(checkpoint):
        return _derive_shortlist(plan,frame,registry,checkpoint)


def freeze_shortlist(plan,frame,registry,review_record,root,*,trusted_reviewers=(),stop_file=None):
    guard=base.ResourceGuard(base.checked_root(root),plan['context_plan']['spec']['dispatch_contract'],stop_file)
    with guard.signals():
        authorize_scope_stage(plan,review_record,root,'screen',trusted_reviewers=trusted_reviewers,frame=frame,guard=guard)
        value=derive_shortlist(plan,frame,registry,guard.check);guard.check()
        path=registry.path('locked_scope_shortlist.json');data=canonical(value)
        if path.exists():
            if path.read_bytes()!=data:raise FileExistsError('Locked shortlist cannot be replaced')
        else:registry.atomic_new('locked_scope_shortlist.json',data)
    return value


def validate_shortlist(plan,frame,registry,value,review_record,root,*,trusted_reviewers=(),stop_file=None):
    guard=base.ResourceGuard(base.checked_root(root),plan['context_plan']['spec']['dispatch_contract'],stop_file)
    with guard.signals():
        authorize_scope_stage(plan,review_record,root,'screen',trusted_reviewers=trusted_reviewers,frame=frame,guard=guard)
        check_seal(value,'shortlist_sha256')
        if canonical(value)!=canonical(derive_shortlist(plan,frame,registry,guard.check)):raise ValueError('Shortlist differs from complete verified family')
    return value
