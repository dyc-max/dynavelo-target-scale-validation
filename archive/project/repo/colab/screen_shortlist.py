"""Freeze a computational top20 only after exact all-TF/all-seed coverage.

Internal descriptive ranks only. No external validation or positive claim.
"""
import json
import numpy as np
from screen_plan import authorize_stage,canonical,identity,seal,check_seal,validate_plan,validate_frame

def percentile(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite complete effects required')
    if len(values)==1:return np.ones(1)
    order=np.argsort(values,kind='stable');ranks=np.empty(len(values));left=0
    while left<len(order):
        right=left+1
        while right<len(order) and values[order[right]]==values[order[left]]:right+=1
        ranks[order[left:right]]=((left+right-1)/2)/(len(values)-1)
        left=right
    return ranks

def _rank_boundary(ids,primary_scores):
    ranks=np.stack([percentile(np.abs(primary_scores[seed])) for seed in (0,1,2)])
    median=np.median(ranks,axis=0)
    order=sorted(range(len(ids)),key=lambda i:(-median[i],ids[i]));selected=order[:min(20,len(ids))]
    cutoff=float(median[selected[-1]])
    return selected,median,[ids[i] for i in range(len(ids)) if median[i]==cutoff]

def _derive_shortlist(plan,frame,registry):
    validate_plan(plan);validate_frame(plan,frame)
    if registry.book['plan_sha256']!=plan['plan_sha256'] or registry.book['frame_sha256']!=frame['frame_sha256']:
        raise ValueError('Registry belongs to another plan/frame')
    coverage=registry.require_full_screen_coverage()
    universe=plan['tf_universe'];ids=[item['stable_id'] for item in universe];positions={name:i for i,name in enumerate(ids)}
    scores={sample:np.full((3,len(ids)),np.nan) for sample in ('primary','stability')}
    for task in registry.book['tasks']:
        saved=registry.resume(task['task_id']);directory=registry.path(saved['manifest_relative_path']).parent
        value=json.loads((directory/'result.json').read_text())
        scores[task['sample_id']][task['seed'],positions[task['tf_stable_id']]]=value['primary_effect']
    if any(not np.isfinite(value).all() for value in scores.values()):raise ValueError('Cannot rank an incomplete effect family')
    ranks={sample:np.stack([percentile(np.abs(values[seed])) for seed in (0,1,2)]) for sample,values in scores.items()}
    selected,median,boundary=_rank_boundary(ids,scores['primary'])
    rows=[]
    for i in selected:
        row={**universe[i],'primary_median_percentile':float(median[i]),
             'primary_effect_by_seed':scores['primary'][:,i].tolist(),'stability_effect_by_seed':scores['stability'][:,i].tolist(),
             'primary_seeds_in_top_5_percent':int((ranks['primary'][:,i]>=.95).sum()),
             'stability_seeds_in_top_5_percent':int((ranks['stability'][:,i]>=.95).sum()),
             'joint_followup':'planned_secondary' if universe[i]['has_selected_motif'] else 'not_applicable',
             'claim_status':'NOT_CONFIRMED'}
        rows.append(row)
    return seal({'schema':'dynavelo_computational_shortlist_v1','plan_sha256':plan['plan_sha256'],'frame_sha256':frame['frame_sha256'],
                 'book_sha256':registry.book['book_sha256'],'coverage_sha256':coverage['coverage_sha256'],
                 'full_tf_denominator':len(ids),'maximum':20,'selected':rows,'boundary_tied_stable_ids':sorted(boundary),
                 'boundary_unselected_tied_stable_ids':sorted(set(boundary)-{ids[i] for i in selected}),
                 'rank_rule':'ascending average-tie percentiles of absolute E per seed; median across seeds; descending median then stable-ID lexical',
                 'ranking_family_sha256':identity({sample:value.tolist() for sample,value in scores.items()}),
                 'no_replacement_after_failure':True,'status':'COMPUTATIONAL_TRUNCATION_ONLY_REVIEW_REQUIRED',
                 'external_confirmation_or_rescue_claim':False},'shortlist_sha256')

def freeze_shortlist(plan,frame,registry,review_record,project_root,*,trusted_reviewers=()):
    authorize_stage(plan,review_record,project_root,'screen',frame=frame,trusted_reviewers=trusted_reviewers)
    result=_derive_shortlist(plan,frame,registry)
    name='locked_shortlist.json';path=registry.path(name);data=canonical(result)
    if path.exists():
        if path.read_bytes()!=data:raise FileExistsError('Locked shortlist cannot be replaced by a more favorable selection')
    else:registry.atomic_new(name,data)
    return result

def validate_shortlist(plan,frame,registry,value,review_record,project_root,*,trusted_reviewers=()):
    authorize_stage(plan,review_record,project_root,'screen',frame=frame,trusted_reviewers=trusted_reviewers)
    check_seal(value,'shortlist_sha256')
    if canonical(value)!=canonical(_derive_shortlist(plan,frame,registry)):
        raise ValueError('Shortlist does not derive from the entire verified frozen family')
    return value
