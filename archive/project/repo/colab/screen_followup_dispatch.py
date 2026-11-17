"""Prospectively reviewed GRN/epsilon/locked-shortlist execution; no approvals.

Uses unchanged core/foundation and the primary dispatch's source/input/resource
and transactional safeguards. No numerical result is a biological PASS.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import uuid
import numpy as np
import torch
import screen_dispatch as base
from human_return_common import safe_target, sha256
from screen_plan import (authorize_stage, canonical, identity, seal, validate_plan,
                         validate_frame, verify_artifact)
from screen_task_registry import TaskRegistry, task_book, validate_book, batches
from screen_shortlist import validate_shortlist
from dynavelo_streaming_core import EdgeAccumulator, batches_without_singleton

OPERATIONS={'grn','epsilon-baseline','epsilon','full-cell-rna','full-cell-joint','full-cell-grn'}
EPSILONS=(1e-5,1e-4,1e-3)

def diagnostic_axes(plan):
    c=plan['spec']['followup_contract']
    fields=[('epsilon_cell_count',len(plan['spec']['cells']),2),
            ('epsilon_rna_regulator_count',len(plan['tf_universe']),1),
            ('epsilon_motif_regulator_count',len(plan['spec']['motifs']),1)]
    for key,maximum,minimum in fields:
        if type(c.get(key)) is not int or not minimum<=c[key]<=maximum:
            raise ValueError('Invalid prospectively frozen diagnostic count: '+key)
    return {'cell_ids':sorted(x['cell_id'] for x in plan['spec']['cells'])[:c['epsilon_cell_count']],
            'rna':sorted(plan['tf_universe'],key=lambda x:x['stable_id'])[:c['epsilon_rna_regulator_count']],
            'motif':sorted(plan['spec']['motifs'],key=lambda x:x['feature_id'])[:c['epsilon_motif_regulator_count']]}

def motif_regulator(row):
    return {'stable_id':row['feature_id'],'feature_id':row['feature_id'],'input_space':'motif'}

def followup_book(plan,frame,shortlist,operation):
    validate_plan(plan);validate_frame(plan,frame)
    if operation not in OPERATIONS:raise ValueError('Unknown operation')
    if operation=='grn':return task_book(plan,'grn',frame)
    diagnostic=diagnostic_axes(plan);geometry=plan['spec']['chunk_geometry']
    if operation.startswith('epsilon'):
        sample='epsilon_fixed';ids=diagnostic['cell_ids']
        regulators=([{**r,'input_space':'rna'} for r in diagnostic['rna']]+[motif_regulator(r) for r in diagnostic['motif']])
        if operation=='epsilon-baseline':regulators=[{'stable_id':'__baseline__','feature_id':'__baseline__','input_space':'none'}]
    else:
        sample='full_common';ids=frame['common_eligible_cell_ids']
        selected=shortlist['selected'];regulators=[{**r,'input_space':'rna'} for r in selected]
        if operation=='full-cell-joint':regulators=[r for r in regulators if r['has_selected_motif']]
        if operation=='full-cell-grn':
            selected_features={r['feature_id'] for r in selected}
            regulators += [motif_regulator(r) for r in plan['spec']['motifs'] if r['tf_rna_feature_id'] in selected_features]
    mode={'epsilon-baseline':'evaluation-fixed','epsilon':'finite-difference','full-cell-rna':'rna-only',
          'full-cell-joint':'official-joint','full-cell-grn':'finite-difference'}[operation]
    eps=EPSILONS if operation=='epsilon' else (1e-4,) if operation=='full-cell-grn' else (None,)
    tasks=[]
    for model in sorted(plan['spec']['models'],key=lambda r:r['seed']):
        for reg in regulators:
            for epsilon in eps:
                body={'schema':'dynavelo_screen_task_v1','plan_sha256':plan['plan_sha256'],
                    'frame_sha256':frame['frame_sha256'],'checkpoint_sha256':model['checkpoint']['sha256'],
                    'seed':model['seed'],'stage':operation,'sample_id':sample,'sample_cell_axis_sha256':identity(ids),
                    'n_cells':len(ids),'tf_stable_id':reg['stable_id'],'feature_id':reg['feature_id'],
                    'input_space':reg['input_space'],'mode':mode,'chunk_geometry':geometry,
                    'batches':batches(len(ids),geometry['cells']),'full_global_minima_sha256':identity(plan['spec']['global_minima']),
                    'epsilon':epsilon,'shortlist_sha256':shortlist['shortlist_sha256'],
                    'followup_contract_sha256':identity(plan['spec']['followup_contract'])}
                task=seal(body,'task_id');task['assigned_shard']=int(task['task_id'][:16],16)%geometry.get('regulator_shards',1);tasks.append(task)
    return seal({'schema':'dynavelo_task_book_v1','stage':operation,'plan_sha256':plan['plan_sha256'],
        'frame_sha256':frame['frame_sha256'],'tasks':tasks,'full_tf_denominator':len(plan['tf_universe']),
        'shortlist_sha256':shortlist['shortlist_sha256'],'shortlist_denominator':len(shortlist['selected']),
        'expected_tasks':len(tasks),'technical_attempt_limit':3,
        'coverage_contract':'All frozen tasks, including failures; joint RNA-only applicability rows separately retained'},'book_sha256')

class FollowupRegistry(TaskRegistry):
    """Exact deterministic new books; inherit unchanged lock/commit/resume format."""
    def __init__(self,root,book,*,plan,frame,shortlist):
        validate_book(book)
        if canonical(book)!=canonical(followup_book(plan,frame,shortlist,book['stage'])):
            raise ValueError('Follow-up book is not the entire deterministic family')
        self.root=base.checked_root(root);self.root.mkdir(parents=True,exist_ok=True)
        self.book=book;self.tasks={t['task_id']:t for t in book['tasks']};self.plan=plan;self.frame=frame
        path=self.path('task_book.json');data=canonical(book)
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Registry already binds a different follow-up book')
        else:self.atomic_new('task_book.json',data)

def stage_for(operation):return 'full-cell-review' if operation in {'full-cell-rna','full-cell-joint'} else 'grn'

def checkpoint(guard):
    if guard is not None:guard.check()

def primary_registry(plan,frame,output,guard=None):
    registry=TaskRegistry(output/'screen',task_book(plan,'screen',frame),plan=plan,frame=frame)
    matched=TaskRegistry(output/'matched-baseline',task_book(plan,'matched-baseline',frame),plan=plan,frame=frame)
    verified={}
    for task in registry.book['tasks']:
        checkpoint(guard)
        directory,_=base.completed_directory(registry,task)
        deps=[t for t in matched.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
        if len(deps)!=1:raise ValueError('Screen dependency not unique')
        key=deps[0]['task_id']
        if key not in verified:verified[key]=base.completed_directory(matched,deps[0])
        _,saved=verified[key]
        expected={'task_id':deps[0]['task_id'],'commit_sha256':saved['manifest_sha256'],'book_sha256':matched.book['book_sha256']}
        if base.read_json(directory/'result.json').get('matched_baseline_dependency')!=expected:
            raise ValueError('Original completed screen dependency changed')
        checkpoint(guard)
    return registry

def authorize_followup(plan,frame,shortlist,screen_review,review_record,root,operation,output,trusted,guard=None):
    if operation not in OPERATIONS:raise ValueError('Unsupported follow-up')
    checkpoint(guard)
    base.validate_output_namespace(plan,root,output)
    authorization=authorize_stage(plan,review_record,root,stage_for(operation),trusted_reviewers=trusted,frame=frame)
    checkpoint(guard)
    contract=base.validate_contract(plan,root,review_record);followup=plan['spec'].get('followup_contract',{})
    if followup.get('schema')!='dynavelo_screen_followup_contract_v1':raise ValueError('Freeze follow-up contract before effects')
    allowed=followup.get('allowed_operations',[])
    if not isinstance(allowed,list) or len(set(allowed))!=len(allowed) or not set(allowed)<=OPERATIONS or operation not in allowed:
        raise PermissionError('Operation not prospectively frozen')
    if followup.get('epsilons')!=list(EPSILONS) or followup.get('selection_rule')!='lexical_original_cell_and_stable_regulator_IDs':
        raise ValueError('No adaptive diagnostic axes/epsilon permitted')
    diagnostic_axes(plan)
    source=verify_artifact(root,plan['spec']['input_artifacts']['followup_module'])
    if sha256(source)!=sha256(__file__):raise ValueError('Actually imported follow-up source not frozen')
    full=TaskRegistry(output/'baseline',task_book(plan,'baseline'),plan=plan)
    if canonical(base.full_frame_from_registry(plan,full,root))!=canonical(frame):raise ValueError('Complete baseline registry/frame mismatch')
    checkpoint(guard)
    screen=primary_registry(plan,frame,output,guard)
    validate_shortlist(plan,frame,screen,shortlist,screen_review,root,trusted_reviewers=trusted)
    checkpoint(guard)
    coverage=screen.require_full_screen_coverage();review=base.read_json(verify_artifact(root,review_record))
    approval=review.get('followup',{})
    expected={'contract_sha256':identity(followup),'screen_coverage_sha256':coverage['coverage_sha256'],
              'shortlist_sha256':shortlist['shortlist_sha256']}
    if any(approval.get(k)!=v for k,v in expected.items()) or operation not in approval.get('allowed_operations',[]):
        raise PermissionError('Follow-up review does not bind exact family/shortlist/operation')
    stability=base.read_json(verify_artifact(root,approval['stability_review']))
    if (stability.get('schema')!='dynavelo_screen_stability_review_v1' or stability.get('decision')!='APPROVED_FOR_EXPLICIT_EXPLORATORY_SCOPE'
        or stability.get('reviewer_id') not in trusted or stability.get('reviewer_id')==plan['spec']['prepared_by']
        or stability.get('independent_of_plan_preparer') is not True or stability.get('plan_sha256')!=plan['plan_sha256']
        or stability.get('frame_sha256')!=frame['frame_sha256'] or stability.get('screen_coverage_sha256')!=coverage['coverage_sha256']
        or stability.get('shortlist_sha256')!=shortlist['shortlist_sha256'] or operation not in stability.get('allowed_operations',[])
        or stability.get('claim_scope')!='exploratory_model_input_sensitivity_only'
        or stability.get('external_confirmation_or_causal_rescue_authorized') is not False):
        raise PermissionError('Independent stability review missing or outside exact family/scope')
    if operation in {'grn','full-cell-grn'}:
        epsreg=FollowupRegistry(output/'epsilon',followup_book(plan,frame,shortlist,'epsilon'),plan=plan,frame=frame,shortlist=shortlist)
        ec=epsreg.coverage()
        if not ec['all_expected_tasks_complete'] or approval.get('epsilon_coverage_sha256')!=ec['coverage_sha256']:
            raise PermissionError('Full fixed epsilon family must be completed and separately reviewed before network')
        er=dependency_registry(plan,frame,shortlist,output,'epsilon');verified={}
        for task in epsreg.book['tasks']:
            checkpoint(guard);verify_completed_dependency(epsreg,er,task,verified)
        summary=epsilon_comparison(epsreg,guard)
        evidence=base.read_json(verify_artifact(root,approval['epsilon_comparison']))
        if canonical(summary)!=canonical(evidence):raise PermissionError('Reviewed epsilon summary differs')
    authorization.update(screen_coverage_sha256=coverage['coverage_sha256'],shortlist_sha256=shortlist['shortlist_sha256'])
    checkpoint(guard)
    return authorization,contract

def cells_for(plan,frame,task):
    return diagnostic_axes(plan)['cell_ids'] if task['sample_id']=='epsilon_fixed' else base.sample_ids(plan,frame,task)

def frame_weights(plan,frame,task,indices):
    if task['sample_id']=='full_common':return np.full(len(indices),1/len(indices),dtype=np.float64)
    rows=frame['samples'][task['sample_id']]['cells']
    if [r['cell_id'] for r in rows]!=[plan['spec']['cells'][i]['cell_id'] for i in indices]:raise ValueError('Weight axis differs')
    weights=np.array([r['cell_weight'] for r in rows],dtype=np.float64)
    if abs(weights.sum()-1)>1e-12:raise ValueError('Frame weights do not sum to one')
    return weights

def group_definitions(plan,frame,task,indices):
    labels=np.array([plan['spec']['cells'][i]['label'] for i in indices],dtype=object)
    full_labels=np.array([r['label'] for r in plan['spec']['cells']],dtype=object)
    eligible=np.array([r['cell_id'] in set(frame['common_eligible_cell_ids']) for r in plan['spec']['cells']])
    windows=np.array(frame['global_time_windows'][str(task['seed'])]['window_index'])
    groups=[('global','all',np.ones(len(indices),dtype=bool),int(eligible.sum()))]
    groups += [('label',label,labels==label,int(((full_labels==label)&eligible).sum())) for label in sorted(set(full_labels))]
    groups += [('branch',name,np.isin(labels,states),int((np.isin(full_labels,states)&eligible).sum())) for name,states in plan['policy']['branches'].items()]
    groups += [('full_original_time_window',str(w),windows[indices]==w,int(((windows==w)&eligible).sum())) for w in range(5)]
    return groups

def regulator_name(plan,task):
    if task['input_space']=='rna':return task['feature_id']
    rows=[r for r in plan['spec']['motifs'] if r['feature_id']==task['feature_id']]
    if len(rows)!=1:raise ValueError('Motif feature-to-TF mapping not unique')
    return rows[0]['tf_rna_feature_id']

def checked_jacobians(plan,task,session):
    """Validate every cell/target block against original fixed complete axes."""
    regulator=regulator_name(plan,task);target_chunk=task['chunk_geometry']['targets']
    expected=((start,stop,channel,left,min(left+target_chunk,len(names)),names[left:left+target_chunk])
        for start,stop in session.batches for channel,names in [('vx',session.population.genes),('vy',session.population.motif_tfs)]
        for left in range(0,len(names),target_chunk))
    count=0
    for block in session.jacobians([regulator],input_space=task['input_space'],target_chunk=target_chunk,epsilon=task['epsilon']):
        wanted=next(expected,None)
        if wanted is None:raise ValueError('Extra derivative block')
        start,stop,channel,left,right,names=wanted
        actual=(block['start'],block['stop'],block['channel'],block['target_start'],block['target_stop'])
        if (actual!=(start,stop,channel,left,right) or list(block['targets'])!=list(names)
            or block['regulator']!=regulator or block['input_space']!=task['input_space']
            or not np.array_equal(block['population_indices'],session.indices[start:stop])
            or block['values'].shape!=(stop-start,right-left) or block['values'].dtype!=np.float64):
            raise ValueError('Derivative block differs from exact regulator/cell/target axes')
        count+=1;yield block
    if next(expected,None) is not None or not count:raise ValueError('Incomplete derivative target/cell family')

def grn_task(plan,frame,task,session,writer,authorization,dependency):
    groups=group_definitions(plan,frame,task,session.indices);weights=frame_weights(plan,frame,task,session.indices)
    axes={'vx':list(session.population.genes),'vy':[r['feature_id'] for r in plan['spec']['motifs']]}
    accum={key:EdgeAccumulator(len(groups),len(names)) for key,names in axes.items()}
    weighted={key:{name:np.zeros((len(groups),len(names)),dtype=np.float64) for name in ('signed_sum','absolute_sum','finite_mass','nonfinite_mass')}
              for key,names in axes.items()}
    for block in checked_jacobians(plan,task,session):
        channel=block['channel'];left=block['target_start'];right=block['target_stop'];s,e=block['start'],block['stop'];values=block['values']
        for group,(_,_,mask,_) in enumerate(groups):
            select=mask[s:e]
            if not select.any():continue
            v=values[select];w=weights[s:e][select,None];finite=np.isfinite(v);valid=np.where(finite,v,0)
            accum[channel].update(v,np.full(len(v),group,dtype=np.int64),left)
            for key,part in [('signed_sum',(valid*w).sum(axis=0)),('absolute_sum',(np.abs(valid)*w).sum(axis=0)),
                             ('finite_mass',(finite*w).sum(axis=0)),('nonfinite_mass',((~finite)*w).sum(axis=0))]:
                weighted[channel][key][group,left:right]+=part
    group_rows=[{'kind':kind,'name':name,'sample_n':int(mask.sum()),'eligible_frame_domain_n':n,
                 'sample_weight_sum':float(weights[mask].sum())} for kind,name,mask,n in groups]
    writer.put('edge_axes.json',{'rna_output_axis':axes['vx'],'motif_output_axis':axes['vy'],
        'motif_output_tf_mapping':list(session.population.motif_tfs),'groups':group_rows,
        'input_space':task['input_space'],'input_feature_id':task['feature_id'],'mapped_rna_feature':regulator_name(plan,task),
        'unit_warning':'RNA and motif input units are distinct; no pooled network score',
        'weighted_mean_recipe':'weighted_signed_sum / weighted_finite_mass; null when mass=0',
        'unweighted_mean_recipe':'sum / count; null when count=0; failures/nonfinite retained'})
    nonfinite=0
    for channel,a in accum.items():
        fields={key:getattr(a,key) for key in ('sum','abs_sum','count','positive','negative','zero','nonfinite')}
        nonfinite+=int(a.nonfinite[0].sum())
        fields.update({'weighted_'+key:value for key,value in weighted[channel].items()})
        np.savez(writer.directory/(channel+'_edges.npz'),**fields)
        # Global count proves every target includes all sampled cells, including nonfinite entries.
        if not np.all(a.count[0]+a.nonfinite[0]==task['n_cells']):raise ValueError('Incomplete target denominator')
    writer.put('derivative_integrity.json',{'nonfinite_entries_global':nonfinite,'all_targets_retained':True})
    if nonfinite:raise ValueError('Nonfinite derivative entries retained in failed attempt; COMPLETE prohibited')
    result=base._result(task,authorization);result.update(input_space=task['input_space'],feature_id=task['feature_id'],
        tf_stable_id=task['tf_stable_id'],epsilon=task['epsilon'],matched_baseline_dependency=dependency,
        groups=len(groups),rna_targets=len(axes['vx']),motif_targets=len(axes['vy']),
        interpretation='Local one-sided full-model input sensitivity, not direct binding or causal GRN')
    return result

def epsilon_task(plan,frame,task,session,writer,authorization,dependency):
    arrays={c:np.lib.format.open_memmap(writer.directory/(c+'_derivative.npy'),mode='w+',dtype=np.float64,
             shape=(task['n_cells'],width)) for c,width in [('vx',session.model.x_dim),('vy',session.model.y_dim)]}
    nonfinite=0
    try:
        for block in checked_jacobians(plan,task,session):
            arrays[block['channel']][block['start']:block['stop'],block['target_start']:block['target_stop']]=block['values']
            nonfinite+=int((~np.isfinite(block['values'])).sum())
        for value in arrays.values():value.flush()
    finally:
        for value in arrays.values():value._mmap.close()
    writer.put('diagnostic_axes.json',{'cell_ids':cells_for(plan,frame,task),'rna_targets':list(session.population.genes),
        'motif_targets':[r['feature_id'] for r in plan['spec']['motifs']],'nonfinite_entries':nonfinite})
    if nonfinite:raise ValueError('Nonfinite epsilon derivatives retained; COMPLETE prohibited')
    result=base._result(task,authorization);result.update(feature_id=task['feature_id'],input_space=task['input_space'],
        epsilon=task['epsilon'],matched_baseline_dependency=dependency,selection='Frozen lexical IDs, no effect-dependent selection')
    return result

def epsilon_comparison(registry,guard=None):
    checkpoint(guard)
    coverage=registry.coverage()
    checkpoint(guard)
    if not coverage['all_expected_tasks_complete']:raise PermissionError('Complete diagnostic family required')
    rows=[]
    for task in registry.book['tasks']:
        checkpoint(guard)
        reference=[t for t in registry.book['tasks'] if all(t[k]==task[k] for k in ('seed','feature_id','input_space')) and t['epsilon']==1e-4]
        if len(reference)!=1:raise ValueError('Diagnostic reference must be unique')
        d,_=base.completed_directory(registry,task);ref,_=base.completed_directory(registry,reference[0])
        for channel in ('vx','vy'):
            a=np.load(d/(channel+'_derivative.npy'),mmap_mode='r',allow_pickle=False)
            b=np.load(ref/(channel+'_derivative.npy'),mmap_mode='r',allow_pickle=False)
            error2=ref2=0.;same=total=0
            if a.shape!=b.shape or a.dtype!=np.float64 or b.dtype!=np.float64:raise ValueError('Diagnostic arrays differ')
            for start in range(0,len(a),256):
                checkpoint(guard)
                x=a[start:start+256];y=b[start:start+256]
                if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite diagnostic payload')
                error2+=float(np.sum((x-y)**2,dtype=np.float64));ref2+=float(np.sum(y*y,dtype=np.float64))
                same+=int((np.sign(x)==np.sign(y)).sum());total+=x.size
            if not np.isfinite(error2+ref2):raise ValueError('Diagnostic aggregate overflow')
            rows.append({'seed':task['seed'],'feature_id':task['feature_id'],'input_space':task['input_space'],
                'epsilon':task['epsilon'],'reference_epsilon':1e-4,'channel':channel,'n_entries':total,
                'absolute_L2':float(np.sqrt(error2)),'reference_L2':float(np.sqrt(ref2)),
                'relative_L2':float(np.sqrt(error2/ref2)) if ref2>0 else None,'sign_agreement':same/total})
            a._mmap.close();b._mmap.close()
    return seal({'schema':'dynavelo_epsilon_comparison_v1','plan_sha256':registry.book['plan_sha256'],
        'book_sha256':registry.book['book_sha256'],'coverage_sha256':coverage['coverage_sha256'],'rows':rows,
        'automatic_stability_PASS':False,'epsilon_selected':None,'claim_scope':'engineering numerical sensitivity description only'},'summary_sha256')

def full_cell_task(plan,frame,task,session,writer,authorization,dependency):
    if task['mode'] not in {'rna-only','official-joint'}:raise ValueError('Unsupported full-cell mode')
    ids=cells_for(plan,frame,task);weights=frame_weights(plan,frame,task,session.indices)
    h=[session.population.gene_index[name] for name in plan['policy']['primary']['H']]
    scale=frame['global_time_windows'][str(task['seed'])]['common_frame_velocity_scale']
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid fixed RMS')
    values={k:np.full(len(ids),np.nan) for k in ('projection_H','delta_time_raw','delta_vx_norm','delta_vy_norm','delta_vz_norm')}
    expected=iter(session.batches)
    for block in session.perturbations([task['feature_id']],mode=task['mode']):
        start,stop=block['start'],block['stop']
        if (next(expected,None)!=(start,stop) or block['regulator']!=task['feature_id'] or block['mode']!=task['mode']
            or block['motif_changed']!=(task['mode']=='official-joint')
            or not np.array_equal(block['population_indices'],session.indices[start:stop])):raise ValueError('Full-cell block/mode mismatch')
        baseline=np.asarray(session.baseline['vx'][start:stop],dtype=np.float64)[:,h];norm=np.linalg.norm(baseline,axis=1)
        if not np.isfinite(norm).all() or np.any(norm<1e-8*scale) or np.any(norm<=0):raise ValueError('No candidate-specific direction remasking')
        values['projection_H'][start:stop]=np.sum(block['delta']['vx'][:,h]*(baseline/norm[:,None]),axis=1)/scale
        values['delta_time_raw'][start:stop]=block['delta']['time']
        for c in ('vx','vy','vz'):values['delta_'+c+'_norm'][start:stop]=np.linalg.norm(block['delta'][c],axis=1)
    if next(expected,None) is not None or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Incomplete/nonfinite full-cell effects')
    result=base._result(task,authorization);result.update(base.summarize_effects(plan,frame,task,values,weights,session.indices))
    result.update(feature_id=task['feature_id'],tf_stable_id=task['tf_stable_id'],matched_baseline_dependency=dependency,
        H=plan['policy']['primary']['H'],full_common_frame_RMS=scale,shortlist_sha256=task['shortlist_sha256'],
        interpretation='Locked top20 follow-up only; no full-TF full-cell ranking or rescue claim')
    stream=io.StringIO(newline='');table=csv.writer(stream,delimiter='\t')
    table.writerow(['cell_id','population_index','label','cell_weight','full_original_window']+list(values))
    windows=frame['global_time_windows'][str(task['seed'])]['window_index']
    for i,position in enumerate(session.indices):
        table.writerow([ids[i],int(position),plan['spec']['cells'][position]['label'],float(weights[i]),windows[position]]+[float(v[i]) for v in values.values()])
    writer.put('cell_effects.tsv',stream.getvalue().encode('utf-8'));return result

def dependency_registry(plan,frame,shortlist,output,operation):
    if operation=='epsilon-baseline':return None
    if operation=='epsilon':
        return FollowupRegistry(output/'epsilon-baseline',followup_book(plan,frame,shortlist,'epsilon-baseline'),plan=plan,frame=frame,shortlist=shortlist)
    return TaskRegistry(output/'matched-baseline',task_book(plan,'matched-baseline',frame),plan=plan,frame=frame)

def dependency_for(registry,task,verified=None):
    matches=[t for t in registry.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
    if len(matches)!=1:raise ValueError('Matched baseline task not unique')
    dep=matches[0]
    for key in ('checkpoint_sha256','sample_cell_axis_sha256','n_cells','chunk_geometry','batches','full_global_minima_sha256'):
        if canonical(dep[key])!=canonical(task[key]):raise ValueError('Matched baseline source/chunks differ')
    if verified is None:directory,saved=base.completed_directory(registry,dep)
    else:
        key=dep['task_id']
        if key not in verified:verified[key]=base.completed_directory(registry,dep)
        directory,saved=verified[key]
    record={'task_id':dep['task_id'],'commit_sha256':saved['manifest_sha256'],'book_sha256':registry.book['book_sha256']}
    return dep,directory,saved,record

def verify_completed_dependency(registry,deps,task,verified=None):
    directory,_=base.completed_directory(registry,task)
    _,_,_,record=dependency_for(deps,task,verified)
    if base.read_json(directory/'result.json').get('matched_baseline_dependency')!=record:
        raise ValueError('Completed follow-up dependency changed')

def applicability(plan,shortlist):
    rows=[{'seed':seed,'stable_id':r['stable_id'],'feature_id':r['feature_id'],
           'status':'NUMERICAL_TASK_REQUIRED' if r['has_selected_motif'] else 'NOT_APPLICABLE_NO_SELECTED_MOTIF',
           'effect_if_not_applicable':None} for seed in (0,1,2) for r in shortlist['selected']]
    return {'schema':'dynavelo_joint_applicability_v1','plan_sha256':plan['plan_sha256'],
        'shortlist_sha256':shortlist['shortlist_sha256'],'rows':rows,'shortlist_seed_denominator':len(rows),
        'applicable_numeric_tasks':sum(r['status']=='NUMERICAL_TASK_REQUIRED' for r in rows),
        'no_replacement_or_zero_imputation':True}

def publish_once(registry,name,value):
    data=canonical(value);path=registry.path(name)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Immutable derived manifest changed: '+name)
    else:registry.atomic_new(name,data)

def dispatch_followup(plan,frame,shortlist,screen_review_record,followup_review_record,root,operation,output_root,*,
                      trusted_reviewers=(),max_tasks=None,shard_index=None,seed=None,stop_file=None,
                      fixture_population=None,fixture_model_loader=None):
    root=base.checked_root(root);output=base.validate_output_namespace(plan,root,output_root)
    # Begin cooperative accounting before source/review/registry hash work.
    guard=base.ResourceGuard(root,plan['spec']['dispatch_contract'],stop_file);guard.check()
    auth,contract=authorize_followup(plan,frame,shortlist,screen_review_record,followup_review_record,root,operation,output,trusted_reviewers,guard)
    fixture=plan['spec']['purpose']=='engineering_fixture'
    if not fixture and (fixture_population is not None or fixture_model_loader is not None):raise PermissionError('No injected human population/model')
    guard.check();maximum=contract['resource_limits']['max_tasks_per_invocation']
    if max_tasks is None:max_tasks=maximum
    if type(max_tasks) is not int or not 1<=max_tasks<=maximum:raise ValueError('Task ceiling exceeds reviewed budget')
    if seed is not None and (type(seed) is not int or seed not in (0,1,2)):raise ValueError('Invalid seed')
    if shard_index is not None and (type(shard_index) is not int or not 0<=shard_index<plan['spec']['chunk_geometry'].get('regulator_shards',1)):
        raise ValueError('Invalid frozen shard')
    population=fixture_population if fixture_population is not None else base.load_population(plan,root,contract)
    base.verify_population(plan,population);torch.set_num_threads(contract['resource_limits']['cpu_threads'])
    registry=FollowupRegistry(output/operation,followup_book(plan,frame,shortlist,operation),plan=plan,frame=frame,shortlist=shortlist)
    deps=dependency_registry(plan,frame,shortlist,output,operation)
    if operation=='full-cell-joint':publish_once(registry,'applicability.json',applicability(plan,shortlist))
    selected=[t for t in registry.book['tasks'] if (seed is None or t['seed']==seed) and (shard_index is None or t['assigned_shard']==shard_index)]
    model=None;current_seed=None;checked_noop=set();attempted=0;status='INVOCATION_FINISHED_NOT_A_SCIENTIFIC_PASS';error=None
    def recheck():
        guard.check()
        authorize_followup(plan,frame,shortlist,screen_review_record,followup_review_record,root,operation,output,trusted_reviewers,guard)
        base.verify_population(plan,population);guard.check()
    with guard.signals():
        try:
            for task in selected:
                guard.check()
                if registry.resume(task['task_id'])['status']=='COMPLETE':
                    if deps is not None:verify_completed_dependency(registry,deps,task)
                    continue
                if attempted>=max_tasks:status='STOPPED_AT_FROZEN_TASK_LIMIT';break
                with registry.locked(task['task_id']):
                    if registry.resume(task['task_id'])['status']=='COMPLETE':
                        if deps is not None:verify_completed_dependency(registry,deps,task)
                        continue
                    recheck();writer=base.FileTaskWriter(registry,task['task_id']);attempted+=1
                    try:
                        if current_seed!=task['seed']:
                            model=None
                            if contract['device']=='cuda:0':torch.cuda.empty_cache()
                            record=next(r for r in plan['spec']['models'] if r['seed']==task['seed'])
                            model=(fixture_model_loader or base.load_model)(root,plan,record,population,contract);current_seed=task['seed']
                        model.eval();model.mode='evaluation-fixed'
                        if str(model.device)!=contract['device'] or any(str(p.device)!=contract['device'] for p in model.parameters()):raise ValueError('Wrong model device')
                        ids=cells_for(plan,frame,task);positions={c:i for i,c in enumerate(population.cell_ids)};indices=[positions[c] for c in ids]
                        if identity(ids)!=task['sample_cell_axis_sha256'] or len(ids)!=task['n_cells']:raise ValueError('Cell axis mismatch')
                        if canonical(task['batches'])!=canonical(batches_without_singleton(len(ids),task['chunk_geometry']['cells'])):raise ValueError('Core chunk mismatch')
                        if operation=='epsilon-baseline':
                            required=len(ids)*(model.x_dim+model.y_dim+model.zx_dim+model.zy_dim+1)*4+1024**2
                        elif operation=='epsilon':required=len(ids)*(model.x_dim+model.y_dim)*8+1024**2
                        elif operation in {'grn','full-cell-grn'}:
                            ng=1+len(set(c['label'] for c in plan['spec']['cells']))+len(plan['policy']['branches'])+5
                            required=ng*(model.x_dim+model.y_dim)*11*8+1024**2
                        else:required=len(ids)*2048+1024**2
                        if required>contract['resource_limits']['task_payload_bytes']:raise base.ResourceStop('Projected task payload ceiling')
                        if operation=='epsilon-baseline':
                            session=base.GuardedSession(model,population,indices,task['chunk_geometry']['cells'],writer.directory/'baseline',checkpoint_sha256=task['checkpoint_sha256'])
                            session.guard=guard;result=base.baseline_task(plan,frame,task,session,writer,auth)
                        else:
                            dep,directory,saved,dependency=dependency_for(deps,task)
                            session=base.attach_baseline(model,population,indices,task,directory/'baseline',guard)
                            cache_key=(task['seed'],task['sample_id'],saved['manifest_sha256'],session.fingerprint)
                            if cache_key not in checked_noop:
                                for block in session.perturbations([population.genes[0]],mode='no-op'):
                                    if any(np.count_nonzero(v) for v in block['delta'].values()):raise ValueError('Restored matched no-op is not exact zero')
                                checked_noop.add(cache_key)
                            function=epsilon_task if operation=='epsilon' else grn_task if operation in {'grn','full-cell-grn'} else full_cell_task
                            result=function(plan,frame,task,session,writer,auth,dependency)
                            if deps.resume(dep['task_id'])!=saved:raise ValueError('Matched cache changed during task')
                            result['restored_matched_noop']='EXACT_ZERO_ALL_CHANNELS_THIS_INVOCATION'
                        session._check_ready();recheck()
                        result.update(resources=guard.snapshot(),followup_dispatcher_sha256=sha256(__file__),engineering_fixture=fixture)
                        writer.put('result.json',result);writer.finish('COMPLETE',contract['resource_limits']['task_payload_bytes']);del session
                    except (base.StopRequested,KeyboardInterrupt):
                        if not (writer.directory/'COMMIT.json').exists():writer.put('INTERRUPTED.json',{'status':'INCOMPLETE','time':base.utc(),'resources':guard.snapshot()})
                        raise
                    except Exception as exc:
                        if not (writer.directory/'COMMIT.json').exists():
                            writer.put('FAILURE.json',{'error_type':type(exc).__name__,'error':str(exc),'time':base.utc(),'resources':guard.snapshot()})
                            writer.finish('FAILED',float('inf'))
                        raise
        except (base.StopRequested,KeyboardInterrupt) as exc:status='INTERRUPTED_INCOMPLETE_TASK_RETAINED';error=str(exc)
        except Exception as exc:status='STOPPED_WITH_FAILURE_ALL_TASKS_RETAINED';error=type(exc).__name__+': '+str(exc)
    result={'schema':'dynavelo_followup_invocation_v1','operation':operation,'status':status,'error':error,'tasks_attempted':attempted,
        'plan_sha256':plan['plan_sha256'],'frame_sha256':frame['frame_sha256'],'shortlist_sha256':shortlist['shortlist_sha256'],
        'resources':guard.snapshot(),'coverage':registry.coverage(),'engineering_fixture':fixture,'scientific_or_external_PASS':False}
    if operation=='epsilon' and result['coverage']['all_expected_tasks_complete']:
        summary=epsilon_comparison(registry,guard);publish_once(registry,'epsilon_comparison.json',summary)
        result['epsilon_summary_sha256']=summary['summary_sha256']
    registry.atomic_new('invocations/'+uuid.uuid4().hex+'.json',canonical(result));return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','frame','shortlist','screen-review-record','followup-review-record','output-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--operation',choices=sorted(OPERATIONS),required=True);p.add_argument('--trusted-reviewer',action='append',required=True)
    p.add_argument('--max-tasks',type=int);p.add_argument('--seed',type=int,choices=[0,1,2]);p.add_argument('--shard-index',type=int);p.add_argument('--stop-file')
    a=p.parse_args();root=base.checked_root(a.root)
    def local(value):
        path=Path(value)
        if not path.is_absolute():raise ValueError('Absolute project path required')
        return safe_target(root,path.relative_to(root).as_posix())
    result=dispatch_followup(base.read_json(local(a.plan)),base.read_json(local(a.frame)),base.read_json(local(a.shortlist)),
        base.read_json(local(a.screen_review_record)),base.read_json(local(a.followup_review_record)),root,a.operation,local(a.output_root),
        trusted_reviewers=a.trusted_reviewer,max_tasks=a.max_tasks,seed=a.seed,shard_index=a.shard_index,stop_file=local(a.stop_file) if a.stop_file else None)
    print(json.dumps({k:v for k,v in result.items() if k!='coverage'},ensure_ascii=False));return 1 if result['error'] else 0

if __name__=='__main__':raise SystemExit(main())
