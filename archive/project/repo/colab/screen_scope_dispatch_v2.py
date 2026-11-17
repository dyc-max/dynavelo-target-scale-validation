"""Scoped primary execution using unchanged streaming arithmetic and full inputs.

Follow-up and cross-VM recovery unsupported. No review/approval creation.
"""
from pathlib import Path
import argparse,csv,io,json,uuid
import numpy as np
import torch
import screen_dispatch as base
from screen_plan import canonical,identity
from screen_scope_plan_runtime_v2 import (scope_task_book,freeze_scope_frame,validate_scope_plan,
                                         seal_cache_envelope,validate_cache_envelope)
from screen_scope_execution_contract_v2 import authorize_scope_stage,validate_namespace
from screen_scope_registry_v2 import ScopeRegistry,ScopeTaskWriter,stage_root
from human_return_common import sha256,safe_target


def result(task,auth):
    return {'schema':'dynavelo_scope_task_result_v2','task_id':task['task_id'],'status':'COMPLETE',
            'numerical_integrity':'PASS','n_cells':task['n_cells'],'mode':task['mode'],
            **{k:task[k] for k in ('scope_plan_sha256','scope_sha256','scope_frame_sha256','context_plan_sha256',
                  'full_context_frame_sha256','calibration_sha256','context_invariants_sha256',
                  'sample_cell_axis_sha256','sample_weights_sha256','checkpoint_sha256','stage','seed','sample_id')},
            'review_sha256':auth['review_sha256'],'claim_status':'EXPLORATORY_MODEL_INPUT_SENSITIVITY_ONLY',
            'biological_or_external_PASS':False}


def full_frame_from_registry(plan,registry,root):
    summaries=[]
    for task in registry.book['tasks']:
        directory,_=base.completed_directory(registry,task)
        path=directory/'baseline_summary.json'
        summaries.append({**base.read_json(path),'artifact':base.record(root,path)})
    return freeze_scope_frame(plan,summaries)


def baseline_task(plan,frame,task,session,writer,auth,registry):
    session.build_baseline();session._check_ready()
    for block in session.perturbations([session.population.genes[0]],mode='no-op'):
        if any(np.count_nonzero(v) for v in block['delta'].values()):raise ValueError('No-op not exact zero')
    context=plan['context_plan'];h=[session.population.gene_index[g] for g in context['policy']['primary']['H']]
    norms=np.empty(len(session.indices),dtype=np.float64)
    for start,stop in session.batches:norms[start:stop]=np.linalg.norm(np.asarray(session.baseline['vx'][start:stop],dtype=np.float64)[:,h],axis=1)
    if frame is not None:
        scale=frame['full_context_frame']['global_time_windows'][str(task['seed'])]['common_frame_velocity_scale']
        if not np.isfinite(norms).all() or np.any(norms<=0) or np.any(norms<1e-8*scale):raise ValueError('Matched directions invalid; no local remasking')
    envelope=seal_cache_envelope(plan,registry.book,task['task_id'],sha256(session.directory/'baseline_manifest.json'),frame=frame)
    writer.put('baseline/scope.json',envelope)
    out=result(task,auth);out.update(no_op='EXACT_ZERO_ALL_CHANNELS_ALL_CHUNKS',
        baseline_manifest_sha256=envelope['core_manifest_sha256'],scope_cache_identity_sha256=envelope['cache_identity_sha256'],
        H=context['policy']['primary']['H'])
    if task['stage']=='baseline':
        writer.put('baseline_summary.json',{'schema':'dynavelo_full_baseline_summary_v1','seed':task['seed'],
            'plan_sha256':context['plan_sha256'],'checkpoint_sha256':task['checkpoint_sha256'],
            'cell_axis_sha256':context['axes_sha256']['cells'],'H':context['policy']['primary']['H'],
            'raw_time_mean':session.baseline['time'].astype(float).tolist(),'direction_norm_H':norms.tolist()})
    return out


def restore_matched(plan,frame,registry,task,model,population,indices,guard):
    directory,saved=base.completed_directory(registry,task)
    cache=base.read_json(directory/'baseline/scope.json')
    validate_cache_envelope(plan,registry.book,task['task_id'],cache,
                           sha256(directory/'baseline/baseline_manifest.json'),frame=frame)
    session=base.attach_baseline(model,population,indices,task,directory/'baseline',guard)
    for block in session.perturbations([population.genes[0]],mode='no-op'):
        if any(np.count_nonzero(v) for v in block['delta'].values()):raise ValueError('Restored matched no-op differs')
    return session,{'task_id':task['task_id'],'book_sha256':registry.book['book_sha256'],'commit_sha256':saved['manifest_sha256']}


def summarize(plan,frame,task,values,weights,indices):
    cells=plan['context_plan']['spec']['cells'];labels=np.array([cells[i]['label'] for i in indices])
    eligible=set(frame['effect_eligible_cell_ids']);full_indices=[i for i,c in enumerate(cells) if c['cell_id'] in eligible]
    windows=np.asarray(frame['full_context_frame']['global_time_windows'][str(task['seed'])]['window_index'])
    def domain(mask,n):
        mass=float(weights[mask].sum())
        return {'sample_n':int(mask.sum()),'effect_frame_domain_n':int(n),'sample_weight_sum':mass,
                'weighted_conditional_mean':{k:float(np.dot(weights[mask],v[mask])/mass) if mass>0 else None for k,v in values.items()},
                'estimator':'descriptive normalized sample-weight domain; no biological replicates or P value'}
    return {'primary_effect':float(np.dot(weights,values['projection_H'])),
        'weighted_effect_frame_secondary':{k:float(np.dot(weights,v)) for k,v in values.items() if k!='projection_H'},
        'by_label':{label:domain(labels==label,frame['effect_state_counts'][label]) for label in plan['scope']['target_labels']},
        'by_full_context_time_window':{str(w):domain(windows[indices]==w,sum(windows[i]==w for i in full_indices)) for w in range(5)},
        'effect_population_n':frame['effect_population_n'],'original_target_n':frame['original_target_n']}


def screen_task(plan,frame,task,session,writer,auth,dependency):
    sample=frame['samples'][task['sample_id']];ids=[r['cell_id'] for r in sample['cells']]
    if ids!=task['sample_cell_ids']:raise ValueError('Exact local sample differs')
    weights=np.asarray([r['cell_weight'] for r in sample['cells']],dtype=np.float64)
    if abs(weights.sum()-1)>1e-12:raise ValueError('Local weights do not sum to one')
    h=[session.population.gene_index[g] for g in plan['context_plan']['policy']['primary']['H']]
    scale=frame['full_context_frame']['global_time_windows'][str(task['seed'])]['common_frame_velocity_scale']
    if not np.isfinite(scale) or scale<=0:raise ValueError('Full-context RMS invalid')
    values={k:np.full(len(ids),np.nan) for k in ('projection_H','delta_time_raw','delta_vx_norm','delta_vy_norm','delta_vz_norm')}
    seen=np.zeros(len(ids),dtype=bool)
    for block in session.perturbations([task['feature_id']],mode='rna-only'):
        start,stop=block['start'],block['stop']
        if (seen[start:stop].any() or block['motif_changed'] or block['regulator']!=task['feature_id']
            or block['mode']!='rna-only' or not np.array_equal(block['population_indices'],session.indices[start:stop])):
            raise ValueError('Perturbation cell/mode coverage mismatch')
        baseline=np.asarray(session.baseline['vx'][start:stop],dtype=np.float64)[:,h];norm=np.linalg.norm(baseline,axis=1)
        if np.any(norm<=0) or np.any(norm<1e-8*scale):raise ValueError('Matched direction failed; cannot exclude per TF')
        values['projection_H'][start:stop]=np.sum(block['delta']['vx'][:,h]*(baseline/norm[:,None]),axis=1)/scale
        values['delta_time_raw'][start:stop]=block['delta']['time']
        for channel in ('vx','vy','vz'):values['delta_'+channel+'_norm'][start:stop]=np.linalg.norm(block['delta'][channel],axis=1)
        seen[start:stop]=True
    if not seen.all() or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Incomplete/nonfinite local effects')
    session._check_ready();out=result(task,auth);out.update(summarize(plan,frame,task,values,weights,session.indices))
    out.update(tf_stable_id=task['tf_stable_id'],feature_id=task['feature_id'],H=plan['context_plan']['policy']['primary']['H'],
               full_context_common_RMS=scale,matched_baseline_dependency=dependency,
               time='raw posterior mean difference; full-context baseline windows unchanged',effect='along matched model velocity, not rescue or causality')
    stream=io.StringIO(newline='');fields=['cell_id','population_index','label','cell_weight','full_context_time_window']+list(values)
    writer_csv=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer_csv.writeheader()
    windows=frame['full_context_frame']['global_time_windows'][str(task['seed'])]['window_index']
    for i,(cell,pos) in enumerate(zip(ids,session.indices)):
        writer_csv.writerow({'cell_id':cell,'population_index':int(pos),'label':sample['cells'][i]['label'],
            'cell_weight':float(weights[i]),'full_context_time_window':windows[pos],**{k:float(v[i]) for k,v in values.items()}})
    writer.put('cell_effects.tsv',stream.getvalue().encode());return out


def dispatch(plan,review_record,root,stage,output_root,*,frame=None,trusted_reviewers=(),
             seed=None,shard_index=None,max_tasks=None,stop_file=None,fixture_population=None,fixture_model_loader=None):
    root=base.checked_root(root);validate_scope_plan(plan);output=validate_namespace(plan,root,output_root)
    context=plan['context_plan'];guard=base.ResourceGuard(root,context['spec']['dispatch_contract'],stop_file);guard.check()
    auth=authorize_scope_stage(plan,review_record,root,stage,trusted_reviewers=trusted_reviewers,frame=frame,guard=guard)
    contract=auth['dispatch_contract'];fixture=plan['purpose']=='engineering_fixture'
    if not fixture and (fixture_population is not None or fixture_model_loader is not None):raise PermissionError('Human data/model injection forbidden')
    guard.check()
    maximum=contract['resource_limits']['max_tasks_per_invocation'];max_tasks=maximum if max_tasks is None else max_tasks
    if type(max_tasks) is not int or not 1<=max_tasks<=maximum:raise ValueError('Invocation task budget exceeded')
    if seed is not None and (type(seed) is not int or seed not in (0,1,2)):raise ValueError('Invalid seed')
    shards=context['spec']['chunk_geometry'].get('regulator_shards',1)
    if type(shards) is not int or shards<1:raise ValueError('Invalid shard count')
    if shard_index is not None and (type(shard_index) is not int or not 0<=shard_index<shards):raise ValueError('Invalid shard')
    # Science/source review has passed before any input matrix or model loader.
    population=fixture_population if fixture_population is not None else base.load_population(context,root,contract)
    base.verify_population(context,population);torch.set_num_threads(contract['resource_limits']['cpu_threads'])
    registry=ScopeRegistry(stage_root(output,stage),scope_task_book(plan,stage,frame),plan=plan,frame=frame)
    if stage!='baseline':
        full=ScopeRegistry(stage_root(output,'baseline'),scope_task_book(plan,'baseline'),plan=plan)
        if canonical(full_frame_from_registry(plan,full,root))!=canonical(frame):raise PermissionError('Scope frame differs from complete full-context registry')
    matched=ScopeRegistry(stage_root(output,'matched-baseline'),scope_task_book(plan,'matched-baseline',frame),plan=plan,frame=frame) if stage=='screen' else None
    tasks=[t for t in registry.book['tasks'] if (seed is None or t['seed']==seed) and (shard_index is None or t['assigned_shard']==shard_index)]
    attempted=0;current_seed=None;model=None;status='INVOCATION_FINISHED_NOT_SCIENTIFIC_PASS';error=None
    def recheck():
        guard.check()
        authorize_scope_stage(plan,review_record,root,stage,trusted_reviewers=trusted_reviewers,frame=frame,guard=guard)
        base.verify_population(context,population);guard.check()
    with guard.signals():
        try:
            with registry.dependency_audit(guard.check):
                for task in tasks:
                    guard.check()
                    if registry.resume(task['task_id'])['status']=='COMPLETE':continue
                    if attempted>=max_tasks:status='STOPPED_AT_FROZEN_TASK_LIMIT';break
                    dependency=None
                    if stage=='screen':
                        candidates=[t for t in matched.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
                        if len(candidates)!=1:raise ValueError('Matched task not unique')
                        dependency=candidates[0];base.completed_directory(matched,dependency)
                        for key in ('sample_cell_ids','sample_cell_axis_sha256','sample_weights_sha256','checkpoint_sha256',
                                    'chunk_geometry','batches','context_invariants_sha256','calibration_sha256'):
                            if dependency[key]!=task[key]:raise ValueError('Matched baseline geometry/scope differs')
                    with registry.locked(task['task_id']):
                        if registry.resume(task['task_id'])['status']=='COMPLETE':continue
                        recheck();writer=ScopeTaskWriter(registry,task['task_id']);attempted+=1
                        try:
                            if current_seed!=task['seed']:
                                record=next(m for m in context['spec']['models'] if m['seed']==task['seed'])
                                model=(fixture_model_loader or base.load_model)(root,context,record,population,contract)
                                current_seed=task['seed']
                            positions={c:i for i,c in enumerate(population.cell_ids)};indices=[positions[c] for c in task['sample_cell_ids']]
                            if identity(task['sample_cell_ids'])!=task['sample_cell_axis_sha256']:raise ValueError('Task axis changed')
                            if stage in ('baseline','matched-baseline'):
                                session=base.GuardedSession(model,population,indices,task['chunk_geometry']['cells'],writer.directory/'baseline',checkpoint_sha256=task['checkpoint_sha256']);session.guard=guard
                                value=baseline_task(plan,frame,task,session,writer,auth,registry)
                            else:
                                session,dep=restore_matched(plan,frame,matched,dependency,model,population,indices,guard)
                                value=screen_task(plan,frame,task,session,writer,auth,dep)
                            writer.put('result.json',value);recheck();writer.finish('COMPLETE',contract['resource_limits']['task_payload_bytes'])
                        except BaseException as exc:
                            if not (writer.directory/'COMMIT.json').exists():
                                writer.put('failure.json',{'error_type':type(exc).__name__,'detail':str(exc),'scientific_PASS':False})
                                writer.finish('FAILED',contract['resource_limits']['task_payload_bytes'])
                            raise
        except (Exception,KeyboardInterrupt) as exc:
            status='STOPPED_WITH_EVIDENCE';error={'type':type(exc).__name__,'detail':str(exc)}
    try:
        with guard.signals():coverage=registry.coverage(checkpoint=guard.check)
    except (base.StopRequested,base.ResourceStop) as exc:
        status='STOPPED_WITH_EVIDENCE';error={'type':type(exc).__name__,'detail':str(exc)}
        coverage={'status':'FULL_AUDIT_DEFERRED_RESOURCE_STOP','all_expected_tasks_complete':False,
                  'expected_tasks':len(registry.tasks),'full_tf_denominator':registry.book['full_tf_denominator'],
                  'book_sha256':registry.book['book_sha256'],'counts':None}
    answer={'schema':'dynavelo_scope_dispatch_invocation_v2','status':status,'error':error,'stage':stage,
            'scope_plan_sha256':plan['scope_plan_sha256'],'scope_frame_sha256':frame['scope_frame_sha256'] if frame else None,
            'tasks_attempted':attempted,'resources':guard.snapshot(),'coverage':coverage,'scientific_PASS':False}
    if stage=='baseline' and answer['coverage']['all_expected_tasks_complete']:
        derived=full_frame_from_registry(plan,registry,root);path=output/'scope_baseline_frame.json';data=canonical(derived)
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Existing full/local frame changed')
        else:_publish_frame(path,data)
        answer.update(scope_frame_path=str(path),scope_frame_status=derived['status'])
    invocation='invocations/'+uuid.uuid4().hex+'.json';registry.atomic_new(invocation,canonical(answer));return answer


def _publish_frame(path,data):
    # Same-directory fsync + exclusive hard-link publication, no overwrite.
    import os
    temporary=path.with_name('.scope_'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','review-record','stage','output-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--frame');parser.add_argument('--trusted-reviewer',action='append',default=[])
    parser.add_argument('--max-tasks',type=int);parser.add_argument('--seed',type=int);parser.add_argument('--shard-index',type=int)
    parser.add_argument('--stop-file');args=parser.parse_args();root=base.checked_root(args.root)
    def local(path):
        p=Path(path).absolute();return safe_target(root,p.relative_to(root).as_posix())
    value=dispatch(base.read_json(local(args.plan)),base.read_json(local(args.review_record)),root,args.stage,local(args.output_root),
        frame=base.read_json(local(args.frame)) if args.frame else None,trusted_reviewers=args.trusted_reviewer,
        seed=args.seed,shard_index=args.shard_index,max_tasks=args.max_tasks,stop_file=local(args.stop_file) if args.stop_file else None)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
