"""Scoped follow-up, unchanged numerical core/review gates, no approvals.

Epsilon uses original full-context lexical axes; effects use target-common IDs.
Cross-VM history migration is unsupported.
"""
from pathlib import Path
import json,uuid,argparse
import numpy as np
import torch
import screen_dispatch as base
import screen_followup_dispatch as old
import screen_scope_dispatch_v2 as primary
from screen_plan import canonical,identity,seal,check_seal,verify_artifact
from screen_task_registry import TaskRegistry,batches
from screen_scope_plan_runtime_v2 import validate_scope_plan,validate_scope_frame,scope_task_book
from screen_scope_execution_contract_v2 import authorize_scope_stage,validate_namespace
from screen_scope_registry_v2 import ScopeRegistry,ScopeTaskWriter,stage_root,validate_result
from screen_scope_shortlist_v2 import validate_shortlist
from human_return_common import sha256,safe_target

OPERATIONS={'epsilon-baseline','epsilon','grn','full-cell-rna','full-cell-joint','full-cell-grn'}
EPSILONS=(1e-5,1e-4,1e-3)
DIR={'epsilon-baseline':'eb','epsilon':'e','grn':'g','full-cell-rna':'fr','full-cell-joint':'fj','full-cell-grn':'fg'}


def followup_contract(plan):
    context=plan['context_plan'];contract=context['spec'].get('followup_contract',{})
    if (contract.get('schema')!='dynavelo_scope_followup_contract_v2'
        or contract.get('selection_rule')!='lexical_original_cell_and_stable_regulator_IDs'
        or contract.get('epsilons')!=list(EPSILONS)
        or contract.get('epsilon_population')!='full_trained_context_lexical_not_effect_eligibility'
        or contract.get('effect_population')!='fixed_target_intersect_full_context_common_frame'):
        raise ValueError('Freeze unchanged epsilon policy and explicit scopes before effects')
    allowed=contract.get('allowed_operations')
    if not isinstance(allowed,list) or not allowed or len(set(allowed))!=len(allowed) or not set(allowed)<=OPERATIONS:
        raise ValueError('Explicit unique operation family required')
    old.diagnostic_axes(context)
    return contract


def followup_book(plan,frame,shortlist,operation):
    validate_scope_plan(plan);validate_scope_frame(plan,frame);contract=followup_contract(plan)
    if operation not in contract['allowed_operations']:raise PermissionError('Operation not prospectively frozen')
    if frame['status']!='READY_FOR_SEPARATE_SCOPE_FRAME_REVIEW':raise PermissionError('Held effect frame')
    check_seal(shortlist,'shortlist_sha256')
    if shortlist.get('scope_plan_sha256')!=plan['scope_plan_sha256'] or shortlist.get('scope_frame_sha256')!=frame['scope_frame_sha256']:
        raise ValueError('Shortlist belongs to different scope/frame')
    context=plan['context_plan'];geometry=context['spec']['chunk_geometry'];diag=old.diagnostic_axes(context)
    if operation.startswith('epsilon'):
        sample='epsilon_fixed_context';ids=diag['cell_ids'];weights=None
        regulators=[{**r,'input_space':'rna'} for r in diag['rna']]+[old.motif_regulator(r) for r in diag['motif']]
        if operation=='epsilon-baseline':regulators=[{'feature_id':'__baseline__','stable_id':'__baseline__','input_space':'none'}]
    elif operation=='grn':
        sample='primary';rows=frame['samples'][sample]['cells'];ids=[r['cell_id'] for r in rows];weights=[r['cell_weight'] for r in rows]
        regulators=[{**r,'input_space':'rna'} for r in context['tf_universe']]+[old.motif_regulator(r) for r in context['spec']['motifs']]
    else:
        sample='full_effect_population';ids=frame['effect_eligible_cell_ids'];weights=[1/len(ids)]*len(ids)
        regulators=[{**r,'input_space':'rna'} for r in shortlist['selected']]
        if operation=='full-cell-joint':regulators=[r for r in regulators if r['has_selected_motif']]
        if operation=='full-cell-grn':
            selected={r['feature_id'] for r in shortlist['selected']}
            regulators += [old.motif_regulator(r) for r in context['spec']['motifs'] if r['tf_rna_feature_id'] in selected]
    mode={'epsilon-baseline':'evaluation-fixed','epsilon':'finite-difference','grn':'finite-difference',
          'full-cell-rna':'rna-only','full-cell-joint':'official-joint','full-cell-grn':'finite-difference'}[operation]
    eps=EPSILONS if operation=='epsilon' else (1e-4,) if operation in {'grn','full-cell-grn'} else (None,)
    tasks=[];nshards=geometry.get('regulator_shards',1)
    if type(nshards) is not int or nshards<1:raise ValueError('Invalid shard count')
    for model in sorted(context['spec']['models'],key=lambda r:r['seed']):
        for reg in regulators:
            for epsilon in eps:
                body={'schema':'dynavelo_scope_followup_task_v2','scope_plan_sha256':plan['scope_plan_sha256'],
                    'scope_sha256':plan['scope_sha256'],'context_plan_sha256':context['plan_sha256'],
                    'scope_frame_sha256':frame['scope_frame_sha256'],'full_context_frame_sha256':frame['full_context_frame_sha256'],
                    'calibration_sha256':frame['calibration_sha256'],'context_invariants_sha256':identity(plan['context_invariants']),
                    'checkpoint_sha256':model['checkpoint']['sha256'],'seed':model['seed'],'stage':operation,'sample_id':sample,
                    'sample_cell_ids':ids,'sample_cell_axis_sha256':identity(ids),'sample_weights_sha256':identity(weights),
                    'n_cells':len(ids),'tf_stable_id':reg['stable_id'],'feature_id':reg['feature_id'],'input_space':reg['input_space'],
                    'mode':mode,'epsilon':epsilon,'chunk_geometry':geometry,'batches':batches(len(ids),geometry['cells']),
                    'shortlist_sha256':shortlist['shortlist_sha256'],'followup_contract_sha256':identity(contract)}
                task=seal(body,'task_id');task['assigned_shard']=int(task['task_id'][:16],16)%nshards;tasks.append(task)
    return seal({'schema':'dynavelo_scope_followup_book_v2','scope_plan_sha256':plan['scope_plan_sha256'],
        'scope_sha256':plan['scope_sha256'],'scope_frame_sha256':frame['scope_frame_sha256'],'stage':operation,
        'shortlist_sha256':shortlist['shortlist_sha256'],'tasks':tasks,'expected_tasks':len(tasks),
        'full_tf_denominator':len(context['tf_universe']),'shortlist_denominator':len(shortlist['selected']),
        'technical_attempt_limit':3,'coverage_contract':'Complete predetermined family; failures and joint nonapplicability retained'},'book_sha256')


class FollowupRegistry(ScopeRegistry):
    def __init__(self,root,book,*,plan,frame,shortlist):
        if canonical(book)!=canonical(followup_book(plan,frame,shortlist,book['stage'])):
            raise ValueError('Follow-up book differs from complete predetermined family')
        self.root=base.checked_root(root);self.plan=plan;self.frame=frame;self.shortlist=shortlist
        self.book=book;self.tasks={t['task_id']:t for t in book['tasks']}
        self._matched=None;self._parent_cache=None;self._audit_checkpoint=None
        execution_root=base.checked_root(plan['context_plan']['spec']['scope_execution_contract']['registry_execution_root'])
        expected='results/screen_runs_v2/'+plan['scope_plan_sha256']+'/'+DIR[book['stage']]
        if self.root!=safe_target(execution_root,expected):raise ValueError('Unique follow-up namespace required')
        self.root.mkdir(parents=True,exist_ok=True);path=self.path('task_book.json');data=canonical(book)
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Immutable follow-up book differs')
        else:self.atomic_new('task_book.json',data)

    def dependency(self,task):
        if self.book['stage']=='epsilon-baseline':return None
        if self._matched is None:
            if self.book['stage']=='epsilon':
                self._matched=FollowupRegistry(self.root.parent/DIR['epsilon-baseline'],
                    followup_book(self.plan,self.frame,self.shortlist,'epsilon-baseline'),plan=self.plan,frame=self.frame,shortlist=self.shortlist)
            else:self._matched=ScopeRegistry(stage_root(self.root.parent,'matched-baseline'),
                    scope_task_book(self.plan,'matched-baseline',self.frame),plan=self.plan,frame=self.frame)
        matches=[t for t in self._matched.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
        if len(matches)!=1:raise ValueError('Exact matched parent not unique')
        dep=matches[0]
        for key in ('checkpoint_sha256','sample_cell_ids','sample_cell_axis_sha256','sample_weights_sha256','n_cells',
                    'chunk_geometry','batches','context_invariants_sha256','calibration_sha256','full_context_frame_sha256'):
            if canonical(dep[key])!=canonical(task[key]):raise ValueError('Dependency geometry/context/weights differ: '+key)
        key=dep['task_id']
        if self._parent_cache is not None and key in self._parent_cache:saved=self._parent_cache[key]
        else:
            saved=self._matched.resume(key)
            if self._parent_cache is not None:self._parent_cache[key]=saved
        if saved['status']!='COMPLETE':raise PermissionError('Complete exact matched baseline required')
        directory=self._matched.path(saved['manifest_relative_path']).parent
        return dep,directory,saved,{'task_id':key,'book_sha256':self._matched.book['book_sha256'],'commit_sha256':saved['manifest_sha256']}

    def resume(self,task_id):
        if self._audit_checkpoint:self._audit_checkpoint()
        saved=TaskRegistry.resume(self,task_id)
        if saved['status']=='COMPLETE':
            task=self.tasks[task_id];directory=self.path(saved['manifest_relative_path']).parent
            value=validate_result(task,base.read_json(directory/'result.json'))
            for key in ('shortlist_sha256','followup_contract_sha256'):
                if value.get(key)!=task[key]:raise ValueError('Follow-up result identity differs')
            if task['mode'] in {'rna-only','official-joint'}:
                if type(value.get('primary_effect')) not in {int,float} or not np.isfinite(value['primary_effect']):raise ValueError('Finite effect required')
            if task['stage']!='epsilon-baseline' and value.get('matched_baseline_dependency')!=self.dependency(task)[3]:
                raise ValueError('Completed follow-up dependency differs')
        return saved


def authorize_followup(plan,frame,shortlist,screen_review,review_record,root,operation,output,trusted,guard):
    guard.check();validate_namespace(plan,root,output);contract=followup_contract(plan)
    if operation not in contract['allowed_operations']:raise PermissionError('Operation not frozen')
    auth=authorize_scope_stage(plan,review_record,root,'screen',trusted_reviewers=trusted,frame=frame,guard=guard)
    for key,path in [('scope_followup_module',__file__),('scope_followup_numerics_module',str(Path(__file__).with_name('screen_scope_followup_numerics_v2.py'))),('legacy_followup_module',old.__file__)]:
        guard.check();artifact=verify_artifact(root,plan['context_plan']['spec']['input_artifacts'][key])
        if sha256(artifact)!=sha256(path):raise ValueError('Actually imported follow-up source not frozen')
    full=ScopeRegistry(stage_root(output,'baseline'),scope_task_book(plan,'baseline'),plan=plan)
    if canonical(primary.full_frame_from_registry(plan,full,root))!=canonical(frame):raise ValueError('Actual full baseline/frame differs')
    screen=ScopeRegistry(stage_root(output,'screen'),scope_task_book(plan,'screen',frame),plan=plan,frame=frame)
    # One bounded dependency audit also covers ranking replay; no per-TF full-parent rehash.
    from screen_scope_shortlist_v2 import derive_shortlist
    authorize_scope_stage(plan,screen_review,root,'screen',trusted_reviewers=trusted,frame=frame,guard=guard)
    with screen.dependency_audit(guard.check):
        derived=derive_shortlist(plan,frame,screen,guard.check)
        if canonical(shortlist)!=canonical(derived):raise ValueError('Shortlist differs from full scope family')
    locked=screen.path('locked_scope_shortlist.json')
    if not locked.is_file() or locked.read_bytes()!=canonical(shortlist):raise ValueError('Actual locked shortlist required')
    review=base.read_json(verify_artifact(root,review_record));approval=review.get('followup',{})
    expected={'contract_sha256':identity(contract),'screen_coverage_sha256':shortlist['coverage_sha256'],
              'shortlist_sha256':shortlist['shortlist_sha256']}
    if any(approval.get(k)!=v for k,v in expected.items()) or operation not in approval.get('allowed_operations',[]):
        raise PermissionError('Follow-up review does not bind exact scope/family/operation')
    stability=base.read_json(verify_artifact(root,approval['stability_review']))
    expected_stability={'schema':'dynavelo_scope_stability_review_v2','decision':'APPROVED_FOR_EXPLICIT_EXPLORATORY_SCOPE',
        'scope_plan_sha256':plan['scope_plan_sha256'],'scope_sha256':plan['scope_sha256'],
        'scope_frame_sha256':frame['scope_frame_sha256'],'full_context_frame_sha256':frame['full_context_frame_sha256'],
        'calibration_sha256':frame['calibration_sha256'],'screen_coverage_sha256':shortlist['coverage_sha256'],
        'shortlist_sha256':shortlist['shortlist_sha256'],'independent_of_plan_preparer':True,
        'claim_scope':'exploratory_model_input_sensitivity_only','external_confirmation_or_causal_rescue_authorized':False}
    if (any(type(stability.get(k))!=type(v) or stability.get(k)!=v for k,v in expected_stability.items())
        or stability.get('reviewer_id') not in trusted or stability.get('reviewer_id')==plan['context_plan']['spec']['prepared_by']
        or operation not in stability.get('allowed_operations',[])):
        raise PermissionError('Independent existing stability review absent or wrong scope')
    if operation in {'grn','full-cell-grn'}:
        from screen_scope_followup_numerics_v2 import epsilon_comparison
        registry=FollowupRegistry(output/DIR['epsilon'],followup_book(plan,frame,shortlist,'epsilon'),plan=plan,frame=frame,shortlist=shortlist)
        summary=epsilon_comparison(registry,guard)
        if approval.get('epsilon_coverage_sha256')!=summary['coverage_sha256']:raise PermissionError('Entire epsilon family not reviewed')
        evidence=base.read_json(verify_artifact(root,approval['epsilon_comparison']))
        if canonical(summary)!=canonical(evidence):raise PermissionError('Reviewed epsilon summary differs')
    guard.check();auth.update(shortlist_sha256=shortlist['shortlist_sha256'],screen_coverage_sha256=shortlist['coverage_sha256'])
    return auth


def epsilon_envelope(plan,frame,book,task,manifest_sha):
    return seal({'schema':'dynavelo_scope_epsilon_cache_v2','scope_plan_sha256':plan['scope_plan_sha256'],
        'scope_sha256':plan['scope_sha256'],'scope_frame_sha256':frame['scope_frame_sha256'],
        'calibration_sha256':frame['calibration_sha256'],'book_sha256':book['book_sha256'],
        'task_id':task['task_id'],'sample_cell_axis_sha256':task['sample_cell_axis_sha256'],
        'shortlist_sha256':task['shortlist_sha256'],'followup_contract_sha256':task['followup_contract_sha256'],
        'core_manifest_sha256':manifest_sha,'selection':'full trained context lexical diagnostic IDs; no effect eligibility'},'cache_sha256')


def baseline_task(plan,frame,task,session,writer,auth,registry):
    session.build_baseline();session._check_ready()
    for block in session.perturbations([session.population.genes[0]],mode='no-op'):
        if any(np.count_nonzero(v) for v in block['delta'].values()):raise ValueError('Diagnostic no-op differs')
    envelope=epsilon_envelope(plan,frame,registry.book,task,sha256(session.directory/'baseline_manifest.json'))
    writer.put('baseline/scope.json',envelope)
    value=primary.result(task,auth);value.update(no_op='EXACT_ZERO_ALL_CHANNELS_ALL_CHUNKS',
        baseline_manifest_sha256=envelope['core_manifest_sha256'],diagnostic_population='full context lexical, no positive-direction gate')
    return value


def restore_dependency(plan,frame,registry,task,model,population,indices,guard):
    dep,directory,saved,record=registry.dependency(task)
    if task['stage']!='epsilon':
        session,actual=primary.restore_matched(plan,frame,registry._matched,dep,model,population,indices,guard)
        if actual!=record:raise ValueError('Matched dependency changed')
    else:
        expected=epsilon_envelope(plan,frame,registry._matched.book,dep,sha256(directory/'baseline/baseline_manifest.json'))
        if canonical(base.read_json(directory/'baseline/scope.json'))!=canonical(expected):raise ValueError('Diagnostic cache scope differs')
        session=base.attach_baseline(model,population,indices,dep,directory/'baseline',guard)
        for block in session.perturbations([population.genes[0]],mode='no-op'):
            if any(np.count_nonzero(v) for v in block['delta'].values()):raise ValueError('Restored diagnostic no-op differs')
    return session,record


def applicability(plan,frame,shortlist):
    value=old.applicability(plan['context_plan'],shortlist)
    value.update(schema='dynavelo_scope_joint_applicability_v2',scope_plan_sha256=plan['scope_plan_sha256'],
        scope_sha256=plan['scope_sha256'],scope_frame_sha256=frame['scope_frame_sha256'],
        full_context_frame_sha256=frame['full_context_frame_sha256'],calibration_sha256=frame['calibration_sha256'],
        effect_population_n=frame['effect_population_n'])
    return value


def dispatch_followup(plan,frame,shortlist,screen_review_record,followup_review_record,root,operation,output_root,*,
        trusted_reviewers=(),max_tasks=None,shard_index=None,seed=None,stop_file=None,
        fixture_population=None,fixture_model_loader=None):
    from screen_scope_followup_numerics_v2 import epsilon_task,grn_task,full_cell_task,epsilon_comparison
    root=base.checked_root(root);output=validate_namespace(plan,root,output_root);context=plan['context_plan']
    guard=base.ResourceGuard(root,context['spec']['dispatch_contract'],stop_file);guard.check()
    with guard.signals():
        auth=authorize_followup(plan,frame,shortlist,screen_review_record,followup_review_record,root,operation,output,trusted_reviewers,guard)
    contract=auth['dispatch_contract'];fixture=plan['purpose']=='engineering_fixture'
    if not fixture and (fixture_population is not None or fixture_model_loader is not None):raise PermissionError('Human model/population injection forbidden')
    maximum=contract['resource_limits']['max_tasks_per_invocation'];max_tasks=maximum if max_tasks is None else max_tasks
    if type(max_tasks) is not int or not 1<=max_tasks<=maximum:raise ValueError('Reviewed invocation task ceiling exceeded')
    if seed is not None and (type(seed) is not int or seed not in (0,1,2)):raise ValueError('Invalid seed')
    shards=context['spec']['chunk_geometry'].get('regulator_shards',1)
    if shard_index is not None and (type(shard_index) is not int or not 0<=shard_index<shards):raise ValueError('Invalid frozen shard')
    guard.check();population=fixture_population if fixture_population is not None else base.load_population(context,root,contract)
    base.verify_population(context,population);torch.set_num_threads(contract['resource_limits']['cpu_threads'])
    registry=FollowupRegistry(output/DIR[operation],followup_book(plan,frame,shortlist,operation),plan=plan,frame=frame,shortlist=shortlist)
    if operation=='full-cell-joint':old.publish_once(registry,'applicability.json',applicability(plan,frame,shortlist))
    tasks=[t for t in registry.book['tasks'] if (seed is None or t['seed']==seed) and (shard_index is None or t['assigned_shard']==shard_index)]
    current_seed=None;model=None;attempted=0;error=None;status='INVOCATION_FINISHED_NOT_A_SCIENTIFIC_PASS'
    # Review/source rechecks retain the same evidence; full family replay occurs
    # once at entry and once at exit, never once per follow-up regulator.
    review=base.read_json(verify_artifact(root,followup_review_record))
    records=[screen_review_record,followup_review_record,review['followup']['stability_review']]
    if operation in {'grn','full-cell-grn'}:records.append(review['followup']['epsilon_comparison'])
    def recheck():
        guard.check()
        authorize_scope_stage(plan,followup_review_record,root,'screen',trusted_reviewers=trusted_reviewers,frame=frame,guard=guard)
        for record in records:guard.check();verify_artifact(root,record)
        for key,path in [('scope_followup_module',__file__),('scope_followup_numerics_module',str(Path(__file__).with_name('screen_scope_followup_numerics_v2.py'))),('legacy_followup_module',old.__file__)]:
            if context['spec']['input_artifacts'][key]['sha256']!=sha256(path):raise ValueError('Imported follow-up source drift')
        base.verify_population(context,population);guard.check()
    with guard.signals():
        try:
            with registry.dependency_audit(guard.check):
                for task in tasks:
                    guard.check()
                    if registry.resume(task['task_id'])['status']=='COMPLETE':continue
                    if attempted>=max_tasks:status='STOPPED_AT_FROZEN_TASK_LIMIT';break
                    registry.dependency(task) # Missing dependencies do not consume a new numerical attempt.
                    with registry.locked(task['task_id']):
                        if registry.resume(task['task_id'])['status']=='COMPLETE':continue
                        recheck();writer=ScopeTaskWriter(registry,task['task_id']);attempted+=1
                        try:
                            if current_seed!=task['seed']:
                                model=None
                                if contract['device']=='cuda:0':torch.cuda.empty_cache()
                                record=next(m for m in context['spec']['models'] if m['seed']==task['seed'])
                                model=(fixture_model_loader or base.load_model)(root,context,record,population,contract);current_seed=task['seed']
                            model.eval();model.mode='evaluation-fixed'
                            if str(model.device)!=contract['device'] or any(str(p.device)!=contract['device'] for p in model.parameters()):raise ValueError('Wrong model device')
                            ids=task['sample_cell_ids'];positions={c:i for i,c in enumerate(population.cell_ids)};indices=[positions[c] for c in ids]
                            if identity(ids)!=task['sample_cell_axis_sha256'] or len(ids)!=task['n_cells']:raise ValueError('Task axis changed')
                            if operation=='epsilon-baseline':required=len(ids)*(model.x_dim+model.y_dim+model.zx_dim+model.zy_dim+1)*4+1024**2
                            elif operation=='epsilon':required=len(ids)*(model.x_dim+model.y_dim)*8+1024**2
                            elif operation in {'grn','full-cell-grn'}:
                                ng=1+len(plan['scope']['target_labels'])+len(context['policy']['branches'])+5
                                required=ng*(model.x_dim+model.y_dim)*11*8+1024**2
                            else:required=len(ids)*2048+1024**2
                            if required>contract['resource_limits']['task_payload_bytes']:raise base.ResourceStop('Projected payload ceiling')
                            if operation=='epsilon-baseline':
                                session=base.GuardedSession(model,population,indices,task['chunk_geometry']['cells'],writer.directory/'baseline',checkpoint_sha256=task['checkpoint_sha256']);session.guard=guard
                                value=baseline_task(plan,frame,task,session,writer,auth,registry)
                            else:
                                session,dependency=restore_dependency(plan,frame,registry,task,model,population,indices,guard)
                                function=epsilon_task if operation=='epsilon' else grn_task if operation in {'grn','full-cell-grn'} else full_cell_task
                                value=function(plan,frame,task,session,writer,auth,dependency)
                            session._check_ready();recheck()
                            value.update(shortlist_sha256=shortlist['shortlist_sha256'],followup_contract_sha256=task['followup_contract_sha256'],
                                resources=guard.snapshot(),followup_dispatcher_sha256=sha256(__file__),engineering_fixture=fixture)
                            writer.put('result.json',value);writer.finish('COMPLETE',contract['resource_limits']['task_payload_bytes']);del session
                        except (base.StopRequested,KeyboardInterrupt):
                            if not (writer.directory/'COMMIT.json').exists():writer.put('INTERRUPTED.json',{'status':'INCOMPLETE','time':base.utc(),'resources':guard.snapshot()})
                            raise
                        except Exception as exc:
                            if not (writer.directory/'COMMIT.json').exists():
                                writer.put('FAILURE.json',{'error_type':type(exc).__name__,'error':str(exc),'time':base.utc(),'resources':guard.snapshot()})
                                writer.finish('FAILED',float('inf'))
                            raise
        except (Exception,KeyboardInterrupt) as exc:status='STOPPED_WITH_EVIDENCE';error={'type':type(exc).__name__,'detail':str(exc)}
    summary=None
    try:
        with guard.signals():
            coverage=registry.coverage(checkpoint=guard.check)
            authorize_followup(plan,frame,shortlist,screen_review_record,followup_review_record,root,operation,output,trusted_reviewers,guard)
            if operation=='epsilon' and coverage['all_expected_tasks_complete']:
                summary=epsilon_comparison(registry,guard);old.publish_once(registry,'epsilon_comparison.json',summary)
    except (base.StopRequested,base.ResourceStop) as exc:
        status='STOPPED_WITH_EVIDENCE';error={'type':type(exc).__name__,'detail':str(exc)}
        coverage={'status':'FULL_AUDIT_DEFERRED_RESOURCE_STOP','all_expected_tasks_complete':False,
            'expected_tasks':len(registry.tasks),'full_tf_denominator':registry.book['full_tf_denominator'],
            'book_sha256':registry.book['book_sha256'],'counts':None}
    answer={'schema':'dynavelo_scope_followup_invocation_v2','operation':operation,'status':status,'error':error,'tasks_attempted':attempted,
        'scope_plan_sha256':plan['scope_plan_sha256'],'scope_sha256':plan['scope_sha256'],'scope_frame_sha256':frame['scope_frame_sha256'],
        'full_context_frame_sha256':frame['full_context_frame_sha256'],'calibration_sha256':frame['calibration_sha256'],
        'shortlist_sha256':shortlist['shortlist_sha256'],'resources':guard.snapshot(),'coverage':coverage,
        'engineering_fixture':fixture,'scientific_or_external_PASS':False}
    if summary:answer['epsilon_summary_sha256']=summary['summary_sha256']
    registry.atomic_new('invocations/'+uuid.uuid4().hex+'.json',canonical(answer));return answer


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
    answer=dispatch_followup(base.read_json(local(a.plan)),base.read_json(local(a.frame)),base.read_json(local(a.shortlist)),
        base.read_json(local(a.screen_review_record)),base.read_json(local(a.followup_review_record)),root,a.operation,local(a.output_root),
        trusted_reviewers=a.trusted_reviewer,max_tasks=a.max_tasks,seed=a.seed,shard_index=a.shard_index,stop_file=local(a.stop_file) if a.stop_file else None)
    print(json.dumps({k:v for k,v in answer.items() if k!='coverage'},ensure_ascii=False));return 1 if answer['error'] else 0


if __name__=='__main__':raise SystemExit(main())
