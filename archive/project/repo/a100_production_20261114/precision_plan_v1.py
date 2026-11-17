"""Separate precision-amended task family; original FP32 artifacts stay frozen."""
from pathlib import Path
import copy,json,sys
ROOT=Path(r'D:\dyc_7\DynaVelo_Research')
sys.path.insert(0,str(ROOT/'repo/colab'))
from screen_plan import canonical,identity,seal,check_seal
from screen_scope_followup_v2 import followup_book
from screen_task_registry import batches,validate_book

NAMESPACE='results/precision_runs_v1/fp64_20261115_v1'
STAGES=('baseline','epsilon','grn','full-cell-grn')

def validate_precision_plan(p):
    check_seal(p,'precision_plan_sha256')
    assert p['schema']=='dynavelo_precision_amendment_plan_v1' and p['namespace_relative']==NAMESPACE
    assert p['dtype']=='float64' and p['original_weight_values']=='exact_float32_values_promoted_to_float64'
    assert p['epsilon_production']==1e-4 and p['diagnostic_epsilons']==[1e-5,1e-4,1e-3]
    assert p['ode_method']=='dopri5' and p['ode_atol']==p['ode_rtol']==1e-7
    assert p['seed_list']==[0,1,2] and p['scientific_pass'] is False
    assert p['review']['sha256'] and p['source_records'] and p['deadline_unix']>0
    return p

def build_books(p,original,frame,shortlist):
    validate_precision_plan(p)
    assert p['original_scope_identity']==original['scope_plan_sha256'] and p['frame_identity']==frame['scope_frame_sha256']
    assert p['shortlist_identity']==shortlist['shortlist_sha256']
    context=original['context_plan'];geometry=context['spec']['chunk_geometry']
    grn=followup_book(original,frame,shortlist,'grn')
    fg=followup_book(original,frame,shortlist,'full-cell-grn')
    eps=followup_book(original,frame,shortlist,'epsilon')
    prototypes={seed:next(t for t in grn['tasks'] if t['seed']==seed) for seed in (0,1,2)}
    ids={
        'full_population':[c['cell_id'] for c in context['spec']['cells']],
        'epsilon_fixed_context':eps['tasks'][0]['sample_cell_ids'],
        'primary':[c['cell_id'] for c in frame['samples']['primary']['cells']],
        'full_effect_population':frame['effect_eligible_cell_ids']}
    weights={'full_population':None,'epsilon_fixed_context':None,
        'primary':[c['cell_weight'] for c in frame['samples']['primary']['cells']],
        'full_effect_population':[1/len(ids['full_effect_population'])]*len(ids['full_effect_population'])}
    assert {k:len(v) for k,v in ids.items()}=={'full_population':6035,'epsilon_fixed_context':128,'primary':192,'full_effect_population':1884}
    def task(template,stage,sample,feature,stable,space,h,parent=None,origin=None):
        body={k:copy.deepcopy(v) for k,v in template.items() if k not in ('task_id','assigned_shard')}
        body.update(schema='dynavelo_precision_task_v1',plan_sha256=p['precision_plan_sha256'],
            frame_sha256=frame['scope_frame_sha256'],precision_plan_sha256=p['precision_plan_sha256'],
            stage=stage,sample_id=sample,sample_cell_ids=ids[sample],n_cells=len(ids[sample]),
            sample_cell_axis_sha256=identity(ids[sample]),sample_weights_sha256=identity(weights[sample]),
            mode='evaluation-fixed' if stage=='baseline' else 'finite-difference',
            feature_id=feature,tf_stable_id=stable,input_space=space,epsilon=h,
            batches=batches(len(ids[sample]),geometry['cells']),parent_baseline_task_id=parent,
            original_fp32_task_id=origin,precision='float64',original_scope_reference_unchanged=True)
        value=seal(body,'task_id');value['assigned_shard']=int(value['task_id'][:16],16)%geometry.get('regulator_shards',1)
        return value
    tasks={stage:[] for stage in STAGES};parents={}
    for seed in (0,1,2):
        for sample in ids:
            t=task(prototypes[seed],'baseline',sample,'__baseline__','__baseline__','none',None)
            parents[(seed,sample)]=t['task_id'];tasks['baseline'].append(t)
    # Qualification retains the lexical64 family and all locked20 RNA/mapped
    # motifs, including known novelty/stability failures. No external-P selection.
    rna={t['tf_stable_id']:{'feature_id':t['feature_id'],'stable_id':t['tf_stable_id']} for t in eps['tasks'] if t['input_space']=='rna'}
    for row in shortlist['selected']:rna[row['stable_id']]={'feature_id':row['feature_id'],'stable_id':row['stable_id']}
    motif={t['feature_id']:{'feature_id':t['feature_id'],'stable_id':t['feature_id']} for t in eps['tasks'] if t['input_space']=='motif'}
    selected={r['feature_id'] for r in shortlist['selected']}
    for row in context['spec']['motifs']:
        if row['tf_rna_feature_id'] in selected:motif[row['feature_id']]={'feature_id':row['feature_id'],'stable_id':row['feature_id']}
    assert len(rna)==52 and len(motif)==38
    full_regulators=[('rna',rna[k]) for k in sorted(rna)]+[('motif',motif[k]) for k in sorted(motif)]
    sentinel=[('rna',next({'feature_id':t['feature_id'],'stable_id':t['tf_stable_id']} for t in eps['tasks'] if t['input_space']=='rna')),
              ('motif',next({'feature_id':t['feature_id'],'stable_id':t['tf_stable_id']} for t in eps['tasks'] if t['input_space']=='motif'))]
    for seed in (0,1,2):
        for sample,regulators in [('epsilon_fixed_context',full_regulators),('primary',sentinel),('full_effect_population',sentinel)]:
            for space,row in regulators:
                for h in p['diagnostic_epsilons']:
                    tasks['epsilon'].append(task(prototypes[seed],'epsilon',sample,row['feature_id'],row['stable_id'],space,h,parents[(seed,sample)]))
    for stage,source in [('grn',grn),('full-cell-grn',fg)]:
        for original_task in source['tasks']:
            sample=original_task['sample_id']
            tasks[stage].append(task(original_task,stage,sample,original_task['feature_id'],original_task['tf_stable_id'],
                original_task['input_space'],p['epsilon_production'],parents[(original_task['seed'],sample)],original_task['task_id']))
    assert {k:len(v) for k,v in tasks.items()}=={'baseline':12,'epsilon':846,'grn':4668,'full-cell-grn':84}
    books={}
    for stage,rows in tasks.items():
        books[stage]=seal({'schema':'dynavelo_task_book_v1','stage':stage,'plan_sha256':p['precision_plan_sha256'],
            'frame_sha256':frame['scope_frame_sha256'],'tasks':rows,'expected_tasks':len(rows),
            'full_tf_denominator':len(context['tf_universe']),'shortlist_denominator':20,'technical_attempt_limit':3,
            'coverage_contract':'Complete amended precision family; all failures and prior FP32 identities retained, no replacement.'},'book_sha256')
        validate_book(books[stage])
    return books

def prepare():
    # Run only when independent modules are final. This locks the new amendment
    # before any broad FP64 qualification result or GRN effect is opened.
    sys.path.insert(0,str(ROOT/'repo/a100_production_20261114'))
    from recovery_io_v1 import read,checked,record,new,now,sha
    report=ROOT/'results/reports/a100_production_20261114'
    attempt=report/'recovery_attempts/a100_recovery_20261115_a23'
    catalog=read(report/'post_screen_v1/followup/queue_catalog.json')
    original,frame,shortlist=[read(checked(ROOT,catalog[k])) for k in ('scope_plan','scope_frame','shortlist')]
    output=report/'precision_amendment_v1';output.mkdir(exist_ok=False)
    sources=['precision_plan_v1.py','precision_worker_v1.py','precision_streaming_v1.py','precision_inference_v1.py',
             'precision_pipeline_v1.py','precision_epsilon_comparison_v1.py']
    source_records=[record(ROOT,ROOT/'repo/a100_production_20261114'/name) for name in sources]
    for value in original['context_plan']['spec']['input_artifacts'].values():
        if isinstance(value,dict) and value.get('relative_path','').startswith('repo/') and value['relative_path'].endswith('.py'):
            checked(ROOT,value)
            if value not in source_records:source_records.append(value)
    for name in ('screen_task_registry.py','screen_plan.py','screen_dispatch.py','screen_scope_followup_numerics_v2.py'):
        value=record(ROOT,ROOT/'repo/colab'/name)
        if value not in source_records:source_records.append(value)
    review_path=output/'execution_review.json'
    review={'schema':'dynavelo_precision_execution_review_v1','created_utc':now(),
        'reviewer_id':'01a1cf01-2fc1-73a3-8732-11023905cf33','decision':'APPROVED_FOR_BOUNDED_FP64_QUALIFICATION_ONLY',
        'allowed_stages':['baseline','epsilon'],'scientific_pass':False,'GRN_authorized':False,
        'basis':'Original FP32 full diagnostic family unstable; original forward/cache exact, separate FP64 two-input probe stabilizes small-step derivatives. Larger qualification required before any GRN.',
        'prior_results_exposure':'FP32 full1191 rank, E1 all20 effects, FP32 epsilon576 and seed0 two-input FP64 precision probe already inspected. This is a new engineering amendment, not retroactive preregistration.',
        'all_original_fp32_outputs_retained':True,'epsilon_production_changed':False,'training_changed':False,
        'sources':source_records,'independent_design':record(ROOT,ROOT/'results/reports/epsilon_review_design_20261115_v1/fp64_qualification_amendment_design.md'),
        'precision_probe':record(ROOT,attempt/'precision_probe.json')}
    new(review_path,review)
    plan=seal({'schema':'dynavelo_precision_amendment_plan_v1','created_utc':now(),'namespace_relative':NAMESPACE,
        'original_scope_plan':catalog['scope_plan'],'scope_frame':catalog['scope_frame'],'shortlist':catalog['shortlist'],
        'original_scope_identity':original['scope_plan_sha256'],'frame_identity':frame['scope_frame_sha256'],
        'shortlist_identity':shortlist['shortlist_sha256'],'source_records':source_records,
        'review':record(ROOT,review_path),'dtype':'float64','original_weight_values':'exact_float32_values_promoted_to_float64',
        'epsilon_production':1e-4,'diagnostic_epsilons':[1e-5,1e-4,1e-3],'seed_list':[0,1,2],
        'ode_method':'dopri5','ode_atol':1e-7,'ode_rtol':1e-7,'workers':8,'cpu_threads_per_worker':1,
        'deadline_unix':read(attempt/'controller_status_v3.json')['deadline_unix'],
        'stop_relative':NAMESPACE+'/STOP','scientific_pass':False,
        'fixed_reference_frame_interpretation':'Original FP32 population, label eligibility, cell sampling, weights, time-window strata and RMS remain explicit frozen reference definitions. Full6035 FP64 baselines are retained to assess precision drift; not presented as newly learned FP64 window boundaries.',
        'qualification_thresholds':{
            'status':'New engineering criteria frozen after the disclosed small precision probe, before broad846 outcomes',
            'small_step_relative_L2_max':.01,'large_step_relative_L2_max':.1,
            'small_step_strong_nonzero_sign_agreement_min':.99,'large_step_strong_nonzero_sign_agreement_min':.95,
            'low_signal_reference_RMS_threshold':1e-8,
            'strong_entry_absolute_derivative_threshold':1e-7,
            'strong_entry_mask':'max(abs(current),abs(reference)) >= threshold; zero-to-strong-nonzero counts as sign disagreement',
            'low_signal_rule':'Retain and label LOW_NUMERICAL_SIGNAL; never use as stable/null biological evidence. Report absolute errors and all zero/nonzero sign categories.',
            'nonlinear_or_failed_row_rule':'Retain explicit row-level HOLD; no epsilon tuning/deletion. Full GRN requires separate actual-family scientific review; reporting any axis additionally requires stable relevant inputs/outputs/windows.',
            'strict_no_op_max_absolute':0,'same_backend_original_nine_output_bitwise':True,'all_rng_bitwise':True},
        'resource_limits':{'system_free_bytes':16*1024**3,'rss_bytes':64*1024**3,'cuda_reserved_bytes':64*1024**3,
            'forward_seconds':120,'wall_seconds':7200,'min_free_disk_bytes':2*1024**3,'task_payload_bytes':1024**3},
        'no_new_external_outcome_access':True,'no_fp32_task_identity_reuse':True},'precision_plan_sha256')
    books=build_books(plan,original,frame,shortlist)
    new(output/'precision_plan.json',plan)
    for stage,book in books.items():new(output/'task_books'/(stage+'.json'),book)
    new(output/'prepared_receipt.json',{'status':'NEW_FP64_FAMILY_FROZEN_NOT_EXECUTED','created_utc':now(),
        'plan':record(ROOT,output/'precision_plan.json'),'counts':{k:len(v['tasks']) for k,v in books.items()},
        'worker_GPU_qualification':'PENDING','scientific_pass':False,'old_runtime_session':'dynavelo-a100-followup-20261115-a1'})
    print(json.dumps({'status':'FP64_PRECISION_FAMILY_PREPARED','precision_plan_sha256':plan['precision_plan_sha256'],'counts':{k:len(v['tasks']) for k,v in books.items()}}))

if __name__=='__main__':prepare()
