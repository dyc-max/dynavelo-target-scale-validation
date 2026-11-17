"""Separate PRIMARY192 h1e-4 descriptive readout. Default metadata only; root approval required.
No full-cell completion, numerical qualification, biological screening, model or cloud calls.
"""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import argparse,csv,hashlib,json,math,os,sys,time
ROOT=Path(r'D:\dyc_7\DynaVelo_Research')
CODE=ROOT/'repo/a100_production_20261114'
NS='results/precision_runs_v1/fp64_20261115_v1'
PREP='results/reports/primary_only_readout_feasibility_20261116_v1'
STATUS='ALL4668_PRIMARY_COMMITS_AND_PARENTS_VERIFIED_DESCRIPTIVE_ONLY'

def require(v,m):
    if not v:raise ValueError(m)

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def record(p):
    p=Path(p).resolve();require(p.is_relative_to(ROOT.resolve()),'Project D path required')
    return {'relative_path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def path(v):
    require(isinstance(v,str) and '\\' not in v and ':' not in v,'Canonical project relative path required')
    q=PurePosixPath(v);require(not q.is_absolute() and q.parts and all(x not in ('.','..') for x in q.parts) and q.as_posix()==v,'Unsafe relative path')
    p=ROOT.joinpath(*q.parts);require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(),'Resolved project path required');return p

def checked(rec):
    require(set(rec)=={'relative_path','bytes','sha256'} and type(rec['bytes']) is int and rec['bytes']>=0,'Exact artifact record required')
    p=path(rec['relative_path']);require(p.is_file() and record(p)==rec,'Artifact bytes/SHA changed: '+rec['relative_path']);return p

def new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def validate_primary_shape(books):
    tasks=books['grn']['tasks'];require(len(tasks)==4668,'Exact complete PRIMARY4668 required')
    require(len({t['task_id'] for t in tasks})==4668,'Unique primary task IDs required')
    counts=Counter();families=defaultdict(list)
    for t in tasks:
        require(t.get('stage')=='grn' and t.get('sample_id')=='primary' and t.get('n_cells')==192 and type(t.get('seed')) is int and t['seed'] in (0,1,2),'PRIMARY192 seed012 only')
        require(t.get('input_space') in ('rna','motif') and isinstance(t.get('feature_id'),str) and t['feature_id'],'Exact input descriptors required')
        counts[(t['seed'],t['input_space'])]+=1;families[(t['input_space'],t['feature_id'])].append(t['seed'])
    require(all(counts[(s,'rna')]==1191 and counts[(s,'motif')]==365 for s in (0,1,2)),'Full1191RNA365motif denominator per seed required')
    require(len(families)==1556 and all(sorted(v)==[0,1,2] for v in families.values()),'Every input must retain all3 seeds')

def validate_census(census,books):
    validate_primary_shape(books);expected={t['task_id'] for t in books['grn']['tasks']}
    require(len(census)==4668 and len({x['task_id'] for x in census})==4668 and {x['task_id'] for x in census}==expected,'Census must equal full PRIMARY book')
    require(all(x.get('stage')=='grn' and set(x['commit'])=={'relative_path','bytes','sha256'} for x in census),'Exact PRIMARY-only COMMIT census required')

def qualification_for(task,channel,hold_ledger):
    require(channel in ('vx','vy'),'Output channel required')
    return {'status':'NOT_TESTED','reason':'No relevant sample/seed/input/target/window step and solver qualification supplied; original16HOLD ledger and axis-interpretation HOLD remain unchanged. CPU evidence does not bridge A100.'}

def rank_descriptive(rows):
    families=defaultdict(list)
    for r in rows:families[tuple(r[k] for k in ('stage','input_space','seed','channel','group_kind','group_name'))].append(r)
    for family in families.values():
        valid=[r for r in family if r['weighted_mean_vector_L2'] is not None]
        require(all(math.isfinite(r['weighted_mean_vector_L2']) and r['weighted_mean_vector_L2']>=0 for r in valid),'Finite nonnegative descriptive scores required')
        valid.sort(key=lambda r:r['weighted_mean_vector_L2']);n=len(valid)
        for r in family:r.update(descriptive_rank=None,descriptive_percentile=None,rank_valid_regulator_count=n,rank_full_family_regulator_count=len(family))
        i=0
        while i<n:
            j=i+1
            while j<n and valid[j]['weighted_mean_vector_L2']==valid[i]['weighted_mean_vector_L2']:j+=1
            mean=(i+1+j)/2
            for r in valid[i:j]:r.update(descriptive_rank=n+1-mean,descriptive_percentile=(mean-1)/(n-1) if n>1 else None)
            i=j

def seal_check(v,key):
    require(hashlib.sha256(canonical({k:x for k,x in v.items() if k!=key})).hexdigest()==v.get(key),'Frozen seal differs: '+key)

def make_contract(output):
    require(output==ROOT/PREP/'prospective_contract_v3.json','Exact new prospective contract path required')
    recovery=ROOT/NS/'lost_runtime_recovery/a100_recovery_20261116_a27/recovery.json';rr=read(recovery)
    plan=read(checked(rr['precision_plan']));review_pointer=ROOT/NS/'actualFP64qualification_review_record.json';review_record=read(review_pointer);review=read(checked(review_record))
    inv=ROOT/PREP/'metadata_inventory.json';require(sha(inv)=='3cbcc28cc09bddc9f56942b6891e677498620058587e8c9708e5e3bce00965bd','Original metadata qualification inventory differs')
    sources=[ROOT/'repo/colab/screen_plan.py',ROOT/'repo/colab/screen_task_registry.py',CODE/'precision_grn_readout_v1.py',CODE/'precision_worker_v1.py',CODE/'precision_plan_v1.py',ROOT/'repo/colab/screen_scope_followup_numerics_v2.py']
    value={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'schema':'dynavelo_primary_descriptive_readout_contract_v1','decision':'PREPARED_ROOT_REVIEW_REQUIRED','execution_authorized':False,'adapter':record(Path(__file__)),
        'sources':[record(x) for x in sources],'precision_plan':rr['precision_plan'],'original_recovery':record(recovery),'native_coverage':read(inv)['native_coverage'],'metadata_qualification':record(inv),
        'books':{s:record(ROOT/NS/s/'task_book.json') for s in ('baseline','epsilon','grn')},'original_book_records':{s:rr['book_records'][s] for s in ('baseline','epsilon','grn')},
        'actual_review_pointer':record(review_pointer),'actual_review':review_record,'epsilon_comparison':review['epsilon_comparison'],
        'scope':{'stage':'grn','sample_id':'primary','n_cells':192,'epsilon':0.0001,'seeds':[0,1,2],'inputs':{'rna':1191,'motif':365},'primary_tasks':4668,'baselines':12,'epsilon_history':846,'fullcell_completed':28,'fullcell_expected':84,'missing_fullcell':56},
        'output_relative':'results/reports/primary_only_grn_descriptive_20261116_v1',
        'qualification_policy':'ALL_NOT_TESTED; original16HOLD separate unchanged; no CPU-to-A100 bridge','interpretation_status':'HOLD','scientific_pass':False,
        'definition':'h1e-4 FP64 local input-sensitivity GRN descriptive summaries only; all PRIMARY inputs, all3 seeds, all targets/groups/negative/zero/NA retained',
        'forbidden_claims':['full5610_complete','full84_complete','1884_crossseed_concordance','rescue','clinical_P','causality','independent_donor_replication','scientific_PASS','new_candidate_screening'],
        'changed_original_files':False,'previously_consumed_E1_remains_consumed':True,'DC_subtype_mapping_promoted':False}
    new(output,value);return {'status':'PROSPECTIVE_PRIMARY_ONLY_CONTRACT_PREPARED_NO_EDGE_VALUES','contract':record(output)}

def load_contract(p,digest):
    require(p.is_absolute() and p.resolve().is_relative_to(ROOT.resolve()) and sha(p)==digest,'Exact contract path/hash required')
    c=read(p);require(c['schema']=='dynavelo_primary_descriptive_readout_contract_v1' and c['decision']=='PREPARED_ROOT_REVIEW_REQUIRED' and c['execution_authorized'] is False,'Prospective separate-scope contract required')
    require(c['adapter']==record(Path(__file__)) and c['scope']=={'stage':'grn','sample_id':'primary','n_cells':192,'epsilon':0.0001,'seeds':[0,1,2],'inputs':{'rna':1191,'motif':365},'primary_tasks':4668,'baselines':12,'epsilon_history':846,'fullcell_completed':28,'fullcell_expected':84,'missing_fullcell':56},'Actual adapter/scope differs')
    for rec in c['sources']+[c['precision_plan'],c['original_recovery'],c['native_coverage'],c['metadata_qualification'],c['actual_review_pointer'],c['actual_review'],c['epsilon_comparison']]:checked(rec)
    for s in ('baseline','epsilon','grn'):
        actual=checked(c['books'][s]); declared=checked(c['original_book_records'][s]); book=read(actual)
        require(actual.read_bytes()==canonical(book) and canonical(book)==canonical(read(declared)),'Returned canonical book differs from frozen original book semantics')
    require(read(checked(c['actual_review_pointer']))==c['actual_review'],'Actual original review pointer differs')
    return c

def audit_family(c):
    """Own exact books, COMMITs, payload SHA, completion pointers and parent audit; no np.load."""
    plan=read(checked(c['precision_plan']));seal_check(plan,'precision_plan_sha256')
    for rec in plan['source_records']+[plan[k] for k in ('review','original_scope_plan','scope_frame','shortlist')]:checked(rec)
    books={s:read(checked(rec)) for s,rec in c['books'].items()};validate_primary_shape(books)
    for s,b in books.items():
        seal_check(b,'book_sha256');require(b['stage']==s and b['expected_tasks']=={'baseline':12,'epsilon':846,'grn':4668}[s] and b['technical_attempt_limit']==3 and b['plan_sha256']==plan['precision_plan_sha256'],'Original exact book contract differs')
        for t in b['tasks']:
            seal_check({k:v for k,v in t.items() if k!='assigned_shard'},'task_id')
            require(t['stage']==s and t['precision_plan_sha256']==plan['precision_plan_sha256'],'Task escaped precision book')
    cov=read(checked(c['native_coverage']));require(cov['status']=='PARTIAL_NATIVE_COVERAGE' and cov['total_accepted_tasks']==5554 and cov['expected_tasks']==5610,'Actual5554-only native evidence required')
    accepted={x['task_id']:x for x in cov['census']};require(len(cov['census'])==len(accepted)==5554,'Actual native census must remain unique')
    review=read(checked(c['actual_review']));require(review['decision']=='APPROVED_FOR_FIXED_FP64_DESCRIPTIVE_GRN' and review['precision_plan_sha256']==plan['precision_plan_sha256'] and review['family_numerical_stability_PASS'] is False and len(review['all_16_HOLD_retained'])==16 and review['failed_rows_upgraded_to_PASS'] is False and review['scientific_pass'] is False and review['interpretation_gate']['status']=='HOLD','Original16HOLD/axis-interpretation gate must remain')
    expected_files={'baseline':{'STARTED.json','result.json','baseline/baseline_manifest.json','baseline/time.npy','baseline/vx.npy','baseline/vy.npy','baseline/vz.npy'},'epsilon':{'STARTED.json','result.json','diagnostic_axes.json','vx_derivative.npy','vy_derivative.npy'},'grn':{'STARTED.json','result.json','derivative_integrity.json','edge_axes.json','vx_edges.npz','vy_edges.npz'}}
    saved={};covers={};primary=[];items=[]
    for stage in ('baseline','epsilon','grn'):
        b=books[stage];rows=[]
        for t in b['tasks']:
            key=t['task_id'];require(key in accepted,'Original task absent from actual native receipt');rec=accepted[key]['commit'];cp=checked(rec);d=read(cp)
            require(d['schema']=='dynavelo_task_commit_v1' and d['status']=='COMPLETE' and d['task']==t and d['book_sha256']==b['book_sha256'] and type(d['attempt']) is int and 1<=d['attempt']<=3,'Exact original COMMIT metadata required')
            rel='tasks/'+key+'/attempt_%03d/COMMIT.json'%d['attempt'];require(rec['relative_path']==NS+'/'+stage+'/'+rel,'COMMIT escaped exact original task attempt')
            ptr=read(ROOT/NS/stage/'tasks'/key/'COMPLETE.json');require(ptr=={'manifest_relative_path':rel,'bytes':rec['bytes'],'sha256':rec['sha256']},'Original completion pointer differs')
            members={x['name']:x for x in d['files']};require(len(members)==len(d['files']) and set(members)==expected_files[stage],'Exact committed payload family required')
            directory=cp.parent
            require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='COMMIT.json'}==set(members),'Uncommitted/omitted payload member detected')
            artifacts={}
            for name,m in members.items():
                require(set(m)=={'name','bytes','sha256'},'Exact member record required')
                rr={'relative_path':(directory/name).relative_to(ROOT).as_posix(),'bytes':m['bytes'],'sha256':m['sha256']};checked(rr);artifacts[name]=rr
            value=read(directory/'result.json')
            expected={'schema':'dynavelo_task_result_v1','task_id':key,'status':'COMPLETE','numerical_integrity':'PASS','n_cells':t['n_cells'],'mode':t['mode'],'sample_cell_axis_sha256':t['sample_cell_axis_sha256'],'plan_sha256':plan['precision_plan_sha256'],'precision_plan_sha256':plan['precision_plan_sha256'],'frame_sha256':t['frame_sha256'],'parent_baseline_task_id':t['parent_baseline_task_id'],'precision_experiment':'FP64_PROMOTED_FROZEN_CHECKPOINT_INPUT_SENSITIVITY_ONLY','precision_dtype':'float64','biological_or_external_PASS':False,'scientific_pass':False}
            expected.update({k:t[k] for k in ('seed','stage','sample_id','checkpoint_sha256')})
            require(all(value.get(k)==v for k,v in expected.items()),'Result escaped original precision task')
            if stage=='baseline':
                require(value.get('parent_baseline_dependency') is None and value.get('no_op')=='EXACT_ZERO_ALL_CHANNELS_ALL_CHUNKS' and value['baseline_manifest_sha256']==artifacts['baseline/baseline_manifest.json']['sha256'],'Baseline no-op/manifest binding differs')
            else:
                parent=saved[t['parent_baseline_task_id']];pt=parent['task'];dep={'task_id':pt['task_id'],'commit_sha256':parent['commit']['sha256'],'book_sha256':books['baseline']['book_sha256']}
                require(all(t[k]==pt[k] for k in ('seed','checkpoint_sha256','sample_id','sample_cell_ids','sample_cell_axis_sha256','n_cells','batches','chunk_geometry')) and value.get('parent_baseline_dependency')==dep and value.get('matched_baseline_dependency')==dep,'Exact matched parent baseline geometry/COMMIT differs')
            for k in ('precision_config_sha256','precision_config'):
                if k in t:require(value.get(k)==t[k],'Original precision configuration differs')
            entry={'task':t,'commit':rec,'result':artifacts['result.json'],'artifacts':artifacts,'parent_baseline_dependency':value.get('parent_baseline_dependency')};saved[key]=entry
            rows.append({'status':'COMPLETE','task_id':key,'manifest_sha256':rec['sha256'],'manifest_relative_path':rel,'files':d['files']})
            if stage=='grn':
                require(t['epsilon']==0.0001 and value.get('review_sha256')==c['actual_review']['sha256'],'Primary h1e-4/review binding differs');primary.append({'task_id':key,'stage':'grn','commit':rec});items.append(entry)
        cv={'schema':'dynavelo_task_coverage_v1','book_sha256':b['book_sha256'],'plan_sha256':b['plan_sha256'],'stage':stage,'full_tf_denominator':b['full_tf_denominator'],'expected_tasks':len(rows),'counts':{'COMPLETE':len(rows),'PENDING':0,'FAILED':0,'INCOMPLETE':0,'CORRUPT':0},'all_expected_tasks_complete':True,'tasks':rows};cv['coverage_sha256']=hashlib.sha256(canonical(cv)).hexdigest();covers[stage]=cv
    validate_census(primary,books)
    comparison=read(checked(c['epsilon_comparison']));require(covers['epsilon']['coverage_sha256']==review['epsilon_coverage_sha256']==comparison['coverage_sha256'],'Complete846 epsilon history binding differs')
    return plan,books,review,covers,primary,items

def execute(contract_path,contract_sha,approval_path,approval_sha):
    require(os.name=='nt' and sys.flags.optimize==0 and Path(sys.executable).resolve()==(ROOT/'repo/.venv/Scripts/python.exe').resolve(),'Exact Windows native Python required')
    c=load_contract(contract_path,contract_sha);approval=read(approval_path)
    require(approval_path.is_absolute() and approval_path.resolve().is_relative_to(ROOT.resolve()) and sha(approval_path)==approval_sha and approval.get('schema')=='dynavelo_primary_descriptive_readout_root_approval_v1' and approval.get('decision')=='APPROVED_PRIMARY_ONLY_DESCRIPTIVE_READOUT' and approval.get('contract')==record(contract_path) and approval.get('adapter')==record(Path(__file__)) and approval.get('scientific_pass') is False,'Separate exact root approval required before any actual payload/value read')
    cache=ROOT/'tmp_cache/primary_only_grn_descriptive_v1';cache.mkdir(parents=True,exist_ok=True)
    for key in ('TMP','TEMP','TMPDIR','PIP_CACHE_DIR','HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR'):
        folder=cache/key;folder.mkdir(parents=True,exist_ok=True);os.environ[key]=str(folder)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    output=path(c['output_relative']);require(output.is_relative_to(ROOT/'results') and not output.exists(),'New separate PRIMARY output required')
    output.mkdir(parents=True,exist_ok=False);new(output/'prospective_contract_v3.json',c);new(output/'root_approval.json',approval)
    started=time.monotonic();plan,books,review,covers,census,items=audit_family(c)
    # Own complete book/COMMIT/payload/parent audits precede original parser imports and edge values.
    sys.path[:0]=[str(CODE),str(ROOT/'repo/colab')]
    import precision_grn_readout_v1 as helper
    import numpy as np
    loaded,original,frame,shortlist,generated=helper.load_definition(checked(c['precision_plan']),c['precision_plan']['sha256'])
    require(canonical(loaded)==canonical(plan) and all(canonical(generated[s])==canonical(books[s]) for s in ('baseline','epsilon','grn')),'Frozen original mathematical book derivation differs')
    parse_edge_payload,expected_axes,validate_axes=helper.parse_edge_payload,helper.expected_axes,helper.validate_axes
    for s,cv in covers.items():new(output/(s+'_coverage.json'),cv)
    new(output/'complete_PRIMARY_commit_census.json',{'schema':'dynavelo_primary_commit_census_v1','primary_tasks':4668,'census':census,'fullcell_complete':28,'fullcell_expected':84,'missing_fullcell':56,'full_original5610_complete':False,'scientific_pass':False})
    new(output/'original16HOLD_unchanged.json',{'actual_review':c['actual_review'],'all_16_HOLD_retained':review['all_16_HOLD_retained'],'interpretation_gate':review['interpretation_gate'],'failed_rows_upgraded_to_PASS':False,'CPU_to_A100_bridge':'NOT_TESTED_UNAVAILABLE','scientific_pass':False})
    inputs={};tasks_by_id={t['task_id']:t for t in books['grn']['tasks']}
    for raw in items:
        task=raw['task'];key=('grn',task['input_space'],task['feature_id']);art=raw['artifacts']
        item={'task':{k:task[k] for k in ('task_id','seed','stage','sample_id','n_cells','input_space','feature_id','tf_stable_id','checkpoint_sha256')},'commit':raw['commit'],'result':raw['result'],'edge_axes':art['edge_axes.json'],'vx_edges':art['vx_edges.npz'],'vy_edges':art['vy_edges.npz'],'parent_baseline_dependency':raw['parent_baseline_dependency'],'qualification':{ch:qualification_for(task,ch,review['all_16_HOLD_retained']) for ch in ('vx','vy')}}
        entry=inputs.setdefault(key,{'stage':'grn','input_space':task['input_space'],'feature_id':task['feature_id'],'tf_stable_id':task['tf_stable_id'],'seed_tasks':[]});entry['seed_tasks'].append(item)
    require(len(inputs)==1556 and all(sorted(x['task']['seed'] for x in e['seed_tasks'])==[0,1,2] for e in inputs.values()),'Complete3seed PRIMARY inputs required')
    summaries=[];consensus=[];group_axes={}
    for key,entry in inputs.items():
        channel_means={c:[] for c in ('vx','vy')};axes_by_seed={}
        for item in sorted(entry['seed_tasks'],key=lambda x:x['task']['seed']):
            task=tasks_by_id[item['task']['task_id']];axes=read(checked(item['edge_axes']))
            validate_axes(axes,expected_axes(original,frame,task));axes_by_seed[task['seed']]=axes
            axis_key=task['stage']+'__seed'+str(task['seed'])
            if axis_key in group_axes and canonical(group_axes[axis_key])!=canonical(axes['groups']):raise ValueError('Groups differ across inputs within same sample/seed')
            group_axes[axis_key]=axes['groups']
            for channel in ('vx','vy'):
                source=checked(item[channel+'_edges']);arrays=parse_edge_payload(source,axes,task)
                channel_means[channel].append(arrays['weighted_signed_mean'])
                for group_index,group in enumerate(axes['groups']):
                    values=arrays['weighted_signed_mean'][group_index];finite=np.isfinite(values)
                    score=float(np.sqrt(np.sum(values[finite]**2,dtype=np.float64))) if finite.any() else None
                    summaries.append({'stage':task['stage'],'input_space':task['input_space'],'feature_id':task['feature_id'],'tf_stable_id':task['tf_stable_id'],
                        'seed':task['seed'],'sample_id':task['sample_id'],'channel':channel,'group_kind':group['kind'],'group_name':group['name'],
                        'sample_n':group['sample_n'],'effect_frame_domain_n':group['effect_frame_domain_n'],'sample_weight_sum':group['sample_weight_sum'],
                        'target_count':len(values),'positive_targets':int((values>0).sum()),'negative_targets':int((values<0).sum()),
                        'zero_targets':int((values==0).sum()),'NA_targets':int((~finite).sum()),'weighted_mean_vector_L2':score,
                        'descriptive_rank':None,'descriptive_percentile':None,'rank_valid_regulator_count':None,'rank_full_family_regulator_count':None,
                        'qualification_status':item['qualification'][channel]['status'],'patient_P_value':None,'scientific_pass':False})
                checked(item[channel+'_edges']) # ensure immutable source persisted through parse
        reference_groups=axes_by_seed[0]['groups']
        for channel,matrices in channel_means.items():
            tensor=np.stack(matrices);assert tensor.shape[0]==3
            # Structural group identity retained across seeds, while n/masses
            # may differ for original seed-specific time-window strata.
            for seed,axes in axes_by_seed.items():
                if [(g['kind'],g['name']) for g in axes['groups']]!=[(g['kind'],g['name']) for g in reference_groups]:raise ValueError('Group names differ across seeds')
            for i,group in enumerate(reference_groups):
                values=tensor[:,i,:];nfinite=np.isfinite(values).sum(axis=0);pos=(values>0).sum(axis=0);neg=(values<0).sum(axis=0);zero=(values==0).sum(axis=0)
                consensus.append({'stage':entry['stage'],'input_space':entry['input_space'],'feature_id':entry['feature_id'],'channel':channel,
                    'group_kind':group['kind'],'group_name':group['name'],'target_count':values.shape[1],
                    'all3_positive_targets':int((pos==3).sum()),'all3_negative_targets':int((neg==3).sum()),'all3_zero_targets':int((zero==3).sum()),
                    'mixed_sign_or_zero_complete_targets':int(((nfinite==3)&(pos!=3)&(neg!=3)&(zero!=3)).sum()),
                    'targets_with_any_NA':int((nfinite<3).sum()),'seeds_are_independent_donors':False,'patient_P_value':None,'scientific_pass':False})
    rank_descriptive(summaries)
    for row in summaries+consensus:row.update(qualification_status='NOT_TESTED',axis_interpretation_status='HOLD')
    for name,rows in [('all_PRIMARY_input_strength_ranks.tsv',summaries),('all_PRIMARY_input_direction_consistency.tsv',consensus)]:
        require(rows,'No descriptive rows generated')
        with (output/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    for rec in c['sources']+[c['adapter'],c['native_coverage'],c['actual_review']]+list(c['books'].values()):checked(rec)
    for raw in items:checked(raw['commit'])
    index={'schema':'dynavelo_primary_precision_grn_backed_index_v1','status':STATUS,'adapter':record(Path(__file__)),'contract':record(contract_path),'root_approval':record(approval_path),'precision_plan':c['precision_plan'],'native5554_coverage':c['native_coverage'],'original_scope_plan':plan['original_scope_plan'],'scope_frame':plan['scope_frame'],'shortlist':plan['shortlist'],'scope':c['scope'],'primary_complete_tasks':4668,'PRIMARY_inputs':1556,'fullcell_complete':28,'fullcell_expected':84,'missing_fullcell':56,'full_original5610_complete':False,'coverage_records':{s:record(output/(s+'_coverage.json')) for s in covers},'actual_qualification_review':c['actual_review'],'qualification_map':None,'qualification_policy':c['qualification_policy'],'unchanged16HOLD':record(output/'original16HOLD_unchanged.json'),'rna_target_axis':[{'feature_id':r['feature_id'],'stable_id':r['stable_id']} for r in original['context_plan']['spec']['genes']],'motif_target_axis':original['context_plan']['spec']['motifs'],'group_axes':group_axes,'inputs':list(inputs.values()),'all_original11_statistics_queryable':True,'negative_zero_NA_retained':True,'frozen20_candidates':shortlist['selected'],'candidate_replacement':False,'candidate_screening_performed':False,'new_external_validation_selected':False,'reference_time_windows':'Original frozen FP32 frame; unchanged PRIMARY192 strata','interpretation':'h1e-4 local input-sensitivity GRN, descriptive only; axis interpretation HOLD','seeds_are_independent_donors':False,'patient_P_values_computed':False,'CPU_to_A100_bridge':'NOT_TESTED_UNAVAILABLE','previously_consumed_E1_remains_consumed':True,'DC_subtype_mapping_promoted':False,'scientific_pass':False,'seconds':time.monotonic()-started}
    new(output/'PRIMARY_backed_grn_index.json',index)
    result={'status':STATUS,'PRIMARY_tasks':4668,'input_families':1556,'strength_rows':len(summaries),'consensus_rows':len(consensus),'all_qualification_NOT_TESTED':True,'original16HOLD_retained':True,'missing_fullcell':56,'full_original5610_complete':False,'axis_interpretation':'HOLD','scientific_pass':False,'seconds':time.monotonic()-started}
    new(output/'DESCRIPTIVE_COMPLETE.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare-contract','metadata','execute']);p.add_argument('--contract');p.add_argument('--contract-sha');p.add_argument('--approval');p.add_argument('--approval-sha');a=p.parse_args()
    if a.mode=='prepare-contract':value=make_contract(ROOT/PREP/'prospective_contract_v3.json')
    else:
        require(a.contract and a.contract_sha,'Exact --contract and --contract-sha required');cp=Path(a.contract)
        if a.mode=='metadata':
            c=load_contract(cp,a.contract_sha);value={'status':'METADATA_ONLY_NO_EDGE_VALUES_OPENED','contract':record(cp),'scope':c['scope'],'execution_authorized':False,'scientific_pass':False}
        else:
            require(a.approval and a.approval_sha,'Root --approval and --approval-sha required');value=execute(cp,a.contract_sha,Path(a.approval),a.approval_sha)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=='__main__':main()
