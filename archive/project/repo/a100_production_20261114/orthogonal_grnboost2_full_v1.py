"""Continuous3006x3 GRNBoost2 family; exact COMMIT resume, no silent failed-task retry."""
from pathlib import Path
from collections import Counter
import argparse,ctypes,hashlib,importlib.metadata,json,logging,math,multiprocessing,os,shutil,sys,time,warnings
import orthogonal_grnboost2_pilot_v1 as base
R,C,P,PARAMS=base.R,base.C,base.P,base.PARAMS
read,rec,checked,new,tsv,cache_env,require=base.read,base.rec,base.checked,base.new,base.tsv,base.cache_env,base.require
O=P/'native_full_v1';PILOT=P/'native_pilot_v1/contract_v2.json';PILOT_SHA='c3a28821c48a0703c573679d4705eb6439b3e20d2394b668ba2a43b848478b1e'
INPUT=R/'data/processed/orthogonal_grn_v1/frozen_RNA_X_v1/manifest.json';INPUT_SHA='9135e0b2bf03dda5856d4dd88eb94893756f42972b755ddb8d90de29a5b145b5'
FITTED=('SUCCESS_GRNBOOST2_TARGET','SUCCESS_GRNBOOST2_TARGET_AFTER_LIBRARY_RETRY');RESOLVED=FITTED+('NOT_TESTED_CONSTANT_TARGET',)
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def status(p,v):
    temp=Path(p).with_suffix('.'+str(time.time_ns())+'.partial');new(temp,v);os.replace(temp,p)
def tasks(axes):
    return [{'ordinal':s*3006+i,'task_id':'seed%d_target%04d'%(s,i),'seed':s,'target_index':i,'target_stable_id':g['stable_id'],'target_symbol':g['symbol']} for s in (0,1,2) for i,g in enumerate(axes['genes'])]

def prepare():
    require(not O.exists(),'Fresh full preparation required');p=base.load_contract(PILOT,PILOT_SHA);require(base.sha(INPUT)==INPUT_SHA,'Exact actual input manifest required');m=read(INPUT);axes=read(checked(p['axes']))
    require(m['contract']==rec(PILOT) and m['axes']==p['axes'] and m['logical_sha256']==p['logical_RNA_X_sha256'] and m['normalization_performed'] is False,'Actual equivalent exported input required')
    family={'schema':'dynavelo_grnboost2_full_family_v1','source':rec(__file__),'pilot_reference_source':p['source'],'input_manifest':rec(INPUT),'axes':p['axes'],'matrix':m['matrix'],'logical_RNA_X_sha256':m['logical_sha256'],'parameters':PARAMS,'early_stop_window_length':25,'seeds':[0,1,2],'target_indices':list(range(3006)),'target_count':3006,'predictor_count':1191,'all20_retained':True,'official_arboreto_core':p['official_arboreto_core'],'interpreter':p['interpreter'],'runtime_install_report':p['runtime_install_report'],'runtime_versions':p['runtime_versions']}
    v={'schema':'dynavelo_grnboost2_full_execution_contract_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'PREPARED_ROOT_REVIEW_REQUIRED_NO_IMPORTANCE_READ','source':rec(__file__),'pilot_contract':rec(PILOT),'family':family,'family_sha256':identity(family),'namespace_relative':'results/orthogonal_grn_v1/full3006_seeds012_v1','workers':8,'BLAS_threads_per_worker':1,'limits':{'wall_seconds':14400,'max_RSS_bytes':10*1024**3,'max_job_commit_bytes':10*1024**3,'min_system_available_bytes':2*1024**3,'startup_available_bytes':12*1024**3,'minimum_D_free_bytes':4*1024**3},'first64_observation':'fixed task ordinals0..63, seed0 native targets0..63; resource/status observation only; continuous dispatch never waits for target acceptance','resume_policy':'Only exact family/source/payload/execution-provenance COMMITs skip. Existing FAILED/incomplete folders remain unresolved; never silently retried. New run ID does not reset tasks.','pilot_reuse':'8 pilot targets remain separate evidence; full family recomputes all9018 in new namespace.','resource_adjustment':'Root may prospectively change wall/workers in a new source-bound execution contract with identical scientific family hash; no automatic cap increase.','method_scope':'same6035 cells/all13 labels/all20 retained; algorithm corroboration, not independent donor validation','original_numerical_HOLD_NOT_TESTED_unchanged':True,'CPU_A100_bridge':'NOT_TESTED_UNAVAILABLE','E1_remains_previously_consumed':True,'DC_subtype_mapping_promoted':False,'scientific_pass':False,'importance_values_read_during_preparation':False}
    O.mkdir(parents=True);new(O/'contract.json',v);new(O/'task_catalog.json',{'family_sha256':v['family_sha256'],'tasks':tasks(axes),'expected_tasks':9018,'order':'seed-major012, native gene index0..3005','scientific_pass':False});return {'status':v['status'],'contract':rec(O/'contract.json'),'expected_tasks':9018}

def load(p,digest):
    require(p.is_absolute() and p.resolve().is_relative_to(R.resolve()) and base.sha(p)==digest,'Exact D contract path/hash required');v=read(p);f=v['family'];require(v['schema']=='dynavelo_grnboost2_full_execution_contract_v1' and v['source']==rec(__file__) and f['source']==v['source'] and identity(f)==v['family_sha256'],'Full source/family differs')
    p=base.load_contract(checked(v['pilot_contract']),v['pilot_contract']['sha256']);m=read(checked(f['input_manifest']));axes=read(checked(f['axes']))
    require(v['pilot_contract']['sha256']==PILOT_SHA and f['input_manifest']['sha256']==INPUT_SHA and f['pilot_reference_source']==p['source'] and f['matrix']==m['matrix'] and f['axes']==m['axes']==p['axes'] and f['logical_RNA_X_sha256']==m['logical_sha256']==p['logical_RNA_X_sha256'],'Actual pilot/input provenance differs')
    require(f['parameters']==PARAMS and f['early_stop_window_length']==25 and f['seeds']==[0,1,2] and f['target_indices']==list(range(3006)) and f['target_count']==3006 and f['predictor_count']==1191 and f['all20_retained'] is True,'Scientific family differs')
    for k in ('official_arboreto_core','interpreter','runtime_install_report'):require(f[k]==p[k],'Runtime provenance differs');checked(f[k])
    require(f['runtime_versions']==p['runtime_versions'] and v['namespace_relative']=='results/orthogonal_grn_v1/full3006_seeds012_v1' and type(v['workers']) is int and 1<=v['workers']<=32 and v['BLAS_threads_per_worker']==1,'Runtime/namespace differs')
    lim=v['limits'];require(type(lim['wall_seconds']) is int and 1<=lim['wall_seconds']<=86400 and lim['max_RSS_bytes']==lim['max_job_commit_bytes']==10*1024**3 and lim['min_system_available_bytes']==2*1024**3 and lim['startup_available_bytes']==12*1024**3 and lim['minimum_D_free_bytes']==4*1024**3,'Resource envelope differs');return v,axes

def approve(a):
    v,axes=load(Path(a.contract),a.contract_sha);rp=Path(a.review);require(rp.is_absolute() and rp.resolve().is_relative_to(R.resolve()) and base.sha(rp)==a.review_sha,'Exact root review required');review=read(rp)
    require(review.get('decision')=='APPROVED_FULL3006_SEEDS012_GRNBOOST2_CONTINUOUS_FAMILY' and review.get('contract')==rec(Path(a.contract)) and review.get('source')==rec(__file__) and review.get('scientific_pass') is False,'Source-bound full root approval required')
    require(os.name=='nt' and Path(sys.executable).resolve()==checked(v['family']['interpreter']).resolve(),'Exact native orthogonal environment required')
    for name,ver in v['family']['runtime_versions'].items():require(importlib.metadata.version(name)==ver,'Runtime drift: '+name)
    cache_env();return v,axes
_EXEC={};_W={}
def execution_ok(record,v):
    if record['sha256'] in _EXEC:return
    ex=read(checked(record));old=read(checked(ex['contract']));review=read(checked(ex['review']))
    require(ex['family_sha256']==old['family_sha256']==v['family_sha256'] and old['family']==v['family'] and old['source']==rec(__file__) and review.get('decision')=='APPROVED_FULL3006_SEEDS012_GRNBOOST2_CONTINUOUS_FAMILY' and review.get('contract')==ex['contract'] and review.get('source')==rec(__file__) and review.get('scientific_pass') is False,'Committed execution/root approval provenance differs');_EXEC[record['sha256']]=True

def committed(folder,task,v):
    cp=folder/'COMMIT.json'
    if not cp.exists():return None
    d=read(cp);require(d.get('schema')=='dynavelo_grnboost2_full_target_commit_v1' and d['family_sha256']==v['family_sha256'] and d['source']==rec(__file__) and d['task']==task and d['result_status'] in RESOLVED,'Exact completed task family/source/identity required');execution_ok(d['execution'],v)
    files={Path(x['relative_path']).name:x for x in d['files']};expect={'STARTED.json','receipt.json'}|({'all_predictors.tsv'} if d['result_status'] in FITTED else set())
    require(len(files)==len(d['files']) and set(files)==expect and {p.name for p in folder.iterdir()}==expect|{'COMMIT.json'},'Exact completed payload family required')
    for name,item in files.items():require(checked(item)==folder/name,'Payload escapes exact task folder')
    receipt=read(folder/'receipt.json');require(receipt['task']==task and receipt['status']==d['result_status'] and receipt['execution']==d['execution'] and receipt['scientific_pass'] is False,'Receipt/COMMIT binding differs')
    if receipt['status'] in FITTED:require(type(receipt.get('n_estimators')) is int and 1<=receipt['n_estimators']<=5000 and receipt['zeros_materialized_only_after_verified_success'] is True and receipt['edges_file']==files['all_predictors.tsv'],'Completed fit proof differs')
    else:require(receipt.get('edges_file') is None and 'n_estimators' not in receipt,'Constant/not-fit must not invent edges/fit')
    return {'task':task,'status':receipt['status'],'commit':rec(cp),'elapsed_seconds':receipt['elapsed_seconds']}

def worker_init(contract,output,execution):
    cache_env();v=read(contract);axes=read(checked(v['family']['axes']))
    import numpy as np
    import arboreto.core as core
    require(dict(core.SGBM_KWARGS)==PARAMS and core.EARLY_STOP_WINDOW_LENGTH==25 and rec(Path(core.__file__))==v['family']['official_arboreto_core'],'Actual official core/defaults differ')
    x=np.load(checked(v['family']['matrix']),mmap_mode='r',allow_pickle=False);require(x.shape==(6035,3006) and x.dtype==np.float32,'Frozen matrix differs');tf=x[:,axes['predictor_indices']];require(np.isfinite(tf).all(),'DATA_HOLD_NONFINITE_PREDICTOR')
    _W.update(v=v,axes=axes,x=x,tf=tf,core=core,output=Path(output),execution=execution,constant_predictor_count=int(np.all(tf==tf[0],axis=0).sum()))

def target_job(task):
    import numpy as np
    from threadpoolctl import threadpool_limits
    w=_W;folder=w['output']/'tasks'/task['task_id'];folder.mkdir(exist_ok=False);started=time.monotonic();target=task['target_stable_id'];y=w['x'][:,task['target_index']]
    new(folder/'STARTED.json',{'task':task,'family_sha256':w['v']['family_sha256'],'execution':w['execution'],'source':rec(__file__),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    result={'task':task,'execution':w['execution'],'constant_predictor_count_retained':w['constant_predictor_count'],'scientific_pass':False};messages=[]
    class Capture(logging.Handler):
        def emit(self,r):messages.append(r.getMessage())
    handler=Capture();logger=logging.getLogger('arboreto.core')
    try:
        if not np.isfinite(y).all():result.update(status='NOT_TESTED_NONFINITE_TARGET',edges_file=None)
        elif np.all(y==y[0]):result.update(status='NOT_TESTED_CONSTANT_TARGET',edges_file=None)
        else:
            logger.addHandler(handler)
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');edges,meta=w['core'].infer_partial_network(regressor_type='GBM',regressor_kwargs=dict(PARAMS),tf_matrix=w['tf'],tf_matrix_gene_names=w['axes']['predictor_stable_ids'],target_gene_name=target,target_gene_expression=y,include_meta=True,early_stop_window_length=25,seed=task['seed'])
            require(list(meta.columns)==['target','n_estimators'] and len(meta)==1 and meta.iloc[0]['target']==target and float(meta.iloc[0]['n_estimators']).is_integer() and 1<=int(meta.iloc[0]['n_estimators'])<=5000,'FAILED_LIBRARY_RETRY_FALLBACK_OR_UNPROVEN_FIT')
            require(list(edges.columns)==['TF','target','importance'] and not edges['TF'].duplicated().any() and set(edges['target'])<=set([target]),'Malformed official links')
            clean=[t for t in w['axes']['predictor_stable_ids'] if t!=target];values={str(row.TF):float(row.importance) for row in edges.itertuples(index=False)}
            require(set(values)<=set(clean) and all(math.isfinite(x) and x>0 for x in values.values()),'Invalid/self/unknown importance')
            tsv(folder/'all_predictors.tsv',[{'TF_stable_id':tf,'target_stable_id':target,'importance':values.get(tf,0.0)} for tf in clean])
            result.update(status='SUCCESS_GRNBOOST2_TARGET_AFTER_LIBRARY_RETRY' if messages else 'SUCCESS_GRNBOOST2_TARGET',n_estimators=int(meta.iloc[0]['n_estimators']),predictor_count_after_official_self_exclusion=len(clean),positive_edge_count=len(values),zero_importance_count=len(clean)-len(values),zeros_materialized_only_after_verified_success=True,edges_file=rec(folder/'all_predictors.tsv'),python_warnings=[str(x.message) for x in caught])
    except BaseException as e:result.update(status='HOLD_TARGET_INFERENCE_FAILED',error_type=type(e).__name__,error=str(e),edges_file=None)
    finally:logger.removeHandler(handler);result.update(library_retry_messages=messages,elapsed_seconds=time.monotonic()-started);new(folder/'receipt.json',result)
    if result['status'] in RESOLVED:
        files=[rec(folder/'STARTED.json'),rec(folder/'receipt.json')]+([result['edges_file']] if result['status'] in FITTED else [])
        new(folder/'COMMIT.json',{'schema':'dynavelo_grnboost2_full_target_commit_v1','family_sha256':w['v']['family_sha256'],'source':rec(__file__),'task':task,'result_status':result['status'],'execution':w['execution'],'files':files,'scientific_pass':False})
        return {'task':task,'status':result['status'],'commit':rec(folder/'COMMIT.json'),'elapsed_seconds':result['elapsed_seconds']}
    new(folder/'FAILED.json',{'task':task,'status':result['status'],'receipt':rec(folder/'receipt.json'),'silent_retry_authorized':False,'scientific_pass':False});return {'task':task,'status':result['status'],'receipt':rec(folder/'receipt.json'),'elapsed_seconds':result['elapsed_seconds']}
def child(a):
    v,axes=approve(a);require(a.run_id and a.run_id.replace('_','').replace('-','').isalnum() and len(a.run_id)<=40,'Unique short run-ID required');out=R/v['namespace_relative'];out.mkdir(parents=True,exist_ok=True);definition={'family':v['family'],'family_sha256':v['family_sha256']}
    if (out/'family.json').exists():require(read(out/'family.json')==definition,'Existing family differs; never reset tasks')
    else:new(out/'family.json',definition)
    (out/'tasks').mkdir(exist_ok=True);run=out/'runs'/a.run_id;run.mkdir(parents=True,exist_ok=False)
    ex={'family_sha256':v['family_sha256'],'contract':rec(Path(a.contract)),'review':rec(Path(a.review)),'source':rec(__file__),'run_id':a.run_id,'job_id':'orthogonal_grn_full_'+a.run_id,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};new(run/'execution.json',ex);exrec=rec(run/'execution.json')
    checked(v['family']['matrix']);alltasks=tasks(axes);byid={t['task_id']:t for t in alltasks};require({p.name for p in (out/'tasks').iterdir()}<=set(byid),'Unexpected task folder outside full family')
    rows={};pending=[]
    for t in alltasks:
        folder=out/'tasks'/t['task_id'];saved=committed(folder,t,v)
        if saved:rows[t['task_id']]=saved
        elif folder.exists():rows[t['task_id']]={'task':t,'status':'EXISTING_FAILED_NOT_RETRIED' if (folder/'FAILED.json').exists() else 'EXISTING_INCOMPLETE_NOT_RETRIED','original_failure_status':read(folder/'FAILED.json').get('status') if (folder/'FAILED.json').exists() else None,'failure_record':rec(folder/'FAILED.json') if (folder/'FAILED.json').exists() else None,'scientific_pass':False}
        else:pending.append(t)
    started=time.monotonic();initial_commits=sum('commit' in x for x in rows.values());new_receipts=0;first64={t['task_id'] for t in alltasks[:64]}
    def progress():
        counts=dict(Counter(r['status'] for r in rows.values()));commits=sum('commit' in r for r in rows.values());snapshot={'family_sha256':v['family_sha256'],'run_id':a.run_id,'expected_tasks':9018,'committed_resolved_tasks':commits,'new_target_receipts':new_receipts,'initial_verified_COMMITs':initial_commits,'statuses':counts,'still_pending_or_active':9018-len(rows),'workers':v['workers'],'elapsed_seconds':time.monotonic()-started,'continuous_dispatch_not_acceptance_gated':True,'scientific_pass':False};status(run/'progress.json',snapshot)
        if first64<=set(rows) and not (run/'first64_resource_observation.json').exists():
            job=R/'tmp_cache/local_owned_jobs'/ex['job_id']/'status.json'
            try:resource={k:x for k,x in read(job).items() if k in ('updated_utc','elapsed_seconds','rss_bytes','peak_rss_bytes','system_available_bytes','stop_reason')}
            except (OSError,ValueError):resource={'status':'OWNED_JOB_RESOURCE_SNAPSHOT_UNAVAILABLE'}
            new(run/'first64_resource_observation.json',{'scope':'fixed original task ordinals0..63, seed0 targets0..63','targets':[rows[t['task_id']] for t in alltasks[:64]],'resource_snapshot':resource,'observation_only_not_scientific_acceptance':True,'dispatch_continued_without_target_acceptance_gate':True,'scientific_pass':False})
        return snapshot
    progress()
    if pending:
        with multiprocessing.get_context('spawn').Pool(processes=v['workers'],initializer=worker_init,initargs=(str(a.contract),str(out),exrec)) as pool:
            for row in pool.imap_unordered(target_job,pending,chunksize=1):
                rows[row['task']['task_id']]=row;new_receipts+=1;snapshot=progress()
                if new_receipts%32==0:print(json.dumps({'status':'FULL_FAMILY_CONTINUOUS_PROGRESS',**snapshot}),flush=True)
    verified=[]
    for t in alltasks:
        saved=committed(out/'tasks'/t['task_id'],t,v)
        if saved:verified.append(saved)
    checked(v['family']['matrix']);load(Path(a.contract),a.contract_sha);counts=dict(Counter(r['status'] for r in verified));allresolved=len(verified)==9018
    result={'schema':'dynavelo_grnboost2_full_run_result_v1','status':'ALL9018_TARGET_SEED_TASKS_RESOLVED_CONSTANTS_EXPLICIT' if allresolved else 'INCOMPLETE_FULL_FAMILY_UNRESOLVED_TASKS_RETAINED','family_sha256':v['family_sha256'],'execution':exrec,'expected_tasks':9018,'committed_resolved_tasks':len(verified),'successful_fits':sum(r['status'] in FITTED for r in verified),'constant_targets_not_fitted':counts.get('NOT_TESTED_CONSTANT_TARGET',0),'all_targets_fitted':len(verified)==9018 and all(r['status'] in FITTED for r in verified),'remaining_unresolved':9018-len(verified),'verified_status_counts':counts,'runtime_status_counts':dict(Counter(r['status'] for r in rows.values())),'new_target_receipts':new_receipts,'initial_verified_COMMITs':initial_commits,'elapsed_seconds':time.monotonic()-started,'no_failed_or_incomplete_task_silently_retried':True,'all20_retained':True,'method_scope':'same cells, not independent donors; no rescue/clinical/directional claims','original_HOLD_NOT_TESTED_and_CPU_A100_bridge_unchanged':True,'scientific_pass':False};new(run/'RUN_COMPLETE.json',result)
    if allresolved:
        census={'family_sha256':v['family_sha256'],'tasks':verified,'expected_tasks':9018,'scientific_pass':False};dest=out/'complete_COMMIT_census.json'
        if dest.exists():require(read(dest)==census,'Existing complete census differs')
        else:new(dest,census)
        marker={'status':result['status'],'family_sha256':v['family_sha256'],'census':rec(dest),'successful_fits':result['successful_fits'],'constant_targets_not_fitted':result['constant_targets_not_fitted'],'all_targets_fitted':result['all_targets_fitted'],'scientific_pass':False}
        if (out/'FULL_COMPLETE.json').exists():require(read(out/'FULL_COMPLETE.json')==marker,'Existing full completion differs')
        else:new(out/'FULL_COMPLETE.json',marker)
    return result

def supervise(a):
    v,axes=approve(a)
    import psutil
    require(a.run_id and a.run_id.replace('_','').replace('-','').isalnum() and len(a.run_id)<=40,'Unique short run-ID required');require(psutil.virtual_memory().available>=v['limits']['startup_available_bytes'],'Need actual12GiB available before reviewed10GiB job');require(shutil.disk_usage(R).free>=v['limits']['minimum_D_free_bytes'],'Need4GiB D free before full export outputs')
    sys.path.insert(0,str(R/'repo/pipeline'));import supervise_local_resume as owner
    original_entry,original_job=owner.ENTRY,owner.OwnedJob
    class BoundedJob(original_job):
        def __init__(self):
            super().__init__();lim=owner.ExtendedLimits();lim.BasicLimitInformation.LimitFlags=0x2000|0x100|0x200;lim.ProcessMemoryLimit=lim.JobMemoryLimit=v['limits']['max_job_commit_bytes']
            if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(lim),ctypes.sizeof(lim)):self.close();raise ctypes.WinError(ctypes.get_last_error())
            actual=owner.ExtendedLimits()
            if not self.k.QueryInformationJobObject(self.handle,9,ctypes.byref(actual),ctypes.sizeof(actual),None) or actual.ProcessMemoryLimit!=lim.ProcessMemoryLimit or actual.JobMemoryLimit!=lim.JobMemoryLimit or actual.BasicLimitInformation.LimitFlags&0x2300!=0x2300:self.close();raise RuntimeError('Kernel10GiB job limits not bound')
    jid='orthogonal_grn_full_'+a.run_id;job=R/'tmp_cache/local_owned_jobs'/jid;require(not job.exists(),'Used resource-owner run-ID cannot be replayed')
    command=[str(checked(v['family']['interpreter'])),'-B',str(Path(__file__)),'child','--contract',a.contract,'--contract-sha',a.contract_sha,'--review',a.review,'--review-sha',a.review_sha,'--run-id',a.run_id]
    spec={'job_id':jid,'job_dir':str(job),'command':command,'python_sha256':v['family']['interpreter']['sha256'],'entry_sha256':v['source']['sha256'],'manifest_sha256':a.contract_sha,'supervisor_sha256':v['source']['sha256'],'wall_seconds':v['limits']['wall_seconds'],'max_rss_bytes':v['limits']['max_RSS_bytes'],'min_free_bytes':v['limits']['min_system_available_bytes'],'stop_file':str(job/'STOP')}
    try:owner.ENTRY=Path(__file__);owner.OwnedJob=BoundedJob;result=owner.supervise(spec)
    finally:owner.ENTRY=original_entry;owner.OwnedJob=original_job
    receipt={'status':'FULL_RUN_EXITED_ROOT_REVIEW_PENDING' if result['exit_code']==0 and result['owned_job_empty'] and not result['errors'] and not result['stop_reason'] else 'HOLD_FULL_RUN_INCOMPLETE_EXISTING_COMMITS_PRESERVED','owned_job_result':result,'family_sha256':v['family_sha256'],'contract':rec(Path(a.contract)),'source':rec(__file__),'other_user_processes_touched':False,'scientific_pass':False};new(job/'full_supervision_receipt.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','metadata','supervise','child'])
    for k in ('contract','contract-sha','review','review-sha','run-id'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.mode=='prepare':value=prepare()
    else:
        require(a.contract and a.contract_sha,'Exact contract arguments required')
        if a.mode=='metadata':v,axes=load(Path(a.contract),a.contract_sha);value={'status':'METADATA_ONLY_NO_EXPRESSION_OR_IMPORTANCE_VALUES','contract':rec(Path(a.contract)),'expected_tasks':9018,'default_workers':v['workers'],'wall_seconds':v['limits']['wall_seconds'],'family_sha256':v['family_sha256']}
        else:require(a.review and a.review_sha and a.run_id,'Exact root review and unique resource run-ID required');value=supervise(a) if a.mode=='supervise' else child(a)
    print(json.dumps(value,allow_nan=False),flush=True)
if __name__=='__main__':multiprocessing.freeze_support();main()
