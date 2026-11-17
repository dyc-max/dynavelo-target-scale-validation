"""Export only frozen PRIMARY192 RNA vx all-group means; default metadata only.

No models, GRNBoost2 files, candidate selection, or cloud operations are imported.
All consumed source bytes are SHA-checked before parsing. Each NPZ is read once.
"""
import argparse, hashlib, io, json, os, time
from pathlib import Path, PurePosixPath

ROOT = Path(r'D:\dyc_7\DynaVelo_Research')
INDEX = 'results/reports/primary_only_grn_descriptive_20261116_v1/PRIMARY_backed_grn_index.json'
INDEX_SHA = '9cb35ac923cd162cb77e4dffc6e85a06bbedc70f3ff45d82047d12682d81b301'
OUT = ROOT/'results/reports/primary_global_diagnostic_cloud_20261116_v1/input_package'
NS = 'results/precision_runs_v1/fp64_20261115_v1'
SCOPE = 'A100_PRIMARY192_ALL1191_RNA_INPUTS_EFFECT_POPULATION_ALL_VX'
FIELDS = ('sum','abs_sum','count','positive','negative','zero','nonfinite','weighted_signed_sum','weighted_absolute_sum','weighted_finite_mass','weighted_nonfinite_mass')
INTS = {'count','positive','negative','zero','nonfinite'}

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def path(s):
    require(isinstance(s,str) and s and '\\' not in s and ':' not in s,'Canonical relative project path required')
    q=PurePosixPath(s)
    require(not q.is_absolute() and all(x not in ('.','..') for x in q.parts) and q.as_posix()==s,'Unsafe project path')
    p=ROOT.joinpath(*q.parts)
    require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(),'Resolved project path required')
    return p

def record(p):
    p=Path(p).resolve(); require(p.is_relative_to(ROOT.resolve()),'Project D path required')
    b=p.read_bytes()
    return {'relative_path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':digest(b)}

def checked(rec):
    require(set(rec)=={'relative_path','bytes','sha256'} and type(rec['bytes']) is int,'Artifact record required')
    b=path(rec['relative_path']).read_bytes()
    require(len(b)==rec['bytes'] and digest(b)==rec['sha256'],'Source size/SHA mismatch: '+rec['relative_path'])
    return b

def obj(rec): return json.loads(checked(rec))
def write(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def seal(x,key): require(digest(canonical({k:v for k,v in x.items() if k!=key}))==x[key],'Logical seal mismatch: '+key)
def portable(p):
    r=record(p);return {'path':p.name,'bytes':r['bytes'],'sha256':r['sha256']}

def compact_proof(i,src,attempt):
    return {'input_index':i,'seed':src['task']['seed'],'task_id':src['task']['task_id'],'attempt':attempt,'payloads':{key:{k:v for k,v in src[key].items() if k!='relative_path'} for key in ('commit','result','edge_axes','vx_edges')},'parent_baseline_task_id':src['parent_baseline_dependency']['task_id']}

def metadata():
    b=path(INDEX).read_bytes();require(digest(b)==INDEX_SHA,'Frozen complete PRIMARY source index differs')
    idx=json.loads(b);irec={'relative_path':INDEX,'bytes':len(b),'sha256':INDEX_SHA}
    require(idx['status']=='ALL4668_PRIMARY_COMMITS_AND_PARENTS_VERIFIED_DESCRIPTIVE_ONLY' and idx['scientific_pass'] is False,'Verified PRIMARY-only index required')
    require(idx['scope']=={'stage':'grn','sample_id':'primary','n_cells':192,'epsilon':0.0001,'seeds':[0,1,2],'inputs':{'rna':1191,'motif':365},'primary_tasks':4668,'baselines':12,'epsilon_history':846,'fullcell_completed':28,'fullcell_expected':84,'missing_fullcell':56},'Frozen source scope differs')
    original=obj(idx['original_scope_plan']);genes=original['context_plan']['spec']['genes'];eligible=[g for g in genes if g['eligible_tf']]
    rna=[x for x in idx['inputs'] if x['input_space']=='rna']
    require(len(rna)==1191 and len(genes)==3006,'Full RNA denominator required')
    require([(x['feature_id'],x['tf_stable_id']) for x in rna]==[(g['feature_id'],g['stable_id']) for g in eligible],'Native eligible RNA order mismatch')
    require(idx['rna_target_axis']==[{'feature_id':g['feature_id'],'stable_id':g['stable_id']} for g in genes],'Original RNA target axis mismatch')
    locked={x['stable_id'] for x in idx['frozen20_candidates']};require(len(locked)==20 and locked<={g['stable_id'] for g in eligible},'Frozen20 identity mismatch')
    axes={'schema':'dynavelo_primary_global_diagnostic_axes_v1','inputs':[{'index':i,'stable_id':g['stable_id'],'gene':g['feature_id'],'locked20':g['stable_id'] in locked} for i,g in enumerate(eligible)],'targets':[{'index':i,'stable_id':g['stable_id'],'gene':g['feature_id']} for i,g in enumerate(genes)]}
    tasks=[s for x in rna for s in x['seed_tasks']]
    require(len(tasks)==3573 and len({s['task']['task_id'] for s in tasks})==3573,'3573 unique RNA tasks required')
    for x in rna:
        require(sorted(s['task']['seed'] for s in x['seed_tasks'])==[0,1,2],'Exact three seeds required')
        for s in x['seed_tasks']:
            t=s['task'];require(t['stage']=='grn' and t['input_space']=='rna' and t['sample_id']=='primary' and t['n_cells']==192 and t['feature_id']==x['feature_id'] and t['tf_stable_id']==x['tf_stable_id'],'Task-index identity mismatch')
    proof_estimate=[compact_proof(i,s,int(PurePosixPath(s['commit']['relative_path']).parent.name.split('_')[1])) for i,x in enumerate(rna) for s in x['seed_tasks']]
    metadata_estimate=len(canonical(proof_estimate))+len(canonical(axes))+100000
    info={'status':'METADATA_ONLY_NO_REAL_VALUES_READ','builder':record(Path(__file__)),'source_index':irec,'tasks':3573,'matrix_raw_bytes':3*1191*3006*8,'matrix_npy_bytes_estimate':3*(1191*3006*8+128),'source_npz_bytes_single_pass':sum(s['vx_edges']['bytes'] for s in tasks),'metadata_estimate_bytes':metadata_estimate,'package_estimate_bytes':3*(1191*3006*8+128)+metadata_estimate,'max_local_memory_bytes':1024**3,'threads':1,'scientific_pass':False,'scope':SCOPE}
    return idx,rna,axes,info

def extract_mean(blob,groups):
    import numpy as np
    selected=[i for i,g in enumerate(groups) if g['kind']=='effect_population' and g['name']=='all']
    require(len(selected)==1,'Exactly one effect_population/all row required');gi=selected[0]
    require(groups[gi]['sample_n']==192 and groups[gi]['effect_frame_domain_n']==1884 and groups[gi]['sample_weight_sum']==1.0,'PRIMARY group context differs')
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        require(set(z.files)==set(FIELDS) and len(z.files)==len(FIELDS),'Exact original eleven statistics required')
        arrays={k:z[k] for k in FIELDS}
    for k,a in arrays.items():
        require(a.shape==(len(groups),3006) and a.dtype==np.dtype('int64' if k in INTS else 'float64') and np.isfinite(a).all(),'Statistic shape/dtype/finiteness: '+k)
        if k in INTS or k in ('abs_sum','weighted_absolute_sum','weighted_finite_mass','weighted_nonfinite_mass'): require((a>=0).all(),'Negative count/mass: '+k)
    a=arrays
    require(np.array_equal(a['positive']+a['negative']+a['zero'],a['count']),'Sign counts do not close')
    require(np.array_equal(a['count']+a['nonfinite'],np.broadcast_to(np.array([g['sample_n'] for g in groups])[:,None],a['count'].shape)),'Sample counts differ')
    require(np.allclose(a['weighted_finite_mass']+a['weighted_nonfinite_mass'],np.array([g['sample_weight_sum'] for g in groups])[:,None],rtol=1e-12,atol=1e-12),'Weighted mass differs')
    require(not np.any(a['nonfinite']) and not np.any(a['weighted_nonfinite_mass']),'Completed source contains nonfinite derivatives')
    require(np.all(a['sum'][a['count']==0]==0) and np.all(a['abs_sum'][a['count']==0]==0),'Empty count has nonzero sum')
    require(np.all(a['weighted_signed_sum'][a['weighted_finite_mass']==0]==0) and np.all(a['weighted_absolute_sum'][a['weighted_finite_mass']==0]==0),'Empty mass has nonzero sum')
    mass=a['weighted_finite_mass'][gi]
    return np.divide(a['weighted_signed_sum'][gi],mass,out=np.full(3006,np.nan,dtype=np.float64),where=mass>0)

def export(idx,rna,axes,info,approval_path):
    import numpy as np, psutil
    ap=Path(approval_path).resolve();require(ap.is_relative_to(ROOT.resolve()) and ap.is_file(),'Root approval must be a project file')
    approval_rec=record(ap);approval=obj(approval_rec)
    require(approval.get('decision')=='APPROVED_PRIMARY_GLOBAL_DIAGNOSTIC_INPUT_EXPORT_V1' and approval.get('builder')==info['builder'] and approval.get('source_index')==info['source_index'] and approval.get('scientific_pass') is False,'Explicit root review of exact builder/index required')
    require(not OUT.exists(),'Refusing to overwrite input package');OUT.mkdir(parents=True)
    write(OUT/'BUILD_STARTED.json',{'schema':'dynavelo_input_export_started_v1','root_approval':approval_rec,'builder':info['builder'],'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    cover={k:obj(idx['coverage_records'][k]) for k in ('grn','baseline')}
    for c in cover.values(): require(c['all_expected_tasks_complete'] is True,'Complete original coverage required')
    lookup={k:{t['task_id']:t for t in v['tasks']} for k,v in cover.items()}
    matrices={s:np.empty((1191,3006),dtype=np.float64) for s in (0,1,2)};proof=[];parents={};contexts={}
    for i,x in enumerate(rna):
        for src in sorted(x['seed_tasks'],key=lambda a:a['task']['seed']):
            tsmall=src['task'];tid=tsmall['task_id'];seed=tsmall['seed'];row=lookup['grn'][tid]
            require(row['status']=='COMPLETE' and src['commit']['relative_path']==NS+'/grn/'+row['manifest_relative_path'] and src['commit']['sha256']==row['manifest_sha256'],'Original selected COMMIT mismatch')
            com=obj(src['commit']);t=com['task'];seal({k:v for k,v in t.items() if k!='assigned_shard'},'task_id')
            require(com['schema']=='dynavelo_task_commit_v1' and com['status']=='COMPLETE' and com['book_sha256']==cover['grn']['book_sha256'] and com['files']==row['files'] and 1<=com['attempt']<=3,'COMMIT contract differs')
            require(all(t[k]==v for k,v in tsmall.items()) and t['precision']=='float64' and t['epsilon']==.0001,'COMMIT task identity differs')
            files={f['name']:f for f in com['files']};require(len(files)==len(com['files']),'Duplicate COMMIT payload name')
            for key,name in (('result','result.json'),('edge_axes','edge_axes.json'),('vx_edges','vx_edges.npz')):
                f=files[name];require(src[key]=={'relative_path':str(PurePosixPath(src['commit']['relative_path']).parent/name),'bytes':f['bytes'],'sha256':f['sha256']},'Payload detached from COMMIT: '+name)
            result=obj(src['result']);ax=obj(src['edge_axes']);dep=src['parent_baseline_dependency']
            require(all(result.get(k)==v for k,v in tsmall.items()) and result['status']=='COMPLETE' and result['numerical_integrity']=='PASS' and result['scientific_pass'] is False and result['precision_dtype']=='float64','Result identity/status differs')
            require(result['parent_baseline_dependency']==dep and result['matched_baseline_dependency']==dep and t['parent_baseline_task_id']==dep['task_id'],'Parent binding differs')
            pid=dep['task_id']
            if pid not in parents:
                pr=lookup['baseline'][pid];require(pr['status']=='COMPLETE' and pr['manifest_sha256']==dep['commit_sha256'] and cover['baseline']['book_sha256']==dep['book_sha256'],'Baseline coverage differs')
                pp=path(NS+'/baseline/'+pr['manifest_relative_path']);pb=pp.read_bytes();require(digest(pb)==dep['commit_sha256'],'Parent COMMIT SHA differs');pc=json.loads(pb)
                require(pc['status']=='COMPLETE' and pc['book_sha256']==dep['book_sha256'] and pc['task']['task_id']==pid,'Parent COMMIT identity differs');seal({k:v for k,v in pc['task'].items() if k!='assigned_shard'},'task_id')
                parents[pid]={'record':{'relative_path':pp.relative_to(ROOT).as_posix(),'bytes':len(pb),'sha256':digest(pb)},'task':pc['task']}
            pt=parents[pid]['task']
            for key in ('seed','checkpoint_sha256','sample_id','sample_cell_ids','sample_cell_axis_sha256','n_cells','batches','chunk_geometry'): require(t[key]==pt[key],'Parent context differs: '+key)
            groups=idx['group_axes']['grn__seed'+str(seed)]
            require(ax['groups']==groups and ax['rna_output_axis']==[g['gene'] for g in axes['targets']] and ax['input_space']=='rna' and ax['input_feature_id']==x['feature_id'],'NPZ axes detached from task')
            context={k:t[k] for k in ('sample_id','n_cells','sample_cell_ids','sample_cell_axis_sha256','sample_weights_sha256','checkpoint_sha256','batches','chunk_geometry','epsilon','precision')};context['groups']=groups
            if seed in contexts: require(contexts[seed]==context,'Within-seed sample/context changed')
            else: contexts[seed]=context
            matrices[seed][i]=extract_mean(checked(src['vx_edges']),groups)
            require(PurePosixPath(src['commit']['relative_path']).parent.as_posix()==NS+'/grn/tasks/'+tid+'/attempt_'+format(com['attempt'],'03d'),'Source directory template differs')
            proof.append(compact_proof(i,src,com['attempt']))
        if i%64==0:
            require(psutil.Process().memory_info().rss<1024**3,'Local memory ceiling exceeded');print(json.dumps({'rows_done':i+1,'all_rows':1191}),flush=True)
    require(len(proof)==3573,'Export incomplete')
    write(OUT/'axes.json',axes)
    provenance={'schema':'dynavelo_primary_global_diagnostic_provenance_v1','scope':SCOPE,'source_index':info['source_index'],'builder':info['builder'],'root_approval':approval_rec,'source_tasks':proof,'source_directory_template':NS+'/grn/tasks/{task_id}/attempt_{attempt:03d}','payload_filenames':{'commit':'COMMIT.json','result':'result.json','edge_axes':'edge_axes.json','vx_edges':'vx_edges.npz'},'source_grn_book_sha256':cover['grn']['book_sha256'],'source_baseline_book_sha256':cover['baseline']['book_sha256'],'parent_commits':parents,'seed_contexts':contexts,'original_scope_plan':idx['original_scope_plan'],'original_coverage_records':idx['coverage_records'],'definition':'Original float64 weighted_signed_sum / weighted_finite_mass; unique effect_population/all row; no averaging across technical seeds','freshly_verified':'Selected COMMITs, result/axes/vx payloads and parent COMMIT identity; parent baseline arrays rely on completed source-index audit','qualification_policy':idx['qualification_policy'],'CPU_to_A100_bridge':idx['CPU_to_A100_bridge'],'scientific_pass':False,'new_candidate_screen':False,'algorithmic_not_biological_validation':True}
    write(OUT/'provenance.json',provenance);mrec=[]
    for seed,a in matrices.items():
        p=OUT/('weighted_signed_mean_seed'+str(seed)+'.npy')
        with p.open('xb') as f: np.save(f,a,allow_pickle=False)
        mrec.append(dict(portable(p),seed=seed,shape=[1191,3006],dtype='float64'))
    contract={'schema':'dynavelo_primary_global_diagnostic_input_v1','scope':SCOPE,'matrices':mrec,'axes':portable(OUT/'axes.json'),'provenance':portable(OUT/'provenance.json'),'scientific_pass':False,'new_candidate_screen':False,'algorithmic_not_biological_validation':True,'context':{'sample':'PRIMARY192','group_kind':'effect_population','group_name':'all','channel':'vx','statistic':'weighted_signed_mean','seeds':'technical_not_independent_donors'},'qualification_policy':idx['qualification_policy']}
    write(OUT/'contract.json',contract);complete={'status':'COMPLETE_INPUT_EXPORT_NO_CLOUD_EXECUTION','contract':portable(OUT/'contract.json'),'payload_bytes':sum(r['bytes'] for r in mrec)+contract['axes']['bytes']+contract['provenance']['bytes'],'scientific_pass':False}
    write(OUT/'BUILD_COMPLETE.json',complete);return complete

def main():
    cache=ROOT/'tmp_cache/primary_global_diagnostic_cloud_v1';cache.mkdir(parents=True,exist_ok=True)
    for k in ('TEMP','TMP','TMPDIR'): os.environ[k]=str(cache)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('metadata','export'),default='metadata');p.add_argument('--root-approval');a=p.parse_args()
    idx,rna,axes,info=metadata()
    if a.mode=='metadata': print(json.dumps(info,indent=2));return
    require(bool(a.root_approval),'Export needs explicit root approval file')
    print(json.dumps(export(idx,rna,axes,info,a.root_approval),indent=2))

if __name__=='__main__': main()
