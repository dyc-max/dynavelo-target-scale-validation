"""Frozen6035x3006 RNA-X export and first8/seed0 GRNBoost2 resource pilot; metadata by default."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,importlib.metadata,json,logging,math,multiprocessing,os,sys,time,warnings
R=Path(r'D:\dyc_7\DynaVelo_Research');C=R/'repo/a100_production_20261114';P=R/'results/reports/orthogonal_GRN_preparation_20261116_v1';O=P/'native_pilot_v1'
ENV=R/'repo/.venv_orthogonal_grn_v1';PARAMS={'learning_rate':.01,'n_estimators':5000,'max_features':.1,'subsample':.9}

def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def rec(p):
    p=Path(p).resolve();require(p.is_relative_to(R.resolve()),'All paths must remain on D project')
    return {'relative_path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(v):
    p=R/v['relative_path'];require(p.resolve().is_relative_to(R.resolve()) and not p.is_symlink() and rec(p)==v,'Artifact bytes/SHA differs: '+v['relative_path']);return p

def new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def tsv(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def cache_env():
    for k in ('TMP','TEMP','TMPDIR','PIP_CACHE_DIR','HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','JOBLIB_TEMP_FOLDER','NUMBA_CACHE_DIR'):
        p=R/'tmp_cache/orthogonal_grn_v1'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')

def make_axes(scope,locked):
    s=scope['context_plan']['spec'];genes=s['genes'];cells=s['cells'];selected={x['stable_id'] for x in locked['selected']}
    require(len(genes)==3006 and len(cells)==6035 and len(s['motifs'])==365 and len({x['feature_id'] for x in genes})==len({x['stable_id'] for x in genes})==3006 and len({x['cell_id'] for x in cells})==6035,'Frozen unique model axes required')
    bysymbol={g['feature_id']:g['stable_id'] for g in genes};reg=[i for i,g in enumerate(genes) if g['eligible_tf']]
    require(len({x['label'] for x in cells})==13 and len(reg)==1191 and len(selected)==20 and selected<=set(g['stable_id'] for g in genes if g['eligible_tf']),'All1191 predictors and all20 retained')
    return {'schema':'dynavelo_orthogonal_grn_native_axes_v1','cells':[{'index':i,**{k:x[k] for k in ('cell_id','label','split')}} for i,x in enumerate(cells)],'genes':[{'index':i,'stable_id':g['stable_id'],'symbol':g['feature_id'],'eligible_tf':g['eligible_tf'],'locked20':g['stable_id'] in selected} for i,g in enumerate(genes)],'predictor_indices':reg,'predictor_stable_ids':[genes[i]['stable_id'] for i in reg],'motifs':[{'index':i,'motif_feature':m['feature_id'],'TF_symbol':m['tf_rna_feature_id'],'TF_stable_id':bysymbol[m['tf_rna_feature_id']]} for i,m in enumerate(s['motifs'])],'pilot_target_indices':list(range(8)),'future_target_indices':list(range(3006)),'future_seeds':[0,1,2],'symbol_collapse':False,'cisTarget_mapping_status':'NOT_TESTED; one-to-one symbol table only','independent_biological_validation':False}

def prepare():
    require(not O.exists(),'Fresh metadata-only preparation namespace required');O.mkdir(parents=True)
    scopep=R/'results/reports/a100_production_20261114/scope_plan.json';scope=read(scopep);s=scope['context_plan']['spec'];pp=read(R/'results/reports/a100_production_20261114/precision_amendment_v1/precision_plan.json');locked=read(checked(pp['shortlist']));axes=make_axes(scope,locked)
    new(O/'axes.json',axes);tsv(O/'cells.tsv',axes['cells']);tsv(O/'genes_stableID_symbol.tsv',axes['genes']);tsv(O/'predictors1191.tsv',[axes['genes'][i] for i in axes['predictor_indices']]);tsv(O/'locked20.tsv',[g for g in axes['genes'] if g['locked20']]);tsv(O/'motifs365.tsv',axes['motifs'])
    install=P/'pip_install_attempt1.json';packages={x['metadata']['name']:x['metadata']['version'] for x in read(install)['install']}
    core=ENV/'Lib/site-packages/arboreto/core.py';require(core.exists(),'Installed official Arboreto source required before freeze')
    sourcepaths=[R/'repo/pipeline/prepare_velocity.py',R/'repo/pipeline/velocity_plan.json',R/'repo/pipeline/assemble_model_inputs.py',R/'repo/pipeline/review_minimal_model_transfer.py',R/'repo/pipeline/supervise_local_resume.py',R/'repo/colab/screen_dispatch.py',R/'repo/colab/dynavelo_streaming_core.py',R/'repo/.venv/Lib/site-packages/scvelo/preprocessing/utils.py']
    contract={'schema':'dynavelo_orthogonal_grnboost2_pilot_contract_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'PREPARED_ROOT_REVIEW_REQUIRED_NO_EXPRESSION_READ','source':rec(__file__),'scope_plan':rec(scopep),'shortlist':pp['shortlist'],'axes':rec(O/'axes.json'),'metadata_tables':[rec(O/n) for n in ('cells.tsv','genes_stableID_symbol.tsv','predictors1191.tsv','locked20.tsv','motifs365.tsv')],'rna':s['input_artifacts']['rna'],'logical_RNA_X_sha256':s['global_minima']['rna_matrix_sha256'],'input_columns':s['dispatch_contract']['input_columns'],'preprocessing_source_records':[rec(p) for p in sourcepaths],'minimal_recipe':rec(R/'data/processed/colab_minimal/HSPC/minimization_recipe.json'),'minimal_independent_review':rec(R/'results/reports/minimal_model_transfer_independent_review.json'),'official_arboreto_core':rec(core),'interpreter':rec(ENV/'Scripts/python.exe'),'runtime_install_report':rec(install),'runtime_versions':packages,'parameters':PARAMS,'early_stop_window_length':25,'seed':0,'targets':axes['genes'][:8],'workers':2,'BLAS_threads_per_worker':1,'scheduler':'native multiprocessing spawn; no Dask client/graph','resource_limits':{'wall_seconds':600,'max_RSS_bytes':6*1024**3,'max_job_commit_bytes':6*1024**3,'min_system_available_bytes':2*1024**3,'startup_available_bytes':6*1024**3},'input_output_relative':'data/processed/orthogonal_grn_v1/frozen_RNA_X_v1','pilot_output_relative':'results/orthogonal_grn_v1/first8_seed0_v1','job_id':'orthogonal_grn_first8_seed0_v1','value_transform':'none; frozen float32 RNA.X, original cell/gene order; not counts/layers/log1p again','future_scope':'all3006 original targets xseeds012/all1191 original predictors requires separate full-run review','constant_target':'NOT_TESTED_CONSTANT_TARGET, no fabricated zero edges','nonfinite_input':'DATA_HOLD, no imputation/row deletion','fallback_policy':'include_meta=True; exact one-target n_estimators>=1 proves fitted model; empty fallback is failure, never zero edges','method_scope':'same-cell algorithm corroboration only; no independent donor/clinical validation','all20_retained':True,'E1_remains_previously_consumed':True,'DC_subtype_mapping_promoted':False,'scientific_pass':False,'model_calls_during_preparation':0,'expression_values_read_during_preparation':False}
    new(O/'contract.json',contract);return {'status':contract['status'],'contract':rec(O/'contract.json'),'label_counts':dict(Counter(x['label'] for x in axes['cells']))}

def load_contract(p,digest):
    require(p.is_absolute() and p.resolve().is_relative_to(R.resolve()) and sha(p)==digest,'Exact D contract path/SHA required');v=read(p)
    require(v['schema']=='dynavelo_orthogonal_grnboost2_pilot_contract_v1' and v['source']==rec(__file__) and v['parameters']==PARAMS and v['early_stop_window_length']==25 and v['seed']==0 and v['workers']==2 and v['BLAS_threads_per_worker']==1,'Frozen method/pilot changed')
    for item in [v['scope_plan'],v['shortlist'],v['axes'],v['official_arboreto_core'],v['interpreter'],v['runtime_install_report'],v['minimal_recipe'],v['minimal_independent_review']]+v['preprocessing_source_records']+v['metadata_tables']:checked(item)
    require(read(checked(v['axes']))==make_axes(read(checked(v['scope_plan'])),read(checked(v['shortlist']))),'Metadata axis derivation changed')
    return v

def approved(a):
    v=load_contract(Path(a.contract),a.contract_sha);p=Path(a.review);require(p.is_absolute() and p.resolve().is_relative_to(R.resolve()) and sha(p)==a.review_sha,'Exact root review path/SHA required');review=read(p)
    require(review.get('decision')=='APPROVED_FROZEN_RNA_X_EXPORT_AND_FIRST8_GRNBOOST2_RESOURCE_PILOT' and review.get('contract')==rec(Path(a.contract)) and review.get('source')==rec(__file__) and review.get('scientific_pass') is False,'Source-bound root review required before expression access/algorithm launch')
    require(os.name=='nt' and Path(sys.executable).resolve()==checked(v['interpreter']).resolve(),'Exact native orthogonal environment required')
    for name,version in v['runtime_versions'].items():require(importlib.metadata.version(name)==version,'Installed package drift: '+name)
    cache_env();return v
def array_digest(x):
    import numpy as np
    h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(x.shape).encode())
    for start in range(0,len(x),256):h.update(np.ascontiguousarray(x[start:start+256]).tobytes())
    return h.hexdigest()

def h5_values(node):
    import h5py,numpy as np
    if isinstance(node,h5py.Group):
        codes=node['codes'][()];cats=h5_values(node['categories']);require(np.all(codes>=0),'Missing categorical metadata');return [cats[int(i)] for i in codes]
    return [v.decode('utf-8') if isinstance(v,bytes) else v.item() if hasattr(v,'item') else v for v in node[()]]

def export_input(a):
    v=approved(a);axes=read(checked(v['axes']));out=R/v['input_output_relative'];require(not out.exists(),'Immutable new input export namespace required');rna=checked(v['rna'])
    import h5py,numpy as np
    from scipy.sparse import csr_matrix
    out.mkdir(parents=True);partial=out/'RNA_X.partial.npy';target=out/'RNA_X.npy'
    try:
        with h5py.File(rna,'r') as f:
            obs,var=f['obs'],f['var'];index=lambda node:h5_values(node[node.attrs['_index']])
            require(index(obs)==[c['cell_id'] for c in axes['cells']] and index(var)==[g['symbol'] for g in axes['genes']],'HDF5 native cell/gene order differs')
            for column,key in ((v['input_columns']['label'],'label'),(v['input_columns']['split'],'split')):require(h5_values(obs[column])==[c[key] for c in axes['cells']],'HDF5 label/split differs')
            require(h5_values(var[v['input_columns']['stable_id']])==[g['stable_id'] for g in axes['genes']] and h5_values(var[v['input_columns']['eligible_tf']])==[g['eligible_tf'] for g in axes['genes']],'Stable IDs/eligible flags differ')
            node=f['X'];shape=(6035,3006);dense=isinstance(node,h5py.Dataset)
            require(tuple(node.shape if dense else node.attrs['shape'])==shape and (node.dtype if dense else node['data'].dtype)==np.dtype('float32'),'Exact original FP32 X schema required')
            x=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=shape);ptr=None if dense else node['indptr'][()]
            if not dense:require(node.attrs['encoding-type']=='csr_matrix','Original CSR encoding required; no alternate layer')
            for start in range(0,shape[0],256):
                stop=min(start+256,shape[0])
                if dense:block=node[start:stop]
                else:
                    lo,hi=int(ptr[start]),int(ptr[stop]);block=csr_matrix((node['data'][lo:hi],node['indices'][lo:hi],ptr[start:stop+1]-lo),shape=(stop-start,shape[1])).toarray()
                require(block.dtype==np.float32 and np.isfinite(block).all() and not (block<0).any(),'DATA_HOLD_NONFINITE_OR_NEGATIVE_OR_DTYPE_CHANGED_X');x[start:stop]=block
        x.flush();logical=array_digest(x);require(logical==v['logical_RNA_X_sha256'],'Frozen full RNA-X logical hash differs; no transformed input accepted');del x
        os.link(partial,target);partial.unlink();checked(v['rna'])
        manifest={'schema':'dynavelo_grnboost2_frozen_RNA_X_export_v1','status':'EXACT_FROZEN_RNA_X_EXPORTED','source':rec(__file__),'contract':rec(Path(a.contract)),'rna':v['rna'],'axes':v['axes'],'matrix':rec(target),'shape':[6035,3006],'dtype':'float32','logical_sha256':logical,'normalization_performed':False,'symbol_collapse':False,'all1191_predictors_retained':True,'all20_retained':True,'scientific_pass':False}
        new(out/'manifest.json',manifest);return manifest
    except BaseException as e:
        new(out/'HOLD.json',{'status':'DATA_HOLD_INPUT_EXPORT_INCOMPLETE','error_type':type(e).__name__,'error':str(e),'scientific_pass':False});raise

_W={}
def worker_init(contract,manifest,output):
    cache_env();v=read(contract);m=read(manifest);axes=read(checked(v['axes']))
    import numpy as np
    import arboreto.core as core
    require(dict(core.SGBM_KWARGS)==PARAMS and core.EARLY_STOP_WINDOW_LENGTH==25,'Official defaults differ')
    x=np.load(checked(m['matrix']),mmap_mode='r',allow_pickle=False);require(x.shape==(6035,3006) and x.dtype==np.float32,'Frozen matrix shape/dtype differs')
    tf=x[:,axes['predictor_indices']];require(np.isfinite(tf).all(),'DATA_HOLD_NONFINITE_PREDICTOR');_W.update(x=x,tf=tf,axes=axes,output=Path(output),core=core,constant_predictor_count=int(np.all(tf==tf[0],axis=0).sum()))

def target_job(index):
    import numpy as np
    from threadpoolctl import threadpool_limits
    w=_W;axes=w['axes'];target=axes['genes'][index]['stable_id'];y=w['x'][:,index];folder=w['output']/('target_%04d'%index);folder.mkdir(exist_ok=False);started=time.monotonic()
    result={'target_index':index,'target_stable_id':target,'target_symbol':axes['genes'][index]['symbol'],'seed':0,'constant_predictor_count_retained':w['constant_predictor_count'],'scientific_pass':False};messages=[]
    class Capture(logging.Handler):
        def emit(self,r):messages.append(r.getMessage())
    handler=Capture();logger=logging.getLogger('arboreto.core')
    try:
        if not np.isfinite(y).all():result.update(status='NOT_TESTED_NONFINITE_TARGET',edges_file=None)
        elif np.all(y==y[0]):result.update(status='NOT_TESTED_CONSTANT_TARGET',edges_file=None)
        else:
            logger.addHandler(handler)
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');edges,meta=w['core'].infer_partial_network(regressor_type='GBM',regressor_kwargs=dict(PARAMS),tf_matrix=w['tf'],tf_matrix_gene_names=axes['predictor_stable_ids'],target_gene_name=target,target_gene_expression=y,include_meta=True,early_stop_window_length=25,seed=0)
            require(list(meta.columns)==['target','n_estimators'] and len(meta)==1 and meta.iloc[0]['target']==target and float(meta.iloc[0]['n_estimators']).is_integer() and 1<=int(meta.iloc[0]['n_estimators'])<=5000,'FAILED_LIBRARY_RETRY_FALLBACK_OR_UNPROVEN_FIT')
            require(list(edges.columns)==['TF','target','importance'] and not edges['TF'].duplicated().any() and set(edges['target'])<=set([target]),'Malformed official links')
            clean=[t for t in axes['predictor_stable_ids'] if t!=target];values={str(row.TF):float(row.importance) for row in edges.itertuples(index=False)}
            require(set(values)<=set(clean) and all(math.isfinite(x) and x>0 for x in values.values()),'Invalid/self/unknown importance')
            tsv(folder/'all_predictors.tsv',[{'TF_stable_id':tf,'target_stable_id':target,'importance':values.get(tf,0.0)} for tf in clean])
            result.update(status='SUCCESS_GRNBOOST2_TARGET_AFTER_LIBRARY_RETRY' if messages else 'SUCCESS_GRNBOOST2_TARGET',n_estimators=int(meta.iloc[0]['n_estimators']),predictor_count_after_official_self_exclusion=len(clean),positive_edge_count=len(values),zero_importance_count=len(clean)-len(values),zeros_materialized_only_after_verified_success=True,edges_file=rec(folder/'all_predictors.tsv'),python_warnings=[str(x.message) for x in caught])
    except BaseException as e:result.update(status='HOLD_TARGET_INFERENCE_FAILED',error_type=type(e).__name__,error=str(e),edges_file=None)
    finally:
        logger.removeHandler(handler);result.update(library_retry_messages=messages,elapsed_seconds=time.monotonic()-started);new(folder/'receipt.json',result)
    return {'target_index':index,'receipt':rec(folder/'receipt.json'),'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}

def pilot_child(a):
    v=approved(a);mp=Path(a.input_manifest);require(mp.is_absolute() and mp.resolve().is_relative_to(R.resolve()) and sha(mp)==a.input_manifest_sha,'Exact approved-input manifest path/SHA required');m=read(mp)
    require(m['status']=='EXACT_FROZEN_RNA_X_EXPORTED' and m['source']==rec(__file__) and m['contract']==rec(Path(a.contract)) and m['axes']==v['axes'] and m['rna']==v['rna'] and m['logical_sha256']==v['logical_RNA_X_sha256'] and m['normalization_performed'] is False,'Frozen exported matrix provenance differs');checked(m['matrix'])
    out=R/v['pilot_output_relative'];require(not out.exists(),'Single-use pilot output required');out.mkdir(parents=True);new(out/'STARTED.json',{'source':rec(__file__),'contract':rec(Path(a.contract)),'input_manifest':rec(mp),'targets':v['targets'],'seed':0,'workers':2,'scientific_pass':False})
    started=time.monotonic();rows=[]
    with multiprocessing.get_context('spawn').Pool(processes=2,initializer=worker_init,initargs=(str(a.contract),str(mp),str(out))) as pool:
        for row in pool.imap_unordered(target_job,range(8),chunksize=1):rows.append(row);new(out/('completion_%02d.json'%row['target_index']),row)
    require(sorted(r['target_index'] for r in rows)==list(range(8)),'Every fixed first8 target requires a receipt');checked(m['matrix'])
    value={'status':'COMPLETE_FIXED_FIRST8_RESOURCE_PILOT_REVIEW_PENDING','targets':sorted(rows,key=lambda x:x['target_index']),'contract':rec(Path(a.contract)),'input_manifest':rec(mp),'workers':2,'seeds':[0],'elapsed_seconds':time.monotonic()-started,'full3006x3_completed':False,'method_scope':'same-cell algorithm corroboration only','clinical_P':None,'rescue_or_causal_direction_inferred':False,'scientific_pass':False};new(out/'COMPLETE.json',value);return value

def supervise(a):
    v=approved(a)
    import ctypes,psutil
    require(psutil.virtual_memory().available>=v['resource_limits']['startup_available_bytes'],'Not enough available RAM for reviewed6GiB pilot envelope')
    require(a.input_manifest and a.input_manifest_sha,'Exact materialized input manifest required')
    sys.path.insert(0,str(R/'repo/pipeline'));import supervise_local_resume as owner
    original_entry,original_job=owner.ENTRY,owner.OwnedJob
    class BoundedJob(original_job):
        def __init__(self):
            super().__init__();limits=owner.ExtendedLimits();limits.BasicLimitInformation.LimitFlags=0x2000|0x100|0x200;limits.ProcessMemoryLimit=limits.JobMemoryLimit=v['resource_limits']['max_job_commit_bytes']
            if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):self.close();raise ctypes.WinError(ctypes.get_last_error())
            bound=owner.ExtendedLimits()
            if not self.k.QueryInformationJobObject(self.handle,9,ctypes.byref(bound),ctypes.sizeof(bound),None) or bound.JobMemoryLimit!=limits.JobMemoryLimit or bound.ProcessMemoryLimit!=limits.ProcessMemoryLimit or bound.BasicLimitInformation.LimitFlags&0x2300!=0x2300:self.close();raise RuntimeError('Kernel job memory limits not bound')
    command=[str(checked(v['interpreter'])),'-B',str(Path(__file__)),'pilot-child','--contract',a.contract,'--contract-sha',a.contract_sha,'--review',a.review,'--review-sha',a.review_sha,'--input-manifest',a.input_manifest,'--input-manifest-sha',a.input_manifest_sha]
    job=R/'tmp_cache/local_owned_jobs'/v['job_id'];require(not job.exists(),'Never silently replay a used pilot job ID')
    spec={'job_id':v['job_id'],'job_dir':str(job),'command':command,'python_sha256':v['interpreter']['sha256'],'entry_sha256':v['source']['sha256'],'manifest_sha256':a.contract_sha,'supervisor_sha256':v['source']['sha256'],'wall_seconds':600,'max_rss_bytes':v['resource_limits']['max_RSS_bytes'],'min_free_bytes':v['resource_limits']['min_system_available_bytes'],'stop_file':str(job/'STOP')}
    try:owner.ENTRY=Path(__file__);owner.OwnedJob=BoundedJob;result=owner.supervise(spec)
    finally:owner.ENTRY=original_entry;owner.OwnedJob=original_job
    receipt={'status':'FINISHED_RESOURCE_PILOT_ROOT_REVIEW_PENDING' if result['exit_code']==0 and result['owned_job_empty'] and not result['errors'] and not result['stop_reason'] else 'HOLD_RESOURCE_PILOT_INCOMPLETE','owned_job_result':result,'contract':rec(Path(a.contract)),'source':rec(__file__),'other_user_processes_touched':False,'scientific_pass':False};new(job/'pilot_supervision_receipt.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','metadata','export','supervise','pilot-child'])
    for k in ('contract','contract-sha','review','review-sha','input-manifest','input-manifest-sha'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.mode=='prepare':v=prepare()
    else:
        require(a.contract and a.contract_sha,'Exact contract arguments required')
        if a.mode=='metadata':c=load_contract(Path(a.contract),a.contract_sha);v={'status':'METADATA_ONLY_NO_EXPRESSION_OR_ALGORITHM','contract':rec(Path(a.contract)),'targets':c['targets'],'workers':2,'wall_seconds':600,'source_review_required':True}
        else:
            require(a.review and a.review_sha,'Exact source-bound root review required')
            v=export_input(a) if a.mode=='export' else supervise(a) if a.mode=='supervise' else pilot_child(a)
    print(json.dumps(v,allow_nan=False),flush=True)
if __name__=='__main__':multiprocessing.freeze_support();main()
