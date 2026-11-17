"""Offline all1191 PRIMARY192 shared-program diagnostics; no screen, P, GRN or model."""
import argparse, csv, datetime, hashlib, json, math, os
from pathlib import Path
import sys, threading, time
SCHEMA='dynavelo_primary_global_diagnostic_input_v1'
LIMIT_BYTES=8*1024**3
WALL_SECONDS=600
BLAS_THREADS=4

def require(ok,message):
    if not ok: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def record(path,root):
    p=Path(path)
    return {'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def bound_path(root,relative):
    require(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'Relative payload path required')
    p=root/relative
    require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink(),'Payload escapes bundle')
    return p
def checked(root,rec):
    p=bound_path(root,rec['path'])
    require(p.is_file() and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],'Payload hash/size mismatch: '+rec['path'])
    return p
def write_json(path,value):
    p=Path(path); temp=p.with_suffix(p.suffix+'.partial')
    with temp.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    os.replace(temp,p)
def write_csv(path,rows):
    require(bool(rows),'Nonempty explicit table required')
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(values):
    import numpy as np
    a=np.asarray(values,dtype=np.float64);require(a.ndim==1 and np.isfinite(a).all(),'Summary values must be finite')
    q=[0.,.05,.25,.5,.75,.95,1.]
    return {'n':int(a.size),'mean':float(a.mean()) if a.size else None,
            'population_std_ddof0':float(a.std(ddof=0)) if a.size else None,
            'quantiles_linear':{str(x):float(y) for x,y in zip(q,np.quantile(a,q,method='linear'))} if a.size else {str(x):None for x in q}}

def top_membership(matrix):
    """Finite nonzero rows only; positive abs-score top50 with exact boundary ties, no fill."""
    import numpy as np
    a=np.asarray(matrix);require(a.ndim==2 and a.dtype==np.float64,'float64 matrix required')
    finite=np.isfinite(a).all(axis=1);nonzero=np.zeros(a.shape[0],dtype=bool)
    nonzero[finite]=np.any(a[finite]!=0,axis=1);usable=finite&nonzero
    membership=np.zeros(a.shape,dtype=bool)
    for i in np.flatnonzero(usable):
        score=np.abs(a[i]);positive=score[score>0];k=min(50,len(positive))
        threshold=np.partition(positive,len(positive)-k)[len(positive)-k]
        membership[i]=(score>0)&(score>=threshold)
    return membership,finite,usable

def spectrum(matrix,normalized):
    """Full Gram eigenspectrum, machine-roundoff-only negative clipping."""
    import numpy as np
    a=np.asarray(matrix,dtype=np.float64);require(a.ndim==2 and np.isfinite(a).all(),'Only explicit finite nonzero rows enter spectrum')
    if not len(a):
        return {'status':'NOT_COMPUTABLE_NO_FINITE_NONZERO_ROWS','rows':0,'top10_singular_values':[],'top10_energy_fractions':[],'top10_cumulative_energy_fractions':[]}
    scales=np.max(np.abs(a),axis=1);require((scales>0).all(),'Zero rows must be reported and excluded explicitly')
    if normalized:
        scaled=a/scales[:,None];a=scaled/np.sqrt(np.sum(scaled*scaled,axis=1))[:,None]
    # A single global scale preserves energy fractions and avoids Gram overflow/underflow.
    global_scale=float(np.max(np.abs(a)));scaled=a/global_scale;gram=scaled@scaled.T
    require(np.isfinite(gram).all(),'Gram nonfinite');eigenvalues=np.linalg.eigvalsh(gram)
    tolerance=64*np.finfo(np.float64).eps*max(1,len(a))*max(1.,float(np.max(np.abs(eigenvalues))))
    require(float(eigenvalues.min())>=-tolerance,'Gram negative eigenvalues beyond machine roundoff')
    negative_n=int((eigenvalues<0).sum());eigenvalues=np.maximum(eigenvalues,0.)[::-1];total=float(eigenvalues.sum())
    require(math.isfinite(total) and total>0,'No usable spectral energy')
    fraction=eigenvalues[:10]/total;singular=np.sqrt(eigenvalues[:10])*global_scale;finite=bool(np.isfinite(singular).all())
    return {'status':'COMPUTABLE','rows':int(len(a)),'columns':int(a.shape[1]),
            'normalization':'row_L2_unit' if normalized else 'original_signed_velocity_matrix',
            'top10_singular_values':singular.tolist() if finite else None,'singular_value_overflow':not finite,
            'top10_energy_fractions':fraction.tolist(),'top10_cumulative_energy_fractions':np.cumsum(fraction).tolist(),
            'all_spectrum_n':int(len(eigenvalues)),'negative_roundoff_eigenvalues_clipped_n':negative_n,
            'negative_roundoff_tolerance_scaled_Gram':tolerance,'global_numeric_scaling':global_scale,'statistical_P_computed':False}

def diagnose(matrix):
    import numpy as np
    membership,finite,usable=top_membership(matrix);indices=np.flatnonzero(usable)
    m=membership[indices].astype(np.float64);sizes=m.sum(axis=1);intersection=m@m.T
    require(np.array_equal(intersection,np.rint(intersection)),'Set intersections must be exact integers')
    union=sizes[:,None]+sizes[None,:]-intersection;require((union>0).all() if union.size else True,'Usable top sets cannot be empty')
    jaccard=intersection/union if len(indices) else np.empty((0,0));pair_values=jaccard[np.triu_indices(len(indices),1)]
    per_row=[];local={int(i):j for j,i in enumerate(indices)}
    for i in range(matrix.shape[0]):
        status='COMPUTABLE' if usable[i] else 'NOT_COMPUTABLE_NONFINITE_ROW' if not finite[i] else 'NOT_COMPUTABLE_ZERO_NORM_ROW'
        other=np.delete(jaccard[local[i]],local[i]) if usable[i] else np.array([])
        per_row.append({'input_index':i,'status':status,'positive_top50_boundary_ties_size':int(membership[i].sum()) if finite[i] else None,
                        'positive_top50_boundary_ties_target_indices':';'.join(map(str,np.flatnonzero(membership[i]).tolist())) if finite[i] else None,
                        'comparable_other_inputs_n':int(len(other)),'mean_Jaccard_to_other_usable_inputs':float(other.mean()) if len(other) else None,
                        'median_Jaccard_to_other_usable_inputs':float(np.median(other)) if len(other) else None})
    total_pairs=matrix.shape[0]*(matrix.shape[0]-1)//2
    result={'all_inputs_retained_n':matrix.shape[0],'finite_nonzero_computable_inputs_n':int(usable.sum()),
            'nonfinite_inputs_n':int((~finite).sum()),'zero_norm_inputs_n':int((finite&~usable).sum()),
            'all_input_pairs_n':total_pairs,'comparable_pairs_n':int(len(pair_values)),'not_computable_pairs_n':total_pairs-len(pair_values),
            'pairwise_Jaccard_distribution':summarize(pair_values),'original_matrix_spectrum':spectrum(matrix[usable],False),
            'row_unit_matrix_spectrum':spectrum(matrix[usable],True),'P_computed':False,'new_candidate_screen':False,'scientific_pass':False}
    return result,per_row,membership.sum(axis=0).astype(int)

def validate_contract(root,contract_path,contract_sha):
    require(sha(contract_path)==contract_sha,'Contract SHA mismatch');c=json.loads(contract_path.read_text(encoding='utf-8'))
    require(c.get('schema')==SCHEMA,'Unsupported input contract schema')
    require(c.get('scope')=='A100_PRIMARY192_ALL1191_RNA_INPUTS_EFFECT_POPULATION_ALL_VX','Fixed diagnostic scope required')
    require(c.get('scientific_pass') is False and c.get('new_candidate_screen') is False,'No scientific promotion or new screen')
    require(len(c['matrices'])==3 and {x['seed'] for x in c['matrices']}=={0,1,2},'All three seeds required')
    records=[c['axes'],c['provenance']]+c['matrices'];require(len({x['path'] for x in records})==len(records),'Unique payload paths required')
    for item in records: checked(root,item)
    axes=json.loads(checked(root,c['axes']).read_text(encoding='utf-8'))
    require(axes.get('schema')=='dynavelo_primary_global_diagnostic_axes_v1','Axes schema differs')
    inputs,targets=axes['inputs'],axes['targets'];require(len(inputs)==1191 and len(targets)==3006,'All1191x3006 required')
    for items in (inputs,targets):
        require([x['index'] for x in items]==list(range(len(items))),'Axis order/indices differ')
        require(len({x['stable_id'] for x in items})==len(items) and all(x['stable_id'] and isinstance(x['gene'],str) for x in items),'Unique stable IDs/labels required')
    require(all(type(x['locked20']) is bool for x in inputs) and sum(x['locked20'] for x in inputs)==20,'Locked20 annotation required without changing scope')
    provenance=json.loads(checked(root,c['provenance']).read_text(encoding='utf-8'))
    require(provenance.get('schema')=='dynavelo_primary_global_diagnostic_provenance_v1' and provenance.get('scientific_pass') is False,'Source provenance schema required')
    for item in c['matrices']: require(item['shape']==[1191,3006] and item['dtype']=='float64','Fixed shape/dtype required')
    return c,axes,provenance

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',required=True);parser.add_argument('--contract',default='contract.json')
    parser.add_argument('--contract-sha',required=True);parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    root,out=Path(args.input_dir).resolve(),Path(args.output_dir).resolve()
    require(root.is_dir() and not out.exists(),'Existing input/fresh output required');require(not out.is_relative_to(root),'Separate output from frozen input bundle');out.mkdir(parents=True)
    started=time.monotonic();done=threading.Event()
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]=str(BLAS_THREADS)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(LIMIT_BYTES,LIMIT_BYTES))
        require(resource.getrlimit(resource.RLIMIT_AS)==(LIMIT_BYTES,LIMIT_BYTES),'Address-space limit not bound')
    except (ImportError,ValueError,OSError) as e:
        write_json(out/'run_receipt.json',{'status':'REFUSED_MEMORY_LIMIT_NOT_BOUND','error':str(e),'scientific_pass':False});return 2
    def watch():
        if not done.wait(WALL_SECONDS):
            try:write_json(out/'run_timeout_receipt.json',{'status':'HARD_TIMEOUT','elapsed_seconds':time.monotonic()-started,'scientific_pass':False})
            finally:os._exit(124)
    threading.Thread(target=watch,daemon=True).start()
    receipt={'schema':'dynavelo_primary_global_diagnostic_run_receipt_v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_sha256':sha(Path(__file__)),'contract_sha256':args.contract_sha,'memory_virtual_address_space_limit_bytes':LIMIT_BYTES,
             'hard_wall_seconds':WALL_SECONDS,'blas_threads_maximum':BLAS_THREADS,'single_process':True,'scientific_pass':False,
             'scope_limit':'A100 PRIMARY192 descriptive shared program only; no new screen; no full1884 endpoint replacement; ALL_NOT_TESTED/HOLD and CPU-A100 bridge unchanged'}
    try:
        contract_path=bound_path(root,args.contract);c,axes,provenance=validate_contract(root,contract_path,args.contract_sha)
        import numpy as np
        from threadpoolctl import threadpool_limits,threadpool_info
        summaries,row_results,frequencies=[],[],[]
        with threadpool_limits(limits=BLAS_THREADS):
            receipt['threadpools']=threadpool_info();require(all(x.get('num_threads',0)<=BLAS_THREADS for x in receipt['threadpools']),'BLAS cap not active')
            for item in sorted(c['matrices'],key=lambda x:x['seed']):
                matrix=np.load(checked(root,item),mmap_mode='r',allow_pickle=False);require(matrix.shape==(1191,3006) and matrix.dtype==np.dtype('float64'),'Actual matrix shape/dtype differs')
                summary,rows,freq=diagnose(matrix);summaries.append({'seed':item['seed'],**summary})
                for row,axis in zip(rows,axes['inputs']):row_results.append({'seed':item['seed'],'input_stable_id':axis['stable_id'],'input_gene':axis['gene'],'locked20_annotation_only':axis['locked20'],**row})
                for j,axis in enumerate(axes['targets']):frequencies.append({'seed':item['seed'],'target_index':j,'target_stable_id':axis['stable_id'],'target_gene':axis['gene'],'occurrence_in_usable_input_top_sets_n':int(freq[j]),'all_input_denominator':1191,'computable_input_denominator':summary['finite_nonzero_computable_inputs_n']})
                del matrix
        for item in [c['axes'],c['provenance']]+c['matrices']:checked(root,item)
        require(sha(contract_path)==args.contract_sha,'Contract changed during analysis')
        write_json(out/'all3seed_shared_program_summary.json',{'schema':'dynavelo_primary_global_diagnostic_result_v1','seeds':summaries,'scientific_pass':False,'scope_limit':receipt['scope_limit']})
        write_csv(out/'all1191_input_diagnostics.csv',row_results);write_csv(out/'all3006_target_occurrence_frequencies.csv',frequencies)
        receipt.update(status='COMPLETE_DESCRIPTIVE_ONLY',seeds=3,input_rows=3573,target_frequency_rows=9018,inputs_pre_and_post_hash_verified=True,numpy_version=np.__version__,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        require(receipt['peak_rss_bytes']<=LIMIT_BYTES,'PeakRSS exceeded limit');write_json(out/'run_receipt.json',receipt)
        write_json(out/'manifest.json',{'schema':'dynavelo_primary_global_diagnostic_output_manifest_v1','files':[record(p,out) for p in sorted(out.iterdir()) if p.is_file()],'scientific_pass':False,'source_sha256':receipt['source_sha256'],'contract_sha256':args.contract_sha})
        print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds'],'output_bytes':sum(p.stat().st_size for p in out.iterdir()),'scientific_pass':False}),flush=True);return 0
    except BaseException as e:
        receipt.update(status='FAILED_NO_SCIENTIFIC_RESULT',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.monotonic()-started);write_json(out/'run_receipt.json',receipt)
        print(json.dumps({'status':receipt['status'],'error':str(e)}),file=sys.stderr,flush=True);return 1
    finally:done.set()

if __name__=='__main__':sys.exit(main())
