"""CPU per-cell finite-difference readout. Inspect/prepare by default, no model.

Execution requires a root review and the complete 696-COMMIT CPU family before
any numerical mmap access. Original16 HOLD and unavailable A100 bridge persist.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,gzip,hashlib,io,json,math,shutil,time
ROOT=Path(r'D:\dyc_7\DynaVelo_Research')
PLAN_SHA='2d4db478e640b2e54be14d1386639a728ff7f904f728eafec8116d6f92f5c581'
COMPARISONS=('small_step','large_step','tolerance')
LIMITS={'small_L2_max':.01,'large_L2_max':.1,'small_strong_sign_min':.99,'large_strong_sign_min':.95,'strong_union_absolute_threshold':1e-7,'low_reference_RMS_threshold':1e-8}
METRICS=('n_entries','absolute_L2','absolute_RMS','maximum_absolute_error','reference_L2','reference_RMS','derivative_L2','derivative_RMS','epsilon_times_derivative_L2','reference_epsilon_times_derivative_L2','relative_L2','sign_agreement','zero_zero','current_zero_reference_nonzero','current_nonzero_reference_zero','opposite_nonzero_sign','same_nonzero_sign','strong_entry_count','strong_entry_fraction','strong_entry_same_sign_count','strong_entry_same_sign_fraction','zero_reference_L2','no_strong_entries')
def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def record(root,p):
    p=Path(p).resolve();return {'relative_path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(root,rec):
    p=root/rec['relative_path'];require(not p.is_symlink() and p.resolve().is_relative_to(root) and p.is_file(),'Missing/redirected artifact: '+rec['relative_path'])
    require(record(root,p)==rec,'Artifact differs: '+rec['relative_path']);return p
def new(p,v):
    with Path(p).open('x',encoding='utf-8') as h:json.dump(v,h,indent=2,allow_nan=False);h.write('\n')
def close(a):
    if getattr(a,'_mmap',None) is not None:a._mmap.close()
def _np():
    import numpy as np
    return np
def window_axis(plan,frame,original_spec,session):
    ids=[c['cell_id'] for c in original_spec['cells']];w=frame['full_context_frame']['global_time_windows'][str(session['seed'])]['window_index']
    require(len(ids)==len(set(ids))==len(w) and all(type(v) is int and 0<=v<5 for v in w),'Frozen original cell/window axis differs')
    mapping=dict(zip(ids,w));selected=session['baseline_geometry']['sample_cell_ids']
    require(len(selected)==len(set(selected)) and all(c in mapping for c in selected),'Unknown/duplicate selected ID')
    return [mapping[c] for c in selected]
def classify_metrics(m,comparison):
    require(comparison in COMPARISONS,'Unknown comparison')
    if m['n_entries']==0:return 'NOT_TESTED_EMPTY_WINDOW',['empty_original_window']
    if m['reference_RMS']<=1e-8:return 'LOW_NUMERICAL_SIGNAL',['reference_RMS_at_or_below_1e-8']
    reasons=[];large=comparison=='large_step'
    if m['relative_L2'] is None or m['relative_L2']>(.1 if large else .01):reasons.append('relative_L2_missing_or_exceeds_limit')
    if m['strong_entry_same_sign_fraction'] is None:reasons.append('no_entries_meet_union_strong_mask')
    elif m['strong_entry_same_sign_fraction']<(.95 if large else .99):reasons.append('strong_union_sign_below_limit')
    return ('HOLD' if reasons else 'STABLE_UNDER_NEW_ENGINEERING_CRITERIA'),reasons
def metrics_columns(current,reference,epsilon,reference_epsilon,comparison):
    np=_np();x=np.asarray(current);y=np.asarray(reference)
    require(x.ndim==2 and x.shape==y.shape and x.dtype==y.dtype==np.dtype('float64'),'Exact aligned FP64 matrices required')
    require(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite derivative entries')
    n,t=x.shape
    if n==0:return [dict.fromkeys(METRICS,None)|{'n_entries':0} for _ in range(t)]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        d=x-y;e=np.sqrt(np.sum(d*d,axis=0,dtype=np.float64));r=np.sqrt(np.sum(y*y,axis=0,dtype=np.float64));c=np.sqrt(np.sum(x*x,axis=0,dtype=np.float64));maximum=np.max(np.abs(d),axis=0)
        xz=x==0;yz=y==0;both=~xz&~yz;same=both&(np.signbit(x)==np.signbit(y));strong=np.maximum(np.abs(x),np.abs(y))>=1e-7
        categories=[np.sum(v,axis=0,dtype=np.int64) for v in (xz&yz,xz&~yz,~xz&yz,both&~same,same)]
        ns=np.sum(strong,axis=0,dtype=np.int64);ss=np.sum(strong&((xz&yz)|same),axis=0,dtype=np.int64)
    require(all(np.isfinite(a).all() for a in (e,r,c,maximum)),'Nonfinite metric accumulation')
    rows=[]
    for j in range(t):
        row=dict(n_entries=n,absolute_L2=float(e[j]),absolute_RMS=float(e[j]/math.sqrt(n)),maximum_absolute_error=float(maximum[j]),reference_L2=float(r[j]),reference_RMS=float(r[j]/math.sqrt(n)),derivative_L2=float(c[j]),derivative_RMS=float(c[j]/math.sqrt(n)),epsilon_times_derivative_L2=float(abs(epsilon)*c[j]),reference_epsilon_times_derivative_L2=float(abs(reference_epsilon)*r[j]),relative_L2=float(e[j]/r[j]) if r[j]>0 else None,sign_agreement=float((categories[0][j]+categories[4][j])/n),**{k:int(a[j]) for k,a in zip(('zero_zero','current_zero_reference_nonzero','current_nonzero_reference_zero','opposite_nonzero_sign','same_nonzero_sign'),categories)},strong_entry_count=int(ns[j]),strong_entry_fraction=float(ns[j]/n),strong_entry_same_sign_count=int(ss[j]),strong_entry_same_sign_fraction=float(ss[j]/ns[j]) if ns[j] else None,zero_reference_L2=bool(r[j]==0),no_strong_entries=bool(ns[j]==0))
        require(all(math.isfinite(v) for v in row.values() if type(v) is float),'Nonfinite derived metric')
        require(sum(row[k] for k in ('zero_zero','current_zero_reference_nonzero','current_nonzero_reference_zero','opposite_nonzero_sign','same_nonzero_sign'))==n,'Incomplete sign decomposition');rows.append(row)
    return rows
def gate_columns(masks):
    np=_np();require(len(masks)==6,'All six RNA masks required');arrays=[np.asarray(v) for v in masks];shape=arrays[0].shape
    require(len(shape)==2 and all(a.shape==shape and a.dtype==np.dtype('bool') for a in arrays),'Aligned bool masks required')
    changed=np.zeros(shape,dtype=bool)
    for a in arrays[1:]:changed|=a!=arrays[0]
    return [{'gate_changed_cells':int(v),'gate_review_status':'NOT_TESTED_EMPTY_WINDOW' if not shape[0] else 'HOLD_FOR_GATE_REVIEW' if v else 'NO_CAPTURED_DECODER_GATE_CHANGE'} for v in np.sum(changed,axis=0,dtype=np.int64)]
def unpack_gate_slice(a,start,stop):
    np=_np();require(a.ndim==2 and a.dtype==np.dtype('uint8') and a.shape[1]==376 and 0<=start<stop<=3006,'Packed mask header/slice differs')
    require(not np.any(a[:,-1]&192),'Nonzero unused packed RNA padding bits')
    return np.unpackbits(a[:,start//8:(stop+7)//8],axis=1,bitorder='little')[:,start%8:start%8+stop-start].astype(bool)
def inspect_plan(plan_path):
    root=ROOT.resolve();p=Path(plan_path).resolve();require(p.is_relative_to(root) and sha(p)==PLAN_SHA,'Exact D CPU plan required');plan=read(p)
    require(plan['schema']=='dynavelo_cpu_fixed28_candidate_proposal_v1' and hashlib.sha256(canonical({k:v for k,v in plan.items() if k!='plan_sha256'})).hexdigest()==plan['plan_sha256'],'CPU plan seal/schema differs')
    for k,v in LIMITS.items():require(plan['qualification'][k]==v,'Frozen numerical limit changed: '+k)
    require(plan['qualification']['original_16_HOLD_unchanged'] and len(plan['fixed_inputs'])==28 and Counter(x['input_space'] for x in plan['fixed_inputs'])=={'rna':20,'motif':8},'Fixed28 inventory differs')
    require(len({(x['input_space'],x['feature_id']) for x in plan['fixed_inputs']})==28,'Duplicate fixed input')
    require(len(plan['sessions'])==12 and plan['counts']['total_tasks']==684 and plan['counts']['derivative_tasks']==672 and sum(len(s['operations']) for s in plan['sessions'])==696,'Complete family denominator differs')
    for rec in plan['source_records']:checked(root,rec)
    original=read(checked(root,plan['original_scope_plan']))['context_plan']['spec'];frame=read(checked(root,plan['scope_frame']))
    require((len(original['cells']),len(original['genes']),len(original['motifs']))==(6035,3006,365),'Frozen model axes differ')
    for s in plan['sessions']:
        g=s['baseline_geometry'];n=192 if s['sample_id']=='primary' else 1884
        require(s['seed'] in (0,1,2) and s['tolerance'] in (1e-7,1e-9) and g['n_cells']==len(g['sample_cell_ids'])==n and g['batches']==[[i,min(i+64,n)] for i in range(0,n,64)],'Frozen geometry differs');window_axis(plan,frame,original,s)
        require([o['kind'] for o in s['operations'][:2]]==['baseline','no-op'],'Baseline/no-op order differs')
        hs=(1e-5,1e-4,1e-3) if s['tolerance']==1e-7 else (1e-4,)
        require({(o['input_space'],o['feature_id'],o['epsilon']) for o in s['operations'][2:]}=={(x['input_space'],x['feature_id'],h) for x in plan['fixed_inputs'] for h in hs},'Missing fixed derivative operation')
    return plan,original,frame
def build_contract(plan_path):
    inspect_plan(plan_path)
    return {'schema':'dynavelo_cpu_fixed28_per_cell_readout_contract_v1','status':'PREPARED_UNEXECUTED_REQUIRES_ROOT_REVIEW','cpu_plan':record(ROOT,plan_path),'source':record(ROOT,Path(__file__)),'thresholds':LIMITS,'comparisons':{'small_step':{'current_tolerance':1e-7,'current_h':1e-5,'reference_tolerance':1e-7,'reference_h':1e-4},'large_step':{'current_tolerance':1e-7,'current_h':1e-3,'reference_tolerance':1e-7,'reference_h':1e-4},'tolerance':{'current_tolerance':1e-7,'current_h':1e-4,'reference_tolerance':1e-9,'reference_h':1e-4}},'expected_COMMITs':696,'raw_tasks':684,'records_per_comparison':2831640,'total_records':8494920,'windows':'Original 6035 cell-ID axis -> frozen seed window_index -> exact session cell IDs; all five including empty','gate_rule':'Per vx target/window union of six B7,D7(1e-5,1e-4,1e-3),B9,D9(1e-4) masks; separate HOLD. vy linear decoder not applicable.','validation_before_values':'All696 COMMITs/exact payload inventories/every SHA, matching CPU baseline logical hashes, cohort and all210 original audits.','actual_derivative_values_read_during_preparation':False,'original_16_HOLD_unchanged':True,'A100_per_cell_bridge':'NOT_TESTED_UNAVAILABLE','scientific_pass':False,'no_model_calls':True,'A100_PASSmap_authorized':False,'target_chunk':128,'output':'1008 gzip TSV shards; full row values and all statuses retained, not means-based qualification'}

def execution_integrity(plan,root,plan_record):
    namespace=root/plan['namespace_relative'];tp=root/'tmp_cache/local_owned_jobs'/plan['job_id']/'terminal.json';terminal=read(tp)
    require(terminal['plan']==plan_record and terminal['status']=='COMPLETE_RAW_CPU_FAMILY_READOUT_PENDING' and not terminal['failures'],'Complete exact CPU cohort required')
    require({p.name for p in namespace.iterdir() if p.is_dir() and p.name.startswith('worker_')}=={'worker_'+str(i) for i in range(4)},'Unexpected worker namespace')
    require(len(terminal['owned_job_cleanup'])==4 and all(v['exit_code']==0 and v['owned_job_empty'] for v in terminal['owned_job_cleanup']),'Successful owned cleanup required')
    review=read(checked(root,terminal['review']));require(review['decision']=='APPROVE_FIXED_CPU_CANDIDATE_EXECUTION' and review['plan']==plan_record,'CPU execution approval differs')
    rows=[read(checked(root,r)) for r in terminal['worker_results']]
    require(len(rows)==4 and {v['worker'] for v in rows}==set(range(4)) and all(v['plan']==plan_record and v['status']=='COMPLETE_RAW_CPU_CANDIDATE_NOT_QUALIFIED' for v in rows),'All four exact workers required')
    for key,n in (('structured_forwards',11484),('original_reference_forwards',210),('completed_tasks',684)):require(sum(v[key] for v in rows)==n,'Raw call denominator differs: '+key)
    observed=[];metadata=[record(root,tp),terminal['review'],*terminal['worker_results']]
    for worker in range(4):
        for p in (namespace/('worker_'+str(worker))).glob('audit_*.json'):
            v=read(p);ordinal=v['ordinal'];matches=[s for s in plan['sessions'] if s['worker']==worker and s['first_ordinal']<=ordinal<s['first_ordinal']+s['structured_forward_count']]
            require(len(matches)==1 and v['all_nine_bitwise_equal'] and v['RNG_unchanged_and_pair_equal'],'Reference audit failed/context differs')
            s=matches[0];op=next(o for o in s['operations'] if o['first_ordinal']<=ordinal<o['first_ordinal']+o['forward_count']);a,b=s['baseline_geometry']['batches'][ordinal-op['first_ordinal']]
            require((v['seed'],v['tolerance'],v['batch_cells'])==(s['seed'],s['tolerance'],b-a) and len(v['array_sha256'])==9,'Reference audit geometry differs')
            observed.append(ordinal);metadata.append(record(root,p))
    require(sorted(observed)==plan['audit_ordinals'] and len(observed)==210,'All210 original audits required');return metadata

def mmap_checked(path,shape,dtype):
    np=_np();a=np.load(path,mmap_mode='r',allow_pickle=False)
    if a.shape!=tuple(shape) or a.dtype!=np.dtype(dtype):close(a);raise ValueError('Array shape/dtype differs: '+str(path))
    return a
def logical_digest(a):
    np=_np();h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
    for start in range(0,len(a),256):
        tile=np.ascontiguousarray(a[start:start+256]);require(np.isfinite(tile).all(),'Nonfinite baseline');h.update(tile.tobytes())
    return h.hexdigest()

def file_payloads(plan,root,plan_record,original):
    """No np.load: all COMMITs, independent axes and payload hashes first."""
    namespace=root/plan['namespace_relative'];census=[];payloads={};axes_records=[];errors=[]
    expected={s['session_id'] for s in plan['sessions']};actual={p.name for p in namespace.iterdir() if p.is_dir() and not p.name.startswith('worker_')}
    require(actual==expected,'Missing/extra CPU session directories')
    for s in plan['sessions']:
        directory=namespace/s['session_id'];bp=directory/s['operations'][0]['operation_id']/'COMMIT.json';parent=record(root,bp) if bp.is_file() else None
        require({p.name for p in directory.iterdir()}=={'baseline','axes.json'}|{o['operation_id'] for o in s['operations']},'Missing/extra session operation directory')
        ap=directory/'axes.json';axes=read(ap);axes_records.append(record(root,ap))
        require(axes=={'cell_ids':s['baseline_geometry']['sample_cell_ids'],'rna_targets':[v['feature_id'] for v in original['genes']],'motif_targets':[v['tf_rna_feature_id'] for v in original['motifs']],'packed_RNA_active_mask':{'dtype':'uint8','bitorder':'little','true_features':3006,'packing_axis':1},'original_windows':plan['scope_frame']},'Exact independent axes.json differs')
        for op in s['operations']:
            f=directory/op['operation_id']/'COMMIT.json'
            try:
                v=read(f)
                require(v['status']=='COMPLETE_RAW_ENGINEERING_ONLY' and v['operation']==op and v['session_id']==s['session_id'] and v['plan']==plan_record and v['backend']=='cpu' and v['tolerance']==s['tolerance'],'Raw COMMIT operation/context differs')
                require(v['scientific_pass'] is False and v['candidate_qualification']=='PENDING_COMPLETE_FAMILY_AND_READOUT','Raw completion cannot claim qualification')
                require(v['matched_baseline_commit']==(None if op['kind']=='baseline' else parent),'Matching CPU baseline COMMIT differs')
                require(v['baseline_manifest']==record(root,directory/'baseline/baseline_manifest.json'),'Matching baseline manifest record differs')
                names={'RNA_active_mask.npy'} if op['kind']=='baseline' else set() if op['kind']=='no-op' else {'RNA_active_mask.npy','vx_derivative.npy','vy_derivative.npy'}
                paths={(f.parent/name).relative_to(root).as_posix() for name in names}
                if op['kind']=='baseline':paths|={(directory/'baseline'/name).relative_to(root).as_posix() for name in ('vx.npy','vy.npy','vz.npy','time.npy','baseline_manifest.json')}
                recs=v['payload'];require(len(recs)==len(paths) and {r['relative_path'] for r in recs}==paths,'Payload inventory missing/duplicate/extra')
                require({p.name for p in f.parent.iterdir()}==names|{'COMMIT.json'},'Extra/partial raw operation file')
                for rec in recs:
                    checked(root,rec);require(rec['relative_path'] not in payloads or payloads[rec['relative_path']]==rec,'Conflicting payload record');payloads[rec['relative_path']]=rec
                census.append(record(root,f))
            except Exception as e:errors.append({'session':s['session_id'],'operation':op['operation_id'],'error':type(e).__name__+': '+str(e)})
    require(not errors,'Incomplete/corrupt family before numerical access: '+json.dumps(errors[:20]))
    require(len(census)==696,'All696 COMMITs required');return census,list(payloads.values()),axes_records

def validate_baselines(plan,root,original):
    models={};cells=[c['cell_id'] for c in original['cells']];positions={c:i for i,c in enumerate(cells)}
    for s in plan['sessions']:
        directory=root/plan['namespace_relative']/s['session_id']/'baseline';m=read(directory/'baseline_manifest.json');g=s['baseline_geometry']
        require(m['schema']=='dynavelo_precision_stream_baseline_v1' and m['mode']=='evaluation-fixed' and m['minima_scope']=='full supplied Population, never cell subset or batch','Baseline semantics differ')
        require(m['selected_indices']==[positions[c] for c in g['sample_cell_ids']] and m['batches']==g['batches'] and m['checkpoint_sha256']==g['checkpoint_sha256'],'Baseline geometry/checkpoint differs')
        require(m['CPU_candidate_contract']=={'plan_sha256':plan['plan_sha256'],'device':'cpu','tolerance':s['tolerance'],'tolerance_adapter':plan['tolerance_ast_audit'],'A100_baseline_reused':False},'Separate matching CPU/tolerance baseline differs')
        pop=m['population'];require(pop['cell_ids']==cells and pop['genes']==[g['feature_id'] for g in original['genes']] and pop['motif_tfs']==[v['tf_rna_feature_id'] for v in original['motifs']],'Baseline population axes differ')
        origin=pop['precision_origin']['original_float32_identity'];require(origin['x_sha256']==original['global_minima']['rna_matrix_sha256'] and origin['y_sha256']==original['global_minima']['motif_matrix_sha256'],'Original FP32 input identity differs')
        n=g['n_cells'];shapes={'vx':(n,3006),'vy':(n,365),'vz':(n,100),'time':(n,)};require(set(m['baseline_arrays'])==set(shapes),'Baseline channels differ')
        for name,shape in shapes.items():
            a=mmap_checked(directory/(name+'.npy'),shape,'float64')
            try:require(m['baseline_arrays'][name]['shape']==list(shape) and logical_digest(a)==m['baseline_arrays'][name]['sha256'],'Baseline logical digest differs')
            finally:close(a)
        key=s['seed'];fp=m['model_state_sha256'];require(key not in models or models[key]==fp,'Seed weights differ across sample/tolerance');models[key]=fp

def pair_rows(root,plan,original,frame,s7,s9,inp,comparison,channel,cfg):
    np=_np();seed=s7['seed'];sample=s7['sample_id'];g=s7['baseline_geometry'];ids=g['sample_cell_ids'];width=3006 if channel=='vx' else 365
    require(ids==s9['baseline_geometry']['sample_cell_ids'],'Cross-tolerance sample axes differ');win=np.asarray(window_axis(plan,frame,original,s7),dtype=np.int64)
    weights=np.asarray([v['cell_weight'] for v in frame['samples']['primary']['cells']] if sample=='primary' else [1/len(ids)]*len(ids),dtype=np.float64)
    require(np.isfinite(weights).all() and (weights>0).all() and hashlib.sha256(canonical(weights.tolist())).hexdigest()==g['sample_weights_sha256'],'Frozen cell weights differ')
    def operation(s,h):return next(o for o in s['operations'] if o['kind']=='derivative' and o['input_space']==inp['input_space'] and o['feature_id']==inp['feature_id'] and o['epsilon']==h)
    def folder(s,op):return root/plan['namespace_relative']/s['session_id']/op['operation_id']
    cs=s7 if cfg['current_tolerance']==1e-7 else s9;rs=s7 if cfg['reference_tolerance']==1e-7 else s9
    x=y=None;gates=[];targets=[v['feature_id'] for v in original['genes' if channel=='vx' else 'motifs']]
    try:
        x=mmap_checked(folder(cs,operation(cs,cfg['current_h']))/(channel+'_derivative.npy'),(len(ids),width),'float64')
        y=mmap_checked(folder(rs,operation(rs,cfg['reference_h']))/(channel+'_derivative.npy'),(len(ids),width),'float64')
        if channel=='vx':
            paths=[folder(s7,s7['operations'][0])]+[folder(s7,operation(s7,h)) for h in (1e-5,1e-4,1e-3)]+[folder(s9,s9['operations'][0]),folder(s9,operation(s9,1e-4))]
            for path in paths:
                a=mmap_checked(path/'RNA_active_mask.npy',(len(ids),376),'uint8');gates.append(a);require(not np.any(a[:,-1]&192),'Nonzero unused packed RNA padding bits')
        for start in range(0,width,128):
            stop=min(start+128,width);xt=np.asarray(x[:,start:stop]);yt=np.asarray(y[:,start:stop])
            bits=[unpack_gate_slice(a,start,stop) for a in gates]
            for window in range(5):
                mask=win==window;xx=xt[mask];yy=yt[mask];mass=float(weights[mask].sum());rows=metrics_columns(xx,yy,cfg['current_h'],cfg['reference_h'],comparison)
                gate=gate_columns([a[mask] for a in bits]) if channel=='vx' else [{'gate_changed_cells':None,'gate_review_status':'NOT_APPLICABLE_LINEAR_MOTIF_OUTPUT'}]*(stop-start)
                meansx=(np.sum(xx*weights[mask,None],axis=0)/mass).tolist() if mass else [None]*(stop-start)
                meansy=(np.sum(yy*weights[mask,None],axis=0)/mass).tolist() if mass else [None]*(stop-start)
                for j,m in enumerate(rows):
                    status,reasons=classify_metrics(m,comparison)
                    yield {'comparison':comparison,'seed':seed,'sample_id':sample,'input_space':inp['input_space'],'input_feature':inp['feature_id'],'channel':channel,'target_index':start+j,'target_feature':targets[start+j],'window':window,**cfg,**m,'numeric_status':status,'numeric_reasons':';'.join(reasons),**gate[j],'original_weight_sum':mass,'weighted_current_mean':meansx[j],'weighted_reference_mean':meansy[j]}
    finally:
        for a in (x,y,*gates):
            if a is not None:close(a)

def write_shard(path,rows):
    from contextlib import closing
    hist=Counter();ghist=Counter();n=0
    with path.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as zipped,io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text,closing(rows):
        writer=None
        for row in rows:
            if writer is None:writer=csv.DictWriter(text,fieldnames=list(row),delimiter='\t',lineterminator='\n');writer.writeheader()
            writer.writerow(row);n+=1;hist[row['numeric_status']]+=1;ghist[row['gate_review_status']]+=1
    return n,dict(hist),dict(ghist)

def run(plan_path,contract_path,contract_sha,review_path,review_sha,output):
    root=ROOT.resolve();require(contract_path.is_absolute() and review_path.is_absolute(),'Absolute D contract/review required')
    cr=record(root,contract_path);rr=record(root,review_path);require(cr['sha256']==contract_sha and rr['sha256']==review_sha,'Readout contract/review hash differs')
    contract=read(contract_path);require(contract==build_contract(plan_path),'Exact prospective contract/source differs');review=read(review_path)
    require(review['decision']=='APPROVE_CPU_FIXED28_NUMERICAL_READOUT' and review['contract']==cr and review['source']==record(root,Path(__file__)),'Actual source-bound root readout approval required')
    require(output.is_absolute() and output.resolve().is_relative_to(root/'results') and not output.exists(),'New absolute D results namespace required')
    plan,original,frame=inspect_plan(plan_path);metadata=execution_integrity(plan,root,record(root,plan_path))
    require(shutil.disk_usage(root).free>=6*1024**3,'Need 6GiB free D before compressed full readout');output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();new(output/'STARTED.json',{'contract':cr,'review':rr,'scientific_pass':False})
    try:
        census,payloads,axes=file_payloads(plan,root,record(root,plan_path),original)
        new(output/'complete_file_hash_census.json',{'COMMITs':census,'payloads':payloads,'uncommitted_axes_independently_verified':axes,'execution_and_audit_metadata':metadata,'all_hashes_verified_before_values':True})
        validate_baselines(plan,root,original)
        sessions={(s['seed'],s['sample_id'],s['tolerance']):s for s in plan['sessions']};require(len(sessions)==12,'Duplicate session key')
        total=0;counts=Counter();shards=[];summaries=[]
        for comparison in COMPARISONS:
          cfg=contract['comparisons'][comparison]
          for seed in (0,1,2):
           for sample in ('primary','full_effect_population'):
            s7=sessions[(seed,sample,1e-7)];s9=sessions[(seed,sample,1e-9)]
            for index,inp in enumerate(plan['fixed_inputs']):
             for channel,width in (('vx',3006),('vy',365)):
              require(shutil.disk_usage(root).free>=2*1024**3,'D disk floor reached; preserve incomplete readout')
              dest=output/comparison/f'seed{seed}_{sample}_input{index:02d}_{channel}.tsv.gz';dest.parent.mkdir(exist_ok=True)
              n,hist,ghist=write_shard(dest,pair_rows(root,plan,original,frame,s7,s9,inp,comparison,channel,cfg));require(n==width*5,'Shard target/window denominator differs')
              total+=n;counts[comparison]+=n;identity={'comparison':comparison,'seed':seed,'sample_id':sample,'input_space':inp['input_space'],'feature_id':inp['feature_id'],'channel':channel}
              shards.append({'artifact':record(root,dest),'rows':n,**identity});summaries.append({**identity,'numeric_counts':hist,'gate_review_counts':ghist,'target_or_candidate_PASS_inferred':False})
        require(total==8494920 and len(shards)==1008 and dict(counts)==dict.fromkeys(COMPARISONS,2831640),'Full three-comparison denominator not retained')
        # No COMPLETE output can survive a changed source after the scan.
        for rec in [*census,*payloads,*axes,*metadata]:checked(root,rec)
        new(output/'all_status_counts.json',{'rows':summaries,'no_averaging_away':True,'all3seed_records_retained':True})
        result={'schema':'dynavelo_cpu_fixed28_numerical_readout_v1','status':'COMPLETE_CPU_PER_CELL_READOUT_ROOT_ADJUDICATION_PENDING','total_records':total,'records_per_comparison':dict(counts),'shards':shards,'contract':cr,'source':record(root,Path(__file__)),'complete_census':record(root,output/'complete_file_hash_census.json'),'status_counts':record(root,output/'all_status_counts.json'),'source_hashes_reverified_after_values':True,'seconds':time.monotonic()-started,'original_16_HOLD_unchanged':True,'A100_per_cell_bridge':'NOT_TESTED_UNAVAILABLE','A100_PASSmap_created':False,'scientific_pass':False,'entire_CPU_backend_qualified':False}
        new(output/'COMPLETE.json',result);return {k:v for k,v in result.items() if k!='shards'}
    except BaseException as e:
        new(output/'HOLD.json',{'status':'HOLD_READOUT_INCOMPLETE','error_type':type(e).__name__,'error':str(e),'seconds':time.monotonic()-started,'scientific_pass':False});raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',default=PLAN_SHA)
    p.add_argument('--prepare-contract',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--contract',type=Path);p.add_argument('--contract-sha');p.add_argument('--review',type=Path);p.add_argument('--review-sha');p.add_argument('--output',type=Path);a=p.parse_args()
    require(a.plan_sha==PLAN_SHA,'Exact original CPU plan SHA required')
    if a.execute:
        require(all((a.contract,a.contract_sha,a.review,a.review_sha,a.output)),'Explicit contract/review/output required')
        value=run(a.plan,a.contract,a.contract_sha,a.review,a.review_sha,a.output)
    else:
        value=build_contract(a.plan)
        if a.prepare_contract:
            require(a.prepare_contract.is_absolute() and a.prepare_contract.resolve().is_relative_to(ROOT/'results/reports'),'New absolute D contract report required');new(a.prepare_contract,value)
        value={'status':'INSPECTED_SOURCE_METADATA_ONLY_NO_RAW_VALUES','contract':value}
    print(json.dumps(value,allow_nan=False))
if __name__=='__main__':main()
