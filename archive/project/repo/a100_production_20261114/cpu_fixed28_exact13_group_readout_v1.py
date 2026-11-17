"""Exact13-group CPU readout: frozen selectors, original metrics, no models/cloud."""
from pathlib import Path
from contextlib import ExitStack
import argparse,ast,hashlib,importlib.util,json,os,shutil,time
R=Path(r'D:\dyc_7\DynaVelo_Research');C=R/'repo/a100_production_20261114'
PLAN=R/'results/reports/cpu_candidate_preparation_20261116_v1/plan.json'
BASE=C/'cpu_fixed28_readout_v1.py';GROUP=R/'repo/colab/screen_scope_followup_numerics_v2.py'
F=R/'results/reports/cpu_exact13_group_feasibility_20261116_v1'
SELECTORS=F/'selectors.json';PROPOSAL=F/'prospective_contract.json'
PRIOR=R/'results/cpu_fixed28_numerical_readout_20261116_v1'
PREP=R/'results/reports/cpu_fixed28_exact13_group_readout_preparation_20261116_v1'
PINS={BASE:'2674d1d87d9980dbe77dcc78aeca0b46e6e548e9fb340a8219bc7d58bf9f1796',GROUP:'ba3d387adc939bc0b644bade8efec12e8a48fb05a70060b538ee45ddc1c05e03',SELECTORS:'ac2a5f2d5278da228ee0a1533813306595da133b1595df4624cce12edbc5c969',PROPOSAL:'6eb6d20721acbd1a43bb311b46a5f7a70a4cb9791dd33f4b92a92fe355b66dba',PRIOR/'COMPLETE.json':'e36d20024c8b1185da699b5e93a8575a1b279519c736c150e3464c95af978271'}
CONFIG={'small_step':(1e-7,1e-5,1e-7,1e-4),'large_step':(1e-7,1e-3,1e-7,1e-4),'tolerance':(1e-7,1e-4,1e-9,1e-4)}
CFG_FIELDS=('current_tolerance','current_h','reference_tolerance','reference_h')
STATUS={'STABLE_UNDER_NEW_ENGINEERING_CRITERIA':0,'HOLD':1,'LOW_NUMERICAL_SIGNAL':2,'NOT_TESTED_EMPTY_GROUP':3}
REASONS={'relative_L2_missing_or_exceeds_limit':1,'no_entries_meet_union_strong_mask':2,'strong_union_sign_below_limit':4,'reference_RMS_at_or_below_1e-8':8,'empty_original_exact_group':16}
INTS={'n_entries','zero_zero','current_zero_reference_nonzero','current_nonzero_reference_zero','opposite_nonzero_sign','same_nonzero_sign','strong_entry_count','strong_entry_same_sign_count'}
BOOLS={'zero_reference_L2','no_strong_entries'}
LIMITATION='AuthorDC exact group only, not subtype or DC x window. No A100 per-cell bridge/map, whole CPU, TF/gene/biological PASS. Original16HOLD unchanged.'
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p):
    p=Path(p).resolve();require(p.is_relative_to(R.resolve()),'D project required');return {'relative_path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def check(v):
    p=R/v['relative_path'];require(not p.is_symlink() and p.resolve()==p.absolute() and rec(p)==v,'Changed/redirected artifact: '+v['relative_path']);return p
def new(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def same_records(left,right):
    a={v['relative_path']:v for v in left};b={v['relative_path']:v for v in right}
    return len(a)==len(left) and len(b)==len(right) and a==b
def load_base():
    for p,h in PINS.items():require(sha(p)==h,'Pinned metadata/source changed: '+str(p))
    s=importlib.util.spec_from_file_location('exact13_base',BASE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def metadata(base):
    """Only source/JSON and metadata arrays, never np.load or raw payload hashes."""
    import numpy as np
    plan,spec,frame=base.inspect_plan(PLAN);original=read(check(plan['original_scope_plan']));frozen=read(SELECTORS);groups=frozen['groups_by_seed_sample']
    nodes=[v for v in ast.parse(GROUP.read_text(encoding='utf-8')).body if isinstance(v,ast.FunctionDef) and v.name in ('frame_weights','group_definitions')]
    require([v.name for v in nodes]==['frame_weights','group_definitions'],'Original metadata functions required');scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(GROUP),'exec'),scope)
    sessions={(s['seed'],s['sample_id'],s['tolerance']):s for s in plan['sessions']};positions={c['cell_id']:i for i,c in enumerate(spec['cells'])}
    require([(g['seed'],g['sample_id']) for g in groups]==[(s,k) for s in (0,1,2) for k in ('primary','full_effect_population')],'All6 geometries in order required')
    for row in groups:
        seed,sample=row['seed'],row['sample_id'];g=sessions[(seed,sample,1e-7)]['baseline_geometry'];ids=row['sample_cell_ids'];n=192 if sample=='primary' else 1884
        require(len(ids)==len(set(ids))==n and ids==g['sample_cell_ids']==sessions[(seed,sample,1e-9)]['baseline_geometry']['sample_cell_ids'],'Frozen sample axes differ')
        expected=[v['cell_id'] for v in frame['samples']['primary']['cells']] if sample=='primary' else frame['effect_eligible_cell_ids'];require(ids==expected,'Original sample order differs')
        indices=np.asarray([positions[c] for c in ids],dtype=np.int64);task={'seed':seed,'sample_id':sample,'sample_cell_ids':ids};w=scope['frame_weights'](original,frame,task,indices)
        require(w.tolist()==row['sample_weights'] and np.isfinite(w).all() and (w>0).all() and hashlib.sha256(base.canonical(w.tolist())).hexdigest()==g['sample_weights_sha256'],'Frozen weights differ')
        defs=scope['group_definitions'](original,frame,task,indices)
        expected=[{'group_index':i,'kind':kind,'name':name,'sample_n':int(mask.sum()),'effect_frame_domain_n':int(domain),'sample_weight_sum':float(w[mask].sum()),'sample_positions':np.flatnonzero(mask).tolist()} for i,(kind,name,mask,domain) in enumerate(defs)]
        require(len(expected)==13 and expected==row['groups'],'Exact original13 selector differs')
    require(all([(g['kind'],g['name']) for g in x['groups']]==[(g['kind'],g['name']) for g in groups[0]['groups']] for x in groups),'Group identities/order differ')
    complete=read(PRIOR/'COMPLETE.json');prior=read(check(complete['complete_census']));require(complete['total_records']==8494920 and len(prior['COMMITs'])==696 and len(prior['payloads'])==2088,'Complete prior census metadata required')
    return plan,spec,frame,groups,prior,complete['complete_census']
def contract(base,plan,spec,groups,census):
    n=22086792;units=n//9;metric_bytes=sum(1 if k in BOOLS else 8 for k in base.METRICS)
    output_bytes=n*(metric_bytes+2+16)+(n//3)*9+n+(n//3)*9+units*6
    return {'schema':'dynavelo_cpu_fixed28_exact13_group_readout_contract_v1','status':'PROSPECTIVE_SOURCE_PREPARED_ROOT_REVIEW_REQUIRED',
      'source':rec(__file__),'cpu_plan':rec(PLAN),'source_and_metadata_records':[rec(p) for p in PINS],
      'original_scope_plan':plan['original_scope_plan'],'scope_frame':plan['scope_frame'],'prior_raw_census':census,
      'selectors':rec(SELECTORS),'thresholds':base.LIMITS,'comparisons':{k:dict(zip(CFG_FIELDS,v)) for k,v in CONFIG.items()},
      'fixed_inputs':plan['fixed_inputs'],'rna_target_axis':[{'feature_id':g['feature_id'],'stable_id':g['stable_id']} for g in spec['genes']],
      'motif_target_axis':spec['motifs'],'group_order':[{k:g[k] for k in ('group_index','kind','name')} for g in groups[0]['groups']],
      'expected_records':n,'all9_target_group_units':units,'case_channel_shards':336,'all9_support_shards':112,
      'metric_order':list(base.METRICS),'metric_dtypes':{k:'int64' if k in INTS else 'uint8' if k in BOOLS else 'float64' for k in base.METRICS},
      'numeric_status_codes':STATUS,'reason_bitmask':REASONS,'gate_codes':{'NO_CAPTURED_DECODER_GATE_CHANGE':0,'HOLD_FOR_GATE_REVIEW':1,'NOT_TESTED_EMPTY_GROUP':2,'NOT_APPLICABLE_LINEAR_MOTIF_OUTPUT':3},
      'layout':'336 uncompressedNPZ: every metric/status/reason/secondarymean[comparison,group,target],gatecount/status[group,target].112ALL9NPZ: numeric[seed,comparison,group,target],gatecount/status[seed,group,target],six flags[group,target].',
      'null_encoding':'floatNaN;int64 -1;nullablebool255. Empty n_entries0; actual zero stays zero. EmptyRNA gatecount0; vy inapplicable count-1.',
      'case_order':'seed012,primary/full_effect_population,fixed28 input order,vx/vy; comparisons small_step,large_step,tolerance; original13groups',
      'all9_rule':'All9 numeric codes0 and no seed gateHOLD. Only ENGINEERING_STABLE_FOR_THIS_TARGET_EXACT_GROUP. Independent anyHOLD/LOW/EMPTY/gateHOLD/all9numericstable retained.',
      'seed_specific_groups':'Each seed uses its own frozen n/selectors/masses; no cross-seed n equality or mean-based qualification.',
      'gate_rule':read(PROPOSAL)['gate_rule'],'integrity':'Reuse original696COMMIT/2088payload prehash/210audit/matching CPU baseline logical checks; exact prior census match; posthash all inputs/sources and output artifacts.',
      'resource_limits':{'startup_D_free_bytes':8*1024**3,'running_D_free_bytes':2*1024**3,'own_RSS_bytes':2*1024**3,'system_available_bytes':2*1024**3,'cooperative_wall_seconds':3600,'target_chunk':128},
      'numeric_array_payload_bytes':output_bytes,'output_bytes_upper':output_bytes+20*1024**2,
      'prospective_refinement':'Adds exact original13 groups after whole-window analysis, without changing original five-window contract/results. Not retrospective preregistration.',
      'optional_A100_13group_consistency':'Separate optional aggregate comparison after complete native family; never per-cell bridge PASS.',
      'scope_limit':LIMITATION,'original16HOLD_unchanged':True,'A100_per_cell_bridge':'NOT_TESTED_UNAVAILABLE','A100_PASSmap_created':False,'scientific_pass':False,
      'raw_array_or_gzip_values_read_during_preparation':False,'model_or_cloud_calls':False}
def allocate(base,width):
    import numpy as np
    out={k:np.full((3,13,width),-1 if k in INTS else 255 if k in BOOLS else np.nan,dtype=np.int64 if k in INTS else np.uint8 if k in BOOLS else np.float64) for k in base.METRICS}
    out.update(numeric_status=np.full((3,13,width),255,dtype=np.uint8),numeric_reason_bits=np.zeros((3,13,width),dtype=np.uint8),
      gate_changed_cells=np.full((13,width),-1,dtype=np.int64),gate_status=np.full((13,width),255,dtype=np.uint8),
      weighted_current_mean=np.full((3,13,width),np.nan),weighted_reference_mean=np.full((3,13,width),np.nan));return out
def store_metrics(base,out,ci,gi,left,right,x,y,weights,mass,comparison,cfg):
    import numpy as np
    rows=base.metrics_columns(x,y,cfg[1],cfg[3],comparison)
    for k in base.METRICS:out[k][ci,gi,left:right]=[(-1 if k in INTS else 255 if k in BOOLS else np.nan) if row[k] is None else row[k] for row in rows]
    for j,row in enumerate(rows):
        status,reasons=base.classify_metrics(row,comparison)
        if row['n_entries']==0:status,reasons='NOT_TESTED_EMPTY_GROUP',['empty_original_exact_group']
        out['numeric_status'][ci,gi,left+j]=STATUS[status];out['numeric_reason_bits'][ci,gi,left+j]=sum(REASONS[v] for v in reasons)
    if mass:
        xmean=np.sum(x*weights[:,None],axis=0)/mass;ymean=np.sum(y*weights[:,None],axis=0)/mass
        require(np.isfinite(xmean).all() and np.isfinite(ymean).all(),'Nonfinite secondary mean')
        out['weighted_current_mean'][ci,gi,left:right]=xmean;out['weighted_reference_mean'][ci,gi,left:right]=ymean
def support_flags(numeric,gate_status,gate_counts):
    import numpy as np
    require(numeric.ndim==4 and numeric.shape[:3]==(3,3,13) and gate_status.shape==gate_counts.shape==(3,13,numeric.shape[-1]),'Complete3seed x3comparison exact13 axes required')
    require(np.isin(numeric,list(STATUS.values())).all() and np.isin(gate_status,[0,1,2,3]).all(),'Invalid evidence code')
    all9=np.all(numeric==0,axis=(0,1));gate=np.any(gate_status==1,axis=0)
    return {'all9_numeric_status':numeric,'all3seed_gate_status':gate_status,'all3seed_gate_changed_cells':gate_counts,
      'any_numeric_HOLD':np.any(numeric==1,axis=(0,1)),'any_numeric_LOW':np.any(numeric==2,axis=(0,1)),
      'any_numeric_EMPTY':np.any(numeric==3,axis=(0,1)),'any_gate_HOLD':gate,'all9_numeric_stable':all9,
      'ENGINEERING_STABLE_FOR_THIS_TARGET_EXACT_GROUP':all9&~gate}
def publish_arrays(path,arrays):
    import numpy as np
    path=Path(path);partial=path.with_name(path.name+'.partial');require(not path.exists() and not partial.exists(),'No overwrite/replay')
    with partial.open('xb') as f:np.savez(f,**arrays)
    os.link(partial,path);partial.unlink();return rec(path)
class Guard:
    def __init__(self,limits):self.limits=limits;self.started=time.monotonic();self.peak=0
    def check(self):
        import psutil
        rss=psutil.Process().memory_info().rss;self.peak=max(self.peak,rss)
        require(rss<=self.limits['own_RSS_bytes'],'Own readout RSS ceiling')
        require(psutil.virtual_memory().available>=self.limits['system_available_bytes'],'Host available-memory floor; no other process touched')
        require(shutil.disk_usage(R).free>=self.limits['running_D_free_bytes'],'D free-space floor')
        require(time.monotonic()-self.started<=self.limits['cooperative_wall_seconds'],'Cooperative readout wall')
def emit_case(base,plan,geometry,s7,s9,inp,channel,guard):
    import numpy as np
    ids=geometry['sample_cell_ids'];n=len(ids);width=3006 if channel=='vx' else 365;weights=np.asarray(geometry['sample_weights'],dtype=np.float64);out=allocate(base,width)
    def op(s,h):return next(o for o in s['operations'] if o['kind']=='derivative' and o['input_space']==inp['input_space'] and o['feature_id']==inp['feature_id'] and o['epsilon']==h)
    def folder(s,o):return R/plan['namespace_relative']/s['session_id']/o['operation_id']
    with ExitStack() as stack:
      def mapped(path,shape,dtype):
        value=base.mmap_checked(path,shape,dtype);stack.callback(base.close,value);return value
      # Four read-only mmap handles, one pair of128-target group tiles evaluated at a time.
      d={('t7',h):mapped(folder(s7,op(s7,h))/(channel+'_derivative.npy'),(n,width),'float64') for h in (1e-5,1e-4,1e-3)}
      d[('t9',1e-4)]=mapped(folder(s9,op(s9,1e-4))/(channel+'_derivative.npy'),(n,width),'float64')
      masks=[]
      if channel=='vx':
        paths=[folder(s7,s7['operations'][0])]+[folder(s7,op(s7,h)) for h in (1e-5,1e-4,1e-3)]+[folder(s9,s9['operations'][0]),folder(s9,op(s9,1e-4))]
        masks=[mapped(p/'RNA_active_mask.npy',(n,376),'uint8') for p in paths]
        require(all(not np.any(a[:,-1]&192) for a in masks),'Invalid packed RNA padding')
      for left in range(0,width,128):
        guard.check();right=min(left+128,width);tiles={k:np.asarray(a[:,left:right]) for k,a in d.items()};bits=[base.unpack_gate_slice(a,left,right) for a in masks]
        for gi,g in enumerate(geometry['groups']):
          select=np.asarray(g['sample_positions'],dtype=np.int64);w=weights[select];mass=g['sample_weight_sum'];require(len(select)==g['sample_n'],'Group selector n differs')
          gate=base.gate_columns([b[select] for b in bits]) if channel=='vx' else None
          out['gate_changed_cells'][gi,left:right]=[v['gate_changed_cells'] for v in gate] if gate is not None else -1
          out['gate_status'][gi,left:right]=[2 if g['sample_n']==0 else 1 if v['gate_review_status']=='HOLD_FOR_GATE_REVIEW' else 0 for v in gate] if gate is not None else 3
          for ci,(comparison,cfg) in enumerate(CONFIG.items()):
            x=tiles[('t7',cfg[1])][select];y=tiles[('t9' if cfg[2]==1e-9 else 't7',cfg[3])][select]
            store_metrics(base,out,ci,gi,left,right,x,y,w,mass,comparison,cfg)
    require(np.isin(out['numeric_status'],list(STATUS.values())).all() and np.isin(out['gate_status'],[0,1,2,3]).all(),'Unwritten status entries')
    guard.check();return out
def progress(out,value):
    p=out/'progress.json';tmp=out/'progress.json.partial'
    new(tmp,value);os.replace(tmp,p)
def execute(a,base,plan,spec,frame,groups,prior,expected):
    import numpy as np
    cp,rp,out=Path(a.contract),Path(a.review),Path(a.output)
    require(all(p.is_absolute() and p.resolve().is_relative_to(R.resolve()) for p in (cp,rp,out)),'Absolute D paths required')
    cr,rr=rec(cp),rec(rp);check(cr);check(rr)
    require(cr['sha256']==a.contract_sha and rr['sha256']==a.review_sha and read(cp)==expected,'Exact contract/review bytes required')
    review=read(rp);require(review['decision']=='APPROVE_CPU_FIXED28_EXACT13_GROUP_READOUT' and review['contract']==cr and review['source']==expected['source'],'Actual source-bound root approval required')
    require(out.resolve()==out.absolute() and out.is_relative_to(R/'results') and not out.exists(),'New unredirected D results output required')
    limits=expected['resource_limits'];require(shutil.disk_usage(R).free>=limits['startup_D_free_bytes'],'Need8GiB D free');guard=Guard(limits);guard.check()
    out.mkdir(parents=True);new(out/'STARTED.json',{'source':expected['source'],'contract':cr,'review':rr,'phase':'PREHASH_NO_NUMERIC_ARRAY_ACCESS_YET','scientific_pass':False})
    started=time.monotonic();artifacts=[];supports={};counts=[];total=0
    try:
      metadata_records=base.execution_integrity(plan,R,rec(PLAN));census,payloads,axes=base.file_payloads(plan,R,rec(PLAN),spec)
      require(same_records(census,prior['COMMITs']) and same_records(payloads,prior['payloads']) and same_records(axes,prior['uncommitted_axes_independently_verified']) and same_records(metadata_records,prior['execution_and_audit_metadata']),'Exact original raw census changed')
      guard.check();new(out/'raw_census.json',{'COMMITs':census,'payloads':payloads,'axes':axes,'execution_and_audit_metadata':metadata_records,'all_hashes_verified_before_numeric_mmap':True})
      # First numeric mmap access occurs after the complete-family physical hashes above.
      base.validate_baselines(plan,R,spec);guard.check();sessions={(s['seed'],s['sample_id'],s['tolerance']):s for s in plan['sessions']}
      for g in groups:
       seed,sample=g['seed'],g['sample_id'];s7,s9=sessions[(seed,sample,1e-7)],sessions[(seed,sample,1e-9)]
       for i,inp in enumerate(plan['fixed_inputs']):
        for channel,width in (('vx',3006),('vy',365)):
          guard.check();arrays=emit_case(base,plan,g,s7,s9,inp,channel,guard);ar=publish_arrays(out/f'seed{seed}_{sample}_input{i:02d}_{channel}.npz',arrays)
          identity={'seed':seed,'sample_id':sample,'input_index':i,'input_space':inp['input_space'],'input_feature':inp['feature_id'],'channel':channel}
          artifacts.append({'artifact':ar,**identity});codes=arrays['numeric_status']
          counts.append({**identity,'numeric_counts':{name:int(np.count_nonzero(codes==code)) for name,code in STATUS.items()},'gate_HOLD':int(np.count_nonzero(arrays['gate_status']==1))});total+=codes.size
          entry=supports.setdefault((sample,i,channel),{'numeric':{},'gate':{},'gate_counts':{},'case_artifacts':[]});require(seed not in entry['numeric'],'Duplicate seed case')
          entry['numeric'][seed]=codes;entry['gate'][seed]=arrays['gate_status'];entry['gate_counts'][seed]=arrays['gate_changed_cells'];entry['case_artifacts'].append(ar)
          del arrays
          progress(out,{'phase':'CASE_SHARDS','completed_case_channel_shards':len(artifacts),'expected':336,'records':total,'seconds':time.monotonic()-started,'own_peak_RSS_bytes':guard.peak,'scientific_pass':False})
      require(total==22086792 and len(artifacts)==336 and len(supports)==112,'Complete fixed28/group family denominator differs')
      support_artifacts=[];support_units=0
      for (sample,i,channel),entry in supports.items():
        guard.check();require(set(entry['numeric'])==set(entry['gate'])==set(entry['gate_counts'])=={0,1,2},'All3 seed evidence cases required')
        numeric=np.stack([entry['numeric'][s] for s in (0,1,2)]);gate=np.stack([entry['gate'][s] for s in (0,1,2)]);gate_counts=np.stack([entry['gate_counts'][s] for s in (0,1,2)])
        flags=support_flags(numeric,gate,gate_counts);ar=publish_arrays(out/f'ALL9_{sample}_input{i:02d}_{channel}.npz',flags);support_units+=flags['all9_numeric_stable'].size
        support_artifacts.append({'artifact':ar,'sample_id':sample,'input_index':i,'channel':channel,'case_artifacts':entry['case_artifacts']})
      require(support_units==2454088 and len(support_artifacts)==112,'All9 support denominator differs')
      progress(out,{'phase':'POSTHASH','case_shards':336,'support_shards':112,'records':total,'scientific_pass':False})
      records=[*census,*payloads,*axes,*metadata_records,cr,rr,expected['source'],expected['cpu_plan'],expected['original_scope_plan'],expected['scope_frame'],expected['prior_raw_census'],*expected['source_and_metadata_records'],*plan['source_records']]
      for r in records:check(r)
      for row in artifacts+support_artifacts:check(row['artifact'])
      guard.check();new(out/'status_counts.json',{'rows':counts,'all_group_target_records_retained':True,'flags_independent_no_averaging_away':True})
      result={'schema':'dynavelo_cpu_fixed28_exact13_group_engineering_support_v1','status':'COMPLETE_EXACT13_GROUP_CPU_ENGINEERING_SUPPORT_ROOT_REVIEW_PENDING','records':total,'all9_target_group_units':support_units,
        'case_channel_shards':artifacts,'all9_support_shards':support_artifacts,'source':expected['source'],'contract':cr,'review':rr,'selectors':expected['selectors'],'raw_census':rec(out/'raw_census.json'),'status_counts':rec(out/'status_counts.json'),
        'all_input_source_output_hashes_reverified':True,'seconds':time.monotonic()-started,'own_peak_RSS_bytes':guard.peak,'original16HOLD_unchanged':True,'A100_per_cell_bridge':'NOT_TESTED_UNAVAILABLE','A100_PASSmap_created':False,'entire_CPU_backend_qualified':False,'TF_gene_biological_PASS':False,'scientific_pass':False,'scope_limit':LIMITATION}
      new(out/'COMPLETE.json',result);return {k:result[k] for k in ('status','records','all9_target_group_units','seconds','own_peak_RSS_bytes','scientific_pass')}
    except BaseException as e:new(out/'HOLD.json',{'status':'HOLD_EXACT13_GROUP_READOUT_INCOMPLETE','error_type':type(e).__name__,'error':str(e),'scientific_pass':False});raise
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    for key in ('contract','contract-sha','review','review-sha','output'):p.add_argument('--'+key)
    a=p.parse_args();base=load_base();plan,spec,frame,groups,prior,census=metadata(base);expected=contract(base,plan,spec,groups,census)
    if a.prepare:
        require(not a.execute and not PREP.exists(),'New metadata preparation required');PREP.mkdir(parents=True);new(PREP/'contract.json',expected)
        value={'status':'PREPARED_SOURCE_METADATA_ONLY_NO_RAW_ARRAY_BYTES_READ','contract':rec(PREP/'contract.json')}
    elif a.execute:
        require(all((a.contract,a.contract_sha,a.review,a.review_sha,a.output)),'Exact contract/review/output arguments required');value=execute(a,base,plan,spec,frame,groups,prior,expected)
    else:value={'status':'METADATA_ONLY_NO_RAW_ARRAY_ACCESS','contract':expected}
    print(json.dumps(value,allow_nan=False))
if __name__=='__main__':main()
