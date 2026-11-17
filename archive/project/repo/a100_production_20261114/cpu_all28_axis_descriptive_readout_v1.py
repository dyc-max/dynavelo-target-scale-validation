"""Prospective descriptive sensitivity readout. Preparation never opens actual NPZ."""
from pathlib import Path
import argparse,csv,hashlib,json,os,shutil,time
R=Path(r'D:\dyc_7\DynaVelo_Research');C=R/'repo/a100_production_20261114'
UP=C/'cpu_fixed28_exact13_group_readout_v1.py';UP_SHA='a8aa26c595175650bff86d9fb084d3eede3b5a7b1ca9d25095516ac5ec79ddf8'
UC=R/'results/reports/cpu_fixed28_exact13_group_readout_preparation_20261116_v1/contract.json'
UO=R/'results/cpu_fixed28_exact13_group_readout_20261116_v1'
PREP=R/'results/reports/cpu_all28_axis_readout_preparation_20261116_v1'
SAMPLES=('primary','full_effect_population');WIDTH={'vx':3006,'vy':365}
SUPPORT=('all9_numeric_status','all3seed_gate_status','all3seed_gate_changed_cells','any_numeric_HOLD','any_numeric_LOW','any_numeric_EMPTY','any_gate_HOLD','all9_numeric_stable','ENGINEERING_STABLE_FOR_THIS_TARGET_EXACT_GROUP')
MS={'MEAN_SIGN_CONSISTENT_AT_EXISTING_STRONG_SCALE':0,'MEAN_HAS_NONFINITE':1,'MEAN_HAS_EXACT_ZERO':2,'MEAN_HAS_LOW_NONZERO_AT_EXISTING_SCALE':3,'MEAN_HAS_MIXED_NONZERO_SIGNS':4}
LIMIT='Signed local velocity sensitivity only: not activation/repression, causal, joint/rescue, new TF screen, external P, scientific PASS or A100 bridge. Original16HOLD and E1 unchanged. Original13 groups overlap; branches are not independent or label x time.'
def require(v,m):
 if not v:raise ValueError(m)
def safe(p):
 p=Path(p);require(p.is_absolute() and p.resolve().is_relative_to(R.resolve()) and p.resolve()==p.absolute() and not p.is_symlink(),'Absolute unredirected D path required');return p
def read(p):return json.loads(safe(p).read_text(encoding='utf-8'))
def sha(p):
 with safe(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p):
 p=safe(p);return {'relative_path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def check(r):
 p=safe(R/r['relative_path']);require(rec(p)==r,'Artifact changed: '+r['relative_path']);return p
def new(p,v):
 with safe(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def metadata_plan():
 """Pinned Python/JSON metadata only; no runner import, NPZ or raw arrays."""
 require(sha(UP)==UP_SHA,'Upstream source pin changed');u=read(UC);require(u['source']==rec(UP),'Source contract mismatch')
 require(len(u['fixed_inputs'])==28 and sum(i['input_space']=='rna' for i in u['fixed_inputs'])==20 and sum(i['input_space']=='motif' for i in u['fixed_inputs'])==8,'Fixed20RNA8motif required')
 require(len(u['rna_target_axis'])==3006 and len(u['motif_target_axis'])==365 and len(u['group_order'])==13 and u['all9_target_group_units']==2454088,'Alltargets/exact13 denominator differs')
 require([g['group_index'] for g in u['group_order']]==list(range(13)) and [sum(g['kind']==k for g in u['group_order']) for k in ('effect_population','label','context_branch_intersect_effect','full_context_time_window')]==[1,3,4,5],'Original1+3+4+5 group order/kinds required')
 require(u['thresholds']['strong_union_absolute_threshold']==1e-7,'Existing strong threshold required')
 cfg={'small_step':[1e-7,1e-5,1e-7,1e-4],'large_step':[1e-7,1e-3,1e-7,1e-4],'tolerance':[1e-7,1e-4,1e-9,1e-4]};fields=('current_tolerance','current_h','reference_tolerance','reference_h')
 require(list(u['comparisons'])==list(cfg) and {k:[v[f] for f in fields] for k,v in u['comparisons'].items()}==cfg,'Frozen comparison order/config differs')
 selectors=read(check(u['selectors']));gg=[{'seed':g['seed'],'sample_id':g['sample_id'],'groups':[{k:x[k] for k in ('group_index','kind','name','sample_n','effect_frame_domain_n','sample_weight_sum')} for x in g['groups']]} for g in selectors['groups_by_seed_sample']]
 require([(g['seed'],g['sample_id']) for g in gg]==[(s,a) for s in (0,1,2) for a in SAMPLES],'All6 original group geometries required')
 return {'schema':'dynavelo_cpu_all28_axis_descriptive_plan_v1','status':'PROSPECTIVE_SOURCE_PREPARED_ROOT_APPROVAL_REQUIRED','source':rec(__file__),'upstream_source':rec(UP),'upstream_contract':rec(UC),'selectors':u['selectors'],'actual_COMPLETE_binding':'REQUIRED_LATER_METADATA_BIND_AND_SOURCE_PLAN_COMPLETE_ROOT_APPROVAL','upstream_COMPLETE_expected_path':str(UO/'COMPLETE.json'),'fixed_inputs':u['fixed_inputs'],'target_axes':{'vx':u['rna_target_axis'],'vy':u['motif_target_axis']},'group_order':u['group_order'],'seed_specific_groups':gg,'comparisons':u['comparisons'],'numeric_status_codes':u['numeric_status_codes'],'gate_codes':u['gate_codes'],'expected_case_shards':336,'expected_support_shards':112,'expected_descriptive_shards':112,'all_target_group_units':2454088,
 'unique_mean_routes':['current[small0]:tol1e-7/h1e-5','reference[small0]==reference[large1]==current[tolerance2]:tol1e-7/h1e-4','current[large1]:tol1e-7/h1e-3','reference[tolerance2]:tol1e-9/h1e-4'],
 'raw_means':'unique_weighted_means[seed3,route4,group13,target]; preserve all12 raw values. Duplicated route arrays must match exactly with equal NaNs; source448 remain immutable.',
 'rank':{'score':'median(abs(strict reference route3 across seeds012))','all3_finite_required':True,'NaN':'unranked retained; never nanmedian','ties':'descending average rank; exact score ties; target index only table presentation','scope':'per input/sample/channel/original group, no RNA-motif cross-unit comparison'},
 'mean_direction':{'existing_strong':1e-7,'rule':'all4 routes x3 seeds finite,abs>=existing1e-7,same nonzero sign','status_codes':MS,'all_flags_independent':True,'does_not_clear_ALL9_or_gate_HOLD_LOW':True},
 'top50':{'rule':'raw strict-reference score, no numerical/gate/direction filtering; k=min(50,finite targets), include all exact boundary ties','nonfinite':'alltarget retained, unranked count explicit, no imputation/refill','layout':'112 TSVs each containing separate tables for all13 groups','HOLD':'keep original raw rank; do not remove and refill'},
 'support':{'keys_copied_unchanged':list(SUPPORT),'verify_against_all3_cases':True},'units':{'rna':'signed velocity sensitivity per RNA input unit','motif':'signed velocity sensitivity per motif input unit; separate scale','channels':'vx RNA targets; vy motif targets; not joint or rescue'},
 'integrity':'Bind actual COMPLETE SHA; physical SHA all336+112 pre/post; recheck source/plan/binding/review/COMPLETE. Do not rehash original19GB raw.',
 'output_size_estimate':{'numeric_array_payload_bytes':2454088*165,'plus':'target/group strings and ZIP headers; compressed112NPZ usually smaller','top50_worst_case':'boundary ties may retain every target; never truncate ties to meet a size target'},
 'resource_limits':{'startup_D_free_bytes':8*1024**3,'running_D_free_bytes':2*1024**3,'own_RSS_bytes':2*1024**3,'system_available_bytes':2*1024**3,'wall_seconds':3600},'scope_limit':LIMIT,'E1_unchanged':True,'original16HOLD_unchanged':True,'A100_bridge':'NOT_TESTED_UNAVAILABLE','scientific_pass':False,'actual_NPZ_opened_during_preparation':False}
def average_descending_ranks(score):
 import numpy as np
 score=np.asarray(score,dtype=np.float64);require(score.ndim==1,'One group vector required');r=np.full(score.shape,np.nan);f=np.flatnonzero(np.isfinite(score));order=f[np.argsort(-score[f],kind='stable')];a=0
 while a<len(order):
  b=a+1
  while b<len(order) and score[order[b]]==score[order[a]]:b+=1
  r[order[a:b]]=(a+1+b)/2;a=b
 return r
def boundary_top(score,k=50):
 import numpy as np
 score=np.asarray(score,dtype=np.float64);f=np.flatnonzero(np.isfinite(score));order=f[np.argsort(-score[f],kind='stable')]
 return order if not len(order) else order[score[order]>=score[order[min(k,len(order))-1]]]
def derive(unique,strong):
 import numpy as np
 a=np.asarray(unique);require(a.ndim==4 and a.shape[:3]==(3,4,13) and a.dtype==np.float64,'3seed x4route x13group float64 required')
 ref=a[:,3];finite3=np.isfinite(ref).all(axis=0);score=np.full(ref.shape[1:],np.nan);score[finite3]=np.median(np.abs(ref),axis=0)[finite3]
 f=np.isfinite(a);allfinite=f.all(axis=(0,1));zero=(f&(a==0)).any(axis=(0,1));low=(f&(np.abs(a)>0)&(np.abs(a)<strong)).any(axis=(0,1));mixed=(f&(a>0)).any(axis=(0,1))&(f&(a<0)).any(axis=(0,1));allstrong=(f&(np.abs(a)>=strong)).all(axis=(0,1));pos=(f&(a>0)).all(axis=(0,1));neg=(f&(a<0)).all(axis=(0,1));consistent=allstrong&(pos|neg)
 code=np.full(score.shape,4,dtype=np.uint8);code[consistent]=0;code[low]=3;code[zero]=2;code[~allfinite]=1
 return {'reference_median_abs':score,'reference_average_descending_rank':np.stack([average_descending_ranks(x) for x in score]),'reference_all3_finite':finite3,'mean_direction_status':code,'mean_all12_finite':allfinite,'mean_any_nonfinite':~allfinite,'mean_any_exact_zero':zero,'mean_any_low_nonzero_at_existing_strong':low,'mean_any_mixed_nonzero_signs':mixed,'mean_all12_abs_at_least_existing_strong':allstrong,'mean_all12_positive':pos,'mean_all12_negative':neg,'MEAN_SIGN_CONSISTENT_AT_EXISTING_STRONG_SCALE':consistent}
def equal_nan(a,b):
 import numpy as np
 return np.array_equal(a,b,equal_nan=True)
def validate_support(support,numeric,gate,counts):
 import numpy as np
 require(set(support)==set(SUPPORT) and numeric.shape[:3]==(3,3,13) and gate.shape==counts.shape==(3,13,numeric.shape[-1]),'Complete support axes/keys required')
 require(numeric.dtype==gate.dtype==np.uint8 and counts.dtype==np.int64 and np.isin(numeric,[0,1,2,3]).all() and np.isin(gate,[0,1,2,3]).all(),'Invalid support dtype/code')
 all9=(numeric==0).all(axis=(0,1));gh=(gate==1).any(axis=0)
 e={'all9_numeric_status':numeric,'all3seed_gate_status':gate,'all3seed_gate_changed_cells':counts,'any_numeric_HOLD':(numeric==1).any(axis=(0,1)),'any_numeric_LOW':(numeric==2).any(axis=(0,1)),'any_numeric_EMPTY':(numeric==3).any(axis=(0,1)),'any_gate_HOLD':gh,'all9_numeric_stable':all9,'ENGINEERING_STABLE_FOR_THIS_TARGET_EXACT_GROUP':all9&~gh}
 require(all(np.array_equal(support[k],v) for k,v in e.items()),'ALL9 support differs from source cases');return e
def inspect_complete(plan,path,digest):
 """Only actual COMPLETE JSON metadata; no NPZ is opened here."""
 p=safe(path);require(p==UO/'COMPLETE.json' and sha(p)==digest,'Exact actual COMPLETE SHA required');v=read(p)
 require(v['status']=='COMPLETE_EXACT13_GROUP_CPU_ENGINEERING_SUPPORT_ROOT_REVIEW_PENDING' and v['all_input_source_output_hashes_reverified'] is True,'Upstream COMPLETE proof required')
 require(v['source']==plan['upstream_source'] and v['contract']==plan['upstream_contract'] and v['all9_target_group_units']==2454088 and v['records']==22086792,'Actual source/contract/family mismatch')
 require(len(v['case_channel_shards'])==336 and len(v['all9_support_shards'])==112,'All336+112 required');cases={};support={}
 for x in v['case_channel_shards']:
  k=(x['seed'],x['sample_id'],x['input_index'],x['channel']);require(k not in cases,'Duplicate case');i=plan['fixed_inputs'][x['input_index']]
  require(x['input_space']==i['input_space'] and x['input_feature']==i['feature_id'],'Input identity differs');require(safe(R/x['artifact']['relative_path'])==UO/f'seed{k[0]}_{k[1]}_input{k[2]:02d}_{k[3]}.npz','Case path differs');cases[k]=x
 require(set(cases)=={(s,a,i,c) for s in (0,1,2) for a in SAMPLES for i in range(28) for c in WIDTH},'Complete case identities differ')
 for x in v['all9_support_shards']:
  k=(x['sample_id'],x['input_index'],x['channel']);require(k not in support,'Duplicate support');require(safe(R/x['artifact']['relative_path'])==UO/f'ALL9_{k[0]}_input{k[1]:02d}_{k[2]}.npz','Support path differs')
  require(x['case_artifacts']==[cases[(s,*k)]['artifact'] for s in (0,1,2)],'Support seed provenance differs');support[k]=x
 require(set(support)=={(a,i,c) for a in SAMPLES for i in range(28) for c in WIDTH},'Support identities differ');records=[x['artifact'] for x in v['case_channel_shards']+v['all9_support_shards']]
 require(len({x['relative_path'] for x in records})==448,'448 unique physical artifacts required');return v,cases,support,records
class Guard:
 def __init__(self,limits):self.limits=limits;self.started=time.monotonic();self.peak=0
 def check(self):
  import psutil
  rss=psutil.Process().memory_info().rss;self.peak=max(self.peak,rss);require(rss<=self.limits['own_RSS_bytes'],'Own RSS ceiling');require(psutil.virtual_memory().available>=self.limits['system_available_bytes'],'Host memory floor');require(shutil.disk_usage(R).free>=self.limits['running_D_free_bytes'],'D free-space floor');require(time.monotonic()-self.started<=self.limits['wall_seconds'],'Cooperative wall budget')
def load_sources(cases,support,width,strong):
 import numpy as np
 means=[];nc=[];gc=[];cc=[]
 for row in cases:
  with np.load(R/row['artifact']['relative_path'],allow_pickle=False) as z:
   x=z['weighted_current_mean'];y=z['weighted_reference_mean'];require(x.dtype==y.dtype==np.float64 and x.shape==y.shape==(3,13,width),'Source mean shape/dtype differs')
   require(equal_nan(y[0],y[1]) and equal_nan(y[0],x[2]),'Duplicated h1e-4 means differ; no tolerance relaxation');means.append(np.stack([x[0],y[0],x[1],y[2]]));nc.append(z['numeric_status']);gc.append(z['gate_status']);cc.append(z['gate_changed_cells'])
 a=np.stack(means);nc=np.stack(nc);gc=np.stack(gc);cc=np.stack(cc);require(nc.shape==(3,3,13,width) and gc.shape==cc.shape==(3,13,width),'Source support target axes differ from means')
 with np.load(R/support['artifact']['relative_path'],allow_pickle=False) as z:sup={k:z[k] for k in z.files}
 validate_support(sup,nc,gc,cc);return {'unique_weighted_means':a,**derive(a,strong),**sup}
def publish_arrays(path,arrays):
 import numpy as np
 p=safe(path);tmp=p.with_name(p.name+'.partial');require(not p.exists() and not tmp.exists(),'No output overwrite')
 with tmp.open('xb') as f:np.savez_compressed(f,**arrays)
 os.link(tmp,p);tmp.unlink();return rec(p)
def top_table(path,arrays,axis,groups):
 import numpy as np
 p=safe(path);tmp=p.with_name(p.name+'.partial');flags=[*SUPPORT[3:],'mean_any_nonfinite','mean_any_exact_zero','mean_any_low_nonzero_at_existing_strong','mean_any_mixed_nonzero_signs','MEAN_SIGN_CONSISTENT_AT_EXISTING_STRONG_SCALE']
 fields=['group_index','group_kind','group_name','target_index','target_feature_id','target_stable_id_or_source_motif_id','raw_strict_reference_average_rank','raw_reference_median_abs','mean_direction_status',*[f'raw_mean_seed{s}_route{r}' for s in (0,1,2) for r in range(4)],'all9_numeric_codes_seed_then_comparison','all3_gate_codes','all3_gate_changed_cells',*flags];summary=[]
 with tmp.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
  for gi,g in enumerate(groups):
   score=arrays['reference_median_abs'][gi];chosen=boundary_top(score);summary.append({'group_index':gi,'finite_ranked_targets':int(np.isfinite(score).sum()),'unranked_nonfinite_targets':int((~np.isfinite(score)).sum()),'raw_top50_with_boundary_ties_rows':len(chosen),'HOLD_not_filtered':True})
   for ti in chosen:
    t=axis[int(ti)];row={'group_index':gi,'group_kind':g['kind'],'group_name':g['name'],'target_index':int(ti),'target_feature_id':t['feature_id'],'target_stable_id_or_source_motif_id':t.get('stable_id',t.get('source_motif_id','')),'raw_strict_reference_average_rank':float(arrays['reference_average_descending_rank'][gi,ti]),'raw_reference_median_abs':float(score[ti]),'mean_direction_status':int(arrays['mean_direction_status'][gi,ti]),'all9_numeric_codes_seed_then_comparison':','.join(map(str,arrays['all9_numeric_status'][:,:,gi,ti].ravel().tolist())),'all3_gate_codes':','.join(map(str,arrays['all3seed_gate_status'][:,gi,ti].tolist())),'all3_gate_changed_cells':','.join(map(str,arrays['all3seed_gate_changed_cells'][:,gi,ti].tolist()))}
    row.update({f'raw_mean_seed{s}_route{r}':float(arrays['unique_weighted_means'][s,r,gi,ti]) for s in (0,1,2) for r in range(4)});row.update({k:bool(arrays[k][gi,ti]) for k in flags});w.writerow(row)
 require(not p.exists(),'No output overwrite');os.link(tmp,p);tmp.unlink();return rec(p),summary
def execute(a):
 import numpy as np
 pp,bp,rp,out=map(safe,(a.plan,a.binding,a.review,a.output));require(sha(pp)==a.plan_sha and sha(bp)==a.binding_sha and sha(rp)==a.review_sha,'Exact plan/binding/review SHA required');plan=read(pp);binding=read(bp);review=read(rp)
 require(plan==metadata_plan() and binding['source']==rec(__file__) and binding['plan']==rec(pp),'Source/plan binding differs');require(review['decision']=='APPROVE_CPU_ALL28_AXIS_DESCRIPTIVE_READOUT' and review['source']==rec(__file__) and review['plan']==rec(pp) and review['binding']==rec(bp) and review['upstream_complete']==binding['upstream_complete'],'Root source/plan/actual-COMPLETE approval required')
 _,cases,support,records=inspect_complete(plan,R/binding['upstream_complete']['relative_path'],binding['upstream_complete']['sha256']);require(records==binding['source_numeric_artifacts'],'Physical source set changed');require(out.is_relative_to(R/'results') and not out.exists(),'New D results output required');require(shutil.disk_usage(R).free>=plan['resource_limits']['startup_D_free_bytes'],'Need8GiB D free');guard=Guard(plan['resource_limits']);guard.check();out.mkdir(parents=True)
 frozen=[rec(__file__),rec(pp),rec(bp),rec(rp),binding['upstream_complete'],plan['upstream_source'],plan['upstream_contract'],plan['selectors']];new(out/'STARTED.json',{'phase':'PREHASH_448_NO_NUMERIC_NPZ_ACCESS_YET','source':rec(__file__),'plan':rec(pp),'binding':rec(bp),'review':rec(rp),'scientific_pass':False});outputs=[];inventory=[];units=0
 try:
  for r in records:guard.check();check(r)
  new(out/'source_448_prehashed.json',{'records':records,'all336_case_and112_support_physical_hashes_verified':True,'original19GB_raw_rehashed':False})
  for sample in SAMPLES:
   for i,inp in enumerate(plan['fixed_inputs']):
    for channel,width in WIDTH.items():
     guard.check();key=(sample,i,channel);arrays=load_sources([cases[(s,*key)] for s in (0,1,2)],support[key],width,plan['mean_direction']['existing_strong']);axis=plan['target_axes'][channel]
     arrays.update(target_feature_id=np.asarray([x['feature_id'] for x in axis]),target_stable_id_or_source_motif_id=np.asarray([x.get('stable_id',x.get('source_motif_id','')) for x in axis]),group_index=np.arange(13),group_kind=np.asarray([g['kind'] for g in plan['group_order']]),group_name=np.asarray([g['name'] for g in plan['group_order']]))
     name=f'{sample}_input{i:02d}_{channel}';nr=publish_arrays(out/(name+'.npz'),arrays);tr,ss=top_table(out/(name+'_raw_top50_groups.tsv'),arrays,axis,plan['group_order']);outputs.extend((nr,tr));units+=13*width;inventory.append({'sample_id':sample,'input_index':i,'input_space':inp['input_space'],'input_feature':inp['feature_id'],'channel':channel,'npz':nr,'table':tr,'groups':ss});del arrays
  require(units==2454088 and len(inventory)==112 and len(outputs)==224,'Alltarget/group family incomplete')
  for r in records+frozen+outputs:guard.check();check(r)
  new(out/'group_table_inventory.json',{'entries':inventory,'all_target_group_units':units,'top50_never_filtered_by_engineering_or_direction':True,'all_original13_groups_separate':True})
  result={'schema':'dynavelo_cpu_all28_axis_descriptive_complete_v1','status':'COMPLETE_DESCRIPTIVE_SENSITIVITY_ROOT_REVIEW_PENDING','source':rec(__file__),'plan':rec(pp),'binding':rec(bp),'review':rec(rp),'upstream_complete':binding['upstream_complete'],'source_numeric_artifacts':records,'source448_physical_hashes_pre_and_post':True,'output_artifacts':outputs,'group_table_inventory':rec(out/'group_table_inventory.json'),'all_target_group_units':units,'descriptive_npz_count':112,'raw_top50_grouped_tsv_count':112,'original19GB_raw_rehashed':False,'seconds':time.monotonic()-guard.started,'own_peak_RSS_bytes':guard.peak,'original16HOLD_unchanged':True,'E1_unchanged':True,'A100_bridge':'NOT_TESTED_UNAVAILABLE','scientific_pass':False,'scope_limit':LIMIT};new(out/'COMPLETE.json',result);return {k:result[k] for k in ('status','all_target_group_units','descriptive_npz_count','seconds','scientific_pass')}
 except BaseException as e:new(out/'HOLD.json',{'status':'HOLD_DESCRIPTIVE_READOUT_INCOMPLETE','error_type':type(e).__name__,'error':str(e),'scientific_pass':False});raise
def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 for k in ('prepare','bind','execute'):g.add_argument('--'+k,action='store_true')
 for k in ('plan','plan-sha','complete','complete-sha','binding','binding-sha','review','review-sha','output'):p.add_argument('--'+k)
 a=p.parse_args()
 if a.prepare:
  require(not PREP.exists(),'Fresh prospective directory required');plan=metadata_plan();PREP.mkdir(parents=True);new(PREP/'plan.json',plan);v={'status':'SOURCE_PLAN_PREPARED_NO_ACTUAL_NPZ_OPENED','source':rec(__file__),'plan':rec(PREP/'plan.json')}
 elif a.bind:
  require(all((a.plan,a.plan_sha,a.complete,a.complete_sha,a.binding)),'Explicit plan/COMPLETE digests and new binding path required');pp=safe(a.plan);plan=read(pp);require(sha(pp)==a.plan_sha and plan==metadata_plan(),'Exact prospective source/plan required');_,_,_,records=inspect_complete(plan,a.complete,a.complete_sha);value={'schema':'dynavelo_cpu_all28_axis_actual_complete_binding_v1','status':'METADATA_BOUND_NOT_APPROVAL_NOT_EXECUTED','source':rec(__file__),'plan':rec(pp),'upstream_complete':rec(a.complete),'source_numeric_artifacts':records,'actual_NPZ_opened':False,'source448_hashes':'REQUIRED_AT_EXECUTION_PRE_AND_POST'};new(a.binding,value);v={'status':value['status'],'binding':rec(a.binding)}
 else:
  require(all((a.plan,a.plan_sha,a.binding,a.binding_sha,a.review,a.review_sha,a.output)),'Explicit source-bound root approval required');v=execute(a)
 print(json.dumps(v,allow_nan=False))
if __name__=='__main__':main()
