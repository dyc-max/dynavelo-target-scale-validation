"""Returned factual QC. Default inspect; no model/cloud/candidate execution."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,re,zipfile
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
ROOT=Path('D:/dyc_7/DynaVelo_Research') if os.name=='nt' else Path('/mnt/d/dyc_7/DynaVelo_Research')
CLOUD='/content/DynaVelo_Research/'
RUN_RE=re.compile(r'8489-MV-1_cuda_seed([012])_\d{8}T\d{12}Z\Z')
OLD_REL='repo/pipeline/audit_seed0_state_reconstruction.py';OLD_SHA='cfc81dc9bec959ca4fad4eabb0a11a176b6e93be4629171b77543ecc5d639b79'
INPUT_SHA={'rna':'245ce8a44dc5b1b9d1835e4f78ef604afb291bfc1841dff1a1777929b3a9a8fe','atac':'968aa6c255b3c45ba1ff6103219a822dd7319aa14e5d9a7a3c50fedc39c3aa29'}
SOURCE_SHA={'repo/colab/train_hspc_colab.py':'f611477ceb94776c0e3aa7982a55bfa42eb006abd3dd9abc3930a44999b01aa7','repo/colab/colab_training.py':'c70c59817b13b4553318ae3e24c59a9c56811bd941e6534c455b3fbfeec4b2c7','repo/colab/dynavelo_colab_adapter.py':'81f3f5c2e05391b8349ed65cd34401b2ceeda75761333502dcbf4b555c91667c'}
STATE_COUNTS={'CMP':275,'DC':175,'Erythrocyte':321,'GMP':594,'Granulocyte':259,'HSC':1307,'LMPP':700,'MEP':426,'Mast Cells':118,'Megakaryocyte':652,'Platelet':227,'Prog DC':402,'Prog MK':579}
FRAMES={'DC_target_1884':('HSC','Prog DC','DC'),'DC_context_3453':('HSC','LMPP','CMP','GMP','Prog DC','DC')}
PAIRS=[('HSC','Erythrocyte'),('HSC','Megakaryocyte'),('HSC','Granulocyte'),('HSC','DC'),('MEP','Erythrocyte'),('Prog MK','Megakaryocyte'),('GMP','Granulocyte'),('Prog DC','DC')]
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def disk(p):return '\\\\?\\'+str(p).replace('/','\\') if os.name=='nt' else str(p)
def sha(p):
 h=hashlib.sha256()
 with open(disk(p),'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with open(disk(p),'rb') as f:return json.loads(f.read().decode('utf-8-sig'))
def safe(root,rel):
 if not isinstance(rel,str) or not rel or PurePosixPath(rel).is_absolute() or '\\' in rel or ':' in rel or any(x in ('','.','..') for x in rel.split('/')):raise ValueError('Unsafe relative path')
 out=Path(root).joinpath(*PurePosixPath(rel).parts)
 if not out.resolve().is_relative_to(Path(root).resolve()) or out.is_symlink():raise ValueError('Redirected path')
 return out
def cloud(root,path):
 if not isinstance(path,str) or not path.startswith(CLOUD):raise ValueError('Exact cloud source path required')
 return safe(root,path[len(CLOUD):])
def record(p,root=ROOT):return {'relative_path':p.relative_to(root).as_posix(),'bytes':os.stat(disk(p)).st_size,'sha256':sha(p)}
def checked(root,rec):
 p=safe(root,rec['relative_path'])
 if sha(p)!=rec['sha256'] or ('bytes' in rec and os.stat(disk(p)).st_size!=rec['bytes']):raise ValueError('Artifact SHA/bytes differ: '+rec['relative_path'])
 return p
def save(p,x):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def old_core():
 p=ROOT/OLD_REL
 if sha(p)!=OLD_SHA:raise ValueError('Reviewed reconstruction core drift')
 spec=importlib.util.spec_from_file_location('_frozen_state_reconstruction_core',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def selections(labels):
 import numpy as np
 return [('all','all',np.ones(len(labels),bool))]+[('state',s,labels==s) for s in sorted(set(labels))]+[('frame',f,np.isin(labels,list(v))) for f,v in FRAMES.items()]
def aggregate(model,baseline,train,valid,labels,features,modality):
 rows=[]
 for kind,name,mask in selections(labels):
  selected=mask&valid;n=int(selected.sum());entries=n*features;ms=float(model[selected].sum());bs=float(baseline[selected].sum())
  rows.append({'modality':modality,'group_type':kind,'group':name,'n_total_cells':int(mask.sum()),'n_training_cells':int((mask&train).sum()),'n_validation_cells':n,'n_features':features,'n_entries':entries,'model_SSE':ms,'training_mean_baseline_SSE':bs,'model_MSE':ms/entries if entries else None,'training_mean_baseline_MSE':bs/entries if entries else None,'MSE_ratio_model_over_baseline':ms/bs if bs>0 else None})
 return rows
def validate_times(mean,sd):
 import numpy as np
 if mean.ndim!=1 or sd.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(sd).all() or not ((mean>0)&(mean<1)&(sd>0)&(sd*sd<mean*(1-mean))).all():raise ValueError('Invalid analytic Beta moments')
def validate_terminal(config,metrics,progress,run,seed):
 if not RUN_RE.fullmatch(run) or int(RUN_RE.fullmatch(run)[1])!=seed or config.get('run_id')!=run or metrics.get('run_id')!=run or config.get('training_seed')!=seed:raise ValueError('Run/seed mismatch')
 a=config['command_arguments']
 if any(config.get(k)!=v for k,v in {'n_cells':6035,'n_train':4828,'n_validation':1207,'n_genes':3006,'n_TF':365,'split_seed':0,'analytic_velocity':False}.items()):raise ValueError('Factual context dimensions differ')
 if any(a.get(k)!=v for k,v in {'seed':seed,'max_epochs':200,'threads':4,'n_samples':8,'evaluation_chunk_cells':64,'batch_size':64,'hidden':200,'zx_dim':50,'zy_dim':50,'learning_rate':.001,'resume_checkpoint':None,'resume_manifest':None,'training_only':False,'benchmark_only':False}.items()):raise ValueError('Not fresh original factual recipe')
 if config.get('status')!='QC_REVIEW_PENDING' or metrics.get('status')!='QC_REVIEW_PENDING' or metrics.get('best_checkpoint_reloaded') is not True or metrics.get('posterior_samples_per_cell')!=8 or metrics.get('evaluation_chunks')!=95:raise ValueError('Completed factual posterior required')
 n,best=metrics.get('epochs_completed'),metrics.get('best_validation_loss_epoch')
 if type(n) is not int or type(best) is not int or not 1<=best<=n<=200:raise ValueError('Invalid actual best epoch')
 chunks=progress.get('completed_chunks')
 if progress.get('complete') is not True or progress.get('total_chunks')!=95 or not isinstance(chunks,list) or len(chunks)!=95:raise ValueError('Incomplete posterior progress')
 for i,x in enumerate(chunks):
  if any(x.get(k)!=v for k,v in {'chunk':i+1,'start':i*64,'stop_exclusive':min((i+1)*64,6035),'n_cells':min(64,6035-i*64)}.items()):raise ValueError('Posterior cell coverage differs')
 return best
def validate_history(raw,metrics):
 import io,math
 rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t'))
 if not raw.endswith(b'\n') or len(rows)!=metrics['epochs_completed'] or not rows:raise ValueError('Incomplete epoch history')
 losses=[]
 for i,row in enumerate(rows,1):
  if None in row or None in row.values():raise ValueError('Torn history row')
  values={k:float(v) for k,v in row.items()}
  if values.get('Epoch')!=i or not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite/noncontiguous training history')
  losses.append(values['loss_test'])
 minimum=min(losses);last_equal_min=max(i+1 for i,v in enumerate(losses) if v==minimum)
 if metrics['best_validation_loss']!=minimum or metrics['best_validation_loss_epoch']!=last_equal_min:raise ValueError('Own-history best loss/epoch differs from frozen last-equal-min metric rule')
 return {'epochs':len(rows),'all_numeric_finite':True,'best_validation_loss':minimum,'best_epoch':last_equal_min,'tie_rule':'Last exact minimum, as unchanged trainer metrics; no new threshold'}
def finite_checkpoint(path):
 import torch
 state=torch.load(disk(path),map_location='cpu',weights_only=True).get('model_state_dict')
 if not isinstance(state,dict) or not state or any(not isinstance(v,torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Nonfinite/invalid own best model state')
 return {'all_state_tensors_finite':True,'state_tensor_count':len(state),'load':'Own hash-verified trained checkpoint; CPU weights_only=True; no model construction/forward'}
def validate_review(root,p):
 run=p.get('run_id','');match=RUN_RE.fullmatch(run)
 if p.get('schema')!='dynavelo_factual_readout_review_v1' or not match or type(p.get('seed')) is not int or p['seed']!=int(match[1]):raise ValueError('Review run/seed required')
 if p.get('script_sha256')!=sha(__file__) or p.get('old_compute_sha256')!=OLD_SHA or p.get('scientific_pass') is not False:raise ValueError('Reviewed code/scope differs')
 if not re.fullmatch('[A-Za-z0-9_-]{8,100}',p.get('output_id','')):raise ValueError('Unique output ID required')
 b=read(checked(root,p['binding']))
 if b.get('allowed_run_ids')!=[run] or b.get('training_seed')!=p['seed'] or b.get('scientific_pass') is not False or b.get('session')!=p.get('generation') or b.get('input_manifest_sha256')!='89aa75e78d34ea72b2d0411977f484ec40e7a580dfe938859fd14a09d1e4752a':raise ValueError('Observed factual run binding differs')
 return b
def return_records(root,review,binding):
 verify_path=checked(root,review['local_verification']);transfer_path=checked(root,review['transfer_manifest']);v=read(verify_path);t=read(transfer_path);run=review['run_id']
 if v.get('status')!='PASS_ALL_DECLARED_RETURN_GROUPS_BYTES_SHA' or v.get('promoted_runs')!=[run] or v.get('transfer_manifest_sha256')!=sha(transfer_path) or v.get('generation')!=review['generation'] or v.get('capture_id')!=t.get('capture_id'):raise ValueError('All-group verified/promoted return required')
 if t.get('generation')!=review['generation'] or t.get('run_ids')!=[run] or t.get('stability5_binding_sha256')!=review['binding']['sha256'] or {a['group'] for a in t['archives']}!={'recovery_and_reports','predictions'}:raise ValueError('Returned capture binding/groups differ')
 records={}
 for archive in t['archives']:
  path=transfer_path.parent/archive['name']
  if Path(archive['name']).name!=archive['name'] or sha(path)!=archive['sha256'] or os.stat(disk(path)).st_size!=archive['bytes']:raise ValueError('Verified ZIP bytes changed')
  with zipfile.ZipFile(disk(path)) as z:
   info=z.getinfo(archive['embedded_manifest'])
   if info.file_size>8*1024**2:raise ValueError('Oversized embedded manifest')
   manifest=json.loads(z.read(info));names=[x['relative_path'] for x in manifest['files']]
   if manifest.get('run_ids')!=[run] or manifest.get('generation')!=review['generation'] or manifest.get('capture_id')!=t['capture_id'] or len(names)!=len(set(n.casefold() for n in names)) or set(z.namelist())!=set(names)|{archive['embedded_manifest']}:raise ValueError('Archive exact census/identity differs')
   for rec in manifest['files']:
    rel=rec['source_relative_path'];safe(root,rel)
    if rel in records or rec.get('source_changed_during_copy') is not False:raise ValueError('Duplicate/mutable returned source')
    records[rel]={'relative_path':rel,'bytes':rec['bytes'],'sha256':rec['sha256']}
 return records

def execute(root,review_path,review_sha):
 import numpy as np,h5py
 root=Path(root);raw=Path(review_path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=review_sha:raise ValueError('Review SHA differs')
 review=json.loads(raw);binding=validate_review(root,review)
 if review.get('execution_authorized') is not True or review.get('reviewer')!='/root':raise PermissionError('Root approval after completed return required')
 records=return_records(root,review,binding);run=review['run_id'];base='checkpoints/GSE284047/'+run+'/';used=[]
 for name in ['binding','local_verification','transfer_manifest']:used.append(record(checked(root,review[name]),root))
 def use(rel):
  if rel not in records:raise ValueError('Artifact missing from verified capture: '+rel)
  p=checked(root,records[rel]);used.append(records[rel]);return p
 config=read(use(base+'config.json'));metrics=read(use(base+'metrics.json'));progress=read(use(base+'evaluation_progress.json'))
 best=validate_terminal(config,metrics,progress,run,review['seed'])
 history=validate_history(use(base+'training_metrics.tsv').read_bytes(),metrics)
 if config['prediction_dir']!=CLOUD+'data/processed/GSE284047/8489-MV-1/model_runs/'+run:raise ValueError('Prediction directory differs')
 actual_inputs={CLOUD+'data/processed/colab_minimal/HSPC/adata_'+k+'.h5ad':v for k,v in INPUT_SHA.items()}
 if config['input_files']!=actual_inputs or 'training_control' in config.get('input_assembly',{}):raise ValueError('Not original factual inputs')
 for rel,expected in SOURCE_SHA.items():
  if sha(safe(root,rel))!=expected:raise ValueError('Frozen trainer/source drift')
 if config['script_sha256']!=SOURCE_SHA['repo/colab/train_hspc_colab.py'] or config['training_implementation_sha256']!=SOURCE_SHA['repo/colab/colab_training.py'] or config['adapter_manifest']['adapter_sha256']!=SOURCE_SHA['repo/colab/dynavelo_colab_adapter.py']:raise ValueError('Run source identities differ')
 if metrics.get('external_validation')!='NOT_PERFORMED' or metrics.get('perturbation')!='NOT_PERFORMED':raise ValueError('Expected factual training-only scope')
 ckpt=cloud(root,metrics['best_checkpoint']);use(ckpt.relative_to(root).as_posix())
 if sha(ckpt)!=metrics['best_checkpoint_sha256']:raise ValueError('Best reload artifact differs')
 checkpoint_check=finite_checkpoint(ckpt)
 statuses=[rel for rel in records if rel.startswith(base+'fresh_job_return_evidence/') and rel.endswith('/status.json')];good=[]
 for rel in statuses:
  s=read(use(rel));prefix=rel[:-len('status.json')]
  if s.get('status')!='PROCESS_EXITED_0_SCIENTIFIC_QC_PENDING' or s.get('exit_code')!=0 or s.get('run_id')!=run or s.get('plan_sha256')!=binding['plan_sha256'] or s.get('binding_sha256')!=binding['run_binding_sha256'] or s.get('error') or s.get('stop_reason'):continue
  for name,key in [('run_binding.json','run_binding_sha256'),('claim.json','claim_sha256'),('initial_config.json','initial_config_sha256')]:
   if sha(use(prefix+name))!=binding[key]:raise ValueError('Returned job evidence differs')
  b=read(safe(root,prefix+'run_binding.json'));c=read(safe(root,prefix+'claim.json'))
  if s.get('child_pid')!=b.get('child_pid') or s.get('nonce')!=c.get('nonce') or b.get('nonce')!=c.get('nonce'):raise ValueError('Terminal PID/nonce differs')
  good.append(rel)
 if not good:raise ValueError('No completed fresh-job evidence')
 console=use(base+'console.log').read_text(encoding='utf-8')
 if [int(x) for x in re.findall(r'^n:\s+(\d+)\s*$',console,re.M)]!=list(range(8))*95:raise ValueError('Actual eight-draw console incomplete')
 core=old_core();all_rows=[];means={};per_cell={};common=None;times=None;input_records=[];time_axes={}
 for modality,key,layer,nvar in [('RNA','rna','x_pred_mean',3006),('motif','atac','y_pred_mean',365)]:
  inp=cloud(root,config['command_arguments'][key]);input_record=record(inp,root)
  if input_record['sha256']!=INPUT_SHA[key]:raise ValueError('Original input changed')
  input_records.append(input_record);pred=use('data/processed/GSE284047/8489-MV-1/model_runs/'+run+'/adata_'+key+'_pred.h5ad')
  if metrics['prediction_files'].get(CLOUD+pred.relative_to(root).as_posix())!=sha(pred) or len(metrics['prediction_files'])!=2:raise ValueError('Prediction hash differs from metrics')
  with h5py.File(inp,'r') as f,h5py.File(pred,'r') as g:result=modality_readout(f,g,config,metrics,key,nvar,core)
  obs,mean,model,baseline,t,sd,axes=result
  if common is None:common=obs[['celltype','split']].copy();times=(t,sd)
  elif not common.equals(obs[['celltype','split']]) or not all(np.array_equal(x,y) for x,y in zip(times,(t,sd))):raise ValueError('Cross-modality metadata/time differs')
  labels=obs['celltype'].astype(str).to_numpy();split=obs['split'].astype(str).to_numpy();train=split=='train';valid=split=='validation'
  means[modality]=mean;per_cell[modality]=(model,baseline);time_axes[key]=axes
  rows=aggregate(model,baseline,train,valid,labels,nvar,modality);all_rows+=rows;reference=metrics['RNA_reconstruction' if key=='rna' else 'motif_reconstruction']
  for name in ('model_MSE','training_mean_baseline_MSE','MSE_ratio_model_over_baseline'):
   if not np.isclose(rows[0][name],reference[name],atol=1e-10,rtol=1e-8):raise ValueError('Global metric agreement failed')
 return finish(root,review_path,review_sha,review,best,all_rows,means,per_cell,common,times,used,input_records,time_axes,core,history,checkpoint_check)

def modality_readout(f,g,config,metrics,key,nvar,core):
 import numpy as np,pandas as pd
 from anndata.experimental import read_elem
 obs,var=read_elem(f['obs']),read_elem(f['var']);po,pv=read_elem(g['obs']),read_elem(g['var'])
 if not obs.index.is_unique or not var.index.is_unique or not obs.index.equals(po.index) or not var.index.equals(pv.index) or len(obs)!=6035 or len(var)!=nvar:raise ValueError('Exact complete paired axes required')
 pd.testing.assert_frame_equal(var,pv,check_exact=True)
 pd.testing.assert_frame_equal(obs.drop(columns=['latent_time']),po.loc[:,[c for c in obs if c!='latent_time']],check_exact=True)
 if not np.array_equal(obs['latent_time'].to_numpy(),po['latent_time_scvelo'].to_numpy(),equal_nan=True):raise ValueError('Retained annotation changed')
 split=obs['split'].astype(str).to_numpy();labels=obs['celltype'].astype(str).to_numpy();train=split=='train';valid=split=='validation'
 if int(train.sum())!=4828 or int(valid.sum())!=1207 or obs['celltype'].value_counts().to_dict()!=STATE_COUNTS:raise ValueError('Fixed split/state population differs')
 expected=set(np.asarray(sorted(obs.index))[np.random.default_rng(0).permutation(6035)[:4828]])
 if set(obs.index[train])!=expected:raise ValueError('Fixed seed0 split membership differs')
 if json.loads(read_elem(g['uns/dynavelo_training_config_json']))!=config:raise ValueError('Embedded config differs')
 embedded=json.loads(read_elem(g['uns/dynavelo_internal_qc_json']))
 if embedded!={k:v for k,v in metrics.items() if k not in ('prediction_files','completed_utc')}:raise ValueError('Embedded metrics differ')
 if len(g['obsp']) or 'neighbors' in g['uns']:raise ValueError('Unexpected inherited graph')
 prefix='x' if key=='rna' else 'y';required={prefix+'_pred_mean',prefix+'_pred_std','v'+prefix+'_pred_mean','v'+prefix+'_pred_std'}
 if set(g['layers'])!=required|({'velocity'} if key=='rna' else set()):raise ValueError('Unexpected posterior layer family')
 for name in required:
  ds=g['layers/'+name]
  if ds.shape!=(6035,nvar) or ds.dtype!=np.float32:raise ValueError('Posterior layer geometry/storage differs')
  for i in range(0,6035,64):
   block=ds[i:i+64]
   if not np.isfinite(block).all() or ((name.endswith('_std') or name=='x_pred_mean') and (block<0).any()):raise ValueError('Invalid posterior values')
 if key=='rna':
  for i in range(0,6035,64):
   if not np.array_equal(f['layers/velocity'][i:i+64],g['layers/velocity'][i:i+64],equal_nan=True):raise ValueError('Velocity prior/mask changed')
 layer=prefix+'_pred_mean';mean,model,baseline=core.compute(f['X'],g['X'],g['layers/'+layer],train,valid,nvar)
 t=np.asarray(po['latent_time_mean'],dtype=float);sd=np.asarray(po['latent_time_std'],dtype=float);validate_times(t,sd)
 axes={'cells_sha256':digest(list(obs.index.astype(str))),'features_sha256':digest(list(var.index.astype(str)))}
 return obs,mean,model,baseline,t,sd,axes

def write_table(path,rows):
 if not rows:raise ValueError('No output rows')
 with path.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def finish(root,review_path,review_sha,review,best,rows,means,per_cell,obs,times,used,inputs,axes,core,history,checkpoint_check):
 import numpy as np
 for rec in used+inputs:checked(root,rec)
 if sha(review_path)!=review_sha or sha(__file__)!=review['script_sha256'] or sha(ROOT/OLD_REL)!=OLD_SHA:raise ValueError('Review/code changed during audit')
 for modality in ('RNA','motif'):
  counts={r['group']:r['n_total_cells'] for r in rows if r['modality']==modality and r['group_type']=='frame'}
  if counts!={'DC_target_1884':1884,'DC_context_3453':3453}:raise ValueError('Frozen frame populations differ')
 out=safe(root,'results/reports/factual_readouts/'+review['output_id']);out.mkdir(parents=True,exist_ok=False)
 write_table(out/'state_and_frame_MSE.tsv',rows);np.savez(out/'full_training_feature_means.npz',**means)
 t,sd=times;time_rows=[{'cell_id':str(obs.index[i]),'author_state':str(obs['celltype'].iloc[i]),'split':str(obs['split'].iloc[i]),'raw_beta_mean':float(t[i]),'analytic_beta_sd':float(sd[i])} for i in range(len(obs))]
 write_table(out/'cell_raw_beta_time.tsv',time_rows)
 valid=obs['split'].astype(str).to_numpy()=='validation';cell=[]
 for i in np.flatnonzero(valid):cell.append({'cell_id':str(obs.index[i]),'author_state':str(obs['celltype'].iloc[i]),'RNA_model_SSE':float(per_cell['RNA'][0][i]),'RNA_baseline_SSE':float(per_cell['RNA'][1][i]),'motif_model_SSE':float(per_cell['motif'][0][i]),'motif_baseline_SSE':float(per_cell['motif'][1][i])})
 write_table(out/'heldout_cell_SSE.tsv',cell)
 report={'schema':'dynavelo_factual_readout_v1','status':'DESCRIPTIVE_INTERNAL_READOUT_NO_SCIENTIFIC_PASS','run_id':review['run_id'],'seed':review['seed'],'best_epoch':best,'review_sha256':review_sha,'binding':review['binding'],'n_cells':6035,'n_train':4828,'n_validation':1207,'frames':FRAMES,'rows':rows,'axes':axes,'baseline':'ALL 4828 original training cells; unchanged across all states/frames','beta_semantics':'Analytic Beta mean/SD, not empirical eight-time-draw mean/SD, SE, CI or elapsed time. No time-identifiability claim from ordinal rank expansion.','source_records':used+inputs,'artifacts':[record(p,root) for p in sorted(out.iterdir())],'script_sha256':sha(__file__),'old_compute_sha256':OLD_SHA,'peak_working_set_bytes':core.rss(),'P_values_generated':False,'scientific_pass':False,'model_execution':False,'created_utc':datetime.now(timezone.utc).isoformat()}
 report.update(training_history=history,own_best_checkpoint=checkpoint_check)
 save(out/'audit.json',report);save(out/'COMPLETE.json',{'audit':record(out/'audit.json',root),'scientific_pass':False});return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path);p.add_argument('--review-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps({'status':'INSPECT_ONLY_NO_RESULTS_READ','requires':'Root-reviewed actual binding + all-groups verified/promoted completed return','dimensions':[6035,4828,1207,3006,365],'frames':FRAMES,'scientific_pass':False}));return
 if a.review is None or not a.review_sha256:raise ValueError('Explicit reviewed artifact/SHA required')
 path=a.review.absolute()
 if not path.is_relative_to(ROOT):raise ValueError('Review outside project')
 result=execute(ROOT,path,a.review_sha256);print(json.dumps({'status':result['status'],'run_id':result['run_id'],'best_epoch':result['best_epoch']}))
if __name__=='__main__':main()
