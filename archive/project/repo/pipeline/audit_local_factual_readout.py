"""Local completed-run evidence + unchanged factual QC. Inspect is the default.

No training, cloud calls, automatic latest-run selection, or scientific PASS.
Capture hashes prediction files only AFTER verified owned-process termination.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, os, re
from datetime import datetime, timezone
from pathlib import Path
import audit_factual_run_readout as q

ROOT=q.ROOT
SELF=Path(__file__).resolve()
ENTRY='repo/pipeline/train_hspc_local_resume.py'
ENTRY_SHA='e6b810b1da4448135646a90db3d9e4dffeafb5382c43e6648e9e9893a56e71a5'
SUPERVISOR='repo/pipeline/supervise_local_resume.py'
SUPERVISOR_SHA='324e6c88ab77548b117757099d47e145452df56d3fab1cecac373f716a8b7b14'
PINS={ENTRY:ENTRY_SHA,SUPERVISOR:SUPERVISOR_SHA,
 'repo/pipeline/audit_factual_run_readout.py':'0ad979f3bfddd2aae263f245c3f04ae1782d01dc93b16db42bea2d45e24fa738',
 'repo/pipeline/compare_factual_raw_times.py':'d034f88927c203891bfd014ed436deb6886435f35c528c2d57708ffebf5775c0',
 q.OLD_REL:q.OLD_SHA}
RUN=re.compile(r'8489-MV-1_(cpu|xpu)_seed1_\d{8}T\d{12}Z\Z')
SCOPE='DESCRIPTIVE_INTERNAL_READOUT_NO_SCIENTIFIC_PASS'
MIGRATION_NOTE=('Seed1 resumes its own CUDA parent on a local backend/version. '
 'Comparison with seed0 combines initialization and backend/version/RNG-stream migration; '
 'it is not a pure initialization experiment. XPU may use framework CPU operator fallback; '
 'no pure-GPU or bitwise-continuation claim. Cells/seeds are not donor replicates.')

def require(ok,message):
 if not ok:raise ValueError(message)

def local(root,value):
 p=Path(value)
 require(p.is_absolute() and p.resolve().is_relative_to(Path(root).resolve()) and '..' not in p.parts,'Absolute project path required')
 require(not p.is_symlink(),'Symlink not accepted')
 return p

def raw(path):
 with open(q.disk(path),'rb') as f:return f.read()

def pinned(path,digest):
 b=raw(path);require(hashlib.sha256(b).hexdigest()==digest,'Trusted raw SHA differs: '+str(path))
 return json.loads(b.decode('utf-8-sig'))

def constants(root):
 """Read literal configuration only; never import the live training entry."""
 require(q.sha(q.safe(root,ENTRY))==ENTRY_SHA,'Frozen local entry drift')
 tree=ast.parse(raw(q.safe(root,ENTRY)).decode('utf-8-sig'));out={}
 for node in tree.body:
  if isinstance(node,ast.Assign):
   for t in node.targets:
    if isinstance(t,ast.Name) and t.id in ('FROZEN','FIT','ARCH','INPUTS'):out[t.id]=ast.literal_eval(node.value)
 require(set(out)=={'FROZEN','FIT','ARCH','INPUTS'},'Entry constants changed')
 return out

def source_records(root,supervisor_sha=SUPERVISOR_SHA):
 require(re.fullmatch('[a-f0-9]{64}',supervisor_sha or '') is not None,'Explicit reviewed supervisor SHA required')
 pins={**PINS,**constants(root)['FROZEN'],SUPERVISOR:supervisor_sha};records=[]
 for rel,h in pins.items():
  p=q.safe(root,rel);require(q.sha(p)==h,'Reviewed source drift: '+rel);records.append(q.record(p,root))
 return records

def command_options(command):
 require(isinstance(command,list) and len(command)>3 and command[1]=='-B','Actual launcher command required')
 options={};i=3
 while i<len(command):
  k=command[i];require(k.startswith('--') and k not in options,'Duplicate/unrecognized command token')
  if k=='--execute':options[k]=True;i+=1
  else:
   require(i+1<len(command) and not command[i+1].startswith('--'),'Missing command value')
   options[k]=command[i+1];i+=2
 require(set(options).issubset({'--manifest','--manifest-sha256','--device','--action','--threads','--execute','--device-evidence','--device-evidence-sha256'}),'Unexpected local command option')
 return options

def final_summary(console):
 decoder=json.JSONDecoder();found=[]
 for m in re.finditer(r'^\{',console,re.M):
  try:value,end=decoder.raw_decode(console[m.start():])
  except json.JSONDecodeError:continue
  if isinstance(value,dict) and set(value)=={'run_id','run_dir','action','parent_completed_epoch'}:
   found.append((value,m.start()+end))
 require(len(found)==1 and not console[found[0][1]:].strip(),'Unique final entry summary must end owned console')
 return found[0][0]

def validate_owned(root,job_id,run,job_sha,terminal_sha,supervisor_sha=SUPERVISOR_SHA):
 require(RUN.fullmatch(run) is not None,'Exact local seed1 run required')
 require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',job_id) is not None,'Invalid job ID')
 d=q.safe(root,'tmp_cache/local_owned_jobs/'+job_id)
 job=pinned(d/'job.json',job_sha);terminal=pinned(d/'terminal.json',terminal_sha);launcher=q.read(d/'launcher.json')
 require(job.get('job_id')==terminal.get('job_id')==job_id and local(root,job['job_dir'])==d,'Wrong job/root binding')
 require(job.get('entry_sha256')==ENTRY_SHA and job.get('supervisor_sha256')==supervisor_sha,'Job source differs')
 require(re.fullmatch('[a-f0-9]{32}',job.get('nonce','')) is not None,'Missing owned-job nonce')
 require(terminal.get('status')=='PROCESS_TERMINATED_NO_SCIENTIFIC_PASS' and type(terminal.get('exit_code')) is int and terminal['exit_code']==0 and terminal.get('errors')==[] and terminal.get('stop_reason')==[] and terminal.get('owned_job_confirmed_empty_before_close') is True,'HOLD: successful empty owned terminal required')
 require(launcher.get('kernel_job_assigned_before_execution') is True,'Owned Job Object evidence absent')
 require(bool(terminal.get('finished_utc')) and terminal.get('checkpoints_and_logs_deleted') is False,'Terminal artifact preservation evidence missing')
 cmd=job['command'];options=command_options(cmd)
 require(local(root,cmd[2])==q.safe(root,ENTRY) and options.get('--action')=='train' and options.get('--execute') is True and options.get('--threads')=='4' and options.get('--device')==RUN.fullmatch(run)[1],'Wrong entry/action/device/threads')
 require(options.get('--manifest-sha256')==job.get('manifest_sha256'),'Parent manifest binding differs')
 interpreter=local(root,cmd[0]);require(interpreter.name.lower()=='python.exe' and q.sha(interpreter)==job['python_sha256'],'Actual interpreter differs')
 seen=terminal.get('observed_processes',[])
 require(isinstance(seen,list) and any(p.get('pid')==launcher.get('pid') and p.get('create_time')==launcher.get('create_time') for p in seen),'Launcher identity absent from owned membership')
 require(any(p.get('command',[])[1:]==cmd[1:] and type(p.get('pid')) is int and type(p.get('create_time')) in (int,float) for p in seen),'Actual training command absent from owned members')
 summary=final_summary(raw(d/'console.log').decode('utf-8-sig'))
 expected_dir=q.safe(root,'checkpoints/GSE284047/'+run)
 require(summary.get('run_id')==run and local(root,summary['run_dir'])==expected_dir and summary.get('action')=='train','Exclusive console run summary mismatch')
 require(not (expected_dir/'failure.json').exists() and not (expected_dir/'short_test_completion.json').exists(),'Failed/short-test run cannot become completed factual evidence')
 records=[q.record(d/name,root) for name in ('job.json','terminal.json','launcher.json','status.json','console.log')]
 records.append(q.record(interpreter,root))
 return job,options,summary,records

def validate_local_terminal(config,metrics,progress,run,device):
 require(config.get('run_id')==metrics.get('run_id')==run and config.get('training_seed')==1,'Local run/seed mismatch')
 require(all(config.get(k)==v for k,v in {'n_cells':6035,'n_train':4828,'n_validation':1207,'n_genes':3006,'n_TF':365,'split_seed':0,'analytic_velocity':False}.items()),'Factual context dimensions differ')
 a=config['command_arguments'];recipe={'seed':1,'max_epochs':200,'threads':4,'n_samples':8,'evaluation_chunk_cells':64,'batch_size':64,'hidden':200,'zx_dim':50,'zy_dim':50,'learning_rate':.001,'training_only':False,'benchmark_only':False,'local_action':'train'}
 require(all(a.get(k)==v for k,v in recipe.items()),'Fixed original factual recipe differs')
 require(config.get('status')==metrics.get('status')=='QC_REVIEW_PENDING' and metrics.get('best_checkpoint_reloaded') is True and metrics.get('posterior_samples_per_cell')==8 and metrics.get('evaluation_chunks')==95,'HOLD: completed factual posterior required')
 n,b=metrics.get('epochs_completed'),metrics.get('best_validation_loss_epoch')
 require(type(n) is int and type(b) is int and 1<=b<=n<=200,'Invalid epoch/best')
 chunks=progress.get('completed_chunks')
 require(progress.get('complete') is True and progress.get('total_chunks')==95 and isinstance(chunks,list) and len(chunks)==95,'HOLD: posterior incomplete')
 for i,x in enumerate(chunks):
  require(all(x.get(k)==v for k,v in {'chunk':i+1,'start':i*64,'stop_exclusive':min((i+1)*64,6035),'n_cells':min(64,6035-i*64)}.items()),'Incomplete exact posterior cell coverage')
 require(config.get('local_runtime',{}).get('device')==device and config['local_runtime'].get('bitwise_continuation_claim') is False,'Backend migration disclosure absent')
 require(metrics.get('external_validation')==metrics.get('perturbation')=='NOT_PERFORMED','Not factual scope')
 return b

def ancestor_records(root,migration):
 """Verify direct and earlier parent bytes; interrupted ancestors need not be posterior-complete."""
 rows=[];seen=set();m=migration
 while True:
  path=local(root,m['parent_manifest_path']);h=m['parent_manifest_sha256']
  require(str(path) not in seen and len(seen)<32,'Cyclic/excessive migration ancestry');seen.add(str(path))
  manifest=pinned(path,h);rows.append(q.record(path,root))
  require(manifest.get('run_id')==m['parent_run_id'] and manifest.get('completed_epoch')==m['parent_completed_epoch'],'Ancestor epoch/run mismatch')
  roles={r['role']:r for r in manifest['required_files']}
  require(len(manifest['required_files'])==3 and set(roles)==set(m['parent_files'])=={'checkpoint','prior_config','best_snapshot'},'Ancestor exact role family differs')
  for role,r in m['parent_files'].items():
   p=local(root,r['path']);expected=roles[role]
   require(p==q.safe(root,expected['relative_path']) and p.parent==path.parent and r['sha256']==expected['sha256'] and r['bytes']==expected['bytes'],'Ancestor role mapping differs')
   actual=q.record(p,root);require(actual['sha256']==r['sha256'] and actual['bytes']==r['bytes'],'Ancestor immutable bytes differ');rows.append(actual)
  prior=q.read(local(root,m['parent_files']['prior_config']['path']))
  require(prior.get('run_id')==manifest['run_id'],'Ancestor prior config run differs')
  if 'local_migration' not in prior:break
  embedded=dict(prior['local_migration']);mh=embedded.pop('migration_manifest_sha256');mp=path.parent/'migration_manifest.json'
  require(pinned(mp,mh)==embedded and embedded.get('new_run_id')==manifest['run_id'],'Ancestor raw/embedded migration differs');rows.append(q.record(mp,root))
  m=embedded
 return rows

def build_capture(root,job_id,run,job_sha,terminal_sha,supervisor_sha=SUPERVISOR_SHA):
 """Only call after root obtains real terminal SHA; does not decode H5AD."""
 sources=source_records(root,supervisor_sha);fixed=constants(root)
 job,options,summary,records=validate_owned(root,job_id,run,job_sha,terminal_sha,supervisor_sha)
 base=q.safe(root,'checkpoints/GSE284047/'+run);pred=q.safe(root,'data/processed/GSE284047/8489-MV-1/model_runs/'+run)
 def use(path):
  path=local(root,path);rec=q.record(path,root);records.append(rec);return q.read(path)
 config=use(base/'config.json');metrics=use(base/'metrics.json');progress=use(base/'evaluation_progress.json')
 best=validate_local_terminal(config,metrics,progress,run,options['--device'])
 migration=use(base/'migration_manifest.json');mh=q.sha(base/'migration_manifest.json')
 require(config.get('local_migration')=={**migration,'migration_manifest_sha256':mh},'Config/raw migration mismatch')
 require(migration.get('schema')=='dynavelo_local_backend_migration_v1' and migration.get('new_run_id')==run and migration.get('action')=='train','Wrong migration')
 records+=ancestor_records(root,migration)
 require(migration.get('parent_manifest_sha256')==job['manifest_sha256'] and migration.get('parent_completed_epoch')==summary['parent_completed_epoch'] and metrics['epochs_completed']>=summary['parent_completed_epoch'],'Parent epoch/manifest mismatch')
 execution=migration['actual_execution']
 require(local(root,execution['root'])==Path(root) and execution.get('device')==options['--device'] and execution.get('threads')==4 and execution.get('source_sha256')==ENTRY_SHA,'Actual execution identity differs')
 require(migration.get('scientific_fit_fixed')==fixed['FIT'] and migration.get('total_epoch_ceiling')==200 and migration.get('early_stop_patience')==10 and migration.get('CUDA_rng_applied_or_converted') is False and migration.get('bitwise_continuation_claim') is False,'Migration numerical scope differs')
 parent=local(root,options['--manifest']);require(local(root,migration['parent_manifest_path'])==parent,'Parent path mismatch')
 pm=pinned(parent,job['manifest_sha256']);records.append(q.record(parent,root))
 require(q.sha(base/'parent_restore_manifest.raw.json')==job['manifest_sha256'] and pm['run_id']==migration['parent_run_id'] and pm['completed_epoch']==migration['parent_completed_epoch'],'Raw parent manifest/epoch differs')
 require(set(migration['parent_files'])=={'checkpoint','prior_config','best_snapshot'},'Parent role census differs')
 parent_roles={r['role']:r for r in pm['required_files']};require(set(parent_roles)==set(migration['parent_files']) and len(pm['required_files'])==3,'Parent manifest role census differs')
 for role,r in migration['parent_files'].items():
  p=local(root,r['path']);expected=parent_roles[role]
  require(p==q.safe(root,expected['relative_path']) and r['sha256']==expected['sha256'] and r['bytes']==expected['bytes'],'Parent exact role differs')
  records.append(q.record(p,root));require(records[-1]['sha256']==r['sha256'] and records[-1]['bytes']==r['bytes'],'Immutable parent bytes drift')
 prior=use(base/'parent_resume_config.raw.json')
 require(q.sha(base/'parent_resume_config.raw.json')==migration['parent_files']['prior_config']['sha256']==config.get('resume_parent_config_sha256'),'Raw parent config mismatch')
 immutable={'inputs':prior['input_files'],'adapter':prior['adapter_manifest'],'runner_sha256':prior['script_sha256'],'training_helper_sha256':prior['training_implementation_sha256'],'architecture':prior['model_defaults'],'fit_arguments':{k:prior['command_arguments'][k] for k in fixed['FIT']}}
 require(migration['historical_immutable_config']==immutable and config['model_defaults']==prior['model_defaults']==fixed['ARCH'],'Historical/actual numerical configuration differs')
 require(config['script_sha256']==config['training_implementation_sha256']==ENTRY_SHA,'Local source falsely labeled')
 adaptation=use(base/'source_adaptation.json');adapter=config['adapter_manifest']
 require(adaptation.get('actual_entry_sha256')==ENTRY_SHA and adaptation.get('frozen_sources')==fixed['FROZEN'] and adaptation.get('helper_sha256')==config.get('compiled_helper_sha256') and adaptation.get('runner_sha256')==config.get('compiled_runner_sha256'),'Compiled adaptation identities differ')
 require(adapter.get('adapter_sha256')==adapter.get('local_resume_source_sha256')==ENTRY_SHA and adapter.get('base_cpu_adapter_sha256')==fixed['FROZEN']['repo/pipeline/dynavelo_cpu_adapter.py'] and adapter.get('device')==options['--device'] and adapter.get('analytic_velocity') is False,'Actual adapter identity differs')
 for key,rel in [('upstream_models_sha256','repo/DynaVelo/dynavelo/models.py'),('upstream_utils_sha256','repo/DynaVelo/dynavelo/utils.py')]:require(adapter.get(key)==fixed['FROZEN'][rel],'Upstream identity differs')
 require(all(execution['packages'].get(k)==v for k,v in config['package_versions'].items()) and config['local_runtime']['torch_version']==execution['packages']['torch'],'Actual package versions differ')
 expected_inputs={str(q.safe(root,rel)):q.INPUT_SHA[key] for key,rel in zip(('rna','atac'),fixed['INPUTS'])}
 require(config['input_files']==expected_inputs and set(migration['actual_inputs'])==set(fixed['INPUTS']) and 'training_control' not in config.get('input_assembly',{}),'Original factual inputs required')
 for key,rel in zip(('rna','atac'),fixed['INPUTS']):
  p=q.safe(root,rel);r=migration['actual_inputs'][rel]
  require(local(root,r['local_path'])==p and r['sha256']==q.INPUT_SHA[key] and prior['input_files'].get(r['historical_path'])==r['sha256'] and local(root,config['command_arguments'][key])==p,'Input migration mapping differs')
  rec=q.record(p,root);require(rec['sha256']==r['sha256'] and rec['bytes']==r['bytes'],'Actual input drift');records.append(rec)
 require(local(root,config['prediction_dir'])==pred and local(root,config['checkpoint_dir'])==base and local(root,config['resume_parent_config'])==base/'parent_resume_config.raw.json','Local run paths differ')
 for k,name in [('resume_checkpoint','last_checkpoint.pth'),('resume_manifest','checkpoint_restore_manifest.json')]:require(local(root,config['command_arguments'][k])==base/name,'Local resume path differs')
 require(local(root,metrics['best_checkpoint'])==base/'model_best.pth' and config['local_model_output_paths']['canonical_best_checkpoint']==metrics['best_checkpoint'],'Short canonical own-best differs')
 restore=use(base/'checkpoint_restore_manifest.json')
 require(restore.get('schema')=='dynavelo_colab_checkpoint_restore_v1' and restore.get('run_id')==run and local(root,restore['cloud_root'])==Path(root) and restore.get('completed_epoch')==metrics['epochs_completed'],'Final local restore identity differs')
 roles={r['role']:r for r in restore['required_files']};require(len(restore['required_files'])==3 and set(roles)=={'checkpoint','prior_config','best_snapshot'},'Final restore role census differs')
 names={'checkpoint':{'last_checkpoint.pth'},'prior_config':{'resume_config.json'},'best_snapshot':{'best_snapshot_0.pth','best_snapshot_1.pth'}}
 for role,r in roles.items():
  p=q.checked(root,r);require(p.parent==base and p.name in names[role],'Final role outside run');records.append(q.record(p,root))
 require(q.sha(base/'model_best.pth')==metrics['best_checkpoint_sha256']==roles['best_snapshot']['sha256'],'Reloaded canonical own-best differs from committed snapshot')
 require(set(metrics['prediction_files'])=={str(pred/('adata_'+k+'_pred.h5ad')) for k in ('rna','atac')},'Exact two local prediction paths required')
 for path,h in metrics['prediction_files'].items():
  p=local(root,path);rec=q.record(p,root);require(rec['sha256']==h,'Prediction SHA differs');records.append(rec)
 gate=use(base/'training_gate.json');require(all(metrics.get(k)==v for k,v in gate.items()),'Training gate/final metric mismatch')
 if migration.get('device_evidence'):
  e=migration['device_evidence'];p=local(root,e['path']);require(options.get('--device-evidence')==str(p) and options.get('--device-evidence-sha256')==e['sha256'],'Device qualification command differs');short=pinned(p,e['sha256']);records.append(q.record(p,root))
  require(short.get('status')=='SHORT_TEST_ONLY_NOT_RESUMABLE' and short.get('local_source_sha256')==ENTRY_SHA and short.get('parent_manifest_sha256')==job['manifest_sha256'] and short.get('device')==options['--device'] and short.get('torch_version')==execution['packages']['torch'],'Device qualification differs')
 elif options['--device']=='xpu':raise ValueError('XPU device qualification missing')
 for name in ('parent_restore_manifest.raw.json','model_best.pth','training_metrics.tsv','validation_velocity_cosines.tsv','console.log'):records.append(q.record(base/name,root))
 require([int(x) for x in re.findall(r'^n:\s+(\d+)\s*$',raw(base/'console.log').decode('utf-8-sig'),re.M)]==list(range(8))*95,'Eight-draw posterior console incomplete')
 unique={r['relative_path']:r for r in records+sources}
 for r in records+sources:require(unique[r['relative_path']]==r,'Source changed during capture');q.checked(root,r)
 return {'schema':'dynavelo_local_completion_evidence_v1','status':'LOCAL_OWNED_TERMINAL_AND_ARTIFACTS_VERIFIED_NO_SCIENTIFIC_PASS','run_id':run,'seed':1,'job_id':job_id,'job_sha256':job_sha,'terminal_sha256':terminal_sha,'supervisor_sha256':supervisor_sha,'best_epoch':best,'completion_summary':summary,'execution':execution,'migration_note':MIGRATION_NOTE,'source_records':list(unique.values()),'script_sha256':q.sha(SELF),'scientific_pass':False,'prediction_matrices_decoded':False,'created_utc':datetime.now(timezone.utc).isoformat()}

def validate_capture(path,digest,run):
 path=local(ROOT,path);e=pinned(path,digest)
 require(e.get('schema')=='dynavelo_local_completion_evidence_v1' and e.get('run_id')==run and RUN.fullmatch(run) and e.get('seed')==1 and e.get('scientific_pass') is False and e.get('script_sha256')==q.sha(SELF),'Completion evidence identity/code differs')
 actual=build_capture(ROOT,e['job_id'],run,e['job_sha256'],e['terminal_sha256'],e['supervisor_sha256'])
 require({k:v for k,v in e.items() if k!='created_utc'}=={k:v for k,v in actual.items() if k!='created_utc'},'Completion evidence/source census drift')
 return e

def function_source(module,name):
 source=raw(Path(module.__file__)).decode('utf-8-sig')
 return ast.get_source_segment(source,next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))

def once(source,old,new):
 require(source.count(old)==1,'Frozen adapter anchor differs: '+old[:80]);return source.replace(old,new,1)

def adapted_finish():
 """Reuse original statistical/output routine; substitute transport provenance only."""
 source=function_source(q,'finish')
 source=once(source,"'results/reports/factual_readouts/'","'results/reports/local_factual_readouts/'")
 source=once(source,"'schema':'dynavelo_factual_readout_v1'","'schema':'dynavelo_local_factual_readout_v1'")
 source=once(source,"'review_sha256':review_sha,'binding':review['binding']","'completion_evidence':review['binding'],'execution':review['execution'],'migration_note':MIGRATION_NOTE,'reused_readout_sha256':PINS['repo/pipeline/audit_factual_run_readout.py']")
 namespace={**vars(q),'__file__':str(SELF),'MIGRATION_NOTE':MIGRATION_NOTE,'PINS':PINS}
 exec(compile(source,'<unchanged_local_factual_finish_transport_metadata>','exec'),namespace)
 return namespace['finish']

def readout(evidence_path,evidence_sha,run,output_id):
 e=validate_capture(evidence_path,evidence_sha,run);check_output_id(output_id)
 import numpy as np,h5py
 root=ROOT;base='checkpoints/GSE284047/'+run+'/';records={r['relative_path']:r for r in e['source_records']};used=[]
 def use(rel):
  require(rel in records,'File absent from completed local inventory: '+rel);used.append(records[rel]);return q.checked(root,records[rel])
 config=q.read(use(base+'config.json'));metrics=q.read(use(base+'metrics.json'));best=e['best_epoch']
 history=q.validate_history(raw(use(base+'training_metrics.tsv')),metrics);checkpoint_check=q.finite_checkpoint(use(base+'model_best.pth'))
 # Entire numerical tail is extracted from pinned v1. Only path-registry spelling changes.
 tail=function_source(q,'execute');tail=tail[tail.index(' core=old_core();'):]
 tail=once(tail,"CLOUD+pred.relative_to(root).as_posix()","str(pred)")
 review={'run_id':run,'seed':1,'output_id':output_id,'script_sha256':q.sha(SELF),'binding':q.record(Path(evidence_path),root),'execution':e['execution']}
 original_finish=adapted_finish()
 def finish(*args):
  validate_capture(evidence_path,evidence_sha,run)
  return original_finish(*args)
 namespace={**vars(q),'__file__':str(SELF),'np':np,'h5py':h5py,'cloud':local,'finish':finish}
 wrapper='def execute_tail(root,review_path,review_sha,review,run,config,metrics,best,history,checkpoint_check,used,use):\n'+tail+'\n'
 exec(compile(wrapper,'<pinned_factual_numerical_tail_local_path_registry>','exec'),namespace)
 return namespace['execute_tail'](root,Path(evidence_path),evidence_sha,review,run,config,metrics,best,history,checkpoint_check,used,use)

def validate_readout(audit_path,audit_sha,run):
 p=local(ROOT,audit_path);a=pinned(p,audit_sha)
 require(a.get('schema')=='dynavelo_local_factual_readout_v1' and a.get('run_id')==run and a.get('seed')==1 and a.get('status')==SCOPE and a.get('scientific_pass') is False and a.get('script_sha256')==q.sha(SELF),'Exact completed local readout required')
 require(q.read(p.parent/'COMPLETE.json')=={'audit':q.record(p,ROOT),'scientific_pass':False},'Readout COMPLETE mismatch')
 for r in a['artifacts']:q.checked(ROOT,r)
 er=a['completion_evidence'];validate_capture(q.checked(ROOT,er),er['sha256'],run)
 return a

def comparison_tail_source(c):
 tail=function_source(c,'execute');tail=tail[tail.index(' time_rec=next'):]
 tail=once(tail,"q.RUN_RE.fullmatch(review.get('run_id',''))","RUN.fullmatch(review.get('run_id',''))")
 tail=once(tail,"'results/reports/factual_time_comparisons/'","'results/reports/local_factual_time_comparisons/'")
 tail=once(tail,"schema='dynavelo_paired_raw_beta_comparison_v1',status='DESCRIPTIVE_SEED_STABILITY_ONLY'","schema='dynavelo_local_paired_raw_beta_comparison_v1',status='DESCRIPTIVE_SEED_AND_BACKEND_MIGRATION_SENSITIVITY_ONLY'")
 tail=once(tail,"review_sha256=review_sha)","local_audit_sha256=review_sha,migration_note=MIGRATION_NOTE,execution=new['execution'],reused_comparison_sha256=PINS['repo/pipeline/compare_factual_raw_times.py'])")
 tail=once(tail," out=q.safe(root,", " validate_readout(review_path,review_sha,review['run_id'])\n for row in result['correlations']:row['coefficient_scope']='descriptive same cells; seed plus backend/version migration; no donor replication'\n out=q.safe(root,")
 return 'def execute_tail(root,review_path,review_sha,review,new_path,new):\n'+tail+'\n'

def compare(audit_path,audit_sha,run,output_id):
 new=validate_readout(audit_path,audit_sha,run);check_output_id(output_id)
 import numpy as np,h5py
 from anndata.experimental import read_elem
 import compare_factual_raw_times as c
 review={'run_id':run,'output_id':output_id,'new_audit':q.record(Path(audit_path),ROOT),'script_sha256':q.sha(SELF)}
 namespace={**vars(c),'__file__':str(SELF),'np':np,'h5py':h5py,'read_elem':read_elem,'RUN':RUN,'MIGRATION_NOTE':MIGRATION_NOTE,'PINS':PINS,'validate_readout':validate_readout}
 exec(compile(comparison_tail_source(c),'<pinned_raw_beta_comparison_local_evidence>','exec'),namespace)
 return namespace['execute_tail'](ROOT,Path(audit_path),audit_sha,review,Path(audit_path),new)

def check_output_id(value):require(re.fullmatch('[A-Za-z0-9_-]{8,100}',value or '') is not None,'Explicit unique output ID required')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('inspect','capture','readout','compare'),default='inspect');p.add_argument('--execute',action='store_true')
 p.add_argument('--run-id');p.add_argument('--job-id');p.add_argument('--job-sha256');p.add_argument('--terminal-sha256');p.add_argument('--supervisor-sha256',default=SUPERVISOR_SHA);p.add_argument('--evidence',type=Path);p.add_argument('--evidence-sha256');p.add_argument('--audit',type=Path);p.add_argument('--audit-sha256');p.add_argument('--output-id')
 a=p.parse_args()
 if a.action=='inspect':
  print(json.dumps({'status':'INSPECT_ONLY_NO_RESULT_FILES_OPENED','required_before_capture':'Explicit local run + owned job JSON SHA + successful empty terminal JSON SHA; exact console final summary; complete migration/final artifacts. Unknown terminal stays HOLD. No Colab ZIP required.','migration_note':MIGRATION_NOTE,'scientific_pass':False}));return
 require(a.execute and RUN.fullmatch(a.run_id or ''),'Explicit execute and exact local seed1 run required');check_output_id(a.output_id)
 if a.action=='capture':
  require(bool(a.job_id and a.job_sha256 and a.terminal_sha256),'Actual job/terminal SHA required; unknown terminal is HOLD')
  value=build_capture(ROOT,a.job_id,a.run_id,a.job_sha256,a.terminal_sha256,a.supervisor_sha256)
  out=q.safe(ROOT,'results/reports/local_completion_evidence/'+a.output_id);out.mkdir(parents=True,exist_ok=False);path=out/'evidence.json';q.save(path,value)
  print(json.dumps({'status':value['status'],'evidence':str(path),'sha256':q.sha(path),'scientific_pass':False}));return
 if a.action=='readout':
  require(a.evidence is not None and bool(a.evidence_sha256),'Pinned completed evidence required');value=readout(a.evidence,a.evidence_sha256,a.run_id,a.output_id)
 else:
  require(a.audit is not None and bool(a.audit_sha256),'Pinned completed local readout required');value=compare(a.audit,a.audit_sha256,a.run_id,a.output_id)
 print(json.dumps({'status':value['status'],'run_id':a.run_id,'scientific_pass':False}))

if __name__=='__main__':
 try:main()
 except (ValueError,FileNotFoundError,KeyError) as error:
  print(json.dumps({'status':'HOLD_LOCAL_EVIDENCE_OR_READOUT_INCOMPLETE','reason':str(error),'scientific_pass':False}));raise SystemExit(2)
