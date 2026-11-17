"""Pinned Windows CPU/XPU resume; inspect by default. No parent/source edits."""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.metadata,itertools,json,math,os,random,re,shutil,sys,tempfile,time,traceback,types
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from assemble_model_inputs import absolute_project_path,sha256,write_json,write_h5ad_atomic
from dynavelo_cpu_adapter import PROJECT_ROOT
ROOT=PROJECT_ROOT;SELF=Path(__file__).resolve();CLOUD='/content/DynaVelo_Research'
FROZEN={
 'repo/colab/colab_training.py':'c70c59817b13b4553318ae3e24c59a9c56811bd941e6534c455b3fbfeec4b2c7',
 'repo/colab/train_hspc_colab.py':'f611477ceb94776c0e3aa7982a55bfa42eb006abd3dd9abc3930a44999b01aa7',
 'repo/colab/dynavelo_colab_adapter.py':'81f3f5c2e05391b8349ed65cd34401b2ceeda75761333502dcbf4b555c91667c',
 'repo/pipeline/dynavelo_cpu_adapter.py':'0f29d3bd4c919d743820c6a022266aba2358a72348aa84385461d57c1aebc05f',
 'repo/pipeline/assemble_model_inputs.py':'e3d76669b4740e81f7760dae922543f1af67627e648bbac76dceed36e13ee3d3',
 'repo/DynaVelo/dynavelo/models.py':'ff9ad39be46382a731d03b8899e4a0ccb7e82c93dfecb6ff1fd572f251e118ca',
 'repo/DynaVelo/dynavelo/utils.py':'34b040e6108c2c9441b431f4524c304b1e88d51fa4f47c7ebc1e734bff6cc523'}
CORE_VERSIONS={'numpy':'1.26.4','scipy':'1.13.1','pandas':'2.2.3','anndata':'0.10.9','h5py':'3.12.1','torchmetrics':'1.6.1','torchdiffeq':'0.2.5'}
FIT={'seed':1,'batch_size':64,'learning_rate':.001,'hidden':200,'zx_dim':50,'zy_dim':50}
ARCH={'num_hidden':200,'zx_dim':50,'zy_dim':50,'k_t':1000,'k_z0':1000,'k_velocity':10000,'k_consistency':10000,'sigma_x':.1,'sigma_y':.1,'mu_pz0':0,'sigma_pz0':1,'alpha_pt':2,'beta_pt':2,'time_variant':False}
INPUTS=('data/processed/colab_minimal/HSPC/adata_rna.h5ad','data/processed/colab_minimal/HSPC/adata_atac.h5ad')
RUN_RE=re.compile(r'8489-MV-1_(?:cuda|cpu|xpu)_seed1_\d{8}T\d{12}Z')
LOGGER_FIELDS={'Epoch','loss_train','loss_test','nll_x','nll_y','loss_vel','loss_con','kl_z0','kl_t'}

def pinned_json(path,digest):
    path=absolute_project_path(path)
    if path.stat().st_size>16*1024**2:raise ValueError('Oversized metadata')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Trusted metadata SHA mismatch')
    return raw,json.loads(raw.decode('utf-8-sig'))

def original_functions(relative,names):
    source=(ROOT/relative).read_text(encoding='utf-8');tree=ast.parse(source)
    found={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    return '\n\n'.join(found[name] for name in names)+'\n'

def once(source,old,new):
    if source.count(old)!=1:raise ValueError('Adaptation anchor differs: '+old[:80])
    return source.replace(old,new,1)

_ns={};exec(compile(original_functions('repo/colab/colab_training.py',['immutable_config']),'<pinned_immutable_config>','exec'),_ns)
immutable_config=_ns['immutable_config']

def source_check(device):
    for path,digest in FROZEN.items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen source drift: '+path)
    versions={n:importlib.metadata.version(n) for n in ('torch',*CORE_VERSIONS)}
    if any(versions[n].split('+')[0]!=v for n,v in CORE_VERSIONS.items()):raise ValueError('Scientific dependency version differs: '+str(versions))
    if device=='cpu' and versions['torch'].split('+')[0]!='2.5.1':raise ValueError('CPU route requires existing torch 2.5.1')
    return versions

def finite_tree(value):
    import torch
    if isinstance(value,torch.Tensor):
        if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite checkpoint tensor')
    elif isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite_tree(v)
    elif isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite checkpoint scalar')

class ShortTestComplete(Exception):pass

def audit_state(state,prior,manifest):
    import torch
    epoch=manifest['completed_epoch']
    if state.get('schema')!='dynavelo_epoch_resume_v1' or state.get('completed_epoch')!=epoch or state.get('immutable_config')!=immutable_config(prior):raise ValueError('Parent epoch/config mismatch')
    entries={r['role']:r for r in manifest['required_files']}
    if state.get('best_snapshot_filename')!=PurePosixPath(entries['best_snapshot']['relative_path']).name or state.get('best_checkpoint_sha256')!=entries['best_snapshot']['sha256']:raise ValueError('Best pointer/SHA differs')
    history=state.get('history');best=None;counter=0
    if not isinstance(history,list) or len(history)!=epoch:raise ValueError('Incomplete history')
    for i,row in enumerate(history,1):
        if type(row.get('Epoch')) is not int or row['Epoch']!=i or not LOGGER_FIELDS.issubset(row):raise ValueError('Noncontiguous/missing full loss history')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in row.values()):raise ValueError('Nonfinite history')
        loss=row['loss_test']
        if best is None:best=loss
        elif best-loss>0:best=loss;counter=0
        elif best-loss<0:counter+=1
        if counter>=10 and i!=epoch:raise ValueError('Continued after original early-stop')
    if state.get('best_loss')!=best or type(state.get('early_stopping_counter')) is not int or state['early_stopping_counter']!=counter:raise ValueError('Best/counter disagrees with history')
    s=state.get('scheduler_state_dict',{});groups=state.get('optimizer_state_dict',{}).get('param_groups',[])
    if s.get('mode')!='min' or s.get('patience')!=5 or s.get('factor')!=.5 or s.get('min_lrs')!=[1e-6] or s.get('last_epoch')!=epoch:raise ValueError('Scheduler differs')
    if len(groups)!=1 or s.get('_last_lr')!=[groups[0].get('lr')] or groups[0].get('fused') or groups[0].get('capturable'):raise ValueError('Adam/scheduler differs')
    rng=state.get('rng_state',{})
    if not {'python','numpy','torch_cpu','torch_cuda','train_sampler'}.issubset(rng) or set(rng)-{'python','numpy','torch_cpu','torch_cuda','torch_xpu','train_sampler'}:raise ValueError('Missing/unknown RNG state')
    for key in ('torch_cpu','train_sampler'):
        if not isinstance(rng[key],torch.Tensor) or rng[key].dtype!=torch.uint8 or rng[key].ndim!=1:raise ValueError('Invalid RNG tensor')
    if not isinstance(rng['torch_cuda'],list):raise ValueError('Invalid CUDA RNG archive')
    # ReduceLROnPlateau(mode=min) intentionally stores mode_worse=+inf.
    # It is a scheduler sentinel, not a learned nonfinite value.
    if s.get('mode_worse')!=float('inf') or not math.isfinite(state.get('elapsed_training_seconds',float('nan'))):raise ValueError('Scheduler sentinel/elapsed metadata differs')
    finite_tree(state['model_state_dict']);finite_tree(state['optimizer_state_dict']);finite_tree(rng)
    return {'completed_epoch':epoch,'next_epoch':epoch+1,'n_history_rows':len(history),'best_epoch_last_tie':max(r['Epoch'] for r in history if r['loss_test']==best),'best_loss':best,'early_stopping_counter':counter,'current_learning_rate':groups[0]['lr'],'saved_cuda_generators':len(rng['torch_cuda']),'already_stopped':counter>=10 or epoch>=200}

def inspect_parent(path,digest,device='cpu'):
    versions=source_check(device);raw,m=pinned_json(path,digest);parent=Path(path).parent;run=m.get('run_id','')
    if not RUN_RE.fullmatch(run) or parent.name!=run or m.get('schema')!='dynavelo_colab_checkpoint_restore_v1' or m.get('cloud_root') not in (CLOUD,str(ROOT)):raise ValueError('Unexpected parent run/root/schema')
    if type(m.get('completed_epoch')) is not int or not 1<=m['completed_epoch']<=200:raise ValueError('Invalid completed epoch')
    entries=m.get('required_files');expected={'checkpoint':{'last_checkpoint.pth'},'prior_config':{'resume_config.json'},'best_snapshot':{'best_snapshot_0.pth','best_snapshot_1.pth'}};records={}
    if not isinstance(entries,list) or len(entries)!=3:raise ValueError('Exactly three recovery inputs required')
    for row in entries:
        role=row.get('role');rel=PurePosixPath(row.get('relative_path',''))
        if role in records or role not in expected or str(rel.parent)!='checkpoints/GSE284047/'+run or rel.name not in expected[role]:raise ValueError('Unsafe/duplicate recovery role')
        p=absolute_project_path(parent/rel.name)
        if type(row.get('bytes')) is not int or not 0<row['bytes']<=128*1024**2 or p.stat().st_size!=row['bytes'] or sha256(p)!=row.get('sha256'):raise ValueError('Parent bytes/SHA differs')
        records[role]={'path':str(p),'bytes':row['bytes'],'sha256':row['sha256']}
    _,prior=pinned_json(records['prior_config']['path'],records['prior_config']['sha256'])
    if prior.get('run_id')!=run or prior.get('model_defaults')!=ARCH or any(prior.get('command_arguments',{}).get(k)!=v for k,v in FIT.items()):raise ValueError('Parent science/fit identity differs')
    if tuple(prior.get(k) for k in ('n_cells','n_genes','n_TF','n_train','n_validation','split_seed'))!=(6035,3006,365,4828,1207,0):raise ValueError('Population/split differs')
    historic=m['cloud_root'];inputs={};paths=[CLOUD+'/'+r if historic==CLOUD else str(ROOT/r) for r in INPUTS]
    if set(prior['input_files'])!=set(paths):raise ValueError('Unexpected historical input paths')
    for relative,oldpath in zip(INPUTS,paths):
        p=absolute_project_path(ROOT/relative);h=sha256(p)
        if h!=prior['input_files'][oldpath]:raise ValueError('Actual input bytes differ')
        inputs[relative]={'historical_path':oldpath,'local_path':str(p),'bytes':p.stat().st_size,'sha256':h}
    if historic==CLOUD:
        if prior.get('script_sha256')!=FROZEN['repo/colab/train_hspc_colab.py'] or prior.get('training_implementation_sha256')!=FROZEN['repo/colab/colab_training.py'] or prior['adapter_manifest'].get('adapter_sha256')!=FROZEN['repo/colab/dynavelo_colab_adapter.py']:raise ValueError('Parent cloud source differs')
    elif prior.get('script_sha256')!=sha256(SELF) or prior.get('training_implementation_sha256')!=sha256(SELF):raise ValueError('Local resume requires exact execution source')
    for key,p in (('upstream_models_sha256','repo/DynaVelo/dynavelo/models.py'),('upstream_utils_sha256','repo/DynaVelo/dynavelo/utils.py')):
        if prior['adapter_manifest'].get(key)!=FROZEN[p]:raise ValueError('Parent upstream source differs')
    if prior.get('analytic_velocity') is not False or prior['adapter_manifest'].get('analytic_velocity') is not False:raise ValueError('Original Jacobian required')
    import torch
    state=torch.load(records['checkpoint']['path'],map_location='cpu',weights_only=True);summary=audit_state(state,prior,m)
    best=torch.load(records['best_snapshot']['path'],map_location='cpu',weights_only=True);finite_tree(best)
    if set(best)!={'model_state_dict','optimizer_state_dict'} or set(best['model_state_dict'])!=set(state['model_state_dict']):raise ValueError('Best/last axes differ')
    for name,t in state['model_state_dict'].items():
        if best['model_state_dict'][name].shape!=t.shape or best['model_state_dict'][name].dtype!=t.dtype:raise ValueError('Best/last shape/dtype differs')
    return {'schema':'dynavelo_local_resume_inspection_v1','status':'INSPECTED_NOT_TRAINING','manifest_path':str(path),'manifest_sha256':digest,'manifest':m,'prior_config':prior,'parent_files':records,'actual_inputs':inputs,'state_summary':summary,'actual_package_versions':versions,'local_source_sha256':sha256(SELF),'frozen_sources':FROZEN,'target_device':device,'bitwise_continuation_claim':False,'model_constructed':False,'training_started':False,'source_trust':'Caller supplies SHA from independently verified final capture; this entry does not attest cloud termination.'}

def rng_functions(device):
    def capture(loader):
        import numpy as np,torch
        v=np.random.get_state()
        result={'python':random.getstate(),'numpy':[v[0],v[1].tolist(),v[2],v[3],v[4]],'torch_cpu':torch.get_rng_state(),'torch_cuda':[], 'train_sampler':loader.batch_sampler.generator.get_state()}
        if device=='xpu':result['torch_xpu']=torch.xpu.get_rng_state_all()
        return result
    def restore(state,loader):
        import numpy as np,torch
        random.setstate(state['python']);v=state['numpy'];np.random.set_state((v[0],np.asarray(v[1],dtype=np.uint32),v[2],v[3],v[4]))
        torch.set_rng_state(state['torch_cpu'].cpu());loader.batch_sampler.generator.set_state(state['train_sampler'].cpu())
        if device=='xpu':
            if 'torch_xpu' in state:torch.xpu.set_rng_state_all([v.cpu() for v in state['torch_xpu']])
            else:torch.xpu.manual_seed_all(1) # explicitly NEW XPU stream; never convert CUDA bytes
    return capture,restore

def local_adapter(device):
    import torch
    if device=='xpu' and (not hasattr(torch,'xpu') or not torch.xpu.is_available()):raise RuntimeError('Requested XPU unavailable; no CPU fallback')
    source=(ROOT/'repo/pipeline/dynavelo_cpu_adapter.py').read_text(encoding='utf-8')
    source=once(source,'if str(kwargs.get("device", "cpu")) != "cpu":','if str(kwargs.get("device", _LOCAL_DEVICE)) != _LOCAL_DEVICE:')
    source=once(source,'kwargs["device"] = "cpu"','kwargs["device"] = _LOCAL_DEVICE')
    source=once(source,'            super().__init__(*args, **kwargs)\n            if analytic_velocity:',
                '            super().__init__(*args, **kwargs)\n            self.to(torch.device(_LOCAL_DEVICE))\n            if analytic_velocity:')
    ns={'__name__':'_dynavelo_explicit_local_adapter','__file__':str(SELF),'_LOCAL_DEVICE':device}
    exec(compile(source,'<explicit_local_adapter_from_pinned_cpu_source>','exec'),ns)
    def load(*args,**kwargs):
        if kwargs.get('analytic_velocity',False):raise ValueError('Only original Jacobians authorized')
        Dataset,Model,meta=ns['load_dynavelo'](*args,**kwargs)
        return Dataset,Model,{**meta,'device':device,'base_cpu_adapter_sha256':FROZEN['repo/pipeline/dynavelo_cpu_adapter.py'],
          'local_resume_source_sha256':sha256(SELF),'local_adapter_compiled_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'device_adaptation':'Explicit constructor device and whole model placement; original losses/Jacobians/ODE retained.'}
    return load

def configure_output_paths(model,run_dir,config):
    """Windows-safe canonical/log filenames; no model parameter changes."""
    canonical=absolute_project_path(run_dir/'model_best.pth')
    log=absolute_project_path(run_dir/'model.log')
    if max(len(str(canonical)+'.partial'),len(str(log)))>=260:raise ValueError('Local output root remains too long')
    config['local_model_output_paths']={'original_model_suffix':model.model_suffix,
      'original_canonical_path_historical_only':str(model.ckpt_path),'canonical_best_checkpoint':str(canonical),
      'epoch_log':str(log),'adaptation':'Filename shortening only; architecture and numerical training unchanged.'}
    model.ckpt_path=str(canonical);model.model_suffix='model'
    write_json(absolute_project_path(run_dir/'config.json'),config)

def construct_modules(binding,migration,run_dir,probe_batches=None):
    device=binding['target_device']
    common=dict(PROJECT_ROOT=ROOT,absolute_project_path=absolute_project_path,sha256=sha256,write_json=write_json,
      write_h5ad_atomic=write_h5ad_atomic,datetime=datetime,timezone=timezone,json=json,random=random,shutil=shutil,
      time=time,math=math,sys=sys,importlib=importlib,configure_output_paths=configure_output_paths,
      _LOCAL_SOURCE=SELF,_LOCAL_DEVICE=device,__file__=str(SELF))
    names=['immutable_config','_save_atomic','_publish_best','_manifest_record','write_checkpoint_restore_manifest','fit_resumable','make_download_manifest']
    hs=original_functions('repo/colab/colab_training.py',names)
    hs=once(hs,'state.get("schema") != "dynavelo_epoch_resume_v1" or state["immutable_config"] != immutable_config(config)',
            'not _validate_migration_state(state, config, verified)')
    helper=types.ModuleType('_dynavelo_local_resume_helper');helper.__dict__.update(common)
    def verified(manifest_path,checkpoint_path):
        if sha256(run_dir/'migration_manifest.json')!=migration['migration_manifest_sha256']:raise ValueError('Migration manifest raw bytes changed')
        if Path(checkpoint_path)!=run_dir/'last_checkpoint.pth' or Path(manifest_path)!=run_dir/'checkpoint_restore_manifest.json':raise ValueError('Wrong derived resume location')
        value=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if value.get('local_migration_manifest_sha256')!=migration['migration_manifest_sha256'] or value['completed_epoch']!=binding['manifest']['completed_epoch']:raise ValueError('Derived migration/epoch binding differs')
        result={'completed_epoch':binding['manifest']['completed_epoch'],'manifest':str(manifest_path),'manifest_sha256':sha256(manifest_path)}
        entries=value.get('required_files',[])
        if len(entries)!=3 or {r['role'] for r in entries}!=set(binding['parent_files']):raise ValueError('Derived manifest roles differ')
        for row in entries:
            role=row['role'];expected=binding['parent_files'][role];p=run_dir/Path(expected['path']).name
            if row['relative_path']!='checkpoints/GSE284047/'+run_dir.name+'/'+p.name or row['sha256']!=expected['sha256'] or row['bytes']!=expected['bytes'] or p.stat().st_size!=expected['bytes'] or sha256(p)!=expected['sha256']:raise ValueError('Derived core differs from immutable parent')
            result[role]=str(p)
        result['best_snapshot_sha256']=binding['parent_files']['best_snapshot']['sha256'];return result
    def validate(state,config,checked):
        audit_state(state,binding['prior_config'],binding['manifest'])
        if config.get('local_migration')!=migration or config['script_sha256']!=sha256(SELF) or config['training_implementation_sha256']!=sha256(SELF):raise ValueError('Actual execution identity differs')
        if config['input_files']!={r['local_path']:r['sha256'] for r in binding['actual_inputs'].values()} or config['model_defaults']!=ARCH or any(config['command_arguments'][k]!=v for k,v in FIT.items()):raise ValueError('Local science/input identity differs')
        if config['adapter_manifest'].get('local_resume_source_sha256')!=sha256(SELF) or config['adapter_manifest'].get('analytic_velocity') is not False or config['adapter_manifest'].get('device')!=device:raise ValueError('Actual adapter identity missing')
        for row in state['history']:
            if not set(config['_logger_fields']).issubset(row):raise ValueError('Full logger history missing')
        return True
    capture,restore=rng_functions(device)
    helper.__dict__.update(_restore_rng=restore,_rng_state=capture,verify_checkpoint_restore_manifest=verified,_validate_migration_state=validate)
    exec(compile(hs,'<local_resume_helper_from_pinned_cloud_source>','exec'),helper.__dict__)
    functions=['batches_without_singleton','dense32','subset_for_dataset','validate_inputs','reconstruction_metrics','velocity_metrics','chunked_evaluate','run']
    ts=original_functions('repo/colab/train_hspc_colab.py',functions)
    start=ts.index('    if not torch.cuda.is_available():');end=ts.index('    rna, atac =',start)
    ts=ts[:start]+'''    gpu_info = {"device": _LOCAL_DEVICE, "torch_version": torch.__version__,
                "threads": args.threads, "precision": "float32; original Jacobian; no AMP",
                "backend_transition": "CUDA RNG archived, not converted; new XPU stream seeded 1 if needed",
                "bitwise_continuation_claim": False}
    if hasattr(torch, "set_float32_matmul_precision"): torch.set_float32_matmul_precision("highest")
'''+ts[end:]
    ts=once(ts,'"cuda_runtime": gpu_info','"local_runtime": gpu_info')
    ts=once(ts,'sha256(PROJECT_ROOT / "repo/colab/colab_training.py")','sha256(_LOCAL_SOURCE)')
    ts=once(ts,'"batching": "CUDA; default 64;','"batching": "Explicit local device; default 64;')
    ts=once(ts,'device="cuda:0", mode="train"','device=_LOCAL_DEVICE, mode="train"')
    ts=once(ts,'    optimizer = FiniteAdam(model.parameters(), lr=args.learning_rate)',
        '    config["_logger_fields"] = list(model.logger.names)\n    configure_output_paths(model, run_dir, config)\n    optimizer = FiniteAdam(model.parameters(), lr=args.learning_rate)')
    runner=types.ModuleType('_dynavelo_local_resume_runner');runner.__dict__.update(common)
    def preserve(directory,config,checkpoint,prior_config=None):
        if Path(directory)!=run_dir or Path(prior_config)!=run_dir/'resume_config.json' or sha256(prior_config)!=binding['parent_files']['prior_config']['sha256']:raise ValueError('Raw parent configuration differs')
        config['local_migration']=migration;config['compiled_helper_sha256']=hashlib.sha256(hs.encode()).hexdigest();config['compiled_runner_sha256']=hashlib.sha256(ts.encode()).hexdigest()
        config['resume_parent_config_sha256']=binding['parent_files']['prior_config']['sha256'];config['resume_parent_config']=str(run_dir/'parent_resume_config.raw.json')
    runner.__dict__.update(load_dynavelo=local_adapter(device),preserve_resume_config=preserve,fit_resumable=helper.fit_resumable,
      benchmark=lambda *a:(_ for _ in ()).throw(ValueError('Use explicit short-test')))
    exec(compile(ts,'<local_resume_runner_from_pinned_cloud_source>','exec'),runner.__dict__)
    if probe_batches is not None:runner.fit_resumable=make_probe(helper,validate,binding,migration,device,probe_batches)
    return helper,runner,{'helper_sha256':hashlib.sha256(hs.encode()).hexdigest(),'runner_sha256':hashlib.sha256(ts.encode()).hexdigest()}

def make_probe(helper,validate,binding,migration,device,n_batches):
    def probe(model,train_loader,valid_loader,optimizer,args,directory,rows,config):
        import numpy as np,torch
        checked=helper.verify_checkpoint_restore_manifest(args.resume_manifest,args.resume_checkpoint)
        state=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=True);validate(state,config,checked)
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.5,min_lr=1e-6)
        scheduler.load_state_dict(state['scheduler_state_dict']);helper._restore_rng(state['rng_state'],train_loader)
        class Limited:
            def __init__(self,loader,n):
                self.batches=list(itertools.islice(loader,n));self.dataset=range(sum(len(b[0]) for b in self.batches))
            def __iter__(self):return iter(self.batches)
        tr,va=Limited(train_loader,n_batches),Limited(valid_loader,1)
        model.mode='train';started=time.monotonic()
        loss=float(model.train_step(tr,optimizer));values=list(map(float,model.test_step(va)))
        if device=='xpu':torch.xpu.synchronize()
        if not np.isfinite([loss,*values]).all():raise FloatingPointError('Nonfinite short test')
        write_json(directory/'short_test.json',{'status':'SHORT_TEST_ONLY_NOT_RESUMABLE','device':device,
          'torch_version':torch.__version__,'local_source_sha256':sha256(SELF),'parent_manifest_sha256':binding['manifest_sha256'],
          'parent_completed_epoch':state['completed_epoch'],'train_batches':len(tr.batches),'validation_batches':len(va.batches),
          'seconds':time.monotonic()-started,'finite_training_loss':loss,'finite_validation_components':values,
          'restored_optimizer_scheduler_rng':True,'original_jacobian_forward_backward_adam':True,
          'checkpoint_saved':False,'scientific_conclusion':None,'model_disposition':'Discard; actual continuation restores immutable parent afresh.'})
        raise ShortTestComplete()
    return probe

def setup_storage():
    if os.name!='nt' or ROOT.drive.upper()!='D:':raise RuntimeError('Windows D: entry only')
    variables={'TMP':'tmp','TEMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'pip','HF_HOME':'hf','TORCH_HOME':'torch',
      'MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','XDG_CACHE_HOME':'xdg','JOBLIB_TEMP_FOLDER':'joblib'}
    for key,relative in variables.items():
        path=absolute_project_path(ROOT/'tmp_cache'/relative);path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    tempfile.tempdir=os.environ['TMPDIR'];os.environ['MPLBACKEND']='Agg'

def execute(binding,action,threads,run_id=None,probe_batches=2,device_evidence=None,device_evidence_sha=None):
    device=binding['target_device']
    if action not in ('train','short-test'):raise ValueError('Explicit execution action required')
    if inspect_parent(binding['manifest_path'],binding['manifest_sha256'],device)!=binding:raise ValueError('Parent/source/input identity changed')
    evidence=None
    if action=='train' and device=='xpu':
        if not device_evidence or not device_evidence_sha:raise ValueError('XPU train needs SHA-pinned local short-test evidence; no fallback')
        _,evidence=pinned_json(device_evidence,device_evidence_sha)
        expected={'status':'SHORT_TEST_ONLY_NOT_RESUMABLE','device':'xpu','torch_version':binding['actual_package_versions']['torch'],
          'local_source_sha256':sha256(SELF),'parent_manifest_sha256':binding['manifest_sha256'],
          'restored_optimizer_scheduler_rng':True,'original_jacobian_forward_backward_adam':True}
        if any(evidence.get(k)!=v for k,v in expected.items()):raise ValueError('XPU short-test evidence not this source/parent/runtime')
    setup_storage()
    run_id=run_id or '8489-MV-1_'+device+'_seed1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not RUN_RE.fullmatch(run_id) or '_'+device+'_' not in run_id:raise ValueError('New explicit-device run ID required')
    run_dir=absolute_project_path(ROOT/'checkpoints/GSE284047'/run_id)
    prediction=absolute_project_path(ROOT/'data/processed/GSE284047/8489-MV-1/model_runs'/run_id)
    cache=absolute_project_path(ROOT/'tmp_cache/prediction_arrays'/run_id)
    if any(p.exists() for p in (run_dir,prediction,cache)):raise FileExistsError('New local run only; parent never overwritten')
    for p in (run_dir,prediction,cache):p.mkdir(parents=True,exist_ok=False)
    migration={'schema':'dynavelo_local_backend_migration_v1','new_run_id':run_id,'action':action,
      'parent_run_id':binding['manifest']['run_id'],'parent_manifest_sha256':binding['manifest_sha256'],
      'parent_manifest_path':binding['manifest_path'],'parent_files':binding['parent_files'],
      'parent_completed_epoch':binding['manifest']['completed_epoch'],'actual_inputs':binding['actual_inputs'],
      'historical_immutable_config':immutable_config(binding['prior_config']),
      'actual_execution':{'root':str(ROOT),'device':device,'threads':threads,'source_sha256':sha256(SELF),'packages':binding['actual_package_versions']},
      'device_evidence':{'path':str(device_evidence),'sha256':device_evidence_sha} if evidence else None,
      'scientific_fit_fixed':FIT,'total_epoch_ceiling':200,'early_stop_patience':10,'raw_parent_bytes_retained':True,
      'CUDA_rng_applied_or_converted':False,'XPU_initial_stream':'Restore prior XPU state if present; otherwise explicitly seed new stream with 1',
      'bitwise_continuation_claim':False,'posterior':{'n_cells':6035,'chunk_cells':64,'expected_chunks':95,'samples_per_cell':8},
      'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(run_dir/'migration_manifest.json',migration)
    migration={**migration,'migration_manifest_sha256':sha256(run_dir/'migration_manifest.json')}
    for role,record in binding['parent_files'].items():
        target=run_dir/Path(record['path']).name
        with Path(record['path']).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024**2);dst.flush();os.fsync(dst.fileno())
        if sha256(target)!=record['sha256']:raise ValueError('Copied parent bytes differ')
    shutil.copyfile(run_dir/'resume_config.json',run_dir/'parent_resume_config.raw.json')
    shutil.copyfile(binding['manifest_path'],run_dir/'parent_restore_manifest.raw.json')
    derived=copy.deepcopy(binding['manifest']);derived.update(run_id=run_id,cloud_root=str(ROOT),local_migration_manifest_sha256=migration['migration_manifest_sha256'],observed_files=[])
    for row in derived['required_files']:row['relative_path']='checkpoints/GSE284047/'+run_id+'/'+PurePosixPath(row['relative_path']).name
    write_json(run_dir/'checkpoint_restore_manifest.json',derived)
    helper,runner,adaptation=construct_modules(binding,migration,run_dir,probe_batches if action=='short-test' else None)
    write_json(run_dir/'source_adaptation.json',{'status':'EXPLICIT_LOCAL_ADAPTATION','frozen_sources':FROZEN,'actual_entry_sha256':sha256(SELF),**adaptation,
      'changes':['isolated local storage','historical immutable validation separately from actual execution identity','applicable RNG restore','explicit CPU/XPU runtime/model placement','actual local source/config hashes','Windows-safe model_best.pth canonical and model.log filenames; original model suffix retained in config'],
      'fit_loop_change':'Only identity predicate; original epoch loop, optimizer calls, scheduler/early-stop/save order retained.'})
    args=argparse.Namespace(**FIT,max_epochs=200,n_samples=8,evaluation_chunk_cells=64,threads=threads,
      rna=ROOT/INPUTS[0],atac=ROOT/INPUTS[1],prediction_root=prediction.parent,resume_checkpoint=run_dir/'last_checkpoint.pth',
      resume_manifest=run_dir/'checkpoint_restore_manifest.json',training_only=False,benchmark_only=False,benchmark_batches=2,local_action=action)
    verification={'status':'LOCAL_PARENT_AND_INPUT_SHA_VERIFIED','resume_verification':helper.verify_checkpoint_restore_manifest(args.resume_manifest,args.resume_checkpoint),
      'parent_manifest_sha256':binding['manifest_sha256'],'migration_manifest_sha256':migration['migration_manifest_sha256']}
    src=(ROOT/'repo/colab/train_hspc_colab.py').read_text(encoding='utf-8');node=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='Tee')
    ns={};exec(compile(ast.get_source_segment(src,node),'<pinned_Tee>','exec'),ns)
    try:
        with (run_dir/'console.log').open('x',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(ns['Tee'](sys.stdout,log)),redirect_stderr(ns['Tee'](sys.stderr,log)):
                runner.run(args,run_id,run_dir,prediction,cache,verification)
    except ShortTestComplete:
        write_json(run_dir/'short_test_completion.json',{'status':'SHORT_TEST_ONLY_NOT_RESUMABLE','run_id':run_id})
    except BaseException as error:
        write_json(run_dir/'failure.json',{'status':'FAILED_LOCAL_EXECUTION','error':repr(error),'traceback':traceback.format_exc(),'biological_convergence':'NOT_ESTABLISHED'})
        raise
    finally:
        for record in binding['parent_files'].values():
            if sha256(record['path'])!=record['sha256']:raise RuntimeError('Protected parent bytes changed')
        if sha256(binding['manifest_path'])!=binding['manifest_sha256']:raise RuntimeError('Protected parent manifest changed')
    return {'run_id':run_id,'run_dir':str(run_dir),'action':action,'parent_completed_epoch':binding['manifest']['completed_epoch']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=absolute_project_path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--device',choices=['cpu','xpu'],default='cpu');p.add_argument('--action',choices=['inspect','short-test','train'],default='inspect')
    p.add_argument('--execute',action='store_true');p.add_argument('--threads',type=int,default=4);p.add_argument('--probe-batches',type=int,default=2)
    p.add_argument('--run-id');p.add_argument('--report',type=absolute_project_path)
    p.add_argument('--device-evidence',type=absolute_project_path);p.add_argument('--device-evidence-sha256')
    a=p.parse_args()
    if not 1<=a.threads<=32 or not 1<=a.probe_batches<=4:p.error('threads 1..32; probe-batches 1..4')
    if a.action!='inspect' and not a.execute:p.error('Explicit --execute required; default inspect never fits')
    binding=inspect_parent(a.manifest,a.manifest_sha256,a.device)
    if a.report:
        if a.report.exists():raise FileExistsError('Inspection report must be new')
        a.report.parent.mkdir(parents=True,exist_ok=True);write_json(a.report,binding)
    result=binding if a.action=='inspect' else execute(binding,a.action,a.threads,a.run_id,a.probe_batches,a.device_evidence,a.device_evidence_sha256)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
