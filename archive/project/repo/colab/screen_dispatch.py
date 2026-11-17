"""Reviewed baseline/all-TF RNA-only numerical dispatch; no approval generator.

GRN, joint, shortlist/full-cell follow-up remain unsupported. All execution uses
the unchanged validated streaming core. Limits are cooperative at forward/task
boundaries; an in-flight native forward is not hard-preempted by Python.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import csv
import io
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import uuid
from types import SimpleNamespace
import numpy as np
import torch
from human_return_common import safe_target, sha256
from screen_plan import authorize_stage, canonical, identity, validate_plan, verify_artifact, freeze_baseline_frame
from screen_task_registry import TaskRegistry, task_book
from dynavelo_streaming_core import Population, StreamingSession, digest_array, model_fingerprint, batches_without_singleton

FROZEN={
 'streaming_core':('dynavelo_streaming_core.py','94a7b832e06c1f0404fd20790e8968f93724d01b732029fd91e22a06a3ae6368'),
 'plan_module':('screen_plan.py','be88a90f502e86028c45075b2f6e6610e9fe430e9081fa11f4d39f18768cd0ab'),
 'registry_module':('screen_task_registry.py','8900035942f7b77bd6ea2c84ddd0586784e71d6a57d95fc46ab59b210c7cb72e'),
 'shortlist_module':('screen_shortlist.py','1ed5344d1b346e2024b8fbc608928615df6ee511eb950d099a8ab047906f6140')}
SUPPORTED={'baseline','matched-baseline','screen'}
ROOTS=(Path(r'D:\dyc_7\DynaVelo_Research'),Path('/content/DynaVelo_Research'))
def utc():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def checked_root(root):
    root=Path(root).absolute()
    if root.resolve()!=root or not any(root==base or root.is_relative_to(base) for base in ROOTS):
        raise ValueError('Unredirected D-project or /content project required; frozen core has no WSL root')
    return root
def record(root,path):
    path=Path(path);safe_target(root,path.relative_to(root).as_posix())
    return {'relative_path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':sha256(path)}

def validate_output_namespace(plan,root,output_root):
    root=checked_root(root);output_root=checked_root(output_root)
    relative=output_root.relative_to(root).as_posix();safe_target(root,relative)
    if plan['spec']['purpose']=='human_exploratory_input_sensitivity':
        if relative!='results/screen_runs/'+plan['plan_sha256']:
            raise ValueError('One human plan has exactly one registry namespace; renaming cannot reset attempts')
        if plan['spec']['dispatch_contract'].get('registry_execution_root')!=str(root):
            raise PermissionError('Cross-project-root migration is not implemented; require audited migration, never an empty fresh registry')
    return output_root

def validate_contract(plan,root,review_record):
    contract=plan['spec'].get('dispatch_contract',{})
    if contract.get('schema')!='dynavelo_screen_dispatch_contract_v1':raise ValueError('Freeze a dispatch contract in plan first')
    if contract.get('device') not in {'cpu','cuda:0'} or contract.get('dtype')!='float32':raise ValueError('Explicit float32 device required')
    versions=contract.get('package_versions',{})
    if versions.get('torch')!=torch.__version__ or versions.get('numpy')!=np.__version__:raise ValueError('Numerical package drift')
    for package in ('anndata','h5py','psutil','torchdiffeq'):
        if versions.get(package)!=importlib.metadata.version(package):raise ValueError('Input/resource package drift: '+package)
    if contract.get('training_variance_rule')!='float64_population_variance_ddof0_feature_blocks256':raise ValueError('Unknown variance/H rule')
    if contract.get('no_op_atol')!=0 or contract.get('analytic_velocity') is not False:raise ValueError('Exact no-op and original Jacobian required')
    inputs=plan['spec']['input_artifacts']
    for role,(name,expected) in FROZEN.items():
        path=verify_artifact(root,inputs[role])
        if sha256(Path(__file__).with_name(name))!=expected or inputs[role]['sha256']!=expected:raise ValueError('Frozen source drift: '+role)
        if path.read_bytes()!=Path(__file__).with_name(name).read_bytes():raise ValueError('Imported and plan source differ')
    if inputs['dispatch_module']['sha256']!=sha256(__file__):raise ValueError('Dispatcher not frozen into plan')
    verify_artifact(root,inputs['dispatch_module'])
    limits=contract.get('resource_limits',{})
    for key in ('wall_seconds','forward_seconds','rss_bytes','cuda_reserved_bytes','min_free_disk_bytes',
                'task_payload_bytes','input_dense_bytes','max_tasks_per_invocation','cpu_threads'):
        if type(limits.get(key)) not in {int,float} or not np.isfinite(limits[key]) or limits[key]<=0:raise ValueError('Explicit positive ceiling required: '+key)
    if any(type(limits[k]) is not int for k in ('max_tasks_per_invocation','cpu_threads')):raise ValueError('Integer limits required')
    review=read_json(verify_artifact(root,review_record))
    evidence=read_json(verify_artifact(root,review['evidence']['resource_budget']))
    if evidence.get('screen_dispatch_contract_sha256')!=identity(contract):raise PermissionError('Resource review does not bind exact dispatch contract')
    return contract

def training_variances(x,train):
    indices=np.flatnonzero(train)
    if not len(indices):raise ValueError('No training cells')
    result=np.empty(x.shape[1],dtype=np.float64)
    for start in range(0,x.shape[1],256):result[start:start+256]=np.var(x[indices,start:start+256].astype(np.float64),axis=0,ddof=0)
    return result

def verify_population(plan,population):
    spec=plan['spec'];minima=spec['global_minima']
    if (list(population.cell_ids)!=[c['cell_id'] for c in spec['cells']] or list(population.genes)!=[g['feature_id'] for g in spec['genes']]
        or list(population.motif_tfs)!=[m['tf_rna_feature_id'] for m in spec['motifs']]):raise ValueError('Owned population axes differ')
    for field,key in [('x','rna_matrix_sha256'),('y','motif_matrix_sha256')]:
        if digest_array(getattr(population,field))!=minima[key]:raise ValueError('Actual full matrix hash differs')
    for field,key in [('x_min','rna'),('y_min','motif')]:
        actual=getattr(population,field);matrix=population.x if field=='x_min' else population.y
        if not np.array_equal(actual,matrix.min(axis=0)) or not np.array_equal(actual,np.asarray(minima[key],dtype=np.float32)):
            raise ValueError('Full-population floor differs; subset minimum forbidden')
    variance=training_variances(population.x,np.array([c['split']=='train' for c in spec['cells']]))
    if not np.array_equal(variance,np.asarray([g['training_variance'] for g in spec['genes']],dtype=np.float64)):
        raise ValueError('Training variance differs from frozen H selection')

def _dense_backed(matrix):
    if np.dtype(matrix.dtype)!=np.dtype('float32'):raise ValueError('Model X must be float32; no implicit conversion')
    result=np.empty(matrix.shape,dtype=np.float32)
    for start in range(0,matrix.shape[0],256):
        block=matrix[start:start+256]
        result[start:start+256]=block.toarray() if hasattr(block,'toarray') else block
    return result

def _input_view(path):
    """AnnData backed loading may read layers: open only X/obs/var groups."""
    import h5py
    from anndata._io.specs import read_elem
    from anndata._core.sparse_dataset import sparse_dataset
    handle=h5py.File(path,'r')
    try:
        obs=read_elem(handle['obs']);var=read_elem(handle['var']);node=handle['X']
        matrix=sparse_dataset(node) if isinstance(node,h5py.Group) else node
        return SimpleNamespace(file=handle,obs=obs,var=var,obs_names=obs.index,var_names=var.index,n_obs=len(obs),n_vars=len(var),X=matrix)
    except BaseException:handle.close();raise

def load_population(plan,root,contract):
    """Only X/identity HDF5 groups; no layers/old graph/raw matrices."""
    columns=contract['input_columns'];inputs=plan['spec']['input_artifacts']
    rna=_input_view(verify_artifact(root,inputs['rna']));atac=None
    try:
        atac=_input_view(verify_artifact(root,inputs['atac']))
        if not rna.obs_names.equals(atac.obs_names):raise ValueError('Paired cell order mismatch')
        cells=[{'cell_id':str(c),'label':str(l),'split':str(s)} for c,l,s in zip(rna.obs_names,rna.obs[columns['label']],rna.obs[columns['split']])]
        if canonical(cells)!=canonical(plan['spec']['cells']):raise ValueError('Cell/label/split metadata differs')
        for name in ('label','split'):
            if not np.array_equal(rna.obs[columns[name]].astype(str),atac.obs[columns[name]].astype(str)):raise ValueError('Paired label/split differs')
        if list(rna.var_names)!=[g['feature_id'] for g in plan['spec']['genes']]:raise ValueError('RNA axis differs')
        if list(atac.var_names)!=[m['feature_id'] for m in plan['spec']['motifs']]:raise ValueError('Motif axis differs')
        for key in ('stable_id','eligible_tf','curated_tf'):
            actual=rna.var[columns[key]].tolist()
            if key.endswith('_tf'):
                if any(not isinstance(v,(bool,np.bool_)) for v in actual):raise ValueError('TF flags must be Boolean')
                actual=[bool(v) for v in actual]
            else:actual=[str(v) for v in actual]
            if actual!=[g[key] for g in plan['spec']['genes']]:raise ValueError('RNA identity/TF annotation changed')
        for field,column in [('tf_rna_feature_id','motif_tf'),('source_motif_id','source_motif_id')]:
            if list(atac.var[columns[column]].astype(str))!=[m[field] for m in plan['spec']['motifs']]:raise ValueError('Motif mapping/provenance changed')
        if 3*(rna.n_obs*rna.n_vars+atac.n_obs*atac.n_vars)*4>contract['resource_limits']['input_dense_bytes']:raise MemoryError('Input load/copy envelope exceeds budget')
        population=Population(_dense_backed(rna.X),_dense_backed(atac.X),rna.obs_names,rna.var_names,atac.var[columns['motif_tf']])
    finally:
        rna.file.close()
        if atac is not None:atac.file.close()
    verify_population(plan,population);return population

class StopRequested(KeyboardInterrupt):pass
class ResourceStop(RuntimeError):pass
class ResourceGuard:
    def __init__(self,root,contract,stop_file=None):
        import psutil
        self.process=psutil.Process();self.root=root;self.contract=contract;self.limits=contract['resource_limits']
        self.started=time.monotonic();self.calls=0;self.forward_seconds=0.;self.peak_rss=0;self.peak_cuda=0;self.interrupted=False
        self.stop_file=Path(stop_file) if stop_file else None
        if self.stop_file:safe_target(root,self.stop_file.relative_to(root).as_posix())
    def check(self):
        if self.interrupted or self.stop_file and self.stop_file.exists():raise StopRequested('Explicit interrupt/stop file')
        rss=self.process.memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
        if rss>self.limits['rss_bytes']:raise ResourceStop('Process RSS ceiling')
        if time.monotonic()-self.started>self.limits['wall_seconds']:raise ResourceStop('Invocation wall ceiling')
        if shutil.disk_usage(self.root).free<self.limits['min_free_disk_bytes']:raise ResourceStop('Free disk reserve')
        if self.contract['device']=='cuda:0':
            reserved=torch.cuda.memory_reserved();self.peak_cuda=max(self.peak_cuda,reserved)
            if reserved>self.limits['cuda_reserved_bytes']:raise ResourceStop('CUDA reserved-memory ceiling')
    def forward(self,function,*args):
        self.check();start=time.monotonic();self.calls+=1
        if self.contract['device']=='cuda:0':torch.cuda.synchronize()
        result=function(*args)
        if self.contract['device']=='cuda:0':torch.cuda.synchronize()
        elapsed=time.monotonic()-start;self.forward_seconds+=elapsed;self.check()
        if elapsed>self.limits['forward_seconds']:raise ResourceStop('Single forward time ceiling')
        return result
    def snapshot(self):
        return {'forward_calls':self.calls,'forward_seconds':self.forward_seconds,'elapsed_seconds':time.monotonic()-self.started,
                'observed_peak_rss_bytes':self.peak_rss,'observed_peak_cuda_reserved_bytes':self.peak_cuda,
                'enforcement':'cooperative before/after forwards; native calls are not hard-preempted'}
    @contextmanager
    def signals(self):
        old={}
        def stop(signum,frame):self.interrupted=True
        for sig in (signal.SIGINT,signal.SIGTERM):old[sig]=signal.signal(sig,stop)
        try:yield
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)

class GuardedSession(StreamingSession):
    def forward(self,x,y):return self.guard.forward(super().forward,x,y)

class FileTaskWriter:
    """Caller holds task lock; reserve attempt before compute, stream commit."""
    def __init__(self,registry,task_id):
        self.registry=registry;self.task=registry.tasks[task_id];self.task_id=task_id
        state=registry.resume(task_id)
        if state['status']=='COMPLETE':raise ValueError('Completed task cannot reserve another attempt')
        folder=registry.path('tasks/'+task_id)
        existing=sorted(p.name for p in folder.iterdir()) if folder.exists() else []
        if existing!=[f'attempt_{i:03d}' for i in range(1,len(existing)+1)]:raise ValueError('Attempt names/gaps/extraneous files require explicit review')
        if any(not (folder/name).is_dir() for name in existing):raise ValueError('Attempt is not a directory')
        self.number=len(existing)+1
        if self.number>registry.book['technical_attempt_limit']:raise ResourceStop('Persistent three-attempt ceiling')
        self.prefix='tasks/'+task_id+f'/attempt_{self.number:03d}/'
        self.directory=registry.path(self.prefix.rstrip('/'));self.directory.mkdir(parents=True,exist_ok=False)
        self.put('STARTED.json',{'task_id':task_id,'attempt':self.number,'created_utc':utc(),'status':'INCOMPLETE'})
    def put(self,name,value):self.registry.atomic_new(self.prefix+name,value if isinstance(value,bytes) else canonical(value))
    def finish(self,status,limit):
        if status not in {'COMPLETE','FAILED'}:raise ValueError('Invalid final task status')
        if status=='COMPLETE':
            v=read_json(self.directory/'result.json');t=self.task
            if (v.get('schema')!='dynavelo_task_result_v1' or v.get('task_id')!=self.task_id or v.get('status')!='COMPLETE'
                or v.get('numerical_integrity')!='PASS' or v.get('n_cells')!=t['n_cells'] or v.get('mode')!=t['mode']
                or v.get('sample_cell_axis_sha256')!=t['sample_cell_axis_sha256']):raise ValueError('Result fails registry contract')
            if t['stage']=='screen' and (type(v.get('primary_effect')) not in {int,float} or not np.isfinite(v['primary_effect'])):raise ValueError('Finite screen effect required')
        files=[];total=0
        for path in sorted(self.directory.rglob('*')):
            if not path.is_file():continue
            name=path.relative_to(self.directory).as_posix();safe_target(self.directory,name)
            if name in {'COMMIT.json','COMPLETE.json'}:raise ValueError('Unexpected pre-existing commit')
            # Windows FlushFileBuffers requires a write-capable owned handle.
            with path.open('r+b') as stream:stream.flush();os.fsync(stream.fileno())
            size=path.stat().st_size;total+=size;files.append({'name':name,'bytes':size,'sha256':sha256(path)})
        if status=='COMPLETE' and total>limit:raise ResourceStop('Task payload ceiling')
        manifest={'schema':'dynavelo_task_commit_v1','task':self.task,'book_sha256':self.registry.book['book_sha256'],
                  'attempt':self.number,'status':status,'files':files,'created_utc':utc()}
        self.registry.atomic_new(self.prefix+'COMMIT.json',canonical(manifest))
        if status=='COMPLETE':
            path=self.directory/'COMMIT.json'
            self.registry.atomic_new('tasks/'+self.task_id+'/COMPLETE.json',canonical({'manifest_relative_path':self.prefix+'COMMIT.json','bytes':path.stat().st_size,'sha256':sha256(path)}))
            return self.registry.resume(self.task_id)
        return {'task_id':self.task_id,'status':'FAILED','attempt':self.number}

def attach_baseline(model,population,indices,task,directory,guard):
    """Read-only restore of core cache; all source/axis/batch/hash keys required."""
    obj=object.__new__(GuardedSession);obj.model=model;obj.population=population;obj.guard=guard;obj._check_mode()
    obj.indices=np.asarray(indices,dtype=np.int64);obj.indices.flags.writeable=False
    obj.batches=batches_without_singleton(len(indices),task['chunk_geometry']['cells']);obj.directory=Path(directory)
    obj.fingerprint=model_fingerprint(model);manifest=read_json(obj.directory/'baseline_manifest.json')
    expected={'schema':'dynavelo_stream_baseline_v1','mode':'evaluation-fixed','population':population.identity,
        'selected_indices':indices,'batches':obj.batches,'checkpoint_sha256':task['checkpoint_sha256'],
        'model_state_sha256':obj.fingerprint,'torch':torch.__version__,'numpy':np.__version__,
        'minima_scope':'full supplied Population, never cell subset or batch'}
    for key,value in expected.items():
        if canonical(manifest.get(key))!=canonical(value):raise ValueError('Matched baseline differs: '+key)
    obj.identity=manifest;obj.baseline={}
    shapes={'vx':(len(indices),model.x_dim),'vy':(len(indices),model.y_dim),'vz':(len(indices),model.zx_dim+model.zy_dim),'time':(len(indices),)}
    for name,shape in shapes.items():
        value=np.load(obj.directory/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if value.shape!=shape or value.dtype!=np.float32:raise ValueError('Baseline shape/dtype differs')
        for start in range(0,len(value),256):
            if not np.isfinite(value[start:start+256]).all():raise ValueError('Nonfinite baseline cache')
        obj.baseline[name]=value
    obj._check_ready();return obj

def completed_directory(registry,task):
    saved=registry.resume(task['task_id'])
    if saved['status']!='COMPLETE':raise PermissionError('Required baseline not complete')
    return registry.path(saved['manifest_relative_path']).parent,saved

def sample_ids(plan,frame,task):
    if task['sample_id']=='full_population':return [c['cell_id'] for c in plan['spec']['cells']]
    if task['sample_id']=='full_common':return frame['common_eligible_cell_ids']
    return [c['cell_id'] for c in frame['samples'][task['sample_id']]['cells']]

def full_frame_from_registry(plan,registry,root):
    summaries=[]
    for task in registry.book['tasks']:
        directory,_=completed_directory(registry,task);path=directory/'baseline_summary.json'
        summaries.append({**read_json(path),'artifact':record(root,path)})
    return freeze_baseline_frame(plan,summaries)

def _result(task,authorization):
    return {'schema':'dynavelo_task_result_v1','task_id':task['task_id'],'status':'COMPLETE','numerical_integrity':'PASS',
        'n_cells':task['n_cells'],'mode':task['mode'],'sample_cell_axis_sha256':task['sample_cell_axis_sha256'],
        'stage':task['stage'],'seed':task['seed'],'sample_id':task['sample_id'],'plan_sha256':task['plan_sha256'],
        'frame_sha256':task['frame_sha256'],'checkpoint_sha256':task['checkpoint_sha256'],'review_sha256':authorization['review_sha256'],
        'claim_status':'EXPLORATORY_ONLY_NOT_CONFIRMED','biological_or_external_PASS':False}

def baseline_task(plan,frame,task,session,writer,authorization):
    session.build_baseline();session._check_ready()
    for block in session.perturbations([session.population.genes[0]],mode='no-op'):
        if any(np.count_nonzero(value) for value in block['delta'].values()):raise ValueError('No-op not exactly zero')
    h=[session.population.gene_index[name] for name in plan['policy']['primary']['H']]
    norms=np.empty(len(session.indices),dtype=np.float64)
    for start,stop in session.batches:
        values=np.asarray(session.baseline['vx'][start:stop],dtype=np.float64)[:,h];norms[start:stop]=np.linalg.norm(values,axis=1)
    if task['stage']=='matched-baseline':
        scale=frame['global_time_windows'][str(task['seed'])]['common_frame_velocity_scale']
        if not np.isfinite(norms).all() or np.any(norms<1e-8*scale) or np.any(norms<=0):raise ValueError('Matched direction invalid; no post-effect remasking')
    result=_result(task,authorization);result.update(no_op='EXACT_ZERO_ALL_CHANNELS_ALL_CHUNKS',
        baseline_manifest_sha256=sha256(session.directory/'baseline_manifest.json'),H=plan['policy']['primary']['H'])
    if task['stage']=='baseline':writer.put('baseline_summary.json',{
        'schema':'dynavelo_full_baseline_summary_v1','seed':task['seed'],'plan_sha256':plan['plan_sha256'],
        'checkpoint_sha256':task['checkpoint_sha256'],'cell_axis_sha256':plan['axes_sha256']['cells'],
        'H':plan['policy']['primary']['H'],'raw_time_mean':session.baseline['time'].astype(float).tolist(),'direction_norm_H':norms.tolist()})
    return result

def summarize_effects(plan,frame,task,values,weights,indices):
    labels=np.array([plan['spec']['cells'][i]['label'] for i in indices],dtype=object)
    all_labels={c['cell_id']:c['label'] for c in plan['spec']['cells']};eligible=set(frame['common_eligible_cell_ids'])
    windows=np.array(frame['global_time_windows'][str(task['seed'])]['window_index'])
    full_indices=[i for i,c in enumerate(plan['spec']['cells']) if c['cell_id'] in eligible]
    def domain(mask,population_n):
        mass=float(weights[mask].sum())
        return {'sample_n':int(mask.sum()),'eligible_frame_domain_n':int(population_n),'sample_weight_sum':mass,
            'weighted_conditional_mean':{key:float(np.dot(weights[mask],value[mask])/mass) if mass>0 else None for key,value in values.items()},
            'estimator':'normalized sample-weight domain description; no biological replication or P value'}
    return {'primary_effect':float(np.dot(weights,values['projection_H'])),
        'weighted_global_secondary':{k:float(np.dot(weights,v)) for k,v in values.items() if k!='projection_H'},
        'by_label':{label:domain(labels==label,sum(all_labels[c]==label for c in eligible)) for label in sorted(set(all_labels.values()))},
        'by_predeclared_branch':{name:domain(np.isin(labels,states),sum(all_labels[c] in states for c in eligible)) for name,states in plan['policy']['branches'].items()},
        'by_full_population_time_window':{str(w):domain(windows[indices]==w,sum(windows[i]==w for i in full_indices)) for w in range(5)}}

def screen_task(plan,frame,task,session,writer,authorization,dependency):
    if task['mode']!='rna-only' or task['input_space']!='rna':raise ValueError('Only uniform RNA-only screen implemented')
    sample=frame['samples'][task['sample_id']];ids=[row['cell_id'] for row in sample['cells']]
    weights=np.asarray([row['cell_weight'] for row in sample['cells']],dtype=np.float64)
    if abs(weights.sum()-1)>1e-12:raise ValueError('Eligible-frame weights do not sum to one')
    h=[session.population.gene_index[name] for name in plan['policy']['primary']['H']]
    scale=frame['global_time_windows'][str(task['seed'])]['common_frame_velocity_scale']
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid full common-frame RMS')
    values={key:np.full(len(ids),np.nan,dtype=np.float64) for key in ('projection_H','delta_time_raw','delta_vx_norm','delta_vy_norm','delta_vz_norm')}
    visited=np.zeros(len(ids),dtype=bool)
    for block in session.perturbations([task['feature_id']],mode='rna-only'):
        start,stop=block['start'],block['stop']
        if (visited[start:stop].any() or block['motif_changed'] or block['regulator']!=task['feature_id']
            or not np.array_equal(block['population_indices'],session.indices[start:stop])):raise ValueError('Block axis/mode/coverage mismatch')
        baseline=np.asarray(session.baseline['vx'][start:stop],dtype=np.float64)[:,h];norms=np.linalg.norm(baseline,axis=1)
        if np.any(norms<1e-8*scale) or np.any(norms<=0):raise ValueError('Matched direction invalid; no TF-specific exclusion')
        values['projection_H'][start:stop]=np.sum(block['delta']['vx'][:,h]*(baseline/norms[:,None]),axis=1)/scale
        values['delta_time_raw'][start:stop]=block['delta']['time']
        for channel in ('vx','vy','vz'):values['delta_'+channel+'_norm'][start:stop]=np.linalg.norm(block['delta'][channel],axis=1)
        visited[start:stop]=True
    if not visited.all() or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Incomplete/nonfinite cell effects')
    session._check_ready();result=_result(task,authorization)
    result.update(summarize_effects(plan,frame,task,values,weights,session.indices))
    result.update(tf_stable_id=task['tf_stable_id'],feature_id=task['feature_id'],full_common_frame_RMS=scale,
        H=plan['policy']['primary']['H'],matched_baseline_dependency=dependency,
        time='raw posterior-mean difference; bins from full original baseline only',
        effect='along current matched model velocity; not rescue, causality or elapsed time')
    stream=io.StringIO(newline='');fields=['cell_id','population_index','label','cell_weight','full_population_window']+list(values)
    writer_csv=csv.DictWriter(stream,fieldnames=fields,delimiter='\t');writer_csv.writeheader()
    windows=frame['global_time_windows'][str(task['seed'])]['window_index']
    for i,(cell,position) in enumerate(zip(ids,session.indices)):
        writer_csv.writerow({'cell_id':cell,'population_index':int(position),'label':sample['cells'][i]['label'],
            'cell_weight':float(weights[i]),'full_population_window':windows[position],**{k:float(v[i]) for k,v in values.items()}})
    writer.put('cell_effects.tsv',stream.getvalue().encode('utf-8'));return result

def load_model(root,plan,model_record,population,contract):
    """Restore exact source-pinned canonical best weights; never fit a model."""
    config=read_json(verify_artifact(root,model_record['config']));metrics=read_json(verify_artifact(root,model_record['metrics']))
    if (config.get('run_id')!=model_record['run_id'] or config.get('training_seed')!=model_record['seed']
        or config.get('analytic_velocity') is not False or metrics.get('run_id')!=model_record['run_id']
        or metrics.get('best_checkpoint_sha256')!=model_record['checkpoint']['sha256'] or metrics.get('best_checkpoint_reloaded') is not True
        or metrics.get('analytic_velocity') is not False or 'BENCHMARK' in config.get('status','') or 'BENCHMARK' in metrics.get('status','')):
        raise ValueError('Config/metrics do not identify canonical-best non-benchmark run')
    if (config.get('n_cells'),config.get('n_genes'),config.get('n_TF'))!=(len(population.cell_ids),len(population.genes),len(population.motif_tfs)):
        raise ValueError('Training/input dimensions differ')
    artifacts=plan['spec']['input_artifacts']
    if sorted(config.get('input_files',{}).values())!=sorted(artifacts[role]['sha256'] for role in ('rna','atac')):
        raise ValueError('Saved training input SHA values differ from inference files')
    for role in ('upstream_models','upstream_utils','device_adapter'):verify_artifact(root,artifacts[role])
    if any(config['adapter_manifest']['upstream_'+name+'_sha256']!=artifacts['upstream_'+name]['sha256'] for name in ('models','utils')):
        raise ValueError('Source differs from saved training provenance')
    if contract['device']=='cuda:0':
        from dynavelo_colab_adapter import load_dynavelo
    else:
        sys.path.insert(0,str(root/'repo/pipeline'))
        from dynavelo_cpu_adapter import load_dynavelo
    _,Model,adapter=load_dynavelo(root,analytic_velocity=False)
    if adapter['adapter_sha256']!=artifacts['device_adapter']['sha256'] or adapter['analytic_velocity'] is not False:raise ValueError('Unreviewed execution adapter')
    if any(adapter['upstream_'+name+'_sha256']!=artifacts['upstream_'+name]['sha256'] for name in ('models','utils')):
        raise ValueError('Actually imported model source differs from frozen source')
    defaults=config['model_defaults']
    expected={'num_hidden','zx_dim','zy_dim','k_t','k_z0','k_velocity','k_consistency','sigma_x','sigma_y','mu_pz0','sigma_pz0','alpha_pt','beta_pt','time_variant'}
    if set(defaults)!=expected:raise ValueError('Unsupported/incomplete model architecture')
    model=Model(x_dim=len(population.genes),y_dim=len(population.motif_tfs),**defaults,seed=model_record['seed'],device=contract['device'],
        mode='evaluation-fixed',dataset_name='screen_inference',sample_name=plan['plan_sha256'][:16]+'_seed'+str(model_record['seed']))
    state=torch.load(verify_artifact(root,model_record['checkpoint']),map_location='cpu',weights_only=True)['model_state_dict']
    model.load_state_dict(state,strict=True);model.eval();model.mode='evaluation-fixed'
    if any(not torch.equal(value.detach().cpu(),state[key]) or not torch.isfinite(value).all() for key,value in model.state_dict().items()):
        raise ValueError('Reloaded state differs or is nonfinite')
    return model

def dispatch(plan,review_record,root,stage,output_root,*,frame=None,trusted_reviewers=(),shard_index=None,
             seed=None,max_tasks=None,stop_file=None,fixture_population=None,fixture_model_loader=None):
    """Run a bounded invocation; failed/incomplete tasks stay in the family."""
    root=checked_root(root);output_root=validate_output_namespace(plan,root,output_root)
    if stage not in SUPPORTED:raise NotImplementedError('GRN/joint/full-cell dispatch unimplemented')
    validate_plan(plan)
    authorization=authorize_stage(plan,review_record,root,stage,trusted_reviewers=trusted_reviewers,frame=frame)
    contract=validate_contract(plan,root,review_record);fixture=plan['spec']['purpose']=='engineering_fixture'
    if not fixture and (fixture_population is not None or fixture_model_loader is not None):raise PermissionError('Injected model/data forbidden for human plans')
    guard=ResourceGuard(root,contract,stop_file);guard.check();maximum=contract['resource_limits']['max_tasks_per_invocation']
    if max_tasks is None:max_tasks=maximum
    if type(max_tasks) is not int or not 1<=max_tasks<=maximum:raise ValueError('Task limit exceeds reviewed ceiling')
    if seed is not None and seed not in (0,1,2):raise ValueError('Invalid seed selector')
    n_shards=plan['spec']['chunk_geometry'].get('regulator_shards',1)
    if shard_index is not None and (type(shard_index) is not int or not 0<=shard_index<n_shards):raise ValueError('Invalid frozen shard selector')
    population=fixture_population if fixture_population is not None else load_population(plan,root,contract)
    verify_population(plan,population);torch.set_num_threads(contract['resource_limits']['cpu_threads'])
    registry=TaskRegistry(output_root/stage,task_book(plan,stage,frame),plan=plan,frame=frame)
    if stage!='baseline':
        full_registry=TaskRegistry(output_root/'baseline',task_book(plan,'baseline'),plan=plan)
        if canonical(full_frame_from_registry(plan,full_registry,root))!=canonical(frame):raise PermissionError('Frame differs from complete baseline registry')
    matched_registry=None
    if stage=='screen':matched_registry=TaskRegistry(output_root/'matched-baseline',task_book(plan,'matched-baseline',frame),plan=plan,frame=frame)
    tasks=[t for t in registry.book['tasks'] if (seed is None or t['seed']==seed) and (shard_index is None or t['assigned_shard']==shard_index)]
    attempted=0;current_seed=None;model=None;restored_noop_checked=set();status='INVOCATION_FINISHED_NOT_A_SCIENTIFIC_PASS';error=None
    def recheck():
        authorize_stage(plan,review_record,root,stage,trusted_reviewers=trusted_reviewers,frame=frame)
        validate_contract(plan,root,review_record);verify_population(plan,population);guard.check()
    with guard.signals():
        try:
            for task in tasks:
                guard.check()
                existing=registry.resume(task['task_id'])
                if existing['status']=='COMPLETE':
                    if stage=='screen':
                        dependencies=[t for t in matched_registry.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
                        if len(dependencies)!=1:raise ValueError('Completed screen baseline dependency not unique')
                        _,saved_dependency=completed_directory(matched_registry,dependencies[0])
                        saved_result=read_json(registry.path(existing['manifest_relative_path']).parent/'result.json')
                        expected_dependency={'task_id':dependencies[0]['task_id'],'commit_sha256':saved_dependency['manifest_sha256'],'book_sha256':matched_registry.book['book_sha256']}
                        if saved_result.get('matched_baseline_dependency')!=expected_dependency:raise ValueError('Completed screen dependency changed')
                    continue
                if attempted>=max_tasks:status='STOPPED_AT_FROZEN_TASK_LIMIT';break
                with registry.locked(task['task_id']):
                    if registry.resume(task['task_id'])['status']=='COMPLETE':continue
                    recheck();writer=FileTaskWriter(registry,task['task_id']);attempted+=1
                    try:
                        model_record=next(m for m in plan['spec']['models'] if m['seed']==task['seed'])
                        if current_seed!=task['seed']:
                            model=None
                            if contract['device']=='cuda:0':torch.cuda.empty_cache()
                            model=(fixture_model_loader or load_model)(root,plan,model_record,population,contract);current_seed=task['seed']
                        model.eval();model.mode='evaluation-fixed'
                        if str(model.device)!=contract['device'] or any(str(p.device)!=contract['device'] for p in model.parameters()):raise ValueError('Loaded device differs')
                        ids=sample_ids(plan,frame,task);positions={c:i for i,c in enumerate(population.cell_ids)};indices=[positions[c] for c in ids]
                        if identity(ids)!=task['sample_cell_axis_sha256'] or len(ids)!=task['n_cells']:raise ValueError('Task cell axis changed')
                        if canonical(task['batches'])!=canonical(batches_without_singleton(len(ids),task['chunk_geometry']['cells'])):raise ValueError('Task/core batches differ')
                        if stage in {'baseline','matched-baseline'}:
                            session=GuardedSession(model,population,indices,task['chunk_geometry']['cells'],writer.directory/'baseline',checkpoint_sha256=task['checkpoint_sha256']);session.guard=guard
                            width=model.x_dim+model.y_dim+model.zx_dim+model.zy_dim+1
                            if len(ids)*width*4+1024**2>contract['resource_limits']['task_payload_bytes']:raise ResourceStop('Projected payload ceiling')
                            result=baseline_task(plan,frame,task,session,writer,authorization)
                        else:
                            deps=[t for t in matched_registry.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
                            if len(deps)!=1:raise ValueError('Matched baseline task not unique')
                            dependency=deps[0];directory,saved=completed_directory(matched_registry,dependency)
                            for key in ('checkpoint_sha256','sample_cell_axis_sha256','n_cells','chunk_geometry','batches','full_global_minima_sha256'):
                                if canonical(task[key])!=canonical(dependency[key]):raise ValueError('Matched source/chunks differ')
                            session=attach_baseline(model,population,indices,task,directory/'baseline',guard)
                            cache_key=(task['seed'],task['sample_id'],saved['manifest_sha256'],session.fingerprint)
                            if cache_key not in restored_noop_checked:
                                for block in session.perturbations([population.genes[0]],mode='no-op'):
                                    if any(np.count_nonzero(v) for v in block['delta'].values()):
                                        raise ValueError('Restored matched cache no-op differs; no cached numerical drift may count as an effect')
                                restored_noop_checked.add(cache_key)
                            result=screen_task(plan,frame,task,session,writer,authorization,{'task_id':dependency['task_id'],'commit_sha256':saved['manifest_sha256'],'book_sha256':matched_registry.book['book_sha256']})
                            result['restored_matched_cache_noop']='EXACT_ZERO_ALL_CHANNELS_CHECKED_THIS_INVOCATION'
                            if matched_registry.resume(dependency['task_id'])!=saved:raise ValueError('Matched cache changed during screen')
                        session._check_ready();recheck()
                        result.update(resource_snapshot=guard.snapshot(),dispatcher_sha256=sha256(__file__),engineering_fixture=fixture)
                        writer.put('result.json',result);writer.finish('COMPLETE',contract['resource_limits']['task_payload_bytes']);del session
                    except (StopRequested,KeyboardInterrupt):
                        if not (writer.directory/'COMMIT.json').exists():writer.put('INTERRUPTED.json',{'status':'INCOMPLETE','time':utc(),'resources':guard.snapshot()})
                        raise
                    except Exception as exc:
                        if not (writer.directory/'COMMIT.json').exists():
                            writer.put('FAILURE.json',{'error_type':type(exc).__name__,'error':str(exc),'time':utc(),'resources':guard.snapshot()});writer.finish('FAILED',float('inf'))
                        raise
        except (StopRequested,KeyboardInterrupt) as exc:status='INTERRUPTED_INCOMPLETE_TASK_RETAINED';error=str(exc)
        except Exception as exc:status='STOPPED_WITH_FAILURE_ALL_TASKS_RETAINED';error=type(exc).__name__+': '+str(exc)
    result={'schema':'dynavelo_dispatch_invocation_v1','status':status,'error':error,'stage':stage,'plan_sha256':plan['plan_sha256'],
        'frame_sha256':frame['frame_sha256'] if frame else None,'tasks_attempted':attempted,'resources':guard.snapshot(),
        'coverage':registry.coverage(),'scientific_or_external_PASS':False,'engineering_fixture':fixture}
    if stage=='baseline' and result['coverage']['all_expected_tasks_complete']:
        derived=full_frame_from_registry(plan,registry,root);path=output_root/'baseline_frame.json';data=canonical(derived)
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing baseline frame changed')
        if not path.exists():
            temporary=path.with_name('.part_'+uuid.uuid4().hex)
            try:
                with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
                os.link(temporary,path)
            finally:
                if temporary.exists():temporary.unlink()
        result.update(baseline_frame_status=derived['status'],baseline_frame_path=str(path))
    registry.atomic_new('invocations/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]+'.json',canonical(result))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--review-record',required=True)
    parser.add_argument('--trusted-reviewer',action='append',required=True);parser.add_argument('--stage',choices=sorted(SUPPORTED),required=True)
    parser.add_argument('--frame');parser.add_argument('--output-root',required=True);parser.add_argument('--shard-index',type=int)
    parser.add_argument('--seed',type=int,choices=[0,1,2]);parser.add_argument('--max-tasks',type=int);parser.add_argument('--stop-file')
    args=parser.parse_args();root=checked_root(args.root)
    def local(path):
        value=Path(path)
        if not value.is_absolute():raise ValueError('Absolute project path required')
        return safe_target(root,value.relative_to(root).as_posix())
    result=dispatch(read_json(local(args.plan)),read_json(local(args.review_record)),root,args.stage,local(args.output_root),
        frame=read_json(local(args.frame)) if args.frame else None,trusted_reviewers=args.trusted_reviewer,
        shard_index=args.shard_index,seed=args.seed,max_tasks=args.max_tasks,stop_file=local(args.stop_file) if args.stop_file else None)
    print(json.dumps({k:v for k,v in result.items() if k!='coverage'},ensure_ascii=False));return 1 if result['error'] else 0

if __name__=='__main__':raise SystemExit(main())
