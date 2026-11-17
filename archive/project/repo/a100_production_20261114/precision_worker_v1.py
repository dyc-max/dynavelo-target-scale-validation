"""Isolated FP64 registry and bounded worker; no original FP32 result writes.

Mathematical definitions come only from precision_plan_v1.build_books. Reuse
original immutable COMMITs, locks, three-attempt cap and reserved stream writer.
"""
from pathlib import Path
from contextlib import contextmanager,ExitStack
import argparse,copy,json,os,random,re,sys,time,traceback
for _thread_env in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_env]='1'
import numpy as np
SELF=Path(__file__).resolve()
ROOT=next((p for p in (Path('/content/DynaVelo_Research'),Path('/mnt/d/dyc_7/DynaVelo_Research'),Path(r'D:\dyc_7\DynaVelo_Research')) if SELF.is_relative_to(p)),None)
if ROOT is None:raise ValueError('Worker source outside project')
sys.path[:0]=[str(ROOT/'repo/colab'),str(ROOT/'repo/a100_production_20261114'),str(ROOT/'repo/posttraining_optimization_20261114')]
import screen_dispatch as base
from screen_plan import canonical,identity,check_seal
from screen_task_registry import TaskRegistry,validate_book,batches
from human_return_common import safe_target
from recovery_io_v1 import read,checked,record,new,now,sha
EXPERIMENT='FP64_PROMOTED_FROZEN_CHECKPOINT_INPUT_SENSITIVITY_ONLY'
NAMESPACE='results/precision_runs_v1/fp64_20261115_v1'

def validate_runtime_packages(contract):
    """Exact original package pins before any model load; no implicit upgrades."""
    import torch
    from importlib.metadata import version
    expected=contract['package_versions']
    required={'torch','numpy','anndata','h5py','psutil','torchdiffeq'}
    if set(expected)!=required:raise ValueError('Complete original runtime package version pins required')
    actual={'torch':str(torch.__version__),'numpy':str(np.__version__)}
    actual.update({name:version(name) for name in sorted(required-{'torch','numpy'})})
    mismatches={name:{'expected':expected[name],'actual':actual[name]} for name in sorted(required) if actual[name]!=expected[name]}
    if mismatches:raise RuntimeError('Precision runtime package version mismatch: '+json.dumps(mismatches,sort_keys=True))
    return actual

def build_books(precision_plan,original_scope_plan,frame,shortlist):
    from precision_plan_v1 import build_books as builder
    return builder(precision_plan,original_scope_plan,frame,shortlist)

class PrecisionRegistry(TaskRegistry):
    def __init__(self,root,book,*,precision_plan,original_scope_plan,frame,shortlist):
        validate_book(book)
        all_books=build_books(precision_plan,original_scope_plan,frame,shortlist)
        if canonical(all_books[book['stage']])!=canonical(book):raise ValueError('Book differs from complete precision family')
        self.root=Path(root).absolute()
        allowed=(Path('/content/DynaVelo_Research'),Path('/mnt/d/dyc_7/DynaVelo_Research'),Path(r'D:\dyc_7\DynaVelo_Research'))
        self.project_root=next((p for p in allowed if self.root.is_relative_to(p)),None)
        if self.project_root is None or self.root.resolve()!=self.root:raise ValueError('Unauthorized registry root')
        if precision_plan['namespace_relative']!=NAMESPACE or self.root!=safe_target(self.project_root,NAMESPACE+'/'+book['stage']):raise ValueError('Unique precision namespace required')
        if book['plan_sha256']!=precision_plan['precision_plan_sha256'] or book['frame_sha256']!=frame['scope_frame_sha256']:raise ValueError('Book plan/frame differs')
        self.plan=self.precision_plan=precision_plan;self.original_scope_plan=original_scope_plan
        self.frame,self.shortlist=frame,shortlist;self.book=book;self.tasks={t['task_id']:t for t in book['tasks']}
        self._all_books=all_books;self._baseline=None;self._parent_cache=None;self._audit_checkpoint=None
        self.root.mkdir(parents=True,exist_ok=True);p=self.path('task_book.json');data=canonical(book)
        if p.exists():
            if p.read_bytes()!=data:raise ValueError('Immutable precision book changed')
        else:self.atomic_new('task_book.json',data)

    @contextmanager
    def dependency_audit(self,checkpoint=None):
        if self._parent_cache is not None:
            if checkpoint:checkpoint()
            yield;return
        self._parent_cache={};self._audit_checkpoint=checkpoint
        try:
            yield
            for key,saved in self._parent_cache.items():
                if checkpoint:checkpoint()
                if self._baseline.resume(key)!=saved:raise ValueError('Parent baseline changed during audit')
        finally:self._parent_cache=None;self._audit_checkpoint=None

    def dependency(self,task):
        parent_id=task.get('parent_baseline_task_id')
        if task['stage']=='baseline':
            if parent_id is not None:raise ValueError('Baseline cannot have parent')
            return None
        if not parent_id:raise PermissionError('Exact precision parent required')
        if self._baseline is None:
            self._baseline=PrecisionRegistry(self.root.parent/'baseline',self._all_books['baseline'],precision_plan=self.precision_plan,original_scope_plan=self.original_scope_plan,frame=self.frame,shortlist=self.shortlist)
        if parent_id not in self._baseline.tasks:raise ValueError('Parent outside precision baseline family')
        parent=self._baseline.tasks[parent_id]
        for key in ('seed','checkpoint_sha256','sample_id','sample_cell_ids','sample_cell_axis_sha256','n_cells','batches','chunk_geometry'):
            if canonical(parent[key])!=canonical(task[key]):raise ValueError('Parent geometry mismatch: '+key)
        saved=self._parent_cache.get(parent_id) if self._parent_cache is not None else None
        if saved is None:
            saved=self._baseline.resume(parent_id)
            if saved['status']!='COMPLETE':raise PermissionError('Complete exact FP64 baseline required')
            if self._parent_cache is not None:self._parent_cache[parent_id]=saved
        dep={'task_id':parent_id,'commit_sha256':saved['manifest_sha256'],'book_sha256':self._baseline.book['book_sha256']}
        return parent,self._baseline.path(saved['manifest_relative_path']).parent,saved,dep

    def validate_result(self,task,value):
        expected={'schema':'dynavelo_task_result_v1','task_id':task['task_id'],'status':'COMPLETE','numerical_integrity':'PASS',
            'n_cells':task['n_cells'],'mode':task['mode'],'sample_cell_axis_sha256':task['sample_cell_axis_sha256'],
            'plan_sha256':self.precision_plan['precision_plan_sha256'],'precision_plan_sha256':self.precision_plan['precision_plan_sha256'],
            'frame_sha256':self.frame['scope_frame_sha256'],'parent_baseline_task_id':task.get('parent_baseline_task_id'),
            'precision_experiment':EXPERIMENT,'precision_dtype':'float64','biological_or_external_PASS':False,'scientific_pass':False}
        for key in ('seed','stage','sample_id','checkpoint_sha256'):
            expected[key]=task[key]
        if any(canonical(value.get(k))!=canonical(v) for k,v in expected.items()):raise ValueError('Result escaped exact FP64 task/context')
        for key in ('precision_config_sha256','precision_config'):
            if key in task and canonical(value.get(key))!=canonical(task[key]):raise ValueError('Precision configuration changed')
        if task['stage']=='baseline':
            if value.get('parent_baseline_dependency') is not None or value.get('no_op')!='EXACT_ZERO_ALL_CHANNELS_ALL_CHUNKS':raise ValueError('Baseline no-op/dependency invalid')
        else:
            dep=self.dependency(task)[3]
            if value.get('parent_baseline_dependency')!=dep or value.get('matched_baseline_dependency')!=dep:raise ValueError('Parent COMMIT binding changed')

    def resume(self,task_id):
        if self._audit_checkpoint:self._audit_checkpoint()
        saved=TaskRegistry.resume(self,task_id)
        if saved['status']=='COMPLETE':
            directory=self.path(saved['manifest_relative_path']).parent;value=read(directory/'result.json')
            self.validate_result(self.tasks[task_id],value)
            if self.book['stage']=='baseline' and value.get('baseline_manifest_sha256')!=sha(directory/'baseline/baseline_manifest.json'):raise ValueError('Baseline manifest binding differs')
        return saved

    def coverage(self,checkpoint=None):
        with self.dependency_audit(checkpoint):return TaskRegistry.coverage(self)

    def commit(self,*args,**kwargs):raise NotImplementedError('Use reserved PrecisionTaskWriter')

class PrecisionTaskWriter(base.FileTaskWriter):
    def finish(self,status,limit):
        if status=='COMPLETE':self.registry.validate_result(self.task,read(self.directory/'result.json'))
        return super().finish(status,limit)

def select_tasks(book,ids):
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or len(ids)>256:raise ValueError('Explicit unique bounded selection required')
    selected=[t for t in book['tasks'] if t['task_id'] in set(ids)]
    if [t['task_id'] for t in selected]!=ids or len({t['seed'] for t in selected})!=1:raise ValueError('Selection requires exact book order and one seed')
    return selected

class PrecisionGuard(base.ResourceGuard):
    def __init__(self,root,contract,stop_file,deadline):
        super().__init__(root,contract,stop_file);self.deadline=float(deadline)
    def check(self):
        super().check()
        import psutil
        if time.time()>=self.deadline:raise base.ResourceStop('Precision plan deadline')
        if psutil.virtual_memory().available<16*1024**3:raise base.ResourceStop('System free-memory floor')

def assert_grn_review(root,precision_plan,original,frame,shortlist,books,guard):
    namespace=root/precision_plan['namespace_relative']
    rec=read(namespace/'actualFP64qualification_review_record.json');review=read(checked(root,rec))
    if review.get('decision')!='APPROVED_FOR_FIXED_FP64_DESCRIPTIVE_GRN' or review.get('precision_plan_sha256')!=precision_plan['precision_plan_sha256']:raise PermissionError('Actual FP64 qualification review required')
    registry=PrecisionRegistry(namespace/'epsilon',books['epsilon'],precision_plan=precision_plan,original_scope_plan=original,frame=frame,shortlist=shortlist)
    coverage=registry.coverage(checkpoint=guard.check)
    if not coverage['all_expected_tasks_complete'] or review.get('epsilon_coverage_sha256')!=coverage['coverage_sha256']:raise PermissionError('Entire exact epsilon family must be reviewed')
    checked(root,review['epsilon_comparison']);return rec,review

def execute(plan_path,plan_sha,stage,ids_path,ids_sha,worker_id):
    import torch
    from filelock import FileLock
    from precision_inference_v1 import make_forward64,installed64
    from precision_streaming_v1 import Population64,StreamingSession64,attach_baseline64
    import screen_scope_followup_numerics_v2 as numerics
    import screen_scope_dispatch_v2 as primary
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}',worker_id):raise ValueError('Unsafe worker ID')
    output=ROOT/'checkpoints/precision_invocations'/worker_id;output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();audits=[];counts={'forward_calls':0};attempted=skipped=0;guard=None
    old_dtype=torch.get_default_dtype()
    def recheck():
        if sha(plan_path)!=plan_sha or sha(ids_path)!=ids_sha:raise ValueError('CLI input bytes changed')
        for rec in precision_plan['source_records']:checked(ROOT,rec)
        checked(ROOT,review_record);guard.check()
    try:
        if not plan_path.is_absolute() or not ids_path.is_absolute() or not plan_path.is_relative_to(ROOT) or not ids_path.is_relative_to(ROOT):raise ValueError('Absolute project paths required')
        if sha(plan_path)!=plan_sha or sha(ids_path)!=ids_sha:raise ValueError('CLI checksum mismatch')
        precision_plan=read(plan_path);check_seal(precision_plan,'precision_plan_sha256')
        original,frame,shortlist=[read(checked(ROOT,precision_plan[k])) for k in ('original_scope_plan','scope_frame','shortlist')]
        for rec in precision_plan['source_records']:checked(ROOT,rec)
        if record(ROOT,SELF) not in precision_plan['source_records']:raise ValueError('Imported worker not pinned')
        initial=read(checked(ROOT,precision_plan['review']))
        allowed=initial.get('allowed_stages',initial.get('allowed_operations',[]))
        if initial.get('decision') in (None,'HOLD','REJECTED') or stage in ('baseline','epsilon') and stage not in allowed:raise PermissionError('Explicit precision stage review required')
        if 'precision_plan_sha256' in initial and initial['precision_plan_sha256']!=precision_plan['precision_plan_sha256']:raise PermissionError('Initial review scope mismatch')
        books=build_books(precision_plan,original,frame,shortlist);book=books[stage]
        payload=read(ids_path);ids=payload['task_ids'] if isinstance(payload,dict) else payload;tasks=select_tasks(book,ids)
        seed=tasks[0]['seed'];namespace=ROOT/precision_plan['namespace_relative']
        registry=PrecisionRegistry(namespace/stage,book,precision_plan=precision_plan,original_scope_plan=original,frame=frame,shortlist=shortlist)
        context=original['context_plan'];contract=copy.deepcopy(context['spec']['dispatch_contract'])
        runtime_packages=validate_runtime_packages(contract)
        if contract['device']!='cuda:0' or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('A100 backend required')
        if torch.cuda.get_device_properties(0).total_memory<75*1024**3:raise RuntimeError('80GB A100 required')
        limits=contract['resource_limits'];limits.update(cpu_threads=1,rss_bytes=64*1024**3,cuda_reserved_bytes=64*1024**3,wall_seconds=7200,forward_seconds=120)
        deadline=precision_plan['deadline_unix'];stop=safe_target(ROOT,precision_plan['stop_relative'])
        guard=PrecisionGuard(ROOT,contract,stop,deadline);guard.check();review_record=precision_plan['review']
        if stage in ('grn','full-cell-grn'):review_record,_=assert_grn_review(ROOT,precision_plan,original,frame,shortlist,books,guard)
        auth={'review_sha256':review_record['sha256']}
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        new(output/'STARTED.json',{'created_utc':now(),'precision_plan':record(ROOT,plan_path),'task_ids':ids,'seed':seed,'gpu':torch.cuda.get_device_name(0),'runtime_package_versions':runtime_packages,'scientific_pass':False})
        torch.set_default_dtype(torch.float32)
        population32=base.load_population(context,ROOT,contract);base.verify_population(context,population32)
        population=Population64.from_float32(population32)
        model_record=next(r for r in context['spec']['models'] if r['seed']==seed)
        model=base.load_model(ROOT,context,model_record,population32,contract)
        expected={k:v.detach().to(dtype=torch.float64).cpu().clone() if v.is_floating_point() else v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.double();model.eval();model.mode='evaluation-fixed';torch.set_default_dtype(torch.float64)
        if any(not torch.equal(v.detach().cpu(),expected[k]) for k,v in model.state_dict().items()):raise ValueError('FP64 weights not exact FP32 promotion')
        del expected
        original_forward=model.forward;candidate=make_forward64(model,checked(ROOT,context['spec']['input_artifacts']['upstream_models']))
        def rng():return random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()]
        def restore(s):random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])
        def same_rng(a,b):return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2]) and len(a[3])==len(b[3]) and all(torch.equal(x,y) for x,y in zip(a[3],b[3]))
        def forward(x,y):
            index=counts['forward_calls'];counts['forward_calls']+=1;test=index==0 or index%127==0
            before=rng() if test else None;value=candidate(x,y)
            if test:
                after=rng();ref=None;wait_started=time.monotonic()
                try:
                    with FileLock(str(namespace/'gpu_original_audit.lock'),timeout=max(1,min(120,deadline-time.time()))):
                        try:
                            guard.check();wait=time.monotonic()-wait_started
                            try:restore(before);ref=original_forward(x,y);torch.cuda.synchronize();original_after=rng()
                            finally:restore(after)
                            equal=len(value)==len(ref)==9 and all(a.dtype==b.dtype and a.shape==b.shape and a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes() for a,b in zip(value,ref))
                            audit={'forward_call':index,'nine_outputs_bitwise_equal':equal,'cpu_cuda_python_numpy_rng_equal':same_rng(after,original_after),'audit_lock_wait_seconds':wait,'seed':seed,'batch_cells':len(x)}
                            audits.append(audit);new(output/('audit_%06d.json'%index),audit)
                            if not equal or not audit['cpu_cuda_python_numpy_rng_equal']:raise ValueError('OriginalFP64/candidate9output/RNG audit failed')
                        finally:
                            ref=None;torch.cuda.empty_cache()
                finally:del ref;torch.cuda.empty_cache()
            return value
        def attach_guard(session):
            raw=session.forward
            def guarded(x,y):return guard.forward(raw,x,y)
            session.forward=guarded;return session
        with ExitStack() as stack:
            stack.enter_context(installed64(model,candidate));model.forward=forward
            with guard.signals(),registry.dependency_audit(guard.check):
                for task in tasks:
                    guard.check()
                    if registry.resume(task['task_id'])['status']=='COMPLETE':skipped+=1;continue
                    dependency=registry.dependency(task)
                    with registry.locked(task['task_id']):
                        if registry.resume(task['task_id'])['status']=='COMPLETE':skipped+=1;continue
                        recheck();writer=PrecisionTaskWriter(registry,task['task_id']);attempted+=1
                        try:
                            positions={n:i for i,n in enumerate(population.cell_ids)};indices=[positions[c] for c in task['sample_cell_ids']]
                            if identity(task['sample_cell_ids'])!=task['sample_cell_axis_sha256'] or len(indices)!=task['n_cells']:raise ValueError('Task sample axis changed')
                            if canonical(task['batches'])!=canonical(batches(len(indices),task['chunk_geometry']['cells'])):raise ValueError('Batch geometry changed')
                            if stage=='baseline':
                                session=attach_guard(StreamingSession64(model,population,indices,task['chunk_geometry']['cells'],writer.directory/'baseline',checkpoint_sha256=task['checkpoint_sha256']))
                                session.build_baseline();session._check_ready();dep=None
                            else:
                                parent,directory,saved,dep=dependency
                                session=attach_guard(attach_baseline64(model,population,indices,parent,directory/'baseline',guard=guard))
                            for block in session.perturbations([population.genes[0]],mode='no-op'):
                                if any(np.count_nonzero(v) for v in block['delta'].values()):raise ValueError('FP64 exact no-op failed')
                            if stage=='baseline':
                                value=primary.result(task,auth);value.update(no_op='EXACT_ZERO_ALL_CHANNELS_ALL_CHUNKS',baseline_manifest_sha256=sha(session.directory/'baseline_manifest.json'))
                            else:
                                function=numerics.epsilon_task if stage=='epsilon' else numerics.grn_task
                                value=function(original,frame,task,session,writer,auth,dep)
                            session._check_ready();recheck()
                            value.update(schema='dynavelo_task_result_v1',plan_sha256=precision_plan['precision_plan_sha256'],frame_sha256=frame['scope_frame_sha256'],
                                precision_plan_sha256=precision_plan['precision_plan_sha256'],precision_experiment=EXPERIMENT,
                                parent_baseline_task_id=task.get('parent_baseline_task_id'),parent_baseline_dependency=dep,
                                biological_or_external_PASS=False,scientific_pass=False,resources=guard.snapshot(),worker_source=record(ROOT,SELF),
                                precision_dtype='float64',original_f32_reference_frame_frozen=True)
                            for key in ('precision_config_sha256','precision_config'):
                                if key in task:value[key]=task[key]
                            writer.put('result.json',value);writer.finish('COMPLETE',limits['task_payload_bytes']);del session
                        except BaseException as exc:
                            if not (writer.directory/'COMMIT.json').exists():
                                writer.put('FAILURE.json',{'error_type':type(exc).__name__,'error':str(exc),'created_utc':now(),'resources':guard.snapshot()});writer.finish('FAILED',float('inf'))
                            raise
        receipt={'status':'COMPLETE_PRECISION_WORKER_NOT_SCIENTIFIC_PASS','created_utc':now(),'stage':stage,'seed':seed,'task_ids':ids,
            'tasks_attempted':attempted,'completed_tasks_skipped':skipped,'audits':audits,**counts,'resources':guard.snapshot(),
            'precision_plan':record(ROOT,plan_path),'candidate_manifest':candidate.manifest,'runtime_package_versions':runtime_packages,'worker_source':record(ROOT,SELF),'scientific_pass':False}
        if attempted and not audits:raise ValueError('Original FP64 audit absent')
        new(output/'worker_receipt.json',receipt);new(output/'terminal.json',{'status':'COMPLETE','exit_code':0,'seconds':time.monotonic()-started,'scientific_pass':False});return receipt
    except BaseException as exc:
        new(output/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'audits':audits,**counts,'seconds':time.monotonic()-started})
        if not (output/'terminal.json').exists():new(output/'terminal.json',{'status':'FAILED_PRESERVE_EVIDENCE','exit_code':1,'scientific_pass':False})
        raise
    finally:torch.set_default_dtype(old_dtype)

def smoke_validate(plan_path,plan_sha,stage,ids_path,ids_sha):
    """Read/hash exact metadata and selection; no registry creation/model/forward."""
    if not plan_path.is_absolute() or not ids_path.is_absolute() or not plan_path.is_relative_to(ROOT) or not ids_path.is_relative_to(ROOT):raise ValueError('Absolute project paths required')
    if sha(plan_path)!=plan_sha or sha(ids_path)!=ids_sha:raise ValueError('CLI checksum mismatch')
    plan=read(plan_path);check_seal(plan,'precision_plan_sha256')
    for source in plan['source_records']:checked(ROOT,source)
    if record(ROOT,SELF) not in plan['source_records']:raise ValueError('Worker source not pinned')
    checked(ROOT,plan['review'])
    original,frame,shortlist=[read(checked(ROOT,plan[k])) for k in ('original_scope_plan','scope_frame','shortlist')]
    books=build_books(plan,original,frame,shortlist);payload=read(ids_path)
    selected=select_tasks(books[stage],payload['task_ids'] if isinstance(payload,dict) else payload)
    return {'status':'PASS_READ_ONLY_PRECISION_PLAN_BOOK_SELECTION','stage':stage,'seed':selected[0]['seed'],
        'selected_tasks':len(selected),'family_counts':{k:len(v['tasks']) for k,v in books.items()},'registry_writes':0,'model_forward_calls':0,
        'runtime_package_guard_required_before_execution':True,'metadata_smoke_does_not_require_GPU_runtime':True,'scientific_pass':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--stage',choices=('baseline','epsilon','grn','full-cell-grn'),required=True)
    parser.add_argument('--task-ids',required=True);parser.add_argument('--task-ids-sha',required=True);parser.add_argument('--worker-id',required=True)
    parser.add_argument('--smoke-validate',action='store_true')
    a=parser.parse_args()
    if a.smoke_validate:print(json.dumps(smoke_validate(Path(a.plan),a.plan_sha,a.stage,Path(a.task_ids),a.task_ids_sha)))
    else:execute(Path(a.plan),a.plan_sha,a.stage,Path(a.task_ids),a.task_ids_sha,a.worker_id)
