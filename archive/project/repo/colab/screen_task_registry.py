"""Immutable task identities, bounded attempts and hash-verified completion.

Storage/provenance infrastructure only; no numerical model execution.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
from human_return_common import safe_relative, safe_target, sha256
from screen_plan import canonical, identity, seal, check_seal, validate_plan, validate_frame

def batches(n,size):
    if n<2:
        raise ValueError('At least two inference cells required')
    edges=list(range(0,n,size))+[n]
    if len(edges)>2 and edges[-1]-edges[-2]==1:
        edges.pop(-2)
    return list(zip(edges[:-1],edges[1:]))

def task_book(plan,stage,frame=None):
    validate_plan(plan)
    if stage not in {'baseline','matched-baseline','screen','grn'}:
        raise NotImplementedError('Full-cell top20 dispatch requires a separately verified shortlist module; not implemented')
    if stage!='baseline':
        validate_frame(plan,frame)
        if frame['status']!='READY_FOR_SEPARATE_FRAME_REVIEW':
            raise ValueError('Held baseline frame cannot produce effect tasks')
    cells=plan['spec']['cells']; geometry=plan['spec']['chunk_geometry']
    n_shards=geometry.get('regulator_shards',1)
    if type(n_shards) is not int or n_shards<1:
        raise ValueError('Invalid regulator shard count')
    if stage=='baseline':
        samples={'full_population':[cell['cell_id'] for cell in cells]}
    elif stage=='matched-baseline':
        samples={name:[cell['cell_id'] for cell in item['cells']] for name,item in frame['samples'].items()}
        samples['full_common']=frame['common_eligible_cell_ids']
    else:
        keys=['primary','stability'] if stage=='screen' else ['primary']
        samples={name:[cell['cell_id'] for cell in frame['samples'][name]['cells']] for name in keys}
    tasks=[]
    for model in sorted(plan['spec']['models'],key=lambda item:item['seed']):
        for sample,ids in samples.items():
            if stage in {'baseline','matched-baseline'}:
                regulators=[{'stable_id':'__baseline__','feature_id':'__baseline__','input_space':'none'}]
            else:
                regulators=[{**item,'input_space':'rna'} for item in plan['tf_universe']]
                if stage=='grn':
                    regulators += [{'stable_id':item['feature_id'],'feature_id':item['feature_id'],'input_space':'motif'} for item in plan['spec']['motifs']]
            for regulator in regulators:
                key={'schema':'dynavelo_screen_task_v1','plan_sha256':plan['plan_sha256'],
                     'frame_sha256':frame['frame_sha256'] if frame else None,
                     'checkpoint_sha256':model['checkpoint']['sha256'],'seed':model['seed'],'stage':stage,'sample_id':sample,
                     'sample_cell_axis_sha256':identity(ids),'n_cells':len(ids),'tf_stable_id':regulator['stable_id'],
                     'feature_id':regulator['feature_id'],'input_space':regulator['input_space'],
                     'mode':'rna-only' if stage=='screen' else 'finite-difference' if stage=='grn' else 'evaluation-fixed',
                     'chunk_geometry':geometry,'batches':batches(len(ids),geometry['cells']),
                     'full_global_minima_sha256':identity(plan['spec']['global_minima']),
                     'epsilon':1e-4 if stage=='grn' else None}
                task=seal(key,'task_id'); task['assigned_shard']=int(task['task_id'][:16],16)%n_shards
                tasks.append(task)
    return seal({'schema':'dynavelo_task_book_v1','stage':stage,'plan_sha256':plan['plan_sha256'],
                 'frame_sha256':frame['frame_sha256'] if frame else None,'tasks':tasks,
                 'full_tf_denominator':len(plan['tf_universe']),'expected_tasks':len(tasks),
                 'technical_attempt_limit':plan['policy']['technical_attempt_limit'],
                 'coverage_contract':'Every expected task is required; failures cannot be removed from the denominator'},'book_sha256')

def validate_book(book):
    check_seal(book,'book_sha256')
    if book.get('schema')!='dynavelo_task_book_v1' or len(book['tasks'])!=book['expected_tasks']:
        raise ValueError('Invalid task book')
    ids=[]
    for task in book['tasks']:
        body={key:value for key,value in task.items() if key!='assigned_shard'}
        check_seal(body,'task_id')
        if task['plan_sha256']!=book['plan_sha256'] or task['frame_sha256']!=book['frame_sha256'] or task['stage']!=book['stage']:
            raise ValueError('Task escaped book identity')
        if task['assigned_shard']!=int(task['task_id'][:16],16)%task['chunk_geometry'].get('regulator_shards',1):
            raise ValueError('Task shard assignment changed')
        ids.append(task['task_id'])
    if len(ids)!=len(set(ids)) or book['technical_attempt_limit']!=3:
        raise ValueError('Duplicate tasks or changed attempt cap')
    return book

class TaskRegistry:
    def __init__(self,root,book,*,plan,frame=None):
        validate_book(book)
        if canonical(task_book(plan,book['stage'],frame))!=canonical(book):
            raise ValueError('Task book does not enumerate the full frozen plan/frame')
        self.root=Path(root).absolute()
        allowed=[Path(r'D:\dyc_7\DynaVelo_Research'),Path('/mnt/d/dyc_7/DynaVelo_Research'),Path('/content/DynaVelo_Research')]
        if self.root.resolve()!=self.root or not any(self.root.is_relative_to(path) for path in allowed):
            raise ValueError('Registry must remain in an authorized project copy')
        self.root.mkdir(parents=True,exist_ok=True)
        self.book=book; self.tasks={item['task_id']:item for item in book['tasks']}
        self.plan=plan; self.frame=frame
        path=self.path('task_book.json')
        data=canonical(book)
        if path.exists():
            if path.read_bytes()!=data:
                raise ValueError('Existing registry belongs to another immutable book')
        else:
            self.atomic_new('task_book.json',data)

    def path(self,name):
        return safe_target(self.root,name)

    def atomic_new(self,name,data):
        path=self.path(name); path.parent.mkdir(parents=True,exist_ok=True)
        self.path(name)
        temporary=path.with_name('.part_'+uuid.uuid4().hex[:12])
        try:
            with temporary.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.link(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()

    @contextmanager
    def locked(self,task_id):
        if task_id not in self.tasks:raise ValueError('Task outside full frozen book')
        name='locks/'+task_id+'.lock'
        self.atomic_new(name,canonical({'task_id':task_id,'pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat()}))
        try:yield
        finally:self.path(name).unlink()

    def resume(self,task_id):
        if task_id not in self.tasks:raise ValueError('Unknown task')
        pointer=self.path('tasks/'+task_id+'/COMPLETE.json')
        if not pointer.exists():
            folder=self.path('tasks/'+task_id)
            attempts=sorted(folder.glob('attempt_*')) if folder.exists() else []
            status='PENDING' if not attempts else 'INCOMPLETE'
            if attempts and (attempts[-1]/'COMMIT.json').exists():status='FAILED'
            return {'status':status,'attempts':len(attempts),'task_id':task_id}
        reference=json.loads(pointer.read_text())
        expected_prefix='tasks/'+task_id+'/attempt_'
        relative=reference['manifest_relative_path'];safe_relative(relative)
        if not relative.startswith(expected_prefix) or not relative.endswith('/COMMIT.json'):
            raise ValueError('Completion pointer escapes task attempts')
        manifest_path=self.path(relative)
        if manifest_path.stat().st_size!=reference['bytes'] or sha256(manifest_path)!=reference['sha256']:
            raise ValueError('Commit manifest changed')
        manifest=json.loads(manifest_path.read_text())
        if (manifest['status']!='COMPLETE' or canonical(manifest['task'])!=canonical(self.tasks[task_id])
                or manifest['book_sha256']!=self.book['book_sha256']):
            raise ValueError('Completed shard belongs to different task/book')
        parent=manifest_path.parent
        declared=[]
        for record in manifest['files']:
            safe_relative(record['name']);declared.append(record['name'])
            path=safe_target(parent,record['name'])
            if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
                raise ValueError('Completed shard payload changed')
        actual=[path.relative_to(parent).as_posix() for path in parent.rglob('*') if path.is_file() and path.name!='COMMIT.json']
        if len(declared)!=len(set(declared)) or set(actual)!=set(declared):
            raise ValueError('Shard members differ from committed manifest')
        return {'status':'COMPLETE','task_id':task_id,'manifest_sha256':reference['sha256'],
                'manifest_relative_path':relative,'files':manifest['files']}

    def commit(self,task_id,payloads,*,status='COMPLETE'):
        if status not in {'COMPLETE','FAILED'} or not payloads:
            raise ValueError('Explicit complete/failed result and evidence required')
        for name,data in payloads.items():
            safe_relative(name)
            if name in {'COMMIT.json','COMPLETE.json'} or not isinstance(data,bytes):
                raise ValueError('Invalid payload name/type')
        if task_id not in self.tasks:raise ValueError('Task outside frozen book')
        if status=='COMPLETE':
            if 'result.json' not in payloads:raise ValueError('Completed task needs a structured result')
            result=json.loads(payloads['result.json'].decode('utf-8'))
            task=self.tasks[task_id]
            if (result.get('schema')!='dynavelo_task_result_v1' or result.get('task_id')!=task_id
                    or result.get('status')!='COMPLETE' or result.get('numerical_integrity')!='PASS'
                    or result.get('n_cells')!=task['n_cells'] or result.get('mode')!=task['mode']
                    or result.get('sample_cell_axis_sha256')!=task['sample_cell_axis_sha256']):
                raise ValueError('Numerical result does not bind exact task/cell axis/mode')
            if task['stage']=='screen' and (type(result.get('primary_effect')) not in {int,float} or not np.isfinite(result['primary_effect'])):
                raise ValueError('Every completed screen task needs its finite prespecified primary effect')
        with self.locked(task_id):
            existing=self.resume(task_id)
            if existing['status']=='COMPLETE':
                proposed={name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in payloads.items()}
                saved={record['name']:{'bytes':record['bytes'],'sha256':record['sha256']} for record in existing['files']}
                if status!='COMPLETE' or proposed!=saved:raise FileExistsError('Cannot replace a completed shard with different evidence')
                return existing
            number=existing['attempts']+1
            if number>self.book['technical_attempt_limit']:
                raise RuntimeError('Persistent three-attempt cap reached; task remains in denominator')
            prefix='tasks/'+task_id+f'/attempt_{number:03d}/'
            self.path(prefix.rstrip('/')).mkdir(parents=True,exist_ok=False)
            files=[]
            for name,data in sorted(payloads.items()):
                self.atomic_new(prefix+name,data)
                files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            manifest={'schema':'dynavelo_task_commit_v1','task':self.tasks[task_id],'book_sha256':self.book['book_sha256'],
                      'attempt':number,'status':status,'files':files,'created_utc':datetime.now(timezone.utc).isoformat()}
            self.atomic_new(prefix+'COMMIT.json',canonical(manifest))
            if status=='COMPLETE':
                path=self.path(prefix+'COMMIT.json')
                self.atomic_new('tasks/'+task_id+'/COMPLETE.json',canonical({'manifest_relative_path':prefix+'COMMIT.json','bytes':path.stat().st_size,'sha256':sha256(path)}))
                return self.resume(task_id)
            return {'status':'FAILED','task_id':task_id,'attempt':number}

    def coverage(self):
        rows=[]
        for task_id in self.tasks:
            try:rows.append(self.resume(task_id))
            except (ValueError,OSError,KeyError,json.JSONDecodeError):rows.append({'task_id':task_id,'status':'CORRUPT'})
        counts={status:sum(row['status']==status for row in rows) for status in ('COMPLETE','PENDING','FAILED','INCOMPLETE','CORRUPT')}
        return seal({'schema':'dynavelo_task_coverage_v1','book_sha256':self.book['book_sha256'],'plan_sha256':self.book['plan_sha256'],
                     'stage':self.book['stage'],'full_tf_denominator':self.book['full_tf_denominator'],'expected_tasks':len(self.tasks),
                     'counts':counts,'all_expected_tasks_complete':counts['COMPLETE']==len(self.tasks),'tasks':rows},'coverage_sha256')

    def require_full_screen_coverage(self):
        coverage=self.coverage()
        if self.book['stage']!='screen' or not coverage['all_expected_tasks_complete']:
            raise PermissionError('Full TF ranking forbidden: all 3 seeds and both samples must complete; failures stay in denominator')
        return coverage
