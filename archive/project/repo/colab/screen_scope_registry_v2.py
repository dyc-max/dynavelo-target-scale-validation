"""Scoped task identity and real COMMIT validation; unchanged atomic primitives."""
import json
import os
from contextlib import contextmanager
import numpy as np
import screen_dispatch as base
from screen_plan import canonical, seal
from screen_task_registry import TaskRegistry
from screen_scope_plan_runtime_v2 import scope_task_book, validate_scope_book
from human_return_common import safe_target,sha256

STAGE_DIR={'baseline':'b','matched-baseline':'m','screen':'s'}

def stage_root(output,stage):
    return output/STAGE_DIR[stage]


def validate_result(task,value):
    expected={'schema':'dynavelo_scope_task_result_v2','task_id':task['task_id'],'status':'COMPLETE',
              'numerical_integrity':'PASS','n_cells':task['n_cells'],'mode':task['mode'],
              **{k:task[k] for k in ('scope_plan_sha256','scope_sha256','scope_frame_sha256','context_plan_sha256',
                  'full_context_frame_sha256','calibration_sha256','context_invariants_sha256',
                  'sample_cell_axis_sha256','sample_weights_sha256','checkpoint_sha256','stage','seed','sample_id')}}
    if any(type(value.get(k))!=type(v) or value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Result escaped exact v2 task/scope/calibration identity')
    if value.get('biological_or_external_PASS') is not False:raise ValueError('Numerical tasks cannot grant scientific PASS')
    if task['stage']=='screen' and (type(value.get('primary_effect')) not in {int,float} or not np.isfinite(value['primary_effect'])):
        raise ValueError('Finite complete primary effect required')
    return value


class ScopeRegistry(TaskRegistry):
    def __init__(self,root,book,*,plan,frame=None):
        validate_scope_book(plan,book,frame)
        self.root=base.checked_root(root);self.plan=plan;self.frame=frame
        self.book=book;self.tasks={t['task_id']:t for t in book['tasks']};self._matched=None
        self._parent_cache=None;self._audit_checkpoint=None
        expected='results/screen_runs_v2/'+plan['scope_plan_sha256']+'/'+STAGE_DIR[book['stage']]
        execution_root=base.checked_root(plan['context_plan']['spec']['scope_execution_contract']['registry_execution_root'])
        if self.root!=safe_target(execution_root,expected):raise ValueError('Registry must use unique exact scope/stage root')
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.path('task_book.json');data=canonical(book)
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Different immutable scope book already exists')
        else:self.atomic_new('task_book.json',data)

    def resume(self,task_id):
        if self._audit_checkpoint:self._audit_checkpoint()
        saved=super().resume(task_id)
        if saved['status']=='COMPLETE':
            directory=self.path(saved['manifest_relative_path']).parent
            value=validate_result(self.tasks[task_id],base.read_json(directory/'result.json'))
            if self.book['stage']=='screen':
                if self._matched is None:
                    self._matched=ScopeRegistry(stage_root(self.root.parent,'matched-baseline'),scope_task_book(self.plan,'matched-baseline',self.frame),plan=self.plan,frame=self.frame)
                task=self.tasks[task_id]
                matches=[t for t in self._matched.book['tasks'] if t['seed']==task['seed'] and t['sample_id']==task['sample_id']]
                if len(matches)!=1:raise ValueError('Matched dependency is not unique')
                dependency=matches[0];key=dependency['task_id']
                if self._parent_cache is not None and key in self._parent_cache:
                    record=self._parent_cache[key]
                else:
                    record=self._matched.resume(key)
                    if self._parent_cache is not None:self._parent_cache[key]=record
                if record['status']!='COMPLETE':raise ValueError('Completed screen has missing/corrupt matched baseline')
                expected={'task_id':dependency['task_id'],'book_sha256':self._matched.book['book_sha256'],
                          'commit_sha256':record['manifest_sha256']}
                if value.get('matched_baseline_dependency')!=expected:raise ValueError('Completed screen dependency changed')
        return saved

    @contextmanager
    def dependency_audit(self,checkpoint=None):
        """Unique-parent verification within ONE audit; reverify before success.

        Never cache across audits/invocations. Core payload hashing is still
        cooperative at file/audit boundaries, not native-I/O hard preemption.
        """
        if self._parent_cache is not None:
            if checkpoint:checkpoint()
            yield
            return
        self._parent_cache={};self._audit_checkpoint=checkpoint
        try:
            if checkpoint:checkpoint()
            yield
            for key,original in self._parent_cache.items():
                if checkpoint:checkpoint()
                if self._matched.resume(key)!=original:raise ValueError('Matched dependency changed during audit')
                if checkpoint:checkpoint()
        finally:
            self._parent_cache=None;self._audit_checkpoint=None

    def commit(self,*args,**kwargs):
        raise NotImplementedError('Use ScopeTaskWriter; inherited v1 result commit is not supported')

    def coverage(self,checkpoint=None):
        rows=[]
        with self.dependency_audit(checkpoint):
            for key in self.tasks:
                try:rows.append(self.resume(key))
                except (ValueError,OSError,KeyError,json.JSONDecodeError):rows.append({'task_id':key,'status':'CORRUPT'})
        counts={name:sum(r['status']==name for r in rows) for name in ('COMPLETE','PENDING','FAILED','INCOMPLETE','CORRUPT')}
        return seal({'schema':'dynavelo_scope_task_coverage_v2','book_sha256':self.book['book_sha256'],
            'scope_plan_sha256':self.book['scope_plan_sha256'],'scope_frame_sha256':self.book['scope_frame_sha256'],
            'scope_sha256':self.book['scope_sha256'],'stage':self.book['stage'],
            'full_tf_denominator':self.book['full_tf_denominator'],'expected_tasks':len(self.tasks),
            'counts':counts,'all_expected_tasks_complete':counts['COMPLETE']==len(self.tasks),'tasks':rows},'coverage_sha256')

    def require_full_screen_coverage(self,checkpoint=None):
        coverage=self.coverage(checkpoint)
        if self.book['stage']!='screen' or not coverage['all_expected_tasks_complete']:
            raise PermissionError('Complete all-TF/three-seed/two-sample audit required')
        return coverage


class ScopeTaskWriter(base.FileTaskWriter):
    def finish(self,status,limit):
        if status not in {'COMPLETE','FAILED'}:raise ValueError('Invalid final task status')
        if status=='COMPLETE':validate_result(self.task,base.read_json(self.directory/'result.json'))
        files=[];total=0
        for path in sorted(self.directory.rglob('*')):
            if not path.is_file():continue
            name=path.relative_to(self.directory).as_posix();safe_target(self.directory,name)
            if name in {'COMMIT.json','COMPLETE.json'}:raise ValueError('Unexpected commit')
            with path.open('r+b') as f:f.flush();os.fsync(f.fileno())
            total+=path.stat().st_size;files.append({'name':name,'bytes':path.stat().st_size,'sha256':sha256(path)})
        if status=='COMPLETE' and total>limit:raise base.ResourceStop('Payload ceiling')
        manifest={'schema':'dynavelo_scope_task_commit_v2','task':self.task,'book_sha256':self.registry.book['book_sha256'],
                  'attempt':self.number,'status':status,'files':files,'created_utc':base.utc()}
        self.registry.atomic_new(self.prefix+'COMMIT.json',canonical(manifest))
        if status=='COMPLETE':
            path=self.directory/'COMMIT.json'
            self.registry.atomic_new('tasks/'+self.task_id+'/COMPLETE.json',canonical({
                'manifest_relative_path':self.prefix+'COMMIT.json','bytes':path.stat().st_size,'sha256':sha256(path)}))
            return self.registry.resume(self.task_id)
        return {'task_id':self.task_id,'status':'FAILED','attempt':self.number}
