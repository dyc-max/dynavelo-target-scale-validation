"""Small explicit-path primitives for a new physical recovery attempt."""
from pathlib import Path, PurePosixPath
from datetime import datetime,timezone
import hashlib,json,os,re

CANONICAL='results/reports/a100_production_20261114'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def record(root,p):
    p=Path(p);return {'relative_path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def path(root,relative):
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative:raise ValueError('Unsafe relative path')
    value=root/relative
    if not value.resolve().is_relative_to(root.resolve()):raise ValueError('Artifact escapes project root')
    return value
def checked(root,rec):
    p=path(root,rec['relative_path'])
    if record(root,p)!=rec:raise ValueError('Artifact checksum differs: '+rec['relative_path'])
    return p
def attempt_root(root,identifier):
    if not re.fullmatch(r'a100_recovery_[0-9]{8}_a[2-9][0-9]*',identifier):raise ValueError('Invalid physical attempt ID')
    return root/CANONICAL/'recovery_attempts'/identifier
def new(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def status(p,value):
    temporary=p.with_name(p.name+'.partial');temporary.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('w',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(p)

def attempt_charges(task_ids,previous,accepted,ceiling=3):
    """Reserve the whole frozen block, including previously unknown remote work."""
    if len(task_ids)!=len(set(task_ids)) or not task_ids:raise ValueError('Invalid task block')
    if set(task_ids)&set(accepted):raise ValueError('Accepted task must not be rerun')
    rows=[]
    for task_id in task_ids:
        prior=previous.get(task_id,0)
        if type(prior) is not int or not 0<=prior<ceiling:raise ValueError('Cross-runtime attempt ceiling exhausted')
        rows.append({'task_id':task_id,'prior_possible_attempts':prior,'new_physical_attempts':1,'charged_total_after_reservation':prior+1,'maximum_attempts':ceiling})
    return rows

def inventory_proves_released(text,exit_code,session):
    return exit_code==0 and 'No active sessions found on server.' in text and '['+session+']' not in text
