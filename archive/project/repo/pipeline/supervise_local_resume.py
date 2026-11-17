"""Windows owned Job Object supervisor for the fixed local resume entry.

Default inspect; no model execution without --execute. The child is assigned
while suspended; all descendants are kernel-owned and killed on owner death.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,json,os,re,subprocess,sys,threading,time,traceback,uuid
from ctypes import wintypes as W
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(r'D:\dyc_7\DynaVelo_Research');ENTRY=ROOT/'repo/pipeline/train_hspc_local_resume.py'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def path(value):
    p=Path(value)
    if not p.is_absolute() or not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Absolute D project path required')
    return p.resolve()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();temp=p.with_suffix('.'+uuid.uuid4().hex+'.partial')
    with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    # Windows readers without FILE_SHARE_DELETE can briefly deny replacement.
    # Keep the complete previous JSON visible, and fail closed if denial lasts.
    deadline=time.monotonic()+1.0
    while True:
        try:
            os.replace(temp,p)
            break
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:
                raise
            time.sleep(.05)
class BasicLimits(ctypes.Structure):
    _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class IOCounters(ctypes.Structure):
    _fields_=[(n,ctypes.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class ExtendedLimits(ctypes.Structure):
    _fields_=[('BasicLimitInformation',BasicLimits),('IoInfo',IOCounters),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
class OwnedJob:
    def __init__(self):
        if os.name!='nt':raise RuntimeError('Windows only')
        self.k=ctypes.WinDLL('kernel32',use_last_error=True)
        defs={'CreateJobObjectW':([ctypes.c_void_p,W.LPCWSTR],W.HANDLE),'SetInformationJobObject':([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD],W.BOOL),
          'AssignProcessToJobObject':([W.HANDLE,W.HANDLE],W.BOOL),'QueryInformationJobObject':([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),
          'TerminateJobObject':([W.HANDLE,W.UINT],W.BOOL),'CloseHandle':([W.HANDLE],W.BOOL),'OpenThread':([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
          'ResumeThread':([W.HANDLE],W.DWORD),'OpenProcess':([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),'IsProcessInJob':([W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)],W.BOOL)}
        for name,(args,result) in defs.items():fn=getattr(self.k,name);fn.argtypes=args;fn.restype=result
        self.handle=self.k.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=ExtendedLimits();limits.BasicLimitInformation.LimitFlags=0x2000
        if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.close();raise ctypes.WinError(ctypes.get_last_error())
        self.proc=None;self.seen={};self.lock=threading.RLock()
    def contains(self,pid):
        h=self.k.OpenProcess(0x1000,False,pid)
        if not h:return False
        try:
            yes=W.BOOL();return bool(self.k.IsProcessInJob(h,self.handle,ctypes.byref(yes)) and yes.value)
        finally:self.k.CloseHandle(h)
    def pids(self):
        buf=ctypes.create_string_buffer(8+ctypes.sizeof(ctypes.c_size_t)*1024)
        if not self.k.QueryInformationJobObject(self.handle,3,buf,len(buf),None):raise ctypes.WinError(ctypes.get_last_error())
        assigned,n=(W.DWORD*2).from_buffer(buf)
        if n>1024 or assigned>1024:raise RuntimeError('Owned process ceiling exceeded')
        return list((ctypes.c_size_t*n).from_buffer(buf,8))
    def observe(self):
        with self.lock:
            current=[]
            for pid in self.pids():
                try:
                    p=psutil.Process(pid);created=p.create_time();rec={'pid':pid,'create_time':created,'ppid':p.ppid(),'exe':p.exe(),'command':p.cmdline()}
                    if not self.contains(pid) or p.create_time()!=created:continue
                    current.append(rec);self.seen[(pid,created)]=rec
                except (psutil.NoSuchProcess,psutil.AccessDenied):continue
            return current
    def launch(self,command,log,env):
        if self.proc is not None:raise RuntimeError('One launch per job')
        proc=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
          creationflags=0x00000004|subprocess.CREATE_NO_WINDOW,close_fds=True) # CREATE_SUSPENDED
        self.proc=proc
        try:
            self.launcher={'pid':proc.pid,'create_time':psutil.Process(proc.pid).create_time()}
            if not self.k.AssignProcessToJobObject(self.handle,int(proc._handle)):raise ctypes.WinError(ctypes.get_last_error())
            self.observe();threads=psutil.Process(proc.pid).threads()
            if len(threads)!=1:raise RuntimeError('Unexpected suspended primary-thread census')
            h=self.k.OpenThread(0x0002,False,threads[0].id)
            if not h:raise ctypes.WinError(ctypes.get_last_error())
            try:
                if self.k.ResumeThread(h)!=1:raise RuntimeError('Unexpected suspend count')
            finally:self.k.CloseHandle(h)
            return proc
        except BaseException:
            proc.kill();proc.wait(timeout=10);raise
    def terminate(self):
        # Kernel job ownership is already bound before ResumeThread. Do not
        # block the deadline behind process metadata reads or a Python lock.
        if not self.k.TerminateJobObject(self.handle,124):raise ctypes.WinError(ctypes.get_last_error())
    def wait_empty(self,seconds=15):
        until=time.monotonic()+seconds
        while self.pids():
            if time.monotonic()>=until:raise RuntimeError('Owned job remains live')
            time.sleep(.05)
        if self.proc is not None:self.proc.wait(timeout=5)
    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None

def inspect(a):
    if os.name!='nt':raise RuntimeError('Windows supervisor only')
    exe=path(a.python);entry=path(ENTRY);manifest=path(a.manifest)
    if exe.name.lower()!='python.exe' or exe.parent.name.lower()!='scripts' or not (exe.parent.parent/'pyvenv.cfg').is_file():raise ValueError('Explicit D-project Windows environment required')
    if sha(entry)!=a.entry_sha256 or sha(manifest)!=a.manifest_sha256:raise ValueError('Pinned entry/manifest differs')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.job_id):raise ValueError('Unsafe job ID')
    if not 1<=a.wall_seconds<=86400 or (a.action=='short-test' and a.wall_seconds>300):raise ValueError('short-test wall <=300; explicit train wall <=86400')
    if not 0<a.max_rss_gib<=64 or not 0<a.min_free_gib<=32:raise ValueError('Explicit positive memory limits required')
    if not 1<=a.threads<=32 or not 1<=a.probe_batches<=4:raise ValueError('Invalid threads/batches')
    job=path(ROOT/'tmp_cache/local_owned_jobs'/a.job_id)
    if job.exists():raise FileExistsError('Job ID already used; never silently restart')
    command=[str(exe),'-B',str(entry),'--manifest',str(manifest),'--manifest-sha256',a.manifest_sha256,
      '--device',a.device,'--action',a.action,'--threads',str(a.threads),'--execute']
    if a.action=='short-test':command+=['--probe-batches',str(a.probe_batches)]
    if a.device_evidence:
        evidence=path(a.device_evidence)
        if sha(evidence)!=a.device_evidence_sha256:raise ValueError('Device evidence SHA differs')
        command+=['--device-evidence',str(evidence),'--device-evidence-sha256',a.device_evidence_sha256]
    if a.action=='train' and a.device=='xpu' and not a.device_evidence:raise ValueError('XPU train requires same-entry short-test evidence')
    return {'status':'INSPECTED_NO_PROCESS_STARTED','job_id':a.job_id,'job_dir':str(job),'command':command,
      'python_sha256':sha(exe),'entry_sha256':a.entry_sha256,'manifest_sha256':a.manifest_sha256,
      'supervisor_sha256':sha(__file__),'wall_seconds':a.wall_seconds,'max_rss_bytes':int(a.max_rss_gib*1024**3),
      'min_free_bytes':int(a.min_free_gib*1024**3),'stop_file':str(path(a.stop_file)) if a.stop_file else str(job/'STOP')}

def supervise(spec):
    import msvcrt
    directory=Path(spec['job_dir']);directory.parent.mkdir(parents=True,exist_ok=True)
    lockfile=(directory.parent/'ACTIVE.lock').open('a+b');lockfile.seek(0)
    if lockfile.read(1)==b'':lockfile.write(b'0');lockfile.flush()
    lockfile.seek(0);msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
    owned=None;timer=None;done=threading.Event();reason=[];errors=[];peak=0
    try:
        directory.mkdir(exist_ok=False);write(directory/'job.json',{**spec,'created_utc':now(),'nonce':uuid.uuid4().hex})
        if Path(spec['stop_file']).exists():raise RuntimeError('Stop file already exists; no launch')
        if psutil.virtual_memory().available<spec['min_free_bytes']:raise RuntimeError('Insufficient system free memory before launch')
        if sha(ENTRY)!=spec['entry_sha256'] or sha(spec['command'][0])!=spec['python_sha256']:raise ValueError('Entry/interpreter drift before launch')
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUTF8='1',MPLBACKEND='Agg')
        for key,rel in {'TMP':'tmp','TEMP':'tmp','TMPDIR':'tmp','PIP_CACHE_DIR':'pip','HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba','JOBLIB_TEMP_FOLDER':'joblib'}.items():
            p=path(ROOT/'tmp_cache'/rel);p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
        owned=OwnedJob()
        def stop(why):
            if not done.is_set():
                if not reason:reason.append(why)
                owned.terminate()
        timer=threading.Timer(spec['wall_seconds'],lambda:stop('HARD_WALL'));timer.daemon=True
        with (directory/'console.log').open('xb') as log:
            started=time.monotonic();proc=owned.launch(spec['command'],log,env);timer.start()
            write(directory/'launcher.json',{'created_utc':now(),**owned.launcher,'kernel_job_assigned_before_execution':True})
            while proc.poll() is None or owned.pids():
                members=owned.observe();rss=0
                for r in members:
                    try:
                        p=psutil.Process(r['pid'])
                        if p.create_time()==r['create_time'] and owned.contains(p.pid):rss+=p.memory_info().rss
                    except psutil.NoSuchProcess:pass
                free=psutil.virtual_memory().available;peak=max(peak,rss)
                write(directory/'status.json',{'updated_utc':now(),'elapsed_seconds':time.monotonic()-started,'current_members':members,'all_observed_members':list(owned.seen.values()),'rss_bytes':rss,'peak_rss_bytes':peak,'system_available_bytes':free,'stop_reason':reason})
                if Path(spec['stop_file']).exists():stop('STOP_FILE')
                elif rss>spec['max_rss_bytes']:stop('RSS_LIMIT')
                elif free<spec['min_free_bytes']:stop('SYSTEM_FREE_LIMIT')
                if proc.poll() is not None and owned.pids():stop('LAUNCHER_EXIT_WITH_LIVE_DESCENDANTS')
                if reason:break
                time.sleep(.5)
    except BaseException as error:
        errors.append(repr(error)+'\n'+traceback.format_exc())
        if not reason:reason.append('SUPERVISOR_EXCEPTION')
    finally:
        if owned is not None:
            try:
                if owned.pids():owned.terminate()
                owned.wait_empty();empty=True
            except BaseException as error:errors.append(repr(error));empty=False
            finally:
                done.set()
                if timer:
                    timer.cancel()
                    if timer.is_alive():timer.join(timeout=5)
                exitcode=owned.proc.returncode if owned.proc is not None else None
                seen=list(owned.seen.values());owned.close()
        else:empty=True;exitcode=None;seen=[];done.set()
        if directory.exists():write(directory/'terminal.json',{'finished_utc':now(),'job_id':spec['job_id'],'exit_code':exitcode,'stop_reason':reason,
          'errors':errors,'observed_processes':seen,'owned_job_confirmed_empty_before_close':empty,'peak_rss_bytes':peak,
          'status':'PROCESS_TERMINATED_NO_SCIENTIFIC_PASS','checkpoints_and_logs_deleted':False})
        lockfile.seek(0);msvcrt.locking(lockfile.fileno(),msvcrt.LK_UNLCK,1);lockfile.close()
    return {'job_dir':str(directory),'exit_code':exitcode,'stop_reason':reason,'errors':errors,'owned_job_empty':empty}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True);p.add_argument('--entry-sha256',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--job-id',required=True)
    p.add_argument('--device',choices=['cpu','xpu'],default='cpu');p.add_argument('--action',choices=['short-test','train'],default='short-test')
    p.add_argument('--wall-seconds',type=int,required=True);p.add_argument('--max-rss-gib',type=float,required=True);p.add_argument('--min-free-gib',type=float,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--probe-batches',type=int,default=2);p.add_argument('--stop-file')
    p.add_argument('--device-evidence');p.add_argument('--device-evidence-sha256');p.add_argument('--execute',action='store_true')
    a=p.parse_args();spec=inspect(a)
    print(json.dumps(supervise(spec) if a.execute else spec,indent=2,allow_nan=False))

if __name__=='__main__':main()
