"""Operations-only dispatcher handoff. Existing worker PIDs/streams remain untouched."""
import hashlib,inspect,json,os,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
I=HERE.parents[1];B=I/'ops/dev_b_20261017';sys.path.insert(0,str(B))
import campaign_common as c
import campaign_supervisor as original
sys.path.insert(0,str(HERE))
from memory_policy import Policy,usage

ADOPTED={}
class ExistingProcess:
    def __init__(self,pid,folder,start):self.pid=pid;self.folder=Path(folder);self.start=start
    def poll(self):
        if (self.folder/'FAILURE.json').exists():return 1
        try:
            if usage(self.pid)['start_abstime']!=self.start:return 0 if (self.folder/'AUDIT.json').exists() else 1
        except OSError:return 0 if (self.folder/'AUDIT.json').exists() else 1
        state=subprocess.run(['/bin/ps','-p',str(self.pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
        if not state or state.startswith('Z'):
            return 0 if (self.folder/'AUDIT.json').exists() else 1
        return None
    def send_signal(self,s):
        if usage(self.pid)['start_abstime']==self.start:os.kill(self.pid,s)
    def terminate(self):self.send_signal(signal.SIGTERM)

def stage_function():
    source=inspect.getsource(original.run_stage)
    old='running={};completed=manifest["full_jobs"]-len(pending)'
    new='running={pid:x for pid,x in ADOPTED.items() if (x[2]["world"],x[2]["arm"]) not in certified};pending=[j for j in pending if j["key"] not in {x[2]["key"] for x in running.values()}];completed=len(certified)'
    assert source.count(old)==1
    source=source.replace(old,new).replace('memory=memory_admission(running)','memory=memory_admission(running,job_arm=pending[0]["arm"])')
    g=dict(original.__dict__);g.update(ADOPTED=ADOPTED,memory_admission=Policy().admission)
    exec(compile(source,str(HERE/'ADAPTED_STAGE.py.txt'),'exec'),g)
    return g['run_stage'],source

def verify():
    c.verify_lock();lock=c.read(HERE/'SOURCE_LOCK.json')
    for name,digest in lock['files'].items():
        if c.sha(HERE/name)!=digest:raise PermissionError('Speedup source mismatch:'+name)
    if c.sha(B/'SOURCE_LOCK.json')!=lock['frozen_DEV_B_lock_sha256']:raise PermissionError('DEV_B source identity changed')
    assert c.read(HERE/'QUALIFICATION.json')['passed']

def inventory(parent,manifest):
    text=subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,stat=,command='],text=True)
    result={}
    for line in text.splitlines():
        fields=line.strip().split(None,3)
        if len(fields)!=4 or int(fields[1])!=parent or 'campaign_worker.py' not in fields[3]:continue
        pid=int(fields[0]);args=shlex.split(fields[3])
        if args[2]!=str(B/'campaign_worker.py'):raise PermissionError('Unexpected worker source')
        get=lambda name:args[args.index(name)+1]
        assert get('--stage')=='DEV_B'
        arm=get('--arm');world=int(get('--world'));key=f'{world}_{arm}'
        job=next(j.copy() for j in manifest['jobs'] if j['key']==key);folder=c.job_folder('DEV_B',job)
        assert get('--out')==str(folder)
        job['folder']=str(folder)
        result[pid]=(ExistingProcess(pid,folder,usage(pid)['start_abstime']),(folder/'job.log').open('a'),job)
    return result

def main():
    verify();assert not (HERE/'LAUNCH.json').exists(),'No automatic controller restart'
    before=c.read(c.RESULTS/'STATUS.json');assert before['status']=='RUNNING' and before['stage']=='DEV_B'
    assert not (c.RESULTS/'FAILURE.json').exists()
    old=c.read(c.RESULTS/'LAUNCH.json')['supervisor_pid'];assert old==46260
    manifest=c.manifest('DEV_B');coffee=None;old_stopped=False;old_retired=False
    try:
        # Stop only dispatch. Workers receive no signal and finish the exact same histories.
        os.kill(old,signal.SIGSTOP);old_stopped=True
        now=time.time();status=c.read(c.RESULTS/'STATUS.json')
        elapsed=status['session_elapsed_seconds']+max(0,min(10,now-status['heartbeat_epoch']))
        session_start=time.monotonic()-elapsed
        ADOPTED.update(inventory(old,manifest))
        accepted={j['key'] for j in manifest['jobs'] if (c.job_folder('DEV_B',j)/'AUDIT.json').exists()}
        adopted_keys={entry[2]['key'] for entry in ADOPTED.values()}
        for job in manifest['jobs']:
            folder=c.job_folder('DEV_B',job)
            if (folder/'EXECUTION_LEDGER.json').exists() and job['key'] not in accepted|adopted_keys:
                raise PermissionError('Unowned incomplete job; never replay:'+job['key'])
        coffee=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        snapshots={pid:dict(job=job,ACTIVE=c.read(Path(job['folder'])/'ACTIVE.json') if (Path(job['folder'])/'ACTIVE.json').exists() else None,
                            live_usage=usage(pid)) for pid,(proc,handle,job) in ADOPTED.items()}
        launch=dict(schema='NATIVE_BIND_OPERATIONAL_MEMORY_SPEEDUP',stage='DEV_B',supervisor_pid=os.getpid(),session_id=os.getsid(0),
            caffeinate_pid=coffee.pid,workers=4,numerical_threads=1,utc=c.now(),log=str(HERE/'supervisor.log'),
            original_supervisor_pid=old,worker_source=str(B/'campaign_worker.py'),source_lock=str(HERE/'SOURCE_LOCK.json'),
            frozen_scientific_source_lock=str(B/'SOURCE_LOCK.json'),original_session_elapsed_seconds_carried=elapsed,
            adopted_worker_pids=sorted(ADOPTED),no_worker_restarted=True,science_unchanged=True)
        c.write(HERE/'HANDOFF.json',dict(launch=launch,before_status=status,workers=snapshots,counts=c.update_index(),
            original_launch=c.read(c.RESULTS/'LAUNCH.json'),original_supervisor_expected_stop=True))
        # Retire the stopped dispatcher, not its workers. Child worker stdout descriptors remain open.
        os.kill(old,signal.SIGKILL);old_retired=True
        c.write(HERE/'LAUNCH.json',launch);c.write(c.RESULTS/'LAUNCH.json',launch)
        fn,source=stage_function();(HERE/'ADAPTED_STAGE.py.txt').write_text(source)
        print('ADOPT',sorted(ADOPTED),'from',old,'elapsed carried',elapsed,flush=True)
        if fn('DEV_B',B/'AUTHORIZATION.json',session_start):
            c.status('ANALYZING',stage='DEV_B',active_jobs=[])
            with (HERE/'analysis.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(B/'campaign_finish.py')],cwd=I,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    c.write(c.RESULTS/'ANALYSIS_HEARTBEAT.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),heartbeat_epoch=time.time()));time.sleep(5)
                if proc.returncode:raise RuntimeError('Frozen full DEV analysis failed;see '+str(HERE/'analysis.log'))
    except BaseException as exc:
        if not old_retired:
            if old_stopped:os.kill(old,signal.SIGCONT)
            c.write(HERE/'HANDOFF_REJECTED.json',dict(error=repr(exc),original_dispatcher_resumed=True,science_unchanged=True))
        else:
            original.join_after_stop(ADOPTED)
            c.write(c.RESULTS/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=c.now(),**c.update_index()))
            c.status('FAILED',stage='DEV_B',error=repr(exc),active_jobs=[])
        raise
    finally:
        if coffee and coffee.poll() is None:coffee.terminate()
if __name__=='__main__':main()
