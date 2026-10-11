"""Detached two-worker dispatcher; no retries, 9h drain/10h hard session limit."""
import fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from confirm_ops import ROOT,RESULTS,OPS,atomic,read,counts,verify_sources
import settings

def dispatch(plan,workers,logdir,tick,done,drain_after=32400,hard_limit=36000):
    active={};index=0;start=time.monotonic();maximum=0
    logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True)
    try:
        while index<len(plan) or active:
            elapsed=time.monotonic()-start
            if elapsed>=hard_limit:raise TimeoutError('10-hour hard session limit; preserve partial work')
            while index<len(plan) and len(active)<workers and elapsed<drain_after:
                world,cmd=plan[index];index+=1
                log=(logdir/f'{world}.log').open('x')
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                active[world]=(child,log);maximum=max(maximum,len(active))
                print(f'DISPATCH world={world} pid={child.pid}',flush=True)
            tick(active,elapsed)
            for world,(child,log) in list(active.items()):
                rc=child.poll()
                if rc is None:continue
                log.close();del active[world]
                if rc!=0:raise RuntimeError(f'world {world}: worker exit {rc}; '+(logdir/f'{world}.log').read_text()[-5000:])
                done(world)
            if not active and elapsed>=drain_after:return dict(stage='PAUSED_SESSION_LIMIT',dispatched=index,max_workers=maximum,next_index=index)
            if active:time.sleep(.5)
        return dict(stage='EXECUTION_COMPLETE',dispatched=index,max_workers=maximum)
    finally:
        for child,log in active.values():
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            log.close()

def main():
    RESULTS.mkdir(parents=True,exist_ok=True);OPS.mkdir(exist_ok=True)
    lock=(OPS/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RESULTS/'LAUNCH.json').exists() or counts()['committed_worlds']:raise ValueError('already launched; no automatic restart')
    source=verify_sources()
    if read(ROOT/'QUALIFICATION.json')['verdict']!='PASS':raise ValueError('qualification not passed')
    if read(ROOT/'AUTHORIZATION.json')['worlds']!=list(settings.CONFIRM_WORLDS):raise ValueError('authorization roster')
    def interrupt(sig,frame):raise InterruptedError('supervisor interrupted; no retries')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    started=time.monotonic();caf=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL)
    def status(stage,**extra):
        atomic(RESULTS/'STATUS.json',dict(stage=stage,heartbeat_unix=time.time(),supervisor_pid=os.getpid(),caffeinate_pid=caf.pid,workers=2,**counts(),**extra))
    atomic(RESULTS/'LAUNCH.json',dict(supervisor_pid=os.getpid(),caffeinate_pid=caf.pid,workers=2,interpreter=sys.executable,
        supervisor_log=str(OPS/'supervisor.log'),source_lock_sha256=source,time_unix=time.time(),worlds=list(settings.CONFIRM_WORLDS),histories=384,training_bytes=1540608,retries=0,drain_after_seconds=32400,session_hard_limit_seconds=36000,GitHub_upload_authorized=False),exclusive=True)
    committed={}
    def tick(active,elapsed):
        verify_sources();entries={}
        for world,(child,_) in active.items():
            item=dict(world=world,worker_pid=child.pid,heartbeat_unix=time.time(),partial_work_not_committed=True)
            entries[str(world)]=item;atomic(RESULTS/'active'/f'{world}.json',item)
        for path in (RESULTS/'active').glob('*.json'):
            if path.stem not in entries:path.unlink()
        status('EXECUTION',active_worlds=entries,elapsed_wall_seconds=elapsed)
    def done(world):
        r=read(RESULTS/'receipts'/f'WORLD_{world}.json.gz')
        if r['world']!=world or r['independent_audit']['verdict']!='PASS' or r['equation_audit']['verdict']!='PASS' or r['training_histories']!=6 or r['training_bytes']!=24072:raise ValueError('worker missing accepted full-world receipt')
        committed[str(world)]=dict(histories=6,training_bytes=24072,path=str(RESULTS/'receipts'/f'WORLD_{world}.json.gz'))
        atomic(RESULTS/'COMMIT_INDEX.json',committed)
        (RESULTS/'active'/f'{world}.json').unlink(missing_ok=True)
        print('COMMITTED '+str(counts()),flush=True)
    try:
        plan=[(w,[sys.executable,'-u',str(ROOT/'runner.py'),'--world',str(w),'--out',str(RESULTS/'receipts'/f'WORLD_{w}.json.gz')]) for w in settings.CONFIRM_WORLDS]
        result=dispatch(plan,2,RESULTS/'logs',tick,done)
        if result['stage']=='PAUSED_SESSION_LIMIT':status('PAUSED_SESSION_LIMIT',**{k:v for k,v in result.items() if k!='stage'},resume_requires_authorization=True);return
        status('ANALYSIS')
        from analysis import analyze,package
        result=analyze();status('PACKAGING',verdict=result['verdict']);package(result)
        status('COMPLETE',verdict=result['verdict'],accepted_audit=True,package_complete=True,elapsed_wall_seconds=time.monotonic()-started)
        print('COMPLETE '+result['verdict'],flush=True)
    except BaseException as exc:
        atomic(RESULTS/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time_unix=time.time(),**counts(),retries=0),exclusive=True)
        status('FAILED',error=repr(exc));raise
    finally:
        caf.terminate();caf.wait(timeout=10)

if __name__=='__main__':main()
