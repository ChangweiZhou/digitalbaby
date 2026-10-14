"""Detached two-worker dispatcher. No retries or science admission override."""
import argparse,gzip,json,os,subprocess,time,sys
from pathlib import Path
from protocol import ROOT,D,atomic,identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('cycle2','cycle3','science'),required=True);a=p.parse_args()
    out=ROOT/('results' if a.stage=='science' else 'cycles/'+a.stage);out.mkdir(parents=True,exist_ok=True)
    start=time.time();session_start=json.loads((ROOT/'ops/WORK_AWAKE.json').read_text())['started_epoch']
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    worlds=D['science_worlds'] if a.stage=='science' else D['cycles'][a.stage[-1]]['worlds']
    if a.stage=='science':assert json.loads((ROOT/'SCIENCE_ADMISSION.json').read_text())['accepted']
    jobs=[(w,m) for w in worlds for m in D['models']];pending=list(jobs);active={};committed=[];failed=[];paused=False
    atomic(out/'MANIFEST.json',dict(stage=a.stage,jobs=[dict(world=w,model=m) for w,m in jobs],source=identity()))
    atomic(out/'LAUNCH.json',dict(stage=a.stage,supervisor_pid=os.getpid(),caffeinate_pid=awake.pid,workers=2,started=start,session_start=session_start,jobs=len(jobs),source=identity()))
    try:
        while pending or active:
            for key,(proc,log) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[key]
                if proc.returncode:failed.append(dict(world=key[0],model=key[1],exit_code=proc.returncode))
                else:
                    cert=out/'receipts'/f'{key[0]}_{key[1]}.json.audit.json'
                    # .json.gz -> with_suffix('.audit.json') produces .json.audit.json.
                    c=json.loads(cert.read_text());assert c['accepted'];committed.append(c)
                    atomic(out/'COMMIT_INDEX.json',dict(committed=committed,jobs=len(committed),records=sum(x['records'] for x in committed),histories=5*len(committed)))
            if failed:pending=[]
            if time.time()-session_start>=8*3600 and pending:paused=True;pending=[]
            while pending and len(active)<2 and not failed and not paused:
                key=pending.pop(0);logpath=out/'logs'/f'{key[0]}_{key[1]}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
                log=logpath.open('a');proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'runner.py'),'--stage',a.stage,'--world',str(key[0]),'--model',key[1]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[key]=(proc,log)
            if time.time()-session_start>=10*3600:
                for proc,_ in active.values():proc.terminate()
                failed.append(dict(error='hard ten-hour boundary; partial files preserved'));break
            atomic(out/'STATUS.json',dict(status='DRAINING_AFTER_FAILURE' if failed else 'PAUSED_DRAINING' if paused else 'RUNNING',stage=a.stage,heartbeat=time.time(),supervisor_pid=os.getpid(),workers=2,
                committed_jobs=len(committed),committed_records=sum(x['records'] for x in committed),active={f'{w}_{m}':p.pid for (w,m),(p,_) in active.items()},pending=len(pending),failures=failed))
            time.sleep(2)
        status='FAILED' if failed else 'PAUSED_SESSION_LIMIT' if paused else 'EXECUTION_ACCEPTED'
        atomic(out/'STATUS.json',dict(status=status,stage=a.stage,heartbeat=time.time(),committed_jobs=len(committed),committed_records=sum(x['records'] for x in committed),failures=failed))
        if failed:
            atomic(out/'FAILURE.json',dict(failures=failed,committed=committed));return 1
        if paused:atomic(out/'SESSION_RESULT.json',dict(status=status,committed=committed));return 0
        if a.stage=='science':
            stats=ROOT.parent.parent/'.venv_loop/bin/python'
            atomic(out/'STATUS.json',dict(status='ANALYSIS',heartbeat=time.time(),committed_jobs=len(committed),committed_records=sum(x['records'] for x in committed),supervisor_pid=os.getpid()))
            subprocess.run([str(stats),str(ROOT/'analysis.py'),'--final'],cwd=ROOT,check=True)
        return 0
    except Exception as e:
        atomic(out/'FAILURE.json',dict(error=repr(e),committed=committed));raise
    finally:
        for proc,log in active.values():
            if not log.closed:log.close()
        awake.terminate()
if __name__=='__main__':raise SystemExit(main())
