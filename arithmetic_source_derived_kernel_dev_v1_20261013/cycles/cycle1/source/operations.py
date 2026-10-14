"""Fixed-cycle technical dispatcher; detached, bounded, no science launch."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from runner import atomic_json
from core import ARMS

ROOT = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',type=int,choices=(2,3),required=True);a=parser.parse_args()
    folder=ROOT/f'cycles/cycle{a.cycle}';folder.mkdir(parents=True,exist_ok=True)
    start=time.time(); pending=list(ARMS); active={}; completed=[]
    failures=[]; handles=[]
    try:
        while pending or active:
            for arm,(proc,handle) in list(active.items()):
                if proc.poll() is None:continue
                handle.close();del active[arm]
                if proc.returncode!=0:failures.append(dict(arm=arm,exit_code=proc.returncode))
                else:completed.append(arm)
            if failures:
                # Let the other active whole technical job finish. No new jobs
                # and no automatic retry; preserve all partial evidence.
                pending=[]
                if not active:break
            while pending and len(active)<2 and not failures:
                if time.time()-start>8*3600:
                    atomic_json(folder/'SESSION_RESULT.json',dict(status='PAUSED_SESSION_LIMIT',remaining=pending,committed=completed));pending=[];break
                arm=pending.pop(0)
                log=folder/'logs'/f'{arm}.log';log.parent.mkdir(parents=True,exist_ok=True)
                handle=log.open('a');handles.append(handle)
                proc=subprocess.Popen([os.sys.executable,'-u',str(ROOT/'runner.py'),'--cycle',str(a.cycle),'--arm',arm],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                active[arm]=(proc,handle)
            if time.time()-start>10*3600:
                for proc,_ in active.values():proc.terminate()
                failures.append(dict(error='hard ten-hour boundary; partial jobs preserved'))
                break
            atomic_json(folder/'STATUS.json',dict(status='RUNNING' if not failures else 'DRAINING_AFTER_FAILURE',cycle=a.cycle,heartbeat=time.time(),started=start,supervisor_pid=os.getpid(),workers=2,active={arm:proc.pid for arm,(proc,_) in active.items()},committed=completed,pending=pending,failures=failures,scientific_jobs=0))
            time.sleep(2)
        final='FAILED' if failures else 'PAUSED_SESSION_LIMIT' if len(completed)<5 else 'ACCEPTED_TECHNICAL_JOBS'
        atomic_json(folder/'STATUS.json',dict(status=final,cycle=a.cycle,heartbeat=time.time(),started=start,supervisor_pid=os.getpid(),workers=2,committed=completed,failures=failures,scientific_jobs=0))
        if failures:
            atomic_json(folder/'FAILURE.json',dict(failures=failures,committed=completed))
            raise RuntimeError('technical worker failed; preserved, no automatic retry')
    finally:
        for handle in handles:
            if not handle.closed:handle.close()


if __name__=='__main__':main()
