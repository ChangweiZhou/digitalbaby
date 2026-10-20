"""Finite engineering-only dispatcher; no pilot or confirmation launch path."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import bootstrap
import json
import subprocess
import sys
import time

from artifacts import HERE,atomic_json,source_identity,source_digest,sha
from audit import audit_job,metric_fixtures


def main():
    started=time.monotonic();awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    spec=json.loads((HERE/'SPEC.json').read_text())
    assert json.loads((HERE/'cycles/cycle1/AUDIT.json').read_text())['accepted']
    assert json.loads((HERE/'cycles/cycle2/AUDIT.json').read_text())['accepted']
    atomic_json(HERE/'cycles/cycle3/METRIC_FIXTURES.json',metric_fixtures())
    roster=[dict(world=w,arm=a,seed=202610208910+k) for k,w in enumerate(spec['engineering_worlds']) for a in spec['arms']]
    source=source_identity()
    atomic_json(HERE/'cycles/cycle3/ROSTER.json',dict(technical_only=True,jobs=roster,
        full_jobs=8,histories=24,record_applications=199680,pilot_jobs=0,confirmation_jobs=0))
    atomic_json(HERE/'cycles/cycle3/SOURCE_LOCK.json',dict(files=source,digest=source_digest(),
        native_asset_audit_sha256=sha(HERE/'cycles/cycle1/NATIVE_ASSET_AUDIT.json'),
        parameter_lock_sha256=sha(HERE/'PARAMETER_LOCK.json')))
    from positive_control import run as parent_control
    parent_control()
    running={};completed=[];queue=list(roster)
    try:
        while queue or running:
            if time.monotonic()-started>8*3600:
                raise RuntimeError('engineering session admission boundary reached; no automatic continuation')
            try:
                import psutil
                available=psutil.virtual_memory().available
                concurrency=min(4,max(0,int((available-512*1024**2)/(512*1024**2))+len(running)))
            except ImportError:concurrency=2
            while queue and len(running)<concurrency:
                job=queue.pop(0);name=f"{job['world']}_{job['arm']}";folder=HERE/'cycles/cycle3/jobs'/name
                folder.mkdir(parents=True,exist_ok=True)
                log=open(folder/'job.log','w')
                command=[sys.executable,'-u',str(HERE/'protocol.py'),'--world',str(job['world']),
                         '--arm',job['arm'],'--seed',str(job['seed'])]
                proc=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                running[name]=(proc,log,folder)
                print('START',name,proc.pid,flush=True)
            for name,(proc,log,folder) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del running[name]
                if code!=0:raise RuntimeError(f'engineering job {name} exited {code}; preserve log; no retry')
                audit=audit_job(folder);completed.append(dict(job=name,**audit))
                print('ACCEPTED',name,audit['technical_records'],flush=True)
            atomic_json(HERE/'STATUS.json',dict(status='CYCLE_3_RUNNING' if running else 'WAITING_RESOURCE',
                technical_only=True,scientific_records=0,completed_jobs=len(completed),jobs_total=8,
                accepted_record_applications=sum(c['technical_records'] for c in completed),
                active_pids={name:p.pid for name,(p,_,_) in running.items()},unstarted=len(queue),
                elapsed_wall_s=time.monotonic()-started))
            atomic_json(HERE/'cycles/cycle3/COMMIT_INDEX.json',dict(accepted_jobs=completed,scientific_records=0))
            if queue or running:time.sleep(5)
        assert len(completed)==8
        assert source_digest()==json.loads((HERE/'cycles/cycle3/SOURCE_LOCK.json').read_text())['digest']
        atomic_json(HERE/'cycles/cycle3/AUDIT.json',dict(accepted=True,jobs=completed,
            full_engineering_jobs=8,continuous_histories=24,record_applications=199680,
            scientific_records=0,elapsed_wall_s=time.monotonic()-started,source_digest=source_digest()))
        print('CYCLE_3_ACCEPTED',len(completed),'wall_s',round(time.monotonic()-started),flush=True)
    except Exception as exc:
        atomic_json(HERE/'cycles/cycle3/FAILURE.json',dict(error=str(exc),accepted_jobs=completed,
            running={name:p.pid for name,(p,_,_) in running.items()},scientific_records=0))
        raise


if __name__=='__main__':main()
