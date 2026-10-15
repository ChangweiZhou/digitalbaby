"""Two-worker TECHNICAL cycles only, safe window admission, no science/publication.

Cycle 2 must finish and be reviewed before a separate cycle-3 invocation.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import core
import provenance
from io_utils import write,read,now
from audit import audit_job


def status(cycle,state,**extra):
    write(core.bootstrap.HERE/'results/STATUS.json',dict(utc=now(),pid=os.getpid(),cycle=cycle,
          state=state,science=False,science_jobs=0,**extra))


def summarize(cycle):
    receipts=sorted((core.bootstrap.HERE/'cycles'/f'cycle{cycle}'/'receipts').glob('*.json.gz'))
    all_jobs=[read(p) for p in receipts]
    for job in all_jobs:
        audit_job(job)
    expected=5 if cycle==2 else 10
    if len(all_jobs)!=expected:
        raise AssertionError(('incomplete cycle',len(all_jobs),expected))
    summary=[]
    for job in all_jobs:
        scores={}
        for history,h in job['histories'].items():
            scores[history]=[dict(stage=m['stage'],n=m['n'],training_cpu=m['cpu'],
                 **{k:sum(r['correct'] for r in rows)/len(rows) for k,rows in m['scores'].items()})
                 for m in h['metrics']]
        summary.append(dict(world=job['world'],version=job['version'],elapsed=job['elapsed'],scores=scores))
    result=dict(accepted=True,science=False,cycle=cycle,jobs=len(all_jobs),histories=3*len(all_jobs),
                records=sum(j['records'] for j in all_jobs),results=summary,
                adoption='NOT_APPLICABLE_TECHNICAL_ONLY')
    if cycle==3:
        import statistics
        result['Cstar_training_CPU']=statistics.median(j['histories']['W']['metrics'][-2]['cpu']
                   for j in all_jobs if j['version']=='KEEP-C16-CE')
    write(core.bootstrap.HERE/'cycles'/f'cycle{cycle}'/'SUMMARY.json',result)
    lines=[f'# Technical cycle {cycle}', '',f"Accepted {len(all_jobs)} full jobs / {result['histories']} correlated histories / {result['records']} record applications.",
           '', 'Technical worlds only. No significance, scientific ranking or core-adoption claim.', '',
           '| World | Version | W taught after OLD+rest | W held after OLD+rest | W held final | N_OLD held final | SH_OLD held final | Wall seconds |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        w=row['scores']['W']; old=[m for m in w if m['stage']=='OLD_REST'][0]
        lines.append(f"| {row['world']} | {row['version']} | {old['old']:.1%} | {old['held']:.1%} | {w[-1]['held']:.1%} | {row['scores']['N_OLD'][-1]['held']:.1%} | {row['scores']['SH_OLD'][-1]['held']:.1%} | {row['elapsed']:.1f} |")
    (core.bootstrap.HERE/'cycles'/f'cycle{cycle}'/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main(cycle):
    root=core.bootstrap.HERE
    if cycle==3 and not read(root/'cycles/cycle2/REVIEW.json')['accepted_for_cycle3']:
        raise ValueError('cycle2 human-agent technical review missing')
    if (root/'cycles'/f'cycle{cycle}'/'SOURCE_LOCK.json').exists():
        raise FileExistsError('no automatic restart or source re-lock')
    provenance.snapshot(cycle)
    worlds=[202610143901] if cycle==2 else [202610143902,202610143903]
    jobs=[(w,v) for w in worlds for v in core.VERSIONS]
    write(root/'cycles'/f'cycle{cycle}'/'MANIFEST.json',dict(science=False,jobs=jobs))
    write(root/'results/LAUNCH.json',dict(utc=now(),pid=os.getpid(),cycle=cycle,workers=2,science=False,
          python=sys.executable,log=str(root/'ops'/f'cycle{cycle}.log')))
    awake=subprocess.Popen(['/usr/bin/caffeinate','-di','-w',str(os.getpid())])
    running={}; completed=[]; start=time.monotonic()
    try:
        while jobs or running:
            provenance.verify(cycle,local_only=True)
            elapsed=time.monotonic()-start
            if elapsed>=36000:
                raise TimeoutError('ten-hour hard session boundary')
            while jobs and len(running)<2 and elapsed<28800:
                w,v=jobs.pop(0)
                folder=root/'cycles'/f'cycle{cycle}'
                output=folder/'receipts'/f'{w}_{v}.json.gz'
                log=folder/'logs'/f'{w}_{v}.log'
                log.parent.mkdir(parents=True,exist_ok=True)
                handle=log.open('w')
                p=subprocess.Popen([sys.executable,'-u',str(root/'worker.py'),'--world',str(w),
                     '--version',v,'--cycle',str(cycle),'--output',str(output)],stdout=handle,stderr=subprocess.STDOUT,cwd=root)
                running[p.pid]=(p,handle,w,v,output)
            for pid,(p,handle,w,v,output) in list(running.items()):
                if p.poll() is None:
                    continue
                handle.close()
                if p.returncode:
                    raise RuntimeError(f'worker {w}/{v} exited {p.returncode}; see its job log')
                certificate=read(output.with_name(output.name.replace('.json.gz','.audit.json')))
                assert certificate['accepted']
                completed.append(dict(world=w,version=v,records=certificate['records']))
                del running[pid]
                write(root/'cycles'/f'cycle{cycle}'/'COMMIT_INDEX.json',completed)
            status(cycle,'RUNNING',committed_jobs=len(completed),committed_records=sum(j['records'] for j in completed),
                   active=[dict(pid=pid,world=x[2],version=x[3]) for pid,x in running.items()],remaining=len(jobs),elapsed=elapsed)
            if jobs and not running and elapsed>=28800:
                status(cycle,'PAUSED_SESSION_LIMIT',committed_jobs=len(completed),remaining=len(jobs))
                return
            time.sleep(2)
        provenance.verify(cycle)
        result=summarize(cycle)
        status(cycle,'CYCLE_COMPLETE_REVIEW_REQUIRED' if cycle==2 else 'THREE_CYCLES_COMPLETED_REVIEW_REQUIRED',
               committed_jobs=result['jobs'],committed_records=result['records'])
    except BaseException as e:
        for p,_,_,_,_ in running.values():
            if p.poll() is None:
                p.terminate()
        write(root/'results/FAILURE.json',dict(utc=now(),cycle=cycle,error=repr(e),committed=completed))
        status(cycle,'FAILED',error=repr(e),committed_jobs=len(completed),committed_records=sum(j['records'] for j in completed))
        raise
    finally:
        if awake.poll() is None:
            awake.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int,choices=(2,3),required=True)
    main(p.parse_args().cycle)
