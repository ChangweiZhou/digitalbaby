"""Fixed one-attempt developmental dispatcher, two workers, no formal admission."""
import os
import sys
import time
import signal
import subprocess
import traceback
import resource
from pathlib import Path
import bootstrap
import config
from artifacts import write,read,now
from provenance import verify


def committed():
    root=bootstrap.HERE/'results'
    a=sorted((root/'track_a').glob('*.audit.json'))
    b=sorted((root/'track_b/receipts').glob('*.audit.json'))
    certs=[read(p) for p in a+b]
    if not all(x['accepted'] for x in certs):raise AssertionError('unaccepted committed audit')
    return dict(native_stages=len(a),addition_jobs=len(b),
                addition_correlated_histories=3*len(b),native_histories=2*len(a),
                native_record_applications=sum(x['record_applications'] for x in certs[:len(a)]),
                addition_record_applications=sum(x['record_applications'] for x in certs[len(a):]))


def finalize():
    import analysis
    summary=analysis.report();root=bootstrap.HERE;results=root/'results'
    certs=sorted((results/'track_a').glob('*.audit.json'))+sorted((results/'track_b/receipts').glob('*.audit.json'))
    assert len(list((results/'track_b/receipts').glob('*.json.gz')))==4
    assert all(read(p)['accepted'] for p in certs)
    assert len(summary['track_b_profiles'])==4
    verify()
    write(results/'FINAL_AUDIT.json',dict(utc=now(),accepted=True,science=False,**committed(),
        certificates=[str(p.relative_to(root)) for p in certs],
        technical_qualification=read(root/'tests/QUALIFICATION.json')['accepted'],
        frozen_sources_unchanged=True,language_task=False,automatic_adoption=False,
        scope='Finite native E1 qualification plus two-world descriptive addition acquisition/HELD/retention curves; no population confirmation.'))
    import zipfile
    members=[root/'README.md',root/'SPEC.json',root/'MODEL_MANIFEST.json',root/'SOURCE_LOCK.json',root/'REPORT.md']
    members+=list(root.glob('*.py'))+list(root.glob('OUTPUT_CODEBOOK.json'))
    members+=list((root/'tests').glob('*.py'))+list((root/'tests').glob('*.json'))+list((root/'tests').glob('*.log'))
    members+=list((root/'ops').glob('AUTHORIZATION.json'))
    members+=[p for p in results.rglob('*') if p.is_file() and p.suffix in ('.json','.gz') and p.name not in ('PACKAGE.json','STATUS.json') and 'active' not in p.parts and p.name!='ACTIVE.json']
    output=results/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(members)):z.write(p,str(p.relative_to(root)))
    from artifacts import digest
    write(results/'PACKAGE.json',dict(utc=now(),complete=True,path=str(output),sha256=digest(output),bytes=output.stat().st_size,
         source_assets='Unchanged retained parents referenced by SOURCE_LOCK.json; not redundantly copied.',science=False))


def main():
    root=bootstrap.HERE;results=root/'results';ops=root/'ops';started=time.monotonic()
    if (results/'LAUNCH.json').exists():raise FileExistsError('one attempt already launched; no automatic resume')
    verify();qualification=read(root/'tests/QUALIFICATION.json')
    if not qualification['accepted']:raise AssertionError('technical qualification required')
    queue=[('A',['--track','A'])]+[(f'{w}_{m}',['--track','B','--world',str(w),'--model',m]) for w in config.TRACK_B_WORLDS for m in config.TRACK_B_MODELS]
    initial=committed()
    if initial['native_stages'] or initial['addition_jobs']:raise AssertionError('preexisting accepted development data')
    launch=dict(utc=now(),pid=os.getpid(),workers=2,science=False,formal_campaign=False,
        fixed_jobs=[name for name,_ in queue],executable=sys.executable,source_lock=str(root/'SOURCE_LOCK.json'),
        admission_seconds=config.SESSION_ADMISSION_SECONDS,hard_seconds=config.SESSION_HARD_SECONDS)
    write(results/'LAUNCH.json',launch)
    caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    launch['caffeinate_pid']=caffeinate.pid;write(results/'LAUNCH.json',launch)
    active={};completed=[];halt=None
    def stop(signum,frame):raise InterruptedError('dispatcher interrupted by signal '+str(signum))
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        while queue or active:
            elapsed=time.monotonic()-started
            if elapsed>=config.SESSION_HARD_SECONDS:raise TimeoutError('hard 10-hour boundary; partial artifacts retained, no retry')
            if elapsed>=config.SESSION_ADMISSION_SECONDS:halt='PAUSED_SESSION_LIMIT'
            while queue and len(active)<2 and not halt:
                name,args=queue.pop(0)
                log=(ops/'logs'/f'{name}.log');log.parent.mkdir(exist_ok=True)
                stream=log.open('x')
                child=subprocess.Popen([sys.executable,str(root/'worker.py'),*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
                active[name]=(child,stream,time.monotonic())
            for name,(child,stream,began) in list(active.items()):
                code=child.poll()
                if code is None:continue
                stream.close();del active[name]
                if code:raise RuntimeError(f'{name} exited {code}; see ops/logs/{name}.log and track FAILURE.json')
                completed.append(dict(job=name,elapsed=time.monotonic()-began,exit_code=code))
            status=dict(utc=now(),state=halt or 'RUNNING_DEVELOPMENT',science=False,
                active=[dict(job=name,pid=x[0].pid) for name,x in active.items()],queued=len(queue),completed_process_jobs=completed,
                elapsed_seconds=time.monotonic()-started,**committed())
            write(results/'STATUS.json',status)
            if halt and not active:
                write(results/'SESSION_RESULT.json',status);return
            time.sleep(1)
        write(results/'STATUS.json',dict(utc=now(),state='ANALYZING',science=False,**committed()))
        finalize()
        costs=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(results/'STATUS.json',dict(utc=now(),state='COMPLETE',science=False,accepted=True,
            completed_process_jobs=completed,**committed(),elapsed_seconds=time.monotonic()-started,
            completed_children_user_cpu=costs.ru_utime,completed_children_system_cpu=costs.ru_stime,
            package_complete=True,automatic_adoption=False,published=False))
    except BaseException as exc:
        for child,stream,_ in active.values():
            if child.poll() is None:child.terminate()
        for child,stream,_ in active.values():
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            stream.close()
        failure=dict(utc=now(),state='FAILED',error=repr(exc),traceback=traceback.format_exc(),science=False,**committed(),automatic_retry=False)
        write(results/'FAILURE.json',failure);write(results/'STATUS.json',failure)
        raise
    finally:
        if caffeinate.poll() is None:caffeinate.terminate();caffeinate.wait()

if __name__=='__main__':main()
