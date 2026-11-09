"""Single guarded producer: three technical cycles, then nine locked lives."""
import os
import subprocess
import sys
import time
import resource
from datetime import datetime, timezone
from pathlib import Path
from e16wr2_common import ROOT,OUT,read,save,sha,verify_lock,require_machine_free,validate_receipt

def main():
    gate=require_machine_free();lock=verify_lock()
    guard=OUT/'e16wr2_campaign.guard'
    guard.mkdir()  # Atomic ownership; stale owners must be investigated, never silently removed.
    save(guard/'owner.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),lock_sha256=sha(ROOT/'E16WR2_LOCK.json')))
    (OUT/'logs').mkdir(exist_ok=True)
    jobs=[dict(j,status='queued') for j in lock['jobs']]
    if any((OUT/f'E16WR2_JOB_{j["key"]}.json').exists() for j in jobs):raise RuntimeError('Prior JOB exists; inspect, do not duplicate')
    running={};t0=time.monotonic();self_cpu0=time.process_time();failed=False
    child0=resource.getrusage(resource.RUSAGE_CHILDREN)
    def state(status):
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        save(ROOT/'E16WR2_STATUS.json',dict(status=status,controller_pid=os.getpid(),initial_gate=gate,
             updated_at=datetime.now(timezone.utc).isoformat(),jobs=jobs,completed=sum(j['status']=='complete' for j in jobs),
             planned_subprocesses=len(jobs),scientific_lives=9,full_default_profiles=0,wall_seconds=time.monotonic()-t0,
             source_lock_sha256=sha(ROOT/'E16WR2_LOCK.json'),
             reaped_child_cpu_seconds=child.ru_utime+child.ru_stime-child0.ru_utime-child0.ru_stime,
             controller_cpu_seconds=time.process_time()-self_cpu0))
    # The technical job is separately charged and is not a scientific life.
    env=dict(os.environ,EXP45_HIDDEN='512',E8_WD='.25',E8_REPLAY_SAMPLING='uniform',E8_REPLAY_TARGETS='self',E8_SCHEDULE='blocked')
    technical_log=OUT/'logs/e16wr2_checks.log'
    state('QUALIFYING')
    start=time.monotonic();tech0=resource.getrusage(resource.RUSAGE_CHILDREN)
    with technical_log.open('xb') as f:
        p=subprocess.Popen([sys.executable,'e16wr2_checks.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        code=p.wait()
    tech1=resource.getrusage(resource.RUSAGE_CHILDREN)
    save(ROOT/'E16WR2_TECHNICAL_JOB.json',dict(pid=p.pid,returncode=code,wall_seconds=time.monotonic()-start,
         process_tree_cpu_seconds=tech1.ru_utime+tech1.ru_stime-tech0.ru_utime-tech0.ru_stime,log=str(technical_log.relative_to(ROOT))))
    if code or read(ROOT/'E16WR2_CHECKS.json')['status']!='PASS':
        state('TECHNICAL_FAILURE');return
    while True:
        for key,(p,handle,start) in list(running.items()):
            rc=p.poll()
            if rc is None:continue
            handle.close();j=next(x for x in jobs if x['key']==key)
            j.update(returncode=rc,wall_seconds=time.monotonic()-start)
            if rc==0:
                try:
                    validate_receipt(j);j.update(status='complete',receipt_sha256=sha(OUT/j['receipt']))
                except Exception as ex:j.update(status='technical_failure',validation_error=repr(ex));failed=True
            else:j.update(status='technical_failure');failed=True
            save(OUT/f'E16WR2_JOB_{key}.json',j);del running[key]
        if not failed:
            for j in jobs:
                if len(running)>=lock['max_workers']:break
                if j['status']!='queued':continue
                require_machine_free();verify_lock()
                handle=(OUT/'logs'/f'e16wr2_{j["key"]}.log').open('xb')
                env=dict(os.environ,**j['env'])
                start=time.monotonic()
                p=subprocess.Popen([sys.executable,'e16wr2_worker.py',j['key']],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                j.update(status='running',pid=p.pid,started_at=datetime.now(timezone.utc).isoformat(),command=[sys.executable,'e16wr2_worker.py',j['key']])
                running[j['key']]=(p,handle,start);save(OUT/f'E16WR2_JOB_{j["key"]}.json',j)
        state('TECHNICAL_FAILURE_WAITING_FOR_ACTIVE_WORKERS' if failed and running else 'TECHNICAL_FAILURE' if failed else 'RUNNING')
        if not running and (failed or all(j['status']=='complete' for j in jobs)):break
        time.sleep(10)
    if not failed: state('RUNS_COMPLETE_VERIFICATION_PENDING')

if __name__=='__main__':main()
