# Operational dispatch adapter; frozen scientific files remain unchanged.
from pathlib import Path
import sys as _sys, json as _json, hashlib as _hashlib
_OPS=Path(__file__).resolve().parent
_ROOT=_OPS.parent.parent
_sys.path.insert(0,str(_ROOT))
EXECUTION_AMENDMENT=_json.loads((_OPS/'AMENDMENT.json').read_text())
def _check_execution_qualification():
    q=_json.loads((_OPS/'QUALIFICATION.json').read_text())
    lock=_json.loads((_OPS/'SOURCE_LOCK.json').read_text())
    assert q['verdict']=='PASS' and q['workers']==8
    assert q['identity']==EXECUTION_AMENDMENT['identity']
    assert _hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==lock['supervisor8_sha256']==q['supervisor8_sha256']
    assert '--prepare' not in _sys.argv, 'Qualification must not be rerun by the execution adapter'
if __name__=='__main__':
    _check_execution_qualification()
"""Detached qualification/science supervision; no result-driven reruns."""
import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
import bootstrap
from v2_core import ARMS
from v2_fixture import SCIENCE_WORLDS, ASSAYS
from v2_integrity import identity, verify
from v2_audit import read, audit_receipt
from compact_storage import atomic_json

ROOT=bootstrap.ROOT
BRANCH='codex/persistent-core-v2-20261003'


def now(): return datetime.now(timezone.utc).isoformat()


def publish(message):
    repo=ROOT.parent
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
    if branch!=BRANCH: raise ValueError('publication branch changed')
    subprocess.run(['git','add','--',ROOT.name],cwd=repo,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
        subprocess.run(['git','commit','-m',message],cwd=repo,check=True)
    subprocess.run(['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=268435456',
                    'push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=repo,check=True,timeout=600)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); args=p.parse_args()
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    import fcntl
    mutex=(results/'supervisor.lock').open('a'); fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'state':'RUNNING','stage':'qualification' if args.prepare else 'execution','pid':os.getpid(),
            'started_utc':now(),'completed_world_arm_assay_jobs':0,'total_world_arm_assay_jobs':768,
            'completed_lives':0,'total_lives':2304,'workers':8,'caffeinate_attached':False,'execution_amendment':'ops/workers8/AMENDMENT.json'}
    caff=None; active={}; completed=[]
    if platform.system()=='Darwin' and shutil.which('caffeinate'):
        caff=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
        status.update(caffeinate_attached=True,caffeinate_pid=caff.pid)
    def stop(sig,frame): raise RuntimeError(f'supervisor interrupted by signal {sig}')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        if args.prepare:
            for cycle in (1,2,3):
                status.update(stage=f'qualification_cycle_{cycle}',heartbeat_utc=now()); atomic_json(results/'STATUS.json',status)
                logpath=results/f'cycle{cycle}.log'
                with logpath.open('w') as f:
                    proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'cycles.py'),'--cycle',str(cycle)],stdout=f,stderr=subprocess.STDOUT)
                    status.update(qualification_pid=proc.pid); atomic_json(results/'STATUS.json',status)
                    while proc.poll() is None:
                        status.update(heartbeat_utc=now()); atomic_json(results/'STATUS.json',status); time.sleep(5)
                    if proc.returncode: raise RuntimeError(f'cycle {cycle} failed; read {logpath}')
            subprocess.run([sys.executable,'-u',str(ROOT/'cycles.py'),'--qualify'],check=True,timeout=60)
        source=verify(); qual=json.loads((results/'QUALIFICATION.json').read_text())
        if qual['verdict']!='PASS' or qual['identity']!=source: raise ValueError('unqualified final source')
        spec=json.loads((ROOT/'SPEC.json').read_text())
        jobs=[(w,a,s) for w in SCIENCE_WORLDS for a in ARMS for s in ASSAYS]
        folder=results/'science'; worker_s=0.
        for job in list(jobs):
            key='_'.join(map(str,job)); receipt=folder/'receipts'/f'{key}.json.gz'
            if receipt.exists():
                d=read(receipt); audit_receipt(d,source); completed.append(job); jobs.remove(job); worker_s+=d['worker_s']
            elif (folder/'checkpoints'/f'{key}.npz').exists(): raise ValueError('partial science requires explicit resume authorization')
        status.update(stage='execution',identity=source,science_started_utc=EXECUTION_AMENDMENT["original_science_started_utc"]); status.pop('qualification_pid',None)
        while jobs or active:
            while jobs and len(active)<8:
                w,a,s=jobs.pop(0); key=f'{w}_{a}_{s}'; logpath=folder/'logs'/f'{key}.workers8.log'; logpath.parent.mkdir(parents=True,exist_ok=True)
                log=logpath.open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--world',str(w),'--arm',a,'--assay',s],
                                      stdout=log,stderr=subprocess.STDOUT)
                active[proc.pid]={'process':proc,'log':log,'job':(w,a,s),'started':time.monotonic(),'logpath':str(logpath)}
            for pid,e in list(active.items()):
                proc=e['process']; code=proc.poll()
                if time.monotonic()-e['started']>spec['max_job_seconds']: raise RuntimeError(f'worker timeout: {e["job"]}')
                if code is None:
                    rss=subprocess.run(['ps','-p',str(pid),'-o','rss='],capture_output=True,text=True)
                    if rss.returncode==0 and rss.stdout.strip().isdigit() and int(rss.stdout.strip())*1024>spec['max_worker_rss_bytes']:
                        raise RuntimeError(f'worker RSS cap: {e["job"]}')
                    continue
                e['log'].close()
                if code: raise RuntimeError(f'worker failed {e["job"]}: exit {code}; {e["logpath"]}')
                w,a,s=e['job']; d=read(folder/'receipts'/f'{w}_{a}_{s}.json.gz'); audit_receipt(d,source)
                worker_s+=d['worker_s']; completed.append(e['job']); del active[pid]
                print(json.dumps({'committed_jobs':len(completed),'world':w,'arm':a,'assay':s}),flush=True)
            used=sum(p.stat().st_size for p in results.rglob('*') if p.is_file())
            charged=worker_s+sum(time.monotonic()-e['started'] for e in active.values())
            if used>spec['disk_cap_bytes'] or charged>spec['worker_budget_seconds']: raise RuntimeError('cumulative resource cap')
            lives=sum(4 if s=='lifetime' else 2 for w,a,s in completed)
            status.update(heartbeat_utc=now(),completed_world_arm_assay_jobs=len(completed),completed_lives=lives,
                          active_jobs=[{'pid':pid,'world':e['job'][0],'arm':e['job'][1],'assay':e['job'][2]} for pid,e in active.items()],
                          committed_worker_s=worker_s,bytes_used=used)
            atomic_json(results/'STATUS.json',status); time.sleep(5)
        status.update(stage='analysis',active_jobs=[]); atomic_json(results/'STATUS.json',status)
        for script in ('v2_analysis.py','final_audit.py'):
            with (results/f'{script}.log').open('w') as f:
                subprocess.run([sys.executable,'-u',str(ROOT/script)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
        summary=json.loads((results/'SUMMARY.json').read_text())
        bundle=results/'RESULT_BUNDLE.zip'; pending=results/'RESULT_BUNDLE.zip.pending'
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(ROOT.rglob('*')):
                if not p.is_file() or any(x in p.parts for x in ('scratch','checkpoints','locks','active','__pycache__')): continue
                if p.name.endswith(('.zip','.pending','.npz','.lock')): continue
                z.write(p,p.relative_to(ROOT))
        os.replace(pending,bundle)
        status.update(state='COMPLETE',stage='accepted',completed_utc=now(),adoption_verdict=summary['verdict'],github_upload='PENDING')
        atomic_json(results/'STATUS.json',status)
        try:
            commit=publish('Complete persistent core V2 factorial with lifecycle and reuse measurements')
            atomic_json(results/'PUBLICATION.json',{'state':'UPLOADED','commit':commit,'branch':BRANCH,'utc':now()})
            status.update(github_upload='UPLOADED',published_commit=commit)
        except Exception:
            status.update(github_upload='FAILED',publication_error=traceback.format_exc())
            atomic_json(results/'PUBLICATION.json',{'state':'FAILED','error':status['publication_error'],'branch':BRANCH})
    except Exception:
        status.update(state='FAILED',error=traceback.format_exc(),stopped_utc=now())
        atomic_json(results/'FAILURE.json',status); print(status['error'],file=sys.stderr,flush=True)
        for e in active.values():
            if e['process'].poll() is None: e['process'].terminate()
        for e in active.values():
            try: e['process'].wait(timeout=30)
            except subprocess.TimeoutExpired: e['process'].kill(); e['process'].wait()
            e['log'].close()
        return 1
    finally:
        if caff is not None: caff.terminate(); caff.wait(); status['caffeinate_released']=True
        atomic_json(results/'STATUS.json',status)
    if status.get('github_upload')=='UPLOADED':
        try: publish('Record final V2 publication and supervisor status')
        except Exception:
            status.update(github_status_upload='FAILED',status_upload_error=traceback.format_exc()); atomic_json(results/'STATUS.json',status)
    return 0


if __name__=='__main__': sys.exit(main())
