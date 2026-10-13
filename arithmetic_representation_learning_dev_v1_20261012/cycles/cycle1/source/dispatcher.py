"""Detached-ready dispatcher, no implicit authorization/retry/resume/upload."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from auditor import audit
from core import CONFIGS
from operations import atomic_json, manifest, may_admit, require_authorization, committed_index
from runner import run_job,source_lock


def worker(job,directory,technical,auth):
    directory=Path(directory)
    hb=directory/'active'/f"{job['job_id']}.json"
    path=directory/'receipts'/f"{job['job_id']}.json.gz"
    r=run_job(job['world'],job['config'],path,technical=technical,
              heartbeat_path=hb,authorization=auth)
    certificate=json.loads((directory/'certificates'/f"{job['job_id']}.json").read_text())
    hb.unlink(missing_ok=True)
    return certificate


def dispatch(directory,workers,authorization=None,technical_jobs=None,resume=False,
             stop_after_completed=None,estimated_job_seconds=300.):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    technical=technical_jobs is not None
    if not technical:
        require_authorization(authorization)
        if stop_after_completed is not None:raise ValueError('synthetic stop forbidden in science')
        qualification=json.loads((Path(__file__).parent/'QUALIFICATION.json').read_text())
        if not qualification.get('ready_for_authorized_science') or qualification.get('source_lock')!=source_lock():
            raise ValueError('three-cycle qualification incomplete')
    jobs=technical_jobs if technical else manifest()
    frozen={'technical':technical,'jobs':jobs,'source_lock':source_lock()}
    frozen_path=directory/'MANIFEST.json'
    if frozen_path.exists():
        if not resume:raise FileExistsError('explicit resume required')
        if json.loads(frozen_path.read_text())!=frozen:raise ValueError('resume manifest/source mismatch')
        if (directory/'FAILURE.json').exists():raise ValueError('failure cannot automatically restart')
        status=json.loads((directory/'STATUS.json').read_text())
        if status['status']!='PAUSED_SESSION_LIMIT':raise ValueError('only safely paused sessions resume')
    else:
        if resume:raise ValueError('nothing to resume')
        atomic_json(frozen_path,frozen)
    expected={(j['world'],j['config']) for j in jobs}
    index=committed_index(directory/'receipts',expected,technical=technical,validate=True)
    done={(r['world'],r['config']) for r in index['rows']}
    queue=[j for j in jobs if (j['world'],j['config']) not in done]
    for name in ('active','receipts','certificates'):(directory/name).mkdir(exist_ok=True)
    atomic_json(directory/'LAUNCH.json',{'pid':os.getpid(),'workers':workers,'epoch':time.time(),
        'technical':technical,'source_lock':source_lock(),'resume':resume,'committed_before':index['committed_jobs'], 'estimated_job_seconds':estimated_job_seconds})
    started=time.monotonic();active={};submitted=0;completed=0;caff=None
    if sys.platform=='darwin':
        caff=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while queue or active:
                elapsed=time.monotonic()-started
                allowed=may_admit(elapsed,estimated_job_seconds)
                if stop_after_completed is not None and completed>=stop_after_completed:
                    allowed=False
                while queue and len(active)<workers and allowed:
                    job=queue.pop(0)
                    future=pool.submit(worker,job,str(directory),technical,authorization)
                    active[future]=job;submitted+=1
                if not active:break
                atomic_json(directory/'STATUS.json',{'status':'RUNNING' if allowed else 'DRAINING',
                    'epoch':time.time(),'technical':technical,'active_jobs':len(active), 'supervisor_pid':os.getpid(),
                    'committed_jobs':len(done),'committed_histories':3*len(done),
                    'committed_records':1536*len(done),'remaining_jobs':len(queue)})
                ready,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                if time.monotonic()-started>10*3600:
                    # Never let Executor.__exit__ wait forever for a hung job.
                    # This is a failed run, not a pause/resume path. Partial files
                    # remain for diagnosis; none are certified or counted.
                    for process in list(pool._processes.values()):
                        process.terminate()
                    for process in list(pool._processes.values()):
                        process.join(timeout=2)
                        if process.is_alive():process.kill()
                    raise TimeoutError('hard session boundary reached; preserve incomplete work, no retry')
                for future in ready:
                    job=active.pop(future)
                    try:
                        certificate=future.result()
                    except BaseException:
                        for process in list(pool._processes.values()):
                            process.terminate()
                        raise
                    done.add((job['world'],job['config']));completed+=1
                if ready:
                    index=committed_index(directory/'receipts',expected,technical=technical)
                    atomic_json(directory/'COMMIT_INDEX.json',index)
        status='PAUSED_SESSION_LIMIT' if queue else 'SCIENCE_EXECUTION_DONE'
        if technical and not queue:status='TECHNICAL_EXECUTION_DONE'
        index=committed_index(directory/'receipts',expected,technical=technical)
        result={'status':status,'epoch':time.time(),'technical':technical,**index,
                'remaining_jobs':len(queue),'active_jobs':0,'wall_seconds':time.monotonic()-started}
        atomic_json(directory/'STATUS.json',result)
        atomic_json(directory/'SESSION_RESULT.json',result)
        if not technical and not queue:
            from finish import finish
            finish(directory/'receipts',directory)
            result['status']='COMPLETED_ACCEPTED_AND_PACKAGED'
            atomic_json(directory/'STATUS.json',result)
        return result
    except BaseException as exc:
        atomic_json(directory/'FAILURE.json',{'error':repr(exc),'epoch':time.time(),
            'committed_jobs':len(done),'technical':technical,'restart_authorized':False})
        raise
    finally:
        if caff is not None:caff.terminate();caff.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--authorization');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--technical-world',type=int);parser.add_argument('--technical-stop',type=int)
    parser.add_argument('--estimated-job-seconds',type=float,default=300.)
    args=parser.parse_args()
    jobs=None if args.technical_world is None else [{'world':args.technical_world,'config':c,
        'job_id':f'{args.technical_world}_{c}'} for c in CONFIGS]
    result=dispatch(args.out,args.workers,args.authorization,jobs,args.resume,args.technical_stop,args.estimated_job_seconds)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
