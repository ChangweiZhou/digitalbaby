"""Detached four-worker finite campaign. No retries, no automatic pause-resume."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import campaign_common as c

WORKERS=4


def request_boundary_pause(running):
    for process,handle,job in running.values():
        if process.poll() is None:process.send_signal(signal.SIGUSR1)


def join_after_stop(running):
    request_boundary_pause(running)
    until=time.monotonic()+120
    while any(p.poll() is None for p,h,j in running.values()) and time.monotonic()<until:
        time.sleep(1)
    for p,h,j in running.values():
        if p.poll() is None:p.terminate()
        h.close()


def memory_admission(running):
    import re
    raw=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
    page=int(re.search(r'page size of (\d+)',raw).group(1))
    stats={k:int(v) for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',raw,re.M)}
    available=page*sum(stats.get(k,0) for k in ('Pages free','Pages inactive','Pages speculative'))
    per_worker=c.read(c.HERE/'RESOURCE_ESTIMATE.json')['single_process_peak_RSS_bytes']
    reserved_growth=0
    for pid,(process,handle,job) in running.items():
        path=c.job_folder('DEV_A',job)/'ACTIVE.json'
        measured=c.read(path).get('peak_rss_bytes',0) if path.exists() else 0
        reserved_growth+=max(0,per_worker-measured)
    required=per_worker+reserved_growth+512*1024**2
    return dict(utc=c.now(),admit=available>=required,available_reclaimable_proxy_bytes=available,
        required_bytes=required,new_worker_peak_reserve_bytes=per_worker,reserved_growth_bytes=reserved_growth,
        note='free+inactive+speculative proxy, not guaranteed free RAM; max4 workers, admission may use fewer')


def run_stage(stage,authorization,session_start,resume_session=False):
    c.verify_lock()
    selected=None
    manifest=c.manifest(stage,selected)
    path=c.RESULTS/stage/"MANIFEST.json"
    if path.exists() and c.read(path)!=manifest:raise PermissionError("Existing manifest differs")
    if not path.exists():c.write(path,manifest)
    certified={(j["world"],j["arm"]) for j in c.committed() if j["stage"]==stage}
    pending=[j for j in manifest["jobs"] if (j["world"],j["arm"]) not in certified]
    running={};completed=manifest["full_jobs"]-len(pending)
    estimate=c.read(c.HERE/"RESOURCE_ESTIMATE.json")["DEV_four_world_session_estimate_hours"]*3600
    worst_job=estimate/max(1,(manifest["full_jobs"]+WORKERS-1)//WORKERS)
    stop_requested=False
    observed_worst_job=worst_job
    memory_admissions=[]
    try:
        while pending or running:
            elapsed=time.monotonic()-session_start
            worst_job=max(worst_job,observed_worst_job)
            if elapsed>=9.75*3600 and not stop_requested:
                stop_requested=True;request_boundary_pause(running)
            if elapsed>=10*3600:raise TimeoutError("Ten-hour hard session boundary reached")
            while pending and len(running)<WORKERS and elapsed<8*3600 and elapsed+worst_job<9.75*3600 and not stop_requested:
                memory=memory_admission(running)
                if not memory['admit']:
                    if not running:
                        c.write(c.RESULTS/'SESSION_RESULT.json',dict(status='PAUSED_RESOURCE_ADMISSION',memory=memory,next_stage=stage))
                        c.status('PAUSED_RESOURCE_ADMISSION',stage=stage,pending_jobs=len(pending),active_jobs=[],memory=memory,next_stage=stage)
                        return False
                    break
                c.write(c.OPS/'MEMORY_ADMISSION.json',memory)
                job=pending.pop(0);folder=c.job_folder(stage,job);folder.mkdir(parents=True,exist_ok=True)
                existing=(folder/"EXECUTION_LEDGER.json").exists()
                if existing and not resume_session:raise PermissionError("Executed incomplete job needs explicit resume")
                if existing and not (folder/"PAUSE.pt").exists():raise PermissionError("No safe checkpoint; do not replay")
                c.verify_lock()
                handle=(folder/"job.log").open("a" if existing else "w")
                command=[sys.executable,"-u",str(c.HERE/"campaign_worker.py"),"--stage",stage,
                         "--arm",job["arm"],"--world",str(job["world"]),"--out",str(folder),
                         "--authorization",str(authorization)]
                if existing:command.append("--resume")
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,cwd=c.HERE)
                running[process.pid]=(process,handle,job)
                print("START",stage,job["key"],"pid",process.pid,flush=True)
            for pid,(process,handle,job) in list(running.items()):
                rc=process.poll()
                if rc is None:continue
                handle.close();folder=c.job_folder(stage,job)
                if rc!=0:
                    error=c.read(folder/"FAILURE.json") if (folder/"FAILURE.json").exists() else dict(error=f"Worker exit {rc}")
                    raise RuntimeError(f"{stage}/{job['key']}: {error['error']}; see {folder/'job.log'}")
                if (folder/"SESSION_RESULT.json").exists() and not (folder/"AUDIT.json").exists():
                    stop_requested=True
                    pending.insert(0,job)
                else:
                    certificate=c.read(folder/"AUDIT.json")
                    if not certificate["accepted"] or c.sha(folder/"RECEIPT.json")!=certificate["receipt_sha256"]:
                        raise AssertionError("Worker lacks immutable independent audit")
                    completed+=1
                    observed_worst_job=max(observed_worst_job,c.read(folder/"RECEIPT.json")["elapsed_seconds"]*1.5)
                    print("COMMIT",stage,job["key"],completed,"/",manifest["full_jobs"],flush=True)
                del running[pid]
            active=[]
            for pid,(process,handle,job) in running.items():
                folder=c.job_folder(stage,job);entry=dict(pid=pid,job=job["key"],world=job["world"],arm=job["arm"],log=str(folder/"job.log"))
                if (folder/"ACTIVE.json").exists():entry.update(c.read(folder/"ACTIVE.json"))
                active.append(entry)
            if pending and not running and (stop_requested or elapsed>=8*3600 or elapsed+worst_job>=9.75*3600):
                c.write(c.RESULTS/"SESSION_RESULT.json",dict(status="PAUSED_SESSION_LIMIT",stage=stage,
                         pending_jobs=len(pending),safe_window_boundary=True,next_stage=stage,utc=c.now()))
                c.status("PAUSED_SESSION_LIMIT",stage=stage,pending_jobs=len(pending),active_jobs=[],next_stage=stage)
                return False
            c.status("RUNNING",stage=stage,stage_committed_jobs=completed,stage_total_jobs=manifest["full_jobs"],
                     pending_jobs=len(pending),active_jobs=active,session_elapsed_seconds=elapsed,
                     workers=WORKERS,session_admission_hours=8,session_hard_hours=10)
            c.write(c.RESULTS/stage/"STATUS.json",dict(status="RUNNING" if pending or running else "COMPLETED",
                     completed_jobs=completed,total_jobs=manifest["full_jobs"],heartbeat_epoch=time.time()))
            if pending or running:time.sleep(5)
        c.write(c.RESULTS/stage/"STATUS.json",dict(status="COMPLETED",completed_jobs=completed,
                total_jobs=manifest["full_jobs"],heartbeat_epoch=time.time()))
        return True
    except BaseException:
        join_after_stop(running)
        raise


def main():
    import traceback
    auth=c.OPS/'AUTHORIZATION.json';c.verify_lock()
    if not c.read(c.HERE/'CAMPAIGN_QUALIFICATION.json').get('passed'):raise PermissionError('Operations unqualified')
    if (c.RESULTS/'LAUNCH.json').exists():raise PermissionError('Never automatically restart a launched session')
    c.admission('DEV_A','NATIVE',202610177101,auth)
    c.OPS.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    coffee=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    launch=dict(schema='NATIVE_BIND_DETACHED_LAUNCH_V2_1',utc=c.now(),stage='DEV_A',supervisor_pid=os.getpid(),
        session_id=os.getsid(0),workers=4,numerical_threads=1,caffeinate_pid=coffee.pid,
        authorization=str(auth),source_lock=str(c.HERE/'CAMPAIGN_SOURCE_LOCK.json'),
        log=str(c.OPS/'supervisor.log'),next_stage_requires_explicit_authorization='DEV_B')
    c.write(c.RESULTS/'LAUNCH.json',launch);c.write(c.OPS/'DETACHED_PROCESS.json',launch)
    try:
        if run_stage('DEV_A',auth,start):
            from campaign_finish import finish
            finish()
            c.status('DEV_A_COMPLETE_AWAITING_DEV_B_AUTHORIZATION',stage='DEV_A',stage_committed_jobs=36,
                stage_total_jobs=36,pending_jobs=0,active_jobs=[],next_stage='DEV_B',workers=0)
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc(),utc=c.now(),**c.update_index())
        c.write(c.RESULTS/'FAILURE.json',failure);c.status('FAILED',stage='DEV_A',error=repr(exc),active_jobs=[])
        raise
    finally:
        if coffee.poll() is None:coffee.terminate()

if __name__=='__main__':main()
