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


def run_stage(stage,authorization,session_start,resume_session=False):
    c.verify_lock()
    selected=None if stage=="DEV" else c.read(c.RESULTS/"DEV/SELECTION_LOCK.json")["selected"]
    manifest=c.manifest(stage,selected)
    path=c.RESULTS/stage/"MANIFEST.json"
    if path.exists() and c.read(path)!=manifest:raise PermissionError("Existing manifest differs")
    if not path.exists():c.write(path,manifest)
    certified={(j["world"],j["arm"]) for j in c.committed() if j["stage"]==stage}
    pending=[j for j in manifest["jobs"] if (j["world"],j["arm"]) not in certified]
    running={};completed=manifest["full_jobs"]-len(pending)
    estimate=c.read(c.HERE/"RESOURCE_ADMISSION.json")["projections"][stage]["projected_wall_hours_conservative"]*3600
    worst_job=estimate/max(1,(manifest["full_jobs"]+WORKERS-1)//WORKERS)
    stop_requested=False
    try:
        while pending or running:
            elapsed=time.monotonic()-session_start
            if elapsed>=9.75*3600 and not stop_requested:
                stop_requested=True;request_boundary_pause(running)
            if elapsed>=10*3600:raise TimeoutError("Ten-hour hard session boundary reached")
            while pending and len(running)<WORKERS and elapsed<8*3600 and elapsed+worst_job<9.75*3600 and not stop_requested:
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


def analysis(stage):
    c.status("ANALYZING",stage=stage,active_jobs=[])
    log=c.OPS/(stage.lower()+"_analysis.log")
    with log.open("w") as handle:
        process=subprocess.Popen([sys.executable,"-u",str(c.HERE/"campaign_stage.py"),"--stage",stage],
            stdout=handle,stderr=subprocess.STDOUT,cwd=c.HERE)
        while process.poll() is None:
            c.write(c.RESULTS/"ANALYSIS_HEARTBEAT.json",dict(stage=stage,pid=process.pid,
                heartbeat_epoch=time.time(),supervisor_pid=os.getpid()))
            time.sleep(5)
        if process.returncode:raise RuntimeError("Analysis/packaging failed; see "+str(log))


def main(stage,authorization,resume_session=False):
    auth=c.read(authorization)
    if stage not in auth.get("sessions",[]):raise PermissionError("Session not human-authorized")
    if resume_session and not auth.get("resume_authorized",False):raise PermissionError("Resume not authorized")
    c.verify_lock()
    if (c.RESULTS/"FAILURE.json").exists():raise PermissionError("Failure present; no automatic restart")
    if (c.RESULTS/"SESSION_RESULT.json").exists() and not resume_session:
        raise PermissionError("Paused session requires explicit resume")
    session_start=time.monotonic();awake=subprocess.Popen(["/usr/bin/caffeinate","-i","-w",str(os.getpid())])
    launch=dict(schema="NATIVE_BIND_DETACHED_LAUNCH_V1",utc=c.now(),supervisor_pid=os.getpid(),
                workers=WORKERS,caffeinate_pid=awake.pid,python=sys.executable,stage=stage,
                source_lock_sha256=c.sha(c.HERE/"SOURCE_LOCK.json"),authorization=str(authorization),
                supervisor_log=str(c.OPS/"supervisor.log"),session_admission_hours=8,session_hard_hours=10,
                maximum_full_jobs=88,maximum_records=311296,conditional_confirmation=True)
    c.write(c.RESULTS/"LAUNCH.json",launch)
    c.write(c.OPS/"DETACHED_PROCESS.json",launch)
    try:
        if not run_stage(stage,authorization,session_start,resume_session):return
        if stage=="DEV":
            analysis("DEV")
            selection=c.read(c.RESULTS/"DEV/SELECTION_LOCK.json")
            if selection["selected"] is None:return
            stage="CONFIRM_A"
            projection=c.read(c.HERE/"RESOURCE_ADMISSION.json")["projections"][stage]["projected_wall_hours_conservative"]*3600
            if time.monotonic()-session_start+projection>8*3600:
                c.write(c.RESULTS/"SESSION_RESULT.json",dict(status="PAUSED_SESSION_LIMIT",next_stage=stage,
                         reason="Next predeclared stage does not fit conservative remaining session budget",utc=c.now()))
                c.status("PAUSED_SESSION_LIMIT",stage="DEV_COMPLETE",next_stage=stage,active_jobs=[])
                return
            if not run_stage(stage,authorization,session_start):return
        if stage=="CONFIRM_A":
            c.write(c.RESULTS/"SESSION_RESULT.json",dict(status="PAUSED_SESSION_LIMIT",next_stage="CONFIRM_B",
                     reason="Predeclared separate second confirmation session; explicit continuation required",utc=c.now()))
            c.status("PAUSED_SESSION_LIMIT",stage="CONFIRM_A_COMPLETE",next_stage="CONFIRM_B",active_jobs=[])
            return
        if stage=="CONFIRM_B":analysis("CONFIRM")
    except BaseException as exc:
        import traceback
        index=c.update_index()
        c.write(c.RESULTS/"FAILURE.json",dict(error=repr(exc),traceback=traceback.format_exc(),utc=c.now(),
                committed_full_jobs=index["full_jobs"],committed_records=index["records"],automatic_retry=False))
        c.status("FAILED",error=repr(exc),active_jobs=[])
        raise
    finally:
        if awake.poll() is None:awake.terminate()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("DEV","CONFIRM_A","CONFIRM_B"),default="DEV")
    p.add_argument("--authorization",required=True);p.add_argument("--resume-session",action="store_true")
    args=p.parse_args();main(args.stage,args.authorization,args.resume_session)
