"""One explicitly authorized detached launch, never a restart or publisher."""
import ast
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
import campaign_common as c


def main():
    if sys.executable!="/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/bin/python":
        raise PermissionError("Wrong pinned interpreter")
    if (c.RESULTS/"LAUNCH.json").exists() or (c.OPS/"DETACHED_PROCESS.json").exists():
        raise PermissionError("Already launched; never automatically restart")
    qualification=c.read(HERE/"CAMPAIGN_QUALIFICATION.json")
    if not qualification["accepted"] or not qualification.get("detached_supervisor_logic_qualified"):
        raise PermissionError("Operations qualification missing")
    if qualification["source_identity"]!=c.identity():raise PermissionError("Qualified operative sources changed")
    for path in HERE.glob("*.py"):ast.parse(path.read_text())
    archive=c.OPS/"prelaunch_technical";archive.mkdir(parents=True,exist_ok=False)
    for path in (HERE/"SOURCE_LOCK.json",HERE/"QUALIFICATION.json",HERE/"REPORT.md"):
        shutil.copy2(path,archive/path.name)
    shutil.copytree(c.RESULTS,archive/"results")
    authorization=dict(schema="NATIVE_BIND_RUN_AUTHORIZATION_V1",human_authorized=True,
        experiment="NATIVE-BIND-V1",stages=["DEV","CONFIRM"],sessions=["DEV","CONFIRM_A"],
        resume_authorized=False,user_request="Okay let us run this in the background.",utc=c.now(),
        unchanged_scientific_kernel=True,conditional_confirmation=True,
        no_automatic_new_designs=True,github_upload_authorized=False,
        confirm_B_requires_explicit_continuation=True)
    c.write(c.OPS/"AUTHORIZATION.json",authorization)
    old=c.read(HERE/"SOURCE_LOCK.json");operative=old["operative_sources"].copy()
    for path in list(HERE.glob("*.py"))+list(c.OPS.glob("*.py"))+[HERE/"CAMPAIGN_QUALIFICATION.json",HERE/"RESOURCE_ADMISSION.json",c.OPS/"AUTHORIZATION.json"]:
        operative[str(path.relative_to(c.ROOT))]=c.sha(path)
    old.update(operative_sources=dict(sorted(operative.items())),utc=c.now(),science_authorized=True,
               campaign_operations_qualified=True,previous_technical_lock=str(archive/"SOURCE_LOCK.json"),
               kernel_identity_unchanged=qualification["kernel_identity"],qualification=str(HERE/"CAMPAIGN_QUALIFICATION.json"))
    c.write(HERE/"SOURCE_LOCK.json",old);c.verify_lock()
    c.write(c.RESULTS/"SUMMARY.json",dict(status="IN_PROGRESS",science=True,committed_jobs=0,committed_records=0))
    c.write(c.RESULTS/"STATUS.json",dict(status="STARTING_DETACHED",stage="DEV",updated_utc=c.now(),
        committed_full_jobs=0,committed_histories=0,committed_records=0,active_jobs=[]))
    # The old FINAL_AUDIT remains the archived technical result, not a science verdict.
    for name in ("FINAL_AUDIT.json","PACKAGE.json","BUNDLE_INVENTORY.json","RESULT_BUNDLE.zip"):
        path=c.RESULTS/name
        if path.exists():path.unlink()
    log=(c.OPS/"supervisor.log").open("w")
    process=subprocess.Popen([sys.executable,"-u",str(HERE/"campaign_supervisor.py"),
              "--stage","DEV","--authorization",str(c.OPS/"AUTHORIZATION.json")],
             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=HERE,start_new_session=True)
    log.close()
    c.write(c.OPS/"LAUNCHER_RETURN.json",dict(supervisor_pid=process.pid,detached=True,start_new_session=True,
        stdin="DEVNULL",utc=c.now(),kernel_unchanged=True))
    # Monitor has independent lifetime and no writing or control endpoint.
    for port in range(8887,8907):
        probe=socket.socket()
        try:probe.bind(("127.0.0.1",port))
        except OSError:probe.close();continue
        probe.close();break
    else:raise RuntimeError("No loopback monitor port available")
    handle=(c.OPS/"monitor.log").open("w")
    monitor=subprocess.Popen([sys.executable,"-u",str(HERE/"campaign_monitor.py"),"--port",str(port)],
             stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,cwd=HERE,start_new_session=True)
    handle.close();url=f"http://127.0.0.1:{port}/"
    for _ in range(30):
        if monitor.poll() is not None:raise RuntimeError("Read-only monitor exited; see monitor.log")
        try:
            with urllib.request.urlopen(url+"api/status",timeout=1) as response:
                assert response.status==200
            break
        except OSError:time.sleep(.2)
    else:raise RuntimeError("Read-only monitor not responding")
    c.write(c.OPS/"MONITOR.json",dict(url=url,pid=monitor.pid,read_only=True,utc=c.now()))
    for _ in range(30):
        if process.poll() is not None:raise RuntimeError("Supervisor exited during startup; inspect status and log")
        if (c.RESULTS/"LAUNCH.json").exists():break
        time.sleep(.2)
    c.write(c.OPS/"BACKGROUND_VERIFICATION.json",dict(supervisor_pid=process.pid,
        monitor_pid=monitor.pid,detached_session=os.getsid(process.pid)==process.pid,
        supervisor_running=process.poll() is None,monitor_http_ok=True,science_restarted=False,utc=c.now()))
    print(dict(supervisor_pid=process.pid,monitor_pid=monitor.pid,url=url,source_lock=c.sha(HERE/"SOURCE_LOCK.json")),flush=True)


if __name__=="__main__":main()
