"""Exercise real dispatcher with synthetic child-process handles, no learner."""
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
import campaign_common as c
import campaign_supervisor as dispatcher

PARENT=Path(__file__).resolve().parent/"dispatcher_qualification"
PARENT.mkdir(exist_ok=True)
attempt=1
while (PARENT/f"attempt{attempt}").exists():attempt+=1
BASE=PARENT/f"attempt{attempt}"
BASE.mkdir()
ORIGINAL_RESULTS=c.RESULTS;ORIGINAL_OPS=c.OPS
real_time_sleep=time.sleep


class Child:
    next_pid=900000;active=0;peak=0;calls=0;fail=False
    def __init__(self,command,**kwargs):
        type(self).next_pid+=1;self.pid=type(self).next_pid
        type(self).calls+=1;type(self).active+=1;type(self).peak=max(type(self).peak,type(self).active)
        self.polls=0;self.returncode=None;self.stopped=False
        self.stage=command[command.index("--stage")+1];self.arm=command[command.index("--arm")+1]
        self.world=int(command[command.index("--world")+1]);self.folder=Path(command[command.index("--out")+1])
        self.is_failure=type(self).fail and type(self).calls==1
    def poll(self):
        if self.returncode is not None:return self.returncode
        self.polls+=1
        if self.polls<3 and not self.stopped:return None
        type(self).active-=1
        if self.is_failure:
            c.write(self.folder/"FAILURE.json",dict(error="SYNTHETIC_WORKER_FAILURE"));self.returncode=1
        elif self.stopped:
            c.write(self.folder/"SESSION_RESULT.json",dict(status="PAUSED_SESSION_LIMIT",synthetic=True));self.returncode=0
        else:
            c.write(self.folder/"RECEIPT.json",dict(synthetic_fixture_only=True,actual_native_records=0))
            c.write(self.folder/"AUDIT.json",dict(accepted=True,world=self.world,arm=self.arm,records=2048,
                    receipt_sha256=c.sha(self.folder/"RECEIPT.json"),synthetic_fixture_only=True))
            self.returncode=0
        return self.returncode
    def send_signal(self,sig):self.stopped=True
    def terminate(self):self.stopped=True;self.polls=3;self.poll()


def setup(name):
    c.RESULTS=BASE/name;c.OPS=c.RESULTS/"ops";c.OPS.mkdir(parents=True,exist_ok=True)
    Child.active=0;Child.peak=0;Child.calls=0;Child.fail=False


def main():
    dispatcher.subprocess=SimpleNamespace(Popen=Child,DEVNULL=dispatcher.subprocess.DEVNULL,STDOUT=dispatcher.subprocess.STDOUT)
    dispatcher.time.sleep=lambda seconds:None
    checks={}
    setup("healthy")
    assert dispatcher.run_stage("DEV",Path("SYNTHETIC_ONLY"),time.monotonic())
    assert c.update_index()["full_jobs"]==24 and Child.calls==24 and Child.peak==4
    calls=Child.calls
    assert dispatcher.run_stage("DEV",Path("SYNTHETIC_ONLY"),time.monotonic()) and Child.calls==calls
    checks["four_worker_dispatch_and_no_replay"]=True
    setup("paused")
    assert not dispatcher.run_stage("DEV",Path("SYNTHETIC_ONLY"),time.monotonic()-8*3600-1)
    assert Child.calls==0 and c.read(c.RESULTS/"SESSION_RESULT.json")["status"]=="PAUSED_SESSION_LIMIT"
    checks["eight_hour_whole_job_admission_stop"]=True
    setup("failure");Child.fail=True
    try:dispatcher.run_stage("DEV",Path("SYNTHETIC_ONLY"),time.monotonic())
    except RuntimeError as exc:assert "SYNTHETIC_WORKER_FAILURE" in str(exc)
    else:raise AssertionError("Failure was not propagated")
    assert Child.calls==4 and Child.active==0
    checks["failure_stops_new_admission_and_preserves_other_windows"]=True
    setup("hard_boundary")
    try:dispatcher.run_stage("DEV",Path("SYNTHETIC_ONLY"),time.monotonic()-36001)
    except TimeoutError:pass
    else:raise AssertionError("Ten-hour boundary not enforced")
    assert Child.calls==0
    checks["ten_hour_hard_session_boundary"]=True
    setup("confirmation_lock")
    c.write(c.RESULTS/"DEV/SELECTION_LOCK.json",dict(selected="BIND",confirmation_CPU_budget=64))
    c.write(c.RESULTS/"DEV/CPU_BUDGET_LOCK.json",dict(C_dev=32,C_star=64))
    c.write(c.RESULTS/"DEV/CONFIRM_ADMISSION_LOCK.json",dict(selected="BIND",
        selection_sha256=c.sha(c.RESULTS/"DEV/SELECTION_LOCK.json"),
        CPU_budget_sha256=c.sha(c.RESULTS/"DEV/CPU_BUDGET_LOCK.json")))
    assert c.manifest("CONFIRM_A","BIND")["full_jobs"]==32
    c.write(c.RESULTS/"DEV/SELECTION_LOCK.json",dict(selected="BIND-FB",confirmation_CPU_budget=64))
    try:c.manifest("CONFIRM_A","BIND")
    except PermissionError:pass
    else:raise AssertionError("Modified confirmation selection was accepted")
    checks["confirmation_selection_and_budget_supplement_lock"]=True
    c.RESULTS=ORIGINAL_RESULTS;c.OPS=ORIGINAL_OPS;dispatcher.time.sleep=real_time_sleep
    path=HERE/"CAMPAIGN_QUALIFICATION.json";q=c.read(path)
    q["dispatcher_checks"]=checks;q["detached_supervisor_logic_qualified"]=True
    q["additional_native_training_records"]=0;q["science_jobs"]=0
    c.write(path,q)
    print(json.dumps(dict(accepted=True,dispatcher_checks=checks,actual_native_records=0)))


if __name__=="__main__":main()
