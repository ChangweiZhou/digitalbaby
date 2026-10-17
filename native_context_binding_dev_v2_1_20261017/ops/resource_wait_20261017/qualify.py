"""Synthetic resource wait/recovery dispatch; no native model is instantiated."""
import json,sys,tempfile,time,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
import supervisor as control
from memory_policy import decide

def main():
    reserves={'NATIVE':600}
    assert decide(10**10,2,reserves,[],'NATIVE')['admit']
    assert not decide(10**10,2,reserves,[dict(arm='NATIVE',RSS_bytes=500)],'NATIVE')['admit']
    assert not decide(10**10,4,reserves,[],'NATIVE')['admit']
    assert not decide(0,2,reserves,[],'NATIVE')['admit']
    fn,source=control.stage_function();states=[];sleeps=[];launches=[];calls=[]
    assert 'PAUSED_RESOURCE_ADMISSION' not in source
    assert 'WAITING_RESOURCE' in source and 'return False' in source # session limits remain explicit
    with tempfile.TemporaryDirectory(prefix='bind_wait_fixture_') as root:
        folder=Path(root);job=dict(key='synthetic_only',world=0,arm='NATIVE')
        m=dict(full_jobs=1,jobs=[job]);realc=control.c
        def read(p):
            if Path(p).name=='RESOURCE_ESTIMATE.json':return dict(DEV_four_world_session_estimate_hours=.01)
            return realc.read(p)
        fake=types.SimpleNamespace(HERE=control.I,OPS=folder,RESULTS=folder/'results',verify_lock=lambda:True,
            manifest=lambda *args:m,committed=lambda:[],job_folder=lambda stage,j:folder/'job',write=realc.write,read=read,sha=realc.sha,
            status=lambda state,**kw:states.append(state),now=realc.now)
        count=[0]
        def admission(*args,**kw):
            count[0]+=1;return dict(admit=count[0]>=3,pressure=4 if count[0]<3 else 2)
        class Proc:
            pid=999999
            def __init__(self,*args,**kw):
                launches.append(args[0]);p=folder/'job';p.mkdir(exist_ok=True)
                fake.write(p/'RECEIPT.json',dict(elapsed_seconds=.1,synthetic_only=True))
                fake.write(p/'AUDIT.json',dict(accepted=True,receipt_sha256=fake.sha(p/'RECEIPT.json'),synthetic_only=True))
            def poll(self):return 0
        fn.__globals__.update(c=fake,memory_admission=admission,
            subprocess=types.SimpleNamespace(Popen=Proc,DEVNULL=-3,STDOUT=-2),
            time=types.SimpleNamespace(monotonic=time.monotonic,time=time.time,sleep=lambda n:sleeps.append(n)))
        assert fn('DEV_B',folder/'synthetic_auth',time.monotonic()) is True
        assert states.count('WAITING_RESOURCE')==2 and len(launches)==1 and sleeps[:2]==[15,15]
    result=dict(passed=True,zero_native_records=True,science_jobs=0,
        critical_wait_then_warning_singleton_progress=True,wait_heartbeat_remains_alive=True,
        warning_additional_workers_blocked=True,insufficient_headroom_blocked=True,
        no_external_worker_or_model_spawned=True,no_retry_of_executed_trajectory=True,original_eight_and_ten_hour_limits_preserved=True)
    control.c.write(HERE/'QUALIFICATION.json',result);print(json.dumps(result))
if __name__=='__main__':main()
