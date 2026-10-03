"""Authorized concurrency transition; drain whole jobs, never replay science."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parent
ROOT=OPS.parent.parent
sys.path.insert(0,str(ROOT))
import bootstrap
from v2_integrity import verify
from compact_storage import atomic_json


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    amendment=json.loads((OPS/'AMENDMENT.json').read_text())
    source=verify()
    assert source==amendment['identity']
    qualification=json.loads((OPS/'QUALIFICATION.json').read_text())
    assert qualification['verdict']=='PASS' and qualification['workers']==8
    old=amendment['original_supervisor_pid']
    folder=ROOT/'results'
    transition=OPS/'transition'
    transition.mkdir(exist_ok=True)
    caff=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:
        began=time.monotonic()
        while True:
            assert not (folder/'FAILURE.json').exists(), 'A real failure appeared before transition'
            current=subprocess.run(['ps','-axo','pid=,ppid=,stat=,command='],capture_output=True,text=True,check=True)
            children=[]
            old_state=None
            for line in current.stdout.splitlines():
                parts=line.strip().split(None,3)
                if len(parts)!=4: continue
                pid,ppid,state,command=parts
                if int(pid)==old: old_state=state
                if int(ppid)==old and str(ROOT/'worker.py') in command and not state.startswith('Z'):
                    children.append(int(pid))
            assert old_state and old_state.startswith('T'), 'Old dispatch must remain paused while workers drain'
            if not children: break
            if time.monotonic()-began>900: raise TimeoutError('Drain timeout; do not interrupt learner trajectories')
            print(json.dumps({'draining_worker_pids':children}),flush=True)
            time.sleep(10)
        cps=list((folder/'science/checkpoints').glob('*.npz'))
        assert not cps, 'Incomplete trajectory found; transition must not replay it'
        for row in amendment['original_status_snapshot']['active_jobs']:
            assert (folder/f"science/receipts/{row['world']}_{row['arm']}_{row['assay']}.json.gz").exists(), 'An active job exited without its receipt'
        receipts={p.name:sha(p) for p in (folder/'science/receipts').glob('*.json.gz')}
        assert len(receipts)>=amendment['original_status_snapshot']['completed_world_arm_assay_jobs']
        for name in ('STATUS.json','LAUNCH.json'):
            (transition/f'original_{name}').write_bytes((folder/name).read_bytes())
        atomic_json(transition/'RECEIPTS_BEFORE.json',receipts)
        os.kill(old,signal.SIGTERM)
        os.kill(old,signal.SIGCONT)
        for _ in range(60):
            state=subprocess.run(['ps','-p',str(old),'-o','stat='],capture_output=True,text=True)
            if state.returncode or not state.stdout.strip() or state.stdout.strip().startswith('Z'): break
            time.sleep(1)
        else: raise RuntimeError('Old supervisor did not stop cleanly')
        failure=json.loads((folder/'FAILURE.json').read_text())
        assert 'supervisor interrupted by signal 15' in failure['error'], 'Unexpected stop reason; no restart permitted'
        assert all(sha(folder/'science/receipts'/name)==value for name,value in receipts.items())
        (folder/'FAILURE.json').replace(transition/'AUTHORIZED_SUPERVISOR_STOP.json')
        assert verify()==source
        record={'authorized':True,'reason':'User requested eight workers','old_pid':old,'committed_jobs_retained':len(receipts),'interrupted_learner_jobs':0,'partial_checkpoints':0,'completed_receipts_unchanged':True,'science_source_identity_unchanged':True,'science_repeated':False}
        atomic_json(transition/'TRANSITION.json',record)
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        logpath=folder/'supervisor8.log'
        with logpath.open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(OPS/'supervisor8.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=environment)
        launch={'pid':proc.pid,'workers':8,'python':sys.executable,'identity':source,'detached_session':True,'authorized_science_after_three_cycles':True,'operational_qualification':str(OPS/'QUALIFICATION.json'),'supervisor':str(OPS/'supervisor8.py'),'log':str(logpath),'committed_jobs_preserved_at_switch':len(receipts),'interrupted_learner_jobs':0}
        atomic_json(folder/'LAUNCH.json',launch)
        for _ in range(60):
            if proc.poll() is not None: raise RuntimeError(f'Eight-worker supervisor exited {proc.returncode}; inspect {logpath}')
            status=json.loads((folder/'STATUS.json').read_text())
            if status.get('pid')==proc.pid and status['state']=='RUNNING' and status['workers']==8 and len(status.get('active_jobs',[]))==8 and status.get('caffeinate_attached'):
                assert all(sha(folder/'science/receipts'/name)==value for name,value in receipts.items())
                record.update(new_pid=proc.pid,active_worker_count=8,caffeinate_pid=status['caffeinate_pid'])
                atomic_json(transition/'TRANSITION.json',record)
                print(json.dumps({'transition':'COMPLETE',**record}),flush=True)
                return
            time.sleep(1)
        raise RuntimeError('Eight-worker dispatch did not become healthy within 60 seconds')
    finally:
        caff.terminate()
        caff.wait()


if __name__=='__main__': main()
