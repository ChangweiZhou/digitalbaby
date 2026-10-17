"""Explicitly authorized continuation of unstarted jobs; resource waits stay alive."""
import inspect,json,os,subprocess,sys,time,traceback,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;I=HERE.parents[1];B=I/'ops/dev_b_20261017'
sys.path.insert(0,str(B))
import campaign_common as c
import campaign_supervisor as original
sys.path.insert(0,str(HERE))
from memory_policy import Policy

def stage_function():
    source=inspect.getsource(original.run_stage)
    source=source.replace('memory=memory_admission(running)','memory=memory_admission(running,job_arm=pending[0]["arm"])')
    old='''                    if not running:
                        c.write(c.RESULTS/'SESSION_RESULT.json',dict(status='PAUSED_RESOURCE_ADMISSION',memory=memory,next_stage=stage))
                        c.status('PAUSED_RESOURCE_ADMISSION',stage=stage,pending_jobs=len(pending),active_jobs=[],memory=memory,next_stage=stage)
                        return False
                    break'''
    new='''                    waiting_memory=memory
                    break'''
    assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace('elapsed=time.monotonic()-session_start','elapsed=time.monotonic()-session_start\n            waiting_memory=None')
    source=source.replace('c.status("RUNNING",stage=stage,stage_committed_jobs=completed,','c.status("WAITING_RESOURCE" if waiting_memory and not running else "RUNNING",stage=stage,resource_wait=waiting_memory if not running else None,next_resource_check_epoch=time.time()+15 if waiting_memory and not running else None,stage_committed_jobs=completed,')
    source=source.replace('status="RUNNING" if pending or running else "COMPLETED"','status=("WAITING_RESOURCE" if waiting_memory and not running else "RUNNING") if pending or running else "COMPLETED"')
    source=source.replace('if pending or running:time.sleep(5)','if pending or running:time.sleep(15 if not running else 5)')
    namespace=dict(original.__dict__);namespace['memory_admission']=Policy().admission
    exec(compile(source,str(HERE/'ADAPTED_STAGE.py.txt'),'exec'),namespace)
    return namespace['run_stage'],source

def main():
    c.verify_lock();lock=c.read(HERE/'SOURCE_LOCK.json')
    for name,digest in lock['files'].items():
        if c.sha(HERE/name)!=digest:raise PermissionError('Continuation lock mismatch:'+name)
    assert c.read(HERE/'QUALIFICATION.json')['passed']
    assert not (HERE/'LAUNCH.json').exists(),'Never automatically relaunch controller'
    state=c.read(c.RESULTS/'STATUS.json');assert state['status']=='PAUSED_RESOURCE_ADMISSION' and not state['active_jobs']
    assert not (c.RESULTS/'FAILURE.json').exists(),'A failed scientific run cannot restart'
    manifest=c.manifest('DEV_B')
    certified={(j['world'],j['arm']) for j in c.committed() if j['stage']=='DEV_B'}
    pending=[j for j in manifest['jobs'] if (j['world'],j['arm']) not in certified]
    for job in pending:
        assert not (c.job_folder('DEV_B',job)/'EXECUTION_LEDGER.json').exists(),'Unfinished trajectory must not be replayed'
    assert len(pending)==19
    # Conservative original session boundary: includes prior sleep/pause wall time, never grants a fresh10h.
    birth=datetime.datetime.fromisoformat(c.read(B/'LAUNCH.json')['utc']).timestamp()
    elapsed=max(0,time.time()-birth);start=time.monotonic()-elapsed
    assert elapsed<8*3600,'Original admission window exhausted'
    c.write(HERE/'PRE_RESUME_STATE.json',dict(previous_status=state,previous_session_result=c.read(c.RESULTS/'SESSION_RESULT.json'),
        pending_jobs=pending,counts=c.update_index(),original_session_wall_seconds_carried=elapsed))
    coffee=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    launch=dict(schema='NATIVE_BIND_RESOURCE_WAIT_CONTINUATION',stage='DEV_B',supervisor_pid=os.getpid(),session_id=os.getsid(0),caffeinate_pid=coffee.pid,
        workers=4,numerical_threads=1,utc=c.now(),log=str(HERE/'supervisor.log'),worker_source=str(B/'campaign_worker.py'),
        source_lock=str(HERE/'SOURCE_LOCK.json'),frozen_scientific_source_lock=str(B/'SOURCE_LOCK.json'),original_elapsed_seconds_carried=elapsed,
        remaining_unstarted_jobs=19,no_trajectory_replayed=True,transient_resource_wait_automatically_rechecked=True)
    c.write(HERE/'LAUNCH.json',launch);c.write(c.RESULTS/'LAUNCH.json',launch)
    # Archive the historical safe-pause marker so it cannot be mistaken for a current pause.
    (c.RESULTS/'SESSION_RESULT.json').rename(HERE/'HISTORICAL_RESOURCE_PAUSE.json')
    try:
        fn,source=stage_function();(HERE/'ADAPTED_STAGE.py.txt').write_text(source)
        if fn('DEV_B',B/'AUTHORIZATION.json',start):
            c.status('ANALYZING',stage='DEV_B',active_jobs=[])
            with (HERE/'analysis.log').open('x') as out:
                proc=subprocess.Popen([sys.executable,'-u',str(B/'campaign_finish.py')],cwd=I,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    c.write(c.RESULTS/'ANALYSIS_HEARTBEAT.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),heartbeat_epoch=time.time()));time.sleep(5)
                if proc.returncode:raise RuntimeError('Frozen DEV analysis failed;see '+str(HERE/'analysis.log'))
    except BaseException as exc:
        c.write(c.RESULTS/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=c.now(),**c.update_index()))
        c.status('FAILED',stage='DEV_B',error=repr(exc),active_jobs=[]);raise
    finally:
        if coffee.poll() is None:coffee.terminate()
if __name__=='__main__':main()
