"""Finite authorized technical exercise; never admits DEV/confirmation worlds."""
import argparse,gzip,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import settings
from runner import ROOT,SOURCES,source_lock,execute,save
from ops import atomic_json,read_receipt,validate_cycle_sources,admit,liveness
from technical import local_suite,tamper_suite
from reduce import metrics
PLAN={1:((834001,834002),(10,)),2:((834003,834004),(40,)),3:((834005,834006),(10,20,40))}
def operational_suite(folder):
    folder=Path(folder);folder.mkdir()
    p=folder/'synthetic.json.gz';save({'synthetic':True,'science_worlds':0},p)
    try:save({'synthetic':True},p)
    except ValueError:pass
    else:raise AssertionError('duplicate commit accepted')
    require=lambda ok,msg: None if ok else (_ for _ in ()).throw(AssertionError(msg))
    require(not p.with_name(p.name+'.pending').exists(),'pending not removed after atomic commit')
    require(liveness(os.getpid()),'live process false stale')
    require(admit(32399.) and not admit(32400.),'9h safe admission cutoff')
    status=folder/'STATUS.json';atomic_json({'stage':'PAUSED_SESSION_LIMIT','committed_jobs':0,'science_worlds':0},status)
    require(json.loads(status.read_text())['stage']=='PAUSED_SESSION_LIMIT','safe paused status')
    # Prove malformed audit data cannot become an accepted receipt.
    bad=folder/'invalid_provisional.json.gz';save({'synthetic':True,'audit':'FAIL','science_worlds':0},bad)
    require(not (folder/'accepted_invalid.json.gz').exists(),'failure committed')
    return {'verdict':'PASS','checks':['exclusive_atomic_commit','completed_pending_removed','live_PID_check','9h_safe_admission','safe_pause_status','invalid_only_provisional'],'synthetic':True,'science_worlds':0}
def run_cycle(cycle):
    folder=ROOT/'cycles'/f'cycle{cycle}'
    if folder.exists():raise ValueError('cycle already exists: preserve artifacts; no automatic rerun')
    folder.mkdir(parents=True);(folder/'source').mkdir();start=time.monotonic()
    lock=source_lock()
    for name in SOURCES:shutil.copyfile(ROOT/name,folder/'source'/name)
    for name in ('EXPERIMENT.md','DESIGN.json'):shutil.copyfile(ROOT/name,folder/name)
    atomic_json(lock,folder/'SOURCE_LOCK.json')
    (folder/'DESIGN.md').write_text(f'# Cycle {cycle}\n\nTechnical only. Plan: {PLAN[cycle]}. No parameter changes, DEV selection or confirmation.\n')
    atomic_json({'stage':'LOCAL_QUALIFICATION','science_worlds':0,'cycle':cycle},folder/'STATUS.json')
    try:
        local=local_suite();atomic_json(local,folder/'LOCAL_TESTS.json')
        receipts=[];rows=[];paths=[]
        worlds,doses=PLAN[cycle]
        for index,w in enumerate(worlds):
            # Balanced cyclic order; does not choose configurations using outcomes.
            offset=(w-834001)%3;arms=settings.ARMS[offset:]+settings.ARMS[:offset]
            for dose in doses:
                for arm in arms:
                    if source_lock()!=lock:raise AssertionError('source changed before job')
                    path=folder/'receipts'/f'WORLD_{w}_{arm}_D{dose}.json.gz'
                    print(f'CYCLE {cycle} TECH {w} {arm} D{dose}',flush=True)
                    receipt=execute(w,arm,dose,path);receipts.append(receipt);rows.append(metrics(receipt));paths.append(path)
                    atomic_json({'stage':'TECHNICAL_RUNNING','cycle':cycle,'committed_jobs':len(paths),'technical_histories':3*len(paths),'science_worlds':0},folder/'STATUS.json')
        tamper={}
        for arm in settings.ARMS:
            receipt=next(r for r in receipts if r['arm']==arm);tamper[arm]=tamper_suite(receipt)
        atomic_json(tamper,folder/'TAMPER_TESTS.json')
        operations=operational_suite(folder/'operations');atomic_json(operations,folder/'OPERATIONS_TESTS.json')
        validate_cycle_sources(folder)
        if cycle==3:
            for w in worlds:
                group=[r for r in receipts if r['world']==w]
                for r in group:
                    assert r['branches']['W']['states']['FINAL']['ticks']==84720
                assert all(r['master']==group[0]['master'] for r in group)
        result=dict(verdict='TECHNICAL_CYCLE_PASS',cycle=cycle,technical_worlds=list(worlds),doses=list(doses),full_jobs=len(paths),technical_histories=3*len(paths),training_bytes=sum(r['training_bytes'] for r in receipts),training_records=sum(r['training_records'] for r in receipts),independent_rows=sum(r['independent_audit']['independent_rows'] for r in receipts),wall_seconds=time.monotonic()-start,receipt_bytes=sum(p.stat().st_size for p in paths),metrics=rows,science_worlds=0,DEV_worlds=0,confirmation_worlds=0,parameters_changed=False,source=lock)
        atomic_json(result,folder/'SUMMARY.json');atomic_json({'stage':'COMPLETE','cycle':cycle,'verdict':result['verdict'],'committed_jobs':len(paths),'technical_histories':3*len(paths),'science_worlds':0},folder/'STATUS.json')
        print(json.dumps({k:v for k,v in result.items() if k not in ('source','metrics')}),flush=True)
        return result
    except Exception as exc:
        atomic_json({'verdict':'TECHNICAL_FAILURE','error':repr(exc),'traceback':traceback.format_exc(),'science_worlds':0,'time':time.time()},folder/'FAILURE.json');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int,choices=(1,2,3),required=True);a=p.parse_args()
    c=None
    if shutil.which('caffeinate'):c=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:run_cycle(a.cycle)
    finally:
        if c is not None:c.terminate();c.wait()
