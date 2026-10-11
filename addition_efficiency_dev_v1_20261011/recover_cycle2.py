"""Authorized technical audit repair; validates saved job, never replays it."""
import json,time,shutil
from runner import ROOT,SOURCES,source_lock,execute,save
from audit import audit
from ops import read_receipt,atomic_json,validate_cycle_sources
from technical import local_suite,tamper_suite
from reduce import metrics
from cycles import operational_suite
folder=ROOT/'cycles/cycle2'
old=json.loads((folder/'SOURCE_LOCK.json').read_text());now=source_lock()
changed=[name for name in SOURCES if old['files'][name]!=now['files'][name]]
assert set(changed)=={'audit.py','technical.py'},changed
assert old['parent']==now['parent']
archive=folder/'source_after_audit_repair';archive.mkdir()
for name in SOURCES:shutil.copyfile(ROOT/name,archive/name)
atomic_json(now,folder/'AUDITOR_REPAIR_SOURCE_LOCK.json')
atomic_json(local_suite(),folder/'LOCAL_TESTS_AFTER_REPAIR.json')
name='WORLD_834004_STABLE_D40.json.gz';raw=folder/'receipts/provisional'/name;final=folder/'receipts'/name
assert not final.exists()
r=read_receipt(raw);assert r['source']==old
start=time.process_time();r['independent_audit']=audit(r,expected_source=old);r['audit_cpu_seconds']=time.process_time()-start
r['audit_repair']={'saved_trajectory_replayed':False,'original_source':old,'validator_source':now,'reason':'Independent accumulation can differ by an ulp at a half-integer. Check actual-value ordinal rendering and actual argmax, while independently checking numeric value/coordinates.'}
save(r,final)
for arm in ('U042-WG1','U021-SF1'):
    path=folder/'receipts'/f'WORLD_834004_{arm}_D40.json.gz'
    assert not path.exists() and not path.with_name(path.name+'.attempt.json').exists()
    print('TECH previously unexecuted',arm,flush=True);execute(834004,arm,40,path)
paths=sorted((folder/'receipts').glob('*.json.gz'));assert len(paths)==6
receipts=[read_receipt(p) for p in paths]
tamper={}
for arm in ('STABLE','U042-WG1','U021-SF1'):
    q=next(r for r in receipts if r['arm']==arm)
    tamper[arm]=tamper_suite(q,expected_source=q['source'])
atomic_json(tamper,folder/'TAMPER_TESTS.json')
atomic_json(operational_suite(folder/'operations'),folder/'OPERATIONS_TESTS.json')
validate_cycle_sources(folder)
summary={'verdict':'TECHNICAL_CYCLE_PASS_AFTER_AUDITOR_REPAIR','cycle':2,'full_jobs':6,'technical_histories':18,'training_bytes':sum(r['training_bytes'] for r in receipts),'training_records':sum(r['training_records'] for r in receipts),'independent_rows':sum(r['independent_audit']['independent_rows'] for r in receipts),'receipt_bytes':sum(p.stat().st_size for p in paths),'metrics':[metrics(r) for r in receipts],'science_worlds':0,'DEV_worlds':0,'confirmation_worlds':0,'parameters_changed':False,'trajectory_replays':0,'source_epochs':[old,now],'boundary_differences_in_repaired_job':receipts[[p.name for p in paths].index(name)]['independent_audit']['numerical_boundary_argmax_differences']}
atomic_json(summary,folder/'SUMMARY.json')
failure=json.loads((folder/'FAILURE.json').read_text());failure.update(resolved=True,resolution='Audit-only repair; preserved provisional job independently verified without reexecution')
atomic_json(failure,folder/'RESOLVED_AUDIT_FAILURE.json');(folder/'FAILURE.json').unlink()
atomic_json({'stage':'COMPLETE','verdict':summary['verdict'],'committed_jobs':6,'technical_histories':18,'science_worlds':0},folder/'STATUS.json')
print(json.dumps({k:v for k,v in summary.items() if k not in ('metrics','source_epochs')}),flush=True)
