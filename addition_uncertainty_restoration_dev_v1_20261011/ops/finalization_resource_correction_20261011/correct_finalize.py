"""Operations-only correction; consume accepted aggregates, never replay training."""
import copy
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import config
from runner import atomic_json

HERE=Path(__file__).resolve().parent

def resource_check(q,free_bytes):
    if q['technical_jobs']!=35 or q['science_jobs']!=0 or q['DEV_worlds']!=0 or q['CONF_worlds']!=0:
        raise ValueError('only completed technical qualification')
    if set(q['resources'])!=set(config.RESTORE_ARMS):raise ValueError('full resource roster')
    if any(not all(k in r for k in ('production_CPU_vs_SUPPORT','max_paired_payload_vs_SUPPORT','peak_RSS_bytes')) for r in q['resources'].values()):
        raise ValueError('complete measured resource charges')
    costs=all(r['production_CPU_vs_SUPPORT']<=1.5 and r['max_paired_payload_vs_SUPPORT']<=2
        and r['peak_RSS_bytes']<=512*1024**2 for r in q['resources'].values())
    space=q['artifact_bytes_conservative']+1024**3<=free_bytes
    return bool(costs and space and q['DEV_hours_conservative']<=8 and
        q['per_full_job_seconds_conservative']+300<=8*3600)

def main():
    original=HERE/'original'
    original.mkdir(exist_ok=False)
    names=('QUALIFICATION.json','SOURCE_LOCK_FINAL.json','FINAL_AUDIT.json','STATUS.json',
           'THREE_CYCLE_REPORT.md','PACKAGE.json','RESULT_BUNDLE.zip','finish.log')
    for name in names:shutil.copy2(ROOT/name,original/name)
    q=json.loads((original/'QUALIFICATION.json').read_text())
    source=config.lock()
    if source!=q['sources'] or source!=json.loads((ROOT/'cycles/cycle3/SOURCE_LOCK.json').read_text())['files']:
        raise AssertionError('qualified scientific/technical sources changed')
    audit=json.loads((original/'FINAL_AUDIT.json').read_text())
    if audit['verdict']!='PASS_THREE_TECHNICAL_CYCLES' or audit['accepted_certificates']!=35:
        raise AssertionError('accepted certificate count')
    # Synthetic correction fixtures: no core invocation or receipt transition audit.
    assert resource_check(q,10*1024**3)
    assert not resource_check(q,2*1024**3)
    bad=copy.deepcopy(q);bad['resources']['SUPPORT-DIR']['production_CPU_vs_SUPPORT']=2.
    assert not resource_check(bad,10*1024**3)
    bad=copy.deepcopy(q);bad['resources']['SUPPORT-DIR']['max_paired_payload_vs_SUPPORT']=3.
    assert not resource_check(bad,10*1024**3)
    bad=copy.deepcopy(q);bad['science_jobs']=1
    try:resource_check(bad,10*1024**3)
    except ValueError:pass
    else:raise AssertionError('science aggregate accepted by technical-only correction')
    free=shutil.disk_usage(ROOT).free
    if not resource_check(q,free):raise AssertionError('actual design resources not qualified')
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record={'type':'OPERATIONS_ONLY_FINALIZATION_CORRECTION','reason':'unlisted inherited 2 GiB archive gate',
       'old_status':q['status'],'measured_free_bytes':free,'estimated_artifact_bytes':q['artifact_bytes_conservative'],
       'reserve_bytes':1024**3,'source_sha256':digest,'science_sources_changed':False,
       'new_training_records':0,'new_trajectories':0,'synthetic_gate_checks':5,
       'CPU_payload_RSS_time_thresholds_changed':False,'scientific_samples_and_gates_changed':False}
    atomic_json(record,HERE/'OPERATIONS_AUDIT.json')
    status='TECHNICAL_QUALIFIED_SCIENCE_NOT_AUTHORIZED'
    q.update(status=status,resource_pass=True,archive_space_check={'estimated_bytes':q['artifact_bytes_conservative'],
       'free_bytes':free,'reserve_bytes':1024**3,'passed':True},operations_amendment=record)
    audit.update(resource_status=status,operations_amendment=record)
    atomic_json(q,ROOT/'QUALIFICATION.json');atomic_json(audit,ROOT/'FINAL_AUDIT.json')
    atomic_json({'stage':status,'technical_jobs':35,'science_jobs':0},ROOT/'STATUS.json')
    atomic_json({'files':source,'operations_files':{str(Path(__file__).relative_to(ROOT)):digest},
                'science_authorized':False},ROOT/'SOURCE_LOCK_FINAL.json')
    report=(original/'THREE_CYCLE_REPORT.md').read_text().replace('RESOURCE_QUALIFICATION_HOLD',status)
    report+='\n## 收尾空间检查修订\n\n收尾器意外继承了旧实验2 GiB归档限额，新设计没有此门槛。保留原资格记录及包在ops/finalization_resource_correction_20261011/original。改为检查实际可用磁盘覆盖保守归档预估并预留1 GiB；CPU、载荷、RSS、时间、样本及科学门不变。修订只读取已接受汇总，没有重新执行或审核任何训练轨迹。原科学和资格源仍与第三轮快照完全一致，独立收尾修订代码另行记入源锁。\n'
    (ROOT/'THREE_CYCLE_REPORT.md').write_text(report)
    (HERE/'AMENDMENT.md').write_text('# Finalization resource correction\n\nThe inherited 2 GiB artifact cap was not in the new design contract. All scientific parameters and measured core-resource gates remain unchanged. Use actual free disk capacity with 1 GiB reserve for the conservative artifact estimate. Preserve original HOLD records and archive. No records or trajectories rerun; five synthetic operations checks cover capacity and gate failures. Frozen cycle-3 sources are unchanged; this finalizer is source-recorded separately.\n')
    archive=ROOT/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p in (archive,ROOT/'PACKAGE.json'):continue
            # Preserve historical package on disk without recursively nesting it.
            if p==original/'RESULT_BUNDLE.zip':continue
            z.write(p,p.relative_to(ROOT))
        for p in sorted(config.DESIGN.glob('*')):
            if p.is_file():z.write(p,'design/'+p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise AssertionError('corrected archive integrity')
    atomic_json({'archive':archive.name,'bytes':archive.stat().st_size,
       'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'science_jobs':0,
       'operations_amendment':'ops/finalization_resource_correction_20261011/OPERATIONS_AUDIT.json'},ROOT/'PACKAGE.json')
    print(json.dumps({'status':status,'technical_jobs':35,'new_trajectories':0,
       'archive_bytes':archive.stat().st_size,'sources_unchanged':True}),flush=True)

if __name__=='__main__':main()
